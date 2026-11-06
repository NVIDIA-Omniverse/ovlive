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

//! The guards of the service's explicit-cursor SCAN-family walks (SDD
//! §22.22, §11.3.2), at three levels.
//!
//! The serving-process guard, of the periodic reference-count reconcile
//! pass's object, carrier and owed walks (`storage::ext_reconcile`): a walk's
//! result is used only when one server process served every step of it, and,
//! for a keyspace walk on a cluster, when each primary's commands went to one
//! named node that serves exactly that primary's slots and the slot map held
//! still from its start to its end.
//!
//! The completeness proofs of the disaster-recovery Jobs' walks (*The DR
//! Jobs' walks*, below): the restore planner's keyspace walks
//! (`storage::dr_job`), the reference-count recount's walks
//! (`storage::ext_reconcile`'s authoritative mode: objects, each object's
//! value indices, carriers, `pending` and `inflight`) and accept-loss's value
//! removal (`dr::job::execute::discard`). A Job uses a walk's result only
//! when the walk is shown complete on the held store by a proof that holds
//! whichever server process answered each step; no Job walk reads a
//! serving-process witness.
//!
//! The coverage guard, of the service's other keyspace walks
//! (`storage::orphan_sweep::scan_keyspace`: the checkpoint tick's pass 1, the
//! external-value backup repair, the orphan sweep's walks and the reconciler's
//! meta walk): every primary of a fresh slot map is walked on a node pinned
//! for it, the pin's ownership is checked where the node answers (and, where
//! it does not, the routing table the pins came from is checked against the
//! map), and a walk that cannot show its coverage says so after its last key
//! (*Coverage walks*, below, including what an unproven pin can still hide).
//!
//! **Why.** A `SCAN`, `ZSCAN`, `HSCAN` or `SSCAN` cursor is continued on
//! whatever process serves the key when the next step is sent. The table a
//! step walks is hashed with a seed that is random per process start (Valkey
//! 9.0.3 `server.c` 7168–7171), so a cursor one process returned and another
//! continues can skip members: after a failover, a restart between two steps,
//! or a slot migration [inferred from source, untested]. SCAN's "every
//! member present throughout is returned" holds only for cursors the same
//! process returned (`valkey-doc` `commands/scan.md`). On a cluster the
//! keyspace `SCAN` cursor also carries the slot index (9.0.3 `kvstore.c`
//! 139–145, 414–456), so a slot that moves between primaries during a walk
//! can skip keys with no process change at all.
//!
//! **The witness** (the serving-process guard). A process is identified by
//! the `run_id` that `INFO server` reports: generated once at process start
//! (9.0.3 `server.c` 2236–2237) and written nowhere else, so a restart —
//! `DEBUG RESTART` included, which keeps the PID — and a promoted replica
//! both report a new one. `master_replid` is no identity (a primary and its
//! replicas share it, and one process can change it), and neither is
//! `process_id`. A run_id is usable only when it is exactly 40 hex
//! characters and not all zeros ([`parse_run_id`]): a constant placeholder
//! would read "unchanged" forever and leave the walk silently unguarded, so
//! anything else is a missing witness ([`WitnessMissing`]). On Azure Managed
//! Redis `INFO server` is answered by a node endpoint that fronts several
//! primary shards, and a primary shard that moved behind an unchanged
//! endpoint and slot map kept every run_id and the map [tested, SDD
//! §11.3.2]: the periodic pass's guard misses such a move, which its two
//! observations a grace apart absorb; the Jobs read no witness.
//!
//! **Keyed walks** ([`KeyedWalk`]: `ZSCAN`, `HSCAN`, `SSCAN` of one key). The
//! walk holds one fred client (one `pool.next()`) for its whole life, so the
//! witness and the steps share that client's connection to the key's node.
//! The witness is `INFO server` routed by the key's slot (on a cluster;
//! unpinned standalone), and it rides in one fred pipeline with a step, so it
//! costs no round trip of its own: the first step is sent as
//! `[INFO server; step; INFO server]` (the walk's start witness, the step and
//! its after-step witness), every later step as `[step; INFO server]`. fred
//! writes a pipeline's commands in order on the one connection, so the
//! after-step witness is answered by the process that answered the step, and
//! the start witness by the process that answered the first step. The cursor
//! pager (`storage::retirement::page_by_cursor`) hands a step's members on
//! only once that step's witness equals the walk's start. Each step is sent
//! with one attempt (`max_attempts: 1`), so fred neither replays it on a
//! reconnected connection nor follows an `-ASK` to a node the slot-routed
//! witness does not read: a step that did not run on the witnessed node
//! fails. `max_redirections: 0` would do nothing (fred restores its default
//! for a zero, `protocol/command.rs` 1882–1888); the attempt budget is the
//! lever. The witness keeps fred's retries, in the pipeline too: when the
//! connection is lost between a step and its witness, fred replays the
//! witness on the reconnected connection, which reads the new process's
//! run_id, and the comparison catches it.
//!
//! **A step's pipeline.** Its replies are read once, each at its command's
//! index (fred's `try_all`, which answers a server error reply for its
//! command alone). A pipeline fred failed as a whole — a lost connection on
//! the step's one attempt, a redirect with no attempt left, a timeout, a
//! routing miss — answers a single error in place of all its replies (a
//! one-element result, fewer than its commands): a failed attempt of the
//! step, nothing of it used. Any other count of replies than one per
//! command fails the walk closed (a `Protocol` step error). Otherwise the
//! replies are judged in command order: a start witness that answered an
//! error ends the walk with a `Read` error, one with no usable run_id with a
//! missing witness; the step's error reply is a failed attempt when
//! transient and ends the walk when not, its after-step witness left unread;
//! the step's reply must be a scan reply; then its after-step witness is
//! judged as the start witness is.
//!
//! **Keyspace walks** ([`KeyspaceWalk`]: `SCAN <cursor> MATCH <pattern> COUNT
//! 100`; a DR Job's walk, [`KeyspaceWalk::start_job`], is the same walk with
//! no witness and `COUNT` [`DR_KEYSPACE_SCAN_COUNT`], its fingerprint the map
//! and the myself lines alone). On a cluster the walk reads a fingerprint at
//! its start and at its end: a fresh `CLUSTER SLOTS` map, each primary's
//! witness, and each
//! primary's `CLUSTER NODES` *myself* line. Each primary's commands go to one
//! named node (`cluster_node`), resolved once at the walk's start from one
//! snapshot of the walk client's own routing table (the node it lists for the
//! first slot of the primary's lowest range, taken whole so fred's TLS name
//! stays on it), one attempt per step, on the walk's one client; the walk
//! visits the primaries one after another until each answers cursor `0`.
//! A slot pin would not do: fred routes a slot through its cached table and,
//! when that table names a node it holds no connection for, sends the
//! command to an arbitrary connection, and a keyless `SCAN`, `INFO` or
//! `CLUSTER NODES` never draws a `MOVED`, so a stale table walks one primary
//! twice and another not at all with every other check passing. A node pin
//! has no fallback. But a node-pinned command that finds no connection makes
//! fred reconnect the whole client, a shared pool member, so before each
//! command or pipeline on a pin (every separate witness read, `CLUSTER
//! NODES`, and each step's pipeline, the end fingerprint's and a retry's
//! re-read included) the walk checks that the client's current routing
//! table still lists the pinned node as a primary, and fails at once,
//! sending nothing, when it does not: the walk's connection or routing view
//! of that node changed (a `Step` error naming the node). One check covers a
//! step's whole pipeline, its witness included: fred's router writes the
//! pipeline's commands in one pass against one routing table, and a step
//! whose own write fails already makes fred defer a whole-client reconnect,
//! so a landed step's witness has no check of its own. Each step's witness
//! rides in its pipeline (`[SCAN; INFO server]`), on the step's pin.
//!
//! The walk errs ([`ServingChange`]) when the two maps differ; when a
//! primary's myself line shows an open slot (`[slot->-…]` migrating or
//! `[slot-<-…]` importing; Valkey prints them on the myself line only, 9.0.3
//! `cluster_legacy.c` 6650–6667) at either end; when the pinned node's own
//! myself line does not list exactly the slots the walk's map gives that
//! primary, at either end, or the table names no node for it
//! ([`ServingChange::NotOwner`]: the client's view was out of date, so before
//! failing the walk re-syncs the client's routing table, bounded by its
//! command timeout, and records how that went); or when a step's witness
//! differs from its primary's start witness. A `CLUSTER NODES` reply with no
//! myself line, a myself line short of its 8 fixed fields, or a slot field
//! outside the `N` / `A-B` grammar ([`parse_myself_line`]), is a missing
//! witness, like an unusable run_id.
//! Standalone, the fingerprint is the witness alone.
//!
//! **A failed step (R1).** One-attempt steps must not make a walk less
//! tolerant of a transient blip than fred's replay made it. So when a step's
//! attempt fails with a transient error ([`step_error_is_transient`]) — its
//! pipeline failed as a whole, or its step answered the error — and the walk
//! has its start witness, the witness is read again, alone, on the same
//! client and route (never the witness that rode in the failed pipeline):
//! unchanged, the step is sent again at the same cursor with its after-step
//! witness (a read step re-sent on the same process is safe), at most
//! [`WALK_STEP_ATTEMPTS`] attempts in all, with a full-jitter backoff
//! between them; changed or missing, the walk ends there. A keyed walk's
//! first pipeline that failed as a whole read no start witness to compare
//! with, so the next attempt resends that whole pipeline, its start witness
//! included: nothing of it was used, and cursor `0` starts a walk on any
//! process. When its start witness answered and only the step failed, the
//! walk has its start and the re-read applies. A step that fails with a
//! deterministic error (a `WRONGTYPE`, a reply that is not a scan reply)
//! would answer the same again, so it ends the walk on its first attempt, a
//! `Step` error, with no re-read. An `INFO server` re-read that fails while
//! a step is being retried spends an attempt and proves nothing either way;
//! the witness that rides with the step that does land still decides. A node
//! pin the client's routing table no longer lists as a primary is refused
//! before anything is sent, wherever it is checked — before a step's
//! pipeline, before any separate witness read (a keyspace walk's start and
//! end fingerprints, a retry's re-read) and before `CLUSTER NODES` — on
//! every attempt, the last one included; a refusal is not a failed witness
//! read, never spends an attempt, and ends the walk at once. So a step error
//! that is not transient, a witness in a step's pipeline that failed or
//! reads no usable run_id, or, during a retry, a changed or missing witness,
//! a refused node pin, or an exhausted retry ends the walk. A keyed walk's
//! caller passes its attempt bound with each step ([`KeyedWalk::step`]); a
//! caller that already retried its steps itself would pass one, so the
//! bounds do not multiply.
//!
//! **The DR Jobs' walks.** A Job walks a held store: hold entry aborts every
//! writer, the quiet period bounds a late one, and after it only the Job
//! writes; none of the walked key families carries a TTL. On a quiet store a
//! sorted set's rank order (score, then member bytes) and every key's
//! cardinality are properties of the data, the same on any process that
//! holds it, so each walk family is shown complete by a proof that does not
//! ask which process answered:
//!
//! - **Sorted sets by rank pages** ([`RankWalk`]). The recount's pager
//!   ([`page_by_rank_counted`]) reads ascending `ZRANGE i i+n-1` pages with
//!   `ZCARD` in each page's pipeline, and is complete only when its pages
//!   returned exactly the `ZCARD` read with the first page and the `ZCARD`
//!   read with the last page equals it. Accept-loss's value removal deletes
//!   as it walks, so it pages by descending absolute ranks from a `ZCARD`
//!   read before its first page: a row it deletes is at or above the current
//!   page's lowest rank, so no rank it has still to read shifts. A page that
//!   answers a number of members other than its ranks span ends its walk
//!   ([`WalkIncomplete::ShortPage`]), and after its last row batch the
//!   removal's end `ZCARD` must equal the start less the rows it deleted.
//! - **Hashes and sets by counted cursor walks** ([`CountedScan`],
//!   [`walk_counted`]). Each `HSCAN` / `SSCAN` step carries `HLEN` / `SCARD`
//!   in its pipeline; the caller de-duplicates the members exactly, and the
//!   walk is complete only when the distinct count equals the cardinality
//!   read with both its first and its last step. A step keeps fred's default
//!   attempts and follows a `-MOVED`: with no witness, which process answers
//!   does not matter.
//! - **Keyspace walks by an agreeing second walk**
//!   ([`KeyspaceWalk::start_job`], [`collect_keyspace_agreed`],
//!   [`keyspace_walks_agree`]). A keyspace has no count specific to a
//!   pattern, so a Job walks it twice and uses the result only when the two
//!   de-duplicated key sets are equal. Each walk keeps everything a guarded
//!   keyspace walk checks but the witness — the node pins, the myself-line
//!   ownership check at both ends, the slot map unchanged from start to end,
//!   no open slot, no dropped pin — and sends `SCAN … COUNT`
//!   [`DR_KEYSPACE_SCAN_COUNT`]. The second walk starts only after the first
//!   answered its end: two walks running at once could hold the same cursor
//!   when the store changes and skip the same keys. The ownership checks
//!   stay because a stale routing table would make both walks miss the same
//!   primary; an unusable myself line is a missing witness, exit 3.
//!
//! A proof that does not hold is a [`WalkIncomplete`]: the store changed
//! while the Job read it. The Jobs exit 1 on it, so Kubernetes re-runs them
//! against the store as it then is; nothing is walked again in-process.
//! What the proofs leave: two walks that each skip the identical non-empty
//! key set (two independent changes skipping the same keys at independently
//! timed cursors), which also bounds a same-node restart inside one keyspace
//! walk, which only the agreeing walk can catch; a miss both walks share
//! that the ownership checks do not see [untested: an Azure Managed Redis
//! endpoint whose myself line lists every slot during a scale-out]; a Job
//! write lost to a failover before the recount (the removal's end `ZCARD`
//! catches a lost delete on its own indices); and a writer that does not
//! honour the hold (SDD §11.3.2).
//!
//! **Coverage walks** ([`KeyspaceWalk::start_coverage`]). The service's
//! periodic keyspace walks act on each key as it arrives and run every few
//! seconds to minutes, so they take the keyspace walk's node pins and
//! ownership check without its serving-process witness, and never stop
//! before walking over a check that does not hold. fred plans its own
//! cluster-wide `SCAN` from the client's cached routing table, which it
//! refreshes only at a (re)connect, on a `MOVED` (which a keyless `SCAN` never
//! draws) or on an explicit sync: after a reshard or scale-out a pool member's
//! stale table would miss a whole primary on every pass. A coverage walk reads
//! a fresh `CLUSTER SLOTS` map at its start (a failed read is a store fault:
//! nothing is walked), resolves each primary's pin from one snapshot of the
//! client's table exactly as a guarded keyspace walk does, and reads each pin's
//! `CLUSTER NODES` (after `Route::ensure_routable`) for one verdict
//! ([`coverage_verdict`]):
//!
//! - the node serves exactly the primary's slots: the primary is walked;
//! - its myself line shows an open slot: walked, the walk recorded unproven
//!   ([`ServingChange::OpenSlot`]);
//! - an unusable myself line, or a `CLUSTER NODES` read that failed: walked,
//!   recorded unproven ([`WitnessMissing`], [`WalkError::Read`]), so a store
//!   that denies the command cannot stop the walks. The pin's ownership is
//!   then unproven: it came from the cached table, which may be stale, so the
//!   walk checks that table against its map instead (*The table check*,
//!   below);
//! - the node serves other slots, or the table names no node for the primary:
//!   not walked, recorded unproven ([`ServingChange::NotOwner`]), and the
//!   client's routing table re-synced, at most once per walk (on its first
//!   `NotOwner` or `StaleTable`, at the start or the end, or at the end of a
//!   walk with an unproven pin, *The table check*), so a later walk on
//!   that client is complete once its re-sync succeeds (a sync that failed,
//!   or that timed out and has not finished in the router, leaves the table
//!   stale, and the next walk misses the same primary and re-syncs again).
//!   The re-sync freshens only the walking pool member's table: each
//!   maintenance-pool member keeps its own, so with N members up to N walks
//!   (more when a sync fails) can each skip a primary before every table is
//!   fresh.
//!
//! A `NotOwner` outranks an open slot: a node that serves other slots is not
//! the primary's, whatever it is migrating.
//!
//! The walk then steps through the walked primaries one after another, each
//! step pinned to its node with one attempt, as a guarded keyspace walk does,
//! but with no witness: a transient step failure is resent at the same
//! cursor, up to [`WALK_STEP_ATTEMPTS`] attempts with the same backoff, and a
//! deterministic one ends the walk on its first attempt. At its end it reads
//! a fresh map again (a map that differs from the start's is recorded,
//! [`ServingChange::SlotMap`]) and each walked primary's verdict again
//! (anything but an owner is recorded; a `NotOwner` re-syncs if the walk has
//! not yet), then, when a pin's ownership is unproven, makes the end table
//! check. Standalone, it is the steps alone.
//!
//! **The table check.** A walk compares its map with a routing table
//! (`table_partition_matches`) at each end where, by then, a pin's ownership
//! is unproven (an unusable myself line or a failed `CLUSTER NODES` read): at
//! the start and again at the end when a start verdict was unproven, at the
//! end only when the first unproven verdict was an end one. The comparison
//! puts the table's ranges grouped by node against the map's grouped by
//! primary, as slot sets, so no host or TLS name spelling enters it. At the
//! start, once every start verdict is in, the table is the snapshot the pins
//! were resolved from: equal, each pin is the table's node for exactly its
//! primary's slots and no node is pinned twice. At the end, after the end
//! verdicts, it is the client's current table, against the end map, so a
//! change during the walk re-syncs early and the next walk on that client is
//! complete once the re-sync succeeds. A table that differs is out of date:
//! the walk is recorded as not covering its map
//! ([`ServingChange::StaleTable`]) and re-syncs the table, under the same
//! once-per-walk rule as a `NotOwner`. Each check follows its end's
//! verdicts, so a re-sync cannot make a `CLUSTER NODES` still to be read
//! refuse its pin; a step on a pin the start re-sync dropped is refused, a
//! store fault. A walk whose every pin was shown by its own myself line (an
//! owner or an open slot), at both ends, makes no comparison and no re-sync:
//! its verdicts already show each primary walked on its owner. What the
//! comparison cannot see: a change of primary that keeps every slot set (a
//! failover the table missed, whose demoted node answers the keyless `SCAN`
//! from its replica copy) [inferred, untested]. So a walk with an unproven
//! pin also re-syncs its table at its end when the end comparison matches,
//! under the same once-per-walk rule, recording nothing and keeping its
//! coverage, whatever the sync's outcome: the next walk on that client pins
//! from a fresh table, so a change the comparison cannot see lasts at most
//! one pass per maintenance-pool member (each member keeps its own table)
//! when the sync succeeds. A sync that fails or times out logs a WARN naming
//! the walk's pattern and the outcome: a failed sync leaves the table as it
//! was, and a timed-out one keeps running in the member's router and may
//! still finish, the walk having only stopped waiting for it, so the table
//! may stay stale.
//!
//! Each step's keys are handed on as they come. After the last cursor `0` and
//! the end checks, the walk answers one recorded reason, once, or completes:
//! the first coverage-losing reason (`NotOwner`, `SlotMap`, `StaleTable`) if
//! one was recorded, otherwise the first proof-only reason (`OpenSlot`, an
//! unusable myself line, a failed `CLUSTER NODES` read); a walk counted as
//! incomplete therefore always answers why. A store fault — a failed step, a
//! refused pin, a failed map read — ends it at once with the fault, and drops
//! any recorded reason.
//! [`KeyspaceWalk::walked_every_primary`] then tells a caller whether every
//! primary of the start map was walked to its end on its pin with the map
//! unchanged, no pin proven wrong and, where a pin's ownership was unproven,
//! the routing table matching the map at each end it was compared (no
//! `NotOwner`, no `SlotMap`, no `StaleTable`, no store fault), which is what
//! decides whether a walk recorded unproven can still be counted complete. It
//! is `true` for a walk whose only reasons are open slots, and unproven pins
//! whose table matched. `true` is not proof that no key was missed: an open
//! slot may skip keys, and an unproven pin can still be the old node in a
//! change the table check cannot see, for at most one pass per
//! maintenance-pool member when its end re-sync succeeds (above). With no
//! witness, a change of serving process within one walk can skip keys for
//! that walk only, undetected; the next walk covers them.
//!
//! **What a caller gets.** [`WalkError`]: a store fault (a step that failed
//! transiently on every attempt or deterministically on one, a guard read
//! that failed, or a node pin the client's
//! routing table no longer lists as a primary, refused before a step's
//! pipeline, a witness or a `CLUSTER NODES` read was sent), a
//! [`ServingChange`] (the
//! walk crossed a change of serving process or slot map, its node pin did
//! not reach the primary's owner, or, in a coverage walk, the routing table
//! its unproven pins came from was out of date), a [`WitnessMissing`], or,
//! from a DR Job's walk, a [`WalkIncomplete`] (its completeness proof did
//! not hold); a coverage walk answers its store fault the same way, and its
//! recorded reason (a `ServingChange`, a `WitnessMissing` or a failed
//! `CLUSTER NODES` read) only after its last key. No state is kept between
//! walks: the witness, the node pins, the fingerprint, a coverage walk's
//! recorded reason and table check, and a Job walk's counts and key sets
//! live and die with the walk on every exit.
//!
//! **The test seam.** Test and `test-fault-injection` builds only:
//! `arm_walk_fault_for_test` makes a chosen witness read of the walks of
//! one key or pattern report a forged run_id (or no run_id line), a chosen
//! step's whole pipeline fail transiently before it is sent (the shape a
//! lost connection gives; on a DR Job's rank walk, a chosen `ZCARD` or
//! `ZRANGE` read, numbered by the walk's reads that answered), a chosen
//! step's own reply in its pipeline read
//! as a transient error while its witnesses' replies stand, a chosen
//! witness in a step's pipeline read as a transient error reply, a chosen
//! `CLUSTER NODES` reply
//! of a keyspace walk read as given, a keyspace walk's primary be pinned to
//! a given node, or resolved to no node (so a guarded or a DR Job's walk
//! fails at its start with `NotOwner`, where a coverage walk skips the
//! primary), a chosen node-route pin check be refused, a chosen
//! primary's end-fingerprint witness report a forged run_id, a coverage
//! walk's chosen routing-table comparison report that the table differs, or
//! a chosen `CLUSTER SLOTS` read come back with its first two primaries'
//! names exchanged (the slots kept, so only the map comparison sees it), or
//! a chosen key or member be dropped from, or repeated in, a chosen step or
//! page reply — or every one — of a chosen DR Job walk of the key or
//! pattern (a keyspace walk's step, either walk of an agreeing pair; a rank
//! page; a counted `HSCAN` / `SSCAN` step), forging a skip so each
//! completeness proof can be shown to fail closed, or a repeat so each walk
//! can be shown to count a member once, until its guard drops; `forge_process_change_for_test` arms a change of
//! serving process at a chosen witness read, from `TEST_RUN_ID_A` to
//! `TEST_RUN_ID_B`. A coverage walk and a Job walk read no witness, so no
//! run_id fault matches them. `KeyspaceWalk::resynced_for_test` tells
//! whether a coverage walk re-synced the client's routing table, which the
//! end re-sync after a matching table shows nowhere else.

use std::sync::Arc;

use fred::clients::{Client, Pipeline};
use fred::interfaces::{ClientLike, ClusterInterface, SortedSetsInterface};
use fred::types::cluster::ClusterRouting;
use fred::types::config::{Options, Server};
use fred::types::{ClusterHash, CustomCommand, InfoKind, Key, Value};

use crate::storage::layer::{
    full_jitter_backoff, OVERWRITE_WAKE_BACKOFF_CAP, OVERWRITE_WAKE_BACKOFF_FLOOR,
};

/// How many times a guarded walk sends one step before it gives up, the
/// first attempt included (a DR Job's keyspace walk included, which resends a
/// transient failure at the same cursor as a coverage walk does): the bound
/// of the restore Job's write step
/// (`dr::job::execute::DR_WRITE_STEP_ATTEMPTS`, which a const assertion
/// there holds equal).
pub const WALK_STEP_ATTEMPTS: u32 = 3;

/// `COUNT` of a keyspace walk's `SCAN` step — a guarded walk's (the
/// periodic reconcile pass's) and a coverage walk's: the orphan sweep's page,
/// for the same reason (large enough to keep round trips down, small enough
/// not to hold the server's event loop). A DR Job's keyspace walk sends
/// [`DR_KEYSPACE_SCAN_COUNT`] instead.
pub const KEYSPACE_SCAN_COUNT: u64 = 100;

/// `COUNT` of a DR Job's keyspace walk's `SCAN` step
/// ([`KeyspaceWalk::start_job`]): ten times [`KEYSPACE_SCAN_COUNT`]. The Job
/// walks each keyspace pattern twice (module docs, *The DR Jobs' walks*)
/// while the fleet is held and nothing else uses the store, so a larger step
/// buys back the round trips the second walk costs (SDD §11.3.2 gives the
/// cost).
pub const DR_KEYSPACE_SCAN_COUNT: u64 = 1_000;

/// The length of a usable run_id: Valkey's `CONFIG_RUN_ID_SIZE`.
const RUN_ID_LEN: usize = 40;

/// What `INFO server` showed in place of a usable run_id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunIdSeen {
    /// The reply has no `run_id:` line.
    NoLine,
    /// The line's value, which is not exactly 40 hex characters, or is all
    /// zeros.
    Value(String),
}

impl std::fmt::Display for RunIdSeen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoLine => write!(f, "no run_id line"),
            Self::Value(value) => write!(f, "run_id {value:?}"),
        }
    }
}

/// The run_id an `INFO server` reply reports, when it is usable: exactly 40
/// hex characters, not all zeros. Anything else — no `run_id:` line, an empty
/// value, `0`, the wrong length or alphabet — is what was seen instead. A
/// placeholder is refused rather than compared: a value that never changes
/// would read "unchanged" forever and leave the walk unguarded.
pub fn parse_run_id(info: &str) -> Result<String, RunIdSeen> {
    let Some(value) = info.lines().find_map(|line| {
        line.trim_end_matches('\r')
            .strip_prefix("run_id:")
            .map(str::to_owned)
    }) else {
        return Err(RunIdSeen::NoLine);
    };
    let usable = value.len() == RUN_ID_LEN
        && value.bytes().all(|b| b.is_ascii_hexdigit())
        && value.bytes().any(|b| b != b'0');
    if usable {
        Ok(value)
    } else {
        Err(RunIdSeen::Value(value))
    }
}

/// One witness reading: the run_id of the process that answered, and the
/// node the client routed it to (its address on a cluster; `None`
/// standalone). Made only by a walk's own read, so a reader cannot forge
/// one.
#[derive(Clone, Debug)]
pub struct Witness {
    run_id: String,
    node: Option<String>,
}

impl Witness {
    /// The run_id.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The node it was read on: its address on a cluster, `None` standalone.
    pub fn node(&self) -> Option<&str> {
        self.node.as_deref()
    }
}

/// `node <address>` on a cluster, `the store` standalone.
fn node_label(node: Option<&str>) -> String {
    node.map_or_else(|| "the store".to_owned(), |node| format!("node {node}"))
}

/// What a `CLUSTER NODES` reply showed in place of a usable myself line
/// ([`parse_myself_line`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MyselfLineSeen {
    /// No line's flags include `myself`.
    NoLine,
    /// The myself line, which has fewer than the 8 fixed fields every node
    /// line carries before its slots: truncated or malformed, not a
    /// replica's (a replica's line has all 8 and no slot field).
    ShortLine(String),
    /// A slot field of the myself line that is neither an open-slot marker
    /// (`[…`) nor `N` / `A-B` with `0 ≤ A ≤ B ≤ 16383`.
    BadToken(String),
}

impl std::fmt::Display for MyselfLineSeen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoLine => write!(f, "no myself line"),
            Self::ShortLine(line) => write!(
                f,
                "a myself line with fewer than its 8 fixed fields ({line:?})"
            ),
            Self::BadToken(token) => write!(
                f,
                "a myself line with the slot field {token:?}, which is neither a slot nor a slot \
                 range"
            ),
        }
    }
}

/// What an unusable witness read showed, by the command that read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WitnessSeen {
    /// `INFO server` reported no usable run_id.
    RunId(RunIdSeen),
    /// A keyspace walk's `CLUSTER NODES` gave no usable myself line.
    MyselfLine(MyselfLineSeen),
}

impl WitnessSeen {
    /// The command that read it.
    pub fn command(&self) -> &'static str {
        match self {
            Self::RunId(_) => "INFO server",
            Self::MyselfLine(_) => "CLUSTER NODES",
        }
    }
}

/// A witness that is not usable: what `INFO server` (the run_id, read by the
/// periodic reconcile pass's guarded walks only) or, for a keyspace walk on
/// a cluster, `CLUSTER NODES` (the myself line) answered on the walk's
/// route. A fixed property of the store, not a fault a retry cures: the DR
/// Jobs exit 3 on an unusable myself line (the service stays held) rather
/// than walk a node they cannot show serves its primary; the periodic
/// reconcile pass forfeits on either. A coverage walk records an unusable
/// myself line and walks on. Not a fred error: it is never counted on
/// `valkey_errors_total`.
#[derive(Clone, Debug)]
pub struct WitnessMissing {
    /// The walk's command: `ZSCAN`, `HSCAN`, `SSCAN` or `SCAN`.
    pub walk: &'static str,
    /// Its key, or its pattern for a keyspace walk.
    pub target: Arc<str>,
    /// The node the read was routed to (`None` standalone).
    pub node: Option<String>,
    /// The command that read it, and what it showed.
    pub seen: WitnessSeen,
}

impl std::fmt::Display for WitnessMissing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let node = node_label(self.node.as_deref());
        match &self.seen {
            WitnessSeen::RunId(seen) => write!(
                f,
                "`INFO server` on {node} reported {seen}, not a usable run_id (40 hex characters, \
                 not all zeros), so the {} walk of {} cannot be shown to run on one server process",
                self.walk, self.target
            ),
            WitnessSeen::MyselfLine(seen) => write!(
                f,
                "`CLUSTER NODES` on {node} answered {seen}, so the {} walk of {} cannot be shown \
                 to reach the node that serves its primary's slots",
                self.walk, self.target
            ),
        }
    }
}

/// A set of hash slots, as sorted, disjoint, non-adjacent inclusive ranges,
/// so two sets are equal exactly when they hold the same slots. Ordered by
/// those ranges, so a list of sets can be sorted and compared whole (a
/// coverage walk's table check).
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SlotSet(Vec<(u16, u16)>);

impl SlotSet {
    /// The union of `ranges` (each `start <= end`), merged.
    pub fn from_ranges(ranges: impl IntoIterator<Item = (u16, u16)>) -> Self {
        let mut ranges: Vec<(u16, u16)> = ranges.into_iter().collect();
        ranges.sort_unstable();
        let mut merged: Vec<(u16, u16)> = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            match merged.last_mut() {
                Some(last) if u32::from(start) <= u32::from(last.1) + 1 => {
                    last.1 = last.1.max(end);
                }
                _ => merged.push((start, end)),
            }
        }
        Self(merged)
    }

    /// The merged ranges.
    pub fn ranges(&self) -> &[(u16, u16)] {
        &self.0
    }
}

impl std::fmt::Display for SlotSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return write!(f, "none");
        }
        let ranges: Vec<String> = self
            .0
            .iter()
            .map(|&(start, end)| {
                if start == end {
                    start.to_string()
                } else {
                    format!("{start}-{end}")
                }
            })
            .collect();
        write!(f, "{}", ranges.join(","))
    }
}

/// Which fingerprint of a keyspace walk found a [`ServingChange::NotOwner`],
/// or which end of a coverage walk found a [`ServingChange::StaleTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkEnd {
    /// The start fingerprint (before any step).
    Start,
    /// The end fingerprint.
    End,
}

impl WalkEnd {
    /// `start` or `end`, as an error line names it.
    fn label(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::End => "end",
        }
    }
}

/// How the routing-table re-sync after a [`ServingChange::NotOwner`], a
/// [`ServingChange::StaleTable`], or a coverage walk's end re-sync after a
/// matching table ended (the `refresh` field of that re-sync's WARN when it
/// failed or timed out).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingRefresh {
    /// `sync_cluster` completed: the client's table is fresh.
    Refreshed,
    /// `sync_cluster` failed with this error.
    Failed(String),
    /// `sync_cluster` did not finish while the caller waited: within the
    /// client's command timeout ([`refresh_routing`]), or within a shorter
    /// wait the caller ended earlier (the store-memory read's startup
    /// deadline, `storage::store_memory`). It keeps running in the member's
    /// router.
    TimedOut {
        /// How long the caller waited, in milliseconds: the command timeout,
        /// or the shorter wait the caller ended.
        waited_ms: u64,
    },
}

impl std::fmt::Display for RoutingRefresh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refreshed => write!(f, "it refreshed its routing table"),
            Self::Failed(e) => write!(f, "it could not refresh its routing table: {e}"),
            Self::TimedOut { waited_ms } => write!(
                f,
                "it could not refresh its routing table: the refresh did not finish within \
                 {waited_ms} ms"
            ),
        }
    }
}

/// A walk whose result cannot be used: it crossed a change of serving process
/// (a guarded walk of the periodic reconcile pass) or, for a keyspace walk on
/// a cluster, of slot ownership. The DR Jobs exit 1 on it, so Kubernetes
/// retries the Job and the walk is taken again from its start; the periodic
/// reconcile pass forfeits. A coverage walk records it and answers it after
/// its last key. Not a fred error: it is never counted on
/// `valkey_errors_total`.
#[derive(Clone, Debug)]
pub enum ServingChange {
    /// A witness read on the walk's route reported another run_id than the
    /// walk's first.
    Process {
        /// The walk's command.
        walk: &'static str,
        /// Its key or pattern.
        target: Arc<str>,
        /// The node the later witness was read on (`None` standalone).
        node: Option<String>,
        /// The run_id at the walk's start (for a keyspace walk, the
        /// primary's).
        before: String,
        /// The run_id read later.
        after: String,
    },
    /// A keyspace walk's `CLUSTER SLOTS` map at its end differs from its
    /// start.
    SlotMap {
        /// The pattern.
        target: Arc<str>,
        /// The start map, `start-end@host:port` per range.
        before: String,
        /// The end map.
        after: String,
    },
    /// A primary's myself line shows a slot migrating or importing.
    OpenSlot {
        /// The pattern.
        target: Arc<str>,
        /// The node it was read on: the pinned node (`host:port`).
        node: String,
        /// The open-slot markers, as `CLUSTER NODES` printed them.
        slots: String,
    },
    /// A keyspace walk's node pin did not reach the node that serves its
    /// primary's slots: the pinned node's own myself line lists other slots
    /// than the walk's map gives that primary (a replica's lists none), or
    /// the client's routing table names no node for the primary at all. The
    /// client's view of the cluster was out of date; the walk re-synced it
    /// (`refresh`) before failing (a coverage walk: before recording it,
    /// leaving the primary unwalked).
    NotOwner {
        /// The pattern.
        target: Arc<str>,
        /// The primary, as the walk's `CLUSTER SLOTS` map names it.
        primary: String,
        /// The node the pin went to (`host:port`), or `None` when the
        /// routing table named none.
        node: Option<String>,
        /// The slots the map gives the primary.
        expected: SlotSet,
        /// The slots the pinned node's myself line lists; `None` when there
        /// was no node to ask.
        served: Option<SlotSet>,
        /// Which fingerprint found it.
        at: WalkEnd,
        /// How the routing-table re-sync ended.
        refresh: RoutingRefresh,
    },
    /// A coverage walk that could not prove a pin's ownership (an unusable
    /// myself line, a failed `CLUSTER NODES` read) found that the client's
    /// routing table does not divide the slots among nodes as the walk's
    /// `CLUSTER SLOTS` map divides them among primaries
    /// (`table_partition_matches`): at the walk's start the table its pins
    /// were resolved from, at its end the client's current one. The client's
    /// view of the cluster was out of date, so an unproven pin may have read a
    /// node that does not serve its primary; the walk re-synced the table
    /// (`refresh`) before recording it. Never found by a guarded walk, which
    /// proves every pin from its own myself line.
    StaleTable {
        /// The pattern.
        target: Arc<str>,
        /// Which end of the walk found it.
        at: WalkEnd,
        /// The walk's map at that end, `start-end@host:port` per range.
        map: String,
        /// The routing table it was compared with, `start-end@host:port` per
        /// range, or `no routing table`.
        table: String,
        /// How the routing-table re-sync ended.
        refresh: RoutingRefresh,
    },
}

impl std::fmt::Display for ServingChange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Process {
                walk,
                target,
                node,
                before,
                after,
            } => write!(
                f,
                "the {walk} walk of {target} crossed a change of serving process on {}: run_id \
                 {before} before, run_id {after} after, so it may have skipped members",
                node_label(node.as_deref())
            ),
            Self::SlotMap {
                target,
                before,
                after,
            } => write!(
                f,
                "the SCAN walk of {target} saw the cluster's slot map change between its start \
                 ({before}) and its end ({after}), so it may have skipped keys"
            ),
            Self::OpenSlot {
                target,
                node,
                slots,
            } => write!(
                f,
                "the SCAN walk of {target} found a slot migrating or importing on node {node} \
                 ({slots}), so it may skip keys"
            ),
            Self::NotOwner {
                target,
                primary,
                node,
                expected,
                served,
                at,
                refresh,
            } => {
                let at = at.label();
                match (node, served) {
                    (Some(node), Some(served)) => write!(
                        f,
                        "the SCAN walk of {target} was routed for primary {primary} (slots \
                         {expected}) to {node}, which serves slots {served} at the walk's {at}: \
                         this client's view of the cluster was out of date, so the walk could \
                         skip keys; {refresh}"
                    ),
                    _ => write!(
                        f,
                        "the SCAN walk of {target} could not be routed for primary {primary} \
                         (slots {expected}): this client's routing table names no node for it at \
                         the walk's {at}, so its view of the cluster was out of date and the walk \
                         could skip keys; {refresh}"
                    ),
                }
            }
            Self::StaleTable {
                target,
                at,
                map,
                table,
                refresh,
            } => write!(
                f,
                "the SCAN walk of {target} could not prove a node it walked serves its primary, \
                 and at the walk's {at} this client's routing table ({table}) does not divide the \
                 slots among nodes as the cluster's slot map ({map}) divides them among \
                 primaries: this client's view of the cluster was out of date, so the walk could \
                 skip keys; {refresh}",
                at = at.label()
            ),
        }
    }
}

/// A DR Job's walk whose completeness proof did not hold (module docs, *The
/// DR Jobs' walks*): the store changed while the Job read it, so the walk
/// may have skipped members. The Jobs exit 1 on it, so Kubernetes retries
/// the Job, which walks again from the start against the store as it then
/// is; nothing is walked again in-process. Not a fred error: it is never
/// counted on `valkey_errors_total`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WalkIncomplete {
    /// A walk of one key ([`page_by_rank_counted`], [`walk_counted`], or the
    /// value removal's descending rank pages) whose members returned (the
    /// distinct ones of a counted walk) were not the key's cardinality at the
    /// walk's start, or whose cardinality at its
    /// end was not the one a complete walk leaves.
    CountMismatch {
        /// The walk's command: `ZRANGE`, `HSCAN` or `SSCAN`.
        walk: &'static str,
        /// The key.
        target: Arc<str>,
        /// The cardinality command: `ZCARD`, `HLEN` or `SCARD`.
        card: &'static str,
        /// The cardinality read at the walk's start.
        at_start: u64,
        /// The members the walk returned: the distinct ones of a counted
        /// walk; every member a rank walk's pages answered, a repeat
        /// included.
        returned: u64,
        /// The cardinality read at the walk's end.
        at_end: u64,
        /// The cardinality a complete walk leaves at its end: `at_start`, or,
        /// for the value removal, `at_start` less the rows it deleted from
        /// the key.
        expected_end: u64,
    },
    /// A page of the value removal's descending rank walk answered a number
    /// of members other than its ranks span: fewer (a member gone from below
    /// it), or more (only a forged repeat, the test seam's `RepeatMember`).
    ShortPage {
        /// The sorted index.
        target: Arc<str>,
        /// The page's lowest rank.
        start: u64,
        /// Its highest rank.
        stop: u64,
        /// The members it answered.
        returned: u64,
    },
    /// The two walks of an agreeing keyspace walk ([`keyspace_walks_agree`])
    /// returned different key sets.
    WalksDisagree {
        /// The pattern.
        target: Arc<str>,
        /// The distinct keys the first walk returned.
        first: u64,
        /// The distinct keys the second walk returned.
        second: u64,
        /// Keys only the first walk returned.
        only_first: u64,
        /// Keys only the second walk returned.
        only_second: u64,
    },
}

impl std::fmt::Display for WalkIncomplete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CountMismatch {
                walk,
                target,
                card,
                at_start,
                returned,
                at_end,
                expected_end,
            } => write!(
                f,
                "the {walk} walk of {target} returned {returned} members, and {card} \
                 read {at_start} at its start and {at_end} at its end, where a complete walk \
                 returns {at_start} and ends at {expected_end}"
            ),
            Self::ShortPage {
                target,
                start,
                stop,
                returned,
            } => write!(
                f,
                "ZRANGE {target} {start} {stop} returned {returned} members, where {} were \
                 expected",
                stop.saturating_sub(*start).saturating_add(1)
            ),
            Self::WalksDisagree {
                target,
                first,
                second,
                only_first,
                only_second,
            } => write!(
                f,
                "two SCAN walks of {target} returned different keys: {first} and {second} \
                 distinct keys, {only_first} found only by the first walk and {only_second} only \
                 by the second"
            ),
        }
    }
}

/// The count check of a DR Job's walk of one key (module docs, *The DR Jobs'
/// walks*): the `walk` walk of `target` returned `returned` members (the
/// distinct ones of a counted walk; every member a rank walk's pages
/// answered, a repeat included), its cardinality command `card` read
/// `at_start` at its start and `at_end` at its end, and a complete walk
/// leaves `expected_end`. Complete only when `returned == at_start` and
/// `at_end == expected_end`; anything else is a
/// [`WalkIncomplete::CountMismatch`]. Pure.
#[allow(clippy::too_many_arguments)]
pub fn check_count(
    walk: &'static str,
    target: &Arc<str>,
    card: &'static str,
    at_start: u64,
    returned: u64,
    at_end: u64,
    expected_end: u64,
) -> Result<(), WalkError> {
    if returned == at_start && at_end == expected_end {
        return Ok(());
    }
    Err(WalkError::incomplete(WalkIncomplete::CountMismatch {
        walk,
        target: Arc::clone(target),
        card,
        at_start,
        returned,
        at_end,
        expected_end,
    }))
}

/// The comparison of an agreeing keyspace walk's two walks of `pattern`
/// (module docs, *The DR Jobs' walks*): `first` and `second`, each walk's
/// de-duplicated keys (or what the caller derived from them, the same way
/// for both). Equal sets agree; anything else is a
/// [`WalkIncomplete::WalksDisagree`] naming how many keys each walk alone
/// returned. Pure.
pub fn keyspace_walks_agree<T: Eq + std::hash::Hash>(
    pattern: &str,
    first: &std::collections::HashSet<T>,
    second: &std::collections::HashSet<T>,
) -> Result<(), WalkError> {
    if first == second {
        return Ok(());
    }
    Err(WalkError::incomplete(WalkIncomplete::WalksDisagree {
        target: Arc::from(pattern),
        first: first.len() as u64,
        second: second.len() as u64,
        only_first: first.difference(second).count() as u64,
        only_second: second.difference(first).count() as u64,
    }))
}

/// Why a guarded walk, or a DR Job's walk, ended without a usable result.
#[derive(Debug)]
pub enum WalkError {
    /// A step failed transiently on every attempt, or failed with a
    /// deterministic error on its first (or its reply was not a scan reply),
    /// or a keyspace walk's node pin is no longer a primary in the client's
    /// routing table, so nothing more was sent to it: that can stop the walk
    /// before any step attempt, or before a witness or `CLUSTER NODES` read
    /// (`Route::ensure_routable`). A DR Job's walk of one key ([`RankWalk`],
    /// [`CountedScan`]) answers a failed page, step or cardinality read
    /// here too, its command as `walk`.
    Step {
        /// The walk's command.
        walk: &'static str,
        /// Its key or pattern.
        target: Arc<str>,
        /// The last attempt's error.
        source: fred::error::Error,
    },
    /// A guard read failed: `INFO server`, `CLUSTER SLOTS` or `CLUSTER
    /// NODES` (`op`), or the `CLUSTER SLOTS` reply could not be read.
    Read {
        /// The command.
        op: &'static str,
        /// Its error.
        source: fred::error::Error,
    },
    /// The walk crossed a change of serving process or slot map, or its node
    /// pin did not reach its primary's slots: exit 1. (A coverage walk also
    /// records an out-of-date routing table here, `StaleTable`.) Boxed: the
    /// `NotOwner` case carries two slot sets ([`WalkError::changed`] builds
    /// it).
    Changed(Box<ServingChange>),
    /// `INFO server` reported no usable run_id, or `CLUSTER NODES` no usable
    /// myself line: exit 3.
    WitnessMissing(WitnessMissing),
    /// A DR Job's walk whose completeness proof did not hold: exit 1. Boxed:
    /// [`WalkError::incomplete`] builds it.
    Incomplete(Box<WalkIncomplete>),
}

impl WalkError {
    /// A [`WalkError::Changed`] for `change`.
    pub fn changed(change: ServingChange) -> Self {
        Self::Changed(Box::new(change))
    }

    /// A [`WalkError::Incomplete`] for `incomplete`.
    pub fn incomplete(incomplete: WalkIncomplete) -> Self {
        Self::Incomplete(Box::new(incomplete))
    }

    /// The fred error a store fault carries, for the caller's error metric;
    /// `None` for a change, a missing witness or an incomplete walk, which
    /// are not fred errors.
    pub fn fred_error(&self) -> Option<&fred::error::Error> {
        match self {
            Self::Step { source, .. } | Self::Read { source, .. } => Some(source),
            Self::Changed(_) | Self::WitnessMissing(_) | Self::Incomplete(_) => None,
        }
    }
}

impl std::fmt::Display for WalkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Step {
                walk,
                target,
                source,
            } => write!(f, "a {walk} step of {target} failed: {source}"),
            Self::Read { op, source } => write!(f, "{op} failed: {source}"),
            Self::Changed(change) => write!(f, "{change}"),
            Self::WitnessMissing(missing) => write!(f, "{missing}"),
            Self::Incomplete(incomplete) => write!(f, "{incomplete}"),
        }
    }
}

impl std::error::Error for WalkError {}

/// Where a walk's commands go.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Route {
    /// A keyed walk on a cluster: the key's slot (`cluster_hash`), so fred
    /// routes it by its table and the server enforces the key's ownership
    /// (a foreign slot answers `-MOVED`, a migrating one `-ASK`).
    Slot(u16),
    /// A keyspace walk's primary on a cluster: one named node
    /// (`cluster_node`), with no fallback to another connection. Every
    /// command on it is sent only while the client's routing table still
    /// lists the node as a primary ([`Route::ensure_routable`]).
    Node(Server),
    /// Standalone: neither.
    None,
}

impl Route {
    /// The options of one of the walk's commands: the route's pin, and one
    /// attempt for a step (`one_attempt`); a witness or fingerprint read
    /// keeps fred's default attempts.
    fn options(&self, one_attempt: bool) -> Options {
        let (cluster_hash, cluster_node) = match self {
            Self::Slot(slot) => (Some(ClusterHash::Custom(*slot)), None),
            Self::Node(server) => (None, Some(server.clone())),
            Self::None => (None, None),
        };
        Options {
            cluster_hash,
            cluster_node,
            max_attempts: one_attempt.then_some(1),
            ..Default::default()
        }
    }

    /// The node an error names (`host:port`): the pinned server for a node
    /// route; for a slot route, the server the client's table routes the
    /// slot to; `None` standalone.
    fn node(&self, client: &Client) -> Option<String> {
        match self {
            Self::Slot(slot) => client
                .cached_cluster_state()
                .and_then(|routing| routing.get_server(*slot).map(ToString::to_string)),
            Self::Node(server) => Some(server.to_string()),
            Self::None => None,
        }
    }

    /// For a node route, whether a command may be sent on it: only while the
    /// client's current routing table lists the pinned node as a primary
    /// ([`pinned_node_listed`]). A node-pinned command that finds no
    /// connection does not fail alone: fred answers it by reconnecting the
    /// whole client (`defer_reconnection` with no server, then a full
    /// `reconnect_with_policy`, `router/commands.rs` 146–153, 343–376), and
    /// the walk client is a shared pool member, so every witness read, step
    /// and retry sent to a node the table has dropped would drop and re-send
    /// its other users' commands in flight. So a pin the table no longer
    /// lists is refused, and the refusal ends the walk at once, sending
    /// nothing, on every path, an R1 retry's witness re-read included. It is
    /// checked once before each command, or once before a step's whole
    /// pipeline: fred writes the pipeline's commands in one pass against one
    /// table, so the witness riding with the step needs no check of its own.
    /// A refusal is the `Step` error naming the node ([`route_refused`];
    /// exit 1 in the DR Jobs). The table drops a node when its slots moved,
    /// and also when a sync dropped a broken connection to it that is not
    /// re-established yet, so a refusal says the walk's connection or routing
    /// view of that node changed. A refusal never spends an R1 attempt and
    /// triggers no routing re-sync: the table it read was just rebuilt by a
    /// completed sync. It reads fred's cached table only. A slot route or a
    /// standalone one is never refused.
    fn ensure_routable(
        &self,
        client: &Client,
        faults: &WalkFaults,
        walk: &'static str,
        target: &Arc<str>,
    ) -> Result<(), WalkError> {
        let Self::Node(server) = self else {
            return Ok(());
        };
        let refused_for_test = faults.refuse_route();
        if !refused_for_test && pinned_node_listed(client.cached_cluster_state().as_ref(), server) {
            return Ok(());
        }
        Err(route_refused(walk, target, server))
    }
}

/// The refusal of a node pin the walk client's routing table no longer lists
/// as a primary ([`Route::ensure_routable`]): the `Step` error of the `walk`
/// walk of `target`, its source a `Routing` error naming `server`. Pure.
pub fn route_refused(walk: &'static str, target: &Arc<str>, server: &Server) -> WalkError {
    WalkError::Step {
        walk,
        target: Arc::clone(target),
        source: fred::error::Error::new(
            fred::error::ErrorKind::Routing,
            format!(
                "pinned node {server} is no longer a primary in this client's routing table: the \
                 walk's connection or routing view of that node changed, so the walk sends \
                 nothing more to it"
            ),
        ),
    }
}

/// The verdict of an R1 retry's witness re-read (`reread`), taken after a
/// failed step attempt, against the walk's `start` witness: `Ok(())` to
/// send the step again, or the error that ends the walk. An unchanged
/// witness lets the step be retried; another run_id ends the walk with the
/// change. An `INFO server` that returned an error (`WalkError::Read`) only
/// spends the attempt: it proves nothing either way, and the witness read
/// after a step that does land still decides. Anything else — a missing
/// witness, a refused node pin ([`route_refused`]) — ends the walk as it is.
/// Pure.
pub fn after_failed_attempt(
    walk: &'static str,
    target: &Arc<str>,
    start: &Witness,
    reread: Result<Witness, WalkError>,
) -> Result<(), WalkError> {
    match reread {
        Ok(now) => match changed_between(walk, target, start, &now) {
            Some(change) => Err(WalkError::changed(change)),
            None => Ok(()),
        },
        Err(WalkError::Read { .. }) => Ok(()),
        Err(ended) => Err(ended),
    }
}

/// Whether `routing` (a client's cached routing table) lists `server` as a
/// primary, compared as fred keys its connections (`Server`'s own equality,
/// host and port in the table's spelling, which the pin was cloned from). No
/// table lists nothing. Pure.
pub fn pinned_node_listed(routing: Option<&ClusterRouting>, server: &Server) -> bool {
    routing.is_some_and(|routing| routing.unique_primary_nodes().contains(server))
}

/// `INFO server` on `route`, read alone as a witness of the `walk` walk of
/// `target`: a keyspace walk's start and end fingerprints, and an R1 retry's
/// re-read. A step's own witnesses ride in its pipeline instead
/// ([`send_guarded_step`]). Keeps fred's retries (module docs).
async fn read_witness(
    client: &Client,
    route: &Route,
    faults: &WalkFaults,
    walk: &'static str,
    target: &Arc<str>,
) -> Result<Witness, WalkError> {
    route.ensure_routable(client, faults, walk, target)?;
    let info: String = client
        .with_options(&route.options(false))
        .info(Some(InfoKind::Server))
        .await
        .map_err(|source| WalkError::Read {
            op: "INFO server",
            source,
        })?;
    let info = faults.witness_reply(info);
    witness_of(walk, target, route.node(client), &info)
}

/// The witness an `INFO server` reply (`info`, already through the test
/// seam) reports for the `walk` walk of `target` on `node`: its run_id, or a
/// missing witness when it reports no usable one. Pure.
fn witness_of(
    walk: &'static str,
    target: &Arc<str>,
    node: Option<String>,
    info: &str,
) -> Result<Witness, WalkError> {
    match parse_run_id(info) {
        Ok(run_id) => Ok(Witness { run_id, node }),
        Err(seen) => Err(WalkError::WitnessMissing(WitnessMissing {
            walk,
            target: Arc::clone(target),
            node,
            seen: WitnessSeen::RunId(seen),
        })),
    }
}

/// `later` against the walk's `start`: the change, when the run_id differs.
fn changed_between(
    walk: &'static str,
    target: &Arc<str>,
    start: &Witness,
    later: &Witness,
) -> Option<ServingChange> {
    (later.run_id != start.run_id).then(|| process_change(walk, target, start, later))
}

/// The change from `start` to `later`, two witnesses of one walk, named by
/// `later`'s node, or `start`'s when `later` names none.
fn process_change(
    walk: &'static str,
    target: &Arc<str>,
    start: &Witness,
    later: &Witness,
) -> ServingChange {
    ServingChange::Process {
        walk,
        target: Arc::clone(target),
        node: later.node.clone().or_else(|| start.node.clone()),
        before: start.run_id.clone(),
        after: later.run_id.clone(),
    }
}

/// Whether a failed step is worth sending again at the same cursor (R1):
/// only a transient error (`storage::client::is_transient_fred_error`: a
/// timeout, an I/O or connection failure, a router that had no connection
/// or spent its attempt budget, a store not yet serving, `TRYAGAIN`). A
/// deterministic one — a `WRONGTYPE`, any other server error, a reply that
/// is not a scan reply — would answer the same again, so it ends the walk on
/// its first attempt. Pure.
pub fn step_error_is_transient(e: &fred::error::Error) -> bool {
    crate::storage::client::is_transient_fred_error(e)
}

/// The witnesses a guarded step's pipeline carries beside the step (module
/// docs, *Keyed walks* and *A step's pipeline*).
#[derive(Clone, Copy, Debug)]
enum WitnessMode<'w> {
    /// A coverage walk's step: no witness, the step alone.
    None,
    /// A keyed walk's first step: `[INFO server; step; INFO server]`, the
    /// walk's start witness, the step and its after-step witness.
    Lead,
    /// A later keyed step, or a guarded keyspace walk's step:
    /// `[step; INFO server]`, the walk's start witness (carried) already
    /// read.
    After(&'w Witness),
}

/// One attempt of a guarded step, as its pipeline's replies read
/// ([`read_step_pipeline`]), when the walk goes on.
#[derive(Debug)]
enum StepAttempt {
    /// The step answered a scan reply, and its witnesses were usable.
    Landed {
        /// The start witness the pipeline led with; `None` when it led with
        /// none.
        lead: Option<Witness>,
        /// The next cursor (`"0"` once the walk is complete).
        next: String,
        /// The step's elements.
        elements: Vec<Value>,
        /// The after-step witness; `None` in a coverage walk.
        witness: Option<Witness>,
    },
    /// The attempt failed transiently: the pipeline as a whole, or the step
    /// alone (R1 decides whether it is sent again).
    Failed {
        /// The start witness the pipeline led with, when it answered (the
        /// step failed alone); `None` otherwise.
        lead: Option<Witness>,
        /// The attempt's error.
        error: fred::error::Error,
    },
}

/// Read one attempt of a guarded step from its pipeline's replies
/// (`results`, fred's `try_all`): a start witness first when the pipeline
/// led with one (`lead`), then the step, then its after-step witness when it
/// carried one (`after`), each witness of the `walk` walk of `target` on
/// `node`. Module docs, *A step's pipeline*:
///
/// - one reply per command: judged in command order. A witness that answered
///   an error is a `Read` error and one with no usable run_id a missing
///   witness; each witness actually read goes through the test seam's
///   `witness_reply` (and `WitnessReplyError`), the start witness first, and
///   a guarded step's reply (its pipeline carries a witness) through its
///   `StepReplyError`. The step's error reply is a
///   failed attempt when transient ([`step_error_is_transient`]) and a
///   `Step` error otherwise, its after-step witness left unread; its reply
///   must be a scan reply ([`split_step_reply`], else a `Step` error);
/// - one error, fewer replies than commands: the pipeline failed as a whole,
///   a failed attempt of the step when transient, a `Step` error otherwise;
/// - any other count (none, too many, or one reply that is not an error, as
///   fred answers a single non-array frame): a `Protocol` `Step` error, so
///   no reply of it is read as a witness or as the step's.
///
/// `Err` ends the walk. Pure but for the seam's counts.
fn read_step_pipeline(
    walk: &'static str,
    target: &Arc<str>,
    lead: bool,
    after: bool,
    node: Option<String>,
    faults: &WalkFaults,
    results: Vec<Result<Value, fred::error::Error>>,
) -> Result<StepAttempt, WalkError> {
    let step_err = |source| WalkError::Step {
        walk,
        target: Arc::clone(target),
        source,
    };
    // A step error, the pipeline's or the step's own: a failed attempt when
    // transient, the end of the walk when not.
    let failed = |lead: Option<Witness>, error: fred::error::Error| {
        if step_error_is_transient(&error) {
            Ok(StepAttempt::Failed { lead, error })
        } else {
            Err(step_err(error))
        }
    };
    let sent = 1 + usize::from(lead) + usize::from(after);
    if results.len() != sent {
        let answered = results.len();
        return match <[_; 1]>::try_from(results) {
            Ok([Err(error)]) => failed(None, error),
            _ => Err(step_err(fred::error::Error::new(
                fred::error::ErrorKind::Protocol,
                format!(
                    "the pipeline of a {walk} step of {target} answered {answered} replies to its \
                     {sent} commands"
                ),
            ))),
        };
    }
    let read_witness_reply = |reply: Option<Result<Value, fred::error::Error>>| {
        // The seam's `WitnessReplyError`: this witness's reply read as a
        // failure (never in a release build).
        let reply = match faults.witness_reply_error() {
            Some(forged) => Some(Err(forged)),
            None => reply,
        };
        let info: String = reply
            .unwrap_or_else(|| {
                Err(fred::error::Error::new(
                    fred::error::ErrorKind::Protocol,
                    "a step's pipeline answered no reply for its witness",
                ))
            })
            .and_then(Value::convert::<String>)
            .map_err(|source| WalkError::Read {
                op: "INFO server",
                source,
            })?;
        witness_of(walk, target, node.clone(), &faults.witness_reply(info))
    };
    // Only a pipeline that carries a witness: a coverage step's one reply
    // cannot be told from a whole-pipeline failure.
    let carries_witness = lead || after;
    let mut results = results.into_iter();
    let lead = if lead {
        Some(read_witness_reply(results.next())?)
    } else {
        None
    };
    // The seam's `StepReplyError` may read a guarded step's reply as a
    // transient error (never in a release build).
    let reply = results.next();
    let reply = if carries_witness {
        reply.map(|reply| faults.step_reply(walk, reply))
    } else {
        reply
    };
    let reply = match reply {
        Some(Ok(reply)) => reply,
        Some(Err(error)) => return failed(lead, error),
        None => {
            return Err(step_err(fred::error::Error::new(
                fred::error::ErrorKind::Protocol,
                "a step's pipeline answered no reply for its step",
            )))
        }
    };
    let (next, elements) = split_step_reply(walk, target, reply).map_err(step_err)?;
    let witness = if after {
        Some(read_witness_reply(results.next())?)
    } else {
        None
    };
    Ok(StepAttempt::Landed {
        lead,
        next,
        elements,
        witness,
    })
}

/// Queue the witness, `INFO server` on `route` at fred's default attempts,
/// on a step's pipeline.
async fn queue_witness(pipe: &Pipeline<Client>, route: &Route) -> Result<(), WalkError> {
    let _: Value = pipe
        .with_options(&route.options(false))
        .info(Some(InfoKind::Server))
        .await
        .map_err(|source| WalkError::Read {
            op: "INFO server",
            source,
        })?;
    Ok(())
}

/// A guarded step that landed ([`send_guarded_step`]).
struct LandedStep {
    /// The start witness the step's pipeline led with, in this attempt or an
    /// earlier one whose step alone failed; `None` when no pipeline led with
    /// one.
    lead: Option<Witness>,
    /// The next cursor (`"0"` once the walk is complete).
    next: String,
    /// The step's elements.
    elements: Vec<Value>,
    /// The after-step witness; `None` in a coverage walk.
    witness: Option<Witness>,
}

/// Send one step, queued on a fresh pipeline of `client` by `queue` (with
/// the route's one-attempt options), with the witnesses `mode` names
/// (module docs, *Keyed walks*), at most `attempts` times: one pin check
/// before each attempt's pipeline ([`Route::ensure_routable`]), its replies
/// read by [`read_step_pipeline`]. After an attempt that failed transiently
/// the walk's start witness — the one `mode` carries, or one an earlier
/// attempt's pipeline led with — is re-read alone against itself (module
/// docs, R1; [`after_failed_attempt`]) and, when it is unchanged, the step is
/// sent again at the same cursor with its after-step witness. A keyed walk's
/// first pipeline that failed as a whole has no start witness, so it is
/// resent whole, start witness included, with no re-read; a coverage walk
/// (`mode` `None`) resends its step with no re-read. A deterministic error
/// ends the walk on that attempt. The landed step, or the deterministic
/// error, witness error, change, missing witness or refused node pin that
/// ended it, or the last attempt's error.
#[allow(clippy::too_many_arguments)]
async fn send_guarded_step<Q, Fut>(
    client: &Client,
    route: &Route,
    faults: &WalkFaults,
    walk: &'static str,
    target: &Arc<str>,
    mode: WitnessMode<'_>,
    attempts: u32,
    queue: Q,
) -> Result<LandedStep, WalkError>
where
    Q: Fn(Pipeline<Client>) -> Fut,
    Fut: std::future::Future<Output = Result<Value, fred::error::Error>>,
{
    let attempts = attempts.max(1);
    let (guarded, passed) = match mode {
        WitnessMode::None => (false, None),
        WitnessMode::Lead => (true, None),
        WitnessMode::After(start) => (true, Some(start)),
    };
    // The start witness an attempt's pipeline led with, kept once it
    // answered, so a later attempt sends `[step; INFO server]`.
    let mut lead: Option<Witness> = None;
    let mut last = None;
    for attempt in 0..attempts {
        if attempt > 0 {
            tokio::time::sleep(full_jitter_backoff(
                attempt - 1,
                OVERWRITE_WAKE_BACKOFF_FLOOR,
                OVERWRITE_WAKE_BACKOFF_CAP,
            ))
            .await;
        }
        // A node pin the client's table has dropped ends the walk here,
        // with nothing sent (`Route::ensure_routable`); one check covers the
        // whole pipeline.
        route.ensure_routable(client, faults, walk, target)?;
        let leads = guarded && passed.is_none() && lead.is_none();
        let results = match faults.step_error(walk) {
            // The seam's step fault: the whole pipeline fails before it is
            // sent, as on a lost connection.
            Some(injected) => vec![Err(injected)],
            None => {
                let pipe = client.pipeline();
                if leads {
                    queue_witness(&pipe, route).await?;
                }
                match queue(pipe.clone()).await {
                    // Not queued, so nothing is sent: the attempt's error.
                    Err(e) => vec![Err(e)],
                    Ok(_) => {
                        if guarded {
                            queue_witness(&pipe, route).await?;
                        }
                        pipe.try_all::<Value>().await
                    }
                }
            }
        };
        let node = if guarded { route.node(client) } else { None };
        match read_step_pipeline(walk, target, leads, guarded, node, faults, results)? {
            StepAttempt::Landed {
                lead: read,
                next,
                elements,
                witness,
            } => {
                faults.step_succeeded();
                return Ok(LandedStep {
                    lead: read.or(lead),
                    next,
                    elements,
                    witness,
                });
            }
            StepAttempt::Failed { lead: read, error } => {
                if read.is_some() {
                    lead = read;
                }
                last = Some(error);
            }
        }
        // R1: a separate re-read against the walk's start, never the
        // witness that rode in the failed pipeline.
        if let Some(start) = passed.or(lead.as_ref()) {
            after_failed_attempt(
                walk,
                target,
                start,
                read_witness(client, route, faults, walk, target).await,
            )?;
        }
    }
    Err(WalkError::Step {
        walk,
        target: Arc::clone(target),
        source: last.unwrap_or_else(|| {
            fred::error::Error::new(fred::error::ErrorKind::Unknown, "no step attempt was made")
        }),
    })
}

/// A scan step's reply: the next cursor and its elements. Anything else is a
/// `Protocol` error naming the walk.
fn split_step_reply(
    walk: &'static str,
    target: &str,
    reply: Value,
) -> Result<(String, Vec<Value>), fred::error::Error> {
    let shape_err = || {
        fred::error::Error::new(
            fred::error::ErrorKind::Protocol,
            format!("{walk} {target} returned an unexpected reply"),
        )
    };
    let mut parts = reply.into_array().into_iter();
    let (Some(next), Some(elements), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(shape_err());
    };
    let next = next.as_string().ok_or_else(shape_err)?;
    let Value::Array(elements) = elements else {
        return Err(shape_err());
    };
    Ok((next, elements))
}

/// One step of a guarded keyed walk that landed ([`KeyedWalk::step`]), or
/// one page of the cursor pager (`storage::retirement::CursorPageReader::
/// read_step`, whose members are the route's reading of the step's
/// elements): the walk's start witness, the next cursor, the members and
/// the step's own after-step witness. The step's members may be used only
/// once `witness` reads `start`'s run_id (the cursor pager checks it).
#[derive(Debug)]
pub struct GuardedStep<M> {
    /// The walk's start witness: the one the step was given, or, on the
    /// walk's first step, the one its pipeline led with.
    pub start: Witness,
    /// The next cursor (`"0"` once the walk is complete).
    pub next: String,
    /// The step's members: from [`KeyedWalk::step`], its reply's elements —
    /// a set's members, or a hash's fields and values or a sorted set's
    /// members and scores interleaved.
    pub members: Vec<M>,
    /// The witness that rode with the step, answered by the process that
    /// answered the step.
    pub witness: Witness,
}

impl<M> GuardedStep<M> {
    /// The same step with its members read by `read` (a route's reading of
    /// a step's elements), or `read`'s error.
    pub fn try_map_members<N, E>(
        self,
        read: impl FnOnce(Vec<M>) -> Result<Vec<N>, E>,
    ) -> Result<GuardedStep<N>, E> {
        Ok(GuardedStep {
            start: self.start,
            next: self.next,
            members: read(self.members)?,
            witness: self.witness,
        })
    }
}

/// A guarded walk of one key's members by `ZSCAN`, `HSCAN` or `SSCAN` steps
/// (module docs, *Keyed walks*), on the one client it holds: the periodic
/// reconcile pass's owed walks. The cursor pager
/// (`storage::retirement::page_by_cursor`) drives it: each step's pipeline
/// carries the step's witness, the first step's the walk's start witness
/// too. A DR Job's walk of one key reads no witness ([`RankWalk`],
/// [`CountedScan`]).
pub struct KeyedWalk<'c> {
    client: &'c Client,
    walk: &'static str,
    key: Arc<str>,
    route: Route,
    faults: WalkFaults,
}

impl<'c> KeyedWalk<'c> {
    /// A walk of `key` by `command` steps on `client`, which it holds for
    /// every step and witness: routed by the key's slot on a cluster,
    /// unpinned standalone.
    pub fn new(client: &'c Client, command: &'static str, key: &str) -> Self {
        let route = if client.is_clustered() {
            Route::Slot(fred::util::redis_keyslot(key.as_bytes()))
        } else {
            Route::None
        };
        Self {
            client,
            walk: command,
            key: Arc::from(key),
            route,
            faults: WalkFaults::for_target(key),
        }
    }

    /// The walked key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The walk's command.
    pub fn command(&self) -> &'static str {
        self.walk
    }

    /// The change from `start`, the walk's first witness, to `later`, a
    /// witness that read another run_id: what the cursor pager hands its
    /// reader when the two differ.
    pub fn change(&self, start: &Witness, later: &Witness) -> ServingChange {
        process_change(self.walk, &self.key, start, later)
    }

    /// One step from `cursor`, asking for `count` elements, in one pipeline
    /// with its witnesses (module docs, *Keyed walks*): with `start`, the
    /// walk's start witness, `[step; INFO server]`; without (the walk's first
    /// step), `[INFO server; step; INFO server]`, whose lead witness becomes
    /// the start. The step's [`GuardedStep`]: the start, the next cursor
    /// (`"0"` once the walk is complete), the step's elements and its
    /// after-step witness, which the caller compares with the start before
    /// using the elements. The step is sent with one attempt, at most
    /// `attempts` times, the start re-read alone after each transient failure
    /// (a first pipeline that failed as a whole is resent whole instead); a
    /// deterministic error ends the walk on its first attempt (module docs,
    /// R1).
    pub async fn step(
        &self,
        cursor: &str,
        count: u64,
        start: Option<&Witness>,
        attempts: u32,
    ) -> Result<GuardedStep<Value>, WalkError> {
        let (walk, route, key) = (self.walk, &self.route, &*self.key);
        let mode = match start {
            Some(start) => WitnessMode::After(start),
            None => WitnessMode::Lead,
        };
        let landed = send_guarded_step(
            self.client,
            route,
            &self.faults,
            walk,
            &self.key,
            mode,
            attempts,
            move |pipe| async move {
                pipe.with_options(&route.options(true))
                    .custom::<Value, Value>(
                        CustomCommand::new_static(walk, ClusterHash::FirstValue, false),
                        vec![
                            Value::from(key),
                            Value::from(cursor),
                            Value::from("COUNT"),
                            Value::from(i64::try_from(count).unwrap_or(i64::MAX)),
                        ],
                    )
                    .await
            },
        )
        .await?;
        // A keyed step's pipeline always carries its after-step witness, and
        // the first one its lead: [`read_step_pipeline`] answers both or
        // fails. Fail closed all the same.
        let (Some(start), Some(witness)) = (start.cloned().or(landed.lead), landed.witness) else {
            return Err(WalkError::Step {
                walk,
                target: Arc::clone(&self.key),
                source: fred::error::Error::new(
                    fred::error::ErrorKind::Protocol,
                    "a keyed step's pipeline answered without its witnesses",
                ),
            });
        };
        Ok(GuardedStep {
            start,
            next: landed.next,
            members: landed.elements,
            witness,
        })
    }
}

/// A rank as `ZRANGE` takes it.
fn zrange_rank(rank: u64) -> i64 {
    i64::try_from(rank).unwrap_or(i64::MAX)
}

/// A DR Job's walk of one sorted set by rank pages (module docs, *The DR
/// Jobs' walks*): `ZRANGE key start stop` pages, a pure function of the
/// set's content, and `ZCARD`, keyed reads routed by the key's slot on the
/// client the walk holds, with no witness. The recount drives it through
/// [`page_by_rank_counted`]; accept-loss's value removal pages it by
/// descending ranks itself, each read one write step.
pub struct RankWalk<'c> {
    client: &'c Client,
    key: Arc<str>,
    one_attempt: bool,
    faults: WalkFaults,
}

impl<'c> RankWalk<'c> {
    /// A walk of the sorted set `key` on `client`. Each read keeps fred's
    /// default attempts, or is sent with one attempt when `one_attempt` (a
    /// caller that retries each read itself, as accept-loss's value removal
    /// does through its write step, so the two bounds do not multiply).
    pub fn new(client: &'c Client, key: &str, one_attempt: bool) -> Self {
        Self {
            client,
            key: Arc::from(key),
            one_attempt,
            faults: WalkFaults::for_target(key),
        }
    }

    /// The walked key.
    pub fn key(&self) -> &Arc<str> {
        &self.key
    }

    /// The options of each read: one attempt, or fred's default.
    fn options(&self) -> Options {
        Options {
            max_attempts: self.one_attempt.then_some(1),
            ..Default::default()
        }
    }

    /// A failed `walk` read of the key, as the walk's error.
    fn failed(&self, walk: &'static str) -> impl Fn(fred::error::Error) -> WalkError + '_ {
        move |source| WalkError::Step {
            walk,
            target: Arc::clone(&self.key),
            source,
        }
    }

    /// `ZCARD` of the key, alone: the value removal's count at an index's
    /// start and at its end. The test seam's `StepError` fault may fail it
    /// transiently before it is sent (never in a release build).
    pub async fn card(&self) -> Result<u64, WalkError> {
        if let Some(injected) = self.faults.rank_read_error("ZCARD") {
            return Err(self.failed("ZCARD")(injected));
        }
        let card = self
            .client
            .with_options(&self.options())
            .zcard::<u64, _>(&*self.key)
            .await
            .map_err(self.failed("ZCARD"))?;
        self.faults.rank_read_answered();
        Ok(card)
    }

    /// The members at ranks `start..=stop`, alone: one page of the value
    /// removal's descending walk. The test seam's `StepError` fault may fail
    /// it transiently before it is sent, and its `DropMember` or
    /// `RepeatMember` fault may drop or repeat a member of it (never in a
    /// release build).
    pub async fn page(&self, start: u64, stop: u64) -> Result<Vec<Value>, WalkError> {
        if let Some(injected) = self.faults.rank_read_error("ZRANGE") {
            return Err(self.failed("ZRANGE")(injected));
        }
        let mut members: Vec<Value> = self
            .client
            .with_options(&self.options())
            .zrange(
                &*self.key,
                zrange_rank(start),
                zrange_rank(stop),
                None,
                false,
                None,
                false,
            )
            .await
            .map_err(self.failed("ZRANGE"))?;
        self.faults.rank_read_answered();
        self.faults.forge_reply(&mut members, Value::as_string);
        Ok(members)
    }

    /// The members at ranks `start..=stop` and the key's `ZCARD`, in one
    /// pipeline: one page of the recount's ascending walk
    /// ([`page_by_rank_counted`]). The test seam's `StepError` fault may fail
    /// it transiently before it is sent, and its `DropMember` or
    /// `RepeatMember` fault may drop or repeat a member of the page (never in
    /// a release build).
    pub async fn page_with_card(
        &self,
        start: u64,
        stop: u64,
    ) -> Result<(Vec<Value>, u64), WalkError> {
        if let Some(injected) = self.faults.rank_read_error("ZRANGE") {
            return Err(self.failed("ZRANGE")(injected));
        }
        let pipe = self.client.pipeline();
        let options = self.options();
        let _: () = pipe
            .with_options(&options)
            .zrange(
                &*self.key,
                zrange_rank(start),
                zrange_rank(stop),
                None,
                false,
                None,
                false,
            )
            .await
            .map_err(self.failed("ZRANGE"))?;
        let _: () = pipe
            .with_options(&options)
            .zcard(&*self.key)
            .await
            .map_err(self.failed("ZCARD"))?;
        let replies: Vec<Value> = pipe.all().await.map_err(self.failed("ZRANGE"))?;
        let protocol = |what: String| WalkError::Step {
            walk: "ZRANGE",
            target: Arc::clone(&self.key),
            source: fred::error::Error::new(fred::error::ErrorKind::Protocol, what),
        };
        let answered = replies.len();
        let Ok([members, card]) = <[Value; 2]>::try_from(replies) else {
            return Err(protocol(format!(
                "a ZRANGE page of {} answered {answered} replies to its 2 commands",
                self.key
            )));
        };
        let Value::Array(mut members) = members else {
            return Err(protocol(format!(
                "a ZRANGE page of {} answered no member list",
                self.key
            )));
        };
        let card: u64 = card.convert().map_err(self.failed("ZCARD"))?;
        self.faults.rank_read_answered();
        self.faults.forge_reply(&mut members, Value::as_string);
        Ok((members, card))
    }
}

/// The per-page side of [`page_by_rank_counted`]: what a DR Job does with
/// each page of a sorted set, and its error for a failed or incomplete walk.
pub(crate) trait RankPageTaker {
    /// The caller's error.
    type Error;

    /// The caller's error for the walk's own: a failed read, or a count
    /// check that did not hold.
    fn walk_error(&self, e: WalkError) -> Self::Error;

    /// Take one page's members. The pager reads the next page only once this
    /// returns, so each page is dropped before the next.
    async fn take_page(&mut self, members: Vec<Value>) -> Result<(), Self::Error>;
}

/// The recount's read of a whole sorted set (module docs, *The DR Jobs'
/// walks*): ascending `ZRANGE i i+n-1` pages of `page_len`, each with the
/// key's `ZCARD` in its pipeline ([`RankWalk::page_with_card`]), each handed
/// to `taker` before the next is read, until a page answers fewer than
/// `page_len` members. Complete only when the pages returned exactly the
/// `ZCARD` read with the first page, and the `ZCARD` read with the last page
/// equals it ([`check_count`]); otherwise the taker's error for a
/// [`WalkIncomplete::CountMismatch`]. Holds one page at a time. The first
/// failure ends it.
pub(crate) async fn page_by_rank_counted<T: RankPageTaker>(
    walk: &RankWalk<'_>,
    page_len: u64,
    taker: &mut T,
) -> Result<(), T::Error> {
    let page_len = page_len.max(1);
    let mut start: u64 = 0;
    let mut returned: u64 = 0;
    let mut at_start: Option<u64> = None;
    loop {
        let stop = start.saturating_add(page_len - 1);
        let (members, card) = walk
            .page_with_card(start, stop)
            .await
            .map_err(|e| taker.walk_error(e))?;
        let first = *at_start.get_or_insert(card);
        let read = members.len() as u64;
        returned = returned.saturating_add(read);
        taker.take_page(members).await?;
        if read < page_len {
            return check_count("ZRANGE", walk.key(), "ZCARD", first, returned, card, first)
                .map_err(|e| taker.walk_error(e));
        }
        start = start.saturating_add(page_len);
    }
}

/// A DR Job's counted cursor walk of one hash (`HSCAN`, counted by `HLEN`)
/// or set (`SSCAN`, counted by `SCARD`) (module docs, *The DR Jobs' walks*):
/// each step and the key's cardinality in one pipeline, keyed, routed by the
/// key's slot with fred's default attempts (a `-MOVED` is followed), on the
/// client the walk holds, with no witness. [`walk_counted`] drives it.
pub struct CountedScan<'c> {
    client: &'c Client,
    command: &'static str,
    card: &'static str,
    key: Arc<str>,
    faults: WalkFaults,
}

/// One step of a [`CountedScan`].
#[derive(Debug)]
pub struct CountedStep {
    /// The next cursor (`"0"` once the walk is complete).
    pub next: String,
    /// The step's members: a set's members, or a hash's fields (their values
    /// dropped).
    pub members: Vec<Value>,
    /// The key's cardinality, read in the step's pipeline.
    pub card: u64,
}

impl<'c> CountedScan<'c> {
    /// A counted `HSCAN` walk of the hash `key` on `client`.
    pub fn hash(client: &'c Client, key: &str) -> Self {
        Self::new(client, "HSCAN", "HLEN", key)
    }

    /// A counted `SSCAN` walk of the set `key` on `client`.
    pub fn set(client: &'c Client, key: &str) -> Self {
        Self::new(client, "SSCAN", "SCARD", key)
    }

    fn new(client: &'c Client, command: &'static str, card: &'static str, key: &str) -> Self {
        Self {
            client,
            command,
            card,
            key: Arc::from(key),
            faults: WalkFaults::for_target(key),
        }
    }

    /// The walked key.
    pub fn key(&self) -> &Arc<str> {
        &self.key
    }

    /// The walk's command: `HSCAN` or `SSCAN`.
    pub fn command(&self) -> &'static str {
        self.command
    }

    /// One step from `cursor`, asking for `count` elements, with the key's
    /// cardinality in the step's pipeline: `[HSCAN key cursor COUNT count;
    /// HLEN key]` or `[SSCAN …; SCARD key]`. An `HSCAN` step's fields are the
    /// first of each pair (`retirement::scan_step_firsts`). A failed command
    /// or a reply of another shape is a `Step` error naming the walk. The
    /// test seam's `DropMember` or `RepeatMember` fault may drop or repeat a
    /// member of the step (never in a release build).
    pub async fn step(&self, cursor: &str, count: u64) -> Result<CountedStep, WalkError> {
        let failed = |source| WalkError::Step {
            walk: self.command,
            target: Arc::clone(&self.key),
            source,
        };
        let key: &str = &self.key;
        let pipe = self.client.pipeline();
        let _: () = pipe
            .custom(
                CustomCommand::new_static(self.command, ClusterHash::FirstValue, false),
                vec![
                    Value::from(key),
                    Value::from(cursor),
                    Value::from("COUNT"),
                    Value::from(i64::try_from(count).unwrap_or(i64::MAX)),
                ],
            )
            .await
            .map_err(failed)?;
        let _: () = pipe
            .custom(
                CustomCommand::new_static(self.card, ClusterHash::FirstValue, false),
                vec![Value::from(key)],
            )
            .await
            .map_err(failed)?;
        let replies: Vec<Value> = pipe.all().await.map_err(failed)?;
        let answered = replies.len();
        let Ok([reply, card]) = <[Value; 2]>::try_from(replies) else {
            return Err(failed(fred::error::Error::new(
                fred::error::ErrorKind::Protocol,
                format!(
                    "a {} step of {key} answered {answered} replies to its 2 commands",
                    self.command
                ),
            )));
        };
        let (next, elements) = split_step_reply(self.command, key, reply).map_err(failed)?;
        let mut members = if self.command == "HSCAN" {
            crate::storage::retirement::scan_step_firsts(self.command, key, elements)
                .map_err(failed)?
        } else {
            elements
        };
        let card: u64 = card.convert().map_err(failed)?;
        self.faults.forge_reply(&mut members, Value::as_string);
        Ok(CountedStep {
            next,
            members,
            card,
        })
    }
}

/// The per-step side of [`walk_counted`]: what a DR Job does with each
/// step's members, which it de-duplicates exactly, and its error for a
/// failed or incomplete walk.
pub(crate) trait CountedStepTaker {
    /// The caller's error.
    type Error;

    /// The caller's error for the walk's own: a failed step, or a count
    /// check that did not hold.
    fn walk_error(&self, e: WalkError) -> Self::Error;

    /// Take one step's members, a member another step already returned
    /// counted once.
    async fn take_step(&mut self, members: Vec<Value>) -> Result<(), Self::Error>;

    /// The distinct members taken so far.
    fn distinct(&self) -> u64;
}

/// A DR Job's counted cursor walk of one hash or set (module docs, *The DR
/// Jobs' walks*): `walk`'s steps of `count` from cursor `"0"` until the
/// server answers `"0"`, each handed to `taker` before the next is issued.
/// Complete only when the distinct members taken equal the cardinality read
/// with the first step and with the last ([`check_count`]); otherwise the
/// taker's error for a [`WalkIncomplete::CountMismatch`]. A full cursor walk
/// returns every member present throughout at least once on one process, and
/// a member a step repeats is counted once; on a quiet store the count check
/// shows a walk complete whichever process answered each step. The first
/// failure ends it; it never restarts from `"0"`.
pub(crate) async fn walk_counted<T: CountedStepTaker>(
    walk: &CountedScan<'_>,
    count: u64,
    taker: &mut T,
) -> Result<(), T::Error> {
    let mut cursor = "0".to_owned();
    let mut at_start: Option<u64> = None;
    loop {
        let step = walk
            .step(&cursor, count)
            .await
            .map_err(|e| taker.walk_error(e))?;
        let first = *at_start.get_or_insert(step.card);
        taker.take_step(step.members).await?;
        if step.next == "0" {
            return check_count(
                walk.command,
                walk.key(),
                walk.card,
                first,
                taker.distinct(),
                step.card,
                first,
            )
            .map_err(|e| taker.walk_error(e));
        }
        cursor = step.next;
    }
}

/// One range of a `CLUSTER SLOTS` map and the primary that serves it. Also
/// built by `storage::store_memory`'s own `CLUSTER SLOTS` parse, which
/// spells `primary` the same way, for [`table_partition_matches`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlotOwner {
    pub(crate) start: u16,
    pub(crate) end: u16,
    /// `host:port` as the reply names it (an empty host: the node that
    /// answered).
    pub(crate) primary: String,
}

/// A `CLUSTER SLOTS` reply as the ranges and their primaries, sorted by
/// slot; `None` for an empty reply or any malformed entry (the same reading
/// as `storage::store_memory::parse_cluster_slots_reply`, keeping each
/// range's primary).
fn parse_slot_owners(reply: &Value) -> Option<Vec<SlotOwner>> {
    let slot = |value: &Value| match value {
        Value::Integer(n) => u16::try_from(*n).ok().filter(|s| *s <= 16_383),
        _ => None,
    };
    let Value::Array(entries) = reply else {
        return None;
    };
    if entries.is_empty() {
        return None;
    }
    let mut owners = Vec::with_capacity(entries.len());
    for entry in entries {
        let Value::Array(fields) = entry else {
            return None;
        };
        let start = slot(fields.first()?)?;
        let end = slot(fields.get(1)?)?;
        if start > end {
            return None;
        }
        let Value::Array(primary) = fields.get(2)? else {
            return None;
        };
        let host = match primary.first()? {
            Value::String(host) => host.to_string(),
            Value::Bytes(host) => std::str::from_utf8(host).ok()?.to_owned(),
            Value::Null => String::new(),
            _ => return None,
        };
        let Value::Integer(port) = primary.get(1)? else {
            return None;
        };
        owners.push(SlotOwner {
            start,
            end,
            primary: format!("{host}:{port}"),
        });
    }
    owners.sort_by_key(|owner| owner.start);
    Some(owners)
}

/// A slot map as an error line names it: `start-end@primary` per range.
pub(crate) fn render_map(map: &[SlotOwner]) -> String {
    map.iter()
        .map(|o| format!("{}-{}@{}", o.start, o.end, o.primary))
        .collect::<Vec<_>>()
        .join(",")
}

/// One primary of a slot map: its name in the map, the slot its walk is
/// resolved by (the first slot of its lowest range), and every slot the map
/// gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MapPrimary {
    primary: String,
    pin_slot: u16,
    slots: SlotSet,
}

/// The primaries of a slot map, in slot order, each with its pin slot and
/// its whole slot set.
fn primaries_of(map: &[SlotOwner]) -> Vec<MapPrimary> {
    let mut primaries: Vec<MapPrimary> = Vec::new();
    for owner in map {
        let range = (owner.start, owner.end);
        match primaries.iter_mut().find(|p| p.primary == owner.primary) {
            Some(p) => {
                p.slots = SlotSet::from_ranges(p.slots.ranges().iter().copied().chain([range]));
            }
            None => primaries.push(MapPrimary {
                primary: owner.primary.clone(),
                pin_slot: owner.start,
                slots: SlotSet::from_ranges([range]),
            }),
        }
    }
    primaries
}

/// Whether `routing`, a client's cached routing table, divides the hash slots
/// among its nodes exactly as `map`, the walk's `CLUSTER SLOTS` map, divides
/// them among its primaries: the table's ranges grouped by node and the map's
/// grouped by primary, each group as one [`SlotSet`], compared as sorted lists
/// of sets. Host and port spellings never enter the comparison, so a table
/// that names a node by another host, address or TLS name than the map still
/// matches. No table, or one with no ranges, differs.
///
/// **What equality proves.** A coverage walk pins each primary to the node
/// the table lists for one of that primary's slots. With the two groupings
/// equal, that node's group in the table is exactly that primary's slot set,
/// and two primaries, whose sets differ, get two different nodes: the pins
/// are every node the table names, once each, whatever the spellings. A pin
/// whose own `CLUSTER NODES` could not be read is then the table's node for
/// exactly its primary's slots.
///
/// **What it cannot see.** A change of primary that keeps every slot set (a
/// failover the table missed, whose demoted node still answers a keyless
/// `SCAN` from its replica copy): the table groups the same slots under the
/// old node, so the groupings still match. A coverage walk with an unproven
/// pin therefore re-syncs its table at its end even when the groupings
/// match, so such a change lasts at most one pass per maintenance-pool
/// member (each keeps its own table) when that re-sync succeeds. Pure.
///
/// **Second user.** The store-memory read (`storage::store_memory`) compares
/// its own pass's `CLUSTER SLOTS` map with its member's table the same way
/// before any `INFO memory`: equality proves the table's distinct primaries
/// are exactly the map's, one each, so its `INFO` targets reach every
/// primary; a table that differs is re-synced once ([`refresh_routing`]),
/// and a pass whose table still differs, or whose re-sync did not complete,
/// reports its primaries unconfirmed. The blind spot above holds there too:
/// such a pass reads the demoted node.
pub(crate) fn table_partition_matches(map: &[SlotOwner], routing: Option<&ClusterRouting>) -> bool {
    let Some(routing) = routing else {
        return false;
    };
    if routing.slots().is_empty() {
        return false;
    }
    let mut from_map: Vec<SlotSet> = primaries_of(map).into_iter().map(|p| p.slots).collect();
    // The table's ranges grouped by node, as fred keys a node (`Server`'s
    // own equality: host and port).
    let mut by_node: Vec<(&Server, Vec<(u16, u16)>)> = Vec::new();
    for range in routing.slots() {
        let span = (range.start, range.end);
        match by_node.iter_mut().find(|(node, _)| **node == range.primary) {
            Some((_, spans)) => spans.push(span),
            None => by_node.push((&range.primary, vec![span])),
        }
    }
    let mut from_table: Vec<SlotSet> = by_node
        .into_iter()
        .map(|(_, spans)| SlotSet::from_ranges(spans))
        .collect();
    from_map.sort_unstable();
    from_table.sort_unstable();
    from_map == from_table
}

/// A routing table as an error line names it: `start-end@host:port` per
/// range, in the table's order, or `no routing table` when there is none or
/// it holds no range.
pub(crate) fn render_table(routing: Option<&ClusterRouting>) -> String {
    match routing {
        Some(routing) if !routing.slots().is_empty() => routing
            .slots()
            .iter()
            .map(|range| format!("{}-{}@{}", range.start, range.end, range.primary))
            .collect::<Vec<_>>()
            .join(","),
        _ => "no routing table".to_owned(),
    }
}

/// What a `CLUSTER NODES` reply's *myself* line — the line of the node that
/// answered — says about that node ([`parse_myself_line`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MyselfLine {
    /// The slots it serves: the union of its `N` and `A-B` slot fields
    /// (empty for a replica, which lists none).
    pub owned: SlotSet,
    /// Its open-slot markers (`[slot->-…]` migrating, `[slot-<-…]`
    /// importing), as printed.
    pub open: Vec<String>,
}

/// The highest hash slot.
const MAX_SLOT: u16 = 16_383;

/// A slot number as `CLUSTER NODES` prints it: decimal digits only, at most
/// [`MAX_SLOT`].
fn parse_slot(text: &str) -> Option<u16> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse::<u16>().ok().filter(|slot| *slot <= MAX_SLOT)
}

/// Read a `CLUSTER NODES` reply's *myself* line: the line whose third field's
/// comma-separated flags include `myself`. Its fields from index 8 on are its
/// slots (Valkey 9.0.3 `cluster_legacy.c` 6561–6572, 6625–6667; the same
/// eight fixed fields on AMR): one that starts with `[` is an open-slot
/// marker, and every other must be `N` or `A-B` with `0 ≤ A ≤ B ≤ 16383`.
/// No myself line, a myself line with fewer than its 8 fixed fields, or a
/// slot field outside that grammar, is what was seen instead (an unusable
/// myself line: exit 3 in the DR Jobs). A replica's line, all 8 fields and
/// no slot, serves the empty set. Pure.
pub fn parse_myself_line(nodes: &str) -> Result<MyselfLine, MyselfLineSeen> {
    /// The fixed fields before the slots: id, address, flags, primary,
    /// ping-sent, pong-received, config epoch, link state.
    const FIXED_FIELDS: usize = 8;
    let line = nodes
        .lines()
        .find(|line| {
            line.split_whitespace()
                .nth(2)
                .is_some_and(|flags| flags.split(',').any(|flag| flag == "myself"))
        })
        .ok_or(MyselfLineSeen::NoLine)?;
    if line.split_whitespace().count() < FIXED_FIELDS {
        return Err(MyselfLineSeen::ShortLine(line.trim_end().to_owned()));
    }
    let mut ranges = Vec::new();
    let mut open = Vec::new();
    for field in line.split_whitespace().skip(8) {
        if field.starts_with('[') {
            open.push(field.to_owned());
            continue;
        }
        let range = match field.split_once('-') {
            Some((start, end)) => parse_slot(start).zip(parse_slot(end)),
            None => parse_slot(field).map(|slot| (slot, slot)),
        };
        match range {
            Some((start, end)) if start <= end => ranges.push((start, end)),
            _ => return Err(MyselfLineSeen::BadToken(field.to_owned())),
        }
    }
    Ok(MyselfLine {
        owned: SlotSet::from_ranges(ranges),
        open,
    })
}

/// The ownership decision: a pinned node serves a primary exactly when its
/// myself line's slots are the slots the walk's map gives that primary —
/// whole sets, never one slot, so a node whose view lags the map, two
/// primaries resolved to one node, or a replica (no slots) all fail. Pure.
pub fn serves_exactly(expected: &SlotSet, owned: &SlotSet) -> bool {
    expected == owned
}

/// What a coverage walk makes of one primary's pinned node, from its
/// `CLUSTER NODES` reply ([`coverage_verdict`]; module docs, *Coverage
/// walks*).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoverageVerdict {
    /// The node serves exactly the primary's slots, none of them open: the
    /// primary is walked, nothing recorded.
    Owner,
    /// The node serves exactly the primary's slots, and its myself line
    /// shows these open-slot markers (as printed, space-separated): the
    /// primary is walked, the walk recorded unproven
    /// ([`ServingChange::OpenSlot`]).
    OpenSlot {
        /// The open-slot markers.
        slots: String,
    },
    /// The reply gave no usable myself line: the primary is walked on this
    /// node, the walk recorded unproven ([`WitnessMissing`]). Whether the node
    /// serves the primary is unknown, so the walk compares its map with the
    /// client's routing table (`table_partition_matches`): a table that
    /// differs is out of date, loses the walk's coverage and is re-synced
    /// ([`ServingChange::StaleTable`]; module docs, *The table check*).
    Unusable(MyselfLineSeen),
    /// The `CLUSTER NODES` read failed: the primary is walked on this node,
    /// the walk recorded unproven (the read's [`WalkError::Read`]). As with
    /// [`Self::Unusable`], the node may not serve the primary, so the walk
    /// compares its map with the routing table.
    ReadFailed,
    /// The node serves other slots (a replica serves none): the primary is
    /// not walked on it, the walk is recorded unproven
    /// ([`ServingChange::NotOwner`]) and the client's routing table is
    /// re-synced, once per walk.
    NotOwner {
        /// The slots the node's myself line lists.
        served: SlotSet,
    },
}

impl CoverageVerdict {
    /// Whether the coverage walk walks the primary on this node: every
    /// verdict but [`Self::NotOwner`].
    pub fn walks(&self) -> bool {
        !matches!(self, Self::NotOwner { .. })
    }

    /// Whether the verdict proves the node serves the primary: only
    /// [`Self::Owner`]. Any other is recorded.
    pub fn proven(&self) -> bool {
        matches!(self, Self::Owner)
    }

    /// Whether the verdict leaves the walk's pin ownership unproven by the
    /// node's own myself line, so the walk must check its routing table
    /// against its map (module docs, *The table check*): an unusable myself
    /// line or a failed read. An owner, an open slot (its myself line listed
    /// exactly the primary's slots) and a `NotOwner` (proven wrong, and
    /// re-synced on its own) do not. Pure.
    fn leaves_ownership_unproven(&self) -> bool {
        match self {
            Self::Unusable(_) | Self::ReadFailed => true,
            Self::Owner | Self::OpenSlot { .. } | Self::NotOwner { .. } => false,
        }
    }
}

/// The coverage walk's verdict on one primary's pinned node: `nodes` is its
/// `CLUSTER NODES` reply, `None` when the read failed; `expected` the slots
/// the walk's map gives the primary. A usable myself line that lists other
/// slots is [`CoverageVerdict::NotOwner`] even when it also shows an open
/// slot: a node that serves other slots is not the primary's, and walking it
/// would count the primary walked when it was not. Pure.
pub fn coverage_verdict(expected: &SlotSet, nodes: Option<&str>) -> CoverageVerdict {
    let Some(nodes) = nodes else {
        return CoverageVerdict::ReadFailed;
    };
    match parse_myself_line(nodes) {
        Err(seen) => CoverageVerdict::Unusable(seen),
        Ok(myself) if !serves_exactly(expected, &myself.owned) => CoverageVerdict::NotOwner {
            served: myself.owned,
        },
        Ok(myself) if !myself.open.is_empty() => CoverageVerdict::OpenSlot {
            slots: myself.open.join(" "),
        },
        Ok(_) => CoverageVerdict::Owner,
    }
}

/// Re-sync `client`'s routing table (`CLUSTER SLOTS`, then the connections
/// it implies: the sync fred runs on a `MOVED`), bounded by the client's
/// command timeout (unbounded when that is `0`), as
/// `ValkeyClient::sync_txn_checkout_routing` does. Called on a result that
/// proves this client's table wrong — a [`ServingChange::NotOwner`], or a
/// coverage walk's [`ServingChange::StaleTable`] — and at the end of a
/// coverage walk with an unproven pin whose table matched, where the
/// comparison cannot rule out a change of primary that keeps every slot set
/// (a coverage walk: once, at the first of these). Also called by the
/// store-memory read (`storage::store_memory`), at most once per pass, when
/// its member's table divides the slots otherwise than the map that pass
/// just read ([`table_partition_matches`]). A sync that times out keeps
/// running in the member's router.
pub(crate) async fn refresh_routing(client: &Client) -> RoutingRefresh {
    let timeout = client.perf_config().default_command_timeout;
    let synced = if timeout.is_zero() {
        Ok(client.sync_cluster().await)
    } else {
        tokio::time::timeout(timeout, client.sync_cluster()).await
    };
    match synced {
        Ok(Ok(())) => RoutingRefresh::Refreshed,
        Ok(Err(e)) => RoutingRefresh::Failed(e.to_string()),
        Err(_elapsed) => RoutingRefresh::TimedOut {
            waited_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
        },
    }
}

/// One primary of a keyspace walk: its pin, its start witness, its cursor.
struct PrimaryWalk {
    /// The primary as the walk's map names it, with its slots (`None`
    /// standalone).
    map: Option<MapPrimary>,
    /// Its node route on a cluster, [`Route::None`] standalone.
    route: Route,
    /// Its start witness; `None` in a coverage walk or a DR Job's walk,
    /// which read none.
    start: Option<Witness>,
    cursor: String,
    done: bool,
}

/// The guard level of a keyspace walk, chosen by its constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyspaceGuard {
    /// [`KeyspaceWalk::start`]: the serving-process witness and the
    /// fingerprint; the first check that does not hold ends the walk (module
    /// docs, *Keyspace walks*).
    Serving,
    /// [`KeyspaceWalk::start_job`]: a DR Job's walk, one of an agreeing pair
    /// (module docs, *The DR Jobs' walks*): the fingerprint's map and myself
    /// lines, no witness; the first check that does not hold ends the walk,
    /// as in [`Self::Serving`].
    Job,
    /// [`KeyspaceWalk::start_coverage`]: node pins and the ownership verdict,
    /// no witness; a check that does not hold is recorded and answered after
    /// the last key (module docs, *Coverage walks*).
    Coverage,
}

impl KeyspaceGuard {
    /// The `COUNT` of the walk's `SCAN` step.
    fn scan_count(self) -> u64 {
        match self {
            Self::Serving | Self::Coverage => KEYSPACE_SCAN_COUNT,
            Self::Job => DR_KEYSPACE_SCAN_COUNT,
        }
    }
}

/// A guarded walk of every key matching a pattern, on every primary (module
/// docs, *Keyspace walks*, *The DR Jobs' walks* and *Coverage walks*), on the
/// client it holds (a handle on the caller's client: the same pool member,
/// the same routing table). Each [`Self::next_page`] answers one step's keys,
/// in a guarded walk already checked against its primary's start witness;
/// the walk is complete only once `next_page` has answered `None`, after the
/// end checks. A guarded walk's keys, and a DR Job's walk's, are usable only
/// then (a Job's only once a second walk agreed, [`collect_keyspace_agreed`]);
/// a coverage walk's are used as they come.
pub struct KeyspaceWalk {
    client: Client,
    pattern: Arc<str>,
    guard: KeyspaceGuard,
    /// The start slot map; `None` standalone.
    map: Option<Vec<SlotOwner>>,
    /// The primaries walked: every primary of the map in a guarded walk;
    /// in a coverage walk, those not left unwalked by a `NotOwner`.
    primaries: Vec<PrimaryWalk>,
    current: usize,
    finished: bool,
    /// Coverage walks: the reason answered once after the last key — the
    /// first coverage-losing reason (`NotOwner`, `SlotMap`, `StaleTable`) if
    /// one was recorded, otherwise the first proof-only reason
    /// ([`Self::record`]); a walk counted as incomplete therefore always
    /// answers why.
    unproven: Option<WalkError>,
    /// Coverage walks: a primary of the start map was not walked on its
    /// owner (a `NotOwner`, at the start or the end), the map changed
    /// (`SlotMap`), or a walk with an unproven pin found the routing table
    /// out of date (`StaleTable`).
    coverage_lost: bool,
    /// Coverage walks: a pin's ownership could not be proven, at the start or
    /// the end — an unusable myself line or a failed `CLUSTER NODES` read
    /// (never an open slot, whose myself line proved the node's slots). Only
    /// such a walk compares its map with the routing table
    /// ([`Self::check_table`]).
    ownership_unproven: bool,
    /// Coverage walks: the client's routing table was re-synced (on the
    /// walk's first `NotOwner` or `StaleTable`, or at its end after an
    /// unproven pin whose table matched); never twice in one walk.
    resynced: bool,
    /// Every walked primary answered cursor `0` and the end checks ran.
    reached_end: bool,
    faults: WalkFaults,
}

impl KeyspaceWalk {
    /// Start a guarded walk of `pattern` on `client`: read the start
    /// fingerprint. On a cluster, each primary of the `CLUSTER SLOTS` map is
    /// pinned to the node one snapshot of the client's own routing table
    /// lists for its pin slot, and that node's witness and myself line are
    /// read on the pin. An open slot, a pin that does not reach the primary's
    /// slots ([`ServingChange::NotOwner`], after a routing re-sync), a missing
    /// witness or an unusable myself line fails it here, before any step.
    pub async fn start(client: &Client, pattern: &str) -> Result<Self, WalkError> {
        Self::begin(client, pattern, KeyspaceGuard::Serving).await
    }

    /// Start a DR Job's walk of `pattern` on `client` (module docs, *The DR
    /// Jobs' walks*): one walk of an agreeing pair
    /// ([`collect_keyspace_agreed`]). It is [`Self::start`]'s walk with no
    /// serving-process witness: on a cluster, the same `CLUSTER SLOTS` map
    /// and node pins, and each pin's myself line read on the pin; an open
    /// slot, a pin that does not reach the primary's slots
    /// ([`ServingChange::NotOwner`], after a routing re-sync) or an unusable
    /// myself line fails it here, before any step. Its steps send `COUNT`
    /// [`DR_KEYSPACE_SCAN_COUNT`], each pinned with one attempt and resent at
    /// the same cursor after a transient failure, up to
    /// [`WALK_STEP_ATTEMPTS`] attempts, as a coverage walk's are; its end
    /// fingerprint is the map and the myself lines again. Standalone, nothing
    /// is read before the first step or after the last.
    pub async fn start_job(client: &Client, pattern: &str) -> Result<Self, WalkError> {
        Self::begin(client, pattern, KeyspaceGuard::Job).await
    }

    /// Start a coverage walk of `pattern` on `client` (module docs,
    /// *Coverage walks*): on a cluster, a fresh `CLUSTER SLOTS` map, each
    /// primary pinned as [`Self::start`] pins it, and each pin's ownership
    /// verdict ([`coverage_verdict`]), read with no witness, then, when a
    /// pin's ownership is unproven, the map against the table snapshot the
    /// pins came from (module docs, *The table check*). Fails here only on a
    /// store fault: the map read, or a refused node pin. A pin that does not
    /// reach its primary's slots leaves that primary unwalked and re-syncs
    /// the routing table, as an out-of-date table found by the comparison
    /// does; these and every other check that does not hold are answered by
    /// [`Self::next_page`] after the last key. Standalone, nothing is read
    /// before the first step.
    pub async fn start_coverage(client: &Client, pattern: &str) -> Result<Self, WalkError> {
        Self::begin(client, pattern, KeyspaceGuard::Coverage).await
    }

    async fn begin(
        client: &Client,
        pattern: &str,
        guard: KeyspaceGuard,
    ) -> Result<Self, WalkError> {
        let pattern: Arc<str> = Arc::from(pattern);
        let faults = WalkFaults::for_target(&pattern);
        let mut walk = Self {
            client: client.clone(),
            pattern,
            guard,
            map: None,
            primaries: Vec::new(),
            current: 0,
            finished: false,
            unproven: None,
            coverage_lost: false,
            ownership_unproven: false,
            resynced: false,
            reached_end: false,
            faults,
        };
        if client.is_clustered() {
            let map = walk.read_map().await?;
            // One snapshot of the table every pin is resolved from. The
            // `Server` is cloned whole, never built from the reply's
            // `host:port`: fred keys its connections by its own table's
            // spelling, and the clone keeps fred's TLS server name.
            let routing = client.cached_cluster_state();
            for (index, primary) in primaries_of(&map).into_iter().enumerate() {
                let resolved = routing
                    .as_ref()
                    .and_then(|routing| routing.get_server(primary.pin_slot).cloned());
                // The test seam's `UnresolvedPin` resolves the primary to no
                // node; its `PinNode` pins it to a given one (never in a
                // release build).
                let pinned = if walk.faults.unresolved_pin(index) {
                    None
                } else {
                    walk.faults.pin_node(index).or(resolved)
                };
                let (route, start) = match (guard, pinned) {
                    (KeyspaceGuard::Serving | KeyspaceGuard::Job, None) => {
                        return Err(walk.not_owner(&primary, None, None, WalkEnd::Start).await);
                    }
                    (KeyspaceGuard::Coverage, None) => {
                        walk.coverage_not_owner(&primary, None, None, WalkEnd::Start)
                            .await;
                        continue;
                    }
                    (KeyspaceGuard::Serving, Some(server)) => {
                        let route = Route::Node(server);
                        let start = walk.read_primary(&primary, &route, WalkEnd::Start).await?;
                        (route, Some(start))
                    }
                    (KeyspaceGuard::Job, Some(server)) => {
                        let route = Route::Node(server);
                        walk.check_owner(&primary, &route, WalkEnd::Start).await?;
                        (route, None)
                    }
                    (KeyspaceGuard::Coverage, Some(server)) => {
                        let route = Route::Node(server);
                        if !walk
                            .check_coverage(&primary, &route, WalkEnd::Start)
                            .await?
                        {
                            continue;
                        }
                        (route, None)
                    }
                };
                walk.primaries.push(PrimaryWalk {
                    map: Some(primary),
                    route,
                    start,
                    cursor: "0".to_owned(),
                    done: false,
                });
            }
            // After every verdict, so a re-sync here cannot make
            // `ensure_routable` refuse a `CLUSTER NODES` still to be read;
            // against the snapshot the pins were resolved from.
            if guard == KeyspaceGuard::Coverage {
                walk.check_table(&map, routing.as_ref(), WalkEnd::Start)
                    .await;
            }
            walk.map = Some(map);
        } else {
            let route = Route::None;
            let start = match guard {
                KeyspaceGuard::Serving => {
                    Some(read_witness(client, &route, &walk.faults, "SCAN", &walk.pattern).await?)
                }
                KeyspaceGuard::Job | KeyspaceGuard::Coverage => None,
            };
            walk.primaries.push(PrimaryWalk {
                map: None,
                route,
                start,
                cursor: "0".to_owned(),
                done: false,
            });
        }
        Ok(walk)
    }

    /// Whether the walk reached its end — every walked primary at cursor `0`
    /// and its end checks run — having walked every primary of its start map
    /// on its pin, with the map unchanged and no pin proven wrong: no
    /// `NotOwner` was found (at the start or the end), no `SlotMap` was
    /// recorded, no `StaleTable` was found, no store fault ended it. A
    /// coverage walk that recorded only an open slot, an unusable myself line
    /// or a failed `CLUSTER NODES` read still answers `true`; for the last two
    /// the pin's ownership is unproven by its node, so it answers `true` only
    /// because the routing table matched the map at each end it was compared:
    /// when a start verdict was unproven, the snapshot the pins came from
    /// against the start map and then the client's table against the end
    /// map; when only an end verdict was, the end comparison alone
    /// (`table_partition_matches`; module docs, *The table check*). For a
    /// coverage walk `true` is not proof that no key was missed: an open slot
    /// may skip keys, the table comparison cannot see a change of primary
    /// that keeps every slot set (such a walk re-syncs its table at its end,
    /// so that miss lasts at most one pass per maintenance-pool member when
    /// the re-sync succeeds), and a change of serving process within the walk
    /// goes undetected (module docs, *Coverage walks*). A guarded walk and a
    /// DR Job's walk answer `true` exactly when they completed. `false`
    /// before the end.
    pub fn walked_every_primary(&self) -> bool {
        self.reached_end && !self.coverage_lost
    }

    /// Test seam: whether a coverage walk has re-synced the client's routing
    /// table: on a `NotOwner` or a `StaleTable`, or at its end after an
    /// unproven pin whose table matched. Always `false` for a guarded walk
    /// or a DR Job's walk, which fails on its `NotOwner` re-sync instead of
    /// recording it.
    /// Compiled out of release builds.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn resynced_for_test(&self) -> bool {
        self.resynced
    }

    /// The next step's keys, or `None` once every walked primary answered
    /// cursor `0` and the end checks held. In a guarded walk, each step is
    /// sent in one pipeline with its witness on the same node route
    /// (`[SCAN; INFO server]`), and its keys come back only after that
    /// witness matched its primary's start witness; the first error ends the
    /// walk. A DR Job's walk sends each step alone and ends on its first
    /// error the same way. In a
    /// coverage walk, a step's keys come back as it lands; after the last
    /// one and the end checks, one recorded reason is answered once (then
    /// `None`) — the first coverage-losing reason (`NotOwner`, `SlotMap`,
    /// `StaleTable`) if one was recorded, otherwise the first proof-only
    /// reason, so a walk counted as incomplete always answers why — and a
    /// store fault ends the walk at once.
    pub async fn next_page(&mut self) -> Result<Option<Vec<Key>>, WalkError> {
        loop {
            if self.finished {
                return Ok(None);
            }
            let Some(primary) = self.primaries.get(self.current) else {
                self.check_end().await?;
                self.finished = true;
                self.reached_end = true;
                return match self.unproven.take() {
                    Some(reason) => Err(reason),
                    None => Ok(None),
                };
            };
            if primary.done {
                self.current += 1;
                continue;
            }
            let (route, cursor) = (&primary.route, primary.cursor.as_str());
            let start = primary.start.as_ref();
            let (client, pattern) = (&self.client, &self.pattern);
            let glob: &str = pattern;
            let count = i64::try_from(self.guard.scan_count()).unwrap_or(i64::MAX);
            // A guarded walk's step carries its witness in its pipeline,
            // `[SCAN; INFO server]`; a coverage walk's and a DR Job's walk's
            // is the step alone.
            let mode = match start {
                Some(start) => WitnessMode::After(start),
                None => WitnessMode::None,
            };
            let landed = send_guarded_step(
                client,
                route,
                &self.faults,
                "SCAN",
                pattern,
                mode,
                WALK_STEP_ATTEMPTS,
                move |pipe| async move {
                    pipe.with_options(&route.options(true))
                        .custom::<Value, Value>(
                            CustomCommand::new_static("SCAN", ClusterHash::FirstKey, false),
                            vec![
                                Value::from(cursor),
                                Value::from("MATCH"),
                                Value::from(glob),
                                Value::from("COUNT"),
                                Value::from(count),
                            ],
                        )
                        .await
                },
            )
            .await?;
            let step_err = |source| WalkError::Step {
                walk: "SCAN",
                target: Arc::clone(pattern),
                source,
            };
            let mut keys = landed
                .elements
                .into_iter()
                .map(Key::try_from)
                .collect::<Result<Vec<Key>, _>>()
                .map_err(step_err)?;
            // The test seam's `DropMember` or `RepeatMember` fault may forge
            // a skip or a repeat in a DR Job's walk (never in a release
            // build).
            if self.guard == KeyspaceGuard::Job {
                self.faults
                    .forge_reply(&mut keys, |key| key.as_str().map(str::to_owned));
            }
            // A guarded walk's step counts only once the witness that rode
            // with it matches its primary's start; a coverage walk and a DR
            // Job's walk have none.
            // A guarded step's pipeline always carries its witness
            // ([`read_step_pipeline`] answers it or fails): fail closed all
            // the same.
            match (start, &landed.witness) {
                (Some(start), Some(now)) => {
                    if let Some(change) = changed_between("SCAN", pattern, start, now) {
                        return Err(WalkError::changed(change));
                    }
                }
                (Some(_), None) => {
                    return Err(step_err(fred::error::Error::new(
                        fred::error::ErrorKind::Protocol,
                        "a guarded SCAN step's pipeline answered without its witness",
                    )))
                }
                (None, _) => {}
            }
            let next = landed.next;
            let primary = &mut self.primaries[self.current];
            primary.done = next == "0";
            primary.cursor = next;
            return Ok(Some(keys));
        }
    }

    /// Every key the walk returns, once it is complete.
    pub async fn collect(mut self) -> Result<Vec<Key>, WalkError> {
        let mut keys = Vec::new();
        while let Some(page) = self.next_page().await? {
            keys.extend(page);
        }
        Ok(keys)
    }

    /// A fresh `CLUSTER SLOTS` map, as an ordinary command on the walk's
    /// client (never fred's own cluster sync).
    async fn read_map(&self) -> Result<Vec<SlotOwner>, WalkError> {
        let reply: Value = self
            .client
            .cluster_slots()
            .await
            .map_err(|source| WalkError::Read {
                op: "CLUSTER SLOTS",
                source,
            })?;
        let map = parse_slot_owners(&reply).ok_or_else(|| WalkError::Read {
            op: "CLUSTER SLOTS",
            source: fred::error::Error::new(
                fred::error::ErrorKind::Protocol,
                "CLUSTER SLOTS answered no slot map, or one that does not parse",
            ),
        })?;
        Ok(self.faults.slot_map_reply(map))
    }

    /// One primary's part of a guarded walk's fingerprint (`at` its start or
    /// its end) on its node route: its witness, then the node's ownership
    /// ([`Self::check_owner`]).
    async fn read_primary(
        &self,
        primary: &MapPrimary,
        route: &Route,
        at: WalkEnd,
    ) -> Result<Witness, WalkError> {
        let witness =
            read_witness(&self.client, route, &self.faults, "SCAN", &self.pattern).await?;
        self.check_owner(primary, route, at).await?;
        Ok(witness)
    }

    /// One primary's ownership check (`at` the walk's start or its end) on
    /// its node route — a guarded walk's after its witness, a DR Job's walk's
    /// alone: the node's `CLUSTER NODES` myself line, read once the pin is
    /// checked routable, which must be usable (else a missing witness), show
    /// no open slot, and list exactly the primary's slots of the walk's map
    /// (else [`ServingChange::NotOwner`], after a routing re-sync).
    async fn check_owner(
        &self,
        primary: &MapPrimary,
        route: &Route,
        at: WalkEnd,
    ) -> Result<(), WalkError> {
        route.ensure_routable(&self.client, &self.faults, "SCAN", &self.pattern)?;
        let nodes = self.read_cluster_nodes(route).await?;
        let myself = parse_myself_line(&nodes).map_err(|seen| {
            WalkError::WitnessMissing(WitnessMissing {
                walk: "SCAN",
                target: Arc::clone(&self.pattern),
                node: route.node(&self.client),
                seen: WitnessSeen::MyselfLine(seen),
            })
        })?;
        if !myself.open.is_empty() {
            return Err(self.open_slot(primary, route, myself.open.join(" ")));
        }
        if !serves_exactly(&primary.slots, &myself.owned) {
            return Err(self
                .not_owner(primary, route.node(&self.client), Some(myself.owned), at)
                .await);
        }
        Ok(())
    }

    /// `CLUSTER NODES` on `route` (its pin checked by the caller), keeping
    /// fred's retries as every guard read does.
    async fn read_cluster_nodes(&self, route: &Route) -> Result<String, WalkError> {
        let nodes: String = self
            .client
            .with_options(&route.options(false))
            .cluster_nodes()
            .await
            .map_err(|source| WalkError::Read {
                op: "CLUSTER NODES",
                source,
            })?;
        Ok(self.faults.cluster_nodes_reply(nodes))
    }

    /// The [`ServingChange::OpenSlot`] of `primary`'s node on `route`.
    fn open_slot(&self, primary: &MapPrimary, route: &Route, slots: String) -> WalkError {
        WalkError::changed(ServingChange::OpenSlot {
            target: Arc::clone(&self.pattern),
            node: route
                .node(&self.client)
                .unwrap_or_else(|| primary.primary.clone()),
            slots,
        })
    }

    /// A coverage walk's check of `primary` on its node `route` (`at` its
    /// start or its end): the node's `CLUSTER NODES` and its
    /// [`coverage_verdict`], anything but an owner recorded
    /// ([`Self::record`]; a `NotOwner` through [`Self::coverage_not_owner`]),
    /// and a verdict that leaves ownership unproven
    /// ([`CoverageVerdict::leaves_ownership_unproven`]: an unusable myself
    /// line or a failed read) marking the walk for [`Self::check_table`].
    /// Whether the primary is walked on that node, or the store fault that
    /// ends the walk: a node pin the client's table no longer lists, refused
    /// before anything is sent.
    async fn check_coverage(
        &mut self,
        primary: &MapPrimary,
        route: &Route,
        at: WalkEnd,
    ) -> Result<bool, WalkError> {
        route.ensure_routable(&self.client, &self.faults, "SCAN", &self.pattern)?;
        let (nodes, failed) = match self.read_cluster_nodes(route).await {
            Ok(nodes) => (Some(nodes), None),
            Err(failed) => (None, Some(failed)),
        };
        let verdict = coverage_verdict(&primary.slots, nodes.as_deref());
        let walks = verdict.walks();
        if verdict.leaves_ownership_unproven() {
            self.ownership_unproven = true;
        }
        match verdict {
            CoverageVerdict::Owner => {}
            CoverageVerdict::OpenSlot { slots } => {
                let reason = self.open_slot(primary, route, slots);
                self.record(reason);
            }
            CoverageVerdict::Unusable(seen) => {
                let reason = WalkError::WitnessMissing(WitnessMissing {
                    walk: "SCAN",
                    target: Arc::clone(&self.pattern),
                    node: route.node(&self.client),
                    seen: WitnessSeen::MyselfLine(seen),
                });
                self.record(reason);
            }
            CoverageVerdict::ReadFailed => {
                if let Some(failed) = failed {
                    self.record(failed);
                }
            }
            CoverageVerdict::NotOwner { served } => {
                let node = route.node(&self.client);
                self.coverage_not_owner(primary, node, Some(served), at)
                    .await;
            }
        }
        Ok(walks)
    }

    /// A coverage walk's `NotOwner` for `primary`: the walk no longer covers
    /// every primary of its map, and when the walk has not yet re-synced (no
    /// earlier `NotOwner` or `StaleTable`) the client's routing table is
    /// re-synced and the change recorded ([`Self::not_owner`]). Otherwise it
    /// re-syncs nothing: the earlier one already did, and already holds the
    /// answered reason (a coverage-losing one, which [`Self::record`] never
    /// replaces), so this one would not be answered.
    async fn coverage_not_owner(
        &mut self,
        primary: &MapPrimary,
        node: Option<String>,
        served: Option<SlotSet>,
        at: WalkEnd,
    ) {
        self.coverage_lost = true;
        if self.resynced {
            return;
        }
        self.resynced = true;
        let reason = self.not_owner(primary, node, served, at).await;
        self.record(reason);
    }

    /// A coverage walk's routing-table check at one end (`at`), made only by
    /// a walk whose ownership is unproven (an unusable myself line or a failed
    /// `CLUSTER NODES` read so far; a walk whose every pin was proven by its
    /// own myself line has nothing for the table to prove). `map` is the
    /// walk's `CLUSTER SLOTS` map at that end and `routing` the table it is
    /// compared with ([`table_partition_matches`]): at the start, the snapshot
    /// the pins were resolved from; at the end, the client's current table.
    /// A table that differs is out of date: the walk loses its coverage and,
    /// when it has not yet re-synced (no earlier `NotOwner` or `StaleTable`),
    /// re-syncs the client's routing table ([`refresh_routing`]) and records
    /// [`ServingChange::StaleTable`]. Otherwise it re-syncs nothing: the
    /// earlier reason already did and holds the answered slot, as in
    /// [`Self::coverage_not_owner`]. A table that matches at the end is
    /// re-synced too, under the same once-per-walk rule, with nothing
    /// recorded and the coverage kept (whatever the sync's outcome; a failed
    /// one, which leaves the table as it was, or a timed-out one, which keeps
    /// running in the member's router and may still finish, logs a WARN, the
    /// table possibly still stale): a change of primary that keeps every
    /// slot set passes the comparison, and the fresh table lets the next walk
    /// on that client pin from it. A matching table at the start is not
    /// re-synced: the end's re-sync, after the walk's last step and verdict,
    /// serves the next walk. Sends nothing but the re-sync.
    async fn check_table(
        &mut self,
        map: &[SlotOwner],
        routing: Option<&ClusterRouting>,
        at: WalkEnd,
    ) {
        if !self.ownership_unproven {
            return;
        }
        // The seam's fault forges the verdict of a comparison actually made.
        let differs = self.faults.table_mismatch() || !table_partition_matches(map, routing);
        if !differs {
            // At the end, a matching table is still re-synced (once per
            // walk): the comparison cannot see a change of primary that keeps
            // every slot set, and a fresh table bounds such a miss to this
            // pass on this pool member. Housekeeping only: no reason is
            // recorded and coverage is not lost, whatever the outcome, since
            // this walk's last command was the end verdicts. A failed sync
            // leaves the table as it was, and a timed-out one keeps running
            // in the member's router and may still finish (the walk only
            // stopped waiting), so the table may stay stale; either is
            // logged, since such a miss can then outlast this pass.
            if at == WalkEnd::End && !self.resynced {
                self.resynced = true;
                let refresh = refresh_routing(&self.client).await;
                match refresh {
                    RoutingRefresh::Refreshed => {}
                    RoutingRefresh::Failed(_) | RoutingRefresh::TimedOut { .. } => {
                        tracing::warn!(
                            pattern = %self.pattern,
                            refresh = %refresh,
                            "SCAN walk's end-of-walk routing-table refresh failed or timed out \
                             after a node's own slot check could not confirm it serves its \
                             primary; a primary change the table comparison cannot see may \
                             persist beyond this pass"
                        );
                    }
                }
            }
            return;
        }
        self.coverage_lost = true;
        if self.resynced {
            return;
        }
        self.resynced = true;
        let refresh = refresh_routing(&self.client).await;
        self.record(WalkError::changed(ServingChange::StaleTable {
            target: Arc::clone(&self.pattern),
            at,
            map: render_map(map),
            table: render_table(routing),
            refresh,
        }));
    }

    /// Record why a coverage walk is unproven. The answered reason is the
    /// first coverage-losing reason (`NotOwner`, `SlotMap`, `StaleTable`) if
    /// one was recorded, otherwise the first proof-only reason (`OpenSlot`, an
    /// unusable myself line, a failed `CLUSTER NODES` read): a coverage-losing
    /// reason replaces a held proof-only one and is never itself replaced, so
    /// a walk counted as incomplete always answers why. Both reasons are
    /// classified by their variant ([`loses_coverage`]), never by
    /// `coverage_lost`, which the callers set before recording.
    fn record(&mut self, reason: WalkError) {
        let replace = match &self.unproven {
            None => true,
            Some(held) => !loses_coverage(held) && loses_coverage(&reason),
        };
        if replace {
            self.unproven = Some(reason);
        }
    }

    /// A [`ServingChange::NotOwner`] for `primary`: the client's routing
    /// table is re-synced first ([`refresh_routing`]), and how that went is
    /// recorded. A guarded walk fails on it; a coverage walk records it
    /// ([`Self::coverage_not_owner`]). Nothing is walked again in this walk.
    async fn not_owner(
        &self,
        primary: &MapPrimary,
        node: Option<String>,
        served: Option<SlotSet>,
        at: WalkEnd,
    ) -> WalkError {
        let refresh = refresh_routing(&self.client).await;
        WalkError::changed(ServingChange::NotOwner {
            target: Arc::clone(&self.pattern),
            primary: primary.primary.clone(),
            node,
            expected: primary.slots.clone(),
            served,
            at,
            refresh,
        })
    }

    /// The walk's end checks, by its guard level.
    async fn check_end(&mut self) -> Result<(), WalkError> {
        match self.guard {
            KeyspaceGuard::Serving => self.check_serving_end().await,
            KeyspaceGuard::Job => self.check_job_end().await,
            KeyspaceGuard::Coverage => self.check_coverage_end().await,
        }
    }

    /// A DR Job's walk's end fingerprint against the start's: the same slot
    /// map, then, per primary on its node route, its node still serving
    /// exactly its slots with none open ([`Self::check_owner`]). Standalone,
    /// nothing.
    async fn check_job_end(&self) -> Result<(), WalkError> {
        let Some(start_map) = &self.map else {
            return Ok(());
        };
        let end_map = self.read_map().await?;
        if end_map != *start_map {
            return Err(WalkError::changed(ServingChange::SlotMap {
                target: Arc::clone(&self.pattern),
                before: render_map(start_map),
                after: render_map(&end_map),
            }));
        }
        for primary in &self.primaries {
            if let Some(map) = &primary.map {
                self.check_owner(map, &primary.route, WalkEnd::End).await?;
            }
        }
        Ok(())
    }

    /// A coverage walk's end checks (module docs, *Coverage walks*): a fresh
    /// slot map against the start's (a change recorded as
    /// [`ServingChange::SlotMap`]), then each walked primary's verdict again
    /// ([`Self::check_coverage`]), then, on a walk whose ownership is
    /// unproven, the end map against the client's current routing table, and
    /// a re-sync of that table, matching or not, once per walk
    /// ([`Self::check_table`]; the start snapshot would only repeat the start
    /// check or the map comparison). Fails only on a store fault: the map
    /// read, or a refused node pin. Standalone, nothing.
    async fn check_coverage_end(&mut self) -> Result<(), WalkError> {
        let end_map = match &self.map {
            Some(start_map) => {
                let end_map = self.read_map().await?;
                if end_map != *start_map {
                    let change = WalkError::changed(ServingChange::SlotMap {
                        target: Arc::clone(&self.pattern),
                        before: render_map(start_map),
                        after: render_map(&end_map),
                    });
                    self.coverage_lost = true;
                    self.record(change);
                }
                Some(end_map)
            }
            None => None,
        };
        let walked: Vec<(MapPrimary, Route)> = self
            .primaries
            .iter()
            .filter_map(|p| Some((p.map.clone()?, p.route.clone())))
            .collect();
        for (primary, route) in walked {
            self.check_coverage(&primary, &route, WalkEnd::End).await?;
        }
        // After the end verdicts, so a re-sync here cannot make
        // `ensure_routable` refuse a `CLUSTER NODES` still to be read.
        if let Some(end_map) = end_map {
            let routing = self.client.cached_cluster_state();
            self.check_table(&end_map, routing.as_ref(), WalkEnd::End)
                .await;
        }
        Ok(())
    }

    /// A guarded walk's end fingerprint against the start's: the same slot
    /// map, then, per primary on its node route, its witness unchanged, no
    /// open slot, and its node still serving exactly its slots. Standalone,
    /// the witness alone.
    async fn check_serving_end(&self) -> Result<(), WalkError> {
        if let Some(start_map) = &self.map {
            let end_map = self.read_map().await?;
            if end_map != *start_map {
                return Err(WalkError::changed(ServingChange::SlotMap {
                    target: Arc::clone(&self.pattern),
                    before: render_map(start_map),
                    after: render_map(&end_map),
                }));
            }
        }
        for (index, primary) in self.primaries.iter().enumerate() {
            // The test seam's `EndWitness` fault matches this read only.
            self.faults.end_witness_of(Some(index));
            let now = match &primary.map {
                Some(map) => self.read_primary(map, &primary.route, WalkEnd::End).await,
                None => {
                    read_witness(
                        &self.client,
                        &primary.route,
                        &self.faults,
                        "SCAN",
                        &self.pattern,
                    )
                    .await
                }
            };
            self.faults.end_witness_of(None);
            let now = now?;
            // Every primary of a guarded walk carries its start witness.
            let Some(start) = &primary.start else {
                continue;
            };
            if let Some(change) = changed_between("SCAN", &self.pattern, start, &now) {
                return Err(WalkError::changed(change));
            }
        }
        Ok(())
    }
}

/// Whether a coverage walk's recorded `reason` means a primary of its map may
/// not have been walked on its owner: a [`ServingChange::NotOwner`], a
/// [`ServingChange::SlotMap`] or a [`ServingChange::StaleTable`] (an unproven
/// pin resolved from an out-of-date table). Every other recorded reason only
/// says a node's ownership could not be proven. Pure.
fn loses_coverage(reason: &WalkError) -> bool {
    matches!(
        reason,
        WalkError::Changed(change)
            if matches!(
                **change,
                ServingChange::NotOwner { .. }
                    | ServingChange::SlotMap { .. }
                    | ServingChange::StaleTable { .. }
            )
    )
}

/// Every key matching `pattern`, by a guarded keyspace walk on `client`
/// ([`KeyspaceWalk`]).
pub async fn collect_keyspace(client: &Client, pattern: &str) -> Result<Vec<Key>, WalkError> {
    KeyspaceWalk::start(client, pattern).await?.collect().await
}

/// The distinct keys one DR Job's walk of `pattern` on `client` returns
/// ([`KeyspaceWalk::start_job`]), once it completed.
pub async fn collect_job_keyspace(
    client: &Client,
    pattern: &str,
) -> Result<std::collections::HashSet<Key>, WalkError> {
    let mut walk = KeyspaceWalk::start_job(client, pattern).await?;
    let mut keys = std::collections::HashSet::new();
    while let Some(page) = walk.next_page().await? {
        keys.extend(page);
    }
    Ok(keys)
}

/// Every key matching `pattern`, by a DR Job's agreeing keyspace walk on
/// `client` (module docs, *The DR Jobs' walks*): two walks
/// ([`collect_job_keyspace`]), the second started only after the first
/// answered its end, and the keys used only when the two de-duplicated sets
/// are equal ([`keyspace_walks_agree`]); otherwise a
/// [`WalkIncomplete::WalksDisagree`]. Each key once. Holds both walks' key
/// sets until the comparison.
pub async fn collect_keyspace_agreed(
    client: &Client,
    pattern: &str,
) -> Result<Vec<Key>, WalkError> {
    let first = collect_job_keyspace(client, pattern).await?;
    // Sequential, never concurrent: two walks running at once could hold the
    // same cursor when the store changes, and skip the same keys.
    let second = collect_job_keyspace(client, pattern).await?;
    keyspace_walks_agree(pattern, &first, &second)?;
    Ok(first.into_iter().collect())
}

// ── The test seam ──────────────────────────────────────────────────────────

/// One fault a test arms on the walks of one key or pattern
/// ([`arm_walk_fault_for_test`]). Compiled out of release builds.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WalkFault {
    /// The walk's witness read number `read` (0-based, over every
    /// `INFO server` the walk sends, in order) reports `run_id:` with
    /// `run_id`, or a reply with no run_id line when `None`, in place of the
    /// server's. A valid id other than the server's forges a change of
    /// process; `"0"`, `""` or a value of the wrong length forges a
    /// placeholder.
    ///
    /// A keyed walk reads, in its first step's pipeline, its start witness
    /// and then that step's after-step witness; then the after-step witness
    /// in each later step's pipeline; plus one R1 re-read after each failed
    /// step attempt once it has its start witness. A failed attempt reads
    /// the start witness its pipeline led with when that answered and the
    /// step alone failed, and never its after-step witness; a first pipeline
    /// that failed as a whole reads none, so the resent pipeline's witnesses
    /// are reads 0 and 1. A keyspace walk reads, on a cluster, one per
    /// primary for its start fingerprint (in slot order), the after-step
    /// witness in each landed step's pipeline, one R1 re-read after each
    /// failed step attempt, then one per primary for its end fingerprint;
    /// standalone, one at its start, the per-step ones, and one at its end.
    /// Where no step attempt fails, a walk's reads are therefore numbered as
    /// one before its first step and one after each step. A coverage walk
    /// and a DR Job's walk read none, so this never matches them.
    RunId {
        /// Which read.
        read: usize,
        /// The run_id it reports; `None` for no run_id line.
        run_id: Option<String>,
    },
    /// The walk's step number `step` (0-based, in order; a keyspace walk
    /// numbers its steps across its primaries) fails with a transient (`IO`)
    /// error, before anything is sent, on each of its first `times`
    /// attempts: its whole pipeline, the witnesses riding with it included,
    /// fails as one, the shape a lost connection gives, so a keyed walk's
    /// failed first pipeline reads no start witness and is resent whole.
    /// Counted across the attempts of every caller, so a step a caller
    /// retried itself counts each call. Matches the keyspace and keyed walks'
    /// steps (a DR Job's keyspace walk's included) and a [`RankWalk`]'s
    /// reads, never a [`CountedScan`]'s step. A [`RankWalk`] numbers its
    /// reads apart from [`Self::DropMember`]'s replies: every read that
    /// answered counts, its `ZCARD` reads alone included, in order, from 0
    /// (the value removal's index walk: its start `ZCARD` is read 0, its
    /// pages reads 1 and on, its end `ZCARD` the last; a recount page, with
    /// its `ZCARD` in its pipeline, is one read). The read's first `times`
    /// attempts fail with the transient error before anything is sent,
    /// counted across the attempts of every caller, so a read the value
    /// removal's write step retries counts each call.
    StepError {
        /// Which step.
        step: usize,
        /// How many of its attempts fail.
        times: u32,
    },
    /// The walk's step number `step` (numbered as [`Self::StepError`]
    /// numbers them) is sent with its pipeline's witnesses, and its own reply
    /// is read as a transient error reply (`LOADING …`, the store not yet
    /// serving) on each of its first `times` attempts that answered one
    /// reply per command and whose lead witness, when the pipeline led with
    /// one, was usable; the witnesses' replies are read as the server sent
    /// them. So the step alone fails: a keyed walk's first step keeps the
    /// start witness its pipeline led with (that read counts, as a
    /// [`Self::RunId`] read), makes its R1 re-read against it, and, while
    /// the walk's own attempts have room (the periodic pass's owed walks, a
    /// guarded keyspace walk), sends its next attempt as
    /// `[step; INFO server]`, while the failed attempt's after-step witness
    /// is left unread and uncounted. A caller that passes one attempt keeps
    /// no lead past the attempt: its own retry is a fresh first step that
    /// resends `[INFO server; step; INFO server]`, so its reads run 0 (the
    /// lead) and 1 (the R1 re-read), then 2 and 3 in the resent pipeline.
    /// Counted apart from `StepError`'s attempts: an attempt `StepError`
    /// fails, which sends nothing, is not one of these. Matches only a step
    /// whose pipeline carries a witness, never a coverage walk's or a DR
    /// Job's walk's: its step's one reply cannot be told apart from a whole
    /// pipeline's failure. What it
    /// cannot show: the server ran the step and only its reply is
    /// discarded, so no store state differs from a landed step's (harmless
    /// for a read step).
    StepReplyError {
        /// Which step.
        step: usize,
        /// How many of its answered attempts read the error.
        times: u32,
    },
    /// The walk's witness read number `read` (numbered as [`Self::RunId`]
    /// numbers them), when it rides in a step's pipeline, reads a transient
    /// error reply (`LOADING …`) in place of its `INFO server` reply, on each
    /// of the first `times` times a witness of that number is read: the walk
    /// sees a failed witness read, `WalkError::Read` of `INFO server`. A
    /// failed read is not counted, as a real one is not, so the witness read
    /// next is again number `read`. Matches only a witness in a step's
    /// pipeline — a keyed walk's start witness before its first step, or a
    /// step's after-step witness — never one read alone (a keyspace walk's
    /// fingerprint, an R1 re-read), whose reply is the server's and whose
    /// number passes. What the walk's caller does with the error is its
    /// own: the periodic pass ends with it; a caller that retries a step
    /// itself would resend it with its witnesses. What it cannot show: the
    /// server answered the `INFO
    /// server`, so the process behind the forged failure is still the one
    /// the step reached.
    WitnessReplyError {
        /// Which read.
        read: usize,
        /// How many times a witness of that number fails.
        times: u32,
    },
    /// A keyspace walk's `CLUSTER NODES` read number `read` (0-based: one
    /// per primary in slot order for its start fingerprint, then one per
    /// primary for its end fingerprint; a coverage walk reads none for a
    /// primary its routing table names no node for, and at its end reads
    /// only the primaries it walked) reads `reply` in place of the server's.
    /// Only a read that answered counts. Forges a myself line listing other
    /// slots (`NotOwner`), an open slot, or an unusable line (no myself line,
    /// a short one, a bad slot field).
    ClusterNodesReply {
        /// Which read.
        read: usize,
        /// The reply read in its place.
        reply: String,
    },
    /// A keyspace walk's primary number `primary` (0-based, in slot order)
    /// is pinned to `server` in place of the node the client's routing
    /// table resolved (or when it resolved none). A `server` the table does
    /// not list as a primary fails the walk before anything is sent to it,
    /// with a `WalkError::Step` (`Route::ensure_routable`), not `NotOwner`;
    /// a listed primary that serves other slots reaches `NotOwner`.
    PinNode {
        /// Which primary.
        primary: usize,
        /// The node it is pinned to.
        server: fred::types::config::Server,
    },
    /// A keyspace walk's primary number `primary` (0-based, in slot order,
    /// as [`Self::PinNode`] numbers them) resolves to no node, as if the
    /// client's routing table named none for its pin slot: on a cluster, at
    /// the walk's start, before its `CLUSTER NODES` is read. A guarded walk
    /// and a DR Job's walk then fail at the start with
    /// [`ServingChange::NotOwner`] (no node, no served slots, after a routing
    /// re-sync); a coverage walk leaves the primary unwalked and records the
    /// same `NotOwner`. Wins over a [`Self::PinNode`] armed on the same
    /// primary. A standalone walk has no pin, so this never matches it.
    UnresolvedPin {
        /// Which primary.
        primary: usize,
    },
    /// A keyspace walk's node-route pin check number `check` (0-based) is
    /// refused as if the client's routing table no longer listed the pinned
    /// node as a primary (the walk ends with `route_refused`'s `Step`);
    /// every other check reads the real table. Models a table that drops the
    /// pin and then lists it again. Checks are numbered in order:
    ///
    /// - at the start, per primary in slot order: its witness, then its
    ///   `CLUSTER NODES`;
    /// - per step attempt: the one check before its pipeline (the step and
    ///   its witness), then, after a failed attempt, its R1 re-read's; the
    ///   witness of a step that lands has no check of its own;
    /// - at the end, per primary: its witness, then its `CLUSTER NODES`.
    ///
    /// With 3 primaries the start is checks 0–5, step 0's first send is 6
    /// and its first re-read is 7; with no failed attempt, step 1's send is
    /// 7. Slot and standalone routes make no check.
    /// A coverage walk and a DR Job's walk read no witness, so their checks
    /// are, in order: per pinned primary at the start, its `CLUSTER NODES`;
    /// one per step attempt; per walked primary at the end, its `CLUSTER
    /// NODES`.
    RefuseRoute {
        /// Which check.
        check: usize,
    },
    /// A keyspace walk's END fingerprint witness of primary number `primary`
    /// (0-based, in slot order; a standalone walk's one end witness is
    /// primary 0) reports `run_id:` with `run_id`, or a reply with no run_id
    /// line when `None`, in place of the server's. Counted apart from the
    /// `RunId` reads, so it targets the end comparison whatever number of
    /// steps the walk takes; where both match one read, this one wins. A
    /// valid id other than the primary's start witness forges a change at
    /// the end; a placeholder forges a missing witness there. Never matches
    /// a coverage walk or a DR Job's walk, which read no witness.
    EndWitness {
        /// Which primary.
        primary: usize,
        /// The run_id it reports; `None` for no run_id line.
        run_id: Option<String>,
    },
    /// A coverage walk's routing-table comparison number `check`
    /// (`table_partition_matches`) reports that the table differs from the
    /// map, whatever the two hold: a forged verdict, as [`Self::RefuseRoute`]
    /// forges a pin check's. Comparisons are numbered from 0, in order, and
    /// counted only when one is made: at an end where, by then, the walk's
    /// ownership is unproven (an unusable myself line or a failed
    /// `CLUSTER NODES` read). A walk with an unproven start verdict makes
    /// comparison 0 at the start and 1 at the end; one whose first unproven
    /// verdict is an end one makes only comparison 0, at the end; a walk that
    /// proves every pin makes none. A guarded walk and a DR Job's walk make
    /// no comparison, so this never matches them.
    TablePartitionDiffers {
        /// Which comparison.
        check: usize,
    },
    /// A keyspace walk's `CLUSTER SLOTS` read number `read` (0-based: 0 the
    /// start map, 1 the end map; at every guard level; only a read that
    /// answered and parsed counts) comes back with its first two primaries,
    /// in slot order, named the other's name: every range keeps its slots,
    /// so the per-node division of the slots is unchanged (no
    /// [`ServingChange::StaleTable`]), but the map differs from one read
    /// without the fault. Armed on one read only, the start (0) or the end
    /// (1), it makes the start and end maps differ, so the walk records a
    /// [`ServingChange::SlotMap`] (a coverage walk then answers
    /// `walked_every_primary() == false`; a guarded walk or a DR Job's walk
    /// fails). Armed on
    /// both reads, 0 and 1, the two maps agree and a walk completes with the
    /// names exchanged. A map with fewer than two primaries is read as it
    /// is.
    SlotMapPrimariesSwapped {
        /// Which read.
        read: usize,
    },
    /// A DR Job's walk of the target loses one key or member from one reply,
    /// as a cursor continued on another process might skip it: so a test
    /// can show each completeness proof fails closed (module docs, *The DR
    /// Jobs' walks*). Matches only the walk number `walk` of the target
    /// (0-based, counting every walk of it that starts while the fault is
    /// armed: an agreeing keyspace walk's first walk and its second are two,
    /// and a pattern or key walked again later in the same Job counts on),
    /// and in it the reply number `step` (0-based, counting only replies
    /// that answered), or every reply of it when `step` is `None`: a
    /// keyspace walk's landed `SCAN` steps across its primaries; a
    /// [`RankWalk`]'s pages (`ZRANGE` reads, never a `ZCARD` read alone); a
    /// [`CountedScan`]'s steps. Drops `member` from that reply when it holds
    /// it, or the reply's first member when `member` is `None` (nothing from
    /// an empty reply). Never matches a guarded or a coverage keyspace walk,
    /// or a [`KeyedWalk`].
    DropMember {
        /// Which walk of the target.
        walk: usize,
        /// Which reply of that walk; `None` for every reply.
        step: Option<usize>,
        /// The key or member dropped; `None` for the reply's first.
        member: Option<String>,
    },
    /// [`Self::DropMember`]'s counterpart: a DR Job's walk of the target
    /// gets one key or member twice in one reply, as a SCAN-family cursor
    /// may return a member again (a rehash between steps), so a test can
    /// show each walk counts a repeated member once. Matches the walk and
    /// the replies [`Self::DropMember`] matches, numbered the same way (both
    /// count the same replies). Repeats `member`, right after it, when the
    /// reply holds it, or the reply's first member when `member` is `None`
    /// (nothing in an empty reply). On a [`RankWalk`] a repeat is a page
    /// answering more members than its ranks span, which each rank walk's
    /// own check reports: on the value removal's descending pages the
    /// page-length check, a [`WalkIncomplete::ShortPage`] whose `returned`
    /// is above the span; on the recount's [`page_by_rank_counted`] the count
    /// check, a [`WalkIncomplete::CountMismatch`] whose `returned` counts the
    /// repeat.
    RepeatMember {
        /// Which walk of the target.
        walk: usize,
        /// Which reply of that walk; `None` for every reply.
        step: Option<usize>,
        /// The key or member repeated; `None` for the reply's first.
        member: Option<String>,
    },
}

/// A usable run_id a test forges a witness read with (40 hex characters,
/// not all zeros). Compiled out of release builds.
#[cfg(any(test, feature = "test-fault-injection"))]
pub const TEST_RUN_ID_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

/// Another usable run_id, so a walk that reads [`TEST_RUN_ID_A`] and then
/// this one crossed a change of serving process. Compiled out of release
/// builds.
#[cfg(any(test, feature = "test-fault-injection"))]
pub const TEST_RUN_ID_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

#[cfg(any(test, feature = "test-fault-injection"))]
mod faults {
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::{SlotOwner, WalkFault, TEST_RUN_ID_A, TEST_RUN_ID_B};

    /// One armed fault.
    struct Armed {
        /// Its guard's id.
        id: u64,
        /// The key or pattern.
        target: String,
        fault: WalkFault,
        /// The walks of `target` started since it was armed: the number the
        /// next one carries, which a `DropMember` or `RepeatMember` fault
        /// matches.
        walks: usize,
    }

    /// Every armed fault.
    static ARMED: Mutex<Vec<Armed>> = Mutex::new(Vec::new());
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    /// Arm `fault` on `target`: its id.
    fn arm(target: &str, fault: WalkFault) -> u64 {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        ARMED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Armed {
                id,
                target: target.to_owned(),
                fault,
                walks: 0,
            });
        id
    }

    /// Test seam: every walk of `target` (a key, or a keyspace walk's
    /// pattern) that starts while the returned guard is held carries `fault`
    /// (see [`WalkFault`] for how reads and steps are numbered). Keyed by the
    /// prefixed key or pattern, so tests sharing a process do not reach each
    /// other's walks. More than one fault may be armed on one target.
    /// Compiled out of release builds.
    pub fn arm_walk_fault_for_test(target: &str, fault: WalkFault) -> ArmedWalkFault {
        ArmedWalkFault {
            ids: vec![arm(target, fault)],
        }
    }

    /// Test seam: forge a change of serving process at witness read `read`
    /// (numbered as [`WalkFault::RunId`] numbers them) of every walk of
    /// `target` that starts while the returned guard is held: reads `0` to
    /// `read - 1` report [`TEST_RUN_ID_A`] and read `read` reports
    /// [`TEST_RUN_ID_B`], whatever the server's own run_id is. So a walk
    /// whose start witness is read 0 sees the change, from A to B, at read
    /// `read` (a standalone keyed or keyspace walk: after its step `read`,
    /// counting from 1, when no step attempt fails). `read` 0 forges only
    /// the start witness, as B: no change. One guard disarms every fault
    /// it armed. Compiled out of release builds.
    pub fn forge_process_change_for_test(target: &str, read: usize) -> ArmedWalkFault {
        let mut ids: Vec<u64> = (0..read)
            .map(|before| {
                arm(
                    target,
                    WalkFault::RunId {
                        read: before,
                        run_id: Some(TEST_RUN_ID_A.to_owned()),
                    },
                )
            })
            .collect();
        ids.push(arm(
            target,
            WalkFault::RunId {
                read,
                run_id: Some(TEST_RUN_ID_B.to_owned()),
            },
        ));
        ArmedWalkFault { ids }
    }

    /// The armed seam of [`arm_walk_fault_for_test`] or
    /// [`forge_process_change_for_test`]; disarms every fault it armed when
    /// dropped. Walks already started keep what they read at their start.
    #[must_use = "the seam is disarmed when this guard drops"]
    #[derive(Debug)]
    pub struct ArmedWalkFault {
        ids: Vec<u64>,
    }

    impl Drop for ArmedWalkFault {
        fn drop(&mut self) {
            ARMED
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retain(|armed| !self.ids.contains(&armed.id));
        }
    }

    /// One walk's faults, taken at its start, and its counts of witness
    /// reads, landed steps, the current step's failed attempts, `CLUSTER
    /// NODES` reads, pin checks, routing-table comparisons, `CLUSTER SLOTS`
    /// reads, the replies a `DropMember` or `RepeatMember` fault is matched
    /// against, and a [`super::RankWalk`]'s reads that answered.
    #[derive(Debug, Default)]
    pub(super) struct WalkFaults {
        faults: Vec<WalkFault>,
        /// The replies offered to a `DropMember` or `RepeatMember` fault so
        /// far.
        member_replies: AtomicUsize,
        /// A `RankWalk`'s reads that answered: the number a `StepError`
        /// fault matches on it.
        rank_reads: AtomicUsize,
        reads: AtomicUsize,
        steps: AtomicUsize,
        attempts: AtomicU32,
        /// The current step's attempts whose reply a `StepReplyError`
        /// fault replaced.
        reply_attempts: AtomicU32,
        /// The witness reads a `WitnessReplyError` fault failed at the
        /// current read number (`reads`); reset when a read counts.
        witness_errors: AtomicU32,
        nodes_reads: AtomicUsize,
        route_checks: AtomicUsize,
        table_checks: AtomicUsize,
        map_reads: AtomicUsize,
        /// The primary whose end-fingerprint witness is being read, plus
        /// one; 0 outside the end fingerprint.
        end_primary: AtomicUsize,
    }

    impl WalkFaults {
        /// The faults armed on `target` for a walk of it starting now: every
        /// one but a `DropMember` or `RepeatMember` armed for another walk
        /// number. Counts the walk on each fault armed on `target`.
        pub(super) fn for_target(target: &str) -> Self {
            let mut armed = ARMED
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut faults = Vec::new();
            for entry in armed.iter_mut().filter(|entry| entry.target == target) {
                let this_walk = entry.walks;
                entry.walks += 1;
                if let WalkFault::DropMember { walk, .. } | WalkFault::RepeatMember { walk, .. } =
                    entry.fault
                {
                    if walk != this_walk {
                        continue;
                    }
                }
                faults.push(entry.fault.clone());
            }
            Self {
                faults,
                ..Self::default()
            }
        }

        /// Under an armed `DropMember` or `RepeatMember` fault for this
        /// reply of the walk, drop or repeat its member (`text` reads a
        /// member's text) or its first, the faults applied in the order
        /// they were armed; every call counts one reply.
        pub(super) fn forge_reply<T: Clone>(
            &self,
            members: &mut Vec<T>,
            text: impl Fn(&T) -> Option<String>,
        ) {
            let reply = self.member_replies.fetch_add(1, Ordering::Relaxed);
            for fault in &self.faults {
                let (step, member, repeat) = match fault {
                    WalkFault::DropMember { step, member, .. } => (step, member, false),
                    WalkFault::RepeatMember { step, member, .. } => (step, member, true),
                    _ => continue,
                };
                if step.is_some_and(|step| step != reply) {
                    continue;
                }
                let at = match member {
                    None => (!members.is_empty()).then_some(0),
                    Some(member) => members
                        .iter()
                        .position(|m| text(m).as_deref() == Some(member.as_str())),
                };
                let Some(at) = at else {
                    continue;
                };
                if repeat {
                    let again = members[at].clone();
                    members.insert(at + 1, again);
                } else {
                    members.remove(at);
                }
            }
        }

        /// The injected error of an armed `StepError` fault for this attempt
        /// of a [`super::RankWalk`]'s current read (its `walk` command,
        /// `ZRANGE` or `ZCARD`), numbered by the walk's reads that answered.
        pub(super) fn rank_read_error(&self, walk: &str) -> Option<fred::error::Error> {
            let read = self.rank_reads.load(Ordering::Relaxed);
            let fails = self.faults.iter().any(|fault| {
                matches!(
                    fault,
                    WalkFault::StepError { step: at, times }
                        if *at == read && self.attempts.load(Ordering::Relaxed) < *times
                )
            });
            if !fails {
                return None;
            }
            self.attempts.fetch_add(1, Ordering::Relaxed);
            Some(fred::error::Error::new(
                fred::error::ErrorKind::IO,
                format!("test seam: read {read} of a rank walk ({walk}) failed before it was sent"),
            ))
        }

        /// A [`super::RankWalk`]'s read answered: the next one starts.
        pub(super) fn rank_read_answered(&self) {
            self.rank_reads.fetch_add(1, Ordering::Relaxed);
            self.attempts.store(0, Ordering::Relaxed);
        }

        /// `info`, or the forged reply of an armed `EndWitness` fault for
        /// the end-fingerprint witness being read, else of an armed `RunId`
        /// fault for this read.
        pub(super) fn witness_reply(&self, info: String) -> String {
            let forged = |run_id: &Option<String>| match run_id {
                Some(run_id) => format!("# Server\r\nrun_id:{run_id}\r\n"),
                None => "# Server\r\n".to_owned(),
            };
            let read = self.reads.fetch_add(1, Ordering::Relaxed);
            self.witness_errors.store(0, Ordering::Relaxed);
            if let Some(end) = self.end_primary.load(Ordering::Relaxed).checked_sub(1) {
                for fault in &self.faults {
                    if let WalkFault::EndWitness { primary, run_id } = fault {
                        if *primary == end {
                            return forged(run_id);
                        }
                    }
                }
            }
            for fault in &self.faults {
                if let WalkFault::RunId { read: at, run_id } = fault {
                    if *at == read {
                        return forged(run_id);
                    }
                }
            }
            info
        }

        /// The end fingerprint is reading `primary`'s witness (`Some`), or
        /// is not (`None`): what an `EndWitness` fault matches.
        pub(super) fn end_witness_of(&self, primary: Option<usize>) {
            self.end_primary
                .store(primary.map_or(0, |p| p + 1), Ordering::Relaxed);
        }

        /// The injected error of an armed `StepError` fault for this attempt
        /// of the current step.
        pub(super) fn step_error(&self, walk: &str) -> Option<fred::error::Error> {
            let step = self.steps.load(Ordering::Relaxed);
            let fails = self.faults.iter().any(|fault| match fault {
                WalkFault::StepError { step: at, times } => {
                    *at == step && self.attempts.load(Ordering::Relaxed) < *times
                }
                WalkFault::RunId { .. }
                | WalkFault::StepReplyError { .. }
                | WalkFault::WitnessReplyError { .. }
                | WalkFault::ClusterNodesReply { .. }
                | WalkFault::PinNode { .. }
                | WalkFault::UnresolvedPin { .. }
                | WalkFault::RefuseRoute { .. }
                | WalkFault::EndWitness { .. }
                | WalkFault::TablePartitionDiffers { .. }
                | WalkFault::SlotMapPrimariesSwapped { .. }
                | WalkFault::DropMember { .. }
                | WalkFault::RepeatMember { .. } => false,
            });
            if !fails {
                return None;
            }
            self.attempts.fetch_add(1, Ordering::Relaxed);
            Some(fred::error::Error::new(
                fred::error::ErrorKind::IO,
                format!("test seam: step {step} of a {walk} walk failed before it was sent"),
            ))
        }

        /// `reply`, the current step's own reply in its pipeline's answer
        /// (called only for a pipeline that carries a witness), or, under an
        /// armed `StepReplyError` fault for this attempt of the current
        /// step, a transient error reply in its place.
        pub(super) fn step_reply(
            &self,
            walk: &str,
            reply: Result<fred::types::Value, fred::error::Error>,
        ) -> Result<fred::types::Value, fred::error::Error> {
            let step = self.steps.load(Ordering::Relaxed);
            let fails = self.faults.iter().any(|fault| {
                matches!(
                    fault,
                    WalkFault::StepReplyError { step: at, times }
                        if *at == step && self.reply_attempts.load(Ordering::Relaxed) < *times
                )
            });
            if !fails {
                return reply;
            }
            self.reply_attempts.fetch_add(1, Ordering::Relaxed);
            Err(not_serving_reply(&format!(
                "the reply of step {step} of a {walk} walk"
            )))
        }

        /// The forged failure of an armed `WitnessReplyError` fault for the
        /// witness read about to be read in a step's pipeline (the current
        /// read number), if any. A failure is not counted as a read.
        pub(super) fn witness_reply_error(&self) -> Option<fred::error::Error> {
            let read = self.reads.load(Ordering::Relaxed);
            let fails = self.faults.iter().any(|fault| {
                matches!(
                    fault,
                    WalkFault::WitnessReplyError { read: at, times }
                        if *at == read && self.witness_errors.load(Ordering::Relaxed) < *times
                )
            });
            if !fails {
                return None;
            }
            self.witness_errors.fetch_add(1, Ordering::Relaxed);
            Some(not_serving_reply(&format!("witness read {read}")))
        }

        /// `nodes`, or the reply of an armed `ClusterNodesReply` fault for
        /// this `CLUSTER NODES` read.
        pub(super) fn cluster_nodes_reply(&self, nodes: String) -> String {
            let read = self.nodes_reads.fetch_add(1, Ordering::Relaxed);
            for fault in &self.faults {
                if let WalkFault::ClusterNodesReply { read: at, reply } = fault {
                    if *at == read {
                        return reply.clone();
                    }
                }
            }
            nodes
        }

        /// Whether an armed `RefuseRoute` fault refuses this node-route pin
        /// check; every call counts one check.
        pub(super) fn refuse_route(&self) -> bool {
            let check = self.route_checks.fetch_add(1, Ordering::Relaxed);
            self.faults
                .iter()
                .any(|fault| matches!(fault, WalkFault::RefuseRoute { check: at } if *at == check))
        }

        /// Whether an armed `TablePartitionDiffers` fault forges this
        /// routing-table comparison as "differs"; every call counts one
        /// comparison, so it is called only for a comparison being made.
        pub(super) fn table_mismatch(&self) -> bool {
            let check = self.table_checks.fetch_add(1, Ordering::Relaxed);
            self.faults.iter().any(|fault| {
                matches!(fault, WalkFault::TablePartitionDiffers { check: at } if *at == check)
            })
        }

        /// `map`, a parsed `CLUSTER SLOTS` read, or, under an armed
        /// `SlotMapPrimariesSwapped` fault for this read, `map` with its first
        /// two primaries in slot order named each other's name; every call
        /// counts one read.
        pub(super) fn slot_map_reply(&self, mut map: Vec<SlotOwner>) -> Vec<SlotOwner> {
            let read = self.map_reads.fetch_add(1, Ordering::Relaxed);
            let armed = self.faults.iter().any(
                |fault| matches!(fault, WalkFault::SlotMapPrimariesSwapped { read: at } if *at == read),
            );
            if !armed {
                return map;
            }
            let Some(first) = map.first().map(|owner| owner.primary.clone()) else {
                return map;
            };
            let Some(second) = map
                .iter()
                .find(|owner| owner.primary != first)
                .map(|owner| owner.primary.clone())
            else {
                return map;
            };
            for owner in &mut map {
                if owner.primary == first {
                    owner.primary.clone_from(&second);
                } else if owner.primary == second {
                    owner.primary.clone_from(&first);
                }
            }
            map
        }

        /// The node an armed `PinNode` fault pins primary `primary` to.
        pub(super) fn pin_node(&self, primary: usize) -> Option<fred::types::config::Server> {
            self.faults.iter().find_map(|fault| match fault {
                WalkFault::PinNode {
                    primary: at,
                    server,
                } if *at == primary => Some(server.clone()),
                _ => None,
            })
        }

        /// Whether an armed `UnresolvedPin` fault resolves primary `primary`
        /// to no node.
        pub(super) fn unresolved_pin(&self, primary: usize) -> bool {
            self.faults.iter().any(
                |fault| matches!(fault, WalkFault::UnresolvedPin { primary: at } if *at == primary),
            )
        }

        /// The current step landed: the next one starts.
        pub(super) fn step_succeeded(&self) {
            self.steps.fetch_add(1, Ordering::Relaxed);
            self.attempts.store(0, Ordering::Relaxed);
            self.reply_attempts.store(0, Ordering::Relaxed);
        }
    }

    /// A transient server error reply, as fred hands a `-LOADING` over
    /// (`ErrorKind::Unknown`, the reply's text as its details, which
    /// `storage::client::is_store_not_serving_reply` reads as the store not
    /// yet serving), naming the forged reply `what`.
    fn not_serving_reply(what: &str) -> fred::error::Error {
        fred::error::Error::new(
            fred::error::ErrorKind::Unknown,
            format!("LOADING Valkey is loading the dataset in memory (test seam: {what})"),
        )
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
use faults::WalkFaults;
#[cfg(any(test, feature = "test-fault-injection"))]
pub use faults::{arm_walk_fault_for_test, forge_process_change_for_test, ArmedWalkFault};

/// A release build's walk faults: none.
#[cfg(not(any(test, feature = "test-fault-injection")))]
#[derive(Debug, Default)]
struct WalkFaults;

#[cfg(not(any(test, feature = "test-fault-injection")))]
impl WalkFaults {
    fn for_target(_target: &str) -> Self {
        Self
    }

    fn witness_reply(&self, info: String) -> String {
        info
    }

    fn step_error(&self, _walk: &str) -> Option<fred::error::Error> {
        None
    }

    fn step_reply(
        &self,
        _walk: &str,
        reply: Result<Value, fred::error::Error>,
    ) -> Result<Value, fred::error::Error> {
        reply
    }

    fn witness_reply_error(&self) -> Option<fred::error::Error> {
        None
    }

    fn cluster_nodes_reply(&self, nodes: String) -> String {
        nodes
    }

    fn refuse_route(&self) -> bool {
        false
    }

    fn table_mismatch(&self) -> bool {
        false
    }

    fn slot_map_reply(&self, map: Vec<SlotOwner>) -> Vec<SlotOwner> {
        map
    }

    fn pin_node(&self, _primary: usize) -> Option<Server> {
        None
    }

    fn unresolved_pin(&self, _primary: usize) -> bool {
        false
    }

    fn end_witness_of(&self, _primary: Option<usize>) {}

    fn step_succeeded(&self) {}

    fn forge_reply<T: Clone>(&self, _members: &mut Vec<T>, _text: impl Fn(&T) -> Option<String>) {}

    fn rank_read_error(&self, _walk: &str) -> Option<fred::error::Error> {
        None
    }

    fn rank_read_answered(&self) {}
}

#[cfg(test)]
mod tests {
    //! The guard's pure decisions: the run_id parse (R3), the `CLUSTER NODES`
    //! myself-line parse and the ownership decision, a coverage walk's
    //! verdict on a pinned node (`coverage_verdict`), whether a routing table
    //! divides the slots among its nodes as a map divides them among its
    //! primaries (`table_partition_matches`), the options each route sends
    //! its commands with, the verdict of an R1 retry's witness re-read
    //! (R1-a), whether a routing table still lists a node pin, which errors
    //! are fred errors, and the DR Jobs' completeness checks (b.f3t: the
    //! count check of a walk of one key and the agreement of two keyspace
    //! walks, with their wording). Beside them, a coverage walk's end: the reason
    //! it recorded answered once, after its last key, and whether it walked
    //! every primary, and its routing-table check (`check_table`: when it
    //! compares, and what a table that differs records), driven on a walk
    //! with no primary left to step, so no command is sent. The reading of a
    //! step's pipeline (`read_step_pipeline`, b.qgz): one reply per command
    //! judged in command order, each as its own command's; one error read
    //! as the whole pipeline's failed attempt; and any other count of
    //! replies, a single non-error frame included, failing closed with no
    //! reply read. One cluster-gated test sends a keyed step to a node that
    //! does not own its key, to show the one-attempt step fails rather than
    //! follow the redirect; its control shows the first pipeline's start and
    //! after-step witnesses read the key's owner, not the node an unpinned
    //! `INFO server` reaches.
    //!
    //! The guarded and coverage walks against a real store, the seam's faults
    //! and every consumer's exit are
    //! `tests/integration/storage/scan_guard_test.rs`'s (and, for a stale
    //! routing table, `scan_guard_stale_table_test.rs`'s),
    //! `storage::ext_reconcile`'s, `storage::orphan_sweep`'s,
    //! `checkpoint::checkpointer`'s, `dr::job::execute::discard`'s and the DR
    //! Job tests'.

    use super::*;

    // Two usable run_ids: the seam's.
    use super::{TEST_RUN_ID_A as RUN_A, TEST_RUN_ID_B as RUN_B};

    /// A Valkey 9.0.3 `INFO server` reply, as the server lays it out, with
    /// `run_id` set to `run_id` (or no `run_id` line when `None`).
    fn valkey_info(run_id: Option<&str>) -> String {
        let mut lines = vec![
            "# Server".to_owned(),
            "redis_version:7.2.4".to_owned(),
            "server_name:valkey".to_owned(),
            "valkey_version:9.0.3".to_owned(),
            "redis_git_sha1:00000000".to_owned(),
            "redis_mode:cluster".to_owned(),
            "process_id:7".to_owned(),
        ];
        if let Some(run_id) = run_id {
            lines.push(format!("run_id:{run_id}"));
        }
        lines.push("tcp_port:7000".to_owned());
        lines.join("\r\n") + "\r\n"
    }

    // ── R3: the run_id parse ───────────────────────────────────────────────

    /// A Valkey-shaped reply and an Azure Managed Redis-shaped one (its
    /// `process_id` is `1`, as AMR reports it) each yield their run_id; so do
    /// an upper-case hex id and one with no carriage returns.
    #[test]
    fn parse_run_id_reads_the_run_id_of_a_valkey_and_an_amr_shaped_reply() {
        let id = "3167b2266793dfa979ea481e11ad7fda032c542c";
        assert_eq!(parse_run_id(&valkey_info(Some(id))), Ok(id.to_owned()));
        let amr = format!(
            "# Server\r\nredis_version:7.4.3\r\nredis_mode:cluster\r\nprocess_id:1\r\nrun_id:{id}\r\n\
             tcp_port:10000\r\n"
        );
        assert_eq!(parse_run_id(&amr), Ok(id.to_owned()));
        let upper = id.to_ascii_uppercase();
        assert_eq!(
            parse_run_id(&valkey_info(Some(&upper))),
            Ok(upper.clone()),
            "hex of either case is a run_id"
        );
        assert_eq!(
            parse_run_id(&format!("# Server\nrun_id:{id}\n")),
            Ok(id.to_owned()),
            "lines without a carriage return"
        );
    }

    /// R3: a run_id is usable only when it is exactly 40 hex characters and
    /// not all zeros. `"0"` (ElastiCache Serverless's constant), an empty
    /// value, 39 characters, 41, forty zeros and a non-hex character are
    /// each what was seen, and a reply with no `run_id` line is `NoLine`.
    #[test]
    fn parse_run_id_refuses_a_placeholder_a_wrong_length_and_a_missing_line_as_missing() {
        let short = "a".repeat(39);
        let long = "a".repeat(41);
        let zeros = "0".repeat(40);
        let not_hex = format!("{}g", "a".repeat(39));
        for value in [
            "0",
            "",
            short.as_str(),
            long.as_str(),
            zeros.as_str(),
            not_hex.as_str(),
        ] {
            assert_eq!(
                parse_run_id(&valkey_info(Some(value))),
                Err(RunIdSeen::Value(value.to_owned())),
                "run_id {value:?} is not usable"
            );
        }
        assert_eq!(parse_run_id(&valkey_info(None)), Err(RunIdSeen::NoLine));
        assert_eq!(parse_run_id(""), Err(RunIdSeen::NoLine));
    }

    /// A missing witness names its command, its node and what was seen, for
    /// the operator's exit-3 line, and neither it nor a change of serving
    /// process is a fred error, so no caller counts either on
    /// `valkey_errors_total`. A store fault is one.
    #[test]
    fn a_missing_witness_names_its_command_and_what_was_seen_and_is_not_a_fred_error() {
        let missing = WitnessMissing {
            walk: "ZSCAN",
            target: Arc::from("le:{7}:obj:flds"),
            node: Some("127.0.0.1:7000".to_owned()),
            seen: WitnessSeen::RunId(RunIdSeen::Value("0".to_owned())),
        };
        assert_eq!(missing.seen.command(), "INFO server");
        let text = missing.to_string();
        for part in [
            "`INFO server`",
            "node 127.0.0.1:7000",
            "run_id \"0\"",
            "ZSCAN walk of le:{7}:obj:flds",
        ] {
            assert!(text.contains(part), "{text:?} names {part:?}");
        }
        let myself = WitnessMissing {
            walk: "SCAN",
            target: Arc::from("le:{*}:obj:hdr"),
            node: Some("127.0.0.1:7001".to_owned()),
            seen: WitnessSeen::MyselfLine(MyselfLineSeen::NoLine),
        };
        assert_eq!(myself.seen.command(), "CLUSTER NODES");
        let text = myself.to_string();
        for part in ["`CLUSTER NODES`", "node 127.0.0.1:7001", "no myself line"] {
            assert!(text.contains(part), "{text:?} names {part:?}");
        }

        assert!(WalkError::WitnessMissing(missing).fred_error().is_none());
        assert!(WalkError::changed(ServingChange::OpenSlot {
            target: Arc::from("p"),
            node: "127.0.0.1:7000".to_owned(),
            slots: "[1->-x]".to_owned(),
        })
        .fred_error()
        .is_none());
        let fault = WalkError::Read {
            op: "INFO server",
            source: fred::error::Error::new(fred::error::ErrorKind::IO, "reset"),
        };
        assert!(
            fault.fred_error().is_some(),
            "a store fault is a fred error"
        );
    }

    // ── The DR Jobs' completeness checks (b.f3t) ───────────────────────────

    /// The count check of a DR Job's walk of one key (`check_count`):
    /// complete only when the walk returned the cardinality read at its
    /// start and its end cardinality is the one a complete walk leaves — the
    /// start for a read-only walk, the start less the rows it deleted for
    /// accept-loss's value removal. A member skipped, a member added under
    /// the walk (whether or not a page returned it) and a delete that did
    /// not land are each a `CountMismatch` naming every figure, worded as
    /// the Jobs' ERROR lines carry it; none is a fred error.
    #[test]
    fn check_count_holds_only_for_the_start_count_returned_and_the_end_a_complete_walk_leaves() {
        let key: Arc<str> = Arc::from("le:{7}:obj:flds");
        for (case, at_end, expected_end) in [
            ("a complete read-only walk", 3, 3),
            ("a complete removal of two rows", 1, 1),
        ] {
            assert!(
                check_count("ZRANGE", &key, "ZCARD", 3, 3, at_end, expected_end).is_ok(),
                "{case}"
            );
        }
        assert!(
            check_count("HSCAN", &key, "HLEN", 0, 0, 0, 0).is_ok(),
            "an empty key"
        );
        for (case, returned, at_end, expected_end) in [
            ("a member skipped", 2, 3, 3),
            ("a member added and returned", 4, 4, 3),
            ("a member added and not returned", 3, 4, 3),
            ("a lost delete", 3, 2, 1),
        ] {
            match check_count("ZRANGE", &key, "ZCARD", 3, returned, at_end, expected_end) {
                Err(WalkError::Incomplete(incomplete)) => {
                    assert_eq!(
                        *incomplete,
                        WalkIncomplete::CountMismatch {
                            walk: "ZRANGE",
                            target: Arc::clone(&key),
                            card: "ZCARD",
                            at_start: 3,
                            returned,
                            at_end,
                            expected_end,
                        },
                        "{case}"
                    );
                    let error = WalkError::Incomplete(incomplete);
                    assert!(error.fred_error().is_none(), "{case}: not a fred error");
                    assert_eq!(
                        error.to_string(),
                        format!(
                            "the ZRANGE walk of le:{{7}}:obj:flds returned {returned} members, \
                             and ZCARD read 3 at its start and {at_end} at its end, where a \
                             complete walk returns 3 and ends at {expected_end}"
                        ),
                        "{case}"
                    );
                }
                other => panic!("{case}: a CountMismatch, got {other:?}"),
            }
        }
        let short = WalkError::incomplete(WalkIncomplete::ShortPage {
            target: Arc::clone(&key),
            start: 256,
            stop: 511,
            returned: 255,
        });
        assert!(
            short.fred_error().is_none(),
            "a short page is not a fred error"
        );
        assert_eq!(
            short.to_string(),
            "ZRANGE le:{7}:obj:flds 256 511 returned 255 members, where 256 were expected"
        );
    }

    /// The agreement check of a DR Job's two keyspace walks
    /// (`keyspace_walks_agree`): equal de-duplicated key sets agree, empty
    /// ones included; any difference is a `WalksDisagree` naming the
    /// pattern, each walk's distinct keys and how many each alone returned,
    /// worded as the Jobs' ERROR lines carry it, and not a fred error.
    #[test]
    fn keyspace_walks_agree_only_on_equal_sets_and_name_what_each_walk_alone_returned() {
        use std::collections::HashSet;

        const PATTERN: &str = "le:{*}:obj:hdr";
        let set =
            |keys: &[&str]| -> HashSet<String> { keys.iter().map(|k| k.to_string()).collect() };
        assert!(keyspace_walks_agree(PATTERN, &set(&["a", "b"]), &set(&["b", "a"])).is_ok());
        assert!(keyspace_walks_agree(PATTERN, &set(&[]), &set(&[])).is_ok());
        for (case, first, second, (only_first, only_second)) in [
            (
                "the first walk skipped a key",
                set(&["a"]),
                set(&["a", "b"]),
                (0, 1),
            ),
            (
                "the second walk skipped a key",
                set(&["a", "b"]),
                set(&["a"]),
                (1, 0),
            ),
            (
                "each walk skipped another key",
                set(&["a", "b"]),
                set(&["a", "c"]),
                (1, 1),
            ),
        ] {
            let (n_first, n_second) = (first.len() as u64, second.len() as u64);
            match keyspace_walks_agree(PATTERN, &first, &second) {
                Err(WalkError::Incomplete(incomplete)) => {
                    assert_eq!(
                        *incomplete,
                        WalkIncomplete::WalksDisagree {
                            target: Arc::from(PATTERN),
                            first: n_first,
                            second: n_second,
                            only_first,
                            only_second,
                        },
                        "{case}"
                    );
                    let error = WalkError::Incomplete(incomplete);
                    assert!(error.fred_error().is_none(), "{case}: not a fred error");
                    assert_eq!(
                        error.to_string(),
                        format!(
                            "two SCAN walks of {PATTERN} returned different keys: {n_first} and \
                             {n_second} distinct keys, {only_first} found only by the first walk \
                             and {only_second} only by the second"
                        ),
                        "{case}"
                    );
                }
                other => panic!("{case}: a WalksDisagree, got {other:?}"),
            }
        }
    }

    // ── The myself line and the ownership decision ─────────────────────────

    /// A Valkey-shaped myself line with a single slot, a range and both
    /// open-slot marker kinds: it owns the slot and the range, and the
    /// markers are open, not owned. Another node's line beside it counts for
    /// nothing.
    #[test]
    fn parse_myself_line_takes_the_slots_and_ranges_as_owned_and_the_markers_as_open() {
        let nodes = "\
292f8b365bb7edb5e285caf0b7e6ddc7265d2f4f 127.0.0.1:7002@17002 master - 0 1426238316232 3 connected 201-16383
07c37dfeb235213a872192d90877d0cd55635b91 127.0.0.1:7001@17001 myself,master - 0 1426238317239 2 connected 7 100-200 [93->-292f8b365bb7edb5e285caf0b7e6ddc7265d2f4f] [77-<-e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca]
";
        let myself = parse_myself_line(nodes).expect("a usable myself line");
        assert_eq!(myself.owned, SlotSet::from_ranges([(7, 7), (100, 200)]));
        assert_eq!(
            myself.open,
            vec![
                "[93->-292f8b365bb7edb5e285caf0b7e6ddc7265d2f4f]".to_owned(),
                "[77-<-e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca]".to_owned()
            ]
        );
    }

    /// Azure Managed Redis's myself line, verbatim from the 2026-10-13
    /// capture (`docs-internal/validation/amr-topology-20261013/captures.md`):
    /// ids `1` / `2`, epoch `0`, the same eight fixed fields, and four
    /// ranges, which are what it owns.
    #[test]
    fn parse_myself_line_reads_the_amr_line_as_its_four_ranges() {
        let nodes = "\
1 <public-ip>:8501 master - 0 0 0 connected 0-2047 4096-6143 8192-10239 12288-14335
2 <public-ip>:8500 myself,master - 0 0 0 connected 2048-4095 6144-8191 10240-12287 14336-16383
";
        let myself = parse_myself_line(nodes).expect("AMR's myself line is usable");
        assert_eq!(
            myself.owned,
            SlotSet::from_ranges([(2048, 4095), (6144, 8191), (10240, 12287), (14336, 16383)])
        );
        assert!(myself.open.is_empty());
    }

    /// A replica's myself line has its eight fixed fields and no slot field:
    /// it owns nothing, so a pin that reaches a replica serves no primary's
    /// slots (`NotOwner`, exit 1), rather than reading as unusable.
    #[test]
    fn parse_myself_line_reads_a_replicas_line_as_owning_no_slot() {
        let nodes = "e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca 127.0.0.1:7004@17004 \
                     myself,slave 07c37dfeb235213a872192d90877d0cd55635b91 0 1426238316232 2 \
                     connected\n";
        let myself = parse_myself_line(nodes).expect("a replica's line is usable");
        assert_eq!(myself.owned, SlotSet::default());
        assert!(!serves_exactly(
            &SlotSet::from_ranges([(0, 5460)]),
            &myself.owned
        ));
    }

    /// No myself line (none at all, or only other nodes' lines), a myself
    /// line short of its eight fixed fields, and a slot field outside the
    /// `N` / `A-B` grammar (`abc`, a reversed range, a slot past 16383, a
    /// sign) are each unusable, and say what was seen.
    #[test]
    fn parse_myself_line_refuses_a_missing_or_short_line_and_a_bad_slot_field() {
        let other = "1 127.0.0.1:7000@17000 master - 0 0 1 connected 0-5460\n";
        assert_eq!(parse_myself_line(""), Err(MyselfLineSeen::NoLine));
        assert_eq!(parse_myself_line(other), Err(MyselfLineSeen::NoLine));
        assert_eq!(
            parse_myself_line("1 127.0.0.1:7000@17000 myself,master - 0\n"),
            Err(MyselfLineSeen::ShortLine(
                "1 127.0.0.1:7000@17000 myself,master - 0".to_owned()
            ))
        );
        for token in ["abc", "5-3", "16384", "0-16384", "-1", "+5", "1-"] {
            let line =
                format!("1 127.0.0.1:7000@17000 myself,master - 0 0 1 connected 0-10 {token}\n");
            assert_eq!(
                parse_myself_line(&line),
                Err(MyselfLineSeen::BadToken(token.to_owned())),
                "slot field {token:?}"
            );
        }
    }

    /// The ownership decision compares whole slot sets: the same slots split
    /// differently (the map's `0-100`, `101-200` against the line's `0-200`)
    /// is an owner; a slot missing, a slot extra and a disjoint set are not.
    #[test]
    fn serves_exactly_compares_whole_slot_sets_however_they_are_split() {
        let expected = SlotSet::from_ranges([(0, 100), (101, 200)]);
        let owned = |text: &str| {
            parse_myself_line(&format!(
                "1 127.0.0.1:7000@17000 myself,master - 0 0 1 connected {text}\n"
            ))
            .expect("usable")
            .owned
        };
        assert!(serves_exactly(&expected, &owned("0-200")));
        assert!(serves_exactly(&expected, &owned("0-50 51-200")));
        assert!(
            !serves_exactly(&expected, &owned("0-199")),
            "one slot missing"
        );
        assert!(
            !serves_exactly(&expected, &owned("0-201")),
            "one slot extra"
        );
        assert!(
            !serves_exactly(&expected, &owned("0-200 300")),
            "an extra slot"
        );
        assert!(!serves_exactly(&expected, &owned("300-400")), "disjoint");
    }

    /// `SlotSet` merges adjacent and overlapping ranges, in any order, and
    /// prints single slots bare, ranges `A-B` and the empty set `none`.
    #[test]
    fn slot_sets_merge_their_ranges_and_print_them_as_the_errors_name_them() {
        let set = SlotSet::from_ranges([(10, 12), (5, 5), (0, 3), (4, 4), (11, 20), (30, 30)]);
        assert_eq!(set.ranges(), &[(0, 5), (10, 20), (30, 30)]);
        assert_eq!(set.to_string(), "0-5,10-20,30");
        assert_eq!(SlotSet::default().to_string(), "none");
    }

    /// `NotOwner` names the pattern, the primary, the node, both slot sets,
    /// the fingerprint that found it and the routing refresh's outcome; with
    /// no node to ask it says the routing table named none.
    #[test]
    fn not_owner_names_the_primary_the_node_both_slot_sets_and_the_refresh() {
        let found = ServingChange::NotOwner {
            target: Arc::from("le:{*}:obj:hdr"),
            primary: "127.0.0.1:7001".to_owned(),
            node: Some("127.0.0.1:7000".to_owned()),
            expected: SlotSet::from_ranges([(0, 0), (5461, 10922)]),
            served: Some(SlotSet::from_ranges([(1, 5460)])),
            at: WalkEnd::Start,
            refresh: RoutingRefresh::Refreshed,
        }
        .to_string();
        for part in [
            "SCAN walk of le:{*}:obj:hdr",
            "primary 127.0.0.1:7001 (slots 0,5461-10922)",
            "to 127.0.0.1:7000, which serves slots 1-5460",
            "at the walk's start",
            "it refreshed its routing table",
        ] {
            assert!(found.contains(part), "{found:?} names {part:?}");
        }
        let unrouted = ServingChange::NotOwner {
            target: Arc::from("p"),
            primary: "127.0.0.1:7002".to_owned(),
            node: None,
            expected: SlotSet::from_ranges([(10923, 16383)]),
            served: None,
            at: WalkEnd::End,
            refresh: RoutingRefresh::TimedOut { waited_ms: 5000 },
        }
        .to_string();
        for part in [
            "primary 127.0.0.1:7002 (slots 10923-16383)",
            "names no node for it at the walk's end",
            "did not finish within 5000 ms",
        ] {
            assert!(unrouted.contains(part), "{unrouted:?} names {part:?}");
        }
    }

    // ── A coverage walk's verdict on a pinned node ─────────────────────────

    /// The slots the coverage-verdict cases give the primary.
    fn first_third() -> SlotSet {
        SlotSet::from_ranges([(0, 5460)])
    }

    /// A `CLUSTER NODES` reply whose myself line lists `slots` (as printed),
    /// with another node's line before it.
    fn nodes_with_myself(slots: &str) -> String {
        format!(
            "1 127.0.0.1:7001@17001 master - 0 0 2 connected 5461-10922\n\
             2 127.0.0.1:7000@17000 myself,master - 0 0 1 connected {slots}\n"
        )
    }

    /// The verdict-to-action table of `coverage_verdict` (module docs,
    /// *Coverage walks*). Only an exact owner is walked with nothing
    /// recorded. An open slot, an unusable myself line (none, a short one, a
    /// bad slot field) and a failed read are each walked and recorded
    /// unproven: a coverage walk never stops before walking over a check it
    /// could not prove. A node serving other slots (one slot fewer, one
    /// extra, a disjoint set) and a replica (no slot) are not walked: a
    /// proven non-owner's keys are another primary's. Only an unusable myself
    /// line and a failed read leave the pin's ownership unproven by the node
    /// (`leaves_ownership_unproven`), so only they make the walk check its
    /// routing table against its map; an open slot's line listed exactly the
    /// primary's slots, and a `NotOwner` is proven wrong and re-syncs itself.
    ///
    /// Fail-on-old, by in-place perturbation: with `leaves_ownership_unproven`
    /// answering `false` for `ReadFailed`, the failed-read row goes red.
    #[test]
    fn coverage_verdict_walks_every_primary_but_a_proven_non_owner() {
        let expected = first_third();
        let marker = "[42->-0000000000000000000000000000000000000002]";
        let open = format!("0-5460 {marker}");
        let replica = "3 127.0.0.1:7003@17003 myself,slave 2 0 0 1 connected\n";
        let short = "2 127.0.0.1:7000@17000 myself,master - 0\n";
        let bad = nodes_with_myself("0-5460 abc");
        let cases: [(&str, Option<String>, CoverageVerdict); 9] = [
            (
                "an exact owner",
                Some(nodes_with_myself("0-5460")),
                CoverageVerdict::Owner,
            ),
            (
                "an exact owner, split otherwise",
                Some(nodes_with_myself("0-100 101-5460")),
                CoverageVerdict::Owner,
            ),
            (
                "an open slot on the owner",
                Some(nodes_with_myself(&open)),
                CoverageVerdict::OpenSlot {
                    slots: marker.to_owned(),
                },
            ),
            (
                "no myself line",
                Some("1 127.0.0.1:7001@17001 master - 0 0 2 connected 0-5460\n".to_owned()),
                CoverageVerdict::Unusable(MyselfLineSeen::NoLine),
            ),
            (
                "a short myself line",
                Some(short.to_owned()),
                CoverageVerdict::Unusable(MyselfLineSeen::ShortLine(short.trim_end().to_owned())),
            ),
            (
                "a bad slot field",
                Some(bad),
                CoverageVerdict::Unusable(MyselfLineSeen::BadToken("abc".to_owned())),
            ),
            ("a failed read", None, CoverageVerdict::ReadFailed),
            (
                "one slot fewer",
                Some(nodes_with_myself("0-5459")),
                CoverageVerdict::NotOwner {
                    served: SlotSet::from_ranges([(0, 5459)]),
                },
            ),
            (
                "a replica",
                Some(replica.to_owned()),
                CoverageVerdict::NotOwner {
                    served: SlotSet::default(),
                },
            ),
        ];
        for (case, nodes, verdict) in cases {
            assert_eq!(
                coverage_verdict(&expected, nodes.as_deref()),
                verdict,
                "{case}"
            );
            let (walks, proven, leaves_unproven) = match &verdict {
                CoverageVerdict::Owner => (true, true, false),
                CoverageVerdict::OpenSlot { .. } => (true, false, false),
                CoverageVerdict::Unusable(_) | CoverageVerdict::ReadFailed => (true, false, true),
                CoverageVerdict::NotOwner { .. } => (false, false, false),
            };
            assert_eq!(
                (
                    verdict.walks(),
                    verdict.proven(),
                    verdict.leaves_ownership_unproven()
                ),
                (walks, proven, leaves_unproven),
                "{case}: (walked, proven, ownership left unproven)"
            );
        }
        for other in ["0-5461", "5461-10922", "0-5460 6000"] {
            let verdict = coverage_verdict(&expected, Some(&nodes_with_myself(other)));
            assert!(
                matches!(verdict, CoverageVerdict::NotOwner { .. }) && !verdict.walks(),
                "slots {other}: a proven non-owner is not walked, got {verdict:?}"
            );
        }
    }

    /// `NotOwner` outranks an open slot: a node whose myself line lists other
    /// slots than the primary's, and also shows a slot migrating, is a
    /// non-owner, not walked. Walking it would count the primary walked when
    /// its keys were never read.
    #[test]
    fn coverage_verdict_ranks_not_owner_above_an_open_slot() {
        let marker = "[5460->-0000000000000000000000000000000000000002]";
        let verdict = coverage_verdict(
            &first_third(),
            Some(&nodes_with_myself(&format!("0-5459 {marker}"))),
        );
        assert_eq!(
            verdict,
            CoverageVerdict::NotOwner {
                served: SlotSet::from_ranges([(0, 5459)]),
            }
        );
        assert!(!verdict.walks());
    }

    // ── A coverage walk's end ──────────────────────────────────────────────

    /// A coverage walk with no primary left to step and no slot map (the
    /// standalone shape: no end check reads anything), carrying the reason
    /// its checks recorded and whether one of them lost coverage
    /// (`coverage_lost`, set by a `NotOwner`, a `SlotMap` or a
    /// `StaleTable`), with every pin's ownership proven and no re-sync yet
    /// (a case sets `ownership_unproven`, `resynced` or `client` itself). Its
    /// client is never connected, so nothing a case sends is answered: a
    /// re-sync (`refresh_routing`) on it waits forever, which the end
    /// re-sync tests send on purpose and bound, and the tests that call
    /// `check_table` or `coverage_not_owner` bound to prove none is sent.
    fn coverage_walk_at_its_end(recorded: Option<WalkError>, coverage_lost: bool) -> KeyspaceWalk {
        let pattern: Arc<str> = Arc::from("le:{*}:obj:hdr");
        KeyspaceWalk {
            client: Client::default(),
            faults: WalkFaults::for_target(&pattern),
            pattern,
            guard: KeyspaceGuard::Coverage,
            map: None,
            primaries: Vec::new(),
            current: 0,
            finished: false,
            unproven: recorded,
            coverage_lost,
            ownership_unproven: false,
            resynced: false,
            reached_end: false,
        }
    }

    fn open_slot_reason() -> WalkError {
        WalkError::changed(ServingChange::OpenSlot {
            target: Arc::from("le:{*}:obj:hdr"),
            node: "127.0.0.1:7000".to_owned(),
            slots: "[42->-x]".to_owned(),
        })
    }

    fn unusable_line_reason() -> WalkError {
        WalkError::WitnessMissing(WitnessMissing {
            walk: "SCAN",
            target: Arc::from("le:{*}:obj:hdr"),
            node: Some("127.0.0.1:7000".to_owned()),
            seen: WitnessSeen::MyselfLine(MyselfLineSeen::NoLine),
        })
    }

    fn failed_read_reason() -> WalkError {
        WalkError::Read {
            op: "CLUSTER NODES",
            source: fred::error::Error::new(fred::error::ErrorKind::IO, "connection reset"),
        }
    }

    fn not_owner_reason() -> WalkError {
        WalkError::changed(ServingChange::NotOwner {
            target: Arc::from("le:{*}:obj:hdr"),
            primary: "127.0.0.1:7001".to_owned(),
            node: Some("127.0.0.1:7000".to_owned()),
            expected: SlotSet::from_ranges([(5461, 10922)]),
            served: Some(first_third()),
            at: WalkEnd::Start,
            refresh: RoutingRefresh::Refreshed,
        })
    }

    fn slot_map_reason() -> WalkError {
        WalkError::changed(ServingChange::SlotMap {
            target: Arc::from("le:{*}:obj:hdr"),
            before: "0-16383@127.0.0.1:7000".to_owned(),
            after: "0-16383@127.0.0.1:7001".to_owned(),
        })
    }

    fn stale_table_reason() -> WalkError {
        WalkError::changed(ServingChange::StaleTable {
            target: Arc::from("le:{*}:obj:hdr"),
            at: WalkEnd::Start,
            map: "0-0@127.0.0.1:7001,1-16383@127.0.0.1:7000".to_owned(),
            table: "0-16383@127.0.0.1:7000".to_owned(),
            refresh: RoutingRefresh::Refreshed,
        })
    }

    /// Which reason a walk answered, by its kind, for comparing.
    fn reason_kind(e: &WalkError) -> &'static str {
        match e {
            WalkError::Changed(change) => match **change {
                ServingChange::OpenSlot { .. } => "OpenSlot",
                ServingChange::NotOwner { .. } => "NotOwner",
                ServingChange::SlotMap { .. } => "SlotMap",
                ServingChange::StaleTable { .. } => "StaleTable",
                ServingChange::Process { .. } => "Process",
            },
            WalkError::WitnessMissing(_) => "WitnessMissing",
            WalkError::Read { .. } => "Read",
            WalkError::Step { .. } => "Step",
            WalkError::Incomplete(_) => "Incomplete",
        }
    }

    /// A coverage walk answers the reason it recorded once, after its last
    /// key (then `None`), and counts as having walked every primary
    /// (`walked_every_primary`, policy 3) only once it reached its end with
    /// coverage intact: an open slot, an unusable myself line or a failed
    /// `CLUSTER NODES` read leaves it `true`; a `NotOwner`, a changed slot
    /// map or an out-of-date routing table (`StaleTable`), each recorded
    /// with `coverage_lost`, makes it `false`. Before its end it is `false`
    /// whatever was recorded, so a walk a store fault ends early never reads
    /// as complete. A walk that recorded nothing completes.
    #[tokio::test]
    async fn a_coverage_walk_answers_its_reason_once_at_its_end_and_walked_every_primary_only_with_coverage_intact(
    ) {
        for (reason, coverage_lost, walked_every_primary) in [
            (open_slot_reason(), false, true),
            (unusable_line_reason(), false, true),
            (failed_read_reason(), false, true),
            (not_owner_reason(), true, false),
            (slot_map_reason(), true, false),
            (stale_table_reason(), true, false),
        ] {
            let kind = reason_kind(&reason);
            let mut walk = coverage_walk_at_its_end(Some(reason), coverage_lost);
            assert!(
                !walk.walked_every_primary(),
                "{kind}: before its end a walk never counts as complete"
            );
            match walk.next_page().await {
                Err(answered) => assert_eq!(reason_kind(&answered), kind),
                Ok(page) => panic!("{kind}: the recorded reason is answered, got {page:?}"),
            }
            assert_eq!(
                walk.walked_every_primary(),
                walked_every_primary,
                "{kind}: walked every primary"
            );
            assert!(
                matches!(walk.next_page().await, Ok(None)),
                "{kind}: the reason is answered once, then the walk is over"
            );
        }

        let mut clean = coverage_walk_at_its_end(None, false);
        assert!(matches!(clean.next_page().await, Ok(None)));
        assert!(
            clean.walked_every_primary(),
            "a clean walk walked every one"
        );
    }

    /// Which reason a coverage walk answers when its checks record two. A
    /// coverage-losing reason (`NotOwner`, `SlotMap`, `StaleTable`) replaces
    /// a recorded proof-only one (`OpenSlot`, `WitnessMissing`, a failed
    /// `CLUSTER NODES` read), so the answer says the walk skipped keys
    /// whenever it may have, and a proof-only one never replaces a
    /// coverage-losing one; among coverage-losing reasons, and among
    /// proof-only ones, the first wins. Each coverage-losing reason is
    /// recorded as the walk records it, with coverage marked lost first, so
    /// `walked_every_primary` agrees with the answer. The `NotOwner`s and
    /// `StaleTable`s go through `record` directly, since
    /// `coverage_not_owner` or `check_table` on a walk that has not
    /// re-synced would send a `sync_cluster` on this unconnected client.
    #[tokio::test]
    async fn a_coverage_walk_answers_a_coverage_losing_reason_over_a_proof_only_one_and_the_first_of_each_kind(
    ) {
        type Reason = fn() -> WalkError;
        let coverage_losing = |kind: &str| matches!(kind, "NotOwner" | "SlotMap" | "StaleTable");
        let cases: [(&str, Reason, Reason, &str); 12] = [
            (
                "a failed CLUSTER NODES read, then a changed map",
                failed_read_reason,
                slot_map_reason,
                "SlotMap",
            ),
            (
                "a NotOwner, then an open slot",
                not_owner_reason,
                open_slot_reason,
                "NotOwner",
            ),
            (
                "a NotOwner, then a changed map",
                not_owner_reason,
                slot_map_reason,
                "NotOwner",
            ),
            (
                "an open slot, then a changed map",
                open_slot_reason,
                slot_map_reason,
                "SlotMap",
            ),
            (
                "an unusable myself line, then a NotOwner",
                unusable_line_reason,
                not_owner_reason,
                "NotOwner",
            ),
            (
                "a failed CLUSTER NODES read, then a NotOwner",
                failed_read_reason,
                not_owner_reason,
                "NotOwner",
            ),
            (
                "a changed map, then a NotOwner",
                slot_map_reason,
                not_owner_reason,
                "SlotMap",
            ),
            (
                "an open slot, then an unusable myself line",
                open_slot_reason,
                unusable_line_reason,
                "OpenSlot",
            ),
            (
                "an unusable myself line, then an out-of-date table",
                unusable_line_reason,
                stale_table_reason,
                "StaleTable",
            ),
            (
                "an out-of-date table, then a NotOwner",
                stale_table_reason,
                not_owner_reason,
                "StaleTable",
            ),
            (
                "a NotOwner, then an out-of-date table",
                not_owner_reason,
                stale_table_reason,
                "NotOwner",
            ),
            (
                "a changed map, then an out-of-date table",
                slot_map_reason,
                stale_table_reason,
                "SlotMap",
            ),
        ];
        for (case, first, second, answered) in cases {
            let mut walk = coverage_walk_at_its_end(None, false);
            for reason in [first(), second()] {
                if coverage_losing(reason_kind(&reason)) {
                    walk.coverage_lost = true;
                }
                walk.record(reason);
            }
            match walk.next_page().await {
                Err(reason) => assert_eq!(reason_kind(&reason), answered, "{case}"),
                Ok(page) => panic!("{case}: a recorded reason is answered, got {page:?}"),
            }
            assert_eq!(
                walk.walked_every_primary(),
                !coverage_losing(answered),
                "{case}: walked every primary"
            );
        }
    }

    /// A `NotOwner` after the walk already re-synced its routing table (on an
    /// earlier `NotOwner`) marks coverage lost without re-syncing again or
    /// replacing the first `NotOwner`: one re-sync per walk.
    #[tokio::test]
    async fn a_coverage_walks_second_not_owner_marks_coverage_lost_without_a_second_resync() {
        let mut resynced = coverage_walk_at_its_end(Some(not_owner_reason()), true);
        resynced.resynced = true;
        let primary = MapPrimary {
            primary: "127.0.0.1:7002".to_owned(),
            pin_slot: 10923,
            slots: SlotSet::from_ranges([(10923, 16383)]),
        };
        // `client` is never connected: a second re-sync would be a
        // `sync_cluster` on it, which this call must not send. Such a sync
        // would queue forever (the client's command timeout is 0), so the
        // bound turns that regression into a failure, not a hang.
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            resynced.coverage_not_owner(&primary, None, None, WalkEnd::End),
        )
        .await
        .expect("a second NotOwner sends no sync_cluster");
        assert!(resynced.coverage_lost && resynced.resynced);
        match resynced.next_page().await {
            Err(WalkError::Changed(change)) => match *change {
                ServingChange::NotOwner { ref primary, .. } => {
                    assert_eq!(
                        primary, "127.0.0.1:7001",
                        "the first NotOwner, not the second"
                    );
                }
                other => panic!("the first NotOwner, got {other:?}"),
            },
            other => panic!("the first NotOwner is answered, got {other:?}"),
        }
        assert!(!resynced.walked_every_primary());
    }

    // ── A coverage walk's routing-table check ──────────────────────────────

    /// A three-primary slot map as a coverage walk reads it, its primaries
    /// named by address.
    fn three_primary_map() -> Vec<SlotOwner> {
        slot_map(&[
            (0, 5460, "10.0.0.1:6379"),
            (5461, 10922, "10.0.0.2:6379"),
            (10923, 16383, "10.0.0.3:6379"),
        ])
    }

    /// A routing table that divides the slots as [`three_primary_map`] does,
    /// under other names.
    fn table_matching_three() -> ClusterRouting {
        routing(&[(0, 5460, 7000), (5461, 10922, 7001), (10923, 16383, 7002)])
    }

    /// A routing table [`three_primary_map`] has moved on from: slot 5461 is
    /// still the first node's.
    fn table_behind_three() -> ClusterRouting {
        routing(&[(0, 5461, 7000), (5462, 10922, 7001), (10923, 16383, 7002)])
    }

    /// The bound on a `check_table` call on an unconnected client. For a
    /// call that must send nothing, a re-sync on the default client would
    /// queue forever (its command timeout is 0), so the bound turns that
    /// regression into a failure, not a hang. The WARN test's call does send
    /// one, on a client swapped for one with a 50 ms command timeout, and
    /// must finish well within it.
    const NO_RESYNC_BOUND: std::time::Duration = std::time::Duration::from_secs(5);

    /// The table check is made only by a walk with an unproven pin. A walk
    /// whose every pin was proven by its own myself line (here: none
    /// unproven, an open slot recorded) checks nothing, even against a table
    /// that differs from its map: it records nothing, keeps its coverage and
    /// sends no re-sync.
    ///
    /// Fail-on-old, by in-place perturbation: with the gate (`if
    /// !self.ownership_unproven { return; }`) removed from `check_table`, the
    /// call re-syncs on the unconnected client and the bound fails it.
    #[tokio::test]
    async fn a_coverage_walk_with_every_pin_proven_makes_no_table_check() {
        let map = three_primary_map();
        let behind = table_behind_three();
        assert!(
            !table_partition_matches(&map, Some(&behind)),
            "non-vacuity: the table differs from the map"
        );
        let mut walk = coverage_walk_at_its_end(Some(open_slot_reason()), false);
        tokio::time::timeout(
            NO_RESYNC_BOUND,
            walk.check_table(&map, Some(&behind), WalkEnd::Start),
        )
        .await
        .expect("a walk with every pin proven sends no sync_cluster");
        assert!(!walk.coverage_lost && !walk.resynced);
        match walk.next_page().await {
            Err(answered) => assert_eq!(reason_kind(&answered), "OpenSlot"),
            Ok(page) => panic!("the open slot is still the answer, got {page:?}"),
        }
        assert!(walk.walked_every_primary());
    }

    /// A walk with an unproven pin whose table divides the slots as its map
    /// does at its START records nothing, keeps its coverage and sends no
    /// re-sync (a matching table is re-synced only at the walk's end: see
    /// the next test): the pins came from a table the map agrees with, so
    /// the walk still counts as having walked every primary, answering its
    /// unusable myself line.
    ///
    /// Fail-on-old, by in-place perturbation: with `check_table` taking
    /// every table as differing, the call re-syncs on the unconnected client
    /// and the bound fails it.
    #[tokio::test]
    async fn a_coverage_walks_table_check_on_a_matching_table_records_nothing() {
        let map = three_primary_map();
        let matching = table_matching_three();
        let mut walk = coverage_walk_at_its_end(Some(unusable_line_reason()), false);
        walk.ownership_unproven = true;
        tokio::time::timeout(
            NO_RESYNC_BOUND,
            walk.check_table(&map, Some(&matching), WalkEnd::Start),
        )
        .await
        .expect("a matching table sends no sync_cluster");
        assert!(!walk.coverage_lost && !walk.resynced);
        match walk.next_page().await {
            Err(answered) => assert_eq!(reason_kind(&answered), "WitnessMissing"),
            Ok(page) => panic!("the unusable line is the answer, got {page:?}"),
        }
        assert!(walk.walked_every_primary());
    }

    /// A walk with an unproven pin whose table matches its map at its END
    /// still re-syncs the table (once per walk): the comparison cannot see a
    /// change of primary that keeps every slot set, and the fresh table
    /// serves the next walk. The re-sync is housekeeping: it records
    /// nothing and keeps the coverage, so the walk still answers its
    /// unusable myself line and counts as having walked every primary.
    ///
    /// On this unconnected client the re-sync queues forever (its command
    /// timeout is 0), so the call is bounded and its bound elapsing is the
    /// evidence that a sync was sent; `resynced` is marked before the sync
    /// is awaited, so it is visible after the bound drops the call.
    ///
    /// Fail-on-old, by in-place perturbation: with the end re-sync after a
    /// matching table removed from `check_table`, the call returns at once
    /// and `resynced` stays `false`.
    #[tokio::test]
    async fn a_coverage_walks_matching_table_at_its_end_is_re_synced_without_losing_coverage() {
        let map = three_primary_map();
        let matching = table_matching_three();
        let mut walk = coverage_walk_at_its_end(Some(unusable_line_reason()), false);
        walk.ownership_unproven = true;
        let synced = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            walk.check_table(&map, Some(&matching), WalkEnd::End),
        )
        .await;
        assert!(
            synced.is_err(),
            "the end check sent a sync_cluster, which waits on this unconnected client"
        );
        assert!(walk.resynced, "the end re-sync is marked, once per walk");
        assert!(!walk.coverage_lost, "a matching table keeps the coverage");
        match walk.next_page().await {
            Err(answered) => assert_eq!(reason_kind(&answered), "WitnessMissing"),
            Ok(page) => panic!("the unusable line is still the answer, got {page:?}"),
        }
        assert!(walk.walked_every_primary());
    }

    /// The end re-sync after a matching table is once per walk: a walk with
    /// an unproven pin that already re-synced (on a start `StaleTable`, say)
    /// sends no second sync at its end, and its answer and coverage stand.
    ///
    /// Fail-on-old, by in-place perturbation: with `&& !self.resynced`
    /// removed from the end branch of `check_table`, the call sends a second
    /// sync, which waits on this unconnected client, and the bound fails it.
    #[tokio::test]
    async fn a_coverage_walk_that_already_re_synced_sends_no_end_re_sync_for_a_matching_table() {
        let map = three_primary_map();
        let matching = table_matching_three();
        let mut walk = coverage_walk_at_its_end(Some(stale_table_reason()), true);
        walk.ownership_unproven = true;
        walk.resynced = true;
        tokio::time::timeout(
            NO_RESYNC_BOUND,
            walk.check_table(&map, Some(&matching), WalkEnd::End),
        )
        .await
        .expect("a walk that already re-synced sends no second sync_cluster");
        assert!(walk.resynced && walk.coverage_lost);
        match walk.next_page().await {
            Err(answered) => assert_eq!(reason_kind(&answered), "StaleTable"),
            Ok(page) => panic!("the start's StaleTable is still the answer, got {page:?}"),
        }
        assert!(!walk.walked_every_primary());
    }

    /// Every WARN `future` logs, as parsed JSON events, captured with
    /// `WithSubscriber` so the capture follows the future across `.await`s
    /// (the shape of `storage::orphan_sweep`'s tests' `capture_warns`).
    async fn capture_warns<F, T>(future: F) -> (T, Vec<serde_json::Value>)
    where
        F: std::future::Future<Output = T>,
    {
        use tracing::instrument::WithSubscriber;

        #[derive(Clone)]
        struct Capture(Arc<std::sync::Mutex<Vec<u8>>>);
        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
            type Writer = Capture;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }
        impl std::io::Write for Capture {
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

        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(Capture(Arc::clone(&buf)))
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

    /// An end re-sync after a matching table that does not complete logs the
    /// WARN operators are told to look for, naming the walk's pattern and
    /// how the refresh ended, and still records nothing: the walk keeps its
    /// coverage and its answer, and counts as re-synced (once per walk). The
    /// walk's client is unconnected with a 50 ms command timeout, so
    /// `refresh_routing` gives up (`TimedOut`, or `Failed` if the client
    /// refuses the sync outright; both log the WARN).
    ///
    /// Fail-on-old, by in-place perturbation: with the `tracing::warn!` call
    /// removed from the end branch of `check_table`, no such WARN is
    /// captured.
    #[tokio::test]
    async fn a_coverage_walks_end_re_sync_that_does_not_complete_logs_one_warn_and_keeps_coverage()
    {
        let map = three_primary_map();
        let matching = table_matching_three();
        let mut walk = coverage_walk_at_its_end(Some(unusable_line_reason()), false);
        walk.ownership_unproven = true;
        walk.client = Client::new(
            fred::types::config::Config::default(),
            Some(fred::types::config::PerformanceConfig {
                default_command_timeout: std::time::Duration::from_millis(50),
                ..Default::default()
            }),
            None,
            None,
        );
        let (bounded, warns) = capture_warns(tokio::time::timeout(
            NO_RESYNC_BOUND,
            walk.check_table(&map, Some(&matching), WalkEnd::End),
        ))
        .await;
        bounded.expect("the refresh gives up within the client's command timeout");
        let refresh_warns: Vec<&serde_json::Value> = warns
            .iter()
            .filter(|event| {
                event["fields"]["message"].as_str().is_some_and(|message| {
                    message.contains("end-of-walk routing-table refresh failed or timed out")
                })
            })
            .collect();
        assert_eq!(
            refresh_warns.len(),
            1,
            "one WARN for the refresh that did not complete; captured: {warns:?}"
        );
        let event = refresh_warns[0];
        assert_eq!(event["level"], "WARN");
        assert_eq!(event["fields"]["pattern"], "le:{*}:obj:hdr");
        assert!(
            event["fields"]["refresh"]
                .as_str()
                .is_some_and(|refresh| refresh.contains("could not refresh its routing table")),
            "the refresh field says how it ended: {event}"
        );
        assert!(walk.resynced, "the end re-sync is marked, once per walk");
        assert!(!walk.coverage_lost, "a matching table keeps the coverage");
        match walk.next_page().await {
            Err(answered) => assert_eq!(reason_kind(&answered), "WitnessMissing"),
            Ok(page) => panic!("the unusable line is still the answer, got {page:?}"),
        }
        assert!(walk.walked_every_primary());
    }

    /// A walk with an unproven pin that already re-synced (its start check
    /// found the table out of date) and finds the table out of date again at
    /// its end — the re-sync failed, say — marks coverage lost without a
    /// second re-sync and keeps answering its first reason, the start's
    /// `StaleTable`: one re-sync per walk. `coverage_lost` starts `false`
    /// here, unlike in a real walk, where the first `StaleTable` already set
    /// it, so the assertion sees this call's own marking.
    ///
    /// Fail-on-old, by in-place perturbation: with `coverage_lost = true`
    /// moved below the `if self.resynced { return; }`, `coverage_lost` stays
    /// `false` and the walk counts as having walked every primary.
    #[tokio::test]
    async fn a_coverage_walks_second_out_of_date_table_marks_coverage_lost_without_a_second_resync()
    {
        let map = three_primary_map();
        let behind = table_behind_three();
        let mut walk = coverage_walk_at_its_end(Some(stale_table_reason()), false);
        walk.ownership_unproven = true;
        walk.resynced = true;
        tokio::time::timeout(
            NO_RESYNC_BOUND,
            walk.check_table(&map, Some(&behind), WalkEnd::End),
        )
        .await
        .expect("a second out-of-date table sends no sync_cluster");
        assert!(walk.coverage_lost && walk.resynced);
        match walk.next_page().await {
            Err(WalkError::Changed(change)) => match *change {
                ServingChange::StaleTable { at, .. } => {
                    assert_eq!(at, WalkEnd::Start, "the first StaleTable, not the second");
                }
                other => panic!("the first StaleTable, got {other:?}"),
            },
            other => panic!("the first StaleTable is answered, got {other:?}"),
        }
        assert!(!walk.walked_every_primary());
    }

    /// `StaleTable` names the pattern, the end of the walk that found it,
    /// the map and the table it compared (each as `start-end@host:port` per
    /// range), that the client's view was out of date, and how the re-sync
    /// went. The table is rendered by `render_table`, which says `no routing
    /// table` for none or an empty one.
    #[test]
    fn stale_table_names_the_pattern_the_end_both_groupings_and_the_refresh() {
        let map = render_map(&three_primary_map());
        let table = render_table(Some(&table_behind_three()));
        assert_eq!(
            table,
            "0-5461@127.0.0.1:7000,5462-10922@127.0.0.1:7001,10923-16383@127.0.0.1:7002"
        );
        assert_eq!(render_table(None), "no routing table");
        assert_eq!(
            render_table(Some(&ClusterRouting::new())),
            "no routing table"
        );
        let found = ServingChange::StaleTable {
            target: Arc::from("le:{*}:obj:hdr"),
            at: WalkEnd::Start,
            map: map.clone(),
            table: table.clone(),
            refresh: RoutingRefresh::Refreshed,
        }
        .to_string();
        for part in [
            "SCAN walk of le:{*}:obj:hdr",
            "at the walk's start",
            &format!("routing table ({table})"),
            &format!("slot map ({map})"),
            "out of date",
            "it refreshed its routing table",
        ] {
            assert!(found.contains(part), "{found:?} names {part:?}");
        }
        let at_end = ServingChange::StaleTable {
            target: Arc::from("p"),
            at: WalkEnd::End,
            map,
            table: render_table(None),
            refresh: RoutingRefresh::TimedOut { waited_ms: 5000 },
        }
        .to_string();
        for part in [
            "at the walk's end",
            "(no routing table)",
            "out of date",
            "did not finish within 5000 ms",
        ] {
            assert!(at_end.contains(part), "{at_end:?} names {part:?}");
        }
    }

    // ── The routes' options ────────────────────────────────────────────────

    /// A keyspace walk's primary on a cluster sends every command to its one
    /// named node and no slot; a keyed walk routes by its key's slot and
    /// names no node; standalone, neither. A step carries one attempt; a
    /// witness or fingerprint read keeps fred's default.
    #[test]
    fn each_route_pins_its_commands_and_only_a_step_carries_one_attempt() {
        let server = Server::new("127.0.0.1", 7001);
        assert_eq!(
            Route::Node(server.clone()).options(true),
            Options {
                cluster_node: Some(server.clone()),
                max_attempts: Some(1),
                ..Default::default()
            }
        );
        assert_eq!(
            Route::Node(server.clone()).options(false),
            Options {
                cluster_node: Some(server.clone()),
                ..Default::default()
            }
        );
        assert_eq!(
            Route::Slot(42).options(true),
            Options {
                cluster_hash: Some(ClusterHash::Custom(42)),
                max_attempts: Some(1),
                ..Default::default()
            }
        );
        assert_eq!(
            Route::Slot(42).options(false),
            Options {
                cluster_hash: Some(ClusterHash::Custom(42)),
                ..Default::default()
            }
        );
        assert_eq!(
            Route::None.options(true),
            Options {
                max_attempts: Some(1),
                ..Default::default()
            }
        );
        assert_eq!(Route::None.options(false), Options::default());
    }

    /// A routing table built from a `CLUSTER SLOTS` reply: one primary per
    /// `(start, end, port)` on `127.0.0.1` ([`routing_on`]).
    fn routing(ranges: &[(i64, i64, i64)]) -> ClusterRouting {
        let on_localhost: Vec<(i64, i64, &str, i64)> = ranges
            .iter()
            .map(|&(start, end, port)| (start, end, "127.0.0.1", port))
            .collect();
        routing_on(&on_localhost)
    }

    /// A node pin is routable only while the table lists it as a primary;
    /// a table that dropped it, and no table at all, list nothing.
    #[test]
    fn a_node_pin_is_listed_only_while_the_routing_table_names_it_a_primary() {
        let table = routing(&[(0, 5460, 7000), (5461, 10922, 7001), (10923, 16383, 7002)]);
        assert!(pinned_node_listed(
            Some(&table),
            &Server::new("127.0.0.1", 7001)
        ));
        let dropped = routing(&[(0, 10922, 7000), (10923, 16383, 7002)]);
        assert!(!pinned_node_listed(
            Some(&dropped),
            &Server::new("127.0.0.1", 7001)
        ));
        assert!(!pinned_node_listed(
            Some(&table),
            &Server::new("127.0.0.1", 1)
        ));
        assert!(!pinned_node_listed(None, &Server::new("127.0.0.1", 7001)));
    }

    // ── The table check's comparison ───────────────────────────────────────

    /// A slot map as a coverage walk reads it: `(start, end, primary)` per
    /// range, sorted by slot as `parse_slot_owners` sorts it.
    fn slot_map(ranges: &[(u16, u16, &str)]) -> Vec<SlotOwner> {
        let mut map: Vec<SlotOwner> = ranges
            .iter()
            .map(|&(start, end, primary)| SlotOwner {
                start,
                end,
                primary: primary.to_owned(),
            })
            .collect();
        map.sort_by_key(|owner| owner.start);
        map
    }

    /// A routing table built from a `CLUSTER SLOTS` reply naming each
    /// range's primary by `(start, end, host, port)`: a client's cached
    /// table, spelled as the client spells it.
    fn routing_on(ranges: &[(i64, i64, &str, i64)]) -> ClusterRouting {
        let reply = Value::Array(
            ranges
                .iter()
                .map(|&(start, end, host, port)| {
                    Value::Array(vec![
                        Value::Integer(start),
                        Value::Integer(end),
                        Value::Array(vec![
                            Value::from(host),
                            Value::Integer(port),
                            Value::from(format!("{host}-{port}")),
                        ]),
                    ])
                })
                .collect(),
        );
        ClusterRouting::from_cluster_slots(reply, "127.0.0.1").expect("the reply parses")
    }

    /// `table_partition_matches` compares the table's ranges grouped by node
    /// with the map's grouped by primary, as slot sets, and nothing else.
    ///
    /// Equal: the same division under other host spellings (an address in
    /// the map, a DNS name in the table); Azure Managed Redis's shape, eight
    /// interleaved ranges on two nodes
    /// (`docs-internal/validation/amr-topology-20261013/captures.md`); one
    /// node's range split into two entries on one side only (either side);
    /// and two primaries whose names the table exchanges (a change the
    /// comparison cannot see, by design: the slot sets are unchanged).
    ///
    /// Unequal: a slot range moved between primaries, a primary added to the
    /// map, a primary removed from it (each still covering every slot on
    /// both sides), no table, and an empty table.
    ///
    /// Fail-on-old, by in-place perturbations of `table_partition_matches`:
    /// keyed by the node's name (`(host:port, slots)` pairs), the spelling
    /// and exchanged-names rows go red; comparing each side's ungrouped
    /// ranges, the split rows go red; comparing only the union of each
    /// side's slots, the moved, added and removed rows go red.
    #[test]
    fn table_partition_matches_compares_slot_sets_per_node_not_names_or_range_lists() {
        let amr_map = slot_map(&[
            (0, 2047, "10.0.0.4:8501"),
            (2048, 4095, "10.0.0.4:8500"),
            (4096, 6143, "10.0.0.4:8501"),
            (6144, 8191, "10.0.0.4:8500"),
            (8192, 10239, "10.0.0.4:8501"),
            (10240, 12287, "10.0.0.4:8500"),
            (12288, 14335, "10.0.0.4:8501"),
            (14336, 16383, "10.0.0.4:8500"),
        ]);
        let amr_table = routing_on(&[
            (0, 2047, "amr.example", 8501),
            (2048, 4095, "amr.example", 8500),
            (4096, 6143, "amr.example", 8501),
            (6144, 8191, "amr.example", 8500),
            (8192, 10239, "amr.example", 8501),
            (10240, 12287, "amr.example", 8500),
            (12288, 14335, "amr.example", 8501),
            (14336, 16383, "amr.example", 8500),
        ]);
        let empty = ClusterRouting::new();
        let cases: Vec<(&str, Vec<SlotOwner>, Option<ClusterRouting>, bool)> = vec![
            (
                "the same division under other host spellings",
                slot_map(&[(0, 8191, "10.0.0.4:8501"), (8192, 16383, "10.0.0.5:8500")]),
                Some(routing_on(&[
                    (0, 8191, "node-1.example", 8501),
                    (8192, 16383, "node-2.example", 8500),
                ])),
                true,
            ),
            (
                "AMR's eight interleaved ranges on two nodes",
                amr_map,
                Some(amr_table),
                true,
            ),
            (
                "a primary's range split in two in the map only",
                slot_map(&[
                    (0, 100, "10.0.0.1:6379"),
                    (101, 5460, "10.0.0.1:6379"),
                    (5461, 16383, "10.0.0.2:6379"),
                ]),
                Some(routing(&[(0, 5460, 7000), (5461, 16383, 7001)])),
                true,
            ),
            (
                "a node's range split in two in the table only",
                slot_map(&[(0, 5460, "10.0.0.1:6379"), (5461, 16383, "10.0.0.2:6379")]),
                Some(routing(&[
                    (0, 100, 7000),
                    (101, 5460, 7000),
                    (5461, 16383, 7001),
                ])),
                true,
            ),
            (
                "two primaries' names exchanged",
                slot_map(&[(0, 8191, "127.0.0.1:7000"), (8192, 16383, "127.0.0.1:7001")]),
                Some(routing(&[(0, 8191, 7001), (8192, 16383, 7000)])),
                true,
            ),
            (
                "a slot range moved between primaries",
                three_primary_map(),
                Some(table_behind_three()),
                false,
            ),
            (
                "a primary added to the map",
                slot_map(&[
                    (0, 5460, "10.0.0.1:6379"),
                    (5461, 10922, "10.0.0.2:6379"),
                    (10923, 13652, "10.0.0.3:6379"),
                    (13653, 16383, "10.0.0.4:6379"),
                ]),
                Some(table_matching_three()),
                false,
            ),
            (
                "a primary removed from the map",
                slot_map(&[(0, 5460, "10.0.0.1:6379"), (5461, 16383, "10.0.0.2:6379")]),
                Some(table_matching_three()),
                false,
            ),
            ("no table", three_primary_map(), None, false),
            ("an empty table", three_primary_map(), Some(empty), false),
        ];
        for (case, map, table, matches) in cases {
            assert_eq!(
                table_partition_matches(&map, table.as_ref()),
                matches,
                "{case}"
            );
        }
    }

    // ── R1-a: the verdict of an R1 retry's witness re-read ────────────────

    /// After a failed step attempt the witness is read again
    /// (`after_failed_attempt`): unchanged, the step is sent again; another
    /// usable run_id ends the walk with the change, naming both; a missing
    /// witness ends it as it is; an `INFO server` that returned an error
    /// only spends the attempt; and a refused node pin
    /// (`route_refused`) ends the walk at once with its `Step`, a `Routing`
    /// error naming the node, never spending an attempt.
    ///
    /// Fail-on-old (in-place perturbation, red then green once restored):
    /// with the verdict's positive `Read` arm turned back into the match it
    /// replaced — a missing witness returned, then the catch-all
    /// `Err(_) => Ok(())` — the refusal returns `Ok(())`.
    #[test]
    fn after_failed_attempt_retries_only_an_unchanged_witness_or_a_failed_info() {
        let walk = "SCAN";
        let target: Arc<str> = Arc::from("le:{*}:obj:hdr");
        let node = Server::new("127.0.0.1", 7001);
        let witness = |run_id: &str| Witness {
            run_id: run_id.to_owned(),
            node: Some(node.to_string()),
        };
        let start = witness(RUN_A);

        assert!(
            after_failed_attempt(walk, &target, &start, Ok(witness(RUN_A))).is_ok(),
            "an unchanged witness: the step is sent again"
        );

        match after_failed_attempt(walk, &target, &start, Ok(witness(RUN_B))) {
            Err(WalkError::Changed(change)) => match *change {
                ServingChange::Process {
                    walk: w,
                    ref target,
                    ref node,
                    ref before,
                    ref after,
                } => {
                    assert_eq!(
                        (
                            w,
                            &**target,
                            node.as_deref(),
                            before.as_str(),
                            after.as_str()
                        ),
                        (
                            "SCAN",
                            "le:{*}:obj:hdr",
                            Some("127.0.0.1:7001"),
                            RUN_A,
                            RUN_B
                        )
                    );
                    let text = change.to_string();
                    assert!(text.contains(RUN_A) && text.contains(RUN_B), "{text}");
                }
                other => panic!("another run_id is a change of process, got {other:?}"),
            },
            other => panic!("another run_id ends the walk, got {other:?}"),
        }

        let missing = WalkError::WitnessMissing(WitnessMissing {
            walk,
            target: Arc::clone(&target),
            node: Some(node.to_string()),
            seen: WitnessSeen::RunId(RunIdSeen::Value("0".to_owned())),
        });
        assert!(
            matches!(
                after_failed_attempt(walk, &target, &start, Err(missing)),
                Err(WalkError::WitnessMissing(WitnessMissing {
                    seen: WitnessSeen::RunId(RunIdSeen::Value(ref v)),
                    ..
                })) if v == "0"
            ),
            "a missing witness ends the walk as it is"
        );

        let failed_info = WalkError::Read {
            op: "INFO server",
            source: fred::error::Error::new(fred::error::ErrorKind::IO, "connection reset"),
        };
        assert!(
            after_failed_attempt(walk, &target, &start, Err(failed_info)).is_ok(),
            "an INFO server that failed spends the attempt"
        );

        match after_failed_attempt(
            walk,
            &target,
            &start,
            Err(route_refused(walk, &target, &node)),
        ) {
            Err(WalkError::Step {
                walk: w,
                target: t,
                source,
            }) => {
                assert_eq!((w, &*t), ("SCAN", "le:{*}:obj:hdr"));
                assert_eq!(*source.kind(), fred::error::ErrorKind::Routing);
                assert!(
                    source.details().contains("pinned node 127.0.0.1:7001"),
                    "the refusal names the node: {source}"
                );
            }
            other => panic!("a refused node pin ends the walk with its Step, got {other:?}"),
        }
    }

    // ── A step's pipeline: reading its replies (`read_step_pipeline`) ──────

    /// The key every pipeline-reading test reads for. No test arms a fault
    /// on it, so its walk faults are none and every witness reply is read
    /// as given.
    const UNARMED_TARGET: &str = "le:{unarmed}:pipeline-reading";

    /// A step's pipeline with the start witness leading it: a keyed walk's
    /// first step, `[INFO server; step; INFO server]`.
    const LEAD: (bool, bool) = (true, true);

    /// A step's pipeline with only its after-step witness: a later keyed
    /// step, or a guarded keyspace step, `[step; INFO server]`.
    const AFTER: (bool, bool) = (false, true);

    /// A coverage walk's step, alone.
    const ALONE: (bool, bool) = (false, false);

    /// One reply of a pipeline (`try_all`'s element).
    type Reply = Result<Value, fred::error::Error>;

    /// A usable `INFO server` reply reporting `run_id`.
    fn info(run_id: &str) -> Reply {
        Ok(Value::from(valkey_info(Some(run_id))))
    }

    /// A scan step's reply: the next cursor `next` and `members`.
    fn scan(next: &str, members: &[&str]) -> Reply {
        Ok(Value::Array(vec![
            Value::from(next),
            Value::Array(members.iter().map(|m| Value::from(*m)).collect()),
        ]))
    }

    /// A transient error ([`step_error_is_transient`]): a lost connection.
    fn transient() -> fred::error::Error {
        fred::error::Error::new(fred::error::ErrorKind::IO, "connection reset")
    }

    /// A deterministic error, as fred reads a `-WRONGTYPE` reply.
    fn wrongtype() -> fred::error::Error {
        fred::error::Error::new(
            fred::error::ErrorKind::InvalidArgument,
            "WRONGTYPE Operation against a key holding the wrong kind of value",
        )
    }

    /// An ACL refusal, as fred reads a `-NOPERM` reply: deterministic.
    fn noperm() -> fred::error::Error {
        fred::error::Error::new(
            fred::error::ErrorKind::Unknown,
            "NOPERM User has no permissions to run the 'info' command",
        )
    }

    /// [`read_step_pipeline`] of a `ZSCAN` walk of `target` in `mode` (a
    /// `(lead, after)` pair) over `results`, with `faults`.
    fn read_pipeline_with(
        target: &str,
        faults: &WalkFaults,
        mode: (bool, bool),
        results: Vec<Reply>,
    ) -> Result<StepAttempt, WalkError> {
        let (lead, after) = mode;
        read_step_pipeline(
            "ZSCAN",
            &Arc::from(target),
            lead,
            after,
            None,
            faults,
            results,
        )
    }

    /// [`read_pipeline_with`] on [`UNARMED_TARGET`].
    fn read_pipeline(mode: (bool, bool), results: Vec<Reply>) -> Result<StepAttempt, WalkError> {
        read_pipeline_with(
            UNARMED_TARGET,
            &WalkFaults::for_target(UNARMED_TARGET),
            mode,
            results,
        )
    }

    /// The landed attempt's parts, `(lead run_id, next, elements, witness
    /// run_id)`; panics, naming `case`, on anything else.
    fn landed(
        case: &str,
        read: Result<StepAttempt, WalkError>,
    ) -> (Option<String>, String, usize, Option<String>) {
        match read {
            Ok(StepAttempt::Landed {
                lead,
                next,
                elements,
                witness,
            }) => (
                lead.map(|w| w.run_id),
                next,
                elements.len(),
                witness.map(|w| w.run_id),
            ),
            other => panic!("{case}: the attempt landed, got {other:?}"),
        }
    }

    /// The failed attempt's lead run_id and error kind; panics, naming
    /// `case`, on anything else.
    fn failed(
        case: &str,
        read: Result<StepAttempt, WalkError>,
    ) -> (Option<String>, fred::error::ErrorKind) {
        match read {
            Ok(StepAttempt::Failed { lead, error }) => {
                (lead.map(|w| w.run_id), error.kind().clone())
            }
            other => panic!("{case}: a failed attempt, got {other:?}"),
        }
    }

    /// The `Step` error's source; panics, naming `case`, on anything else.
    fn step_error(case: &str, read: Result<StepAttempt, WalkError>) -> fred::error::Error {
        match read {
            Err(WalkError::Step {
                walk,
                target,
                source,
            }) => {
                assert_eq!((walk, &*target), ("ZSCAN", UNARMED_TARGET), "{case}");
                source
            }
            other => panic!("{case}: a Step error, got {other:?}"),
        }
    }

    /// One reply per command is read in command order: the lead witness
    /// (index 0, [`RUN_A`]) becomes the attempt's lead, the step's reply its
    /// cursor and elements, the after-step witness (index 2, [`RUN_B`]) its
    /// witness. A coverage step alone
    /// carries neither witness, and its one reply is the step's. Each
    /// witness actually read goes through the seam, the lead first: with a
    /// change forged at read 1, the lead reads [`RUN_A`] and the after-step
    /// witness [`RUN_B`].
    ///
    /// Fail-on-old (in-place perturbation of `read_step_pipeline`, red then
    /// green once restored): with a landed lead attempt's two witnesses
    /// exchanged, the lead row reads [`RUN_B`] as its lead and [`RUN_A`] as
    /// its after-step witness.
    #[test]
    fn read_step_pipeline_reads_one_reply_per_command_in_command_order() {
        assert_eq!(
            landed(
                "lead",
                read_pipeline(
                    LEAD,
                    vec![info(RUN_A), scan("17", &["m0", "0"]), info(RUN_B)]
                )
            ),
            (
                Some(RUN_A.to_owned()),
                "17".to_owned(),
                2,
                Some(RUN_B.to_owned())
            ),
            "lead: the lead is index 0's run_id, the after-step witness index 2's"
        );
        assert_eq!(
            landed(
                "after",
                read_pipeline(AFTER, vec![scan("0", &["m0", "0"]), info(RUN_B)])
            ),
            (None, "0".to_owned(), 2, Some(RUN_B.to_owned()))
        );
        assert_eq!(
            landed("alone", read_pipeline(ALONE, vec![scan("5", &["k"])])),
            (None, "5".to_owned(), 1, None)
        );

        let target = "le:{pipeline-reading-order}:obj:flds";
        let forged = forge_process_change_for_test(target, 1);
        let faults = WalkFaults::for_target(target);
        drop(forged);
        // Two real run_ids, so the forged ones show which read the seam
        // replaced, not which reply was read.
        let (real_lead, real_after) = (
            "3167b2266793dfa979ea481e11ad7fda032c542c",
            "4278c3377804e0ba08afb592f22be8eb143d653d",
        );
        let read = read_pipeline_with(
            target,
            &faults,
            LEAD,
            vec![info(real_lead), scan("0", &[]), info(real_after)],
        );
        match read {
            Ok(StepAttempt::Landed {
                lead: Some(lead),
                witness: Some(witness),
                ..
            }) => assert_eq!(
                (lead.run_id(), witness.run_id()),
                (RUN_A, RUN_B),
                "the seam reads the lead as read 0 and the after-step witness as read 1"
            ),
            other => panic!("the forged pipeline lands, got {other:?}"),
        }
    }

    /// One error for a pipeline of two or three commands is the pipeline
    /// failed as a whole (a lost connection, a redirect with no attempt
    /// left, a timeout): a failed attempt with no lead when the error is
    /// transient, so R1 decides; the `Step` error carrying it when it is
    /// not. For both pipeline shapes.
    #[test]
    fn read_step_pipeline_reads_one_error_as_the_whole_pipelines_failed_attempt() {
        for (shape, mode) in [("lead", LEAD), ("after", AFTER)] {
            assert_eq!(
                failed(shape, read_pipeline(mode, vec![Err(transient())])),
                (None, fred::error::ErrorKind::IO),
                "{shape}: a transient whole failure is a failed attempt with no lead"
            );
            let source = step_error(shape, read_pipeline(mode, vec![Err(wrongtype())]));
            assert!(
                source.details().contains("WRONGTYPE"),
                "{shape}: a deterministic whole failure is the Step error carrying it: {source}"
            );
        }
    }

    /// Decider's clarification 1 (b.qgz): every other count of replies than
    /// one per command fails the walk closed, a `Protocol` `Step` error
    /// naming both counts — none, too many, two errors for three commands,
    /// and the one `Ok` reply fred answers for a single frame that is not an
    /// array (`vendor/fred-10.1.0/src/clients/pipeline.rs` `try_send_all`).
    /// For both pipeline shapes, that one `Ok` is never read as a witness
    /// or as the step's reply: an `INFO server`-shaped one, a scan-shaped
    /// one and an integer all fail the same way, and the seam reads no
    /// witness for it (its read 0, forged as no run_id, is still unread
    /// afterwards).
    ///
    /// Fail-on-old (in-place perturbation of `read_step_pipeline`, red then
    /// green once restored): with the count check answering only the
    /// one-error case (any other count read on, reply by reply), every row
    /// fails. Among them, the lead pipeline's `INFO server`-shaped and
    /// integer single `Ok`s are read as its lead witness (the seam's read 0
    /// spent, a missing witness), the after pipeline's scan-shaped one as the
    /// step's reply (a `Read` error for its missing witness), and a reply too
    /// many is read on to a witness.
    #[test]
    fn read_step_pipeline_fails_closed_on_any_other_count_of_replies_than_its_commands() {
        let target = "le:{pipeline-reading-count}:obj:flds";
        // Every row checked, then every wrong one named.
        let mut wrong = Vec::new();
        for (shape, mode, sent) in [("lead", LEAD, 3), ("after", AFTER, 2)] {
            let mut shapes: Vec<(&str, Vec<Reply>)> = vec![
                ("no reply", Vec::new()),
                ("one reply too many", {
                    let mut replies = vec![scan("0", &[]), info(RUN_A), info(RUN_A)];
                    if mode == LEAD {
                        replies.insert(0, info(RUN_A));
                    }
                    replies
                }),
                ("one INFO-shaped Ok", vec![info(RUN_A)]),
                ("one scan-shaped Ok", vec![scan("0", &["m0", "0"])]),
                ("one integer Ok", vec![Ok(Value::Integer(1))]),
            ];
            if mode == LEAD {
                shapes.push(("two errors", vec![Err(transient()), Err(transient())]));
                shapes.push(("two replies", vec![info(RUN_A), scan("0", &[])]));
            }
            for (case, replies) in shapes {
                let answered = replies.len();
                let armed = arm_walk_fault_for_test(
                    target,
                    WalkFault::RunId {
                        read: 0,
                        run_id: None,
                    },
                );
                let faults = WalkFaults::for_target(target);
                drop(armed);
                let read = read_pipeline_with(target, &faults, mode, replies);
                let refusal = format!("answered {answered} replies to its {sent} commands");
                match read {
                    Err(WalkError::Step { walk, source, .. })
                        if walk == "ZSCAN"
                            && *source.kind() == fred::error::ErrorKind::Protocol
                            && source.details().contains(&refusal) => {}
                    other => wrong.push(format!(
                        "{shape}, {case}: not a Protocol Step error naming both counts: {other:?}"
                    )),
                }
                if faults.witness_reply(valkey_info(Some(RUN_A))) != "# Server\r\n" {
                    wrong.push(format!(
                        "{shape}, {case}: a reply was read as a witness (the seam's read 0 is spent)"
                    ));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "every count other than one reply per command fails closed, unread:\n{}",
            wrong.join("\n")
        );
    }

    /// With one reply per command, each reply is judged as its own
    /// command's, in command order: a lead witness that answered an error
    /// is a `Read` error of `INFO server` and one with no run_id a missing
    /// witness; the step's deterministic error is its `Step` error and its
    /// transient one a failed attempt keeping the lead that answered, in
    /// both cases with the after-step witness unread (here an error reply,
    /// which would otherwise end the walk as a `Read`); a step reply that
    /// is not a scan reply is a `Protocol` `Step` error; after a step that
    /// answered, an after-step witness that answered an error is a `Read`
    /// error and one with no run_id a missing witness.
    #[test]
    fn read_step_pipeline_judges_each_reply_as_its_own_commands() {
        let no_run_id = || Ok(Value::from(valkey_info(None)));
        let is_read = |case: &str, read: Result<StepAttempt, WalkError>| match read {
            Err(WalkError::Read { op, source }) => {
                assert_eq!(op, "INFO server", "{case}");
                assert!(source.details().contains("NOPERM"), "{case}: {source}");
            }
            other => panic!("{case}: a Read error of INFO server, got {other:?}"),
        };
        let is_missing = |case: &str, read: Result<StepAttempt, WalkError>| match read {
            Err(WalkError::WitnessMissing(missing)) => assert_eq!(
                missing.seen,
                WitnessSeen::RunId(RunIdSeen::NoLine),
                "{case}"
            ),
            other => panic!("{case}: a missing witness, got {other:?}"),
        };

        is_read(
            "lead error",
            read_pipeline(LEAD, vec![Err(noperm()), scan("0", &[]), info(RUN_A)]),
        );
        is_missing(
            "lead with no run_id",
            read_pipeline(LEAD, vec![no_run_id(), scan("0", &[]), info(RUN_A)]),
        );

        for (shape, mode) in [("lead", LEAD), ("after", AFTER)] {
            let with_step = |step: Reply, after: Reply| {
                let mut replies = vec![step, after];
                if mode == LEAD {
                    replies.insert(0, info(RUN_A));
                }
                read_pipeline(mode, replies)
            };
            let lead = (mode == LEAD).then(|| RUN_A.to_owned());

            let source = step_error(shape, with_step(Err(wrongtype()), Err(noperm())));
            assert_eq!(
                *source.kind(),
                fred::error::ErrorKind::InvalidArgument,
                "{shape}: the step's deterministic error, its after-step witness unread: {source}"
            );
            assert_eq!(
                failed(shape, with_step(Err(transient()), Err(noperm()))),
                (lead, fred::error::ErrorKind::IO),
                "{shape}: the step's transient error is a failed attempt keeping the lead, its \
                 after-step witness unread"
            );
            let source = step_error(shape, with_step(Ok(Value::Integer(1)), info(RUN_A)));
            assert_eq!(
                *source.kind(),
                fred::error::ErrorKind::Protocol,
                "{shape}: a step reply that is not a scan reply: {source}"
            );
            is_read(
                &format!("{shape}: after-step witness error"),
                with_step(scan("0", &[]), Err(noperm())),
            );
            is_missing(
                &format!("{shape}: after-step witness with no run_id"),
                with_step(scan("0", &[]), no_run_id()),
            );
        }
    }

    // ── One-attempt keyed steps on a real cluster ──────────────────────────

    /// The members a keyed walk's sorted set holds: past the compact
    /// encoding's 128, so a step answers about its `COUNT`.
    const KEYED_MEMBERS: usize = 200;

    /// A keyed step sent to a primary that does not own its key fails rather
    /// than following the redirect: on a cluster, a `ZSCAN` step pinned to a
    /// slot another primary owns is answered `-MOVED`, which a one-attempt
    /// step does not follow, so every attempt fails (its witness, read on the
    /// same route, unchanged) and the walk ends with the `Step` error. The
    /// same step on the key's own route answers members (the control).
    ///
    /// The control's witnesses carry the step's pin (b.qgz): the key is
    /// chosen so its owner is not the primary an unpinned `INFO server`
    /// reaches (fred routes that by its first argument, `server`), and the
    /// start witness the first step's pipeline led with, and its after-step
    /// witness, both read the run_id of `INFO server` sent to the key's owner
    /// by node.
    ///
    /// Cluster-gated (`LIVE_EDIT_TEST_CLUSTER_URL`, run by CI's
    /// `test-cluster:` job): it skips without the variable. Not run in this
    /// repository's sandbox, which has no cluster; its first run is CI's.
    ///
    /// Fail-on-old (in-place perturbations, not run in this sandbox): with
    /// `Route::options` giving a step fred's default attempts, fred follows
    /// the `-MOVED` to the owner and the pinned step answers its members;
    /// with a step's pipelined witnesses queued without the route's options
    /// (`queue_witness` sending a bare `INFO server`), they reach `server`'s
    /// slot's owner and the control's witnesses read another node's run_id.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_keyed_step_pinned_to_a_cluster_non_owner_fails_instead_of_following_the_redirect() {
        use fred::interfaces::SortedSetsInterface;

        use crate::storage::test_support::{test_cluster_url, TestValkey};

        let Some(url) = test_cluster_url() else {
            eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
            return;
        };
        let tv = TestValkey::setup_with_url(url, 1).await;
        let client = tv.client().pool.next();
        assert!(client.is_clustered(), "the cluster URL names a cluster");
        let routing = client
            .cached_cluster_state()
            .expect("a connected cluster client has a routing table");
        let owner_of = |name: &str| {
            routing
                .get_server(fred::util::redis_keyslot(name.as_bytes()))
                .cloned()
                .unwrap_or_else(|| panic!("the table names the owner of {name:?}"))
        };
        // The primary an unpinned `INFO server` reaches.
        let info_owner = owner_of("server");
        let key = (0..64)
            .map(|tag| format!("{}{{keyed-step-{tag}}}:index", tv.prefix()))
            .find(|key| owner_of(key) != info_owner)
            .expect("fixture: some hash tag lands on another primary than `server`'s slot");
        let owner = owner_of(&key);
        let members: Vec<(f64, String)> = (0..KEYED_MEMBERS)
            .map(|i| (i as f64, format!("m{i:03}")))
            .collect();
        let _: i64 = client
            .zadd(&key, None, None, false, false, members)
            .await
            .expect("seeding the index must succeed");
        let foreign = routing
            .slots()
            .iter()
            .find(|range| range.primary != owner)
            .map(|range| range.start)
            .expect("a three-shard cluster has another primary");
        let run_id_on = |node: Server| async move {
            let info: String = client
                .with_options(&Options {
                    cluster_node: Some(node.clone()),
                    ..Default::default()
                })
                .info(Some(InfoKind::Server))
                .await
                .unwrap_or_else(|e| panic!("INFO server on {node}: {e}"));
            parse_run_id(&info).unwrap_or_else(|seen| panic!("{node} reports {seen}"))
        };
        let owners_run_id = run_id_on(owner.clone()).await;
        assert_ne!(
            run_id_on(info_owner.clone()).await,
            owners_run_id,
            "fixture: an unpinned witness would read another process"
        );

        let walk = KeyedWalk::new(client, "ZSCAN", &key);
        let step = walk
            .step("0", 10, None, WALK_STEP_ATTEMPTS)
            .await
            .expect("control: the step on the key's own route answers");
        assert!(
            !step.members.is_empty(),
            "control: the owner answers members"
        );
        assert_eq!(
            (step.start.run_id(), step.witness.run_id()),
            (owners_run_id.as_str(), owners_run_id.as_str()),
            "control: the pipeline's start and after-step witnesses read the key's owner {owner}, \
             not {info_owner}"
        );

        let misrouted = KeyedWalk {
            client,
            walk: "ZSCAN",
            key: Arc::from(key.as_str()),
            route: Route::Slot(foreign),
            faults: WalkFaults::for_target(&key),
        };
        let start = read_witness(
            misrouted.client,
            &misrouted.route,
            &misrouted.faults,
            misrouted.walk,
            &misrouted.key,
        )
        .await
        .expect("the non-owner answers its INFO server");
        match misrouted
            .step("0", 10, Some(&start), WALK_STEP_ATTEMPTS)
            .await
        {
            Err(WalkError::Step { walk, target, .. }) => {
                assert_eq!((walk, &*target), ("ZSCAN", key.as_str()));
            }
            other => panic!(
                "a step sent to a primary that does not own its key fails, it is not redirected; \
                 got {:?}",
                other.map(|step| (step.next, step.members.len()))
            ),
        }
        println!("CLUSTER-GATED RAN: a_keyed_step_pinned_to_a_cluster_non_owner_fails");
    }
}
