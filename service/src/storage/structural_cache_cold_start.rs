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

//! The structural cache's cold start: the consistent full read that creates an
//! object's cache entry, or the uncacheable verdict that keeps the object out
//! of the cache (structural-edits SDD SR-4.3 steps 1, 3 and 4; SR-3.2's
//! dangling seeding; SR-2.5's metrics).
//!
//! Section references: `SR-n.m` is the structural-edits spec SDD (Spec Bee
//! `b.xes`); `§n.m` is `docs-internal/sdd.md`.
//!
//! # Where it sits
//!
//! A cold start (or a reload, which is the same read after a dropped entry) is
//! the first structural edit of an object on a replica. Its caller, the read
//! side (`storage::structural_cache_read`), runs the repair script in probe
//! mode first (step 0, `structural_cache_sync`), which answers the
//! commit-token check, the header mapping, and an object without
//! `chs_complete` — recorded through this module's verdict, with no full
//! read — before any full read. The same probe reports the object's five
//! index cardinalities, the ones the atomic read's step 2 reads, at no extra
//! round trip; the read side hands them to the cold start
//! ([`cold_start_probed`]), which routes on them. An index key of a foreign
//! type is reported as no cardinalities, not as a failed probe, and the cold
//! start then runs without them ("Without a probe", below).
//!
//! # The route
//!
//! In order; the first step that answers ends the cold start. The route is
//! decided from the probe before any permit is taken:
//!
//! 0. **The probe's cardinalities** give the atomic read's own step-2 item
//!    bound ([`snapshot_item_bound`]) before any full read.
//! 1. **An unmarked object** (no `chs_complete`): an
//!    [`UncacheableReason::NoChsComplete`] verdict. It cannot enumerate its
//!    dangling lists at any ceiling (SR-3.2).
//! 2. **The weight check** (policy decision P7): when a lower bound on the
//!    entry's weight ([`state_weight_lower_bound`]: the weight estimate's
//!    per-entry constants times the cardinalities, strings excluded) already
//!    exceeds the per-entry cap, an [`UncacheableReason::OverWeight`] verdict
//!    carrying that bound, with no full read. On a well-formed object whose
//!    indexes do not change between the probe and the read, a lower bound
//!    never refuses an object `admit` would have taken. Two cases can: an
//!    object that shrinks after the probe is judged on the probe's larger
//!    cardinalities, and a corrupt index (a member whose hash is missing)
//!    over-counts. Either way the object is judged one negative-entry TTL
//!    early, which costs it the uncacheable path, never a wrong entry. The
//!    entry is compact (per value row its key and external hash), so a
//!    layer's inline value bytes never count against the cap. What the check
//!    cannot see (P16) is what [`state_weight_lower_bound`] leaves out:
//!    every string, name and hash (section names, the interned field names,
//!    the boxed external hashes), the children table and the children lists.
//!    On a compact entry those can be most of its weight, so an entry over
//!    the cap by them passes this check. It then goes on through the route:
//!    a layer the plain read can take pays that read before `admit` refuses
//!    it `OverWeight`; a layer bound for the fenced read meets the deadline
//!    check (step 5) first, which refuses it `OverDeadline` with no read when
//!    its fenced read cannot fit the configured RPC deadline, and otherwise
//!    it pays the fenced read before `admit` refuses it.
//! 3. **The route predictor** ([`plain_read_predicted_to_fit`], P5/P11):
//!    the bound within the plain item ceiling, and the bound times
//!    [`STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM`] within the plain byte
//!    ceiling, sends the cold start to the plain read (step 4); otherwise to
//!    the deadline check and the fenced read (steps 5 and 6), estimated at
//!    the bound and the bound times that constant.
//! 4. **The plain atomic read**: [`read_object_snapshot_slim_with_ceilings`]
//!    for the `structural_cache` consumer at the client's shared ceilings,
//!    one `EVALSHA` with a proven cut, slim (the script's `slim` flag: no
//!    inline value is sent) and with the script's `return_dangling` flag
//!    set, so the dangling parents and the header's `commit_chain`,
//!    `superseded` and `chs_complete` come from the same cut. It holds one of
//!    the cache's per-replica plain permits
//!    ([`StructuralCache::acquire_plain_cold_start_read_permit`]) across that
//!    one read, handed to `load` with the snapshot (below), so at most
//!    [`STRUCTURAL_COLD_START_READ_PERMITS`](crate::storage::structural_cache::STRUCTURAL_COLD_START_READ_PERMITS)
//!    cold-start reads run on a replica at once, however many objects cold
//!    start together. A `TOO_LARGE` answer — a predictor miss, or a layer
//!    that grew since the probe — gives the permit back and is judged on its
//!    own cardinalities: unmarked → `NoChsComplete`; a weight lower bound
//!    over the cap → `OverWeight`; otherwise the deadline check and the
//!    fenced read, sized from the answer
//!    ([`LayerSize::from_too_large`](crate::storage::snapshot_fence::LayerSize::from_too_large),
//!    which scales a mid-read refusal's tallies to the whole layer). A miss
//!    is the one case a probed cold start reads twice; it shows on the
//!    `structural_cache` consumer's atomic `too_large` count, followed by a
//!    fenced read.
//! 5. **The deadline check**, on every way to the fenced read: when the
//!    fenced read's waiting cap at the size estimate, with the replica's
//!    whole configured RPC deadline (`LIVE_EDIT_RPC_DEADLINE_MS`) left, is
//!    zero ([`fenced_read_over_deadline`]) — the deadline does not cover the
//!    read's expected time plus one WATCH attempt's — an
//!    [`UncacheableReason::OverDeadline`] verdict, with no fenced read. It
//!    reads the layer's size and the configuration only, so no call on the
//!    replica could load the layer: the verdict is recorded, and the layer's
//!    later edits route to the WATCH path without a probe until it expires.
//!    It never reads what is left of the call's own deadline: a call merely
//!    short of time fails its fenced read for itself only (step 6). With no
//!    deadline in scope ([`cold_start`], [`cold_start_probed`]) the check is
//!    skipped. An operator clears the verdict by raising the RPC deadline or
//!    splitting the layer; it lifts when its negative entry expires.
//! 6. **The fenced read**: b.y3m's [`read_object_snapshot_fenced`] for the
//!    `structural_cache` consumer, with the cold start's own hook
//!    (`storage::structural_cache_fenced`): the cache's one-at-a-time and
//!    plain permits, a fenced-class catch-up read permit and a catch-up memory
//!    grant, all given back during every `RETRY_AFTER` sleep, the waiting
//!    capped so time is left for this read and one WATCH attempt. It reads at
//!    a proven cut at any size, with the structural facts under the fence. It
//!    ends:
//!    - verified → `load` the entry, [`ColdStartRoute::Fenced`];
//!    - `NotFound` → returned; the read side re-probes, as for the atomic
//!      read;
//!    - the cap, a capacity wait timing out, or
//!      [`FENCE_READ_ATTEMPTS`](crate::storage::snapshot_fence::FENCE_READ_ATTEMPTS)
//!      broken attempts → [`UncacheableReason::FencedReadFailed`], **for this
//!      call only** (step 5 has already refused a layer no call could read):
//!      no negative entry, no
//!      `live_edit_structural_cold_start_reads_total` tick, one DEBUG verdict
//!      line, and the call takes the WATCH path; the next call cold-starts
//!      afresh;
//!    - the capacity closed at shutdown → the shutdown error
//!      (`Unavailable`), never the WATCH path;
//!    - any other classified error → returned.
//!
//! Neither read ever falls back to the unfenced fan-out: a `TooLarge` answer
//! is this consumer's own outcome. Under write traffic the fan-out never
//! completes, which is the failure this cache exists to remove. And no cold
//! start takes the raised-ceiling atomic read: that read
//! ([`raised_cold_start_ceilings`], [`STRUCTURAL_COLD_START_ITEMS_FACTOR`],
//! [`STRUCTURAL_COLD_START_BYTES_FACTOR`]) has no production caller: the
//! sampled audit (`storage::structural_cache_audit`, SR-3.6) takes the fenced
//! read too, and only the measurement tool still measures the raised read.
//!
//! **Without a probe.** [`cold_start`] has no cardinalities, so its reads
//! decide: steps 1 to 3 are skipped and the plain read always runs first, its
//! `TOO_LARGE` reply's cardinalities and bound standing in for the probe's. A
//! layer over the plain byte ceiling and under the item bound then reads
//! twice: the predictor's job, which only a probed cold start has. The read
//! side takes this route only when an index key holds a foreign type, so the
//! probe reports no cardinalities: the plain read then reaches that index and
//! refuses it with its per-family error, ticking
//! `live_edit_snapshot_read_type_error_total{key_family}`. Tests drive it
//! directly.
//!
//! **What the read side learns of the size.** An uncacheable answer carries
//! the layer's size when the cold start knows the layer is above the plain
//! ceilings ([`ColdStart::Uncacheable`]'s `above_ceiling`): the probe's bound
//! over the item ceiling, a `TOO_LARGE` answer, a failed fenced read's latest
//! size, or an `OverWeight` entry whose read measured above the ceilings —
//! whatever the verdict, an unmarked object's `NoChsComplete` included, since
//! its structural edits take the WATCH path too. The read side answers an
//! unmarked object's `NoChsComplete` from its probe, before any cold start,
//! and carries the same probe bound ([`probed_above_ceiling`]). The
//! structural route classifies its WATCH fallback from it (policy decision
//! P12), so that fallback registers its writer intent before its first
//! `WATCH`. A layer that is merely predicted over the byte ceiling carries no
//! size.

//! # The entry it builds
//!
//! The read decodes the snapshot straight into the compact form
//! ([`CompactState`]: per value row its key and validated external hash, each
//! time sample keyed by its stored bits), which is then normalized with
//! `domain::replay::normalize_state` (SR-3.2), at the read's cut `seq`
//! and `commit_chain`. Its `dangling_parents` is the read's `return_dangling`
//! element — the `obj:chs` members whose parent is neither the root nor a live
//! section, enumerated in the same script as the cut — unioned with whatever
//! `normalize_state` drops. The two are not the same computation:
//! `normalize_state` sees only the lists the read returned, and the script
//! already left out every list under a dangling parent, so on a well-formed
//! object it drops nothing and the read's element is the whole set. It can
//! drop a list only where the index is corrupt — an `obj:secs` member whose
//! section hash is missing counts as live to the script but is absent from the
//! decoded sections — and that parent's stored list is, to the entry, exactly
//! a list it does not hold, so it belongs in the set too.
//!
//! After normalizing, `load` trims each children list to its length: the
//! decoder collects a list from a filtered iterator, which leaves a short list
//! spare capacity, and the entry is the decoded state itself, not a copy.
//! `admit`'s compaction
//! ([`compact_state`](crate::storage::structural_cache::compact_state)) trims
//! the hash tables, not the
//! lists, so this is the one place a cold-started entry's lists are trimmed.
//!
//! The entry goes to [`StructuralCache::admit`], which compacts and weighs it
//! and hands it back checked out to the caller. Over the per-entry weight cap,
//! `admit` records an [`UncacheableReason::OverWeight`] negative entry.
//!
//! **What the read holds, and where it is given back.** Both routes hand
//! `load` what their read held ([`ColdStartHold`]): the cache's read permits
//! and, for a fenced read, its memory grant inside `T`. `load` gives them
//! back at three named points:
//!
//! - **Seam A**, the first statement after the header checks (superseded,
//!   unmarked; each of those exits gives the whole hold back first), and the
//!   cold start's **charge point** (CR0): for a structural `UpdateLayer`
//!   (`cold_start_for`'s `commit_charge`) the decoded entry — its state at
//!   its estimated weight, and beside it its plan indexes' build at its peak
//!   (`PlanIndexes::build_peak_bytes`) — enters the call's
//!   upload commit charge, never waiting, while the read permits still bound
//!   the read, so the entry is never outside both (SDD §5.4). Then the read
//!   permits drop, so they end with the read. A refused charge still loads,
//!   admits and stores the entry, and answers [`ColdStart::NeedsCharge`]
//!   instead of reading the commit token: the call waits holding nothing,
//!   and its re-run is a hit. Since no charge then holds the entry, the read
//!   permits and the grant (seam B) are kept until it is admitted and put
//!   back, where the cache's budget counts it, and drop only then.
//!   **A refused plain read's hold is bounded.** Under its read permits it
//!   then holds the decoded state and the indexes' build beside its reply,
//!   which the pod memory limit's cold-start term covers only up to
//!   r × (1 + b) × the plain byte ceiling
//!   ([`plain_cold_start_hold_allowance_bytes`], 5,535,891 B at the default;
//!   SDD §21.2). So a refused plain read whose state and build are over that
//!   allowance loads nothing: its snapshot, read permits and ticket drop at
//!   seam A, before the state is normalized or the build runs, nothing is
//!   stored, and it answers [`ColdStart::NeedsCharge`] with the same target,
//!   ticking no route and logging one DEBUG line
//!   ([`COLD_START_HOLD_OVER_ALLOWANCE_MSG`]). The call waits for its target
//!   holding nothing and reads again under it, so that read's CR0 is granted
//!   unless the layer grew meanwhile. On every weighed layer shape the hold
//!   is far below the allowance. A fenced read's refused hold is unchanged.
//! - **Seam B**, the decode point: the read decoded straight into the
//!   entry's compact state, which `load` moves out of the snapshot and which
//!   lives on as the entry. Once that state is normalized and its dangling
//!   set built, the memory grant counted in `T` for the read's reply drops
//!   here (after the put-back instead, when the charge at seam A was
//!   refused).
//! - **Seam C**, `admit`: the only place a cold-started entry is compacted
//!   (its hash tables; `load` already trimmed its children lists) and
//!   weighed, which the read side's `ReadyEntry { path: StructuralFull }`
//!   reaches before the structural route's first plan (`plan_in_place`).
//!   Nothing but the token `HGET` and the repair sits between them.
//!
//! **Retirement during the read.** The cold start takes its epoch before its
//! first read ([`StructuralCache::begin_cold_start`]) and admits under it. So
//! when the object is invalidated while the cold start reads — a `DeleteLayer`
//! or overwrite retiring it, an audit dropping it — `admit` refuses the
//! entry, whose cut may predate the retirement, and [`cold_start`] answers
//! [`ColdStart::Invalidated`]; an invalidation after the admission refuses the
//! guard's put-back instead. Either way a retired object's entry is never
//! restored (the spec SDD's Mechanism lifecycle). An invalidation before the
//! cold start began, such as a reload's own `discard`, refuses nothing. An
//! uncacheable verdict found during such a race is still recorded: a negative
//! entry for a retired object id is never consulted again, since object ids
//! are not reused, and expires with its TTL.
//!
//! An
//! admitted entry that mini-moka's own admission later refuses is kept in the
//! cache's side store and served as a hit: it is a loaded entry here, never an
//! uncacheable verdict, and never ticks the `uncacheable` route.
//!
//! # The commit token
//!
//! After a read that loads an entry (or finds the object superseded), one
//! same-slot `HGET obj:wtok <client_id>` reads the call's commit-token record
//! (SR-4.3 step 1), with the read's permits and memory already given back. It
//! runs after the read, so it sees every commit the cut
//! includes — the call's own landed commit among them, should an earlier
//! attempt's in-doubt commit have landed after the probe. That is all the
//! token check needs, so the `HGET` need not share the read's cut. The caller
//! checks the record before planning (SR-4.2a). A failed `HGET` fails the
//! cold start, but the loaded entry is put back (the guard's `Drop`): it is a
//! valid cut, so the retry repairs it rather than reading it again. A record
//! key that changed to a foreign type after the probe answers
//! [`ColdStart::CommitTokenRecordWrongType`], the probe's own refusal, with
//! the loaded entry put back the same way; a `TYPE` read, on that path only,
//! supplies the type name the refusal carries.
//!
//! # What the caller maps
//!
//! [`cold_start`] does not apply SR-4.2's header mapping. A read that finds no
//! header, or a header without a decimal `seq`, fails `NotFound` (the atomic read's rule,
//! which cannot tell the two apart); a superseded header answers
//! [`ColdStart::Superseded`] with the token record, and no entry. The caller
//! re-probes to classify the first and maps the second after its token check,
//! as SR-4.2's header mapping requires. [`ColdStart::Invalidated`] is also
//! answered by a re-probe, and [`ColdStart::CommitTokenRecordWrongType`] is
//! returned as the probe's refusal would be.
//!
//! # Metrics
//!
//! Each cold start that produces an entry or an uncacheable verdict — not an
//! error before its verdict or admission, a superseded object, an entry
//! refused because the object was invalidated while it read, a refused plain
//! read over its hold allowance (seam A), or a disabled cache — ticks
//! `live_edit_structural_cold_start_reads_total{route}` once, under the route
//! that ended it, and observes its wall time — from its start, every permit
//! wait included, to the verdict or the admission, before the token `HGET`;
//! a verdict from step 1, 2 or 5 on the probe's figures reads nothing in
//! between — on
//! `live_edit_structural_cold_start_read_duration_seconds{route}`. Every read
//! it issues also shows on `live_edit_snapshot_read_*{consumer="structural_cache"}`,
//! a fenced one on `path="fenced"` with its fence waits and holds under the
//! same consumer, and its `P_f` wait on
//! `live_edit_catchup_snapshot_read_wait_seconds{consumer="structural_cache",class="fenced"}`.
//! The read side's probe verdict for an object without `chs_complete` ticks
//! the `uncacheable` route the same way, its wall time the probe's. A fenced
//! read that failed for its call ticks no route: it is neither an entry nor
//! a recorded verdict.
//!
//! # Reporting a verdict
//!
//! Every uncacheable verdict the structural route reaches is reported once,
//! by [`report_uncacheable`]: one tick of
//! `live_edit_structural_cache_uncacheable_total{reason}` and one line
//! ([`UNCACHEABLE_VERDICT_MSG`], INFO; DEBUG for `reload_limit` and
//! `fenced_read_failed`, whose rates nothing bounds) naming the session and
//! layer (when the caller passed them, [`VerdictLayer`]), the object, the
//! reason and what the verdict compared — for `over_weight` the estimated
//! weight against the per-entry cap (the weight check's lower bound, when
//! step 2 refused before any read); for `fenced_read_failed` its `cause`
//! (`cap`, `capacity_timeout` or `broken`), the attempts, the layer's latest
//! size, and the time waited against the cap ([`FencedReadFailure`]); for
//! `over_deadline` the size estimate it was judged at, the configured RPC
//! deadline, and what the fenced read would keep back of it at that size,
//! the read's expected time plus one WATCH attempt's (`rpc_deadline_ms`,
//! `reserve_ms`; [`OverDeadline`]). The verdicts this module and the read
//! side reach are the cold start's
//! (above), the probe's `no_chs_complete`, the read side's `reload_limit`,
//! and an explicit put-back the per-entry cap refused. A verdict recorded as a
//! negative entry is reported by whoever first takes it
//! ([`StructuralCache::take_unreported`], exactly once per negative entry):
//! the site that reached it, at once, with the bounds; or, for a verdict no
//! caller heard — the implicit put-back of a dropped call's over-cap entry —
//! the first structural lookup it routes, with the weight the entry carries.
//! A verdict already reported is never reported again by the lookups it
//! routes, so a hot uncacheable layer logs once per negative-entry TTL
//! (`STRUCTURAL_CACHE_IDLE_TTL`) per replica, not per edit. A disabled cache
//! reports nothing: it is a configuration, not a verdict.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use fred::interfaces::{HashesInterface, KeysInterface};

use crate::domain::delta::RewriteCount;
use crate::domain::replay::normalize_state;
use crate::domain::types::CompactState;
use crate::error::ServiceError;
use crate::observability::metrics::{
    SharedMetrics, STRUCTURAL_COLD_START_ROUTE_ATOMIC, STRUCTURAL_COLD_START_ROUTE_FENCED,
    STRUCTURAL_COLD_START_ROUTE_UNCACHEABLE,
};
use crate::storage::client::{
    is_wrongtype_reply, map_transient_fred_err, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA,
};
use crate::storage::commit_charge::{ObjectStateNeed, SharedCommitCharge};
use crate::storage::diff_read::{
    read_object_snapshot_slim_with_ceilings, snapshot_item_bound, AtomicSnapshotOf,
    SlimSnapshotReadConsumer, SnapshotCardinalities, SnapshotCeilings, SnapshotReadOf,
};
use crate::storage::keys::KeySchema;
use crate::storage::layer::commit_token_record_wrong_type_refusal;
use crate::storage::snapshot_fence::read::read_object_snapshot_fenced;
use crate::storage::snapshot_fence::writer::RemainingDeadlineFn;
use crate::storage::snapshot_fence::{coefficients, FenceCoefficients, LayerSize};
use crate::storage::structural_cache::{
    estimate_state_bytes, state_weight_lower_bound, AdmitRefused, CacheEntry, CheckedOut,
    ColdStartTicket, StructuralCache, UncacheableReason,
};
use crate::storage::structural_cache_fenced::{
    failed_for_the_call, max_size, shutting_down, wait_cap, wait_reserve, ColdStartFenceHook,
    ColdStartHold, FencedFailure,
};

/// The raised-ceiling atomic read's item ceiling, as a multiple of the shared
/// atomic-read item ceiling: 6, chosen after measurement (below) over the
/// spec's 4×.
///
/// **No production caller.** Neither a cold start nor the sampled audit takes
/// the raised read: above the plain ceilings both take b.y3m's fenced read
/// (module docs, "The route"; `storage::structural_cache_audit`). The
/// measurement tool's `shardblock` subcommand still measures it, outside the
/// cache and its permits (tests take those through
/// [`StructuralCache::acquire_raised_cold_start_read_permits`]), and the
/// load generator's `--cold-start-factors` and the measurement scripts name
/// it, so it keeps its name.
///
/// **Sizing.** Both factors were sized so that the largest gate layer, 5,500
/// harness sections, plus 25 % — 6,875 sections — fits the raised read.
///
/// **The margin is judged by the pre-read item bound, not the item tally.**
/// Whether the raised read answers is decided by `object_snapshot_read.lua`'s
/// step-2 item bound before any entry is read: its bound counts 2 + 6 items
/// per section, 1 + 2 per tombstone, 1 + 5 per field, 1 + 3 per time sample,
/// plus the `obj:chs` members. That is stricter than the tally the read
/// counts as it goes (about 117 bound items against about 103 tallied for
/// the same harness loadtest content). On the harness layer the bound is
/// about 29.25 items per section, so 6,875 sections are 6,875 × 29.25 ≈ 201k
/// bound items. At the default item ceiling of 37,554, 5.35× is needed:
/// 4× (150,216) misses, and 5× (187,770) fits only about 6,419 sections
/// (+16.7 %). 6× gives 225,324.
///
/// Measured on this build: 6,875 harness sections have a bound of 201,058,
/// which needs 5.35×, so 6× leaves 12.1 % headroom.
///
/// **Its shard block.** At 6× items and 8× bytes the raised read blocks the
/// shard for 138–169 ms on a private Valkey. On AMR, which project SDD §21.5
/// puts at 1.30–1.50× the sandbox, that is about 180–220 ms at 1.3× and about
/// 207–254 ms at §21.5's conservative 1.502×. Writers committing meanwhile
/// waited up to 144–163 ms
/// (`docs-internal/validation/structural-edits-b2wo-20261002/epic4-measurements.md`).
/// That cost is why cold starts no longer take it.
pub const STRUCTURAL_COLD_START_ITEMS_FACTOR: u64 = 6;

/// The raised-ceiling atomic read's byte ceiling, as a multiple of the shared
/// atomic-read byte ceiling: 8, chosen after measurement (below), at the top
/// of the spec's expected 4–8× range. No production
/// caller (see [`STRUCTURAL_COLD_START_ITEMS_FACTOR`]).
///
/// Epic 1 measured the harness's 5,000-section layer at about 2.4 MB (18.9
/// bytes per tallied item, against the 11.98 the shared ceilings then sat on), so
/// the largest gate layer plus 25 % is about 3.3 MB. At the 449,938-byte
/// ceiling it was sized against (the default then), 8× is 3,599,504 bytes,
/// which fits.
///
/// Measured on this build: 6,875 sections read 3,315,626 bytes (7.37×), so 8×
/// leaves 8.6 % headroom. Bytes bind before items on this shape, at about
/// 7,458 sections: a 7,383-section layer loads and a 7,535-section one is
/// refused.
pub const STRUCTURAL_COLD_START_BYTES_FACTOR: u64 = 8;

/// The route predictor's bytes per bound item, `K` (policy decisions P5 and
/// P11): a probed cold start takes the plain read only when the probe's item
/// bound ([`snapshot_item_bound`]) times `K` is within the plain byte
/// ceiling, and the bound itself within the plain item ceiling
/// ([`plain_read_predicted_to_fit`]). Otherwise it takes the fenced read,
/// with no plain read first.
///
/// **Why a predictor.** `object_snapshot_read.lua` checks its item bound
/// before reading anything but its byte ceiling only on its running tally.
/// So a layer over the plain byte ceiling and under the item bound would cost
/// a plain read that reads every row hash and refuses on bytes, then the
/// fenced read.
///
/// **The rule (P11).** `K = ⌈max D⌉`, where `D` is the read's byte tally
/// divided by its pre-read item bound, taken over the item-dominated layer
/// shapes — Kitchen_set, field-heavy prims, and reference-plus-transform-op
/// assets (the demo shape) — and the gze real-traffic layer. So no
/// item-dominated layer ever reads twice. Measured (re-taken when the
/// `too_large` trigger below fires):
///
/// | Layer | Largest `D` | Source |
/// |---|---|---|
/// | Kitchen_set live layer, 1–6 copies | 16.23 | S1 static run |
/// | field-heavy prims, 250–10,000 | 16.84 | S1 static run |
/// | reference + transform ops, 250–5,000 assets | 17.75 (model; 17.62 the largest server-read `D`) | S1 static run |
/// | gze real-traffic layer, every size its history passed through | 15.84 (its full history: 21 sections, bound 497, 7,873 B) | captured-history tally (`docs-internal/validation/structural-edits-b2wo-20261002/epic4-measurements.md`, "Cold-start tallies and the factor margins"; SDD §21.5) |
///
/// The S1 figures are in
/// `docs-internal/validation/structural-edits-b2wo-20261002/s1-shapes-results.md`
/// (`bee/b.2wo` `61414e73`), which also names the generator that authored
/// the six layer shapes. So `K = ⌈17.75⌉ = 18`, below the 20 the
/// predictor shipped with, which had been chosen above three-attribute
/// prims' 16.5 and Kitchen_set's 11.98 before the shapes were measured.
///
/// **Not yet measured: real traffic at large sizes.** gze is a 21-section
/// capture, and its `D` was still rising with size (12.08 at one section,
/// 15.84 at 21). A real layer near the plain ceilings may be denser. A probed
/// cold start that the predictor sent to the plain read and that read
/// refused shows as a tick of
/// `live_edit_snapshot_read_total{consumer="structural_cache",path="atomic",outcome="too_large"}`
/// followed by a `path="fenced"` read and a
/// `live_edit_structural_cold_start_reads_total{route="fenced"}` tick.
/// **Trigger:** if that `too_large` series rises on real traffic (dev or a
/// customer) for layers of item-dominated content, re-derive `K` from those
/// layers' `D` (tally a copy of the layer with the internal structural-cache
/// measurement harness).
///
/// **The band.** A layer whose bound is within the item ceiling but whose
/// bound times `K` is over the byte ceiling goes fenced. Layers lighter than
/// the densest item-dominated shape in that band (Kitchen-like layers near
/// the item ceiling) pay one fenced read where a plain read would have
/// fitted: a hold of tens to a few hundred milliseconds, and colliding
/// writers on that layer wait one `poll` step. No shard is blocked.
///
/// **Misses.** A byte-dense layer (large inline arrays, recorded rigs) has a
/// `D` far above `K`, so the predictor sends it to the plain read, which
/// refuses mid-read once its byte tally passes the ceiling — after about
/// `max_bytes ÷ D` items, a few milliseconds of shard time — and the cold
/// start then reads fenced. Each miss shows on
/// `live_edit_snapshot_read_total{consumer="structural_cache",path="atomic",outcome="too_large"}`,
/// followed by a `path="fenced"` read. On a probed cold start that series
/// counts only a miss or a layer that grew past the plain ceilings between
/// the probe and the read. Guaranteeing one read would take an O(n) byte pass
/// before every plain read.
pub const STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM: u64 = 18;

/// `r`, in hundredths: 280, so r = 2.8 — the plain cold start's decoded
/// state over its reply, as the pod memory limit's cold-start term counts it
/// (the chart-limit record's `cold_start_entry_to_read_ratio`; SDD §21.2,
/// *Cold-start reads*). The configuration's chart-limit test pins it to the
/// record, so the term and [`plain_cold_start_hold_allowance_bytes`] cannot
/// drift apart.
pub const COLD_START_ENTRY_TO_READ_CENTI: u64 = 280;

/// `b`, in hundredths: 227, so b = 2.27 — the plan indexes' build at its
/// peak over the decoded state, as the same term counts it (the chart-limit
/// record's `cold_start_index_build_peak_ratio`), pinned to the record the
/// same way.
pub const COLD_START_BUILD_PEAK_CENTI: u64 = 227;

/// The most a plain cold start whose charge at its decode (CR0) was refused
/// may hold under its read permits beside its reply: the decoded state at
/// its estimated weight plus the plan indexes' build at its peak, against
/// ⌊`byte_ceiling` × r × (1 + b)⌋ ([`COLD_START_ENTRY_TO_READ_CENTI`],
/// [`COLD_START_BUILD_PEAK_CENTI`]) — the pod memory limit's per-read
/// cold-start term less its reply. 5,535,891 B at the default 604,619 B
/// ceiling. Exact in `u128`, saturating at `u64::MAX`.
///
/// A refused plain cold start whose hold is over it loads nothing (module
/// docs, seam A), so the term holds for every layer shape by construction.
pub const fn plain_cold_start_hold_allowance_bytes(byte_ceiling: u64) -> u64 {
    let allowance = byte_ceiling as u128
        * COLD_START_ENTRY_TO_READ_CENTI as u128
        * (100 + COLD_START_BUILD_PEAK_CENTI as u128)
        / 10_000;
    if allowance > u64::MAX as u128 {
        u64::MAX
    } else {
        allowance as u64
    }
}

/// The line a refused plain cold start whose hold is over its allowance
/// logs, at DEBUG (module docs, seam A).
pub(crate) const COLD_START_HOLD_OVER_ALLOWANCE_MSG: &str =
    "structural cache cold start held more than its allowance after its charge was refused; \
     it loads nothing and reads again once the call holds its charge";

/// The message of the line every uncacheable verdict logs — INFO (DEBUG
/// for `reload_limit` and `fenced_read_failed`) — ([`report_uncacheable`]):
/// the string operators search for.
pub(crate) const UNCACHEABLE_VERDICT_MSG: &str =
    "structural cache found the object uncacheable; its structural edits take the WATCH path";

/// The session and layer of the structural call that reached a verdict: what
/// the verdict line names beside the object
/// ([`StructuralReadRequest::with_layer`]).
///
/// [`StructuralReadRequest::with_layer`]: crate::storage::structural_cache_read::StructuralReadRequest::with_layer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VerdictLayer<'a> {
    /// The call's session.
    pub(crate) session_id: &'a str,
    /// The call's layer.
    pub(crate) layer_name: &'a str,
}

/// What a [`UncacheableReason::FencedReadFailed`] verdict reports: why the
/// fenced cold-start read failed for its call, and how far it got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FencedReadFailure {
    /// `cap`, `capacity_timeout` or `broken`.
    pub(crate) cause: &'static str,
    /// The attempts the read started.
    pub(crate) attempts: u32,
    /// The largest layer size the read knew of: its latest, the route's
    /// estimate when that put the layer above the ceilings, or what the store
    /// reported. When the layer is above the ceilings, the size the WATCH
    /// fallback is classified at.
    pub(crate) size: LayerSize,
    /// How long the fenced route waited and read.
    pub(crate) waited: Duration,
    /// The waiting cap at that size.
    pub(crate) cap: Duration,
}

/// What an [`UncacheableReason::OverDeadline`] verdict reports: the size it
/// was judged at and what that size's fenced read would need of the
/// configured RPC deadline it was judged against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OverDeadline {
    /// The cold start's size estimate: the probe's, or a `TOO_LARGE`
    /// answer's.
    pub(crate) size: LayerSize,
    /// The replica's configured RPC deadline (`LIVE_EDIT_RPC_DEADLINE_MS`).
    pub(crate) rpc_deadline: Duration,
    /// What the waiting cap keeps back of it at `size`: the fenced read's
    /// expected time plus one WATCH attempt's ([`wait_reserve`]), at or above
    /// the deadline.
    pub(crate) reserve: Duration,
}

/// One uncacheable verdict, as [`report_uncacheable`] counts and logs it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Verdict<'a> {
    pub(crate) object_id: u64,
    pub(crate) layer: Option<VerdictLayer<'a>>,
    pub(crate) reason: UncacheableReason,
    /// The estimated state weight an `OverWeight` verdict refused, when
    /// known: the weight the per-entry cap reads, without the plan indexes.
    pub(crate) weight_bytes: Option<u64>,
    /// Set for a `FencedReadFailed` verdict only.
    pub(crate) fenced: Option<FencedReadFailure>,
    /// Set for an `OverDeadline` verdict only.
    pub(crate) over_deadline: Option<OverDeadline>,
}

impl<'a> Verdict<'a> {
    /// `reason` for `object_id`, its weight taken from an `OverWeight`
    /// reason, with no fenced-read or deadline report.
    pub(crate) fn new(
        object_id: u64,
        layer: Option<VerdictLayer<'a>>,
        reason: UncacheableReason,
    ) -> Self {
        let weight_bytes = match reason {
            UncacheableReason::OverWeight(weight) => Some(weight),
            _ => None,
        };
        Self {
            object_id,
            layer,
            reason,
            weight_bytes,
            fenced: None,
            over_deadline: None,
        }
    }

    fn with_fenced(mut self, fenced: FencedReadFailure) -> Self {
        self.fenced = Some(fenced);
        self
    }

    fn with_over_deadline(mut self, over_deadline: OverDeadline) -> Self {
        self.over_deadline = Some(over_deadline);
        self
    }
}

/// Report one uncacheable verdict (module docs, "Reporting a verdict"): tick
/// `live_edit_structural_cache_uncacheable_total{reason}` and log one
/// [`UNCACHEABLE_VERDICT_MSG`] line — INFO, or DEBUG for `reload_limit` and
/// `fenced_read_failed`. Called once per verdict: for a recorded verdict only
/// by whoever took it with [`StructuralCache::take_unreported`]; the
/// per-call verdicts, never recorded, where they are reached. A `Disabled`
/// reason reports nothing. Absent fields — the session and layer when the
/// caller passed none, the bounds a reason does not compare — are left off
/// the line, not logged empty.
pub(crate) fn report_uncacheable(
    metrics: Option<&SharedMetrics>,
    cache: &StructuralCache,
    verdict: &Verdict<'_>,
) {
    let Some(reason) = verdict.reason.label() else {
        return;
    };
    if let Some(m) = metrics {
        m.inc_structural_cache_uncacheable(reason);
    }
    let fenced = verdict.fenced;
    let over_deadline = verdict.over_deadline;
    let size = fenced
        .map(|f| f.size)
        .or_else(|| over_deadline.map(|o| o.size));
    let over_weight = matches!(verdict.reason, UncacheableReason::OverWeight(_));
    macro_rules! verdict_line {
        ($level:ident) => {
            tracing::$level!(
                session_id = verdict.layer.map(|l| l.session_id),
                layer_name = verdict.layer.map(|l| l.layer_name),
                object_id = verdict.object_id,
                reason,
                weight_bytes = verdict.weight_bytes,
                max_entry_bytes = over_weight.then(|| cache.max_entry_bytes()),
                cause = fenced.map(|f| f.cause),
                attempts = fenced.map(|f| f.attempts),
                size_items = size.map(|s| s.items),
                size_bytes = size.map(|s| s.bytes),
                waited_ms = fenced.map(|f| f.waited.as_millis() as u64),
                cap_ms = fenced.map(|f| f.cap.as_millis() as u64),
                rpc_deadline_ms = over_deadline.map(|o| o.rpc_deadline.as_millis() as u64),
                reserve_ms = over_deadline.map(|o| o.reserve.as_millis() as u64),
                "{}",
                UNCACHEABLE_VERDICT_MSG
            )
        };
    }
    // `reload_limit` and `fenced_read_failed` are never recorded, so nothing
    // bounds their rate: while the cause recurs (a rolling deploy's busy older
    // writer; a layer whose fence stays busy) every structural edit on the
    // layer can reach it again. So they log at DEBUG and their rate is the
    // counter's (SDD §17.2's log-volume rules); every other verdict is
    // bounded by its negative entry's TTL and logs at INFO.
    if matches!(
        verdict.reason,
        UncacheableReason::ReloadLimit | UncacheableReason::FencedReadFailed
    ) {
        verdict_line!(debug);
    } else {
        verdict_line!(info);
    }
}

/// The route that ended one cold start: the `route` label of the cold-start
/// metrics (SR-2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdStartRoute {
    /// The plain atomic read, at the shared ceilings, loaded the entry.
    Atomic,
    /// b.y3m's fenced read loaded the entry: a layer above the plain ceilings
    /// or predicted over the plain byte ceiling, read at a proven cut under
    /// the cache's one-at-a-time and plain permits, a fenced-class catch-up
    /// read permit and a catch-up memory grant
    /// (`storage::structural_cache_fenced`).
    Fenced,
    /// The cold start ended in an uncacheable verdict, recorded as a negative
    /// entry.
    Uncacheable,
}

impl ColdStartRoute {
    /// The metric label (a
    /// [`crate::observability::metrics::STRUCTURAL_COLD_START_ROUTES`] value).
    pub fn label(self) -> &'static str {
        match self {
            Self::Atomic => STRUCTURAL_COLD_START_ROUTE_ATOMIC,
            Self::Fenced => STRUCTURAL_COLD_START_ROUTE_FENCED,
            Self::Uncacheable => STRUCTURAL_COLD_START_ROUTE_UNCACHEABLE,
        }
    }
}

/// An entry a cold start loaded, checked out to the call that read it.
#[derive(Debug)]
pub struct ColdStartLoaded {
    /// The new entry, at the read's `(seq, commit_chain)` cut, checked out:
    /// the caller plans on it and settles the guard (SR-3.4). Not yet stored —
    /// the guard's put-back stores it, and its `Drop` does if nothing else
    /// settles it.
    pub entry: CheckedOut,
    /// Which read loaded it: [`ColdStartRoute::Atomic`] or
    /// [`ColdStartRoute::Fenced`].
    pub route: ColdStartRoute,
    /// The call's commit-token record (`HGET obj:wtok <client_id>`), read
    /// after the read; `None` when the field is absent. The caller checks it
    /// before planning (SR-4.2a).
    pub commit_token: Option<String>,
}

/// What [`cold_start`] found.
//
// `clippy::large_enum_variant` allowed: `Loaded` carries the checkout guard
// by value, like `CacheLookup::Hit`. The value is built and matched once per
// cold start and never stored, so its size costs one move.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum ColdStart {
    /// A consistent full read loaded the object's entry.
    Loaded(ColdStartLoaded),
    /// The read's header carries `superseded`: nothing was admitted, and the
    /// caller maps it to `FailedPrecondition(SUPERSEDED)` after its token
    /// check, which `commit_token` answers (SR-4.2).
    Superseded {
        /// As in [`ColdStartLoaded::commit_token`].
        commit_token: Option<String>,
    },
    /// The object is uncacheable. For every reason but
    /// [`UncacheableReason::Disabled`] and
    /// [`UncacheableReason::FencedReadFailed`] (this call only) the verdict is
    /// recorded as a negative entry, so later lookups route the object without
    /// another read until it expires (SR-3.4).
    Uncacheable {
        /// Why.
        reason: UncacheableReason,
        /// The layer's size, when the cold start knows the layer is above the
        /// plain atomic-read ceilings (module docs, "What the read side learns
        /// of the size"); `None` otherwise.
        above_ceiling: Option<LayerSize>,
    },
    /// The object was invalidated while the cold start read it — retired by
    /// a `DeleteLayer` or a `CreateLayer` overwrite, or dropped by an audit —
    /// so the read's cut may predate the retirement. Nothing was stored or
    /// recorded, and no commit token was read. The caller re-probes: a retired
    /// object answers as gone or superseded there, and an audit-dropped one
    /// cold-starts again.
    Invalidated,
    /// The commit-token record key holds a foreign type — it changed type
    /// after the probe, which would have refused it. `refusal` is the commit
    /// backends' byte-identical `WRONGTYPE_KEY` refusal, the probe's answer
    /// too, for the caller to answer as
    /// `StructuralRead::CommitTokenRecordWrongType` (counted, logged and
    /// returned by its caller). A loaded entry was put back first: it is a
    /// valid cut. Answered after a superseded read too, as the commit paths
    /// check the token before `superseded`.
    CommitTokenRecordWrongType {
        /// The `Internal` refusal.
        refusal: ServiceError,
    },
    /// With a commit charge only (`cold_start_for`'s `commit_charge`): the
    /// decoded entry could not be held in the call's upload commit charge
    /// without waiting at seam A (CR0). It was admitted and stored — unless
    /// it came from the plain read and its state and build were over that
    /// read's allowance ([`plain_cold_start_hold_allowance_bytes`]), when
    /// nothing was loaded or stored — and the read permits and grant
    /// dropped; no commit token was read. The caller waits for `target`
    /// object-state bytes, holding nothing, and its next read is a cache hit,
    /// or, when nothing was stored, a cold start read again under the held
    /// charge.
    NeedsCharge {
        /// The whole target: the entry's state and its plan-index build.
        target: u64,
    },
}

/// The raised-ceiling read's ceilings: `plain` (the shared ceilings) times
/// [`STRUCTURAL_COLD_START_ITEMS_FACTOR`] and
/// [`STRUCTURAL_COLD_START_BYTES_FACTOR`], saturating. No production read
/// takes them: neither a cold start nor the sampled audit, which reads
/// fenced above the plain ceilings; only the measurement tools do.
pub fn raised_cold_start_ceilings(plain: SnapshotCeilings) -> SnapshotCeilings {
    SnapshotCeilings {
        max_items: plain
            .max_items
            .saturating_mul(STRUCTURAL_COLD_START_ITEMS_FACTOR),
        max_bytes: plain
            .max_bytes
            .saturating_mul(STRUCTURAL_COLD_START_BYTES_FACTOR),
    }
}

/// Cold-start `object_id` into `cache` for the structural call whose
/// `client_id` is `client_id`, with no probe's cardinalities: the plain read
/// always runs first, and above the plain ceilings the fenced read (module
/// docs, "Without a probe"; the read side routes through
/// [`cold_start_probed`]'s steps whenever its probe reports cardinalities),
/// with no deadline in scope. The module documentation describes the routes,
/// the entry and the metrics.
///
/// It does not consult the cache first: the caller has already looked the
/// object up ([`StructuralCache::lookup`]), which routes an object with a live
/// negative entry without a cold start, and has run the probe-mode repair.
///
/// On a disabled cache it reads nothing and answers
/// [`ColdStart::Uncacheable`] with [`UncacheableReason::Disabled`], recording
/// and ticking nothing (SR-3.3).
///
/// # Errors
///
/// - `NotFound` when a read finds no header, or a header without a decimal
///   `seq`; the caller re-probes to tell the two apart.
/// - A read's own classified error (transient → `Aborted`, permanent →
///   `Internal`), as [`read_object_snapshot_with_ceilings`] and
///   [`read_object_snapshot_fenced`] return it.
/// - `Unavailable` when the replica is shutting down during a fenced read
///   (its capacity closed).
/// - The commit-token `HGET`'s error (or, after a `WRONGTYPE`, the `TYPE`
///   read's), classified on the data pool; a loaded entry is put back first.
///   A record key of a foreign type is not an error: it answers
///   [`ColdStart::CommitTokenRecordWrongType`].
pub async fn cold_start(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    object_id: u64,
    client_id: &str,
) -> Result<ColdStart, ServiceError> {
    cold_start_for(
        client, keys, cache, object_id, client_id, None, None, None, None, None,
    )
    .await
}

/// [`cold_start`] after a probe that read `cards`: the route the read side
/// takes (module docs, "The route"). `cards` are the probe's index
/// cardinalities as the atomic read's step 2 reports them
/// (`storage::structural_cache_sync::SyncIndexCardinalities::snapshot_cardinalities`),
/// `children_index` `None` on an unmarked object.
///
/// # Errors
///
/// As [`cold_start`].
pub async fn cold_start_probed(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    object_id: u64,
    client_id: &str,
    cards: SnapshotCardinalities,
) -> Result<ColdStart, ServiceError> {
    cold_start_for(
        client,
        keys,
        cache,
        object_id,
        client_id,
        None,
        Some(cards),
        None,
        None,
        None,
    )
    .await
}

/// Whether step 3's predictor sends a cold start whose probe found the item
/// bound `item_bound` to the plain read under `plain`: the bound within the
/// item ceiling, and the bound times
/// [`STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM`] within the byte ceiling.
/// Saturating, so a bound too large to price is never predicted to fit.
pub fn plain_read_predicted_to_fit(item_bound: u64, plain: SnapshotCeilings) -> bool {
    item_bound <= plain.max_items
        && item_bound.saturating_mul(STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM) <= plain.max_bytes
}

/// Whether a cold start bound for the fenced read of a layer estimated at
/// `estimate` is refused [`UncacheableReason::OverDeadline`] (module docs,
/// "The route", the deadline check): the fenced read's waiting cap under
/// `coefficients`, with the replica's whole configured RPC deadline
/// `rpc_deadline` left, is zero ([`wait_cap`]). It reads the layer's size
/// and the configuration only, never what is left of a call's deadline, so
/// its answer holds for every call on the replica.
pub fn fenced_read_over_deadline(
    coefficients: &FenceCoefficients,
    rpc_deadline: Duration,
    estimate: LayerSize,
) -> bool {
    wait_cap(coefficients, Some(rpc_deadline), estimate).is_zero()
}

/// Whether `size` is above either plain ceiling: what classifies a WATCH
/// fallback (module docs, "What the read side learns of the size").
fn above(size: LayerSize, plain: SnapshotCeilings) -> bool {
    size.items > plain.max_items || size.bytes > plain.max_bytes
}

/// The route's size estimate from a probe's cardinalities: the item bound
/// ([`snapshot_item_bound`]), and that bound times
/// [`STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM`] (saturating) as its bytes.
fn probe_estimate(cards: &SnapshotCardinalities) -> LayerSize {
    let bound = snapshot_item_bound(cards);
    LayerSize {
        items: bound,
        bytes: bound.saturating_mul(STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM),
    }
}

/// The size a probe alone knows a layer to be above the plain ceilings at:
/// [`probe_estimate`] when its item bound is over the plain item ceiling,
/// `None` otherwise, since a byte estimate is only a prediction (module docs,
/// "What the read side learns of the size"). The probed cold start's verdicts
/// carry it, and so does the read side's probe-time `NoChsComplete` verdict,
/// which answers before any cold start (`storage::structural_cache_read`).
pub(crate) fn probed_above_ceiling(
    cards: &SnapshotCardinalities,
    plain: SnapshotCeilings,
) -> Option<LayerSize> {
    let estimate = probe_estimate(cards);
    (estimate.items > plain.max_items).then_some(estimate)
}

/// The structural `UpdateLayer`'s upload commit charge, as a cold start
/// grows it at seam A (CR0; module docs), with a test's override of the
/// refused plain cold start's hold allowance.
#[derive(Clone, Copy)]
pub(crate) struct ColdStartCharge<'a> {
    charge: &'a dyn SharedCommitCharge,
    /// Replaces [`plain_cold_start_hold_allowance_bytes`] of the plain byte
    /// ceiling when set (`LayerStore::with_plain_cold_start_hold_allowance_for_test`).
    #[cfg(any(test, feature = "test-support"))]
    hold_allowance_for_test: Option<u64>,
}

impl<'a> ColdStartCharge<'a> {
    /// `charge`, with the production allowance.
    pub(crate) fn new(charge: &'a dyn SharedCommitCharge) -> Self {
        Self {
            charge,
            #[cfg(any(test, feature = "test-support"))]
            hold_allowance_for_test: None,
        }
    }

    /// **Test-only.** The same charge, with a refused plain cold start's
    /// hold allowance `allowance` bytes when `Some`.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn with_hold_allowance_for_test(mut self, allowance: Option<u64>) -> Self {
        self.hold_allowance_for_test = allowance;
        self
    }

    /// What a plain cold start whose charge was refused may hold beside its
    /// reply under the plain byte ceiling `byte_ceiling`:
    /// [`plain_cold_start_hold_allowance_bytes`], or a test's override.
    fn hold_allowance(&self, byte_ceiling: u64) -> u64 {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(allowance) = self.hold_allowance_for_test {
            return allowance;
        }
        plain_cold_start_hold_allowance_bytes(byte_ceiling)
    }
}

/// [`cold_start`] for the call on `layer`, which the verdict line of an
/// uncacheable verdict names ([`report_uncacheable`]); `None` leaves the
/// session and layer off the line. `probed` is the probe's cardinalities
/// ([`cold_start_probed`]), or `None` when no probe ran ([`cold_start`]).
/// `rpc_deadline` is the replica's configured RPC deadline, which the
/// deadline check judges a fenced-bound layer against (module docs, "The
/// route"); `None` when no deadline is in scope, which skips the check.
/// `remaining` reads the calling RPC's deadline left, once, when a fenced
/// read starts; `None` when no deadline is in scope. The read side's entry
/// point, which passes its probe's cardinalities, or `None` when the probe
/// found an index key of a foreign type and reported none. `commit_charge`
/// is the structural `UpdateLayer`'s upload commit charge, which `load`
/// grows at seam A (CR0; module docs), and against which a refused plain
/// read's hold is judged; `None` charges nothing, as for every caller but
/// that route.
//
// `clippy::too_many_arguments` allowed: the read side's request, unpacked.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn cold_start_for(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    object_id: u64,
    client_id: &str,
    layer: Option<VerdictLayer<'_>>,
    probed: Option<SnapshotCardinalities>,
    rpc_deadline: Option<Duration>,
    remaining: Option<&RemainingDeadlineFn>,
    commit_charge: Option<ColdStartCharge<'_>>,
) -> Result<ColdStart, ServiceError> {
    if cache.budget_bytes() == 0 {
        return Ok(ColdStart::Uncacheable {
            reason: UncacheableReason::Disabled,
            above_ceiling: None,
        });
    }
    // Registered before the first read, so a retirement or audit drop of the
    // object while it reads refuses the admission (module docs). Every early
    // return below drops it, releasing the registration.
    let ticket = cache.begin_cold_start();
    let started = Instant::now();
    let metrics = client.metrics_snapshot();
    let oid = object_id.to_string();
    let plain = SnapshotCeilings::of_client(client);
    let ctx = Ctx {
        client,
        keys,
        cache,
        metrics: metrics.as_ref(),
        object_id,
        layer,
        client_id,
        plain,
        started,
        commit_charge,
    };

    // Steps 1–3's decisions on the probe's cardinalities, before any full
    // read or permit (module docs, "The route"). `Some(estimate)`: straight
    // to the fenced read; `None`: the plain read first.
    let fenced_estimate = match probed {
        // No probe ran: the reads decide, the plain read first.
        None => None,
        Some(cards) => {
            let estimate = probe_estimate(&cards);
            let bound = estimate.items;
            // Only a bound over the item ceiling is known to be above the
            // ceilings; a byte estimate is a prediction.
            let known_above = probed_above_ceiling(&cards, plain);
            // Step 1: an unmarked object cannot enumerate its dangling lists
            // at any ceiling (SR-3.2).
            if cards.children_index.is_none() {
                return Ok(ctx.verdict(UncacheableReason::NoChsComplete, known_above));
            }
            // Step 2: the weight check (P7). A lower bound at the probe's
            // cardinalities: it refuses an admissible object only if the
            // object shrank since the probe or its index is corrupt, which
            // costs that object the uncacheable path, never a wrong entry
            // (module docs).
            let weight = state_weight_lower_bound(&cards);
            if weight > cache.max_entry_bytes() {
                return Ok(ctx.verdict(UncacheableReason::OverWeight(weight), known_above));
            }
            // Step 3's predictor (P5, P11).
            (!plain_read_predicted_to_fit(bound, plain)).then_some((estimate, known_above))
        }
    };

    let (estimate, known_above) = match fenced_estimate {
        Some(routed) => routed,
        None => {
            // Step 4: a plain permit, held across this one read and handed to
            // `load` with its snapshot (seam A drops it, or the put-back on a
            // refused charge); a `TOO_LARGE` answer gives it back here,
            // before the fenced read's permits.
            // Dropped on every exit, including `?` and cancellation.
            let permit = cache.acquire_plain_cold_start_read_permit().await;
            let plain_read = read_object_snapshot_slim_with_ceilings(
                client,
                keys,
                &oid,
                SlimSnapshotReadConsumer::StructuralCache,
                plain,
            )
            .await?;
            match plain_read {
                SnapshotReadOf::Snapshot(snapshot) => {
                    return load(
                        &ctx,
                        ticket,
                        *snapshot,
                        ColdStartRoute::Atomic,
                        ColdStartHold::plain(permit),
                    )
                    .await;
                }
                // A predictor miss (or a layer that grew since the probe), or,
                // unprobed, every layer above the plain ceilings.
                SnapshotReadOf::TooLarge(too_large) => {
                    drop(permit);
                    let size = LayerSize::from_too_large(&too_large, None);
                    // The read's own step-2 cardinalities report the marker:
                    // an unmarked object cannot enumerate its dangling lists
                    // at any ceiling (SR-3.2).
                    if too_large.cardinalities.children_index.is_none() {
                        return Ok(ctx.verdict(UncacheableReason::NoChsComplete, Some(size)));
                    }
                    // The weight check (P7) on the answer's cardinalities, before
                    // any fenced read: the unprobed path's first chance at it.
                    let weight = state_weight_lower_bound(&too_large.cardinalities);
                    if weight > cache.max_entry_bytes() {
                        return Ok(ctx.verdict(UncacheableReason::OverWeight(weight), Some(size)));
                    }
                    (size, Some(size))
                }
            }
        }
    };

    // Step 5: the deadline check, on every way to the fenced read — the
    // predictor's, and a plain read's `TOO_LARGE` answer. Judged at the
    // replica's configured RPC deadline, never at the call's deadline left:
    // a layer whose fenced read could not fit even a whole deadline is
    // refused for every call, so the verdict is recorded and later edits
    // route without a probe; a call that is merely short of time fails
    // below, for itself only. No deadline in scope, no check.
    if let Some(rpc_deadline) = rpc_deadline {
        let coefficients = coefficients(client);
        if fenced_read_over_deadline(&coefficients, rpc_deadline, estimate) {
            let over_deadline = OverDeadline {
                size: estimate,
                rpc_deadline,
                reserve: wait_reserve(&coefficients, estimate),
            };
            let reason = UncacheableReason::OverDeadline;
            record_verdict(
                cache,
                ctx.metrics,
                &Verdict::new(object_id, layer, reason).with_over_deadline(over_deadline),
                started,
            );
            return Ok(ColdStart::Uncacheable {
                reason,
                above_ceiling: known_above,
            });
        }
    }

    // Step 6: the fenced read, under the cold start's own hook. The deadline
    // left is read once, here.
    let mut hook = ColdStartFenceHook::new(client, cache, estimate, remaining.and_then(|f| f()));
    let read = read_object_snapshot_fenced(
        client,
        keys,
        object_id,
        SlimSnapshotReadConsumer::StructuralCache,
        estimate,
        &mut hook,
    )
    .await;
    match read {
        Ok(snapshot) => {
            let Some(hold) = hook.take_hold() else {
                return Err(ServiceError::Internal(format!(
                    "the fenced cold-start read of object {object_id} verified without holding \
                     its cold-start read permits"
                )));
            };
            drop(hook);
            load(&ctx, ticket, snapshot, ColdStartRoute::Fenced, hold).await
        }
        Err(error) => {
            if hook.failure() == Some(FencedFailure::Closed) {
                return Err(shutting_down());
            }
            let Some(cause) = failed_for_the_call(&hook, &error) else {
                return Err(error);
            };
            // For this call only (module docs): no negative entry, no route
            // tick; the capacity the hook held is given back first.
            // Known above the ceilings when the route said so before the
            // read, or the store said so during it (a completed fan-out's
            // measure, or the fence key's stored size): the largest of what
            // is known, so a stale smaller size never hides a larger one.
            let known = max_size(known_above, hook.learned());
            let above_ceiling = known.filter(|size| above(*size, plain));
            // The line reports the largest size the read knew, its own latest
            // included, so it names the size the fallback is classified at.
            let size = max_size(Some(hook.size()), known).unwrap_or(hook.size());
            let failure = FencedReadFailure {
                cause,
                attempts: hook.attempts(),
                size,
                waited: hook.elapsed(),
                cap: hook.wait_cap(size),
            };
            drop(hook);
            let reason = UncacheableReason::FencedReadFailed;
            report_uncacheable(
                ctx.metrics,
                cache,
                &Verdict::new(object_id, layer, reason).with_fenced(failure),
            );
            Ok(ColdStart::Uncacheable {
                reason,
                above_ceiling,
            })
        }
    }
}

/// One cold start's fixed inputs.
struct Ctx<'a> {
    client: &'a ValkeyClient,
    keys: &'a KeySchema,
    cache: &'a StructuralCache,
    metrics: Option<&'a SharedMetrics>,
    object_id: u64,
    layer: Option<VerdictLayer<'a>>,
    client_id: &'a str,
    plain: SnapshotCeilings,
    started: Instant,
    /// The structural `UpdateLayer`'s commit charge, grown at seam A.
    commit_charge: Option<ColdStartCharge<'a>>,
}

impl Ctx<'_> {
    /// Record and report `reason` as this cold start's verdict
    /// ([`record_verdict`]), answering with `above_ceiling`.
    fn verdict(&self, reason: UncacheableReason, above_ceiling: Option<LayerSize>) -> ColdStart {
        record_verdict(
            self.cache,
            self.metrics,
            &Verdict::new(self.object_id, self.layer, reason),
            self.started,
        );
        ColdStart::Uncacheable {
            reason,
            above_ceiling,
        }
    }
}

/// Turn a snapshot the `structural_cache` consumer read, with what its read
/// held, into the cold start's answer: a superseded or unmarked header, an
/// over-weight entry, an object invalidated since `ticket`, or the loaded
/// entry with the commit-token record. Gives `hold` back at seams A and B
/// (module docs, "The entry it builds").
async fn load(
    ctx: &Ctx<'_>,
    ticket: ColdStartTicket,
    mut snapshot: AtomicSnapshotOf<CompactState>,
    route: ColdStartRoute,
    hold: ColdStartHold,
) -> Result<ColdStart, ServiceError> {
    let Ctx {
        client,
        keys,
        cache,
        metrics,
        object_id,
        layer,
        client_id,
        plain,
        started,
        commit_charge,
    } = *ctx;
    let oid = object_id.to_string();
    let Some(facts) = snapshot.structural.take() else {
        return Err(ServiceError::Internal(format!(
            "the structural_cache snapshot read of object {object_id} carried no structural \
             facts: read_object_snapshot sets return_dangling for that consumer"
        )));
    };
    if facts.is_superseded() {
        // Nothing to load: everything the read held goes back before the
        // token read.
        drop(hold);
        drop(snapshot);
        return Ok(
            match read_commit_token(client, keys, &oid, client_id).await? {
                TokenRead::Record(commit_token) => ColdStart::Superseded { commit_token },
                TokenRead::WrongType(refusal) => ColdStart::CommitTokenRecordWrongType { refusal },
            },
        );
    }
    if !facts.chs_complete_marked() {
        drop(hold);
        let measured = snapshot.measured_size();
        return Ok(ctx.verdict(
            UncacheableReason::NoChsComplete,
            above(measured, plain).then_some(measured),
        ));
    }
    let ColdStartHold {
        read_permits,
        memory,
    } = hold;
    // Seam A, the charge point (CR0): the decoded entry enters the call's
    // upload commit charge — its whole state, and beside it the plan
    // indexes' build at its peak (`PlanIndexes::build_peak_bytes_for`, from
    // the decoded state's row, list-entry and value-row counts), never
    // waiting — while the read permits still bound the read, so no window is
    // outside both. Grow-only: a charge already holding more (the handler's
    // hint, a recharge's wait) keeps it until the attempt's peak. Then the
    // read permits end with the read. A refused charge still loads and
    // stores the entry below, then answers `NeedsCharge`: the call waits
    // holding nothing, and its re-run is a hit. On a refusal the entry is
    // held by no charge, so the read permits — and below, the fenced read's
    // grant — stay held until the entry is stored and put back, where the
    // cache's own budget counts it: the decoded entry is never outside all
    // of them while it is normalized, built and admitted. The one exception
    // is a refused plain read whose state and build are over its allowance,
    // below, which loads nothing.
    // `(target, hold)`: the refused charge's whole target, and the state
    // and build it would have held.
    let refused_charge = commit_charge.and_then(|charge| {
        let state_bytes = estimate_state_bytes(&snapshot.state);
        let build_peak_bytes =
            crate::domain::plan_index::PlanIndexes::build_peak_bytes_for(&snapshot.state);
        let need =
            ObjectStateNeed::cached_growing(state_bytes, build_peak_bytes, RewriteCount::default());
        let hold = state_bytes.saturating_add(build_peak_bytes);
        charge
            .charge
            .try_hold_at_least_need(&need)
            .err()
            .map(|target| (target, hold))
    });
    // A refused plain read holds its state and build under its read permits
    // only within the allowance the pod memory limit's cold-start term gives
    // it beside its reply (`plain_cold_start_hold_allowance_bytes`). Over
    // it, it loads nothing: the snapshot, the read permits and the ticket
    // drop here, before the state is normalized or the build runs, nothing
    // is stored or admitted, and the call waits for its target holding
    // nothing, then reads again — under its charge, which CR0 then grants.
    // No route ticks: a read that loads nothing is neither an entry nor a
    // verdict. A fenced read's refused hold is unchanged: its grant stays
    // held until the put-back.
    if let (Some((target, hold)), Some(charge), ColdStartRoute::Atomic) =
        (refused_charge, commit_charge, route)
    {
        let allowance = charge.hold_allowance(plain.max_bytes);
        if hold > allowance {
            tracing::debug!(
                session_id = layer.map(|l| l.session_id),
                layer_name = layer.map(|l| l.layer_name),
                object_id,
                hold_bytes = hold,
                allowance_bytes = allowance,
                target_bytes = target,
                "{}",
                COLD_START_HOLD_OVER_ALLOWANCE_MSG
            );
            drop(snapshot);
            drop(read_permits);
            drop(memory);
            drop(ticket);
            return Ok(ColdStart::NeedsCharge { target });
        }
    }
    let refused_charge = refused_charge.map(|(target, _)| target);
    let read_permits = match refused_charge {
        Some(_) => Some(read_permits),
        None => {
            drop(read_permits);
            None
        }
    };

    let measured = snapshot.measured_size();
    let cut = snapshot.cut;
    let AtomicSnapshotOf { mut state, .. } = snapshot;
    let mut dangling_parents: HashSet<u64> = facts.dangling_parents.into_iter().collect();
    dangling_parents.extend(normalize_state(&mut state));
    // The decoder collects each children list from a filtered iterator, so
    // a short list keeps spare capacity (a one-child list holds four slots).
    // The entry is the decoded state itself, not a copy of it, so trim the
    // lists once, here: `compact_state` (admit, put-back) trims tables, not
    // lists. From then on each list a commit or replayed record touches is
    // trimmed again as it is normalized (`domain::plan_index::normalize_touched`),
    // so the entry's lists stay exact. One pass over the lists per cold start.
    for kids in state.children.values_mut() {
        kids.shrink_to_fit();
    }
    // Seam B, the decode point: the snapshot was decoded straight into the
    // entry's compact state, so the memory counted in `T` for the read goes
    // now — after the put-back instead when CR0 was refused (above).
    let memory = match refused_charge {
        Some(_) => Some(memory),
        None => {
            drop(memory);
            None
        }
    };
    cache.before_admit(object_id);

    let entry = CacheEntry::new(object_id, facts.commit_chain, cut, dangling_parents, state);
    // Seam C: `admit` compacts and weighs the entry, the only place a
    // cold-started entry is weighed before the structural route's first
    // plan (`plan_in_place`), which the read side's `ReadyEntry` reaches after
    // the token `HGET` and the repair only.
    let entry = match cache.admit(ticket, entry) {
        Ok(entry) => entry,
        // Unreachable in practice: the budget never changes, and `cold_start`
        // answered a disabled cache before reading. Never a verdict, never
        // ticked.
        Err(AdmitRefused::Uncacheable(UncacheableReason::Disabled)) => {
            return Ok(ColdStart::Uncacheable {
                reason: UncacheableReason::Disabled,
                above_ceiling: None,
            })
        }
        // `admit` recorded the negative entry itself: `OverWeight`, the only
        // reason it records.
        Err(AdmitRefused::Uncacheable(reason)) => {
            observe(metrics, ColdStartRoute::Uncacheable, started);
            if cache.take_unreported(object_id).is_some() {
                report_uncacheable(metrics, cache, &Verdict::new(object_id, layer, reason));
            }
            return Ok(ColdStart::Uncacheable {
                reason,
                above_ceiling: above(measured, plain).then_some(measured),
            });
        }
        // Neither an entry nor a verdict: not ticked, no token read.
        Err(AdmitRefused::Invalidated) => return Ok(ColdStart::Invalidated),
    };
    observe(metrics, route, started);
    // CR0 refused, and a plain read's hold within its allowance (seam A): the
    // admitted entry is stored for the call's re-run, and only then do the
    // permits and grant it kept go. No token read: the re-run's repair
    // checks the token.
    if let Some(target) = refused_charge {
        entry.put_back_held();
        drop(read_permits);
        drop(memory);
        return Ok(ColdStart::NeedsCharge { target });
    }
    // On a granted charge both went at seams A and B.
    debug_assert!(read_permits.is_none() && memory.is_none());
    // On an error the guard drops and puts the entry back (module docs).
    match read_commit_token(client, keys, &oid, client_id).await? {
        TokenRead::Record(commit_token) => Ok(ColdStart::Loaded(ColdStartLoaded {
            entry,
            route,
            commit_token,
        })),
        TokenRead::WrongType(refusal) => {
            // A valid cut: kept for the next call, as on the probe's refusal.
            entry.put_back_held();
            Ok(ColdStart::CommitTokenRecordWrongType { refusal })
        }
    }
}

/// Record `verdict` as its object's negative entry, tick the `uncacheable`
/// route with the wall time since `started`, and report the verdict
/// ([`report_uncacheable`]), taking it first
/// ([`StructuralCache::take_unreported`]) so no lookup reports it again.
///
/// Also the read side's verdict for an object its probe found without
/// `chs_complete` (SR-4.3 step 0, `storage::structural_cache_read`): that
/// cold start ends at the probe, with no full read, and is counted here like
/// every other uncacheable verdict.
pub(crate) fn record_verdict(
    cache: &StructuralCache,
    metrics: Option<&SharedMetrics>,
    verdict: &Verdict<'_>,
    started: Instant,
) -> ColdStart {
    cache.record_uncacheable(verdict.object_id, verdict.reason);
    observe(metrics, ColdStartRoute::Uncacheable, started);
    // Taken at once: under the object's structural permit nothing else can
    // take it between the two critical sections, and should something, it is
    // still reported exactly once — by that taker.
    if cache.take_unreported(verdict.object_id).is_some() {
        report_uncacheable(metrics, cache, verdict);
    }
    ColdStart::Uncacheable {
        reason: verdict.reason,
        above_ceiling: None,
    }
}

/// Tick the cold-start metrics for one cold start that `route` ended.
fn observe(metrics: Option<&SharedMetrics>, route: ColdStartRoute, started: Instant) {
    if let Some(m) = metrics {
        m.observe_structural_cold_start(route.label(), started.elapsed());
    }
}

/// What [`read_commit_token`] found.
enum TokenRead {
    /// The call's record; `None` when the field or the key is absent.
    Record(Option<String>),
    /// The record key holds a foreign type: the commit backends'
    /// byte-identical `WRONGTYPE_KEY` refusal.
    WrongType(ServiceError),
}

/// The call's commit-token record: one same-slot `HGET obj:wtok <client_id>`
/// on the data pool, after the read (SR-4.3 step 1).
///
/// A record key of a foreign type answers the server's `WRONGTYPE`. The probe
/// before the cold start already refused such a key, so here it means the key
/// changed type since; the answer is the same as the probe's, the commit
/// backends' byte-identical `WRONGTYPE_KEY` refusal
/// ([`commit_token_record_wrong_type_refusal`]), which names the key's type. So
/// one `TYPE obj:wtok` follows, on this path only, to read it. Should that
/// find the key a hash or gone again — it changed type twice between the two
/// commands — the `HGET`'s own error is answered, classified as any other.
async fn read_commit_token(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    client_id: &str,
) -> Result<TokenRead, ServiceError> {
    let wtok_key = keys.obj_write_token(object_id);
    let error = match client
        .pool
        .hget::<Option<String>, _, _>(&wtok_key, client_id)
        .await
    {
        Ok(record) => return Ok(TokenRead::Record(record)),
        Err(error) => error,
    };
    let map = |e: fred::error::Error| {
        map_transient_fred_err(
            FredErrOrigin::client(client, POOL_LABEL_DATA),
            "HGET obj:wtok (structural cold-start commit token)",
            e,
        )
    };
    if !is_wrongtype_reply(&error) {
        return Err(map(error));
    }
    let key_type: String = client.pool.r#type(&wtok_key).await.map_err(|e| {
        map_transient_fred_err(
            FredErrOrigin::client(client, POOL_LABEL_DATA),
            "TYPE obj:wtok (structural cold-start commit token)",
            e,
        )
    })?;
    if key_type == "hash" || key_type == "none" {
        return Err(map(error));
    }
    Ok(TokenRead::WrongType(
        commit_token_record_wrong_type_refusal(&wtok_key, &key_type),
    ))
}
