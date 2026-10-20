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

//! Upload admission: one per-replica memory budget, `B` (derived from the pod
//! memory limit, `config::memory_plan`), that every `CreateLayer` and
//! `UpdateLayer` reserves from before it reads its content, plus one overflow
//! slot beyond it (SDD §5.4).
//!
//! An upload reads its header, is authorized (and, on `UpdateLayer`, its
//! target resolved), and only then calls [`UploadAdmissionGate::admit`]. The
//! returned [`UploadReservation`] is a local of the handler: it lives across
//! the body, the parse, the external-value store, every storage retry and the
//! recovery re-issue, and its `Drop` releases everything on every exit —
//! return, error, deadline, client reset and panic.
//!
//! **No upload under `LIVE_EDIT_MAX_LAYER_SIZE` is refused for its size.** The
//! budget bounds how many uploads run at once; an upload whose processing does
//! not fit runs in the **overflow slot**, one at a time, beside the in-budget
//! uploads. So the in-budget reservations never exceed `B`, at most one upload
//! is charged beyond `B`, and total upload memory is at most `B` plus that one
//! upload's charge. A structural `UpdateLayer` also holds memory for the
//! object it edits; its commit charges that too, before it allocates it (the
//! object-state charge, below), with three residuals on the WATCH path, all
//! charged after they arrive: names longer than any its read has seen yet, a
//! small set index's whole membership answering one index page, and a
//! hashtable-encoded set index's scan step overshooting its share by a few
//! members (below).
//!
//! **Charges.** Admission reserves `F + A`:
//!
//! - `F` ([`FIXED_CHARGE_BYTES`]) is a per-call fixed charge, so `B ÷ F` also
//!   caps how many uploads are admitted at once;
//! - `A` ([`RECEIVE_ALLOWANCE_BYTES`]) is the transport allowance: twice
//!   tonic's decode limit, held from admission to the end of the body.
//!
//! While the body is read the reservation grows by
//! [`RECEIVE_CHARGE_FACTOR`] × each chunk's length (the chunks as received
//! plus their one concatenation), accepted or refused at once
//! ([`UploadReservation::try_grow_in_budget`]). `B` is floored at
//! `F + A + 2 × LIVE_EDIT_MAX_LAYER_SIZE` ([`upload_budget_floor_bytes`]), so
//! the receive phase of any upload under the size cap fits an empty budget. At
//! the end of the body [`UploadReservation::end_receive`] releases `A` and the
//! session slot, and the reservation shrinks to `F` + body length.
//!
//! Then two processing steps, each through
//! [`UploadReservation::reserve_processing`]:
//!
//! 1. before the delta is parsed, [`parse_charge`]: `F` + body + `e_ext` ×
//!    external-value bytes + `k_parse` × delta bytes;
//! 2. after the parse, once the commit path is known, [`processing_charge`]:
//!    the larger of the parse charge and `F` + body + `e_ext` × external-value
//!    bytes + `byte[path]` × delta bytes + `element[path]` × delta elements +
//!    `reorder[path]` × reorder-list entries ([`CommitPath`],
//!    [`DeltaFootprint`]). Decoded memory scales with the number of delta
//!    elements, not with the bytes they arrived in. The handlers run the
//!    per-delta checks (§9.4 caps, name characters, sample times,
//!    self-parenting) on the parsed delta before this step, so a delta that
//!    fails them is refused `INVALID_ARGUMENT` holding only the parse charge,
//!    never the processing charge.
//!
//! **The object-state charge.** A structural `UpdateLayer`'s commit holds
//! memory that grows with the object it edits, not the delta: on the cached
//! route the held cache entry the plan runs on in place, its stream repair's
//! reads and replay, the tables the plan grows, the rows and lists it
//! rewrites and the sections its depth check walks; on the WATCH path the
//! attempt's fixed cost, its whole-object read, the rewrite and the commands
//! its transaction queues. [`object_state_charge`] prices it
//! ([`cached_object_state_charge`], [`repair_transient_charge`],
//! [`watch_object_state_charge`]) from the counts the commit reports as an
//! `ObjectStateNeed`, at constants T7 fitted, each route's rewrite at its own
//! prices ([`RewritePrices`]); the WATCH read's three are still
//! provisional (below). The stream repair's terms — its two reply terms
//! ([`REPAIR_REPLY_ENTRY_BYTES`], [`REPAIR_REPLY_BYTE_CENTI`]) and its growth
//! term ([`REPAIR_GROWTH_CENTI`]) — are fitted by T7 from the fail-closed
//! fits in `docs-internal/validation/repair-charge-t7-b-fsz-20261010/`
//! (`fits.json`). The handler takes a hint of it
//! with the second processing step (`LayerStore::object_state_hint`); the
//! commit then holds it on top of the processing charge through the storage
//! seam [`CommitCharge`], implemented here by [`UploadCommitCharge`] over
//! [`UploadReservation::try_hold_processing`], a processing step that never
//! waits, at each charge point before the memory it covers. Until the
//! attempt's peak every charge point only grows the charge, the stream
//! repair's (before each of its reads, before replaying each and before
//! applying each record) and the
//! plan's included (it asks for more as its rewrite grows); after it, the commit
//! trims to its exact need once (the cached route's CR2, the WATCH path's
//! WA3); CR2's need keeps, beside the planned entry, the table rebuilds the
//! plan's rollback could still need. A refused growth makes the commit
//! release everything it holds and
//! wait in [`UploadReservation::reserve_processing`], holding nothing else,
//! before it resumes.
//!
//! The structural cache's sampled audit holds its repair of the entry's copy
//! in the same budget, priced the same way less the copy's own weight,
//! through its own holder ([`UploadBudgetAuditCharges`]): no admission, never
//! waiting, never on the overflow slot, refused while an upload is queued or
//! when it would leave less than one admission's room, and capped at
//! [`SAMPLED_AUDIT_CHARGE_CAP_BYTES`]. A refused audit skips.
//!
//! **The residuals** (SDD §5.4). The WATCH path's read charges each window
//! and index page before it is issued, from per-command reply bounds that
//! size every name at the longest the read has seen, at least
//! `diff_read::NAME_BYTES_SIZING_FLOOR` (64 B). A set index's `SSCAN` step is
//! charged for its whole share, at least what its `COUNT` asks for, and at
//! least twice that for a share above one. Three kinds of reply can exceed
//! what was charged before they were issued, and are charged after they
//! arrive:
//!
//! - **Names** (outside §21.2's line, in the unsafe direction, until its own
//!   Issue lands). A page or window of names longer than that is charged
//!   after it arrives, one at a time per attempt: at most 32,715,522 B for a
//!   field-index page and 8,404,709 B for a section window at
//!   `MAX_NAME_BYTES` (4,096), and up to the transaction pool's attempts at
//!   once (about 2.09 GB at the chart's 64) [code + model].
//! - **A compact-encoded set index.** A round-1 `SSCAN` step on a set the
//!   store keeps in its compact encoding (intset or listpack) answers the
//!   set's whole membership whatever its `COUNT`, which can exceed the share
//!   charged for it. That set's scan then ends, so it happens at most once
//!   per set index per read: at most max(`set-max-intset-entries`,
//!   `set-max-listpack-entries`) × 20 B (a decimal id) per index — 10,240 B
//!   at the store's defaults (512 and 128) — and 30,720 B per read for the
//!   three set indices (sections, deleted sections, `obj:chs`) [code +
//!   Valkey's documented defaults]. An operator who raises those limits
//!   raises it in proportion.
//! - **A hashtable-encoded set index's step overshoot** (small). A round-1
//!   `SSCAN` step on a set the store keeps as a hashtable can return more
//!   than its charged share: the store walks the table until it has
//!   collected `COUNT` members, and each bucket it visits adds its whole
//!   chain — during a rehash, also every bucket of the larger table that the
//!   smaller table's bucket expands to [Valkey 8.0 `scanGenericCommand`,
//!   `src/db.c`]. Such a step can exceed its share by a few members ×
//!   `diff_read::ID_REPLY_BOUND` (20 B). Over a read it never exceeds the
//!   set's own size × 20 B, but no constant in the code bounds it.
//!
//! **Waiting.** Only two calls wait, and neither waits on anything that is
//! itself waiting on the gate:
//!
//! - [`UploadAdmissionGate::admit`] waits holding nothing from the gate, for at
//!   most the admission wait. The queue is first-come-first-served: a waiter
//!   held back only by its own session's cap is skipped, and a waiter held
//!   back by bytes keeps its place and holds those behind it (all admissions
//!   need the same `F + A`).
//! - [`UploadReservation::reserve_processing`] keeps the in-budget bytes it
//!   already holds while it waits, and waits only for room in `B` or for the
//!   overflow slot, whose holder never waits on the gate (it may wait outside
//!   it, at the external-value release budget: SDD §5.4, and the chained
//!   `UpdateLayer` paragraph below). It grows in budget when the target fits;
//!   otherwise it takes the free overflow slot if no processing waiter is
//!   queued; otherwise it queues for whichever comes first — room in `B`, or
//!   reaching the head of the processing queue with the slot free. It
//!   is bounded by the RPC deadline (the handler future is dropped) and races
//!   the token its caller passes: the handlers, a commit's recharge through
//!   [`UploadCommitCharge`]'s [`CommitCharge::hold`], and the `UpdateLayer`
//!   handler's hold of one extra copy of the external values before the
//!   checkpoint-recovery re-issue's second pre-store (through the same
//!   [`CommitCharge::hold`], holding nothing else; given back, without a
//!   wait, before the re-issue's first attempt) pass the graceful shutdown's
//!   `begun` token, cancelled when the paced stream ending starts.
//!
//! Every other charge is accepted or refused at once —
//! [`UploadReservation::try_hold_processing`] included: it takes
//! [`UploadReservation::reserve_processing`]'s steps that need no wait and is
//! refused, with nothing changed, where that would queue. The slot holder never
//! waits on the gate — it grows in place — so the slot always frees, and the
//! slot goes to processing waiters in FIFO order, so every processing waiter
//! is granted after at most the processing of the slot holders ahead of it.
//! On every release the gate first grants each processing waiter that now fits
//! in `B`, then gives a free slot to the processing-queue head (whose bytes
//! leave `B` with it, so the in-budget grants run again), then serves
//! admissions: work in progress goes before new work. That is why the gate
//! cannot deadlock even though processing waiters hold bytes in `B`, and why a
//! small arrival cannot overtake a slot waiter.
//!
//! A chained `UpdateLayer` (`UpdateLayerRequest.chained_header`) also waits,
//! outside the gate: in the chain gate (`storage::chain_gate`), between the
//! end of its body and its first processing step, for its predecessor's
//! commit. It holds its reservation there — `F` plus its content — and
//! nothing else of the gate: no processing charge, no overflow slot. It waits
//! on no permit the gate grants, only on a commit, so it adds no edge to the
//! gate's waits. Its bytes do stay in `B` while it waits: if waiting chained
//! calls fill `B`, a predecessor's admission waits at most the admission wait
//! and is then refused `UPLOAD_CAPACITY` with nothing applied (a receive-growth
//! charge that does not fit is refused at once), and a client cancel of the
//! waiters frees their bytes at once. Past admission, a predecessor's
//! processing step, and its commit's recharge between attempts
//! ([`CommitCharge::hold`]), are never refused for capacity: each waits for
//! room in `B` (which waiting successors hold) or for the overflow slot. The
//! slot's holder may itself be waiting at the external-value release budget
//! (SDD §5.4): the `CreateLayer` and `UpdateLayer` handlers await it after
//! both processing steps, holding their processing charge, and so the slot if
//! they run in it. So the predecessor, and the chain waiting behind it, can
//! wait up to that holder's release-budget wait. It is still deadlock-free,
//! because there the slot's holder waits only on GC, never on a chained call.
//!
//! **Fairness.** Each session may have at most
//! [`UploadAdmissionGate::per_session_cap`] uploads still receiving their
//! content: `max(1, ⌊B ÷ 4A⌋)`, derived from `B`.
//!
//! **Lock order.** Admission and both processing steps come before the
//! external-value store, the per-object write semaphore, pool slots, and every
//! other permit an upload takes later. Nothing awaits the gate while holding
//! a per-object permit, a transaction checkout, a cold-start permit or grant,
//! or a checked-out cache entry: growth taken while holding any of them never
//! waits ([`UploadReservation::try_hold_processing`]), and a refused growth
//! releases them all before the call waits
//! ([`UploadReservation::reserve_processing`], through
//! [`CommitCharge::hold`]). A chained `UpdateLayer`'s chain-gate wait sits
//! between admission and the first processing step, so it holds no permit an
//! upload takes after it.
//!
//! **Processing-wait observations.** `live_edit_upload_processing_wait_seconds`
//! records one observation per waiting step: each
//! [`UploadReservation::reserve_processing`] call that grows the
//! reservation, the handler's two steps, a commit's recharge, and the
//! `UpdateLayer` handler's hold of one extra copy of the external values
//! before the checkpoint-recovery re-issue's second pre-store (through
//! [`CommitCharge::hold`]; one per re-issue with external values). The commit's
//! non-waiting growth ([`UploadReservation::try_hold_processing`]) records
//! none, so the histogram stays one unit, a real wait.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::domain::delta::{own_element_count, MutationAllowance, RewriteCount};
use crate::error::{
    refusal_metadata, refusal_reason, RefusalDetail, ServiceError, CAPACITY_RETRY_DELAY,
};
use crate::grpc::graceful_shutdown::GracefulShutdown;
use crate::observability::metrics::{
    SharedMetrics, ADMISSION_RESULT_ADMITTED, ADMISSION_RESULT_CANCELLED,
    ADMISSION_RESULT_REFUSED_GROWTH, ADMISSION_RESULT_REFUSED_SATURATED,
    ADMISSION_RESULT_REFUSED_WAITER_CAP, UPLOAD_PHASE_PROCESSING, UPLOAD_PHASE_RECEIVING,
    UPLOAD_RPC_CREATE_LAYER, UPLOAD_RPC_UPDATE_LAYER,
};
use crate::storage::commit_charge::{
    AuditCharge, AuditChargeSource, CommitCharge, ObjectStateNeed, QueuedCommands, RepairTransient,
    SharedCommitCharge,
};

const MIB: u64 = 1024 * 1024;

/// Stand-in end for a wait the clock cannot represent (about 30 years).
const FAR_FUTURE: Duration = Duration::from_secs(86_400 * 365 * 30);

/// The upload budget `B` at the reference pod memory limit, 384 MiB: 7.5 % of
/// 5,120 MiB (`config::memory_plan`). What a `Config` built field by field,
/// rather than derived from a limit, sets its upload budget to.
pub const UPLOAD_MEMORY_BUDGET_MIB_DEFAULT: u64 = 384;

/// The default `LIVE_EDIT_MAX_LAYER_SIZE`: 104857600 (100 MiB). The budget's
/// default and floor are derived from the configured size cap.
pub const MAX_LAYER_SIZE_DEFAULT_BYTES: u64 = 104_857_600;

/// The ceiling of the derived `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS` default:
/// 2000 ([`upload_admission_wait_ms_default`]). A waiting upload leaves its
/// unread content in the connection's flow-control window, so the wait bounds
/// how long it can stall other streams sharing that connection.
pub const UPLOAD_ADMISSION_WAIT_MS_DEFAULT: u64 = 2_000;

/// The `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS` default for a given
/// `LIVE_EDIT_RPC_DEADLINE_MS`: `min(`[`UPLOAD_ADMISSION_WAIT_MS_DEFAULT`]`,
/// rpc_deadline_ms / 2)`. Below every non-zero deadline, so an operator who
/// tightens the deadline never has to set the wait too; it is 0 — never
/// wait — only for a deadline under 2 ms.
pub const fn upload_admission_wait_ms_default(rpc_deadline_ms: u64) -> u64 {
    let half = rpc_deadline_ms / 2;
    if half < UPLOAD_ADMISSION_WAIT_MS_DEFAULT {
        half
    } else {
        UPLOAD_ADMISSION_WAIT_MS_DEFAULT
    }
}

/// Default `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`: 64, which holds the queued
/// uploads' unread content to about 64 MiB.
pub const UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT: usize = 64;

/// tonic's decode limit for one request message. Nothing in this crate calls
/// `max_decoding_message_size`, so it is tonic's unconfigured default (4 MiB).
/// tonic buffers a whole message before decoding it, and prost copies the
/// chunk's bytes out of that buffer.
pub const TONIC_DEFAULT_DECODE_LIMIT_BYTES: u64 = 4 * MIB;

/// `A`: the transport allowance an upload holds from admission to the end of
/// its body — twice [`TONIC_DEFAULT_DECODE_LIMIT_BYTES`], for one message
/// buffered by tonic and its decoded copy.
pub const RECEIVE_ALLOWANCE_BYTES: u64 = 2 * TONIC_DEFAULT_DECODE_LIMIT_BYTES;

/// While the body is read, each received byte is charged this many times: the
/// chunk as received, and its share of the one exact-size concatenation made
/// at the end of the stream.
pub const RECEIVE_CHARGE_FACTOR: u64 = 2;

// The charge constants below are set by the T7 memory envelope measurement
// (`tests/upload_memory_envelope/`), with no headroom beyond its × 1.25. T7
// measures a debug build and counts the Rust heap only; its report prints each
// run's figures and the constants that one run recommends, and its envelope
// assertion fails when the production constants under-cover an upload.
//
// The derivation rule (the one statement of it; the SDD, T7 and its report
// point here). T7 fits each path's terms in a chain, each net of this run's own
// `F` (the path's header-only peak) and body: `byte`, then `element` net of
// `byte`, then `reorder` net of `byte` and `element`, then `e_ext` net of
// `byte` and `element`. Each chained term is netted against, for every earlier
// term, the larger of this run's figure and that term's production basis (the
// production constant ÷ 1.25, or ÷ 125 for a hundredths constant), so a run
// that measures an earlier term low cannot inflate a later one. Because one
// run's figures vary (the slow path's long-reorder peak varies by a few
// hundred bytes across runs, and before the connection-buffer limits took one
// of two values, `reorder[UpdateSlow]` below), re-derive the constants when
// an upload path changes: run T7 several times under nextest, and set each
// constant to ⌈its largest netted figure over those runs × 1.25⌉ (the report's
// `netted` line; `recommended` is that × 1.25, rounded up; `k_parse` plus
// 100). A whole-binary `cargo test` is not needed: its netted figures have
// measured no higher than nextest's (the slow path's structure-dense element
// and `Create`'s `e_ext` lower, the rest equal), and a lower earlier figure
// cannot raise a later term under this netting. When an earlier term's
// constant changes, re-run T7 before deriving the terms after it, which are
// netted against its production basis. Each constant's doc records its
// nextest figure. A constant whose doc states an adoption condition
// ([`WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES`]'s T7 bound check: every WATCH read
// segment and WA3, on both transports, at or under its commit-only charge)
// is wired at ⌈largest × 1.25⌉ only if that value meets the condition, and
// otherwise at the smallest value that does; that constant's doc records
// which.
//
// `UpdateSlow` covers every commit route charged `CommitPath::UpdateSlow` —
// structural deltas and resets on any backend, every delta under `txn` — and
// each `UpdateSlow` constant is its largest netted figure over all of them: T7
// measures the path on the `lua` backend's cached structural route, from a
// cold start and with the object already cached, and on the WATCH/MULTI/EXEC
// path under `txn`, which also takes every field and time-sample delta there
// (`lua-watch-structural` and `lua`'s fallbacks commit structural deltas on
// the same WATCH code). `F` is the largest fixed cost over every path and
// route: each route's header-only peak, and on `UpdateSlow` also the peak of
// a one-child structural edit net of its body, byte and element terms (an
// empty `UpdateLayer` commits on the fast path under `lua`, so only that edit
// reaches the cached structural route's fixed cost). So `F` and each
// `UpdateSlow` constant come from the most expensive route measured,
// whichever backend is configured; `txn` stays in the derivation by decision,
// though it is a rollback value. The per-element term is netted of each
// route's own object-state need at its peak, which the commit charges beside
// the processing charge (the cached route's held entry, growth, rewrite and
// walk; the WATCH path's queued commands), so a route's object-state memory
// is not counted twice. Today the WATCH path under `txn` binds `F` (its
// one-child structural edit), the cached route's cold start binds
// `byte[UpdateSlow]` and has the largest netted `element[UpdateSlow]`
// figure, which the constant, only ever raised, sits above, and
// `reorder[UpdateSlow]` sits above every route's figure; under the default
// `lua` backend the WATCH-bound `F` therefore charges more than its own
// routes need. T7's
// floor-binding note prints only on the route that binds the term (its
// `Route::binds`, which follows these bindings), since on the others the
// constant is set above their figure by design. Under raise-only a binding
// route's own recommendation can sit below its constant without anything
// being wrong: today the cached cold start recommends byte 714 against the
// kept 840 (its figure has fallen since the constant was set), and
// `UpdateFast`'s byte figure recommends 586 or 587 by run.
//
// The object-state terms continue the `UpdateSlow` chain after `e_ext`:
// `queued` (the WATCH commit's queued commands, from stalled dense fixtures,
// netted against the cached route's element basis), then `element` re-netted
// (since the cached route plans in place, from its element-dense edits, each
// route's figure net of its own object-state need at its peak), then the
// cached route's in-place rewrite terms — `rewrite_list` (each charge
// point's segment of a one-child edit with the object already cached, net of
// the segment before the first charge point and of its growth term, per list
// entry, at several base sizes and both content shapes), `rewrite_row` (from
// moving the first child of a long list to the end, every key changing,
// netted against one new child on the same base) and `rewrite_walk_entry`
// (the depth check's walks measured alone, per section walked) — then
// `rewrite_name_index` (the index segment of a two-probe edit, attributed by
// a peak-time allocation tracker; netting the edit's commit peak against the
// one-probe edit's on the same base reads less, since the one-probe commit
// peaks after its plan's index is gone), then `rewrite_adoption_entry` (the
// recording segment of a live re-key plus a birth; netting the commit peak
// against the same delta re-keying a childless section, matched run to run,
// reads less: 13.1 B per id) — both measured when the cached route still
// built those indexes per plan, and now pricing the WATCH path's plan only —
// then the WATCH attempt's fixed cost (the empty-base edit's processing
// phase net of its processing charge's terms — `F`, the body, `byte` and
// `element` — each at its production basis, or at the run's own figure where
// the run gives one and it is larger (`element[UpdateSlow]`)), then the WATCH
// snapshot factors (`W_NAME_BYTE` from the named
// fixtures, then `W_CMD`, `W_ENTRY`, `W_ITEM`, `W_BYTE`). They charge the
// memory a structural `UpdateLayer` holds for the object it edits, which
// grows with the object rather than the delta ([`object_state_charge`]); the
// WATCH snapshot factors are netted of the rewrite terms, from bases of at
// least three sizes in two content shapes at two windows. The netted fits
// are netted against the earlier terms' production basis as above. Fitted by
// this rule, each doc naming its section of the fit record that holds it:
// `queued`, `rewrite_name_index`, `rewrite_adoption_entry` and `W_NAME_BYTE`
// over 4 or 5 runs
// (`docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`);
// `element[UpdateSlow]`, `element[UpdateFast]`, the cached route's
// `rewrite_list`, `rewrite_row` and `rewrite_walk_entry`, and the WATCH
// path's `rewrite_list`
// (`docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`);
// the WATCH attempt's fixed cost (`W_FIXED`) and the per-command snapshot
// factor (`W_CMD`, on the window-script transport; the per-plain-command
// factor keeps the fit of the read when it was all plain) over 24 runs per
// window transport
// (`docs-internal/validation/watch-window-script-b-gvf-20261010/`,
// `results.json`). `element[UpdateSlow]` and the WATCH path's `rewrite_row`
// are netted of `queued`, so they hold only with it wired, and
// `element[UpdateSlow]` is netted of the cached route's rewrite prices, so it
// holds only with them wired. `W_CMD` is fitted from the WATCH read with the
// provisional snapshot factors at their wired values, since T7 cannot yet
// separate them, so it carries the read's whole excess. Three are wired
// above the rule's figure, each doc giving both: `element[UpdateSlow]`, kept
// at 3,476 as constants are only raised; the WATCH path's `rewrite_row`,
// kept at the 4,376 its snapshot factors were netted of; and
// `rewrite_adoption_entry`, wired at 33 against the rule's 30, to cover one
// cross-run pairing of its commit peak. The cached route's stream-repair
// terms are fitted by T7 in fits that fail closed: `repair_reply_byte`
// (`b`) from the single-entry repair reads, then `repair_reply_entry` (`a`)
// from the multi-entry reads net of `b`'s largest figure, each read net of
// `F`'s basis; and `repair_growth`, which this rule's netting does not
// identify (it nets to 0), from a direct per-record measure of each
// replayed record's peak net of the terms RP2 prices beside it. Each is
// ⌈its largest figure × 1.25⌉ over every row and run the record's manifest
// names, and its fit script refuses to fit when a member is missing
// (`docs-internal/validation/repair-charge-t7-b-fsz-20261010/`,
// `fits-manifest.json`, `scripts/fit_repair.py`, `fits.json`).
// **The other values are provisional** (`W_ENTRY`, `W_ITEM` and `W_BYTE`):
// set conservatively, at or above every figure each constant's doc records,
// not yet fitted by T7, which must fit them by this rule. A provisional
// constant is lowered to its fit only once repeated runs confirm no run
// exceeds it, and never below its largest measured figure × 1.25.

/// `F`: the fixed per-call charge, in bytes — the largest fixed cost over
/// every commit path and route (derivation rule above). Set by T7 from the
/// most expensive route, measured on a debug build with the Rust heap counted:
/// 72,160 B, a one-child structural edit on the WATCH path under `txn`, net of
/// its body, byte and element terms (71,354 B in most runs) — the route that
/// binds. Elsewhere: an empty `UpdateLayer` under `txn` (WATCH) 43,734 B; an
/// empty `UpdateLayer` on `lua`, which takes the fast path, 32,581–38,554 B
/// (the structural-edit release raised it from 37,818 B); the cached
/// structural route's one-child edit, net of its terms, 45,969 B from a cold
/// start and 45,077 B with the object cached; `Create`'s header-only upload
/// 35,851 B. The budget floor ([`upload_budget_floor_bytes`]), the admission
/// charge and the startup log ([`log_upload_admission_startup`]) read it.
pub const FIXED_CHARGE_BYTES: u64 = 90_200;

/// The per-byte factors below are in hundredths: a factor of `250` charges
/// 2.5 bytes per byte of the part it applies to, rounded up.
pub const EXPANSION_SCALE: u64 = 100;

/// `e_ext`, in hundredths: processing memory per byte of the OMPF's
/// external-value parts (the one owned copy shared by the store and the
/// mirror, and the Valkey client's write buffer), net of the delta terms. One
/// store's worth: the `UpdateLayer` checkpoint-recovery re-issue's second
/// pre-store holds one more copy, which its handler charges on top for that
/// store only ([`UploadCommitCharge`]). Set
/// by T7 from the blob fixtures (derivation rule above), measured on a debug
/// build with the Rust heap counted: 2.466 per byte, `UpdateFast` (`Create`
/// 2.003; `UpdateSlow` 2.158 on the WATCH path, 2.001 on the cached route).
pub const EXT_EXPANSION_CENTI: u64 = 309;

/// `k_parse`, in hundredths: memory per byte of the OMPF's delta part while
/// the delta is parsed — the cxx bridge's intermediate commands and the owned
/// delta they convert into — plus 100 for the C++ parser's transient
/// temporaries (one string or reorder list at a time, never more than the
/// delta part itself), which the Rust heap count cannot see. Set by T7's
/// parse-only pass (derivation rule above), measured on a debug build with the
/// Rust heap counted: 11.901 per byte, the structure-dense fixture;
/// [`parse_charge`] reads it.
pub const PARSE_EXPANSION_CENTI: u64 = 1_588;

/// `element[Create]`, in bytes per delta element (struct command, set field or
/// time sample) on `CreateLayer`: the parsed element, its `ObjectState` row
/// and the materialization commands queued for it. Set by T7 from the dense
/// fixtures, net of the per-byte term (derivation rule above), measured on a
/// debug build with the Rust heap counted: 6,405.9 B; [`processing_charge`]
/// reads it.
pub const ELEMENT_CHARGE_CREATE_BYTES: u64 = 8_008;
/// `element[UpdateFast]`, in bytes per delta element on the Lua fast path: the
/// parsed element and its script arguments. Set by T7 as
/// [`ELEMENT_CHARGE_CREATE_BYTES`] is: 1,451.3 B, the field-dense fixture at
/// 20,000 elements, identical over 4 runs under nextest (1,408.4 B at 30,000
/// elements, within 5 %, so not a buffer's step; the time-sample-dense
/// fixture 1,263.2 B and 1,232.6 B)
/// (`docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`,
/// § `element[UpdateFast]`); ⌈× 1.25⌉ is 1,815. It was 1,707, from 1,365.3 B
/// when last fitted; the rise is not attributed to one change.
pub const ELEMENT_CHARGE_UPDATE_FAST_BYTES: u64 = 1_815;
/// `element[UpdateSlow]`, in bytes per delta element on every route charged
/// `UpdateSlow`: the parsed element, and on the cached structural route its
/// plan's commands and the plan journal's record of the element's first
/// touch, on the WATCH/MULTI/EXEC path one attempt's clone and the commands
/// queued for it. The journal holds that record until the commit settles —
/// a set field's slot in the journal's per-section map, that map, the
/// field's boxed name and its pre-image; a time sample's slot and pre-image;
/// the pre-image of each row a create, a diff-change or a move writes — and
/// no rewrite or growth term prices it (`RewriteCount::rows` excludes the
/// delta's own elements), so this term covers it. Set by T7 as
/// [`ELEMENT_CHARGE_CREATE_BYTES`] is (the `UpdateSlow` routes in the
/// derivation rule above), each route's figure netted of the object-state
/// need that route holds at its peak, which the commit charges beside this
/// term: the WATCH path's of its queued commands (as since `queued` got its
/// own term), the cached route's of its held entry, growth and rebuilds 1:1
/// and its rows, list entries and walked sections at their prices ÷ 1.25.
/// The cached route's cold start is the largest — 20,000 new children,
/// 2,543.4 B per element (2,951.8 B before the netting; 2,542.3 B with the
/// object cached), above one new child with 20,000 new fields (2,133.9 B),
/// overwriting 20,000 existing fields (2,134.9 B) or with 20,000 time
/// samples (1,773.1 B); the WATCH path's at most 1,900.2 B (20,000 new
/// children, its transaction queue's slots netted, as `queued` was fitted)
/// (`docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`,
/// § `element[UpdateSlow]`); ⌈× 1.25⌉ is 3,180, and the constant is kept at
/// 3,476 (2,780.6 B when the cached route planned on a copy) as constants are
/// only raised. The netting holds only with the cached route's rewrite
/// prices wired at the values it was netted of
/// ([`CACHED_REWRITE_ROW_CHARGE_BYTES`] 4,574,
/// [`CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES`] 287,
/// [`REWRITE_WALK_ENTRY_CHARGE_BYTES`] 124) and its in-place growth term
/// charged (un-netted, the cached route needs 3,690). Valid only with
/// [`QUEUED_COMMAND_CHARGE_BYTES`] wired at 1,896 or above: before it, the
/// WATCH path's field-dense fixture bound at 8,438.6 B with its queue
/// counted in.
pub const ELEMENT_CHARGE_UPDATE_SLOW_BYTES: u64 = 3_476;

/// `byte[Create]`, in hundredths: memory per byte of the delta part beyond its
/// elements (inline values) on `CreateLayer`. Set by T7 from the value-heavy
/// fixture, before the element term (derivation rule above), measured on a
/// debug build with the Rust heap counted: 2.454 per byte;
/// [`processing_charge`] reads it.
pub const DELTA_BYTE_CHARGE_CREATE_CENTI: u64 = 307;
/// `byte[UpdateFast]`, in hundredths. Set by T7 as
/// [`DELTA_BYTE_CHARGE_CREATE_CENTI`] is: 4.6884 per byte, in the runs whose
/// header-only peak, which it is net of, measured 36,866 B (4.6880 when it
/// measures 38,554 B).
pub const DELTA_BYTE_CHARGE_UPDATE_FAST_CENTI: u64 = 587;
/// `byte[UpdateSlow]`, in hundredths. Set by T7 as
/// [`DELTA_BYTE_CHARGE_CREATE_CENTI`] is, from the slow path's own
/// value-heavy fixture, from the most expensive route: 6.713 per byte, the
/// cached structural route's cold start, which binds — 6.708 since T7 runs
/// that route's code once before measuring, a once-per-process cost the
/// earlier figure carried, kept at 840 as constants are only raised (warm
/// 6.707; the WATCH path under `txn` 5.470).
pub const DELTA_BYTE_CHARGE_UPDATE_SLOW_CENTI: u64 = 840;

/// `reorder[Create]`, in bytes per reorder-list entry: zero. Set by T7 from
/// the long-reorder fixture, net of the byte and element terms (derivation
/// rule above), measured on a debug build with the Rust heap counted: 0.0 B.
pub const REORDER_ENTRY_CHARGE_CREATE_BYTES: u64 = 0;
/// `reorder[UpdateFast]`: zero — a reorder never takes the fast path, so T7
/// has nothing to measure.
pub const REORDER_ENTRY_CHARGE_UPDATE_FAST_BYTES: u64 = 0;
/// `reorder[UpdateSlow]`, in bytes per reorder-list entry: each entry rewrites
/// the reordered child's section hash. Set by T7 as
/// [`REORDER_ENTRY_CHARGE_CREATE_BYTES`] is: set from 5,260.4 B, the WATCH
/// path before the structural-edit release, and kept. That path's
/// long-reorder peak then took one of two values across runs, a connection
/// whose buffers had already grown against a fresh one: since the
/// connection-buffer limits (`storage::connection_buffers`) the fixture
/// regrows its buffers on every call, and five runs peaked at 275,476,243 B
/// four times and 275,476,545 B once. On the current routes the largest
/// netted figure is 3,689.3 B (the WATCH path; the cached route nets to 0).
/// Reorders of a layer's existing children cost per rewritten row, not per
/// entry: a short list can change every child's key (before the
/// connection-buffer limits, moving the first of 20,000 children to the end
/// peaked at 132.6 MB on the WATCH path against 65.1 MB for one new child on
/// the same base; `CACHED_REWRITE_ROW_CHARGE_BYTES` and
/// `WATCH_REWRITE_ROW_CHARGE_BYTES` have the current per-row figures). The commit charges those rows, and the object's state the route
/// holds, through the object-state terms below, not through this one.
pub const REORDER_ENTRY_CHARGE_UPDATE_SLOW_BYTES: u64 = 6_576;

// The object-state terms below are wired: every structural commit charges
// them ([`object_state_charge`]), so none may sit below a figure measured for
// it (the derivation rule above). Most are fitted by T7 on the compact form
// (each doc records its basis and runs); `W_ENTRY`, `W_ITEM` and `W_BYTE`
// stay provisional until T7 fits them. Each is
// ⌈its recorded figure × 1.25⌉ or above, from figures measured on a debug
// build with the Rust heap counted (MB = 10^6 B); the provisional
// `W_ENTRY` and `W_ITEM` were measured before the compact cache and decode
// (b.mr6), which shrank what a structural edit holds for its object, so they
// over-reserve on the current tree. Each attributes every byte of its figure
// to itself: where two terms share one measured excess (`W_ENTRY` and
// `W_ITEM`), both cover all of it.
//
// They and their charge functions are `pub(crate)`: the storage side counts
// (`storage::commit_charge::ObjectStateNeed`), and only this module prices.

// The rewrite terms are priced per route: each `ObjectStateNeed` variant
// names the route its commit runs on, so [`object_state_charge`] and
// [`object_state_rewrite_allowance`] price a need with its own route's set
// ([`RewritePrices`]). `rewrite_row` and `rewrite_list` differ by route: the
// cached route's plan in place keeps a journal the WATCH path's plan on its
// own read does not, and the WATCH snapshot factors were netted of the WATCH
// row's value. `rewrite_name_index`, `rewrite_adoption_entry` and
// `rewrite_walk_entry` are one price on both routes, each counted on one
// route only (the indexes on the WATCH path's plan, the walk on the cached
// route's hooked plan).

/// `rewrite_row` on the cached route, in bytes per rewritten row: each
/// mutation a structural plan emits beyond the delta's own elements
/// (`RewriteCount::rows`) — a re-keyed child, a cascaded or reset deletion,
/// and each row a reset's or cascade's projection asks for before it is
/// emitted — and the plan journal's first-touch record of each such row,
/// alive through the commit, and what a rollback re-allocates to restore it
/// (the journal's records of the delta's own elements are the per-element
/// term's, [`ELEMENT_CHARGE_UPDATE_SLOW_BYTES`]). **Fitted** by T7
/// (derivation rule above) on the cached route planning in place: moving the
/// first of 20,000 children to the end, which re-keys every child, nets
/// against one new child on the same base at 3,658.5 B per re-keyed row, the
/// largest over two runs, with the object cached and from a cold start
/// alike; a reset's and a cascade's deleted rows cost less (1,435.9–1,646.7 B
/// per row with the list term netted), and a deleting edit rolled back after
/// CR2 stays within its charge
/// (`docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`,
/// § `rewrite_row`); ⌈× 1.25⌉ is 4,574 (3,500.4 B when the route planned
/// on a copy). The WATCH path's row is [`WATCH_REWRITE_ROW_CHARGE_BYTES`].
pub(crate) const CACHED_REWRITE_ROW_CHARGE_BYTES: u64 = 4_574;

/// `rewrite_row` on the WATCH/MULTI/EXEC path, in bytes per rewritten row
/// (`RewriteCount::rows`, as on the cached route). **Fitted** by T7
/// (derivation rule above) when every `UpdateSlow` route shared one row
/// price: moving the first of 20,000 children to the end netted against one
/// new child on the same base at 3,500.4 B per re-keyed row on the cached
/// route, then planning on a copy, which bound, identical over 5 runs
/// (`docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`,
/// § `rewrite_row`); ⌈× 1.25⌉ is 4,376. Kept at that value because the
/// WATCH snapshot factors were netted of it, so it holds together with them
/// (the WATCH path's own row nets lower, below). The WATCH path nets to
/// 793.6 B per row once its commit's queued commands are netted of
/// [`QUEUED_COMMAND_CHARGE_BYTES`] (5,916.3 B un-netted; 793.5 B on the
/// same fixture after the cached route began planning in place), so this
/// value is valid only with that term wired at 1,896 or above (else at least
/// 7,396)
/// (`docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`,
/// § WATCH prices).
pub(crate) const WATCH_REWRITE_ROW_CHARGE_BYTES: u64 = 4_376;

/// `rewrite_list` on the cached route, in bytes per entry of a children list
/// a structural plan rewrites or lists (`RewriteCount::list_entries`): a
/// parent's listing is projected at its whole length before the plan builds
/// it, and counts the larger of that and what the plan emits for the parent.
/// **Fitted** by T7 (derivation rule above) on the cached route planning in
/// place, with the plan's journal alive through the commit: each charge
/// point's segment with the object cached, its peak less the segment before
/// the first charge point and its growth term (the held entry predates the
/// commit, so its peak excludes it), per list entry the commit reports, over
/// one-new-child edits on 1,000, 5,000, 7,168 and 20,000 children, harness
/// layers of 1,250 and 2,500 prims, 2,141 sections and 5,000 sections with
/// 8-B and long names, and two new children on 5,000, 7,168 and 20,000:
/// 229.6 B, one new child on 1,000 children, in the commit's segment (CR2 to
/// the end: the digests, the commit's encoding of the rewritten list, and
/// the settle), 220.6–224.0 B on the others
/// (`docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`,
/// § `rewrite_list`); ⌈× 1.25⌉ is 287. The route charged a plan copy
/// beside the entry when this term was last fitted (184.0 B net of it,
/// `docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`),
/// which also covered the commit's encoding of the list. The WATCH path's
/// list entry is [`WATCH_REWRITE_LIST_ENTRY_CHARGE_BYTES`].
pub(crate) const CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES: u64 = 287;

/// `rewrite_list` on the WATCH/MULTI/EXEC path, in bytes per list entry
/// (`RewriteCount::list_entries`, as on the cached route). **Fitted** by T7
/// (derivation rule above) on the WATCH path's own commit: one new child
/// under a section of 20,000 children against one under an empty section
/// beside it, on the same base (the same read, retained state and queued
/// commands), the commit's segment after the apply differing by the
/// rewritten list alone — 199.243 B per list entry, the same over two runs
/// (197.8 B at 5,000 children)
/// (`docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`,
/// § WATCH prices); ⌈× 1.25⌉ is 250. It was 230 when every `UpdateSlow`
/// route shared one list price, set by the cached route planning on a copy
/// (184.0 B net of the copy's charge,
/// `docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`,
/// § `rewrite_list`), which the WATCH path's own figure exceeds. The WATCH
/// snapshot factors were netted of 230; a higher list price only lowers what
/// they would net to, so they hold with it.
pub(crate) const WATCH_REWRITE_LIST_ENTRY_CHARGE_BYTES: u64 = 250;

/// `rewrite_name_index`, in bytes per section a structural plan's name index
/// holds (`RewriteCount::indexed_sections`): the `(parent_id, name)` index a
/// plan builds at its second name probe, projected at its whole size — the
/// object's sections plus every entry the delta can add to it, one per
/// birth, rename or reparent (`StructuralInserts::name_index_inserts`) —
/// before it is built, and built at that capacity, so it never grows.
/// **Fitted** by T7 (derivation rule above): two new children on 7,167
/// children, the cached route with the object cached, whose index segment a
/// peak-time allocation tracker attributes — the index table (671,752 B:
/// 16,384 buckets for 7,172 entries) plus 7,172 one-id `Vec`s at 32 B, so
/// 125.7 B per section, at the worst bucket ratio (an index just past 7/8 of
/// a power of two), over 4 runs
/// (`docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`,
/// § `rewrite_name_index`; 99.2 B at 20,000 sections). It is exactly the
/// model: hashbrown's 41 B per bucket at up to 2 × 8/7 buckets per entry (the
/// table rounds its capacity up to a power of two at a 7/8 load), plus the
/// 32 B `Vec`. ⌈× 1.25⌉ is 158.
/// Charged exactly, never scaled by the rewrite headroom: its size is known
/// when it is built.
pub(crate) const REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES: u64 = 158;

/// `rewrite_adoption_entry`, in bytes per id a structural plan's adoption
/// index records (`RewriteCount::adoption_entries`): the rows naming an id
/// the delta can birth, live or not, so a `DiffChangeSection` of a live
/// section in a delta with any birth records all of that section's children.
/// Each id is projected before it is pushed, by the index's build and by the
/// name hook that keeps it, and never retired before the plan ends.
/// **Fitted** by T7 (derivation rule above): re-keying a section of 20,000
/// children with one birth, on the cached route, records 20,000 ids, and the
/// segment recording ids up to the 16,894th peaks 393,152 B above the
/// index's — at the 16,385th push, the `Vec<u64>` growing from 16,384 to
/// 32,768 ids, old and new buffers at once — so 24.0 B per id, identical
/// over 4 runs
/// (`docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`,
/// § `rewrite_adoption_entry`), as modelled: 8 B per id in a `Vec<u64>` that
/// doubles as it grows, up to 16 B steady and 24 B while a push reallocates.
/// ⌈× 1.25⌉ is 30. At the commit peak, against the same delta re-keying a
/// childless section, the ids' 262,144 B buffer is still live: 262,280 B
/// above the control, 13.1 B per id, matched run to run. One cross-run
/// pairing adds an unrelated 262,144 B allocation present at that peak in
/// some runs and not others, for 26.22 B per id, whose ⌈× 1.25⌉ is 33; it is
/// wired at 33 to cover that pairing. Scaled by the rewrite headroom like
/// rows and list entries (`storage::commit_charge::ChargeRewriteBudget`),
/// since the count grows one id at a time.
pub(crate) const REWRITE_ADOPTION_ENTRY_CHARGE_BYTES: u64 = 33;

/// `rewrite_walk_entry`, in bytes per section a structural plan's depth check
/// walks (`RewriteCount::walked_sections`): on the cached route the
/// incremental check projects each hop of its walks up (its memo and path
/// entries) and each children list before it pushes it onto its walk down,
/// and a full walk (an entry whose indexes cannot answer incrementally)
/// projects every section, which CR1 already holds; the walk's memo, path,
/// visited set and stack live until the check ends. **Fitted** by T7
/// (derivation rule above), each walk measured alone on a compact state, its
/// peak per section it walks as the plan projects them: the incremental
/// check's walk up from a new leaf under a 9,000-deep chain, 9,003 sections
/// at 99.2 B each — every hop's memo entry, path entry and on-path set entry
/// live at once, the memo map mid-growth — which binds; its walk down a moved
/// section's 20,000 children, 38.1 B; the full walk, which varied between
/// runs, 54.1 B on 20,000 children of one section (41.0 B in run 1), 19.7 B
/// on the 2,500-prim harness layer (18.0 B) and 11.1 B on the dev shape at
/// 7,167 sections; the chain's leaf on the cached
/// route, its plan's segment net of its other terms, 97.3 B
/// (`docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`,
/// § `rewrite_walk_entry`); ⌈99.2 × 1.25⌉ is 124.
pub(crate) const REWRITE_WALK_ENTRY_CHARGE_BYTES: u64 = 124;

/// `W_FIXED`, in bytes: a structural WATCH attempt's fixed cost beyond what
/// `F` ([`FIXED_CHARGE_BYTES`]) covers — the boxed attempt future, the fence
/// probe joined to the header read, writer-intent registration, the
/// snapshot read joined to the stream-collision probe and the fan-out's first
/// windows — held from the head of every structural WATCH attempt, rather
/// than raising `F` for every upload. **Fitted** by T7 (derivation rule
/// above): one new child on an empty section on the WATCH path under `txn`
/// (`slow_path_one_new_child_on_empty_base_stays_within_its_charge_b26e`),
/// its processing phase from WA0 to WA1 (charged processing plus `W_FIXED`
/// alone, before the read counts anything), which holds the read's first
/// window (the header and the index counts, one plain future per command)
/// and the read's pass future, boxed onto its `watch-read` thread
/// (`FanoutThreads::run_pass`). On the shipped window shape, over 24 runs
/// per transport on a client whose connections had already carried traffic,
/// it peaked at most 160,874 B (the script transport; the plain one at most
/// 147,543 B); a call that is the first to use its connections adds fred's
/// router channel block, 14,624 B, so 175,498 B binds. Net of the earlier
/// terms of the fixture's processing charge, each at its production basis, or
/// at the run's own figure where the run gives one and it is larger
/// (`element[UpdateSlow]`; the derivation rule above) — `F`'s
/// basis, 72,160 B; the body's 152 B, 1:1; `byte[UpdateSlow]`'s basis × its
/// 136-B delta, 913.92 B; and for its one element `element[UpdateSlow]`'s
/// basis, 2,780.8 B, since the WATCH path's own figure, netted of its queued
/// commands, is at most 1,900.2 B ([`ELEMENT_CHARGE_UPDATE_SLOW_BYTES`]):
/// 76,006.72 B — that is 99,491.28 B, and ⌈× 1.25⌉ is 124,365
/// (`docs-internal/validation/watch-window-script-b-gvf-20261010/`,
/// `results.json` `w_fixed_refit`; `scripts/m3_wfixed_check.py`, which reads
/// each basis from this file). The step-2 read inside it carries the commit
/// release ring's `HGET obj:wrel |n|`, whose reply is at most 3 bytes, inside
/// the fit. At 124,365 the fixture's processing charge is 219,336 B: its warm
/// peak is 0.734 of it, 0.800 with the first-use block.
///
/// Who pays more against main [model, from the constants;
/// `scripts/m3_one_at_a_time.py`, `raw/m3-one-at-a-time.json`]: only
/// structural WATCH attempts hold `W_FIXED`. Per attempt, with the
/// per-command factor's refit beside it, the charge moves by +42,387 B
/// (`W_FIXED`, 81,978 → 124,365), plus up to 82,175 B (round 1's index pages,
/// at most five plain commands at
/// [`WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES`]), plus the request-byte
/// projection × `W_BYTE`, less 6,278 B × min(wire commands, 2,141) (the
/// per-command factor, 16,435 → 10,157 B). That breaks even at about 20 wire
/// commands (about 10 entries): only near-empty layers pay more, at most
/// about 125 KB per structural WATCH attempt beside that projection, and
/// every larger layer pays less, by up to about 13.4 MB at a full window. No
/// size or admission limit worsens.
pub(crate) const WATCH_ATTEMPT_FIXED_BYTES: u64 = 124_365;

/// `W_CMD`, the WATCH snapshot's per-command factor, in bytes per snapshot
/// command in flight at once during a structural edit's whole-object read on
/// the WATCH/MULTI/EXEC path: `min(commands, FANOUT_WINDOW)` wire commands
/// (a value row's two-command read counts two), one window's commands and
/// replies awaiting it. The read issues the windows of its rounds 2 and 3 as
/// one window script each (`storage::lua::FANOUT_WINDOW_READ_SCRIPT`), whose
/// window holds one future and its entries' arguments and replies; its plain
/// windows (round 1, a lone children list over τ, and every window after a
/// script fallback), each command with its own future, are priced apart, by
/// [`WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES`], so this factor prices what
/// the read holds per command in flight beyond every other term.
/// **Fitted** by T7 (derivation rule above) on the script transport: per
/// command in flight, each read segment's commit-only peak less every other
/// term of the charge — the fitted ones ([`WATCH_ATTEMPT_FIXED_BYTES`] at its
/// wired 124,365 B, `W_NAME_BYTE` × name bytes,
/// [`WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES`] × plain commands in flight)
/// at their production basis, the provisional ones (`W_ENTRY` × entries,
/// `W_ITEM` × items, `W_BYTE` × the window's bytes) at their wired values —
/// over bases of 5,000 and 20,000 children, 2,141 children and fields with
/// 8-B names, harness 250, 1,250 and 2,500 prims, the 1,024-B-name base and
/// 5,000 sections named 150 B, at the 2,141- and 1,000-command windows, 24
/// runs per transport on the shipped window shape. The largest is 8,125.0 B,
/// the 1,024-B-name base at the 1,000 window during its round-1 index pages,
/// whose names arrive longer than the read has sized them (the residual,
/// module docs), in one script-transport run; every other base nets below
/// zero; ⌈× 1.25⌉ is 10,157. The fit is adopted only if every read segment
/// and WA3 of every T7 base, on both transports, holds at or under its
/// commit-only charge at it, else the smallest value that bounds every one
/// is wired: none is over at 10,157 (worst 0.866, that segment), so it is
/// wired (`docs-internal/validation/watch-window-script-b-gvf-20261010/`,
/// `results.json` `w_cmd_refit`).
///
/// That segment's peak is not attributed: in 2 of 76 complete runs on the
/// merged and shipped trees it rose about 3.04–3.08 MB above its usual
/// 10.26–10.30 MB (13,343,990 B on the plain transport, 13,342,520 B on the
/// script one), and every other run stayed at its usual peak. Should a
/// refit put any segment above 0.9 of its charge, the excess is attributed
/// before that value is wired. `W_CMD` and `W_BYTE` are not yet separable
/// (every saturated base moves both in-flight commands and window bytes
/// together between the two windows), so this term carries the whole excess
/// until a window-bytes arm splits them.
pub(crate) const WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES: u64 = 10_157;

/// `W_PLAIN_CMD`, the WATCH snapshot's per-plain-command factor, in bytes per
/// plain wire command in flight at once (`WatchSnapshot::plain_in_flight`):
/// a window the read issues as plain commands, each command holding its own
/// future — a round-1 index page, a lone children list too long for a script
/// window, and every window after the read falls back from its window script
/// on a cluster-state refusal. A script window holds one future whatever its
/// entries, priced by [`WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES`]; this term keeps
/// the charge a bound on the plain windows whatever that factor is fitted to.
/// **Fitted** by T7 (derivation rule above) on exactly these per-command
/// futures: the WATCH read's plain windows, two commands per value row, its
/// value-row read inline — 13,148 B per command at most, the per-command
/// fit of the read when every window of it was plain
/// (`docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`,
/// § `W_CMD`); ⌈× 1.25⌉ is 16,435. `storage::diff_read`'s size pin
/// `bounded_command_future_holds_the_value_row_read_inline_b26e` keeps that
/// future's shape.
pub(crate) const WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES: u64 = 16_435;

/// `W_ENTRY`, the WATCH snapshot's per-entry factor, in bytes per entry of
/// the object (its sections, tombstones, fields, time samples, and two per
/// children list — its `LLEN` and its ranged `LRANGE` — as the read's count
/// windows report them): the round-1 index members and the command lists the
/// read holds until each round is issued. Fitted on short names; the bytes
/// of long names are [`WATCH_SNAPSHOT_NAME_BYTE_CHARGE_BYTES`]'s, fitted
/// first, and this is to be re-netted of them.
/// **Provisional**, until a T7 fit separates it from
/// [`WATCH_SNAPSHOT_ITEM_CHARGE_BYTES`]: the largest excess per entry of any
/// measured WATCH peak over its processing charge, `W_FIXED`, `W_CMD` ×
/// commands in flight (`W_CMD` then 16,435, every window of the read plain)
/// and `W_BYTE` × the byte window — 185.1 B per entry,
/// one new prim on the 2,500-prim harness base (about 55,006 entries,
/// 257,556 items) at the 1,000-command window, 24,982,660 B, measured before
/// the compact decode (b.mr6) — ⌈× 1.25⌉ is 232. Every other measured WATCH
/// row (bases of 0 to 20,000 children and 250 to 2,500 prims, at both
/// windows) needs less.
pub(crate) const WATCH_SNAPSHOT_ENTRY_CHARGE_BYTES: u64 = 232;

/// `W_ITEM`, the WATCH snapshot's per-item factor, in bytes per item the read
/// has tallied (hash fields and members), which its decoded state retains.
/// **Provisional**, until a T7 fit separates it from `W_ENTRY`: the same
/// largest excess as [`WATCH_SNAPSHOT_ENTRY_CHARGE_BYTES`], attributed to
/// items instead — 39.5 B per item on the same row — ⌈× 1.25⌉ is 50. To be
/// re-netted of [`WATCH_SNAPSHOT_NAME_BYTE_CHARGE_BYTES`] by that fit, as
/// `W_ENTRY` is.
pub(crate) const WATCH_SNAPSHOT_ITEM_CHARGE_BYTES: u64 = 50;

/// `W_NAME_BYTE`, the WATCH snapshot's per-name-byte factor, in bytes per
/// byte of the names the read retains: the round-1 field-index members' bytes
/// and the section names' bytes it has read so far. `W_ENTRY` and `W_ITEM`
/// charge an entry by count, fitted on short names; a field entry keeps its
/// name about three times over — its index member, its round-2 key
/// (`obj:fld:{member}`) and its command's id — and the compact state interns
/// the decoded name once more, so at `MAX_NAME_BYTES` (4,096) one entry holds
/// about 100 × `W_ENTRY`, which no count term covers. **Fitted** by T7
/// (derivation rule above) from the named fixtures (2,141 sections and 2,141
/// fields with 1,024-B against 8-B names): 1.3105 B per name byte retained at
/// WA3, over 5 runs; in the read phase the names retained cost 1.000 B per
/// byte, and a reply's incoming names at most 1.2 B per byte, which the
/// per-window reply term charges
/// (`docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`,
/// § `W_NAME_BYTE`).
/// ⌈1.3105 × 1.25⌉ is 2; with it and the other WATCH factors as wired, the
/// named reads measured at most 0.826 of their commit-only charge.
pub(crate) const WATCH_SNAPSHOT_NAME_BYTE_CHARGE_BYTES: u64 = 2;

/// `W_BYTE`, the WATCH snapshot's per-byte factor, in hundredths per byte of
/// one fan-out window's replies and script request: at the count windows
/// (WA1), the read's projected reply and request bound from its counts at the
/// name floor, at most `FANOUT_WINDOW_BYTES`
/// (`diff_read::projected_read_reply_bound`,
/// `diff_read::projected_read_request_bound`), so a small object's read
/// holds about its own replies' bound rather than a whole window; before a
/// window or index page is issued, the sum of its commands' per-command reply
/// bounds, plus a script window's request bytes (the bound the read sizes it
/// within, `FANOUT_WINDOW_BYTES` at most unless one command alone exceeds
/// it); after it, the bytes its replies held. The read never
/// transfers an inline value (a value row is read by its length and its other
/// columns), so every reply is
/// bounded before it is issued except names longer than the read has seen,
/// a compact-encoded set index's whole membership answering one index page,
/// and a hashtable-encoded set index's scan step overshooting its share by a
/// few members (the residuals, module docs). Under the compact decode a
/// window's replies are dropped as it completes, so the term is per window,
/// not per read.
/// **Provisional**: the WATCH path under `txn` measured 5.470 per byte of a
/// delta's inline values, which end as the same rows (the `byte[UpdateSlow]`
/// figure above); × 1.25 is 6.838, until T7 refits it on the read's current
/// reply shape.
pub(crate) const WATCH_SNAPSHOT_REPLY_BYTE_CHARGE_CENTI: u64 = 684;

/// The per-queued-command factor, in bytes per slot of a connection's
/// command queues a WATCH commit fills, charged per next power of two of the
/// commands queued at once, because fred's pipeline buffer and its
/// connection's in-flight queue both double (`VecDeque` growth): the
/// transaction pipeline, the external-value pre-image read and the keyspace
/// probe ([`queued_command_charge`]). **Fitted** by T7
/// (derivation rule above) from stalled WATCH commits, whose pipeline is
/// held on the transaction connection until the whole queue is in flight:
/// the field-dense fixture (60,006 transaction commands, 0.916 of their
/// 65,536 next-power-of-two slots), net of the earlier terms and the cached
/// route's element basis, is 1,516.6 B per transaction slot, counting those
/// slots alone (the pre-image read's and the probe's left out, which only
/// raises it), identical over 5 runs
/// (`docs-internal/validation/upload-commit-charge-b26e-s2-20261008/fits.md`,
/// § `queued`); ⌈× 1.25⌉ is 1,896. `element[UpdateSlow]` and `rewrite_row`
/// are netted of it, so they are valid only with it wired at this value or
/// above.
pub(crate) const QUEUED_COMMAND_CHARGE_BYTES: u64 = 1_896;

/// `repair_reply_entry` (`a`), in bytes per stream entry of one stream-repair
/// read's reply on the cached route (`storage::structural_cache_repair`; RP0
/// at the read's entry bound, RP1 and RP2 at the chunk's entries): what each
/// decoded entry holds beyond its bytes — its id and payload copied into
/// `String`s, its field vector, the reply's per-entry frame structure. The
/// reply is priced in two terms, this one per entry and
/// [`REPAIR_REPLY_BYTE_CENTI`] per byte, so a read's bound is priced at what
/// a full read of small records and a full read of large ones each hold,
/// rather than at the small records' per-byte cost over the whole byte bound.
/// **Fitted** by T7 (derivation rule above) after `b`, from the multi-entry
/// repair reads (256 small records, 256 field writes, 256 × 1 KiB; 30 reads
/// over 5 runs), each read's peak net of `F`'s basis (72,160 B) and of
/// `b`'s largest figure × its reply bytes, per entry. Wired at 864 from the
/// first fit, at a first read of 113 entries: 690.70 B per entry, the
/// 256-small-records row's 143-entry second read; ⌈× 1.25⌉. The refit at
/// the wired first read (91 entries; the current members) fits 543, whose
/// first-read step would be 149 entries, so the fit has not converged: 864
/// is kept because it covers every member, which `fit_repair.py` checks,
/// and is lowered only by a refit that converges and still covers them
/// (`docs-internal/validation/repair-charge-t7-b-fsz-20261010/fits.json`,
/// `reply_fit`; its members in `fits-manifest.json`). Re-derive it whenever
/// `b` is refitted (`a` is netted of `b`'s figure), or when the per-entry
/// decode (the entry's id and payload strings, its field vector, the
/// reply's per-entry frame structure) or the read's entry bounds change;
/// the first read's entry bound
/// (`storage::structural_cache_sync::STRUCTURAL_REPAIR_INITIAL_ENTRIES`)
/// follows it.
pub(crate) const REPAIR_REPLY_ENTRY_BYTES: u64 = 864;

/// `repair_reply_byte` (`b`), in hundredths per byte of one stream-repair
/// read's reply on the cached route (RP0 at the read's byte bound, RP1 and
/// RP2 at the chunk's byte tally): the reply frame in the connection's read
/// buffer, which a fresh buffer grows to by doubling, the old and new
/// buffers held at once, plus the decoded entries' OMPF fields, zero-copy
/// `Bytes` of that frame. The per-entry rest is
/// [`REPAIR_REPLY_ENTRY_BYTES`]. **Fitted** by T7 (derivation rule above)
/// from the single-entry repair reads (one record of 2 MiB external value,
/// 3 MiB of prims, 1 MiB of sections, a 250 KiB inline value; 20 reads over
/// 5 runs), each read's peak net of `F`'s basis (72,160 B), per reply byte.
/// The largest is 2.954229 B per reply byte: the 2 MiB external record, one
/// entry of 2,097,578 B peaking at 6,268,886 B in every run; ⌈× 125⌉ is 370.
/// The refit at the wired first read fits 369; 370 is kept with `a` until a
/// refit converges, and covers every member
/// (`docs-internal/validation/repair-charge-t7-b-fsz-20261010/fits.json`,
/// `reply_fit`; its members in `fits-manifest.json`). Re-derive it, and `a`
/// with it, when the repair read's reply decode changes (`SyncStreamEntry`,
/// the repair script's reply format, the zero-copy OMPF fields) or fred's
/// read-buffer growth or retention limits change, when `F`
/// ([`FIXED_CHARGE_BYTES`], every read's netting basis) changes, or when the
/// repair's read bounds change (`STRUCTURAL_REPAIR_MAX_BYTES`, the first
/// read's byte bound).
pub(crate) const REPAIR_REPLY_BYTE_CENTI: u64 = 370;

/// `repair_growth`, in hundredths per byte of the delta parts a stream
/// repair replays (RP1 over a round's records, RP2 over one record's): what
/// the replay adds to the held entry beyond the table, children-list and
/// name-index-vector growth, the listings and depth walk, and the cascade,
/// reset and list-delete sink scratch that RP2 prices per record
/// (`storage::structural_cache::estimate_replay_record_growth`) — so it
/// carries what scales with the record's own elements, not a cascade's
/// deleted subtree.
/// **Fitted** by T7 from a direct per-record measure, since the derivation
/// rule's netting does not identify it (its netted figure is 0): each
/// replayed record's peak above the replay's start (its decode, which
/// `k_parse` prices, excluded), net of what the held
/// entry had already grown (on an indexed entry its weights; on the audit's
/// index-less copy the growth RP2 priced for the records before it), of the
/// growth and scratch RP2 prices for the record (1:1) and of its rewrite at
/// the cached prices' basis (÷ 1.25), per byte of the delta parts RP2
/// carries for it (an indexed entry's: the record's own; the copy's: every
/// record's so far). Over 28 rows, on both holders and through each sink
/// the row's repair takes, 2,116 records per run and 3 runs, the largest is
/// 0.999453 B per delta byte: one record carrying a 250 KiB inline value
/// (256,128 B of delta) replayed onto an indexed entry, the value's copy
/// into the entry; ⌈× 125⌉ is 125
/// (`docs-internal/validation/repair-charge-t7-b-fsz-20261010/fits.json`;
/// its members in `fits-manifest.json`). Re-derive it when the replay's
/// apply changes what a record allocates and keeps beyond the priced terms
/// (how the compact state stores an inline value, a name or a row:
/// `domain::types`, `domain::delta`'s apply in `ApplyMode::Replay`, the name
/// interner); when `estimate_replay_record_growth` or, on the audit's copy,
/// `estimate_replay_record_growth_bounded` changes a growth, scratch or
/// rewrite term, or `CopyBucketBound::applied` changes how the copy's
/// bucket bound rises, or the cached rewrite prices change; when RP2's held
/// term or its per-record normalization changes; or when a record shape
/// outside the measured rows (large inline time-sample values, say) enters
/// production replay at scale, or T7's counting basis changes. One record's
/// parse is priced at `k_parse` ([`PARSE_EXPANSION_CENTI`]), the upload's
/// own: replay parses with the same `parse_delta`.
pub(crate) const REPAIR_GROWTH_CENTI: u64 = 125;

/// Which upload RPC a reservation belongs to: the `rpc` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadRpc {
    CreateLayer,
    UpdateLayer,
}

impl UploadRpc {
    /// `rpc` label value on the upload admission metrics.
    pub fn label(self) -> &'static str {
        match self {
            UploadRpc::CreateLayer => UPLOAD_RPC_CREATE_LAYER,
            UploadRpc::UpdateLayer => UPLOAD_RPC_UPDATE_LAYER,
        }
    }
}

/// The path an upload's commit takes, which decides its processing charge:
/// `CreateLayer`'s materialization, `UpdateLayer`'s Lua fast path, or its
/// WATCH/MULTI/EXEC slow path (the `Txn` backend included). For `UpdateLayer`
/// it comes from `LayerStore::takes_fast_path`, the predicate the commit
/// dispatch itself uses, so the charge and the dispatch cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitPath {
    Create,
    UpdateFast,
    UpdateSlow,
}

impl CommitPath {
    /// `element[path]`, in bytes per delta element.
    pub const fn element_charge_bytes(self) -> u64 {
        match self {
            CommitPath::Create => ELEMENT_CHARGE_CREATE_BYTES,
            CommitPath::UpdateFast => ELEMENT_CHARGE_UPDATE_FAST_BYTES,
            CommitPath::UpdateSlow => ELEMENT_CHARGE_UPDATE_SLOW_BYTES,
        }
    }

    /// `byte[path]`, in hundredths per delta byte.
    pub const fn delta_byte_charge_centi(self) -> u64 {
        match self {
            CommitPath::Create => DELTA_BYTE_CHARGE_CREATE_CENTI,
            CommitPath::UpdateFast => DELTA_BYTE_CHARGE_UPDATE_FAST_CENTI,
            CommitPath::UpdateSlow => DELTA_BYTE_CHARGE_UPDATE_SLOW_CENTI,
        }
    }

    /// `reorder[path]`, in bytes per reorder-list entry.
    pub const fn reorder_entry_charge_bytes(self) -> u64 {
        match self {
            CommitPath::Create => REORDER_ENTRY_CHARGE_CREATE_BYTES,
            CommitPath::UpdateFast => REORDER_ENTRY_CHARGE_UPDATE_FAST_BYTES,
            CommitPath::UpdateSlow => REORDER_ENTRY_CHARGE_UPDATE_SLOW_BYTES,
        }
    }
}

/// The counts a parsed delta's processing charge scales with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeltaFootprint {
    /// Struct commands + set fields + time samples
    /// ([`own_element_count`], the count a plan's `RewriteCount::rows`
    /// excludes).
    pub elements: u64,
    /// The summed `children_list` lengths of the `ReorderChildren` commands.
    pub reorder_entries: u64,
}

impl DeltaFootprint {
    /// The footprint of a parsed delta; `None` (empty content, or a container
    /// with no delta) has none.
    pub fn of(delta: Option<&oodf_ffi::OwnedDelta>) -> Self {
        let Some(delta) = delta else {
            return Self::default();
        };
        let count = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        let reorder_entries = delta
            .struct_commands
            .iter()
            .filter_map(|command| match command {
                oodf_ffi::StructCommand::ReorderChildren(reorder) => {
                    Some(count(reorder.children_list.len()))
                }
                _ => None,
            })
            .fold(0u64, u64::saturating_add);
        Self {
            elements: own_element_count(delta),
            reorder_entries,
        }
    }
}

/// What [`UploadAdmissionGate::admit`] reserves: `F + A`.
pub const fn admission_charge() -> u64 {
    FIXED_CHARGE_BYTES + RECEIVE_ALLOWANCE_BYTES
}

/// The smallest budget that carries the receive phase of an upload of
/// `max_upload_bytes` (`LIVE_EDIT_MAX_LAYER_SIZE`) on its own:
/// `F + A + RECEIVE_CHARGE_FACTOR × max_upload_bytes`, saturating. With `B` at
/// least this, no upload under the size cap can be refused for its size.
pub const fn upload_budget_floor_bytes(max_upload_bytes: u64) -> u64 {
    admission_charge().saturating_add(RECEIVE_CHARGE_FACTOR.saturating_mul(max_upload_bytes))
}

/// The upload budget `B` at the reference pod memory limit for a given
/// `LIVE_EDIT_MAX_LAYER_SIZE`, the pod memory plan's rule at 5,120 MiB:
/// `max(`[`UPLOAD_MEMORY_BUDGET_MIB_DEFAULT`]`, ⌈floor ÷ 1 MiB⌉)`
/// ([`upload_budget_floor_bytes`]) — 384 at the default size cap, larger
/// when the cap is raised past about 188 MiB.
pub const fn upload_memory_budget_mib_default(max_upload_bytes: u64) -> u64 {
    let floor_mib = upload_budget_floor_bytes(max_upload_bytes).div_ceil(MIB);
    if floor_mib > UPLOAD_MEMORY_BUDGET_MIB_DEFAULT {
        floor_mib
    } else {
        UPLOAD_MEMORY_BUDGET_MIB_DEFAULT
    }
}

/// `len × centi ÷ 100`, rounded up, saturating.
fn scaled(len: u64, centi: u64) -> u64 {
    let scaled = (u128::from(len) * u128::from(centi)).div_ceil(u128::from(EXPANSION_SCALE));
    u64::try_from(scaled).unwrap_or(u64::MAX)
}

/// `F` + body + `e_ext` × external-value bytes: the part of every processing
/// charge that does not depend on the delta.
fn base_charge(body_len: u64, ext_len: u64) -> u64 {
    FIXED_CHARGE_BYTES
        .saturating_add(body_len)
        .saturating_add(scaled(ext_len, EXT_EXPANSION_CENTI))
}

/// The first processing step's charge, held from before the delta is parsed:
/// `F` + body + `e_ext` × external-value bytes + `k_parse` × delta bytes. The
/// part lengths come from the zero-copy OMPF container parse.
pub fn parse_charge(body_len: u64, delta_len: u64, ext_len: u64) -> u64 {
    base_charge(body_len, ext_len).saturating_add(scaled(delta_len, PARSE_EXPANSION_CENTI))
}

/// The second processing step's charge, held from once the delta is parsed and
/// its commit path known to the end of the call: the larger of
/// [`parse_charge`] and `F` + body + `e_ext` × external-value bytes +
/// `byte[path]` × delta bytes + `element[path]` × elements + `reorder[path]` ×
/// reorder-list entries.
pub fn processing_charge(
    path: CommitPath,
    body_len: u64,
    delta_len: u64,
    ext_len: u64,
    footprint: DeltaFootprint,
) -> u64 {
    let commit = base_charge(body_len, ext_len)
        .saturating_add(scaled(delta_len, path.delta_byte_charge_centi()))
        .saturating_add(
            path.element_charge_bytes()
                .saturating_mul(footprint.elements),
        )
        .saturating_add(
            path.reorder_entry_charge_bytes()
                .saturating_mul(footprint.reorder_entries),
        );
    commit.max(parse_charge(body_len, delta_len, ext_len))
}

/// One route's rewrite prices, in bytes per unit of each [`RewriteCount`]
/// count: the cached route's ([`Self::CACHED`]) or the WATCH/MULTI/EXEC
/// path's ([`Self::WATCH`]). [`Self::of`] picks a need's own route's set; a
/// need is never priced at the other route's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RewritePrices {
    /// `rewrite_row`.
    pub(crate) row: u64,
    /// `rewrite_list`.
    pub(crate) list_entry: u64,
    /// `rewrite_name_index`.
    pub(crate) name_index_entry: u64,
    /// `rewrite_adoption_entry`.
    pub(crate) adoption_entry: u64,
    /// `rewrite_walk_entry`.
    pub(crate) walk_entry: u64,
}

impl RewritePrices {
    /// The cached route's: [`CACHED_REWRITE_ROW_CHARGE_BYTES`],
    /// [`CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES`] and
    /// [`REWRITE_WALK_ENTRY_CHARGE_BYTES`], fitted on its plan in place. A
    /// plan in place builds neither the name index nor the adoption index, so
    /// their prices ([`REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES`],
    /// [`REWRITE_ADOPTION_ENTRY_CHARGE_BYTES`]) are here but count nothing.
    pub(crate) const CACHED: Self = Self {
        row: CACHED_REWRITE_ROW_CHARGE_BYTES,
        list_entry: CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES,
        name_index_entry: REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES,
        adoption_entry: REWRITE_ADOPTION_ENTRY_CHARGE_BYTES,
        walk_entry: REWRITE_WALK_ENTRY_CHARGE_BYTES,
    };

    /// The WATCH path's: [`WATCH_REWRITE_ROW_CHARGE_BYTES`],
    /// [`WATCH_REWRITE_LIST_ENTRY_CHARGE_BYTES`],
    /// [`REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES`] and
    /// [`REWRITE_ADOPTION_ENTRY_CHARGE_BYTES`] — the values the WATCH
    /// snapshot factors were netted of, the list entry raised to its own
    /// figure's rule. Its plan runs without hooks, so it reports no walked
    /// sections, and the walk's price here counts nothing.
    pub(crate) const WATCH: Self = Self {
        row: WATCH_REWRITE_ROW_CHARGE_BYTES,
        list_entry: WATCH_REWRITE_LIST_ENTRY_CHARGE_BYTES,
        name_index_entry: REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES,
        adoption_entry: REWRITE_ADOPTION_ENTRY_CHARGE_BYTES,
        walk_entry: REWRITE_WALK_ENTRY_CHARGE_BYTES,
    };

    /// The set for `need`'s route.
    pub(crate) fn of(need: &ObjectStateNeed) -> Self {
        match need {
            ObjectStateNeed::Cached { .. } => Self::CACHED,
            ObjectStateNeed::Watch { .. } => Self::WATCH,
        }
    }
}

/// The rewrite part of a structural edit's object-state term at `prices`,
/// its route's set: `rewrite_row` × rows + `rewrite_list` × list entries +
/// `rewrite_name_index` × indexed sections + `rewrite_adoption_entry` ×
/// adoption entries + `rewrite_walk_entry` × walked sections, plus the
/// projected rollback rebuilds 1:1 (`RewriteCount::rollback_rebuild_bytes`,
/// already bytes, so no constant), saturating. While a plan runs its rewrite
/// includes what its build sites projected before allocating (a parent's
/// listing, the name index, the adoption index's ids, a reset's, a
/// cascade's or an adoption's rows, the depth walk, the tables and
/// name-index vectors its rollback could rebuild — an upper bound, each
/// rebuild priced exactly); after it, the exact kept count, with no index,
/// adoption, walk or rebuild entries.
pub(crate) fn rewrite_charge(prices: RewritePrices, rewrite: RewriteCount) -> u64 {
    rewrite.bytes(
        prices.row,
        prices.list_entry,
        prices.name_index_entry,
        prices.adoption_entry,
        prices.walk_entry,
    )
}

/// The stream-repair part of a structural edit's object-state term on the
/// cached route: the read's reply in two terms, [`REPAIR_REPLY_ENTRY_BYTES`]
/// × its entries + [`REPAIR_REPLY_BYTE_CENTI`] × its bytes, then `k_parse`
/// ([`PARSE_EXPANSION_CENTI`]) × the replayed record's delta part (the
/// largest one at RP1) + [`REPAIR_GROWTH_CENTI`] × the replayed delta parts,
/// saturating. `k_parse` is the upload's own: replay parses each record with
/// the same `parse_delta`.
pub(crate) fn repair_transient_charge(repair: RepairTransient) -> u64 {
    REPAIR_REPLY_ENTRY_BYTES
        .saturating_mul(repair.reply_entry_count)
        .saturating_add(scaled(repair.reply_bytes, REPAIR_REPLY_BYTE_CENTI))
        .saturating_add(scaled(repair.parse_delta_bytes, PARSE_EXPANSION_CENTI))
        .saturating_add(scaled(repair.replay_delta_bytes, REPAIR_GROWTH_CENTI))
}

/// The most one sampled audit may hold on `B` ([`UploadBudgetAuditCharges`]):
/// one upload admission, [`admission_charge`] (`F + A`, 8,478,808 B). A
/// constant, not a share of `B`: an audit's need follows the layer it checks
/// (its sections, its pending records), not the budget, and the crowding it
/// may cause is counted in admissions — with the audit's rule of always
/// leaving one admission's room free, an audit costs at most one concurrent
/// admission, for its few milliseconds. An audit whose repair would need more
/// skips (`storage::structural_cache_audit`, `AuditSkip::ChargeOverCap`).
pub const SAMPLED_AUDIT_CHARGE_CAP_BYTES: u64 = admission_charge();

/// The object-state term of a structural edit on the cached route, held on
/// top of its processing charge: `held` + `growth` + [`rewrite_charge`] at
/// [`RewritePrices::CACHED`] + [`repair_transient_charge`], saturating,
/// `held` and `growth` priced 1:1.
///
/// The route plans in place on the checked-out entry and copies no state.
/// `held` is that entry's estimated weight, state and plan indexes, all of it
/// held outside every other charge because a checked-out entry's resident
/// slot can be evicted under the call; `growth` what the attempt can
/// allocate beside it — the plan-index build's peak at a cold start (CR0),
/// the tables the plan can grow in place at CR1, the tables, lists and
/// name-index vectors one replayed record grows at the repair's RP2 — and,
/// once the plan's growth is in `held` (CR2), the tables its rollback could
/// still have to rebuild at their pre-plan length (usually 0); `rewrite` the
/// plan's rewrite — zero before the plan but for a full depth walk's
/// sections, its kept count plus its build-site projections (its rollback
/// rebuilds, 1:1, included) while it plans, its exact kept count after; at
/// RP2 the replayed record's listings, depth walk and (on an entry without
/// plan indexes) per-delta indexes; `repair` the stream repair's transient —
/// one read's reply, and while its chunk replays a record's parse and the
/// entry's growth (RP0, RP1, RP2), zero at the hint, CR0, CR1 and CR2.
pub(crate) fn cached_object_state_charge(
    held: u64,
    growth: u64,
    rewrite: RewriteCount,
    repair: RepairTransient,
) -> u64 {
    held.saturating_add(growth)
        .saturating_add(rewrite_charge(RewritePrices::CACHED, rewrite))
        .saturating_add(repair_transient_charge(repair))
}

/// The read-and-rewrite part of a structural edit's object-state term on the
/// WATCH/MULTI/EXEC path: the whole-object read's snapshot —
/// [`WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES`] × commands in flight +
/// [`WATCH_SNAPSHOT_ITEM_CHARGE_BYTES`] × items +
/// [`WATCH_SNAPSHOT_REPLY_BYTE_CHARGE_CENTI`] × one window's reply bytes —
/// plus [`rewrite_charge`] at [`RewritePrices::WATCH`], saturating. [`object_state_charge`] adds the
/// attempt's fixed cost, the per-entry term and the queued commands.
pub(crate) fn watch_object_state_charge(
    commands_in_flight: u64,
    items: u64,
    reply_bytes: u64,
    rewrite: RewriteCount,
) -> u64 {
    WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES
        .saturating_mul(commands_in_flight)
        .saturating_add(WATCH_SNAPSHOT_ITEM_CHARGE_BYTES.saturating_mul(items))
        .saturating_add(scaled(reply_bytes, WATCH_SNAPSHOT_REPLY_BYTE_CHARGE_CENTI))
        .saturating_add(rewrite_charge(RewritePrices::WATCH, rewrite))
}

/// The queued-command part of a WATCH commit's object-state term:
/// [`QUEUED_COMMAND_CHARGE_BYTES`] × the next power of two of each
/// connection's commands queued at once — the transaction pipeline, the
/// external-value pre-image read and the keyspace probe — summed, saturating:
/// every queue's worst case at once, so no bound depends on reply timing. A
/// count of zero costs nothing.
pub(crate) fn queued_command_charge(queued: QueuedCommands) -> u64 {
    let slots = |n: u64| {
        if n == 0 {
            0
        } else {
            n.checked_next_power_of_two().unwrap_or(u64::MAX)
        }
    };
    QUEUED_COMMAND_CHARGE_BYTES.saturating_mul(
        slots(queued.txn)
            .saturating_add(slots(queued.pre_image))
            .saturating_add(slots(queued.probe)),
    )
}

/// The object-state term, in bytes, of what a structural commit reports it
/// is about to hold (`ObjectStateNeed`): the price the commit charge's
/// `target` answers, and the handler's hint. Each variant is priced at its
/// own route's rewrite prices ([`RewritePrices::of`]).
///
/// - Cached route: [`cached_object_state_charge`].
/// - WATCH path: [`WATCH_ATTEMPT_FIXED_BYTES`] when the attempt's fixed cost
///   is held, + [`WATCH_SNAPSHOT_ENTRY_CHARGE_BYTES`] × entries +
///   [`WATCH_SNAPSHOT_NAME_BYTE_CHARGE_BYTES`] × retained name bytes +
///   [`WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES`] × plain commands in flight
///   + [`watch_object_state_charge`] + [`queued_command_charge`], saturating.
pub(crate) fn object_state_charge(need: &ObjectStateNeed) -> u64 {
    match *need {
        ObjectStateNeed::Cached {
            held,
            growth,
            rewrite,
            repair,
        } => cached_object_state_charge(held, growth, rewrite, repair),
        ObjectStateNeed::Watch {
            snapshot,
            fixed,
            rewrite,
            queued,
        } => {
            let fixed = if fixed { WATCH_ATTEMPT_FIXED_BYTES } else { 0 };
            fixed
                .saturating_add(WATCH_SNAPSHOT_ENTRY_CHARGE_BYTES.saturating_mul(snapshot.entries))
                .saturating_add(
                    WATCH_SNAPSHOT_NAME_BYTE_CHARGE_BYTES.saturating_mul(snapshot.name_bytes),
                )
                .saturating_add(
                    WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES
                        .saturating_mul(snapshot.plain_in_flight),
                )
                .saturating_add(watch_object_state_charge(
                    snapshot.in_flight,
                    snapshot.items,
                    snapshot.window_bytes,
                    rewrite,
                ))
                .saturating_add(queued_command_charge(queued))
        }
    }
}

/// What a plan may spend on its rewrite with `held` object-state bytes held:
/// `held` less the target of `need` with no rewrite, priced at `need`'s own
/// route's rewrite prices ([`RewritePrices::of`]), the same set
/// [`object_state_charge`] prices its rewrite at.
pub(crate) fn object_state_rewrite_allowance(
    held: u64,
    need: &ObjectStateNeed,
) -> MutationAllowance {
    let without_rewrite = object_state_charge(&need.with_rewrite(RewriteCount::default()));
    let prices = RewritePrices::of(need);
    MutationAllowance {
        bytes: held.saturating_sub(without_rewrite),
        row_bytes: prices.row,
        list_entry_bytes: prices.list_entry,
        index_entry_bytes: prices.name_index_entry,
        adoption_entry_bytes: prices.adoption_entry,
        walk_entry_bytes: prices.walk_entry,
    }
}

/// Test-support only: [`object_state_charge`], for a test binary's own
/// `CommitCharge` that must price a need as the production charge does (T7's
/// fits' `NeedRecorder`). Compiled out of the production binary.
#[cfg(any(test, feature = "test-support"))]
pub fn object_state_charge_for_test(need: &ObjectStateNeed) -> u64 {
    object_state_charge(need)
}

/// Test-support only: [`object_state_rewrite_allowance`], for the same test
/// charges as [`object_state_charge_for_test`]. Compiled out of the
/// production binary.
#[cfg(any(test, feature = "test-support"))]
pub fn object_state_rewrite_allowance_for_test(
    held: u64,
    need: &ObjectStateNeed,
) -> MutationAllowance {
    object_state_rewrite_allowance(held, need)
}

/// The largest body `budget` can carry through its receive phase:
/// `F + A + RECEIVE_CHARGE_FACTOR × L ≤ budget`.
fn largest_receivable(budget: u64) -> u64 {
    budget.saturating_sub(admission_charge()) / RECEIVE_CHARGE_FACTOR
}

/// SHA-256: the length of the hash an external-value reference carries.
const EXT_HASH_LEN: usize = 32;

/// The delta and body lengths, in bytes, of the canonical single-field blob
/// upload carrying a blob of `blob_len` bytes: an OMPF container of one delta
/// and that one external value, the delta one `SetField` that references the
/// value (`ext_value_index` 1, its 32-byte hash, `ext_value_size` =
/// `blob_len`) with an empty key name and every other field at its default.
///
/// Both come from the real encoders, `oodf_ffi::build_delta` and
/// [`OmpfContainer::serialize`](crate::domain::ompf::OmpfContainer::serialize).
/// The container is serialized around a one-byte stand-in for the blob: its
/// framing (magic, part count and one fixed-width offset per part) does not
/// depend on a part's length, so the body is that framing + delta + `blob_len`.
/// Pure, no I/O. `None` only if an encoder refuses the shape.
fn canonical_blob_upload_lengths(blob_len: u64) -> Option<(u64, u64)> {
    let delta = canonical_blob_delta(blob_len);
    let delta_bytes = oodf_ffi::build_delta(&delta).ok()?;
    let stand_in = [0u8; 1];
    let container =
        crate::domain::ompf::OmpfContainer::serialize(&delta_bytes, &[&stand_in]).ok()?;
    let delta_len = u64::try_from(delta_bytes.len()).ok()?;
    let framing = u64::try_from(container.len())
        .ok()?
        .checked_sub(delta_len)?
        .checked_sub(1)?;
    Some((
        delta_len,
        framing.checked_add(delta_len)?.checked_add(blob_len)?,
    ))
}

/// The canonical single-field blob upload's delta for a `blob_len`-byte blob
/// ([`canonical_blob_upload_lengths`]).
fn canonical_blob_delta(blob_len: u64) -> oodf_ffi::OwnedDelta {
    oodf_ffi::OwnedDelta {
        set_fields: vec![oodf_ffi::SetField {
            value_or_ext_hash: vec![0; EXT_HASH_LEN],
            ext_value_size: blob_len,
            ext_value_index: 1,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// The object-state term the canonical single-field blob upload's commit
/// holds on the WATCH path, which takes every `UpdateSlow` delta under `txn`:
/// the term of its commit point after the apply — the plan's rewrite and the
/// commands its transaction, its external-value pre-image read and its
/// keyspace probe queue ([`queued_command_charge`]) — with no fixed cost and
/// no read, the delta being fast-shaped. The plan is the real engine's,
/// applied to a header-only object (`apply_delta_bounded` under an unbounded
/// allowance, so its budget is never asked), and the queue counts are the
/// commit's own prediction
/// (`storage::layer::watch_queued_commands`), so the figure is exact for the
/// charge. Pure, no I/O. `None` only if the engine refuses the delta.
fn canonical_blob_watch_commit_term(blob_len: u64) -> Option<u64> {
    let header: HashMap<String, String> =
        [("seq", "1"), ("first_ver", "1"), ("next_set_order", "1")]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
    let mut state: crate::domain::types::CompactState =
        crate::storage::diff_read::header_to_object_state(&header, 1);
    let mut stats = crate::domain::delta::ApplyDeltaStats::default();
    let plan = crate::domain::delta::apply_delta_bounded(
        &mut state,
        Some(canonical_blob_delta(blob_len)),
        MutationAllowance::UNBOUNDED,
        &mut |_: RewriteCount| None::<u64>,
        &mut stats,
    )
    .ok()?;
    let crate::domain::delta::BoundedPlan::Within { mutations, rewrite } = plan else {
        return None;
    };
    let queued = crate::storage::layer::watch_queued_commands(&mutations, true);
    Some(object_state_charge(&ObjectStateNeed::Watch {
        snapshot: crate::storage::commit_charge::WatchSnapshot::default(),
        fixed: false,
        rewrite,
        queued,
    }))
}

/// Whether the canonical single-field blob upload of a `blob_len`-byte blob
/// ([`canonical_blob_upload_lengths`]) runs entirely within `budget`: its
/// receive phase (`F + A + RECEIVE_CHARGE_FACTOR × body`), its parse charge,
/// and its processing charge (one element, no reorder entries) on every
/// commit path — on `UpdateSlow` with the object-state term its WATCH commit
/// holds on top ([`canonical_blob_watch_commit_term`]).
fn canonical_blob_upload_fits(budget: u64, blob_len: u64) -> bool {
    let Some((delta_len, body_len)) = canonical_blob_upload_lengths(blob_len) else {
        return false;
    };
    let Some(watch_commit_term) = canonical_blob_watch_commit_term(blob_len) else {
        return false;
    };
    let receive_peak =
        admission_charge().saturating_add(RECEIVE_CHARGE_FACTOR.saturating_mul(body_len));
    let footprint = DeltaFootprint {
        elements: 1,
        reorder_entries: 0,
    };
    receive_peak <= budget
        && [
            CommitPath::Create,
            CommitPath::UpdateFast,
            CommitPath::UpdateSlow,
        ]
        .into_iter()
        .all(|path| {
            let charge = processing_charge(path, body_len, delta_len, blob_len, footprint);
            let charge = match path {
                CommitPath::UpdateSlow => charge.saturating_add(watch_commit_term),
                CommitPath::Create | CommitPath::UpdateFast => charge,
            };
            charge <= budget
        })
}

/// The largest blob, in bytes, whose canonical single-field blob upload fits
/// within `budget` on every commit path, rather than running in the overflow
/// slot; 0 when none does. A blob length: the upload itself is that plus its
/// delta and framing. On `UpdateSlow` the charge counted includes the
/// object-state term its WATCH commit holds after the apply
/// ([`canonical_blob_upload_fits`]).
///
/// Exact for that shape ([`canonical_blob_upload_lengths`]: one `SetField`
/// referencing the one external value, empty key name, defaults elsewhere):
/// an upload of that shape carrying a blob of exactly this size fits, and one
/// byte more does not. An upload with a larger delta — a non-empty key name,
/// a non-default field, more elements or blobs — is charged more, so a blob
/// of this size in it may not fit. Found by a binary search over the blob
/// length (the charges grow with it), bounded by what the receive phase can
/// carry: a few dozen encoder calls and plans of a tiny delta, run once at
/// startup.
pub fn largest_in_budget_blob_upload(budget: u64) -> u64 {
    if !canonical_blob_upload_fits(budget, 1) {
        return 0;
    }
    // Invariant: `low` fits; `high` is past the receive phase's own limit, so
    // it cannot.
    let (mut low, mut high) = (1u64, largest_receivable(budget).saturating_add(1));
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if canonical_blob_upload_fits(budget, mid) {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

/// The per-session receiving cap for `budget`: `max(1, ⌊B ÷ 4A⌋)`.
pub fn per_session_cap_for(budget: u64) -> usize {
    usize::try_from(budget / (4 * RECEIVE_ALLOWANCE_BYTES))
        .unwrap_or(usize::MAX)
        .max(1)
}

/// The gate's settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadAdmissionConfig {
    /// `B`, in bytes, derived from the pod memory limit
    /// (`config::memory_plan`). At least [`upload_budget_floor_bytes`] of
    /// `max_upload_bytes`.
    pub budget_bytes: u64,
    /// The admission wait (`LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS`). Zero means
    /// never wait.
    pub wait: Duration,
    /// The admission queue's bound (`LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`).
    pub max_waiters: usize,
    /// The largest upload body the service accepts
    /// (`LIVE_EDIT_MAX_LAYER_SIZE`); the budget is floored by it.
    pub max_upload_bytes: u64,
}

impl Default for UploadAdmissionConfig {
    /// The production defaults at the default 30 s RPC deadline and size cap
    /// and the reference pod memory limit: the derived wait is its 2000 ms
    /// ceiling and the derived budget 384 MiB.
    fn default() -> Self {
        Self {
            budget_bytes: upload_memory_budget_mib_default(MAX_LAYER_SIZE_DEFAULT_BYTES) * MIB,
            wait: Duration::from_millis(UPLOAD_ADMISSION_WAIT_MS_DEFAULT),
            max_waiters: UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            max_upload_bytes: MAX_LAYER_SIZE_DEFAULT_BYTES,
        }
    }
}

/// Why an upload was refused a reservation, or refused growth of one. No
/// refusal is for the upload's size: the budget floor and the overflow slot
/// make that unreachable under `LIVE_EDIT_MAX_LAYER_SIZE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadRefusal {
    /// Capacity stayed full for the whole admission wait (or the wait is 0):
    /// the memory budget, or the session's cap on uploads still receiving
    /// their content.
    Saturated,
    /// The admission queue is at `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`.
    WaiterCap,
    /// The service is shutting down: refused at admission or while waiting
    /// for processing room.
    ShuttingDown,
    /// The budget cannot hold an in-budget growth of an admitted reservation
    /// right now ([`UploadReservation::try_grow_in_budget`]): a receive-phase
    /// growth, which refuses the upload, or a best-effort growth after the
    /// body, such as a `CreateLayer` re-send's re-spawn charges, which only
    /// skips that work.
    Growth,
}

impl UploadRefusal {
    /// `result` label on `live_edit_upload_admission_total`. A
    /// [`UploadRefusal::Growth`] is counted as `refused_growth` only while the
    /// upload is still receiving its content; a growth refused after
    /// [`UploadReservation::end_receive`] is not counted.
    pub fn metric_result(self) -> &'static str {
        match self {
            UploadRefusal::Saturated => ADMISSION_RESULT_REFUSED_SATURATED,
            UploadRefusal::WaiterCap => ADMISSION_RESULT_REFUSED_WAITER_CAP,
            UploadRefusal::ShuttingDown => ADMISSION_RESULT_CANCELLED,
            UploadRefusal::Growth => ADMISSION_RESULT_REFUSED_GROWTH,
        }
    }

    /// The error a refusal answers: always `UNAVAILABLE` — this call applied
    /// nothing; what that promises a client about a re-send is stated once, at
    /// [`REFUSAL_DETAIL_DOMAIN`](crate::error::REFUSAL_DETAIL_DOMAIN) — with a
    /// retry hint ([`ServiceError::UnavailableWithDetail`]): a capacity
    /// refusal carries
    /// [`UPLOAD_CAPACITY`](refusal_reason::UPLOAD_CAPACITY), metadata `cause`
    /// = `saturated` / `waiter_cap` / `growth`. A shutdown refusal maps to
    /// the shutdown status's message and detail here too, identical on the
    /// wire to the shared shutdown status
    /// (`crate::grpc::transport::server_shutting_down`): the handlers' own
    /// admission and processing waits answer that status directly, and the
    /// in-commit recharge wait ([`UploadCommitCharge`]), which answers
    /// through the storage call as a `ServiceError`, answers this mapping.
    /// Either way the completion line is marked shutdown-caused.
    pub fn into_service_error(self) -> ServiceError {
        let capacity = |cause: &str, message: &str| {
            RefusalDetail::new(refusal_reason::UPLOAD_CAPACITY, Some(CAPACITY_RETRY_DELAY))
                .with_metadata(refusal_metadata::CAUSE, cause)
                .into_unavailable_error(message)
        };
        match self {
            UploadRefusal::Saturated => capacity(
                refusal_metadata::SATURATED,
                "this replica's upload capacity (its memory budget, sized from the pod's memory \
                 limit, or this session's limit on uploads still sending content) stayed full \
                 for the whole admission wait; nothing was applied, retry",
            ),
            UploadRefusal::WaiterCap => capacity(
                refusal_metadata::WAITER_CAP,
                "this replica's upload admission queue is full \
                 (LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS); nothing was applied, retry",
            ),
            UploadRefusal::Growth => capacity(
                refusal_metadata::GROWTH,
                "this replica's upload memory budget (sized from the pod's memory limit) cannot \
                 hold this upload's content right now; nothing was applied, retry",
            ),
            UploadRefusal::ShuttingDown => RefusalDetail::server_shutting_down()
                .into_unavailable_error(crate::grpc::transport::SERVER_SHUTTING_DOWN_MSG),
        }
    }
}

/// One queued admission's hand-off cell: the gate sets `granted` under its
/// lock, after reserving for the waiter, and then notifies it.
#[derive(Default)]
struct WaiterSignal {
    granted: AtomicBool,
    notify: Notify,
}

struct QueuedWaiter {
    id: u64,
    session: Arc<str>,
    signal: Arc<WaiterSignal>,
}

/// [`ProcessingSignal::state`]: not granted yet.
const PROCESSING_PENDING: u8 = 0;
/// [`ProcessingSignal::state`]: granted in budget.
const PROCESSING_IN_BUDGET: u8 = 1;
/// [`ProcessingSignal::state`]: granted the overflow slot.
const PROCESSING_OVERFLOW: u8 = 2;

/// One queued processing step's hand-off cell: the gate books the grant under
/// its lock, records which kind it was in `state`, and then notifies it.
#[derive(Default)]
struct ProcessingSignal {
    state: AtomicU8,
    notify: Notify,
}

struct ProcessingWaiter {
    /// The waiting reservation's id.
    reservation: u64,
    /// What the reservation holds in budget while it waits.
    held: u64,
    /// What it waits to hold.
    target: u64,
    signal: Arc<ProcessingSignal>,
}

/// The overflow slot's holder.
#[derive(Debug, Clone, Copy)]
struct Overflow {
    reservation: u64,
    bytes: u64,
}

#[derive(Default)]
struct GateState {
    /// Bytes reserved in budget by every live reservation, granted waiters
    /// included. Never above `B`.
    reserved: u64,
    /// The highest `reserved` has ever been.
    high_water: u64,
    /// The one upload charged beyond `B`, if any.
    overflow: Option<Overflow>,
    /// The highest an overflow holder's charge has ever been.
    overflow_high_water: u64,
    /// Uploads still receiving their content, per session. An entry is
    /// removed when it reaches zero, so the map holds at most `⌊B ÷ A⌋`
    /// entries.
    receiving: HashMap<Arc<str>, usize>,
    /// Admissions waiting, oldest first.
    queue: VecDeque<QueuedWaiter>,
    next_waiter_id: u64,
    /// Processing steps waiting for room in budget or for the overflow slot,
    /// oldest first.
    processing: VecDeque<ProcessingWaiter>,
    /// The test-only charge observer
    /// ([`UploadAdmissionGate::set_charge_observer_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    charge_observer: Option<ChargeObserver>,
}

impl GateState {
    fn receiving_in(&self, session: &str) -> usize {
        self.receiving.get(session).copied().unwrap_or(0)
    }

    /// Report the gate's charges to the test-only observer, if one is set.
    /// Under the lock, after every change to them.
    #[cfg(any(test, feature = "test-support"))]
    fn charge_changed(&self) {
        if let Some(observer) = &self.charge_observer {
            observer(GateCharge {
                reserved_bytes: self.reserved,
                overflow_bytes: self.overflow.map_or(0, |o| o.bytes),
            });
        }
    }

    /// No observer outside tests.
    #[cfg(not(any(test, feature = "test-support")))]
    fn charge_changed(&self) {}
}

/// What the gate holds after a change, as the test-only charge observer sees
/// it ([`UploadAdmissionGate::set_charge_observer_for_test`]).
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateCharge {
    /// Bytes reserved in budget, by every live reservation.
    pub reserved_bytes: u64,
    /// The overflow slot holder's charge; 0 when the slot is free.
    pub overflow_bytes: u64,
}

/// The test-only charge observer's type
/// ([`UploadAdmissionGate::set_charge_observer_for_test`]).
#[cfg(any(test, feature = "test-support"))]
pub type ChargeObserver = Arc<dyn Fn(GateCharge) + Send + Sync>;

/// The per-replica upload budget and its overflow slot; see the module docs.
pub struct UploadAdmissionGate {
    budget: u64,
    per_session_cap: usize,
    wait: Duration,
    max_waiters: usize,
    max_upload_bytes: u64,
    next_reservation_id: AtomicU64,
    state: Mutex<GateState>,
    metrics: Option<SharedMetrics>,
}

impl std::fmt::Debug for UploadAdmissionGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadAdmissionGate")
            .field("budget", &self.budget)
            .field("per_session_cap", &self.per_session_cap)
            .field("wait", &self.wait)
            .field("max_waiters", &self.max_waiters)
            .field("max_upload_bytes", &self.max_upload_bytes)
            .field("reserved", &self.reserved_bytes())
            .field("overflow", &self.overflow_bytes())
            .field("waiters", &self.waiters())
            .field("processing_waiters", &self.processing_waiters())
            .finish()
    }
}

/// How a queued admission's wait ended.
enum WaitEnd {
    Granted,
    Deadline,
    Shutdown,
}

/// How a processing step ended, for the caller's log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessingGrant {
    /// This step moved the upload into the overflow slot: it now runs beyond
    /// the budget. False when it was already there, or stayed in budget.
    pub entered_overflow: bool,
    /// How long the step waited for room or for the slot.
    pub waited: Duration,
}

impl UploadAdmissionGate {
    /// A gate over `config`. Records the budget on
    /// `live_edit_upload_budget_bytes`.
    ///
    /// # Panics
    ///
    /// When `config.budget_bytes` is below [`upload_budget_floor_bytes`] of
    /// `config.max_upload_bytes`: such a gate could refuse an upload under the
    /// size cap for its size. Configuration rejects that at startup.
    pub fn new(config: UploadAdmissionConfig, metrics: Option<SharedMetrics>) -> Self {
        let floor = upload_budget_floor_bytes(config.max_upload_bytes);
        assert!(
            config.budget_bytes >= floor,
            "upload budget {} bytes is below its floor of {floor} bytes for a {}-byte size cap",
            config.budget_bytes,
            config.max_upload_bytes
        );
        if let Some(m) = &metrics {
            m.set_upload_budget_bytes(config.budget_bytes);
        }
        Self {
            budget: config.budget_bytes,
            per_session_cap: per_session_cap_for(config.budget_bytes),
            wait: config.wait,
            max_waiters: config.max_waiters,
            max_upload_bytes: config.max_upload_bytes,
            next_reservation_id: AtomicU64::new(0),
            state: Mutex::new(GateState::default()),
            metrics,
        }
    }

    /// `B`, in bytes.
    pub fn budget_bytes(&self) -> u64 {
        self.budget
    }

    /// The size cap the budget is floored by.
    pub fn max_upload_bytes(&self) -> u64 {
        self.max_upload_bytes
    }

    /// Uploads one session may have receiving their content at once.
    pub fn per_session_cap(&self) -> usize {
        self.per_session_cap
    }

    /// The admission wait.
    pub fn wait(&self) -> Duration {
        self.wait
    }

    /// The admission queue's bound.
    pub fn max_waiters(&self) -> usize {
        self.max_waiters
    }

    /// Bytes reserved in budget right now.
    pub fn reserved_bytes(&self) -> u64 {
        self.lock().reserved
    }

    /// The highest the in-budget reservation total has ever been. Never above
    /// [`Self::budget_bytes`].
    pub fn high_water_bytes(&self) -> u64 {
        self.lock().high_water
    }

    /// The overflow slot holder's charge right now; 0 when the slot is free.
    pub fn overflow_bytes(&self) -> u64 {
        self.lock().overflow.map_or(0, |o| o.bytes)
    }

    /// The highest an overflow slot holder's charge has ever been.
    pub fn overflow_high_water_bytes(&self) -> u64 {
        self.lock().overflow_high_water
    }

    /// Admissions queued right now.
    pub fn waiters(&self) -> usize {
        self.lock().queue.len()
    }

    /// Processing steps queued for room or the overflow slot right now.
    pub fn processing_waiters(&self) -> usize {
        self.lock().processing.len()
    }

    /// Uploads of `session_id` still receiving their content right now.
    pub fn receiving_uploads(&self, session_id: &str) -> usize {
        self.lock().receiving_in(session_id)
    }

    /// Set — or, with `None`, clear — the gate's charge observer: called under
    /// the gate's lock after every change to what the gate holds (a
    /// reservation reserved, grown, shrunk or released, in budget or on the
    /// overflow slot), with the totals that change leaves ([`GateCharge`]).
    /// One operation can report more than one change in one lock hold — a
    /// step into the overflow slot reports its in-budget bytes leaving `B`,
    /// then the slot taken — and the last is the state it leaves.
    ///
    /// For T7 (`tests/upload_memory_envelope/`), which runs one upload at a
    /// time, so the totals are that upload's charge: it closes its allocator
    /// window and opens the next at each report, and checks each window's
    /// peak against the charge in force during it. The observer runs under
    /// the gate's lock, so it must not call the gate.
    ///
    /// Test-only: compiled only under `cfg(test)` or the `test-support`
    /// feature, never into the production binary.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_charge_observer_for_test(&self, observer: Option<ChargeObserver>) {
        self.lock().charge_observer = observer;
    }

    fn lock(&self) -> MutexGuard<'_, GateState> {
        // Nothing panics while holding the lock; recover the state rather than
        // wedging every upload if something ever did.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Admit one upload of `session_id`: reserve `F + A` and take one of the
    /// session's receiving slots.
    ///
    /// Immediate when the budget has room, the session is under its cap and
    /// no waiter the gate could serve is queued ahead. Otherwise the call
    /// joins the queue and waits, holding nothing from the gate, until it is
    /// granted, the admission wait ends ([`UploadRefusal::Saturated`]) or
    /// `shutdown` fires ([`UploadRefusal::ShuttingDown`]). A wait of zero never
    /// queues; a full queue refuses [`UploadRefusal::WaiterCap`] at once.
    /// Dropping the returned future leaves the queue (or releases a grant that
    /// raced the drop) and serves the next waiter.
    ///
    /// Records one `live_edit_upload_admission_total{rpc, result}` and one
    /// `live_edit_upload_admission_wait_seconds{rpc}` observation per call,
    /// a dropped wait included (`cancelled`).
    pub async fn admit(
        self: &Arc<Self>,
        rpc: UploadRpc,
        session_id: &str,
        shutdown: &CancellationToken,
    ) -> Result<UploadReservation, UploadRefusal> {
        let started = tokio::time::Instant::now();
        // At most `B`: the constructor's floor is at least `F + A`.
        let need = admission_charge();
        if shutdown.is_cancelled() {
            return Err(self.refuse(rpc, UploadRefusal::ShuttingDown, started));
        }
        let session: Arc<str> = Arc::from(session_id);

        let (id, signal) = {
            let mut st = self.lock();
            let session_has_room = st.receiving_in(&session) < self.per_session_cap;
            let bytes_fit = st.reserved.saturating_add(need) <= self.budget;
            // A queued waiter the gate could serve (its session has room) is
            // held back by bytes: it keeps its place ahead of this arrival.
            let servable_waiter_queued = st
                .queue
                .iter()
                .any(|w| st.receiving_in(&w.session) < self.per_session_cap);
            if session_has_room && bytes_fit && !servable_waiter_queued {
                self.grant_locked(&mut st, &session, need);
                drop(st);
                return Ok(self.admitted(rpc, session, need, started));
            }
            if self.wait.is_zero() {
                drop(st);
                return Err(self.refuse(rpc, UploadRefusal::Saturated, started));
            }
            if st.queue.len() >= self.max_waiters {
                drop(st);
                return Err(self.refuse(rpc, UploadRefusal::WaiterCap, started));
            }
            let id = st.next_waiter_id;
            st.next_waiter_id += 1;
            let signal = Arc::new(WaiterSignal::default());
            st.queue.push_back(QueuedWaiter {
                id,
                session: Arc::clone(&session),
                signal: Arc::clone(&signal),
            });
            self.add_waiters(1);
            (id, signal)
        };

        let mut pending = PendingAdmission {
            gate: self,
            rpc,
            id,
            signal: Arc::clone(&signal),
            session: Arc::clone(&session),
            started,
            settled: false,
        };
        // A wait too long for the clock to represent ends decades away rather
        // than panicking (configuration bounds it below the RPC deadline).
        let deadline = started
            .checked_add(self.wait)
            .unwrap_or_else(|| started + FAR_FUTURE);
        let end = loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break WaitEnd::Shutdown,
                () = signal.notify.notified() => {
                    if signal.granted.load(Ordering::Acquire) {
                        break WaitEnd::Granted;
                    }
                }
                () = tokio::time::sleep_until(deadline) => break WaitEnd::Deadline,
            }
        };
        pending.settled = true;
        match end {
            WaitEnd::Granted => Ok(self.admitted(rpc, session, need, started)),
            WaitEnd::Deadline => {
                if self.leave_queue(id, &signal, &session, false) {
                    // Granted at the deadline: the grant stands.
                    Ok(self.admitted(rpc, session, need, started))
                } else {
                    Err(self.refuse(rpc, UploadRefusal::Saturated, started))
                }
            }
            WaitEnd::Shutdown => {
                self.leave_queue(id, &signal, &session, true);
                Err(self.refuse(rpc, UploadRefusal::ShuttingDown, started))
            }
        }
    }

    /// Take waiter `id` out of the queue. Returns whether it had already been
    /// granted; a grant is released when `release_grant` is set and kept
    /// otherwise. Serves the queues either way.
    fn leave_queue(
        &self,
        id: u64,
        signal: &WaiterSignal,
        session: &Arc<str>,
        release_grant: bool,
    ) -> bool {
        let mut st = self.lock();
        let granted = signal.granted.load(Ordering::Acquire);
        if granted {
            if release_grant {
                self.release_locked(&mut st, admission_charge(), Some(session));
            }
        } else if let Some(at) = st.queue.iter().position(|w| w.id == id) {
            st.queue.remove(at);
            self.add_waiters(-1);
        }
        self.serve_all(&mut st);
        granted
    }

    /// Reserve `need` for one admission of `session`. Under the lock.
    fn grant_locked(&self, st: &mut GateState, session: &Arc<str>, need: u64) {
        self.grow_locked(st, need);
        *st.receiving.entry(Arc::clone(session)).or_insert(0) += 1;
    }

    /// Add `bytes` to the in-budget total. Under the lock; the caller has
    /// checked they fit.
    fn grow_locked(&self, st: &mut GateState, bytes: u64) {
        st.reserved = st.reserved.saturating_add(bytes);
        st.high_water = st.high_water.max(st.reserved);
        self.add_reserved(bytes);
        st.charge_changed();
    }

    /// Release `bytes` from the budget, and a receiving slot of `session` when
    /// given. Under the lock; the caller serves the queues after.
    fn release_locked(&self, st: &mut GateState, bytes: u64, session: Option<&Arc<str>>) {
        st.reserved = st.reserved.saturating_sub(bytes);
        self.add_reserved_negative(bytes);
        st.charge_changed();
        if let Some(session) = session {
            if let Some(count) = st.receiving.get_mut(&**session) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    st.receiving.remove(&**session);
                }
            }
        }
    }

    /// Put `reservation` in the overflow slot at `bytes`. Under the lock; the
    /// slot is free.
    ///
    /// Does not count `live_edit_upload_overflow_runs_total`: a slot handed to
    /// a queued waiter may never be taken (shutdown or a dropped call can end
    /// the wait first), so the run is counted where the reservation takes the
    /// grant ([`Self::count_overflow_run`]).
    fn occupy_overflow_locked(&self, st: &mut GateState, reservation: u64, bytes: u64) {
        st.overflow = Some(Overflow { reservation, bytes });
        st.overflow_high_water = st.overflow_high_water.max(bytes);
        if let Some(m) = &self.metrics {
            m.set_upload_overflow_bytes(bytes);
        }
        st.charge_changed();
    }

    /// Count one upload that took the overflow slot and goes on to run beyond
    /// the budget: one `live_edit_upload_overflow_runs_total{rpc}`, paired
    /// with the [`ProcessingGrant::entered_overflow`] the caller logs.
    fn count_overflow_run(&self, rpc: UploadRpc) {
        if let Some(m) = &self.metrics {
            m.inc_upload_overflow_runs(rpc.label());
        }
    }

    /// Re-book the overflow holder at `bytes`. Under the lock.
    fn rebook_overflow_locked(&self, st: &mut GateState, bytes: u64) {
        if let Some(overflow) = st.overflow.as_mut() {
            overflow.bytes = bytes;
            st.overflow_high_water = st.overflow_high_water.max(bytes);
            if let Some(m) = &self.metrics {
                m.set_upload_overflow_bytes(bytes);
            }
            st.charge_changed();
        }
    }

    /// Free the overflow slot if `reservation` holds it. Under the lock; the
    /// caller serves the queues after.
    fn vacate_overflow_locked(&self, st: &mut GateState, reservation: u64) {
        if st.overflow.is_some_and(|o| o.reservation == reservation) {
            st.overflow = None;
            if let Some(m) = &self.metrics {
                m.set_upload_overflow_bytes(0);
            }
            st.charge_changed();
        }
    }

    /// Serve both queues after a release. Under the lock.
    ///
    /// Processing first — work in progress goes before new work: each queued
    /// processing step that now fits in budget is granted there; then a free
    /// overflow slot goes to the processing queue's head, whose in-budget
    /// bytes leave `B` with it, so the in-budget pass runs again over the
    /// room that frees; then admissions. Without that second pass a new
    /// admission could take room a queued processing step now fits into.
    fn serve_all(&self, st: &mut GateState) {
        self.grant_processing_in_budget_locked(st);
        if st.overflow.is_none() {
            if let Some(head) = st.processing.pop_front() {
                // Its in-budget bytes move out of `B` onto the slot.
                self.release_locked(st, head.held, None);
                self.occupy_overflow_locked(st, head.reservation, head.target);
                self.add_processing_waiters(-1);
                head.signal
                    .state
                    .store(PROCESSING_OVERFLOW, Ordering::Release);
                head.signal.notify.notify_one();
                self.grant_processing_in_budget_locked(st);
            }
        }
        self.serve_queue(st);
    }

    /// Grant, in queue order, each queued processing step whose growth now
    /// fits in budget; one that does not fit keeps its place and is skipped.
    /// Under the lock.
    fn grant_processing_in_budget_locked(&self, st: &mut GateState) {
        let mut at = 0;
        while at < st.processing.len() {
            let waiter = &st.processing[at];
            let more = waiter.target.saturating_sub(waiter.held);
            if st.reserved.saturating_add(more) > self.budget {
                at += 1;
                continue;
            }
            let Some(waiter) = st.processing.remove(at) else {
                break;
            };
            self.grow_locked(st, more);
            self.add_processing_waiters(-1);
            waiter
                .signal
                .state
                .store(PROCESSING_IN_BUDGET, Ordering::Release);
            waiter.signal.notify.notify_one();
        }
    }

    /// Grant queued admissions in order: skip one whose session is at its
    /// cap, stop at the first the budget cannot hold. Under the lock.
    fn serve_queue(&self, st: &mut GateState) {
        let need = admission_charge();
        let mut at = 0;
        while at < st.queue.len() {
            if st.receiving_in(&st.queue[at].session) >= self.per_session_cap {
                at += 1;
                continue;
            }
            if st.reserved.saturating_add(need) > self.budget {
                break;
            }
            let Some(waiter) = st.queue.remove(at) else {
                break;
            };
            self.grant_locked(st, &waiter.session, need);
            self.add_waiters(-1);
            waiter.signal.granted.store(true, Ordering::Release);
            waiter.signal.notify.notify_one();
        }
    }

    /// A sampled audit's charge ([`UploadBudgetAuditCharges`]) holding
    /// `held` bytes in budget asks to hold exactly `target`, never waiting.
    /// Returns whether `target` is now held; `held` is updated under the lock.
    ///
    /// - Above `cap`: refused.
    /// - At or below what it holds: granted; a lower target gives the
    ///   difference back and serves the queues (the exact trim).
    /// - A growth is granted only when no upload waits — neither an admission
    ///   nor a processing step is queued, so the audit never goes ahead of
    ///   one — and when the budget keeps one admission's room
    ///   ([`admission_charge`]) free after it. It never takes the overflow
    ///   slot: an audit runs inside `B` or not at all.
    ///
    /// Moves only the in-budget total and `live_edit_upload_reserved_bytes`:
    /// no admission, waiter or overflow metric.
    fn try_hold_audit(&self, held: &AtomicU64, target: u64, cap: u64) -> bool {
        if target > cap {
            return false;
        }
        let mut st = self.lock();
        let now = held.load(Ordering::Acquire);
        if target <= now {
            if target < now {
                self.release_locked(&mut st, now - target, None);
                held.store(target, Ordering::Release);
                self.serve_all(&mut st);
            }
            return true;
        }
        if !st.queue.is_empty() || !st.processing.is_empty() {
            return false;
        }
        let more = target - now;
        if st
            .reserved
            .saturating_add(more)
            .saturating_add(admission_charge())
            > self.budget
        {
            return false;
        }
        self.grow_locked(&mut st, more);
        held.store(target, Ordering::Release);
        true
    }

    /// Give back the `bytes` a dropped audit charge held, and serve the
    /// queues.
    fn release_audit(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut st = self.lock();
        self.release_locked(&mut st, bytes, None);
        self.serve_all(&mut st);
    }

    /// The reservation for an admission whose `need` is already reserved.
    fn admitted(
        self: &Arc<Self>,
        rpc: UploadRpc,
        session: Arc<str>,
        need: u64,
        started: tokio::time::Instant,
    ) -> UploadReservation {
        self.observe_admission(rpc, ADMISSION_RESULT_ADMITTED, started);
        if let Some(m) = &self.metrics {
            m.add_uploads_in_flight(UPLOAD_PHASE_RECEIVING, 1);
        }
        UploadReservation {
            gate: Arc::clone(self),
            id: self.next_reservation_id.fetch_add(1, Ordering::Relaxed),
            rpc,
            session,
            reserved: need,
            receiving: true,
            in_overflow: false,
        }
    }

    fn refuse(
        &self,
        rpc: UploadRpc,
        refusal: UploadRefusal,
        started: tokio::time::Instant,
    ) -> UploadRefusal {
        self.observe_admission(rpc, refusal.metric_result(), started);
        refusal
    }

    fn observe_admission(
        &self,
        rpc: UploadRpc,
        result: &'static str,
        started: tokio::time::Instant,
    ) {
        if let Some(m) = &self.metrics {
            m.observe_upload_admission(rpc.label(), result, started.elapsed());
        }
    }

    fn count_refusal(&self, rpc: UploadRpc, result: &'static str) {
        if let Some(m) = &self.metrics {
            m.inc_upload_admission_refusal(rpc.label(), result);
        }
    }

    fn observe_processing_wait(&self, rpc: UploadRpc, waited: Duration) {
        if let Some(m) = &self.metrics {
            m.observe_upload_processing_wait(rpc.label(), waited);
        }
    }

    fn add_waiters(&self, delta: i64) {
        if let Some(m) = &self.metrics {
            m.add_upload_admission_waiters(delta);
        }
    }

    fn add_processing_waiters(&self, delta: i64) {
        if let Some(m) = &self.metrics {
            m.add_upload_overflow_waiters(delta);
        }
    }

    fn add_reserved(&self, bytes: u64) {
        if let Some(m) = &self.metrics {
            m.add_upload_reserved_bytes(i64::try_from(bytes).unwrap_or(i64::MAX));
        }
    }

    fn add_reserved_negative(&self, bytes: u64) {
        if let Some(m) = &self.metrics {
            m.add_upload_reserved_bytes(-i64::try_from(bytes).unwrap_or(i64::MAX));
        }
    }
}

/// A queued admission in flight. If the `admit` future is dropped before its
/// wait settles, this leaves the queue — or releases a grant that raced the
/// drop — and serves the next waiter.
struct PendingAdmission<'a> {
    gate: &'a Arc<UploadAdmissionGate>,
    rpc: UploadRpc,
    id: u64,
    signal: Arc<WaiterSignal>,
    session: Arc<str>,
    started: tokio::time::Instant,
    settled: bool,
}

impl Drop for PendingAdmission<'_> {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        self.gate
            .leave_queue(self.id, &self.signal, &self.session, true);
        self.gate
            .observe_admission(self.rpc, ADMISSION_RESULT_CANCELLED, self.started);
    }
}

/// One admitted upload's reservation. Grows and shrinks with the upload, in
/// budget or — once a processing step moved it there — in the overflow slot.
/// Its `Drop` releases every byte it holds, the slot if it holds it, and the
/// session slot if the upload was still receiving.
pub struct UploadReservation {
    gate: Arc<UploadAdmissionGate>,
    /// Unique per gate; names the overflow slot's holder.
    id: u64,
    rpc: UploadRpc,
    session: Arc<str>,
    /// Bytes held: in budget, or on the overflow slot when `in_overflow`.
    reserved: u64,
    receiving: bool,
    in_overflow: bool,
}

impl std::fmt::Debug for UploadReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadReservation")
            .field("rpc", &self.rpc)
            .field("reserved", &self.reserved)
            .field("receiving", &self.receiving)
            .field("in_overflow", &self.in_overflow)
            .finish()
    }
}

impl UploadReservation {
    /// The RPC this reservation belongs to.
    pub fn rpc(&self) -> UploadRpc {
        self.rpc
    }

    /// Bytes this reservation holds, in budget or on the overflow slot.
    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    /// Whether the upload still holds its receive allowance and session slot.
    pub fn is_receiving(&self) -> bool {
        self.receiving
    }

    /// Whether the upload holds the overflow slot: it runs beyond the budget.
    pub fn in_overflow(&self) -> bool {
        self.in_overflow
    }

    /// Grow the reservation to `target` bytes in budget, at once or not at all
    /// — it never waits, and it ignores both queues. A target at or below the
    /// current reservation changes nothing; the overflow holder books the
    /// growth on its slot instead.
    ///
    /// Refuses [`UploadRefusal::Growth`] when the budget cannot hold the
    /// growth now. A refusal changes nothing. While the upload is still
    /// receiving, a refusal counts one `live_edit_upload_admission_total{rpc,
    /// result="refused_growth"}`: that is the receive growth the client is
    /// refused for. A refusal after [`Self::end_receive`] — a best-effort
    /// growth no client is refused for — counts nothing.
    pub fn try_grow_in_budget(&mut self, target: u64) -> Result<(), UploadRefusal> {
        if target <= self.reserved {
            return Ok(());
        }
        let gate = Arc::clone(&self.gate);
        let mut st = gate.lock();
        if self.in_overflow {
            gate.rebook_overflow_locked(&mut st, target);
            self.reserved = target;
            return Ok(());
        }
        let others = st.reserved.saturating_sub(self.reserved);
        if others.saturating_add(target) > gate.budget {
            drop(st);
            if self.receiving {
                gate.count_refusal(self.rpc, ADMISSION_RESULT_REFUSED_GROWTH);
            }
            return Err(UploadRefusal::Growth);
        }
        gate.grow_locked(&mut st, target - self.reserved);
        self.reserved = target;
        Ok(())
    }

    /// One processing step: hold at least `target` bytes, in budget or in the
    /// overflow slot. Never refused for size.
    ///
    /// In order:
    ///
    /// 1. a target at or below what the reservation holds changes nothing;
    /// 2. the overflow holder books the target on its slot in place, without
    ///    waiting;
    /// 3. a target that fits the budget grows the reservation in budget;
    /// 4. otherwise, when the slot is free and no processing step is queued,
    ///    the reservation takes the slot, its in-budget bytes moving out of
    ///    the budget onto it;
    /// 5. otherwise it queues and waits for whichever comes first: room in the
    ///    budget, or reaching the queue's head with the slot free.
    ///
    /// The wait has no bound of its own — the RPC deadline drops it — and
    /// races `shutdown`, which refuses [`UploadRefusal::ShuttingDown`]. A wait
    /// ended by shutdown or by dropping the future leaves the queue; a grant
    /// that raced it is released, except that a slot grant keeps the slot at
    /// what the reservation held before, until the reservation drops. Either
    /// ending counts one `live_edit_upload_admission_total{rpc,
    /// result="cancelled"}`.
    ///
    /// Records one `live_edit_upload_processing_wait_seconds{rpc}` observation
    /// per call that grows the reservation, ~0 when granted at once. Counts one
    /// `live_edit_upload_overflow_runs_total{rpc}` exactly when it returns
    /// [`ProcessingGrant::entered_overflow`]; a slot grant that a shutdown or a
    /// dropped call ended first is not counted.
    pub async fn reserve_processing(
        &mut self,
        target: u64,
        shutdown: &CancellationToken,
    ) -> Result<ProcessingGrant, UploadRefusal> {
        let started = tokio::time::Instant::now();
        let at_once = ProcessingGrant {
            entered_overflow: false,
            waited: Duration::ZERO,
        };
        if target <= self.reserved {
            return Ok(at_once);
        }
        let gate = Arc::clone(&self.gate);
        let signal = {
            let mut st = gate.lock();
            if let Some(entered_overflow) = self.grant_processing_at_once_locked(&mut st, target) {
                drop(st);
                return Ok(self.granted_at_once(entered_overflow, started));
            }
            let signal = Arc::new(ProcessingSignal::default());
            st.processing.push_back(ProcessingWaiter {
                reservation: self.id,
                held: self.reserved,
                target,
                signal: Arc::clone(&signal),
            });
            gate.add_processing_waiters(1);
            signal
        };

        let mut pending = PendingProcessing {
            reservation: self,
            signal: Arc::clone(&signal),
            target,
            started,
            settled: false,
        };
        let shut_down = loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break true,
                () = signal.notify.notified() => {
                    if signal.state.load(Ordering::Acquire) != PROCESSING_PENDING {
                        break false;
                    }
                }
            }
        };
        pending.settled = true;
        if shut_down {
            pending.abandon();
            return Err(UploadRefusal::ShuttingDown);
        }
        let entered_overflow = pending.take_grant();
        let waited = started.elapsed();
        gate.observe_processing_wait(pending.reservation.rpc, waited);
        Ok(ProcessingGrant {
            entered_overflow,
            waited,
        })
    }

    /// A processing step that never waits: hold exactly `target` bytes, at
    /// once or not at all. For a charge taken where waiting on the gate is
    /// forbidden (the lock order in the module docs), so it never queues and
    /// never waits.
    ///
    /// In order:
    ///
    /// 1. a target equal to what the reservation holds changes nothing, and a
    ///    target below it shrinks the reservation to it, giving the
    ///    difference back to the gate ([`Self::shrink_to`]);
    /// 2. the overflow holder books the target on its slot in place;
    /// 3. a target that fits the budget grows the reservation in budget;
    /// 4. otherwise, when the slot is free and no processing step is queued,
    ///    the reservation takes the slot, its in-budget bytes moving out of
    ///    the budget onto it;
    /// 5. otherwise it is refused: `None`, and nothing changed.
    ///
    /// Steps 2–4 are [`Self::reserve_processing`]'s, with one of its metrics:
    /// one `live_edit_upload_overflow_runs_total{rpc}` exactly when it
    /// returns [`ProcessingGrant::entered_overflow`]. It records no
    /// `live_edit_upload_processing_wait_seconds` observation: it never
    /// waits, and a commit calls it at every charge
    /// point, so its ~0 observations would dilute the histogram, whose unit
    /// is a real wait (module docs). As after [`Self::reserve_processing`],
    /// the caller logs the entry into the slot
    /// ([`log_upload_runs_beyond_budget`]). A shrink or a refusal records
    /// nothing.
    pub fn try_hold_processing(&mut self, target: u64) -> Option<ProcessingGrant> {
        if target <= self.reserved {
            self.shrink_to(target);
            return Some(ProcessingGrant {
                entered_overflow: false,
                waited: Duration::ZERO,
            });
        }
        let gate = Arc::clone(&self.gate);
        let mut st = gate.lock();
        let entered_overflow = self.grant_processing_at_once_locked(&mut st, target)?;
        drop(st);
        if entered_overflow {
            self.gate.count_overflow_run(self.rpc);
        }
        Some(ProcessingGrant {
            entered_overflow,
            waited: Duration::ZERO,
        })
    }

    /// Steps 2–4 of [`Self::reserve_processing`], under the gate's lock, for a
    /// `target` above what the reservation holds: book it on the slot in
    /// place, grow in budget when it fits, or take the free slot when no
    /// processing step is queued. `Some(entered_overflow)` when granted;
    /// `None`, with nothing changed, when the step would have to queue. The
    /// caller drops the lock and then records the grant's metrics:
    /// [`Self::reserve_processing`] through [`Self::granted_at_once`] (the
    /// overflow run and a ~0 processing wait), and
    /// [`Self::try_hold_processing`] the overflow run alone, with no
    /// processing-wait observation, since it never waits.
    fn grant_processing_at_once_locked(&mut self, st: &mut GateState, target: u64) -> Option<bool> {
        let gate = &self.gate;
        if self.in_overflow {
            gate.rebook_overflow_locked(st, target);
            self.reserved = target;
            return Some(false);
        }
        let others = st.reserved.saturating_sub(self.reserved);
        if others.saturating_add(target) <= gate.budget {
            gate.grow_locked(st, target - self.reserved);
            self.reserved = target;
            return Some(false);
        }
        if st.overflow.is_none() && st.processing.is_empty() {
            gate.release_locked(st, self.reserved, None);
            gate.occupy_overflow_locked(st, self.id, target);
            self.reserved = target;
            self.in_overflow = true;
            gate.serve_all(st);
            return Some(true);
        }
        None
    }

    /// The metrics of a processing step granted without waiting, after the
    /// lock is dropped: the overflow run when it entered the slot, and its
    /// ~0 processing wait.
    fn granted_at_once(
        &self,
        entered_overflow: bool,
        started: tokio::time::Instant,
    ) -> ProcessingGrant {
        if entered_overflow {
            self.gate.count_overflow_run(self.rpc);
        }
        self.gate
            .observe_processing_wait(self.rpc, started.elapsed());
        ProcessingGrant {
            entered_overflow,
            waited: Duration::ZERO,
        }
    }

    /// Shrink the reservation to `target` bytes and serve the queues. A target
    /// at or above the current reservation changes nothing.
    pub fn shrink_to(&mut self, target: u64) {
        if target >= self.reserved {
            return;
        }
        let gate = Arc::clone(&self.gate);
        let mut st = gate.lock();
        if self.in_overflow {
            gate.rebook_overflow_locked(&mut st, target);
        } else {
            gate.release_locked(&mut st, self.reserved - target, None);
        }
        self.reserved = target;
        gate.serve_all(&mut st);
    }

    /// The body is fully received: release the receive allowance `A` and the
    /// session's receiving slot, and serve the queues. Exactly once — a second
    /// call changes nothing.
    pub fn end_receive(&mut self) {
        if !self.receiving {
            return;
        }
        self.receiving = false;
        let gate = Arc::clone(&self.gate);
        let mut st = gate.lock();
        let allowance = RECEIVE_ALLOWANCE_BYTES.min(self.reserved);
        self.reserved -= allowance;
        if self.in_overflow {
            gate.rebook_overflow_locked(&mut st, self.reserved);
            gate.release_locked(&mut st, 0, Some(&self.session));
        } else {
            gate.release_locked(&mut st, allowance, Some(&self.session));
        }
        gate.serve_all(&mut st);
        drop(st);
        if let Some(m) = &gate.metrics {
            m.add_uploads_in_flight(UPLOAD_PHASE_RECEIVING, -1);
            m.add_uploads_in_flight(UPLOAD_PHASE_PROCESSING, 1);
        }
    }
}

impl Drop for UploadReservation {
    fn drop(&mut self) {
        let gate = Arc::clone(&self.gate);
        let mut st = gate.lock();
        let session = self.receiving.then_some(&self.session);
        if self.in_overflow {
            gate.vacate_overflow_locked(&mut st, self.id);
            gate.release_locked(&mut st, 0, session);
        } else {
            gate.release_locked(&mut st, self.reserved, session);
        }
        self.reserved = 0;
        gate.serve_all(&mut st);
        drop(st);
        if let Some(m) = &gate.metrics {
            let phase = if self.receiving {
                UPLOAD_PHASE_RECEIVING
            } else {
                UPLOAD_PHASE_PROCESSING
            };
            m.add_uploads_in_flight(phase, -1);
        }
    }
}

/// A queued processing step in flight. If the `reserve_processing` future is
/// dropped before its wait settles, this leaves the queue — or releases a
/// grant that raced the drop — and serves the next waiter.
struct PendingProcessing<'a> {
    reservation: &'a mut UploadReservation,
    signal: Arc<ProcessingSignal>,
    target: u64,
    started: tokio::time::Instant,
    settled: bool,
}

impl PendingProcessing<'_> {
    /// Book the grant the gate made on the reservation. Returns whether it was
    /// the overflow slot, and counts the overflow run when it was. The gate
    /// already moved the bytes under its lock.
    fn take_grant(&mut self) -> bool {
        let overflow = self.signal.state.load(Ordering::Acquire) == PROCESSING_OVERFLOW;
        self.reservation.reserved = self.target;
        self.reservation.in_overflow = overflow;
        if overflow {
            self.reservation
                .gate
                .count_overflow_run(self.reservation.rpc);
        }
        overflow
    }

    /// End the wait without proceeding: leave the queue, or release a grant
    /// that raced the ending. An in-budget grant gives back its growth; a slot
    /// grant keeps the slot at what the reservation held, until it drops (its
    /// in-budget bytes already left the budget). Counts the step `cancelled`
    /// and serves the queues.
    fn abandon(&mut self) {
        let gate = Arc::clone(&self.reservation.gate);
        let rpc = self.reservation.rpc;
        let mut st = gate.lock();
        match self.signal.state.load(Ordering::Acquire) {
            PROCESSING_IN_BUDGET => {
                gate.release_locked(&mut st, self.target - self.reservation.reserved, None);
            }
            PROCESSING_OVERFLOW => {
                gate.rebook_overflow_locked(&mut st, self.reservation.reserved);
                self.reservation.in_overflow = true;
            }
            _ => {
                let id = self.reservation.id;
                if let Some(at) = st.processing.iter().position(|w| w.reservation == id) {
                    st.processing.remove(at);
                    gate.add_processing_waiters(-1);
                }
            }
        }
        gate.serve_all(&mut st);
        drop(st);
        gate.count_refusal(rpc, ADMISSION_RESULT_CANCELLED);
        gate.observe_processing_wait(rpc, self.started.elapsed());
    }
}

impl Drop for PendingProcessing<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.abandon();
        }
    }
}

/// The INFO line "upload runs beyond the upload budget", logged once per
/// upload, by the caller of the processing step that returned
/// [`ProcessingGrant::entered_overflow`]: the RPC, the session and layer, the
/// step's charge (`charge_bytes`), `B` and how long the step waited.
pub(crate) fn log_upload_runs_beyond_budget(
    reservation: &UploadReservation,
    session_id: &str,
    layer_name: &str,
    charge_bytes: u64,
    waited: Duration,
) {
    tracing::info!(
        rpc = reservation.rpc().label(),
        session_id = %session_id,
        layer_name = %layer_name,
        charge_bytes,
        upload_budget_bytes = reservation.gate.budget_bytes(),
        waited_ms = u64::try_from(waited.as_millis()).unwrap_or(u64::MAX),
        "upload runs beyond the upload budget"
    );
}

/// The upload implementation of the storage seam [`CommitCharge`]: an
/// `UpdateLayer` call's reservation, after its processing steps, as its
/// commit sees it.
///
/// It holds the call's base processing charge (`base`, the second processing
/// step's charge without the handler's object-state hint) throughout, and the
/// commit's object-state term on top: [`CommitCharge::held`] is what the
/// reservation holds beyond `base` — at first the hint — and a target of `t`
/// holds `base + t`. [`CommitCharge::target`] prices a need with
/// [`object_state_charge`], and [`CommitCharge::rewrite_allowance`] is
/// [`object_state_rewrite_allowance`] of what is held. [`CommitCharge::try_hold`] is
/// [`UploadReservation::try_hold_processing`]; [`CommitCharge::hold`] shrinks
/// to a lower target, and otherwise is [`UploadReservation::reserve_processing`],
/// which may wait. A step that enters the overflow slot logs the line every
/// processing step does ([`log_upload_runs_beyond_budget`]), with the
/// whole charge as `charge_bytes`.
///
/// A [`CommitCharge::hold`] refused because the service is shutting down
/// answers the shared shutdown status the admission and processing waits
/// answer — `UNAVAILABLE "server shutting down"` with the
/// [`SERVER_SHUTTING_DOWN`](refusal_reason::SERVER_SHUTTING_DOWN) detail and
/// its retry hint ([`UploadRefusal::into_service_error`]) — and is recorded
/// ([`Self::shut_down`]), so the handler can mark its completion line
/// shutdown-caused. The call's tail keeps the detail only on its first
/// attempt: after a counted retry an earlier attempt may have committed, so
/// the answer is the plain outcome-unknown `UNAVAILABLE` (SDD §18.1,
/// `storage::layer::outcome_unknown_after_retry`).
///
/// `shutdown` is the service's graceful-shutdown handle. A waiting
/// [`CommitCharge::hold`] reads its [`GracefulShutdown::begun`] token on the
/// spot and hands it straight to [`UploadReservation::reserve_processing`] —
/// the read the shutdown-token guard pins. The charge keeps the handle, never
/// a token.
///
/// The `UpdateLayer` handler holds on it too, between its calls: before the
/// checkpoint-recovery re-issue's second pre-store of the RPC's external
/// values it grows the charge by one copy of their bytes through
/// [`CommitCharge::hold`] (a wait, holding nothing else; a shutdown refusal is
/// recorded here as any is), and after that store — on success and on failure
/// — trims it back to what it held through [`CommitCharge::try_hold`], which
/// never waits. So the extra copy is never part of [`CommitCharge::held`]
/// while a commit runs (`grpc::layer`'s `prestore_for_recovery_reissue`).
pub struct UploadCommitCharge<'a> {
    reservation: &'a mut UploadReservation,
    base: u64,
    shutdown: &'a GracefulShutdown,
    session_id: &'a str,
    layer_name: &'a str,
    shut_down: bool,
}

impl<'a> UploadCommitCharge<'a> {
    /// The commit charge of the call holding `reservation`, whose processing
    /// charge is `base`. `shutdown` is the service's graceful-shutdown
    /// handle: a waiting [`CommitCharge::hold`] races its `begun` token, read
    /// on the spot; `session_id` and `layer_name` are the call's, for the
    /// overflow line.
    pub fn new(
        reservation: &'a mut UploadReservation,
        base: u64,
        shutdown: &'a GracefulShutdown,
        session_id: &'a str,
        layer_name: &'a str,
    ) -> Self {
        Self {
            reservation,
            base,
            shutdown,
            session_id,
            layer_name,
            shut_down: false,
        }
    }

    /// Whether a [`CommitCharge::hold`] was refused because the service is
    /// shutting down.
    pub fn shut_down(&self) -> bool {
        self.shut_down
    }

    fn log_if_entered(&self, grant: ProcessingGrant, charge_bytes: u64) {
        if grant.entered_overflow {
            log_upload_runs_beyond_budget(
                self.reservation,
                self.session_id,
                self.layer_name,
                charge_bytes,
                grant.waited,
            );
        }
    }
}

#[async_trait]
impl CommitCharge for UploadCommitCharge<'_> {
    fn held(&self) -> u64 {
        self.reservation.reserved().saturating_sub(self.base)
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        object_state_charge(need)
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance(self.held(), need)
    }

    fn try_hold(&mut self, target: u64) -> bool {
        let charge = self.base.saturating_add(target);
        match self.reservation.try_hold_processing(charge) {
            Some(grant) => {
                self.log_if_entered(grant, charge);
                true
            }
            None => false,
        }
    }

    async fn hold(&mut self, target: u64) -> Result<(), ServiceError> {
        let charge = self.base.saturating_add(target);
        if charge < self.reservation.reserved() {
            self.reservation.shrink_to(charge);
            return Ok(());
        }
        match self
            .reservation
            .reserve_processing(charge, self.shutdown.begun())
            .await
        {
            Ok(grant) => {
                self.log_if_entered(grant, charge);
                Ok(())
            }
            Err(refusal) => {
                if matches!(refusal, UploadRefusal::ShuttingDown) {
                    self.shut_down = true;
                }
                Err(refusal.into_service_error())
            }
        }
    }
}

/// The sampled audit's charges on the upload budget `B`
/// (`storage::commit_charge::AuditChargeSource`): each audit opens one,
/// which holds its repair's memory in budget beside the uploads, priced as a
/// structural call's repair is ([`object_state_charge`]) less the copy's own
/// weight, which the line's audit term funds.
///
/// Its handle is no [`UploadReservation`]: it takes no admission and no
/// session slot, ticks no admission or in-flight metric, and moves only the
/// in-budget total and `live_edit_upload_reserved_bytes`. A growth never
/// waits and never takes the overflow slot, is refused while any upload is
/// queued, keeps one admission's room ([`admission_charge`]) free, and is
/// refused above [`SAMPLED_AUDIT_CHARGE_CAP_BYTES`]
/// (`UploadAdmissionGate::try_hold_audit`). A lower target gives the
/// difference back; dropping the handle gives back all of it and serves the
/// queues, on every exit of the audit (its compare, a skip, an error, the
/// shutdown's cancellation, a panic's unwind).
pub struct UploadBudgetAuditCharges {
    gate: Arc<UploadAdmissionGate>,
}

impl UploadBudgetAuditCharges {
    /// The audit charges on `gate`'s budget.
    pub fn new(gate: Arc<UploadAdmissionGate>) -> Self {
        Self { gate }
    }
}

impl AuditChargeSource for UploadBudgetAuditCharges {
    fn open(&self, base: u64) -> Box<dyn AuditCharge> {
        Box::new(UploadBudgetAuditCharge {
            gate: Arc::clone(&self.gate),
            base,
            cap: SAMPLED_AUDIT_CHARGE_CAP_BYTES,
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        })
    }
}

/// One audit's charge ([`UploadBudgetAuditCharges`]). `held` is the bytes it
/// holds in budget, changed only under the gate's lock.
struct UploadBudgetAuditCharge {
    gate: Arc<UploadAdmissionGate>,
    /// The copy's own weight, outside `B`: every target is the need's price
    /// less it.
    base: u64,
    cap: u64,
    held: AtomicU64,
    peak: AtomicU64,
}

impl SharedCommitCharge for UploadBudgetAuditCharge {
    fn held(&self) -> u64 {
        self.held.load(Ordering::Acquire)
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        object_state_charge(need).saturating_sub(self.base)
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance(self.held().saturating_add(self.base), need)
    }

    fn try_hold(&self, target: u64) -> bool {
        let held = self.gate.try_hold_audit(&self.held, target, self.cap);
        if held {
            self.peak.fetch_max(target, Ordering::AcqRel);
        }
        held
    }

    fn try_hold_at_least(&self, target: u64) -> bool {
        target <= self.held() || self.try_hold(target)
    }
}

impl AuditCharge for UploadBudgetAuditCharge {
    fn cap(&self) -> u64 {
        self.cap
    }

    fn peak(&self) -> u64 {
        self.peak.load(Ordering::Acquire)
    }
}

impl Drop for UploadBudgetAuditCharge {
    fn drop(&mut self) {
        self.gate.release_audit(self.held.swap(0, Ordering::AcqRel));
    }
}

/// The one startup line, at INFO: the budget, the per-session cap, the
/// admission settings, the size cap (`max_layer_size`), and the largest blob
/// that fits within the budget when sent as the one external value of the
/// canonical single-field blob upload of [`largest_in_budget_blob_upload`]
/// (`largest_in_budget_blob_upload_bytes`: the blob's size, not the upload's,
/// which adds its delta and framing). A larger delta may not fit. Larger
/// uploads, up to `LIVE_EDIT_MAX_LAYER_SIZE`, are still admitted and run one
/// at a time in the overflow slot beyond the budget — at the defaults a
/// 100 MiB blob does — so this is a statement of the configuration, not a
/// warning.
pub fn log_upload_admission_startup(gate: &UploadAdmissionGate) {
    let budget = gate.budget_bytes();
    tracing::info!(
        upload_budget_bytes = budget,
        per_session_receiving_cap = gate.per_session_cap(),
        admission_wait_ms = u64::try_from(gate.wait().as_millis()).unwrap_or(u64::MAX),
        admission_max_waiters = gate.max_waiters(),
        largest_in_budget_blob_upload_bytes = largest_in_budget_blob_upload(budget),
        max_layer_size = gate.max_upload_bytes(),
        "Upload admission budget configured: a blob of up to \
         largest_in_budget_blob_upload_bytes, sent as the one external value of a single \
         empty-named field (the upload adds its delta and framing on top), fits within the \
         budget, and a larger delta may not; larger uploads, up to LIVE_EDIT_MAX_LAYER_SIZE \
         (max_layer_size), are admitted and run one at a time in the overflow slot beyond the \
         budget"
    );
}

#[cfg(test)]
mod tests {
    //! The gate driven directly, one poll at a time.
    //!
    //! Queue order is decided on an admission's FIRST poll: [`admit`] joins the
    //! queue synchronously, before its first await, and a grant is made under
    //! the gate's lock by whichever call releases bytes. So each test pins its
    //! waiters, first-polls them in arrival order (each must park), releases,
    //! and then polls the parked ones newest-first: exactly the expected one
    //! must be ready. Nothing here sleeps; the one wait that has to elapse
    //! runs on paused time.
    //!
    //! [`admit`]: UploadAdmissionGate::admit

    use std::future::Future;
    use std::pin::Pin;
    use std::task::Poll;

    use super::*;
    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{
        metrics_scrape_only, Metrics, UPLOAD_ADMISSION_RESULTS, UPLOAD_PHASES, UPLOAD_RPCS,
    };

    /// `F + A`: what every admission reserves.
    const C: u64 = admission_charge();

    /// `4A`: room for exactly three admissions by bytes (`3C ≤ 4A < 4C`, the
    /// compile-time fixture self-check below) and a per-session receiving cap
    /// of one ([`small_budget_admits_one_upload_per_session`]).
    const SMALL_BUDGET: u64 = 4 * RECEIVE_ALLOWANCE_BYTES;

    /// An admission wait no test here lets elapse on its own.
    const LONG_WAIT: Duration = Duration::from_secs(3_600);

    fn gate(budget_bytes: u64, wait: Duration, max_waiters: usize) -> Arc<UploadAdmissionGate> {
        Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes,
                wait,
                max_waiters,
                max_upload_bytes: 0,
            },
            None,
        ))
    }

    fn gate_with_metrics(
        budget_bytes: u64,
        wait: Duration,
        max_waiters: usize,
    ) -> (Arc<UploadAdmissionGate>, Arc<Metrics>) {
        let metrics = Arc::new(metrics_scrape_only("upload-admission-test"));
        let gate = Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes,
                wait,
                max_waiters,
                max_upload_bytes: 0,
            },
            Some(Arc::clone(&metrics)),
        ));
        (gate, metrics)
    }

    /// Poll `fut` exactly once.
    async fn poll_once<F: Future>(fut: Pin<&mut F>) -> Poll<F::Output> {
        let mut fut = Some(fut);
        std::future::poll_fn(move |cx| {
            let f = fut.take().expect("polled once");
            Poll::Ready(f.poll(cx))
        })
        .await
    }

    /// An admission that must be granted on its first poll.
    async fn admit_now(
        gate: &Arc<UploadAdmissionGate>,
        session: &str,
        shutdown: &CancellationToken,
    ) -> UploadReservation {
        let mut fut = std::pin::pin!(gate.admit(UploadRpc::UpdateLayer, session, shutdown));
        match poll_once(fut.as_mut()).await {
            Poll::Ready(Ok(reservation)) => reservation,
            other => panic!("session {session:?} must be admitted at once; got {other:?}"),
        }
    }

    /// The value of one rendered sample, or 0 when the series is absent.
    fn metric(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_value(&render(metrics), name, labels).unwrap_or(0.0)
    }

    fn admission_count(metrics: &Metrics, result: &str) -> f64 {
        metric(
            metrics,
            "live_edit_upload_admission_total",
            &[("rpc", UPLOAD_RPC_UPDATE_LAYER), ("result", result)],
        )
    }

    // ── Fixture arithmetic ──────────────────────────────────────────────────

    /// Fixture self-check, at compile time: `SMALL_BUDGET` holds exactly three
    /// admissions.
    const _: () = assert!(3 * C <= SMALL_BUDGET && 4 * C > SMALL_BUDGET);

    #[test]
    fn small_budget_admits_one_upload_per_session() {
        assert_eq!(per_session_cap_for(SMALL_BUDGET), 1);
    }

    // ── Queue order ─────────────────────────────────────────────────────────

    /// With the budget full, waiters from distinct sessions are granted in
    /// arrival order, one per release.
    #[tokio::test]
    async fn admit_serves_byte_bound_waiters_first_come_first_served() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut held = Vec::new();
        for session in ["h0", "h1", "h2"] {
            held.push(admit_now(&g, session, &token).await);
        }
        assert_eq!(g.reserved_bytes(), 3 * C);

        let sessions = ["w0", "w1", "w2"];
        let mut waiting: Vec<Option<Pin<Box<_>>>> = sessions
            .iter()
            .map(|s| Some(Box::pin(g.admit(UploadRpc::UpdateLayer, s, &token))))
            .collect();
        for (index, waiter) in waiting.iter_mut().enumerate() {
            let waiter = waiter.as_mut().expect("nothing served yet");
            assert!(
                poll_once(waiter.as_mut()).await.is_pending(),
                "fixture self-check: with the budget full, waiter {index} must park"
            );
            assert_eq!(g.waiters(), index + 1);
        }

        let mut served = Vec::new();
        for release in 0..sessions.len() {
            drop(held.pop().expect("one holder per release"));
            let mut ready = Vec::new();
            for index in (0..sessions.len()).rev() {
                let Some(waiter) = waiting[index].as_mut() else {
                    continue;
                };
                if let Poll::Ready(granted) = poll_once(waiter.as_mut()).await {
                    served.push(granted.expect("a granted waiter is admitted"));
                    waiting[index] = None;
                    ready.push(index);
                }
            }
            assert_eq!(
                ready,
                vec![release],
                "release {release} must grant exactly the oldest waiter, {release}"
            );
        }
        assert_eq!(g.waiters(), 0);
        assert_eq!(g.reserved_bytes(), 3 * C);
    }

    /// A waiter held back only by its own session's receiving cap is skipped,
    /// so a later waiter of another session takes the freed bytes; the
    /// skipped one is granted as soon as its session's slot frees.
    #[tokio::test]
    async fn admit_skips_a_waiter_held_back_by_its_session_cap() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut s = admit_now(&g, "s", &token).await;
        let t = admit_now(&g, "t", &token).await;
        let _u = admit_now(&g, "u", &token).await;

        let mut w_s = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "s", &token));
        let mut w_v = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "v", &token));
        assert!(poll_once(w_s.as_mut()).await.is_pending());
        assert!(poll_once(w_v.as_mut()).await.is_pending());
        assert_eq!(g.waiters(), 2);

        drop(t);
        assert!(
            poll_once(w_s.as_mut()).await.is_pending(),
            "session s is at its cap of {} receiving upload(s): its waiter must stay queued",
            g.per_session_cap()
        );
        let _v = match poll_once(w_v.as_mut()).await {
            Poll::Ready(Ok(r)) => r,
            other => {
                panic!("the freed bytes must go to the waiter behind the capped one; got {other:?}")
            }
        };

        s.end_receive();
        assert!(
            matches!(poll_once(w_s.as_mut()).await, Poll::Ready(Ok(_))),
            "ending s's receive phase frees its slot and its allowance: the skipped waiter \
             must be granted"
        );
        assert_eq!(g.waiters(), 0);
    }

    /// A new arrival whose session has room still queues behind a waiter the
    /// gate could serve once bytes free: it does not take the next release.
    #[tokio::test]
    async fn admit_newcomer_queues_behind_a_byte_bound_waiter() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut held = Vec::new();
        for session in ["h0", "h1", "h2"] {
            held.push(admit_now(&g, session, &token).await);
        }
        let mut first = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "first", &token));
        assert!(poll_once(first.as_mut()).await.is_pending());
        let mut newcomer = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "newcomer", &token));
        assert!(poll_once(newcomer.as_mut()).await.is_pending());

        drop(held.pop());
        assert!(
            poll_once(newcomer.as_mut()).await.is_pending(),
            "the newcomer must not overtake the byte-bound waiter queued before it"
        );
        assert!(matches!(
            poll_once(first.as_mut()).await,
            Poll::Ready(Ok(_))
        ));
    }

    // ── Growth ──────────────────────────────────────────────────────────────

    /// Growth never waits and ignores the queue: with a waiter queued it is
    /// granted when the bytes fit, and refused `Growth` when they do not fit
    /// now. A refusal changes nothing.
    #[tokio::test]
    async fn try_grow_in_budget_answers_at_once_even_with_a_waiter_queued() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = admit_now(&g, "a", &token).await;
        let _b = admit_now(&g, "b", &token).await;
        let _c = admit_now(&g, "c", &token).await;
        let mut waiter = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "d", &token));
        assert!(poll_once(waiter.as_mut()).await.is_pending());

        let free = SMALL_BUDGET - 3 * C;
        assert!(
            free > 0 && free < C,
            "fixture self-check: {free} bytes free"
        );
        let grown = r.reserved() + free;
        assert_eq!(r.try_grow_in_budget(grown), Ok(()));
        assert_eq!(r.reserved(), grown);
        assert_eq!(g.reserved_bytes(), SMALL_BUDGET);
        assert!(poll_once(waiter.as_mut()).await.is_pending());

        assert_eq!(r.try_grow_in_budget(grown + 1), Err(UploadRefusal::Growth));
        assert_eq!(r.reserved(), grown, "a refused growth changes nothing");
        assert_eq!(g.reserved_bytes(), SMALL_BUDGET);
        assert_eq!(
            r.try_grow_in_budget(1),
            Ok(()),
            "a target below the reservation is a no-op"
        );
        assert_eq!(r.reserved(), grown);
        assert_eq!(g.high_water_bytes(), SMALL_BUDGET);
    }

    /// `end_receive` and `shrink_to` each serve the queue when they free
    /// enough: here the freed allowance alone is `F` short, and the shrink
    /// that follows frees the rest.
    #[tokio::test]
    async fn shrink_to_serves_a_queued_waiter() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = admit_now(&g, "a", &token).await;
        let _b = admit_now(&g, "b", &token).await;
        let _c = admit_now(&g, "c", &token).await;
        r.try_grow_in_budget(r.reserved() + (SMALL_BUDGET - 3 * C))
            .expect("fill the budget exactly");
        let mut waiter = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "d", &token));
        assert!(poll_once(waiter.as_mut()).await.is_pending());

        r.end_receive();
        assert_eq!(g.reserved_bytes(), SMALL_BUDGET - RECEIVE_ALLOWANCE_BYTES);
        assert!(
            poll_once(waiter.as_mut()).await.is_pending(),
            "fixture self-check: the freed allowance is {FIXED_CHARGE_BYTES} bytes short of an \
             admission"
        );

        r.shrink_to(r.reserved() - FIXED_CHARGE_BYTES);
        let Poll::Ready(Ok(_granted)) = poll_once(waiter.as_mut()).await else {
            panic!("the shrink must serve the waiter");
        };
        assert_eq!(g.reserved_bytes(), SMALL_BUDGET);
    }

    // ── Bounded waiting ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn admit_refuses_waiter_cap_at_once_when_the_queue_is_full() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 1);
        let _held: Vec<_> =
            futures::future::join_all(["a", "b", "c"].map(|s| admit_now(&g, s, &token))).await;
        let mut queued = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "q", &token));
        assert!(poll_once(queued.as_mut()).await.is_pending());

        let mut over = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "over", &token));
        assert!(
            matches!(
                poll_once(over.as_mut()).await,
                Poll::Ready(Err(UploadRefusal::WaiterCap))
            ),
            "a full queue refuses on the first poll"
        );
        assert_eq!(g.waiters(), 1);
    }

    /// The wait ends refused at the admission wait, and not a tick before.
    #[tokio::test(start_paused = true)]
    async fn admit_wait_is_refused_saturated_at_the_deadline_not_before() {
        const WAIT: Duration = Duration::from_millis(100);
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, WAIT, 8);
        let _held: Vec<_> =
            futures::future::join_all(["a", "b", "c"].map(|s| admit_now(&g, s, &token))).await;
        let mut waiter = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "w", &token));
        assert!(poll_once(waiter.as_mut()).await.is_pending());

        tokio::time::advance(WAIT - Duration::from_millis(1)).await;
        assert!(
            poll_once(waiter.as_mut()).await.is_pending(),
            "refused before the admission wait elapsed"
        );
        tokio::time::advance(Duration::from_millis(1)).await;
        assert!(matches!(
            poll_once(waiter.as_mut()).await,
            Poll::Ready(Err(UploadRefusal::Saturated))
        ));
        assert_eq!(g.waiters(), 0, "a refused wait leaves the queue");
        assert_eq!(g.reserved_bytes(), 3 * C, "and reserves nothing");
    }

    #[tokio::test]
    async fn admit_with_a_zero_wait_never_queues() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, Duration::ZERO, 8);
        let held: Vec<_> =
            futures::future::join_all(["a", "b", "c"].map(|s| admit_now(&g, s, &token))).await;
        let mut refused = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "w", &token));
        assert!(matches!(
            poll_once(refused.as_mut()).await,
            Poll::Ready(Err(UploadRefusal::Saturated))
        ));
        assert_eq!(g.waiters(), 0);
        drop(held);
        let _admitted = admit_now(&g, "w", &token).await;
    }

    // ── Shutdown ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn admit_refuses_queued_and_new_admissions_once_shutdown_fires() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut held: Vec<_> =
            futures::future::join_all(["a", "b", "c"].map(|s| admit_now(&g, s, &token))).await;
        let mut queued = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "q", &token));
        assert!(poll_once(queued.as_mut()).await.is_pending());

        token.cancel();
        assert!(matches!(
            poll_once(queued.as_mut()).await,
            Poll::Ready(Err(UploadRefusal::ShuttingDown))
        ));
        assert_eq!(g.waiters(), 0);

        held.pop();
        let mut fresh = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "fresh", &token));
        assert!(
            matches!(
                poll_once(fresh.as_mut()).await,
                Poll::Ready(Err(UploadRefusal::ShuttingDown))
            ),
            "a new admission is refused after shutdown even with room in the budget"
        );
        assert_eq!(g.reserved_bytes(), 2 * C, "no refusal took bytes");
    }

    // ── Dropped waits ───────────────────────────────────────────────────────

    /// A wait dropped before its grant leaves the queue; the next release
    /// serves the waiter behind it.
    #[tokio::test]
    async fn dropped_wait_leaves_the_queue_and_the_next_waiter_is_served() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut held: Vec<_> =
            futures::future::join_all(["a", "b", "c"].map(|s| admit_now(&g, s, &token))).await;
        let mut first = Box::pin(g.admit(UploadRpc::UpdateLayer, "first", &token));
        let mut second = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "second", &token));
        assert!(poll_once(first.as_mut()).await.is_pending());
        assert!(poll_once(second.as_mut()).await.is_pending());

        drop(first);
        assert_eq!(g.waiters(), 1);
        held.pop();
        assert!(matches!(
            poll_once(second.as_mut()).await,
            Poll::Ready(Ok(_))
        ));
    }

    /// A wait dropped after the gate granted it, but before it was polled
    /// again, gives the grant back — and that serves the next waiter.
    #[tokio::test]
    async fn dropped_wait_whose_grant_raced_the_drop_releases_it_to_the_next_waiter() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut held: Vec<_> =
            futures::future::join_all(["a", "b", "c"].map(|s| admit_now(&g, s, &token))).await;
        let mut first = Box::pin(g.admit(UploadRpc::UpdateLayer, "first", &token));
        let mut second = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "second", &token));
        assert!(poll_once(first.as_mut()).await.is_pending());
        assert!(poll_once(second.as_mut()).await.is_pending());

        held.pop();
        assert_eq!(
            g.reserved_bytes(),
            3 * C,
            "precondition: the release granted the first waiter"
        );
        assert_eq!(g.receiving_uploads("first"), 1);
        drop(first);
        assert_eq!(
            g.receiving_uploads("first"),
            0,
            "the raced grant was released"
        );
        let Poll::Ready(Ok(_second)) = poll_once(second.as_mut()).await else {
            panic!("the released grant must serve the next waiter");
        };
        assert_eq!(g.reserved_bytes(), 3 * C);
        assert_eq!(g.waiters(), 0);
    }

    // ── Release ─────────────────────────────────────────────────────────────

    /// `end_receive` releases the allowance and the session slot exactly once,
    /// and `Drop` releases the rest; every gauge returns to zero.
    #[tokio::test]
    async fn reservation_end_receive_releases_once_and_drop_releases_everything() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let reserved = |m: &Metrics| metric(m, "live_edit_upload_reserved_bytes", &[]);
        let in_flight = |m: &Metrics, phase: &str| {
            metric(m, "live_edit_uploads_in_flight", &[("phase", phase)])
        };

        let mut r = admit_now(&g, "s", &token).await;
        assert_eq!(g.reserved_bytes(), C);
        assert_eq!(g.receiving_uploads("s"), 1);
        assert_eq!(reserved(&metrics), C as f64);
        assert_eq!(in_flight(&metrics, UPLOAD_PHASE_RECEIVING), 1.0);

        const GROWTH: u64 = 1_000;
        r.try_grow_in_budget(C + GROWTH)
            .expect("room for the growth");
        r.end_receive();
        let after = C + GROWTH - RECEIVE_ALLOWANCE_BYTES;
        assert!(!r.is_receiving());
        assert_eq!(r.reserved(), after);
        assert_eq!(g.reserved_bytes(), after);
        assert_eq!(g.receiving_uploads("s"), 0);
        assert_eq!(in_flight(&metrics, UPLOAD_PHASE_RECEIVING), 0.0);
        assert_eq!(in_flight(&metrics, UPLOAD_PHASE_PROCESSING), 1.0);

        r.end_receive();
        assert_eq!(r.reserved(), after, "a second end_receive changes nothing");
        assert_eq!(g.reserved_bytes(), after);
        assert_eq!(in_flight(&metrics, UPLOAD_PHASE_PROCESSING), 1.0);

        drop(r);
        assert_eq!(g.reserved_bytes(), 0);
        assert_eq!(reserved(&metrics), 0.0);
        assert_eq!(in_flight(&metrics, UPLOAD_PHASE_PROCESSING), 0.0);
        assert_eq!(in_flight(&metrics, UPLOAD_PHASE_RECEIVING), 0.0);
        assert_eq!(g.high_water_bytes(), C + GROWTH);
    }

    /// Aborting the task that holds a reservation — still receiving — releases
    /// its bytes and its session slot; aborting a task parked in the admission
    /// queue removes it from the queue and counts it `cancelled`.
    #[tokio::test]
    async fn task_abort_releases_a_reservation_and_a_queued_wait() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let (admitted_tx, admitted_rx) = tokio::sync::oneshot::channel();
        let holder = tokio::spawn({
            let g = Arc::clone(&g);
            let token = token.clone();
            async move {
                let _r = g
                    .admit(UploadRpc::UpdateLayer, "held", &token)
                    .await
                    .expect("admitted");
                admitted_tx.send(()).expect("the test is listening");
                std::future::pending::<()>().await;
            }
        });
        admitted_rx.await.expect("the holder was admitted");
        let _b = admit_now(&g, "b", &token).await;
        let _c = admit_now(&g, "c", &token).await;
        assert_eq!(g.reserved_bytes(), 3 * C);

        let waiter = tokio::spawn({
            let g = Arc::clone(&g);
            let token = token.clone();
            async move { g.admit(UploadRpc::UpdateLayer, "w", &token).await }
        });
        // The spawned waiter joins the queue on its first poll; yield until it
        // has. The current-thread runtime runs it at the first yield.
        while g.waiters() == 0 {
            tokio::task::yield_now().await;
        }

        waiter.abort();
        assert!(waiter.await.expect_err("aborted").is_cancelled());
        assert_eq!(g.waiters(), 0);
        assert_eq!(admission_count(&metrics, ADMISSION_RESULT_CANCELLED), 1.0);

        holder.abort();
        assert!(holder.await.expect_err("aborted").is_cancelled());
        assert_eq!(g.reserved_bytes(), 2 * C);
        assert_eq!(g.receiving_uploads("held"), 0);
    }

    // ── Metrics ─────────────────────────────────────────────────────────────

    /// Every `(rpc, result)` cell of the admission counter (ten: five results,
    /// none a refusal for size), every `rpc` cell of the overflow-run counter,
    /// both phases of the in-flight gauge, and the five unlabelled gauges exist
    /// from the first scrape, at zero (the budget gauge at the configured
    /// budget once a gate is built), each under its own type. The
    /// processing-wait histogram is not pre-registered: it appears with its
    /// first observation (see the counting test below).
    #[test]
    fn upload_admission_metrics_are_seeded_at_zero() {
        use crate::observability::metrics::test_util::{has_metric, metric_type};

        let metrics = metrics_scrape_only("upload-admission-seed");
        let text = render(&metrics);
        assert_eq!(UPLOAD_ADMISSION_RESULTS.len() * UPLOAD_RPCS.len(), 10);
        assert!(
            !text.contains("refused_too_large"),
            "no result is a refusal for size any more"
        );
        for rpc in UPLOAD_RPCS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_upload_overflow_runs_total",
                    &[("rpc", rpc)]
                ),
                Some(0.0),
                "overflow-run cell rpc={rpc} must be seeded"
            );
        }
        for (name, kind) in [
            ("live_edit_upload_admission_total", "counter"),
            ("live_edit_upload_overflow_runs_total", "counter"),
            ("live_edit_upload_reserved_bytes", "gauge"),
            ("live_edit_upload_overflow_bytes", "gauge"),
            ("live_edit_upload_overflow_waiters", "gauge"),
            ("live_edit_upload_admission_waiters", "gauge"),
            ("live_edit_upload_budget_bytes", "gauge"),
            ("live_edit_uploads_in_flight", "gauge"),
        ] {
            assert_eq!(metric_type(&text, name).as_deref(), Some(kind), "{name}");
        }
        assert!(
            !has_metric(&text, "live_edit_upload_processing_wait_seconds"),
            "the processing-wait histogram is not pre-registered"
        );
        for rpc in UPLOAD_RPCS {
            for result in UPLOAD_ADMISSION_RESULTS {
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_upload_admission_total",
                        &[("rpc", rpc), ("result", result)]
                    ),
                    Some(0.0),
                    "cell rpc={rpc} result={result} must be seeded"
                );
            }
        }
        for phase in UPLOAD_PHASES {
            assert_eq!(
                sample_value(&text, "live_edit_uploads_in_flight", &[("phase", phase)]),
                Some(0.0),
                "phase {phase} must be seeded"
            );
        }
        for name in [
            "live_edit_upload_reserved_bytes",
            "live_edit_upload_overflow_bytes",
            "live_edit_upload_overflow_waiters",
            "live_edit_upload_admission_waiters",
            "live_edit_upload_budget_bytes",
        ] {
            assert_eq!(sample_value(&text, name, &[]), Some(0.0), "{name}");
        }
    }

    /// Each decision moves its own `result` cell by one, the waiters gauge
    /// follows the queue, and the budget gauge reports the budget.
    #[tokio::test]
    async fn upload_admission_metrics_count_each_decision_in_its_cell() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 1);
        assert_eq!(
            metric(&metrics, "live_edit_upload_budget_bytes", &[]),
            SMALL_BUDGET as f64
        );

        let mut r = admit_now(&g, "a", &token).await;
        let _b = admit_now(&g, "b", &token).await;
        let _c = admit_now(&g, "c", &token).await;
        assert_eq!(admission_count(&metrics, ADMISSION_RESULT_ADMITTED), 3.0);

        let mut queued = Box::pin(g.admit(UploadRpc::UpdateLayer, "q", &token));
        assert!(poll_once(queued.as_mut()).await.is_pending());
        assert_eq!(
            metric(&metrics, "live_edit_upload_admission_waiters", &[]),
            1.0
        );

        let mut over = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "over", &token));
        assert!(poll_once(over.as_mut()).await.is_ready());
        assert_eq!(
            admission_count(&metrics, ADMISSION_RESULT_REFUSED_WAITER_CAP),
            1.0
        );

        assert!(r.try_grow_in_budget(SMALL_BUDGET).is_err());
        assert_eq!(
            admission_count(&metrics, ADMISSION_RESULT_REFUSED_GROWTH),
            1.0
        );

        drop(queued);
        assert_eq!(admission_count(&metrics, ADMISSION_RESULT_CANCELLED), 1.0);
        assert_eq!(
            metric(&metrics, "live_edit_upload_admission_waiters", &[]),
            0.0
        );

        let zero_wait = gate_with_metrics(SMALL_BUDGET, Duration::ZERO, 1);
        let _held: Vec<_> =
            futures::future::join_all(["a", "b", "c"].map(|s| admit_now(&zero_wait.0, s, &token)))
                .await;
        let mut refused = std::pin::pin!(zero_wait.0.admit(UploadRpc::UpdateLayer, "w", &token));
        assert!(poll_once(refused.as_mut()).await.is_ready());
        assert_eq!(
            admission_count(&zero_wait.1, ADMISSION_RESULT_REFUSED_SATURATED),
            1.0
        );
        assert_eq!(
            metric(
                &metrics,
                "live_edit_upload_admission_total",
                &[
                    ("rpc", UPLOAD_RPC_CREATE_LAYER),
                    ("result", ADMISSION_RESULT_ADMITTED)
                ]
            ),
            0.0,
            "an UpdateLayer decision must not tick the CreateLayer cell"
        );
    }

    /// A growth refused after `end_receive` — the `CreateLayer` re-send's
    /// best-effort charges — counts no `refused_growth`: no client is refused
    /// for it. The same refusal while the upload is still receiving its
    /// content counts one.
    #[tokio::test]
    async fn try_grow_in_budget_refused_after_end_receive_counts_no_refused_growth() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut done = admit_now(&g, "done", &token).await;
        done.end_receive();
        let mut receiving = admit_now(&g, "receiving", &token).await;
        let _other = admit_now(&g, "other", &token).await;

        assert_eq!(
            done.try_grow_in_budget(SMALL_BUDGET),
            Err(UploadRefusal::Growth)
        );
        assert_eq!(
            admission_count(&metrics, ADMISSION_RESULT_REFUSED_GROWTH),
            0.0,
            "a refusal after the content is received refuses no client"
        );

        assert_eq!(
            receiving.try_grow_in_budget(SMALL_BUDGET),
            Err(UploadRefusal::Growth)
        );
        assert_eq!(
            admission_count(&metrics, ADMISSION_RESULT_REFUSED_GROWTH),
            1.0,
            "the receive-phase refusal is counted"
        );
    }

    // ── Processing: the overflow slot (T14) ────────────────────────────────

    /// A processing target no room in `SMALL_BUDGET` can hold: it runs in the
    /// overflow slot or waits for it.
    const OVER: u64 = SMALL_BUDGET + 1;

    /// An admitted upload whose content is received: it holds `F` in budget.
    async fn received(
        gate: &Arc<UploadAdmissionGate>,
        session: &str,
        shutdown: &CancellationToken,
    ) -> UploadReservation {
        let mut r = admit_now(gate, session, shutdown).await;
        r.end_receive();
        r
    }

    /// A processing step that must be granted on its first poll.
    async fn process_now(
        r: &mut UploadReservation,
        target: u64,
        shutdown: &CancellationToken,
    ) -> ProcessingGrant {
        let mut fut = std::pin::pin!(r.reserve_processing(target, shutdown));
        match poll_once(fut.as_mut()).await {
            Poll::Ready(Ok(grant)) => grant,
            other => panic!("the processing step must be granted at once; got {other:?}"),
        }
    }

    fn overflow_runs(metrics: &Metrics) -> f64 {
        metric(
            metrics,
            "live_edit_upload_overflow_runs_total",
            &[("rpc", UPLOAD_RPC_UPDATE_LAYER)],
        )
    }

    /// A target that fits the budget is granted there at once, even with a
    /// processing step queued for the slot: only a target that cannot fit
    /// waits.
    #[tokio::test]
    async fn reserve_processing_fitting_target_is_granted_in_budget_with_slot_waiters_queued() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        assert!(
            process_now(&mut holder, OVER, &token)
                .await
                .entered_overflow
        );
        assert_eq!(g.overflow_bytes(), OVER);
        assert_eq!(g.reserved_bytes(), 0, "the holder's bytes left the budget");

        let mut waiter = received(&g, "waiter", &token).await;
        let mut waiting = Box::pin(waiter.reserve_processing(OVER, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        assert_eq!(g.processing_waiters(), 1);

        let mut small = received(&g, "small", &token).await;
        let grant = process_now(&mut small, C, &token).await;
        assert!(!grant.entered_overflow, "a fitting target stays in budget");
        assert_eq!(small.reserved(), C);
        assert!(!small.in_overflow());
        assert_eq!(g.reserved_bytes(), FIXED_CHARGE_BYTES + C);
        assert!(poll_once(waiting.as_mut()).await.is_pending());
    }

    /// The slot goes to processing waiters in arrival order, one holder at a
    /// time, and a holder's `Drop` hands it to the next.
    #[tokio::test]
    async fn reserve_processing_slot_is_fifo_and_the_holders_drop_grants_the_next() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;

        let mut first = received(&g, "first", &token).await;
        let mut second = received(&g, "second", &token).await;
        let mut first_wait = Box::pin(first.reserve_processing(OVER + 1, &token));
        let mut second_wait = Box::pin(second.reserve_processing(OVER + 2, &token));
        assert!(poll_once(first_wait.as_mut()).await.is_pending());
        assert!(poll_once(second_wait.as_mut()).await.is_pending());
        assert_eq!(g.processing_waiters(), 2);
        assert_eq!(
            metric(&metrics, "live_edit_upload_overflow_waiters", &[]),
            2.0
        );

        drop(holder);
        assert!(
            poll_once(second_wait.as_mut()).await.is_pending(),
            "the later waiter must not overtake the earlier one"
        );
        let Poll::Ready(Ok(grant)) = poll_once(first_wait.as_mut()).await else {
            panic!("the holder's drop must hand the slot to the oldest waiter");
        };
        assert!(grant.entered_overflow);
        assert_eq!(g.overflow_bytes(), OVER + 1);
        assert_eq!(
            metric(&metrics, "live_edit_upload_overflow_bytes", &[]),
            (OVER + 1) as f64
        );
        assert_eq!(g.processing_waiters(), 1);

        drop(first_wait);
        drop(first);
        let Poll::Ready(Ok(grant)) = poll_once(second_wait.as_mut()).await else {
            panic!("the next holder's drop must hand the slot on");
        };
        assert!(grant.entered_overflow);
        assert_eq!(g.overflow_bytes(), OVER + 2);
        assert_eq!(overflow_runs(&metrics), 3.0);
        drop(second_wait);
        drop(second);
        assert_eq!(g.overflow_bytes(), 0);
        assert_eq!(g.reserved_bytes(), 0, "nothing is left reserved");
        assert_eq!(g.overflow_high_water_bytes(), OVER + 2);
    }

    /// The holder books further growth on its slot in place and never waits —
    /// not even with a waiter queued — through either growth call.
    #[tokio::test]
    async fn reserve_processing_holder_grows_in_place_without_waiting() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;
        let mut waiter = received(&g, "waiter", &token).await;
        let mut waiting = Box::pin(waiter.reserve_processing(OVER, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());

        let grant = process_now(&mut holder, 2 * OVER, &token).await;
        assert!(!grant.entered_overflow, "it was already in the slot");
        assert_eq!(grant.waited, Duration::ZERO);
        assert_eq!(g.overflow_bytes(), 2 * OVER);
        assert_eq!(holder.try_grow_in_budget(3 * OVER), Ok(()));
        assert_eq!(g.overflow_bytes(), 3 * OVER);
        assert_eq!(holder.reserved(), 3 * OVER);
        assert_eq!(
            g.reserved_bytes(),
            FIXED_CHARGE_BYTES,
            "the holder's growth never touches the budget"
        );
        assert!(poll_once(waiting.as_mut()).await.is_pending());
    }

    /// A slot waiter is granted in budget when room frees there before the
    /// slot does; the holder keeps the slot.
    #[tokio::test]
    async fn reserve_processing_slot_waiter_is_granted_in_budget_when_room_frees_first() {
        const TARGET: u64 = 20 * MIB;
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;
        let first = admit_now(&g, "first", &token).await;
        let _second = admit_now(&g, "second", &token).await;
        let mut waiter = received(&g, "waiter", &token).await;
        // Fixture self-check: TARGET does not fit beside both admissions, and
        // fits beside one.
        const _: () = assert!(2 * C + TARGET > SMALL_BUDGET && C + TARGET <= SMALL_BUDGET);

        let mut waiting = Box::pin(waiter.reserve_processing(TARGET, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        drop(first);
        let Poll::Ready(Ok(grant)) = poll_once(waiting.as_mut()).await else {
            panic!("freed room must grant the waiter in budget");
        };
        assert!(!grant.entered_overflow);
        drop(waiting);
        assert!(!waiter.in_overflow());
        assert_eq!(waiter.reserved(), TARGET);
        assert_eq!(g.reserved_bytes(), C + TARGET);
        assert_eq!(g.overflow_bytes(), OVER, "the holder keeps the slot");
        assert_eq!(g.processing_waiters(), 0);
    }

    /// Shutdown refuses a processing step waiting for the slot (counted
    /// `cancelled`), but not the holder, which grows in place and runs to its
    /// end.
    #[tokio::test]
    async fn reserve_processing_shutdown_refuses_slot_waiters_but_not_the_holder() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;
        let mut waiter = received(&g, "waiter", &token).await;
        let mut waiting = Box::pin(waiter.reserve_processing(OVER, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());

        token.cancel();
        assert!(matches!(
            poll_once(waiting.as_mut()).await,
            Poll::Ready(Err(UploadRefusal::ShuttingDown))
        ));
        drop(waiting);
        assert_eq!(g.processing_waiters(), 0);
        assert!(!waiter.in_overflow());
        assert_eq!(waiter.reserved(), FIXED_CHARGE_BYTES, "nothing was granted");
        assert_eq!(admission_count(&metrics, ADMISSION_RESULT_CANCELLED), 1.0);

        let grant = process_now(&mut holder, 2 * OVER, &token).await;
        assert!(!grant.entered_overflow);
        assert_eq!(g.overflow_bytes(), 2 * OVER);
        assert_eq!(
            overflow_runs(&metrics),
            1.0,
            "only the holder ran in the slot"
        );
    }

    /// A slot grant that loses to shutdown — granted under the lock, but the
    /// wait sees shutdown first — is not a run: it counts no overflow run and
    /// returns no grant (so the handler logs no "runs beyond the budget"
    /// line). The waiter keeps the slot at what it held until it drops, and
    /// then the slot moves on.
    #[tokio::test]
    async fn reserve_processing_slot_grant_lost_to_shutdown_counts_no_overflow_run() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;
        let mut waiter = received(&g, "waiter", &token).await;
        let mut waiting = Box::pin(waiter.reserve_processing(OVER, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());

        drop(holder);
        assert_eq!(
            g.overflow_bytes(),
            OVER,
            "precondition: the holder's drop granted the waiter the slot"
        );
        token.cancel();
        assert!(matches!(
            poll_once(waiting.as_mut()).await,
            Poll::Ready(Err(UploadRefusal::ShuttingDown))
        ));
        drop(waiting);
        assert_eq!(overflow_runs(&metrics), 1.0, "only the first holder's run");
        assert_eq!(admission_count(&metrics, ADMISSION_RESULT_CANCELLED), 1.0);
        assert!(waiter.in_overflow());
        assert_eq!(
            g.overflow_bytes(),
            FIXED_CHARGE_BYTES,
            "kept at what it held"
        );
        drop(waiter);
        assert_eq!(g.overflow_bytes(), 0);
        assert_eq!(g.reserved_bytes(), 0);
    }

    /// A slot grant that loses to a dropped call is not a run either; and a
    /// wait dropped before any grant leaves the queue, so the next waiter is
    /// served.
    #[tokio::test]
    async fn reserve_processing_dropped_waits_leave_the_queue_and_count_no_overflow_run() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;
        let mut gone = received(&g, "gone", &token).await;
        let mut granted_then_dropped = received(&g, "granted", &token).await;
        let mut last = received(&g, "last", &token).await;
        let mut gone_wait = Box::pin(gone.reserve_processing(OVER, &token));
        let mut granted_wait = Box::pin(granted_then_dropped.reserve_processing(OVER, &token));
        let mut last_wait = Box::pin(last.reserve_processing(OVER, &token));
        assert!(poll_once(gone_wait.as_mut()).await.is_pending());
        assert!(poll_once(granted_wait.as_mut()).await.is_pending());
        assert!(poll_once(last_wait.as_mut()).await.is_pending());

        drop(gone_wait);
        assert_eq!(g.processing_waiters(), 2, "the dropped wait left the queue");
        drop(holder);
        assert_eq!(
            g.overflow_bytes(),
            OVER,
            "precondition: the slot went to the next waiter"
        );
        drop(granted_wait);
        assert_eq!(overflow_runs(&metrics), 1.0, "a dropped grant is no run");
        assert!(granted_then_dropped.in_overflow());
        assert!(poll_once(last_wait.as_mut()).await.is_pending());
        drop(granted_then_dropped);
        let Poll::Ready(Ok(grant)) = poll_once(last_wait.as_mut()).await else {
            panic!("the slot must move on to the last waiter");
        };
        assert!(grant.entered_overflow);
        assert_eq!(overflow_runs(&metrics), 2.0);
        assert_eq!(admission_count(&metrics, ADMISSION_RESULT_CANCELLED), 2.0);
        drop(gone);
    }

    /// The round-1 deadlock shape: the budget full of the residues of uploads
    /// that each need more than the whole budget. Each queues for the slot,
    /// and they run one at a time in arrival order — none waits forever on
    /// room only another's completion could free.
    #[tokio::test]
    async fn reserve_processing_budget_full_of_slot_waiters_residues_drains_in_fifo_order() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut a = received(&g, "a", &token).await;
        let mut b = received(&g, "b", &token).await;
        let mut c = received(&g, "c", &token).await;
        let third = SMALL_BUDGET / 3;
        for r in [&mut a, &mut b] {
            r.try_grow_in_budget(third).expect("room for a third");
        }
        c.try_grow_in_budget(SMALL_BUDGET - 2 * third)
            .expect("the rest");
        assert_eq!(
            g.reserved_bytes(),
            SMALL_BUDGET,
            "the budget is full of residues"
        );

        assert!(process_now(&mut a, OVER, &token).await.entered_overflow);
        let mut b_wait = Box::pin(b.reserve_processing(OVER, &token));
        let mut c_wait = Box::pin(c.reserve_processing(OVER, &token));
        assert!(poll_once(b_wait.as_mut()).await.is_pending());
        assert!(poll_once(c_wait.as_mut()).await.is_pending());

        drop(a);
        assert!(poll_once(c_wait.as_mut()).await.is_pending());
        let Poll::Ready(Ok(grant)) = poll_once(b_wait.as_mut()).await else {
            panic!("b is next");
        };
        assert!(grant.entered_overflow);
        drop(b_wait);
        drop(b);
        let Poll::Ready(Ok(grant)) = poll_once(c_wait.as_mut()).await else {
            panic!("c is last");
        };
        assert!(grant.entered_overflow);
        drop(c_wait);
        drop(c);
        assert_eq!(overflow_runs(&metrics), 3.0);
        assert_eq!(g.reserved_bytes(), 0);
        assert_eq!(g.overflow_bytes(), 0);
        assert!(g.high_water_bytes() <= SMALL_BUDGET);
    }

    /// Fill `gate`: the overflow slot held (`holder`), one received upload per
    /// name in `sessions` (each holding `F`), and a `filler` grown to fill the
    /// rest of the budget exactly. A processing step of any received upload
    /// then waits for room or for the slot.
    async fn full_budget_with_the_slot_held<const N: usize>(
        gate: &Arc<UploadAdmissionGate>,
        shutdown: &CancellationToken,
        sessions: [&str; N],
    ) -> (UploadReservation, [UploadReservation; N], UploadReservation) {
        let mut holder = received(gate, "holder", shutdown).await;
        assert!(
            process_now(&mut holder, OVER, shutdown)
                .await
                .entered_overflow
        );
        let mut uploads = Vec::with_capacity(N);
        for session in sessions {
            uploads.push(received(gate, session, shutdown).await);
        }
        let mut filler = admit_now(gate, "filler", shutdown).await;
        let room = SMALL_BUDGET - gate.reserved_bytes();
        filler
            .try_grow_in_budget(filler.reserved() + room)
            .expect("room for the filler");
        assert_eq!(gate.reserved_bytes(), SMALL_BUDGET, "the budget is full");
        let uploads = uploads.try_into().expect("one upload per session");
        (holder, uploads, filler)
    }

    /// A processing step granted in budget whose wait then ends without taking
    /// the grant — its future dropped, or shutdown seen first — gives the
    /// growth back, and that serves the waiter behind it. The ending is counted
    /// `cancelled`; the reservation keeps only what it held before.
    #[tokio::test]
    async fn reserve_processing_in_budget_grant_lost_to_a_drop_or_shutdown_serves_the_next_waiter()
    {
        const MORE: u64 = 4 * MIB;
        for lost_to_shutdown in [false, true] {
            let token = CancellationToken::new();
            let first_token = CancellationToken::new();
            let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
            let (_holder, [mut first, mut second], mut filler) =
                full_budget_with_the_slot_held(&g, &token, ["first", "second"]).await;
            let first_target = first.reserved() + MORE;
            let second_target = second.reserved() + MORE;
            let mut first_wait = Box::pin(first.reserve_processing(first_target, &first_token));
            let mut second_wait = Box::pin(second.reserve_processing(second_target, &token));
            assert!(poll_once(first_wait.as_mut()).await.is_pending());
            assert!(poll_once(second_wait.as_mut()).await.is_pending());

            filler.shrink_to(filler.reserved() - MORE);
            assert_eq!(
                g.processing_waiters(),
                1,
                "precondition: the freed room granted the first waiter in budget"
            );
            assert_eq!(g.reserved_bytes(), SMALL_BUDGET);
            if lost_to_shutdown {
                first_token.cancel();
                assert!(
                    matches!(
                        poll_once(first_wait.as_mut()).await,
                        Poll::Ready(Err(UploadRefusal::ShuttingDown))
                    ),
                    "shutdown seen before the grant ends the wait"
                );
            }
            drop(first_wait);
            assert_eq!(
                first.reserved(),
                FIXED_CHARGE_BYTES,
                "shutdown {lost_to_shutdown}: the lost grant leaves the reservation as it was"
            );
            assert!(!first.in_overflow());
            let Poll::Ready(Ok(grant)) = poll_once(second_wait.as_mut()).await else {
                panic!(
                    "shutdown {lost_to_shutdown}: the released growth must serve the next waiter"
                );
            };
            assert!(!grant.entered_overflow);
            drop(second_wait);
            assert_eq!(second.reserved(), second_target);
            assert_eq!(g.reserved_bytes(), SMALL_BUDGET);
            assert_eq!(g.processing_waiters(), 0);
            assert_eq!(admission_count(&metrics, ADMISSION_RESULT_CANCELLED), 1.0);
        }
    }

    /// Work in progress goes before new work: with a processing step and an
    /// admission both queued, a release with room for only one of them grants
    /// the processing step, and the admission keeps waiting.
    #[tokio::test]
    async fn serve_grants_a_processing_waiter_before_a_queued_admission() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let (_holder, [mut worker], mut filler) =
            full_budget_with_the_slot_held(&g, &token, ["worker"]).await;
        let target = worker.reserved() + C;
        let mut working = Box::pin(worker.reserve_processing(target, &token));
        assert!(poll_once(working.as_mut()).await.is_pending());
        let mut arrival = Box::pin(g.admit(UploadRpc::UpdateLayer, "arrival", &token));
        assert!(poll_once(arrival.as_mut()).await.is_pending());
        assert_eq!((g.processing_waiters(), g.waiters()), (1, 1));

        filler.shrink_to(filler.reserved() - C);
        assert!(
            poll_once(arrival.as_mut()).await.is_pending(),
            "a new admission must not take the room a processing step waits for"
        );
        let Poll::Ready(Ok(grant)) = poll_once(working.as_mut()).await else {
            panic!("the freed room must grant the processing step");
        };
        assert!(!grant.entered_overflow);
        assert_eq!(g.waiters(), 1);
    }

    /// A processing waiter that does not fit the freed room is skipped, not a
    /// barrier: a later waiter that does fit is granted, and the earlier one
    /// keeps its place.
    #[tokio::test]
    async fn serve_skips_a_processing_waiter_that_does_not_fit_and_grants_a_later_one() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let (_holder, [mut big, mut small], mut filler) =
            full_budget_with_the_slot_held(&g, &token, ["big", "small"]).await;
        let big_target = big.reserved() + 2 * C;
        let small_target = small.reserved() + C;
        let mut big_wait = Box::pin(big.reserve_processing(big_target, &token));
        let mut small_wait = Box::pin(small.reserve_processing(small_target, &token));
        assert!(poll_once(big_wait.as_mut()).await.is_pending());
        assert!(poll_once(small_wait.as_mut()).await.is_pending());

        filler.shrink_to(filler.reserved() - C);
        assert!(
            poll_once(big_wait.as_mut()).await.is_pending(),
            "fixture self-check: the freed room is too small for the earlier waiter"
        );
        let Poll::Ready(Ok(grant)) = poll_once(small_wait.as_mut()).await else {
            panic!("the later waiter fits the freed room and must be granted past the earlier one");
        };
        assert!(!grant.entered_overflow);
        assert_eq!(
            g.processing_waiters(),
            1,
            "the earlier waiter keeps its place"
        );
        assert_eq!(g.reserved_bytes(), SMALL_BUDGET);
    }

    /// Work in progress goes before new work after a slot hand-off too. When
    /// the holder drops, the processing-queue head takes the slot and its
    /// in-budget bytes leave the budget; a second processing waiter that fits
    /// only in that freed room is granted there before a queued admission
    /// that would also fit is served.
    #[tokio::test]
    async fn serve_grants_a_processing_waiter_the_room_a_slot_hand_off_frees_before_an_admission() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        assert!(
            process_now(&mut holder, OVER, &token)
                .await
                .entered_overflow
        );
        let mut head = received(&g, "head", &token).await;
        head.try_grow_in_budget(C)
            .expect("room for the head's in-budget bytes");
        let mut second = received(&g, "second", &token).await;
        let mut filler = admit_now(&g, "filler", &token).await;
        let room = SMALL_BUDGET - g.reserved_bytes();
        filler
            .try_grow_in_budget(filler.reserved() + room)
            .expect("room for the filler");
        assert_eq!(g.reserved_bytes(), SMALL_BUDGET, "the budget is full");

        let mut head_wait = Box::pin(head.reserve_processing(OVER, &token));
        let second_target = second.reserved() + C;
        let mut second_wait = Box::pin(second.reserve_processing(second_target, &token));
        let mut arrival = Box::pin(g.admit(UploadRpc::UpdateLayer, "arrival", &token));
        assert!(poll_once(head_wait.as_mut()).await.is_pending());
        assert!(poll_once(second_wait.as_mut()).await.is_pending());
        assert!(poll_once(arrival.as_mut()).await.is_pending());
        assert_eq!((g.processing_waiters(), g.waiters()), (2, 1));

        // Frees the slot; the head's C in-budget bytes follow it out of B:
        // room for exactly one of the second waiter (C) and the arrival (C).
        drop(holder);
        let Poll::Ready(Ok(grant)) = poll_once(head_wait.as_mut()).await else {
            panic!("the holder's drop must hand the slot to the queue's head");
        };
        assert!(grant.entered_overflow);
        assert!(
            poll_once(arrival.as_mut()).await.is_pending(),
            "a new admission must not take the room the head's hand-off freed while a \
             processing step waits for it"
        );
        let Poll::Ready(Ok(grant)) = poll_once(second_wait.as_mut()).await else {
            panic!("the room the hand-off freed must grant the waiting processing step");
        };
        assert!(!grant.entered_overflow);
        assert_eq!(g.reserved_bytes(), SMALL_BUDGET);
        assert_eq!((g.processing_waiters(), g.waiters()), (0, 1));
    }

    /// `live_edit_upload_overflow_waiters` falls back to zero however a slot
    /// wait ends — dropped, shut down, granted in budget, granted the slot —
    /// and `live_edit_upload_overflow_bytes` falls back to zero once the last
    /// holder vacates the slot.
    #[tokio::test]
    async fn overflow_gauges_fall_back_to_zero_however_a_wait_ends() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let waiters = |m: &Metrics| metric(m, "live_edit_upload_overflow_waiters", &[]);
        let slot_bytes = |m: &Metrics| metric(m, "live_edit_upload_overflow_bytes", &[]);
        let (holder, [mut dropped, mut shut, mut fits, mut next], filler) =
            full_budget_with_the_slot_held(&g, &token, ["dropped", "shut", "fits", "next"]).await;
        assert_eq!(slot_bytes(&metrics), OVER as f64);

        let mut wait = Box::pin(dropped.reserve_processing(OVER, &token));
        assert!(poll_once(wait.as_mut()).await.is_pending());
        assert_eq!(waiters(&metrics), 1.0);
        drop(wait);
        assert_eq!(waiters(&metrics), 0.0, "after a dropped wait");

        let shut_token = CancellationToken::new();
        let mut wait = Box::pin(shut.reserve_processing(OVER, &shut_token));
        assert!(poll_once(wait.as_mut()).await.is_pending());
        assert_eq!(waiters(&metrics), 1.0);
        shut_token.cancel();
        assert!(matches!(
            poll_once(wait.as_mut()).await,
            Poll::Ready(Err(UploadRefusal::ShuttingDown))
        ));
        drop(wait);
        assert_eq!(waiters(&metrics), 0.0, "after a shutdown");

        let fits_target = fits.reserved() + C;
        let mut wait = Box::pin(fits.reserve_processing(fits_target, &token));
        assert!(poll_once(wait.as_mut()).await.is_pending());
        assert_eq!(waiters(&metrics), 1.0);
        drop(filler);
        let Poll::Ready(Ok(grant)) = poll_once(wait.as_mut()).await else {
            panic!("the freed room must grant the step in budget");
        };
        assert!(!grant.entered_overflow);
        drop(wait);
        assert_eq!(waiters(&metrics), 0.0, "after an in-budget grant");

        let mut wait = Box::pin(next.reserve_processing(OVER + 1, &token));
        assert!(poll_once(wait.as_mut()).await.is_pending());
        assert_eq!(waiters(&metrics), 1.0);
        drop(holder);
        let Poll::Ready(Ok(grant)) = poll_once(wait.as_mut()).await else {
            panic!("the holder's drop must hand the slot on");
        };
        assert!(grant.entered_overflow);
        drop(wait);
        assert_eq!(waiters(&metrics), 0.0, "after a slot grant");
        assert_eq!(slot_bytes(&metrics), (OVER + 1) as f64);
        drop(next);
        assert_eq!(
            slot_bytes(&metrics),
            0.0,
            "the last holder vacated the slot"
        );
    }

    /// The processing-wait histogram appears with the first processing step,
    /// recording one observation per step that grows the reservation.
    #[tokio::test]
    async fn reserve_processing_records_one_processing_wait_per_growing_step() {
        use crate::observability::metrics::test_util::{histogram_bucket, metric_type};

        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = received(&g, "r", &token).await;
        process_now(&mut r, C, &token).await;
        process_now(&mut r, OVER, &token).await;
        process_now(&mut r, 1, &token).await; // holds more already: not recorded
        let text = render(&metrics);
        assert_eq!(
            metric_type(&text, "live_edit_upload_processing_wait_seconds").as_deref(),
            Some("histogram")
        );
        assert_eq!(
            histogram_bucket(
                &text,
                "live_edit_upload_processing_wait_seconds",
                "+Inf",
                &[("rpc", UPLOAD_RPC_UPDATE_LAYER)]
            ),
            Some(2.0)
        );
    }

    // ── Processing without waiting: `try_hold_processing` (b.26e) ──────────

    /// `live_edit_upload_processing_wait_seconds{rpc="update_layer"}`
    /// observations so far; 0 before the first.
    fn processing_waits(metrics: &Metrics) -> f64 {
        use crate::observability::metrics::test_util::histogram_bucket;

        histogram_bucket(
            &render(metrics),
            "live_edit_upload_processing_wait_seconds",
            "+Inf",
            &[("rpc", UPLOAD_RPC_UPDATE_LAYER)],
        )
        .unwrap_or(0.0)
    }

    /// A target that fits the budget is held there at once, exactly, even with
    /// a processing step queued for the slot — the queued step is neither
    /// served nor overtaken — and records neither a processing wait (it never
    /// waits: an in-commit charge is not a processing step) nor an overflow
    /// run.
    #[tokio::test]
    async fn try_hold_processing_fitting_target_grows_in_budget_with_a_slot_waiter_queued() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        assert!(
            process_now(&mut holder, OVER, &token)
                .await
                .entered_overflow
        );
        let mut waiter = received(&g, "waiter", &token).await;
        let mut waiting = Box::pin(waiter.reserve_processing(OVER, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        let mut r = received(&g, "r", &token).await;
        let (waits, runs) = (processing_waits(&metrics), overflow_runs(&metrics));

        let target = r.reserved() + C;
        let grant = r
            .try_hold_processing(target)
            .expect("a target that fits the budget is held at once");

        assert!(!grant.entered_overflow);
        assert_eq!(grant.waited, Duration::ZERO);
        assert_eq!(r.reserved(), target);
        assert!(!r.in_overflow());
        assert_eq!(g.reserved_bytes(), FIXED_CHARGE_BYTES + target);
        assert_eq!(g.overflow_bytes(), OVER, "the holder keeps the slot");
        assert_eq!(g.processing_waiters(), 1, "the queued step is untouched");
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        assert_eq!(
            processing_waits(&metrics),
            waits,
            "an in-commit growth is no processing wait"
        );
        assert_eq!(overflow_runs(&metrics), runs);
    }

    /// The slot holder books a higher target on its slot in place, with a
    /// step queued for the slot, never touching the budget or the queue,
    /// and records neither a processing wait nor an overflow run.
    #[tokio::test]
    async fn try_hold_processing_slot_holder_grows_in_place() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;
        let mut waiter = received(&g, "waiter", &token).await;
        let mut waiting = Box::pin(waiter.reserve_processing(OVER, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        let in_budget = g.reserved_bytes();
        let waits = processing_waits(&metrics);

        let grant = holder
            .try_hold_processing(2 * OVER)
            .expect("the slot holder grows in place");

        assert!(!grant.entered_overflow, "it was already in the slot");
        assert_eq!(holder.reserved(), 2 * OVER);
        assert!(holder.in_overflow());
        assert_eq!(g.overflow_bytes(), 2 * OVER);
        assert_eq!(g.reserved_bytes(), in_budget, "the budget is untouched");
        assert_eq!(g.processing_waiters(), 1);
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        assert_eq!(overflow_runs(&metrics), 1.0, "only the holder's entry");
        assert_eq!(
            processing_waits(&metrics),
            waits,
            "a growth in place is no processing wait"
        );
    }

    /// With the slot free and no processing step queued, a target the budget
    /// cannot hold takes the slot: the reservation's in-budget bytes leave the
    /// budget with it, and exactly one overflow run is counted — a further
    /// growth on the slot counts none. It never waits, so it records no
    /// processing-wait observation: one is recorded per step that waited.
    #[tokio::test]
    async fn try_hold_processing_takes_the_free_slot_and_counts_one_overflow_run() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = received(&g, "r", &token).await;
        let _other = received(&g, "other", &token).await;
        assert_eq!(g.processing_waiters(), 0, "precondition: no step queued");

        let grant = r
            .try_hold_processing(OVER)
            .expect("the free slot is taken at once");

        assert!(grant.entered_overflow);
        assert_eq!(grant.waited, Duration::ZERO);
        assert!(r.in_overflow());
        assert_eq!(r.reserved(), OVER);
        assert_eq!(g.overflow_bytes(), OVER);
        assert_eq!(
            g.reserved_bytes(),
            FIXED_CHARGE_BYTES,
            "r's in-budget bytes left the budget; only the other upload's remain"
        );
        assert_eq!(overflow_runs(&metrics), 1.0);
        assert_eq!(
            processing_waits(&metrics),
            0.0,
            "a slot entry that never waited is no processing wait"
        );

        r.try_hold_processing(OVER + 1)
            .expect("the holder grows in place");
        assert_eq!(g.overflow_bytes(), OVER + 1);
        assert_eq!(overflow_runs(&metrics), 1.0, "one entry, one run");
        assert_eq!(processing_waits(&metrics), 0.0);
    }

    /// A target that fits neither the budget nor a free slot is refused at
    /// once with nothing changed — not the reservation, the gate, the queue
    /// or any metric — and it never queues: with a step already queued for
    /// the slot, the holder's drop hands the slot to that step, not to the
    /// refused reservation.
    #[tokio::test]
    async fn try_hold_processing_refused_changes_nothing_and_never_queues() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;
        let mut r = received(&g, "r", &token).await;
        let before = render(&metrics);
        let (in_budget, slot, held) = (g.reserved_bytes(), g.overflow_bytes(), r.reserved());

        assert!(
            r.try_hold_processing(OVER).is_none(),
            "the slot is held: refused"
        );
        assert_eq!(r.reserved(), held);
        assert!(!r.in_overflow());
        assert_eq!((g.reserved_bytes(), g.overflow_bytes()), (in_budget, slot));
        assert_eq!(g.processing_waiters(), 0, "a refusal never queues");
        assert_eq!(render(&metrics), before, "a refusal records no metric");

        let mut waiter = received(&g, "waiter", &token).await;
        let mut waiting = Box::pin(waiter.reserve_processing(OVER, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        assert!(
            r.try_hold_processing(OVER).is_none(),
            "the slot is held and a step is queued for it: refused"
        );
        assert_eq!(g.processing_waiters(), 1, "still only the queued step");

        drop(holder);
        let Poll::Ready(Ok(grant)) = poll_once(waiting.as_mut()).await else {
            panic!("the holder's drop must hand the slot to the queued step");
        };
        assert!(grant.entered_overflow);
        assert!(!r.in_overflow(), "the refused reservation never queued");
        assert_eq!(r.reserved(), held);
        assert_eq!(
            overflow_runs(&metrics),
            2.0,
            "the holder's and the waiter's"
        );
    }

    /// A target below what the reservation holds shrinks it to exactly that
    /// target and gives the difference back, which serves a queued waiter
    /// the freed bytes now fit; the slot holder's shrink re-books its slot.
    /// A shrink records no processing wait and no overflow run.
    #[tokio::test]
    async fn try_hold_processing_lower_target_shrinks_serves_the_queue_and_records_nothing() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = admit_now(&g, "a", &token).await;
        let _b = admit_now(&g, "b", &token).await;
        let _c = admit_now(&g, "c", &token).await;
        r.try_grow_in_budget(SMALL_BUDGET - 2 * C)
            .expect("fill the budget exactly");
        let mut waiter = std::pin::pin!(g.admit(UploadRpc::UpdateLayer, "d", &token));
        assert!(
            poll_once(waiter.as_mut()).await.is_pending(),
            "fixture self-check: the budget is too full for an admission"
        );

        let target = r.reserved() - C;
        let grant = r
            .try_hold_processing(target)
            .expect("a shrink is always held");

        assert!(!grant.entered_overflow);
        assert_eq!(r.reserved(), target);
        let Poll::Ready(Ok(_granted)) = poll_once(waiter.as_mut()).await else {
            panic!("the bytes the shrink gave back must serve the queued admission");
        };
        assert_eq!(processing_waits(&metrics), 0.0, "a shrink is no wait");
        assert_eq!(overflow_runs(&metrics), 0.0);

        let (slot_gate, slot_metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&slot_gate, "holder", &token).await;
        assert!(
            holder
                .try_hold_processing(2 * OVER)
                .expect("the free slot")
                .entered_overflow
        );
        holder
            .try_hold_processing(OVER)
            .expect("the holder's shrink");
        assert_eq!(holder.reserved(), OVER);
        assert!(holder.in_overflow());
        assert_eq!(
            slot_gate.overflow_bytes(),
            OVER,
            "the slot is re-booked at the target"
        );
        assert_eq!(overflow_runs(&slot_metrics), 1.0, "only the entry counted");
        assert_eq!(
            processing_waits(&slot_metrics),
            0.0,
            "neither the entry nor the shrink is a wait"
        );
    }

    /// A target equal to what the reservation holds changes nothing: the gate
    /// reports no change to its charge observer, and nothing is recorded.
    #[tokio::test]
    async fn try_hold_processing_equal_target_changes_nothing() {
        let token = CancellationToken::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = received(&g, "r", &token).await;
        process_now(&mut r, C, &token).await;
        let reports = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&reports);
        g.set_charge_observer_for_test(Some(Arc::new(move |c| {
            sink.lock().expect("observer lock").push(c);
        })));
        let before = render(&metrics);

        let grant = r.try_hold_processing(C).expect("an equal target is held");

        assert!(!grant.entered_overflow);
        assert_eq!(r.reserved(), C);
        assert_eq!(g.reserved_bytes(), C);
        assert!(
            reports.lock().expect("observer lock").is_empty(),
            "an equal target changes nothing the gate holds"
        );
        assert_eq!(render(&metrics), before, "and records nothing");
    }

    // ── The test-only charge observer (b.26e T7) ────────────────────────────

    /// The observer is called after every change to what the gate holds —
    /// admission, growth, the end of the receive phase, a step into the slot,
    /// the release — and the last report of each operation is the state it
    /// left. Cleared, it is called no more.
    #[tokio::test]
    async fn charge_observer_last_report_is_the_gate_state_after_every_change() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let reports = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&reports);
        g.set_charge_observer_for_test(Some(Arc::new(move |c| {
            sink.lock().expect("observer lock").push(c);
        })));
        let taken = || std::mem::take(&mut *reports.lock().expect("observer lock"));
        let now = |g: &UploadAdmissionGate| GateCharge {
            reserved_bytes: g.reserved_bytes(),
            overflow_bytes: g.overflow_bytes(),
        };

        let mut r = admit_now(&g, "r", &token).await;
        assert_eq!(taken().last(), Some(&now(&g)), "after admission");
        assert_eq!(now(&g).reserved_bytes, C);
        r.try_grow_in_budget(C + 1).expect("room");
        assert_eq!(taken().last(), Some(&now(&g)), "after a receive growth");
        r.end_receive();
        assert_eq!(taken().last(), Some(&now(&g)), "after the receive phase");
        assert!(
            r.try_hold_processing(OVER)
                .expect("free slot")
                .entered_overflow
        );
        let entry = taken();
        assert_eq!(entry.last(), Some(&now(&g)), "after the step into the slot");
        assert_eq!(
            now(&g),
            GateCharge {
                reserved_bytes: 0,
                overflow_bytes: OVER
            }
        );
        drop(r);
        assert_eq!(taken().last(), Some(&now(&g)), "after the release");
        assert_eq!(
            now(&g),
            GateCharge {
                reserved_bytes: 0,
                overflow_bytes: 0
            }
        );

        g.set_charge_observer_for_test(None);
        let _s = admit_now(&g, "s", &token).await;
        assert!(taken().is_empty(), "a cleared observer is not called");
    }

    // ── The commit-charge seam (b.26e) ──────────────────────────────────────

    /// [`UploadCommitCharge`]: every target is an object-state term held on
    /// top of the base processing charge, `held` is what the reservation
    /// holds beyond the base, and a lower target — through either call —
    /// shrinks back towards the base, never below it.
    #[tokio::test]
    async fn upload_commit_charge_holds_each_target_on_top_of_the_base() {
        let gs = GracefulShutdown::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = received(&g, "r", gs.begun()).await;
        let base = C;
        process_now(&mut r, base, gs.begun()).await;
        let mut charge = UploadCommitCharge::new(&mut r, base, &gs, "s", "layer");
        assert_eq!(charge.held(), 0);

        assert!(charge.try_hold(3_000));
        assert_eq!(charge.held(), 3_000);
        assert_eq!(g.reserved_bytes(), base + 3_000);
        assert!(charge.try_hold(1_000), "a lower target is a shrink");
        assert_eq!(charge.held(), 1_000);
        assert_eq!(g.reserved_bytes(), base + 1_000);

        charge.hold(5_000).await.expect("fits the budget");
        assert_eq!(charge.held(), 5_000);
        assert_eq!(g.reserved_bytes(), base + 5_000);
        charge.hold(5_000).await.expect("an equal target");
        assert_eq!(charge.held(), 5_000);
        charge.hold(0).await.expect("a shrink");
        assert_eq!(charge.held(), 0);
        assert!(!charge.shut_down());
        // The charge's borrow of `r` ends at its last use above.
        assert_eq!(r.reserved(), base, "the base itself is never given back");
        assert_eq!(g.reserved_bytes(), base);
    }

    /// [`UploadCommitCharge`] prices a need with the object-state formulas
    /// (`target` is [`object_state_charge`], the cached and WATCH formulas
    /// recomputed here term by term), and its rewrite allowance is what it
    /// holds beyond the need's target with no rewrite, at the rewrite prices:
    /// zero before anything is held, the difference once more is.
    #[tokio::test]
    async fn upload_commit_charge_target_is_the_formulas_and_its_allowance_what_is_held_beyond_b26e(
    ) {
        let gs = GracefulShutdown::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = received(&g, "r", gs.begun()).await;
        let base = C;
        process_now(&mut r, base, gs.begun()).await;
        let mut charge = UploadCommitCharge::new(&mut r, base, &gs, "s", "layer");
        let rewrite = RewriteCount {
            rows: 2,
            list_entries: 7,
            indexed_sections: 3,
            adoption_entries: 5,
            walked_sections: 0,
            rollback_rebuild_bytes: 0,
        };
        let cached = ObjectStateNeed::cached_growing(900, 1_000, rewrite);
        assert_eq!(
            charge.target(&cached),
            900 + 1_000 + rewrite_charge(RewritePrices::CACHED, rewrite)
        );
        let snapshot = crate::storage::commit_charge::WatchSnapshot {
            entries: 11,
            in_flight: 4,
            plain_in_flight: 5,
            items: 13,
            window_bytes: 100,
            name_bytes: 17,
        };
        let queued = QueuedCommands {
            txn: 12,
            pre_image: 3,
            probe: 0,
        };
        let watch = ObjectStateNeed::Watch {
            snapshot,
            fixed: true,
            rewrite,
            queued,
        };
        assert_eq!(
            charge.target(&watch),
            WATCH_ATTEMPT_FIXED_BYTES
                + 11 * WATCH_SNAPSHOT_ENTRY_CHARGE_BYTES
                + 17 * WATCH_SNAPSHOT_NAME_BYTE_CHARGE_BYTES
                + 5 * WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES
                + 4 * WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES
                + 13 * WATCH_SNAPSHOT_ITEM_CHARGE_BYTES
                + (100 * WATCH_SNAPSHOT_REPLY_BYTE_CHARGE_CENTI).div_ceil(EXPANSION_SCALE)
                + rewrite_charge(RewritePrices::WATCH, rewrite)
                + QUEUED_COMMAND_CHARGE_BYTES * (16 + 4),
            "the queued counts priced at next-power-of-two slots, an empty queue at none"
        );
        assert_eq!(
            charge.target(&ObjectStateNeed::watch_fixed()),
            WATCH_ATTEMPT_FIXED_BYTES,
            "a structural WATCH attempt's head: its fixed cost alone"
        );

        let no_rewrite = charge.target(&cached.with_rewrite(RewriteCount::default()));
        assert_eq!(
            charge.rewrite_allowance(&cached).bytes,
            0,
            "nothing held, nothing to spend"
        );
        assert!(charge.try_hold(no_rewrite + 4_321));
        let allowance = charge.rewrite_allowance(&cached);
        assert_eq!(
            allowance.bytes, 4_321,
            "held beyond the need without its rewrite"
        );
        assert_eq!(
            (
                allowance.row_bytes,
                allowance.list_entry_bytes,
                allowance.index_entry_bytes,
                allowance.adoption_entry_bytes
            ),
            (
                CACHED_REWRITE_ROW_CHARGE_BYTES,
                CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES,
                REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES,
                REWRITE_ADOPTION_ENTRY_CHARGE_BYTES
            )
        );
    }

    /// The window script's record (b.gvf), which carries the WATCH attempt's
    /// `W_FIXED` and `W_CMD` refits.
    const WINDOW_SCRIPT_RESULTS_JSON: &str =
        "docs-internal/validation/watch-window-script-b-gvf-20261010/results.json";

    /// The committed record at `path` (relative to the crate root), parsed;
    /// `None` in the published source tree, which has no `docs-internal/`
    /// (only there: in this repository a missing record fails).
    fn committed_record(path: &str) -> Option<serde_json::Value> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("docs-internal").is_dir() {
            eprintln!("skipped: no docs-internal/ (published source tree)");
            return None;
        }
        let raw = std::fs::read_to_string(root.join(path))
            .unwrap_or_else(|e| panic!("the committed record {path}: {e}"));
        Some(serde_json::from_str(&raw).expect("the record is valid JSON"))
    }

    /// `W_FIXED` is the decider-ruled refit (b.gvf), 124,365 B, and the
    /// window script's record derives it (`w_fixed_refit`): the empty-base
    /// edit's warm processing-phase peak, the largest of its runs, plus fred's
    /// first-use router channel block, less the earlier terms of the fixture's
    /// processing charge, each at its production basis — `F`,
    /// `byte[UpdateSlow]` × the delta and `element[UpdateSlow]` × the elements
    /// at their constants ÷ 1.25 (÷ 125 for the hundredths), the body 1:1 —
    /// the element term at the larger of its basis and the WATCH path's own
    /// figure, × 1.25 rounded up. The bases are
    /// computed here from the constants, in thousandths of a byte, so a
    /// re-wired earlier constant fails here until `W_FIXED` is re-derived.
    /// `W_CMD` is that record's refit on the
    /// window-script transport (`w_cmd_refit`, computed by
    /// `scripts/m3_wcmd_fit.py` into the file `w_cmd_refit.output` names): the
    /// candidate ⌈largest per-command excess × 1.25⌉ when every recorded WATCH
    /// segment, on both transports, is at or under its charge at it (the
    /// adoption check), else the smallest value at which none is over, never
    /// below the candidate. `W_PLAIN_CMD` keeps the plain read's per-command
    /// fit, the figure `W_CMD` held before the refit. A constant re-wired
    /// without its record, or the reverse, fails here.
    #[test]
    fn watch_fixed_and_command_factors_are_the_window_script_records_refits_b_gvf() {
        assert_eq!(
            WATCH_ATTEMPT_FIXED_BYTES, 124_365,
            "W_FIXED is the decider-ruled refit"
        );
        let Some(results) = committed_record(WINDOW_SCRIPT_RESULTS_JSON) else {
            return;
        };
        let u64_at = |pointer: &str| {
            results
                .pointer(pointer)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| panic!("results.json carries {pointer} as a whole number"))
        };
        let f64_at = |pointer: &str| {
            results
                .pointer(pointer)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or_else(|| panic!("results.json carries {pointer} as a number"))
        };
        // A recorded figure in thousandths of a byte (the records give at
        // most two decimals).
        let milli = |bytes: f64| (bytes * 1_000.0).round() as u64;

        let warm_peak = results
            .pointer("/w_fixed_refit/runs_warm_client_bytes")
            .and_then(serde_json::Value::as_array)
            .expect("results.json carries w_fixed_refit.runs_warm_client_bytes")
            .iter()
            .map(|run| {
                run.as_u64()
                    .expect("a run's peak is a whole number of bytes")
            })
            .max()
            .expect("at least one run");
        let binding = warm_peak + u64_at("/w_fixed_refit/first_use_channel_block_bytes");
        assert_eq!(binding, u64_at("/w_fixed_refit/binding_peak_bytes"));

        // The earlier terms at their production bases, in thousandths of a
        // byte: a constant ÷ 1.25 is × 800, a hundredths constant ÷ 125 is ×
        // 8; the element term at the larger of its basis and the WATCH path's
        // own figure.
        let body = u64_at("/w_fixed_refit/fixture/body_bytes");
        let delta = u64_at("/w_fixed_refit/fixture/delta_bytes");
        let elements = u64_at("/w_fixed_refit/fixture/elements");
        let element_basis = ELEMENT_CHARGE_UPDATE_SLOW_BYTES * 800;
        let watch_element = milli(f64_at("/w_fixed_refit/watch_element_figure_bytes"));
        assert!(
            watch_element <= element_basis,
            "the WATCH path's element figure is at or below the basis, so the basis nets"
        );
        let bases = FIXED_CHARGE_BYTES * 800
            + body * 1_000
            + DELTA_BYTE_CHARGE_UPDATE_SLOW_CENTI * delta * 8
            + element_basis.max(watch_element) * elements;
        assert_eq!(
            milli(f64_at("/w_fixed_refit/earlier_terms_bases_bytes")),
            bases,
            "the record's bases are the production bases of the constants wired"
        );
        // The fixture's processing step at the constants is what its runs
        // printed, so its counts are the record's.
        assert_eq!(
            FIXED_CHARGE_BYTES
                + body
                + (DELTA_BYTE_CHARGE_UPDATE_SLOW_CENTI * delta).div_ceil(100)
                + ELEMENT_CHARGE_UPDATE_SLOW_BYTES * elements,
            u64_at("/w_fixed_refit/processing_step_charge_bytes")
        );
        let netted = binding * 1_000 - bases;
        assert_eq!(milli(f64_at("/w_fixed_refit/netted_bytes")), netted);
        assert_eq!(
            WATCH_ATTEMPT_FIXED_BYTES,
            (netted * 5).div_ceil(4_000),
            "W_FIXED = ⌈netted × 1.25⌉"
        );
        assert_eq!(WATCH_ATTEMPT_FIXED_BYTES, u64_at("/w_fixed_refit/wired"));

        let largest = results
            .pointer("/w_cmd_refit/largest_per_command_bytes")
            .and_then(serde_json::Value::as_f64)
            .expect("results.json carries w_cmd_refit.largest_per_command_bytes");
        let candidate = (largest * 1.25).ceil() as u64;
        assert_eq!(
            candidate,
            u64_at("/w_cmd_refit/fit_candidate"),
            "the fit's candidate is ⌈largest × 1.25⌉"
        );

        // The fit's output: the adoption check at the candidate, with every
        // segment over its charge there (its peak, its charge at the
        // candidate, its commands in flight).
        let fit_output = results
            .pointer("/w_cmd_refit/output")
            .and_then(serde_json::Value::as_str)
            .expect("results.json names w_cmd_refit.output, the fit's output");
        let record_dir = std::path::Path::new(WINDOW_SCRIPT_RESULTS_JSON)
            .parent()
            .expect("the record has a directory");
        let fit = committed_record(
            record_dir
                .join(fit_output)
                .to_str()
                .expect("the record's paths are UTF-8"),
        )
        .expect("results.json exists, so docs-internal/ does");
        let fit_u64 = |value: &serde_json::Value, pointer: &str| {
            value
                .pointer(pointer)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| panic!("{fit_output} carries {pointer} as a whole number"))
        };
        // `W_CMD` is netted of `W_FIXED`'s basis, so its fit must be the one
        // netted at the `W_FIXED` wired now: re-wiring `W_FIXED` without
        // re-running the fit fails here.
        assert_eq!(
            fit_u64(&fit, "/w_fixed_used/value"),
            WATCH_ATTEMPT_FIXED_BYTES,
            "the W_CMD fit was netted at the W_FIXED wired now"
        );
        assert_eq!(fit_u64(&fit, "/fit/candidate"), candidate);
        assert_eq!(fit_u64(&fit, "/check_at_candidate/w_cmd"), candidate);
        let over_at_candidate = fit
            .pointer("/check_at_candidate/over")
            .and_then(serde_json::Value::as_array)
            .expect("the fit's output lists the segments over their charge at the candidate");

        // The bound rule. A segment's charge grows with W_CMD by its commands
        // in flight, so a segment at or under its charge at the candidate stays
        // so at any larger value; each one over needs W_CMD at least
        // ⌈(peak − its charge without W_CMD) / in flight⌉.
        let expected = over_at_candidate
            .iter()
            .map(|seg| {
                let peak = fit_u64(seg, "/peak");
                let charge = fit_u64(seg, "/charge");
                let in_flight = fit_u64(seg, "/in_flight");
                assert!(
                    in_flight > 0,
                    "a segment over its charge with no command in flight is bounded by no W_CMD"
                );
                let without_w_cmd = charge - candidate * in_flight;
                (peak - without_w_cmd).div_ceil(in_flight)
            })
            .fold(candidate, u64::max);
        assert_eq!(
            WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES, expected,
            "W_CMD is the candidate if no recorded segment is over its charge there, \
             else the smallest value at which none is"
        );
        assert_eq!(
            WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES,
            u64_at("/w_cmd_refit/wired")
        );
        assert_eq!(fit_u64(&fit, "/wired"), WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES);
        assert_eq!(
            fit_u64(&fit, "/check_at_wired/w_cmd"),
            WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES
        );
        assert_eq!(
            fit.pointer("/check_at_wired/over")
                .and_then(serde_json::Value::as_array)
                .map(Vec::len),
            Some(0),
            "no recorded segment is over its charge at the W_CMD wired"
        );
        assert_eq!(
            WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES,
            u64_at("/w_cmd_refit/previous"),
            "W_PLAIN_CMD keeps the plain read's per-command fit"
        );
        assert_eq!(
            results.pointer("/w_cmd_refit/bound_check/adopted"),
            Some(&serde_json::Value::Bool(true)),
            "W_CMD is wired only once its bound check adopted it"
        );
    }

    /// The seam's two growths where the target needs the slot and the slot
    /// is held: `try_hold` is refused with nothing changed, and `hold` waits
    /// for the slot and is granted it when the holder drops.
    #[tokio::test]
    async fn upload_commit_charge_try_hold_is_refused_and_hold_waits_for_the_slot() {
        let gs = GracefulShutdown::new();
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", gs.begun()).await;
        process_now(&mut holder, OVER, gs.begun()).await;
        let mut r = received(&g, "r", gs.begun()).await;
        let base = r.reserved();
        let mut charge = UploadCommitCharge::new(&mut r, base, &gs, "s", "layer");

        assert!(!charge.try_hold(OVER), "neither the budget nor the slot");
        assert_eq!(charge.held(), 0);
        assert_eq!(g.processing_waiters(), 0, "try_hold never queues");

        let mut holding = Box::pin(charge.hold(OVER));
        assert!(poll_once(holding.as_mut()).await.is_pending());
        assert_eq!(g.processing_waiters(), 1, "hold queues for the slot");
        drop(holder);
        let Poll::Ready(Ok(())) = poll_once(holding.as_mut()).await else {
            panic!("the holder's drop must grant the waiting hold the slot");
        };
        drop(holding);
        assert_eq!(charge.held(), OVER);
        assert_eq!(g.overflow_bytes(), base + OVER);
        assert_eq!(overflow_runs(&metrics), 2.0);
        assert!(!charge.shut_down());
    }

    /// A `hold` waiting for room when the service shuts down answers the
    /// shared shutdown status — `UNAVAILABLE "server shutting down"` with the
    /// `SERVER_SHUTTING_DOWN` `ErrorInfo` and its retry hint, the same status
    /// the handlers' own admission and processing waits answer — grants
    /// nothing, and is recorded on [`UploadCommitCharge::shut_down`] for the
    /// handler's completion line.
    #[tokio::test]
    async fn upload_commit_charge_hold_ended_by_shutdown_answers_unavailable_and_records_it() {
        let gs = GracefulShutdown::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", gs.begun()).await;
        process_now(&mut holder, OVER, gs.begun()).await;
        let mut r = received(&g, "r", gs.begun()).await;
        let base = r.reserved();
        let mut charge = UploadCommitCharge::new(&mut r, base, &gs, "s", "layer");

        let mut holding = Box::pin(charge.hold(OVER));
        assert!(poll_once(holding.as_mut()).await.is_pending());
        gs.cancel();
        let Poll::Ready(Err(err)) = poll_once(holding.as_mut()).await else {
            panic!("shutdown must end the waiting hold with an error");
        };
        drop(holding);

        let ServiceError::UnavailableWithDetail { message, detail } = err else {
            panic!("expected UNAVAILABLE with the shutdown detail, got {err:?}");
        };
        assert_eq!(
            message,
            crate::grpc::transport::server_shutting_down().message()
        );
        assert_eq!(
            *detail,
            RefusalDetail::server_shutting_down(),
            "the shared shutdown detail: SERVER_SHUTTING_DOWN and its retry hint"
        );
        let status = tonic::Status::from(ServiceError::UnavailableWithDetail { message, detail });
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(charge.shut_down());
        assert_eq!(charge.held(), 0, "nothing was granted");
        assert_eq!(g.processing_waiters(), 0, "the wait left the queue");
    }

    /// The slot holder's growth and shrink in place re-book the slot, and
    /// each reports exactly one change to the charge observer: the slot at
    /// its new size, the budget untouched.
    #[tokio::test]
    async fn charge_observer_reports_each_slot_rebooking_in_place() {
        let token = CancellationToken::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        process_now(&mut holder, OVER, &token).await;
        let in_budget = g.reserved_bytes();
        let reports = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&reports);
        g.set_charge_observer_for_test(Some(Arc::new(move |c| {
            sink.lock().expect("observer lock").push(c);
        })));
        let taken = || std::mem::take(&mut *reports.lock().expect("observer lock"));

        holder
            .try_hold_processing(2 * OVER)
            .expect("the holder grows in place");
        assert_eq!(
            taken(),
            vec![GateCharge {
                reserved_bytes: in_budget,
                overflow_bytes: 2 * OVER,
            }],
            "one report for the growth in place"
        );
        holder
            .try_hold_processing(OVER)
            .expect("the holder shrinks in place");
        assert_eq!(
            taken(),
            vec![GateCharge {
                reserved_bytes: in_budget,
                overflow_bytes: OVER,
            }],
            "one report for the shrink in place"
        );
    }

    /// Capture sink for [`capture_json`]: the tree-wide `CapturingWriter`
    /// idiom (`grpc::rpc_log::tests`).
    #[derive(Clone)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

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
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Run `f` to completion under a thread-local JSON capture subscriber and
    /// return every event it emitted, parsed. These tests run on a
    /// current-thread runtime and poll the futures under test in place, so
    /// every line they emit is emitted on this thread.
    async fn capture_json<F: Future<Output = ()>>(f: F) -> Vec<serde_json::Value> {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(Arc::clone(&buf)))
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            f.await;
        }
        let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
        String::from_utf8(raw)
            .expect("captured tracing output must be UTF-8")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect()
    }

    /// The captured "upload runs beyond the upload budget" lines.
    fn beyond_budget_lines(events: &[serde_json::Value]) -> Vec<&serde_json::Value> {
        events
            .iter()
            .filter(|e| e["fields"]["message"] == "upload runs beyond the upload budget")
            .collect()
    }

    /// A `try_hold` that takes the free slot logs one "upload runs beyond the
    /// upload budget" INFO line with the whole charge (`base` + target) and
    /// the call's session and layer; a further growth on the slot logs none.
    #[tokio::test]
    async fn upload_commit_charge_try_hold_into_the_slot_logs_one_line_with_the_whole_charge() {
        let gs = GracefulShutdown::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut r = received(&g, "r", gs.begun()).await;
        let base = r.reserved();
        let events = capture_json(async {
            let mut charge = UploadCommitCharge::new(&mut r, base, &gs, "s-try", "layer-try");
            assert!(charge.try_hold(OVER), "the free slot");
            assert!(charge.try_hold(OVER + 1), "growth on the slot");
        })
        .await;

        let lines = beyond_budget_lines(&events);
        assert_eq!(lines.len(), 1, "one line, for the entry only: {events:?}");
        let line = lines[0];
        assert_eq!(line["level"], "INFO");
        assert_eq!(line["fields"]["charge_bytes"], base + OVER);
        assert_eq!(line["fields"]["session_id"], "s-try");
        assert_eq!(line["fields"]["layer_name"], "layer-try");
    }

    /// A `hold` that waits for the slot and is granted it logs one "upload
    /// runs beyond the upload budget" INFO line with the whole charge (`base`
    /// + target).
    #[tokio::test]
    async fn upload_commit_charge_hold_into_the_slot_logs_one_line_with_the_whole_charge() {
        let gs = GracefulShutdown::new();
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", gs.begun()).await;
        process_now(&mut holder, OVER, gs.begun()).await;
        let mut r = received(&g, "r", gs.begun()).await;
        let base = r.reserved();
        let events = capture_json(async {
            let mut charge = UploadCommitCharge::new(&mut r, base, &gs, "s-hold", "layer-hold");
            let mut holding = Box::pin(charge.hold(OVER));
            assert!(poll_once(holding.as_mut()).await.is_pending());
            drop(holder);
            let Poll::Ready(Ok(())) = poll_once(holding.as_mut()).await else {
                panic!("the holder's drop must grant the waiting hold the slot");
            };
        })
        .await;

        let lines = beyond_budget_lines(&events);
        assert_eq!(lines.len(), 1, "one line for the entry: {events:?}");
        let line = lines[0];
        assert_eq!(line["level"], "INFO");
        assert_eq!(line["fields"]["charge_bytes"], base + OVER);
        assert_eq!(line["fields"]["session_id"], "s-hold");
        assert_eq!(line["fields"]["layer_name"], "layer-hold");
    }

    /// The upload charge's element count is the domain's
    /// [`own_element_count`]: struct commands + set fields + time samples,
    /// the count a plan's rewrite rows exclude.
    #[test]
    fn delta_footprint_elements_is_the_deltas_own_element_count() {
        let delta = oodf_ffi::OwnedDelta {
            struct_commands: vec![
                oodf_ffi::StructCommand::DeleteSection(
                    oodf_ffi::CommandDeleteSection {
                        parent_id: 0,
                        section_id: 1,
                    }
                );
                3
            ],
            set_fields: vec![oodf_ffi::SetField::default(); 5],
            time_samples: vec![oodf_ffi::TimeSample::default(); 7],
            ..Default::default()
        };
        let footprint = DeltaFootprint::of(Some(&delta));
        assert_eq!(footprint.elements, 15);
        assert_eq!(footprint.elements, own_element_count(&delta));
        assert_eq!(DeltaFootprint::of(None).elements, 0);
    }

    /// The object-state charges are each term times its constant, summed,
    /// the cached route's held entry and growth 1:1, and saturate instead of
    /// overflowing — every rewrite count included: rows, list entries, the
    /// name index's sections (`NI`), the adoption index's ids (`AE`) and the
    /// depth walk's sections (`WK`).
    #[test]
    fn object_state_charges_sum_their_terms_and_saturate() {
        let rewrite = RewriteCount {
            rows: 3,
            list_entries: 10,
            indexed_sections: 20,
            adoption_entries: 40,
            walked_sections: 80,
            rollback_rebuild_bytes: 0,
        };
        let rewrite_bytes = 3 * CACHED_REWRITE_ROW_CHARGE_BYTES
            + 10 * CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES
            + 20 * REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES
            + 40 * REWRITE_ADOPTION_ENTRY_CHARGE_BYTES
            + 80 * REWRITE_WALK_ENTRY_CHARGE_BYTES;
        let watch_rewrite_bytes = 3 * WATCH_REWRITE_ROW_CHARGE_BYTES
            + 10 * WATCH_REWRITE_LIST_ENTRY_CHARGE_BYTES
            + 20 * REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES
            + 40 * REWRITE_ADOPTION_ENTRY_CHARGE_BYTES
            + 80 * REWRITE_WALK_ENTRY_CHARGE_BYTES;
        assert_eq!(
            rewrite_charge(RewritePrices::CACHED, rewrite),
            rewrite_bytes
        );
        assert_eq!(
            rewrite_charge(RewritePrices::WATCH, rewrite),
            watch_rewrite_bytes
        );
        for prices in [RewritePrices::CACHED, RewritePrices::WATCH] {
            assert_eq!(rewrite_charge(prices, RewriteCount::default()), 0);
        }
        for (one, price, term) in [
            (
                RewriteCount {
                    indexed_sections: 1,
                    ..RewriteCount::default()
                },
                REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES,
                "NI",
            ),
            (
                RewriteCount {
                    adoption_entries: 1,
                    ..RewriteCount::default()
                },
                REWRITE_ADOPTION_ENTRY_CHARGE_BYTES,
                "AE",
            ),
            (
                RewriteCount {
                    walked_sections: 1,
                    ..RewriteCount::default()
                },
                REWRITE_WALK_ENTRY_CHARGE_BYTES,
                "WK",
            ),
        ] {
            for prices in [RewritePrices::CACHED, RewritePrices::WATCH] {
                assert_eq!(
                    rewrite_charge(prices, one),
                    price,
                    "one {term} unit alone, {prices:?}"
                );
            }
        }

        assert_eq!(
            cached_object_state_charge(1_000, 7, rewrite, RepairTransient::default()),
            1_000 + 7 + rewrite_bytes
        );
        assert_eq!(
            cached_object_state_charge(1, 0, RewriteCount::default(), RepairTransient::default()),
            1,
            "the held entry is charged 1:1"
        );
        assert_eq!(
            watch_object_state_charge(2, 5, 100, rewrite),
            2 * WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES
                + 5 * WATCH_SNAPSHOT_ITEM_CHARGE_BYTES
                + (100 * WATCH_SNAPSHOT_REPLY_BYTE_CHARGE_CENTI).div_ceil(EXPANSION_SCALE)
                + watch_rewrite_bytes
        );

        let huge = RewriteCount {
            rows: u64::MAX,
            list_entries: u64::MAX,
            indexed_sections: u64::MAX,
            adoption_entries: u64::MAX,
            walked_sections: u64::MAX,
            rollback_rebuild_bytes: u64::MAX,
        };
        for prices in [RewritePrices::CACHED, RewritePrices::WATCH] {
            assert_eq!(rewrite_charge(prices, huge), u64::MAX);
        }
        assert_eq!(
            cached_object_state_charge(u64::MAX, u64::MAX, huge, RepairTransient::default()),
            u64::MAX
        );
        assert_eq!(
            watch_object_state_charge(u64::MAX, u64::MAX, u64::MAX, huge),
            u64::MAX
        );
        // Each plan-only count saturates alone, the other counts at zero.
        for huge_one in [
            RewriteCount {
                indexed_sections: u64::MAX,
                ..RewriteCount::default()
            },
            RewriteCount {
                adoption_entries: u64::MAX,
                ..RewriteCount::default()
            },
            RewriteCount {
                walked_sections: u64::MAX,
                ..RewriteCount::default()
            },
        ] {
            for prices in [RewritePrices::CACHED, RewritePrices::WATCH] {
                assert_eq!(rewrite_charge(prices, huge_one), u64::MAX, "{huge_one:?}");
            }
            assert_eq!(
                cached_object_state_charge(1, 1, huge_one, RepairTransient::default()),
                u64::MAX,
                "{huge_one:?}"
            );
        }
    }

    /// A plan's rewrite allowance prices every count at its constant — the
    /// name index's and the adoption index's included — and is what is held
    /// beyond the need's target with no rewrite.
    #[test]
    fn object_state_rewrite_allowance_carries_every_rewrite_price() {
        let need = ObjectStateNeed::Cached {
            held: 10_000,
            growth: 10_000,
            rewrite: RewriteCount {
                rows: 5,
                ..RewriteCount::default()
            },
            repair: RepairTransient::default(),
        };
        let without_rewrite = cached_object_state_charge(
            10_000,
            10_000,
            RewriteCount::default(),
            RepairTransient::default(),
        );
        let allowance = object_state_rewrite_allowance(without_rewrite + 777, &need);
        assert_eq!(
            allowance,
            MutationAllowance {
                bytes: 777,
                row_bytes: CACHED_REWRITE_ROW_CHARGE_BYTES,
                list_entry_bytes: CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES,
                index_entry_bytes: REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES,
                adoption_entry_bytes: REWRITE_ADOPTION_ENTRY_CHARGE_BYTES,
                walk_entry_bytes: REWRITE_WALK_ENTRY_CHARGE_BYTES,
            }
        );
        assert_eq!(
            object_state_rewrite_allowance(without_rewrite - 1, &need).bytes,
            0,
            "held below the no-rewrite target allows nothing"
        );
    }

    /// The cached route's stream-repair transient (RP0, RP1) is priced on top
    /// of the rest of its need: the reply at `repair_reply_byte` (and
    /// `repair_reply_entry` per entry, 0 here), the largest
    /// record's delta part at `k_parse` and the replayed delta parts at
    /// `repair_growth`, each in hundredths rounded up, saturating. A zero
    /// transient prices nothing, so the needs that carry none (`cached`,
    /// `cached_growing`) price as before.
    #[test]
    fn repair_transient_is_priced_on_top_of_the_cached_need_and_saturates() {
        let repair = RepairTransient {
            reply_entry_count: 0,
            reply_bytes: 1_001,
            parse_delta_bytes: 333,
            replay_delta_bytes: 77,
        };
        let transient = (1_001 * REPAIR_REPLY_BYTE_CENTI).div_ceil(EXPANSION_SCALE)
            + (333 * PARSE_EXPANSION_CENTI).div_ceil(EXPANSION_SCALE)
            + (77 * REPAIR_GROWTH_CENTI).div_ceil(EXPANSION_SCALE);
        assert_eq!(repair_transient_charge(repair), transient);
        assert_eq!(repair_transient_charge(RepairTransient::default()), 0);
        for (one, centi, part) in [
            (
                RepairTransient {
                    reply_bytes: 1,
                    ..RepairTransient::default()
                },
                REPAIR_REPLY_BYTE_CENTI,
                "reply",
            ),
            (
                RepairTransient {
                    parse_delta_bytes: 1,
                    ..RepairTransient::default()
                },
                PARSE_EXPANSION_CENTI,
                "parse",
            ),
            (
                RepairTransient {
                    replay_delta_bytes: 1,
                    ..RepairTransient::default()
                },
                REPAIR_GROWTH_CENTI,
                "growth",
            ),
        ] {
            assert_eq!(
                repair_transient_charge(one),
                centi.div_ceil(EXPANSION_SCALE),
                "one byte of {part} alone, rounded up"
            );
            let huge = RepairTransient {
                reply_entry_count: 0,
                reply_bytes: one.reply_bytes * u64::MAX,
                parse_delta_bytes: one.parse_delta_bytes * u64::MAX,
                replay_delta_bytes: one.replay_delta_bytes * u64::MAX,
            };
            assert_eq!(repair_transient_charge(huge), u64::MAX, "{part} saturates");
            assert_eq!(
                cached_object_state_charge(1, 1, RewriteCount::default(), huge),
                u64::MAX,
                "{part} saturates the cached need"
            );
        }

        let rewrite = RewriteCount {
            rows: 3,
            ..RewriteCount::default()
        };
        assert_eq!(
            cached_object_state_charge(1_000, 7, rewrite, repair),
            cached_object_state_charge(1_000, 7, rewrite, RepairTransient::default()) + transient,
            "the transient is on top of the entry, its growth and the rewrite"
        );
        assert_eq!(
            object_state_charge(&ObjectStateNeed::cached_repair(1_000, repair)),
            object_state_charge(&ObjectStateNeed::cached(1_000, RewriteCount::default()))
                + transient,
            "a repair charge point holds the entry, as the hint does, and the transient"
        );
    }

    /// The first repair read's bounds — its entry bound
    /// (`STRUCTURAL_REPAIR_INITIAL_ENTRIES`) and byte bound
    /// (`STRUCTURAL_REPAIR_INITIAL_BYTES`) — are the step the first-read rule
    /// reaches at the wired reply constants, so a refit of either reply term
    /// that moves the step fails here until the bounds follow it. The rule
    /// prices a read of `E` entries and `B` bytes as RP0 does,
    /// `P(E, B) = a × E + b × B`, against 1 MiB:
    ///
    /// 1. `E` is the read's full entry bound (256) when `P(256, 256 KiB)` fits;
    /// 2. else the largest `E` in `[64, 256]` with `P(E, 256 KiB)` within it;
    /// 3. else, when even `P(64, 256 KiB)` exceeds it, the byte bound becomes
    ///    128 KiB and `E` is chosen by steps 1 and 2 at 128 KiB.
    ///
    /// At the fitted `a` = 864 B and `b` = 370 centi the step is 2: 91
    /// entries at 256 KiB, priced 1,048,557 B, the figure the fit record
    /// states, which is pinned beside the computed step. The same pin ties
    /// `REPAIR_GROWTH_CENTI` (125) to its fit in that record: it is the one
    /// place a test states MR-A's fitted values as literals; every other test
    /// reads them by name.
    #[test]
    fn first_read_bound_is_the_rules_step() {
        use crate::storage::structural_cache_sync::{
            STRUCTURAL_REPAIR_INITIAL_BYTES, STRUCTURAL_REPAIR_INITIAL_ENTRIES,
            STRUCTURAL_REPAIR_MAX_ENTRIES,
        };
        const LIMIT: u64 = 1024 * 1024;
        const MIN_ENTRIES: u64 = 64;
        let price = |entries: u64, bytes: u64| {
            repair_transient_charge(RepairTransient {
                reply_entry_count: entries,
                reply_bytes: bytes,
                ..RepairTransient::default()
            })
        };
        // Steps 1 and 2 at one byte bound: the largest entry bound in
        // [64, 256] whose read fits, the full 256 first.
        let entries_at = |bytes: u64| {
            (MIN_ENTRIES..=STRUCTURAL_REPAIR_MAX_ENTRIES)
                .rev()
                .find(|&entries| price(entries, bytes) <= LIMIT)
        };
        let (entries, bytes) = match entries_at(256 * 1024) {
            Some(entries) => (entries, 256 * 1024),
            None => (
                entries_at(128 * 1024).expect(
                    "the rule has no step past 128 KiB: at 128 KiB even 64 entries price over \
                     1 MiB",
                ),
                128 * 1024,
            ),
        };
        assert_eq!(
            (
                STRUCTURAL_REPAIR_INITIAL_ENTRIES,
                STRUCTURAL_REPAIR_INITIAL_BYTES
            ),
            (entries, bytes),
            "the first read's (entries, bytes) bounds are the rule's step at a = \
             {REPAIR_REPLY_ENTRY_BYTES} B and b = {REPAIR_REPLY_BYTE_CENTI} centi"
        );
        assert!(
            price(
                STRUCTURAL_REPAIR_INITIAL_ENTRIES,
                STRUCTURAL_REPAIR_INITIAL_BYTES
            ) <= LIMIT,
            "the first read's reply prices within 1 MiB"
        );
        // The one literal pin of MR-A's wired repair constants, each as the
        // fit record states it
        // (`docs-internal/validation/repair-charge-t7-b-fsz-20261010/fits.json`,
        // each constant's `wired` field): `a` and `b` with the first read's
        // step and its RP0 reply price, and `REPAIR_GROWTH_CENTI`. The record
        // also holds the latest refit's `fitted` figures; the wired values
        // move only by a refit that converges, and this tuple and that record
        // move together. The record is read by hand, not with
        // `include_str!`: `docs-internal/` is not in the published source tree,
        // so a build reading it there would fail.
        assert_eq!(
            (
                REPAIR_REPLY_ENTRY_BYTES,
                REPAIR_REPLY_BYTE_CENTI,
                entries,
                bytes,
                price(entries, bytes),
                REPAIR_GROWTH_CENTI
            ),
            (864, 370, 91, 262_144, 1_048_557, 125),
            "the first read's step, its RP0 reply price and the per-record growth factor as \
             the fit record states them"
        );
    }

    /// W2-T1: each route's need is priced at its own route's rewrite set,
    /// through the one choke point (`object_state_charge`) and the plan's
    /// allowance alike: a `Cached` need's rows and list entries at the cached
    /// route's prices, a `Watch` need's at the WATCH path's — the values the
    /// WATCH snapshot factors were netted of. Fails if the WATCH path is priced
    /// at the cached row or list entry, or the cached route at the WATCH ones.
    #[test]
    fn object_state_charge_prices_each_route_at_its_own_rewrite_set() {
        use crate::storage::commit_charge::{QueuedCommands, WatchSnapshot};
        const _: () = assert!(CACHED_REWRITE_ROW_CHARGE_BYTES != WATCH_REWRITE_ROW_CHARGE_BYTES);
        const _: () = assert!(
            CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES != WATCH_REWRITE_LIST_ENTRY_CHARGE_BYTES
        );
        let row = RewriteCount {
            rows: 1,
            ..RewriteCount::default()
        };
        let list_entry = RewriteCount {
            list_entries: 1,
            ..RewriteCount::default()
        };
        let cached = |rewrite| ObjectStateNeed::cached_growing(0, 0, rewrite);
        let watch = |rewrite| ObjectStateNeed::Watch {
            snapshot: WatchSnapshot::default(),
            fixed: false,
            rewrite,
            queued: QueuedCommands::default(),
        };
        for (route, need, row_price, list_price) in [
            (
                "cached",
                &cached as &dyn Fn(RewriteCount) -> ObjectStateNeed,
                CACHED_REWRITE_ROW_CHARGE_BYTES,
                CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES,
            ),
            (
                "watch",
                &watch,
                WATCH_REWRITE_ROW_CHARGE_BYTES,
                WATCH_REWRITE_LIST_ENTRY_CHARGE_BYTES,
            ),
        ] {
            let base = object_state_charge(&need(RewriteCount::default()));
            assert_eq!(
                object_state_charge(&need(row)) - base,
                row_price,
                "{route}: one row at the route's rewrite_row"
            );
            assert_eq!(
                object_state_charge(&need(list_entry)) - base,
                list_price,
                "{route}: one list entry at the route's rewrite_list"
            );
            let allowance = object_state_rewrite_allowance(base + 777, &need(row));
            assert_eq!(
                (
                    allowance.bytes,
                    allowance.row_bytes,
                    allowance.list_entry_bytes
                ),
                (777, row_price, list_price),
                "{route}: the allowance carries the route's prices"
            );
        }
    }

    /// The plan-scratch prices are at least their largest measured basis ×
    /// 1.25: `NI` ≥ ⌈125.7 × 1.25⌉ (hashbrown's 41 B per bucket at up to
    /// 2 × 8/7 buckets per entry, plus a 32 B one-id `Vec`, measured exactly
    /// at the worst bucket ratio) and `AE` ≥ ⌈26.22 × 1.25⌉ = 33 (the largest
    /// cross-run figure measured for an adoption-index id, above the 24 B a
    /// doubling `Vec` of 8 B ids holds while a push reallocates); on the
    /// cached route planning in place with its journal alive through the
    /// commit, `rewrite_list` ≥ ⌈229.6 × 1.25⌉ = 287 (one new child on 1,000
    /// children, the commit's segment), `rewrite_row` ≥ ⌈3,658.5 × 1.25⌉ =
    /// 4,574 (the 20,000-child re-key netted against one new child) and
    /// `rewrite_walk_entry` ≥ ⌈99.2 × 1.25⌉ = 124 (the incremental depth
    /// check's walk up a 9,000-deep chain); on the WATCH path, `rewrite_row`
    /// at 4,376, the value its snapshot factors were netted of (above
    /// ⌈793.6 × 1.25⌉), and `rewrite_list` ≥ ⌈199.25 × 1.25⌉ = 250 (its own
    /// commit's list segment); `element[UpdateSlow]` ≥ ⌈2,543.4 × 1.25⌉ =
    /// 3,180 (20,000 new children from a cold start, net of the cached
    /// route's object-state need at the commit's peak); and
    /// `element[UpdateFast]` ≥ ⌈1,451.3 × 1.25⌉ = 1,815 (the field-dense
    /// fixture). And the couplings the constants' docs state hold: the WATCH
    /// `rewrite_row` below 7,396 B and `element[UpdateSlow]` below 10,549 B
    /// each need the queued-command term at 1,896 or above, and
    /// `element[UpdateSlow]` below 3,690 needs the cached rewrite prices it
    /// was netted of. Checked at compile time, so lowering one term alone
    /// fails the test build.
    #[test]
    fn plan_scratch_prices_cover_their_measured_bases_and_couplings() {
        const NI_MODEL_TENTHS: u64 = 1_257;
        const AE_BASIS_HUNDREDTHS: u64 = 2_622;
        const _: () = assert!((NI_MODEL_TENTHS * 125).div_ceil(1_000) == 158);
        const _: () = assert!(
            REWRITE_NAME_INDEX_ENTRY_CHARGE_BYTES >= (NI_MODEL_TENTHS * 125).div_ceil(1_000)
        );
        const _: () = assert!((AE_BASIS_HUNDREDTHS * 125).div_ceil(10_000) == 33);
        const _: () = assert!(
            REWRITE_ADOPTION_ENTRY_CHARGE_BYTES >= (AE_BASIS_HUNDREDTHS * 125).div_ceil(10_000)
        );
        // The cached route's in-place bases, in tenths of a byte (the fit
        // record
        // `docs-internal/validation/upload-commit-charge-inplace-b-ug2-20261010/fits.md`).
        const LIST_BASIS_TENTHS: u64 = 2_296;
        const ROW_BASIS_TENTHS: u64 = 36_585;
        const WALK_BASIS_TENTHS: u64 = 992;
        const _: () = assert!((LIST_BASIS_TENTHS * 125).div_ceil(1_000) == 287);
        const _: () = assert!((ROW_BASIS_TENTHS * 125).div_ceil(1_000) == 4_574);
        const _: () = assert!((WALK_BASIS_TENTHS * 125).div_ceil(1_000) == 124);
        const _: () = assert!(
            CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES >= (LIST_BASIS_TENTHS * 125).div_ceil(1_000)
        );
        const _: () =
            assert!(CACHED_REWRITE_ROW_CHARGE_BYTES >= (ROW_BASIS_TENTHS * 125).div_ceil(1_000));
        const _: () =
            assert!(REWRITE_WALK_ENTRY_CHARGE_BYTES >= (WALK_BASIS_TENTHS * 125).div_ceil(1_000));
        // The WATCH path's set: its row the value the WATCH snapshot
        // factors were netted of, above its own figure (793.6 B net of
        // `queued`); its list entry its own figure's rule (199.25 B, in
        // hundredths, rounded up from 199.243).
        const WATCH_ROW_BASIS_TENTHS: u64 = 7_936;
        const WATCH_LIST_BASIS_HUNDREDTHS: u64 = 19_925;
        const _: () = assert!(WATCH_REWRITE_ROW_CHARGE_BYTES == 4_376);
        const _: () = assert!(
            WATCH_REWRITE_ROW_CHARGE_BYTES >= (WATCH_ROW_BASIS_TENTHS * 125).div_ceil(1_000)
        );
        const _: () = assert!((WATCH_LIST_BASIS_HUNDREDTHS * 125).div_ceil(10_000) == 250);
        const _: () = assert!(
            WATCH_REWRITE_LIST_ENTRY_CHARGE_BYTES
                >= (WATCH_LIST_BASIS_HUNDREDTHS * 125).div_ceil(10_000)
        );
        // `element[UpdateSlow]`'s largest netted figure over its routes: the
        // cached route's cold start, 20,000 new children, net of its
        // object-state need at the commit's peak at the production bases.
        // The netting holds only with the cached rewrite set it was netted
        // of (287 / 4,574 / 124) and the in-place growth term wired, so that
        // coupling is pinned beside it. Kept at 3,476 by raise-only.
        const ELEMENT_BASIS_TENTHS: u64 = 25_434;
        const _: () = assert!((ELEMENT_BASIS_TENTHS * 125).div_ceil(1_000) == 3_180);
        const _: () = assert!(
            ELEMENT_CHARGE_UPDATE_SLOW_BYTES >= (ELEMENT_BASIS_TENTHS * 125).div_ceil(1_000)
        );
        const _: () = assert!(
            ELEMENT_CHARGE_UPDATE_SLOW_BYTES >= 3_690
                || (CACHED_REWRITE_LIST_ENTRY_CHARGE_BYTES >= 287
                    && CACHED_REWRITE_ROW_CHARGE_BYTES >= 4_574
                    && REWRITE_WALK_ENTRY_CHARGE_BYTES >= 124)
        );
        const _: () = assert!(
            WATCH_REWRITE_ROW_CHARGE_BYTES >= 7_396 || QUEUED_COMMAND_CHARGE_BYTES >= 1_896
        );
        const _: () = assert!(
            ELEMENT_CHARGE_UPDATE_SLOW_BYTES >= 10_549 || QUEUED_COMMAND_CHARGE_BYTES >= 1_896
        );
        // `element[UpdateFast]`: the field-dense fixture at 20,000 elements,
        // 1,451.3 B (1,408.4 B at 30,000).
        const FAST_ELEMENT_BASIS_TENTHS: u64 = 14_513;
        const _: () = assert!((FAST_ELEMENT_BASIS_TENTHS * 125).div_ceil(1_000) == 1_815);
        const _: () = assert!(
            ELEMENT_CHARGE_UPDATE_FAST_BYTES >= (FAST_ELEMENT_BASIS_TENTHS * 125).div_ceil(1_000)
        );
    }

    // ── Refusal statuses ────────────────────────────────────────────────────

    /// Every refusal answers `UNAVAILABLE` carrying a retry hint whose
    /// `ErrorInfo` tells the client this call applied nothing (the re-send
    /// contract is `crate::error::REFUSAL_DETAIL_DOMAIN`'s).
    /// None is for the upload's size. A capacity refusal is `UPLOAD_CAPACITY`
    /// with metadata `cause` naming what was full and a `[500 ms, 1500 ms)`
    /// delay; a shutdown refusal is the shutdown status's message and detail.
    #[test]
    fn refusal_into_service_error_is_unavailable_with_its_retry_hint() {
        use crate::error::SHUTDOWN_RETRY_DELAY;
        for (refusal, reason, cause, range) in [
            (
                UploadRefusal::Saturated,
                refusal_reason::UPLOAD_CAPACITY,
                Some(refusal_metadata::SATURATED),
                CAPACITY_RETRY_DELAY,
            ),
            (
                UploadRefusal::WaiterCap,
                refusal_reason::UPLOAD_CAPACITY,
                Some(refusal_metadata::WAITER_CAP),
                CAPACITY_RETRY_DELAY,
            ),
            (
                UploadRefusal::Growth,
                refusal_reason::UPLOAD_CAPACITY,
                Some(refusal_metadata::GROWTH),
                CAPACITY_RETRY_DELAY,
            ),
            (
                UploadRefusal::ShuttingDown,
                refusal_reason::SERVER_SHUTTING_DOWN,
                None,
                SHUTDOWN_RETRY_DELAY,
            ),
        ] {
            let (message, detail) = match refusal.into_service_error() {
                ServiceError::UnavailableWithDetail { message, detail } => (message, detail),
                other => panic!("{refusal:?} must carry a retry hint; got {other:?}"),
            };
            let metadata: Vec<(&str, String)> = cause
                .map(|c| vec![(refusal_metadata::CAUSE, c.to_owned())])
                .unwrap_or_default();
            assert_eq!(
                *detail,
                RefusalDetail {
                    reason,
                    metadata,
                    retry_delay: Some(range),
                },
                "{refusal:?}"
            );
            match refusal {
                UploadRefusal::ShuttingDown => assert_eq!(message, "server shutting down"),
                _ => assert!(
                    message.contains("nothing was applied, retry"),
                    "{refusal:?}: {message}"
                ),
            }
            let status =
                tonic::Status::from(ServiceError::UnavailableWithDetail { message, detail });
            assert_eq!(status.code(), tonic::Code::Unavailable, "{refusal:?}");
        }
    }

    // ── Charges and the default arithmetic (T8) ─────────────────────────────

    /// The charges round each per-byte term up, take the larger of the parse
    /// and commit terms, and saturate instead of overflowing.
    #[test]
    fn charges_round_up_take_the_larger_term_and_saturate() {
        assert_eq!(
            parse_charge(0, 1, 0),
            FIXED_CHARGE_BYTES + PARSE_EXPANSION_CENTI.div_ceil(EXPANSION_SCALE)
        );
        assert_eq!(
            parse_charge(0, 0, 1),
            FIXED_CHARGE_BYTES + EXT_EXPANSION_CENTI.div_ceil(EXPANSION_SCALE)
        );
        let one_element = DeltaFootprint {
            elements: 1,
            reorder_entries: 0,
        };
        for path in [
            CommitPath::Create,
            CommitPath::UpdateFast,
            CommitPath::UpdateSlow,
        ] {
            assert_eq!(
                processing_charge(path, 0, 0, 0, one_element),
                FIXED_CHARGE_BYTES + path.element_charge_bytes(),
                "{path:?}"
            );
            assert!(
                processing_charge(path, 0, 1_000, 0, DeltaFootprint::default())
                    >= parse_charge(0, 1_000, 0),
                "{path:?}: never below the parse charge"
            );
            assert_eq!(
                processing_charge(
                    path,
                    u64::MAX,
                    u64::MAX,
                    u64::MAX,
                    DeltaFootprint {
                        elements: u64::MAX,
                        reorder_entries: u64::MAX
                    }
                ),
                u64::MAX
            );
        }
    }

    /// The canonical single-field blob upload, rebuilt here independently of
    /// `canonical_blob_upload_lengths`: one `SetField` with an empty key name
    /// referencing external value 1 of `blob_len` bytes. Returns its delta
    /// bytes; the body is the OMPF framing of two parts (magic, part count and
    /// one offset per part: 4 + 4 + 2 × 8 = 24 bytes) + delta + blob.
    fn canonical_blob_delta(blob_len: u64) -> Vec<u8> {
        oodf_ffi::build_delta(&oodf_ffi::OwnedDelta {
            set_fields: vec![oodf_ffi::SetField {
                section_id: 0,
                key_name: String::new(),
                value_or_ext_hash: vec![0xAB; 32],
                ext_value_size: blob_len,
                ext_value_index: 1,
                set_order: 0,
            }],
            ..Default::default()
        })
        .expect("the canonical delta builds")
    }

    /// The OMPF framing of a delta and one external value, by the wire format.
    const TWO_PART_OMPF_FRAMING: u64 = 4 + 4 + 2 * 8;

    /// The object-state term the canonical upload's WATCH commit holds on
    /// `UpdateSlow` after its apply (WA3), derived by hand from the commit's
    /// shape rather than by running the engine. The delta is one `SetField` on
    /// a header-only object, fast-shaped, so the commit holds no fixed cost and
    /// no read, only:
    ///
    /// - its rewrite: the plan emits the `SetField` and the header, one row
    ///   beyond the delta's one element — one `rewrite_row`, at the WATCH
    ///   path's price ([`WATCH_REWRITE_ROW_CHARGE_BYTES`]);
    /// - its queued commands, by next power of two per connection:
    ///   - the transaction: `SetField`'s 3 + the header's 1, two external-index
    ///     writes for its one value row, the six frame commands (`MULTI`, the
    ///     commit-token record, `XADD`, `XTRIM`, `CLIENT ID`, `EXEC`), and,
    ///     because it writes a value row, the commit release ring's `HDEL`,
    ///     `HSET` and `PEXPIRE`: 15, so 16 slots;
    ///   - the pre-image read: two commands for its one value row, 2 slots
    ///     (it reads no whole `obj:ext` index);
    ///   - the keyspace probe: one key per queued mutation command (4) plus the
    ///     five commit-level keys (`obj:ext`, the commit-token record, the
    ///     commit release ring, the snapshot fence and the writer-intent key),
    ///     plus the ring's window `HMGET` that runs beside it for the one value
    ///     row: 10, so 16 slots.
    const CANONICAL_BLOB_WATCH_COMMIT_TERM: u64 =
        WATCH_REWRITE_ROW_CHARGE_BYTES + QUEUED_COMMAND_CHARGE_BYTES * (16 + 2 + 16);

    /// The engine-run WATCH commit term of the canonical upload
    /// (`canonical_blob_watch_commit_term`) is the hand derivation's, whatever
    /// the blob's size: an external value's bytes never reach the commit's
    /// rows or queue.
    #[test]
    fn canonical_blob_watch_commit_term_is_one_row_and_its_queued_slots() {
        for blob_len in [1u64, 4_096, 1_000_003, 100 * MIB] {
            assert_eq!(
                canonical_blob_watch_commit_term(blob_len),
                Some(CANONICAL_BLOB_WATCH_COMMIT_TERM),
                "blob {blob_len}"
            );
        }
    }

    /// Whether the canonical upload of a `blob_len`-byte blob runs within
    /// `budget` on every path — receive phase, parse charge, and the
    /// processing charge of each commit path, on `UpdateSlow` with its WATCH
    /// commit's object-state term on top — recomputed from
    /// [`canonical_blob_delta`] and [`CANONICAL_BLOB_WATCH_COMMIT_TERM`].
    fn canonical_upload_fits_independently(budget: u64, blob_len: u64) -> bool {
        let delta_len = canonical_blob_delta(blob_len).len() as u64;
        let body = TWO_PART_OMPF_FRAMING + delta_len + blob_len;
        let one_element = DeltaFootprint {
            elements: 1,
            reorder_entries: 0,
        };
        admission_charge() + RECEIVE_CHARGE_FACTOR * body <= budget
            && parse_charge(body, delta_len, blob_len) <= budget
            && [
                CommitPath::Create,
                CommitPath::UpdateFast,
                CommitPath::UpdateSlow,
            ]
            .into_iter()
            .all(|path| {
                let charge = processing_charge(path, body, delta_len, blob_len, one_element);
                let object_state = match path {
                    CommitPath::UpdateSlow => CANONICAL_BLOB_WATCH_COMMIT_TERM,
                    CommitPath::Create | CommitPath::UpdateFast => 0,
                };
                charge + object_state <= budget
            })
    }

    /// The independent recomputation's framing is the real container's: a
    /// canonical upload actually serialized, around a real blob, is framing +
    /// delta + blob long — and `canonical_blob_upload_lengths` agrees.
    #[test]
    fn canonical_blob_upload_lengths_match_a_real_serialized_upload() {
        for blob_len in [1u64, 4_096, 1_000_003] {
            let delta = canonical_blob_delta(blob_len);
            let blob = vec![0x5A; blob_len as usize];
            let body = crate::domain::ompf::OmpfContainer::serialize(&delta, &[&blob])
                .expect("serializes")
                .len() as u64;
            assert_eq!(
                body,
                TWO_PART_OMPF_FRAMING + delta.len() as u64 + blob_len,
                "blob {blob_len}"
            );
            assert_eq!(
                canonical_blob_upload_lengths(blob_len),
                Some((delta.len() as u64, body)),
                "blob {blob_len}"
            );
        }
    }

    /// `largest_in_budget_blob_upload` is exactly the largest blob whose
    /// canonical single-field upload fits within the budget on every path
    /// (receive phase, parse charge, each path's processing charge, and on
    /// `UpdateSlow` its WATCH commit's object-state term): that blob fits, and
    /// one byte more does not — checked by an independent recomputation, at
    /// several budgets including the default.
    #[test]
    fn largest_in_budget_blob_upload_is_the_largest_that_fits() {
        for budget in [
            SMALL_BUDGET,
            100 * MIB,
            UPLOAD_MEMORY_BUDGET_MIB_DEFAULT * MIB,
            1_024 * MIB,
        ] {
            let blob = largest_in_budget_blob_upload(budget);
            assert!(blob > 0, "budget {budget}: some blob fits");
            assert!(
                canonical_upload_fits_independently(budget, blob),
                "budget {budget}: a {blob}-byte blob's canonical upload must fit"
            );
            assert!(
                !canonical_upload_fits_independently(budget, blob + 1),
                "budget {budget}: a {}-byte blob's canonical upload fits too, so {blob} is not \
                 the largest",
                blob + 1
            );
        }
        assert_eq!(
            largest_in_budget_blob_upload(admission_charge() - 1),
            0,
            "a budget below one admission fits no upload"
        );
    }

    /// T8: at the defaults (`B` = 384 MiB, a 100 MiB size cap) the largest
    /// legal single-blob upload — the canonical single-field upload whose body
    /// is exactly the 100 MiB size cap — is admitted on every commit path and
    /// runs in the overflow slot, never refused for its size.
    ///
    /// The default budget clears its floor and the upload's receive phase fits
    /// it, but its processing charge is over `B`. So, driven through a gate at
    /// the default configuration the way the handler drives it, its
    /// processing step takes the overflow slot at once and ticks
    /// `overflow_runs_total{rpc}`. With the rest of the budget then filled by
    /// other uploads, a second upload over `B` queues for the slot instead of
    /// becoming a second holder: total upload memory stays within `B` plus
    /// this one slot charge. The round-1 gate refused this upload
    /// `INVALID_ARGUMENT`.
    #[tokio::test]
    async fn default_budget_runs_a_size_cap_single_blob_upload_in_the_overflow_slot_on_every_path()
    {
        const SIZE_CAP: u64 = 100 * MIB;
        let config = UploadAdmissionConfig::default();
        let budget = config.budget_bytes;
        assert_eq!(budget, UPLOAD_MEMORY_BUDGET_MIB_DEFAULT * MIB);
        assert_eq!(config.max_upload_bytes, SIZE_CAP);
        assert!(budget >= upload_budget_floor_bytes(SIZE_CAP));
        // The blob that makes the canonical upload's body exactly the size
        // cap: the cap less the delta and framing a one-byte blob adds.
        let (_, one_byte_body) = canonical_blob_upload_lengths(1).expect("encodes");
        let blob_len = SIZE_CAP - (one_byte_body - 1);
        let (delta_len, body) = canonical_blob_upload_lengths(blob_len).expect("encodes");
        assert_eq!(
            body, SIZE_CAP,
            "fixture self-check: the body is exactly the size cap, the largest the service accepts"
        );
        let receive_peak = admission_charge() + RECEIVE_CHARGE_FACTOR * body;
        assert!(
            receive_peak <= budget,
            "the receive phase of a size-cap blob upload must fit the default budget"
        );
        assert!(
            largest_in_budget_blob_upload(budget) < blob_len,
            "a size-cap blob runs beyond the default budget, in the overflow slot"
        );
        let footprint = DeltaFootprint {
            elements: 1,
            reorder_entries: 0,
        };
        let token = CancellationToken::new();
        for path in [
            CommitPath::Create,
            CommitPath::UpdateFast,
            CommitPath::UpdateSlow,
        ] {
            let rpc = match path {
                CommitPath::Create => UploadRpc::CreateLayer,
                CommitPath::UpdateFast | CommitPath::UpdateSlow => UploadRpc::UpdateLayer,
            };
            let parse = parse_charge(body, delta_len, blob_len);
            let charge = processing_charge(path, body, delta_len, blob_len, footprint);
            assert!(
                charge > budget,
                "{path:?}: fixture self-check: the upload charges {charge} bytes, within the \
                 default budget of {budget}, so it would not need the overflow slot"
            );
            let metrics = Arc::new(metrics_scrape_only("upload-admission-t8"));
            let g = Arc::new(UploadAdmissionGate::new(config, Some(Arc::clone(&metrics))));
            let admit = |session: &'static str| {
                let g = Arc::clone(&g);
                let token = token.clone();
                async move {
                    let mut fut = std::pin::pin!(g.admit(rpc, session, &token));
                    match poll_once(fut.as_mut()).await {
                        Poll::Ready(Ok(reservation)) => reservation,
                        other => panic!("{session:?} must be admitted at once; got {other:?}"),
                    }
                }
            };

            // Receive phase, as `streaming::read_upload_body` books it.
            let mut blob = admit("blob").await;
            assert_eq!(
                blob.try_grow_in_budget(receive_peak),
                Ok(()),
                "{path:?}: the receive phase fits the default budget"
            );
            blob.end_receive();
            blob.shrink_to(FIXED_CHARGE_BYTES + body);
            let mut next = admit("next").await;
            next.end_receive();

            // Both processing steps, as the handler takes them.
            for (step, target) in [("parse", parse), ("processing", charge)] {
                let mut fut = std::pin::pin!(blob.reserve_processing(target, &token));
                match poll_once(fut.as_mut()).await {
                    Poll::Ready(Ok(_)) => {}
                    other => panic!(
                        "{path:?}: the {step} step of a size-cap single-blob upload must be \
                         granted at once in the overflow slot, not refused; got {other:?}"
                    ),
                }
            }
            assert!(blob.in_overflow(), "{path:?}: the upload holds the slot");
            assert_eq!(blob.reserved(), charge, "{path:?}");
            assert_eq!(g.overflow_bytes(), charge, "{path:?}");
            assert_eq!(
                metric(
                    &metrics,
                    "live_edit_upload_overflow_runs_total",
                    &[("rpc", rpc.label())]
                ),
                1.0,
                "{path:?}: one overflow run"
            );

            // Fill the rest of the budget; a second over-budget upload queues.
            let mut filler = admit("filler").await;
            let room = budget - g.reserved_bytes();
            filler
                .try_grow_in_budget(filler.reserved() + room)
                .expect("room for the filler");
            assert_eq!(g.reserved_bytes(), budget, "{path:?}: the budget is full");
            let mut waiting = Box::pin(next.reserve_processing(charge, &token));
            assert!(
                poll_once(waiting.as_mut()).await.is_pending(),
                "{path:?}: a second upload over the budget waits for the slot"
            );
            assert_eq!(g.processing_waiters(), 1, "{path:?}");
            assert_eq!(
                g.overflow_bytes(),
                charge,
                "{path:?}: never a second slot holder"
            );
            assert!(
                g.reserved_bytes() + g.overflow_bytes() <= budget + charge,
                "{path:?}: total upload memory within B plus one slot charge"
            );
            assert!(g.high_water_bytes() <= budget, "{path:?}");
            drop(waiting);
        }
    }

    /// The per-session cap is `max(1, ⌊B ÷ 4A⌋)`: 12 at the default budget,
    /// never below one.
    #[test]
    fn per_session_cap_is_a_quarter_of_the_receive_allowances_and_at_least_one() {
        assert_eq!(
            per_session_cap_for(UPLOAD_MEMORY_BUDGET_MIB_DEFAULT * MIB),
            12
        );
        assert_eq!(per_session_cap_for(4 * RECEIVE_ALLOWANCE_BYTES - 1), 1);
        assert_eq!(per_session_cap_for(0), 1);
        assert_eq!(per_session_cap_for(8 * RECEIVE_ALLOWANCE_BYTES), 2);
    }

    /// The unset admission wait is `min(2000, deadline / 2)`.
    #[test]
    fn upload_admission_wait_default_is_half_the_deadline_capped_at_2000() {
        for (deadline, want) in [
            (30_000, 2_000),
            (4_000, 2_000),
            (3_999, 1_999),
            (1_000, 500),
            (3, 1),
            (1, 0),
            (0, 0),
        ] {
            assert_eq!(
                upload_admission_wait_ms_default(deadline),
                want,
                "deadline {deadline}"
            );
        }
    }

    // ── The sampled audit's charge (UploadBudgetAuditCharges, b.fsz PD2) ────

    /// One audit's charge on `gate`, its copy weighing `base`.
    fn audit_charge(gate: &Arc<UploadAdmissionGate>, base: u64) -> Box<dyn AuditCharge> {
        UploadBudgetAuditCharges::new(Arc::clone(gate)).open(base)
    }

    /// The per-audit cap is one upload admission (`F + A`), a constant: the
    /// figure the docs state, and what every audit charge answers.
    #[test]
    fn sampled_audit_charge_cap_is_one_upload_admission() {
        assert_eq!(SAMPLED_AUDIT_CHARGE_CAP_BYTES, admission_charge());
        assert_eq!(
            SAMPLED_AUDIT_CHARGE_CAP_BYTES, 8_478_808,
            "the cap the docs state"
        );
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        assert_eq!(audit_charge(&g, 0).cap(), SAMPLED_AUDIT_CHARGE_CAP_BYTES);
    }

    /// A target is the need's price less the copy's own weight (which the
    /// line's audit term funds), never below 0, and the rewrite allowance is
    /// read on what is held plus that weight, as a structural call's charge
    /// reads it on what it holds.
    #[test]
    fn audit_charge_target_is_the_needs_price_less_the_copy_weight() {
        const BASE: u64 = 1_000_000;
        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let c = audit_charge(&g, BASE);
        let need = ObjectStateNeed::cached_repair(
            BASE,
            RepairTransient {
                reply_entry_count: 3,
                reply_bytes: 4_096,
                parse_delta_bytes: 512,
                replay_delta_bytes: 512,
            },
        );

        assert_eq!(c.target(&need), object_state_charge(&need) - BASE);
        assert_eq!(
            c.target(&ObjectStateNeed::cached(BASE / 2, RewriteCount::default())),
            0,
            "a need priced below the copy's weight asks nothing"
        );
        assert!(c.try_hold(10_000));
        assert_eq!(
            c.rewrite_allowance(&need),
            object_state_rewrite_allowance(10_000 + BASE, &need)
        );
    }

    /// A growth is granted while the budget keeps one admission's room free
    /// after it, and refused one byte past that, changing nothing; a growth
    /// past the cap is refused whatever room the budget has; a target at or
    /// below what is held is granted, a lower one giving the difference back.
    /// The peak is the most held at once.
    #[test]
    fn audit_charge_grows_only_with_one_admissions_room_left_and_within_its_cap() {
        const ROOM: u64 = 3 * MIB;
        let g = gate(admission_charge() + ROOM, LONG_WAIT, 8);
        let c = audit_charge(&g, 0);

        assert!(!c.try_hold(ROOM + 1), "one byte past one admission's room");
        assert_eq!(
            (c.held(), g.reserved_bytes()),
            (0, 0),
            "refused: nothing moved"
        );
        assert!(c.try_hold(ROOM), "exactly one admission's room left");
        assert_eq!((c.held(), g.reserved_bytes()), (ROOM, ROOM));
        assert!(c.try_hold_at_least(ROOM - 1), "grow-only: already held");
        assert_eq!(c.held(), ROOM);
        assert!(c.try_hold(MIB), "a lower target is granted");
        assert_eq!(
            (c.held(), g.reserved_bytes(), c.peak()),
            (MIB, MIB, ROOM),
            "the difference given back; the peak kept"
        );

        let roomy = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let capped = audit_charge(&roomy, 0);
        assert!(
            SMALL_BUDGET - admission_charge() > SAMPLED_AUDIT_CHARGE_CAP_BYTES + 1,
            "fixture: the budget has room past the cap"
        );
        assert!(
            capped.try_hold(SAMPLED_AUDIT_CHARGE_CAP_BYTES),
            "at the cap"
        );
        assert!(
            !capped.try_hold(SAMPLED_AUDIT_CHARGE_CAP_BYTES + 1),
            "past the cap, with room for it"
        );
        assert_eq!(roomy.reserved_bytes(), SAMPLED_AUDIT_CHARGE_CAP_BYTES);
    }

    /// While an upload waits — an admission queued (held back by its
    /// session's receiving cap, so the budget has room) or a processing step
    /// queued for the slot — an audit's growth is refused at once, with room
    /// for it: it never goes ahead of an upload, and never queues itself (the
    /// queues are unchanged). Once the queue is empty it grows.
    #[tokio::test]
    async fn audit_charge_growth_is_refused_while_an_upload_is_queued_and_never_queues() {
        let token = CancellationToken::new();

        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let _first = admit_now(&g, "s", &token).await;
        let mut second = Box::pin(g.admit(UploadRpc::UpdateLayer, "s", &token));
        assert!(
            poll_once(second.as_mut()).await.is_pending(),
            "fixture: the session's receiving cap queues its second admission"
        );
        let c = audit_charge(&g, 0);
        assert!(
            g.budget_bytes() - g.reserved_bytes() > admission_charge() + MIB,
            "fixture: the budget has room for the audit"
        );
        let reserved = g.reserved_bytes();
        assert!(!c.try_hold(MIB), "an admission is queued: refused");
        assert_eq!(
            (g.waiters(), g.reserved_bytes(), c.held()),
            (1, reserved, 0)
        );
        drop(second);
        assert!(c.try_hold(MIB), "the queue is empty: granted");

        let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
        let mut holder = received(&g, "holder", &token).await;
        assert!(
            process_now(&mut holder, OVER, &token)
                .await
                .entered_overflow
        );
        let mut waiter = received(&g, "waiter", &token).await;
        let mut waiting = Box::pin(waiter.reserve_processing(OVER, &token));
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        let c = audit_charge(&g, 0);
        let reserved = g.reserved_bytes();
        assert!(!c.try_hold(MIB), "a processing step is queued: refused");
        assert_eq!(
            (g.processing_waiters(), g.reserved_bytes(), c.held()),
            (1, reserved, 0)
        );
        drop(waiting);
        assert!(c.try_hold(MIB), "the queue is empty: granted");
    }

    /// An audit's growth the budget cannot hold is refused even with the
    /// overflow slot free and nothing queued, where an upload's processing
    /// step would take the slot: the audit runs in budget or not at all, and
    /// the slot stays free for the next upload.
    #[tokio::test]
    async fn audit_charge_never_takes_the_overflow_slot() {
        let token = CancellationToken::new();
        // Two admissions' room: with `r` holding `F`, the audit's room (one
        // admission left free) is under its cap, so the budget refuses it.
        let (g, metrics) = gate_with_metrics(2 * admission_charge(), LONG_WAIT, 8);
        let mut r = received(&g, "r", &token).await;
        let c = audit_charge(&g, 0);
        let over = g.budget_bytes() - g.reserved_bytes() - admission_charge() + 1;
        assert!(
            over <= c.cap(),
            "fixture: refused by the budget, not the cap"
        );

        assert!(!c.try_hold(over), "past the budget's room, the slot free");
        assert_eq!(
            (g.overflow_bytes(), c.held(), overflow_runs(&metrics)),
            (0, 0, 0.0)
        );
        assert!(
            r.try_hold_processing(OVER)
                .expect("the slot is still free")
                .entered_overflow,
            "the slot was left free for the upload"
        );
    }

    /// A shrink gives the difference back at once and serves a processing
    /// step waiting for that room; dropping a charge gives everything back
    /// and serves the queue too — on a panic's unwind as well.
    #[tokio::test]
    async fn audit_charge_shrink_and_drop_give_back_and_serve_the_queue_even_on_a_panic() {
        for end in ["shrink", "drop", "panic"] {
            let token = CancellationToken::new();
            let g = gate(SMALL_BUDGET, LONG_WAIT, 8);
            let mut holder = received(&g, "holder", &token).await;
            assert!(
                process_now(&mut holder, OVER, &token)
                    .await
                    .entered_overflow
            );
            let mut r = received(&g, "r", &token).await;
            let c = audit_charge(&g, 0);
            assert!(
                c.try_hold(c.cap()),
                "{end}: fixture: the audit holds its cap"
            );
            // `r` asks one byte past the room the audit left: it waits for
            // the audit's bytes (the slot is held).
            let target = r.reserved() + (g.budget_bytes() - g.reserved_bytes()) + 1;
            let mut waiting = Box::pin(r.reserve_processing(target, &token));
            assert!(
                poll_once(waiting.as_mut()).await.is_pending(),
                "{end}: fixture: the step waits"
            );

            match end {
                "shrink" => assert!(c.try_hold(0)),
                "drop" => drop(c),
                _ => {
                    let unwound =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                            let _held = c;
                            panic!("an audit panicking while it holds its charge");
                        }));
                    assert!(unwound.is_err(), "fixture: the audit panicked");
                }
            }

            let Poll::Ready(Ok(grant)) = poll_once(waiting.as_mut()).await else {
                panic!("{end}: the bytes given back must serve the waiting step");
            };
            assert!(!grant.entered_overflow, "{end}: in budget");
            drop(waiting);
            assert_eq!(
                g.reserved_bytes(),
                r.reserved(),
                "{end}: the audit holds nothing; only the upload's bytes remain"
            );
        }
    }

    /// An audit's charge moves only the in-budget total and
    /// `live_edit_upload_reserved_bytes`: no admission, in-flight, waiter,
    /// overflow or processing-wait series, whether it grows, shrinks or is
    /// dropped.
    #[test]
    fn audit_charge_moves_only_the_reserved_bytes_series() {
        let (g, metrics) = gate_with_metrics(SMALL_BUDGET, LONG_WAIT, 8);
        let others = |text: String| -> Vec<String> {
            text.lines()
                .filter(|line| !line.contains("live_edit_upload_reserved_bytes"))
                .map(str::to_owned)
                .collect()
        };
        let reserved = || metric(&metrics, "live_edit_upload_reserved_bytes", &[]);
        let before = others(render(&metrics));
        let c = audit_charge(&g, 0);

        assert!(c.try_hold(2 * MIB));
        assert_eq!(reserved(), (2 * MIB) as f64);
        assert!(c.try_hold(MIB));
        assert_eq!(reserved(), MIB as f64);
        drop(c);
        assert_eq!(reserved(), 0.0);
        assert_eq!(
            others(render(&metrics)),
            before,
            "no series but the reserved bytes moved"
        );
    }
}
