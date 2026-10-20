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

//! PRD AC-8 — the restore-interruption **fault-injection matrix**: Task
//! `t2.dfh.yj.u3`, Subtask `t3.dfh.yj.u3.uc`.
//!
//! ## The two claims this file owns
//!
//! Every arm injects a fault at one point of the composed-restore pipeline and
//! asserts the same two things, which is what makes it a matrix rather than a
//! pile of cases:
//!
//! - **(a) No partially-applied object is ever observable.**
//!   [`assert_no_partial_object`] reads the durable state and admits exactly two
//!   shapes: the object's whole `le:{oid}:obj:*` keyspace is ABSENT, or the
//!   object equals the fully-restored expected state on the faithful-column
//!   basis. Never anything between the two.
//! - **(b) Retry converges.** Once the fault is cleared, driving the SAME
//!   production entry point (`handle_stat_layer` → the recovery hook) lands the
//!   exact expected state, including the composed head's cursor columns.
//!
//! ## Injection points (the matrix), and what each one lands on
//!
//! | # | Injection point | Instrument | Landed observable shape | Path |
//! |---|---|---|---|---|
//! | 1 | mid-gather fetch, CLASSIFIED error | store `get_layer_checkpoint` → `Transient` | propagates VERBATIM to `RecoveryOutcome::Error`; NO lineage cell moves | composed |
//! | 2 | mid-gather fetch, MISSING member | store `get_layer_checkpoint` → `Ok(None)` | `stepped_down_link_broken`, restore lands on the older generation | base-only |
//! | 3 | composition | corrupt one tail record's OMPF payload | `stepped_down_compose`, restore lands on the older generation | base-only |
//! | 4 | between ext pre-store and object write | [`RESTORE_SEAM_AFTER_EXT_PRESTORE`] via `RecoveryHook::with_restore_fault_injector_for_test` | `RecoveryOutcome::Error`, nothing materialized | composed |
//! | 5 | SR-3.6 gap re-base PUT | store `put_layer_checkpoint` targeted at `(seeded_seq, Base)` | the RESTORE is unaffected (the re-base is best-effort); the base simply does not land | composed |
//! | 6 | degraded adapter — TWO wedged members | store `get_layer_checkpoint` that never returns | `stepped_down_budget` per candidate, ONE reserve per wedged MEMBER, floor answer still reached | base-only |
//! | 7 | b.4et stream high-water probe | [`RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE`] `Transient` x every attempt, over a SPARED stream | `RecoveryOutcome::Error`, nothing materialized, `unresolved{reason="probe_unreadable"}` | composed |
//! | 8 | b.4et stale-stream `UNLINK` | [`RECOVERY_SEAM_STALE_STREAM_UNLINK`] `Transient` x every attempt, over a spared stream carrying a tombstone top | `RecoveryOutcome::Error`, nothing materialized, the stale stream still present, `unresolved{reason="unlink_unverified"}` | composed |
//! | 9 | b.4et stream-key `TYPE` classification | [`RECOVERY_SEAM_STREAM_KEY_TYPE`] TERMINAL, paired with a `Transient` probe seam, over a spared stream | `RecoveryOutcome::Error`, nothing materialized, `unresolved{reason="probe_unreadable"}` | composed |
//! | 10 | between the b.f4t residue WIPE and the object write | [`RESTORE_SEAM_AFTER_EXT_PRESTORE`] over a keyspace seeded with surviving residue | `RecoveryOutcome::Error`, residue already gone, nothing materialized, retry converges | composed |
//! | 11 | INSIDE the wipe's `WATCH` … `EXEC` window — a peer lands a structurally-COMPLETE header | [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`] returning `Ok` after `HSET`ting the 12 canonical fields | tripped `EXEC` ⇒ `PeerLandedHeader` ⇒ `AlreadyPresent` ⇒ `RecoveryOutcome::Restored`; nothing wiped, nothing written, the peer's header serves the `StatLayer` | composed |
//! | 12 | INSIDE the same window — a SINGLE-field header write | the same seam returning `Ok` after `HSET superseded=1` | tripped `EXEC` ⇒ `Transient` ⇒ UNAVAILABLE; nothing wiped, nothing written, and a second attempt converges | composed |
//! | 13 | INSIDE the same window — the seam REFUSES | the same seam returning `Err(Transient)` | the `WATCH` is released and the error propagates verbatim; nothing wiped, nothing written, and a second attempt converges | composed |
//! | 14 | degraded adapter — ONE wedged member at the SHIPPED budget ratio (`budget == 2 x reserve`) | store `get_layer_checkpoint` that never returns, over a hook `Config` at [`SHIPPED_RATIO_RESERVE`] | `stepped_down_budget` once, floor answer still reached — the b.bu8 accounting claim at production's own ratio | base-only |
//! | 15 | degraded adapter — a slow LIST ([`INCONCLUSIVE_LIST_DELAY`]) *plus* a wedged head, leaving the healthy base UNREAD | store `list_layer_checkpoints` that sleeps, over the same shipped-ratio `Config` | `exhausted_budget`, `RecoveryOutcome::Error`, retriable UNAVAILABLE — never `not_recoverable`/NOT_FOUND (b.hxk); nothing materialized | none resolved |
//!
//! Cross-cutting, per the Subtask's arm 6: arms 1 / 4 / 5 / 7 / 8 / 9 / 10 / 11
//! / 12 / 13 run on the **composed** path and arms 2 / 3 / 6 / 14 resolve on the
//! **base-only** path, so claim (a) is proven not to be composition-specific.
//! Arm 15 resolves on neither — it is the arm where the selection produces no
//! lineage at all — so it proves claim (a) on the third shape.
//!
//! ## Arms 6 and 14 are one claim split by BUDGET RATIO, not a duplicate
//!
//! Both drive a wedged `get_layer_checkpoint` and both assert the floor answer,
//! so the split needs stating or a later reader will collapse them:
//!
//! - **Arm 6** owns the selection-wide `LinkFact::Wedged` cache — a wedged
//!   member is fetched at most once for the WHOLE selection, not once per
//!   candidate that reaches it. Proving that needs two *actually fetched*
//!   wedges plus the surviving lineage's own fetches, so it needs a budget well
//!   above two reserves ([`WEDGE_BUDGET_MS`], eight). At production's ratio the
//!   second wedge would be declined un-fetched and the claim would pass
//!   vacuously.
//! - **Arm 14** owns the b.bu8 accounting at the ratio production actually runs:
//!   `checkpoint_recovery_fetch_budget()` is `max(rpc_deadline_ms, 2 x
//!   checkpoint_request_timeout_ms)`, i.e. exactly two reserves at the
//!   defaults, where one wedged member plus the floor answer IS the whole
//!   budget. Pre-b.bu8 the base was declined there by the very threshold meant
//!   to protect it, so arm 6's headroom is precisely what hid the defect.
//!
//! Neither subsumes the other: arm 6's budget cannot see the defect, and arm
//! 14's ratio cannot fund the caching claim.
//!
//! ## The two windows around the b.f4t residue wipe, and which seam covers each
//!
//! b.f4t inserted a pre-write residue wipe at `apply_restore` step 3b, between
//! the decode and the SR-3.4 ext pre-store. That created **two** distinct
//! windows, and each has its own seam:
//!
//! - **After the wipe, before `write_object`** — arm 10. The existing
//!   [`RESTORE_SEAM_AFTER_EXT_PRESTORE`] already sits there, so arm 10 reuses
//!   it rather than inventing a second seam.
//! - **Inside the wipe, between its `WATCH` on `obj:hdr` and its `EXEC`** —
//!   arms 11 / 12 / 13, through
//!   [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`](live_edit::checkpoint::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC),
//!   the production constant b.f4t landed for exactly this purpose (this file
//!   still invents no seam of its own). It is the only seam in the repo that
//!   fires between a `WATCH` and its `EXEC`, and that position is what makes
//!   the wipe's tripped-`EXEC` arm reachable on purpose rather than by timing:
//!   its two verdicts differ only in WHICH write landed on `obj:hdr` inside a
//!   window made of real round trips, so without the seam a test could pick one
//!   only with a `sleep` — which the test-writing guide forbids as a
//!   synchronisation primitive, and which would in any case yield
//!   `PeerLandedHeader` OR `Transient` OR `Wiped` depending on load.
//!
//! The seam's delivery precondition is wired here by construction: the injector
//! rides `ExtPrestore`, so a hook installing `with_restore_fault_injector_for_test`
//! must ALSO carry `with_ext_value_store` or the seam is inert. `CheckpointHarness::
//! install_recovery_hook` rebuilds from `svc.ext_value_store`, which
//! [`setup_matrix_harness`] populates with a real `ValkeyExternalValueStore` —
//! and each of arms 11 / 12 / 13 asserts its injector actually fired, so an
//! inert seam fails the arm rather than passing it vacuously.
//!
//! The same verdict reached one round trip earlier — the post-`WATCH` re-read
//! rather than the tripped `EXEC` — stays covered by `src/checkpoint/residue.rs`'s
//! unit lane, which observes the `ResidueOutcome` enum directly; these arms are
//! the composed, handler-level half that the unit lane cannot see (the apply
//! folds `AlreadyPresent` and `Applied` into one `Ok`).
//!
//! ## Why the b.4et arms belong in this matrix and not only in their own file
//!
//! `e2e_object_id_preservation_test.rs` owns the b.4et **arm table** — which
//! probe/`TYPE` combination proceeds, which refuses, what each one reports, and
//! the seq arithmetic each produces. Arms 7 and 8 here make a different claim,
//! the one this file owns and that file does not: that a refusal *between* the
//! lineage gather and the object write leaves NO partially-applied object
//! behind, and that clearing the fault converges on the exact composed head.
//! They are the first arms to interrupt the restore at a point that is neither
//! inside the gather nor inside the apply, which is precisely why the invariant
//! needs re-proving there.
//!
//! ## The mid-gather contract, READ FROM THE LANDED CODE
//!
//! `src/checkpoint/lineage.rs` settles the propagate-vs-step-down question, and
//! the three landed behaviours are complementary rather than contradictory —
//! this file asserts each as landed and names it here:
//!
//! 1. **A classified store error propagates verbatim.** `fetch_member` does
//!    `Ok(result) => result?`, and both `gather_restore_lineage` and
//!    `LineageSelection::next_restorable` document "Store errors propagate
//!    **verbatim**: `Transient`, `Permanent`, `NotImplemented`". Stepping down on
//!    a transient error would serve older state where a retry gets the newest.
//!    Test 1 asserts that shape, including that NO `stepped_down_*` cell moves.
//! 2. **A fetch that overruns its per-fetch allowance is `stepped_down_budget`.**
//!    `fetch_member` wraps every `get_layer_checkpoint` in
//!    `tokio::time::timeout(allowance, ..)` — a kind-dependent cap of at most
//!    one reserve, taken off the live remainder — and records the overrun as a
//!    selection-wide `LinkFact::Wedged`. Tests 6 and 14 assert that shape, at
//!    two budget ratios; see "Arms 6 and 14" above for why both are needed.
//! 3. **A missing member (`Ok(None)`) is `stepped_down_link_broken`.**
//!    `fetch_member`'s `let Some(blob) = fetched else` arm mints
//!    `LinkFact::Absent { at_fetch: true }`, whose `reason_label()` is
//!    `"stepped_down_link_broken"`. Test 2 asserts that shape.
//!
//! A fourth behaviour rides on shape 2 and is a separate claim from it: **what
//! a budget verdict means to the CLIENT depends on the BASES, not on the
//! step-downs.** Post-b.hxk `next_restorable`'s drain splits on
//! `unadjudicated_bases()`. An exhaustion that adjudicated every listed `Base`
//! is `Exhaustion::AllBroken` -> `exhausted` -> `Ok(None)` ->
//! `not_recoverable` -> terminal NOT_FOUND; one that left a listed `Base`
//! unread is `Exhaustion::BudgetInconclusive` -> `exhausted_budget` ->
//! `CheckpointError::Transient` -> `error` -> retriable UNAVAILABLE. Test 15
//! asserts the second, and Tests 6 and 14 are what make it a genuine split
//! rather than "any budget step-down is retriable": both tick
//! `stepped_down_budget` and both end `restored`.
//!
//! No contradiction with the Task-2 ticket set was found in the landed code, so
//! nothing here is asserted "around" a defect. Had one existed it would have
//! been escalated to the Task rather than encoded in an assertion.
//!
//! ## The `stepped_down_budget` DEADLINE arm is delegated, not dropped
//!
//! Shape 2 above has two producers: a fetch that overruns its per-fetch
//! allowance (Tests 6 and 14 drive it end-to-end) and a *selection* whose
//! remaining deadline can
//! no longer fund one more fetch. The second needs a controllable clock, and
//! `tokio::time::pause` is not usable in this binary — the tests are
//! `flavor = "multi_thread"` and every step touches a live Valkey, so pausing
//! time stalls real I/O rather than advancing the selection. The **exact
//! arithmetic** of that arm is therefore still **delegated by citation** to
//! `t3.dfh.yj.wg.xn`'s per-fetch-timeout unit coverage in
//! `src/checkpoint/lineage.rs`'s `mod tests`, whose PM condition C1 mandates
//! the never-returning-fetch → budget-step-down case under
//! `tokio::time::pause`. It is named here so the obligation cannot silently
//! drop: the Subtask's original six arms — plus arm 14, the second overrun
//! producer at the shipped budget ratio — **plus the cited deadline
//! delegation**, which stays delegated for its arithmetic.
//!
//! Arm 15 narrows what is delegated without replacing it. It drives a
//! zero-allowance decline end-to-end (a real slow LIST, a real wedge, real wall
//! clock), so the *consequence* — which status the client gets when a listed
//! base is never read — is asserted here at the handler rather than cited. What
//! stays delegated is the arithmetic that produces it: exact `elapsed`
//! equalities, per-candidate allowance values, and the
//! `unadjudicated_bases` count over mixed fact shapes, none of which a live
//! clock can pin.
//!
//! ## Scope fence — what this file deliberately does NOT restate
//!
//! - `t3.dfh.yj.b7.74` (`e2e_composed_restore_ext_order_test.rs`, Test 2) owns
//!   the SR-3.4 **durable-pair** claim at the ext-prestore seam: fired exactly
//!   once, `obj:hdr` absent AND the tail-carried ext hashes present at that
//!   instant, plus the accepted ref-count posture. Test 4 here is narrowed to
//!   the **retry-convergence half** of that seam (PM condition C6) and asserts
//!   the file-wide no-partial-object invariant, nothing about ext ordering or
//!   ref counts.
//! - `t3.dfh.yj.4f.gw` (`restore_rebase_test.rs`) owns the re-base **cursor-column
//!   matrix** — `restore_seq_bump` retirement, `last_checkpoint_seq` /
//!   `last_checkpoint_ms` stamping across the persisted / skipped / failed /
//!   shed arms, and the `refused_seq_gap` capture consequence. Test 5 here
//!   asserts only that the RESTORED OBJECT is complete and consistent when the
//!   re-base PUT fails, and that a later attempt lands the base at `seeded_seq`.
//!
//! ## Feature gating — the SILENT second activation condition
//!
//! The module is **unconditionally compiled**. A `#![cfg(feature = ...)]` at the
//! top would make the whole file compile to nothing and report green under the
//! default feature set, which is the failure mode the test-writing guide's
//! "Valkey Integration Tests" section warns about for a *silent* activation
//! condition: nothing in a passing run tells you the body never ran.
//!
//! EIGHT tests — [`the_ext_prestore_seam_interrupt_converges_on_retry`], the
//! three b.4et arms, and the four b.f4t residue arms (10 / 11 / 12 / 13) — name
//! `CheckpointFaultInjector`, whose own cfg is precisely
//! `feature = "test-checkpoint-fault-injection"` (not `any(test, ...)`), so
//! those tests and their private support items carry
//! `#[cfg(feature = "test-checkpoint-fault-injection")]` and nothing else does.
//! Under the default feature set six of the fourteen arms run; the other eight
//! are silently absent. The invocation that runs all fourteen:
//!
//! ```text
//! cargo nextest run -p live-edit --test integration \
//!     --features test-support,test-checkpoint-fault-injection \
//!     -E 'test(/restore_fault_injection_matrix/)'
//! ```
//!
//! Every adapter-side arm is expressed with a
//! [`DelegatingCheckpointStore`](super::helpers::DelegatingCheckpointStore)
//! newtype rather than the op-label-keyed injector precisely so it stays
//! ungated: the injector dispatches on the method label alone and could not
//! separate the re-base PUT from the loop's, nor one lineage member's fetch from
//! another's.
//!
//! ## Synchronization budget — no `sleep` is used for coordination
//!
//! | Wait | Bound | What a timeout means |
//! |---|---|---|
//! | first loop-written base lands | [`LOOP_BASE_DEADLINE_SECS`] (10 s, bounded poll; the helper clamps up to its own load-tolerant floor) | the checkpoint loop never assembled a base |
//! | loop task drains after `cancel()` | [`LOOP_DRAIN_BUDGET`] (5 s) | a wedged loop; never observed, it returns at its next `select!` |
//! | bootstrap / writer tracker drains | [`TRACKER_IDLE_DEADLINE_SECS`] (10 s, clamped) | a spawned SR-3.6 re-base never completed |
//! | wedged member fetch (Test 6 only) | [`WEDGE_RESERVE`] (1 s) x 2 members | not a wait the test performs — it is the production per-fetch cap the arm exists to exercise |
//! | wedged member fetch (Test 14 only) | [`SHIPPED_RATIO_RESERVE`] (3 s) x 1 member | as above; Test 14 drives ONE wedge, so it pays one reserve — and its reserve is the larger of the two because that arm runs at production's ratio and so has no headroom to absorb runner load (see the constant) |
//! | slow LIST + clamped wedge (Test 15 only) | [`INCONCLUSIVE_LIST_DELAY`] (4 s) + the remainder it leaves (2 s) = one whole 6 s budget | as above; the wedge there is clamped to what the LIST left, so the two terms sum to the budget by construction rather than adding to it |
//!
//! Worst case per test is roughly 10 s (fixture) + 10 s (drain) + up to 6 s of
//! injected adapter latency — Test 6 pays `2 x WEDGE_RESERVE`, Test 14 one
//! `SHIPPED_RATIO_RESERVE`, Test 15 one whole budget, and no other arm wedges at
//! all — comfortably inside `.config/nextest.toml`'s 120 s hard kill. A
//! SIGKILLed test prints none of its diagnostics.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; under the
//! project sandbox that variable is preset (CLAUDE.md).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{ClientLike, HashesInterface, KeysInterface};
use fred::types::scan::ScanType;
use futures::StreamExt;
use tonic::Request;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    BaseBootstrap, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind,
    CheckpointMeta, CheckpointStore, LayerArtifact,
};
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES;
use live_edit::storage::diff_read::ObjectSnapshot;
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
// Read-only reuse, exactly as `e2e_composed_restore_ext_order_test.rs` does:
// the surgical ext-namespace wipe is a correctness contract (explicit hashes
// only — a wildcard would also catch the `le:ext:{gc}:*` bookkeeping sets and
// change what the assertion means), and one home for it keeps the files from
// drifting apart. It is `pub(super)` so siblings can consume it.
use super::e2e_ext_value_recovery_test::wipe_ext_namespace_for_hashes;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ─────────────────────────────────────────────────────────────────────────────
// Named waits and fixture constants — see the module header's budget table
// ─────────────────────────────────────────────────────────────────────────────

/// Checkpoint-loop tick cadence. `1` is the value every e2e recovery file in
/// this module uses: the harness sets `interval = quiesce = 0`, so a
/// just-written layer is due on the very next tick.
const TICK_SECS: u64 = 1;

/// Deadline handed to `wait_for_checkpoint_at_least`. The helper clamps it up
/// to its own load-tolerant floor, so this is the *lower* bound on how long a
/// contended Valkey may take before the wait is called a failure.
const LOOP_BASE_DEADLINE_SECS: u64 = 10;

/// Bound on the post-`cancel()` drain of the checkpoint loop's `JoinHandle`.
/// The loop returns at its next `select!` on the cancellation token, so this is
/// never the wait actually observed — it exists so a wedged loop fails the test
/// promptly instead of hanging the binary.
const LOOP_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// Deadline handed to `helpers::wait_for_writer_idle` when draining the shared
/// `TaskTracker` that both the session-checkpoint writer and the SR-3.6 re-base
/// spawn onto. Clamped up to the helpers' load-tolerant floor.
const TRACKER_IDLE_DEADLINE_SECS: u64 = 10;

/// Test 6's per-fetch reserve — `Config::checkpoint_request_timeout_ms` on the
/// hook's own `Config`. It is the wall-clock cost of ONE wedged member, so it is
/// dialled far below the 30 s production value: two wedged members then cost 2 s
/// rather than 60 s. It must stay comfortably above the latency of a *healthy*
/// in-memory fetch, or a legitimate member would be mistaken for a wedged one.
const WEDGE_RESERVE: Duration = Duration::from_millis(1_000);

/// Test 6's outer selection budget — `Config::rpc_deadline_ms` on the hook's own
/// `Config`, which `checkpoint_recovery_fetch_budget()` folds to
/// `max(rpc_deadline_ms, 2 x checkpoint_request_timeout_ms)`.
///
/// Deliberately well ABOVE two reserves, and **kept** so after b.bu8. Test 6's
/// claim is that TWO wedged members are each addressed at most once and the
/// surviving lineage's own fetches still land afterwards; three members'-worth
/// of speculative budget is what that claim structurally requires, whatever the
/// per-fetch accounting rule is. Post-b.bu8 the base's own fetch is reserved out
/// of the budget by member kind, so the floor answer no longer *depends* on this
/// headroom — but the two wedged members do, because a `Tail` allowance is
/// `min(reserve, remaining - reserve)` and the second wedge only gets to spend
/// anything while more than one reserve is still standing above the
/// reservation. Dialling this down to two reserves would silently convert Test 6
/// into a weaker copy of Test 14.
///
/// The shipped ratio — `budget == 2 x reserve`, which is what the formula yields
/// at the production defaults and the posture b.bu8 was actually about — is
/// covered by its own arm, Test 14
/// ([`one_wedged_member_at_the_shipped_two_reserve_ratio_still_reaches_the_floor`]),
/// at [`SHIPPED_RATIO_RESERVE`].
const WEDGE_BUDGET_MS: u64 = 8_000;

/// Test 14's reserve AND its `rpc_deadline_ms`, which is what makes that arm's
/// budget exactly two reserves: `max(3 s, 2 x 3 s) = 6 s`.
///
/// One number for both knobs, deliberately — the arm's whole precondition is the
/// RATIO, and spelling it once makes the `2 x reserve` assertion in the arm a
/// restatement of the formula rather than of two independently-chosen literals.
/// It also makes the value safe to move: raising it cannot perturb the ratio,
/// because both terms of `max(rpc_deadline_ms, 2 x checkpoint_request_timeout_ms)`
/// scale together.
///
/// Scaled down from production (30 s reserve / 60 s budget) for the same reason
/// [`WEDGE_RESERVE`] is: the wedged member costs its whole allowance of REAL
/// wall clock, so the arm's runtime is this constant. It is 3 s rather than
/// [`WEDGE_RESERVE`]'s 1 s because this arm alone has no headroom by
/// construction, and the reserve bounds more than the wedge. It does not bound
/// the LIST: `list_layer_checkpoints` runs BEFORE the first member is addressed
/// and no per-fetch cap covers it — the deadline is sampled before the LIST and
/// consulted only after — so a slow store spends the budget there first and can
/// return with `remaining < reserve` (see `gather_restore_lineage` and the
/// `list_delay` note in `src/checkpoint/lineage.rs`). What the reserve caps is
/// the MEMBER fetches, at one reserve each: the wedge first, then the healthy
/// base `get`, whose fetch and decode are what the slack the wedge leaves has
/// to fund. That is the number sized here. A 1 s reserve left that
/// non-wedge slack at 1 s, thin enough that a contended runner could time out a
/// HEALTHY base fetch and turn the arm's claim (`restored` == 1) red for load
/// rather than for the accounting defect it exists to catch. 3 s triples the
/// absolute slack while preserving the ratio exactly, and still must stay
/// comfortably above the latency of a healthy in-memory fetch — which, at three
/// orders of magnitude below it, it is.
///
/// The arm's wedge duration scales WITH this constant: the wedge lands on the
/// gather's un-gated first fetch, whose allowance is `min(reserve, remaining)`,
/// so Test 14 pays exactly one of these and the module header's budget table
/// row moves with it.
const SHIPPED_RATIO_RESERVE: Duration = Duration::from_millis(3_000);

/// Test 15's LIST cost. Strictly ABOVE `budget − reserve` — 4 s against a 6 s
/// budget and a 3 s reserve — which is the whole precondition of that arm.
///
/// It leaves 2 s, i.e. **under one reserve**, so the gather's un-gated first
/// fetch is clamped to the remainder rather than granted a flat reserve. The
/// wedge armed on that fetch then consumes exactly what is left, and the
/// perfectly healthy base behind it is declined at a **zero** allowance —
/// unread, and so unadjudicated. That is the shape b.hxk is about: an
/// exhaustion holding no evidence at all about the artifact that could have
/// rooted a lineage.
///
/// Sized against [`SHIPPED_RATIO_RESERVE`] rather than independently: the arm's
/// claim is the RATIO of the LIST to the budget, so if that constant moves this
/// one must move with it. The arm asserts the relation below rather than
/// trusting the two literals.
///
/// The arm's wall clock is this plus the clamped wedge — 4 s + 2 s = the 6 s
/// budget — which is the same order as Tests 6 and 14 and far inside the 120 s
/// hard kill.
const INCONCLUSIVE_LIST_DELAY: Duration = Duration::from_millis(4_000);

/// `client_id` on every UpdateLayer header. Only needs to be stable and
/// non-empty; self-echo suppression is not under test.
const CLIENT_ID: &str = "restore-matrix-client";

const SESSION_BASE_PATH: &str = "/stage";
const CREATED_BY: &str = "restore-matrix-user";

/// The section every delta in this file writes into.
const ROOT_SECTION_ID: u64 = 42;

/// Section minted by the first tail-range commit, so the replayed range carries
/// structural work and not only field writes.
const TAIL_SECTION_ID: u64 = 1_000;

/// Authoring rank stamped on every `SetField`. Every field in this file is
/// written exactly once, so the value only has to be stable.
const AUTHORING_RANK: u64 = 0;

/// Byte length of the external value the tail-range carries. Large enough to be
/// unambiguously an out-of-band OMPF part, small enough to stay free.
const EXT_BLOB_LEN: usize = 4 * 1024;

/// The one-based OMPF part index the ext-carrying `SetField` points at. Part 0
/// is the OODF delta; `delta_to_ompf_bytes(delta, &[blob])` puts `blob` at
/// Part 1.
const EXT_VALUE_INDEX: u32 = 1;

/// How many real commits the fixture drives ABOVE the loop-written base. Three,
/// so the `Chained` layout can put one commit in each of three linked tails and
/// the middle one can be the ext-value commit.
const TAIL_RANGE_COMMITS: u64 = 3;

/// The unparseable payload planted in place of a real record's OMPF for the
/// composition-failure arm. Fails `OmpfContainer` parsing at the magic, which
/// the compose engine classifies `ComposeError::OmpfParse` — `Permanent` by
/// construction, so the selection steps down rather than retrying.
const CORRUPT_RECORD_PAYLOAD: &[u8] = b"not-an-ompf-container-not-even-close";

/// Marker carried on the injected gather error's boxed source, so Test 1's
/// status assertion is pinned to THIS injection rather than to any UNAVAILABLE
/// the restore path could produce for another reason.
const GATHER_ERROR_MARKER: &str = "restore-matrix-gather-fetch-fault";

/// Marker carried on the injected re-base PUT failure's boxed source.
const REBASE_PUT_ERROR_MARKER: &str = "restore-matrix-rebase-put-fault";

// ─────────────────────────────────────────────────────────────────────────────
// Payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// CreateLayer payload: one section plus one inline field, so the loop's base
/// carries real state.
fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: ROOT_SECTION_ID,
            section_name: "restore-matrix-root".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// A pre-base commit: one new inline field on the root section.
///
/// `base_version: 1` (any non-zero) marks this an INCREMENTAL diff. With
/// `base_version == 0` and `is_diff == true` the domain engine takes the reset
/// branch and wipes prior state, which would empty the fixture.
fn pre_base_commit_delta(index: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: format!("pre-base-field-{index}"),
            value_or_ext_hash: vec![0x10 + index as u8, 0x11, 0x12],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Tail-range commit 1: a new section plus an inline field inside it.
/// Structural work in the replayed range, so the composed comparison is not
/// purely field-shaped.
fn tail_structural_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ROOT_SECTION_ID,
            section_id: TAIL_SECTION_ID,
            section_name: "restore-matrix-tail-section".to_string(),
            section_type: 2,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: TAIL_SECTION_ID,
            key_name: "tail-inline-field".to_string(),
            value_or_ext_hash: vec![0xC0, 0xDE, 0x01],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Tail-range commit 2: ONE external-value `SetField`.
///
/// **Load-bearing for the whole matrix (Task-3 PM constraint).**
/// `RESTORE_SEAM_AFTER_EXT_PRESTORE` sits inside `if !ext_values.is_empty()` in
/// `apply.rs`, so a zero-ext lineage arms the seam and never fires it — the
/// restore just succeeds and the arm goes vacuously green. Every lineage this
/// file composes carries this commit.
fn tail_ext_value_delta(hash: &[u8; 32], blob_len: usize) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: TAIL_SECTION_ID,
            key_name: "tail-ext-field".to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: blob_len as u64,
            ext_value_index: EXT_VALUE_INDEX,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Tail-range commit 3: a trailing inline field, so the head of the replayed
/// range is not the ext commit itself.
fn tail_trailing_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: TAIL_SECTION_ID,
            key_name: "tail-trailing-field".to_string(),
            value_or_ext_hash: vec![0xC0, 0xDE, 0x02],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Deterministic external-value payload. An RNG would make a failing run
/// unreproducible for no benefit — the bytes only have to be non-trivial and
/// stable so `ext_content_hash` of them is the fixture's own contract.
fn ext_blob() -> Vec<u8> {
    (0..EXT_BLOB_LEN).map(|i| (i % 251) as u8).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The fault-injecting store — one newtype, every adapter-side arm
// ─────────────────────────────────────────────────────────────────────────────

/// What an armed `get_layer_checkpoint` does instead of delegating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GetFault {
    /// Return `CheckpointError::Transient` carrying [`GATHER_ERROR_MARKER`].
    /// The landed gather contract propagates this verbatim (module header).
    Transient,
    /// Return `Ok(None)` — the contract-documented concurrent-prune race, which
    /// the landed gather turns into `stepped_down_link_broken`.
    Absent,
    /// Never return. The production per-fetch `tokio::time::timeout(reserve, ..)`
    /// is what ends it, which is the whole point of the arm; the pending future
    /// is dropped by that timeout, so nothing leaks.
    Wedge,
}

/// One `put_layer_checkpoint` attempt, as the wrapper saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PutAttempt {
    snap_seq: u64,
    kind: CheckpointKind,
}

/// `CheckpointStore` newtype that can fail, blank, or wedge the fetch of a
/// NAMED `(snap_seq, kind)` and can fail the PUT of a named `(snap_seq, Base)`.
///
/// # Why a store wrapper and not the `CheckpointFaultInjector`
///
/// The injector dispatches on the method-name label alone
/// (`"get_layer_checkpoint"`, `"put_layer_checkpoint"`, …). Every arm in this
/// file needs to separate one lineage member's fetch from another's, or the
/// SR-3.6 re-base PUT from the checkpoint loop's — a distinction the label
/// cannot express. `RecordingRebaseStore` in `restore_rebase_test.rs` is the
/// precedent, and Task 4 landed no production injector seam on the re-base, so
/// this is the sanctioned instrument rather than a second-best one. It also
/// keeps five of the six arms UNGATED.
///
/// # Delegation
///
/// Composed atop [`DelegatingCheckpointStore`](super::helpers::DelegatingCheckpointStore)
/// and forwards **every** method, including the four with non-trivial trait
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting those would
/// not leave a call un-instrumented — it would SUBSTITUTE behaviour behind the
/// wrapper and make downstream assertions vacuously green, the trap that
/// helper's own doc records.
struct MatrixFaultStore {
    delegate: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    get_faults: Mutex<HashMap<(u64, CheckpointKind), GetFault>>,
    get_calls: Mutex<Vec<(u64, CheckpointKind)>>,
    put_attempts: Mutex<Vec<PutAttempt>>,
    fail_base_put_at: Mutex<Option<u64>>,
    /// Wall clock `list_layer_checkpoints` burns before answering — the LIST
    /// half of Test 15's degraded adapter.
    ///
    /// A separate switch from [`GetFault`] because the two are bounded by
    /// different things, and the gap between them is the whole of that arm:
    /// every member fetch is capped by the gather's per-fetch allowance, while
    /// the enumeration is capped only by the adapter's own ceiling (540 s at
    /// stock knobs). The gather samples the caller's deadline BEFORE the LIST
    /// and consults it only after, so a slow enumeration can return having
    /// already spent the budget — the one cost no cap in this module covers.
    list_delay: Mutex<Duration>,
    list_calls: Mutex<usize>,
}

impl MatrixFaultStore {
    fn new() -> Self {
        Self {
            delegate: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            get_faults: Mutex::new(HashMap::new()),
            get_calls: Mutex::new(Vec::new()),
            put_attempts: Mutex::new(Vec::new()),
            fail_base_put_at: Mutex::new(None),
            list_delay: Mutex::new(Duration::ZERO),
            list_calls: Mutex::new(0),
        }
    }

    /// Make every subsequent `list_layer_checkpoints` cost `delay`. Armed AFTER
    /// the fixture is built, so the fixture's own enumerations stay free.
    fn arm_list_delay(&self, delay: Duration) {
        *self.list_delay.lock().unwrap_or_else(|p| p.into_inner()) = delay;
    }

    fn clear_list_delay(&self) {
        *self.list_delay.lock().unwrap_or_else(|p| p.into_inner()) = Duration::ZERO;
    }

    /// A mark into the LIST counter, taken immediately before the act step —
    /// same reasoning as [`Self::get_calls_mark`].
    fn list_calls_mark(&self) -> usize {
        *self.list_calls.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn list_calls_since(&self, mark: usize) -> usize {
        *self.list_calls.lock().unwrap_or_else(|p| p.into_inner()) - mark
    }

    fn arm_get(&self, snap_seq: u64, kind: CheckpointKind, fault: GetFault) {
        self.get_faults
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert((snap_seq, kind), fault);
    }

    /// Clear every armed fetch fault — the "fault cleared" half of every
    /// retry-convergence assertion.
    fn clear_get_faults(&self) {
        self.get_faults
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }

    /// A mark into the fetch log, to be taken immediately before the act step.
    ///
    /// Required, not cosmetic: `setup_matrix_fixture` fetches the loop-written
    /// base to read its meta, so an absolute count over the whole log would
    /// attribute that fixture-time call to the injection and every
    /// "exactly once" assertion would read one too many.
    fn get_calls_mark(&self) -> usize {
        self.get_calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len()
    }

    /// How many `get_layer_checkpoint` calls this wrapper has seen for
    /// `(snap_seq, kind)` SINCE `mark`.
    ///
    /// The non-vacuity probe for every fetch arm ("the targeted member really
    /// was addressed") and the caching proof for Test 6 ("a member that ate a
    /// whole reserve is never re-fetched, however many candidates re-reach it").
    fn get_calls_since(&self, mark: usize, snap_seq: u64, kind: CheckpointKind) -> usize {
        self.get_calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .skip(mark)
            .filter(|call| **call == (snap_seq, kind))
            .count()
    }

    /// Fail every `Base` PUT at exactly `snap_seq`. Targets the SR-3.6 re-base
    /// alone: the fixture's own base is written at `S0` by the checkpoint loop
    /// long before this is armed.
    fn arm_base_put_failure(&self, snap_seq: u64) {
        *self
            .fail_base_put_at
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(snap_seq);
    }

    fn clear_base_put_failure(&self) {
        *self
            .fail_base_put_at
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = None;
    }

    fn base_put_attempts_at(&self, snap_seq: u64) -> usize {
        self.put_attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|a| a.kind == CheckpointKind::Base && a.snap_seq == snap_seq)
            .count()
    }
}

#[async_trait]
impl CheckpointStore for MatrixFaultStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Recorded BEFORE the failure arm: the test wants to know the call was
        // ISSUED, whatever happens to it afterwards.
        self.put_attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(PutAttempt {
                snap_seq,
                kind: meta.kind,
            });

        let fail_at = *self
            .fail_base_put_at
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if meta.kind == CheckpointKind::Base && fail_at == Some(snap_seq) {
            return Err(CheckpointError::Transient {
                source: REBASE_PUT_ERROR_MARKER.into(),
            });
        }

        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would report "no such generation"
    // for a generation the delegate holds, to the restore selection whose whole
    // job is deciding what exists.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        let key = (handle.seq(), handle.kind);
        self.get_calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(key);
        // Read the verdict out and DROP the std mutex guard before any await —
        // holding it across one would make the future non-`Send`.
        let fault = self
            .get_faults
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
            .copied();
        match fault {
            Some(GetFault::Transient) => Err(CheckpointError::Transient {
                source: GATHER_ERROR_MARKER.into(),
            }),
            Some(GetFault::Absent) => Ok(None),
            Some(GetFault::Wedge) => {
                // The production per-fetch cap ends this; see `GetFault::Wedge`.
                std::future::pending::<()>().await;
                unreachable!("a pending future never resolves")
            }
            None => self.delegate.get_layer_checkpoint(handle).await,
        }
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        *self.list_calls.lock().unwrap_or_else(|p| p.into_inner()) += 1;
        // Read the delay out and DROP the std mutex guard before any await —
        // holding it across one would make the future non-`Send`.
        let delay = *self.list_delay.lock().unwrap_or_else(|p| p.into_inner());
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        self.delegate.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.delegate.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.delegate
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.delegate.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.delegate.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.delegate.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.delegate.list_live_ext_set().await
    }

    // ── The four non-trivial trait defaults ──────────────────────────────
    //
    // Forwarded because omitting them SUBSTITUTES behaviour rather than merely
    // skipping instrumentation: `read_ext_refs_sidecar`'s default returns an
    // EMPTY set, `list_layer_artifacts`' an empty vec, and both would make
    // downstream assertions vacuously green behind the wrapper.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.delegate.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.delegate
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.delegate.publish_worst_case()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracles and the no-partial-object probe
// ─────────────────────────────────────────────────────────────────────────────

/// The live state at one instant, as both a decoded snapshot and the raw
/// `obj:hdr` map — the two views a restored object is compared against.
struct Oracle {
    snapshot: ObjectSnapshot,
    header: HashMap<String, String>,
}

async fn capture_oracle(harness: &CheckpointHarness, object_id: u64) -> Oracle {
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test must succeed while the live layer is intact");
    let header = read_obj_header(harness, object_id).await;
    Oracle { snapshot, header }
}

/// `HGETALL le:{object_id}:obj:hdr`.
async fn read_obj_header(harness: &CheckpointHarness, object_id: u64) -> HashMap<String, String> {
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    harness
        .tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// Parse a `u64` column off an `obj:hdr` map with a diagnostic on absence.
fn header_u64(header: &HashMap<String, String>, field: &str) -> u64 {
    let raw = header.get(field).unwrap_or_else(|| {
        panic!(
            "obj:hdr MUST carry `{field}`; got fields: {:?}",
            header.keys().collect::<Vec<_>>()
        )
    });
    raw.parse::<u64>()
        .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}: {e}"))
}

/// Every `le:{object_id}:obj:*` key currently present.
///
/// The absence half of [`assert_no_partial_object`] probes the WHOLE per-object
/// keyspace rather than `obj:hdr` alone. `obj:hdr` is what
/// `apply_checkpoint`'s structural-header short-circuit keys on, so it is the column that
/// decides whether a retry is even possible — but a fault that wrote sections
/// or fields and then failed before the header would leave the header absent
/// and still be a partially-applied object. Scanning the family is what
/// excludes that.
async fn obj_keys(harness: &CheckpointHarness, object_id: u64) -> Vec<String> {
    let tv = &harness.tv;
    let pattern = format!("{}{{{object_id}}}:obj:*", tv.key_schema().prefix());
    let client = tv.client().pool.next();
    let mut stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<fred::types::Key, fred::error::Error>> + Send>,
    > = if client.is_clustered() {
        Box::pin(client.scan_cluster_buffered(pattern, Some(100), None::<ScanType>))
    } else {
        Box::pin(client.scan_buffered(pattern, Some(100), None::<ScanType>))
    };
    let mut found = Vec::new();
    while let Some(item) = stream.next().await {
        if let Ok(key) = item {
            let rendered: Option<&str> = key.as_str();
            found.push(rendered.unwrap_or("<non-utf8 key>").to_string());
        }
    }
    found.sort();
    found
}

/// Which of the two admissible shapes [`assert_no_partial_object`] observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectPresence {
    /// No `le:{oid}:obj:*` key exists at all.
    Absent,
    /// The object exists and equals the expected fully-restored state.
    FullyRestored,
}

/// **The AC-8 core, shared by every arm.**
///
/// Admits exactly two durable shapes and nothing between them:
///
/// - the object's entire `le:{oid}:obj:*` keyspace is ABSENT — the interruption
///   left no trace, and a retry can still run because
///   `apply_checkpoint`'s structural-header short-circuit has nothing to short-circuit on;
/// - the object EXISTS and equals `expected` on the faithful-column basis —
///   `helpers::assert_faithful_columns`, whose doc-comment carries the
///   authoritative excluded-column list — the restore completed.
///
/// A half-written object is a `obj:hdr` that exists while the sections / fields
/// / time samples disagree with `expected`, and that is what the second arm's
/// comparison excludes. Which of the two shapes an arm requires is the caller's
/// assertion; this helper's job is that it is one of them.
///
/// `FaithfulColumnsOptions::default()`: nothing here is asymmetric between the
/// sides — every arm captures its oracle before the wipe and issues no commit
/// after the restore — so no field key is excluded.
async fn assert_no_partial_object(
    harness: &CheckpointHarness,
    object_id: u64,
    expected: &Oracle,
    context: &str,
) -> ObjectPresence {
    assert_no_partial_object_ignoring(harness, object_id, expected, &[], context).await
}

/// [`assert_no_partial_object`], but with a list of `obj:*` keys the ARM ITSELF
/// planted before the restore.
///
/// Only the b.4et stream arms need it. Their fixture deliberately spares
/// `le:{oid}:obj:stream` — that is the whole condition under test — and the
/// stream is not materialized object state, so its presence alongside an absent
/// header is not the partially-applied shape this helper exists to exclude.
/// Passing it explicitly, rather than blanket-excluding the stream key for
/// every arm, keeps the default assertion at full strength: on every other arm
/// a surviving stream WOULD be a finding.
async fn assert_no_partial_object_ignoring(
    harness: &CheckpointHarness,
    object_id: u64,
    expected: &Oracle,
    ignored: &[String],
    context: &str,
) -> ObjectPresence {
    let keys: Vec<String> = obj_keys(harness, object_id)
        .await
        .into_iter()
        .filter(|key| !ignored.contains(key))
        .collect();
    let header = read_obj_header(harness, object_id).await;
    if header.is_empty() {
        assert!(
            keys.is_empty(),
            "{context}: `obj:hdr` is absent but {} other `obj:*` key(s) survive: {keys:?}. That \
             is a partially-applied object — the header is what apply's structural short-circuit \
             keys on, so state written without it is unreachable AND unrepairable",
            keys.len()
        );
        return ObjectPresence::Absent;
    }

    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .unwrap_or_else(|e| {
        panic!(
            "{context}: `obj:hdr` is present, so the object must be fully readable; \
             read_object_state_for_test failed: {e}"
        )
    });
    assert_faithful_columns(
        &restored,
        &expected.snapshot,
        FaithfulColumnsOptions::default(),
        context,
    );
    assert_eq!(
        restored.object_id, object_id,
        "{context}: the restore must materialize under the ORIGINAL object_id"
    );
    ObjectPresence::FullyRestored
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric helpers — before/after deltas on the zero-seeded families, never
// absolute values, and never a log-string assertion
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot every cell of `live_edit_checkpoint_restore_lineage_total` for a
/// before/after delta. The whole pinned vocabulary, never one cell: a step-down
/// that ticks the WRONG reason is a different defect from one that does not
/// tick at all, and only a full sweep tells them apart.
fn lineage_cells(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    CHECKPOINT_RESTORE_LINEAGE_OUTCOMES
        .iter()
        .map(|outcome| {
            (
                *outcome,
                read_restore_lineage_counter(&harness.metrics, outcome),
            )
        })
        .collect()
}

/// Assert the per-outcome deltas since `before`. Outcomes absent from
/// `expected` must be FLAT — that is what makes "a classified error does not
/// step down" (Test 1) assertable at all.
fn assert_lineage_deltas(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    expected: &[(&str, u64)],
    context: &str,
) {
    for (outcome, pre) in before {
        let want = expected
            .iter()
            .find(|(name, _)| name == outcome)
            .map(|(_, value)| *value)
            .unwrap_or(0);
        let got = read_restore_lineage_counter(&harness.metrics, outcome) - pre;
        assert_eq!(
            got, want,
            "{context}: lineage outcome {outcome:?} MUST tick {want} time(s), got {got}. This \
             family is NOT one-tick-per-attempt — one attempt that steps down twice before \
             resolving ticks three cells"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The fixture: a real-commit lineage whose replayed range carries an ext value
// ─────────────────────────────────────────────────────────────────────────────

/// How the three tail-range commits are packaged into durable tail generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailLayout {
    /// ONE tail covering `(S0, S0+3]`. The shape every arm but Test 6 uses.
    Single,
    /// THREE linked tails, one commit each: `(S0, S0+1]`, `(S0+1, S0+2]`,
    /// `(S0+2, S0+3]`, all rooted at `S0`. Gives the degraded-adapter arm
    /// distinct members to wedge and a chain long enough that one wedged member
    /// is re-reached by a later candidate.
    Chained,
}

/// Everything one matrix arm needs.
struct RestoreMatrixFixture {
    session_id: String,
    layer_name: String,
    object_id: u64,
    /// `S0` — the loop-written base's `snap_seq`, and the layer's live sequence
    /// when the base was assembled.
    s0: u64,
    /// `S` — the head of the replayed range.
    head_seq: u64,
    /// The live state at `S0`: what a step-down to the base must reproduce.
    s0_oracle: Oracle,
    /// The live state at `S`: what an uninterrupted composed restore must
    /// reproduce.
    head_oracle: Oracle,
    /// `(sequence_num, the OMPF that commit sent)` for the replayed range, in
    /// commit order — the exact bytes the durable tails carry.
    records: Vec<(u64, Bytes)>,
    /// Hex spelling of the external value's content hash — the ext key-schema
    /// input, and the only handle the arms need (the raw bytes are 74's and the
    /// AC-3 lane's subject, not this file's).
    ext_hash_hex: String,
    /// The `band` `apply_restore` will apply, derived from the landed formula
    /// `max(checkpoint_seq_safety_margin, observed_commit_span)` with the
    /// fixture's spans asserted below the margin, so it IS the margin.
    band: u64,
}

impl RestoreMatrixFixture {
    /// `seq` a restore of the composed lineage seeds: `restore_floor + band`,
    /// with `restore_floor == head_seq` because `wipe_layer_state` UNLINKs the
    /// stream too and the high-water probe folds to `None`.
    fn composed_seeded_seq(&self) -> u64 {
        self.head_seq + self.band
    }
}

/// CreateSession → CreateLayer → two pre-base commits → wait for the loop's base
/// → **fence the loop** → three tail-range commits (the middle one carrying an
/// external value) → hand-put the durable tail(s) from the exact OMPF bytes
/// those commits sent.
///
/// The fence (`cancel()` + a bounded join) is the shape
/// `e2e_composed_restore_test` uses: after it no tick exists to land a
/// generation, so retention cannot prune the fixture and no fresh base can
/// appear at a sequence the hand-put tails do not link onto.
///
/// The base is written BEFORE the ext-carrying commit on purpose. The assembler
/// serializes base OMPFs with zero external parts, so a base written after it
/// would carry the field but not the bytes; keeping the ext commit strictly
/// inside the replayed range is what makes `ComposedLineage::ext_values`
/// non-empty and the SR-3.4 seam reachable at all (Task-3 PM constraint).
///
/// Kept **file-local** deliberately: `helpers.rs` is off-limits to this Subtask
/// and sibling Task-6 lanes are authored in this directory in parallel, the
/// same posture `e2e_ext_value_recovery_test.rs`'s module header records.
async fn setup_matrix_fixture(
    harness: &mut CheckpointHarness,
    name: &str,
    layout: TailLayout,
) -> RestoreMatrixFixture {
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(&format!("{name}-session"), SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");
    let layer_name = format!("{name}-layer");

    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, &layer_name),
        make_create_data(ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed while building the lineage fixture")
        .into_inner()
        .object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    let mut committed_seq = 0;
    for index in 0..2 {
        let (seq, _) = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            &pre_base_commit_delta(index),
            &[],
            1_000 + index,
        )
        .await;
        committed_seq = seq;
    }

    // Gate on the committed sequence, not merely "any base": with
    // `interval = quiesce = 0` the loop can fire between CreateLayer and the
    // commits and land a generation that predates them, which would leave `S0`
    // non-deterministic and every link's arithmetic meaningless.
    let base_handle = wait_for_checkpoint_at_least(
        &harness.store,
        object_id,
        committed_seq,
        LOOP_BASE_DEADLINE_SECS,
    )
    .await;

    // ── The fence ────────────────────────────────────────────────────────
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, loop_task).await;

    let s0 = base_handle.snap_seq;
    assert_eq!(
        s0, committed_seq,
        "fixture precondition: the loop's base MUST have been assembled at the layer's live \
         sequence ({committed_seq}). A higher value means a commit landed that this fixture \
         does not know about, and a hand-put tail's `from_seq = S0` link would name a \
         predecessor covering commits nothing replays"
    );

    let base_meta = harness
        .store
        .get_layer_checkpoint(&base_handle)
        .await
        .expect("get_layer_checkpoint on the loop's base must succeed")
        .expect("the loop's base must be fetchable")
        .meta;

    let s0_oracle = capture_oracle(harness, object_id).await;
    assert_eq!(
        s0_oracle.snapshot.sequence_num, s0,
        "fixture precondition: the S0 oracle must be captured at the base's own sequence"
    );
    assert_eq!(
        header_u64(&s0_oracle.header, "seq"),
        s0_oracle.snapshot.sequence_num,
        "fixture precondition: the live obj:hdr and the decoded snapshot must agree on `seq`"
    );
    assert_eq!(
        s0_oracle
            .header
            .get("restore_seq_bump")
            .map(|raw| raw.as_str())
            .unwrap_or("0"),
        "0",
        "fixture precondition: a layer that has only ever been written live MUST carry no \
         guard band. A non-zero value means a restore already ran and every post-restore \
         assertion here would be reading a residue"
    );

    // ── The replayed range ───────────────────────────────────────────────
    let blob = ext_blob();
    let ext_hash = ext_content_hash(&blob);
    let ext_hash_hex = sha256_to_hex(&ext_hash);

    let deltas = [
        (tail_structural_delta(), Vec::new()),
        (tail_ext_value_delta(&ext_hash, blob.len()), vec![&blob[..]]),
        (tail_trailing_delta(), Vec::new()),
    ];
    let mut records: Vec<(u64, Bytes)> = Vec::with_capacity(deltas.len());
    let mut expected = s0 + 1;
    for (index, (delta, externals)) in deltas.iter().enumerate() {
        let record = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            delta,
            externals,
            2_000 + index as u64,
        )
        .await;
        assert_eq!(
            record.0, expected,
            "fixture precondition: commit sequences must be contiguous. Commit {index} \
             returned {}, expected {expected}. A gap would be rejected downstream as a \
             RecordSeqGap and the arm would silently assert the step-down path",
            record.0
        );
        expected += 1;
        records.push(record);
    }
    let head_seq = records.last().expect("three commits were driven").0;
    assert_eq!(
        head_seq,
        s0 + TAIL_RANGE_COMMITS,
        "fixture precondition: the replayed range must cover exactly {TAIL_RANGE_COMMITS} \
         commits above S0"
    );

    // Non-vacuity for the Task-3 PM constraint, asserted for EVERY arm and not
    // only the feature-gated seam one: the replayed range really does carry an
    // external value. `RESTORE_SEAM_AFTER_EXT_PRESTORE` sits inside
    // `if !ext_values.is_empty()`, so a fixture that silently lost its ext
    // commit would arm the seam, never fire it, and let the restore just
    // succeed. The live UpdateLayer path pre-flighted the value through
    // `store_external_values`, so both content-addressed keys exist right now;
    // every arm wipes them before restoring so no later presence claim can pass
    // on this residue.
    for key in [
        harness.tv.key_schema().ext_meta(&ext_hash_hex),
        harness.tv.key_schema().ext_data(&ext_hash_hex),
    ] {
        let present: bool = harness
            .tv
            .client()
            .pool
            .exists(&key)
            .await
            .expect("EXISTS on the ext key must succeed");
        assert!(
            present,
            "fixture precondition: `{key}` MUST exist after the ext-value commit. Without a \
             real external value in the replayed range the SR-3.4 seam is unreachable and the \
             seam arm goes vacuously green"
        );
    }

    let head_oracle = capture_oracle(harness, object_id).await;
    assert_eq!(
        head_oracle.snapshot.sequence_num, head_seq,
        "fixture precondition: the head oracle must be captured at the last commit's sequence"
    );

    // ── The durable tail generation(s) ───────────────────────────────────
    match layout {
        TailLayout::Single => {
            put_tail_generation_from_records(
                &harness.store,
                object_id,
                &session_id,
                &layer_name,
                s0,
                s0,
                &records,
                HAND_PUT_WRITTEN_AT_UNIX_MS,
                0,
            )
            .await;
        }
        TailLayout::Chained => {
            for (index, record) in records.iter().enumerate() {
                put_tail_generation_from_records(
                    &harness.store,
                    object_id,
                    &session_id,
                    &layer_name,
                    s0,
                    s0 + index as u64,
                    std::slice::from_ref(record),
                    HAND_PUT_WRITTEN_AT_UNIX_MS,
                    0,
                )
                .await;
            }
        }
    }

    // `band = max(margin, observed_commit_span)`. Both candidate spans are
    // asserted below the margin, so the band IS the margin and every
    // `seq == floor + band` assertion below stays exact without depending on
    // which meta sources the span.
    let margin = harness.config.checkpoint_seq_safety_margin;
    for (label, span) in [
        ("base", base_meta.observed_commit_span),
        ("replayed range", TAIL_RANGE_COMMITS),
    ] {
        assert!(
            span < margin,
            "fixture precondition: the {label} observed_commit_span ({span}) must stay below \
             the safety margin ({margin}), so `band = max(margin, span)` is the margin"
        );
    }

    RestoreMatrixFixture {
        session_id,
        layer_name,
        object_id,
        s0,
        head_seq,
        s0_oracle,
        head_oracle,
        records,
        ext_hash_hex,
        band: margin,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wiring + drive helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A harness whose checkpoint store is `fault_store` and whose external-value
/// store is a real `ValkeyExternalValueStore` over the harness's own
/// `TestValkey`.
///
/// The ext store has to be real: the fixture's ext-carrying `UpdateLayer` goes
/// through `store_external_values`, and an in-process `MockExternalValueStore`
/// short-circuits that against a map, creating none of the `le:ext:*` keys the
/// composed path later re-stores.
///
/// It is built through [`CheckpointHarnessOptions::ext_value_store_factory`]
/// rather than swapped onto `svc.ext_value_store` after construction. The factory
/// runs at the one point where the harness's `TestValkey` exists and nothing has
/// been wired yet, so the SAME `Arc` reaches the service AND the recovery hook.
/// The field swap it replaces reached only the service and left the harness's own
/// hook holding a different store — the reason every arm here used to install a
/// hand-built hook of its own.
///
/// `.with_checkpoint_store(..)` is deliberately NOT chained: arming the SR-3.2
/// lazy fetch-on-miss route would let a value the composed path failed to
/// pre-store silently self-heal out of the adapter, and the SR-3.4 eager
/// pre-store is the only route this file's arms mean to exercise.
async fn setup_matrix_harness(fault_store: &Arc<MatrixFaultStore>) -> CheckpointHarness {
    // `tails_off_options`, not `CheckpointHarnessOptions::new`: the tail flag is
    // pinned explicitly OFF rather than inherited, because the binary default is
    // `true` since b.3n1 and every arm below counts adapter CALLS through
    // `MatrixFaultStore` and asserts an exact durable shape at an injected
    // interruption. A tails-on harness plants an SR-7.1 bootstrap base at
    // CreateLayer and spawns an SR-3.6 re-base after any restore that lands —
    // both of them extra, asynchronous writers through the very store the call
    // counts and the fault seams key on. The re-base is `restore_rebase_test`'s
    // and `e2e_rebase_lifecycle_test`'s subject, not this file's; the one arm
    // here that does drive it builds an enabled `BaseBootstrap` by hand
    // (`a_failed_gap_rebase_leaves_a_complete_object_and_a_later_attempt_converges`).
    let mut opts = tails_off_options(TICK_SECS);
    opts.store = Some(Arc::clone(fault_store) as Arc<dyn CheckpointStore>);
    opts.ext_value_store_factory = Some(Box::new(|ctx: ExtStoreCtx<'_>| {
        // `.with_metrics(..)` because `main.rs` does — without it every
        // `gc_underflow_total` observation is silently dropped.
        Arc::new(
            ValkeyExternalValueStore::new(ctx.tv.client().clone(), ctx.tv.key_schema().clone())
                .with_metrics(Arc::clone(ctx.metrics)),
        ) as Arc<dyn ExternalValueStore>
    }));
    setup_checkpoint_harness_with_options(opts).await
}

// ── Recovery-hook posture ───────────────────────────────────────────────────
//
// Every arm restores through the harness's OWN `svc.recovery`. That hook is now
// the shipped shape: `build_recovery_hook_for_e2e` installs both builders
// `main.rs` installs, including `with_ext_value_store` over the single `Arc` the
// factory above returns, so a composed lineage carrying an external value
// pre-stores it instead of failing closed at the apply core's `Permanent` arm.
// (`harness_parity_test.rs` asserts that property directly; the arms below
// consume it.) The bootstrap the harness threads is its own, whose three-way gate
// is off because `setup_matrix_harness` pins the tail feature off — so no arm
// spawns an SR-3.6 re-base unless it asks for one.
//
// The three arms that need MORE than the stock hook — the SR-3.4 seam injector
// (Test 4), an explicitly-enabled re-base vehicle (Test 5), and the dialled-down
// budget `Config` (Test 6) — chain it through
// `CheckpointHarness::install_recovery_hook{,_with_config}`, which rebuilds from
// the harness's own store / layer store / config / metrics / ext store /
// bootstrap. Nothing in this file re-derives that wiring by hand.

/// A `StatLayerRequest` carrying the stub bearer token the harness's
/// `AllowAllPermissions` / `StubUserinfo` wiring expects.
fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Issue one `UpdateLayer` and return `(sequence_num, the OMPF bytes it sent)`.
///
/// Returning the sent bytes is the point: they are what
/// `put_tail_generation_from_records` encodes into the tail envelope, so the
/// artifact carries exactly what the live commit carried — including, for the
/// ext-value commit, the blob as OMPF Part 1.
async fn drive_commit(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: &OwnedDelta,
    external_values: &[&[u8]],
    client_timestamp: u64,
) -> (u64, Bytes) {
    let ompf = delta_to_ompf_bytes(delta, external_values);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf.clone()),
    ]);
    let sequence_num = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("UpdateLayer must succeed while building the lineage fixture")
        .into_inner()
        .sequence_num;
    (sequence_num, Bytes::from(ompf))
}

/// Wipe the layer's whole `obj:*` keyspace and the ext namespace for the
/// fixture's hash — the state every restore attempt in this file starts from.
///
/// Both surfaces, because a post-restore ext-presence assertion that passed on
/// the live upload's residue would prove nothing. `wipe_layer_state` leaves the
/// layer mapping intact, so apply's SR-2.4 identity guard still resolves.
async fn wipe_for_restore(harness: &CheckpointHarness, fixture: &RestoreMatrixFixture) {
    wipe_layer_state(&harness.tv, fixture.object_id).await;
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&fixture.ext_hash_hex)).await;
    assert!(
        obj_keys(harness, fixture.object_id).await.is_empty(),
        "wipe precondition: no `obj:*` key may survive, or the recovery branch would not fire \
         and every assertion below would read pre-wipe state"
    );
}

/// Section id of the planted residue member. Deliberately NOT one the composed
/// head carries, so a survivor is visible as content the checkpoint never
/// contained rather than hiding inside the restored set.
#[cfg(feature = "test-checkpoint-fault-injection")]
const RESIDUE_SECTION_ID: &str = "9999";

/// The `obj:*` keys one call to [`plant_header_less_residue`] created.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct PlantedResidue {
    index: String,
    record: String,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl PlantedResidue {
    /// Both keys, in the shape `EXISTS` and the `ignored` parameter of
    /// [`assert_no_partial_object_ignoring`] / [`assert_retry_converges_ignoring`]
    /// take.
    fn keys(&self) -> Vec<String> {
        vec![self.index.clone(), self.record.clone()]
    }
}

/// Plant the b.kf1 orphan shape the b.f4t wipe exists for: one index member
/// plus its per-record hash, under an object with **no header**.
///
/// Called after [`wipe_for_restore`], so the eleven-key residue probe finds
/// exactly this and the wipe takes its `WATCH` path rather than short-circuiting
/// `Clean` — which is what puts
/// [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`](live_edit::checkpoint::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC)
/// on the code path at all. Header-less is load-bearing in the other direction
/// too: `apply_restore`'s step-1 structural-presence gate must not short-circuit
/// the attempt as `AlreadyPresent` before the wipe is ever reached.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn plant_header_less_residue(
    harness: &CheckpointHarness,
    object_id: u64,
    section_id: &str,
) -> PlantedResidue {
    use fred::interfaces::SetsInterface;

    let oid_str = object_id.to_string();
    let planted = PlantedResidue {
        index: harness.tv.key_schema().obj_sections_index(&oid_str),
        record: harness.tv.key_schema().obj_section(&oid_str, section_id),
    };
    let _: () = harness
        .tv
        .client()
        .pool
        .sadd(&planted.index, section_id)
        .await
        .expect("SADD residue index member");
    let _: () = harness
        .tv
        .client()
        .pool
        .hset(&planted.record, [("name", "stale"), ("parent_id", "0")])
        .await
        .expect("HSET residue record");
    planted
}

// The metrics-handle installer this file's arms depend on —
// `helpers::wire_client_metrics` — is shared with
// `restore_write_fence_test.rs` and `e2e_object_id_preservation_test.rs`.
// Without it every
// `live_edit_storage_watch_voided_total{path="restore_residue_wipe"}` and
// `live_edit_checkpoint_restore_residue_wiped_total` assertion below would read
// a permanent zero and pass whatever the wipe did, because the residue wipe
// reports through `ValkeyClient::metrics_snapshot()`, which is `None` until the
// hooks are installed.
//
// That the wiring is live is proven inside the arms themselves rather than
// assumed: each of arms 11 / 12 / 13 ends on a converging retry whose wipe
// commits, and asserts `residue_wiped_metric` moved by exactly one there. A dead
// handle would fail that assertion, which is what keeps the "unchanged"
// assertions on the interrupted attempts from being vacuous.

/// Live value of one
/// `live_edit_storage_watch_voided_total{path="restore_residue_wipe",disposition}`
/// cell. The whole `path × disposition` grid is pre-initialised in
/// `Metrics::new`, so an absent series is a wiring fault, not a zero.
#[cfg(feature = "test-checkpoint-fault-injection")]
fn residue_watch_voided_metric(harness: &CheckpointHarness, disposition: &str) -> u64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(&harness.metrics),
        "live_edit_storage_watch_voided_total",
        &[
            ("path", "restore_residue_wipe"),
            ("disposition", disposition),
        ],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_storage_watch_voided_total{{path=\"restore_residue_wipe\", \
             disposition=\"{disposition}\"}} must be pre-initialised; an absent series means the \
             cell was renamed or dropped, not that it read zero"
        )
    }) as u64
}

/// The two dispositions the residue wipe's own code can tick on a store that
/// answers `CLIENT ID` — i.e. the two whose movement is a finding rather than an
/// environment report.
///
/// `witness_unavailable` / `fingerprint_unavailable` are deliberately excluded:
/// both describe a backing store that would not give the wipe a stable
/// in-`MULTI` `CLIENT ID` or a connection id, which is a property of the Valkey
/// under the test rather than of the arm, and pinning them would make these arms
/// fail on a proxied managed store for a reason they do not test.
/// `compensated` has no producer on this path at all.
#[cfg(feature = "test-checkpoint-fault-injection")]
const RESIDUE_WATCH_DISPOSITIONS_UNDER_TEST: [&str; 2] =
    ["aborted_pre_exec", "committed_unwatched"];

/// Assert that neither disposition in [`RESIDUE_WATCH_DISPOSITIONS_UNDER_TEST`]
/// moved.
///
/// `aborted_pre_exec` is the L2 pre-dispatch `Refuse` arm and
/// `committed_unwatched` is the L3 witness contradiction on a COMMITTED `EXEC`.
/// A tripped `EXEC` is neither: it is `WatchedExec::Conflict`, which the wipe
/// answers by re-reading the header and returning `PeerLandedHeader` or
/// `Transient` — ticking nothing. The seam's own `Err` arm unwatches and
/// propagates and likewise ticks nothing, even though it *does* release the
/// `WATCH`, which is what distinguishes it from the L2 refusal it otherwise
/// resembles.
#[cfg(feature = "test-checkpoint-fault-injection")]
fn assert_residue_watch_dispositions_flat(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    context: &str,
) {
    for (disposition, pre) in before {
        assert_eq!(
            residue_watch_voided_metric(harness, disposition),
            *pre,
            "{context}: storage_watch_voided_total{{path=\"restore_residue_wipe\", \
             disposition=\"{disposition}\"}} MUST NOT move. A tripped EXEC is a Conflict, not a \
             pre-EXEC abort and not an unfenced commit; ticking either cell here would report a \
             defence layer firing that did not"
        );
    }
}

/// Snapshot both cells of [`RESIDUE_WATCH_DISPOSITIONS_UNDER_TEST`] for a
/// before/after comparison.
#[cfg(feature = "test-checkpoint-fault-injection")]
fn residue_watch_dispositions(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    RESIDUE_WATCH_DISPOSITIONS_UNDER_TEST
        .iter()
        .map(|disposition| {
            (
                *disposition,
                residue_watch_voided_metric(harness, disposition),
            )
        })
        .collect()
}

/// Drive the production entry point and require it to succeed.
async fn restore_expecting_ok(
    harness: &CheckpointHarness,
    fixture: &RestoreMatrixFixture,
    context: &str,
) {
    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "{context}: StatLayer MUST restore and return Ok, got {:?}: {}. An UNAVAILABLE \
             means the restore errored; a NOT_FOUND means no lineage verified",
            status.code(),
            status.message()
        )
    });
}

/// Drive the production entry point and require it to fail UNAVAILABLE,
/// returning the `Status` so a caller can pin more.
///
/// UNAVAILABLE is itself the classification assertion, not merely "an error":
/// `handle_stat_layer` routes `RecoveryOutcome::Error` through
/// `ServiceError::from_recovery_error`, which maps a **`Transient`** onto
/// UNAVAILABLE and every other `CheckpointError` onto INTERNAL. So this helper
/// asserts `recovery_error_is_retriable()` answered `true` for whatever the
/// restore raised — which is the whole verdict for the arms whose failure is
/// produced by production code rather than by an injector-chosen error.
async fn restore_expecting_unavailable_status(
    harness: &CheckpointHarness,
    fixture: &RestoreMatrixFixture,
    context: &str,
) -> tonic::Status {
    let status = match handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    {
        Err(status) => status,
        Ok(response) => panic!(
            "{context}: an interrupted restore MUST surface an error, never Ok. StatLayer \
             returned {:?}",
            response.into_inner()
        ),
    };
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "{context}: the handler maps a Transient RecoveryOutcome::Error onto UNAVAILABLE \
         (retriable) and every other CheckpointError onto INTERNAL, so the code IS the \
         classification. Got {:?}: {}",
        status.code(),
        status.message()
    );
    status
}

/// [`restore_expecting_unavailable_status`], additionally requiring the surfaced
/// message to carry `marker` — pinned to THIS injection so the assertion cannot
/// pass on some other UNAVAILABLE the restore path could produce.
///
/// Usable only where the injector chose the error, so the marker is the test's
/// own string rather than a copy of production prose.
async fn restore_expecting_unavailable(
    harness: &CheckpointHarness,
    fixture: &RestoreMatrixFixture,
    marker: &str,
    context: &str,
) {
    let status = restore_expecting_unavailable_status(harness, fixture, context).await;
    assert!(
        status.message().contains(marker),
        "{context}: the surfaced status MUST carry THIS injection's source ({marker}); got: {}",
        status.message()
    );
}

/// Whether [`assert_retry_converges`] also asserts the composed path's two
/// restore-cursor columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CursorCheck {
    /// Assert `last_checkpoint_seq == S` and `seq == restore_floor + band` —
    /// the columns the composed restore itself stamps.
    ComposedRestore,
    /// Skip them. A hook carrying a live SR-3.6 re-base RE-STAMPS both on
    /// success (`obj:hdr seq == last_checkpoint_seq == seeded_seq`), so reading
    /// them after the restore returns is both a different claim and a race
    /// against a fire-and-forget task. That matrix is `t3.dfh.yj.4f.gw`'s;
    /// Test 5 asserts the base landing at `seeded_seq` instead.
    SkippedRebaseRestamps,
}

/// **The retry-convergence half of every arm.** The fault is already cleared by
/// the caller; this drives the SAME production entry point again over the SAME
/// lineage and asserts the exact composed head state, plus — when the arm's hook
/// carries no re-base — the two cursor columns the composed path is responsible
/// for.
///
/// `restore_seq_bump` is deliberately NOT asserted here even on the
/// [`CursorCheck::ComposedRestore`] arm: it is `t3.dfh.yj.4f.gw`'s column
/// (`restore_rebase_test.rs` owns the retired / unchanged matrix across the
/// re-base's persisted, skipped, failed and shed arms).
async fn assert_retry_converges(
    harness: &CheckpointHarness,
    fixture: &RestoreMatrixFixture,
    cursor: CursorCheck,
    context: &str,
) {
    assert_retry_converges_ignoring(harness, fixture, cursor, &[], context).await
}

/// [`assert_retry_converges`], but with the same planted-key allowance
/// [`assert_no_partial_object_ignoring`] documents — for the b.4et arms whose
/// fixture spares `le:{oid}:obj:stream`.
///
/// The expected seeded sequence is UNCHANGED by that survival, and not by
/// accident: the fixture's spared stream carries the head commit's own
/// `{head_seq}-0`, so the high-water probe folds in `head_seq` — exactly the
/// `restore_floor` the composed head already supplies. The arms therefore test
/// the refusal machinery without also moving the arithmetic the other five arms
/// pin.
async fn assert_retry_converges_ignoring(
    harness: &CheckpointHarness,
    fixture: &RestoreMatrixFixture,
    cursor: CursorCheck,
    ignored: &[String],
    context: &str,
) {
    assert!(
        obj_keys(harness, fixture.object_id)
            .await
            .into_iter()
            .all(|key| ignored.contains(&key)),
        "{context}: the retry must start from an absent object, or the structural-header short-circuit \
         answers AlreadyPresent and the convergence claim is vacuous"
    );

    let restored_pre = layer_recovery_metric(harness, "restored");
    let error_pre = layer_recovery_metric(harness, "error");
    let lineage_pre = lineage_cells(harness);

    restore_expecting_ok(harness, fixture, context).await;

    assert_eq!(
        layer_recovery_metric(harness, "restored") - restored_pre,
        1,
        "{context}: exactly ONE `restored` tick on the converging retry"
    );
    assert_eq!(
        layer_recovery_metric(harness, "error") - error_pre,
        0,
        "{context}: `error` MUST stay flat once the fault is cleared"
    );
    assert_lineage_deltas(
        harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        &format!("{context}: the cleared-fault retry"),
    );

    let presence = assert_no_partial_object(
        harness,
        fixture.object_id,
        &fixture.head_oracle,
        &format!("{context}: after the converging retry"),
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::FullyRestored,
        "{context}: the converging retry MUST materialize the object"
    );

    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("post-retry read_object_state_for_test must succeed");
    // b.2ti: what the restore PERSISTS on the three diff-FILTERED version
    // columns is uniformly `seeded_seq` — one write of the whole object — so the
    // per-commit spread the oracle carries is not a comparable column here.
    // `Section::first_version` is exempt from that re-stamp and comes back
    // preserved; the retry converges on `gathered_composed` (asserted just
    // above), so `ComposedLineage` splits it at the lineage root `fixture.s0`.
    // `composed_seeded_seq()` is `head_seq + band`, the same value the
    // `CursorCheck::ComposedRestore` arm asserts on the header below; it is
    // correct on BOTH arms because the SR-3.6 re-base re-stamps the header, not
    // the entries. The presence half of the old assertion survives: every entry
    // the replayed range wrote must exist after the retry. `fixture.s0` is
    // passed twice — presence-range floor and lineage root — which coincide on a
    // single-generation fixture (see the helper's rustdoc).
    assert_restored_entry_versions_flattened(
        &restored,
        &fixture.head_oracle.snapshot,
        RestoredVersionRanges {
            replayed_range_floor: fixture.s0,
            lineage_root: fixture.s0,
        },
        fixture.composed_seeded_seq(),
        RestoredFirstVersions::ComposedLineage,
        FaithfulColumnsOptions::default(),
        context,
    );

    if cursor == CursorCheck::SkippedRebaseRestamps {
        return;
    }

    let header = read_obj_header(harness, fixture.object_id).await;
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        fixture.head_seq,
        "{context}: last_checkpoint_seq MUST be the COMPOSED HEAD ({}), not the base's \
         snap_seq ({}). The base's value under-states what the restore materialized and seeds \
         the writer cursor below sequence numbers the replayed range already used",
        fixture.head_seq,
        fixture.s0
    );
    assert_eq!(
        header_u64(&header, "seq"),
        fixture.composed_seeded_seq(),
        "{context}: seq MUST be `restore_floor + band` = {} + {}. The stream was wiped with \
         the rest of `obj:*`, so the high-water probe folds to None and the floor is the \
         composed head alone",
        fixture.head_seq,
        fixture.band
    );
}

/// The base-only sibling of [`assert_retry_converges`]'s cursor assertions, for
/// the arms whose step-down verdict resolves on the base.
fn assert_base_only_cursor(
    header: &HashMap<String, String>,
    fixture: &RestoreMatrixFixture,
    context: &str,
) {
    assert_eq!(
        header_u64(header, "last_checkpoint_seq"),
        fixture.s0,
        "{context}: a step-down to the base restores S0 ({}), so last_checkpoint_seq must be \
         S0 — a composed head here would mean the broken candidate was used anyway",
        fixture.s0
    );
    assert_eq!(
        header_u64(header, "seq"),
        fixture.s0 + fixture.band,
        "{context}: seq MUST be S0 + band = {} + {}",
        fixture.s0,
        fixture.band
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 1 — mid-gather fetch failure, CLASSIFIED: propagates verbatim
// ═════════════════════════════════════════════════════════════════════════════

/// A classified adapter error raised during a lineage member fetch **propagates
/// verbatim** to `RecoveryOutcome::Error` — it does NOT become a step-down.
///
/// This is the landed contract, read out of `src/checkpoint/lineage.rs`:
/// `fetch_member` does `Ok(result) => result?`, and both `gather_restore_lineage`
/// and `LineageSelection::next_restorable` document "Store errors propagate
/// **verbatim**: `Transient`, `Permanent`, `NotImplemented`. A broken lineage is
/// never an error — it is a verdict." Stepping down on a transient error would
/// serve older state where a retry gets the newest, so the sharp assertion here
/// is that **no** `stepped_down_*` cell moves: the whole zero-seeded lineage
/// vocabulary is required flat.
///
/// The fault is aimed at the BASE's fetch, reached only after the head tail has
/// already been probed and its link followed — so the failure is genuinely
/// mid-gather rather than at the first call.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_classified_gather_fetch_error_propagates_and_writes_nothing() {
    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "gather-error", TailLayout::Single).await;

    wipe_for_restore(&harness, &fixture).await;
    store.arm_get(fixture.s0, CheckpointKind::Base, GetFault::Transient);

    // Marked HERE, not at zero: `setup_matrix_fixture` fetched the loop-written
    // base to read its meta, and counting that call as an injection hit would
    // make the "exactly once" assertions below read one too many.
    let fetch_mark = store.get_calls_mark();
    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let lineage_pre = lineage_cells(&harness);

    restore_expecting_unavailable(
        &harness,
        &fixture,
        GATHER_ERROR_MARKER,
        "a Transient raised on the base's fetch",
    )
    .await;

    // Non-vacuity: the targeted member really was addressed, and the head tail
    // was probed first (so this is mid-gather, not a first-call failure).
    assert_eq!(
        store.get_calls_since(fetch_mark, fixture.s0, CheckpointKind::Base),
        1,
        "the base's fetch MUST have been attempted exactly once — without it the injection \
         never fired and the assertions below are vacuous"
    );
    assert_eq!(
        store.get_calls_since(fetch_mark, fixture.head_seq, CheckpointKind::Tail),
        1,
        "the head tail MUST have been fetched BEFORE the base: the walk follows its header's \
         `from_seq` link to reach the base at all, which is what makes this failure mid-gather"
    );

    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "a propagated classified error MUST land on the `error` recovery-attempt cell exactly \
         once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing was materialized"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — a restorable lineage is on the store; the failure \
         is the injected backend error, not an absent lineage"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[],
        "a CLASSIFIED store error during a member fetch MUST NOT step down (landed contract: \
         lineage.rs `fetch_member` does `Ok(result) => result?`). Any `stepped_down_*` tick \
         here would mean DR served older state where a retry would have got the newest",
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        "interrupted by a classified gather error",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::Absent,
        "a gather that never produced a lineage cannot have written anything"
    );

    // ── Claim (b) ────────────────────────────────────────────────────────
    store.clear_get_faults();
    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        "after clearing the gather fetch fault",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 2 — a MISSING lineage member is a step-down, and a step-down is correct
// ═════════════════════════════════════════════════════════════════════════════

/// A member the listing holds but whose fetch answers `Ok(None)` — the
/// contract-documented concurrent-prune race — is a **broken link**, not an
/// error: the selection steps down to the next older restorable generation and
/// the restore SUCCEEDS on it.
///
/// Landed shape, read from `lineage.rs`: `fetch_member`'s
/// `let Some(blob) = fetched else` arm mints `LinkFact::Absent { at_fetch: true }`,
/// whose `LineageBreak::reason_label()` is `"stepped_down_link_broken"`.
///
/// This arm is what makes the distinction in Test 1 meaningful: the two faults
/// are one method call apart at the same instrument, and they land on opposite
/// verdicts. It also proves claim (a) on a path where the object DOES get
/// written — the state must equal the older generation's oracle exactly, never
/// a blend of the two.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_missing_lineage_member_steps_down_and_restores_the_older_generation() {
    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "missing-member", TailLayout::Single).await;

    wipe_for_restore(&harness, &fixture).await;
    store.arm_get(fixture.head_seq, CheckpointKind::Tail, GetFault::Absent);

    // See Test 1 on why the count is marked rather than absolute.
    let fetch_mark = store.get_calls_mark();
    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let lineage_pre = lineage_cells(&harness);

    restore_expecting_ok(
        &harness,
        &fixture,
        "a lineage whose head tail vanished between the listing and the fetch",
    )
    .await;

    assert_eq!(
        store.get_calls_since(fetch_mark, fixture.head_seq, CheckpointKind::Tail),
        1,
        "the vanished member MUST have been addressed exactly once — non-vacuity"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "a step-down is a SUCCESSFUL restore: exactly one `restored` tick"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat — a broken link is a verdict, never an error"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("stepped_down_link_broken", 1), ("gathered_base", 1)],
        "an `Ok(None)` on the head tail's fetch",
    );

    // ── Claim (a), on the path where the object IS written ───────────────
    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.s0_oracle,
        "stepped down to the base after a missing member",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::FullyRestored,
        "the step-down restore MUST materialize the OLDER generation completely"
    );
    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_base_only_cursor(&header, &fixture, "after a link-broken step-down");

    // ── Claim (b) ────────────────────────────────────────────────────────
    // The step-down already succeeded, so the retry needs the object gone again
    // — `apply_checkpoint`'s structural-header short-circuit would otherwise answer
    // `AlreadyPresent` and prove nothing about the restored generation.
    store.clear_get_faults();
    wipe_for_restore(&harness, &fixture).await;
    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        "after the pruned member became reachable again",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 3 — composition failure
// ═════════════════════════════════════════════════════════════════════════════

/// A tail whose record payload cannot be parsed makes the composition engine
/// reject the lineage `Permanent`, which the selection records as
/// `stepped_down_compose` and answers by restoring the next older lineage.
///
/// The corruption is planted at the RECORD level, not the envelope's: the
/// gather module never parses a record payload (it only peeks the 44-byte
/// header), so the lineage verifies and is handed to `compose_from_artifacts`,
/// which fails at `ComposeError::OmpfParse`. That is what makes this a
/// *composition* injection point rather than a decode one — `stepped_down_decode`
/// is a different cell and it must stay flat.
///
/// The verdict is a WARN-worthy step-down, never an `error` outcome: every
/// `ComposeError` is `Permanent` by construction (the engine does no I/O, so a
/// retry against the same bytes reaches the same verdict), so the loop continues
/// to the next candidate rather than surfacing an error the caller would retry.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_composition_failure_steps_down_without_writing_a_partial_object() {
    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "compose-fault", TailLayout::Single).await;

    // Re-put the SAME (object_id, head_seq, Tail) generation with one record's
    // OMPF replaced by unparseable bytes. Both shipped adapters are
    // last-writer-wins on an identical `(object_id, snap_seq, kind)`, so this
    // corrupts the lineage in place rather than adding a second candidate.
    let mut corrupted = fixture.records.clone();
    let last = corrupted.len() - 1;
    corrupted[last].1 = Bytes::from_static(CORRUPT_RECORD_PAYLOAD);
    put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.s0,
        fixture.s0,
        &corrupted,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    wipe_for_restore(&harness, &fixture).await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let lineage_pre = lineage_cells(&harness);

    restore_expecting_ok(
        &harness,
        &fixture,
        "a lineage whose head tail carries an unparseable record",
    )
    .await;

    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "a compose step-down still restores: exactly one `restored` tick"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat — a rejected composition is a step-down verdict, not an error"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("stepped_down_compose", 1), ("gathered_base", 1)],
        "an unparseable record payload. `stepped_down_decode` must stay flat: the envelope \
         header peeked fine and the rejection came from the COMPOSITION engine, not the codec",
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.s0_oracle,
        "stepped down to the base after a composition failure",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::FullyRestored,
        "the step-down restore MUST materialize the OLDER generation completely — a partial \
         replay of the tail up to the corrupt record would fail the column comparison"
    );
    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_base_only_cursor(&header, &fixture, "after a compose step-down");

    // ── Claim (b) ────────────────────────────────────────────────────────
    // "Clearing" a composition fault means repairing the artifact: the fault is
    // in the durable bytes, not in a live seam.
    put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.s0,
        fixture.s0,
        &fixture.records,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;
    wipe_for_restore(&harness, &fixture).await;
    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        "after the corrupt record was repaired",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 4 — the ext pre-store / object write seam (feature-gated)
// ═════════════════════════════════════════════════════════════════════════════

/// Op-targeted ONE-SHOT fault injector: fires at `target_op` exactly once, then
/// self-disarms.
///
/// The self-disarm IS the instrument for claim (b) here — the retry runs against
/// the same hook with the seam quiet, which is the "clear the fault, drive the
/// same production entry point again" shape every other arm expresses through
/// the store wrapper.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct OneShotSeamInjector {
    target_op: &'static str,
    armed: std::sync::atomic::AtomicBool,
    fired: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl OneShotSeamInjector {
    fn new(target_op: &'static str) -> Self {
        Self {
            target_op,
            armed: std::sync::atomic::AtomicBool::new(true),
            fired: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn fire_count(&self) -> usize {
        self.fired.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for OneShotSeamInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op == self.target_op && self.armed.swap(false, std::sync::atomic::Ordering::AcqRel) {
            self.fired.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Err(CheckpointError::Transient {
                source: SEAM_ERROR_MARKER.into(),
            });
        }
        Ok(())
    }
}

/// Marker on the seam injection's boxed source, so the status assertion is
/// pinned to THIS injection.
#[cfg(feature = "test-checkpoint-fault-injection")]
const SEAM_ERROR_MARKER: &str = "restore-matrix-ext-prestore-seam-fault";

/// SR-3.4 / AC-8: a restore interrupted **between the external-value pre-store
/// and the single `write_object`** writes no object, and the retry converges to
/// the exact composed head state.
///
/// Consumes Task 3's own named constant
/// [`RESTORE_SEAM_AFTER_EXT_PRESTORE`](live_edit::checkpoint::RESTORE_SEAM_AFTER_EXT_PRESTORE)
/// through the hook's landed builder `with_restore_fault_injector_for_test`.
/// **No second seam is invented anywhere in this file** — every other arm drives
/// a store wrapper.
///
/// # Narrowed to the convergence half (PM condition C6)
///
/// The **durable-pair** claim at this instant — that `obj:hdr` is absent while
/// the tail-carried ext hashes are already present, plus the accepted ref-count
/// posture — is `t3.dfh.yj.b7.74`'s
/// (`e2e_composed_restore_ext_order_test.rs`, Test 2), which fires the seam
/// exactly once for that purpose. This test asserts the file-wide
/// no-partial-object invariant and then the half nobody else makes: that after
/// the seam falls quiet the SAME hook, over the SAME lineage, converges to the
/// exact expected state and cursor columns.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_ext_prestore_seam_interrupt_converges_on_retry() {
    use live_edit::checkpoint::adapters::CheckpointFaultInjector;
    use live_edit::checkpoint::RESTORE_SEAM_AFTER_EXT_PRESTORE;

    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "seam-retry", TailLayout::Single).await;

    let injector = Arc::new(OneShotSeamInjector::new(RESTORE_SEAM_AFTER_EXT_PRESTORE));
    harness.install_recovery_hook(|hook| {
        hook.with_restore_fault_injector_for_test(
            Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>
        )
    });

    wipe_for_restore(&harness, &fixture).await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let lineage_pre = lineage_cells(&harness);

    restore_expecting_unavailable(
        &harness,
        &fixture,
        SEAM_ERROR_MARKER,
        "interrupted at RESTORE_SEAM_AFTER_EXT_PRESTORE",
    )
    .await;

    assert_eq!(
        injector.fire_count(),
        1,
        "the seam MUST have fired. A zero here means the lineage carried NO external value and \
         the seam — which sits inside `if !ext_values.is_empty()` — was armed but unreachable, \
         which would make this whole arm vacuous (Task-3 PM constraint)"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "the interrupted restore MUST land on the `error` recovery-attempt cell exactly once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing was materialized"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — the lineage composed fine; the failure is the seam"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        "the seam fires AFTER a successful gather + compose, so the lineage resolved composed \
         and no step-down occurred",
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        "interrupted at the SR-3.4 seam",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::Absent,
        "the seam fires BEFORE the single write_object MULTI/EXEC, so no object key may exist. \
         Once a structurally-complete `obj:hdr` exists apply's step-1 short-circuit answers AlreadyPresent for every \
         later attempt and the retry below could never repair anything"
    );

    // ── Claim (b) — the half this lane owns ──────────────────────────────
    // The injector self-disarmed on its single fire; nothing else changes.
    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        "after the seam fell quiet",
    )
    .await;
    assert_eq!(
        injector.fire_count(),
        1,
        "the seam MUST NOT fire again — the convergence above is over a quiet seam, not a \
         second injection that happened to be tolerated"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 10 (b.f4t) — an interrupt AFTER the residue wipe and BEFORE the write
// ═════════════════════════════════════════════════════════════════════════════

/// b.f4t: a restore interrupted **after the pre-write residue wipe and before
/// `write_object`** leaves nothing partially observable, and the retry
/// converges.
///
/// # Why this window needed its own arm
///
/// Before b.f4t the restore path had no destructive step: every arm of this
/// matrix interrupted a restore that had only READ. Step 3b changed that — the
/// wipe `UNLINK`s a surviving `obj:*` keyspace — so there is now an instant at
/// which the old state is gone and the new state is not yet written. That is a
/// state no earlier arm could produce, and the AC-8 claim has to be re-proven
/// against it: an interruption there must leave readers seeing a clean
/// `NotFound`, not a half-dissolved object.
///
/// # The seam is the existing one, deliberately
///
/// [`RESTORE_SEAM_AFTER_EXT_PRESTORE`] fires after step 3b's wipe and before
/// step 5's `write_object`, so it lands inside exactly the window this arm
/// names. No second seam is invented (see the module header).
///
/// # What each assertion adds
///
/// - **the residue is gone.** Positive evidence that the wipe COMMITTED before
///   the interrupt — without it the arm could pass on a restore that refused
///   before reaching step 3b at all, proving nothing about this window.
/// - **`ObjectPresence::Absent`.** The AC-8 core: wipe-then-die leaves the
///   object absent, which is one of the two admissible durable shapes. The
///   inadmissible one — some index members surviving under no header — is
///   exactly what a wipe that had torn halfway would look like.
/// - **the retry converges.** The wipe is idempotent by construction (every arm
///   that mutates runs while the object is header-less), so the second attempt
///   re-probes, finds a clean keyspace, and lands the exact composed head.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_interrupt_after_the_residue_wipe_leaves_nothing_observable_and_converges() {
    use live_edit::checkpoint::adapters::CheckpointFaultInjector;
    use live_edit::checkpoint::RESTORE_SEAM_AFTER_EXT_PRESTORE;

    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "residue-seam", TailLayout::Single).await;

    let injector = Arc::new(OneShotSeamInjector::new(RESTORE_SEAM_AFTER_EXT_PRESTORE));
    harness.install_recovery_hook(|hook| {
        hook.with_restore_fault_injector_for_test(
            Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>
        )
    });

    wipe_for_restore(&harness, &fixture).await;

    // The b.kf1 orphan shape the wipe exists for — see
    // [`plant_header_less_residue`].
    let residue = plant_header_less_residue(&harness, fixture.object_id, RESIDUE_SECTION_ID).await;

    restore_expecting_unavailable(
        &harness,
        &fixture,
        SEAM_ERROR_MARKER,
        "interrupted after the b.f4t residue wipe",
    )
    .await;

    assert_eq!(
        injector.fire_count(),
        1,
        "the seam MUST have fired — a zero means the restore refused before reaching this \
         window and the arm proves nothing about it"
    );

    let residue_alive = count_existing(&harness, residue.keys()).await;
    assert_eq!(
        residue_alive, 0,
        "the residue MUST already be gone when the seam fires. That is what places the \
         interrupt in the window this arm names — after the wipe committed, before the object \
         was written — rather than merely somewhere before the write"
    );

    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        "interrupted between the residue wipe and write_object",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::Absent,
        "wipe-then-interrupt must leave the object ABSENT. Surviving index members under no \
         header would be a torn wipe — and worse than the pre-fix merge, because a later \
         retry would union the restored set into whatever was left"
    );

    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        "after the residue wipe was interrupted and the seam fell quiet",
    )
    .await;
    assert_eq!(
        injector.fire_count(),
        1,
        "the seam MUST NOT fire again — the convergence above is over a quiet seam"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests 11–13 (b.f4t) — INSIDE the residue wipe's WATCH … EXEC window
// ═════════════════════════════════════════════════════════════════════════════

/// `created_by` on the header the injector lands as a peer replica's restore.
///
/// Deliberately different from [`CREATED_BY`], which is what this fixture's own
/// `write_object` would stamp. That difference is the sharpest "nothing was
/// written" witness in arm 11: `StatLayer` echoes `created_by` straight out of
/// the live header, so a response carrying [`CREATED_BY`] would mean this
/// replica's restore overwrote the peer's — the exact thing the fence exists to
/// prevent — while one carrying this value means the peer's header is what
/// served.
#[cfg(feature = "test-checkpoint-fault-injection")]
const PEER_CREATED_BY: &str = "restore-matrix-peer-replica";

/// Marker on the wipe-window seam's injected refusal (arm 13), so that arm's
/// status assertion is pinned to THIS injection.
#[cfg(feature = "test-checkpoint-fault-injection")]
const WIPE_SEAM_ERROR_MARKER: &str = "restore-matrix-wipe-window-seam-fault";

/// The 12 canonical `obj:hdr` fields — deliberately only those — standing in for
/// the header a peer replica's completed restore lands.
///
/// All twelve rather than the two
/// `apply::header_fields_are_structurally_complete` reads: seeding only the two
/// fields the predicate happens to read would pass whatever that predicate's
/// field list became, which is part of what is under test. No more than the
/// twelve, because the arms need the header only to look structurally complete.
/// A real peer's completed restore writes a superset: the children-index marker
/// `chs_complete`, the fence identity `commit_chain` and the three
/// checkpoint-cursor columns (`last_checkpoint_seq`, `last_checkpoint_ms`,
/// `restore_seq_bump`) as well.
///
/// So this fixture header is UNMARKED. Once the atomic snapshot read trusts
/// `obj:chs` only on marked objects (b.y3m Epic 2, Task 4), a read of an object
/// carrying this header takes the unmarked fallback path, where a real peer's
/// restored object would be marked.
#[cfg(feature = "test-checkpoint-fault-injection")]
fn peer_header_fields() -> Vec<(&'static str, String)> {
    vec![
        ("seq", "500".to_owned()),
        ("first_ver", "0".to_owned()),
        ("src_fmt", "usda".to_owned()),
        ("next_set_order", "0".to_owned()),
        ("session_id", "peer-session".to_owned()),
        ("layer_name", "peer-layer".to_owned()),
        ("created_by", PEER_CREATED_BY.to_owned()),
        ("created_at", "1700000000000".to_owned()),
        ("modified_at", "1700000000000".to_owned()),
        ("last_client_id", String::new()),
        ("last_client_ts", "0".to_owned()),
        ("last_server_ts", "1700000000000".to_owned()),
    ]
}

/// What [`WipeWindowInjector`] does when it fires.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WipeWindowAction {
    /// `HSET` all twelve canonical header fields, then return `Ok(())` — a peer
    /// replica's restore landing inside the window. The wipe's `EXEC` then trips
    /// and its post-conflict re-read finds a structurally-complete header.
    LandCompleteHeader,
    /// `HSET superseded=1` and nothing else, then return `Ok(())` — the most
    /// concrete real producer of a single-field header write, a racing
    /// `CreateLayer` overwrite. The `EXEC` trips on the same `WATCH`, but the
    /// re-read finds a header that is still structurally incomplete.
    LandSupersededFlagOnly,
    /// Return `Err(Transient)`, having written nothing — the seam's own refusal
    /// exit path.
    Refuse,
}

/// Fires once at
/// [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`](live_edit::checkpoint::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC),
/// then self-disarms so the converging retry runs over a quiet seam.
///
/// # Why it writes to Valkey from inside a synchronous callback
///
/// `CheckpointFaultInjector::maybe_inject` is sync, and the two `Ok(())` actions
/// have to put a real write on `obj:hdr` **while the wipe holds its `WATCH`** —
/// that is the entire point of the seam's position. `block_in_place` +
/// `Handle::block_on` is the sanctioned way to drive one await from a sync
/// callback on a multi-thread runtime (every test in this file is
/// `flavor = "multi_thread"`), and it cannot deadlock against the wipe: the
/// injected `HSET` goes to the multiplexed data pool while the wipe holds an
/// exclusive `txn_pool` slot, so the two never contend for a connection.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct WipeWindowInjector {
    client: live_edit::storage::client::ValkeyClient,
    header_key: String,
    action: WipeWindowAction,
    armed: std::sync::atomic::AtomicBool,
    fired: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl WipeWindowInjector {
    fn new(
        client: live_edit::storage::client::ValkeyClient,
        header_key: String,
        action: WipeWindowAction,
    ) -> Self {
        Self {
            client,
            header_key,
            action,
            armed: std::sync::atomic::AtomicBool::new(true),
            fired: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn fire_count(&self) -> usize {
        self.fired.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for WipeWindowInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        use live_edit::checkpoint::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC;

        if op != RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC
            || !self.armed.swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return Ok(());
        }
        self.fired.fetch_add(1, std::sync::atomic::Ordering::AcqRel);

        let fields = match self.action {
            WipeWindowAction::Refuse => {
                return Err(CheckpointError::Transient {
                    source: WIPE_SEAM_ERROR_MARKER.into(),
                })
            }
            WipeWindowAction::LandCompleteHeader => peer_header_fields(),
            WipeWindowAction::LandSupersededFlagOnly => vec![("superseded", "1".to_owned())],
        };
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let _: () = self
                    .client
                    .pool
                    .hset(&self.header_key, fields)
                    .await
                    .expect("the injected obj:hdr write must succeed inside the WATCH window");
            })
        });
        Ok(())
    }
}

/// Build the harness + fixture + planted residue every wipe-window arm starts
/// from, with `injector` installed on the harness's own recovery hook.
///
/// The hook is rebuilt through `CheckpointHarness::install_recovery_hook`, which
/// re-wires `svc.ext_value_store` — load-bearing rather than incidental, because
/// the seam's injector rides `ExtPrestore` and is therefore installed only when
/// the hook carries an external-value store. Each arm additionally asserts its
/// injector fired, so an inert seam fails the arm instead of passing it.
///
/// **Call it as `Box::pin(setup_wipe_window_arm(..)).await`.** An `async fn`'s
/// state is inlined into its caller's, and this one already carries
/// [`setup_matrix_fixture`]'s state plus the whole `CheckpointHarness` it
/// returns; awaited inline it overflows std's default 2 MiB test-thread stack before the
/// first round trip (observed: `has overflowed its stack` at ~0.1 s, on every
/// arm). Boxing moves that state to the heap. The other arms in this file
/// inline their setup instead, which is why they never met this.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn setup_wipe_window_arm(
    name: &str,
    action: WipeWindowAction,
) -> (
    CheckpointHarness,
    RestoreMatrixFixture,
    Arc<WipeWindowInjector>,
    PlantedResidue,
    String,
) {
    use live_edit::checkpoint::adapters::CheckpointFaultInjector;

    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, name, TailLayout::Single).await;
    wire_client_metrics(&harness).await;

    let header_key = harness
        .tv
        .key_schema()
        .obj_header(&fixture.object_id.to_string());
    let injector = Arc::new(WipeWindowInjector::new(
        harness.tv.client().clone(),
        header_key.clone(),
        action,
    ));
    harness.install_recovery_hook(|hook| {
        hook.with_restore_fault_injector_for_test(
            Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>
        )
    });

    wipe_for_restore(&harness, &fixture).await;
    let residue = plant_header_less_residue(&harness, fixture.object_id, RESIDUE_SECTION_ID).await;

    (harness, fixture, injector, residue, header_key)
}

/// The fixture's two content-addressed ext keys must NOT exist.
///
/// The SR-3.4 pre-store is step 4b — strictly AFTER the step-3b wipe — so an arm
/// that refused inside the wipe cannot have reached it. This is the second
/// independent "nothing was written" witness in each of arms 11 / 12 / 13, and
/// the one that would catch a refusal reported from the wrong step.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn assert_ext_values_not_prestored(
    harness: &CheckpointHarness,
    fixture: &RestoreMatrixFixture,
    context: &str,
) {
    let alive = count_existing(
        harness,
        vec![
            harness.tv.key_schema().ext_meta(&fixture.ext_hash_hex),
            harness.tv.key_schema().ext_data(&fixture.ext_hash_hex),
        ],
    )
    .await;
    assert_eq!(
        alive, 0,
        "{context}: neither ext key may exist. The SR-3.4 pre-store runs at step 4b, strictly \
         after the step-3b wipe, so a present blob means the restore got PAST the wipe and this \
         arm is not observing the window it names"
    );
}

/// After a converging retry: the planted residue was REPLACED, not merged.
///
/// Deliberately not "both planted keys are gone". The index key `obj:secs` is
/// legitimately alive again — `write_object` re-creates it holding the composed
/// head's sections — so the observable claim is about its CONTENTS plus the
/// per-record hash:
///
/// * `obj:sec:{RESIDUE_SECTION_ID}` must not exist. A surviving record is the
///   b.9fn orphan shape: unreachable from any index, and no retirement path can
///   enumerate it.
/// * the planted id must not be a member. Its presence is the pre-fix merge —
///   `write_object`'s `SADD` unioning the restored set into the survivor, after
///   which every snapshot read serves a section the checkpoint never contained.
/// * the index must be non-empty, or the two assertions above would hold for a
///   restore that materialized nothing.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn assert_planted_residue_replaced(
    harness: &CheckpointHarness,
    object_id: u64,
    residue: &PlantedResidue,
    context: &str,
) {
    assert_eq!(
        count_existing(harness, vec![residue.record.clone()]).await,
        0,
        "{context}: the planted per-record hash `{}` MUST be gone once the wipe commits",
        residue.record
    );
    let members = section_index_members(harness, object_id).await;
    assert!(
        !members.contains(&RESIDUE_SECTION_ID.to_string()),
        "{context}: obj:secs MUST equal exactly the restored set. The planted member surviving \
         alongside it is the pre-fix merge. Got {members:?}"
    );
    assert!(
        !members.is_empty(),
        "{context}: non-vacuity — the restored state carries sections, so an EMPTY obj:secs means \
         nothing was materialized and the two assertions above are trivially true"
    );
}

/// Read `SMEMBERS le:{oid}:obj:secs`, sorted.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn section_index_members(harness: &CheckpointHarness, object_id: u64) -> Vec<String> {
    use fred::interfaces::SetsInterface;

    let key = harness
        .tv
        .key_schema()
        .obj_sections_index(&object_id.to_string());
    let mut members: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(&key)
        .await
        .expect("SMEMBERS obj:secs must succeed");
    members.sort();
    members
}

/// **Arm 11.** A peer replica landing a structurally-COMPLETE `obj:hdr` inside
/// the wipe's `WATCH` … `EXEC` window is a **fence**: the `EXEC` trips, the wipe
/// deletes nothing, `apply_restore` reports `AlreadyPresent`, and the peer's
/// state is what serves the RPC.
///
/// # Why this needs the seam
///
/// The verdict is decided by the post-conflict re-read, not by the trip itself.
/// Reaching it requires a write to land on `obj:hdr` between the wipe's
/// post-`WATCH` re-read and its `EXEC` — a window made of five concurrent
/// enumerations and a `MULTI` queue, i.e. real round trips. A racing writer
/// would land somewhere arbitrary and yield `PeerLandedHeader` OR `Transient` OR
/// `Wiped` by luck;
/// [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`](live_edit::checkpoint::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC)
/// makes the choice deterministic.
///
/// # What each assertion adds
///
/// - **`StatLayer` answers Ok with the PEER's `created_by`.** The fence held AND
///   nothing overwrote it: this replica's own `write_object` would have stamped
///   [`CREATED_BY`].
/// - **`restored` +1, `error` flat.** `AlreadyPresent` counts as `restored` —
///   the pre-existing contract that keeps `sum by (result) == attempts` — and
///   the fence is emphatically not an error.
/// - **The residue and the peer's header both survive.** "Nothing was wiped" is
///   the only observation that separates a fence from a race the peer merely won
///   after the `UNLINK` had already committed.
/// - **`obj:secs` still holds ONLY the planted member.** `write_object`'s `SADD`
///   would have unioned the composed head's sections into it, so this is the
///   sharpest proof the materializer never ran.
/// - **No `last_checkpoint_seq`, no ext blobs.** The two columns steps 4b / 5
///   would have written.
/// - **Neither watch-voided disposition moves.** A tripped `EXEC` is a
///   `Conflict`, not the L2 pre-dispatch `Refuse` (`aborted_pre_exec`) and not an
///   unfenced commit (`committed_unwatched`).
/// - **Once the peer's header is lost again, the SAME residue is wiped and the
///   restore converges.** The fence left the keyspace in a state a later attempt
///   still handles, and the `+1` on the residue counter there is what proves the
///   "unchanged" readings above were taken through a live metrics handle rather
///   than an uninstalled one.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_landing_a_complete_header_inside_the_wipe_window_fences_the_restore() {
    let (harness, fixture, injector, residue, header_key) = Box::pin(setup_wipe_window_arm(
        "wipe-peer-header",
        WipeWindowAction::LandCompleteHeader,
    ))
    .await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let lineage_pre = lineage_cells(&harness);
    let wiped_pre = residue_wiped_metric(&harness);
    let dispositions_pre = residue_watch_dispositions(&harness);
    // b.h4m: this arm is the suite's only producer of the MIDDLE fence stage.
    let fenced_pre = restore_fenced_cells(&harness);

    let response = handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "a peer-landed header is a FENCE, not an error: apply reports AlreadyPresent, the \
             hook reports Restored, and StatLayer's retried header read finds the peer's header. \
             Got {:?}: {}",
            status.code(),
            status.message()
        )
    })
    .into_inner();

    // b.h4m: `stage="residue_wipe"` and nothing else. `presence_probe` moving
    // would mean the peer's header was already there when step 1 read it — i.e.
    // the seam is not driving the window this arm names — and `object_write`
    // moving would mean the attempt got past the wipe to the materialization,
    // which the `AlreadyPresent` verdict is supposed to have prevented.
    assert_restore_fenced_deltas(
        &harness,
        &fenced_pre,
        &[("residue_wipe", 1)],
        "a peer landing a complete header inside the wipe window: this arm's whole claim is that \
         the peer was noticed by the residue wipe's own WATCH",
    );

    assert_eq!(
        injector.fire_count(),
        1,
        "the wipe-window seam MUST have fired exactly once. A zero means the wipe never armed its \
         WATCH — the residue probe short-circuited Clean, or the injector never reached the wipe \
         because the hook carried no ext value store — and every assertion below would be vacuous"
    );
    assert_eq!(
        response.created_by_user_name, PEER_CREATED_BY,
        "StatLayer MUST serve the PEER's header. `{CREATED_BY}` here would mean this replica's \
         write_object overwrote a restore that had already landed"
    );

    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "AlreadyPresent counts as `restored` — the layer scope's result vocabulary is \
         `restored | not_recoverable | error` and the presence short-circuit has always used the \
         first"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat: a peer landing the restore is the fence working, not a failure"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — the lineage composed and the mapping resolved"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        "the wipe runs after the gather, so the lineage resolved composed before the fence",
    );

    // ── Nothing was wiped ────────────────────────────────────────────────
    assert_eq!(
        count_existing(&harness, residue.keys()).await,
        2,
        "the planted residue MUST survive untouched. Wiping under a peer's freshly landed header \
         would delete the state that peer just restored — the whole reason the tripped EXEC \
         re-reads instead of assuming"
    );
    assert_eq!(
        section_index_members(&harness, fixture.object_id).await,
        vec![RESIDUE_SECTION_ID.to_string()],
        "obj:secs must still hold ONLY the planted member. Any composed-head section here means \
         write_object's SADD ran, i.e. the AlreadyPresent verdict did not stop the materialization"
    );
    assert_eq!(
        residue_wiped_metric(&harness),
        wiped_pre,
        "live_edit_checkpoint_restore_residue_wiped_total ticks on the committed-EXEC arm ONLY; a \
         tick here would claim a wipe that must not have happened"
    );
    assert_residue_watch_dispositions_flat(
        &harness,
        &dispositions_pre,
        "a peer-landed header trips the EXEC",
    );

    // ── Nothing was written ──────────────────────────────────────────────
    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_eq!(
        header.get("created_by").map(String::as_str),
        Some(PEER_CREATED_BY),
        "the live header must still be the peer's, field for field; got {header:?}"
    );
    assert!(
        !header.contains_key("last_checkpoint_seq"),
        "`last_checkpoint_seq` is stamped by write_object's own MULTI/EXEC, so its presence would \
         mean step 5 ran after the fence said the layer was already present. Got {header:?}"
    );
    assert_ext_values_not_prestored(&harness, &fixture, "fenced by a peer-landed header").await;

    // ── The fence left a keyspace a later attempt still converges on ─────
    //
    // Simulating the peer's header being lost AGAIN (the b.kf1 orphan class, one
    // round later) is what re-opens the restore: with the header gone the wipe
    // finds the same residue it declined to touch and completes normally. The
    // `+1` below is also this arm's proof that the metric handle it has been
    // reading is live — see `wire_client_metrics`.
    let _: () = harness
        .tv
        .client()
        .pool
        .del(&header_key)
        .await
        .expect("DEL the peer's header to re-open the restore");

    assert_retry_converges_ignoring(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        &residue.keys(),
        "after the peer's header was lost again",
    )
    .await;
    assert_eq!(
        residue_wiped_metric(&harness) - wiped_pre,
        1,
        "the converging attempt DID wipe, so the residue counter must move exactly once — this is \
         also what proves the flat readings above were taken through a live metrics handle"
    );
    assert_planted_residue_replaced(
        &harness,
        fixture.object_id,
        &residue,
        "the residue the fence spared, once the wipe actually commits",
    )
    .await;
    assert_eq!(
        injector.fire_count(),
        1,
        "the seam MUST NOT fire again — the convergence above is over a quiet seam"
    );
}

/// **Arm 12.** A SINGLE-field write on `obj:hdr` inside the same window trips the
/// same `EXEC`, but is NOT evidence a peer landed anything: the verdict is
/// `Transient`, nothing is wiped, nothing is written, and a second attempt
/// converges.
///
/// # The defect this fences
///
/// Inferring `PeerLandedHeader` from the trip alone would report
/// `ApplyOutcome::AlreadyPresent` — and therefore
/// `checkpoint_recovery_attempts_total{result="restored"}` — for a restore that
/// never ran, handing the caller its own `NotFound` straight back while the
/// metric claimed a recovery. `HSET superseded=1` is the concrete producer: a
/// racing `CreateLayer` overwrite sets exactly that one field.
///
/// The second attempt is the other half of the claim. The wipe is idempotent by
/// construction — every arm that mutates runs while the object is header-less or
/// header-incomplete — so the re-drive must find the residue AND the partial
/// header, remove both, and land the exact composed head. A surviving
/// `superseded=1` would make the restored layer permanently unwritable on both
/// commit backends, which is why it is asserted gone by name.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_partial_header_write_inside_the_wipe_window_is_transient_and_converges() {
    let (harness, fixture, injector, residue, header_key) = Box::pin(setup_wipe_window_arm(
        "wipe-partial-header",
        WipeWindowAction::LandSupersededFlagOnly,
    ))
    .await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let wiped_pre = residue_wiped_metric(&harness);
    let dispositions_pre = residue_watch_dispositions(&harness);

    // UNAVAILABLE is the assertion: the handler maps a `Transient`
    // `RecoveryOutcome::Error` there and every other CheckpointError onto
    // INTERNAL. Nothing is pinned on the message, which is production prose the
    // injector did not choose — the state assertions below are what identify
    // this failure.
    let status = restore_expecting_unavailable_status(
        &harness,
        &fixture,
        "a single-field obj:hdr write inside the wipe's WATCH window",
    )
    .await;

    assert_eq!(
        injector.fire_count(),
        1,
        "the wipe-window seam MUST have fired exactly once, or the UNAVAILABLE above came from \
         somewhere else entirely"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat. A tick here is the exact metric dishonesty this arm fences: a \
         tripped EXEC read as `AlreadyPresent` reports a recovery that never ran"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "the Transient verdict lands on the `error` recovery-attempt cell exactly once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — the mapping still resolves; the wipe simply lost its \
         WATCH"
    );

    // ── Nothing was wiped, nothing was written ───────────────────────────
    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_eq!(
        header,
        HashMap::from([("superseded".to_owned(), "1".to_owned())]),
        "obj:hdr must carry EXACTLY the injected single field. More fields mean write_object ran \
         over a header the wipe was fencing; fewer mean the injected write never landed inside \
         the window and this arm observed nothing. (StatLayer said: {})",
        status.message()
    );
    assert_eq!(
        count_existing(&harness, residue.keys()).await,
        2,
        "the planted residue MUST survive: the EXEC tripped, so the UNLINK never committed"
    );
    assert_eq!(
        section_index_members(&harness, fixture.object_id).await,
        vec![RESIDUE_SECTION_ID.to_string()],
        "obj:secs must still hold ONLY the planted member — no composed-head section may have \
         been unioned in"
    );
    assert_eq!(
        residue_wiped_metric(&harness),
        wiped_pre,
        "nothing was wiped, so the residue counter must not move"
    );
    assert_residue_watch_dispositions_flat(
        &harness,
        &dispositions_pre,
        "a single-field write trips the EXEC",
    );
    assert_ext_values_not_prestored(&harness, &fixture, "tripped by a single-field header write")
        .await;

    // ── The idempotent re-drive ──────────────────────────────────────────
    let mut ignored = residue.keys();
    ignored.push(header_key.clone());
    assert_retry_converges_ignoring(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        &ignored,
        "after the wipe lost its WATCH to a single-field write",
    )
    .await;

    assert_eq!(
        residue_wiped_metric(&harness) - wiped_pre,
        1,
        "the re-drive's wipe DID commit, so the residue counter moves exactly once — this is also \
         what proves the flat readings above were taken through a live metrics handle"
    );
    assert_planted_residue_replaced(
        &harness,
        fixture.object_id,
        &residue,
        "the residue the re-drive found",
    )
    .await;
    let repaired = read_obj_header(&harness, fixture.object_id).await;
    assert!(
        !repaired.contains_key("superseded"),
        "a surviving `superseded=1` makes the restored layer permanently UNWRITABLE on both \
         commit backends, and write_object's HSET cannot remove it — only the wipe can. Got \
         {repaired:?}"
    );
    assert_eq!(
        injector.fire_count(),
        1,
        "the seam MUST NOT fire again — the convergence above is over a quiet seam"
    );
}

/// **Arm 13.** The seam's own refusal exit: an injected `Err` inside the window
/// releases the `WATCH` and propagates verbatim, having wiped and written
/// nothing; the retry converges.
///
/// # What is distinct here
///
/// This is the one arm where the wipe abandons a `WATCH` it armed **without**
/// dispatching anything — the same disposition as the L2 pre-dispatch `Refuse`
/// arm, reached one step later. The pinned consequence is that it does NOT tick
/// `storage_watch_voided_total{disposition="aborted_pre_exec"}`: that cell is the
/// L2 identity check refusing, and an operator watching it must not see this
/// test-only exit — nor, more importantly, must a future production refusal
/// added at this position be able to borrow the L2 cell's meaning silently.
///
/// The `WATCH` really was released is not asserted directly — a released watch
/// set has no observable spelling from outside the connection — but the
/// converging retry below takes a `txn_pool` slot and commits its own watched
/// `UNLINK`, which is the observable consequence a leaked watch set would
/// threaten.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refusal_inside_the_wipe_window_wipes_nothing_and_converges() {
    let (harness, fixture, injector, residue, _header_key) = Box::pin(setup_wipe_window_arm(
        "wipe-seam-refusal",
        WipeWindowAction::Refuse,
    ))
    .await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let wiped_pre = residue_wiped_metric(&harness);
    let dispositions_pre = residue_watch_dispositions(&harness);

    restore_expecting_unavailable(
        &harness,
        &fixture,
        WIPE_SEAM_ERROR_MARKER,
        "refused at the wipe's WATCH…EXEC seam",
    )
    .await;

    assert_eq!(
        injector.fire_count(),
        1,
        "the wipe-window seam MUST have fired exactly once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "the propagated injected error lands on the `error` recovery-attempt cell exactly once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing was materialized"
    );

    // ── Nothing wiped, nothing written ───────────────────────────────────
    let presence = assert_no_partial_object_ignoring(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        &residue.keys(),
        "refused inside the residue wipe's WATCH window",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::Absent,
        "the refusal is before the EXEC and long before write_object, so no header may exist"
    );
    assert_eq!(
        count_existing(&harness, residue.keys()).await,
        2,
        "the planted residue MUST survive: the seam refused before the UNLINK was dispatched, so \
         `nothing was wiped` is the contract, not merely the likely outcome"
    );
    assert_eq!(
        residue_wiped_metric(&harness),
        wiped_pre,
        "nothing was wiped, so the residue counter must not move"
    );
    assert_residue_watch_dispositions_flat(
        &harness,
        &dispositions_pre,
        "the seam refused and unwatched, which is NOT the L2 pre-dispatch Refuse arm",
    );
    assert_ext_values_not_prestored(&harness, &fixture, "refused at the wipe seam").await;

    // ── Retry converges over a quiet seam ────────────────────────────────
    assert_retry_converges_ignoring(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        &residue.keys(),
        "after the wipe-window seam fell quiet",
    )
    .await;
    assert_eq!(
        residue_wiped_metric(&harness) - wiped_pre,
        1,
        "the converging attempt's wipe commits, so the residue counter moves exactly once — also \
         this arm's proof that the flat readings above came from a live metrics handle"
    );
    assert_planted_residue_replaced(
        &harness,
        fixture.object_id,
        &residue,
        "the residue the refusal spared, once the wipe commits",
    )
    .await;
    assert_eq!(
        injector.fire_count(),
        1,
        "the seam MUST NOT fire again — the convergence above is over a quiet seam"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests 7–8 — the b.4et stream-adjudication seams (feature-gated)
// ═════════════════════════════════════════════════════════════════════════════

/// Op-targeted ALWAYS-ON fault injector: fires a `Transient` at `target_op`
/// every time, and is disarmed by the test between the interruption half and
/// the convergence half.
///
/// Distinct from [`OneShotSeamInjector`] because the b.4et seams sit INSIDE a
/// bounded retry: a one-shot would be absorbed by attempt 2 and the restore
/// would simply succeed, so the arm would prove nothing about the refusal. The
/// explicit `disarm()` is this file's usual "clear the fault, drive the same
/// production entry point again" shape, expressed on an injector that must fail
/// a whole budget rather than a single call.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct AlwaysSeamInjector {
    target_op: &'static str,
    armed: std::sync::atomic::AtomicBool,
    fired: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl AlwaysSeamInjector {
    fn new(target_op: &'static str) -> Self {
        Self {
            target_op,
            armed: std::sync::atomic::AtomicBool::new(true),
            fired: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn fire_count(&self) -> usize {
        self.fired.load(std::sync::atomic::Ordering::Acquire)
    }

    fn disarm(&self) {
        self.armed
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for AlwaysSeamInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op != self.target_op || !self.armed.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(());
        }
        self.fired.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Err(CheckpointError::Transient {
            source: B4ET_SEAM_ERROR_MARKER.into(),
        })
    }
}

/// Marker on the b.4et injections' boxed source, so the status assertions are
/// pinned to THIS injection.
#[cfg(feature = "test-checkpoint-fault-injection")]
const B4ET_SEAM_ERROR_MARKER: &str = "restore-matrix-b4et-seam-fault";

// The attempts the b.4et adjudication spends per Valkey round-trip
// (`RECOVERY_STREAM_PROBE_RETRY.max_attempts`) are the shared mirror
// `RECOVERY_STREAM_PROBE_ATTEMPTS` in `super::helpers`. This file used to carry
// its own copy under a second name (`B4ET_PROBE_ATTEMPTS`), which hid that it
// was the same production fact `e2e_object_id_preservation_test.rs` mirrored
// too. The helpers copy carries the mirror-not-import reasoning and names the
// unit test that pins the relationship.

// `stream_unresolved_metric` / `stream_unresolved_cells` /
// `assert_stream_unresolved_deltas` live in `super::helpers` (glob-imported at
// the top of this file) — same reasoning as [`lineage_cells`] for the sweep
// shape, and shared because `e2e_object_id_preservation_test.rs` carried a
// byte-identical copy.

/// [`wipe_for_restore`], sparing `le:{oid}:obj:stream`.
///
/// The b.4et arms need a surviving stream: it is the positive evidence that
/// turns an unreadable probe into a refusal rather than a benign fold to
/// `snap_seq`. Everything else — the `obj:*` state keys and the fixture's ext
/// namespace — goes exactly as [`wipe_for_restore`] takes it, so the restore
/// under test is the same composed restore every other arm drives.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn wipe_for_restore_sparing_stream(
    harness: &CheckpointHarness,
    fixture: &RestoreMatrixFixture,
) -> String {
    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, fixture.object_id).await;
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&fixture.ext_hash_hex)).await;
    assert_eq!(
        obj_keys(harness, fixture.object_id).await,
        vec![stream_key.clone()],
        "wipe precondition: the per-object stream and NOTHING else may survive, or the arm is \
         not testing the surviving-stream condition it claims to"
    );
    stream_key
}

/// AC-8 for the b.4et **probe** refusal: a restore that refuses because it
/// cannot read a surviving stream's high-water writes no object at all, and the
/// retry with the fault cleared converges on the exact composed head.
///
/// # Why this arm is not a restatement of the b.4et file's own coverage
///
/// `e2e_object_id_preservation_test.rs` owns the arm table — which combination
/// refuses, what it reports, what seq each proceeding arm produces. What it
/// does NOT establish is this file's invariant, and the refusal is the first
/// interruption point that is neither inside the lineage gather nor inside the
/// apply: it sits between them, after a full gather + compose has already
/// succeeded. So the "no partially-applied object" claim genuinely needs
/// re-proving here — and, unlike arms 1–6, the object's keyspace is NOT empty
/// at that moment (the spared stream is there), which is exactly the shape a
/// naive absence check would misread.
///
/// The convergence half is the other reason: the refusal is `Transient`, so it
/// advertises itself as retriable. This arm is what makes that advertisement
/// true rather than merely stated — with the seam quiet the probe reads the
/// spared stream's own `{head_seq}-0`, folds it into a floor the composed head
/// already supplied, and the restore lands at the same `seeded_seq` the other
/// composed arms assert.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_stream_probe_seam_refusal_writes_nothing_and_converges_on_retry() {
    use live_edit::checkpoint::adapters::CheckpointFaultInjector;
    use live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE;

    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "b4et-probe", TailLayout::Single).await;

    let injector = Arc::new(AlwaysSeamInjector::new(
        RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE,
    ));
    harness.install_recovery_hook(|hook| {
        hook.with_restore_fault_injector_for_test(
            Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>
        )
    });

    let stream_key = wipe_for_restore_sparing_stream(&harness, &fixture).await;
    let ignored = vec![stream_key.clone()];

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let lineage_pre = lineage_cells(&harness);
    let unresolved_pre = stream_unresolved_cells(&harness);

    restore_expecting_unavailable(
        &harness,
        &fixture,
        "probe_unreadable",
        "refused at the b.4et high-water probe",
    )
    .await;

    assert_eq!(
        injector.fire_count(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the seam MUST fire once per bounded-retry attempt. One fire means the retry is absent \
         (a single reconnect would then cost a restore); more than {RECOVERY_STREAM_PROBE_ATTEMPTS} means \
         the budget is unbounded and the refusal is paid for under the per-object admission mutex"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "the refused restore MUST land on the `error` recovery-attempt cell exactly once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing was materialized"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — the lineage resolved fine; the failure is the probe"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        "the b.4et probe runs AFTER a successful gather + compose, so the lineage resolved \
         composed and no step-down occurred",
    );
    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_pre,
        &[("probe_unreadable", 1)],
        "a probe that could not read a surviving stream",
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object_ignoring(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        &ignored,
        "refused at the b.4et high-water probe",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::Absent,
        "the refusal happens BEFORE apply_restore is called at all, so no object key may exist. \
         Pre-b.4et this path folded to `None` and materialized the object at `snap_seq + band` \
         underneath the very stream it could not read"
    );

    // ── Claim (b) — the fault is genuinely retriable ─────────────────────
    injector.disarm();
    assert_retry_converges_ignoring(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        &ignored,
        "after the b.4et probe seam fell quiet",
    )
    .await;
    assert_eq!(
        injector.fire_count(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the seam MUST NOT fire again — the convergence above is over a quiet seam, not a second \
         injection that happened to be tolerated"
    );
}

/// AC-8 for the b.4et **`UNLINK`** refusal: the same two claims, one step later
/// in the adjudication.
///
/// The probe SUCCEEDS here and reads a top the plausibility gate refuses (an
/// auto-ID `deleted` tombstone — production's own shape, whose wall-clock ID
/// lands three orders of magnitude above the bound); the SR-2.4 identity guard
/// authorizes the cleanup; and every `UNLINK` attempt is injected.
///
/// The extra assertion this arm carries over arm 7 is that the stale stream is
/// STILL THERE afterwards. That is not incidental — it is why the restore was
/// refused. Pre-b.4et the `UNLINK` reply was bound as `()` and its errors were
/// swallowed, so the restore proceeded and seeded a header underneath exactly
/// this surviving top: a layer that comes back readable and permanently
/// unwritable.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_stale_stream_unlink_seam_refusal_writes_nothing_and_converges_on_retry() {
    use fred::interfaces::StreamsInterface;
    use live_edit::checkpoint::adapters::CheckpointFaultInjector;
    use live_edit::checkpoint::RECOVERY_SEAM_STALE_STREAM_UNLINK;

    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "b4et-unlink", TailLayout::Single).await;

    let injector = Arc::new(AlwaysSeamInjector::new(RECOVERY_SEAM_STALE_STREAM_UNLINK));
    harness.install_recovery_hook(|hook| {
        hook.with_restore_fault_injector_for_test(
            Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>
        )
    });

    let stream_key = wipe_for_restore_sparing_stream(&harness, &fixture).await;
    let ignored = vec![stream_key.clone()];

    // The tombstone, in exactly the shape `LayerStore::delete_layer` writes it:
    // auto-ID (so Valkey mints a wall-clock entry ID), one `payload=deleted`
    // field. That ID is what the plausibility gate refuses, which is the only
    // way to reach the `UNLINK` at all.
    let tombstone_id: String = harness
        .tv
        .client()
        .pool
        .xadd(
            &stream_key,
            false,
            None,
            "*",
            vec![("payload", fred::types::Value::from("deleted"))],
        )
        .await
        .expect("XADD of the `deleted` tombstone must succeed");

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let lineage_pre = lineage_cells(&harness);
    let unresolved_pre = stream_unresolved_cells(&harness);

    restore_expecting_unavailable(
        &harness,
        &fixture,
        "unlink_unverified",
        "refused at the b.4et stale-stream UNLINK",
    )
    .await;

    assert_eq!(
        injector.fire_count(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the UNLINK gets the same bounded retry budget as the probe — a removal that fails once \
         must not cost a restore"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "the refused restore MUST land on the `error` recovery-attempt cell exactly once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing was materialized"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        "the UNLINK runs after a successful gather + compose",
    );
    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_pre,
        &[("unlink_unverified", 1)],
        "a stale-stream removal that could not be verified",
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object_ignoring(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        &ignored,
        "refused at the b.4et stale-stream UNLINK",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::Absent,
        "the refusal happens before apply_restore, so no object key may exist"
    );

    let survived: bool = harness
        .tv
        .client()
        .pool
        .exists(&stream_key)
        .await
        .expect("EXISTS obj:stream must succeed");
    assert!(
        survived,
        "the stale stream (tombstone {tombstone_id}) MUST still be present — that is precisely \
         why the restore was refused. Pre-b.4et the swallowed UNLINK error let the restore \
         proceed and seed a header underneath this very top"
    );

    // ── Claim (b) ────────────────────────────────────────────────────────
    // With the seam quiet the gate's UNLINK lands, so the stream is gone by the
    // time the retry's assertions run — which is why the convergence half still
    // passes the ignore list: it is consulted BEFORE the retry, when the stream
    // is still there.
    injector.disarm();
    assert_retry_converges_ignoring(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        &ignored,
        "after the b.4et UNLINK seam fell quiet",
    )
    .await;
    assert_eq!(
        injector.fire_count(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the seam MUST NOT fire again during the converging retry"
    );
}

/// Arm 9's injector: the probe seam answered `Transient` forever (which is what
/// makes the classification `TYPE` reachable) and the `TYPE` seam answered with
/// a TERMINAL kind.
///
/// A separate type rather than a mode on [`AlwaysSeamInjector`] because the two
/// ops carry different KINDS, and that difference is the arm's own claim: the
/// terminal `TYPE` must stop its retry dead while the transient probe spends
/// its whole budget, so a single-kind injector could not express the case.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct ProbeAndTypeSeamInjector {
    armed: std::sync::atomic::AtomicBool,
    probe_fired: std::sync::atomic::AtomicUsize,
    type_fired: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl ProbeAndTypeSeamInjector {
    fn new() -> Self {
        Self {
            armed: std::sync::atomic::AtomicBool::new(true),
            probe_fired: std::sync::atomic::AtomicUsize::new(0),
            type_fired: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn disarm(&self) {
        self.armed
            .store(false, std::sync::atomic::Ordering::Release);
    }

    fn probe_fires(&self) -> usize {
        self.probe_fired.load(std::sync::atomic::Ordering::Acquire)
    }

    fn type_fires(&self) -> usize {
        self.type_fired.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for ProbeAndTypeSeamInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if !self.armed.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(());
        }
        if op == live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE {
            self.probe_fired
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Err(CheckpointError::Transient {
                source: B4ET_SEAM_ERROR_MARKER.into(),
            });
        }
        if op == live_edit::checkpoint::RECOVERY_SEAM_STREAM_KEY_TYPE {
            self.type_fired
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Err(CheckpointError::Permanent {
                source: B4ET_SEAM_ERROR_MARKER.into(),
            });
        }
        Ok(())
    }
}

/// AC-8 for the b.4et **`TYPE` classification** refusal — the arm where the
/// round-trip that decides what an unreadable probe MEANS is itself unreadable.
///
/// Same two claims as arms 7 and 8, at the third and last interruption point
/// the b.4et adjudication adds. It earns its own row because it is the only one
/// whose fault is TERMINAL rather than transient: the interruption is not a
/// blip the enclosing retry can absorb, so "nothing partial survived" and "the
/// retry converges once cleared" are being asserted across a hard failure
/// rather than a slow one.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_stream_key_type_seam_refusal_writes_nothing_and_converges_on_retry() {
    use live_edit::checkpoint::adapters::CheckpointFaultInjector;

    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "b4et-keytype", TailLayout::Single).await;

    let injector = Arc::new(ProbeAndTypeSeamInjector::new());
    harness.install_recovery_hook(|hook| {
        hook.with_restore_fault_injector_for_test(
            Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>
        )
    });

    let stream_key = wipe_for_restore_sparing_stream(&harness, &fixture).await;
    let ignored = vec![stream_key.clone()];

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let lineage_pre = lineage_cells(&harness);
    let unresolved_pre = stream_unresolved_cells(&harness);

    restore_expecting_unavailable(
        &harness,
        &fixture,
        "probe_unreadable",
        "refused at the b.4et TYPE classification",
    )
    .await;

    assert_eq!(
        injector.probe_fires(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the transient probe seam must still spend its whole budget — that is what makes the \
         classification TYPE reachable at all"
    );
    assert_eq!(
        injector.type_fires(),
        1,
        "a TERMINAL fault MUST stop the TYPE retry dead. Retrying a permanently-failing \
         round-trip only lengthens the per-object admission-mutex hold for a verdict that \
         cannot change"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "the refused restore MUST land on the `error` recovery-attempt cell exactly once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing was materialized"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        "the TYPE classification runs after a successful gather + compose",
    );
    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_pre,
        &[("probe_unreadable", 1)],
        "a classification TYPE that could not be read",
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object_ignoring(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        &ignored,
        "refused at the b.4et TYPE classification",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::Absent,
        "the refusal happens before apply_restore, so no object key may exist"
    );

    // ── Claim (b) ────────────────────────────────────────────────────────
    injector.disarm();
    assert_retry_converges_ignoring(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        &ignored,
        "after the b.4et TYPE seam fell quiet",
    )
    .await;
    assert_eq!(
        injector.type_fires(),
        1,
        "the seam MUST NOT fire again during the converging retry"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 5 — SR-3.6 gap re-base PUT failure
// ═════════════════════════════════════════════════════════════════════════════

/// A failing SR-3.6 post-restore re-base leaves the RESTORED OBJECT complete and
/// consistent — the re-base is a spawned best-effort task, not part of the
/// restore — and a later attempt with the fault cleared lands the base at
/// `seeded_seq` and converges.
///
/// # Instrument
///
/// Task 4 landed **no** `CheckpointFaultInjector` seam on the re-base, so the
/// PUT is failed through the store wrapper, keyed on `(seeded_seq, Base)`. That
/// targeting is what the op-label injector structurally cannot do: the re-base
/// PUT and the checkpoint loop's share the `"put_layer_checkpoint"` label.
/// `seeded_seq` is known in advance from the landed formula — `restore_floor +
/// band`, with the floor equal to the composed head because the wipe takes the
/// stream too — and the arm asserts the PUT was actually attempted at that
/// sequence, so a mis-targeted arm fails loudly instead of passing vacuously.
///
/// # Fence
///
/// The re-base **cursor-column matrix** (`restore_seq_bump` retirement, the
/// stamped `last_checkpoint_seq` / `last_checkpoint_ms` across the persisted /
/// skipped / failed / shed arms, and the `refused_seq_gap` capture consequence)
/// is `t3.dfh.yj.4f.gw`'s in `restore_rebase_test.rs`. Nothing here re-asserts
/// it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_gap_rebase_leaves_a_complete_object_and_a_later_attempt_converges() {
    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "rebase-put", TailLayout::Single).await;

    // The SR-3.6 vehicle, wired the way `main.rs` wires it but constructed here
    // with `enabled = true` explicitly rather than through the harness's tail
    // knobs: the harness's own bootstrap gate is derived from
    // `checkpoint_tail_enabled`, and flipping that would also change what the
    // checkpoint loop writes and therefore what `S0` means.
    //
    // It is installed through `install_recovery_hook` all the same, so the rest
    // of the wiring (store, layer store, config, metrics, and the ONE ext store
    // the service holds) stays the harness's own rather than being re-forked
    // around this one substitution. `with_base_bootstrap` assigns rather than
    // rejects, so the enabled unit below simply replaces the harness's disabled
    // one on the rebuilt hook. `harness.base_bootstrap` — the `Arc` `CreateLayer`
    // holds — is deliberately NOT the unit under test here: the fixture's
    // CreateLayer must keep running under the harness's own gate.
    let bootstrap = Arc::new(BaseBootstrap::new(
        Arc::clone(&harness.store),
        harness.tv.client().clone(),
        harness.tv.key_schema().clone(),
        harness.session_checkpoint_tracker.clone(),
        Arc::clone(&harness.metrics),
        harness.config.service_label.clone(),
        true,
    ));
    harness.install_recovery_hook(|hook| hook.with_base_bootstrap(Arc::clone(&bootstrap)));

    let seeded_seq = fixture.composed_seeded_seq();
    wipe_for_restore(&harness, &fixture).await;
    store.arm_base_put_failure(seeded_seq);

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let writes_error_pre = read_layer_writes_counter(&harness.metrics, "error");

    restore_expecting_ok(
        &harness,
        &fixture,
        "a composed restore whose SR-3.6 re-base PUT is failed",
    )
    .await;
    // The re-base is fire-and-forget; drain before observing its effects.
    wait_for_writer_idle(
        &harness.session_checkpoint_tracker,
        TRACKER_IDLE_DEADLINE_SECS,
    )
    .await;

    assert_eq!(
        store.base_put_attempts_at(seeded_seq),
        1,
        "the re-base MUST have attempted a Base PUT at seeded_seq ({seeded_seq}) — without it \
         the fault never fired and every assertion below is vacuous. A zero here also means \
         the bootstrap gate or the spawn point moved"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "the RESTORE is unaffected by the re-base failure: exactly one `restored` tick. The \
         re-base is spawned, never awaited, so a failing PUT cannot fail the restore"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat — the recovery attempt itself succeeded"
    );
    assert_eq!(
        read_layer_writes_counter(&harness.metrics, "error") - writes_error_pre,
        1,
        "the failed re-base MUST be observable on checkpoint_writes_total{{result=\"error\", \
         kind=\"layer\"}} — a best-effort task that fails silently is the posture SR-7.2 \
         forbids"
    );

    // ── Claim (a): the restore itself is whole ───────────────────────────
    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        "after a composed restore whose re-base PUT failed",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::FullyRestored,
        "a failed re-base must leave the restored object COMPLETE — it runs strictly after the \
         single write_object MULTI/EXEC has committed"
    );
    let generations = harness
        .store
        .list_layer_checkpoints(fixture.object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        !generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.seq() == seeded_seq),
        "no base may exist at seeded_seq ({seeded_seq}) while the PUT is failing; got {:?}",
        generations
            .iter()
            .map(|h| (h.seq(), h.kind.as_str()))
            .collect::<Vec<_>>()
    );

    // ── Claim (b) ────────────────────────────────────────────────────────
    // The next restore opportunity for this layer: wipe and re-drive. The
    // lineage is unchanged and the stream is wiped again, so `seeded_seq` is
    // the same number, which is what makes "the base lands where the failed
    // attempt would have put it" an exact assertion.
    //
    // `CursorCheck::SkippedRebaseRestamps`: this is the ONE arm whose hook
    // carries a live re-base, and a successful one re-stamps `obj:hdr` to
    // `seq == last_checkpoint_seq == seeded_seq`. Reading those columns after
    // the restore returns would be both a different claim (gw's matrix) and a
    // race against a fire-and-forget task. The convergence evidence this arm
    // owns is the state equality above plus the base landing below.
    store.clear_base_put_failure();
    wipe_for_restore(&harness, &fixture).await;
    let writes_success_pre = read_layer_writes_counter(&harness.metrics, "success");
    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::SkippedRebaseRestamps,
        "after clearing the re-base PUT fault",
    )
    .await;
    wait_for_writer_idle(
        &harness.session_checkpoint_tracker,
        TRACKER_IDLE_DEADLINE_SECS,
    )
    .await;

    assert_eq!(
        read_layer_writes_counter(&harness.metrics, "success") - writes_success_pre,
        1,
        "the cleared-fault re-base MUST tick checkpoint_writes_total{{result=\"success\", \
         kind=\"layer\"}} exactly once"
    );
    let generations = harness
        .store
        .list_layer_checkpoints(fixture.object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.seq() == seeded_seq),
        "the converging attempt MUST land the SR-3.6 base at seeded_seq ({seeded_seq}) — that \
         is what closes the gap the restore opened; got {:?}",
        generations
            .iter()
            .map(|h| (h.seq(), h.kind.as_str()))
            .collect::<Vec<_>>()
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 6 — degraded adapter: TWO wedged members above a healthy lineage
// ═════════════════════════════════════════════════════════════════════════════

/// A backend whose fetches never return costs the selection **at most one
/// reserve per wedged MEMBER, not one per candidate that reaches it**, and the
/// selection still reaches its answer inside the outer budget.
///
/// "At most", not "one": post-b.bu8 an allowanced `Tail` is capped at
/// `min(reserve, remaining - reserve)`, so a wedged member spends a WHOLE
/// reserve only while more than one reserve is still standing above the base's
/// reservation. That holds for both wedges below precisely because
/// [`WEDGE_BUDGET_MS`] gives this arm eight reserves; it is a property of these
/// knobs, not the general claim. At the shipped two-reserve ratio the same
/// member is cut short or declined un-fetched — Test 14 and the unit lane's
/// `a_slow_but_healthy_tail_is_cut_short_to_protect_the_bases_reserve` own that
/// side. What is unqualified here, and what this arm actually proves, is the
/// PER-MEMBER half: the cost is bounded by the number of wedged members, never
/// by the number of candidates that reach them.
///
/// # The topology, and what each candidate costs
///
/// Three linked tails over a base: `T3 → T2 → T1 → base`. `T2` and `T1` are
/// wedged. The landed walk (`lineage.rs`) then produces:
///
/// 1. candidate `T3` — probes `T3` (healthy), follows its `from_seq` link to
///    `T2`, whose fetch eats a whole reserve → `stepped_down_budget`;
/// 2. candidate `T2` — the overrun was cached selection-wide as
///    `LinkFact::Wedged`, so this candidate steps down at **zero fetches**.
///    That is the property the arm exists for: without the cached fact a single
///    wedged member is re-fetched once per candidate that reaches it, and the
///    outer preemptive timeout fires before the floor is ever addressed;
/// 3. candidate `T1` — a second, distinct wedged member: one more reserve →
///    `stepped_down_budget`;
/// 4. candidate `base` — healthy, funded, and the answer.
///
/// Three `stepped_down_budget` ticks, two reserves of wall clock, and a restore.
/// The caching claim is asserted on the wrapper's own call log rather than on
/// elapsed time, so it is deterministic under load.
///
/// # Why the budget knobs are dialled down here — and why the RATIO is not
///
/// `WEDGE_RESERVE` / `WEDGE_BUDGET_MS` replace the 30 s / 60 s production
/// numbers on this hook's `Config` alone. Scaling is forced: two wedged members
/// at the production reserve would cost 60 s of real wall clock and sit close to
/// nextest's 120 s hard kill. The knobs are the two the landed formula reads
/// (`checkpoint_recovery_fetch_budget()` = `max(rpc_deadline_ms, 2 x
/// checkpoint_request_timeout_ms)`), so the SHAPE under test is production's.
///
/// The RATIO here is not production's, and that is deliberate rather than
/// incidental — see [`WEDGE_BUDGET_MS`]. This arm's claim needs three
/// members'-worth of speculative budget because it drives two *actually
/// fetched* wedges; at production's two-reserve ratio the second wedge would be
/// declined un-fetched and the caching claim would pass vacuously.
///
/// b.bu8 resolved the question that made the gap matter. Pre-b.bu8 the floor
/// answer was reached here only *because* of the headroom — the accounting could
/// not deliver it at the shipped two-reserve ratio, which is the defect — so this
/// arm's `restored` tick was evidence about a posture no deployment runs in.
/// Post-b.bu8 the base's fetch is reserved out of the budget by member kind, so
/// the floor answer no longer depends on headroom, and the shipped ratio has its
/// own arm: Test 14,
/// [`one_wedged_member_at_the_shipped_two_reserve_ratio_still_reaches_the_floor`].
/// The two are complementary — this one owns the selection-wide
/// `LinkFact::Wedged` cache, that one owns the accounting at the ratio
/// production actually runs.
///
/// The sibling `stepped_down_budget` producer — a selection whose remaining
/// *deadline* cannot fund one more fetch — is delegated by citation to
/// `t3.dfh.yj.wg.xn`'s unit coverage; see the module header.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_wedged_members_cost_one_reserve_each_and_the_selection_still_lands() {
    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "wedged", TailLayout::Chained).await;

    // Derived from `harness.config` and mutated by two fields, not rebuilt from
    // `test_config()`: everything else the hook reads must stay the harness's own
    // value or the arm silently runs under a different restore posture.
    let mut cfg = harness.config.clone();
    cfg.checkpoint_request_timeout_ms = WEDGE_RESERVE.as_millis() as u64;
    cfg.rpc_deadline_ms = WEDGE_BUDGET_MS;
    assert!(
        cfg.checkpoint_recovery_fetch_budget() > WEDGE_RESERVE * 3,
        "arm precondition: the outer budget ({:?}) must fund the two wedged members AND the \
         surviving lineage's own fetches, or this arm would assert an exhaustion instead of \
         the floor answer",
        cfg.checkpoint_recovery_fetch_budget()
    );
    harness.install_recovery_hook_with_config(Some(cfg), |hook| hook);

    // `T2` and `T1` — the two middle members of the chain. `T3` (the head) stays
    // healthy so the walk follows a real link into the first wedged member.
    let wedged_a = fixture.s0 + 2;
    let wedged_b = fixture.s0 + 1;
    store.arm_get(wedged_a, CheckpointKind::Tail, GetFault::Wedge);
    store.arm_get(wedged_b, CheckpointKind::Tail, GetFault::Wedge);

    wipe_for_restore(&harness, &fixture).await;

    // See Test 1 on why the count is marked rather than absolute.
    let fetch_mark = store.get_calls_mark();
    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let lineage_pre = lineage_cells(&harness);

    restore_expecting_ok(
        &harness,
        &fixture,
        "a degraded adapter with two wedged members",
    )
    .await;

    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "the floor answer MUST still be reached: exactly one `restored` tick. A zero here \
         means the selection burned its budget above the floor — the stranded-recovery class \
         SR-2.5 and FR-6 forbid"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat ON THIS ARM. An overrunning fetch is a budget VERDICT rather \
         than an error, and this selection reaches a floor ANSWER, so there is nothing left \
         for an error to classify. A tick here means the outer preemptive timeout fired \
         before the cooperative check, which is exactly the `Transient` → `Error` stranding \
         the per-fetch cap exists to eliminate. \
         NOT unconditional across budget arms (b.hxk): a selection that exhausts having left a \
         listed base UNREAD ends `error` by design, because it cannot claim the layer is \
         unrestorable — Test 15. What keeps this arm flat is that it resolves"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("stepped_down_budget", 3), ("gathered_base", 1)],
        "three candidates decline (the head on its wedged link, the first wedged member on the \
         CACHED fact, the second on its own overrun) and the base answers",
    );

    // ── The caching claim, on the call log (deterministic under load) ─────
    assert_eq!(
        store.get_calls_since(fetch_mark, wedged_a, CheckpointKind::Tail),
        1,
        "a member that ate a whole reserve without answering MUST be fetched at most ONCE for \
         the whole selection. It is reached twice — as candidate T3's link and as a candidate \
         in its own right — and the second reach must consult the cached `LinkFact::Wedged` \
         and cost no fetch. Two calls here means one wedged member costs one reserve PER \
         CANDIDATE and a degraded backend blows the outer budget before reaching the floor"
    );
    assert_eq!(
        store.get_calls_since(fetch_mark, wedged_b, CheckpointKind::Tail),
        1,
        "the second wedged member is likewise probed exactly once"
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.s0_oracle,
        "stepped down to the base past two wedged members",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::FullyRestored,
        "the floor restore MUST materialize the base generation completely"
    );
    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_base_only_cursor(&header, &fixture, "after two budget step-downs");

    // ── Claim (b) ────────────────────────────────────────────────────────
    // With the wedges cleared the whole chain verifies and composes to the head,
    // which is a strictly better answer than the floor the degraded run reached.
    store.clear_get_faults();
    wipe_for_restore(&harness, &fixture).await;
    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        "after the backend stopped wedging",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 14 — the SHIPPED budget ratio: ONE wedged member above a healthy base
// ═════════════════════════════════════════════════════════════════════════════

/// The same degraded-adapter claim as Test 6, at the ratio production actually
/// runs: **`budget == 2 x reserve`**, one wedged member, and the floor answer
/// still reached.
///
/// # Why this arm exists next to Test 6
///
/// Test 6 dials its budget to eight reserves so it can afford TWO wedged
/// members plus the surviving lineage's own fetches — its claim is about the
/// selection-wide `LinkFact::Wedged` cache, and that claim needs the headroom.
/// The consequence is that it cannot see the b.bu8 defect at all: with six
/// spare reserves standing, any accounting rule reaches the floor.
///
/// The shipped ratio has no spare reserve. `checkpoint_recovery_fetch_budget()`
/// is `max(rpc_deadline_ms, 2 x checkpoint_request_timeout_ms)`, which at the
/// defaults is exactly twice the reserve, so one wedged member and the floor
/// answer is the WHOLE budget. Pre-b.bu8 that did not fit: the wedged fetch ran
/// a flat reserve, the remainder fell under one reserve by the listing and the
/// walk around it, and the base's own fetch was declined by the very threshold
/// gate meant to protect it — `Ok(None)` -> `NotRecoverable` -> `NOT_FOUND`,
/// with a healthy base one generation below. The allowance rule reserves the
/// base's fetch by member kind instead, so the base is funded out of what is
/// left however the tail above it behaved.
///
/// # The topology, and what each candidate costs
///
/// [`TailLayout::Single`] — ONE tail over the base — because "one wedged member
/// above a healthy base" is the shape whose arithmetic is tightest, and because
/// it lands the wedge on the gather's **un-gated first fetch**: the one arm not
/// charged against the base's reservation, where the base gets
/// `min(reserve, remaining)` rather than a full reserve. That is the stated
/// exception in `gather_restore_lineage`'s property (ii), so it is the arm most
/// worth proving end-to-end.
///
/// 1. candidate `tail@head_seq` — the un-gated first fetch, wedged, capped at
///    [`SHIPPED_RATIO_RESERVE`] -> `stepped_down_budget`;
/// 2. candidate `base@s0` — funded out of the remainder, and the answer.
///
/// One `stepped_down_budget` tick, one reserve of wall clock, and a restore.
/// The two-wedged-member / cached-fact shape stays Test 6's; the allowanced
/// (non-arm-1) tail leaving the base a FULL reserve stays the unit lane's
/// (`src/checkpoint/lineage.rs`,
/// `one_wedged_member_cannot_starve_the_floor_at_the_shipped_two_reserve_budget`),
/// which can pin the arithmetic exactly under `tokio::time::pause`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_wedged_member_at_the_shipped_two_reserve_ratio_still_reaches_the_floor() {
    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "shipped-ratio", TailLayout::Single).await;

    // Derived from `harness.config` and mutated by two fields, exactly as
    // Test 6 does and for the same reason.
    let mut cfg = harness.config.clone();
    cfg.checkpoint_request_timeout_ms = SHIPPED_RATIO_RESERVE.as_millis() as u64;
    cfg.rpc_deadline_ms = SHIPPED_RATIO_RESERVE.as_millis() as u64;
    assert_eq!(
        cfg.checkpoint_recovery_fetch_budget(),
        SHIPPED_RATIO_RESERVE * 2,
        "arm precondition: the budget must be EXACTLY two reserves — production's ratio, and the \
         posture in which the pre-b.bu8 accounting could not fund the floor answer. Any headroom \
         above it and this arm degenerates into Test 6 and stops discriminating"
    );
    harness.install_recovery_hook_with_config(Some(cfg), |hook| hook);

    store.arm_get(fixture.head_seq, CheckpointKind::Tail, GetFault::Wedge);
    wipe_for_restore(&harness, &fixture).await;

    // See Test 1 on why the count is marked rather than absolute.
    let fetch_mark = store.get_calls_mark();
    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let lineage_pre = lineage_cells(&harness);

    restore_expecting_ok(
        &harness,
        &fixture,
        "one wedged member at the shipped two-reserve budget ratio",
    )
    .await;

    assert_eq!(
        store.get_calls_since(fetch_mark, fixture.head_seq, CheckpointKind::Tail),
        1,
        "the wedged member MUST have been addressed exactly once — non-vacuity"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "THE CLAIM. At the SHIPPED budget ratio one wedged member must still leave the base \
         fundable: exactly one `restored` tick. A zero here is the b.bu8 defect — the base's own \
         fetch declined by the threshold meant to protect it, surfacing as NOT_FOUND with a \
         healthy base one generation below, which is the stranded-recovery class SR-2.5 and FR-6 \
         forbid"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat ON THIS ARM — an overrunning fetch is a budget VERDICT rather \
         than an error, and this selection reaches the floor answer. Not unconditional across \
         budget arms: see Test 15, where an exhaustion that left a listed base unread ends \
         `error` by design (b.hxk)"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("stepped_down_budget", 1), ("gathered_base", 1)],
        "ONE candidate declines on its own wedged head fetch and the base answers — with a \
         single tail there is no second candidate to reach the wedged member through the cached \
         fact, which is what keeps this arm's count at one where Test 6's is three",
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.s0_oracle,
        "stepped down to the base at the shipped budget ratio",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::FullyRestored,
        "the floor restore MUST materialize the base generation completely"
    );
    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_base_only_cursor(&header, &fixture, "after one budget step-down");

    // ── Claim (b) ────────────────────────────────────────────────────────
    store.clear_get_faults();
    wipe_for_restore(&harness, &fixture).await;
    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        "after the backend stopped wedging, at the shipped budget ratio",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 15 — a budget exhausted with a HEALTHY base unread is RETRIABLE (b.hxk)
// ═════════════════════════════════════════════════════════════════════════════

/// A slow enumeration plus one wedged member leaves the layer's only full
/// generation **unread**, and the client must be told to come back — `UNAVAILABLE`
/// — not that its layer has no backup.
///
/// # The distinction this arm owns, and why the other budget arms cannot make it
///
/// Tests 6 and 14 both drive `stepped_down_budget` and both reach the floor
/// answer, so in both the budget verdict is a step-down *on the way to a
/// restore*. This arm is the case where the budget runs out with nothing left to
/// answer with, and the question it settles is what that means:
///
/// - Every listed `Base` was adjudicated ⇒ the selection has positive evidence
///   about the layer, `Exhaustion::AllBroken` ⇒ `Ok(None)` ⇒
///   `RecoveryOutcome::NotRecoverable` ⇒ terminal `NOT_FOUND`, and
///   `recovery_attempts_total{result="not_recoverable"}`.
/// - Some listed `Base` was never read ⇒ the selection has evidence about the
///   *clock*, not the layer, `Exhaustion::BudgetInconclusive` ⇒
///   `CheckpointError::Transient` ⇒ `RecoveryOutcome::Error` ⇒ retriable
///   `UNAVAILABLE`, and `recovery_attempts_total{result="error"}`.
///
/// This fixture is the second, and it is the production shape rather than a
/// corner: the base on the store is **perfectly healthy** — the retry half below
/// restores from it untouched — and the only reason the layer did not come back
/// is that the adapter was slow. Answering `NOT_FOUND` there tells a client its
/// durable data is gone, and the standard retry policy does not retry a
/// `NOT_FOUND`, so a transient store slowdown is reported as permanent data
/// loss.
///
/// # The fault, and why it takes BOTH halves
///
/// [`INCONCLUSIVE_LIST_DELAY`] alone cannot produce it. The `Base` allowance is
/// `min(reserve, remaining)`, never declined while any budget is left, so a slow
/// LIST over a healthy base still funds the base out of whatever remains and the
/// restore succeeds. What makes the base unreachable is the remainder hitting
/// **zero**, and the only thing that can spend a sub-reserve remainder to zero
/// is the gather's un-gated first fetch — hence the wedge on the head tail,
/// clamped by arm 1's `min(reserve, remaining)` to precisely what the LIST left.
/// The LIST is the half no cap covers (see [`INCONCLUSIVE_LIST_DELAY`]); the
/// wedge is the half that converts the shortfall into a zero.
///
/// Neither is a synchronisation `sleep`: both model adapter latency, which is
/// the fault under injection, exactly as [`GetFault::Wedge`] does.
///
/// # Why `stepped_down_budget` ticking is NOT the same claim
///
/// Two `stepped_down_budget` ticks fire here, and Tests 6 and 14 fire them too
/// while ending `restored`. So the step-down reasons cannot decide the client's
/// answer, and a predicate keyed on them would make Test 6's and Test 14's
/// perfectly-conclusive runs retriable as well. What decides it is the
/// `exhausted_budget` cell, which is why this arm asserts that cell moving and
/// `exhausted` staying flat rather than reading the step-downs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_budget_exhausted_with_a_healthy_base_unread_is_unavailable_not_not_found() {
    let store = Arc::new(MatrixFaultStore::new());
    let mut harness = setup_matrix_harness(&store).await;
    let fixture = setup_matrix_fixture(&mut harness, "inconclusive", TailLayout::Single).await;

    // Derived from `harness.config` and mutated by two fields, exactly as
    // Tests 6 and 14 do and for the same reason.
    let mut cfg = harness.config.clone();
    cfg.checkpoint_request_timeout_ms = SHIPPED_RATIO_RESERVE.as_millis() as u64;
    cfg.rpc_deadline_ms = SHIPPED_RATIO_RESERVE.as_millis() as u64;
    let budget = cfg.checkpoint_recovery_fetch_budget();
    assert_eq!(
        budget,
        SHIPPED_RATIO_RESERVE * 2,
        "arm precondition: production's ratio, as Test 14 uses"
    );
    assert!(
        INCONCLUSIVE_LIST_DELAY > budget - SHIPPED_RATIO_RESERVE
            && INCONCLUSIVE_LIST_DELAY < budget,
        "arm precondition: the LIST must leave STRICTLY LESS than one reserve and STRICTLY MORE \
         than nothing. Below `budget - reserve` the un-gated first fetch gets a full reserve and \
         the wedge cannot drain the remainder to zero, so the base would be funded and the \
         restore would succeed; at or above `budget` the outer preemptive wrapper elapses inside \
         the LIST itself and the arm would assert a plain timeout instead of the cooperative \
         verdict. Got a {INCONCLUSIVE_LIST_DELAY:?} LIST against a {budget:?} budget and a \
         {SHIPPED_RATIO_RESERVE:?} reserve"
    );
    harness.install_recovery_hook_with_config(Some(cfg), |hook| hook);

    wipe_for_restore(&harness, &fixture).await;

    // Armed AFTER the fixture and the wipe, so only the restore's own
    // enumeration pays.
    store.arm_list_delay(INCONCLUSIVE_LIST_DELAY);
    store.arm_get(fixture.head_seq, CheckpointKind::Tail, GetFault::Wedge);

    // See Test 1 on why the counts are marked rather than absolute.
    let fetch_mark = store.get_calls_mark();
    let list_mark = store.list_calls_mark();
    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let lineage_pre = lineage_cells(&harness);

    let status = restore_expecting_unavailable_status(
        &harness,
        &fixture,
        "a budget exhausted before the layer's only base could be read",
    )
    .await;
    store.clear_list_delay();

    // ── THE CLAIM: the status the client actually gets ───────────────────
    //
    // `restore_expecting_unavailable_status` already asserted the CODE, which
    // is the whole verdict: pre-b.hxk this trace answered `Ok(None)` ->
    // `NotRecoverable` -> NOT_FOUND, which the standard client retry policy
    // does not retry.
    //
    // The message is pinned only on its two operator-actionable NUMBERS, not
    // as prose: the budget that ran out (the knob to raise, cross-checked
    // against `checkpoint_adapter_ceiling_elapsed_total`) and the fact that
    // generations went unread (what distinguishes this from real data loss).
    // Wording beyond those is free to change.
    assert!(
        status
            .message()
            .contains(&format!("{}ms", budget.as_millis())),
        "the surfaced status MUST name the budget that ran out — it is the knob an operator \
         acts on. Got: {}",
        status.message()
    );
    assert!(
        status.message().contains("unread"),
        "and it MUST say that generations went UNREAD, which is what tells an operator this is \
         adapter latency rather than a layer with no usable backup. Got: {}",
        status.message()
    );

    // ── Non-vacuity: the fault really landed, and the base really did not ─
    assert_eq!(
        store.list_calls_since(list_mark),
        1,
        "the restore's own enumeration MUST have run exactly once under the delay — the LIST is \
         half the fault, and a zero here means the arm proved nothing"
    );
    assert_eq!(
        store.get_calls_since(fetch_mark, fixture.head_seq, CheckpointKind::Tail),
        1,
        "the wedged head MUST have been addressed exactly once — the other half of the fault"
    );
    assert_eq!(
        store.get_calls_since(fetch_mark, fixture.s0, CheckpointKind::Base),
        0,
        "THE PREMISE. The base was never fetched, so the selection learned NOTHING about it — \
         and it is a perfectly healthy base, as the retry below proves by restoring from it \
         untouched. A `1` here means the base was adjudicated after all and this arm is no \
         longer testing the inconclusive case"
    );

    // ── The recovery-attempt classification ──────────────────────────────
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "an inconclusive exhaustion is an ERROR attempt: it routes through \
         `CheckpointError::Transient` -> `RecoveryOutcome::Error`, which is what carries the \
         retriable status to the client"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "and `not_recoverable` MUST stay flat. That cell is the affirmative claim `nothing here \
         can be restored`, and this selection never read the artifact that could have restored \
         it. A tick here is the b.hxk defect, visible on the dashboard as checkpoint loss that \
         did not happen"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing was materialized"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("stepped_down_budget", 2), ("exhausted_budget", 1)],
        "the head declines on its clamped wedge and the base declines at a zero allowance, then \
         the selection exhausts INCONCLUSIVELY. `exhausted` MUST stay flat: that cell means the \
         selection adjudicated every full generation and none is usable, which is a statement \
         about the stored artifacts — the one thing this run cannot say",
    );

    // ── Claim (a) ────────────────────────────────────────────────────────
    let presence = assert_no_partial_object(
        &harness,
        fixture.object_id,
        &fixture.head_oracle,
        "interrupted by an inconclusive budget exhaustion",
    )
    .await;
    assert_eq!(
        presence,
        ObjectPresence::Absent,
        "a selection that never resolved a lineage cannot have written anything"
    );

    // ── Claim (b) ────────────────────────────────────────────────────────
    //
    // Also the proof that the retriable answer was the RIGHT one: the same
    // store, unmodified except that it is no longer slow, restores the whole
    // composed head. The layer was never unrestorable.
    store.clear_get_faults();
    assert_retry_converges(
        &harness,
        &fixture,
        CursorCheck::ComposedRestore,
        "after the adapter stopped being slow — the retry the UNAVAILABLE invited",
    )
    .await;
}
