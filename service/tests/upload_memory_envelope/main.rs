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

//! The upload memory envelope (b.715 T7, SDD §5.4): an admitted upload's heap
//! never grows past what its reservation charged, and the measurements the
//! charge constants in `grpc::upload_admission` are fitted from.
//!
//! A dedicated binary because it installs a counting `#[global_allocator]`,
//! which applies to the whole binary — in `integration` it would count every
//! other test's allocations too. It wraps the system allocator, so the counts
//! are glibc `malloc`'s requests as Rust makes them. A `realloc` that moves its
//! block books the new block before freeing the old one, as the copy needs
//! both. Allocations outside the Rust heap (the OODF parser's C++ side) are not
//! counted; glibc's own in-use figure (`mallinfo2`) is reported beside the
//! Rust peak for the largest delta fixture. The counts are those of the build
//! the binary runs in, and CI runs it with the test-only cargo features on
//! (`test-fault-injection` among them), which compile test seams into the
//! measured calls' paths. An unarmed seam must allocate nothing, or that
//! build measures heap the release build never allocates and the charges
//! never price (`storage::layer::park_between_attempts`).
//! So validate these rows with CI's feature set (the repo Lint command's
//! features), not `test-support` alone.
//!
//! Each fixture runs the real `CreateLayer` / `UpdateLayer` handler against
//! Valkey, with the real external-value store. `UpdateLayer` runs on a tiny
//! base layer: on the Lua fast path, and charged `UpdateSlow` on each route
//! that path covers ([`Route`]) — the `lua` backend's cached structural route
//! from a cold start and with the object cached, and the WATCH/MULTI/EXEC
//! path under `txn`, which there also takes the field-dense and
//! time-sample-dense fixtures (every fixture asserted against its charge
//! alone: b.26e Stage 2 charges the WATCH commit queue's worst case, so no
//! fixture's bound depends on reply timing any more), the whole WATCH set run
//! once on each of its read's window transports (below); a reset fixture
//! (structure-dense content sent as a reset) runs on each. The large-base
//! reproducers named `*_stay(s)_within_its_charge_b26e`, all but the
//! refused-plan one below, run a small structural
//! update or a reset on a base of thousands of children (a full 7,168-section
//! table among them, and two new children, whose second birth builds the
//! name index), the harness-shaped layer, the long-name layer or an empty
//! section on all three routes instead, and on the WATCH route alone a
//! structural edit of a layer of large inline values and an external-value
//! rewrite on a 20,000-hash `obj:ext` — the WATCH route of each on both of its
//! read's window transports (b.gvf: the production one, window scripts but
//! for the windows `storage::diff_read`'s bounded fan-out names as plain, and
//! every window plain, the transport a read falls back to) — each phase
//! against the charge in force
//! (on the cached route's cold start, the read of the uncached object runs
//! under its own permit or `T` grant and is reported, not checked:
//! [`Measured::labels`]); the `cached_hit_repairing_*_within_its_charge_fsz`
//! tests (b.fsz) hold a cache hit's stream repair — of records a second
//! store committed after the cache warm-up ([`Fixture::pending`]): 256 small
//! records (new prims), 256 field writes, 256 records of about 1 KiB, one
//! record of about 250 KiB, one record with a 2 MiB external
//! part, about 1 MiB of new sections read
//! in two reads, and the regression's structural record of about 3 MiB — to
//! the charge in force in every phase, on a small base, each after a twin
//! run and a read through every member of the general pool that shape the
//! connections' at-rest read buffers ([`warm_every_pool_member`]), and the
//! 1 KiB, 250 KiB, 2 MiB and 3 MiB rows also cold, with neither; on large
//! bases, the per-record charge point's (RP2) rows — a creation into a
//! sections table at its load limit (D1), a child added to a list of 30,000
//! (D2) and 256 prims under a parent of exactly 4,096 children (its
//! name-index vector's resize) — each after a twin run; the in-process
//! rows hold one record's replay to RP2's hold beside the entry: the sampled
//! audit's index-less copy at about 8 MiB and at the per-entry cap, for an
//! asset placement, a field into a full fields table and a section into a
//! full sections table (`audit_replay_of_*_stays_within_its_hold_fsz`), and
//! RP2's scratch (b.fsz dq-1) — cascades of a 1,000-section group, of a
//! placed asset, a group and through each cascading command on the dev
//! shape, of a 20,000-child section, resets of a 9,000-deep chain and of the
//! dev shape, and the list-delete sink's rows — on an indexed entry and on
//! its copy, through the discarding and the list-delete sink
//! (`replay_scratch_of_*_stays_within_its_hold_fsz`), with the parent
//! listing's pin, one child added to a list of 30,000 on an indexed entry
//! (`replay_of_a_child_added_to_a_long_list_on_an_indexed_entry_*`);
//! `first_difference_holds_no_copy_of_either_state_fsz` holds the audit's
//! compare to 4 KiB of heap on a 20,000-section state; the `retirement_*_fsz`
//! rows (b.fsz MR-D4) hold a retirement's reference walk of long-named rows
//! with large inline values to its page bound (`RETIREMENT_ROW_BYTES`), the
//! whole release of such an object (the composed route, no reference) to
//! `W`, its release of 10,000 and 20,000 references to a slope of at most `h`
//! per reference (`RETIREMENT_REFERENCE_CHARGE_BYTES`) over a constant within
//! `W` (the size reads in flight, at most `RELEASE_SIZE_READS_IN_FLIGHT`), and
//! its teardown of a compact set step to `RETIREMENT_SET_MEMBER_BYTES` per
//! member; the `#[ignore]`d
//! `stream_repair_segments_are_reported_fsz` reports the same repairs'
//! segments for the two-term reply fit; and the `#[ignore]`d
//! `repair_growth_records_are_reported_fsz` measures, in process, each
//! replayed record of every row above for the per-byte growth fit (its
//! section's doc); one more large-base
//! reproducer,
//! `slow_path_refused_plan_on_a_full_budget_stays_within_its_charge_b26e`,
//! deletes a section beside one of 64,000 fields on the cached route with
//! the object cached alone, on a gate whose budget is full but for the
//! call's room and whose overflow slot is held, so the plan's first growth
//! is refused and the call waits for its exact need: each phase outside the
//! refused plan is checked against the charge in force, and the refused
//! plan's phases against the documented residual (peak ≤ the next target,
//! the charge the call waited for); the `stalled_watch_*_stays_within_its_charge_b26e`
//! tests (`test-fault-injection`) hold each of the WATCH set's dense fixtures
//! to its charge with its whole commit pipeline queued behind a stalled
//! transaction connection, on both window transports; the `#[ignore]`d
//! `cached_route_plan_rows_are_reported_b26e` reports the cached route's
//! plan rows (a cascade, an edit beside a long list, a reset, a re-key of a
//! long-listed section) phase by phase; and the `#[ignore]`d
//! `*_phases_are_reported_b26e` runs report, without asserting, each phase of
//! one new child on bases from an empty section to 20,000 children and on
//! harness-shaped layers of three sizes, and of moving the first of 20,000
//! children to the end, the WATCH path at two item ceilings, with what the
//! object-state terms multiply (the M1 fit's inputs); the `#[ignore]`d
//! `long_reorder_list_on_the_watch_path_is_reported_b26e` measures that one
//! fixture alone, for repeated runs; the `#[ignore]`d
//! `object_state_segments_are_reported_b26e` reports, for each structural
//! commit route, the heap peak between every two object-state charge points
//! with the counts each point priced (the Stage-2 fits); the `#[ignore]`d
//! `stalled_watch_commit_queue_is_reported_b26e` (`test-fault-injection`)
//! holds the WATCH commit pipeline behind a stalled transaction connection
//! until it is all queued, for the queued-command fit; the `#[ignore]`d
//! `plans_per_commit_under_a_charge_are_reported_b26e` reports how many plans
//! each structural commit made under the call's real charge; and the
//! `#[ignore]`d b.ug2 Stage 6b reports measure the in-place route's re-fit
//! inputs: `cached_route_element_shapes_are_reported_ug2` (the per-element
//! term on the cached route, element-dense structural edits through the
//! handler), `depth_walks_are_reported_ug2` (the depth check's full and
//! incremental walks, alone) and, with `test-fault-injection`,
//! `rollback_after_cr2_arms_are_reported_ug2` (a deleting edit rolled back
//! after CR2 on a `STALE`, its segments split around the rollback). The
//! object-state segments report also carries the in-place re-fit's cases
//! (`ug2-*`: the walks, resets, cascades and element-heavy edits). The `m3_*`
//! arms (b.26e Stage 2, M3) run several one-new-child structural
//! `UpdateLayer`s at once through the handler on one gate at the chart's
//! default `B`, and hold the heap's peak growth over the arm within the
//! gate's in-budget and overflow high-water marks (plus, on the cold arms,
//! the high-water of `T`, inside which their cold starts' fenced reads take
//! their grants, and `P_f × X`, the reads' window transient `T` funds beside
//! the grants), and each stretch of one gate total within that total and
//! what the calls hold outside the gate: four demo arms on 5,000-child
//! bases, and three `#[ignore]`d stress arms on 20,000-child bases. The
//! `post_release_*_stays_within_the_at_rest_bound_b26e` tests hold the heap
//! repeated uploads leave live after their release within what the Valkey
//! connections may keep once drained (`storage::connection_buffers`), and
//! each client to the warm rule read off its retention report, on a
//! standalone store and, as `post_release_cluster_*`, on a cluster; and the
//! tests after them check each connection buffer limit's holder: the stream
//! reader's (on a cluster too, across a retarget), the retry buffer, a fred
//! `Transaction`'s EXEC reply and the reader read limit under mixed reply
//! sizes; then one warm connection per client, the idle shape and the
//! cluster routing table's owned strings (tests 17, 18, 19 and 25: test 18
//! runs everywhere, its cluster twin and the others only with
//! `LIVE_EDIT_TEST_CLUSTER_URL`, as do the cluster cases above); and that
//! fred's connection-buffer sweep allocates nothing on a pass that changes no
//! buffer (test 8, counted by the allocator while fred's in-sweep flag is set
//! on the measured client's own thread: standalone everywhere, its cluster
//! twin only with `LIVE_EDIT_TEST_CLUSTER_URL`). The content arrives one
//! gRPC message per HTTP/2 DATA frame, 1 MiB per message, the way a client
//! sends it; the input frames stay alive across the call, standing in for the
//! transport buffers the receive allowance `A` covers, so they count as
//! baseline. The handler future is boxed inside the measured window, as tonic
//! boxes it. Each fixture but the refused-plan reproducer and the M3 arms
//! gets a fresh gate, so the larger of the gate's in-budget and overflow
//! high-water marks is that one call's largest reservation: its CHARGE (the
//! call's high-water). The refused-plan reproducer shares its gate with a
//! filler and a slot holder and reads each phase net of them; an M3 arm's
//! calls share theirs, so its phases are the arm's, not one call's.
//!
//! **Phases.** The gate's test-only charge observer
//! (`UploadAdmissionGate::set_charge_observer_for_test`) reports every change
//! to what the gate holds, and with one upload on a fresh gate that is the
//! call's charge. At each report the measured window closes and the next one
//! opens, so each [`Phase`] has one charge in force and its own heap peak
//! (above the live bytes when the call began): before admission, the receive
//! phase (which holds `A`), the parse step, the processing step, and the
//! release. Checking each phase against its own charge sees an under-charge
//! the whole call's high-water hides: the high-water includes `A` (8 MiB),
//! held only while the content is received, so a processing phase can run
//! up to `A` over its charge and still stay under it.
//!
//! **Assertions.**
//!
//! - The envelope, per phase: for every fixture, each phase's heap growth peak
//!   is at most the charge in force during it, at the production constants
//!   ([`phase_limits`] says which charge that is for a phase holding nothing)
//!   — except the refused-plan reproducer's refused plan, whose phases are
//!   held to the documented residual instead (peak ≤ the next target).
//! - The fit: each path's mixed fixture — fitted from nothing — stays within
//!   the charge this run's own fitted coefficients (× 1.25) predict, so the
//!   per-path model holds for content it was not fitted on.
//! - The parse: `oodf_ffi::parse_delta` alone stays within `k_parse` less its
//!   100-hundredths allowance for the C++ side.
//! - The census: a fast-path upload of 64 inline values of exactly 65,521 B
//!   makes exactly 64 allocations of that size — one owned copy of each value
//!   — so a reinstated handler-side copy of the delta (128) fails it, which
//!   the 25 % margin of the envelope cannot catch.
//! - I5, the external-value copies: a fast-path upload of 32 external values
//!   of exactly 131,071 B makes one owned copy of each, and the store path
//!   never holds more than `STORE_VALUES_MAX_INFLIGHT` plus one finished copy
//!   per router thread alive at once (9 on the census runtime: fred drops a
//!   finished command's copy only after sending its reply); with an
//!   ext-mirror sender wired it still makes exactly one per value, the mirror
//!   sharing it.
//! - The ext-mirror's per-item allowance (b.mhq T13): the heap one queued
//!   mirror item holds beyond its value's bytes — the hash string, the
//!   copy's owner and charge slot, the channel slot — fits the one KiB each
//!   item is charged beyond its value (`ITEM_OVERHEAD_KIB`).
//!
//! **The fit**, per commit path, in this order (the `F` in steps 1–4 is the
//! path's header-only peak; `byte*` and `element*` are the coefficients the later terms are
//! netted with, which the derivation rule in the comment above the constants
//! in `grpc::upload_admission` defines):
//!
//! 1. `byte[path]` from the value-heavy fixture: `(peak − F − body) ÷ delta`;
//! 2. `element[path]` from the dense fixtures: `(peak − F − body −
//!    byte*·delta) ÷ elements`;
//! 3. `reorder[path]` from the long-reorder fixture, a reorder of children
//!    created in the same delta: `(peak − F − body − byte*·delta −
//!    element*·elements) ÷ reorder entries`;
//! 4. `e_ext` from the blob fixtures, net of `byte*` and `element*` (the blob
//!    fixtures carry no reorder list);
//! 5. on `UpdateSlow`, the one-child structural edit's fixed cost: `peak −
//!    body − byte*·delta − element*·elements`. The path's `F` is the larger of
//!    that and the header-only peak.
//!
//! The report prints this run's own chain (`measured`: `byte*` and
//! `element*` are this run's figures), the netted chain (`netted`), and the
//! netted chain's recommendation, ⌈figure × 1.25⌉ (`F`, `element` and
//! `reorder` in bytes; `byte`, `e_ext` and `k_parse` in hundredths, `k_parse`
//! plus 100). The `netted` line carries a floor-binding note for each earlier
//! term (byte, element) that the measured route binds — the route its
//! constant is derived from — whose recommendation from this run's own chain
//! is below the path's production constant: the later terms were then netted
//! against the production basis, not this run's own chain, so this run's own
//! recommended set is not self-consistent. On a route that does not bind a
//! term the constant comes from a costlier route, so no note is printed. A
//! run that leaves fixtures out ([`run_path_where`]) prints an "as expected"
//! note instead: the WATCH long-reorder report leaves out field-dense, which
//! `element[UpdateSlow]` is derived from. `.config/nextest.toml` keeps the
//! report on a passing run. One run's recommendation is not the production
//! constant: the derivation rule says which runs it is taken over.
//!
//! Every test that measures, or touches the allocator's tracking state,
//! takes [`MEASURE`] first, so under `cargo test` (one process, parallel
//! threads) no two measure at once; under nextest each runs in its own
//! process anyway. Those that take it through [`measure_lock`] install the
//! test logger first: with `RUST_LOG` set, the service's and fred's log lines
//! go to stderr ([`init_test_logging`]), and the byte figures are for
//! diagnosis only; without it, no logger is installed.

#[path = "../common/mod.rs"]
mod common;

/// The integration suite's gRPC helpers; this binary uses a few of them, so
/// the rest (and their re-exports) are unused here.
#[path = "../integration/grpc/helpers.rs"]
#[allow(dead_code, unused_imports)]
mod helpers;

use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use bytes::{BufMut, Bytes};
use fred::interfaces::ClientLike;
use http_body::Frame;
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
    CommandReorderChildren, OwnedDelta, SetField, StructCommand, TimeSample,
};
use prost::Message;
use tonic::codec::Streaming;
use tonic::Request;

use fred::live_edit::retention::test_support::{
    request_report, set_sweep_for_test, take_report, ConnectionReport, SweepForTest,
};
use fred::live_edit::retention::{RetentionLimits, FRESH_BUFFER_BYTES, IN_FLIGHT_SLOT_BYTES};
use live_edit::checkpoint::ext_mirror_channel;
use live_edit::config::{
    snapshot_fanout_window, UpdateLayerBackend, RPC_DEADLINE_MS_DEFAULT,
    SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT, SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
};
use live_edit::domain::delta::{
    apply_delta, apply_delta_bounded, ApplyDeltaStats, BoundedPlan, MutationAllowance, RewriteCount,
};
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{CreateLayerRequest, UpdateLayerRequest};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::upload_admission::{
    object_state_charge_for_test, object_state_rewrite_allowance_for_test, parse_charge,
    processing_charge, CommitPath, DeltaFootprint, GateCharge, UploadAdmissionConfig,
    UploadAdmissionGate, DELTA_BYTE_CHARGE_CREATE_CENTI, DELTA_BYTE_CHARGE_UPDATE_FAST_CENTI,
    DELTA_BYTE_CHARGE_UPDATE_SLOW_CENTI, ELEMENT_CHARGE_CREATE_BYTES,
    ELEMENT_CHARGE_UPDATE_FAST_BYTES, ELEMENT_CHARGE_UPDATE_SLOW_BYTES, EXT_EXPANSION_CENTI,
    FIXED_CHARGE_BYTES, PARSE_EXPANSION_CENTI, REORDER_ENTRY_CHARGE_CREATE_BYTES,
    REORDER_ENTRY_CHARGE_UPDATE_SLOW_BYTES,
};
use live_edit::observability::metrics::test_util::{render, sample_sum, sample_value};
use live_edit::observability::metrics::{
    STRUCTURAL_CACHE_UNCACHEABLE_REASONS, UPDATE_LAYER_CONFLICT_KINDS, UPDATE_LAYER_PATHS,
    UPDATE_LAYER_PATH_RESET, UPDATE_LAYER_PATH_STRUCTURAL, UPDATE_LAYER_PATH_STRUCTURAL_FULL,
    UPDATE_LAYER_PATH_WATCH, UPDATE_LAYER_RETRY_OUTCOMES, UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS,
};
use live_edit::storage::commit_charge::{CommitCharge, ObjectStateNeed, RepairTransient};
use live_edit::storage::connection_buffers::{
    at_rest_bound_bytes, PoolSizes, Topology, BACKCHANNEL_BYTES, GENERAL_LIMITS,
    IDLE_CONNECTION_BYTES, READER_LIMITS,
};
use live_edit::storage::diff_read::{
    read_object_snapshot_slim, read_object_snapshot_slim_with_ceilings, SlimSnapshotReadConsumer,
    SnapshotCeilings, SnapshotReadOf, WindowTransport,
};
use live_edit::storage::external_values::ValkeyExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::memory_budget::MemoryBudget;
use live_edit::storage::notifications::{consume_layer_stream, LayerStreamConsumer, StreamCursor};
use live_edit::storage::session::SessionStore;
use live_edit::storage::stream_mux::XREAD_REPLY_TARGET_BYTES;
use live_edit::storage::structural_cache_cold_start::plain_read_predicted_to_fit;
use live_edit::storage::structural_cache_sync::STRUCTURAL_REPAIR_INITIAL_BYTES;
use live_edit::storage::test_support::{
    connect_test_client, ConnStall, TestValkey, STORE_VALUES_MAX_INFLIGHT,
};

use common::owned_delta::delta_part_len;
use common::upload_stream::{AttestedStreaming, EndsWithEndStream};
use helpers::{
    build_service, make_create_data, make_create_header, make_request, make_update_data,
    make_update_header, streaming_create, AllowAllPermissions,
};

// ═════════════════════════════════════════════════════════════════════════════
// The counting allocator
// ═════════════════════════════════════════════════════════════════════════════

/// Bytes allocated and not yet freed, process-wide.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// The highest [`LIVE`] has been since the last [`begin_window`].
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// The lowest [`LIVE`] has been since the last [`begin_window`]: below the
/// window's baseline only when memory allocated before the window was freed
/// in it ([`window_dip`]), which may have understated [`end_window`]'s peak:
/// not at all if it was freed after the peak, by more than the dip if it was
/// freed while the measured call held memory of its own.
static TROUGH: AtomicUsize = AtomicUsize::new(usize::MAX);
/// The allocation size the census counts; 0 counts nothing.
static CENSUS_SIZE: AtomicUsize = AtomicUsize::new(0);
/// Allocations of exactly [`CENSUS_SIZE`] since the census began.
static CENSUS_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Allocations of exactly [`CENSUS_SIZE`] made in the window and not yet
/// freed (a block of that size made before the window and freed in it takes
/// one off, so the figure can only under-count).
static CENSUS_LIVE: AtomicIsize = AtomicIsize::new(0);
/// The highest [`CENSUS_LIVE`] has been since the census began.
static CENSUS_LIVE_PEAK: AtomicIsize = AtomicIsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Ordering::SeqCst) + by;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

fn shrank(by: usize) {
    let now = LIVE.fetch_sub(by, Ordering::SeqCst).saturating_sub(by);
    TROUGH.fetch_min(now, Ordering::SeqCst);
}

/// Whether an allocation or free of `size` is one the census counts: that
/// size, inside a measurement window — only the measured call's allocations.
fn census_counts(size: usize) -> bool {
    let counted = CENSUS_SIZE.load(Ordering::Relaxed);
    counted != 0 && size == counted && WINDOW_OPEN.load(Ordering::Relaxed)
}

/// Count an allocation of `size`, and one more of that size alive.
fn census(size: usize) {
    if census_counts(size) {
        CENSUS_COUNT.fetch_add(1, Ordering::SeqCst);
        let alive = CENSUS_LIVE.fetch_add(1, Ordering::SeqCst) + 1;
        CENSUS_LIVE_PEAK.fetch_max(alive, Ordering::SeqCst);
    }
}

/// Count one fewer allocation of `size` alive.
fn census_freed(size: usize) {
    if census_counts(size) {
        CENSUS_LIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Start a census of allocations of exactly `size`.
fn begin_census(size: usize) {
    CENSUS_COUNT.store(0, Ordering::SeqCst);
    CENSUS_LIVE.store(0, Ordering::SeqCst);
    CENSUS_LIVE_PEAK.store(0, Ordering::SeqCst);
    CENSUS_SIZE.store(size, Ordering::SeqCst);
}

/// End the census: `(allocations made, most alive at once)`.
fn end_census() -> (usize, isize) {
    CENSUS_SIZE.store(0, Ordering::SeqCst);
    (
        CENSUS_COUNT.load(Ordering::SeqCst),
        CENSUS_LIVE_PEAK.load(Ordering::SeqCst),
    )
}

/// Allocations of at least this many bytes are counted in [`LARGE_COUNT`]
/// (b.26e tests 18 and 19: a new read buffer); 0 counts nothing. A `realloc`
/// that grows a block to at least this size counts as one.
static LARGE_MIN: AtomicUsize = AtomicUsize::new(0);
/// Allocations of at least [`LARGE_MIN`] bytes since [`begin_large_count`].
static LARGE_COUNT: AtomicUsize = AtomicUsize::new(0);
/// The largest of them.
static LARGE_LARGEST: AtomicUsize = AtomicUsize::new(0);

/// Count an allocation of `size` when it is at least [`LARGE_MIN`].
fn large(size: usize) {
    let min = LARGE_MIN.load(Ordering::Relaxed);
    if min != 0 && size >= min {
        LARGE_COUNT.fetch_add(1, Ordering::SeqCst);
        LARGE_LARGEST.fetch_max(size, Ordering::SeqCst);
    }
}

/// Start counting allocations of at least `min` bytes, process-wide.
fn begin_large_count(min: usize) {
    LARGE_COUNT.store(0, Ordering::SeqCst);
    LARGE_LARGEST.store(0, Ordering::SeqCst);
    LARGE_MIN.store(min, Ordering::SeqCst);
}

/// Stop counting: `(allocations counted, the largest of them)`.
fn end_large_count() -> (usize, usize) {
    LARGE_MIN.store(0, Ordering::SeqCst);
    (
        LARGE_COUNT.load(Ordering::SeqCst),
        LARGE_LARGEST.load(Ordering::SeqCst),
    )
}

/// Whether allocations made inside fred's connection-buffer sweep are counted
/// in [`SWEEP_ALLOCS`] (b.26e test 8). The sweep sets a per-thread flag while
/// it runs (`fred::live_edit::retention::test_support::in_sweep`); every
/// allocation, zeroed allocation and `realloc` made with it set on the
/// measured thread ([`SWEEP_THREAD`]) counts.
static SWEEP_COUNT: AtomicBool = AtomicBool::new(false);
/// Allocations made inside a sweep since [`begin_sweep_count`].
static SWEEP_ALLOCS: AtomicUsize = AtomicUsize::new(0);
/// The largest of them.
static SWEEP_LARGEST: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Set on the one thread test 8 runs its client on ([`on_sweep_thread`]):
    /// only sweeps run there are counted, so another client's sweep elsewhere
    /// in the process (the fixture's own, cutting back after its setup
    /// traffic) cannot be counted as the measured client's.
    static SWEEP_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Count an allocation of `size` when it is made inside a sweep on the
/// measured thread ([`SWEEP_THREAD`]). Both flags are const-initialised
/// thread-local `Cell`s, so reading them never allocates or recurses into the
/// allocator.
fn sweep_alloc(size: usize) {
    if SWEEP_COUNT.load(Ordering::Relaxed)
        && SWEEP_THREAD.try_with(std::cell::Cell::get).unwrap_or(false)
        && fred::live_edit::retention::test_support::in_sweep()
    {
        SWEEP_ALLOCS.fetch_add(1, Ordering::SeqCst);
        SWEEP_LARGEST.fetch_max(size, Ordering::SeqCst);
    }
}

/// Start counting allocations made inside a sweep on the measured thread.
fn begin_sweep_count() {
    SWEEP_ALLOCS.store(0, Ordering::SeqCst);
    SWEEP_LARGEST.store(0, Ordering::SeqCst);
    SWEEP_COUNT.store(true, Ordering::SeqCst);
}

/// Stop counting: `(allocations made inside a sweep, the largest of them)`.
fn end_sweep_count() -> (usize, usize) {
    SWEEP_COUNT.store(false, Ordering::SeqCst);
    (
        SWEEP_ALLOCS.load(Ordering::SeqCst),
        SWEEP_LARGEST.load(Ordering::SeqCst),
    )
}

struct Counting;

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged and only updates counters (and, while a post-release report
// tracks, the allocation map) on the side.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded verbatim; the caller upholds `alloc`'s contract.
        let p = unsafe { System.alloc(layout) };
        let state = in_tracker();
        if !p.is_null() && state != Some(true) {
            grew(layout.size());
            census(layout.size());
            large(layout.size());
            sweep_alloc(layout.size());
            if state == Some(false) {
                track_alloc(p, layout.size());
            }
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`.
        let p = unsafe { System.alloc_zeroed(layout) };
        let state = in_tracker();
        if !p.is_null() && state != Some(true) {
            grew(layout.size());
            census(layout.size());
            large(layout.size());
            sweep_alloc(layout.size());
            if state == Some(false) {
                track_alloc(p, layout.size());
            }
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let state = in_tracker();
        // Untracked before the block is freed, so another thread that is
        // handed the same address cannot have its entry removed.
        if state == Some(false) {
            track_free(ptr);
        }
        // SAFETY: as `alloc`; `ptr` came from this allocator with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        if state != Some(true) {
            shrank(layout.size());
            census_freed(layout.size());
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let state = in_tracker();
        if state == Some(false) {
            track_free(ptr);
        }
        // SAFETY: as `dealloc`.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if state == Some(true) {
            return p;
        }
        if !p.is_null() {
            census_freed(layout.size());
            census(new_size);
            if new_size > layout.size() {
                large(new_size);
            }
            sweep_alloc(new_size);
            if p == ptr {
                // Resized in place: only the difference was ever live.
                if new_size >= layout.size() {
                    grew(new_size - layout.size());
                } else {
                    shrank(layout.size() - new_size);
                }
            } else {
                // Moved: the new block existed beside the old one while the
                // contents were copied.
                grew(new_size);
                shrank(layout.size());
            }
            if state == Some(false) {
                track_alloc(p, new_size);
            }
        } else if state == Some(false) {
            // The old block is still live, unmoved.
            track_alloc(ptr, layout.size());
        }
        p
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Allocation tracking, for the post-release reports only (b.26e)
// ─────────────────────────────────────────────────────────────────────────────
//
// Off unless a post-release report turns it on ([`start_tracking`]), so the
// envelope tests see only the one `in_tracker` read per allocation. While on,
// every counted allocation is recorded by address in [`TRACKED`] (the large
// ones with a backtrace of where they were made, or last resized), and every
// free removes its entry, so what is left in the map is the window's
// allocations still live. The map, the backtraces and the report's own
// strings are allocated with this thread's tracker flag set, which keeps
// them out of [`LIVE`]: everything the tracker allocates it also frees with
// the flag set.

/// Whether new allocations are recorded in [`TRACKED`].
static TRACK: AtomicBool = AtomicBool::new(false);
/// Whether [`TRACKED`] holds a map that frees must be removed from.
static TRACK_MAP: AtomicBool = AtomicBool::new(false);
/// Tracked allocations at least this large carry a backtrace.
static TRACK_BACKTRACE_MIN: AtomicUsize = AtomicUsize::new(usize::MAX);

/// One live tracked allocation.
struct Tracked {
    size: usize,
    backtrace: Option<std::backtrace::Backtrace>,
    /// The tracker event ([`TRACK_EVENT`]) that recorded it.
    event: u64,
}

// The peak-time view (b.26e M0a), for a tracked fixture's call only
// ([`begin_fixture_tracking`]): every tracked allocation and free is
// numbered ([`TRACK_EVENT`]), the event at which [`LIVE`] reached its highest
// since tracking began is kept ([`TRACK_PEAK_EVENT`]), and each tracked
// allocation freed while tracking is kept with the event that freed it
// ([`TRACKED_FREED`]). The allocations live at the peak are then those
// recorded at or before it and not freed by then
// ([`report_tracked_at_peak`]), which the end-of-call report cannot show: by
// then the transients of the peak are gone.

/// Whether frees are kept in [`TRACKED_FREED`] for the peak-time view.
static TRACK_AT_PEAK: AtomicBool = AtomicBool::new(false);
/// The tracker's event counter: one per tracked allocation and free.
static TRACK_EVENT: AtomicU64 = AtomicU64::new(0);
/// The highest [`LIVE`] since tracking began.
static TRACK_PEAK_LIVE: AtomicUsize = AtomicUsize::new(0);
/// [`LIVE`] when tracking began.
static TRACK_BASE_LIVE: AtomicUsize = AtomicUsize::new(0);
/// The event of the allocation that reached [`TRACK_PEAK_LIVE`].
static TRACK_PEAK_EVENT: AtomicU64 = AtomicU64::new(0);
/// The tracked allocations freed while the peak-time view is on, each with
/// the event of its free.
static TRACKED_FREED: std::sync::Mutex<Vec<(Tracked, u64)>> = std::sync::Mutex::new(Vec::new());

/// The live tracked allocations, by address.
static TRACKED: std::sync::Mutex<Option<std::collections::HashMap<usize, Tracked>>> =
    std::sync::Mutex::new(None);

thread_local! {
    /// Set while this thread runs tracker code: its allocations and frees are
    /// neither counted nor tracked.
    static IN_TRACKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread is inside the tracker; `None` once its thread-locals
/// are gone (counted, never tracked).
fn in_tracker() -> Option<bool> {
    IN_TRACKER.try_with(std::cell::Cell::get).ok()
}

/// Marks this thread as inside the tracker until dropped.
struct TrackerGuard(bool);

impl TrackerGuard {
    fn enter() -> Self {
        Self(IN_TRACKER.with(|c| c.replace(true)))
    }
}

impl Drop for TrackerGuard {
    fn drop(&mut self) {
        IN_TRACKER.with(|c| c.set(self.0));
    }
}

fn tracked_map() -> std::sync::MutexGuard<'static, Option<std::collections::HashMap<usize, Tracked>>>
{
    TRACKED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Record a counted allocation of `size` at `ptr` while tracking is on.
fn track_alloc(ptr: *mut u8, size: usize) {
    if !TRACK.load(Ordering::Relaxed) {
        return;
    }
    let _guard = TrackerGuard::enter();
    // Captured before the map's lock is taken, so the map's lock is never
    // held while waiting for the backtrace lock. Never on a panicking thread:
    // the panic hook (std's default, printing a backtrace) holds std's
    // non-reentrant backtrace lock while it allocates, and a capture here
    // would wait on that lock on the thread that holds it. Other code that
    // captures a std backtrace while tracking is on would deadlock the same
    // way; the crate's own source captures none.
    let backtrace = (size >= TRACK_BACKTRACE_MIN.load(Ordering::Relaxed)
        && !std::thread::panicking())
    .then(std::backtrace::Backtrace::force_capture);
    let event = TRACK_EVENT.fetch_add(1, Ordering::SeqCst) + 1;
    let mut map = tracked_map();
    if let Some(map) = map.as_mut() {
        map.insert(
            ptr as usize,
            Tracked {
                size,
                backtrace,
                event,
            },
        );
    }
    drop(map);
    // Counted before it was tracked (`grew` runs first), so a new peak is
    // this allocation's.
    let live = LIVE.load(Ordering::SeqCst);
    if live > TRACK_PEAK_LIVE.fetch_max(live, Ordering::SeqCst) {
        TRACK_PEAK_EVENT.store(event, Ordering::SeqCst);
    }
}

/// Remove `ptr`'s entry, if tracked; with the peak-time view on, keep it as
/// freed at this event.
fn track_free(ptr: *mut u8) {
    if !TRACK_MAP.load(Ordering::Relaxed) {
        return;
    }
    let _guard = TrackerGuard::enter();
    let removed = tracked_map()
        .as_mut()
        .and_then(|map| map.remove(&(ptr as usize)));
    if let Some(tracked) = removed {
        if TRACK_AT_PEAK.load(Ordering::Relaxed) {
            let event = TRACK_EVENT.fetch_add(1, Ordering::SeqCst) + 1;
            TRACKED_FREED
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((tracked, event));
        }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Whether a measurement window is open (read by the glibc sampler).
static WINDOW_OPEN: AtomicBool = AtomicBool::new(false);
/// glibc's in-use bytes when the current window opened.
static GLIBC_BASELINE: AtomicU64 = AtomicU64::new(0);

/// Start a measurement window: the peak restarts at the live bytes now, which
/// are returned as the window's baseline. glibc's baseline is taken here too,
/// after every input of the measured call has been built.
fn begin_window() -> usize {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    GLIBC_BASELINE.store(glibc_in_use(), Ordering::SeqCst);
    let now = LIVE.load(Ordering::SeqCst);
    PEAK.store(now, Ordering::SeqCst);
    TROUGH.store(now, Ordering::SeqCst);
    WINDOW_OPEN.store(true, Ordering::SeqCst);
    now
}

/// Close the window: its heap growth peak above `baseline`.
fn end_window(baseline: usize) -> u64 {
    WINDOW_OPEN.store(false, Ordering::SeqCst);
    PEAK.load(Ordering::SeqCst).saturating_sub(baseline) as u64
}

/// How far [`LIVE`] fell below `baseline`, the window's, since the window
/// opened: a lower bound on the memory allocated before the window and freed
/// in it. Read before the next [`begin_window`].
fn window_dip(baseline: usize) -> u64 {
    baseline.saturating_sub(TROUGH.load(Ordering::SeqCst)) as u64
}

/// Serializes the measuring tests within one process; take it through
/// [`measure_lock`].
static MEASURE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// [`MEASURE`], with the test logger installed first ([`init_test_logging`]).
async fn measure_lock() -> tokio::sync::MutexGuard<'static, ()> {
    init_test_logging();
    MEASURE.lock().await
}

/// Install, once per process, a stderr logger filtered by `RUST_LOG`
/// (`tracing_subscriber`'s `EnvFilter`), with fred's `log` records bridged
/// into it (`tracing-log`, which `try_init` installs), so a run with
/// `RUST_LOG` set shows the service's and fred's own lines — the unresponsive
/// verdicts and near misses among them — beside the measurements. Nothing is
/// installed when `RUST_LOG` is unset (CI's runs): the logger formats each
/// event on the heap the counting allocator measures, so a default run
/// measures exactly what it did without one. A logger already set in the
/// process is kept. The subscriber itself is installed before any test's
/// reference is read, so its own state is baseline — but the fmt layer
/// formats each event into a per-thread buffer that it only clears, never
/// shrinks. That buffer is first allocated at a thread's first logged event
/// and grows to the longest line it has formatted. With `RUST_LOG` set, that
/// can happen inside a measured window and stay live after the call, counting
/// toward a case's after-call figure and its plateau. Such a run's byte
/// figures are therefore for diagnosis only: an overrun seen with `RUST_LOG`
/// set does not rule the logger out. CI's runs leave `RUST_LOG` unset and are
/// unaffected.
fn init_test_logging() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if std::env::var_os("RUST_LOG").is_none() {
            return;
        }
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();
    });
}

/// glibc's in-use bytes: small-block arenas plus mmapped blocks.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn glibc_in_use() -> u64 {
    // SAFETY: `mallinfo2` takes no arguments and only reads allocator state.
    let info = unsafe { libc::mallinfo2() };
    (info.uordblks + info.hblkhd) as u64
}

// ═════════════════════════════════════════════════════════════════════════════
// Phases: one window per charge
// ═════════════════════════════════════════════════════════════════════════════

/// One stretch of a measured call over which the gate's charge did not
/// change.
#[derive(Clone, Copy, Debug)]
struct Phase {
    /// What the gate held during the phase, in budget and on the overflow
    /// slot: with one upload on a fresh gate, that upload's charge.
    charge: GateCharge,
    /// The heap growth peak during the phase, above the live bytes when the
    /// call began (so memory the call allocated earlier and still holds
    /// counts in every later phase).
    peak: u64,
    /// The catch-up memory budget `T`'s grants and top-ups at their highest
    /// during the phase, as an M3 arm's sampler read them ([`M3Sampler`]:
    /// [`T_SAMPLED`], [`T_PHASE_MAX`]); 0 when no sampler runs.
    t_max: u64,
}

/// `T`'s grants and top-ups at the sampler's latest reading; 0 when no M3
/// sampler runs.
static T_SAMPLED: AtomicU64 = AtomicU64::new(0);
/// The highest [`T_SAMPLED`] since the open phase began: each phase close
/// takes it ([`Phase::t_max`]) and restarts it at the latest reading.
static T_PHASE_MAX: AtomicU64 = AtomicU64::new(0);

/// No charge: what the gate holds before admission and after the release.
const NOTHING: GateCharge = GateCharge {
    reserved_bytes: 0,
    overflow_bytes: 0,
};

impl Phase {
    /// The phase's charge: in budget and on the slot together.
    fn held(&self) -> u64 {
        self.charge
            .reserved_bytes
            .saturating_add(self.charge.overflow_bytes)
    }
}

/// Room for every phase of one call — one per receive chunk, a few more per
/// step — so the observer never grows its log, and so never allocates,
/// inside the window it measures.
const MAX_PHASES: usize = 4_096;

struct PhaseLog {
    /// The live bytes when the call began.
    baseline: usize,
    /// The charge of the phase now open.
    open_charge: GateCharge,
    phases: Vec<Phase>,
}

/// The gate's charge observer for one measured call: at each report it closes
/// the open phase at the window's peak so far and opens the next at the
/// reported charge, restarting the peak at the live bytes now.
///
/// The observer runs under the gate's lock, after every change; one operation
/// can report more than one change (a step into the overflow slot reports its
/// in-budget bytes leaving the budget, then the slot taken). Such a change
/// leaves a phase with nothing held and no allocation in it, which
/// [`phase_limits`] checks against the slot charge that follows it.
#[derive(Clone)]
struct PhaseRecorder(Arc<std::sync::Mutex<PhaseLog>>);

impl PhaseRecorder {
    /// A recorder installed as `gate`'s charge observer.
    fn install(gate: &UploadAdmissionGate) -> Self {
        let recorder = Self(Arc::new(std::sync::Mutex::new(PhaseLog {
            baseline: 0,
            open_charge: NOTHING,
            phases: Vec::with_capacity(MAX_PHASES),
        })));
        let observer = recorder.clone();
        gate.set_charge_observer_for_test(Some(Arc::new(move |c: GateCharge| {
            observer.close_phase(c);
        })));
        recorder
    }

    fn log(&self) -> std::sync::MutexGuard<'_, PhaseLog> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Open the call's window ([`begin_window`]) and its first phase, with
    /// nothing held.
    fn begin(&self) {
        let baseline = begin_window();
        let mut log = self.log();
        log.baseline = baseline;
        log.open_charge = NOTHING;
        log.phases.clear();
        T_PHASE_MAX.store(T_SAMPLED.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    /// Close the open phase and open one at `charge`. Allocates nothing.
    fn close_phase(&self, charge: GateCharge) {
        let mut log = self.log();
        let peak = PEAK.load(Ordering::SeqCst).saturating_sub(log.baseline) as u64;
        let t_max = T_PHASE_MAX.swap(T_SAMPLED.load(Ordering::SeqCst), Ordering::SeqCst);
        let open = log.open_charge;
        log.phases.push(Phase {
            charge: open,
            peak,
            t_max,
        });
        log.open_charge = charge;
        PEAK.store(LIVE.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    /// Close the window ([`end_window`]) and the last phase: every phase, in
    /// order, and the live bytes the call left above its baseline.
    fn finish(&self) -> (Vec<Phase>, u64) {
        let mut log = self.log();
        let tail = end_window(log.baseline);
        let residual = LIVE.load(Ordering::SeqCst).saturating_sub(log.baseline) as u64;
        let open = log.open_charge;
        log.phases.push(Phase {
            charge: open,
            peak: tail,
            t_max: T_PHASE_MAX.load(Ordering::SeqCst),
        });
        assert!(
            log.phases.len() <= MAX_PHASES,
            "fixture self-check: {} phases, past the log's {MAX_PHASES}: the observer grew its \
             log inside the window",
            log.phases.len()
        );
        (std::mem::take(&mut log.phases), residual)
    }
}

/// What a phase holding nothing is, by where it falls in the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unheld {
    /// The first phase: the call before admission.
    PreAdmission,
    /// Between the two reports of a step into the overflow slot — the
    /// reservation's in-budget bytes leaving the budget, then the slot
    /// taken — which come in one hold of the gate's lock: the next phase
    /// holds the slot.
    SlotEntry,
    /// After the last charge: the call returning.
    Released,
    /// In the middle of the call with nothing charged and no slot entry
    /// next: the reservation was released and charged again. Any heap the
    /// call holds there is uncharged.
    Uncharged,
}

/// What phase `i`, which holds nothing, is ([`Unheld`]).
fn unheld(phases: &[Phase], i: usize) -> Unheld {
    if i == 0 {
        Unheld::PreAdmission
    } else if !phases[i + 1..].iter().any(|q| q.held() > 0) {
        Unheld::Released
    } else if phases[i + 1].charge.overflow_bytes > 0 {
        Unheld::SlotEntry
    } else {
        Unheld::Uncharged
    }
}

/// The charge each phase is checked against: the charge in force during it.
/// A phase with nothing held is checked by what it is ([`Unheld`]):
///
/// - before admission: the admission's charge, which `F` covers;
/// - a slot entry's gap, with no allocation in it: the slot's charge, the
///   next phase's;
/// - after the release: the call returning what it held under the last
///   charge, plus `cache_entry`, the structural-cache entry a cold start
///   stored ([`Measured::cache_entry`]; 0 on every other route). That entry
///   outlives the call — it is the cache's, not the upload's, once stored —
///   so it is reported, not charged. Nothing else the call leaves live is
///   allowed for: the Valkey connections cut their buffers back before the
///   call is answered (`storage::connection_buffers`), so their growth is
///   gone by the release, and what the handler allocates after its release
///   is checked against the last charge;
/// - anywhere else in the call: 0. A reservation released and charged again
///   mid-call leaves the call holding its heap with nothing charged, so any
///   growth above the call's start there is an overrun.
fn phase_limits(phases: &[Phase], cache_entry: u64) -> Vec<u64> {
    phases
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if p.held() > 0 {
                return p.held();
            }
            match unheld(phases, i) {
                Unheld::PreAdmission => phases[1..]
                    .iter()
                    .map(Phase::held)
                    .find(|&c| c > 0)
                    .unwrap_or(0),
                Unheld::SlotEntry => phases[i + 1].held(),
                Unheld::Released => phases[..i]
                    .iter()
                    .rev()
                    .map(Phase::held)
                    .find(|&c| c > 0)
                    .map_or(0, |c| c.saturating_add(cache_entry)),
                Unheld::Uncharged => 0,
            }
        })
        .collect()
}

/// The label of a [`Route::CachedCold`] commit's phases that read the
/// uncached object under the read's own permit or `T` grant, before the
/// decoded entry is charged to the upload ([`Measured::labels`]): reported,
/// not checked against the upload's charge.
const COLD_START_READ: &str = "cold-start-read";

/// A name for each phase, from the charges the handler's steps take: until
/// the first fall in the charge, admission and the receive phase (each
/// received chunk grows it); then the parse step's target (`parse`) and, at
/// or above the processing step's target, `processing`: the processing
/// step's own charge and every in-commit object-state growth on top of it (a
/// structural commit's charge points, and the handler's hint, which the
/// processing step already holds), so the label covers the whole commit. A
/// processing target equal to the parse target holds no new charge, so its
/// phase reads `processing`; any other charge after the receive phase is
/// `post-receive`. A phase holding nothing is `pre-admission`, `slot-entry`,
/// `released` or `uncharged` ([`Unheld`]).
fn phase_labels(phases: &[Phase], parse_target: u64, processing_target: u64) -> Vec<&'static str> {
    let mut received = false;
    let mut previous = 0;
    phases
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let held = p.held();
            if held == 0 {
                return match unheld(phases, i) {
                    Unheld::PreAdmission => "pre-admission",
                    Unheld::SlotEntry => "slot-entry",
                    Unheld::Released => "released",
                    Unheld::Uncharged => "uncharged",
                };
            }
            if held < previous {
                received = true;
            }
            previous = held;
            if !received {
                "receive"
            } else if held >= processing_target {
                "processing"
            } else if held == parse_target {
                "parse"
            } else {
                "post-receive"
            }
        })
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures
// ═════════════════════════════════════════════════════════════════════════════

const MIB: usize = 1024 * 1024;

/// Content per message, as clients chunk it.
const CHUNK: usize = MIB;

/// Entries in each dense fixture: large enough that the fixed overhead is
/// noise next to the expansion, small enough for a debug build.
const DENSE: usize = 20_000;

/// Inline values in a value-heavy fixture, and their size.
const VALUES: usize = 64;
const VALUE_LEN: usize = 64 * 1024;

/// The census value size: prime, so no other allocation of an upload's
/// processing has it by coincidence.
const CENSUS_VALUE_LEN: usize = 65_521;

/// The section every field, time sample and value hangs off.
const SECTION: u64 = 1;

fn create_section(parent_id: u64, section_id: u64, name: String) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: name,
        section_type: 0,
    })
}

fn field(key: String, value: Vec<u8>) -> SetField {
    SetField {
        section_id: SECTION,
        key_name: key,
        value_or_ext_hash: value,
        ext_value_size: 0,
        ext_value_index: 0,
        set_order: 0,
    }
}

fn time_sample(i: usize) -> TimeSample {
    TimeSample {
        section_id: SECTION,
        time: i as f64,
        value_or_ext_hash: (i as u64).to_le_bytes().to_vec(),
        ext_value_size: 0,
        ext_value_index: 0,
    }
}

fn external_field(key: String, blob: &[u8], index: u32) -> SetField {
    SetField {
        section_id: SECTION,
        key_name: key,
        value_or_ext_hash: live_edit::storage::external_values::ext_content_hash(blob).to_vec(),
        ext_value_size: blob.len() as u64,
        ext_value_index: index,
        set_order: 0,
    }
}

/// `count` deterministic, distinct blobs of `len` bytes.
fn blobs(count: usize, len: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..count)
        .map(|b| {
            (0..len)
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(b as u8) ^ salt)
                .collect()
        })
        .collect()
}

/// `count` inline values of `len` bytes, distinct per key.
fn value_fields(count: usize, len: usize) -> Vec<SetField> {
    (0..count)
        .map(|i| field(format!("v{i}"), vec![i as u8; len]))
        .collect()
}

/// Which handler path a fixture runs on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    Create,
    /// `UpdateLayer` with no structural command: the Lua fast path.
    UpdateFast,
    /// `UpdateLayer` charged `CommitPath::UpdateSlow`: structural commands and
    /// resets (`is_diff` with `base_version` 0) on any backend, and every delta
    /// under `txn`. Which commit route it takes is the measurement's
    /// [`Route`].
    UpdateSlow,
}

/// The commit route an `UpdateSlow` measurement takes. Every route is charged
/// `CommitPath::UpdateSlow`, so its constants must cover each of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// The service's default `lua` backend, with no route check: every
    /// `Create` and `UpdateFast` measurement.
    Default,
    /// `lua`, on a fresh structural cache: the cached structural route's cold
    /// start, which reads the object into the cache (`path`
    /// `structural_full`).
    CachedCold,
    /// `lua`, with the object already in the structural cache (warmed by one
    /// unmeasured structural commit): the cached route's plan in place on the
    /// held entry (`path` `structural`).
    CachedWarm,
    /// `txn`: the WATCH/MULTI/EXEC path (`path` `watch`) for every delta. The
    /// same commit code takes structural deltas under `lua-watch-structural`
    /// and `lua`'s fallbacks (an uncacheable object, a repeated cache
    /// divergence); `txn` also sends field and time-sample deltas there.
    Watch,
}

impl Route {
    /// The backend this route's measurements run on.
    fn backend(self) -> UpdateLayerBackend {
        match self {
            Route::Default | Route::CachedCold | Route::CachedWarm => UpdateLayerBackend::Lua,
            Route::Watch => UpdateLayerBackend::Txn,
        }
    }

    /// The `path` label an `UpdateSlow` commit of `delta` on this route must
    /// record; `None` when the route is not checked. Under `lua` a reset
    /// commits on `reset`, from a cache hit or a cold start alike.
    fn path_label(self, delta: Option<&OwnedDelta>) -> Option<&'static str> {
        let reset = delta.is_some_and(|d| d.is_diff && d.base_version == 0);
        match self {
            Route::Default => None,
            Route::CachedCold | Route::CachedWarm if reset => Some(UPDATE_LAYER_PATH_RESET),
            Route::CachedCold => Some(UPDATE_LAYER_PATH_STRUCTURAL_FULL),
            Route::CachedWarm => Some(UPDATE_LAYER_PATH_STRUCTURAL),
            Route::Watch => Some(UPDATE_LAYER_PATH_WATCH),
        }
    }

    /// Whether this route binds the production `term` (`"byte"` or
    /// `"element"`) of its path: the route the constant is derived from (the
    /// constants' docs in `grpc::upload_admission`; update this when a
    /// re-derivation moves one). On `Create` and `UpdateFast` the one route
    /// binds both; on `UpdateSlow` the cached route's cold start binds
    /// `byte`, and has the largest `element` figure net of the route's own
    /// object-state need, which the commit charges beside the processing
    /// charge (`cached_route_element_shapes_are_reported_ug2`'s
    /// `net_of_need`). The run path's own element figure is not netted of
    /// that need, so on this route it reads above the constant and prints no
    /// note; the constant, only ever raised, sits above the netted figure.
    fn binds(self, term: &str) -> bool {
        match self {
            Route::Default => true,
            Route::CachedCold => term == "byte" || term == "element",
            Route::CachedWarm | Route::Watch => false,
        }
    }
}

/// A root-level section the [`Route::CachedWarm`] warm-up commit creates, away
/// from `SECTION`'s children, so it changes no fixture's reorder or
/// children list.
const WARM_SECTION: u64 = 2;

impl Path {
    /// The production constants for this path: `(byte centi, element bytes,
    /// reorder-entry bytes)`.
    fn production(self) -> (u64, u64, u64) {
        match self {
            Path::Create => (
                DELTA_BYTE_CHARGE_CREATE_CENTI,
                ELEMENT_CHARGE_CREATE_BYTES,
                REORDER_ENTRY_CHARGE_CREATE_BYTES,
            ),
            Path::UpdateFast => (
                DELTA_BYTE_CHARGE_UPDATE_FAST_CENTI,
                ELEMENT_CHARGE_UPDATE_FAST_BYTES,
                0,
            ),
            Path::UpdateSlow => (
                DELTA_BYTE_CHARGE_UPDATE_SLOW_CENTI,
                ELEMENT_CHARGE_UPDATE_SLOW_BYTES,
                REORDER_ENTRY_CHARGE_UPDATE_SLOW_BYTES,
            ),
        }
    }
}

/// What a fixture's measurement fits or checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Feeds {
    /// `F`: the header-only upload.
    Fixed,
    /// `F` on a structural route: a one-child structural edit on a tiny base,
    /// whose peak net of its body and its byte and element terms is that
    /// route's fixed cost (a header-only `UpdateLayer` commits on the fast path
    /// under `lua`, so it never reaches the cached structural route).
    FixedStructural,
    /// `byte[path]`: inline values, few elements.
    Bytes,
    /// `element[path]`: many small elements.
    Elements,
    /// `reorder[path]`: a long reorder list.
    Reorder,
    /// `e_ext`: external values.
    Ext,
    /// Fitted from nothing: the cross-check.
    Mixed,
}

struct Fixture {
    name: &'static str,
    feeds: Feeds,
    /// The parsed form; `None` for the header-only upload.
    delta: Option<OwnedDelta>,
    /// The OMPF content; empty for the header-only upload.
    ompf: Vec<u8>,
    /// On the update paths, the layer the measured upload edits.
    base: Base,
    /// On [`Route::CachedWarm`], records a second store commits after the
    /// cache warm-up and before the measured call, so the call's cache hit
    /// repairs its entry by replaying them ([`commit_pending`]); empty for
    /// every other fixture.
    pending: Vec<PendingRecord>,
    /// On [`Route::CachedWarm`], a premise checked on the entry the measured
    /// call's cache hit takes — the store's own entry, read just before the
    /// call and outside its window ([`measure_on`]); `None` for every other
    /// fixture.
    entry_premise: Option<fn(&live_edit::storage::structural_cache::CacheEntry)>,
}

/// One record a second store commits ahead of a measured cache hit
/// ([`Fixture::pending`]): its delta and the external values its OMPF
/// carries.
#[derive(Clone)]
struct PendingRecord {
    delta: OwnedDelta,
    external: Vec<Vec<u8>>,
}

/// The base layer an update fixture edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base {
    /// One section, `SECTION`, with this many empty children of its own (ids
    /// from 10, as [`children`] makes them); 0 for the usual one-empty-section
    /// base.
    Children(usize),
    /// The load harness's seeded layer at 5,003 sections, fields included
    /// ([`harness_base_delta`]).
    Harness,
    /// The same shape at this many prims instead of [`HARNESS_PRIMS`]: 4 ×
    /// prims + 3 sections (the M1 fit's second content shape).
    HarnessPrims(u64),
    /// [`Base::Children`]'s `SECTION` with this many children, and beside it
    /// [`SECOND_SECTION`], an empty root-level section: a base where an edit
    /// under the empty section lists no long children list (the planning
    /// clone's fit basis).
    ChildrenAndEmptySection(usize),
    /// [`inline_value_base_delta`]: `SECTION` with [`INLINE_VALUE_ROWS`]
    /// inline values of [`INLINE_VALUE_LEN`] bytes and
    /// [`INLINE_VALUE_CHILDREN`] children.
    InlineValues,
    /// [`named_base`] with names of this many bytes: [`NAMED_ENTRIES`]
    /// sections and as many fields, each carrying an external value.
    Named(usize),
    /// [`external_rows_base`]: `SECTION` with this many fields, each carrying
    /// its own external value, so `obj:ext` holds as many distinct hashes.
    ExternalRows(usize),
    /// [`chain_base_delta`]: `SECTION` and below it a chain of this many
    /// sections, each the only child of the one before — a tree this deep.
    Chain(usize),
    /// `SECTION` with this many small inline fields keyed `f0…`
    /// ([`small_fields`]).
    Fields(usize),
    /// `SECTION` with `keep` small inline fields keyed `k0…`, and beside it
    /// [`SECOND_SECTION`] with `doomed` keyed `d0…`, every key a distinct
    /// interned name: deleting [`SECOND_SECTION`] purges `doomed` field rows
    /// and names from tables holding `keep + doomed`.
    SplitFields { keep: usize, doomed: usize },
    /// [`dev_assets_base_delta`]: the dev shape's placed assets, `assets` of
    /// them, padded with empty sections to exactly `sections` sections.
    DevAssets { assets: usize, sections: usize },
}

/// [`Base::ChildrenAndEmptySection`]'s empty root-level section (clear of
/// [`WARM_SECTION`] and of the harness ids).
const SECOND_SECTION: u64 = 3;

/// [`Base::InlineValues`]: one fan-out window's worth of value rows at the
/// default item ceiling (2,141 commands), each an inline value of 32 KiB —
/// larger than either per-command charge (`W_CMD` on a script window,
/// `W_PLAIN_CMD` on a plain one), so a read that transferred them would
/// receive far more than the charge it took before the window —
/// beside a long children list. 2,141 × 32 KiB (about 70 MB) keeps the base
/// layer under the handler's 100 MiB layer size cap.
const INLINE_VALUE_ROWS: usize = 2_141;
const INLINE_VALUE_LEN: usize = 32 * 1024;
const INLINE_VALUE_CHILDREN: usize = 5_000;

/// The `CreateLayer` delta of [`Base::InlineValues`]: `SECTION` with
/// [`INLINE_VALUE_CHILDREN`] empty children (ids from 10) and
/// [`INLINE_VALUE_ROWS`] fields keyed `v0…`, each an inline value of
/// [`INLINE_VALUE_LEN`] bytes, distinct per key.
fn inline_value_base_delta() -> OwnedDelta {
    OwnedDelta {
        struct_commands: std::iter::once(create_section(0, SECTION, "base".to_owned()))
            .chain(children(INLINE_VALUE_CHILDREN, false))
            .collect(),
        set_fields: (0..INLINE_VALUE_ROWS)
            .map(|i| {
                let mut value = vec![i as u8; INLINE_VALUE_LEN];
                value[..4].copy_from_slice(&(i as u32).to_le_bytes());
                field(format!("v{i}"), value)
            })
            .collect(),
        ..Default::default()
    }
}

/// `count` blobs of [`NAMED_EXT_LEN`] bytes, each carrying its index, so all
/// are distinct whatever `count` is ([`blobs`] repeats from 256 on).
fn distinct_blobs(count: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..count)
        .map(|b| {
            let mut blob = vec![salt; NAMED_EXT_LEN];
            blob[..8].copy_from_slice(&(b as u64).to_le_bytes());
            blob
        })
        .collect()
}

/// The `CreateLayer` delta of [`Base::ExternalRows`] and its external
/// values: `SECTION` with `rows` fields keyed `x0…`, each carrying its own
/// distinct external value, so `obj:ext` holds `rows` distinct hashes.
fn external_rows_base(rows: usize) -> (OwnedDelta, Vec<Vec<u8>>) {
    let values = distinct_blobs(rows, 0x3c);
    let set_fields = values
        .iter()
        .enumerate()
        .map(|(i, value)| external_field(format!("x{i}"), value, i as u32 + 1))
        .collect();
    (
        OwnedDelta {
            struct_commands: vec![create_section(0, SECTION, "base".to_owned())],
            set_fields,
            ..Default::default()
        },
        values,
    )
}

impl Fixture {
    fn new(name: &'static str, feeds: Feeds, delta: OwnedDelta, external: &[&[u8]]) -> Self {
        let ompf = common::owned_delta::delta_to_ompf_bytes(&delta, external);
        Self {
            name,
            feeds,
            delta: Some(delta),
            ompf,
            base: Base::Children(0),
            pending: Vec::new(),
            entry_premise: None,
        }
    }

    fn header_only() -> Self {
        Self {
            name: "header-only",
            feeds: Feeds::Fixed,
            delta: None,
            ompf: Vec::new(),
            base: Base::Children(0),
            pending: Vec::new(),
            entry_premise: None,
        }
    }

    /// This fixture, measured on `base`.
    fn on_base(self, base: Base) -> Self {
        Self { base, ..self }
    }

    /// This fixture, measured on [`Route::CachedWarm`] after a second store
    /// commits `pending` ([`Fixture::pending`]).
    fn after_pending(self, pending: Vec<PendingRecord>) -> Self {
        Self { pending, ..self }
    }

    /// This fixture, its measured cache hit's entry checked by `premise`
    /// ([`Fixture::entry_premise`]).
    fn with_entry_premise(
        self,
        premise: Option<fn(&live_edit::storage::structural_cache::CacheEntry)>,
    ) -> Self {
        Self {
            entry_premise: premise,
            ..self
        }
    }
}

/// The structural commands a fixture on `path` starts with: a `CreateLayer`
/// creates `SECTION` itself; an `UpdateLayer` finds it in its base layer.
fn own_section(path: Path) -> Vec<StructCommand> {
    if path == Path::Create {
        vec![create_section(0, SECTION, "base".to_owned())]
    } else {
        Vec::new()
    }
}

/// `n` new child sections of `SECTION`, ids from 10, and a reorder of them
/// when `reorder` is set.
fn children(n: usize, reorder: bool) -> Vec<StructCommand> {
    let ids: Vec<u64> = (0..n as u64).map(|i| 10 + i).collect();
    let mut out: Vec<StructCommand> = ids
        .iter()
        .map(|&id| create_section(SECTION, id, format!("c{id}")))
        .collect();
    if reorder {
        out.push(StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: SECTION,
            children_list_id: 0,
            children_list: ids.iter().rev().copied().collect(),
        }));
    }
    out
}

/// Every fixture of `path`, header-only first.
fn fixtures(path: Path) -> Vec<Fixture> {
    let base = own_section(path);
    let with = |struct_commands: Vec<StructCommand>,
                set_fields: Vec<SetField>,
                time_samples: Vec<TimeSample>| OwnedDelta {
        struct_commands: base.iter().cloned().chain(struct_commands).collect(),
        set_fields,
        time_samples,
        ..Default::default()
    };
    let mut out = vec![Fixture::header_only()];

    // F on the structural routes: the smallest structural edit, one new child.
    if path == Path::UpdateSlow {
        out.push(Fixture::new(
            "one-child-structural",
            Feeds::FixedStructural,
            with(children(1, false), Vec::new(), Vec::new()),
            &[],
        ));
    }

    // byte[path]: 64 inline values of 64 KiB. The slow path needs a
    // structural command to take it: one new child section.
    let value_structure = if path == Path::UpdateSlow {
        children(1, false)
    } else {
        Vec::new()
    };
    out.push(Fixture::new(
        "value-heavy",
        Feeds::Bytes,
        with(value_structure, value_fields(VALUES, VALUE_LEN), Vec::new()),
        &[],
    ));

    // element[path].
    if path != Path::UpdateSlow {
        out.push(Fixture::new(
            "field-dense",
            Feeds::Elements,
            with(
                Vec::new(),
                (0..DENSE)
                    .map(|i| field(format!("f{i}"), (i as u64).to_le_bytes().to_vec()))
                    .collect(),
                Vec::new(),
            ),
            &[],
        ));
        out.push(Fixture::new(
            "time-sample-dense",
            Feeds::Elements,
            with(
                Vec::new(),
                Vec::new(),
                (0..DENSE).map(time_sample).collect(),
            ),
            &[],
        ));
    }
    if path != Path::UpdateFast {
        out.push(Fixture::new(
            "structure-dense",
            Feeds::Elements,
            with(children(DENSE, false), Vec::new(), Vec::new()),
            &[],
        ));
        // reorder[path].
        out.push(Fixture::new(
            "long-reorder-list",
            Feeds::Reorder,
            with(children(DENSE, true), Vec::new(), Vec::new()),
            &[],
        ));
    }
    // element[UpdateSlow] again, as a reset (`is_diff`, `base_version` 0)
    // that replaces the layer with the structure-dense content: a reset is
    // charged `UpdateSlow` and commits on `lua`'s `reset` path.
    if path == Path::UpdateSlow {
        out.push(Fixture::new(
            "reset-structure-dense",
            Feeds::Elements,
            OwnedDelta {
                is_diff: true,
                base_version: 0,
                struct_commands: std::iter::once(create_section(0, SECTION, "base".to_owned()))
                    .chain(children(DENSE, false))
                    .collect(),
                ..Default::default()
            },
            &[],
        ));
    }

    // e_ext: one large blob, and more blobs than the store keeps in flight
    // at once (8). The slow path again takes one structural command.
    let salt = path as u8;
    for (name, values) in [
        ("one-large-blob", blobs(1, 16 * MIB, salt)),
        ("many-small-blobs", blobs(32, 128 * 1024, salt ^ 0x5a)),
    ] {
        let refs: Vec<&[u8]> = values.iter().map(Vec::as_slice).collect();
        let structure = if path == Path::UpdateSlow {
            children(1, false)
        } else {
            Vec::new()
        };
        let set_fields = refs
            .iter()
            .enumerate()
            .map(|(i, b)| external_field(format!("blob{i}"), b, i as u32 + 1))
            .collect();
        out.push(Fixture::new(
            name,
            Feeds::Ext,
            with(structure, set_fields, Vec::new()),
            &refs,
        ));
    }

    // The cross-check: a bit of everything the path takes.
    let mixed_structure = match path {
        Path::UpdateFast => Vec::new(),
        _ => children(2_000, true),
    };
    let mixed_blobs = blobs(4, 256 * 1024, salt ^ 0x33);
    let mixed_refs: Vec<&[u8]> = mixed_blobs.iter().map(Vec::as_slice).collect();
    let mut mixed_fields: Vec<SetField> = (0..4_000)
        .map(|i| field(format!("m{i}"), (i as u64).to_le_bytes().to_vec()))
        .collect();
    mixed_fields.extend(value_fields(16, VALUE_LEN));
    mixed_fields.extend(
        mixed_refs
            .iter()
            .enumerate()
            .map(|(i, b)| external_field(format!("mb{i}"), b, i as u32 + 1)),
    );
    out.push(Fixture::new(
        "mixed",
        Feeds::Mixed,
        with(
            mixed_structure,
            mixed_fields,
            (0..4_000).map(time_sample).collect(),
        ),
        &mixed_refs,
    ));
    out
}

// ═════════════════════════════════════════════════════════════════════════════
// The measured call
// ═════════════════════════════════════════════════════════════════════════════

/// One gRPC frame per message.
fn grpc_frame<T: Message>(msg: &T) -> Bytes {
    let encoded = msg.encode_to_vec();
    let mut buf = bytes::BytesMut::with_capacity(5 + encoded.len());
    buf.put_u8(0);
    buf.put_u32(encoded.len() as u32);
    buf.extend_from_slice(&encoded);
    buf.freeze()
}

/// A request over `frames`, one HTTP/2 DATA frame each, ending with
/// END_STREAM and carrying its end-of-stream record, as the LiveService stack
/// hands an upload to its handler.
fn request<T: Message + Default + Send + 'static>(frames: &[Bytes]) -> Request<Streaming<T>> {
    let items: Vec<Result<Frame<Bytes>, Infallible>> =
        frames.iter().cloned().map(|b| Ok(Frame::data(b))).collect();
    // `StreamBody` never reports END_STREAM; the wrapper reports it once the
    // frames are exhausted, as h2 does for a client that finished its stream.
    let body = EndsWithEndStream::new(http_body_util::StreamBody::new(futures::stream::iter(
        items,
    )));
    make_request(AttestedStreaming::new(body))
}

/// The service under measurement: the real external-value store, a 100 MiB
/// size cap, and a fresh gate at the production defaults, on the default
/// `lua` backend.
fn service(tv: &TestValkey) -> LiveEditService {
    service_for(tv, Route::Default)
}

/// One metrics registry for every service this process measures, as
/// production has one per replica: a fresh registry per call would put each
/// metric series' first-use allocation inside the measured window. The
/// unmeasured warm-up calls ([`warm_up`]) create the series, and run the
/// once-per-process code of each route, before anything is measured.
fn shared_metrics() -> Arc<live_edit::observability::metrics::Metrics> {
    static METRICS: std::sync::OnceLock<Arc<live_edit::observability::metrics::Metrics>> =
        std::sync::OnceLock::new();
    Arc::clone(METRICS.get_or_init(|| Arc::new(common::metrics_scrape_only("t7"))))
}

/// [`service`] on `route`'s backend, with a fresh `LayerStore` (so a fresh,
/// empty structural cache) recording into [`shared_metrics`], which the route
/// check reads.
fn service_for(tv: &TestValkey, route: Route) -> LiveEditService {
    service_on(tv, route, tv.client().clone())
}

/// [`service_for`] with its store on `client`.
fn service_on(
    tv: &TestValkey,
    route: Route,
    client: live_edit::storage::client::ValkeyClient,
) -> LiveEditService {
    service_on_with(tv, route, client, |store| store)
}

/// [`service_on`] with `tweak` applied to its store before it is shared
/// (a test hook, say).
fn service_on_with(
    tv: &TestValkey,
    route: Route,
    client: live_edit::storage::client::ValkeyClient,
    tweak: impl FnOnce(LayerStore) -> LayerStore,
) -> LiveEditService {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let metrics = shared_metrics();
    let layer_store = Arc::new(tweak(
        LayerStore::new(client, tv.key_schema().clone(), 4096)
            .with_update_layer_backend(route.backend())
            .with_metrics(Arc::clone(&metrics)),
    ));
    // A memory measurement: the shadow check's copy of each planned entry
    // is exactly what the route no longer allocates.
    layer_store
        .structural_cache()
        .set_shadow_check_for_test(false);
    // The recovery hook holds the store it restores into; rebuild it over this
    // one (it is disabled either way), as
    // `helpers::build_service_with_layer_store` does.
    let recovery =
        helpers::build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&metrics));
    LiveEditService {
        metrics,
        layer_store,
        recovery,
        ext_value_store: Arc::new(ValkeyExternalValueStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        )),
        max_layer_size: 100 * MIB,
        upload_admission: Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig::default(),
            None,
        )),
        ..base
    }
}

#[derive(Clone)]
struct Measured {
    path: Path,
    route: Route,
    name: &'static str,
    feeds: Feeds,
    body: u64,
    delta: u64,
    ext: u64,
    elements: u64,
    reorder: u64,
    /// The whole call's heap growth peak: the largest phase peak.
    peak: u64,
    /// The call's high-water: its largest reservation, `A` included.
    charge: u64,
    /// The call's phases, in order ([`PhaseRecorder`]).
    phases: Vec<Phase>,
    /// The first processing step's target ([`parse_charge`]).
    parse_target: u64,
    /// The second processing step's target ([`processing_charge`]).
    processing_target: u64,
    /// The live bytes the call left above its baseline when it returned, after
    /// its reservation was released: whatever outlives the call, such as the
    /// entry a cached route's cold start put in the structural cache.
    residual: u64,
    /// On [`Route::CachedCold`], the bytes the entry the cold start stored in
    /// the structural cache holds: what invalidating it after the call freed.
    /// 0 on every other route. The one residue the released phase is allowed
    /// ([`phase_limits`]).
    cache_entry: u64,
    /// On the update paths, the object the measured upload edited; 0 on
    /// `Create`.
    object_id: u64,
    /// The handler call's wall time (report only: [`measure_on`] prints it).
    wall: std::time::Duration,
    /// What the measured call planned and committed ([`PlanCounts`]).
    plans: PlanCounts,
}

/// What one measured call planned and committed (plans per commit under a
/// charge), as the differences of counters read just before and
/// just after the handler call, outside its measured window ([`measure_on`]).
#[derive(Clone, Copy, Debug, Default)]
struct PlanCounts {
    /// The cached route's plans, made in place on the held entry
    /// (`StructuralCache::in_place_plans_for_test`): one per plan.
    in_place_plans: u64,
    /// `structural_update` whole-object reads (`live_edit_snapshot_read_total`,
    /// every path and outcome): one per WATCH plan. Recorded only when the
    /// store's client carries the registry's hooks; 0 otherwise.
    watch_reads: f64,
    /// Commits (`live_edit_update_layer_attempts_histogram_count`, every path).
    commits: f64,
    /// The attempts those commits took (`…_sum`).
    attempts: f64,
    /// Waits for a larger charge (`live_edit_update_layer_recharges_total`,
    /// every path).
    recharges: f64,
}

impl PlanCounts {
    /// The counters now, on `svc`'s store and registry. Renders the registry,
    /// so it allocates: call it outside a measured window.
    fn read(svc: &LiveEditService) -> Self {
        let text = render(&svc.metrics);
        let sum =
            |name: &str, labels: &[(&str, &str)]| sample_sum(&text, name, labels).unwrap_or(0.0);
        Self {
            in_place_plans: svc.layer_store.structural_cache().in_place_plans_for_test(),
            watch_reads: sum(
                "live_edit_snapshot_read_total",
                &[("consumer", "structural_update")],
            ),
            commits: sum("live_edit_update_layer_attempts_histogram_count", &[]),
            attempts: sum("live_edit_update_layer_attempts_histogram_sum", &[]),
            recharges: sum("live_edit_update_layer_recharges_total", &[]),
        }
    }

    /// What changed since `before`.
    fn since(self, before: Self) -> Self {
        Self {
            in_place_plans: self.in_place_plans - before.in_place_plans,
            watch_reads: self.watch_reads - before.watch_reads,
            commits: self.commits - before.commits,
            attempts: self.attempts - before.attempts,
            recharges: self.recharges - before.recharges,
        }
    }
}

impl Measured {
    /// [`phase_labels`], with the cold start's read named on
    /// [`Route::CachedCold`] ([`COLD_START_READ`]).
    ///
    /// On that route the structural commit first reads the object it has
    /// not cached, holding the read's own permit or, for a fenced read, its
    /// grant of the catch-up build budget `T` — not the upload budget `B` —
    /// across the read and the decode, and only then charges the decoded
    /// entry to the upload (CR0, at seam A, before those permits drop). So
    /// the processing phases from the processing step to
    /// that first object-state growth hold the read under its own budget, by
    /// design, and are reported, not checked against the upload's charge.
    /// The charge cannot tell the read apart from the handler's own work in
    /// those phases (the external values' store, under the same processing
    /// step), so that work goes unchecked here; it is the same on every route
    /// and checked, under the same charge, on [`Route::CachedWarm`] and
    /// [`Route::Watch`].
    /// They are the processing-labelled phases holding exactly the processing
    /// step's charge (a fresh cache gives no hint) before the first phase
    /// that holds more — only when one does: a commit that charged nothing
    /// beyond its processing step (a delta that takes no structural plan, as
    /// the header-only fixture's) read no object, and none of its phases is
    /// exempt.
    fn labels(&self) -> Vec<&'static str> {
        let mut labels = phase_labels(&self.phases, self.parse_target, self.processing_target);
        if self.route != Route::CachedCold {
            return labels;
        }
        let first_growth = self
            .phases
            .iter()
            .zip(&labels)
            .position(|(p, label)| *label == "processing" && p.held() > self.processing_target);
        let Some(first_growth) = first_growth else {
            return labels;
        };
        for (p, label) in self.phases[..first_growth]
            .iter()
            .zip(labels.iter_mut())
            .filter(|(_, label)| **label == "processing")
        {
            if p.held() == self.processing_target {
                *label = COLD_START_READ;
            }
        }
        labels
    }

    /// Each phase whose peak passed the charge in force during it
    /// ([`phase_limits`]), described. The cold start's read
    /// ([`COLD_START_READ`]) runs under its own permit or grant, not the
    /// upload's charge, and is not checked.
    fn phase_overruns(&self) -> Vec<String> {
        let labels = self.labels();
        self.phases
            .iter()
            .zip(phase_limits(&self.phases, self.cache_entry))
            .zip(labels)
            .enumerate()
            .filter(|(_, ((_, _), label))| *label != COLD_START_READ)
            .filter(|(_, ((p, limit), _))| p.peak > *limit)
            .map(|(i, ((p, limit), label))| {
                format!(
                    "phase {i} ({label}): {} B over a {limit} B charge ({:.3}x)",
                    p.peak,
                    p.peak as f64 / limit.max(1) as f64
                )
            })
            .collect()
    }

    /// The phases, consecutive phases of one label merged into one entry
    /// carrying their highest peak against its charge, as `label peak/charge
    /// (ratio)`.
    fn phase_summary(&self) -> String {
        let labels = self.labels();
        let mut merged: Vec<(&str, u64, u64)> = Vec::new();
        for ((p, limit), label) in self
            .phases
            .iter()
            .zip(phase_limits(&self.phases, self.cache_entry))
            .zip(labels)
        {
            match merged.last_mut() {
                Some((last, peak, charge)) if *last == label => {
                    if p.peak as f64 / limit.max(1) as f64 > *peak as f64 / (*charge).max(1) as f64
                    {
                        (*peak, *charge) = (p.peak, limit);
                    }
                }
                _ => merged.push((label, p.peak, limit)),
            }
        }
        merged
            .iter()
            .map(|(label, peak, charge)| {
                format!(
                    "{label} {peak}/{charge} ({:.3})",
                    *peak as f64 / (*charge).max(1) as f64
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// The most held in any phase labelled `label`; 0 when none is. On
    /// `processing`, the processing step's charge plus the largest
    /// object-state term the commit took.
    fn phase_held_max(&self, label: &str) -> u64 {
        let labels = phase_labels(&self.phases, self.parse_target, self.processing_target);
        self.phases
            .iter()
            .zip(labels)
            .filter(|(_, l)| *l == label)
            .map(|(p, _)| p.held())
            .max()
            .unwrap_or(0)
    }

    /// The highest peak among the phases labelled `label`; 0 when none is.
    fn phase_peak(&self, label: &str) -> u64 {
        let labels = phase_labels(&self.phases, self.parse_target, self.processing_target);
        self.phases
            .iter()
            .zip(labels)
            .filter(|(_, l)| *l == label)
            .map(|(p, _)| p.peak)
            .max()
            .unwrap_or(0)
    }
}

/// Root ids of [`Base::Harness`]: the root, `World` and its one zone (ids
/// clear of [`WARM_SECTION`]).
const HARNESS_WORLD: u64 = 4;
const HARNESS_ZONE: u64 = 5;
/// The first of [`Base::Harness`]'s prim ids; each prim's attributes take
/// the next three.
const HARNESS_FIRST_PRIM: u64 = 100;
/// [`Base::Harness`]'s prims: 1,250 × (the prim + 3 attributes) + root,
/// `World` and the zone = 5,003 sections.
const HARNESS_PRIMS: u64 = 1_250;
/// `SdfSpecType`s, as the harness's client writes them into `section_type`.
const SPEC_ATTRIBUTE: u8 = 1;
const SPEC_PRIM: u8 = 6;
const SPEC_PSEUDO_ROOT: u8 = 7;

/// The `CreateLayer` delta of [`Base::Harness`]: the load harness's seeded
/// layer at `--seed-prims sections=5000`, in its shape and sizes — 5,003
/// sections (the root, `World`, one zone, 1,250 prims of three attributes each)
/// and 17,504 fields (two on `World`, the zone and each prim, four on each
/// attribute, 2–24 B) — with a
/// fixed topology instead of the harness's random one: every even prim under
/// the zone, every odd one under the prim before it. `prims` is
/// [`HARNESS_PRIMS`] for [`Base::Harness`]; [`Base::HarnessPrims`] scales it.
fn harness_base_delta(prims: u64) -> OwnedDelta {
    let create = |parent_id, section_id, name: String, section_type| {
        StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name,
            section_type,
        })
    };
    let set = |section_id: u64, key: &str, len: usize| SetField {
        section_id,
        key_name: key.to_owned(),
        value_or_ext_hash: (0..len)
            .map(|i| b'A' + ((section_id as usize + i) % 26) as u8)
            .collect(),
        ext_value_size: 0,
        ext_value_index: 0,
        set_order: 0,
    };
    let mut struct_commands = vec![
        create(0, SECTION, "root".to_owned(), SPEC_PSEUDO_ROOT),
        create(SECTION, HARNESS_WORLD, "World".to_owned(), SPEC_PRIM),
        create(HARNESS_WORLD, HARNESS_ZONE, "Zone0".to_owned(), SPEC_PRIM),
    ];
    let mut set_fields = Vec::new();
    for id in [HARNESS_WORLD, HARNESS_ZONE] {
        set_fields.push(set(id, "specifier", 4));
        set_fields.push(set(id, "typeName", 9));
    }
    for i in 0..prims {
        let prim = HARNESS_FIRST_PRIM + 4 * i;
        let parent = if i % 2 == 0 { HARNESS_ZONE } else { prim - 4 };
        struct_commands.push(create(
            parent,
            prim,
            format!("w{}_p{}", i % 8, i / 8 + 1),
            SPEC_PRIM,
        ));
        set_fields.push(set(prim, "specifier", 4));
        set_fields.push(set(prim, "typeName", 8));
        for (k, attr) in ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
            .into_iter()
            .enumerate()
        {
            let id = prim + 1 + k as u64;
            struct_commands.push(create(prim, id, attr.to_owned(), SPEC_ATTRIBUTE));
            for (key, len) in [
                ("typeName", 8),
                ("custom", 2),
                ("variability", 4),
                ("default", 24),
            ] {
                set_fields.push(set(id, key, len));
            }
        }
    }
    OwnedDelta {
        struct_commands,
        set_fields,
        ..Default::default()
    }
}

/// One new prim under [`Base::Harness`]'s zone.
fn one_new_prim_in_harness_zone() -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: HARNESS_ZONE,
            section_id: 99,
            section_name: "extra".to_owned(),
            section_type: SPEC_PRIM,
        })],
        ..Default::default()
    }
}

/// A new session for one measured call, created through the store.
///
/// Boxed, like every handler call and layer-store call `base_layer` and
/// `measure_on` await. Inline, each call's future is held inside every helper
/// future above it (`base_layer`, `measure_on`, `measure_route`, `warm_up`,
/// `run_path`), and in a debug build each of those helpers' poll frames on the
/// test thread's stack holds several copies of the future it awaits: the
/// session-create future alone is about 45 KB there, and the stack was about
/// 1 MB deep where the measured handler began. The WATCH route's `UpdateLayer`
/// handler path takes about 1.2 MB more in a debug build (about 86 KB in
/// release), which overflowed std's default 2 MiB test-thread stack; boxed, that test
/// passes at 1.4 MiB. Tonic polls handlers from the heap, so boxing changes
/// nothing the handler sees; the setup calls' boxes (this one, `base_layer`'s
/// `CreateLayer`, the cache warm-up) are made outside every measurement
/// window.
async fn new_session(tv: &TestValkey, label: &str) -> String {
    Box::pin(
        SessionStore::new(tv.client().clone(), tv.key_schema().clone()).create(
            label,
            "/stage",
            "test-user",
        ),
    )
    .await
    .expect("session creation")
}

/// The `CreateLayer` delta of `base`.
fn base_delta(base: Base) -> OwnedDelta {
    base_content(base).0
}

/// The `CreateLayer` delta of `base` and the external values its OMPF
/// carries (none but on [`Base::Named`] and [`Base::ExternalRows`]).
fn base_content(base: Base) -> (OwnedDelta, Vec<Vec<u8>>) {
    let children_of_section = |n| OwnedDelta {
        struct_commands: std::iter::once(create_section(0, SECTION, "base".to_owned()))
            .chain(children(n, false))
            .collect(),
        ..Default::default()
    };
    match base {
        Base::Children(n) => (children_of_section(n), Vec::new()),
        Base::Harness => (harness_base_delta(HARNESS_PRIMS), Vec::new()),
        Base::HarnessPrims(prims) => (harness_base_delta(prims), Vec::new()),
        Base::ChildrenAndEmptySection(n) => {
            let mut delta = children_of_section(n);
            delta
                .struct_commands
                .push(create_section(0, SECOND_SECTION, "second".to_owned()));
            (delta, Vec::new())
        }
        Base::InlineValues => (inline_value_base_delta(), Vec::new()),
        Base::Named(name_len) => named_base(name_len),
        Base::ExternalRows(rows) => external_rows_base(rows),
        Base::Chain(n) => (chain_base_delta(n), Vec::new()),
        Base::Fields(n) => (
            OwnedDelta {
                struct_commands: vec![create_section(0, SECTION, "base".to_owned())],
                set_fields: small_fields(SECTION, "f", n),
                ..Default::default()
            },
            Vec::new(),
        ),
        Base::SplitFields { keep, doomed } => (
            OwnedDelta {
                struct_commands: vec![
                    create_section(0, SECTION, "base".to_owned()),
                    create_section(0, SECOND_SECTION, "second".to_owned()),
                ],
                set_fields: small_fields(SECTION, "k", keep)
                    .into_iter()
                    .chain(small_fields(SECOND_SECTION, "d", doomed))
                    .collect(),
                ..Default::default()
            },
            Vec::new(),
        ),
        Base::DevAssets { assets, sections } => {
            (dev_assets_base_delta(assets, sections), Vec::new())
        }
    }
}

/// `n` small inline fields on `section_id`, keyed `{prefix}0…`, each value
/// its index's 8 bytes.
fn small_fields(section_id: u64, prefix: &str, n: usize) -> Vec<SetField> {
    (0..n)
        .map(|i| SetField {
            section_id,
            ..field(format!("{prefix}{i}"), (i as u64).to_le_bytes().to_vec())
        })
        .collect()
}

/// [`Base::Chain`]'s `CreateLayer` delta: `SECTION` at the root, then ids
/// `10 .. 10 + n`, the first under `SECTION` and each later one under the
/// one before, so the deepest, [`chain_bottom`], is at depth `n + 1`.
fn chain_base_delta(n: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: std::iter::once(create_section(0, SECTION, "base".to_owned()))
            .chain((0..n as u64).map(|i| {
                let parent = if i == 0 { SECTION } else { 10 + i - 1 };
                create_section(parent, 10 + i, format!("c{i}"))
            }))
            .collect(),
        ..Default::default()
    }
}

/// The deepest section of [`Base::Chain`]`(n)`.
fn chain_bottom(n: usize) -> u64 {
    10 + n as u64 - 1
}

/// The first id [`dev_assets_base_delta`] allocates: its first group's;
/// `DEV_ASSET0_ROOT` follows it.
const DEV_FIRST_ID: u64 = 1_000_000;

/// The root prim of [`dev_assets_base_delta`]'s first asset, which has four
/// prims (28 sections, 105 field rows). Gated as its only users are.
#[cfg(feature = "test-fault-injection")]
const DEV_ASSET0_ROOT: u64 = DEV_FIRST_ID + 1;

/// [`Base::DevAssets`]'s `CreateLayer` delta, the dev shape of the step-0
/// record: `SECTION` (`World`), a `Group_<g>` under it per 100 placed
/// assets, and each asset a root prim with 3 or 4 child prims, alternating
/// (4 prims first), every prim carrying 6 attribute sections — 28 or 35
/// sections an asset, 31.5 on average. Field rows: `specifier` and
/// `typeName` on every prim and `references` on the asset root; `custom`,
/// `default`, `typeName` and `variability` on every attribute; values are 8
/// inline bytes. Then a `Pad` section under `SECTION` and empty sections
/// under it, up to exactly `sections` sections in all (none if the assets
/// already reach it).
fn dev_assets_base_delta(assets: usize, sections: usize) -> OwnedDelta {
    let mut next = DEV_FIRST_ID;
    let mut id = || {
        let id = next;
        next += 1;
        id
    };
    let mut struct_commands = vec![create_section(0, SECTION, "World".to_owned())];
    let mut set_fields = Vec::new();
    let mut fields_on = |section_id: u64, keys: &[&str]| {
        for key in keys {
            set_fields.push(SetField {
                section_id,
                ..field((*key).to_owned(), section_id.to_le_bytes().to_vec())
            });
        }
    };
    let mut group = 0;
    for asset in 0..assets {
        if asset % 100 == 0 {
            group = id();
            struct_commands.push(create_section(
                SECTION,
                group,
                format!("Group_{}", asset / 100),
            ));
        }
        let root = id();
        let prims = if asset % 2 == 0 { 4 } else { 5 };
        let mut prim_ids = vec![root];
        struct_commands.push(create_section(group, root, format!("Asset_{asset}")));
        fields_on(root, &["specifier", "typeName", "references"]);
        for p in 1..prims {
            let prim = id();
            struct_commands.push(create_section(root, prim, format!("Prim_{p}")));
            fields_on(prim, &["specifier", "typeName"]);
            prim_ids.push(prim);
        }
        for prim in prim_ids {
            for a in 0..6 {
                let attr = id();
                struct_commands.push(create_section(prim, attr, format!("attr_{a}")));
                fields_on(attr, &["custom", "default", "typeName", "variability"]);
            }
        }
    }
    if struct_commands.len() < sections {
        let pad = id();
        struct_commands.push(create_section(SECTION, pad, "Pad".to_owned()));
        while struct_commands.len() < sections {
            let section = id();
            struct_commands.push(create_section(pad, section, format!("p{section}")));
        }
    }
    OwnedDelta {
        struct_commands,
        set_fields,
        ..Default::default()
    }
}

/// A layer whose base state is `base`, for the update paths.
async fn base_layer(tv: &TestValkey, label: &str, base: Base) -> (String, u64) {
    let session_id = new_session(tv, label).await;
    // The measured services' layer size cap ([`service_on_with`]), above the
    // test configuration's 10 MiB: [`Base::InlineValues`] is about 70 MB.
    let setup = LiveEditService {
        max_layer_size: 100 * MIB,
        ..build_service(tv, Arc::new(AllowAllPermissions))
    };
    let (delta, values) = base_content(base);
    let external: Vec<&[u8]> = values.iter().map(Vec::as_slice).collect();
    let base = common::owned_delta::delta_to_ompf_bytes(&delta, &external);
    let messages = std::iter::once(make_create_header(&session_id, "layer"))
        .chain(base.chunks(CHUNK).map(|c| make_create_data(c.to_vec())))
        .collect();
    let response = Box::pin(handle_create_layer(
        &setup,
        make_request(streaming_create(messages)),
    ))
    .await
    .expect("base layer creation");
    (session_id, response.into_inner().object_id)
}

/// Run one upload of `fixture` on `path` and measure it, on a fresh
/// [`service`].
async fn measure(tv: &TestValkey, path: Path, fixture: &Fixture, label: &str) -> Measured {
    measure_on(service(tv), tv, path, Route::Default, fixture, label).await
}

/// [`measure`] on `route`, through a fresh [`service_for`] it.
async fn measure_route(
    tv: &TestValkey,
    path: Path,
    route: Route,
    fixture: &Fixture,
    label: &str,
) -> Measured {
    measure_on(service_for(tv, route), tv, path, route, fixture, label).await
}

/// The `live_edit_update_layer_attempts_histogram` count for `path_label` on
/// `svc`'s metrics: one per `UpdateLayer` commit on that path.
fn commits_on(svc: &LiveEditService, path_label: &str) -> f64 {
    sample_value(
        &render(&svc.metrics),
        "live_edit_update_layer_attempts_histogram_count",
        &[("path", path_label)],
    )
    .unwrap_or(0.0)
}

/// Whether `B26E_TRACK_FIXTURE` names `fixture`: [`measure_on`] then tracks
/// every allocation of its measured call and prints the tracked allocations
/// still live when the handler returns and after [`SETTLE`] (report only;
/// `B26E_BACKTRACE_MIN` sets the smallest allocation backtraced). Off by
/// default.
fn fixture_tracked(fixture: &str) -> bool {
    std::env::var("B26E_TRACK_FIXTURE").is_ok_and(|v| v == fixture)
}

/// Start tracking the measured call of `fixture` when [`fixture_tracked`],
/// with the peak-time view on.
fn begin_fixture_tracking(fixture: &str) {
    if fixture_tracked(fixture) {
        begin_peak_tracking();
    }
}

/// Start tracking every allocation, with the peak-time view on, from the
/// live bytes now (`B26E_BACKTRACE_MIN` sets the smallest allocation
/// backtraced).
fn begin_peak_tracking() {
    {
        let backtrace_min = std::env::var("B26E_BACKTRACE_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(POST_RELEASE_BACKTRACE_MIN);
        let _guard = TrackerGuard::enter();
        let freed = std::mem::take(
            &mut *TRACKED_FREED
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        drop(freed);
        let live = LIVE.load(Ordering::SeqCst);
        TRACK_BASE_LIVE.store(live, Ordering::SeqCst);
        TRACK_PEAK_LIVE.store(live, Ordering::SeqCst);
        TRACK_PEAK_EVENT.store(0, Ordering::SeqCst);
        TRACK_AT_PEAK.store(true, Ordering::SeqCst);
        start_tracking(backtrace_min);
    }
}

/// Print what `fixture`'s tracked call still holds, `when`, when
/// [`fixture_tracked`]; at its return, first what it held at its peak
/// ([`report_tracked_at_peak`]).
fn report_fixture_tracking(fixture: &str, label: &str, when: &str) {
    if fixture_tracked(fixture) {
        stop_recording();
        if when == "at return" {
            report_tracked_at_peak(&format!("{label}, at its peak"), 16);
        }
        report_tracked(&format!("{label}, {when}"), 12);
    }
}

/// Run one upload of `fixture` on `path` through `svc` and measure it, phase
/// by phase ([`PhaseRecorder`], set as the gate's charge observer for the
/// call). `svc` must carry a fresh gate that no other upload uses: its
/// high-water marks are read as the call's high-water, and its charge as the
/// call's. On an `UpdateSlow` [`Route`] other than `Default`, the measured
/// commit is checked to have taken that route.
async fn measure_on(
    svc: LiveEditService,
    tv: &TestValkey,
    path: Path,
    route: Route,
    fixture: &Fixture,
    label: &str,
) -> Measured {
    let (body, delta, ext) = {
        let c = OmpfContainer::parse(&fixture.ompf).expect("fixtures are valid OMPF");
        (
            fixture.ompf.len() as u64,
            c.delta_bytes.len() as u64,
            c.external_values.iter().map(|v| v.len() as u64).sum(),
        )
    };
    let footprint = DeltaFootprint::of(fixture.delta.as_ref());
    let chunks: Vec<Vec<u8>> = fixture.ompf.chunks(CHUNK).map(<[u8]>::to_vec).collect();
    let recorder = PhaseRecorder::install(&svc.upload_admission);
    let mut plans = PlanCounts::default();
    let (phases, residual, cache_entry, object_id, wall) = match path {
        Path::Create => {
            assert_eq!(
                fixture.base,
                Base::Children(0),
                "fixture self-check: {label}: a CreateLayer has no base layer"
            );
            let session_id = new_session(tv, label).await;
            let frames: Vec<Bytes> =
                std::iter::once(grpc_frame(&make_create_header(&session_id, "layer")))
                    .chain(chunks.into_iter().map(|c| grpc_frame(&make_create_data(c))))
                    .collect();
            begin_fixture_tracking(fixture.name);
            recorder.begin();
            let started = std::time::Instant::now();
            let result = Box::pin(handle_create_layer(
                &svc,
                request::<CreateLayerRequest>(&frames),
            ))
            .await;
            let wall = started.elapsed();
            let (phases, residual) = recorder.finish();
            report_fixture_tracking(fixture.name, label, "at return");
            result.unwrap_or_else(|e| panic!("{label}: CreateLayer must commit: {e:?}"));
            (phases, residual, 0, 0, wall)
        }
        Path::UpdateFast | Path::UpdateSlow => {
            let (session_id, object_id) = base_layer(tv, label, fixture.base).await;
            if route == Route::CachedWarm {
                // Unmeasured, outside the gate: one structural commit through
                // this store loads the object into its structural cache.
                let warm = OwnedDelta {
                    struct_commands: vec![create_section(0, WARM_SECTION, "warm".to_owned())],
                    ..Default::default()
                };
                let relay = Bytes::from(common::owned_delta::delta_to_ompf_bytes(&warm, &[]));
                // Boxed, as every store and handler call here is (see
                // `new_session`).
                Box::pin(svc.layer_store.update_layer(
                    &session_id,
                    "layer",
                    object_id,
                    Some(warm),
                    Some(relay),
                    "t7-warmup",
                    0,
                ))
                .await
                .unwrap_or_else(|e| panic!("{label}: the cache warm-up must commit: {e:?}"));
            }
            if !fixture.pending.is_empty() {
                assert_eq!(
                    route,
                    Route::CachedWarm,
                    "fixture self-check: {label}: pending records are for a cache hit's repair"
                );
                commit_pending(tv, &session_id, object_id, &fixture.pending, label).await;
            }
            if let Some(premise) = fixture.entry_premise {
                assert_eq!(
                    route,
                    Route::CachedWarm,
                    "fixture self-check: {label}: an entry premise is for a cache hit"
                );
                // The entry the measured call will check out, as the store
                // holds it. Checked out untouched and dropped, so it goes back
                // unchanged (`CheckedOut`'s `Drop`), outside the window.
                let held = svc
                    .layer_store
                    .structural_cache()
                    .checkout(object_id)
                    .unwrap_or_else(|| panic!("{label}: the warm-up cached the object"));
                premise(held.entry());
                drop(held);
            }
            let checked = route
                .path_label(fixture.delta.as_ref())
                .filter(|_| !svc.layer_store.takes_fast_path(&fixture.delta));
            let before = checked.map(|l| commits_on(&svc, l));
            let frames: Vec<Bytes> = std::iter::once(grpc_frame(&make_update_header(
                &session_id,
                "layer",
                object_id,
            )))
            .chain(chunks.into_iter().map(|c| grpc_frame(&make_update_data(c))))
            .collect();
            let plans_before = PlanCounts::read(&svc);
            begin_fixture_tracking(fixture.name);
            recorder.begin();
            let started = std::time::Instant::now();
            let result = Box::pin(handle_update_layer(
                &svc,
                request::<UpdateLayerRequest>(&frames),
            ))
            .await;
            let wall = started.elapsed();
            let (phases, residual) = recorder.finish();
            report_fixture_tracking(fixture.name, label, "at return");
            plans = PlanCounts::read(&svc).since(plans_before);
            result.unwrap_or_else(|e| panic!("{label}: UpdateLayer must commit: {e:?}"));
            if let (Some(path_label), Some(before)) = (checked, before) {
                assert_eq!(
                    commits_on(&svc, path_label),
                    before + 1.0,
                    "fixture self-check: {label} committed on the {route:?} route (path \
                     {path_label})"
                );
            }
            // The entry a cold start stored: what invalidating it frees.
            let cache_entry = if route == Route::CachedCold {
                let held = live_now();
                svc.layer_store.structural_cache().invalidate(object_id);
                (held - live_now()).max(0) as u64
            } else {
                0
            };
            (phases, residual, cache_entry, object_id, wall)
        }
    };
    svc.upload_admission.set_charge_observer_for_test(None);
    eprintln!(
        "T7 call {label}: wall {:.3} s; {plans:?}",
        wall.as_secs_f64()
    );
    if fixture_tracked(fixture.name) {
        tokio::time::sleep(SETTLE).await;
        report_fixture_tracking(fixture.name, label, "after SETTLE");
        end_tracking();
    }
    let commit_path = match path {
        Path::Create => CommitPath::Create,
        Path::UpdateFast => CommitPath::UpdateFast,
        Path::UpdateSlow => CommitPath::UpdateSlow,
    };
    if path != Path::Create && (fixture.delta.is_some() || route == Route::Watch) {
        assert_eq!(
            svc.layer_store.takes_fast_path(&fixture.delta),
            path == Path::UpdateFast,
            "fixture self-check: {label} is charged the {path:?} commit path"
        );
    }
    Measured {
        path,
        route,
        name: fixture.name,
        feeds: fixture.feeds,
        body,
        delta,
        ext,
        elements: footprint.elements,
        reorder: footprint.reorder_entries,
        peak: phases.iter().map(|p| p.peak).max().unwrap_or(0),
        // An upload over the budget runs in the overflow slot, charged there
        // instead of in budget.
        charge: svc
            .upload_admission
            .high_water_bytes()
            .max(svc.upload_admission.overflow_high_water_bytes()),
        phases,
        parse_target: parse_charge(body, delta, ext),
        processing_target: processing_charge(commit_path, body, delta, ext, footprint),
        residual,
        cache_entry,
        object_id,
        wall,
        plans,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The fit
// ═════════════════════════════════════════════════════════════════════════════

/// The margin each recommended constant carries over its figure: × 1.25
/// (× 125 for a constant in hundredths).
const HEADROOM: f64 = 1.25;

/// [`HEADROOM`] for a constant in hundredths: × 125.
const HEADROOM_CENTI: f64 = HEADROOM * 100.0;

/// The floor under the coefficients a chain nets its later terms with: per
/// delta byte and per element, raw (not × [`HEADROOM`]).
#[derive(Debug, Clone, Copy)]
struct NetFloor {
    byte: f64,
    element: f64,
}

impl NetFloor {
    /// No floor: each term net of this run's own earlier figures.
    const NONE: Self = Self {
        byte: 0.0,
        element: 0.0,
    };

    /// The production basis of `path`'s byte and element terms: each
    /// production constant ÷ [`HEADROOM`] (and ÷ 100 for the hundredths one).
    fn production(path: Path) -> Self {
        let (byte_centi, element, _) = path.production();
        Self {
            byte: byte_centi as f64 / HEADROOM_CENTI,
            element: element as f64 / HEADROOM,
        }
    }
}

/// One path's measured coefficients, raw (not yet × [`HEADROOM`]).
#[derive(Debug, Clone, Copy)]
struct Fit {
    /// `F`: the larger of `fixed_header` and `fixed_structural`.
    fixed: f64,
    /// The header-only peak, which every other term is net of.
    fixed_header: f64,
    /// The one-child structural edit's peak net of its body and its byte and
    /// element terms (the netting coefficients); 0 without one.
    fixed_structural: f64,
    /// Per delta byte.
    byte: f64,
    /// Per element, net of `net_byte`.
    element: f64,
    /// Per reorder-list entry, net of `net_byte` and `net_element`.
    reorder: f64,
    /// Per external-value byte, net of `net_byte` and `net_element`.
    ext: f64,
    /// The per-byte coefficient the later terms are net of: the larger of
    /// `byte` and the floor's.
    net_byte: f64,
    /// The per-element coefficient the later terms are net of: the larger of
    /// `element` and the floor's.
    net_element: f64,
}

impl Fit {
    /// Fit `results` in the documented order — byte, element, reorder, ext —
    /// netting each later term against the larger of this run's earlier
    /// figure and `floor`'s.
    fn of(results: &[Measured], floor: NetFloor) -> Self {
        let fixed_header = results
            .iter()
            .filter(|m| m.feeds == Feeds::Fixed)
            .map(|m| m.peak as f64)
            .fold(0.0, f64::max);
        let over = |m: &Measured| (m.peak as f64 - fixed_header - m.body as f64).max(0.0);
        let max_of = |feeds: Feeds, per: &dyn Fn(&Measured) -> f64| {
            results
                .iter()
                .filter(|m| m.feeds == feeds)
                .map(per)
                .fold(0.0_f64, f64::max)
        };
        let byte = max_of(Feeds::Bytes, &|m| over(m) / m.delta.max(1) as f64);
        let net_byte = byte.max(floor.byte);
        let element = max_of(Feeds::Elements, &|m| {
            (over(m) - net_byte * m.delta as f64).max(0.0) / m.elements.max(1) as f64
        });
        let net_element = element.max(floor.element);
        let reorder = max_of(Feeds::Reorder, &|m| {
            (over(m) - net_byte * m.delta as f64 - net_element * m.elements as f64).max(0.0)
                / m.reorder.max(1) as f64
        });
        let ext = max_of(Feeds::Ext, &|m| {
            (over(m) - net_byte * m.delta as f64 - net_element * m.elements as f64).max(0.0)
                / m.ext.max(1) as f64
        });
        let fixed_structural = max_of(Feeds::FixedStructural, &|m| {
            (m.peak as f64
                - m.body as f64
                - net_byte * m.delta as f64
                - net_element * m.elements as f64)
                .max(0.0)
        });
        Self {
            fixed: fixed_header.max(fixed_structural),
            fixed_header,
            fixed_structural,
            byte,
            element,
            reorder,
            ext,
            net_byte,
            net_element,
        }
    }

    /// The recommended constants: ⌈figure × [`HEADROOM`]⌉ — `F`, element and
    /// reorder in bytes; byte and ext in hundredths.
    fn recommended(self) -> Recommended {
        Recommended {
            fixed: (self.fixed * HEADROOM).ceil() as u64,
            byte_centi: (self.byte * HEADROOM_CENTI).ceil() as u64,
            element: (self.element * HEADROOM).ceil() as u64,
            reorder: (self.reorder * HEADROOM).ceil() as u64,
            ext_centi: (self.ext * HEADROOM_CENTI).ceil() as u64,
        }
    }

    /// The earlier terms (byte, element) that `route` binds ([`Route::binds`])
    /// for which this run's own recommendation is below `path`'s production
    /// constant, as `(term, recommended, production)`. Called on the run's
    /// own chain: for those terms the netted chain's later terms were netted
    /// against the production basis, not this run's figure, so this run's own
    /// recommended set is not self-consistent — which signals something only
    /// on the route the constant is derived from. On another route of the same
    /// path the constant is set by a costlier route and the run's figure is
    /// below it by design, so no note is printed. The comparison is in
    /// integers, so a figure below its basis by less than one unit of the
    /// constant (one byte, or one hundredth) recommends the constant itself and
    /// prints no note.
    fn floor_binding(self, path: Path, route: Route) -> Vec<(&'static str, u64, u64)> {
        let (byte_centi, element, _) = path.production();
        let own = self.recommended();
        let mut binding = Vec::new();
        if route.binds("byte") && own.byte_centi < byte_centi {
            binding.push(("byte", own.byte_centi, byte_centi));
        }
        if route.binds("element") && own.element < element {
            binding.push(("element", own.element, element));
        }
        binding
    }
}

#[derive(Debug, Clone, Copy)]
struct Recommended {
    fixed: u64,
    byte_centi: u64,
    element: u64,
    reorder: u64,
    ext_centi: u64,
}

impl Recommended {
    /// The commit charge these constants give `m` (without the parse floor).
    fn charge(self, m: &Measured) -> u64 {
        self.fixed
            + m.body
            + (m.ext * self.ext_centi).div_ceil(100)
            + (m.delta * self.byte_centi).div_ceil(100)
            + self.element * m.elements
            + self.reorder * m.reorder
    }
}

/// The unmeasured calls a run makes on `path` and `route` before it measures
/// anything: they load the scripts, fill the pools, create the metric series
/// and run the commit routes' once-per-process code — on
/// `UpdateSlow`, a structural commit and a reset too, since the header-only
/// upload commits on the fast path under `lua`.
async fn warm_up(tv: &TestValkey, path: Path, route: Route, label: &str) {
    measure_route(tv, path, route, &Fixture::header_only(), label).await;
    if path == Path::UpdateSlow {
        let structural = Fixture::new(
            "warmup-structural",
            Feeds::Mixed,
            OwnedDelta {
                struct_commands: children(1, false),
                ..Default::default()
            },
            &[],
        );
        let reset = Fixture::new(
            "warmup-reset",
            Feeds::Mixed,
            OwnedDelta {
                is_diff: true,
                base_version: 0,
                struct_commands: std::iter::once(create_section(0, SECTION, "base".to_owned()))
                    .chain(children(1, false))
                    .collect(),
                ..Default::default()
            },
            &[],
        );
        for fixture in [structural, reset] {
            let label = format!("{label}-{}", fixture.name);
            measure_route(tv, path, route, &fixture, &label).await;
        }
    }
}

/// Every fixture `path` is measured with on `route`: [`fixtures`], and on
/// [`Route::Watch`] also the field-dense and time-sample-dense fixtures, which
/// `txn` sends to the WATCH path charged `UpdateSlow`.
fn fixtures_on(path: Path, route: Route) -> Vec<Fixture> {
    let mut out = fixtures(path);
    if route == Route::Watch {
        let at = out
            .iter()
            .position(|f| f.feeds == Feeds::Elements)
            .unwrap_or(out.len());
        let fast_only = fixtures(Path::UpdateFast)
            .into_iter()
            .filter(|f| f.feeds == Feeds::Elements);
        out.splice(at..at, fast_only);
    }
    out
}

/// Measure every fixture of `path` on `route`, print the report, and assert
/// the envelope and the cross-check.
async fn run_path(path: Path, route: Route) {
    run_path_where(path, route, |_| true).await;
}

/// The WATCH read's window transports (b.gvf) a measurement on `route`
/// runs on. On [`Route::Watch`] both: the production one, where each window
/// of the read's rounds 2 and 3 is one window script (but round 1, a lone
/// children list over τ and every window after a script fallback, which are
/// plain), and the plain one, every window plain commands — what a read that
/// falls back issues, each command holding its own future, priced by
/// `W_PLAIN_CMD`. Every other route reads no object through the WATCH read,
/// so it runs on the production transport alone.
fn transports_on(route: Route) -> &'static [WindowTransport] {
    if route == Route::Watch {
        &[WindowTransport::Script, WindowTransport::Plain]
    } else {
        &[WindowTransport::Script]
    }
}

/// The suffix a run on `transport` adds to its labels and session names,
/// which must differ between two runs on one fixture's store: none on the
/// production transport, `-plain` on the plain one.
fn transport_suffix(transport: WindowTransport) -> &'static str {
    match transport {
        WindowTransport::Script => "",
        WindowTransport::Plain => "-plain",
    }
}

/// [`run_path`] over the fixtures of `path` on `route` that `keep` keeps,
/// once per transport of [`transports_on`]: each transport's run is warmed
/// up, measured, fitted and reported on its own, and every run's envelope
/// and cross-check are asserted after the last one is reported. The plain
/// transport is forced through the test-support switch
/// (`ValkeyClient::set_watch_read_transport_for_test`) on the fixture's
/// client, which every measured service shares, and reset after.
async fn run_path_where(path: Path, route: Route, keep: fn(&Fixture) -> bool) {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut runs = Vec::new();
    for &transport in transports_on(route) {
        let suffix = transport_suffix(transport);
        tv.client().set_watch_read_transport_for_test(transport);
        warm_up(
            &tv,
            path,
            route,
            &format!("warmup-{path:?}-{route:?}{suffix}"),
        )
        .await;

        let all = fixtures_on(path, route);
        let total = all.len();
        let mut results = Vec::new();
        for fixture in all.into_iter().filter(keep) {
            let label = format!("{path:?}-{route:?}{suffix}-{}", fixture.name);
            results.push(measure_route(&tv, path, route, &fixture, &label).await);
        }
        tv.client()
            .set_watch_read_transport_for_test(WindowTransport::Script);
        let own = Fit::of(&results, NetFloor::NONE);
        let fit = Fit::of(&results, NetFloor::production(path));
        // The cross-check predicts with this run's own chain, which depends
        // on nothing but this run's measurements.
        let predicting = own.recommended();
        report(
            path,
            route,
            transport,
            &results,
            results.len() < total,
            own,
            fit,
            predicting,
        );
        runs.push((transport, results, predicting));
    }

    for (transport, results, predicting) in &runs {
        for m in results {
            let overruns = m.phase_overruns();
            assert!(
                overruns.is_empty(),
                "{:?} on {:?} ({transport:?} windows) {}: the upload's heap grew past the charge \
                 in force at the production constants in {overruns:?} — the reservation \
                 under-covers that phase. Body {} B, delta part {} B ({} elements, {} reorder \
                 entries), external-value part {} B; phases: {}. Re-derive the constants by the \
                 rule in the comment above them in grpc::upload_admission — one run's \
                 recommendation can be too low.",
                m.path,
                m.route,
                m.name,
                m.body,
                m.delta,
                m.elements,
                m.reorder,
                m.ext,
                m.phase_summary()
            );
        }
        for m in results.iter().filter(|m| m.feeds == Feeds::Mixed) {
            let predicted = predicting.charge(m);
            assert!(
                m.peak <= predicted,
                "{:?} ({transport:?} windows) {}: the cross-check grew {} bytes, past the \
                 {predicted} bytes the fitted coefficients ({predicting:?}) predict — the \
                 per-path model does not hold for content it was not fitted on",
                m.path,
                m.name,
                m.peak
            );
        }
    }
}

/// `partial`: the run left out some of the path's fixtures
/// ([`run_path_where`]). Every floor-binding note of a partial run is then
/// replaced by an "as expected" one. The one partial run today (the
/// long-reorder report) is on WATCH, which binds no term ([`Route::binds`]),
/// so it prints no note; a partial run on a binding route that kept a term's
/// binding fixture would need the plain note back.
#[allow(clippy::too_many_arguments)]
fn report(
    path: Path,
    route: Route,
    transport: WindowTransport,
    results: &[Measured],
    partial: bool,
    own: Fit,
    fit: Fit,
    predicting: Recommended,
) {
    let (byte, element, reorder) = path.production();
    let recommended = fit.recommended();
    eprintln!("== upload memory envelope: {path:?} on route {route:?} ({transport:?} windows) ==");
    eprintln!(
        "production constants: F={FIXED_CHARGE_BYTES} B e_ext={EXT_EXPANSION_CENTI} \
         k_parse={PARSE_EXPANSION_CENTI} byte={byte} (hundredths) element={element} \
         reorder={reorder} (bytes)"
    );
    eprintln!(
        "{:<12} {:<18} {:>9} {:>9} {:>9} {:>7} {:>7} {:>11} {:>11} {:>6}",
        "path", "fixture", "body", "delta", "ext", "elems", "reord", "peak", "high-water", "pk/hw"
    );
    for m in results {
        eprintln!(
            "{:<12} {:<18} {:>9} {:>9} {:>9} {:>7} {:>7} {:>11} {:>11} {:>6.3}",
            format!("{:?}", m.path),
            m.name,
            m.body,
            m.delta,
            m.ext,
            m.elements,
            m.reorder,
            m.peak,
            m.charge,
            m.peak as f64 / m.charge.max(1) as f64,
        );
    }
    eprintln!(
        "phases (each phase's peak / the charge in force, consecutive phases of one step \
         merged at their highest ratio; then the bytes the call left live when it returned, \
         and of them the structural-cache entry a cold start stored):"
    );
    for m in results {
        eprintln!(
            "  {:<18} {} || left {} (cache entry {})",
            m.name,
            m.phase_summary(),
            m.residual,
            m.cache_entry
        );
    }
    eprintln!(
        "measured {path:?} (this run's own chain, each term net of this run's earlier \
         figures): F={:.0} B  byte={:.3}/B  element={:.1} B  reorder={:.1} B  e_ext={:.3}/B",
        own.fixed, own.byte, own.element, own.reorder, own.ext
    );
    eprintln!(
        "F {path:?} on {route:?}: header-only peak {:.0} B; one-child structural edit net of \
         its body, byte and element terms {:.0} B (netted chain; 0 when not measured); F is the \
         larger",
        fit.fixed_header, fit.fixed_structural
    );
    let floor_note: String = own
        .floor_binding(path, route)
        .into_iter()
        .map(|(term, recommended, production)| {
            if partial {
                format!(
                    "  [{term} {recommended} below its production constant {production}, as \
                     expected: this run leaves out the fixture it is derived from; later terms \
                     were netted against the production basis]"
                )
            } else {
                format!(
                    "  [floor binding: this run recommends {term} {recommended} below its \
                     production constant {production}; later terms were netted against the \
                     production basis]"
                )
            }
        })
        .collect();
    eprintln!(
        "netted {path:?} (each later term net of max(this run, production / {HEADROOM}): byte \
         {:.3}/B, element {:.1} B): F={:.0} B  byte={:.3}/B  element={:.1} B  reorder={:.1} B  \
         e_ext={:.3}/B{floor_note}",
        fit.net_byte, fit.net_element, fit.fixed, fit.byte, fit.element, fit.reorder, fit.ext
    );
    eprintln!(
        "recommended {path:?} (netted x{HEADROOM}, ceil; this run only — production follows the \
         derivation rule in grpc::upload_admission): F={} B  byte={} centi  element={} B  \
         reorder={} B  e_ext={} centi",
        recommended.fixed,
        recommended.byte_centi,
        recommended.element,
        recommended.reorder,
        recommended.ext_centi
    );
    for m in results.iter().filter(|m| m.feeds == Feeds::Mixed) {
        eprintln!(
            "cross-check {path:?} {}: peak {} B, predicted by this run's own fit {} B, \
             production charge {} B",
            m.name,
            m.peak,
            predicting.charge(m),
            m.charge
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn create_layer_uploads_stay_within_their_charge() {
    run_path(Path::Create, Route::Default).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn update_layer_fast_path_uploads_stay_within_their_charge() {
    run_path(Path::UpdateFast, Route::Default).await;
}

/// `UpdateSlow` on the cached structural route's cold start (`lua`, a fresh
/// structural cache).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn update_layer_slow_path_cached_cold_uploads_stay_within_their_charge() {
    run_path(Path::UpdateSlow, Route::CachedCold).await;
}

/// `UpdateSlow` on the cached structural route with the object cached (`lua`).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn update_layer_slow_path_cached_warm_uploads_stay_within_their_charge() {
    run_path(Path::UpdateSlow, Route::CachedWarm).await;
}

/// `UpdateSlow` on the WATCH/MULTI/EXEC path (`txn`), every fixture:
/// [`run_path`]'s whole set, the fit and the cross-check included, once on
/// each of the WATCH read's window transports ([`transports_on`]).
///
/// The WATCH commit queues its commands on the txn connection — three per
/// value row, two per section — and a fast-shaped delta reads its
/// external-value pre-image first on a data connection; fred holds every
/// command not yet answered in queues that double as they fill, so how far
/// they grow depends on reply timing. Before b.26e Stage 2 no term charged
/// that queue, and the field-dense, time-sample-dense and mixed fixtures ran
/// only in an `#[ignore]`d copy of this test (field-dense measured 0.80 of its
/// charge in one run and 1.30 in a slower one). Stage 2 charges the queue's
/// deterministic worst case, next-power-of-two slots of every connection's
/// queue at once (`QUEUED_COMMAND_CHARGE_BYTES`), so
/// every fixture is asserted against its charge alone, however slow the
/// replies; [`stalled_watch_commit_queue_is_reported_b26e`] forces that worst
/// case.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn update_layer_slow_path_watch_all_uploads_stay_within_their_charge() {
    run_path(Path::UpdateSlow, Route::Watch).await;
}

/// The large-base reproducers (b.26e): slow-path `UpdateLayer`s of a small
/// delta on `base` — thousands of children of `SECTION`, the harness-shaped
/// layer, the long-name layer, or an empty section.
///
/// Each measures one call on each `UpdateSlow` route — [`Route::CachedCold`],
/// [`Route::CachedWarm`] and [`Route::Watch`] — and asserts each phase's heap
/// growth is at most the charge in force during it at the production
/// constants ([`Measured::phase_overruns`]): in the processing phase, where
/// the commit route loads the object, that is the processing-step charge
/// ([`processing_charge`]), not the call's high-water, which includes the
/// receive allowance `A` (about 8 MiB) released when the body ends and would
/// hide a base smaller than that; above it, the object-state term each
/// in-commit charge point added. On [`Route::CachedCold`] the read of the
/// uncached object is under its own permit or `T` grant, not the upload's
/// charge, and is reported, not checked ([`Measured::labels`]). The fixture
/// is tagged [`Feeds::Mixed`] only
/// because a [`Fixture`] needs a tag; it is measured here, never in
/// [`run_path`], so it never reaches a fit.
async fn assert_slow_path_update_on_large_base_within_charge(
    name: &'static str,
    base: Base,
    delta: OwnedDelta,
) {
    assert_slow_path_update_within_charge_on(
        name,
        base,
        delta,
        &[],
        &[Route::CachedCold, Route::CachedWarm, Route::Watch],
    )
    .await;
}

/// [`assert_slow_path_update_on_large_base_within_charge`] on `routes` only,
/// with `external` the external values the update's OMPF carries.
///
/// [`Route::Watch`] is measured on both of the WATCH read's window
/// transports ([`transports_on`]). The plain transport is forced through the
/// test-support switch (`ValkeyClient::set_watch_read_transport_for_test`)
/// on the fixture's client, which every measured service shares, and reset
/// after.
async fn assert_slow_path_update_within_charge_on(
    name: &'static str,
    base: Base,
    delta: OwnedDelta,
    external: &[&[u8]],
    routes: &[Route],
) {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut over = Vec::new();
    for &route in routes {
        for &transport in transports_on(route) {
            // Session names are unique per base path: the plain run's
            // sessions carry their own name (leaked: a few bytes per test).
            let run_name: &'static str = match transport {
                WindowTransport::Script => name,
                WindowTransport::Plain => {
                    Box::leak(format!("{name}{}", transport_suffix(transport)).into_boxed_str())
                }
            };
            tv.client().set_watch_read_transport_for_test(transport);
            let m = measure_slow_path_update(&tv, run_name, base, &delta, external, route).await;
            tv.client()
                .set_watch_read_transport_for_test(WindowTransport::Script);
            over.extend(
                m.phase_overruns()
                    .into_iter()
                    .map(|overrun| format!("{route:?} ({transport:?} windows): {overrun}")),
            );
        }
    }
    assert!(
        over.is_empty(),
        "b.26e {name}: the upload's heap grew past the charge in force on {over:?} — the \
         object-state terms the commit route charges do not cover the object it holds"
    );
}

/// One `UpdateSlow` call of `delta` (with `external` values) on a fresh
/// `base` layer, on `route`, after the route's warm-up, measured phase by
/// phase and printed: its peak against the processing-step charge and the
/// call's high-water, the largest charge held while processing (the
/// processing step plus the largest object-state term the commit took), and
/// each phase's peak against the charge in force.
async fn measure_slow_path_update(
    tv: &TestValkey,
    name: &'static str,
    base: Base,
    delta: &OwnedDelta,
    external: &[&[u8]],
    route: Route,
) -> Measured {
    warm_up(
        tv,
        Path::UpdateSlow,
        route,
        &format!("{name}-{route:?}-warmup"),
    )
    .await;
    let fixture = Fixture::new(name, Feeds::Mixed, delta.clone(), external).on_base(base);
    let label = format!("{name}-{route:?}");
    let m = measure_route(tv, Path::UpdateSlow, route, &fixture, &label).await;
    eprintln!(
        "== b.26e {name} on {route:?}: peak {} B; processing-step charge {} B ({:.0}x); call \
         high-water {} B ({:.1}x, includes A); held max in processing {} B; base {base:?} ==\n   \
         phases: {}",
        m.peak,
        m.processing_target,
        m.peak as f64 / m.processing_target.max(1) as f64,
        m.charge,
        m.peak as f64 / m.charge.max(1) as f64,
        m.phase_held_max("processing"),
        m.phase_summary()
    );
    m
}

/// One new child of `SECTION`, on a base of `base_children` of them.
fn one_new_child(base_children: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![create_section(
            SECTION,
            10 + base_children as u64,
            "extra".to_owned(),
        )],
        ..Default::default()
    }
}

/// A reorder of `SECTION`'s children by `children_list`.
fn reorder_of(children_list: Vec<u64>) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: SECTION,
            children_list_id: 0,
            children_list,
        })],
        ..Default::default()
    }
}

/// b.26e reproducer: one new child on a base of 5,000 children. The delta is
/// one element, but the slow path loads the object's whole state.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_one_new_child_on_5k_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-one-new-child-5k",
        Base::Children(DENSE / 4),
        one_new_child(DENSE / 4),
    )
    .await;
}

/// b.26e reproducer: one new child on a base of 20,000 children.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_one_new_child_on_20k_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-one-new-child-20k",
        Base::Children(DENSE),
        one_new_child(DENSE),
    )
    .await;
}

/// b.26e reproducer: a true swap of the first two of 20,000 children —
/// `[11, 10, 12]` on `[10, 11, 12, …]` gives `[11, 10, 12, …]` (each named
/// child is followed by the unnamed children after it), so only those two
/// change position and key.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_swap_two_children_on_20k_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-swap-two-20k",
        Base::Children(DENSE),
        reorder_of(vec![11, 10, 12]),
    )
    .await;
}

/// b.26e reproducer: a two-entry reorder that moves the first of 20,000
/// children to the end — `[11, 10]` on `[10, 11, 12, …]` gives `[11, 12, …,
/// 10]` (child 11 carries the unnamed children after it) — so every child
/// changes position, and each position keeps its key, so every child's key
/// changes and all 20,000 rows are rewritten.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_move_first_child_to_end_on_20k_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-move-first-to-end-20k",
        Base::Children(DENSE),
        reorder_of(vec![11, 10]),
    )
    .await;
}

/// b.26e reproducer: one new prim under the zone of the harness-shaped
/// 5,003-section layer, fields included ([`Base::Harness`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_one_new_prim_on_harness_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-one-new-prim-harness",
        Base::Harness,
        one_new_prim_in_harness_zone(),
    )
    .await;
}

/// b.26e reproducer: a reset (`is_diff`, `base_version` 0) of a base of
/// 20,000 children to one empty section — a one-element delta that deletes
/// every row the base holds.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_reset_of_20k_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-reset-20k",
        Base::Children(DENSE),
        OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![create_section(0, SECTION, "base".to_owned())],
            ..Default::default()
        },
    )
    .await;
}

/// b.26e: one new child on an empty section — the small-object end of the
/// reproducers. Before Stage 2 the WATCH route's processing phase peaked at
/// 105,159 B against a 102,044 B processing-step charge (1.03×): the
/// structural WATCH attempt's fixed cost was outside `F`. Stage 2 charges it
/// as `W_FIXED` from the head of every structural WATCH attempt (WA0); this
/// is that charge's regression check.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_one_new_child_on_empty_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-one-new-child-0",
        Base::Children(0),
        one_new_child(0),
    )
    .await;
}

/// `n` new children of `SECTION`, on a base of `base_children` of them: from
/// the second, each birth probes a `(parent, name)` the plan has probed
/// before, so the plan builds the name index.
fn new_children(base_children: usize, n: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: (0..n)
            .map(|i| {
                create_section(
                    SECTION,
                    10 + (base_children + i) as u64,
                    format!("extra{i}"),
                )
            })
            .collect(),
        ..Default::default()
    }
}

/// b.26e: one new child on a base of 7,168 sections — the
/// largest table at 8,192 buckets (7/8 full), so the plan in place on the
/// cached entry grows (rehashes) that table on the new child's insert; CR1's
/// in-place growth term charges the doubled table before the plan runs
/// (`estimate_in_place_growth_bytes`): each phase within the charge in
/// force, on all three routes.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_one_new_child_on_full_table_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-one-new-child-7168",
        Base::Children(7_167),
        one_new_child(7_167),
    )
    .await;
}

/// b.26e: two new children on a base of 5,000 children —
/// the second birth's name probe builds the name index over the parent's
/// 5,000 children, which the plan asks its budget for before it
/// builds it: each phase within the charge in force, on all three routes.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_two_new_children_on_5k_base_stay_within_their_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-two-new-children-5k",
        Base::Children(DENSE / 4),
        new_children(DENSE / 4, 2),
    )
    .await;
}

/// b.26e: [`slow_path_two_new_children_on_5k_base_stay_within_their_charge_b26e`]
/// on a base of 20,000 children.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_two_new_children_on_20k_base_stay_within_their_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-two-new-children-20k",
        Base::Children(DENSE),
        new_children(DENSE, 2),
    )
    .await;
}

/// b.26e: a structural WATCH edit of a layer whose
/// value rows are large inline values ([`Base::InlineValues`]: one fan-out
/// window of 32 KiB values beside a 5,000-entry children list) stays within
/// its charge in every phase. The WATCH read (`structural_update`) takes each
/// value row's length and columns, never its value, and charges every window
/// before it is issued at its replies' bound; a read that transferred the
/// values would receive a window of 2,141 × 32 KiB with only the commands'
/// charge held.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_one_new_child_on_inline_value_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_within_charge_on(
        "b26e-one-new-child-inline-values",
        Base::InlineValues,
        one_new_child(INLINE_VALUE_CHILDREN),
        &[],
        &[Route::Watch],
    )
    .await;
}

/// b.26e: one new child on the long-name base
/// ([`Base::Named`] at 1,024-byte names: 2,141 sections and as many fields,
/// each named 1,024 bytes) stays within its charge in every phase, on all
/// three routes. The WATCH read's charge counts the names it retains by their
/// bytes (`W_NAME_BYTE`), and sizes each window's reply bound at the longest
/// name seen; counts of entries and items alone do not cover them.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_one_new_child_on_long_name_base_stays_within_its_charge_b26e() {
    assert_slow_path_update_on_large_base_within_charge(
        "b26e-one-new-child-long-names",
        Base::Named(1_024),
        one_new_child(NAMED_ENTRIES),
    )
    .await;
}

/// A refused plan on a budget too small for it: deleting a section with
/// 1,000 fields beside a section of 64,000 fields, through the handler on
/// [`Route::CachedWarm`], with
/// the gate's budget full but for the room the call needs up to its plan
/// (sized by an unconstrained calibration run of the same edit on an
/// identical base: the most the call holds before its plan's first growth,
/// which is what its hit's first repair read, RP0, holds — the growth before
/// it) and the overflow slot held, so the plan's first in-plan growth is
/// refused. The plan then counts on to its exact need, puts its entry back
/// and the call waits for that need; a thread releases the slot once the
/// call waits, and the call commits in the slot, with one recharge. Each
/// phase outside the refused plan stays within the charge in force. Inside
/// the refused plan's phases — from the processing step, whose charge grows
/// once before the refusal, at RP0, to the wait's grant — the plan's scratch
/// runs above what is held, by design (it counts on so its need is exact),
/// and only the documented residual is asserted: peak − held ≤ the next
/// target − held, the next target being the charge the call waited for.
///
/// That residual bound is the whole rewrite's price, so on its own it cannot
/// tell a growth refused inside the plan from one refused at the exact trim
/// after it, with the plan uncharged; the calibration run, where every
/// growth fits, is asserted within its charge in every phase too, and an
/// uncharged plan fails there.
///
/// The gate's charge observer reports the whole gate, so each phase is read
/// net of the filler (in budget) and the slot holder (on the slot, until it
/// is released).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_refused_plan_on_a_full_budget_stays_within_its_charge_b26e() {
    use live_edit::grpc::upload_admission::UploadRpc;
    use tokio_util::sync::CancellationToken;

    /// How long the releaser waits for the call to queue before it gives up
    /// (the test then fails on the call not having waited).
    const RELEASE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(60);
    /// Fields kept on `SECTION`: enough that the processing step's
    /// object-state hint (the cached entry it plans on in place, its state and
    /// plan indexes, about 170 B per field) holds more than the receive phase
    /// (`F` + `A` + the body's growth, about 8.5 MB), so the plan's first
    /// growth passes everything the call held before it — at least about
    /// 50,000; 64,000 leaves about 2.5 MB of margin (80,000 made a base's
    /// one-`EXEC` creation time out on the shared store). The self-check
    /// below confirms it.
    const KEPT_FIELDS: usize = 64_000;
    /// Fields on `SECOND_SECTION`, which the edit deletes: enough that its
    /// cascade's projected rows (`rewrite_row`, 4,574 B each on the cached
    /// route, about 4.6 MB for 1,000) pass the reply term the hit's first
    /// repair read (RP0) leaves held beyond the hint (the first read's reply
    /// term at `STRUCTURAL_REPAIR_INITIAL_ENTRIES` and
    /// `STRUCTURAL_REPAIR_INITIAL_BYTES`, within 1 MiB), which the
    /// plan spends as rewrite allowance before it asks its budget: with one
    /// field the plan never grew the charge. The self-check below confirms
    /// it.
    const DOOMED_FIELDS: usize = 1_000;
    let name = "b26e-refused-plan-delete-1k-beside-64k-fields";
    let base = Base::SplitFields {
        keep: KEPT_FIELDS,
        doomed: DOOMED_FIELDS,
    };
    // Delete `SECOND_SECTION` and its fields: no insert, so CR1 adds nothing
    // to the hint (no table can grow) and is a no-op after the hit's first
    // repair read (RP0), which already holds the hint and that read's reply
    // term; the plan's first growth is then its cascade's projection, inside
    // the plan, past what RP0 left held. The heavy part of the base
    // is a table, not a children list, so the commit stays cheap: a large
    // list costs the commit per entry (60,000 children took 15 s a call in a
    // debug build, re-keying them 30 s).
    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: SECOND_SECTION,
        })],
        ..Default::default()
    };
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;

    // Calibration: the same edit on an identical base, on a fresh gate,
    // where every growth fits; within its charge in every phase.
    let calibration =
        measure_slow_path_update(&tv, name, base, &delta, &[], Route::CachedWarm).await;
    assert!(
        calibration.phase_overruns().is_empty(),
        "b.26e {name}, unrefused: the upload's heap grew past the charge in force: {:?}",
        calibration.phase_overruns()
    );
    let labels = calibration.labels();
    let processing = labels
        .iter()
        .position(|label| *label == "processing")
        .expect("fixture self-check: the calibration call has a processing step");
    let step = calibration.phases[processing].held();
    // The processing step holds the hint. The commit's first charge point
    // after it is the hit's first repair read (RP0), which grows past the
    // hint by that read's reply term alone — the object is cached and nothing
    // is pending, so the entry weighs what the hint priced; the plan's first
    // growth is the first growth after it. The reply term is the two-term
    // reply's, at the first read's bounds as the repair asks for them
    // (`RepairBounds::first_read`): its entry bound and its byte bound alike.
    let price = |repair| object_state_charge_for_test(&ObjectStateNeed::cached_repair(0, repair));
    let first_read_bounds =
        live_edit::storage::structural_cache_repair::RepairBounds::PRODUCTION.first_read();
    assert_eq!(
        first_read_bounds.reply_bytes, STRUCTURAL_REPAIR_INITIAL_BYTES,
        "fixture self-check: the first read's byte bound"
    );
    let first_read_term = price(first_read_bounds) - price(RepairTransient::default());
    let first_read = calibration.phases[processing..]
        .iter()
        .position(|p| p.held() > step)
        .map(|i| processing + i)
        .expect("fixture self-check: the calibration call's first repair read grows its charge");
    let read_held = calibration.phases[first_read].held();
    assert_eq!(
        read_held,
        step + first_read_term,
        "fixture self-check: the commit's first growth is the hit's first repair read (RP0), its \
         reply term past the hint"
    );
    let growth = calibration.phases[first_read..]
        .iter()
        .position(|p| p.held() > read_held)
        .map(|i| first_read + i)
        .expect("fixture self-check: the calibration call's plan grows its charge");
    let room = calibration.phases[..growth]
        .iter()
        .map(Phase::held)
        .max()
        .unwrap_or(0);
    assert!(
        calibration.phases[growth].held() > room,
        "fixture self-check: the plan's first growth ({} B) passes everything held before it \
         ({room} B; the processing step {step} B)",
        calibration.phases[growth].held()
    );

    // The gate: its budget full but for `room`, and the slot held.
    let gate = Arc::new(UploadAdmissionGate::new(
        UploadAdmissionConfig::default(),
        None,
    ));
    let budget = gate.budget_bytes();
    let holders = CancellationToken::new();
    let mut slot = gate
        .admit(UploadRpc::UpdateLayer, "refused-plan-slot", &holders)
        .await
        .expect("the slot holder is admitted");
    slot.end_receive();
    assert!(
        slot.reserve_processing(budget + 1, &holders)
            .await
            .expect("the slot is free")
            .entered_overflow,
        "fixture: the slot is held"
    );
    let mut filler = gate
        .admit(UploadRpc::UpdateLayer, "refused-plan-filler", &holders)
        .await
        .expect("the filler is admitted");
    let others = gate.reserved_bytes() - filler.reserved();
    filler
        .try_grow_in_budget(budget - others - room)
        .expect("the filler leaves the call its room");
    let (in_budget, on_slot) = (gate.reserved_bytes(), gate.overflow_bytes());

    // The releaser: frees the slot once the call waits for its recharge.
    // A thread polling the gate, which allocates nothing while the call is
    // measured.
    let done = Arc::new(AtomicBool::new(false));
    let releaser = {
        let (gate, done) = (Arc::clone(&gate), Arc::clone(&done));
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + RELEASE_DEADLINE;
            while gate.processing_waiters() == 0
                && !done.load(Ordering::SeqCst)
                && std::time::Instant::now() < deadline
            {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            drop(slot);
        })
    };
    let svc = LiveEditService {
        upload_admission: Arc::clone(&gate),
        ..service_on(&tv, Route::CachedWarm, tv.client().clone())
    };
    let fixture = Fixture::new(name, Feeds::Mixed, delta.clone(), &[]).on_base(base);
    let label = format!("{name}-refused");
    let m = measure_on(
        svc,
        &tv,
        Path::UpdateSlow,
        Route::CachedWarm,
        &fixture,
        &label,
    )
    .await;
    done.store(true, Ordering::SeqCst);
    releaser.join().expect("the releaser thread");
    drop(filler);

    // Each phase net of the filler and of the slot holder.
    let phases: Vec<Phase> = m
        .phases
        .iter()
        .map(|p| Phase {
            charge: GateCharge {
                reserved_bytes: p.charge.reserved_bytes.saturating_sub(in_budget),
                overflow_bytes: if p.charge.overflow_bytes == on_slot {
                    0
                } else {
                    p.charge.overflow_bytes
                },
            },
            peak: p.peak,
            t_max: p.t_max,
        })
        .collect();
    // The refused plan's phases: the processing phases before the call runs
    // in the slot, after its wait. With no wait there are none, and every
    // phase is checked against its charge.
    let labels = phase_labels(&phases, m.parse_target, m.processing_target);
    let slot_run = phases.iter().position(|p| p.charge.overflow_bytes > 0);
    let next_target = slot_run.map_or(0, |s| phases[s].held());
    let refused: Vec<usize> = (0..slot_run.unwrap_or(0))
        .filter(|&i| labels[i] == "processing")
        .collect();
    let limits = phase_limits(&phases, 0);
    let mut over = Vec::new();
    for (i, (p, limit)) in phases.iter().zip(&limits).enumerate() {
        let (bound, what) = if refused.contains(&i) {
            (
                next_target,
                "the refused plan's residual bound (the next target)",
            )
        } else {
            (*limit, "the charge in force")
        };
        if p.peak > bound {
            over.push(format!(
                "phase {i} ({}): {} B over {what}, {bound} B (held {} B)",
                labels[i],
                p.peak,
                p.held()
            ));
        }
    }
    eprintln!(
        "== b.26e {name}, refused: room {room} B; next target {next_target} B; refused phases \
         {refused:?}; {:?} ==",
        m.plans
    );
    assert!(
        over.is_empty(),
        "b.26e {name}: a phase outside the refused plan passed its charge, or the refused plan \
         passed its documented residual: {over:?}"
    );
    assert_eq!(
        (
            m.plans.recharges,
            m.plans.in_place_plans,
            slot_run.is_some()
        ),
        (1.0, 2, true),
        "fixture self-check: a growth inside the plan was refused, the call waited once, \
         planned again and ran in the slot: {:?}",
        m.plans
    );
    assert!(
        refused
            .iter()
            .all(|&i| phases[i].held() == step || phases[i].held() == room)
            && refused.last().is_some_and(|&i| phases[i].held() == room),
        "fixture self-check: from its processing step ({step} B) to its wait the call held that \
         step and, from its first repair read on, its room ({room} B), so its plan's first \
         growth was the one refused: {:?}",
        refused
            .iter()
            .map(|&i| phases[i].held())
            .collect::<Vec<_>>()
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// b.fsz (a): a cache hit's stream repair, charged per round (RP0, RP1) and
// per record (RP2)
// ═════════════════════════════════════════════════════════════════════════════
//
// A structural call whose object is cached repairs the held entry from the
// object's stream before it plans: it reads the records committed since the
// entry's cut and replays them into the entry in place. Each round holds its
// read at the read's entry and byte bounds before issuing it (RP0), before
// replaying the chunk it read the chunk's reply, its largest record's parse
// and the entry's growth by its delta parts (RP1), and before each record
// applies what that record grows the entry by beyond its delta bytes (RP2),
// on top of the processing step and the handler's hint. The tests below commit records through a
// second store after the measured store cached the object, so the measured
// call's hit replays them, and hold every phase of that call to the charge
// in force. Before b.fsz the repair ran uncharged, under the processing step
// and the hint alone.

/// The first section id a pending record creates: clear of the base's
/// children (ids from 10), the measured edit's and the cache warm-up's.
const PENDING_FIRST_ID: u64 = 1_000_000;

/// `n` new prims under `SECTION`, ids from `first`: each a section and the
/// two fields a prim's spec carries (`specifier`, `typeName`), as the load
/// harness's client writes them.
fn new_prims(first: u64, n: u64) -> OwnedDelta {
    let ids = first..first + n;
    let set = |section_id: u64, key: &str, value: &[u8]| SetField {
        section_id,
        key_name: key.to_owned(),
        value_or_ext_hash: value.to_vec(),
        ext_value_size: 0,
        ext_value_index: 0,
        set_order: 0,
    };
    OwnedDelta {
        struct_commands: ids
            .clone()
            .map(|id| create_section(SECTION, id, format!("p{id}")))
            .collect(),
        set_fields: ids
            .flat_map(|id| [set(id, "specifier", b"def"), set(id, "typeName", b"Xform")])
            .collect(),
        ..Default::default()
    }
}

/// `n` new child sections of `SECTION`, ids from `first`, named `c{id}`:
/// structure alone, the most entry growth per delta byte.
fn new_sections(first: u64, n: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: (first..first + n)
            .map(|id| create_section(SECTION, id, format!("c{id}")))
            .collect(),
        ..Default::default()
    }
}

/// `build(n)` at the `n` whose delta part is about `bytes`, from a
/// 1,000-element sample: such a delta grows linearly in `n`.
fn sized_to(bytes: u64, build: impl Fn(u64) -> OwnedDelta) -> OwnedDelta {
    let per = delta_part_len(&build(1_000)) as f64 / 1_000.0;
    build((bytes as f64 / per).round() as u64)
}

/// A pending record of `delta` alone.
fn pending_of(delta: OwnedDelta) -> PendingRecord {
    PendingRecord {
        delta,
        external: Vec::new(),
    }
}

/// `n` small pending records, each one new prim (240 bytes of delta).
/// At 256 they fill one repair read's entry bound.
fn pending_small_records(n: u64) -> Vec<PendingRecord> {
    (0..n)
        .map(|i| pending_of(new_prims(PENDING_FIRST_ID + i, 1)))
        .collect()
}

/// `n` pending field writes, each one 8-byte value on `SECTION` under a key
/// of its own: the smallest records an editing client sends, 136 bytes of
/// delta, so a read of 256 holds the most per-entry overhead per byte.
fn pending_field_writes(n: u64) -> Vec<PendingRecord> {
    (0..n)
        .map(|i| {
            pending_of(OwnedDelta {
                set_fields: vec![field(format!("w{i}"), i.to_le_bytes().to_vec())],
                ..Default::default()
            })
        })
        .collect()
}

/// One pending record: a new prim carrying a 2 MiB external value (a
/// screenshot attached to a prim), whose OMPF is mostly the external part.
fn pending_external_record() -> Vec<PendingRecord> {
    let blob = blobs(1, 2 * MIB, 0x7e).remove(0);
    let mut delta = new_prims(PENDING_FIRST_ID, 1);
    delta.set_fields.push(SetField {
        section_id: PENDING_FIRST_ID,
        key_name: "screenshot".to_owned(),
        value_or_ext_hash: live_edit::storage::external_values::ext_content_hash(&blob).to_vec(),
        ext_value_size: blob.len() as u64,
        ext_value_index: 1,
        set_order: 0,
    });
    vec![PendingRecord {
        delta,
        external: vec![blob],
    }]
}

/// One pending record of about 1 MiB of new sections: above the repair's
/// first read bound (256 KiB), so its first read answers `TOO_LARGE` and the
/// repair reads it again at its size, within the 4 MiB cap: two reads, no
/// reload.
fn pending_sections_record() -> Vec<PendingRecord> {
    let delta = sized_to(MIB as u64, |n| new_sections(PENDING_FIRST_ID, n));
    vec![pending_of(delta)]
}

/// One pending structural record of about 3 MiB of new prims, read the same
/// way: the regression's record.
fn pending_prims_record() -> Vec<PendingRecord> {
    let delta = sized_to(3 * MIB as u64, |n| new_prims(PENDING_FIRST_ID, n));
    vec![pending_of(delta)]
}

/// The stream-repair rows: the regression's record, and the shapes around
/// it — the two-term reply fit's single-entry rows (the 2 MiB external, 1 MiB
/// sections, 3 MiB prims and 250 KiB records) and its multi-entry rows (256
/// field writes, 256 prims, 256 records of 1 KiB).
fn stream_repair_rows() -> Vec<(&'static str, Vec<PendingRecord>)> {
    vec![
        ("fsz-repair-256-small-records", pending_small_records(256)),
        ("fsz-repair-256-field-writes", pending_field_writes(256)),
        ("fsz-repair-2mib-external-record", pending_external_record()),
        ("fsz-repair-1mib-sections-record", pending_sections_record()),
        ("fsz-repair-3mib-prims-record", pending_prims_record()),
        ("fsz-repair-256-1kib-records", pending_values(256, 1024)),
        ("fsz-repair-250kib-record", pending_values(1, 250 * 1024)),
    ]
}

/// Commit `pending` to `object_id` through a second store — a service of
/// its own on the cached route (`lua`), with a structural cache of its own,
/// so its commits never touch the measured store's — each record through
/// the `UpdateLayer` handler as a client sends it, its external values
/// stored first. Unmeasured. The cached route, not the WATCH path, so a
/// row of hundreds of structural records on a large base (the index-resize
/// row's 256 prims under 4,096 children) commits in a few milliseconds each
/// rather than a whole-object read each; what the measured call replays is
/// the same either way: each record's relay OMPF, the client's bytes.
async fn commit_pending(
    tv: &TestValkey,
    session_id: &str,
    object_id: u64,
    pending: &[PendingRecord],
    label: &str,
) {
    let peer = service_for(tv, Route::CachedWarm);
    for (i, record) in pending.iter().enumerate() {
        let external: Vec<&[u8]> = record.external.iter().map(Vec::as_slice).collect();
        let ompf = common::owned_delta::delta_to_ompf_bytes(&record.delta, &external);
        let frames: Vec<Bytes> = std::iter::once(grpc_frame(&make_update_header(
            session_id, "layer", object_id,
        )))
        .chain(
            ompf.chunks(CHUNK)
                .map(|c| grpc_frame(&make_update_data(c.to_vec()))),
        )
        .collect();
        Box::pin(handle_update_layer(
            &peer,
            request::<UpdateLayerRequest>(&frames),
        ))
        .await
        .unwrap_or_else(|e| panic!("{label}: pending record {i} must commit: {e:?}"));
    }
}

/// One structural edit (a new child of `SECTION`) on [`Route::CachedWarm`]
/// of a small base, after a second store committed `pending`: the call's
/// cache hit replays them before it plans. Each phase's heap growth must
/// stay within the charge in force ([`Measured::phase_overruns`]). The route
/// check in [`measure_on`] proves the call committed on `structural`, a hit:
/// a repair that reloaded the object would commit on `structural_full`, and a
/// hit that had not replayed the records could not commit at their header. A
/// small base keeps the hint's entry term and CR1's growth term, which the
/// repair does not spend, from covering the repair by accident.
///
/// The same scenario runs once first on a twin layer, its result not
/// checked, and then every member of the store's general pool reads
/// pipelined replies through its own connection, each within the
/// read-buffer limit ([`warm_every_pool_member`]), so the measured call
/// finds every connection's read buffer already at the most it may keep at
/// rest, which a repair read does not grow past.
/// That buffer is a connection's, not the call's: what a drained connection
/// keeps is counted at rest in `C_conn`
/// (`storage::connection_buffers::at_rest_bound_bytes`, outside the upload
/// budget), which charges each general client one warm connection at its
/// class limits — `kept_read_bytes`, `4 × max(read_buffer_bytes, 64 KiB)` of
/// read buffer — plus an idle 8 KiB one per further primary. Fred's
/// retention (`vendor/fred-10.1.0/src/live_edit/retention.rs`) keeps that
/// shape at every moment, whichever call grew a buffer: it elects at most
/// one warm connection per client (`elect_streaming`, a single index), and
/// cuts every other drained connection that carried traffic to the idle
/// shape, a fresh 8 KiB read buffer (`cut`), at the router's next pass
/// (`sweep`) or, for a reply that drains it, at the reply pop before the
/// caller is answered (`on_reply`); a read buffer that held a frame over the
/// class limit is replaced by a fresh one once drained (`on_read`). The
/// repair read goes through a member of the general data pool
/// (`client.pool.next()`, round-robin over its members), so warming only the
/// twin's member left the measured call's read to grow another member's
/// buffer from its idle shape after its charge was trimmed — memory `C_conn`
/// already funds. Warming every member holds the call to its own memory.
async fn assert_cached_hit_repair_within_charge(name: &'static str, pending: Vec<PendingRecord>) {
    assert_cached_hit_repair_within_charge_after(name, pending, true).await;
}

/// [`assert_cached_hit_repair_within_charge`] with no twin run first: the
/// cold case, where the measured call's repair read finds the client's
/// connections as the route's warm-up left them, so a large reply grows its
/// read buffer from its fresh shape inside the measured window (the buffer
/// doubling through its last step holds both allocations at once). Its
/// replies above the read-buffer limit leave no buffer at rest after them,
/// so the twin run's reason does not apply to it; it gates the read's
/// transient at its worst.
async fn assert_cached_hit_repair_within_charge_cold(
    name: &'static str,
    pending: Vec<PendingRecord>,
) {
    assert_cached_hit_repair_within_charge_after(name, pending, false).await;
}

/// The body of the two above, on the small base: with `twin`, the twin run
/// and the warming of every pool member first.
async fn assert_cached_hit_repair_within_charge_after(
    name: &'static str,
    pending: Vec<PendingRecord>,
    twin: bool,
) {
    assert_cached_hit_repair_within_charge_on(
        name,
        Base::Children(0),
        one_new_child(0),
        pending,
        twin,
    )
    .await;
}

/// Grow every member of `tv`'s general data pool's connection read buffer
/// to the most it may keep at rest: four `GET`s of one value, pipelined on
/// each member's own client so their replies can arrive together, each
/// reply kept 4 KiB under the class's read-buffer limit
/// (`GENERAL_LIMITS.read_buffer_bytes`, 64 KiB) so no frame passes it — a
/// frame past the limit would leave a fresh 8 KiB buffer behind once drained
/// (`on_read`), undoing the warming. A buffer so grown is within what the
/// warm connection keeps (`RetentionLimits::kept_read_bytes`, under four
/// times the larger of its largest frame and 64 KiB), which `C_conn` counts
/// at rest for every general client, whichever call grew it; a repair read
/// grows a buffer so shaped no further. Unmeasured. The value's key is under
/// the test's prefix.
async fn warm_every_pool_member(tv: &TestValkey) {
    use fred::interfaces::KeysInterface;
    let reply_bytes = GENERAL_LIMITS.read_buffer_bytes - 4 * 1024;
    let key = format!("{}t7-warm-every-member", tv.prefix());
    tv.client()
        .pool
        .set::<(), _, _>(&key, vec![0x57_u8; reply_bytes], None, None, false)
        .await
        .expect("SET the warming value");
    let members = tv.client().pool.clients();
    assert!(
        members.len() > 1,
        "fixture: a pool of several members, whose round-robin the warming covers"
    );
    for member in members {
        let reads = futures::future::join_all((0..4).map(|_| member.get::<Bytes, _>(&key))).await;
        for read in reads {
            assert_eq!(
                read.expect("GET the warming value").len(),
                reply_bytes,
                "fixture: the warming reply"
            );
        }
    }
}

/// The largest reply a repair read of `pending` can bring: the stream
/// entries' OMPFs together, which every read's reply holds at most.
fn pending_reply_bytes(pending: &[PendingRecord]) -> usize {
    pending
        .iter()
        .map(|record| {
            let external: Vec<&[u8]> = record.external.iter().map(Vec::as_slice).collect();
            common::owned_delta::delta_to_ompf_bytes(&record.delta, &external).len()
        })
        .sum()
}

/// [`assert_cached_hit_repair_within_charge_after`] on `base`, the measured
/// call's edit `edit`: the general form the rows on other bases use.
async fn assert_cached_hit_repair_within_charge_on(
    name: &'static str,
    base: Base,
    edit: OwnedDelta,
    pending: Vec<PendingRecord>,
    twin: bool,
) -> Measured {
    assert_cached_hit_repair_within_charge_checked(name, base, edit, pending, twin, None).await
}

/// [`assert_cached_hit_repair_within_charge_on`], with `entry_premise`
/// checked on the store's entry before each measured call
/// ([`Fixture::entry_premise`]).
async fn assert_cached_hit_repair_within_charge_checked(
    name: &'static str,
    base: Base,
    edit: OwnedDelta,
    pending: Vec<PendingRecord>,
    twin: bool,
    entry_premise: Option<fn(&live_edit::storage::structural_cache::CacheEntry)>,
) -> Measured {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    warm_up(
        &tv,
        Path::UpdateSlow,
        Route::CachedWarm,
        &format!("{name}-warmup"),
    )
    .await;
    let pending_delta: u64 = pending.iter().map(|r| delta_part_len(&r.delta)).sum();
    let reply_bytes = pending_reply_bytes(&pending);
    let fixture = |pending| {
        Fixture::new(name, Feeds::Mixed, edit.clone(), &[])
            .on_base(base)
            .after_pending(pending)
            .with_entry_premise(entry_premise)
    };
    if twin {
        measure_route(
            &tv,
            Path::UpdateSlow,
            Route::CachedWarm,
            &fixture(pending.clone()),
            &format!("{name}-twin"),
        )
        .await;
        warm_every_pool_member(&tv).await;
    }
    let m = measure_route(
        &tv,
        Path::UpdateSlow,
        Route::CachedWarm,
        &fixture(pending),
        name,
    )
    .await;
    eprintln!(
        "== b.fsz {name} ({}): pending delta parts {pending_delta} B, replies {reply_bytes} B; \
         peak {} B; processing-step charge {} B; held max in processing {} B; the peak's phase \
         {}; worst phase {} ==\n   phases: {}\n   each phase, peak/charge: {}",
        if twin {
            "after a twin run, every pool member warmed"
        } else {
            "cold"
        },
        m.peak,
        m.processing_target,
        m.phase_held_max("processing"),
        peak_phase(&m),
        worst_phase_ratio(&m),
        m.phase_summary(),
        m.phases
            .iter()
            .zip(phase_limits(&m.phases, m.cache_entry))
            .enumerate()
            .map(|(i, (p, limit))| format!("{i}:{}/{limit}", p.peak))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let over = m.phase_overruns();
    assert!(
        over.is_empty(),
        "b.fsz {name}: the cache hit's stream repair grew the heap past the charge in force in \
         {over:?} — its reads and replay are not covered by their charge points (RP0, RP1, \
         RP2); phases: {}",
        m.phase_summary()
    );
    m
}

/// The regression (b.fsz (a)): a cache hit whose repair replays one
/// structural record of about 3 MiB, committed by a second store, keeps every
/// phase within the charge in force. Its first read (256 KiB) answers
/// `TOO_LARGE`, the repair reads the record at its size, parses it and
/// replays it into the entry, each held first. With the repair uncharged
/// the parse alone (`k_parse` × 3 MiB) runs under the processing step and
/// the hint of a small base.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_3mib_structural_record_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge("fsz-repair-3mib-prims-record", pending_prims_record())
        .await;
}

/// [`cached_hit_repairing_a_3mib_structural_record_stays_within_its_charge_fsz`]
/// cold ([`assert_cached_hit_repair_within_charge_cold`]): the 3 MiB read
/// grows the read buffer through its 2 MiB to 4 MiB step inside the
/// measured window, and every phase still stays within its charge.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_3mib_structural_record_cold_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge_cold(
        "fsz-repair-3mib-prims-record-cold",
        pending_prims_record(),
    )
    .await;
}

/// [`cached_hit_repairing_a_record_with_a_2mib_external_part_stays_within_its_charge_fsz`]
/// cold ([`assert_cached_hit_repair_within_charge_cold`]): the read of a
/// reply just over 2 MiB, whose buffer step from 2 MiB to 4 MiB holds about
/// three times the reply at once, stays within its charge.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_record_with_a_2mib_external_part_cold_stays_within_its_charge_fsz()
{
    assert_cached_hit_repair_within_charge_cold(
        "fsz-repair-2mib-external-record-cold",
        pending_external_record(),
    )
    .await;
}

/// A cache hit whose repair replays 256 small records (one new prim each) in
/// one read keeps every phase within its charge. Its CR2 phase, the trim to
/// the plan's exact need, has the least margin of the repair rows (about
/// 1.4 KB), so it is the first to fail when the measured build allocates on
/// the call's path what the release build does not (the module doc).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_256_small_records_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge(
        "fsz-repair-256-small-records",
        pending_small_records(256),
    )
    .await;
}

/// A cache hit whose repair replays 256 field writes, the smallest records
/// (most per-entry overhead per reply byte), in one read keeps every phase
/// within its charge.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_256_field_writes_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge(
        "fsz-repair-256-field-writes",
        pending_field_writes(256),
    )
    .await;
}

/// A cache hit whose repair replays a record carrying a 2 MiB external value
/// — a reply of about 2 MiB with a small delta part to parse and replay —
/// keeps every phase within its charge.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_record_with_a_2mib_external_part_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge(
        "fsz-repair-2mib-external-record",
        pending_external_record(),
    )
    .await;
}

/// A cache hit whose repair replays a record of about 1 MiB of new sections
/// — between the first read's 256 KiB and the 4 MiB cap, so two reads and no
/// reload — keeps every phase within its charge.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_1mib_record_over_two_reads_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge(
        "fsz-repair-1mib-sections-record",
        pending_sections_record(),
    )
    .await;
}

// ─────────────────────────────────────────────────────────────────────────────
// b.fsz C: the rows of the two-term reply and per-record (RP2) charge
// ─────────────────────────────────────────────────────────────────────────────

/// `n` pending field writes on `SECTION`, each a value of `value_len` bytes
/// under a key of its own: with `value_len` 1 KiB, records of about 1 KiB
/// that fill a first read with its full entry bound within its byte bound
/// (RP0's worst case, the reply's per-entry and per-byte terms both at
/// work).
fn pending_values(n: u64, value_len: usize) -> Vec<PendingRecord> {
    (0..n)
        .map(|i| {
            pending_of(OwnedDelta {
                set_fields: vec![field(format!("v{i}"), vec![(i % 251) as u8; value_len])],
                ..Default::default()
            })
        })
        .collect()
}

/// A new section `section_id` under the root: the measured edit of the
/// large-base rows, away from their long children list, so its plan lists
/// only the root's two children.
fn new_root_section(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![create_section(0, section_id, format!("r{section_id}"))],
        ..Default::default()
    }
}

/// The compact state a [`Route::CachedWarm`] measurement caches for
/// `base`, built in process as the cache builds and stores it: the base's
/// content and the warm-up's section, compacted (`compact_state`,
/// `CacheEntry::compact_and_weigh`). The large-base rows' premises are
/// checked on it.
fn cached_shape_of(base: Base) -> live_edit::domain::types::CompactState {
    let mut delta = base_delta(base);
    delta
        .struct_commands
        .push(create_section(0, WARM_SECTION, "warm".to_owned()));
    let mut state = compact_of(delta);
    live_edit::storage::structural_cache::compact_state(&mut state);
    for list in state.children.values_mut() {
        list.shrink_to_fit();
    }
    state
}

/// The sections of the D1 base once warmed: `SECTION`, its children and the
/// warm-up's section, 28,672 = 7/8 of 32,768 buckets — a compacted table at
/// its load limit (hashbrown's bucket count for a capacity of 15 or more does
/// not depend on its group width).
const D1_CHILDREN: usize = 28_670;

/// D2's children of `SECTION`: a list whose doubling is ~0.46 MB, on a
/// sections table with room.
const D2_CHILDREN: usize = 30_000;

/// The cached route's per-record growth, tables (A4 (ii), D1): a cache hit
/// whose repair replays one record creating a section into a sections table
/// at its load limit keeps every phase within its charge. The record's
/// per-record charge point (RP2) holds the doubled table before the replay
/// allocates it; no per-byte term covers it (a one-section record of a few
/// dozen delta bytes doubling a 2.4 MB table). Self-check: the table's
/// length is its capacity, on the cached shape built in process and on the
/// store's own entry, which the measured call's cache hit takes
/// ([`d1_store_entry_premise`]). Holds at
/// group width 8 (aarch64) and 16 (x86_64) by construction: the record only
/// inserts, so no tombstone decides the growth, at a capacity past 15
/// hashbrown's bucket count does not depend on the width, and the trailing
/// control bytes the widths differ by (8 B) are nothing beside the
/// megabytes of margin either way.
///
/// Slow: 59.7–60.4 s measured in a debug build (a twin run and the measured
/// run, each on a fresh 28,670-child base), past the global 40 s slow mark;
/// `.config/nextest.toml` raises its kill deadline with the other
/// `cached_hit_repairing_*` rows and puts it in the large-base test group.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_creation_into_a_full_sections_table_stays_within_its_charge_fsz() {
    let shape = cached_shape_of(Base::Children(D1_CHILDREN));
    assert_eq!(
        (shape.sections.len(), shape.sections.capacity()),
        (D1_SECTIONS, D1_SECTIONS),
        "fixture self-check: the cached sections table is at its load limit"
    );
    assert_cached_hit_repair_within_charge_checked(
        "fsz-d1-full-sections-table",
        Base::Children(D1_CHILDREN),
        new_root_section(PENDING_FIRST_ID + 1),
        vec![pending_of(new_root_section(PENDING_FIRST_ID))],
        true,
        Some(d1_store_entry_premise),
    )
    .await;
}

/// The D1 base's sections once warmed: `SECTION`, its [`D1_CHILDREN`]
/// children and the warm-up's section.
const D1_SECTIONS: usize = D1_CHILDREN + 2;

/// D1's premise on the entry the measured call's cache hit takes, as the
/// store holds it: its sections table is at its load limit, so the pending
/// record's one new section doubles it.
fn d1_store_entry_premise(entry: &live_edit::storage::structural_cache::CacheEntry) {
    let sections = &entry.state.sections;
    assert_eq!(
        (sections.len(), sections.capacity()),
        (D1_SECTIONS, D1_SECTIONS),
        "fixture self-check: the store's cached sections table is at its load limit"
    );
}

/// The long-list envelope row (A4 (ii), D2): a cache hit whose repair
/// replays one record adding a child under a parent of 30,000 children keeps
/// every phase of the real call within its charge. The parent's list is at
/// its length (trimmed, as every touched list is), so the record reallocates
/// it to twice that, old and new alive at once, and builds the parent's
/// listing; RP2 holds the list's doubling (its list term), the name vector's
/// growth and the listing as a rewrite count. Self-checks, on the cached
/// shape built in process and on the store's own entry, which the measured
/// call's cache hit takes ([`d2_store_entry_premise`]): the list is at its
/// length, and the sections table has room, so no table term prices
/// anything. Holds at either group width by construction: the list is a
/// `Vec`, whose growth no group width touches, and the table has room at
/// either width.
///
/// An envelope row, not a pin of any one term: no real replay grows a list
/// without building its parent's listing first, and the listing's price
/// (per entry, far above the doubling's 16 B) covers the list term, while
/// the parent's name vector has room at this length, so zeroing either the
/// list or the name-vector term leaves this row green (measured).
/// The listing term is pinned in process by
/// `replay_of_a_child_added_to_a_long_list_on_an_indexed_entry_stays_within_its_hold_fsz`,
/// and the list and name-vector terms by `estimate_replay_record_growth`'s
/// unit tests against std's real growth.
///
/// Slow: 65.1–65.2 s measured in a debug build (a twin run and the measured
/// run, each on a fresh 30,000-child base), past the global 40 s slow mark;
/// `.config/nextest.toml` raises its kill deadline with the other
/// `cached_hit_repairing_*` rows and puts it in the large-base test group.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_child_added_to_a_long_list_stays_within_its_charge_fsz() {
    let shape = cached_shape_of(Base::Children(D2_CHILDREN));
    let list = &shape.children[&SECTION];
    assert_eq!(
        (list.len(), list.capacity()),
        (D2_CHILDREN, D2_CHILDREN),
        "fixture self-check: the long list is at its length"
    );
    assert_eq!(
        live_edit::domain::alloc_model::table_growth_bytes(
            shape.sections.len(),
            shape.sections.capacity(),
            1,
            1,
        ),
        0,
        "fixture self-check: the sections table has room for the record's section"
    );
    assert_cached_hit_repair_within_charge_checked(
        "fsz-d2-long-list",
        Base::Children(D2_CHILDREN),
        new_root_section(PENDING_FIRST_ID + 1),
        vec![pending_of(OwnedDelta {
            struct_commands: vec![create_section(
                SECTION,
                PENDING_FIRST_ID,
                "pending".to_owned(),
            )],
            ..Default::default()
        })],
        true,
        Some(d2_store_entry_premise),
    )
    .await;
}

/// D2's premises on the entry the measured call's cache hit takes, as the
/// store holds it: `SECTION`'s list is at its length, so the pending
/// record's new child reallocates it, and the sections table has room for
/// that child's section.
fn d2_store_entry_premise(entry: &live_edit::storage::structural_cache::CacheEntry) {
    let list = &entry.state.children[&SECTION];
    assert_eq!(
        (list.len(), list.capacity()),
        (D2_CHILDREN, D2_CHILDREN),
        "fixture self-check: the store's cached long list is at its length"
    );
    assert_eq!(
        live_edit::domain::alloc_model::table_growth_bytes(
            entry.state.sections.len(),
            entry.state.sections.capacity(),
            1,
            1,
        ),
        0,
        "fixture self-check: the store's cached sections table has room for the record's section"
    );
}

/// The two-term reply's per-entry rows: a cache hit whose repair replays 256
/// records of about 1 KiB each — a first read full at its entry bound
/// (`STRUCTURAL_REPAIR_INITIAL_ENTRIES`) within its byte bound, then the rest
/// — keeps every phase within its charge, after a twin run.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_256_records_of_1kib_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge(
        "fsz-repair-256-1kib-records",
        pending_values(256, 1024),
    )
    .await;
}

/// [`cached_hit_repairing_256_records_of_1kib_stays_within_its_charge_fsz`]
/// cold: its reads' replies, past 64 KiB, grow the read buffer from its
/// fresh shape inside the measured window.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_256_records_of_1kib_cold_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge_cold(
        "fsz-repair-256-1kib-records-cold",
        pending_values(256, 1024),
    )
    .await;
}

/// The two-term reply's per-byte rows: a cache hit whose repair replays one
/// record of about 250 KiB, within the first read's 256 KiB byte bound,
/// keeps every phase within its charge, after a twin run.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_250kib_record_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge(
        "fsz-repair-250kib-record",
        pending_values(1, 250 * 1024),
    )
    .await;
}

/// [`cached_hit_repairing_a_250kib_record_stays_within_its_charge_fsz`]
/// cold.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_a_250kib_record_cold_stays_within_its_charge_fsz() {
    assert_cached_hit_repair_within_charge_cold(
        "fsz-repair-250kib-record-cold",
        pending_values(1, 250 * 1024),
    )
    .await;
}

/// The index-resize row: a cache hit whose repair replays 256 one-prim
/// records under a parent of exactly 4,096 children keeps every phase within
/// its charge. The parent's name-index vector, 4,096 entries at its
/// capacity when built, doubles at the first record; its children list,
/// trimmed after every record, reallocates at each; RP2 prices both per
/// record. After a twin run.
///
/// Slow: 97.5–98.0 s measured in a debug build (a twin run and the measured
/// run, each committing and repairing 256 pending records), near the global
/// 120 s kill; `.config/nextest.toml` raises its kill deadline with the
/// other `cached_hit_repairing_*` rows and puts it in the large-base test
/// group.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cached_hit_repairing_256_prims_under_a_parent_of_4096_children_stays_within_its_charge_fsz(
) {
    assert_cached_hit_repair_within_charge_on(
        "fsz-repair-index-resize",
        Base::Children(4_096),
        one_new_child(4_096),
        pending_small_records(256),
        true,
    )
    .await;
}

/// A production constant of the stream repair, in hundredths per byte, read
/// back through the charge of a need carrying one million bytes of the part
/// `part` sets: `repair_reply` and `repair_growth` are crate-private.
fn repair_factor_centi(part: fn(u64) -> RepairTransient) -> u64 {
    let price = |repair| object_state_charge_for_test(&ObjectStateNeed::cached_repair(0, repair));
    (price(part(1_000_000)) - price(RepairTransient::default())) / 10_000
}

/// A stream repair charge point: RP0 (`Read`), before a read, its reply
/// bound alone; RP1 (`Replay`), before a chunk's replay, its reply, its
/// largest record's parse and its delta parts; RP2 (`Record`), before each
/// record of that chunk applies, the chunk's reply and that record's own
/// parse and delta part with its priced growth and scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RepairPoint {
    Read,
    Replay,
    Record,
}

/// The stream repair charge point each segment's need is, in order, with
/// its transient. An RP2 parses and replays its record as an RP1 parses and
/// replays its chunk, so they are told apart by their order, as the repair
/// prices them (and as `structural_cache_read_test`'s `repair_points`
/// does): a need with a repair transient that follows an RP1 still owed its
/// RP2s is an RP2 — whatever its delta part — the RP1's chunk owing one per
/// entry (`reply_entry_count`); otherwise one that parses is an RP1, and one
/// that only reads is an RP0. Any other need (a cold start's CR0, the plan's
/// CR1) is none and ends the RP2s still owed. A chunk whose records have no
/// readable delta part would read as an RP0; the rows here all parse.
fn repair_segments(
    needs: &[Option<ObjectStateNeed>],
) -> Vec<Option<(RepairPoint, RepairTransient)>> {
    let mut owed: u64 = 0;
    needs
        .iter()
        .map(|need| match need {
            Some(ObjectStateNeed::Cached { repair, .. })
                if *repair != RepairTransient::default() =>
            {
                let point = if owed > 0 {
                    owed -= 1;
                    RepairPoint::Record
                } else if repair.parse_delta_bytes > 0 {
                    owed = repair.reply_entry_count;
                    RepairPoint::Replay
                } else {
                    RepairPoint::Read
                };
                Some((point, *repair))
            }
            _ => {
                owed = 0;
                None
            }
        })
        .collect()
}

/// b.fsz (a): the stream repair's fit. For each row of
/// [`stream_repair_rows`] — 256 small records (new prims), 256 field writes,
/// a record with a 2 MiB external part, about 1 MiB of new sections, about
/// 3 MiB of new prims, 256 records of about 1 KiB and one of about 250 KiB —
/// one structural edit on a cached small base after a second store committed
/// the row, through the store with the fits' recorder ([`NeedRecorder`]),
/// whose segments split at every charge point: RP0, RP1 and each record's
/// RP2, told apart by their order ([`repair_segments`]). Prints each `SEG`,
/// each RP2 segment as a `REPAIR-RECORD` line (one record's apply, under its
/// own point, left out of the two fits below), each read segment as a
/// `REPLY-READ` line (its peak, the chunk's entries and reply bytes: the
/// two-term reply fit's input, `b` from the single-entry reads and `a` from
/// the others net of `b`, with a `REPLY-FIT` line of this run's figures),
/// then the one-term fit by the derivation rule in `grpc::upload_admission`
/// as it was first taken, and a closing `REPLY-RUN-END` line, the marker
/// that the run is whole:
///
/// 1. `repair_reply` from each read's segment — the RP0 segment the chunk's
///    RP1 follows — net of `F`'s production basis, per byte of the chunk's
///    reply (RP1's `reply_bytes`);
/// 2. `repair_growth` from each RP1 segment, net of `F`'s basis, the reply
///    at the larger of step 1's figure and `repair_reply`'s production basis,
///    and the largest record's parse at `k_parse`'s basis, per delta byte
///    replayed;
/// 3. beside it, the replay's growth measured directly: what invalidating
///    the entry the commit left frees, less what the warmed entry of the same
///    base frees, per delta byte replayed (the edit's own child is in it, a
///    few hundred bytes). Step 2 cannot separate the growth where one
///    record's parse overlaps it, and nets to nothing when the parse at
///    `k_parse`'s basis, or the reply at the read's figure, covers the whole
///    segment. The commit's residual is no measure of it: a read leaves the
///    warm connection's read buffer at its at-rest keep (`C_conn`), which the
///    residual counts in.
///
/// Each figure's recommendation is ⌈figure × 125⌉ hundredths. One run's
/// figures are not the constants: the rule takes the largest over several
/// runs. Report only.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.fsz stream-repair fit segments"]
async fn stream_repair_segments_are_reported_fsz() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let f_basis = FIXED_CHARGE_BYTES as f64 / HEADROOM;
    let k_parse_basis = PARSE_EXPANSION_CENTI as f64 / HEADROOM_CENTI;
    let reply_centi = repair_factor_centi(|bytes| RepairTransient {
        reply_bytes: bytes,
        ..RepairTransient::default()
    });
    let growth_centi = repair_factor_centi(|bytes| RepairTransient {
        replay_delta_bytes: bytes,
        ..RepairTransient::default()
    });
    // What invalidating `object`'s entry on `store` frees.
    let entry_bytes = |store: &LayerStore, object: u64| {
        let held = live_now();
        store.structural_cache().invalidate(object);
        (held - live_now()).max(0) as u64
    };

    // Once-per-process code of the route and of a hit's replay, unmeasured;
    // and the warmed base entry's bytes, which the direct growth is net of.
    let base_entry = {
        let label = "fsz-fit-process-warmup";
        let store = fit_store(&tv, FitRoute::CachedWarm, tv.client().clone());
        let (session, object) = base_layer(&tv, label, Base::Children(0)).await;
        warm_cache(&store, &session, object, label).await;
        commit_pending(&tv, &session, object, &pending_small_records(1), label).await;
        measure_commit(&store, &session, object, &one_new_child(0), label).await;
        let label = "fsz-fit-base-entry";
        let (session, object) = base_layer(&tv, label, Base::Children(0)).await;
        warm_cache(&store, &session, object, label).await;
        entry_bytes(&store, object)
    };

    // (row, read segment's peak, the chunk's reply bytes, its entries)
    let mut reads = Vec::new();
    // (row, replay segment's peak, its transient)
    let mut replays = Vec::new();
    // (row, the entry's bytes after the commit, the delta parts replayed)
    let mut entries = Vec::new();
    for (name, pending) in stream_repair_rows() {
        let store = fit_store(&tv, FitRoute::CachedWarm, tv.client().clone());
        let (session, object) = base_layer(&tv, name, Base::Children(0)).await;
        warm_cache(&store, &session, object, name).await;
        commit_pending(&tv, &session, object, &pending, name).await;
        let (segments, residual, wall) =
            measure_commit(&store, &session, object, &one_new_child(0), name).await;
        let replayed: u64 = pending.iter().map(|r| delta_part_len(&r.delta)).sum();
        // The entry the commit left (the store's only one): its estimate
        // against its heap.
        let weight = store.structural_cache().weight_bytes();
        let entry = entry_bytes(&store, object);
        entries.push((name, entry, replayed));
        eprintln!(
            "REPAIR-ENTRY {name}: the cached entry's weight {weight} B; invalidating it freed \
             {entry} B (the warmed base entry {base_entry} B)"
        );
        for (i, s) in segments.iter().enumerate() {
            eprintln!(
                "SEG {name} CachedWarm #{i} peak {} need {:?}",
                s.peak, s.need
            );
        }
        eprintln!(
            "COMMIT {name} CachedWarm: {} segments, residual {residual}, wall_s {:.3}",
            segments.len(),
            wall.as_secs_f64()
        );
        // Each segment's charge point; RP2's segments (one record's apply)
        // are neither a read's nor a chunk's, and are left out of both fits.
        let points = repair_segments(&segments.iter().map(|s| s.need).collect::<Vec<_>>());
        for (pair, point) in segments.windows(2).zip(points.windows(2)) {
            if let [Some((RepairPoint::Read, _)), Some((RepairPoint::Replay, replay))] = point {
                reads.push((
                    name,
                    pair[0].peak,
                    replay.reply_bytes,
                    replay.reply_entry_count,
                ));
            }
        }
        for (i, (s, point)) in segments.iter().zip(&points).enumerate() {
            match point {
                Some((RepairPoint::Replay, replay)) => replays.push((name, s.peak, *replay)),
                Some((RepairPoint::Record, record)) => eprintln!(
                    "REPAIR-RECORD {name} #{i}: peak {} B, reply {} B, parse {} B, replay {} B",
                    s.peak, record.reply_bytes, record.parse_delta_bytes, record.replay_delta_bytes
                ),
                _ => {}
            }
        }
    }
    assert!(
        !reads.is_empty() && !replays.is_empty(),
        "fixture self-check: the rows' repairs read and replayed under the recorder"
    );

    let net = |peak: u64, terms: f64| (peak as f64 - f_basis - terms).max(0.0);
    let mut reply_figure = 0.0_f64;
    for (name, peak, reply, entries) in &reads {
        let figure = net(*peak, 0.0) / *reply as f64;
        eprintln!(
            "REPAIR-READ {name}: peak {peak} B, reply {reply} B, net of F {f_basis:.0} B: \
             {figure:.4}/B"
        );
        // The two-term reply fit's input (the record's `scripts/fit_repair.py`
        // fits `b` from the single-entry reads and `a` from the others).
        eprintln!(
            "REPLY-READ row={name} entries={entries} bytes={reply} peak={peak} \
             f_basis={f_basis:.1}"
        );
        reply_figure = reply_figure.max(figure);
    }
    // This run's two-term figures, for a glance; the fit takes them over the
    // runs its manifest requires. `b` from the single-entry reads, net of
    // `F`'s basis, per reply byte; `a` from the reads of several entries, net
    // of `F`'s basis and `b` at this run's figure, per entry.
    let b_run = reads
        .iter()
        .filter(|(_, _, _, entries)| *entries == 1)
        .map(|(_, peak, reply, _)| net(*peak, 0.0) / *reply as f64)
        .fold(0.0_f64, f64::max);
    let a_run = reads
        .iter()
        .filter(|(_, _, _, entries)| *entries > 1)
        .map(|(_, peak, reply, entries)| net(*peak, b_run * *reply as f64) / *entries as f64)
        .fold(0.0_f64, f64::max);
    eprintln!(
        "REPLY-FIT this run: b {b_run:.4}/B (⌈× 125⌉ {} centi), a {a_run:.1} B per entry (⌈× \
         1.25⌉ {} B)",
        (b_run * HEADROOM_CENTI).ceil() as u64,
        (a_run * HEADROOM).ceil() as u64
    );
    let reply_net = reply_figure.max(reply_centi as f64 / HEADROOM_CENTI);
    let mut growth_figure = 0.0_f64;
    for (name, peak, r) in &replays {
        let netted = net(
            *peak,
            reply_net * r.reply_bytes as f64 + k_parse_basis * r.parse_delta_bytes as f64,
        );
        let figure = netted / r.replay_delta_bytes as f64;
        eprintln!(
            "REPAIR-REPLAY {name}: peak {peak} B, reply {} B, parse {} B, replay {} B, net of F, \
             the reply at {reply_net:.4}/B and k_parse at {k_parse_basis:.4}/B: {figure:.4}/B",
            r.reply_bytes, r.parse_delta_bytes, r.replay_delta_bytes
        );
        growth_figure = growth_figure.max(figure);
    }
    // The growth the replay leaves in the entry, measured directly.
    let mut entry_figure = 0.0_f64;
    for (name, entry, replayed) in &entries {
        let figure = entry.saturating_sub(base_entry) as f64 / *replayed as f64;
        eprintln!(
            "REPAIR-GROWTH {name}: the entry grew {} B for {replayed} B of delta parts: \
             {figure:.4}/B",
            entry.saturating_sub(base_entry)
        );
        entry_figure = entry_figure.max(figure);
    }
    eprintln!(
        "REPAIR-FIT this run: repair_reply {reply_figure:.4}/B, recommended {} centi (production \
         {reply_centi}); repair_growth {growth_figure:.4}/B (net of the reply at {reply_net:.4}/B), \
         recommended {} centi (production {growth_centi}); the entry's growth measured directly \
         {entry_figure:.4}/B, ⌈× 125⌉ {} centi",
        (reply_figure * HEADROOM_CENTI).ceil() as u64,
        (growth_figure * HEADROOM_CENTI).ceil() as u64,
        (entry_figure * HEADROOM_CENTI).ceil() as u64
    );
    // The marker that the run is whole: every row measured.
    eprintln!(
        "REPLY-RUN-END rows={} reads={}",
        stream_repair_rows().len(),
        reads.len()
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// M3: concurrent structural edits on one gate (b.26e Stage 2)
// ═════════════════════════════════════════════════════════════════════════════
//
// Every test above measures one call on a gate of its own. The M3 arms run
// several one-new-child structural `UpdateLayer`s at once through the real
// handler, all on ONE gate at the production defaults (`B` = 384 MiB, the
// budget at the chart's 5 GiB pod memory limit, with its overflow slot), and hold
// the heap's peak growth over the whole arm to what the gate and the
// catch-up memory budget `T` held at their highest: the gate's in-budget
// high-water (≤ `B`) + the overflow slot's high-water + (cold arms, whose
// cold starts take the fenced read and its grant inside `T`) the `T`
// grant's high-water + `P_f × X`, the fenced read stages' window transient,
// which no grant reserves, at the figure `T`'s derivation funds
// (`storage::memory_budget`'s bound `T × (1 + P_f) + P_f × X`); each solo
// cold start's read is held to that `X` on its own. That bound compares
// high-waters, and at 5,000 children
// the receive allowances of the calls admitted at once (`A`, 8 MiB each)
// dominate the gate's, so it cannot see a call's object state held
// uncharged; each arm also holds every stretch of one gate total to that
// total plus what the calls hold outside the gate ([`run_m3_arm`]), which
// sees it on the warm and WATCH arms only — on a cold 5k arm the allowances
// for the fenced reads (`T` in the phase + `P_f × X`) and the cache entries
// left live (≈ 3.6 MB) cover a 5k call's ≈ 3 MB object state, so A4 stays
// green without its object-state charge (0.896); A1/A2 and the single-call
// CachedCold reproducers pin that charge. The
// charge observer reports the whole gate, so a phase cannot be told apart by
// call: the per-phase figures are arm-level, and each call's own phases are
// reported for one solo call of the same edit on the arm's own store, on a
// gate of its own, before the arm.

/// The demo arms' base: 5,000 children (`SECTION`'s), just above the
/// largest real dev layer measured (3,786 prims).
const M3_DEMO_CHILDREN: usize = 5_000;
/// The demo arms' concurrency: one human and four agents.
const M3_DEMO_CALLS: usize = 5;
/// The stress arms' base.
const M3_STRESS_CHILDREN: usize = DENSE;
/// The stress arms' concurrency.
const M3_STRESS_CALLS: usize = 16;
/// `P_f × X` as the default `T`'s derivation funds it, in tenths of a MiB:
/// 143.0 MiB at `P_f` = 1, the window term
/// `config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT`'s derivation lists
/// (`docs-internal/validation/fanout-executor-b26e-s2-20261009/derivation.json`
/// `window_term`: the structural WATCH read's 149,987,424 B maximum on the
/// build before the fan-out arena, kept by the record's PD-X2 as the
/// conservative bound). It is not the record's rule value,
/// `window_term_by_rule` (142.1 MiB, the catch-up's recorded `X` model,
/// `x_catchup`), which the term exceeds by 0.9 MiB. `X` is a fenced read
/// stage's fan-out window transient, one term the catch-up, the structural
/// cache's cold start and the WATCH read share, which the read's grant does
/// not reserve (`storage::memory_budget`'s module docs: resident catch-up
/// memory is at most `T × (1 + P_f) + P_f × X`).
const M3_PF_X_MODEL_MIB_TENTHS: u64 = 1_430;

/// The record [`M3_PF_X_MODEL_MIB_TENTHS`] is read from.
const M3_X_MODEL_RECORD: &str =
    "docs-internal/validation/fanout-executor-b26e-s2-20261009/derivation.json";

/// `P_f × X` and `X` (per fenced read stage), in bytes, at the default
/// `P_f`, from [`M3_PF_X_MODEL_MIB_TENTHS`] — checked against its record's
/// `p_f × window_term` when the record is in the tree (a published tree may
/// leave it out), so a re-derivation of `T` that moves the term moves this
/// with it.
fn m3_x_model() -> (u64, u64) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(M3_X_MODEL_RECORD);
    if let Ok(text) = std::fs::read_to_string(&path) {
        let record: serde_json::Value =
            serde_json::from_str(&text).expect("the fan-out executor derivation record parses");
        let number = |pointer: &str| {
            record
                .pointer(pointer)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or_else(|| panic!("{pointer} is recorded as a number"))
        };
        let recorded = number("/p_f") * number("/window_term");
        assert_eq!(
            (recorded * 10.0).round() as u64,
            M3_PF_X_MODEL_MIB_TENTHS,
            "fixture self-check: M3's P_f x X is the derivation's p_f x window_term"
        );
    }
    let p_f = u64::from(live_edit::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT);
    let pf_x = M3_PF_X_MODEL_MIB_TENTHS * MIB as u64 / 10;
    (pf_x, pf_x / p_f.max(1))
}

/// `P_f`, the fenced read concurrency the cold arms' bounds assume: the
/// default, which their `P_f × X` term ([`m3_x_model`]) is taken at. A cold
/// arm checks its client's fenced read gate runs at it; that the arm's
/// fenced reads then stay within one `P_f × X` term holds by construction —
/// the structural cache's one-at-a-time cold-start permit, and that gate.
const M3_P_F: usize = live_edit::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT as usize;

/// How often the arm's sampler reads `T` and the gate's queues.
const M3_SAMPLE_PERIOD: std::time::Duration = std::time::Duration::from_micros(200);

/// One M3 arm: `calls` concurrent one-new-child structural `UpdateLayer`s
/// ([`m3_edit`]), call `i` on base layer `i % layers`, each layer
/// [`Base::Children`]`(children)` in a session of its own, on `route`.
#[derive(Clone, Copy, Debug)]
struct M3Arm {
    name: &'static str,
    children: usize,
    calls: usize,
    layers: usize,
    route: Route,
}

/// Call `i`'s edit of an M3 arm on a base of `children`: one new child of
/// `SECTION`, its id and name its own, so calls on one layer never collide
/// and each is found exactly once in the result.
fn m3_edit(children: usize, i: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![create_section(
            SECTION,
            m3_child_id(children, i),
            format!("extra{i}"),
        )],
        ..Default::default()
    }
}

/// The id of call `i`'s new child ([`m3_edit`]).
fn m3_child_id(children: usize, i: usize) -> u64 {
    10 + (children + i) as u64
}

/// The counters an M3 arm reads before and after its calls, outside the
/// measured window: summed over every label but the ones named.
#[derive(Clone, Copy, Debug, Default)]
struct M3Counts {
    /// Plans, commits, attempts and recharges ([`PlanCounts`]).
    plans: PlanCounts,
    /// Commits on the arm's route's `path`.
    on_path: f64,
    /// Processing steps that grew the reservation
    /// (`live_edit_upload_processing_wait_seconds` observations) and, of
    /// them, those granted within 1 ms.
    processing_steps: f64,
    processing_quick: f64,
    /// Admissions, and those granted within 1 ms
    /// (`live_edit_upload_admission_wait_seconds`).
    admissions: f64,
    admissions_quick: f64,
    /// Uploads that ran in the overflow slot.
    overflow_runs: f64,
    /// Cold starts the fenced read ended
    /// (`live_edit_structural_cold_start_reads_total{route="fenced"}`).
    fenced: f64,
    /// Fenced cold starts that failed for their call, which then took the
    /// WATCH path (`live_edit_structural_cache_uncacheable_total{reason=
    /// "fenced_read_failed"}`).
    fenced_failed: f64,
    /// Commits on the WATCH path (`path="watch"`).
    on_watch: f64,
    /// Commits on the cached route with the object cached
    /// (`path="structural"`).
    on_structural: f64,
    /// Waits for a larger charge by an attempt on `path="structural_full"`:
    /// a cold start refused its charge (CR0), or whose repair after it was
    /// refused a read, a chunk's replay or a record's apply (RP0, RP1, RP2;
    /// RP2 stops between two records of a round, at a valid, normalized
    /// cut), stores its entry and waits, and
    /// its resumed attempt is a cache hit, which commits on `structural`.
    recharges_cold: f64,
    /// Structural calls that committed through the WATCH path instead
    /// (`live_edit_update_layer_structural_fallbacks_total`).
    fallbacks: f64,
    /// The object-state bytes the calls held when they returned their
    /// commits (`live_edit_upload_object_state_bytes`, sum).
    object_state_bytes: f64,
}

impl M3Counts {
    /// The counters now, on `svc`'s registry. Renders it, so it allocates:
    /// call it outside a measured window.
    fn read(svc: &LiveEditService, path_label: &str) -> Self {
        let text = render(&svc.metrics);
        let sum =
            |name: &str, labels: &[(&str, &str)]| sample_sum(&text, name, labels).unwrap_or(0.0);
        Self {
            plans: PlanCounts::read(svc),
            on_path: sum(
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", path_label)],
            ),
            processing_steps: sum("live_edit_upload_processing_wait_seconds_count", &[]),
            processing_quick: sum(
                "live_edit_upload_processing_wait_seconds_bucket",
                &[("le", "0.001")],
            ),
            admissions: sum("live_edit_upload_admission_wait_seconds_count", &[]),
            admissions_quick: sum(
                "live_edit_upload_admission_wait_seconds_bucket",
                &[("le", "0.001")],
            ),
            overflow_runs: sum("live_edit_upload_overflow_runs_total", &[]),
            fenced: sum(
                "live_edit_structural_cold_start_reads_total",
                &[("route", "fenced")],
            ),
            fenced_failed: sum(
                "live_edit_structural_cache_uncacheable_total",
                &[("reason", "fenced_read_failed")],
            ),
            on_watch: sum(
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", UPDATE_LAYER_PATH_WATCH)],
            ),
            on_structural: sum(
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", UPDATE_LAYER_PATH_STRUCTURAL)],
            ),
            recharges_cold: sum(
                "live_edit_update_layer_recharges_total",
                &[("path", UPDATE_LAYER_PATH_STRUCTURAL_FULL)],
            ),
            fallbacks: sum("live_edit_update_layer_structural_fallbacks_total", &[]),
            object_state_bytes: sum("live_edit_upload_object_state_bytes_sum", &[]),
        }
    }

    /// What changed since `before`.
    fn since(self, before: Self) -> Self {
        Self {
            plans: self.plans.since(before.plans),
            on_path: self.on_path - before.on_path,
            processing_steps: self.processing_steps - before.processing_steps,
            processing_quick: self.processing_quick - before.processing_quick,
            admissions: self.admissions - before.admissions,
            admissions_quick: self.admissions_quick - before.admissions_quick,
            overflow_runs: self.overflow_runs - before.overflow_runs,
            fenced: self.fenced - before.fenced,
            fenced_failed: self.fenced_failed - before.fenced_failed,
            on_watch: self.on_watch - before.on_watch,
            on_structural: self.on_structural - before.on_structural,
            recharges_cold: self.recharges_cold - before.recharges_cold,
            fallbacks: self.fallbacks - before.fallbacks,
            object_state_bytes: self.object_state_bytes - before.object_state_bytes,
        }
    }
}

/// The labelled counters an M3 arm breaks its attempts and routes down by,
/// each as `(name, label, values)`: the conflicts that send an attempt round
/// again, by kind; the retries a committed or exhausted call took, by
/// outcome; the waits for a larger charge, by path; the structural calls
/// sent to the WATCH path, by reason (the repeat `CACHE_DIVERGED` and the
/// size gates); and the uncacheable verdicts, by reason (a fenced read that
/// failed for its call among them).
const M3_BREAKDOWNS: &[(&str, &str, &[&str])] = &[
    (
        "live_edit_update_layer_conflicts_total",
        "kind",
        UPDATE_LAYER_CONFLICT_KINDS,
    ),
    (
        "live_edit_update_layer_retries_total",
        "outcome",
        UPDATE_LAYER_RETRY_OUTCOMES,
    ),
    (
        "live_edit_update_layer_recharges_total",
        "path",
        UPDATE_LAYER_PATHS,
    ),
    (
        "live_edit_update_layer_structural_fallbacks_total",
        "reason",
        UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS,
    ),
    (
        "live_edit_structural_cache_uncacheable_total",
        "reason",
        STRUCTURAL_CACHE_UNCACHEABLE_REASONS,
    ),
];

/// Every [`M3_BREAKDOWNS`] cell on `svc`'s registry now, as `(counter,
/// label value, count)`, plus `live_edit_update_layer_commit_already_landed_total`
/// (an attempt that found its call's commit already landed: a retry after an
/// outcome-unknown commit). Renders the registry: call it outside a measured
/// window.
fn m3_breakdown(svc: &LiveEditService) -> Vec<(&'static str, &'static str, f64)> {
    let text = render(&svc.metrics);
    let mut cells: Vec<(&'static str, &'static str, f64)> = M3_BREAKDOWNS
        .iter()
        .flat_map(|&(name, label, values)| {
            let text = &text;
            values.iter().map(move |&value| {
                (
                    name,
                    value,
                    sample_sum(text, name, &[(label, value)]).unwrap_or(0.0),
                )
            })
        })
        .collect();
    cells.push((
        "live_edit_update_layer_commit_already_landed_total",
        "",
        sample_sum(
            &text,
            "live_edit_update_layer_commit_already_landed_total",
            &[],
        )
        .unwrap_or(0.0),
    ));
    cells
}

/// What an M3 arm's sampler saw at its highest.
#[derive(Clone, Copy, Debug, Default)]
struct M3Sampled {
    /// The catch-up memory budget `T`'s grants and top-ups together.
    t_high_water: u64,
    /// Processing steps queued on the gate at once.
    processing_waiters: usize,
    /// Admissions queued on the gate at once.
    admission_waiters: usize,
    /// The store's fenced cold-start reads in flight at once
    /// (`StructuralCache::cold_start_reads_in_flight_for_test`): a cold arm's
    /// sign that a fenced read ran.
    fenced_reads: usize,
    samples: u64,
}

/// A thread reading `T` and the gate's queues every [`M3_SAMPLE_PERIOD`]
/// while an arm runs. `MemoryBudget` keeps no high-water mark of its own, so
/// `T`'s is sampled: a grant held for less than a period can be missed,
/// which only lowers the arm's bounds (the safe direction: it can fail an
/// arm that holds, never pass one that does not). Each reading also goes to
/// the phase recorder ([`Phase::t_max`]), so a phase is checked against `T`
/// as it stood during that phase. It also reads how many fenced cold-start
/// reads `cache` has in flight, so a cold arm can check that one ran (that
/// count is bounded by the cache's one-at-a-time permit by construction, so
/// it shows no breach of `P_f`). It allocates nothing while it samples.
struct M3Sampler {
    stop: Arc<AtomicBool>,
    thread: std::thread::JoinHandle<M3Sampled>,
}

impl M3Sampler {
    fn start(
        t: Arc<MemoryBudget>,
        gate: Arc<UploadAdmissionGate>,
        cache: live_edit::storage::structural_cache::StructuralCache,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut seen = M3Sampled::default();
                loop {
                    let last = stop.load(Ordering::SeqCst);
                    let t_now = t.reserved_bytes() + t.overcommit_bytes();
                    seen.t_high_water = seen.t_high_water.max(t_now);
                    // For the phase recorder: each phase's `T` at its highest.
                    T_SAMPLED.store(t_now, Ordering::SeqCst);
                    T_PHASE_MAX.fetch_max(t_now, Ordering::SeqCst);
                    seen.processing_waiters =
                        seen.processing_waiters.max(gate.processing_waiters());
                    seen.admission_waiters = seen.admission_waiters.max(gate.waiters());
                    seen.fenced_reads = seen
                        .fenced_reads
                        .max(cache.cold_start_reads_in_flight_for_test().fenced);
                    seen.samples += 1;
                    if last {
                        break seen;
                    }
                    std::thread::sleep(M3_SAMPLE_PERIOD);
                }
            })
        };
        Self { stop, thread }
    }

    fn stop(self) -> M3Sampled {
        self.stop.store(true, Ordering::SeqCst);
        let seen = self.thread.join().expect("the M3 sampler thread");
        T_SAMPLED.store(0, Ordering::SeqCst);
        T_PHASE_MAX.store(0, Ordering::SeqCst);
        seen
    }
}

/// The client an arm's store runs on: its own, with its data and
/// transaction pools at `pool` and the registry's hooks installed
/// ([`shared_metrics`]), as the composition root installs them — so the
/// cold start's route ticks, and the pools record what they do, inside the
/// window as in production. On a cold arm its plain atomic-read ceilings
/// send a cold start of `object_id` (a base of the arm's size) to the fenced
/// read, with its grant inside `T`: the production ceilings when they
/// already do (the item bound, read off an early-out `TOO_LARGE` answer,
/// over what the plain read is predicted to fit), otherwise the item ceiling
/// one under that bound, the byte ceiling left at its production value.
/// Every other arm runs at the production ceilings. Returns the client and
/// which ceilings it runs at.
async fn m3_client(
    tv: &TestValkey,
    cold: bool,
    object_id: u64,
    pool: u32,
) -> (live_edit::storage::client::ValkeyClient, String) {
    let mut config = common::test_config();
    config.valkey_pool_size = pool;
    let ceilings = if cold {
        m3_fenced_ceilings(tv, object_id, &mut config).await
    } else {
        "production".to_owned()
    };
    let client = connect_test_client(&config)
        .await
        .expect("a client with the arm's pools and ceilings connects");
    client.install_observability_hooks(shared_metrics()).await;
    (client, ceilings)
}

/// Set `config`'s plain ceilings so a cold start of `object_id` takes the
/// fenced read ([`m3_client`]), and say which they are.
async fn m3_fenced_ceilings(
    tv: &TestValkey,
    object_id: u64,
    config: &mut live_edit::config::Config,
) -> String {
    let read = Box::pin(read_object_snapshot_slim_with_ceilings(
        tv.client(),
        tv.key_schema(),
        &object_id.to_string(),
        SlimSnapshotReadConsumer::StructuralCache,
        SnapshotCeilings {
            max_items: 1,
            max_bytes: 1 << 40,
        },
    ))
    .await
    .expect("the base's early-out read");
    let SnapshotReadOf::TooLarge(too_large) = read else {
        panic!("fixture: a one-item ceiling answers every base TOO_LARGE");
    };
    let bound = too_large.item_bound;
    let production = SnapshotCeilings {
        max_items: SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        max_bytes: SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
    };
    if plain_read_predicted_to_fit(bound, production) {
        config.snapshot_atomic_read_max_items = bound - 1;
        format!(
            "item ceiling {} (the item bound {bound} less one: the production ceilings send this \
             base's cold start to the plain read)",
            bound - 1
        )
    } else {
        format!("production (the item bound {bound} sends the cold start to the fenced read)")
    }
}

/// `SECTION`'s children list on `object_id`, read whole.
async fn m3_children_of_section(tv: &TestValkey, object_id: u64) -> Vec<u64> {
    let read = Box::pin(read_object_snapshot_slim_with_ceilings(
        tv.client(),
        tv.key_schema(),
        &object_id.to_string(),
        SlimSnapshotReadConsumer::StructuralCache,
        SnapshotCeilings {
            max_items: 1 << 40,
            max_bytes: 1 << 40,
        },
    ))
    .await
    .expect("the edited layer reads");
    let SnapshotReadOf::Snapshot(snapshot) = read else {
        panic!("an unbounded read is never answered TOO_LARGE");
    };
    snapshot
        .state
        .children
        .get(&SECTION)
        .cloned()
        .unwrap_or_default()
}

/// Run `arm` once and assert its envelope: the heap's peak growth over the
/// arm within the gate's in-budget high-water + the overflow slot's
/// high-water + (on [`Route::CachedCold`]) `T`'s high-water + `P_f × X`
/// ([`m3_x_model`]); each phase's peak within the gate total in force during
/// it + `F` per call + the arm's residual (+ on a cold arm, while a grant is
/// held, `T` in the phase + `P_f × X`); on every arm, that residual within
/// what the window grew the structural-cache entries by, measured by
/// invalidating them (on a warm arm, less each layer's entry before the
/// window, measured on a twin layer), + `F` per call; on a cold arm, the
/// solo call's fenced read's heap above its grant within `X`; every call
/// committed exactly once on the arm's route — on a cold arm, or on the
/// WATCH path after its fenced read failed for it, or on `structural` after
/// its cold start waited for its charge — or was refused with nothing
/// applied (a documented capacity refusal: `UNAVAILABLE`, "nothing was applied"), and
/// on a demo arm none is refused; every committed call's child is in its
/// layer exactly once and no refused call's is. A cold arm also self-checks
/// its fixture: its client's fenced read gate runs at the default `P_f` the
/// `P_f × X` term is taken at, every layer's cold start took the fenced read
/// (loaded by it or failed for its call), the solo call's cold start took the
/// fenced read with a grant, and a fenced read was seen in flight.
///
/// In order, outside the measured window: the route's warm-up
/// ([`warm_up`]); the base layers; the store, on its own client
/// ([`m3_client`]); one solo call of the same edit on
/// the store, on a gate of its own ([`measure_on`]), which runs the route's
/// once-per-process code and gives one call's own phases; on
/// [`Route::CachedWarm`], one unmeasured structural commit per layer to
/// cache it, and the same on one twin layer, whose entry is then invalidated
/// to measure what each layer's entry holds before the window. Then every
/// call is spawned at once on the multi-thread runtime with the gate's
/// charge observer and the sampler running, and awaited.
async fn run_m3_arm(arm: M3Arm) {
    assert!(
        arm.layers > 0 && arm.calls.is_multiple_of(arm.layers),
        "fixture: {arm:?} spreads its calls evenly over its layers"
    );
    assert!(
        arm.route != Route::CachedCold || arm.layers == arm.calls,
        "fixture: {arm:?}: each call of a cold arm is its layer's first edit"
    );
    let name = arm.name;
    let cold = arm.route == Route::CachedCold;
    let _serial = measure_lock().await;
    // Every WATCH racer holds its own transaction connection, as on the
    // chart (`valkey.poolSize` 64).
    let pool = u32::try_from(arm.calls.max(4)).expect("an arm's call count fits u32");
    let tv = TestValkey::setup_with_pool_size(pool).await;
    let base = Base::Children(arm.children);
    warm_up(&tv, Path::UpdateSlow, arm.route, &format!("{name}-warmup")).await;
    let mut layers = Vec::with_capacity(arm.layers);
    for i in 0..arm.layers {
        layers.push(base_layer(&tv, &format!("{name}-layer{i}"), base).await);
    }
    let (client, ceilings) = m3_client(&tv, cold, layers[0].1, pool).await;
    let t_budget = Arc::clone(&client.catchup_build_budget);
    // The cold arms' bounds take `P_f × X` at the default `P_f`: the client's
    // fenced read gate must run at it.
    let fenced_gate_limit = client.catchup_fenced_read_gate.limit();
    if cold {
        assert_eq!(
            fenced_gate_limit, M3_P_F,
            "fixture self-check: {name}: the arm's client runs the fenced read concurrency P_f \
             the bounds' P_f x X term is taken at"
        );
    }
    // As the composition root wires it. No call here runs under a request
    // deadline (a debug build cannot meet the production one), so a cold
    // start's fenced read waits at most its size's `reader_wait_max`, the
    // waiting cap with no deadline in scope; one that reaches it fails for
    // its call, which takes the WATCH path.
    let store_svc = service_on_with(&tv, arm.route, client, |store| {
        store.with_fence_wait_deadline(
            std::time::Duration::from_millis(RPC_DEADLINE_MS_DEFAULT),
            Arc::new(live_edit::grpc::deadline::remaining),
        )
    });
    let gate_on_metrics = || {
        Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig::default(),
            Some(Arc::clone(&store_svc.metrics)),
        ))
    };

    // One call of the same edit alone, on a gate of its own, with `T`
    // sampled across it: on a cold arm, the grant its cold start's fenced
    // read held.
    let solo_fixture =
        Fixture::new(name, Feeds::Mixed, m3_edit(arm.children, 0), &[]).on_base(base);
    let solo_gate = gate_on_metrics();
    let solo_sampler = M3Sampler::start(
        Arc::clone(&t_budget),
        Arc::clone(&solo_gate),
        store_svc.layer_store.structural_cache().clone(),
    );
    let solo = measure_on(
        LiveEditService {
            upload_admission: solo_gate,
            ..store_svc.clone()
        },
        &tv,
        Path::UpdateSlow,
        arm.route,
        &solo_fixture,
        &format!("{name}-solo"),
    )
    .await;
    let solo_seen = solo_sampler.stop();
    let solo_t = solo_seen.t_high_water;
    // The solo call's cold-start read: its phases hold the processing step's
    // charge, and the read's memory is its grant's, inside `T`, but for its
    // fan-out window's transient `X`, which the grant does not reserve. Per
    // phase, the heap above the processing step and the grant as it stood
    // then: the read's `X`, at its largest.
    let solo_read = solo
        .phases
        .iter()
        .zip(solo.labels())
        .filter(|(_, label)| *label == COLD_START_READ)
        .map(|(p, _)| {
            (
                p.peak.saturating_sub(p.held() + p.t_max),
                p.peak,
                p.held(),
                p.t_max,
            )
        })
        .max();

    // On a warm arm, what one layer's entry holds before the window (`E0`):
    // a twin of the arm's layers, built on the same base and warmed by the
    // same commit, its entry invalidated here and the freed bytes taken as
    // `entries` is after the window. The arm's layers' entries hold the same
    // before the window, so `entries` less `E0` per layer is what the
    // window's commits grew them by; that `entries` is at least `E0` per
    // layer is checked where the growth is taken.
    let mut twin_entry = 0;
    if arm.route == Route::CachedWarm {
        let twin = base_layer(&tv, &format!("{name}-twin"), base).await;
        for (session_id, object_id) in layers.iter().chain(std::iter::once(&twin)) {
            let warm = OwnedDelta {
                struct_commands: vec![create_section(0, WARM_SECTION, "warm".to_owned())],
                ..Default::default()
            };
            let relay = Bytes::from(common::owned_delta::delta_to_ompf_bytes(&warm, &[]));
            Box::pin(store_svc.layer_store.update_layer(
                session_id,
                "layer",
                *object_id,
                Some(warm),
                Some(relay),
                "m3-warmup",
                0,
            ))
            .await
            .unwrap_or_else(|e| panic!("{name}: the cache warm-up must commit: {e:?}"));
        }
        let held = live_now();
        store_svc.layer_store.structural_cache().invalidate(twin.1);
        twin_entry = (held - live_now()).max(0) as u64;
        assert!(
            twin_entry > 0,
            "fixture self-check: {name}: invalidating the warmed twin's entry freed nothing"
        );
    }

    let gate = gate_on_metrics();
    assert_eq!(
        gate.budget_bytes(),
        384 * MIB as u64,
        "fixture: the gate runs at the chart's default B"
    );
    let svc = Arc::new(LiveEditService {
        upload_admission: Arc::clone(&gate),
        ..store_svc.clone()
    });
    let path_label = arm
        .route
        .path_label(Some(&m3_edit(arm.children, 0)))
        .expect("an M3 route records its path");
    let mut requests = Vec::with_capacity(arm.calls);
    for i in 0..arm.calls {
        let (session_id, object_id) = &layers[i % arm.layers];
        let ompf = common::owned_delta::delta_to_ompf_bytes(&m3_edit(arm.children, i), &[]);
        let frames: Vec<Bytes> = std::iter::once(grpc_frame(&make_update_header(
            session_id, "layer", *object_id,
        )))
        .chain(
            ompf.chunks(CHUNK)
                .map(|c| grpc_frame(&make_update_data(c.to_vec()))),
        )
        .collect();
        requests.push(request::<UpdateLayerRequest>(&frames));
    }
    let before = M3Counts::read(&svc, path_label);
    let breakdown_before = m3_breakdown(&svc);

    // The measured window: every call at once.
    let recorder = PhaseRecorder::install(&gate);
    let sampler = M3Sampler::start(
        Arc::clone(&t_budget),
        Arc::clone(&gate),
        svc.layer_store.structural_cache().clone(),
    );
    let mut handles = Vec::with_capacity(arm.calls);
    let mut outcomes = Vec::with_capacity(arm.calls);
    let arm_start = std::time::Instant::now();
    recorder.begin();
    for (i, req) in requests.into_iter().enumerate() {
        let svc = Arc::clone(&svc);
        handles.push(tokio::spawn(async move {
            let started = std::time::Instant::now();
            let result = Box::pin(handle_update_layer(&svc, req)).await.map(drop);
            (i, started, std::time::Instant::now(), result)
        }));
    }
    for handle in handles {
        outcomes.push(handle.await.expect("an M3 call's task"));
    }
    let (phases, residual) = recorder.finish();
    let wall = arm_start.elapsed();
    let sampled = sampler.stop();
    gate.set_charge_observer_for_test(None);

    // The structural-cache entries left live: the cache's, outside B, once
    // each call is released (what invalidating them frees). On a cold arm
    // the entries the cold starts stored in the window; on a warm arm the
    // warmed entries as the commits replaced them, which include `E0` per
    // layer from before the window (`twin_entry`); on a WATCH arm, whose
    // `txn` route never fills the cache, none.
    if arm.route == Route::Watch {
        assert_eq!(
            svc.layer_store.structural_cache().entry_count(),
            0,
            "fixture self-check: {name}: the `txn` route left the structural cache empty"
        );
    }
    let entries = {
        let held = live_now();
        for (_, object_id) in &layers {
            svc.layer_store.structural_cache().invalidate(*object_id);
        }
        (held - live_now()).max(0) as u64
    };
    let counts = M3Counts::read(&svc, path_label).since(before);
    // What sent attempts round again and which way the calls went: every
    // breakdown cell that moved over the arm.
    let breakdown: Vec<String> = m3_breakdown(&svc)
        .into_iter()
        .zip(breakdown_before)
        .filter(|((_, _, after), (_, _, before))| after > before)
        .map(|((name, value, after), (_, _, before))| {
            format!(
                "{}{{{value}}} {}",
                name.trim_start_matches("live_edit_update_layer_")
                    .trim_start_matches("live_edit_"),
                after - before
            )
        })
        .collect();

    // Every call committed once, or was refused with nothing applied.
    let mut refused = Vec::new();
    let mut failed = Vec::new();
    for (i, _, _, result) in &outcomes {
        match result {
            Ok(()) => {}
            Err(status)
                if status.code() == tonic::Code::Unavailable
                    && status.message().contains("nothing was applied") =>
            {
                refused.push(*i);
            }
            Err(status) => failed.push(format!("call {i}: {status:?}")),
        }
    }
    let committed = arm.calls - refused.len();
    let mut misapplied = Vec::new();
    for (l, (_, object_id)) in layers.iter().enumerate() {
        let list = m3_children_of_section(&tv, *object_id).await;
        let mut landed = 0;
        for i in (0..arm.calls).filter(|i| i % arm.layers == l) {
            let id = m3_child_id(arm.children, i);
            let found = list.iter().filter(|&&c| c == id).count();
            let want = usize::from(!refused.contains(&i));
            landed += want;
            if found != want {
                misapplied.push(format!(
                    "call {i}: child {id} is in layer {l} {found} times, not {want}"
                ));
            }
        }
        if list.len() != arm.children + landed {
            misapplied.push(format!(
                "layer {l}: {} children, not the base's {} + {landed}",
                list.len(),
                arm.children
            ));
        }
    }

    // The envelope.
    let peak = phases.iter().map(|p| p.peak).max().unwrap_or(0);
    let in_budget = gate.high_water_bytes();
    let slot = gate.overflow_high_water_bytes();
    // On a cold arm the documented catch-up bound (`storage::memory_budget`):
    // `T`'s grants and top-ups, here their high-water, and beside them
    // `P_f × X`, the fenced read stages' window transient no grant reserves,
    // at the figure `T`'s derivation funds ([`m3_x_model`]), never the
    // measured one.
    let (pf_x, x_model) = m3_x_model();
    let t_term = if cold { sampled.t_high_water } else { 0 };
    let x_term = if cold { pf_x } else { 0 };
    let bound = in_budget + slot + t_term + x_term;
    // Arm-level phases: each stretch of one gate total, its peak against that
    // total plus what the calls may hold outside the gate at any moment: `F`
    // per call, for a call before its admission or returning after its
    // release (the single-call envelope holds those to the admission charge
    // and the last charge, which `F` covers); the arm's residual, what
    // outlives the calls (the cache entries the commits left, which are the
    // cache's once their calls are released: on a cold arm the entries the
    // cold starts stored, on a warm one the growth of each entry its commit
    // replaced, on a WATCH arm none), as the single-call envelope allows a
    // cold start's entry after its release, and held below to that growth of
    // the entries + `F` per call on every arm; and on a cold arm `T` as it
    // stood in the phase ([`Phase::t_max`]) and, only while a grant is held
    // — a fenced read in flight — `P_f × X`. So a phase with a read in flight
    // holds that read's heap above its grant and every charge to `X`: the
    // per-read check.
    let outside = arm.calls as u64 * FIXED_CHARGE_BYTES;
    let extra = |p: &Phase| {
        let catch_up = if cold && p.t_max > 0 {
            p.t_max + pf_x
        } else {
            0
        };
        outside + residual + catch_up
    };
    let phase_over: Vec<String> = phases
        .iter()
        .enumerate()
        .filter(|(_, p)| p.peak > p.held() + extra(p))
        .map(|(i, p)| {
            format!(
                "phase {i}: {} B over {} B held + {} B ({:.3}x)",
                p.peak,
                p.held(),
                extra(p),
                p.peak as f64 / (p.held() + extra(p)) as f64
            )
        })
        .collect();
    let worst_phase = phases
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.peak as f64 / (p.held() + extra(p)) as f64, *p))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let unheld_peak = phases
        .iter()
        .filter(|p| p.held() == 0)
        .map(|p| p.peak)
        .max()
        .unwrap_or(0);
    let per_call = |v: f64| v / arm.calls as f64;
    let mut by_end: Vec<_> = outcomes.iter().collect();
    by_end.sort_by_key(|(_, _, ended, _)| *ended);
    let timeline: Vec<String> = by_end
        .iter()
        .map(|(i, started, ended, result)| {
            format!(
                "#{i} {}..{} ms{}",
                started.duration_since(arm_start).as_millis(),
                ended.duration_since(arm_start).as_millis(),
                if result.is_ok() { "" } else { " refused" }
            )
        })
        .collect();
    eprintln!(
        "M3 {name}: {} calls on {} layer(s) of {} children, {:?}; ceilings {ceilings}\n   \
         peak {peak} B; B high-water {in_budget} B; slot high-water {slot} B; T high-water {} \
         B (sampled, {} samples; in the bound: {}); P_f x X {x_term} B; bound {bound} B; \
         peak/bound {:.3}\n   \
         per call: recharges {:.2}, attempts {:.2}, plans in place {:.2}, processing waits over \
         1 ms {:.2} (of {:.2} steps), admission waits over 1 ms {:.2}; commits {} on path {} \
         ({} expected), refused {refused:?}; overflow runs {}; fenced cold starts {} (failed for \
         their call {}; commits on watch {}); WATCH fallbacks {}; mean object-state bytes held \
         at commit {:.0}\n   \
         gate queues at most: processing {}, admission {}; fenced reads at once {}; \
         structural-cache entries freed {entries} B{}; residual {residual} B\n   \
         attempts {} for {} commits ({} extra); retries, conflicts and routes moved: \
         {breakdown:?}\n   \
         arm-level worst phase: {}; unheld phases' peak {unheld_peak} B; {} phases\n   \
         wall {:.1} s (indicative: slot B shared)\n   \
         completion order: {}\n   \
         gate total by phase, MB (phase peak in brackets; arms of up to 80 phases): {}",
        arm.calls,
        arm.layers,
        arm.children,
        arm.route,
        sampled.t_high_water,
        sampled.samples,
        if cold { "yes" } else { "no" },
        peak as f64 / bound.max(1) as f64,
        per_call(counts.plans.recharges),
        per_call(counts.plans.attempts),
        per_call(counts.plans.in_place_plans as f64),
        per_call(counts.processing_steps - counts.processing_quick),
        per_call(counts.processing_steps),
        per_call(counts.admissions - counts.admissions_quick),
        counts.plans.commits,
        counts.on_path,
        committed,
        counts.overflow_runs,
        counts.fenced,
        counts.fenced_failed,
        counts.on_watch,
        counts.fallbacks,
        counts.object_state_bytes / counts.plans.commits.max(1.0),
        sampled.processing_waiters,
        sampled.admission_waiters,
        sampled.fenced_reads,
        if arm.route == Route::CachedWarm {
            format!(" (each held {twin_entry} B before the window, by the twin)")
        } else {
            String::new()
        },
        counts.plans.attempts,
        counts.plans.commits,
        counts.plans.attempts - counts.plans.commits,
        worst_phase.map_or("none".to_owned(), |(i, ratio, p)| format!(
            "phase {i} peak {} B over held {} B + {} B (T in the phase {} B) ({ratio:.3})",
            p.peak,
            p.held(),
            extra(&p),
            p.t_max
        )),
        phases.len(),
        wall.as_secs_f64(),
        timeline.join(", "),
        if phases.len() <= 80 {
            phases
                .iter()
                .map(|p| format!("{:.2} [{:.2}]", p.held() as f64 / 1e6, p.peak as f64 / 1e6))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            "-".to_owned()
        }
    );
    eprintln!(
        "M3 {name} solo: peak {} B; call high-water {} B (largest charge one call held alone); \
         held max in processing {} B; worst phase over its charge {}; cold-start read {}; wall \
         {:.1} s; {:?}\n   \
         phases: {}",
        solo.peak,
        solo.charge,
        solo.phase_held_max("processing"),
        worst_phase_ratio(&solo),
        // The fan-out window's transient `X` (its commands in flight and
        // their replies) is outside the read's grant by design
        // (`storage::memory_budget`'s module docs), checked below against
        // the `X` `T`'s derivation funds.
        solo_read.map_or("none".to_owned(), |(x, peak, held, t)| format!(
            "peak {peak} B over the processing step's {held} B + the grant {t} B in that phase \
             (T's high-water over the call {solo_t} B): X {x} B against X_model {x_model} B \
             ({:.3})",
            x as f64 / x_model.max(1) as f64
        )),
        solo.wall.as_secs_f64(),
        solo.plans,
        solo.phase_summary()
    );

    assert!(
        failed.is_empty(),
        "b.26e M3 {name}: a call failed other than by a documented capacity refusal: {failed:?}"
    );
    assert!(
        misapplied.is_empty(),
        "b.26e M3 {name}: a call's edit was not applied exactly once (or a refused one was \
         applied): {misapplied:?}"
    );
    // On a cold arm a fenced read that fails for its call — its waiting cap
    // reached while it queues behind the replica's one fenced read at a time
    // — sends that call to the WATCH path, for that call only (the cold
    // start's route, step 6): every other cold start loads its layer. And a
    // cold start whose charge at seam A is refused still stores its entry and
    // waits (CR0, `ColdStart::NeedsCharge`), as does one whose repair after
    // it is refused a read, a chunk's replay or a record's apply (RP0, RP1,
    // RP2; an RP2 refusal stops between two records of a round, the entry
    // a valid, normalized cut), both on `structural_full`;
    // its resumed attempt is a cache hit and commits on `structural`.
    let committed = committed as f64;
    let (fell_back, resumed) = if cold {
        (counts.fenced_failed, counts.recharges_cold)
    } else {
        (0.0, 0.0)
    };
    // Commits on (the route's own path, `structural`, `watch`).
    let expected = match arm.route {
        Route::Watch => (committed, 0.0, committed),
        Route::CachedCold => (committed - fell_back - resumed, resumed, fell_back),
        _ => (committed, committed, 0.0),
    };
    assert_eq!(
        (
            counts.plans.commits,
            (counts.on_path, counts.on_structural, counts.on_watch)
        ),
        (committed, expected),
        "b.26e M3 {name}: every call that answered OK committed once, on the {:?} route (path \
         {path_label}){}",
        arm.route,
        if cold {
            ", on the WATCH path after its fenced read failed for it, or on `structural` after \
             its cold start waited for its charge"
        } else {
            ""
        }
    );
    if cold {
        assert_eq!(
            counts.fenced + counts.fenced_failed,
            arm.layers as f64,
            "fixture self-check: {name}: every layer's cold start took the fenced read"
        );
        // The solo call's cold start was found, and took the fenced read with
        // a grant inside `T`: the per-read check below has a read to check.
        assert!(
            solo_read.is_some_and(|(_, _, _, grant)| grant > 0),
            "fixture self-check: {name}: the solo call's cold start took the fenced read with a \
             grant ({solo_read:?})"
        );
        // A fenced read ran. (One `P_f × X` term in both bounds holds by
        // construction: the cache's one-at-a-time cold-start permit and the
        // client's fenced read gate, checked at `P_f` above.)
        let fenced_at_once = sampled.fenced_reads.max(solo_seen.fenced_reads);
        assert!(
            fenced_at_once >= 1,
            "fixture self-check: {name}: no fenced cold-start read was seen in flight"
        );
    }
    // What the arm left live is what the window grew the structural-cache
    // entries by, measured apart by invalidating them, and at most `F` per
    // call more: on a cold arm the whole entries, which the window stored; on
    // a warm arm the entries less `E0` per layer (the twin's); on a WATCH arm,
    // whose cache is empty, nothing, so `residual ≤ F` per call. The residual
    // every phase's allowance carries is no larger than that, so memory that
    // outlives its call widens that allowance by at most `F` per call.
    // Each warm layer's entry starts the window at `E0` and its commits only
    // add a child, so the entries hold at least `E0` per layer at its end
    // unless one left the cache (evicted, dropped from the side store, or
    // invalidated by an audit) or the twin overstated `E0`. Either makes
    // `entries_grown` smaller, which only tightens the check: a missing
    // entry's bytes leave `residual` (the net live heap since the baseline)
    // too, so the unclamped comparison stays sound. What would loosen it is
    // clamping a negative growth to 0, which adds the shortfall (up to `E0`
    // per missing entry, or the twin's overstatement) back into the
    // allowance; so when `entries` falls below `E0` per layer the fixture
    // fails here instead. The error that does loosen the bound is a twin
    // that understates `E0`: the allowance grows by the shortfall per layer,
    // and nothing here catches it.
    let layers_e0 = arm.layers as u64 * twin_entry;
    let entries_grown = entries.checked_sub(layers_e0).unwrap_or_else(|| {
        panic!(
            "fixture self-check: {name}: every warm layer's entry is resident at the window's \
             end and holds at least the twin's E0: invalidating the {} layers' entries freed \
             {entries} B, less than {} x the twin's {twin_entry} B = {layers_e0} B",
            arm.layers, arm.layers
        )
    });
    assert!(
        residual <= entries_grown + outside,
        "b.26e M3 {name}: the arm left {residual} B live, more than what the window grew the \
         structural-cache entries by, {entries_grown} B (invalidating them freed {entries} B, \
         less {} layer(s) x the {twin_entry} B each held before the window), + {outside} B \
         ({} calls x F), on the {:?} route",
        arm.layers,
        arm.calls,
        arm.route
    );
    // A demo arm's five calls fit the budget at once: none may be refused.
    // A stress arm may meet documented capacity refusals.
    if arm.children == M3_DEMO_CHILDREN {
        assert!(
            refused.is_empty(),
            "b.26e M3 {name}: a demo arm's call was refused: {refused:?}"
        );
    }
    let mut over = Vec::new();
    if peak > bound {
        over.push(format!(
            "the heap grew {peak} B over the arm, past the gate's in-budget high-water \
             {in_budget} B + the overflow slot's high-water {slot} B + the T grant's high-water \
             {t_term} B + P_f x X {x_term} B = {bound} B ({:.3}x)",
            peak as f64 / bound.max(1) as f64
        ));
    }
    // The per-read check on the solo call: its fenced read's heap above the
    // processing step and its grant, phase by phase, within the `X` that `T`'s
    // derivation funds. In this debug build each in-flight command's future is
    // larger than in a release build, so the check is no looser here than
    // there for that part of `X`; the figure is a release RSS one, which
    // M5a's release build compares on its own terms.
    if let Some((x, read_peak, held, t)) = solo_read.filter(|_| cold) {
        if x > x_model {
            over.push(format!(
                "the solo call's fenced cold-start read grew {read_peak} B over the processing \
                 step's {held} B + its grant {t} B: {x} B outside the grant, past the X that T's \
                 derivation funds, {x_model} B ({:.3}x)",
                x as f64 / x_model.max(1) as f64
            ));
        }
    }
    // The arm's bound compares high-waters: at 5,000 children the receive
    // allowances of the calls admitted at once (`A`, 8 MiB each) dominate the
    // gate's, so it cannot see the object state a call holds uncharged. Each
    // phase against the gate total in force during it can, on the warm and
    // WATCH arms only: on a cold 5k arm the allowance while a read is in
    // flight (`T` in the phase + `P_f × X`) and the residual (the cache
    // entries, ≈ 3.6 MB) cover a 5k call's ≈ 3 MB object state, so A4 stays
    // green without its object-state charge (0.896 under that perturbation).
    if !phase_over.is_empty() {
        over.push(format!(
            "{} of {} phases passed the gate total in force + {outside} B outside the gate \
             ({} calls x F) + the arm's residual {residual} B{}: {:?}",
            phase_over.len(),
            phases.len(),
            arm.calls,
            if cold {
                format!(" + (while a grant is held) T in the phase + P_f x X {pf_x} B")
            } else {
                String::new()
            },
            &phase_over[..phase_over.len().min(8)]
        ));
    }
    assert!(
        over.is_empty(),
        "b.26e M3 {name}: memory the calls held that neither the upload gate nor the catch-up \
         budget charged: {over:?}"
    );
}

/// The phase holding `m`'s peak, against the charge in force during it, as
/// `phase i (label) peak/limit (ratio)`.
fn peak_phase(m: &Measured) -> String {
    m.phases
        .iter()
        .zip(phase_limits(&m.phases, m.cache_entry))
        .zip(m.labels())
        .enumerate()
        .max_by_key(|(_, ((p, _), _))| p.peak)
        .map_or_else(
            || "none".to_owned(),
            |(i, ((p, limit), label))| {
                format!(
                    "phase {i} ({label}) {}/{limit} ({:.3})",
                    p.peak,
                    p.peak as f64 / limit.max(1) as f64
                )
            },
        )
}

/// `m`'s worst phase against the charge in force during it, as `phase i
/// (label) peak/limit (ratio)`, the cold start's read excepted
/// ([`Measured::labels`]).
fn worst_phase_ratio(m: &Measured) -> String {
    m.phases
        .iter()
        .zip(phase_limits(&m.phases, m.cache_entry))
        .zip(m.labels())
        .enumerate()
        .filter(|(_, (_, label))| *label != COLD_START_READ)
        .map(|(i, ((p, limit), label))| {
            (
                p.peak as f64 / limit.max(1) as f64,
                format!(
                    "phase {i} ({label}) {}/{limit} ({:.3})",
                    p.peak,
                    p.peak as f64 / limit.max(1) as f64
                ),
            )
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map_or_else(|| "none".to_owned(), |(_, s)| s)
}

/// M3 A1 (demo): five agents' one-new-child edits at once, each on its own
/// 5,000-child layer, under `lua` with every layer cached.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn m3_a1_lua_five_cached_layers_stay_within_the_gate_b26e() {
    run_m3_arm(M3Arm {
        name: "m3-a1-lua-5-layers-warm",
        children: M3_DEMO_CHILDREN,
        calls: M3_DEMO_CALLS,
        layers: M3_DEMO_CALLS,
        route: Route::CachedWarm,
    })
    .await;
}

/// M3 A2 (demo): the same five edits all on one cached 5,000-child layer
/// under `lua`: they queue on the object's structural permit, one planning
/// at a time.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn m3_a2_lua_one_cached_layer_stays_within_the_gate_b26e() {
    run_m3_arm(M3Arm {
        name: "m3-a2-lua-1-layer-warm",
        children: M3_DEMO_CHILDREN,
        calls: M3_DEMO_CALLS,
        layers: 1,
        route: Route::CachedWarm,
    })
    .await;
}

/// M3 A3 (demo): the five edits on one 5,000-child layer under `txn`, the
/// WATCH path, racing for the object.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn m3_a3_txn_one_layer_stays_within_the_gate_b26e() {
    run_m3_arm(M3Arm {
        name: "m3-a3-txn-1-layer",
        children: M3_DEMO_CHILDREN,
        calls: M3_DEMO_CALLS,
        layers: 1,
        route: Route::Watch,
    })
    .await;
}

/// M3 A4 (demo): five edits on five uncached 5,000-child layers under
/// `lua`, each layer's cold start through the fenced read with its grant
/// inside `T`. Its per-phase check does not see a call's object state held
/// uncharged: the allowances for the fenced reads (`T` in the phase +
/// `P_f × X`) and the cache entries left live (≈ 3.6 MB) cover a 5k call's
/// ≈ 3 MB object state, so it stays green without that charge (0.896). A1,
/// A2 and the single-call CachedCold reproducers
/// (`slow_path_*_within_*_charge_b26e` on [`Route::CachedCold`], among them
/// `slow_path_one_new_child_on_5k_base_stays_within_its_charge_b26e`) pin it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn m3_a4_lua_five_cold_layers_stay_within_the_gate_and_t_b26e() {
    run_m3_arm(M3Arm {
        name: "m3-a4-lua-5-layers-cold",
        children: M3_DEMO_CHILDREN,
        calls: M3_DEMO_CALLS,
        layers: M3_DEMO_CALLS,
        route: Route::CachedCold,
    })
    .await;
}

/// M3 B1 (stress, pass/fail): sixteen edits at once on one 20,000-child
/// layer under `txn`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: b.26e M3 stress arm, 16 concurrent edits of a 20,000-child layer; minutes in a \
            debug build, run by hand (.config/nextest.toml)"]
async fn m3_b1_txn_one_layer_under_stress_stays_within_the_gate_b26e() {
    run_m3_arm(M3Arm {
        name: "m3-b1-txn-1-layer-20k",
        children: M3_STRESS_CHILDREN,
        calls: M3_STRESS_CALLS,
        layers: 1,
        route: Route::Watch,
    })
    .await;
}

/// M3 B2 (stress, pass/fail): sixteen edits at once, each on its own cached
/// 20,000-child layer, under `lua`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: b.26e M3 stress arm, 16 cached 20,000-child layers; minutes in a debug build, \
            run by hand (.config/nextest.toml)"]
async fn m3_b2_lua_sixteen_cached_layers_under_stress_stay_within_the_gate_b26e() {
    run_m3_arm(M3Arm {
        name: "m3-b2-lua-16-layers-warm-20k",
        children: M3_STRESS_CHILDREN,
        calls: M3_STRESS_CALLS,
        layers: M3_STRESS_CALLS,
        route: Route::CachedWarm,
    })
    .await;
}

/// M3 B3 (stress, pass/fail): sixteen edits at once, each on its own
/// uncached 20,000-child layer, under `lua` — every active layer of a
/// replica cold at once, as after a restart — each cold start through the
/// fenced read with its grant inside `T`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: b.26e M3 stress arm, 16 cold 20,000-child layers; minutes in a debug build, \
            run by hand (.config/nextest.toml)"]
async fn m3_b3_lua_sixteen_cold_layers_under_stress_stay_within_the_gate_and_t_b26e() {
    run_m3_arm(M3Arm {
        name: "m3-b3-lua-16-layers-cold-20k",
        children: M3_STRESS_CHILDREN,
        calls: M3_STRESS_CALLS,
        layers: M3_STRESS_CALLS,
        route: Route::CachedCold,
    })
    .await;
}

/// A section id clear of the ids of every base it is used with (their
/// children run from 10 to 20,009).
const FAR_SECTION: u64 = 30_000;

/// The adoption-index report row's delta on [`Base::ChildrenAndEmptySection`]: a
/// `DiffChangeSection` re-key of the root-level section `section_id` (its
/// `section_order` changes, under the same parent and name), which makes it
/// a candidate whose every naming row the adoption index records, and one
/// new root-level section, a birth, which builds that index. On `SECTION`
/// the index records its every child; on [`SECOND_SECTION`], childless,
/// none, on the same base (the AE fit's control).
fn rekey_section_and_birth(section_id: u64) -> OwnedDelta {
    let name = if section_id == SECTION {
        "base"
    } else {
        "second"
    };
    OwnedDelta {
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id,
                section_name: name.to_owned(),
                section_type: 0,
                section_order: 1,
            }),
            create_section(0, FAR_SECTION, "Y".to_owned()),
        ],
        ..Default::default()
    }
}

/// The cached route's plan report rows: `(name, base, delta)`.
fn cached_cold_report_rows() -> Vec<(&'static str, Base, OwnedDelta)> {
    vec![
        // (iii): delete a section holding 5,000 descendants.
        (
            "b26e-delete-5k-descendants",
            Base::Children(DENSE / 4),
            OwnedDelta {
                struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id: 0,
                    section_id: SECTION,
                })],
                ..Default::default()
            },
        ),
        // (iv): one new child under an empty second section of a bare
        // 20,000-child object: no long listing absorbs the depth walk.
        (
            "b26e-one-new-child-beside-20k",
            Base::ChildrenAndEmptySection(DENSE),
            OwnedDelta {
                struct_commands: vec![create_section(
                    SECOND_SECTION,
                    FAR_SECTION,
                    "extra".to_owned(),
                )],
                ..Default::default()
            },
        ),
        // (v): a reset of a 2,000-prim harness-shaped layer to one section.
        (
            "b26e-reset-2000-prims",
            Base::HarnessPrims(2_000),
            OwnedDelta {
                is_diff: true,
                base_version: 0,
                struct_commands: vec![create_section(0, SECTION, "root".to_owned())],
                ..Default::default()
            },
        ),
        // (vi): the re-key of a live 20,000-child section and one birth, and
        // the same delta re-keying a childless section of the same base (the
        // AE fit's control).
        (
            "b26e-rekey-20k-children",
            Base::ChildrenAndEmptySection(DENSE),
            rekey_section_and_birth(SECTION),
        ),
        (
            "b26e-rekey-childless",
            Base::ChildrenAndEmptySection(DENSE),
            rekey_section_and_birth(SECOND_SECTION),
        ),
    ]
}

/// b.26e, report only: each row of
/// [`cached_cold_report_rows`] on the cached route's cold start and with the
/// object cached, phase by phase against the charge in force
/// ([`measure_slow_path_update`]), with its overruns. `B26E_FIT_ONLY` keeps
/// the rows whose name contains it.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e Stage 2 cached-route plan rows"]
async fn cached_route_plan_rows_are_reported_b26e() {
    let only = std::env::var("B26E_FIT_ONLY").unwrap_or_default();
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    for (name, base, delta) in cached_cold_report_rows() {
        if !name.contains(only.as_str()) {
            continue;
        }
        for route in [Route::CachedCold, Route::CachedWarm] {
            let m = measure_slow_path_update(&tv, name, base, &delta, &[], route).await;
            eprintln!(
                "ROW {name} {route:?}: worst {} overruns {:?}",
                worst_phase(&m),
                m.phase_overruns()
            );
        }
    }
}

/// The checked phase (not the cold start's read) with the largest peak
/// against the charge in force, as `label peak/limit (ratio)`.
fn worst_phase(m: &Measured) -> String {
    m.phases
        .iter()
        .zip(phase_limits(&m.phases, m.cache_entry))
        .zip(m.labels())
        .filter(|(_, label)| *label != COLD_START_READ)
        .map(|((p, limit), label)| (label, p.peak, limit))
        .max_by(|a, b| {
            let ratio = |(_, peak, limit): &(&str, u64, u64)| *peak as f64 / (*limit).max(1) as f64;
            ratio(a).total_cmp(&ratio(b))
        })
        .map_or_else(String::new, |(label, peak, limit)| {
            format!(
                "{label} {peak}/{limit} ({:.3})",
                peak as f64 / limit.max(1) as f64
            )
        })
}

/// The rows of [`Base::ExternalRows`] the external-rewrite edit runs on: at least
/// 20,000 distinct hashes in `obj:ext`.
const EXTERNAL_ROWS: usize = 20_000;

/// b.26e: on `txn`, a fast-shaped edit that rewrites one
/// external field to a new external value, on an object whose `obj:ext`
/// holds 20,000 distinct hashes, stays within its charge in every phase. The
/// commit's external-value pre-image read fetches only the rewritten row's
/// columns and the counts of the hashes it holds or receives, never the
/// whole index, so its replies are bounded by the commit's own rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn slow_path_external_rewrite_on_a_20k_hash_index_stays_within_its_charge_b26e() {
    let new_value = vec![0xa5; NAMED_EXT_LEN];
    assert_slow_path_update_within_charge_on(
        "b26e-external-rewrite-20k-hashes",
        Base::ExternalRows(EXTERNAL_ROWS),
        OwnedDelta {
            set_fields: vec![external_field("x0".to_owned(), &new_value, 1)],
            ..Default::default()
        },
        &[&new_value],
        &[Route::Watch],
    )
    .await;
}

/// The `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS` the attribution runs below
/// also measure the WATCH path at, besides the default 37,554. The item
/// ceiling sets the fan-out window of the whole-object read that path loads
/// its state with (`config::snapshot_fanout_window`: ⌊ceiling ÷ 5.266⌋
/// commands in flight at once, capped at `config::FANOUT_WINDOW_MAX`): 1,000
/// here instead of the default's 2,141.
const LOW_ITEM_CEILING: u64 = 5_266;

/// The rewrite the delta engine plans for `delta` on `base`'s state, built
/// from its `CreateLayer` delta: the `RewriteCount` the object-state charge's
/// rewrite term multiplies (report only, for the M1 fit).
fn planned_rewrite(base: Base, delta: &OwnedDelta) -> RewriteCount {
    let mut state = ObjectState::default();
    apply_delta(&mut state, Some(base_delta(base))).expect("the base delta applies");
    let unbounded = MutationAllowance {
        bytes: u64::MAX,
        row_bytes: 1,
        list_entry_bytes: 1,
        index_entry_bytes: 1,
        adoption_entry_bytes: 1,
        walk_entry_bytes: 1,
    };
    // An unbounded allowance never asks its budget for more.
    match apply_delta_bounded(
        &mut state,
        Some(delta.clone()),
        unbounded,
        &mut |_: RewriteCount| None::<u64>,
        &mut ApplyDeltaStats::default(),
    )
    .expect("the edit applies to its base")
    {
        BoundedPlan::Within { rewrite, .. } => rewrite,
        BoundedPlan::Overflow { needed } => needed,
    }
}

/// b.26e attribution (M0) and the object-state fit's inputs (M1), report
/// only: one small structural edit on `base`, measured phase by phase on each
/// `UpdateSlow` route — the cached route with the object cached, its cold
/// start, and the WATCH path under `txn` at the default item ceiling and at
/// [`LOW_ITEM_CEILING`] — printing each phase's peak against the charge in
/// force, and the bytes each call left live when it returned (on the cold
/// start, its cache entry among them). Then one `M1` line per route with what
/// the object-state terms multiply: the cache's weight after the warm call,
/// the fan-out window, the snapshot tallies (`items`, `reply_bytes`,
/// `entries`) and the rewrite the engine plans ([`planned_rewrite`]). Each
/// route edits its own object, but every line carries the one set of tallies
/// read from the WATCH row's object (one unmeasured structural-update read
/// after the calls, so of its state with the edit). The CachedWarm row's
/// `weight` is its own object's cache entry, which also holds
/// [`WARM_SECTION`], the root section the unmeasured warm-up commit creates:
/// one section more than the WATCH object the tallies describe, negligible
/// at thousands of children but about 5 % of the items on an empty base.
/// Asserts nothing new:
/// the per-phase envelope on these bases is what the charge points to come
/// must make true.
async fn report_slow_path_edit_phases(name: &'static str, base: Base, delta: OwnedDelta) {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let fixture = || Fixture::new(name, Feeds::Mixed, delta.clone(), &[]).on_base(base);
    let mut rows = Vec::new();
    let mut warm_weight = 0;
    for route in [Route::CachedWarm, Route::CachedCold, Route::Watch] {
        warm_up(
            &tv,
            Path::UpdateSlow,
            route,
            &format!("{name}-{route:?}-warmup"),
        )
        .await;
        let label = format!("{name}-{route:?}");
        let svc = service_for(&tv, route);
        let store = Arc::clone(&svc.layer_store);
        let m = measure_on(svc, &tv, Path::UpdateSlow, route, &fixture(), &label).await;
        if route == Route::CachedWarm {
            warm_weight = store.structural_cache().weight_bytes();
        }
        rows.push((format!("{route:?}"), m));
    }
    // The WATCH path again, through a store whose own client has the lower
    // item ceiling, warmed through that client first.
    let client = connect_low_ceiling_client().await;
    let structural_warmup = Fixture::new(
        "warmup-structural",
        Feeds::Mixed,
        OwnedDelta {
            struct_commands: children(1, false),
            ..Default::default()
        },
        &[],
    );
    for warmup in [Fixture::header_only(), structural_warmup] {
        let label = format!("{name}-low-ceiling-{}", warmup.name);
        let svc = service_on(&tv, Route::Watch, client.clone());
        measure_on(svc, &tv, Path::UpdateSlow, Route::Watch, &warmup, &label).await;
    }
    let svc = service_on(&tv, Route::Watch, client);
    let label = format!("{name}-Watch-low-ceiling");
    let m = measure_on(svc, &tv, Path::UpdateSlow, Route::Watch, &fixture(), &label).await;
    rows.push((format!("Watch@{LOW_ITEM_CEILING}"), m));

    eprintln!("== b.26e M0 {name} on {base:?}: per-phase peaks (B) / charge in force ==");
    eprintln!(
        "{:<12} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "route",
        "pre-admit",
        "receive",
        "parse",
        "processing",
        "proc-charge",
        "released",
        "residual",
        "high-water"
    );
    for (route, m) in &rows {
        eprintln!(
            "{:<12} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            route,
            m.phase_peak("pre-admission"),
            m.phase_peak("receive"),
            m.phase_peak("parse"),
            m.phase_peak("processing"),
            m.processing_target,
            m.phase_peak("released"),
            m.residual,
            m.charge
        );
        eprintln!("  phases: {}", m.phase_summary());
    }

    let watched = rows
        .iter()
        .find(|(route, _)| route == "Watch")
        .map(|(_, m)| m.object_id)
        .expect("the WATCH route was measured");
    let read = Box::pin(read_object_snapshot_slim(
        tv.client(),
        tv.key_schema(),
        &watched.to_string(),
        SlimSnapshotReadConsumer::StructuralUpdate,
    ))
    .await
    .expect("the edited object reads");
    let SnapshotReadOf::Snapshot(snapshot) = read else {
        panic!("{name}: the structural-update read is never answered TOO_LARGE");
    };
    let (items, reply_bytes, entries) = (
        snapshot.item_count,
        snapshot.reply_bytes,
        snapshot.entry_count,
    );
    drop(snapshot);
    let rewrite = planned_rewrite(base, &delta);
    for (route, m) in &rows {
        let (weight, window) = match route.as_str() {
            "CachedWarm" => (warm_weight, 0),
            "Watch" => (
                0,
                snapshot_fanout_window(common::test_config().snapshot_atomic_read_max_items),
            ),
            "CachedCold" => (0, 0),
            _ => (0, snapshot_fanout_window(LOW_ITEM_CEILING)),
        };
        eprintln!(
            "M1 {name} {route}: proc_peak {} proc_charge {} proc_held_max {} body {} delta {} \
             elements {} reorder {} weight {weight} window {window} items {items} reply_bytes \
             {reply_bytes} entries {entries} rewrite_rows {} rewrite_list {} residual {} \
             cache_entry {} wall_s {:.3}",
            m.phase_peak("processing"),
            m.processing_target,
            m.phase_held_max("processing"),
            m.body,
            m.delta,
            m.elements,
            m.reorder,
            rewrite.rows,
            rewrite.list_entries,
            m.residual,
            m.cache_entry,
            m.wall.as_secs_f64()
        );
    }
}

/// A client on `TEST_VALKEY_URL` with [`LOW_ITEM_CEILING`] as its item
/// ceiling, for a store under a [`TestValkey`]'s key prefix ([`service_on`]).
async fn connect_low_ceiling_client() -> live_edit::storage::client::ValkeyClient {
    let mut config = common::test_config();
    config.snapshot_atomic_read_max_items = LOW_ITEM_CEILING;
    connect_test_client(&config)
        .await
        .expect("a client with its own item ceiling connects")
}

/// b.26e M0: one new child on an empty section.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M0 per-phase attribution"]
async fn slow_path_one_new_child_on_empty_base_phases_are_reported_b26e() {
    report_slow_path_edit_phases("m0-one-new-child-0", Base::Children(0), one_new_child(0)).await;
}

/// b.26e M0: one new child on 1,000 empty children.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M0 per-phase attribution"]
async fn slow_path_one_new_child_on_1k_base_phases_are_reported_b26e() {
    report_slow_path_edit_phases(
        "m0-one-new-child-1k",
        Base::Children(1_000),
        one_new_child(1_000),
    )
    .await;
}

/// b.26e M0: one new child on 5,000 empty children.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M0 per-phase attribution"]
async fn slow_path_one_new_child_on_5k_base_phases_are_reported_b26e() {
    report_slow_path_edit_phases(
        "m0-one-new-child-5k",
        Base::Children(5_000),
        one_new_child(5_000),
    )
    .await;
}

/// b.26e M0: one new child on 20,000 empty children.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M0 per-phase attribution"]
async fn slow_path_one_new_child_on_20k_base_phases_are_reported_b26e() {
    report_slow_path_edit_phases(
        "m0-one-new-child-20k",
        Base::Children(DENSE),
        one_new_child(DENSE),
    )
    .await;
}

/// b.26e M0: one new prim under the zone of the harness-shaped 5,003-section
/// layer, fields included.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M0 per-phase attribution"]
async fn slow_path_one_new_prim_on_harness_base_phases_are_reported_b26e() {
    report_slow_path_edit_phases(
        "m0-one-new-prim-harness",
        Base::Harness,
        one_new_prim_in_harness_zone(),
    )
    .await;
}

/// b.26e M1, a measurement, not a regression check: moving the first of
/// 20,000 children to the end, which re-keys every child. Reports each
/// route's phases and its `M1` line ([`report_slow_path_edit_phases`]), the
/// input the `rewrite_row` term is fitted from; asserts nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M1 object-state fit"]
async fn slow_path_move_first_child_to_end_on_20k_base_phases_are_reported_b26e() {
    report_slow_path_edit_phases(
        "m1-move-first-to-end-20k",
        Base::Children(DENSE),
        reorder_of(vec![11, 10]),
    )
    .await;
}

/// b.26e M1, a measurement, not a regression check: one new prim under the
/// zone of the harness shape at 250 prims (1,003 sections), the object-state
/// fit's second content shape at its small end. Reports each route's phases
/// and its `M1` line ([`report_slow_path_edit_phases`]); asserts nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M1 object-state fit"]
async fn slow_path_one_new_prim_on_250_prim_harness_base_phases_are_reported_b26e() {
    report_slow_path_edit_phases(
        "m1-one-new-prim-harness-250",
        Base::HarnessPrims(250),
        one_new_prim_in_harness_zone(),
    )
    .await;
}

/// b.26e M1, a measurement, not a regression check: one new prim under the
/// zone of the harness shape at 2,500 prims (10,003 sections), the
/// object-state fit's second content shape at its large end. Reports each
/// route's phases and its `M1` line ([`report_slow_path_edit_phases`]);
/// asserts nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M1 object-state fit"]
async fn slow_path_one_new_prim_on_2500_prim_harness_base_phases_are_reported_b26e() {
    report_slow_path_edit_phases(
        "m1-one-new-prim-harness-2500",
        Base::HarnessPrims(2_500),
        one_new_prim_in_harness_zone(),
    )
    .await;
}

/// b.26e M1 (b.rnt), a measurement, not a regression check: the
/// long-reorder-list fixture alone on the WATCH path, run repeatedly (for
/// instance with nextest's `--stress-count`) to see whether its peak varies
/// from run to run, as it once took one of two values. Reports the fixture's
/// peak, phases and netted reorder figure; the header-only upload runs first,
/// since that figure is net of its peak (`F`). It goes through [`run_path_where`],
/// so the per-phase envelope is asserted too, but that check belongs to the
/// live [`update_layer_slow_path_watch_all_uploads_stay_within_their_charge`],
/// which measures the same fixture after others. Alone, the fixture peaked
/// higher than after the full WATCH set's earlier fixtures (M1: 275.5
/// against 251.8 MB, each the same in every run), so a figure from one
/// context does not stand for the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M1 long-reorder peak repeats"]
async fn long_reorder_list_on_the_watch_path_is_reported_b26e() {
    run_path_where(Path::UpdateSlow, Route::Watch, |f| {
        f.feeds == Feeds::Fixed || f.name == "long-reorder-list"
    })
    .await;
}

// ─────────────────────────────────────────────────────────────────────────────
// b.26e Stage 2 fits (M0b, M1): a commit's object-state segments
// ─────────────────────────────────────────────────────────────────────────────
//
// The handler-level phases above split a call only where the gate's charge
// changes, and price each need at the production constants, so they show
// whether a charge covers, not what each count costs. The fits need the
// counts each charge point reports: [`NeedRecorder`] is a `CommitCharge` of
// the test's own, passed straight to `LayerStore::update_layer_with_summary`,
// that records every need a charge point prices and splits the measured
// commit into one segment per charge point, each with its heap peak above
// the live bytes when the commit began. It never refuses, and answers `held`
// 0 so every charge point (grow-only ones too) reaches `try_hold`. Its
// rewrite allowance is the production one (`object_state_rewrite_allowance`,
// at the production prices of rows, list entries, name-index sections and
// adoption-index ids) on what it has granted so far: zero before the plan's
// first ask, so a plan's build sites ask before they allocate, each ask is
// granted and recorded, and the segments split at the projections — the
// fits see the peak before a projection apart from the one after it, as a
// production plan's growth sees them (b.26e Stage 2).

/// Room for every charge point of one commit, so the recorder never grows
/// its log inside the window it measures.
const MAX_SEGMENTS: usize = 4_096;

/// One stretch of a measured commit between two of its charge points.
#[derive(Clone, Copy, Debug)]
struct Segment {
    /// The need held during it: the one the charge point that opened it
    /// priced; `None` before the first charge point.
    need: Option<ObjectStateNeed>,
    /// The heap growth peak during it, above the live bytes when the commit
    /// began.
    peak: u64,
}

struct SegmentLog {
    baseline: usize,
    /// The need the last `target` call priced, and its production price: the
    /// next `try_hold`'s.
    priced: Option<(ObjectStateNeed, u64)>,
    /// The need held by the segment now open.
    open: Option<ObjectStateNeed>,
    /// The largest production price granted so far: what a never-refusing
    /// production charge would hold, which the plan's rewrite allowance is
    /// answered from.
    granted: u64,
    segments: Vec<Segment>,
}

/// Whether [`NeedRecorder`] reports each segment's allocations live at its
/// peak as it closes it (`B26E_TRACK_SEGMENTS`, [`report_commit_segments`]).
static SEGMENT_TRACKING: AtomicBool = AtomicBool::new(false);

/// The peak-time view of segment `segment`, just closed: what was live at
/// its peak ([`report_tracked_at_peak`], growth above the commit's start),
/// then the peak restarts at the live bytes now, as the segment's own does.
fn report_segment_at_peak(segment: usize) {
    report_tracked_at_peak(&format!("segment #{segment}, at its peak"), 12);
    let _guard = TrackerGuard::enter();
    TRACK_PEAK_LIVE.store(LIVE.load(Ordering::SeqCst), Ordering::SeqCst);
    TRACK_PEAK_EVENT.store(TRACK_EVENT.load(Ordering::SeqCst), Ordering::SeqCst);
}

/// The live bytes when the last [`NeedRecorder`] commit began: what a split
/// of its segments ([`split_segment`]) is reported above.
static RECORDER_BASELINE: AtomicUsize = AtomicUsize::new(0);

/// The most splits one measured commit records ([`split_segment`]).
#[cfg(feature = "test-fault-injection")]
const MAX_SPLITS: usize = 8;
/// Splits recorded since [`begin_splits`].
#[cfg(feature = "test-fault-injection")]
static SPLIT_N: AtomicUsize = AtomicUsize::new(0);
/// Each split's [`PEAK`] and [`LIVE`], absolute, as it was taken.
#[cfg(feature = "test-fault-injection")]
static SPLIT_PEAK: [AtomicUsize; MAX_SPLITS] = [const { AtomicUsize::new(0) }; MAX_SPLITS];
#[cfg(feature = "test-fault-injection")]
static SPLIT_LIVE: [AtomicUsize; MAX_SPLITS] = [const { AtomicUsize::new(0) }; MAX_SPLITS];

/// Forget the splits of the last measured commit.
#[cfg(feature = "test-fault-injection")]
fn begin_splits() {
    SPLIT_N.store(0, Ordering::SeqCst);
}

/// Split the open segment of a [`NeedRecorder`] commit where a test hook
/// runs, without a charge point: record the peak so far and the live bytes,
/// and restart the peak at the live bytes now, so the segment's own peak
/// (taken at its next charge point or the commit's end) covers only what
/// runs after the split. Allocates nothing.
#[cfg(feature = "test-fault-injection")]
fn split_segment() {
    let n = SPLIT_N.fetch_add(1, Ordering::SeqCst);
    if n < MAX_SPLITS {
        SPLIT_PEAK[n].store(PEAK.load(Ordering::SeqCst), Ordering::SeqCst);
        SPLIT_LIVE[n].store(LIVE.load(Ordering::SeqCst), Ordering::SeqCst);
    }
    PEAK.store(LIVE.load(Ordering::SeqCst), Ordering::SeqCst);
}

/// Each split since [`begin_splits`]: its peak and live bytes above the
/// recorder's baseline.
#[cfg(feature = "test-fault-injection")]
fn splits() -> Vec<(u64, u64)> {
    let baseline = RECORDER_BASELINE.load(Ordering::SeqCst);
    let n = SPLIT_N.load(Ordering::SeqCst).min(MAX_SPLITS);
    (0..n)
        .map(|i| {
            (
                SPLIT_PEAK[i]
                    .load(Ordering::SeqCst)
                    .saturating_sub(baseline) as u64,
                SPLIT_LIVE[i]
                    .load(Ordering::SeqCst)
                    .saturating_sub(baseline) as u64,
            )
        })
        .collect()
}

/// The fits' `CommitCharge` (module section above).
struct NeedRecorder(std::sync::Mutex<SegmentLog>);

impl NeedRecorder {
    fn new() -> Self {
        Self(std::sync::Mutex::new(SegmentLog {
            baseline: 0,
            priced: None,
            open: None,
            granted: 0,
            segments: Vec::with_capacity(MAX_SEGMENTS),
        }))
    }

    fn log(&self) -> std::sync::MutexGuard<'_, SegmentLog> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Open the commit's window and its first segment.
    fn begin(&self) {
        let baseline = begin_window();
        RECORDER_BASELINE.store(baseline, Ordering::SeqCst);
        let mut log = self.log();
        log.baseline = baseline;
        log.priced = None;
        log.open = None;
        log.granted = 0;
        log.segments.clear();
    }

    /// Close the window and the last segment: every segment, and the live
    /// bytes the commit left above its baseline.
    fn finish(&self) -> (Vec<Segment>, u64) {
        let mut log = self.log();
        let tail = end_window(log.baseline);
        let residual = LIVE.load(Ordering::SeqCst).saturating_sub(log.baseline) as u64;
        let open = log.open;
        log.segments.push(Segment {
            need: open,
            peak: tail,
        });
        assert!(
            log.segments.len() <= MAX_SEGMENTS,
            "fixture self-check: {} segments, past the log's {MAX_SEGMENTS}",
            log.segments.len()
        );
        (std::mem::take(&mut log.segments), residual)
    }
}

#[async_trait::async_trait]
impl CommitCharge for NeedRecorder {
    fn held(&self) -> u64 {
        0
    }

    /// Records the need and its production price, and answers 1, above the
    /// `held` of 0, so the charge point reaches `try_hold` and splits.
    fn target(&self, need: &ObjectStateNeed) -> u64 {
        self.log().priced = Some((*need, object_state_charge_for_test(need)));
        1
    }

    /// The production allowance on what has been granted so far, at the
    /// production prices — zero before the first ask beyond the need's other
    /// terms.
    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance_for_test(self.log().granted, need)
    }

    fn try_hold(&mut self, _target: u64) -> bool {
        let closed = {
            let mut log = self.log();
            let peak = PEAK.load(Ordering::SeqCst).saturating_sub(log.baseline) as u64;
            let open = log.open;
            log.segments.push(Segment { need: open, peak });
            log.open = log.priced.map(|(need, _)| need);
            if let Some((_, price)) = log.priced {
                log.granted = log.granted.max(price);
            }
            PEAK.store(LIVE.load(Ordering::SeqCst), Ordering::SeqCst);
            log.segments.len() - 1
        };
        if SEGMENT_TRACKING.load(Ordering::SeqCst) {
            report_segment_at_peak(closed);
        }
        true
    }

    async fn hold(&mut self, _target: u64) -> Result<(), ServiceError> {
        panic!("the need recorder never refuses a charge, so nothing waits on it")
    }
}

/// [`Base`]-like fixture of the long-name fit (`W_NAME_BYTE`):
/// `NAMED_ENTRIES` child sections of `SECTION`, ids from 10, each named
/// `name_len` bytes, and as many fields, one on each child, keyed `name_len`
/// bytes, each carrying an external value.
const NAMED_ENTRIES: usize = 2_141;

/// The bytes of each named fixture's external value.
const NAMED_EXT_LEN: usize = 64;

/// `prefix` padded with `n` to `len` bytes.
fn padded_name(prefix: String, len: usize) -> String {
    let pad = len.saturating_sub(prefix.len());
    prefix + "n".repeat(pad).as_str()
}

/// The named fixture's `CreateLayer` delta and its external values.
fn named_base(name_len: usize) -> (OwnedDelta, Vec<Vec<u8>>) {
    let values = blobs(NAMED_ENTRIES, NAMED_EXT_LEN, 0x5a);
    let mut struct_commands = vec![create_section(0, SECTION, "base".to_owned())];
    let mut set_fields = Vec::with_capacity(NAMED_ENTRIES);
    for (i, value) in values.iter().enumerate() {
        let id = 10 + i as u64;
        struct_commands.push(create_section(
            SECTION,
            id,
            padded_name(format!("s{i:06}"), name_len),
        ));
        let mut field = external_field(
            padded_name(format!("k{i:06}"), name_len),
            value,
            i as u32 + 1,
        );
        field.section_id = id;
        set_fields.push(field);
    }
    (
        OwnedDelta {
            struct_commands,
            set_fields,
            ..Default::default()
        },
        values,
    )
}

/// The `CreateLayer` delta of the window-bytes arm's base (M0b): `count`
/// child sections of `SECTION`, ids from 10, each named `name_len` bytes, and
/// nothing else. Two such bases that differ only in `name_len` read the same
/// commands in the same windows; only the section rows' reply bytes and the
/// names the read keeps differ.
fn named_sections_base(count: usize, name_len: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: std::iter::once(create_section(0, SECTION, "base".to_owned()))
            .chain((0..count).map(|i| {
                create_section(
                    SECTION,
                    10 + i as u64,
                    padded_name(format!("c{i:06}"), name_len),
                )
            }))
            .collect(),
        ..Default::default()
    }
}

/// A base of the fits: one of [`Base`]'s, the named fixture, or the
/// window-bytes arm's named sections.
#[derive(Clone, Copy, Debug)]
enum FitBase {
    Plain(Base),
    /// [`named_base`] with names of this many bytes.
    Named(usize),
    /// [`named_sections_base`]: this many sections, names of this many
    /// bytes.
    NamedSections(usize, usize),
}

/// A layer whose state is `base`, created through the handler.
async fn fit_base_layer(tv: &TestValkey, label: &str, base: FitBase) -> (String, u64) {
    let delta = match base {
        FitBase::Plain(base) => return base_layer(tv, label, base).await,
        FitBase::Named(name_len) => return base_layer(tv, label, Base::Named(name_len)).await,
        FitBase::NamedSections(count, name_len) => named_sections_base(count, name_len),
    };
    let session_id = new_session(tv, label).await;
    let setup = build_service(tv, Arc::new(AllowAllPermissions));
    let ompf = common::owned_delta::delta_to_ompf_bytes(&delta, &[]);
    let messages = std::iter::once(make_create_header(&session_id, "layer"))
        .chain(ompf.chunks(CHUNK).map(|c| make_create_data(c.to_vec())))
        .collect();
    let response = Box::pin(handle_create_layer(
        &setup,
        make_request(streaming_create(messages)),
    ))
    .await
    .expect("named base layer creation");
    (session_id, response.into_inner().object_id)
}

/// The route of one fit measurement: an `UpdateSlow` route, and on WATCH the
/// fan-out window ([`LOW_ITEM_CEILING`] gives 1,000 commands).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FitRoute {
    CachedWarm,
    CachedCold,
    Watch,
    WatchLow,
}

/// A store for `route` on `client`, recording into [`shared_metrics`].
fn fit_store(
    tv: &TestValkey,
    route: FitRoute,
    client: live_edit::storage::client::ValkeyClient,
) -> Arc<LayerStore> {
    let backend = match route {
        FitRoute::CachedWarm | FitRoute::CachedCold => UpdateLayerBackend::Lua,
        FitRoute::Watch | FitRoute::WatchLow => UpdateLayerBackend::Txn,
    };
    let store = LayerStore::new(client, tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
        .with_metrics(shared_metrics());
    // A memory measurement: no shadow copy of the planned entry.
    store.structural_cache().set_shadow_check_for_test(false);
    Arc::new(store)
}

/// M1e's reply-byte input: one unmeasured read of `object` with the cold
/// start's consumer and reply shape (`structural_cache`, whole rows; the
/// tallies are the same on every read path), after the measured
/// commit, printed as a `COLDREAD` line with its item, reply-byte and entry
/// tallies. The commit added one section, so the figures are the cold
/// start's read plus that one row.
async fn report_cold_read_tallies(tv: &TestValkey, name: &str, object: u64) {
    // Unbounded ceilings (as the structural cache's measurement tool reads
    // its tallies), so no fit base is answered `TOO_LARGE`.
    let unbounded = 1 << 40;
    let read = Box::pin(read_object_snapshot_slim_with_ceilings(
        tv.client(),
        tv.key_schema(),
        &object.to_string(),
        SlimSnapshotReadConsumer::StructuralCache,
        SnapshotCeilings {
            max_items: unbounded,
            max_bytes: unbounded,
        },
    ))
    .await
    .expect("the edited object reads");
    let SnapshotReadOf::Snapshot(snapshot) = read else {
        panic!("{name}: an unbounded read is never answered TOO_LARGE");
    };
    eprintln!(
        "COLDREAD {name}: items {} reply_bytes {} entries {}",
        snapshot.item_count, snapshot.reply_bytes, snapshot.entry_count
    );
}

/// One structural commit of `delta` through `store`, with the commit's
/// segments measured ([`NeedRecorder`]).
async fn measure_commit(
    store: &LayerStore,
    session_id: &str,
    object_id: u64,
    delta: &OwnedDelta,
    label: &str,
) -> (Vec<Segment>, u64, std::time::Duration) {
    let relay = Bytes::from(common::owned_delta::delta_to_ompf_bytes(delta, &[]));
    let delta = Arc::new(Some(delta.clone()));
    let mut recorder = NeedRecorder::new();
    recorder.begin();
    let started = std::time::Instant::now();
    let (result, _summary) = Box::pin(store.update_layer_with_summary(
        session_id,
        "layer",
        object_id,
        delta,
        Some(relay),
        "t7-fit",
        0,
        None,
        &mut recorder,
    ))
    .await;
    let wall = started.elapsed();
    let (segments, residual) = recorder.finish();
    result.unwrap_or_else(|e| panic!("{label}: the measured commit must land: {e:?}"));
    (segments, residual, wall)
}

/// The cache warm-up [`Route::CachedWarm`] uses: one unmeasured structural
/// commit through `store`, which loads the object into its cache.
async fn warm_cache(store: &LayerStore, session_id: &str, object_id: u64, label: &str) {
    let warm = OwnedDelta {
        struct_commands: vec![create_section(0, WARM_SECTION, "warm".to_owned())],
        ..Default::default()
    };
    let relay = Bytes::from(common::owned_delta::delta_to_ompf_bytes(&warm, &[]));
    Box::pin(store.update_layer(
        session_id,
        "layer",
        object_id,
        Some(warm),
        Some(relay),
        "t7-warmup",
        0,
    ))
    .await
    .unwrap_or_else(|e| panic!("{label}: the cache warm-up must commit: {e:?}"));
}

/// The fits' report: every segment of one structural edit of `delta` on
/// `base`, on each of `routes`, as `SEG` lines (the need held and the peak
/// during it), and one `COMMIT` line per route with the commit's peak and
/// the segment it fell in. Each route's once-per-process code runs first, in
/// an unmeasured commit on an empty base. Asserts nothing.
async fn report_commit_segments(name: &str, base: FitBase, delta: OwnedDelta, routes: &[FitRoute]) {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let low = connect_low_ceiling_client().await;
    for &route in routes {
        let client = if route == FitRoute::WatchLow {
            low.clone()
        } else {
            tv.client().clone()
        };
        // Once-per-process code of this route, unmeasured.
        let store = fit_store(&tv, route, client.clone());
        let warm_label = format!("{name}-{route:?}-process-warmup");
        let (session, object) = base_layer(&tv, &warm_label, Base::Children(0)).await;
        if route == FitRoute::CachedWarm {
            warm_cache(&store, &session, object, &warm_label).await;
        }
        measure_commit(&store, &session, object, &one_new_child(0), &warm_label).await;

        let label = format!("{name}-{route:?}");
        let store = fit_store(&tv, route, client);
        let (session, object) = fit_base_layer(&tv, &label, base).await;
        if route == FitRoute::CachedWarm {
            warm_cache(&store, &session, object, &label).await;
        }
        // `B26E_TRACK_SEGMENTS=<case>-<route>` (this label): each segment's
        // allocations live at its peak ([`report_segment_at_peak`]).
        let tracked = std::env::var("B26E_TRACK_SEGMENTS").is_ok_and(|v| v == label);
        if tracked {
            begin_peak_tracking();
            SEGMENT_TRACKING.store(true, Ordering::SeqCst);
        }
        let (segments, residual, wall) =
            measure_commit(&store, &session, object, &delta, &label).await;
        if tracked {
            SEGMENT_TRACKING.store(false, Ordering::SeqCst);
            stop_recording();
            report_tracked_at_peak(&format!("{label}: the closing segment, at its peak"), 12);
            end_tracking();
        }
        if route == FitRoute::CachedCold {
            report_cold_read_tallies(&tv, name, object).await;
        }
        let weight = store.structural_cache().weight_bytes();
        for (i, s) in segments.iter().enumerate() {
            eprintln!(
                "SEG {name} {route:?} #{i} peak {} need {:?}",
                s.peak, s.need
            );
        }
        let (at, top) = segments
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.peak)
            .expect("a commit has at least its closing segment");
        eprintln!(
            "COMMIT {name} {route:?}: peak {} at segment {at} of {} need {:?} residual \
             {residual} cache_weight_after {weight} wall_s {:.3}",
            top.peak,
            segments.len(),
            top.need,
            wall.as_secs_f64()
        );
    }
}

const ALL_FIT_ROUTES: &[FitRoute] = &[
    FitRoute::CachedWarm,
    FitRoute::CachedCold,
    FitRoute::Watch,
    FitRoute::WatchLow,
];

/// b.26e Stage 2 fits (M0b, M1): each commit's object-state segments
/// ([`report_commit_segments`]) for one new child on 0 / 1,000 / 5,000 /
/// 20,000 children, one new prim on the harness shape at 250 / 1,250 /
/// 2,500 prims, one new child on the named fixtures (names of 1,024 and of 8
/// bytes), one new child on the window-bytes arm's 5,000 sections named 8
/// and 150 bytes ([`named_sections_base`]: the same commands per window at
/// 2,141 and 1,000, so only reply bytes and retained names differ), and
/// moving the first of 20,000 children to the end; then, on the cached
/// route only, the C3 fits' cases (`c3-*`: two new children on 5,000, 7,168
/// and 20,000 sections and one on 7,168, for `NI`; the re-key of a
/// 20,000-child section with one birth, and the same re-keying a childless
/// section of the same base, for `AE`; one new child under an empty section
/// beside 20,000 children, row (iv), the retired plan-copy term's basis);
/// then, on the cached route only, the in-place re-fit's cases
/// ([`ug2_fit_cases`]). Report only; the
/// cold-start route also prints a `COLDREAD` line, its read's tallies
/// ([`report_cold_read_tallies`]). `B26E_FIT_ONLY` keeps the cases whose name
/// contains it, `B26E_FIT_ROUTES=cached` the cached route's two routes only;
/// `B26E_TRACK_SEGMENTS=<case>-<route>` attributes that
/// commit's segment peaks ([`report_segment_at_peak`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e Stage 2 object-state fit segments"]
async fn object_state_segments_are_reported_b26e() {
    let only = std::env::var("B26E_FIT_ONLY").unwrap_or_default();
    let cases: Vec<(&str, FitBase, OwnedDelta)> = vec![
        (
            "child-0",
            FitBase::Plain(Base::Children(0)),
            one_new_child(0),
        ),
        (
            "child-1k",
            FitBase::Plain(Base::Children(1_000)),
            one_new_child(1_000),
        ),
        (
            "child-5k",
            FitBase::Plain(Base::Children(5_000)),
            one_new_child(5_000),
        ),
        (
            "child-20k",
            FitBase::Plain(Base::Children(DENSE)),
            one_new_child(DENSE),
        ),
        (
            "prim-250",
            FitBase::Plain(Base::HarnessPrims(250)),
            one_new_prim_in_harness_zone(),
        ),
        (
            "prim-1250",
            FitBase::Plain(Base::HarnessPrims(1_250)),
            one_new_prim_in_harness_zone(),
        ),
        (
            "prim-2500",
            FitBase::Plain(Base::HarnessPrims(2_500)),
            one_new_prim_in_harness_zone(),
        ),
        (
            "named-1024",
            FitBase::Named(1_024),
            one_new_child(NAMED_ENTRIES),
        ),
        ("named-8", FitBase::Named(8), one_new_child(NAMED_ENTRIES)),
        (
            "secnames-8",
            FitBase::NamedSections(5_000, 8),
            one_new_child(5_000),
        ),
        (
            "secnames-150",
            FitBase::NamedSections(5_000, 150),
            one_new_child(5_000),
        ),
        (
            "move-first-to-end-20k",
            FitBase::Plain(Base::Children(DENSE)),
            reorder_of(vec![11, 10]),
        ),
    ];
    // `B26E_FIT_ROUTES=cached` keeps the cached route's two only, `=watch`
    // the WATCH path at its 2,141-command window only (the WATCH rewrite
    // prices' own figures, R2-1), and skips the cached-only cases below.
    let routes_var = std::env::var("B26E_FIT_ROUTES").unwrap_or_default();
    let routes = match routes_var.as_str() {
        "cached" => &[FitRoute::CachedWarm, FitRoute::CachedCold][..],
        "watch" => &[FitRoute::Watch][..],
        _ => ALL_FIT_ROUTES,
    };
    for (name, base, delta) in cases {
        if name.contains(only.as_str()) {
            report_commit_segments(name, base, delta, routes).await;
        }
    }
    if routes_var == "watch" {
        // WATCH's per-list-entry figure (R2-1): one new child under a
        // section of 5,000 or 20,000 children against one under an empty
        // section beside it, on the same base, so the read and its retained
        // state are the same and the commit's segments differ by the
        // rewritten list.
        for (size, under, beside) in [
            (5_000, "ug2-watch-under-5k", "ug2-watch-beside-5k"),
            (DENSE, "ug2-watch-under-20k", "ug2-watch-beside-20k"),
        ] {
            for (name, parent, id) in [
                (under, SECTION, 10 + size as u64),
                (beside, SECOND_SECTION, FAR_SECTION),
            ] {
                if name.contains(only.as_str()) {
                    report_commit_segments(
                        name,
                        FitBase::Plain(Base::ChildrenAndEmptySection(size)),
                        OwnedDelta {
                            struct_commands: vec![create_section(parent, id, "extra".to_owned())],
                            ..Default::default()
                        },
                        routes,
                    )
                    .await;
                }
            }
        }
        return;
    }
    // The b.26e C3 fits' cases, on the cached route only, where the plan's
    // projections split a warm commit's segments: two new children less one
    // on the same base, at 5,000, 7,168 sections and 20,000 (the `NI` fit's
    // cases; a plan in place builds no name index, so on this route they now
    // measure a second birth's list entries and, at 7,168, a full table); the
    // re-key of a 20,000-child section and one birth less the same delta
    // re-keying a childless section of the same base (`AE`); one new child
    // beside 20,000 children, under an empty section (row (iv), the retired
    // plan-copy term's basis).
    let c3_cases: Vec<(&str, FitBase, OwnedDelta)> = vec![
        (
            "c3-two-children-5k",
            FitBase::Plain(Base::Children(5_000)),
            new_children(5_000, 2),
        ),
        (
            "c3-child-7168",
            FitBase::Plain(Base::Children(7_167)),
            one_new_child(7_167),
        ),
        (
            "c3-two-children-7168",
            FitBase::Plain(Base::Children(7_167)),
            new_children(7_167, 2),
        ),
        (
            "c3-two-children-20k",
            FitBase::Plain(Base::Children(DENSE)),
            new_children(DENSE, 2),
        ),
        (
            "c3-rekey-20k",
            FitBase::Plain(Base::ChildrenAndEmptySection(DENSE)),
            rekey_section_and_birth(SECTION),
        ),
        (
            "c3-rekey-childless",
            FitBase::Plain(Base::ChildrenAndEmptySection(DENSE)),
            rekey_section_and_birth(SECOND_SECTION),
        ),
        (
            "c3-beside-20k",
            FitBase::Plain(Base::ChildrenAndEmptySection(DENSE)),
            OwnedDelta {
                struct_commands: vec![create_section(
                    SECOND_SECTION,
                    FAR_SECTION,
                    "extra".to_owned(),
                )],
                ..Default::default()
            },
        ),
    ];
    for (name, base, delta) in c3_cases {
        if name.contains(only.as_str()) {
            report_commit_segments(
                name,
                base,
                delta,
                &[FitRoute::CachedWarm, FitRoute::CachedCold],
            )
            .await;
        }
    }
    // The in-place route's re-fit cases (b.ug2 Stage 6b), on the cached
    // route only: the depth check's walks (`rewrite_walk_entry`), a reset, a
    // cascade, and the field- and time-sample-heavy structural edits whose
    // journal records each own element (`element[UpdateSlow]`).
    for (name, base, delta) in ug2_fit_cases() {
        if name.contains(only.as_str()) {
            report_commit_segments(
                name,
                base,
                delta,
                &[FitRoute::CachedWarm, FitRoute::CachedCold],
            )
            .await;
        }
    }
}

/// The chain depth of the walk-up fit case: a new leaf under it is at depth
/// 9,002, inside `MAX_SECTION_TREE_DEPTH` (10,000).
const UG2_CHAIN: usize = 9_000;

/// The in-place route's re-fit cases (b.ug2 Stage 6b), each a name, a base
/// and the measured delta:
///
/// - `ug2-walk-chain-9k`: one new leaf under the bottom of a 9,000-deep
///   chain, whose incremental depth check walks 9,002 hops up;
/// - `ug2-walk-move-20k`: `SECTION`, with 20,000 children, moved under the
///   empty [`SECOND_SECTION`], whose check walks those 20,000 down;
/// - `ug2-reset-20k` / `ug2-reset-prim-2500`: a reset to one section of
///   20,000 children, and of the 2,500-prim harness layer (fields included);
/// - `ug2-delete-20k` / `ug2-delete-fields-20k`: deleting `SECTION`, which
///   cascades to its 20,000 children, or purges its 20,000 field rows;
/// - `ug2-fields-20k` / `ug2-samples-20k`: one new child and 20,000 new
///   fields, or time samples, on `SECTION`;
/// - `ug2-overwrite-fields-20k`: one new child and an overwrite of each of
///   `SECTION`'s 20,000 fields, so the journal records each row's pre-image.
fn ug2_fit_cases() -> Vec<(&'static str, FitBase, OwnedDelta)> {
    let with_child = |set_fields: Vec<SetField>, time_samples: Vec<TimeSample>| OwnedDelta {
        struct_commands: vec![create_section(SECTION, 10, "extra".to_owned())],
        set_fields,
        time_samples,
        ..Default::default()
    };
    let reset = OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![create_section(0, SECTION, "base".to_owned())],
        ..Default::default()
    };
    let delete_section = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: SECTION,
        })],
        ..Default::default()
    };
    vec![
        (
            "ug2-walk-chain-9k",
            FitBase::Plain(Base::Chain(UG2_CHAIN)),
            OwnedDelta {
                struct_commands: vec![create_section(
                    chain_bottom(UG2_CHAIN),
                    FAR_SECTION,
                    "leaf".to_owned(),
                )],
                ..Default::default()
            },
        ),
        (
            "ug2-walk-move-20k",
            FitBase::Plain(Base::ChildrenAndEmptySection(DENSE)),
            OwnedDelta {
                struct_commands: vec![StructCommand::MoveSection(CommandMoveSection {
                    old_parent_id: 0,
                    new_parent_id: SECOND_SECTION,
                    section_id: SECTION,
                    new_name: "base".to_owned(),
                })],
                ..Default::default()
            },
        ),
        (
            "ug2-reset-20k",
            FitBase::Plain(Base::Children(DENSE)),
            reset.clone(),
        ),
        (
            "ug2-reset-prim-2500",
            FitBase::Plain(Base::HarnessPrims(2_500)),
            reset,
        ),
        (
            "ug2-delete-20k",
            FitBase::Plain(Base::Children(DENSE)),
            delete_section.clone(),
        ),
        (
            "ug2-delete-fields-20k",
            FitBase::Plain(Base::Fields(DENSE)),
            delete_section,
        ),
        (
            "ug2-fields-20k",
            FitBase::Plain(Base::Children(0)),
            with_child(small_fields(SECTION, "n", DENSE), Vec::new()),
        ),
        (
            "ug2-samples-20k",
            FitBase::Plain(Base::Children(0)),
            with_child(Vec::new(), (0..DENSE).map(time_sample).collect()),
        ),
        (
            "ug2-overwrite-fields-20k",
            FitBase::Plain(Base::Fields(DENSE)),
            with_child(
                (0..DENSE)
                    .map(|i| field(format!("f{i}"), (i as u64 + 1).to_le_bytes().to_vec()))
                    .collect(),
                Vec::new(),
            ),
        ),
    ]
}

/// b.26e Stage 2: plans per commit under a charge. Each case is
/// one `UpdateLayer` through the handler ([`measure_on`]), so its commit runs
/// under the call's real `UploadCommitCharge` over its gate reservation, with
/// the default budget (384 MiB) and no other upload: every in-commit growth
/// fits, so none should wait, and each commit should plan once — one
/// planning clone on the cached route, one `structural_update` read on WATCH
/// (counted on a client of its own with the registry's hooks installed; the
/// fits' clients carry none, so their reads record no series).
/// The cases include small objects whose plan rewrites many rows (moving the
/// first of 500 or 2,000 children to the end), whose rewrite grows the charge
/// while planning (`ChargeRewriteBudget` extensions) now that WA1's byte
/// window no longer holds a whole window for them. Prints one `PLANS` line per
/// case and route; `B26E_FIT_ONLY` keeps the cases whose name contains it.
/// Report only.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e Stage 2 plans per commit under a charge"]
async fn plans_per_commit_under_a_charge_are_reported_b26e() {
    let only = std::env::var("B26E_FIT_ONLY").unwrap_or_default();
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let cases: Vec<(&'static str, Base, OwnedDelta)> = vec![
        ("plans-child-0", Base::Children(0), one_new_child(0)),
        (
            "plans-child-5k",
            Base::Children(5_000),
            one_new_child(5_000),
        ),
        (
            "plans-prim-250",
            Base::HarnessPrims(250),
            one_new_prim_in_harness_zone(),
        ),
        (
            "plans-move-first-500",
            Base::Children(500),
            reorder_of(vec![11, 10]),
        ),
        (
            "plans-move-first-2k",
            Base::Children(2_000),
            reorder_of(vec![11, 10]),
        ),
        (
            "plans-move-first-20k",
            Base::Children(DENSE),
            reorder_of(vec![11, 10]),
        ),
    ];
    // The stores' own client, with the registry's hooks installed, so its
    // reads record `live_edit_snapshot_read_total` (the WATCH plans).
    let client = connect_test_client(&common::test_config())
        .await
        .expect("the plans client connects");
    client.install_observability_hooks(shared_metrics()).await;
    for route in [Route::CachedWarm, Route::Watch] {
        warm_up(
            &tv,
            Path::UpdateSlow,
            route,
            &format!("plans-{route:?}-warmup"),
        )
        .await;
        // The hooked client's first structural commit on this route, which
        // creates its metric series, unmeasured.
        let warmup = Fixture::new("plans-warmup", Feeds::Mixed, one_new_child(0), &[]);
        let svc = service_on(&tv, route, client.clone());
        let label = format!("plans-{route:?}-client-warmup");
        measure_on(svc, &tv, Path::UpdateSlow, route, &warmup, &label).await;
        for (name, base, delta) in &cases {
            if !name.contains(only.as_str()) {
                continue;
            }
            let fixture = Fixture::new(name, Feeds::Mixed, delta.clone(), &[]).on_base(*base);
            let label = format!("{name}-{route:?}");
            let svc = service_on(&tv, route, client.clone());
            let m = measure_on(svc, &tv, Path::UpdateSlow, route, &fixture, &label).await;
            let p = m.plans;
            let plans = match route {
                Route::CachedWarm => p.in_place_plans as f64,
                _ => p.watch_reads,
            };
            eprintln!(
                "PLANS {name} {route:?}: plans {plans} commits {} plans_per_commit {:.2} attempts \
                 {} recharges {} in_place_plans {} watch_reads {} proc_held_max {} overruns {:?}",
                p.commits,
                plans / p.commits.max(1.0),
                p.attempts,
                p.recharges,
                p.in_place_plans,
                p.watch_reads,
                m.phase_held_max("processing"),
                m.phase_overruns()
            );
        }
    }
}

/// b.ug2 Stage 6b and R2-2: `element[UpdateSlow]` per route, each netted of
/// the route's own object-state need at its peak. The cached route plans in
/// place, so its journal records the first touch of each of the delta's own
/// elements until the commit settles. On [`Route::CachedWarm`],
/// [`Route::CachedCold`] and [`Route::Watch`], after the route's warm-up,
/// measures through the handler the header-only upload and the value-heavy
/// fixture (the chain's `F` and byte terms) and four element-dense
/// structural fixtures — 20,000 new children (structure-dense), one new
/// child with 20,000 new fields or 20,000 time samples on `SECTION`, and one
/// new child overwriting each of `SECTION`'s 20,000 fields (the journal then
/// holds each row's pre-image) — and on WATCH also its run path's own
/// element fixtures. Per fixture it prints an `ELEMENT` line: its element
/// figure netted as [`Fit::of`] nets it (the peak less the header-only peak,
/// its body and the larger of this run's byte figure and the production
/// basis × its delta bytes, per element), the same net of the object-state
/// need at its commit's peak ([`need_at_commit_peak`]) — on the cached route
/// all of it at the production bases ([`need_basis`]), on WATCH its queued
/// commands alone ([`queued_basis`]) — and its phases against the charge in
/// force. `B26E_FIT_ONLY` keeps the fixtures whose name contains it (the
/// header-only and value-heavy ones always run); `UG2_ELEMENT_ROUTES=cached`
/// or `=watch` keeps those routes. Report only.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.ug2 Stage 6b element fit per UpdateSlow route"]
async fn cached_route_element_shapes_are_reported_ug2() {
    let only = std::env::var("B26E_FIT_ONLY").unwrap_or_default();
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let child = || create_section(SECTION, 10 + DENSE as u64, "extra".to_owned());
    let element_fixtures = || {
        vec![
            Fixture::new(
                "structure-dense",
                Feeds::Elements,
                OwnedDelta {
                    struct_commands: children(DENSE, false),
                    ..Default::default()
                },
                &[],
            ),
            Fixture::new(
                "fields-structural",
                Feeds::Elements,
                OwnedDelta {
                    struct_commands: vec![child()],
                    set_fields: small_fields(SECTION, "n", DENSE),
                    ..Default::default()
                },
                &[],
            ),
            Fixture::new(
                "samples-structural",
                Feeds::Elements,
                OwnedDelta {
                    struct_commands: vec![child()],
                    time_samples: (0..DENSE).map(time_sample).collect(),
                    ..Default::default()
                },
                &[],
            ),
            Fixture::new(
                "overwrite-structural",
                Feeds::Elements,
                OwnedDelta {
                    struct_commands: vec![child()],
                    set_fields: (0..DENSE)
                        .map(|i| field(format!("f{i}"), (i as u64 + 1).to_le_bytes().to_vec()))
                        .collect(),
                    ..Default::default()
                },
                &[],
            )
            .on_base(Base::Fields(DENSE)),
        ]
    };
    // `UG2_ELEMENT_ROUTES=cached` or `=watch` keeps those routes only.
    let routes = match std::env::var("UG2_ELEMENT_ROUTES").as_deref() {
        Ok("cached") => vec![Route::CachedWarm, Route::CachedCold],
        Ok("watch") => vec![Route::Watch],
        _ => vec![Route::CachedWarm, Route::CachedCold, Route::Watch],
    };
    for route in routes {
        warm_up(
            &tv,
            Path::UpdateSlow,
            route,
            &format!("ug2-elements-{route:?}-warmup"),
        )
        .await;
        let mut fixtures = vec![
            Fixture::header_only(),
            Fixture::new(
                "value-heavy",
                Feeds::Bytes,
                OwnedDelta {
                    struct_commands: children(1, false),
                    set_fields: value_fields(VALUES, VALUE_LEN),
                    ..Default::default()
                },
                &[],
            ),
        ];
        fixtures.extend(
            element_fixtures()
                .into_iter()
                .filter(|f| f.name.contains(only.as_str())),
        );
        // On WATCH also its run path's own element fixtures (its field- and
        // time-sample-dense ones, which `txn` sends there, and the reset).
        if route == Route::Watch {
            fixtures.extend(
                fixtures_on(Path::UpdateSlow, Route::Watch)
                    .into_iter()
                    .filter(|f| {
                        f.feeds == Feeds::Elements
                            && f.name != "structure-dense"
                            && f.name.contains(only.as_str())
                    }),
            );
        }
        let mut results = Vec::new();
        // Each element fixture's object-state need at its commit's peak, from
        // a `NeedRecorder` commit of the same delta on the same base and
        // route (R2-2): what the call holds beside its processing charge
        // there, which the element figure is netted of.
        let mut needs = Vec::new();
        for fixture in &fixtures {
            let label = format!("ug2-elements-{route:?}-{}", fixture.name);
            results.push(measure_route(&tv, Path::UpdateSlow, route, fixture, &label).await);
            let need = if fixture.feeds == Feeds::Elements {
                Some(need_at_commit_peak(&tv, route, fixture, &format!("{label}-need")).await)
            } else {
                None
            };
            needs.push(need);
        }
        let fit = Fit::of(&results, NetFloor::production(Path::UpdateSlow));
        let fixed_header = results
            .iter()
            .filter(|m| m.feeds == Feeds::Fixed)
            .map(|m| m.peak as f64)
            .fold(0.0, f64::max);
        for (m, need) in results.iter().zip(&needs) {
            let excess = if m.feeds == Feeds::Elements {
                (m.peak as f64 - fixed_header - m.body as f64 - fit.net_byte * m.delta as f64)
                    .max(0.0)
            } else {
                0.0
            };
            let per = |bytes: f64| bytes / m.elements.max(1) as f64;
            let (segment, need) = need.unwrap_or((0, None));
            // The cached route nets its whole object-state need at the
            // production bases; WATCH, as before, its queued commands alone
            // (`queued`'s basis per slot), since `queued` was fitted as the
            // WATCH commit's excess over the element basis.
            let basis = need.as_ref().map_or(0.0, |need| match need {
                ObjectStateNeed::Cached { .. } => need_basis(need),
                ObjectStateNeed::Watch { .. } => queued_basis(need),
            });
            eprintln!(
                "ELEMENT {route:?} {}: peak {} body {} delta {} elements {} netted_element {:.1} \
                 net_of_need {:.1} (header-only {:.0}, net_byte {:.3}; need at the commit's \
                 peak, segment {segment}, basis {basis:.0}: {need:?}) worst_phase {} phases {}",
                m.name,
                m.peak,
                m.body,
                m.delta,
                m.elements,
                per(excess),
                per((excess - basis).max(0.0)),
                fixed_header,
                fit.net_byte,
                worst_phase_ratio(m),
                m.phase_summary()
            );
        }
    }
}

/// The object-state need held at the peak of one commit of `fixture`'s delta
/// on its base, on the cached `route` (warmed as [`Route::CachedWarm`]
/// warms, or cold): the need of the [`NeedRecorder`] segment whose
/// commit-only peak is the commit's highest, with that segment's index. The
/// element fits net a handler-measured peak of the same fixture of it
/// ([`need_basis`]).
async fn need_at_commit_peak(
    tv: &TestValkey,
    route: Route,
    fixture: &Fixture,
    label: &str,
) -> (usize, Option<ObjectStateNeed>) {
    let fit_route = match route {
        Route::CachedWarm => FitRoute::CachedWarm,
        Route::CachedCold => FitRoute::CachedCold,
        Route::Watch => FitRoute::Watch,
        Route::Default => {
            panic!("fixture self-check: {label}: the need pairing is for an UpdateSlow route")
        }
    };
    let delta = fixture
        .delta
        .as_ref()
        .expect("fixture self-check: an element fixture carries a delta");
    let store = fit_store(tv, fit_route, tv.client().clone());
    let (session, object) = base_layer(tv, label, fixture.base).await;
    if route == Route::CachedWarm {
        warm_cache(&store, &session, object, label).await;
    }
    let (segments, _, _) = measure_commit(&store, &session, object, delta, label).await;
    let (at, top) = segments
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| s.peak)
        .expect("a commit has at least its closing segment");
    (at, top.need)
}

/// `need`'s object-state term at the production bases (R2-2): its parts
/// priced 1:1 — the held entry, its growth and the rollback rebuilds — at
/// their bytes, and its rewrite counts at each production price ÷
/// [`HEADROOM`]. Read through the production pricing
/// (`object_state_charge_for_test`): the need priced with only its rebuild
/// bytes as its rewrite is the 1:1 part, and the rest is the priced rewrite.
fn need_basis(need: &ObjectStateNeed) -> f64 {
    let rewrite = need.rewrite();
    let one_to_one = object_state_charge_for_test(&need.with_rewrite(RewriteCount {
        rollback_rebuild_bytes: rewrite.rollback_rebuild_bytes,
        ..RewriteCount::default()
    }));
    let priced = object_state_charge_for_test(need);
    one_to_one as f64 + priced.saturating_sub(one_to_one) as f64 / HEADROOM
}

/// A WATCH `need`'s queued-command term at its production basis
/// (`QUEUED_COMMAND_CHARGE_BYTES` ÷ [`HEADROOM`] per slot), read through the
/// production pricing as the need less the same need with no queued
/// commands; 0 for a cached need.
fn queued_basis(need: &ObjectStateNeed) -> f64 {
    let ObjectStateNeed::Watch {
        snapshot,
        fixed,
        rewrite,
        ..
    } = *need
    else {
        return 0.0;
    };
    let without = object_state_charge_for_test(&ObjectStateNeed::Watch {
        snapshot,
        fixed,
        rewrite,
        queued: live_edit::storage::commit_charge::QueuedCommands::default(),
    });
    object_state_charge_for_test(need).saturating_sub(without) as f64 / HEADROOM
}

/// The element fixtures' sizes [`fast_path_element_sizes_are_reported_ug2`]
/// measures: the run path's ([`DENSE`]) and a second, larger one, so a
/// per-element figure that moves between them by more than 5 % reads as a
/// buffer's step rather than a per-element cost.
const FAST_ELEMENT_SIZES: [usize; 2] = [DENSE, 30_000];

/// b.ug2 R2-5: `element[UpdateFast]` at two sizes. On the fast path, after
/// its warm-up, measures through the handler the header-only upload and the
/// value-heavy fixture (the chain's `F` and byte terms), then the field-dense
/// and time-sample-dense fixtures at each of [`FAST_ELEMENT_SIZES`], and
/// prints per fixture an `ELEMENTFAST` line with its element figure netted as
/// [`Fit::of`] nets it (peak less the header-only peak, its body and the
/// larger of this run's byte figure and the production basis × its delta
/// bytes, per element) and its worst phase against the charge in force.
/// Report only.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.ug2 R2-5 fast-path element fit at two sizes"]
async fn fast_path_element_sizes_are_reported_ug2() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    warm_up(&tv, Path::UpdateFast, Route::Default, "ug2-fast-warmup").await;
    let mut fixtures = vec![
        Fixture::header_only(),
        Fixture::new(
            "value-heavy",
            Feeds::Bytes,
            OwnedDelta {
                set_fields: value_fields(VALUES, VALUE_LEN),
                ..Default::default()
            },
            &[],
        ),
    ];
    for n in FAST_ELEMENT_SIZES {
        fixtures.push(Fixture::new(
            if n == DENSE {
                "field-dense-20k"
            } else {
                "field-dense-30k"
            },
            Feeds::Elements,
            OwnedDelta {
                set_fields: (0..n)
                    .map(|i| field(format!("f{i}"), (i as u64).to_le_bytes().to_vec()))
                    .collect(),
                ..Default::default()
            },
            &[],
        ));
        fixtures.push(Fixture::new(
            if n == DENSE {
                "time-sample-dense-20k"
            } else {
                "time-sample-dense-30k"
            },
            Feeds::Elements,
            OwnedDelta {
                time_samples: (0..n).map(time_sample).collect(),
                ..Default::default()
            },
            &[],
        ));
    }
    let mut results = Vec::new();
    for fixture in &fixtures {
        let label = format!("ug2-fast-{}", fixture.name);
        results.push(measure_route(&tv, Path::UpdateFast, Route::Default, fixture, &label).await);
    }
    let fit = Fit::of(&results, NetFloor::production(Path::UpdateFast));
    for m in &results {
        let element = if m.feeds == Feeds::Elements {
            (m.peak as f64 - fit.fixed_header - m.body as f64 - fit.net_byte * m.delta as f64)
                .max(0.0)
                / m.elements.max(1) as f64
        } else {
            0.0
        };
        eprintln!(
            "ELEMENTFAST {}: peak {} body {} delta {} elements {} netted_element {element:.1} \
             (header-only {:.0}, net_byte {:.4}) worst_phase {}",
            m.name,
            m.peak,
            m.body,
            m.delta,
            m.elements,
            fit.fixed_header,
            fit.net_byte,
            worst_phase_ratio(m),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// b.ug2 Stage 6b: the depth check's walks, measured alone
// ─────────────────────────────────────────────────────────────────────────────

/// The heap growth peak of `f`, above the live bytes when it began.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let baseline = begin_window();
    let out = f();
    (out, end_window(baseline))
}

/// The compact state `delta` creates from nothing.
fn compact_of(delta: OwnedDelta) -> live_edit::domain::types::CompactState {
    let mut state = ObjectState::default();
    apply_delta(&mut state, Some(delta)).expect("the base applies");
    live_edit::domain::types::compact(&state)
}

/// b.ug2 Stage 6b: `rewrite_walk_entry`'s basis, the depth check's walks
/// measured alone on compact states, outside any commit: the full walk
/// (`check_section_tree_depth_capped`, an entry whose plan indexes cannot
/// answer the check incrementally) on a 9,000-deep chain, on 20,000
/// children of one section, on the 2,500-prim harness layer and on the dev
/// shape at 7,167 sections; and the cached route's incremental check from
/// the listings a plan made — a new leaf under the chain's bottom (9,002
/// hops up), and the 20,000-child section moved under an empty one (20,000
/// sections walked down) — each as one `WALK` line with its peak, the
/// sections it walks (`walked`, as the plan projects them) and the peak per
/// walked section. Report only.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.ug2 Stage 6b depth-walk fit"]
async fn depth_walks_are_reported_ug2() {
    use live_edit::domain::limits::MAX_SECTION_TREE_DEPTH;
    use live_edit::domain::plan_index::{full_depth_verdict, incremental_depth_verdict_for_test};
    let _serial = measure_lock().await;
    let report = |name: &str, walked: u64, peak: u64| {
        eprintln!(
            "WALK {name}: walked {walked} peak {peak} per_walked {:.1}",
            peak as f64 / walked.max(1) as f64
        );
    };
    let full_bases = [
        ("full-chain-9k", base_delta(Base::Chain(UG2_CHAIN))),
        ("full-children-20k", base_delta(Base::Children(DENSE))),
        ("full-prim-2500", base_delta(Base::HarnessPrims(2_500))),
        (
            "full-dev-7167",
            base_delta(Base::DevAssets {
                assets: 220,
                sections: 7_167,
            }),
        ),
    ];
    for (name, delta) in full_bases {
        let state = compact_of(delta);
        let walked = state.sections.len() as u64;
        let (verdict, peak) = peak_of(|| full_depth_verdict(&state, MAX_SECTION_TREE_DEPTH));
        verdict.expect("the base is within the depth cap");
        report(name, walked, peak);
    }

    // A new leaf under the chain's bottom: one listing, walked up to the root.
    let mut chain = compact_of(base_delta(Base::Chain(UG2_CHAIN)));
    let bottom = chain_bottom(UG2_CHAIN);
    apply_delta(
        &mut chain,
        Some(OwnedDelta {
            struct_commands: vec![create_section(bottom, FAR_SECTION, "leaf".to_owned())],
            ..Default::default()
        }),
    )
    .expect("the leaf applies");
    let listed = [(bottom, FAR_SECTION)];
    let (verdict, peak) =
        peak_of(|| incremental_depth_verdict_for_test(&chain, &listed, MAX_SECTION_TREE_DEPTH));
    assert!(
        matches!(verdict, Some(Ok(()))),
        "the chain's leaf is answered incrementally, within the cap: {verdict:?}"
    );
    // Hops up: the leaf, the 9,000 chain sections and `SECTION`; then the
    // leaf as the one root walked down.
    report("incremental-chain-9k", UG2_CHAIN as u64 + 2 + 1, peak);

    // `SECTION`, with 20,000 children, moved under the empty second section.
    let mut broad = compact_of(base_delta(Base::ChildrenAndEmptySection(DENSE)));
    apply_delta(
        &mut broad,
        Some(OwnedDelta {
            struct_commands: vec![StructCommand::MoveSection(CommandMoveSection {
                old_parent_id: 0,
                new_parent_id: SECOND_SECTION,
                section_id: SECTION,
                new_name: "base".to_owned(),
            })],
            ..Default::default()
        }),
    )
    .expect("the move applies");
    let listed = [(SECOND_SECTION, SECTION)];
    let (verdict, peak) =
        peak_of(|| incremental_depth_verdict_for_test(&broad, &listed, MAX_SECTION_TREE_DEPTH));
    assert!(
        matches!(verdict, Some(Ok(()))),
        "the move is answered incrementally, within the cap: {verdict:?}"
    );
    // Hops up: `SECTION` and the second section; then `SECTION` as the root
    // walked down and its 20,000 children.
    report("incremental-move-20k", 2 + 1 + DENSE as u64, peak);
}

// ─────────────────────────────────────────────────────────────────────────────
// b.ug2 Stage 6b (R-T6): a rollback after CR2
// ─────────────────────────────────────────────────────────────────────────────

/// b.ug2 Stage 6b, R-T6: a deleting structural edit whose commit answers
/// `STALE` after CR2, so its plan is rolled back under CR2's held need, on the
/// cached route with the object cached ([`NeedRecorder`]). Four arms, each a
/// deleting delta on a base whose table is full ("short": the plan's removals
/// can leave it without the room to re-create them, so its rollback may
/// rebuild it at its pre-plan length, which the plan projects and CR2 keeps
/// in its growth term) or has room ("normal"):
///
/// - `rt6-dev-delete-{short,normal}`: deleting the first placed asset (28
///   sections, 105 field rows) of a dev-shape layer whose section table is
///   at 7,168 of 7,168 rows once the warm-up's section is in, or at 6,001;
/// - `rt6-fields-cascade-{short,normal}`: deleting a section with 5,000
///   field rows, each its own interned name, from a field table (and name
///   map) at 28,672 of 28,672 rows, or at 23,000.
///
/// The structural pre-dispatch hook, at the first attempt's commit, splits
/// the segment ([`split_segment`]: the digests and the commit's build so
/// far), lands another replica's one-field fast-path commit, and splits
/// again; the between-attempts hook splits once more before the retry, so
/// the third split's peak is the `EVALSHA`, its `STALE` answer and the
/// rollback alone. Prints every segment (`SEG`), the splits (`SPLIT`: peak
/// and live bytes above the commit's start) and the cache's rollback and
/// rebuild counts. Report only.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.ug2 Stage 6b rollback-after-CR2 arms"]
async fn rollback_after_cr2_arms_are_reported_ug2() {
    rollback_after_cr2_arms().await;
}

/// [`rollback_after_cr2_arms_are_reported_ug2`]'s pre-dispatch hook body:
/// once `armed`, split the segment, land `other`'s one-field fast-path
/// commit on `object` (so the measured commit answers `STALE`), and split
/// again; disarmed, nothing.
#[cfg(feature = "test-fault-injection")]
fn rt6_competing_commit(
    other: Arc<LayerStore>,
    session: String,
    object: u64,
    armed: Arc<AtomicBool>,
) -> futures::future::BoxFuture<'static, ()> {
    Box::pin(async move {
        if !armed.swap(false, Ordering::SeqCst) {
            return;
        }
        split_segment();
        let poke = OwnedDelta {
            set_fields: vec![field("stale-poke".to_owned(), vec![7; 8])],
            ..Default::default()
        };
        let relay = Bytes::from(common::owned_delta::delta_to_ompf_bytes(&poke, &[]));
        Box::pin(other.update_layer(
            &session,
            "layer",
            object,
            Some(poke),
            Some(relay),
            "rt6-other",
            0,
        ))
        .await
        .expect("the other replica's commit lands");
        split_segment();
    })
}

/// [`rollback_after_cr2_arms_are_reported_ug2`]'s body.
#[cfg(feature = "test-fault-injection")]
async fn rollback_after_cr2_arms() {
    let only = std::env::var("B26E_FIT_ONLY").unwrap_or_default();
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let delete = |parent_id: u64, section_id: u64| OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id,
            section_id,
        })],
        ..Default::default()
    };
    let arms = [
        (
            "rt6-dev-delete-short",
            Base::DevAssets {
                assets: 220,
                sections: 7_167,
            },
            delete(DEV_FIRST_ID, DEV_ASSET0_ROOT),
        ),
        (
            "rt6-dev-delete-normal",
            Base::DevAssets {
                assets: 190,
                sections: 6_000,
            },
            delete(DEV_FIRST_ID, DEV_ASSET0_ROOT),
        ),
        (
            "rt6-fields-cascade-short",
            Base::SplitFields {
                keep: 23_672,
                doomed: 5_000,
            },
            delete(0, SECOND_SECTION),
        ),
        (
            "rt6-fields-cascade-normal",
            Base::SplitFields {
                keep: 18_000,
                doomed: 5_000,
            },
            delete(0, SECOND_SECTION),
        ),
    ];
    // Once-per-process code of the route, unmeasured.
    {
        let store = fit_store(&tv, FitRoute::CachedWarm, tv.client().clone());
        let label = "rt6-process-warmup";
        let (session, object) = base_layer(&tv, label, Base::Children(0)).await;
        warm_cache(&store, &session, object, label).await;
        measure_commit(&store, &session, object, &one_new_child(0), label).await;
    }
    for (name, base, delta) in arms {
        if !name.contains(only.as_str()) {
            continue;
        }
        let (session, object) = base_layer(&tv, name, base).await;
        // Another replica, with its own cache: the competing commit's.
        let other = Arc::new(
            LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                .with_update_layer_backend(UpdateLayerBackend::Lua),
        );
        let armed = Arc::new(AtomicBool::new(false));
        let hook: live_edit::storage::watched_txn::WatchWindowHook = {
            let (other, session, armed) = (Arc::clone(&other), session.clone(), Arc::clone(&armed));
            Arc::new(move |_client: &fred::clients::Client| {
                rt6_competing_commit(
                    Arc::clone(&other),
                    session.clone(),
                    object,
                    Arc::clone(&armed),
                )
            })
        };
        let between: live_edit::storage::layer::BetweenAttemptsHook =
            Arc::new(|_attempt: u32| -> futures::future::BoxFuture<'static, ()> {
                Box::pin(async { split_segment() })
            });
        let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_metrics(shared_metrics())
            .with_structural_commit_pre_dispatch_hook_for_test(hook)
            .with_between_attempts_hook_for_test(between);
        // A memory measurement: no shadow copy of the planned entry.
        store.structural_cache().set_shadow_check_for_test(false);
        warm_cache(&store, &session, object, name).await;
        let cache = store.structural_cache();
        let (rollbacks, rebuilds) = (
            cache.rollbacks_for_test(),
            cache.rollback_rebuilds_for_test(),
        );
        armed.store(true, Ordering::SeqCst);
        begin_splits();
        let (segments, residual, wall) =
            measure_commit(&store, &session, object, &delta, name).await;
        assert!(
            !armed.load(Ordering::SeqCst),
            "{name}: fixture: the competing commit ran"
        );
        for (i, s) in segments.iter().enumerate() {
            eprintln!(
                "SEG {name} CachedWarm #{i} peak {} need {:?}",
                s.peak, s.need
            );
        }
        for (i, (peak, live)) in splits().into_iter().enumerate() {
            eprintln!("SPLIT {name} #{i} peak {peak} live {live}");
        }
        eprintln!(
            "RT6 {name}: rollbacks {} rollback_rebuilds {} residual {residual} wall_s {:.3}",
            cache.rollbacks_for_test() - rollbacks,
            cache.rollback_rebuilds_for_test() - rebuilds,
            wall.as_secs_f64()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// b.26e Stage 2: the stalled WATCH commit, for the queued-command fit
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the WATCH stall hook ([`stall_hook`]) parks the transaction
/// connection; off, it does nothing and allocates nothing.
#[cfg(feature = "test-fault-injection")]
static STALL_ON: AtomicBool = AtomicBool::new(false);

/// Stalls released, and the transaction connection's in-flight queue at each
/// release (the `BLPOP` plus the whole pipeline), in a ring the releaser
/// writes without allocating.
#[cfg(feature = "test-fault-injection")]
static STALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "test-fault-injection")]
static STALL_QUEUED: [AtomicUsize; 64] = [const { AtomicUsize::new(0) }; 64];
/// Stalls released by the backstop deadline rather than on a settled queue.
#[cfg(feature = "test-fault-injection")]
static STALL_BACKSTOPS: AtomicUsize = AtomicUsize::new(0);
/// Stall releases that failed or found the stall already ended (its `BLPOP`
/// answered before the release), counted by the releaser.
#[cfg(feature = "test-fault-injection")]
static STALL_RELEASE_FAILURES: AtomicUsize = AtomicUsize::new(0);

/// The longest a stall waits for its queue to settle before the backstop
/// releases it.
#[cfg(feature = "test-fault-injection")]
const STALL_SETTLE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(20);

/// The WATCH stall hook, installed through
/// `LayerStore::with_pre_dispatch_hook_for_test`, which awaits it with the
/// guarded transaction client right before the commit pipeline is sent.
/// With [`STALL_ON`] set it parks a `BLPOP` on that connection
/// (`ConnStall`, `observer` carrying its `CLIENT LIST` polls and the
/// release), so the whole pipeline queues behind it, and spawns a releaser:
/// it reads the client's retention report and releases the stall once the
/// connection's in-flight queue holds more than the `BLPOP` and is unchanged
/// across two consecutive reports, each answered by its own router pass (one
/// pass every 500 ms at least). So the queue reaches the `BLPOP` plus the
/// whole pipeline by construction, not by timing. The releaser records the
/// queue it released at ([`STALL_QUEUED`]).
#[cfg(feature = "test-fault-injection")]
fn stall_hook(
    observer: fred::clients::Client,
    key: String,
) -> live_edit::storage::watched_txn::WatchWindowHook {
    use futures::FutureExt;
    Arc::new(move |client: &fred::clients::Client| {
        if !STALL_ON.load(Ordering::SeqCst) {
            return std::future::ready(()).boxed();
        }
        let observer = observer.clone();
        let key = key.clone();
        async move {
            let stall = ConnStall::park(client, &observer, key).await;
            let txn = client.clone();
            tokio::spawn(async move {
                let deadline = tokio::time::Instant::now() + STALL_SETTLE_DEADLINE;
                let mut last = 0;
                let queued = loop {
                    let report = connection_report("stalled txn connection", &txn).await;
                    let queued = report.iter().map(|c| c.queued).max().unwrap_or(0);
                    if queued > 1 && queued == last {
                        break queued;
                    }
                    if tokio::time::Instant::now() >= deadline {
                        STALL_BACKSTOPS.fetch_add(1, Ordering::SeqCst);
                        break queued;
                    }
                    last = queued;
                };
                let n = STALLS.fetch_add(1, Ordering::SeqCst);
                STALL_QUEUED[n % STALL_QUEUED.len()].store(queued, Ordering::SeqCst);
                // A detached task's panic would reach no test: a release that
                // failed, or found the stall already gone (its `BLPOP` timed
                // out), is counted for the measuring test to check.
                let released = stall.release().await;
                if !matches!(released, Ok(ref v) if !v.is_null()) {
                    STALL_RELEASE_FAILURES.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
        .boxed()
    })
}

/// The stalled fit fixtures: the WATCH full set's dense ones.
#[cfg(feature = "test-fault-injection")]
const STALLED_FIXTURES: &[&str] = &[
    "field-dense",
    "time-sample-dense",
    "mixed",
    "structure-dense",
];

/// b.26e Stage 2: the WATCH commit queue at its deterministic
/// worst case. Each of [`STALLED_FIXTURES`] commits on the WATCH path under
/// `txn` with its pipeline held behind a stalled transaction connection
/// ([`stall_hook`]) until the whole pipeline is queued, on a client whose
/// fast unresponsive verdict is off (the stall outlasts its 2 s) and whose
/// command timeout is 30 s. Prints one `STALLED` line per fixture: the
/// processing peak, the largest charge held in processing, the per-delta
/// sizes and `n`, the pipeline's commands (the released queue less the
/// `BLPOP`), with each phase's peak against its charge at the production
/// constants. Self-checks: every measured fixture's commit was stalled and
/// released on a settled queue, and at least one has `n / next_pow2(n) ≥
/// 0.9`, so a fit from these runs sees a queue near the top of its power of
/// two. The phases' overruns are reported here, and asserted per fixture by
/// the `stalled_watch_*_stays_within_its_charge_b26e` tests.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e Stage 2 stalled WATCH commit queue"]
async fn stalled_watch_commit_queue_is_reported_b26e() {
    let ratios: Vec<f64> = measure_stalled_watch(STALLED_FIXTURES, WindowTransport::Script)
        .await
        .iter()
        .map(|s| s.ratio)
        .collect();
    assert!(
        ratios.iter().any(|&r| r >= 0.9),
        "fixture self-check: at least one stalled fixture's pipeline fills at least 0.9 of its \
         power of two: {ratios:?}"
    );
}

/// One stalled WATCH commit ([`measure_stalled_watch`]).
#[cfg(feature = "test-fault-injection")]
struct Stalled {
    m: Measured,
    /// The pipeline's commands: the queue it was released at, less the
    /// `BLPOP`.
    n: usize,
    /// `n` ÷ its next power of two.
    ratio: f64,
}

/// b.26e Stage 2: the WATCH commit queue at its
/// deterministic worst case, asserted. Each of the WATCH full set's dense
/// fixtures commits with its whole pipeline queued behind a stalled
/// transaction connection ([`measure_stalled_watch`]) — fred's in-flight
/// queue at its largest, whatever the replies' timing — and every phase
/// stays within the charge in force: the queued-command term
/// (`QUEUED_COMMAND_CHARGE_BYTES` × next-power-of-two slots of every
/// connection's queue) covers it. The stalled commit queue is held to its
/// charge on both of the WATCH read's window transports ([`transports_on`]).
/// These fixtures are dense deltas on a near-empty base (one section, no
/// children), so their WATCH read is a few commands per window on either
/// transport: what is dense here is the commit, not the read. A read of a
/// layer dense in value rows on the plain transport, where each plain
/// command holds its own future priced by `W_PLAIN_CMD`, is held to its
/// charge by the large-base reproducers through
/// [`assert_slow_path_update_within_charge_on`] (e.g.
/// [`slow_path_one_new_child_on_inline_value_base_stays_within_its_charge_b26e`]).
/// One test per fixture, so each stays within nextest's per-test timeout.
#[cfg(feature = "test-fault-injection")]
async fn assert_stalled_watch_within_charge(fixture: &'static str) {
    let mut over = Vec::new();
    for &transport in transports_on(Route::Watch) {
        let stalled = measure_stalled_watch(&[fixture], transport).await;
        assert_eq!(
            stalled.len(),
            1,
            "fixture self-check: {fixture} is one of the WATCH set's fixtures"
        );
        let overruns = stalled[0].m.phase_overruns();
        if !overruns.is_empty() {
            over.push(format!(
                "{transport:?} windows, {} commands queued: {overruns:?}",
                stalled[0].n
            ));
        }
    }
    assert!(
        over.is_empty(),
        "b.26e {fixture}, stalled: the WATCH commit's heap grew past the charge in force with \
         its whole pipeline queued: {over:?}"
    );
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stalled_watch_field_dense_stays_within_its_charge_b26e() {
    assert_stalled_watch_within_charge("field-dense").await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stalled_watch_time_sample_dense_stays_within_its_charge_b26e() {
    assert_stalled_watch_within_charge("time-sample-dense").await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stalled_watch_mixed_stays_within_its_charge_b26e() {
    assert_stalled_watch_within_charge("mixed").await;
}

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stalled_watch_structure_dense_stays_within_its_charge_b26e() {
    assert_stalled_watch_within_charge("structure-dense").await;
}

/// The stalled WATCH commits of the fixtures `names` names, measured and
/// printed ([`stalled_watch_commit_queue_is_reported_b26e`]'s `STALLED`
/// lines). Self-checks that every measured fixture's commit was stalled and
/// released on a settled queue, by a release that held, and that the released
/// queue less the `BLPOP` is the commit's whole pipeline — its mutations'
/// commands, its frame and its external-value index writes — within the
/// queue its charge predicted
/// (`test_support::watch_queue_prediction_for_test`). Every WATCH read of the
/// run, the warm-up's included, issues its windows on `transport`, set on
/// both of the run's clients (each the run's own, dropped with it).
#[cfg(feature = "test-fault-injection")]
async fn measure_stalled_watch(names: &[&str], transport: WindowTransport) -> Vec<Stalled> {
    use live_edit::storage::client::{with_liveness_overrides_for_test, LivenessOverrides};
    use live_edit::storage::test_support::watch_queue_prediction_for_test;
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut config = common::test_config();
    config.valkey_command_timeout_ms = 30_000;
    let overrides = LivenessOverrides {
        fast_timeout: Some(std::time::Duration::from_secs(120)),
        ..Default::default()
    };
    let client = with_liveness_overrides_for_test(overrides, connect_test_client(&config))
        .await
        .expect("the stalled run's client connects");
    tv.client().set_watch_read_transport_for_test(transport);
    client.set_watch_read_transport_for_test(transport);
    let observer = tv.client().pool.clients()[0].clone();
    let key = format!("{}t7-stall", tv.prefix());
    let svc_for = |client: live_edit::storage::client::ValkeyClient| {
        service_on_with(&tv, Route::Watch, client, |store| {
            store.with_pre_dispatch_hook_for_test(stall_hook(observer.clone(), key.clone()))
        })
    };
    let suffix = transport_suffix(transport);
    warm_up(
        &tv,
        Path::UpdateSlow,
        Route::Watch,
        &format!("stalled-warmup{suffix}"),
    )
    .await;
    let mut stalled = Vec::new();
    for fixture in fixtures_on(Path::UpdateSlow, Route::Watch)
        .into_iter()
        .filter(|f| names.contains(&f.name))
    {
        let label = format!("stalled{suffix}-{}", fixture.name);
        let svc = svc_for(client.clone());
        let (stalls, backstops, failures) = (
            STALLS.load(Ordering::SeqCst),
            STALL_BACKSTOPS.load(Ordering::SeqCst),
            STALL_RELEASE_FAILURES.load(Ordering::SeqCst),
        );
        STALL_ON.store(true, Ordering::SeqCst);
        let m = measure_on(svc, &tv, Path::UpdateSlow, Route::Watch, &fixture, &label).await;
        STALL_ON.store(false, Ordering::SeqCst);
        let released = STALLS.load(Ordering::SeqCst) - stalls;
        let failed = STALL_RELEASE_FAILURES.load(Ordering::SeqCst) - failures;
        assert!(
            released >= 1 && STALL_BACKSTOPS.load(Ordering::SeqCst) == backstops && failed == 0,
            "fixture self-check: {label}: the commit was stalled and released on a settled \
             queue ({released} stalls, {} backstop releases, {failed} failed releases)",
            STALL_BACKSTOPS.load(Ordering::SeqCst) - backstops
        );
        let queued = STALL_QUEUED[(STALLS.load(Ordering::SeqCst) - 1) % STALL_QUEUED.len()]
            .load(Ordering::SeqCst);
        let n = queued.saturating_sub(1);
        // The released queue, less the `BLPOP`, is exactly the commit's
        // whole pipeline: every command its mutations queue
        // (`commands_for`); the frame every WATCH commit here queues —
        // `MULTI`, the commit-token record, `XADD`, the `CLIENT ID` witness
        // (armed on every server this runs against) and `EXEC`, with no
        // relay `XTRIM` (a fresh layer's stream has nothing to trim) and no
        // commit release ring write (a fresh base replaces no external value);
        // and one `obj:ext` index write per external value row. So a release
        // before the pipeline was all queued fails here. And the charge's
        // prediction (`watch_queued_commands`, an upper bound) covers it.
        let delta = fixture
            .delta
            .as_ref()
            .expect("fixture self-check: the stalled fixtures carry a delta");
        let mut base = ObjectState::default();
        apply_delta(&mut base, Some(base_delta(fixture.base))).expect("the base delta applies");
        let prediction = watch_queue_prediction_for_test(&base, delta);
        let external_rows = delta
            .set_fields
            .iter()
            .filter(|f| f.ext_value_index != 0)
            .count()
            + delta
                .time_samples
                .iter()
                .filter(|s| s.ext_value_index != 0)
                .count();
        let whole = prediction.mutation_commands + 5 + external_rows as u64;
        assert_eq!(
            n as u64, whole,
            "fixture self-check: {label}: the stall released at {n} queued commands, not the \
             commit's whole pipeline ({} mutation commands, five frame commands and \
             {external_rows} external-value index writes)",
            prediction.mutation_commands
        );
        assert!(
            whole <= prediction.queued.txn,
            "fixture self-check: {label}: the pipeline ({whole}) is within its charge's \
             prediction ({})",
            prediction.queued.txn
        );
        let ratio = n as f64 / n.max(1).next_power_of_two() as f64;
        eprintln!(
            "STALLED {}{suffix}: proc_peak {} proc_charge {} proc_held_max {} body {} delta {} \
             elements {} reorder {} n_txn {n} whole {whole} predicted {} next_pow2 {} ratio \
             {ratio:.3} stalls {released} wall_s {:.3}; overruns {:?}; phases: {}",
            fixture.name,
            m.phase_peak("processing"),
            m.processing_target,
            m.phase_held_max("processing"),
            m.body,
            m.delta,
            m.elements,
            m.reorder,
            prediction.queued.txn,
            n.max(1).next_power_of_two(),
            m.wall.as_secs_f64(),
            m.phase_overruns(),
            m.phase_summary()
        );
        stalled.push(Stalled { m, n, ratio });
    }
    stalled
}

/// The parse alone — `oodf_ffi::parse_delta` over each distinct delta — sets
/// `k_parse`: its Rust-heap peak per delta byte, × 1.25, plus 100 for the C++
/// side's transient temporaries, which the Rust counts cannot see. Asserts the
/// Rust peak stays within the production `k_parse` less that allowance.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn delta_parse_alone_stays_within_k_parse() {
    let _serial = measure_lock().await;
    let mut deltas: Vec<(String, OwnedDelta)> = Vec::new();
    for path in [Path::Create, Path::UpdateFast, Path::UpdateSlow] {
        for fixture in fixtures(path) {
            if let Some(delta) = fixture.delta {
                deltas.push((format!("{path:?}-{}", fixture.name), delta));
            }
        }
    }
    let rust_budget_centi = PARSE_EXPANSION_CENTI.saturating_sub(100);
    eprintln!("== delta parse alone ==");
    let mut worst = 0.0_f64;
    let mut over = Vec::new();
    for (label, delta) in deltas {
        let bytes = oodf_ffi::build_delta(&delta).expect("fixtures build");
        drop(delta);
        let len = bytes.len() as u64;
        let baseline = begin_window();
        let parsed = oodf_ffi::parse_delta(&bytes).expect("fixtures parse");
        let peak = end_window(baseline);
        drop(parsed);
        let ratio = peak as f64 / len.max(1) as f64;
        worst = worst.max(ratio);
        eprintln!("{label:<32} delta {len:>9} B  parse peak {peak:>11} B  ratio {ratio:.3}");
        if peak > (len * rust_budget_centi).div_ceil(100) {
            over.push(format!("{label}: {peak} B for {len} B"));
        }
    }
    eprintln!(
        "measured k_parse (Rust heap) max {worst:.3}/B  ->  recommended k_parse = ceil(x{HEADROOM}) + \
         100 = {} centi (production {PARSE_EXPANSION_CENTI})",
        (worst * HEADROOM_CENTI).ceil() as u64 + 100
    );
    assert!(
        over.is_empty(),
        "the parse's Rust heap passed k_parse ({PARSE_EXPANSION_CENTI} hundredths, less the \
         100 C++ allowance) for: {over:?}"
    );
}

/// The allocation census: a fast-path `UpdateLayer` of 64 inline values of
/// exactly 65,521 B makes exactly 64 allocations of that size over the whole
/// call — one owned copy of each value, the parsed delta's. A handler-side
/// deep copy of the delta (the pre-b.715 recovery backup) makes 128. The
/// envelope's 25 % margin cannot see that copy (it costs 1-18 % of a peak),
/// so this counts it directly.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn fast_path_update_makes_one_owned_copy_of_each_inline_value() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    measure(
        &tv,
        Path::UpdateFast,
        &Fixture::header_only(),
        "census-warmup",
    )
    .await;
    let fixture = Fixture::new(
        "census",
        Feeds::Bytes,
        OwnedDelta {
            set_fields: value_fields(VALUES, CENSUS_VALUE_LEN),
            ..Default::default()
        },
        &[],
    );
    begin_census(CENSUS_VALUE_LEN);
    let m = measure(&tv, Path::UpdateFast, &fixture, "census").await;
    let (counted, _) = end_census();
    eprintln!(
        "== census: {counted} allocations of exactly {CENSUS_VALUE_LEN} B for {VALUES} inline \
         values (peak {} B) ==",
        m.peak
    );
    // Only allocations inside the measured window are counted: the fixture's
    // own inputs are built before it opens, and `measure`'s path self-check
    // after it closes.
    assert_eq!(
        counted, VALUES,
        "a fast-path UpdateLayer must hold exactly one owned copy of each inline value; {counted} \
         allocations of {CENSUS_VALUE_LEN} B means the delta (or its values) was copied"
    );
}

/// External values in the I5 census fixture: more than the store keeps in
/// flight.
const EXT_CENSUS_BLOBS: usize = 32;
/// Their size: prime (2^17 − 1), so no other allocation of the upload has it
/// by coincidence.
const EXT_CENSUS_LEN: usize = 131_071;
/// The external-value store's in-flight bound, the real
/// `STORE_VALUES_MAX_INFLIGHT` (re-exported under `test-support`).
const EXT_STORE_MAX_INFLIGHT: isize = STORE_VALUES_MAX_INFLIGHT as isize;
/// The worker threads of the census test's runtime (its
/// `#[tokio::test(worker_threads = …)]`, which must match; the test checks).
/// The test's own future — the handler, its `buffered` stream and the store
/// futures — runs on the test thread; fred's routers run on these workers.
const CENSUS_WORKER_THREADS: usize = 1;
/// The most store-path copies alive at once: [`EXT_STORE_MAX_INFLIGHT`] plus
/// one per runtime thread other than the polling one that can run a router —
/// here the [`CENSUS_WORKER_THREADS`] workers — so 9.
///
/// The store keeps at most [`EXT_STORE_MAX_INFLIGHT`] stores in flight, but a
/// finished store's copy lives in the ARGV of fred's `Command` until the
/// router drops it inside the free function
/// `fred::protocol::responders::respond_to_caller`
/// (`vendor/fred-10.1.0/src/protocol/responders.rs`), right after sending the
/// reply, with no await between. In that gap the woken store future (on the
/// test thread) can finish and `buffered` start the next copy, so each router
/// thread holds at most one such finished copy at a time. Measured by the
/// Engineer's instrumented runs: a peak of 8 in 39 of 40 runs, 9 in 1, the
/// lingering copy freed inside `respond_to_caller` on the worker. See the
/// comment in `ValkeyExternalValueStore::store_values`.
const EXT_COPIES_ALIVE_CEILING: isize = EXT_STORE_MAX_INFLIGHT + CENSUS_WORKER_THREADS as isize;

/// A fast-path `UpdateLayer` fixture of [`EXT_CENSUS_BLOBS`] external values
/// of [`EXT_CENSUS_LEN`] bytes, and the values (alive until the caller drops
/// them, after the measured window).
fn ext_census_fixture() -> (Fixture, Vec<Vec<u8>>) {
    let values = blobs(EXT_CENSUS_BLOBS, EXT_CENSUS_LEN, 0x17);
    let refs: Vec<&[u8]> = values.iter().map(Vec::as_slice).collect();
    let set_fields = refs
        .iter()
        .enumerate()
        .map(|(i, b)| external_field(format!("blob{i}"), b, i as u32 + 1))
        .collect();
    let fixture = Fixture::new(
        "ext-census",
        Feeds::Ext,
        OwnedDelta {
            set_fields,
            ..Default::default()
        },
        &refs,
    );
    assert_ne!(
        fixture.ompf.len() % CHUNK,
        EXT_CENSUS_LEN,
        "fixture self-check: no content chunk has the census size"
    );
    (fixture, values)
}

/// I5: the external-value store makes one owned copy of each external value,
/// and the store path never holds more than its in-flight bound plus one
/// finished copy per router thread ([`EXT_COPIES_ALIVE_CEILING`]: 8 + 1 on
/// this runtime) — it copies a value only as a store starts, at most
/// `STORE_VALUES_MAX_INFLIGHT` stores at a time. The peak is checked to reach
/// the in-flight bound (`buffered`'s first poll starts that many stores), so
/// the ceiling is exercised.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn external_value_store_path_holds_at_most_its_inflight_bound_plus_one_finished_copy() {
    assert_eq!(
        tokio::runtime::Handle::current().metrics().num_workers(),
        CENSUS_WORKER_THREADS,
        "fixture self-check: the ceiling counts this runtime's worker threads"
    );
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    measure(
        &tv,
        Path::UpdateFast,
        &Fixture::header_only(),
        "ext-census-warmup",
    )
    .await;
    let (fixture, _values) = ext_census_fixture();
    begin_census(EXT_CENSUS_LEN);
    measure(&tv, Path::UpdateFast, &fixture, "ext-census").await;
    let (made, most_alive) = end_census();
    eprintln!(
        "== I5 census: {made} allocations of exactly {EXT_CENSUS_LEN} B for {EXT_CENSUS_BLOBS} \
         external values, at most {most_alive} alive at once =="
    );
    assert_eq!(
        made, EXT_CENSUS_BLOBS,
        "one owned copy of each external value"
    );
    assert!(
        most_alive <= EXT_COPIES_ALIVE_CEILING,
        "{most_alive} external-value copies were alive at once, past {EXT_COPIES_ALIVE_CEILING} \
         (the store's in-flight bound of {EXT_STORE_MAX_INFLIGHT}, plus one finished copy per \
         router thread): the store path no longer bounds its copies"
    );
    assert!(
        most_alive >= EXT_STORE_MAX_INFLIGHT,
        "fixture self-check: {most_alive} copies at most, below the in-flight bound of \
         {EXT_STORE_MAX_INFLIGHT}, so the ceiling was not exercised"
    );
}

/// I5, shared with the mirror: with an ext-mirror sender wired, the store
/// still makes exactly one copy of each external value — the mirror's
/// payload is that copy, not a second one — and every value reaches the
/// mirror's queue. The queue is not drained, so every payload is still alive
/// when the window closes.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn external_value_store_makes_one_copy_per_value_with_a_mirror_sender_wired() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    measure(
        &tv,
        Path::UpdateFast,
        &Fixture::header_only(),
        "ext-mirror-census-warmup",
    )
    .await;
    let (fixture, _values) = ext_census_fixture();
    let base = service(&tv);
    // A budget no census value can exhaust, so every value is enqueued.
    let (sender, mut rx) = ext_mirror_channel(
        2 * EXT_CENSUS_BLOBS,
        std::num::NonZeroU64::new(1 << 30).expect("non-zero"),
        Arc::clone(&base.metrics),
    );
    let svc = LiveEditService {
        ext_value_store: Arc::new(
            ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_ext_mirror_sender(sender),
        ),
        ..base
    };
    begin_census(EXT_CENSUS_LEN);
    measure_on(
        svc,
        &tv,
        Path::UpdateFast,
        Route::Default,
        &fixture,
        "ext-mirror-census",
    )
    .await;
    let (made, most_alive) = end_census();
    eprintln!(
        "== I5 mirror census: {made} allocations of exactly {EXT_CENSUS_LEN} B for \
         {EXT_CENSUS_BLOBS} external values with a mirror sender, {most_alive} alive at most =="
    );
    let mut mirrored = 0;
    while let Ok(item) = rx.try_recv() {
        assert_eq!(item.bytes().len(), EXT_CENSUS_LEN);
        mirrored += 1;
    }
    assert_eq!(
        mirrored, EXT_CENSUS_BLOBS,
        "precondition: every value was enqueued for the mirror"
    );
    assert_eq!(
        made, EXT_CENSUS_BLOBS,
        "the mirror must share each value's one owned copy, not make a second"
    );
}

/// T13 (b.mhq): the heap one queued ext-mirror item holds beyond its value's
/// bytes — the hash string, the store stamp, the copy's owner and its charge
/// slot, the channel's slot — fits the per-item allowance the mirror charges each
/// value on top of its size (`ITEM_OVERHEAD_KIB`, one KiB), so the byte
/// budget bounds the memory the queue really holds. Measured as the live
/// heap the channel's creation and 256 queued 4 KiB values add, less the
/// values' own bytes, per item; the store path's copy and send, as
/// `store_values` makes them (a 64-character hash string cloned per send).
/// The channel's creation is inside the window, so its fixed allocations
/// count against the items too.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn ext_mirror_heap_per_queued_item_beyond_its_value_fits_the_per_item_allowance() {
    use live_edit::checkpoint::ext_mirror::ITEM_OVERHEAD_KIB;

    const ITEMS: usize = 256;
    const VALUE_LEN: usize = 4096;
    let _serial = measure_lock().await;
    let metrics = shared_metrics();
    let payload = vec![0x5Au8; VALUE_LEN];
    let hash_hex = "ab".repeat(32);
    // Warm-up, outside the window: the metric cells' first use.
    {
        let (sender, _rx) = ext_mirror_channel(
            1,
            std::num::NonZeroU64::new(1 << 20).expect("non-zero"),
            Arc::clone(&metrics),
        );
        sender.send(
            hash_hex.clone(),
            sender.copy_value(&payload),
            "1760000000000".to_owned(),
        );
    }

    let before = LIVE.load(Ordering::SeqCst);
    let (sender, rx) = ext_mirror_channel(
        ITEMS,
        std::num::NonZeroU64::new(1 << 30).expect("non-zero"),
        Arc::clone(&metrics),
    );
    for _ in 0..ITEMS {
        // The store stamp a queued item holds, the 13-digit decimal string
        // the store script returns.
        sender.send(
            hash_hex.clone(),
            sender.copy_value(&payload),
            "1760000000000".to_owned(),
        );
    }
    let held = LIVE.load(Ordering::SeqCst) - before;
    assert_eq!(
        sender.available_budget_bytes(),
        sender.budget_bytes()
            - (ITEMS as u64) * (VALUE_LEN as u64 / 1024 + ITEM_OVERHEAD_KIB) * 1024,
        "fixture self-check: every item is queued and charged"
    );
    assert!(
        held >= ITEMS * VALUE_LEN,
        "fixture self-check: the window sees the {ITEMS} values' bytes ({held} B held)"
    );
    let per_item = (held - ITEMS * VALUE_LEN) as f64 / ITEMS as f64;
    eprintln!(
        "== T13: {per_item:.1} B of heap per queued ext-mirror item beyond its value, \
         against {} B allowed ==",
        ITEM_OVERHEAD_KIB * 1024
    );
    assert!(
        per_item <= (ITEM_OVERHEAD_KIB * 1024) as f64,
        "a queued ext-mirror item holds {per_item:.1} B beyond its value, more than the \
         {ITEM_OVERHEAD_KIB} KiB each is charged for it: the byte budget no longer bounds the \
         queue's memory"
    );
    drop((sender, rx));
}

/// glibc's in-use bytes (`mallinfo2`), sampled every millisecond inside the
/// measured window of the largest delta fixture, beside the Rust-counted peak.
/// The baseline is taken when the window opens, after the inputs are built.
/// Report only; no assertion: the gap includes glibc's own overhead and
/// fragmentation as well as any allocation outside the Rust heap, and the
/// samples can miss short peaks.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn largest_delta_fixture_glibc_in_use_is_reported() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let fixture = fixtures(Path::Create)
        .into_iter()
        .filter(|f| f.feeds != Feeds::Ext && f.feeds != Feeds::Mixed)
        .max_by_key(|f| f.ompf.len())
        .expect("delta fixtures exist");
    measure(&tv, Path::Create, &Fixture::header_only(), "glibc-warmup").await;

    let stop = Arc::new(AtomicBool::new(false));
    let sampler = std::thread::spawn({
        let stop = Arc::clone(&stop);
        move || {
            let mut max = 0u64;
            while !stop.load(Ordering::SeqCst) {
                if WINDOW_OPEN.load(Ordering::SeqCst) {
                    let above =
                        glibc_in_use().saturating_sub(GLIBC_BASELINE.load(Ordering::SeqCst));
                    max = max.max(above);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            max
        }
    });
    let m = measure(&tv, Path::Create, &fixture, "glibc-largest").await;
    stop.store(true, Ordering::SeqCst);
    let glibc_peak = sampler.join().expect("sampler");
    eprintln!(
        "== glibc in-use, largest delta fixture ({}, {} B delta part) ==\n\
         Rust-counted peak {} B; glibc in-use peak above the window's baseline (1 ms samples) \
         {} B; gap {} B; charge {} B",
        m.name,
        m.delta,
        m.peak,
        glibc_peak,
        glibc_peak as i64 - m.peak as i64,
        m.charge
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// b.26e: post-release live memory and connection buffer retention
// ═════════════════════════════════════════════════════════════════════════════
//
// A call can leave heap live after its reservation is released, outside every
// charge. The largest such residue was the Valkey connections' buffers, which
// stock fred never shrinks; the connection buffer limits
// (`storage::connection_buffers`, the vendored fred's
// `live_edit::retention`) cut each one back once its traffic drains, so what
// the connections of a replica keep is bounded by `at_rest_bound_bytes`. The
// tests below hold that: the same upload repeated on one service and one
// client built by the production `connect` must leave the heap within that
// bound after every call, and stop growing once every connection the case can
// use has carried a large call. Each prints its per-call figures, then tears
// the service and the client's pools down one at a time (which one held
// it?); with `B26E_TRACK=1` every allocation the calls made is tracked by
// address and the large ones by the backtrace that made them (which type
// holds it?). The instrument sees the Rust heap only: glibc's own caches and
// the OODF parser's C++ side are outside it. glibc's in-use figure is printed
// beside it, and is comparable only with the tracker off (the default): the
// tracker's own map is in glibc's figure and not in the Rust one.
//
// The same cases run against a cluster too (test 11's cluster cases, gated on
// `LIVE_EDIT_TEST_CLUSTER_URL`), bounded at the cluster's primary count. After
// the calls every case also reads each general client's retention report
// (`fred::live_edit::retention::test_support`, [`connection_report`]) and
// holds it to the warm rule ([`warm_rule_violations`]): exactly one warm
// connection on a client with one connection or with read progress, every
// other drained connection in the idle shape. That is what sees a client
// keeping more than one warm connection, or none. Test 11c runs through the
// same code on a cluster with a tighter byte bound of its own
// (`OneWarmFigures::bound`: one warm connection per client its calls load)
// and, with test 17, the warm rule's standing negative controls
// ([`assert_warm_rule_discriminates`]).
//
// The rest of the section holds each retention rule to its own holder: the
// stream reader's class limits (test 12, and its cluster variant across a
// retarget), the retry buffer after a replay (test 15), a fred
// `Transaction`'s EXEC reply read outside the read loop (test 22) and the
// reader read limit under mixed reply sizes (test 23); then the in-flight
// slot report (M6); then one warm connection per client and the idle shape
// (test 18 and its cluster twin, tests 17 and 19), the routing table's owned
// strings (test 25), and the cached-route residue report.

/// Tracked allocations at least this large carry a backtrace in the
/// post-release reports, unless `B26E_BACKTRACE_MIN` sets another size for
/// the measured calls: connection buffers and command queues, and few of the
/// per-element allocations.
const POST_RELEASE_BACKTRACE_MIN: usize = 16 * 1024;

/// How long a post-release report waits after a call or a teardown step
/// before it reads the live bytes, so work the call left running (a router
/// dropping a finished command, a spawned task) has finished.
const SETTLE: std::time::Duration = std::time::Duration::from_millis(300);

/// The frames of a backtrace a post-release report names a site by.
const SITE_FRAMES: usize = 14;

/// Start tracking every counted allocation, in an empty map.
fn start_tracking(backtrace_min: usize) {
    let _guard = TrackerGuard::enter();
    TRACK_BACKTRACE_MIN.store(backtrace_min, Ordering::SeqCst);
    let previous = tracked_map().replace(std::collections::HashMap::with_capacity(1 << 16));
    drop(previous);
    TRACK_MAP.store(true, Ordering::SeqCst);
    TRACK.store(true, Ordering::SeqCst);
}

/// Stop recording new allocations; frees still leave the map.
fn stop_recording() {
    TRACK.store(false, Ordering::SeqCst);
}

/// Stop tracking and drop the map and the peak-time view's frees.
fn end_tracking() {
    stop_recording();
    let _guard = TrackerGuard::enter();
    TRACK_MAP.store(false, Ordering::SeqCst);
    TRACK_AT_PEAK.store(false, Ordering::SeqCst);
    let map = tracked_map().take();
    drop(map);
    let freed = std::mem::take(
        &mut *TRACKED_FREED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    drop(freed);
}

/// The tracked allocations still live: `(count, bytes)`.
fn tracked_totals() -> (usize, u64) {
    let _guard = TrackerGuard::enter();
    let map = tracked_map();
    map.as_ref().map_or((0, 0), |m| {
        (m.len(), m.values().map(|t| t.size as u64).sum())
    })
}

/// `name` without a trailing `::h<16 hex digits>` symbol hash.
fn without_symbol_hash(name: &str) -> &str {
    match name.rfind("::h") {
        Some(at)
            if name.len() - at == 19 && name[at + 3..].bytes().all(|b| b.is_ascii_hexdigit()) =>
        {
            &name[..at]
        }
        _ => name,
    }
}

/// Where `backtrace`'s allocation was made: its first [`SITE_FRAMES`]
/// function names past the allocator's own frames, innermost first.
fn allocation_site(backtrace: &std::backtrace::Backtrace) -> String {
    const ALLOCATOR_FRAMES: &[&str] = &[
        "std::backtrace",
        "std::sys",
        "upload_memory_envelope::track_alloc",
        "<upload_memory_envelope::Counting",
        "__rustc",
        "__rust_",
        "alloc::alloc",
        "<alloc::alloc::Global",
        "alloc::raw_vec",
        "<alloc::raw_vec",
    ];
    backtrace
        .to_string()
        .lines()
        .filter_map(|line| {
            let (index, name) = line.trim_start().split_once(": ")?;
            index.parse::<usize>().ok()?;
            Some(without_symbol_hash(name).to_owned())
        })
        .filter(|name| !ALLOCATOR_FRAMES.iter().any(|f| name.starts_with(f)))
        .take(SITE_FRAMES)
        .collect::<Vec<_>>()
        .join("\n        <- ")
}

/// Print the tracked allocations still live: the large ones grouped by
/// [`allocation_site`], the largest `top` sites by bytes, and the rest by
/// power-of-two size class.
fn report_tracked(title: &str, top: usize) {
    let _guard = TrackerGuard::enter();
    let grouped = {
        let map = tracked_map();
        let Some(map) = map.as_ref() else {
            return;
        };
        group_tracked(map.values())
    };
    print_tracked(title, top, grouped);
}

/// The tracked allocations live at the tracked call's peak (the peak-time
/// view, above): recorded at or before the peak's event and not freed by it,
/// grouped as [`report_tracked`] groups them, with the peak's growth above
/// the live bytes when tracking began. Allocations made before tracking
/// began and freed during it lower the growth, not this list, so the list
/// can exceed the growth.
fn report_tracked_at_peak(title: &str, top: usize) {
    let _guard = TrackerGuard::enter();
    let peak_event = TRACK_PEAK_EVENT.load(Ordering::SeqCst);
    let growth = TRACK_PEAK_LIVE
        .load(Ordering::SeqCst)
        .saturating_sub(TRACK_BASE_LIVE.load(Ordering::SeqCst));
    let grouped = {
        let map = tracked_map();
        let Some(map) = map.as_ref() else {
            return;
        };
        let freed = TRACKED_FREED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let live_at_peak = map.values().filter(|t| t.event <= peak_event).chain(
            freed
                .iter()
                .filter(|(t, freed_at)| t.event <= peak_event && *freed_at > peak_event)
                .map(|(t, _)| t),
        );
        group_tracked(live_at_peak)
    };
    print_tracked(
        &format!("{title} (growth {growth} B at tracker event {peak_event})"),
        top,
        grouped,
    );
}

/// Tracked allocations grouped for a report: the backtraced ones by
/// [`allocation_site`] (count, bytes, largest), the rest by power-of-two size
/// class (count, bytes), and the totals (count, bytes).
type GroupedTracked = (
    std::collections::HashMap<String, (usize, u64, usize)>,
    std::collections::BTreeMap<u32, (usize, u64)>,
    (usize, u64),
);

/// Group `tracked` ([`GroupedTracked`]). Called inside the tracker.
fn group_tracked<'a>(tracked: impl Iterator<Item = &'a Tracked>) -> GroupedTracked {
    let mut sites: std::collections::HashMap<String, (usize, u64, usize)> =
        std::collections::HashMap::new();
    let mut classes: std::collections::BTreeMap<u32, (usize, u64)> =
        std::collections::BTreeMap::new();
    let (mut count, mut bytes) = (0, 0);
    for t in tracked {
        count += 1;
        bytes += t.size as u64;
        match &t.backtrace {
            Some(backtrace) => {
                let site = sites.entry(allocation_site(backtrace)).or_default();
                site.0 += 1;
                site.1 += t.size as u64;
                site.2 = site.2.max(t.size);
            }
            None => {
                let class = classes
                    .entry(usize::BITS - t.size.leading_zeros())
                    .or_default();
                class.0 += 1;
                class.1 += t.size as u64;
            }
        }
    }
    (sites, classes, (count, bytes))
}

/// Print a grouped report: the largest `top` sites by bytes, then the size
/// classes. Called inside the tracker, with no tracker lock held.
fn print_tracked(title: &str, top: usize, grouped: GroupedTracked) {
    let (sites, classes, (count, bytes)) = grouped;
    // Built whole and printed after the map's lock is released, so no thread
    // waits on the map while this one waits on stderr.
    let backtrace_min = TRACK_BACKTRACE_MIN.load(Ordering::SeqCst);
    let mut out = format!(
        "-- tracked live: {title}: {count} allocations, {bytes} B (backtraced at >= \
         {backtrace_min} B) --\n"
    );
    let mut ranked: Vec<_> = sites.into_iter().collect();
    ranked.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    for (site, (n, total, largest)) in ranked.iter().take(top) {
        out.push_str(&format!(
            "  {total:>11} B in {n:>5} allocations (largest {largest} B) at\n        {site}\n"
        ));
    }
    let rest: u64 = ranked.iter().skip(top).map(|(_, (_, b, _))| b).sum();
    out.push_str(&format!(
        "  {rest} B more at {} other large sites\n",
        ranked.len().saturating_sub(top)
    ));
    for (class, (n, total)) in &classes {
        out.push_str(&format!(
            "  under {backtrace_min} B, size < 2^{class}: {n:>7} allocations, {total:>11} B\n"
        ));
    }
    eprint!("{out}");
}

/// The live counted bytes.
fn live_now() -> i64 {
    LIVE.load(Ordering::SeqCst) as i64
}

/// glibc's in-use bytes; 0 where `mallinfo2` is not available.
fn glibc_now() -> i64 {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        glibc_in_use() as i64
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        0
    }
}

/// [`service_on`] `client`, with its session and external-value stores on
/// `client` too, so every connection the measured uploads use belongs to
/// `client`, and quitting `client`'s pools closes all of them.
fn isolated_service(
    tv: &TestValkey,
    route: Route,
    client: live_edit::storage::client::ValkeyClient,
) -> LiveEditService {
    let svc = service_on(tv, route, client.clone());
    LiveEditService {
        session_store: Arc::new(SessionStore::new(client.clone(), tv.key_schema().clone())),
        ext_value_store: Arc::new(ValkeyExternalValueStore::new(
            client,
            tv.key_schema().clone(),
        )),
        ..svc
    }
}

/// One upload's input: its frames, one gRPC message per frame, and the
/// session its layer `"layer"` is in.
struct UploadInput {
    frames: Vec<Bytes>,
    session_id: String,
}

/// The input of one upload of `fixture` on `path`: a `CreateLayer` in a new
/// session, or an `UpdateLayer` of a new [`base_layer`] (made through `tv`'s
/// own client).
async fn upload_frames(tv: &TestValkey, path: Path, fixture: &Fixture, label: &str) -> UploadInput {
    let chunks = fixture.ompf.chunks(CHUNK).map(<[u8]>::to_vec);
    match path {
        Path::Create => {
            let session_id = new_session(tv, label).await;
            let frames = std::iter::once(grpc_frame(&make_create_header(&session_id, "layer")))
                .chain(chunks.map(|c| grpc_frame(&make_create_data(c))))
                .collect();
            UploadInput { frames, session_id }
        }
        Path::UpdateFast | Path::UpdateSlow => {
            let (session_id, object_id) = base_layer(tv, label, fixture.base).await;
            update_input(session_id, object_id, fixture)
        }
    }
}

/// The input of an `UpdateLayer` of `fixture` to the existing layer
/// `"layer"` of `session_id`, object `object_id`.
fn update_input(session_id: String, object_id: u64, fixture: &Fixture) -> UploadInput {
    let frames = std::iter::once(grpc_frame(&make_update_header(
        &session_id,
        "layer",
        object_id,
    )))
    .chain(
        fixture
            .ompf
            .chunks(CHUNK)
            .map(|c| grpc_frame(&make_update_data(c.to_vec()))),
    )
    .collect();
    UploadInput { frames, session_id }
}

/// One upload of `frames` on `path` through `svc`, which must commit.
async fn upload_once(svc: &LiveEditService, path: Path, frames: &[Bytes], name: &str, call: usize) {
    let result = match path {
        Path::Create => Box::pin(handle_create_layer(
            svc,
            request::<CreateLayerRequest>(frames),
        ))
        .await
        .map(drop),
        Path::UpdateFast | Path::UpdateSlow => Box::pin(handle_update_layer(
            svc,
            request::<UpdateLayerRequest>(frames),
        ))
        .await
        .map(drop),
    };
    result.unwrap_or_else(|e| panic!("{name} call {call}: the upload must commit: {e:?}"));
}

// ─────────────────────────────────────────────────────────────────────────────
// The warm rule, read through the retention report
// ─────────────────────────────────────────────────────────────────────────────
//
// A client's warm connection is a per-connection state (its `warm` and `idle`
// flags, its read progress and its buffers' capacities), which the vendored
// fred reports per client in test-support builds
// (`fred::live_edit::retention::test_support`, fred's
// `live-edit-retention-report` feature, enabled by live-edit's
// `test-support`). The tests below read it there rather than infer it from
// heap bytes, so the check does not depend on the primary count or on the
// limits' constants. Every report is taken outside a measured window and
// dropped before the next live-bytes read: the counting allocator counts the
// report's `Vec`.

/// How long [`connection_report`] waits for a client's router to answer: 20
/// unresponsive-check intervals. A router passes at least once per interval
/// (500 ms, set on every builder `connect` makes) with or without traffic, and
/// answers at the first pass whose sweep began after the request.
const REPORT_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// How often [`connection_report`] looks for the answer. Never a hot loop: the
/// router's `try_lock` on the report slot yields to a test thread holding it,
/// leaving the request for its next pass.
const REPORT_POLL: std::time::Duration = std::time::Duration::from_millis(10);

/// `client`'s retention report: one [`ConnectionReport`] per connection, as
/// the first router pass whose sweep began after this call left it, so a
/// sweep mode set before this call is the mode that sweep ran. `client` must
/// be a client (a pool member, a `txn` slot's client, the pub/sub client),
/// never a `Pool`, whose `inner` round-robins across its members. Panics,
/// naming `label` and the client, when no answer comes within
/// [`REPORT_WAIT`]: a router ended by `QUIT` or an abort answers nothing.
async fn connection_report<C: ClientLike>(label: &str, client: &C) -> Vec<ConnectionReport> {
    request_report(client);
    let deadline = tokio::time::Instant::now() + REPORT_WAIT;
    loop {
        if let Some(report) = take_report(client) {
            return report;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{label} (fred client {}): its router answered no retention report within \
             {REPORT_WAIT:?}; a router passes at least every unresponsive-check interval, so it \
             has ended (QUIT or abort) or stopped passing",
            client.id()
        );
        tokio::time::sleep(REPORT_POLL).await;
    }
}

/// Whether `conn` is in the idle shape a cut leaves and nothing has touched
/// since: marked idle, no queue allocation, an unallocated write buffer and a
/// fresh [`FRESH_BUFFER_BYTES`] read buffer. Exact from the report: a buffer
/// nothing was consumed from reports its whole allocation as its capacity.
fn in_idle_shape(conn: &ConnectionReport) -> bool {
    conn.idle
        && conn.queue_capacity == 0
        && conn.write_capacity == 0
        && conn.read_capacity == FRESH_BUFFER_BYTES
}

/// The warm rule over one client's retention `report`, its connections
/// expected to carry `expected` limits. Each violation is tagged with its
/// rule, `[L]` or `[V1]` to `[V4]` ([`breaks`] reads the tag):
///
/// - **L**: every connection carries `Some(expected)`.
/// - **V1**: at most one connection is warm.
/// - **V2**: every connection that is drained and not warm is in the idle
///   shape ([`in_idle_shape`]): the sweep cut it, and the cut freed what it
///   held.
/// - **V3**: on a client with one connection, or with read progress on any
///   connection, exactly one connection is warm.
/// - **V4**: the warm connection, once drained, keeps at most
///   `in_flight_commands` queue slots.
///
/// V3 is keyed on read progress, not on "at most one", so it sees both ways
/// the rule can break. With no sweep, no connection is ever marked warm and
/// every drained one is cut at its reply, so the clients that carried traffic
/// have progress and no warm connection. And it needs no list of which
/// clients carried traffic to hold on correct code: the election always names
/// a warm connection once any connection has progress (or a client has one
/// connection), and a client with several connections and no progress has
/// none, every drained connection of it idle-shaped after its first sweep.
/// Every connection warm breaks V1, and a cut that frees nothing breaks V2.
fn warm_rule_violations(
    label: &str,
    expected: RetentionLimits,
    report: &[ConnectionReport],
) -> Vec<String> {
    let mut out = Vec::new();
    let describe = |i: usize| format!("connection {i} ({})", report[i].server);
    for (i, conn) in report.iter().enumerate() {
        if conn.limits != Some(expected) {
            out.push(format!(
                "[L] {label}: {} carries limits {:?}, not {expected:?}",
                describe(i),
                conn.limits
            ));
        }
    }
    let warm: Vec<usize> = (0..report.len()).filter(|&i| report[i].warm).collect();
    if warm.len() > 1 {
        out.push(format!(
            "[V1] {label}: connections {warm:?} of {} are warm; at most one may be",
            report.len()
        ));
    }
    for (i, conn) in report.iter().enumerate() {
        if conn.drained && !conn.warm && !in_idle_shape(conn) {
            out.push(format!(
                "[V2] {label}: {} is drained and not warm, but not in the idle shape: idle {}, \
                 queue {} slots, write {} B, read {} B (want idle, 0, 0, {FRESH_BUFFER_BYTES})",
                describe(i),
                conn.idle,
                conn.queue_capacity,
                conn.write_capacity,
                conn.read_capacity
            ));
        }
    }
    let progressed: Vec<usize> = (0..report.len())
        .filter(|&i| report[i].read_progress)
        .collect();
    if (report.len() == 1 || !progressed.is_empty()) && warm.len() != 1 {
        out.push(format!(
            "[V3] {label}: {} of its {} connections are warm (read progress on {progressed:?}); \
             a client with one connection, or with read progress, has exactly one",
            warm.len(),
            report.len()
        ));
    }
    for &i in &warm {
        let conn = &report[i];
        if conn.drained && conn.queue_capacity > expected.in_flight_commands {
            out.push(format!(
                "[V4] {label}: the warm {} is drained with a {}-slot queue, past the {}-slot limit",
                describe(i),
                conn.queue_capacity,
                expected.in_flight_commands
            ));
        }
    }
    out
}

/// Whether `violations` ([`warm_rule_violations`]) include one of `rule`
/// (`"L"`, `"V1"` … `"V4"`).
fn breaks(violations: &[String], rule: &str) -> bool {
    let tag = format!("[{rule}]");
    violations.iter().any(|v| v.starts_with(&tag))
}

/// The standing negative controls of the warm-rule check (amendment 4 R6), on
/// `client`, which has carried traffic and whose connections are drained:
///
/// - with its sweep `Skipped` (no warm connection, as a build without the
///   sweep), the check must report V3;
/// - on a `multi`-connection client, with `EveryConnectionWarm` (the
///   per-connection rule alone), it must report V1;
/// - back on the sweep `AsBuilt`, it must report nothing.
///
/// So the check is shown to tell a broken warm rule from the built one on
/// whatever store the test runs on, CI's cluster included. The mode is set
/// back to `AsBuilt` before anything is asserted.
async fn assert_warm_rule_discriminates<C: ClientLike>(label: &str, client: &C, multi: bool) {
    let check = |report: &[ConnectionReport], mode: &str| {
        warm_rule_violations(&format!("{label} ({mode})"), GENERAL_LIMITS, report)
    };
    set_sweep_for_test(client, SweepForTest::Skipped);
    let skipped = check(&connection_report(label, client).await, "sweep skipped");
    let every_warm = if multi {
        set_sweep_for_test(client, SweepForTest::EveryConnectionWarm);
        Some(check(
            &connection_report(label, client).await,
            "every connection warm",
        ))
    } else {
        None
    };
    set_sweep_for_test(client, SweepForTest::AsBuilt);
    let as_built = check(&connection_report(label, client).await, "as built");
    assert!(
        breaks(&skipped, "V3"),
        "negative control: {label} with its sweep skipped keeps no warm connection, which the \
         warm-rule check must report as V3; it reported {skipped:?}"
    );
    if let Some(every_warm) = every_warm {
        assert!(
            breaks(&every_warm, "V1"),
            "negative control: {label} with every connection warm must be reported as V1; it \
             reported {every_warm:?}"
        );
    }
    assert!(
        as_built.is_empty(),
        "{label}: back on the sweep as built, the warm rule must hold again: {as_built:?}"
    );
}

/// The rules a report breaks, from [`warm_rule_violations`]'s tags, sorted
/// and each once.
fn broken_rules(violations: &[String]) -> Vec<String> {
    let mut rules: Vec<String> = violations
        .iter()
        .filter_map(|v| {
            v.strip_prefix('[')
                .and_then(|rest| rest.split_once(']'))
                .map(|(rule, _)| rule.to_owned())
        })
        .collect();
    rules.sort();
    rules.dedup();
    rules
}

/// [`warm_rule_violations`] as a decision table (pure, runs everywhere): each
/// row a client's report and the rules it must break, no more and no fewer.
/// Connections are built from three shapes: the warm one (drained, a 256-slot
/// queue, grown buffers, read progress), the idle shape a cut leaves, and the
/// same shape with read progress (a connection that carried traffic and was
/// cut since). The "none warm, read progress on one" row is what tells a
/// client the sweep never ran from one it ran on: it fails with V3 relaxed to
/// "at most one warm".
#[test]
fn warm_rule_violations_tell_one_warm_connection_from_none_or_several_b26e() {
    let limits = GENERAL_LIMITS;
    let idle = ConnectionReport {
        server: "primary:6379".to_owned(),
        limits: Some(limits),
        warm: false,
        idle: true,
        drained: true,
        read_progress: false,
        queued: 0,
        queue_capacity: 0,
        write_capacity: 0,
        read_capacity: FRESH_BUFFER_BYTES,
    };
    let cut = ConnectionReport {
        read_progress: true,
        ..idle.clone()
    };
    let warm = ConnectionReport {
        warm: true,
        idle: false,
        read_progress: true,
        queue_capacity: limits.in_flight_commands,
        write_capacity: 16 * 1024,
        read_capacity: 64 * 1024,
        ..idle.clone()
    };
    let with = |conn: &ConnectionReport, change: fn(&mut ConnectionReport)| {
        let mut conn = conn.clone();
        change(&mut conn);
        conn
    };
    let rows: Vec<(&str, Vec<ConnectionReport>, &[&str])> = vec![
        (
            "one connection, warm with read progress",
            vec![warm.clone()],
            &[],
        ),
        ("one connection, not warm", vec![idle.clone()], &["V3"]),
        (
            "three connections, one warm (a full queue), two idle-shaped",
            vec![cut.clone(), warm.clone(), idle.clone()],
            &[],
        ),
        (
            "three connections, two warm",
            vec![
                with(&warm, |c| c.read_progress = false),
                with(&warm, |c| c.read_progress = false),
                idle.clone(),
            ],
            &["V1"],
        ),
        (
            "none warm, read progress on one",
            vec![idle.clone(), cut.clone(), idle.clone()],
            &["V3"],
        ),
        (
            "none warm, no read progress, all idle-shaped",
            vec![idle.clone(), idle.clone(), idle.clone()],
            &[],
        ),
        (
            "a drained connection not warm and not marked idle",
            vec![warm.clone(), with(&idle, |c| c.idle = false), idle.clone()],
            &["V2"],
        ),
        (
            "a drained connection not warm, keeping a 64-slot queue",
            vec![
                warm.clone(),
                with(&idle, |c| c.queue_capacity = 64),
                idle.clone(),
            ],
            &["V2"],
        ),
        (
            "a drained connection not warm, keeping a 16 KiB read buffer",
            vec![
                warm.clone(),
                with(&idle, |c| c.read_capacity = 16 * 1024),
                idle.clone(),
            ],
            &["V2"],
        ),
        (
            "a drained connection not warm, keeping an 8 KiB write buffer",
            vec![
                warm.clone(),
                with(&idle, |c| c.write_capacity = 8 * 1024),
                idle.clone(),
            ],
            &["V2"],
        ),
        (
            "a connection not warm and not drained, with a 64-slot queue",
            vec![
                warm.clone(),
                with(&idle, |c| {
                    c.drained = false;
                    c.idle = false;
                    c.queue_capacity = 64;
                }),
                idle.clone(),
            ],
            &[],
        ),
        (
            "the warm connection drained with a queue one slot past the limit",
            vec![
                with(&warm, |c| {
                    c.queue_capacity = GENERAL_LIMITS.in_flight_commands + 1
                }),
                idle.clone(),
            ],
            &["V4"],
        ),
        (
            "a connection without limits",
            vec![warm.clone(), with(&idle, |c| c.limits = None)],
            &["L"],
        ),
    ];
    let wrong: Vec<String> = rows
        .iter()
        .filter_map(|(name, report, want)| {
            let violations = warm_rule_violations(name, limits, report);
            let got = broken_rules(&violations);
            (got != *want).then(|| format!("{name}: broke {got:?}, want {want:?}: {violations:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "warm_rule_violations: {wrong:#?}");
}

/// The clients of `client`'s `txn` pool, one per slot: every slot checked
/// out at once (so each checkout is a different slot), each slot's client
/// cloned, then every checkout released. The clones share their client's
/// state, so they serve reports and sweep modes; they never send a command.
///
/// Take them right after connect, before any call a report is to judge, and
/// keep them: each checkout runs the checkout's probe (`DISCARD`, `UNWATCH`)
/// on its slot, which on a cluster goes to an arbitrary primary and gives that
/// connection the newest read progress, so a checkout between a call and a
/// report would move the slot's warm connection to the probe's and the report
/// would show what the probe left, not what the call did.
async fn txn_slot_clients(
    client: &live_edit::storage::client::ValkeyClient,
) -> Vec<fred::clients::Client> {
    let mut held = Vec::with_capacity(client.txn_pool.slots());
    for _ in 0..client.txn_pool.slots() {
        held.push(client.acquire_txn().await);
    }
    held.iter().map(|slot| (**slot).clone()).collect()
}

/// Every general-class client `client` holds now, with its retention report:
/// each member of the data, maintenance and snapshot pools (re-read here, as a
/// member replacement swaps a member's client), each `txn` slot's client and
/// the pub/sub client. The `txn` slots' clients are `txn_slots`, the clones
/// [`txn_slot_clients`] took right after connect: reaching them now through
/// the checkout would run its probe on them first. Each is labelled by the
/// slot that holds it now, and a slot holding a client not among them (a
/// replacement swapped it) fails as a fixture self-check.
async fn general_client_reports(
    client: &live_edit::storage::client::ValkeyClient,
    txn_slots: &[fred::clients::Client],
) -> Vec<(String, Vec<ConnectionReport>)> {
    let mut out = Vec::new();
    for (pool_label, pool) in [
        ("data", &client.pool),
        ("maintenance", &client.maintenance_pool),
        ("snapshot", &client.snapshot_pool),
    ] {
        for (i, member) in pool.clients().iter().enumerate() {
            let label = format!("{pool_label} member {i}");
            let report = connection_report(&label, member).await;
            out.push((label, report));
        }
    }
    for (i, id) in client.txn_slot_client_ids_for_test().iter().enumerate() {
        let slot = txn_slots
            .iter()
            .find(|slot| slot.id() == id.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "fixture self-check: txn slot {i} holds client {id}, not one of the clients \
                     taken at connect: a replacement swapped it"
                )
            });
        let label = format!("txn slot {i}");
        let report = connection_report(&label, slot).await;
        out.push((label, report));
    }
    let label = "pub/sub client".to_owned();
    let report = connection_report(&label, &client.subscriber).await;
    out.push((label, report));
    out
}

/// One line per client of `reports`: its connections, which is warm, which
/// have read progress, and each one's queue slots.
fn describe_reports(reports: &[(String, Vec<ConnectionReport>)]) -> String {
    reports
        .iter()
        .map(|(label, report)| {
            let flags: Vec<String> = report
                .iter()
                .map(|c| {
                    format!(
                        "{}{}{}q{}",
                        if c.warm { "W" } else { "-" },
                        if c.idle { "I" } else { "-" },
                        if c.read_progress { "P" } else { "-" },
                        c.queue_capacity
                    )
                })
                .collect();
            format!("{label}: [{}]", flags.join(" "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Test 11's bound, formula (B) for `pools` on `topology` without K, the
/// member-replacement clients (`PoolSizes::replacement_clients`).
/// `at_rest_bound_bytes` counts K always, so its bound holds through a
/// replacement; no member replacement runs in test 11 (its self-check), so
/// K's clients never exist there, and counting them would loosen the bound by
/// K general clients: 8 beside the 5 of a pool of 1 (the retiring cap's 4 and
/// one connecting client for each of its four pools), 2.6 times the general
/// term. Built from the public functions: one general client's term is what
/// the bound gains when the data pool gains a member, which leaves K as it
/// is, and K of those come off.
fn at_rest_bound_without_replacements(pools: PoolSizes, topology: Topology) -> u64 {
    let one_more = PoolSizes {
        data: pools.data + 1,
        ..pools
    };
    assert_eq!(
        one_more.replacement_clients(),
        pools.replacement_clients(),
        "fixture self-check: one more data member adds no member-replacement client"
    );
    let bound = at_rest_bound_bytes(pools, topology);
    let general_client = at_rest_bound_bytes(one_more, topology) - bound;
    bound - pools.replacement_clients() as u64 * general_client
}

/// The fred client id of every pooled general client of `client`: each data,
/// maintenance and snapshot member's and each `txn` slot's. A member
/// replacement changes one, so test 11 compares them across its calls.
fn pooled_client_ids(client: &live_edit::storage::client::ValkeyClient) -> Vec<String> {
    let mut ids: Vec<String> = [
        &client.pool,
        &client.maintenance_pool,
        &client.snapshot_pool,
    ]
    .iter()
    .flat_map(|pool| pool.clients().into_iter().map(|m| m.id().to_owned()))
    .collect();
    ids.extend(client.txn_slot_client_ids_for_test());
    ids
}

/// One post-release case.
struct PostReleaseCase {
    name: String,
    path: Path,
    route: Route,
    /// The client's data and txn pool size.
    pool_size: u32,
    /// One fixture per call, in order.
    calls: Vec<Fixture>,
    /// The cluster to run on (a `redis-cluster://` URL, [`cluster_url`]);
    /// `None` runs on `TEST_VALKEY_URL`'s standalone store.
    cluster_url: Option<String>,
    /// Where the calls' objects go and what bounds them.
    shape: CaseShape,
}

/// Where a post-release case's calls land and what bounds the heap they
/// leave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaseShape {
    /// Each call's object wherever its id puts it; bounded by
    /// [`at_rest_bound_without_replacements`] for the client's pools and
    /// topology (test 11).
    AtRest,
    /// Test 11c, on a cluster: call k's object on primary (k − 1) mod P, in
    /// the routing table's order; bounded by [`OneWarmFigures::bound`] for
    /// the [`UPLOAD_PATH_TOUCHED_CLIENTS`] clients the calls load.
    OneWarmPerTouchedClient,
}

impl CaseShape {
    /// What the case's bound is, for its report header.
    fn bound_name(self) -> &'static str {
        match self {
            CaseShape::AtRest => {
                "at-rest bound (at_rest_bound_bytes without the member-replacement clients)"
            }
            CaseShape::OneWarmPerTouchedClient => {
                "one-warm-connection bound (OneWarmFigures::bound: the touched clients at one \
                 warm connection each, the untouched clients at the idle shape)"
            }
        }
    }

    /// What a failure most likely means, for its message.
    fn overrun_cause(self) -> &'static str {
        match self {
            CaseShape::AtRest => {
                "at_rest_bound_bytes counts every pooled client at its class limits \
                 (storage::connection_buffers); a connection left out of the limits, or a holder \
                 no longer cut back once it drains, keeps its largest load. A [L] or [V1]-[V4] \
                 line is the warm rule read off a client's retention report \
                 (warm_rule_violations): its limits, more than one warm connection, a drained \
                 connection not cut to the idle shape, no warm connection (no sweep), or a warm \
                 queue past the limit"
            }
            CaseShape::OneWarmPerTouchedClient => {
                "OneWarmFigures::bound counts the clients the calls load (data and txn) at one \
                 warm connection each, clustered_client_bound(P), and every other general \
                 client at the idle shape; past it, a touched client kept more than its one warm \
                 connection's limits, or the connection buffer limits are off, and the teardown \
                 lines below say which client held it. A [L] or [V1]-[V4] line is the warm rule \
                 read off a client's retention report (warm_rule_violations; retention::sweep / \
                 elect): more than one warm connection, a drained connection not cut to the idle \
                 shape, no warm connection (no sweep), or a warm queue past the limit"
            }
        }
    }
}

impl PostReleaseCase {
    /// The connections one call can carry a large load on: the data pool's,
    /// plus one more — the txn slot the WATCH path checks out (the first free
    /// one, so the same one while the calls run one at a time) or the
    /// snapshot lane's one connection (`common::test_config`'s
    /// `snapshot_pool_size`) a cold start reads on. Once every one of them has
    /// carried a large call, more calls must not grow the heap
    /// ([`PostRelease::plateau`]). On a cluster each of these is a client
    /// with a connection per primary, of which only its warm one keeps the
    /// class limits, so the count is the same.
    fn touchable(&self) -> usize {
        self.pool_size as usize + 1
    }

    /// Whether each call's layer is deleted after the call, before its live
    /// bytes are read: a cold start stores its object in the structural cache,
    /// which keeps it until the object is retired, and deleting the layer
    /// retires it.
    fn deletes_layers(&self) -> bool {
        self.route == Route::CachedCold
    }
}

/// The live bytes after one teardown step, above the reference taken before
/// the first call.
struct TeardownStep {
    step: &'static str,
    live: i64,
    glibc: i64,
    tracked_count: usize,
    tracked_bytes: u64,
}

/// Wait [`SETTLE`], then read the live bytes after `step`.
async fn teardown_step(step: &'static str, reference: (i64, i64)) -> TeardownStep {
    tokio::time::sleep(SETTLE).await;
    let (tracked_count, tracked_bytes) = tracked_totals();
    TeardownStep {
        step,
        live: live_now() - reference.0,
        glibc: glibc_now() - reference.1,
        tracked_count,
        tracked_bytes,
    }
}

/// One call of a post-release case.
struct CallRow {
    call: usize,
    /// Live bytes when the handler returned, above the call's start.
    returned: i64,
    /// Live bytes after the layer's deletion, if the case deletes it, and
    /// [`SETTLE`], above the call's start.
    settled: i64,
    /// Live bytes then, above the reference taken before the first call.
    total: i64,
    /// glibc's in-use bytes then, above the reference.
    glibc: i64,
    /// The upload's wall time, from the handler's call to its return.
    wall: std::time::Duration,
}

/// What [`measure_post_release`] found.
struct PostRelease {
    name: String,
    /// The bound every call's heap must stay within ([`CaseShape`]):
    /// [`at_rest_bound_without_replacements`] for the case's client on
    /// [`CaseShape::AtRest`]; [`OneWarmFigures::bound`] on
    /// [`CaseShape::OneWarmPerTouchedClient`].
    bound: u64,
    /// The case's [`CaseShape`], which says what `bound` is and what a call
    /// past it most likely means.
    shape: CaseShape,
    rows: Vec<CallRow>,
    /// The warm rule's violations ([`warm_rule_violations`]) over every
    /// general client's retention report, read after the calls drained.
    warm_rule: Vec<String>,
    /// [`PostReleaseCase::touchable`].
    touchable: usize,
    /// Each teardown step and the live bytes above the reference after it.
    teardown: Vec<(&'static str, i64)>,
}

/// The most the heap may grow over a case's last calls, once it has made as
/// many calls as it has connections to use: what one general-class
/// connection keeps once drained, which leaves room for one connection
/// reached late (on test 11's cases, by the data pool's round-robin). A
/// residue that grows by more than that over those calls fails it.
fn plateau_tolerance() -> i64 {
    (GENERAL_LIMITS.kept_queue_bytes()
        + GENERAL_LIMITS.kept_write_bytes()
        + GENERAL_LIMITS.kept_read_bytes()) as i64
}

impl PostRelease {
    /// The heap's growth from the first call after as many calls as the case
    /// has connections to use (call `touchable + 1`) to the last call; `None`
    /// when the case has too few calls for one.
    fn plateau(&self) -> Option<i64> {
        let from = self.rows.get(self.touchable)?;
        let last = self.rows.last()?;
        (last.call > from.call).then(|| last.total - from.total)
    }

    /// Each call that left the heap above the bound, a plateau past its
    /// tolerance, and each break of the warm rule, described.
    fn violations(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .rows
            .iter()
            .filter(|r| r.total > self.bound as i64)
            .map(|r| {
                format!(
                    "{}: call {} left {} B live above the reference, past its bound of {} B",
                    self.name, r.call, r.total, self.bound
                )
            })
            .collect();
        if let Some(growth) = self.plateau().filter(|&g| g > plateau_tolerance()) {
            out.push(format!(
                "{}: the heap grew {growth} B from call {} to call {}, past the {} B one \
                 connection keeps: it grows with the calls",
                self.name,
                self.touchable + 1,
                self.rows.len(),
                plateau_tolerance()
            ));
        }
        out.extend(self.warm_rule.iter().cloned());
        out
    }

    /// What a failure message needs to be diagnosed from the log alone: the
    /// heap above the reference after each call, and what each teardown step
    /// freed — after the service is dropped, each client quit in turn, so a
    /// client's step is about what that client kept (its connections'
    /// buffers and its own structures).
    fn diagnosis(&self) -> String {
        let calls: Vec<String> = self
            .rows
            .iter()
            .map(|r| format!("call {}: {} B", r.call, r.total))
            .collect();
        let freed: Vec<String> = self
            .teardown
            .windows(2)
            .map(|w| format!("{}: {} B", w[1].0, w[0].1 - w[1].1))
            .collect();
        format!(
            "heap above the reference after each call [{}]; freed by each teardown step [{}]",
            calls.join(", "),
            freed.join(", ")
        )
    }
}

/// Run `case`'s uploads one after another through one service on one fresh
/// client built by the production `connect`, reading the live bytes each
/// leaves and in total, then tear the service and the client down a part at
/// a time, reporting what each part freed. Between the calls and the
/// teardown, reads every general client's retention report for the warm rule.
/// Prints the figures and the reports; asserts only its fixture self-checks
/// and, on [`CaseShape::OneWarmPerTouchedClient`], the warm rule's negative
/// controls ([`PostRelease::violations`] says what broke).
async fn measure_post_release(case: PostReleaseCase) -> PostRelease {
    let touchable = case.touchable();
    let deletes_layers = case.deletes_layers();
    let PostReleaseCase {
        name,
        path,
        route,
        pool_size,
        calls,
        cluster_url,
        shape,
    } = case;
    // B26E_TRACK=1 runs the allocation tracker. Its own map is uncounted, so
    // outside the Rust figures, but inside glibc's in-use figure.
    let track = std::env::var("B26E_TRACK").is_ok_and(|v| v == "1");
    let mut config = common::test_config();
    let tv = match &cluster_url {
        Some(url) => {
            config.valkey_url = url.clone();
            TestValkey::setup_with_url(url.clone(), pool_size).await
        }
        None => TestValkey::setup().await,
    };
    config.valkey_pool_size = pool_size;
    // No subscription runs here, so no stream reader connects (checked below):
    // R, the readers that have connected, is 0.
    let pools = PoolSizes {
        readers: 0,
        ..PoolSizes::from_config(&config)
    };
    tokio::time::sleep(SETTLE).await;
    let setup_reference = (live_now(), glibc_now());
    if track {
        start_tracking(POST_RELEASE_BACKTRACE_MIN);
    }
    let client = connect_test_client(&config)
        .await
        .expect("the case's own client connects");
    // Each txn slot's client, for its retention report after the calls: taken
    // now, before any call, since a checkout probes its slot
    // ([`txn_slot_clients`]). Dropped before the teardown.
    let txn_slots = txn_slot_clients(&client).await;
    // The bound's topology is the store's: on a cluster, P is the primary
    // count of the data pool's routing table.
    let topology = topology_of(&client.pool.clients()[0]);
    assert_eq!(
        topology.clustered(),
        cluster_url.is_some(),
        "fixture self-check: {name}: the client is clustered exactly when the case runs on a \
         cluster"
    );
    let bound = match shape {
        CaseShape::AtRest => at_rest_bound_without_replacements(pools, topology),
        CaseShape::OneWarmPerTouchedClient => {
            assert!(
                topology.clustered() && topology.primaries() >= 2,
                "fixture self-check: {name}: test 11c runs on a cluster of two or more primaries, \
                 so each client it checks has more than one connection to choose a warm one \
                 among; got {topology:?}"
            );
            assert_eq!(
                (
                    pools.data,
                    pools.txn,
                    client.pool.clients().len(),
                    client.txn_pool.slots()
                ),
                (1, 1, 1, 1),
                "fixture self-check: {name}: the data and txn pools have one member each, so \
                 the calls load exactly those two clients"
            );
            assert_eq!(
                calls.len(),
                topology.primaries() + 2,
                "fixture self-check: {name}: P + 2 calls"
            );
            OneWarmFigures::of(
                pools.general_clients(),
                UPLOAD_PATH_TOUCHED_CLIENTS,
                topology.primaries(),
            )
            .bound
        }
    };
    let svc = isolated_service(&tv, route, client.clone());

    // Unmeasured: load the scripts and create the metric series through
    // this service, as `warm_up` does.
    let mut warmups = vec![Fixture::header_only()];
    if path == Path::UpdateSlow {
        warmups.push(Fixture::new(
            "warmup-structural",
            Feeds::Mixed,
            OwnedDelta {
                struct_commands: children(1, false),
                ..Default::default()
            },
            &[],
        ));
    }
    for (i, warmup) in warmups.iter().enumerate() {
        let input = upload_frames(&tv, path, warmup, &format!("{name}-warmup-{i}")).await;
        upload_once(&svc, path, &input.frames, &name, 0).await;
    }
    tokio::time::sleep(SETTLE).await;
    eprintln!(
        "== b.26e post-release {name}: the client, the service and the warm-up left {} B live \
         (glibc {} B) ==",
        live_now() - setup_reference.0,
        glibc_now() - setup_reference.1
    );
    if track {
        stop_recording();
        report_tracked(&format!("{name}: client + service + warm-up"), 6);
        end_tracking();
    }

    // Every call's input is built before the reference is read, and kept
    // alive until the end.
    let mut inputs = Vec::with_capacity(calls.len());
    match shape {
        CaseShape::AtRest => {
            for (i, fixture) in calls.iter().enumerate() {
                inputs.push(upload_frames(&tv, path, fixture, &format!("{name}-{}", i + 1)).await);
            }
        }
        CaseShape::OneWarmPerTouchedClient => {
            inputs = placed_inputs(&tv, &client, &calls, &name).await;
        }
    }
    let fixture_name = calls.first().map_or("none", |f| f.name);
    let body = calls.first().map_or(0, |f| f.ompf.len());
    drop(calls);
    let mut rows: Vec<CallRow> = Vec::with_capacity(inputs.len());
    let mut steps: Vec<TeardownStep> = Vec::with_capacity(8);
    // Which client each pooled member and `txn` slot holds, to show no member
    // replacement ran during the calls (the bound leaves its clients out).
    let ids_before = pooled_client_ids(&client);
    tokio::time::sleep(SETTLE).await;

    // B26E_BACKTRACE_MIN=<bytes> backtraces smaller allocations too.
    let backtrace_min = std::env::var("B26E_BACKTRACE_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(POST_RELEASE_BACKTRACE_MIN);
    let reference = (live_now(), glibc_now());
    if track {
        start_tracking(backtrace_min);
    }
    for (i, input) in inputs.iter().enumerate() {
        let before = live_now();
        let started = std::time::Instant::now();
        upload_once(&svc, path, &input.frames, &name, i + 1).await;
        let wall = started.elapsed();
        let returned = live_now() - before;
        if deletes_layers {
            svc.layer_store
                .delete_layer(&input.session_id, "layer", Arc::clone(&svc.ext_value_store))
                .await
                .unwrap_or_else(|e| panic!("{name} call {}: the layer deletes: {e:?}", i + 1));
        }
        tokio::time::sleep(SETTLE).await;
        let after = live_now();
        rows.push(CallRow {
            call: i + 1,
            returned,
            settled: after - before,
            total: after - reference.0,
            glibc: glibc_now() - reference.1,
            wall,
        });
    }
    if track {
        stop_recording();
    }
    assert_eq!(
        client.stream_mux.connected_readers(),
        0,
        "fixture self-check: no stream reader connected, so the bound counts none"
    );
    assert_eq!(
        (
            pooled_client_ids(&client),
            client.member_replacement_tasks_for_test()
        ),
        (ids_before, 0),
        "fixture self-check: {name}: no member replacement ran during the calls, so the bound \
         counts no member-replacement client"
    );

    // The warm rule, off every general client's retention report now that the
    // calls have drained. Outside every measured window: the rows are read,
    // and the reports are dropped before the teardown's first live-bytes read,
    // with no request left pending.
    let reports = general_client_reports(&client, &txn_slots).await;
    assert_eq!(
        reports.len(),
        pools.general_clients(),
        "fixture self-check: {name}: a report from every general client the bound counts"
    );
    let warm_rule: Vec<String> = reports
        .iter()
        .flat_map(|(label, report)| {
            warm_rule_violations(&format!("{name}: {label}"), GENERAL_LIMITS, report)
        })
        .collect();
    eprintln!(
        "== b.26e post-release {name}: retention reports after the calls (W warm, I idle, P read \
         progress, q queue slots): {} ==",
        describe_reports(&reports)
    );
    if shape == CaseShape::OneWarmPerTouchedClient {
        // Both regressions are visible only on the clients the calls loaded,
        // and only once every one of their connections has read progress.
        for touched in ["data member 0", "txn slot 0"] {
            let (_, report) = reports
                .iter()
                .find(|(label, _)| label == touched)
                .unwrap_or_else(|| panic!("fixture self-check: {name}: a report of the {touched}"));
            assert!(
                report.len() == topology.primaries() && report.iter().all(|c| c.read_progress),
                "fixture self-check: {name}: the {touched} has a connection to each of the {} \
                 primaries, every one with read progress (each carried a call): {report:?}",
                topology.primaries()
            );
        }
    }
    drop(reports);
    // The negative controls run only on a case that kept the warm rule: one
    // that broke it fails with its own violations, asserted with the case's
    // diagnosis by `assert_post_release_case`, not with a control's.
    if shape == CaseShape::OneWarmPerTouchedClient {
        if warm_rule.is_empty() {
            let data_member = client.pool.clients()[0].clone();
            assert_warm_rule_discriminates(&format!("{name}: data member 0"), &data_member, true)
                .await;
            assert_warm_rule_discriminates(&format!("{name}: txn slot 0"), &txn_slots[0], true)
                .await;
        } else {
            eprintln!(
                "== b.26e post-release {name}: negative controls skipped: the case broke the \
                 warm rule itself =="
            );
        }
    }
    drop(txn_slots);

    let mut measured = PostRelease {
        name: name.clone(),
        bound,
        shape,
        rows,
        warm_rule,
        touchable,
        teardown: Vec::new(),
    };
    eprintln!(
        "== b.26e post-release {name}: {path:?} on {route:?}, fixture {fixture_name} ({body} B \
         body), {} calls, pool size {pool_size}{}; {} {bound} B ({pools:?}) ==",
        measured.rows.len(),
        match (deletes_layers, shape) {
            (true, _) => ", each call's layer deleted after it",
            (false, CaseShape::OneWarmPerTouchedClient) => {
                ", call k's object on primary (k - 1) mod P"
            }
            (false, CaseShape::AtRest) => "",
        },
        shape.bound_name()
    );
    eprintln!(
        "{:>5} {:>14} {:>14} {:>14} {:>14}",
        "call", "left@return", "left@settle", "total-left", "glibc-total"
    );
    for r in &measured.rows {
        eprintln!(
            "{:>5} {:>14} {:>14} {:>14} {:>14}",
            r.call, r.returned, r.settled, r.total, r.glibc
        );
    }
    for r in &measured.rows {
        eprintln!(
            "== b.26e post-release {name}: call {} wall time {:.3} s ==",
            r.call,
            r.wall.as_secs_f64()
        );
    }
    match measured.plateau() {
        Some(growth) => eprintln!(
            "plateau: {growth} B from call {} to call {} (tolerance {} B)",
            touchable + 1,
            measured.rows.len(),
            plateau_tolerance()
        ),
        None => eprintln!(
            "plateau: not checked, {} calls for {touchable} connections the case can use",
            measured.rows.len()
        ),
    }
    if track {
        report_tracked(&format!("{name} after the calls"), 10);
    }

    steps.push(teardown_step("after the calls", reference).await);
    drop(svc);
    steps.push(teardown_step("service dropped", reference).await);
    let _ = client.snapshot_pool.quit().await;
    steps.push(teardown_step("snapshot pool quit", reference).await);
    let _ = client.maintenance_pool.quit().await;
    steps.push(teardown_step("maintenance pool quit", reference).await);
    let _ = client.subscriber.quit().await;
    steps.push(teardown_step("pub/sub client quit", reference).await);
    let _ = client.pool.quit().await;
    steps.push(teardown_step("data pool quit", reference).await);
    client.quit().await;
    steps.push(teardown_step("client quit (txn, readers)", reference).await);
    drop(client);
    steps.push(teardown_step("client dropped", reference).await);

    eprintln!(
        "{:<28} {:>14} {:>14} {:>14} {:>9}",
        "teardown", "live-left", "glibc-left", "tracked-left", "tracked#"
    );
    for s in &steps {
        eprintln!(
            "{:<28} {:>14} {:>14} {:>14} {:>9}",
            s.step, s.live, s.glibc, s.tracked_bytes, s.tracked_count
        );
    }
    if track {
        report_tracked(&format!("{name} after the teardown"), 6);
        end_tracking();
    }
    drop(inputs);
    measured.teardown = steps.iter().map(|s| (s.step, s.live)).collect();
    measured
}

/// Test 11c's inputs: call k's base layer (made through `tv`'s own client)
/// on an object whose `{object_id}` slot is owned by primary (k − 1) mod P,
/// in `client`'s data member's routing-table order. Each is found by making
/// empty base layers until one lands there, at most
/// [`OBJECTS_TO_FIND_ANOTHER_PRIMARY`] tries per call.
async fn placed_inputs(
    tv: &TestValkey,
    client: &live_edit::storage::client::ValkeyClient,
    calls: &[Fixture],
    name: &str,
) -> Vec<UploadInput> {
    use fred::interfaces::ClusterInterface;
    let member = client.pool.clients()[0].clone();
    let primaries = member
        .cached_cluster_state()
        .expect("a connected cluster client has its routing")
        .unique_primary_nodes();
    let mut inputs = Vec::with_capacity(calls.len());
    for (i, fixture) in calls.iter().enumerate() {
        let want = &primaries[i % primaries.len()];
        let mut placed = None;
        for attempt in 0..OBJECTS_TO_FIND_ANOTHER_PRIMARY {
            let label = format!("{name}-{}-{attempt}", i + 1);
            let (session_id, object_id) = base_layer(tv, &label, fixture.base).await;
            if stream_owner(&member, tv.key_schema(), object_id) == *want {
                placed = Some((session_id, object_id));
                break;
            }
        }
        let (session_id, object_id) = placed.unwrap_or_else(|| {
            panic!(
                "fixture self-check: {name}: no object of {OBJECTS_TO_FIND_ANOTHER_PRIMARY} landed \
                 on {want} for call {}",
                i + 1
            )
        });
        assert_eq!(
            stream_owner(&member, tv.key_schema(), object_id),
            *want,
            "fixture self-check: {name}: call {}'s object is on primary {} mod {}",
            i + 1,
            i,
            primaries.len()
        );
        inputs.push(update_input(session_id, object_id, fixture));
    }
    inputs
}

/// `calls` uploads of [`DENSE`]-style field-dense content of `fields` fields.
fn field_dense_calls(fields: usize, calls: usize) -> Vec<Fixture> {
    (0..calls)
        .map(|_| {
            Fixture::new(
                "field-dense",
                Feeds::Elements,
                OwnedDelta {
                    set_fields: (0..fields)
                        .map(|i| field(format!("f{i}"), (i as u64).to_le_bytes().to_vec()))
                        .collect(),
                    ..Default::default()
                },
                &[],
            )
        })
        .collect()
}

/// `calls` uploads of one external value of `len` bytes each, distinct per
/// call (so no call finds its value already stored).
fn one_blob_calls(len: usize, calls: usize) -> Vec<Fixture> {
    (0..calls)
        .map(|call| {
            let blob = blobs(1, len, call as u8 ^ 0xa5).remove(0);
            Fixture::new(
                "one-blob",
                Feeds::Ext,
                OwnedDelta {
                    set_fields: vec![external_field("blob0".to_owned(), &blob, 1)],
                    ..Default::default()
                },
                &[blob.as_slice()],
            )
        })
        .collect()
}

/// `calls` uploads of [`VALUES`] inline values of [`VALUE_LEN`] bytes.
fn value_heavy_calls(calls: usize) -> Vec<Fixture> {
    (0..calls)
        .map(|_| {
            Fixture::new(
                "value-heavy",
                Feeds::Bytes,
                OwnedDelta {
                    set_fields: value_fields(VALUES, VALUE_LEN),
                    ..Default::default()
                },
                &[],
            )
        })
        .collect()
}

/// The case `spec` names: `<kind>:<size>:<pool size>:<calls>`, where kind is
/// `watch-field-dense` or `fast-field-dense` (size = fields), `fast-blob` or
/// `create-blob` (size = blob bytes), `fast-value-heavy` (size unused), or
/// `cold-children` (one new child on a base of size children, each call a new
/// object on the cached route's cold start, its layer deleted after it).
fn post_release_case(spec: &str) -> PostReleaseCase {
    let parts: Vec<&str> = spec.split(':').collect();
    let [kind, size, pool_size, calls] = parts[..] else {
        panic!("B26E_POST_RELEASE_CASE `{spec}`: want <kind>:<size>:<pool size>:<calls>");
    };
    let size: usize = size.parse().expect("size");
    let pool_size: u32 = pool_size.parse().expect("pool size");
    let n: usize = calls.parse().expect("calls");
    let (path, route, calls) = match kind {
        "watch-field-dense" => (Path::UpdateSlow, Route::Watch, field_dense_calls(size, n)),
        "fast-field-dense" => (Path::UpdateFast, Route::Default, field_dense_calls(size, n)),
        "fast-blob" => (Path::UpdateFast, Route::Default, one_blob_calls(size, n)),
        "create-blob" => (Path::Create, Route::Default, one_blob_calls(size, n)),
        "fast-value-heavy" => (Path::UpdateFast, Route::Default, value_heavy_calls(n)),
        "cold-children" => (
            Path::UpdateSlow,
            Route::CachedCold,
            (0..n)
                .map(|_| {
                    Fixture::new("one-new-child", Feeds::Mixed, one_new_child(size), &[])
                        .on_base(Base::Children(size))
                })
                .collect(),
        ),
        _ => panic!("B26E_POST_RELEASE_CASE: unknown kind `{kind}`"),
    };
    PostReleaseCase {
        name: spec.to_owned(),
        path,
        route,
        pool_size,
        calls,
        cluster_url: None,
        shape: CaseShape::AtRest,
    }
}

/// Test 11 (b.26e F): `spec`'s calls leave the heap within the at-rest bound
/// of their client's pools after every call, and stop growing it once every
/// connection the case can use has carried a large call; and once they have
/// drained, every general client keeps the warm rule
/// ([`warm_rule_violations`] over its retention report). On a standalone
/// store every client has one connection, which the rule then requires to
/// be warm: a direct check that the sweep runs on every client's router,
/// which fails on V3 for every client without it. `spec` must have at least
/// two calls past [`PostReleaseCase::touchable`], so the plateau is checked.
async fn assert_post_release_within_the_at_rest_bound(spec: &str) {
    let _serial = measure_lock().await;
    assert_post_release_case(post_release_case(spec)).await;
}

/// Test 11's cluster cases: `spec` as [`assert_post_release_within_the_at_rest_bound`]
/// runs it, on the cluster `LIVE_EDIT_TEST_CLUSTER_URL` names. They hold
/// formula (B) without its member-replacement clients at the routing table's
/// primary count ([`at_rest_bound_without_replacements`] on
/// `Topology::cluster(P)`, [`topology_of`]) after every call, and the
/// plateau: bound checks, which fail with the connection buffer limits
/// reverted or their wiring removed (each touched connection then keeps its
/// in-flight queue at the pipeline's size: 65,536 slots, 24.6 MB, on the
/// WATCH cases).
///
/// The warm rule they check directly, as every case of test 11 does: each
/// general client's retention report must show one warm connection on a
/// client with read progress and every other drained connection in the idle
/// shape ([`warm_rule_violations`]), at pools of 2, 4 and 8 and on every
/// upload path here. Every connection warm then fails V1, and no sweep fails
/// V3 on every client the calls loaded. The bytes cannot see either: (B)
/// counts every general client at its class limits, so the clients the cases
/// barely use leave more slack than every connection warm adds, and no sweep
/// keeps less.
///
/// Prints a `SKIP:` line and passes without a cluster.
async fn assert_cluster_post_release_within_the_at_rest_bound(spec: &str) {
    let _serial = measure_lock().await;
    let Some(url) = cluster_url(&format!("cluster post-release {spec}")) else {
        return;
    };
    assert_post_release_case(PostReleaseCase {
        cluster_url: Some(url),
        ..post_release_case(spec)
    })
    .await;
}

/// The body of the assertions of tests 11 and 11c, on `case`'s store and
/// against its [`CaseShape`]'s bound.
async fn assert_post_release_case(case: PostReleaseCase) {
    let spec = case.name.clone();
    assert!(
        case.calls.len() >= case.touchable() + 2,
        "fixture self-check: {spec}: {} calls leave no plateau past the {} connections the \
         case can use",
        case.calls.len(),
        case.touchable()
    );
    let measured = measure_post_release(case).await;
    let violations = measured.violations();
    assert!(
        violations.is_empty(),
        "the Valkey connections kept more than their limits allow, or broke the warm rule, once \
         the calls drained: {violations:?}. {}. {}",
        measured.shape.overrun_cause(),
        measured.diagnosis()
    );
}

/// Test 11: field-dense content of [`DENSE`] fields on the WATCH path, pool
/// of 2. Before the connection buffer limits each call left about 24.6 MB in
/// the in-flight queue of the data connection that read its pre-image and in
/// the txn connection's (75.3 MB after the pool was covered).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_watch_field_dense_on_a_pool_of_2_stays_within_the_at_rest_bound_b26e() {
    assert_post_release_within_the_at_rest_bound(&format!("watch-field-dense:{DENSE}:2:5")).await;
}

/// Test 11: as above, pool of 4 (124.75 MB before the limits).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_watch_field_dense_on_a_pool_of_4_stays_within_the_at_rest_bound_b26e() {
    assert_post_release_within_the_at_rest_bound(&format!("watch-field-dense:{DENSE}:4:7")).await;
}

/// Test 11: as above, pool of 8 (223.6 MB before the limits).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_watch_field_dense_on_a_pool_of_8_stays_within_the_at_rest_bound_b26e() {
    assert_post_release_within_the_at_rest_bound(&format!("watch-field-dense:{DENSE}:8:11")).await;
}

/// Tests 11 and 14: one 16 MiB external value per call, ten calls, pool of
/// 4. Each value is written in one frame, so the write buffer of the data
/// connection that sent it grows to hold it (33.4 MB kept before the limits,
/// on the two connections the round-robin sent them to). A flush leaves that
/// buffer empty with its allocation whole and its `capacity()` near zero, so
/// this is the case a write-buffer rule keyed on `BytesMut::capacity()` never
/// cuts back (test 14): the rule must judge the buffer by what it held.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_one_16_mib_blob_per_call_stays_within_the_at_rest_bound_b26e() {
    assert_post_release_within_the_at_rest_bound(&format!("fast-blob:{}:4:10", 16 * MIB)).await;
}

/// Test 11: [`VALUES`] inline values of [`VALUE_LEN`] bytes per call, ten
/// calls, pool of 4: the fast path's script call and relay together, about
/// 9.8 MB in one write buffer (39.2 MB kept before the limits).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_value_heavy_stays_within_the_at_rest_bound_b26e() {
    assert_post_release_within_the_at_rest_bound("fast-value-heavy:0:4:10").await;
}

/// Test 11: one new child on a base of [`DENSE`] children per call, each a
/// new object read whole by the cached route's cold start, each call's layer
/// deleted after it so its structural-cache entry goes. Pool of 1, the
/// smallest bound (five general clients, no reader).
///
/// This case guards the bound on the cold start's read path; it does not by
/// itself catch the limits' removal. The read is windowed, so no reply is
/// large, and without the limits the connections it reads on keep only their
/// grown queues and buffers: 1.5 MB at a pool of 1, and 2.4–12 MB after four
/// calls at pools of 4 and 8 (M6, 2026-10-06, varying run to run with which
/// connections carry the reads) — under the bound at every pool measured. The
/// other cases fail without the limits.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_cold_children_stays_within_the_at_rest_bound_b26e() {
    assert_post_release_within_the_at_rest_bound(&format!("cold-children:{DENSE}:1:4")).await;
}

/// Test 11 on a cluster: the WATCH field-dense case at a pool of 2, against
/// the bound at the cluster's primary count
/// ([`assert_cluster_post_release_within_the_at_rest_bound`]). Cluster-gated.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_cluster_watch_field_dense_on_a_pool_of_2_stays_within_the_at_rest_bound_b26e()
{
    assert_cluster_post_release_within_the_at_rest_bound(&format!("watch-field-dense:{DENSE}:2:5"))
        .await;
}

/// Test 11 on a cluster: as above, pool of 4. Cluster-gated.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_cluster_watch_field_dense_on_a_pool_of_4_stays_within_the_at_rest_bound_b26e()
{
    assert_cluster_post_release_within_the_at_rest_bound(&format!("watch-field-dense:{DENSE}:4:7"))
        .await;
}

/// Test 11 on a cluster: as above, pool of 8. Cluster-gated.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_cluster_watch_field_dense_on_a_pool_of_8_stays_within_the_at_rest_bound_b26e()
{
    assert_cluster_post_release_within_the_at_rest_bound(&format!(
        "watch-field-dense:{DENSE}:8:11"
    ))
    .await;
}

/// Tests 11 and 14 on a cluster: one 16 MiB external value per call, ten
/// calls, pool of 4. Cluster-gated.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_cluster_one_16_mib_blob_per_call_stays_within_the_at_rest_bound_b26e() {
    assert_cluster_post_release_within_the_at_rest_bound(&format!("fast-blob:{}:4:10", 16 * MIB))
        .await;
}

/// Test 11 on a cluster: the value-heavy fast-path upload, ten calls, pool
/// of 4. Cluster-gated.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_cluster_value_heavy_stays_within_the_at_rest_bound_b26e() {
    assert_cluster_post_release_within_the_at_rest_bound("fast-value-heavy:0:4:10").await;
}

/// Test 11 on a cluster: the 20,000-child cold start, each call's layer
/// deleted after it, pool of 1. Cluster-gated.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_cluster_cold_children_stays_within_the_at_rest_bound_b26e() {
    assert_cluster_post_release_within_the_at_rest_bound(&format!("cold-children:{DENSE}:1:4"))
        .await;
}

/// The clients test 11c's calls load: the data pool's one member, which reads
/// each call's pre-image, and the txn pool's one member, which runs its WATCH
/// transaction. The maintenance, snapshot and pub/sub clients are untouched.
const UPLOAD_PATH_TOUCHED_CLIENTS: usize = 2;

// Test 11c's field-dense WATCH upload must put more commands in flight on
// each touched connection than the general in-flight limit, so each grows its
// queue past the limit and V4 (the warm connection's queue cut back to the
// limit once drained) has something to check.
const _: () = assert!(DENSE > GENERAL_LIMITS.in_flight_commands);

/// Test 11c's figures for `general` general clients, `touched` of them
/// loaded on each of `primaries` primaries, all from the constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OneWarmFigures {
    /// What the clients keep with one warm connection each:
    /// `touched × clustered_client_bound(P) + (general − touched) × (P ×
    /// IDLE_CONNECTION_BYTES + BACKCHANNEL_BYTES)`. The untouched clients'
    /// term lets each one's connections and backchannel be replaced by fresh
    /// ones once.
    bound: u64,
}

impl OneWarmFigures {
    fn of(general: usize, touched: usize, primaries: usize) -> Self {
        let untouched_client = primaries * IDLE_CONNECTION_BYTES + BACKCHANNEL_BYTES;
        let bound = touched as u64 * clustered_client_bound(primaries) as u64
            + ((general - touched) * untouched_client) as u64;
        Self { bound }
    }
}

/// Test 11c's bound (pure, runs everywhere), checked here because 11c itself
/// runs only on CI's cluster: at test 11c's pools (G = 5 general clients at a
/// pool of 1, of which 2 are loaded), `2 × clustered_client_bound(P) + 3 ×
/// (P × IDLE_CONNECTION_BYTES + BACKCHANNEL_BYTES)` is 1,159,168 B on 2
/// primaries, 1,200,128 B on 3 (CI's cluster) and 1,241,088 B on 4.
#[test]
fn one_warm_per_touched_client_bound_counts_the_cases_clients_b26e() {
    let mut config = common::test_config();
    config.valkey_pool_size = 1;
    let general = PoolSizes::from_config(&config).general_clients();
    assert_eq!(general, 5, "test 11c's general clients at a pool of 1");
    for (primaries, bound) in [(2, 1_159_168), (3, 1_200_128), (4, 1_241_088)] {
        assert_eq!(
            OneWarmFigures::of(general, UPLOAD_PATH_TOUCHED_CLIENTS, primaries).bound,
            bound,
            "test 11c's bound on {primaries} primaries"
        );
    }
}

/// Test 11c (b.26e F3, cluster-gated): one warm connection per client on
/// the upload path. The field-dense WATCH upload at [`DENSE`] fields, P + 2
/// calls, through a client from the production `connect` at a pool of 1, so
/// the calls load exactly two clients: the data member (each call's
/// 40,001-command pre-image pipeline) and the txn member (its WATCH
/// transaction, at least 20,000 commands), both carrying more than
/// `in_flight_commands` in flight. Call k's object is placed, before the
/// reference, on primary (k − 1) mod P in the routing table's order, so
/// every primary's connection of both clients carries a call.
///
/// Then, as on every case of test 11, each of the five general clients'
/// retention reports must keep the warm rule ([`warm_rule_violations`]):
/// on the two loaded clients, every one of whose P connections has read
/// progress, exactly one warm connection with its queue cut back to the
/// limit, and every other connection cut to the idle shape. Every connection
/// warm (the per-connection rule alone) fails V1; no sweep fails V3; a cut
/// that frees nothing fails V2. Then the standing negative controls
/// ([`assert_warm_rule_discriminates`]) on the data member and the txn slot
/// show, on this cluster, that the check reports a skipped sweep (V3) and
/// every connection warm (V1), and nothing once the sweep runs as built.
///
/// Its bytes: after every call the heap is at most [`OneWarmFigures::bound`]
/// above the reference (1,200,128 B on 3 primaries), and the plateau holds
/// from call 3 on, which fail with the connection buffer limits reverted or
/// their wiring removed (a 65,536-slot queue per touched connection).
///
/// Self-checks: the client is clustered, with P ≥ 2 from [`topology_of`] (the
/// warm rule needs more than one connection per client to choose among); the
/// data and txn pools have one member each; P + 2 calls, each call's object
/// on its primary; `DENSE > in_flight_commands` (at compile time); no stream
/// reader connected; the data member's and the txn slot's reports have P
/// connections, every one with read progress. A failure prints each call's
/// heap, each client's report and what each client freed at teardown.
///
/// Runs only with `LIVE_EDIT_TEST_CLUSTER_URL` (CI's `test-cluster:` job);
/// prints a `SKIP:` line otherwise. Its bound is checked everywhere by
/// `one_warm_per_touched_client_bound_counts_the_cases_clients_b26e`, and
/// the warm rule's decisions by
/// `warm_rule_violations_tell_one_warm_connection_from_none_or_several_b26e`.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn post_release_cluster_one_warm_connection_per_client_on_the_upload_path_b26e() {
    let _serial = measure_lock().await;
    let Some(url) = cluster_url("t11c") else {
        return;
    };
    // P, for the call count: the cluster's routing, read through a probe
    // before anything is measured.
    let primaries = {
        let probe = TestValkey::setup_with_url(url.clone(), 1).await;
        topology_of(&probe.client().pool.clients()[0]).primaries()
    };
    assert_post_release_case(PostReleaseCase {
        name: format!("t11c watch-field-dense:{DENSE}:1:{} placed", primaries + 2),
        path: Path::UpdateSlow,
        route: Route::Watch,
        pool_size: 1,
        calls: field_dense_calls(DENSE, primaries + 2),
        cluster_url: Some(url),
        shape: CaseShape::OneWarmPerTouchedClient,
    })
    .await;
}

/// Test 11's cluster arm, its topology rule (pure, runs everywhere): the
/// bound's topology comes from the client's routing table — none is a
/// standalone store, and a table is a cluster of its distinct primaries,
/// not of its slot ranges. A routing of four slot ranges over three
/// primaries (one owning two ranges) is `Topology::cluster(3)`, so the
/// cluster cases are held to formula (B) at P = 3.
#[test]
fn at_rest_bound_topology_is_the_routing_tables_primary_count_b26e() {
    use fred::bytes_utils::Str;
    use fred::types::Value;
    let range = |start: i64, end: i64, host: &str, id: &str| {
        Value::Array(vec![
            Value::Integer(start),
            Value::Integer(end),
            Value::Array(vec![
                Value::String(Str::from(host)),
                Value::Integer(7000),
                Value::String(Str::from(id)),
            ]),
        ])
    };
    let slots = Value::Array(vec![
        range(0, 5_460, "node-a.example.com", "a"),
        range(5_461, 10_922, "node-b.example.com", "b"),
        range(10_923, 13_000, "node-c.example.com", "c"),
        range(13_001, 16_383, "node-a.example.com", "a"),
    ]);
    let routing = fred::types::cluster::ClusterRouting::from_cluster_slots(slots, "127.0.0.1")
        .expect("the CLUSTER SLOTS reply parses");
    assert_eq!(
        topology_of_routing(Some(&routing)),
        Topology::cluster(3),
        "four slot ranges over three primaries"
    );
    assert_eq!(topology_of_routing(None), Topology::standalone());
}

/// Test 11 and M6 over any cases: each case in `B26E_POST_RELEASE_CASE`
/// (`;`-separated, [`post_release_case`]), by default test 11's cases at
/// pools of 2, 4 and 8 — field-dense on the WATCH path, one 16 MiB external
/// value and the value-heavy upload on the fast path, and the 20,000-child
/// cold start. Every case runs and prints its figures; the test then fails if
/// any left the heap above its at-rest bound after a call, or kept growing it
/// once every connection it can use was covered (a case with too few calls
/// for that prints "plateau: not checked"). `B26E_TRACK=1` runs the
/// allocation tracker; `B26E_BACKTRACE_MIN` sets the smallest allocation the
/// measured calls backtrace.
///
/// The default set takes far longer than nextest's per-test kill, so run it
/// under `cargo test`:
///
/// ```text
/// cargo test -p live-edit --features test-support --test upload_memory_envelope \
///     -- --ignored --exact post_release_live_memory_cases_stay_within_the_at_rest_bound_b26e \
///     --nocapture
/// ```
///
/// or one case at a time, under either runner:
///
/// ```text
/// B26E_POST_RELEASE_CASE=watch-field-dense:20000:4:7 cargo test -p live-edit \
///     --features test-support --test upload_memory_envelope \
///     -- --ignored --exact post_release_live_memory_cases_stay_within_the_at_rest_bound_b26e \
///     --nocapture
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: b.26e post-release live memory over chosen cases (M6), long-running"]
async fn post_release_live_memory_cases_stay_within_the_at_rest_bound_b26e() {
    let _serial = measure_lock().await;
    let specs = std::env::var("B26E_POST_RELEASE_CASE").unwrap_or_else(|_| {
        [2, 4, 8]
            .iter()
            .map(|pool| {
                format!(
                    "watch-field-dense:{DENSE}:{pool}:{};fast-blob:{}:{pool}:10;\
                     fast-value-heavy:0:{pool}:10;cold-children:{DENSE}:{pool}:4",
                    pool + 3,
                    16 * MIB
                )
            })
            .collect::<Vec<_>>()
            .join(";")
    });
    let mut violations = Vec::new();
    for spec in specs.split(';').filter(|s| !s.is_empty()) {
        violations.extend(
            measure_post_release(post_release_case(spec))
                .await
                .violations(),
        );
    }
    assert!(
        violations.is_empty(),
        "post-release cases past their at-rest bound: {violations:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: the stream reader's class limits
// ─────────────────────────────────────────────────────────────────────────────

/// Bound on waiting for a stream reader to be parked in its `XREAD BLOCK`
/// again; unreachable on the healthy path.
const READER_PARK_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Bound on waiting for a relayed entry to reach its subscriber.
const DELIVERY_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// A value-heavy fast-path delta, [`VALUES`] inline values of [`VALUE_LEN`]
/// bytes (4 MiB), its values salted by `salt` so successive commits differ.
fn salted_value_heavy_delta(salt: u8) -> OwnedDelta {
    OwnedDelta {
        set_fields: (0..VALUES)
            .map(|i| field(format!("v{i}"), vec![i as u8 ^ salt; VALUE_LEN]))
            .collect(),
        ..Default::default()
    }
}

/// Commit [`salted_value_heavy_delta`] to `object_id` through `svc`'s store,
/// relaying its OMPF verbatim: the stream entry carries the whole 4 MiB.
/// Returns the new sequence number and the relay's size.
async fn commit_value_heavy_relay(
    svc: &LiveEditService,
    session_id: &str,
    object_id: u64,
    salt: u8,
) -> (u64, usize) {
    let delta = salted_value_heavy_delta(salt);
    let relay = Bytes::from(common::owned_delta::delta_to_ompf_bytes(&delta, &[]));
    let len = relay.len();
    // Boxed, as every store and handler call here is (see `new_session`).
    let (seq, _) = Box::pin(svc.layer_store.update_layer(
        session_id,
        "layer",
        object_id,
        Some(delta),
        Some(relay),
        "t12",
        0,
    ))
    .await
    .unwrap_or_else(|e| panic!("the value-heavy commit (salt {salt}) must land: {e:?}"));
    (seq, len)
}

/// Wait until the server shows `reader`'s one connection blocked in an
/// `XREAD` (`CLIENT LIST`: `cmd=xread`, flag `b`), asked over `observer`:
/// the reader has finished handling what it read and is waiting again.
async fn await_reader_parked(observer: &fred::clients::Client, reader: &fred::clients::Client) {
    await_reader_parked_on(observer, None, reader).await;
}

/// The `CLIENT LIST` line of connection `id`, asked over `observer` — on a
/// cluster, of `node`, the one node the connection can be listed on; `None`
/// when the server lists no such connection.
async fn client_list_line(
    observer: &fred::clients::Client,
    node: Option<&fred::types::config::Server>,
    id: &str,
) -> Option<String> {
    use fred::interfaces::ClientInterface;
    let ids = Some(vec![id.to_owned()]);
    let listing: String = match node {
        Some(node) => {
            observer
                .with_cluster_node(node.clone())
                .client_list::<String, String>(None, ids)
                .await
        }
        None => observer.client_list::<String, String>(None, ids).await,
    }
    .expect("CLIENT LIST ID on the observer");
    listing
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(str::to_owned)
}

/// [`await_reader_parked`], asking `node` on a cluster (the node `reader`'s
/// one connection is to).
async fn await_reader_parked_on(
    observer: &fred::clients::Client,
    node: Option<&fred::types::config::Server>,
    reader: &fred::clients::Client,
) {
    use fred::interfaces::ClientInterface;
    let ids = reader.connection_ids();
    assert_eq!(
        ids.len(),
        1,
        "a reader client holds one connection; got {ids:?}"
    );
    let id = ids.values().next().expect("checked above").to_string();
    let field_of = |line: &str, name: &str| -> String {
        line.split(' ')
            .find_map(|kv| kv.strip_prefix(&format!("{name}=")).map(str::to_owned))
            .unwrap_or_default()
    };
    let deadline = tokio::time::Instant::now() + READER_PARK_WAIT;
    loop {
        let parked = client_list_line(observer, node, &id)
            .await
            .is_some_and(|l| field_of(&l, "cmd") == "xread" && field_of(&l, "flags").contains('b'));
        if parked {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "reader connection {id} was not parked in XREAD within {READER_PARK_WAIT:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// Test 12 (b.26e F): once a stream reader has delivered an entry far above
/// its read limit, it keeps at most its class limits: an in-flight queue of
/// `READER_LIMITS.in_flight_commands` slots, its write buffer, a read buffer
/// under `READER_LIMITS.kept_read_bytes()` (256 KiB, a quarter of one XREAD
/// reply target), and a backchannel's two 8 KiB buffers — `171,520 + 256 KiB
/// + 16,384` bytes, computed from the constants.
///
/// One reader, one subscription. The same 4 MiB value-heavy commit runs once
/// before the subscription starts (so the data and txn connections, which
/// write it, are at their kept shape and no reader has read it) and once
/// after, through a client built by the production `connect` with a pool of
/// 1. The live bytes from the reader parked before the second commit to the
/// reader parked again after delivering it are what the reader keeps, plus
/// the heap the commit itself leaves (nothing past what the first commit
/// left). Without the reader's limits its read buffer keeps the 4 MiB reply.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn stream_reader_keeps_at_most_its_class_limits_after_a_relay_far_above_its_read_limit_b26e()
{
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut config = common::test_config();
    config.valkey_pool_size = 1;
    config.subscriber_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own client connects");
    let svc = isolated_service(&tv, Route::Default, client.clone());
    let (session_id, object_id) = base_layer(&tv, "t12-reader", Base::Children(0)).await;
    let observer = tv.client().pool.next().clone();

    let (warm_seq, _) = commit_value_heavy_relay(&svc, &session_id, object_id, 0x11).await;
    let consumer = consume_layer_stream(
        &client,
        tv.key_schema(),
        object_id,
        StreamCursor::From(format!("{warm_seq}-0")),
        tokio::time::Instant::now() + READER_PARK_WAIT,
    )
    .await
    .expect("the subscription starts");
    // Readers connect lazily, once a subscription needs them.
    let deadline = tokio::time::Instant::now() + READER_PARK_WAIT;
    let reader = loop {
        if let Some(reader) = client.stream_mux.reader_client_for_test(0) {
            break reader;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the reader did not connect to serve the subscription within {READER_PARK_WAIT:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };
    await_reader_parked(&observer, &reader).await;
    tokio::time::sleep(SETTLE).await;
    let reference = live_now();

    let (seq, relay_len) = commit_value_heavy_relay(&svc, &session_id, object_id, 0x22).await;
    let (_, notification) = tokio::time::timeout(DELIVERY_WAIT, consumer.recv())
        .await
        .expect("the entry is delivered in time")
        .expect("the subscription is live");
    let delivered = match &notification {
        live_edit::storage::notifications::LayerNotification::Update {
            sequence_num,
            relay_ompf,
            ..
        } => (*sequence_num, relay_ompf.as_ref().map_or(0, Bytes::len)),
        other => panic!("fixture self-check: an update is delivered, not {other:?}"),
    };
    assert_eq!(
        delivered,
        (seq, relay_len),
        "fixture self-check: the reader delivered the measured commit's whole relay"
    );
    assert!(
        relay_len as u64 > 2 * XREAD_REPLY_TARGET_BYTES,
        "fixture self-check: the relay ({relay_len} B) is far above the reader's read limit"
    );
    drop(notification);
    await_reader_parked(&observer, &reader).await;
    tokio::time::sleep(SETTLE).await;
    let kept = live_now() - reference;

    let bound = reader_class_bound();
    eprintln!(
        "== b.26e test 12: after a {relay_len} B relay the heap is {kept} B above the reader \
         parked before it; reader class bound {bound} B =="
    );
    assert!(
        kept <= bound,
        "the stream reader kept {kept} B after delivering a {relay_len} B entry, past its class \
         limits ({bound} B): its read buffer was not cut back once it drained"
    );
    consumer.unsubscribe().await;
    drop(reader);
    drop(svc);
    client.quit().await;
}

/// What test 12 and its cluster variant allow a stream reader to keep: one
/// connection at the reader class limits — its in-flight queue, its write
/// buffer and a read buffer under `READER_LIMITS.kept_read_bytes()` (within
/// one XREAD reply target) — and its backchannel's two 8 KiB buffers,
/// computed from the constants.
fn reader_class_bound() -> i64 {
    (READER_LIMITS.kept_queue_bytes()
        + READER_LIMITS.kept_write_bytes()
        + READER_LIMITS.kept_read_bytes()
        + BACKCHANNEL_BYTES) as i64
}

/// Test 12's bound (pure, runs everywhere): [`reader_class_bound`] is one
/// reader connection and its backchannel, `171,520 + 256 KiB + 16,384` bytes
/// at the 64 KiB reader read limit — so the cluster variant, which cannot run
/// here, asserts against the same figure the standalone test does.
#[test]
fn stream_reader_class_bound_is_one_connection_and_its_backchannel_b26e() {
    assert_eq!(reader_class_bound(), 171_520 + 262_144 + 16_384);
}

/// Base layers [`cluster_stream_reader_retargeted_to_another_primary_keeps_one_connection_b26e`]
/// creates looking for one whose stream is on another primary than the
/// first's; with three primaries, nearly every object's is.
const OBJECTS_TO_FIND_ANOTHER_PRIMARY: usize = 32;

/// The primary that owns `object_id`'s stream in `member`'s routing table.
fn stream_owner(
    member: &fred::clients::Client,
    schema: &live_edit::storage::keys::KeySchema,
    object_id: u64,
) -> fred::types::config::Server {
    use fred::interfaces::ClusterInterface;
    let slot = fred::util::redis_keyslot(schema.layer_stream(object_id).as_bytes());
    member
        .cached_cluster_state()
        .expect("a connected cluster client has its routing")
        .get_server(slot)
        .cloned()
        .expect("the routing names the stream slot's owner")
}

/// Wait until `client`'s reader 0 holds a connection to `node` alone, and
/// return its fred client.
async fn await_reader_on(
    client: &live_edit::storage::client::ValkeyClient,
    node: &fred::types::config::Server,
) -> fred::clients::Client {
    let deadline = tokio::time::Instant::now() + READER_PARK_WAIT;
    loop {
        if let Some(reader) = client.stream_mux.reader_client_for_test(0) {
            if reader.active_connections() == vec![node.clone()] {
                return reader;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "reader 0 did not come to hold one connection, to {node}, within \
             {READER_PARK_WAIT:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// The next notification of `consumer`, which must be the update committed at
/// `seq` carrying a `len`-byte relay.
async fn expect_relay(consumer: &LayerStreamConsumer, seq: u64, len: usize) {
    let (_, notification) = tokio::time::timeout(DELIVERY_WAIT, consumer.recv())
        .await
        .expect("the entry is delivered in time")
        .expect("the subscription is live");
    let delivered = match &notification {
        live_edit::storage::notifications::LayerNotification::Update {
            sequence_num,
            relay_ompf,
            ..
        } => (*sequence_num, relay_ompf.as_ref().map_or(0, Bytes::len)),
        other => panic!("fixture self-check: an update is delivered, not {other:?}"),
    };
    assert_eq!(
        delivered,
        (seq, len),
        "fixture self-check: the reader delivered the commit's whole relay"
    );
}

/// Test 12 on a cluster (b.26e F, Invariant 10; cluster-gated): a stream
/// reader whose only subscription moves from an object whose stream is on
/// primary A to one on primary B retargets — the old client is retired with
/// its connection and buffers, and the new one holds one connection, to B —
/// and keeps at most [`reader_class_bound`] (`171,520 + 256 KiB + 16,384`
/// bytes) once it has delivered an entry far above its read limit on each.
///
/// One reader, pools of 1, a client built by the production `connect`.
/// Before the reference: the same 4 MiB value-heavy commit on each object
/// (so the writers' connections to both primaries are at their kept shape),
/// B's last, and the reader parked on A. Then a commit on A, delivered; the
/// subscription moved to B; a commit on B, delivered; the reader parked
/// again. The writers end as they began, warm on B's primary, so the heap
/// above the reference is the reader's.
///
/// Each failure it names: a reader that is a cluster client holds a
/// connection per primary (the one-connection checks); a retarget that
/// leaves the old client alive leaves its connection listed on A; and a
/// reader that keeps a reply's read buffer keeps at least 4 MiB.
///
/// Runs only with `LIVE_EDIT_TEST_CLUSTER_URL` (CI's `test-cluster:` job);
/// prints a `SKIP:` line otherwise. Its bound is checked everywhere by
/// `stream_reader_class_bound_is_one_connection_and_its_backchannel_b26e`.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cluster_stream_reader_retargeted_to_another_primary_keeps_one_connection_b26e() {
    let _serial = measure_lock().await;
    let test = "t12c";
    let Some(url) = cluster_url(test) else {
        return;
    };
    let tv = TestValkey::setup_with_url(url.clone(), 1).await;
    let mut config = common::test_config();
    config.valkey_url = url;
    config.valkey_pool_size = 1;
    config.subscriber_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own cluster client connects");
    let member = client.pool.clients()[0].clone();
    assert!(
        member.is_clustered(),
        "{CLUSTER_URL_VAR} must point at a clustered deployment"
    );
    let svc = isolated_service(&tv, Route::Default, client.clone());
    let observer = tv.client().pool.next().clone();

    let (session_a, object_a) = base_layer(&tv, "t12c-a", Base::Children(0)).await;
    let node_a = stream_owner(&member, tv.key_schema(), object_a);
    let mut on_b = None;
    for i in 0..OBJECTS_TO_FIND_ANOTHER_PRIMARY {
        let (session, object) = base_layer(&tv, &format!("t12c-b{i}"), Base::Children(0)).await;
        if stream_owner(&member, tv.key_schema(), object) != node_a {
            on_b = Some((session, object));
            break;
        }
    }
    let (session_b, object_b) =
        on_b.expect("fixture self-check: some object's stream is on another primary");
    let node_b = stream_owner(&member, tv.key_schema(), object_b);
    drop(member);

    let (warm_a, _) = commit_value_heavy_relay(&svc, &session_a, object_a, 0x11).await;
    let (warm_b, _) = commit_value_heavy_relay(&svc, &session_b, object_b, 0x12).await;
    let consumer_a = consume_layer_stream(
        &client,
        tv.key_schema(),
        object_a,
        StreamCursor::From(format!("{warm_a}-0")),
        tokio::time::Instant::now() + READER_PARK_WAIT,
    )
    .await
    .expect("the subscription on A starts");
    let (old_name, old_conn) = {
        use fred::interfaces::{ClientInterface, ClientLike};
        let old = await_reader_on(&client, &node_a).await;
        await_reader_parked_on(&observer, Some(&node_a), &old).await;
        let conn = old
            .connection_ids()
            .values()
            .next()
            .expect("the reader's one connection")
            .to_string();
        (old.id().to_owned(), conn)
    };
    tokio::time::sleep(SETTLE).await;
    let reference = live_now();

    let (seq_a, len_a) = commit_value_heavy_relay(&svc, &session_a, object_a, 0x21).await;
    expect_relay(&consumer_a, seq_a, len_a).await;
    consumer_a.unsubscribe().await;
    let consumer_b = consume_layer_stream(
        &client,
        tv.key_schema(),
        object_b,
        StreamCursor::From(format!("{warm_b}-0")),
        tokio::time::Instant::now() + READER_PARK_WAIT,
    )
    .await
    .expect("the subscription on B starts");
    let new_name = {
        use fred::interfaces::ClientLike;
        let reader = await_reader_on(&client, &node_b).await;
        await_reader_parked_on(&observer, Some(&node_b), &reader).await;
        reader.id().to_owned()
    };
    let (seq_b, len_b) = commit_value_heavy_relay(&svc, &session_b, object_b, 0x22).await;
    expect_relay(&consumer_b, seq_b, len_b).await;
    {
        let reader = await_reader_on(&client, &node_b).await;
        await_reader_parked_on(&observer, Some(&node_b), &reader).await;
    }
    tokio::time::sleep(SETTLE).await;
    let kept = live_now() - reference;

    let deadline = tokio::time::Instant::now() + READER_PARK_WAIT;
    while client_list_line(&observer, Some(&node_a), &old_conn)
        .await
        .is_some()
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the reader's old connection (id {old_conn}) is still listed on {node_a} after its \
             subscription moved to {node_b}: the retarget left the old client alive"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_ne!(
        old_name, new_name,
        "the retarget replaced the reader's client"
    );
    assert_eq!(
        client.stream_mux.connected_readers(),
        1,
        "one reader, holding one connection"
    );
    let bound = reader_class_bound();
    eprintln!(
        "CLUSTER-GATED RAN: {test}: the reader moved from {node_a} to {node_b}; the heap is \
         {kept} B above the reader parked on A before either entry (bound {bound} B)"
    );
    assert!(
        kept <= bound,
        "the stream reader kept {kept} B after delivering a {len_a} B entry on {node_a} and a \
         {len_b} B entry on {node_b}, past one connection at its class limits ({bound} B)"
    );
    consumer_b.unsubscribe().await;
    drop(svc);
    client.quit().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15: the retry buffer after a replay
// ─────────────────────────────────────────────────────────────────────────────

/// GETs in flight when test 15 kills their connection: more than fifteen
/// times the general in-flight limit (`GENERAL_LIMITS.in_flight_commands`,
/// 256), so a retry buffer kept at that limit, or at the replay's size, is
/// plain next to what an empty one keeps.
const REPLAYED_COMMANDS: usize = 4_000;

/// Bound on waiting for the server to have read a stalled pipeline, and for
/// the pipeline's replies after its replay.
const REPLAY_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// What test 15 allows a connection's replacement and the replay to leave
/// beside what the connection they replace kept: a few small allocations a
/// reconnect makes (the new socket, its codec state), measured at under
/// 2 KiB. A retry buffer kept at the general in-flight limit is 96,256 B.
const RECONNECT_NOISE_BYTES: i64 = 32 * 1024;

/// One `GET key` command as the server reads it (RESP2 array of two bulk
/// strings).
fn get_frame_len(key: &str) -> u64 {
    format!("*2\r\n$3\r\nGET\r\n${}\r\n{key}\r\n", key.len()).len() as u64
}

/// A pipeline of `GET` on each of `keys` on `member`; every reply must be a
/// nil (the keys do not exist).
async fn get_pipeline(member: fred::clients::Client, keys: Vec<String>) {
    use fred::interfaces::KeysInterface;
    let pipeline = member.pipeline();
    for key in &keys {
        let _: () = pipeline
            .get(key)
            .await
            .expect("a GET queues in the pipeline");
    }
    let replies: Vec<Result<Option<String>, fred::error::Error>> = pipeline.try_all().await;
    assert_eq!(replies.len(), keys.len(), "one reply per GET");
    for (i, reply) in replies.iter().enumerate() {
        assert!(
            matches!(reply, Ok(None)),
            "GET {i} must answer nil after its replay, got {reply:?}"
        );
    }
}

/// Park `member`'s connection in a `BLPOP` on `stall_key` ([`ConnStall`]),
/// send a pipeline of `GET` on `keys` behind it, wait until the server has
/// read every one of them (so each is in flight, written and unanswered),
/// then kill the connection: fred moves the GETs to its retry buffer,
/// reconnects and replays them, and every one is answered.
async fn kill_with_gets_in_flight(
    member: &fred::clients::Client,
    observer: &fred::clients::Client,
    keys: &[String],
    stall_key: String,
) {
    let stall = ConnStall::park(member, observer, stall_key).await;
    let read_before = stall
        .net_in()
        .await
        .expect("the stalled connection is alive");
    let want = read_before + keys.iter().map(|k| get_frame_len(k)).sum::<u64>();
    let pipeline = tokio::spawn(get_pipeline(member.clone(), keys.to_vec()));
    let deadline = tokio::time::Instant::now() + REPLAY_WAIT;
    loop {
        let read = stall
            .net_in()
            .await
            .expect("the stalled connection is alive until the kill");
        if read >= want {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the server read {read} B of the {want} B the stall and its GETs make within \
             {REPLAY_WAIT:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let stalled = stall.kill().await;
    assert!(
        stalled.is_err(),
        "fixture self-check: the kill, not a release, ended the stall; got {stalled:?}"
    );
    tokio::time::timeout(REPLAY_WAIT, pipeline)
        .await
        .expect("the replayed GETs are answered in time")
        .expect("the pipeline task does not panic");
}

/// Test 15 (b.26e F): a replay leaves the retry buffer holding nothing. A
/// data-pool connection (pool of 1, built by the production `connect`) is
/// killed with [`REPLAYED_COMMANDS`] GETs in flight; fred moves them to its
/// retry buffer, reconnects and replays them. Once every replayed GET is
/// answered the heap is back at the reference taken before the kill — the
/// connection's replacement at the shape it was in — within
/// [`RECONNECT_NOISE_BYTES`]. A retry buffer kept at its replay's size
/// (stock fred: 4,096 slots, 1.5 MB) or at the in-flight limit (256 slots,
/// 96,256 B) fails it.
///
/// The kill is deterministic: the connection is held in a server-side
/// `BLPOP` ([`ConnStall`]), and the kill waits until the server has read
/// every GET behind it, so each is in flight when the connection dies.
///
/// Before the reference: the same pipeline without a kill (the connection
/// at its kept shape), a kill with one GET in flight (a reconnect's first-time
/// allocations, and a retry buffer of at most a few slots), and the pipeline
/// again on the new connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn retry_buffer_keeps_nothing_once_a_replay_longer_than_the_queue_limit_empties_it_b26e() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut config = common::test_config();
    config.valkey_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own client connects");
    let member = client.pool.clients()[0].clone();
    let observer = tv.client().pool.next().clone();
    let keys: Vec<String> = (0..REPLAYED_COMMANDS)
        .map(|i| format!("{}t15:{i:05}", tv.prefix()))
        .collect();
    let stall_key = |i: usize| format!("{}t15-stall:{i}", tv.prefix());

    get_pipeline(member.clone(), keys.clone()).await;
    kill_with_gets_in_flight(&member, &observer, &keys[..1], stall_key(0)).await;
    get_pipeline(member.clone(), keys.clone()).await;
    tokio::time::sleep(SETTLE).await;
    let reference = live_now();

    kill_with_gets_in_flight(&member, &observer, &keys, stall_key(1)).await;
    tokio::time::sleep(SETTLE).await;
    let kept = live_now() - reference;
    eprintln!(
        "== b.26e test 15: after a replay of {REPLAYED_COMMANDS} GETs the heap is {kept} B above \
         the reference taken before the kill (allowance {RECONNECT_NOISE_BYTES} B; a retry \
         buffer at the in-flight limit is {} B) ==",
        GENERAL_LIMITS.kept_queue_bytes()
    );
    assert!(
        kept <= RECONNECT_NOISE_BYTES,
        "the heap is {kept} B above where it was before the kill once the replay of \
         {REPLAYED_COMMANDS} GETs was answered: the retry buffer kept its allocation after the \
         replay emptied it"
    );
    drop(member);
    client.quit().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 22: a fred Transaction's EXEC reply
// ─────────────────────────────────────────────────────────────────────────────

/// The value test 22's transaction reads: its EXEC reply is one frame of
/// about this size, above the general read limit
/// (`GENERAL_LIMITS.read_buffer_bytes`, 64 KiB).
const EXEC_VALUE_LEN: usize = 300_000;

/// What a call may leave live when it returns, in test 22: the general read
/// limit. A read buffer kept after reading the EXEC reply holds at least the
/// reply.
const EXEC_LEFT_LIMIT: i64 = 64 * 1024;

/// The value test 22's unmeasured transaction reads: its EXEC reply fits the
/// connection's starting 8 KiB read buffer.
const EXEC_WARMUP_VALUE_LEN: usize = 1_024;

/// `MULTI`, `GET key`, `EXEC` on `client` through fred's `Transaction`; the
/// reply is checked to be the `len`-byte value and dropped.
async fn exec_get(client: &fred::clients::Client, key: &str, len: usize) {
    use fred::interfaces::{KeysInterface, TransactionInterface};
    let trx = client.multi();
    let _: () = trx.get(key).await.expect("GET queues in the transaction");
    let reply: Vec<Bytes> = trx.exec(true).await.expect("EXEC answers");
    assert_eq!(
        reply.iter().map(Bytes::len).collect::<Vec<_>>(),
        vec![len],
        "fixture self-check: EXEC answers the one GET"
    );
}

/// Test 22 (b.26e F, regression i): a fred `Transaction` (`client.multi()`,
/// as CreateLayer commits) reads its replies through `Connection::read`, off
/// the router's read loop, so its EXEC reply's read buffer is cut back only
/// by the read rule `live_edit::next` applies to every frame taken that way.
/// On a txn member from the production `connect`, a transaction whose EXEC
/// reply is about [`EXEC_VALUE_LEN`] bytes leaves less than
/// [`EXEC_LEFT_LIMIT`] live when it returns.
///
/// Before the reference: the same transaction on a value of
/// [`EXEC_WARMUP_VALUE_LEN`] bytes, so the transaction path's first-use
/// allocations are made while the connection's read buffer stays at its
/// starting size, and the measured EXEC reply has to grow it.
///
/// The read rule runs at the `next` hook and again in the sweep at the top
/// of the router's next pass, before the router yields, so this test cannot
/// tell which of the two cut the buffer; code review checks the `next` call.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn fred_transaction_exec_reply_above_the_read_limit_is_released_by_the_end_of_the_call_b26e()
{
    use fred::interfaces::KeysInterface;
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut config = common::test_config();
    config.valkey_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own client connects");
    let warmup_key = format!("{}t22:warmup", tv.prefix());
    let key = format!("{}t22:value", tv.prefix());
    for (key, len) in [(&warmup_key, EXEC_WARMUP_VALUE_LEN), (&key, EXEC_VALUE_LEN)] {
        let _: () = tv
            .client()
            .pool
            .set(key, vec![b'v'; len], None, None, false)
            .await
            .expect("SET of a value the transactions read");
    }

    let txn = client.acquire_txn().await;
    exec_get(&txn, &warmup_key, EXEC_WARMUP_VALUE_LEN).await;
    let before = live_now();
    exec_get(&txn, &key, EXEC_VALUE_LEN).await;
    let left = live_now() - before;
    eprintln!(
        "== b.26e test 22: a transaction with a {EXEC_VALUE_LEN} B EXEC reply left {left} B \
         live when it returned (limit {EXEC_LEFT_LIMIT} B) =="
    );
    assert!(
        left < EXEC_LEFT_LIMIT,
        "a fred Transaction whose EXEC reply is about {EXEC_VALUE_LEN} B left {left} B live when \
         it returned: the read buffer that held the reply was not cut back before the caller \
         was answered"
    );
    drop(txn);
    client.quit().await;
}

/// Fields in test 22's unmeasured CreateLayer: its EXEC reply fits the txn
/// connection's starting 8 KiB read buffer.
const CREATE_WARMUP_FIELDS: usize = 100;

/// Test 22, CreateLayer: a CreateLayer of [`DENSE`] fields commits its whole
/// object in one fred `Transaction`, whose EXEC reply (one reply per queued
/// write) is above the general read limit, and the call leaves less than
/// [`EXEC_LEFT_LIMIT`] live when it returns. Through a client built by the
/// production `connect`, pool of 1; one unmeasured CreateLayer of
/// [`CREATE_WARMUP_FIELDS`] fields first, so the path's first-use
/// allocations are made while the txn connection's read buffer stays at its
/// starting size, and the measured EXEC reply has to grow it.
///
/// This holds Invariant 9 for CreateLayer end to end; it does not single out
/// the `live_edit::next` hook. After the object's transaction, CreateLayer's
/// commit point (`create_layer_commit.lua`) runs on a txn checkout, which
/// with calls one at a time is the same slot, and its replies go through the
/// read loop, which cuts the EXEC reply's buffer before the call returns
/// with or without the hook. The transaction test above is the hook's
/// regression.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn create_layer_of_20k_fields_releases_its_exec_reply_by_the_end_of_the_call_b26e() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut config = common::test_config();
    config.valkey_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own client connects");
    let svc = isolated_service(&tv, Route::Default, client.clone());
    let fixture = |fields: usize| {
        Fixture::new(
            "field-dense",
            Feeds::Elements,
            OwnedDelta {
                struct_commands: own_section(Path::Create),
                set_fields: (0..fields)
                    .map(|i| field(format!("f{i}"), (i as u64).to_le_bytes().to_vec()))
                    .collect(),
                ..Default::default()
            },
            &[],
        )
    };
    let warmup = upload_frames(
        &tv,
        Path::Create,
        &fixture(CREATE_WARMUP_FIELDS),
        "t22-warmup",
    )
    .await;
    upload_once(&svc, Path::Create, &warmup.frames, "t22-create", 0).await;
    let measured = upload_frames(&tv, Path::Create, &fixture(DENSE), "t22-create").await;
    tokio::time::sleep(SETTLE).await;

    let before = live_now();
    upload_once(&svc, Path::Create, &measured.frames, "t22-create", 1).await;
    let left = live_now() - before;
    eprintln!(
        "== b.26e test 22: a CreateLayer of {DENSE} fields left {left} B live when it returned \
         (limit {EXEC_LEFT_LIMIT} B) =="
    );
    assert!(
        left < EXEC_LEFT_LIMIT,
        "a CreateLayer of {DENSE} fields left {left} B live when it returned: the txn \
         connection's read buffer kept its EXEC reply"
    );
    drop(svc);
    client.quit().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 23: the reader read limit under mixed reply sizes
// ─────────────────────────────────────────────────────────────────────────────

/// A single-node fred client on `TEST_VALKEY_URL` with the stream readers'
/// connection buffer limits, speaking `version`. Built here rather than
/// through a stream reader, whose one connection is held in an `XREAD BLOCK`
/// and cannot be given plain `GET`s; it carries the same limits
/// (`READER_LIMITS`, the one setting `ValkeyClient::reader_builder` adds that
/// bears on retention).
async fn reader_class_client(version: fred::types::RespVersion) -> fred::clients::Client {
    retention_client(READER_LIMITS, version).await
}

/// A single-node fred client on `TEST_VALKEY_URL` with connection buffer
/// limits `limits`, speaking `version` (RESP3 through `HELLO 3`). live-edit's
/// own clients speak RESP2 today; the RESP3 codec carries the same retention
/// hooks (`protocol/codec.rs`, `resp3_encode_frame` / `resp3_decode_frame`),
/// which these clients exercise.
async fn retention_client(
    limits: fred::live_edit::retention::RetentionLimits,
    version: fred::types::RespVersion,
) -> fred::clients::Client {
    let url = common::test_config().valkey_url;
    let mut config = fred::types::config::Config::from_url(&url).expect("TEST_VALKEY_URL parses");
    config.version = version;
    let mut builder = fred::types::Builder::from_config(config);
    builder.with_connection_config(|connection| {
        connection.live_edit.retention = Some(limits);
    });
    let client = builder.build().expect("the retention client builds");
    client.init().await.expect("the retention client connects");
    client
}

/// What a drained reader connection may keep: its class limits' queue and
/// write terms, and a read buffer under one XREAD reply target — the
/// per-reader reply memory the stream multiplexer accounts for, which
/// `at_rest_bound_bytes` leaves the reader's read buffer to. Today
/// `READER_LIMITS.kept_read_bytes()` is a quarter of that target (256 KiB;
/// the compile-time guard in `storage::connection_buffers` keeps it at or
/// under the target); the test uses the target itself so a read limit raised
/// past a quarter of it fails here rather than raising the bound with it.
fn reader_connection_bound() -> i64 {
    (READER_LIMITS.kept_queue_bytes()
        + READER_LIMITS.kept_write_bytes()
        + XREAD_REPLY_TARGET_BYTES as usize) as i64
}

/// Store a value of each length in `lens` under `tv`'s prefix; returns the
/// key of each, in order.
async fn store_values_of(tv: &TestValkey, lens: &[usize]) -> Vec<String> {
    use fred::interfaces::KeysInterface;
    let mut keys = Vec::with_capacity(lens.len());
    for &len in lens {
        let key = format!("{}t23:{len}", tv.prefix());
        let _: () = tv
            .client()
            .pool
            .set(&key, vec![b'r'; len], None, None, false)
            .await
            .expect("SET of a test 23 value");
        keys.push(key);
    }
    keys
}

/// Run `replies` on a fresh [`reader_class_client`] speaking `version` and
/// return what its connection keeps once they are dropped: the live bytes
/// above those after the client connected and answered one small `GET`.
/// Self-checks that the largest reply was read (the window's peak passed it).
async fn reader_class_kept<F, Fut>(
    version: fred::types::RespVersion,
    replies: F,
    largest: usize,
) -> i64
where
    F: FnOnce(fred::clients::Client) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    use fred::interfaces::KeysInterface;
    let client = reader_class_client(version).await;
    let _: Option<Bytes> = client
        .get("t23:absent")
        .await
        .expect("a small GET on the reader-class client");
    let reference = live_now();
    let baseline = begin_window();
    replies(client.clone()).await;
    let peak = end_window(baseline);
    let kept = live_now() - reference;
    assert!(
        peak >= largest as u64,
        "fixture self-check: the replies were read (peak {peak} B, largest reply {largest} B)"
    );
    client.quit().await.expect("the reader-class client quits");
    kept
}

/// Test 23 (b.26e F, regression ii), sequence (a): a reader-class connection
/// reads replies of 1,000,000, 200,000 and 900,000 B, each dropped before the
/// next, and then keeps at most [`reader_connection_bound`]: its class
/// limits' queue and write terms and a read buffer under one XREAD reply
/// target. A read buffer keeps just under four times the largest frame it
/// held below its read limit (the decoder leaves consumed replies as an
/// offset in the same allocation), so at a read limit of 1 MiB (the target)
/// this sequence leaves a 2 MiB read buffer; the reader limit, 64 KiB, cuts
/// it back.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn reader_read_buffer_stays_within_its_limits_after_mixed_replies_each_dropped_b26e() {
    assert_mixed_replies_each_dropped_keep_within_the_reader_limits(
        fred::types::RespVersion::RESP2,
    )
    .await;
}

/// Test 23, sequence (a), on RESP3 (`HELLO 3`): the same replies, decoded by
/// fred's RESP3 codec, whose decode hook (`resp3_decode_frame` →
/// `live_edit::on_decode`) records each frame's size. Without it the read
/// rule never sees a frame above the read limit, the read buffer is never
/// cut back, and the sequence keeps about 2 MiB. live-edit speaks RESP2
/// today; this holds the hook for a switch to RESP3.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn reader_read_buffer_stays_within_its_limits_after_mixed_replies_each_dropped_on_resp3_b26e()
{
    assert_mixed_replies_each_dropped_keep_within_the_reader_limits(
        fred::types::RespVersion::RESP3,
    )
    .await;
}

/// Test 23 (a)'s body, on a reader-class client speaking `version`.
async fn assert_mixed_replies_each_dropped_keep_within_the_reader_limits(
    version: fred::types::RespVersion,
) {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let keys = store_values_of(&tv, &[1_000_000, 200_000, 900_000]).await;
    let kept = reader_class_kept(
        version.clone(),
        |client| async move {
            use fred::interfaces::KeysInterface;
            for (key, len) in keys.iter().zip([1_000_000, 200_000, 900_000]) {
                let reply: Bytes = client.get(key).await.expect("GET");
                assert_eq!(
                    reply.len(),
                    len,
                    "fixture self-check: the reply is the value"
                );
            }
        },
        1_000_000,
    )
    .await;
    let bound = reader_connection_bound();
    eprintln!(
        "== b.26e test 23 (a), {version:?}: the connection kept {kept} B (bound {bound} B) =="
    );
    assert!(
        kept <= bound,
        "after replies of 1,000,000 / 200,000 / 900,000 B, each dropped, the reader-class \
         connection on {version:?} kept {kept} B, past its queue and write limits and one XREAD \
         reply target ({bound} B)"
    );
}

/// The value the RESP3 write case SETs: far above the general write limit
/// (`GENERAL_LIMITS.write_buffer_bytes`, 64 KiB), so its one encoded frame
/// grows the write buffer to hold it.
const RESP3_SET_VALUE_LEN: usize = 1024 * 1024;

/// The write rule on RESP3 (b.26e F, runs everywhere): on a
/// general-class client speaking RESP3 (`HELLO 3`), one `SET` of a
/// [`RESP3_SET_VALUE_LEN`]-byte value, and the connection keeps at most
/// `GENERAL_LIMITS.kept_write_bytes()` above the reference taken after a
/// small command. fred's RESP3 encode hook (`resp3_encode_frame` →
/// `live_edit::on_encode`) records the write buffer's length after each
/// encode; without it the write rule never sees a buffer above the limit and
/// the connection keeps the 1 MiB write buffer. live-edit speaks RESP2 today
/// (its write rule is held there by test 11's blob case); this holds the
/// hook for a switch to RESP3.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn general_class_write_buffer_is_cut_back_after_a_large_set_on_resp3_b26e() {
    use fred::interfaces::KeysInterface;
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let key = format!("{}resp3-write", tv.prefix());
    let client = retention_client(GENERAL_LIMITS, fred::types::RespVersion::RESP3).await;
    let _: Option<Bytes> = client
        .get(&key)
        .await
        .expect("a small GET on the RESP3 client");
    let value = vec![b'w'; RESP3_SET_VALUE_LEN];
    tokio::time::sleep(SETTLE).await;
    let reference = live_now();
    let baseline = begin_window();
    let _: () = client
        .set(&key, value.as_slice(), None, None, false)
        .await
        .expect("the SET lands");
    let peak = end_window(baseline);
    tokio::time::sleep(SETTLE).await;
    let kept = live_now() - reference;
    assert!(
        peak >= RESP3_SET_VALUE_LEN as u64,
        "fixture self-check: the SET's frame was encoded (peak {peak} B)"
    );
    let bound = GENERAL_LIMITS.kept_write_bytes() as i64;
    eprintln!(
        "== b.26e RESP3 write case: after a {RESP3_SET_VALUE_LEN} B SET the connection kept \
         {kept} B (bound {bound} B) =="
    );
    assert!(
        kept <= bound,
        "after one {RESP3_SET_VALUE_LEN} B SET on RESP3, the general-class connection kept \
         {kept} B, past its write limit's kept bytes ({bound} B): its write buffer was not cut \
         back once it flushed"
    );
    drop(value);
    client.quit().await.expect("the RESP3 client quits");
}

/// Test 23, sequence (b): a reply of 300,000 B is held while one of
/// 900,000 B is read, then both are dropped, and the connection keeps at
/// most [`reader_connection_bound`]. A held reply makes the read buffer
/// allocate afresh for the next, so at a read limit of 1 MiB this leaves
/// about 1.71 MiB.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn reader_read_buffer_stays_within_its_limits_after_a_reply_held_across_a_larger_one_b26e() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let keys = store_values_of(&tv, &[300_000, 900_000]).await;
    let kept = reader_class_kept(
        fred::types::RespVersion::RESP2,
        |client| async move {
            use fred::interfaces::KeysInterface;
            let held: Bytes = client.get(&keys[0]).await.expect("GET 300,000 B");
            let next: Bytes = client.get(&keys[1]).await.expect("GET 900,000 B");
            assert_eq!(
                (held.len(), next.len()),
                (300_000, 900_000),
                "fixture self-check: the replies are the values"
            );
        },
        900_000,
    )
    .await;
    let bound = reader_connection_bound();
    eprintln!("== b.26e test 23 (b): the connection kept {kept} B (bound {bound} B) ==");
    assert!(
        kept <= bound,
        "after a 300,000 B reply held across a 900,000 B one, the reader-class connection kept \
         {kept} B, past its queue and write limits and one XREAD reply target ({bound} B)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// M6: the in-flight slot (report only)
// ─────────────────────────────────────────────────────────────────────────────

/// M6 (b.26e F), report only: one in-flight queue slot is
/// `IN_FLIGHT_SLOT_BYTES` (376 B, `size_of::<fred Command>()`), which
/// `RetentionLimits::kept_queue_bytes` multiplies by. A pipeline of 16,384
/// GETs grows the queue to 16,384 slots and one of 16,385 to 32,768
/// (`VecDeque` doubles), so only the second makes allocations of exactly
/// 32,768 × 376 = 12,320,768 B — the connection's in-flight queue, and the
/// pipeline's own queue of commands, which grows the same way. A census of
/// each size over each pipeline counts them; a stalled run then holds the
/// 16,385 GETs in flight and prints the live allocations of that size with
/// where they were made.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e M6 in-flight slot size"]
async fn in_flight_slot_size_is_reported_b26e() {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut config = common::test_config();
    config.valkey_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own client connects");
    let member = client.pool.clients()[0].clone();
    let keys = |n: usize| -> Vec<String> {
        (0..n)
            .map(|i| format!("{}m6:{i:05}", tv.prefix()))
            .collect()
    };
    eprintln!("== b.26e M6: IN_FLIGHT_SLOT_BYTES = {IN_FLIGHT_SLOT_BYTES} ==");
    for commands in [16_384_usize, 16_385] {
        for slots in [16_384_usize, 32_768] {
            let size = slots * IN_FLIGHT_SLOT_BYTES;
            let input = keys(commands);
            begin_census(size);
            let baseline = begin_window();
            get_pipeline(member.clone(), input).await;
            end_window(baseline);
            let (made, _) = end_census();
            eprintln!(
                "pipeline of {commands} GETs: {made} allocations of exactly {size} B \
                 ({slots} slots x {IN_FLIGHT_SLOT_BYTES} B)"
            );
        }
    }
    let observer = tv.client().pool.next().clone();
    let stall = ConnStall::park(&member, &observer, format!("{}m6-stall", tv.prefix())).await;
    let input = keys(16_385);
    let want =
        stall.net_in().await.expect("alive") + input.iter().map(|k| get_frame_len(k)).sum::<u64>();
    start_tracking(32_768 * IN_FLIGHT_SLOT_BYTES);
    let pipeline = tokio::spawn(get_pipeline(member.clone(), input));
    let deadline = tokio::time::Instant::now() + REPLAY_WAIT;
    while stall.net_in().await.expect("alive") < want {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the stalled pipeline was not read in time"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    stop_recording();
    report_tracked("M6: a 16,385-GET pipeline held in flight behind a BLPOP", 4);
    end_tracking();
    let released = stall.release().await;
    assert!(
        matches!(&released, Ok(v) if !v.is_null()),
        "the stall held until its release; got {released:?}"
    );
    tokio::time::timeout(REPLAY_WAIT, pipeline)
        .await
        .expect("the pipeline is answered after the release")
        .expect("the pipeline task does not panic");
    drop(member);
    client.quit().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// F pass B: one warm connection per client, the idle shape, and the routing
// table's owned strings (tests 17, 18, 19 and 25)
// ─────────────────────────────────────────────────────────────────────────────

/// The value each GET in tests 17, 18 and 19 reads: under the general read
/// limit (64 KiB), so the warm connection keeps the read buffer it grew for
/// it, and above 16 KiB, so a fresh 8 KiB read buffer has to grow again to
/// read it.
const WARM_VALUE_LEN: usize = 48 * 1024;

/// Calls in tests 18 and 19.
const WARM_CALLS: usize = 10;

/// The smallest allocation a read buffer growing from its fresh 8 KiB makes
/// (it at least doubles): one at or above this, made during a call, is a new
/// read buffer.
const NEW_READ_BUFFER_MIN: usize = 2 * FRESH_BUFFER_BYTES;

/// `GET key` on `client`; the reply is checked to be [`WARM_VALUE_LEN`]
/// bytes and dropped.
async fn get_warm_value(client: &fred::clients::Client, key: &str) {
    use fred::interfaces::KeysInterface;
    let reply: Bytes = client.get(key).await.expect("GET of the warm value");
    assert_eq!(
        reply.len(),
        WARM_VALUE_LEN,
        "fixture self-check: the reply is the value"
    );
}

/// Run [`WARM_CALLS`] GETs of `key` on `client` one at a time, returning,
/// for each call, the allocations of at least [`NEW_READ_BUFFER_MIN`] it made
/// and the largest of them.
async fn warm_gets(client: &fred::clients::Client, key: &str) -> Vec<(usize, usize)> {
    let mut calls = Vec::with_capacity(WARM_CALLS);
    for _ in 0..WARM_CALLS {
        begin_large_count(NEW_READ_BUFFER_MIN);
        get_warm_value(client, key).await;
        calls.push(end_large_count());
    }
    calls
}

/// The calls among `calls` (1-based) from `first` on that made an allocation
/// of at least [`NEW_READ_BUFFER_MIN`], described.
fn calls_that_grew_a_read_buffer(calls: &[(usize, usize)], first: usize) -> Vec<String> {
    calls
        .iter()
        .enumerate()
        .skip(first - 1)
        .filter(|(_, (made, _))| *made > 0)
        .map(|(i, (made, largest))| {
            format!(
                "call {}: {made} allocations >= {NEW_READ_BUFFER_MIN} B, largest {largest} B",
                i + 1
            )
        })
        .collect()
}

/// Store a [`WARM_VALUE_LEN`]-byte value at `key` through `tv`'s own client.
async fn store_warm_value(tv: &TestValkey, key: &str) {
    use fred::interfaces::KeysInterface;
    let _: () = tv
        .client()
        .pool
        .set(key, vec![b'w'; WARM_VALUE_LEN], None, None, false)
        .await
        .expect("SET of the warm value");
}

/// Test 18 (b.26e F3): a client with one connection treats it as warm, so it
/// keeps the read buffer its class limits allow between calls. Ten GETs of a
/// [`WARM_VALUE_LEN`]-byte value, one at a time, on the one connection of a
/// data-pool member (pool of 1, built by the production `connect`): the first
/// grows the connection's read buffer to hold the reply; calls 2–10 make no
/// allocation of [`NEW_READ_BUFFER_MIN`] or more. A connection not treated as
/// warm is cut to the idle shape at every drain (an 8 KiB read buffer), and
/// each call regrows it.
///
/// After the calls, the member's retention report shows its one connection
/// warm, with read progress, and the warm rule holds
/// ([`warm_rule_violations`]); without the sweep it fails there, on V3. Then
/// the negative control ([`assert_warm_rule_discriminates`]): with the sweep
/// skipped the check reports V3, and back on the sweep as built nothing. This
/// is the local proof of the retention report's wiring (fred's
/// `live-edit-retention-report` feature through live-edit's `test-support`),
/// its plumbing and V3, on a standalone store.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn a_single_connection_client_stays_warm_so_repeated_replies_allocate_no_new_read_buffer_b26e(
) {
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let mut config = common::test_config();
    config.valkey_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own client connects");
    let member = client.pool.clients()[0].clone();
    let key = format!("{}t18:value", tv.prefix());
    store_warm_value(&tv, &key).await;

    let calls = warm_gets(&member, &key).await;
    eprintln!("== b.26e test 18 (standalone): (allocations >= {NEW_READ_BUFFER_MIN} B, largest) per call: {calls:?} ==");
    let label = "test 18's data member";
    let report = connection_report(label, &member).await;
    eprintln!(
        "== b.26e test 18 (standalone): retention report {} ==",
        describe_reports(&[(label.to_owned(), report.clone())])
    );
    assert_eq!(
        report.len(),
        1,
        "fixture self-check: a standalone client holds one connection: {report:?}"
    );
    let violations = warm_rule_violations(label, GENERAL_LIMITS, &report);
    assert!(
        violations.is_empty(),
        "a one-connection client's connection must be its warm one: {violations:?}"
    );
    assert!(
        report[0].read_progress,
        "fixture self-check: the connection made read progress: {report:?}"
    );
    assert_warm_rule_discriminates(label, &member, false).await;
    assert!(
        calls[0].0 > 0,
        "fixture self-check: the first call grew the read buffer from its fresh 8 KiB: {calls:?}"
    );
    let regrown = calls_that_grew_a_read_buffer(&calls, 2);
    assert!(
        regrown.is_empty(),
        "a one-connection client must keep its connection warm, so its read buffer survives the \
         drain between calls; these calls grew a new one: {regrown:?}"
    );
    drop(member);
    client.quit().await;
}

/// The cluster URL variable the cluster-gated tests below read
/// (`tests/integration/storage/mod.rs` names it the same, and CI's
/// `test-cluster:` job sets it).
const CLUSTER_URL_VAR: &str = "LIVE_EDIT_TEST_CLUSTER_URL";

/// For a cluster-gated test: `LIVE_EDIT_TEST_CLUSTER_URL`, or `None`, with a
/// `SKIP:` line printed, when the variable is unset or not a
/// `redis-cluster://` URL. A set but unreachable cluster fails the test that
/// connects to it.
fn cluster_url(test: &str) -> Option<String> {
    let Ok(url) = std::env::var(CLUSTER_URL_VAR) else {
        eprintln!(
            "SKIP: {test}: {CLUSTER_URL_VAR} not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run it"
        );
        return None;
    };
    if !url.starts_with("redis-cluster://") {
        eprintln!("SKIP: {test}: {CLUSTER_URL_VAR} is not a redis-cluster:// URL");
        return None;
    }
    Some(url)
}

/// The topology `at_rest_bound_bytes` is computed for on `member`'s store:
/// standalone for a single-node client, else a cluster of the primaries its
/// routing table names ([`topology_of_routing`]).
fn topology_of(member: &fred::clients::Client) -> Topology {
    use fred::interfaces::ClusterInterface;
    let routing = member
        .is_clustered()
        .then(|| member.cached_cluster_state())
        .flatten();
    assert_eq!(
        routing.is_some(),
        member.is_clustered(),
        "fixture self-check: a connected cluster client has its routing table"
    );
    topology_of_routing(routing.as_ref())
}

/// [`topology_of`]'s rule: no routing table is a standalone store; a routing
/// table is a cluster of `unique_primary_nodes().len()` primaries (P).
fn topology_of_routing(routing: Option<&fred::types::cluster::ClusterRouting>) -> Topology {
    routing.map_or(Topology::standalone(), |routing| {
        Topology::cluster(routing.unique_primary_nodes().len())
    })
}

/// For a cluster-gated test: a [`TestValkey`] on `LIVE_EDIT_TEST_CLUSTER_URL`
/// (for its prefix and cleanup) and a client of the test's own, built by the
/// production `connect` with pools of 1, so its data pool's one member is a
/// clustered general client with a connection to every primary. `None`, with
/// a `SKIP:` line printed, without a cluster ([`cluster_url`]).
async fn cluster_client(
    test: &str,
) -> Option<(TestValkey, live_edit::storage::client::ValkeyClient)> {
    let url = cluster_url(test)?;
    let tv = TestValkey::setup_with_url(url.clone(), 1).await;
    let mut config = common::test_config();
    config.valkey_url = url;
    config.valkey_pool_size = 1;
    let client = connect_test_client(&config)
        .await
        .expect("the test's own cluster client connects");
    assert!(
        client.pool.clients()[0].is_clustered(),
        "{CLUSTER_URL_VAR} must point at a clustered deployment"
    );
    Some((tv, client))
}

/// One hash tag per primary of `member`'s cluster: a tag whose slot that
/// primary owns, so a key carrying it is read on that primary's connection.
/// In the routing table's primary order.
fn tag_per_primary(member: &fred::clients::Client) -> Vec<String> {
    use fred::interfaces::ClusterInterface;
    let routing = member
        .cached_cluster_state()
        .expect("a connected cluster client has its routing");
    routing
        .unique_primary_nodes()
        .iter()
        .map(|primary| {
            (0..)
                .map(|i| format!("t{i}"))
                .find(|tag| {
                    routing.get_server(fred::util::redis_keyslot(tag.as_bytes())) == Some(primary)
                })
                .expect("some tag hashes to every primary")
        })
        .collect()
}

/// What a clustered general client keeps once drained, with `primaries`
/// connections: one warm connection at the general class limits, every other
/// connection at the idle shape (an 8 KiB read buffer, no queue, no write
/// buffer), and its `CLUSTER SLOTS` backchannel's two 8 KiB buffers —
/// `505,856 + (P − 1) × 8,192 + 16,384`, computed from the constants.
fn clustered_client_bound(primaries: usize) -> i64 {
    (GENERAL_LIMITS.kept_queue_bytes()
        + GENERAL_LIMITS.kept_write_bytes()
        + GENERAL_LIMITS.kept_read_bytes()
        + (primaries - 1) * FRESH_BUFFER_BYTES
        + BACKCHANNEL_BYTES) as i64
}

/// GETs per primary in tests 17 and 19: the general in-flight limit (256). A
/// pipeline at the limit grows its connection's queue to exactly the limit,
/// which the class rules keep, and its replies stay under the read limit, so
/// the class limits alone would let each connection keep what the pipeline
/// grew; its held replies are 12 MiB (256 × 48 KiB).
const PIPELINE_PER_PRIMARY: usize = GENERAL_LIMITS.in_flight_commands;

/// A pipeline of [`PIPELINE_PER_PRIMARY`] GETs of `key` on `member`.
async fn pipeline_of_warm_gets(member: &fred::clients::Client, key: &str) {
    use fred::interfaces::KeysInterface;
    let pipeline = member.pipeline();
    for _ in 0..PIPELINE_PER_PRIMARY {
        let _: () = pipeline
            .get(key)
            .await
            .expect("a GET queues in the pipeline");
    }
    let replies: Vec<Result<Bytes, fred::error::Error>> = pipeline.try_all().await;
    assert!(
        replies
            .iter()
            .all(|r| matches!(r, Ok(v) if v.len() == WARM_VALUE_LEN)),
        "fixture self-check: every GET answered the value"
    );
}

/// Tests 17 and 19's common part, on a cluster client's one data-pool
/// member: store a [`WARM_VALUE_LEN`]-byte value on every primary, take the
/// reference after one router pass, then run a [`PIPELINE_PER_PRIMARY`]-GET
/// pipeline on each primary in turn and one more GET on the last (the router
/// pass that follows the traffic: every command runs the sweep first).
/// Returns the member, each primary's key in order (the routing table's
/// primary order), and the reference.
async fn load_every_primary(
    tv: &TestValkey,
    client: &live_edit::storage::client::ValkeyClient,
    test: &str,
) -> (fred::clients::Client, Vec<String>, i64) {
    let member = client.pool.clients()[0].clone();
    let keys: Vec<String> = tag_per_primary(&member)
        .iter()
        .map(|tag| format!("{}{{{tag}}}:{test}", tv.prefix()))
        .collect();
    assert!(
        keys.len() >= 2,
        "fixture self-check: the cluster has at least two primaries"
    );
    for key in &keys {
        store_warm_value(tv, key).await;
    }
    let _: () = member.ping(None).await.expect("PING: one router pass");
    tokio::time::sleep(SETTLE).await;
    let reference = live_now();
    for key in &keys {
        pipeline_of_warm_gets(&member, key).await;
    }
    get_warm_value(&member, keys.last().expect("checked above")).await;
    tokio::time::sleep(SETTLE).await;
    (member, keys, reference)
}

/// Each primary of `member`'s cluster as its connections' `server` reads
/// (`host:port`), in the routing table's primary order: the order
/// [`tag_per_primary`] and [`load_every_primary`] use.
fn primary_servers(member: &fred::clients::Client) -> Vec<String> {
    use fred::interfaces::ClusterInterface;
    member
        .cached_cluster_state()
        .expect("a connected cluster client has its routing")
        .unique_primary_nodes()
        .iter()
        .map(ToString::to_string)
        .collect()
}

/// The warm connection of `report`, when exactly one is warm.
fn the_warm_connection(report: &[ConnectionReport]) -> Option<&ConnectionReport> {
    let mut warm = report.iter().filter(|c| c.warm);
    let first = warm.next()?;
    warm.next().is_none().then_some(first)
}

/// Test 17 (b.26e F3, cluster-gated): idle connections keep the idle shape.
/// One clustered general client (pool of 1, the production pool builder)
/// carries a [`PIPELINE_PER_PRIMARY`]-GET pipeline of a
/// [`WARM_VALUE_LEN`]-byte value on every primary, the last primary's last,
/// then one more GET there. Each pipeline is at the class limits, so the
/// per-connection limits alone would let every connection keep it (the
/// 256-slot queue, 96,256 B, on each).
///
/// After the router pass that follows, the member's retention report must
/// show P connections, every one with read progress and drained
/// (self-checks); the warm rule ([`warm_rule_violations`]): one warm
/// connection, every other one cut to the idle shape; and the warm one is
/// the last primary's, its queue at exactly the limit. That is
/// deterministic: the read loop takes at most one frame per connection per
/// poll, so warmth moves to a pipeline's connection after its first reply
/// and stays while it carries the rest, and that connection keeps the
/// queue the pipeline grew to the limit. Every connection warm (the
/// per-connection rule alone) fails V1, no sweep fails V3, a cut that frees
/// nothing fails V2. Then the standing negative controls
/// ([`assert_warm_rule_discriminates`]) show on this cluster that the check
/// reports a skipped sweep and every connection warm.
///
/// Its bytes: the heap is at most [`clustered_client_bound`] above the
/// reference taken before the traffic (538,624 B on three primaries), formula
/// (B)'s per-client term: one warm connection at the class limits, the others
/// at the idle shape, and the `CLUSTER SLOTS` backchannel. That bound alone
/// does not tell one warm connection from several at these limits; the report
/// does.
///
/// Runs only with `LIVE_EDIT_TEST_CLUSTER_URL` (CI's `test-cluster:` job);
/// prints a `SKIP:` line otherwise. The election and cut decisions it rests
/// on are covered everywhere by the pure test 20
/// (`storage::connection_buffers::tests`), and the warm rule's by
/// `warm_rule_violations_tell_one_warm_connection_from_none_or_several_b26e`.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cluster_idle_connections_keep_the_idle_shape_after_traffic_on_every_primary_b26e() {
    let _serial = measure_lock().await;
    let test = "t17";
    let Some((tv, client)) = cluster_client(test).await else {
        return;
    };
    let (member, keys, reference) = load_every_primary(&tv, &client, test).await;
    let kept = live_now() - reference;
    let bound = clustered_client_bound(keys.len());
    let label = format!("{test}'s data member");
    let report = connection_report(&label, &member).await;
    eprintln!(
        "CLUSTER-GATED RAN: {test} primaries={}: kept {kept} B after the traffic (bound {bound} \
         B); retention report {}",
        keys.len(),
        describe_reports(&[(label.clone(), report.clone())])
    );
    assert!(
        report.len() == keys.len() && report.iter().all(|c| c.read_progress && c.drained),
        "fixture self-check: the member has a connection to each of the {} primaries, every one \
         with read progress (each carried a pipeline) and drained: {report:?}",
        keys.len()
    );
    let violations = warm_rule_violations(&label, GENERAL_LIMITS, &report);
    assert!(
        violations.is_empty(),
        "after a {PIPELINE_PER_PRIMARY}-GET pipeline on each primary the member must keep one \
         warm connection and every other one in the idle shape: {violations:?}"
    );
    let last = primary_servers(&member)
        .pop()
        .expect("checked above: the cluster has primaries");
    let warm = the_warm_connection(&report).expect("checked above: one warm connection");
    assert_eq!(
        (warm.server.as_str(), warm.queue_capacity),
        (last.as_str(), GENERAL_LIMITS.in_flight_commands),
        "the warm connection is the last primary's, the one that carried the last traffic, and \
         keeps the queue its pipeline grew to the limit: {report:?}"
    );
    drop(report);
    assert_warm_rule_discriminates(&label, &member, true).await;
    assert!(
        kept <= bound,
        "a clustered client kept {kept} B after carrying a {PIPELINE_PER_PRIMARY}-GET pipeline on \
         each of {} primaries, past one warm connection plus the idle shape on the others \
         ({bound} B): a connection other than the warm one kept its class limits",
        keys.len()
    );
    drop(member);
    client.quit().await;
}

/// Test 19 (b.26e F3, cluster-gated): warmth follows traffic. After test 17's
/// traffic, whose last call leaves the last primary's connection warm with
/// its 256-slot queue (checked on the member's retention report first), ten
/// GETs on the first primary only:
///
/// - by the report taken after the first call (answered by a router pass
///   after it), warmth has moved to the first primary's connection, and the
///   old warm connection is cut to the idle shape with no queue: it let go of
///   its whole 256-slot queue, `in_flight_commands × IN_FLIGHT_SLOT_BYTES` =
///   96,256 B, read exactly off the two reports. The heap's drop by the end
///   of the second call is printed beside it, not asserted: the old warm
///   connection's read buffer is guaranteed only at least about 48 KiB (a
///   pipelined reply still held when it reallocated keeps the larger of the
///   partial frame and the carried-over capacity), so the bytes it frees can
///   fall a few KB short of any fixed figure on correct code;
/// - calls 3–10 make no allocation of [`NEW_READ_BUFFER_MIN`] or more: the
///   new warm connection keeps its read buffer.
///
/// An election that never moves leaves the old connection warm, which fails
/// the identity check at once (and cuts the new one at every drain, so every
/// call regrows a read buffer). An election that moved on every reply cannot
/// be told apart here, where one connection carries all the traffic; test
/// 20's in-flight and tie rows hold that warmth stays put.
///
/// Runs only with `LIVE_EDIT_TEST_CLUSTER_URL` (CI's `test-cluster:` job);
/// prints a `SKIP:` line otherwise. The election rows are covered everywhere
/// by the pure test 20.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cluster_warmth_follows_traffic_to_another_primary_b26e() {
    let _serial = measure_lock().await;
    let test = "t19";
    let Some((tv, client)) = cluster_client(test).await else {
        return;
    };
    let (member, keys, _) = load_every_primary(&tv, &client, test).await;
    let servers = primary_servers(&member);
    let (first_server, last_server) = (servers[0].clone(), servers[servers.len() - 1].clone());
    let label = format!("{test}'s data member");
    let connection_to = |report: &[ConnectionReport], server: &str| -> usize {
        report
            .iter()
            .position(|c| c.server == server)
            .unwrap_or_else(|| panic!("{label}: no connection to {server}: {report:?}"))
    };
    let report = connection_report(&label, &member).await;
    let old_queue_before = report[connection_to(&report, &last_server)].queue_capacity;
    let violations = warm_rule_violations(&format!("{label} before"), GENERAL_LIMITS, &report);
    assert!(
        violations.is_empty()
            && the_warm_connection(&report).is_some_and(|c| c.server == last_server)
            && old_queue_before == GENERAL_LIMITS.in_flight_commands,
        "fixture self-check: after test 17's traffic the last primary's ({last_server}) \
         connection is warm with a {}-slot queue: {report:?}; {violations:?}",
        GENERAL_LIMITS.in_flight_commands
    );
    drop((report, violations));
    let first = &keys[0];
    let before = live_now();
    let mut calls = Vec::with_capacity(WARM_CALLS);
    let mut after_second = 0;
    let mut old_queue_after = None;
    let mut not_moved = None;
    for call in 1..=WARM_CALLS {
        begin_large_count(NEW_READ_BUFFER_MIN);
        get_warm_value(&member, first).await;
        calls.push(end_large_count());
        if call == 1 {
            // Outside the call's window, and dropped before the next
            // live-bytes read: only plain figures outlive it, and a
            // description only on a failure.
            let report = connection_report(&label, &member).await;
            let old = &report[connection_to(&report, &last_server)];
            let violations =
                warm_rule_violations(&format!("{label} after call 1"), GENERAL_LIMITS, &report);
            let moved = the_warm_connection(&report).is_some_and(|c| c.server == first_server);
            old_queue_after = Some(old.queue_capacity);
            not_moved = (!moved || !in_idle_shape(old) || !violations.is_empty()).then(|| {
                format!(
                    "after the first call on the first primary ({first_server}) warmth must have \
                     moved to its connection within one router pass, and the old warm one \
                     ({last_server}) been cut to the idle shape: {report:?}; {violations:?}"
                )
            });
        }
        if call == 2 {
            after_second = live_now();
        }
    }
    let old_queue_after = old_queue_after.expect("the report after call 1 was taken");
    let released = (old_queue_before - old_queue_after) * IN_FLIGHT_SLOT_BYTES;
    let dropped = before - after_second;
    eprintln!(
        "CLUSTER-GATED RAN: {test} primaries={}: after the first call on the first primary the \
         old warm connection released {released} B of queue (want {} B); the heap dropped \
         {dropped} B by the second call; (allocations >= {NEW_READ_BUFFER_MIN} B, largest) per \
         call: {calls:?}",
        keys.len(),
        GENERAL_LIMITS.kept_queue_bytes()
    );
    if let Some(not_moved) = not_moved {
        panic!("{not_moved}");
    }
    assert_eq!(
        released,
        GENERAL_LIMITS.in_flight_commands * IN_FLIGHT_SLOT_BYTES,
        "the old warm connection let go of its whole {}-slot queue",
        GENERAL_LIMITS.in_flight_commands
    );
    let regrown = calls_that_grew_a_read_buffer(&calls, 3);
    assert!(
        regrown.is_empty(),
        "the connection carrying the traffic must stay warm and keep its read buffer: {regrown:?}"
    );
    drop(member);
    client.quit().await;
}

/// Test 18's cluster twin (cluster-gated): on a clustered general client, a
/// connection that carries all the traffic is the warm one and keeps its
/// read buffer. Two unmeasured GETs of a [`WARM_VALUE_LEN`]-byte value on one
/// primary first: warmth moves to its connection at the router pass after
/// the first (the connection warm at connect may be another primary's), and
/// the second grows its read buffer. The member's retention report then shows
/// that primary's connection warm, under the warm rule. Then ten more GETs
/// make no allocation of [`NEW_READ_BUFFER_MIN`] or more.
///
/// Runs only with `LIVE_EDIT_TEST_CLUSTER_URL`; prints a `SKIP:` line
/// otherwise.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cluster_the_connection_carrying_the_traffic_stays_warm_b26e() {
    let _serial = measure_lock().await;
    let test = "t18c";
    let Some((tv, client)) = cluster_client(test).await else {
        return;
    };
    let member = client.pool.clients()[0].clone();
    let tag = tag_per_primary(&member).remove(0);
    let tag_primary = primary_servers(&member).remove(0);
    let key = format!("{}{{{tag}}}:{test}", tv.prefix());
    store_warm_value(&tv, &key).await;

    begin_large_count(NEW_READ_BUFFER_MIN);
    get_warm_value(&member, &key).await;
    get_warm_value(&member, &key).await;
    let warm_up = end_large_count();
    let label = format!("{test}'s data member");
    let report = connection_report(&label, &member).await;
    let violations = warm_rule_violations(&label, GENERAL_LIMITS, &report);
    assert!(
        violations.is_empty()
            && the_warm_connection(&report).is_some_and(|c| c.server == tag_primary),
        "after the two warm-ups the connection carrying the traffic, the tag's primary's \
         ({tag_primary}), must be the warm one: {report:?}; {violations:?}"
    );
    drop((report, violations));
    let calls = warm_gets(&member, &key).await;
    eprintln!(
        "CLUSTER-GATED RAN: {test}: the two warm-up calls made {warm_up:?}; (allocations >= \
         {NEW_READ_BUFFER_MIN} B, largest) per measured call: {calls:?}"
    );
    assert!(
        warm_up.0 > 0,
        "fixture self-check: the warm-up grew the read buffer: {warm_up:?}"
    );
    let regrown = calls_that_grew_a_read_buffer(&calls, 1);
    assert!(
        regrown.is_empty(),
        "the connection carrying all the traffic must be warm and keep its read buffer: \
         {regrown:?}"
    );
    drop(member);
    client.quit().await;
}

/// Topology syncs in test 25.
const SYNCS: usize = 4;

/// What test 25 allows the syncs to leave: the routing table's own copies of
/// its strings (a few hundred bytes per slot range) and allocator noise. A
/// `CLUSTER SLOTS` reply kept alive by a routing string is a whole read
/// buffer, at least 8,192 B.
const SYNC_RESIDUE_LIMIT: i64 = 4_096;

/// Test 25 (b.26e F4b, cluster-gated): a clustered general client's routing
/// table owns its strings, so no `CLUSTER SLOTS` reply outlives its sync. On
/// a pool-1 clustered client from the production builder, with the
/// reference taken after connect and one router pass, [`SYNCS`] topology
/// syncs (`sync_cluster`), each followed by a `PING` (a router pass, which
/// cuts the backchannel the sync used), leave the heap less than
/// [`SYNC_RESIDUE_LIMIT`] above the reference. Without the copy at the end of
/// `ClusterRouting::rebuild`, each sync's reply allocation stays pinned by
/// the routing state, connection keys and the next backchannel's server, one
/// or more allocations of at least 8,192 B.
///
/// Runs only with `LIVE_EDIT_TEST_CLUSTER_URL` (CI's `test-cluster:` job);
/// prints a `SKIP:` line otherwise. The copy itself is covered everywhere by
/// the pure test 24 (`storage::connection_buffers::tests`).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cluster_topology_syncs_leave_no_reply_buffer_pinned_by_the_routing_table_b26e() {
    use fred::interfaces::ClusterInterface;
    let _serial = measure_lock().await;
    let test = "t25";
    let Some((_tv, client)) = cluster_client(test).await else {
        return;
    };
    let member = client.pool.clients()[0].clone();
    let _: () = member.ping(None).await.expect("PING: one router pass");
    tokio::time::sleep(SETTLE).await;
    let reference = live_now();
    for _ in 0..SYNCS {
        member.sync_cluster().await.expect("a topology sync");
        let _: () = member.ping(None).await.expect("PING after the sync");
    }
    tokio::time::sleep(SETTLE).await;
    let kept = live_now() - reference;
    eprintln!(
        "CLUSTER-GATED RAN: {test}: {SYNCS} syncs left {kept} B above the reference (limit \
         {SYNC_RESIDUE_LIMIT} B)"
    );
    assert!(
        kept < SYNC_RESIDUE_LIMIT,
        "{SYNCS} topology syncs left {kept} B live: a CLUSTER SLOTS reply's read buffer is kept \
         alive by the routing table's strings"
    );
    drop(member);
    client.quit().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8 (b.26e f3b): the sweep allocates nothing on an idle pass
// ─────────────────────────────────────────────────────────────────────────────

/// The value test 8 reads: 1 KiB, far under the general read limit, so no
/// class rule fires on the connection carrying it.
const IDLE_VALUE_LEN: usize = 1024;

/// GETs in test 8's measured stretch, one at a time: each is its own command,
/// so its own router pass, and every pass sweeps first.
const IDLE_SWEEP_CALLS: usize = 8;

/// The fewest sweeps test 8's measured stretch must see, so a zero count
/// cannot come from sweeps that never ran (b.26e test 8: at least 4 idle
/// passes).
const MIN_IDLE_SWEEPS: u64 = 4;

/// Store an [`IDLE_VALUE_LEN`]-byte value at `key` through `tv`'s own client.
async fn store_idle_value(tv: &TestValkey, key: &str) {
    use fred::interfaces::KeysInterface;
    let _: () = tv
        .client()
        .pool
        .set(key, vec![b'i'; IDLE_VALUE_LEN], None, None, false)
        .await
        .expect("SET of the idle value");
}

/// `GET key` on `client`; the reply is checked to be [`IDLE_VALUE_LEN`] bytes
/// and dropped.
async fn get_idle_value(client: &fred::clients::Client, key: &str) {
    use fred::interfaces::KeysInterface;
    let reply: Bytes = client.get(key).await.expect("GET of the idle value");
    assert_eq!(
        reply.len(),
        IDLE_VALUE_LEN,
        "fixture self-check: the reply is the value"
    );
}

/// A general-class fred client of test 8's own on `url`, standalone or
/// clustered: the general class limits, and the unresponsive check at the
/// production interval (500 ms), so its router also passes without traffic
/// and answers a retention report. Built bare rather than by the production
/// `connect`, whose other pools' clients would sweep on the measured thread
/// too, each cutting back after its own connect-time traffic at a time the
/// test does not control.
async fn sweep_test_client(url: &str) -> fred::clients::Client {
    let config = fred::types::config::Config::from_url(url).expect("the store URL parses");
    let mut builder = fred::types::Builder::from_config(config);
    builder.with_connection_config(|connection| {
        connection.live_edit.retention = Some(GENERAL_LIMITS);
        connection.unresponsive = fred::types::config::UnresponsiveConfig {
            max_timeout: Some(std::time::Duration::from_secs(45)),
            interval: std::time::Duration::from_millis(500),
        };
    });
    let client = builder.build().expect("test 8's client builds");
    client.init().await.expect("test 8's client connects");
    client
}

/// Run `scenario` on a thread of its own ([`SWEEP_THREAD`] set), on a
/// current-thread runtime: a client the scenario builds runs its router task
/// on that thread, so only its sweeps are counted. The scenario's panic is
/// re-raised here.
async fn on_sweep_thread<T, F, Fut>(scenario: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = T>,
{
    let thread = std::thread::Builder::new()
        .name("b26e-test-8-sweep".to_owned())
        .spawn(move || {
            SWEEP_THREAD.with(|flag| flag.set(true));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("the measured thread's runtime builds");
            runtime.block_on(scenario())
        })
        .expect("the measured thread starts");
    tokio::task::spawn_blocking(move || thread.join())
        .await
        .expect("the join of the measured thread")
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

/// What test 8 saw over its measured stretch.
#[derive(Debug)]
struct IdleSweeps {
    /// Sweeps the measured client ran (`sweep_stats` before and after).
    sweeps: u64,
    /// Allocations made inside sweeps on the measured thread.
    allocations: usize,
    /// The largest of them, in bytes.
    largest: usize,
}

/// [`IDLE_SWEEP_CALLS`] GETs of `key` on `client`, one at a time, with the
/// allocations made inside its sweeps counted.
async fn idle_sweeps(client: &fred::clients::Client, key: &str) -> IdleSweeps {
    use fred::live_edit::retention::test_support::sweep_stats;
    let before = sweep_stats(client);
    begin_sweep_count();
    for _ in 0..IDLE_SWEEP_CALLS {
        get_idle_value(client, key).await;
    }
    let (allocations, largest) = end_sweep_count();
    let after = sweep_stats(client);
    IdleSweeps {
        sweeps: after.passes - before.passes,
        allocations,
        largest,
    }
}

/// Test 8's assertions on `seen`: enough sweeps ran (a self-check, so the
/// zero below is not vacuous), and none of them allocated.
fn assert_idle_sweeps_allocate_nothing(test: &str, seen: &IdleSweeps) {
    assert!(
        seen.sweeps >= MIN_IDLE_SWEEPS,
        "fixture self-check: {test}: the measured client ran at least {MIN_IDLE_SWEEPS} sweeps \
         over {IDLE_SWEEP_CALLS} GETs: {seen:?}"
    );
    assert_eq!(
        seen.allocations, 0,
        "{test}: the connection-buffer sweep allocated on passes that changed no buffer ({} \
         sweeps, every connection drained and already in its settled state; the largest \
         allocation {} B): the sweep must walk the router's connections without collecting \
         them, their views or their actions (retention::sweep, liveness::connections_mut)",
        seen.sweeps, seen.largest
    );
}

/// Test 8 (b.26e f3b): fred's connection-buffer sweep allocates nothing on a
/// pass that changes no buffer. A standalone general-class client
/// ([`sweep_test_client`]) on a thread of its own ([`on_sweep_thread`]) reads
/// an [`IDLE_VALUE_LEN`]-byte value twice, unmeasured; its retention report
/// then shows its one connection warm, drained and with read progress, the
/// state every later pass keeps (the class rules have nothing to cut back on a
/// connection that carried 1 KiB replies). Then [`IDLE_SWEEP_CALLS`] more
/// GETs: the client runs at least [`MIN_IDLE_SWEEPS`] sweeps
/// (`test_support::sweep_stats`), and the counting allocator sees no
/// allocation made while fred's in-sweep flag was set on that thread. A sweep
/// that collects the router's connections into a `Vec` on every pass makes
/// one allocation per pass here.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn retention_sweep_allocates_nothing_on_an_idle_pass_b26e() {
    let _serial = measure_lock().await;
    let test = "test 8 (standalone)";
    let tv = TestValkey::setup().await;
    let key = format!("{}t8:value", tv.prefix());
    store_idle_value(&tv, &key).await;
    let url = common::test_config().valkey_url;
    let (report, seen) = on_sweep_thread(move || async move {
        let client = sweep_test_client(&url).await;
        get_idle_value(&client, &key).await;
        get_idle_value(&client, &key).await;
        let report = connection_report("test 8's client", &client).await;
        let seen = idle_sweeps(&client, &key).await;
        client.quit().await.expect("test 8's client quits");
        (report, seen)
    })
    .await;
    eprintln!(
        "== b.26e {test}: {seen:?} over {IDLE_SWEEP_CALLS} GETs; retention report before: {} ==",
        describe_reports(&[("test 8's client".to_owned(), report.clone())])
    );
    assert!(
        report.len() == 1 && report[0].warm && report[0].drained && report[0].read_progress,
        "fixture self-check: {test}: the client's one connection is warm, drained and has made \
         read progress before the measured GETs: {report:?}"
    );
    assert_idle_sweeps_allocate_nothing(test, &seen);
}

/// The server, `host:port`, of the primary that owns `key`'s slot in
/// `client`'s routing table: the `server` its connection's report entry
/// carries.
fn primary_of(client: &fred::clients::Client, key: &str) -> String {
    use fred::interfaces::ClusterInterface;
    client
        .cached_cluster_state()
        .expect("a connected cluster client has its routing")
        .get_server(fred::util::redis_keyslot(key.as_bytes()))
        .expect("every slot has a primary")
        .to_string()
}

/// Test 8's cluster twin (cluster-gated): on a clustered general-class client
/// ([`sweep_test_client`]) with a connection to every primary, the sweep
/// allocates nothing on a pass that changes no buffer. Unmeasured first: one
/// GET of an [`IDLE_VALUE_LEN`]-byte value on every primary, the first
/// primary's last, then one more there, so every connection made read
/// progress, warmth moved to the first primary's connection and every other
/// connection was cut to the idle shape (the sweep's cuts allocate, which is
/// why they come before the measurement). The retention report shows that
/// settled state: one connection per primary, all drained, the warm one the
/// first primary's, the warm rule holding ([`warm_rule_violations`]). Then
/// [`IDLE_SWEEP_CALLS`] GETs on the first primary: warmth stays, every other
/// connection is already idle, and the sweeps, at least [`MIN_IDLE_SWEEPS`],
/// make no allocation. A sweep that collects the connections, their views
/// and their actions makes three allocations per pass here.
///
/// Runs only with `LIVE_EDIT_TEST_CLUSTER_URL` (CI's `test-cluster:` job);
/// prints a `SKIP:` line otherwise. The standalone test above covers the
/// sweep everywhere, and the election the cluster sweep folds is covered by
/// the pure test 20 (`storage::connection_buffers::tests`).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn cluster_retention_sweep_allocates_nothing_on_an_idle_pass_b26e() {
    let _serial = measure_lock().await;
    let test = "t8c";
    let Some(url) = cluster_url(test) else {
        return;
    };
    let tv = TestValkey::setup_with_url(url.clone(), 1).await;
    let keys: Vec<String> = tag_per_primary(&tv.client().pool.clients()[0])
        .iter()
        .map(|tag| format!("{}{{{tag}}}:{test}", tv.prefix()))
        .collect();
    assert!(
        keys.len() >= 2,
        "fixture self-check: the cluster has at least two primaries"
    );
    for key in &keys {
        store_idle_value(&tv, key).await;
    }
    let primaries = keys.len();
    let (report, warm_server, seen) = on_sweep_thread(move || async move {
        let client = sweep_test_client(&url).await;
        assert!(
            client.is_clustered(),
            "{CLUSTER_URL_VAR} must point at a clustered deployment"
        );
        for key in keys.iter().rev() {
            get_idle_value(&client, key).await;
        }
        get_idle_value(&client, &keys[0]).await;
        let report = connection_report("t8c's client", &client).await;
        let warm_server = primary_of(&client, &keys[0]);
        let seen = idle_sweeps(&client, &keys[0]).await;
        client.quit().await.expect("t8c's client quits");
        (report, warm_server, seen)
    })
    .await;
    eprintln!(
        "CLUSTER-GATED RAN: {test} primaries={primaries}: {seen:?} over {IDLE_SWEEP_CALLS} GETs \
         on {warm_server}; retention report before: {}",
        describe_reports(&[("t8c's client".to_owned(), report.clone())])
    );
    let violations = warm_rule_violations("t8c's client", GENERAL_LIMITS, &report);
    assert!(
        report.len() == primaries
            && report.iter().all(|c| c.read_progress && c.drained)
            && violations.is_empty()
            && the_warm_connection(&report).is_some_and(|c| c.server == warm_server),
        "fixture self-check: {test}: before the measured GETs the client has a drained \
         connection with read progress to each of the {primaries} primaries, the first \
         primary's ({warm_server}) warm and every other one in the idle shape: {report:?}; \
         {violations:?}"
    );
    assert_idle_sweeps_allocate_nothing(test, &seen);
}

/// b.26e attribution, report only: what an `UpdateLayer` of the `mixed`
/// fixture on a cached structural route leaves live when it returns, beyond
/// the structural-cache entry it stores or grows. One envelope run measured
/// 13.1 MB left at return on the cold start, of which invalidating its entry
/// freed 3.69 MB; a later one measured 3.34 MB there (all of it the entry)
/// and 13.09 MB on the warm route. For each cached route, on a client of the
/// test's own (production `connect`) and a fresh service, every allocation
/// of the measured call is tracked; the live tracked allocations are printed
/// by site at return, after the object's cache entry is invalidated, after
/// the service is dropped, and after the client quits. `B26E_BACKTRACE_MIN`
/// sets the smallest allocation backtraced (default 16 KiB).
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "manual: report only — b.26e cached-route mixed residue attribution"]
async fn cached_route_mixed_residue_is_attributed_b26e() {
    let _serial = measure_lock().await;
    for route in [Route::CachedCold, Route::CachedWarm] {
        attribute_mixed_residue(route).await;
    }
}

/// One route of [`cached_route_mixed_residue_is_attributed_b26e`].
async fn attribute_mixed_residue(route: Route) {
    let tv = TestValkey::setup().await;
    let client = connect_test_client(&common::test_config())
        .await
        .expect("the report's own client connects");
    let path = Path::UpdateSlow;
    // Unmeasured: scripts, metric series and the route's once-per-process
    // code, through a service on this client.
    for (i, warmup) in [
        Fixture::header_only(),
        Fixture::new(
            "warmup-structural",
            Feeds::Mixed,
            OwnedDelta {
                struct_commands: children(1, false),
                ..Default::default()
            },
            &[],
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let svc = isolated_service(&tv, route, client.clone());
        let label = format!("residue-warmup-{i}");
        let input = upload_frames(&tv, path, &warmup, &label).await;
        upload_once(&svc, path, &input.frames, "residue-warmup", 0).await;
    }
    let fixture = fixtures(path)
        .into_iter()
        .find(|f| f.name == "mixed")
        .expect("the mixed fixture");
    let svc = isolated_service(&tv, route, client.clone());
    let (session_id, object_id) = base_layer(&tv, "residue-mixed", fixture.base).await;
    if route == Route::CachedWarm {
        // Unmeasured, as `measure_on` does: one structural commit through this
        // store loads the object into its structural cache.
        let warm = OwnedDelta {
            struct_commands: vec![create_section(0, WARM_SECTION, "warm".to_owned())],
            ..Default::default()
        };
        let relay = Bytes::from(common::owned_delta::delta_to_ompf_bytes(&warm, &[]));
        Box::pin(svc.layer_store.update_layer(
            &session_id,
            "layer",
            object_id,
            Some(warm),
            Some(relay),
            "residue-warmup",
            0,
        ))
        .await
        .expect("the cache warm-up commits");
    }
    let chunks: Vec<Vec<u8>> = fixture.ompf.chunks(CHUNK).map(<[u8]>::to_vec).collect();
    let frames: Vec<Bytes> = std::iter::once(grpc_frame(&make_update_header(
        &session_id,
        "layer",
        object_id,
    )))
    .chain(chunks.into_iter().map(|c| grpc_frame(&make_update_data(c))))
    .collect();
    drop(fixture);
    tokio::time::sleep(SETTLE).await;

    let backtrace_min = std::env::var("B26E_BACKTRACE_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(POST_RELEASE_BACKTRACE_MIN);
    let reference = (live_now(), glibc_now());
    start_tracking(backtrace_min);
    upload_once(&svc, path, &frames, "residue-mixed", 1).await;
    let at_return = live_now() - reference.0;
    stop_recording();
    eprintln!(
        "== b.26e residue: {route:?} mixed left {at_return} B live at return (glibc {} B) ==",
        glibc_now() - reference.1
    );
    report_tracked(&format!("{route:?} mixed, at return"), 12);

    let mut steps = vec![teardown_step("after SETTLE", reference).await];
    svc.layer_store.structural_cache().invalidate(object_id);
    steps.push(teardown_step("cache entry invalidated", reference).await);
    report_tracked(
        &format!("{route:?} mixed, after the cache entry is invalidated"),
        12,
    );
    drop(svc);
    steps.push(teardown_step("service dropped", reference).await);
    report_tracked(&format!("{route:?} mixed, after the service is dropped"), 8);
    client.quit().await;
    drop(client);
    steps.push(teardown_step("client quit and dropped", reference).await);
    report_tracked(&format!("{route:?} mixed, after the client is gone"), 8);
    eprintln!(
        "{:<28} {:>14} {:>14} {:>14} {:>9}",
        "step", "live-left", "glibc-left", "tracked-left", "tracked#"
    );
    for s in &steps {
        eprintln!(
            "{:<28} {:>14} {:>14} {:>14} {:>9}",
            s.step, s.live, s.glibc, s.tracked_bytes, s.tracked_count
        );
    }
    end_tracking();
    drop(frames);
}

/// The allocation tracker never deadlocks a panicking thread. With every
/// allocation backtraced, a panic whose hook captures and prints a backtrace
/// — as std's default hook does when `RUST_BACKTRACE` is set, holding std's
/// non-reentrant backtrace lock while it allocates — runs to its end.
#[test]
fn allocation_tracker_lets_a_panic_hook_capture_a_backtrace() {
    const SELF_CHECK: &str = "tracker-self-check";
    let _serial = MEASURE.blocking_lock();
    let previous = std::panic::take_hook();
    // Only the self-check thread's panic captures a backtrace; any other
    // panic — this test's own failure — prints its message without one, so
    // a failure is reported even while a deadlocked thread holds the
    // backtrace lock.
    std::panic::set_hook(Box::new(|info| {
        if std::thread::current().name() == Some(SELF_CHECK) {
            let backtrace = std::backtrace::Backtrace::force_capture();
            drop(backtrace.to_string());
        } else {
            eprintln!("{info}");
        }
    }));
    start_tracking(1);
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name(SELF_CHECK.to_owned())
        .spawn(move || {
            let _ = std::panic::catch_unwind(|| panic!("tracker self-check"));
            let _ = done_tx.send(());
        })
        .expect("the self-check thread starts");
    let finished = done_rx.recv_timeout(std::time::Duration::from_secs(30));
    end_tracking();
    if finished.is_ok() {
        // A deadlocked hook still holds the hook lock, which `set_hook` would
        // wait for forever; so the previous hook comes back only on success.
        std::panic::set_hook(previous);
    }
    assert!(
        finished.is_ok(),
        "the panicking thread did not finish within 30 s: the tracker's backtrace capture \
         waited on the backtrace lock its own panic hook held"
    );
}

/// The phase checks see a reservation released and charged again mid-call:
/// a gap with nothing charged is checked against nothing, so heap the call
/// holds through it is an overrun — unless the next phase holds the overflow
/// slot, when the gap is a step into the slot and is checked against the
/// slot's charge.
#[test]
fn phase_limits_check_a_mid_call_gap_against_nothing_unless_the_slot_follows() {
    let at = |reserved_bytes, overflow_bytes, peak| Phase {
        charge: GateCharge {
            reserved_bytes,
            overflow_bytes,
        },
        peak,
        t_max: 0,
    };
    let recharged = [
        at(0, 0, 100),
        at(1_000, 0, 500),
        at(0, 0, 800),
        at(1_000, 0, 900),
        at(0, 0, 50),
    ];
    assert_eq!(
        phase_limits(&recharged, 20),
        vec![1_000, 1_000, 0, 1_000, 1_020]
    );
    assert_eq!(phase_labels(&recharged, 0, 0)[2], "uncharged");

    let slot_entry = [
        at(0, 0, 100),
        at(1_000, 0, 500),
        at(0, 0, 800),
        at(0, 5_000, 4_000),
        at(0, 0, 50),
    ];
    assert_eq!(
        phase_limits(&slot_entry, 20),
        vec![1_000, 1_000, 5_000, 5_000, 5_020]
    );
    assert_eq!(phase_labels(&slot_entry, 0, 0)[2], "slot-entry");
}

/// Test 16 (b.26e F): the released phase is checked against the last charge
/// plus the structural-cache entry a cold start stored, and nothing else the
/// call left live. A released-phase peak above the last charge fails though
/// the call left that much live ([`Measured::residual`]), and passes once
/// the cold start's entry covers it.
#[test]
fn released_phase_allows_the_cold_start_cache_entry_and_nothing_else_the_call_left() {
    let at = |reserved_bytes, peak| Phase {
        charge: GateCharge {
            reserved_bytes,
            overflow_bytes: 0,
        },
        peak,
        t_max: 0,
    };
    let measured = |residual, cache_entry| Measured {
        path: Path::UpdateSlow,
        route: Route::CachedCold,
        name: "released-phase-self-check",
        feeds: Feeds::Mixed,
        body: 0,
        delta: 0,
        ext: 0,
        elements: 0,
        reorder: 0,
        peak: 1_300,
        charge: 1_000,
        phases: vec![at(0, 100), at(1_000, 900), at(0, 1_300)],
        parse_target: 0,
        processing_target: 0,
        residual,
        cache_entry,
        object_id: 0,
        wall: std::time::Duration::ZERO,
        plans: PlanCounts::default(),
    };
    let left_live = measured(400, 0);
    assert_eq!(
        left_live.phase_overruns().len(),
        1,
        "300 B over the last charge in the released phase is an overrun, however much the call \
         left live: {:?}",
        left_live.phase_overruns()
    );
    assert!(
        left_live.phase_overruns()[0].contains("released"),
        "the overrun is the released phase's: {:?}",
        left_live.phase_overruns()
    );
    let cold_start = measured(400, 400);
    assert!(
        cold_start.phase_overruns().is_empty(),
        "a cold start's stored entry is allowed in the released phase: {:?}",
        cold_start.phase_overruns()
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// b.fsz: one replayed record in process — the audit's copy, the cached entry
// ═════════════════════════════════════════════════════════════════════════════
//
// The sampled audit repairs a copy of a cache entry that holds no plan
// indexes (`CacheEntry::copy_without_plan`), holding each record's need on
// the upload budget first (RP2, `storage::structural_cache_repair`); the
// cached route replays onto the entry with its indexes. These rows build a
// state in process, decode one record outside the measured window, and
// measure, inside it, the record's apply and the normalization that follows
// it — `apply_decoded_record_into` and the round's `normalize_children` and
// `normalize_state` on the copy; `apply_decoded_record_indexed` and
// `normalize_touched` on an indexed entry — against what RP2 holds beside the
// entry: the record's growth and rewrite (`estimate_replay_record_growth`;
// on the copy `estimate_replay_record_growth_bounded`) and its reply, parse
// and per-byte growth, its need's price less the entry it is held beside. No
// Valkey: the reply term then covers nothing the window holds, so these rows
// check the record's own terms.

/// The per-entry cap on a cached state at the default structural-cache
/// budget (a quarter of 256 MiB): the size the audit's large rows are at.
const AUDIT_ENTRY_CAP: u64 = 64 * MIB as u64;

/// One record's replay measured in process: the entry's weight it is held
/// beside, the window's peak, RP2's hold beyond that weight (its need's
/// price less it), the record's priced growth, scratch and rewrite, its
/// delta part, and the state's estimated weight after it with the bound RP2
/// keeps for it (the weight before, the priced growth and the per-byte
/// growth term).
#[derive(Debug)]
struct ReplayedRecord {
    weight: u64,
    peak: u64,
    hold: u64,
    growth: u64,
    scratch: u64,
    rewrite: RewriteCount,
    delta: u64,
    after: u64,
    bound: u64,
}

/// `repair_growth`, in hundredths per delta byte, read back through the
/// charge (it is crate-private).
fn repair_growth_centi() -> u64 {
    let price = |replayed| {
        object_state_charge_for_test(&ObjectStateNeed::cached_repair(
            0,
            RepairTransient {
                replay_delta_bytes: replayed,
                ..RepairTransient::default()
            },
        ))
    };
    (price(1_000_000) - price(0)) / 10_000
}

/// Replay `delta` onto `state` in process, as the audit's copy (no plan
/// indexes) when `indexes` is `None`, else as the cached entry holding them
/// (`indexes` follow the record), and measure it ([`ReplayedRecord`]).
/// `state` must stand at its `sequence_num`. When the entry has `dangling`
/// parents the record replays through the list-delete sink
/// (`MutationSink::ListDeletes`), else through the discarding sink — the
/// sinks the repair picks the same way (`structural_cache_repair`'s
/// `replay_chunk`, which this in-process replay cannot call). RP2 holds the
/// record's priced growth and scratch together
/// (`ReplayRecordGrowth::growth` + `scratch`), as `RecordCharge::try_hold`
/// does; the bound on the state after keeps its growth alone, since the
/// scratch is freed before the record returns.
fn replay_one_record(
    state: &mut live_edit::domain::types::CompactState,
    indexes: Option<&mut live_edit::domain::plan_index::PlanIndexes>,
    delta: &OwnedDelta,
    dangling: bool,
) -> ReplayedRecord {
    use live_edit::domain::delta::MutationSink;
    use live_edit::domain::plan_index::{normalize_touched, PlanTouches};
    use live_edit::domain::replay::{
        apply_decoded_record_indexed, apply_decoded_record_into, decode_stream_record,
        normalize_children, normalize_state,
    };
    use live_edit::storage::structural_cache::{
        estimate_replay_record_growth, estimate_replay_record_growth_bounded, estimate_state_bytes,
        CopyBucketBound, HeldCapacities,
    };

    let ompf = common::owned_delta::delta_to_ompf_bytes(delta, &[]);
    let delta_bytes = OmpfContainer::parse(&ompf)
        .expect("the record's OMPF parses")
        .delta_bytes
        .len() as u64;
    let seq = state.sequence_num + 1;
    let record =
        decode_stream_record(state.sequence_num, Some(&ompf), seq).expect("the record decodes");
    let weight = estimate_state_bytes(state)
        + indexes
            .as_deref()
            .map_or(0, |indexes| indexes.weight_bytes_recomputed());
    // RP2's estimate: on the indexed entry at its held capacities; on the
    // copy from its bucket bound (`unindexed_record_growth`), started, as for
    // a copy of an entry built by inserts alone, at its live capacities.
    let estimate = match indexes.as_deref() {
        Some(indexes) => {
            estimate_replay_record_growth(state, Some(indexes), record.delta(), dangling)
        }
        None => {
            let bound = CopyBucketBound::new(state, HeldCapacities::live(state));
            estimate_replay_record_growth_bounded(state, &bound, record.delta(), dangling).0
        }
    };
    let need = ObjectStateNeed::cached_repair_record(
        weight,
        estimate.growth + estimate.scratch,
        estimate.rewrite,
        RepairTransient {
            reply_entry_count: 1,
            reply_bytes: ompf.len() as u64,
            parse_delta_bytes: delta_bytes,
            replay_delta_bytes: delta_bytes,
        },
    );
    let hold = object_state_charge_for_test(&need) - weight;
    let mut sink = if dangling {
        MutationSink::list_deletes()
    } else {
        MutationSink::Discard
    };
    let ((), peak) = peak_of(|| match indexes {
        Some(indexes) => {
            let mut touches = PlanTouches::default();
            apply_decoded_record_indexed(state, record, &mut sink, indexes, &mut touches)
                .expect("the record applies");
            normalize_touched(state, indexes, &touches.lists);
        }
        None => {
            apply_decoded_record_into(state, record, &mut sink).expect("the record applies");
            normalize_children(state);
            normalize_state(state);
        }
    });
    drop(sink);
    let after = estimate_state_bytes(state);
    let bound = weight
        .saturating_add(estimate.growth)
        .saturating_add((delta_bytes * repair_growth_centi()).div_ceil(100));
    ReplayedRecord {
        weight,
        peak,
        hold,
        growth: estimate.growth,
        scratch: estimate.scratch,
        rewrite: estimate.rewrite,
        delta: delta_bytes,
        after,
        bound,
    }
}

/// One `REPLAY-RECORD` report line for `row`.
fn report_replayed(row: &str, r: &ReplayedRecord) {
    eprintln!(
        "REPLAY-RECORD {row}: weight {} B; peak {} B; RP2 hold beyond the entry {} B \
         (peak/hold {:.3}); growth {} B; scratch {} B; rewrite {:?}; delta {} B; after {} B \
         against the bound {} B",
        r.weight,
        r.peak,
        r.hold,
        r.peak as f64 / r.hold.max(1) as f64,
        r.growth,
        r.scratch,
        r.rewrite,
        r.delta,
        r.after,
        r.bound
    );
}

/// Sections and fields for an in-process base: `sections` sections, in
/// groups of up to `per_group` under the root (the group ids from 1, their
/// children from 10,000,000), and `fields` small inline field rows spread
/// over the sections in order, up to eight per section, named `f0`…`f7`.
fn audit_base(sections: u64, fields: u64, per_group: u64) -> ObjectState {
    use live_edit::domain::types::{Field, FieldValue, Section};
    let section = |id: u64, parent: u64, order: u32| Section {
        section_id: id,
        parent_id: parent,
        name: format!("s{id}"),
        section_type: 1,
        section_order: order,
        first_version: 1,
        last_version: 1,
    };
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..ObjectState::default()
    };
    let groups = sections.div_ceil(per_group).max(1);
    let mut ids: Vec<u64> = Vec::with_capacity(sections as usize);
    let mut root = Vec::new();
    for group in 1..=groups.min(sections) {
        state
            .sections
            .insert(group, section(group, 0, root.len() as u32));
        root.push(group);
        ids.push(group);
    }
    state.children.insert(0, root);
    let mut next = 10_000_000;
    let mut group = 1;
    while (ids.len() as u64) < sections {
        let list = state.children.entry(group).or_default();
        state
            .sections
            .insert(next, section(next, group, list.len() as u32));
        list.push(next);
        ids.push(next);
        next += 1;
        group = group % groups + 1;
    }
    for (i, &id) in ids.iter().cycle().take(fields as usize).enumerate() {
        let name = format!("f{}", i as u64 / sections);
        state.fields.insert(
            (id, name.clone()),
            Field {
                section_id: id,
                key_name: name,
                value: FieldValue::Inline(vec![1; 8]),
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );
    }
    state
}

/// [`audit_base`] in groups of 1,000, compacted as a stored entry is.
fn audit_entry_state(sections: u64, fields: u64) -> live_edit::domain::types::CompactState {
    let mut state = live_edit::domain::types::compact(&audit_base(sections, fields, 1_000));
    live_edit::storage::structural_cache::compact_state(&mut state);
    for list in state.children.values_mut() {
        list.shrink_to_fit();
    }
    state
}

/// What a section and a field row of [`audit_entry_state`] weigh, as
/// `estimate_state_bytes` charges them, each read off a small state.
fn audit_unit_weights() -> (u64, u64) {
    use live_edit::storage::structural_cache::estimate_state_bytes;
    let sections = estimate_state_bytes(&audit_entry_state(10_000, 0));
    let with_fields = estimate_state_bytes(&audit_entry_state(10_000, 40_000));
    (sections / 10_000, (with_fields - sections) / 40_000)
}

/// Whether `built` is within [0.95, 1.05] of `weight`.
fn near_weight(built: u64, weight: u64) -> bool {
    built * 20 >= weight * 19 && built * 20 <= weight * 21
}

/// Bring `state` up to about `weight` bytes, as `estimate_state_bytes`
/// weighs it, without changing any table's layout: the tables' weight steps
/// with their bucket counts, so the counts fix the shape and the sections'
/// names carry the rest. Each name, in id order, takes the same few bytes
/// more — the shortfall spread over every section, rounded up — until the
/// state reaches the weight, each step counted as the estimate counts a name
/// (its allocation), so the state lands within one name's step of it.
/// Panics, as a fixture self-check, when the state is past [`near_weight`]
/// of `weight`.
fn pad_to_weight(state: &mut live_edit::domain::types::CompactState, weight: u64) {
    use live_edit::domain::alloc_model::allocation_bytes;
    use live_edit::storage::structural_cache::estimate_state_bytes;
    let mut built = estimate_state_bytes(state);
    if built < weight {
        let mut ids: Vec<u64> = state.sections.keys().copied().collect();
        ids.sort_unstable();
        let per_name = (weight - built).div_ceil(ids.len().max(1) as u64) + 1;
        for id in ids {
            if built >= weight {
                break;
            }
            let row = state.sections.get_mut(&id).expect("a section");
            let old = allocation_bytes(row.name.capacity() as u64);
            let mut name = std::mem::take(&mut row.name);
            name.push_str(&"p".repeat(per_name as usize));
            name.shrink_to_fit();
            built += allocation_bytes(name.capacity() as u64) - old;
            row.name = name;
        }
    }
    let built = estimate_state_bytes(state);
    assert!(
        near_weight(built, weight),
        "fixture self-check: the padded state weighs {built} B, within [0.95, 1.05] of {weight} B"
    );
}

/// The largest table length at a load-limit boundary — 7/8 of 2^k buckets,
/// k ≥ 4 — at most `limit`.
fn load_limit_at_most(limit: u64) -> u64 {
    let mut buckets = 16u64;
    while (buckets * 2) / 8 * 7 <= limit {
        buckets *= 2;
    }
    buckets / 8 * 7
}

/// The next load-limit boundary below `length` (itself one).
fn load_limit_below(length: u64) -> u64 {
    (length / 7 * 8 / 2) / 8 * 7
}

/// The three record shapes of the audit's envelope rows (b.fsz PD2 §4).
#[derive(Clone, Copy, Debug)]
enum AuditShape {
    /// "Place one asset": two sections created under one group, the second
    /// under the first, an existing section moved under the new asset root,
    /// and fields on the new ones. The base is two sections short of a
    /// load-limit boundary, so the sections table takes the two births
    /// without growing, while the per-delta name index — the sections, the
    /// births and the move — lands just past 7/8 of its buckets, its worst
    /// ratio. (Just past 7/8 by births alone would double the sections table
    /// too, whose growth would then dominate the record's need and hide the
    /// index-less terms this shape is for.)
    PlaceAsset,
    /// One field set into a fields table at its load limit.
    FieldIntoFullTable,
    /// One section created into a sections table at its load limit.
    SectionIntoFullTable,
}

/// The base and the record of `shape` at about `weight` bytes: the table
/// the shape is about at its largest load-limit boundary whose base, with no
/// padding, weighs at most 0.95 of `weight` (from a per-unit estimate, then
/// one boundary lower while it is over), then padded to the weight
/// ([`pad_to_weight`]).
fn audit_row(
    shape: AuditShape,
    weight: u64,
) -> (live_edit::domain::types::CompactState, OwnedDelta) {
    use live_edit::domain::types::FieldRows;
    use live_edit::storage::structural_cache::estimate_state_bytes;
    let (per_section, per_field) = audit_unit_weights();
    let birth = |parent: u64, id: u64| create_section(parent, id, format!("asset{id}"));
    let set = |section_id: u64, key: &str| SetField {
        section_id,
        ..field(key.to_owned(), vec![2; 16])
    };
    let room = weight / 20 * 19;
    // The base of the shape at a boundary `length`.
    let base_at = |length: u64| match shape {
        AuditShape::PlaceAsset => audit_entry_state(length - 2, 0),
        AuditShape::SectionIntoFullTable => audit_entry_state(length, 0),
        AuditShape::FieldIntoFullTable => audit_entry_state(length / 8 + 1, length),
    };
    let per_unit = match shape {
        AuditShape::FieldIntoFullTable => per_field + per_section / 8,
        _ => per_section,
    };
    let mut length = load_limit_at_most(room / per_unit.max(1));
    let mut state = base_at(length);
    while estimate_state_bytes(&state) > room {
        length = load_limit_below(length);
        state = base_at(length);
    }
    pad_to_weight(&mut state, weight);
    let delta = match shape {
        AuditShape::PlaceAsset => OwnedDelta {
            struct_commands: vec![
                birth(1, 900_000_001),
                birth(900_000_001, 900_000_002),
                // The first group's first child, re-parented under the asset.
                StructCommand::MoveSection(CommandMoveSection {
                    old_parent_id: 1,
                    new_parent_id: 900_000_001,
                    section_id: 10_000_000,
                    new_name: "placed".to_owned(),
                }),
            ],
            set_fields: vec![
                set(900_000_001, "xformOp:translate"),
                set(900_000_002, "typeName"),
            ],
            ..Default::default()
        },
        AuditShape::FieldIntoFullTable => {
            assert_eq!(
                state.fields.row_room(),
                0,
                "fixture self-check: the fields table is at its load limit"
            );
            // A name no row holds: a new row (and interned name) on a
            // section that exists.
            OwnedDelta {
                set_fields: vec![set(1, "brand-new")],
                ..Default::default()
            }
        }
        AuditShape::SectionIntoFullTable => {
            assert_eq!(
                state.sections.len(),
                state.sections.capacity(),
                "fixture self-check: the sections table is at its load limit"
            );
            OwnedDelta {
                struct_commands: vec![birth(1, 900_000_001)],
                ..Default::default()
            }
        }
    };
    eprintln!(
        "AUDIT-SIZE {shape:?}: {} sections, {} field rows, {} B",
        state.sections.len(),
        state.fields.len(),
        estimate_state_bytes(&state)
    );
    (state, delta)
}

/// One audit envelope row: `shape` at about `weight`, its record replayed
/// onto the copy, which holds no plan indexes. The window's peak must be
/// within RP2's hold on the upload budget beyond the copy (its need's
/// price less the copy's weight: the audit charge's own target net of the
/// copy, `UploadBudgetAuditCharges`), and the copy's estimated weight after
/// it within the bound RP2 keeps for it; the line reports the B target and
/// whether it is within the per-audit cap, past which the audit skips.
fn assert_audit_record_within_its_hold(row: &str, shape: AuditShape, weight: u64) {
    let _serial = MEASURE.blocking_lock();
    let (mut copy, delta) = audit_row(shape, weight);
    let built = live_edit::storage::structural_cache::estimate_state_bytes(&copy);
    assert!(
        near_weight(built, weight),
        "fixture self-check: {row}'s state weighs {built} B, within [0.95, 1.05] of {weight} B \
         (the cap rows' target, 0.95 of the 64 MiB cap, so within [0.9, 1.0] of the cap)"
    );
    let replayed = replay_one_record(&mut copy, None, &delta, false);
    report_replayed(row, &replayed);
    eprintln!(
        "AUDIT-ROW {row}: B target {} B against the per-audit cap {} B ({})",
        replayed.hold,
        live_edit::grpc::upload_admission::SAMPLED_AUDIT_CHARGE_CAP_BYTES,
        if replayed.hold > live_edit::grpc::upload_admission::SAMPLED_AUDIT_CHARGE_CAP_BYTES {
            "over the cap: the audit skips"
        } else {
            "within the cap"
        }
    );
    assert!(
        replayed.peak <= replayed.hold,
        "b.fsz {row}: the audit copy's replay of one record peaked at {} B, past its RP2 hold of \
         {} B: {replayed:?}",
        replayed.peak,
        replayed.hold
    );
    assert!(
        replayed.after <= replayed.bound,
        "b.fsz {row}: the copy's weight after the record ({} B) is past what RP2 keeps for it \
         ({} B): {replayed:?}",
        replayed.after,
        replayed.bound
    );
}

/// The sampled audit's repair of one "place one asset" record onto its
/// index-less copy of an ~8 MiB entry stays within its hold on the upload
/// budget: the per-delta name index (just past 7/8 of its buckets, its worst
/// ratio), the adoption index and the full depth walk are its rewrite, the
/// sections table taking the births without growing. Holds at either group
/// width: the indexes are built at their projected capacity, insert-only,
/// and the tables' bucket counts past a capacity of 15 do not depend on the
/// width.
#[test]
fn audit_replay_of_an_asset_placement_on_an_8_mib_entry_stays_within_its_hold_fsz() {
    assert_audit_record_within_its_hold(
        "fsz-audit-asset-8mib",
        AuditShape::PlaceAsset,
        8 * MIB as u64,
    );
}

/// [`audit_replay_of_an_asset_placement_on_an_8_mib_entry_stays_within_its_hold_fsz`]
/// at the per-entry cap ([0.9, 1.0] × 64 MiB).
#[test]
fn audit_replay_of_an_asset_placement_at_the_entry_cap_stays_within_its_hold_fsz() {
    assert_audit_record_within_its_hold(
        "fsz-audit-asset-cap",
        AuditShape::PlaceAsset,
        AUDIT_ENTRY_CAP * 95 / 100,
    );
}

/// The audit's replay of one field set into a fields table at its load
/// limit on an ~8 MiB entry stays within its hold: the doubled table is the
/// record's priced growth. Holds at either group width: insert-only, at a
/// capacity past 15.
#[test]
fn audit_replay_of_a_field_into_a_full_fields_table_on_an_8_mib_entry_stays_within_its_hold_fsz() {
    assert_audit_record_within_its_hold(
        "fsz-audit-field-table-8mib",
        AuditShape::FieldIntoFullTable,
        8 * MIB as u64,
    );
}

/// [`audit_replay_of_a_field_into_a_full_fields_table_on_an_8_mib_entry_stays_within_its_hold_fsz`]
/// at the per-entry cap.
#[test]
fn audit_replay_of_a_field_into_a_full_fields_table_at_the_entry_cap_stays_within_its_hold_fsz() {
    assert_audit_record_within_its_hold(
        "fsz-audit-field-table-cap",
        AuditShape::FieldIntoFullTable,
        AUDIT_ENTRY_CAP * 95 / 100,
    );
}

/// The audit's replay of one section created into a sections table at its
/// load limit on an ~8 MiB entry stays within its hold: the doubled table,
/// beside the per-delta indexes and walk. Holds at either group width:
/// insert-only, at a capacity past 15.
#[test]
fn audit_replay_of_a_section_into_a_full_sections_table_on_an_8_mib_entry_stays_within_its_hold_fsz(
) {
    assert_audit_record_within_its_hold(
        "fsz-audit-section-table-8mib",
        AuditShape::SectionIntoFullTable,
        8 * MIB as u64,
    );
}

/// [`audit_replay_of_a_section_into_a_full_sections_table_on_an_8_mib_entry_stays_within_its_hold_fsz`]
/// at the per-entry cap.
#[test]
fn audit_replay_of_a_section_into_a_full_sections_table_at_the_entry_cap_stays_within_its_hold_fsz()
{
    assert_audit_record_within_its_hold(
        "fsz-audit-section-table-cap",
        AuditShape::SectionIntoFullTable,
        AUDIT_ENTRY_CAP * 95 / 100,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// b.fsz dq-1: RP2's scratch — cascades, resets and the list-delete sink
// ─────────────────────────────────────────────────────────────────────────────
//
// A replayed record's apply allocates scratch it frees before it returns,
// none of which the entry keeps: each cascade's subtree walk (its stack,
// result and visited set), the plan's `deleted_lists` and `purge_ids`, with
// plan indexes the record's touched-list set, the compact field purge's
// probe vector and the name interner's free list; a reset's registrations;
// and, through the list-delete sink, the sink's ids and the one mutation it
// builds at a time. RP2 prices it per record in `ReplayRecordGrowth::scratch`
// (`domain::delta::cascade_scratch_bytes` at the pushes `subtree_pushes`
// counts over the subtrees the record can delete, `reset_scratch_bytes`,
// `list_delete_sink_bytes`), held beside the record's growth while it
// applies and never kept in the audit copy's held growth. Before it, a
// cascade's scratch was left to the per-byte growth term, which does not
// scale with the deleted subtree: one 104-byte `DeleteSection` of 1,000
// sections peaked at 43,848 B on an indexed entry against a 9,505 B hold.
//
// The rows replay one record in process ([`replay_one_record`]) on each
// holder — an entry with plan indexes (the cached route) and its index-less
// copy (the sampled audit) — through each sink the repair picks: the
// discarding sink (no dangling parents) and the list-delete sink
// (`MutationSink::ListDeletes`, dangling parents). The copy's index-less
// rewrite (the per-delta name and adoption indexes and the full depth walk,
// every section) covers a cascade's scratch on its own, so only the indexed
// arms discriminate the scratch term; the copy's arms validate it.
//
// Why each row holds at hashbrown group width 8 (this aarch64 sandbox) and
// 16 (CI's x86_64) by construction: every collection the scratch model
// prices starts empty in its plan or sink and is only inserted into or
// pushed onto, so no tombstone decides a growth; the model prices each table
// with 16 trailing control bytes, the wider group's, so a table at width 16
// allocates no more than it is priced at; bucket counts past a capacity of
// 15 do not depend on the width; and the model prices the walk's stack at
// its most for any visiting order, so a hash order that differs by width
// cannot push the walk past it. Where a record inserts into the entry's own
// tables, the row's self-check confirms they have room, so its growth is the
// same at either width and does not hide the scratch.

/// The stored shape `delta` creates from nothing: compacted as the cache
/// stores an entry (`compact_state`, every list trimmed to its length).
fn stored_shape_of(delta: OwnedDelta) -> live_edit::domain::types::CompactState {
    let mut state = compact_of(delta);
    live_edit::storage::structural_cache::compact_state(&mut state);
    for list in state.children.values_mut() {
        list.shrink_to_fit();
    }
    state
}

/// The id of the one section of `state` named `name` under `parent`.
fn section_named(state: &live_edit::domain::types::CompactState, parent: u64, name: &str) -> u64 {
    let ids: Vec<u64> = state
        .sections
        .values()
        .filter(|row| row.parent_id == parent && row.name == name)
        .map(|row| row.section_id)
        .collect();
    match ids[..] {
        [id] => id,
        _ => panic!("fixture: one section named {name:?} under {parent}; found {ids:?}"),
    }
}

/// The sections and field rows a record removed from `before`, read off
/// `after` — the rows' self-check of the subtree they delete.
fn removed_rows(
    before: &live_edit::domain::types::CompactState,
    after: &live_edit::domain::types::CompactState,
) -> (usize, usize) {
    use live_edit::domain::types::FieldRows;
    (
        before.sections.len() - after.sections.len(),
        before.fields.len() - after.fields.len(),
    )
}

/// The holders a scratch row replays on: with plan indexes (`true`, the
/// cached route) and without (`false`, the sampled audit's copy).
const BOTH_HOLDERS: [bool; 2] = [true, false];

/// The sinks a scratch row replays through: discarding (`false`) and the
/// list-delete sink (`true`).
const BOTH_SINKS: [bool; 2] = [false, true];

/// One scratch row (the section's header above): `delta` replayed once per
/// holder in `holders` and sink in `sinks` onto a clone of `base`, each arm's
/// window peak held to RP2's hold beyond the entry, and `check` run on the
/// state before and after each arm (the row's self-check of what the record
/// deleted). Answers each arm's measure, by its row name
/// (`fsz-scratch-{record}-{indexed|copy}-{discard|list-deletes}`).
fn assert_replay_scratch_within_its_hold(
    record: &str,
    base: &live_edit::domain::types::CompactState,
    delta: &OwnedDelta,
    holders: &[bool],
    sinks: &[bool],
    check: &dyn Fn(
        &live_edit::domain::types::CompactState,
        &live_edit::domain::types::CompactState,
    ),
) -> Vec<(String, ReplayedRecord)> {
    let _serial = MEASURE.blocking_lock();
    let mut arms = Vec::new();
    for &indexed in holders {
        for &list_deletes in sinks {
            let mut state = base.clone();
            let mut indexes =
                indexed.then(|| live_edit::domain::plan_index::PlanIndexes::build(&state));
            let row = format!(
                "fsz-scratch-{record}-{}-{}",
                if indexed { "indexed" } else { "copy" },
                if list_deletes {
                    "list-deletes"
                } else {
                    "discard"
                }
            );
            let replayed = replay_one_record(&mut state, indexes.as_mut(), delta, list_deletes);
            report_replayed(&row, &replayed);
            check(base, &state);
            arms.push((row, replayed));
        }
    }
    let failures: Vec<String> = arms
        .iter()
        .filter(|(_, r)| r.peak > r.hold)
        .map(|(row, r)| format!("{row}: peak {} B over its hold {} B", r.peak, r.hold))
        .collect();
    assert!(
        failures.is_empty(),
        "b.fsz: a replayed record's scratch passed its RP2 hold: {failures:?}"
    );
    arms
}

/// The deleted-count self-check of a scratch row: the record removed
/// exactly `sections` sections and `rows` field rows.
fn removes(sections: usize, rows: usize) -> impl Fn(&CompactState, &CompactState) {
    move |before, after| {
        assert_eq!(
            removed_rows(before, after),
            (sections, rows),
            "fixture self-check: the record removes (sections, field rows)"
        );
    }
}

/// The scratch rows' compact state type.
type CompactState = live_edit::domain::types::CompactState;

/// A `DeleteSection` of `section_id`, listed under `parent_id`.
fn delete_section(parent_id: u64, section_id: u64) -> StructCommand {
    StructCommand::DeleteSection(CommandDeleteSection {
        parent_id,
        section_id,
    })
}

/// A cascade's scratch (T1): one `DeleteSection` (104 delta bytes) of a
/// group of 1,000 sections, each carrying three field rows, on a
/// 20,000-section entry ([`audit_entry_state`]). Before RP2 priced a
/// cascade's scratch, the indexed entry peaked at 43,848 B against a
/// 9,505 B hold (4.6×): the per-byte growth term of a record this small
/// cannot carry a 1,000-section subtree. Now its hold carries the cascade
/// model at the group's 1,001 pushes. Zeroing `cascade_scratch_bytes` fails
/// the indexed arms. Widths 8 and 16: see the section header — the delete
/// inserts into no table of the entry.
#[test]
fn replay_scratch_of_a_cascade_stays_within_its_hold_fsz() {
    let (base, cascade) = scratch_cascade_row();
    assert_replay_scratch_within_its_hold(
        "cascade",
        &base,
        &cascade,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(1_000, 3_000),
    );
}

/// [`replay_scratch_of_a_cascade_stays_within_its_hold_fsz`]'s entry and
/// record (shared with the growth fit, [`growth_fit_rows`]).
fn scratch_cascade_row() -> (CompactState, OwnedDelta) {
    let cascade = OwnedDelta {
        struct_commands: vec![delete_section(0, 1)],
        ..Default::default()
    };
    (audit_entry_state(20_000, 60_000), cascade)
}

/// The list-delete sink's rows (T7): a record of 2,000 field writes
/// replayed through the list-delete sink, as the repair replays every record
/// of an entry that holds a dangling parent (`MutationSink::ListDeletes`,
/// which keeps the ids of the lists the record deletes and builds every
/// other mutation only to drop it). The sink holds one mutation at a time —
/// a `SetField`'s copy of its row is the record's own delta bytes, the
/// per-byte growth term's — so RP2's scratch prices the sink's floor (its
/// ids and two names of `MAX_NAME_BYTES` in flight) and nothing per row.
/// Under the collecting sink it replaced, the 2,000 rows' copies were all
/// alive at once. Widths 8 and 16: the sink is a `Vec`, which no group width
/// touches, and the fields table and name interner the 2,000 new rows grow
/// are priced as growth from their capacities, whose bucket counts past 15
/// do not depend on the width — the model's 16 trailing control bytes cover
/// the wider group.
#[test]
fn replay_scratch_of_the_list_delete_sink_stays_within_its_hold_fsz() {
    let (base, writes) = scratch_list_deleted_writes_row();
    assert_replay_scratch_within_its_hold(
        "list-deleted-writes",
        &base,
        &writes,
        &BOTH_HOLDERS,
        &[true],
        &|before, after| {
            use live_edit::domain::types::FieldRows;
            assert_eq!(
                after.fields.len(),
                before.fields.len() + 2_000,
                "fixture self-check: the record writes 2,000 new field rows"
            );
        },
    );
}

/// [`replay_scratch_of_the_list_delete_sink_stays_within_its_hold_fsz`]'s
/// entry and record (shared with the growth fit, [`growth_fit_rows`]).
fn scratch_list_deleted_writes_row() -> (CompactState, OwnedDelta) {
    let writes = OwnedDelta {
        set_fields: (0..2_000u64)
            .map(|i| SetField {
                section_id: 10_000_000 + i,
                ..field(format!("w{i}"), vec![3; 32])
            })
            .collect(),
        ..Default::default()
    };
    (audit_entry_state(20_000, 60_000), writes)
}

/// The dev shape's placed assets, as `Base::DevAssets` seeds them: 220
/// assets (`Group_0` and `Group_1` of 100, `Group_2` of 20) padded to
/// 7,167 sections.
const DEV_SHAPE_ASSETS: usize = 220;
/// [`DEV_SHAPE_ASSETS`]'s padded section count.
const DEV_SHAPE_SECTIONS: usize = 7_167;
/// A placed asset of 4 prims ([`dev_assets_base_delta`]): its root, 3 child
/// prims and 6 attribute sections on each of the 4 — 28 sections; and its
/// field rows, 3 on the root, 2 on each child prim and 4 on each attribute
/// section — 105.
const DEV_ASSET_OF_4_PRIMS: (usize, usize) = (28, 105);
/// A group of 100 placed assets, half of 4 prims and half of 5 (35 sections
/// and 131 rows each): the group's own section and 3,150 below it, and
/// 11,800 field rows.
const DEV_GROUP_OF_100: (usize, usize) = (3_151, 11_800);

/// [`DEV_SHAPE_ASSETS`] at [`DEV_SHAPE_SECTIONS`], stored.
fn dev_shape() -> CompactState {
    stored_shape_of(dev_assets_base_delta(DEV_SHAPE_ASSETS, DEV_SHAPE_SECTIONS))
}

/// The self-check that `state`'s sections and children tables take
/// `births` more rows without growing: the record's growth then prices no
/// table, at either group width, and does not hide its scratch.
fn assert_tables_have_room(state: &CompactState, births: u64) {
    use live_edit::domain::alloc_model::table_growth_bytes;
    assert_eq!(
        (
            table_growth_bytes(state.sections.len(), state.sections.capacity(), births, 1),
            table_growth_bytes(state.children.len(), state.children.capacity(), births, 1),
        ),
        (0, 0),
        "fixture self-check: the sections and children tables have room for {births} births"
    );
}

/// T2a, a validation row: a placed asset (`Asset_0`, 4 prims, 28 sections,
/// 105 field rows) deleted from the dev shape. Its group's listing (100
/// assets) carries most of the hold; the cascade of 28 pushes adds a few KB.
/// Widths 8 and 16: see the section header.
#[test]
fn replay_scratch_of_a_placed_asset_delete_on_the_dev_shape_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_dev_asset_delete_row();
    let (sections, rows) = DEV_ASSET_OF_4_PRIMS;
    assert_replay_scratch_within_its_hold(
        "dev-asset-delete",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(sections, rows),
    );
}

/// T2b: `Group_0` (100 placed assets, 3,151 sections, 11,800 field rows)
/// deleted from the dev shape. The world's listing is 4 entries, so no other
/// term covers the cascade on the indexed entry: zeroing
/// `cascade_scratch_bytes` fails the indexed arms. Widths 8 and 16: see the
/// section header.
#[test]
fn replay_scratch_of_a_group_delete_on_the_dev_shape_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_dev_group_delete_row();
    let (sections, rows) = DEV_GROUP_OF_100;
    assert_replay_scratch_within_its_hold(
        "dev-group-delete",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(sections, rows),
    );
}

/// T2c: a `CreateSection` of a new `Group_0` under the world, whose
/// same-name sibling — the dev shape's `Group_0`, 3,151 sections — it
/// cascades away. The record names no existing section: the cascade's root
/// is found only at the `(parent, name)` the creation introduces (with plan
/// indexes through `PlanIndexes::rows_named`, without them in the pass over
/// the sections). Dropping those conflict roots fails the indexed arms.
/// Self-check: the tables take the birth without growing. Widths 8 and 16:
/// see the section header.
#[test]
fn replay_scratch_of_a_same_name_creation_on_the_dev_shape_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_dev_same_name_creation_row();
    assert_tables_have_room(&base, 1);
    let (sections, rows) = DEV_GROUP_OF_100;
    assert_replay_scratch_within_its_hold(
        "dev-same-name-creation",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        // The newborn takes one of the removed sections' places.
        &removes(sections - 1, rows),
    );
}

/// T2d, a validation row: `Group_0` moved under a parent that does not
/// exist, which deletes it (`MoveSection`'s missing-parent edge) with its
/// 3,151 sections. A move carries a subtree into the depth check's walk
/// down, so RP2 prices the full walk too, which covers the cascade beside
/// it: this row validates the move's cascade term rather than singling it
/// out (the move's roots are pinned by the unit tests of
/// `estimate_replay_record_growth`). So dropping `MoveSection`'s cascade
/// roots from the estimate (the dq-1 perturbation P-n3) fails those unit
/// tests (T4a) and leaves this row green: the full walk term covers the
/// cascade. Widths 8 and 16: see the section header.
#[test]
fn replay_scratch_of_a_move_to_a_missing_parent_on_the_dev_shape_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_dev_move_to_missing_parent_row();
    let (sections, rows) = DEV_GROUP_OF_100;
    assert_replay_scratch_within_its_hold(
        "dev-move-to-missing-parent",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(sections, rows),
    );
}

/// T2e, a validation row: a `DiffChangeSection` re-parents `Group_0` (3,151
/// sections) under `Asset_200`'s root (`Group_2`'s first asset, 4 prims,
/// 28 sections), and a `DeleteSection` of that asset then cascades both —
/// 3,179 sections, 11,905 field rows. The re-parented group is a cascade
/// root only through its `DiffChangeSection`. A re-parent carries a subtree
/// into the depth check's walk down, so RP2 prices the full walk too, which
/// covers the cascade: this row validates the term (the `DiffChangeSection`
/// roots are pinned by the unit tests of `estimate_replay_record_growth`).
/// So dropping the `DiffChangeSection` roots (the dq-1 perturbation P-n4)
/// fails those unit tests (T4a) and leaves this row green: the full walk
/// term covers the cascade. Widths 8 and 16: see the section header.
#[test]
fn replay_scratch_of_a_reparent_then_delete_on_the_dev_shape_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_dev_reparent_then_delete_row();
    let (group_sections, group_rows) = DEV_GROUP_OF_100;
    let (asset_sections, asset_rows) = DEV_ASSET_OF_4_PRIMS;
    assert_replay_scratch_within_its_hold(
        "dev-reparent-then-delete",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(group_sections + asset_sections, group_rows + asset_rows),
    );
}

/// The orphaned subtree [`replay_scratch_of_an_adopting_birth_deleted_again_on_the_dev_shape_stays_within_its_hold_fsz`]
/// adds to the dev shape: a section naming a parent that does not exist
/// (made as a client makes one, by a `DiffChangeSection`, which stores no
/// list under the missing parent) with this many children of its own.
const ORPHANED_CHILDREN: u64 = 3_150;
/// The missing parent the orphaned subtree names.
const ORPHANS_PARENT: u64 = 7_000_000;
/// The orphaned subtree's root.
const ORPHANED_ROOT: u64 = 7_000_001;

/// T2e (births and orphans), a validation row: on the dev shape plus an
/// orphaned subtree (a section under the missing [`ORPHANS_PARENT`] with
/// [`ORPHANED_CHILDREN`] children), a record creates [`ORPHANS_PARENT`] —
/// which adopts the orphan and so its subtree — and deletes it again: the
/// cascade takes the birth, the orphan and its children. Nothing the record
/// names has a row before it, so the cascade's roots are its birth and the
/// orphan it adopts alone. An adopting birth carries a subtree into the depth
/// check's walk down, so RP2 prices the full walk too, which covers the
/// cascade: this row validates the term (births and orphans are pinned by
/// the unit tests of `estimate_replay_record_growth`). So dropping births
/// and orphans from the walk's roots (the dq-1 perturbation P-n11) fails
/// those unit tests (T4a) and leaves this row green: the full walk term
/// covers the cascade. Self-check: the tables take the birth (and its new
/// list) without growing. Widths 8 and 16: see the section header.
#[test]
fn replay_scratch_of_an_adopting_birth_deleted_again_on_the_dev_shape_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_dev_adopting_birth_deleted_row();
    assert_eq!(
        base.sections.get(&ORPHANED_ROOT).map(|row| row.parent_id),
        Some(ORPHANS_PARENT),
        "fixture self-check: the orphan names the missing parent"
    );
    assert_tables_have_room(&base, 1);
    assert_replay_scratch_within_its_hold(
        "dev-adopting-birth-deleted",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(1 + ORPHANED_CHILDREN as usize, 0),
    );
}

/// T2f: a section of 20,000 children (`Base::Children`) deleted — 20,001
/// sections. Its parent's listing is the root's single entry, so on the
/// indexed entry only the cascade term covers the walk's collections and
/// the purge's sets (about 0.9 MB): zeroing `cascade_scratch_bytes` fails
/// the indexed arms. Widths 8 and 16: see the section header.
#[test]
fn replay_scratch_of_a_20000_child_section_delete_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_20k_child_section_delete_row();
    assert_replay_scratch_within_its_hold(
        "20k-child-section-delete",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(DENSE + 1, 0),
    );
}

/// The reset record of the reset rows: the object's content replaced by
/// `SECTION` alone.
fn reset_to_one_section(name: &str) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![create_section(0, SECTION, name.to_owned())],
        ..Default::default()
    }
}

/// T2g: a reset of a 9,000-deep chain (`Base::Chain`, 9,001 sections and as
/// many children lists, each of one entry). A reset registers every list and
/// deletes every section's (`reset_scratch_bytes`), scratch that scales with
/// the lists, which the full walk's per-section term does not cover when
/// every section holds a list: zeroing the reset term fails the indexed
/// arms. Widths 8 and 16: see the section header — the reset's own tables
/// are built from empty, insert-only.
#[test]
fn replay_scratch_of_a_reset_of_a_deep_chain_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_chain_reset_row();
    assert_eq!(
        (base.sections.len(), base.children.len()),
        (SCRATCH_CHAIN + 1, SCRATCH_CHAIN + 1),
        "fixture self-check: every section but the bottom, and the root, hold a list"
    );
    assert_replay_scratch_within_its_hold(
        "chain-reset",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(SCRATCH_CHAIN, 0),
    );
}

/// The chain reset row's depth: a 9,000-deep chain.
const SCRATCH_CHAIN: usize = 9_000;

/// T2g at the dev shape, a validation row: the dev shape reset to its world
/// alone. Its lists are a fraction of its sections, so the full walk's term
/// covers the reset's registrations there.
#[test]
fn replay_scratch_of_a_reset_of_the_dev_shape_stays_within_its_hold_fsz() {
    use live_edit::domain::types::FieldRows;
    let (base, delta) = scratch_dev_reset_row();
    let rows = base.fields.len();
    assert_replay_scratch_within_its_hold(
        "dev-reset",
        &base,
        &delta,
        &BOTH_HOLDERS,
        &BOTH_SINKS,
        &removes(DEV_SHAPE_SECTIONS - 1, rows),
    );
}

/// T9, the listing pin (b.fsz dq-2): one child added under a parent of
/// 30,000 children whose list is at its length, replayed on an entry with
/// plan indexes. The replay builds the parent's listing, about 134 B per
/// entry, beside the list's doubling (16 B per entry) and the name vector's
/// growth; RP2 prices the listing as a rewrite count (`list_entries`, the
/// parent's 30,001 entries at the cached route's list-entry price). Zeroing
/// `list_entries` fails it. Zeroing RP2's list term leaves it green — the
/// listing's price covers the doubling — and so does zeroing its
/// name-vector term, which prices nothing here (the parent's name vector,
/// built with room, takes the record's entry without growing) — measured: so
/// this row pins the listing alone; those two terms are pinned by the unit
/// tests of `estimate_replay_record_growth` against std's real growth.
/// Self-checks:
/// the list is at its length, the sections table has room, and the record's
/// rewrite counts the whole listing. Widths 8 and 16: the list and the name
/// vector are `Vec`s, which no group width touches, and the sections table
/// has room at either width.
#[test]
fn replay_of_a_child_added_to_a_long_list_on_an_indexed_entry_stays_within_its_hold_fsz() {
    let (base, delta) = scratch_d2_long_list_row();
    let list = &base.children[&SECTION];
    assert_eq!(
        (list.len(), list.capacity()),
        (D2_CHILDREN, D2_CHILDREN),
        "fixture self-check: the long list is at its length"
    );
    assert_tables_have_room(&base, 1);
    let arms = assert_replay_scratch_within_its_hold(
        "d2-long-list",
        &base,
        &delta,
        &[true],
        &[false],
        &|before, after| {
            assert_eq!(
                after.sections.len(),
                before.sections.len() + 1,
                "fixture self-check: the record adds one section"
            );
        },
    );
    let [(_, replayed)] = &arms[..] else {
        panic!("fixture: one arm");
    };
    assert!(
        replayed.rewrite.list_entries > D2_CHILDREN as u64,
        "fixture self-check: RP2 prices the parent's whole listing: {:?}",
        replayed.rewrite
    );
}

/// T2a's entry and record: `Asset_0` deleted from the dev shape.
fn scratch_dev_asset_delete_row() -> (CompactState, OwnedDelta) {
    let base = dev_shape();
    let group = section_named(&base, SECTION, "Group_0");
    let asset = section_named(&base, group, "Asset_0");
    let delta = OwnedDelta {
        struct_commands: vec![delete_section(group, asset)],
        ..Default::default()
    };
    (base, delta)
}

/// T2b's entry and record: `Group_0` deleted from the dev shape.
fn scratch_dev_group_delete_row() -> (CompactState, OwnedDelta) {
    let base = dev_shape();
    let group = section_named(&base, SECTION, "Group_0");
    let delta = OwnedDelta {
        struct_commands: vec![delete_section(SECTION, group)],
        ..Default::default()
    };
    (base, delta)
}

/// T2c's entry and record: a new `Group_0` created under the world.
fn scratch_dev_same_name_creation_row() -> (CompactState, OwnedDelta) {
    let delta = OwnedDelta {
        struct_commands: vec![create_section(
            SECTION,
            PENDING_FIRST_ID * 900,
            "Group_0".to_owned(),
        )],
        ..Default::default()
    };
    (dev_shape(), delta)
}

/// T2d's entry and record: `Group_0` moved under a missing parent.
fn scratch_dev_move_to_missing_parent_row() -> (CompactState, OwnedDelta) {
    let base = dev_shape();
    let group = section_named(&base, SECTION, "Group_0");
    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: SECTION,
            new_parent_id: PENDING_FIRST_ID * 901,
            section_id: group,
            new_name: "Group_0".to_owned(),
        })],
        ..Default::default()
    };
    (base, delta)
}

/// T2e's (re-parent) entry and record: `Group_0` re-parented under
/// `Asset_200`'s root, then that asset deleted.
fn scratch_dev_reparent_then_delete_row() -> (CompactState, OwnedDelta) {
    let base = dev_shape();
    let group = section_named(&base, SECTION, "Group_0");
    let group_2 = section_named(&base, SECTION, "Group_2");
    let asset = section_named(&base, group_2, "Asset_200");
    let delta = OwnedDelta {
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: asset,
                section_id: group,
                section_name: "Group_0".to_owned(),
                section_type: 0,
                section_order: 0,
            }),
            delete_section(group_2, asset),
        ],
        ..Default::default()
    };
    (base, delta)
}

/// T2e's (births and orphans) entry and record: the dev shape plus an
/// orphaned subtree, and a record that creates the orphan's missing parent
/// and deletes it again.
fn scratch_dev_adopting_birth_deleted_row() -> (CompactState, OwnedDelta) {
    let mut seed = dev_assets_base_delta(DEV_SHAPE_ASSETS, DEV_SHAPE_SECTIONS);
    seed.struct_commands
        .push(StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ORPHANS_PARENT,
            section_id: ORPHANED_ROOT,
            section_name: "Orphaned".to_owned(),
            section_type: 0,
            section_order: 0,
        }));
    seed.struct_commands.extend(
        (1..=ORPHANED_CHILDREN)
            .map(|i| create_section(ORPHANED_ROOT, ORPHANED_ROOT + i, format!("o{i}"))),
    );
    let delta = OwnedDelta {
        struct_commands: vec![
            create_section(SECTION, ORPHANS_PARENT, "Adopter".to_owned()),
            delete_section(SECTION, ORPHANS_PARENT),
        ],
        ..Default::default()
    };
    (stored_shape_of(seed), delta)
}

/// T2f's entry and record: the 20,000-child section deleted.
fn scratch_20k_child_section_delete_row() -> (CompactState, OwnedDelta) {
    let delta = OwnedDelta {
        struct_commands: vec![delete_section(0, SECTION)],
        ..Default::default()
    };
    (stored_shape_of(base_delta(Base::Children(DENSE))), delta)
}

/// T2g's (chain) entry and record: a reset of the [`SCRATCH_CHAIN`]-deep
/// chain.
fn scratch_chain_reset_row() -> (CompactState, OwnedDelta) {
    (
        stored_shape_of(base_delta(Base::Chain(SCRATCH_CHAIN))),
        reset_to_one_section("base"),
    )
}

/// T2g's (dev shape) entry and record: the dev shape reset to its world.
fn scratch_dev_reset_row() -> (CompactState, OwnedDelta) {
    (dev_shape(), reset_to_one_section("World"))
}

/// T9's entry and record: one child added under a parent of
/// [`D2_CHILDREN`] children whose list is at its length.
fn scratch_d2_long_list_row() -> (CompactState, OwnedDelta) {
    let delta = OwnedDelta {
        struct_commands: vec![create_section(
            SECTION,
            PENDING_FIRST_ID,
            "pending".to_owned(),
        )],
        ..Default::default()
    };
    (
        stored_shape_of(base_delta(Base::Children(D2_CHILDREN))),
        delta,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// b.fsz §C: the growth fit's direct per-record measure (report only)
// ─────────────────────────────────────────────────────────────────────────────
//
// `repair_growth` (`REPAIR_GROWTH_CENTI`) is fitted from its direct measure,
// per replayed record, not from the segment netting (which nets it to 0;
// `docs-internal/validation/repair-charge-t7-b-fsz-20261010/report.md` § 2).
// Each row's entry is built in process as the cache stores it, and its
// records — each one stream entry's OMPF, as the client sent it — are
// replayed onto it in order, one window over the whole replay, on each
// holder: the entry with its plan indexes (the cached route's repair) and
// its index-less copy (the sampled audit's). Each record is decoded before
// its own measuring, so its parse (priced by `k_parse`, not by
// `repair_growth`) is held at the start of the record's measure and is left
// out of it. For record `k`:
//
// - `G_k`, the heap's peak while the record applies and normalizes, above
//   the live bytes when the replay began, less the decoded record: what the
//   earlier records left live, plus the record's own peak above the live
//   bytes once it is decoded;
// - the part of `G_k` RP2 holds by itself: with plan indexes, the entry's
//   weight now less its weight when the replay began (RP2 holds the weight
//   now, `CacheEntry::weights_now`); on the copy, the growth RP2 priced for
//   the records before (RP2 holds `W0 + G`, never re-weighing it,
//   `CopyWeight`);
// - the record's own priced terms, as RP2 holds them: its growth and scratch
//   1:1 (`estimate_replay_record_growth`; on the copy, priced from its bucket
//   bound, `estimate_replay_record_growth_bounded`, raised after each record
//   as the repair raises it) and its rewrite at its basis (its
//   price at the cached rewrite prices ÷ 1.25, so a rewrite priced above
//   what it holds cannot hide growth);
//
// and `f_k` is what is left, if anything, per byte of the delta parts the
// per-byte term prices at that RP2: the record's own on the indexed entry,
// and on the copy those of every record replayed so far (its RP2 carries
// `D<k + δ_k`). The fit is ⌈the largest `f_k` over every record, row, holder,
// sink and run × 125⌉ hundredths, never below the measure; the record's
// `scripts/fit_repair.py` takes it over the runs its `fits-manifest.json`
// requires and fails closed on a missing one.

/// One row of the growth fit (the section above): the entry as the cache
/// stores it, the records replayed onto it in order — each one stream
/// entry's OMPF, as the client sent it — and the sinks it replays through
/// (`false` discarding, `true` the list-delete sink).
struct GrowthRow {
    state: CompactState,
    records: Vec<Vec<u8>>,
    sinks: &'static [bool],
}

/// One record's direct measure ([`growth_fit_records`]).
#[derive(Debug)]
struct GrowthRecord {
    delta: u64,
    denom: u64,
    g: i64,
    held_net: i64,
    growth: u64,
    scratch: u64,
    rewrite_priced: u64,
    residual: f64,
    f: f64,
}

/// A handler row's growth-fit form: `base` as [`cached_shape_of`] builds
/// the cached entry its measured call hits, and `pending`'s records as the
/// client sent them, external values included.
fn handler_growth_row(base: Base, pending: Vec<PendingRecord>) -> GrowthRow {
    GrowthRow {
        state: cached_shape_of(base),
        records: pending
            .iter()
            .map(|record| {
                let external: Vec<&[u8]> = record.external.iter().map(Vec::as_slice).collect();
                common::owned_delta::delta_to_ompf_bytes(&record.delta, &external)
            })
            .collect(),
        sinks: &[false],
    }
}

/// An in-process row's growth-fit form: one record, `delta`, onto `state`.
fn one_record_growth_row(
    (state, delta): (CompactState, OwnedDelta),
    sinks: &'static [bool],
) -> GrowthRow {
    GrowthRow {
        state,
        records: vec![common::owned_delta::delta_to_ompf_bytes(&delta, &[])],
        sinks,
    }
}

/// A growth-fit row: its name and how to build it.
type GrowthRowBuilder = (&'static str, Box<dyn Fn() -> GrowthRow>);

/// Every row of the growth fit, by name: each gating T7 handler row that
/// replays records (its records and the entry its measured call hits; its
/// twin and cold variants replay the same records onto the same entry), the
/// audit envelope rows, and the RP2 scratch rows. The record's
/// `fits-manifest.json` names the same rows; the fit fails on one missing.
fn growth_fit_rows() -> Vec<GrowthRowBuilder> {
    let handler = |base: Base, pending: fn() -> Vec<PendingRecord>| -> Box<dyn Fn() -> GrowthRow> {
        Box::new(move || handler_growth_row(base, pending()))
    };
    let audit = |shape: AuditShape, weight: u64| -> Box<dyn Fn() -> GrowthRow> {
        Box::new(move || one_record_growth_row(audit_row(shape, weight), &[false]))
    };
    let scratch = |row: fn() -> (CompactState, OwnedDelta),
                   sinks: &'static [bool]|
     -> Box<dyn Fn() -> GrowthRow> {
        Box::new(move || one_record_growth_row(row(), sinks))
    };
    vec![
        (
            "fsz-repair-256-small-records",
            handler(Base::Children(0), || pending_small_records(256)),
        ),
        (
            "fsz-repair-256-field-writes",
            handler(Base::Children(0), || pending_field_writes(256)),
        ),
        (
            "fsz-repair-2mib-external-record",
            handler(Base::Children(0), pending_external_record),
        ),
        (
            "fsz-repair-1mib-sections-record",
            handler(Base::Children(0), pending_sections_record),
        ),
        (
            "fsz-repair-3mib-prims-record",
            handler(Base::Children(0), pending_prims_record),
        ),
        (
            "fsz-repair-256-1kib-records",
            handler(Base::Children(0), || pending_values(256, 1024)),
        ),
        (
            "fsz-repair-250kib-record",
            handler(Base::Children(0), || pending_values(1, 250 * 1024)),
        ),
        (
            "fsz-repair-index-resize",
            handler(Base::Children(4_096), || pending_small_records(256)),
        ),
        (
            "fsz-d1-full-sections-table",
            handler(Base::Children(D1_CHILDREN), || {
                vec![pending_of(new_root_section(PENDING_FIRST_ID))]
            }),
        ),
        (
            "fsz-d2-long-list",
            handler(Base::Children(D2_CHILDREN), || {
                vec![pending_of(OwnedDelta {
                    struct_commands: vec![create_section(
                        SECTION,
                        PENDING_FIRST_ID,
                        "pending".to_owned(),
                    )],
                    ..Default::default()
                })]
            }),
        ),
        (
            "fsz-audit-asset-8mib",
            audit(AuditShape::PlaceAsset, 8 * MIB as u64),
        ),
        (
            "fsz-audit-asset-cap",
            audit(AuditShape::PlaceAsset, AUDIT_ENTRY_CAP * 95 / 100),
        ),
        (
            "fsz-audit-field-table-8mib",
            audit(AuditShape::FieldIntoFullTable, 8 * MIB as u64),
        ),
        (
            "fsz-audit-field-table-cap",
            audit(AuditShape::FieldIntoFullTable, AUDIT_ENTRY_CAP * 95 / 100),
        ),
        (
            "fsz-audit-section-table-8mib",
            audit(AuditShape::SectionIntoFullTable, 8 * MIB as u64),
        ),
        (
            "fsz-audit-section-table-cap",
            audit(AuditShape::SectionIntoFullTable, AUDIT_ENTRY_CAP * 95 / 100),
        ),
        (
            "fsz-scratch-cascade",
            scratch(scratch_cascade_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-list-deleted-writes",
            scratch(scratch_list_deleted_writes_row, &[true]),
        ),
        (
            "fsz-scratch-dev-asset-delete",
            scratch(scratch_dev_asset_delete_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-dev-group-delete",
            scratch(scratch_dev_group_delete_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-dev-same-name-creation",
            scratch(scratch_dev_same_name_creation_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-dev-move-to-missing-parent",
            scratch(scratch_dev_move_to_missing_parent_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-dev-reparent-then-delete",
            scratch(scratch_dev_reparent_then_delete_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-dev-adopting-birth-deleted",
            scratch(scratch_dev_adopting_birth_deleted_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-20k-child-section-delete",
            scratch(scratch_20k_child_section_delete_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-chain-reset",
            scratch(scratch_chain_reset_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-dev-reset",
            scratch(scratch_dev_reset_row, &BOTH_SINKS),
        ),
        (
            "fsz-scratch-d2-long-list",
            scratch(scratch_d2_long_list_row, &[false]),
        ),
    ]
}

/// The cached route's rewrite price of `rewrite` (`rewrite_charge` at
/// `RewritePrices::CACHED`, crate-private), read back through the charge.
fn cached_rewrite_price(rewrite: RewriteCount) -> u64 {
    let price = |rewrite| {
        object_state_charge_for_test(&ObjectStateNeed::cached_repair_record(
            0,
            0,
            rewrite,
            RepairTransient::default(),
        ))
    };
    price(rewrite) - price(RewriteCount::default())
}

/// Replay `row`'s records in order onto a clone of its entry, with its plan
/// indexes when `indexed`, through the list-delete sink when
/// `list_deletes`, and measure each record directly (the section above).
fn growth_fit_records(row: &GrowthRow, indexed: bool, list_deletes: bool) -> Vec<GrowthRecord> {
    use live_edit::domain::delta::MutationSink;
    use live_edit::domain::plan_index::{normalize_touched, PlanIndexes, PlanTouches};
    use live_edit::domain::replay::{
        apply_decoded_record_indexed, apply_decoded_record_into, decode_stream_record,
        normalize_children, normalize_state,
    };
    use live_edit::storage::structural_cache::{
        estimate_replay_record_growth, estimate_replay_record_growth_bounded, estimate_state_bytes,
        indexed_state_bytes, CopyBucketBound, HeldCapacities,
    };
    let mut state = row.state.clone();
    let mut indexes = indexed.then(|| PlanIndexes::build(&state));
    // The entry's weight as RP2 reads it: with plan indexes, its weight now
    // (`CacheEntry::weights_now`: each table at its held capacity); the
    // copy's is read only at the start (`W0`).
    let weight = |state: &CompactState, indexes: Option<&PlanIndexes>| match indexes {
        Some(indexes) => indexed_state_bytes(state, indexes) + indexes.weight_bytes(),
        None => estimate_state_bytes(state),
    };
    // The copy's bucket bound: the audit starts the bound at the committed
    // entry's held capacities; these rows' states are built by inserts
    // alone, so held equals live.
    let mut bound = (!indexed).then(|| CopyBucketBound::new(&state, HeldCapacities::live(&state)));
    let deltas: Vec<u64> = row
        .records
        .iter()
        .map(|ompf| {
            OmpfContainer::parse(ompf)
                .expect("the record's OMPF parses")
                .delta_bytes
                .len() as u64
        })
        .collect();
    let mut out = Vec::with_capacity(row.records.len());
    let w0 = weight(&state, indexes.as_ref());
    let (mut grown, mut replayed) = (0u64, 0u64);
    let live0 = LIVE.load(Ordering::SeqCst) as i64;
    for (ompf, &delta) in row.records.iter().zip(&deltas) {
        let live_before = LIVE.load(Ordering::SeqCst) as i64;
        let seq = state.sequence_num + 1;
        let record =
            decode_stream_record(state.sequence_num, Some(ompf), seq).expect("the record decodes");
        // RP2's estimate: on the indexed entry at its held capacities, on the
        // copy from its bucket bound (`unindexed_record_growth`), with what
        // the bound's raise reads once the record applies.
        let (estimate, priced) = match bound.as_ref() {
            Some(bound) => {
                let (estimate, priced) = estimate_replay_record_growth_bounded(
                    &state,
                    bound,
                    record.delta(),
                    list_deletes,
                );
                (estimate, Some(priced))
            }
            None => (
                estimate_replay_record_growth(
                    &state,
                    indexes.as_ref(),
                    record.delta(),
                    list_deletes,
                ),
                None,
            ),
        };
        let held_net = if indexed {
            weight(&state, indexes.as_ref()) as i64 - w0 as i64
        } else {
            grown as i64
        };
        let mut sink = if list_deletes {
            MutationSink::list_deletes()
        } else {
            MutationSink::Discard
        };
        let ((), peak) = peak_of(|| match indexes.as_mut() {
            Some(indexes) => {
                let mut touches = PlanTouches::default();
                apply_decoded_record_indexed(&mut state, record, &mut sink, indexes, &mut touches)
                    .expect("the record applies");
                normalize_touched(&mut state, indexes, &touches.lists);
            }
            None => {
                apply_decoded_record_into(&mut state, record, &mut sink)
                    .expect("the record applies");
                normalize_children(&mut state);
                normalize_state(&mut state);
            }
        });
        drop(sink);
        // The copy's bound raised from what the record priced, read once the
        // record and its normalization have run (`CopyWeight::applied`).
        if let (Some(bound), Some(priced)) = (bound.as_mut(), priced.as_ref()) {
            bound.applied(priced, &state);
        }
        let g = live_before - live0 + peak as i64;
        let rewrite_priced = cached_rewrite_price(estimate.rewrite);
        let denom = if indexed { delta } else { replayed + delta };
        let residual = g as f64
            - held_net as f64
            - estimate.growth as f64
            - estimate.scratch as f64
            - rewrite_priced as f64 / HEADROOM;
        out.push(GrowthRecord {
            delta,
            denom,
            g,
            held_net,
            growth: estimate.growth,
            scratch: estimate.scratch,
            rewrite_priced,
            residual,
            f: residual.max(0.0) / denom.max(1) as f64,
        });
        grown += estimate.growth;
        replayed += delta;
    }
    out
}

/// b.fsz §C: the growth fit's direct per-record measure (the section above)
/// over every row of [`growth_fit_rows`], on both holders and each of the
/// row's sinks. Prints one `GROWTH-RECORD` line per record, one `GROWTH-ROW`
/// line per row, holder and sink with its largest `f_k`, and a closing
/// `GROWTH-RUN-END` line, the marker that the run is whole. Report only:
/// asserts nothing of the figures. `FSZ_GROWTH_ONLY` keeps the rows whose
/// name contains it (a partial run, which the fit refuses as a member).
#[test]
#[ignore = "manual: report only — b.fsz §C growth fit's direct per-record measure"]
fn repair_growth_records_are_reported_fsz() {
    let _serial = MEASURE.blocking_lock();
    let only = std::env::var("FSZ_GROWTH_ONLY").unwrap_or_default();
    let wired = repair_growth_centi();
    let mut arms = 0;
    let mut rows = 0;
    for (name, build) in growth_fit_rows() {
        if !name.contains(only.as_str()) {
            continue;
        }
        let row = build();
        rows += 1;
        for indexed in [true, false] {
            for &list_deletes in row.sinks {
                arms += 1;
                let regime = if indexed { "indexed" } else { "copy" };
                let sink = if list_deletes {
                    "list-deletes"
                } else {
                    "discard"
                };
                let records = growth_fit_records(&row, indexed, list_deletes);
                let n = records.len();
                let mut worst = (0.0_f64, 0usize);
                for (k, r) in records.iter().enumerate() {
                    eprintln!(
                        "GROWTH-RECORD row={name} regime={regime} sink={sink} k={k} of={n} \
                         delta={} denom={} g={} held_net={} growth={} scratch={} \
                         rewrite_priced={} residual={:.1} f={:.6} covered_at_wired={}",
                        r.delta,
                        r.denom,
                        r.g,
                        r.held_net,
                        r.growth,
                        r.scratch,
                        r.rewrite_priced,
                        r.residual,
                        r.f,
                        r.residual <= (r.denom * wired) as f64 / 100.0
                    );
                    if r.f > worst.0 {
                        worst = (r.f, k);
                    }
                }
                eprintln!(
                    "GROWTH-ROW row={name} regime={regime} sink={sink} records={n} max_f={:.6} \
                     at_k={}",
                    worst.0, worst.1
                );
            }
        }
    }
    eprintln!(
        "GROWTH-RUN-END rows={rows} arms={arms} only={only:?} repair_growth_centi_wired={wired}"
    );
}

/// The sampled audit's compare (`first_difference`) holds no copy of either
/// state: beside the two states it holds only the smallest differing key it
/// has found and the detail. On a state of 20,000 sections in 2,001 lists,
/// 60,000 field rows and a time sample per section — each part differing in
/// every row in turn (every section renamed, every list reversed, every
/// field's and every sample's external hash flipped), and a control in which
/// only the dangling parents differ — the compare names each part's
/// smallest key with a heap peak within 4 KiB. The states are built outside
/// the window. A compare that collected and sorted a part's keys would hold
/// one reference per key (about 320 KB for the sections, 32 KB for the
/// lists), and one that built row maps a copy of the rows (MBs). The states
/// hold no tables a group width changes the compare's allocations by: it
/// allocates nothing per row whatever the tables' layout.
#[test]
fn first_difference_holds_no_copy_of_either_state_fsz() {
    use live_edit::domain::types::{FieldValue, TimeSample};
    use live_edit::storage::structural_cache_audit::first_difference_for_test;
    use ordered_float::OrderedFloat;
    use std::collections::HashSet;
    const PEAK_LIMIT: u64 = 4 * 1024;
    let _serial = MEASURE.blocking_lock();
    let external = |tag: u8| FieldValue::External {
        hash: [tag; 32],
        size: 64,
        ext_value_index: 0,
    };
    let mut base = audit_base(20_000, 60_000, 10);
    for row in base.fields.values_mut() {
        row.value = external(1);
    }
    let ids: Vec<u64> = base.sections.keys().copied().collect();
    for &id in &ids {
        base.time_samples.insert(
            (id, OrderedFloat(0.5)),
            TimeSample {
                section_id: id,
                time: OrderedFloat(0.5),
                value: external(1),
                version: 1,
            },
        );
    }
    assert_eq!(
        (base.sections.len(), base.children.len(), base.fields.len()),
        (20_000, 2_001, 60_000),
        "fixture self-check: the state's parts"
    );
    let cached = live_edit::domain::types::compact(&base);
    let none = HashSet::new();
    let changed = |change: &dyn Fn(&mut ObjectState)| {
        let mut fresh = base.clone();
        change(&mut fresh);
        live_edit::domain::types::compact(&fresh)
    };
    let cases: [(
        &str,
        live_edit::domain::types::CompactState,
        HashSet<u64>,
        &str,
    ); 5] = [
        (
            "every section renamed",
            changed(&|s| s.sections.values_mut().for_each(|row| row.name.push('~'))),
            HashSet::new(),
            "section 1",
        ),
        (
            "every list reversed",
            changed(&|s| s.children.values_mut().for_each(|list| list.reverse())),
            HashSet::new(),
            "children list of 0",
        ),
        (
            "every field's hash flipped",
            changed(&|s| {
                s.fields
                    .values_mut()
                    .for_each(|row| row.value = external(2))
            }),
            HashSet::new(),
            "field \"f0\" of section 1",
        ),
        (
            "every sample's hash flipped",
            changed(&|s| {
                s.time_samples
                    .values_mut()
                    .for_each(|row| row.value = external(2))
            }),
            HashSet::new(),
            "time sample at 0.5 of section 1",
        ),
        (
            "only the dangling parents (control)",
            changed(&|_| {}),
            HashSet::from([999_999]),
            "dangling parent 999999",
        ),
    ];
    for (case, fresh, fresh_dangling, named) in cases {
        let (difference, peak) =
            peak_of(|| first_difference_for_test(&cached, &none, &fresh, &fresh_dangling));
        eprintln!("FIRST-DIFFERENCE {case}: {difference:?}, peak {peak} B");
        assert_eq!(difference.as_deref(), Some(named), "{case}");
        assert!(
            peak <= PEAK_LIMIT,
            "{case}: the compare's heap peak, {peak} B, is past {PEAK_LIMIT} B: it held a copy \
             of a part"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// b.fsz MR-D4 (D4-T2) — a retirement's heap: the reference walk, the release
// and the paged teardown, against the retirement allowance's model
// ═════════════════════════════════════════════════════════════════════════════
//
// A retirement reserves its charge on the retirement allowance
// (`storage::retirement::retirement_charge_bytes`): its page at
// `RETIREMENT_ROW_BYTES` per member — at most the full page bound `W` =
// `P` × `RETIREMENT_ROW_BYTES` its walk and teardown hold — with a row for
// each of the release's size reads in flight (at most
// `RELEASE_SIZE_READS_IN_FLIGHT`, read once the walk's pages are dropped),
// and `h` = `RETIREMENT_REFERENCE_CHARGE_BYTES` per distinct reference its
// release carries. Each row below holds one term to its constant, the heap
// counted process-wide (the walk's pass runs on a `watch-read` thread), after
// a twin run that warms the connections' read buffers. The fixtures are
// written straight into the store.

/// `P` for the rows: the smallest pod's page (`R_ref` = 5 MiB).
const RETIREMENT_PAGE_LEN: u64 = live_edit::storage::retirement::RETIREMENT_MIN_PAGE_LEN;

/// Write `rows` field rows of `object_id` named at `MAX_NAME_BYTES`, each
/// holding an inline `value` of `value_bytes` (never transferred by the
/// walk), and index them in `obj:flds`; plus a header.
async fn seed_long_named_inline_rows(
    tv: &TestValkey,
    object_id: u64,
    rows: u64,
    value_bytes: usize,
) {
    use fred::interfaces::{HashesInterface, SortedSetsInterface};
    let keys = tv.key_schema();
    let oid = object_id.to_string();
    let pool = &tv.client().pool;
    let _: () = pool
        .hset(keys.obj_header(&oid), [("seq", "1")])
        .await
        .expect("HSET header");
    let value = "A".repeat(value_bytes);
    for i in 0..rows {
        // A distinct name of exactly `MAX_NAME_BYTES`: its member's hex form
        // is the longest row key a commit can write.
        let mut name = format!("{i:08}");
        name.push_str(&"n".repeat(live_edit::domain::limits::MAX_NAME_BYTES - name.len()));
        let _: () = pool
            .hset(
                keys.obj_field(&oid, "7", &name),
                [
                    ("value", value.as_str()),
                    ("set_order", "1"),
                    ("version", "1"),
                ],
            )
            .await
            .expect("HSET row");
        let _: i64 = pool
            .zadd(
                keys.obj_fields_index(&oid),
                None,
                None,
                false,
                false,
                (
                    i as f64,
                    live_edit::storage::keys::KeySchema::field_member("7", &name),
                ),
            )
            .await
            .expect("ZADD member");
    }
}

/// D4-T2, `RETIREMENT_ROW_BYTES`: the reference walk of 3 pages × `P` field
/// rows, each named at `MAX_NAME_BYTES` and holding a 64 KiB inline value,
/// stays within `W` = `P` × `RETIREMENT_ROW_BYTES` with no reference to
/// carry. The values add up to 12 MiB, which the release's read before the
/// walk (the ext-GC snapshot read, removed by b.fsz MR-D4) transferred and
/// decoded whole, then cloned, so this bound fails on that read by an order
/// of magnitude; the walk reads each row's shape by `HSTRLEN` and `HMGET`
/// and never transfers a value. The peak must also reach a page's row keys
/// (`P` × 8 KiB): the page's rows were in flight together, so the bound is
/// not met by an idle walk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retirement_walk_of_long_named_inline_rows_stays_within_its_page_bound_fsz() {
    use live_edit::storage::retirement::{
        walk_object_references, RetirementWalk, RETIREMENT_ROW_BYTES,
    };
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let rows = 3 * RETIREMENT_PAGE_LEN;
    seed_long_named_inline_rows(&tv, 610_001, rows, 64 * 1024).await;
    let walk = RetirementWalk::unreserved().with_page_len(RETIREMENT_PAGE_LEN);
    let run = || walk_object_references(tv.client(), tv.key_schema(), "610001", &walk);
    run().await.expect("the twin walk must succeed");

    let base = begin_window();
    let walked = run().await.expect("the walk must succeed");
    let peak = end_window(base);

    let bound = RETIREMENT_PAGE_LEN * RETIREMENT_ROW_BYTES;
    let floor = RETIREMENT_PAGE_LEN * 2 * live_edit::domain::limits::MAX_NAME_BYTES as u64;
    eprintln!(
        "RETIREMENT-ROW rows={rows} page={RETIREMENT_PAGE_LEN} peak={peak} bound={bound} \
         per_row={}",
        peak / RETIREMENT_PAGE_LEN
    );
    assert_eq!(walked.rows, rows, "fixture: every row walked");
    assert!(
        walked.hashes.is_empty(),
        "fixture: inline rows name no hash"
    );
    assert!(
        peak <= bound,
        "the walk's heap peak, {peak} B, is past W = P × RETIREMENT_ROW_BYTES = {bound} B"
    );
    assert!(
        peak >= floor,
        "the walk's heap peak, {peak} B, never held a page of row keys ({floor} B): the page's \
         rows were not read together, and the bound above is not exercised"
    );
}

/// D4-T2, the composed route: the whole release of an object of 3 × `P`
/// long-named field rows each holding a 64 KiB inline value and no reference
/// (`schedule_old_object_ext_gc`: the walk, the begin, nothing to enqueue),
/// at the reference pod's page, stays within its `W`. The values add up to
/// 24 MiB, which the release's read before b.fsz MR-D4 (the ext-GC snapshot
/// read of the whole object) transferred, decoded and cloned: on the build
/// before it this row's peak measured 42,577,956 B against that build's `W`
/// of 5,242,880 B (debug; the reference page was then 128 rows).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retirement_release_of_long_named_inline_rows_stays_within_its_page_bound_fsz() {
    use live_edit::storage::external_values::MockExternalValueStore;
    use live_edit::storage::layer::schedule_old_object_ext_gc_for_test;
    use live_edit::storage::retirement::RetirementPaging;
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let paging = RetirementPaging::reference();
    let rows = 3 * paging.page_len();
    seed_long_named_inline_rows(&tv, 615_001, rows, 64 * 1024).await;
    seed_long_named_inline_rows(&tv, 615_002, rows, 64 * 1024).await;
    let ext = MockExternalValueStore::new();
    schedule_old_object_ext_gc_for_test(tv.client(), tv.key_schema(), "615001", &ext, None)
        .await
        .expect("the twin release must succeed");

    let base = begin_window();
    schedule_old_object_ext_gc_for_test(tv.client(), tv.key_schema(), "615002", &ext, None)
        .await
        .expect("the release must succeed");
    let peak = end_window(base);

    let bound = paging.page_bytes();
    eprintln!(
        "RETIREMENT-RELEASE-ROWS rows={rows} page={} peak={peak} bound={bound}",
        paging.page_len()
    );
    assert_eq!(
        ext.pending_count(),
        0,
        "fixture: inline rows name no reference"
    );
    assert!(
        peak <= bound,
        "the release's heap peak, {peak} B, is past the reference pod's W = {bound} B: the \
         release read more of the object than one page of rows"
    );
}

/// D4-T2, `RETIREMENT_REFERENCE_CHARGE_BYTES` (`h`): the whole release of an
/// object whose `obj:ext` records `N` distinct references and that has no row
/// — the walk, the begin that records them on the carrier, the attempt-mode
/// enqueue, the clear — measured at `N` = 10,000 and 20,000, through the
/// in-memory external-value store (whose enqueue takes the members as they
/// are) and through the real store (whose enqueue first reads each value's
/// size for the release ledger, at most `RELEASE_SIZE_READS_IN_FLIGHT` at a
/// time, and sends the members as script arguments). Per store, the slope
/// `Δpeak ÷ ΔN` is what one more reference holds and must be at most `h`; the
/// constant `peak − slope × N` is what the release holds whatever its size —
/// the walk's page and the size reads in flight — and must be within the
/// charge's row term for this object: with no row and more references than
/// reads in flight, a row per read, `RELEASE_SIZE_READS_IN_FLIGHT` ×
/// `RETIREMENT_ROW_BYTES`; and each peak is within what the route reserves
/// for it, `retirement_charge_bytes(1, N)` (that row term + `h × N`),
/// whatever the reference pod's full page `W`. One read per
/// reference all at once, as before the size reads were capped, holds a
/// per-reference term several times `h` in the real store's slope. Prints
/// `RETIREMENT-H profile={debug|release}` with each store's slope and
/// constant, and `W` beside them.
///
/// The counter is process-wide, so a window opens only once every pool's
/// router has dropped the commands before it (`quiesce`): fred answers a
/// caller before it drops the command and its arguments, and the twin
/// release's last large command, freed inside the next window, once took
/// ~2.9 MB off `peak_10000` and its slope past `h` (1,050 B).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retirement_release_of_many_references_stays_within_h_per_reference_fsz() {
    use fred::interfaces::HashesInterface;
    use live_edit::storage::external_values::{ExternalValueStore, MockExternalValueStore};
    use live_edit::storage::layer::schedule_old_object_ext_gc_for_test;
    use live_edit::storage::retirement::{
        retirement_charge_bytes, RetirementPaging, RELEASE_SIZE_READS_IN_FLIGHT,
        RETIREMENT_REFERENCE_CHARGE_BYTES, RETIREMENT_ROW_BYTES,
    };
    const SMALL: u64 = 10_000;
    const LARGE: u64 = 20_000;
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    /// `oid`'s header and `refs` distinct references, each hash naming its
    /// object so no two objects share one.
    async fn seed(tv: &TestValkey, oid: &str, refs: u64) {
        let keys = tv.key_schema();
        let _: () = tv
            .client()
            .pool
            .hset(keys.obj_header(oid), [("seq", "1")])
            .await
            .expect("HSET header");
        let fields: Vec<(String, String)> = (0..refs)
            .map(|i| (format!("{oid:0>32}{i:032x}"), "1".to_owned()))
            .collect();
        let _: () = tv
            .client()
            .pool
            .hset(keys.obj_ext_index(oid), fields)
            .await
            .expect("HSET obj:ext");
    }
    async fn release(tv: &TestValkey, ext: &dyn ExternalValueStore, oid: &str) {
        schedule_old_object_ext_gc_for_test(tv.client(), tv.key_schema(), oid, ext, None)
            .await
            .expect("the release must succeed");
    }
    /// One round trip on every connection of each pool a release uses (data,
    /// snapshot, maintenance). fred answers a command's caller before it drops
    /// the command, arguments included (`respond_to_caller`), so a release's
    /// last large command — a begin or an enqueue carrying every member, a
    /// few MB at `LARGE` — can be freed by its router task after the caller
    /// has opened the next window, and that window's peak is then understated
    /// by the arguments' size. A router answers commands in order, so once
    /// each one has answered a `PING` issued after the release returned, no
    /// earlier command is left to drop.
    async fn quiesce(tv: &TestValkey) {
        use fred::interfaces::ClientLike;
        let client = tv.client();
        for pool in [
            &client.pool,
            &client.snapshot_pool,
            &client.maintenance_pool,
        ] {
            for member in pool.clients() {
                let _: () = member.ping(None).await.expect("PING: one router pass");
            }
        }
    }
    /// The heap peak of the release of `oid`, measured in a window opened
    /// after [`quiesce`], and how far the live heap fell below the window's
    /// baseline in it ([`window_dip`]): above 0 only when memory allocated
    /// before the window was freed in it, which may have understated the
    /// peak.
    async fn measured(tv: &TestValkey, ext: &dyn ExternalValueStore, oid: &str) -> (u64, u64) {
        quiesce(tv).await;
        let base = begin_window();
        release(tv, ext, oid).await;
        let peak = end_window(base);
        (peak, window_dip(base))
    }
    /// How many times [`peaks`] measures before it gives up on windows that
    /// dipped.
    const ATTEMPTS: u64 = 3;
    /// The dip a window may show and still count: the quiesce's own `PING`s
    /// are dropped late the same way, a few hundred bytes each. 64 KiB is
    /// about 6.6 B per reference at `SMALL`, against `h` = 1,024 B.
    const DIP_TOLERANCE: u64 = 64 * 1024;
    /// Through `ext`, after a twin release at the larger count: the heap peak
    /// of the release of `SMALL`, then of `LARGE`, references, each from a
    /// window that freed nothing allocated before it beyond
    /// [`DIP_TOLERANCE`]. A window that did is measured again on fresh
    /// objects (oids from `first`, three per attempt), at most [`ATTEMPTS`]
    /// times, and printed as `RETIREMENT-H-DIPPED`; if every attempt dipped,
    /// the test fails naming the disturbance rather than a slope. The dip
    /// sees only a late free that takes the heap below the baseline, so the
    /// quiesce, not the check, is what keeps the windows clean; the check
    /// catches a new source of late frees.
    async fn peaks(tv: &TestValkey, ext: &dyn ExternalValueStore, first: u64) -> (u64, u64) {
        for attempt in 0..ATTEMPTS {
            let oids = [0, 1, 2].map(|i| (first + 3 * attempt + i).to_string());
            seed(tv, &oids[0], LARGE).await;
            seed(tv, &oids[1], SMALL).await;
            seed(tv, &oids[2], LARGE).await;
            release(tv, ext, &oids[0]).await;
            let (small, dip_small) = measured(tv, ext, &oids[1]).await;
            let (large, dip_large) = measured(tv, ext, &oids[2]).await;
            if dip_small <= DIP_TOLERANCE && dip_large <= DIP_TOLERANCE {
                return (small, large);
            }
            eprintln!(
                "RETIREMENT-H-DIPPED attempt={attempt} peak_{SMALL}={small} \
                 dip_{SMALL}={dip_small} peak_{LARGE}={large} dip_{LARGE}={dip_large}"
            );
        }
        panic!(
            "every one of {ATTEMPTS} measurements freed memory allocated before its window, so \
             none measured a release's peak; something besides the routers quiesced here frees \
             late (RETIREMENT-H-DIPPED lines above)"
        );
    }
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let w = RetirementPaging::reference().page_bytes();
    let h = RETIREMENT_REFERENCE_CHARGE_BYTES;
    let rows = RELEASE_SIZE_READS_IN_FLIGHT as u64 * RETIREMENT_ROW_BYTES;

    let mock = MockExternalValueStore::new();
    let in_memory = peaks(&tv, &mock, 620_001).await;
    let real = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone());
    let valkey = peaks(&tv, &real, 620_101).await;

    for (store, (small, large)) in [("in-memory", in_memory), ("Valkey", valkey)] {
        let slope = large.saturating_sub(small) / (LARGE - SMALL);
        let constant = large.saturating_sub(slope * LARGE);
        eprintln!(
            "RETIREMENT-H profile={profile} store={store} peak_{SMALL}={small} \
             peak_{LARGE}={large} slope={slope} constant={constant} h={h} w={w}"
        );
        assert!(
            large > small && small >= SMALL * 64,
            "{store}: fixture: the release carried every reference, and more of them held more \
             ({small} B for {SMALL}, {large} B for {LARGE})"
        );
        assert!(
            slope <= h,
            "{store}: one more reference holds {slope} B, past h = {h} B"
        );
        assert!(
            constant <= rows,
            "{store}: the release holds {constant} B whatever its size, past a row per size \
             read in flight = {rows} B"
        );
        for (refs, peak) in [(SMALL, small), (LARGE, large)] {
            let charge = retirement_charge_bytes(1, refs);
            assert!(
                peak <= charge,
                "{store}: the release of {refs} references peaked at {peak} B, past its \
                 charge retirement_charge_bytes(1, {refs}) = {charge} B"
            );
        }
    }
}

/// D4-T2, `RETIREMENT_SET_MEMBER_BYTES`: the paged teardown of an object
/// whose sections index is a compact set of 512 integer ids (an intset,
/// which an `SSCAN` step answers whole whatever its `COUNT`) holds at most
/// 512 × `RETIREMENT_SET_MEMBER_BYTES`: the step's members, the two keys each
/// names and the `UNLINK` copies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retirement_teardown_of_a_compact_set_step_stays_within_its_member_charge_fsz() {
    use fred::interfaces::{HashesInterface, SetsInterface};
    use live_edit::storage::retirement::{RetirementPaging, Teardown, RETIREMENT_SET_MEMBER_BYTES};
    const MEMBERS: u64 = 512;
    let _serial = measure_lock().await;
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let seed = |oid: &'static str| {
        let tv = &tv;
        async move {
            let keys = tv.key_schema();
            let pool = &tv.client().pool;
            let _: () = pool
                .hset(keys.obj_header(oid), [("seq", "1")])
                .await
                .expect("HSET header");
            let ids: Vec<String> = (1..=MEMBERS).map(|i| i.to_string()).collect();
            let _: () = pool
                .sadd(keys.obj_sections_index(oid), ids.clone())
                .await
                .expect("SADD sections");
            for id in &ids {
                let _: () = pool
                    .hset(keys.obj_section(oid, id), [("name", "s")])
                    .await
                    .expect("HSET section");
            }
        }
    };
    seed("630001").await;
    seed("630002").await;
    let paging = RetirementPaging::for_allowance(5 << 20);
    Teardown::new(&tv.client().pool, keys, "630001", paging)
        .run()
        .await
        .expect("the twin teardown must succeed");

    let base = begin_window();
    Teardown::new(&tv.client().pool, keys, "630002", paging)
        .run()
        .await
        .expect("the teardown must succeed");
    let peak = end_window(base);

    let bound = MEMBERS * RETIREMENT_SET_MEMBER_BYTES;
    eprintln!(
        "RETIREMENT-SET-MEMBER members={MEMBERS} peak={peak} bound={bound} per_member={}",
        peak / MEMBERS
    );
    assert!(
        peak <= bound,
        "the teardown's heap peak, {peak} B, is past {MEMBERS} × RETIREMENT_SET_MEMBER_BYTES = \
         {bound} B"
    );
    let left: u64 =
        fred::interfaces::KeysInterface::exists(&tv.client().pool, keys.obj_section("630002", "1"))
            .await
            .expect("EXISTS");
    assert_eq!(left, 0, "fixture: the teardown removed the sections");
}
