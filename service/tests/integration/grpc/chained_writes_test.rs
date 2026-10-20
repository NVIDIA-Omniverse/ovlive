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

//! b.pav — chained `UpdateLayer` writes (`UpdateLayerRequest.chained_header`)
//! through the real handler: the chain gate (`storage::chain_gate`) that
//! holds a chained call after its content is read until the latest operation
//! recorded for its `(object_id, client_id)` is its predecessor (or its own),
//! its refusals, its metrics and its completion line, and the
//! `INVALID_UPDATE` choke point (scope (a)); and the commit-time guard — the
//! gate's `ready` is advisory, and every commit route, on every backend,
//! refuses an attempt whose record does not name the call's predecessor
//! (scope (b)).
//!
//! Every ordering test asserts what a client observes — the commit order
//! through each answer's `sequence_num`, the outcome, the status code and its
//! `ErrorInfo`.
//!
//! | Test | Group | What it pins |
//! |---|---|---|
//! | [`chained_writes_sent_in_order_to_two_replicas_commit_in_chain_order`] | 1 | one writer's chain alternating between two replicas commits in chain order, each `APPLIED` |
//! | [`chained_writes_sent_out_of_order_to_two_replicas_commit_in_chain_order`] | 1, 16, 22 | the chain sent last-first, its later writes parked on two replicas, commits in chain order, never overtaken; `ready` cells and a detection lag only for the calls that slept |
//! | [`depth_three_chain_keeps_waiting_while_the_record_moves_through_each_predecessor`] | 2 | the third write keeps waiting while the record is the first, then commits after the second |
//! | [`predecessor_refused_at_admission_leaves_its_successor_waiting_until_the_client_cancels_it`] | 3, 22 | a predecessor refused `UPLOAD_CAPACITY` at admission (the budget held by its waiting successor) does not end the successor; the client's cancel applies nothing and returns the reservation and both gauges |
//! | [`predecessor_lost_in_transport_is_cancelled_and_its_resend_releases_the_successor`] | 6 | a predecessor whose stream ended without END_STREAM is `CANCELLED` with nothing applied; its re-send commits and then its successor |
//! | [`passed_predecessor_with_no_wait_is_refused_at_once_naming_the_latest_write`] / [`passed_predecessor_without_no_wait_waits_until_its_deadline_cap`] / [`passed_predecessor_without_no_wait_waits_until_cancelled`] | 8 (a) | a predecessor the record has already passed: `no_wait` answers `ABORTED` at once, otherwise the call waits to its deadline cap or its cancel |
//! | [`no_wait_matrix_answers_from_the_record_alone`] | 9 | `no_wait` against a predecessor, own, other, absent and no-operation record |
//! | [`many_writers_chained_in_reverse_at_the_writer_limit_never_deadlock_on_the_fast_route`] / [`many_writers_chained_in_reverse_at_the_writer_limit_never_deadlock_on_the_structural_route`] | 10 | three writers × eight writes, sent last-first to two replicas whose per-object writer limit is 2, all commit in chain order, on the route each test names |
//! | [`chained_call_waiting_when_the_layer_is_deleted_is_answered_not_found_without_status_details`] / [`fault_injected::chained_call_waiting_when_create_layer_supersedes_its_object_is_answered_failed_precondition`] / [`chained_call_waiting_when_its_header_is_removed_and_the_layer_is_replaced_is_answered_failed_precondition`] | 11, Q2 | a replaced or deleted object ends a waiting call `FAILED_PRECONDITION` / `NOT_FOUND`, plain, outcome `replaced`; a removed header under a live mapping keeps it waiting |
//! | [`chained_call_to_a_stale_object_id_is_refused_failed_precondition_without_status_details`] | Q2 | the step-3 stale-object refusal of a chained call is plain |
//! | [`chained_header_read_as_a_pre_chaining_replica_reads_it_is_refused_before_any_chunk`] | 12 | the fail-closed old replica: the frame it decodes is refused `INVALID_ARGUMENT` with no details having read no chunk |
//! | [`every_invalid_argument_a_chained_call_meets_carries_invalid_update`] | 13 | every `INVALID_ARGUMENT` producer reachable by a chained call that has no reason of its own carries `INVALID_UPDATE`; the unchained twin stays plain with the same message |
//! | [`size_refusals_a_chained_call_meets_carry_their_own_reason_in_place_of_invalid_update`] | 13 (b.tbw) | a size refusal keeps its own reason on a chained call — `DELTA_TOO_LARGE`, or `CHILDREN_LIST_TOO_LONG` from the domain bound or the parser's cap — exactly as on the unchained twin; a content error on the same service still carries `INVALID_UPDATE` |
//! | [`fault_injected::planning_invalid_argument_after_a_counted_retry_is_plain_unavailable`] / [`fault_injected::planning_invalid_argument_on_the_reissue_after_a_two_attempt_first_call_is_plain_unavailable`] | 14 (a) | after a counted retry a chained `INVALID_ARGUMENT` becomes a plain `UNAVAILABLE`, outcome unknown |
//! | [`chained_call_waiting_at_shutdown_is_refused_server_shutting_down_at_stage_chain`] | 19, 22 | graceful shutdown ends a waiting call `UNAVAILABLE "server shutting down"`, INFO at stage `chain`, outcome `shutdown` |
//! | [`chained_call_whose_every_probe_fails_is_refused_chain_check_unavailable_near_its_deadline`] | 20, 22 | every probe failing: `UNAVAILABLE` `CHAIN_CHECK_UNAVAILABLE` + `RetryInfo` at the deadline cap, WARN, outcome `unavailable`; with `no_wait` too (it keeps probing) |
//! | [`stalled_store_ends_a_chained_call_chain_check_unavailable_before_its_deadline_end_to_end`] / [`stalled_store_after_one_read_ends_a_chained_call_aborted_before_its_deadline_end_to_end`] | stalled-store probe | each probe's own time limit ends the call before its `grpc-timeout`, never `DEADLINE_EXCEEDED` |
//! | [`waiting_chained_call_is_refused_aborted_before_its_grpc_timeout_end_to_end`] | 3 | without a cancel, `grpc-timeout` bounds the wait: `ABORTED` `last_record=operation` before the deadline |
//! | [`waiting_chained_call_is_refused_before_its_deadline_reaching_a_tonic_client_with_its_own_timer_over_a_delayed_link`] / [`stalled_store_ends_a_chained_call_chain_check_unavailable_reaching_a_tonic_client_with_its_own_timer_over_a_delayed_link`] | 28 | a tonic client whose own deadline timer starts before the server's clock, over a link adding 20 ms each way, reads the gate's `ABORTED` (or `CHAIN_CHECK_UNAVAILABLE`), never its own timeout; one completion, no abandoned line |
//! | [`late_arrival_with_no_more_than_its_margin_left_is_checked_once_and_does_not_wait`] | 27 | a call reaching the gate with no more than its margin left: one probe, no sleep — `timeout` naming the record, `ready` for a chain's first write, `timeout` or `unavailable` with no time left |
//! | [`late_arrival_whose_final_probe_reads_replaced_is_probed_again_after_a_same_object_re_resolve`] | C, review 1 | a late arrival re-resolved to the same object is probed again and answered from that probe: `ready` once its object is back and its predecessor committed, else `ABORTED` `timeout` from the replaced-again arm |
//! | [`same_object_re_resolve_finishing_after_d_minus_half_the_margin_is_refused_at_once_naming_the_freshest_record`] | C | the re-entered probe's limit has passed: given up at once, `ABORTED` `timeout` naming the freshest record |
//! | [`waiter_whose_same_object_re_resolve_finishes_before_d_minus_half_the_margin_keeps_its_floor`] | C | the re-entered final probe is given up by `D − M/2`, not halfway to `D` |
//! | [`chained_call_whose_header_is_gone_under_a_live_mapping_is_refused_with_no_wait_at_once_and_otherwise_at_its_cap`] | review 1 | replaced again after re-resolving to the same object: `no_wait` refused at once (`refused`), otherwise at the cap (`timeout`); never `FAILED_PRECONDITION` / `NOT_FOUND` |
//! | [`chained_call_whose_re_resolve_recovers_the_same_object_keeps_waiting_and_a_second_loss_is_not_found`] | review 1 | a re-resolve that runs the call's one recovery and finds the same object keeps the call waiting; a second missing mapping in that call is `NOT_FOUND`, no second recovery |
//! | [`chained_call_meeting_a_wrong_typed_record_key_leaves_the_gate_ready_and_is_refused_wrongtype_key`] / [`chained_call_meeting_a_wrong_typed_header_leaves_the_gate_ready_and_is_refused_by_the_commit`] | review 1 | a wrong-typed key ends the wait at once as `ready` (one probe, counted `error`); the commit refuses it `INTERNAL` (`WRONGTYPE_KEY` for the record key, the store's `WRONGTYPE` for the header) |
//! | [`gate_probe_reads_superseded_one_as_replaced_and_any_other_value_or_none_as_live`] | probe | the `HMGET … superseded` classification: `1` replaced, absent or another value live, no header replaced |
//! | [`fault_injected::chained_call_issues_no_start_record_read_and_its_unchained_twin_does`] | review 2, E8 | neither the handler nor the store issues a call-start record read for a chained call, whose writer guard is its predecessor |
//! | [`chain_metrics_seed_the_probe_and_gauge_cells_but_not_the_histograms`] | 22 | the probe counter and both gauges at zero from the first scrape; neither histogram before its first observation |
//! | [`chained_completion_lines_carry_chained_and_their_link`] | 24 | `chained = true` on every chained call's line (a validation refusal's included, with no link fields); PNC INFO at stage `chain`; the unchained line carries no `chained` |
//! | [`ping_reports_both_capability_bits_end_to_end`] | 23 | Ping through `live_service_server` answers `operation_ids` and `chained_writes` |
//! | [`chained_call_parked_at_the_gate_and_reset_after_end_stream_applies_nothing_end_to_end`] / [`chained_call_parked_at_the_gate_whose_connection_closes_applies_nothing_end_to_end`] | 4 | rider 1, a real HTTP/2 client against the full service: one abandoned line, one `CANCELLED` tick, `cancelled` observed, gauges and reserved bytes back, nothing applied, and the predecessor's later commit is not followed by the cancelled call's |
//! | `chained_call_whose_record_moves_after_its_gate_is_refused_predecessor_not_committed_by_its_commit_on_the_*_route` (three) | 16b, 17 | a record moved after the gate's `ready` (another operation, a write without one, the record deleted) is refused by the commit on the fast, structural and WATCH routes: `ABORTED` `PREDECESSOR_NOT_COMMITTED` with each `last_record` value, INFO at stage `storage`, one attempt, nothing applied, never overtaken |
//! | [`pending_predecessor_is_refused_aborted_ahead_of_a_wrong_typed_fence_or_release_record_in_every_cell`] | 25, 15 | in every backend × delta-class cell a pending predecessor is answered `ABORTED`, not the `WRONGTYPE_KEY` of a wrong-typed fence key or `obj:wrel` |
//! | [`chained_commit_answers_its_own_record_then_superseded_before_its_predecessor_in_every_cell`] | 15 | in every cell a re-send whose record is its own is `ALREADY_COMMITTED`, and a superseded object is `FAILED_PRECONDITION` ahead of a pending predecessor |
//! | [`chained_call_whose_record_moved_is_refused_predecessor_not_committed_before_planning_in_every_structural_cell`] | 15 | a pending predecessor is answered before planning: `ABORTED`, never the planner's `INVALID_UPDATE` |
//! | `chained_call_restored_mid_chain_is_refused_aborted_last_record_absent_and_hands_back_its_own_preflight_increment_on_the_*_route` (three) | 18 | a restore between the gate and the commit loses the predecessor's record: the re-issue, which pre-stored for itself, is refused `ABORTED` `last_record=absent` and, refused on its first attempt, hands back its own pre-flight increment (one `Resend` member); after GC the external value keeps the first call's leaked increment |
//! | `fault_injected::recovery_reissue_answered_from_a_client_resend_hands_back_its_own_preflight_increment_on_the_*_route` (three) | 18 (b) | the first call lands, loses its reply and ends `NOT_FOUND`; the restore releases its reference; the client's re-send of the operation commits on the restored object before the re-issue: the re-issue is answered `ALREADY_COMMITTED` (same bytes) or `ALREADY_EXISTS` (other bytes), hands back its own pre-flight increment (the re-issue's `Resend` member, none for the first call), and after GC the external value counts exactly its live references |
//! | `fault_injected::recovery_reissue_committing_after_a_restore_counts_every_live_reference_on_the_*_route` (three) | b.5wb A | the first call lands, loses its reply and ends `NOT_FOUND`; the restore, from a checkpoint without the value, releases its reference; the re-issue pre-stores for itself and commits: with zero or one reference held elsewhere, the value counts exactly its live references after GC, its data stays, and the upload's content was checked once |
//! | `fault_injected::recovery_reissue_committing_after_a_base_holding_the_value_counts_every_live_reference_on_the_*_route` (three) | b.5wb B | the same, restored from a base that already holds the first call's commit: the re-issue's increment is surplus and handed back, and the value counts exactly its live references |
//! | [`fault_injected::recovery_reissue_committing_after_an_object_keyspace_loss_and_a_base_holding_the_value_counts_every_live_reference`] | b.5wb E | the same after every `obj:*` key of the object is lost |
//! | [`fault_injected::recovery_reissue_committing_after_an_object_keyspace_loss_and_a_base_without_the_value_leaks_one_increment`] / [`fault_injected::recovery_reissue_committing_after_a_composed_restore_holding_the_value_leaks_one_increment`] | b.5wb D, C | the leak-direction cases: the value counts its live references plus one, never fewer |
//! | [`fault_injected::recovery_reissue_whose_pre_store_is_refused_after_a_two_attempt_first_call_is_plain_unavailable`] / [`fault_injected::recovery_reissue_whose_pre_store_is_refused_after_a_one_attempt_first_call_answers_the_refusal`] | b.5wb | a refused re-pre-store ends the RPC before the re-issue: a plain `UNAVAILABLE`, outcome unknown, after a counted retry, else the refusal with its detail; the re-issue made no attempt and the failure line keeps the first call's summary |
//! | `fault_injected::predecessor_refusal_after_a_counted_retry_is_plain_unavailable_on_the_*_route` (three) / [`fault_injected::predecessor_refusal_on_the_reissue_after_a_two_attempt_first_call_is_plain_unavailable`] | 14 (b) | a predecessor refusal at the commit after a counted retry, in storage or on the recovery re-issue, is a plain `UNAVAILABLE`, outcome unknown, that does not claim the call applied nothing |
//! | `fault_injected::dropped_chained_call_handed_to_a_stalled_store_*` (two) | 5 (P11, variant S) | a dropped call's commit written to a stalled connection lands late while the record is still its predecessor's, and never after its re-send and successor moved it |
//! | [`fault_injected::dropped_chained_call_queued_through_a_reconnect_is_never_written_and_its_resend_commits_once`] | P11, rider 1 (variant R) | a dropped call's commit still queued in fred's channel while its connection reconnects is never written (the vendored fred's change 9): the record stays its predecessor's, and its re-send and successor then commit once each, in chain order |
//! | `fault_injected::stale_predecessor_*` (two) | 7 | a dropped predecessor's late commit and its re-send, in both orders: each write lands exactly once, in chain order |
//! | [`fault_injected::two_dropped_chained_calls_handed_over_late_land_in_chain_order`] | 8b | two dropped calls' late commits land late, in chain order |
//!
//! # Wait budget
//!
//! Every wait in a test is bounded by that test's one [`TEST_BUDGET`]
//! deadline ([`test_deadline`]); a healthy run meets each in milliseconds,
//! except the waits a test stages on purpose: a call held to its deadline cap,
//! under [`SHORT_DEADLINE`], [`PROBED_CAP_DEADLINE`], a [`CLIENT_TIMEOUT`] or
//! the [`TONIC_TIMEOUT`],
//! which such a test meets once or twice, and the gate tests driven directly
//! that wait towards a deadline at most [`GATE_WAITER_DEADLINE`] away (the
//! two late-arrival tests' 9 s deadlines are never waited for: a late
//! arrival takes no sleep). A call a test
//! parks and ends some other way runs under [`CAPPED_DEADLINE`], which only a
//! regression reaches. The handed-over-commit tests bound each wait for a
//! commit to reach, and to run on, its held connection by `HANDOVER_WAIT`
//! (10 s; the connection stall's own waits are as long), at most a few each.
//! All are far under nextest's 120 s hang-killer.
//!
//! # The margin
//!
//! A waiting call is refused once only its margin `M = max(T/10, 50 ms)` is
//! left before its deadline (`storage::chain_gate`, the deadline cap), so the
//! budgets above leave: [`SHORT_DEADLINE`] 60 ms, [`PROBED_CAP_DEADLINE`]
//! 300 ms, [`CLIENT_TIMEOUT`] 150 ms, [`TONIC_TIMEOUT`] 300 ms,
//! [`CAPPED_DEADLINE`] 3 s, the gate tests'
//! [`GATE_BUDGET`] 1 s, and the late-arrival tests'
//! [`LATE_REPROBE_BUDGET`] 10 s.

use std::sync::Arc;
use std::time::Duration;

use bytes::{BufMut, Bytes};
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, StreamsInterface};
use prost::Message;
use tokio::time::Instant;
use tonic::{Code, Status};

use live_edit::error::{
    refusal_metadata, refusal_reason, ServiceError, CAPACITY_RETRY_DELAY, REFUSAL_DETAIL_DOMAIN,
};
use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer, handle_update_layer};
use live_edit::grpc::proto::{
    update_layer_request, ChainedUpdateLayerHeader, DeleteLayerRequest, OperationOutcome,
    UpdateLayerHeader, UpdateLayerRequest, UpdateLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::upload_admission::{
    admission_charge, UploadAdmissionConfig, UploadAdmissionGate, FIXED_CHARGE_BYTES,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, CHAIN_OUTCOMES, CHAIN_OUTCOME_CANCELLED, CHAIN_OUTCOME_READY, CHAIN_OUTCOME_REFUSED,
    CHAIN_OUTCOME_REPLACED, CHAIN_OUTCOME_SHUTDOWN, CHAIN_OUTCOME_TIMEOUT,
    CHAIN_OUTCOME_UNAVAILABLE, CHAIN_PROBE_ERROR, CHAIN_PROBE_OK,
};
use live_edit::storage::chain_gate::{chain_deadline_margin, ChainGate, ChainLink, GateExit};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{connect_test_client, TestValkey};
use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

use super::helpers::*;
use super::log_capture::{
    assert_common_fields, assert_field, assert_field_absent, capture, purge_keys, ThreadCapture,
    GRPC_CODE_SLUGS, INTERNAL_ERROR_MSG, RPC_ABANDONED_MSG, RPC_FAILURE_MSG,
};
use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline};
use crate::common::refusal_detail::{
    assert_no_refusal_detail, assert_refusal_detail, assert_shutdown_status, decode,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets
// ═════════════════════════════════════════════════════════════════════════════

/// One test's whole wait budget. Unreachable on a healthy run.
const TEST_BUDGET: Duration = Duration::from_secs(60);

/// The deadline a call held to its deadline cap is given, in a test that
/// counts no more than one probe of it. Its margin is 60 ms, so the call
/// waits about 540 ms before its final probe. Not for a test that counts
/// probes up to the cap: a call slowed by load on its way to the gate can
/// arrive with no more than its margin left and is then checked once and
/// refused (a late arrival), so a count of two or more would be a timing
/// assertion — such a test uses [`PROBED_CAP_DEADLINE`].
const SHORT_DEADLINE: Duration = Duration::from_millis(600);

/// The deadline a call held to its deadline cap is given, in a test that
/// counts two or more of its probes before the cap: 3 s, so its margin is
/// 300 ms and it waits about 2.7 s — seconds in which to reach the gate and
/// make several probe round trips, however loaded the runner (test-writing
/// guide, § Performance-Sensitive Tests).
const PROBED_CAP_DEADLINE: Duration = Duration::from_secs(3);

/// The deadline of a call a test parks and then ends some other way: far past
/// anything the test waits for (the cap would end it 3 s before it), so the
/// cap never ends it first.
const CAPPED_DEADLINE: Duration = Duration::from_secs(30);

/// The `grpc-timeout` of the raw-client end-to-end deadline tests that count
/// no more than one probe: a 150 ms margin. The stalled-after-one-read case,
/// which counts two, runs under [`PROBED_CAP_DEADLINE`].
const CLIENT_TIMEOUT: Duration = Duration::from_millis(1_500);

fn test_deadline() -> Instant {
    Instant::now() + TEST_BUDGET
}

async fn settled<T>(deadline: Instant, what: &str, fut: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout_at(deadline, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not finish within the test's {TEST_BUDGET:?}"))
}

/// Poll `cond` until it holds, failing at `deadline`.
async fn until(deadline: Instant, what: &str, mut cond: impl FnMut() -> bool) {
    while !cond() {
        assert!(
            Instant::now() < deadline,
            "{what} did not happen within the test's {TEST_BUDGET:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Run `fut` with an RPC deadline `within` from now in scope, as the metrics
/// layer installs one for a served call: the handler reads it at its entry as
/// the call's deadline budget, which sets the chain gate's bound and margin.
async fn within<F: std::future::Future>(within: Duration, fut: F) -> F::Output {
    live_edit::grpc::deadline::scope(Instant::now() + within, fut).await
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures
// ═════════════════════════════════════════════════════════════════════════════

/// The layer every test writes.
const LAYER: &str = "layer";

/// The `client_id` of the chain under test.
const WRITER: &str = "chain-writer";

/// A layer at `seq = 1`.
#[derive(Clone, Debug)]
struct Target {
    session_id: String,
    object_id: u64,
}

/// A session with one layer at `seq = 1`, created through a separate
/// service, so no service under test has observed anything yet.
async fn new_target(tv: &TestValkey, label: &str) -> Target {
    let setup = build_service(tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = create_session_and_layer(tv, &setup, label, LAYER).await;
    Target {
        session_id,
        object_id,
    }
}

/// `build_service`, whose layer store is `tune`d and counts into the service's
/// metrics (the overtaken counter and the commit-path histograms among them),
/// over `client` — the fixture's own, or a peer replica's.
fn service_over(
    tv: &TestValkey,
    client: ValkeyClient,
    tune: impl FnOnce(LayerStore) -> LayerStore,
) -> LiveEditService {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let keys = tv.key_schema().clone();
    let layer_store = Arc::new(tune(
        LayerStore::new(client.clone(), keys.clone(), 4096).with_metrics(Arc::clone(&base.metrics)),
    ));
    LiveEditService {
        recovery: build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&base.metrics)),
        session_store: Arc::new(SessionStore::new(client.clone(), keys.clone())),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            client.clone(),
            keys.clone(),
        ),
        layer_store,
        valkey: client,
        ..base
    }
}

/// [`service_over`] the fixture's own client, untuned.
fn chain_service(tv: &TestValkey) -> LiveEditService {
    service_over(tv, tv.client().clone(), |store| store)
}

/// A second replica on the same store and key prefix: a `ValkeyClient` of
/// its own, a gate, an upload budget and metrics of its own.
async fn peer_replica(
    tv: &TestValkey,
    tune: impl FnOnce(LayerStore) -> LayerStore,
) -> LiveEditService {
    let peer = connect_test_client(&crate::common::test_config())
        .await
        .expect("the peer replica's client must connect");
    service_over(tv, peer, tune)
}

/// An OMPF setting one inline field, distinct per `tag`.
fn field_ompf(tag: u8) -> Vec<u8> {
    delta_to_ompf_bytes(
        &delta_with_set_field_inline(1, &format!("k{tag}"), vec![tag; 16], 0),
        &[],
    )
}

/// A structural delta creating each `(parent, section)` in `sections`, named
/// after the section: the structural route under the `lua` backend.
fn sections_ompf(sections: &[(u64, u64)]) -> Vec<u8> {
    let delta = OwnedDelta {
        is_diff: false,
        struct_commands: sections
            .iter()
            .map(|&(parent_id, section_id)| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id,
                    section_id,
                    section_name: format!("s{section_id}"),
                    section_type: 1,
                })
            })
            .collect(),
        ..Default::default()
    };
    delta_to_ompf_bytes(&delta, &[])
}

/// A delta the domain refuses while planning, whatever the object's state: it
/// creates section 70 twice (`DuplicateSection`, `INVALID_ARGUMENT`).
fn duplicate_section_ompf() -> Vec<u8> {
    sections_ompf(&[(0, 70), (0, 70)])
}

/// A chained header from `client`.
fn chained_header_from(
    client: &str,
    target: &Target,
    operation_id: &str,
    previous_operation_id: &str,
    no_wait: bool,
) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::ChainedHeader(
            ChainedUpdateLayerHeader {
                header: Some(UpdateLayerHeader {
                    session_id: target.session_id.clone(),
                    layer_name: LAYER.to_owned(),
                    object_id: target.object_id,
                    timestamp_microseconds_since_epoch: 1_000,
                    client_id: client.to_owned(),
                    operation_id: operation_id.to_owned(),
                }),
                previous_operation_id: previous_operation_id.to_owned(),
                no_wait,
            },
        )),
    }
}

/// A chained header from [`WRITER`].
fn chained_header(
    target: &Target,
    operation_id: &str,
    previous_operation_id: &str,
    no_wait: bool,
) -> UpdateLayerRequest {
    chained_header_from(WRITER, target, operation_id, previous_operation_id, no_wait)
}

/// An unchained header from `client` carrying `operation_id` (empty = none).
fn plain_header_from(client: &str, target: &Target, operation_id: &str) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: target.session_id.clone(),
                layer_name: LAYER.to_owned(),
                object_id: target.object_id,
                timestamp_microseconds_since_epoch: 1_000,
                client_id: client.to_owned(),
                operation_id: operation_id.to_owned(),
            },
        )),
    }
}

/// `messages` through `handle_update_layer`, as a client that sent them all
/// and ended its stream.
async fn update(
    svc: &LiveEditService,
    messages: Vec<UpdateLayerRequest>,
) -> Result<UpdateLayerResponse, Status> {
    handle_update_layer(svc, make_request(streaming_update(messages)))
        .await
        .map(tonic::Response::into_inner)
}

/// One chained write from `client`: the header, then `payload` as one chunk.
async fn chained_from(
    svc: &LiveEditService,
    client: &str,
    target: &Target,
    operation_id: &str,
    previous_operation_id: &str,
    no_wait: bool,
    payload: Vec<u8>,
) -> Result<UpdateLayerResponse, Status> {
    update(
        svc,
        vec![
            chained_header_from(client, target, operation_id, previous_operation_id, no_wait),
            make_update_data(payload),
        ],
    )
    .await
}

/// One chained write from [`WRITER`].
async fn chained(
    svc: &LiveEditService,
    target: &Target,
    operation_id: &str,
    previous_operation_id: &str,
    no_wait: bool,
    payload: Vec<u8>,
) -> Result<UpdateLayerResponse, Status> {
    chained_from(
        svc,
        WRITER,
        target,
        operation_id,
        previous_operation_id,
        no_wait,
        payload,
    )
    .await
}

/// One unchained write from `client` (`operation_id` empty = none).
async fn plain_from(
    svc: &LiveEditService,
    client: &str,
    target: &Target,
    operation_id: &str,
    payload: Vec<u8>,
) -> Result<UpdateLayerResponse, Status> {
    update(
        svc,
        vec![
            plain_header_from(client, target, operation_id),
            make_update_data(payload),
        ],
    )
    .await
}

type Pending = tokio::task::JoinHandle<Result<UpdateLayerResponse, Status>>;

/// A chained write from [`WRITER`] on its own task, under an RPC deadline
/// `deadline` from now: a call a test parks in the gate.
fn spawn_chained(
    svc: &Arc<LiveEditService>,
    target: &Target,
    operation_id: &str,
    previous_operation_id: &str,
    no_wait: bool,
    payload: Vec<u8>,
    deadline: Duration,
) -> Pending {
    let svc = Arc::clone(svc);
    let messages = vec![
        chained_header(target, operation_id, previous_operation_id, no_wait),
        make_update_data(payload),
    ];
    tokio::spawn(async move { within(deadline, update(&svc, messages)).await })
}

/// The answer of a spawned call.
async fn answer(
    deadline: Instant,
    what: &str,
    call: Pending,
) -> Result<UpdateLayerResponse, Status> {
    settled(deadline, what, call)
        .await
        .unwrap_or_else(|e| panic!("{what}: the call's task failed: {e}"))
}

/// The wire outcome of `response`.
fn outcome(response: &UpdateLayerResponse) -> OperationOutcome {
    OperationOutcome::try_from(response.operation_outcome).unwrap_or_else(|_| {
        panic!(
            "operation_outcome {} is not an OperationOutcome value",
            response.operation_outcome
        )
    })
}

/// Assert `result` committed as `seq`, `APPLIED`.
#[track_caller]
fn assert_applied(result: &Result<UpdateLayerResponse, Status>, seq: u64, what: &str) {
    let response = result
        .as_ref()
        .unwrap_or_else(|s| panic!("{what}: must commit; got {s:?}"));
    assert_eq!(
        (response.sequence_num, outcome(response)),
        (seq, OperationOutcome::Applied),
        "{what}: the commit's position in the object's history"
    );
}

// ── Store reads ──────────────────────────────────────────────────────────────

/// The object's committed sequence number.
async fn seq(tv: &TestValkey, object_id: u64) -> u64 {
    let raw: String = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_header(&object_id.to_string()), "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");
    raw.parse().expect("seq is numeric")
}

/// The commits on the object's stream after CreateLayer's `1-0`.
async fn commits(tv: &TestValkey, object_id: u64) -> usize {
    let entries: Vec<(String, std::collections::HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", None)
        .await
        .expect("XRANGE must succeed");
    entries.len().saturating_sub(1)
}

/// `client`'s commit record on the object (`obj:wtok`).
async fn record(tv: &TestValkey, object_id: u64, client: &str) -> Option<String> {
    tv.client()
        .pool
        .hget(
            tv.key_schema().obj_write_token(&object_id.to_string()),
            client,
        )
        .await
        .expect("HGET obj:wtok must succeed")
}

/// Whether `record` is the record of `operation_id`.
fn names(record: &Option<String>, operation_id: &str) -> bool {
    record
        .as_deref()
        .is_some_and(|r| r.starts_with(&format!("op.{operation_id}:")))
}

// ── Metrics ──────────────────────────────────────────────────────────────────

fn cell(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    sample_value(&render(metrics), name, labels)
}

/// `live_edit_update_layer_chain_waiters`, seeded at zero.
fn waiters(metrics: &Metrics) -> u64 {
    cell(metrics, "live_edit_update_layer_chain_waiters", &[])
        .expect("live_edit_update_layer_chain_waiters is seeded") as u64
}

/// `live_edit_update_layer_chain_waiting_bytes`, seeded at zero.
fn waiting_bytes(metrics: &Metrics) -> u64 {
    cell(metrics, "live_edit_update_layer_chain_waiting_bytes", &[])
        .expect("live_edit_update_layer_chain_waiting_bytes is seeded") as u64
}

/// `live_edit_update_layer_chain_probes_total{result}`, seeded at zero.
fn probes(metrics: &Metrics, result: &str) -> u64 {
    cell(
        metrics,
        "live_edit_update_layer_chain_probes_total",
        &[("result", result)],
    )
    .unwrap_or_else(|| {
        panic!("live_edit_update_layer_chain_probes_total{{result={result}}} is seeded")
    }) as u64
}

/// `live_edit_update_layer_chain_wait_seconds_count{outcome}`; zero before the
/// cell's first observation (histograms are not pre-registered).
fn waits(metrics: &Metrics, outcome: &str) -> u64 {
    cell(
        metrics,
        "live_edit_update_layer_chain_wait_seconds_count",
        &[("outcome", outcome)],
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_update_layer_chain_detect_lag_seconds_count`; zero before its
/// first observation.
fn lags(metrics: &Metrics) -> u64 {
    cell(
        metrics,
        "live_edit_update_layer_chain_detect_lag_seconds_count",
        &[],
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_update_layer_operation_overtaken_total{backend}` summed over
/// both backends.
fn overtaken(metrics: &Metrics) -> u64 {
    ["lua", "txn"]
        .iter()
        .map(|backend| {
            cell(
                metrics,
                "live_edit_update_layer_operation_overtaken_total",
                &[("backend", backend)],
            )
            .unwrap_or_else(|| panic!("the overtaken {backend} cell is seeded")) as u64
        })
        .sum()
}

/// Assert every gate exit `metrics` observed: each `(outcome, n)` in
/// `expected` exactly `n` times and every other outcome never, and both
/// gauges back at zero — the state every exit must leave.
#[track_caller]
fn assert_gate_exits(metrics: &Metrics, expected: &[(&str, u64)], what: &str) {
    for outcome in CHAIN_OUTCOMES {
        let want = expected
            .iter()
            .find(|(o, _)| o == outcome)
            .map_or(0, |(_, n)| *n);
        assert_eq!(
            waits(metrics, outcome),
            want,
            "{what}: chain_wait_seconds{{outcome={outcome}}}"
        );
    }
    assert_eq!(
        (waiters(metrics), waiting_bytes(metrics)),
        (0, 0),
        "{what}: both gate gauges must be back at zero once every call has left the gate"
    );
}

/// Wait until `n` chained calls wait in `metrics`' gate.
async fn parked(deadline: Instant, metrics: &Metrics, n: u64, what: &str) {
    until(deadline, what, || waiters(metrics) == n).await;
}

/// Wait until `metrics` counted at least `n` successful probes.
async fn probed(deadline: Instant, metrics: &Metrics, n: u64, what: &str) {
    until(deadline, what, || probes(metrics, CHAIN_PROBE_OK) >= n).await;
}

// ── Status assertions ────────────────────────────────────────────────────────

/// The metadata of a refusal naming `operation_id` as the latest record.
fn last_operation(operation_id: &str) -> Vec<(&'static str, String)> {
    vec![
        (
            refusal_metadata::LAST_RECORD,
            refusal_metadata::OPERATION.to_owned(),
        ),
        (refusal_metadata::LAST_OPERATION_ID, operation_id.to_owned()),
    ]
}

/// The metadata of a refusal whose latest record is `last_record` (no
/// operation id).
fn last_record(last_record: &str) -> Vec<(&'static str, String)> {
    vec![(refusal_metadata::LAST_RECORD, last_record.to_owned())]
}

/// Assert `status` is the predecessor refusal: `ABORTED`, an `ErrorInfo`
/// `PREDECESSOR_NOT_COMMITTED` in the service's domain carrying exactly
/// `metadata`, and no `RetryInfo`.
#[track_caller]
fn assert_predecessor_not_committed(status: &Status, metadata: &[(&'static str, String)]) {
    assert_eq!(status.code(), Code::Aborted, "{status:?}");
    let wire = decode(status).unwrap_or_else(|| {
        panic!("the predecessor refusal must carry an ErrorInfo; got {status:?}")
    });
    let mut want: Vec<(String, String)> = metadata
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    want.sort();
    assert_eq!(
        (wire.reason.as_str(), wire.domain.as_str(), &wire.metadata),
        (
            refusal_reason::PREDECESSOR_NOT_COMMITTED,
            REFUSAL_DETAIL_DOMAIN,
            &want
        ),
        "the predecessor refusal's reason, domain and metadata; status {status:?}"
    );
    assert_eq!(
        wire.retry_delay, None,
        "the predecessor refusal is not a capacity refusal and carries no RetryInfo"
    );
}

/// Assert `status` is `INVALID_ARGUMENT` carrying `INVALID_UPDATE` in the
/// service's domain, with no metadata and no `RetryInfo`.
#[track_caller]
fn assert_invalid_update(status: &Status, what: &str) {
    assert_eq!(status.code(), Code::InvalidArgument, "{what}: {status:?}");
    let wire = decode(status).unwrap_or_else(|| {
        panic!(
            "{what}: a chained call's INVALID_ARGUMENT must carry INVALID_UPDATE; got {status:?}"
        )
    });
    assert_eq!(
        (
            wire.reason.as_str(),
            wire.domain.as_str(),
            wire.metadata.len(),
            wire.retry_delay
        ),
        (
            refusal_reason::INVALID_UPDATE,
            REFUSAL_DETAIL_DOMAIN,
            0,
            None
        ),
        "{what}: INVALID_UPDATE carries no metadata and no RetryInfo"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 1 — in order and out of order, across replicas
// ═════════════════════════════════════════════════════════════════════════════

/// One writer's three-write chain sent in order, alternating between two
/// replicas (separate clients, gates and metrics on one store): each write
/// commits directly after its predecessor, `APPLIED`, at the next sequence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_writes_sent_in_order_to_two_replicas_commit_in_chain_order() {
    let tv = TestValkey::setup().await;
    let a = chain_service(&tv);
    let b = peer_replica(&tv, |store| store).await;
    let target = new_target(&tv, "chain-in-order").await;

    let ops = ["in-order-1", "in-order-2", "in-order-3"];
    for (i, replica) in [&a, &b, &a].into_iter().enumerate() {
        let previous = if i == 0 { "" } else { ops[i - 1] };
        let result = chained(
            replica,
            &target,
            ops[i],
            previous,
            false,
            field_ompf(i as u8),
        )
        .await;
        assert_applied(&result, i as u64 + 2, ops[i]);
    }

    assert_eq!(seq(&tv, target.object_id).await, 4);
    assert_eq!(commits(&tv, target.object_id).await, 3);
    assert!(
        names(&record(&tv, target.object_id, WRITER).await, "in-order-3"),
        "the writer's record is its chain's last write"
    );
    assert_gate_exits(&a.metrics, &[(CHAIN_OUTCOME_READY, 2)], "replica A");
    assert_gate_exits(&b.metrics, &[(CHAIN_OUTCOME_READY, 1)], "replica B");
}

/// The same chain sent last-first: write 3 parks on replica A, write 2 on
/// replica B, and only then is write 1 sent. Each commits only directly after
/// its predecessor: 1, 2, 3 at sequences 2, 3, 4, each `APPLIED`.
///
/// The parked writes' records moved while they waited, which is exactly when
/// an unchained operation-id call is overtaken: a chained call never is
/// (group 16, the gate half). Each parked write had read the record once
/// before write 1 was sent, so it saw `ready` only after a sleep, and only
/// those two calls observe the detection lag.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_writes_sent_out_of_order_to_two_replicas_commit_in_chain_order() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let a = Arc::new(chain_service(&tv));
    let b = Arc::new(peer_replica(&tv, |store| store).await);
    let target = new_target(&tv, "chain-out-of-order").await;

    let third = spawn_chained(
        &a,
        &target,
        "out-of-order-3",
        "out-of-order-2",
        false,
        field_ompf(3),
        CAPPED_DEADLINE,
    );
    parked(deadline, &a.metrics, 1, "write 3 parked on replica A").await;
    let second = spawn_chained(
        &b,
        &target,
        "out-of-order-2",
        "out-of-order-1",
        false,
        field_ompf(2),
        CAPPED_DEADLINE,
    );
    parked(deadline, &b.metrics, 1, "write 2 parked on replica B").await;
    probed(deadline, &a.metrics, 1, "write 3's first probe").await;
    probed(deadline, &b.metrics, 1, "write 2's first probe").await;
    assert_eq!(
        seq(&tv, target.object_id).await,
        1,
        "nothing commits before the chain's first write"
    );

    let first = chained(&a, &target, "out-of-order-1", "", false, field_ompf(1)).await;
    assert_applied(&first, 2, "write 1");
    assert_applied(
        &answer(deadline, "write 2", second).await,
        3,
        "write 2, after write 1",
    );
    assert_applied(
        &answer(deadline, "write 3", third).await,
        4,
        "write 3, after write 2",
    );

    assert_eq!(commits(&tv, target.object_id).await, 3);
    assert_eq!(
        (overtaken(&a.metrics), overtaken(&b.metrics)),
        (0, 0),
        "a chained call whose record moved while it waited is never overtaken"
    );
    assert_gate_exits(&a.metrics, &[(CHAIN_OUTCOME_READY, 2)], "replica A");
    assert_gate_exits(&b.metrics, &[(CHAIN_OUTCOME_READY, 1)], "replica B");
    assert_eq!(
        (lags(&a.metrics), lags(&b.metrics)),
        (1, 1),
        "the detection lag is observed only on a ready exit after a sleep: write 3 on A and \
         write 2 on B, never write 1, which was ready at its first probe"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 2 — depth-3 pipelining
// ═════════════════════════════════════════════════════════════════════════════

/// Write 3 parks; write 1 commits, so the record moves to write 1 — still not
/// write 3's predecessor, and write 3 keeps waiting (it has probed the moved
/// record since); then write 2 commits after write 1, and write 3 after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn depth_three_chain_keeps_waiting_while_the_record_moves_through_each_predecessor() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(chain_service(&tv));
    let target = new_target(&tv, "chain-depth-3").await;

    let third = spawn_chained(
        &svc,
        &target,
        "depth-3",
        "depth-2",
        false,
        field_ompf(3),
        CAPPED_DEADLINE,
    );
    parked(deadline, &svc.metrics, 1, "write 3 parked").await;

    assert_applied(
        &chained(&svc, &target, "depth-1", "", false, field_ompf(1)).await,
        2,
        "write 1",
    );
    // Two more successful probes after write 1 committed: whichever probe was
    // in flight across the commit, the later one read write 1's record.
    let after_first = probes(&svc.metrics, CHAIN_PROBE_OK);
    probed(
        deadline,
        &svc.metrics,
        after_first + 2,
        "write 3 probing the moved record",
    )
    .await;
    assert!(
        !third.is_finished(),
        "write 3's predecessor is write 2: a record at write 1 must keep it waiting"
    );
    assert_eq!(waiters(&svc.metrics), 1);

    assert_applied(
        &chained(&svc, &target, "depth-2", "depth-1", false, field_ompf(2)).await,
        3,
        "write 2",
    );
    assert_applied(
        &answer(deadline, "write 3", third).await,
        4,
        "write 3, directly after write 2",
    );
    assert_eq!(commits(&tv, target.object_id).await, 3);
    assert_eq!(overtaken(&svc.metrics), 0);
    assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_READY, 3)], "depth 3");
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 3 — a predecessor refused at admission
// ═════════════════════════════════════════════════════════════════════════════

/// The upload budget holds exactly one upload's admission (`F + A`) plus a
/// little: a successor waiting in the gate keeps `F` plus its content, so its
/// predecessor's admission charge does not fit and, with no admission wait,
/// is refused `UPLOAD_CAPACITY` at once — the waiting chained calls filling
/// the budget, the hazard `grpc::upload_admission`'s module doc names.
fn budget_of_one_admission() -> UploadAdmissionConfig {
    UploadAdmissionConfig {
        budget_bytes: admission_charge() + 4_096,
        wait: Duration::ZERO,
        max_waiters: 4,
        max_upload_bytes: 0,
    }
}

/// The predecessor P is refused at admission while its successor S waits:
/// the service never ends S for its predecessor's failure, so S keeps waiting
/// — holding `F` plus its content, which both gauges and the reservation show
/// — until the client cancels it. The cancel applies nothing and returns the
/// reservation and both gauges; P's re-send then commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn predecessor_refused_at_admission_leaves_its_successor_waiting_until_the_client_cancels_it()
{
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let base = chain_service(&tv);
    let svc = Arc::new(LiveEditService {
        upload_admission: Arc::new(UploadAdmissionGate::new(
            budget_of_one_admission(),
            Some(Arc::clone(&base.metrics)),
        )),
        ..base
    });
    let target = new_target(&tv, "chain-admission").await;
    let s_payload = field_ompf(2);
    let held = FIXED_CHARGE_BYTES + s_payload.len() as u64;

    let successor = spawn_chained(
        &svc,
        &target,
        "admission-s",
        "admission-p",
        false,
        s_payload,
        CAPPED_DEADLINE,
    );
    parked(deadline, &svc.metrics, 1, "the successor parked").await;
    assert_eq!(
        (
            waiting_bytes(&svc.metrics),
            svc.upload_admission.reserved_bytes()
        ),
        (held, held),
        "a waiting call holds its reservation's fixed charge plus its content, and nothing else"
    );

    let refused = chained(&svc, &target, "admission-p", "", false, field_ompf(1))
        .await
        .expect_err("the predecessor must be refused at admission");
    assert_eq!(refused.code(), Code::Unavailable, "{refused:?}");
    assert_eq!(
        decode(&refused).map(|w| w.reason),
        Some(refusal_reason::UPLOAD_CAPACITY.to_owned()),
        "precondition: the predecessor was refused at admission; got {refused:?}"
    );
    assert!(
        !successor.is_finished() && waiters(&svc.metrics) == 1,
        "the service never refuses a waiting call because its predecessor's call failed"
    );

    successor.abort();
    let ended = settled(deadline, "the cancelled successor", successor).await;
    assert!(
        ended.is_err_and(|e| e.is_cancelled()),
        "precondition: the successor was dropped while it waited"
    );
    assert_gate_exits(
        &svc.metrics,
        &[(CHAIN_OUTCOME_CANCELLED, 1)],
        "the cancel at the gate",
    );
    assert_eq!(svc.upload_admission.reserved_bytes(), 0);
    assert_eq!(
        cell(&svc.metrics, "live_edit_upload_reserved_bytes", &[]),
        Some(0.0),
        "the reservation's bytes return with the cancel"
    );
    assert_eq!(seq(&tv, target.object_id).await, 1, "nothing applied");
    assert_eq!(record(&tv, target.object_id, WRITER).await, None);

    assert_applied(
        &chained(&svc, &target, "admission-p", "", false, field_ompf(1)).await,
        2,
        "the predecessor's re-send, once the budget is free",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 6 — a predecessor lost in transport and re-sent
// ═════════════════════════════════════════════════════════════════════════════

/// A request body that yields `frames` and then ends WITHOUT END_STREAM: a
/// stream lost in transport (the `RST_STREAM(NO_ERROR)` or `GOAWAY` shape).
fn lost_in_transport(frames: Vec<Bytes>) -> AttestedStreaming<UpdateLayerRequest> {
    let frames = futures::stream::iter(
        frames
            .into_iter()
            .map(|b| Ok::<_, std::convert::Infallible>(http_body::Frame::data(b))),
    );
    AttestedStreaming::new(http_body_util::StreamBody::new(frames))
}

/// One message as a gRPC frame.
fn grpc_frame<T: Message>(msg: &T) -> Bytes {
    let encoded = msg.encode_to_vec();
    let mut buf = bytes::BytesMut::with_capacity(5 + encoded.len());
    buf.put_u8(0);
    buf.put_u32(encoded.len() as u32);
    buf.extend_from_slice(&encoded);
    buf.freeze()
}

/// P is lost in transport — its stream ends without END_STREAM, so it is
/// `CANCELLED` with nothing applied — while its successor S waits. The
/// client re-sends P (P′, the same operation); P′ commits, and S commits
/// directly after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn predecessor_lost_in_transport_is_cancelled_and_its_resend_releases_the_successor() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(chain_service(&tv));
    let target = new_target(&tv, "chain-lost").await;

    let successor = spawn_chained(
        &svc,
        &target,
        "lost-s",
        "lost-p",
        false,
        field_ompf(2),
        CAPPED_DEADLINE,
    );
    parked(deadline, &svc.metrics, 1, "the successor parked").await;

    let lost = handle_update_layer(
        &svc,
        make_request(lost_in_transport(vec![
            grpc_frame(&chained_header(&target, "lost-p", "", false)),
            grpc_frame(&make_update_data(field_ompf(1))),
        ])),
    )
    .await
    .map(tonic::Response::into_inner)
    .expect_err("a predecessor whose stream ended without END_STREAM must not commit");
    assert_eq!(lost.code(), Code::Cancelled, "{lost:?}");
    assert_eq!(seq(&tv, target.object_id).await, 1, "P applied nothing");
    assert!(!successor.is_finished(), "S keeps waiting for P");

    assert_applied(
        &chained(&svc, &target, "lost-p", "", false, field_ompf(1)).await,
        2,
        "P′, P's re-send",
    );
    assert_applied(
        &answer(deadline, "S", successor).await,
        3,
        "S, directly after P′",
    );
    assert_eq!(commits(&tv, target.object_id).await, 2);
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 8 (a) — a passed predecessor
// ═════════════════════════════════════════════════════════════════════════════

/// The writer's three operations committed with ordinary (unchained)
/// operation-id calls: the record is `passed-3`, past `passed-1` and
/// `passed-2`.
async fn stage_passed_chain(svc: &LiveEditService, target: &Target) {
    for (i, op) in ["passed-1", "passed-2", "passed-3"].into_iter().enumerate() {
        let staged = plain_from(svc, WRITER, target, op, field_ompf(i as u8)).await;
        assert_applied(&staged, i as u64 + 2, op);
    }
}

/// A re-send of `passed-2` (predecessor `passed-1`) after `passed-3`
/// committed, with `no_wait`: the record is neither its predecessor nor its
/// own, so it is refused `ABORTED` at once, from its first probe, naming the
/// latest write — and applies nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn passed_predecessor_with_no_wait_is_refused_at_once_naming_the_latest_write() {
    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);
    let target = new_target(&tv, "chain-passed-no-wait").await;
    stage_passed_chain(&svc, &target).await;

    let refused = within(
        CAPPED_DEADLINE,
        chained(&svc, &target, "passed-2", "passed-1", true, field_ompf(1)),
    )
    .await
    .expect_err("a passed predecessor must be refused under no_wait");
    assert_predecessor_not_committed(&refused, &last_operation("passed-3"));
    assert_eq!(
        (
            probes(&svc.metrics, CHAIN_PROBE_OK),
            probes(&svc.metrics, CHAIN_PROBE_ERROR)
        ),
        (1, 0),
        "no_wait answers from its first probe: no sleep, no second probe"
    );
    assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_REFUSED, 1)], "no_wait");
    assert_eq!(seq(&tv, target.object_id).await, 4, "nothing applied");
}

/// The same re-send without `no_wait` waits — the record never becomes its
/// predecessor again — until its deadline cap, then is refused `ABORTED`
/// naming the latest write, outcome `timeout`, having applied nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn passed_predecessor_without_no_wait_waits_until_its_deadline_cap() {
    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);
    let target = new_target(&tv, "chain-passed-capped").await;
    stage_passed_chain(&svc, &target).await;

    let refused = within(
        PROBED_CAP_DEADLINE,
        chained(&svc, &target, "passed-2", "passed-1", false, field_ompf(1)),
    )
    .await
    .expect_err("a call whose predecessor never becomes the record is refused at its cap");
    assert_predecessor_not_committed(&refused, &last_operation("passed-3"));
    assert!(
        refused.message().contains("within the call's deadline"),
        "the cap's refusal names the deadline; got {refused:?}"
    );
    assert!(
        probes(&svc.metrics, CHAIN_PROBE_OK) >= 2,
        "the call waited: more than one probe"
    );
    assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_TIMEOUT, 1)], "capped");
    assert_eq!(seq(&tv, target.object_id).await, 4, "nothing applied");
}

/// The same re-send without `no_wait`, cancelled by its client while it
/// waits: nothing applied, outcome `cancelled`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn passed_predecessor_without_no_wait_waits_until_cancelled() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(chain_service(&tv));
    let target = new_target(&tv, "chain-passed-cancel").await;
    stage_passed_chain(&svc, &target).await;

    let waiting = spawn_chained(
        &svc,
        &target,
        "passed-2",
        "passed-1",
        false,
        field_ompf(1),
        CAPPED_DEADLINE,
    );
    parked(deadline, &svc.metrics, 1, "the re-send parked").await;
    probed(
        deadline,
        &svc.metrics,
        2,
        "the re-send probing the passed record",
    )
    .await;
    assert!(
        !waiting.is_finished(),
        "a passed predecessor keeps the call waiting"
    );
    waiting.abort();
    assert!(settled(deadline, "the cancel", waiting)
        .await
        .is_err_and(|e| e.is_cancelled()));
    assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_CANCELLED, 1)], "cancelled");
    assert_eq!(seq(&tv, target.object_id).await, 4, "nothing applied");
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 9 — the no_wait matrix
// ═════════════════════════════════════════════════════════════════════════════

/// `no_wait` answers from the record alone, one probe per call, for each
/// shape of the latest record of the call's `(object_id, client_id)` — each
/// row its own `client_id` on one layer:
///
/// | Record | Answer |
/// |---|---|
/// | the predecessor's | commits, `APPLIED` |
/// | its own (a re-send of a committed write) | `ALREADY_COMMITTED`, the original sequence |
/// | another operation's | `ABORTED`, `last_record=operation`, `last_operation_id` |
/// | none, with a predecessor named | `ABORTED`, `last_record=absent` |
/// | none, starting a chain | commits, `APPLIED` |
/// | a commit without an `operation_id` | `ABORTED`, `last_record=no_operation_id` |
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_wait_matrix_answers_from_the_record_alone() {
    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);
    let target = new_target(&tv, "chain-no-wait-matrix").await;
    let oid = target.object_id;

    // Staged records, one per row's client_id.
    assert_applied(
        &plain_from(&svc, "row-predecessor", &target, "m-prev", field_ompf(1)).await,
        2,
        "staging the predecessor row",
    );
    let own_payload = field_ompf(2);
    let own = plain_from(&svc, "row-own", &target, "m-own", own_payload.clone()).await;
    assert_applied(&own, 3, "staging the own row");
    assert_applied(
        &plain_from(&svc, "row-other", &target, "m-other", field_ompf(3)).await,
        4,
        "staging the other row",
    );
    let no_op_staged = plain_from(&svc, "row-no-op", &target, "", field_ompf(4))
        .await
        .expect("staging the no-operation-id row must commit");
    assert_eq!(
        (no_op_staged.sequence_num, outcome(&no_op_staged)),
        (5, OperationOutcome::Unspecified),
        "staging the no-operation-id row: a commit without an operation id"
    );

    // The predecessor's record: commits.
    assert_applied(
        &chained_from(
            &svc,
            "row-predecessor",
            &target,
            "m-next",
            "m-prev",
            true,
            field_ompf(5),
        )
        .await,
        6,
        "no_wait over its predecessor's record",
    );

    // Its own record: answered ALREADY_COMMITTED with the original commit.
    let again = chained_from(
        &svc,
        "row-own",
        &target,
        "m-own",
        "m-before",
        true,
        own_payload,
    )
    .await
    .expect("a re-send of a committed chained write is answered OK");
    assert_eq!(
        (again.sequence_num, outcome(&again)),
        (3, OperationOutcome::AlreadyCommitted),
        "the re-send reports the original commit and applies nothing"
    );

    // Another operation's record.
    let other = chained_from(
        &svc,
        "row-other",
        &target,
        "m-x",
        "m-y",
        true,
        field_ompf(6),
    )
    .await
    .expect_err("another operation's record refuses a no_wait call");
    assert_predecessor_not_committed(&other, &last_operation("m-other"));

    // No record, a predecessor named.
    let absent = chained_from(
        &svc,
        "row-absent",
        &target,
        "m-x",
        "m-y",
        true,
        field_ompf(7),
    )
    .await
    .expect_err("no record refuses a no_wait call that names a predecessor");
    assert_predecessor_not_committed(&absent, &last_record(refusal_metadata::ABSENT));

    // No record, a chain's first write: commits.
    assert_applied(
        &chained_from(
            &svc,
            "row-absent-start",
            &target,
            "m-start",
            "",
            true,
            field_ompf(8),
        )
        .await,
        7,
        "no_wait starting a chain on no record",
    );

    // A commit without an operation id.
    let no_op = chained_from(
        &svc,
        "row-no-op",
        &target,
        "m-x",
        "m-y",
        true,
        field_ompf(9),
    )
    .await
    .expect_err("a commit without an operation id refuses a no_wait call");
    assert_predecessor_not_committed(&no_op, &last_record(refusal_metadata::NO_OPERATION_ID));

    assert_eq!(
        seq(&tv, oid).await,
        7,
        "only the two rows that commit applied anything"
    );
    assert_eq!(
        probes(&svc.metrics, CHAIN_PROBE_OK),
        6,
        "one probe per no_wait call, never a sleep"
    );
    assert_gate_exits(
        &svc.metrics,
        &[(CHAIN_OUTCOME_READY, 3), (CHAIN_OUTCOME_REFUSED, 3)],
        "the matrix",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 10 — no deadlock at the writer limit
// ═════════════════════════════════════════════════════════════════════════════

/// Writers in [`many_writers_chained_in_reverse`].
const WRITERS: usize = 3;
/// Writes per writer.
const WRITES: usize = 8;
/// Each replica's per-object writer limit: fewer than the writers' calls in
/// flight, so a waiter holding a writer permit would deadlock the chain.
const WRITER_PERMITS: u32 = 2;

/// Every writer's chain sent last-first, alternating between two replicas
/// whose per-object writer limit is [`WRITER_PERMITS`]: every call commits,
/// `APPLIED`, and each writer's writes commit in chain order. A waiting call
/// holds no writer permit, so the chain's later writes cannot starve its
/// first.
///
/// `path` is the commit path the test names, the `path` label of
/// `live_edit_update_layer_attempts_histogram`: some of the calls must have
/// committed on it, summed over both replicas — under contention a call may
/// fall back to another route, so without this a test named for a route
/// could pass with every call committed elsewhere.
async fn many_writers_chained_in_reverse(
    label: &str,
    path: &str,
    payload: impl Fn(usize, usize) -> Vec<u8>,
) {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let limit = |store: LayerStore| store.with_update_layer_per_object_concurrency(WRITER_PERMITS);
    let a = Arc::new(service_over(&tv, tv.client().clone(), limit));
    let b = Arc::new(peer_replica(&tv, limit).await);
    let target = new_target(&tv, label).await;
    let op = |w: usize, i: usize| format!("{label}-w{w}-{i}");

    let mut calls = Vec::new();
    for i in (0..WRITES).rev() {
        for w in 0..WRITERS {
            let replica = if (w + i) % 2 == 0 { &a } else { &b };
            let previous = if i == 0 { String::new() } else { op(w, i - 1) };
            let svc = Arc::clone(replica);
            let messages = vec![
                chained_header_from(&format!("writer-{w}"), &target, &op(w, i), &previous, false),
                make_update_data(payload(w, i)),
            ];
            calls.push((
                w,
                i,
                tokio::spawn(async move { within(CAPPED_DEADLINE, update(&svc, messages)).await }),
            ));
        }
    }

    let mut seqs = vec![vec![0u64; WRITES]; WRITERS];
    for (w, i, call) in calls {
        let what = format!("writer {w} write {i}");
        let response = answer(deadline, &what, call)
            .await
            .unwrap_or_else(|s| panic!("{what}: every chained call must commit; got {s:?}"));
        assert_eq!(outcome(&response), OperationOutcome::Applied, "{what}");
        seqs[w][i] = response.sequence_num;
    }
    for (w, writer) in seqs.iter().enumerate() {
        assert!(
            writer.windows(2).all(|pair| pair[0] < pair[1]),
            "writer {w}'s writes must commit in chain order; sequences {writer:?}"
        );
    }
    let total = (WRITERS * WRITES) as u64;
    assert_eq!(seq(&tv, target.object_id).await, 1 + total);
    assert_eq!(commits(&tv, target.object_id).await, total as usize);
    assert_eq!(
        waits(&a.metrics, CHAIN_OUTCOME_READY) + waits(&b.metrics, CHAIN_OUTCOME_READY),
        total
    );
    let on_path: f64 = [&a, &b]
        .iter()
        .filter_map(|replica| {
            live_edit::observability::metrics::test_util::sample_sum(
                &render(&replica.metrics),
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", path)],
            )
        })
        .sum();
    assert!(
        on_path > 0.0,
        "{label}: some calls must commit on the {path} path the test names; none did"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_writers_chained_in_reverse_at_the_writer_limit_never_deadlock_on_the_fast_route() {
    many_writers_chained_in_reverse(
        "limit-fast",
        live_edit::observability::metrics::UPDATE_LAYER_PATH_FAST,
        |w, i| field_ompf((w * WRITES + i) as u8),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_writers_chained_in_reverse_at_the_writer_limit_never_deadlock_on_the_structural_route(
) {
    many_writers_chained_in_reverse(
        "limit-structural",
        live_edit::observability::metrics::UPDATE_LAYER_PATH_STRUCTURAL,
        |w, i| sections_ompf(&[(0, 100 + (w * WRITES + i) as u64)]),
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 11 — the object replaced or deleted mid-chain; Q2
// ═════════════════════════════════════════════════════════════════════════════

/// `CreateLayer` overwriting `target`'s layer through `svc`: the new object.
async fn overwrite(svc: &LiveEditService, target: &Target) -> u64 {
    handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            make_create_header(&target.session_id, LAYER),
            make_create_data(delta_to_ompf_bytes(
                &crate::common::owned_delta::empty_delta(),
                &[],
            )),
        ])),
    )
    .await
    .expect("the CreateLayer overwrite must succeed")
    .into_inner()
    .object_id
}

/// A waiting chained call whose layer is deleted: its probe finds the header
/// gone, the layer resolves to nothing, and the call is answered `NOT_FOUND`
/// with no status details (Q2: a chained call's NOT_FOUND is plain, and
/// settling), outcome `replaced`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_waiting_when_the_layer_is_deleted_is_answered_not_found_without_status_details(
) {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(chain_service(&tv));
    let target = new_target(&tv, "chain-deleted").await;

    let waiting = spawn_chained(
        &svc,
        &target,
        "deleted-s",
        "deleted-p",
        false,
        field_ompf(1),
        CAPPED_DEADLINE,
    );
    parked(deadline, &svc.metrics, 1, "the call parked").await;

    let mut delete = tonic::Request::new(DeleteLayerRequest {
        session_id: target.session_id.clone(),
        layer_name: LAYER.to_owned(),
    });
    delete
        .extensions_mut()
        .insert(live_edit::grpc::BearerToken::new("test-token"));
    handle_delete_layer(&svc, delete)
        .await
        .expect("DeleteLayer must succeed");

    let status = answer(deadline, "the waiting call", waiting)
        .await
        .expect_err("a waiting call whose layer was deleted must fail");
    assert_eq!(status.code(), Code::NotFound, "{status:?}");
    assert_no_refusal_detail(&status);
    assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_REPLACED, 1)], "deleted");
}

/// The header-removed shape: the object's header vanishes while its mapping
/// still names it (the object a `CreateLayer` overwrite's cleanup is about to
/// replace). The waiting call re-resolves, finds the same object and keeps
/// waiting; the overwrite then moves the mapping, and the call is answered
/// `FAILED_PRECONDITION` with no status details, outcome `replaced`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_waiting_when_its_header_is_removed_and_the_layer_is_replaced_is_answered_failed_precondition(
) {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(chain_service(&tv));
    let target = new_target(&tv, "chain-header-removed").await;

    let waiting = spawn_chained(
        &svc,
        &target,
        "removed-s",
        "removed-p",
        false,
        field_ompf(1),
        CAPPED_DEADLINE,
    );
    parked(deadline, &svc.metrics, 1, "the call parked").await;

    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().obj_header(&target.object_id.to_string()))
        .await
        .expect("DEL obj:hdr must succeed");
    let before = probes(&svc.metrics, CHAIN_PROBE_OK);
    probed(
        deadline,
        &svc.metrics,
        before + 2,
        "probes of the removed header",
    )
    .await;
    assert!(
        !waiting.is_finished() && waiters(&svc.metrics) == 1,
        "while the mapping still names the object, the call keeps waiting"
    );
    assert_eq!(waits(&svc.metrics, CHAIN_OUTCOME_REPLACED), 0);

    let replacement = overwrite(&svc, &target).await;
    let status = answer(deadline, "the waiting call", waiting)
        .await
        .expect_err("a waiting call whose object was replaced must fail");
    assert_eq!(status.code(), Code::FailedPrecondition, "{status:?}");
    assert_eq!(
        status.message(),
        format!(
            "object_id mismatch: expected {replacement}, got {}",
            target.object_id
        )
    );
    assert_no_refusal_detail(&status);
    assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_REPLACED, 1)], "replaced");
}

/// Mark `target`'s object `superseded=1` (`on`), as a `CreateLayer` overwrite
/// does before it moves the mapping — a probe then reads `replaced` while the
/// mapping still names the object — or remove the mark again (`HDEL`; the
/// commit script treats any `superseded` field as set).
async fn set_superseded(tv: &TestValkey, target: &Target, on: bool) {
    let header_key = tv.key_schema().obj_header(&target.object_id.to_string());
    let pool = &tv.client().pool;
    if on {
        let _: i64 = pool
            .hset(&header_key, [("superseded", "1")])
            .await
            .expect("HSET superseded=1");
    } else {
        let _: i64 = pool
            .hdel(&header_key, "superseded")
            .await
            .expect("HDEL superseded");
    }
}

/// `DEL` `target`'s object header, leaving its mapping naming it: the shape a
/// `CreateLayer` overwrite's cleanup leaves before it moves the mapping, or a
/// header lost under a live mapping. Every gate probe then reads `replaced`,
/// and every re-resolve finds the same object.
async fn remove_header(tv: &TestValkey, target: &Target) {
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().obj_header(&target.object_id.to_string()))
        .await
        .expect("DEL obj:hdr must succeed");
}

/// Review round 1, item 1 — replaced again after re-resolving to the same
/// object (`ChainGate::wait`'s arm for a `replaced` probe after
/// `keep_waiting`). The object's header is gone while its mapping still names
/// it, so each probe reads `replaced` and each re-resolve finds the same
/// object; the writer's record is `gone-0`, not the call's predecessor.
///
/// - With `no_wait`: the first probe reads `replaced`, the re-resolve names the
///   same object, and the second probe reads `replaced` again — a call that
///   cannot wait is then refused at once: `ABORTED`
///   `PREDECESSOR_NOT_COMMITTED` naming the record, outcome `refused`, after
///   exactly those two probes. Without that arm it would go back to the
///   handler to re-resolve and probe again, over and over, and be answered
///   only once its re-entered probe's limit (`D − M/2`) had passed, outcome
///   `timeout`.
/// - Without `no_wait`, under [`PROBED_CAP_DEADLINE`]: it keeps waiting — probing
///   and re-resolving, the header still gone — until its margin, then
///   `ABORTED` naming the record, outcome `timeout`.
///
/// Never `FAILED_PRECONDITION` or `NOT_FOUND`: the layer still names the
/// object. Nothing applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_whose_header_is_gone_under_a_live_mapping_is_refused_with_no_wait_at_once_and_otherwise_at_its_cap(
) {
    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);

    let target = new_target(&tv, "chain-gone-no-wait").await;
    assert_applied(
        &plain_from(&svc, WRITER, &target, "gone-0", field_ompf(0)).await,
        2,
        "the writer's latest write",
    );
    remove_header(&tv, &target).await;
    let refused = within(
        CAPPED_DEADLINE,
        chained(&svc, &target, "gone-2", "gone-1", true, field_ompf(2)),
    )
    .await
    .expect_err("a no_wait call that reads replaced after a same-object re-resolve is refused");
    assert_predecessor_not_committed(&refused, &last_operation("gone-0"));
    assert_eq!(
        (
            probes(&svc.metrics, CHAIN_PROBE_OK),
            probes(&svc.metrics, CHAIN_PROBE_ERROR)
        ),
        (2, 0),
        "no_wait: the probe before the re-resolve and the one after it, both read; no third"
    );
    assert_gate_exits(
        &svc.metrics,
        &[(CHAIN_OUTCOME_REFUSED, 1)],
        "no_wait, replaced again",
    );
    assert!(
        names(&record(&tv, target.object_id, WRITER).await, "gone-0"),
        "nothing applied: the record is still the writer's latest write"
    );

    let svc = chain_service(&tv);
    let target = new_target(&tv, "chain-gone-capped").await;
    assert_applied(
        &plain_from(&svc, WRITER, &target, "gone-b-0", field_ompf(0)).await,
        2,
        "the writer's latest write",
    );
    remove_header(&tv, &target).await;
    let refused = within(
        PROBED_CAP_DEADLINE,
        chained(&svc, &target, "gone-b-2", "gone-b-1", false, field_ompf(2)),
    )
    .await
    .expect_err("a waiting call that reads replaced to its cap is refused at the cap");
    assert_predecessor_not_committed(&refused, &last_operation("gone-b-0"));
    assert!(
        refused.message().contains("within the call's deadline"),
        "the cap's refusal names the deadline; got {refused:?}"
    );
    assert!(
        probes(&svc.metrics, CHAIN_PROBE_OK) >= 3,
        "the call kept waiting across re-resolves: more than two probes read"
    );
    assert_gate_exits(
        &svc.metrics,
        &[(CHAIN_OUTCOME_TIMEOUT, 1)],
        "replaced again, at the cap",
    );
    assert!(
        names(&record(&tv, target.object_id, WRITER).await, "gone-b-0"),
        "nothing applied: the record is still the writer's latest write"
    );
}

/// Review round 1, item 1 — the `recovered_once` the gate's re-resolve shares
/// with the handler's step 3. A waiting call's probe reads `replaced` (the
/// header marked `superseded`) and its re-resolve finds no mapping (the proxy
/// in front of the store answers that one `GET` with nothing): the call runs
/// its one recovery — the object is still there, so the hook reports it
/// restored — and the mapping, read again, names the same object. So the call
/// keeps waiting, and commits directly after its predecessor once the object
/// is live and the predecessor commits.
///
/// A second call loses its mapping twice: the first loss is recovered and it
/// keeps waiting; the second, in the same call, is answered `NOT_FOUND`
/// (plain, settling, outcome `replaced`) with no second recovery — one per
/// call, step 3's and the gate's together.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_whose_re_resolve_recovers_the_same_object_keeps_waiting_and_a_second_loss_is_not_found(
) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let target = new_target(&tv, "chain-recovered").await;

    // The next `lose` mapping GETs through the proxy read no mapping.
    let lose = Arc::new(AtomicUsize::new(0));
    let rule: ProxyRule = {
        let lose = Arc::clone(&lose);
        let mapping_key = tv
            .key_schema()
            .layer_mapping(&target.session_id, LAYER)
            .into_bytes();
        Arc::new(move |args: &[Vec<u8>]| {
            let is_mapping_get = args
                .first()
                .is_some_and(|name| name.eq_ignore_ascii_case(b"GET"))
                && args.get(1) == Some(&mapping_key);
            (is_mapping_get
                && lose
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                    .is_ok())
            .then(|| ProxyAction::Eval("return false".to_owned()))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let mut svc = service_through(&tv, &proxy).await;
    let metrics = crate::common::replace_metrics(&mut svc);
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = true;
    config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
    svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
        Arc::new(live_edit::checkpoint::adapters::InMemoryCheckpointStore::new())
            as Arc<dyn live_edit::checkpoint::CheckpointStore>,
        Arc::clone(&svc.layer_store),
        Arc::new(config),
        Arc::clone(&svc.metrics),
    ));
    let svc = Arc::new(svc);
    let restored = || crate::common::recovery_metric(&metrics, "restored");
    let supersede = |on: bool| set_superseded(&tv, &target, on);

    // The first call: one loss, recovered; it keeps waiting, then commits.
    let waiting = spawn_chained(
        &svc,
        &target,
        "rec-s",
        "rec-p",
        false,
        field_ompf(2),
        CAPPED_DEADLINE,
    );
    parked(deadline, &metrics, 1, "the call parked").await;
    lose.store(1, Ordering::SeqCst);
    supersede(true).await;
    until(deadline, "the re-resolve's recovery", || restored() == 1).await;
    supersede(false).await;
    assert_eq!(
        lose.load(Ordering::SeqCst),
        0,
        "precondition: the re-resolve read no mapping"
    );
    let before = probes(&metrics, CHAIN_PROBE_OK);
    probed(deadline, &metrics, before + 2, "probes after the recovery").await;
    assert!(
        !waiting.is_finished() && waiters(&metrics) == 1,
        "the recovered mapping names the same object: the call keeps waiting"
    );
    assert_eq!(waits(&metrics, CHAIN_OUTCOME_REPLACED), 0);
    assert_applied(
        &chained(
            &chain_service(&tv),
            &target,
            "rec-p",
            "",
            false,
            field_ompf(1),
        )
        .await,
        2,
        "the predecessor",
    );
    assert_applied(
        &answer(deadline, "the recovered call", waiting).await,
        3,
        "the recovered call, directly after its predecessor",
    );

    // The second call: a second loss in one call is not recovered again.
    let waiting = spawn_chained(
        &svc,
        &target,
        "rec-s3",
        "rec-s2",
        false,
        field_ompf(3),
        CAPPED_DEADLINE,
    );
    parked(deadline, &metrics, 1, "the second call parked").await;
    lose.store(1, Ordering::SeqCst);
    supersede(true).await;
    until(deadline, "the second call's recovery", || restored() == 2).await;
    lose.store(1, Ordering::SeqCst);
    let status = answer(deadline, "the second call", waiting)
        .await
        .expect_err("a second lost mapping in one call is not recovered again");
    assert_eq!(status.code(), Code::NotFound, "{status:?}");
    assert_no_refusal_detail(&status);
    assert_eq!(
        restored(),
        2,
        "one recovery per call: the second loss ran none"
    );
    assert_gate_exits(
        &metrics,
        &[(CHAIN_OUTCOME_READY, 1), (CHAIN_OUTCOME_REPLACED, 1)],
        "recovered, then lost again",
    );
    assert_eq!(
        seq(&tv, target.object_id).await,
        3,
        "the second call applied nothing"
    );
    supersede(false).await;
}

/// Q2 at step 3: a chained call naming a stale object is refused
/// `FAILED_PRECONDITION`, plain, before it reaches the gate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_to_a_stale_object_id_is_refused_failed_precondition_without_status_details() {
    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);
    let target = new_target(&tv, "chain-stale").await;
    let stale = Target {
        object_id: target.object_id + 1_000,
        ..target.clone()
    };
    let status = chained(&svc, &stale, "stale-s", "", false, field_ompf(1))
        .await
        .expect_err("a stale object_id must be refused");
    assert_eq!(status.code(), Code::FailedPrecondition, "{status:?}");
    assert_no_refusal_detail(&status);
    assert_gate_exits(&svc.metrics, &[], "refused before the gate");
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 12 — a pre-chaining replica fails closed
// ═════════════════════════════════════════════════════════════════════════════

/// A request body over `frames` that counts the frames its reader pulled and
/// reports END_STREAM once they are all read.
struct CountedBody {
    frames: std::collections::VecDeque<Bytes>,
    pulled: Arc<std::sync::atomic::AtomicUsize>,
}

impl http_body::Body for CountedBody {
    type Data = Bytes;
    type Error = std::convert::Infallible;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Bytes>, Self::Error>>> {
        let next = self.frames.pop_front();
        if next.is_some() {
            self.pulled
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        std::task::Poll::Ready(next.map(|b| Ok(http_body::Frame::data(b))))
    }

    fn is_end_stream(&self) -> bool {
        self.frames.is_empty()
    }
}

/// What a replica built before chained writes decodes of a chained call's
/// first frame: `chained_header` is field 3 of the `message_type` oneof, an
/// unknown field to it, so it reads a message with neither arm set (pinned on
/// the wire by `codegen::proto_types`). Its refusal of that frame is this
/// service's unchanged first-frame check: `INVALID_ARGUMENT` "first message
/// has no header or data", with NO status details — the client's sign that
/// the replica does not chain — having read no chunk and reserved nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_header_read_as_a_pre_chaining_replica_reads_it_is_refused_before_any_chunk() {
    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);
    let target = new_target(&tv, "chain-old-replica").await;

    // The chained frame, re-encoded as an old replica decoded it: the unknown
    // field 3 skipped, no arm set.
    let chained = chained_header(&target, "old-s", "old-p", true).encode_to_vec();
    assert_eq!(
        chained.first(),
        Some(&0x1a),
        "precondition: chained_header is field 3, length-delimited"
    );
    let as_old = UpdateLayerRequest { message_type: None };
    let pulled = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let body = CountedBody {
        frames: [
            grpc_frame(&as_old),
            grpc_frame(&make_update_data(field_ompf(1))),
            grpc_frame(&make_update_data(field_ompf(2))),
        ]
        .into_iter()
        .collect(),
        pulled: Arc::clone(&pulled),
    };
    let status = handle_update_layer(&svc, make_request(AttestedStreaming::new(body)))
        .await
        .expect_err("a first frame with neither arm must be refused");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert_eq!(status.message(), "first message has no header or data");
    assert_no_refusal_detail(&status);
    assert_eq!(
        pulled.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "refused from the first frame: no chunk read"
    );
    assert_eq!(
        svc.upload_admission.high_water_bytes(),
        0,
        "nothing reserved"
    );
    assert_eq!(seq(&tv, target.object_id).await, 1);
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 13 — INVALID_UPDATE on every INVALID_ARGUMENT producer
// ═════════════════════════════════════════════════════════════════════════════

/// One `INVALID_ARGUMENT` producer: the chained call's messages and, where
/// one exists, the unchained call refused by the same producer.
struct Producer {
    name: &'static str,
    chained: Vec<UpdateLayerRequest>,
    unchained: Option<Vec<UpdateLayerRequest>>,
}

/// `header` with `edit` applied to its `UpdateLayerHeader`.
fn edited(
    mut request: UpdateLayerRequest,
    edit: impl Fn(&mut UpdateLayerHeader),
) -> UpdateLayerRequest {
    match request.message_type.as_mut() {
        Some(update_layer_request::MessageType::Header(h)) => edit(h),
        Some(update_layer_request::MessageType::ChainedHeader(c)) => edit(
            c.header
                .as_mut()
                .expect("the chained header carries its header"),
        ),
        _ => panic!("fixture: not a header"),
    }
    request
}

/// Every `INVALID_ARGUMENT` a chained call can meet that carries no reason of
/// its own, end to end through the handler: the §B chain rules, the header
/// rules, the chunk count, the size cap, a second header, malformed OODF, a
/// per-delta check and planning on the call's first attempt. Each chained
/// refusal carries `INVALID_UPDATE` (and keeps the producer's own message);
/// the unchained call refused by the same producer carries no details at all.
/// Nothing applies. The size refusals, which carry their own reason, are the
/// next test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_invalid_argument_a_chained_call_meets_carries_invalid_update() {
    let tv = TestValkey::setup().await;
    let svc = LiveEditService {
        max_layer_size: SMALL_SIZE_CAP,
        ..chain_service(&tv)
    };
    let t = new_target(&tv, "chain-invalid-update").await;
    let chained_head = |op: &str| chained_header(&t, op, "", false);
    let plain_head = |op: &str| plain_header_from(WRITER, &t, op);
    let tiny = || make_update_data(vec![0u8]);
    let over_count = chunk_count_cap_for_test() + 1;
    let self_parenting = sections_ompf(&[(7, 7)]);

    let producers = vec![
        Producer {
            name: "chained_header without its header",
            chained: vec![UpdateLayerRequest {
                message_type: Some(update_layer_request::MessageType::ChainedHeader(
                    ChainedUpdateLayerHeader {
                        header: None,
                        previous_operation_id: String::new(),
                        no_wait: false,
                    },
                )),
            }],
            unchained: None,
        },
        Producer {
            name: "an empty header.operation_id",
            chained: vec![chained_header(&t, "", "", false)],
            unchained: None,
        },
        Producer {
            name: "a previous_operation_id outside the charset",
            chained: vec![chained_header(&t, "iu-1", "op:1", false)],
            unchained: None,
        },
        Producer {
            name: "an over-long previous_operation_id",
            chained: vec![chained_header(&t, "iu-1", &"a".repeat(65), false)],
            unchained: None,
        },
        Producer {
            name: "previous_operation_id equal to operation_id",
            chained: vec![chained_header(&t, "iu-1", "iu-1", false)],
            unchained: None,
        },
        Producer {
            name: "an operation_id outside the charset",
            chained: vec![chained_head("op 1")],
            unchained: Some(vec![plain_head("op 1")]),
        },
        Producer {
            name: "an operation_id without a client_id",
            chained: vec![edited(chained_head("iu-1"), |h| h.client_id.clear())],
            unchained: Some(vec![edited(plain_head("iu-1"), |h| h.client_id.clear())]),
        },
        Producer {
            name: "a zero object_id",
            chained: vec![edited(chained_head("iu-1"), |h| h.object_id = 0)],
            unchained: Some(vec![edited(plain_head("iu-1"), |h| h.object_id = 0)]),
        },
        Producer {
            name: "too many chunks",
            chained: std::iter::once(chained_head("iu-1"))
                .chain((0..over_count).map(|_| tiny()))
                .collect(),
            unchained: Some(
                std::iter::once(plain_head("iu-1"))
                    .chain((0..over_count).map(|_| tiny()))
                    .collect(),
            ),
        },
        Producer {
            name: "content over the size cap",
            chained: vec![
                chained_head("iu-1"),
                make_update_data(vec![0u8; SMALL_SIZE_CAP + 1]),
            ],
            unchained: Some(vec![
                plain_head("iu-1"),
                make_update_data(vec![0u8; SMALL_SIZE_CAP + 1]),
            ]),
        },
        Producer {
            name: "a second header",
            chained: vec![chained_head("iu-1"), tiny(), chained_head("iu-1")],
            unchained: Some(vec![plain_head("iu-1"), tiny(), plain_head("iu-1")]),
        },
        Producer {
            name: "content that is not OMPF",
            chained: vec![chained_head("iu-1"), make_update_data(b"not-ompf".to_vec())],
            unchained: Some(vec![
                plain_head("iu-1"),
                make_update_data(b"not-ompf".to_vec()),
            ]),
        },
        Producer {
            name: "a per-delta check (self-parenting)",
            chained: vec![
                chained_head("iu-1"),
                make_update_data(self_parenting.clone()),
            ],
            unchained: Some(vec![plain_head("iu-1"), make_update_data(self_parenting)]),
        },
        Producer {
            name: "planning on the first attempt (DuplicateSection)",
            chained: vec![
                chained_head("iu-1"),
                make_update_data(duplicate_section_ompf()),
            ],
            unchained: Some(vec![
                plain_head("iu-1"),
                make_update_data(duplicate_section_ompf()),
            ]),
        },
    ];

    for producer in producers {
        let what = producer.name;
        let chained_status = update(&svc, producer.chained)
            .await
            .expect_err("every producer must refuse the chained call");
        assert_invalid_update(&chained_status, what);
        if let Some(unchained) = producer.unchained {
            let plain_status = update(&svc, unchained)
                .await
                .expect_err("every producer must refuse the unchained call");
            assert_eq!(
                plain_status.code(),
                Code::InvalidArgument,
                "{what}: {plain_status:?}"
            );
            assert_no_refusal_detail(&plain_status);
            assert_eq!(
                chained_status.message(),
                plain_status.message(),
                "{what}: the choke point keeps the producer's message"
            );
        }
    }
    assert_eq!(seq(&tv, t.object_id).await, 1, "nothing applied");
}

/// The chunk-count cap of the size-capped service (`SMALL_SIZE_CAP`).
fn chunk_count_cap_for_test() -> usize {
    live_edit::grpc::streaming::chunk_count_cap(SMALL_SIZE_CAP)
}

/// `delta`'s OMPF body as `UpdateLayer` chunks of 1 MiB: well under tonic's
/// 4 MiB message limit.
fn update_chunks(delta: &OwnedDelta) -> Vec<UpdateLayerRequest> {
    delta_to_ompf_bytes(delta, &[])
        .chunks(1 << 20)
        .map(|chunk| make_update_data(chunk.to_vec()))
        .collect()
}

/// 60,000 created sections and 60,000 inline set fields: each kind under the
/// old per-kind limit of 100,000, their sum over the 99,999-entry limit.
fn over_the_entry_limit() -> OwnedDelta {
    OwnedDelta {
        struct_commands: (1..=60_000u64)
            .map(|id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: id,
                    section_name: format!("s{id}"),
                    section_type: 1,
                })
            })
            .collect(),
        set_fields: (0..60_000)
            .map(|i| oodf_ffi::SetField {
                section_id: 1,
                key_name: format!("k{i}"),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            })
            .collect(),
        ..Default::default()
    }
}

/// One `ReorderChildren` of section 1 naming `n` ids, which the size checks
/// refuse before the delta is applied, so the ids need not exist.
fn reorder_of(n: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::ReorderChildren(
            oodf_ffi::CommandReorderChildren {
                section_id: 1,
                children_list_id: 0,
                children_list: (1..=n).collect(),
            },
        )],
        ..Default::default()
    }
}

/// Group 13's size cases (b.tbw): a chained call whose delta is refused for
/// its size carries the size reason — `DELTA_TOO_LARGE` with its counts, or
/// `CHILDREN_LIST_TOO_LONG` with the limit and the list's length, from the
/// domain bound (100,001 ids) or the parser's defensive cap (1,000,001) — in
/// place of `INVALID_UPDATE`, through the choke point unchanged; the
/// unchained call refused by the same producer carries the same refusal, so
/// a chained call loses nothing by being chained. A content error with no
/// reason of its own, on the same service, still carries `INVALID_UPDATE`.
/// Nothing applies.
///
/// Needs a service whose size cap holds several MB (the fixture's 10 MB),
/// unlike the producer table above, which needs a small cap for its own
/// size-cap case.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn size_refusals_a_chained_call_meets_carry_their_own_reason_in_place_of_invalid_update() {
    use crate::common::refusal_detail::{assert_children_list_too_long, assert_delta_too_large};

    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);
    let t = new_target(&tv, "chain-size-refusal").await;
    let with = |head: UpdateLayerRequest, delta: &OwnedDelta| {
        std::iter::once(head)
            .chain(update_chunks(delta))
            .collect::<Vec<_>>()
    };
    type Check = fn(&Status);
    let producers: Vec<(&str, OwnedDelta, Check)> = vec![
        (
            "a delta over the entry limit",
            over_the_entry_limit(),
            |s| assert_delta_too_large(s, 60_000, 60_000, 0),
        ),
        (
            "a children list one past the domain's limit",
            reorder_of(100_001),
            |s| assert_children_list_too_long(s, 100_001),
        ),
        (
            "a children list past the parser's defensive cap",
            reorder_of(1_000_001),
            |s| assert_children_list_too_long(s, 1_000_001),
        ),
    ];

    for (what, delta, check) in producers {
        let chained_status = update(&svc, with(chained_header(&t, "size-1", "", false), &delta))
            .await
            .expect_err("a size refusal must refuse the chained call");
        check(&chained_status);

        let plain_status = update(&svc, with(plain_header_from(WRITER, &t, "size-1"), &delta))
            .await
            .expect_err("a size refusal must refuse the unchained call");
        check(&plain_status);
        assert_eq!(
            chained_status.message(),
            plain_status.message(),
            "{what}: the choke point keeps the producer's message"
        );
    }

    let content_error = update(
        &svc,
        vec![
            chained_header(&t, "size-2", "", false),
            make_update_data(sections_ompf(&[(7, 7)])),
        ],
    )
    .await
    .expect_err("a self-parenting delta must be refused");
    assert_invalid_update(
        &content_error,
        "a content error with no reason of its own, on the same service",
    );
    assert_eq!(seq(&tv, t.object_id).await, 1, "nothing applied");
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 19 — shutdown in the gate
// ═════════════════════════════════════════════════════════════════════════════

/// Graceful shutdown's `begun` while a chained call waits: the call is
/// refused `UNAVAILABLE "server shutting down"` with the shutdown retry
/// hint, logged once at INFO at stage `chain`, outcome `shutdown`, its
/// reservation released and nothing applied. The handler's on-the-spot
/// observation of `begun` (the shutdown guard's `ALLOWED_OBSERVATIONS`
/// entry for `src/grpc/layer.rs`) is what ends it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_waiting_at_shutdown_is_refused_server_shutting_down_at_stage_chain() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(chain_service(&tv));
    let target = new_target(&tv, "chain-shutdown").await;

    let waiting = {
        let svc = Arc::clone(&svc);
        let messages = vec![
            chained_header(&target, "shutdown-s", "shutdown-p", false),
            make_update_data(field_ompf(1)),
        ];
        tokio::spawn(async move { capture(within(CAPPED_DEADLINE, update(&svc, messages))).await })
    };
    parked(deadline, &svc.metrics, 1, "the call parked").await;

    let held_open = begin_paced_ending_held_open(&svc.shutdown).await;
    let (result, captured) = settled(deadline, "the waiting call", waiting)
        .await
        .expect("the call's task must not panic");
    let status = result.expect_err("a call waiting at shutdown must be refused");
    assert_shutdown_status(&status);
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "UpdateLayer", "chain", "UNAVAILABLE");
    assert_field(event, "chained", true.into());
    assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_SHUTDOWN, 1)], "shutdown");
    assert_eq!(svc.upload_admission.reserved_bytes(), 0);
    assert_eq!(seq(&tv, target.object_id).await, 1, "nothing applied");
    drop(held_open);
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 20 — CHAIN_CHECK_UNAVAILABLE; the stalled store
// ═════════════════════════════════════════════════════════════════════════════

use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

/// Whether `args` is the gate probe's `HMGET <obj:hdr> seq superseded`.
fn is_gate_header_probe(args: &[Vec<u8>]) -> bool {
    args.first()
        .is_some_and(|name| name.eq_ignore_ascii_case(b"HMGET"))
        && args.iter().any(|a| a.as_slice() == b"superseded")
}

/// A proxy that answers every gate probe's header read with an error reply,
/// so no probe of any chained call can read the record.
async fn failing_probe_proxy() -> ScriptReplyProxy {
    let rule: ProxyRule = Arc::new(|args: &[Vec<u8>]| {
        is_gate_header_probe(args).then(|| {
            ProxyAction::Eval(error_reply_script(
                "ERR the test proxy refuses the chain probe",
            ))
        })
    });
    ScriptReplyProxy::start(test_valkey_addr(), rule).await
}

/// A service whose layer store reaches the store only through `proxy` (the
/// gate's probes among its commands); everything else as [`chain_service`].
async fn service_through(tv: &TestValkey, proxy: &ScriptReplyProxy) -> LiveEditService {
    let client = client_through(proxy).await;
    LiveEditService {
        session_store: Arc::new(SessionStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        )),
        ..service_over(tv, client, |store| store)
    }
}

/// Every probe failing: the call cannot learn whether its predecessor
/// committed. It keeps probing — with `no_wait` too: a failed probe only
/// counts, whatever `no_wait` (the deferred option of 2026-10-08) — until its
/// deadline cap, then is refused `UNAVAILABLE` `CHAIN_CHECK_UNAVAILABLE` with
/// a `RetryInfo` in the capacity range, logged at WARN at stage `chain`
/// (P7 amended: not client-caused), outcome `unavailable`, nothing applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_whose_every_probe_fails_is_refused_chain_check_unavailable_near_its_deadline()
{
    let tv = TestValkey::setup().await;
    let proxy = failing_probe_proxy().await;
    let svc = service_through(&tv, &proxy).await;
    let target = new_target(&tv, "chain-check-unavailable").await;

    for (no_wait, op) in [(false, "unread-s"), (true, "unread-no-wait")] {
        let what = if no_wait {
            "with no_wait"
        } else {
            "without no_wait"
        };
        let errors_before = probes(&svc.metrics, CHAIN_PROBE_ERROR);
        let (result, captured) = capture(within(
            PROBED_CAP_DEADLINE,
            chained(&svc, &target, op, "unread-p", no_wait, field_ompf(1)),
        ))
        .await;
        let status = result.expect_err("a call whose every probe failed must be refused");
        assert_refusal_detail(
            &status,
            refusal_reason::CHAIN_CHECK_UNAVAILABLE,
            &[],
            CAPACITY_RETRY_DELAY,
        );
        let event = captured.only(RPC_FAILURE_MSG);
        assert_common_fields(event, "WARN", "UpdateLayer", "chain", "UNAVAILABLE");
        assert!(
            probes(&svc.metrics, CHAIN_PROBE_ERROR) >= errors_before + 2,
            "{what}: a failed probe does not end the wait; the call probed again"
        );
    }
    assert_eq!(probes(&svc.metrics, CHAIN_PROBE_OK), 0);
    assert!(
        proxy.hits() >= 4,
        "vacuity guard: the proxy refused the probes"
    );
    assert_gate_exits(
        &svc.metrics,
        &[(CHAIN_OUTCOME_UNAVAILABLE, 2)],
        "every probe failed",
    );
    assert_eq!(seq(&tv, target.object_id).await, 1, "nothing applied");
}

// ═════════════════════════════════════════════════════════════════════════════
// The gate's deadline cap and probe, driven directly
// ═════════════════════════════════════════════════════════════════════════════

/// The deadline budget `T` of the gate tests driven directly: 10 s, so their
/// margin `M` is 1 s ([`gate_margin`]). Each test sets the call's deadline `D`
/// itself, as if the call had already spent part of `T` on its way to the
/// gate.
const GATE_BUDGET: Duration = Duration::from_secs(10);

/// The furthest deadline a gate test driven directly waits towards: 1.5 s, so
/// its first probe (at `D − 1.5 s`) comes before the margin and the call is a
/// waiter, not a late arrival.
const GATE_WAITER_DEADLINE: Duration = Duration::from_millis(1_500);

/// The deadline budget of the late-arrival tests that stage `D` 9 s away:
/// 100 s, so their margin `M` is 10 s, the call is a late arrival, and its
/// final probe — and, in the re-probe test, the window it stages between its
/// two probes — is seconds wide.
const LATE_REPROBE_BUDGET: Duration = Duration::from_secs(100);

/// [`GATE_BUDGET`]'s margin `M`: 1 s.
fn gate_margin() -> Duration {
    chain_deadline_margin(GATE_BUDGET)
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// A fresh registry for one gate's exits and probes.
fn gate_metrics() -> Arc<Metrics> {
    Arc::new(crate::common::metrics_scrape_only("test"))
}

/// Enter the gate directly, as the handler does at its step 5b, for
/// `operation_id` after `previous_operation_id` under `client_id`, with the
/// deadline `deadline` and the budget [`GATE_BUDGET`], counting into
/// `metrics`.
fn enter_gate<'a>(
    store: &'a LayerStore,
    metrics: &Arc<Metrics>,
    target: &Target,
    client_id: &'a str,
    operation_id: &'a str,
    previous_operation_id: &'a str,
    deadline: Instant,
) -> ChainGate<'a> {
    enter_gate_with_budget(
        store,
        metrics,
        target,
        client_id,
        operation_id,
        previous_operation_id,
        deadline,
        GATE_BUDGET,
    )
}

/// [`enter_gate`] with the deadline budget `budget` in place of
/// [`GATE_BUDGET`].
#[allow(clippy::too_many_arguments)]
fn enter_gate_with_budget<'a>(
    store: &'a LayerStore,
    metrics: &Arc<Metrics>,
    target: &Target,
    client_id: &'a str,
    operation_id: &'a str,
    previous_operation_id: &'a str,
    deadline: Instant,
    budget: Duration,
) -> ChainGate<'a> {
    ChainGate::enter(
        store,
        Some(Arc::clone(metrics)),
        target.object_id,
        client_id,
        ChainLink {
            operation_id,
            previous_operation_id,
            no_wait: false,
        },
        deadline,
        budget,
        1,
    )
}

/// `(ok, error)` of `live_edit_update_layer_chain_probes_total`.
fn probe_counts(metrics: &Metrics) -> (u64, u64) {
    (
        probes(metrics, CHAIN_PROBE_OK),
        probes(metrics, CHAIN_PROBE_ERROR),
    )
}

/// The status of a gate refusal.
#[track_caller]
fn refused(exit: GateExit, what: &str) -> Status {
    match exit {
        GateExit::Refused(error) => error.into(),
        other => panic!("{what}: the gate must refuse the call; got {other:?}"),
    }
}

/// Group 27 (F8): a call that reaches the gate with no more than its margin
/// left is checked once and does not wait. Driven directly against the store
/// with `T` = [`LATE_REPROBE_BUDGET`] (`M` = 10 s) and `D` 9 s away: inside
/// the margin, yet room for hundreds of probe sleeps (each at most 20 ms) —
/// which the gate took when its cap was a store round trip (`2·rtt99`)
/// before the deadline, probing many times before it answered. The deadline
/// is staged far ahead so the one probe has seconds to answer however loaded
/// the runner (test-writing guide, § Performance-Sensitive Tests), as in the
/// re-probe test below.
///
/// - A pending predecessor: exactly one probe, read, then `ABORTED`
///   `PREDECESSOR_NOT_COMMITTED` naming the record it read, outcome
///   `timeout` — never `CHAIN_CHECK_UNAVAILABLE` on a healthy store, since
///   the probe has half of the 9 s (4.5 s) to answer.
/// - A chain's first write over no record: `ready` from that one probe.
/// - No time left at all (`D` = now, `T` = [`GATE_BUDGET`]): still exactly
///   one probe, answered `timeout` or `unavailable` (the probe's limit is its
///   own start, so it is usually given up before it reads the record).
///
/// Counter-based: the probe count is the claim, not the time taken.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn late_arrival_with_no_more_than_its_margin_left_is_checked_once_and_does_not_wait() {
    let tv = TestValkey::setup().await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let svc = chain_service(&tv);
    let target = new_target(&tv, "gate-late-arrival").await;
    let left = Duration::from_secs(9);
    assert!(
        left < chain_deadline_margin(LATE_REPROBE_BUDGET),
        "precondition: 9 s is inside the 10 s margin"
    );
    assert_applied(
        &plain_from(&svc, WRITER, &target, "late-0", field_ompf(0)).await,
        2,
        "the writer's latest write",
    );

    let metrics = gate_metrics();
    let exit = enter_gate_with_budget(
        &store,
        &metrics,
        &target,
        WRITER,
        "late-2",
        "late-1",
        Instant::now() + left,
        LATE_REPROBE_BUDGET,
    )
    .wait()
    .await;
    let status = refused(exit, "a pending predecessor");
    assert_predecessor_not_committed(&status, &last_operation("late-0"));
    assert_eq!(
        probe_counts(&metrics),
        (1, 0),
        "a pending predecessor: one probe, read — no sleep, no second probe"
    );
    assert_gate_exits(
        &metrics,
        &[(CHAIN_OUTCOME_TIMEOUT, 1)],
        "a late arrival, its predecessor pending",
    );

    let metrics = gate_metrics();
    let exit = enter_gate_with_budget(
        &store,
        &metrics,
        &target,
        "late-first-writer",
        "first-1",
        "",
        Instant::now() + left,
        LATE_REPROBE_BUDGET,
    )
    .wait()
    .await;
    assert!(
        matches!(exit, GateExit::Ready),
        "a chain's first write over no record is ready; got {exit:?}"
    );
    assert_eq!(
        record(&tv, target.object_id, "late-first-writer").await,
        None,
        "the first writer has no record on the object, which is what its one probe read"
    );
    assert_eq!(
        probe_counts(&metrics),
        (1, 0),
        "a chain's first write: one probe"
    );
    assert_gate_exits(
        &metrics,
        &[(CHAIN_OUTCOME_READY, 1)],
        "a late arrival, ready",
    );

    let metrics = gate_metrics();
    let exit = enter_gate(
        &store,
        &metrics,
        &target,
        WRITER,
        "late-2",
        "late-1",
        Instant::now(),
    )
    .wait()
    .await;
    let status = refused(exit, "no time left");
    let outcome = if status.code() == Code::Aborted {
        assert_predecessor_not_committed(&status, &last_operation("late-0"));
        CHAIN_OUTCOME_TIMEOUT
    } else {
        assert_refusal_detail(
            &status,
            refusal_reason::CHAIN_CHECK_UNAVAILABLE,
            &[],
            CAPACITY_RETRY_DELAY,
        );
        CHAIN_OUTCOME_UNAVAILABLE
    };
    let (ok, error) = probe_counts(&metrics);
    assert_eq!(ok + error, 1, "no time left: still exactly one probe");
    assert_gate_exits(&metrics, &[(outcome, 1)], "no time left");
    assert_eq!(
        seq(&tv, target.object_id).await,
        2,
        "the gate applies nothing"
    );
}

/// Part C (F8): a late arrival whose final probe reads `replaced`, and whose
/// layer then resolves to the same object (the handler calls
/// [`ChainGate::keep_waiting`], here at once), is probed a second time, and
/// answered from that probe — not from the first. The call's budget is
/// [`LATE_REPROBE_BUDGET`] (`M` = 10 s) and `D` is 9 s away: inside the
/// margin, so the call is a late arrival; and the second probe's limit,
/// `D − M/2`, is 4 s after entry, so the staging between the two probes (an
/// `HDEL` and a full handler commit) and the second probe itself have seconds
/// to finish, however loaded the runner (test-writing guide, § Performance-
/// Sensitive Tests: the deadline is staged far ahead).
///
/// - The object comes back (the `superseded` mark removed) and the
///   predecessor commits before the second probe: `ready`, on the
///   predecessor's record.
/// - The object stays replaced: the second probe reads `replaced` again, and a
///   call past its margin cannot wait — `ABORTED` `PREDECESSOR_NOT_COMMITTED`
///   naming the freshest record, outcome `timeout`, from the replaced-again
///   arm (review round 1, item 1, without `no_wait`). Without that arm the
///   gate would hand the call back to re-resolve again, which this test
///   observes as a second `replaced` exit.
///
/// Exactly two probes each, both read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn late_arrival_whose_final_probe_reads_replaced_is_probed_again_after_a_same_object_re_resolve(
) {
    let tv = TestValkey::setup().await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let svc = chain_service(&tv);
    let left = Duration::from_secs(9);
    assert!(
        left < chain_deadline_margin(LATE_REPROBE_BUDGET),
        "precondition: 9 s is inside the 10 s margin"
    );

    // The object comes back, and the predecessor commits.
    let target = new_target(&tv, "gate-late-replaced-ready").await;
    set_superseded(&tv, &target, true).await;
    let metrics = gate_metrics();
    let mut gate = enter_gate_with_budget(
        &store,
        &metrics,
        &target,
        WRITER,
        "back-s",
        "back-p",
        Instant::now() + left,
        LATE_REPROBE_BUDGET,
    );
    assert!(
        matches!(gate.wait().await, GateExit::Replaced),
        "the late arrival's one probe reads replaced"
    );
    set_superseded(&tv, &target, false).await;
    assert_applied(
        &plain_from(&svc, WRITER, &target, "back-p", field_ompf(1)).await,
        2,
        "the predecessor, while the call re-resolves",
    );
    gate.keep_waiting();
    let exit = gate.wait().await;
    assert!(
        matches!(exit, GateExit::Ready),
        "the second probe reads the predecessor's record: ready; got {exit:?}"
    );
    drop(gate);
    assert!(
        names(&record(&tv, target.object_id, WRITER).await, "back-p"),
        "ready on the predecessor's record, the writer's record on the object"
    );
    assert_eq!(
        probe_counts(&metrics),
        (2, 0),
        "probed again after the re-resolve"
    );
    assert_gate_exits(
        &metrics,
        &[(CHAIN_OUTCOME_READY, 1)],
        "a late arrival, ready after the re-resolve",
    );

    // The object stays replaced.
    let target = new_target(&tv, "gate-late-replaced-again").await;
    assert_applied(
        &plain_from(&svc, WRITER, &target, "again-0", field_ompf(0)).await,
        2,
        "the writer's latest write",
    );
    set_superseded(&tv, &target, true).await;
    let metrics = gate_metrics();
    let mut gate = enter_gate_with_budget(
        &store,
        &metrics,
        &target,
        WRITER,
        "again-2",
        "again-1",
        Instant::now() + left,
        LATE_REPROBE_BUDGET,
    );
    assert!(
        matches!(gate.wait().await, GateExit::Replaced),
        "the late arrival's one probe reads replaced"
    );
    gate.keep_waiting();
    let status = refused(gate.wait().await, "replaced again past the margin");
    drop(gate);
    assert_predecessor_not_committed(&status, &last_operation("again-0"));
    assert_eq!(
        probe_counts(&metrics),
        (2, 0),
        "probed again after the re-resolve, and answered from that probe"
    );
    assert_gate_exits(
        &metrics,
        &[(CHAIN_OUTCOME_TIMEOUT, 1)],
        "a late arrival, replaced again",
    );
}

/// Part C (F8): a waiter whose same-object re-resolve finishes after
/// `D − M/2` is refused at once: its re-entered final probe's limit
/// (`D − M/2`) has already passed, so that probe is given up the moment it is
/// sent (counted `error`), and the call is answered from the freshest record
/// it read — `ABORTED` `PREDECESSOR_NOT_COMMITTED`, outcome `timeout`.
///
/// The waiter's first probe, 1.5 s before `D` (`M` = 1 s), reads the object
/// `superseded`; the test then plays a re-resolve that finishes at
/// `D − 0.4M`. Without the cap on the re-entered probe it would be given half
/// of what is left (to `D − 0.2M`) and read the record — two probes read,
/// none given up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_object_re_resolve_finishing_after_d_minus_half_the_margin_is_refused_at_once_naming_the_freshest_record(
) {
    let tv = TestValkey::setup().await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let svc = chain_service(&tv);
    let target = new_target(&tv, "gate-late-re-resolve").await;
    assert_applied(
        &plain_from(&svc, WRITER, &target, "slow-0", field_ompf(0)).await,
        2,
        "the writer's latest write",
    );
    set_superseded(&tv, &target, true).await;
    let metrics = gate_metrics();
    let d = Instant::now() + GATE_WAITER_DEADLINE;
    let m = gate_margin();
    let mut gate = enter_gate(&store, &metrics, &target, WRITER, "slow-2", "slow-1", d);
    assert!(
        matches!(gate.wait().await, GateExit::Replaced),
        "the waiter's first probe reads replaced"
    );
    assert_eq!(
        probe_counts(&metrics),
        (1, 0),
        "precondition: one probe, before the margin"
    );

    tokio::time::sleep_until(d - m.mul_f64(0.4)).await;
    gate.keep_waiting();
    let status = refused(gate.wait().await, "a re-resolve finishing after D − M/2");
    drop(gate);
    assert_predecessor_not_committed(&status, &last_operation("slow-0"));
    assert_eq!(
        probe_counts(&metrics),
        (1, 1),
        "the re-entered probe is given up at once, its limit D − M/2 passed; the refusal names \
         the record the first probe read"
    );
    assert_gate_exits(
        &metrics,
        &[(CHAIN_OUTCOME_TIMEOUT, 1)],
        "a re-resolve finishing late",
    );
}

/// Part C (F8): a waiter whose same-object re-resolve finishes before
/// `D − M/2` keeps its floor — its re-entered final probe is given up by
/// `D − M/2`, not halfway between its start and `D`.
///
/// The gate's store sits behind a proxy that passes the first probe and holds
/// the second's header read 350 ms. The waiter's first probe, 1.5 s before `D`
/// (`M` = 1 s), reads the object `superseded`; the test plays a re-resolve
/// that finishes at `D − 0.8M`, and the second probe starts then. Its answer
/// would arrive about `D − 0.45M`: after `D − M/2`, where the capped limit
/// gives it up (counted `error`, and the call refused from the freshest record
/// read, outcome `timeout`), but before `D − 0.4M`, the uncapped limit, where
/// it would be read. The proxy's hold is a minimum, so a slow run only makes
/// the answer later, never earlier.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn waiter_whose_same_object_re_resolve_finishes_before_d_minus_half_the_margin_keeps_its_floor(
) {
    let tv = TestValkey::setup().await;
    let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let rule: ProxyRule = {
        let seen = Arc::clone(&seen);
        Arc::new(move |args: &[Vec<u8>]| {
            if !is_gate_header_probe(args) {
                return None;
            }
            (seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1)
                .then_some(ProxyAction::Delay(ms(350)))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let store = LayerStore::new(client_through(&proxy).await, tv.key_schema().clone(), 4096);
    let svc = chain_service(&tv);
    let target = new_target(&tv, "gate-floor-kept").await;
    assert_applied(
        &plain_from(&svc, WRITER, &target, "floor-0", field_ompf(0)).await,
        2,
        "the writer's latest write",
    );
    set_superseded(&tv, &target, true).await;
    let metrics = gate_metrics();
    let d = Instant::now() + GATE_WAITER_DEADLINE;
    let m = gate_margin();
    let mut gate = enter_gate(&store, &metrics, &target, WRITER, "floor-2", "floor-1", d);
    assert!(
        matches!(gate.wait().await, GateExit::Replaced),
        "the waiter's first probe reads replaced"
    );
    assert_eq!(
        probe_counts(&metrics),
        (1, 0),
        "precondition: one probe, before the margin"
    );

    tokio::time::sleep_until(d - m.mul_f64(0.8)).await;
    gate.keep_waiting();
    let status = refused(gate.wait().await, "a held re-entered probe");
    drop(gate);
    assert_eq!(
        proxy.hits(),
        1,
        "vacuity guard: the proxy held the second probe"
    );
    assert_predecessor_not_committed(&status, &last_operation("floor-0"));
    assert_eq!(
        probe_counts(&metrics),
        (1, 1),
        "the re-entered probe, answered after D − M/2, is given up at D − M/2"
    );
    assert_gate_exits(&metrics, &[(CHAIN_OUTCOME_TIMEOUT, 1)], "the floor kept");
}

/// The probe's header read classifies `superseded` exactly: `1` — the only
/// value a `CreateLayer` overwrite writes, and the value the WATCH-path
/// guards test — reads as replaced; the field absent, or holding any other
/// value, reads as a live object (here pending, so a `no_wait` call is
/// refused); a header without `seq` reads as replaced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gate_probe_reads_superseded_one_as_replaced_and_any_other_value_or_none_as_live() {
    let tv = TestValkey::setup().await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let target = new_target(&tv, "gate-superseded").await;
    let header_key = tv.key_schema().obj_header(&target.object_id.to_string());
    let pool = &tv.client().pool;

    async fn probe_once(store: &LayerStore, target: &Target) -> GateExit {
        let mut gate = ChainGate::enter(
            store,
            None,
            target.object_id,
            WRITER,
            ChainLink {
                operation_id: "superseded-s",
                previous_operation_id: "superseded-p",
                no_wait: true,
            },
            Instant::now() + CAPPED_DEADLINE,
            CAPPED_DEADLINE,
            1,
        );
        gate.wait().await
    }

    assert!(
        matches!(probe_once(&store, &target).await, GateExit::Refused(_)),
        "no superseded field: a live object"
    );
    let _: i64 = pool
        .hset(&header_key, [("superseded", "0")])
        .await
        .expect("HSET superseded=0");
    assert!(
        matches!(probe_once(&store, &target).await, GateExit::Refused(_)),
        "superseded=0, a value nothing writes: a live object"
    );
    let _: i64 = pool
        .hset(&header_key, [("superseded", "1")])
        .await
        .expect("HSET superseded=1");
    assert!(
        matches!(probe_once(&store, &target).await, GateExit::Replaced),
        "superseded=1: replaced"
    );
    let _: i64 = pool
        .hdel(&header_key, vec!["superseded", "seq"])
        .await
        .expect("HDEL superseded seq");
    assert!(
        matches!(probe_once(&store, &target).await, GateExit::Replaced),
        "a header without seq: replaced (gone)"
    );
}

/// Which of the probe's two keys a [`wrong_typed_key_ends_the_wait_at_once`]
/// case retypes.
#[derive(Clone, Copy, Debug)]
enum Retyped {
    /// `obj:wtok`: the probe's `HGET` fails, so it reads no record.
    RecordKey,
    /// `obj:hdr`, the writer's record in place: the probe's `HGET` reads the
    /// record and its `HMGET` fails.
    Header,
}

/// Review round 1, item 1 — the gate's wrong-type exit. A chained call whose
/// probe meets a key holding a foreign type is let through at once: one
/// probe, counted `error`, outcome `ready` — the commit refuses the keyspace
/// before it compares anything — and its commit refuses it as a keyspace
/// type error, a server fault answered `INTERNAL` with the detail on the
/// ERROR line only: the commit's own `WRONGTYPE_KEY` for the record key, the
/// store's `WRONGTYPE` for the header, which the commit reads first. Without
/// that exit the call would count each failed probe and wait to its cap (here
/// [`SHORT_DEADLINE`]'s), then be refused `CHAIN_CHECK_UNAVAILABLE`: a retry
/// hint for a keyspace no retry can fix. Nothing applied.
async fn wrong_typed_key_ends_the_wait_at_once(retyped: Retyped, label: &str) {
    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);
    let target = new_target(&tv, label).await;
    let oid = target.object_id.to_string();
    let key = match retyped {
        Retyped::RecordKey => tv.key_schema().obj_write_token(&oid),
        Retyped::Header => {
            assert_applied(
                &plain_from(&svc, WRITER, &target, "wt-0", field_ompf(0)).await,
                2,
                "the writer's latest write, which the probe reads",
            );
            tv.key_schema().obj_header(&oid)
        }
    };
    let _: () = tv
        .client()
        .pool
        .set(&key, "not-a-hash", None, None, false)
        .await
        .expect("retyping the key must succeed");

    let (result, captured) = capture(within(
        SHORT_DEADLINE,
        chained(&svc, &target, "wt-2", "wt-1", false, field_ompf(2)),
    ))
    .await;
    let status = result.expect_err("a wrong-typed key must fail the commit");
    assert_eq!(
        (status.code(), status.message()),
        (Code::Internal, INTERNAL_ERROR_MSG),
        "{retyped:?}: the commit's keyspace refusal, not the gate's; got {status:?}"
    );
    let event = captured.only(INTERNAL_ERROR_MSG);
    assert_field(event, "stage", "storage".into());
    let detail = event["fields"]["error"].as_str().unwrap_or_default();
    match retyped {
        Retyped::RecordKey => assert!(
            detail.contains("WRONGTYPE_KEY") && detail.contains(key.as_str()),
            "the commit's type check refuses the record key WRONGTYPE_KEY; got {event}"
        ),
        Retyped::Header => assert!(
            detail.contains("WRONGTYPE"),
            "the commit's first header read fails the store's WRONGTYPE; got {event}"
        ),
    }
    assert_eq!(
        probe_counts(&svc.metrics),
        (0, 1),
        "{retyped:?}: one probe, counted error — the wait ends at once"
    );
    assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_READY, 1)], label);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_meeting_a_wrong_typed_record_key_leaves_the_gate_ready_and_is_refused_wrongtype_key(
) {
    wrong_typed_key_ends_the_wait_at_once(Retyped::RecordKey, "chain-wrongtype-record").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_meeting_a_wrong_typed_header_leaves_the_gate_ready_and_is_refused_by_the_commit(
) {
    wrong_typed_key_ends_the_wait_at_once(Retyped::Header, "chain-wrongtype-header").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 22 — seeding
// ═════════════════════════════════════════════════════════════════════════════

/// A fresh registry renders both probe cells and both gauges at zero, so a
/// probe-error rate or a waiting-bytes ratio has a series from the first
/// scrape; the two histograms are NOT pre-registered — no
/// `chain_wait_seconds` cell for any outcome and no detection-lag series
/// until each is first observed (as SDD §17.1 states).
#[test]
fn chain_metrics_seed_the_probe_and_gauge_cells_but_not_the_histograms() {
    use live_edit::observability::metrics::test_util::has_metric;

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);
    for result in live_edit::observability::metrics::CHAIN_PROBE_RESULTS {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_update_layer_chain_probes_total",
                &[("result", result)]
            ),
            Some(0.0),
            "probes_total{{result={result}}} is seeded at zero"
        );
    }
    for gauge in [
        "live_edit_update_layer_chain_waiters",
        "live_edit_update_layer_chain_waiting_bytes",
    ] {
        assert_eq!(
            sample_value(&text, gauge, &[]),
            Some(0.0),
            "{gauge} is seeded at zero"
        );
    }
    for outcome in CHAIN_OUTCOMES {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_update_layer_chain_wait_seconds_count",
                &[("outcome", outcome)]
            ),
            None,
            "chain_wait_seconds{{outcome={outcome}}} is not pre-registered"
        );
    }
    assert!(
        !has_metric(&text, "live_edit_update_layer_chain_wait_seconds")
            && !has_metric(&text, "live_edit_update_layer_chain_detect_lag_seconds"),
        "neither histogram has a series before its first observation"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Group 24 — the completion line
// ═════════════════════════════════════════════════════════════════════════════

/// A chained call's completion line carries `chained = true` — server-derived
/// from the arm, so recorded before any validator — and, once its link passed
/// validation, `previous_operation_id` and `no_wait`:
///
/// - a chained call refused by its link's validation: INFO at stage `header`,
///   `chained = true`, its validated `operation_id`, and neither
///   `previous_operation_id` nor `no_wait` (they were not validated);
/// - a predecessor refusal: INFO (P7 amended) at stage `chain`, `ABORTED`,
///   both link fields;
/// - an unchained call's line carries no `chained` field at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_completion_lines_carry_chained_and_their_link() {
    let tv = TestValkey::setup().await;
    let svc = chain_service(&tv);
    let target = new_target(&tv, "chain-log").await;

    let (result, captured) = capture(chained(
        &svc,
        &target,
        "log-s",
        "bad:previous",
        false,
        field_ompf(1),
    ))
    .await;
    assert_invalid_update(
        &result.expect_err("an invalid link is refused"),
        "invalid link",
    );
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "UpdateLayer", "header", "INVALID_ARGUMENT");
    assert_field(event, "chained", true.into());
    assert_field(event, "operation_id", "log-s".into());
    assert_field_absent(event, "previous_operation_id", "it failed its validation");
    assert_field_absent(event, "no_wait", "recorded only with a validated link");

    let (result, captured) = capture(chained(
        &svc,
        &target,
        "log-s",
        "log-p",
        true,
        field_ompf(1),
    ))
    .await;
    assert_predecessor_not_committed(
        &result.expect_err("a pending predecessor refuses a no_wait call"),
        &last_record(refusal_metadata::ABSENT),
    );
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "UpdateLayer", "chain", "ABORTED");
    assert_field(event, "chained", true.into());
    assert_field(event, "operation_id", "log-s".into());
    assert_field(event, "previous_operation_id", "log-p".into());
    assert_field(event, "no_wait", true.into());

    let (result, captured) = capture(update(
        &svc,
        vec![
            plain_header_from(WRITER, &target, "bad op"),
            make_update_data(field_ompf(1)),
        ],
    ))
    .await;
    assert_eq!(result.expect_err("refused").code(), Code::InvalidArgument);
    let event = captured.only(RPC_FAILURE_MSG);
    assert_field_absent(event, "chained", "an unchained call");
    assert_field_absent(event, "previous_operation_id", "an unchained call");
    assert_field_absent(event, "no_wait", "an unchained call");
}

// ═════════════════════════════════════════════════════════════════════════════
// End to end: a real server
// ═════════════════════════════════════════════════════════════════════════════

/// A real server wired as `main.rs` wires it: the production HTTP/2 settings,
/// `MetricsLayer` with its RPC deadline, and the LiveService built by
/// `transport::live_service_server` (the auth interceptor and the
/// end-of-stream record) around `svc`.
struct ChainServer {
    addr: std::net::SocketAddr,
    /// The metrics layer's registry, where `live_endpoint_calls_total` ticks.
    layer_metrics: Arc<Metrics>,
    task: tokio::task::JoinHandle<()>,
}

/// The metrics layer's `service` label on a [`ChainServer`].
const SERVER_LABEL: &str = "test";

/// The bearer every end-to-end call carries.
const BEARER: &str = "Bearer chained-writes-test";

impl ChainServer {
    async fn start(svc: LiveEditService, rpc_deadline_ms: u64, deadline: Instant) -> Self {
        use live_edit::grpc::metrics_layer::MetricsLayer;
        use live_edit::grpc::transport::{live_service_server, production_server_builder};

        let layer_metrics = Arc::new(crate::common::metrics_scrape_only(SERVER_LABEL));
        let metrics_layer = MetricsLayer::new(
            Arc::clone(&layer_metrics),
            SERVER_LABEL,
            rpc_deadline_ms,
            2_000,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a random port");
        let addr = listener.local_addr().expect("local addr");
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            let _ = ready_tx.send(());
            production_server_builder()
                .layer(metrics_layer)
                .add_service(live_service_server(svc))
                .serve_with_incoming(incoming)
                .await
                .expect("test server error");
        });
        settled(deadline, "server readiness", ready_rx)
            .await
            .expect("the server task dropped its ready channel");
        Self {
            addr,
            layer_metrics,
            task,
        }
    }

    /// `live_endpoint_calls_total{endpoint, grpc_code}` on this server.
    fn calls(&self, endpoint: &str, grpc_code: &str) -> u64 {
        sample_value(
            &render(&self.layer_metrics),
            "live_endpoint_calls_total",
            &[
                ("endpoint", endpoint),
                ("service", SERVER_LABEL),
                ("grpc_code", grpc_code),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    /// Every completion `endpoint` recorded, whatever its code.
    fn total_calls(&self, endpoint: &str) -> u64 {
        GRPC_CODE_SLUGS
            .iter()
            .map(|code| self.calls(endpoint, code))
            .sum()
    }

    async fn client(
        &self,
    ) -> live_edit::grpc::proto::live_service_client::LiveServiceClient<tonic::transport::Channel>
    {
        let channel = tonic::transport::Channel::from_shared(format!("http://{}", self.addr))
            .expect("valid URI")
            .connect()
            .await
            .expect("connect to the test server");
        live_edit::grpc::proto::live_service_client::LiveServiceClient::new(channel)
    }
}

impl Drop for ChainServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// `messages` as one `UpdateLayer` to `server` from a raw HTTP/2 client
/// carrying a `grpc-timeout` of `timeout`, and the status the server
/// answers.
///
/// Raw on purpose: this client arms no deadline timer of its own, so these
/// tests pin the server's answer independently of any client's timer — what
/// it answers, and that it answers before the deadline it computed from the
/// header, never `DEADLINE_EXCEEDED` or an abandoned call. A client that runs
/// its own timer, started before the server's clock (tonic arms one in
/// `GrpcTimeout::call`, before the request leaves), is covered by group 28
/// ([`tonic_update`] over a [`DelayedLink`]) and, for grpc-core, by the
/// conformance case (group 29).
async fn served_update(
    server: &ChainServer,
    messages: Vec<UpdateLayerRequest>,
    timeout: Duration,
    deadline: Instant,
) -> Status {
    let mut call = RawCall::open(server.addr, deadline, Some(timeout)).await;
    let last = messages.len() - 1;
    for (i, message) in messages.iter().enumerate() {
        call.send(message, i == last);
    }
    call.status(deadline).await
}

/// Assert the server recorded exactly one `UpdateLayer` completion, with
/// `grpc_code` — so not `DEADLINE_EXCEEDED` and not an abandoned `CANCELLED`.
async fn assert_one_completion(server: &ChainServer, deadline: Instant, grpc_code: &str) {
    until(deadline, "the server recorded the call", || {
        server.total_calls("UpdateLayer") > 0
    })
    .await;
    assert_eq!(
        (
            server.calls("UpdateLayer", grpc_code),
            server.total_calls("UpdateLayer")
        ),
        (1, 1),
        "one completion, {grpc_code}: never DEADLINE_EXCEEDED or an abandoned CANCELLED"
    );
}

/// Without a client cancel, a waiting call's `grpc-timeout` bounds it: the
/// gate refuses it `ABORTED` `PREDECESSOR_NOT_COMMITTED` naming the writer's
/// latest write (`last_record=operation`) before the deadline elapses, so the
/// client reads the refusal, not `DEADLINE_EXCEEDED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn waiting_chained_call_is_refused_aborted_before_its_grpc_timeout_end_to_end() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let target = new_target(&tv, "chain-grpc-timeout").await;
    let staging = chain_service(&tv);
    assert_applied(
        &chained(&staging, &target, "timeout-0", "", false, field_ompf(0)).await,
        2,
        "the chain's committed write",
    );
    let svc = chain_service(&tv);
    let gate_metrics = Arc::clone(&svc.metrics);
    let server = ChainServer::start(svc, 30_000, deadline).await;

    let status = served_update(
        &server,
        vec![
            chained_header(&target, "timeout-2", "timeout-1", false),
            make_update_data(field_ompf(2)),
        ],
        CLIENT_TIMEOUT,
        deadline,
    )
    .await;
    assert_predecessor_not_committed(&status, &last_operation("timeout-0"));
    assert_one_completion(&server, deadline, "ABORTED").await;
    assert_gate_exits(&gate_metrics, &[(CHAIN_OUTCOME_TIMEOUT, 1)], "grpc-timeout");
    assert_eq!(seq(&tv, target.object_id).await, 2, "nothing applied");
}

/// A proxy that holds every gate probe's header read past any deadline here,
/// after letting the first `pass` through.
async fn stalling_probe_proxy(pass: usize) -> ScriptReplyProxy {
    let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
        if !is_gate_header_probe(args) {
            return None;
        }
        (seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= pass)
            .then_some(ProxyAction::Delay(Duration::from_secs(60)))
    });
    ScriptReplyProxy::start(test_valkey_addr(), rule).await
}

/// A stalled store: every probe goes unanswered. Each probe has its own time
/// limit, so the call answers `UNAVAILABLE` `CHAIN_CHECK_UNAVAILABLE` before
/// its `grpc-timeout` — never `DEADLINE_EXCEEDED`, which would leave the
/// client unable to tell whether the call can still commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_store_ends_a_chained_call_chain_check_unavailable_before_its_deadline_end_to_end()
{
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let target = new_target(&tv, "chain-stalled").await;
    let proxy = stalling_probe_proxy(0).await;
    let svc = service_through(&tv, &proxy).await;
    let gate_metrics = Arc::clone(&svc.metrics);
    let server = ChainServer::start(svc, 30_000, deadline).await;

    let status = served_update(
        &server,
        vec![
            chained_header(&target, "stalled-s", "stalled-p", false),
            make_update_data(field_ompf(1)),
        ],
        CLIENT_TIMEOUT,
        deadline,
    )
    .await;
    assert_refusal_detail(
        &status,
        refusal_reason::CHAIN_CHECK_UNAVAILABLE,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    assert_one_completion(&server, deadline, "UNAVAILABLE").await;
    assert!(proxy.hits() >= 1, "vacuity guard: the proxy held a probe");
    assert_eq!(probes(&gate_metrics, CHAIN_PROBE_OK), 0);
    assert_gate_exits(
        &gate_metrics,
        &[(CHAIN_OUTCOME_UNAVAILABLE, 1)],
        "stalled store",
    );
}

/// The store stalls after one probe read the record: the call's later probes
/// are cut at their own time limits, and it answers from the freshest record
/// read — `ABORTED` `PREDECESSOR_NOT_COMMITTED`, `last_record=absent` —
/// before its `grpc-timeout`, never `DEADLINE_EXCEEDED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_store_after_one_read_ends_a_chained_call_aborted_before_its_deadline_end_to_end() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let target = new_target(&tv, "chain-stalled-after-one").await;
    let proxy = stalling_probe_proxy(1).await;
    let svc = service_through(&tv, &proxy).await;
    let gate_metrics = Arc::clone(&svc.metrics);
    let server = ChainServer::start(svc, 30_000, deadline).await;

    // Two probes are counted below (the one read, then one cut at its limit):
    // a multi-second `grpc-timeout`, so a call slowed on its way to the gate
    // is still a waiter, not a late arrival checked once.
    let status = served_update(
        &server,
        vec![
            chained_header(&target, "stalled-s", "stalled-p", false),
            make_update_data(field_ompf(1)),
        ],
        PROBED_CAP_DEADLINE,
        deadline,
    )
    .await;
    assert_predecessor_not_committed(&status, &last_record(refusal_metadata::ABSENT));
    assert_one_completion(&server, deadline, "ABORTED").await;
    assert_eq!(probes(&gate_metrics, CHAIN_PROBE_OK), 1);
    assert!(probes(&gate_metrics, CHAIN_PROBE_ERROR) >= 1);
    assert_gate_exits(
        &gate_metrics,
        &[(CHAIN_OUTCOME_TIMEOUT, 1)],
        "stalled after one read",
    );
}

// ── Group 28: a client that runs its own deadline timer ──────────────────────

/// The one-way latency [`DelayedLink`] adds in each direction in group 28:
/// 20 ms, a round trip of 40 ms — well above the old cap's `2·rtt99`
/// (≈ 2.8 ms) and well under half the [`TONIC_TIMEOUT`]'s margin (150 ms).
const LINK_DELAY: Duration = Duration::from_millis(20);

/// The deadline group 28's tonic client sets on its request: 3 s, so the
/// gate's margin is 300 ms.
const TONIC_TIMEOUT: Duration = Duration::from_secs(3);

/// A TCP proxy in front of `upstream` that holds every byte, in each
/// direction, for a fixed delay before passing it on, in order: a link whose
/// one-way latency is that delay, whatever crosses it. Aborting it drops
/// every connection it relays.
struct DelayedLink {
    addr: std::net::SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl DelayedLink {
    async fn start(upstream: std::net::SocketAddr, delay: Duration) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the delayed link");
        let addr = listener.local_addr().expect("the delayed link's address");
        let task = tokio::spawn(async move {
            let mut relays = tokio::task::JoinSet::new();
            while let Ok((client, _)) = listener.accept().await {
                relays.spawn(async move {
                    let Ok(server) = tokio::net::TcpStream::connect(upstream).await else {
                        return;
                    };
                    let (client_read, client_write) = client.into_split();
                    let (server_read, server_write) = server.into_split();
                    tokio::join!(
                        delayed_copy(client_read, server_write, delay),
                        delayed_copy(server_read, client_write, delay),
                    );
                });
            }
        });
        Self { addr, task }
    }
}

impl Drop for DelayedLink {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Copy `from` to `to`, each chunk written `delay` after it was read, in
/// order; the end of `from` reaches `to` (as a write shutdown) `delay` after
/// it too. Reading never waits on writing, so the delay is a latency, not a
/// rate limit.
async fn delayed_copy(
    mut from: tokio::net::tcp::OwnedReadHalf,
    mut to: tokio::net::tcp::OwnedWriteHalf,
    delay: Duration,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(Instant, Option<Vec<u8>>)>();
    let reader = async move {
        let mut buf = vec![0u8; 16 * 1024];
        loop {
            let chunk = match from.read(&mut buf).await {
                Ok(0) | Err(_) => None,
                Ok(n) => Some(buf[..n].to_vec()),
            };
            let end = chunk.is_none();
            if tx.send((Instant::now() + delay, chunk)).is_err() || end {
                return;
            }
        }
    };
    let writer = async move {
        while let Some((due, chunk)) = rx.recv().await {
            tokio::time::sleep_until(due).await;
            match chunk {
                Some(bytes) => {
                    if to.write_all(&bytes).await.is_err() {
                        return;
                    }
                }
                None => {
                    let _ = to.shutdown().await;
                    return;
                }
            }
        }
    };
    tokio::join!(reader, writer);
}

/// `messages` as one `UpdateLayer` from a tonic client connected to `addr`,
/// with `timeout` set on the request, and the status the client reads. tonic
/// sends `timeout` as the `grpc-timeout` header and arms its own timer from
/// it in `GrpcTimeout::call`, before the request leaves: its deadline starts
/// before the server's, and on expiry the client reads its own `CANCELLED`
/// "Timeout expired", whatever the server answers later.
async fn tonic_update(
    addr: std::net::SocketAddr,
    messages: Vec<UpdateLayerRequest>,
    timeout: Duration,
) -> Status {
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .expect("valid URI")
        .connect()
        .await
        .expect("connect through the delayed link");
    let mut client = live_edit::grpc::proto::live_service_client::LiveServiceClient::new(channel);
    let mut request = tonic::Request::new(tokio_stream::iter(messages));
    request.set_timeout(timeout);
    request
        .metadata_mut()
        .insert("authorization", BEARER.parse().expect("ASCII bearer"));
    client
        .update_layer(request)
        .await
        .expect_err("the chained call must be refused")
}

/// What the gate does with group 28's waiting call.
#[derive(Clone, Copy, Debug)]
enum TonicCase {
    /// The predecessor never commits: `ABORTED` at the cap.
    Pending,
    /// Every probe stalls: `CHAIN_CHECK_UNAVAILABLE`.
    StalledStore,
}

/// Group 28 (F8), end to end: a tonic client with its own deadline timer
/// ([`TONIC_TIMEOUT`], margin 300 ms), over a [`DelayedLink`] adding
/// [`LINK_DELAY`] each way, to the full server stack ([`ChainServer`]). Its
/// chained call's predecessor never commits. The client's timer starts at
/// least one link delay before the server's clock, and the answer takes one
/// more to come back; the gate's margin leaves room for both, so:
///
/// - the client reads the gate's refusal — `ABORTED` `PREDECESSOR_NOT_COMMITTED`
///   (`last_record=operation`), or with every probe stalled
///   `CHAIN_CHECK_UNAVAILABLE` with its `RetryInfo` — never its own
///   `CANCELLED` "Timeout expired" or a `DEADLINE_EXCEEDED`;
/// - the server records exactly that one completion, and no abandoned line;
/// - nothing is applied.
///
/// Under the old cap (an answer `2·rtt99` before the server's deadline) the
/// answer reached this client about 40 ms after its own timer fired; here it
/// arrives with more than 100 ms to spare. Current-thread so the
/// [`ThreadCapture`] sees the server's lines.
async fn chained_call_answer_reaches_a_tonic_client_with_its_own_timer(
    case: TonicCase,
    label: &str,
) {
    let deadline = test_deadline();
    let capture = ThreadCapture::start();
    let tv = TestValkey::setup().await;
    let target = new_target(&tv, label).await;
    assert_applied(
        &chained(
            &chain_service(&tv),
            &target,
            "tonic-0",
            "",
            false,
            field_ompf(0),
        )
        .await,
        2,
        "the chain's committed write",
    );
    let proxy = match case {
        TonicCase::Pending => None,
        TonicCase::StalledStore => Some(stalling_probe_proxy(0).await),
    };
    let svc = match &proxy {
        None => chain_service(&tv),
        Some(proxy) => service_through(&tv, proxy).await,
    };
    let gate_metrics = Arc::clone(&svc.metrics);
    let server = ChainServer::start(svc, 30_000, deadline).await;
    let link = DelayedLink::start(server.addr, LINK_DELAY).await;
    assert!(
        LINK_DELAY * 2 < chain_deadline_margin(TONIC_TIMEOUT) / 2,
        "precondition: the round trip is under half the margin"
    );

    let status = settled(
        deadline,
        "the tonic call",
        tonic_update(
            link.addr,
            vec![
                chained_header(&target, "tonic-2", "tonic-1", false),
                make_update_data(field_ompf(2)),
            ],
            TONIC_TIMEOUT,
        ),
    )
    .await;
    let (grpc_code, outcome) = match case {
        TonicCase::Pending => {
            assert_predecessor_not_committed(&status, &last_operation("tonic-0"));
            ("ABORTED", CHAIN_OUTCOME_TIMEOUT)
        }
        TonicCase::StalledStore => {
            assert_refusal_detail(
                &status,
                refusal_reason::CHAIN_CHECK_UNAVAILABLE,
                &[],
                CAPACITY_RETRY_DELAY,
            );
            assert!(
                proxy.as_ref().is_some_and(|p| p.hits() >= 1),
                "vacuity guard: the proxy held a probe"
            );
            ("UNAVAILABLE", CHAIN_OUTCOME_UNAVAILABLE)
        }
    };
    assert_one_completion(&server, deadline, grpc_code).await;
    let captured = capture.snapshot();
    assert!(
        captured.with_message(RPC_ABANDONED_MSG).is_empty(),
        "{case:?}: the server answered the call; it did not abandon it; full capture:\n{}",
        captured.text
    );
    let event = captured.only(RPC_FAILURE_MSG);
    assert_field(event, "grpc_code", grpc_code.into());
    assert_field(event, "stage", "chain".into());
    assert_gate_exits(&gate_metrics, &[(outcome, 1)], label);
    assert_eq!(seq(&tv, target.object_id).await, 2, "nothing applied");

    drop(link);
    drop(server);
    drop(proxy);
    drop(capture);
    purge_keys(&tv).await;
}

#[tokio::test]
async fn waiting_chained_call_is_refused_before_its_deadline_reaching_a_tonic_client_with_its_own_timer_over_a_delayed_link(
) {
    chained_call_answer_reaches_a_tonic_client_with_its_own_timer(
        TonicCase::Pending,
        "chain-tonic-timer",
    )
    .await;
}

#[tokio::test]
async fn stalled_store_ends_a_chained_call_chain_check_unavailable_reaching_a_tonic_client_with_its_own_timer_over_a_delayed_link(
) {
    chained_call_answer_reaches_a_tonic_client_with_its_own_timer(
        TonicCase::StalledStore,
        "chain-tonic-stalled",
    )
    .await;
}

/// Group 23: `Ping` through the full service stack reports this build's two
/// capability bits, `operation_ids` and `chained_writes`, both true.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ping_reports_both_capability_bits_end_to_end() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let server = ChainServer::start(chain_service(&tv), 30_000, deadline).await;
    let mut request = tonic::Request::new(live_edit::grpc::proto::PingRequest {
        client_timestamp_microseconds_since_epoch: 7,
    });
    request
        .metadata_mut()
        .insert("authorization", BEARER.parse().expect("ASCII bearer"));
    let reply = server
        .client()
        .await
        .ping(request)
        .await
        .expect("Ping must succeed")
        .into_inner();
    assert_eq!(
        (
            reply.client_timestamp_microseconds_since_epoch,
            reply.operation_ids,
            reply.chained_writes
        ),
        (7, true, true)
    );
}

// ── Rider 1: a parked call cancelled on the wire ─────────────────────────────

/// One `UpdateLayer` call from a raw HTTP/2 client: the frames, resets and
/// connection losses a real client produces.
struct RawCall {
    _send_request: h2::client::SendRequest<Bytes>,
    response: Option<h2::client::ResponseFuture>,
    body: h2::SendStream<Bytes>,
    driver: tokio::task::JoinHandle<()>,
}

impl RawCall {
    /// Open a connection and send the head of an `UpdateLayer` call, with a
    /// `grpc-timeout` header when `grpc_timeout` is set (no client-side timer
    /// is armed: nothing here enforces it).
    async fn open(
        addr: std::net::SocketAddr,
        deadline: Instant,
        grpc_timeout: Option<Duration>,
    ) -> Self {
        let tcp = tokio::net::TcpStream::connect(addr)
            .await
            .expect("TCP connect");
        let (send_request, connection) =
            settled(deadline, "the h2 handshake", h2::client::handshake(tcp))
                .await
                .expect("h2 handshake");
        let driver = tokio::spawn(async move {
            let _ = connection.await;
        });
        let mut send_request = settled(deadline, "the connection ready", send_request.ready())
            .await
            .expect("the connection accepts a stream");
        let mut head = http::Request::builder()
            .method("POST")
            .uri(format!(
                "http://{addr}/nvidia.omniverse.live.v1alpha.LiveService/UpdateLayer"
            ))
            .header("content-type", "application/grpc")
            .header("te", "trailers")
            .header("authorization", BEARER);
        if let Some(timeout) = grpc_timeout {
            head = head.header("grpc-timeout", format!("{}m", timeout.as_millis()));
        }
        let head = head.body(()).expect("valid head");
        let (response, body) = send_request
            .send_request(head, false)
            .expect("UpdateLayer head");
        Self {
            _send_request: send_request,
            response: Some(response),
            body,
            driver,
        }
    }

    /// The status the server answered the call with: from a trailers-only
    /// response's head, or from the trailers after the body.
    async fn status(&mut self, deadline: Instant) -> Status {
        let response = self.response.take().expect("taken once");
        let response = settled(deadline, "the response head", response)
            .await
            .expect("the server answers");
        if let Some(status) = Status::from_header_map(response.headers()) {
            return status;
        }
        let mut body = response.into_body();
        while let Some(chunk) = settled(deadline, "the response body", body.data()).await {
            let chunk = chunk.expect("the response body");
            let _ = body.flow_control().release_capacity(chunk.len());
        }
        let trailers = settled(deadline, "the response trailers", body.trailers())
            .await
            .expect("the response trailers")
            .expect("a gRPC response ends with trailers");
        Status::from_header_map(&trailers).expect("the trailers carry grpc-status")
    }

    /// Send `msg` as one DATA frame, with END_STREAM if `end_of_stream`.
    fn send(&mut self, msg: &UpdateLayerRequest, end_of_stream: bool) {
        self.body
            .send_data(grpc_frame(msg), end_of_stream)
            .expect("send a DATA frame");
    }

    /// Close the TCP connection, without a GOAWAY or a reset.
    async fn close(&mut self, deadline: Instant) {
        self.driver.abort();
        let stopped = settled(deadline, "the connection driver stopped", &mut self.driver).await;
        assert!(
            stopped.is_err_and(|e| e.is_cancelled()),
            "the connection must still have been open"
        );
    }
}

impl Drop for RawCall {
    fn drop(&mut self) {
        self.driver.abort();
    }
}

/// How the client ends a call parked at the gate.
#[derive(Clone, Copy, Debug)]
enum WireEnding {
    /// `RST_STREAM(CANCEL)` after the request's END_STREAM.
    ResetAfterEndStream,
    /// The TCP connection closes.
    TcpClose,
}

/// Rider 1 end to end: a chained call whose content is complete (END_STREAM
/// sent) parks at the gate of a full `LiveEditService` behind a predecessor
/// that has not committed; the client then ends it on the wire. hyper drops
/// the handler at the gate's await, which spawns nothing, so:
///
/// - the metrics layer's drop guard logs one `RPC abandoned before
///   completion` line and ticks `CANCELLED` once, and no `RPC failed` line;
/// - the gate's drop guard observes `cancelled` and returns both gauges, and
///   the reservation's bytes return;
/// - the header `seq` and the stream are unchanged;
/// - when the predecessor later commits, the cancelled call does not follow
///   it: its own `no_wait` re-send then applies (not `ALREADY_COMMITTED`),
///   and the stream holds exactly the predecessor and that re-send.
///
/// Current-thread so the [`ThreadCapture`] sees the line the server's
/// connection task emits.
async fn parked_call_ended_on_the_wire_applies_nothing(ending: WireEnding, label: &str) {
    let deadline = test_deadline();
    let capture = ThreadCapture::start();
    let tv = TestValkey::setup().await;
    let target = new_target(&tv, label).await;
    let svc = chain_service(&tv);
    let gate_metrics = Arc::clone(&svc.metrics);
    let gate = Arc::clone(&svc.upload_admission);
    let server = ChainServer::start(svc, 50_000, deadline).await;

    let payload = field_ompf(2);
    let mut call = RawCall::open(server.addr, deadline, None).await;
    call.send(&chained_header(&target, "wire-s", "wire-p", false), false);
    call.send(&make_update_data(payload.clone()), true);
    parked(deadline, &gate_metrics, 1, "the call parked at the gate").await;
    assert_eq!(
        waiting_bytes(&gate_metrics),
        FIXED_CHARGE_BYTES + payload.len() as u64
    );

    match ending {
        WireEnding::ResetAfterEndStream => call.body.send_reset(h2::Reason::CANCEL),
        WireEnding::TcpClose => call.close(deadline).await,
    }
    until(deadline, "the handler dropped at the gate", || {
        waiters(&gate_metrics) == 0
    })
    .await;
    until(deadline, "the server recorded the call", || {
        server.total_calls("UpdateLayer") > 0
    })
    .await;
    assert_eq!(
        (
            server.calls("UpdateLayer", "CANCELLED"),
            server.total_calls("UpdateLayer")
        ),
        (1, 1),
        "{ending:?}: one CANCELLED tick for the dropped call"
    );
    let captured = capture.snapshot();
    let abandoned = captured.only(RPC_ABANDONED_MSG);
    assert_eq!(abandoned["level"], "INFO", "{abandoned}");
    assert_field(abandoned, "endpoint", "UpdateLayer".into());
    assert_field(abandoned, "grpc_code", "CANCELLED".into());
    assert!(
        captured.with_message(RPC_FAILURE_MSG).is_empty(),
        "{ending:?}: the handler was dropped, not answered; full capture:\n{}",
        captured.text
    );
    assert_gate_exits(&gate_metrics, &[(CHAIN_OUTCOME_CANCELLED, 1)], label);
    until(deadline, "the reservation released", || {
        gate.reserved_bytes() == 0
    })
    .await;
    assert_eq!(
        cell(&gate_metrics, "live_edit_upload_reserved_bytes", &[]),
        Some(0.0)
    );
    assert_eq!(seq(&tv, target.object_id).await, 1, "nothing applied");
    assert_eq!(
        commits(&tv, target.object_id).await,
        0,
        "the stream is unchanged"
    );

    let client = chain_service(&tv);
    assert_applied(
        &chained(&client, &target, "wire-p", "", false, field_ompf(1)).await,
        2,
        "the predecessor, committing late",
    );
    assert_applied(
        &chained(&client, &target, "wire-s", "wire-p", true, payload).await,
        3,
        "the cancelled call's own re-send: APPLIED, so the cancelled call never committed",
    );
    assert_eq!(commits(&tv, target.object_id).await, 2);

    drop(call);
    drop(server);
    drop(capture);
    purge_keys(&tv).await;
}

#[tokio::test]
async fn chained_call_parked_at_the_gate_and_reset_after_end_stream_applies_nothing_end_to_end() {
    parked_call_ended_on_the_wire_applies_nothing(
        WireEnding::ResetAfterEndStream,
        "chain-wire-reset",
    )
    .await;
}

#[tokio::test]
async fn chained_call_parked_at_the_gate_whose_connection_closes_applies_nothing_end_to_end() {
    parked_call_ended_on_the_wire_applies_nothing(WireEnding::TcpClose, "chain-wire-close").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The commit-time guard — groups 15, 16b, 17, 18, 25
// ═════════════════════════════════════════════════════════════════════════════

use futures::future::BoxFuture;
use live_edit::config::UpdateLayerBackend;
use live_edit::storage::external_values::{
    ExtRelease, ExtValueProbe, ExtValueVerify, ExternalValueStore, GcBatchResult, GcPassState,
    GcPolicy, MockExternalValueStore, ReleaseAttemptOutcome,
};

/// The external-value store of a call under test: `inner` (the mock, or a
/// Valkey store where a test counts references), which first runs a one-shot
/// `intruder` when an upload stores its values. A chained call stores them
/// after its chain gate and before its commit's first attempt — the window
/// between the gate's advisory `ready` and the commit, which only the
/// commit's predecessor guard covers.
///
/// The second store call — the checkpoint-recovery re-issue's own pre-store
/// (b.5wb), when the first call ended `NOT_FOUND` and a restore landed —
/// first runs the one-shot `at_second`, then answers `refuse_second` instead
/// of storing when one is set.
struct IntrudingExtStore {
    inner: Arc<dyn ExternalValueStore>,
    intruder: std::sync::Mutex<Option<BoxFuture<'static, ()>>>,
    stores: std::sync::atomic::AtomicUsize,
    at_second: std::sync::Mutex<Option<BoxFuture<'static, ()>>>,
    refuse_second: std::sync::Mutex<Option<ServiceError>>,
}

impl IntrudingExtStore {
    fn over(inner: Arc<dyn ExternalValueStore>) -> Self {
        Self {
            inner,
            intruder: std::sync::Mutex::new(None),
            stores: std::sync::atomic::AtomicUsize::new(0),
            at_second: std::sync::Mutex::new(None),
            refuse_second: std::sync::Mutex::new(None),
        }
    }

    /// Before a store: count it, run the intruder on the first, and on the
    /// second run `at_second` and answer `refuse_second` if set.
    async fn intrude(&self) -> Result<(), ServiceError> {
        let call = self
            .stores
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        let intruder = self.intruder.lock().unwrap().take();
        if let Some(intruder) = intruder {
            intruder.await;
        }
        if call == 2 {
            let at_second = self.at_second.lock().unwrap().take();
            if let Some(at_second) = at_second {
                at_second.await;
            }
            if let Some(refusal) = self.refuse_second.lock().unwrap().take() {
                return Err(refusal);
            }
        }
        Ok(())
    }

    /// How many store calls reached the store.
    fn stores(&self) -> usize {
        self.stores.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl ExternalValueStore for IntrudingExtStore {
    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        self.intrude().await?;
        self.inner.store_values(values).await
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.intrude().await?;
        self.inner.store_verified_values(values).await
    }

    fn ext_value_verify(&self) -> ExtValueVerify<'_> {
        self.inner.ext_value_verify()
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        self.inner.load_values(hashes).await
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner.read_batch(hashes).await
    }

    async fn probe_values(&self, hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
        self.inner.probe_values(hashes).await
    }

    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<ReleaseAttemptOutcome, ServiceError> {
        self.inner
            .enqueue_retirement_attempt(object_id, attempt, releases, send_by)
            .await
    }

    async fn process_pending(
        &self,
        policy: &GcPolicy,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(
        &self,
    ) -> Option<&Arc<live_edit::storage::ext_release_gate::ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

/// The one external value every [`external_value_ompf`] delta holds.
fn external_blob() -> Vec<u8> {
    (0..256u32).map(|i| (i % 251) as u8).collect()
}

/// An OMPF whose delta sets one field held out of line, in one external
/// value ([`external_blob`]), plus `struct_commands`: the call stores the
/// value (the [`IntrudingExtStore`] seam) before its commit.
fn external_value_ompf(struct_commands: Vec<StructCommand>) -> Vec<u8> {
    let blob = external_blob();
    let hash = live_edit::storage::external_values::ext_content_hash(&blob).to_vec();
    let delta = OwnedDelta {
        struct_commands,
        ..crate::common::owned_delta::delta_with_set_field_external(
            1,
            "blob",
            hash,
            blob.len() as u64,
            1,
            0,
        )
    };
    delta_to_ompf_bytes(&delta, &[&blob])
}

/// The routes a chained call's commit takes, each with its own place for the
/// predecessor guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// The `lua` fast path: the script's PREDECESSOR step, after its step 1c.
    Fast,
    /// The cached structural route under `lua`: its read side, after the
    /// superseded check and before planning, then the script's PREDECESSOR
    /// step in its `seq`-fenced mode.
    Structural,
    /// The WATCH path: its predecessor check, after its superseded guard.
    Watch,
}

/// Where a chained call commits: the store's backend and the class of the
/// call's delta.
#[derive(Clone, Copy, Debug)]
struct Cell {
    backend: UpdateLayerBackend,
    /// The delta creates a section (structural or WATCH route), rather than
    /// only setting a field.
    structural: bool,
}

/// Every backend × delta class: `lua` (fast, structural), `lua-watch-structural`
/// (fast, WATCH) and `txn` (WATCH, WATCH).
const CELLS: [Cell; 6] = [
    Cell::new(UpdateLayerBackend::Lua, false),
    Cell::new(UpdateLayerBackend::Lua, true),
    Cell::new(UpdateLayerBackend::LuaWatchStructural, false),
    Cell::new(UpdateLayerBackend::LuaWatchStructural, true),
    Cell::new(UpdateLayerBackend::Txn, false),
    Cell::new(UpdateLayerBackend::Txn, true),
];

impl Cell {
    const fn new(backend: UpdateLayerBackend, structural: bool) -> Self {
        Self {
            backend,
            structural,
        }
    }

    /// The cell a route is first taken in: `lua` for the fast and structural
    /// routes, `txn` with a field delta for the WATCH path.
    fn of(route: Route) -> Self {
        match route {
            Route::Fast => Self::new(UpdateLayerBackend::Lua, false),
            Route::Structural => Self::new(UpdateLayerBackend::Lua, true),
            Route::Watch => Self::new(UpdateLayerBackend::Txn, false),
        }
    }

    /// The route the cell's commit takes (`LayerStore`'s dispatch).
    fn route(self) -> Route {
        match (self.backend, self.structural) {
            (UpdateLayerBackend::Txn, _) | (UpdateLayerBackend::LuaWatchStructural, true) => {
                Route::Watch
            }
            (_, false) => Route::Fast,
            (UpdateLayerBackend::Lua, true) => Route::Structural,
        }
    }

    /// A store tuned for the cell.
    fn tune(self, store: LayerStore) -> LayerStore {
        store.with_update_layer_backend(self.backend)
    }

    /// The payload of write `n` in this cell, with no external value: a
    /// structural cell's creates section `60 + n`.
    fn payload(self, n: u8) -> Vec<u8> {
        if self.structural {
            sections_ompf(&[(0, 60 + u64::from(n))])
        } else {
            field_ompf(n)
        }
    }

    /// The payload of write `n` in this cell carrying one external value, so
    /// the call meets the [`IntrudingExtStore`] seam before its commit.
    fn ext_payload(self, n: u8) -> Vec<u8> {
        external_value_ompf(if self.structural {
            vec![create_section(60 + u64::from(n))]
        } else {
            vec![]
        })
    }
}

/// `CreateSection(section_id)` under the root.
fn create_section(section_id: u64) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id: 0,
        section_id,
        section_name: format!("s{section_id}"),
        section_type: 1,
    })
}

/// The `operation_id` of the write an [`Intrusion::OperationWrite`] lands.
const INTRUDER: &str = "guard-intruder";

/// What lands between a chained call's gate and its commit: after the gate's
/// `ready` on its predecessor's record, while the call stores its external
/// value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Intrusion {
    /// Another write of the same `client_id` carrying an operation id
    /// ([`INTRUDER`]), committed by another replica: `last_record=operation`.
    OperationWrite,
    /// Another write of the same `client_id` with no operation id:
    /// `last_record=no_operation_id`.
    PlainWrite,
    /// The writer's record deleted (`HDEL obj:wtok <client_id>`):
    /// `last_record=absent`.
    RecordDeleted,
    /// The record left as it is: P's, so S's predecessor is the record (only
    /// the intruder's [`Then`] step acts).
    Untouched,
}

impl Intrusion {
    /// The metadata of the predecessor refusal the record it leaves answers.
    fn metadata(self) -> Vec<(&'static str, String)> {
        match self {
            Intrusion::OperationWrite => last_operation(INTRUDER),
            Intrusion::PlainWrite => last_record(refusal_metadata::NO_OPERATION_ID),
            Intrusion::RecordDeleted => last_record(refusal_metadata::ABSENT),
            Intrusion::Untouched => unreachable!("an untouched record answers no refusal"),
        }
    }

    /// The object's `seq` once it landed, over the predecessor's 2.
    fn seq(self) -> u64 {
        match self {
            Intrusion::OperationWrite | Intrusion::PlainWrite => 3,
            Intrusion::RecordDeleted | Intrusion::Untouched => 2,
        }
    }
}

/// What the intruder does to the object after its [`Intrusion`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Then {
    Nothing,
    /// `HSET obj:hdr superseded 1`: a `CreateLayer` overwrite's mark.
    Supersede,
    /// A string on the snapshot fence key (script step 1d, WATCH step 2b).
    RetypeFence,
    /// A string on the commit release record `obj:wrel` (script step 1e,
    /// WATCH step 2c).
    RetypeReleaseRecord,
    /// `DEL obj:hdr`: the object gone, for checkpoint recovery to restore.
    DeleteHeader,
}

/// A chained call S's commit-guard fixture: a service in `cell` over the
/// fixture's client whose external-value store runs a one-shot intruder, a
/// peer replica for the intruding write, and a target on which the chain's
/// first write P (`guard-p`) committed at `seq` 2 through the service — so a
/// structural cell's cache entry for the object is warm and S's read is a hit.
struct GuardFixture {
    tv: TestValkey,
    svc: LiveEditService,
    ext: Arc<IntrudingExtStore>,
    peer: Arc<LiveEditService>,
    target: Target,
}

/// The predecessor every [`GuardFixture`] call S names.
const GUARD_P: &str = "guard-p";

/// The chained call every [`GuardFixture`] stages.
const GUARD_S: &str = "guard-s";

async fn guard_fixture(cell: Cell, label: &str) -> GuardFixture {
    guard_fixture_over(cell, label, |_| Arc::new(MockExternalValueStore::new())).await
}

/// [`guard_fixture`] whose intruding store wraps the store `inner` builds on
/// the fixture's Valkey.
async fn guard_fixture_over(
    cell: Cell,
    label: &str,
    inner: impl FnOnce(&TestValkey) -> Arc<dyn ExternalValueStore>,
) -> GuardFixture {
    let tv = TestValkey::setup().await;
    let target = new_target(&tv, label).await;
    let ext = Arc::new(IntrudingExtStore::over(inner(&tv)));
    let svc = LiveEditService {
        ext_value_store: Arc::clone(&ext) as Arc<dyn ExternalValueStore>,
        ..service_over(&tv, tv.client().clone(), |store| cell.tune(store))
    };
    let peer = Arc::new(peer_replica(&tv, |store| store).await);
    assert_applied(
        &chained(&svc, &target, GUARD_P, "", false, cell.payload(1)).await,
        2,
        "P, the predecessor",
    );
    GuardFixture {
        tv,
        svc,
        ext,
        peer,
        target,
    }
}

impl GuardFixture {
    /// Arm the one-shot intruder: `intrusion`, then `then`. The returned flag
    /// turns true once it ran — the vacuity guard that it landed between S's
    /// gate and its commit.
    fn stage(&self, intrusion: Intrusion, then: Then) -> Arc<std::sync::atomic::AtomicBool> {
        let landed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (peer, target, client, keys, flag) = (
            Arc::clone(&self.peer),
            self.target.clone(),
            self.tv.client().clone(),
            self.tv.key_schema().clone(),
            Arc::clone(&landed),
        );
        *self.ext.intruder.lock().unwrap() = Some(Box::pin(async move {
            let oid = target.object_id.to_string();
            let write = match intrusion {
                Intrusion::OperationWrite => Some(INTRUDER),
                Intrusion::PlainWrite => Some(""),
                Intrusion::RecordDeleted | Intrusion::Untouched => None,
            };
            if let Some(operation_id) = write {
                let landed = tokio::spawn(async move {
                    plain_from(&peer, WRITER, &target, operation_id, field_ompf(9)).await
                })
                .await
                .expect("the intruder's task must not panic")
                .expect("the intruder's write must commit");
                assert_eq!(
                    landed.sequence_num, 3,
                    "the intruder, after S's gate, directly after P"
                );
            }
            if intrusion == Intrusion::RecordDeleted {
                let removed: i64 = client
                    .pool
                    .hdel(keys.obj_write_token(&oid), WRITER)
                    .await
                    .expect("HDEL obj:wtok <writer>");
                assert_eq!(removed, 1, "the intruder deleted P's record");
            }
            let retype = |key: String| {
                let client = client.clone();
                async move {
                    let _: i64 = client.pool.del(&key).await.expect("DEL before retyping");
                    let _: () = client
                        .pool
                        .set(&key, "not-a-hash", None, None, false)
                        .await
                        .expect("retyping the key must succeed");
                }
            };
            match then {
                Then::Nothing => {}
                Then::Supersede => {
                    let _: i64 = client
                        .pool
                        .hset(keys.obj_header(&oid), [("superseded", "1")])
                        .await
                        .expect("HSET superseded=1");
                }
                Then::RetypeFence => retype(keys.obj_snapshot_fence(&oid)).await,
                Then::RetypeReleaseRecord => retype(keys.obj_write_releases(&oid)).await,
                Then::DeleteHeader => {
                    let _: i64 = client
                        .pool
                        .del(keys.obj_header(&oid))
                        .await
                        .expect("DEL obj:hdr");
                }
            }
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }));
        landed
    }

    /// S — `guard-s` after P — sent with `payload`, its completion line
    /// captured.
    async fn send_s(
        &self,
        payload: Vec<u8>,
    ) -> (
        Result<UpdateLayerResponse, Status>,
        super::log_capture::Captured,
    ) {
        capture(chained(
            &self.svc,
            &self.target,
            GUARD_S,
            GUARD_P,
            false,
            payload,
        ))
        .await
    }
}

/// Assert `staged` ran.
#[track_caller]
fn assert_intruded(staged: &std::sync::atomic::AtomicBool, what: &str) {
    assert!(
        staged.load(std::sync::atomic::Ordering::SeqCst),
        "{what}: precondition: the intruder ran between S's gate and its commit"
    );
}

/// Why `result` is not the predecessor refusal `ABORTED`
/// `PREDECESSOR_NOT_COMMITTED` carrying exactly `metadata`, or `None` when it
/// is ([`assert_predecessor_not_committed`], without panicking, for the tests
/// that collect every cell's answer).
fn not_predecessor_refusal(
    result: &Result<UpdateLayerResponse, Status>,
    metadata: &[(&'static str, String)],
) -> Option<String> {
    let status = match result {
        Ok(response) => return Some(format!("committed: {response:?}")),
        Err(status) => status,
    };
    let Some(wire) = decode(status).filter(|_| status.code() == Code::Aborted) else {
        return Some(format!("not ABORTED with an ErrorInfo: {status:?}"));
    };
    let mut want: Vec<(String, String)> = metadata
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    want.sort();
    (wire.reason != refusal_reason::PREDECESSOR_NOT_COMMITTED
        || wire.domain != REFUSAL_DETAIL_DOMAIN
        || wire.metadata != want
        || wire.retry_delay.is_some())
    .then(|| format!("the wrong ErrorInfo {wire:?}, want metadata {want:?}: {status:?}"))
}

// ── Groups 16b and 17: never overtaken; a refusal at the commit is terminal ──

/// Groups 16b and 17 on `route`. A chained call S passes its gate on its
/// predecessor P's record. After the gate's `ready` and before S's commit —
/// while S stores its external value, where no start-record read could
/// re-take the record — the record moves, in each of three ways. S's commit
/// then refuses itself on its first attempt: `ABORTED`
/// `PREDECESSOR_NOT_COMMITTED` naming the record it read, for each of the
/// three `last_record` values; logged once, at INFO from stage `storage`,
/// with exactly one attempt (a terminal verdict, never retried in-loop); and
/// nothing applied. S is never answered overtaken and the overtaken counter
/// stays at zero on both backend cells (group 16b's commit half; the gate
/// half is [`chained_writes_sent_out_of_order_to_two_replicas_commit_in_chain_order`]).
async fn record_moved_after_the_gate_is_refused_by_the_commit(route: Route) {
    let cell = Cell::of(route);
    for intrusion in [
        Intrusion::OperationWrite,
        Intrusion::PlainWrite,
        Intrusion::RecordDeleted,
    ] {
        let what = format!("{route:?}, {intrusion:?}");
        let fx = guard_fixture(cell, &format!("guard-moved-{route:?}-{intrusion:?}")).await;
        let staged = fx.stage(intrusion, Then::Nothing);

        let (result, captured) = fx.send_s(cell.ext_payload(2)).await;

        assert_intruded(&staged, &what);
        let status =
            result.expect_err("S's commit must refuse itself: its predecessor is not the record");
        assert_predecessor_not_committed(&status, &intrusion.metadata());
        let event = captured.only(RPC_FAILURE_MSG);
        assert_common_fields(event, "INFO", "UpdateLayer", "storage", "ABORTED");
        assert_field(event, "attempt", 1.into());
        assert_eq!(
            overtaken(&fx.svc.metrics),
            0,
            "{what}: a chained call is never overtaken"
        );
        assert_eq!(
            seq(&fx.tv, fx.target.object_id).await,
            intrusion.seq(),
            "{what}: S applied nothing"
        );
        let latest = record(&fx.tv, fx.target.object_id, WRITER).await;
        let left_by_the_intruder = match intrusion {
            Intrusion::OperationWrite => names(&latest, INTRUDER),
            Intrusion::PlainWrite => latest.as_deref().is_some_and(|r| !r.starts_with("op.")),
            Intrusion::RecordDeleted => latest.is_none(),
            Intrusion::Untouched => unreachable!("not staged here"),
        };
        assert!(
            left_by_the_intruder,
            "{what}: the record is the intruder's, untouched by S; got {latest:?}"
        );
        assert_gate_exits(&fx.svc.metrics, &[(CHAIN_OUTCOME_READY, 2)], &what);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_whose_record_moves_after_its_gate_is_refused_predecessor_not_committed_by_its_commit_on_the_fast_route(
) {
    record_moved_after_the_gate_is_refused_by_the_commit(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_whose_record_moves_after_its_gate_is_refused_predecessor_not_committed_by_its_commit_on_the_structural_route(
) {
    record_moved_after_the_gate_is_refused_by_the_commit(Route::Structural).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_whose_record_moves_after_its_gate_is_refused_predecessor_not_committed_by_its_commit_on_the_watch_route(
) {
    record_moved_after_the_gate_is_refused_by_the_commit(Route::Watch).await;
}

// ── Groups 15 and 25: the guard order, end to end ────────────────────────────

/// Group 25 (F7), and group 15's PREDECESSOR-before-FENCED pair end to end,
/// in every backend × delta-class cell. S's predecessor is pending at its
/// commit (the intruder's operation write), and then either the snapshot
/// fence key or the commit release record `obj:wrel` is made to hold a
/// string. S is refused `ABORTED` `PREDECESSOR_NOT_COMMITTED` naming the
/// intruder, never `INTERNAL` with the commit's `WRONGTYPE_KEY`: the
/// predecessor check sits before the fence (script step 1d, WATCH step 2b)
/// and before the release record's type (script step 1e, WATCH step 2c), and
/// the structural read side answers before either is reached. A predecessor
/// check placed after the fence or after step 1e turns that cell's answer
/// into the keyspace refusal. Nothing applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pending_predecessor_is_refused_aborted_ahead_of_a_wrong_typed_fence_or_release_record_in_every_cell(
) {
    let mut failures = Vec::new();
    for cell in CELLS {
        for then in [Then::RetypeFence, Then::RetypeReleaseRecord] {
            let what = format!("{cell:?} → {:?}, {then:?}", cell.route());
            let fx = guard_fixture(cell, "guard-wrongtype").await;
            let staged = fx.stage(Intrusion::OperationWrite, then);

            let (result, _) = fx.send_s(cell.ext_payload(2)).await;

            assert_intruded(&staged, &what);
            if let Some(why) =
                not_predecessor_refusal(&result, &Intrusion::OperationWrite.metadata())
            {
                failures.push(format!("{what}: {why}"));
            }
            if seq(&fx.tv, fx.target.object_id).await != 3 {
                failures.push(format!("{what}: S applied"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Group 15, the order `own-key > SUPERSEDED > PREDECESSOR` end to end, in
/// every backend × delta-class cell (`PREDECESSOR > FENCED` is
/// [`pending_predecessor_is_refused_aborted_ahead_of_a_wrong_typed_fence_or_release_record_in_every_cell`],
/// `PREDECESSOR > planning` is
/// [`chained_call_whose_record_moved_is_refused_predecessor_not_committed_before_planning_in_every_structural_cell`]).
///
/// - **own-key before PREDECESSOR.** S commits; its re-send (same operation,
///   bytes and predecessor) passes the gate on S's own record and is answered
///   `ALREADY_COMMITTED` at S's `seq`. The record names S, not S's
///   predecessor, so a predecessor check ahead of the own-key match would
///   refuse the re-send `ABORTED`.
/// - **SUPERSEDED before PREDECESSOR.** The intruder's operation write leaves
///   S's predecessor pending and the object is marked superseded: S is
///   answered `FAILED_PRECONDITION` (superseded) with no status details,
///   not `ABORTED`. On the structural route this is the read side's cache
///   hit (the fixture's entry is warm); the probe and cold-start sites are
///   pinned in `structural_cache_read_test`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_commit_answers_its_own_record_then_superseded_before_its_predecessor_in_every_cell(
) {
    let mut failures = Vec::new();
    for cell in CELLS {
        let what = format!("{cell:?} → {:?}", cell.route());

        let fx = guard_fixture(cell, "guard-order-own").await;
        let first = chained(
            &fx.svc,
            &fx.target,
            GUARD_S,
            GUARD_P,
            false,
            cell.payload(2),
        )
        .await;
        if first.as_ref().map(|r| (r.sequence_num, outcome(r))).ok()
            != Some((3, OperationOutcome::Applied))
        {
            failures.push(format!(
                "{what}: precondition: S commits at 3; got {first:?}"
            ));
            continue;
        }
        let resent = chained(
            &fx.svc,
            &fx.target,
            GUARD_S,
            GUARD_P,
            false,
            cell.payload(2),
        )
        .await;
        match &resent {
            Ok(r) if (r.sequence_num, outcome(r)) == (3, OperationOutcome::AlreadyCommitted) => {}
            other => failures.push(format!(
                "{what}: own-key: S's re-send must be answered ALREADY_COMMITTED at 3; got {other:?}"
            )),
        }
        if seq(&fx.tv, fx.target.object_id).await != 3 {
            failures.push(format!("{what}: own-key: the re-send applied"));
        }

        let fx = guard_fixture(cell, "guard-order-superseded").await;
        let staged = fx.stage(Intrusion::OperationWrite, Then::Supersede);
        let (result, _) = fx.send_s(cell.ext_payload(2)).await;
        assert_intruded(&staged, &what);
        match &result {
            Err(status)
                if status.code() == Code::FailedPrecondition
                    && status.message().contains("superseded")
                    && decode(status).is_none() => {}
            other => failures.push(format!(
                "{what}: superseded: S must be refused FAILED_PRECONDITION superseded, plain; got \
                 {other:?}"
            )),
        }
        if seq(&fx.tv, fx.target.object_id).await != 3 {
            failures.push(format!("{what}: superseded: S applied"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Group 15: the read side answers before planning. In each cell whose
/// delta is structural (the structural route under `lua`, the WATCH path
/// under `lua-watch-structural` and `txn`), S carries a delta the domain
/// refuses while planning (it creates section 70 twice) and its record has
/// moved past its predecessor: S is refused `ABORTED`
/// `PREDECESSOR_NOT_COMMITTED` naming the intruder — the predecessor is
/// checked before anything is planned — never `INVALID_ARGUMENT`
/// `INVALID_UPDATE`. The control, per cell: the same call with its
/// predecessor the record is refused by planning, so the delta does reach
/// the planner on that route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_whose_record_moved_is_refused_predecessor_not_committed_before_planning_in_every_structural_cell(
) {
    let duplicate = || external_value_ompf(vec![create_section(70), create_section(70)]);
    let mut failures = Vec::new();
    for cell in CELLS.into_iter().filter(|cell| cell.structural) {
        let what = format!("{cell:?} → {:?}", cell.route());

        let fx = guard_fixture(cell, "guard-before-planning").await;
        let staged = fx.stage(Intrusion::OperationWrite, Then::Nothing);
        let (result, _) = fx.send_s(duplicate()).await;
        assert_intruded(&staged, &what);
        if let Some(why) = not_predecessor_refusal(&result, &Intrusion::OperationWrite.metadata()) {
            failures.push(format!("{what}: {why}"));
        }

        let control = guard_fixture(cell, "guard-before-planning-control").await;
        let (result, _) = control.send_s(duplicate()).await;
        match &result {
            Err(status) if status.code() == Code::InvalidArgument => {
                assert_invalid_update(status, &format!("{what}: control"));
            }
            other => failures.push(format!(
                "{what}: control: with its predecessor the record, the planner must refuse S; got \
                 {other:?}"
            )),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ── Group 18: restore mid-chain ──────────────────────────────────────────────

/// `ref_count` of `hash`, `None` when its blob is gone.
async fn ref_count(tv: &TestValkey, hash: &[u8; 32]) -> Option<i64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(
            tv.key_schema()
                .ext_meta(&live_edit::storage::keys::sha256_to_hex(hash)),
            "ref_count",
        )
        .await
        .expect("HGET ref_count must succeed");
    raw.map(|v| v.parse().expect("ref_count is an integer"))
}

/// Every member of `le:ext:{gc}:pending`.
async fn pending_members(tv: &TestValkey) -> Vec<String> {
    tv.client()
        .pool
        .smembers(tv.key_schema().ext_pending())
        .await
        .expect("SMEMBERS ext pending")
}

/// Group 18 on `route`. P committed; S passes its gate on P's record; before
/// S's first attempt the object's header is deleted (the intruder: the
/// between-attempts hook runs only before a retry). S's attempt finds the
/// object gone, and the handler restores it from its checkpoint (seeded at
/// `seq` 1, so without P's record: a restore loses the records), pre-stores
/// S's external value H again for the re-issue (b.5wb), and re-issues the
/// commit once with the same writer guard, S's predecessor. P's record is
/// gone, so the re-issue's commit refuses S `ABORTED`
/// `PREDECESSOR_NOT_COMMITTED` `last_record=absent` — `ABORTED`, not a plain
/// `UNAVAILABLE`, because the first call made one attempt — and S applied
/// nothing on the restored object.
///
/// The re-issue is refused on its own first attempt, so it hands back its
/// own pre-flight increment, as any call does: exactly one `Resend` member
/// of H on the object is pending — the re-issue's; the first call, ended
/// `NOT_FOUND`, hands back nothing. H also holds one reference outside the
/// object, so its count stays readable: after GC it is that reference plus
/// the first call's increment, which leaks to the reference-count
/// reconciler (its commit never landed, so nothing spent it).
///
/// Fail-on-old: before b.5wb the re-issue reused the first call's pre-store
/// and handed back nothing, so no `Resend` member was pending.
async fn restore_mid_chain_is_refused_with_last_record_absent(route: Route) {
    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::CheckpointStore;
    use live_edit::storage::external_values::{process_pending_gc, ValkeyExternalValueStore};

    let cell = Cell::of(route);
    let valkey_ext: std::sync::Mutex<Option<Arc<ValkeyExternalValueStore>>> =
        std::sync::Mutex::new(None);
    let mut fx = guard_fixture_over(cell, &format!("guard-restore-{route:?}"), |tv| {
        let ext = Arc::new(ValkeyExternalValueStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        *valkey_ext.lock().unwrap() = Some(Arc::clone(&ext));
        ext
    })
    .await;
    let valkey_ext = valkey_ext
        .into_inner()
        .unwrap()
        .expect("the fixture built the Valkey store");
    let blob = external_blob();
    let hash = live_edit::storage::external_values::ext_content_hash(&blob);
    valkey_ext
        .store_values(&[(&hash, blob.as_slice())])
        .await
        .expect("the reference held elsewhere must be stored");
    let metrics = crate::common::replace_metrics(&mut fx.svc);
    let checkpoints = Arc::new(InMemoryCheckpointStore::new());
    crate::common::seed_layer_checkpoint(
        &checkpoints,
        fx.target.object_id,
        &fx.target.session_id,
        LAYER,
        1,
    )
    .await;
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = true;
    config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
    fx.svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
        checkpoints as Arc<dyn CheckpointStore>,
        Arc::clone(&fx.svc.layer_store),
        Arc::new(config),
        Arc::clone(&fx.svc.metrics),
    ));
    let staged = fx.stage(Intrusion::Untouched, Then::DeleteHeader);

    let (result, _) = fx.send_s(cell.ext_payload(2)).await;

    assert_intruded(&staged, &format!("{route:?}"));
    assert_eq!(
        crate::common::recovery_metric(&metrics, "restored"),
        1,
        "{route:?}: precondition: S's attempt found the object gone and the restore landed"
    );
    let status = result.expect_err("the re-issue must refuse S: its predecessor's record is gone");
    assert_predecessor_not_committed(&status, &last_record(refusal_metadata::ABSENT));
    assert_eq!(
        record(&fx.tv, fx.target.object_id, WRITER).await,
        None,
        "{route:?}: the restore left no record, and S, which would have recorded its own \
         operation, applied nothing to the restored object"
    );
    let pending = pending_members(&fx.tv).await;
    let resend_prefix = format!(
        "{}:{}:rsd",
        live_edit::storage::keys::sha256_to_hex(&hash),
        fx.target.object_id
    );
    assert_eq!(
        pending.len(),
        1,
        "{route:?}: one member pending, the re-issue's hand-back; got {pending:?}"
    );
    assert!(
        pending[0].starts_with(&resend_prefix),
        "{route:?}: the recovery re-issue, refused on its first attempt, hands back its own \
         pre-flight increment of H as a Resend member; got {pending:?}"
    );
    assert_eq!(
        ref_count(&fx.tv, &hash).await,
        Some(3),
        "{route:?}: before GC, H counts the reference elsewhere and both pre-stores"
    );
    assert_eq!(
        fx.ext.stores(),
        2,
        "{route:?}: the first call and the re-issue each pre-stored H"
    );
    process_pending_gc(valkey_ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
    assert_eq!(
        ref_count(&fx.tv, &hash).await,
        Some(2),
        "{route:?}: after GC, the reference elsewhere and the first call's leaked increment"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_restored_mid_chain_is_refused_aborted_last_record_absent_and_hands_back_its_own_preflight_increment_on_the_fast_route(
) {
    restore_mid_chain_is_refused_with_last_record_absent(Route::Fast).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_restored_mid_chain_is_refused_aborted_last_record_absent_and_hands_back_its_own_preflight_increment_on_the_structural_route(
) {
    restore_mid_chain_is_refused_with_last_record_absent(Route::Structural).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_call_restored_mid_chain_is_refused_aborted_last_record_absent_and_hands_back_its_own_preflight_increment_on_the_watch_route(
) {
    restore_mid_chain_is_refused_with_last_record_absent(Route::Watch).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Seams (test-fault-injection)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod fault_injected {
    //! The cases that drive `LayerStore`'s `test-fault-injection` seams: the
    //! overwrite-cleanup hold, the attempt fault queue, the between-attempts
    //! hook, the start-record read's fault queue, the fast and structural
    //! routes' pre-dispatch hooks, the WATCH window hook, the lost-reply
    //! seams (the post-`EVALSHA` fault queue, the substituted `EXEC` verdict
    //! and the reconcile read's fault queue) and the pinned commit tokens.
    //! Compiled out without
    //! `--features test-fault-injection`.

    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
    use std::sync::Mutex;

    use fred::clients::Client;
    use fred::interfaces::ClientLike;
    use futures::FutureExt;

    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::CheckpointStore;
    use live_edit::storage::client::LivenessOverrides;
    use live_edit::storage::external_values::{process_pending_gc, ValkeyExternalValueStore};
    use live_edit::storage::keys::sha256_to_hex;
    use live_edit::storage::layer::BetweenAttemptsHook;
    use live_edit::storage::notifications::RELAY_OMPF_FIELD;
    use live_edit::storage::test_support::ConnStall;
    use live_edit::storage::watched_txn::{
        WatchWindowHook, WatchedExecOverride, WatchedExecOverrideQueue,
    };

    use super::*;
    use crate::storage::b_zt3_support::proxied_client_with;
    use crate::storage::snapshot_read_support::ReplyGate;

    // ── Group 11: the superseded shape ───────────────────────────────────────

    /// The superseded shape: a `CreateLayer` overwrite while a chained call
    /// waits, its old object's cleanup held so the old header stays, marked
    /// `superseded=1`. The waiting call reads it as replaced, re-resolves to
    /// the new object and is answered `FAILED_PRECONDITION` naming both,
    /// with no status details (Q2), outcome `replaced`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chained_call_waiting_when_create_layer_supersedes_its_object_is_answered_failed_precondition(
    ) {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let cleanup = Arc::new(tokio::sync::Barrier::new(2));
        let svc = Arc::new(service_over(&tv, tv.client().clone(), |store| {
            store.with_overwrite_cleanup_gate_for_test(Arc::clone(&cleanup))
        }));
        let target = new_target(&tv, "chain-superseded").await;

        let waiting = spawn_chained(
            &svc,
            &target,
            "superseded-s",
            "superseded-p",
            false,
            field_ompf(1),
            CAPPED_DEADLINE,
        );
        parked(deadline, &svc.metrics, 1, "the call parked").await;

        let replacement = overwrite(&svc, &target).await;
        let superseded: Option<String> = tv
            .client()
            .pool
            .hget(
                tv.key_schema().obj_header(&target.object_id.to_string()),
                "superseded",
            )
            .await
            .expect("HGET superseded");
        assert_eq!(
            superseded.as_deref(),
            Some("1"),
            "precondition: the old header stays, superseded, while its cleanup is held"
        );
        let status = answer(deadline, "the waiting call", waiting)
            .await
            .expect_err("a waiting call whose object was superseded must fail");
        assert_eq!(status.code(), Code::FailedPrecondition, "{status:?}");
        assert_eq!(
            status.message(),
            format!(
                "object_id mismatch: expected {replacement}, got {}",
                target.object_id
            )
        );
        assert_no_refusal_detail(&status);
        assert_gate_exits(&svc.metrics, &[(CHAIN_OUTCOME_REPLACED, 1)], "superseded");
        assert_eq!(
            seq(&tv, replacement).await,
            1,
            "nothing applied to the new object"
        );

        settled(deadline, "the held cleanup released", cleanup.wait()).await;
    }

    // ── Review round 1, item 2: no start-record read for a chained call ─────

    /// A chained call issues no call-start record read, at either of its two
    /// sites: the handler's (`grpc::layer`, the `chain.is_none()` guard on
    /// `spawn_start_record_read`) and the store's
    /// (`LayerStore::update_layer_with_operation` reads one only for an
    /// operation without a writer guard, and a chained call's operation
    /// carries its predecessor as its guard). With one failure queued for that
    /// read, a chained call commits and leaves the failure queued; the
    /// unchained operation-id twin then takes it and is refused `UNAVAILABLE`
    /// `UPSTREAM_UNAVAILABLE` — the vacuity check that the seam is live.
    ///
    /// Why it matters beyond the wasted read: a start record is the guard of
    /// an unchained call, which is refused overtaken once the record moves.
    /// Read for a chained call, it would either fail the call on a failed read
    /// that its guard never needed, or — attached as its guard — answer a
    /// record that moved past the call overtaken instead of the contract's
    /// `PREDECESSOR_NOT_COMMITTED`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chained_call_issues_no_start_record_read_and_its_unchained_twin_does() {
        let tv = TestValkey::setup().await;
        let faults = Arc::new(Mutex::new(VecDeque::from([fred::error::Error::new(
            fred::error::ErrorKind::IO,
            "injected start-record read failure",
        )])));
        let svc = service_over(&tv, tv.client().clone(), |store| {
            store.with_start_record_read_fred_error_injector_for_test(Arc::clone(&faults))
        });
        let target = new_target(&tv, "chain-no-start-read").await;

        assert_applied(
            &chained(&svc, &target, "no-read-1", "", false, field_ompf(1)).await,
            2,
            "the chained call",
        );
        assert_eq!(
            faults.lock().unwrap().len(),
            1,
            "the chained call issued no start-record read: the queued failure is untouched"
        );

        let status = plain_from(&svc, "twin-writer", &target, "no-read-twin", field_ompf(2))
            .await
            .expect_err("the unchained operation-id twin takes the queued failure");
        // Refused before its commit marker: the b.xd1 pre-commit hint.
        crate::common::refusal_detail::assert_upstream_unavailable(&status);
        assert!(
            faults.lock().unwrap().is_empty(),
            "vacuity: the unchained twin's start-record read took the queued failure"
        );
        assert_eq!(
            seq(&tv, target.object_id).await,
            2,
            "the twin applied nothing"
        );
    }

    // ── Group 14 (a): after a counted retry, plain UNAVAILABLE ───────────────

    type FaultQueue = Arc<Mutex<VecDeque<ServiceError>>>;

    /// Fail the next attempt `faults` reaches with the in-loop retry signal.
    fn fail_next_attempt(faults: &FaultQueue) {
        faults.lock().unwrap().push_back(ServiceError::Aborted(
            "injected transient failure".to_owned(),
        ));
    }

    /// A chained call whose first attempt fails transiently and whose second
    /// meets a planning `INVALID_ARGUMENT` (`DuplicateSection`) is answered a
    /// plain `UNAVAILABLE`, outcome unknown — the first attempt may have sent
    /// its commit — never `INVALID_ARGUMENT` + `INVALID_UPDATE`, which would
    /// promise it applied nothing. The unchained call under the same faults
    /// keeps its `INVALID_ARGUMENT`: the rule is the chained contract's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn planning_invalid_argument_after_a_counted_retry_is_plain_unavailable() {
        let tv = TestValkey::setup().await;
        let faults: FaultQueue = Arc::new(Mutex::new(VecDeque::new()));
        let svc = service_over(&tv, tv.client().clone(), |store| {
            store.with_fault_injector_for_test(Arc::clone(&faults))
        });
        let target = new_target(&tv, "chain-retry-invalid").await;

        fail_next_attempt(&faults);
        let status = chained(
            &svc,
            &target,
            "retry-s",
            "",
            false,
            duplicate_section_ompf(),
        )
        .await
        .expect_err("a planning refusal must fail the call");
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert!(
            status.message().starts_with("UpdateLayer outcome unknown:")
                && status.message().contains("attempt 2")
                && status.message().contains("delta error"),
            "the outcome-unknown answer names the attempt and the planning refusal; got {status:?}"
        );
        assert_no_refusal_detail(&status);
        assert!(
            faults.lock().unwrap().is_empty(),
            "the injected fault was taken"
        );

        fail_next_attempt(&faults);
        let plain = update(
            &svc,
            vec![
                plain_header_from(WRITER, &target, "retry-plain"),
                make_update_data(duplicate_section_ompf()),
            ],
        )
        .await
        .expect_err("the unchained twin is refused too");
        assert_eq!(plain.code(), Code::InvalidArgument, "{plain:?}");
        assert_no_refusal_detail(&plain);

        assert_eq!(seq(&tv, target.object_id).await, 1, "nothing applied");
    }

    /// A chained call whose first `update_layer` call makes two attempts —
    /// the first failing transiently, the second finding the header gone —
    /// is restored from its checkpoint and re-issued once; the re-issue's
    /// planning `INVALID_ARGUMENT` on its own first attempt is still answered
    /// a plain `UNAVAILABLE` naming all three attempts: the first call's
    /// retry is counted across the re-issue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn planning_invalid_argument_on_the_reissue_after_a_two_attempt_first_call_is_plain_unavailable(
    ) {
        let tv = TestValkey::setup().await;
        let target = new_target(&tv, "chain-reissue-invalid").await;
        let faults: FaultQueue = Arc::new(Mutex::new(VecDeque::new()));
        fail_next_attempt(&faults);
        let header_key = tv.key_schema().obj_header(&target.object_id.to_string());
        let wiper = tv.client().clone();
        let wipe_before_attempt_2: BetweenAttemptsHook = Arc::new(move |attempt: u32| {
            let (wiper, header_key) = (wiper.clone(), header_key.clone());
            async move {
                if attempt == 2 {
                    let _: i64 = wiper
                        .pool
                        .del(header_key)
                        .await
                        .expect("DEL obj:hdr before attempt 2");
                }
            }
            .boxed()
        });
        let mut svc = service_over(&tv, tv.client().clone(), |store| {
            store
                .with_fault_injector_for_test(Arc::clone(&faults))
                .with_between_attempts_hook_for_test(wipe_before_attempt_2)
        });
        let metrics = crate::common::replace_metrics(&mut svc);
        let checkpoints = Arc::new(InMemoryCheckpointStore::new());
        crate::common::seed_layer_checkpoint(
            &checkpoints,
            target.object_id,
            &target.session_id,
            LAYER,
            1,
        )
        .await;
        let mut config = crate::common::test_config();
        config.checkpoint_enabled = true;
        config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
        svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
            checkpoints as Arc<dyn CheckpointStore>,
            Arc::clone(&svc.layer_store),
            Arc::new(config),
            Arc::clone(&svc.metrics),
        ));

        let status = chained(
            &svc,
            &target,
            "reissue-s",
            "",
            false,
            duplicate_section_ompf(),
        )
        .await
        .expect_err("the re-issue's planning refusal must fail the call");
        assert_eq!(
            crate::common::recovery_metric(&metrics, "restored"),
            1,
            "precondition: the second attempt found the header gone and the restore landed"
        );
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert!(
            status.message().starts_with("UpdateLayer outcome unknown:")
                && status.message().contains("attempt 3")
                && status.message().contains("delta error"),
            "the outcome-unknown answer counts the first call's two attempts and the \
             re-issue's one; got {status:?}"
        );
        assert_no_refusal_detail(&status);
    }

    // ── Group 14 (b): a predecessor refusal after a counted retry ───────────

    /// Assert `status` is the after-retry answer to a predecessor refusal at
    /// the commit: a plain `UNAVAILABLE`, outcome unknown, naming attempt
    /// `attempt` and the refusal, with no `ErrorInfo` — and without the
    /// refusal's own "this call applied nothing", which an earlier attempt's
    /// possible commit makes untrue (the ending the answer strips).
    #[track_caller]
    fn assert_predecessor_refusal_after_a_retry(status: &Status, attempt: u32, what: &str) {
        assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
        let message = status.message();
        assert!(
            message.starts_with("UpdateLayer outcome unknown:")
                && message.contains(&format!("attempt {attempt} was refused"))
                && message.contains("is not the latest operation recorded"),
            "{what}: the outcome-unknown answer names attempt {attempt} and the predecessor \
             refusal; got {status:?}"
        );
        assert!(
            !message.contains("applied nothing"),
            "{what}: after a counted retry the answer must not promise that the call applied \
             nothing; got {status:?}"
        );
        assert_no_refusal_detail(status);
    }

    /// Group 14 (b) on `route`. S's first attempt fails transiently; before
    /// its second, a same-`client_id` operation write from another replica
    /// moves the record past S's predecessor P. The second attempt's commit
    /// refuses S by its predecessor guard; but the first attempt may have
    /// committed with its reply lost (then a successor of it could have moved
    /// the record), so S is answered a plain `UNAVAILABLE`, outcome unknown,
    /// naming attempt 2 — logged at WARN from stage `storage`, two attempts —
    /// never `ABORTED` `PREDECESSOR_NOT_COMMITTED`, whose contract is "this
    /// call applied nothing". Nothing of S applied, and it is not overtaken.
    async fn predecessor_refusal_after_a_counted_retry_is_plain_unavailable(route: Route) {
        let tv = TestValkey::setup().await;
        let target = new_target(&tv, &format!("chain-retry-pnc-{route:?}")).await;
        let cell = Cell::of(route);
        let faults: FaultQueue = Arc::new(Mutex::new(VecDeque::new()));
        let peer = Arc::new(peer_replica(&tv, |store| store).await);
        let intruded = Arc::new(AtomicBool::new(false));
        let intrude_before_attempt_2: BetweenAttemptsHook = {
            let (peer, target, intruded) =
                (Arc::clone(&peer), target.clone(), Arc::clone(&intruded));
            Arc::new(move |attempt: u32| {
                let (peer, target, intruded) =
                    (Arc::clone(&peer), target.clone(), Arc::clone(&intruded));
                async move {
                    if attempt != 2 {
                        return;
                    }
                    let landed = tokio::spawn(async move {
                        plain_from(&peer, WRITER, &target, INTRUDER, field_ompf(9)).await
                    })
                    .await
                    .expect("the intruder's task must not panic");
                    assert_applied(&landed, 3, "the intruder, before S's second attempt");
                    intruded.store(true, SeqCst);
                }
                .boxed()
            })
        };
        let svc = service_over(&tv, tv.client().clone(), |store| {
            cell.tune(store)
                .with_fault_injector_for_test(Arc::clone(&faults))
                .with_between_attempts_hook_for_test(intrude_before_attempt_2)
        });
        assert_applied(
            &chained(&svc, &target, "retry-p", "", false, cell.payload(1)).await,
            2,
            "P, the predecessor",
        );

        fail_next_attempt(&faults);
        let (result, captured) = capture(chained(
            &svc,
            &target,
            "retry-s",
            "retry-p",
            false,
            cell.payload(2),
        ))
        .await;

        assert!(
            intruded.load(SeqCst) && faults.lock().unwrap().is_empty(),
            "{route:?}: precondition: attempt 1 took the injected fault and the intruder landed \
             before attempt 2"
        );
        let status = result.expect_err("S's second attempt must be refused");
        assert_predecessor_refusal_after_a_retry(&status, 2, &format!("{route:?}"));
        let event = captured.only(RPC_FAILURE_MSG);
        assert_common_fields(event, "WARN", "UpdateLayer", "storage", "UNAVAILABLE");
        assert_field(event, "attempt", 2.into());
        assert_eq!(overtaken(&svc.metrics), 0, "{route:?}: never overtaken");
        assert_eq!(
            seq(&tv, target.object_id).await,
            3,
            "{route:?}: S applied nothing"
        );
        assert!(
            names(&record(&tv, target.object_id, WRITER).await, INTRUDER),
            "{route:?}: the intruder's write is the record"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn predecessor_refusal_after_a_counted_retry_is_plain_unavailable_on_the_fast_route() {
        predecessor_refusal_after_a_counted_retry_is_plain_unavailable(Route::Fast).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn predecessor_refusal_after_a_counted_retry_is_plain_unavailable_on_the_structural_route(
    ) {
        predecessor_refusal_after_a_counted_retry_is_plain_unavailable(Route::Structural).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn predecessor_refusal_after_a_counted_retry_is_plain_unavailable_on_the_watch_route() {
        predecessor_refusal_after_a_counted_retry_is_plain_unavailable(Route::Watch).await;
    }

    /// Group 14 (b), the re-issue: the twin of
    /// [`planning_invalid_argument_on_the_reissue_after_a_two_attempt_first_call_is_plain_unavailable`]
    /// for a predecessor refusal. P committed; S's first `update_layer` call
    /// makes two attempts — the first failing transiently, the second finding
    /// the header gone — so the object is restored from its checkpoint, which
    /// loses P's record, and the commit is re-issued once. The re-issue's
    /// commit refuses S by its predecessor guard on its own first attempt
    /// (`last_record=absent`), and the handler answers it a plain
    /// `UNAVAILABLE` naming all three attempts: the first call's retry is
    /// counted across the re-issue, so its possible lost commit leaves the
    /// outcome unknown. Nothing of S applied.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn predecessor_refusal_on_the_reissue_after_a_two_attempt_first_call_is_plain_unavailable(
    ) {
        let tv = TestValkey::setup().await;
        let target = new_target(&tv, "chain-reissue-pnc").await;
        let faults: FaultQueue = Arc::new(Mutex::new(VecDeque::new()));
        let header_key = tv.key_schema().obj_header(&target.object_id.to_string());
        let wiper = tv.client().clone();
        let wipe_before_attempt_2: BetweenAttemptsHook = Arc::new(move |attempt: u32| {
            let (wiper, header_key) = (wiper.clone(), header_key.clone());
            async move {
                if attempt == 2 {
                    let _: i64 = wiper
                        .pool
                        .del(header_key)
                        .await
                        .expect("DEL obj:hdr before attempt 2");
                }
            }
            .boxed()
        });
        let mut svc = service_over(&tv, tv.client().clone(), |store| {
            store
                .with_fault_injector_for_test(Arc::clone(&faults))
                .with_between_attempts_hook_for_test(wipe_before_attempt_2)
        });
        let metrics = crate::common::replace_metrics(&mut svc);
        let checkpoints = Arc::new(InMemoryCheckpointStore::new());
        crate::common::seed_layer_checkpoint(
            &checkpoints,
            target.object_id,
            &target.session_id,
            LAYER,
            1,
        )
        .await;
        let mut config = crate::common::test_config();
        config.checkpoint_enabled = true;
        config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
        svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
            checkpoints as Arc<dyn CheckpointStore>,
            Arc::clone(&svc.layer_store),
            Arc::new(config),
            Arc::clone(&svc.metrics),
        ));
        assert_applied(
            &chained(&svc, &target, "reissue-p", "", false, field_ompf(1)).await,
            2,
            "P, the predecessor",
        );

        fail_next_attempt(&faults);
        let status = chained(
            &svc,
            &target,
            "reissue-s",
            "reissue-p",
            false,
            field_ompf(2),
        )
        .await
        .expect_err("the re-issue's predecessor refusal must fail the call");

        assert_eq!(
            crate::common::recovery_metric(&metrics, "restored"),
            1,
            "precondition: the second attempt found the header gone and the restore landed"
        );
        assert_predecessor_refusal_after_a_retry(&status, 3, "the re-issue");
        assert_eq!(
            record(&tv, target.object_id, WRITER).await,
            None,
            "the restore left no record, and S, which would have recorded its own operation, \
             applied nothing to the restored object"
        );
    }

    // ── Group 18 (b): the re-issue answered from a client's re-send ─────────

    /// The commit tokens a group 18 (b) service pins: its first call's, then
    /// its recovery re-issue's.
    const FIRST_CALL_TOKEN: u128 = 0x18b0_0000_0000_0000_0000_0000_0000_0001;
    const REISSUE_TOKEN: u128 = 0x18b0_0000_0000_0000_0000_0000_0000_0002;

    /// The operation under test: a chain's first write, so the client's
    /// re-send of it passes the gate on the restored object, which a restore
    /// leaves with no record.
    const RESENT_S: &str = "resent-s";

    /// What the client's re-send carries, against the request the re-issue
    /// repeats.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Resend {
        /// The same bytes: the re-issue's verdict is earlier-call, answered
        /// `ALREADY_COMMITTED` at the re-send's `seq`.
        Same,
        /// Other bytes under the same `operation_id`: the verdict is
        /// conflict, answered `ALREADY_EXISTS`.
        Different,
    }

    /// The client's re-send, landed through a peer replica at the first
    /// commit dispatch after the object is lost and restored — the re-issue's
    /// first attempt, before it reads the record — and only once.
    struct ResendStage {
        header_key: String,
        probe: ValkeyClient,
        peer: Arc<LiveEditService>,
        target: Target,
        payload: Vec<u8>,
        /// Set by the between-attempts hook once it deleted the header.
        lost: AtomicBool,
        /// The commit seam that landed the re-send: `fast`, `structural` or
        /// `watch`, naming the route the re-issue's attempt took.
        site: Mutex<Option<&'static str>>,
        resent: Mutex<Option<Result<UpdateLayerResponse, Status>>>,
    }

    impl ResendStage {
        /// The seam `site` of a commit route: the fast route's and the
        /// structural route's pre-dispatch hooks, and the WATCH window hook.
        fn hook(self: &Arc<Self>, site: &'static str) -> WatchWindowHook {
            let this = Arc::clone(self);
            Arc::new(move |_: &Client| {
                let this = Arc::clone(&this);
                async move { this.land(site).await }.boxed()
            })
        }

        /// Land the re-send if the object was lost and is back (restored),
        /// and no seam landed it yet. The first call's attempts reach here
        /// before the loss or with the header gone, and pass.
        async fn land(&self, site: &'static str) {
            if !self.lost.load(SeqCst) || self.site.lock().unwrap().is_some() {
                return;
            }
            let present: i64 = self
                .probe
                .pool
                .exists(self.header_key.as_str())
                .await
                .expect("EXISTS obj:hdr");
            if present == 0 {
                return;
            }
            *self.site.lock().unwrap() = Some(site);
            let (peer, target, payload) = (
                Arc::clone(&self.peer),
                self.target.clone(),
                self.payload.clone(),
            );
            let resent =
                tokio::spawn(
                    async move { chained(&peer, &target, RESENT_S, "", false, payload).await },
                )
                .await
                .expect("the re-send's task must not panic");
            *self.resent.lock().unwrap() = Some(resent);
        }
    }

    /// Group 18 (b) on `route`: a recovery re-issue answered from a client's
    /// re-send hands back its own pre-flight increment. S, a chain's first
    /// write referencing one external value H, pre-stores H for its first
    /// call. That call's attempt 1 commits — H's reference is now the
    /// object's — and loses its reply; before attempt 2 the object's header
    /// is deleted, so attempt 2 ends `NOT_FOUND` and the handler restores the
    /// object from its checkpoint (`seq` 1, without H), releasing the landed
    /// reference (`before − after` over `obj:ext`). The handler then
    /// pre-stores H again, for the re-issue (b.5wb). At the re-issue's first
    /// commit dispatch the client's re-send of S — `resend`'s bytes, also
    /// referencing H, through another replica — commits on the restored
    /// object, so the re-issue is answered from that earlier call's record:
    ///
    /// - the same bytes ([`Resend::Same`]): `ALREADY_COMMITTED` at the
    ///   re-send's `seq`;
    /// - other bytes ([`Resend::Different`]): `ALREADY_EXISTS`.
    ///
    /// Either answer is the same as before the gate, and either verdict
    /// hands back the re-issue's own pre-flight increment, as it would any
    /// call's: exactly the `Resend(<re-issue token>)` member is pending,
    /// none for the first call's token, whose increment its landed commit
    /// spent. Before GC H counts four — the reference elsewhere, the first
    /// call's spent increment, the re-issue's and the re-send's — and after
    /// GC (the restore's release and the hand-back) exactly its live
    /// references: the one held elsewhere and the re-send's.
    ///
    /// Fail-on-old: before b.5wb the re-issue reused the first call's
    /// pre-store and handed back nothing on these verdicts, so no
    /// `Resend(<re-issue token>)` member was pending.
    async fn reissue_answered_from_a_client_resend_hands_back_its_own(
        route: Route,
        resend: Resend,
    ) {
        let what = format!("{route:?}, {resend:?}");
        let cell = Cell::of(route);
        let tv = TestValkey::setup().await;
        let target = new_target(&tv, &format!("resend-reissue-{route:?}-{resend:?}")).await;
        let blob = external_blob();
        let hash = live_edit::storage::external_values::ext_content_hash(&blob);
        let ext = Arc::new(ValkeyExternalValueStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        ext.store_values(&[(&hash, blob.as_slice())])
            .await
            .expect("the reference held elsewhere must be stored");
        assert_eq!(
            ref_count(&tv, &hash).await,
            Some(1),
            "{what}: precondition: H's one reference, held elsewhere"
        );

        let payload = cell.ext_payload(2);
        let resent_payload = match resend {
            Resend::Same => payload.clone(),
            Resend::Different => external_value_ompf(vec![create_section(90)]),
        };
        let peer = Arc::new(LiveEditService {
            ext_value_store: Arc::clone(&ext) as Arc<dyn ExternalValueStore>,
            ..peer_replica(&tv, |store| store).await
        });
        let header_key = tv.key_schema().obj_header(&target.object_id.to_string());
        let stage = Arc::new(ResendStage {
            header_key: header_key.clone(),
            probe: tv.client().clone(),
            peer,
            target: target.clone(),
            payload: resent_payload,
            lost: AtomicBool::new(false),
            site: Mutex::new(None),
            resent: Mutex::new(None),
        });

        // Attempt 2 of the first call: check attempt 1 landed (the record
        // names S under the first call's token), then lose the object.
        let first_landed = Arc::new(AtomicBool::new(false));
        let lose_object: BetweenAttemptsHook = {
            let (stage, client, wtok, landed) = (
                Arc::clone(&stage),
                tv.client().clone(),
                tv.key_schema()
                    .obj_write_token(&target.object_id.to_string()),
                Arc::clone(&first_landed),
            );
            Arc::new(move |attempt: u32| {
                let (stage, client, wtok, header_key, landed) = (
                    Arc::clone(&stage),
                    client.clone(),
                    wtok.clone(),
                    header_key.clone(),
                    Arc::clone(&landed),
                );
                async move {
                    if attempt != 2 || stage.lost.load(SeqCst) {
                        return;
                    }
                    let latest: Option<String> = client
                        .pool
                        .hget(wtok, WRITER)
                        .await
                        .expect("HGET obj:wtok before attempt 2");
                    landed.store(
                        names_with_token(&latest, RESENT_S, FIRST_CALL_TOKEN),
                        SeqCst,
                    );
                    let _: i64 = client
                        .pool
                        .del(header_key)
                        .await
                        .expect("DEL obj:hdr before attempt 2");
                    stage.lost.store(true, SeqCst);
                }
                .boxed()
            })
        };

        // The first call's attempt 1 commits and loses its reply: the
        // post-`EVALSHA` injector on the `lua` routes; on the WATCH path an
        // in-doubt `EXEC` whose reconcile read fails too.
        let timeout = || {
            Arc::new(Mutex::new(VecDeque::from([fred::error::Error::new(
                fred::error::ErrorKind::Timeout,
                "staged: reply lost",
            )])))
        };
        let (post_eval, reconcile) = (timeout(), timeout());
        let exec: WatchedExecOverrideQueue = Arc::new(Mutex::new(VecDeque::from([
            WatchedExecOverride::ReplaceOutcomeAfterExec,
        ])));
        let tokens = Arc::new(Mutex::new(VecDeque::from([
            FIRST_CALL_TOKEN,
            REISSUE_TOKEN,
        ])));
        let mut svc = service_over(&tv, tv.client().clone(), |store| {
            let store = cell
                .tune(store)
                .with_commit_tokens_for_test(tokens)
                .with_between_attempts_hook_for_test(lose_object)
                .with_lua_commit_pre_dispatch_hook_for_test(stage.hook("fast"))
                .with_structural_commit_pre_dispatch_hook_for_test(stage.hook("structural"))
                .with_watch_window_hook_for_test(stage.hook("watch"));
            match route {
                Route::Fast | Route::Structural => {
                    store.with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&post_eval))
                }
                Route::Watch => store
                    .with_watched_exec_override_for_test(Arc::clone(&exec))
                    .with_reconcile_xrange_fred_error_injector_for_test(Arc::clone(&reconcile)),
            }
        });
        svc.ext_value_store = Arc::clone(&ext) as Arc<dyn ExternalValueStore>;
        let metrics = crate::common::replace_metrics(&mut svc);
        let checkpoints = Arc::new(InMemoryCheckpointStore::new());
        crate::common::seed_layer_checkpoint(
            &checkpoints,
            target.object_id,
            &target.session_id,
            LAYER,
            1,
        )
        .await;
        let mut config = crate::common::test_config();
        config.checkpoint_enabled = true;
        config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
        svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
            checkpoints as Arc<dyn CheckpointStore>,
            Arc::clone(&svc.layer_store),
            Arc::new(config),
            Arc::clone(&svc.metrics),
        ));

        let result = settled(
            test_deadline(),
            "S",
            chained(&svc, &target, RESENT_S, "", false, payload),
        )
        .await;

        let drained = match route {
            Route::Fast | Route::Structural => post_eval.lock().unwrap().is_empty(),
            Route::Watch => exec.lock().unwrap().is_empty() && reconcile.lock().unwrap().is_empty(),
        };
        assert!(
            drained && first_landed.load(SeqCst),
            "{what}: precondition: the first call's attempt 1 committed under its token and lost \
             its reply"
        );
        assert_eq!(
            crate::common::recovery_metric(&metrics, "restored"),
            1,
            "{what}: precondition: attempt 2 found the object gone and the restore landed"
        );
        let expected_site = match route {
            Route::Fast => "fast",
            Route::Structural => "structural",
            Route::Watch => "watch",
        };
        assert_eq!(
            *stage.site.lock().unwrap(),
            Some(expected_site),
            "{what}: precondition: the re-send landed on the restored object at the re-issue's \
             first commit dispatch, on the route the cell names"
        );
        let resent = stage
            .resent
            .lock()
            .unwrap()
            .take()
            .expect("the seam that fired stored the re-send's answer");
        let resent_seq = match &resent {
            Ok(response) if outcome(response) == OperationOutcome::Applied => response.sequence_num,
            other => panic!("{what}: precondition: the re-send commits; got {other:?}"),
        };
        match resend {
            Resend::Same => {
                let response = result.unwrap_or_else(|s| {
                    panic!("{what}: the re-issue is answered from the re-send; got {s:?}")
                });
                assert_eq!(
                    (response.sequence_num, outcome(&response)),
                    (resent_seq, OperationOutcome::AlreadyCommitted),
                    "{what}: the earlier-call answer is unchanged"
                );
            }
            Resend::Different => {
                let status = result.expect_err("the re-issue must be refused as a conflict");
                assert_eq!(status.code(), Code::AlreadyExists, "{what}: {status:?}");
                assert_eq!(
                    status.message(),
                    live_edit::storage::layer::OPERATION_CONFLICT_MESSAGE,
                    "{what}: the conflict answer is unchanged"
                );
            }
        }
        assert_eq!(
            seq(&tv, target.object_id).await,
            resent_seq,
            "{what}: the re-issue applied nothing after the re-send"
        );
        let latest = record(&tv, target.object_id, WRITER).await;
        assert!(
            names(&latest, RESENT_S)
                && !names_with_token(&latest, RESENT_S, FIRST_CALL_TOKEN)
                && !names_with_token(&latest, RESENT_S, REISSUE_TOKEN),
            "{what}: the record is the re-send's; got {latest:?}"
        );

        let pending = pending_members(&tv).await;
        let resend_members: Vec<&String> = pending.iter().filter(|m| m.contains(":rsd")).collect();
        assert_eq!(
            resend_members,
            vec![&ExtRelease::resend(hash, target.object_id, REISSUE_TOKEN).member()],
            "{what}: the recovery re-issue answered from an earlier call's record hands back its \
             own pre-flight increment — its Resend member alone, none for the first call's \
             token; pending {pending:?}"
        );
        assert_eq!(
            ref_count(&tv, &hash).await,
            Some(4),
            "{what}: before GC H counts the reference elsewhere, the first call's spent \
             increment, the re-issue's and the re-send's"
        );
        process_pending_gc(ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
        assert_eq!(
            ref_count(&tv, &hash).await,
            Some(2),
            "{what}: after GC H counts its live references, the one held elsewhere and the \
             re-send's on the restored object (pending before GC: {pending:?})"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_answered_from_a_client_resend_hands_back_its_own_preflight_increment_on_the_fast_route(
    ) {
        for resend in [Resend::Same, Resend::Different] {
            reissue_answered_from_a_client_resend_hands_back_its_own(Route::Fast, resend).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_answered_from_a_client_resend_hands_back_its_own_preflight_increment_on_the_structural_route(
    ) {
        for resend in [Resend::Same, Resend::Different] {
            reissue_answered_from_a_client_resend_hands_back_its_own(Route::Structural, resend)
                .await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_answered_from_a_client_resend_hands_back_its_own_preflight_increment_on_the_watch_route(
    ) {
        for resend in [Resend::Same, Resend::Different] {
            reissue_answered_from_a_client_resend_hands_back_its_own(Route::Watch, resend).await;
        }
    }

    // ── b.5wb: the recovery re-issue pays for its own pre-store ────────────
    //
    // The first call of an `UpdateLayer` RPC commits H's reference and loses
    // its reply; the object is lost before its attempt 2, which ends
    // `NOT_FOUND`; the handler restores the object from its checkpoint,
    // pre-stores H again for the re-issue, and re-issues the commit, which
    // commits. Whatever the restore kept or released, H must count at least
    // its live references after GC: exactly them where the accounting can be
    // exact (A, B, E), one more where the first call's increment is orphaned
    // (C, D) — never fewer, which would let GC delete a blob a live layer
    // references. Before b.5wb the re-issue spent the first call's
    // increment a second time, so A, B and E counted one fewer than their
    // live references.

    /// What a re-issue scenario loses before the first call's attempt 2.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Loss {
        /// `DEL obj:hdr`: the object's other keys — its `obj:ext` index among
        /// them — survive, so the restore releases what that index holds and
        /// the restored state lacks.
        Header,
        /// Every `obj:*` key of the object: the restore finds no index and
        /// releases nothing.
        ObjectKeys,
    }

    /// The checkpoint a re-issue scenario's restore selects.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Restored {
        /// A base at `seq` 1, from before the first call: without H.
        Without,
        /// A base assembled from the object right after the first call's
        /// commit, alone: H on its row, and nothing pre-stored by the
        /// restore (a base-only restore pre-stores nothing).
        BaseWith,
        /// The base at `seq` 1 and a tail record of the first call's commit:
        /// a composed lineage, whose restore pre-stores H itself.
        ComposedWith,
    }

    /// One re-issue scenario.
    #[derive(Clone, Copy, Debug)]
    struct Reissue {
        route: Route,
        /// References to H held outside the object.
        elsewhere: i64,
        loss: Loss,
        restored: Restored,
    }

    /// The operation every re-issue scenario's call carries: a chain's first
    /// write, which the restored object — holding no record — lets commit.
    const REISSUED_S: &str = "reissued-s";

    /// The object a reference to H held elsewhere is released from.
    const ELSEWHERE_OBJECT: u64 = 0xE15E;

    /// A re-issue scenario's payload on `route`: one external value H, plus,
    /// on the structural route, a `DiffChangeSection` of section 62 — an
    /// upsert, so the re-issue applies whether or not the restored object
    /// already holds the section the first call created.
    fn reissue_payload(route: Route) -> Vec<u8> {
        external_value_ompf(match route {
            Route::Structural => vec![StructCommand::DiffChangeSection(
                oodf_ffi::CommandDiffChangeSection {
                    parent_id: 0,
                    section_id: 62,
                    section_name: "s62".to_owned(),
                    section_type: 1,
                    section_order: 0,
                },
            )],
            Route::Fast | Route::Watch => vec![],
        })
    }

    /// Records the route the re-issue's first commit dispatch takes: the
    /// first dispatch after the object was lost that finds its header back.
    struct ReissueRouteProbe {
        header_key: String,
        probe: ValkeyClient,
        lost: AtomicBool,
        site: Mutex<Option<&'static str>>,
    }

    impl ReissueRouteProbe {
        fn hook(self: &Arc<Self>, site: &'static str) -> WatchWindowHook {
            let this = Arc::clone(self);
            Arc::new(move |_: &Client| {
                let this = Arc::clone(&this);
                async move { this.saw(site).await }.boxed()
            })
        }

        async fn saw(&self, site: &'static str) {
            if !self.lost.load(SeqCst) || self.site.lock().unwrap().is_some() {
                return;
            }
            let present: i64 = self
                .probe
                .pool
                .exists(self.header_key.as_str())
                .await
                .expect("EXISTS obj:hdr");
            if present == 1 {
                *self.site.lock().unwrap() = Some(site);
            }
        }
    }

    /// `live_edit_ext_value_verify_total{result="ok"}`.
    fn verified_ok(metrics: &Metrics) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_ext_value_verify_total",
            &[("result", "ok")],
        )
        .expect("the ok cell is seeded") as u64
    }

    /// What [`reissue_after_a_restore`] leaves to assert on.
    struct Reissued {
        tv: TestValkey,
        target: Target,
        hash: [u8; 32],
        ext: Arc<ValkeyExternalValueStore>,
        result: Result<UpdateLayerResponse, Status>,
        /// `live_edit_ext_value_verify_total{result="ok"}` gained during the
        /// RPC.
        verified: u64,
    }

    /// Run `case`: S (a chain's first write referencing H, [`REISSUED_S`])
    /// through a service whose ext store is a Valkey store counting into the
    /// service's registry, with `case.elsewhere` references to H held outside
    /// the object. The first call's attempt 1 commits under
    /// [`FIRST_CALL_TOKEN`] and loses its reply; before attempt 2 the
    /// checkpoint `case.restored` names is put in place and `case.loss`
    /// loses the object; attempt 2 ends `NOT_FOUND`, the restore lands, and
    /// the re-issue runs under [`REISSUE_TOKEN`]. Asserts every precondition:
    /// the reply was lost after the first commit landed, one restore landed
    /// from the lineage `case.restored` names, and the re-issue dispatched on
    /// `case.route`.
    async fn reissue_after_a_restore(case: Reissue) -> Reissued {
        use crate::checkpoint::helpers::{
            put_tail_generation_from_records, read_restore_lineage_counter,
            HAND_PUT_WRITTEN_AT_UNIX_MS,
        };
        use live_edit::checkpoint::assembler::{
            assemble_layer_checkpoint, AssemblyDecision, NoLeaseToKeep,
        };

        let Reissue {
            route,
            elsewhere,
            loss,
            restored,
        } = case;
        let what = format!("{case:?}");
        let cell = Cell::of(route);
        let tv = TestValkey::setup().await;
        let target = new_target(
            &tv,
            &format!("reissue-{route:?}-{elsewhere}-{loss:?}-{restored:?}"),
        )
        .await;
        let blob = external_blob();
        let hash = live_edit::storage::external_values::ext_content_hash(&blob);
        let payload = reissue_payload(route);

        let checkpoints = Arc::new(InMemoryCheckpointStore::new());
        if restored != Restored::BaseWith {
            crate::common::seed_layer_checkpoint(
                &checkpoints,
                target.object_id,
                &target.session_id,
                LAYER,
                1,
            )
            .await;
        }
        let header_key = tv.key_schema().obj_header(&target.object_id.to_string());
        let probe = Arc::new(ReissueRouteProbe {
            header_key: header_key.clone(),
            probe: tv.client().clone(),
            lost: AtomicBool::new(false),
            site: Mutex::new(None),
        });

        // Before attempt 2 of the first call: check attempt 1 landed (the
        // record names S under the first call's token), put the checkpoint
        // in place, then lose the object.
        let first_landed = Arc::new(AtomicBool::new(false));
        let lose_object: BetweenAttemptsHook = {
            let (probe, client, keys, prefix, target, checkpoints, payload, landed) = (
                Arc::clone(&probe),
                tv.client().clone(),
                tv.key_schema().clone(),
                tv.prefix().to_owned(),
                target.clone(),
                Arc::clone(&checkpoints),
                payload.clone(),
                Arc::clone(&first_landed),
            );
            Arc::new(move |attempt: u32| {
                let (probe, client, keys, prefix, target, checkpoints, payload, landed) = (
                    Arc::clone(&probe),
                    client.clone(),
                    keys.clone(),
                    prefix.clone(),
                    target.clone(),
                    Arc::clone(&checkpoints),
                    payload.clone(),
                    Arc::clone(&landed),
                );
                async move {
                    if attempt != 2 || probe.lost.load(SeqCst) {
                        return;
                    }
                    let oid = target.object_id;
                    let latest: Option<String> = client
                        .pool
                        .hget(keys.obj_write_token(&oid.to_string()), WRITER)
                        .await
                        .expect("HGET obj:wtok before attempt 2");
                    landed.store(
                        names_with_token(&latest, REISSUED_S, FIRST_CALL_TOKEN),
                        SeqCst,
                    );
                    match restored {
                        Restored::Without => {}
                        Restored::BaseWith => {
                            let decision = assemble_layer_checkpoint(
                                &client,
                                &keys,
                                oid,
                                checkpoints.as_ref(),
                                crate::common::RECOVERY_SERVICE_LABEL,
                                &mut NoLeaseToKeep,
                            )
                            .await
                            .expect("the base after the first commit assembles");
                            match decision {
                                AssemblyDecision::Written(written) => assert_eq!(
                                    written.snap_seq, 2,
                                    "the base holds the first call's commit"
                                ),
                                other => panic!("the base must be written; got {other:?}"),
                            }
                        }
                        Restored::ComposedWith => {
                            put_tail_generation_from_records(
                                &(Arc::clone(&checkpoints) as Arc<dyn CheckpointStore>),
                                oid,
                                &target.session_id,
                                LAYER,
                                1,
                                1,
                                &[(2, Bytes::from(payload))],
                                HAND_PUT_WRITTEN_AT_UNIX_MS,
                                0,
                            )
                            .await;
                        }
                    }
                    match loss {
                        Loss::Header => {
                            let _: i64 = client
                                .pool
                                .del(keys.obj_header(&oid.to_string()))
                                .await
                                .expect("DEL obj:hdr before attempt 2");
                        }
                        Loss::ObjectKeys => {
                            let lost = crate::common::scan_object_keys(&client, &prefix, oid).await;
                            let _: () = client
                                .pool
                                .unlink(lost)
                                .await
                                .expect("UNLINK the object's keys before attempt 2");
                            assert!(
                                crate::common::scan_object_keys(&client, &prefix, oid)
                                    .await
                                    .is_empty(),
                                "every obj:* key of the object is gone"
                            );
                        }
                    }
                    probe.lost.store(true, SeqCst);
                }
                .boxed()
            })
        };

        // The first call's attempt 1 commits and loses its reply, as in
        // group 18 (b).
        let timeout = || {
            Arc::new(Mutex::new(VecDeque::from([fred::error::Error::new(
                fred::error::ErrorKind::Timeout,
                "staged: reply lost",
            )])))
        };
        let (post_eval, reconcile) = (timeout(), timeout());
        let exec: WatchedExecOverrideQueue = Arc::new(Mutex::new(VecDeque::from([
            WatchedExecOverride::ReplaceOutcomeAfterExec,
        ])));
        let tokens = Arc::new(Mutex::new(VecDeque::from([
            FIRST_CALL_TOKEN,
            REISSUE_TOKEN,
        ])));
        let mut svc = service_over(&tv, tv.client().clone(), |store| {
            let store = cell
                .tune(store)
                .with_commit_tokens_for_test(tokens)
                .with_between_attempts_hook_for_test(lose_object)
                .with_lua_commit_pre_dispatch_hook_for_test(probe.hook("fast"))
                .with_structural_commit_pre_dispatch_hook_for_test(probe.hook("structural"))
                .with_watch_window_hook_for_test(probe.hook("watch"));
            match route {
                Route::Fast | Route::Structural => {
                    store.with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&post_eval))
                }
                Route::Watch => store
                    .with_watched_exec_override_for_test(Arc::clone(&exec))
                    .with_reconcile_xrange_fred_error_injector_for_test(Arc::clone(&reconcile)),
            }
        });
        let metrics = crate::common::replace_metrics(&mut svc);
        let ext = Arc::new(
            ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_metrics(Arc::clone(&metrics)),
        );
        svc.ext_value_store = Arc::clone(&ext) as Arc<dyn ExternalValueStore>;
        for _ in 0..elsewhere {
            ext.store_values(&[(&hash, blob.as_slice())])
                .await
                .expect("a reference held elsewhere must be stored");
        }
        let mut config = crate::common::test_config();
        config.checkpoint_enabled = true;
        config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
        svc.recovery = Arc::new(
            live_edit::checkpoint::RecoveryHook::new(
                checkpoints as Arc<dyn CheckpointStore>,
                Arc::clone(&svc.layer_store),
                Arc::new(config),
                Arc::clone(&svc.metrics),
            )
            .with_ext_value_store(Arc::clone(&ext) as Arc<dyn ExternalValueStore>),
        );
        let verified_before = verified_ok(&metrics);

        let result = settled(
            test_deadline(),
            "S",
            chained(&svc, &target, REISSUED_S, "", false, payload),
        )
        .await;

        let drained = match route {
            Route::Fast | Route::Structural => post_eval.lock().unwrap().is_empty(),
            Route::Watch => exec.lock().unwrap().is_empty() && reconcile.lock().unwrap().is_empty(),
        };
        assert!(
            drained && first_landed.load(SeqCst),
            "{what}: precondition: the first call's attempt 1 committed under its token and lost \
             its reply"
        );
        assert_eq!(
            crate::common::recovery_metric(&metrics, "restored"),
            1,
            "{what}: precondition: attempt 2 found the object gone and the restore landed"
        );
        let lineage = match restored {
            Restored::ComposedWith => "gathered_composed",
            Restored::Without | Restored::BaseWith => "gathered_base",
        };
        assert_eq!(
            read_restore_lineage_counter(&metrics, lineage),
            1,
            "{what}: precondition: the restore selected the lineage the case names"
        );
        let expected_site = match route {
            Route::Fast => "fast",
            Route::Structural => "structural",
            Route::Watch => "watch",
        };
        assert_eq!(
            *probe.site.lock().unwrap(),
            Some(expected_site),
            "{what}: precondition: the re-issue dispatched on the route the case names"
        );
        let verified = verified_ok(&metrics) - verified_before;
        Reissued {
            tv,
            target,
            hash,
            ext,
            result,
            verified,
        }
    }

    /// Run `case`, whose re-issue commits, and assert the accounting: the
    /// re-issue commits `APPLIED` under its own token; no call hands back
    /// (no `Resend` member for either token); the upload's one reference was
    /// checked once (`live_edit_ext_value_verify_total{result="ok"}` +1, the
    /// re-pre-store checking nothing again); `owed` releases of H on the
    /// object are pending (the restore's, or the re-issue's surplus
    /// hand-back with its commit); H counts `before_gc` before GC and
    /// `after_gc` after it, with its data present; and, with a reference
    /// held elsewhere, releasing it leaves H counted and its data present.
    async fn reissue_commits_and_counts(case: Reissue, owed: usize, before_gc: i64, after_gc: i64) {
        let what = format!("{case:?}");
        let run = reissue_after_a_restore(case).await;
        let response = run
            .result
            .as_ref()
            .unwrap_or_else(|s| panic!("{what}: the re-issue commits; got {s:?}"));
        assert_eq!(
            outcome(response),
            OperationOutcome::Applied,
            "{what}: the re-issue's own commit"
        );
        assert!(
            names_with_token(
                &record(&run.tv, run.target.object_id, WRITER).await,
                REISSUED_S,
                REISSUE_TOKEN
            ),
            "{what}: the record is the re-issue's commit"
        );
        assert_eq!(
            run.verified, 1,
            "{what}: the upload's one external-value reference was checked once, not again by \
             the re-pre-store"
        );
        let pending = pending_members(&run.tv).await;
        for token in [FIRST_CALL_TOKEN, REISSUE_TOKEN] {
            let member = ExtRelease::resend(run.hash, run.target.object_id, token).member();
            assert!(
                !pending.contains(&member),
                "{what}: a committing re-issue hands nothing back; {member} is pending in \
                 {pending:?}"
            );
        }
        let of_h = format!("{}:{}:", sha256_to_hex(&run.hash), run.target.object_id);
        assert_eq!(
            pending.iter().filter(|m| m.starts_with(&of_h)).count(),
            owed,
            "{what}: the releases of H on the object pending before GC; got {pending:?}"
        );
        assert_eq!(
            ref_count(&run.tv, &run.hash).await,
            Some(before_gc),
            "{what}: H before GC (pending {pending:?})"
        );
        process_pending_gc(run.ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
        let data_key = run.tv.key_schema().ext_data(&sha256_to_hex(&run.hash));
        let data_present = |tv: &TestValkey| {
            let (client, key) = (tv.client().clone(), data_key.clone());
            async move {
                let present: i64 = client.pool.exists(key).await.expect("EXISTS ext:data");
                present == 1
            }
        };
        assert_eq!(
            ref_count(&run.tv, &run.hash).await,
            Some(after_gc),
            "{what}: H after GC (pending before GC {pending:?})"
        );
        assert!(
            data_present(&run.tv).await,
            "{what}: H's data survives GC: the restored layer references it"
        );
        if case.elsewhere > 0 {
            run.ext
                .schedule_release(&[ExtRelease::retirement(run.hash, ELSEWHERE_OBJECT)])
                .await
                .expect("release the reference held elsewhere");
            process_pending_gc(run.ext.as_ref(), None, GcPolicy::IMMEDIATE).await;
            assert_eq!(
                ref_count(&run.tv, &run.hash).await,
                Some(after_gc - 1),
                "{what}: H after the reference elsewhere is released"
            );
            assert!(
                data_present(&run.tv).await,
                "{what}: H's data survives the release of the reference elsewhere: the restored \
                 layer still references it"
            );
        }
    }

    /// b.5wb A on `route`, with zero and with one reference to H held
    /// elsewhere: the restore, from a checkpoint without H, releases the
    /// first call's landed reference (its member pending before GC); the
    /// re-issue pre-stores H for itself and commits. Before GC H counts E +
    /// 2 (the first call's spent increment and the re-issue's), after GC
    /// E + 1, exactly its live references, and its data stays; with E = 1,
    /// releasing the reference elsewhere leaves 1 and the data.
    ///
    /// Fail-on-old: the re-issue spent the first call's increment again, so
    /// E = 1 counted 1 after GC and the release deleted the blob, and E = 0
    /// deleted it at the first GC.
    async fn reissue_after_a_restore_without_the_value(route: Route) {
        for elsewhere in [0, 1] {
            let case = Reissue {
                route,
                elsewhere,
                loss: Loss::Header,
                restored: Restored::Without,
            };
            // Owed: the restore's release of the first call's reference.
            reissue_commits_and_counts(case, 1, elsewhere + 2, elsewhere + 1).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_a_restore_counts_every_live_reference_on_the_fast_route(
    ) {
        reissue_after_a_restore_without_the_value(Route::Fast).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_a_restore_counts_every_live_reference_on_the_structural_route(
    ) {
        reissue_after_a_restore_without_the_value(Route::Structural).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_a_restore_counts_every_live_reference_on_the_watch_route(
    ) {
        reissue_after_a_restore_without_the_value(Route::Watch).await;
    }

    /// b.5wb B (variant (i), base-only) on `route`, with zero and with one
    /// reference elsewhere: the restore's base, assembled right after the
    /// first call's commit, already holds H, so the restore releases
    /// nothing; the re-issue pre-stores H and commits onto an object that
    /// holds it, so its increment is surplus and handed back with its commit
    /// (`surplus_acquire`). Before GC E + 2, after GC E + 1, exactly its live
    /// references.
    ///
    /// Fail-on-old: the surplus hand-back gave back the first call's
    /// increment, which its commit had spent: E after GC, one below.
    async fn reissue_after_a_base_holding_the_value(route: Route) {
        for elsewhere in [0, 1] {
            let case = Reissue {
                route,
                elsewhere,
                loss: Loss::Header,
                restored: Restored::BaseWith,
            };
            // Owed: the re-issue's surplus hand-back.
            reissue_commits_and_counts(case, 1, elsewhere + 2, elsewhere + 1).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_a_base_holding_the_value_counts_every_live_reference_on_the_fast_route(
    ) {
        reissue_after_a_base_holding_the_value(Route::Fast).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_a_base_holding_the_value_counts_every_live_reference_on_the_structural_route(
    ) {
        reissue_after_a_base_holding_the_value(Route::Structural).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_a_base_holding_the_value_counts_every_live_reference_on_the_watch_route(
    ) {
        reissue_after_a_base_holding_the_value(Route::Watch).await;
    }

    /// b.5wb E, fast route: every `obj:*` key of the object is lost, its
    /// `obj:ext` index with it, and the restore's base holds H: the restore
    /// releases nothing and materializes H, and the re-issue's increment is
    /// surplus and handed back. With one reference elsewhere: 3 before GC,
    /// 2 after, exactly its live references.
    ///
    /// Fail-on-old: 1 after GC, one below.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_an_object_keyspace_loss_and_a_base_holding_the_value_counts_every_live_reference(
    ) {
        let case = Reissue {
            route: Route::Fast,
            elsewhere: 1,
            loss: Loss::ObjectKeys,
            restored: Restored::BaseWith,
        };
        // Owed: the re-issue's surplus hand-back.
        reissue_commits_and_counts(case, 1, 3, 2).await;
    }

    /// b.5wb D, fast route, a leak-direction pin: every `obj:*` key of the
    /// object is lost and the restore's base lacks H, so nothing releases
    /// the first call's landed reference, which went with the object; the
    /// re-issue's commit spends its own increment. With one reference
    /// elsewhere: 3 before and after GC — the live references (2) plus the
    /// orphaned increment, never fewer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_an_object_keyspace_loss_and_a_base_without_the_value_leaks_one_increment(
    ) {
        let case = Reissue {
            route: Route::Fast,
            elsewhere: 1,
            loss: Loss::ObjectKeys,
            restored: Restored::Without,
        };
        // Owed: nothing — no index survived to release from, and the
        // re-issue's commit spent its own increment.
        reissue_commits_and_counts(case, 0, 3, 3).await;
    }

    /// b.5wb C, fast route, a leak-direction pin: the restore composes the
    /// base at `seq` 1 with a tail record of the first call's commit, so it
    /// holds H and pre-stores H itself; the surviving `obj:ext` index also
    /// holds H, so it releases nothing; the re-issue's increment is surplus
    /// and handed back. With one reference elsewhere: 4 before GC, 3 after —
    /// the live references (2) plus the first call's increment, never fewer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_committing_after_a_composed_restore_holding_the_value_leaks_one_increment(
    ) {
        let case = Reissue {
            route: Route::Fast,
            elsewhere: 1,
            loss: Loss::Header,
            restored: Restored::ComposedWith,
        };
        // Owed: the re-issue's surplus hand-back.
        reissue_commits_and_counts(case, 1, 4, 3).await;
    }

    /// A refused re-pre-store ends the RPC before the re-issue (b.5wb). S
    /// carries H through an [`IntrudingExtStore`] over a Valkey store: its
    /// first store deletes the object's header, so the first call ends
    /// `NOT_FOUND`, after `first_call_attempts` attempts (a transient
    /// failure first when 2); the restore lands; the handler's second store
    /// of H, for the re-issue, is refused by the backing store
    /// (`STORE_WRITE_REFUSED`, out of memory). Asserts the restore landed
    /// and the staged failure was taken; returns the rest ([`Refused`]).
    async fn reissue_with_a_refused_pre_store(first_call_attempts: u32) -> Refused {
        use live_edit::storage::client::StoreWriteRefusal;

        let tv = TestValkey::setup().await;
        let target = new_target(&tv, &format!("reissue-refused-{first_call_attempts}")).await;
        let faults: FaultQueue = Arc::new(Mutex::new(VecDeque::new()));
        if first_call_attempts == 2 {
            fail_next_attempt(&faults);
        }
        let mut svc = service_over(&tv, tv.client().clone(), |store| {
            store.with_fault_injector_for_test(Arc::clone(&faults))
        });
        let ext = Arc::new(IntrudingExtStore::over(Arc::new(
            ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone()),
        )));
        let header_key = tv.key_schema().obj_header(&target.object_id.to_string());
        let wiper = tv.client().clone();
        *ext.intruder.lock().unwrap() = Some(Box::pin(async move {
            let _: i64 = wiper
                .pool
                .del(header_key)
                .await
                .expect("DEL obj:hdr at the first store");
        }));
        let restored_seq = Arc::new(std::sync::atomic::AtomicU64::new(0));
        {
            let (client, keys, oid, restored_seq) = (
                tv.client().clone(),
                tv.key_schema().clone(),
                target.object_id,
                Arc::clone(&restored_seq),
            );
            *ext.at_second.lock().unwrap() = Some(Box::pin(async move {
                let seq: Option<u64> = client
                    .pool
                    .hget(keys.obj_header(&oid.to_string()), "seq")
                    .await
                    .expect("HGET seq at the re-pre-store");
                restored_seq.store(
                    seq.expect("the restore landed before the re-pre-store"),
                    SeqCst,
                );
            }));
        }
        *ext.refuse_second.lock().unwrap() =
            Some(StoreWriteRefusal::OutOfMemory.into_service_error());
        svc.ext_value_store = Arc::clone(&ext) as Arc<dyn ExternalValueStore>;
        let metrics = crate::common::replace_metrics(&mut svc);
        let checkpoints = Arc::new(InMemoryCheckpointStore::new());
        crate::common::seed_layer_checkpoint(
            &checkpoints,
            target.object_id,
            &target.session_id,
            LAYER,
            1,
        )
        .await;
        let mut config = crate::common::test_config();
        config.checkpoint_enabled = true;
        config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
        svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
            checkpoints as Arc<dyn CheckpointStore>,
            Arc::clone(&svc.layer_store),
            Arc::new(config),
            Arc::clone(&svc.metrics),
        ));

        let (result, captured) = capture(chained(
            &svc,
            &target,
            REISSUED_S,
            "",
            false,
            reissue_payload(Route::Fast),
        ))
        .await;

        assert_eq!(
            crate::common::recovery_metric(&metrics, "restored"),
            1,
            "precondition: the first call ended NOT_FOUND and the restore landed"
        );
        assert!(
            faults.lock().unwrap().is_empty(),
            "precondition: the staged transient failure was taken"
        );
        let after = seq(&tv, target.object_id).await;
        Refused {
            result,
            captured,
            restored_seq: restored_seq.load(SeqCst),
            after,
            stores: ext.stores(),
        }
    }

    /// What [`reissue_with_a_refused_pre_store`] leaves to assert on.
    struct Refused {
        result: Result<UpdateLayerResponse, Status>,
        captured: super::super::log_capture::Captured,
        /// The object's `seq` when the re-pre-store was refused (the restored
        /// one); 0 when no second store ran.
        restored_seq: u64,
        /// The object's `seq` after the RPC.
        after: u64,
        /// The store calls that reached the external-value store.
        stores: usize,
    }

    /// After a first call of two attempts (a transient failure, then
    /// `NOT_FOUND`), a refused re-pre-store answers a plain `UNAVAILABLE`,
    /// outcome unknown, naming attempt 2 — the first call's two attempts and
    /// the re-issue's none — with no refusal detail: the first call's first
    /// attempt may have landed, so "this call applied nothing" cannot be
    /// promised. The re-issue made no attempt (the object stays at its
    /// restored `seq`), and the failure line carries the first call's
    /// summary (`attempt` 2).
    ///
    /// Fail-on-old: the RPC stored once, so nothing refused; the re-issue
    /// ran and committed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_whose_pre_store_is_refused_after_a_two_attempt_first_call_is_plain_unavailable(
    ) {
        let refused = reissue_with_a_refused_pre_store(2).await;

        let status = refused
            .result
            .expect_err("a refused re-pre-store must fail the RPC");
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        assert!(
            status.message().starts_with("UpdateLayer outcome unknown:")
                && status.message().contains("attempt 2 was refused")
                && status.message().contains("out of memory"),
            "the outcome-unknown answer names the first call's two attempts and the refusal; \
             got {status:?}"
        );
        assert_no_refusal_detail(&status);
        assert_eq!(
            refused.stores, 2,
            "precondition: the first call's pre-store and the refused re-pre-store"
        );
        assert_eq!(
            refused.after, refused.restored_seq,
            "the re-issue made no attempt: the object stays at its restored seq"
        );
        let event = refused.captured.only(RPC_FAILURE_MSG);
        assert_field(event, "attempt", 2.into());
    }

    /// After a first call of one attempt (`NOT_FOUND` at once, group 18's
    /// shape), a refused re-pre-store answers the refusal verbatim, its
    /// `STORE_WRITE_REFUSED` detail and retry hint included: no attempt of
    /// the RPC can have landed. The re-issue made no attempt, and the
    /// failure line carries the first call's summary (`attempt` 1).
    ///
    /// Fail-on-old: the re-issue ran and committed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovery_reissue_whose_pre_store_is_refused_after_a_one_attempt_first_call_answers_the_refusal(
    ) {
        use live_edit::error::{refusal_metadata, refusal_reason, STORE_WRITE_REFUSED_RETRY_DELAY};
        use live_edit::storage::client::StoreWriteRefusal;

        let refused = reissue_with_a_refused_pre_store(1).await;

        let status = refused
            .result
            .expect_err("a refused re-pre-store must fail the RPC");
        let refusal = StoreWriteRefusal::OutOfMemory;
        assert_eq!(
            status.message(),
            refusal
                .into_service_error()
                .to_string()
                .trim_start_matches("unavailable: "),
            "the refusal's message, verbatim"
        );
        assert_refusal_detail(
            &status,
            refusal_reason::STORE_WRITE_REFUSED,
            &[(refusal_metadata::CONDITION, refusal.reason_label())],
            STORE_WRITE_REFUSED_RETRY_DELAY,
        );
        assert_eq!(
            refused.stores, 2,
            "precondition: the first call's pre-store and the refused re-pre-store"
        );
        assert_eq!(
            refused.after, refused.restored_seq,
            "the re-issue made no attempt: the object stays at its restored seq"
        );
        let event = refused.captured.only(RPC_FAILURE_MSG);
        assert_field(event, "attempt", 1.into());
    }

    // ── P11 residual (rider 1), groups 5, 7 and 8b: a handed-over commit ────
    //
    // A client cancel cannot take back a commit fred has already written to
    // its connection: the store runs it whenever it reads it. Each test of
    // variant S writes a chained call's fast-route commit to a connection
    // that cannot run it yet (a server-side stall), drops the caller, moves
    // the chain on (or not), and then lets the commit run: it lands only while
    // the record is still its predecessor's, and never twice. A commit fred
    // has NOT yet written when its caller is dropped — variant R, queued in
    // fred's channel behind a reconnect — is never written at all (the
    // vendored fred's change 9). Fast route only: a late structural commit is
    // refused by its `seq` fence anyway, and a late WATCH-route `EXEC` that
    // runs unwatched tears against its `XADD` fence — the torn-commit residual
    // of SDD §7.5.2 / §18.2.

    /// Bound on each wait for a handed-over commit: to reach its connection,
    /// and to have run once let go.
    const HANDOVER_WAIT: Duration = Duration::from_secs(10);

    /// The commit tokens a [`HandoverFixture`] service pins, in call order.
    const FIRST_TOKEN: u128 = 0x1111_2222_3333_4444_5555_6666_7777_8888;
    const SECOND_TOKEN: u128 = 0x9999_aaaa_bbbb_cccc_dddd_eeee_ffff_0001;

    /// A client of the test's own whose data pool is ONE connection, so the
    /// connection a test stalls is the one every commit of its service rides,
    /// and the fixture's own client stays free for reads.
    async fn one_member_client() -> ValkeyClient {
        let mut config = crate::common::test_config();
        config.valkey_pool_size = 1;
        connect_test_client(&config)
            .await
            .expect("the one-member client must connect")
    }

    /// The fast route's pre-dispatch seam of a store whose next ARMED commit
    /// is handed over to a stalled connection: the hook parks a [`ConnStall`]
    /// on the very connection the commit's `EVALSHA` is about to be written
    /// to, so the `EVALSHA` is written and queues behind the stall.
    struct StalledHandover {
        armed: AtomicBool,
        /// Another connection of the same client: the stall's `CLIENT LIST`
        /// polls and its releasing `LPUSH`.
        observer: Client,
        key: String,
        parked: Mutex<Option<(ConnStall, u64)>>,
        handed: tokio::sync::Notify,
    }

    impl StalledHandover {
        fn new(client: &ValkeyClient, key: String) -> Arc<Self> {
            Arc::new(Self {
                armed: AtomicBool::new(false),
                observer: client.maintenance_pool.next().clone(),
                key,
                parked: Mutex::new(None),
                handed: tokio::sync::Notify::new(),
            })
        }

        fn hook(self: &Arc<Self>) -> WatchWindowHook {
            let this = Arc::clone(self);
            Arc::new(move |client: &Client| {
                let (this, client) = (Arc::clone(&this), client.clone());
                async move {
                    if !this.armed.swap(false, SeqCst) {
                        return;
                    }
                    let stall = ConnStall::park(&client, &this.observer, this.key.clone()).await;
                    let baseline = stall.net_in().await.unwrap_or(0);
                    *this.parked.lock().unwrap() = Some((stall, baseline));
                    this.handed.notify_one();
                }
                .boxed()
            })
        }

        /// Hand the store's next commit over to a stalled connection.
        fn arm(&self) {
            self.armed.store(true, SeqCst);
        }

        /// Wait until the armed commit's `EVALSHA` has reached the stalled
        /// connection (its bytes read by the server, queued behind the
        /// block), and return the stall, still holding.
        async fn queued(&self) -> ConnStall {
            tokio::time::timeout(HANDOVER_WAIT, self.handed.notified())
                .await
                .expect("the armed commit never reached its dispatch");
            let (stall, baseline) = self
                .parked
                .lock()
                .unwrap()
                .take()
                .expect("the hook parked a stall");
            let deadline = Instant::now() + HANDOVER_WAIT;
            while stall.net_in().await.unwrap_or(0) <= baseline {
                assert!(
                    Instant::now() < deadline,
                    "the commit's EVALSHA never reached the stalled connection"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            stall
        }
    }

    /// A service whose commits ride `client`'s one data connection, with a
    /// [`StalledHandover`] seam and the commit tokens [`FIRST_TOKEN`],
    /// [`SECOND_TOKEN`] pinned for its first two calls.
    async fn handover_service(
        tv: &TestValkey,
        label: &str,
    ) -> (Arc<LiveEditService>, Arc<StalledHandover>, ValkeyClient) {
        let own = one_member_client().await;
        let handover = StalledHandover::new(&own, format!("{}{label}-stall", tv.prefix()));
        let tokens = Arc::new(Mutex::new(VecDeque::from([FIRST_TOKEN, SECOND_TOKEN])));
        let svc = Arc::new(service_over(tv, own.clone(), |store| {
            store
                .with_lua_commit_pre_dispatch_hook_for_test(handover.hook())
                .with_commit_tokens_for_test(tokens)
        }));
        (svc, handover, own)
    }

    /// Send the chained write `operation_id` after `previous` from `svc` on
    /// its own task with `handover` armed, wait until its commit is queued
    /// behind the stall, and drop its caller. Returns the stall, still
    /// holding.
    async fn hand_over_and_drop(
        svc: &Arc<LiveEditService>,
        handover: &StalledHandover,
        target: &Target,
        operation_id: &str,
        previous: &str,
        payload: Vec<u8>,
    ) -> ConnStall {
        handover.arm();
        let call = spawn_chained(
            svc,
            target,
            operation_id,
            previous,
            false,
            payload,
            CAPPED_DEADLINE,
        );
        let stall = handover.queued().await;
        call.abort();
        assert!(
            call.await.unwrap_err().is_cancelled(),
            "precondition: {operation_id}'s caller was dropped with its commit already written \
             to the stalled connection"
        );
        stall
    }

    /// Release `stall` — asserting it held until now — and wait, through a
    /// command sent after it on the same one connection of `own`, until the
    /// commit queued behind it has run.
    async fn release_and_fence(stall: ConnStall, own: &ValkeyClient) {
        let released = stall.release().await;
        assert!(
            matches!(&released, Ok(v) if !v.is_null()),
            "the stall must have held until its release; got {released:?}"
        );
        tokio::time::timeout(
            HANDOVER_WAIT,
            own.pool.next().ping::<fred::types::Value>(None),
        )
        .await
        .expect("the fence must be answered once the stall is released")
        .expect("the fence PING must succeed");
    }

    /// The relay OMPF of every commit on the object's stream after `after`,
    /// with its entry id, in stream order.
    async fn relayed_after(tv: &TestValkey, object_id: u64, after: u64) -> Vec<(String, Bytes)> {
        let entries: Vec<(String, std::collections::HashMap<String, Bytes>)> = tv
            .client()
            .pool
            .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", None)
            .await
            .expect("XRANGE must succeed");
        entries
            .into_iter()
            .filter(|(id, _)| {
                id.split('-')
                    .next()
                    .and_then(|seq| seq.parse::<u64>().ok())
                    .is_some_and(|seq| seq > after)
            })
            .map(|(id, fields)| {
                let ompf = fields.get(RELAY_OMPF_FIELD).cloned().unwrap_or_else(|| {
                    panic!("stream entry {id} must carry the writer's relay OMPF")
                });
                (id, ompf)
            })
            .collect()
    }

    /// `(id, ompf)` stream entries, for comparing with [`relayed_after`].
    fn entries(expected: &[(u64, Vec<u8>)]) -> Vec<(String, Bytes)> {
        expected
            .iter()
            .map(|(seq, ompf)| (format!("{seq}-0"), Bytes::from(ompf.clone())))
            .collect()
    }

    /// Whether `record` is `operation_id`'s, made by the call that carried
    /// the commit token `token`.
    fn names_with_token(record: &Option<String>, operation_id: &str, token: u128) -> bool {
        names(record, operation_id)
            && record
                .as_deref()
                .is_some_and(|r| r.contains(&format!(":{token:032x}:")))
    }

    /// Which case of a P11 scenario runs.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Late {
        /// K: the record is still the predecessor's when the commit runs.
        RecordStillPredecessors,
        /// M: before it runs, the call's re-send S′ (another replica,
        /// `no_wait`) and its successor T commit.
        RecordMovedPast,
    }

    /// The P11 cases' common end: S (`p11-s` after `p11-p`, payload
    /// `field_ompf(2)`, the second pinned token) was handed over and dropped
    /// with P at `seq` 2; for [`Late::RecordMovedPast`], `peer` commits S′ and
    /// T (`p11-t`) now. `run` then lets S's commit run. K: S lands at 3, its
    /// own commit (its token in the record), its relay OMPF the one entry
    /// after P. M: S's late commit is refused by its predecessor guard — the
    /// record names T — so `seq` stays 4 and the entries after P are exactly
    /// S′'s and T's.
    async fn p11_case<F: std::future::Future<Output = ()>>(
        tv: &TestValkey,
        target: &Target,
        peer: &LiveEditService,
        late: Late,
        run: F,
    ) {
        assert_eq!(
            seq(tv, target.object_id).await,
            2,
            "precondition: S's caller was dropped before its commit ran"
        );
        assert!(
            names(&record(tv, target.object_id, WRITER).await, "p11-p"),
            "precondition: the record is still P's"
        );
        if late == Late::RecordMovedPast {
            assert_applied(
                &chained(peer, target, "p11-s", "p11-p", true, field_ompf(2)).await,
                3,
                "S′, S re-sent through another replica with no_wait",
            );
            assert_applied(
                &chained(peer, target, "p11-t", "p11-s", false, field_ompf(3)).await,
                4,
                "T, S's successor",
            );
        }

        run.await;

        match late {
            Late::RecordStillPredecessors => {
                assert_eq!(
                    seq(tv, target.object_id).await,
                    3,
                    "K: S's handed-over commit lands, late, directly after P"
                );
                assert!(
                    names_with_token(
                        &record(tv, target.object_id, WRITER).await,
                        "p11-s",
                        SECOND_TOKEN
                    ),
                    "K: the record is S's own commit, made by the dropped call's EVALSHA"
                );
                assert_eq!(
                    relayed_after(tv, target.object_id, 2).await,
                    entries(&[(3, field_ompf(2))]),
                    "K: exactly one entry after P, S's"
                );
            }
            Late::RecordMovedPast => {
                assert_eq!(
                    seq(tv, target.object_id).await,
                    4,
                    "M: S's late commit must not land after T"
                );
                assert!(
                    names(&record(tv, target.object_id, WRITER).await, "p11-t"),
                    "M: the record is still T's"
                );
                assert_eq!(
                    relayed_after(tv, target.object_id, 2).await,
                    entries(&[(3, field_ompf(2)), (4, field_ompf(3))]),
                    "M: the entries after P are exactly S′'s and T's; a third is S's late \
                     commit, applied after T"
                );
            }
        }
    }

    /// Group 5 — the P11 residual, variant S (a stalled store): S's commit
    /// `EVALSHA` is written to its one connection, held in a server-side
    /// `BLPOP` ([`ConnStall`]), and S's caller is dropped; the stall is then
    /// released (not killed), so the store runs the queued `EVALSHA`, and a
    /// `PING` on the same connection fences it.
    async fn stalled_store_handover(late: Late) {
        let tv = TestValkey::setup().await;
        let target = new_target(&tv, &format!("p11-stalled-{late:?}")).await;
        let (svc, handover, own) = handover_service(&tv, "p11-stalled").await;
        let peer = peer_replica(&tv, |store| store).await;
        assert_applied(
            &chained(&svc, &target, "p11-p", "", false, field_ompf(1)).await,
            2,
            "P, the predecessor",
        );

        let stall =
            hand_over_and_drop(&svc, &handover, &target, "p11-s", "p11-p", field_ompf(2)).await;

        p11_case(&tv, &target, &peer, late, release_and_fence(stall, &own)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_chained_call_handed_to_a_stalled_store_lands_late_while_the_record_is_its_predecessors(
    ) {
        stalled_store_handover(Late::RecordStillPredecessors).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_chained_call_handed_to_a_stalled_store_never_lands_after_the_record_moved_past_its_predecessor(
    ) {
        stalled_store_handover(Late::RecordMovedPast).await;
    }

    /// The P11 residual, variant R (the rider's literal case), which the
    /// vendored fred's change 9 closed: S's commit `EVALSHA` waits in fred's
    /// command channel while its one connection reconnects, and S's caller is
    /// dropped. The client runs through a proxy. S's pre-dispatch hook sends a
    /// marker `GET` on the commit's own connection, which the proxy answers by
    /// closing it, and waits for the `GET` to fail; fred's reconnect
    /// handshake then sends its `PING`, whose reply the proxy holds behind a
    /// gate. The hook returns, and S's `EVALSHA` is queued with no suspension
    /// point in between, so S's caller is dropped with the commit in the
    /// channel. Opening the gate completes the handshake; the router then
    /// skips the queued `EVALSHA`, whose caller gave up, and a `GET` sent
    /// after it on the same connection fences that.
    ///
    /// The commit is never written: no frame carrying S's commit token
    /// reaches the store after the marker, `seq` stays at P's 2 with the
    /// record P's and nothing after P on the stream; S's re-send S′ (another
    /// replica, `no_wait`) and its successor T then commit at 3 and 4, once
    /// each. Before change 9 fred wrote the queued `EVALSHA` on the new
    /// connection and S landed late at 3, directly after P (and, had the
    /// record moved past P first, its predecessor guard refused it).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_chained_call_queued_through_a_reconnect_is_never_written_and_its_resend_commits_once(
    ) {
        let tv = TestValkey::setup().await;
        let target = new_target(&tv, "p11-reconnect").await;
        let marker = format!("{}p11-reconnect-marker", tv.prefix());
        let fence = format!("{}p11-reconnect-fence", tv.prefix());
        let gate = ReplyGate::new();
        let ping_armed = Arc::new(AtomicBool::new(false));
        // Frames the proxy forwards carrying each pinned commit token: P's
        // (the self-check that a commit's frames carry its token at all), and
        // S's after the marker closed its connection.
        let (first_hex, second_hex) = (
            format!("{FIRST_TOKEN:032x}"),
            format!("{SECOND_TOKEN:032x}"),
        );
        let first_token_frames = Arc::new(AtomicUsize::new(0));
        let second_token_frames_after_marker = Arc::new(AtomicUsize::new(0));
        let marker_closed = Arc::new(AtomicBool::new(false));
        let rule: ProxyRule = {
            let (marker, gate, ping_armed) =
                (marker.clone(), gate.clone(), Arc::clone(&ping_armed));
            let (first_frames, second_frames, marker_closed) = (
                Arc::clone(&first_token_frames),
                Arc::clone(&second_token_frames_after_marker),
                Arc::clone(&marker_closed),
            );
            Arc::new(move |args: &[Vec<u8>]| {
                let carries = |hex: &str| {
                    args.iter()
                        .any(|a| String::from_utf8_lossy(a).contains(hex))
                };
                if carries(&first_hex) {
                    first_frames.fetch_add(1, SeqCst);
                }
                if marker_closed.load(SeqCst) && carries(&second_hex) {
                    second_frames.fetch_add(1, SeqCst);
                }
                let name = args.first()?;
                if name.eq_ignore_ascii_case(b"GET")
                    && args
                        .get(1)
                        .is_some_and(|key| key.as_slice() == marker.as_bytes())
                {
                    marker_closed.store(true, SeqCst);
                    return Some(ProxyAction::Close);
                }
                (name.eq_ignore_ascii_case(b"PING") && ping_armed.swap(false, SeqCst))
                    .then(|| ProxyAction::HoldReplies(gate.clone()))
            })
        };
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        let own = proxied_client_with(&proxy, 1, LivenessOverrides::default(), |_| {}).await;

        let armed = Arc::new(AtomicBool::new(false));
        let handed = Arc::new(tokio::sync::Notify::new());
        let commit_connection: Arc<Mutex<Option<(Client, bool)>>> = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (armed, ping_armed, handed, slot, marker) = (
                Arc::clone(&armed),
                Arc::clone(&ping_armed),
                Arc::clone(&handed),
                Arc::clone(&commit_connection),
                marker.clone(),
            );
            Arc::new(move |client: &Client| {
                let (armed, ping_armed, handed, slot, marker, client) = (
                    Arc::clone(&armed),
                    Arc::clone(&ping_armed),
                    Arc::clone(&handed),
                    Arc::clone(&slot),
                    marker.clone(),
                    client.clone(),
                );
                async move {
                    if !armed.swap(false, SeqCst) {
                        return;
                    }
                    ping_armed.store(true, SeqCst);
                    let once = fred::types::config::Options {
                        max_attempts: Some(1),
                        ..Default::default()
                    };
                    let marker_get: Result<Option<String>, fred::error::Error> =
                        client.with_options(&once).get(&marker).await;
                    *slot.lock().unwrap() = Some((client, marker_get.is_err()));
                    handed.notify_one();
                }
                .boxed()
            })
        };
        let tokens = Arc::new(Mutex::new(VecDeque::from([FIRST_TOKEN, SECOND_TOKEN])));
        let svc = Arc::new(service_over(&tv, own, |store| {
            store
                .with_lua_commit_pre_dispatch_hook_for_test(hook)
                .with_commit_tokens_for_test(tokens)
        }));
        let peer = peer_replica(&tv, |store| store).await;
        assert_applied(
            &chained(&svc, &target, "p11-p", "", false, field_ompf(1)).await,
            2,
            "P, the predecessor (which also loads the script)",
        );

        armed.store(true, SeqCst);
        let call = spawn_chained(
            &svc,
            &target,
            "p11-s",
            "p11-p",
            false,
            field_ompf(2),
            CAPPED_DEADLINE,
        );
        tokio::time::timeout(HANDOVER_WAIT, handed.notified())
            .await
            .expect("S never reached its commit dispatch");
        call.abort();
        assert!(
            call.await.unwrap_err().is_cancelled(),
            "precondition: S's caller was dropped"
        );
        let (connection, marker_failed) = commit_connection
            .lock()
            .unwrap()
            .take()
            .expect("the hook kept S's commit connection");
        assert!(
            marker_failed,
            "precondition: the proxy closed S's commit connection under the marker GET"
        );
        gate.reply_held().await;
        assert!(
            connection.inner().counters.cmd_buffer_len.load(SeqCst) >= 1,
            "precondition: S's EVALSHA waits in the reconnecting router's channel"
        );
        assert_eq!(
            proxy.hits(),
            2,
            "precondition: the marker closed the connection and the reconnect's PING is held"
        );

        assert!(
            first_token_frames.load(SeqCst) >= 1,
            "self-check: P's commit frames carried P's token through the proxy, so a written S \
             commit would be counted by its own"
        );

        gate.open();
        let fenced: Result<Option<String>, fred::error::Error> =
            tokio::time::timeout(HANDOVER_WAIT, connection.get(&fence))
                .await
                .expect("the fence must be answered once the reconnect completes");
        fenced.expect("the fence GET must succeed");

        assert_eq!(
            second_token_frames_after_marker.load(SeqCst),
            0,
            "S's commit, queued in fred's channel when its caller was dropped, must never be \
             written: no frame carrying its token after the marker"
        );
        assert_eq!(
            seq(&tv, target.object_id).await,
            2,
            "S's dropped commit never lands: the object stays at P's seq"
        );
        assert!(
            names(&record(&tv, target.object_id, WRITER).await, "p11-p"),
            "the record is still P's"
        );
        assert_eq!(
            relayed_after(&tv, target.object_id, 2).await,
            Vec::new(),
            "nothing after P on the stream"
        );

        assert_applied(
            &chained(&peer, &target, "p11-s", "p11-p", true, field_ompf(2)).await,
            3,
            "S′, S re-sent through another replica with no_wait, commits directly after P",
        );
        assert_applied(
            &chained(&peer, &target, "p11-t", "p11-s", false, field_ompf(3)).await,
            4,
            "T, S's successor",
        );
        assert_eq!(
            relayed_after(&tv, target.object_id, 2).await,
            entries(&[(3, field_ompf(2)), (4, field_ompf(3))]),
            "the entries after P are exactly S′'s and T's, once each"
        );
        assert_eq!(
            second_token_frames_after_marker.load(SeqCst),
            0,
            "S's commit stayed unwritten while the chain moved on"
        );
    }

    /// Group 7: the stale predecessor and its re-send, in both orders. P0 —
    /// the chain's first write P, its commit handed to a stalled connection
    /// and its caller dropped — and P′, P re-sent through another replica
    /// with `no_wait` (same operation, bytes and predecessor).
    ///
    /// - Order 1: P0 lands late first (its own token in the record); P′ is
    ///   then answered `ALREADY_COMMITTED` at P0's `seq`; S commits once,
    ///   after it.
    /// - Order 2: P′ is `APPLIED` and S commits; P0's late `EVALSHA` then
    ///   finds S's record, which is neither its own operation nor — P being a
    ///   chain's first write — no record, so its predecessor guard refuses it.
    ///
    /// Either way P and S each land exactly once, in chain order.
    async fn stale_predecessor_and_its_resend(p0_first: bool) {
        let tv = TestValkey::setup().await;
        let target = new_target(&tv, &format!("p7-stale-predecessor-{p0_first}")).await;
        let (svc, handover, own) = handover_service(&tv, "p7").await;
        let peer = peer_replica(&tv, |store| store).await;

        let stall = hand_over_and_drop(&svc, &handover, &target, "p7-p", "", field_ompf(1)).await;
        assert_eq!(
            seq(&tv, target.object_id).await,
            1,
            "precondition: P0's caller was dropped before its commit ran"
        );

        if p0_first {
            release_and_fence(stall, &own).await;
            assert!(
                names_with_token(
                    &record(&tv, target.object_id, WRITER).await,
                    "p7-p",
                    FIRST_TOKEN
                ),
                "order 1: P0's own late commit is the record"
            );
            let resent = chained(&peer, &target, "p7-p", "", true, field_ompf(1)).await;
            let resent = resent.expect("P′ must be answered");
            assert_eq!(
                (resent.sequence_num, outcome(&resent)),
                (2, OperationOutcome::AlreadyCommitted),
                "order 1: P′ is answered from P0's commit"
            );
            assert_applied(
                &chained(&peer, &target, "p7-s", "p7-p", false, field_ompf(2)).await,
                3,
                "order 1: S, after P",
            );
        } else {
            assert_applied(
                &chained(&peer, &target, "p7-p", "", true, field_ompf(1)).await,
                2,
                "order 2: P′",
            );
            assert_applied(
                &chained(&peer, &target, "p7-s", "p7-p", false, field_ompf(2)).await,
                3,
                "order 2: S, after P′",
            );
            release_and_fence(stall, &own).await;
            assert!(
                names(&record(&tv, target.object_id, WRITER).await, "p7-s"),
                "order 2: P0's late commit must not land after S: the record stays S's"
            );
        }

        assert_eq!(
            seq(&tv, target.object_id).await,
            3,
            "P and S each land exactly once"
        );
        assert_eq!(
            relayed_after(&tv, target.object_id, 1).await,
            entries(&[(2, field_ompf(1)), (3, field_ompf(2))]),
            "the stream holds P once, then S"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_predecessor_landing_before_its_resend_is_answered_already_committed_and_its_successor_commits_once(
    ) {
        stale_predecessor_and_its_resend(true).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_predecessor_running_after_its_resend_and_successor_is_refused_by_its_commit() {
        stale_predecessor_and_its_resend(false).await;
    }

    /// Group 8b: two dropped calls' late commits land in chain order. S1
    /// (after P) is handed over to a stalled connection on replica A and its
    /// caller dropped; S2 (after S1) waits in replica B's gate. Releasing
    /// stall 1 lands S1 at 3; B's gate then lets S2 through, its commit is
    /// handed over to B's own stalled connection, and its caller is dropped
    /// too; releasing stall 2 lands S2 at 4. Each caller was dropped before
    /// its commit ran (`seq` unchanged at the drop).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_dropped_chained_calls_handed_over_late_land_in_chain_order() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let target = new_target(&tv, "p8b-two-late").await;
        let (a, handover_a, own_a) = handover_service(&tv, "p8b-a").await;
        let (b, handover_b, own_b) = handover_service(&tv, "p8b-b").await;
        assert_applied(
            &chained(&a, &target, "p8b-p", "", false, field_ompf(1)).await,
            2,
            "P, the predecessor",
        );

        let stall_1 =
            hand_over_and_drop(&a, &handover_a, &target, "p8b-s1", "p8b-p", field_ompf(2)).await;
        assert_eq!(
            seq(&tv, target.object_id).await,
            2,
            "S1's caller was dropped before its commit ran"
        );
        handover_b.arm();
        let s2 = spawn_chained(
            &b,
            &target,
            "p8b-s2",
            "p8b-s1",
            false,
            field_ompf(3),
            CAPPED_DEADLINE,
        );
        parked(deadline, &b.metrics, 1, "S2 waiting in B's gate").await;
        probed(deadline, &b.metrics, 1, "S2's first probe").await;

        release_and_fence(stall_1, &own_a).await;
        assert_eq!(
            seq(&tv, target.object_id).await,
            3,
            "S1's late commit lands, directly after P"
        );

        let stall_2 = handover_b.queued().await;
        s2.abort();
        assert!(
            s2.await.unwrap_err().is_cancelled(),
            "precondition: S2's caller was dropped with its commit already written to B's \
             stalled connection"
        );
        assert_eq!(
            seq(&tv, target.object_id).await,
            3,
            "S2's caller was dropped before its commit ran"
        );
        release_and_fence(stall_2, &own_b).await;

        assert_eq!(
            seq(&tv, target.object_id).await,
            4,
            "S2's late commit lands, directly after S1"
        );
        assert!(
            names(&record(&tv, target.object_id, WRITER).await, "p8b-s2"),
            "the record is S2's"
        );
        assert_eq!(
            relayed_after(&tv, target.object_id, 2).await,
            entries(&[(3, field_ompf(2)), (4, field_ompf(3))]),
            "S1 then S2, once each"
        );
    }
}
