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

//! Upload admission through the `CreateLayer` / `UpdateLayer` handlers
//! (b.715, SDD §5.4): each upload reads its header, is authorized (and, on
//! `UpdateLayer`, its target resolved), reserves from the replica's upload
//! budget, and only then reads its content.
//!
//! The content arrives through a [`ScriptedBody`]: an HTTP request body fed
//! one gRPC frame per HTTP/2 DATA frame from a channel the test holds, which
//! counts the frames the handler pulled. A test can therefore stall an upload
//! mid-content (by holding the channel open), end it with END_STREAM, fail its
//! transport, reset it the way a client's `RST_STREAM(CANCEL)` arrives, or end
//! it without END_STREAM the way hyper delivers a `RST_STREAM(NO_ERROR)` or a
//! keepalive `GOAWAY`. Every upload request carries its body's end-of-stream
//! record (`grpc::request_end`), as the LiveService stack hands it over.
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | `*_is_refused_from_the_header_alone` (six) | T1, T5 | a missing permission, an unknown session, an unknown layer and a stale `object_id` are refused having pulled only the header frame, and take no reservation |
//! | `*_authorized_reads_every_content_frame_and_commits` (two) | T1 | an authorized upload reads every frame and answers OK, and its reservation is released |
//! | `*_whose_first_frame_is_a_chunk_is_refused_before_any_permission_check` | T1 | a malformed first frame never reaches the permission service |
//! | `update_layer_second_header_after_content_is_invalid_argument_at_stage_body` | T1 | a malformed later frame is `INVALID_ARGUMENT`, logged at stage `body` |
//! | `stalled_uploads_filling_the_budget_leave_every_further_call_admitted_or_refused_unavailable` | T3; b.kgb test 11 | no partial-reservation deadlock: with the budget held by half-sent uploads every further call ends `UNAVAILABLE` with the `UPLOAD_CAPACITY` retry hint whose `cause` matches its message (`saturated` / `waiter_cap`), none hangs, and the stalled ones still commit |
//! | `concurrent_uploads_stay_within_the_budget_and_refused_ones_write_nothing` | T4 | many delayed uploads to distinct objects and to one object never reserve past the budget; each answer is OK (committed) or `UNAVAILABLE` (nothing written) |
//! | `update_layer_structural_commit_retried_holds_its_charge_and_releases_it` | T4 | a structural `UpdateLayer` retried by the fault injector commits under its processing charge and releases it |
//! | `*_releases_its_reservation` (four, plus the two `*_meets_the_server_deadline_*`) | T5 | a parse error, a storage error (`INTERNAL`), a transport error, a dropped handler and the server's RPC deadline (`MetricsLayer`, both RPCs) all release the reservation; an auth refusal takes none (the T1 rows) |
//! | `update_layer_queued_at_shutdown_answers_server_shutting_down_with_one_info_line` | T5; b.kgb test 7 | a queued admission ends at t0, when graceful shutdown's paced stream ending starts (`begun`) — not at the end of the paced ending — with `UNAVAILABLE "server shutting down"` and its retry hint |
//! | `a_session_over_its_receiving_cap_waits_while_another_sessions_upload_is_admitted` | T6 | the per-session cap: one session's excess waits and is refused at the wait budget while another session's upload commits |
//! | `create_layer_whose_session_was_deleted_after_authorization_is_not_found_and_maps_nothing` | T9 | a `CreateLayer` authorized before a `DeleteSession` is refused at its commit point and writes no mapping, index entry or record |
//! | `upload_admission_metrics_follow_a_handled_upload` | T12 | the handler's admission ticks the service's metrics and returns the gauges to zero |
//! | `overflow_slot::over_budget_dense_uploads_commit_through_the_overflow_slot_on_every_path` | T13 | a dense `CreateLayer`, fast-path and slow-path `UpdateLayer` each charged more than the whole budget commit, each counted once in `overflow_runs_total{rpc}`, with the in-budget peak within the budget |
//! | `overflow_slot::two_over_budget_uploads_run_one_at_a_time_in_the_overflow_slot` | T4 | the second over-budget upload queues while the first holds the slot; both commit; `overflow_runs_total` = 2 |
//! | `overflow_slot::create_layer_waiting_for_the_slot_*` (two) | T5; b.kgb test 7 | the start of the paced stream ending (`begun`) while waiting for the slot answers `UNAVAILABLE "server shutting down"` with its retry hint (one INFO line at stage `parse`, no slot-run line, nothing applied) and the holder still commits; the RPC deadline answers `DEADLINE_EXCEEDED`; both release the reservation |
//! | `overflow_slot::*_over_cap_delta_is_refused_at_its_parse_without_a_processing_charge` (two) | b.715 regression | a delta over a §9.4 cap is refused `INVALID_ARGUMENT` at stage `parse`, before its processing charge: no slot, no growth past the parse charge, no external value stored, reservation released |
//! | `overflow_slot::create_layer_over_the_budget_logs_one_slot_run_line` | T13 | one INFO "upload runs beyond the upload budget" line with `rpc`, `charge_bytes` and `upload_budget_bytes` |
//! | `overflow_slot::create_layer_resend_of_an_over_budget_create_is_answered_without_the_slot` | T5 | an `ALREADY_COMMITTED` re-send's charges never wait: original `object_id`, no slot, no `refused_growth`, re-spawn skipped |
//! | `overflow_slot::create_layer_resend_of_an_over_cap_delta_skips_its_respawn_for_the_delta_error` | b.715 regression | the re-send path runs the per-delta checks before its processing charge |
//! | `overflow_slot::create_layer_holds_its_parse_charge_before_parsing_its_delta` | T5 | the parse charge is held before the parse: with the slot held, malformed content whose parse charge is over the budget queues rather than being refused at once |
//! | `update_layer_starts_its_start_record_read_only_after_its_whole_body` | I3 | the start-record read starts only once the whole body is received |
//! | `overflow_slot::update_layer_starts_its_start_record_read_before_its_processing_wait` | I3 | the start-record read starts immediately after the body: it has started while the upload waits in the processing queue |
//! | `update_layer_cached_edit_waits_for_its_hint_at_the_processing_step_b26e` | b.26e Stage 2 (the handler's object-state hint) | a structural edit on a cached object, with the budget full and the slot held, waits at its processing step for its object-state hint with the object's per-object permits free, then runs in the freed slot and commits with no recharge: its first charge point, the hit's repair read (RP0), grows past the hint there, where growth is booked in place |
//! | `update_layer_cached_edit_with_its_hint_in_budget_recharges_once_at_its_first_repair_read_fsz` | b.fsz (a), the hint held in budget | a structural edit on a cached object takes its processing step and hint in budget; with the budget then full and the slot held, a processing step queued for it, the hit's first repair read (RP0) is refused, and the call recharges once, keeping its processing step and hint (its reservation, read off the gate's charge observer, never below them), granted in budget at RP0's target, then commits once |
//! | `update_layer_recharge_wait_at_shutdown_answers_server_shutting_down_at_stage_storage_b26e` | b.26e Stage 2 (the in-commit recharge wait, handler arm) | shutdown during a WATCH commit's in-commit recharge wait answers the shared shutdown status (`UNAVAILABLE "server shutting down"`, the `SERVER_SHUTTING_DOWN` detail and its retry hint), one INFO line at stage `storage`, nothing applied |
//! | `update_layer_recharge_wait_at_shutdown_after_a_counted_retry_answers_outcome_unknown_b26e` | b.26e Stage 2 (the in-commit recharge wait, handler arm), SDD §18.1 | the same wait on attempt 2, after a counted retry, answers the plain outcome-unknown `UNAVAILABLE` with no detail, at stage `storage`, nothing applied |
//! | `update_layer_recovery_reissue_recharges_through_the_calls_commit_charge_b26e` | b.26e Stage 2 (the object-state hold reused by the recovery re-issue) | after a wiped object's checkpoint restore, the re-issued WATCH commit charges through the call's commit charge: with the budget full and the slot held it waits for one recharge holding no per-object permit, then commits |
//! | `reissue_extra_copy::update_layer_recovery_reissue_holds_one_copy_of_its_external_values_for_its_pre_store_b5wb` | b.5wb (the re-issue's extra-copy charge) | the checkpoint-recovery re-issue's own pre-store grows the reservation by one copy of the stored bytes right before it, gives it back right after, before the re-issue's commit dispatch, and records one processing-wait observation |
//! | `reissue_extra_copy::update_layer_recovery_reissue_waiting_to_hold_its_extra_copy_at_shutdown_answers_server_shutting_down_b5wb` | b.5wb | with the budget full and the slot held, that hold waits as one processing waiter; shutdown there answers the shutdown refusal verbatim (the first call made one attempt), at INFO, and the re-issue never runs |
//! | `cancelled_upload::*` | b.682 regression tests (not a b.715 design item) | an upload stream that did not end with END_STREAM — a client's `CANCEL`, a `RST_STREAM(NO_ERROR)`, the server's keepalive `GOAWAY` — answers `CANCELLED` with nothing committed, created, mapped or stored and its reservation released, at the handler and end to end; a header followed by END_STREAM still commits an empty update or creates an empty layer; a request with no end-of-stream record is `INTERNAL` having read nothing (see that module) |
//!
//! # Wait budget
//!
//! Every wait is a deadline, not a sleep, and the healthy path meets each in
//! milliseconds, except the two waits named below that the healthy path waits
//! out in real time. Each test takes ONE deadline when it starts
//! ([`test_deadline`]: [`TEST_BUDGET`], 60 s, from then) and bounds every wait
//! it makes by that one instant — each handler result, each poll for a
//! condition ([`until`]), each drain of spawned uploads, a parked holder's
//! arrival and the deadline server's readiness — through
//! `tokio::time::timeout_at` ([`settled`]). However many waits a test makes,
//! they share the one budget; a per-wait timeout would instead multiply by the
//! wait count.
//!
//! | Test | Worst case |
//! |---|---|
//! | every test in this file | [`TEST_BUDGET`]: 60 s |
//!
//! The `cancelled_upload` tests that loop over two or three endings share
//! their one deadline across the loop. Two tests wait in real time on the
//! healthy path:
//!
//! * `cancelled_upload::update_layer_whose_connection_meets_the_keepalive_timeout_*`
//!   waits out the production keepalive (`transport::HTTP2_KEEPALIVE_INTERVAL`
//!   plus `HTTP2_KEEPALIVE_TIMEOUT`, 35 s today), inside its 60 s budget; a
//!   `const` check in that module holds the sum inside the budget.
//! * [`stalled_uploads_filling_the_budget_leave_every_further_call_admitted_or_refused_unavailable`]
//!   (T3): its two queued calls wait out the whole 5 s admission wait
//!   (`QUEUED_WAIT`) before they are refused `saturated`. The wait is that long
//!   so they are still queued when the later calls reach admission and find
//!   the queue full.
//!
//! All are under nextest's 120 s kill. Not counted: fixture setup (sessions
//! and base layers created through a separate service) and read-backs from
//! Valkey, which are round trips bounded by the client's command timeout, not
//! waits on the behaviour under test. Waits inside a handler — the admission
//! waits configured here ([`SHORT_WAIT`], T3's 5 s `QUEUED_WAIT`, T4's 5 s), a
//! slot wait, the 5 s RPC deadline of the slot-deadline test — are inside its
//! bounded result. The few `tokio::time::sleep` calls in
//! [`concurrent_uploads_stay_within_the_budget_and_refused_ones_write_nothing`]
//! stagger content delivery — the "delayed streams" the scenario names — and
//! synchronize nothing; [`until`] sleeps 5 ms between probes.

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::{BufMut, Bytes};
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use http_body::Frame;
use prost::Message;
use tokio::sync::mpsc;
use tonic::codec::{BufferSettings, Streaming};
use tonic::{Code, Request, Status};
use tonic_prost::ProstCodec;

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    create_layer_request, update_layer_request, CreateLayerHeader, CreateLayerRequest,
    UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::upload_admission::{
    admission_charge, UploadAdmissionConfig, UploadAdmissionGate, RECEIVE_ALLOWANCE_BYTES,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline, empty_delta};
use crate::common::refusal_detail::{assert_refusal_detail, assert_shutdown_status};
use live_edit::error::{refusal_metadata, refusal_reason, CAPACITY_RETRY_DELAY};

use super::helpers::*;
use super::log_capture::{assert_common_fields, capture, RPC_FAILURE_MSG};

/// One test's whole wait budget: every wait for a condition or a handler
/// result in a test ends by [`test_deadline`]. Unreachable on a healthy run.
const TEST_BUDGET: Duration = Duration::from_secs(60);

/// The instant every wait of the calling test ends by: [`TEST_BUDGET`] from
/// now. Take it once, as the test starts.
fn test_deadline() -> tokio::time::Instant {
    tokio::time::Instant::now() + TEST_BUDGET
}

/// A short admission wait, so a refusal at the wait budget arrives quickly,
/// and far above a local round trip. T6's per-session-cap test configures it;
/// T3's saturation test uses its own, longer `QUEUED_WAIT` instead.
const SHORT_WAIT: Duration = Duration::from_millis(300);

/// `F + A`: what every admission reserves.
const C: u64 = admission_charge();

/// `4A`: room for three admissions by bytes and a per-session receiving cap
/// of one (`grpc::upload_admission`'s unit tests pin the arithmetic).
const SMALL_BUDGET: u64 = 4 * RECEIVE_ALLOWANCE_BYTES;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

// ═════════════════════════════════════════════════════════════════════════════
// A scripted request body
// ═════════════════════════════════════════════════════════════════════════════

/// One step of a [`ScriptedBody`].
enum Step {
    /// One HTTP/2 DATA frame.
    Data(Bytes),
    /// The body's error as a client `RST_STREAM(CANCEL)` produces it: an
    /// `h2::Error` with reason `CANCEL`, which tonic maps to `CANCELLED` and
    /// then swallows into a clean end of stream.
    Cancel,
    /// A transport failure that is not a cancel.
    Fail,
    /// The end of the body without END_STREAM: hyper's clean end for a
    /// `RST_STREAM(NO_ERROR)` (what Envoy v1.36 forwards for a client's
    /// cancel) and for the server's own keepalive `GOAWAY(NO_ERROR)`.
    EndWithoutEndStream,
}

/// A request body fed from a channel. Ends with END_STREAM when the sender is
/// dropped; stalls while it is held and empty. Reports `is_end_stream()` as h2
/// does: true only once it has ended with END_STREAM.
struct ScriptedBody {
    steps: mpsc::UnboundedReceiver<Step>,
    pulled: Arc<AtomicUsize>,
    /// Set when the body ended because the sender was dropped: END_STREAM.
    end_stream: bool,
}

impl http_body::Body for ScriptedBody {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        match self.steps.poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                self.end_stream = true;
                Poll::Ready(None)
            }
            Poll::Ready(Some(Step::EndWithoutEndStream)) => Poll::Ready(None),
            Poll::Ready(Some(Step::Data(bytes))) => {
                self.pulled.fetch_add(1, Ordering::SeqCst);
                Poll::Ready(Some(Ok(Frame::data(bytes))))
            }
            Poll::Ready(Some(Step::Cancel)) => {
                Poll::Ready(Some(Err(Box::new(h2::Error::from(h2::Reason::CANCEL)))))
            }
            Poll::Ready(Some(Step::Fail)) => Poll::Ready(Some(Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "connection reset",
            ))))),
        }
    }

    fn is_end_stream(&self) -> bool {
        self.end_stream
    }
}

/// The test's end of a [`ScriptedBody`].
struct Feed {
    tx: Option<mpsc::UnboundedSender<Step>>,
    pulled: Arc<AtomicUsize>,
}

impl Feed {
    fn step(&self, step: Step) {
        self.tx
            .as_ref()
            .expect("the feed is still open")
            .send(step)
            .expect("the body is still alive");
    }

    /// Send one message as its own frame.
    fn send<T: Message>(&self, msg: &T) {
        self.step(Step::Data(grpc_frame(msg)));
    }

    /// Send one message as its own frame if the handler still holds the body:
    /// a refused upload drops its body, and the rest of its content goes
    /// nowhere.
    fn send_if_open<T: Message>(&self, msg: &T) {
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.send(Step::Data(grpc_frame(msg)));
        }
    }

    /// Send raw bytes as one DATA frame.
    fn send_raw(&self, bytes: Bytes) {
        self.step(Step::Data(bytes));
    }

    fn cancel(&self) {
        self.step(Step::Cancel);
    }

    fn fail(&self) {
        self.step(Step::Fail);
    }

    /// End the body without END_STREAM (see [`Step::EndWithoutEndStream`]).
    fn end_without_end_stream(&self) {
        self.step(Step::EndWithoutEndStream);
    }

    /// End the body cleanly, with END_STREAM.
    fn end(&mut self) {
        self.tx = None;
    }

    /// DATA frames the handler pulled so far.
    fn pulled(&self) -> usize {
        self.pulled.load(Ordering::SeqCst)
    }
}

/// A [`ScriptedBody`] and the test's end of it.
fn scripted_body() -> (ScriptedBody, Feed) {
    let (tx, steps) = mpsc::unbounded_channel();
    let pulled = Arc::new(AtomicUsize::new(0));
    (
        ScriptedBody {
            steps,
            pulled: Arc::clone(&pulled),
            end_stream: false,
        },
        Feed {
            tx: Some(tx),
            pulled,
        },
    )
}

/// An upload request over a [`ScriptedBody`], carrying the test bearer and
/// the body's end-of-stream record, as the LiveService stack hands an upload
/// to its handler.
fn upload<T: Message + Default + Send + 'static>() -> (Request<Streaming<T>>, Feed) {
    let (body, feed) = scripted_body();
    (make_request(AttestedStreaming::new(body)), feed)
}

/// As [`upload`], but with NO end-of-stream record: the request a LiveService
/// stack built without `live_service_server` would hand the handler.
fn upload_without_a_record<T: Message + Default + Send + 'static>() -> (Request<Streaming<T>>, Feed)
{
    let (body, feed) = scripted_body();
    let decoder = ProstCodec::<(), T>::raw_decoder(BufferSettings::default());
    (
        make_request(Streaming::new_request(decoder, body, None, None)),
        feed,
    )
}

/// An upload whose every frame is already sent and whose body has ended.
fn sent<T: Message + Default + Send + 'static>(messages: &[T]) -> (Request<Streaming<T>>, Feed) {
    let (request, mut feed) = upload::<T>();
    for msg in messages {
        feed.send(msg);
    }
    feed.end();
    (request, feed)
}

/// One message as a gRPC frame: compression flag, big-endian length, bytes.
fn grpc_frame<T: Message>(msg: &T) -> Bytes {
    let encoded = msg.encode_to_vec();
    let mut buf = bytes::BytesMut::with_capacity(5 + encoded.len());
    buf.put_u8(0);
    buf.put_u32(encoded.len() as u32);
    buf.extend_from_slice(&encoded);
    buf.freeze()
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures
// ═════════════════════════════════════════════════════════════════════════════

/// [`build_service`] with its upload gate built over `config` and the
/// service's metrics.
fn service_with_gate(
    tv: &TestValkey,
    permissions: Arc<dyn live_edit::auth::permissions::PermissionsClient>,
    config: UploadAdmissionConfig,
) -> LiveEditService {
    let base = build_service(tv, permissions);
    LiveEditService {
        upload_admission: Arc::new(UploadAdmissionGate::new(
            config,
            Some(Arc::clone(&base.metrics)),
        )),
        ..base
    }
}

fn small_gate(wait: Duration, max_waiters: usize) -> UploadAdmissionConfig {
    UploadAdmissionConfig {
        budget_bytes: SMALL_BUDGET,
        wait,
        max_waiters,
        // No size-cap floor, so the budget can stay small.
        max_upload_bytes: 0,
    }
}

/// An OMPF upload setting one inline field, distinct per `tag`.
fn field_ompf(tag: u8) -> Vec<u8> {
    delta_to_ompf_bytes(
        &delta_with_set_field_inline(1, &format!("key-{tag}"), vec![tag; 64], 0),
        &[],
    )
}

/// `bytes` split into `parts` contiguous chunks (the last takes the rest).
fn split(bytes: &[u8], parts: usize) -> Vec<Vec<u8>> {
    let size = bytes.len().div_ceil(parts).max(1);
    bytes.chunks(size).map(<[u8]>::to_vec).collect()
}

fn update_header_with(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    client_id: &str,
    operation_id: &str,
) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: 1_000,
                client_id: client_id.to_owned(),
                operation_id: operation_id.to_owned(),
            },
        )),
    }
}

fn create_header_with(
    session_id: &str,
    layer_name: &str,
    operation_id: &str,
) -> CreateLayerRequest {
    CreateLayerRequest {
        message_type: Some(create_layer_request::MessageType::Header(
            CreateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                operation_id: operation_id.to_owned(),
            },
        )),
    }
}

/// A header followed by `ompf` in `parts` chunks.
fn update_messages(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    ompf: &[u8],
    parts: usize,
) -> Vec<UpdateLayerRequest> {
    std::iter::once(make_update_header(session_id, layer_name, object_id))
        .chain(split(ompf, parts).into_iter().map(make_update_data))
        .collect()
}

fn create_messages(
    session_id: &str,
    layer_name: &str,
    ompf: &[u8],
    parts: usize,
) -> Vec<CreateLayerRequest> {
    std::iter::once(make_create_header(session_id, layer_name))
        .chain(split(ompf, parts).into_iter().map(make_create_data))
        .collect()
}

/// A session with one layer at `seq = 1`, created through a separate
/// allow-all service, so the service under test's gate has admitted nothing
/// yet and its budget is untouched.
async fn setup_layer_named(tv: &TestValkey, session_name: &str, layer_name: &str) -> (String, u64) {
    let setup = build_service(tv, Arc::new(AllowAllPermissions));
    create_session_and_layer(tv, &setup, session_name, layer_name).await
}

async fn new_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

/// The layer's committed sequence number (`obj:hdr` `seq`).
async fn seq(tv: &TestValkey, object_id: u64) -> u64 {
    let raw: String = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_header(&object_id.to_string()), "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");
    raw.parse().expect("seq is numeric")
}

/// Poll `cond` until it holds, failing at the test's `deadline`.
async fn until(deadline: tokio::time::Instant, what: &str, mut cond: impl FnMut() -> bool) {
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not happen within the test's {TEST_BUDGET:?} budget"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// A handler result (or any awaited outcome), bounded by the test's
/// `deadline`.
async fn settled<T>(
    deadline: tokio::time::Instant,
    what: &str,
    fut: impl std::future::Future<Output = T>,
) -> T {
    tokio::time::timeout_at(deadline, fut)
        .await
        .unwrap_or_else(|_| {
            panic!("{what} did not finish within the test's {TEST_BUDGET:?} budget: it hung")
        })
}

/// What a refusal from the header alone looks like: `code`, exactly the
/// header's frame pulled, and no reservation ever taken.
fn assert_refused_from_the_header(
    status: &Status,
    code: Code,
    feed: &Feed,
    gate: &UploadAdmissionGate,
    what: &str,
) {
    assert_eq!(status.code(), code, "{what}: {status:?}");
    assert_eq!(
        feed.pulled(),
        1,
        "{what}: only the header frame may be read before the call is authorized, its target \
         resolved and its upload admitted; the content was read first"
    );
    assert_eq!(
        gate.high_water_bytes(),
        0,
        "{what}: a call refused before admission must take no reservation"
    );
}

/// An OMPF big enough to send in several frames.
fn multi_frame_ompf() -> Vec<u8> {
    let ompf = field_ompf(7);
    assert!(ompf.len() >= 4, "fixture self-check: {} bytes", ompf.len());
    ompf
}

/// The frames every T1 refusal test sends: a header and four content frames.
const CONTENT_FRAMES: usize = 4;

// ═════════════════════════════════════════════════════════════════════════════
// T1 — refused from the header, before any content is read
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_without_write_permission_is_refused_from_the_header_alone() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));
    let session_id = new_session(&tv, "t1-create-denied").await;

    let (request, feed) = sent(&create_messages(
        &session_id,
        "layer",
        &multi_frame_ompf(),
        CONTENT_FRAMES,
    ));
    let status = settled(deadline, "the handler", handle_create_layer(&svc, request))
        .await
        .expect_err("a caller without write permission must be refused");
    assert_refused_from_the_header(
        &status,
        Code::PermissionDenied,
        &feed,
        &svc.upload_admission,
        "CreateLayer without permission",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_on_an_unknown_session_is_refused_from_the_header_alone() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (request, feed) = sent(&create_messages(
        "no-such-session",
        "layer",
        &multi_frame_ompf(),
        CONTENT_FRAMES,
    ));
    let status = settled(deadline, "the handler", handle_create_layer(&svc, request))
        .await
        .expect_err("an unknown session must be refused");
    assert_refused_from_the_header(
        &status,
        Code::NotFound,
        &feed,
        &svc.upload_admission,
        "CreateLayer on an unknown session",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_without_write_permission_is_refused_from_the_header_alone() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let setup = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &setup, "t1-update-denied", "layer").await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let (request, feed) = sent(&update_messages(
        &session_id,
        "layer",
        object_id,
        &multi_frame_ompf(),
        CONTENT_FRAMES,
    ));
    let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect_err("a caller without write permission must be refused");
    assert_refused_from_the_header(
        &status,
        Code::PermissionDenied,
        &feed,
        &svc.upload_admission,
        "UpdateLayer without permission",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_on_an_unknown_session_is_refused_from_the_header_alone() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (request, feed) = sent(&update_messages(
        "no-such-session",
        "layer",
        1,
        &multi_frame_ompf(),
        CONTENT_FRAMES,
    ));
    let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect_err("an unknown session must be refused");
    assert_refused_from_the_header(
        &status,
        Code::NotFound,
        &feed,
        &svc.upload_admission,
        "UpdateLayer on an unknown session",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_on_an_unknown_layer_is_refused_from_the_header_alone() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "t1-unknown-layer").await;

    let (request, feed) = sent(&update_messages(
        &session_id,
        "no-such-layer",
        1,
        &multi_frame_ompf(),
        CONTENT_FRAMES,
    ));
    let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect_err("an unknown layer must be refused");
    assert_refused_from_the_header(
        &status,
        Code::NotFound,
        &feed,
        &svc.upload_admission,
        "UpdateLayer on an unknown layer",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_with_a_stale_object_id_is_refused_from_the_header_alone() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = setup_layer_named(&tv, "t1-stale", "layer").await;

    let (request, feed) = sent(&update_messages(
        &session_id,
        "layer",
        object_id + 1_000,
        &multi_frame_ompf(),
        CONTENT_FRAMES,
    ));
    let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect_err("a stale object_id must be refused");
    assert_refused_from_the_header(
        &status,
        Code::FailedPrecondition,
        &feed,
        &svc.upload_admission,
        "UpdateLayer with a stale object_id",
    );
}

/// The release every admitted upload ends with: nothing reserved, no session
/// slot held, and evidence it was admitted at all.
fn assert_admitted_and_released(gate: &UploadAdmissionGate, session_id: &str, what: &str) {
    assert!(
        gate.high_water_bytes() >= C,
        "{what}: precondition: the upload was admitted"
    );
    assert_eq!(
        gate.reserved_bytes(),
        0,
        "{what}: the reservation must be released"
    );
    assert_eq!(
        gate.receiving_uploads(session_id),
        0,
        "{what}: the session's receiving slot must be released"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_authorized_reads_every_content_frame_and_commits() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "t1-create-ok").await;

    let (request, feed) = sent(&create_messages(
        &session_id,
        "layer",
        &multi_frame_ompf(),
        CONTENT_FRAMES,
    ));
    let response = settled(deadline, "the handler", handle_create_layer(&svc, request))
        .await
        .expect("an authorized upload commits");
    assert_ne!(response.into_inner().object_id, 0);
    assert_eq!(feed.pulled(), 1 + CONTENT_FRAMES);
    assert_admitted_and_released(&svc.upload_admission, &session_id, "CreateLayer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_authorized_reads_every_content_frame_and_commits() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = setup_layer_named(&tv, "t1-update-ok", "layer").await;

    let (request, feed) = sent(&update_messages(
        &session_id,
        "layer",
        object_id,
        &multi_frame_ompf(),
        CONTENT_FRAMES,
    ));
    let response = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect("an authorized upload commits");
    assert_eq!(response.into_inner().sequence_num, 2);
    assert_eq!(feed.pulled(), 1 + CONTENT_FRAMES);
    assert_admitted_and_released(&svc.upload_admission, &session_id, "UpdateLayer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_whose_first_frame_is_a_chunk_is_refused_before_any_permission_check() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let permissions = Arc::new(RecordingPermissions::allowing());
    let svc = build_service(&tv, Arc::clone(&permissions) as _);
    let session_id = new_session(&tv, "t1-create-chunk-first").await;

    let (request, _feed) = sent(&[
        make_create_data(multi_frame_ompf()),
        make_create_header(&session_id, "layer"),
    ]);
    let status = settled(deadline, "the handler", handle_create_layer(&svc, request))
        .await
        .expect_err("a chunk before the header must be refused");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert!(
        status.message().contains("first message must be a header"),
        "{status:?}"
    );
    assert!(
        permissions.calls().is_empty(),
        "a malformed first frame must be refused before the permission service is asked; \
         probed: {:?}",
        permissions.calls()
    );
    assert_eq!(svc.upload_admission.high_water_bytes(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_whose_first_frame_is_a_chunk_is_refused_before_any_permission_check() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let permissions = Arc::new(RecordingPermissions::allowing());
    let svc = build_service(&tv, Arc::clone(&permissions) as _);

    let (request, _feed) = sent(&[make_update_data(multi_frame_ompf())]);
    let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect_err("a chunk before the header must be refused");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert!(
        permissions.calls().is_empty(),
        "a malformed first frame must be refused before the permission service is asked; \
         probed: {:?}",
        permissions.calls()
    );
    assert_eq!(svc.upload_admission.high_water_bytes(), 0);
}

/// A malformed frame after content has begun is `INVALID_ARGUMENT`, logged
/// once at INFO at stage `body` — the admitted upload reading its content.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_second_header_after_content_is_invalid_argument_at_stage_body() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = setup_layer_named(&tv, "t1-second-header", "layer").await;

    let (request, _feed) = sent(&[
        make_update_header(&session_id, "layer", object_id),
        make_update_data(multi_frame_ompf()),
        make_update_header(&session_id, "layer", object_id),
    ]);
    let (result, captured) = settled(
        deadline,
        "the handler",
        capture(handle_update_layer(&svc, request)),
    )
    .await;
    let status = result.expect_err("a second header must be refused");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert!(status.message().contains("second header"), "{status:?}");

    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "UpdateLayer", "body", "INVALID_ARGUMENT");
    assert_admitted_and_released(&svc.upload_admission, &session_id, "second header");
    assert_eq!(seq(&tv, object_id).await, 1, "nothing was committed");
}

// ═════════════════════════════════════════════════════════════════════════════
// T3 — no partial-reservation deadlock; bounded waits
// ═════════════════════════════════════════════════════════════════════════════

/// Three half-sent uploads hold the whole budget (each admitted, one chunk
/// received, the rest withheld). Every further call is admitted or refused
/// `UNAVAILABLE` — here all are refused, at the wait budget or the waiter cap
/// — and none hangs; the stalled uploads then finish and commit, because no
/// holder ever waits on the gate.
///
/// The further calls are ordered on the gate itself, not on timing: the first
/// `QUEUE_CAP` are spawned and the test waits until the gate reports them
/// queued; only then are the rest spawned, so each of those finds the queue
/// full and is refused `waiter_cap` at once, while the queued two wait out
/// their admission wait and are refused `saturated`. The admission wait is
/// `QUEUED_WAIT`, long enough that the queued two are still parked when the
/// later calls — each doing its own authorization and layer reads first —
/// reach admission.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stalled_uploads_filling_the_budget_leave_every_further_call_admitted_or_refused_unavailable(
) {
    const STALLED: usize = 3;
    const FURTHER: usize = 6;
    /// The admission queue's capacity (`max_waiters`).
    const QUEUE_CAP: usize = 2;
    /// The admission wait: the queued calls' refusal comes only after it, and
    /// every later call must reach admission before it, which on the healthy
    /// path takes milliseconds.
    const QUEUED_WAIT: Duration = Duration::from_secs(5);
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(service_with_gate(
        &tv,
        Arc::new(AllowAllPermissions),
        small_gate(QUEUED_WAIT, QUEUE_CAP),
    ));
    let gate = Arc::clone(&svc.upload_admission);

    let mut stalled = Vec::new();
    for i in 0..STALLED {
        let (session_id, object_id) =
            setup_layer_named(&tv, &format!("t3-stalled-{i}"), "layer").await;
        let ompf = field_ompf(i as u8);
        let chunks = split(&ompf, 2);
        let (request, feed) = upload::<UpdateLayerRequest>();
        feed.send(&make_update_header(&session_id, "layer", object_id));
        feed.send(&make_update_data(chunks[0].clone()));
        let task = tokio::spawn({
            let svc = Arc::clone(&svc);
            async move {
                handle_update_layer(&svc, request)
                    .await
                    .map(|r| r.into_inner())
            }
        });
        stalled.push((session_id, object_id, chunks, feed, task));
    }
    until(deadline, "the stalled uploads holding the budget", || {
        stalled
            .iter()
            .all(|(s, _, _, feed, _)| gate.receiving_uploads(s) == 1 && feed.pulled() == 2)
    })
    .await;
    assert!(
        gate.reserved_bytes() + C > SMALL_BUDGET,
        "fixture self-check: no room for another admission"
    );

    let mut prepared = Vec::new();
    for i in 0..FURTHER {
        let (session_id, object_id) =
            setup_layer_named(&tv, &format!("t3-further-{i}"), "layer").await;
        prepared.push(sent(&update_messages(
            &session_id,
            "layer",
            object_id,
            &field_ompf(100 + i as u8),
            1,
        )));
    }
    let spawn_call = |request| {
        tokio::spawn({
            let svc = Arc::clone(&svc);
            async move { handle_update_layer(&svc, request).await.map(|_| ()) }
        })
    };
    let mut feeds = Vec::new();
    let mut prepared = prepared.into_iter();
    // Fill the admission queue, and wait until the gate reports it full.
    let mut queued = Vec::new();
    for (request, feed) in prepared.by_ref().take(QUEUE_CAP) {
        feeds.push(feed);
        queued.push(spawn_call(request));
    }
    until(
        deadline,
        "the first calls filling the admission queue",
        || gate.waiters() == QUEUE_CAP,
    )
    .await;
    // Only now the rest: each finds the queue full.
    let mut overflow = Vec::new();
    for (request, feed) in prepared {
        feeds.push(feed);
        overflow.push(spawn_call(request));
    }

    let mut causes = std::collections::BTreeSet::new();
    let mut assert_refused = |what: &str, result: Result<(), Status>, cause: &str| {
        let status = result.expect_err("the budget is held by the stalled uploads");
        assert_eq!(
            status.code(),
            Code::Unavailable,
            "{what}: a full budget must answer UNAVAILABLE (retry); got {status:?}"
        );
        // The message and the retry hint both name what was full.
        let message = if cause == refusal_metadata::WAITER_CAP {
            "admission queue is full"
        } else {
            "stayed full for the whole admission wait"
        };
        assert!(
            status.message().contains(message),
            "{what}: refused `{cause}`, so the message names it; got {status:?}"
        );
        assert_refusal_detail(
            &status,
            refusal_reason::UPLOAD_CAPACITY,
            &[(refusal_metadata::CAUSE, cause)],
            CAPACITY_RETRY_DELAY,
        );
        causes.insert(cause.to_owned());
    };
    for (i, task) in overflow.into_iter().enumerate() {
        let result = settled(deadline, "a call past the full queue", task)
            .await
            .expect("the handler task must not panic");
        assert_refused(
            &format!("overflow call {i}"),
            result,
            refusal_metadata::WAITER_CAP,
        );
    }
    for (i, task) in queued.into_iter().enumerate() {
        let result = settled(deadline, "a queued call", task)
            .await
            .expect("the handler task must not panic");
        assert_refused(
            &format!("queued call {i}"),
            result,
            refusal_metadata::SATURATED,
        );
    }
    assert_eq!(
        causes.len(),
        2,
        "with {FURTHER} further calls and a queue of {QUEUE_CAP}, the queued calls wait out the \
         budget (`saturated`) and the rest find the queue full (`waiter_cap`); got {causes:?}"
    );
    drop(feeds);

    for (session_id, object_id, chunks, mut feed, task) in stalled {
        feed.send(&make_update_data(chunks[1].clone()));
        feed.end();
        let response = settled(deadline, "a stalled upload", task)
            .await
            .expect("the handler task must not panic")
            .expect("a stalled upload commits once its content arrives");
        assert_eq!(response.sequence_num, 2, "session {session_id}");
        assert_eq!(seq(&tv, object_id).await, 2);
    }
    assert_eq!(gate.reserved_bytes(), 0);
    assert!(gate.high_water_bytes() <= SMALL_BUDGET);
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — the global bound
// ═════════════════════════════════════════════════════════════════════════════

/// Twelve uploads with staggered content — six to distinct objects, six to
/// one object from distinct `client_id`s — against a budget of three
/// admissions and a short queue. The gate's high-water mark never passes the
/// budget; every answer is OK or `UNAVAILABLE`; each object's sequence number
/// counts exactly its OK answers, so no refused call wrote anything.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_uploads_stay_within_the_budget_and_refused_ones_write_nothing() {
    const DISTINCT: usize = 6;
    const SHARED: usize = 6;
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(service_with_gate(
        &tv,
        Arc::new(AllowAllPermissions),
        UploadAdmissionConfig {
            budget_bytes: SMALL_BUDGET,
            wait: Duration::from_secs(5),
            max_waiters: 4,
            max_upload_bytes: 0,
        },
    ));
    let gate = Arc::clone(&svc.upload_admission);

    /// One upload's task: OK, or the refusal it answered.
    type UploadTask = tokio::task::JoinHandle<Result<(), Status>>;
    // (object_id, tasks targeting it)
    let mut targets: Vec<(u64, Vec<UploadTask>)> = Vec::new();
    let spawn_upload = |session_id: String, object_id: u64, client: String, i: usize| {
        let svc = Arc::clone(&svc);
        tokio::spawn(async move {
            let ompf = field_ompf(i as u8);
            let (request, mut feed) = upload::<UpdateLayerRequest>();
            feed.send(&update_header_with(
                &session_id,
                "layer",
                object_id,
                &client,
                "",
            ));
            let handler = tokio::spawn(async move { handle_update_layer(&svc, request).await });
            // Staggered delivery: the "delayed streams" of the scenario.
            for chunk in split(&ompf, 3) {
                tokio::time::sleep(Duration::from_millis(5 * (i as u64 % 4))).await;
                feed.send_if_open(&make_update_data(chunk));
            }
            feed.end();
            handler
                .await
                .expect("the handler task must not panic")
                .map(|_| ())
        })
    };

    for i in 0..DISTINCT {
        let (session_id, object_id) =
            setup_layer_named(&tv, &format!("t4-distinct-{i}"), "layer").await;
        let task = spawn_upload(session_id, object_id, "client".to_owned(), i);
        targets.push((object_id, vec![task]));
    }
    let (shared_session, shared_object) = setup_layer_named(&tv, "t4-shared", "layer").await;
    let mut shared_tasks = Vec::new();
    for i in 0..SHARED {
        shared_tasks.push(spawn_upload(
            shared_session.clone(),
            shared_object,
            format!("client-{i}"),
            DISTINCT + i,
        ));
    }
    targets.push((shared_object, shared_tasks));

    let mut refused = 0usize;
    for (object_id, tasks) in targets {
        let mut ok = 0u64;
        for task in tasks {
            match settled(deadline, "an upload", task)
                .await
                .expect("no panic")
            {
                Ok(()) => ok += 1,
                Err(status) => {
                    assert_eq!(
                        status.code(),
                        Code::Unavailable,
                        "every refusal under load must be UNAVAILABLE; got {status:?}"
                    );
                    refused += 1;
                }
            }
        }
        assert_eq!(
            seq(&tv, object_id).await,
            1 + ok,
            "object {object_id}: its sequence must count exactly its OK answers — a refused \
             upload must write nothing"
        );
    }
    assert!(
        gate.high_water_bytes() <= SMALL_BUDGET,
        "the gate reserved {} bytes at its peak, past the budget of {SMALL_BUDGET}",
        gate.high_water_bytes()
    );
    assert!(
        gate.high_water_bytes() > 2 * C,
        "fixture self-check: the uploads overlapped (peak {} bytes); refused {refused}",
        gate.high_water_bytes()
    );
    assert_eq!(gate.reserved_bytes(), 0);
    assert_eq!(gate.waiters(), 0);
}

/// A structural `UpdateLayer` whose commit the fault injector fails three
/// times on its WATCH/MULTI/EXEC path commits on its fourth attempt under the
/// processing charge it reserved before the first, and releases it after.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_commit_retried_holds_its_charge_and_releases_it() {
    use crate::common::owned_delta::delta_with_diff_change_section;
    use live_edit::config::UpdateLayerBackend;
    use live_edit::grpc::upload_admission::{processing_charge, CommitPath, DeltaFootprint};
    use live_edit::storage::layer::LayerStore;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    const INJECTED: usize = 3;
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let base = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &base, "t4-structural", "layer").await;

    let injected: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(
        (0..INJECTED)
            .map(|_| fred::error::Error::new(fred::error::ErrorKind::Timeout, "t4: injected"))
            .collect(),
    ));
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Txn)
            .with_update_layer_retry_max_attempts(INJECTED as u32 + 1)
            .with_fred_error_injector_for_test(Arc::clone(&injected)),
    );
    let svc = build_service_with_layer_store(&tv, layer_store);

    let delta = delta_with_diff_change_section(0, 2, "child", 0, 0);
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let delta_len = oodf_ffi::build_delta(&delta).expect("builds").len() as u64;
    let (request, _feed) = sent(&update_messages(&session_id, "layer", object_id, &ompf, 2));
    let response = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect("the structural commit succeeds after the injected failures");
    assert_eq!(response.into_inner().sequence_num, 2);
    assert!(
        injected.lock().unwrap().is_empty(),
        "precondition: every injected failure was consumed, so the commit was retried"
    );

    let gate = &svc.upload_admission;
    let charge = processing_charge(
        CommitPath::UpdateSlow,
        ompf.len() as u64,
        delta_len,
        0,
        DeltaFootprint::of(Some(&delta)),
    );
    assert!(
        gate.high_water_bytes() >= charge,
        "the processing charge ({charge} bytes) must be reserved before the commit; peak {}",
        gate.high_water_bytes()
    );
    assert_eq!(gate.reserved_bytes(), 0, "and released when the call ends");
}

// ═════════════════════════════════════════════════════════════════════════════
// The overflow slot: an upload whose processing does not fit the budget runs
// beyond it, one at a time (T4, T5, T13)
// ═════════════════════════════════════════════════════════════════════════════

mod overflow_slot {
    //! Uploads whose processing charge exceeds the whole budget, against a
    //! gate at `SMALL_BUDGET` (floored for a 0-byte size cap, so a budget of
    //! three admissions is valid). None is refused for its size: each runs in
    //! the overflow slot, the in-budget reservations never pass the budget,
    //! and at most one upload holds the slot.
    //!
    //! The dense fixtures are sized so their charge exceeds `SMALL_BUDGET` at
    //! the current constants, and each test checks that first.
    //!
    //! The tests that need an upload to hold the slot while another arrives
    //! park the holder's `CreateLayer` in `LayerStore`'s pre-commit hook — it
    //! holds the slot from its parse until its call ends — so they are
    //! compiled only with `--features test-fault-injection`.

    use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

    use live_edit::domain::limits::MAX_NAME_BYTES;
    use live_edit::grpc::proto::OperationOutcome;
    use live_edit::grpc::upload_admission::{
        parse_charge, processing_charge, CommitPath, DeltaFootprint, RECEIVE_CHARGE_FACTOR,
    };
    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::storage::external_values::ValkeyExternalValueStore;
    use live_edit::storage::layer::{operation_request_digest, CreateOperationRecord};

    use super::super::log_capture::assert_field;
    use super::*;

    /// The section every dense field hangs off.
    const SECTION: u64 = 1;

    /// Fields in a dense `CreateLayer`.
    const DENSE: usize = 6_000;
    /// Fields in a dense fast-path `UpdateLayer`, whose per-element charge is
    /// about a quarter of the other paths'.
    const DENSE_FAST: usize = 20_000;
    /// New sections in a dense slow-path `UpdateLayer`: its per-element
    /// charge (`element[UpdateSlow]`) is under half of `Create`'s, so it
    /// needs more elements than `DENSE` to pass the budget
    /// (`assert_over_budget` checks it).
    const DENSE_SLOW: usize = 10_000;

    fn create_section(parent_id: u64, section_id: u64) -> StructCommand {
        StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: format!("s{section_id}"),
            section_type: 0,
        })
    }

    fn fields(n: usize) -> Vec<SetField> {
        (0..n)
            .map(|i| SetField {
                section_id: SECTION,
                key_name: format!("f{i}"),
                value_or_ext_hash: (i as u64).to_le_bytes().to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            })
            .collect()
    }

    /// A dense `CreateLayer` delta: `SECTION`, then `DENSE` fields on it.
    fn dense_create() -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![create_section(0, SECTION)],
            set_fields: fields(DENSE),
            ..Default::default()
        }
    }

    /// A dense fast-path `UpdateLayer` delta: fields only.
    fn dense_update_fast() -> OwnedDelta {
        OwnedDelta {
            set_fields: fields(DENSE_FAST),
            ..Default::default()
        }
    }

    /// A dense slow-path `UpdateLayer` delta: `DENSE_SLOW` new sections.
    fn dense_update_slow() -> OwnedDelta {
        OwnedDelta {
            struct_commands: (0..DENSE_SLOW as u64)
                .map(|i| create_section(SECTION, 100 + i))
                .collect(),
            ..Default::default()
        }
    }

    /// Fixture self-check: `delta` on `path` is charged more than the whole
    /// `SMALL_BUDGET`, so it cannot run in budget.
    fn assert_over_budget(delta: &OwnedDelta, path: CommitPath) -> Vec<u8> {
        let ompf = delta_to_ompf_bytes(delta, &[]);
        let delta_len = oodf_ffi::build_delta(delta).expect("builds").len() as u64;
        let charge = processing_charge(
            path,
            ompf.len() as u64,
            delta_len,
            0,
            DeltaFootprint::of(Some(delta)),
        );
        assert!(
            charge > SMALL_BUDGET,
            "fixture self-check: {path:?} charges {charge} bytes, within the {SMALL_BUDGET}-byte \
             budget"
        );
        ompf
    }

    fn overflow_gate() -> UploadAdmissionConfig {
        UploadAdmissionConfig {
            budget_bytes: SMALL_BUDGET,
            wait: Duration::from_secs(10),
            max_waiters: 8,
            max_upload_bytes: 0,
        }
    }

    fn overflow_runs(svc: &LiveEditService, rpc: &str) -> f64 {
        sample_value(
            &render(&svc.metrics),
            "live_edit_upload_overflow_runs_total",
            &[("rpc", rpc)],
        )
        .unwrap_or(f64::NAN)
    }

    /// A session and a layer whose base state is `SECTION`, created through a
    /// separate service.
    async fn base_layer(tv: &TestValkey, session_name: &str) -> (String, u64) {
        let setup = build_service(tv, Arc::new(AllowAllPermissions));
        let session_id = new_session(tv, session_name).await;
        let base = delta_to_ompf_bytes(
            &OwnedDelta {
                struct_commands: vec![create_section(0, SECTION)],
                ..Default::default()
            },
            &[],
        );
        let (request, _feed) = sent(&create_messages(&session_id, "layer", &base, 1));
        let object_id = handle_create_layer(&setup, request)
            .await
            .expect("base layer")
            .into_inner()
            .object_id;
        (session_id, object_id)
    }

    /// T13: a dense `CreateLayer`, a dense fast-path `UpdateLayer` and a dense
    /// slow-path `UpdateLayer`, each charged more than the whole budget, all
    /// commit — none is refused for its size. Each runs in the overflow slot
    /// (one `overflow_runs_total` per upload, by `rpc`), and the in-budget
    /// reservations never pass the budget. The round-1 gate refused these.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn over_budget_dense_uploads_commit_through_the_overflow_slot_on_every_path() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = service_with_gate(&tv, Arc::new(AllowAllPermissions), overflow_gate());
        let gate = &svc.upload_admission;

        let create = dense_create();
        let create_ompf = assert_over_budget(&create, CommitPath::Create);
        let session_id = new_session(&tv, "t13-overflow-create").await;
        let (request, _feed) = sent(&create_messages(&session_id, "layer", &create_ompf, 4));
        let created = settled(deadline, "the handler", handle_create_layer(&svc, request))
            .await
            .expect("an over-budget CreateLayer commits through the overflow slot");
        assert_ne!(created.into_inner().object_id, 0);
        assert_eq!(overflow_runs(&svc, "create_layer"), 1.0);

        let fast = dense_update_fast();
        assert!(
            svc.layer_store.takes_fast_path(&Some(fast.clone())),
            "fixture self-check: the fast path"
        );
        let fast_ompf = assert_over_budget(&fast, CommitPath::UpdateFast);
        let (session_id, object_id) = base_layer(&tv, "t13-overflow-fast").await;
        let (request, _feed) = sent(&update_messages(
            &session_id,
            "layer",
            object_id,
            &fast_ompf,
            4,
        ));
        let updated = settled(deadline, "the handler", handle_update_layer(&svc, request))
            .await
            .expect("an over-budget fast-path UpdateLayer commits through the overflow slot");
        assert_eq!(updated.into_inner().sequence_num, 2);
        assert_eq!(seq(&tv, object_id).await, 2);

        let slow = dense_update_slow();
        assert!(
            !svc.layer_store.takes_fast_path(&Some(slow.clone())),
            "fixture self-check: the slow path"
        );
        let slow_ompf = assert_over_budget(&slow, CommitPath::UpdateSlow);
        let (session_id, object_id) = base_layer(&tv, "t13-overflow-slow").await;
        let (request, _feed) = sent(&update_messages(
            &session_id,
            "layer",
            object_id,
            &slow_ompf,
            4,
        ));
        let updated = settled(deadline, "the handler", handle_update_layer(&svc, request))
            .await
            .expect("an over-budget slow-path UpdateLayer commits through the overflow slot");
        assert_eq!(updated.into_inner().sequence_num, 2);

        assert_eq!(overflow_runs(&svc, "update_layer"), 2.0);
        assert!(
            gate.high_water_bytes() <= SMALL_BUDGET,
            "in-budget peak {} past the budget",
            gate.high_water_bytes()
        );
        assert!(
            gate.overflow_high_water_bytes() > SMALL_BUDGET,
            "the uploads were charged beyond the budget, in the slot"
        );
        assert_eq!(gate.reserved_bytes(), 0);
        assert_eq!(gate.overflow_bytes(), 0);
    }

    /// One `SetField` whose `key_name` is one byte over `MAX_NAME_BYTES`: the
    /// cheapest §9.4 cap to exceed.
    fn over_cap_field() -> SetField {
        SetField {
            section_id: SECTION,
            key_name: "k".repeat(MAX_NAME_BYTES + 1),
            value_or_ext_hash: vec![1],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }
    }

    /// `delta` plus one field referencing `blob` as external value 1.
    fn with_external_value(mut delta: OwnedDelta, blob: &[u8]) -> OwnedDelta {
        delta.set_fields.push(SetField {
            section_id: SECTION,
            key_name: "blob".to_owned(),
            value_or_ext_hash: live_edit::storage::external_values::ext_content_hash(blob).to_vec(),
            ext_value_size: blob.len() as u64,
            ext_value_index: 1,
            set_order: 0,
        });
        delta
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// A delta that fails its per-delta checks is refused at the parse, before
    /// its processing charge. The fixture is a dense delta whose processing
    /// charge is over the whole budget while its parse charge fits, plus one
    /// name over `MAX_NAME_BYTES` and one external value, on `path`. The
    /// upload must answer `INVALID_ARGUMENT`, log one INFO completion line at
    /// stage `parse`, take no overflow slot (no `overflow_runs_total`, no slot
    /// bytes ever), hold nothing in budget beyond its receive phase and its
    /// parse charge, release its reservation, and store no external value.
    ///
    /// With the per-delta checks after the second processing step instead,
    /// this upload would take the slot, store its external value, and only
    /// then be refused by the storage-side backstop — on `UpdateLayer` at
    /// stage `storage`.
    async fn over_cap_delta_is_refused_at_its_parse_without_a_processing_charge(path: CommitPath) {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = LiveEditService {
            ext_value_store: Arc::new(ValkeyExternalValueStore::new(
                tv.client().clone(),
                tv.key_schema().clone(),
            )),
            ..service_with_gate(&tv, Arc::new(AllowAllPermissions), overflow_gate())
        };
        let gate = Arc::clone(&svc.upload_admission);
        let blob: Vec<u8> = (0..4_096).map(|i| (i % 251) as u8).collect();
        let base = match path {
            CommitPath::Create => dense_create(),
            CommitPath::UpdateFast => dense_update_fast(),
            CommitPath::UpdateSlow => unreachable!("one dense fixture per RPC"),
        };
        let mut delta = with_external_value(base, &blob);
        delta.set_fields.push(over_cap_field());
        let ompf = delta_to_ompf_bytes(&delta, &[&blob]);
        let body = ompf.len() as u64;
        let delta_len = oodf_ffi::build_delta(&delta).expect("builds").len() as u64;
        let ext_len = blob.len() as u64;
        let parse = parse_charge(body, delta_len, ext_len);
        let charge = processing_charge(
            path,
            body,
            delta_len,
            ext_len,
            DeltaFootprint::of(Some(&delta)),
        );
        assert!(
            parse <= SMALL_BUDGET && charge > SMALL_BUDGET,
            "fixture self-check: the parse charge ({parse}) fits the {SMALL_BUDGET}-byte budget \
             and the processing charge ({charge}) does not, so the late check would take the slot"
        );
        let receive_peak = C + RECEIVE_CHARGE_FACTOR * body;

        // Both handler futures are boxed. Inline, each is held again by every
        // wrapper around it (the async block, `capture`, `settled`'s
        // timeout), which in a debug build made this function's future about
        // 100 KB; the test thread's `block_on` stack holds several copies of
        // that future before the body runs (about 1.3 MB of std's default 2 MiB
        // test-thread stack), and the refused UpdateLayer's handler path
        // then overflowed it.
        let (session_id, rpc, endpoint, (result, captured), object_id) = match path {
            CommitPath::Create => {
                let session_id = new_session(&tv, "a-over-cap-create").await;
                let (request, _feed) = sent(&create_messages(&session_id, "layer", &ompf, 4));
                let outcome = settled(
                    deadline,
                    "the over-cap CreateLayer",
                    capture(async {
                        Box::pin(handle_create_layer(&svc, request))
                            .await
                            .map(|_| ())
                    }),
                )
                .await;
                (session_id, "create_layer", "CreateLayer", outcome, None)
            }
            _ => {
                let (session_id, object_id) = base_layer(&tv, "a-over-cap-update").await;
                let (request, _feed) =
                    sent(&update_messages(&session_id, "layer", object_id, &ompf, 4));
                let outcome = settled(
                    deadline,
                    "the over-cap UpdateLayer",
                    capture(async {
                        Box::pin(handle_update_layer(&svc, request))
                            .await
                            .map(|_| ())
                    }),
                )
                .await;
                (
                    session_id,
                    "update_layer",
                    "UpdateLayer",
                    outcome,
                    Some(object_id),
                )
            }
        };

        let status = result.expect_err("a delta over a §9.4 cap must be refused");
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().contains("delta error"), "{status:?}");
        let event = captured.only(RPC_FAILURE_MSG);
        assert_common_fields(event, "INFO", endpoint, "parse", "INVALID_ARGUMENT");
        assert_eq!(
            overflow_runs(&svc, rpc),
            0.0,
            "{path:?}: a refused delta must not run in the overflow slot"
        );
        assert_eq!(
            gate.overflow_high_water_bytes(),
            0,
            "{path:?}: the overflow slot was never charged"
        );
        assert_eq!(
            gate.high_water_bytes(),
            receive_peak.max(parse),
            "{path:?}: nothing beyond the receive phase and the parse charge was ever held"
        );
        assert_eq!(gate.reserved_bytes(), 0, "{path:?}: released");
        assert_eq!(gate.receiving_uploads(&session_id), 0);
        let stored: Option<Vec<u8>> = tv
            .client()
            .pool
            .get(tv.key_schema().ext_data(&hex(
                &live_edit::storage::external_values::ext_content_hash(&blob),
            )))
            .await
            .expect("GET the external value");
        assert_eq!(stored, None, "{path:?}: no external value may be stored");
        match object_id {
            Some(object_id) => assert_eq!(seq(&tv, object_id).await, 1, "nothing committed"),
            None => {
                let mapping: Option<String> = tv
                    .client()
                    .pool
                    .get(tv.key_schema().layer_mapping(&session_id, "layer"))
                    .await
                    .expect("GET the mapping");
                assert_eq!(mapping, None, "no layer created");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_over_cap_delta_is_refused_at_its_parse_without_a_processing_charge() {
        over_cap_delta_is_refused_at_its_parse_without_a_processing_charge(CommitPath::Create)
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_over_cap_delta_is_refused_at_its_parse_without_a_processing_charge() {
        over_cap_delta_is_refused_at_its_parse_without_a_processing_charge(CommitPath::UpdateFast)
            .await;
    }

    /// The INFO line an upload logs when it enters the overflow slot.
    const SLOT_RUN_MSG: &str = "upload runs beyond the upload budget";

    /// The WARN line a `CreateLayer` re-send logs when it cannot rebuild the
    /// creation state for its re-spawn.
    const RESEND_SKIPPED_MSG: &str = "CreateLayer re-send could not rebuild the creation state; \
                                      its first-base re-spawn is skipped (the ALREADY_COMMITTED \
                                      answer stands)";

    /// `messages` for a `CreateLayer` carrying `operation_id`.
    fn create_messages_with_operation(
        session_id: &str,
        operation_id: &str,
        ompf: &[u8],
    ) -> Vec<CreateLayerRequest> {
        std::iter::once(create_header_with(session_id, "layer", operation_id))
            .chain(split(ompf, 4).into_iter().map(make_create_data))
            .collect()
    }

    /// An upload that enters the overflow slot logs exactly one INFO line
    /// saying so, carrying its `rpc`, the charge it runs under and the budget
    /// it runs beyond.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_over_the_budget_logs_one_slot_run_line() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = service_with_gate(&tv, Arc::new(AllowAllPermissions), overflow_gate());
        let delta = dense_create();
        let ompf = assert_over_budget(&delta, CommitPath::Create);
        let delta_len = oodf_ffi::build_delta(&delta).expect("builds").len() as u64;
        let charge = processing_charge(
            CommitPath::Create,
            ompf.len() as u64,
            delta_len,
            0,
            DeltaFootprint::of(Some(&delta)),
        );
        let session_id = new_session(&tv, "slot-run-line").await;
        let (request, _feed) = sent(&create_messages(&session_id, "layer", &ompf, 4));

        let (result, captured) = settled(
            deadline,
            "the over-budget CreateLayer",
            capture(handle_create_layer(&svc, request)),
        )
        .await;
        result.expect("an over-budget CreateLayer commits through the overflow slot");
        let line = captured.only(SLOT_RUN_MSG);
        assert_eq!(line["level"], "INFO", "{line}");
        assert_field(line, "rpc", "create_layer".into());
        assert_field(line, "charge_bytes", charge.into());
        assert_field(line, "upload_budget_bytes", SMALL_BUDGET.into());
    }

    /// A `CreateLayer` re-send answered `ALREADY_COMMITTED` never waits on the
    /// gate: its re-spawn charges are taken in budget or not at all. Re-sending
    /// the over-budget dense create with its operation id answers
    /// `ALREADY_COMMITTED` with the original `object_id`; the processing
    /// charge, which does not fit, only skips the re-spawn (one WARN). It takes
    /// no slot (`overflow_runs_total` stays at the original's one) and counts
    /// no `refused_growth`: no client is refused for it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_resend_of_an_over_budget_create_is_answered_without_the_slot() {
        const OPERATION: &str = "resend-over-budget";
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = service_with_gate(&tv, Arc::new(AllowAllPermissions), overflow_gate());
        let gate = Arc::clone(&svc.upload_admission);
        let ompf = assert_over_budget(&dense_create(), CommitPath::Create);
        let session_id = new_session(&tv, "resend-over-budget").await;

        let (request, _feed) = sent(&create_messages_with_operation(
            &session_id,
            OPERATION,
            &ompf,
        ));
        let original = settled(deadline, "the original", handle_create_layer(&svc, request))
            .await
            .expect("the original commits through the overflow slot")
            .into_inner();
        assert_eq!(overflow_runs(&svc, "create_layer"), 1.0);

        let (request, _feed) = sent(&create_messages_with_operation(
            &session_id,
            OPERATION,
            &ompf,
        ));
        let (result, captured) = settled(
            deadline,
            "the re-send",
            capture(handle_create_layer(&svc, request)),
        )
        .await;
        let resent = result.expect("the re-send is answered").into_inner();
        assert_eq!(resent.object_id, original.object_id);
        assert_eq!(
            resent.operation_outcome,
            OperationOutcome::AlreadyCommitted as i32
        );
        assert_eq!(
            overflow_runs(&svc, "create_layer"),
            1.0,
            "the re-send must not take the slot"
        );
        assert_eq!(
            sample_value(
                &render(&svc.metrics),
                "live_edit_upload_admission_total",
                &[("rpc", "create_layer"), ("result", "refused_growth")]
            ),
            Some(0.0),
            "a re-spawn charge refused after the content is received refuses no client"
        );
        let warned = captured.only(RESEND_SKIPPED_MSG);
        assert!(
            warned["fields"]["error"]
                .as_str()
                .is_some_and(|e| e.contains("upload memory budget")),
            "the re-spawn is skipped for want of budget: {warned}"
        );
        assert!(captured.with_message(SLOT_RUN_MSG).is_empty());
        assert_eq!(gate.reserved_bytes(), 0);
        assert_eq!(gate.overflow_bytes(), 0);
    }

    /// The re-send path runs the per-delta checks before its processing
    /// charge too. A re-send whose recorded digest matches an over-cap delta
    /// — seeded here, since no committed call can carry one — whose
    /// processing charge is over the budget answers `ALREADY_COMMITTED` and
    /// skips its re-spawn for the delta's own error ("delta error"), not for
    /// want of budget, which is what checking after the charge would report.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_resend_of_an_over_cap_delta_skips_its_respawn_for_the_delta_error() {
        const OPERATION: &str = "resend-over-cap";
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = service_with_gate(&tv, Arc::new(AllowAllPermissions), overflow_gate());
        let session_id = new_session(&tv, "resend-over-cap").await;

        let (request, _feed) = sent(&create_messages_with_operation(
            &session_id,
            OPERATION,
            &field_ompf(1),
        ));
        let original = settled(deadline, "the original", handle_create_layer(&svc, request))
            .await
            .expect("the original commits")
            .into_inner();

        let mut over_cap = dense_create();
        over_cap.set_fields.push(over_cap_field());
        let over_cap_ompf = assert_over_budget(&over_cap, CommitPath::Create);
        let record_key = tv
            .key_schema()
            .create_layer_operation(&session_id, OPERATION);
        let recorded: String = tv
            .client()
            .pool
            .get(&record_key)
            .await
            .expect("GET the operation record");
        let mut record =
            CreateOperationRecord::parse(&record_key, &recorded).expect("a valid record");
        record.digest = operation_request_digest("layer", &over_cap_ompf);
        let _: () = tv
            .client()
            .pool
            .set(&record_key, record.render(), None, None, false)
            .await
            .expect("seed the over-cap digest");

        let (request, _feed) = sent(&create_messages_with_operation(
            &session_id,
            OPERATION,
            &over_cap_ompf,
        ));
        let (result, captured) = settled(
            deadline,
            "the re-send",
            capture(handle_create_layer(&svc, request)),
        )
        .await;
        let resent = result.expect("the re-send is answered").into_inner();
        assert_eq!(resent.object_id, original.object_id);
        assert_eq!(
            resent.operation_outcome,
            OperationOutcome::AlreadyCommitted as i32
        );
        let warned = captured.only(RESEND_SKIPPED_MSG);
        assert!(
            warned["fields"]["error"]
                .as_str()
                .is_some_and(|e| e.contains("delta error")),
            "the re-spawn is skipped for the delta's own error: {warned}"
        );
        assert_eq!(overflow_runs(&svc, "create_layer"), 0.0);
        assert_eq!(svc.upload_admission.reserved_bytes(), 0);
    }

    /// A `CreateLayer` pre-commit hook that parks the FIRST call to reach it
    /// until released, and lets every later one through: the parked call keeps
    /// holding the overflow slot it took at its parse.
    #[cfg(feature = "test-fault-injection")]
    struct ParkedHolder {
        hook: live_edit::storage::watched_txn::WatchWindowHook,
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    #[cfg(feature = "test-fault-injection")]
    impl ParkedHolder {
        fn new() -> Self {
            let entered = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let calls = Arc::new(AtomicUsize::new(0));
            let hook: live_edit::storage::watched_txn::WatchWindowHook = Arc::new({
                let entered = Arc::clone(&entered);
                let release = Arc::clone(&release);
                move |_client| {
                    let first = calls.fetch_add(1, Ordering::SeqCst) == 0;
                    let entered = Arc::clone(&entered);
                    let release = Arc::clone(&release);
                    Box::pin(async move {
                        if first {
                            entered.notify_one();
                            release.notified().await;
                        }
                    })
                }
            });
            Self {
                hook,
                entered,
                release,
            }
        }

        /// A service at the overflow gate whose `CreateLayer` commits pass this
        /// hook.
        fn service(&self, tv: &TestValkey) -> LiveEditService {
            self.service_with(tv, |store| store)
        }

        /// [`Self::service`], with `seam` applied to its `LayerStore` too.
        fn service_with(
            &self,
            tv: &TestValkey,
            seam: impl FnOnce(
                live_edit::storage::layer::LayerStore,
            ) -> live_edit::storage::layer::LayerStore,
        ) -> LiveEditService {
            let store = Arc::new(seam(
                live_edit::storage::layer::LayerStore::new(
                    tv.client().clone(),
                    tv.key_schema().clone(),
                    4096,
                )
                .with_create_layer_pre_commit_hook_for_test(Arc::clone(&self.hook)),
            ));
            let base = build_service_with_layer_store(tv, store);
            LiveEditService {
                upload_admission: Arc::new(UploadAdmissionGate::new(
                    overflow_gate(),
                    Some(Arc::clone(&base.metrics)),
                )),
                ..base
            }
        }

        /// Start an over-budget `CreateLayer` and wait, until the test's
        /// `deadline`, for it to park at its commit point, holding the
        /// overflow slot.
        async fn park(
            &self,
            tv: &TestValkey,
            svc: &Arc<LiveEditService>,
            session_name: &str,
            deadline: tokio::time::Instant,
        ) -> tokio::task::JoinHandle<Result<(), Status>> {
            let ompf = assert_over_budget(&dense_create(), CommitPath::Create);
            let session_id = new_session(tv, session_name).await;
            let (request, _feed) = sent(&create_messages(&session_id, "layer", &ompf, 4));
            let task = tokio::spawn({
                let svc = Arc::clone(svc);
                async move { handle_create_layer(&svc, request).await.map(|_| ()) }
            });
            settled(
                deadline,
                "the holder parked at its commit point",
                self.entered.notified(),
            )
            .await;
            assert!(
                svc.upload_admission.overflow_bytes() > SMALL_BUDGET,
                "precondition: the parked holder holds the overflow slot"
            );
            task
        }

        fn release(&self) {
            self.release.notify_one();
        }
    }

    /// T4: two over-budget uploads at once. The second queues for the slot
    /// while the first holds it — never two holders — and runs after it; both
    /// commit, `overflow_runs_total` counts 2, and the in-budget reservations
    /// never pass the budget.
    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_over_budget_uploads_run_one_at_a_time_in_the_overflow_slot() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let parked = ParkedHolder::new();
        let svc = Arc::new(parked.service(&tv));
        let gate = Arc::clone(&svc.upload_admission);
        let first = parked.park(&tv, &svc, "t4-overflow-first", deadline).await;
        let first_charge = gate.overflow_bytes();

        let ompf = assert_over_budget(&dense_create(), CommitPath::Create);
        let session_id = new_session(&tv, "t4-overflow-second").await;
        let (request, _feed) = sent(&create_messages(&session_id, "layer", &ompf, 4));
        let second = tokio::spawn({
            let svc = Arc::clone(&svc);
            async move { handle_create_layer(&svc, request).await.map(|_| ()) }
        });
        until(deadline, "the second upload queued for the slot", || {
            gate.processing_waiters() == 1
        })
        .await;
        assert_eq!(
            gate.overflow_bytes(),
            first_charge,
            "the first upload still holds the slot alone"
        );
        assert!(gate.reserved_bytes() <= SMALL_BUDGET);

        parked.release();
        settled(deadline, "the first upload", first)
            .await
            .expect("no panic")
            .expect("the first over-budget upload commits");
        settled(deadline, "the second upload", second)
            .await
            .expect("no panic")
            .expect("the second over-budget upload commits after the first");
        assert_eq!(overflow_runs(&svc, "create_layer"), 2.0);
        assert!(gate.high_water_bytes() <= SMALL_BUDGET);
        assert_eq!(gate.overflow_bytes(), 0);
        assert_eq!(gate.processing_waiters(), 0);
        assert_eq!(gate.reserved_bytes(), 0);
    }

    /// T5: the start of graceful shutdown's paced stream ending (`begun`, t0)
    /// while waiting for the slot ends the wait at once with `UNAVAILABLE
    /// "server shutting down"` and its retry hint — one INFO completion line at
    /// stage `parse`, no "upload runs beyond the upload budget" line, nothing
    /// applied, the reservation released — while the holder runs to its end and
    /// commits. The service shutdown token is not yet cancelled when the waiter
    /// is refused: its wait races `begun`.
    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_waiting_for_the_slot_at_shutdown_answers_server_shutting_down() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let parked = ParkedHolder::new();
        let svc = Arc::new(parked.service(&tv));
        let gate = Arc::clone(&svc.upload_admission);
        let holder = parked
            .park(&tv, &svc, "t5-slot-shutdown-holder", deadline)
            .await;

        let ompf = assert_over_budget(&dense_create(), CommitPath::Create);
        let session_id = new_session(&tv, "t5-slot-shutdown-waiter").await;
        let (request, _feed) = sent(&create_messages(&session_id, "layer", &ompf, 4));
        let shutdown = svc.shutdown.clone();
        let begin_once_queued = async {
            until(deadline, "the waiter queued for the slot", || {
                gate.processing_waiters() == 1
            })
            .await;
            begin_paced_ending_held_open(&shutdown).await
        };
        let ((result, captured), paced) = tokio::join!(
            settled(
                deadline,
                "the waiting CreateLayer",
                capture(handle_create_layer(&svc, request))
            ),
            begin_once_queued
        );
        let status = result.expect_err("a slot wait must end at shutdown");
        assert_eq!(status.code(), Code::Unavailable, "{status:?}");
        crate::common::refusal_detail::assert_shutdown_status(&status);
        assert!(
            !svc.shutdown.is_cancelled(),
            "the waiting upload was refused at t0, while the paced ending was still under way"
        );
        let event = captured.only(RPC_FAILURE_MSG);
        assert_common_fields(event, "INFO", "CreateLayer", "parse", "UNAVAILABLE");
        assert!(
            captured
                .with_message("upload runs beyond the upload budget")
                .is_empty(),
            "a waiter that never ran in the slot logs no slot-run line"
        );
        let mapping: Option<String> = tv
            .client()
            .pool
            .get(tv.key_schema().layer_mapping(&session_id, "layer"))
            .await
            .expect("GET the mapping");
        assert_eq!(mapping, None, "nothing applied");
        assert_eq!(gate.processing_waiters(), 0);
        assert_eq!(
            gate.reserved_bytes(),
            0,
            "the waiter's reservation is released"
        );

        parked.release();
        settled(deadline, "the holder", holder)
            .await
            .expect("no panic")
            .expect("shutdown does not refuse the slot's holder");
        assert_eq!(overflow_runs(&svc, "create_layer"), 1.0);
        assert_eq!(gate.overflow_bytes(), 0);
        drop(paced);
    }

    /// T5: the RPC deadline while waiting for the slot answers
    /// `DEADLINE_EXCEEDED` and releases the dropped call's reservation.
    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_waiting_for_the_slot_meets_the_deadline_and_releases_its_reservation() {
        /// Long enough for the call to be received and parsed, so it is the
        /// slot wait that the deadline ends (checked below).
        const DEADLINE_MS: u64 = 5_000;
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let parked = ParkedHolder::new();
        let svc = Arc::new(parked.service(&tv));
        let gate = Arc::clone(&svc.upload_admission);
        let holder = parked
            .park(&tv, &svc, "t5-slot-deadline-holder", deadline)
            .await;
        let server = DeadlineServer::start((*svc).clone(), DEADLINE_MS, deadline).await;

        let ompf = assert_over_budget(&dense_create(), CommitPath::Create);
        let session_id = new_session(&tv, "t5-slot-deadline-waiter").await;
        let messages = create_messages(&session_id, "layer", &ompf, 4);
        let mut client = server.client().await;
        let call = tokio::spawn(async move {
            client
                .create_layer(with_bearer(futures::stream::iter(messages)))
                .await
                .map(|_| ())
        });
        until(deadline, "the call queued for the slot", || {
            gate.processing_waiters() == 1
        })
        .await;
        let status = settled(deadline, "the deadline-bound CreateLayer", call)
            .await
            .expect("no panic")
            .expect_err("a call still waiting for the slot at its deadline must not succeed");
        assert_eq!(status.code(), Code::DeadlineExceeded, "{status:?}");
        until(deadline, "the waiter's reservation released", || {
            gate.processing_waiters() == 0 && gate.reserved_bytes() == 0
        })
        .await;

        parked.release();
        settled(deadline, "the holder", holder)
            .await
            .expect("no panic")
            .expect("the holder commits");
        assert_eq!(gate.overflow_bytes(), 0);
    }

    /// The first processing step — the parse charge — is held BEFORE the delta
    /// is parsed. With the slot held, an upload whose parse charge alone is
    /// over the budget (a 3 MiB delta part, at `k_parse`) and whose delta
    /// bytes are not a delta queues for the slot; it does not answer
    /// `INVALID_ARGUMENT` at once, which is what parsing before the charge
    /// would do. Once the holder finishes, it takes the slot, parses, and is
    /// refused.
    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_holds_its_parse_charge_before_parsing_its_delta() {
        const DELTA_PART: usize = 3 * 1024 * 1024;
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let parked = ParkedHolder::new();
        let svc = Arc::new(parked.service(&tv));
        let gate = Arc::clone(&svc.upload_admission);
        let holder = parked.park(&tv, &svc, "parse-first-holder", deadline).await;

        let ompf = live_edit::domain::ompf::OmpfContainer::serialize(&vec![0xEE; DELTA_PART], &[])
            .expect("an OMPF container around bytes that are not a delta");
        let parse = parse_charge(ompf.len() as u64, DELTA_PART as u64, 0);
        assert!(
            parse > SMALL_BUDGET,
            "fixture self-check: the parse charge ({parse}) alone is over the budget"
        );
        let session_id = new_session(&tv, "parse-first-waiter").await;
        let (request, _feed) = sent(&create_messages(&session_id, "layer", &ompf, 4));
        let call = tokio::spawn({
            let svc = Arc::clone(&svc);
            async move { handle_create_layer(&svc, request).await.map(|_| ()) }
        });
        until(deadline, "the malformed upload queued, or answered", || {
            gate.processing_waiters() == 1 || call.is_finished()
        })
        .await;
        assert_eq!(
            gate.processing_waiters(),
            1,
            "the parse charge must be held before the parse: the upload waits for the slot \
             rather than being refused at once"
        );

        parked.release();
        settled(deadline, "the holder", holder)
            .await
            .expect("no panic")
            .expect("the holder commits");
        let status = settled(deadline, "the malformed upload", call)
            .await
            .expect("no panic")
            .expect_err("bytes that are not a delta are refused once parsed");
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert_eq!(overflow_runs(&svc, "create_layer"), 2.0);
        assert_eq!(gate.reserved_bytes(), 0);
        assert_eq!(gate.overflow_bytes(), 0);
    }

    /// I3's other leg: an `UpdateLayer` with an operation id starts its
    /// start-record read as soon as its body is received — BEFORE its
    /// processing steps, which may wait. With the slot held and a parse charge
    /// over the budget (a 3 MiB delta part that is not a delta), the upload
    /// waits in the processing queue; the layer store's one injected
    /// start-record failure, consumed synchronously when the read starts, is
    /// already gone while it waits. Once the holder finishes, the upload takes
    /// the slot, parses, and is refused `INVALID_ARGUMENT`, nothing committed.
    #[cfg(feature = "test-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_starts_its_start_record_read_before_its_processing_wait() {
        use std::collections::VecDeque;
        use std::sync::Mutex;

        const DELTA_PART: usize = 3 * 1024 * 1024;
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let failures: Arc<Mutex<VecDeque<fred::error::Error>>> =
            Arc::new(Mutex::new(VecDeque::from([fred::error::Error::new(
                fred::error::ErrorKind::IO,
                "i3: start-record read failed",
            )])));
        let parked = ParkedHolder::new();
        let svc = Arc::new(parked.service_with(&tv, |store| {
            store.with_start_record_read_fred_error_injector_for_test(Arc::clone(&failures))
        }));
        let gate = Arc::clone(&svc.upload_admission);
        let (session_id, object_id) = base_layer(&tv, "i3-processing-wait").await;
        let holder = parked
            .park(&tv, &svc, "i3-processing-wait-holder", deadline)
            .await;

        let ompf = live_edit::domain::ompf::OmpfContainer::serialize(&vec![0xEE; DELTA_PART], &[])
            .expect("an OMPF container around bytes that are not a delta");
        let parse = parse_charge(ompf.len() as u64, DELTA_PART as u64, 0);
        assert!(
            parse > SMALL_BUDGET,
            "fixture self-check: the parse charge ({parse}) alone is over the budget"
        );
        let messages: Vec<UpdateLayerRequest> = std::iter::once(update_header_with(
            &session_id,
            "layer",
            object_id,
            "client",
            "i3-processing-wait",
        ))
        .chain(split(&ompf, 4).into_iter().map(make_update_data))
        .collect();
        let (request, _feed) = sent(&messages);
        let task = tokio::spawn({
            let svc = Arc::clone(&svc);
            async move { handle_update_layer(&svc, request).await.map(|_| ()) }
        });
        until(
            deadline,
            "the upload queued for the slot, or answered",
            || gate.processing_waiters() == 1 || task.is_finished(),
        )
        .await;
        assert_eq!(
            gate.processing_waiters(),
            1,
            "fixture self-check: the upload waits in the processing queue"
        );
        assert!(
            failures.lock().unwrap().is_empty(),
            "the start-record read must start once the body is received, before the processing \
             wait: the injected failure is still queued"
        );

        parked.release();
        settled(deadline, "the holder", holder)
            .await
            .expect("no panic")
            .expect("the holder commits");
        let status = settled(deadline, "the waiting UpdateLayer", task)
            .await
            .expect("no panic")
            .expect_err("bytes that are not a delta are refused once parsed");
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert_eq!(seq(&tv, object_id).await, 1, "nothing committed");
        assert_eq!(gate.reserved_bytes(), 0);
        assert_eq!(gate.overflow_bytes(), 0);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// I3 — the start-record read starts immediately after the body
// ═════════════════════════════════════════════════════════════════════════════

/// I3's "immediately after the body" leg: an authorized `UpdateLayer` with an
/// operation id starts its start-record read only once its whole content is
/// received. The layer store fails the next start-record read (one injected
/// failure, consumed synchronously when the read starts). Stalled mid-body,
/// the upload has not started it: the failure is still queued. Once the body
/// ends the read starts, the failure is consumed, and the call answers
/// `UNAVAILABLE` `UPSTREAM_UNAVAILABLE` (refused before its commit marker,
/// b.xd1) with nothing committed.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_starts_its_start_record_read_only_after_its_whole_body() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use live_edit::storage::layer::LayerStore;

    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let failures: Arc<Mutex<VecDeque<fred::error::Error>>> =
        Arc::new(Mutex::new(VecDeque::from([fred::error::Error::new(
            fred::error::ErrorKind::IO,
            "i3: start-record read failed",
        )])));
    let base = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_start_record_read_fred_error_injector_for_test(Arc::clone(&failures)),
    );
    let recovery =
        build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&base.metrics));
    let svc = Arc::new(LiveEditService {
        layer_store,
        recovery,
        ..base
    });
    let gate = Arc::clone(&svc.upload_admission);
    let (session_id, object_id) = setup_layer_named(&tv, "i3-after-body", "layer").await;
    let chunks = split(&field_ompf(1), 2);

    let (request, mut feed) = upload::<UpdateLayerRequest>();
    feed.send(&update_header_with(
        &session_id,
        "layer",
        object_id,
        "client",
        "i3-operation",
    ));
    feed.send(&make_update_data(chunks[0].clone()));
    let task = tokio::spawn({
        let svc = Arc::clone(&svc);
        async move { handle_update_layer(&svc, request).await.map(|_| ()) }
    });
    until(
        deadline,
        "the upload authorized, admitted and reading its body",
        || (gate.receiving_uploads(&session_id) == 1 && feed.pulled() == 2) || task.is_finished(),
    )
    .await;
    assert_eq!(
        failures.lock().unwrap().len(),
        1,
        "mid-body, the start-record read must not have started"
    );
    assert!(
        !task.is_finished(),
        "the upload ended before its body did: it must still be reading its content"
    );

    feed.send(&make_update_data(chunks[1].clone()));
    feed.end();
    let status = settled(deadline, "the UpdateLayer", task)
        .await
        .expect("no panic")
        .expect_err("the call's start-record read fails");
    crate::common::refusal_detail::assert_upstream_unavailable(&status);
    assert!(
        failures.lock().unwrap().is_empty(),
        "the read started once the body was received: the injected failure was consumed"
    );
    assert_eq!(seq(&tv, object_id).await, 1, "nothing committed");
    assert_eq!(gate.reserved_bytes(), 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// b.26e Stage 2 — the handler's object-state hint and the in-commit recharge
// ═════════════════════════════════════════════════════════════════════════════

/// A layer of `n` sections under the root, ids from 100, each named
/// `s{id}`: the base of the object-state tests below.
#[cfg(feature = "test-fault-injection")]
fn sections_under_root(n: u64) -> live_edit::domain::types::ObjectState {
    use live_edit::domain::types::{ObjectState, Section};

    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_owned(),
        ..Default::default()
    };
    let ids: Vec<u64> = (100..100 + n).collect();
    for (order, id) in ids.iter().enumerate() {
        state.sections.insert(
            *id,
            Section {
                section_id: *id,
                parent_id: 0,
                name: format!("s{id}"),
                section_type: 1,
                section_order: order as u32,
                first_version: 1,
                last_version: 1,
            },
        );
    }
    state.children.insert(0, ids);
    state
}

/// Sections of the hint tests' base (the processing-step wait's and the
/// in-budget RP0 refusal's), each named [`HINT_BASE_NAME_BYTES`] long: enough that its structural-cache entry's charge (the hint: the
/// entry's state and plan indexes) exceeds the receive allowance `A` an
/// upload gives back at the end of its body, the most the budget has room
/// for when the processing step runs, and the test's receive slack (about
/// 8.65 MB). With short names the hint does not grow linearly with the
/// sections: 6,270,144 B at 30,000 and 7,053,504 B at 46,000, the sections
/// table's bucket array the same, so short names would need about 58,000
/// sections, and every structural commit on the base rewrites the root's
/// whole children list, which at 60,000 overran the test's time budget in a
/// debug build. Long names carry the weight instead, about 250 B more per
/// section. The test's self-check confirms the margin.
#[cfg(feature = "test-fault-injection")]
const HINT_BASE_SECTIONS: u64 = 30_000;

/// Each hint-test base section's name length, in bytes
/// ([`HINT_BASE_SECTIONS`]).
#[cfg(feature = "test-fault-injection")]
const HINT_BASE_NAME_BYTES: usize = 256;

/// A structural `UpdateLayer` on a cached object takes its
/// object-state hint — the cached entry it plans on in place — with its second
/// processing step, a step that may wait holding nothing. With the budget
/// full (a filler, and room only for the call's own admission, so at the
/// processing step only its receive allowance is free) and the overflow slot
/// held, the call waits at that processing step, before its commit, with the
/// object's per-object permits free; once the slot is free it runs there and
/// commits with no recharge. Its commit's first charge point, the hit's
/// repair read (RP0), grows past the hint by that read's reply term, and in
/// the slot that growth is booked in place, so it is granted. Without the
/// hint the processing step fits, the commit's RP0 is refused, and the call
/// recharges (`live_edit_update_layer_recharges_total{path="structural"}` 1);
/// with the hint held in budget and no room for that growth, RP0 is refused
/// too
/// ([`update_layer_cached_edit_with_its_hint_in_budget_recharges_once_at_its_first_repair_read_fsz`]).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_cached_edit_waits_for_its_hint_at_the_processing_step_b26e() {
    use live_edit::config::UpdateLayerBackend;
    use live_edit::grpc::upload_admission::{object_state_charge_for_test, UploadRpc};
    use live_edit::storage::commit_charge::ObjectStateNeed;
    use live_edit::storage::external_values::MockExternalValueStore;
    use live_edit::storage::layer::LayerStore;
    use live_edit::storage::test_support::relay::RelayedUpdates;
    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};
    use tokio_util::sync::CancellationToken;

    const K: usize = 16;
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("hint"));
    let ls = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_update_layer_per_object_concurrency(K as u32)
            .with_metrics(Arc::clone(&metrics)),
    );
    let svc = LiveEditService {
        upload_admission: Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes: SMALL_BUDGET,
                wait: Duration::from_secs(10),
                max_waiters: 8,
                max_upload_bytes: 0,
            },
            None,
        )),
        ..build_service_with_layer_store(&tv, Arc::clone(&ls))
    };
    let gate = Arc::clone(&svc.upload_admission);

    // A large base, cached on this store by one structural commit.
    let session_id = new_session(&tv, "hint").await;
    let mut base = sections_under_root(HINT_BASE_SECTIONS);
    for section in base.sections.values_mut() {
        section.name = format!("{:x<HINT_BASE_NAME_BYTES$}", section.name);
    }
    let object_id = ls
        .create_layer(
            &session_id,
            "layer",
            "test-user",
            &base,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("the base is created");
    let create = |id: u64| OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: id,
            section_name: format!("new{id}"),
            section_type: 1,
        })],
        ..Default::default()
    };
    settled(
        deadline,
        "the warming commit",
        ls.update_layer_relayed(&session_id, "layer", object_id, Some(create(1)), "warm", 1),
    )
    .await
    .expect("the warming commit lands");
    let weight = ls
        .structural_cache()
        .weight_hint(object_id)
        .expect("fixture: the base is cached");
    let hint = object_state_charge_for_test(&ObjectStateNeed::cached(weight, Default::default()));
    // Room for the update's receive growth beside its admission.
    const RECEIVE_SLACK: u64 = 256 * 1024;
    assert!(
        hint > RECEIVE_ALLOWANCE_BYTES + RECEIVE_SLACK,
        "fixture self-check: the hint ({hint} B) exceeds the receive allowance and the slack, \
         the room the processing step finds"
    );
    let recharges = || {
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_recharges_total",
            &[("path", "structural")],
        )
        .unwrap_or(0.0) as u64
    };
    let before_seq = seq(&tv, object_id).await;

    // The slot held, and the budget full but for one admission.
    let shutdown = CancellationToken::new();
    let mut slot = gate
        .admit(UploadRpc::UpdateLayer, "hint-slot", &shutdown)
        .await
        .expect("the slot holder is admitted");
    slot.end_receive();
    assert!(
        slot.reserve_processing(SMALL_BUDGET + 1, &shutdown)
            .await
            .expect("the slot is free")
            .entered_overflow,
        "fixture: the slot is held"
    );
    let mut filler = gate
        .admit(UploadRpc::UpdateLayer, "hint-filler", &shutdown)
        .await
        .expect("the filler is admitted");
    let others = gate.reserved_bytes() - filler.reserved();
    filler
        .try_grow_in_budget(SMALL_BUDGET - others - C - RECEIVE_SLACK)
        .expect("the filler leaves room for one admission and its receive growth");

    let ompf = delta_to_ompf_bytes(&create(2), &[]);
    let (request, _feed) = sent(&update_messages(&session_id, "layer", object_id, &ompf, 2));
    let task = tokio::spawn({
        let svc = Arc::new(svc);
        async move { handle_update_layer(&svc, request).await.map(|_| ()) }
    });
    until(
        deadline,
        "the update waits at its processing step, or answered",
        || gate.processing_waiters() == 1 || task.is_finished(),
    )
    .await;
    if task.is_finished() {
        panic!(
            "fixture: the update must wait in the processing queue; it answered {:?}",
            task.await
        );
    }
    assert_eq!(
        gate.processing_waiters(),
        1,
        "fixture: the update waits in the processing queue"
    );
    assert!(
        ls.admission_permits_available_for_test(object_id)
            .is_none_or(|available| available == (1, K)),
        "the waiting update holds no per-object permit"
    );
    assert_eq!(
        recharges(),
        0,
        "it waits at the processing step for its hint, not in a recharge"
    );

    drop(slot);
    settled(deadline, "the update", task)
        .await
        .expect("no panic")
        .expect("the update commits in the freed slot");
    assert_eq!(seq(&tv, object_id).await, before_seq + 1, "one commit");
    assert_eq!(
        recharges(),
        0,
        "the commit's first charge point (RP0) grew past the hint in the slot, which books \
         growth in place: no recharge"
    );
    drop(filler);
}

/// The hint held in budget (b.fsz (a)): a structural `UpdateLayer` on a
/// cached object takes its processing step and its object-state hint in
/// budget, and the budget is then full — a filler leaves the call room for
/// its step and half of what its first charge point grows past the hint —
/// with the overflow slot held and a processing step queued for it. On a
/// hit that first charge point is the repair's first read (RP0), which grows
/// past the hint by that read's reply term: it does not fit, and the slot is
/// not free, so it is refused. The call releases its per-object permits and
/// entry, keeps its processing step and hint, and waits once for RP0's
/// target; the filler's release grants it in budget, and it commits once.
///
/// Its reservation is read off the gate's charge observer, net of the
/// filler and the queued step (the slot holder is off budget): it never
/// falls below the processing step and hint from that step's grant through
/// the recharge's, and the recharge's grant holds the processing charge and
/// RP0's whole target, in budget.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_cached_edit_with_its_hint_in_budget_recharges_once_at_its_first_repair_read_fsz(
) {
    use live_edit::config::UpdateLayerBackend;
    use live_edit::domain::ompf::OmpfContainer;
    use live_edit::grpc::upload_admission::{
        object_state_charge_for_test, processing_charge, CommitPath, DeltaFootprint, GateCharge,
        UploadRpc,
    };
    use live_edit::storage::commit_charge::{ObjectStateNeed, RepairTransient};
    use live_edit::storage::external_values::MockExternalValueStore;
    use live_edit::storage::layer::LayerStore;
    use live_edit::storage::structural_cache_sync::{
        STRUCTURAL_REPAIR_INITIAL_BYTES, STRUCTURAL_REPAIR_INITIAL_ENTRIES,
        STRUCTURAL_REPAIR_MAX_ENTRIES,
    };
    use live_edit::storage::test_support::relay::RelayedUpdates;
    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};
    use tokio_util::sync::CancellationToken;

    const K: usize = 16;
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("hint-in-budget"));
    let ls = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_update_layer_per_object_concurrency(K as u32)
            .with_metrics(Arc::clone(&metrics)),
    );
    let svc = LiveEditService {
        upload_admission: Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes: SMALL_BUDGET,
                wait: Duration::from_secs(10),
                max_waiters: 8,
                max_upload_bytes: 0,
            },
            None,
        )),
        ..build_service_with_layer_store(&tv, Arc::clone(&ls))
    };
    let gate = Arc::clone(&svc.upload_admission);

    // A large base of long-named sections, as the hint test's, cached on
    // this store by one structural commit.
    let session_id = new_session(&tv, "hint-in-budget").await;
    let mut base = sections_under_root(HINT_BASE_SECTIONS);
    for section in base.sections.values_mut() {
        section.name = format!("{:x<HINT_BASE_NAME_BYTES$}", section.name);
    }
    let object_id = ls
        .create_layer(
            &session_id,
            "layer",
            "test-user",
            &base,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("the base is created");
    let create = |id: u64| OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: id,
            section_name: format!("new{id}"),
            section_type: 1,
        })],
        ..Default::default()
    };
    settled(
        deadline,
        "the warming commit",
        ls.update_layer_relayed(&session_id, "layer", object_id, Some(create(1)), "warm", 1),
    )
    .await
    .expect("the warming commit lands");
    let weight = ls
        .structural_cache()
        .weight_hint(object_id)
        .expect("fixture: the base is cached");

    // The call's processing step (its processing charge and the hint), and
    // what RP0 grows it to.
    let delta = create(2);
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let delta_len = OmpfContainer::parse(&ompf)
        .expect("the update's OMPF parses")
        .delta_bytes
        .len() as u64;
    let processing = processing_charge(
        CommitPath::UpdateSlow,
        ompf.len() as u64,
        delta_len,
        0,
        DeltaFootprint::of(Some(&delta)),
    );
    let hint = object_state_charge_for_test(&ObjectStateNeed::cached(weight, Default::default()));
    let rp0 = object_state_charge_for_test(&ObjectStateNeed::cached_repair(
        weight,
        RepairTransient {
            reply_entry_count: STRUCTURAL_REPAIR_INITIAL_ENTRIES.min(STRUCTURAL_REPAIR_MAX_ENTRIES),
            reply_bytes: STRUCTURAL_REPAIR_INITIAL_BYTES,
            ..RepairTransient::default()
        },
    ));
    let step = processing + hint;
    let growth = rp0 - hint;
    assert!(
        hint > C && growth > 0,
        "fixture self-check: the hint ({hint} B) holds more than the call's admission ({C} B), so \
         room for the step admits the call, and RP0 grows past it ({growth} B)"
    );
    let recharges = |path: &str| {
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_recharges_total",
            &[("path", path)],
        )
        .unwrap_or(0.0) as u64
    };
    let before_seq = seq(&tv, object_id).await;

    // The slot held, a processing step queued for it, and the budget full
    // but for the call's step and half of RP0's growth past it.
    let shutdown = CancellationToken::new();
    let mut slot = gate
        .admit(UploadRpc::UpdateLayer, "rp0-slot", &shutdown)
        .await
        .expect("the slot holder is admitted");
    slot.end_receive();
    assert!(
        slot.reserve_processing(SMALL_BUDGET + 1, &shutdown)
            .await
            .expect("the slot is free")
            .entered_overflow,
        "fixture: the slot is held"
    );
    let on_slot = gate.overflow_bytes();
    let mut queued = gate
        .admit(UploadRpc::UpdateLayer, "rp0-queued", &shutdown)
        .await
        .expect("the queued step's upload is admitted");
    queued.end_receive();
    let queued_in_budget = queued.reserved();
    let queued_task = tokio::spawn({
        let shutdown = shutdown.clone();
        async move {
            // Over the budget, with the slot held: it waits for the slot.
            let _ = queued.reserve_processing(SMALL_BUDGET + 1, &shutdown).await;
            queued
        }
    });
    until(deadline, "the queued step waits for the slot", || {
        gate.processing_waiters() == 1
    })
    .await;
    let mut filler = gate
        .admit(UploadRpc::UpdateLayer, "rp0-filler", &shutdown)
        .await
        .expect("the filler is admitted");
    let others = gate.reserved_bytes() - filler.reserved();
    filler
        .try_grow_in_budget(SMALL_BUDGET - others - step - growth / 2)
        .expect("the filler leaves the call its step and half of RP0's growth");
    let filled = filler.reserved();

    // Every change to what the gate holds from here, in order.
    let reports: Arc<std::sync::Mutex<Vec<GateCharge>>> = Arc::default();
    gate.set_charge_observer_for_test(Some(Arc::new({
        let reports = Arc::clone(&reports);
        move |charge: GateCharge| reports.lock().unwrap().push(charge)
    })));
    let (request, _feed) = sent(&update_messages(&session_id, "layer", object_id, &ompf, 2));
    let task = tokio::spawn({
        let svc = Arc::new(svc);
        async move { handle_update_layer(&svc, request).await.map(|_| ()) }
    });
    until(
        deadline,
        "the update waits for its recharge behind the queued step, or answered",
        || gate.processing_waiters() == 2 || task.is_finished(),
    )
    .await;
    if task.is_finished() {
        panic!(
            "fixture: the update must wait for its recharge; it answered {:?}",
            task.await
        );
    }
    assert_eq!(
        recharges("structural"),
        1,
        "the hit's first repair read (RP0) was refused: one recharge, on `structural`"
    );
    assert!(
        ls.admission_permits_available_for_test(object_id)
            .is_none_or(|available| available == (1, K)),
        "the recharging update holds no per-object permit"
    );
    // Every report before this one is the call's alone, with the filler and
    // the queued step in budget; the next is the filler's release.
    let filler_released_at = reports.lock().unwrap().len();
    drop(filler);
    settled(deadline, "the update", task)
        .await
        .expect("no panic")
        .expect("the update commits after its recharge");
    gate.set_charge_observer_for_test(None);

    assert_eq!(seq(&tv, object_id).await, before_seq + 1, "one commit");
    assert_eq!(
        (recharges("structural"), recharges("structural_full")),
        (1, 0),
        "exactly one recharge, on the hit's path"
    );
    let reports = reports.lock().unwrap().clone();
    let call = |i: usize, charge: &GateCharge| {
        let others = if i < filler_released_at {
            filled + queued_in_budget
        } else {
            queued_in_budget
        };
        charge.reserved_bytes.saturating_sub(others)
    };
    let stepped = reports
        .iter()
        .enumerate()
        .position(|(i, charge)| call(i, charge) == step)
        .expect("the call's processing step was granted at its processing charge and hint");
    // The recharge's grant: the first growth of the call's reservation from
    // the filler's release on, to the processing charge and the whole target
    // of the refused point — RP0's, not a later point's such as CR1's.
    let recharged = (filler_released_at.max(1)..reports.len())
        .find(|&i| call(i, &reports[i]) > call(i - 1, &reports[i - 1]))
        .expect("the recharge was granted");
    assert_eq!(
        call(recharged, &reports[recharged]),
        processing + rp0,
        "the recharge is granted at the processing charge and RP0's whole target: the refused \
         point was the hit's first repair read; reports: {reports:?}"
    );
    for (i, charge) in reports.iter().enumerate().take(recharged + 1).skip(stepped) {
        assert!(
            call(i, charge) >= step,
            "report {i}: the call holds {} B, below its processing step and hint ({step} B), \
             across its recharge; reports: {reports:?}",
            call(i, charge)
        );
        assert_eq!(
            charge.overflow_bytes, on_slot,
            "report {i}: the slot stays the holder's; the call's step and recharge are in budget"
        );
    }
    queued_task.abort();
    let _ = queued_task.await;
    drop(slot);
}

/// Sections of the in-commit shutdown test's base: enough that a WATCH
/// read's count window charges a whole window of commands in flight
/// (`W_CMD` × `FANOUT_WINDOW`, tens of MB), far past the receive allowance
/// the budget has room for when the commit starts.
#[cfg(feature = "test-fault-injection")]
const RECHARGE_BASE_SECTIONS: u64 = 1_000;

/// The in-commit recharge wait's handler-level arm (SDD §17.2.1): a structural `UpdateLayer` on
/// the WATCH path (`txn`) whose read cannot grow its charge — the budget full
/// but for the call's own admission, the overflow slot held — suspends and
/// waits on the gate for its recharge, holding nothing. Graceful shutdown
/// beginning during that in-commit wait, on the call's first attempt, answers
/// the shared shutdown status — `UNAVAILABLE "server shutting down"` with the
/// `SERVER_SHUTTING_DOWN` detail and its retry hint, as the admission and
/// processing waits answer — logged once at INFO at stage `storage`, with
/// nothing applied.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_recharge_wait_at_shutdown_answers_server_shutting_down_at_stage_storage_b26e()
{
    recharge_wait_ended_by_shutdown(false).await;
}

/// The same in-commit recharge wait ended by shutdown, but on the call's
/// second attempt, after a counted retry (attempt 1 `ABORTED` by the
/// test-only fault injector, before any store command): an earlier attempt
/// may have committed, so the refusal detail's "this call applied nothing"
/// cannot be promised and the answer is the plain outcome-unknown
/// `UNAVAILABLE` (SDD §18.1), with no detail, still logged at stage `storage`
/// and still applying nothing.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_recharge_wait_at_shutdown_after_a_counted_retry_answers_outcome_unknown_b26e()
{
    recharge_wait_ended_by_shutdown(true).await;
}

/// The body of the two in-commit shutdown tests above; `counted_retry`
/// makes attempt 1 fail `ABORTED` first, so the recharge wait runs on
/// attempt 2.
#[cfg(feature = "test-fault-injection")]
async fn recharge_wait_ended_by_shutdown(counted_retry: bool) {
    use crate::common::refusal_detail::assert_no_refusal_detail;
    use live_edit::config::UpdateLayerBackend;
    use live_edit::error::ServiceError;
    use live_edit::grpc::upload_admission::UploadRpc;
    use live_edit::storage::external_values::MockExternalValueStore;
    use live_edit::storage::layer::LayerStore;
    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;

    const RECEIVE_SLACK: u64 = 256 * 1024;
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("recharge-shutdown"));
    let injected = Arc::new(Mutex::new(VecDeque::new()));
    if counted_retry {
        injected
            .lock()
            .expect("injector lock")
            .push_back(ServiceError::Aborted(
                "injected attempt-1 conflict".to_owned(),
            ));
    }
    let ls = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Txn)
            .with_update_layer_retry_max_attempts(3)
            .with_fault_injector_for_test(Arc::clone(&injected))
            .with_metrics(Arc::clone(&metrics)),
    );
    let svc = Arc::new(LiveEditService {
        upload_admission: Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes: SMALL_BUDGET,
                wait: TEST_BUDGET * 2,
                max_waiters: 8,
                max_upload_bytes: 0,
            },
            None,
        )),
        ..build_service_with_layer_store(&tv, Arc::clone(&ls))
    });
    let gate = Arc::clone(&svc.upload_admission);

    let session_id = new_session(&tv, "recharge-shutdown").await;
    let object_id = ls
        .create_layer(
            &session_id,
            "layer",
            "test-user",
            &sections_under_root(RECHARGE_BASE_SECTIONS),
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("the base is created");
    let before_seq = seq(&tv, object_id).await;
    let recharges = || {
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_recharges_total",
            &[("path", "watch")],
        )
        .unwrap_or(0.0) as u64
    };

    // The slot held, and the budget full but for one admission and its
    // receive growth.
    let holders = CancellationToken::new();
    let mut slot = gate
        .admit(UploadRpc::UpdateLayer, "recharge-slot", &holders)
        .await
        .expect("the slot holder is admitted");
    slot.end_receive();
    assert!(
        slot.reserve_processing(SMALL_BUDGET + 1, &holders)
            .await
            .expect("the slot is free")
            .entered_overflow,
        "fixture: the slot is held"
    );
    let mut filler = gate
        .admit(UploadRpc::UpdateLayer, "recharge-filler", &holders)
        .await
        .expect("the filler is admitted");
    let others = gate.reserved_bytes() - filler.reserved();
    filler
        .try_grow_in_budget(SMALL_BUDGET - others - C - RECEIVE_SLACK)
        .expect("the filler leaves room for one admission and its receive growth");

    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 7,
            section_name: "new".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    };
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let (request, _feed) = sent(&update_messages(&session_id, "layer", object_id, &ompf, 2));
    let shutdown = svc.shutdown.clone();
    let begin_once_recharging = async {
        until(deadline, "the call waits for its recharge", || {
            gate.processing_waiters() == 1 && recharges() == 1
        })
        .await;
        begin_paced_ending_held_open(&shutdown).await
    };
    let ((result, captured), paced) = tokio::join!(
        settled(
            deadline,
            "the recharging UpdateLayer",
            capture(handle_update_layer(&svc, request))
        ),
        begin_once_recharging
    );
    let status = result.expect_err("an in-commit recharge wait must end at shutdown");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    if counted_retry {
        assert_eq!(
            status.message(),
            "UpdateLayer outcome unknown: an earlier attempt of this call may have committed \
             before attempt 2 was refused (server shutting down)",
            "{status:?}"
        );
        assert_no_refusal_detail(&status);
        assert!(
            injected.lock().expect("injector lock").is_empty(),
            "attempt 1 consumed the injected conflict"
        );
    } else {
        assert_shutdown_status(&status);
    }
    drop(paced);
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "UpdateLayer", "storage", "UNAVAILABLE");
    assert_eq!(recharges(), 1, "the call waited in one recharge");
    assert_eq!(seq(&tv, object_id).await, before_seq, "nothing applied");
    assert!(
        ls.admission_permits_available_for_test(object_id)
            .is_none_or(|available| available.1
                == live_edit::storage::layer::DEFAULT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY as usize),
        "the call holds no per-object permit"
    );
    drop((slot, filler));
}

/// The checkpoint-recovery re-issue charges through the call's commit
/// charge. A structural `UpdateLayer` on the WATCH path (`txn`) whose object
/// was wiped finds its header gone before its read charges anything
/// (`NOT_FOUND`), the recovery hook restores the object from a checkpoint of
/// [`RECHARGE_BASE_SECTIONS`] sections, and the handler re-issues the commit
/// with the same charge. With the budget full but for the call's own
/// admission and the overflow slot held, the re-issued attempt's read cannot
/// grow that charge, so it suspends and waits on the gate for its recharge,
/// holding no per-object permit — after the restore, so the wait is the
/// re-issue's. Once the slot is free it commits, with the one recharge. A
/// re-issue under no charge would never wait and would read the restored
/// object outside the budget.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_recovery_reissue_recharges_through_the_calls_commit_charge_b26e() {
    use crate::common::{recovery_metric, wipe_object_keys, RECOVERY_SERVICE_LABEL};
    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::{CheckpointMeta, CheckpointStore, RecoveryHook};
    use live_edit::config::UpdateLayerBackend;
    use live_edit::domain::ompf::OmpfContainer;
    use live_edit::grpc::upload_admission::UploadRpc;
    use live_edit::storage::external_values::MockExternalValueStore;
    use live_edit::storage::layer::LayerStore;
    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};
    use tokio_util::sync::CancellationToken;

    const RECEIVE_SLACK: u64 = 256 * 1024;
    /// The checkpoint's `snap_seq`.
    const SNAP_SEQ: u64 = 3;
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    // One registry for the store's commit metrics and the recovery hook's,
    // under the label `recovery_metric` reads.
    let metrics = Arc::new(crate::common::metrics_scrape_only(RECOVERY_SERVICE_LABEL));
    let ls = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Txn)
            .with_metrics(Arc::clone(&metrics)),
    );
    let mut svc = LiveEditService {
        upload_admission: Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes: SMALL_BUDGET,
                wait: TEST_BUDGET * 2,
                max_waiters: 8,
                max_upload_bytes: 0,
            },
            None,
        )),
        ..build_service_with_layer_store(&tv, Arc::clone(&ls))
    };
    svc.metrics = Arc::clone(&metrics);

    let session_id = new_session(&tv, "recovery-recharge").await;
    let object_id = ls
        .create_layer(
            &session_id,
            "layer",
            "test-user",
            &sections_under_root(1),
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("the layer is created");
    // The checkpoint: a reset delta re-creating the sections of
    // `sections_under_root(RECHARGE_BASE_SECTIONS)`, so the restored object is
    // large enough that its WATCH read's count window outgrows the room.
    let restored = OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: (100..100 + RECHARGE_BASE_SECTIONS)
            .map(|id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: id,
                    section_name: format!("s{id}"),
                    section_type: 1,
                })
            })
            .collect(),
        ..Default::default()
    };
    let checkpoint = OmpfContainer::serialize(
        &oodf_ffi::build_delta(&restored).expect("the checkpoint delta builds"),
        &[],
    )
    .expect("the checkpoint OMPF serializes");
    let store = InMemoryCheckpointStore::new();
    store
        .put_layer_checkpoint(
            object_id,
            SNAP_SEQ,
            bytes::Bytes::from(checkpoint),
            CheckpointMeta {
                creator_replica_id: RECOVERY_SERVICE_LABEL.to_owned(),
                written_at_unix_ms: 1_700_000_000_000,
                snap_seq: Some(SNAP_SEQ),
                session_id: session_id.clone(),
                layer_name: "layer".to_owned(),
                created_by: "test-user".to_owned(),
                server_ts: 1_700_000_000_000,
                ..CheckpointMeta::default()
            },
        )
        .await
        .expect("the checkpoint is seeded");
    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = true;
    cfg.service_label = RECOVERY_SERVICE_LABEL.to_owned();
    svc.recovery = Arc::new(RecoveryHook::new(
        Arc::new(store) as Arc<dyn CheckpointStore>,
        Arc::clone(&svc.layer_store),
        Arc::new(cfg),
        Arc::clone(&svc.metrics),
    ));
    let svc = Arc::new(svc);
    let gate = Arc::clone(&svc.upload_admission);
    wipe_object_keys(&tv, object_id).await;
    let recharges = || {
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_recharges_total",
            &[("path", "watch")],
        )
        .unwrap_or(0.0) as u64
    };

    // The slot held, and the budget full but for one admission and its
    // receive growth.
    let holders = CancellationToken::new();
    let mut slot = gate
        .admit(UploadRpc::UpdateLayer, "recovery-slot", &holders)
        .await
        .expect("the slot holder is admitted");
    slot.end_receive();
    assert!(
        slot.reserve_processing(SMALL_BUDGET + 1, &holders)
            .await
            .expect("the slot is free")
            .entered_overflow,
        "fixture: the slot is held"
    );
    let mut filler = gate
        .admit(UploadRpc::UpdateLayer, "recovery-filler", &holders)
        .await
        .expect("the filler is admitted");
    let others = gate.reserved_bytes() - filler.reserved();
    filler
        .try_grow_in_budget(SMALL_BUDGET - others - C - RECEIVE_SLACK)
        .expect("the filler leaves room for one admission and its receive growth");

    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 7,
            section_name: "new".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    };
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let (request, _feed) = sent(&update_messages(&session_id, "layer", object_id, &ompf, 2));
    let task = tokio::spawn({
        let svc = Arc::clone(&svc);
        async move { handle_update_layer(&svc, request).await.map(|_| ()) }
    });
    until(
        deadline,
        "the re-issued commit waits for its recharge, or the call answered",
        || (gate.processing_waiters() == 1 && recharges() == 1) || task.is_finished(),
    )
    .await;
    if task.is_finished() {
        panic!(
            "the re-issued commit must wait on the gate for its recharge; the call answered \
             {:?} with {} recharges",
            task.await,
            recharges()
        );
    }
    assert_eq!(
        recovery_metric(&metrics, "restored"),
        1,
        "the wait is the re-issue's: the object was restored before it"
    );
    assert!(
        ls.admission_permits_available_for_test(object_id)
            .is_none_or(|available| available.1
                == live_edit::storage::layer::DEFAULT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY as usize),
        "the waiting re-issue holds no per-object permit"
    );
    let restored_seq = seq(&tv, object_id).await;

    drop(slot);
    settled(deadline, "the re-issued UpdateLayer", task)
        .await
        .expect("no panic")
        .expect("the re-issued commit lands in the freed slot");
    assert_eq!(
        seq(&tv, object_id).await,
        restored_seq + 1,
        "the re-issue commits once on the restored object"
    );
    assert_eq!(recharges(), 1, "the re-issue recharged once");
    drop(filler);
}

/// b.5wb: the checkpoint-recovery re-issue's own pre-store holds one more
/// copy of the upload's external values on the call's commit charge while it
/// runs. An `UpdateLayer` carrying one external value of [`BLOB_LEN`] bytes
/// stores it through an [`ObservedStore`], whose first store deletes the
/// object's header, so the first call ends `NOT_FOUND`; the recovery hook
/// restores the object from a checkpoint at `seq` 1, and the handler stores
/// the value again before re-issuing the commit. The gate's charge observer,
/// the store calls and the fast route's commit dispatches are recorded in one
/// ordered log ([`Event`]).
#[cfg(feature = "test-fault-injection")]
mod reissue_extra_copy {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use futures::future::BoxFuture;
    use futures::FutureExt;

    use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
    use live_edit::checkpoint::{CheckpointStore, RecoveryHook};
    use live_edit::error::ServiceError;
    use live_edit::grpc::upload_admission::{UploadReservation, UploadRpc};
    use live_edit::observability::metrics::test_util::histogram_bucket;
    use live_edit::observability::metrics::Metrics;
    use live_edit::storage::external_values::{
        ext_content_hash, ExtRelease, ExtValueProbe, ExtValueVerify, ExternalValueStore,
        GcBatchResult, GcPassState, GcPolicy, ReleaseAttemptOutcome, ValkeyExternalValueStore,
    };
    use live_edit::storage::layer::LayerStore;
    use tokio_util::sync::CancellationToken;

    use super::*;

    /// The external value's size: the extra copy the re-pre-store holds.
    const BLOB_LEN: usize = 4_096;

    /// One entry of a run's ordered log.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Event {
        /// The gate's in-budget reservation total after a change (the
        /// test-only charge observer).
        Charge(u64),
        /// The `n`-th store call reached the external-value store.
        Store(usize),
        /// A fast-route commit was about to be dispatched.
        Dispatch,
    }

    type Log = Arc<Mutex<Vec<Event>>>;

    /// A hook run at one store call, before it stores.
    type AtStore = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>;

    /// A Valkey external-value store that logs each store call and, before
    /// the `n`-th, runs the `n`-th queued hook.
    struct ObservedStore {
        inner: ValkeyExternalValueStore,
        log: Log,
        stores: AtomicUsize,
        at_store: Mutex<VecDeque<AtStore>>,
    }

    impl ObservedStore {
        async fn observe(&self) {
            let n = self.stores.fetch_add(1, Ordering::SeqCst) + 1;
            self.log.lock().unwrap().push(Event::Store(n));
            let hook = self.at_store.lock().unwrap().pop_front();
            if let Some(hook) = hook {
                hook().await;
            }
        }
    }

    #[async_trait::async_trait]
    impl ExternalValueStore for ObservedStore {
        async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
            self.observe().await;
            self.inner.store_values(values).await
        }

        async fn store_verified_values(
            &self,
            values: &[(&[u8; 32], &[u8])],
        ) -> Result<(), ServiceError> {
            self.observe().await;
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

    /// `live_edit_upload_processing_wait_seconds{rpc="update_layer"}`
    /// observations so far.
    fn processing_waits(metrics: &Metrics) -> u64 {
        histogram_bucket(
            &render(metrics),
            "live_edit_upload_processing_wait_seconds",
            "+Inf",
            &[("rpc", "update_layer")],
        )
        .unwrap_or(0.0) as u64
    }

    /// The service, its gate and log, a layer at `seq` 1 with a checkpoint
    /// at `seq` 1, and the content of an update carrying one external value.
    struct Fixture {
        tv: TestValkey,
        svc: Arc<LiveEditService>,
        gate: Arc<UploadAdmissionGate>,
        metrics: Arc<Metrics>,
        ext: Arc<ObservedStore>,
        log: Log,
        session_id: String,
        object_id: u64,
        ompf: Vec<u8>,
    }

    impl Fixture {
        /// The update, its content in two chunks.
        fn request(&self) -> Request<Streaming<UpdateLayerRequest>> {
            let (request, _feed) = sent(&update_messages(
                &self.session_id,
                "layer",
                self.object_id,
                &self.ompf,
                2,
            ));
            request
        }

        /// Queue the store hooks: the first store deletes the object's
        /// header, so the first call ends `NOT_FOUND`, then runs
        /// `then_at_first`; the second runs `at_second`.
        fn at_stores(&self, then_at_first: AtStore, at_second: AtStore) {
            let (client, header_key) = (
                self.tv.client().clone(),
                self.tv.key_schema().obj_header(&self.object_id.to_string()),
            );
            let first: AtStore = Box::new(move || {
                async move {
                    let _: i64 = client
                        .pool
                        .del(header_key)
                        .await
                        .expect("DEL obj:hdr at the first store");
                    then_at_first().await;
                }
                .boxed()
            });
            self.ext.at_store.lock().unwrap().extend([first, at_second]);
        }

        /// The ordered log so far.
        fn log(&self) -> Vec<Event> {
            self.log.lock().unwrap().clone()
        }
    }

    /// A hook that does nothing.
    fn nothing() -> AtStore {
        Box::new(|| async {}.boxed())
    }

    /// The [`Fixture`] on a gate of `budget_bytes`.
    async fn fixture(label: &str, budget_bytes: u64) -> Fixture {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(
            crate::common::RECOVERY_SERVICE_LABEL,
        ));
        let log: Log = Arc::default();
        let dispatches: live_edit::storage::watched_txn::WatchWindowHook = {
            let log = Arc::clone(&log);
            Arc::new(move |_: &fred::clients::Client| {
                log.lock().unwrap().push(Event::Dispatch);
                async {}.boxed()
            })
        };
        let ls = Arc::new(
            LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                .with_lua_commit_pre_dispatch_hook_for_test(dispatches)
                .with_metrics(Arc::clone(&metrics)),
        );
        let gate = Arc::new(UploadAdmissionGate::new(
            UploadAdmissionConfig {
                budget_bytes,
                wait: TEST_BUDGET * 2,
                max_waiters: 8,
                max_upload_bytes: 0,
            },
            Some(Arc::clone(&metrics)),
        ));
        {
            let log = Arc::clone(&log);
            gate.set_charge_observer_for_test(Some(Arc::new(move |charge| {
                log.lock()
                    .unwrap()
                    .push(Event::Charge(charge.reserved_bytes));
            })));
        }
        let ext = Arc::new(ObservedStore {
            inner: ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone()),
            log: Arc::clone(&log),
            stores: AtomicUsize::new(0),
            at_store: Mutex::new(VecDeque::new()),
        });
        let (session_id, object_id) = setup_layer_named(&tv, label, "layer").await;
        let checkpoints = InMemoryCheckpointStore::new();
        crate::common::seed_layer_checkpoint(&checkpoints, object_id, &session_id, "layer", 1)
            .await;
        let mut cfg = crate::common::test_config();
        cfg.checkpoint_enabled = true;
        cfg.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
        let svc = LiveEditService {
            upload_admission: Arc::clone(&gate),
            ext_value_store: Arc::clone(&ext) as Arc<dyn ExternalValueStore>,
            metrics: Arc::clone(&metrics),
            ..build_service_with_layer_store(&tv, Arc::clone(&ls))
        };
        let svc = LiveEditService {
            recovery: Arc::new(RecoveryHook::new(
                Arc::new(checkpoints) as Arc<dyn CheckpointStore>,
                Arc::clone(&svc.layer_store),
                Arc::new(cfg),
                Arc::clone(&metrics),
            )),
            ..svc
        };
        let blob: Vec<u8> = (0..BLOB_LEN).map(|i| (i % 251) as u8).collect();
        let delta = crate::common::owned_delta::delta_with_set_field_external(
            1,
            "blob",
            ext_content_hash(&blob).to_vec(),
            BLOB_LEN as u64,
            1,
            0,
        );
        Fixture {
            tv,
            svc: Arc::new(svc),
            gate,
            metrics,
            ext,
            log,
            session_id,
            object_id,
            ompf: delta_to_ompf_bytes(&delta, &[&blob]),
        }
    }

    /// The re-issue's pre-store grows the call's reservation by exactly the
    /// stored bytes ([`BLOB_LEN`], one copy) right before the second store,
    /// and shrinks it back to what it held before that growth right after
    /// it, before the re-issue's first commit dispatch: the extra copy is
    /// held for the store only, never under the commit. The growth is a
    /// waiting step, so it records one more
    /// `live_edit_upload_processing_wait_seconds` observation between the
    /// two stores. The re-issue then commits.
    ///
    /// Fail-on-old: the RPC stored once, so there was no second store and no
    /// growth.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_recovery_reissue_holds_one_copy_of_its_external_values_for_its_pre_store_b5wb(
    ) {
        let deadline = test_deadline();
        let fx = fixture("reissue-extra-copy", 64 * 1024 * 1024).await;
        let waits_at: Arc<Mutex<Vec<u64>>> = Arc::default();
        let at = |waits_at: &Arc<Mutex<Vec<u64>>>| -> AtStore {
            let (waits_at, metrics) = (Arc::clone(waits_at), Arc::clone(&fx.metrics));
            Box::new(move || {
                async move {
                    waits_at.lock().unwrap().push(processing_waits(&metrics));
                }
                .boxed()
            })
        };
        fx.at_stores(at(&waits_at), at(&waits_at));
        let before_seq = seq(&fx.tv, fx.object_id).await;

        let response = settled(
            deadline,
            "the recovered UpdateLayer",
            handle_update_layer(&fx.svc, fx.request()),
        )
        .await
        .expect("the re-issue commits");

        assert_eq!(
            crate::common::recovery_metric(&fx.metrics, "restored"),
            1,
            "precondition: the first call ended NOT_FOUND and the restore landed"
        );
        assert!(
            response.into_inner().sequence_num > before_seq,
            "the re-issue committed on the restored object"
        );
        let log = fx.log();
        let second = log
            .iter()
            .position(|e| *e == Event::Store(2))
            .unwrap_or_else(|| panic!("the re-issue's pre-store ran; log {log:?}"));
        let charge = |i: usize| match log.get(i) {
            Some(Event::Charge(reserved)) => *reserved,
            other => panic!("event {i} is a gate change; got {other:?} in {log:?}"),
        };
        // What the reservation held before the growth: the last gate change
        // before it (the first call's commit dispatch, which changes nothing,
        // may sit between).
        let prior = log[..second - 1]
            .iter()
            .rev()
            .find_map(|e| match e {
                Event::Charge(reserved) => Some(*reserved),
                Event::Store(_) | Event::Dispatch => None,
            })
            .unwrap_or_else(|| panic!("the call held a charge before; log {log:?}"));
        let (grown, given_back) = (charge(second - 1), charge(second + 1));
        assert_eq!(
            grown,
            prior + BLOB_LEN as u64,
            "right before the second store the reservation grew by one copy of the stored \
             bytes; log {log:?}"
        );
        assert_eq!(
            given_back, prior,
            "right after it the reservation is back to what it held; log {log:?}"
        );
        let dispatch = log[second..]
            .iter()
            .position(|e| *e == Event::Dispatch)
            .map(|i| i + second)
            .unwrap_or_else(|| panic!("the re-issue dispatched its commit; log {log:?}"));
        assert!(
            second + 1 < dispatch,
            "the extra copy was given back before the re-issue's first commit dispatch; log \
             {log:?}"
        );
        let waits_at = waits_at.lock().unwrap().clone();
        assert_eq!(
            waits_at.len(),
            2,
            "precondition: both stores ran their hooks"
        );
        assert_eq!(
            waits_at[1],
            waits_at[0] + 1,
            "the extra-copy hold is one waiting step: one processing-wait observation between \
             the two stores"
        );
        assert_eq!(fx.ext.stores.load(Ordering::SeqCst), 2);
    }

    /// With the budget full and the overflow slot held when the re-issue's
    /// pre-store asks for its extra copy, the call waits for it as one
    /// processing waiter, holding nothing else; graceful shutdown beginning
    /// there answers by the re-issue's rule — the first call made one
    /// attempt, so the shutdown refusal verbatim (`UNAVAILABLE "server
    /// shutting down"`, its detail and retry hint) — logged once at INFO,
    /// and the re-issue never runs: the second store never happens and the
    /// restored object carries no commit of the call.
    ///
    /// Fail-on-old: no hold, so no wait; the re-issue committed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_recovery_reissue_waiting_to_hold_its_extra_copy_at_shutdown_answers_server_shutting_down_b5wb(
    ) {
        let deadline = test_deadline();
        let fx = fixture("reissue-extra-copy-shutdown", SMALL_BUDGET).await;
        let gate = Arc::clone(&fx.gate);

        // The slot held by another upload.
        let holders = CancellationToken::new();
        let mut slot = gate
            .admit(UploadRpc::UpdateLayer, "extra-copy-slot", &holders)
            .await
            .expect("the slot holder is admitted");
        slot.end_receive();
        assert!(
            slot.reserve_processing(SMALL_BUDGET + 1, &holders)
                .await
                .expect("the slot is free")
                .entered_overflow,
            "fixture: the slot is held"
        );
        // A filler that takes the rest of the budget at the first store, once
        // the call holds its processing charge.
        let filler: Arc<Mutex<Option<UploadReservation>>> = Arc::new(Mutex::new(Some(
            gate.admit(UploadRpc::UpdateLayer, "extra-copy-filler", &holders)
                .await
                .expect("the filler is admitted"),
        )));
        let fill: AtStore = {
            let (gate, filler) = (Arc::clone(&gate), Arc::clone(&filler));
            Box::new(move || {
                async move {
                    let mut filler = filler.lock().unwrap();
                    let filler = filler.as_mut().expect("the filler is held");
                    let others = gate.reserved_bytes() - filler.reserved();
                    filler
                        .try_grow_in_budget(gate.budget_bytes() - others)
                        .expect("the filler takes the rest of the budget");
                }
                .boxed()
            })
        };
        fx.at_stores(fill, nothing());

        let shutdown = fx.svc.shutdown.clone();
        let answered = std::sync::atomic::AtomicBool::new(false);
        let begin_once_waiting = async {
            until(
                deadline,
                "the re-issue's pre-store waits for its extra copy, or the call answered",
                || gate.processing_waiters() == 1 || answered.load(Ordering::SeqCst),
            )
            .await;
            begin_paced_ending_held_open(&shutdown).await
        };
        let call = async {
            let answer = capture(handle_update_layer(&fx.svc, fx.request())).await;
            answered.store(true, Ordering::SeqCst);
            answer
        };
        let ((result, captured), paced) = tokio::join!(
            settled(deadline, "the recovered UpdateLayer", call),
            begin_once_waiting
        );

        let status = result.expect_err("the extra-copy wait must end at shutdown");
        assert_shutdown_status(&status);
        drop(paced);
        assert_eq!(
            crate::common::recovery_metric(&fx.metrics, "restored"),
            1,
            "precondition: the wait came after the restore"
        );
        let event = captured.only(RPC_FAILURE_MSG);
        assert_eq!(
            event["level"], "INFO",
            "a shutdown ending; full event: {event}"
        );
        assert_eq!(
            fx.ext.stores.load(Ordering::SeqCst),
            1,
            "the re-issue's pre-store never ran"
        );
        let record: Option<String> = fx
            .tv
            .client()
            .pool
            .hget(
                fx.tv
                    .key_schema()
                    .obj_write_token(&fx.object_id.to_string()),
                "test-client",
            )
            .await
            .expect("HGET obj:wtok");
        assert_eq!(
            record, None,
            "the restored object carries no commit of the call: the re-issue never ran"
        );
        assert_eq!(gate.processing_waiters(), 0);
        drop((slot, filler));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// T5 — the reservation is released on every exit
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_parse_error_after_admission_releases_its_reservation() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = setup_layer_named(&tv, "t5-parse", "layer").await;

    let (request, _feed) = sent(&[
        make_update_header(&session_id, "layer", object_id),
        make_update_data(vec![0xEE; 256]),
    ]);
    let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect_err("content that is not OMPF must be refused");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert!(status.message().contains("malformed OMPF"), "{status:?}");
    assert_admitted_and_released(&svc.upload_admission, &session_id, "parse error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_storage_error_releases_its_reservation() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = setup_layer_named(&tv, "t5-storage", "layer").await;
    // Retype the object's header: the commit fails WRONGTYPE.
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().obj_header(&object_id.to_string()),
            "not-a-hash",
            None,
            None,
            false,
        )
        .await
        .expect("retyping obj:hdr must succeed");

    let (request, _feed) = sent(&update_messages(
        &session_id,
        "layer",
        object_id,
        &field_ompf(1),
        1,
    ));
    let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect_err("a wrong-typed header must fail the call");
    // A WRONGTYPE commit is a permanent keyspace refusal: a server fault,
    // answered `INTERNAL` with the detail withheld from the client.
    assert_eq!(status.code(), Code::Internal, "{status:?}");
    assert_eq!(status.message(), "internal error", "{status:?}");
    assert_admitted_and_released(&svc.upload_admission, &session_id, "storage error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_transport_error_mid_content_is_cancelled_and_releases_its_reservation() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = setup_layer_named(&tv, "t5-transport", "layer").await;
    let chunks = split(&field_ompf(1), 2);

    let (request, feed) = upload::<UpdateLayerRequest>();
    feed.send(&make_update_header(&session_id, "layer", object_id));
    feed.send(&make_update_data(chunks[0].clone()));
    feed.fail();
    let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
        .await
        .expect_err("a transport failure mid-content must fail the call");
    assert_eq!(status.code(), Code::Cancelled, "{status:?}");
    assert_admitted_and_released(&svc.upload_admission, &session_id, "transport error");
    assert_eq!(seq(&tv, object_id).await, 1);
}

/// The handler future dropped mid-content — what hyper does to a call whose
/// client reset it while the handler waits — releases the reservation and the
/// session slot through `Drop`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upload_handler_dropped_mid_content_releases_its_reservation() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(build_service(&tv, Arc::new(AllowAllPermissions)));
    let (session_id, object_id) = setup_layer_named(&tv, "t5-dropped", "layer").await;
    let gate = Arc::clone(&svc.upload_admission);
    let chunks = split(&field_ompf(1), 2);

    let (request, feed) = upload::<UpdateLayerRequest>();
    feed.send(&make_update_header(&session_id, "layer", object_id));
    feed.send(&make_update_data(chunks[0].clone()));
    let task = tokio::spawn({
        let svc = Arc::clone(&svc);
        async move { handle_update_layer(&svc, request).await.map(|_| ()) }
    });
    until(
        deadline,
        "the upload admitted and reading its content",
        || gate.receiving_uploads(&session_id) == 1 && feed.pulled() == 2,
    )
    .await;

    task.abort();
    assert!(settled(deadline, "the aborted handler", task)
        .await
        .expect_err("aborted")
        .is_cancelled());
    assert_admitted_and_released(&gate, &session_id, "dropped handler");
    assert_eq!(seq(&tv, object_id).await, 1);
}

/// An admission queued when graceful shutdown's paced stream ending starts
/// ends at once, at t0 — its wait races `begun`, not the service shutdown
/// token, which is cancelled only when the paced ending finishes — with
/// `UNAVAILABLE "server shutting down"` and its retry hint (nothing was
/// applied), and one INFO completion line at stage `admission` — the expected
/// ending on a draining pod, not a WARN. An upload waiting for the service
/// token instead would wait out the whole paced ending.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_queued_at_shutdown_answers_server_shutting_down_with_one_info_line() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(service_with_gate(
        &tv,
        Arc::new(AllowAllPermissions),
        UploadAdmissionConfig {
            budget_bytes: C,
            // Longer than the whole test: only shutdown can end the wait.
            wait: TEST_BUDGET * 2,
            max_waiters: 4,
            max_upload_bytes: 0,
        },
    ));
    let gate = Arc::clone(&svc.upload_admission);
    let (holder_session, holder_object) =
        setup_layer_named(&tv, "t5-shutdown-holder", "layer").await;
    let (session_id, object_id) = setup_layer_named(&tv, "t5-shutdown-queued", "layer").await;

    // The holder takes the whole budget and stalls before its content.
    let (holder_request, holder_feed) = upload::<UpdateLayerRequest>();
    holder_feed.send(&make_update_header(&holder_session, "layer", holder_object));
    let holder = tokio::spawn({
        let svc = Arc::clone(&svc);
        async move { handle_update_layer(&svc, holder_request).await.map(|_| ()) }
    });
    until(deadline, "the holder admitted", || {
        gate.receiving_uploads(&holder_session) == 1
    })
    .await;

    let (request, _feed) = sent(&update_messages(
        &session_id,
        "layer",
        object_id,
        &field_ompf(1),
        1,
    ));
    let shutdown = svc.shutdown.clone();
    let begin_once_queued = async {
        until(deadline, "the call queued for admission", || {
            gate.waiters() == 1
        })
        .await;
        begin_paced_ending_held_open(&shutdown).await
    };
    let ((result, captured), paced) = tokio::join!(
        settled(
            deadline,
            "the queued UpdateLayer",
            capture(handle_update_layer(&svc, request))
        ),
        begin_once_queued
    );
    let status = result.expect_err("a queued admission must end at shutdown");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_shutdown_status(&status);
    assert!(
        !svc.shutdown.is_cancelled(),
        "the queued upload was refused at t0, while the paced ending was still under way"
    );
    drop(paced);
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "UpdateLayer", "admission", "UNAVAILABLE");
    assert_eq!(gate.waiters(), 0);
    assert_eq!(seq(&tv, object_id).await, 1);

    holder.abort();
    let _ = settled(deadline, "the aborted holder", holder).await;
    assert_eq!(gate.reserved_bytes(), 0);
    drop(holder_feed);
}

// ── The server deadline ──────────────────────────────────────────────────────

/// A real server, wired as `main.rs` wires it — the production HTTP/2
/// settings (`transport::production_server_builder`: the 30 s / 5 s
/// keepalive), `MetricsLayer` with its RPC deadline, and the LiveService built
/// by `transport::live_service_server` (the auth interceptor and the
/// end-of-stream record) — around `svc`.
struct DeadlineServer {
    addr: std::net::SocketAddr,
    /// The metrics layer's registry, where `live_endpoint_calls_total` ticks.
    layer_metrics: Arc<live_edit::observability::metrics::Metrics>,
    task: tokio::task::JoinHandle<()>,
}

/// The metrics layer's `service` label on a [`DeadlineServer`].
const SERVER_SERVICE_LABEL: &str = "test";

impl DeadlineServer {
    /// Start the server and wait, until the test's `deadline`, for it to
    /// accept connections.
    async fn start(
        svc: LiveEditService,
        rpc_deadline_ms: u64,
        deadline: tokio::time::Instant,
    ) -> Self {
        use live_edit::grpc::metrics_layer::MetricsLayer;
        use live_edit::grpc::transport::{live_service_server, production_server_builder};

        let layer_metrics = Arc::new(crate::common::metrics_scrape_only(SERVER_SERVICE_LABEL));
        let metrics_layer = MetricsLayer::new(
            Arc::clone(&layer_metrics),
            SERVER_SERVICE_LABEL,
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
                ("service", SERVER_SERVICE_LABEL),
                ("grpc_code", grpc_code),
            ],
        )
        .unwrap_or(0.0) as u64
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

impl Drop for DeadlineServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn with_bearer<T>(inner: T) -> Request<T> {
    let mut request = Request::new(inner);
    request.metadata_mut().insert(
        "authorization",
        tonic::metadata::MetadataValue::try_from("Bearer upload-admission-test")
            .expect("valid metadata"),
    );
    request
}

/// The RPC deadline the deadline tests configure: short, so an admitted upload
/// whose content never arrives meets it quickly.
const SHORT_DEADLINE_MS: u64 = 300;

/// An admitted `UpdateLayer` whose content never arrives meets the server's
/// RPC deadline (`LIVE_EDIT_RPC_DEADLINE_MS`, enforced by `MetricsLayer`):
/// the client gets `DEADLINE_EXCEEDED` and the dropped handler's reservation
/// is released. The direct middleware-deadline coverage for `UpdateLayer`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_admitted_and_stalled_meets_the_server_deadline_and_releases_its_reservation()
{
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = setup_layer_named(&tv, "t5-deadline-update", "layer").await;
    let gate = Arc::clone(&svc.upload_admission);
    let server = DeadlineServer::start(svc, SHORT_DEADLINE_MS, deadline).await;

    let (tx, rx) = mpsc::channel::<UpdateLayerRequest>(4);
    tx.send(make_update_header(&session_id, "layer", object_id))
        .await
        .expect("queue the header");
    let status = settled(
        deadline,
        "the deadline-bound UpdateLayer",
        server
            .client()
            .await
            .update_layer(with_bearer(tokio_stream::wrappers::ReceiverStream::new(rx))),
    )
    .await
    .expect_err("an upload whose content never arrives must not succeed");
    assert_eq!(status.code(), Code::DeadlineExceeded, "{status:?}");
    until(deadline, "the reservation released", || {
        gate.reserved_bytes() == 0
    })
    .await;
    assert!(
        gate.high_water_bytes() >= C,
        "precondition: it was admitted"
    );
    assert_eq!(gate.receiving_uploads(&session_id), 0);
    drop(tx);
}

/// The `CreateLayer` twin of the deadline test above.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_admitted_and_stalled_meets_the_server_deadline_and_releases_its_reservation()
{
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "t5-deadline-create").await;
    let gate = Arc::clone(&svc.upload_admission);
    let server = DeadlineServer::start(svc, SHORT_DEADLINE_MS, deadline).await;

    let (tx, rx) = mpsc::channel::<CreateLayerRequest>(4);
    tx.send(make_create_header(&session_id, "layer"))
        .await
        .expect("queue the header");
    let status = settled(
        deadline,
        "the deadline-bound CreateLayer",
        server
            .client()
            .await
            .create_layer(with_bearer(tokio_stream::wrappers::ReceiverStream::new(rx))),
    )
    .await
    .expect_err("an upload whose content never arrives must not succeed");
    assert_eq!(status.code(), Code::DeadlineExceeded, "{status:?}");
    until(deadline, "the reservation released", || {
        gate.reserved_bytes() == 0
    })
    .await;
    assert!(
        gate.high_water_bytes() >= C,
        "precondition: it was admitted"
    );
    assert_eq!(gate.receiving_uploads(&session_id), 0);
    drop(tx);
}

// ═════════════════════════════════════════════════════════════════════════════
// T6 — fairness across sessions
// ═════════════════════════════════════════════════════════════════════════════

/// The per-session receiving cap (one here): a session's second upload waits
/// behind its first while another session's upload is admitted and commits,
/// and the waiting one is refused `UNAVAILABLE` at the wait budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_session_over_its_receiving_cap_waits_while_another_sessions_upload_is_admitted() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(service_with_gate(
        &tv,
        Arc::new(AllowAllPermissions),
        small_gate(SHORT_WAIT, 8),
    ));
    let gate = Arc::clone(&svc.upload_admission);
    assert_eq!(gate.per_session_cap(), 1, "fixture self-check");
    let (busy, busy_object) = setup_layer_named(&tv, "t6-busy", "layer").await;
    let (other, other_object) = setup_layer_named(&tv, "t6-other", "layer").await;

    // The busy session's first upload is admitted and stalls before content.
    let (first_request, mut first_feed) = upload::<UpdateLayerRequest>();
    first_feed.send(&make_update_header(&busy, "layer", busy_object));
    let first = tokio::spawn({
        let svc = Arc::clone(&svc);
        async move { handle_update_layer(&svc, first_request).await.map(|_| ()) }
    });
    until(deadline, "the busy session's first upload admitted", || {
        gate.receiving_uploads(&busy) == 1
    })
    .await;

    // Its second waits on the session cap, though the budget has room.
    let (second_request, _second_feed) = sent(&update_messages(
        &busy,
        "layer",
        busy_object,
        &field_ompf(2),
        1,
    ));
    let second = tokio::spawn({
        let svc = Arc::clone(&svc);
        async move { handle_update_layer(&svc, second_request).await.map(|_| ()) }
    });
    until(deadline, "the busy session's second upload queued", || {
        gate.waiters() == 1
    })
    .await;

    // Another session is admitted at once and commits.
    let (other_request, _other_feed) = sent(&update_messages(
        &other,
        "layer",
        other_object,
        &field_ompf(3),
        1,
    ));
    settled(
        deadline,
        "the other session's upload",
        handle_update_layer(&svc, other_request),
    )
    .await
    .expect("another session's upload is admitted and commits");
    assert_eq!(seq(&tv, other_object).await, 2);
    assert_eq!(
        gate.waiters(),
        1,
        "the busy session's second is still waiting"
    );

    let refused = settled(deadline, "the busy session's second upload", second)
        .await
        .expect("no panic")
        .expect_err("the busy session's excess is refused at the wait budget");
    assert_eq!(refused.code(), Code::Unavailable, "{refused:?}");

    first_feed.send(&make_update_data(field_ompf(1)));
    first_feed.end();
    settled(deadline, "the busy session's first upload", first)
        .await
        .expect("no panic")
        .expect("the first upload commits");
    assert_eq!(
        seq(&tv, busy_object).await,
        2,
        "only the busy session's first committed"
    );
    assert_eq!(gate.reserved_bytes(), 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// T9 — a session deleted between authorization and commit
// ═════════════════════════════════════════════════════════════════════════════

/// A `CreateLayer` authorized before a `DeleteSession` is refused at its
/// commit point, `NOT_FOUND "session was deleted"`, and writes no layer
/// mapping, no layers-index entry and no operation record. The session is
/// deleted while the upload is admitted and waiting for its content, which
/// arrives afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_whose_session_was_deleted_after_authorization_is_not_found_and_maps_nothing()
{
    const LAYER: &str = "t9-layer";
    const OPERATION_ID: &str = "t9-operation";
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = Arc::new(build_service(&tv, Arc::new(AllowAllPermissions)));
    let gate = Arc::clone(&svc.upload_admission);
    let session_id = new_session(&tv, "t9-deleted").await;

    let (request, mut feed) = upload::<CreateLayerRequest>();
    feed.send(&create_header_with(&session_id, LAYER, OPERATION_ID));
    let task = tokio::spawn({
        let svc = Arc::clone(&svc);
        async move {
            handle_create_layer(&svc, request)
                .await
                .map(|r| r.into_inner())
        }
    });
    until(deadline, "the create authorized and admitted", || {
        gate.receiving_uploads(&session_id) == 1
    })
    .await;

    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .delete(&session_id)
        .await
        .expect("DeleteSession succeeds");
    feed.send(&make_create_data(delta_to_ompf_bytes(&empty_delta(), &[])));
    feed.end();

    let status = settled(deadline, "the create", task)
        .await
        .expect("no panic")
        .expect_err("a create into a deleted session must be refused");
    assert_eq!(status.code(), Code::NotFound, "{status:?}");
    assert_eq!(status.message(), "session was deleted");

    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let mapping: Option<String> = pool
        .get(keys.layer_mapping(&session_id, LAYER))
        .await
        .expect("GET the mapping");
    assert_eq!(
        mapping, None,
        "no layer mapping may be written into a deleted session"
    );
    let indexed: bool = pool
        .sismember(
            keys.layers_index(&session_id),
            live_edit::storage::keys::hex_encode(LAYER),
        )
        .await
        .expect("SISMEMBER the layers index");
    assert!(
        !indexed,
        "the layer must not be listed in the deleted session's index"
    );
    let record: Option<String> = pool
        .get(keys.create_layer_operation(&session_id, OPERATION_ID))
        .await
        .expect("GET the operation record");
    assert_eq!(record, None, "no operation record may be written");
    assert_eq!(gate.reserved_bytes(), 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// T12 — the handler's admission reaches the service's metrics
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upload_admission_metrics_follow_a_handled_upload() {
    let deadline = test_deadline();
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "t12").await;
    let metric = |name: &str, labels: &[(&str, &str)]| {
        sample_value(&render(&svc.metrics), name, labels).unwrap_or(f64::NAN)
    };

    let (request, _feed) = sent(&create_messages(&session_id, "layer", &field_ompf(1), 2));
    settled(deadline, "the handler", handle_create_layer(&svc, request))
        .await
        .expect("the upload commits");

    assert_eq!(
        metric(
            "live_edit_upload_admission_total",
            &[("rpc", "create_layer"), ("result", "admitted")]
        ),
        1.0
    );
    assert_eq!(
        metric(
            "live_edit_upload_admission_total",
            &[("rpc", "update_layer"), ("result", "admitted")]
        ),
        0.0
    );
    assert_eq!(metric("live_edit_upload_reserved_bytes", &[]), 0.0);
    assert_eq!(
        metric("live_edit_uploads_in_flight", &[("phase", "receiving")]),
        0.0
    );
    assert_eq!(
        metric("live_edit_uploads_in_flight", &[("phase", "processing")]),
        0.0
    );
    assert_eq!(
        metric("live_edit_upload_budget_bytes", &[]),
        svc.upload_admission.budget_bytes() as f64
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// An upload stream that did not end with END_STREAM — regression tests of b.682
// ═════════════════════════════════════════════════════════════════════════════

mod cancelled_upload {
    //! Regression tests of b.682: an upload is applied only if its request
    //! stream ended with HTTP/2 END_STREAM (SDD §5.4). Every other ending —
    //! a client's `RST_STREAM(CANCEL)`, the `RST_STREAM(NO_ERROR)` Envoy v1.36
    //! forwards for a client's cancel, the server's own keepalive
    //! `GOAWAY(NO_ERROR)` — answers `CANCELLED` with nothing committed,
    //! created, mapped or stored, and the upload's reservation released.
    //!
    //! Before the fix, tonic turned a `CANCEL` into a clean end of stream
    //! (tonic 0.14.6 `codec/decode.rs:260-263`) and hyper did the same for
    //! the `NO_ERROR` endings (hyper 1.11.0 `body/incoming.rs:250-255`), so
    //! the handler committed whatever prefix had arrived: an empty update
    //! (`seq` 1 → 2), an empty layer created and mapped, or — cut inside the
    //! last external value, since an OMPF container records only where each
    //! part starts — the truncated value stored under the full value's hash.
    //! The handler now reads the end-of-stream record
    //! (`grpc::request_end`) at each `Ok(None)`.
    //!
    //! The handler-level tests drive the endings through a [`ScriptedBody`]
    //! ([`Step::Cancel`], [`Step::EndWithoutEndStream`]; a dropped feed is
    //! END_STREAM). The end-to-end tests drive them through a raw HTTP/2
    //! client against a [`DeadlineServer`], built through
    //! `transport::live_service_server` as `main.rs` builds it. By the time
    //! the server answers a truncated upload its client can no longer read
    //! the answer — the stream is reset or the connection shut down — so
    //! those tests assert the server-side outcome: the one INFO `RPC failed`
    //! line at stage `body`, one `CANCELLED` tick on
    //! `live_endpoint_calls_total`, no abandoned-call line, the sequence
    //! unchanged, nothing stored, the reservation released. They run on a
    //! current-thread runtime so a [`ThreadCapture`] sees the line the
    //! server's connection task emits.
    //!
    //! Each refusal test asserts the code, not only that nothing was applied:
    //! the upload content check (`storage::external_values::ext_content_hash`)
    //! alone would refuse the truncated-value inputs `INVALID_ARGUMENT` at
    //! stage `parse`, so `CANCELLED` at stage `body` is what keeps these tests
    //! about the stream's ending.
    //!
    //! The `update_layer_*_meets_the_keepalive_timeout_*` test waits out the
    //! production keepalive — a PING after [`HTTP2_KEEPALIVE_INTERVAL`]
    //! without a frame from the client, then [`HTTP2_KEEPALIVE_TIMEOUT`] for
    //! its acknowledgement — so its healthy path takes their sum, which the
    //! `const` checks beside [`E2E_RPC_DEADLINE_MS`] hold inside the server's
    //! RPC deadline and [`TEST_BUDGET`].

    use live_edit::grpc::transport::{HTTP2_KEEPALIVE_INTERVAL, HTTP2_KEEPALIVE_TIMEOUT};
    use live_edit::storage::external_values::ValkeyExternalValueStore;

    use super::*;
    use crate::grpc::log_capture::{
        assert_field, purge_keys, ThreadCapture, GRPC_CODE_SLUGS, INTERNAL_ERROR_MSG,
        RPC_ABANDONED_MSG,
    };

    // ── Fixtures ─────────────────────────────────────────────────────────────

    /// The external value's length.
    const BLOB_LEN: usize = 4_096;

    /// How far before the end of the OMPF the truncated upload stops: inside
    /// the external value, the container's last part.
    const CUT_INSIDE_BLOB: usize = 1_000;

    /// An `UpdateLayer` whose delta sets one field held out of line, in a
    /// [`BLOB_LEN`]-byte external value that is the OMPF's last part, claimed
    /// under its content address (`ext_content_hash`, the hash the upload
    /// content check verifies).
    struct BlobUpdate {
        ompf: Vec<u8>,
        blob: Vec<u8>,
        hash_hex: String,
    }

    impl BlobUpdate {
        fn new() -> Self {
            let blob: Vec<u8> = (0..BLOB_LEN).map(|i| (i % 251) as u8).collect();
            let hash = live_edit::storage::external_values::ext_content_hash(&blob).to_vec();
            let hash_hex = hash.iter().map(|b| format!("{b:02x}")).collect();
            let delta = crate::common::owned_delta::delta_with_set_field_external(
                1,
                "blob",
                hash,
                BLOB_LEN as u64,
                1,
                0,
            );
            let ompf = delta_to_ompf_bytes(&delta, &[&blob]);
            Self {
                ompf,
                blob,
                hash_hex,
            }
        }

        /// The OMPF cut inside its last part: it still parses, with the
        /// external value shortened — what a commit of a truncated upload
        /// would store under the full value's hash.
        fn truncated(&self) -> Vec<u8> {
            let cut = self.ompf.len() - CUT_INSIDE_BLOB;
            assert!(
                cut > self.ompf.len() - BLOB_LEN,
                "fixture self-check: the cut falls inside the external value, the last part"
            );
            self.ompf[..cut].to_vec()
        }
    }

    /// [`build_service`] with the real external-value store, so what an
    /// upload stores under a hash can be read back.
    fn ext_service(tv: &TestValkey) -> LiveEditService {
        LiveEditService {
            ext_value_store: Arc::new(ValkeyExternalValueStore::new(
                tv.client().clone(),
                tv.key_schema().clone(),
            )),
            ..build_service(tv, Arc::new(AllowAllPermissions))
        }
    }

    /// The bytes stored under `hash_hex`, if any.
    async fn stored_value(tv: &TestValkey, hash_hex: &str) -> Option<Vec<u8>> {
        tv.client()
            .pool
            .get(tv.key_schema().ext_data(hash_hex))
            .await
            .expect("GET the external value")
    }

    /// The layer mapping of `layer` in `session_id`, if any.
    async fn mapping(tv: &TestValkey, session_id: &str, layer: &str) -> Option<String> {
        tv.client()
            .pool
            .get(tv.key_schema().layer_mapping(session_id, layer))
            .await
            .expect("GET the mapping")
    }

    /// How a handler-level upload's body ends after its last frame.
    #[derive(Clone, Copy, Debug)]
    enum Ending {
        /// A client's `RST_STREAM(CANCEL)` ([`Step::Cancel`]).
        Cancel,
        /// The end without END_STREAM ([`Step::EndWithoutEndStream`]).
        WithoutEndStream,
        /// END_STREAM: the client finished its stream.
        EndStream,
    }

    fn finish(feed: &mut Feed, ending: Ending) {
        match ending {
            Ending::Cancel => feed.cancel(),
            Ending::WithoutEndStream => feed.end_without_end_stream(),
            Ending::EndStream => feed.end(),
        }
    }

    /// What every refused truncated upload answers: `CANCELLED`, under the
    /// documented `stream receive error:` prefix, its reservation and session
    /// slot released.
    fn assert_cancelled_and_released(
        result: &Result<impl std::fmt::Debug, Status>,
        gate: &UploadAdmissionGate,
        session_id: &str,
        what: &str,
    ) {
        let status = result
            .as_ref()
            .expect_err(&format!("{what}: a truncated upload must not answer OK"));
        assert_eq!(status.code(), Code::Cancelled, "{what}: {status:?}");
        assert!(
            status.message().starts_with("stream receive error:"),
            "{what}: {status:?}"
        );
        assert_admitted_and_released(gate, session_id, what);
    }

    // ── UpdateLayer: header only ─────────────────────────────────────────────

    /// An `UpdateLayer` of a header alone, ending as `ending`: the handler's
    /// answer (its sequence number) and the layer's sequence after it.
    async fn update_header_only(
        tv: &TestValkey,
        svc: &LiveEditService,
        label: &str,
        ending: Ending,
    ) -> (String, Result<u64, Status>, u64) {
        let deadline = test_deadline();
        let (session_id, object_id) = setup_layer_named(tv, label, "layer").await;
        let (request, mut feed) = upload::<UpdateLayerRequest>();
        feed.send(&make_update_header(&session_id, "layer", object_id));
        finish(&mut feed, ending);
        let result = settled(deadline, "the handler", handle_update_layer(svc, request))
            .await
            .map(|r| r.into_inner().sequence_num);
        (session_id, result, seq(tv, object_id).await)
    }

    /// T1: an `UpdateLayer` cancelled right after its header commits nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_cancelled_after_its_header_commits_nothing() {
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, result, seq_after) =
            update_header_only(&tv, &svc, "b682-update-header", Ending::Cancel).await;
        assert_cancelled_and_released(&result, &svc.upload_admission, &session_id, "cancel");
        assert_eq!(seq_after, 1, "a cancelled UpdateLayer must commit nothing");
    }

    /// T3: an `UpdateLayer` whose stream ends without END_STREAM right after
    /// its header — the shape of Envoy's `RST_STREAM(NO_ERROR)` — commits
    /// nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_ended_without_end_stream_after_its_header_commits_nothing() {
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, result, seq_after) = update_header_only(
            &tv,
            &svc,
            "b682-update-header-no-eos",
            Ending::WithoutEndStream,
        )
        .await;
        assert_cancelled_and_released(&result, &svc.upload_admission, &session_id, "no END_STREAM");
        assert_eq!(seq_after, 1, "a truncated UpdateLayer must commit nothing");
    }

    /// T4 (d), the counterpart of the two above: a header followed by
    /// END_STREAM is a valid empty update (SDD §8.4), committed as one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_header_only_with_end_stream_commits_an_empty_update() {
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, result, seq_after) =
            update_header_only(&tv, &svc, "b682-update-header-eos", Ending::EndStream).await;
        assert_eq!(
            result.expect("a header-only upload that ended with END_STREAM commits"),
            2
        );
        assert_eq!(seq_after, 2, "the empty update is committed");
        assert_admitted_and_released(&svc.upload_admission, &session_id, "END_STREAM");
    }

    // ── CreateLayer: header only ─────────────────────────────────────────────

    /// A `CreateLayer` of a header alone, ending as `ending`: the handler's
    /// answer (its object id) and the layer's mapping after it.
    async fn create_header_only(
        tv: &TestValkey,
        svc: &LiveEditService,
        label: &str,
        ending: Ending,
    ) -> (String, Result<u64, Status>, Option<String>) {
        let deadline = test_deadline();
        let session_id = new_session(tv, label).await;
        let (request, mut feed) = upload::<CreateLayerRequest>();
        feed.send(&make_create_header(&session_id, "layer"));
        finish(&mut feed, ending);
        let result = settled(deadline, "the handler", handle_create_layer(svc, request))
            .await
            .map(|r| r.into_inner().object_id);
        let mapped = mapping(tv, &session_id, "layer").await;
        (session_id, result, mapped)
    }

    /// T1: a `CreateLayer` cancelled right after its header creates nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_cancelled_after_its_header_creates_nothing() {
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, result, mapped) =
            create_header_only(&tv, &svc, "b682-create-header", Ending::Cancel).await;
        assert_cancelled_and_released(&result, &svc.upload_admission, &session_id, "cancel");
        assert_eq!(mapped, None, "a cancelled CreateLayer must create no layer");
    }

    /// T3: a `CreateLayer` whose stream ends without END_STREAM right after
    /// its header creates nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_ended_without_end_stream_after_its_header_creates_nothing() {
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, result, mapped) = create_header_only(
            &tv,
            &svc,
            "b682-create-header-no-eos",
            Ending::WithoutEndStream,
        )
        .await;
        assert_cancelled_and_released(&result, &svc.upload_admission, &session_id, "no END_STREAM");
        assert_eq!(mapped, None, "a truncated CreateLayer must create no layer");
    }

    /// T4 (d), the counterpart of the two above: a header followed by
    /// END_STREAM creates an empty layer and maps it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_header_only_with_end_stream_creates_an_empty_layer() {
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, result, mapped) =
            create_header_only(&tv, &svc, "b682-create-header-eos", Ending::EndStream).await;
        let object_id =
            result.expect("a header-only upload that ended with END_STREAM creates a layer");
        assert_ne!(object_id, 0);
        assert_eq!(
            mapped,
            Some(object_id.to_string()),
            "the empty layer is created and mapped"
        );
        assert_admitted_and_released(&svc.upload_admission, &session_id, "END_STREAM");
    }

    // ── UpdateLayer: mid-content ─────────────────────────────────────────────

    /// T2: an `UpdateLayer` cancelled after part of its content — whole
    /// content frames, then half of the next — commits nothing and answers
    /// `CANCELLED`. Before the fix it committed nothing either, but only
    /// because the OODF verifier refused the cut delta `INVALID_ARGUMENT`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_cancelled_mid_content_commits_nothing() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) = setup_layer_named(&tv, "b682-update-partial", "layer").await;
        let chunks = split(&field_ompf(9), 3);

        let (request, feed) = upload::<UpdateLayerRequest>();
        feed.send(&make_update_header(&session_id, "layer", object_id));
        feed.send(&make_update_data(chunks[0].clone()));
        let partial = grpc_frame(&make_update_data(chunks[1].clone()));
        feed.send_raw(partial.slice(..partial.len() / 2));
        feed.cancel();
        let result = settled(deadline, "the handler", handle_update_layer(&svc, request))
            .await
            .map(|r| r.into_inner().sequence_num);
        assert_cancelled_and_released(&result, &svc.upload_admission, &session_id, "cancel");
        assert_eq!(
            seq(&tv, object_id).await,
            1,
            "a cancelled UpdateLayer must commit nothing"
        );
    }

    /// An `UpdateLayer` of a header and the [`BlobUpdate`] cut inside its
    /// external value — whole gRPC frames, so the decoder holds no partial
    /// frame — ending as `ending`. Returns the session, the handler's answer,
    /// the sequence after it and what is stored under the value's hash.
    async fn update_inside_the_last_external_value(
        tv: &TestValkey,
        svc: &LiveEditService,
        label: &str,
        ending: Ending,
    ) -> (String, Result<u64, Status>, u64, Option<Vec<u8>>) {
        let deadline = test_deadline();
        let (session_id, object_id) = setup_layer_named(tv, label, "layer").await;
        let fixture = BlobUpdate::new();

        let (request, mut feed) = upload::<UpdateLayerRequest>();
        feed.send(&make_update_header(&session_id, "layer", object_id));
        feed.send(&make_update_data(fixture.truncated()));
        finish(&mut feed, ending);
        let result = settled(deadline, "the handler", handle_update_layer(svc, request))
            .await
            .map(|r| r.into_inner().sequence_num);
        (
            session_id,
            result,
            seq(tv, object_id).await,
            stored_value(tv, &fixture.hash_hex).await,
        )
    }

    /// T1: an `UpdateLayer` cancelled part-way through its LAST external
    /// value commits nothing and stores nothing under the value's hash.
    /// Before the fix it committed, with 3,096 of the 4,096 bytes stored
    /// under the full value's hash.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_cancelled_inside_its_last_external_value_commits_nothing() {
        let tv = TestValkey::setup().await;
        let svc = ext_service(&tv);
        let (session_id, result, seq_after, stored) =
            update_inside_the_last_external_value(&tv, &svc, "b682-update-blob", Ending::Cancel)
                .await;
        assert_cancelled_and_released(&result, &svc.upload_admission, &session_id, "cancel");
        assert_eq!(seq_after, 1, "a cancelled UpdateLayer must commit nothing");
        assert_eq!(
            stored.map(|v| v.len()),
            None,
            "nothing may be stored under the external value's hash"
        );
    }

    /// T3: the same upload ending without END_STREAM — Envoy's
    /// `RST_STREAM(NO_ERROR)` shape, mid-content — commits and stores
    /// nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_ended_without_end_stream_inside_its_last_external_value_commits_nothing()
    {
        let tv = TestValkey::setup().await;
        let svc = ext_service(&tv);
        let (session_id, result, seq_after, stored) = update_inside_the_last_external_value(
            &tv,
            &svc,
            "b682-update-blob-no-eos",
            Ending::WithoutEndStream,
        )
        .await;
        assert_cancelled_and_released(&result, &svc.upload_admission, &session_id, "no END_STREAM");
        assert_eq!(seq_after, 1, "a truncated UpdateLayer must commit nothing");
        assert_eq!(
            stored.map(|v| v.len()),
            None,
            "nothing may be stored under the external value's hash"
        );
    }

    // ── The missing record ───────────────────────────────────────────────────

    /// T8: an upload request carrying no end-of-stream record — a LiveService
    /// stack built without `live_service_server` — is refused `INTERNAL`
    /// having read nothing and reserved nothing, with one ERROR `internal
    /// error` line at stage `header`. Fail closed: without the record a
    /// reset upload would read as a complete one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upload_without_an_end_of_stream_record_is_internal_having_read_nothing() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) = setup_layer_named(&tv, "b682-no-record", "layer").await;

        let (update, mut update_feed) = upload_without_a_record::<UpdateLayerRequest>();
        update_feed.send(&make_update_header(&session_id, "layer", object_id));
        update_feed.end();
        let (create, mut create_feed) = upload_without_a_record::<CreateLayerRequest>();
        create_feed.send(&make_create_header(&session_id, "other-layer"));
        create_feed.end();

        // Both handler futures are boxed, as in `overflow_slot`'s
        // over-cap refusal: inline, each is held again by `capture` and
        // `settled`, which in a debug build brought this test within 256 KiB
        // of std's default 2 MiB test-thread stack.
        let (update_result, update_log) = settled(
            deadline,
            "the UpdateLayer",
            capture(Box::pin(handle_update_layer(&svc, update))),
        )
        .await;
        let (create_result, create_log) = settled(
            deadline,
            "the CreateLayer",
            capture(Box::pin(handle_create_layer(&svc, create))),
        )
        .await;

        for (what, status, feed, log) in [
            (
                "UpdateLayer",
                update_result.err(),
                &update_feed,
                &update_log,
            ),
            (
                "CreateLayer",
                create_result.err(),
                &create_feed,
                &create_log,
            ),
        ] {
            let status = status.unwrap_or_else(|| panic!("{what}: must be refused"));
            assert_eq!(status.code(), Code::Internal, "{what}: {status:?}");
            assert_eq!(
                feed.pulled(),
                0,
                "{what}: nothing may be read without the record"
            );
            let event = log.only(INTERNAL_ERROR_MSG);
            assert_common_fields(event, "ERROR", what, "header", "INTERNAL");
            assert!(
                event["fields"]["error"]
                    .as_str()
                    .is_some_and(|e| e.contains("no end-of-stream record")),
                "{what}: the ERROR line names the missing record; {event}"
            );
        }
        assert_eq!(
            svc.upload_admission.high_water_bytes(),
            0,
            "nothing reserved"
        );
        assert_eq!(seq(&tv, object_id).await, 1);
        assert_eq!(mapping(&tv, &session_id, "other-layer").await, None);
    }

    // ── The completion line and the metrics ──────────────────────────────────

    /// T9: a truncated body logs exactly one INFO `RPC failed` line at stage
    /// `body`, `grpc_code` `CANCELLED`, under the `stream receive error:`
    /// prefix — for a reset (`CANCEL`) and for an ending without END_STREAM
    /// alike, each with its own message.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_truncated_body_logs_one_info_cancelled_line_at_stage_body() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        for (ending, message_prefix) in [
            (
                Ending::Cancel,
                "stream receive error: the upload stream was reset before it ended (",
            ),
            (
                Ending::WithoutEndStream,
                "stream receive error: the upload stream ended without END_STREAM",
            ),
        ] {
            let (session_id, object_id) =
                setup_layer_named(&tv, &format!("b682-log-body-{ending:?}"), "layer").await;
            let (request, mut feed) = upload::<UpdateLayerRequest>();
            feed.send(&make_update_header(&session_id, "layer", object_id));
            feed.send(&make_update_data(field_ompf(1)));
            finish(&mut feed, ending);
            let (result, captured) = settled(
                deadline,
                "the handler",
                capture(handle_update_layer(&svc, request)),
            )
            .await;
            let status = result.expect_err("a truncated upload must be refused");
            assert_eq!(status.code(), Code::Cancelled, "{ending:?}: {status:?}");
            let event = captured.only(RPC_FAILURE_MSG);
            assert_common_fields(event, "INFO", "UpdateLayer", "body", "CANCELLED");
            assert!(
                event["fields"]["status_message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with(message_prefix)),
                "{ending:?}: {event}"
            );
            assert_eq!(seq(&tv, object_id).await, 1);
        }
    }

    /// T9: a stream that ends before its first message other than with
    /// END_STREAM — reset, or ended without it — is `CANCELLED`, logged once
    /// at INFO at stage `header`, having reserved nothing. A stream that
    /// carried nothing and ended with END_STREAM stays the malformed request
    /// it was: `INVALID_ARGUMENT`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_stream_ended_before_its_first_message_is_cancelled_at_stage_header() {
        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        for (ending, code) in [
            (Ending::Cancel, Code::Cancelled),
            (Ending::WithoutEndStream, Code::Cancelled),
            (Ending::EndStream, Code::InvalidArgument),
        ] {
            let (request, mut feed) = upload::<UpdateLayerRequest>();
            finish(&mut feed, ending);
            let (result, captured) = settled(
                deadline,
                "the handler",
                capture(handle_update_layer(&svc, request)),
            )
            .await;
            let status = result.expect_err("a stream with no header must be refused");
            assert_eq!(status.code(), code, "{ending:?}: {status:?}");
            let event = captured.only(RPC_FAILURE_MSG);
            assert_eq!(event["level"], "INFO", "{ending:?}: {event}");
            assert_field(event, "stage", "header".into());
        }
        assert_eq!(
            svc.upload_admission.high_water_bytes(),
            0,
            "nothing reserved"
        );
    }

    /// T10: a cancel while the content arrives adds nothing to
    /// `live_edit_upload_admission_total` beyond the upload's `admitted`, and
    /// the in-flight gauges and the reserved bytes return to zero.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_cancelled_mid_content_counts_only_its_admission() {
        use live_edit::observability::metrics::{
            ADMISSION_RESULT_ADMITTED, UPLOAD_ADMISSION_RESULTS,
        };

        let deadline = test_deadline();
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) = setup_layer_named(&tv, "b682-metrics", "layer").await;
        let metric = |name: &str, labels: &[(&str, &str)]| {
            sample_value(&render(&svc.metrics), name, labels).unwrap_or(0.0)
        };

        let (request, feed) = upload::<UpdateLayerRequest>();
        feed.send(&make_update_header(&session_id, "layer", object_id));
        feed.send(&make_update_data(field_ompf(1)));
        feed.cancel();
        let status = settled(deadline, "the handler", handle_update_layer(&svc, request))
            .await
            .expect_err("a cancelled upload must be refused");
        assert_eq!(status.code(), Code::Cancelled, "{status:?}");

        for result in UPLOAD_ADMISSION_RESULTS {
            let want = if *result == ADMISSION_RESULT_ADMITTED {
                1.0
            } else {
                0.0
            };
            assert_eq!(
                metric(
                    "live_edit_upload_admission_total",
                    &[("rpc", "update_layer"), ("result", result)]
                ),
                want,
                "live_edit_upload_admission_total{{result=\"{result}\"}}"
            );
        }
        for phase in ["receiving", "processing"] {
            assert_eq!(
                metric("live_edit_uploads_in_flight", &[("phase", phase)]),
                0.0,
                "live_edit_uploads_in_flight{{phase=\"{phase}\"}}"
            );
        }
        assert_eq!(metric("live_edit_upload_reserved_bytes", &[]), 0.0);
    }

    // ── End to end: a raw HTTP/2 client against a real server ────────────────

    type RawConnection = Pin<Box<h2::client::Connection<tokio::net::TcpStream, Bytes>>>;

    /// One `UpdateLayer` upload from a raw HTTP/2 client — the frames, resets
    /// and silences a real client or proxy produces — to a [`DeadlineServer`].
    /// The connection is driven by a task the test can stop without closing
    /// the connection ([`Self::stop_answering`]).
    struct RawUpload {
        _send_request: h2::client::SendRequest<Bytes>,
        response: Option<h2::client::ResponseFuture>,
        body: h2::SendStream<Bytes>,
        stop: Option<tokio::sync::oneshot::Sender<()>>,
        driver: tokio::task::JoinHandle<Option<RawConnection>>,
        /// The connection once the driver has stopped: held, never polled,
        /// so the TCP connection stays open and the server's PINGs go
        /// unanswered.
        held: Option<RawConnection>,
    }

    impl RawUpload {
        /// Open a connection and send the head of an `UpdateLayer` call,
        /// carrying the test bearer.
        async fn open(server: &DeadlineServer, deadline: tokio::time::Instant) -> Self {
            let tcp = tokio::net::TcpStream::connect(server.addr)
                .await
                .expect("TCP connect");
            let (send_request, connection) =
                settled(deadline, "the h2 handshake", h2::client::handshake(tcp))
                    .await
                    .expect("h2 handshake");
            let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
            let driver = tokio::spawn(async move {
                let mut connection: RawConnection = Box::pin(connection);
                tokio::select! {
                    _ = &mut connection => None,
                    _ = stopped => Some(connection),
                }
            });
            let mut send_request = settled(deadline, "the connection ready", send_request.ready())
                .await
                .expect("the connection accepts a stream");
            let head = http::Request::builder()
                .method("POST")
                .uri(format!(
                    "http://{}/nvidia.omniverse.live.v1alpha.LiveService/UpdateLayer",
                    server.addr
                ))
                .header("content-type", "application/grpc")
                .header("te", "trailers")
                .header("authorization", "Bearer upload-admission-test")
                .body(())
                .expect("valid head");
            let (response, body) = send_request
                .send_request(head, false)
                .expect("UpdateLayer head");
            Self {
                _send_request: send_request,
                response: Some(response),
                body,
                stop: Some(stop),
                driver,
                held: None,
            }
        }

        /// Send `msg` as one DATA frame, with END_STREAM if `end_of_stream`.
        fn send(&mut self, msg: &UpdateLayerRequest, end_of_stream: bool) {
            self.body
                .send_data(grpc_frame(msg), end_of_stream)
                .expect("send a DATA frame");
        }

        /// Reset the stream with `reason`.
        fn reset(&mut self, reason: h2::Reason) {
            self.body.send_reset(reason);
        }

        /// Stop driving the connection — so nothing more is read from it and
        /// the server's keepalive PING goes unacknowledged — and keep it
        /// open.
        async fn stop_answering(&mut self, deadline: tokio::time::Instant) {
            self.stop
                .take()
                .expect("stopped once")
                .send(())
                .expect("the connection is still being driven");
            let driver = &mut self.driver;
            self.held = settled(deadline, "the connection driver stopped", driver)
                .await
                .expect("the driver must not panic");
            assert!(
                self.held.is_some(),
                "the connection ended before the test stopped answering it"
            );
        }

        /// The response, for a call the server answered.
        async fn response(
            &mut self,
            deadline: tokio::time::Instant,
        ) -> http::Response<h2::RecvStream> {
            let response = self.response.take().expect("taken once");
            settled(deadline, "the response head", response)
                .await
                .expect("the server answers")
        }
    }

    impl Drop for RawUpload {
        fn drop(&mut self) {
            self.driver.abort();
        }
    }

    /// The RPC deadline of the end-to-end server: past the keepalive test's
    /// wait ([`HTTP2_KEEPALIVE_INTERVAL`] + [`HTTP2_KEEPALIVE_TIMEOUT`]) by at
    /// least [`KEEPALIVE_DEADLINE_MARGIN`], and inside [`TEST_BUDGET`].
    const E2E_RPC_DEADLINE_MS: u64 = 50_000;

    /// How far the keepalive's ending must come before the RPC deadline: room
    /// for the handler to answer, and for scheduling under a loaded runner,
    /// so the deadline never ends the keepalive test's call first.
    const KEEPALIVE_DEADLINE_MARGIN: Duration = Duration::from_secs(10);

    // The keepalive ends the silent upload's stream at least the margin before
    // the RPC deadline would, and the deadline — so every end-to-end call —
    // fits inside one test's budget.
    const _: () = {
        let keepalive_ms =
            HTTP2_KEEPALIVE_INTERVAL.as_millis() + HTTP2_KEEPALIVE_TIMEOUT.as_millis();
        assert!(
            keepalive_ms + KEEPALIVE_DEADLINE_MARGIN.as_millis() <= E2E_RPC_DEADLINE_MS as u128
        );
        assert!((E2E_RPC_DEADLINE_MS as u128) < TEST_BUDGET.as_millis());
    };

    /// A current-thread end-to-end fixture: a [`DeadlineServer`] with the
    /// real external-value store over a layer at `seq = 1`, under a
    /// [`ThreadCapture`].
    struct EndToEnd {
        tv: TestValkey,
        capture: ThreadCapture,
        server: DeadlineServer,
        gate: Arc<UploadAdmissionGate>,
        session_id: String,
        object_id: u64,
    }

    impl EndToEnd {
        async fn start(label: &str, deadline: tokio::time::Instant) -> Self {
            let capture = ThreadCapture::start();
            let tv = TestValkey::setup().await;
            let svc = ext_service(&tv);
            let gate = Arc::clone(&svc.upload_admission);
            let (session_id, object_id) = setup_layer_named(&tv, label, "layer").await;
            // An RPC deadline past every wait here. At the 30 s default
            // (`LIVE_EDIT_RPC_DEADLINE_MS`) the deadline would end the
            // keepalive test's call, whose client falls silent, before the
            // keepalive interval plus its timeout did; a deployment with a
            // longer deadline meets the keepalive ending instead, which is
            // the one under test.
            let server = DeadlineServer::start(svc, E2E_RPC_DEADLINE_MS, deadline).await;
            Self {
                tv,
                capture,
                server,
                gate,
                session_id,
                object_id,
            }
        }

        /// Wait until the upload has been admitted and is reading content.
        async fn admitted(&self, deadline: tokio::time::Instant) {
            until(
                deadline,
                "the upload admitted and reading its content",
                || self.gate.receiving_uploads(&self.session_id) == 1,
            )
            .await;
        }

        /// Wait until the upload has received a content chunk: its
        /// reservation grew past the admission charge.
        async fn received_content(&self, deadline: tokio::time::Instant) {
            until(deadline, "the content chunk received", || {
                self.gate.reserved_bytes() > C
            })
            .await;
        }

        /// Wait for the server to record the call, and return its total
        /// `live_endpoint_calls_total` ticks.
        async fn completed(&self, deadline: tokio::time::Instant) -> u64 {
            let total = || -> u64 {
                GRPC_CODE_SLUGS
                    .iter()
                    .map(|code| self.server.calls("UpdateLayer", code))
                    .sum()
            };
            until(deadline, "the server recorded the call", || total() > 0).await;
            total()
        }

        /// The server-side outcome of a truncated upload: one `CANCELLED`
        /// completion, logged once at INFO at stage `body` under
        /// `message_prefix` — not dropped as an abandoned call — the
        /// reservation released, the sequence unchanged and nothing stored
        /// under `hash_hex`.
        async fn assert_cancelled_with_nothing_applied(
            &self,
            deadline: tokio::time::Instant,
            message_prefix: &str,
            hash_hex: Option<&str>,
        ) {
            let total = self.completed(deadline).await;
            let captured = self.capture.snapshot();
            assert_eq!(
                (self.server.calls("UpdateLayer", "CANCELLED"), total),
                (1, 1),
                "one call, completed CANCELLED; full capture:\n{}",
                captured.text
            );
            let event = captured.only(RPC_FAILURE_MSG);
            assert_common_fields(event, "INFO", "UpdateLayer", "body", "CANCELLED");
            assert!(
                event["fields"]["status_message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with(message_prefix)),
                "the completion names the ending; {event}"
            );
            assert!(
                captured.with_message(RPC_ABANDONED_MSG).is_empty(),
                "the handler answered CANCELLED itself rather than being dropped; full \
                 capture:\n{}",
                captured.text
            );
            until(deadline, "the reservation released", || {
                self.gate.reserved_bytes() == 0
            })
            .await;
            assert_admitted_and_released(&self.gate, &self.session_id, "end to end");
            assert_eq!(
                seq(&self.tv, self.object_id).await,
                1,
                "a truncated UpdateLayer must commit nothing"
            );
            if let Some(hash_hex) = hash_hex {
                assert_eq!(
                    stored_value(&self.tv, hash_hex).await.map(|v| v.len()),
                    None,
                    "nothing may be stored under the external value's hash"
                );
            }
        }

        /// `TestValkey` cannot clean up on a current-thread runtime.
        async fn finish(self) {
            purge_keys(&self.tv).await;
        }
    }

    /// The `CANCELLED` detail of a stream reset by its client.
    const RESET_BEFORE_END: &str =
        "stream receive error: the upload stream was reset before it ended (";

    /// The `CANCELLED` detail of a stream that ended without END_STREAM.
    const ENDED_WITHOUT_END_STREAM: &str =
        "stream receive error: the upload stream ended without END_STREAM";

    /// T1/T5: a raw HTTP/2 client sends an `UpdateLayer`'s header, waits until
    /// the server has admitted the upload, then resets the stream with
    /// `CANCEL` — what a gRPC client dropping the call sends. The server
    /// answers `CANCELLED` and commits nothing. Before the fix: `seq` 1 → 2.
    #[tokio::test]
    async fn update_layer_reset_by_its_client_after_its_header_commits_nothing() {
        let deadline = test_deadline();
        let e2e = EndToEnd::start("b682-e2e-cancel", deadline).await;
        let mut client = RawUpload::open(&e2e.server, deadline).await;
        client.send(
            &make_update_header(&e2e.session_id, "layer", e2e.object_id),
            false,
        );
        e2e.admitted(deadline).await;

        client.reset(h2::Reason::CANCEL);
        e2e.assert_cancelled_with_nothing_applied(deadline, RESET_BEFORE_END, None)
            .await;
        drop(client);
        e2e.finish().await;
    }

    /// The decider's Envoy shape: a raw HTTP/2 client sends a header and the
    /// [`BlobUpdate`] cut inside its external value, then resets the stream
    /// with `RST_STREAM(NO_ERROR)` — what Envoy v1.36 sends upstream for a
    /// client's cancel, and what hyper turns into a clean end of the body.
    /// The server answers `CANCELLED`, commits nothing and stores nothing.
    #[tokio::test]
    async fn update_layer_reset_no_error_inside_its_last_external_value_commits_nothing() {
        let deadline = test_deadline();
        let e2e = EndToEnd::start("b682-e2e-no-error", deadline).await;
        let fixture = BlobUpdate::new();
        let mut client = RawUpload::open(&e2e.server, deadline).await;
        client.send(
            &make_update_header(&e2e.session_id, "layer", e2e.object_id),
            false,
        );
        e2e.admitted(deadline).await;
        client.send(&make_update_data(fixture.truncated()), false);
        e2e.received_content(deadline).await;

        client.reset(h2::Reason::NO_ERROR);
        e2e.assert_cancelled_with_nothing_applied(
            deadline,
            ENDED_WITHOUT_END_STREAM,
            Some(&fixture.hash_hex),
        )
        .await;
        drop(client);
        e2e.finish().await;
    }

    /// The decider's keepalive shape: a raw HTTP/2 client sends a header and
    /// the [`BlobUpdate`] cut inside its external value, then stops reading
    /// its connection without closing it. The server's production keepalive
    /// (`transport::production_server_builder`) PINGs after
    /// [`HTTP2_KEEPALIVE_INTERVAL`] without a frame and, unacknowledged for
    /// [`HTTP2_KEEPALIVE_TIMEOUT`], shuts the connection down with
    /// `GOAWAY(NO_ERROR)`, which hyper turns into a clean end of the body. The
    /// server answers `CANCELLED`, commits nothing and stores nothing.
    ///
    /// Takes the keepalive interval plus its timeout on the healthy path,
    /// waited out in real time (a paused clock would also fire the Valkey
    /// client's timeouts).
    #[tokio::test]
    async fn update_layer_whose_connection_meets_the_keepalive_timeout_inside_its_last_external_value_commits_nothing(
    ) {
        let deadline = test_deadline();
        let e2e = EndToEnd::start("b682-e2e-keepalive", deadline).await;
        let fixture = BlobUpdate::new();
        let mut client = RawUpload::open(&e2e.server, deadline).await;
        client.send(
            &make_update_header(&e2e.session_id, "layer", e2e.object_id),
            false,
        );
        e2e.admitted(deadline).await;
        client.send(&make_update_data(fixture.truncated()), false);
        e2e.received_content(deadline).await;

        client.stop_answering(deadline).await;
        e2e.assert_cancelled_with_nothing_applied(
            deadline,
            ENDED_WITHOUT_END_STREAM,
            Some(&fixture.hash_hex),
        )
        .await;
        drop(client);
        e2e.finish().await;
    }

    /// T5: the header sent with END_STREAM on its own DATA frame is a
    /// complete, empty update, end to end: OK, `seq` 1 → 2. The record reads
    /// `Complete` through the real wrapper chain.
    #[tokio::test]
    async fn update_layer_header_with_end_stream_commits_an_empty_update_end_to_end() {
        let deadline = test_deadline();
        let e2e = EndToEnd::start("b682-e2e-header-eos", deadline).await;
        let mut client = RawUpload::open(&e2e.server, deadline).await;
        client.send(
            &make_update_header(&e2e.session_id, "layer", e2e.object_id),
            true,
        );

        assert_eq!(e2e.completed(deadline).await, 1);
        assert_eq!(e2e.server.calls("UpdateLayer", "OK"), 1);
        assert_eq!(
            client.response(deadline).await.status(),
            http::StatusCode::OK
        );
        assert_eq!(
            seq(&e2e.tv, e2e.object_id).await,
            2,
            "the empty update is committed"
        );
        drop(client);
        e2e.finish().await;
    }

    /// The decider's positive case: a client that sends its header and whole
    /// content and half-closes its stream (END_STREAM on the last DATA frame)
    /// still commits, end to end: OK, `seq` 1 → 2, the external value stored
    /// whole under its hash.
    #[tokio::test]
    async fn update_layer_half_closed_after_its_content_commits_end_to_end() {
        let deadline = test_deadline();
        let e2e = EndToEnd::start("b682-e2e-half-close", deadline).await;
        let fixture = BlobUpdate::new();
        let mut client = RawUpload::open(&e2e.server, deadline).await;
        client.send(
            &make_update_header(&e2e.session_id, "layer", e2e.object_id),
            false,
        );
        client.send(&make_update_data(fixture.ompf.clone()), true);

        assert_eq!(e2e.completed(deadline).await, 1);
        assert_eq!(
            e2e.server.calls("UpdateLayer", "OK"),
            1,
            "full capture:\n{}",
            e2e.capture.snapshot().text
        );
        assert_eq!(
            client.response(deadline).await.status(),
            http::StatusCode::OK
        );
        assert_eq!(
            seq(&e2e.tv, e2e.object_id).await,
            2,
            "the update is committed"
        );
        assert_eq!(
            stored_value(&e2e.tv, &fixture.hash_hex).await.as_deref(),
            Some(fixture.blob.as_slice()),
            "the external value is stored whole under its hash"
        );
        drop(client);
        e2e.finish().await;
    }
}
