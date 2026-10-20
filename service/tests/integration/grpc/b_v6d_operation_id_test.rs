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

//! b.v6d — the client `operation_id` through the real `UpdateLayer` /
//! `CreateLayer` handlers: the wire answer table, header validation, and the
//! `CreateLayer` handler's re-send obligations (the first-base re-spawn and the
//! re-driven post-commit tail).
//!
//! The storage-level verdicts (both backends, external-value hand-back,
//! concurrency, torn commits) are in `tests/integration/storage/b_v6d_tests.rs`;
//! this file pins what only the handler adds.
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`update_layer_answers_each_row_of_the_operation_table`] | answer table, T1 | no id → `UNSPECIFIED`; first commit → `APPLIED`; re-send → `ALREADY_COMMITTED` with the original seq/timestamp; different bytes → `ALREADY_EXISTS` |
//! | [`update_layer_resend_after_an_overwrite_or_delete_fails_like_any_write`] | T8 | a re-send to a replaced object is `FAILED_PRECONDITION`, to a deleted layer `NOT_FOUND` — never answered from the record |
//! | [`update_layer_invalid_operation_id_is_refused_before_the_body_is_buffered`] / [`create_layer_invalid_operation_id_is_refused_before_the_body_is_buffered`] | T11 | over-bound and bad-charset ids are `INVALID_ARGUMENT` from the header validator, not the size cap |
//! | [`update_layer_operation_id_without_a_client_id_is_refused_before_the_body_is_buffered`] | T30, T11 (P6) | an `UpdateLayer` `operation_id` with an empty `client_id` is `INVALID_ARGUMENT` before buffering; an empty `client_id` without an id is accepted; `CreateLayer` is unaffected |
//! | [`update_layer_recovery_reissue_keeps_the_call_start_record`] / [`update_layer_recovery_reissue_with_an_absent_start_record_applies`] | T29 | the checkpoint-recovery re-issue carries the start record read at call start: a writer whose record the restore removed is overtaken (`UNAVAILABLE`), and its fresh re-send applies; a writer with no record applies on the re-issue |
//! | [`create_layer_resend_is_answered_already_committed_with_the_original_object`] | T13 | the wire answer, the layer still maps to the original object, and the counter did not move |
//! | [`create_layer_resend_with_different_bytes_is_refused_already_exists`] | T5 | the handler's pre-allocation read refuses a conflicting re-send |
//! | [`lost_post_commit_reply::create_layer_resend_after_a_dropped_overwrite_redrives_its_tail`] | T14 | the original commits and is dropped before its tail (its reply lost, the call parked in the post-failed-run hook); the re-send is `ALREADY_COMMITTED`, re-appends `deleted` to the replaced object's stream, and enqueues a session checkpoint naming the new object |
//! | [`create_layer_resend_answered_by_the_handlers_record_read_spawns_the_recorded_objects_base`] | Amendment 1b | a re-send answered by the handler's record read spawns the creation base of an object whose original create spawned none |
//! | [`lost_post_commit_reply::create_layer_dropped_after_its_commit_point_is_rooted_once_by_its_resend`] | T20, Amendment 1b | the dropped original spawns no creation base; the re-send's spawn runs once and persists one (`persisted`, `skipped_generation_present` and `writes_total{success,layer}` cells) |
//! | [`lost_post_commit_reply::create_layer_duplicate_answered_by_the_commit_script_spawns_the_recorded_objects_base`] | T15, Amendment 1b | a duplicate that loses the race at the commit script spawns the creation base for the RECORDED (winning) object |
//! | [`failed_start_record_read::update_layer_unauthorized_write_is_permission_denied_before_its_start_record_read_starts`] | G1 | the start-record read starts after authorization (and the body), so an unauthorized write is `PERMISSION_DENIED` and never starts it — never `UNAVAILABLE` |
//! | [`failed_start_record_read::update_layer_authorized_write_whose_start_record_read_failed_is_unavailable_and_applies_nothing`] | G1, T28 through the handler | an authorized write whose read failed is `UNAVAILABLE` with nothing committed or recorded; its re-send is `APPLIED` as the first commit |
//! | [`overwrite_second_try::create_layer_overwrite_with_an_operation_id_commits_on_its_second_try`] / [`overwrite_second_try::create_layer_overwrite_without_an_operation_id_commits_on_its_second_try`] | Amendment R4-1 | an overwrite whose first commit try fails before its script runs answers OK with the new `object_id` (`APPLIED` / `UNSPECIFIED`), the mapping names it, an `UpdateLayer` of it commits, and one of the old object is `FAILED_PRECONDITION` |
//!
//! # Feature gate
//!
//! `lost_post_commit_reply`, `failed_start_record_read` and
//! `overwrite_second_try` drive `LayerStore`'s `test-fault-injection` seams (the
//! create commit-dispatch fault injector, the create post-failed-run and
//! pre-commit hooks, the start-record read injector) and are compiled out
//! without `--features test-fault-injection`. Everything above them needs only
//! Valkey.
//!
//! # Wait budget
//!
//! Handler calls are unbounded here (each is a handful of round trips against
//! a healthy Valkey, or a restore from an in-memory checkpoint store); the
//! waits are the checkpoint task drains, each bounded by
//! `wait_for_writer_idle`'s load-tolerant floor, and at most three per test,
//! and in the dropped-overwrite tests one wait (15 s) for the call to park.

use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use tokio_util::task::TaskTracker;
use tonic::Code;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    BaseBootstrap, CheckpointKind, CheckpointStore, SessionCheckpointWriter,
};
use live_edit::domain::types::ObjectState;
use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer, handle_update_layer};
use live_edit::grpc::proto::{
    create_layer_request, update_layer_request, CreateLayerHeader, CreateLayerRequest,
    DeleteLayerRequest, OperationOutcome, UpdateLayerHeader, UpdateLayerRequest,
    UpdateLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{operation_request_digest, CreateOperation, LayerStore};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    assert_header_refused_before_buffering, build_disabled_recovery_hook, build_service,
    build_service_with_size_cap, make_create_data, make_request, make_update_data,
    streaming_create, streaming_update, AllowAllPermissions, SMALL_SIZE_CAP,
};
use crate::checkpoint::helpers::{
    gather_base_bootstrap_cells, read_layer_writes_counter, wait_for_writer_idle,
};
use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline, empty_delta};

/// The operation id the scenarios send: a UUIDv4 in canonical text, the shape
/// the proto comment recommends.
const OPERATION_ID: &str = "0b9e8f3a-1c2d-4e5f-8a6b-7c8d9e0f1a2b";

/// The `client_id` every `UpdateLayer` here carries.
const CLIENT_ID: &str = "bv6d-grpc-client";

/// A body one byte over the size-capped service's `max_layer_size`, for the
/// header-before-body cases.
const OVER_LIMIT_CHUNK_SIZE: usize = SMALL_SIZE_CAP + 1;

/// The charset refusal the header validator answers.
const CHARSET_MESSAGE: &str =
    "operation_id may contain only ASCII letters, digits, '-', '.', '_' and '~'";

// ═════════════════════════════════════════════════════════════════════════════
// Request builders
// ═════════════════════════════════════════════════════════════════════════════

/// An `UpdateLayer` header from [`CLIENT_ID`] carrying `operation_id` (empty =
/// none).
fn update_header(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    timestamp: u64,
    operation_id: &str,
) -> UpdateLayerRequest {
    update_header_from(
        CLIENT_ID,
        session_id,
        layer_name,
        object_id,
        timestamp,
        operation_id,
    )
}

/// An `UpdateLayer` header from `client_id` carrying `operation_id` (empty =
/// none).
fn update_header_from(
    client_id: &str,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    timestamp: u64,
    operation_id: &str,
) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: timestamp,
                client_id: client_id.to_owned(),
                operation_id: operation_id.to_owned(),
            },
        )),
    }
}

/// A `CreateLayer` header carrying `operation_id` (empty = none).
fn create_header(session_id: &str, layer_name: &str, operation_id: &str) -> CreateLayerRequest {
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

/// An OMPF payload setting field `color` to `value`.
fn field_payload(value: u8) -> Vec<u8> {
    delta_to_ompf_bytes(
        &delta_with_set_field_inline(1, "color", vec![value], 0),
        &[],
    )
}

/// One `UpdateLayer` through the handler.
async fn update(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    timestamp: u64,
    operation_id: &str,
    payload: Vec<u8>,
) -> Result<UpdateLayerResponse, tonic::Status> {
    handle_update_layer(
        svc,
        make_request(streaming_update(vec![
            update_header(session_id, layer_name, object_id, timestamp, operation_id),
            make_update_data(payload),
        ])),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// One `CreateLayer` through the handler, answering `(object_id, outcome)`.
async fn create(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    operation_id: &str,
    payload: Vec<u8>,
) -> Result<(u64, OperationOutcome), tonic::Status> {
    let response = handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            create_header(session_id, layer_name, operation_id),
            make_create_data(payload),
        ])),
    )
    .await?
    .into_inner();
    Ok((response.object_id, outcome(response.operation_outcome)))
}

/// Decode a wire `operation_outcome`, failing on a value outside the enum.
fn outcome(raw: i32) -> OperationOutcome {
    OperationOutcome::try_from(raw)
        .unwrap_or_else(|_| panic!("operation_outcome {raw} is not an OperationOutcome value"))
}

async fn new_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), "test-user")
        .await
        .expect("session creation must succeed")
}

async fn object_id_counter(tv: &TestValkey) -> u64 {
    let raw: Option<String> = tv
        .client()
        .pool
        .get(tv.key_schema().counter_object_id())
        .await
        .expect("GET object-id counter must succeed");
    raw.map_or(0, |v| v.parse().expect("the counter is decimal"))
}

async fn mapping(tv: &TestValkey, session_id: &str, layer_name: &str) -> Option<u64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .get(tv.key_schema().layer_mapping(session_id, layer_name))
        .await
        .expect("GET layer mapping must succeed");
    raw.map(|v| v.parse().expect("the mapping holds a decimal object id"))
}

// ═════════════════════════════════════════════════════════════════════════════
// UpdateLayer
// ═════════════════════════════════════════════════════════════════════════════

/// Every row of the `UpdateLayer` answer table, on one layer, through the
/// handler — so the wire `operation_outcome` values, not just the storage
/// enum, are pinned:
///
/// 1. no `operation_id` → `UNSPECIFIED`;
/// 2. a first commit carrying X → `APPLIED`;
/// 3. a re-send of X with the same bytes and a fresh timestamp →
///    `ALREADY_COMMITTED`, with row 2's `sequence_num` and server timestamp;
/// 4. X over different bytes → `ALREADY_EXISTS`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_answers_each_row_of_the_operation_table() {
    const LAYER: &str = "bv6d-grpc-update-table";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "bv6d-grpc-update-table").await;
    let (object_id, _) = create(
        &svc,
        &session_id,
        LAYER,
        "",
        delta_to_ompf_bytes(&empty_delta(), &[]),
    )
    .await
    .expect("setup CreateLayer must succeed");

    let plain = update(
        &svc,
        &session_id,
        LAYER,
        object_id,
        1_000,
        "",
        field_payload(1),
    )
    .await
    .expect("an id-less update must commit");
    assert_eq!(
        (plain.sequence_num, outcome(plain.operation_outcome)),
        (2, OperationOutcome::Unspecified)
    );

    let applied = update(
        &svc,
        &session_id,
        LAYER,
        object_id,
        2_000,
        OPERATION_ID,
        field_payload(2),
    )
    .await
    .expect("the first commit carrying the id must commit");
    assert_eq!(
        (applied.sequence_num, outcome(applied.operation_outcome)),
        (3, OperationOutcome::Applied)
    );

    let resent = update(
        &svc,
        &session_id,
        LAYER,
        object_id,
        3_000,
        OPERATION_ID,
        field_payload(2),
    )
    .await
    .expect("the re-send must be answered OK");
    assert_eq!(
        (
            resent.sequence_num,
            resent.server_timestamp_microseconds_since_epoch,
            outcome(resent.operation_outcome)
        ),
        (
            applied.sequence_num,
            applied.server_timestamp_microseconds_since_epoch,
            OperationOutcome::AlreadyCommitted
        ),
        "the re-send reports the ORIGINAL commit. sequence_num 4 is the edit applied twice"
    );

    let conflict = update(
        &svc,
        &session_id,
        LAYER,
        object_id,
        4_000,
        OPERATION_ID,
        field_payload(9),
    )
    .await
    .expect_err("the same id over different bytes must be refused");
    assert_eq!(conflict.code(), Code::AlreadyExists);
    assert_eq!(
        conflict.message(),
        live_edit::storage::layer::OPERATION_CONFLICT_MESSAGE
    );

    // Fields read as bytes: an entry's relay `ompf` field is binary.
    let entries: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", None)
        .await
        .expect("XRANGE must succeed");
    let ids: Vec<String> = entries.into_iter().map(|(id, _)| id).collect();
    assert_eq!(
        ids,
        ["1-0", "2-0", "3-0"],
        "two commits after CreateLayer's seed: neither the re-send nor the conflict wrote"
    );
}

/// T8: a re-send is a write to the object its header names, and is checked
/// like one BEFORE its record is consulted. After an overwrite replaced that
/// object the re-send is `FAILED_PRECONDITION` (a stale `object_id`); after the
/// layer was deleted it is `NOT_FOUND`. It is never answered
/// `ALREADY_COMMITTED` from the replaced object's record — the client must
/// learn its object is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_resend_after_an_overwrite_or_delete_fails_like_any_write() {
    const LAYER: &str = "bv6d-grpc-t8";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "bv6d-grpc-t8").await;
    let empty = || delta_to_ompf_bytes(&empty_delta(), &[]);

    let (original, _) = create(&svc, &session_id, LAYER, "", empty())
        .await
        .expect("setup CreateLayer must succeed");
    update(
        &svc,
        &session_id,
        LAYER,
        original,
        1_000,
        OPERATION_ID,
        field_payload(3),
    )
    .await
    .expect("the original update must commit");

    let (replacement, _) = create(&svc, &session_id, LAYER, "", empty())
        .await
        .expect("the overwrite must succeed");
    assert_ne!(replacement, original);
    let after_overwrite = update(
        &svc,
        &session_id,
        LAYER,
        original,
        2_000,
        OPERATION_ID,
        field_payload(3),
    )
    .await
    .expect_err("a re-send to a replaced object must fail");
    assert_eq!(
        after_overwrite.code(),
        Code::FailedPrecondition,
        "a re-send after an overwrite is a stale object_id; got {after_overwrite:?}"
    );

    let mut delete = tonic::Request::new(DeleteLayerRequest {
        session_id: session_id.clone(),
        layer_name: LAYER.to_owned(),
    });
    delete
        .extensions_mut()
        .insert(live_edit::grpc::BearerToken::new("test-token"));
    handle_delete_layer(&svc, delete)
        .await
        .expect("DeleteLayer must succeed");
    let after_delete = update(
        &svc,
        &session_id,
        LAYER,
        original,
        3_000,
        OPERATION_ID,
        field_payload(3),
    )
    .await
    .expect_err("a re-send to a deleted layer must fail");
    assert_eq!(
        after_delete.code(),
        Code::NotFound,
        "a re-send after DeleteLayer is NOT_FOUND; got {after_delete:?}"
    );
}

/// T11 (handler half): an `operation_id` over 64 bytes, or carrying a byte
/// outside the allowed set, is refused `INVALID_ARGUMENT` by the header
/// validator — before any chunk is buffered, so an over-cap body behind it is
/// never read to the cap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_invalid_operation_id_is_refused_before_the_body_is_buffered() {
    const LAYER: &str = "bv6d-grpc-t11-update";
    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);
    let session_id = new_session(&tv, "bv6d-grpc-t11-update").await;
    let (object_id, _) = create(
        &svc,
        &session_id,
        LAYER,
        "",
        delta_to_ompf_bytes(&empty_delta(), &[]),
    )
    .await
    .expect("setup CreateLayer must succeed");

    for (operation_id, message) in [
        ("a".repeat(65), "operation_id exceeds 64 byte limit"),
        ("op:1".to_owned(), CHARSET_MESSAGE),
    ] {
        let status = handle_update_layer(
            &svc,
            make_request(streaming_update(vec![
                update_header(&session_id, LAYER, object_id, 1_000, &operation_id),
                make_update_data(vec![0xAB; OVER_LIMIT_CHUNK_SIZE]),
            ])),
        )
        .await
        .expect_err("an invalid operation_id must fail");
        assert_header_refused_before_buffering(&status, message);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_invalid_operation_id_is_refused_before_the_body_is_buffered() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);
    let session_id = new_session(&tv, "bv6d-grpc-t11-create").await;

    for (operation_id, message) in [
        ("a".repeat(65), "operation_id exceeds 64 byte limit"),
        ("op{1}".to_owned(), CHARSET_MESSAGE),
    ] {
        let status = handle_create_layer(
            &svc,
            make_request(streaming_create(vec![
                create_header(&session_id, "bv6d-grpc-t11-create", &operation_id),
                make_create_data(vec![0xAB; OVER_LIMIT_CHUNK_SIZE]),
            ])),
        )
        .await
        .expect_err("an invalid operation_id must fail");
        assert_header_refused_before_buffering(&status, message);
    }
}

/// T30 / T11 (P6): an `UpdateLayer` `operation_id` requires a non-empty
/// `client_id` — its record and its start record are both keyed by
/// `client_id`, so writers sharing the empty one could not be told apart and
/// would overtake each other's calls. Refused `INVALID_ARGUMENT` by the header
/// validator, before an over-cap body behind it is buffered. An empty
/// `client_id` WITHOUT an operation id is still accepted, and `CreateLayer`
/// (which has no `client_id`) is unaffected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_operation_id_without_a_client_id_is_refused_before_the_body_is_buffered() {
    const LAYER: &str = "bv6d-grpc-t30";
    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);
    let session_id = new_session(&tv, "bv6d-grpc-t30").await;
    let (object_id, created) = create(
        &svc,
        &session_id,
        LAYER,
        OPERATION_ID,
        delta_to_ompf_bytes(&empty_delta(), &[]),
    )
    .await
    .expect("CreateLayer with an operation_id needs no client_id");
    assert_eq!(created, OperationOutcome::Applied);

    let status = handle_update_layer(
        &svc,
        make_request(streaming_update(vec![
            update_header_from("", &session_id, LAYER, object_id, 1_000, OPERATION_ID),
            make_update_data(vec![0xAB; OVER_LIMIT_CHUNK_SIZE]),
        ])),
    )
    .await
    .expect_err("an operation_id with an empty client_id must be refused");
    assert_header_refused_before_buffering(&status, "operation_id requires a non-empty client_id");

    let accepted = handle_update_layer(
        &svc,
        make_request(streaming_update(vec![
            update_header_from("", &session_id, LAYER, object_id, 2_000, ""),
            make_update_data(field_payload(30)),
        ])),
    )
    .await
    .expect("an empty client_id without an operation_id is still accepted")
    .into_inner();
    assert_eq!(
        (accepted.sequence_num, outcome(accepted.operation_outcome)),
        (2, OperationOutcome::Unspecified),
        "the refusal applied nothing, so the accepted call is the first commit"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T29 — the start record is not re-read across a checkpoint recovery
// ═════════════════════════════════════════════════════════════════════════════

/// A service on `tv` whose recovery hook is ENABLED over an in-memory store
/// holding one base checkpoint of `object_id` (`layer`, `snap_seq`), with a
/// fresh metrics registry the hook reports into.
async fn recovering_service(
    tv: &TestValkey,
    session_id: &str,
    layer: &str,
    object_id: u64,
    snap_seq: u64,
) -> (
    LiveEditService,
    live_edit::observability::metrics::SharedMetrics,
) {
    let mut svc = build_service(tv, Arc::new(AllowAllPermissions));
    let metrics = crate::common::replace_metrics(&mut svc);
    let store = Arc::new(InMemoryCheckpointStore::new());
    crate::common::seed_layer_checkpoint(&store, object_id, session_id, layer, snap_seq).await;
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = true;
    config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
    svc.recovery = Arc::new(live_edit::checkpoint::RecoveryHook::new(
        store as Arc<dyn CheckpointStore>,
        Arc::clone(&svc.layer_store),
        Arc::new(config),
        Arc::clone(&svc.metrics),
    ));
    (svc, metrics)
}

/// Delete `object_id`'s header only: the commit then answers `NotFound` and
/// the handler restores the object from its checkpoint and re-issues the
/// commit. The rest of the object's keys — the commit-token record included —
/// survive until the restore wipes them.
async fn wipe_header(tv: &TestValkey, object_id: u64) {
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("DEL obj:hdr must succeed");
}

/// One `UpdateLayer` from `client_id` through the handler.
async fn update_from(
    svc: &LiveEditService,
    client_id: &str,
    session_id: &str,
    layer: &str,
    object_id: u64,
    operation_id: &str,
    payload: Vec<u8>,
) -> Result<UpdateLayerResponse, tonic::Status> {
    handle_update_layer(
        svc,
        make_request(streaming_update(vec![
            update_header_from(client_id, session_id, layer, object_id, 5_000, operation_id),
            make_update_data(payload),
        ])),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// T29: the start record is read ONCE per call — at call start — and the
/// checkpoint-recovery re-issue carries that same record rather than reading
/// again. A restore rebuilds the object without its commit-token records, so a
/// writer whose record existed at call start finds the field moved (gone): the
/// re-issue is overtaken, `UNAVAILABLE`, nothing applied. Re-reading after the
/// restore would reopen the window the guard closes. The client's fresh
/// re-send then starts from the restored state and applies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_recovery_reissue_keeps_the_call_start_record() {
    const LAYER: &str = "bv6d-grpc-t29";
    const WRITER_ID: &str = "bv6d-t29-writer";
    let tv = TestValkey::setup().await;
    let plain = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "bv6d-grpc-t29").await;
    let (object_id, _) = create(
        &plain,
        &session_id,
        LAYER,
        "",
        delta_to_ompf_bytes(&empty_delta(), &[]),
    )
    .await
    .expect("setup CreateLayer must succeed");
    update_from(
        &plain,
        WRITER_ID,
        &session_id,
        LAYER,
        object_id,
        "bv6d-t29-a",
        field_payload(1),
    )
    .await
    .expect("the writer's operation A must commit, leaving its record");
    let (svc, metrics) = recovering_service(&tv, &session_id, LAYER, object_id, 2).await;

    wipe_header(&tv, object_id).await;
    let reissued = update_from(
        &svc,
        WRITER_ID,
        &session_id,
        LAYER,
        object_id,
        "bv6d-t29-b",
        field_payload(2),
    )
    .await
    .expect_err("the re-issue after the restore must be overtaken");
    assert_eq!(
        crate::common::recovery_metric(&metrics, "restored"),
        1,
        "precondition: the call took the recovery path and the restore landed"
    );
    assert_eq!(
        (reissued.code(), reissued.message()),
        (
            Code::Unavailable,
            live_edit::storage::layer::OPERATION_OVERTAKEN_MESSAGE
        ),
        "the call started with A's record, and the restore removed it: the re-issue must be \
         overtaken. APPLIED here means the start record was read again after the restore"
    );
    // The overtaken answer is matched on its message prefix and never carries
    // the refusal detail, whose ErrorInfo would tell the client a re-send is
    // safe.
    crate::common::refusal_detail::assert_no_refusal_detail(&reissued);
    let record: Option<String> = tv
        .client()
        .pool
        .hget(
            tv.key_schema().obj_write_token(&object_id.to_string()),
            WRITER_ID,
        )
        .await
        .expect("HGET of the record must succeed");
    assert_eq!(
        record, None,
        "nothing applied: an applied re-issue would have recorded operation B"
    );

    let resent = update_from(
        &svc,
        WRITER_ID,
        &session_id,
        LAYER,
        object_id,
        "bv6d-t29-b",
        field_payload(2),
    )
    .await
    .expect("the client's fresh re-send must apply");
    assert_eq!(
        outcome(resent.operation_outcome),
        OperationOutcome::Applied,
        "the re-send starts from the restored state, where its record is absent"
    );
}

/// T29, the absent case: a writer with no record at call start carries an
/// ABSENT start record through the restore, and the restored object has none
/// either — so the re-issue applies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_recovery_reissue_with_an_absent_start_record_applies() {
    const LAYER: &str = "bv6d-grpc-t29-absent";
    let tv = TestValkey::setup().await;
    let plain = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "bv6d-grpc-t29-absent").await;
    let (object_id, _) = create(
        &plain,
        &session_id,
        LAYER,
        "",
        delta_to_ompf_bytes(&empty_delta(), &[]),
    )
    .await
    .expect("setup CreateLayer must succeed");
    let (svc, metrics) = recovering_service(&tv, &session_id, LAYER, object_id, 1).await;

    wipe_header(&tv, object_id).await;
    let reissued = update_from(
        &svc,
        "bv6d-t29-fresh-writer",
        &session_id,
        LAYER,
        object_id,
        "bv6d-t29-c",
        field_payload(3),
    )
    .await
    .expect("the re-issue after the restore must apply");
    assert_eq!(
        crate::common::recovery_metric(&metrics, "restored"),
        1,
        "precondition: the call took the recovery path and the restore landed"
    );
    assert_eq!(
        outcome(reissued.operation_outcome),
        OperationOutcome::Applied
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// CreateLayer
// ═════════════════════════════════════════════════════════════════════════════

/// T13 through the handler: a create re-send is answered with the original
/// `object_id` and `ALREADY_COMMITTED`, the first create answered `APPLIED`, an
/// id-less create `UNSPECIFIED`; the re-send allocated nothing and the layer
/// still maps to the original object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_resend_is_answered_already_committed_with_the_original_object() {
    const LAYER: &str = "bv6d-grpc-t13";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "bv6d-grpc-t13").await;

    let (_, unspecified) = create(
        &svc,
        &session_id,
        "bv6d-grpc-t13-plain",
        "",
        field_payload(4),
    )
    .await
    .expect("an id-less create must commit");
    assert_eq!(unspecified, OperationOutcome::Unspecified);

    let (original, applied) = create(&svc, &session_id, LAYER, OPERATION_ID, field_payload(4))
        .await
        .expect("the original create must commit");
    assert_eq!(applied, OperationOutcome::Applied);
    let counter = object_id_counter(&tv).await;

    let resent = create(&svc, &session_id, LAYER, OPERATION_ID, field_payload(4))
        .await
        .expect("the re-send must be answered OK");
    assert_eq!(
        resent,
        (original, OperationOutcome::AlreadyCommitted),
        "the re-send must answer the original object_id. A new id is the layer replaced by its \
         own re-send"
    );
    assert_eq!(
        object_id_counter(&tv).await,
        counter,
        "the re-send allocated nothing"
    );
    assert_eq!(mapping(&tv, &session_id, LAYER).await, Some(original));
}

/// T5 through the handler: the same id over different bytes is refused
/// `ALREADY_EXISTS` by the handler's pre-allocation record read, allocates
/// nothing, and leaves the layer on its original object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_resend_with_different_bytes_is_refused_already_exists() {
    const LAYER: &str = "bv6d-grpc-t5-create";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "bv6d-grpc-t5-create").await;

    let (original, _) = create(&svc, &session_id, LAYER, OPERATION_ID, field_payload(5))
        .await
        .expect("the original create must commit");
    let counter = object_id_counter(&tv).await;

    let refused = create(&svc, &session_id, LAYER, OPERATION_ID, field_payload(6))
        .await
        .expect_err("a different request under a recorded id must be refused");
    assert_eq!(refused.code(), Code::AlreadyExists);
    assert_eq!(object_id_counter(&tv).await, counter, "nothing allocated");
    assert_eq!(mapping(&tv, &session_id, LAYER).await, Some(original));
}

// ═════════════════════════════════════════════════════════════════════════════
// The CreateLayer handler's re-send obligations
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on each checkpoint-task drain (clamped up to the helper's
/// load-tolerant floor).
const DRAIN_SECS: u64 = 10;

/// A service whose `CreateLayer` path is fully wired for the re-send
/// obligations: an ENABLED first-base bootstrap and a session-checkpoint
/// writer over one in-memory checkpoint store, sharing one task tracker and
/// one metrics registry.
struct Fixture {
    tv: TestValkey,
    svc: LiveEditService,
    store: Arc<dyn CheckpointStore>,
    metrics: Arc<Metrics>,
    tracker: TaskTracker,
    session_id: String,
}

/// Build the fixture around a `LayerStore` the caller decorates with a
/// seam (`decorate` receives the store with its checkpoint writer already
/// installed).
async fn fixture(name: &str, decorate: impl FnOnce(LayerStore) -> LayerStore) -> Fixture {
    let tv = TestValkey::setup().await;
    let metrics: Arc<Metrics> = Arc::new(crate::common::metrics_scrape_only("test-v6d"));
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let tracker = TaskTracker::new();
    let writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(&metrics),
        "test-v6d".to_owned(),
        true,
    ));
    let layer_store = Arc::new(decorate(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_checkpoint_writer(writer)
            .with_metrics(Arc::clone(&metrics)),
    ));
    let base_bootstrap = Arc::new(BaseBootstrap::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(&metrics),
        "test-v6d".to_owned(),
        true,
    ));
    let base = build_service(&tv, Arc::new(AllowAllPermissions));
    let svc = LiveEditService {
        recovery: build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&metrics)),
        layer_store,
        base_bootstrap,
        ..base
    };
    let session_id = new_session(&tv, name).await;
    Fixture {
        tv,
        svc,
        store,
        metrics,
        tracker,
        session_id,
    }
}

impl Fixture {
    async fn drain(&self) {
        wait_for_writer_idle(&self.tracker, DRAIN_SECS).await;
    }

    /// Every generation the checkpoint store holds for `object_id`.
    async fn generations(&self, object_id: u64) -> Vec<live_edit::checkpoint::CheckpointHandle> {
        self.store
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints must succeed")
    }

    /// The creation-base counter cells that tell one bootstrap run from two:
    /// `(base_bootstrap_total{persisted,creation_base},
    /// base_bootstrap_total{skipped_generation_present,creation_base},
    /// checkpoint_writes_total{success,layer})`.
    ///
    /// The in-memory checkpoint store keeps ONE generation per
    /// `(object_id, snap_seq)`, and every creation base targets
    /// `(object, 1)`, so a generation count cannot distinguish one run from
    /// two at the same key. These cells can: each run ticks exactly one
    /// `base_bootstrap_total` cell.
    fn creation_base_cells(&self) -> (u64, u64, u64) {
        let cells = gather_base_bootstrap_cells(&self.metrics);
        let cell = |outcome: &str| {
            cells
                .get(&(outcome.to_owned(), "creation_base".to_owned()))
                .copied()
                .unwrap_or_else(|| {
                    panic!("base_bootstrap_total{{{outcome},creation_base}} is seeded")
                }) as u64
        };
        (
            cell("persisted"),
            cell("skipped_generation_present"),
            read_layer_writes_counter(&self.metrics, "success"),
        )
    }
}

/// Amendment 1b, the handler's record read: the original create committed
/// at the storage layer, which spawns no creation base, so the recorded
/// object has no lineage root. The client's re-send is answered by the
/// handler's own record `GET` (Step 2b), before the store is reached — and
/// that answer must spawn the recorded object's creation base, rebuilt from
/// the re-sent bytes. Otherwise a create whose reply was lost before its own
/// spawn leaves the layer base-less until it quiesces.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_resend_answered_by_the_handlers_record_read_spawns_the_recorded_objects_base()
{
    const LAYER: &str = "bv6d-grpc-step2b";
    let fx = fixture("bv6d-grpc-step2b", |store| store).await;
    let payload = field_payload(10);

    let original = LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .create_layer_with_operation(
            &fx.session_id,
            LAYER,
            "test-user",
            &ObjectState {
                sequence_num: 1,
                first_version: 1,
                source_format: "usda".to_owned(),
                ..Default::default()
            },
            Arc::new(MockExternalValueStore::new()),
            Some(CreateOperation::new(
                OPERATION_ID,
                operation_request_digest(LAYER, &payload),
            )),
        )
        .await
        .expect("the original create must commit at the storage layer")
        .object_id;
    fx.drain().await;
    assert!(
        fx.generations(original).await.is_empty(),
        "precondition: a storage-layer create spawns no creation base"
    );

    let resent = create(&fx.svc, &fx.session_id, LAYER, OPERATION_ID, payload)
        .await
        .expect("the re-send must be answered OK");
    assert_eq!(
        resent,
        (original, OperationOutcome::AlreadyCommitted),
        "precondition: the re-send is answered with the recorded object"
    );
    fx.drain().await;

    assert_eq!(
        fx.generations(original)
            .await
            .iter()
            .map(|h| (h.snap_seq, h.kind))
            .collect::<Vec<_>>(),
        [(1, CheckpointKind::Base)],
        "the handler's record-read answer must spawn the recorded object's creation base"
    );
    assert_eq!(
        fx.creation_base_cells(),
        (1, 0, 1),
        "exactly one creation-base run, which persisted: (persisted, \
         skipped_generation_present, writes_total{{success,layer}})"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The same obligations after a lost post-commit reply (test-fault-injection
// seams)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod lost_post_commit_reply {
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;

    use fred::clients::Client;
    use futures::FutureExt;

    use live_edit::checkpoint::SessionCheckpoint;
    use live_edit::storage::layer::CreateCommitFault;
    use live_edit::storage::notifications::{DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD};
    use live_edit::storage::watched_txn::WatchWindowHook;

    use super::*;

    impl Fixture {
        /// The object the session checkpoint maps `layer` to.
        async fn checkpointed_mapping(&self, layer: &str) -> Option<u64> {
            let blob = self
                .store
                .get_session_checkpoint(&self.session_id)
                .await
                .expect("get_session_checkpoint must succeed")?;
            let checkpoint = SessionCheckpoint::from_bytes(&blob.bytes)
                .expect("the session checkpoint must decode");
            checkpoint.layer_mappings.get(layer).copied()
        }

        /// How many `payload=deleted` entries `object_id`'s stream holds.
        /// Fields are read as bytes: an entry's relay `ompf` field is binary.
        async fn deleted_entries(&self, object_id: u64) -> usize {
            let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = self
                .tv
                .client()
                .pool
                .xrange_values(self.tv.key_schema().layer_stream(object_id), "-", "+", None)
                .await
                .expect("XRANGE must succeed");
            entries
                .iter()
                .filter(|(_, fields)| {
                    fields.get(NOTIFICATION_PAYLOAD_FIELD).map(|p| &p[..])
                        == Some(DELETED_STREAM_PAYLOAD.as_bytes())
                })
                .count()
        }
    }

    /// Bound on the wait for a dropped call to reach its post-failed-run hook.
    const PARK_BUDGET: std::time::Duration = std::time::Duration::from_secs(15);

    /// The T14 / T20 staging: a layer exists at `original`; an overwrite
    /// carrying the operation id runs its commit, which lands (the mapping now
    /// names the new object) with its reply lost; and the call is dropped
    /// right there — parked in the post-failed-run hook, before its check —
    /// as a deadline or a cancellation drops it. The client got no answer;
    /// the overwrite tail never ran and no creation base was spawned. Returns
    /// `(fixture, original, payload)`.
    async fn dropped_overwrite(name: &str, layer: &str) -> (Fixture, u64, Vec<u8>) {
        let faults: Arc<Mutex<VecDeque<CreateCommitFault>>> = Arc::new(Mutex::new(VecDeque::new()));
        let parked = Arc::new(tokio::sync::Notify::new());
        let park: WatchWindowHook = {
            let parked = Arc::clone(&parked);
            Arc::new(move |_client: &Client| {
                let parked = Arc::clone(&parked);
                async move {
                    parked.notify_one();
                    std::future::pending::<()>().await;
                }
                .boxed()
            })
        };
        let fx = fixture(name, |store| {
            store
                .with_create_layer_commit_fault_injector_for_test(Arc::clone(&faults))
                .with_create_layer_post_failed_run_hook_for_test(park)
        })
        .await;
        let (original, _) = create(&fx.svc, &fx.session_id, layer, "", field_payload(7))
            .await
            .expect("setup CreateLayer must succeed");
        fx.drain().await;
        assert_eq!(
            fx.checkpointed_mapping(layer).await,
            Some(original),
            "precondition: the session checkpoint names the original object"
        );

        faults.lock().unwrap().push_back(CreateCommitFault {
            error: fred::error::Error::new(
                fred::error::ErrorKind::IO,
                "bv6d: the commit's reply was lost",
            ),
            script_ran: true,
        });
        let payload = field_payload(8);
        let call = create(
            &fx.svc,
            &fx.session_id,
            layer,
            OPERATION_ID,
            payload.clone(),
        );
        tokio::select! {
            answered = call => panic!(
                "the overwrite must park after its landed run, not answer; got {answered:?}"
            ),
            () = parked.notified() => {}
            () = tokio::time::sleep(PARK_BUDGET) => panic!(
                "the overwrite did not reach its post-failed-run hook within {PARK_BUDGET:?}"
            ),
        }
        // The call future is dropped here, as a deadline would drop it.
        assert!(
            faults.lock().unwrap().is_empty(),
            "precondition: the commit run took the lost-reply fault"
        );
        fx.drain().await;
        (fx, original, payload)
    }

    /// T14: the original overwrite committed (the mapping names its new object)
    /// and its call was dropped before the overwrite tail ran, so the replaced
    /// object's subscribers were never told and no session checkpoint captured
    /// the new mapping. The client's re-send is answered `ALREADY_COMMITTED`
    /// with the new object and re-drives both: a `deleted` entry lands on the
    /// replaced object's stream, and the session checkpoint now names the new
    /// object.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_resend_after_a_dropped_overwrite_redrives_its_tail() {
        const LAYER: &str = "bv6d-grpc-t14";
        let (fx, original, payload) = dropped_overwrite("bv6d-grpc-t14", LAYER).await;
        let committed = mapping(&fx.tv, &fx.session_id, LAYER)
            .await
            .expect("the commit point flipped the mapping");
        assert_ne!(committed, original, "precondition: the overwrite committed");
        assert_eq!(
            fx.deleted_entries(original).await,
            0,
            "precondition: the dropped call never woke the replaced object's stream"
        );
        assert_eq!(
            fx.checkpointed_mapping(LAYER).await,
            Some(original),
            "precondition: the dropped call enqueued no session checkpoint"
        );

        let resent = create(&fx.svc, &fx.session_id, LAYER, OPERATION_ID, payload)
            .await
            .expect("the re-send must be answered OK");
        assert_eq!(
            resent,
            (committed, OperationOutcome::AlreadyCommitted),
            "the re-send reports the object the original committed"
        );
        fx.drain().await;

        assert_eq!(
            fx.deleted_entries(original).await,
            1,
            "the re-send must re-drive the replaced object's `deleted` wake-up"
        );
        assert_eq!(
            fx.checkpointed_mapping(LAYER).await,
            Some(committed),
            "the re-send must enqueue a session checkpoint capturing the new mapping"
        );
    }

    /// T20 / Amendment 1b: the original call committed and was dropped before
    /// it answered, so its new object IS the layer but the handler never
    /// reached its creation-base spawn — the object has no lineage root. The
    /// client's re-send is answered from the handler's record read and spawns
    /// the recorded object's creation base: exactly ONE, persisted.
    ///
    /// Counted on the bootstrap cells as well as the generation list: the
    /// in-memory store keeps one generation per `(object_id, snap_seq)`, so a
    /// generation count cannot tell one persisting run from two at the same
    /// key; the cells can.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_dropped_after_its_commit_point_is_rooted_once_by_its_resend() {
        const LAYER: &str = "bv6d-grpc-t20";
        let (fx, _, payload) = dropped_overwrite("bv6d-grpc-t20", LAYER).await;
        let committed = mapping(&fx.tv, &fx.session_id, LAYER)
            .await
            .expect("the commit point flipped the mapping");
        assert!(
            fx.generations(committed).await.is_empty(),
            "precondition: the dropped call spawned no creation base"
        );
        assert_eq!(
            fx.creation_base_cells(),
            (1, 0, 1),
            "precondition: only the setup create's creation base persisted, as (persisted, \
             skipped_generation_present, writes_total{{success,layer}})"
        );

        create(&fx.svc, &fx.session_id, LAYER, OPERATION_ID, payload)
            .await
            .expect("the re-send must be answered OK");
        fx.drain().await;

        assert_eq!(
            fx.generations(committed)
                .await
                .iter()
                .map(|h| (h.snap_seq, h.kind))
                .collect::<Vec<_>>(),
            [(1, CheckpointKind::Base)],
            "the re-send must root the object the dropped call made the layer: one Base at \
             snap_seq 1"
        );
        assert_eq!(
            fx.creation_base_cells(),
            (2, 0, 2),
            "the re-send's spawn runs exactly once and persists: (1, 0, 1) is a re-send that \
             spawned nothing; a persisted or writes count of 3 is a second creation base counted \
             for one object"
        );
    }

    /// T15 / Amendment 1b, the commit-script race: a duplicate passes the
    /// handler's record read and the `MGET` (no record yet), and the winner
    /// commits inside the window before its script. The duplicate is answered
    /// with the WINNER's `object_id`, and the handler spawns the creation base
    /// for that recorded object — the winner here ran at the storage layer and
    /// spawned nothing, so the only base can come from the duplicate's answer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_duplicate_answered_by_the_commit_script_spawns_the_recorded_objects_base()
    {
        const LAYER: &str = "bv6d-grpc-t15";
        let winner_slot: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
        let tv_for_hook: Arc<Mutex<Option<(LayerStore, String)>>> = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (winner_slot, tv_for_hook) = (winner_slot.clone(), tv_for_hook.clone());
            Arc::new(move |_client: &Client| {
                let (winner_slot, tv_for_hook) = (winner_slot.clone(), tv_for_hook.clone());
                async move {
                    let (store, session_id) = tv_for_hook
                        .lock()
                        .unwrap()
                        .take()
                        .expect("the competing store is installed before the call");
                    let digest = live_edit::storage::layer::operation_request_digest(
                        LAYER,
                        &field_payload(9),
                    );
                    let commit = store
                        .create_layer_with_operation(
                            &session_id,
                            LAYER,
                            "test-user",
                            &ObjectState {
                                sequence_num: 1,
                                first_version: 1,
                                source_format: "usda".to_owned(),
                                ..Default::default()
                            },
                            Arc::new(MockExternalValueStore::new()),
                            Some(CreateOperation::new(OPERATION_ID, digest)),
                        )
                        .await
                        .expect("the competing duplicate must commit — it is uncontended");
                    *winner_slot.lock().unwrap() = Some(commit.object_id);
                }
                .boxed()
            })
        };
        let fx = fixture("bv6d-grpc-t15", |store| {
            store.with_create_layer_pre_commit_hook_for_test(hook)
        })
        .await;
        *tv_for_hook.lock().unwrap() = Some((
            LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096),
            fx.session_id.clone(),
        ));

        let answered = create(
            &fx.svc,
            &fx.session_id,
            LAYER,
            OPERATION_ID,
            field_payload(9),
        )
        .await
        .expect("the duplicate that lost the race must be answered OK");
        let winner = winner_slot
            .lock()
            .unwrap()
            .expect("precondition: the hook ran and the competing duplicate committed");
        assert_eq!(answered, (winner, OperationOutcome::AlreadyCommitted));
        fx.drain().await;

        assert_eq!(
            fx.generations(winner)
                .await
                .iter()
                .map(|h| (h.snap_seq, h.kind))
                .collect::<Vec<_>>(),
            [(1, CheckpointKind::Base)],
            "the duplicate's answer must spawn the creation base for the RECORDED object"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// G1 — the handler's start-record read starts only once the write is
// authorized and its whole call received (test-fault-injection seam)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod failed_start_record_read {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use live_edit::auth::permissions::PermissionsClient;

    use super::super::helpers::DenyAllPermissions;
    use super::*;

    /// The injected start-record read failures still queued.
    type ReadFailures = Arc<Mutex<VecDeque<fred::error::Error>>>;

    /// A service over `permissions` whose layer store fails the next
    /// call-start read of a start record (one injected transport error, the
    /// `HGET` never sent), and the queue holding that failure.
    fn service_with_a_failing_start_record_read(
        tv: &TestValkey,
        permissions: Arc<dyn PermissionsClient>,
    ) -> (LiveEditService, ReadFailures) {
        let failures: ReadFailures =
            Arc::new(Mutex::new(VecDeque::from([fred::error::Error::new(
                fred::error::ErrorKind::IO,
                "bv6d: start-record read failed",
            )])));
        let base = build_service(tv, permissions);
        let layer_store = Arc::new(
            LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                .with_start_record_read_fred_error_injector_for_test(Arc::clone(&failures)),
        );
        let recovery =
            build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&base.metrics));
        let svc = LiveEditService {
            layer_store,
            recovery,
            ..base
        };
        (svc, failures)
    }

    /// A session with one layer at `seq = 1`, created through an allow-all
    /// service: `(session_id, object_id)`.
    async fn layer(tv: &TestValkey, layer: &str) -> (String, u64) {
        let plain = build_service(tv, Arc::new(AllowAllPermissions));
        let session_id = new_session(tv, layer).await;
        let (object_id, _) = create(
            &plain,
            &session_id,
            layer,
            "",
            delta_to_ompf_bytes(&empty_delta(), &[]),
        )
        .await
        .expect("setup CreateLayer must succeed");
        (session_id, object_id)
    }

    fn assert_read_started(failures: &ReadFailures) {
        assert!(
            failures.lock().unwrap().is_empty(),
            "precondition: the call started its start-record read (the injected failure was \
             consumed)"
        );
    }

    /// An unauthorized write is refused `PERMISSION_DENIED` without starting
    /// its start-record read at all: the read starts only once the whole call
    /// is received, which is after authorization (b.715), so the injected
    /// read failure stays queued. Were the read started before authorization
    /// again, its failure (`UNAVAILABLE`) could reach an unauthorized caller —
    /// telling it to retry, and letting the backing store's state decide an
    /// authorization answer. The authorized sibling below consumes the same
    /// injected failure, so the seam is live and "still queued" is not vacuous.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_unauthorized_write_is_permission_denied_before_its_start_record_read_starts(
    ) {
        const LAYER: &str = "bv6d-grpc-g1-denied";
        let tv = TestValkey::setup().await;
        let (session_id, object_id) = layer(&tv, LAYER).await;
        let (svc, failures) =
            service_with_a_failing_start_record_read(&tv, Arc::new(DenyAllPermissions));

        let refused = update(
            &svc,
            &session_id,
            LAYER,
            object_id,
            1_000,
            OPERATION_ID,
            field_payload(1),
        )
        .await
        .expect_err("an unauthorized write must be refused");

        assert_eq!(
            refused.code(),
            Code::PermissionDenied,
            "authorization answers first; UNAVAILABLE here is the start-record read's failure \
             reaching an unauthorized caller: {refused:?}"
        );
        assert_eq!(
            failures.lock().unwrap().len(),
            1,
            "an unauthorized write must not start its start-record read: the read starts after \
             the body, and the body is read only once the write is authorized"
        );
    }

    /// An authorized write whose start-record read failed ends `UNAVAILABLE`
    /// before any commit attempt, carrying `UPSTREAM_UNAVAILABLE` (b.xd1: the
    /// read precedes the commit marker): nothing is applied and no record is
    /// written, so the client's re-send with the same id is the first commit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_authorized_write_whose_start_record_read_failed_is_unavailable_and_applies_nothing(
    ) {
        const LAYER: &str = "bv6d-grpc-g1-allowed";
        let tv = TestValkey::setup().await;
        let (session_id, object_id) = layer(&tv, LAYER).await;
        let (svc, failures) =
            service_with_a_failing_start_record_read(&tv, Arc::new(AllowAllPermissions));

        let refused = update(
            &svc,
            &session_id,
            LAYER,
            object_id,
            1_000,
            OPERATION_ID,
            field_payload(2),
        )
        .await
        .expect_err("a write whose start-record read failed must be refused");

        assert_read_started(&failures);
        crate::common::refusal_detail::assert_upstream_unavailable(&refused);
        // Fields read as bytes: an entry's relay `ompf` field is binary.
        let entries: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = tv
            .client()
            .pool
            .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", None)
            .await
            .expect("XRANGE must succeed");
        assert_eq!(
            entries.into_iter().map(|(id, _)| id).collect::<Vec<_>>(),
            ["1-0"],
            "nothing may be committed after CreateLayer's seed"
        );
        let record: Option<String> = tv
            .client()
            .pool
            .hget(
                tv.key_schema().obj_write_token(&object_id.to_string()),
                CLIENT_ID,
            )
            .await
            .expect("HGET of the record must succeed");
        assert_eq!(record, None, "nothing may be recorded for the refused call");

        let resent = update(
            &svc,
            &session_id,
            LAYER,
            object_id,
            2_000,
            OPERATION_ID,
            field_payload(2),
        )
        .await
        .expect("the re-send, whose read succeeds, must commit");
        assert_eq!(
            (resent.sequence_num, outcome(resent.operation_outcome)),
            (2, OperationOutcome::Applied),
            "the refused call applied nothing, so its re-send is the first commit"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Amendment R4-1 — an overwrite whose first commit try failed commits on its
// second (test-fault-injection seam)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod overwrite_second_try {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use live_edit::storage::layer::CreateCommitFault;

    use super::*;

    /// Amendment R4-1 through the handlers: an overwrite `CreateLayer` whose
    /// commit script fails transiently before it runs (one injected `IO`
    /// fault: the connection dropped before the write) has already marked the
    /// old object superseded. The commit is tried once more, so the call
    /// answers OK with the NEW `object_id`, the mapping names it, and an
    /// `UpdateLayer` of it commits; the old object refuses its writers
    /// `FAILED_PRECONDITION`.
    ///
    /// Without the second try the `CreateLayer` answered `UNAVAILABLE` and left
    /// the superseded old object mapped, so every `UpdateLayer` to the layer
    /// failed `FAILED_PRECONDITION` until a re-send.
    async fn overwrite_commits_on_its_second_try(
        name: &str,
        operation_id: &str,
        expected: OperationOutcome,
    ) {
        const LAYER: &str = "bv6d-grpc-r41";
        let faults: Arc<Mutex<VecDeque<CreateCommitFault>>> = Arc::new(Mutex::new(VecDeque::new()));
        let fx = fixture(name, |store| {
            store.with_create_layer_commit_fault_injector_for_test(Arc::clone(&faults))
        })
        .await;
        let (original, _) = create(&fx.svc, &fx.session_id, LAYER, "", field_payload(1))
            .await
            .expect("setup CreateLayer must succeed");
        faults.lock().unwrap().push_back(CreateCommitFault {
            error: fred::error::Error::new(fred::error::ErrorKind::IO, "bv6d: connection reset"),
            script_ran: false,
        });

        let (replaced, answered) = create(
            &fx.svc,
            &fx.session_id,
            LAYER,
            operation_id,
            field_payload(2),
        )
        .await
        .expect("the overwrite must commit on its second try");

        assert!(
            faults.lock().unwrap().is_empty(),
            "precondition: the first try's failure was injected"
        );
        assert_ne!(replaced, original, "the overwrite answers its NEW object");
        assert_eq!(answered, expected);
        assert_eq!(mapping(&fx.tv, &fx.session_id, LAYER).await, Some(replaced));
        let updated = update(
            &fx.svc,
            &fx.session_id,
            LAYER,
            replaced,
            5_000,
            "",
            field_payload(3),
        )
        .await
        .expect("an UpdateLayer of the new object must commit, not fail FAILED_PRECONDITION");
        assert_eq!(updated.sequence_num, 2);
        let stale = update(
            &fx.svc,
            &fx.session_id,
            LAYER,
            original,
            6_000,
            "",
            field_payload(4),
        )
        .await
        .expect_err("an UpdateLayer of the replaced object must be refused");
        assert_eq!(stale.code(), Code::FailedPrecondition, "{stale:?}");
        fx.drain().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_with_an_operation_id_commits_on_its_second_try() {
        overwrite_commits_on_its_second_try(
            "bv6d-grpc-r41-id",
            OPERATION_ID,
            OperationOutcome::Applied,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_overwrite_without_an_operation_id_commits_on_its_second_try() {
        overwrite_commits_on_its_second_try(
            "bv6d-grpc-r41-no-id",
            "",
            OperationOutcome::Unspecified,
        )
        .await;
    }
}
