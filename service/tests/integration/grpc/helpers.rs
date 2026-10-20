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

// Shared test helpers for gRPC integration tests.
//
// Mock auth clients, service builder, streaming helpers (CreateLayer,
// UpdateLayer and JoinSession), message builders, JoinSession presence
// fixture steps (scenario budget, seam and condition waits, the router feed,
// remote registration and eviction, probes), and external presence-channel
// observers (a raw subscriber, one on a chosen cluster node, and `Tap`), shared
// across the gRPC test modules.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::BufMut;
use dashmap::DashMap;
use prost::Message;
use tonic::codec::{BufferSettings, Streaming};
use tonic::Request;
use tonic_prost::ProstCodec;

use live_edit::auth::permissions::PermissionsClient;
use live_edit::auth::userinfo::UserinfoClient;
use live_edit::auth::{AuthError, AuthResult, Permission, UserInfo};
use live_edit::checkpoint::{
    adapters::InMemoryCheckpointStore, CheckpointStore, RecoveryHook, SessionRecoveryHook,
};
use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::proto::{
    create_layer_request, join_session_request, join_session_response, update_layer_request,
    CreateLayerHeader, CreateLayerRequest, JoinSessionHeader, JoinSessionRequest,
    JoinSessionResponse, SessionMessageRequest, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::Metrics;
use live_edit::presence::PresenceEvent;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

// ── Mock auth clients ─────────────────────────────────────────────────────────

pub struct AllowAllPermissions;

#[async_trait]
impl PermissionsClient for AllowAllPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        Ok(AuthResult::Allowed)
    }
}

pub struct DenyAllPermissions;

#[async_trait]
impl PermissionsClient for DenyAllPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        Ok(AuthResult::Denied)
    }
}

/// Auth stub that always returns `Unauthenticated` — simulates the
/// permissions service rejecting the bearer token as invalid/expired.
pub struct UnauthenticatedPermissions;

#[async_trait]
impl PermissionsClient for UnauthenticatedPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        Ok(AuthResult::Unauthenticated)
    }
}

/// Auth stub that always returns `PathNotFound` — simulates storage-service
/// returning 404 for the requested base path.
pub struct PathNotFoundPermissions;

#[async_trait]
impl PermissionsClient for PathNotFoundPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        Ok(AuthResult::PathNotFound)
    }
}

/// The `backend_status` [`InvalidRequestPermissions`] answers with.
pub const INVALID_REQUEST_BACKEND_STATUS: &str = "HTTP 400";

/// Auth stub that always returns `InvalidRequest { "HTTP 400" }` — simulates
/// the storage service refusing the permission probe as a malformed request
/// (a base path that is not a resource address it accepts).
pub struct InvalidRequestPermissions;

#[async_trait]
impl PermissionsClient for InvalidRequestPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        Ok(AuthResult::InvalidRequest {
            backend_status: INVALID_REQUEST_BACKEND_STATUS,
        })
    }
}

/// Mock `PermissionsClient` that allows everything **and records every path it
/// was probed with**, in call order.
///
/// The sibling doubles above all ignore their `path` argument, so none of them
/// can witness *which* spelling of the base path reached the permission probe.
/// That is exactly the b.uoc contract: `CreateSession` / `ListSessions` must
/// authorize against the canonicalized path, not the caller's raw one, so that
/// live-edit's auth-cache keys, its Valkey keys, and the permission service's
/// own normalized resource ids all agree on one string.
///
/// Hold an `Arc<RecordingPermissions>` in the test, pass a clone to
/// [`build_service`], and assert on [`RecordingPermissions::calls`] afterwards.
pub struct RecordingPermissions {
    result: AuthResult,
    calls: Mutex<Vec<(String, Permission)>>,
}

impl RecordingPermissions {
    /// Records every probe and answers `Allowed`.
    pub fn allowing() -> Self {
        Self {
            result: AuthResult::Allowed,
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Every `(path, permission)` pair the service probed, in call order.
    pub fn calls(&self) -> Vec<(String, Permission)> {
        self.calls.lock().expect("recording mutex poisoned").clone()
    }
}

#[async_trait]
impl PermissionsClient for RecordingPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        path: &str,
        permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        self.calls
            .lock()
            .expect("recording mutex poisoned")
            .push((path.to_owned(), permission));
        Ok(self.result.clone())
    }
}

pub struct StubUserinfo;

#[async_trait]
impl UserinfoClient for StubUserinfo {
    async fn get_current_user(&self, _token: &str) -> Result<UserInfo, AuthError> {
        Ok(UserInfo {
            id: "test-user".into(),
            display_name: "Test User".into(),
            email: Some("test@example.com".into()),
            username: "testuser".into(),
        })
    }
}

// ── Recovery hook test builder ───────────────────────────────────────────────

/// Build a `RecoveryHook` configured with `checkpoint_enabled=false`.
///
/// The hook's `try_restore_layer` short-circuits to `NotRecoverable` in this
/// state without touching the store, admission map, or metric — exactly the
/// behavior gRPC integration tests want when checkpointing is not part of the
/// system under test. Plumbing this into every `LiveEditService` construction
/// keeps the recovery field present without coupling unrelated tests to the
/// checkpoint subsystem.
pub fn build_disabled_recovery_hook(
    layer_store: Arc<LayerStore>,
    metrics: Arc<Metrics>,
) -> Arc<RecoveryHook> {
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = false;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    Arc::new(RecoveryHook::new(
        store,
        layer_store,
        Arc::new(config),
        metrics,
    ))
}

/// Build a `SessionRecoveryHook` configured with `checkpoint_enabled=false`.
///
/// Sibling helper to [`build_disabled_recovery_hook`] for the session-state
/// recovery surface added in Subtask `t3.fgj.7v.fp.rd`. The hook's
/// `try_recover_session` short-circuits to `NoCheckpoint` in this state
/// without touching the store, admission map, or metric — exactly the
/// behaviour gRPC integration tests want when session-checkpointing is not
/// part of the system under test. Plumbing this into every `LiveEditService`
/// construction keeps the field present without coupling unrelated tests to
/// the checkpoint subsystem.
pub fn build_disabled_session_recovery_hook(
    tv: &TestValkey,
    metrics: Arc<Metrics>,
) -> Arc<SessionRecoveryHook> {
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = false;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    Arc::new(SessionRecoveryHook::new(
        store,
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::new(config),
        metrics,
    ))
}

// ── Service builder ──────────────────────────────────────────────────────────

/// Build a LiveEditService backed by a real Valkey session store and the given
/// permissions client. The userinfo client defaults to `StubUserinfo`, which
/// ignores the bearer token and returns a fixed `test-user` identity — fine for
/// tests that do not care about JWT decode. Tests that need to exercise the
/// real `JwtUserinfoClient` (or any other concrete `UserinfoClient`) should
/// call [`build_service_with_userinfo`] instead.
pub fn build_service(tv: &TestValkey, permissions: Arc<dyn PermissionsClient>) -> LiveEditService {
    build_service_with_userinfo(tv, permissions, Arc::new(StubUserinfo))
}

/// Like [`build_service`] but lets the caller inject a concrete
/// `UserinfoClient` (e.g. a real `JwtUserinfoClient` for the SR-2.x integration
/// tests). All other wiring is identical to `build_service`.
pub fn build_service_with_userinfo(
    tv: &TestValkey,
    permissions: Arc<dyn PermissionsClient>,
    userinfo: Arc<dyn UserinfoClient>,
) -> LiveEditService {
    let session_store = Arc::new(SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let recovery = build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&metrics));
    let session_recovery = build_disabled_session_recovery_hook(tv, Arc::clone(&metrics));
    let base_bootstrap = crate::common::build_disabled_base_bootstrap(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&metrics),
    );

    LiveEditService {
        catchup_snapshot_enabled: false,
        session_store,
        layer_store,
        ext_value_store: Arc::new(MockExternalValueStore::new()),
        session_tokens: Arc::new(DashMap::new()),
        userinfo_client: userinfo,
        permissions_client: permissions,
        max_layer_size: 10 * 1024 * 1024, // 10 MB
        upload_admission: Arc::new(live_edit::grpc::upload_admission::UploadAdmissionGate::new(
            Default::default(),
            Some(Arc::clone(&metrics)),
        )),
        snapshot_flights: live_edit::storage::snapshot_flights::SnapshotFlights::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ),
        valkey: tv.client().clone(),
        key_schema: tv.key_schema().clone(),
        presence_router: Arc::new(live_edit::presence::LocalPresenceRouter::new()),
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        presence_cleanup_tracker: tokio_util::task::TaskTracker::new(),
        subscription_tracker: tokio_util::task::TaskTracker::new(),
        shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown::new(),
        metrics,
        service_label: "test".to_string(),
        session_base_path_cache: live_edit::grpc::session_cache::SessionBasePathCache::new(),
        recovery,
        session_recovery,
        base_bootstrap,
        service_token_provider: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_xrange_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        outer_send_deadline_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_chunk_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        run_catch_up_inner_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_out_of_window_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_state_read_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        catch_up_build_turn_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        capacity_budget_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        join_registration_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        presence_replay_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        lease_keeper_seam_for_test: None,
        #[cfg(feature = "test-fault-injection")]
        presence_channel_seam_for_test: None,
    }
}

/// `build_service` with `layer_store` swapped in, so a test can install a
/// `LayerStore`-level fault seam before the store is shared into the service.
/// The recovery hook is rebuilt over the same store so the two never disagree
/// (it is disabled either way).
#[cfg(feature = "test-fault-injection")]
pub fn build_service_with_layer_store(
    tv: &TestValkey,
    layer_store: Arc<LayerStore>,
) -> LiveEditService {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let recovery =
        build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&base.metrics));
    LiveEditService {
        layer_store,
        recovery,
        ..base
    }
}

/// A `LayerStore` whose `delete_layer` consults `fault_map` by layer name
/// before touching Valkey (the map entry is returned verbatim, every time).
#[cfg(feature = "test-fault-injection")]
pub fn layer_store_with_name_faults(
    tv: &TestValkey,
    fault_map: Arc<Mutex<std::collections::HashMap<String, live_edit::error::ServiceError>>>,
) -> Arc<LayerStore> {
    Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_delete_layer_by_name_fault_injector_for_test(fault_map),
    )
}

// ── SR-2.x verbatim error wording ────────────────────────────────────────────
//
// Re-exported from `tests/common/mod.rs` so both the `integration` and `misc`
// test binaries share a single source of truth for the SR-2.4 / SR-2.5
// pin-strings. The constants MUST stay character-for-character identical to
// the strings produced by
// `src/auth/userinfo_jwt.rs::JwtUserinfoClient::get_current_user` — drift
// there is a Task 2 contract violation. Routing both binaries through the
// shared `common` module means a reword surfaces as a single compile-time fix
// rather than as silent no-ops in sibling tests across either binary.
//
// The re-export (rather than removing the local names) preserves the
// `use super::helpers::*;` import pattern that every gRPC integration test
// file already uses — callers continue to name `SR_2_4_OPAQUE_TOKEN_MSG` /
// `SR_2_5_MISSING_CLAIM_MSG` unchanged.
pub use crate::common::{SR_2_4_OPAQUE_TOKEN_MSG, SR_2_5_MISSING_CLAIM_MSG};

// ── JWT builder ──────────────────────────────────────────────────────────────

/// Re-export of the shared `make_jwt` from `tests/common/mod.rs`. Existing
/// `use super::helpers::*;` call sites in `session_test`, `create_layer_test`,
/// `join_session_test` resolve `make_jwt` through this re-export, so the
/// single source of truth lives in the cross-binary `common` module while
/// per-RPC test files keep their existing import shape.
pub use crate::common::make_jwt;

// ── Userinfo "no such user" fallback ─────────────────────────────────────────

/// Entra object id of an app the userinfo directory does not hold (invented).
pub const UNLISTED_APP_OID: &str = "11111111-2222-3333-4444-555555555555";

/// An Entra app-only access token (`idtyp=app`, `oid == sub`) for the app
/// `app_displayname`, whose principal the userinfo directory does not hold.
pub fn unlisted_app_jwt(app_displayname: &str) -> String {
    make_jwt(&serde_json::json!({
        "idtyp": "app",
        "app_displayname": app_displayname,
        "azp": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "oid": UNLISTED_APP_OID,
        "sub": UNLISTED_APP_OID,
    }))
}

/// The userinfo client a deployment with the userinfo service and
/// authorization both on gets from `build_userinfo_client`, pointed at a REST
/// userinfo service that answers every caller "no such user" (404 with its
/// `not_found` body). The returned `MockServer` must outlive the client.
pub async fn userinfo_answering_not_found() -> (Arc<dyn UserinfoClient>, wiremock::MockServer) {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1alpha/users/me"))
        .respond_with(ResponseTemplate::new(404).set_body_raw(
            br#"{"error":"not_found","message":"not found"}"#.to_vec(),
            "application/json",
        ))
        .mount(&mock_server)
        .await;

    let mut config = crate::common::test_config();
    config.auth_protocol = live_edit::config::AuthProtocol::Rest;
    config.authz_enabled = true;
    config.userinfo_service_enabled = true;
    config.userinfo_endpoint = Some(mock_server.uri());
    let client = live_edit::auth::build_userinfo_client(
        &config,
        &live_edit::auth::fast_retry(1),
        Arc::new(crate::common::metrics_scrape_only("test")),
    )
    .await
    .expect("REST userinfo builder must succeed");
    (client, mock_server)
}

// ── Streaming helpers ────────────────────────────────────────────────────────

pub use crate::common::upload_stream::{AttestedStreaming, IntoHandlerRequest};

/// Encode protobuf messages into a gRPC-framed body that ends with
/// END_STREAM, decoded as a `tonic::Streaming` paired with its end-of-stream
/// record (`crate::common::upload_stream`). Each message is framed as:
///   [0u8 compression_flag] [4-byte big-endian length] [protobuf bytes]
pub fn streaming_create(
    messages: Vec<CreateLayerRequest>,
) -> AttestedStreaming<CreateLayerRequest> {
    AttestedStreaming::from_messages(&messages)
}

/// Encode `UpdateLayerRequest` messages as [`streaming_create`] does, for the
/// client-streaming UpdateLayer RPC.
///
/// Promoted here from `update_layer_test.rs` (where it was a private fn) so the
/// ompf-invariant module can drive real UpdateLayer commits through
/// `handle_update_layer` without duplicating the framer.
pub fn streaming_update(
    messages: Vec<UpdateLayerRequest>,
) -> AttestedStreaming<UpdateLayerRequest> {
    AttestedStreaming::from_messages(&messages)
}

// ── Message builders ─────────────────────────────────────────────────────────

/// Build an UpdateLayerRequest header message. Promoted from
/// `update_layer_test.rs`; `client_id` / timestamp are fixed test constants —
/// callers that need to vary either build the header inline.
pub fn make_update_header(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: 1000,
                client_id: "test-client".to_owned(),
                operation_id: String::new(),
            },
        )),
    }
}

/// Build an UpdateLayerRequest data chunk message. Promoted from
/// `update_layer_test.rs`.
pub fn make_update_data(payload: Vec<u8>) -> UpdateLayerRequest {
    UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Chunk(
            live_edit::grpc::proto::UpdateLayerChunk { data: payload },
        )),
    }
}

/// Build a CreateLayerRequest header message.
pub fn make_create_header(session_id: &str, layer_name: &str) -> CreateLayerRequest {
    CreateLayerRequest {
        message_type: Some(create_layer_request::MessageType::Header(
            CreateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                operation_id: String::new(),
            },
        )),
    }
}

/// Build a CreateLayerRequest data chunk message.
pub fn make_create_data(payload: Vec<u8>) -> CreateLayerRequest {
    CreateLayerRequest {
        message_type: Some(create_layer_request::MessageType::Chunk(
            live_edit::grpc::proto::CreateLayerChunk { data: payload },
        )),
    }
}

// ── Request builder ──────────────────────────────────────────────────────────

/// Wrap a request stream in a tonic::Request with a BearerToken extension —
/// and, for an [`AttestedStreaming`], its end-of-stream and body-progress
/// records.
pub fn make_request<T>(stream: impl IntoHandlerRequest<T>) -> Request<Streaming<T>> {
    let mut request = stream.into_handler_request();
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ── The upload size cap ─────────────────────────────────────────────────────

/// A small `max_layer_size` for tests that need an upload to reach the size
/// cap: the b.y2t header-before-body tests and the b.75i size-cap refusal
/// tests. Deliberately far below tonic's 4 MiB per-message decode limit, so a
/// body one byte over THIS cap reaches `read_upload_body`'s size check instead of
/// being refused by the decoder (`CANCELLED`) before the size cap could run —
/// which is what a body over the default 10 MB cap would hit.
pub const SMALL_SIZE_CAP: usize = 4096;

/// [`build_service`] with `max_layer_size` overridden.
pub fn build_service_with_size_cap(
    tv: &TestValkey,
    permissions: Arc<dyn PermissionsClient>,
    max_layer_size: usize,
) -> LiveEditService {
    LiveEditService {
        max_layer_size,
        ..build_service(tv, permissions)
    }
}

/// The fragment of the size cap's refusal message (`grpc::streaming::read_upload_body`).
/// Its presence in a header refusal means the body was buffered to the cap
/// before the header was looked at.
pub const SIZE_CAP_MESSAGE_FRAGMENT: &str = "layer content exceeds maximum size";

/// The size cap's whole refusal message for a service built with `max_size`.
pub fn size_cap_message(max_size: usize) -> String {
    format!("{SIZE_CAP_MESSAGE_FRAGMENT} of {max_size} bytes")
}

// ── Header-before-body refusals (b.y2t) ─────────────────────────────────────

/// Assert `status` is the header validator's `INVALID_ARGUMENT`, carrying
/// `header_message_fragment`, and not the size cap's.
///
/// The CODE cannot separate the two: since b.75i the size cap answers
/// `INVALID_ARGUMENT` as well, so the b.y2t tests in `create_layer_test` and
/// `update_layer_test` — which send an over-cap body behind an invalid header
/// — tell "header validated first" from "body buffered first" by the message.
pub fn assert_header_refused_before_buffering(
    status: &tonic::Status,
    header_message_fragment: &str,
) {
    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "an invalid header must be refused INVALID_ARGUMENT; got {status:?}"
    );
    assert!(
        !status.message().contains(SIZE_CAP_MESSAGE_FRAGMENT),
        "the refusal must come from header validation, not the size cap — the body was \
         buffered to the cap before the header was inspected; got {:?}",
        status.message()
    );
    assert!(
        status.message().contains(header_message_fragment),
        "the refusal must be the header validator's, naming {header_message_fragment:?}; got \
         {:?}",
        status.message()
    );
}

// ── Result helpers ───────────────────────────────────────────────────────────

/// The `Err` status of a handler call that must have failed, or a panic naming
/// `context`.
///
/// `Result::expect_err` is unavailable for the streaming handlers: their `Ok`
/// half is a boxed response stream, which is not `Debug`.
pub fn expect_err_status<T>(result: Result<T, tonic::Status>, context: &str) -> tonic::Status {
    match result {
        Err(status) => status,
        Ok(_) => panic!("{context}"),
    }
}

// ── JoinSession stream helpers ───────────────────────────────────────────────

/// The response stream `handle_join_session` returns.
pub type JoinResponseStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<JoinSessionResponse, tonic::Status>> + Send>,
>;

/// Encode one `JoinSessionRequest` as a gRPC frame: compression flag, 4-byte
/// big-endian length, protobuf bytes.
pub fn encode_join_frame(msg: &JoinSessionRequest) -> bytes::Bytes {
    let encoded = msg.encode_to_vec();
    let mut buf = bytes::BytesMut::with_capacity(5 + encoded.len());
    buf.put_u8(0); // no compression
    buf.put_u32(encoded.len() as u32);
    buf.extend_from_slice(&encoded);
    buf.freeze()
}

/// A JoinSession header message.
pub fn join_header(session_id: &str, user_application: &str) -> JoinSessionRequest {
    JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Header(
            JoinSessionHeader {
                session_id: session_id.to_owned(),
                user_application: user_application.to_owned(),
                client_keepalive: false,
            },
        )),
    }
}

/// A JoinSession client message, which the join publishes as a `PeerMessage`.
pub fn join_message(data: Vec<u8>) -> JoinSessionRequest {
    JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Message(
            SessionMessageRequest { data },
        )),
    }
}

/// A `tonic::Streaming<JoinSessionRequest>` fed by the returned sender, which
/// takes frames from [`encode_join_frame`]. The client side stays open until
/// the sender is dropped, which the join's loop reads as the client closing its
/// stream.
pub fn open_join_streaming() -> (
    Streaming<JoinSessionRequest>,
    tokio::sync::mpsc::UnboundedSender<bytes::Bytes>,
) {
    use futures::StreamExt;
    use http_body_util::StreamBody;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
    let rx_stream = UnboundedReceiverStream::new(rx)
        .map(|b| Ok::<_, std::io::Error>(http_body::Frame::data(b)));
    let body = StreamBody::new(rx_stream);
    let decoder = ProstCodec::<(), JoinSessionRequest>::raw_decoder(BufferSettings::default());
    (Streaming::new_request(decoder, body, None, None), tx)
}

/// [`open_join_streaming`] with the records the LiveService stack attaches to
/// every request body, the body-progress record among them: what an opted-in
/// join needs to be armed for the client keepalive (SDD §13.2). Each `Bytes`
/// sent is one DATA frame of the body, so a test can split one gRPC frame
/// over several sends to model a message still arriving.
pub fn open_attested_join_streaming() -> (
    AttestedStreaming<JoinSessionRequest>,
    tokio::sync::mpsc::UnboundedSender<bytes::Bytes>,
) {
    use futures::StreamExt;
    use http_body_util::StreamBody;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
    let rx_stream = UnboundedReceiverStream::new(rx)
        .map(|b| Ok::<_, std::io::Error>(http_body::Frame::data(b)));
    (AttestedStreaming::new(StreamBody::new(rx_stream)), tx)
}

/// Wrap a JoinSession client stream in a request carrying a bearer token.
pub fn make_join_request(
    stream: Streaming<JoinSessionRequest>,
) -> Request<Streaming<JoinSessionRequest>> {
    make_request(stream)
}

/// Open a JoinSession on `svc`: send the header on a fresh client stream and
/// call the handler, which must admit the join. Returns the response stream
/// and the client-side sender. Waits for nothing the spawned join task does.
pub async fn open_join(
    svc: &LiveEditService,
    session_id: &str,
    user_application: &str,
) -> (
    JoinResponseStream,
    tokio::sync::mpsc::UnboundedSender<bytes::Bytes>,
) {
    let (stream, tx) = open_join_streaming();
    tx.send(encode_join_frame(&join_header(
        session_id,
        user_application,
    )))
    .expect("the client channel must accept the header frame");
    let response = live_edit::grpc::presence::handle_join_session(svc, make_join_request(stream))
        .await
        .unwrap_or_else(|s| panic!("the JoinSession handler must admit the join: {s:?}"));
    (response.into_inner(), tx)
}

/// Read the stream's next item within `within`, which must be
/// `SessionConnected`, and return its connection id.
pub async fn expect_session_connected(
    stream: &mut JoinResponseStream,
    within: std::time::Duration,
) -> String {
    use futures::StreamExt;
    let first = tokio::time::timeout(within, stream.next())
        .await
        .unwrap_or_else(|_| panic!("SessionConnected did not arrive within {within:?}"))
        .expect("the join stream ended before SessionConnected")
        .unwrap_or_else(|s| panic!("the join stream failed before SessionConnected: {s:?}"));
    match first.response_type {
        Some(join_session_response::ResponseType::Connected(c)) => c.connection_id,
        other => panic!("the join stream's first message must be SessionConnected; got {other:?}"),
    }
}

// ── JoinSession presence: store readers, seeding, drains, session setup ──────

/// Bound on each wait the helpers in this section perform. Unreachable on the
/// healthy path.
pub const PRESENCE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Create a session in the store. With `with_token`, also register the
/// session's cancellation token on `svc`, as `CreateSession` on this replica
/// would; without it, the session has no token entry here, as for a session
/// created on another replica. Returns the session id and that token (an
/// unregistered one when `with_token` is false).
pub async fn create_session(
    tv: &TestValkey,
    svc: &LiveEditService,
    name: &str,
    with_token: bool,
) -> (String, tokio_util::sync::CancellationToken) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let token = tokio_util::sync::CancellationToken::new();
    if with_token {
        svc.session_tokens
            .insert(session_id.clone(), token.clone().into());
    }
    (session_id, token)
}

/// The connection presence-store reads go through: the maintenance pool, so a
/// read never queues behind a stalled data-pool connection and never moves the
/// data pool's round-robin cursor, which a test may rely on to know which data
/// connection a join's next command takes.
pub fn store_reader(tv: &TestValkey) -> &fred::clients::Client {
    tv.client().maintenance_pool.next()
}

/// Every presence key under `session_id`'s hash tag — the member set and each
/// connection's info hash, lease and teardown marker — sorted.
pub async fn presence_keys(tv: &TestValkey, session_id: &str) -> Vec<String> {
    use futures::StreamExt;

    let pattern = format!("{}*", tv.key_schema().presence_members(session_id));
    let mut keys = Vec::new();
    let mut scan = store_reader(tv).scan_buffered(&pattern, Some(100), None);
    while let Some(key) = scan.next().await {
        keys.push(key.expect("SCAN must succeed").as_str_lossy().into_owned());
    }
    keys.sort();
    keys
}

/// Whether `session_id` is in the session index (`le:presence:active_sessions`).
pub async fn is_indexed(tv: &TestValkey, session_id: &str) -> bool {
    use fred::interfaces::SetsInterface;
    store_reader(tv)
        .sismember(tv.key_schema().presence_active_sessions(), session_id)
        .await
        .expect("SISMEMBER active_sessions must succeed")
}

/// Whether `cid` is in `session_id`'s presence member set.
pub async fn is_member(tv: &TestValkey, session_id: &str, cid: &str) -> bool {
    use fred::interfaces::SetsInterface;
    store_reader(tv)
        .sismember(tv.key_schema().presence_members(session_id), cid)
        .await
        .expect("SISMEMBER must succeed")
}

/// A join's response-channel delivery slots: `handle_join_session` makes the
/// channel 33 deep and reserves one slot for the final status. A replay with
/// more members to send than this, to a client that has stopped reading, waits
/// on a send. Ungated: `presence_resync_test` sizes its forced lag from it in
/// every feature shape.
pub const RESPONSE_DELIVERY_SLOTS: usize = 32;

/// Other members for a replay of one page (under `REPLAY_PAGE`) that overfills
/// the response channel, so a client that reads only `SessionConnected` leaves
/// the replay waiting on a send.
#[cfg(feature = "test-fault-injection")]
pub const ONE_PAGE_OVERFILLING_THE_CHANNEL: usize = RESPONSE_DELIVERY_SLOTS + 8;

/// `n` distinct connection ids for [`seed_presence_members`], sorting in seed
/// order.
#[cfg(feature = "test-fault-injection")]
pub fn seeded_cids(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("seeded-{i:04}")).collect()
}

/// The `user_application` of every member [`seed_presence_members`] writes.
#[cfg(feature = "test-fault-injection")]
pub const SEEDED_APP: &str = "seeded-app";

/// The `user_name` [`seed_presence_members`] writes for `cid`.
#[cfg(feature = "test-fault-injection")]
pub fn seeded_user_name(cid: &str) -> String {
    format!("User {cid}")
}

/// Seed `cids` into `session_id`'s presence state straight through the store,
/// as joins on other replicas would have left them: each a member, with a
/// lease and a complete info hash (`user_name` [`seeded_user_name`],
/// `user_application` [`SEEDED_APP`]). One pipeline on the maintenance pool;
/// the member set gains `cids` in one `SADD`, in `cids` order.
///
/// Nothing is published and the session index is not touched, so no local
/// stream or peer hears of these members. A test that needs a member announced
/// registers it through the register script instead.
///
/// Compiled with `test-fault-injection` only: every caller drives the replay
/// seam.
#[cfg(feature = "test-fault-injection")]
pub async fn seed_presence_members(tv: &TestValkey, session_id: &str, cids: &[String]) {
    use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};

    if cids.is_empty() {
        return;
    }
    let ks = tv.key_schema();
    let lease_ttl_secs = live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS as i64;
    let pipeline = store_reader(tv).pipeline();
    let _: () = pipeline
        .sadd(ks.presence_members(session_id), cids.to_vec())
        .await
        .expect("queuing the seeded SADD must succeed");
    for cid in cids {
        let _: () = pipeline
            .hset(
                ks.presence_user(session_id, cid),
                [
                    ("user_name", seeded_user_name(cid)),
                    ("user_application", SEEDED_APP.to_owned()),
                ],
            )
            .await
            .expect("queuing a seeded HSET must succeed");
        let _: () = pipeline
            .set(
                ks.presence_lease(session_id, cid),
                "1",
                Some(fred::types::Expiration::EX(lease_ttl_secs)),
                None,
                false,
            )
            .await
            .expect("queuing a seeded lease SET must succeed");
    }
    let _: Vec<fred::types::Value> = pipeline
        .all()
        .await
        .expect("seeding the presence members must succeed");
}

/// Close `svc`'s cleanup tracker, wait (up to [`PRESENCE_WAIT`]) for every join
/// task and lease keeper on it to finish, and check none is left.
pub async fn drain_tracker(svc: &LiveEditService) {
    svc.presence_cleanup_tracker.close();
    tokio::time::timeout(PRESENCE_WAIT, svc.presence_cleanup_tracker.wait())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "the join tasks and lease keepers did not drain within {PRESENCE_WAIT:?}; {} left",
                svc.presence_cleanup_tracker.len()
            )
        });
    assert!(
        svc.presence_cleanup_tracker.is_empty(),
        "no join task or lease keeper may be left running"
    );
}

/// Read `stream` to its end under one [`PRESENCE_WAIT`] deadline: every message
/// it delivered, and its terminal status (`None` for an `OK` end).
pub async fn read_to_end(
    stream: &mut JoinResponseStream,
    what: &str,
) -> (Vec<JoinSessionResponse>, Option<tonic::Status>) {
    read_to_end_within(stream, what, PRESENCE_WAIT).await
}

/// [`read_to_end`] under a `within` deadline.
pub async fn read_to_end_within(
    stream: &mut JoinResponseStream,
    what: &str,
    within: std::time::Duration,
) -> (Vec<JoinSessionResponse>, Option<tonic::Status>) {
    use futures::StreamExt;
    tokio::time::timeout(within, async {
        let mut delivered = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(msg) => delivered.push(msg),
                Err(status) => {
                    assert!(
                        stream.next().await.is_none(),
                        "{what}: an error status must be the stream's terminal item"
                    );
                    return (delivered, Some(status));
                }
            }
        }
        (delivered, None)
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: the join stream did not end within {within:?}"))
}

/// [`read_to_end`] for a stream that must end `OK`: every message it delivered.
pub async fn read_to_ok_end(
    stream: &mut JoinResponseStream,
    what: &str,
) -> Vec<JoinSessionResponse> {
    let (delivered, status) = read_to_end(stream, what).await;
    if let Some(status) = status {
        panic!(
            "{what}: the join stream must end OK, but ended {:?} ({}) after {delivered:?}",
            status.code(),
            status.message()
        );
    }
    delivered
}

// ── JoinSession presence: scenario budget, waits, router feed, store steps ───
//
// Shared by the presence lifecycle tests (`join_lifecycle_test`), the
// replay-duplicate tests (`presence_replay_dedup_test`) and the presence
// sweeper tests (`presence_sweeper_test`).

/// Whole-test deadline for a presence scenario, below nextest's 120 s
/// hang-killer, so a stuck test fails with its own message before the killer
/// would discard it.
pub const SCENARIO_BUDGET: std::time::Duration = std::time::Duration::from_secs(100);

/// Interval between probes of a [`poll_until`] condition.
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

/// Run a test body under [`SCENARIO_BUDGET`].
pub async fn scenario<F: std::future::Future<Output = ()>>(body: F) {
    tokio::time::timeout(SCENARIO_BUDGET, body)
        .await
        .unwrap_or_else(|_| panic!("the scenario did not finish within {SCENARIO_BUDGET:?}"));
}

/// Poll `check` until it holds, failing after `within`.
pub async fn poll_until<F, Fut>(what: &str, within: std::time::Duration, mut check: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + within;
    loop {
        if check().await {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not happen within {within:?}"
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Wait (up to [`PRESENCE_WAIT`]) for an arrival to park at `gate`.
#[cfg(feature = "test-fault-injection")]
pub async fn wait_parked(gate: &live_edit::grpc::presence::PresenceSeamGate, what: &str) {
    tokio::time::timeout(PRESENCE_WAIT, gate.wait_until_parked())
        .await
        .unwrap_or_else(|_| panic!("{what} did not park within {PRESENCE_WAIT:?}"));
}

/// `live_edit_pubsub_relay_lagged_total{relay, service}` on `metrics`: pub/sub
/// messages a replica-wide relay (`presence` or `session_deleted`) skipped.
/// Every cell is seeded, so an absent series is a label bug.
pub fn pubsub_relay_lagged(metrics: &Metrics, service: &str, relay: &str) -> u64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(metrics),
        "live_edit_pubsub_relay_lagged_total",
        &[("relay", relay), ("service", service)],
    )
    .unwrap_or_else(|| panic!("live_edit_pubsub_relay_lagged_total{{relay=\"{relay}\"}} is seeded"))
        as u64
}

/// The session the router feed's readiness probe runs on; no test uses it.
const ROUTER_FEED_WARMUP_SESSION: &str = "router-feed-warmup";

/// Forward the presence channel into `svc`'s local router, as `main.rs` does,
/// so local JoinSession streams receive published events, and return once the
/// feed is forwarding. Cancel the returned token to stop it.
///
/// The feed reads `svc.valkey`'s subscriber client — the one `svc`'s joins
/// `SUBSCRIBE` on — so a service built over a `ValkeyClient` of its own (a
/// second replica) gets a relay of its own.
///
/// `spawn_presence_subscriber` takes the feed's pub/sub receiver before it
/// returns, so the feed sees every event published after that. This still
/// returns only after a probe published on a warm-up session's channel has
/// come through the feed into the router, re-publishing it until one does, so
/// a test that joins at once starts against a feed seen forwarding end to end;
/// the warm-up channel and router entry are then released.
pub async fn spawn_router_feed(
    tv: &TestValkey,
    svc: &LiveEditService,
) -> tokio_util::sync::CancellationToken {
    use fred::interfaces::PubsubInterface;

    let cancel = tokio_util::sync::CancellationToken::new();
    live_edit::presence::spawn_presence_subscriber(
        svc.valkey.subscriber.clone(),
        Arc::clone(&svc.presence_router),
        tv.key_schema().clone(),
        Arc::clone(&svc.metrics),
        cancel.clone(),
    );

    let channel = tv
        .key_schema()
        .presence_events_channel(ROUTER_FEED_WARMUP_SESSION);
    let subscriber = &svc.valkey.subscriber;
    let (mut warmup_rx, _) = svc.presence_router.subscribe(ROUTER_FEED_WARMUP_SESSION);
    subscriber
        .subscribe(&channel)
        .await
        .expect("SUBSCRIBE on the warm-up channel must succeed");
    tokio::time::timeout(PRESENCE_WAIT, async {
        loop {
            publish_probe(tv, ROUTER_FEED_WARMUP_SESSION, "router-feed-ready").await;
            if let Ok(Ok(_)) = tokio::time::timeout(POLL_INTERVAL, warmup_rx.recv()).await {
                return;
            }
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!("the router feed did not forward a warm-up probe within {PRESENCE_WAIT:?}")
    });
    subscriber
        .unsubscribe(&channel)
        .await
        .expect("UNSUBSCRIBE on the warm-up channel must succeed");
    svc.presence_router.unsubscribe(ROUTER_FEED_WARMUP_SESSION);
    cancel
}

/// Register `cid` in the session directly through the store, as a join on
/// another replica would (index add, then register, whose script publishes
/// `joined`), so it is a member no local token or teardown touches. Its lease
/// is the production lease TTL, and nothing renews it.
#[cfg(feature = "test-fault-injection")]
pub async fn register_remote_peer(tv: &TestValkey, session_id: &str, cid: &str) {
    use live_edit::presence::{ConnectedUser, DEFAULT_PRESENCE_LEASE_TTL_SECS};
    use live_edit::storage::client::{FredErrOrigin, POOL_LABEL_DATA};
    use live_edit::storage::presence::{
        add_session_to_index, register_connection, store_time_ms, RegisterOutcome,
    };

    let origin = FredErrOrigin::client(tv.client(), POOL_LABEL_DATA);
    let ks = tv.key_schema();
    add_session_to_index(tv.client(), origin, ks, session_id)
        .await
        .expect("indexing the peer's session must succeed");
    let now_ms = store_time_ms(tv.client(), origin, ks, session_id)
        .await
        .expect("store_time_ms must succeed");
    let joined = PresenceEvent::PeerJoined(ConnectedUser {
        connection_id: cid.to_owned(),
        user_name: "Remote Peer".to_owned(),
        user_application: "remote-app".to_owned(),
    })
    .encode();
    let outcome = register_connection(
        tv.client(),
        origin,
        ks,
        session_id,
        cid,
        "Remote Peer",
        "remote-app",
        DEFAULT_PRESENCE_LEASE_TTL_SECS,
        now_ms + DEFAULT_PRESENCE_LEASE_TTL_SECS * 1000,
        &joined,
    )
    .await
    .expect("registering the remote peer must succeed");
    assert_eq!(outcome, RegisterOutcome::Registered { added: true });
}

/// Evict `cid` the way a sweep does once its lease has lapsed: delete the
/// lease, then run the test-only atomic eviction, which removes the member,
/// deletes the info hash and publishes `left` in one script.
pub async fn evict(tv: &TestValkey, session_id: &str, cid: &str) {
    use fred::interfaces::KeysInterface;
    use live_edit::storage::client::{FredErrOrigin, POOL_LABEL_DATA};

    let deleted: i64 = store_reader(tv)
        .del(tv.key_schema().presence_lease(session_id, cid))
        .await
        .expect("DEL lease must succeed");
    assert_eq!(deleted, 1, "precondition: the live connection has a lease");
    let eviction = live_edit::storage::presence::evict_connection_for_test(
        tv.client(),
        FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
        tv.key_schema(),
        session_id,
        cid,
        &PresenceEvent::PeerLeft {
            connection_id: cid.to_owned(),
        }
        .encode(),
    )
    .await
    .expect("the eviction must succeed");
    assert!(
        eviction.removed,
        "precondition: the eviction removed the member"
    );
}

/// A `PeerJoined` for a made-up connection, published straight onto the
/// channel: proof that a local stream still receives presence events, or a
/// sentinel a stream is read up to.
pub async fn publish_probe(tv: &TestValkey, session_id: &str, probe_cid: &str) {
    let probe = PresenceEvent::PeerJoined(live_edit::presence::ConnectedUser {
        connection_id: probe_cid.to_owned(),
        user_name: "Probe".to_owned(),
        user_application: "probe-app".to_owned(),
    });
    live_edit::presence::publish_presence_event(
        &tv.client().pool,
        tv.key_schema(),
        session_id,
        &probe,
    )
    .await
    .expect("publishing the probe must succeed");
}

/// A `PeerMessage` from a made-up connection, published straight onto the
/// channel: filler a test can have a lagging stream lose without losing a
/// join or leave, which would end the stream for a resync (SDD §13.2).
#[cfg(feature = "test-fault-injection")]
pub async fn publish_message_probe(tv: &TestValkey, session_id: &str, probe_cid: &str) {
    let probe = PresenceEvent::PeerMessage {
        connection_id: probe_cid.to_owned(),
        data: b"probe".to_vec(),
    };
    live_edit::presence::publish_presence_event(
        &tv.client().pool,
        tv.key_schema(),
        session_id,
        &probe,
    )
    .await
    .expect("publishing the message probe must succeed");
}

/// Read `stream` (up to [`PRESENCE_WAIT`]) until it delivers a `PeerJoined`
/// for `cid`.
pub async fn expect_peer_joined(stream: &mut JoinResponseStream, cid: &str, what: &str) {
    use futures::StreamExt;
    tokio::time::timeout(PRESENCE_WAIT, async {
        while let Some(item) = stream.next().await {
            let msg = item.unwrap_or_else(|s| panic!("{what}: the join stream failed: {s:?}"));
            if let Some(join_session_response::ResponseType::PeerJoined(p)) = msg.response_type {
                if p.connection_id == cid {
                    return;
                }
            }
        }
        panic!("{what}: the join stream ended before PeerJoined for {cid}");
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: no PeerJoined for {cid} within {PRESENCE_WAIT:?}"));
}

// ── External presence observers ──────────────────────────────────────────────

/// The pub/sub message channel capacity of every test observer's subscriber
/// client. fred's default (32) lags under a burst — a sweep chunk publishes up
/// to `SWEEP_MEMBER_BATCH` (100) `left`s in one script — and a lagged observer
/// would lose events silently, reading as an ordering or delivery bug in the
/// code under test.
pub const OBSERVER_CHANNEL_CAPACITY: usize = 4096;

/// A RESP2 subscriber client on `url`, with a pub/sub message channel of
/// [`OBSERVER_CHANNEL_CAPACITY`], connected.
async fn observer_subscriber_client(url: &str) -> fred::clients::SubscriberClient {
    use fred::interfaces::ClientLike;
    use fred::types::config::PerformanceConfig;
    use fred::types::{Builder, RespVersion};

    let config = fred::types::config::Config::from_url(url)
        .unwrap_or_else(|e| panic!("{url} must parse as a fred config: {e:?}"));
    let mut builder = Builder::from_config(config);
    builder
        .with_config(|c| c.version = RespVersion::RESP2)
        .set_performance_config(PerformanceConfig {
            broadcast_channel_capacity: OBSERVER_CHANNEL_CAPACITY,
            ..Default::default()
        });
    let subscriber = builder
        .build_subscriber_client()
        .expect("subscriber client build must succeed");
    subscriber.init().await.expect("subscriber must connect");
    subscriber
}

/// Subscribe `subscriber` to `channel` and forward the decoded presence events
/// published on it. The forwarder panics if the subscriber client's message
/// channel lags — an observer that skipped messages cannot vouch for what was
/// published — and ends when the client's channel closes or the receiver is
/// dropped. A panic ends the forwarder, so the receiver then reads `None`.
async fn forward_presence_events(
    subscriber: &fred::clients::SubscriberClient,
    channel: &str,
) -> (
    tokio::sync::mpsc::UnboundedReceiver<PresenceEvent>,
    tokio::task::JoinHandle<()>,
) {
    use fred::interfaces::{EventInterface, PubsubInterface};
    use tokio::sync::broadcast::error::RecvError;

    let mut messages = subscriber.message_rx();
    subscriber
        .subscribe(channel)
        .await
        .expect("SUBSCRIBE on presence-events channel must succeed");

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<PresenceEvent>();
    let wanted = channel.to_owned();
    let handle = tokio::spawn(async move {
        loop {
            let msg = match messages.recv().await {
                Ok(msg) => msg,
                Err(RecvError::Lagged(n)) => panic!(
                    "the test observer of {wanted} lagged and lost {n} messages; raise \
                     OBSERVER_CHANNEL_CAPACITY ({OBSERVER_CHANNEL_CAPACITY})"
                ),
                Err(RecvError::Closed) => break,
            };
            if msg.channel.as_bytes() != wanted.as_bytes() {
                continue;
            }
            let Ok(payload) = msg.value.convert::<String>() else {
                continue;
            };
            if let Ok(event) = PresenceEvent::parse(&payload) {
                if tx.send(event).is_err() {
                    break;
                }
            }
        }
    });
    (rx, handle)
}

/// Subscribe a fresh `SubscriberClient` to one Valkey pub/sub channel and
/// return a futures `Receiver` of decoded `PresenceEvent`s. The caller is
/// responsible for keeping the returned task `JoinHandle` alive until the
/// observation deadline; dropping it cancels the subscriber loop. This is
/// the minimal "another replica observes the channel" rig used to assert
/// that PeerLeft actually lands on the wire.
///
/// The client's message channel holds [`OBSERVER_CHANNEL_CAPACITY`] messages,
/// and the forwarding task panics rather than skip any (see
/// [`forward_presence_events`]).
pub async fn spawn_external_channel_subscriber(
    tv: &TestValkey,
    session_id: &str,
) -> (
    tokio::sync::mpsc::UnboundedReceiver<PresenceEvent>,
    tokio::task::JoinHandle<()>,
    fred::clients::SubscriberClient,
) {
    // Build an independent subscriber connection — production wiring would
    // use the second replica's own `ValkeyClient::subscriber`, but to keep
    // this test single-process we just open one more subscriber against
    // the same Valkey.
    let url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let subscriber = observer_subscriber_client(&url).await;
    let channel = tv.key_schema().presence_events_channel(session_id);
    let (rx, handle) = forward_presence_events(&subscriber, &channel).await;
    (rx, handle, subscriber)
}

/// Subscribe a standalone connection to one cluster node and forward the
/// decoded presence events published on `channel`. Cluster pub/sub fans every
/// PUBLISH out to every node, so this sees the channel as a peer attached to
/// `node` would. Same channel capacity and lag handling as
/// [`spawn_external_channel_subscriber`].
pub async fn subscribe_on_node(
    node: &fred::types::config::Server,
    channel: &str,
) -> (
    tokio::sync::mpsc::UnboundedReceiver<PresenceEvent>,
    tokio::task::JoinHandle<()>,
    fred::clients::SubscriberClient,
) {
    let subscriber =
        observer_subscriber_client(&format!("redis://{}:{}", node.host, node.port)).await;
    let (rx, handle) = forward_presence_events(&subscriber, channel).await;
    (rx, handle, subscriber)
}

/// An independent subscriber on one session's presence-events channel: a peer
/// on another replica. It uses its own connection, never the service's
/// subscriber, which teardown unsubscribes.
pub struct Tap {
    session_id: String,
    channel: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<PresenceEvent>,
    task: tokio::task::JoinHandle<()>,
    subscriber: fred::clients::SubscriberClient,
    fences: u64,
}

impl Tap {
    /// A tap on the store `tv` is connected to.
    pub async fn open(tv: &TestValkey, session_id: &str) -> Self {
        let (rx, task, subscriber) = spawn_external_channel_subscriber(tv, session_id).await;
        Self::new(tv, session_id, rx, task, subscriber)
    }

    /// A tap subscribed on one cluster `node`, as a peer attached to that node
    /// would see the channel. Compiled with `test-fault-injection` only: its
    /// callers drive the sweeper seam.
    #[cfg(feature = "test-fault-injection")]
    pub async fn open_on_node(
        tv: &TestValkey,
        session_id: &str,
        node: &fred::types::config::Server,
    ) -> Self {
        let channel = tv.key_schema().presence_events_channel(session_id);
        let (rx, task, subscriber) = subscribe_on_node(node, &channel).await;
        Self::new(tv, session_id, rx, task, subscriber)
    }

    fn new(
        tv: &TestValkey,
        session_id: &str,
        rx: tokio::sync::mpsc::UnboundedReceiver<PresenceEvent>,
        task: tokio::task::JoinHandle<()>,
        subscriber: fred::clients::SubscriberClient,
    ) -> Self {
        Tap {
            session_id: session_id.to_owned(),
            channel: tv.key_schema().presence_events_channel(session_id),
            rx,
            task,
            subscriber,
            fences: 0,
        }
    }

    /// Every event published on the channel since the last read, proven
    /// complete by a fence: a `PeerMessage` published after everything under
    /// test, read up to. A single node delivers one channel's messages to one
    /// subscriber in the order it executed the PUBLISHes. On a cluster the
    /// fence is published from the session's `{sid}` node, as every presence
    /// script's publish is, so it orders behind them only if every event under
    /// test came from that node: a cluster test waits for the events it
    /// expects ([`Self::until`]) before it drains.
    pub async fn drain(&mut self, tv: &TestValkey) -> Vec<PresenceEvent> {
        self.fences += 1;
        let fence = format!("fence-{}", self.fences);
        live_edit::presence::publish_presence_event(
            &tv.client().pool,
            tv.key_schema(),
            &self.session_id,
            &PresenceEvent::PeerMessage {
                connection_id: fence.clone(),
                data: Vec::new(),
            },
        )
        .await
        .expect("publishing the fence must succeed");
        let rx = &mut self.rx;
        tokio::time::timeout(PRESENCE_WAIT, async move {
            let mut seen = Vec::new();
            loop {
                match rx.recv().await {
                    Some(PresenceEvent::PeerMessage { connection_id, .. })
                        if connection_id == fence =>
                    {
                        return seen;
                    }
                    Some(event) => seen.push(event),
                    None => panic!("the tap's forwarding task ended"),
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the fence did not arrive within {PRESENCE_WAIT:?}"))
    }

    /// Every event up to and including the first that matches `wanted`.
    pub async fn until(
        &mut self,
        what: &str,
        wanted: impl Fn(&PresenceEvent) -> bool,
    ) -> Vec<PresenceEvent> {
        let rx = &mut self.rx;
        tokio::time::timeout(PRESENCE_WAIT, async move {
            let mut seen = Vec::new();
            loop {
                let event = rx.recv().await.expect("the tap's forwarding task ended");
                let done = wanted(&event);
                seen.push(event);
                if done {
                    return seen;
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{what} did not reach the tap within {PRESENCE_WAIT:?}"))
    }

    /// Every event up to and including the `n`-th that matches `wanted`,
    /// under one [`PRESENCE_WAIT`] deadline. Compiled with
    /// `test-fault-injection` only: its callers drive the sweeper seam.
    #[cfg(feature = "test-fault-injection")]
    pub async fn until_count(
        &mut self,
        what: &str,
        n: usize,
        wanted: impl Fn(&PresenceEvent) -> bool,
    ) -> Vec<PresenceEvent> {
        let rx = &mut self.rx;
        let mut seen = Vec::new();
        let mut matched = 0;
        tokio::time::timeout(PRESENCE_WAIT, async {
            while matched < n {
                let event = rx.recv().await.expect("the tap's forwarding task ended");
                if wanted(&event) {
                    matched += 1;
                }
                seen.push(event);
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!("only {matched} of {n} {what} reached the tap within {PRESENCE_WAIT:?}")
        });
        seen
    }

    pub async fn close(self) {
        use fred::interfaces::{ClientLike, PubsubInterface};
        let _ = self.subscriber.unsubscribe(self.channel).await;
        let _ = self.subscriber.quit().await;
        let _ = tokio::time::timeout(PRESENCE_WAIT, self.task).await;
    }
}

pub fn is_joined(event: &PresenceEvent, cid: &str) -> bool {
    matches!(event, PresenceEvent::PeerJoined(user) if user.connection_id == cid)
}

pub fn is_left(event: &PresenceEvent, cid: &str) -> bool {
    matches!(event, PresenceEvent::PeerLeft { connection_id } if connection_id == cid)
}

pub fn is_message(event: &PresenceEvent, cid: &str) -> bool {
    matches!(event, PresenceEvent::PeerMessage { connection_id, .. } if connection_id == cid)
}

/// Poll the receiver for a `PeerLeft` matching `connection_id` until the
/// deadline elapses. Returns `true` on hit, `false` on timeout. Other
/// events are drained-and-ignored.
pub async fn wait_for_peer_left(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<PresenceEvent>,
    connection_id: &str,
    deadline: std::time::Duration,
) -> bool {
    let result = tokio::time::timeout(deadline, async {
        while let Some(ev) = rx.recv().await {
            if let PresenceEvent::PeerLeft { connection_id: cid } = ev {
                if cid == connection_id {
                    return true;
                }
            }
        }
        false
    })
    .await;
    matches!(result, Ok(true))
}

// ── Composite helpers ────────────────────────────────────────────────────────

/// Create a session and layer via the CreateLayer handler, returning
/// (session_id, object_id).
pub async fn create_session_and_layer(
    tv: &TestValkey,
    svc: &LiveEditService,
    session_name: &str,
    layer_name: &str,
) -> (String, u64) {
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create(session_name, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let ompf_bytes = crate::common::owned_delta::delta_to_ompf_bytes(
        &crate::common::owned_delta::empty_delta(),
        &[],
    );

    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf_bytes),
    ]);
    let request = make_request(stream);

    let response = handle_create_layer(svc, request)
        .await
        .expect("create_layer must succeed for test setup");

    let object_id = response.into_inner().object_id;
    (session_id, object_id)
}

/// A graceful shutdown whose paced stream ending has started and is held
/// open: `begun` is cancelled — so new streaming RPCs are refused and waiting
/// uploads end — while the service shutdown token (the parent) stays
/// uncancelled for about 20 s, so a refusal the test observes in that window
/// came from `begun` or a born-cancelled enrollment, never from the parent.
/// Dropping it cancels the handle, ending the held-open window.
pub struct PacedEndingHeldOpen {
    shutdown: live_edit::grpc::graceful_shutdown::GracefulShutdown,
    /// [`PACED_ENDING_HOLDERS`] placeholder streams, enrolled before the start
    /// so the window is long: at 1/s they stretch the paced ending to its
    /// 20 s cap.
    _holders: Vec<live_edit::grpc::graceful_shutdown::StreamEnrollment>,
}

/// Placeholder streams [`begin_paced_ending_held_open`] enrolls.
pub const PACED_ENDING_HOLDERS: usize = 20;

impl Drop for PacedEndingHeldOpen {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

/// Start `shutdown`'s paced stream ending as `main.rs` does once the drain
/// ends — [`end_streams_after_drain`](live_edit::grpc::transport::end_streams_after_drain)
/// with an already-resolved drain — at 1 stream per second over at most 20 s,
/// and return once `begun` has fired (t0, `STREAM_END_DELAY_AFTER_GOAWAY`
/// later). Streams the test enrolled before the call are in the snapshot, in
/// random order among [`PACED_ENDING_HOLDERS`] placeholders, one ending
/// about every second from t0 + 0.9 s; the parent is cancelled about 20 s
/// after t0.
pub async fn begin_paced_ending_held_open(
    shutdown: &live_edit::grpc::graceful_shutdown::GracefulShutdown,
) -> PacedEndingHeldOpen {
    use live_edit::grpc::graceful_shutdown::StreamPacing;
    use rand::SeedableRng;
    let holders = (0..PACED_ENDING_HOLDERS)
        .map(|_| shutdown.enroll_stream().1)
        .collect();
    live_edit::grpc::transport::end_streams_after_drain(
        async {},
        shutdown.clone(),
        StreamPacing {
            rate_per_sec: 1,
            max_window: std::time::Duration::from_secs(20),
        },
        rand::rngs::StdRng::seed_from_u64(0),
    )
    .await;
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        shutdown.begun().cancelled(),
    )
    .await
    .expect("the paced ending starts the GOAWAY head start after the drain");
    assert!(
        !shutdown.is_cancelled(),
        "fixture: the service shutdown token stays uncancelled while the paced ending is held \
         open"
    );
    PacedEndingHeldOpen {
        shutdown: shutdown.clone(),
        _holders: holders,
    }
}
