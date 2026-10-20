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

//! Integration tests for `build_permissions_client` / `build_userinfo_client`.
//!
//! Exercises four combinations for the permissions helper (REST/gRPC × authz
//! enabled/disabled), both protocol branches for the userinfo helper, and the
//! userinfo helper's authorization gate on the not-found fallback, using
//! `wiremock` for REST and an inline tonic mock server for gRPC.
//!
//! The tonic mocks here are freshly written rather than extracted from
//! `src/auth/mod.rs::grpc_tests` because the existing mocks live behind
//! `#[cfg(test)]` and are not reachable from an integration-test binary. The
//! shapes are minimal — only the two RPCs under test (`Stat` and
//! `GetCurrentUser`) return meaningful values; every other RPC is
//! `unimplemented!()`.
//!
//! All tests assert behavior (not the concrete type of the returned trait
//! object), so the no-op branch is verified by observing that
//! `check_permission` returns `Allowed` without the mock receiving any
//! requests.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tonic::{Request, Response, Status};
use wiremock::matchers::{any, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use live_edit::auth::storage_service_proto::file_object_service_server::{
    FileObjectService, FileObjectServiceServer,
};
use live_edit::auth::storage_service_proto::*;
use live_edit::auth::userinfo_proto::user_info_service_server::{
    UserInfoService, UserInfoServiceServer,
};
use live_edit::auth::userinfo_proto::*;
use live_edit::auth::{
    build_permissions_client, build_userinfo_client, fast_retry, AuthError, AuthResult, Permission,
};
use live_edit::config::{AuthProtocol, Config};

// ── Config test-builder ──────────────────────────────────────────────────────

/// Build a `Config` suitable for exercising the auth builders.
///
/// b.ohv item 8: delegates to `common::test_config()` and overrides only the
/// auth-relevant fields the builders look at (`auth_protocol`,
/// `authz_enabled`, `userinfo_service_enabled`, `storage_service_endpoint`,
/// `userinfo_endpoint`, `auth_retry_initial_delay_ms`). The builders ignore
/// every other field, so taking the common defaults keeps the test surface
/// minimal.
///
/// `userinfo_endpoint` is taken as a raw `String` and wrapped in `Some(...)`
/// internally — this keeps the call-site diff for b.g3z mechanical (the
/// `String` parameter already matched the prior `Config::userinfo_endpoint`
/// type). Pass `userinfo_service_enabled = true` at all existing call sites
/// to preserve the pre-b.g3z "external userinfo backend assumed" semantics
/// that these REST/gRPC `build_userinfo_client` dispatch tests rely on —
/// they exercise the userinfo backend, NOT the new JWT-extraction path.
fn test_config(
    auth_protocol: AuthProtocol,
    authz_enabled: bool,
    userinfo_service_enabled: bool,
    storage_endpoint: Option<String>,
    userinfo_endpoint: String,
) -> Config {
    let mut cfg = crate::common::test_config();
    cfg.auth_protocol = auth_protocol;
    cfg.authz_enabled = authz_enabled;
    cfg.userinfo_service_enabled = userinfo_service_enabled;
    cfg.storage_service_endpoint = storage_endpoint;
    cfg.userinfo_endpoint = Some(userinfo_endpoint);
    // The original helper used a sub-millisecond retry initial delay to keep
    // the fast-retry tests snappy; preserve that here.
    cfg.auth_retry_initial_delay_ms = 1;
    cfg
}

// ── tonic mock: FileObjectService (minimal) ──────────────────────────────────

#[derive(Clone, Default)]
struct MockStorageService {
    stat_calls: Arc<Mutex<u32>>,
}

#[tonic::async_trait]
impl FileObjectService for MockStorageService {
    type EnumerateStream = tokio_stream::Empty<Result<EnumerateResponse, Status>>;

    async fn enumerate(
        &self,
        _req: Request<EnumerateRequest>,
    ) -> Result<Response<Self::EnumerateStream>, Status> {
        unimplemented!()
    }

    async fn stat(&self, _req: Request<StatRequest>) -> Result<Response<StatResponse>, Status> {
        *self.stat_calls.lock().unwrap() += 1;
        Ok(Response::new(StatResponse {
            resource_info: None,
        }))
    }

    type ReadStream = tokio_stream::Empty<Result<ReadResponse, Status>>;

    async fn read(&self, _req: Request<ReadRequest>) -> Result<Response<Self::ReadStream>, Status> {
        unimplemented!()
    }

    type ReadFromAddressStream = tokio_stream::Empty<Result<ReadFromAddressResponse, Status>>;

    async fn read_from_address(
        &self,
        _req: Request<ReadFromAddressRequest>,
    ) -> Result<Response<Self::ReadFromAddressStream>, Status> {
        unimplemented!()
    }

    async fn fetch_write_type_info(
        &self,
        _req: Request<FetchWriteTypeInfoRequest>,
    ) -> Result<Response<FetchWriteTypeInfoResponse>, Status> {
        unimplemented!()
    }

    type WriteStream = tokio_stream::Empty<Result<WriteResponse, Status>>;

    async fn write(
        &self,
        _req: Request<tonic::Streaming<WriteRequest>>,
    ) -> Result<Response<Self::WriteStream>, Status> {
        unimplemented!()
    }

    async fn complete_redirect_upload(
        &self,
        _req: Request<CompleteRedirectUploadRequest>,
    ) -> Result<Response<CompleteRedirectUploadResponse>, Status> {
        unimplemented!()
    }

    async fn upload_part(
        &self,
        _req: Request<UploadPartRequest>,
    ) -> Result<Response<UploadPartResponse>, Status> {
        unimplemented!()
    }

    async fn complete_multipart_upload(
        &self,
        _req: Request<CompleteMultipartUploadRequest>,
    ) -> Result<Response<CompleteMultipartUploadResponse>, Status> {
        unimplemented!()
    }

    async fn abort_multipart_upload(
        &self,
        _req: Request<AbortMultipartUploadRequest>,
    ) -> Result<Response<AbortMultipartUploadResponse>, Status> {
        unimplemented!()
    }

    async fn delete(
        &self,
        _req: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        unimplemented!()
    }
}

/// Start a mock gRPC storage-service server on a random port; returns its
/// `http://HOST:PORT` URI and a call-counter handle.
///
/// Listener-ready handshake (b.ohv item 5): the spawned task fires a
/// oneshot once it has the `TcpListenerStream` wired into the tonic
/// `Server::serve_with_incoming` future. This replaces the old TCP-connect
/// polling loop with 10 ms cadence — same correctness, deterministic
/// wakeup, no observed-from-outside wall-clock dependence.
async fn start_grpc_storage_mock() -> (String, Arc<Mutex<u32>>) {
    let mock = MockStorageService::default();
    let counter = Arc::clone(&mock.stat_calls);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let _ = ready_tx.send(());
        tonic::transport::Server::builder()
            .add_service(FileObjectServiceServer::new(mock))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });
    let _ = tokio::time::timeout(Duration::from_secs(5), ready_rx)
        .await
        .expect("storage mock server did not signal readiness within 5s");
    (format!("http://{addr}"), counter)
}

// ── tonic mock: UserInfoService (minimal) ────────────────────────────────────

#[derive(Clone)]
struct MockUserInfoService {
    user: User,
    calls: Arc<Mutex<u32>>,
}

impl MockUserInfoService {
    fn new(user: User) -> Self {
        Self {
            user,
            calls: Arc::new(Mutex::new(0)),
        }
    }
}

#[tonic::async_trait]
impl UserInfoService for MockUserInfoService {
    async fn get_current_user(
        &self,
        _req: Request<GetCurrentUserRequest>,
    ) -> Result<Response<GetCurrentUserResponse>, Status> {
        *self.calls.lock().unwrap() += 1;
        Ok(Response::new(GetCurrentUserResponse {
            user: Some(self.user.clone()),
        }))
    }

    async fn list_users(
        &self,
        _req: Request<ListUsersRequest>,
    ) -> Result<Response<ListUsersResponse>, Status> {
        unimplemented!()
    }

    async fn get_user(
        &self,
        _req: Request<GetUserRequest>,
    ) -> Result<Response<GetUserResponse>, Status> {
        unimplemented!()
    }

    async fn list_groups(
        &self,
        _req: Request<ListGroupsRequest>,
    ) -> Result<Response<ListGroupsResponse>, Status> {
        unimplemented!()
    }

    async fn get_group(
        &self,
        _req: Request<GetGroupRequest>,
    ) -> Result<Response<GetGroupResponse>, Status> {
        unimplemented!()
    }

    async fn list_group_members(
        &self,
        _req: Request<ListGroupMembersRequest>,
    ) -> Result<Response<ListGroupMembersResponse>, Status> {
        unimplemented!()
    }

    async fn list_user_groups(
        &self,
        _req: Request<ListUserGroupsRequest>,
    ) -> Result<Response<ListUserGroupsResponse>, Status> {
        unimplemented!()
    }

    async fn get_user_group(
        &self,
        _req: Request<GetUserGroupRequest>,
    ) -> Result<Response<GetUserGroupResponse>, Status> {
        unimplemented!()
    }

    async fn get_group_member(
        &self,
        _req: Request<GetGroupMemberRequest>,
    ) -> Result<Response<GetGroupMemberResponse>, Status> {
        unimplemented!()
    }
}

fn sample_user() -> User {
    User {
        id: "user-42".into(),
        display_name: "Jane Doe".into(),
        given_name: Some("Jane".into()),
        family_name: Some("Doe".into()),
        email: Some("jane@example.com".into()),
        username: "jdoe".into(),
        active: Some(true),
        job_title: Some("Engineer".into()),
    }
}

async fn start_grpc_userinfo_mock() -> (String, Arc<Mutex<u32>>) {
    let mock = MockUserInfoService::new(sample_user());
    let counter = Arc::clone(&mock.calls);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let _ = ready_tx.send(());
        tonic::transport::Server::builder()
            .add_service(UserInfoServiceServer::new(mock))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });
    let _ = tokio::time::timeout(Duration::from_secs(5), ready_rx)
        .await
        .expect("userinfo mock server did not signal readiness within 5s");
    (format!("http://{addr}"), counter)
}

// ── permissions: REST + authz enabled ────────────────────────────────────────

#[tokio::test]
async fn build_permissions_client_rest_authz_enabled() {
    let mock_server = MockServer::start().await;
    // The REST storage client stats the address:
    // `HEAD /v1alpha/fileobject/by-address/{path}`, no query string (b.uoc),
    // with the caller-supplied path fully percent-encoded as an opaque
    // resource address — so `/p` becomes `%2Fp`. `204 No Content` is the
    // faithful stat answer to a HEAD; both 200 and 204 mean `Allowed`.
    Mock::given(method("HEAD"))
        .and(path("/v1alpha/fileobject/by-address/%2Fp"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config(
        AuthProtocol::Rest,
        true,
        true,
        Some(mock_server.uri()),
        "http://userinfo.example.com".to_string(),
    );
    let client = build_permissions_client(&config, &fast_retry(3))
        .await
        .expect("rest+authz builder should succeed");

    let result = client
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();
    assert_eq!(result, AuthResult::Allowed);
    // wiremock's `.expect(1)` verifies on drop that the mock was hit once.
}

// ── permissions: gRPC + authz enabled ────────────────────────────────────────

#[tokio::test]
async fn build_permissions_client_grpc_authz_enabled() {
    let (endpoint, counter) = start_grpc_storage_mock().await;

    let config = test_config(
        AuthProtocol::Grpc,
        true,
        true,
        Some(endpoint),
        "http://userinfo.example.com".to_string(),
    );
    let client = build_permissions_client(&config, &fast_retry(3))
        .await
        .expect("grpc+authz builder should succeed");

    let result = client
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();
    assert_eq!(result, AuthResult::Allowed);
    assert_eq!(*counter.lock().unwrap(), 1);
}

// ── permissions: REST + authz disabled → no-op (no HTTP call) ────────────────

#[tokio::test]
async fn build_permissions_client_rest_authz_disabled() {
    let mock_server = MockServer::start().await;
    // Mount a mock that MUST NOT be hit — `.expect(0)` asserts on drop.
    // Matched with `any()` rather than a specific verb so the assertion stays
    // "the no-op client makes no HTTP call at all": a method-pinned matcher
    // would silently stop guarding anything the day the probe's verb changes
    // (exactly what happened when b.uoc moved the probe from GET to HEAD).
    Mock::given(any())
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&mock_server)
        .await;

    let config = test_config(
        AuthProtocol::Rest,
        false,
        true,
        Some(mock_server.uri()),
        "http://userinfo.example.com".to_string(),
    );
    let client = build_permissions_client(&config, &fast_retry(1))
        .await
        .expect("rest+!authz should return the no-op client");

    // No-op returns `Allowed` for both permissions without any network call.
    let read = client
        .check_permission("tok", "/any/path", Permission::Read)
        .await
        .unwrap();
    let write = client
        .check_permission("tok", "/any/path", Permission::Write)
        .await
        .unwrap();
    assert_eq!(read, AuthResult::Allowed);
    assert_eq!(write, AuthResult::Allowed);
    // `.expect(0)` verifies on drop that the mock was not contacted.
}

// ── permissions: gRPC + authz disabled → no-op (no connect) ──────────────────

#[tokio::test]
async fn build_permissions_client_grpc_authz_disabled() {
    // Storage endpoint points at a closed port. If the helper ignored
    // `authz_enabled = false` and tried the gRPC branch, `.connect().await`
    // would fail. The no-op branch must short-circuit before any connect.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed_addr = listener.local_addr().unwrap();
    drop(listener);

    let config = test_config(
        AuthProtocol::Grpc,
        false,
        true,
        Some(format!("http://{closed_addr}")),
        "http://userinfo.example.com".to_string(),
    );
    let client = build_permissions_client(&config, &fast_retry(1))
        .await
        .expect("grpc+!authz should return the no-op client, not connect");

    let result = client
        .check_permission("tok", "/any/path", Permission::Read)
        .await
        .unwrap();
    assert_eq!(result, AuthResult::Allowed);
}

// ── userinfo: REST ───────────────────────────────────────────────────────────

#[tokio::test]
async fn build_userinfo_client_rest_succeeds() {
    let mock_server = MockServer::start().await;
    let body = r#"{"id":"u1","display_name":"Jane","email":"jane@x.com","username":"jane"}"#;
    Mock::given(method("GET"))
        .and(path("/api/v1alpha/users/me"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body.as_bytes().to_vec(), "application/json"),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = test_config(AuthProtocol::Rest, false, true, None, mock_server.uri());
    let client = build_userinfo_client(
        &config,
        &fast_retry(3),
        Arc::new(crate::common::metrics_scrape_only("test")),
    )
    .await
    .expect("rest userinfo builder should succeed");

    let info = client.get_current_user("tok").await.unwrap();
    assert_eq!(info.id, "u1");
    assert_eq!(info.display_name, "Jane");
    assert_eq!(info.email, Some("jane@x.com".to_owned()));
    assert_eq!(info.username, "jane");
}

// ── userinfo: gRPC ───────────────────────────────────────────────────────────

#[tokio::test]
async fn build_userinfo_client_grpc_succeeds() {
    let (endpoint, counter) = start_grpc_userinfo_mock().await;

    let config = test_config(AuthProtocol::Grpc, false, true, None, endpoint);
    let client = build_userinfo_client(
        &config,
        &fast_retry(3),
        Arc::new(crate::common::metrics_scrape_only("test")),
    )
    .await
    .expect("grpc userinfo builder should succeed");

    let info = client.get_current_user("tok").await.unwrap();
    assert_eq!(info.id, "user-42");
    assert_eq!(info.display_name, "Jane Doe");
    assert_eq!(info.email, Some("jane@example.com".to_owned()));
    assert_eq!(info.username, "jdoe");
    assert_eq!(*counter.lock().unwrap(), 1);
}

// ── userinfo: JWT dispatch (userinfo service disabled) ───────────────────────

#[tokio::test]
async fn build_userinfo_client_userinfo_service_disabled_returns_jwt_client() {
    // `auth_protocol = AuthProtocol::Rest` is arbitrary — the JWT branch at
    // the top of `build_userinfo_client` short-circuits ahead of the protocol
    // match (SR-3.1), so the value never reaches the REST/gRPC dispatch.
    // `userinfo_endpoint = None` matches the b.g3z `Option<String>` shape: the
    // builder must not require an endpoint when the userinfo service is off.
    // The `test_config` helper unconditionally wraps the userinfo endpoint in
    // `Some(...)`, so we override the field to `None` after construction to
    // exercise the actual production invariant.
    let mut config = test_config(AuthProtocol::Rest, false, false, None, String::new());
    config.userinfo_endpoint = None;

    let client = build_userinfo_client(
        &config,
        &fast_retry(1),
        Arc::new(crate::common::metrics_scrape_only("test")),
    )
    .await
    .expect("disabled userinfo service should construct JwtUserinfoClient in-process");

    // Behavior assertion 1: opaque (non-JWT) bearer surfaces the SR-2.4
    // verbatim error string. Pinned by `src/auth/userinfo_jwt.rs`; the
    // shared `crate::common::SR_2_4_OPAQUE_TOKEN_MSG` constant is the
    // single source of truth across both test binaries, so a reword in
    // production turns into a compile-time mismatch in one place rather
    // than a silent no-op here.
    let err = client
        .get_current_user("opaque-not-a-jwt")
        .await
        .expect_err("opaque bearer must be rejected by the JWT client");
    assert!(
        matches!(err, AuthError::InvalidResponse(ref msg) if msg == crate::common::SR_2_4_OPAQUE_TOKEN_MSG),
        "expected SR-2.4 opaque-token InvalidResponse, got {err:?}",
    );

    // Behavior assertion 2: a well-formed 3-segment JWT carrying a `name`
    // claim decodes in-process and the decoded claim flows out as
    // `UserInfo.display_name`. Together with assertion 1 this proves the
    // returned trait object is a `JwtUserinfoClient` without naming the
    // concrete type, matching the file's module-doc-comment convention.
    let jwt = crate::common::make_jwt(&serde_json::json!({"name": "Alice Example"}));
    let info = client
        .get_current_user(&jwt)
        .await
        .expect("name-only JWT should decode via the in-process JWT client");
    assert_eq!(info.display_name, "Alice Example");
}

// ── userinfo: not-found fallback gated on authorization ──────────────────────
//
// The userinfo service's "no such user" answer (REST 404 + `not_found` body) is
// answered from the bearer's JWT claims only when authorization is on — the
// claim decode trusts the bearer only once the permissions service validated
// it. With authorization off the builder returns the bare backend and the
// answer stays an error.

/// Mount the userinfo service's not-found answer on `GET /api/v1alpha/users/me`,
/// expected exactly once.
async fn mount_userinfo_not_found(mock_server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/api/v1alpha/users/me"))
        .respond_with(ResponseTemplate::new(404).set_body_raw(
            br#"{"error":"not_found","message":"not found"}"#.to_vec(),
            "application/json",
        ))
        .expect(1)
        .mount(mock_server)
        .await;
}

/// An Entra app-only access token for an app the directory does not hold.
fn app_only_jwt() -> String {
    crate::common::make_jwt(&serde_json::json!({
        "idtyp": "app",
        "app_displayname": "Asset Agent",
        "oid": "11111111-2222-3333-4444-555555555555",
        "sub": "11111111-2222-3333-4444-555555555555",
    }))
}

fn fallbacks(metrics: &live_edit::observability::metrics::Metrics) -> Option<f64> {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(&render(metrics), "live_edit_userinfo_fallbacks_total", &[])
}

#[tokio::test]
async fn build_userinfo_client_rest_authz_enabled_falls_back_to_jwt_label_on_user_not_found() {
    let mock_server = MockServer::start().await;
    mount_userinfo_not_found(&mock_server).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));

    let config = test_config(AuthProtocol::Rest, true, true, None, mock_server.uri());
    let client = build_userinfo_client(&config, &fast_retry(3), Arc::clone(&metrics))
        .await
        .expect("rest userinfo builder should succeed");

    let info = client
        .get_current_user(&app_only_jwt())
        .await
        .expect("with authorization on, a not-found answer falls back to the JWT claims");
    assert_eq!(info.display_name, "Asset Agent (app)");
    assert_eq!(info.id, "11111111-2222-3333-4444-555555555555");
    assert_eq!(
        fallbacks(&metrics),
        Some(1.0),
        "the builder must hand the decorator the metrics it was given"
    );
    // `.expect(1)` verifies on drop that the userinfo service was asked once.
}

#[tokio::test]
async fn build_userinfo_client_rest_authz_disabled_keeps_user_not_found() {
    let mock_server = MockServer::start().await;
    mount_userinfo_not_found(&mock_server).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));

    let config = test_config(AuthProtocol::Rest, false, true, None, mock_server.uri());
    let client = build_userinfo_client(&config, &fast_retry(3), Arc::clone(&metrics))
        .await
        .expect("rest userinfo builder should succeed");

    let err = client
        .get_current_user(&app_only_jwt())
        .await
        .expect_err("with authorization off nothing validated the bearer: no fallback");
    assert!(
        matches!(err, AuthError::UserNotFound),
        "expected UserNotFound, got {err:?}"
    );
    assert_eq!(fallbacks(&metrics), Some(0.0));
}

// ── error path: gRPC permissions builder connect failure ─────────────────────

#[tokio::test]
async fn build_permissions_client_grpc_connect_failure() {
    // Bind then drop a listener to obtain a port that refuses connections.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed_addr = listener.local_addr().unwrap();
    drop(listener);

    let config = test_config(
        AuthProtocol::Grpc,
        true, // authz ON → the gRPC branch actually runs
        true, // userinfo service ON (preserves pre-b.g3z semantics)
        Some(format!("http://{closed_addr}")),
        "http://userinfo.example.com".to_string(),
    );
    let err = match build_permissions_client(&config, &fast_retry(1)).await {
        Ok(_) => panic!("grpc+authz pointed at closed port should fail"),
        Err(e) => e,
    };

    assert!(
        matches!(err, AuthError::Unreachable),
        "expected Unreachable, got {err:?}"
    );
}
