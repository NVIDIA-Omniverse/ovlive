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

// Integration tests for DeleteLayer and StatLayer handlers.
//
// Tests call `handle_delete_layer` and `handle_stat_layer` directly with a
// LiveEditService backed by real Valkey (TestValkey fixture) and mock auth.

use std::sync::Arc;

use tonic::Request;

use live_edit::grpc::layer::{handle_delete_layer, handle_stat_layer};
use live_edit::grpc::proto::{DeleteLayerRequest, StatLayerRequest};
use live_edit::grpc::BearerToken;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── Request builders ────────────────────────────────────────────────────────

fn delete_layer_request(session_id: &str, layer_name: &str) -> Request<DeleteLayerRequest> {
    let mut request = Request::new(DeleteLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

fn stat_layer_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═══════════════════════════════════════════════════════════════════════════════
// DeleteLayer tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Happy path: delete an existing layer ─────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_happy_path_removes_layer() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "delete-happy", "target-layer").await;

    let response = handle_delete_layer(&svc, delete_layer_request(&session_id, "target-layer"))
        .await
        .expect("delete of existing layer must succeed");

    // DeleteLayerResponse has no fields — just verify we got one.
    let _ = response.into_inner();

    // Verify the layer is gone by trying to stat it.
    let stat_status = handle_stat_layer(&svc, stat_layer_request(&session_id, "target-layer"))
        .await
        .expect_err("stat after delete must fail");
    assert_eq!(
        stat_status.code(),
        tonic::Code::NotFound,
        "deleted layer must return NOT_FOUND on stat, got: {}",
        stat_status.code()
    );
}

// ── 2. Session not found → NOT_FOUND ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_session_not_found_returns_not_found() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let status = handle_delete_layer(
        &svc,
        delete_layer_request("nonexistent-session", "some-layer"),
    )
    .await
    .expect_err("non-existent session must fail");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "non-existent session must return NOT_FOUND, got: {}",
        status.code()
    );
}

// ── 3. Layer not found → NOT_FOUND ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_layer_not_found_returns_not_found() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "delete-notfound", "existing-layer").await;

    let status = handle_delete_layer(&svc, delete_layer_request(&session_id, "nonexistent-layer"))
        .await
        .expect_err("non-existent layer must fail");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "non-existent layer must return NOT_FOUND, got: {}",
        status.code()
    );
}

// ── 4. Auth denied → PERMISSION_DENIED ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_auth_denied_returns_permission_denied() {
    let tv = TestValkey::setup().await;
    // Create with AllowAll so the layer exists, then test with DenyAll.
    let allow_svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &allow_svc, "delete-denied", "layer-to-deny").await;

    let deny_svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let status = handle_delete_layer(
        &deny_svc,
        delete_layer_request(&session_id, "layer-to-deny"),
    )
    .await
    .expect_err("denied auth must fail");

    assert_eq!(
        status.code(),
        tonic::Code::PermissionDenied,
        "denied auth must return PERMISSION_DENIED, got: {}",
        status.code()
    );
}

// ── 5. Empty session_id → INVALID_ARGUMENT ──────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_empty_session_id_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let status = handle_delete_layer(&svc, delete_layer_request("", "some-layer"))
        .await
        .expect_err("empty session_id must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "empty session_id must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}

// ── 6. Invalid layer name → INVALID_ARGUMENT ────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_invalid_layer_name_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "delete-invalid-name", "good-layer").await;

    let status = handle_delete_layer(&svc, delete_layer_request(&session_id, "../../etc/passwd"))
        .await
        .expect_err("invalid layer name must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "invalid layer name must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// StatLayer tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 7. Happy path: stat an existing layer ───────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_happy_path_returns_metadata() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stat-happy", "my-layer").await;

    let response = handle_stat_layer(&svc, stat_layer_request(&session_id, "my-layer"))
        .await
        .expect("stat of existing layer must succeed");

    let resp = response.into_inner();
    assert!(
        !resp.created_by_user_name.is_empty(),
        "created_by_user_name must be non-empty"
    );
    assert!(resp.created_at > 0, "created_at must be non-zero");
    assert!(
        resp.last_modified_at > 0,
        "last_modified_at must be non-zero"
    );
}

// ── 8. Session not found → NOT_FOUND ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_session_not_found_returns_not_found() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let status = handle_stat_layer(
        &svc,
        stat_layer_request("nonexistent-session", "some-layer"),
    )
    .await
    .expect_err("non-existent session must fail");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "non-existent session must return NOT_FOUND, got: {}",
        status.code()
    );
}

// ── 9. Layer not found → NOT_FOUND ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_layer_not_found_returns_not_found() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stat-notfound", "existing-layer").await;

    let status = handle_stat_layer(&svc, stat_layer_request(&session_id, "nonexistent-layer"))
        .await
        .expect_err("non-existent layer must fail");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "non-existent layer must return NOT_FOUND, got: {}",
        status.code()
    );
}

// ── 10. Auth denied → PERMISSION_DENIED ─────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_auth_denied_returns_permission_denied() {
    let tv = TestValkey::setup().await;
    // Create with AllowAll so the layer exists, then test with DenyAll.
    let allow_svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &allow_svc, "stat-denied", "layer-to-deny").await;

    let deny_svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let status = handle_stat_layer(&deny_svc, stat_layer_request(&session_id, "layer-to-deny"))
        .await
        .expect_err("denied auth must fail");

    assert_eq!(
        status.code(),
        tonic::Code::PermissionDenied,
        "denied auth must return PERMISSION_DENIED, got: {}",
        status.code()
    );
}

// ── 11. Empty session_id → INVALID_ARGUMENT ─────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_empty_session_id_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let status = handle_stat_layer(&svc, stat_layer_request("", "some-layer"))
        .await
        .expect_err("empty session_id must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "empty session_id must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}

// ── 12. Invalid layer name → INVALID_ARGUMENT ───────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_invalid_layer_name_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stat-invalid-name", "good-layer").await;

    let status = handle_stat_layer(&svc, stat_layer_request(&session_id, "../../etc/passwd"))
        .await
        .expect_err("invalid layer name must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "invalid layer name must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}
