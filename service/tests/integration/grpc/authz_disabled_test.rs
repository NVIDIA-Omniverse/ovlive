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

// Integration test: service operates correctly when authz is disabled.
//
// Uses the real NoOpPermissionsClient (not the test AllowAllPermissions mock)
// to validate the production wiring path end-to-end.
// Covers Epic AC #8 (t3.5ci.ny.bg.ha).

use std::sync::Arc;

use tonic::Request;

use live_edit::auth::NoOpPermissionsClient;
use live_edit::grpc::proto::CreateSessionRequest;
use live_edit::grpc::session::handle_create_session;
use live_edit::grpc::BearerToken;
use live_edit::storage::test_support::TestValkey;

use super::helpers::build_service;

fn create_session_request(path: &str, name: &str) -> Request<CreateSessionRequest> {
    let mut request = Request::new(CreateSessionRequest {
        path: path.to_owned(),
        name: name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ── t3.5ci.ny.bg.ha: authz-disabled wiring ─────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_succeeds_with_noop_permissions_client() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(NoOpPermissionsClient));

    let response = handle_create_session(
        &svc,
        create_session_request("/stage", "authz-disabled-test"),
    )
    .await
    .expect("CreateSession must succeed with NoOpPermissionsClient");

    let resp = response.into_inner();
    assert!(
        !resp.session_id.is_empty(),
        "session_id must be non-empty when authz is disabled"
    );
}
