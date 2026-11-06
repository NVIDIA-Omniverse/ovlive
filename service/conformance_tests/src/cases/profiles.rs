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

//! The cases of the optional profiles (`deployment`, `permissions`,
//! `multi_replica`): the port of `test_profiles.py`.

use tonic::Code;
use tonic_health::pb::health_check_response::ServingStatus;

use crate::client::LIVE_SERVICE_NAME;
use crate::fixtures::{fixture, Fixture};
use crate::proto;
use crate::runner::{Case, CaseFuture, Ctx, Profile};
use crate::{check, check_eq, expect_code};

const FILE: &str = "profiles";

/// This module's cases, in definition order.
pub fn cases() -> Vec<Case> {
    vec![
        Case::new(
            FILE,
            "test_live_service_health_is_serving_without_auth",
            Profile::Deployment,
            false,
            test_live_service_health_is_serving_without_auth,
        ),
        Case::new(
            FILE,
            "test_denied_base_path_returns_permission_denied",
            Profile::Permissions,
            false,
            test_denied_base_path_returns_permission_denied,
        ),
        Case::new(
            FILE,
            "test_update_across_replica_endpoints",
            Profile::MultiReplica,
            true,
            test_update_across_replica_endpoints,
        ),
    ]
}

fn test_live_service_health_is_serving_without_auth(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let response = client
            .health(LIVE_SERVICE_NAME, client.settings().timeout)
            .await?;
        check_eq!(response.status, ServingStatus::Serving as i32);
        Ok(())
    })
}

fn test_denied_base_path_returns_permission_denied(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        // The preflight refuses a `permissions` run without
        // `CONFORMANCE_DENIED_PATH`; unset, the request names no path, as the
        // Python suite's `path=None` did, and fails the expected code.
        let path = client.settings().denied_path.clone().unwrap_or_default();
        expect_code!(
            client
                .list_sessions(proto::ListSessionsRequest { path })
                .await,
            Code::PermissionDenied
        );
        Ok(())
    })
}

fn test_update_across_replica_endpoints(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        // The runner builds the peer client for `multi_replica` cases and
        // closes it after the case, as the Python `finally: peer.close()` did.
        let peer = ctx.peer()?;
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let mut stream = peer.subscribe(session, created.object_id, 1).await?;
        stream.caught_up(created.object_id, 1).await?;
        let response = client
            .update(session, created.object_id, fixture(Fixture::Update))
            .await?;
        let (header, payload) = stream.delivery().await?;
        check_eq!(header.object_id, created.object_id);
        check_eq!(header.sequence_num, response.sequence_num);
        check_eq!(response.sequence_num, 2);
        check!(payload.as_slice() == fixture(Fixture::Update));
        Ok(())
    })
}
