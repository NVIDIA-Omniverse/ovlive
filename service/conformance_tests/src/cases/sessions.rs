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

//! Ping, session CRUD, identifier validation, the bearer requirement and
//! session isolation: the port of `test_sessions.py`.

use tonic::Code;

use crate::client::{Auth, CallError};
use crate::fixtures::{fixture, Fixture};
use crate::proto;
use crate::runner::{Case, CaseFuture, Ctx, Profile};
use crate::{check, check_eq, expect_code};

const FILE: &str = "sessions";

/// This module's cases, in definition order.
pub fn cases() -> Vec<Case> {
    let mut cases = vec![
        Case::new(
            FILE,
            "test_ping_echoes_timestamp",
            Profile::Core,
            false,
            test_ping_echoes_timestamp,
        ),
        Case::new(
            FILE,
            "test_session_create_list_delete_and_duplicate_name",
            Profile::Core,
            false,
            test_session_create_list_delete_and_duplicate_name,
        ),
    ];
    cases.extend(EmptyIdentifier::ALL.into_iter().map(|rpc| {
        Case::with_param(
            FILE,
            "test_empty_identifiers_are_invalid",
            rpc.name(),
            Profile::Core,
            false,
            rpc,
            test_empty_identifiers_are_invalid,
        )
    }));
    cases.extend([
        Case::new(
            FILE,
            "test_anonymous_request_is_rejected",
            Profile::Core,
            false,
            test_anonymous_request_is_rejected,
        ),
        Case::new(
            FILE,
            "test_sessions_isolate_layer_names",
            Profile::Core,
            true,
            test_sessions_isolate_layer_names,
        ),
        Case::new(
            FILE,
            "test_session_delete_cascades_to_layer_and_subscription",
            Profile::Core,
            true,
            test_session_delete_cascades_to_layer_and_subscription,
        ),
    ]);
    cases
}

/// A `conformance-` or `anonymous-` name unique to this run: the prefix and a
/// v4 UUID as 32 lowercase hex digits, as Python's `uuid.uuid4().hex`.
fn unique_name(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

fn test_ping_echoes_timestamp(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let response = ctx
            .client
            .ping(proto::PingRequest {
                client_timestamp_microseconds_since_epoch: 12345,
            })
            .await?;
        check_eq!(response.client_timestamp_microseconds_since_epoch, 12345);
        check!(response.server_timestamp_microseconds_since_epoch > 0);
        Ok(())
    })
}

fn test_session_create_list_delete_and_duplicate_name(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let base_path = client.settings().base_path.clone();
        let name = unique_name("conformance");
        let session = client.session_named(&name).await?;
        let rows = client
            .list_sessions(proto::ListSessionsRequest {
                path: base_path.clone(),
            })
            .await?
            .sessions;
        let matching: Vec<&proto::Session> = rows
            .iter()
            .filter(|row| row.session_id == session)
            .collect();
        check_eq!(matching.len(), 1);
        check_eq!(matching[0].name, name);
        check!(matching[0].created_at > 0);
        expect_code!(client.session_named(&name).await, Code::AlreadyExists);
        client
            .delete_session(proto::DeleteSessionRequest {
                session_id: session.clone(),
            })
            .await?;
        let rows = client
            .list_sessions(proto::ListSessionsRequest { path: base_path })
            .await?
            .sessions;
        check!(!rows.iter().any(|row| row.session_id == session));
        Ok(())
    })
}

/// The RPC of a `test_empty_identifiers_are_invalid` parametrization, each
/// sent with an all-default request: no path, session id or layer name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmptyIdentifier {
    CreateSession,
    ListSessions,
    DeleteSession,
    StatLayer,
    DeleteLayer,
}

impl EmptyIdentifier {
    /// In the Python parametrize list's order.
    const ALL: [EmptyIdentifier; 5] = [
        EmptyIdentifier::CreateSession,
        EmptyIdentifier::ListSessions,
        EmptyIdentifier::DeleteSession,
        EmptyIdentifier::StatLayer,
        EmptyIdentifier::DeleteLayer,
    ];

    /// The RPC's name: the parametrization's id.
    fn name(self) -> &'static str {
        match self {
            EmptyIdentifier::CreateSession => "CreateSession",
            EmptyIdentifier::ListSessions => "ListSessions",
            EmptyIdentifier::DeleteSession => "DeleteSession",
            EmptyIdentifier::StatLayer => "StatLayer",
            EmptyIdentifier::DeleteLayer => "DeleteLayer",
        }
    }
}

fn test_empty_identifiers_are_invalid(ctx: &mut Ctx, rpc: EmptyIdentifier) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let result: Result<(), CallError> = match rpc {
            EmptyIdentifier::CreateSession => client
                .create_session(proto::CreateSessionRequest::default())
                .await
                .map(drop),
            EmptyIdentifier::ListSessions => client
                .list_sessions(proto::ListSessionsRequest::default())
                .await
                .map(drop),
            EmptyIdentifier::DeleteSession => client
                .delete_session(proto::DeleteSessionRequest::default())
                .await
                .map(drop),
            EmptyIdentifier::StatLayer => client
                .stat_layer(proto::StatLayerRequest::default())
                .await
                .map(drop),
            EmptyIdentifier::DeleteLayer => client
                .delete_layer(proto::DeleteLayerRequest::default())
                .await
                .map(drop),
        };
        expect_code!(result, Code::InvalidArgument);
        Ok(())
    })
}

fn test_anonymous_request_is_rejected(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let request = proto::CreateSessionRequest {
            path: client.settings().base_path.clone(),
            name: unique_name("anonymous"),
        };
        expect_code!(
            client.create_session_as(request, Auth::Anonymous).await,
            Code::Unauthenticated
        );
        Ok(())
    })
}

fn test_sessions_isolate_layer_names(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let other = client.session().await?;
        client.create(session, fixture(Fixture::Initial)).await?;
        expect_code!(
            client
                .stat_layer(proto::StatLayerRequest {
                    session_id: other,
                    layer_name: "layer".into(),
                })
                .await,
            Code::NotFound
        );
        Ok(())
    })
}

fn test_session_delete_cascades_to_layer_and_subscription(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let client = &ctx.client;
        let session = &ctx.session;
        let created = client.create(session, fixture(Fixture::Initial)).await?;
        let mut stream = client.subscribe(session, created.object_id, 1).await?;
        stream.caught_up(created.object_id, 1).await?;
        client
            .delete_session(proto::DeleteSessionRequest {
                session_id: session.clone(),
            })
            .await?;
        expect_code!(stream.next().await, Code::NotFound);
        expect_code!(
            client
                .stat_layer(proto::StatLayerRequest {
                    session_id: session.clone(),
                    layer_name: "layer".into(),
                })
                .await,
            Code::NotFound
        );
        Ok(())
    })
}
