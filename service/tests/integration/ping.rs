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

// Integration tests for the Ping RPC over a real gRPC connection.

use std::time::{SystemTime, UNIX_EPOCH};

use tonic::transport::Channel;

use crate::common::{proto, spawn_test_server};

async fn live_client(
    addr: std::net::SocketAddr,
) -> proto::live_service_client::LiveServiceClient<Channel> {
    let endpoint = format!("http://{}", addr);
    proto::live_service_client::LiveServiceClient::connect(endpoint)
        .await
        .expect("failed to connect to test server")
}

#[tokio::test]
async fn ping_over_grpc_echoes_client_timestamp() {
    let handle = spawn_test_server().await;
    let mut client = live_client(handle.addr).await;

    let client_ts: u64 = 1_700_000_000_000_000;
    let resp = client
        .ping(proto::PingRequest {
            client_timestamp_microseconds_since_epoch: client_ts,
        })
        .await
        .expect("Ping RPC failed")
        .into_inner();

    assert_eq!(
        resp.client_timestamp_microseconds_since_epoch, client_ts,
        "client_timestamp must be echoed exactly"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn ping_over_grpc_server_timestamp_is_plausible() {
    let handle = spawn_test_server().await;
    let mut client = live_client(handle.addr).await;

    let before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;

    let resp = client
        .ping(proto::PingRequest {
            client_timestamp_microseconds_since_epoch: 0,
        })
        .await
        .expect("Ping RPC failed")
        .into_inner();

    let after = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;

    assert!(
        resp.server_timestamp_microseconds_since_epoch >= before,
        "server_timestamp must be >= time before RPC call"
    );
    assert!(
        resp.server_timestamp_microseconds_since_epoch <= after,
        "server_timestamp must be <= time after RPC call"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn ping_over_grpc_server_shuts_down_cleanly() {
    let handle = spawn_test_server().await;
    let addr = handle.addr;
    let mut client = live_client(addr).await;

    // Confirm the server is up.
    client
        .ping(proto::PingRequest {
            client_timestamp_microseconds_since_epoch: 42,
        })
        .await
        .expect("Ping before shutdown must succeed");

    // Shut down and verify it completes without panicking.
    handle.shutdown().await;
}
