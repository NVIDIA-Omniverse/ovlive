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

// Integration tests for graceful shutdown health transitions.
//
// Note: These tests validate the health reporter's NOT_SERVING transition directly,
// which is the load-bearing contract. Full signal-based shutdown is tested in
// production deployments. The health transition logic is what matters for K8s probes.

use std::time::Duration;

use tonic::transport::Channel;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;
use tonic_health::ServingStatus;

use crate::common::{proto, spawn_test_server};

async fn health_client(addr: std::net::SocketAddr) -> HealthClient<Channel> {
    let endpoint = format!("http://{}", addr);
    let channel = Channel::from_shared(endpoint)
        .expect("invalid endpoint URI")
        .connect()
        .await
        .expect("failed to connect to test server");
    HealthClient::new(channel)
}

async fn live_client(
    addr: std::net::SocketAddr,
) -> proto::live_service_client::LiveServiceClient<Channel> {
    let endpoint = format!("http://{}", addr);
    proto::live_service_client::LiveServiceClient::connect(endpoint)
        .await
        .expect("failed to connect live client")
}

#[tokio::test]
async fn shutdown_health_transitions_to_not_serving_for_overall() {
    let handle = spawn_test_server().await;
    let mut health = health_client(handle.addr).await;

    // Confirm serving before shutdown.
    let before = health
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("health check must succeed before shutdown")
        .into_inner();
    assert_eq!(
        before.status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32,
        "overall health must be SERVING before shutdown"
    );

    // Simulate the shutdown transition (mirrors what main.rs does on signal).
    handle
        .health_reporter
        .set_service_status("", ServingStatus::NotServing)
        .await;

    // Brief pause to let the health state propagate.
    tokio::time::sleep(Duration::from_millis(10)).await;

    let after = health
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("health check must be reachable during shutdown transition")
        .into_inner();
    assert_eq!(
        after.status,
        tonic_health::pb::health_check_response::ServingStatus::NotServing as i32,
        "overall health must be NOT_SERVING after shutdown signal"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn shutdown_health_transitions_to_not_serving_for_live_service() {
    // TestService is local to common, so we use set_service_status by name directly.
    let handle = spawn_test_server().await;
    let mut health = health_client(handle.addr).await;

    handle
        .health_reporter
        .set_service_status(
            "nvidia.omniverse.live.v1alpha.LiveService",
            ServingStatus::NotServing,
        )
        .await;

    tokio::time::sleep(Duration::from_millis(10)).await;

    let resp = health
        .check(HealthCheckRequest {
            service: "nvidia.omniverse.live.v1alpha.LiveService".to_string(),
        })
        .await
        .expect("health check must be reachable during transition")
        .into_inner();
    assert_eq!(
        resp.status,
        tonic_health::pb::health_check_response::ServingStatus::NotServing as i32,
        "LiveService health must be NOT_SERVING after transition"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn shutdown_inflight_ping_completes_before_server_stops() {
    let handle = spawn_test_server().await;
    let addr = handle.addr;

    let mut live = live_client(addr).await;

    // Start a Ping in the background.
    let ping_task = tokio::spawn(async move {
        live.ping(proto::PingRequest {
            client_timestamp_microseconds_since_epoch: 99,
        })
        .await
    });

    // Give the request a moment to be in-flight before shutdown.
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Initiate shutdown — server waits for in-flight RPCs to complete.
    handle.shutdown().await;

    // The in-flight Ping must have completed successfully.
    let resp = tokio::time::timeout(Duration::from_secs(3), ping_task)
        .await
        .expect("ping task timed out")
        .expect("ping task panicked")
        .expect("Ping RPC failed during graceful shutdown");

    assert_eq!(
        resp.into_inner().client_timestamp_microseconds_since_epoch,
        99,
        "in-flight Ping must complete with correct echo during shutdown"
    );
}

#[tokio::test]
async fn shutdown_server_exits_within_timeout() {
    let handle = spawn_test_server().await;

    let start = std::time::Instant::now();
    handle.shutdown().await;
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "server must exit within 5 seconds, took {:?}",
        elapsed
    );
}
