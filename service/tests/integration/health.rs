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

// Integration tests for gRPC health checks.

use tonic::transport::Channel;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;

use crate::common::spawn_test_server;

async fn health_client(addr: std::net::SocketAddr) -> HealthClient<Channel> {
    let endpoint = format!("http://{}", addr);
    let channel = Channel::from_shared(endpoint)
        .expect("invalid endpoint URI")
        .connect()
        .await
        .expect("failed to connect to test server");
    HealthClient::new(channel)
}

#[tokio::test]
async fn health_overall_service_returns_serving() {
    let handle = spawn_test_server().await;
    let mut client = health_client(handle.addr).await;

    let resp = client
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("health check for \"\" must succeed")
        .into_inner();

    assert_eq!(
        resp.status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32,
        "overall health (service=\"\") must be SERVING"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn health_live_service_returns_serving() {
    let handle = spawn_test_server().await;
    let mut client = health_client(handle.addr).await;

    let resp = client
        .check(HealthCheckRequest {
            service: "nvidia.omniverse.live.v1alpha.LiveService".to_string(),
        })
        .await
        .expect("health check for LiveService must succeed")
        .into_inner();

    assert_eq!(
        resp.status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32,
        "LiveService health must be SERVING"
    );

    handle.shutdown().await;
}

#[tokio::test]
async fn health_unregistered_service_returns_not_found() {
    let handle = spawn_test_server().await;
    let mut client = health_client(handle.addr).await;

    let result = client
        .check(HealthCheckRequest {
            service: "nvidia.omniverse.nonexistent.v1.FakeService".to_string(),
        })
        .await;

    // tonic-health returns NOT_FOUND for unregistered services
    match result {
        Err(status) => {
            assert_eq!(
                status.code(),
                tonic::Code::NotFound,
                "unregistered service must return NOT_FOUND, got {:?}",
                status.code()
            );
        }
        Ok(resp) => {
            // Some tonic-health versions return SERVICE_UNKNOWN (4) instead
            assert_eq!(
                resp.into_inner().status,
                tonic_health::pb::health_check_response::ServingStatus::ServiceUnknown as i32,
                "unregistered service should be NOT_FOUND or SERVICE_UNKNOWN"
            );
        }
    }

    handle.shutdown().await;
}
