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

// Integration tests: metrics HTTP server and gRPC server coexist on separate ports.
//
// Note: The Ping handler is not yet instrumented with Metrics (that happens in a
// later epic). The endpoint_calls_total counter will remain 0 after a Ping call
// until handler instrumentation is added. These tests verify both servers can start
// concurrently and respond to requests simultaneously.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use prometheus::{Encoder, Registry, TextEncoder};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use crate::common::{proto, spawn_test_server};

/// Spawn a minimal axum /metrics server. Returns (addr, task_handle).
async fn spawn_metrics_server(registry: Arc<Registry>) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind metrics port");
    let addr = listener.local_addr().expect("no local addr");

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(registry);

    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    (addr, task)
}

async fn metrics_handler(State(registry): State<Arc<Registry>>) -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let families = registry.gather();
    let mut buf = Vec::new();
    encoder.encode(&families, &mut buf).unwrap_or_default();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        buf,
    )
}

// ── tests ──────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_servers_start_without_address_conflicts() {
    let grpc_handle = spawn_test_server().await;
    let registry = Arc::new(Registry::new());
    let (metrics_addr, metrics_task) = spawn_metrics_server(registry).await;

    // Both servers must be on distinct addresses.
    assert_ne!(
        grpc_handle.addr, metrics_addr,
        "gRPC and metrics servers must not share a port"
    );

    grpc_handle.shutdown().await;
    metrics_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ping_and_metrics_respond_simultaneously() {
    // Spawn both servers.
    let grpc_handle = spawn_test_server().await;
    let registry = Arc::new(Registry::new());
    let (metrics_addr, metrics_task) = spawn_metrics_server(registry).await;

    let grpc_addr = grpc_handle.addr;

    // Issue gRPC Ping and HTTP GET /metrics concurrently.
    let ping_task = tokio::spawn(async move {
        let endpoint = format!("http://{grpc_addr}");
        let mut client = proto::live_service_client::LiveServiceClient::connect(endpoint)
            .await
            .expect("failed to connect to gRPC server");
        client
            .ping(proto::PingRequest {
                client_timestamp_microseconds_since_epoch: 12345,
            })
            .await
            .expect("Ping must succeed")
            .into_inner()
    });

    let http_task = tokio::spawn(async move {
        let client = reqwest::Client::new();
        client
            .get(format!("http://{metrics_addr}/metrics"))
            .send()
            .await
            .expect("GET /metrics must succeed")
    });

    let (ping_resp, http_resp) = tokio::join!(ping_task, http_task);

    let ping_resp = ping_resp.expect("ping task must not panic");
    assert_eq!(
        ping_resp.client_timestamp_microseconds_since_epoch, 12345,
        "Ping must echo client timestamp"
    );

    let http_resp = http_resp.expect("http task must not panic");
    assert_eq!(
        http_resp.status().as_u16(),
        200,
        "GET /metrics must return 200 while gRPC server is running"
    );

    grpc_handle.shutdown().await;
    metrics_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_shutdown_does_not_crash_metrics_server() {
    let grpc_handle = spawn_test_server().await;
    let registry = Arc::new(Registry::new());
    let (metrics_addr, metrics_task) = spawn_metrics_server(registry).await;

    // Shut down only the gRPC server.
    grpc_handle.shutdown().await;

    // Metrics server must still be responsive after gRPC shuts down.
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{metrics_addr}/metrics"))
        .send()
        .await
        .expect("GET /metrics must remain reachable after gRPC shutdown");

    assert_eq!(
        resp.status().as_u16(),
        200,
        "metrics server must remain alive after gRPC server stops"
    );

    metrics_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_endpoint_calls_total_reflects_counter_state() {
    // NOTE: The gRPC Ping handler is not yet wired to Metrics, so
    // endpoint_calls_total stays at 0 after a Ping. This test verifies that
    // the counter infrastructure is correct: manual inc() is visible via /metrics.
    use prometheus::{CounterVec, Opts};

    let registry = Arc::new(Registry::new());

    let endpoint_calls = CounterVec::new(
        Opts::new("live_endpoint_calls_total", "Total gRPC endpoint calls"),
        &["endpoint", "service"],
    )
    .unwrap();
    registry.register(Box::new(endpoint_calls.clone())).unwrap();

    // Simulate what a wired Ping handler would do.
    endpoint_calls
        .with_label_values(&["Ping", "test-pod"])
        .inc();

    let (metrics_addr, metrics_task) = spawn_metrics_server(registry).await;

    let client = reqwest::Client::new();
    let body = client
        .get(format!("http://{metrics_addr}/metrics"))
        .send()
        .await
        .expect("GET /metrics must succeed")
        .text()
        .await
        .expect("body must be UTF-8");

    assert!(
        body.contains("live_endpoint_calls_total"),
        "body must contain live_endpoint_calls_total"
    );
    // Counter was incremented once; value should be 1.
    assert!(
        body.contains("} 1"),
        "endpoint_calls_total must show value 1 after one increment, got:\n{body}"
    );

    metrics_task.abort();
}
