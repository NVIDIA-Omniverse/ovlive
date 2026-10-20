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

// Axum HTTP server for the metrics port: GET /metrics in Prometheus text
// format, and GET /scaling/v1, the autoscaler's load snapshot (see
// `observability::scaling`).
//
// The metrics themselves are recorded via the OpenTelemetry metrics API
// (OMPE-98174); the `opentelemetry-prometheus-text-exporter` `PrometheusExporter`
// handed to `serve_metrics_with_scaling` (or `serve_metrics`) is the
// always-attached OTel scrape reader. Its
// `export()` drives a fresh SDK collect (invoking any observable-gauge
// callbacks) and writes the Prometheus text exposition directly — no separate
// `prometheus::Registry` / `gather()` / `TextEncoder` step needed.

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use live_edit_scaler::snapshot::SCALING_SNAPSHOT_PATH;
use opentelemetry_prometheus_text_exporter::PrometheusExporter;
use tokio::net::TcpListener;

use crate::observability::scaling::ScalingEndpoint;

#[derive(Clone)]
struct MetricsState {
    exporter: PrometheusExporter,
    scaling: Arc<ScalingEndpoint>,
}

/// Serve Prometheus metrics on the given TCP listener.
///
/// [`serve_metrics_with_scaling`] with an endpoint nothing registers on, so
/// `GET /scaling/v1` answers 503 for the server's life. For servers that carry
/// no autoscaler snapshot (tests that exercise `/metrics` alone).
pub async fn serve_metrics(listener: TcpListener, exporter: PrometheusExporter) {
    serve_metrics_with_scaling(listener, exporter, Arc::new(ScalingEndpoint::new())).await;
}

/// Serve the metrics port on the given TCP listener: `GET /metrics` and
/// `GET /scaling/v1`.
///
/// `GET /metrics` renders all metric families by calling
/// `PrometheusExporter::export`, which runs a collect() on the OpenTelemetry
/// meter provider (including observable-gauge callbacks) and encodes them in
/// the standard Prometheus text exposition format.
///
/// `GET /scaling/v1` is the autoscaler's load snapshot — an internal contract
/// between the service and `live-edit scaler`, not a customer API. It answers
/// `200` with the JSON snapshot once a source is registered on `scaling`, and
/// `503` before (the server starts ahead of the dependencies the source reads).
///
/// This function runs until the provided listener is closed or the task is
/// aborted (e.g. via `JoinHandle::abort()` during shutdown).
pub async fn serve_metrics_with_scaling(
    listener: TcpListener,
    exporter: PrometheusExporter,
    scaling: Arc<ScalingEndpoint>,
) {
    let state = MetricsState { exporter, scaling };
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route(SCALING_SNAPSHOT_PATH, get(scaling_handler))
        .with_state(state);

    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| tracing::error!(error = %e, "metrics server error"));
}

async fn metrics_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    let mut buf = Vec::new();
    if let Err(e) = state.exporter.export(&mut buf) {
        tracing::error!(error = %e, "failed to encode metrics");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            [("content-type", "text/plain")],
            b"failed to encode metrics".to_vec(),
        );
    }
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        buf,
    )
}

async fn scaling_handler(State(state): State<MetricsState>) -> impl IntoResponse {
    let Some(snapshot) = state.scaling.snapshot().await else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("content-type", "text/plain")],
            b"scaling snapshot not available yet: the service is starting".to_vec(),
        );
    };
    match serde_json::to_vec(&snapshot) {
        Ok(body) => (StatusCode::OK, [("content-type", "application/json")], body),
        Err(e) => {
            tracing::error!(error = %e, "failed to encode the scaling snapshot");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [("content-type", "text/plain")],
                b"failed to encode the scaling snapshot".to_vec(),
            )
        }
    }
}
