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

// Integration test for b.nrt item 4 — auth metrics visible in /metrics output.
//
// The inline unit test `metrics_increments_cache_hits_and_misses` asserts
// against the prometheus in-memory counter handles directly. This test goes
// one layer further: it stands up the axum /metrics handler used in
// production, exercises AuthCache so that hits/misses/call-duration samples
// are recorded, then scrapes /metrics and parses the Prometheus text
// exposition format to verify:
//
//   - all four b.nrt auth metric families are present,
//   - `auth_cache_hits_total{kind="permissions|userinfo"}` and
//     `auth_cache_misses_total{kind="permissions|userinfo"}` are
//     *pre-initialized at zero* on first scrape (the reason they must
//     show up even with no cache activity — operators need to see the
//     series exist),
//   - after traffic, counters reflect the actual hit/miss pattern.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use opentelemetry_prometheus_text_exporter::PrometheusExporter;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use live_edit::auth::cache::AuthCache;
use live_edit::auth::permissions::PermissionsClient;
use live_edit::auth::userinfo::UserinfoClient;
use live_edit::auth::{AuthError, AuthResult, Permission, UserInfo};

/// Spawn an axum metrics server on a random port and return (addr, task_handle).
///
/// Serves `/metrics` from the same `PrometheusExporter` the `Metrics`
/// instance under test owns (the OTel scrape reader), mirroring production
/// `serve_metrics` — so the scraped text reflects the actual instrument
/// state rather than a disconnected throwaway registry.
///
/// Replaces a 20 ms blanket sleep with an explicit oneshot "listener
/// ready" signal (b.ohv item 5). The server task fires the signal once
/// `axum::serve` is about to run on the listener — that is, after every
/// sync setup step. Subsequent client connects target a ready accept loop.
async fn spawn_metrics_server(exporter: PrometheusExporter) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind to random port");
    let addr = listener.local_addr().expect("no local addr");

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(exporter);

    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        // Signal readiness before entering the serve future — any
        // subsequent test-side connect is guaranteed to find the accept
        // loop driven by this task.
        let _ = ready_tx.send(());
        axum::serve(listener, app).await.ok();
    });

    // Bounded wait so a missed signal can't hang the test indefinitely.
    let _ = tokio::time::timeout(Duration::from_secs(5), ready_rx)
        .await
        .expect("metrics server task did not signal readiness within 5s");
    (addr, task)
}

async fn metrics_handler(State(exporter): State<PrometheusExporter>) -> impl IntoResponse {
    let mut buf = Vec::new();
    exporter.export(&mut buf).unwrap_or_default();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        buf,
    )
}

// ── test auth doubles ──────────────────────────────────────────────────────

struct AllowAllPermissions;

#[async_trait]
impl PermissionsClient for AllowAllPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        Ok(AuthResult::Allowed)
    }
}

struct StubUserinfo;

#[async_trait]
impl UserinfoClient for StubUserinfo {
    async fn get_current_user(&self, _token: &str) -> Result<UserInfo, AuthError> {
        Ok(UserInfo {
            id: "u1".into(),
            display_name: "U".into(),
            email: None,
            username: "u".into(),
        })
    }
}

/// Count occurrences of a substring in `haystack`.
fn count_matches(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

/// Parse the numeric sample value from a single Prometheus text-format line,
/// e.g. `live_edit_auth_cache_hits_total{kind="permissions"} 3` → 3.0.
/// Returns None if the line does not match `metric_name{label_key="label_value"} <f64>`.
fn parse_counter_value(
    body: &str,
    metric_name: &str,
    label_key: &str,
    label_value: &str,
) -> Option<f64> {
    for line in body.lines() {
        // Skip HELP and TYPE lines and anything that doesn't look like a sample.
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if !line.starts_with(metric_name) {
            continue;
        }
        // Match the exact label key=value pair to avoid false matches across
        // sibling series (e.g. hits vs misses both start with the same prefix
        // when scanning for family names).
        let label_pat = format!("{label_key}=\"{label_value}\"");
        if !line.contains(&label_pat) {
            continue;
        }
        // After the closing '}', expect ` <number>`.
        if let Some(rest) = line.rsplit_once(' ') {
            if let Ok(v) = rest.1.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

// ── tests ──────────────────────────────────────────────────────────────────

/// Pre-initialization: the four auth metrics defined for b.nrt must be
/// visible in /metrics output even before any auth traffic occurs, so
/// operators aren't confused by missing series during quiet periods.
#[tokio::test]
async fn auth_metric_families_pre_initialized_at_zero_on_first_scrape() {
    // Construct Metrics so that all series are registered and (for
    // counters with enumerated labels) pre-initialized.
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-pod"));

    let (addr, task) = spawn_metrics_server(metrics.prometheus_exporter()).await;

    let body = reqwest::Client::new()
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("GET /metrics must succeed")
        .text()
        .await
        .expect("body must be UTF-8");

    task.abort();

    // The two cache-hit/miss counters are pre-initialized for both `kind`
    // label values so they appear at zero even with no activity. The
    // histogram and error counters have label values that can't all be
    // enumerated at startup (`backend` × `protocol`, unbounded `code`),
    // so they only appear once sampled — those are exercised by the
    // traffic test below.
    assert!(
        body.contains("live_edit_auth_cache_hits_total"),
        "missing live_edit_auth_cache_hits_total in /metrics, got:\n{body}"
    );
    assert!(
        body.contains("live_edit_auth_cache_misses_total"),
        "missing live_edit_auth_cache_misses_total in /metrics, got:\n{body}"
    );

    // Both `kind` labels must be pre-initialized at zero so Prometheus
    // captures the existence of the series immediately.
    assert_eq!(
        parse_counter_value(
            &body,
            "live_edit_auth_cache_hits_total",
            "kind",
            "permissions"
        ),
        Some(0.0),
        "permissions hits must be pre-initialized at zero, got body:\n{body}"
    );
    assert_eq!(
        parse_counter_value(&body, "live_edit_auth_cache_hits_total", "kind", "userinfo"),
        Some(0.0),
        "userinfo hits must be pre-initialized at zero, got body:\n{body}"
    );
    assert_eq!(
        parse_counter_value(
            &body,
            "live_edit_auth_cache_misses_total",
            "kind",
            "permissions"
        ),
        Some(0.0),
        "permissions misses must be pre-initialized at zero, got body:\n{body}"
    );
    assert_eq!(
        parse_counter_value(
            &body,
            "live_edit_auth_cache_misses_total",
            "kind",
            "userinfo"
        ),
        Some(0.0),
        "userinfo misses must be pre-initialized at zero, got body:\n{body}"
    );
}

/// After running real traffic through `AuthCache`, /metrics output must
/// reflect the actual hit/miss counters (not just the existence of the
/// series). Guards against the cache being wired with a no-op metrics
/// handle or a detached registry.
#[tokio::test]
async fn auth_cache_traffic_reflected_in_metrics_endpoint_output() {
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-pod"));

    let cache = AuthCache::new_with_negative_ttl(
        Arc::new(AllowAllPermissions),
        Arc::new(StubUserinfo),
        300,
        30,
        5,
        1_000,
        Some(Arc::clone(&metrics)),
    );

    // First call for this (token, path, permission): miss → 1 permissions miss.
    cache
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();
    // Second call with same args: hit → 1 permissions hit.
    cache
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();

    // First userinfo call: miss → 1 userinfo miss.
    cache.get_current_user("tok").await.unwrap();
    // Second userinfo call: hit → 1 userinfo hit.
    cache.get_current_user("tok").await.unwrap();

    let (addr, task) = spawn_metrics_server(metrics.prometheus_exporter()).await;

    let body = reqwest::Client::new()
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("GET /metrics must succeed")
        .text()
        .await
        .expect("body must be UTF-8");

    task.abort();

    // Counter values must match the traffic pattern above.
    assert_eq!(
        parse_counter_value(
            &body,
            "live_edit_auth_cache_misses_total",
            "kind",
            "permissions"
        ),
        Some(1.0),
        "expected one permissions miss in /metrics output, got:\n{body}"
    );
    assert_eq!(
        parse_counter_value(
            &body,
            "live_edit_auth_cache_hits_total",
            "kind",
            "permissions"
        ),
        Some(1.0),
        "expected one permissions hit in /metrics output, got:\n{body}"
    );
    assert_eq!(
        parse_counter_value(
            &body,
            "live_edit_auth_cache_misses_total",
            "kind",
            "userinfo"
        ),
        Some(1.0),
        "expected one userinfo miss in /metrics output, got:\n{body}"
    );
    assert_eq!(
        parse_counter_value(&body, "live_edit_auth_cache_hits_total", "kind", "userinfo"),
        Some(1.0),
        "expected one userinfo hit in /metrics output, got:\n{body}"
    );

    // Belt-and-braces: the raw metric family names must appear. If label
    // parsing above ever regresses, at least this catches absence.
    assert!(count_matches(&body, "live_edit_auth_cache_hits_total") >= 1);
    assert!(count_matches(&body, "live_edit_auth_cache_misses_total") >= 1);
}

/// The `live_edit_auth_call_duration_seconds` histogram family only
/// appears in /metrics output after an observation is recorded — this
/// test uses `InstrumentedPermissionsClient` (the production wiring) to
/// record a sample and then verifies the histogram is scrape-visible.
#[tokio::test]
async fn auth_call_duration_histogram_appears_in_metrics_after_observation() {
    use live_edit::auth::InstrumentedPermissionsClient;

    let metrics = Arc::new(crate::common::metrics_scrape_only("test-pod"));

    let inner: Arc<dyn PermissionsClient> = Arc::new(AllowAllPermissions);
    let wrapped =
        InstrumentedPermissionsClient::new(inner, Arc::clone(&metrics), "storage", "rest");

    wrapped
        .check_permission("tok", "/p", Permission::Read)
        .await
        .unwrap();

    let (addr, task) = spawn_metrics_server(metrics.prometheus_exporter()).await;
    let body = reqwest::Client::new()
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("GET /metrics must succeed")
        .text()
        .await
        .expect("body must be UTF-8");
    task.abort();

    assert!(
        body.contains("live_edit_auth_call_duration_seconds"),
        "missing live_edit_auth_call_duration_seconds in /metrics after observation, got:\n{body}"
    );
    // The bucket lines carry the backend and protocol labels; verify the
    // exact pair we recorded with appears.
    assert!(
        body.contains("backend=\"storage\"") && body.contains("protocol=\"rest\""),
        "histogram must carry backend=storage protocol=rest labels, got:\n{body}"
    );
    // Histogram emits *_count and *_sum series; both should be present.
    assert!(
        body.contains("live_edit_auth_call_duration_seconds_count"),
        "missing _count series in /metrics, got:\n{body}"
    );
    assert!(
        body.contains("live_edit_auth_call_duration_seconds_sum"),
        "missing _sum series in /metrics, got:\n{body}"
    );
}

/// `live_edit_auth_errors_total` is not pre-initialized — the `code`
/// label is unbounded (classification of `AuthError` variants). This
/// test drives one error through `InstrumentedPermissionsClient` and
/// then verifies the family appears in scrape output with the correct
/// backend + code labels.
#[tokio::test]
async fn auth_errors_counter_appears_in_metrics_after_first_error() {
    use live_edit::auth::InstrumentedPermissionsClient;

    struct TimeoutPerms;
    #[async_trait]
    impl PermissionsClient for TimeoutPerms {
        async fn check_permission(
            &self,
            _token: &str,
            _path: &str,
            _permission: Permission,
        ) -> Result<AuthResult, AuthError> {
            Err(AuthError::Timeout)
        }
    }

    let metrics = Arc::new(crate::common::metrics_scrape_only("test-pod"));

    let inner: Arc<dyn PermissionsClient> = Arc::new(TimeoutPerms);
    let wrapped =
        InstrumentedPermissionsClient::new(inner, Arc::clone(&metrics), "storage", "grpc");
    let _ = wrapped
        .check_permission("tok", "/p", Permission::Read)
        .await;

    let (addr, task) = spawn_metrics_server(metrics.prometheus_exporter()).await;
    let body = reqwest::Client::new()
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("GET /metrics must succeed")
        .text()
        .await
        .expect("body must be UTF-8");
    task.abort();

    assert!(
        body.contains("live_edit_auth_errors_total"),
        "missing live_edit_auth_errors_total in /metrics after error, got:\n{body}"
    );
    // The counter value line must reflect the single observed timeout.
    assert_eq!(
        parse_counter_value(&body, "live_edit_auth_errors_total", "code", "timeout"),
        Some(1.0),
        "expected one storage timeout error reflected in /metrics, got:\n{body}"
    );
}
