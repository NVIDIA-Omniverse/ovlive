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

//! Metric-instrumented wrappers for the permissions and userinfo clients.
//!
//! Wraps the raw REST/gRPC client behind each trait so that every backend
//! call (including the ones that retry internally) records a single
//! duration sample and, on failure, increments the error counter by
//! classified `AuthError` code. The wrappers know their backend + protocol
//! labels and forward them to the [`Metrics`] facade.
//!
//! An error that is a verdict rather than a fault — the userinfo service's
//! not-found answer, [`AuthError::UserNotFound`] — still records its duration
//! sample but is never counted as an auth error ([`auth_error_code`] returns
//! `None` for it). With authorization on, the userinfo fallback decorator sits
//! beneath this wrapper and turns that answer into an `Ok`, so a fallback is a
//! success sample here.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;

use crate::auth::permissions::PermissionsClient;
use crate::auth::userinfo::UserinfoClient;
use crate::auth::{auth_error_code, AuthError, AuthResult, Permission, UserInfo};
use crate::observability::metrics::SharedMetrics;

/// Metric-instrumented wrapper around a [`PermissionsClient`].
pub struct InstrumentedPermissionsClient {
    inner: Arc<dyn PermissionsClient>,
    metrics: SharedMetrics,
    backend: &'static str,  // "storage"
    protocol: &'static str, // "rest" | "grpc"
}

impl InstrumentedPermissionsClient {
    pub fn new(
        inner: Arc<dyn PermissionsClient>,
        metrics: SharedMetrics,
        backend: &'static str,
        protocol: &'static str,
    ) -> Self {
        Self {
            inner,
            metrics,
            backend,
            protocol,
        }
    }
}

#[async_trait]
impl PermissionsClient for InstrumentedPermissionsClient {
    async fn check_permission(
        &self,
        token: &str,
        path: &str,
        permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        let start = Instant::now();
        let result = self.inner.check_permission(token, path, permission).await;
        self.metrics
            .observe_auth_call(self.backend, self.protocol, start.elapsed());
        if let Some(code) = result.as_ref().err().and_then(auth_error_code) {
            self.metrics.inc_auth_error(self.backend, code);
        }
        result
    }
}

/// Metric-instrumented wrapper around a [`UserinfoClient`].
pub struct InstrumentedUserinfoClient {
    inner: Arc<dyn UserinfoClient>,
    metrics: SharedMetrics,
    backend: &'static str,  // "userinfo"
    protocol: &'static str, // "rest" | "grpc"
}

impl InstrumentedUserinfoClient {
    pub fn new(
        inner: Arc<dyn UserinfoClient>,
        metrics: SharedMetrics,
        backend: &'static str,
        protocol: &'static str,
    ) -> Self {
        Self {
            inner,
            metrics,
            backend,
            protocol,
        }
    }
}

#[async_trait]
impl UserinfoClient for InstrumentedUserinfoClient {
    async fn get_current_user(&self, token: &str) -> Result<UserInfo, AuthError> {
        let start = Instant::now();
        let result = self.inner.get_current_user(token).await;
        self.metrics
            .observe_auth_call(self.backend, self.protocol, start.elapsed());
        if let Some(code) = result.as_ref().err().and_then(auth_error_code) {
            self.metrics.inc_auth_error(self.backend, code);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::auth::permissions::mock::MockPermissionsClient;
    use crate::auth::permissions::PermissionsClient;
    use crate::auth::userinfo::mock::MockUserinfoClient;
    use crate::auth::userinfo::UserinfoClient;
    use crate::auth::{AuthError, AuthResult, Permission, UserInfo};
    use crate::observability::metrics::metrics_scrape_only;
    use crate::observability::metrics::test_util::{render, sample_value};

    use super::{InstrumentedPermissionsClient, InstrumentedUserinfoClient};

    /// Read a histogram's observation count (`<name>_count`) from rendered text.
    fn hist_count(text: &str, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_value(text, &format!("{name}_count"), labels).unwrap_or(0.0)
    }

    /// Read a counter cell, treating an absent (never-incremented) series as 0.
    fn counter(text: &str, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_value(text, name, labels).unwrap_or(0.0)
    }

    fn user() -> UserInfo {
        UserInfo {
            id: "u1".into(),
            display_name: "Jane".into(),
            email: None,
            username: "jane".into(),
        }
    }

    #[tokio::test]
    async fn permissions_success_records_duration_no_error_counter() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn PermissionsClient> =
            Arc::new(MockPermissionsClient::new(Ok(AuthResult::Allowed)));
        let wrapped = InstrumentedPermissionsClient::new(inner, metrics.clone(), "storage", "rest");

        wrapped
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap();

        let text = render(&metrics);
        assert_eq!(
            hist_count(
                &text,
                "live_edit_auth_call_duration_seconds",
                &[("backend", "storage"), ("protocol", "rest")],
            ),
            1.0
        );
        assert_eq!(
            counter(
                &text,
                "live_edit_auth_errors_total",
                &[("backend", "storage"), ("code", "timeout")],
            ),
            0.0
        );
    }

    #[tokio::test]
    async fn permissions_timeout_increments_error_counter() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn PermissionsClient> =
            Arc::new(MockPermissionsClient::new(Err(AuthError::Timeout)));
        let wrapped = InstrumentedPermissionsClient::new(inner, metrics.clone(), "storage", "grpc");

        let err = wrapped
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, AuthError::Timeout));

        assert_eq!(
            counter(
                &render(&metrics),
                "live_edit_auth_errors_total",
                &[("backend", "storage"), ("code", "timeout")],
            ),
            1.0
        );
    }

    #[tokio::test]
    async fn userinfo_invalid_response_increments_invalid_response_code() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn UserinfoClient> = Arc::new(MockUserinfoClient::new(Err(
            AuthError::InvalidResponse("bad".into()),
        )));
        let wrapped = InstrumentedUserinfoClient::new(inner, metrics.clone(), "userinfo", "rest");

        let _ = wrapped.get_current_user("tok").await;

        assert_eq!(
            counter(
                &render(&metrics),
                "live_edit_auth_errors_total",
                &[("backend", "userinfo"), ("code", "invalid_response")],
            ),
            1.0
        );
    }

    #[tokio::test]
    async fn userinfo_success_records_duration_no_error_counter() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn UserinfoClient> = Arc::new(MockUserinfoClient::new(Ok(user())));
        let wrapped = InstrumentedUserinfoClient::new(inner, metrics.clone(), "userinfo", "grpc");

        wrapped.get_current_user("tok").await.unwrap();

        assert_eq!(
            hist_count(
                &render(&metrics),
                "live_edit_auth_call_duration_seconds",
                &[("backend", "userinfo"), ("protocol", "grpc")],
            ),
            1.0
        );
    }

    // ── b.j5t item 6: per-variant `AuthError` metric label coverage ─────
    //
    // The b.nrt error-counter wiring gives every `AuthError` variant that is a
    // backend fault (`Timeout`, `Unreachable`, `InvalidResponse`) a distinct
    // `code` label via `auth_error_code(...)`. `UserNotFound` is a verdict,
    // not a fault: `auth_error_code` returns `None` and it is never counted
    // (pinned by `userinfo_user_not_found_records_duration_no_error_counter`).
    // Pre-b.j5t only `Timeout` was covered; below adds permissions-arm
    // coverage for `Unreachable` and `InvalidResponse` so every labelled
    // mapping has a guard.

    #[tokio::test]
    async fn permissions_unreachable_increments_unreachable_code() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn PermissionsClient> =
            Arc::new(MockPermissionsClient::new(Err(AuthError::Unreachable)));
        let wrapped = InstrumentedPermissionsClient::new(inner, metrics.clone(), "storage", "grpc");

        let _ = wrapped
            .check_permission("tok", "/p", Permission::Read)
            .await;

        let text = render(&metrics);
        assert_eq!(
            counter(
                &text,
                "live_edit_auth_errors_total",
                &[("backend", "storage"), ("code", "unreachable")],
            ),
            1.0
        );
        // Negative-control: the timeout label must NOT have been bumped.
        assert_eq!(
            counter(
                &text,
                "live_edit_auth_errors_total",
                &[("backend", "storage"), ("code", "timeout")],
            ),
            0.0
        );
    }

    /// `Ok(InvalidRequest)` — the storage service refusing the probe as a
    /// malformed request — is a verdict on the client's request, not an auth
    /// backend fault: it records a call-duration sample like any completed
    /// probe and bumps no `live_edit_auth_errors_total` cell, so a client
    /// sending bad base paths cannot inflate the backend-error rate operators
    /// alert on. The check is that the counter has no sample line at all, not
    /// that today's `code` labels read 0: the registry is this test's own, so
    /// an untouched counter renders no sample, and counting the verdict under
    /// any label — `invalid_response` (where a 400 counted before it became a
    /// verdict) or one added later — produces one.
    #[tokio::test]
    async fn permissions_invalid_request_records_duration_no_error_counter() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn PermissionsClient> =
            Arc::new(MockPermissionsClient::new(Ok(AuthResult::InvalidRequest {
                backend_status: "HTTP 400",
            })));
        let wrapped = InstrumentedPermissionsClient::new(inner, metrics.clone(), "storage", "rest");

        let result = wrapped
            .check_permission("tok", "load-test/perf", Permission::Read)
            .await;
        assert!(
            matches!(result, Ok(AuthResult::InvalidRequest { .. })),
            "the wrapper must pass the verdict through unchanged; got {result:?}"
        );

        let text = render(&metrics);
        assert_eq!(
            hist_count(
                &text,
                "live_edit_auth_call_duration_seconds",
                &[("backend", "storage"), ("protocol", "rest")],
            ),
            1.0
        );
        let error_samples: Vec<&str> = text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .filter(|line| line.split(['{', ' ']).next() == Some("live_edit_auth_errors_total"))
            .collect();
        assert!(
            error_samples.is_empty(),
            "InvalidRequest must not count as an auth error under any label; got {error_samples:?}"
        );
    }

    /// Every `live_edit_auth_errors_total` sample line in `text`, under any
    /// label. The registry is the test's own, so an untouched counter renders
    /// none.
    fn auth_error_samples(text: &str) -> Vec<&str> {
        text.lines()
            .filter(|line| !line.starts_with('#'))
            .filter(|line| line.split(['{', ' ']).next() == Some("live_edit_auth_errors_total"))
            .collect()
    }

    /// With authorization on, the fallback decorator sits beneath this
    /// wrapper: the userinfo service's not-found answer arrives here as an
    /// `Ok`, so it records one success duration sample and is never counted
    /// as an auth error, under any label.
    #[tokio::test]
    async fn userinfo_fallback_records_duration_no_error_counter() {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let metrics = Arc::new(metrics_scrape_only("t"));
        let backend: Arc<dyn UserinfoClient> =
            Arc::new(MockUserinfoClient::new(Err(AuthError::UserNotFound)));
        let fallback: Arc<dyn UserinfoClient> = Arc::new(crate::auth::UserinfoFallbackClient::new(
            backend,
            metrics.clone(),
        ));
        let wrapped =
            InstrumentedUserinfoClient::new(fallback, metrics.clone(), "userinfo", "rest");
        let app_only = format!(
            "{}.{}.",
            URL_SAFE_NO_PAD.encode(br#"{"alg":"none"}"#),
            URL_SAFE_NO_PAD.encode(br#"{"idtyp":"app","oid":"app-oid","sub":"app-oid"}"#),
        );

        let info = wrapped
            .get_current_user(&app_only)
            .await
            .expect("the decorator answers not-found from the claims");
        assert_eq!(info.display_name, "app-oid (app)");

        let text = render(&metrics);
        assert_eq!(
            hist_count(
                &text,
                "live_edit_auth_call_duration_seconds",
                &[("backend", "userinfo"), ("protocol", "rest")],
            ),
            1.0
        );
        let error_samples = auth_error_samples(&text);
        assert!(
            error_samples.is_empty(),
            "a fallback must not count as an auth error under any label; got {error_samples:?}"
        );
    }

    /// With authorization off there is no decorator and the not-found answer
    /// reaches this wrapper as `Err(UserNotFound)`: a verdict, not a backend
    /// fault, so it records its duration sample and still no error sample.
    #[tokio::test]
    async fn userinfo_user_not_found_records_duration_no_error_counter() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn UserinfoClient> =
            Arc::new(MockUserinfoClient::new(Err(AuthError::UserNotFound)));
        let wrapped = InstrumentedUserinfoClient::new(inner, metrics.clone(), "userinfo", "grpc");

        let err = wrapped.get_current_user("tok").await.unwrap_err();
        assert!(
            matches!(err, AuthError::UserNotFound),
            "the wrapper passes the error through unchanged; got {err:?}"
        );

        let text = render(&metrics);
        assert_eq!(
            hist_count(
                &text,
                "live_edit_auth_call_duration_seconds",
                &[("backend", "userinfo"), ("protocol", "grpc")],
            ),
            1.0
        );
        let error_samples = auth_error_samples(&text);
        assert!(
            error_samples.is_empty(),
            "UserNotFound must not count as an auth error under any label; got {error_samples:?}"
        );
    }

    #[tokio::test]
    async fn permissions_invalid_response_increments_invalid_response_code() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn PermissionsClient> = Arc::new(MockPermissionsClient::new(Err(
            AuthError::InvalidResponse("oh no".into()),
        )));
        let wrapped = InstrumentedPermissionsClient::new(inner, metrics.clone(), "storage", "rest");

        let _ = wrapped
            .check_permission("tok", "/p", Permission::Read)
            .await;

        assert_eq!(
            counter(
                &render(&metrics),
                "live_edit_auth_errors_total",
                &[("backend", "storage"), ("code", "invalid_response")],
            ),
            1.0
        );
    }

    #[tokio::test]
    async fn userinfo_unreachable_increments_unreachable_code() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn UserinfoClient> =
            Arc::new(MockUserinfoClient::new(Err(AuthError::Unreachable)));
        let wrapped = InstrumentedUserinfoClient::new(inner, metrics.clone(), "userinfo", "rest");

        let _ = wrapped.get_current_user("tok").await;

        assert_eq!(
            counter(
                &render(&metrics),
                "live_edit_auth_errors_total",
                &[("backend", "userinfo"), ("code", "unreachable")],
            ),
            1.0
        );
    }

    #[tokio::test]
    async fn userinfo_timeout_increments_timeout_code() {
        let metrics = Arc::new(metrics_scrape_only("t"));
        let inner: Arc<dyn UserinfoClient> =
            Arc::new(MockUserinfoClient::new(Err(AuthError::Timeout)));
        let wrapped = InstrumentedUserinfoClient::new(inner, metrics.clone(), "userinfo", "grpc");

        let _ = wrapped.get_current_user("tok").await;

        assert_eq!(
            counter(
                &render(&metrics),
                "live_edit_auth_errors_total",
                &[("backend", "userinfo"), ("code", "timeout")],
            ),
            1.0
        );
    }
}
