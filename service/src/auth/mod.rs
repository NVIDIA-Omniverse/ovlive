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

pub mod builder;
pub mod cache;
pub mod instrumented;
pub mod permissions;
pub mod rest;
pub mod service_token;
pub mod tls;
pub mod userinfo;
pub mod userinfo_fallback;
pub mod userinfo_jwt;

pub use builder::{build_permissions_client, build_userinfo_client};
pub use instrumented::{InstrumentedPermissionsClient, InstrumentedUserinfoClient};
pub use permissions::NoOpPermissionsClient;
pub use rest::{RestStorageServiceClient, RestUserinfoClient};
pub use userinfo_fallback::UserinfoFallbackClient;
pub use userinfo_jwt::JwtUserinfoClient;

use async_trait::async_trait;
use thiserror::Error;

/// Permission level for path-level authorization checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Permission {
    Read,
    Write,
}

/// Result returned by the user permissions service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    Allowed,
    /// The bearer token itself is missing, expired, or otherwise invalid
    /// (HTTP 401 / gRPC `Unauthenticated`). Clients should refresh the
    /// token and retry; distinct from `Denied`, which means the token is
    /// valid but the principal lacks permission.
    Unauthenticated,
    /// The bearer token is valid, but the principal is not permitted
    /// (HTTP 403 / gRPC `PermissionDenied`).
    Denied,
    /// The auth service returned 404 — the path does not exist.
    PathNotFound,
    /// The backend refused the request itself as malformed (HTTP 400 / 414,
    /// gRPC `InvalidArgument`). It is a verdict on the request, not on the
    /// principal or on whether the path exists, and it never grants access.
    ///
    /// `backend_status` names which answer the backend gave. It is a fixed
    /// string from a closed set — `"HTTP 400"`, `"HTTP 414"`,
    /// `"INVALID_ARGUMENT"` — written by the classifiers in `rest.rs` and
    /// [`GrpcStorageServiceClient`], so it can never carry backend text: the
    /// auth path's code-only, no-body posture (b.ohw H-auth-5) holds by
    /// construction.
    InvalidRequest {
        backend_status: &'static str,
    },
}

/// Errors from external auth service calls.
#[derive(Debug, Clone, Error)]
pub enum AuthError {
    #[error("auth service timed out")]
    Timeout,
    #[error("auth service unreachable")]
    Unreachable,
    #[error("auth service returned an invalid response: {0}")]
    InvalidResponse(String),
    /// The userinfo service answered that it holds no user for the bearer's
    /// principal (REST 404 with the userinfo service's `not_found` body, or
    /// gRPC `NOT_FOUND`). A valid answer, not a fault: it is not retried and
    /// never counted on `live_edit_auth_errors_total`. With authorization on,
    /// [`UserinfoFallbackClient`] answers it with the bearer's JWT claims.
    ///
    /// A unit variant on purpose: the outcome is the whole signal, so no
    /// backend body, status message or principal id can ride along into logs
    /// (b.ohw H-auth-5).
    #[error("userinfo service has no user for the caller")]
    UserNotFound,
}

/// User identity resolved from a bearer token via the userinfo service.
#[derive(Debug, Clone)]
pub struct UserInfo {
    pub id: String,
    pub display_name: String,
    pub email: Option<String>,
    pub username: String,
}

// ── Generated proto types for external services ──────────────────────────────

/// Storage service (FileObjectService) generated gRPC client.
pub mod storage_service_proto {
    tonic::include_proto!("nvidia.omniverse.storage.fileobject.v1beta");
}

/// Userinfo service generated gRPC client.
pub mod userinfo_proto {
    tonic::include_proto!("nvidia.omniverse.userinfo.v1alpha");
}

// ── Retry helper for transient auth errors ──────────────────────────────────

use std::time::Duration;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, ClientTlsConfig};

/// Classify a `tonic::transport::Error` from a `Channel::connect()` attempt
/// into the appropriate `AuthError`. TLS handshake failures (e.g.
/// `AlertReceived(HandshakeFailure)`, untrusted cert, SNI mismatch) are
/// permanent and not retried; everything else stays `Unreachable` so the
/// retry loop covers genuinely transient TCP/DNS/timeout failures.
///
/// b.j5t item 1 (DRY follow-up): the underlying source-chain walker is the
/// shared [`tls::has_rustls_error_in_chain`] — pre-DRY this file owned a
/// bit-for-bit copy of `auth::rest::is_tls_error`'s walker, which invited
/// drift the moment a reqwest/hyper bump rearranged how `rustls::Error`
/// appears in the chain.
fn classify_grpc_connect_error(err: tonic::transport::Error) -> AuthError {
    if tls::has_rustls_error_in_chain(&err) {
        return AuthError::InvalidResponse(
            "auth backend TLS handshake failed (cert / trust mismatch — not retried)".to_owned(),
        );
    }
    AuthError::Unreachable
}

/// Classify a per-RPC `tonic::Status::Unavailable` whose underlying transport
/// error carries a `rustls::Error` in its source chain as a permanent TLS
/// failure (b.j5t item 1). Returns `Some(AuthError::InvalidResponse(...))`
/// when the status is a TLS error, `None` otherwise — callers fall through
/// to the standard `Unreachable` classification.
fn maybe_grpc_status_tls_error(status: &tonic::Status) -> Option<AuthError> {
    // tonic 0.14 surfaces transport-level handshake failures by storing the
    // originating `hyper::Error` (which wraps `rustls::Error`) in the
    // `Status::source()` chain. We walk that chain via the shared
    // `tls::has_rustls_error_in_chain` helper — same walker the REST path
    // uses to walk `reqwest::Error`.
    if tls::has_rustls_error_in_chain(status) {
        Some(AuthError::InvalidResponse(
            "auth backend TLS handshake failed (cert / trust mismatch — not retried)".to_owned(),
        ))
    } else {
        None
    }
}

/// Retry configuration for auth service calls.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the first). Minimum 1.
    pub max_attempts: u32,
    /// Initial delay before the first retry.
    pub initial_delay: Duration,
    /// Multiplier applied to the delay after each retry.
    pub backoff_factor: u32,
    /// Maximum delay between retries.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            backoff_factor: 2,
            max_delay: Duration::from_secs(1),
        }
    }
}

/// Fast-retry `RetryConfig` for tests: 1 ms initial delay, capped at 10 ms,
/// so exhaustion paths complete in well under a second. Shared helper used
/// by the auth-module unit tests and by `tests/auth_builder.rs`.
pub fn fast_retry(max_attempts: u32) -> RetryConfig {
    RetryConfig {
        max_attempts,
        initial_delay: Duration::from_millis(1),
        backoff_factor: 2,
        max_delay: Duration::from_millis(10),
    }
}

/// Returns `true` if the error is transient and the call should be retried.
pub(crate) fn is_transient(err: &AuthError) -> bool {
    matches!(err, AuthError::Timeout | AuthError::Unreachable)
}

/// Classify an [`AuthError`] into a short metric-label code. Used as the
/// `code` label for `live_edit_auth_errors_total`.
///
/// `None` means the error is a verdict, not a backend fault, and is not
/// counted at all: [`AuthError::UserNotFound`] is a valid "no such
/// principal" answer, the userinfo twin of [`AuthResult::PathNotFound`]
/// (SDD §14.2), whether or not authorization is on. The match stays
/// exhaustive so a future variant must pick a code or `None` here.
pub(crate) fn auth_error_code(err: &AuthError) -> Option<&'static str> {
    match err {
        AuthError::Timeout => Some("timeout"),
        AuthError::Unreachable => Some("unreachable"),
        AuthError::InvalidResponse(_) => Some("invalid_response"),
        AuthError::UserNotFound => None,
    }
}

/// Execute `f` with retry on transient [`AuthError`]s using exponential backoff
/// with full jitter.
///
/// **Jitter (b.nrt).** The naive exponential backoff (delay doubling each
/// attempt) deterministically synchronizes concurrent callers: a transient
/// outage that fails N in-flight auth RPCs makes all N retry at exactly the
/// same wall-clock moments, amplifying load into the failing dependency at
/// the worst possible time. Full jitter (`uniform(0, delay)`) spreads retries
/// uniformly across the backoff window and is the recommended AWS
/// Architecture-Blog pattern for this failure mode.
///
/// TODO(b.nrt follow-up): add an optional circuit breaker in front of this
/// retry loop so repeated failures can fast-fail the call instead of burning
/// the full retry budget on every request during a known-down dependency.
pub(crate) async fn retry_on_transient<T, F, Fut>(
    config: &RetryConfig,
    f: F,
) -> Result<T, AuthError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, AuthError>>,
{
    use rand::Rng;

    let max = config.max_attempts.max(1);
    let mut delay = config.initial_delay;

    // The loop always exits via `return`: a successful attempt returns `Ok`,
    // a non-transient error returns `Err`, and the final transient attempt
    // (where `attempt + 1 == max`) falls into the wildcard `Err(e) =>` arm
    // because the guard `attempt + 1 < max` no longer holds. There is no
    // path that exits the for-loop normally, so the post-loop fallback
    // `Err(last_err.unwrap_or(...))` that lived here pre-b.ohw was dead
    // code (b.ohw medium).
    for attempt in 0..max {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) if is_transient(&e) && attempt + 1 < max => {
                // Full jitter: pick a uniform random delay in [0, delay].
                // Zero-length delays collapse to a zero sleep rather than
                // panicking on the empty range.
                let upper_ms = delay.as_millis() as u64;
                let sleep_ms = if upper_ms == 0 {
                    0
                } else {
                    rand::rng().random_range(0..=upper_ms)
                };
                let sleep_for = Duration::from_millis(sleep_ms);

                tracing::warn!(
                    attempt = attempt + 1,
                    max_attempts = max,
                    error = %e,
                    backoff_window_ms = upper_ms,
                    sleep_ms,
                    "transient auth error, retrying with jitter"
                );
                tokio::time::sleep(sleep_for).await;
                // Advance the backoff *window* (not the actual sleep); the
                // next retry samples full jitter from [0, new_delay].
                delay = (delay * config.backoff_factor).min(config.max_delay);
            }
            Err(e) => return Err(e),
        }
    }

    // Unreachable: `max.max(1)` guarantees at least one iteration, and every
    // iteration returns. `unreachable!` is the right tool — the previous
    // `Err(last_err.unwrap_or(...))` masked any logic regression in the
    // transient classifier as a silent `Err(Unreachable)` instead of a panic.
    unreachable!("retry_on_transient loop exits via return on every iteration")
}

// ── Real gRPC client: Storage Service (permissions via Stat) ─────────────────

/// gRPC client that checks path-level access by calling the storage service's
/// Stat RPC. Both Read and Write permissions map to the same Stat call — the
/// storage service checks "can this user access this path?" without
/// distinguishing read vs write.
pub struct GrpcStorageServiceClient {
    client: storage_service_proto::file_object_service_client::FileObjectServiceClient<Channel>,
    timeout: Duration,
    retry_config: RetryConfig,
}

impl GrpcStorageServiceClient {
    pub async fn connect(
        endpoint: &str,
        timeout_ms: u64,
        retry_config: RetryConfig,
    ) -> Result<Self, AuthError> {
        let (uri, use_tls) = normalize_endpoint(endpoint)?;
        let mut builder = Channel::from_shared(uri)
            .map_err(|e| AuthError::InvalidResponse(format!("invalid endpoint: {e}")))?
            // b.j5t item 3: cap the connect handshake (DNS + TCP + TLS) at
            // `auth_timeout_ms` so a DNS hang or unreachable backend at
            // startup fails inside the configured budget instead of blocking
            // indefinitely. Per-RPC deadlines still apply once connected.
            .connect_timeout(Duration::from_millis(timeout_ms));
        if use_tls {
            builder = builder
                .tls_config(ClientTlsConfig::new().with_native_roots())
                .map_err(|e| AuthError::InvalidResponse(format!("tls config: {e}")))?;
        }
        // b.j5t item 1: route connect failures through the TLS-aware
        // classifier so a handshake error short-circuits to permanent
        // `InvalidResponse` rather than getting retried as `Unreachable`.
        let channel = builder
            .connect()
            .await
            .map_err(classify_grpc_connect_error)?;

        Ok(Self {
            client: storage_service_proto::file_object_service_client::FileObjectServiceClient::new(
                channel,
            ),
            timeout: Duration::from_millis(timeout_ms),
            retry_config,
        })
    }
}

/// Public wrapper around [`normalize_endpoint`]'s validation step, used by
/// `Config::from_env` to reject bare hostnames at startup (b.ohw C-6 /
/// H-auth-1). Returns the human-readable failure reason on rejection.
///
/// Intentionally discards the (uri, use_tls) tuple — config parsing only
/// cares whether the value is valid; the auth-client constructors run
/// `normalize_endpoint` again at connect time to obtain the normalized form.
/// Calling the same function from both paths means a value that passes here
/// is guaranteed to pass at connect.
pub fn validate_auth_endpoint_uri(endpoint: &str) -> Result<(), String> {
    normalize_endpoint(endpoint)
        .map(|_| ())
        .map_err(|e| match e {
            AuthError::InvalidResponse(msg) => msg,
            // Other variants are unreachable from `normalize_endpoint`, which only
            // ever produces `InvalidResponse` on failure. Render them defensively.
            other => other.to_string(),
        })
}

/// Normalize a user-provided auth endpoint into a URI tonic / reqwest can
/// consume, returning the URI together with whether the channel must use TLS.
///
/// Fail-closed posture (b.ohw C-6 / H-auth-1):
/// - Bare hostnames (`storage.prod.internal`) are REJECTED with
///   `AuthError::InvalidResponse`. The pre-b.ohw helper silently rewrote
///   them to `http://...`, so an operator that omitted the scheme by accident
///   would downgrade the auth channel to plaintext — bearer tokens then
///   transit in the clear. The new contract requires an explicit scheme so
///   the operator's intent is unambiguous.
/// - The port-substring `contains(":443")` was replaced with proper
///   right-most-`:` parsing followed by exact equality. Hosts deployed on
///   ports 4430..4439, 44300..44399, etc. used to be silently TLS-flipped
///   into a connection failure with a confusing SNI error.
///
/// Accepted forms (case-sensitive scheme):
/// - `https://host[:port]` → unchanged, TLS on.
/// - `http://host[:port]` → unchanged, TLS off (UNLESS the explicit port is
///   exactly 443, in which case TLS is forced and the scheme is rewritten
///   to `https://` to match what tonic expects). This preserves the existing
///   behavior where `http://host:443` was treated as a TLS endpoint, but now
///   uses exact-equality port parsing rather than a string-contains check.
///
/// Returning `https://...` ensures tonic derives the correct SNI name for
/// the TLS handshake.
pub(crate) fn normalize_endpoint(endpoint: &str) -> Result<(String, bool), AuthError> {
    let trimmed = endpoint.trim();
    if trimmed.is_empty() {
        return Err(AuthError::InvalidResponse(
            "auth endpoint must not be empty".to_owned(),
        ));
    }

    let (scheme_is_https, after_scheme) = if let Some(rest) = trimmed.strip_prefix("https://") {
        (true, rest)
    } else if let Some(rest) = trimmed.strip_prefix("http://") {
        (false, rest)
    } else {
        return Err(AuthError::InvalidResponse(format!(
            "auth endpoint {endpoint:?} must include an explicit scheme (https:// or http://); \
             bare hostnames are rejected to prevent a silent plaintext downgrade"
        )));
    };

    // `after_scheme` looks like `host[:port][/path...]`. Extract the
    // authority (everything before the first `/`) so we can pull the port
    // out without matching a `:443` that lives inside a path.
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
    if authority.is_empty() {
        return Err(AuthError::InvalidResponse(format!(
            "auth endpoint {endpoint:?} is missing a host"
        )));
    }

    // Parse the port (right-most ':' so an IPv6 literal `[::1]:443` still
    // splits correctly — bracketed hosts terminate after `]`).
    let explicit_port_443 = if authority.starts_with('[') {
        // IPv6 literal: `[host]:port` — split on the `:` that follows `]`.
        match authority.rfind("]:") {
            Some(idx) => authority[idx + 2..] == *"443",
            None => false,
        }
    } else if let Some((_host, port)) = authority.rsplit_once(':') {
        port == "443"
    } else {
        false
    };

    let use_tls = scheme_is_https || explicit_port_443;
    let normalized = if use_tls {
        format!("https://{after_scheme}")
    } else {
        format!("http://{after_scheme}")
    };
    Ok((normalized, use_tls))
}

#[async_trait]
impl permissions::PermissionsClient for GrpcStorageServiceClient {
    async fn check_permission(
        &self,
        token: &str,
        path: &str,
        _permission: Permission, // Read and Write both map to Stat
    ) -> Result<AuthResult, AuthError> {
        let client = self.client.clone();
        let timeout = self.timeout;
        let path_owned = path.to_owned();
        let bearer = format!("Bearer {token}");
        let bearer_value = MetadataValue::try_from(&bearer)
            .map_err(|e| AuthError::InvalidResponse(format!("invalid token: {e}")))?;

        retry_on_transient(&self.retry_config, || {
            let mut client = client.clone();
            let mut request = tonic::Request::new(storage_service_proto::StatRequest {
                resource_address: path_owned.clone(),
            });
            request
                .metadata_mut()
                .insert("authorization", bearer_value.clone());
            request.set_timeout(timeout);

            async move {
                match client.stat(request).await {
                    Ok(_) => Ok(AuthResult::Allowed),
                    Err(status) => match status.code() {
                        tonic::Code::NotFound => Ok(AuthResult::PathNotFound),
                        tonic::Code::PermissionDenied => Ok(AuthResult::Denied),
                        tonic::Code::Unauthenticated => Ok(AuthResult::Unauthenticated),
                        // The storage service refused the Stat request as
                        // malformed — the gRPC twin of REST's 400 / 414 in
                        // `rest::classify_status`. A verdict, not a fault:
                        // `Ok`, so it is neither retried nor counted as an
                        // auth error, and it never allows. The status is the
                        // fixed string, never `status.message()` (b.ohw
                        // H-auth-5). `OutOfRange` deliberately stays on the
                        // wildcard below: a ≤4 KiB address in a protobuf
                        // field has no range to exceed, so that code would
                        // mean a contract break.
                        tonic::Code::InvalidArgument => Ok(AuthResult::InvalidRequest {
                            backend_status: "INVALID_ARGUMENT",
                        }),
                        tonic::Code::DeadlineExceeded => Err(AuthError::Timeout),
                        // b.j5t item 1: a `tonic::Status::Unavailable` with
                        // a `rustls::Error` in its source chain is a TLS
                        // failure mid-RPC (e.g. mid-stream renegotiation, or
                        // a backend swapping out its cert without grace).
                        // Permanent, not retried — mirrors the REST path's
                        // `is_tls_error` short-circuit. `ResourceExhausted`
                        // never carries a rustls error, so we only branch
                        // on `Unavailable`.
                        tonic::Code::Unavailable => maybe_grpc_status_tls_error(&status)
                            .map(Err)
                            .unwrap_or(Err(AuthError::Unreachable)),
                        tonic::Code::ResourceExhausted => Err(AuthError::Unreachable),
                        // b.ohw medium: classify gRPC `Aborted` and `Internal` as
                        // transient so the retry loop picks them up — REST
                        // already retries 429/500/503 and inconsistent
                        // classification leaks through to operators as
                        // "REST recovers, gRPC doesn't" during a brief
                        // backend blip.
                        tonic::Code::Aborted | tonic::Code::Internal => Err(AuthError::Unreachable),
                        // b.ohw H-auth-5: drop `status.message()` from the
                        // wrapped error string. A buggy or malicious backend
                        // that echoes the bearer token into status.message()
                        // would otherwise leak it through `ServiceError::Internal`
                        // into operator logs. Keep only the gRPC code, which
                        // is enough signal to triage.
                        code => Err(AuthError::InvalidResponse(format!(
                            "auth backend returned {code}"
                        ))),
                    },
                }
            }
        })
        .await
    }
}

// ── Real gRPC client: Userinfo Service ───────────────────────────────────────

/// gRPC client that resolves a bearer token to user identity by calling the
/// userinfo service's GetCurrentUser RPC.
pub struct GrpcUserinfoClient {
    client: userinfo_proto::user_info_service_client::UserInfoServiceClient<Channel>,
    timeout: Duration,
    retry_config: RetryConfig,
}

impl GrpcUserinfoClient {
    pub async fn connect(
        endpoint: &str,
        timeout_ms: u64,
        retry_config: RetryConfig,
    ) -> Result<Self, AuthError> {
        let (uri, use_tls) = normalize_endpoint(endpoint)?;
        let mut builder = Channel::from_shared(uri)
            .map_err(|e| AuthError::InvalidResponse(format!("invalid endpoint: {e}")))?
            // b.j5t item 3: bound connect handshake at `auth_timeout_ms`
            // (see matching comment in `GrpcStorageServiceClient::connect`).
            .connect_timeout(Duration::from_millis(timeout_ms));
        if use_tls {
            builder = builder
                .tls_config(ClientTlsConfig::new().with_native_roots())
                .map_err(|e| AuthError::InvalidResponse(format!("tls config: {e}")))?;
        }
        // b.j5t item 1: TLS-aware connect-error classification.
        let channel = builder
            .connect()
            .await
            .map_err(classify_grpc_connect_error)?;

        Ok(Self {
            client: userinfo_proto::user_info_service_client::UserInfoServiceClient::new(channel),
            timeout: Duration::from_millis(timeout_ms),
            retry_config,
        })
    }
}

#[async_trait]
impl userinfo::UserinfoClient for GrpcUserinfoClient {
    async fn get_current_user(&self, token: &str) -> Result<UserInfo, AuthError> {
        let client = self.client.clone();
        let timeout = self.timeout;
        let bearer = format!("Bearer {token}");
        let bearer_value = MetadataValue::try_from(&bearer)
            .map_err(|e| AuthError::InvalidResponse(format!("invalid token: {e}")))?;

        retry_on_transient(&self.retry_config, || {
            let mut client = client.clone();
            let mut request = tonic::Request::new(userinfo_proto::GetCurrentUserRequest {});
            request
                .metadata_mut()
                .insert("authorization", bearer_value.clone());
            request.set_timeout(timeout);

            async move {
                let response = client.get_current_user(request).await.map_err(|status| {
                    match status.code() {
                        // Not worded as a verdict on the caller's token: see
                        // the matching 401/403 arm in the REST userinfo
                        // client (`auth::rest`).
                        tonic::Code::Unauthenticated => AuthError::InvalidResponse(
                            "userinfo service rejected the bearer (UNAUTHENTICATED)".to_owned(),
                        ),
                        // The userinfo service holds no user for the bearer's
                        // principal: the gRPC twin of REST's 404 with the
                        // `not_found` body. Unlike REST, no body-shape check
                        // is needed to tell it from a route fault — an
                        // unknown method answers `UNIMPLEMENTED`, so
                        // `NOT_FOUND` can only be that answer. Never built
                        // from `status.message()` (b.ohw H-auth-5).
                        tonic::Code::NotFound => AuthError::UserNotFound,
                        tonic::Code::DeadlineExceeded => AuthError::Timeout,
                        // b.j5t item 1: TLS-aware per-RPC error classification
                        // — see the matching comment in `stat` above.
                        tonic::Code::Unavailable => {
                            maybe_grpc_status_tls_error(&status).unwrap_or(AuthError::Unreachable)
                        }
                        tonic::Code::ResourceExhausted => AuthError::Unreachable,
                        // b.ohw medium: classify Aborted / Internal as
                        // transient — see the matching comment in the
                        // storage-service `stat` arm above.
                        tonic::Code::Aborted | tonic::Code::Internal => AuthError::Unreachable,
                        // b.ohw H-auth-5: drop `status.message()` from
                        // the wrapped string — keep only the code.
                        code => AuthError::InvalidResponse(format!("auth backend returned {code}")),
                    }
                })?;

                let user = response.into_inner().user.ok_or_else(|| {
                    AuthError::InvalidResponse("response missing user field".to_owned())
                })?;

                Ok(UserInfo {
                    id: user.id,
                    display_name: user.display_name,
                    email: user.email,
                    username: user.username,
                })
            }
        })
        .await
    }
}

// ── Stubs (test only) ────────────────────────────────────────────────────────

/// Stub permissions client that always returns an error.
/// Only used in tests — production uses GrpcStorageServiceClient.
///
/// b.ohw low: gated `#[cfg(test)]` because `pub` test-only types in the
/// library API surface invite accidental wiring into production paths.
#[cfg(test)]
pub struct StubPermissionsClient;

#[cfg(test)]
#[async_trait]
impl permissions::PermissionsClient for StubPermissionsClient {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        Err(AuthError::InvalidResponse(
            "permissions client not configured".to_owned(),
        ))
    }
}

/// Stub userinfo client that always returns an error.
/// Only used in tests — production uses GrpcUserinfoClient.
#[cfg(test)]
pub struct StubUserinfoClient;

#[cfg(test)]
#[async_trait]
impl userinfo::UserinfoClient for StubUserinfoClient {
    async fn get_current_user(&self, _token: &str) -> Result<UserInfo, AuthError> {
        Err(AuthError::InvalidResponse(
            "userinfo client not configured".to_owned(),
        ))
    }
}

#[cfg(test)]
mod grpc_tests {
    #![allow(clippy::result_large_err)]

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use tonic::{Request, Response, Status};

    use crate::auth::permissions::PermissionsClient;
    use crate::auth::storage_service_proto::file_object_service_server::{
        FileObjectService, FileObjectServiceServer,
    };
    use crate::auth::storage_service_proto::*;
    use crate::auth::userinfo::UserinfoClient;
    use crate::auth::userinfo_proto::user_info_service_server::{
        UserInfoService, UserInfoServiceServer,
    };
    use crate::auth::userinfo_proto::*;
    use crate::auth::{
        AuthError, AuthResult, GrpcStorageServiceClient, GrpcUserinfoClient, Permission,
        RetryConfig,
    };

    // ── Mock FileObjectService ──────────────────────────────────────────────

    /// Captured metadata from the most recent request to the mock storage service.
    #[derive(Debug, Clone, Default)]
    struct CapturedRequest {
        authorization: Option<String>,
        resource_address: Option<String>,
    }

    /// A mock FileObjectService that returns a configurable response from `stat`
    /// and captures request metadata for assertion.
    #[derive(Clone)]
    struct MockStorageService {
        stat_result: Arc<dyn Fn() -> Result<StatResponse, Status> + Send + Sync>,
        captured: Arc<Mutex<CapturedRequest>>,
    }

    impl MockStorageService {
        fn returning_ok() -> Self {
            Self {
                stat_result: Arc::new(|| {
                    Ok(StatResponse {
                        resource_info: None,
                    })
                }),
                captured: Arc::new(Mutex::new(CapturedRequest::default())),
            }
        }

        fn returning_status(code: tonic::Code) -> Self {
            Self {
                stat_result: Arc::new(move || Err(Status::new(code, "mock error"))),
                captured: Arc::new(Mutex::new(CapturedRequest::default())),
            }
        }

        fn capturing() -> Self {
            Self {
                stat_result: Arc::new(|| {
                    Ok(StatResponse {
                        resource_info: None,
                    })
                }),
                captured: Arc::new(Mutex::new(CapturedRequest::default())),
            }
        }

        /// Returns `code` for the first `fail_count` calls, then Ok.
        fn failing_then_ok(code: tonic::Code, fail_count: usize) -> (Self, Arc<AtomicUsize>) {
            Self::failing_with_message_then_ok(code, "transient", fail_count)
        }

        /// Returns `code` with status message `message` for the first
        /// `fail_count` calls, then Ok.
        fn failing_with_message_then_ok(
            code: tonic::Code,
            message: &'static str,
            fail_count: usize,
        ) -> (Self, Arc<AtomicUsize>) {
            let counter = Arc::new(AtomicUsize::new(0));
            let counter_clone = Arc::clone(&counter);
            let svc = Self {
                stat_result: Arc::new(move || {
                    let n = counter_clone.fetch_add(1, Ordering::SeqCst);
                    if n < fail_count {
                        Err(Status::new(code, message))
                    } else {
                        Ok(StatResponse {
                            resource_info: None,
                        })
                    }
                }),
                captured: Arc::new(Mutex::new(CapturedRequest::default())),
            };
            (svc, counter)
        }
    }

    #[tonic::async_trait]
    impl FileObjectService for MockStorageService {
        type EnumerateStream = tokio_stream::Empty<Result<EnumerateResponse, Status>>;

        async fn enumerate(
            &self,
            _request: Request<EnumerateRequest>,
        ) -> Result<Response<Self::EnumerateStream>, Status> {
            unimplemented!()
        }

        async fn stat(
            &self,
            request: Request<StatRequest>,
        ) -> Result<Response<StatResponse>, Status> {
            // Capture metadata
            {
                let mut cap = self.captured.lock().unwrap();
                cap.authorization = request
                    .metadata()
                    .get("authorization")
                    .map(|v| v.to_str().unwrap().to_owned());
                cap.resource_address = Some(request.into_inner().resource_address);
            }
            (self.stat_result)().map(Response::new)
        }

        type ReadStream = tokio_stream::Empty<Result<ReadResponse, Status>>;

        async fn read(
            &self,
            _request: Request<ReadRequest>,
        ) -> Result<Response<Self::ReadStream>, Status> {
            unimplemented!()
        }

        type ReadFromAddressStream = tokio_stream::Empty<Result<ReadFromAddressResponse, Status>>;

        async fn read_from_address(
            &self,
            _request: Request<ReadFromAddressRequest>,
        ) -> Result<Response<Self::ReadFromAddressStream>, Status> {
            unimplemented!()
        }

        async fn fetch_write_type_info(
            &self,
            _request: Request<FetchWriteTypeInfoRequest>,
        ) -> Result<Response<FetchWriteTypeInfoResponse>, Status> {
            unimplemented!()
        }

        type WriteStream = tokio_stream::Empty<Result<WriteResponse, Status>>;

        async fn write(
            &self,
            _request: Request<tonic::Streaming<WriteRequest>>,
        ) -> Result<Response<Self::WriteStream>, Status> {
            unimplemented!()
        }

        async fn complete_redirect_upload(
            &self,
            _request: Request<CompleteRedirectUploadRequest>,
        ) -> Result<Response<CompleteRedirectUploadResponse>, Status> {
            unimplemented!()
        }

        async fn upload_part(
            &self,
            _request: Request<UploadPartRequest>,
        ) -> Result<Response<UploadPartResponse>, Status> {
            unimplemented!()
        }

        async fn complete_multipart_upload(
            &self,
            _request: Request<CompleteMultipartUploadRequest>,
        ) -> Result<Response<CompleteMultipartUploadResponse>, Status> {
            unimplemented!()
        }

        async fn abort_multipart_upload(
            &self,
            _request: Request<AbortMultipartUploadRequest>,
        ) -> Result<Response<AbortMultipartUploadResponse>, Status> {
            unimplemented!()
        }

        async fn delete(
            &self,
            _request: Request<DeleteRequest>,
        ) -> Result<Response<DeleteResponse>, Status> {
            unimplemented!()
        }
    }

    /// Start a mock storage server on a random port and return a connected client.
    async fn start_storage_mock(
        mock: MockStorageService,
    ) -> (GrpcStorageServiceClient, Arc<Mutex<CapturedRequest>>) {
        start_storage_mock_with_retry(mock, RetryConfig::default()).await
    }

    /// Start a mock storage server with a custom retry config.
    async fn start_storage_mock_with_retry(
        mock: MockStorageService,
        retry_config: RetryConfig,
    ) -> (GrpcStorageServiceClient, Arc<Mutex<CapturedRequest>>) {
        let captured = Arc::clone(&mock.captured);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(FileObjectServiceServer::new(mock))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        let endpoint = format!("http://{addr}");
        let start = std::time::Instant::now();
        let deadline = std::time::Duration::from_secs(5);
        let client = loop {
            match GrpcStorageServiceClient::connect(&endpoint, 5000, retry_config.clone()).await {
                Ok(c) => break c,
                Err(e) => {
                    if start.elapsed() >= deadline {
                        panic!("storage mock never became reachable within {deadline:?}: {e:?}");
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
            }
        };
        (client, captured)
    }

    // ── Mock UserInfoService ────────────────────────────────────────────────

    /// Captured metadata from the most recent request to the mock userinfo service.
    #[derive(Debug, Clone, Default)]
    struct CapturedUserinfoRequest {
        authorization: Option<String>,
    }

    /// A mock UserInfoService that returns a configurable response from
    /// `get_current_user` and captures request metadata for assertion.
    #[derive(Clone)]
    struct MockUserInfoService {
        result: Arc<dyn Fn() -> Result<GetCurrentUserResponse, Status> + Send + Sync>,
        captured: Arc<Mutex<CapturedUserinfoRequest>>,
    }

    impl MockUserInfoService {
        fn returning_user(user: User) -> Self {
            Self {
                result: Arc::new(move || {
                    Ok(GetCurrentUserResponse {
                        user: Some(user.clone()),
                    })
                }),
                captured: Arc::new(Mutex::new(CapturedUserinfoRequest::default())),
            }
        }

        fn returning_empty_response() -> Self {
            Self {
                result: Arc::new(|| Ok(GetCurrentUserResponse { user: None })),
                captured: Arc::new(Mutex::new(CapturedUserinfoRequest::default())),
            }
        }

        fn returning_status(code: tonic::Code) -> Self {
            Self {
                result: Arc::new(move || Err(Status::new(code, "mock error"))),
                captured: Arc::new(Mutex::new(CapturedUserinfoRequest::default())),
            }
        }

        fn capturing_with_user(user: User) -> Self {
            Self {
                result: Arc::new(move || {
                    Ok(GetCurrentUserResponse {
                        user: Some(user.clone()),
                    })
                }),
                captured: Arc::new(Mutex::new(CapturedUserinfoRequest::default())),
            }
        }

        /// Returns `code` for the first `fail_count` calls, then returns `user`.
        fn failing_then_user(
            code: tonic::Code,
            fail_count: usize,
            user: User,
        ) -> (Self, Arc<AtomicUsize>) {
            let counter = Arc::new(AtomicUsize::new(0));
            let counter_clone = Arc::clone(&counter);
            let svc = Self {
                result: Arc::new(move || {
                    let n = counter_clone.fetch_add(1, Ordering::SeqCst);
                    if n < fail_count {
                        Err(Status::new(code, "transient"))
                    } else {
                        Ok(GetCurrentUserResponse {
                            user: Some(user.clone()),
                        })
                    }
                }),
                captured: Arc::new(Mutex::new(CapturedUserinfoRequest::default())),
            };
            (svc, counter)
        }
    }

    #[tonic::async_trait]
    impl UserInfoService for MockUserInfoService {
        async fn get_current_user(
            &self,
            request: Request<GetCurrentUserRequest>,
        ) -> Result<Response<GetCurrentUserResponse>, Status> {
            {
                let mut cap = self.captured.lock().unwrap();
                cap.authorization = request
                    .metadata()
                    .get("authorization")
                    .map(|v| v.to_str().unwrap().to_owned());
            }
            (self.result)().map(Response::new)
        }

        async fn list_users(
            &self,
            _request: Request<ListUsersRequest>,
        ) -> Result<Response<ListUsersResponse>, Status> {
            unimplemented!()
        }

        async fn get_user(
            &self,
            _request: Request<GetUserRequest>,
        ) -> Result<Response<GetUserResponse>, Status> {
            unimplemented!()
        }

        async fn list_groups(
            &self,
            _request: Request<ListGroupsRequest>,
        ) -> Result<Response<ListGroupsResponse>, Status> {
            unimplemented!()
        }

        async fn get_group(
            &self,
            _request: Request<GetGroupRequest>,
        ) -> Result<Response<GetGroupResponse>, Status> {
            unimplemented!()
        }

        async fn list_group_members(
            &self,
            _request: Request<ListGroupMembersRequest>,
        ) -> Result<Response<ListGroupMembersResponse>, Status> {
            unimplemented!()
        }

        async fn list_user_groups(
            &self,
            _request: Request<ListUserGroupsRequest>,
        ) -> Result<Response<ListUserGroupsResponse>, Status> {
            unimplemented!()
        }

        async fn get_user_group(
            &self,
            _request: Request<GetUserGroupRequest>,
        ) -> Result<Response<GetUserGroupResponse>, Status> {
            unimplemented!()
        }

        async fn get_group_member(
            &self,
            _request: Request<GetGroupMemberRequest>,
        ) -> Result<Response<GetGroupMemberResponse>, Status> {
            unimplemented!()
        }
    }

    /// Start a mock userinfo server on a random port and return a connected client.
    async fn start_userinfo_mock(
        mock: MockUserInfoService,
    ) -> (GrpcUserinfoClient, Arc<Mutex<CapturedUserinfoRequest>>) {
        start_userinfo_mock_with_retry(mock, RetryConfig::default()).await
    }

    /// Start a mock userinfo server with a custom retry config.
    async fn start_userinfo_mock_with_retry(
        mock: MockUserInfoService,
        retry_config: RetryConfig,
    ) -> (GrpcUserinfoClient, Arc<Mutex<CapturedUserinfoRequest>>) {
        let captured = Arc::clone(&mock.captured);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(UserInfoServiceServer::new(mock))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        let endpoint = format!("http://{addr}");
        let start = std::time::Instant::now();
        let deadline = std::time::Duration::from_secs(5);
        let client = loop {
            match GrpcUserinfoClient::connect(&endpoint, 5000, retry_config.clone()).await {
                Ok(c) => break c,
                Err(e) => {
                    if start.elapsed() >= deadline {
                        panic!("userinfo mock never became reachable within {deadline:?}: {e:?}");
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
            }
        };
        (client, captured)
    }

    // ── Helper: make a full User proto message ──────────────────────────────

    fn make_full_user() -> User {
        User {
            id: "user-42".into(),
            display_name: "Jane Doe".into(),
            given_name: Some("Jane".into()),
            family_name: Some("Doe".into()),
            email: Some("jane@example.com".into()),
            username: "jdoe".into(),
            active: Some(true),
            job_title: Some("Engineer".into()),
        }
    }

    // ── GrpcStorageServiceClient tests ──────────────────────────────────────

    #[tokio::test]
    async fn stat_ok_returns_allowed() {
        let mock = MockStorageService::returning_ok();
        let (client, _) = start_storage_mock(mock).await;

        let result = client
            .check_permission("test-token", "/scene/layer1", Permission::Read)
            .await
            .unwrap();

        assert_eq!(result, AuthResult::Allowed);
    }

    #[tokio::test]
    async fn stat_not_found_returns_path_not_found() {
        let mock = MockStorageService::returning_status(tonic::Code::NotFound);
        let (client, _) = start_storage_mock(mock).await;

        let result = client
            .check_permission("test-token", "/missing/path", Permission::Read)
            .await
            .unwrap();

        assert_eq!(result, AuthResult::PathNotFound);
    }

    #[tokio::test]
    async fn stat_permission_denied_returns_denied() {
        let mock = MockStorageService::returning_status(tonic::Code::PermissionDenied);
        let (client, _) = start_storage_mock(mock).await;

        let result = client
            .check_permission("bad-token", "/scene/layer1", Permission::Read)
            .await
            .unwrap();

        assert_eq!(result, AuthResult::Denied);
    }

    /// gRPC `Unauthenticated` (invalid/expired bearer) maps to
    /// `AuthResult::Unauthenticated`, distinct from `PermissionDenied`
    /// (which maps to `AuthResult::Denied`). Conflating them breaks
    /// token-refresh flows in standard gRPC clients.
    #[tokio::test]
    async fn stat_unauthenticated_returns_unauthenticated() {
        let mock = MockStorageService::returning_status(tonic::Code::Unauthenticated);
        let (client, _) = start_storage_mock(mock).await;

        let result = client
            .check_permission("expired-token", "/scene/layer1", Permission::Read)
            .await
            .unwrap();

        assert_eq!(result, AuthResult::Unauthenticated);
    }

    #[tokio::test]
    async fn stat_unavailable_returns_unreachable() {
        let mock = MockStorageService::returning_status(tonic::Code::Unavailable);
        let (client, _) = start_storage_mock(mock).await;

        let err = client
            .check_permission("test-token", "/scene/layer1", Permission::Read)
            .await
            .unwrap_err();

        assert!(matches!(err, AuthError::Unreachable));
    }

    #[tokio::test]
    async fn bearer_token_sent_in_metadata() {
        let mock = MockStorageService::capturing();
        let (client, captured) = start_storage_mock(mock).await;

        client
            .check_permission("my-secret-token", "/some/path", Permission::Read)
            .await
            .unwrap();

        let cap = captured.lock().unwrap();
        assert_eq!(cap.authorization.as_deref(), Some("Bearer my-secret-token"));
    }

    #[tokio::test]
    async fn read_and_write_both_call_stat() {
        // Permission::Read calls stat
        let mock_read = MockStorageService::capturing();
        let captured_read = Arc::clone(&mock_read.captured);
        let (client_read, _) = start_storage_mock(mock_read).await;

        client_read
            .check_permission("token-r", "/path/read", Permission::Read)
            .await
            .unwrap();

        {
            let cap_r = captured_read.lock().unwrap();
            assert_eq!(cap_r.resource_address.as_deref(), Some("/path/read"));
        }

        // Permission::Write also calls stat
        let mock_write = MockStorageService::capturing();
        let captured_write = Arc::clone(&mock_write.captured);
        let (client_write, _) = start_storage_mock(mock_write).await;

        client_write
            .check_permission("token-w", "/path/write", Permission::Write)
            .await
            .unwrap();

        let cap_w = captured_write.lock().unwrap();
        assert_eq!(cap_w.resource_address.as_deref(), Some("/path/write"));
    }

    // ── GrpcUserinfoClient tests ────────────────────────────────────────────

    #[tokio::test]
    async fn get_current_user_returns_user_info() {
        let user = make_full_user();
        let mock = MockUserInfoService::returning_user(user);
        let (client, _) = start_userinfo_mock(mock).await;

        let info = client.get_current_user("test-token").await.unwrap();

        assert_eq!(info.id, "user-42");
        assert_eq!(info.display_name, "Jane Doe");
        assert_eq!(info.email, Some("jane@example.com".to_owned()));
        assert_eq!(info.username, "jdoe");
    }

    #[tokio::test]
    async fn get_current_user_email_none() {
        let user = User {
            id: "user-99".into(),
            display_name: "No Email User".into(),
            given_name: None,
            family_name: None,
            email: None,
            username: "noemail".into(),
            active: None,
            job_title: None,
        };
        let mock = MockUserInfoService::returning_user(user);
        let (client, _) = start_userinfo_mock(mock).await;

        let info = client.get_current_user("test-token").await.unwrap();

        assert_eq!(info.id, "user-99");
        assert_eq!(info.display_name, "No Email User");
        assert_eq!(info.email, None);
        assert_eq!(info.username, "noemail");
    }

    #[tokio::test]
    async fn get_current_user_unauthenticated_returns_error() {
        let mock = MockUserInfoService::returning_status(tonic::Code::Unauthenticated);
        let (client, _) = start_userinfo_mock(mock).await;

        let err = client.get_current_user("bad-token").await.unwrap_err();

        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg)
                if msg == "userinfo service rejected the bearer (UNAUTHENTICATED)"),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn get_current_user_unavailable_returns_unreachable() {
        let mock = MockUserInfoService::returning_status(tonic::Code::Unavailable);
        let (client, _) = start_userinfo_mock(mock).await;

        let err = client.get_current_user("test-token").await.unwrap_err();

        assert!(matches!(err, AuthError::Unreachable));
    }

    #[tokio::test]
    async fn get_current_user_missing_user_field_returns_error() {
        let mock = MockUserInfoService::returning_empty_response();
        let (client, _) = start_userinfo_mock(mock).await;

        let err = client.get_current_user("test-token").await.unwrap_err();

        assert!(matches!(err, AuthError::InvalidResponse(msg) if msg.contains("missing user")));
    }

    #[tokio::test]
    async fn bearer_token_sent_in_userinfo_metadata() {
        let user = make_full_user();
        let mock = MockUserInfoService::capturing_with_user(user);
        let (client, captured) = start_userinfo_mock(mock).await;

        client.get_current_user("my-userinfo-token").await.unwrap();

        let cap = captured.lock().unwrap();
        assert_eq!(
            cap.authorization.as_deref(),
            Some("Bearer my-userinfo-token")
        );
    }

    // ── Retry behavior: Storage Service ────────────────────────────────────

    #[tokio::test]
    async fn storage_retries_unavailable_then_succeeds() {
        let (mock, counter) = MockStorageService::failing_then_ok(tonic::Code::Unavailable, 2);
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_storage_mock_with_retry(mock, retry_cfg).await;

        let result = client
            .check_permission("token", "/scene/layer1", Permission::Read)
            .await
            .unwrap();

        assert_eq!(result, AuthResult::Allowed);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn storage_retries_deadline_exceeded_then_succeeds() {
        let (mock, counter) = MockStorageService::failing_then_ok(tonic::Code::DeadlineExceeded, 1);
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_storage_mock_with_retry(mock, retry_cfg).await;

        let result = client
            .check_permission("token", "/scene/layer1", Permission::Read)
            .await
            .unwrap();

        assert_eq!(result, AuthResult::Allowed);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn storage_exhausts_retries_returns_last_error() {
        let mock = MockStorageService::returning_status(tonic::Code::Unavailable);
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_storage_mock_with_retry(mock, retry_cfg).await;

        let err = client
            .check_permission("token", "/scene/layer1", Permission::Read)
            .await
            .unwrap_err();

        assert!(matches!(err, AuthError::Unreachable));
    }

    #[tokio::test]
    async fn storage_does_not_retry_permission_denied() {
        // Use failing_then_ok so we can observe the call count
        let (mock, counter) = MockStorageService::failing_then_ok(tonic::Code::PermissionDenied, 3);
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_storage_mock_with_retry(mock, retry_cfg).await;

        // PermissionDenied maps to Ok(Denied), not an error — so the first call succeeds
        let result = client
            .check_permission("token", "/scene/layer1", Permission::Read)
            .await
            .unwrap();

        assert_eq!(result, AuthResult::Denied);
        // Only one call was made — no retry needed since it returned Ok
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    /// The storage service's `INVALID_ARGUMENT` on `Stat` is the gRPC twin of
    /// REST's 400: a verdict that the request is malformed. It must come back
    /// as `Ok(InvalidRequest { "INVALID_ARGUMENT" })` from one call (the mock
    /// would fail three times, so a retried or `InvalidResponse` mapping shows
    /// up as a count above 1 or an `Err`), and the backend's status message —
    /// here a token-shaped sentinel — must not reach the result (b.ohw
    /// H-auth-5).
    #[tokio::test]
    async fn storage_invalid_argument_returns_invalid_request_no_retry_no_message_leak() {
        let sentinel = super::status_message_scrubbing_tests::TOKEN_SENTINEL;
        let (mock, counter) = MockStorageService::failing_with_message_then_ok(
            tonic::Code::InvalidArgument,
            sentinel,
            3,
        );
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_storage_mock_with_retry(mock, retry_cfg).await;

        let result = client
            .check_permission("token", "load-test/perf", Permission::Read)
            .await;

        assert!(
            matches!(
                result,
                Ok(AuthResult::InvalidRequest {
                    backend_status: "INVALID_ARGUMENT"
                })
            ),
            "INVALID_ARGUMENT must be Ok(InvalidRequest {{ \"INVALID_ARGUMENT\" }}); \
             got {result:?}"
        );
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "INVALID_ARGUMENT is a verdict and must not be retried"
        );
        let rendered = format!("{result:?}");
        assert!(
            !rendered.contains(sentinel) && !rendered.contains("LEAKEDTOKEN"),
            "the backend's status message must not reach the result; got {rendered}"
        );
    }

    #[tokio::test]
    async fn storage_no_retry_with_max_attempts_one() {
        let mock = MockStorageService::returning_status(tonic::Code::Unavailable);
        let retry_cfg = RetryConfig {
            max_attempts: 1,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_storage_mock_with_retry(mock, retry_cfg).await;

        let err = client
            .check_permission("token", "/scene/layer1", Permission::Read)
            .await
            .unwrap_err();

        assert!(matches!(err, AuthError::Unreachable));
    }

    // ── Retry behavior: Userinfo Service ───────────────────────────────────

    #[tokio::test]
    async fn userinfo_retries_unavailable_then_succeeds() {
        let user = make_full_user();
        let (mock, counter) =
            MockUserInfoService::failing_then_user(tonic::Code::Unavailable, 2, user);
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_userinfo_mock_with_retry(mock, retry_cfg).await;

        let info = client.get_current_user("test-token").await.unwrap();

        assert_eq!(info.id, "user-42");
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn userinfo_retries_deadline_exceeded_then_succeeds() {
        let user = make_full_user();
        let (mock, counter) =
            MockUserInfoService::failing_then_user(tonic::Code::DeadlineExceeded, 1, user);
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_userinfo_mock_with_retry(mock, retry_cfg).await;

        let info = client.get_current_user("test-token").await.unwrap();

        assert_eq!(info.id, "user-42");
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn userinfo_exhausts_retries_returns_last_error() {
        let mock = MockUserInfoService::returning_status(tonic::Code::Unavailable);
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_userinfo_mock_with_retry(mock, retry_cfg).await;

        let err = client.get_current_user("test-token").await.unwrap_err();

        assert!(matches!(err, AuthError::Unreachable));
    }

    #[tokio::test]
    async fn userinfo_does_not_retry_unauthenticated() {
        // Unauthenticated maps to InvalidResponse, which is non-transient.
        // Use a counter-based mock to verify only one call is made.
        let user = make_full_user();
        let (mock, counter) =
            MockUserInfoService::failing_then_user(tonic::Code::Unauthenticated, 3, user);
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_userinfo_mock_with_retry(mock, retry_cfg).await;

        let err = client.get_current_user("test-token").await.unwrap_err();

        assert!(matches!(err, AuthError::InvalidResponse(_)));
        // Only one call — no retries for non-transient errors
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    /// gRPC `NOT_FOUND` is the userinfo service's "no user for this
    /// principal" answer: `UserNotFound`, not retried, and built without the
    /// status message (b.ohw H-auth-5) — a backend echoing the bearer there
    /// must not get it into the error.
    #[tokio::test]
    async fn userinfo_not_found_returns_user_not_found_not_retried_without_status_message() {
        const SENTINEL: &str = "Bearer eyJhbGciOiJIUzI1NiJ9.LEAKEDTOKEN.signature";
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_in_mock = Arc::clone(&calls);
        // Fails `NOT_FOUND` on every attempt the retry budget allows, so a
        // retry would show as a second call.
        let mock = MockUserInfoService {
            result: Arc::new(move || {
                calls_in_mock.fetch_add(1, Ordering::SeqCst);
                Err(Status::new(tonic::Code::NotFound, SENTINEL))
            }),
            captured: Arc::new(Mutex::new(CapturedUserinfoRequest::default())),
        };
        let retry_cfg = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };
        let (client, _) = start_userinfo_mock_with_retry(mock, retry_cfg).await;

        let err = client.get_current_user("test-token").await.unwrap_err();

        assert!(
            matches!(err, AuthError::UserNotFound),
            "gRPC NOT_FOUND must be UserNotFound, got {err:?}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "UserNotFound is an answer, not a fault: no retry"
        );
        let rendered = format!("{err:?} / {err}");
        assert!(
            !rendered.contains("LEAKEDTOKEN"),
            "the status message must not reach the error, got: {rendered}"
        );
    }
}

// ── Unit tests for retry_on_transient ───────────────────────────────────────

#[cfg(test)]
mod retry_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn is_transient_timeout() {
        assert!(is_transient(&AuthError::Timeout));
    }

    #[test]
    fn is_transient_unreachable() {
        assert!(is_transient(&AuthError::Unreachable));
    }

    #[test]
    fn is_not_transient_invalid_response() {
        assert!(!is_transient(&AuthError::InvalidResponse("bad".into())));
    }

    #[tokio::test]
    async fn retry_succeeds_on_first_attempt() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };

        let result: Result<&str, AuthError> = retry_on_transient(&config, || {
            let c = Arc::clone(&c);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok("ok")
            }
        })
        .await;

        assert_eq!(result.unwrap(), "ok");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_succeeds_after_transient_failures() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };

        let result: Result<&str, AuthError> = retry_on_transient(&config, || {
            let c = Arc::clone(&c);
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(AuthError::Timeout)
                } else {
                    Ok("recovered")
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), "recovered");
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_exhausts_all_attempts() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };

        let result: Result<&str, AuthError> = retry_on_transient(&config, || {
            let c = Arc::clone(&c);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(AuthError::Unreachable)
            }
        })
        .await;

        assert!(matches!(result, Err(AuthError::Unreachable)));
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_does_not_retry_non_transient() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };

        let result: Result<&str, AuthError> = retry_on_transient(&config, || {
            let c = Arc::clone(&c);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(AuthError::InvalidResponse("permanent".into()))
            }
        })
        .await;

        assert!(matches!(result, Err(AuthError::InvalidResponse(_))));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_with_max_attempts_zero_still_attempts_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let config = RetryConfig {
            max_attempts: 0,
            initial_delay: std::time::Duration::from_millis(1),
            ..Default::default()
        };

        let result: Result<&str, AuthError> = retry_on_transient(&config, || {
            let c = Arc::clone(&c);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok("ok")
            }
        })
        .await;

        assert_eq!(result.unwrap(), "ok");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}

// ── b.ohw C-6 + H-auth-1: normalize_endpoint regression tests ──────────────
//
// These guard the fail-closed posture: bare hostnames are rejected (so an
// operator dropped scheme can't silently downgrade to plaintext) and the
// `:443` detection uses exact-equality port parsing (so ports like 4430 and
// 44300 stay http, not silently flipped to TLS).

#[cfg(test)]
mod normalize_endpoint_tests {
    use super::*;

    /// Helper: assert that `endpoint` is rejected with an `InvalidResponse`
    /// error whose message matches `predicate`. The body of the error message
    /// is the operator-facing failure reason, so we assert on its substring
    /// rather than treating it as opaque.
    fn assert_rejected(endpoint: &str, predicate: impl Fn(&str) -> bool) {
        match normalize_endpoint(endpoint) {
            Err(AuthError::InvalidResponse(msg)) => {
                assert!(
                    predicate(&msg),
                    "rejection message did not match expectation; got: {msg}"
                );
            }
            other => panic!("expected InvalidResponse rejection for {endpoint:?}, got {other:?}"),
        }
    }

    // ── port-substring false-positive guard (H-auth-1) ──────────────────

    #[test]
    fn port_4430_is_http_not_https() {
        // The pre-fix `contains(":443")` substring test would flip port 4430
        // to TLS. The fix uses exact equality: port == "443".
        let (uri, use_tls) = normalize_endpoint("http://host.example.com:4430").unwrap();
        assert_eq!(uri, "http://host.example.com:4430");
        assert!(!use_tls, "host:4430 must NOT be classified as TLS");
    }

    #[test]
    fn port_44300_is_http_not_https() {
        // Five-digit port that contains "443" as a prefix. The pre-fix
        // substring check matched this and flipped to TLS, breaking plaintext
        // services on operator-chosen high ports.
        let (uri, use_tls) = normalize_endpoint("http://host.example.com:44300").unwrap();
        assert_eq!(uri, "http://host.example.com:44300");
        assert!(!use_tls, "host:44300 must NOT be classified as TLS");
    }

    #[test]
    fn port_4435_is_http_not_https() {
        // Adjacent port in the formerly-misclassified 4430..4439 range.
        let (_uri, use_tls) = normalize_endpoint("http://h:4435").unwrap();
        assert!(!use_tls);
    }

    #[test]
    fn explicit_port_443_on_http_scheme_upgrades_to_tls() {
        // Documented carve-out: `http://host:443` is treated as TLS and the
        // scheme is rewritten to https so tonic derives the correct SNI name.
        // This matches the pre-fix behavior with exact-equality port parsing.
        let (uri, use_tls) = normalize_endpoint("http://host.example.com:443").unwrap();
        assert!(use_tls, "exact port 443 must enable TLS");
        assert_eq!(uri, "https://host.example.com:443");
    }

    #[test]
    fn https_scheme_stays_https() {
        let (uri, use_tls) = normalize_endpoint("https://host.example.com:443").unwrap();
        assert!(use_tls);
        assert_eq!(uri, "https://host.example.com:443");
    }

    #[test]
    fn https_scheme_no_port_stays_https() {
        let (uri, use_tls) = normalize_endpoint("https://host.example.com").unwrap();
        assert!(use_tls);
        assert_eq!(uri, "https://host.example.com");
    }

    #[test]
    fn http_scheme_arbitrary_port_stays_http() {
        let (uri, use_tls) = normalize_endpoint("http://host.example.com:8080").unwrap();
        assert!(!use_tls);
        assert_eq!(uri, "http://host.example.com:8080");
    }

    #[test]
    fn port_443_inside_path_does_not_flip_tls() {
        // Pre-fix substring match would catch ":443" inside the path. The fix
        // splits authority from path before parsing the port.
        let (uri, use_tls) = normalize_endpoint("http://host.example.com/api/v:443/stat").unwrap();
        assert!(!use_tls, "':443' in the path must not flip TLS");
        assert_eq!(uri, "http://host.example.com/api/v:443/stat");
    }

    #[test]
    fn ipv6_literal_with_port_443_enables_tls() {
        // IPv6 authority is `[::1]:443` — the right-most-`:` heuristic alone
        // would split inside the address, so the helper has explicit
        // `[host]:port` handling. Verify the carve-out works for IPv6 too.
        let (uri, use_tls) = normalize_endpoint("http://[::1]:443").unwrap();
        assert!(use_tls);
        assert_eq!(uri, "https://[::1]:443");
    }

    #[test]
    fn ipv6_literal_with_port_4430_stays_http() {
        let (_uri, use_tls) = normalize_endpoint("http://[::1]:4430").unwrap();
        assert!(!use_tls);
    }

    // ── bare-hostname rejection (C-6) ───────────────────────────────────

    #[test]
    fn bare_hostname_rejected() {
        // The chosen behavior is rejection (b.ohw fix-order item 3): a bare
        // hostname could be silently rewritten to http:// pre-fix, so an
        // operator that pasted "storage.prod.internal" without a scheme would
        // route bearer tokens in the clear. Now it fails fast.
        assert_rejected("storage.prod.internal", |msg| {
            msg.contains("explicit scheme")
                && msg.contains("https://")
                && msg.contains("plaintext downgrade")
        });
    }

    #[test]
    fn bare_hostname_with_port_rejected() {
        // Even with a port, the absence of a scheme is rejected — port alone
        // cannot disambiguate operator intent (50051 could be plaintext gRPC
        // OR a TLS service depending on the operator).
        assert_rejected("storage.prod.internal:50051", |msg| {
            msg.contains("explicit scheme")
        });
    }

    #[test]
    fn empty_endpoint_rejected() {
        assert_rejected("", |msg| msg.contains("must not be empty"));
    }

    #[test]
    fn whitespace_only_endpoint_rejected() {
        // Pre-trim guard: a value of "   " trims to empty.
        assert_rejected("   ", |msg| msg.contains("must not be empty"));
    }

    #[test]
    fn ftp_scheme_rejected() {
        // Only http:// and https:// are accepted — alternative schemes are
        // rejected with the same "explicit scheme" message because they hit
        // the bare-hostname path (no `http://` or `https://` prefix).
        assert_rejected("ftp://host.example.com", |msg| {
            msg.contains("explicit scheme")
        });
    }

    #[test]
    fn uppercase_https_scheme_rejected_case_sensitive() {
        // The doc comment promises case-sensitive scheme matching. Operators
        // who paste "HTTPS://host" hit the bare-hostname path and get a clear
        // rejection rather than a silent fallback.
        assert_rejected("HTTPS://host.example.com", |msg| {
            msg.contains("explicit scheme")
        });
    }

    #[test]
    fn scheme_only_with_no_host_rejected() {
        assert_rejected("https://", |msg| msg.contains("missing a host"));
    }

    #[test]
    fn endpoint_is_trimmed_before_validation() {
        // Whitespace surrounding a valid value should be trimmed (b.3z7
        // medium parity), so " https://host " parses identically to
        // "https://host".
        let (uri, use_tls) = normalize_endpoint("  https://host.example.com  ").unwrap();
        assert!(use_tls);
        assert_eq!(uri, "https://host.example.com");
    }

    // ── validate_auth_endpoint_uri (C-6: config-time rejection) ─────────

    #[test]
    fn validate_auth_endpoint_uri_accepts_https() {
        validate_auth_endpoint_uri("https://host.example.com").unwrap();
    }

    #[test]
    fn validate_auth_endpoint_uri_accepts_http() {
        validate_auth_endpoint_uri("http://host.example.com:8080").unwrap();
    }

    #[test]
    fn validate_auth_endpoint_uri_rejects_bare_hostname() {
        // Mirror the auth-client constructor rejection at the config-validation
        // boundary so an operator dropped scheme is caught at startup, not
        // later when an auth call first attempts a connection.
        let err = validate_auth_endpoint_uri("storage.prod.internal").unwrap_err();
        assert!(
            err.contains("explicit scheme"),
            "validator should surface the same rejection text as normalize_endpoint, got: {err}"
        );
    }

    #[test]
    fn validate_auth_endpoint_uri_rejects_empty() {
        let err = validate_auth_endpoint_uri("").unwrap_err();
        assert!(err.contains("must not be empty"), "got: {err}");
    }
}

// ── b.ohw H-auth-5: status.message() scrubbing regression tests ─────────────
//
// A buggy or malicious backend that echoes a bearer token in `status.message()`
// would, pre-fix, leak it through the wrapped `AuthError::InvalidResponse`
// string and into operator logs / `ServiceError::Internal`. The fix keeps only
// the gRPC code (or HTTP code, REST side) in the wrapped message.

#[cfg(test)]
mod status_message_scrubbing_tests {
    use super::*;
    use crate::auth::permissions::PermissionsClient;
    use crate::auth::userinfo::UserinfoClient;
    use std::sync::Arc;
    use std::time::Duration;
    use tonic::{Request, Response, Status};

    use crate::auth::storage_service_proto::file_object_service_server::{
        FileObjectService, FileObjectServiceServer,
    };
    use crate::auth::storage_service_proto::*;
    use crate::auth::userinfo_proto::user_info_service_server::{
        UserInfoService, UserInfoServiceServer,
    };
    use crate::auth::userinfo_proto::*;

    /// Token-shaped sentinel that a buggy backend might echo into the status
    /// message. The scrubbing fix must prevent this from showing up anywhere
    /// in the surfaced error. Shared with `grpc_tests`' `INVALID_ARGUMENT`
    /// test, which checks the same no-leak contract on the `Ok` arm.
    pub(super) const TOKEN_SENTINEL: &str = "Bearer eyJhbGciOiJIUzI1NiJ9.LEAKEDTOKEN.signature";

    /// Storage mock that returns a non-mapped gRPC code with a token-like
    /// status message. Code chosen: `OutOfRange` — falls through the explicit
    /// arms in the `stat` handler (NotFound/Permission*/Unauthenticated/
    /// InvalidArgument/DeadlineExceeded/Unavailable/ResourceExhausted/Aborted/
    /// Internal) and hits the wildcard arm where the message scrubbing
    /// matters. It also pins that `OutOfRange` stays a contract break
    /// (`InvalidResponse`) rather than joining `InvalidArgument` as a
    /// malformed-request verdict.
    #[derive(Clone)]
    struct LeakyStorageService;

    #[tonic::async_trait]
    impl FileObjectService for LeakyStorageService {
        type EnumerateStream = tokio_stream::Empty<Result<EnumerateResponse, Status>>;
        async fn enumerate(
            &self,
            _r: Request<EnumerateRequest>,
        ) -> Result<Response<Self::EnumerateStream>, Status> {
            unimplemented!()
        }
        async fn stat(&self, _r: Request<StatRequest>) -> Result<Response<StatResponse>, Status> {
            // OutOfRange falls into the wildcard arm of the storage-service
            // status match where status.message() previously leaked through.
            Err(Status::new(tonic::Code::OutOfRange, TOKEN_SENTINEL))
        }
        type ReadStream = tokio_stream::Empty<Result<ReadResponse, Status>>;
        async fn read(
            &self,
            _r: Request<ReadRequest>,
        ) -> Result<Response<Self::ReadStream>, Status> {
            unimplemented!()
        }
        type ReadFromAddressStream = tokio_stream::Empty<Result<ReadFromAddressResponse, Status>>;
        async fn read_from_address(
            &self,
            _r: Request<ReadFromAddressRequest>,
        ) -> Result<Response<Self::ReadFromAddressStream>, Status> {
            unimplemented!()
        }
        async fn fetch_write_type_info(
            &self,
            _r: Request<FetchWriteTypeInfoRequest>,
        ) -> Result<Response<FetchWriteTypeInfoResponse>, Status> {
            unimplemented!()
        }
        type WriteStream = tokio_stream::Empty<Result<WriteResponse, Status>>;
        async fn write(
            &self,
            _r: Request<tonic::Streaming<WriteRequest>>,
        ) -> Result<Response<Self::WriteStream>, Status> {
            unimplemented!()
        }
        async fn complete_redirect_upload(
            &self,
            _r: Request<CompleteRedirectUploadRequest>,
        ) -> Result<Response<CompleteRedirectUploadResponse>, Status> {
            unimplemented!()
        }
        async fn upload_part(
            &self,
            _r: Request<UploadPartRequest>,
        ) -> Result<Response<UploadPartResponse>, Status> {
            unimplemented!()
        }
        async fn complete_multipart_upload(
            &self,
            _r: Request<CompleteMultipartUploadRequest>,
        ) -> Result<Response<CompleteMultipartUploadResponse>, Status> {
            unimplemented!()
        }
        async fn abort_multipart_upload(
            &self,
            _r: Request<AbortMultipartUploadRequest>,
        ) -> Result<Response<AbortMultipartUploadResponse>, Status> {
            unimplemented!()
        }
        async fn delete(
            &self,
            _r: Request<DeleteRequest>,
        ) -> Result<Response<DeleteResponse>, Status> {
            unimplemented!()
        }
    }

    #[derive(Clone)]
    struct LeakyUserinfoService;

    #[tonic::async_trait]
    impl UserInfoService for LeakyUserinfoService {
        async fn get_current_user(
            &self,
            _r: Request<GetCurrentUserRequest>,
        ) -> Result<Response<GetCurrentUserResponse>, Status> {
            // Same wildcard-arm idea as the storage mock above.
            Err(Status::new(tonic::Code::OutOfRange, TOKEN_SENTINEL))
        }
        async fn list_users(
            &self,
            _r: Request<ListUsersRequest>,
        ) -> Result<Response<ListUsersResponse>, Status> {
            unimplemented!()
        }
        async fn get_user(
            &self,
            _r: Request<GetUserRequest>,
        ) -> Result<Response<GetUserResponse>, Status> {
            unimplemented!()
        }
        async fn list_groups(
            &self,
            _r: Request<ListGroupsRequest>,
        ) -> Result<Response<ListGroupsResponse>, Status> {
            unimplemented!()
        }
        async fn get_group(
            &self,
            _r: Request<GetGroupRequest>,
        ) -> Result<Response<GetGroupResponse>, Status> {
            unimplemented!()
        }
        async fn list_group_members(
            &self,
            _r: Request<ListGroupMembersRequest>,
        ) -> Result<Response<ListGroupMembersResponse>, Status> {
            unimplemented!()
        }
        async fn list_user_groups(
            &self,
            _r: Request<ListUserGroupsRequest>,
        ) -> Result<Response<ListUserGroupsResponse>, Status> {
            unimplemented!()
        }
        async fn get_user_group(
            &self,
            _r: Request<GetUserGroupRequest>,
        ) -> Result<Response<GetUserGroupResponse>, Status> {
            unimplemented!()
        }
        async fn get_group_member(
            &self,
            _r: Request<GetGroupMemberRequest>,
        ) -> Result<Response<GetGroupMemberResponse>, Status> {
            unimplemented!()
        }
    }

    async fn start_leaky_storage() -> GrpcStorageServiceClient {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(FileObjectServiceServer::new(LeakyStorageService))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        let endpoint = format!("http://{addr}");
        let start = std::time::Instant::now();
        let deadline = Duration::from_secs(5);
        loop {
            match GrpcStorageServiceClient::connect(&endpoint, 5000, fast_retry(1)).await {
                Ok(c) => break c,
                Err(e) => {
                    if start.elapsed() >= deadline {
                        panic!("storage mock not reachable: {e:?}");
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
        }
    }

    async fn start_leaky_userinfo() -> GrpcUserinfoClient {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(UserInfoServiceServer::new(LeakyUserinfoService))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        let endpoint = format!("http://{addr}");
        let start = std::time::Instant::now();
        let deadline = Duration::from_secs(5);
        loop {
            match GrpcUserinfoClient::connect(&endpoint, 5000, fast_retry(1)).await {
                Ok(c) => break c,
                Err(e) => {
                    if start.elapsed() >= deadline {
                        panic!("userinfo mock not reachable: {e:?}");
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
        }
    }

    /// Storage gRPC arm: a non-mapped gRPC code whose status message contains
    /// a token-shaped string must produce an `InvalidResponse` whose wrapped
    /// message does NOT contain the token text.
    #[tokio::test]
    async fn grpc_storage_status_message_does_not_leak_through_invalid_response() {
        let client = start_leaky_storage().await;
        let err = client
            .check_permission("real-token", "/p", Permission::Read)
            .await
            .unwrap_err();

        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(
                    !msg.contains(TOKEN_SENTINEL),
                    "wrapped error must not include the backend's status.message() — \
                     a buggy backend echoing a bearer token would otherwise leak it. \
                     Got: {msg}"
                );
                assert!(
                    !msg.contains("LEAKEDTOKEN"),
                    "even partial token text must be absent — got: {msg}"
                );
                // The contract is "auth backend returned <code>" — assert the
                // code is present so we know we're hitting the scrubbed format.
                assert!(
                    msg.contains("auth backend returned"),
                    "scrubbed message format should be 'auth backend returned <code>', got: {msg}"
                );
            }
            other => panic!("expected InvalidResponse for non-mapped gRPC code, got {other:?}"),
        }
    }

    /// Userinfo gRPC arm: same scrubbing contract as the storage arm.
    #[tokio::test]
    async fn grpc_userinfo_status_message_does_not_leak_through_invalid_response() {
        let client = start_leaky_userinfo().await;
        let err = client.get_current_user("real-token").await.unwrap_err();

        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(
                    !msg.contains(TOKEN_SENTINEL),
                    "wrapped userinfo error must not include status.message() — got: {msg}"
                );
                assert!(!msg.contains("LEAKEDTOKEN"), "got: {msg}");
                assert!(
                    msg.contains("auth backend returned"),
                    "expected scrubbed format 'auth backend returned <code>', got: {msg}"
                );
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }
    }

    /// End-to-end: AuthError::InvalidResponse with a token-shaped wrapped
    /// message must NOT flow through to `Status::internal`'s message body —
    /// `ServiceError::Internal` redacts on conversion to gRPC `Status`. This
    /// is the last line of defense even if the wrapped string somehow does
    /// contain a token (defense-in-depth with the scrubbing fix above).
    #[test]
    fn auth_error_invalid_response_redacts_in_status_internal() {
        use crate::error::ServiceError;
        let auth_err = AuthError::InvalidResponse(format!("hypothetical leak: {TOKEN_SENTINEL}"));
        let svc_err: ServiceError = auth_err.into();
        let status: tonic::Status = svc_err.into();
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(
            !status.message().contains(TOKEN_SENTINEL),
            "Status::internal must not leak the wrapped detail to clients — got: {}",
            status.message()
        );
        assert!(
            !status.message().contains("LEAKEDTOKEN"),
            "must not leak even partial token bytes — got: {}",
            status.message()
        );
        // Sanity-check the redacted contract: callers see a generic message.
        assert_eq!(status.message(), "internal error");
    }

    /// Wraps the leaky service through the Arc<dyn PermissionsClient> path —
    /// the same path production handlers take. Confirms the scrubbing applies
    /// regardless of whether the call goes through the inherent method or the
    /// trait dispatch.
    #[tokio::test]
    async fn dyn_permissions_client_scrubs_status_message() {
        let client: Arc<dyn PermissionsClient> = Arc::new(start_leaky_storage().await);
        let err = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap_err();
        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(!msg.contains(TOKEN_SENTINEL));
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }
    }

    /// Wraps the leaky userinfo service through the Arc<dyn UserinfoClient>
    /// path. Mirrors the test above.
    #[tokio::test]
    async fn dyn_userinfo_client_scrubs_status_message() {
        let client: Arc<dyn UserinfoClient> = Arc::new(start_leaky_userinfo().await);
        let err = client.get_current_user("tok").await.unwrap_err();
        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(!msg.contains(TOKEN_SENTINEL));
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }
    }
}

// ── b.ohw medium: gRPC ResourceExhausted/Aborted/Internal retry parity ─────
//
// Pre-fix REST retried HTTP 429/500/503 but the gRPC arm only retried
// Unavailable + DeadlineExceeded. A backend hiccup that surfaced as Aborted
// or Internal was not retried, so REST and gRPC paths diverged during a
// transient outage. The fix classifies these three codes as Unreachable so
// they participate in the retry loop.

#[cfg(test)]
mod grpc_transient_classification_tests {
    use super::*;
    use crate::auth::permissions::PermissionsClient;
    use crate::auth::userinfo::UserinfoClient;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tonic::{Request, Response, Status};

    use crate::auth::storage_service_proto::file_object_service_server::{
        FileObjectService, FileObjectServiceServer,
    };
    use crate::auth::storage_service_proto::*;
    use crate::auth::userinfo_proto::user_info_service_server::{
        UserInfoService, UserInfoServiceServer,
    };
    use crate::auth::userinfo_proto::*;

    /// Storage mock that fails the first `fail_count` calls with `code` then
    /// returns Ok. Reusable across the three retried-code parametric cases.
    #[derive(Clone)]
    struct FlakyStorage {
        code: tonic::Code,
        fail_count: usize,
        counter: Arc<AtomicUsize>,
    }

    #[tonic::async_trait]
    impl FileObjectService for FlakyStorage {
        type EnumerateStream = tokio_stream::Empty<Result<EnumerateResponse, Status>>;
        async fn enumerate(
            &self,
            _r: Request<EnumerateRequest>,
        ) -> Result<Response<Self::EnumerateStream>, Status> {
            unimplemented!()
        }
        async fn stat(&self, _r: Request<StatRequest>) -> Result<Response<StatResponse>, Status> {
            let n = self.counter.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                Err(Status::new(self.code, "transient"))
            } else {
                Ok(Response::new(StatResponse {
                    resource_info: None,
                }))
            }
        }
        type ReadStream = tokio_stream::Empty<Result<ReadResponse, Status>>;
        async fn read(
            &self,
            _r: Request<ReadRequest>,
        ) -> Result<Response<Self::ReadStream>, Status> {
            unimplemented!()
        }
        type ReadFromAddressStream = tokio_stream::Empty<Result<ReadFromAddressResponse, Status>>;
        async fn read_from_address(
            &self,
            _r: Request<ReadFromAddressRequest>,
        ) -> Result<Response<Self::ReadFromAddressStream>, Status> {
            unimplemented!()
        }
        async fn fetch_write_type_info(
            &self,
            _r: Request<FetchWriteTypeInfoRequest>,
        ) -> Result<Response<FetchWriteTypeInfoResponse>, Status> {
            unimplemented!()
        }
        type WriteStream = tokio_stream::Empty<Result<WriteResponse, Status>>;
        async fn write(
            &self,
            _r: Request<tonic::Streaming<WriteRequest>>,
        ) -> Result<Response<Self::WriteStream>, Status> {
            unimplemented!()
        }
        async fn complete_redirect_upload(
            &self,
            _r: Request<CompleteRedirectUploadRequest>,
        ) -> Result<Response<CompleteRedirectUploadResponse>, Status> {
            unimplemented!()
        }
        async fn upload_part(
            &self,
            _r: Request<UploadPartRequest>,
        ) -> Result<Response<UploadPartResponse>, Status> {
            unimplemented!()
        }
        async fn complete_multipart_upload(
            &self,
            _r: Request<CompleteMultipartUploadRequest>,
        ) -> Result<Response<CompleteMultipartUploadResponse>, Status> {
            unimplemented!()
        }
        async fn abort_multipart_upload(
            &self,
            _r: Request<AbortMultipartUploadRequest>,
        ) -> Result<Response<AbortMultipartUploadResponse>, Status> {
            unimplemented!()
        }
        async fn delete(
            &self,
            _r: Request<DeleteRequest>,
        ) -> Result<Response<DeleteResponse>, Status> {
            unimplemented!()
        }
    }

    #[derive(Clone)]
    struct FlakyUserinfo {
        code: tonic::Code,
        fail_count: usize,
        counter: Arc<AtomicUsize>,
    }

    #[tonic::async_trait]
    impl UserInfoService for FlakyUserinfo {
        async fn get_current_user(
            &self,
            _r: Request<GetCurrentUserRequest>,
        ) -> Result<Response<GetCurrentUserResponse>, Status> {
            let n = self.counter.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_count {
                Err(Status::new(self.code, "transient"))
            } else {
                Ok(Response::new(GetCurrentUserResponse {
                    user: Some(User {
                        id: "u".into(),
                        display_name: "U".into(),
                        given_name: None,
                        family_name: None,
                        email: None,
                        username: "u".into(),
                        active: None,
                        job_title: None,
                    }),
                }))
            }
        }
        async fn list_users(
            &self,
            _r: Request<ListUsersRequest>,
        ) -> Result<Response<ListUsersResponse>, Status> {
            unimplemented!()
        }
        async fn get_user(
            &self,
            _r: Request<GetUserRequest>,
        ) -> Result<Response<GetUserResponse>, Status> {
            unimplemented!()
        }
        async fn list_groups(
            &self,
            _r: Request<ListGroupsRequest>,
        ) -> Result<Response<ListGroupsResponse>, Status> {
            unimplemented!()
        }
        async fn get_group(
            &self,
            _r: Request<GetGroupRequest>,
        ) -> Result<Response<GetGroupResponse>, Status> {
            unimplemented!()
        }
        async fn list_group_members(
            &self,
            _r: Request<ListGroupMembersRequest>,
        ) -> Result<Response<ListGroupMembersResponse>, Status> {
            unimplemented!()
        }
        async fn list_user_groups(
            &self,
            _r: Request<ListUserGroupsRequest>,
        ) -> Result<Response<ListUserGroupsResponse>, Status> {
            unimplemented!()
        }
        async fn get_user_group(
            &self,
            _r: Request<GetUserGroupRequest>,
        ) -> Result<Response<GetUserGroupResponse>, Status> {
            unimplemented!()
        }
        async fn get_group_member(
            &self,
            _r: Request<GetGroupMemberRequest>,
        ) -> Result<Response<GetGroupMemberResponse>, Status> {
            unimplemented!()
        }
    }

    async fn run_storage_with_code(code: tonic::Code, fail_count: usize) -> Arc<AtomicUsize> {
        let counter = Arc::new(AtomicUsize::new(0));
        let svc = FlakyStorage {
            code,
            fail_count,
            counter: Arc::clone(&counter),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(FileObjectServiceServer::new(svc))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        let endpoint = format!("http://{addr}");
        let start = std::time::Instant::now();
        let client = loop {
            match GrpcStorageServiceClient::connect(&endpoint, 5000, fast_retry(5)).await {
                Ok(c) => break c,
                Err(_) if start.elapsed() < Duration::from_secs(5) => {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(e) => panic!("mock not reachable: {e:?}"),
            }
        };
        client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .expect("retried code should eventually succeed");
        counter
    }

    async fn run_userinfo_with_code(code: tonic::Code, fail_count: usize) -> Arc<AtomicUsize> {
        let counter = Arc::new(AtomicUsize::new(0));
        let svc = FlakyUserinfo {
            code,
            fail_count,
            counter: Arc::clone(&counter),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(UserInfoServiceServer::new(svc))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        let endpoint = format!("http://{addr}");
        let start = std::time::Instant::now();
        let client = loop {
            match GrpcUserinfoClient::connect(&endpoint, 5000, fast_retry(5)).await {
                Ok(c) => break c,
                Err(_) if start.elapsed() < Duration::from_secs(5) => {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(e) => panic!("mock not reachable: {e:?}"),
            }
        };
        client
            .get_current_user("tok")
            .await
            .expect("retried code should eventually succeed");
        counter
    }

    #[tokio::test]
    async fn storage_resource_exhausted_is_retried_then_succeeds() {
        // REST parity with HTTP 429: ResourceExhausted is the gRPC analog
        // and pre-fix wasn't retried — the fix classifies it as transient.
        let counter = run_storage_with_code(tonic::Code::ResourceExhausted, 2).await;
        assert_eq!(
            counter.load(Ordering::SeqCst),
            3,
            "ResourceExhausted should be retried (REST 429 parity)"
        );
    }

    #[tokio::test]
    async fn storage_aborted_is_retried_then_succeeds() {
        // Aborted classifies as transient under the b.ohw fix; pre-fix it
        // fell through to InvalidResponse and exited after one attempt.
        let counter = run_storage_with_code(tonic::Code::Aborted, 2).await;
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn storage_internal_is_retried_then_succeeds() {
        // gRPC Code::Internal classifies as transient (REST 500 parity);
        // pre-fix it fell through to InvalidResponse and was not retried.
        let counter = run_storage_with_code(tonic::Code::Internal, 2).await;
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn userinfo_resource_exhausted_is_retried_then_succeeds() {
        let counter = run_userinfo_with_code(tonic::Code::ResourceExhausted, 2).await;
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn userinfo_aborted_is_retried_then_succeeds() {
        let counter = run_userinfo_with_code(tonic::Code::Aborted, 2).await;
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn userinfo_internal_is_retried_then_succeeds() {
        let counter = run_userinfo_with_code(tonic::Code::Internal, 2).await;
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    // ── b.j5t item 1: gRPC TLS handshake parity with REST ────────────────────
    //
    // Pre-fix the gRPC auth client classified a TLS handshake failure as
    // `AuthError::Unreachable` (transient) and retried it for the full retry
    // budget. The fix downcasts the connect-error chain (and per-RPC
    // `tonic::Status` errors with `Unavailable` + rustls in the source) to
    // `AuthError::InvalidResponse(...)`, mirroring `is_tls_error` on the REST
    // path. Below: a plain-TCP listener that sends a well-formed TLS fatal
    // alert when the client attempts a TLS handshake, identical to the REST
    // tls test's shape.

    /// Spawn a plain-TCP listener that accepts a single connection and
    /// writes a fatal TLS alert (`handshake_failure`) in response to the
    /// client's ClientHello. rustls processes this as
    /// `rustls::Error::AlertReceived(HandshakeFailure)` — the exact shape
    /// the `is_tls_error_in_chain` walker downcasts to.
    async fn spawn_grpc_tls_alert_server() -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => return,
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 5];
                    let _ = socket.read_exact(&mut buf).await;
                    const TLS_ALERT: [u8; 7] = [0x15, 0x03, 0x03, 0x00, 0x02, 0x02, 0x28];
                    let _ = socket.write_all(&TLS_ALERT).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        addr
    }

    #[tokio::test]
    async fn grpc_storage_tls_handshake_failure_classified_as_invalid_response_not_retried() {
        let addr = spawn_grpc_tls_alert_server().await;
        let endpoint = format!("https://127.0.0.1:{}", addr.port());

        // 5 retries with `fast_retry`'s 1ms initial / 10ms max would take
        // ~50ms+ if classification regressed back to `Unreachable`. The
        // fix issues exactly one attempt.
        let start = std::time::Instant::now();
        let err = GrpcStorageServiceClient::connect(&endpoint, 2_000, fast_retry(5))
            .await
            .err()
            .expect("TLS handshake to garbage server should fail");
        let elapsed = start.elapsed();

        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(
                    msg.contains("TLS handshake failed"),
                    "gRPC TLS errors must surface as InvalidResponse with TLS-tagged message, got: {msg}"
                );
                assert!(
                    msg.contains("not retried"),
                    "operator-facing message must say 'not retried', got: {msg}"
                );
            }
            other => {
                panic!("TLS handshake failure must classify as InvalidResponse, got {other:?}")
            }
        }

        // Connect-time TLS classification uses no retry loop (the connect
        // happens once at construction); the assertion above already
        // proves the classifier downcast fired. The elapsed-time bound
        // here guards against a regression that lengthens the connect
        // path with internal retries.
        assert!(
            elapsed < Duration::from_millis(1500),
            "gRPC TLS handshake failure should not retry — elapsed {elapsed:?} suggests a retry storm"
        );
    }

    #[tokio::test]
    async fn grpc_userinfo_tls_handshake_failure_classified_as_invalid_response_not_retried() {
        let addr = spawn_grpc_tls_alert_server().await;
        let endpoint = format!("https://127.0.0.1:{}", addr.port());

        let err = GrpcUserinfoClient::connect(&endpoint, 2_000, fast_retry(5))
            .await
            .err()
            .expect("TLS handshake to garbage server should fail");
        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(
                    msg.contains("TLS handshake failed"),
                    "userinfo gRPC TLS errors must classify as InvalidResponse, got: {msg}"
                );
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }
    }

    // ── b.j5t item 1 follow-up: direct unit coverage on per-RPC TLS classifier ─
    //
    // The two TLS regression tests above (`grpc_storage_tls_handshake_failure_…`
    // / `grpc_userinfo_tls_handshake_failure_…`) only exercise the connect-time
    // path via `classify_grpc_connect_error`. The per-RPC arms in
    // `GrpcStorageServiceClient::stat` and `GrpcUserinfoClient::get_current_user`
    // invoke `maybe_grpc_status_tls_error` for `tonic::Code::Unavailable`
    // statuses, and that helper is structurally uncovered — a swapped argument,
    // inverted boolean, or branch swap inside it would slip past every
    // existing test. The tests below pin its wiring directly by constructing
    // a synthetic `tonic::Status` with a `rustls::Error` planted in the
    // public `Status::set_source` chain (the same chain `is_tls_error_in_chain`
    // walks via `std::error::Error::source()`) and asserting the Some/None
    // branches.

    #[test]
    fn maybe_grpc_status_tls_error_unavailable_with_rustls_source_returns_invalid_response() {
        // Plant a rustls::Error directly in Status::source(). The walker's
        // first downcast (`e.downcast_ref::<rustls::Error>()`) must fire and
        // the helper must return Some(InvalidResponse(...)). Status::source()
        // returns the inner Arc<dyn Error> set by `set_source`, so the chain
        // looks like: Status (no rustls) → rustls::Error (downcast hits).
        let mut status = tonic::Status::unavailable("transport error");
        let tls_err = rustls::Error::General("synthetic handshake failure".to_owned());
        status.set_source(std::sync::Arc::new(tls_err));

        match maybe_grpc_status_tls_error(&status) {
            Some(AuthError::InvalidResponse(msg)) => {
                assert!(
                    msg.contains("TLS handshake"),
                    "per-RPC TLS classifier must surface a TLS-tagged message, got: {msg}"
                );
                assert!(
                    msg.contains("not retried"),
                    "operator-facing message must say 'not retried', got: {msg}"
                );
            }
            other => panic!(
                "Status::unavailable with rustls in source chain must classify as Some(InvalidResponse), got {other:?}"
            ),
        }
    }

    #[test]
    fn maybe_grpc_status_tls_error_unavailable_without_rustls_returns_none() {
        // Negative branch: an Unavailable status with no source set (or a
        // source that is not a rustls::Error) must return None so the caller
        // falls through to AuthError::Unreachable and the standard retry
        // path. Guards against a regression that inverts the boolean inside
        // `is_tls_error_in_chain` or short-circuits the helper unconditionally.
        let status_no_source = tonic::Status::unavailable("plain transient blip");
        assert!(
            maybe_grpc_status_tls_error(&status_no_source).is_none(),
            "Status with no source chain must classify as None (transient), not InvalidResponse"
        );

        // Also cover the "source is set but not a rustls::Error" shape — a
        // generic io::Error with a non-rustls inner error. Both the top-level
        // downcast and the io::Error::get_ref() descent must miss.
        let mut status_non_tls_source = tonic::Status::unavailable("transport error");
        let inner_io = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "peer closed");
        status_non_tls_source.set_source(std::sync::Arc::new(inner_io));
        assert!(
            maybe_grpc_status_tls_error(&status_non_tls_source).is_none(),
            "Status with non-rustls source must classify as None (transient), not InvalidResponse"
        );
    }

    #[test]
    fn maybe_grpc_status_tls_error_rustls_nested_inside_io_error_returns_invalid_response() {
        // Reproduces the real-world shape: hyper/tonic wrap rustls::Error
        // inside an io::Error, which hides the inner error from
        // `Error::source()` and requires the `io::Error::get_ref()` descent
        // in `is_tls_error_in_chain` to find it. Without this branch the
        // downcast on rustls never fires for the actual on-wire shape.
        let mut status = tonic::Status::unavailable("transport error");
        let tls_err = rustls::Error::General("wrapped handshake failure".to_owned());
        let io_err = std::io::Error::new(std::io::ErrorKind::InvalidData, tls_err);
        status.set_source(std::sync::Arc::new(io_err));

        match maybe_grpc_status_tls_error(&status) {
            Some(AuthError::InvalidResponse(msg)) => {
                assert!(
                    msg.contains("TLS handshake"),
                    "rustls nested inside io::Error must still classify as TLS, got: {msg}"
                );
            }
            other => panic!(
                "rustls wrapped in io::Error must classify as Some(InvalidResponse) via get_ref() descent, got {other:?}"
            ),
        }
    }

    // ── b.j5t item 3: gRPC connect_timeout bounds DNS hang ──────────────────
    //
    // Pre-fix `Channel::from_shared(...).connect().await` had no
    // `.connect_timeout()` configured, so a DNS hang at startup could
    // block indefinitely. The fix sets `.connect_timeout(auth_timeout_ms)`
    // so the connect path fails inside the configured budget.

    #[tokio::test]
    async fn grpc_storage_connect_timeout_bounds_unreachable_endpoint() {
        // 192.0.2.0/24 (TEST-NET-1, RFC 5737) is a documentation-only range
        // that drops on the public internet. Connecting to an address in
        // this range hangs at the TCP SYN stage on most networks until the
        // OS gives up — exactly the shape `connect_timeout` is meant to
        // bound.
        let endpoint = "http://192.0.2.1:50051".to_owned();
        // 500 ms is generous enough to absorb scheduling jitter on busy
        // CI runners but short enough that a hang is unambiguous.
        let timeout_ms = 500u64;

        let start = std::time::Instant::now();
        let result = GrpcStorageServiceClient::connect(&endpoint, timeout_ms, fast_retry(1)).await;
        let elapsed = start.elapsed();

        // The exact error class isn't load-bearing (could surface as
        // `Unreachable` or `InvalidResponse` depending on the platform's
        // SYN-drop classification), but it MUST fail within the timeout
        // budget plus a generous slack — pre-fix this call would block
        // for the OS default ~30s+.
        assert!(
            result.is_err(),
            "connect to TEST-NET-1 address must fail, got Ok"
        );
        assert!(
            elapsed < Duration::from_millis(timeout_ms * 4),
            "connect_timeout did not bound the unreachable-endpoint connect — elapsed {elapsed:?}"
        );
    }
}
