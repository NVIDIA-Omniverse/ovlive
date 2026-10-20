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

//! REST-based auth service clients (storage service + userinfo).

use std::time::Duration;

use async_trait::async_trait;
use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use reqwest::StatusCode;
use serde::Deserialize;

use crate::auth::{
    normalize_endpoint, permissions, retry_on_transient, userinfo, AuthError, AuthResult,
    Permission, RetryConfig, UserInfo,
};

/// Maximum userinfo response body size (b.j5t item 2). The userinfo DTO is
/// four short JSON strings — well under 1 KiB in practice. A 64 KiB ceiling
/// leaves three orders of magnitude of headroom for future fields while
/// preventing a malicious or broken backend from OOM-ing the service by
/// streaming a 100 MB response body into `reqwest::Response::json()`. Pre-fix
/// the call buffered the entire body before deserialization with no upper
/// bound.
const USERINFO_MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// Percent-encoding set for the storage-api `by-address` path segment, reached
/// only through [`encode_resource_address`]. Treats the
/// caller-supplied resource address as opaque: encodes every ASCII byte
/// except the RFC 3986 unreserved set (`A-Z a-z 0-9 - . _ ~`). This matches
/// what the storage service's `by-address` endpoint expects for full-URL
/// resource addresses such as Azure blob URLs — e.g.
/// `https://host/container/key.usd` → `https%3A%2F%2Fhost%2Fcontainer%2Fkey.usd`.
///
/// b.ohw low: derived from `NON_ALPHANUMERIC` and then carved back to
/// preserve the four RFC 3986 unreserved punctuation bytes (`-`, `.`, `_`,
/// `~`). The pre-fix table was an explicit allow-list of 30+ ASCII bytes
/// that drifted from the RFC over time; the `NON_ALPHANUMERIC` base is a
/// single canonical reference and the `remove` calls document precisely
/// what the encoder leaves alone.
///
/// **Deliberately distinct from `domain::base_path`'s `BASE_PATH_ENCODE_SET`**
/// (b.uoc), which is this set plus `.remove(b'/')`. That one canonicalizes a
/// *path*, so `/` stays a separator; this one packs a whole URL into a single
/// URL segment, so `/` must be escaped. Do not merge them.
const PATH_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

/// Percent-encode a resource address for the storage `by-address` URL segment.
///
/// The one encoder for that endpoint's address segment, with two production
/// callers: `RestStorageServiceClient::check_permission` (the permission
/// probe) and the checkpoint lane's storage-api adapter
/// (`checkpoint::adapters::storage_api`, every artifact URL it renders). Both
/// address the same endpoint, so both must encode identically. Also exercised
/// directly by `tests/proptests.rs`.
pub fn encode_resource_address(path: &str) -> String {
    utf8_percent_encode(path, PATH_ENCODE_SET).to_string()
}

/// REST client that checks path-level access via the storage service's
/// `HEAD /v1alpha/fileobject/by-address/{path}` (stat-by-address) endpoint.
pub struct RestStorageServiceClient {
    client: reqwest::Client,
    endpoint: String,
    timeout: Duration,
    retry_config: RetryConfig,
}

impl RestStorageServiceClient {
    pub fn new(
        endpoint: &str,
        timeout_ms: u64,
        retry_config: RetryConfig,
    ) -> Result<Self, AuthError> {
        let (uri, _use_tls) = normalize_endpoint(endpoint)?;
        let trimmed = uri.trim_end_matches('/').to_owned();

        // b.j5t item 2: set a connect-phase timeout in addition to the
        // per-request `.timeout()` applied at call time. Pre-fix the
        // builder had neither, so a DNS hang on the first call could
        // block indefinitely. The per-request timeout still acts as a
        // tighter ceiling on the overall RPC.
        //
        // b.j5t follow-up: the connect-phase budget is sized off the same
        // `auth_timeout_ms` knob (`LIVE_EDIT_AUTH_TIMEOUT_MS`) that already
        // drives `.timeout()` and the gRPC clients' `.connect_timeout()`.
        // Pre-fix this was a hardcoded `REST_CONNECT_TIMEOUT_MS = 5_000`
        // constant — the asymmetry made docs ("both protocols use
        // LIVE_EDIT_AUTH_TIMEOUT_MS") lie and forced operators to remember
        // two different budgets for the same handshake phase across REST
        // vs gRPC.
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(timeout_ms))
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .map_err(|e| AuthError::InvalidResponse(format!("reqwest client: {e}")))?;

        Ok(Self {
            client,
            endpoint: trimmed,
            timeout: Duration::from_millis(timeout_ms),
            retry_config,
        })
    }
}

/// Classify a non-2xx HTTP status into the appropriate `AuthResult` / `AuthError`.
///
/// 401 and 403 map to distinct `AuthResult` variants so callers can surface
/// the right gRPC status code back to the client (`Unauthenticated` vs
/// `PermissionDenied`). Conflating them breaks token-refresh semantics in
/// standard gRPC client libraries.
///
/// **204 is an allow, not an anomaly** (b.uoc). The permission probe is a
/// `HEAD` against `stat_by_address`, whose success contract is `204 No Content`
/// with the metadata carried in response headers — it has no body to return.
/// Without this arm 204 falls through the catch-all to a permanent,
/// non-retriable `InvalidResponse`, and *every* permission check fails closed.
///
/// **400 and 414 are a verdict on the request, not a contract break.** The
/// storage service refused the address as malformed (400; observed for
/// scheme-less base paths) or too long (414; the percent-encoded address can
/// be about 3× the 4 KiB base-path cap). Both become
/// [`AuthResult::InvalidRequest`], carrying the fixed status string. As an
/// `Ok` it is not retried, not counted as an auth error, and never allows.
/// 422 and every other unlisted status stay on the catch-all
/// `InvalidResponse` (`INTERNAL`, ERROR): a HEAD has no body for 422 to
/// describe, and statuses such as 405 (a storage service without HEAD
/// by-address) or 3xx are deployment or contract faults that must stay loud
/// rather than blame the client.
fn classify_status(status: StatusCode) -> Result<AuthResult, AuthError> {
    match status.as_u16() {
        200 | 204 => Ok(AuthResult::Allowed),
        401 => Ok(AuthResult::Unauthenticated),
        403 => Ok(AuthResult::Denied),
        404 => Ok(AuthResult::PathNotFound),
        400 => Ok(AuthResult::InvalidRequest {
            backend_status: "HTTP 400",
        }),
        414 => Ok(AuthResult::InvalidRequest {
            backend_status: "HTTP 414",
        }),
        429 | 500 | 502 | 503 | 504 => Err(AuthError::Unreachable),
        // b.ohw H-auth-5: keep only the numeric status (no reason phrase
        // from `StatusCode`'s Display) and no body. Aligns with the gRPC
        // path's "code only" posture so a buggy backend can't leak token
        // material via the response status line.
        code => Err(AuthError::InvalidResponse(format!(
            "auth backend returned HTTP {code}"
        ))),
    }
}

/// Walk the error-source chain of a `reqwest::Error` looking for a
/// `rustls::Error` (b.ohw H-auth-4).
///
/// Thin protocol-specific shim around [`crate::auth::tls::has_rustls_error_in_chain`]
/// (b.j5t DRY follow-up). The walker logic itself is shared with the gRPC
/// auth path — see the `auth::tls` module docs for the why-walk-two-chains
/// rationale.
///
/// Version-bound assumption: works against `reqwest = "0.12"` paired with
/// `rustls = "0.23"` and `hyper-rustls = "0.27"`. A reqwest bump that
/// changes which crate wraps TLS errors in the source chain may require
/// updating the shared walker.
fn is_tls_error(err: &reqwest::Error) -> bool {
    crate::auth::tls::has_rustls_error_in_chain(err)
}

/// Classify a `reqwest::Error` into an `AuthError`, tagging the
/// `InvalidResponse` message with the given `service_label` for log clarity.
///
/// b.ohw H-auth-4: TLS handshake failures are classified as permanent
/// (`InvalidResponse`) rather than transient (`Unreachable`). Pre-fix every
/// non-status, non-timeout error fell through to `Unreachable` and was then
/// retried `max_attempts` times — a cert-rotation incident or a CA mismatch
/// would pound the backend with failing handshakes for the full retry
/// budget. Connect-level (TCP) and DNS failures stay transient because
/// they recover quickly on cluster topology changes.
fn classify_request_error(err: reqwest::Error, service_label: &'static str) -> AuthError {
    if err.is_timeout() {
        return AuthError::Timeout;
    }
    // Inspect TLS BEFORE `is_connect()` because reqwest classifies a TLS
    // handshake failure that happened during connection establishment as
    // a connect error — without this check it would slip through to
    // `Unreachable` and be retried.
    if is_tls_error(&err) {
        return AuthError::InvalidResponse(format!(
            "{service_label} TLS handshake failed (cert / trust mismatch — not retried)"
        ));
    }
    if err.is_connect() {
        return AuthError::Unreachable;
    }
    // DNS / connection-reset errors surface through the hyper stack;
    // reqwest does not expose dedicated predicates for them, so we treat any
    // non-timeout, non-TLS, non-connect transport error as Unreachable when
    // it has no HTTP status (i.e. the request never completed at the HTTP
    // layer).
    if err.status().is_none() {
        return AuthError::Unreachable;
    }
    // Scrub the URL from the rendered error before letting it into
    // `InvalidResponse` (b.vvs item 9). `reqwest::Error::Display` includes the
    // request URL by default; the URL embeds the encoded resource address
    // (full blob URL with container + filename) which is unnecessary for
    // diagnostics and would leak into operator logs / cached error strings.
    // `without_url()` strips the URL while keeping the underlying status /
    // body explanation.
    let scrubbed = err.without_url();
    AuthError::InvalidResponse(format!("{service_label} request failed: {scrubbed}"))
}

/// A body read that the transport completed: either the whole body, or the
/// cap breach that stopped it. Kept apart from the transport error (the
/// `Err` beside it) so a caller can treat an oversize body differently from a
/// failed connection.
enum CappedBody {
    Within(bytes::Bytes),
    /// The `InvalidResponse` naming the breach.
    OverCap(AuthError),
}

/// Read up to `cap` bytes from `response`'s body, returning
/// [`CappedBody::OverCap`] the moment the accumulated byte count exceeds
/// `cap`. The chunk stream is dropped on the cap-exceeded path so the
/// underlying connection is freed immediately rather than continuing to drain
/// a hostile gigabyte. A transport failure mid-body is the `Err`.
///
/// b.j5t item 2: `reqwest::Response::bytes()` and `.json()` both buffer the
/// entire body before returning, which lets a malicious backend OOM the
/// service by streaming an arbitrarily large response. Streaming with an
/// inline cap caps memory at `cap + chunk_size` regardless of what the
/// backend tries to send.
async fn read_bounded_body(
    mut response: reqwest::Response,
    cap: usize,
) -> Result<CappedBody, AuthError> {
    use bytes::BytesMut;

    let mut buf = BytesMut::with_capacity(std::cmp::min(cap, 4096));
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if buf.len().saturating_add(chunk.len()) > cap {
                    return Ok(CappedBody::OverCap(AuthError::InvalidResponse(format!(
                        "userinfo response exceeded {cap}-byte cap during streaming read"
                    ))));
                }
                buf.extend_from_slice(&chunk);
            }
            Ok(None) => return Ok(CappedBody::Within(buf.freeze())),
            Err(e) => {
                return Err(classify_request_error(e, "userinfo service"));
            }
        }
    }
}

#[async_trait]
impl permissions::PermissionsClient for RestStorageServiceClient {
    async fn check_permission(
        &self,
        token: &str,
        path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        // The caller's path is treated as an opaque resource address — every
        // byte outside the RFC 3986 unreserved set is percent-encoded, so a
        // full blob URL like `https://host/container/file.usd` survives as a
        // single encoded path segment on the wire.
        let encoded_path = encode_resource_address(path);
        // b.uoc: `HEAD` (stat-by-address), not `GET` (read-by-address). Same
        // authorization semantics — both map to the upstream `read` action —
        // but the GET is the *read* endpoint: it streams the layer body, which
        // this probe discards, its `max_page_size` param is ignored, and in a
        // redirect-serving deployment it can answer `300 Multiple Choices`,
        // which `classify_status` treats as a permanent `InvalidResponse` and
        // which would fail every authorization check. HEAD also restores parity
        // with the gRPC backend, which already probes `stat`.
        let url = format!(
            "{}/v1alpha/fileobject/by-address/{}",
            self.endpoint, encoded_path
        );
        // Proactively validate the bearer header value so invalid-token bytes
        // (e.g. CR/LF) surface as `InvalidResponse("invalid token")` — matching
        // the gRPC client's `MetadataValue::try_from` guard — rather than being
        // swallowed into a generic `Unreachable` by the reqwest send path.
        let bearer = reqwest::header::HeaderValue::try_from(format!("Bearer {token}"))
            .map_err(|_| AuthError::InvalidResponse("invalid token".to_owned()))?;

        retry_on_transient(&self.retry_config, || {
            let client = self.client.clone();
            let url = url.clone();
            let bearer = bearer.clone();
            let timeout = self.timeout;
            async move {
                let response = client
                    .head(&url)
                    .header("Authorization", bearer)
                    .timeout(timeout)
                    .send()
                    .await
                    .map_err(|e| classify_request_error(e, "storage service"))?;
                classify_status(response.status())
            }
        })
        .await
    }
}

/// REST client that resolves a bearer token to user identity via the userinfo
/// service's `GET /api/v1alpha/users/me` endpoint (PRD §5.5).
pub struct RestUserinfoClient {
    client: reqwest::Client,
    endpoint: String,
    timeout: Duration,
    retry_config: RetryConfig,
}

/// Wire-format DTO for the userinfo `GET /api/v1alpha/users/me` JSON body.
/// Field names mirror the gRPC `User` proto's snake_case baseline.
#[derive(Debug, Deserialize)]
struct UserInfoDto {
    id: String,
    display_name: String,
    email: Option<String>,
    username: String,
}

/// Wire-format DTO for the userinfo service's error body
/// (`{"error":"<tag>","message":"..."}`). Only the machine-readable tag is
/// read; `message` and any other field are ignored and never kept.
#[derive(Debug, Deserialize)]
struct UserinfoErrorDto {
    error: String,
}

/// The userinfo service's error tag for "no user for this principal".
const USERINFO_NOT_FOUND_TAG: &str = "not_found";

/// Read a userinfo response body under [`USERINFO_MAX_RESPONSE_BYTES`]: the
/// `Content-Length` fast rejection, then the streaming cap.
///
/// b.j5t item 2: `response.json()` would buffer arbitrary sizes, so a
/// malicious userinfo backend could OOM the service by streaming a multi-GB
/// body. `Content-Length` is consulted as a fast-path rejection before any
/// bytes are read, but a backend may omit it (chunked transfer-encoding), so
/// the cap is still enforced on the streaming path inside
/// [`read_bounded_body`]. The header check is an optimization; the streaming
/// cap is the correctness primitive — never trust the header alone. Either
/// breach is [`CappedBody::OverCap`]; a transport error mid-body is the `Err`,
/// classified by `read_bounded_body`.
async fn read_userinfo_body(response: reqwest::Response) -> Result<CappedBody, AuthError> {
    if let Some(len) = response.content_length() {
        if len as usize > USERINFO_MAX_RESPONSE_BYTES {
            return Ok(CappedBody::OverCap(AuthError::InvalidResponse(format!(
                "userinfo response exceeded {USERINFO_MAX_RESPONSE_BYTES}-byte cap (Content-Length={len})"
            ))));
        }
    }
    read_bounded_body(response, USERINFO_MAX_RESPONSE_BYTES).await
}

impl RestUserinfoClient {
    pub fn new(
        endpoint: &str,
        timeout_ms: u64,
        retry_config: RetryConfig,
    ) -> Result<Self, AuthError> {
        let (uri, _use_tls) = normalize_endpoint(endpoint)?;
        let trimmed = uri.trim_end_matches('/').to_owned();

        // b.j5t item 2: set connect + per-request timeouts on the userinfo
        // client (the storage client got the same treatment above). The
        // request body itself is bounded inside `get_current_user` via a
        // streaming `.bytes()` read with a 64 KiB ceiling — `.timeout()`
        // alone doesn't stop a malicious backend trickling out megabytes of
        // valid JSON within the timeout window.
        //
        // b.j5t follow-up: connect-phase budget shares `auth_timeout_ms`
        // with the gRPC clients — see the matching comment in
        // `RestStorageServiceClient::new`.
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(timeout_ms))
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .map_err(|e| AuthError::InvalidResponse(format!("reqwest client: {e}")))?;

        Ok(Self {
            client,
            endpoint: trimmed,
            timeout: Duration::from_millis(timeout_ms),
            retry_config,
        })
    }
}

#[async_trait]
impl userinfo::UserinfoClient for RestUserinfoClient {
    async fn get_current_user(&self, token: &str) -> Result<UserInfo, AuthError> {
        let url = format!("{}/api/v1alpha/users/me", self.endpoint);
        // Proactively validate the bearer header value — see the parallel
        // comment in `RestStorageServiceClient::check_permission`.
        let bearer = reqwest::header::HeaderValue::try_from(format!("Bearer {token}"))
            .map_err(|_| AuthError::InvalidResponse("invalid token".to_owned()))?;

        retry_on_transient(&self.retry_config, || {
            let client = self.client.clone();
            let url = url.clone();
            let bearer = bearer.clone();
            let timeout = self.timeout;
            async move {
                let response = client
                    .get(&url)
                    .header("Authorization", bearer)
                    .timeout(timeout)
                    .send()
                    .await
                    .map_err(|e| classify_request_error(e, "userinfo service"))?;

                let status = response.status();
                match status.as_u16() {
                    200 => {
                        // b.j5t item 2: the body is read under
                        // USERINFO_MAX_RESPONSE_BYTES before deserializing
                        // (see `read_userinfo_body`).
                        let body = match read_userinfo_body(response).await? {
                            CappedBody::Within(body) => body,
                            CappedBody::OverCap(err) => return Err(err),
                        };
                        let dto: UserInfoDto = serde_json::from_slice(&body).map_err(|e| {
                            AuthError::InvalidResponse(format!("malformed userinfo response: {e}"))
                        })?;
                        Ok(UserInfo {
                            id: dto.id,
                            display_name: dto.display_name,
                            email: dto.email,
                            username: dto.username,
                        })
                    }
                    // Deliberately not worded as a verdict on the caller's
                    // token. A userinfo 401/403 is not proof the bearer is
                    // invalid or expired: the userinfo service does not
                    // necessarily validate the bearer at all, and it can
                    // also answer 401 for a failure of its own upstream
                    // credential or for a bearer it cannot extract an
                    // identity from. The message says only what happened.
                    code @ (401 | 403) => Err(AuthError::InvalidResponse(format!(
                        "userinfo service rejected the bearer (HTTP {code})"
                    ))),
                    // The userinfo service holds no user for the bearer's
                    // principal. Only the service's own `not_found` body
                    // proves that: a 404 from a wrong path prefix in
                    // `LIVE_EDIT_USERINFO_ENDPOINT` (a route the server does
                    // not have) answers with an empty or foreign body, and
                    // that misconfiguration must stay loud, so any other 404
                    // — empty, non-JSON, another tag, oversize — keeps the
                    // catch-all `InvalidResponse` below. The body is read
                    // under the same cap as the 200 arm and never logged or
                    // wrapped (b.ohw H-auth-5); a transport error while
                    // reading it propagates as on the 200 arm.
                    404 => {
                        let body = match read_userinfo_body(response).await? {
                            CappedBody::Within(body) => body,
                            // Oversize is not the service's `not_found`
                            // body: fall through to the 404 catch-all.
                            CappedBody::OverCap(_) => bytes::Bytes::new(),
                        };
                        match serde_json::from_slice::<UserinfoErrorDto>(&body) {
                            Ok(dto) if dto.error == USERINFO_NOT_FOUND_TAG => {
                                Err(AuthError::UserNotFound)
                            }
                            _ => Err(AuthError::InvalidResponse(
                                "auth backend returned HTTP 404".to_owned(),
                            )),
                        }
                    }
                    429 | 500 | 502 | 503 | 504 => Err(AuthError::Unreachable),
                    // b.ohw H-auth-5: status-only message — never include the
                    // `StatusCode` `Display` form (which appends the reason
                    // phrase) or body in the wrapped error string.
                    code => Err(AuthError::InvalidResponse(format!(
                        "auth backend returned HTTP {code}"
                    ))),
                }
            }
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{fast_retry, permissions::PermissionsClient};
    use std::time::Duration;
    use wiremock::matchers::{any, header, method};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    /// Per-attempt timeout for tests whose subject is *not* timing.
    ///
    /// A deadline, not a sleep: the healthy path against a local wiremock
    /// server returns in single-digit milliseconds and never approaches this,
    /// so the budget costs nothing and only has to be unreachable under load.
    /// It is sized that way deliberately — this binary runs ~40 tests in
    /// parallel, each with its own runtime and its own wiremock server, and at
    /// the previous 200 ms a starved task occasionally blew the budget on the
    /// *healthy* path. The test then reported `AuthError::Timeout` instead of
    /// the classification it exists to assert, which is a red run that says
    /// nothing about the code under test.
    pub(super) const TEST_TIMEOUT_MS: u64 = 5_000;

    /// Per-attempt timeout for the two tests that deliberately provoke a
    /// timeout. Paired with a response delay an order of magnitude larger, so
    /// every attempt is guaranteed to exhaust it.
    pub(super) const TIMEOUT_PROBE_MS: u64 = 200;

    /// Build a `RestStorageServiceClient` pointed at `mock_server` with the
    /// non-timing [`TEST_TIMEOUT_MS`] budget and the given retry configuration.
    fn start_rest_storage_mock(
        mock_server: &MockServer,
        retry_config: RetryConfig,
    ) -> RestStorageServiceClient {
        RestStorageServiceClient::new(&mock_server.uri(), TEST_TIMEOUT_MS, retry_config).unwrap()
    }

    /// A loopback address guaranteed to refuse connections, for the
    /// "backend is down" tests.
    ///
    /// Port 1 is privileged — binding it needs root, which the test process
    /// does not have — and sits below Linux's ephemeral range
    /// (`net.ipv4.ip_local_port_range`, 32768+ by default), so no
    /// concurrently-starting `MockServer` in this binary can ever be handed
    /// it. The obvious alternative (bind `:0`, read the port, drop the
    /// listener) returns that port to the ephemeral pool, and with ~40
    /// wiremock servers starting in parallel here another test does
    /// occasionally win the race and answer the request — an intermittent
    /// failure that says nothing about the code under test.
    pub(super) const REFUSED_ADDR: &str = "http://127.0.0.1:1";

    /// Wiremock matcher that compares against the raw (un-decoded) request URI
    /// starting at `/v1alpha`. Needed because wiremock's built-in `path()`
    /// matcher decodes percent-escapes before comparing, which would hide
    /// whether e.g. `/` is encoded as `%2F` on the wire.
    struct RawUriMatcher(&'static str);
    impl wiremock::Match for RawUriMatcher {
        fn matches(&self, request: &Request) -> bool {
            let uri = request.url.as_str();
            match uri.find("/v1alpha") {
                Some(idx) => &uri[idx..] == self.0,
                None => false,
            }
        }
    }

    /// `204 No Content` is the storage API's actual success answer to a `HEAD`
    /// against `stat_by_address`: there is no body, and the metadata rides in
    /// response headers. This is *the* status the b.uoc probe sees in
    /// production, so it gets a test of its own — without the `204` arm in
    /// `classify_status` a 204 falls through the catch-all to a permanent,
    /// non-retriable `InvalidResponse` and every permission check in the
    /// service fails closed.
    #[tokio::test]
    async fn stat_204_returns_allowed() {
        let mock_server = MockServer::start().await;
        // The caller's path is opaque: every `/` is percent-encoded so the
        // address occupies a single URL path segment.
        Mock::given(method("HEAD"))
            .and(RawUriMatcher(
                "/v1alpha/fileobject/by-address/%2Fscene%2Flayer1",
            ))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let result = client
            .check_permission("tok", "/scene/layer1", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Allowed);
    }

    /// `200 OK` stays an allow alongside 204, deliberately: a proxy or a
    /// future endpoint revision that answers the probe with a body must not
    /// start denying access.
    #[tokio::test]
    async fn stat_200_returns_allowed() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(RawUriMatcher(
                "/v1alpha/fileobject/by-address/%2Fscene%2Flayer1",
            ))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let result = client
            .check_permission("tok", "/scene/layer1", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Allowed);
    }

    /// Pin the probe's *shape* off the recorded request rather than off a
    /// matcher, so the assertion is about what was actually sent: exactly one
    /// request, method `HEAD`, and no query string at all (b.uoc).
    ///
    /// The matcher-based tests above would all still pass if the client also
    /// issued a stray `GET`, or appended a query parameter no `RawUriMatcher`
    /// happens to name — this one would not, because it matches `any()` and
    /// then inspects every request the server received. Pre-b.uoc the probe
    /// was `GET .../by-address/{addr}?max_page_size=1`: the *read* endpoint,
    /// which streams the layer body this probe discards, ignores
    /// `max_page_size`, and in a redirect-serving deployment can answer `300
    /// Multiple Choices` — a permanent `InvalidResponse` that fails every
    /// authorization check.
    #[tokio::test]
    async fn stat_probe_is_head_with_no_query_string() {
        let mock_server = MockServer::start().await;
        // Deliberately unconditional: a stray request of any method or shape
        // is recorded rather than 404'd, so the count below can see it.
        Mock::given(any())
            .respond_with(ResponseTemplate::new(204))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(1));
        let result = client
            .check_permission("tok", "/scene/layer1", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Allowed);

        let requests = mock_server.received_requests().await.unwrap();
        let methods: Vec<String> = requests.iter().map(|r| r.method.to_string()).collect();
        assert_eq!(
            requests.len(),
            1,
            "the probe must issue exactly one request; methods seen: {methods:?}"
        );

        let request = &requests[0];
        assert_eq!(
            request.method.as_str(),
            "HEAD",
            "the permission probe must be a HEAD (stat-by-address), never a GET \
             (read-by-address, which streams the layer body)"
        );
        assert_eq!(
            request.url.query(),
            None,
            "the probe URL must carry no query string, got {:?}",
            request.url.query()
        );
        // Byte-exact on the raw URI, for the reason `RawUriMatcher` exists:
        // `Url::path()` alone would not prove the query is absent, and a
        // decoded comparison would hide whether `/` is encoded on the wire.
        let uri = request.url.as_str();
        let idx = uri
            .find("/v1alpha")
            .unwrap_or_else(|| panic!("probe URI must contain /v1alpha, got {uri:?}"));
        assert_eq!(
            &uri[idx..],
            "/v1alpha/fileobject/by-address/%2Fscene%2Flayer1",
            "the probe must target stat-by-address with the opaque-encoded address"
        );
    }

    #[tokio::test]
    async fn stat_404_returns_path_not_found() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let result = client
            .check_permission("tok", "/missing", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::PathNotFound);
    }

    /// 401 Unauthorized (invalid/expired bearer) → `AuthResult::Unauthenticated`,
    /// distinct from 403 so the handler can surface gRPC `UNAUTHENTICATED` and
    /// clients know to refresh the token rather than give up.
    #[tokio::test]
    async fn stat_401_returns_unauthenticated() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let result = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Unauthenticated);
    }

    /// 403 Forbidden (authenticated but lacks permission) → `AuthResult::Denied`.
    #[tokio::test]
    async fn stat_403_returns_denied() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let result = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Denied);
    }

    /// Helper: asserts that `status` is retried (up to `max_attempts`) and the
    /// final success (200) is returned, with call count == fail_count + 1.
    async fn assert_transient_status_retried_then_succeeds(status: u16, fail_count: u64) {
        let mock_server = MockServer::start().await;
        // Fail the first N calls with `status`, succeed after that.
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(status))
            .up_to_n_times(fail_count)
            .expect(fail_count)
            .mount(&mock_server)
            .await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry((fail_count as u32) + 1));
        let result = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Allowed);

        // wiremock verifies the `.expect()` counts on drop — also check total
        let total_calls = mock_server.received_requests().await.unwrap().len() as u64;
        assert_eq!(total_calls, fail_count + 1);
    }

    #[tokio::test]
    async fn stat_transient_5xx_and_429_retried_then_succeeds() {
        // Collapsed from five per-status tests. 503 uses fail_count=2 to
        // exercise a 2-retry chain (one extra transient failure before
        // success) — the others use fail_count=1. Asymmetry is intentional
        // and documented here so future readers don't "normalize" it.
        for (status, fail_count) in [(429u16, 1), (500, 1), (502, 1), (503, 2), (504, 1)] {
            assert_transient_status_retried_then_succeeds(status, fail_count).await;
        }
    }

    /// Helper: asserts that `status` is the storage service refusing the probe
    /// as a malformed request — `Ok(InvalidRequest)` carrying
    /// `expected_backend_status` — and that it is answered from one request
    /// under a three-attempt retry budget, so it is a verdict and not a
    /// transient error that burned the budget.
    async fn assert_invalid_request_not_retried(status: u16, expected_backend_status: &str) {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let result = client
            .check_permission("tok", "load-test/perf", Permission::Read)
            .await;
        assert!(
            matches!(
                result,
                Ok(AuthResult::InvalidRequest { backend_status })
                    if backend_status == expected_backend_status
            ),
            "HTTP {status} must be Ok(InvalidRequest {{ backend_status: \
             {expected_backend_status:?} }}); got {result:?}"
        );

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(
            calls, 1,
            "HTTP {status} is a verdict on the request and must not be retried"
        );
    }

    /// 400 is the storage service refusing the address as malformed (seen for
    /// scheme-less base paths). It is a verdict on the request, so it must
    /// not become `InvalidResponse` (`INTERNAL`, which Kit retries) and must
    /// not be retried here.
    #[tokio::test]
    async fn stat_400_returns_invalid_request_no_retry() {
        assert_invalid_request_not_retried(400, "HTTP 400").await;
    }

    /// 414 is the storage service refusing the address as too long; the
    /// percent-encoded address can be about 3x the base-path byte cap. Same
    /// verdict as 400, with its own status string.
    #[tokio::test]
    async fn stat_414_returns_invalid_request_no_retry() {
        assert_invalid_request_not_retried(414, "HTTP 414").await;
    }

    /// Pins the decision that 422 is **not** a request verdict: a HEAD has no
    /// body for a 422 to describe, so it stays on the catch-all permanent
    /// `InvalidResponse` (`INTERNAL`, ERROR) alongside 405 below, rather than
    /// joining 400 / 414 as `InvalidRequest`.
    #[tokio::test]
    async fn stat_422_returns_invalid_response_no_retry() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(422))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let err = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, AuthError::InvalidResponse(_)));

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 1);
    }

    /// A storage service that does not route `HEAD` on `stat_by_address`
    /// answers `405 Method Not Allowed`, and that is the deployment
    /// prerequisite b.uoc introduced: the permission probe is a `HEAD`, so a
    /// backend without that route makes **every** authorization check fail —
    /// permanently, with no retry to ride out, since 405 is not a transient
    /// condition and hammering it would only multiply the failure.
    ///
    /// Pinned here because that consequence is now documented as an operator
    /// prerequisite (`docs/service/docs/authentication.md`, SDD §14.2, PRD,
    /// CHANGELOG) while nothing in the
    /// code pinned it: reclassifying 405 as transient would burn the retry
    /// budget on a misconfiguration that cannot self-heal, and reclassifying
    /// it as `Allowed` would open the probe entirely. The assertion is
    /// deliberately two-part — the variant *and* the call count — because
    /// only the count distinguishes "permanent" from "retried and gave up".
    #[tokio::test]
    async fn stat_405_returns_invalid_response_no_retry() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(405))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let err = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap_err();
        assert!(
            matches!(err, AuthError::InvalidResponse(_)),
            "405 (backend does not route HEAD) must be a permanent \
             InvalidResponse, not a transient error or an allow; got {err:?}"
        );

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(
            calls, 1,
            "405 must not be retried — the misconfiguration cannot self-heal"
        );
    }

    #[tokio::test]
    async fn stat_request_timeout_returns_timeout_and_retries() {
        let mock_server = MockServer::start().await;
        // Delay the response far longer than TIMEOUT_PROBE_MS so every attempt
        // is guaranteed to exhaust its budget.
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
            .mount(&mock_server)
            .await;

        // The one storage test that wants the tight budget — it is asserting
        // on the timeout itself, so it must not inherit TEST_TIMEOUT_MS.
        let client =
            RestStorageServiceClient::new(&mock_server.uri(), TIMEOUT_PROBE_MS, fast_retry(3))
                .unwrap();
        let err = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, AuthError::Timeout));

        // Timeout is transient → all 3 attempts must have fired.
        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn stat_connection_refused_returns_unreachable() {
        let client =
            RestStorageServiceClient::new(REFUSED_ADDR, TEST_TIMEOUT_MS, fast_retry(1)).unwrap();
        let err = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap_err();
        assert!(
            matches!(err, AuthError::Unreachable),
            "a refused connection is transient (backend down / rolling restart) \
             and must classify as Unreachable, got {err:?}"
        );
    }

    #[tokio::test]
    async fn stat_retry_exhausts_returns_last_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let err = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, AuthError::Unreachable));

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn stat_bearer_token_sent_in_header() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(header("authorization", "Bearer my-token"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(3));
        let result = client
            .check_permission("my-token", "/p", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Allowed);
    }

    #[tokio::test]
    async fn stat_url_encodes_path_as_opaque_resource_address() {
        let mock_server = MockServer::start().await;
        // Proves three properties of the opaque-encoding contract:
        //   1. reserved chars `/` and space are percent-encoded (`%2F`, `%20`),
        //   2. RFC 3986 unreserved chars `.` `_` `-` are preserved,
        //   3. no ad-hoc leading-slash normalization — the caller's `/` is
        //      encoded like any other byte.
        Mock::given(method("HEAD"))
            .and(RawUriMatcher(
                "/v1alpha/fileobject/by-address/%2Fscene%2Fmy%20layer.usda",
            ))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(1));
        let result = client
            .check_permission("tok", "/scene/my layer.usda", Permission::Read)
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Allowed);
    }

    #[tokio::test]
    async fn stat_full_url_path_is_percent_encoded() {
        // Regression test for b.3k8: full URL-form resource addresses
        // (e.g. object-store blob URLs) must have their `://` and `/` encoded
        // so the storage service sees a single opaque path segment. Verified
        // against a deployed storage service: the encoded form returns 200, the
        // un-encoded form returns 404.
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(RawUriMatcher(
                "/v1alpha/fileobject/by-address/\
                 https%3A%2F%2Fstorage.example.com\
                 %2Fcontainer%2Ftest%2Fdo-not-remove.usd",
            ))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(1));
        let result = client
            .check_permission(
                "tok",
                "https://storage.example.com/container/test/do-not-remove.usd",
                Permission::Write,
            )
            .await
            .unwrap();
        assert_eq!(result, AuthResult::Allowed);
    }

    #[tokio::test]
    async fn stat_read_and_write_both_issue_same_request() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(RawUriMatcher(
                "/v1alpha/fileobject/by-address/%2Fsame%2Fpath",
            ))
            .respond_with(ResponseTemplate::new(200))
            .expect(2)
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(1));
        client
            .check_permission("tok", "/same/path", Permission::Read)
            .await
            .unwrap();
        client
            .check_permission("tok", "/same/path", Permission::Write)
            .await
            .unwrap();
        // wiremock's `.expect(2)` asserts on drop that exactly 2 matching
        // requests arrived — with the same URL shape for both permissions.
    }

    #[tokio::test]
    async fn stat_no_retry_with_max_attempts_one() {
        let mock_server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&mock_server)
            .await;

        let client = start_rest_storage_mock(&mock_server, fast_retry(1));
        let err = client
            .check_permission("tok", "/p", Permission::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, AuthError::Unreachable));

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 1);
    }
}

#[cfg(test)]
mod userinfo_tests {
    use super::tests::{REFUSED_ADDR, TEST_TIMEOUT_MS, TIMEOUT_PROBE_MS};
    use super::*;
    use crate::auth::{fast_retry, userinfo::UserinfoClient};
    use std::time::Duration;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    /// Build a `RestUserinfoClient` pointed at `mock_server` with the
    /// non-timing [`TEST_TIMEOUT_MS`] budget and the given retry
    /// configuration. Mirrors `start_rest_storage_mock` from the
    /// storage-client tests, including its timeout rationale.
    fn start_rest_userinfo_mock(
        mock_server: &MockServer,
        retry_config: RetryConfig,
    ) -> RestUserinfoClient {
        RestUserinfoClient::new(&mock_server.uri(), TEST_TIMEOUT_MS, retry_config).unwrap()
    }

    /// Build a `200 OK` JSON response with `application/json` content-type.
    /// Avoids a `serde_json` dep — body is a raw string we control byte-for-byte.
    fn json_response(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(200)
            .insert_header("content-type", "application/json")
            .set_body_raw(body.as_bytes().to_vec(), "application/json")
    }

    /// Standard happy-path JSON body with all four fields populated.
    const HAPPY_BODY: &str =
        r#"{"id":"u1","display_name":"Jane","email":"j@x.com","username":"jane"}"#;

    #[tokio::test]
    async fn userinfo_200_returns_user_info() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1alpha/users/me"))
            .respond_with(json_response(HAPPY_BODY))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let info = client.get_current_user("tok").await.unwrap();
        assert_eq!(info.id, "u1");
        assert_eq!(info.display_name, "Jane");
        assert_eq!(info.email, Some("j@x.com".to_owned()));
        assert_eq!(info.username, "jane");
    }

    #[tokio::test]
    async fn userinfo_200_email_null_returns_none() {
        let mock_server = MockServer::start().await;
        let body = r#"{"id":"u1","display_name":"Jane","email":null,"username":"jane"}"#;
        Mock::given(method("GET"))
            .respond_with(json_response(body))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let info = client.get_current_user("tok").await.unwrap();
        assert_eq!(info.email, None);
    }

    #[tokio::test]
    async fn userinfo_200_email_missing_returns_none() {
        let mock_server = MockServer::start().await;
        // No `email` key at all — Option<String> defaults to None.
        let body = r#"{"id":"u1","display_name":"Jane","username":"jane"}"#;
        Mock::given(method("GET"))
            .respond_with(json_response(body))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let info = client.get_current_user("tok").await.unwrap();
        assert_eq!(info.email, None);
    }

    #[tokio::test]
    async fn userinfo_401_returns_invalid_response() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        match err {
            AuthError::InvalidResponse(msg) => {
                assert_eq!(msg, "userinfo service rejected the bearer (HTTP 401)");
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }

        // 401 is non-transient → exactly one call.
        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 1);
    }

    #[tokio::test]
    async fn userinfo_403_returns_invalid_response() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        match err {
            AuthError::InvalidResponse(msg) => {
                assert_eq!(msg, "userinfo service rejected the bearer (HTTP 403)");
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 1);
    }

    // ── 404: the userinfo service's "no such user" vs. a route fault ────────
    //
    // A 404 is the typed `UserNotFound` outcome only when its bounded body is
    // the userinfo service's own `{"error":"not_found",…}`. Any other 404 —
    // empty, non-JSON, another tag, or a body over the cap — is what a wrong
    // path prefix in the endpoint produces, so it keeps the loud catch-all
    // `InvalidResponse("auth backend returned HTTP 404")`. Neither is retried.

    /// The userinfo service's not-found body, byte-for-byte.
    const NOT_FOUND_BODY: &str = r#"{"error":"not_found","message":"not found"}"#;

    /// A `404` carrying `body` as `application/json`.
    fn not_found_response(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(404).set_body_raw(body.as_bytes().to_vec(), "application/json")
    }

    #[tokio::test]
    async fn userinfo_404_with_not_found_body_returns_user_not_found_not_retried() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1alpha/users/me"))
            .respond_with(not_found_response(NOT_FOUND_BODY))
            .mount(&mock_server)
            .await;

        // Three attempts allowed, so a retry of the answer would show as a
        // second request.
        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(
            matches!(err, AuthError::UserNotFound),
            "a 404 with the userinfo service's `not_found` body must be UserNotFound, got {err:?}"
        );

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 1, "UserNotFound is an answer, not a fault: no retry");
    }

    #[tokio::test]
    async fn userinfo_404_without_the_not_found_body_returns_invalid_response_not_retried() {
        for (case, response) in [
            ("empty body", ResponseTemplate::new(404)),
            ("non-JSON body", not_found_response("Not Found")),
            (
                "another error tag",
                not_found_response(r#"{"error":"bad_request"}"#),
            ),
        ] {
            let mock_server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(response)
                .mount(&mock_server)
                .await;

            let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
            let err = client.get_current_user("tok").await.unwrap_err();
            assert!(
                matches!(err, AuthError::InvalidResponse(ref msg) if msg == "auth backend returned HTTP 404"),
                "{case}: a 404 without the `not_found` body is a route fault and must stay \
                 the catch-all InvalidResponse, got {err:?}"
            );

            let calls = mock_server.received_requests().await.unwrap().len();
            assert_eq!(calls, 1, "{case}: a 404 is never retried");
        }
    }

    #[tokio::test]
    async fn userinfo_404_not_found_body_message_is_not_carried_into_the_error() {
        // b.ohw H-auth-5: the body is read for its tag only. A backend echoing
        // a bearer into `message` must not get it into the error, whose
        // `Debug` / `Display` reach logs and the negative cache.
        const SENTINEL: &str = "eyJhbGciOiJSUzI1NiJ9.LEAKEDTOKEN.c2ln";
        let body = format!(r#"{{"error":"not_found","message":"{SENTINEL}"}}"#);
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(not_found_response(&body))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(1));
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(matches!(err, AuthError::UserNotFound), "got {err:?}");
        let rendered = format!("{err:?} / {err}");
        assert!(
            !rendered.contains("LEAKEDTOKEN"),
            "the 404 body's message must not reach the error, got: {rendered}"
        );
    }

    #[tokio::test]
    async fn userinfo_404_not_found_body_over_the_cap_returns_invalid_response() {
        // The body IS the `not_found` shape, padded past the 64 KiB cap
        // (wiremock declares it in Content-Length). Were the cap skipped on
        // the 404 arm, it would parse as `not_found`; it must instead be
        // treated as not the userinfo service's body.
        let padding = "x".repeat(USERINFO_MAX_RESPONSE_BYTES + 1);
        let body = format!(r#"{{"error":"not_found","message":"{padding}"}}"#);
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(not_found_response(&body))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(
            matches!(err, AuthError::InvalidResponse(ref msg) if msg == "auth backend returned HTTP 404"),
            "an oversize 404 body must not be read as `not_found`, got {err:?}"
        );

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 1);
    }

    /// Helper: assert that `status` is treated as transient — first
    /// `fail_count` calls return `status`, then a 200 succeeds.
    async fn assert_transient_status_retried_then_succeeds(status: u16, fail_count: u64) {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(status))
            .up_to_n_times(fail_count)
            .expect(fail_count)
            .mount(&mock_server)
            .await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(HAPPY_BODY.as_bytes().to_vec(), "application/json"),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry((fail_count as u32) + 1));
        let info = client.get_current_user("tok").await.unwrap();
        assert_eq!(info.id, "u1");

        let total_calls = mock_server.received_requests().await.unwrap().len() as u64;
        assert_eq!(total_calls, fail_count + 1);
    }

    #[tokio::test]
    async fn userinfo_transient_5xx_and_429_retried_then_succeeds() {
        // Collapsed from five per-status tests. 503 uses fail_count=2 to
        // exercise a 2-retry chain (one extra transient failure before
        // success) — the others use fail_count=1. Mirrors the storage-client
        // test above; asymmetry is intentional.
        for (status, fail_count) in [(429u16, 1), (500, 1), (502, 1), (503, 2), (504, 1)] {
            assert_transient_status_retried_then_succeeds(status, fail_count).await;
        }
    }

    #[tokio::test]
    async fn userinfo_400_returns_invalid_response_no_retry() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(400))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(matches!(err, AuthError::InvalidResponse(_)));

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 1);
    }

    #[tokio::test]
    async fn userinfo_422_returns_invalid_response_no_retry() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(422))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(matches!(err, AuthError::InvalidResponse(_)));

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 1);
    }

    #[tokio::test]
    async fn userinfo_request_timeout_returns_timeout() {
        let mock_server = MockServer::start().await;
        // Delay the response by 5 s so every attempt exhausts TIMEOUT_PROBE_MS.
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(HAPPY_BODY.as_bytes().to_vec(), "application/json")
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&mock_server)
            .await;

        // The one userinfo test that wants the tight budget — it is asserting
        // on the timeout itself, so it must not inherit TEST_TIMEOUT_MS.
        let client =
            RestUserinfoClient::new(&mock_server.uri(), TIMEOUT_PROBE_MS, fast_retry(3)).unwrap();
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(matches!(err, AuthError::Timeout));

        // Timeout is transient → all 3 attempts must have fired.
        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn userinfo_connection_refused_returns_unreachable() {
        let client = RestUserinfoClient::new(REFUSED_ADDR, TEST_TIMEOUT_MS, fast_retry(1)).unwrap();
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(
            matches!(err, AuthError::Unreachable),
            "a refused connection is transient (backend down / rolling restart) \
             and must classify as Unreachable, got {err:?}"
        );
    }

    #[tokio::test]
    async fn userinfo_malformed_json_on_200_returns_invalid_response() {
        let mock_server = MockServer::start().await;
        // 200 with a non-JSON body. Malformed JSON on 200 is non-transient —
        // `.expect(1)` asserts the call is NOT retried.
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("not json at all")
                    .insert_header("content-type", "application/json"),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(matches!(err, AuthError::InvalidResponse(_)));
    }

    #[tokio::test]
    async fn userinfo_missing_required_field_returns_invalid_response() {
        let mock_server = MockServer::start().await;
        // Missing the required `id` field — serde rejects the body, JSON
        // parse failure → InvalidResponse, not retried.
        let body = r#"{"display_name":"Jane","email":"j@x.com","username":"jane"}"#;
        Mock::given(method("GET"))
            .respond_with(json_response(body))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(matches!(err, AuthError::InvalidResponse(_)));
    }

    #[tokio::test]
    async fn userinfo_retry_exhausts_returns_last_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let err = client.get_current_user("tok").await.unwrap_err();
        assert!(matches!(err, AuthError::Unreachable));

        let calls = mock_server.received_requests().await.unwrap().len();
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn userinfo_bearer_token_sent_in_header() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(header("authorization", "Bearer my-token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(HAPPY_BODY.as_bytes().to_vec(), "application/json"),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let info = client.get_current_user("my-token").await.unwrap();
        assert_eq!(info.id, "u1");
    }

    #[tokio::test]
    async fn userinfo_url_matches_contract() {
        // Assert the raw request URI exactly: path is /api/v1alpha/users/me
        // with no query string. The storage client's probe carries none
        // either since b.uoc; this test owns the userinfo side of that
        // contract, `stat_probe_is_head_with_no_query_string` the other.
        struct UserinfoUriMatcher;
        impl wiremock::Match for UserinfoUriMatcher {
            fn matches(&self, request: &Request) -> bool {
                let url = &request.url;
                url.path() == "/api/v1alpha/users/me" && url.query().is_none()
            }
        }

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(UserinfoUriMatcher)
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(HAPPY_BODY.as_bytes().to_vec(), "application/json"),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(1));
        let info = client.get_current_user("tok").await.unwrap();
        assert_eq!(info.id, "u1");
    }

    // ── b.ohw H-auth-4: TLS handshake errors are NOT retried ────────────
    //
    // Strategy: spin up a plain TCP listener that immediately writes
    // arbitrary non-TLS bytes (or just RSTs the connection mid-handshake),
    // then point a reqwest client at `https://127.0.0.1:<port>`. The client
    // attempts a TLS handshake, the bytes it reads are not valid TLS
    // records, and rustls surfaces a `rustls::Error` that propagates up the
    // `reqwest::Error` source chain. Pre-fix this would be classified as
    // `Unreachable` (transient) and retried `max_attempts` times — an
    // operator hitting cert rotation would pound the backend with failing
    // handshakes for the full retry budget. The fix downcasts through the
    // source chain and short-circuits to `InvalidResponse`, NOT retried.

    // The storage-side TLS test needs `PermissionsClient` and `Permission`
    // in scope so it can invoke the same trait method production callers
    // use. The userinfo-side test reuses `UserinfoClient` already imported
    // at the module top.
    use crate::auth::permissions::PermissionsClient as _PermissionsClient;
    use crate::auth::Permission as _Permission;

    /// Spawn a TCP listener that accepts a connection, lets the client
    /// write its TLS ClientHello, then replies with a well-formed TLS
    /// fatal-alert record (`handshake_failure`). rustls processes this as
    /// `rustls::Error::AlertReceived(HandshakeFailure)` — exactly the
    /// shape the `is_tls_error` downcast catches.
    ///
    /// Why not "send garbage HTTP bytes"? Pre-fix attempts revealed reqwest
    /// classifies non-TLS server responses as `is_connect()` (the TCP
    /// connection succeeded but the protocol bridge failed), which masks
    /// the rustls error in the source chain. A valid-shape TLS alert
    /// guarantees the rustls state machine produces an explicit
    /// `rustls::Error` that propagates intact.
    async fn spawn_tls_alert_server() -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // Accept connections in a loop in case the client retries.
            // (The test asserts there is no retry; the loop is purely
            // defensive — a regression that re-enables retry should not
            // hang on a missing accept.)
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => return,
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    // Read at least the 5-byte TLS record header from the
                    // client's ClientHello so the client has had a chance
                    // to enter the TLS state machine before we reply.
                    let mut buf = [0u8; 5];
                    let _ = socket.read_exact(&mut buf).await;

                    // TLS Alert record:
                    //   ContentType: 0x15 (Alert)
                    //   Version:     0x03 0x03 (TLS 1.2 — accepted by rustls)
                    //   Length:      0x00 0x02
                    //   Level:       0x02 (fatal)
                    //   Description: 0x28 (handshake_failure)
                    const TLS_ALERT: [u8; 7] = [0x15, 0x03, 0x03, 0x00, 0x02, 0x02, 0x28];
                    let _ = socket.write_all(&TLS_ALERT).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        addr
    }

    #[tokio::test]
    async fn stat_tls_handshake_failure_classified_as_invalid_response_not_retried() {
        // The pre-fix bug: `err.status().is_none()` fell through to
        // `Unreachable` → `is_transient` → retried. The fix inspects the
        // source chain for `rustls::Error` and short-circuits to
        // `InvalidResponse` (permanent). We assert two properties:
        // (a) the returned error is `InvalidResponse`, and
        // (b) the call count is exactly 1 — no retries happened.
        let addr = spawn_tls_alert_server().await;
        let endpoint = format!("https://127.0.0.1:{}", addr.port());

        // Use 5 attempts so a regression that classifies TLS as transient
        // is unambiguous: pre-fix would issue 5 connection attempts; the
        // fix issues exactly 1.
        let client = RestStorageServiceClient::new(&endpoint, 200, fast_retry(5)).unwrap();
        let start = std::time::Instant::now();
        let err = _PermissionsClient::check_permission(&client, "tok", "/p", _Permission::Read)
            .await
            .unwrap_err();
        let elapsed = start.elapsed();

        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(
                    msg.contains("TLS handshake failed"),
                    "TLS errors must surface as InvalidResponse with a TLS-tagged message, got: {msg}"
                );
                assert!(
                    msg.contains("not retried"),
                    "the operator-facing message should say 'not retried' so log readers \
                     understand why the retry loop stopped, got: {msg}"
                );
            }
            other => {
                panic!("TLS handshake failure must classify as InvalidResponse, got {other:?}")
            }
        }

        // Even with `fast_retry(5)`'s 1ms initial / 10ms max delays, five
        // attempts plus inter-retry sleeps would take ~50ms+ here. A
        // single-attempt TLS failure typically completes well under 100ms.
        // Use a generous bound so we don't flake on slow hosts but still
        // catch a regression that issues 5 retries.
        assert!(
            elapsed < Duration::from_millis(800),
            "TLS handshake failure should not retry — elapsed time {elapsed:?} suggests a retry storm"
        );
    }

    #[tokio::test]
    async fn userinfo_tls_handshake_failure_classified_as_invalid_response_not_retried() {
        let addr = spawn_tls_alert_server().await;
        let endpoint = format!("https://127.0.0.1:{}", addr.port());
        let client = RestUserinfoClient::new(&endpoint, 200, fast_retry(5)).unwrap();

        let start = std::time::Instant::now();
        let err = client.get_current_user("tok").await.unwrap_err();
        let elapsed = start.elapsed();

        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(
                    msg.contains("TLS handshake failed"),
                    "userinfo TLS errors must classify as InvalidResponse, got: {msg}"
                );
            }
            other => panic!("expected InvalidResponse, got {other:?}"),
        }
        assert!(
            elapsed < Duration::from_millis(800),
            "no retry on TLS handshake failure — elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn userinfo_unknown_extra_fields_parse_cleanly() {
        // The DTO does not use `#[serde(deny_unknown_fields)]`, so extra
        // fields in the JSON must not break parsing.
        let mock_server = MockServer::start().await;
        let body = r#"{"id":"u1","display_name":"Jane","email":"j@x.com","username":"jane","extra_unknown_field":"ignored","another":42}"#;
        Mock::given(method("GET"))
            .respond_with(json_response(body))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let info = client.get_current_user("tok").await.unwrap();
        assert_eq!(info.id, "u1");
        assert_eq!(info.username, "jane");
    }

    // ── b.j5t item 2: REST userinfo response body size cap ─────────────────
    //
    // Pre-fix `response.json::<UserInfoDto>()` buffered arbitrary response
    // sizes; a malicious or broken backend could OOM the service. The fix
    // streams the body with an inline 64 KiB ceiling and bails out the
    // moment the cap is exceeded. Two attack shapes are covered here:
    //
    //   1. Honest Content-Length header > cap → rejected before reading
    //      bytes (fast path).
    //   2. Chunked transfer-encoding without Content-Length → streaming
    //      read accumulates and bails inside the chunk loop.

    /// Build a 200 KiB JSON body that's valid JSON (deserialization is
    /// never reached because the cap fires first) but exceeds the 64 KiB
    /// cap.
    fn oversized_userinfo_body(bytes: usize) -> String {
        // Pad the `display_name` field with a long ASCII run so the total
        // body weight crosses the cap.
        let padding = "x".repeat(bytes);
        format!(r#"{{"id":"u1","display_name":"{padding}","email":"j@x.com","username":"jane"}}"#,)
    }

    #[tokio::test]
    async fn userinfo_response_body_exceeding_cap_via_content_length_returns_invalid_response() {
        // wiremock sets Content-Length from the body bytes, so this test
        // exercises the fast-path cap check based on the declared length.
        let mock_server = MockServer::start().await;
        // 200 KiB — well above the 64 KiB cap.
        let oversized = oversized_userinfo_body(200 * 1024);
        Mock::given(method("GET"))
            .and(path("/api/v1alpha/users/me"))
            .respond_with(json_response(&oversized))
            .expect(1)
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(1));
        let err = client.get_current_user("tok").await.unwrap_err();
        match err {
            AuthError::InvalidResponse(msg) => {
                assert!(
                    msg.contains("cap"),
                    "expected an oversized-body InvalidResponse, got: {msg}"
                );
            }
            other => {
                panic!("oversized response body must classify as InvalidResponse, got {other:?}")
            }
        }
        // Critically: no OOM. The test reaching this assertion at all is the
        // primary regression signal — the pre-fix code would buffer the
        // entire body before returning.
    }

    #[tokio::test]
    async fn userinfo_response_at_cap_boundary_still_parses_when_valid_json() {
        // A response *under* the cap parses cleanly. We use a body that's
        // ~32 KiB (well under 64 KiB) but larger than the happy path's
        // single-line JSON, to exercise the streaming chunk loop on a
        // realistic-sized payload.
        let mock_server = MockServer::start().await;
        let padding = "y".repeat(32 * 1024);
        let body = format!(
            r#"{{"id":"u1","display_name":"{padding}","email":"j@x.com","username":"jane"}}"#,
        );
        Mock::given(method("GET"))
            .respond_with(json_response(&body))
            .mount(&mock_server)
            .await;

        let client = start_rest_userinfo_mock(&mock_server, fast_retry(3));
        let info = client.get_current_user("tok").await.unwrap();
        assert_eq!(info.id, "u1");
        assert!(info.display_name.len() >= 32 * 1024);
    }

    // ── b.j5t item 2 (streaming cap): chunked transfer-encoding ─────────────
    //
    // Critical to AC2: the Content-Length-based fast path is an
    // optimization, NOT the load-bearing line of defense. A malicious or
    // broken userinfo backend can omit Content-Length entirely (HTTP/1.1
    // `Transfer-Encoding: chunked`) or lie about it, and the cap must
    // still fire while the body streams in. Pre-fix the call buffered
    // arbitrary bytes via `response.json()` — an attacker could OOM the
    // service by trickling out megabytes of valid-shape JSON under the
    // request timeout. The fix's `read_bounded_body` walks chunks and
    // bails the moment the accumulated byte count crosses the cap.
    //
    // Strategy: a hand-rolled TCP listener that speaks HTTP/1.1 directly.
    // wiremock cannot construct a chunked response without Content-Length
    // — it always populates the latter from the body bytes — so we drop
    // to raw sockets, identical in shape to `spawn_tls_alert_server` above.

    /// Spawn an HTTP/1.1 server on a random local port. It accepts a
    /// single connection, reads the client's request line + headers, then
    /// responds with a `200 OK` whose body is sent in HTTP/1.1 chunked
    /// transfer-encoding *without* a Content-Length header. The server
    /// writes 128 KiB worth of JSON in 8 KiB chunks (so the cap is
    /// guaranteed to be crossed across multiple `chunk().await` iterations,
    /// not a single jumbo chunk). The server never sends the
    /// zero-length terminating chunk — it just drops the connection once
    /// it's written enough bytes, simulating a malicious backend that
    /// keeps feeding the client garbage.
    ///
    /// Returns the bound address. The accept task self-terminates after
    /// one connection.
    async fn spawn_chunked_oversized_userinfo_server() -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // One connection is enough — the client will not retry an
            // InvalidResponse (it's non-transient).
            let (mut socket, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => return,
            };
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            // Drain the request line + headers. HTTP/1.1 ends headers
            // with `\r\n\r\n`. Read until we see it, with a small buffer.
            let mut req_buf = Vec::with_capacity(1024);
            let mut tmp = [0u8; 256];
            loop {
                match socket.read(&mut tmp).await {
                    Ok(0) => return,
                    Ok(n) => {
                        req_buf.extend_from_slice(&tmp[..n]);
                        if req_buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                        if req_buf.len() > 8192 {
                            // Don't read indefinitely if the client
                            // somehow doesn't send a well-formed request.
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }

            // Send the status line + headers — chunked, no Content-Length.
            let headers = b"HTTP/1.1 200 OK\r\n\
                            Content-Type: application/json\r\n\
                            Transfer-Encoding: chunked\r\n\
                            \r\n";
            if socket.write_all(headers).await.is_err() {
                return;
            }

            // Write 128 KiB total in 8 KiB chunks. The 64 KiB cap fires
            // partway through the loop — `read_bounded_body` returns
            // InvalidResponse and drops the connection, so a `write_all`
            // failure after that is expected and benign.
            //
            // Chunk format (per RFC 7230 §4.1):
            //   <hex size>\r\n
            //   <payload bytes>\r\n
            let chunk_size_hex = format!("{:x}\r\n", 8 * 1024);
            let payload = vec![b'x'; 8 * 1024];
            for _ in 0..16 {
                if socket.write_all(chunk_size_hex.as_bytes()).await.is_err() {
                    return;
                }
                if socket.write_all(&payload).await.is_err() {
                    return;
                }
                if socket.write_all(b"\r\n").await.is_err() {
                    return;
                }
            }

            // Intentionally never write the zero-length terminating
            // chunk (`0\r\n\r\n`). The client should have bailed long
            // before reaching the loop end.
            let _ = socket.shutdown().await;
        });
        addr
    }

    /// Streaming-cap regression: a backend speaking chunked
    /// transfer-encoding (no Content-Length) that streams a body
    /// exceeding the 64 KiB cap must trigger `read_bounded_body`'s
    /// in-loop guard and return `AuthError::InvalidResponse` with the
    /// "during streaming read" tag — proving the streaming code path
    /// fired rather than the Content-Length fast path.
    #[tokio::test]
    async fn userinfo_response_chunked_oversized_body_caught_on_streaming_path() {
        let addr = spawn_chunked_oversized_userinfo_server().await;
        let endpoint = format!("http://127.0.0.1:{}", addr.port());

        // Generous per-call timeout so the timeout itself never fires
        // ahead of the cap. If pre-fix code regressed back in, the test
        // would either OOM (no cap at all) or hit this timeout. The cap
        // path fires immediately on the first oversized accumulation, so
        // a fast bound is plenty.
        let client = RestUserinfoClient::new(&endpoint, 5_000, fast_retry(1)).unwrap();
        let start = std::time::Instant::now();
        let err = client.get_current_user("tok").await.unwrap_err();
        let elapsed = start.elapsed();

        match err {
            AuthError::InvalidResponse(msg) => {
                // The "streaming read" sub-string is the load-bearing
                // proof — that exact phrasing is only emitted by
                // `read_bounded_body`, not by the Content-Length
                // fast-path branch. Distinguishing the two ensures this
                // test cannot pass via the fast path even if a future
                // refactor accidentally adds a Content-Length header.
                assert!(
                    msg.contains("streaming read"),
                    "oversized chunked body must trip the streaming-path cap, got: {msg}"
                );
                assert!(
                    msg.contains("cap"),
                    "message must reference the cap, got: {msg}"
                );
            }
            other => {
                panic!("oversized chunked body must classify as InvalidResponse, got {other:?}")
            }
        }

        // The cap fires inside `read_bounded_body` after only a few 8 KiB
        // chunks. A timeout-driven failure would take ~5s. Anything under
        // a second is unambiguously the cap firing, not a timeout.
        assert!(
            elapsed < Duration::from_millis(2_000),
            "streaming cap must fire promptly — elapsed {elapsed:?} suggests the cap was bypassed"
        );
    }

    // ── b.vvs item 9: classify_request_error scrubs URL from wrapped error ──
    //
    // `reqwest::Error::Display` includes the full request URL by default. The
    // storage-service URL embeds the *caller-supplied* path as an opaque
    // resource address — e.g. a customer's full object URL
    // `https://<storage-host>/<container>/<filename>` — which would
    // otherwise leak into:
    //
    //   * operator logs (the `InvalidResponse(...)` string flows through
    //     `tracing::error!` at the gRPC layer),
    //   * the auth cache's negative-decision entries (the rendered error is
    //     stored as the cache value's `Display` payload for the negative TTL
    //     window — pre-fix every cached failure carried the full URL).
    //
    // ### Why this test is a source-grep instead of an end-to-end runtime test
    //
    // The URL-scrub branch in `classify_request_error` only fires when the
    // `reqwest::Error` has `is_timeout=false`, is not a TLS error, has
    // `is_connect=false`, AND `err.status().is_some()`. In practice, the
    // only `reqwest::Error` instances that satisfy `status().is_some()` come
    // from `.error_for_status()` chains — which this codebase does NOT use
    // (HTTP status is handled explicitly via `classify_status` against the
    // status code, with the response body read separately). The branch is
    // therefore *defensive* — coverage for a future refactor that introduces
    // an `.error_for_status()` call site or a middleware that produces
    // status-bearing transport errors. Multiple attempts to drive the branch
    // via runtime fixtures (truncated Content-Length, chunked-without-end,
    // raw response.bytes() against a half-closed connection) all landed in
    // `Unreachable` instead, because reqwest does not associate a status
    // with body-read errors on the streaming path.
    //
    // The source-grep is the least-bad regression guard: it cannot detect
    // a *runtime* hole, but it does catch the specific failure mode b.vvs
    // item 9 set out to prevent — a future PR that drops `without_url()` or
    // re-introduces `{err}` (without the scrub) on the URL-bearing branch.
    // Coupling the assertion to the *function name* `classify_request_error`
    // narrows the source check to the exact call site, so an accidental
    // broadening of the scrub elsewhere doesn't make this test pass for the
    // wrong reason.

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn classify_request_error_calls_without_url_on_the_status_bearing_branch() {
        // Source-level regression for b.vvs item 9. The load-bearing
        // invariants pinned here:
        //
        //   1. The function `classify_request_error` exists and references
        //      `without_url` — if a future refactor strips the call entirely
        //      (e.g. "simplify the error wrapping"), this assertion fires.
        //
        //   2. The status-bearing branch (the one that fires after the
        //      timeout / TLS / connect / status-none short-circuits) wraps a
        //      *scrubbed* error variable named `scrubbed` rather than the
        //      raw `{err}`. Pinning the variable name `scrubbed` instead of
        //      a substring like `without_url` keeps the assertion semantic
        //      — anyone renaming the variable as part of a deliberate
        //      cleanup will see this test fail and have a clear signal that
        //      the contract being protected is "the format-arg must be the
        //      scrubbed value".
        let source = include_str!("rest.rs");

        // Slice the source to just the classify_request_error body so the
        // assertion can't spuriously match `without_url` mentions elsewhere
        // (e.g. in test code below). The function is fewer than 50 lines.
        let fn_start = source
            .find("fn classify_request_error(")
            .expect("classify_request_error must exist (b.vvs item 9)");
        // The function ends before the next `fn ` at column 0; conservative
        // upper bound at +4 KB is plenty for a ~40-line helper.
        let body = &source[fn_start..fn_start + 4096.min(source.len() - fn_start)];

        assert!(
            body.contains("without_url"),
            "b.vvs item 9: classify_request_error must call \
             `reqwest::Error::without_url()` on the status-bearing branch; got body:\n{body}"
        );
        assert!(
            body.contains("let scrubbed"),
            "b.vvs item 9: the URL-scrub variable must be bound as `let scrubbed = ...`; \
             a regression that inlines `format!(\"...{{err}}\")` without `without_url()` \
             would skip the scrub. Body:\n{body}"
        );
        // Negative pin: the InvalidResponse format on the status-bearing
        // branch must NOT directly interpolate the raw `err` value. Pre-fix
        // the code did `format!(\"... {err}\")` — the regression we're
        // guarding against is exactly that shape returning.
        //
        // We pin the post-fix pattern: the format string interpolates
        // `{scrubbed}`. A regression that renames the variable but keeps
        // the scrub is still safe; a regression that drops the scrub and
        // interpolates `{err}` again fails this assertion.
        assert!(
            body.contains("{scrubbed}"),
            "b.vvs item 9: InvalidResponse format must interpolate the scrubbed \
             value (`{{scrubbed}}`), not the raw `{{err}}` value that includes the URL. \
             Body:\n{body}"
        );
    }
}
