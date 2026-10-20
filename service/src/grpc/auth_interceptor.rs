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

// Auth interceptor: extracts the bearer token from the `authorization` metadata
// header and inserts it into request extensions as a `BearerToken`.
//
// This interceptor only *extracts* the token. It does not validate or enforce
// permissions — that is done by individual RPC handlers calling the external
// permissions service (SDD §5.2, §14).
//
// A rejection here happens before any handler runs, so this is the completion
// point that logs it: one WARN `RPC failed` at `stage=header` naming the
// endpoint and the [`AuthRejection`] (SDD §17.2.1). A tonic interceptor never
// sees the request path, so the endpoint comes from the `RpcCompletion` ledger
// `super::metrics_layer` inserted into the request extensions.
//
// Security: the header value, and so the token, is never logged or traced at
// any level (SDD §23.3) — the rejection line carries only the fixed
// `AuthRejection` label.

use std::sync::Arc;

use tonic::{Request, Status};

use super::metrics_layer::OTHER_ENDPOINT;
use super::rpc_log::{log_rpc_failure, RpcCompletion, RpcLogContext};
use super::BearerToken;

/// Which header check refused the request. One value per rejection site;
/// emitted as the `rejection` field of the rejection line. The wire answer is
/// the same bare `UNAUTHENTICATED` for all four.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AuthRejection {
    /// No `authorization` header.
    MissingHeader,
    /// The header value is not visible ASCII.
    InvalidHeaderValue,
    /// The `Bearer ` prefix is present but nothing follows it.
    EmptyToken,
    /// The header does not start with `Bearer ` (case-insensitive).
    NotBearer,
}

impl AuthRejection {
    /// Stable lowercase label for the `rejection` log field.
    pub(crate) fn label(self) -> &'static str {
        match self {
            AuthRejection::MissingHeader => "missing_header",
            AuthRejection::InvalidHeaderValue => "invalid_header_value",
            AuthRejection::EmptyToken => "empty_token",
            AuthRejection::NotBearer => "not_bearer",
        }
    }
}

/// Tonic interceptor that extracts the `authorization` header bearer token.
///
/// On success, inserts `BearerToken` into request extensions so RPC handlers
/// can forward the token to the permissions service without re-parsing headers.
/// Nothing is logged on success.
///
/// Returns `UNAUTHENTICATED` with an empty message, and logs one rejection
/// line, if:
/// - The `authorization` header is absent.
/// - The header value is not valid ASCII.
/// - The header does not start with `Bearer ` (case-insensitive).
/// - The token after stripping the `Bearer ` prefix is empty.
//
// `result_large_err`: tonic::Status is large, but this is the required return
// type for a tonic interceptor — we cannot box it.
#[allow(clippy::result_large_err)]
pub fn auth_interceptor(mut request: Request<()>) -> Result<Request<()>, Status> {
    match extract_bearer_token(&request) {
        Ok(token) => {
            request.extensions_mut().insert(BearerToken::new(token));
            Ok(request)
        }
        Err(rejection) => {
            let status = Status::unauthenticated("");
            log_rejection(&request, rejection, &status);
            Err(status)
        }
    }
}

/// Emit the rejection's completion line and mark the request's ledger
/// reported, so the metrics layer's backstop does not log it again.
///
/// Absent a ledger — the interceptor called outside `super::metrics_layer`,
/// which only unit tests do — the endpoint is [`OTHER_ENDPOINT`].
fn log_rejection(request: &Request<()>, rejection: AuthRejection, status: &Status) {
    let endpoint = request
        .extensions()
        .get::<Arc<RpcCompletion>>()
        .map_or(OTHER_ENDPOINT, |completion| completion.endpoint());
    let mut ctx = RpcLogContext::for_request(endpoint, request);
    ctx.rejection(rejection.label());
    log_rpc_failure(&ctx, status);
}

/// Parse the `authorization` header and return the raw token string.
fn extract_bearer_token(request: &Request<()>) -> Result<String, AuthRejection> {
    let header_value = request
        .metadata()
        .get("authorization")
        .ok_or(AuthRejection::MissingHeader)?;

    let header_str = header_value
        .to_str()
        .map_err(|_| AuthRejection::InvalidHeaderValue)?;

    let token = strip_bearer_prefix(header_str)?;

    if token.is_empty() {
        return Err(AuthRejection::EmptyToken);
    }

    Ok(token.to_owned())
}

/// Strip the `Bearer ` prefix (case-insensitive on "Bearer").
///
/// Returns the token substring if the prefix is present, or
/// [`AuthRejection::NotBearer`] if the header value does not start with
/// `bearer ` (case-insensitive).
fn strip_bearer_prefix(header: &str) -> Result<&str, AuthRejection> {
    const PREFIX: &str = "bearer ";
    if header.len() >= PREFIX.len() && header[..PREFIX.len()].eq_ignore_ascii_case(PREFIX) {
        Ok(&header[PREFIX.len()..])
    } else {
        Err(AuthRejection::NotBearer)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tonic::metadata::MetadataValue;
    use tracing_subscriber::fmt::MakeWriter;

    use super::*;

    /// A `MakeWriter` that captures all bytes written to it into a shared buffer.
    #[derive(Clone)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

    impl CapturingWriter {
        fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
            let buf = Arc::new(Mutex::new(Vec::new()));
            (Self(buf.clone()), buf)
        }
    }

    impl<'a> MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn request_with_header(value: &str) -> Request<()> {
        let mut req = Request::new(());
        req.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from(value).expect("invalid metadata value"),
        );
        req
    }

    #[test]
    fn valid_bearer_token_is_extracted() {
        let req = request_with_header("Bearer eyABC123");
        let result = auth_interceptor(req).expect("should succeed");
        let token = result
            .extensions()
            .get::<BearerToken>()
            .expect("BearerToken must be in extensions");
        assert_eq!(token.as_str(), "eyABC123");
    }

    #[test]
    fn bearer_prefix_is_case_insensitive() {
        for prefix in ["bearer ", "BEARER ", "Bearer ", "bEaReR "] {
            let req = request_with_header(&format!("{prefix}token123"));
            let result = auth_interceptor(req).expect("case-insensitive prefix should succeed");
            let token = result.extensions().get::<BearerToken>().unwrap();
            assert_eq!(token.as_str(), "token123", "failed for prefix: {prefix:?}");
        }
    }

    #[test]
    fn missing_authorization_header_returns_unauthenticated() {
        let req = Request::new(());
        let err = auth_interceptor(req).expect_err("missing header should fail");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn empty_token_after_prefix_returns_unauthenticated() {
        let req = request_with_header("Bearer ");
        let err = auth_interceptor(req).expect_err("empty token should fail");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn header_without_bearer_prefix_returns_unauthenticated() {
        let req = request_with_header("Token abc123");
        let err = auth_interceptor(req).expect_err("non-Bearer scheme should fail");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
    }

    // ── b.jr8: behavior pinned when multiple `authorization` headers present ──
    //
    // The interceptor uses `metadata.get("authorization")` which returns only
    // the FIRST value — later `authorization` headers on the same request are
    // ignored. This test pins that behavior explicitly so any future refactor
    // (e.g. switching to `get_all`) has to deliberately break the contract,
    // not silently shift to a different "first vs last vs any" semantic.
    //
    // Rationale for "pin, don't change": gRPC / HTTP-2 allow multiple headers
    // with the same name, but for bearer auth a second authorization header is
    // almost certainly a client bug or a smuggling attempt. Neither merging
    // them nor selecting the second one gives a better security posture, and
    // "first wins" matches the default behavior of every other
    // `MetadataMap::get` caller in the service.

    #[test]
    fn multiple_authorization_headers_first_wins() {
        let mut req = Request::new(());
        req.metadata_mut().append(
            "authorization",
            MetadataValue::try_from("Bearer first-token").unwrap(),
        );
        req.metadata_mut().append(
            "authorization",
            MetadataValue::try_from("Bearer second-token").unwrap(),
        );
        let result = auth_interceptor(req).expect("first header must parse successfully");
        let token = result
            .extensions()
            .get::<BearerToken>()
            .expect("BearerToken must be in extensions");
        assert_eq!(
            token.as_str(),
            "first-token",
            "pinned behavior: first `authorization` header wins when multiple are present"
        );
    }

    #[test]
    fn multiple_authorization_headers_first_valid_second_invalid_still_first_wins() {
        // The second header intentionally does NOT start with `bearer ` — if
        // the interceptor ever regressed to "last wins", this would produce
        // UNAUTHENTICATED. We pin that the first (valid) header is the one
        // consulted regardless of what the second contains.
        let mut req = Request::new(());
        req.metadata_mut().append(
            "authorization",
            MetadataValue::try_from("Bearer good-token").unwrap(),
        );
        req.metadata_mut().append(
            "authorization",
            MetadataValue::try_from("NotBearer abc").unwrap(),
        );
        let result = auth_interceptor(req).expect("first header must parse successfully");
        let token = result
            .extensions()
            .get::<BearerToken>()
            .expect("BearerToken must be in extensions");
        assert_eq!(token.as_str(), "good-token");
    }

    #[test]
    fn multiple_authorization_headers_first_invalid_rejects_even_if_second_valid() {
        // Conversely, if the first header is invalid the interceptor must
        // reject — it does not scan forward to find a valid one. This pins
        // the rule that "first wins" is absolute, not a fallback.
        let mut req = Request::new(());
        req.metadata_mut().append(
            "authorization",
            MetadataValue::try_from("NotBearer abc").unwrap(),
        );
        req.metadata_mut().append(
            "authorization",
            MetadataValue::try_from("Bearer good-token").unwrap(),
        );
        let err = auth_interceptor(req).expect_err("invalid first header must reject");
        assert_eq!(err.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn bearer_token_debug_does_not_expose_value() {
        let token = BearerToken::new("super-secret");
        let debug_str = format!("{token:?}");
        assert!(
            !debug_str.contains("super-secret"),
            "token value must not appear in Debug output: {debug_str}"
        );
        assert!(debug_str.contains("***"));
    }

    /// Verify that the raw token string does not appear in any tracing output
    /// emitted while the interceptor processes a request (SDD §23.3).
    #[test]
    fn bearer_token_does_not_appear_in_tracing_output() {
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::TRACE)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            const SECRET: &str = "s3cr3t-t0k3n-value";
            let req = request_with_header(&format!("Bearer {SECRET}"));
            let _ = auth_interceptor(req);
        });

        let guard = captured.lock().unwrap();
        let output = String::from_utf8_lossy(&guard);
        assert!(
            !output.contains("s3cr3t-t0k3n-value"),
            "token value must not appear in tracing output: {output}"
        );
    }

    // ── Rejection line (SDD §17.2.1) ────────────────────────────────────────
    //
    // A rejection here happens before any handler runs, so the interceptor is
    // the completion point that must log it. The line is the deliverable (the
    // commonest client misconfiguration otherwise leaves only a counter tick),
    // so it is pinned structurally: level, message, every field, and the
    // ledger mark that stops the metrics layer's backstop logging it again.

    /// Secret material embedded in a rejected request. Distinctive so a sweep
    /// over the whole capture buffer cannot match anything else.
    const REJECTED_SECRET: &str = "rejected-s3cr3t-value-9c1e";

    /// Run `f` under a thread-local JSON capture subscriber, returning the
    /// parsed events and the raw text (the raw text is what the secret sweep
    /// reads, so a leak into a field this file never names is still caught).
    fn capture_json<T>(f: impl FnOnce() -> T) -> (T, Vec<serde_json::Value>, String) {
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(writer)
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        let value = tracing::subscriber::with_default(subscriber, f);
        let text = String::from_utf8(captured.lock().unwrap().clone())
            .expect("captured tracing output must be UTF-8");
        let events = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect();
        (value, events, text)
    }

    /// Insert the ledger the metrics layer would have inserted.
    fn with_ledger(
        mut req: Request<()>,
        endpoint: &'static str,
    ) -> (Request<()>, Arc<RpcCompletion>) {
        let completion = Arc::new(RpcCompletion::new(endpoint));
        req.extensions_mut().insert(Arc::clone(&completion));
        (req, completion)
    }

    /// A request whose `authorization` value is not visible ASCII.
    ///
    /// Built through `http::HeaderValue::from_bytes`, which admits obs-text
    /// (0x80..=0xFF) that `to_str()` then refuses — the only route to the
    /// `InvalidHeaderValue` rejection, since `MetadataValue::try_from(&str)`
    /// refuses such bytes up front.
    fn request_with_non_ascii_header(prefix_and_secret: &str) -> Request<()> {
        let mut bytes = prefix_and_secret.as_bytes().to_vec();
        bytes.push(0x80);
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "authorization",
            http::HeaderValue::from_bytes(&bytes).expect("obs-text is a legal header byte"),
        );
        let mut req = Request::new(());
        *req.metadata_mut() = tonic::metadata::MetadataMap::from_headers(headers);
        req
    }

    /// One rejected request per `AuthRejection`, each carrying
    /// [`REJECTED_SECRET`] somewhere the interceptor reads or could be tempted
    /// to log.
    fn rejection_cases() -> Vec<(AuthRejection, Request<()>)> {
        // Missing: no `authorization` header — the secret rides a misspelt
        // lookalike header, the classic client typo this rejection reports.
        let mut missing = Request::new(());
        missing.metadata_mut().insert(
            "authorisation",
            MetadataValue::try_from(format!("Bearer {REJECTED_SECRET}")).unwrap(),
        );
        // Empty token: the first header is a bare `Bearer ` (first wins), the
        // secret sits in a second one the interceptor never consults.
        let mut empty = Request::new(());
        empty
            .metadata_mut()
            .append("authorization", MetadataValue::try_from("Bearer ").unwrap());
        empty.metadata_mut().append(
            "authorization",
            MetadataValue::try_from(format!("Bearer {REJECTED_SECRET}")).unwrap(),
        );
        vec![
            (AuthRejection::MissingHeader, missing),
            (
                AuthRejection::InvalidHeaderValue,
                request_with_non_ascii_header(&format!("Bearer {REJECTED_SECRET}")),
            ),
            (AuthRejection::EmptyToken, empty),
            (
                AuthRejection::NotBearer,
                request_with_header(&format!("Token {REJECTED_SECRET}")),
            ),
        ]
    }

    /// Each rejection shape answers a bare `UNAUTHENTICATED` and logs exactly
    /// one WARN `RPC failed` naming the ledger's endpoint, `stage=header`, the
    /// empty status message and its own `rejection` label — and marks the
    /// ledger reported.
    #[test]
    fn each_rejection_shape_logs_one_warn_with_the_endpoint_and_its_rejection_label() {
        for (rejection, req) in rejection_cases() {
            let (req, completion) = with_ledger(req, "UpdateLayer");
            let (result, events, _text) = capture_json(|| auth_interceptor(req));

            let status = result.expect_err("every rejection shape must be refused");
            assert_eq!(status.code(), tonic::Code::Unauthenticated, "{rejection:?}");
            assert_eq!(
                status.message(),
                "",
                "{rejection:?}: the wire answer stays the bare UNAUTHENTICATED"
            );

            assert_eq!(
                events.len(),
                1,
                "{rejection:?}: exactly one line per rejection; got {events:?}"
            );
            let event = &events[0];
            assert_eq!(event["level"], "WARN", "{rejection:?}");
            let fields = &event["fields"];
            assert_eq!(fields["message"], super::super::rpc_log::RPC_FAILURE_MSG);
            assert_eq!(fields["endpoint"], "UpdateLayer", "{rejection:?}");
            assert_eq!(fields["stage"], "header", "{rejection:?}");
            assert_eq!(fields["grpc_code"], "UNAUTHENTICATED", "{rejection:?}");
            assert_eq!(fields["status_message"], "", "{rejection:?}");
            assert_eq!(fields["rejection"], rejection.label(), "{rejection:?}");
            for absent in ["session_id", "layer_name", "client_id", "base_path"] {
                assert!(
                    fields.get(absent).is_none(),
                    "{rejection:?}: the interceptor knows no request identifier; got {fields:?}"
                );
            }
            assert!(
                completion.is_reported(),
                "{rejection:?}: the ledger must be marked, or the metrics layer's backstop logs \
                 this UNAUTHENTICATED a second time off the trailer"
            );
        }
    }

    /// No rejection shape puts any byte of the header value — and so of the
    /// token — anywhere in the log output.
    ///
    /// Swept over the raw buffer, so a leak into a field nobody thought to
    /// check (a `Debug` of the request, the metadata, a future field) is
    /// caught.
    #[test]
    fn no_rejection_shape_logs_the_header_value() {
        for (rejection, req) in rejection_cases() {
            let (req, _completion) = with_ledger(req, "StatLayer");
            let (result, events, text) = capture_json(|| auth_interceptor(req));
            assert!(result.is_err(), "{rejection:?} must be refused");
            assert_eq!(events.len(), 1, "precondition: the line was emitted");
            assert!(
                !text.contains(REJECTED_SECRET),
                "{rejection:?}: header material reached the log: {text}"
            );
        }
    }

    /// Without a ledger — the interceptor driven outside the metrics layer —
    /// the line still fires, under the `other` endpoint bucket.
    #[test]
    fn rejection_without_a_ledger_logs_under_the_other_endpoint() {
        let (result, events, _text) = capture_json(|| auth_interceptor(Request::new(())));
        assert!(result.is_err());
        assert_eq!(events.len(), 1, "got {events:?}");
        assert_eq!(events[0]["fields"]["endpoint"], OTHER_ENDPOINT);
        assert_eq!(events[0]["fields"]["rejection"], "missing_header");
    }

    /// A successful extraction logs nothing, at any level, and leaves the
    /// ledger unmarked.
    ///
    /// Every authenticated RPC passes through here, so a line on success would
    /// be one per request; and a marked ledger would make the backstop skip a
    /// later tonic-level failure of the same call.
    #[test]
    fn successful_extraction_logs_nothing_and_leaves_the_ledger_unmarked() {
        let (req, completion) = with_ledger(request_with_header("Bearer ok-token"), "Ping");
        let (result, events, _text) = capture_json(|| auth_interceptor(req));
        assert!(result.is_ok());
        assert!(events.is_empty(), "success must be silent; got {events:?}");
        assert!(!completion.is_reported());
    }

    /// The four `rejection` labels are distinct, stable and lowercase — they
    /// are what an operator filters a denial burst by.
    #[test]
    fn auth_rejection_labels_are_distinct_and_stable() {
        let pairs = [
            (AuthRejection::MissingHeader, "missing_header"),
            (AuthRejection::InvalidHeaderValue, "invalid_header_value"),
            (AuthRejection::EmptyToken, "empty_token"),
            (AuthRejection::NotBearer, "not_bearer"),
        ];
        for (rejection, want) in pairs {
            assert_eq!(rejection.label(), want);
        }
        let mut labels: Vec<_> = pairs.iter().map(|(r, _)| r.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), pairs.len());
    }
}
