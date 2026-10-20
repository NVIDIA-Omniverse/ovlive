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

//! Userinfo fallback: a principal the userinfo service does not list gets the
//! display name its own JWT claims give it.
//!
//! The userinfo lookup only picks the name shown for a caller who has already
//! been authorized on the same RPC; "the directory holds no user for this
//! principal" ([`AuthError::UserNotFound`]) is therefore not a reason to fail
//! the RPC. [`UserinfoFallbackClient`] wraps the REST or gRPC userinfo backend
//! and answers that one outcome with [`JwtUserinfoClient`]'s claim chain on
//! the same bearer.
//!
//! **Placement.** `build_userinfo_client` builds it directly around the
//! backend, beneath `InstrumentedUserinfoClient` and `AuthCache`. A fallback
//! is therefore an `Ok` to both: a success duration sample (never an auth
//! error), and a positive-cache entry bounded by the bearer's `exp`, so the
//! userinfo service is not re-asked every negative-cache TTL. With the cache's
//! singleflight, the WARN and counter fire once per bearer per positive-cache
//! life, per replica.
//!
//! **Built only with authorization on.** The claim decode trusts the bearer
//! only because the permissions service validated it on the same RPC (see the
//! `SECURITY` comment in `userinfo_jwt`); with authorization off nothing
//! validated it, so the builder returns the bare backend and the not-found
//! answer fails the RPC as before.
//!
//! **Only `UserNotFound` falls back.** Every other result passes through
//! untouched: a userinfo 401/403 can mean the userinfo service's own
//! credential failed for every caller, other permanent errors are deployment
//! or contract faults, and transient errors already end the RPC retryable
//! `UNAVAILABLE` — a cached fallback there would pin a claim-derived name past
//! the outage.

use std::sync::Arc;

use async_trait::async_trait;

use crate::auth::userinfo::UserinfoClient;
use crate::auth::{AuthError, JwtUserinfoClient, UserInfo};
use crate::observability::metrics::SharedMetrics;

/// Userinfo decorator that answers the userinfo service's not-found outcome
/// from the bearer's JWT claims. See the module docs.
pub struct UserinfoFallbackClient {
    inner: Arc<dyn UserinfoClient>,
    claims: JwtUserinfoClient,
    metrics: SharedMetrics,
}

impl UserinfoFallbackClient {
    /// Wrap `inner`, the REST or gRPC userinfo backend.
    pub fn new(inner: Arc<dyn UserinfoClient>, metrics: SharedMetrics) -> Self {
        Self {
            inner,
            claims: JwtUserinfoClient::new(),
            metrics,
        }
    }
}

#[async_trait]
impl UserinfoClient for UserinfoFallbackClient {
    async fn get_current_user(&self, token: &str) -> Result<UserInfo, AuthError> {
        match self.inner.get_current_user(token).await {
            Err(AuthError::UserNotFound) => match self.claims.get_current_user(token).await {
                Ok(info) => {
                    // Emitted only once the fallback has succeeded, and with
                    // no `.await` after it: a caller cancelled before this
                    // point emits nothing, and `AuthCache` caches the result
                    // in the same poll that resolves it, so every emission
                    // matches one positive-cache entry. `principal_id` is the
                    // claim id (`oid`, else `sub`) so an operator can find the
                    // principal; never the bearer or the claim-derived name.
                    tracing::warn!(
                        principal_id = %info.id,
                        "userinfo service has no user for the caller; using the display name \
                         from the bearer's JWT claims"
                    );
                    self.metrics.inc_userinfo_fallback();
                    Ok(info)
                }
                // The claim chain could not name the caller (an opaque bearer,
                // or a JWT with no usable claim): keep the original outcome.
                // Unreachable with the userinfo service this targets, whose
                // not-found answer implies a decodable `oid`.
                Err(_) => Err(AuthError::UserNotFound),
            },
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use tracing::instrument::WithSubscriber;

    use super::UserinfoFallbackClient;
    use crate::auth::cache::AuthCache;
    use crate::auth::permissions::mock::MockPermissionsClient;
    use crate::auth::userinfo::mock::MockUserinfoClient;
    use crate::auth::userinfo::UserinfoClient;
    use crate::auth::{AuthError, AuthResult, UserInfo};
    use crate::observability::metrics::metrics_scrape_only;
    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{Metrics, SharedMetrics};

    const FALLBACKS: &str = "live_edit_userinfo_fallbacks_total";
    /// The WARN line's message; the log line is a deliverable of the fallback
    /// (test-writing guide § What NOT to Test, carve-out).
    const FALLBACK_WARN_MSG: &str = "userinfo service has no user for the caller; using the \
                                     display name from the bearer's JWT claims";

    /// Entra object id of the app's service principal (invented).
    const APP_OID: &str = "11111111-2222-3333-4444-555555555555";
    const APP_DISPLAY_NAME: &str = "Asset Agent";

    /// An unsigned JWT carrying `claims`; the decorator decodes it without
    /// verification (the permissions service validated it on the RPC).
    fn jwt(claims: serde_json::Value) -> String {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"none"}"#);
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        format!("{header}.{payload}.")
    }

    /// An Entra app-only access token (`idtyp=app`, `oid == sub`) whose `exp`
    /// is an hour out, so a cache bound by it never expires during a test.
    fn app_only_jwt() -> String {
        let exp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after the epoch")
            .as_secs()
            + 3600;
        jwt(serde_json::json!({
            "idtyp": "app",
            "app_displayname": APP_DISPLAY_NAME,
            "azp": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "oid": APP_OID,
            "sub": APP_OID,
            "exp": exp,
        }))
    }

    fn directory_user() -> UserInfo {
        UserInfo {
            id: "u1".into(),
            display_name: "Jane".into(),
            email: None,
            username: "jane".into(),
        }
    }

    /// The decorator over a mock backend answering `inner`, the mock's call
    /// log, and the metrics it counts on.
    fn decorator(
        inner: Result<UserInfo, AuthError>,
    ) -> (
        UserinfoFallbackClient,
        Arc<Mutex<Vec<crate::auth::userinfo::mock::UserinfoCall>>>,
        SharedMetrics,
    ) {
        let backend = MockUserinfoClient::new(inner);
        let calls = Arc::clone(&backend.calls);
        let metrics = Arc::new(metrics_scrape_only("fallback-test"));
        let client = UserinfoFallbackClient::new(Arc::new(backend), Arc::clone(&metrics));
        (client, calls, metrics)
    }

    fn fallbacks(metrics: &Metrics) -> f64 {
        sample_value(&render(metrics), FALLBACKS, &[]).expect("the counter is seeded at zero")
    }

    #[derive(Clone)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Await `fut` under a JSON capture subscriber (`WithSubscriber`, so the
    /// capture follows the future across polls) and return its output and
    /// every event it emitted, parsed. JSON so an absent field is a missing
    /// key and every field value can be inspected structurally.
    async fn capture_json<T>(fut: impl Future<Output = T>) -> (T, Vec<serde_json::Value>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(Arc::clone(&buf)))
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        let out = fut.with_subscriber(subscriber).await;
        let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let events = String::from_utf8(raw)
            .expect("captured tracing output must be UTF-8")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect();
        (out, events)
    }

    fn fallback_warns(events: &[serde_json::Value]) -> Vec<&serde_json::Value> {
        events
            .iter()
            .filter(|e| e["fields"]["message"] == FALLBACK_WARN_MSG)
            .collect()
    }

    #[tokio::test]
    async fn get_current_user_user_not_found_with_app_only_jwt_returns_claim_label_counts_and_warns(
    ) {
        let (client, calls, metrics) = decorator(Err(AuthError::UserNotFound));
        let bearer = app_only_jwt();

        let (result, events) = capture_json(client.get_current_user(&bearer)).await;

        let info = result.expect("a not-found answer must fall back to the JWT claims");
        assert_eq!(info.display_name, format!("{APP_DISPLAY_NAME} (app)"));
        assert_eq!(info.id, APP_OID);
        assert_eq!(calls.lock().unwrap().len(), 1, "the backend is asked once");
        assert_eq!(
            fallbacks(&metrics),
            1.0,
            "one fallback ticks the counter once"
        );

        let warns = fallback_warns(&events);
        assert_eq!(
            warns.len(),
            1,
            "exactly one fallback WARN; events: {events:?}"
        );
        let warn = warns[0];
        assert_eq!(warn["level"], "WARN");
        assert_eq!(warn["fields"]["principal_id"], APP_OID);
        let fields = warn["fields"]
            .as_object()
            .expect("the event's fields are an object");
        let mut field_names: Vec<&str> = fields.keys().map(String::as_str).collect();
        field_names.sort_unstable();
        assert_eq!(
            field_names,
            ["message", "principal_id"],
            "the WARN carries the principal id and nothing else"
        );
        for (name, value) in fields {
            let value = value.to_string();
            assert!(
                !value.contains(&bearer),
                "field {name} must not carry the bearer: {value}"
            );
            assert!(
                !value.contains(APP_DISPLAY_NAME),
                "field {name} must not carry the claim-derived name: {value}"
            );
        }
    }

    #[tokio::test]
    async fn get_current_user_inner_ok_passes_through_without_counting() {
        let (client, _, metrics) = decorator(Ok(directory_user()));

        let (result, events) = capture_json(client.get_current_user(&app_only_jwt())).await;

        let info = result.expect("a directory answer passes through");
        assert_eq!(
            info.display_name, "Jane",
            "the directory's name, not the claims'"
        );
        assert_eq!(info.id, "u1");
        assert_eq!(fallbacks(&metrics), 0.0);
        assert!(fallback_warns(&events).is_empty(), "events: {events:?}");
    }

    #[tokio::test]
    async fn get_current_user_inner_errors_other_than_user_not_found_pass_through_without_counting()
    {
        // A userinfo 401 can mean the userinfo service's own credential failed
        // for every caller; transient errors must stay retryable. None of them
        // falls back, even with a bearer the claim chain could name.
        for inner in [
            AuthError::InvalidResponse("userinfo service rejected the bearer (HTTP 401)".into()),
            AuthError::Timeout,
            AuthError::Unreachable,
        ] {
            let (client, _, metrics) = decorator(Err(inner.clone()));

            let (result, events) = capture_json(client.get_current_user(&app_only_jwt())).await;

            let err = result.expect_err("only UserNotFound falls back");
            assert_eq!(
                format!("{err:?}"),
                format!("{inner:?}"),
                "the error passes through unchanged"
            );
            assert_eq!(fallbacks(&metrics), 0.0, "{inner:?} must not count");
            assert!(
                fallback_warns(&events).is_empty(),
                "{inner:?} must not warn; events: {events:?}"
            );
        }
    }

    #[tokio::test]
    async fn get_current_user_user_not_found_with_opaque_bearer_stays_user_not_found() {
        let (client, _, metrics) = decorator(Err(AuthError::UserNotFound));

        let (result, events) = capture_json(client.get_current_user("opaque-bearer")).await;

        let err = result.expect_err("an opaque bearer has no claims to fall back to");
        assert!(matches!(err, AuthError::UserNotFound), "got {err:?}");
        assert_eq!(fallbacks(&metrics), 0.0);
        assert!(fallback_warns(&events).is_empty(), "events: {events:?}");
    }

    /// Placement: `AuthCache` over the decorator over the backend, as
    /// `build_userinfo_client` + `main` wire it. The fallback is an `Ok` to
    /// the cache, so it is positive-cached under the bearer's `exp`: a second
    /// lookup asks neither the backend nor the decorator. Were the decorator
    /// above the cache, the backend's `UserNotFound` would be negative-cached
    /// and the decorator would fall back (and count) on every call; were the
    /// fallback negative-cached, the second call would fail.
    #[tokio::test]
    async fn auth_cache_over_fallback_positive_caches_the_fallback() {
        let backend = MockUserinfoClient::new(Err(AuthError::UserNotFound));
        let backend_calls = Arc::clone(&backend.calls);
        let metrics = Arc::new(metrics_scrape_only("fallback-placement"));
        let fallback = UserinfoFallbackClient::new(Arc::new(backend), Arc::clone(&metrics));
        // Positive and negative TTLs far longer than the test, so only the
        // cache's arm choice can decide whether the backend is asked again.
        let cache = AuthCache::new(
            Arc::new(MockPermissionsClient::new(Ok(AuthResult::Allowed))),
            Arc::new(fallback),
            600,
            1_000,
        );
        let bearer = app_only_jwt();

        let first = cache
            .get_current_user(&bearer)
            .await
            .expect("first lookup falls back");
        let second = cache
            .get_current_user(&bearer)
            .await
            .expect("second lookup is served the cached fallback");

        assert_eq!(first.display_name, format!("{APP_DISPLAY_NAME} (app)"));
        assert_eq!(second.display_name, first.display_name);
        assert_eq!(
            backend_calls.lock().unwrap().len(),
            1,
            "the backend is asked once"
        );
        assert_eq!(fallbacks(&metrics), 1.0, "one fallback per cached entry");
    }
}
