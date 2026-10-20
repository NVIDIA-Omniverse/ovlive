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

//! Delegating fred [`CredentialProvider`] that suppresses the periodic re-AUTH
//! cadence while keeping the credential itself byte-identical (b.j64, SDD §15.3).
//!
//! ## Why this exists
//!
//! Under the passwordless Valkey auth modes (`accessTokenFile` and `iam` alike —
//! this is mode-agnostic) fred spawns one credential-refresh task **per client**
//! whenever [`CredentialProvider::refresh_interval`] returns `Some`, and that
//! task re-issues `AUTH` on the interval forever
//! (`fred-10.1.0/src/modules/inner.rs:879-887`, `spawn_credential_refresh` at
//! `:386`). On the dedicated pub/sub [`SubscriberClient`](fred::clients::SubscriberClient)
//! that re-AUTH is **structurally impossible**: the connection is permanently in
//! RESP2 subscribe state (`main.rs` subscribes `le:session:deleted` at startup),
//! and Valkey answers any `AUTH` there with
//!
//! ```text
//! ERR Can't execute 'auth': only (P|S)SUBSCRIBE / (P|S)UNSUBSCRIBE / PING / QUIT / RESET are allowed in this context
//! ```
//!
//! fred classifies that reply as `ErrorKind::Unknown` (non-fatal): the
//! connection is not dropped, no metric moves, and the only effect is a `warn!`
//! at `inner.rs:423` every `[300 s, 360 s]` — roughly 288 WARNs per replica per
//! day, forever, on every passwordless deployment. That cadence trains operators
//! to ignore Valkey auth warnings, which is exactly the alarm-fatigue that let
//! the b.9uq delivery bug hide.
//!
//! ## Why suppressing the schedule is the correct fix, not log filtering
//!
//! The refresh on that one connection is **already a no-op**: the pub/sub client
//! renews its credential through the *reconnect* path instead —
//! `reconnect_on_auth_error = true` (set for every connection in
//! `apply_connect_timeout`) plus the handshake `AUTH`, which calls
//! [`CredentialProvider::fetch`] afresh, plus
//! `SubscriberClient::manage_subscriptions()` re-subscribing the tracked
//! channels. Nothing is lost by not scheduling a command the server refuses.
//!
//! Filtering the log instead was evaluated and disqualified: the only reachable
//! `EnvFilter` granularity is the target `fred::modules::inner`, whose three
//! `warn!` sites are *all* credential-refresh warnings, so a filter would also
//! mute a genuine `WRONGPASS` / token-unavailable warning on the data, txn,
//! maintenance and snapshot pools — the connections where fred's refresh is
//! load-bearing.
//!
//! ## The stream readers (b.19i)
//!
//! The same wrapper is installed on every stream reader client, for a
//! different reason: there fred's refresh *would* reach the server, but as a
//! second, independent task writing to a connection the reader keeps parked
//! in `XREAD BLOCK`, where it can queue behind the block. So fred schedules
//! nothing there, and each reader re-`AUTH`s its own connection between its
//! reads from the unwrapped provider (`storage::stream_mux`'s reader; see
//! `storage::client::derive_reader_config`).
//!
//! ## What is preserved
//!
//! [`Self::fetch`] delegates **verbatim** to the same inner `Arc`, so the
//! credential a wrapped connection presents at handshake is byte-identical to
//! the one every other connection presents. This wrapper narrows the *cadence*,
//! never the credential — which is why it does not weaken the "one
//! credential-provider policy across all six connections" guarantee (SDD §15.3).
//!
//! [`Debug`] forwards to the inner provider, so the bearer-bytes-never-logged
//! invariant (SR-6.2, extended to Valkey auth) is **inherited** rather than
//! re-implemented: both `TokenFileCredentialProvider` and `IamCredentialProvider`
//! already redact, and a future provider that redacts will redact here too.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use fred::error::Error as FredError;
use fred::types::config::{CredentialProvider, Server};

/// Wraps a fred [`CredentialProvider`] so fred never schedules a periodic
/// re-`AUTH` for the client it is installed on, while the credential itself is
/// unchanged.
///
/// Installed on two kinds of connection, each by its own derivation inside
/// `storage::client::build_fred_config`:
/// - the dedicated pub/sub `SubscriberClient`, by
///   `storage::client::derive_pubsub_config`, only under RESP2 (see that
///   function for the protocol gate) — it renews its credential on the
///   reconnect path;
/// - every stream reader client, by `storage::client::derive_reader_config`,
///   under any RESP version (b.19i) — each reader re-`AUTH`s its own
///   connection between its reads, from the unwrapped provider.
pub(crate) struct NoRefreshCredentialProvider(Arc<dyn CredentialProvider>);

impl NoRefreshCredentialProvider {
    /// Wrap `inner`. The `Arc` is shared, not cloned-by-value: the wrapper and
    /// every other connection's provider are the *same* object, so there is no
    /// second credential source to drift.
    pub(crate) fn new(inner: Arc<dyn CredentialProvider>) -> Self {
        Self(inner)
    }
}

#[async_trait::async_trait]
impl CredentialProvider for NoRefreshCredentialProvider {
    /// Verbatim delegation — same inner `Arc`, same `server` argument, same
    /// `Result`. The credential presented at handshake is identical to every
    /// other connection's.
    async fn fetch(
        &self,
        server: Option<&Server>,
    ) -> Result<(Option<String>, Option<String>), FredError> {
        self.0.fetch(server).await
    }

    /// `None` — fred's `ClientInner::reset_credential_refresh_task` only spawns
    /// the refresh task when this is `Some`, so returning `None` means the task is
    /// never created at all (not created-and-erroring). This is the whole point
    /// of the wrapper.
    fn refresh_interval(&self) -> Option<Duration> {
        None
    }
}

/// Forwards to the inner provider's `Debug`, inheriting its redaction rather
/// than re-implementing it. Prefixed so a log line still identifies the wrapper.
impl fmt::Debug for NoRefreshCredentialProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NoRefreshCredentialProvider")
            .field(&self.0)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the delegating wrapper itself (b.j64). The *wiring* tests
    //! — that `build_fred_config` installs this on the pub/sub config under
    //! RESP2 only and on the stream readers' config under any RESP version
    //! (b.19i), never on the shared one, with non-auth fields at parity — live
    //! in `storage::client`'s test module next to the seam they guard.

    use std::sync::atomic::{AtomicUsize, Ordering};

    use fred::error::ErrorKind as FredErrorKind;

    use super::*;

    /// Distinctive needle the `Debug`-forwarding test looks for.
    const INNER_DEBUG_NEEDLE: &str = "StubInnerProviderDebugNeedle";

    /// A stub inner provider that records how many times `fetch` was called,
    /// returns a fixed credential pair (or a fixed error), and reports a
    /// non-`None` `refresh_interval` so the wrapper's `None` is provably the
    /// wrapper's doing and not inherited.
    struct StubProvider {
        fetches: AtomicUsize,
        /// Last `server` argument seen, rendered — proves verbatim pass-through.
        last_server: std::sync::Mutex<Option<String>>,
        fail: bool,
    }

    impl StubProvider {
        fn new(fail: bool) -> Self {
            Self {
                fetches: AtomicUsize::new(0),
                last_server: std::sync::Mutex::new(None),
                fail,
            }
        }
    }

    #[async_trait::async_trait]
    impl CredentialProvider for StubProvider {
        async fn fetch(
            &self,
            server: Option<&Server>,
        ) -> Result<(Option<String>, Option<String>), FredError> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            *self.last_server.lock().unwrap() = server.map(|s| format!("{s}"));
            if self.fail {
                return Err(FredError::new(
                    FredErrorKind::Auth,
                    "stub credential failure",
                ));
            }
            Ok((
                Some("stub-user".to_string()),
                Some("stub-token".to_string()),
            ))
        }

        fn refresh_interval(&self) -> Option<Duration> {
            Some(Duration::from_secs(300))
        }
    }

    impl fmt::Debug for StubProvider {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(INNER_DEBUG_NEEDLE)
        }
    }

    #[tokio::test]
    async fn fetch_reaches_the_inner_provider_on_every_call_and_returns_it_unchanged() {
        let inner = Arc::new(StubProvider::new(false));
        let wrapper = NoRefreshCredentialProvider::new(inner.clone());
        let expected = (
            Some("stub-user".to_string()),
            Some("stub-token".to_string()),
        );

        let first = wrapper.fetch(None).await.expect("first delegated fetch");
        let second = wrapper.fetch(None).await.expect("second delegated fetch");

        assert_eq!(
            first, expected,
            "the wrapper must return the inner provider's credential unchanged — it \
             narrows the re-AUTH CADENCE, never the credential"
        );
        assert_eq!(
            second, expected,
            "…and must keep returning it unchanged on every subsequent call"
        );
        // WHY TWO CALLS. A single call cannot tell verbatim delegation from a
        // wrapper-local cache that memoized the first result, and no-cache is
        // THE load-bearing property here: with the periodic refresh suppressed,
        // the reconnect handshake's fresh `fetch()` is the pub/sub connection's
        // ONLY credential-renewal path. A caching wrapper would hand that
        // handshake the boot-time token forever — silently, until the process
        // restarts — while passing every value-equality assertion above.
        assert_eq!(
            inner.fetches.load(Ordering::SeqCst),
            2,
            "each fetch must reach the inner provider — two calls, two inner \
             fetches. A lower count means the wrapper served a cached credential, \
             which on the pub/sub connection means presenting the boot-time token \
             until the process restarts (the reconnect handshake's fetch is the \
             only renewal path left once the periodic re-AUTH is suppressed)"
        );
    }

    #[tokio::test]
    async fn fetch_with_a_server_argument_forwards_it_verbatim() {
        // The `server` argument is per-node context fred passes on cluster
        // handshakes. A wrapper that dropped it (passing `None`) would silently
        // change behaviour for any future provider that signs per-node.
        let inner = Arc::new(StubProvider::new(false));
        let wrapper = NoRefreshCredentialProvider::new(inner.clone());
        let server = Server::new("node-7.example.com", 6379);

        let _ = wrapper.fetch(Some(&server)).await.expect("delegated fetch");

        assert_eq!(
            inner.last_server.lock().unwrap().as_deref(),
            Some(format!("{server}").as_str()),
            "the wrapper must forward the `server` argument verbatim"
        );
    }

    #[tokio::test]
    async fn fetch_with_a_failing_inner_provider_propagates_the_error_kind() {
        // A token-unavailable `Err(Auth)` must still surface as `Err(Auth)` so
        // the existing PING readiness posture (SDD §15.4 / §15.5) is unchanged
        // on the pub/sub connection.
        let failing = Arc::new(StubProvider::new(true));
        let wrapper = NoRefreshCredentialProvider::new(failing);

        let err = wrapper
            .fetch(None)
            .await
            .expect_err("a failing inner provider must surface as an error");

        assert!(
            matches!(err.kind(), FredErrorKind::Auth),
            "the wrapper must not reclassify the inner error kind, got {:?}",
            err.kind()
        );
    }

    #[test]
    fn refresh_interval_is_none_even_when_the_inner_provider_schedules_one() {
        let inner = Arc::new(StubProvider::new(false));
        assert!(
            inner.refresh_interval().is_some(),
            "precondition: the inner provider must schedule a refresh, otherwise \
             this test cannot tell the wrapper's `None` from an inherited one"
        );

        let wrapper = NoRefreshCredentialProvider::new(inner);
        assert!(
            wrapper.refresh_interval().is_none(),
            "the wrapper MUST return None — fred's reset_credential_refresh_task \
             only spawns the per-client refresh task when this is Some, and on \
             the RESP2 pub/sub connection that task's AUTH is structurally refused"
        );
    }

    #[test]
    fn debug_forwards_to_the_inner_provider() {
        // Redaction is INHERITED, not re-implemented: a future inner provider
        // that redacts must redact here too. Forwarding is the mechanism.
        let inner = Arc::new(StubProvider::new(false));
        let rendered = format!("{:?}", NoRefreshCredentialProvider::new(inner));
        assert!(
            rendered.contains(INNER_DEBUG_NEEDLE),
            "Debug must forward to the inner provider so its redaction is \
             inherited; got: {rendered}"
        );
        assert!(
            rendered.contains("NoRefreshCredentialProvider"),
            "…while still naming the wrapper so a log line is unambiguous; got: {rendered}"
        );
    }
}
