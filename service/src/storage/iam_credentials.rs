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

//! AWS IAM credential provider for Valkey AUTH (Epic 4 `t3.tr5.we.6w`, SDD §15.5).
//!
//! The AWS analog of
//! [`TokenFileCredentialProvider`](super::token_file_credentials::TokenFileCredentialProvider).
//! Bridges an AWS credential source (the standard provider chain — IRSA in
//! production) into fred's [`CredentialProvider`] trait so every `AUTH` /
//! `HELLO` fred issues carries a freshly minted **SigV4 pre-signed token** as
//! the password and the IAM-enabled ACL username as the username.
//!
//! ## SigV4 pre-signed token
//!
//! Amazon MemoryDB and ElastiCache authenticate IAM clients with a short-lived
//! token that is an AWS SigV4 **pre-signed request URI** for a synthetic
//! `Action=connect&User=<acl-user>` GET against the cluster host, with the
//! scheme (`https://`) stripped off. The token is valid for 15 minutes
//! (`900 s`, the `X-Amz-Expires` presign bound). We regenerate it on every
//! [`CredentialProvider::fetch`] rather than caching: HMAC-SHA256 signing is
//! cheap, and a fresh token per call removes any staleness window.
//!
//! The presign core is a **pure function** of
//! `(credentials, region, cluster_host, service, timestamp)` — see
//! [`IamCredentialProvider::presign_at`], whose signing timestamp is injected
//! (never read from the wall clock inside the signing math) so the token is
//! deterministic under test. [`CredentialProvider::fetch`] is the only caller
//! that reads the real clock.
//!
//! ## Service selector and signing host
//!
//! [`ValkeyIamService`] selects the SigV4 *service name*: `memorydb` for
//! MemoryDB, `elasticache` for ElastiCache. The signing **host** is taken from
//! `valkey_iam_cluster_host` — used verbatim for MemoryDB (the cluster name),
//! and **lowercased** for ElastiCache (ElastiCache lowercases cache names at
//! creation; signing against a mixed-case name fails auth).
//!
//! ## Per-cluster, not per-node
//!
//! The signing host is the cluster/cache name, so one token authenticates every
//! node behind it. fred passes the specific target `server` to `fetch`, but this
//! provider deliberately ignores it and signs once against the configured
//! cluster host (see the comment in [`CredentialProvider::fetch`]).
//!
//! ## Re-AUTH cadence and per-client jitter (mirrors SR-4.2)
//!
//! AWS force-disconnects an IAM-authenticated connection after 12 hours; the
//! periodic re-AUTH with a fresh token is the prolongation mechanism, so
//! [`CredentialProvider::refresh_interval`] MUST return `Some` — `None` would
//! still hit a fleet-wide 12-hour mass disconnect. The cadence is a short
//! `base_interval` (well under the 15-minute token expiry, so re-AUTH always
//! fires with headroom) plus a fresh per-call jitter draw. As with the
//! access-token-file provider, this single instance is shared as one `Arc` across all pools /
//! clients, fred spawns a fresh credential-refresh task (calling
//! `refresh_interval` afresh) per pooled client on each connect/reconnect, and
//! each stream reader calls it afresh every time it schedules its own next
//! re-AUTH — so a fresh per-call random draw yields genuine per-client stagger
//! and avoids a synchronized re-AUTH thundering herd. The realized interval is
//! uniform in `[base_interval, base_interval + jitter_bound]`.
//!
//! The stream readers do not run fred's refresh (b.19i): their configs carry
//! this provider wrapped so fred schedules nothing there, and each reader takes
//! the same tick itself — `fetch(None)`, which here always mints a credential —
//! and sends the `AUTH` on its own connection between its reads, waking an
//! in-flight read first. The 12-hour prolongation therefore reaches the
//! readers' connections on the same cadence.
//!
//! ## Token-unavailable handling
//!
//! When AWS credentials cannot be resolved (STS unreachable, web-identity token
//! missing/expired) or signing fails, `fetch()` returns
//! [`FredErrorKind::Auth`] with a **generic** message — no token or credential
//! bytes — the same discipline the access-token-file provider uses. `Auth` (not a transient
//! kind) keeps classification consistent with
//! [`is_transient_fred_error`](super::client::is_transient_fred_error), which
//! treats auth failures as non-transient. The failed `AUTH` trips the existing
//! Valkey `PING` readiness probe → `NOT_SERVING`; no new `ProbeConfig` slot.
//!
//! ## Bearer-bytes-never-logged invariant (extends SR-6.2 to the IAM surface)
//!
//! This struct is credential-bearing: `fetch()` mints a live token and the
//! username, region, and cluster host are credential-adjacent. The hand-written
//! [`Debug`] impl prints none of them — only the non-secret scheduling fields
//! and the public service selector. Grep this module before adding any logging
//! or `Debug` field.

use std::fmt;
use std::time::{Duration, SystemTime};

use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    sign, SignableBody, SignableRequest, SignatureLocation, SigningSettings,
};
use aws_sigv4::sign::v4;
use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
use fred::types::config::{CredentialProvider, Server};
use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use rand::Rng;

use crate::config::ValkeyIamService;

/// Presigned-token validity, in seconds — the `X-Amz-Expires` bound. AWS caps
/// this at 15 minutes for the ElastiCache / MemoryDB connect action.
const TOKEN_VALIDITY_SECS: u64 = 900;

/// AWS SigV4 query-string percent-encoding set: URI-encode every byte **except**
/// the RFC 3986 unreserved characters `A-Za-z0-9-_.~`. Matches the encoding AWS
/// applies when canonicalizing (and later verifying) the query string, so the
/// pre-signed values we append decode back to exactly what was signed.
const AWS_QUERY_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

/// fred [`CredentialProvider`] that authenticates Valkey connections with a
/// freshly minted AWS SigV4 pre-signed token (SDD §15.5).
///
/// Construct via [`Self::new`]; wrap in `Arc` and install on fred's
/// `FredConfig::credential_provider` from the storage
/// `apply_fred_config_overrides` seam (Epic 4 wiring Subtask).
pub struct IamCredentialProvider {
    /// Injectable AWS credential source. In production this is the standard AWS
    /// provider chain (IRSA — an STS `AssumeRoleWithWebIdentity` round-trip the
    /// chain resolves and caches); under test it is a stub. Credential-bearing —
    /// never printed by [`Debug`].
    credentials_provider: SharedCredentialsProvider,
    /// IAM-enabled Valkey ACL username (`Config::valkey_iam_username`). Returned
    /// as the AUTH username **and** embedded as the `User` connect-action query
    /// parameter that the token authorizes. Credential-adjacent — never printed
    /// by [`Debug`].
    username: String,
    /// AWS region for SigV4 signing (`Config::valkey_iam_region`).
    /// Credential-adjacent — never printed by [`Debug`].
    region: String,
    /// Cluster/cache host the token is signed against
    /// (`Config::valkey_iam_cluster_host`) — the cluster name for MemoryDB, the
    /// cache name (lowercased at sign time) for ElastiCache. Credential-adjacent
    /// — never printed by [`Debug`].
    cluster_host: String,
    /// AWS managed backend — selects the SigV4 service name (`memorydb` vs
    /// `elasticache`) and whether the signing host is lowercased. A public,
    /// non-secret selector.
    service: ValkeyIamService,
    /// Base re-AUTH cadence. Kept short relative to the 15-minute token expiry so
    /// re-AUTH always fires with headroom (and prolongs the connection ahead of
    /// AWS's 12-hour force-disconnect).
    base_interval: Duration,
    /// Upper bound on the per-client stagger added to `base_interval`. The
    /// realized jitter is uniform in `[0, jitter_bound]`, drawn fresh on each
    /// [`Self::refresh_interval`] call.
    jitter_bound: Duration,
}

impl IamCredentialProvider {
    /// Construct a provider.
    ///
    /// * `credentials_provider` — the AWS credential source (the standard
    ///   provider chain / IRSA in production; a stub under test).
    /// * `username` — the IAM-enabled ACL username
    ///   (`Config::valkey_iam_username`).
    /// * `region` — AWS region for SigV4 signing
    ///   (`Config::valkey_iam_region`).
    /// * `cluster_host` — cluster/cache host to sign against
    ///   (`Config::valkey_iam_cluster_host`).
    /// * `service` — the AWS managed backend selector
    ///   (`Config::valkey_iam_service`).
    /// * `base_interval` — base re-AUTH cadence.
    /// * `jitter_bound` — upper bound of the per-client stagger.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        credentials_provider: SharedCredentialsProvider,
        username: String,
        region: String,
        cluster_host: String,
        service: ValkeyIamService,
        base_interval: Duration,
        jitter_bound: Duration,
    ) -> Self {
        Self {
            credentials_provider,
            username,
            region,
            cluster_host,
            service,
            base_interval,
            jitter_bound,
        }
    }

    /// SigV4 service name for the configured backend.
    fn service_name(&self) -> &'static str {
        match self.service {
            ValkeyIamService::MemoryDb => "memorydb",
            ValkeyIamService::ElastiCache => "elasticache",
        }
    }

    /// Host the token is signed against: the cluster host verbatim for MemoryDB,
    /// lowercased for ElastiCache (ElastiCache lowercases cache names at
    /// creation, so signing against a mixed-case name fails auth).
    fn signing_host(&self) -> String {
        match self.service {
            ValkeyIamService::MemoryDb => self.cluster_host.clone(),
            ValkeyIamService::ElastiCache => self.cluster_host.to_ascii_lowercase(),
        }
    }

    /// Mint a SigV4 pre-signed connect token at a caller-supplied timestamp.
    ///
    /// Pure over `(credentials, self.region, self.cluster_host, self.service,
    /// now)` — the signing timestamp is injected, never read from the wall
    /// clock, so the output is deterministic for a fixed set of inputs. The
    /// returned token is the pre-signed request URI with the `https://` scheme
    /// stripped (the shape MemoryDB / ElastiCache expect as the AUTH password).
    ///
    /// Any signing failure is mapped to a generic [`FredErrorKind::Auth`] error
    /// with no credential or token bytes.
    fn presign_at(&self, credentials: &Credentials, now: SystemTime) -> Result<String, FredError> {
        let signing_host = self.signing_host();

        // Synthetic connect request the token authorizes. The `User` value is
        // AWS-query-encoded so the URI parses and round-trips to the ACL
        // username. `Action=connect` is fixed.
        //
        // v1 deliberately OMITS the ElastiCache serverless-only
        // `ResourceType=ServerlessCache` query parameter: this provider targets
        // cluster-mode / replication-group (MemoryDB clusters, ElastiCache
        // non-serverless), not serverless caches.
        let user_enc = utf8_percent_encode(&self.username, AWS_QUERY_ENCODE_SET).to_string();
        let base_uri = format!("https://{signing_host}/?Action=connect&User={user_enc}");

        // The `From<Credentials>` target (`Identity`) lives in a transitive
        // crate; let inference name it from the builder's `.identity(&_)` arg so
        // we need not add aws-smithy-runtime-api as a direct dependency.
        let identity = credentials.clone().into();

        // `SigningSettings` is `#[non_exhaustive]`, so mutate a default rather
        // than a struct literal.
        let mut settings = SigningSettings::default();
        settings.signature_location = SignatureLocation::QueryParams;
        settings.expires_in = Some(Duration::from_secs(TOKEN_VALIDITY_SECS));

        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(self.service_name())
            .time(now)
            .settings(settings)
            .build()
            .map_err(|_| Self::generic_auth_error())?;
        let params = signing_params.into();

        let signable = SignableRequest::new(
            "GET",
            &base_uri,
            std::iter::empty::<(&str, &str)>(),
            SignableBody::empty(),
        )
        .map_err(|_| Self::generic_auth_error())?;

        let (instructions, _signature) = sign(signable, &params)
            .map_err(|_| Self::generic_auth_error())?
            .into_parts();

        // Query-param signing places every `X-Amz-*` parameter (including the
        // signature) in `params`; append them to the base URI, AWS-encoding each
        // value so the presigned URL AWS later re-canonicalizes matches what was
        // signed. `Action` / `User` are carried verbatim from `base_uri` — they
        // are byte-identical between the signed input and this final URL.
        let (_headers, query_params) = instructions.into_parts();
        let mut presigned = base_uri;
        for (name, value) in query_params {
            presigned.push('&');
            presigned.push_str(name);
            presigned.push('=');
            presigned.push_str(&utf8_percent_encode(&value, AWS_QUERY_ENCODE_SET).to_string());
        }

        // The AUTH password is the presigned URI without its scheme.
        let token = presigned
            .strip_prefix("https://")
            .map(str::to_owned)
            .unwrap_or(presigned);
        Ok(token)
    }

    /// Generic token-unavailable error — no credential or token bytes.
    fn generic_auth_error() -> FredError {
        FredError::new(
            FredErrorKind::Auth,
            "iam valkey credential unavailable: AWS credentials could not be resolved or signed",
        )
    }

    /// Fresh per-client stagger in `[0, jitter_bound]`.
    ///
    /// Draws fresh entropy from the thread RNG on **every** call — NOT a seeded,
    /// reproducible draw. fred spawns a fresh credential-refresh task (and thus
    /// calls `refresh_interval` afresh) per client on each connect/reconnect, so
    /// a fresh per-call draw spreads clients across the jitter band rather than
    /// re-AUTHing in lockstep.
    fn jitter(&self) -> Duration {
        let bound_ms = self.jitter_bound.as_millis();
        if bound_ms == 0 {
            return Duration::ZERO;
        }
        // `as_millis()` is u128; cap at u64 for `random_range`. A jitter bound
        // beyond ~584 million years is not a real configuration, so the
        // saturating cast is purely defensive.
        let bound_ms = u64::try_from(bound_ms).unwrap_or(u64::MAX);
        Duration::from_millis(rand::rng().random_range(0..=bound_ms))
    }
}

#[async_trait::async_trait]
impl CredentialProvider for IamCredentialProvider {
    /// Return `(Some(acl_username), Some(presigned_token))` for the next `AUTH` /
    /// `HELLO`.
    ///
    /// Resolves AWS credentials through the injected provider (an STS
    /// `AssumeRoleWithWebIdentity` round-trip the chain caches under IRSA), then
    /// mints a SigV4 pre-signed token at the real clock. When credentials cannot
    /// be resolved — or signing fails — returns [`FredErrorKind::Auth`] with a
    /// **generic** message (no token bytes) so the connection's AUTH fails rather
    /// than sending a credential AWS is about to reject. `Auth` (not a transient
    /// kind) keeps classification consistent with
    /// [`is_transient_fred_error`](super::client::is_transient_fred_error).
    ///
    /// The `server` argument is intentionally ignored: the token is signed
    /// against the configured cluster/cache host and authenticates every node
    /// behind it (per-cluster, not per-node), so there is no per-target signing
    /// to do.
    async fn fetch(
        &self,
        _server: Option<&Server>,
    ) -> Result<(Option<String>, Option<String>), FredError> {
        let credentials = self
            .credentials_provider
            .provide_credentials()
            .await
            .map_err(|_| Self::generic_auth_error())?;

        let token = self.presign_at(&credentials, SystemTime::now())?;
        Ok((Some(self.username.clone()), Some(token)))
    }

    /// Re-AUTH cadence: `base_interval` plus a fresh per-client stagger drawn on
    /// every call (see [`Self::jitter`]). Always `Some` — the periodic re-AUTH
    /// prolongs the connection ahead of AWS's 12-hour force-disconnect, so
    /// `None` would still mass-disconnect the fleet. Kept short relative to the
    /// 15-minute token expiry so each re-AUTH mints a fresh token with headroom.
    fn refresh_interval(&self) -> Option<Duration> {
        Some(self.base_interval + self.jitter())
    }
}

/// Hand-written redacting [`Debug`] — the trait requires `Debug` and this struct
/// holds credential-bearing / credential-adjacent state (the AWS credential
/// source, the ACL username, region, and cluster host). Prints only the
/// non-secret scheduling fields and the public service selector; the credential
/// source, username, region, cluster host, and any token bytes never appear.
/// Extends the bearer-bytes-never-logged invariant (SR-6.2) to the IAM Valkey
/// surface.
impl fmt::Debug for IamCredentialProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IamCredentialProvider")
            .field("username", &"<redacted>")
            .field("region", &"<redacted>")
            .field("cluster_host", &"<redacted>")
            .field("service", &self.service)
            .field("base_interval", &self.base_interval)
            .field("jitter_bound", &self.jitter_bound)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `IamCredentialProvider` — Subtask `t3.tr5.we.6w.2a`.
    //!
    //! Coverage targets (parent Task `t2.tr5.we.6w` acceptance criteria, SDD
    //! §15.5), mirroring the Epic 3 access-token-file provider's `#[cfg(test)]` discipline
    //! (`CapturingWriter` canary, in-band jitter asserts, redacting `Debug`):
    //!
    //! - Deterministic SigV4 presign for the **MemoryDB** selector: byte-stable
    //!   across repeated calls at a fixed injected timestamp, signs against
    //!   service name `memorydb`, and signs the cluster host **verbatim** (no
    //!   case folding). `fetch()` returns `username = valkey_iam_username`.
    //! - Deterministic SigV4 presign for the **ElastiCache** selector: signs
    //!   against service name `elasticache` and **lowercases** the signing host
    //!   (the one behavioural divergence from the MemoryDB path).
    //! - `refresh_interval()` always in-band `[base, base + jitter_bound]`,
    //!   spread across repeated calls (per-client stagger, no lockstep — the
    //!   12-hour mass-disconnect guard), and short enough that the max interval
    //!   leaves headroom well under the 900 s / 15 min token expiry.
    //! - Credential-unavailable (STS unreachable / web-identity token missing)
    //!   yields `Err(ErrorKind::Auth)` — non-transient, no silent connect, and
    //!   no credential/token bytes in the message.
    //! - `Debug` never prints the username, region, cluster host, or token
    //!   bytes; and no 8-byte slice of any credential canary reaches the logs
    //!   (bearer-bytes-never-logged, SR-6.2 extended to the IAM Valkey surface).
    //!
    //! The signing timestamp is injected through the private
    //! [`IamCredentialProvider::presign_at`] entry point, so the SigV4 asserts
    //! are deterministic (no wall-clock read in the signing math). The jitter
    //! draws fresh thread-RNG entropy and is NOT seedable, so the
    //! `refresh_interval()` tests assert the statistical in-band / spread
    //! invariants rather than a fixed-seed equality.

    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, SystemTime};

    use aws_credential_types::provider::error::CredentialsError;
    use aws_credential_types::provider::{future, ProvideCredentials, SharedCredentialsProvider};
    use aws_credential_types::Credentials;
    use tracing_subscriber::fmt::MakeWriter;

    use super::*;

    // ── Fixed test inputs ───────────────────────────────────────────────────

    /// Distinctive canary access key — never a real AWS key; chosen so any
    /// contiguous 8-byte slice is improbable to collide with tracing/tokio
    /// noise, for the bearer-bytes-never-logged assertion.
    const ACCESS_KEY: &str = "AKIAcanaryACCESSkey0000000";
    /// Distinctive canary secret access key.
    const SECRET_KEY: &str = "secretCanaryKEYwJalrXUtnFEMIcanary00";
    /// Distinctive canary session token (IRSA/STS creds carry one).
    const SESSION_TOKEN: &str = "sessionCanaryTOKENfedcba9876543210canary";
    /// Fixed AWS region for SigV4 signing.
    const REGION: &str = "us-east-1";
    /// IAM-enabled ACL username = `valkey_iam_username`. A distinctive needle so
    /// the redaction test can search for it, and long enough for the 8-byte
    /// slice search to be meaningful.
    const USERNAME: &str = "iam-acl-canary-user";
    /// Mixed-case MemoryDB cluster host — signed **verbatim** (case preserved).
    /// The test hosts keep an endpoint's shape under example.com: the signer
    /// only copies or lowercases the host, never parses it.
    const MEMORYDB_HOST: &str = "MyCluster-AbC123.abcdef.clustercfg.memorydb.us-east-1.example.com";
    /// Mixed-case ElastiCache cache host — signed **lowercased**. Contains
    /// neither `memorydb` nor `elasticache` as a substring, so the service-name
    /// assertion cannot be satisfied spuriously by the host.
    const ELASTICACHE_HOST: &str = "MyCache-AbC123.abcdef.clustercfg.use1.cache.example.com";
    /// Fixed injected signing timestamp (deterministic — never the wall clock).
    const FIXED_TIME_SECS: u64 = 1_700_000_000;

    fn fixed_time() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(FIXED_TIME_SECS)
    }

    /// Fixed static credentials (access key / secret / session token), used both
    /// as the injected `presign_at` input and — wrapped in a
    /// `SharedCredentialsProvider` — as the `fetch()` credential source.
    fn fixed_credentials() -> Credentials {
        Credentials::new(
            ACCESS_KEY,
            SECRET_KEY,
            Some(SESSION_TOKEN.to_string()),
            None,
            "iam-credentials-test",
        )
    }

    /// Build a provider backed by static `fixed_credentials()` for the given
    /// backend selector and cluster host. Fixed `base_interval` (30 s) /
    /// `jitter_bound` (10 s); `refresh_interval()` draws fresh entropy each call.
    fn build_provider(service: ValkeyIamService, cluster_host: &str) -> IamCredentialProvider {
        IamCredentialProvider::new(
            SharedCredentialsProvider::new(fixed_credentials()),
            USERNAME.to_string(),
            REGION.to_string(),
            cluster_host.to_string(),
            service,
            Duration::from_secs(30),
            Duration::from_secs(10),
        )
    }

    // ── Deterministic SigV4 presign — MemoryDB selector ─────────────────────

    #[test]
    fn presign_memorydb_is_deterministic_and_signs_host_verbatim() {
        let provider = build_provider(ValkeyIamService::MemoryDb, MEMORYDB_HOST);
        let creds = fixed_credentials();

        // Byte-stable across repeated calls at the same injected timestamp
        // (pure function of the inputs — no wall-clock read).
        let token_a = provider
            .presign_at(&creds, fixed_time())
            .expect("presign a");
        let token_b = provider
            .presign_at(&creds, fixed_time())
            .expect("presign b");
        assert_eq!(
            token_a, token_b,
            "presign_at must be byte-stable for fixed (credentials, region, host, service, time)"
        );

        // Signs against SigV4 service name `memorydb` — the credential scope
        // ends `…/memorydb/aws4_request`, AWS-query-encoded (slashes → %2F).
        assert!(
            token_a.contains("memorydb%2Faws4_request"),
            "MemoryDB token must sign against service name `memorydb`, got: {token_a}"
        );
        assert!(
            !token_a.contains("elasticache"),
            "MemoryDB token must not name the elasticache service, got: {token_a}"
        );

        // The signing host is used VERBATIM (mixed case preserved) — the token
        // is the presigned URI with `https://` stripped, so it starts with the
        // authority. MemoryDB does NOT lowercase.
        assert!(
            token_a.starts_with(&format!("{MEMORYDB_HOST}/?Action=connect")),
            "MemoryDB must sign the cluster host verbatim (case preserved), got: {token_a}"
        );
        assert!(
            token_a.contains("MyCluster-AbC123"),
            "MemoryDB host mixed-case must survive verbatim, got: {token_a}"
        );
    }

    #[tokio::test]
    async fn fetch_returns_valkey_iam_username() {
        let provider = build_provider(ValkeyIamService::MemoryDb, MEMORYDB_HOST);
        let (username, password) = provider
            .fetch(None)
            .await
            .expect("fetch must succeed with resolvable static credentials");
        assert_eq!(
            username,
            Some(USERNAME.to_string()),
            "fetch() username must equal valkey_iam_username"
        );
        let token = password.expect("fetch must return a presigned token password");
        assert!(
            token.starts_with(&format!("{MEMORYDB_HOST}/?Action=connect")),
            "fetch() password must be the presigned connect URI (scheme-stripped), got: {token}"
        );
    }

    #[tokio::test]
    async fn fetch_ignores_per_node_server_and_signs_configured_cluster_host() {
        // Per-cluster, not per-node: fred passes the specific target node to
        // `fetch`, but the token always signs against the configured
        // `cluster_host` so one token authenticates every node. A regression that
        // started signing against the passed-in `server` would slip past every
        // other test (they all call `fetch(None)`), so pass a DIFFERENT node host
        // and assert the token still signs the configured cluster host verbatim.
        let provider = build_provider(ValkeyIamService::MemoryDb, MEMORYDB_HOST);
        let other_node = Server::new("some-other-node-host", 6379);

        let (_username, password) = provider
            .fetch(Some(&other_node))
            .await
            .expect("fetch must succeed with resolvable static credentials");
        let token = password.expect("fetch must return a presigned token password");

        // The token authority is the configured cluster host verbatim — the
        // per-node `server` arg does NOT influence host selection.
        assert!(
            token.starts_with(&format!("{MEMORYDB_HOST}/?Action=connect")),
            "fetch() must sign the configured cluster_host verbatim, ignoring the per-node \
             server arg, got: {token}"
        );
        assert!(
            !token.contains("some-other-node-host"),
            "the passed-in per-node server host must NEVER appear in the signed token, got: {token}"
        );
    }

    // ── Deterministic SigV4 presign — ElastiCache selector ──────────────────

    #[test]
    fn presign_elasticache_is_deterministic_and_lowercases_host() {
        let provider = build_provider(ValkeyIamService::ElastiCache, ELASTICACHE_HOST);
        let creds = fixed_credentials();

        let token_a = provider
            .presign_at(&creds, fixed_time())
            .expect("presign a");
        let token_b = provider
            .presign_at(&creds, fixed_time())
            .expect("presign b");
        assert_eq!(
            token_a, token_b,
            "presign_at must be byte-stable for fixed inputs (ElastiCache selector)"
        );

        // Signs against SigV4 service name `elasticache`.
        assert!(
            token_a.contains("elasticache%2Faws4_request"),
            "ElastiCache token must sign against service name `elasticache`, got: {token_a}"
        );
        assert!(
            !token_a.contains("memorydb"),
            "ElastiCache token must not name the memorydb service, got: {token_a}"
        );

        // The one behavioural divergence: the signing host is LOWERCASED
        // (ElastiCache lowercases cache names at creation; signing a mixed-case
        // name fails auth). The token must start with the all-lowercase host and
        // never carry the original mixed-case form.
        let lower_host = ELASTICACHE_HOST.to_ascii_lowercase();
        assert!(
            token_a.starts_with(&format!("{lower_host}/?Action=connect")),
            "ElastiCache must sign the host lowercased, got: {token_a}"
        );
        assert!(
            !token_a.contains("MyCache-AbC123"),
            "ElastiCache must not sign the mixed-case host verbatim, got: {token_a}"
        );
    }

    /// The memorydb-vs-elasticache divergence pinned in one place: identical
    /// fixed inputs, only the selector differs, and the two tokens must differ
    /// in exactly the two documented axes — service signing name and host case.
    #[test]
    fn memorydb_and_elasticache_diverge_only_on_service_name_and_host_case() {
        let creds = fixed_credentials();
        // Same mixed-case host fed to both selectors so the ONLY inputs that
        // vary are the service selector's two effects.
        let host = "MixedCase-Host123.abc.clustercfg.example.us-east-1.example.com";

        let memorydb = build_provider(ValkeyIamService::MemoryDb, host)
            .presign_at(&creds, fixed_time())
            .expect("memorydb presign");
        let elasticache = build_provider(ValkeyIamService::ElastiCache, host)
            .presign_at(&creds, fixed_time())
            .expect("elasticache presign");

        assert_ne!(
            memorydb, elasticache,
            "the two selectors must produce distinct tokens for the same inputs"
        );
        // MemoryDB: verbatim host + memorydb scope.
        assert!(memorydb.starts_with("MixedCase-Host123"));
        assert!(memorydb.contains("memorydb%2Faws4_request"));
        // ElastiCache: lowercased host + elasticache scope.
        assert!(elasticache.starts_with("mixedcase-host123"));
        assert!(elasticache.contains("elasticache%2Faws4_request"));
    }

    // ── refresh_interval(): every draw stays within bounds ──────────────────

    #[test]
    fn refresh_interval_is_always_within_bounds() {
        let provider = build_provider(ValkeyIamService::MemoryDb, MEMORYDB_HOST);
        // base_interval = 30 s, jitter_bound = 10 s. Each call draws fresh
        // entropy (no seed), so assert the in-band invariant for EVERY draw.
        let base = Duration::from_secs(30);
        let ceiling = Duration::from_secs(40); // base + jitter_bound
        for _ in 0..200 {
            let interval = provider
                .refresh_interval()
                .expect("refresh_interval must be Some — None mass-disconnects the fleet");
            assert!(
                interval >= base && interval <= ceiling,
                "every drawn interval must stay within [{base:?}, {ceiling:?}], got {interval:?}"
            );
        }
    }

    // ── refresh_interval(): zero jitter bound → exactly base_interval ───────

    #[test]
    fn refresh_interval_with_zero_jitter_bound_returns_base_interval_exactly() {
        // The `jitter()` guard short-circuits to `Duration::ZERO` when the bound
        // is zero (a config with no per-client stagger), so `refresh_interval()`
        // must return EXACTLY `base_interval` on every draw — no RNG range over
        // an empty band.
        let base = Duration::from_secs(30);
        let provider = IamCredentialProvider::new(
            SharedCredentialsProvider::new(fixed_credentials()),
            USERNAME.to_string(),
            REGION.to_string(),
            MEMORYDB_HOST.to_string(),
            ValkeyIamService::MemoryDb,
            base,
            Duration::ZERO,
        );
        for _ in 0..50 {
            let interval = provider
                .refresh_interval()
                .expect("refresh_interval must be Some");
            assert_eq!(
                interval, base,
                "with a zero jitter bound every draw must equal base_interval exactly, got \
                 {interval:?}"
            );
        }
    }

    // ── refresh_interval(): per-client stagger — no lockstep ────────────────

    #[test]
    fn refresh_interval_spreads_across_repeated_calls_no_lockstep() {
        // Fresh per-call entropy is the mechanism that staggers the fleet's
        // re-AUTHs and avoids a synchronized 12-hour mass-disconnect. fred reads
        // a client's refresh_interval() once per client (all clients share this
        // provider via `Arc`), so repeated calls on one provider must spread
        // rather than lock in step. A jitter-free interval would re-AUTH every
        // client together.
        let provider = build_provider(ValkeyIamService::MemoryDb, MEMORYDB_HOST);
        let base = Duration::from_secs(30);
        let ceiling = Duration::from_secs(40);

        let mut samples = std::collections::HashSet::new();
        for _ in 0..200 {
            let interval = provider
                .refresh_interval()
                .expect("refresh_interval is Some");
            assert!(
                interval >= base && interval <= ceiling,
                "staggered interval {interval:?} must stay within [{base:?}, {ceiling:?}]"
            );
            samples.insert(interval);
        }

        // With a 10 s (10_000 ms) bound over 200 uniform draws the probability
        // of every sample being identical is ~(1/10_001)^199 — asserting "more
        // than one distinct value" is a safe proxy for genuine per-call jitter.
        assert!(
            samples.len() > 1,
            "repeated refresh_interval() calls must spread (per-call jitter, no lockstep); \
             saw only {} distinct value(s)",
            samples.len()
        );
    }

    // ── refresh_interval(): headroom well under the 900 s token expiry ──────

    #[test]
    fn refresh_interval_ceiling_leaves_headroom_before_token_expiry() {
        // The largest re-AUTH interval is `base_interval + jitter_bound`. For a
        // re-AUTH to always mint a fresh token BEFORE the current one's 900 s /
        // 15 min expiry, that ceiling must be short relative to the token
        // validity. We read the provider's ACTUAL configured band (private
        // fields, same-module access) rather than re-declared literals.
        let provider = build_provider(ValkeyIamService::MemoryDb, MEMORYDB_HOST);
        let ceiling = provider.base_interval + provider.jitter_bound;

        // Token validity — the `X-Amz-Expires` bound the provider itself signs.
        let expiry = Duration::from_secs(TOKEN_VALIDITY_SECS);

        let headroom = expiry
            .checked_sub(ceiling)
            .expect("max re-AUTH interval must be shorter than the 900 s token expiry");
        assert!(
            headroom >= Duration::from_secs(600),
            "base + max_jitter ({ceiling:?}) must leave generous headroom (≥ 600 s) before the \
             900 s token expiry so re-AUTH always refreshes first; headroom was {headroom:?}"
        );
        // Belt-and-suspenders: at least two re-AUTH cycles fit inside one token
        // lifetime, so a missed tick still refreshes before expiry.
        assert!(
            ceiling * 2 <= expiry,
            "two full re-AUTH cycles ({:?}) must fit within the token expiry ({expiry:?})",
            ceiling * 2
        );
    }

    // ── Credential-unavailable → Err(Auth), no silent connect, no bytes ─────

    /// Stub credential source that always fails to resolve — models STS being
    /// unreachable or the web-identity token file being missing/expired.
    #[derive(Debug)]
    struct UnavailableCredentials;

    impl ProvideCredentials for UnavailableCredentials {
        fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            future::ProvideCredentials::new(async {
                // The source string carries a distinctive needle; the provider
                // MUST strip it from the surfaced error (generic message only).
                Err(CredentialsError::not_loaded(
                    "web-identity-token-file-missing: STS AssumeRoleWithWebIdentity unreachable",
                ))
            })
        }
    }

    #[tokio::test]
    async fn fetch_errors_auth_when_credentials_unresolvable() {
        let provider = IamCredentialProvider::new(
            SharedCredentialsProvider::new(UnavailableCredentials),
            USERNAME.to_string(),
            REGION.to_string(),
            MEMORYDB_HOST.to_string(),
            ValkeyIamService::MemoryDb,
            Duration::from_secs(30),
            Duration::from_secs(10),
        );

        let err = provider.fetch(None).await.expect_err(
            "fetch MUST fail AUTH when credentials are unresolvable, not send a blank credential",
        );
        // ErrorKind::Auth (non-transient) so fred fails the connection AUTH →
        // the Valkey PING readiness probe trips → NOT_SERVING. `is_transient_*`
        // treats Auth as non-transient (no needless retry of a bad credential).
        assert!(
            matches!(err.kind(), FredErrorKind::Auth),
            "credential-unavailable must surface as ErrorKind::Auth, got {:?}",
            err.kind()
        );

        // The generic message must NOT leak the underlying source detail (which
        // could carry token-file paths / credential context).
        let rendered = format!("{err}");
        assert!(
            !rendered.contains("web-identity-token-file-missing"),
            "error message must not leak the credential-source detail, got: {rendered}"
        );
        assert!(
            !rendered.contains("AssumeRoleWithWebIdentity"),
            "error message must not leak the credential-source detail, got: {rendered}"
        );
    }

    // ── Debug redaction ─────────────────────────────────────────────────────

    #[test]
    fn debug_redacts_username_region_host_and_prints_no_credentials() {
        let provider = build_provider(ValkeyIamService::MemoryDb, MEMORYDB_HOST);
        let rendered = format!("{provider:?}");
        assert!(
            !rendered.contains(USERNAME),
            "Debug must not print the ACL username, got: {rendered}"
        );
        assert!(
            !rendered.contains(REGION),
            "Debug must not print the region, got: {rendered}"
        );
        assert!(
            !rendered.contains("MyCluster-AbC123") && !rendered.contains(MEMORYDB_HOST),
            "Debug must not print the cluster host, got: {rendered}"
        );
        assert!(
            !rendered.contains(ACCESS_KEY) && !rendered.contains(SECRET_KEY),
            "Debug must not print any credential bytes, got: {rendered}"
        );
        // The non-secret scheduling fields + public selector are fine to print.
        assert!(
            rendered.contains("IamCredentialProvider") && rendered.contains("<redacted>"),
            "Debug should still identify the type and mark redacted fields, got: {rendered}"
        );
    }

    // ── No credential bytes in logs (CapturingWriter canary) ────────────────

    /// A `MakeWriter` that captures every byte written into a shared buffer.
    /// Local copy of the access-token-file provider's `CapturingWriter` (that one lives in a
    /// private `#[cfg(test)]` module and is not reachable cross-module).
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

    impl Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn no_credential_bytes_in_logs_across_fetch_and_refresh() {
        // Capture at INFO — this provider's own fetch/refresh/presign paths emit
        // NO tracing events, so any credential leak would come from our code
        // (future log additions), not the aws-sigv4 dependency's TRACE-level
        // internals (which Debug-print the access key by design and are out of
        // scope for OUR bearer-bytes-never-logged invariant). A WARN sentinel
        // proves the capture pipeline is live so the leak assertion is not
        // vacuous.
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::INFO)
            .finish();
        let _dispatch_guard = tracing::subscriber::set_default(subscriber);

        // Success path: mints a real token from the canary credentials.
        let ok_provider = build_provider(ValkeyIamService::ElastiCache, ELASTICACHE_HOST);
        let (_, password) = ok_provider.fetch(None).await.expect("fetch success");
        assert!(password.is_some(), "success path returns a token");
        let _ = ok_provider.refresh_interval();
        let _ = ok_provider.presign_at(&fixed_credentials(), fixed_time());

        // Error path: credential-unavailable.
        let err_provider = IamCredentialProvider::new(
            SharedCredentialsProvider::new(UnavailableCredentials),
            USERNAME.to_string(),
            REGION.to_string(),
            MEMORYDB_HOST.to_string(),
            ValkeyIamService::MemoryDb,
            Duration::from_secs(30),
            Duration::from_secs(10),
        );
        let _ = err_provider.fetch(None).await;

        // Sentinel — proves the subscriber is capturing (guards against a
        // vacuous pass on an empty buffer).
        tracing::warn!(target: "iam_credentials_test", "iam-log-pipeline-live-sentinel");

        drop(_dispatch_guard);

        let guard = captured.lock().unwrap();
        let output = String::from_utf8_lossy(&guard);
        assert!(
            output.contains("iam-log-pipeline-live-sentinel"),
            "capture pipeline must be live (sentinel absent means a vacuous test), got: {output}"
        );

        // Bearer-bytes-never-logged — pinned at 8-byte slice granularity. Slide
        // an 8-byte window over each credential canary and assert NO contiguous
        // 8-byte slice appears in captured output.
        for canary in [ACCESS_KEY, SECRET_KEY, SESSION_TOKEN] {
            let bytes = canary.as_bytes();
            assert!(
                bytes.len() >= 8,
                "canary must be ≥ 8 bytes for the slice search to be meaningful"
            );
            for window in bytes.windows(8) {
                let slice = std::str::from_utf8(window).expect("ASCII canary");
                assert!(
                    !output.contains(slice),
                    "captured log output must not contain any 8-byte slice of a credential canary \
                     (slice={slice:?}, canary={canary:?}, output={output})"
                );
            }
        }
    }
}
