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

//! Access-token-file credential provider for Valkey AUTH (Epic 3 `t3.tr5.jv`, SDD SR-4.1 / SR-4.2).
//!
//! Bridges the Redis-scoped [`ServiceTokenProvider`] into fred's
//! [`CredentialProvider`] trait so every `AUTH` / `HELLO` fred issues carries a
//! live file-sourced access token as the password and the configured identity as
//! the username (SDD SR-4.1). The token is conventionally minted by an Azure
//! Entra managed-identity sidecar — in which case the username is the managed
//! identity's **Object ID**, NOT the client ID — but this provider is agnostic
//! to the token issuer and only reads the file the sidecar writes.
//!
//! ## Why a second `ServiceTokenProvider`
//!
//! The outbound storage-api adapter (Epic 5) already runs one
//! [`ServiceTokenProvider`] with a 60 s refresh lead. The Valkey access-token
//! path wants a *wider* proactive re-read lead — Azure's guidance is to fetch a
//! fresh token **≥ 3 min before expiry** — so the wiring layer constructs a
//! second, Redis-scoped instance via
//! [`ServiceTokenProvider::new_with_refresh_lead`] and hands it to this
//! provider. This module reads that instance through
//! [`ServiceTokenProvider::cached_bearer_if_unexpired`]; it never touches the
//! storage-api instance.
//!
//! ## The lead is a re-read cadence, not a withholding window (b.d6b)
//!
//! `fetch()` reads
//! [`ServiceTokenProvider::cached_bearer_if_unexpired`], which serves the
//! cached token for as long as it is actually unexpired — which is all the
//! refresh-cadence guidance ever asked for. It has to: fred needs a password
//! for every `AUTH` / `HELLO`, so an accessor that treated the lead as a
//! *withholding* window — returning `None` for the whole ≥ 3 min run-up to
//! expiry — turned it into a daily ≥ 3 min window in which **every** Valkey
//! connection attempt failed `AUTH` and the readiness probe tripped, while a
//! working token sat in the cache. That was this path before b.d6b. The
//! withholding accessor has since lost its other consumers too (b.e49) and been
//! deleted outright (b.mq8); serving the cached token is now the provider's
//! only bearer policy.
//!
//! Serving it is never worse than withholding it:
//! failure-mode symmetry — if the backing store really would reject a
//! near-expiry token, sending it fails `AUTH` exactly as withholding it did.
//! The win is at connection *establishment*: pre-change a `fetch()` `Err`
//! failed `Connection::setup` and fred's reconnect policy retried into the void
//! for the width of the lead; post-change the connect succeeds, and the
//! connection is simply dropped at the token's real expiry and re-established
//! with the by-then-cached fresh one.
//!
//! ## Thundering-herd stagger (SR-4.2)
//!
//! fred calls [`CredentialProvider::refresh_interval`] once per `Client` to
//! decide how often to re-AUTH that connection, and adds **no jitter of its
//! own** — a fixed `Duration` re-AUTHs every replica's every connection in
//! lockstep. At Helm scale (up to ~1,176 clients across 3 replicas once every
//! stream reader has connected; every pooled cluster client broadcasts
//! `_AuthAllCluster`) that is precisely the
//! synchronized-refresh thundering herd Azure warns against. The provider is installed once on the shared
//! `FredConfig` and cloned as a single `Arc` across all pools / clients, so a
//! per-instance stagger would give every client in a replica the *same*
//! interval — no per-client spread. Instead [`Self::refresh_interval`] draws
//! **fresh entropy on every call**: fred spawns a fresh credential-refresh task
//! (and thus calls `refresh_interval` afresh) per pooled client on each
//! connect/reconnect, and each stream reader calls it afresh every time it
//! schedules its own next re-AUTH tick (fred schedules none on a reader
//! client — see `storage::stream_mux`'s reader), so a fresh per-call random
//! draw yields genuine per-client stagger (SR-4.2). The realized interval is
//! uniform in `[base_interval, base_interval + jitter_bound]`.
//!
//! ## Rotation-gated re-AUTH (b.7td)
//!
//! fred re-`AUTH`s **every** pooled connection on [`Self::refresh_interval`],
//! and on a cluster each pooled client's re-`AUTH` fans out to every primary
//! (`CommandKind::_AuthAllCluster`). The stream readers do not run fred's
//! refresh: their configs carry this provider wrapped so fred schedules
//! nothing, and each reader takes the same tick itself, between its reads,
//! re-`AUTH`ing only its one connection (b.19i). At Helm scale the pooled
//! refresh is a large, permanent
//! broadcast rate whose only purpose is to carry a credential that changes
//! roughly once a day — and it is not free. Two costs were observed on a
//! cluster-mode managed backing store:
//!
//! - **Head-of-line blocking with no bound of its own.** The cluster `AUTH`
//!   path waits on a reply from every primary with no per-command deadline of
//!   its own, so one slow or wedged node stalls the whole refresh. Those stalls
//!   are the `Failed to refresh credentials` warnings in the pod logs.
//! - **A genuine `WATCH` → `EXEC` hazard.** A refresh that triggers a
//!   reconnect wipes the connection's watch set; an `EXEC` that follows then
//!   commits unwatched.
//!
//! The gate removes the broadcasts that carry nothing new. `fetch()` is called
//! with `Some(server)` at connection setup (the handshake, which *must* always
//! produce a credential) and with `None` on the periodic tick — and fred's
//! refresh loop sends **nothing at all** when the provider returns
//! `password = None` (fred-10.1.0 `src/modules/inner.rs`,
//! `spawn_credential_refresh`: the `if let Some(password) = password` guarding
//! the `hello` / `auth` call is the **last statement in the loop body**, so a
//! `None` falls through it to the next `wait_with_interrupt` with nothing
//! written to the socket. There is no `else` arm; the two `continue`s in that
//! loop belong to the earlier fetch-error and not-`Connected` checks, not to
//! this guard). A stream reader's own tick follows the same rule: a `None`
//! password sends nothing and wakes nothing. So the periodic path can decide,
//! per tick, whether a re-`AUTH` is worth issuing:
//!
//! - The provider fingerprints the cached token in memory and records the
//!   instant the fingerprint last *changed* — the rotation epoch.
//! - The first token the provider ever sees — on a handshake or a tick — is
//!   recorded with **no window open**, and that tick sends nothing (b.19i).
//!   Every connection that exists at that moment authenticated with that same
//!   token at its handshake, so there is nothing to fan out; treating the first
//!   sighting as a rotation re-`AUTH`ed every pooled connection 60–90 s after
//!   every startup.
//! - A call whose fingerprint differs from a prior observation is the first
//!   observation of a rotation: it re-records, opens the window, and (on a
//!   tick) sends.
//! - A tick within the provider's `auth_fanout_window` of that change sends
//!   too. The provider is a single shared `Arc` with no per-client identity, so
//!   it cannot know *which* clients have already re-`AUTH`ed; holding the
//!   window open long enough for every client to take at least one tick inside
//!   it is what gets the new credential onto all of them.
//! - Every other tick returns `(Some(username), None)` and nothing is sent.
//!
//! Rotation pickup **improves** rather than degrading, because the cadence is
//! shortened at the same time (see [`Self::refresh_interval`]): the pre-gate
//! 300–360 s cadence would usually miss the window in which the replacement
//! token is cached but the old one has not yet expired, so connections were
//! typically dropped at expiry and re-established. A 60–90 s cadence lands
//! inside that window far more often. The net effect is fewer broadcasts *and*
//! more successful prolongations.
//!
//! The fingerprint is an in-memory `u64` and is **never** logged, never
//! exported, and never leaves this struct — see the invariant below.
//!
//! ## Bearer-bytes-never-logged invariant (extends SR-6.2 to Valkey auth)
//!
//! This struct is credential-bearing: `fetch()` reads a live token, and the
//! username is the configured AUTH identity. The hand-written [`Debug`] impl
//! prints neither. The `fetch()` error path returns a generic descriptor with no
//! token bytes — in particular it must never grow a "reason" that distinguishes
//! *why* no credential was available, since the distinguishing state is derived
//! from the token itself. Grep this module before adding any logging or `Debug`
//! field.
//!
//! ## Identity self-check (b.cbe)
//!
//! [`valkey_identity_verdict`] compares the token's own **non-secret** identity
//! claims (`oid`, `appid`/`azp` — see
//! [`TokenClaims`](crate::auth::service_token::TokenClaims)) against the
//! configured AUTH username, so that the single opaque
//! `WRONGPASS invalid username-password pair` a managed backing store returns
//! for every distinct cause can be attributed to one of them.
//!
//! It changes nothing about [`TokenFileCredentialProvider::fetch`]. The verdict
//! travels to the **log**, never into the fred error: growing a "reason" on the
//! `fetch()` error is what the paragraph above forbids, and it would also
//! misdirect — that error means *we have no credential to send*, which is a
//! different fault from *the server rejected the credential we sent*.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
use fred::types::config::{CredentialProvider, Server};
use rand::Rng;
use sha2::{Digest, Sha256};

use crate::auth::service_token::{ServiceTokenProvider, TokenClaims};

/// The token the provider last observed, and the fan-out window it opened, if
/// any.
///
/// `fingerprint` identifies the cached token *without retaining it* — see
/// [`token_fingerprint`]. `window_opened_at` is when this fingerprint replaced
/// a different one, which starts the fan-out window; `None` when the provider
/// has seen no other token, i.e. this fingerprint was its first sighting
/// (b.19i): every connection open at that moment authenticated with this
/// token at its handshake, so there is nothing to fan out.
#[derive(Debug, Clone, Copy)]
struct RotationEpoch {
    fingerprint: u64,
    window_opened_at: Option<Instant>,
}

/// Reduce a token to a 64-bit in-memory identity.
///
/// SHA-256 truncated to its first 8 bytes. The point is only equality — "is
/// this the same credential I last saw?" — so any collision-resistant digest
/// would do; SHA-256 is chosen because the crate already depends on it and
/// because a non-cryptographic hasher would make a deliberate collision (and
/// thus a suppressed rotation) cheap to construct.
///
/// **The result never leaves the provider.** It is not logged, not printed by
/// [`Debug`], not exported as a metric label, and not returned by any public
/// method. Storing the digest rather than the token also means the rotation
/// state adds no second copy of credential material to the process (SR-6.2).
fn token_fingerprint(token: &str) -> u64 {
    let digest = Sha256::digest(token.as_bytes());
    let mut head = [0u8; 8];
    head.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(head)
}

/// fred [`CredentialProvider`] that authenticates Valkey connections with a live
/// file-sourced access token (SDD SR-4.1 / SR-4.2).
///
/// Construct via [`Self::new`]; wrap in `Arc` and install on fred's
/// `FredConfig::credential_provider` from the storage `apply_fred_config_overrides`
/// seam (Epic 3 wiring Subtask).
pub struct TokenFileCredentialProvider {
    /// Redis-scoped token provider constructed with a ≥ 3 min proactive
    /// re-read lead (see module docs). Read-only from here via
    /// [`ServiceTokenProvider::cached_bearer_if_unexpired`], which ignores that
    /// lead — see the module's "The lead is a re-read cadence, not a
    /// withholding window" section for why nothing here may gate on it.
    provider: Arc<ServiceTokenProvider>,
    /// Valkey AUTH username (SDD SR-4.1) — conventionally the token issuer's
    /// identity, e.g. an Azure managed identity's **Object ID** — sourced from
    /// `Config::valkey_token_username`. Credential-adjacent — never printed by
    /// [`Debug`].
    username: String,
    /// Base re-AUTH cadence. Kept short relative to the ~24 h token lifetime so
    /// a re-AUTH *may* land in the window where the rotated token is already
    /// cached, prolonging the connection past the old token's expiry instead of
    /// letting it drop there. That window's width is the **sidecar's** refresh
    /// lead less up to one token-file poll interval, not the service's own
    /// lead — see [`Self::refresh_interval`]. Sized by the caller; see
    /// `storage::client::VALKEY_TOKEN_REAUTH_BASE_INTERVAL`.
    base_interval: Duration,
    /// Upper bound on the per-client stagger added to `base_interval`. The
    /// realized jitter is uniform in `[0, jitter_bound]`, drawn fresh on each
    /// [`Self::refresh_interval`] call.
    jitter_bound: Duration,
    /// How long after a rotation is first observed the provider keeps answering
    /// the periodic `fetch(None)` with a credential (b.7td).
    ///
    /// The provider is installed once on the shared `FredConfig` and cloned as a
    /// single `Arc` across every pool and client, so it has no per-client
    /// identity and cannot track who has already re-`AUTH`ed. A time window is
    /// the only mechanism available, and it must be **at least**
    /// `base_interval + jitter_bound`: a client whose tick fired microseconds
    /// before the rotation was observed will not tick again for that long.
    ///
    /// Sized by the caller alongside the cadence, for the same reason the
    /// cadence is — all three numbers are one design and belong at one site. See
    /// `storage::client::VALKEY_TOKEN_AUTH_FANOUT_WINDOW`, where a `const`
    /// assertion mechanizes the pairing against the two cadence constants.
    auth_fanout_window: Duration,
    /// The token last observed and the fan-out window it opened, if any
    /// (b.7td; a first sighting opens none, b.19i), or `None` before the
    /// first `fetch()` that found a cached token. Read and written under a plain
    /// `std::sync::Mutex` — the critical section is a `u64` compare and an
    /// `Instant` subtraction with no `await` inside it, so an async mutex would
    /// buy nothing and cost a scheduler hop on a path fred takes for every
    /// connection handshake.
    ///
    /// Credential-adjacent: it holds a digest of the live token. Never printed
    /// by [`Debug`], never logged.
    rotation: Mutex<Option<RotationEpoch>>,
}

impl TokenFileCredentialProvider {
    /// Construct a provider.
    ///
    /// * `provider` — the Redis-scoped [`ServiceTokenProvider`] (built with a
    ///   ≥ 3 min refresh lead via
    ///   [`ServiceTokenProvider::new_with_refresh_lead`]).
    /// * `username` — the AUTH username, conventionally the token issuer's
    ///   identity (`Config::valkey_token_username`).
    /// * `base_interval` — base tick cadence.
    /// * `jitter_bound` — upper bound of the per-client stagger.
    /// * `auth_fanout_window` — how long a rotation keeps answering periodic
    ///   ticks with a credential; must be `>= base_interval + jitter_bound`.
    pub fn new(
        provider: Arc<ServiceTokenProvider>,
        username: String,
        base_interval: Duration,
        jitter_bound: Duration,
        auth_fanout_window: Duration,
    ) -> Self {
        debug_assert!(
            auth_fanout_window >= base_interval + jitter_bound,
            "TokenFileCredentialProvider: auth_fanout_window ({auth_fanout_window:?}) must be at \
             least base_interval + jitter_bound ({:?}), or a client whose tick fired just before \
             a rotation was observed gets no tick inside the window and never re-AUTHs with the \
             new token",
            base_interval + jitter_bound
        );
        Self {
            provider,
            username,
            base_interval,
            jitter_bound,
            auth_fanout_window,
            rotation: Mutex::new(None),
        }
    }

    /// Decide whether this `fetch(None)` tick should carry a credential, and
    /// fold the observation into the rotation epoch (b.7td).
    ///
    /// Returns `true` when the tick should re-`AUTH` — either because this
    /// tick is the first observation of a token that replaced another one, or
    /// because it falls inside `auth_fanout_window` of one.
    ///
    /// **A first sighting opens no window (b.19i).** With no token observed
    /// before, this call records the fingerprint with no window open and
    /// returns `false`, whichever path made it: every connection that exists
    /// at that moment authenticated with this same token at its handshake, so
    /// a re-`AUTH` would carry nothing new. Only a fingerprint that differs
    /// from a prior observation opens the window. Treating the first sighting
    /// as a rotation made every pooled client re-`AUTH` 60–90 s after every
    /// startup.
    ///
    /// Also called from the handshake path, where the return value is
    /// **ignored**: a handshake must always send the credential, but it is also
    /// a perfectly good observation, and letting a handshake that sees a
    /// changed token open the window means a reconnect that picks up a fresh
    /// token does not leave the periodic path to rediscover it a tick later.
    ///
    /// A poisoned lock is treated as "send". Losing the gate degrades to the
    /// pre-b.7td behaviour — more broadcasts, never a missed rotation — which
    /// is the right direction to fail in, and is strictly better than
    /// propagating a panic out of fred's connection setup.
    fn observe_and_should_reauth(&self, token: &str) -> bool {
        let fingerprint = token_fingerprint(token);
        let now = Instant::now();
        let Ok(mut rotation) = self.rotation.lock() else {
            return true;
        };
        match *rotation {
            Some(epoch) if epoch.fingerprint == fingerprint => epoch
                .window_opened_at
                .is_some_and(|opened| now.duration_since(opened) <= self.auth_fanout_window),
            // The token changed since the last observation: this call opens
            // the fan-out window.
            Some(_) => {
                *rotation = Some(RotationEpoch {
                    fingerprint,
                    window_opened_at: Some(now),
                });
                true
            }
            // The first sighting: record it, open nothing.
            None => {
                *rotation = Some(RotationEpoch {
                    fingerprint,
                    window_opened_at: None,
                });
                false
            }
        }
    }

    /// Fresh per-client stagger in `[0, jitter_bound]`.
    ///
    /// Draws fresh entropy from the thread RNG on **every** call — NOT a seeded,
    /// reproducible draw. fred spawns a fresh credential-refresh task (and thus
    /// calls `refresh_interval` afresh) per client on each connect/reconnect, so
    /// a fresh per-call draw spreads clients across the jitter band rather than
    /// re-AUTHing in lockstep (SR-4.2).
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
impl CredentialProvider for TokenFileCredentialProvider {
    /// Return `(Some(username), Some(token))` for the next `AUTH` / `HELLO`.
    ///
    /// The password is a live file-sourced token read from the Redis-scoped
    /// [`ServiceTokenProvider`] via
    /// [`ServiceTokenProvider::cached_bearer_if_unexpired`], so a cached token
    /// authenticates for its whole real lifetime — including the ≥ 3 min
    /// proactive-re-read lead at the end of it, during which the polling task
    /// is already hunting for the replacement (b.d6b).
    ///
    /// `Err(`[`FredErrorKind::Auth`]`)` survives for the two states that are
    /// still genuine credential-death, and only those: **no token cached**
    /// (sidecar has not written the file yet, or has never written a parseable
    /// one) and **cached token already expired**. Failing `AUTH` there is
    /// correct — there is no credential to send, and a blank / garbage one is
    /// never an option.
    ///
    /// The error message is deliberately **generic** and must stay that way: it
    /// names neither the token nor which of the two states it is in, both of
    /// which are derived from token material (SR-6.2). `Auth` — not a transient
    /// kind — keeps the classification consistent with
    /// `storage::client::is_transient_fred_error`, which treats auth failures
    /// as non-transient.
    ///
    /// ## `server` selects the caller (b.7td)
    ///
    /// fred passes `Some(server)` from connection setup and `None` from the
    /// periodic credential-refresh task — and a stream reader passes `None`
    /// from its own re-AUTH tick (b.19i) — and that is the only signal
    /// distinguishing them:
    ///
    /// - **`Some(_)` — handshake.** Always returns the credential. A connection
    ///   with no password never authenticates, so gating here would fail
    ///   connections outright. The observation is still folded into the
    ///   rotation epoch, so a reconnect that picks up a changed token opens the
    ///   fan-out window immediately rather than leaving the periodic path to
    ///   rediscover it a tick later. The provider's first sighting opens no
    ///   window, on this path or the tick's.
    /// - **`None` — periodic tick.** Returns `(Some(username), None)` unless
    ///   this token replaced an earlier one and the call is still inside the
    ///   provider's `auth_fanout_window` of that change; nothing is sent when
    ///   the password is `None`. See the module's "Rotation-gated re-AUTH"
    ///   section.
    ///
    /// The `Err(Auth)` contract above is identical on both paths and is
    /// evaluated **before** the gate: a tick with no usable credential is
    /// genuine credential death and must be reported as such, never quietly
    /// suppressed into a no-op.
    async fn fetch(
        &self,
        server: Option<&Server>,
    ) -> Result<(Option<String>, Option<String>), FredError> {
        let Some(token) = self.provider.cached_bearer_if_unexpired() else {
            return Err(FredError::new(
                FredErrorKind::Auth,
                "accessTokenFile valkey credential unavailable: no unexpired service-identity \
                 token is cached",
            ));
        };

        let is_handshake = server.is_some();
        // Called unconditionally — the handshake path ignores the verdict but
        // must still register the observation.
        let should_reauth = self.observe_and_should_reauth(&token);
        if is_handshake || should_reauth {
            Ok((Some(self.username.clone()), Some(token)))
        } else {
            // Username without password: fred's refresh loop (and a stream
            // reader's own tick) skips the AUTH / HELLO entirely for this tick.
            // Deliberately unlogged — this is the steady-state case, once per
            // client per minute.
            Ok((Some(self.username.clone()), None))
        }
    }

    /// Tick cadence: `base_interval` plus a fresh per-client stagger drawn on
    /// every call (see [`Self::jitter`]).
    ///
    /// **This is the rate at which the provider is *consulted*, not the rate at
    /// which fred re-`AUTH`s (b.7td).** Most ticks now return no password and
    /// send nothing on the wire — see [`Self::fetch`] and the module's
    /// "Rotation-gated re-AUTH" section. Once the broadcast is gated, a short
    /// cadence is cheap, and a short cadence is what makes rotation pickup
    /// *reliable*: the window in which the replacement token is cached but the
    /// old one has not yet expired is set by the **sidecar's** refresh lead less
    /// up to one `LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS`, and the shipped
    /// 60–90 s band lands inside it far more often than the 300–360 s band it
    /// replaces did.
    ///
    /// The two properties this cadence has to satisfy are coupled, and both are
    /// mechanized as `const` assertions next to the constants in
    /// `storage::client`:
    ///
    /// 1. `base_interval + jitter_bound` must not exceed
    ///    the provider's `auth_fanout_window`, or a client whose tick fired just
    ///    before a rotation was observed would not tick again inside the window
    ///    and would never receive the new credential.
    /// 2. The prolongation floor handed to the Redis-scoped
    ///    `ServiceTokenProvider` must cover **two** such intervals — one for the
    ///    first client to observe the rotation, one for the last client to take
    ///    its next tick — since a rotation with less headroom than that before
    ///    the outgoing token's expiry cannot be guaranteed to reach every live
    ///    connection.
    ///
    /// Missing the window still costs a reconnect, not an auth outage.
    /// [`Self::fetch`] serves any unexpired cached token, and every pooled
    /// connection was AUTHed with the same token, so a total miss drops them
    /// together at that token's expiry and they immediately re-establish with
    /// the by-then-cached fresh one — a blip once per token lifetime. Sizing
    /// the sidecar lead to cover the band is a deployment recommendation
    /// carried in the operator docs, and deliberately **not** a startup check
    /// (b.is1): the term it constrains is the *sidecar's* refresh lead, which is
    /// not a live-edit config value and is invisible at boot. What ships instead
    /// is a runtime observation — the prolongation floor is passed to the
    /// Redis-scoped `ServiceTokenProvider` (wired in `main.rs`), which records
    /// the real per-rotation headroom as
    /// `live_edit_service_token_refresh_headroom_seconds{role="valkey"}` and
    /// WARNs once per rotation that lands under it. The caller sizes every
    /// half; see `storage::client::VALKEY_TOKEN_REAUTH_BASE_INTERVAL`.
    ///
    /// Still returns `Some` unconditionally: fred spawns a pooled client's
    /// refresh task only when this is `Some`, a stream reader schedules its own
    /// re-AUTH tick only when this is `Some` (its client's config carries the
    /// provider wrapped to return `None`, so fred schedules nothing there), and
    /// the gate lives in `fetch()`, not here.
    fn refresh_interval(&self) -> Option<Duration> {
        Some(self.base_interval + self.jitter())
    }
}

/// Hand-written redacting [`Debug`] — the trait requires `Debug` and this struct
/// holds credential-adjacent state (the token provider and the AUTH
/// username). Prints only the non-secret scheduling fields; the token and
/// username never appear. Extends the bearer-bytes-never-logged invariant
/// (SR-6.2) to the Valkey auth surface.
impl fmt::Debug for TokenFileCredentialProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenFileCredentialProvider")
            .field("username", &"<redacted>")
            .field("base_interval", &self.base_interval)
            .field("jitter_bound", &self.jitter_bound)
            // Non-secret scheduling parameter, printed for the same reason the
            // other two are. The `rotation` epoch beside it is deliberately
            // absent: it holds a digest of the live token (b.7td).
            .field("auth_fanout_window", &self.auth_fanout_window)
            .finish_non_exhaustive()
    }
}

/// Verdict of cross-checking a token's own identity claims against the
/// configured Valkey AUTH username (b.cbe).
///
/// The backing store returns one opaque `WRONGPASS invalid username-password
/// pair` for every distinct cause of an AUTH rejection, so the discriminating
/// information is not in the server's response and no amount of parsing it can
/// recover which cause fired. This service holds the two halves the server does
/// not expose — its own configuration and the token bytes it is about to send —
/// and this is where they are compared.
///
/// **Judges `oid`, reports everything else.** There is deliberately no
/// hard-coded audience allowlist and no issuer allowlist: the `accessTokenFile`
/// mode is issuer-agnostic by design (SDD §15.3 / §15.4), so pinning an expected
/// `aud` here would make a correct non-Azure deployment log a false accusation.
/// The audience is *printed* in the diagnostic and the expected-value guidance
/// lives in the hint text and the operator docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IdentityVerdict {
    /// The cached credential is not a decodable JWT (opaque token, or no token
    /// cached at all). No self-check is possible — and this is not itself a
    /// fault, so the caller skips the identity half of the diagnostic silently.
    NotAJwt,
    /// The token decoded but carries neither `oid` nor `appid`/`azp`, so there
    /// is nothing to compare the configured username against.
    NoIdentityClaim,
    /// Healthy: the configured username equals the token's `oid` claim.
    MatchesOid,
    /// **The exact-cause case.** The configured username equals the token's
    /// client-id claim (`appid`, else `azp`) and is not its `oid` — i.e. the
    /// operator configured the client ID where the Object ID was required.
    MatchesClientIdClaim {
        /// The `oid` the username should be set to instead, when the token
        /// carries one.
        oid: Option<String>,
        /// The client-id claim the username currently matches. Non-secret and
        /// printed verbatim: it is exactly the value the operator must replace.
        client_id_claim: String,
    },
    /// The configured username matches neither identity claim — the token was
    /// minted for a different identity than the one configured (federation
    /// misbinding, or a stale username).
    MatchesNeither {
        oid: Option<String>,
        client_id_claim: Option<String>,
    },
}

/// Placeholder printed for a claim the token does not carry. Never an empty
/// string: an empty rendering reads as "the value is blank" rather than "the
/// claim is absent".
const ABSENT_CLAIM: &str = "<absent>";

impl IdentityVerdict {
    /// Short, stable label for the structured `identity_verdict` log field.
    /// Kept greppable and free of claim values so dashboards and alert rules
    /// can key off it.
    pub(crate) fn label(&self) -> &'static str {
        match self {
            IdentityVerdict::NotAJwt => "not-a-jwt",
            IdentityVerdict::NoIdentityClaim => "no-identity-claim",
            IdentityVerdict::MatchesOid => "matches-oid",
            IdentityVerdict::MatchesClientIdClaim { .. } => "matches-client-id-claim",
            IdentityVerdict::MatchesNeither { .. } => "matches-neither",
        }
    }

    /// The cause-specific sentence for the operator hint.
    ///
    /// **The configured username is never echoed.** Every statement is made
    /// *relationally* ("the configured username equals the token's `appid`"),
    /// and the concrete values printed are the token's own non-secret claims.
    /// That keeps this consistent with the redacting [`Debug`] on
    /// [`TokenFileCredentialProvider`] while staying fully actionable — the
    /// value the operator has to substitute (`oid`) is printed, and in the
    /// client-id case so is the wrong value, because it *is* a claim.
    pub(crate) fn guidance(&self) -> String {
        match self {
            // Covers BOTH shapes that reach this arm: no credential cached at
            // all (the sidecar has not written one yet, or the cached entry
            // expired), and a cached credential that is not a decodable JWT.
            // Asserting the second alone would be a false statement in the
            // empty-cache case and would send the operator inspecting a token
            // that does not exist.
            IdentityVerdict::NotAJwt => "no cached Valkey credential is available to inspect, or \
                 the cached credential is not a decodable JWT, so no token-claims self-check is \
                 available (an opaque token is legitimate — it just carries no readable \
                 identity). Check whether the credential sidecar is writing the token file, then \
                 check the AUTH username and the token's audience against the backing store by \
                 hand."
                .to_string(),
            IdentityVerdict::NoIdentityClaim => "the token decoded but carries neither an `oid` \
                 nor an `appid`/`azp` claim, so its identity cannot be cross-checked against \
                 LIVE_EDIT_VALKEY_TOKEN_USERNAME. Confirm with the token issuer which claim \
                 carries the identity the backing store's access policy is keyed by."
                .to_string(),
            IdentityVerdict::MatchesOid => "LIVE_EDIT_VALKEY_TOKEN_USERNAME equals the token's \
                 `oid` claim, which is the expected identity binding — causes (1) and (3) below \
                 are ruled out from this service's own data, so look at (2) and especially (4)."
                .to_string(),
            IdentityVerdict::MatchesClientIdClaim {
                oid,
                client_id_claim,
            } => format!(
                "LIVE_EDIT_VALKEY_TOKEN_USERNAME is set to the token's client-id claim \
                 ({client_id_claim}), not to its `oid`. This is the most common cause of an \
                 opaque AUTH rejection: a managed backing store keys its data-plane access \
                 policy by the identity's Object ID, not its client ID. Set \
                 LIVE_EDIT_VALKEY_TOKEN_USERNAME to {}.",
                match oid {
                    Some(oid) => format!("`{oid}`"),
                    None =>
                        "the identity's Object ID (this token carries no `oid` claim to read it \
                         from)"
                            .to_string(),
                }
            ),
            IdentityVerdict::MatchesNeither {
                oid,
                client_id_claim,
            } => format!(
                "LIVE_EDIT_VALKEY_TOKEN_USERNAME matches neither the token's `oid` ({}) nor its \
                 client-id claim ({}) — the token was minted for a different identity than the \
                 one configured (federation misbinding, or a stale username). Set \
                 LIVE_EDIT_VALKEY_TOKEN_USERNAME to the `oid` above, or point the credential \
                 sidecar at the identity you intended.",
                oid.as_deref().unwrap_or(ABSENT_CLAIM),
                client_id_claim.as_deref().unwrap_or(ABSENT_CLAIM),
            ),
        }
    }
}

/// Cross-check a token's identity claims against the configured Valkey AUTH
/// username (b.cbe).
///
/// Pure and total: no I/O, no logging, no clock. `claims == None` means the
/// cached credential was not a decodable JWT (or nothing is cached) and yields
/// [`IdentityVerdict::NotAJwt`].
///
/// Lives here, next to the provider, because this is the module that owns the
/// configured username — the other half of the comparison is
/// `Config::valkey_token_username`, which is what
/// [`TokenFileCredentialProvider`] sends as the AUTH username.
///
/// Evaluation order is deliberate: `oid` is checked first, so a token whose
/// `oid` and `appid` happen to be equal reports the healthy verdict rather than
/// accusing the operator of the client-id mistake.
pub(crate) fn valkey_identity_verdict(
    claims: Option<&TokenClaims>,
    configured_username: &str,
) -> IdentityVerdict {
    let Some(claims) = claims else {
        return IdentityVerdict::NotAJwt;
    };

    if claims.oid.as_deref() == Some(configured_username) {
        return IdentityVerdict::MatchesOid;
    }
    if let Some(client_id_claim) = claims.client_id_claim.as_deref() {
        if client_id_claim == configured_username {
            return IdentityVerdict::MatchesClientIdClaim {
                oid: claims.oid.clone(),
                client_id_claim: client_id_claim.to_string(),
            };
        }
    }
    if claims.oid.is_none() && claims.client_id_claim.is_none() {
        return IdentityVerdict::NoIdentityClaim;
    }
    IdentityVerdict::MatchesNeither {
        oid: claims.oid.clone(),
        client_id_claim: claims.client_id_claim.clone(),
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `TokenFileCredentialProvider` — Subtask `t3.tr5.jv.2b.dj`.
    //!
    //! Coverage targets (parent Task `t2.tr5.jv.2b` acceptance criteria, SDD
    //! SR-4.1 / SR-4.2):
    //!
    //! - `fetch()` returns `(username = Object ID, password = current token)`
    //!   once the sidecar token file is present.
    //! - `fetch()` reflects a rewritten token after one poll interval (the
    //!   Redis-scoped `ServiceTokenProvider` re-reads the file on its tick).
    //! - Token-unavailable contract: `fetch()` yields `Err(ErrorKind::Auth)` (a
    //!   live `AUTH` failure), never a blank / garbage credential, for the two
    //!   states that are genuine credential-death — **no token cached** (the
    //!   sidecar has not written a parseable file yet) and **cached token
    //!   already expired**. A token inside the ≥ 3 min proactive re-read lead
    //!   is NOT one of them: `fetch()` reads `cached_bearer_if_unexpired()`, so
    //!   a cached token authenticates for its whole real lifetime (b.d6b). That
    //!   AUTH failure trips the existing Valkey `PING` readiness probe →
    //!   `NOT_SERVING` (SDD §17.3 "Valkey probe (always)", no smoothing); no
    //!   new `ProbeConfig` slot is added for the access-token-file path.
    //! - `refresh_interval()` draws a fresh per-call stagger that always stays
    //!   within `[base_interval, base_interval + jitter_bound]`, and repeated
    //!   calls on a single provider spread across the band rather than locking
    //!   in step (SR-4.2 thundering-herd per-client stagger).
    //! - `Debug` never prints the Object-ID username or token bytes
    //!   (bearer-bytes-never-logged, SR-6.2 extended to Valkey auth).
    //! - Rotation-gated re-AUTH (b.7td): which `fetch()` calls carry a
    //!   credential and which return `(Some(username), None)` so nothing is
    //!   sent — the gate states (a first sighting, which records the token and
    //!   opens no window (b.19i); a changed token, which opens it; inside the
    //!   fan-out window; outside it with an unchanged token), plus the two
    //!   ungated contracts that sit around it (the handshake always sends — and
    //!   registers the epoch it observed, so a rotation picked up by a
    //!   reconnect does not fan out a second time; credential death is
    //!   reported before the gate is consulted). The fan-out window is a constructor parameter, which is
    //!   what makes the gate testable at all — see [`TINY_FANOUT_WINDOW`] for
    //!   why the window is bought with real wall time and not a paused clock.
    //! - The rotation fingerprint is credential-derived state and obeys the
    //!   same never-logged / never-`Debug`-printed invariant as the token.
    //!
    //! **What none of this proves.** The field failure b.7td was filed for — a
    //! periodic re-`AUTH` landing on a pooled connection that was carrying
    //! leaked transaction state, on a cluster-mode managed backing store — is
    //! not reproducible here or anywhere in this suite. It needs a real
    //! cluster, a real token rotation and the managed proxy. What is tested is
    //! that the broadcast the failure rode in on is now issued only when a
    //! rotation actually needs it.
    //!
    //! - Composition with the token provider's stale-token fast-poll phase
    //!   (b.qam): a late sidecar rotation moves `fetch()` from `Err(Auth)` back
    //!   to `Ok` sub-second rather than on the next steady poll. This is the
    //!   Valkey-shaped statement of the fix — once the cached token actually
    //!   expires there is no credential left to send, so every `AUTH` fails
    //!   until the replacement is read, and each of those seconds trips the
    //!   readiness probe. The ≥ 3 min lead's job is to put the polling task
    //!   into that fast-poll phase *ahead* of expiry, while `fetch()` is still
    //!   authenticating normally off the cached token (b.d6b).
    //!
    //! The `service_token.rs` harness helpers (`write_token_file`,
    //! `fresh_metrics`, `TEST_POLL_INTERVAL`, `await_one_poll`) live in that
    //! module's private `#[cfg(test)]` block and are NOT reachable from here,
    //! so this module carries minimal local equivalents that write the same
    //! `{"access_token","expires_on"}` JSON the provider reads and drive the
    //! same 20 ms poll interval.

    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use chrono::Utc;
    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;
    use tracing_subscriber::fmt::MakeWriter;

    use super::*;

    // ── Test helpers (local copies of the service_token.rs harness) ─────────

    /// A `MakeWriter` that captures every byte written to it into a shared
    /// buffer. Local copy of `service_token.rs::tests::CapturingWriter` — that
    /// one lives in a private `#[cfg(test)]` module and is not reachable
    /// cross-module, so the bearer-bytes-never-logged test carries its own.
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

    /// Poll interval used by every polling-driven test — 20 ms, matching the
    /// `service_token.rs` harness `TEST_POLL_INTERVAL`.
    const TEST_POLL_INTERVAL: Duration = Duration::from_millis(20);

    /// Spend one poll interval plus scheduling slack.
    ///
    /// **Only for negative observations** — where the elapsed window *is* the
    /// assertion. It is not a bound on anything for a positive assertion: the
    /// token provider's reads run on the blocking pool, so under a loaded
    /// runner they can take far longer, and the test-writing guide's
    /// § Synchronization Patterns rules the pattern out. [`poll_until`] and
    /// [`await_cached_token`] are the positive-assertion equivalents.
    async fn await_one_poll() {
        tokio::time::sleep(TEST_POLL_INTERVAL + Duration::from_millis(40)).await;
    }

    /// Budget for the "the token provider's polling task has observed an
    /// on-disk change" waits. A ceiling a healthy run exits in single-digit
    /// milliseconds, spent in full only on a genuine hang; the `service_token`
    /// harness carries the same constant for the same reason.
    const POLL_PICKUP_BUDGET: Duration = Duration::from_millis(2000);

    /// Bounded probe loop — the test-writing guide's "wait for some background
    /// condition" pattern. Returns `true` if `predicate` held before `budget`
    /// elapsed. Local equivalent of `service_token.rs::tests::poll_until`,
    /// which lives in that module's private `#[cfg(test)]` block.
    async fn poll_until(budget: Duration, mut predicate: impl FnMut() -> bool) -> bool {
        let started = std::time::Instant::now();
        loop {
            if predicate() {
                return true;
            }
            if started.elapsed() >= budget {
                return false;
            }
            tokio::time::sleep(PROBE_INTERVAL).await;
        }
    }

    /// Wait until the token provider's polling task has cached `expected`.
    ///
    /// Watches the cache (`cached_bearer_if_unexpired()`), never `fetch()`,
    /// so the wait is not itself an observation the rotation gate folds in.
    /// Each test then chooses which `fetch()` call is the provider's first
    /// sighting of a token, and which sees a changed one. Since b.19i that
    /// choice decides the outcome: a first sighting opens no fan-out window,
    /// so a `fetch(None)` wait would leave the gate in a state the test never
    /// named.
    async fn await_cached_token(provider: &TokenFileCredentialProvider, expected: &str) {
        assert!(
            poll_until(POLL_PICKUP_BUDGET, || provider
                .provider
                .cached_bearer_if_unexpired()
                .as_deref()
                == Some(expected))
            .await,
            "the polling task must cache {expected:?} within {POLL_PICKUP_BUDGET:?}"
        );
    }

    /// The handshake call (`fetch(Some(server))`), which always carries the
    /// credential whatever the gate says. The tests that check *which
    /// credential* the provider presents ask through it: since b.19i a
    /// periodic tick on the provider's first sighting carries no password.
    async fn handshake_fetch(
        provider: &TokenFileCredentialProvider,
    ) -> (Option<String>, Option<String>) {
        let server = handshake_server();
        provider
            .fetch(Some(&server))
            .await
            .expect("a handshake with a live cached token must produce a credential")
    }

    /// Poll interval for the stale-token fast-poll composition test.
    ///
    /// [`TEST_POLL_INTERVAL`] (20 ms) is *shorter* than the provider's
    /// `FAST_POLL_RETRY_INTERVAL` (250 ms), so under it a "recovered within a
    /// second" assertion is satisfied by the steady poll alone and proves
    /// nothing about the fast cadence. 60 s is 240× the fast cadence, which
    /// makes any sub-second recovery attributable to the fast phase and to
    /// nothing else.
    const LONG_POLL_INTERVAL: Duration = Duration::from_secs(60);

    /// Probe granularity for the bounded wait loops in the stale-token test.
    /// Small relative to the 250 ms fast cadence so the observed transition
    /// instants are not quantisation artefacts.
    const PROBE_INTERVAL: Duration = Duration::from_millis(5);

    /// Local mirror of the token provider's private `FAST_POLL_RETRY_INTERVAL`
    /// (`auth::service_token`), which is module-private and not nameable here.
    /// Used only to size budgets in
    /// [`fetch_recovers_sub_second_from_a_late_rotation`]; those are ceilings a
    /// healthy run exits early from, not contracts, so a stale copy degrades a
    /// diagnostic rather than an assertion.
    const FAST_POLL_RETRY_INTERVAL: Duration = Duration::from_millis(250);

    /// Local mirror of the token provider's private `STARTUP_FAST_POLL_WINDOW`
    /// in its b.d6b role: the wall-clock bound on the *post-expiry* fast-poll
    /// window, i.e. how long after a cached token dies the provider keeps
    /// hunting for its replacement at [`FAST_POLL_RETRY_INTERVAL`] before
    /// reverting to the steady poll.
    ///
    /// Same standing as the mirror above — it sizes the "did the fixture stall
    /// past the window?" diagnostic in
    /// [`fetch_recovers_sub_second_from_a_late_rotation`], never an assertion
    /// about production behaviour. The bound itself is pinned inside
    /// `auth::service_token` by
    /// `post_expiry_fast_poll_stops_after_the_startup_window`, where the
    /// constant is nameable.
    const POST_EXPIRY_FAST_WINDOW: Duration = Duration::from_secs(20);

    /// [`write_token_file`] with an arbitrary — in particular, **sub-second** —
    /// lifetime.
    ///
    /// The token provider's stale-token phase is driven by `chrono::Utc::now()`
    /// (wall clock), which `#[tokio::test(start_paused = true)]` does not
    /// virtualise, so an expiry-driven test cannot buy its wall-time budget
    /// from a paused clock. It buys it from short-lived *tokens* instead. Same
    /// helper, same rationale, as `service_token.rs::tests::write_token_file_in`.
    async fn write_token_file_in(
        path: &std::path::Path,
        token: &str,
        expires_in: chrono::Duration,
    ) {
        let expires_on = Utc::now() + expires_in;
        let body = format!(
            r#"{{"access_token":"{}","expires_on":"{}"}}"#,
            token,
            expires_on.to_rfc3339()
        );
        tokio::fs::write(path, body).await.expect("write fixture");
    }

    /// The ≥ 3 min widened proactive re-read lead the access-token-file Valkey
    /// path uses (Azure asks that a fresh token be *fetched* ≥ 3 min before
    /// expiry, SR-4.1). It is not a window in which the current token stops
    /// working — see `fetch_serves_cached_token_inside_the_refresh_lead`.
    const WIDE_EXPIRY_SKEW: chrono::Duration = chrono::Duration::seconds(180);

    /// The same ≥ 3 min window as a [`std::time::Duration`] — the headroom the
    /// max re-AUTH interval (`base_interval + jitter_bound`) must leave ahead of
    /// the token lifetime, so at least one re-AUTH lands inside the lead once
    /// the replacement token is cached (SR-4.2; see
    /// [`refresh_interval_ceiling_leaves_three_min_headroom`] for the full
    /// reasoning).
    const THREE_MIN: Duration = Duration::from_secs(180);

    /// A managed-identity Object ID (SR-4.1 — Object ID, NOT the client ID).
    /// A GUID-shaped fixture so the redaction test has a distinctive needle.
    const OBJECT_ID: &str = "11111111-2222-3333-4444-555555555555";

    /// Construct a fresh, self-contained `Metrics` so tests never touch a
    /// global registry (mirrors `service_token.rs::fresh_metrics`).
    fn fresh_metrics() -> Arc<crate::observability::metrics::Metrics> {
        Arc::new(crate::observability::metrics::metrics_scrape_only(
            "token-file-credentials-test",
        ))
    }

    /// Write the sidecar token-file JSON the provider reads: an
    /// `{"access_token","expires_on"}` object with `expires_on` `N` seconds
    /// from now (negative = already expired). Byte-shape identical to
    /// `service_token.rs::write_token_file`.
    async fn write_token_file(path: &std::path::Path, token: &str, expires_in_secs: i64) {
        let expires_on = Utc::now() + chrono::Duration::seconds(expires_in_secs);
        let body = format!(
            r#"{{"access_token":"{}","expires_on":"{}"}}"#,
            token,
            expires_on.to_rfc3339()
        );
        tokio::fs::write(path, body).await.expect("write fixture");
    }

    /// Build an `TokenFileCredentialProvider` backed by a Redis-scoped
    /// `ServiceTokenProvider` pointed at `path` with the given pre-expiry
    /// `expiry_skew`. Fixed `base_interval` (30 s) / `jitter_bound` (10 s).
    /// `refresh_interval()` draws fresh entropy on each call — there is no seed.
    /// Returns the provider plus the cancellation token so the test can stop the
    /// polling task.
    fn build_token_file_provider(
        path: std::path::PathBuf,
        expiry_skew: chrono::Duration,
    ) -> (TokenFileCredentialProvider, CancellationToken) {
        build_token_file_provider_with_poll(path, TEST_POLL_INTERVAL, expiry_skew)
    }

    /// [`build_token_file_provider`] with an explicit poll interval.
    ///
    /// Separate because [`TEST_POLL_INTERVAL`] is shorter than the token
    /// provider's fast-poll cadence and so cannot discriminate a fast-cadence
    /// pickup from a steady one — see [`LONG_POLL_INTERVAL`].
    fn build_token_file_provider_with_poll(
        path: std::path::PathBuf,
        poll_interval: Duration,
        expiry_skew: chrono::Duration,
    ) -> (TokenFileCredentialProvider, CancellationToken) {
        build_token_file_provider_with_band(
            path,
            poll_interval,
            expiry_skew,
            FIXTURE_BASE_INTERVAL,
            FIXTURE_JITTER_BOUND,
            FIXTURE_FANOUT_WINDOW,
        )
    }

    /// [`build_token_file_provider_with_poll`] with an explicit cadence band and
    /// fan-out window (b.7td).
    ///
    /// The rotation-gate tests need a window they can outlive inside a test —
    /// see [`TINY_FANOUT_WINDOW`] — and `new()`'s `debug_assert!` requires
    /// `auth_fanout_window >= base_interval + jitter_bound`, so the cadence has
    /// to shrink with it. Every other test keeps the 30 s / 10 s / 60 s fixture
    /// band through the two wrappers above.
    fn build_token_file_provider_with_band(
        path: std::path::PathBuf,
        poll_interval: Duration,
        expiry_skew: chrono::Duration,
        base_interval: Duration,
        jitter_bound: Duration,
        auth_fanout_window: Duration,
    ) -> (TokenFileCredentialProvider, CancellationToken) {
        let cancel = CancellationToken::new();
        let token_provider = Arc::new(ServiceTokenProvider::new_with_refresh_lead(
            path,
            poll_interval,
            expiry_skew,
            fresh_metrics(),
            cancel.clone(),
        ));
        let provider = TokenFileCredentialProvider::new(
            token_provider,
            OBJECT_ID.to_string(),
            base_interval,
            jitter_bound,
            auth_fanout_window,
        );
        (provider, cancel)
    }

    /// Base tick cadence every non-gate test's provider is built with.
    const FIXTURE_BASE_INTERVAL: Duration = Duration::from_secs(30);
    /// Jitter bound paired with [`FIXTURE_BASE_INTERVAL`].
    const FIXTURE_JITTER_BOUND: Duration = Duration::from_secs(10);
    /// Fan-out window for the fixture band, sized by the same rule the
    /// production constants follow: `>= base + jitter` (40 s), with slack.
    ///
    /// Wide enough that no test can accidentally outlive it: a test that opens
    /// the window by staging a rotation stays inside it for its whole run, so
    /// an "every tick inside the window carries the token" assertion cannot
    /// be failed by a slow runner.
    const FIXTURE_FANOUT_WINDOW: Duration = Duration::from_secs(60);

    /// Fan-out window for the rotation-gate tests: short enough that a test can
    /// outlive it in wall-clock time (b.7td).
    ///
    /// **The window cannot be advanced by a paused clock.**
    /// [`RotationEpoch::window_opened_at`] holds a `std::time::Instant`, and
    /// `#[tokio::test(start_paused = true)]` virtualises `tokio::time` only —
    /// `tokio::time::advance` moves `tokio::time::Instant::now()`, never
    /// `std::time::Instant::now()`. So the gate tests buy their window from real
    /// wall time, and buy it cheaply by shrinking the window rather than by
    /// waiting out a production-sized one.
    const TINY_FANOUT_WINDOW: Duration = Duration::from_millis(60);
    /// Base tick cadence paired with [`TINY_FANOUT_WINDOW`]. Only has to satisfy
    /// `base + jitter <= window`; nothing in the gate tests waits on it, because
    /// `fetch()` is called directly rather than through fred's refresh loop.
    const TINY_BASE_INTERVAL: Duration = Duration::from_millis(20);
    /// Jitter bound paired with [`TINY_BASE_INTERVAL`].
    const TINY_JITTER_BOUND: Duration = Duration::from_millis(10);

    /// A provider on the [`TINY_FANOUT_WINDOW`] band, for the rotation-gate
    /// tests.
    fn build_gated_provider(
        path: std::path::PathBuf,
    ) -> (TokenFileCredentialProvider, CancellationToken) {
        build_token_file_provider_with_band(
            path,
            TEST_POLL_INTERVAL,
            WIDE_EXPIRY_SKEW,
            TINY_BASE_INTERVAL,
            TINY_JITTER_BOUND,
            TINY_FANOUT_WINDOW,
        )
    }

    /// Spend more than one [`TINY_FANOUT_WINDOW`] of real time.
    ///
    /// Same standing as [`await_one_poll`], and sound for the same reason: it
    /// backs a **negative** observation ("the gate has closed"), where the
    /// elapsed window IS the assertion, and the property is monotone — once
    /// `now - window_opened_at` has passed the window it never comes back inside it,
    /// so a slow runner can only make the assertion more true. The 90 ms slack
    /// over the 60 ms window covers scheduler jitter.
    async fn spend_the_fanout_window() {
        tokio::time::sleep(TINY_FANOUT_WINDOW + Duration::from_millis(90)).await;
    }

    /// A fred [`Server`] to pass as `fetch(Some(..))`'s handshake argument.
    ///
    /// Never connected to: `fetch()` only reads whether the argument is `Some`,
    /// which is the entire signal distinguishing fred's connection-setup call
    /// from its periodic refresh tick.
    fn handshake_server() -> Server {
        Server::new("127.0.0.1", 6379)
    }

    // ── fetch(): username = Object ID, password = current token ─────────────
    //
    // These tests ask through the handshake (`fetch(Some(server))`), which
    // always carries the credential. A periodic tick on the provider's first
    // sighting of a token carries none (b.19i); the gate tests further down
    // pin that.

    #[tokio::test]
    async fn fetch_returns_object_id_username_and_current_token_password() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);
        await_cached_token(&provider, "token-A").await;

        let (username, password) = handshake_fetch(&provider).await;
        assert_eq!(
            username,
            Some(OBJECT_ID.to_string()),
            "username must be the managed identity's Object ID (SR-4.1)"
        );
        assert_eq!(
            password,
            Some("token-A".to_string()),
            "password must be the current token bytes"
        );

        cancel.cancel();
    }

    // ── fetch(): refresh visibility after a file rewrite ────────────────────

    #[tokio::test]
    async fn fetch_reflects_rewritten_token_after_one_poll() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_token_file_provider(path.clone(), WIDE_EXPIRY_SKEW);
        await_cached_token(&provider, "token-A").await;
        let (_, before) = handshake_fetch(&provider).await;
        assert_eq!(
            before,
            Some("token-A".to_string()),
            "precondition: before the rewrite fetch() presents token-A, or the assertion below \
             cannot tell a reflected rewrite from a token that was always token-B"
        );

        // Sidecar rotates the token; on the provider's next tick fetch() must
        // reflect the new bytes without any provider restart.
        write_token_file(&path, "token-B", 3600).await;
        await_cached_token(&provider, "token-B").await;
        let (username, password) = handshake_fetch(&provider).await;
        assert_eq!(
            username,
            Some(OBJECT_ID.to_string()),
            "username stays the Object ID across a token rotation"
        );
        assert_eq!(
            password,
            Some("token-B".to_string()),
            "password must reflect the rotated token after one poll interval"
        );

        cancel.cancel();
    }

    // ── fetch(): token-unavailable → Err(Auth), never a blank credential ────

    #[tokio::test]
    async fn fetch_errors_auth_when_no_token_file() {
        let dir = tempdir().expect("tempdir");
        // Path never written — the polling task's NotFound branch leaves the
        // cache empty, so `cached_bearer_if_unexpired()` (what fetch() reads)
        // returns None: the nothing-cached half of the Err(Auth) contract,
        // reached without any expiry travel.
        let path = dir.path().join("never-written.json");
        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);
        await_one_poll().await;

        let err = provider.fetch(None).await.expect_err(
            "fetch MUST fail AUTH when no token is available, not send a blank credential",
        );
        // ErrorKind::Auth (not a transient kind) so fred fails the connection
        // AUTH → the Valkey PING probe trips → readiness NOT_SERVING
        // (SDD §17.3 "Valkey probe (always)", no smoothing; no new
        // ProbeConfig slot for the access-token-file path).
        assert!(
            matches!(err.kind(), FredErrorKind::Auth),
            "token-unavailable must surface as ErrorKind::Auth, got {:?}",
            err.kind()
        );

        cancel.cancel();
    }

    /// A cached token inside the ≥ 3 min refresh lead is still a working
    /// credential and MUST authenticate (b.d6b).
    ///
    /// The inverse of what this test asserted pre-b.d6b, and the headline
    /// regression: `fetch()` used to read a second, *withholding* accessor whose
    /// policy turned the lead into a ≥ 3 min daily window in which every fred
    /// `AUTH` / `HELLO` failed and the Valkey readiness probe tripped — while a
    /// perfectly valid token sat in the cache. `fetch()` now reads
    /// `cached_bearer_if_unexpired()`, which ignores the lead.
    ///
    /// That withholding accessor lost its last consumer in b.e49 and was deleted
    /// in b.mq8, so serving the token is the provider's only bearer policy. The
    /// provider-side half of this contract is pinned by `service_token.rs`'s
    /// `cached_bearer_if_unexpired_serves_a_token_inside_the_refresh_lead` — the
    /// same fixture shape, one layer down.
    #[tokio::test]
    async fn fetch_serves_cached_token_inside_the_refresh_lead() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token-soon.json");
        // Expires in 150 s — INSIDE the 3 min (180 s) refresh lead, and 150 s
        // away from being a dead credential.
        write_token_file(&path, "token-soon", 150).await;

        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);
        // Probe the REAL precondition — a populated cache — rather than
        // spending a fixed poll interval. Pre-b.d6b this test documented that
        // it could not: a not-yet-read file and a cached-but-withheld token
        // were indistinguishable through this module's surface. The new
        // `cached_token_expires_on()` accessor closes that gap, so the wait is
        // now a bounded probe on the thing that must be true and the assertion
        // below can no longer pass on a stalled read.
        assert!(
            poll_until(POLL_PICKUP_BUDGET, || provider
                .provider
                .cached_token_expires_on()
                .is_some())
            .await,
            "the polling task must land the 150 s fixture in the cache within \
             {POLL_PICKUP_BUDGET:?}; without that read this test would assert nothing"
        );

        // Through the handshake: `Err(Auth)` is evaluated before the rotation
        // gate on both paths, and the handshake is the one that carries the
        // token on a first sighting (b.19i).
        let server = handshake_server();
        let (username, password) = provider.fetch(Some(&server)).await.expect(
            "a still-unexpired token inside the ≥ 3 min refresh lead must authenticate: the lead \
             is proactive-re-read guidance, not a statement that the token has stopped working",
        );
        assert_eq!(
            username,
            Some(OBJECT_ID.to_string()),
            "username must be the managed identity's Object ID (SR-4.1)"
        );
        assert_eq!(
            password,
            Some("token-soon".to_string()),
            "password must be the cached token, served for its whole real lifetime"
        );

        cancel.cancel();
    }

    /// The complement: an *expired* cached token is not a credential, and
    /// `fetch()` must still refuse it.
    ///
    /// b.d6b narrowed the `Err(Auth)` surface to exactly two states — nothing
    /// cached (`fetch_errors_auth_when_no_token_file`) and cached-but-expired
    /// (here). Non-vacuous by construction: the wait below is on the cache
    /// reporting an expiry in the *past*, so this cannot be the not-yet-read
    /// state the sibling test already covers.
    #[tokio::test]
    async fn fetch_errors_auth_when_cached_token_expired() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token-expired.json");
        // Already 10 s dead when the sidecar file is first read.
        write_token_file(&path, "token-dead", -10).await;

        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);
        assert!(
            poll_until(POLL_PICKUP_BUDGET, || provider
                .provider
                .cached_token_expires_on()
                .is_some_and(|expires_on| expires_on <= Utc::now()))
            .await,
            "the polling task must cache the already-expired fixture within {POLL_PICKUP_BUDGET:?}; \
             until it does, the Err below would be the empty-cache state, not the expired one"
        );

        let err = provider.fetch(None).await.expect_err(
            "an expired cached token is not a credential; fetch() must fail AUTH rather than send it",
        );
        // ErrorKind::Auth (not a transient kind) so fred fails the connection
        // AUTH → the Valkey PING probe trips → readiness NOT_SERVING, matching
        // `fetch_errors_auth_when_no_token_file`. The two states are
        // deliberately indistinguishable from the error itself (SR-6.2: the
        // distinguishing state is derived from token material).
        assert!(
            matches!(err.kind(), FredErrorKind::Auth),
            "an expired token must surface as ErrorKind::Auth, got {:?}",
            err.kind()
        );

        cancel.cancel();
    }

    // ── Rotation-gated re-AUTH: which ticks carry a credential (b.7td) ──────
    //
    // fred sends nothing on a tick whose `password` is `None`
    // (`fred-10.1.0/src/modules/inner.rs`, `spawn_credential_refresh`: the
    // `if let Some(password) = password` guarding the `hello` / `auth` call is
    // the last statement in the loop body, so a `None` falls through it with
    // nothing written — there is no `else` and no `continue` on that guard),
    // so "the provider returned `(Some(username), None)`" IS "no
    // `AUTH` went on the wire" — the whole point of the gate, and the only
    // observable this crate has for it. A stream reader's own re-AUTH tick
    // (b.19i) follows the same rule. The tests below drive the gate's states
    // through `fetch()` and never touch `observe_and_should_reauth` directly:
    // the provider's first sighting of a token (b.19i: recorded, no window, no
    // password, whether a handshake or a tick saw it), a changed token (opens
    // the window and carries it), inside the window, and outside it.
    //
    // Since a first sighting opens no window, a test that needs a window open
    // stages a real rotation: the provider sees token-A first, then the sidecar
    // writes token-B.

    /// The steady state the gate exists to create: the token has not changed
    /// and the fan-out window has closed, so the periodic tick carries the
    /// username and **no password**, and fred issues nothing.
    ///
    /// Pre-b.7td every one of these ticks was a real `AUTH`, and on a cluster
    /// each one fanned out to every primary (`_AuthAllCluster`) — the
    /// unbounded head-of-line broadcast the field failure surfaced through.
    #[tokio::test]
    async fn fetch_periodic_tick_sends_no_password_once_the_fanout_window_closes() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_gated_provider(path.clone());
        // The handshake is the provider's first sighting of token-A: it opens
        // no window (b.19i).
        await_cached_token(&provider, "token-A").await;
        let _ = handshake_fetch(&provider).await;

        // The sidecar rotates. The first tick to see token-B opens the window,
        // and carries the new token.
        write_token_file(&path, "token-B", 3600).await;
        await_cached_token(&provider, "token-B").await;
        let (_, opening) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            opening,
            Some("token-B".to_string()),
            "precondition: the rotation must open the fan-out window, or the closed-window \
             assertions below cannot tell 'the window closed' from 'no window ever opened'"
        );

        spend_the_fanout_window().await;

        // Three consecutive ticks, because one suppressed tick followed by a
        // re-opening would leave the broadcast rate halved rather than gated.
        for tick in 1..=3 {
            let (username, password) = provider
                .fetch(None)
                .await
                .expect("an unexpired cached token is never a credential failure");
            assert_eq!(
                username,
                Some(OBJECT_ID.to_string()),
                "tick {tick}: the username must still be sent — fred keys its refresh on the \
                 password being absent, not the pair being absent"
            );
            assert_eq!(
                password, None,
                "tick {tick}: an unchanged token outside the fan-out window must produce NO \
                 password, so fred's refresh loop skips the AUTH entirely. A `Some(..)` here is \
                 the pre-b.7td permanent broadcast — on a cluster, one `_AuthAllCluster` fan-out \
                 per client per tick, forever, carrying a credential that changes once a day"
            );
        }

        cancel.cancel();
    }

    /// Inside the window the answer is the opposite, and must be, or the
    /// fan-out never completes.
    ///
    /// The provider is a single shared `Arc` with no per-client identity, so it
    /// cannot know which clients have already re-`AUTH`ed; holding the window
    /// open for every client to take at least one tick inside it is the only
    /// mechanism available. A gate that closed after the first *observation*
    /// rather than after the *window* would re-`AUTH` exactly one connection
    /// per rotation and leave every other one to be dropped at expiry.
    ///
    /// Runs on the wide [`FIXTURE_FANOUT_WINDOW`] deliberately: the assertion
    /// is "still inside the window", and against a 60 ms window a stalled
    /// runner could leave it while the test was mid-loop.
    ///
    /// The window is opened by a staged rotation (token-A seen first, then
    /// token-B), since a first sighting opens none (b.19i).
    #[tokio::test]
    async fn fetch_periodic_ticks_inside_the_fanout_window_keep_carrying_the_token() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_token_file_provider(path.clone(), WIDE_EXPIRY_SKEW);
        await_cached_token(&provider, "token-A").await;
        let _ = handshake_fetch(&provider).await;
        write_token_file(&path, "token-B", 3600).await;
        await_cached_token(&provider, "token-B").await;

        for tick in 1..=5 {
            let (_, password) = provider.fetch(None).await.expect("cached token is live");
            assert_eq!(
                password,
                Some("token-B".to_string()),
                "tick {tick}: every tick inside the {FIXTURE_FANOUT_WINDOW:?} fan-out window \
                 must carry the credential — that window is what gets a newly rotated token \
                 onto EVERY pooled connection, not just the one that happened to observe it"
            );
        }

        cancel.cancel();
    }

    /// A rotation re-opens the gate: the first tick that sees new token bytes
    /// carries them, even though the previous window had closed.
    ///
    /// This is the whole reason the gate is safe to have. Suppressing ticks is
    /// only correct if a real rotation is never suppressed with them. Staged
    /// across two rotations (A → B, window spent, B → C), so the gate being
    /// re-opened is one a rotation had opened and time had closed.
    #[tokio::test]
    async fn fetch_periodic_tick_sends_again_when_the_sidecar_rotates_the_token() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_gated_provider(path.clone());
        await_cached_token(&provider, "token-A").await;
        let _ = handshake_fetch(&provider).await;
        write_token_file(&path, "token-B", 3600).await;
        await_cached_token(&provider, "token-B").await;
        let (_, opening) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            opening,
            Some("token-B".to_string()),
            "precondition: the first rotation opens the window"
        );
        spend_the_fanout_window().await;

        // Precondition: the gate is shut. Without this the assertion below
        // would also pass on a provider that never gated anything.
        let (_, password) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            password, None,
            "precondition: the unchanged token's window must have closed before the rotation, \
             or this test cannot distinguish 'the rotation re-opened the gate' from 'the gate \
             was never shut'"
        );

        // The sidecar rotates again. The next tick to observe the new bytes
        // must carry them — a suppressed rotation is a connection that gets
        // dropped at the old token's expiry instead of prolonged.
        write_token_file(&path, "token-C", 3600).await;
        await_cached_token(&provider, "token-C").await;
        let (_, rotated) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            rotated,
            Some("token-C".to_string()),
            "the first tick to see a rotated token must carry it, whatever the previous \
             window did"
        );

        cancel.cancel();
    }

    // ── First sighting opens no window (b.19i) ──────────────────────────────
    //
    // Every connection that exists when the provider first sees a token
    // authenticated with that same token at its handshake, so there is nothing
    // to fan out. Treating the first sighting as a rotation made every pooled
    // client re-AUTH 60–90 s after every startup — on a cluster, one
    // `_AuthAllCluster` broadcast per client, queued behind any in-flight read.
    //
    // These run on the wide [`FIXTURE_FANOUT_WINDOW`] where they assert "no
    // window", so a gate that wrongly opened one is still inside it when the
    // tick is asked — the assertion cannot pass by the window having lapsed.

    /// The startup shape: connections handshake first (the provider's first
    /// sighting), then the periodic ticks run with the token unchanged. None
    /// of those ticks may carry a password.
    #[tokio::test]
    async fn fetch_periodic_tick_after_the_first_handshake_sends_no_password() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);
        await_cached_token(&provider, "token-A").await;
        let (_, handshake) = handshake_fetch(&provider).await;
        assert_eq!(
            handshake,
            Some("token-A".to_string()),
            "the first handshake still carries the token: a first sighting opens no window, but \
             the handshake is never gated"
        );

        for tick in 1..=3 {
            let (username, password) = provider.fetch(None).await.expect("cached token is live");
            assert_eq!(
                username,
                Some(OBJECT_ID.to_string()),
                "tick {tick}: the username is still sent"
            );
            assert_eq!(
                password, None,
                "tick {tick}: the handshake was the provider's first sighting of token-A, and \
                 every connection authenticated with it there — a tick carrying it is the \
                 startup re-AUTH burst (every pooled client, 60–90 s after every pod start) \
                 that b.19i removes"
            );
        }

        cancel.cancel();
    }

    /// The same rule when the first `fetch()` to see the token is a periodic
    /// tick — no prior handshake through this provider. The rule does not
    /// depend on which path saw the token first.
    #[tokio::test]
    async fn fetch_first_periodic_tick_with_no_prior_handshake_sends_no_password() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);
        await_cached_token(&provider, "token-A").await;

        for tick in 1..=2 {
            let (username, password) = provider.fetch(None).await.expect("cached token is live");
            assert_eq!(username, Some(OBJECT_ID.to_string()));
            assert_eq!(
                password, None,
                "tick {tick}: a first sighting on the periodic path records the token and opens \
                 no window, exactly as one on the handshake path does"
            );
        }

        cancel.cancel();
    }

    /// After a first sighting, a changed token opens the window: the tick
    /// that first sees it carries it, and once the window has passed the
    /// unchanged token sends nothing again. Started from a tick's first
    /// sighting (the window-closes test above starts from a handshake's).
    #[tokio::test]
    async fn fetch_a_changed_token_after_a_first_sighting_opens_the_fanout_window() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_gated_provider(path.clone());
        await_cached_token(&provider, "token-A").await;
        let (_, first) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            first, None,
            "precondition: a first sighting opens no window"
        );

        write_token_file(&path, "token-B", 3600).await;
        await_cached_token(&provider, "token-B").await;
        let (_, opening) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            opening,
            Some("token-B".to_string()),
            "a token that replaced one the provider had seen is a rotation: the first tick to \
             see it opens the fan-out window and carries it. Without this, a first sighting \
             with no window would also swallow every real rotation"
        );

        spend_the_fanout_window().await;
        let (_, after) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            after, None,
            "the window the rotation opened closes on its own schedule"
        );

        cancel.cancel();
    }

    /// The handshake path is never gated.
    ///
    /// fred passes `Some(server)` from connection setup, and a connection with
    /// no password never authenticates — so gating there would not save a
    /// broadcast, it would fail the connection outright. Asserted in the state
    /// where the periodic path is provably suppressed, which is the only state
    /// where the two answers differ — here the provider's first sighting,
    /// which opens no window (b.19i).
    #[tokio::test]
    async fn fetch_handshake_carries_the_token_whatever_the_gate_says() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_gated_provider(path);
        await_cached_token(&provider, "token-A").await;

        let (_, gated) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            gated, None,
            "precondition: the periodic path must be suppressed here, or the handshake \
             assertion below proves nothing about the gate"
        );

        let server = handshake_server();
        let (username, password) = provider
            .fetch(Some(&server))
            .await
            .expect("a handshake with a live cached token must produce a credential");
        assert_eq!(
            username,
            Some(OBJECT_ID.to_string()),
            "the handshake username must still be the Object ID (SR-4.1)"
        );
        assert_eq!(
            password,
            Some("token-A".to_string()),
            "the handshake MUST carry the token whatever the rotation gate says: fred sends no \
             AUTH when the password is None, and a connection that never AUTHs never connects"
        );

        cancel.cancel();
    }

    /// The handshake does not just ignore the gate — it *feeds* it (b.7td
    /// round 2).
    ///
    /// [`TokenFileCredentialProvider::fetch`] calls `observe_and_should_reauth`
    /// unconditionally and only the *verdict* is ignored on the handshake path.
    /// That placement is load-bearing and it is the fragile kind: moving the
    /// call into the `else` branch — where a reader who sees the return value
    /// discarded would naturally put it — leaves every handshake invisible to
    /// the rotation epoch, and nothing else in the suite notices, because a
    /// handshake that carries the token still carries the token.
    ///
    /// What it costs is one whole extra fan-out per rotation picked up by a
    /// reconnect: the connection AUTHs with the new token during its handshake,
    /// the epoch still names the *old* one, and the next periodic tick therefore
    /// reads as a fresh rotation and re-`AUTH`s everything a second time — on a
    /// cluster, a second `_AuthAllCluster` broadcast, which is the shape b.7td
    /// exists to bound.
    ///
    /// The wait for the rotated token deliberately watches the token cache
    /// ([`await_cached_token`]), not `fetch()`: a `fetch(None)` there would
    /// itself be the first observation of token-B and would register the
    /// epoch, so the test would pass regardless of what the handshake does.
    /// The handshake has to be the first call to see the new bytes.
    ///
    /// What this does not tell apart: a handshake that registered token-B
    /// *with no window* would also leave the final tick silent. That the
    /// handshake opens the window is pinned by
    /// [`fetch_handshake_that_sees_a_changed_token_opens_the_fanout_window`].
    #[tokio::test]
    async fn fetch_handshake_registers_the_rotation_epoch_it_observed() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_gated_provider(path.clone());
        await_cached_token(&provider, "token-A").await;

        let (_, password) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            password, None,
            "precondition: the gate must be shut on token-A before the rotation (a first \
             sighting opens no window), or the final assertion cannot tell 'the handshake \
             opened a window that then closed' from 'a window was open all along'"
        );

        // The sidecar rotates. Observed through the token provider's cache, NOT
        // through `fetch()` — see the doc comment.
        write_token_file(&path, "token-B", 3600).await;
        await_cached_token(&provider, "token-B").await;

        // The handshake is the first call in the process to see token-B.
        let server = handshake_server();
        let (_, handshake_password) = provider
            .fetch(Some(&server))
            .await
            .expect("a handshake with a live cached token must produce a credential");
        assert_eq!(
            handshake_password,
            Some("token-B".to_string()),
            "the handshake must carry the rotated token — the ungated contract this test builds on"
        );

        spend_the_fanout_window().await;

        let (_, tick_password) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            tick_password, None,
            "the fan-out window for token-B must have been opened BY THE HANDSHAKE and must have \
             closed with it. A `Some(\"token-B\")` here means the handshake authenticated with a \
             token the rotation epoch never recorded, so this tick reads as a changed token \
             and opens a window of its own — one extra `_AuthAllCluster` fan-out per rotation \
             that a reconnect happened to pick up first, which is exactly the broadcast b.7td \
             gates"
        );

        cancel.cancel();
    }

    /// A handshake that sees a changed token opens the fan-out window (b.7td),
    /// not just records it: the next tick, inside the window, carries the
    /// token on to the connections that did not reconnect.
    ///
    /// The complement of the test above, which cannot see this: a handshake
    /// that recorded token-B as if it were a first sighting (no window) would
    /// leave that test's final tick silent too, while here it leaves this
    /// tick silent — every connection that did not happen to reconnect keeps
    /// the old token until it expires. On the wide [`FIXTURE_FANOUT_WINDOW`],
    /// so "inside the window" holds however slow the runner is.
    #[tokio::test]
    async fn fetch_handshake_that_sees_a_changed_token_opens_the_fanout_window() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_token_file_provider(path.clone(), WIDE_EXPIRY_SKEW);
        await_cached_token(&provider, "token-A").await;
        let _ = handshake_fetch(&provider).await;
        let (_, gated) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            gated, None,
            "precondition: no window is open on token-A (its first sighting)"
        );

        write_token_file(&path, "token-B", 3600).await;
        await_cached_token(&provider, "token-B").await;
        // The handshake is the first call to see token-B.
        let (_, handshake) = handshake_fetch(&provider).await;
        assert_eq!(handshake, Some("token-B".to_string()));

        let (_, tick) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            tick,
            Some("token-B".to_string()),
            "the handshake saw token-B replace token-A, which is a rotation: it must open the \
             fan-out window so the periodic ticks carry token-B to every other connection. A \
             `None` here means the handshake treated the changed token as a first sighting"
        );

        cancel.cancel();
    }

    /// Credential death is evaluated **before** the gate.
    ///
    /// The two answers are not interchangeable: `Ok((username, None))` means
    /// "nothing to send, the connection is fine", while `Err(Auth)` means "this
    /// process has no usable credential" — which trips the Valkey `PING`
    /// readiness probe (SDD §17.3, no smoothing). An implementation that
    /// consulted the gate first would answer a dead credential with the gate's
    /// silent shape and hide a total auth outage behind a suppressed tick.
    ///
    /// Driven from the *gated* state on purpose: with the window still open the
    /// two orderings are indistinguishable, because both would go on to look
    /// for a token. The gated state here is the provider's first sighting of
    /// token-A, which opens no window (b.19i).
    #[tokio::test]
    async fn fetch_reports_credential_death_rather_than_the_gates_silent_no_password() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, cancel) = build_gated_provider(path.clone());
        await_cached_token(&provider, "token-A").await;

        let (_, password) = provider.fetch(None).await.expect("cached token is live");
        assert_eq!(
            password, None,
            "precondition: the gate must be shut before the credential dies, or this test \
             cannot tell the two evaluation orders apart"
        );

        // The sidecar's next write is a token that is already dead. The cache
        // takes it (the previous-good token is retained only across *parse*
        // failures), so from here `cached_bearer_if_unexpired()` is empty.
        write_token_file(&path, "token-dead", -10).await;
        assert!(
            poll_until(POLL_PICKUP_BUDGET, || provider
                .provider
                .cached_token_expires_on()
                .is_some_and(|expires_on| expires_on <= Utc::now()))
            .await,
            "the polling task must cache the already-expired rotation within \
             {POLL_PICKUP_BUDGET:?}; until it does, the credential is still live and the \
             assertion below would be about the wrong state"
        );

        let outcome = provider.fetch(None).await;
        let err = match outcome {
            Err(err) => err,
            Ok((_, None)) => panic!(
                "an expired credential answered with the ROTATION GATE's shape — \
                 `Ok((Some(username), None))`. That is the gate being consulted before the \
                 token is checked: fred would treat it as 'nothing to re-AUTH, carry on' and \
                 the pod would keep reporting ready while holding no usable credential at all"
            ),
            Ok((_, Some(_))) => {
                panic!("an expired token must never be sent as a credential (SR-4.1)")
            }
        };
        assert!(
            matches!(err.kind(), FredErrorKind::Auth),
            "credential death must surface as ErrorKind::Auth, got {:?}",
            err.kind()
        );

        cancel.cancel();
    }

    /// The constructor's cadence / window pairing must actually fire.
    ///
    /// A window shorter than `base_interval + jitter_bound` fails silently and
    /// slowly in production: a client whose tick fired microseconds before a
    /// rotation was observed never ticks again inside the window, so it never
    /// receives the new credential and is dropped at expiry instead of
    /// prolonged — indistinguishable from ordinary connection churn. The
    /// shipped constants are additionally guarded by a `const` assertion in
    /// `storage::client`; this covers the constructor itself, which is what any
    /// future second wiring site would go through.
    ///
    /// `#[cfg(debug_assertions)]` because the check is a `debug_assert!`: a
    /// release build has deliberately nothing to observe.
    #[cfg(debug_assertions)]
    #[tokio::test]
    #[should_panic(expected = "auth_fanout_window")]
    async fn new_rejects_a_fanout_window_shorter_than_the_tick_band() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        let cancel = CancellationToken::new();
        let token_provider = Arc::new(ServiceTokenProvider::new_with_refresh_lead(
            path,
            TEST_POLL_INTERVAL,
            WIDE_EXPIRY_SKEW,
            fresh_metrics(),
            cancel.clone(),
        ));

        let _ = TokenFileCredentialProvider::new(
            token_provider,
            OBJECT_ID.to_string(),
            FIXTURE_BASE_INTERVAL,
            FIXTURE_JITTER_BOUND,
            // One second short of `base + jitter` — the smallest mis-sizing
            // that breaks the fan-out.
            FIXTURE_BASE_INTERVAL + FIXTURE_JITTER_BOUND - Duration::from_secs(1),
        );
    }

    // ── Late rotation: Err(Auth) → Ok sub-second, not on the steady poll ────

    /// b.qam composition test, re-shaped for b.d6b: pins the *production
    /// wiring* — this credential provider on top of a `ServiceTokenProvider`
    /// whose stale-token fast-poll phase is armed — against a Valkey-shaped
    /// refresh lead.
    ///
    /// **What moved.** Pre-b.d6b `fetch()` failed `AUTH` for the whole
    /// pre-expiry lead, so the outage this test measured *started at lead
    /// entry*. That premise is gone: an in-lead token now authenticates
    /// (`fetch_serves_cached_token_inside_the_refresh_lead`). The outage the
    /// composition can still suffer starts at the cached token's **actual
    /// expiry**, and that is where this test now drives it from — which makes
    /// the same test simultaneously the regression test for b.d6b's *bounded
    /// post-expiry fast-poll window*, the mechanism that keeps the 250 ms
    /// cadence running through exactly that outage instead of reverting to the
    /// steady poll at the worst possible moment.
    ///
    /// The failure guarded is unchanged in shape and worse in severity: with
    /// no usable credential, every fred `AUTH` / `HELLO` fails and the Valkey
    /// `PING` readiness probe goes `NOT_SERVING` (SDD §17.3, no smoothing) for
    /// as long as it takes the token provider to notice the replacement file.
    /// On the steady cadence alone that is up to a full poll interval; on the
    /// post-expiry fast cadence it is one `FAST_POLL_RETRY_INTERVAL`.
    ///
    /// Structure, and why each number is what it is:
    ///
    /// - [`LONG_POLL_INTERVAL`] (60 s), not the module's 20 ms
    ///   [`TEST_POLL_INTERVAL`], is the discriminator. A recovery observed
    ///   inside a 1 s budget cannot have come from a steady tick. It is now a
    ///   *strictly* clean discriminator: across the post-expiry window the
    ///   steady arm is guard-disabled outright, so the fast arm is the only
    ///   thing that can read the file.
    /// - The lead and the token lifetime are scaled down from the production
    ///   ≥ 3 min / ~24 h to 1 s / 1.5 s, keeping the test on the real clock
    ///   (the phase is `Utc::now()`-driven, so a paused clock cannot advance a
    ///   token toward expiry) and near 1.5 s of wall time. The lead no longer
    ///   moves the AUTH-failure onset; it is still scaled because it names the
    ///   proactive re-read window and keeps the fixture production-shaped.
    /// - The rotation is written *at the moment the AUTH failure is first
    ///   observed*, i.e. microseconds into a post-expiry window that is
    ///   [`POST_EXPIRY_FAST_WINDOW`] (20 s) wide. That margin is ~80× the
    ///   pre-b.d6b fixture's, so the elaborate "fixture aged out" machinery
    ///   this test used to carry collapses to the single check below.
    #[tokio::test]
    async fn fetch_recovers_sub_second_from_a_late_rotation() {
        /// Scaled-down stand-in for the ≥ 3 min production refresh lead. With
        /// [`LONG_POLL_INTERVAL`] as the poll interval the phase's entry lead
        /// (`refresh_lead + poll_interval`) is dominated by the 60 s term, so
        /// the phase is armed from the first loop head whatever this is; the
        /// value exists to keep the fixture's shape, not to time anything.
        const SCALED_LEAD: chrono::Duration = chrono::Duration::milliseconds(1000);
        /// Scaled-down stand-in for the ~24 h token lifetime. `fetch()`
        /// succeeds for the whole of it, then fails AUTH at expiry.
        const SCALED_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(1500);
        /// Budget for the first successful `fetch()`. A ceiling on the t≈0
        /// read, spent only on a stall — and strictly inside
        /// [`SCALED_LIFETIME`], so a run that blows it has not silently turned
        /// the precondition into an expiry observation.
        const LOAD_BUDGET: Duration = Duration::from_millis(1000);
        /// Budget for the AUTH failure to appear. Expected at
        /// [`SCALED_LIFETIME`] (~1.5 s); generous headroom above it because
        /// overshooting costs nothing now — the post-expiry window is 20 s.
        const EXPIRY_ONSET_BUDGET: Duration = Duration::from_millis(6000);
        /// Budget for the recovery. Four [`FAST_POLL_RETRY_INTERVAL`]s, and 60×
        /// below [`LONG_POLL_INTERVAL`] — that ratio is the discriminator.
        const RECOVERY_BUDGET: Duration =
            Duration::from_millis(4 * FAST_POLL_RETRY_INTERVAL.as_millis() as u64);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("late-rotation.json");
        write_token_file_in(&path, "token-A", SCALED_LIFETIME).await;

        let (provider, cancel) =
            build_token_file_provider_with_poll(path.clone(), LONG_POLL_INTERVAL, SCALED_LEAD);
        // Every probe below is a handshake (`fetch(Some(server))`): the
        // question is whether a credential exists, which the handshake always
        // answers, while a periodic tick would also consult the rotation gate
        // (and carries no password on a first sighting, b.19i). `Err(Auth)` is
        // evaluated before the gate on both paths.
        let server = handshake_server();

        // Precondition: a live token authenticates normally.
        let started = std::time::Instant::now();
        loop {
            match provider.fetch(Some(&server)).await {
                Ok((_, password)) => {
                    assert_eq!(
                        password,
                        Some("token-A".to_string()),
                        "the initial credential must be the token the sidecar wrote"
                    );
                    break;
                }
                Err(e) => assert!(
                    started.elapsed() < LOAD_BUDGET,
                    "the token file was present at construction, so fetch() must succeed once \
                     the t≈0 read lands; still failing after {:?} with {:?}",
                    started.elapsed(),
                    e.kind()
                ),
            }
            tokio::time::sleep(PROBE_INTERVAL).await;
        }

        // The sidecar is late: the cached token runs all the way out and the
        // credential dies. Every AUTH now fails. This is the total-outage state
        // the post-expiry fast-poll window exists to shorten.
        let started = std::time::Instant::now();
        loop {
            match provider.fetch(Some(&server)).await {
                Err(e) => {
                    assert!(
                        matches!(e.kind(), FredErrorKind::Auth),
                        "an expired token must surface as ErrorKind::Auth, got {:?}",
                        e.kind()
                    );
                    break;
                }
                Ok(_) => assert!(
                    started.elapsed() < EXPIRY_ONSET_BUDGET,
                    "fetch() must start failing AUTH once the cached token passes its \
                     {SCALED_LIFETIME:?} expiry; it was still succeeding after {:?}",
                    started.elapsed()
                ),
            }
            tokio::time::sleep(PROBE_INTERVAL).await;
        }

        // The rotation finally lands — inside the bounded post-expiry window,
        // where the fast cadence is still running.
        let auth_failed_at = std::time::Instant::now();
        write_token_file(&path, "token-B", 3600).await;

        let started = std::time::Instant::now();
        let recovered = loop {
            if let Ok((username, password)) = provider.fetch(Some(&server)).await {
                break Some((started.elapsed(), username, password));
            }
            if started.elapsed() >= RECOVERY_BUDGET {
                break None;
            }
            tokio::time::sleep(PROBE_INTERVAL).await;
        };

        let (elapsed, username, password) = recovered.unwrap_or_else(|| {
            // Fixture-health check, in the failure path only. The post-expiry
            // fast cadence is bounded by POST_EXPIRY_FAST_WINDOW measured from
            // the provider's retirement, which is at or after the moment
            // `fetch()` first failed — so elapsed-since-that-moment is a
            // conservative lower bound on how much window is left. A run that
            // stalls for 20 s here has learned nothing about the fix either
            // way, and must not report as a regression.
            assert!(
                auth_failed_at.elapsed() < POST_EXPIRY_FAST_WINDOW,
                "FIXTURE STALLED — not a b.d6b regression. {:?} elapsed between the first AUTH \
                 failure and the end of the recovery probe, at or beyond the \
                 {POST_EXPIRY_FAST_WINDOW:?} post-expiry fast-poll window, so the provider had \
                 legitimately reverted to the {LONG_POLL_INTERVAL:?} steady cadence before token B \
                 was ever looked for. Re-run; if it recurs the runner is too loaded for this test.",
                auth_failed_at.elapsed()
            );
            panic!(
                "a late rotation must restore AUTH within {RECOVERY_BUDGET:?} via the token \
                 provider's post-expiry fast-poll window; fetch() was still failing, which means \
                 the credential stays broken until the {LONG_POLL_INTERVAL:?} steady poll — the \
                 total-credential-outage case b.d6b exists to shorten. The window was still open \
                 ({:?} of {POST_EXPIRY_FAST_WINDOW:?} elapsed), so this is the real regression.",
                auth_failed_at.elapsed()
            )
        });
        assert_eq!(
            username,
            Some(OBJECT_ID.to_string()),
            "username stays the Object ID across the recovery (recovered in {elapsed:?})"
        );
        assert_eq!(
            password,
            Some("token-B".to_string()),
            "the restored credential must be the rotated token, not the stale one"
        );

        cancel.cancel();
    }

    // ── refresh_interval(): every draw stays within bounds (SR-4.2) ─────────

    #[tokio::test]
    async fn refresh_interval_is_always_within_bounds() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;
        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);

        // base_interval = 30 s, jitter_bound = 10 s. Each call draws fresh
        // entropy (there is no seed), so assert the in-band invariant holds for
        // EVERY draw across a large sample rather than any determinism claim.
        let base = Duration::from_secs(30);
        let ceiling = Duration::from_secs(40); // base + jitter_bound
        for _ in 0..200 {
            let interval = provider
                .refresh_interval()
                .expect("refresh_interval is Some");
            assert!(
                interval >= base && interval <= ceiling,
                "every drawn interval must stay within [{base:?}, {ceiling:?}], got {interval:?}"
            );
        }

        cancel.cancel();
    }

    // ── refresh_interval(): per-client stagger — no lockstep (SR-4.2) ───────

    #[tokio::test]
    async fn refresh_interval_spreads_across_repeated_calls_no_lockstep() {
        // SR-4.2 per-client stagger: `refresh_interval()` draws fresh entropy on
        // every call, so repeated calls on a SINGLE provider must spread across
        // the jitter band rather than returning one locked-in value. This is the
        // guarantee that prevents the `_AuthAllCluster` thundering herd — fred
        // reads a client's `refresh_interval()` once per client (each client
        // shares this provider via `Arc`), so a fresh per-call draw is precisely
        // the mechanism that staggers clients across the fleet. A jitter-free /
        // seed-locked interval would re-AUTH every client in lockstep.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);

        let base = Duration::from_secs(30);
        let ceiling = Duration::from_secs(40); // base + jitter_bound (10 s = 10_000 ms)

        let mut samples = std::collections::HashSet::new();
        for _ in 0..200 {
            let interval = provider
                .refresh_interval()
                .expect("refresh_interval is Some");
            // Every staggered draw still sits inside the shared jitter band.
            assert!(
                interval >= base && interval <= ceiling,
                "staggered interval {interval:?} must stay within [{base:?}, {ceiling:?}]"
            );
            samples.insert(interval);
        }

        // Flake-proof: with a 10 s (10_000 ms) bound over 200 uniform draws the
        // probability of every sample being identical is astronomically small
        // (~(1/10_001)^199), so asserting "more than one distinct value" is a
        // safe proxy for "genuine per-call jitter, no lockstep".
        assert!(
            samples.len() > 1,
            "repeated refresh_interval() calls must produce a spread of values \
             (per-call jitter, no lockstep); saw only {} distinct value(s)",
            samples.len()
        );

        cancel.cancel();
    }

    // ── refresh_interval(): ≥ 3 min headroom before token lifetime (SR-4.2) ──

    #[tokio::test]
    async fn refresh_interval_ceiling_leaves_three_min_headroom() {
        // The largest possible re-AUTH interval is `base_interval + jitter_bound`
        // (max jitter), and it must leave ≥ 3 min of headroom ahead of the
        // configured token lifetime. The assertion is unchanged by b.d6b; its
        // REASON is not, so do not read the old one back in from the sibling
        // tests' shape.
        //
        // It is no longer "a maximally-staggered client could next re-AUTH
        // inside the lead window, where the accessor withholds and `fetch()`
        // returns `Err(Auth)`" — that is simply false now:
        // `fetch()` reads `cached_bearer_if_unexpired()`, which serves a cached
        // token for its whole real lifetime, lead or no lead.
        //
        // The live reason is about connection *longevity*. fred re-AUTHs on
        // this cadence, and a re-AUTH only carries a connection past the
        // current token's death if it lands while a FRESH token is already
        // cached. The replacement is only reliably on disk (and re-read by the
        // polling task, which is what the ≥ 3 min lead drives) once the lead
        // has opened. So the cadence ceiling must fit at least one whole
        // re-AUTH inside that final ≥ 3 min stretch; a ceiling with less
        // headroom can spend a client's last pre-expiry re-AUTH on the
        // about-to-die token, after which the connection dies at expiry with no
        // scheduled re-AUTH left to pick the replacement up. The authoritative
        // statement of that requirement lives on
        // `storage::client::VALKEY_TOKEN_REAUTH_BASE_INTERVAL`, which also
        // records the (milder, post-b.d6b) consequence of missing it: reconnect
        // churn rather than a hard AUTH failure.
        //
        // We read the provider's ACTUAL configured band (private fields,
        // same-module access) rather than re-declared literals, so the assert
        // covers the real construction.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);

        // Ceiling of the re-AUTH cadence, from the provider's real config.
        let ceiling = provider.base_interval + provider.jitter_bound;

        // Realistic access-token lifetime (~24 h; SDD SR-4.1 / SR-4.2).
        let token_lifetime = Duration::from_secs(24 * 60 * 60);

        let headroom = token_lifetime
            .checked_sub(ceiling)
            .expect("max re-AUTH interval must be shorter than the token lifetime");
        assert!(
            headroom >= THREE_MIN,
            "base + max_jitter ({ceiling:?}) must leave ≥ 3 min before the token \
             lifetime ({token_lifetime:?}); headroom was {headroom:?}"
        );

        cancel.cancel();
    }

    // ── Debug never prints the Object-ID username or token bytes (SR-6.2) ────

    #[tokio::test]
    async fn debug_redacts_object_id_and_token_state() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "super-secret-token", 3600).await;
        let (provider, cancel) = build_token_file_provider(path, WIDE_EXPIRY_SKEW);
        // Establish the premise before rendering: the assertions below are
        // about a provider that is *holding* a token — and, through a
        // `fetch()` that saw it, a rotation epoch fingerprinting it — and
        // `Debug` redacting an empty one proves nothing.
        await_cached_token(&provider, "super-secret-token").await;
        let (_, password) = handshake_fetch(&provider).await;
        assert_eq!(password.as_deref(), Some("super-secret-token"));

        let rendered = format!("{provider:?}");
        assert!(
            !rendered.contains(OBJECT_ID),
            "Debug must not print the Object-ID username, got: {rendered}"
        );
        assert!(
            !rendered.contains("super-secret-token"),
            "Debug must not print token bytes, got: {rendered}"
        );

        // b.7td: the rotation epoch holds a digest OF the live token. It is
        // credential-derived state, so it is held to the same rule as the token
        // itself — a `#[derive(Debug)]` regression on the struct would print it
        // through the `RotationEpoch` field and put a stable, comparable
        // identifier for the current credential into every log line that
        // renders the provider.
        let fingerprint = token_fingerprint("super-secret-token");
        assert!(
            !rendered.contains(&fingerprint.to_string())
                && !rendered.contains(&format!("{fingerprint:x}")),
            "Debug must not print the rotation fingerprint (decimal {fingerprint} / hex \
             {fingerprint:x}), got: {rendered}"
        );
        assert!(
            !rendered.contains("rotation"),
            "the `rotation` field must be omitted from Debug entirely, not rendered with a \
             redaction placeholder — the placeholder is what a later edit turns back into a \
             value. Got: {rendered}"
        );

        cancel.cancel();
    }

    // ── The rotation fingerprint never reaches the logs (b.7td, SR-6.2) ──────

    /// The gate's own state is credential-derived and must obey the same
    /// bearer-bytes-never-logged invariant as the token.
    ///
    /// A fingerprint in the logs is weaker than a leaked token but not
    /// harmless: it is a stable, comparable identifier for the live credential,
    /// so a log aggregator could tell which pods share a token and exactly when
    /// each rotation landed. The gate is also the one place tempted to log —
    /// "suppressed a re-AUTH, fingerprint unchanged" is the obvious debug line
    /// to add — which is why the suppression path is deliberately silent and
    /// why this test drives every branch of the gate rather than just one.
    #[tokio::test]
    async fn no_rotation_fingerprint_in_logs_across_the_gates_branches() {
        const TOKEN_A: &str = "tkn-b7td-rotation-canary-alpha";
        const TOKEN_B: &str = "tkn-b7td-rotation-canary-bravo";

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, TOKEN_A, 3600).await;

        // Installed before construction so the polling task — which runs on
        // this same thread under the default current-thread runtime — inherits
        // the dispatcher too. Same reasoning as
        // `no_token_bytes_in_logs_across_fetch_refresh_and_parse_error`.
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        let dispatch_guard = tracing::subscriber::set_default(subscriber);

        let (provider, cancel) = build_gated_provider(path.clone());
        let tick_password = |outcome: Result<(Option<String>, Option<String>), FredError>| {
            outcome.expect("cached token is live").1
        };

        // Branch 1 — first sighting (records, opens no window, sends nothing).
        await_cached_token(&provider, TOKEN_A).await;
        assert_eq!(tick_password(provider.fetch(None).await), None);
        // Branch 2 — handshake (sends unconditionally).
        let server = handshake_server();
        assert_eq!(
            tick_password(provider.fetch(Some(&server)).await).as_deref(),
            Some(TOKEN_A)
        );
        // Branch 3 — rotation observed (opens the window and sends).
        write_token_file(&path, TOKEN_B, 3600).await;
        await_cached_token(&provider, TOKEN_B).await;
        assert_eq!(
            tick_password(provider.fetch(None).await).as_deref(),
            Some(TOKEN_B)
        );
        // Branch 4 — inside the window (sends). Not asserted: on the 60 ms
        // window a stalled runner may already be past it, and either answer
        // still drives a gate branch through the logs.
        let _ = provider.fetch(None).await;
        // Branch 5 — window closed, token unchanged (suppressed).
        spend_the_fanout_window().await;
        assert_eq!(tick_password(provider.fetch(None).await), None);

        // Non-vacuity. Every branch above is deliberately silent, so an
        // absence-only assertion over an empty buffer would pass on a capture
        // pipeline that was never wired. The token provider's parse-error WARN
        // is a log site that reliably fires, so drive it and wait on the event
        // itself rather than on a poll interval. It also leaves the cache
        // holding the previous-good token, which is a further gate branch.
        let malformed =
            format!("{{\"access_token\":\"{TOKEN_B}\",\"expires_on\":\"not-rfc3339\"}} GARBAGE");
        tokio::fs::write(&path, malformed)
            .await
            .expect("write malformed fixture");
        assert!(
            poll_until(POLL_PICKUP_BUDGET, || captured
                .lock()
                .unwrap()
                .windows(4)
                .any(|w| w == b"WARN"))
            .await,
            "the malformed rotation must reach the token provider's parse-error WARN site \
             within {POLL_PICKUP_BUDGET:?}; without a live capture the no-leak assertions \
             below would pass vacuously"
        );
        let _ = provider.fetch(None).await;

        cancel.cancel();
        drop(dispatch_guard);

        let guard = captured.lock().expect("capture mutex is never poisoned");
        let output = String::from_utf8_lossy(&guard);
        assert!(
            output.contains("WARN"),
            "expected the parse-error WARN in the captured output; an empty buffer would make \
             every assertion below vacuous. Got: {output}"
        );

        for (label, token) in [("outgoing", TOKEN_A), ("rotated", TOKEN_B)] {
            let fingerprint = token_fingerprint(token);
            for rendering in [format!("{fingerprint}"), format!("{fingerprint:x}")] {
                assert!(
                    !output.contains(&rendering),
                    "the {label} token's rotation fingerprint ({rendering}) must never be \
                     logged — it is a stable identifier for the live credential (b.7td / \
                     SR-6.2). Captured output: {output}"
                );
            }
            // The token bytes themselves are covered at 8-byte granularity by
            // the sibling test; this is the cheap non-vacuity check that the
            // gate's own branches did not open a second leak of the plaintext.
            assert!(
                !output.contains(token),
                "the {label} token bytes must never be logged. Captured output: {output}"
            );
        }
    }

    // ── No token bytes in logs across fetch / refresh / parse-error sites ────
    //    (bearer-bytes-never-logged, SR-4.1 extends SR-6.2 to Valkey auth)

    #[tokio::test]
    async fn no_token_bytes_in_logs_across_fetch_refresh_and_parse_error() {
        // Hand-picked canary token — any contiguous 8-byte slice of this string
        // MUST NOT appear in captured log output across the access-token-file provider's
        // fetch / refresh log sites OR the underlying token provider's
        // parse-error WARN site. Chosen so it is improbable to collide with
        // anything tokio / tracing emits (mirrors the `SECRET_TOKEN` canary in
        // `service_token.rs::malformed_json_logs_warn_and_increments_parse_counter`).
        const SECRET_TOKEN: &str = "tkn-tokenfile-cafef00d-8badf00d-canary";

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");

        // Start from a VALID token file carrying the canary so the fetch()
        // success path actually handles the real token bytes (the password it
        // returns IS the canary) before we ever drive the parse-error path.
        write_token_file(&path, SECRET_TOKEN, 3600).await;

        // Install the capturing subscriber BEFORE constructing the provider so
        // the spawned polling task is captured too. The default `#[tokio::test]`
        // runtime is `current_thread`, so the spawned task runs on the same OS
        // thread as the test body and inherits this thread-local default
        // dispatcher — same reasoning as ~line 799 of service_token.rs.
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::TRACE)
            .finish();
        let _dispatch_guard = tracing::subscriber::set_default(subscriber);

        let (provider, cancel) = build_token_file_provider(path.clone(), WIDE_EXPIRY_SKEW);
        await_cached_token(&provider, SECRET_TOKEN).await;

        // fetch() success path: the returned password IS the canary token —
        // this is the credential-bearing call whose bytes must never reach the
        // logs. Asked through the handshake, which always carries it; the
        // periodic path (a first sighting there carries no password, b.19i)
        // and refresh_interval() are on the same credential-bearing struct, so
        // exercise them too.
        let (_, password) = handshake_fetch(&provider).await;
        assert_eq!(
            password,
            Some(SECRET_TOKEN.to_string()),
            "fetch() must return the loaded canary token as the password"
        );
        let _ = provider.fetch(None).await;
        let _ = provider.refresh_interval();

        // Rotate the sidecar file to MALFORMED JSON that still embeds the
        // canary. The polling task's next tick hits the parse-error WARN site —
        // if the impl ever logged the file bytes, the canary would surface
        // there. (Byte-shape identical to the service_token.rs malformed
        // fixture: valid `access_token` key, garbage tail breaks JSON parse.)
        let malformed = format!(
            "{{\"access_token\":\"{SECRET_TOKEN}\",\"expires_on\":\"not-rfc3339\"}} <-- GARBAGE TAIL"
        );
        tokio::fs::write(&path, malformed)
            .await
            .expect("write malformed fixture");
        // Wait on the WARN itself, not on a poll interval: the buffer being
        // non-empty is the precondition the leak assertion at the end depends
        // on, and a fixed sleep on a loaded runner does not guarantee the
        // parse-error read has happened.
        assert!(
            poll_until(POLL_PICKUP_BUDGET, || captured
                .lock()
                .unwrap()
                .windows(4)
                .any(|w| w == b"WARN"))
            .await,
            "the malformed rotation must reach the token provider's parse-error WARN site within \
             {POLL_PICKUP_BUDGET:?}; without that read the no-leak assertion below would pass \
             vacuously"
        );

        // Drive fetch() again across the parse-error state. The previous-good
        // cache is retained on a parse failure, so this still exercises the
        // credential-return path; we only care that no bytes leak, not the
        // return value here.
        let _ = provider.fetch(None).await;

        cancel.cancel();

        // Stop capturing before asserting on the buffer.
        drop(_dispatch_guard);

        let guard = captured.lock().unwrap();
        let output = String::from_utf8_lossy(&guard);

        // A WARN must have fired (the parse-error log site). This confirms the
        // capture pipeline is actually live — an empty buffer would make the
        // leak assertion below vacuously pass.
        assert!(
            output.contains("WARN"),
            "expected at least one WARN event (parse-error log site) in captured output, got: {output}"
        );

        // Bearer-bytes-never-logged invariant — pinned at the 8-byte slice
        // granularity. Slide an 8-byte window over the canary and assert that
        // NO contiguous 8-byte slice of the token bytes appears in the captured
        // output.
        let canary_bytes = SECRET_TOKEN.as_bytes();
        assert!(
            canary_bytes.len() >= 8,
            "canary must be at least 8 bytes wide for the slice-search to be meaningful"
        );
        for window in canary_bytes.windows(8) {
            // The canary is ASCII, so any 8-byte window is valid UTF-8.
            let slice = std::str::from_utf8(window).expect("ASCII canary");
            assert!(
                !output.contains(slice),
                "captured log output must not contain any 8-byte slice of the canary token \
                 (slice={slice:?}, full canary={SECRET_TOKEN:?}, output={output})"
            );
        }
    }

    // ── Identity self-check: valkey_identity_verdict (b.cbe) ─────────────────
    //
    // `valkey_identity_verdict` is pure and total, so every case below is a
    // plain `#[test]` with no runtime, no file and no clock. It is the whole
    // discriminating step of the b.cbe diagnostic: a managed backing store
    // returns ONE opaque `WRONGPASS invalid username-password pair` for every
    // cause, and this function is where the two halves the server never sees —
    // our configured AUTH username and the claims of the token we sent — are
    // compared. A wrong verdict is worse than none: it sends an operator at
    // the wrong cause with the authority of a diagnosis.
    //
    // The `guidance()` assertions here check that each verdict's sentence
    // carries the *values an operator must act on* (the `oid` to set, the
    // wrong value currently configured) and names the env var — not its exact
    // wording. Verdict `label()`s are pinned as a set because they are the
    // structured `identity_verdict` log field dashboards and alert rules key
    // off, i.e. an interface rather than a message.

    /// A client-id claim distinct from [`OBJECT_ID`] — the value an operator
    /// wrongly configures as the AUTH username in the cause-(1) case.
    const CLIENT_ID: &str = "99999999-8888-7777-6666-555555555555";

    /// Build a `TokenClaims` carrying only the two identity claims the verdict
    /// judges. The reported-but-unjudged claims (`aud`, `tid`, `iss`, `exp`)
    /// are left at their defaults: including them would suggest they
    /// participate in the decision, and they deliberately do not.
    fn identity_claims(oid: Option<&str>, client_id_claim: Option<&str>) -> TokenClaims {
        TokenClaims {
            oid: oid.map(str::to_string),
            client_id_claim: client_id_claim.map(str::to_string),
            ..TokenClaims::default()
        }
    }

    #[test]
    fn valkey_identity_verdict_maps_each_claims_and_username_pairing_to_its_verdict() {
        // The verdict matrix, folded from eight near-identical single-assert
        // bodies. Every row is (claims, configured username, expected verdict,
        // rationale); the rationale is what a red run reads, and it is the only
        // thing that differed between the original bodies.
        //
        // The two ORDER-SENSITIVE cases stay standalone below — they assert the
        // evaluation order and the comparison semantics rather than a mapping,
        // and folding them in would bury the property in a table row.
        const UNRELATED: &str = "00000000-0000-0000-0000-000000000000";

        let cases: [(Option<TokenClaims>, &str, IdentityVerdict, &str); 8] = [
            (
                None,
                OBJECT_ID,
                IdentityVerdict::NotAJwt,
                "no cached claims — what an opaque token, a SigV4 credential and an unpopulated \
                 cache all look like; legitimate states, not faults",
            ),
            (
                Some(identity_claims(None, None)),
                OBJECT_ID,
                IdentityVerdict::NoIdentityClaim,
                "the token decoded but carries neither identity claim, so nothing can be \
                 cross-checked",
            ),
            (
                Some(identity_claims(Some(OBJECT_ID), Some(CLIENT_ID))),
                OBJECT_ID,
                IdentityVerdict::MatchesOid,
                "the healthy binding: the configured username IS the token's Object ID",
            ),
            (
                Some(identity_claims(Some(OBJECT_ID), Some(CLIENT_ID))),
                CLIENT_ID,
                IdentityVerdict::MatchesClientIdClaim {
                    oid: Some(OBJECT_ID.to_string()),
                    client_id_claim: CLIENT_ID.to_string(),
                },
                "the exact-cause case (client ID configured where the Object ID was required); \
                 the verdict must carry BOTH values — the wrong one to identify it and the right \
                 one to substitute",
            ),
            (
                Some(identity_claims(None, Some(CLIENT_ID))),
                CLIENT_ID,
                IdentityVerdict::MatchesClientIdClaim {
                    oid: None,
                    client_id_claim: CLIENT_ID.to_string(),
                },
                "same cause on a token carrying no `oid`: the verdict still fires, and carries \
                 `None` so the guidance can say there is no Object ID to read",
            ),
            (
                Some(identity_claims(Some(OBJECT_ID), Some(CLIENT_ID))),
                UNRELATED,
                IdentityVerdict::MatchesNeither {
                    oid: Some(OBJECT_ID.to_string()),
                    client_id_claim: Some(CLIENT_ID.to_string()),
                },
                "the username matches neither claim — federation misbinding or a stale username",
            ),
            (
                Some(identity_claims(Some(OBJECT_ID), None)),
                CLIENT_ID,
                IdentityVerdict::MatchesNeither {
                    oid: Some(OBJECT_ID.to_string()),
                    client_id_claim: None,
                },
                "NOT `NoIdentityClaim`: one claim WAS readable and it disagreed, which is a \
                 finding, whereas `NoIdentityClaim` says nothing could be checked",
            ),
            (
                Some(identity_claims(Some(OBJECT_ID), Some(CLIENT_ID))),
                "",
                IdentityVerdict::MatchesNeither {
                    oid: Some(OBJECT_ID.to_string()),
                    client_id_claim: Some(CLIENT_ID.to_string()),
                },
                "the `render_hint` path passes `\"\"` when `valkey_token_username` is `None`; an \
                 empty username must compare equal to no claim (which is why \
                 `decode_token_claims` drops empty claims), and \"set it to this `oid`\" is the \
                 right advice for an unset variable",
            ),
        ];

        for (claims, username, expected, rationale) in cases {
            assert_eq!(
                valkey_identity_verdict(claims.as_ref(), username),
                expected,
                "wrong verdict — {rationale}"
            );
        }
    }

    #[test]
    fn valkey_identity_verdict_oid_equal_to_client_id_prefers_the_healthy_verdict() {
        // Evaluation order, pinned: a token whose `oid` and `appid` happen to
        // be the same string must report the healthy match rather than
        // accusing the operator of the client-id mistake.
        let claims = identity_claims(Some(OBJECT_ID), Some(OBJECT_ID));

        assert_eq!(
            valkey_identity_verdict(Some(&claims), OBJECT_ID),
            IdentityVerdict::MatchesOid
        );
    }

    #[test]
    fn valkey_identity_verdict_comparison_is_case_sensitive() {
        // Pins the implementation's actual comparison (`&str` equality, no
        // case folding) rather than an assumption about it. Real directory
        // Object IDs are canonically lower-case GUIDs and the sidecar mints
        // them, so an upper-cased `LIVE_EDIT_VALKEY_TOKEN_USERNAME` is an
        // operator typo — and reporting it as "matches neither" is the correct
        // and useful answer, because the backing store's own comparison is the
        // one that decides and this service cannot assume it folds either.
        // `OBJECT_ID` is all-numeric, so it cannot express a case difference —
        // this fixture carries hex letters on purpose.
        const MIXED_CASE_OID: &str = "abcdef01-2345-6789-abcd-ef0123456789";
        let upper = MIXED_CASE_OID.to_uppercase();
        assert_ne!(
            upper, MIXED_CASE_OID,
            "fixture must actually differ in case"
        );
        let claims = identity_claims(Some(MIXED_CASE_OID), None);

        assert_eq!(
            valkey_identity_verdict(Some(&claims), &upper),
            IdentityVerdict::MatchesNeither {
                oid: Some(MIXED_CASE_OID.to_string()),
                client_id_claim: None,
            }
        );
    }

    #[test]
    fn identity_verdict_labels_are_distinct_and_claim_free() {
        // `label()` is the structured `identity_verdict` log field, so it is an
        // interface: dashboards and alert rules match on these strings, and two
        // verdicts sharing one label would make a broken deployment
        // indistinguishable from a healthy one on a graph.
        let verdicts = [
            IdentityVerdict::NotAJwt,
            IdentityVerdict::NoIdentityClaim,
            IdentityVerdict::MatchesOid,
            IdentityVerdict::MatchesClientIdClaim {
                oid: Some(OBJECT_ID.to_string()),
                client_id_claim: CLIENT_ID.to_string(),
            },
            IdentityVerdict::MatchesNeither {
                oid: Some(OBJECT_ID.to_string()),
                client_id_claim: Some(CLIENT_ID.to_string()),
            },
        ];

        let labels: Vec<&str> = verdicts.iter().map(IdentityVerdict::label).collect();
        let mut unique = labels.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            labels.len(),
            "every verdict needs its own label; got {labels:?}"
        );

        for (verdict, label) in verdicts.iter().zip(&labels) {
            // A label is a fixed dimension value, so it must never carry a
            // per-token claim — that would make the field unbounded-cardinality
            // and would put identity values somewhere the redaction review
            // never looks.
            assert!(
                !label.contains(OBJECT_ID) && !label.contains(CLIENT_ID),
                "{verdict:?} label {label:?} must not embed a claim value"
            );
        }
    }

    #[test]
    fn guidance_for_client_id_match_names_the_oid_to_substitute() {
        let guidance = IdentityVerdict::MatchesClientIdClaim {
            oid: Some(OBJECT_ID.to_string()),
            client_id_claim: CLIENT_ID.to_string(),
        }
        .guidance();

        assert!(
            guidance.contains(OBJECT_ID),
            "the operator's whole action is substituting the `oid`, so it must be printed: \
             {guidance}"
        );
        assert!(
            guidance.contains(CLIENT_ID),
            "the wrongly-configured value must be named so the operator can recognise it: \
             {guidance}"
        );
        assert!(
            guidance.contains("LIVE_EDIT_VALKEY_TOKEN_USERNAME"),
            "the guidance must name the lever: {guidance}"
        );
    }

    #[test]
    fn guidance_for_client_id_match_without_an_oid_claim_still_names_the_lever() {
        // The token matched on its client id but carries no `oid`, so there is
        // no concrete value to print. The sentence must stay actionable rather
        // than degrading to a dangling "set it to ." or printing a placeholder
        // as if it were the value.
        let guidance = IdentityVerdict::MatchesClientIdClaim {
            oid: None,
            client_id_claim: CLIENT_ID.to_string(),
        }
        .guidance();

        assert!(
            guidance.contains("LIVE_EDIT_VALKEY_TOKEN_USERNAME") && guidance.contains("Object ID"),
            "with no `oid` to print, the guidance must still name the lever and what to set it \
             to: {guidance}"
        );
        assert!(
            !guidance.contains(ABSENT_CLAIM),
            "the absent-claim placeholder must not be presented as the value to configure: \
             {guidance}"
        );
    }

    #[test]
    fn guidance_for_matches_neither_reports_both_claims_including_absent_ones() {
        let guidance = IdentityVerdict::MatchesNeither {
            oid: Some(OBJECT_ID.to_string()),
            client_id_claim: None,
        }
        .guidance();

        assert!(
            guidance.contains(OBJECT_ID),
            "the token's real `oid` is the value to configure: {guidance}"
        );
        assert!(
            guidance.contains(ABSENT_CLAIM),
            "a claim the token does not carry must render as an explicit absence marker rather \
             than an empty gap that reads as a blank value: {guidance}"
        );
    }

    #[test]
    fn guidance_for_matches_oid_narrows_to_the_remaining_causes() {
        // The healthy-identity verdict's whole value is the *derived* claim it
        // licenses — two of the four causes are ruled out from this service's
        // own data — so a guidance sentence that merely said "looks fine" would
        // leave the operator exactly where the opaque WRONGPASS did.
        let guidance = IdentityVerdict::MatchesOid.guidance();

        assert!(
            guidance.contains("LIVE_EDIT_VALKEY_TOKEN_USERNAME") && guidance.contains("`oid`"),
            "the healthy verdict must state the binding it verified: {guidance}"
        );
        assert!(
            guidance.contains("ruled out"),
            "the healthy verdict must narrow the search rather than just reassuring: {guidance}"
        );
    }

    #[test]
    fn guidance_for_unreadable_identity_verdicts_does_not_accuse_a_cause() {
        // `NotAJwt` is a legitimate deployment (opaque token) and
        // `NoIdentityClaim` is an unanswerable question; neither may be
        // rendered as a fault. The regression this guards is a copy-paste of
        // the cause-(1) sentence into these arms, which would tell an operator
        // running a perfectly valid opaque-token deployment to change a
        // correctly-set variable.
        for verdict in [IdentityVerdict::NotAJwt, IdentityVerdict::NoIdentityClaim] {
            let guidance = verdict.guidance();
            assert!(
                !guidance.contains("most common cause"),
                "{verdict:?} must not assert a cause it did not establish: {guidance}"
            );
            assert!(
                guidance.contains("Check") || guidance.contains("Confirm"),
                "{verdict:?} must hand the operator a manual next step: {guidance}"
            );
        }
    }
}
