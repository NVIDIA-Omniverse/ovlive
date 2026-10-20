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

//! Service-identity Bearer token provider (Epic `t1.fgj.dp`, SDD SR-6.1 / SR-6.2 / SR-6.4).
//!
//! Polls a sidecar-written JSON token file at a configurable interval and
//! exposes the most recently parsed `access_token` via a synchronous,
//! non-yielding accessor. Designed for the Epic 5 outbound storage-api
//! adapter which, for every checkpoint write / read, needs to attach a
//! `Bearer <token>` header on the hot path without awaiting.
//!
//! Design summary:
//!
//! - The token file is written by a credential-provisioner sidecar using
//!   write-temp + rename semantics, so a partial-write race on read is not
//!   possible. We rely on this in `read_and_update` — we read the file in a
//!   single `tokio::fs::read_to_string` call and do not need to coordinate
//!   with the writer.
//! - The cache is held in a `std::sync::RwLock<Option<CachedToken>>` so that
//!   `cached_bearer_if_unexpired()` can be synchronous (non-`async`) and never
//!   yields.
//!   `tokio::sync::RwLock` is deliberately rejected: callers in the Epic 5
//!   outbound-storage-api hot path must not be forced to await for an
//!   uncontended read.
//! - On parse failure, the cache is NOT updated — the previously-cached
//!   token (if any) continues to satisfy `cached_bearer_if_unexpired()` until
//!   the next successful poll. This is the "previous-good-cache" invariant
//!   referenced by SDD §14.6, pinned in code so a future refactor doesn't
//!   accidentally blow away a still-valid token on transient corruption upstream.
//! - On `NotFound`: the polling task tolerates it silently (cache stays
//!   empty per SR-6.2 startup-absent path); `force_refresh()` callers (the
//!   Epic 5 401-retry path) get an `Err(Io)` so they can distinguish
//!   "refresh attempted, file still absent" from "refresh succeeded".
//! - Cold start runs a bounded fast-poll phase: until the cache holds its
//!   first token the polling task re-reads at `FAST_POLL_RETRY_INTERVAL`
//!   rather than `poll_interval`, so a sidecar write landing a few hundred
//!   milliseconds after boot is cached immediately instead of one full poll
//!   interval later. See `run_polling_task`.
//! - The symmetric case — a *warmed-up* provider whose cached token is
//!   about to go stale because the sidecar's rotation is late — re-enters
//!   the same fast cadence: once the cached `expires_on` falls inside
//!   `refresh_lead + poll_interval` of now, the task re-reads at
//!   `FAST_POLL_RETRY_INTERVAL` until either a token with a different
//!   `expires_on` lands (silent retirement) or the cached token actually
//!   expires (one WARN, then a bounded post-expiry window at the same fast
//!   cadence, then back to the steady one). Without it a late rotation costs
//!   up to a full `poll_interval` of "no cached credential". See `StalePhase`
//!   and `run_polling_task`.
//! - One bearer accessor, one production policy. `cached_bearer_if_unexpired()`
//!   serves any not-yet-expired token and is what **every** production
//!   consumer reads: the Valkey `accessTokenFile` credential provider, the
//!   storage-api checkpoint adapter, and the health supervisor. A second,
//!   *withholding* accessor once returned `None` for a cached token that was
//!   inside the per-instance `refresh_lead`. Withholding a still-valid
//!   credential ahead of its expiry bought nothing and cost real failures —
//!   every `AUTH` for the width of the lead on the Valkey path (b.d6b), a
//!   self-inflicted 401 round trip once per token lifetime on the storage-api
//!   path (b.e49) — so it was deleted outright (b.mq8). The lead is
//!   refresh-cadence guidance, not a rule about when a token stops working; do
//!   not reintroduce an accessor that treats it as one.
//! - Every observed rotation is measured for *prolongation headroom* — how long
//!   before the outgoing token's expiry the replacement was actually picked up —
//!   and recorded as `live_edit_service_token_refresh_headroom_seconds{role}`.
//!   An instance constructed with a `prolongation_floor` (the Valkey one, whose
//!   floor is the rotation fan-out ceiling — twice the maximum re-AUTH tick
//!   interval its credential consumer schedules, 180 s) additionally WARNs once
//!   per rotation that lands under it. This is the
//!   runtime substitute for a startup check that cannot exist: the term the
//!   deployment recommendation constrains is the *sidecar's* refresh lead, which
//!   is not a live-edit config value (b.is1).
//!
//! ## Bearer-bytes-never-logged invariant (SR-6.2)
//!
//! Every `tracing::` call site in this module emits only generic descriptors —
//! never the file contents, never the parsed `access_token`, never any byte
//! slice that could carry token material. The `Parse` error variant carries
//! only the JSON / RFC3339 error category as a string, never the underlying
//! bytes. Grep this module for `tracing::` before changing logging.

use std::io;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::auth::userinfo_jwt::deserialize_optional_string;
use crate::config::Config;
use crate::observability::metrics::Metrics;

/// Default `refresh_lead` (the [`run_polling_task`] parameter that owns it):
/// how far ahead of `expires_on` the polling task starts hunting for a
/// replacement token.
///
/// **Purely a re-read cadence.** It is not a statement that a token inside the
/// window has stopped working, and no production consumer treats it as one:
/// the Valkey credential path (b.d6b) and the storage-api checkpoint adapter
/// and health supervisor (b.e49) all read
/// [`ServiceTokenProvider::cached_bearer_if_unexpired`], which ignores the lead
/// and serves any token that has not actually expired. Its only consumer is the
/// polling task's stale-phase entry lead (see [`StalePhase`]); the accessor that
/// applied it as a *withholding* threshold was deleted outright (b.mq8) once its
/// last production caller was retired.
const DEFAULT_REFRESH_LEAD: chrono::Duration = chrono::Duration::seconds(60);

/// Re-read cadence shared by both fast-poll phases: the cold-start phase
/// (active until the cache holds its first successfully-parsed token) and the
/// stale-token phase (active while a warmed-up cache's token is inside
/// `refresh_lead + poll_interval` of expiry, and over the bounded window just
/// past its expiry). See [`StalePhase`].
///
/// **Why 250 ms.** Both phases exist because the steady `poll_interval`
/// (default 15 s) is far too coarse for a token-file *transition* that the
/// service is blocked on: the read is a sub-millisecond `stat` + small-file
/// read, so re-reading four times a second costs nothing measurable, and it
/// collapses a transition the operator would otherwise wait a whole poll
/// interval for into sub-second. Both phases are bounded *per cached token*
/// (below): each entry retires on a wall-clock window or on that token's own
/// expiry, so no single arming runs for the process lifetime. In aggregate the
/// cadence can still run continuously in one degenerate configuration — a
/// token lifetime `<= refresh_lead + poll_interval`, where every rotation lands
/// already inside the entry lead and immediately re-arms the phase. Dead for no
/// consumer, in either sub-band: every production reader is
/// `cached_bearer_if_unexpired()`, which serves a token for its whole life
/// regardless of the lead (b.d6b for the Valkey credential path, b.e49 for the
/// storage-api adapter and the health supervisor). The deployment works; it
/// just never leaves the stale phase, because every fresh token is already
/// within `refresh_lead + poll_interval` of expiry when it is cached. Harmless:
/// the only cost is the 250 ms cadence itself — a `stat` plus a small-file read,
/// on a file the poll loop was going to re-read anyway — and it is bounded by
/// the clamp below, so it can never poll faster than four times a second.
///
/// **Why clamped.** Clamped to `poll_interval` at use so an operator-configured
/// interval shorter than this is never *slowed down* by either fast phase.
///
/// Per-phase notes:
///
/// - *Cold start.* `tokio::time::interval` fires tick #0 immediately at
///   process start, which on a cold pod start lands *before* the
///   credential-provisioner sidecar has written the token file (observed at
///   ~t+0.5 s). That first read is a tolerated NotFound (SR-6.2
///   startup-absent path); with only `poll_interval` to fall back on, the
///   token that appeared at t+0.5 s would not be cached until ~t+15 s — long
///   enough to stall the first Valkey connect under
///   `LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile` past the liveness-probe
///   grace window. Bounded by [`STARTUP_FAST_POLL_WINDOW`].
/// - *Stale token.* A late sidecar rotation on a warmed-up provider costs up
///   to one `poll_interval` of no usable credential once the cached token has
///   actually expired — correct fail-closed behavior, but a needless ~15 s hole
///   in outbound auth and in Valkey `AUTH` refresh. Re-reading at this cadence
///   over exactly the window
///   SDD §15.4's invariant covers shrinks that to ~250 ms. Bounded by the
///   cached token's own expiry, plus [`STARTUP_FAST_POLL_WINDOW`] past it
///   (b.d6b: recovery from a token that *has* expired is the most urgent case,
///   not the least) — see [`StalePhase`].
const FAST_POLL_RETRY_INTERVAL: Duration = Duration::from_millis(250);

/// Upper bound on the two *credential-less* [`FAST_POLL_RETRY_INTERVAL`]
/// fast-poll windows: the cold-start phase, and the post-expiry
/// `StalePhase::RetiredFor` window (b.d6b).
///
/// Bounded so a token file that never appears (misconfigured sidecar, wrong
/// `LIVE_EDIT_SERVICE_TOKEN_FILE` path) — or one that stops being rewritten
/// after the cached token dies — does not leave the polling task stat-ing a
/// dead path four times a second for the process lifetime. On expiry of the
/// cold-start window the task logs one WARN and reverts to `poll_interval`;
/// the post-expiry window is silent on close (its WARN was already emitted at
/// retirement).
///
/// Deliberately sized *below* the default Valkey connect budget
/// (`LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`, default 30 s). Under
/// `LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile` the process exits when
/// `ValkeyClient::connect` gives up, so a window at or above that budget
/// makes the expiry WARN structurally unreachable on exactly the path whose
/// misconfiguration it exists to diagnose — the pod is already gone by the
/// time the window closes. 20 s leaves ~10 s of headroom for the WARN to
/// reach the log pipeline before the exit. Raising this at or above the
/// connect budget silently re-breaks that diagnostic.
///
/// The stale-token phase's *pre-expiry* half (`StalePhase::Active`)
/// deliberately does NOT take this window: it is condition-bounded (it
/// self-terminates at the cached token's actual `expires_on`, and re-arms only
/// for a *different* cached expiry), and the 20 s here is *shorter* than the
/// ~60 s sidecar lateness that half exists to ride out — inheriting it would
/// end the fast cadence while the credential is still perfectly good.
///
/// Its *post-expiry* half (`StalePhase::RetiredFor`) does take it, and the
/// derivation transfers verbatim: past `expires_on` there is no usable
/// credential at all, so under `LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile`
/// the process is heading for the same `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`
/// exit the cold-start sizing above is written against. Twenty seconds of
/// 250 ms re-reads is the whole budget that exists before the pod dies.
const STARTUP_FAST_POLL_WINDOW: Duration = Duration::from_secs(20);

/// State of the stale-token fast-poll phase, owned task-locally by
/// [`run_polling_task`] and keyed on the cached token's `expires_on`.
///
/// The phase re-arms the [`FAST_POLL_RETRY_INTERVAL`] cadence on a *warmed-up*
/// provider whose cached token is about to go stale, so a late sidecar
/// rotation is picked up sub-second instead of on the next steady tick. Five
/// design points, all load-bearing:
///
/// - **Entry predicate is `expires_on <= now + refresh_lead + poll_interval`**,
///   not `+ refresh_lead` alone. The extra `poll_interval` arms the phase one
///   steady tick *before* the lead window opens; without it, entry and the next
///   steady read coincide whenever the sidecar's refresh lead falls between
///   `refresh_lead` and `refresh_lead + poll_interval`, and the phase is a
///   no-op on exactly the deployments that need it. The sum is SDD §15.4's
///   left-hand side, so the phase is active over precisely the window that
///   invariant says the sidecar lead must dominate.
/// - **`refresh_lead` is per-instance** (the [`run_polling_task`] parameter,
///   chosen by the constructor the wiring layer picked), never the
///   [`DEFAULT_REFRESH_LEAD`] constant: the Valkey credential instance runs a ≥ 3 min
///   lead (≈ 195 s window at the default poll interval), the storage-api
///   instance 60 s (≈ 75 s window).
/// - **"A fresher token appeared" is "the cached `expires_on` changed"** — not
///   a file mtime, not a byte comparison. That composes with the
///   previous-good-cache invariant for free: a parse failure leaves
///   `expires_on` untouched, so the phase correctly stays active. And the
///   entry predicate is monotone for a fixed token, so a token that is inside
///   the window can never fall back out of it — no flapping.
/// - **Bound is the token's own expiry, re-armed once per distinct cached
///   expiry.** `Active(E)` becomes `RetiredFor { expiry: E, .. }` when `E`
///   passes; a different cached expiry `E'` resets to `Idle` (and may then
///   re-enter as `Active(E')`). Re-entry for the *same* `E` after retirement is
///   refused, which is what makes the retirement WARN one-per-distinct-expiry
///   rather than per-tick.
/// - **Retirement does not immediately drop back to the steady cadence**
///   (b.d6b). `RetiredFor` means the cached token is actually dead — every
///   consumer is now credential-less, which is the *most* urgent moment to be
///   hunting for the replacement, not the least. So [`Self::wants_fast_poll`]
///   keeps the fast arm live for [`STARTUP_FAST_POLL_WINDOW`] past the
///   retirement instant (`at`), reusing the cold-start window's derivation: a
///   post-expiry process under `LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile` is
///   heading for the same `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` exit that
///   window is sized against. Before b.d6b, recovery from total credential
///   loss reverted to the steady poll (up to `poll_interval`, default 15 s)
///   exactly when the outage became total.
/// - **That post-expiry window is per-*retirement*, not per-outage.** `at` is
///   re-stamped on every entry into `RetiredFor`, and the previous bullet's
///   "a different cached expiry resets to `Idle`" applies to already-expired
///   tokens too: a sidecar that writes a stream of *distinct* already-expired
///   tokens drives `RetiredFor(E, at) → Idle → RetiredFor(E', at')` on each
///   one, so the 250 ms cadence (and one WARN per distinct expiry) keeps
///   running indefinitely rather than lapsing [`STARTUP_FAST_POLL_WINDOW`]
///   after the first. Deliberate: each newly-observed expired token is
///   evidence the sidecar is alive and writing, so a usable one may be moments
///   away — the case the fast cadence exists for. A *single* expired token
///   that never changes retires once and correctly lapses back to the steady
///   poll. Both halves are pinned by
///   `post_expiry_fast_poll_stops_after_the_startup_window`, whose two-token
///   shape depends on this; do not "fix" the re-stamping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StalePhase {
    /// No cached token, or the cached token is comfortably outside the entry
    /// window. The steady-state timer arm is the live one.
    Idle,
    /// Fast-polling for a replacement of the token expiring at this instant.
    Active(DateTime<Utc>),
    /// The token expiring at `expiry` expired without a replacement landing;
    /// the WARN has been emitted and the phase will not re-arm for this expiry
    /// again. Only a *different* cached expiry leaves this state.
    ///
    /// `at` is the retirement instant, and it is the anchor for the bounded
    /// post-expiry fast-poll window (see [`Self::wants_fast_poll`]). A
    /// `tokio::time::Instant`, not `std::time::Instant`, so paused-clock tests
    /// can advance past the window with virtual time — same reason
    /// [`ParseFailureGate`] uses one.
    RetiredFor {
        expiry: DateTime<Utc>,
        at: tokio::time::Instant,
    },
}

impl StalePhase {
    /// The cached `expires_on` this phase is keyed on, if any. Used at the
    /// loop head to detect that the cache now holds a *different* token.
    fn keyed_expiry(self) -> Option<DateTime<Utc>> {
        match self {
            StalePhase::Idle => None,
            StalePhase::Active(e) | StalePhase::RetiredFor { expiry: e, .. } => Some(e),
        }
    }

    /// Whether the stale [`FAST_POLL_RETRY_INTERVAL`] arm should be the live
    /// one for this phase — and, by exact complement, whether the steady-state
    /// arm should be guard-disabled. The two `select!` guards are written from
    /// this one predicate so they can never overlap or both go dark.
    ///
    /// True while `Active` (hunting for a replacement before the cached token
    /// dies) and, per b.d6b, for [`STARTUP_FAST_POLL_WINDOW`] after a
    /// `RetiredFor` retirement (hunting for a replacement *after* it died —
    /// see the type-level docs for why that is the more urgent of the two).
    /// Bounded rather than open-ended for the same reason the cold-start phase
    /// is: a sidecar that has stopped writing entirely must not leave the task
    /// stat-ing a dead path four times a second for the process lifetime.
    fn wants_fast_poll(self) -> bool {
        match self {
            StalePhase::Idle => false,
            StalePhase::Active(_) => true,
            StalePhase::RetiredFor { at, .. } => at.elapsed() < STARTUP_FAST_POLL_WINDOW,
        }
    }
}

/// Caller identity for `read_and_update`. Polling and `force_refresh()`
/// share the same parse / cache-update body but differ in two ways: how
/// `NotFound` is treated — polling silently tolerates an absent file
/// (startup-absent path, SR-6.2), while `force_refresh()` propagates it so
/// the Epic 5 401-retry path can distinguish "refresh failed" from "refresh
/// succeeded with stale-still-empty cache" — and how read-failure emission
/// is paced: the WARN on both failure branches (non-`NotFound` I/O error and
/// parse failure) plus, on the parse branch only, the accompanying metric
/// increment (see [`ParseFailureGate`]).
#[derive(Debug, Clone, Copy)]
enum ReadCaller<'a> {
    /// A polling-task read. `Some(gate)` paces read-failure emission,
    /// `None` leaves it exact (one WARN — and, on the parse branch, one
    /// increment — per read); the `NotFound` behavior is identical either
    /// way — only emission pacing differs.
    ///
    /// `Some(gate)` is used by **both** fast-poll arms — the cold-start arm
    /// (and the t≈0 pre-loop read that precedes it) and the stale-token arm
    /// (see [`StalePhase`]) — which re-read up to 4×/s: a token file that is
    /// present but malformed, or present but unreadable (EACCES, wrong-type
    /// path, bad mount), would otherwise amplify one operator-visible fault
    /// into ~80 identical WARNs per 20 s — and, when the fault is a parse
    /// failure, as many metric increments. Both arms are handed the *same*
    /// gate instance: they are mutually exclusive by construction (the
    /// `select!` guards are `fast_poll_active` and
    /// `stale_phase.wants_fast_poll()`, and the stale phase is only evaluated
    /// when `!fast_poll_active`), so sharing one instance yields
    /// the intended "at most one failure WARN per `poll_interval`" bound in
    /// whichever phase is running rather than one budget per phase.
    ///
    /// `None` is used by the steady-state arm, where `poll_interval` *is*
    /// the rate limit already. Gating there would be a second, redundant
    /// limiter at the same nominal period, and — because the gate stamps at
    /// emission time (after the async file read) while ticks fire on
    /// tick-time deadlines — per-read filesystem jitter would push
    /// `now - prev` to either side of `min_interval`, nondeterministically
    /// dropping roughly a third of steady-state WARNs and
    /// `live_edit_service_token_parse_errors_total` increments.
    Polling(Option<&'a ParseFailureGate>),
    /// An explicit [`ServiceTokenProvider::force_refresh`] read. Emission is
    /// exact — one WARN per failed read (plus one metric increment when the
    /// failure is a parse failure), ungated — because these are
    /// caller-driven, one-per-401 events rather than a timer cadence, and the
    /// Epic 5 retry path's diagnostics depend on the one-to-one
    /// correspondence.
    ForceRefresh,
}

/// Rate-limit gate for the polling path's read-failure emission: the WARN on
/// both of `read_and_update`'s failure branches — a non-`NotFound` I/O error
/// (present-but-unreadable file) and a parse failure (present-but-malformed
/// file) — plus, on the parse branch only, the accompanying
/// `live_edit_service_token_parse_errors_total` increment. One gate instance
/// serves both branches: a given read either fails I/O or reaches the parser,
/// never both, so they never compete for an emission slot, and the shared
/// instance yields "at most one startup-phase failure WARN per
/// `min_interval`" across whichever fault the file exhibits.
///
/// Owned by [`run_polling_task`] (one per task, never shared across
/// providers) and handed to [`read_and_update`] through
/// `ReadCaller::Polling(Some(..))` — from the cold-start fast-poll arm, the
/// t≈0 pre-loop read that precedes it, and the stale-token fast-poll arm
/// ([`StalePhase`]). The steady-state arm passes `None`. Deliberately not
/// reachable from [`ReadCaller::ForceRefresh`] — see that variant's docs.
///
/// A `std::sync::Mutex` rather than a `Cell`: the gate is borrowed across an
/// `await` inside `read_and_update`, so `&Self` must be `Send`, which
/// requires `Self: Sync`.
#[derive(Debug)]
struct ParseFailureGate {
    /// Minimum spacing between emissions on the *gated* polling reads (the
    /// cold-start fast phase and the stale-token fast phase). Set to the
    /// steady-state `poll_interval` so either fast phase's up-to-4×/s
    /// re-reads can't emit faster than the steady cadence alone would. The
    /// two phases never run concurrently, so one `min_interval` gives the
    /// intended bound in both. The steady-state arm is ungated, so this
    /// bound never applies to it — see [`ReadCaller::Polling`].
    min_interval: Duration,
    /// `tokio::time::Instant`, not `std::time::Instant`: paused-clock tests
    /// advance virtual time only, and a std clock would never open the gate.
    last_emitted: std::sync::Mutex<Option<tokio::time::Instant>>,
}

impl ParseFailureGate {
    fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_emitted: std::sync::Mutex::new(None),
        }
    }

    /// Returns `true` at most once per `min_interval`. The first call always
    /// passes, so a parse failure is never fully silent.
    fn allow(&self) -> bool {
        let now = tokio::time::Instant::now();
        let mut last = self
            .last_emitted
            .lock()
            .expect("ServiceTokenProvider parse-failure gate Mutex poisoned");
        match *last {
            Some(prev) if now.saturating_duration_since(prev) < self.min_interval => false,
            _ => {
                *last = Some(now);
                true
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CachedToken {
    access_token: String,
    expires_on: DateTime<Utc>,
    /// Non-secret identity claims projected out of `access_token` at
    /// cache-write time, or `None` when the token is not a decodable JWT.
    /// See [`TokenClaims`] and [`decode_token_claims`].
    claims: Option<TokenClaims>,
}

/// The **fixed, named** set of non-secret claims projected out of a JWT-shaped
/// access token, used by the Valkey `accessTokenFile` auth diagnostics (b.cbe).
///
/// ## Why a named projection and not the raw payload
///
/// The bearer-bytes-never-logged invariant (SR-6.2, module docs above) is what
/// makes this type safe to exist at all: the diagnostic path *prints* what is in
/// here, so the type itself is the enforcement point. Every field is a
/// non-secret identity/routing claim (`aud`, `oid`, `appid`/`azp`, `tid`, `iss`,
/// `exp`) — none of them is credential material, and none of them can be used to
/// reconstruct the token. Keeping the shape closed is deliberate: neither the
/// raw payload JSON nor a `serde_json::Value` claim bag may be stored here,
/// because either would let an unreviewed claim (`sub` on a token that carries a
/// secret in it, a vendor extension, a nested object) reach a log line by
/// accident. Adding a field is a deliberate review step; widening to a bag is
/// not permitted.
///
/// ## Decode is best-effort and unverified
///
/// Same trust posture as [`crate::auth::userinfo_jwt`]: the payload is
/// base64url-decoded and JSON-parsed, and **no signature, issuer, audience or
/// expiry verification happens**. Nothing here is consumed for an authorization
/// decision — the sole consumer is an operator-facing diagnostic that explains
/// *why the backing store rejected our AUTH*. A token that is not a JWT at all
/// (an opaque string, a SigV4 blob) decodes to `None` and the diagnostic
/// silently degrades to the mode-level advice.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenClaims {
    /// `aud` — the audience(s) the token was minted for. A JWT `aud` is a
    /// string **or** an array of strings (RFC 7519 §4.1.3), so this is always a
    /// `Vec`: modelling it as a single `Option<String>` silently drops an
    /// array-valued audience, which is precisely the wrong-scope case the
    /// diagnostic exists to surface. Empty when the claim is absent or is
    /// neither of the two legal shapes.
    pub aud: Vec<String>,
    /// `oid` — the token subject's directory Object ID. This is the identity a
    /// managed backing store keys its data-plane access policy by, and the
    /// value the Valkey AUTH username is expected to equal.
    pub oid: Option<String>,
    /// The token's **client-id** claim: `appid` when present, else `azp`. Held
    /// separately from `oid` because "the operator configured the client ID
    /// where the Object ID was required" is the single most common cause of an
    /// opaque `WRONGPASS`, and distinguishing it needs both values.
    pub client_id_claim: Option<String>,
    /// `tid` — the issuing tenant. Reported, never judged.
    pub tid: Option<String>,
    /// `iss` — the issuer. Reported, never judged.
    pub iss: Option<String>,
    /// `exp` — expiry as a JWT NumericDate (seconds since the Unix epoch), so
    /// an `i64` rather than a string. Reported alongside the cache's own
    /// `expires_on` so an operator can see whether the sidecar's stated expiry
    /// and the token's own agree.
    pub exp: Option<i64>,
}

/// Raw serde shape for the payload segment. Every field is optional and
/// shape-tolerant: a claim of the wrong JSON type collapses to "missing"
/// rather than failing the whole decode, so one unexpected claim never costs
/// the operator the rest of the diagnostic.
#[derive(Debug, Deserialize)]
struct RawJwtClaims {
    #[serde(default, deserialize_with = "deserialize_audience")]
    aud: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    oid: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    appid: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    azp: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    tid: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    iss: Option<String>,
    #[serde(default, deserialize_with = "deserialize_numeric_date")]
    exp: Option<i64>,
}

/// Deserialize a JWT `aud` claim, which RFC 7519 §4.1.3 defines as either a
/// single string or an array of strings.
///
/// [`deserialize_optional_string`] cannot cover this: it collapses an array to
/// `None`, which would report "no audience" for the perfectly ordinary
/// multi-audience token and hide the wrong-scope failure the diagnostic is
/// meant to expose. Non-string array members are skipped; any other JSON shape
/// yields an empty vec.
fn deserialize_audience<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(items) => items
            .into_iter()
            .filter_map(|item| match item {
                serde_json::Value::String(s) => Some(s),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    })
}

/// Deserialize a JWT NumericDate claim (`exp`) as `Option<i64>`.
///
/// NumericDate is a *number* of seconds since the epoch, so the string helper
/// would report every real `exp` as missing. Non-numeric shapes collapse to
/// `None`; a fractional value truncates toward zero.
fn deserialize_numeric_date<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(match value {
        serde_json::Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        _ => None,
    })
}

/// Best-effort, verification-free projection of a token's non-secret claims.
///
/// Returns `None` — silently, with no log line and no metric — whenever the
/// input is not a decodable JWT: not exactly three `.`-separated segments, a
/// payload segment that is not base64url (no padding), or payload bytes that
/// are not JSON. That is the issuer-agnostic contract: an opaque token, or the
/// SigV4 credential the `iam` auth mode builds, is not an error condition here,
/// it just means no claims-level self-check is available.
///
/// Shares the decoding substrate with [`crate::auth::userinfo_jwt`] — the same
/// three-segment split, the same `URL_SAFE_NO_PAD` engine, and literally the
/// same [`deserialize_optional_string`] helper — but deliberately not
/// `JwtUserinfoClient` itself: that type is an authenticated-user display-name
/// resolver bound to the `UserinfoClient` trait, and this is a claims
/// projection — written for the service-identity diagnostic, now also read by
/// the auth cache (below) — with a different claim set and a different
/// failure contract (`None`, not an `AuthError`).
///
/// Two callers, with two contracts:
///
/// - **Service-identity diagnostics** (this module): runs on the polling task
///   at cache-write time, never on a `fetch()` /
///   `cached_bearer_if_unexpired()` hot path.
/// - **The request path's auth cache** (`auth::cache::bearer_cache_bound`):
///   runs on every positive `AuthCache` insert — an `Allowed` permission
///   verdict or a successful userinfo lookup — for the CALLER's bearer, and
///   reads only `exp`. It is a verification-free read that can only SHORTEN
///   how long that entry is served (the configured TTL still caps it); a
///   forged `exp` can at worst cost an earlier backend call. `None`, or a
///   claim set without `exp`, means "no bound" and the configured TTL alone
///   applies. Keep it cheap and side-effect-free for that caller: it now
///   sits on a cache miss of an RPC.
pub(crate) fn decode_token_claims(token: &str) -> Option<TokenClaims> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload = URL_SAFE_NO_PAD.decode(parts[1]).ok()?;

    // Parse to a free-form `Value` and require a JSON *object* before typing it.
    // RFC 7519 §7.2 makes the payload a JSON object, and without this guard a
    // JSON array payload does not fail — serde's derived impl accepts a sequence
    // and fills the fields POSITIONALLY (aud, oid, appid, azp, tid, iss, exp),
    // so `["ignored","1111-…","client",…]` lands a real-looking `oid` and can
    // produce a `MatchesOid` verdict by pure coincidence. That is an
    // authoritative misdiagnosis from a diagnostic whose whole job is to be
    // trusted, and it silently couples behaviour to field-declaration order.
    // Every non-object payload is "not a decodable JWT" -> `None`.
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    if !value.is_object() {
        return None;
    }
    let raw: RawJwtClaims = serde_json::from_value(value).ok()?;

    // A present-but-empty claim is treated as missing, matching the fallback
    // semantics in `userinfo_jwt`: an empty `oid` must not compare equal to an
    // empty configured username and report a healthy identity match.
    fn non_empty(value: Option<String>) -> Option<String> {
        value.filter(|s| !s.is_empty())
    }

    Some(TokenClaims {
        aud: raw.aud.into_iter().filter(|s| !s.is_empty()).collect(),
        oid: non_empty(raw.oid),
        client_id_claim: non_empty(raw.appid).or_else(|| non_empty(raw.azp)),
        tid: non_empty(raw.tid),
        iss: non_empty(raw.iss),
        exp: raw.exp,
    })
}

/// JSON shape written by the credential-provisioner sidecar (SDD SR-6.1).
/// `expires_on` is an RFC3339 timestamp; `chrono`'s default `Deserialize`
/// impl for `DateTime<Utc>` accepts RFC3339.
#[derive(Debug, Deserialize)]
struct TokenFileJson {
    expires_on: DateTime<Utc>,
    access_token: String,
}

/// Errors surfaced from [`ServiceTokenProvider::force_refresh`].
///
/// The polling task tolerates I/O / parse failures internally (logging and
/// metric-ticking), so these only appear on the explicit-refresh path used
/// by the Epic 5 401-retry caller. Bearer-bytes-never-logged invariant: the
/// `Parse` variant carries only a generic descriptor string + the error
/// category — never the file contents or the parsed token bytes.
#[derive(Debug, Error)]
pub enum ServiceTokenError {
    /// The token file is missing OR another I/O error fired while reading
    /// it. Polling-path NotFound is filtered out before reaching this
    /// variant; only `force_refresh()` callers see `Err(Io)` for NotFound.
    #[error("service token file I/O error: {0}")]
    Io(#[from] io::Error),
    /// JSON or RFC3339 parse failed. Carries only a generic descriptor; the
    /// file contents are NEVER included to honor the
    /// bearer-bytes-never-logged invariant (SR-6.2).
    #[error("{0}")]
    Parse(String),
    /// Returned from `force_refresh()` when the cancellation token has
    /// been signaled and the polling task has shut down before the request
    /// could be serviced. The request oneshot is dropped without a
    /// response, and `force_refresh()` maps that into this variant so
    /// callers see a meaningful failure rather than an opaque recv-error.
    /// Carried explicitly so future tree-shake / dead-code-warning passes
    /// don't drop it.
    #[error("service token provider cancelled")]
    Cancelled,
}

struct Inner {
    /// `std::sync::RwLock` (NOT `tokio::sync::RwLock`) so
    /// `cached_bearer_if_unexpired()` stays synchronous and never yields — see
    /// module docs.
    cache: RwLock<Option<CachedToken>>,
    path: PathBuf,
    metrics: Arc<Metrics>,
    /// Channel into the polling task used by [`ServiceTokenProvider::force_refresh`].
    /// Each request carries a oneshot reply slot so the caller can await
    /// the resulting `Result`.
    force_refresh_tx: mpsc::Sender<oneshot::Sender<Result<(), ServiceTokenError>>>,
}

impl Inner {
    /// `expires_on` of the currently cached token, or `None` when the cache
    /// has never been populated by a successful parse.
    ///
    /// The cache is the single source of truth for token expiry — there is
    /// deliberately no task-local mirror and no `AtomicI64` shadow copy, so
    /// this and [`ServiceTokenProvider::cached_bearer_if_unexpired`] can never
    /// disagree about which token is live.
    ///
    /// **Projects only the `Copy` timestamp, never the token bytes.** The
    /// read guard is dropped before returning, so no caller can widen this
    /// into an accessor that hands out `access_token` — the
    /// bearer-bytes-never-logged invariant (SR-6.2) depends on that.
    ///
    /// Same `std::sync::RwLock` read-guard discipline as
    /// `cached_bearer_if_unexpired()`, and the substrate the public
    /// [`ServiceTokenProvider::cached_token_expires_on`] accessor delegates to
    /// rather than re-deriving its own lock usage.
    ///
    /// A poisoned lock reports `None`, matching
    /// `cached_bearer_if_unexpired`'s `.read().ok()?` posture. That is a
    /// total-function default, NOT a graceful-degradation path:
    /// `read_and_update` `.expect()`s on this same lock, so a poisoned cache
    /// means the polling task — the only writer — has already panicked, and no
    /// later read can populate the cache. The branch exists so this cannot
    /// itself panic, not because polling keeps working through it.
    fn cached_expires_on(&self) -> Option<DateTime<Utc>> {
        self.cache.read().ok()?.as_ref().map(|c| c.expires_on)
    }

    /// The non-secret [`TokenClaims`] projected from the currently cached
    /// token, or `None` when the cache has never been populated **or** the
    /// cached token is not a decodable JWT.
    ///
    /// Same discipline as [`Self::cached_expires_on`] and for the same reasons:
    /// it projects a *named, non-secret* subset (never `access_token`), the
    /// read guard is dropped before the value is returned, and a poisoned lock
    /// reports `None` matching `cached_bearer_if_unexpired`'s `.read().ok()?`
    /// posture.
    ///
    /// The clone is cheap and happens under the read guard; the decode itself
    /// already ran on the polling task at cache-write time
    /// ([`read_and_update`]), so no caller pays JSON-parsing cost here.
    fn cached_claims(&self) -> Option<TokenClaims> {
        let guard = self.cache.read().ok()?;
        let claims = guard.as_ref().and_then(|c| c.claims.clone());
        drop(guard);
        claims
    }

    /// Whether the cache has ever been populated by a successful parse.
    ///
    /// Deliberately NOT `cached_bearer_if_unexpired().is_some()`: this is the
    /// cold-start phase's *retirement* predicate in [`run_polling_task`], and
    /// the condition being tested is "no token has ever loaded", which is
    /// exactly "cache is `None`". A token that parsed fine but has since run
    /// out makes the bearer accessor report `None` while the cache is
    /// populated, so gating on it would leave the phase **un-retired** on a
    /// warm restart straight into an already-expired token: the
    /// [`FAST_POLL_RETRY_INTERVAL`] cadence would stay pinned for the full
    /// [`STARTUP_FAST_POLL_WINDOW`], and the phase would then close on that
    /// window with the "file still absent or unparseable" WARN — emitted
    /// against a file that is present and parses fine. `fast_poll_active` is
    /// monotone (set once before the loop, only ever cleared), so the failure
    /// mode is a phase that never retires, never one that re-arms.
    ///
    /// A warmed-up provider whose token is nearing or past its expiry is the
    /// *stale-token* phase's job ([`StalePhase`]) — that one keys on the cached
    /// expiry, re-arms once per distinct expiry, and carries its own bounded
    /// post-expiry window.
    ///
    /// Expressed through [`Self::cached_expires_on`] so the cache's
    /// lock discipline lives at exactly one site.
    fn has_cached_token(&self) -> bool {
        self.cached_expires_on().is_some()
    }
}

/// Cached service-identity Bearer token, refreshed from a sidecar-written
/// JSON file on a configurable poll interval.
///
/// Public surface:
///
/// - [`Self::new`] — construct + spawn the polling task. The task runs until
///   `cancel` is signaled.
/// - [`Self::cached_bearer_if_unexpired`] — the provider's **sole bearer
///   accessor**, synchronous and non-withholding: `Some(token)` whenever the
///   cache holds a token that has not actually expired, with the per-instance
///   refresh lead ignored. Every production consumer reads it: the Valkey
///   `accessTokenFile` credential provider (b.d6b), the storage-api checkpoint
///   adapter, and the health supervisor (b.e49). A second, *withholding*
///   accessor that returned `None` inside the lead was deleted once its last
///   production caller was retired (b.mq8) — do not reintroduce one.
/// - [`Self::cached_token_expires_on`] — synchronous expiry projection (the
///   `Copy` timestamp only, never the token bytes).
/// - [`Self::cached_token_claims`] — synchronous projection of the cached
///   token's named non-secret claims ([`TokenClaims`]), decoded at cache-write
///   time. Consumed by the Valkey `accessTokenFile` auth diagnostic.
/// - [`Self::force_refresh`] — async on-demand re-read. Used by the Epic 5
///   401-retry path to bypass the poll interval after a stale-token
///   rejection from upstream.
///
/// Wrap in `Arc` for sharing across the gRPC service-state struct and the
/// Epic 5 storage-api adapter.
pub struct ServiceTokenProvider {
    inner: Arc<Inner>,
}

impl ServiceTokenProvider {
    /// Role label for the instance backing `LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile`.
    ///
    /// A stalled token pickup here is fatal: `main.rs` exits the process when
    /// the Valkey connect budget is spent, so the operator is looking at a
    /// crash-looping pod.
    pub const ROLE_VALKEY: &'static str = "valkey";
    /// Role label for the instance backing the outbound storage-api
    /// checkpoint adapter.
    ///
    /// A stalled token pickup here only degrades checkpoint auth (outbound
    /// calls go unauthenticated / retry on 401); the service stays up.
    pub const ROLE_STORAGE_API: &'static str = "storage-api";
    /// Role label used by the skew-only constructors, which predate the
    /// label and are still used by tests and non-`main.rs` call sites.
    const ROLE_UNSPECIFIED: &'static str = "unspecified";

    /// Construct a provider and spawn its polling task.
    ///
    /// The task reads `path` once immediately, then — until that first
    /// read (or a later one) successfully populates the cache — re-reads at
    /// [`FAST_POLL_RETRY_INTERVAL`], bounded by [`STARTUP_FAST_POLL_WINDOW`],
    /// so a token the sidecar writes shortly after boot is picked up
    /// sub-second instead of on the next `poll_interval` tick. It re-enters
    /// that same cadence later, per [`StalePhase`], whenever the cached
    /// token nears expiry without a replacement having landed. Otherwise it
    /// loops on `tokio::time::interval(poll_interval)` with
    /// `MissedTickBehavior::Skip` so a long pause (e.g. tokio scheduling
    /// stall) does not produce a burst of catch-up ticks. On every tick
    /// and every `force_refresh()` request it re-reads `path` and updates
    /// the cache. The task exits cleanly when `cancel.cancelled()` fires;
    /// any outstanding `force_refresh()` request oneshot is dropped, which
    /// surfaces to the caller as [`ServiceTokenError::Cancelled`].
    pub fn new(
        path: PathBuf,
        poll_interval: Duration,
        metrics: Arc<Metrics>,
        cancel: CancellationToken,
    ) -> Self {
        // Delegate with the default 60 s skew so the storage-api call site
        // (main.rs) and the health.rs test helper keep their exact behavior.
        Self::new_with_refresh_lead(path, poll_interval, DEFAULT_REFRESH_LEAD, metrics, cancel)
    }

    /// [`Self::new`] plus a static role label.
    ///
    /// Additive surface: [`Self::new`] and [`Self::new_with_refresh_lead`]
    /// keep their existing signatures (many call sites, including the test
    /// harnesses, depend on them) and pass [`Self::ROLE_UNSPECIFIED`]. The
    /// label is carried into the polling task's operator-facing log lines —
    /// the startup fast-poll-window expiry WARN, the stale-token phase-entry
    /// INFO, and the cached-token-expired retirement WARN — where it
    /// distinguishes the two production instances: the same message means
    /// "the pod is about to die" for [`Self::ROLE_VALKEY`] and "the
    /// checkpoint adapter will degrade" for [`Self::ROLE_STORAGE_API`], and
    /// `path` alone does not tell an operator which one they are reading.
    pub fn new_with_role(
        path: PathBuf,
        poll_interval: Duration,
        role: &'static str,
        metrics: Arc<Metrics>,
        cancel: CancellationToken,
    ) -> Self {
        Self::spawn(
            path,
            poll_interval,
            DEFAULT_REFRESH_LEAD,
            None,
            role,
            metrics,
            cancel,
        )
    }

    /// [`Self::new_with_refresh_lead`] plus a static role label — see
    /// [`Self::new_with_role`] for what the label is for — and the instance's
    /// recommended prolongation floor (see [`run_polling_task`], which owns it).
    ///
    /// The only constructor that takes the floor, because it is the only one
    /// the wiring layer uses for an instance whose credential is consumed by
    /// long-lived connections. Every other constructor passes `None`, which
    /// keeps the headroom observation gauge-only.
    pub fn new_with_refresh_lead_and_role(
        path: PathBuf,
        poll_interval: Duration,
        refresh_lead: chrono::Duration,
        prolongation_floor: Option<chrono::Duration>,
        role: &'static str,
        metrics: Arc<Metrics>,
        cancel: CancellationToken,
    ) -> Self {
        Self::spawn(
            path,
            poll_interval,
            refresh_lead,
            prolongation_floor,
            role,
            metrics,
            cancel,
        )
    }

    /// Construct a provider with an explicit proactive re-read lead.
    ///
    /// Identical to [`Self::new`] except the caller chooses `refresh_lead`
    /// instead of inheriting the 60 s [`DEFAULT_REFRESH_LEAD`] default. The
    /// value is handed to [`run_polling_task`], where it does exactly one
    /// thing: drive the stale-token phase's proactive re-read window. It is not
    /// a validity floor and no accessor consults it —
    /// [`Self::cached_bearer_if_unexpired`] serves any cached token that has
    /// not actually expired. See [`run_polling_task`] for the full semantics.
    ///
    /// The access-token-file Valkey path passes a ≥ 3 min lead (Azure asks that
    /// a fresh token be fetched ≥ 3 min before expiry, SDD SR-4.1); the
    /// storage-api path uses [`Self::new`] and keeps the 60 s lead.
    pub fn new_with_refresh_lead(
        path: PathBuf,
        poll_interval: Duration,
        refresh_lead: chrono::Duration,
        metrics: Arc<Metrics>,
        cancel: CancellationToken,
    ) -> Self {
        Self::spawn(
            path,
            poll_interval,
            refresh_lead,
            None,
            Self::ROLE_UNSPECIFIED,
            metrics,
            cancel,
        )
    }

    /// Shared construction body for every public constructor above.
    ///
    /// `refresh_lead`, `role` and `prolongation_floor` are passed to
    /// [`run_polling_task`] rather than stored on [`Inner`]: every site that
    /// reads any of them — the cold-start fast-poll-window expiry WARN, the
    /// stale-token phase's entry lead, entry INFO and retirement WARN, and the
    /// rotation-headroom observation — is inside the task, so widening the
    /// shared state buys nothing. `refresh_lead` joined them in b.mq8, when the
    /// deletion of the withholding accessor removed its last reader outside the
    /// task; keep it that way, so a future accessor cannot quietly grow a
    /// dependency on the lead.
    fn spawn(
        path: PathBuf,
        poll_interval: Duration,
        refresh_lead: chrono::Duration,
        prolongation_floor: Option<chrono::Duration>,
        role: &'static str,
        metrics: Arc<Metrics>,
        cancel: CancellationToken,
    ) -> Self {
        // Capacity 8 is generous: each in-flight refresh request occupies
        // one slot until the polling task consumes it. Realistic load is
        // single-digit concurrent 401-retry callers fanning into the
        // single polling task; backpressure above that bound is fine.
        // `force_refresh()` uses `send(...).await`, so a full channel
        // simply parks the caller until the polling task dequeues a
        // slot — no requests are dropped, and the wait is bounded by
        // how long `read_and_update` takes per dequeued request.
        let (force_refresh_tx, force_refresh_rx) = mpsc::channel(8);

        let inner = Arc::new(Inner {
            cache: RwLock::new(None),
            path,
            metrics,
            force_refresh_tx,
        });

        // Spawn the polling task. It owns a clone of `inner` so the
        // provider handle drop does not stop polling; only `cancel`
        // shuts the task down. This matches the b.w29 / b.fgj
        // background-task pattern: cancellation is the sole shutdown
        // signal, never relying on refcount.
        let polling_inner = Arc::clone(&inner);
        tokio::spawn(run_polling_task(
            polling_inner,
            poll_interval,
            refresh_lead,
            role,
            prolongation_floor,
            force_refresh_rx,
            cancel,
        ));

        Self { inner }
    }

    /// Return the cached Bearer token whenever it has not *actually* expired —
    /// the provider's sole bearer accessor (b.d6b, b.mq8).
    ///
    /// The per-instance refresh lead is ignored: a token whose `expires_on` is
    /// one second away is still a working credential, so it is served; only "no
    /// token cached" and "cached token already expired" yield `None`.
    ///
    /// **Why the lead is not consulted.** The Valkey `accessTokenFile` path
    /// installs a fred `CredentialProvider` whose `fetch()` must produce a
    /// password for every `AUTH` / `HELLO`. Under the withholding rule this
    /// accessor replaced, the ≥ 3 min lead turned into a ≥ 3 min window in
    /// which every connection attempt failed `AUTH` outright, tripping the
    /// Valkey readiness probe, while a perfectly valid token sat in the cache.
    /// The lead is refresh-cadence guidance — "fetch the next token early" —
    /// not an instruction to stop using the current one. Failure-mode symmetry
    /// makes that safe: if the backing store really would reject a near-expiry
    /// token, sending it fails `AUTH` exactly as withholding it did, so this is
    /// never worse. The lead keeps its real job — driving the stale-token
    /// phase's proactive re-reads in [`run_polling_task`] — untouched.
    ///
    /// b.e49 carried the same argument to the other two consumers, so this is
    /// the accessor for *every* production reader: the storage-api checkpoint
    /// adapter (where withholding guaranteed an unauthenticated 401 once per
    /// token lifetime, and post-b.33e that reached DR-restore clients as a
    /// non-retriable `INTERNAL`) and the health supervisor (where it flipped a
    /// replica `NOT_SERVING` — a routing actuator — while it held a working
    /// credential). With no consumer left, the withholding accessor itself was
    /// deleted in b.mq8; do not reintroduce one.
    ///
    /// Synchronous (non-`async`) by design — the storage-api checkpoint hot
    /// path must not yield. Acquires a `std::sync::RwLock` read guard;
    /// uncontended reads take a single atomic compare-exchange. See
    /// [`Inner::cached_expires_on`] for why the `.read().ok()?` poisoned-lock
    /// posture is a total-function default rather than a graceful-degradation
    /// path.
    pub fn cached_bearer_if_unexpired(&self) -> Option<String> {
        let guard = self.inner.cache.read().ok()?;
        let cached = guard.as_ref()?;
        if cached.expires_on > Utc::now() {
            Some(cached.access_token.clone())
        } else {
            None
        }
    }

    /// `expires_on` of the currently cached token, or `None` when the cache has
    /// never been populated by a successful parse.
    ///
    /// One-line delegate to [`Inner::cached_expires_on`] — the single source of
    /// truth for token expiry. Deliberately not a second lock-discipline site
    /// and deliberately not backed by a shadow copy, so this can never disagree
    /// with [`Self::cached_bearer_if_unexpired`] about which token is live.
    ///
    /// Projects only the `Copy` timestamp, never the token bytes
    /// (bearer-bytes-never-logged, SR-6.2).
    pub fn cached_token_expires_on(&self) -> Option<DateTime<Utc>> {
        self.inner.cached_expires_on()
    }

    /// The non-secret [`TokenClaims`] of the currently cached token, or `None`
    /// when nothing is cached or the cached token is not a decodable JWT
    /// (b.cbe).
    ///
    /// One-line delegate to [`Inner::cached_claims`], mirroring
    /// [`Self::cached_token_expires_on`]'s relationship to
    /// [`Inner::cached_expires_on`] — the cache stays the single source of
    /// truth and this cannot disagree with the bearer accessors about which
    /// token is live.
    ///
    /// Consumed by the Valkey `accessTokenFile` auth diagnostic
    /// (`storage::client`), which turns the opaque server-side
    /// `WRONGPASS` / `NOAUTH` into a cause-specific operator hint. Projects
    /// only the named non-secret claims, never the token bytes
    /// (bearer-bytes-never-logged, SR-6.2).
    pub fn cached_token_claims(&self) -> Option<TokenClaims> {
        self.inner.cached_claims()
    }

    /// Trigger an immediate re-read of the token file via the polling task
    /// and await the outcome.
    ///
    /// Used by the Epic 5 outbound storage-api adapter when an upstream
    /// 401 indicates the cached token has gone stale and the operator
    /// wants to fail fast on the retry instead of waiting up to
    /// `service_token_poll_interval_secs` for the next poll. The polling
    /// task is the sole writer of the cache so request-coalescing is
    /// implicit: many concurrent `force_refresh()` calls funnel into the
    /// task's `select!`, each gets its own oneshot reply, and the task
    /// runs `read_and_update` once per dequeued request.
    pub async fn force_refresh(&self) -> Result<(), ServiceTokenError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self.inner.force_refresh_tx.send(reply_tx).await.is_err() {
            // Polling task has shut down — channel rx dropped.
            return Err(ServiceTokenError::Cancelled);
        }
        match reply_rx.await {
            Ok(result) => result,
            Err(_recv_err) => {
                // Polling task dropped the oneshot mid-flight (cancelled
                // after dequeuing the request). Surface as `Cancelled`
                // rather than an opaque recv-error per the variant's
                // docstring.
                Err(ServiceTokenError::Cancelled)
            }
        }
    }
}

/// Read-and-update routine shared by the polling tick and the
/// `force_refresh()` handler. `caller` controls the NotFound branch and the
/// parse-failure emission pacing only — see [`ReadCaller`] for the
/// rationale; everything else is identical.
async fn read_and_update(inner: &Inner, caller: ReadCaller<'_>) -> Result<(), ServiceTokenError> {
    // Atomic-rename semantics on the writer side (credential-provisioner
    // sidecar writes a tempfile + renames into place) guarantee that this
    // single `read_to_string` either observes the previous full file or
    // the new full file — never a torn / partial read. See SDD SR-6.1 for
    // the writer contract.
    let contents = match tokio::fs::read_to_string(&inner.path).await {
        Ok(s) => s,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return match caller {
                // Polling-task NotFound is the SR-6.2 startup-absent path
                // — sidecar may not yet have written the file. Stay silent
                // (cache untouched, `cached_bearer_if_unexpired()` continues to
                // return `None` until the first successful poll). Identical for
                // gated and ungated polling reads: the gate paces the two
                // failure branches below, never this one — which emits
                // nothing to pace.
                ReadCaller::Polling(_) => Ok(()),
                // `force_refresh()` callers (the Epic 5 401-retry path)
                // need the NotFound signal so they can distinguish
                // "refresh failed" from "refresh succeeded with empty
                // cache".
                ReadCaller::ForceRefresh => Err(ServiceTokenError::Io(e)),
            };
        }
        Err(e) => {
            // Non-NotFound I/O error (EACCES, ENOTDIR, EIO, …): log at WARN
            // (no file content), then surface to both callers. Polling-tick
            // callers don't have a place to surface to — `run_polling_task`
            // just discards the result — but `force_refresh()` callers will
            // see it.
            //
            // Emission is caller-scoped exactly as on the parse branch below:
            // a present-but-unreadable file (wrong mode, wrong-type path, bad
            // mount) is a startup-persistent fault, so the cold-start
            // fast-poll arm's up-to-4×/s re-reads would otherwise amplify one
            // operator-visible fault into ~80 identical WARNs across the
            // startup window. Sharing the single gate instance with the parse
            // branch cannot starve either class: one read either fails I/O
            // *or* reaches the parser, never both, so the two branches never
            // compete for the same emission slot within a read — the shared
            // gate just yields "at most one startup-phase failure WARN per
            // `poll_interval`" over whichever failure the file currently
            // exhibits. No metric here:
            // `live_edit_service_token_parse_errors_total` counts parse
            // failures only and deliberately does not tick on I/O errors.
            let emit = match caller {
                ReadCaller::ForceRefresh | ReadCaller::Polling(None) => true,
                ReadCaller::Polling(Some(gate)) => gate.allow(),
            };
            if emit {
                tracing::warn!(
                    kind = ?e.kind(),
                    "failed to read service token file (file content is never logged)"
                );
            }
            return Err(ServiceTokenError::Io(e));
        }
    };

    // Parse JSON. On failure, tick the metric, log WARN with NO token
    // bytes (the bearer-bytes-never-logged invariant — `serde_json`'s
    // error messages include byte offsets but not raw bytes; we belt-and-
    // suspender by stripping the error to a category prefix only), and
    // return Err. The cache is NOT touched — previous-good-cache invariant.
    let parsed: TokenFileJson = match serde_json::from_str::<TokenFileJson>(&contents) {
        Ok(p) => p,
        Err(e) => {
            // Emission is caller-scoped. `force_refresh()` reads and
            // steady-state polling reads keep exact
            // one-WARN-and-one-increment-per-read semantics — in steady
            // state `poll_interval` is already the rate limit, so the
            // documented "every parse failure ticks the counter (bounded
            // only by read cadence)" contract holds 1:1. Only the
            // cold-start fast-poll reads carry a gate, so a single
            // malformed file can't amplify into a startup WARN storm and an
            // equally inflated parse-error counter. Both the log line and
            // the metric move together — a counter that ticks while the WARN
            // is suppressed would be just as misleading.
            let emit = match caller {
                ReadCaller::ForceRefresh | ReadCaller::Polling(None) => true,
                ReadCaller::Polling(Some(gate)) => gate.allow(),
            };
            if emit {
                inner.metrics.inc_service_token_parse_error();
                // `serde_json::Error::classify` returns one of `Io`, `Syntax`,
                // `Data`, `Eof`. We log that category only — no byte slice,
                // no error message that could echo input. SR-6.2.
                tracing::warn!(
                    category = ?e.classify(),
                    "failed to parse service token file (token bytes are never logged)"
                );
            }
            return Err(ServiceTokenError::Parse(format!(
                "failed to parse token file: {:?}",
                e.classify()
            )));
        }
    };

    // Write the new cache entry. `expect` on the lock-poisoned path is
    // acceptable here — a poisoned RwLock means a previous writer panicked,
    // which is a programmer bug worth crashing on.
    // Project the non-secret identity claims once, here, at cache-write time
    // (b.cbe). Doing it on the polling task keeps every JSON decode off
    // `fetch()` / `cached_bearer_if_unexpired()`, which are synchronous and on
    // the Valkey AUTH and outbound-checkpoint hot paths; a token that is not a
    // JWT yields `None` silently and costs a three-way `split` and nothing else.
    let claims = decode_token_claims(&parsed.access_token);
    let new_cached = CachedToken {
        access_token: parsed.access_token,
        expires_on: parsed.expires_on,
        claims,
    };
    let mut write_guard = inner
        .cache
        .write()
        .expect("ServiceTokenProvider cache RwLock poisoned");
    *write_guard = Some(new_cached);
    Ok(())
}

/// `refresh_lead` — per-instance **proactive re-read lead**: how far ahead of a
/// cached token's `expires_on` this task starts hunting for its replacement. One
/// `poll_interval` is added to it below to get the stale-token phase's entry
/// lead (`stale_entry_lead`), so the fast cadence is already running by the time
/// the sidecar's rotation is due. It is also reported as `refresh_lead_secs` on
/// all three of this task's lead-bearing log sites: the b.is1 rotation-headroom
/// WARN at the loop head, the stale-token phase-entry INFO, and the retirement
/// WARN.
///
/// **Driving that phase is its whole job.** It is not a validity floor and no
/// accessor consults it: every production consumer (Valkey credential provider,
/// b.d6b; storage-api checkpoint adapter and health supervisor, b.e49) reads
/// [`ServiceTokenProvider::cached_bearer_if_unexpired`], which serves any cached
/// token that has not actually expired. The accessor that once applied the lead
/// as a withholding threshold was deleted in b.mq8. Withholding a still-valid
/// credential was never what the "refresh ≥ 3 min before expiry" guidance asked
/// for.
///
/// Defaults to [`DEFAULT_REFRESH_LEAD`] (60 s) via
/// [`ServiceTokenProvider::new`], which is what the storage-api instance uses;
/// the access-token-file Valkey path constructs a second instance with a ≥ 3 min
/// lead via [`ServiceTokenProvider::new_with_refresh_lead`] (Azure asks that a
/// fresh token be *fetched* ≥ 3 min before expiry, SDD SR-4.1).
///
/// `prolongation_floor` — recommended minimum **prolongation headroom** for this
/// instance, or `None` when the instance has no such recommendation (b.is1).
///
/// Headroom is measured per observed rotation as `old_expiry − pickup_instant`
/// (see the loop head below). When it falls below this floor, one WARN naming
/// the two operator levers is emitted. When it is `None` the headroom is still
/// recorded as a gauge; only the WARN is suppressed.
///
/// **Set by the wiring layer, not derived here.** The floor is a property of
/// what consumes the credential, and `auth` must not gain a dependency edge on
/// `storage` to learn it: the Valkey instance's floor is the rotation fan-out
/// ceiling (`VALKEY_TOKEN_ROTATION_FANOUT_CEILING`), passed in from `main.rs`,
/// because a re-AUTH firing later than the headroom cannot carry the fresh
/// credential onto a live connection. The storage-api instance passes
/// `None`: its consumer re-reads on an upstream 401 and holds no long-lived
/// credentialed connection, so there is no band to stay ahead of.
///
/// Both are task parameters rather than [`Inner`] fields for the same reason
/// `role` is (see [`ServiceTokenProvider::spawn`]): their only readers are
/// inside this task.
async fn run_polling_task(
    inner: Arc<Inner>,
    poll_interval: Duration,
    refresh_lead: chrono::Duration,
    role: &'static str,
    prolongation_floor: Option<chrono::Duration>,
    mut force_refresh_rx: mpsc::Receiver<oneshot::Sender<Result<(), ServiceTokenError>>>,
    cancel: CancellationToken,
) {
    let started_at = tokio::time::Instant::now();

    // Paces both fast phases' read-failure WARNs — I/O-error and
    // parse-failure branches alike, plus the parse branch's metric — since
    // they re-read up to 4×/s. Handed to the t≈0 read, the cold-start
    // fast-poll arm and the stale-token fast-poll arm; the steady-state arm
    // reads ungated, since `poll_interval` is already its rate limit. One
    // shared instance across both fast arms: they are mutually exclusive by
    // construction, so a per-phase budget would buy nothing. Task-local: one
    // gate per provider, never shared, and it dies with the task.
    let parse_failure_gate = ParseFailureGate::new(poll_interval);

    let mut interval = tokio::time::interval(poll_interval);
    // Skip catch-up bursts after a long pause — if the tokio runtime
    // stalled for several poll intervals, we don't want to fire N reads
    // in rapid succession; just resume at the next nominal tick.
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    // Already-cancelled-at-spawn short circuit. `cancel.cancelled()` is only
    // consulted inside the `select!` below, so without this the explicit
    // t≈0 read would still hit the filesystem for a task that has been told
    // to shut down (shutdown racing startup, or a test that cancels before
    // the spawned task is first polled).
    if cancel.is_cancelled() {
        tracing::debug!("service token polling task cancelled before initial read");
        return;
    }

    // Immediate t≈0 read, performed explicitly rather than by consuming
    // `interval`'s tick #0. `tokio::time::interval` completes tick #0 at
    // construction, so doing the read here preserves the warm-restart
    // cadence byte-for-byte (first read at task start, next read one
    // `poll_interval` later, via the `interval.reset()` at the warm-up
    // transition below).
    // Gated: this read stands in for the fast phase's tick #0, so it shares
    // the fast phase's amplification budget.
    let _ = read_and_update(&inner, ReadCaller::Polling(Some(&parse_failure_gate))).await;

    // Cold-start fast-poll phase (see `FAST_POLL_RETRY_INTERVAL`). Active
    // until either the cache holds its first token or
    // `STARTUP_FAST_POLL_WINDOW` elapses. Clamp to `poll_interval` so a
    // shorter operator-configured interval is never slowed by this phase;
    // the stale-token phase below shares the same clamped cadence.
    let fast_poll_cadence = FAST_POLL_RETRY_INTERVAL.min(poll_interval);
    let mut fast_poll_active = true;

    // Entry lead for the stale-token phase (see `StalePhase`):
    // `refresh_lead + poll_interval`, i.e. SDD §15.4's left-hand sum. The
    // per-instance lead is used, never the `DEFAULT_REFRESH_LEAD` constant. A
    // `poll_interval` beyond chrono's representable range is not reachable
    // from `Config` (u64 seconds parsed into a `Duration`, but the process
    // would have to be configured with a ~292-million-year interval); fall
    // back to the lead alone rather than panicking if it ever is. Both
    // fallible steps are covered: the conversion (`from_std`) and the sum
    // (`Duration::add` panics on overflow, so `checked_add`).
    let stale_entry_lead = {
        let lead =
            chrono::Duration::from_std(poll_interval).unwrap_or_else(|_| chrono::Duration::zero());
        refresh_lead.checked_add(&lead).unwrap_or(refresh_lead)
    };
    let mut stale_phase = StalePhase::Idle;
    // When the current `StalePhase::Active` was entered — the `fast_polled_secs`
    // field of the retirement WARN. Only read while `Active`.
    let mut stale_entered_at = tokio::time::Instant::now();

    // Cached expiry as of the previous loop-head pass — the substrate for the
    // rotation-headroom observation below (b.is1).
    //
    // Deliberately a task-local mirror rather than `StalePhase::keyed_expiry()`:
    // the phase is armed only when a token nears expiry with no replacement
    // cached, so in the healthy regime (the sidecar rotates well ahead of the
    // stale-entry lead) it stays `Idle` and would report `None` across every
    // rotation there is to measure — precisely the rotations the prolongation
    // recommendation is about. This mirror is updated on every pass regardless
    // of phase, so it sees them all.
    //
    // Holds the `Copy` timestamp only, never token bytes (SR-6.2), and never
    // becomes the authority on which token is live — that stays
    // `Inner::cached_expires_on()`.
    let mut last_cached_expiry: Option<DateTime<Utc>> = None;

    // Long-lived `Interval`s rather than `sleep(...)` futures built inside the
    // `select!`. A `select!` arm's future is constructed fresh on every loop
    // iteration, so a `sleep` arm restarts its countdown whenever any *other*
    // arm wins — and a sustained `force_refresh()` stream is exactly that:
    // the Epic 5 storage-api adapter force-refreshes on every upstream 401,
    // which is most likely during this same cold-start window. That could
    // starve the fast poll indefinitely. `Interval::tick` is cancel-safe and
    // holds its deadline across iterations, so a losing branch costs nothing.
    // The argument applies verbatim to the stale-token arm: a 401 storm is at
    // least as likely while the cached token is expiring as at cold start.
    //
    // `interval_at(now + period, …)`: plain `interval()` completes tick #0
    // immediately, and the explicit read above already stands in for it. The
    // stale interval is `reset()` on phase entry, so its pre-entry deadline is
    // never observed.
    let mut startup_interval = tokio::time::interval_at(
        tokio::time::Instant::now() + fast_poll_cadence,
        fast_poll_cadence,
    );
    startup_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut stale_interval = tokio::time::interval_at(
        tokio::time::Instant::now() + fast_poll_cadence,
        fast_poll_cadence,
    );
    stale_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    loop {
        // Warm-up transition, checked at the loop head so it covers every
        // path that can populate the cache — the immediate read above, a
        // fast-poll read, and a `force_refresh()` request that beat the
        // poller to it. `interval.reset()` starts the steady-state cadence
        // from the successful load rather than from process start (and
        // discards `interval`'s still-pending tick #0, which the explicit
        // read above stands in for).
        if fast_poll_active && inner.has_cached_token() {
            fast_poll_active = false;
            interval.reset();
        }

        // Window bound, also at the loop head rather than inside the
        // fast-poll arm: the bound must hold no matter which arm fires. If
        // it lived in the timer arm, a `force_refresh()` stream that keeps
        // winning the `select!` would leave the phase un-retired and the
        // one-shot WARN unemitted for the process lifetime — the same
        // starvation the `Interval` above defends against, but silent.
        // Ordered after the cache check so a successful load retires the
        // phase without WARNing. Bearer-bytes-never-logged (SR-6.2): role,
        // path and elapsed only, never file contents or token material.
        if fast_poll_active && started_at.elapsed() >= STARTUP_FAST_POLL_WINDOW {
            tracing::warn!(
                role,
                path = %inner.path.display(),
                elapsed_secs = started_at.elapsed().as_secs(),
                "service token file still absent or unparseable after startup fast-poll \
                 window; falling back to the steady-state poll interval (file content is \
                 never logged)"
            );
            fast_poll_active = false;
            interval.reset();
        }

        // One snapshot of the cached expiry per loop-head pass, shared by the
        // rotation-headroom observation and the stale-phase transitions below.
        // Nothing between here and the `select!` awaits, and the polling task is
        // the cache's only writer, so a single read is coherent across the whole
        // head. Read unconditionally (not under `!fast_poll_active`) so the
        // rotation observation is independent of both fast phases; the cost is
        // one extra uncontended read-lock acquisition per cold-start iteration.
        let cached_expiry = inner.cached_expires_on();

        // Rotation prolongation-headroom observation (b.is1).
        //
        // A `Some(prev) → Some(curr)` change of the cached expiry is the
        // pickup of a *replacement* token (the same "a fresher token appeared"
        // signal the stale-phase reset below keys on — not file mtime, not a
        // byte comparison; a parse failure leaves `expires_on` untouched by the
        // previous-good-cache invariant and so does not look like a rotation).
        //
        // `headroom = prev − now` is the window in which a periodic re-AUTH can
        // still carry the fresh credential onto a connection that was AUTHed
        // with the old token: after `prev` the backing store drops that
        // connection regardless. Measuring from `now` rather than from a
        // configured lead folds the real pickup delay (poll cadence, sidecar
        // write time, scheduling) into the number, which makes it strictly
        // tighter than any static comparison of the two leads — and it is the
        // only form available at all, since the sidecar's refresh lead is not a
        // live-edit config value and cannot be checked at startup.
        //
        // Signed: negative means the replacement landed after the old token had
        // already expired, i.e. the whole pool was dropped and re-established
        // rather than prolonged.
        //
        // SR-6.2: a derived duration and, in the WARN, configured seconds —
        // never the `expires_on` timestamps themselves, never token bytes.
        if let Some(curr) = cached_expiry {
            if let Some(prev) = last_cached_expiry.replace(curr) {
                if prev != curr {
                    let headroom = prev - Utc::now();
                    inner
                        .metrics
                        .set_service_token_refresh_headroom_seconds(role, headroom.num_seconds());

                    // One WARN per observed rotation, and only for an instance
                    // that carries a floor (see this function's
                    // `prolongation_floor` parameter) — the storage-api instance
                    // has no re-AUTH band to stay ahead of, so it records the
                    // gauge and never warns.
                    if let Some(floor) = prolongation_floor {
                        if headroom < floor {
                            // Second lever, keyed on the role. Only the Valkey
                            // instance's lead is
                            // `LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS`; the
                            // constructor that takes a floor is `pub`, and
                            // nothing structurally stops a future non-Valkey
                            // instance from carrying one, so name the env var
                            // only where it is the right one and stay generic
                            // otherwise rather than sending an operator to a
                            // variable that does not drive this instance.
                            let lead_lever = if role == ServiceTokenProvider::ROLE_VALKEY {
                                "LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS"
                            } else {
                                "this instance's refresh lead"
                            };
                            // The target is stated as the value already in this
                            // line, not as "the sidecar's lead minus one poll
                            // interval": the operator cannot read the sidecar's
                            // lead from anywhere, whereas
                            // `observed_headroom_secs >= sidecar_lead − poll`
                            // always holds (pickup lags the sidecar's write by
                            // at most one steady poll), so the printed value is
                            // a self-sufficient — and never over-stated —
                            // stand-in for that target.
                            tracing::warn!(
                                role,
                                path = %inner.path.display(),
                                observed_headroom_secs = headroom.num_seconds(),
                                recommended_floor_secs = floor.num_seconds(),
                                refresh_lead_secs = refresh_lead.num_seconds(),
                                "replacement service token was picked up with less headroom \
                                 before the previous token's expiry than the periodic re-AUTH \
                                 interval needs to prolong a live connection across the \
                                 rotation; connections AUTHed with the old token will be \
                                 dropped at its expiry and immediately re-established. Either \
                                 raise the sidecar's own refresh lead, or raise {} to at least \
                                 the observed_headroom_secs value in this line so pickup rides \
                                 the fast-poll phase and the poll term drops out (token \
                                 bytes are never logged)",
                                lead_lever
                            );
                        }
                    }
                }
            }
        }

        // Stale-token phase transitions (see `StalePhase`), evaluated at the
        // loop head for the same reason the two checks above are: every path
        // that can change the cache — a fast-poll read, a steady read, a
        // `force_refresh()` that beat the poller — funnels through here, and a
        // check living inside a timer arm could be starved by a sustained
        // `force_refresh()` stream.
        //
        // Guarded on `!fast_poll_active` so the two fast phases are mutually
        // exclusive by construction: their `select!` guards can then never
        // both be live, and the shared `parse_failure_gate` bounds whichever
        // one is running.
        //
        // Cost on a steady-state iteration: one `Utc::now()` and one
        // uncontended read-lock acquisition. `cached_bearer_if_unexpired()` is
        // untouched — nothing here signals the hot path.
        if !fast_poll_active {
            // A cached expiry different from the one the phase is keyed on
            // means a *different token* is now cached — which is exactly the
            // "a fresher token appeared" signal (not file mtime, not a byte
            // comparison). Reset to `Idle` so the new token is evaluated on
            // its own merits, and so a retired expiry can never be re-armed.
            // A parse failure leaves `expires_on` untouched by the
            // previous-good-cache invariant, so it correctly does NOT look
            // like a fresher token here.
            if let Some(keyed) = stale_phase.keyed_expiry() {
                if cached_expiry != Some(keyed) {
                    if matches!(stale_phase, StalePhase::Active(_)) {
                        // Retirement #1 of 2 — the happy one: the replacement
                        // landed. Silent (a recovering sidecar is not an
                        // operator event), and `interval.reset()` restarts the
                        // steady cadence from the load rather than from the
                        // tick that preceded phase entry — same reason the
                        // cold-start warm-up transition above resets.
                        interval.reset();
                    }
                    stale_phase = StalePhase::Idle;
                }
            }

            if let Some(expires_on) = cached_expiry {
                let now = Utc::now();

                // Time-to-expiry gauge (b.d6b), recorded from the values this
                // loop head already computes. Negative once the cached token
                // has expired, which is the operator-meaningful part: the
                // series crosses zero exactly when the credential dies.
                //
                // SR-6.2: a *derived duration*, never the `expires_on`
                // timestamp and never token material — the same granularity
                // the phase-entry INFO below already logs as `secs_to_expiry`.
                // Recorded on every loop-head pass rather than on transitions
                // so the series stays live between them.
                inner
                    .metrics
                    .set_service_token_expires_in_seconds(role, (expires_on - now).num_seconds());

                if expires_on <= now {
                    // Retirement #2 of 2: the cached token expired with no
                    // replacement. Fires once per distinct cached expiry —
                    // including the warm-restart-into-an-already-expired-token
                    // case, where the phase was never entered at all
                    // (`fast_polled_secs = 0`), because that is the same
                    // operator-visible fault. `RetiredFor` refuses re-entry
                    // for this expiry, which is what makes it one-shot.
                    //
                    // Message deliberately distinct from the cold-start
                    // window WARN above: that one is substring-matched by the
                    // b.o98 window tests, which assert an exact count.
                    //
                    // SR-6.2: role, path and derived durations only — never
                    // `expires_on` itself, never file contents, never token
                    // material.
                    if !matches!(stale_phase, StalePhase::RetiredFor { .. }) {
                        let fast_polled_secs = match stale_phase {
                            StalePhase::Active(_) => stale_entered_at.elapsed().as_secs(),
                            _ => 0,
                        };
                        tracing::warn!(
                            role,
                            path = %inner.path.display(),
                            refresh_lead_secs = refresh_lead.num_seconds(),
                            fast_polled_secs,
                            "cached service token expired before a refreshed one appeared; \
                             leaving the stale-token fast-poll phase and continuing to \
                             fast-poll over the bounded post-expiry window before reverting \
                             to the steady poll interval (token bytes are never logged)"
                        );
                        if matches!(stale_phase, StalePhase::Active(_)) {
                            // `reset_immediately()`, not `reset()`. Throughout
                            // the `Active` window the steady arm was
                            // guard-disabled, so `interval`'s deadline is
                            // stale/elapsed and carries no useful information;
                            // a plain `reset()` would pin the next steady read
                            // at `T_expiry + poll_interval` — a guaranteed
                            // worst-case delay at exactly the moment
                            // `cached_bearer_if_unexpired()` starts returning
                            // `None` to every caller. Fire one catch-up read at
                            // retirement instead and re-anchor the steady grid
                            // there, so the replacement token is picked up as
                            // soon as it lands rather than a full interval
                            // later. (Contrast retirement #1 above, where a
                            // replacement is already cached and restarting the
                            // cadence from that load is the right anchor.)
                            //
                            // Still load-bearing after b.d6b even though the
                            // fast arm now stays live across the post-expiry
                            // window: the window is bounded, and when it closes
                            // this re-anchored grid is what the steady arm
                            // resumes on.
                            interval.reset_immediately();
                        }
                        // b.d6b: anchor the bounded post-expiry fast-poll
                        // window at the retirement instant. `wants_fast_poll()`
                        // reads `at`, keeping the stale arm live — and the
                        // steady arm guard-disabled — for
                        // `STARTUP_FAST_POLL_WINDOW` past this point, so
                        // recovery from a *total* credential outage rides the
                        // 250 ms cadence instead of reverting to the steady
                        // poll at the worst possible moment.
                        //
                        // No `stale_interval.reset()` here, deliberately, on
                        // either path into `RetiredFor`:
                        //
                        // - From `Active`: the interval is already ticking at
                        //   the fast cadence and must not lose its deadline.
                        // - From `Idle`: the arm was guard-disabled, so the
                        //   interval kept its own schedule in the dark and its
                        //   deadline is whatever that schedule left. On a warm
                        //   restart straight into an already-expired token the
                        //   interval was constructed microseconds ago as
                        //   `interval_at(now + fast_poll_cadence, …)`, so the
                        //   first post-expiry read lands one ~250 ms cadence
                        //   in — fast enough that resetting would buy nothing.
                        //   On a long-running process that reaches `RetiredFor`
                        //   from `Idle` (the dark interval has been ticking
                        //   unobserved for far longer than one cadence) the
                        //   deadline is elapsed and `MissedTickBehavior::Skip`
                        //   fires one catch-up read immediately.
                        //
                        // Either way the arm is reading at the fast cadence
                        // within one cadence of retirement, which is the
                        // behavior a process with no usable credential needs.
                        stale_phase = StalePhase::RetiredFor {
                            expiry: expires_on,
                            at: tokio::time::Instant::now(),
                        };
                    }
                } else if matches!(stale_phase, StalePhase::Idle)
                    && now
                        .checked_add_signed(stale_entry_lead)
                        .is_some_and(|entry_at| expires_on <= entry_at)
                {
                    // `checked_add_signed`, not `+`: `DateTime<Utc> + TimeDelta`
                    // panics on a sum outside chrono's `DateTime` range, and
                    // this task is `tokio::spawn`ed with its `JoinHandle`
                    // dropped — a panic here is *swallowed*, silently killing
                    // the poller and `force_refresh()` with it while the pod
                    // coasts to credential death with no log line. `None` (the
                    // lead is so large the sum is unrepresentable) means "not
                    // inside the window", so the phase simply stays `Idle` and
                    // the steady poll continues. `Config` rejects such leads at
                    // startup (`VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS`), but this
                    // must not depend on that: the constructors are `pub` and
                    // take a caller-chosen `chrono::Duration` (b.is1).
                    //
                    // Entry. `+ poll_interval` on top of the refresh lead arms
                    // the phase one steady tick BEFORE the lead window opens —
                    // see `StalePhase` for why `+ lead` alone makes this a
                    // no-op on the deployments that need it.
                    //
                    // INFO, not DEBUG: it marks the one window in which the
                    // provider is running its 250 ms cadence rather than the
                    // steady poll, so an operator reading the logs around a
                    // rotation can tell "the replacement was late enough that we
                    // went hunting for it" from "the rotation was picked up on an
                    // ordinary tick". Entry alone is NOT a misconfiguration
                    // signal, even though post-b.7td the two thresholds now
                    // coincide. Entry fires whenever the sidecar's lead is under
                    // `refresh_lead + poll_interval` — 195 s at defaults — and
                    // the operator-facing sidecar-lead recommendation is now
                    // that same 195 s, because b.7td's rotation gate cut the
                    // prolongation floor to `VALKEY_TOKEN_ROTATION_FANOUT_CEILING`
                    // (180 s) plus one token-file poll. (Pre-b.7td the
                    // recommendation was ~375 s and entry sat well inside it.)
                    // The coincidence is a defaults-only artifact — it breaks as
                    // soon as `LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS` or
                    // `LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS` moves — and
                    // the two still measure different things: this line fires on
                    // the *observation* that no replacement had landed yet, on
                    // every rotation at or under the sum, whether or not the
                    // replacement then arrived with plenty of headroom. The
                    // prolongation signal remains the per-rotation headroom WARN
                    // at the loop head (b.is1), which measures the realized
                    // headroom against the floor.
                    //
                    // SR-6.2: derived durations only, never the `expires_on`
                    // timestamp.
                    tracing::info!(
                        role,
                        path = %inner.path.display(),
                        secs_to_expiry = (expires_on - now).num_seconds(),
                        refresh_lead_secs = refresh_lead.num_seconds(),
                        "cached service token is approaching expiry with no replacement yet; \
                         entering the stale-token fast-poll phase (token bytes are never \
                         logged)"
                    );
                    stale_phase = StalePhase::Active(expires_on);
                    stale_entered_at = tokio::time::Instant::now();
                    // First stale tick one cadence from entry, not from
                    // whenever this `Interval` was last read.
                    stale_interval.reset();
                }
            }
        }

        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                tracing::debug!("service token polling task cancelled");
                return;
            }
            Some(reply_tx) = force_refresh_rx.recv() => {
                let result = read_and_update(&inner, ReadCaller::ForceRefresh).await;
                // Ignore send error: caller dropped the oneshot rx. The
                // cache update (if any) is still durable, so the next
                // `cached_bearer_if_unexpired()` reads the fresh token.
                let _ = reply_tx.send(result);
            }
            _ = startup_interval.tick(), if fast_poll_active => {
                // Cold start: no token has loaded yet. Same tolerated-failure
                // contract as the steady-state arm below, just at a much
                // shorter cadence so a token the sidecar writes shortly
                // after boot is cached sub-second. Retirement of this phase
                // (success or window expiry) is handled at the loop head.
                // Gated: this is the cadence the gate exists to bound.
                let caller = ReadCaller::Polling(Some(&parse_failure_gate));
                let _ = read_and_update(&inner, caller).await;
            }
            _ = stale_interval.tick(), if stale_phase.wants_fast_poll() => {
                // Warmed-up cache whose token is inside
                // `refresh_lead + poll_interval` of expiry (`Active`), or has
                // just expired outright and is inside the bounded post-expiry
                // window (`RetiredFor`, b.d6b): re-read at the fast cadence so
                // a late sidecar rotation is picked up in ~250 ms rather than
                // on the next steady tick. Retirement (fresher token, or the
                // cached token actually expiring) is handled at the loop head.
                // Gated with the SAME instance the cold-start arm uses — the
                // two arms never run concurrently, so one `poll_interval`
                // budget covers whichever is live.
                let caller = ReadCaller::Polling(Some(&parse_failure_gate));
                let _ = read_and_update(&inner, caller).await;
            }
            // Exact complement of the arm above, so the two are mutually
            // exclusive and exactly one of them is live whenever the cold-start
            // phase is done. Written from the same `wants_fast_poll()`
            // predicate rather than re-spelled, so widening that predicate can
            // never leave both arms live (double-cadence) or both dark
            // (polling stops).
            _ = interval.tick(), if !fast_poll_active && !stale_phase.wants_fast_poll() => {
                // Polling-tick failures are tolerated — non-NotFound I/O
                // errors and parse errors already logged + metric-ticked
                // inside `read_and_update`. Discard the Result here so a
                // transient sidecar hiccup doesn't kill the polling task.
                //
                // Ungated (`None`): `poll_interval` is already the rate
                // limit here, so emission is exactly 1:1 with failed reads
                // on both failure branches. A gate at the same nominal
                // period would stamp at emission time (post-read) against
                // tick-time deadlines, and per-read filesystem jitter would
                // silently drop a fraction of the WARNs and parse-error
                // increments.
                let _ = read_and_update(&inner, ReadCaller::Polling(None)).await;
            }
        }
    }
}

/// Emit the V1 service-identity startup warning, if applicable.
///
/// Called exactly once from `main.rs`, right after `Config::from_env`
/// returns and before any dependency step (port bind, backing-store
/// connect). Factored into a `pub` helper so integration tests (which cannot
/// reach `main.rs` from the integration-test binary) can call it directly
/// with synthetic `Config` values and capture the emitted log output.
///
/// **SR-6.4 unsupported V1 mode.** When the operator set
/// `LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED=true`, warn that the
/// in-process OAuth client-credentials flow is not implemented in V1. The
/// flag has no other effect: outbound storage-api calls authenticate only
/// with the token read from `LIVE_EDIT_SERVICE_TOKEN_FILE`.
///
/// The `StorageApi` checkpoint adapter without `LIVE_EDIT_SERVICE_TOKEN_FILE`
/// needs no warning here: `Config::from_env` refuses that combination, so
/// the process exits with a configuration error naming the variable before
/// this helper runs.
pub fn log_service_token_startup_warnings(config: &Config) {
    if config.service_token_client_credentials_enabled {
        tracing::warn!(
            "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED=true selects the in-process OAuth \
             client-credentials flow, which is NOT implemented in V1. The setting has no other \
             effect: outbound storage-api calls are authenticated only with the token read from \
             LIVE_EDIT_SERVICE_TOKEN_FILE. Unset it or set it to false."
        );
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `ServiceTokenProvider` — Subtask `t3.fgj.dp.io.vx`.
    //!
    //! Coverage targets (parent Task `t2.fgj.dp.io` acceptance criteria):
    //!
    //! - #3: polling picks up rewritten token within one interval.
    //! - #4: `force_refresh()` reads immediately without waiting for next tick.
    //! - #5: startup-absent file tolerated, cache flips on first write.
    //! - #6: malformed JSON warn + counter increment + recovery.
    //! - #7: the refresh lead is a re-read cadence and **not** a validity
    //!   floor — a cached token inside the lead is still served, and only an
    //!   actually-expired one is withheld. Acceptance #7 originally read "60 s
    //!   pre-expiry guard, pinned at 30 s, 61 s, and already-expired"; b.e49
    //!   retired every production consumer of that guard and b.mq8 deleted the
    //!   withholding accessor that implemented it, so the surviving contract is
    //!   the one pinned here.
    //! - #9: no token bytes appear in captured log output on parse failure.
    //!
    //! Bearer accessor (b.d6b, b.mq8), under
    //! `── cached_bearer_if_unexpired()`: the provider's sole bearer accessor
    //! serves any cached token that has not actually expired — the per-instance
    //! refresh lead is ignored — and `cached_token_expires_on` is pinned
    //! empty-then-populated.
    //!
    //! Cold-start fast-poll phase (`FAST_POLL_RETRY_INTERVAL` /
    //! `STARTUP_FAST_POLL_WINDOW`), under `── Cold-start fast-poll phase`:
    //!
    //! - a token written shortly *after* construction is cached sub-second,
    //!   far inside the steady-state poll interval;
    //! - a warm restart (file already present) is cached by the explicit
    //!   t≈0 read, without relying on a fast-poll tick;
    //! - the phase is retired on "cache non-empty", not on
    //!   `cached_bearer_if_unexpired().is_some()`, so a cached token the
    //!   accessor declines to serve does not pin the *startup* phase in the
    //!   fast cadence. The observable for this is the stale-token phase's
    //!   expiry WARN, not a cadence measurement — see
    //!   `expired_cached_token_still_retires_the_fast_poll_phase` for why no
    //!   cadence measurement can discriminate the two phases here;
    //! - the phase is retired by a cache that `force_refresh()` filled, not
    //!   only by one a fast-poll read filled (the loop-head warm-up check);
    //! - cancelling mid-phase actually stops the task — the fast-poll sleep
    //!   arm does not outlive the cancel;
    //! - the phase is bounded by `STARTUP_FAST_POLL_WINDOW`: exactly one WARN
    //!   (carrying no token bytes), then the steady cadence resumes.
    //!
    //! Stale-token fast-poll phase ([`StalePhase`], b.qam), under
    //! `── Stale-token fast-poll re-entry (b.qam) ──`:
    //!
    //! - a warmed-up cache whose token is inside `expiry_skew + poll_interval`
    //!   re-enters the fast cadence and picks up a late rotation sub-second,
    //!   far inside the steady interval;
    //! - a healthy rotation (token far from expiry) never enters the phase —
    //!   the guard against an "always fast-poll" implementation;
    //! - the entry predicate is re-evaluated at later loop heads, so a token
    //!   that drifts *into* the window arms the phase;
    //! - the `+ poll_interval` term buys a full steady tick of head start: at
    //!   the arming instant the cached token still has more than `expiry_skew`
    //!   of life left, where an entry lead of `expiry_skew` alone would arm the
    //!   phase only once the token had already drifted inside the lead;
    //! - retirement on a fresher token is silent and restores the steady
    //!   cadence, whether that token arrived on a fast-poll read or through
    //!   `force_refresh()`; and when the replacement is *itself* inside the
    //!   entry window (the chronically-short-lead sidecar) the same loop
    //!   iteration that retires the phase re-arms it;
    //! - retirement at the cached token's actual expiry emits exactly one WARN
    //!   (carrying no token bytes) and does not re-arm for that same expiry,
    //!   including the warm-restart-into-an-already-expired-token case where
    //!   the phase is never entered at all;
    //! - retirement does NOT drop back to the steady cadence (b.d6b): the
    //!   250 ms arm stays live for a replacement written after the token died,
    //!   bounded by `STARTUP_FAST_POLL_WINDOW` past the retirement instant;
    //! - a *different* cached expiry does re-arm the phase — the suppression is
    //!   expiry-keyed, not latched;
    //! - the stale arm's reads are paced by the same `ParseFailureGate` the
    //!   cold-start arm uses, so a malformed file cannot amplify into a WARN /
    //!   parse-error-counter storm at the 250 ms cadence.
    //!
    //! Rotation prolongation headroom (b.is1), under `── Rotation prolongation
    //! headroom`:
    //!
    //! - a rotation is measured against the **outgoing** token's expiry, and
    //!   the recorded value is asserted in a range that contains the outgoing
    //!   token's lifetime and excludes the replacement's — measuring against
    //!   the wrong one would report a comfortable headroom on exactly the
    //!   deployments the WARN exists to catch;
    //! - the value is signed at the measurement site: a replacement picked up
    //!   after the outgoing token had already died records a negative headroom;
    //! - the WARN is gated on the instance carrying a prolongation floor, fires
    //!   once per rotation rather than once per loop head, names the floor and
    //!   the operator lever, and carries no token bytes; an instance built
    //!   without a floor (`new_with_role`, the storage-api shape) records the
    //!   gauge under its own role label and never warns;
    //! - neither a first token load nor a failed parse is a rotation — both
    //!   leave the series absent, which is what makes an absent series readable
    //!   as "no rotation observed yet";
    //! - and the whole observation **recurs**: two consecutive rotations each
    //!   move the gauge to their own measurement and each emit their own WARN.
    //!   Every other test here sees at most one rotation, so this is the only
    //!   one an implementation that latches after the first — the steady state
    //!   of any pod that outlives one token lifetime — cannot pass;
    //! - and the WARN's operator lever is keyed on the role: it names
    //!   `LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS` only on the instance that
    //!   variable actually drives, and stays generic on any other. Production
    //!   wires no non-Valkey instance with a floor today, which is precisely why
    //!   that branch needs a test — nothing else reaches it.
    //!
    //! Unrepresentable refresh leads (b.is1), under `── Unrepresentable refresh
    //! leads`: the provider's one surviving `Utc::now() + lead` site — the
    //! polling task's loop head — resolves a lead too large to add as "not
    //! inside the window" rather than panicking. (The second such site went
    //! with the withholding accessor in b.mq8.) `Config` rejects that range,
    //! but the constructors are `pub`, so the guard must hold independently of
    //! the config gate — and it fails silently (the task's `JoinHandle` is
    //! dropped), so nothing but a test observes it.
    //!
    //! **`start_paused = true` is unusable for the stale-phase tests.** The
    //! phase is driven by `chrono::Utc::now()` (wall clock), which tokio's
    //! paused clock does not virtualise, so a paused test can advance timers
    //! forever without ever moving a token toward expiry. Those tests therefore
    //! run on the real clock with **sub-second** token lifetimes and sub-second
    //! `expiry_skew`s (see [`write_token_file_in`] and [`SUB_SECOND_SKEW`]),
    //! keeping each one to roughly 0.5–2 s of wall time while
    //! [`LONG_POLL_INTERVAL`] stays the discriminator against the steady
    //! cadence.
    //!
    //! **Wall-clock shape.** Four groups, with materially different costs:
    //!
    //! - The cache / parse / `force_refresh` tests drive the 20 ms
    //!   [`TEST_POLL_INTERVAL`] and finish in tens of milliseconds each.
    //! - The rotation-headroom group also runs on the real clock — the headroom
    //!   is `Utc::now()`-derived, same constraint — but never waits out a token
    //!   lifetime: its fixtures live for tens of seconds and are replaced
    //!   immediately, so the only fixed cost is
    //!   [`NO_MORE_OUTPUT_WINDOW`] (300 ms) where a test spends a negative
    //!   observation window.
    //! - The stale-phase group runs on the real clock (see the note above) and
    //!   is the expensive one. Each of its tests waits out a real sub-second
    //!   token lifetime, and most then hold a deliberately *negative*
    //!   observation window open for a further 1–1.5 s to prove the fast
    //!   cadence is gone. Those negative windows are spent in full on every
    //!   run, pass or fail, which puts the slowest tests here in the 2–3 s
    //!   range. The positive budgets (`ENTRY_BUDGET`, `WARN_BUDGET`,
    //!   `PICKUP_BUDGET`, …) are ceilings a healthy run exits early from and
    //!   only a failing run pays.
    //! - The two `STARTUP_FAST_POLL_WINDOW` tests have to burn the whole 20 s
    //!   window, which is a module constant with no test knob, so they run on a
    //!   paused tokio clock (`#[tokio::test(start_paused = true)]`, the
    //!   convention already used in `src/grpc/metrics_layer.rs::tests`) and
    //!   advance virtual time in `FAST_POLL_RETRY_INTERVAL` steps — wall-clock
    //!   cost stays in the milliseconds. Both step through `step_paused_until`;
    //!   see its doc comment for why a virtual-step budget is a sound bound
    //!   even against the task's blocking-pool reads.
    //!
    //! Summed sequentially the module is ~20 s; under `cargo nextest`'s
    //! process-per-test parallelism it completes in ~2.5 s, which is just the
    //! slowest single test. That cost is inherent to covering a
    //! `Utc::now()`-driven phase, not incidental — a paused clock cannot
    //! substitute here — so budget for it rather than trimming the negative
    //! windows, which are the assertions that fail an "always fast-poll"
    //! implementation.
    //!
    //! Consequence for new tests: **never synchronise a positive assertion with
    //! a fixed sleep.** With ~37 tests in flight the polling task's blocking
    //! reads are not bounded by any sleep short enough to be worth writing.
    //! [`await_bearer`], [`await_cached`], [`await_parse_errors_above`] and
    //! [`poll_until`] wait on the observable; [`await_one_poll`] is reserved for
    //! negative windows, where the elapsed time *is* the assertion.
    //!
    //! Note on test isolation: each test constructs its own `Metrics` instance
    //! (which owns a fresh, self-contained metric set) so the
    //! `live_edit_service_token_parse_errors_total` counter starts at zero. We
    //! never reach into a global registry.

    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;
    use tracing_subscriber::fmt::MakeWriter;

    use super::*;

    // ── Test helpers ────────────────────────────────────────────────────────

    /// A `MakeWriter` that captures all bytes written to it into a shared
    /// buffer — mirrors the `CapturingWriter` pattern in
    /// `src/grpc/auth_interceptor.rs::tests` so the bearer-bytes-never-logged
    /// invariant test can assert on captured stdout.
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

    /// Construct a fresh `Metrics` so each test observes a clean counter.
    fn fresh_metrics() -> Arc<Metrics> {
        Arc::new(crate::observability::metrics::metrics_scrape_only(
            "service-token-test",
        ))
    }

    /// Read the unlabeled `live_edit_service_token_parse_errors_total` counter
    /// by rendering the scrape exposition. The family is absent until the
    /// first parse error (OTel counters materialise on first record and this
    /// one is not pre-initialised), so an absent series reads as 0.
    fn parse_error_count(metrics: &Metrics) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_service_token_parse_errors_total",
            &[],
        )
        .unwrap_or(0.0) as u64
    }

    /// Read `live_edit_service_token_expires_in_seconds{role=…}` out of the
    /// scrape exposition (b.d6b). `None` until the polling task first records
    /// it — OTel gauges materialise no series before their first `record()`,
    /// and this one is written at the loop head only once a token is cached.
    fn expires_in_seconds(metrics: &Metrics, role: &str) -> Option<f64> {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_service_token_expires_in_seconds",
            &[("role", role)],
        )
    }

    /// Read `live_edit_service_token_refresh_headroom_seconds{role=…}` out of
    /// the scrape exposition (b.is1). `None` until the polling task observes a
    /// token *rotation*; a first load is not one, so an absent series is the
    /// observable for "no rotation seen yet" and is what the negative tests in
    /// the `── Rotation prolongation headroom` section assert on.
    fn refresh_headroom_seconds(metrics: &Metrics, role: &str) -> Option<f64> {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_service_token_refresh_headroom_seconds",
            &[("role", role)],
        )
    }

    /// Poll interval used by every polling-task test. Deliberately far below
    /// the production default so a test that must observe a *steady* tick does
    /// not pay for one.
    const TEST_POLL_INTERVAL: Duration = Duration::from_millis(20);

    /// Slack added on top of one poll interval when *spending* an observation
    /// window (see [`await_one_poll`]).
    const POLL_SLACK: Duration = Duration::from_millis(40);

    /// Spend one poll interval plus slack.
    ///
    /// **Only for negative observations** — "the cache is still empty after N
    /// ticks", where the elapsed window *is* the assertion. Do NOT use it to
    /// synchronise before a positive assertion: the polling task's reads run on
    /// the blocking pool, so on a loaded runner (this module runs ~37 tests in
    /// parallel under nextest) 60 ms is not a bound on anything, and the
    /// test-writing guide's § Synchronization Patterns rules out exactly that
    /// use. [`await_bearer`] / [`await_cached`] are the positive-assertion
    /// equivalents.
    async fn await_one_poll() {
        tokio::time::sleep(TEST_POLL_INTERVAL + POLL_SLACK).await;
    }

    /// Budget for the "the polling task has observed an on-disk change" waits.
    ///
    /// A *ceiling*, not a delay: a healthy run satisfies these predicates in
    /// single-digit milliseconds and returns immediately, so the budget is only
    /// ever spent on a genuine hang. Sized well above any plausible
    /// blocking-pool queue delay for one small `tokio::fs` read.
    const POLL_PICKUP_BUDGET: Duration = Duration::from_millis(2000);

    /// Wait until `cached_bearer_if_unexpired()` reports `expected`.
    ///
    /// The positive-assertion replacement for `await_one_poll(); assert_eq!(…)`
    /// — same invariant, but a stalled read extends the wait instead of failing
    /// the test and misattributing the blame to the logic under test.
    async fn await_bearer(provider: &ServiceTokenProvider, expected: &str) {
        poll_until(POLL_PICKUP_BUDGET, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some(expected)
        })
        .await
        .unwrap_or_else(|| {
            panic!(
                "the polling task must cache {expected:?} within {POLL_PICKUP_BUDGET:?}; \
                 cached_bearer_if_unexpired() was still {:?}",
                provider.cached_bearer_if_unexpired()
            )
        });
    }

    /// Wait until the polling task has landed *some* token in the cache.
    ///
    /// The precondition every `…_returns_none` test needs:
    /// `cached_bearer_if_unexpired() == None` is also what an unread file looks
    /// like, so without this the assertion passes vacuously on a stalled read
    /// and proves nothing about the expiry check it exists to pin.
    async fn await_cached(provider: &ServiceTokenProvider) {
        poll_until(POLL_PICKUP_BUDGET, || provider.inner.has_cached_token())
            .await
            .expect("the polling task must land the fixture token in the cache");
    }

    /// Wait until the parse-error counter has moved past `baseline`.
    async fn await_parse_errors_above(metrics: &Metrics, baseline: u64) {
        poll_until(POLL_PICKUP_BUDGET, || parse_error_count(metrics) > baseline)
            .await
            .unwrap_or_else(|| {
                panic!(
                    "the parse-error counter must move past {baseline} within \
                     {POLL_PICKUP_BUDGET:?}; it was still {}",
                    parse_error_count(metrics)
                )
            });
    }

    /// Poll interval used by the cold-start tests: far larger than any of
    /// their assertion budgets, so an observed cache flip can never be
    /// attributed to the steady-state `interval.tick()`. This is what makes
    /// those tests genuine b.o98 regression tests — before the fix, a token
    /// that appeared after the t≈0 read could not be picked up until this
    /// interval elapsed.
    const LONG_POLL_INTERVAL: Duration = Duration::from_secs(60);

    /// Probe granularity for [`poll_until`]. Small relative to
    /// [`FAST_POLL_RETRY_INTERVAL`] so the measured pickup latency is a
    /// meaningful number rather than a quantisation artefact.
    const PROBE_INTERVAL: Duration = Duration::from_millis(5);

    /// Bounded probe loop: evaluate `predicate` until it returns `true` or
    /// `budget` elapses. Returns the elapsed wall time on success, `None` if
    /// the budget ran out.
    ///
    /// This is the test-writing guide's "wait for some background condition"
    /// pattern (poll observable state in a bounded loop) rather than a bare
    /// "sleep long enough" wait — the polling task exposes no readiness
    /// signal, so `cached_bearer_if_unexpired()` is the observable.
    ///
    /// Real-clock only: uses `std::time::Instant`, so it must not be called
    /// from a `start_paused` test.
    async fn poll_until(budget: Duration, mut predicate: impl FnMut() -> bool) -> Option<Duration> {
        let start = std::time::Instant::now();
        loop {
            if predicate() {
                return Some(start.elapsed());
            }
            if start.elapsed() >= budget {
                return None;
            }
            tokio::time::sleep(PROBE_INTERVAL).await;
        }
    }

    /// Advance a paused tokio clock in [`FAST_POLL_RETRY_INTERVAL`] steps until
    /// `predicate` holds. Returns the number of steps taken, or `None` once
    /// `virtual_step_budget` steps of virtual time have elapsed with the
    /// predicate still false.
    ///
    /// The step budget is a sound failure trigger even though the polling
    /// task's `tokio::fs` reads run on the **blocking pool**. On the
    /// current-thread runtime — the only flavour `start_paused` permits —
    /// tokio inhibits clock auto-advance for the entire life of a blocking
    /// task: from `spawn_blocking` (so queue time counts, not just execution)
    /// until the task is released, which then unparks the driver. See
    /// `runtime::blocking::schedule::BlockingSchedule::{new,release}` →
    /// `time::Clock::{inhibit_auto_advance,allow_auto_advance}`, consumed by
    /// `runtime::time::Driver::park_thread_timeout`. So while a read is in
    /// flight this loop's `sleep` parks in *real* time rather than advancing:
    /// blocking-pool contention costs wall time, never steps, and the budget
    /// cannot be burned out from under a read the pool has not run yet.
    ///
    /// What the budget does have to tolerate is work the runtime cannot see —
    /// a plain OS thread, or a task blocking the executor without yielding —
    /// which does not inhibit auto-advance and can burn thousands of virtual
    /// steps per real second. Nothing in this module does that (every wait
    /// here is a tokio timer, a `tokio::fs` read, or a channel), and the
    /// residual jitter is small: over 300 trials, including 20-way
    /// oversubscription, the step count observed past a satisfied predicate
    /// never exceeded 2 — against callers' budgets of 20 steps and up. Keep
    /// that headroom and let the step count be the sole bound, so a genuine
    /// failure reports at its virtual budget instead of after a fixed
    /// real-time penalty.
    async fn step_paused_until(
        virtual_step_budget: usize,
        mut predicate: impl FnMut() -> bool,
    ) -> Option<usize> {
        let mut steps = 0usize;
        loop {
            if predicate() {
                return Some(steps);
            }
            if steps >= virtual_step_budget {
                return None;
            }
            steps += 1;
            tokio::time::sleep(FAST_POLL_RETRY_INTERVAL).await;
        }
    }

    /// Snapshot the captured log buffer as text.
    fn captured_text(buf: &Arc<Mutex<Vec<u8>>>) -> String {
        let guard = buf.lock().unwrap();
        String::from_utf8_lossy(&guard).into_owned()
    }

    /// Assert that no contiguous 8-byte slice of `canary` appears in
    /// `output` — the bearer-bytes-never-logged invariant (SR-6.2), pinned
    /// at the 8-byte granularity used by
    /// `malformed_json_logs_warn_and_increments_parse_counter`.
    fn assert_no_token_bytes(output: &str, canary: &str) {
        let canary_bytes = canary.as_bytes();
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
                 (slice={slice:?}, full canary={canary:?}, output={output})"
            );
        }
    }

    /// Write a token file with the given access token expiring N seconds
    /// from now. Uses `tempfile`-style write-temp + rename via
    /// `tokio::fs::write` (which is itself a single write — the provider's
    /// production contract is that the credential-provisioner sidecar
    /// performs an atomic rename, but for tests the file is small enough
    /// that the single write is functionally indistinguishable).
    async fn write_token_file(path: &std::path::Path, token: &str, expires_in_secs: i64) {
        write_token_file_in(path, token, chrono::Duration::seconds(expires_in_secs)).await
    }

    /// [`write_token_file`] with an arbitrary — in particular, **sub-second** —
    /// lifetime.
    ///
    /// The stale-token phase ([`StalePhase`]) is the only thing in this module
    /// driven by `chrono::Utc::now()` rather than by a tokio timer, and
    /// `Utc::now()` is *not* virtualised by `#[tokio::test(start_paused =
    /// true)]`: a paused test can advance tokio's clock indefinitely without
    /// ever moving a cached token one millisecond closer to expiry, so the
    /// expiry-driven transitions never fire and the test hangs on its budget.
    /// The stale-phase tests therefore run on the real clock and buy their
    /// wall-time budget by making the *token* short-lived instead — hundreds of
    /// milliseconds rather than the production hours — while keeping
    /// [`LONG_POLL_INTERVAL`] as the discriminator against the steady cadence.
    async fn write_token_file_in(
        path: &std::path::Path,
        token: &str,
        expires_in: chrono::Duration,
    ) {
        tokio::fs::write(path, token_file_body_in(token, expires_in))
            .await
            .expect("write fixture");
    }

    /// Body of a token file naming `token` and expiring `expires_in_secs` from
    /// now (negative values produce an already-expired token).
    fn token_file_body(token: &str, expires_in_secs: i64) -> String {
        token_file_body_in(token, chrono::Duration::seconds(expires_in_secs))
    }

    /// [`token_file_body`] with an arbitrary (possibly sub-second) lifetime —
    /// see [`write_token_file_in`] for why sub-second lifetimes exist here.
    fn token_file_body_in(token: &str, expires_in: chrono::Duration) -> String {
        let expires_on = Utc::now() + expires_in;
        format!(
            r#"{{"access_token":"{}","expires_on":"{}"}}"#,
            token,
            expires_on.to_rfc3339()
        )
    }

    /// Blocking counterpart to [`write_token_file`].
    ///
    /// Used by the tests that must not yield to the tokio scheduler between
    /// writing the fixture and the next step: on the default current-thread
    /// test runtime an `.await` is the only place the polling task can be
    /// scheduled, so a blocking write makes "the poller has not seen this file
    /// yet" a structural guarantee rather than a timing assumption.
    fn write_token_file_blocking(path: &std::path::Path, token: &str, expires_in_secs: i64) {
        std::fs::write(path, token_file_body(token, expires_in_secs)).expect("write fixture");
    }

    /// Install a thread-local capturing `tracing` subscriber and return its
    /// guard plus the shared buffer.
    ///
    /// Must be called *before* the provider is constructed so the spawned
    /// polling task's events land in the buffer: `#[tokio::test]` is a
    /// current-thread runtime, so the task runs on this same OS thread and
    /// inherits the thread-local default dispatcher. Drop the guard before
    /// asserting on the buffer.
    fn capture_logs() -> (tracing::subscriber::DefaultGuard, Arc<Mutex<Vec<u8>>>) {
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            // `fmt` colourises by default whether or not the writer is a tty,
            // and the escape codes land *between* a field name and its `=`.
            // That leaves a structured field unmatchable by a plain
            // `contains("name=value")` — which is how a test asserts that a
            // WARN carries the number an operator needs (see
            // `rotation_below_prolongation_floor_records_headroom_and_warns_once`).
            // Message-substring matches are unaffected either way.
            .with_ansi(false)
            .with_max_level(tracing::Level::TRACE)
            .finish();
        (tracing::subscriber::set_default(subscriber), captured)
    }

    /// Distinctive substring of the stale-token phase's **entry** INFO. Chosen
    /// so it cannot collide with either the cold-start window WARN
    /// ([`STARTUP_WINDOW_WARN_MARKER`]) or the stale phase's own expiry WARN
    /// ([`STALE_EXPIRY_WARN_MARKER`]).
    const STALE_ENTRY_INFO_MARKER: &str = "entering the stale-token fast-poll phase";

    /// Distinctive substring of the stale-token phase's **expiry** WARN.
    const STALE_EXPIRY_WARN_MARKER: &str = "leaving the stale-token fast-poll phase";

    /// Distinctive substring of the *cold-start* window-expiry WARN. Shared
    /// with the two `STARTUP_FAST_POLL_WINDOW` tests, which assert an exact
    /// count on it.
    const STARTUP_WINDOW_WARN_MARKER: &str = "startup fast-poll window";

    /// Distinctive substring of the rotation **prolongation-headroom** WARN
    /// (b.is1). Shares no wording with the three markers above, so a test can
    /// assert an exact count on it while the stale-phase lines come and go.
    const HEADROOM_WARN_MARKER: &str = "picked up with less headroom";

    /// Count occurrences of `marker` in the captured log buffer.
    fn marker_count(buf: &Arc<Mutex<Vec<u8>>>, marker: &str) -> usize {
        captured_text(buf).matches(marker).count()
    }

    /// Build a provider against `path` with an explicit `poll_interval` *and*
    /// `expiry_skew` — the stale-token phase's entry lead is the sum of the
    /// two, so both must be controllable for those tests.
    fn build_provider_with_skew_and_interval(
        path: std::path::PathBuf,
        poll_interval: Duration,
        expiry_skew: chrono::Duration,
    ) -> (ServiceTokenProvider, Arc<Metrics>, CancellationToken) {
        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new_with_refresh_lead(
            path,
            poll_interval,
            expiry_skew,
            metrics.clone(),
            cancel.clone(),
        );
        (provider, metrics, cancel)
    }

    /// Build a provider against `path` with a fresh metrics + cancellation
    /// token, using an explicit `expiry_skew`. Returns the provider plus its
    /// metrics handle (for counter assertions) and the cancellation token (so
    /// the test can shut down the polling task on drop if it cares).
    fn build_provider_with_skew(
        path: std::path::PathBuf,
        expiry_skew: chrono::Duration,
    ) -> (ServiceTokenProvider, Arc<Metrics>, CancellationToken) {
        build_provider_with_skew_and_interval(path, TEST_POLL_INTERVAL, expiry_skew)
    }

    /// Build a provider with the storage-api default 60 s skew ([`DEFAULT_REFRESH_LEAD`]).
    /// Preserves the pre-parameterization behaviour byte-for-byte for the
    /// existing tests that assert the 60 s window.
    fn build_provider(
        path: std::path::PathBuf,
    ) -> (ServiceTokenProvider, Arc<Metrics>, CancellationToken) {
        build_provider_with_skew(path, DEFAULT_REFRESH_LEAD)
    }

    /// Build a Valkey-role provider carrying an explicit prolongation floor
    /// (b.is1) — the only shape `new_with_refresh_lead_and_role` exists for.
    ///
    /// The refresh lead is [`SUB_SECOND_SKEW`] rather than the production
    /// 3 min: the headroom tests below use token lifetimes of tens of seconds,
    /// and a 3 min lead would put every one of those fixtures inside the
    /// stale-token phase's entry window, mixing that phase's transitions into
    /// tests that are about the rotation observation. The floor is what those
    /// tests vary; the lead is held out of the way.
    fn build_provider_with_floor(
        path: std::path::PathBuf,
        prolongation_floor: Option<chrono::Duration>,
    ) -> (ServiceTokenProvider, Arc<Metrics>, CancellationToken) {
        build_provider_with_role_lead_and_floor(
            path,
            ServiceTokenProvider::ROLE_VALKEY,
            SUB_SECOND_SKEW,
            prolongation_floor,
        )
    }

    /// [`build_provider_with_floor`] with the role and the refresh lead opened
    /// up too — every argument `new_with_refresh_lead_and_role` takes.
    ///
    /// Two callers need something other than the Valkey-role /
    /// [`SUB_SECOND_SKEW`] default: the test covering the WARN's role-keyed
    /// operator lever (a non-Valkey role that nonetheless carries a floor), and
    /// the test covering the arithmetic guards (a lead so large that
    /// `Utc::now() + lead` is unrepresentable).
    fn build_provider_with_role_lead_and_floor(
        path: std::path::PathBuf,
        role: &'static str,
        refresh_lead: chrono::Duration,
        prolongation_floor: Option<chrono::Duration>,
    ) -> (ServiceTokenProvider, Arc<Metrics>, CancellationToken) {
        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new_with_refresh_lead_and_role(
            path,
            TEST_POLL_INTERVAL,
            refresh_lead,
            prolongation_floor,
            role,
            metrics.clone(),
            cancel.clone(),
        );
        (provider, metrics, cancel)
    }

    /// Wait until the rotation-headroom gauge for `role` reads something other
    /// than `prev`, then return that new value.
    ///
    /// The positive-assertion wait for the whole `── Rotation prolongation
    /// headroom` section: a loop-head pass *observing* the rotation is what
    /// moves the gauge, so reading the value straight after a file rewrite would
    /// race the pickup.
    ///
    /// `prev` is the caller's last observation — `None` before any rotation, so
    /// the first call waits on the series simply appearing. Keying the wait on
    /// **change** rather than on presence is what makes the second and later
    /// rotations in a test observable at all: once the first rotation has landed
    /// the series exists forever after, so a presence wait would return the
    /// *stale* value immediately and let a once-ever-latched implementation (a
    /// `warned: bool`, a first-rotation-only guard) pass a test that believes it
    /// is watching rotation #2.
    ///
    /// The new value is captured inside the predicate rather than re-read after
    /// it: a re-read could pick up a *third* rotation's value and silently
    /// attribute it to the wait that ended on the second.
    async fn await_headroom_change(metrics: &Metrics, role: &str, prev: Option<f64>) -> f64 {
        let observed = std::cell::Cell::new(None::<f64>);
        poll_until(POLL_PICKUP_BUDGET, || {
            match refresh_headroom_seconds(metrics, role) {
                Some(secs) if Some(secs) != prev => {
                    observed.set(Some(secs));
                    true
                }
                _ => false,
            }
        })
        .await
        .unwrap_or_else(|| {
            panic!(
                "the polling task must record a rotation-headroom point for role {role:?} \
                 differing from the previous observation ({prev:?}) within {POLL_PICKUP_BUDGET:?} \
                 of the replacement token being written; the gauge read {:?}",
                refresh_headroom_seconds(metrics, role)
            )
        });
        observed
            .get()
            .expect("the predicate above stored the value it matched on")
    }

    /// Widened proactive re-read lead for the access-token-file Valkey
    /// provider: Azure requires a fresh token *fetched* ≥ 3 min before expiry
    /// (SDD SR-4.1). 3 min exactly.
    const WIDE_EXPIRY_SKEW: chrono::Duration = chrono::Duration::seconds(180);

    /// Proactive re-read lead used by the stale-token phase tests.
    ///
    /// Deliberately sub-second: those tests run on the real clock (see
    /// [`write_token_file_in`]) with token lifetimes in the hundreds of
    /// milliseconds, and the lead is one of the two terms in the phase's entry
    /// window. It has almost no effect on that window in those tests: the entry
    /// lead is `expiry_skew + poll_interval`, and with [`LONG_POLL_INTERVAL`]
    /// the 60 s term dominates by two orders of magnitude.
    const SUB_SECOND_SKEW: chrono::Duration = chrono::Duration::milliseconds(500);

    // ── cached_bearer_if_unexpired(): basic cache lifecycle ─────────────────

    #[tokio::test]
    async fn cached_bearer_if_unexpired_returns_none_before_first_successful_read() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("missing.json");
        // Provider points at a path that does not exist. The polling task
        // may have fired its `interval.tick()` once (the first tick fires
        // immediately), but with NotFound the cache stays empty.
        let (provider, _metrics, cancel) = build_provider(path);
        // Read before any await — even if the polling task has had a chance
        // to schedule, NotFound leaves the cache empty.
        assert_eq!(provider.cached_bearer_if_unexpired(), None);
        cancel.cancel();
    }

    #[tokio::test]
    async fn cached_bearer_if_unexpired_returns_token_after_first_poll_tick() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, _metrics, cancel) = build_provider(path);
        await_bearer(&provider, "token-A").await;

        cancel.cancel();
    }

    // ── Polling pickup of rewrites (acceptance #3) ──────────────────────────

    #[tokio::test]
    async fn polling_picks_up_rewritten_token_within_one_interval() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        let (provider, _metrics, cancel) = build_provider(path.clone());
        await_bearer(&provider, "token-A").await;

        write_token_file(&path, "token-B", 3600).await;
        await_bearer(&provider, "token-B").await;

        cancel.cancel();
    }

    // ── force_refresh() (acceptance #4) ─────────────────────────────────────

    #[tokio::test]
    async fn force_refresh_reads_immediately_without_waiting_for_next_tick() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token.json");
        write_token_file(&path, "token-A", 3600).await;

        // Use a LONG poll interval so the polling tick definitely does not
        // fire during the test window — that way an observed flip can only
        // be attributed to the explicit `force_refresh()` call.
        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            Duration::from_secs(60),
            metrics.clone(),
            cancel.clone(),
        );
        // Drive the initial cache fill via force_refresh (the first
        // interval.tick() also fires immediately, but force_refresh is the
        // deterministic path).
        provider.force_refresh().await.expect("initial refresh");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("token-A".to_string())
        );

        // Rewrite the file and force-refresh — NO sleep between the rewrite
        // and the assertion: any flip must come from force_refresh.
        write_token_file(&path, "token-B", 3600).await;
        provider
            .force_refresh()
            .await
            .expect("post-rewrite refresh");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("token-B".to_string())
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn force_refresh_returns_err_on_io_failure() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("never-created.json");
        // Use a LONG poll interval so the polling task does not race with
        // our explicit force_refresh request.
        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path,
            Duration::from_secs(60),
            metrics.clone(),
            cancel.clone(),
        );

        let err = provider
            .force_refresh()
            .await
            .expect_err("missing file must surface as Err");

        // Tighten: the variant MUST be `Io(_)` — a future refactor that
        // mapped NotFound to `Parse` or `Cancelled` would silently lose
        // the distinct-error-mode signal the Epic 5 401-retry path relies
        // on. Bind the inner error to a name so the asserts below can
        // inspect it.
        assert!(
            matches!(err, ServiceTokenError::Io(_)),
            "expected ServiceTokenError::Io, got {err:?}"
        );

        // No access-token bytes — there is no on-disk content yet, so the
        // best we can do is assert the Display string is a generic I/O
        // description and not something like a JSON fragment. Defensive:
        // the file is absent, so no real-world token bytes could leak, but
        // the assertion documents the invariant.
        let displayed = format!("{err}");
        assert!(
            !displayed.contains("access_token"),
            "Display string must not include the access_token JSON key: {displayed}"
        );

        cancel.cancel();
    }

    // ── Startup-absent recovery (acceptance #5) ─────────────────────────────

    #[tokio::test]
    async fn startup_absent_file_tolerated_and_recovers_on_first_write() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("late-arrival.json");
        let (provider, _metrics, cancel) = build_provider(path.clone());

        // Wait through several poll ticks while the file is absent. The
        // polling task must NOT panic, error out, or otherwise wedge — the
        // SR-6.2 startup-absent path simply leaves the cache empty.
        for _ in 0..3 {
            await_one_poll().await;
            assert_eq!(provider.cached_bearer_if_unexpired(), None);
        }

        // Sidecar finally writes the file → next poll picks it up.
        //
        // NOTE: this test's `TEST_POLL_INTERVAL` is 20 ms, so "the next poll"
        // here is only 20 ms away and the assertion says nothing about how a
        // *production* 15 s interval behaves. That gap is what b.o98 was:
        // see `cold_start_picks_up_late_written_token_far_inside_poll_interval`
        // below, which pins the same scenario against a 60 s interval.
        write_token_file(&path, "late-token", 3600).await;
        await_bearer(&provider, "late-token").await;

        cancel.cancel();
    }

    // ── Cold-start fast-poll phase (b.o98) ──────────────────────────────────

    /// b.o98 regression. Under `LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile`
    /// the provider's first read fires at t≈0, before the credential
    /// sidecar has written the token file (observed at ~t+0.5 s). The
    /// NotFound is tolerated, and pre-fix the *next* read waited a full
    /// `poll_interval` (production default 15 s), stalling the first Valkey
    /// connect ~22 s and tripping the liveness probe.
    ///
    /// The steady-state interval here is [`LONG_POLL_INTERVAL`] (60 s), so a
    /// pass cannot come from an `interval.tick()`: the assertion budget is
    /// 2 s. Pre-fix, the earliest possible flip was t+60 s.
    #[tokio::test]
    async fn cold_start_picks_up_late_written_token_far_inside_poll_interval() {
        /// The sidecar's write latency, scaled down from the ~0.5 s observed
        /// in production. Long enough that the t≈0 read reliably lands first
        /// (making this a genuine "file absent on first read" scenario).
        const SIDECAR_WRITE_DELAY: Duration = Duration::from_millis(150);
        /// Assertion budget: generous against `FAST_POLL_RETRY_INTERVAL`
        /// (250 ms) and scheduling jitter, but 30× below `LONG_POLL_INTERVAL`.
        const PICKUP_BUDGET: Duration = Duration::from_secs(2);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("late-sidecar-write.json");

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            LONG_POLL_INTERVAL,
            metrics.clone(),
            cancel.clone(),
        );

        // Precondition: file absent, cache empty.
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            None,
            "cache must be empty before the sidecar writes the token file"
        );

        // Sidecar writes the token shortly after boot.
        tokio::time::sleep(SIDECAR_WRITE_DELAY).await;
        write_token_file(&path, "sidecar-token", 3600).await;

        let elapsed = poll_until(PICKUP_BUDGET, || {
            provider.cached_bearer_if_unexpired().is_some()
        })
        .await;
        let elapsed = elapsed.unwrap_or_else(|| {
            panic!(
                "token written {SIDECAR_WRITE_DELAY:?} after construction was not cached within \
                 {PICKUP_BUDGET:?} — the cold-start fast-poll phase is not running, so the first \
                 pickup is stalled until the {LONG_POLL_INTERVAL:?} steady-state tick (b.o98)"
            )
        });
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("sidecar-token".to_string()),
            "the cached token must be the one the sidecar wrote (pickup took {elapsed:?})"
        );

        cancel.cancel();
    }

    /// Warm restart (pod restart with the token file already on disk) must
    /// not regress: the explicit t≈0 read caches the token immediately, with
    /// no dependence on the fast-poll cadence.
    ///
    /// The budget is deliberately *below* [`FAST_POLL_RETRY_INTERVAL`] so this
    /// discriminates "cached by the immediate read" from "cached by the first
    /// fast-poll tick". If it ever flakes on a loaded runner, raise the
    /// budget — but keep it under `FAST_POLL_RETRY_INTERVAL` or the test stops
    /// discriminating.
    #[tokio::test]
    async fn warm_restart_caches_token_on_the_immediate_first_read() {
        const IMMEDIATE_READ_BUDGET: Duration = Duration::from_millis(200);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("already-present.json");
        write_token_file(&path, "warm-token", 3600).await;

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            LONG_POLL_INTERVAL,
            metrics.clone(),
            cancel.clone(),
        );

        assert!(
            IMMEDIATE_READ_BUDGET < FAST_POLL_RETRY_INTERVAL,
            "budget must stay below the fast-poll cadence for this test to discriminate"
        );
        let elapsed = poll_until(IMMEDIATE_READ_BUDGET, || {
            provider.cached_bearer_if_unexpired().is_some()
        })
        .await
        .unwrap_or_else(|| {
            panic!(
                "a token present at construction must be cached by the immediate t≈0 read, \
                     within {IMMEDIATE_READ_BUDGET:?}"
            )
        });
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("warm-token".to_string()),
            "warm restart must surface the on-disk token (read took {elapsed:?})"
        );

        cancel.cancel();
    }

    /// The fast-poll cadence is `FAST_POLL_RETRY_INTERVAL.min(poll_interval)`:
    /// an operator who configured a poll interval *shorter* than 250 ms must
    /// not have startup reads slowed down to 250 ms by the cold-start phase.
    ///
    /// With `TEST_POLL_INTERVAL` = 20 ms, an unclamped fast-poll would not
    /// re-read until t ≈ 250 ms; the token here lands at t ≈ 30 ms and must be
    /// cached well before that.
    #[tokio::test]
    async fn fast_poll_cadence_is_clamped_to_a_shorter_poll_interval() {
        /// Early enough that an unclamped 250 ms fast-poll would still be
        /// ~220 ms away when the file lands.
        const WRITE_DELAY: Duration = Duration::from_millis(30);
        /// Comfortably above `TEST_POLL_INTERVAL` (20 ms) yet below the gap an
        /// unclamped cadence would leave.
        const PICKUP_BUDGET: Duration = Duration::from_millis(150);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("short-interval.json");
        let (provider, _metrics, cancel) = build_provider(path.clone());

        tokio::time::sleep(WRITE_DELAY).await;
        write_token_file(&path, "short-interval-token", 3600).await;

        let elapsed = poll_until(PICKUP_BUDGET, || {
            provider.cached_bearer_if_unexpired().is_some()
        })
        .await
        .unwrap_or_else(|| {
            panic!(
                "a {TEST_POLL_INTERVAL:?} poll interval must not be slowed to \
                     {FAST_POLL_RETRY_INTERVAL:?} by the cold-start phase — the token was still \
                     uncached {PICKUP_BUDGET:?} after it was written"
            )
        });
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("short-interval-token".to_string()),
            "the short-interval poll must cache the token (pickup took {elapsed:?})"
        );

        cancel.cancel();
    }

    /// The *cold-start* fast-poll phase is retired on "the cache holds a
    /// token", NOT on `cached_bearer_if_unexpired().is_some()`.
    ///
    /// A token that parses fine but has since run out makes the bearer accessor
    /// return `None` while the cache is populated. Gating the cold-start phase
    /// on the bearer would re-arm the *startup* phase for a steady-state
    /// condition — and a steady-state concern leaking into the startup path is
    /// what this test exists to prevent. A warmed-up provider whose token is
    /// nearing or past expiry is the *stale-token* phase's job
    /// ([`StalePhase`]), which keys on the cached expiry rather than on the
    /// accessor.
    ///
    /// The fixture is an already-expired token because that is now the only
    /// shape for which the cache is populated and the bearer accessor still
    /// answers `None`: b.mq8 deleted the withholding accessor, so "cached but
    /// not served" and "cached but dead" are the same state. Its sibling
    /// `warm_restart_into_an_already_expired_token_warns_without_entering_the_phase`
    /// shares the fixture but owns a different subject — the one-shot-ness of
    /// the retirement WARN. This one owns the cold-start retirement predicate.
    ///
    /// # Why this test is asserted on logs rather than on cadence
    ///
    /// The original form of this test used a cadence proxy: cache the token,
    /// overwrite the file with a far-future one, and assert the replacement was
    /// *not* observed inside several [`FAST_POLL_RETRY_INTERVAL`]s — on the
    /// theory that only a task still in the fast phase could see it before the
    /// [`LONG_POLL_INTERVAL`] steady tick. That proxy stopped discriminating
    /// when the stale-token phase ([`StalePhase`], b.qam) landed, and it cannot
    /// be repaired by retuning: any fixture whose bearer reads `None` off a
    /// populated cache is by construction one the stale phase reacts to, and
    /// b.d6b then keeps the same 250 ms cadence live across the bounded
    /// post-expiry window. A correct implementation therefore *does* fast-poll
    /// here, which is the intended behaviour, not the regression the old
    /// assertion described.
    ///
    /// The surviving discriminator is which phase is doing the fast polling,
    /// and that is observable only in the log. The stale-phase transitions are
    /// evaluated under `if !fast_poll_active`, so a cold-start phase that
    /// failed to retire would suppress them entirely: **the presence of the
    /// stale-expiry WARN is proof that `fast_poll_active` went false**, i.e.
    /// that the cold-start phase retired on cache-population rather than
    /// hanging on `cached_bearer_if_unexpired()`.
    #[tokio::test]
    async fn expired_cached_token_still_retires_the_fast_poll_phase() {
        /// Budget for the immediate t≈0 read to land the expired token in
        /// the cache, and for the loop head that follows it to evaluate the
        /// phase transitions. Generously above the (sub-millisecond) work
        /// involved.
        const FIRST_READ_SETTLE: Duration = Duration::from_millis(500);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("already-expired.json");
        // Died 10 s ago — parses fine and populates the cache, but the bearer
        // accessor declines to serve it.
        write_token_file(&path, "already-expired-token", -10).await;

        let (_dispatch_guard, captured) = capture_logs();

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            LONG_POLL_INTERVAL,
            metrics.clone(),
            cancel.clone(),
        );

        // Wait on the real precondition — "the cache holds the expired
        // token" — rather than assuming a fixed settle time. `has_cached_token()`
        // is the exact predicate the polling task's warm-up check gates on, so
        // a slow t≈0 read extends the wait instead of failing the test and
        // misattributing the blame to the retirement logic.
        poll_until(FIRST_READ_SETTLE, || provider.inner.has_cached_token())
            .await
            .expect("the expired token must be cached by the immediate t≈0 read");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            None,
            "the fixture must be the cached-but-not-served shape this test needs: the cache is \
             populated (asserted above) yet the bearer accessor declines the dead token"
        );
        assert_eq!(
            parse_error_count(&metrics),
            0,
            "the expired fixture is valid JSON — no parse errors expected"
        );

        // The discriminator. The stale-phase transitions live behind
        // `if !fast_poll_active`, so this WARN can only be emitted once the
        // cold-start phase has retired — which, with the cache populated but
        // `cached_bearer_if_unexpired()` still `None`, can only have happened on
        // the cache-non-empty check.
        poll_until(FIRST_READ_SETTLE, || {
            marker_count(&captured, STALE_EXPIRY_WARN_MARKER) > 0
        })
        .await
        .expect(
            "the stale-token phase never reported the expiry, which means the cold-start phase \
             never retired: with the cache populated and cached_bearer_if_unexpired() still None, \
             a cold-start phase gated on the bearer would stay active and suppress the \
             stale-phase transitions entirely",
        );

        // Corroboration, not the discriminator: the cold-start phase retired
        // via the cache-non-empty check, so it never reached its
        // `STARTUP_FAST_POLL_WINDOW` bound. (Weaker than the assertion above by
        // construction — the window is 20 s and this test observes well under
        // 1 s — so it can only catch a *premature* window WARN, never confirm
        // the absence of a late one.)
        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STARTUP_WINDOW_WARN_MARKER).count(),
            0,
            "the cold-start phase must retire on cache-population, not by burning its window: \
             {output}"
        );

        // Liveness control: the task is still alive and serving reads.
        write_token_file(&path, "far-future-token", 3600).await;
        provider.force_refresh().await.expect("force_refresh");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("far-future-token".to_string()),
            "the polling task must still service force_refresh after leaving the cold-start phase"
        );

        cancel.cancel();
    }

    /// The fast-poll phase is retired by a cache that `force_refresh()`
    /// populated, not only by one a fast-poll read populated.
    ///
    /// `run_polling_task` re-checks `has_cached_token()` at the *loop head* —
    /// covering every path that can fill the cache — precisely so a load that
    /// arrived via the `force_refresh` arm ends the phase. A check placed
    /// inside the fast-poll arm instead would leave a task whose first token
    /// came from `force_refresh()` stat-ing the file at the 250 ms cadence for
    /// the rest of the process lifetime.
    ///
    /// Observable: a cadence measurement. After the force-refresh load, a
    /// second token written to the file must NOT be observed within several
    /// [`FAST_POLL_RETRY_INTERVAL`]s, because only a still-fast-polling task
    /// could see it before the [`LONG_POLL_INTERVAL`] steady tick.
    ///
    /// That shape is sound *here* and is not sound in
    /// `expired_cached_token_still_retires_the_fast_poll_phase`, which once
    /// shared it and has since been rewritten onto log discrimination. There
    /// the cached token is dead, so the stale-token phase's bounded post-expiry
    /// window keeps the fast cadence running for a legitimate reason and no
    /// cadence measurement can tell the two phases apart. Here the loaded token
    /// is an hour from expiry, so the stale phase stays dormant and a fast
    /// cadence can only mean an un-retired cold-start phase.
    #[tokio::test]
    async fn force_refresh_populated_cache_retires_the_fast_poll_phase() {
        /// Six fast-poll intervals — a task still in the fast phase would have
        /// re-read the file several times over.
        const OBSERVATION_WINDOW: Duration = Duration::from_millis(1500);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("force-refresh-first.json");

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            LONG_POLL_INTERVAL,
            metrics.clone(),
            cancel.clone(),
        );

        // Precondition: file absent, cache empty, task still fast-polling.
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            None,
            "cache must be empty while the token file is absent"
        );

        // Write blocking, then refresh with no intervening `.await`: the
        // polling task cannot be scheduled in between on the current-thread
        // runtime, so every fast-poll read that precedes the refresh saw an
        // absent file. `select!`'s `biased` ordering then prefers the queued
        // refresh request over an already-ready fast-poll timer, so the load
        // that populates the cache is the `force_refresh()` one.
        write_token_file_blocking(&path, "refresh-token", 3600);
        provider
            .force_refresh()
            .await
            .expect("force_refresh must load the freshly written file");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("refresh-token".to_string()),
            "force_refresh must be the load that populates the cache"
        );

        // Swap in a second token. Only a task still on the fast cadence could
        // observe this before the 60 s steady tick.
        write_token_file_blocking(&path, "second-token", 3600);
        let picked_up = poll_until(OBSERVATION_WINDOW, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("second-token")
        })
        .await;
        assert!(
            picked_up.is_none(),
            "a cache populated by force_refresh() must retire the fast-poll phase at the loop-head \
             warm-up check; the replacement token was picked up after {picked_up:?}, which means \
             the task was still re-reading at the {FAST_POLL_RETRY_INTERVAL:?} cadence"
        );

        // Liveness control: the task is still alive and serving reads — the
        // assertion above is "steady cadence", not "polling task died".
        provider.force_refresh().await.expect("force_refresh");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("second-token".to_string()),
            "the polling task must still service force_refresh after leaving the fast phase"
        );

        cancel.cancel();
    }

    /// Cancelling while the cold-start fast-poll phase is active must actually
    /// stop the polling task.
    ///
    /// The fast-poll arm is a fresh `tokio::time::sleep` future constructed per
    /// loop iteration rather than a long-lived `interval`, so it is the arm
    /// most at risk of outliving a cancel if the `select!` ever loses its
    /// `cancel.cancelled()` branch — and the one the steady-state cancellation
    /// path never exercises. Pinned here: after cancellation a token written to
    /// the file is never cached, and `force_refresh()` reports
    /// [`ServiceTokenError::Cancelled`].
    #[tokio::test]
    async fn cancel_during_fast_poll_phase_stops_the_polling_task() {
        /// Budget for the task to observe cancellation and drop its
        /// force-refresh receiver.
        const SHUTDOWN_BUDGET: Duration = Duration::from_secs(2);
        /// Six fast-poll intervals — a task still fast-polling would have
        /// re-read the file several times over.
        const OBSERVATION_WINDOW: Duration = Duration::from_millis(1500);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("cancel-during-fast-poll.json");

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            LONG_POLL_INTERVAL,
            metrics.clone(),
            cancel.clone(),
        );

        // Precondition: the file is absent so the cache never populates, which
        // keeps the task in the fast-poll phase right up to the cancel.
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            None,
            "cache must be empty while the token file is absent"
        );

        cancel.cancel();

        // The polling task owns the force-refresh receiver, so a closed sender
        // is the observable for "the task returned" — waiting on it before
        // writing the fixture removes any race between shutdown and the write.
        poll_until(SHUTDOWN_BUDGET, || {
            provider.inner.force_refresh_tx.is_closed()
        })
        .await
        .expect("the polling task must exit the fast-poll sleep arm on cancellation");

        // A token appearing after shutdown must never be cached.
        write_token_file(&path, "post-cancel-token", 3600).await;
        let picked_up = poll_until(OBSERVATION_WINDOW, || {
            provider.cached_bearer_if_unexpired().is_some()
        })
        .await;
        assert!(
            picked_up.is_none(),
            "a cancelled polling task must stop reading the token file, but a token written after \
             cancellation was cached after {picked_up:?}"
        );

        let err = provider
            .force_refresh()
            .await
            .expect_err("force_refresh must fail once the polling task has shut down");
        assert!(
            matches!(err, ServiceTokenError::Cancelled),
            "expected ServiceTokenError::Cancelled, got {err:?}"
        );
    }

    /// [`STARTUP_FAST_POLL_WINDOW`] bounds the fast-poll phase: a token file
    /// that never becomes loadable (misconfigured sidecar, wrong path) must
    /// not leave the task stat-ing the path four times a second forever. On
    /// expiry it logs exactly one WARN and reverts to `poll_interval`.
    ///
    /// Runs on a paused clock and advances virtual time in
    /// [`FAST_POLL_RETRY_INTERVAL`] steps — [`STARTUP_FAST_POLL_WINDOW`] is a
    /// module constant with no test knob, and a real-clock test would burn
    /// that window of suite wall time. Stepping at exactly the task's own cadence keeps
    /// the auto-advance bounded to one fast-poll interval per step even while
    /// the task is inside a real filesystem read.
    ///
    /// The fixture is unparseable *and* carries a canary token, so the same
    /// run pins the SR-6.2 bearer-bytes-never-logged invariant on the WARN
    /// path. (An absent file drives the identical arm — the WARN is gated on
    /// `!has_cached_token()`, not on the failure mode — but leaves no token
    /// bytes on disk for the canary to catch.)
    #[tokio::test(start_paused = true)]
    async fn fast_poll_window_expiry_logs_exactly_one_warn_without_token_bytes() {
        const SECRET_TOKEN: &str = "tkn-window-canary-0123456789abcdef";
        /// Distinctive substring of the window-expiry WARN. Shared with the
        /// stale-phase tests, which assert this marker is *absent*; chosen so
        /// it collides with neither the per-read parse WARN (which fires once
        /// per [`FAST_POLL_RETRY_INTERVAL`] for the whole window) nor the
        /// stale-token phase's own entry / expiry lines.
        const WINDOW_WARN_MARKER: &str = STARTUP_WINDOW_WARN_MARKER;
        /// 4× the nominal step count for the window
        /// ([`STARTUP_FAST_POLL_WINDOW`] / [`FAST_POLL_RETRY_INTERVAL`]), so
        /// scheduling drift under the paused clock cannot fail the test.
        /// Derived from the constants — this is a ceiling on a loop that exits
        /// at the WARN, so it only ever needs to stay above the nominal count.
        const MAX_STEPS: usize = (STARTUP_FAST_POLL_WINDOW.as_millis()
            / FAST_POLL_RETRY_INTERVAL.as_millis()) as usize
            * 4;
        /// Extra steps (10 virtual seconds) after the WARN, during which a
        /// task still on the fast cadence would emit ~40 more WARNs.
        const POST_WARN_STEPS: usize = 40;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("never-parses.json");
        // Never parses → the cache never populates → the fast-poll phase can
        // only be retired by the window bound.
        let body = format!("{{\"access_token\":\"{SECRET_TOKEN}\" <-- GARBAGE, NOT JSON");
        tokio::fs::write(&path, body).await.expect("write fixture");

        let (_dispatch_guard, captured) = capture_logs();

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            LONG_POLL_INTERVAL,
            metrics.clone(),
            cancel.clone(),
        );

        // Precondition for the WARN arm: the file never loads, so the cache
        // stays empty and the phase can only end at the window bound.
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            None,
            "an unparseable file must leave the cache empty"
        );

        // The WARN is emitted at the polling task's loop head, i.e. only after
        // its in-flight blocking `tokio::fs` read returns. That read cannot
        // silently consume the step budget: a blocking task in flight inhibits
        // clock auto-advance, so the stepping loop parks in real time until it
        // completes (see `step_paused_until`).
        let steps = step_paused_until(MAX_STEPS, || {
            captured_text(&captured).matches(WINDOW_WARN_MARKER).count() > 0
        })
        .await;
        assert!(
            steps.is_some(),
            "no fast-poll-window WARN after {MAX_STEPS} × {FAST_POLL_RETRY_INTERVAL:?} of virtual \
             time (window is {STARTUP_FAST_POLL_WINDOW:?}) — the fast-poll phase is unbounded"
        );

        // Keep advancing: the WARN must be a one-shot, not a per-tick log.
        for _ in 0..POST_WARN_STEPS {
            tokio::time::sleep(FAST_POLL_RETRY_INTERVAL).await;
        }

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(WINDOW_WARN_MARKER).count(),
            1,
            "the fast-poll-window fallback must WARN exactly once, not on every subsequent tick"
        );
        assert_no_token_bytes(&output, SECRET_TOKEN);

        cancel.cancel();
    }

    /// After [`STARTUP_FAST_POLL_WINDOW`] expires the task must fall back to
    /// the steady `poll_interval` — not stay on the
    /// [`FAST_POLL_RETRY_INTERVAL`] cadence, and not stop polling altogether.
    ///
    /// Paused clock, same rationale as
    /// `fast_poll_window_expiry_logs_exactly_one_warn_without_token_bytes`.
    /// After burning the window with an unloadable file, a valid token is
    /// written: it must NOT appear within a few fast-poll intervals (the fast
    /// cadence is retired) but MUST appear once the steady interval elapses
    /// (the task is still polling).
    #[tokio::test(start_paused = true)]
    async fn fast_poll_window_expiry_reverts_to_steady_poll_cadence() {
        /// Distinctive substring of the window-expiry WARN — the observable
        /// this test uses to know the phase has actually been retired.
        const WINDOW_WARN_MARKER: &str = STARTUP_WINDOW_WARN_MARKER;
        /// Virtual-step budget for burning the window:
        /// [`STARTUP_FAST_POLL_WINDOW`] / [`FAST_POLL_RETRY_INTERVAL`], plus a
        /// small margin for drift. Derived from the constants rather than
        /// written as a literal so a constant change can't silently
        /// under-budget the burn.
        const WINDOW_STEPS: usize = (STARTUP_FAST_POLL_WINDOW.as_millis()
            / FAST_POLL_RETRY_INTERVAL.as_millis()) as usize
            + 8;
        /// 20 fast-poll intervals (5 virtual seconds), but — per the
        /// `assert!` below — well inside the [`LONG_POLL_INTERVAL`] steady
        /// interval that starts at retirement.
        const FAST_CADENCE_PROBE_STEPS: usize = 20;
        /// Enough steps to cover two steady intervals after the fallback.
        const STEADY_CADENCE_STEPS: usize =
            (LONG_POLL_INTERVAL.as_millis() / FAST_POLL_RETRY_INTERVAL.as_millis()) as usize * 2
                + 40;

        // The probe below asserts "still None", and the probe span starts at
        // the *observed* retirement (the WARN), which is where
        // `interval.reset()` puts the first steady tick. So the probe must
        // stay strictly inside one `LONG_POLL_INTERVAL` or that steady read
        // legitimately caches the token and the probe fails for a reason
        // unrelated to the invariant under test.
        assert!(
            FAST_POLL_RETRY_INTERVAL * (FAST_CADENCE_PROBE_STEPS as u32) < LONG_POLL_INTERVAL,
            "the post-retirement probe must land strictly before the first steady tick one \
             {LONG_POLL_INTERVAL:?} after retirement; shrink FAST_CADENCE_PROBE_STEPS"
        );

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("late-then-valid.json");
        tokio::fs::write(&path, b"garbage not json")
            .await
            .expect("write fixture");

        let (_dispatch_guard, captured) = capture_logs();

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            LONG_POLL_INTERVAL,
            metrics.clone(),
            cancel.clone(),
        );

        // Burn the fast-poll window, waiting for the retirement WARN rather
        // than for a fixed number of virtual steps. Two reasons:
        //
        // 1. A fixed burn says nothing about where the polling task sits in
        //    its loop when the burn ends — a last fast-poll read can still be
        //    in flight, and it would observe the valid token this test writes
        //    below. That is a legitimate fast-cadence pickup, failing the
        //    "must still be None" probe for a reason that is not the
        //    invariant under test.
        // 2. The WARN is emitted at the task's loop head, so by construction
        //    no fast-poll read is in flight once it appears, and the probe
        //    window below starts at the true retirement instant.
        let burn_steps = step_paused_until(WINDOW_STEPS, || {
            captured_text(&captured).matches(WINDOW_WARN_MARKER).count() > 0
        })
        .await;
        assert!(
            burn_steps.is_some(),
            "the fast-poll phase never retired after {WINDOW_STEPS} × {FAST_POLL_RETRY_INTERVAL:?} \
             of virtual time (window is {STARTUP_FAST_POLL_WINDOW:?})"
        );
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            None,
            "an unloadable file must leave the cache empty for the whole window"
        );

        // The file becomes valid. The fast cadence is gone, so nothing should
        // observe this for a full steady interval.
        write_token_file(&path, "post-window-token", 3600).await;
        for _ in 0..FAST_CADENCE_PROBE_STEPS {
            tokio::time::sleep(FAST_POLL_RETRY_INTERVAL).await;
            assert_eq!(
                provider.cached_bearer_if_unexpired(),
                None,
                "after the fast-poll window expires the task must be on the {LONG_POLL_INTERVAL:?} \
                 steady cadence, not still re-reading every {FAST_POLL_RETRY_INTERVAL:?}"
            );
        }

        // ...but the steady cadence must still be running. The cache flip is
        // published only after the steady tick's blocking read returns; that
        // read holds off auto-advance while it runs, so the step budget below
        // measures virtual time the task actually saw (see
        // `step_paused_until`).
        let steady_steps = step_paused_until(STEADY_CADENCE_STEPS, || {
            provider.cached_bearer_if_unexpired().is_some()
        })
        .await;
        assert!(
            steady_steps.is_some(),
            "the steady-state poll never picked the token up after the fast-poll window expired \
             ({STEADY_CADENCE_STEPS} × {FAST_POLL_RETRY_INTERVAL:?} of virtual time) — the polling \
             task stopped reading entirely"
        );
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("post-window-token".to_string()),
            "the steady-state poll must eventually cache the now-valid token"
        );

        cancel.cancel();
    }

    // ── Stale-token fast-poll re-entry (b.qam) ──────────────────────────────
    //
    // These tests all run on the REAL clock. `#[tokio::test(start_paused =
    // true)]` is structurally unusable here: the stale phase's transitions are
    // driven by `chrono::Utc::now()`, which tokio's paused clock does not
    // virtualise, so a paused test advances timers without ever moving a
    // cached token toward expiry — entry, retirement and the expiry WARN never
    // fire and the test burns its budget. The wall-time budget is bought by
    // making the *tokens* short-lived (`write_token_file_in`) and the skew
    // sub-second (`SUB_SECOND_SKEW`) instead, keeping each test to roughly
    // 0.5–1.5 s. `LONG_POLL_INTERVAL` (60 s) remains the discriminator: any
    // cache change observed inside a sub-second budget can only have come from
    // the fast cadence, never from a steady tick.

    /// Headline b.qam regression. A warmed-up provider whose cached token is
    /// inside `expiry_skew + poll_interval` of expiry re-enters the
    /// [`FAST_POLL_RETRY_INTERVAL`] cadence, so a *late* sidecar rotation is
    /// picked up sub-second instead of costing up to a full `poll_interval` of
    /// running on a token that has meanwhile expired.
    ///
    /// The steady interval is [`LONG_POLL_INTERVAL`] (60 s) and the pickup
    /// budget is 1 s, so a pass cannot be attributed to `interval.tick()`.
    /// Pre-fix, the earliest possible pickup was the 60 s steady tick.
    #[tokio::test]
    async fn stale_cache_re_enters_fast_poll_and_picks_up_late_rotation() {
        /// Lifetime of the cached token. Well inside the phase's ~60 s entry
        /// lead, so the phase arms at the first loop head; long enough that the
        /// rotation below lands while the phase is still `Active` rather than
        /// after the expiry retirement.
        const STALE_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(1500);
        /// Budget for the t≈0 read plus the loop head that arms the phase.
        const ENTRY_BUDGET: Duration = Duration::from_millis(500);
        /// Four fast-poll intervals — and 60× below [`LONG_POLL_INTERVAL`],
        /// which is the entire discriminator.
        const PICKUP_BUDGET: Duration = Duration::from_millis(1000);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("late-rotation.json");
        write_token_file_in(&path, "stale-token", STALE_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect(
            "a cached token inside `expiry_skew + poll_interval` of expiry must arm the \
             stale-token fast-poll phase at the first loop head",
        );

        // The sidecar's rotation lands late — after the phase armed. This is
        // the write that pre-fix would not have been observed for 60 s.
        write_token_file(&path, "rotated-token", 3600).await;
        let picked_up = poll_until(PICKUP_BUDGET, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("rotated-token")
        })
        .await;
        assert!(
            picked_up.is_some(),
            "a rotation written while the stale-token phase is active must be cached within \
             {PICKUP_BUDGET:?} (the {FAST_POLL_RETRY_INTERVAL:?} cadence); it was not, so the \
             task is waiting on the {LONG_POLL_INTERVAL:?} steady tick (b.qam)"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            1,
            "the phase must arm exactly once for one distinct cached expiry: {output}"
        );
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            0,
            "a recovering sidecar is not an operator event — retirement on a fresher token must \
             be silent: {output}"
        );

        cancel.cancel();
    }

    /// Dormancy guard: a provider whose cached token is nowhere near expiry
    /// must NOT be fast-polling.
    ///
    /// This is the test that fails an "always fast-poll" implementation — the
    /// simplest wrong way to fix b.qam. The token is 1 h from expiry, far
    /// outside the ~60 s entry lead, so the task must sit on the
    /// [`LONG_POLL_INTERVAL`] steady tick: a replacement written to the file is
    /// invisible across several [`FAST_POLL_RETRY_INTERVAL`]s, and the entry
    /// INFO never fires.
    #[tokio::test]
    async fn healthy_rotation_never_enters_the_stale_phase() {
        /// Budget for the t≈0 read to land the healthy token in the cache.
        const CACHE_BUDGET: Duration = Duration::from_millis(500);
        /// Six fast-poll intervals — a fast-polling task would have re-read the
        /// file several times over — yet still far inside one
        /// [`LONG_POLL_INTERVAL`].
        const OBSERVATION_WINDOW: Duration = Duration::from_millis(1500);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("healthy-rotation.json");
        write_token_file(&path, "healthy-A", 3600).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(CACHE_BUDGET, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("healthy-A")
        })
        .await
        .expect("the healthy token must be cached by the immediate t≈0 read");

        write_token_file(&path, "healthy-B", 3600).await;
        let picked_up = poll_until(OBSERVATION_WINDOW, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("healthy-B")
        })
        .await;
        assert!(
            picked_up.is_none(),
            "a provider whose token is an hour from expiry must be on the {LONG_POLL_INTERVAL:?} \
             steady cadence; the replacement was observed after {picked_up:?}, i.e. the task is \
             fast-polling unconditionally rather than only inside the stale window"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            0,
            "the stale-token phase must not arm for a token far outside its entry window — in a \
             correctly-configured deployment this INFO never fires at all: {output}"
        );
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            0,
            "no token expired, so no retirement WARN may fire: {output}"
        );

        // Liveness control: the task is alive and serving reads — the assertion
        // above is "steady cadence", not "polling task died".
        provider.force_refresh().await.expect("force_refresh");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("healthy-B".to_string()),
            "the polling task must still service force_refresh while dormant"
        );

        cancel.cancel();
    }

    /// The entry predicate is re-evaluated at *later* loop heads, not only once
    /// at warm-up: a token that starts outside the entry window and drifts into
    /// it arms the phase.
    ///
    /// This is what the `+ poll_interval` term in the entry lead buys (see
    /// [`StalePhase`]). The predicate can only be evaluated when the task
    /// reaches its loop head, and in steady state that happens once per
    /// `poll_interval` — so the window must open a full steady tick *before*
    /// the bare refresh-lead window does, or entry and the next steady read
    /// coincide and the phase is a no-op.
    ///
    /// A moderate poll interval is required here rather than
    /// [`LONG_POLL_INTERVAL`]: the entry lead is `expiry_skew + poll_interval`,
    /// so a 60 s interval would put the entry window 60 s wide and no fixture
    /// could start outside it without a 60 s test. The observable is therefore
    /// the entry INFO's absence-then-presence, not a cadence measurement.
    #[tokio::test]
    async fn stale_phase_entry_predicate_is_re_evaluated_at_later_loop_heads() {
        /// Steady interval, and half the entry lead.
        const DRIFT_POLL_INTERVAL: Duration = Duration::from_millis(200);
        /// Skew; entry lead is this plus `DRIFT_POLL_INTERVAL` = 400 ms.
        const DRIFT_SKEW: chrono::Duration = chrono::Duration::milliseconds(200);
        /// Token lifetime. Five times the 400 ms entry lead, so the entry
        /// window opens ~1600 ms after the fixture is written.
        const DRIFT_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(2000);
        /// Budget for the t≈0 read to land the drifting token in the cache.
        /// Spent only on a stall — a healthy run satisfies the predicate in
        /// low single-digit milliseconds — but it is spent *before* the
        /// negative assertion below, so it counts against that assertion's
        /// distance to the entry window. See [`NEGATIVE_ASSERTION_SLACK`].
        const CACHE_BUDGET: Duration = Duration::from_millis(400);
        /// Dwell before sampling the negative assertion, so several steady
        /// loop heads have evaluated the entry predicate and declined.
        const BEFORE_WINDOW: Duration = Duration::from_millis(400);
        /// Minimum distance the negative-assertion instant must keep from the
        /// instant the entry window opens.
        ///
        /// This test's negative assertion ("the phase has NOT armed yet") is
        /// the one that a *correct* implementation can fail if the sample
        /// lands too late, and its instant is budget-coupled: it occurs at
        /// worst `CACHE_BUDGET + BEFORE_WINDOW` after the fixture is written,
        /// because a stalled t≈0 read spends the cache budget before the dwell
        /// even starts. Sizing the token lifetime so the window opens a full
        /// [`NEGATIVE_ASSERTION_SLACK`] beyond that worst case is what
        /// decouples the two; the `assert!` below enforces the arithmetic so a
        /// future constant tweak cannot silently re-couple them.
        const NEGATIVE_ASSERTION_SLACK: Duration = Duration::from_millis(800);
        /// Budget for the entry once the window has opened. Measured from the
        /// negative assertion, so it must cover the remaining distance to the
        /// ~1600 ms window opening plus a steady tick — with slack of its own.
        const ENTRY_BUDGET: Duration = Duration::from_millis(2500);

        // `DRIFT_LIFETIME - (DRIFT_SKEW + DRIFT_POLL_INTERVAL)` after the
        // fixture write. Computed rather than written as a literal so the
        // guard below tracks the constants.
        let window_opens_at =
            Duration::from_millis((DRIFT_LIFETIME - DRIFT_SKEW).num_milliseconds() as u64)
                - DRIFT_POLL_INTERVAL;
        assert!(
            CACHE_BUDGET + BEFORE_WINDOW + NEGATIVE_ASSERTION_SLACK <= window_opens_at,
            "the negative assertion must land at least {NEGATIVE_ASSERTION_SLACK:?} before the \
             entry window opens ({window_opens_at:?} after the fixture write), but its worst-case \
             instant is {:?}; raise DRIFT_LIFETIME or shrink the budgets",
            CACHE_BUDGET + BEFORE_WINDOW
        );

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("drifts-into-window.json");
        write_token_file_in(&path, "drifting-token", DRIFT_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) =
            build_provider_with_skew_and_interval(path.clone(), DRIFT_POLL_INTERVAL, DRIFT_SKEW);

        poll_until(CACHE_BUDGET, || provider.inner.has_cached_token())
            .await
            .expect("the drifting token must be cached by the immediate t≈0 read");

        // Still outside the entry window: the token has ~1600 ms of life left
        // against a 400 ms lead, and several steady loop heads have already
        // evaluated the predicate and correctly declined.
        tokio::time::sleep(BEFORE_WINDOW).await;
        assert_eq!(
            marker_count(&captured, STALE_ENTRY_INFO_MARKER),
            0,
            "the phase must not arm while the cached token is outside the \
             `expiry_skew + poll_interval` entry window"
        );

        // ...and then it drifts in, at a loop head that is not the warm-up one.
        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect(
            "a token that drifts into the entry window must arm the phase at a later loop head; \
             evaluating the predicate only once at warm-up would leave a late rotation costing a \
             full poll interval (b.qam)",
        );

        cancel.cancel();
    }

    /// The `+ poll_interval` term in the entry lead, pinned as a **two-sided
    /// margin** on the cached token's remaining life at the arming instant.
    ///
    /// Its sibling above proves the predicate is re-evaluated at later loop
    /// heads, but it would still pass with the lead reduced to `refresh_lead`
    /// alone — the phase would just arm later. This test is the one that fails
    /// under that mutation.
    ///
    /// The argument. The phase arms at the first loop head at or after
    /// `expires_on - (refresh_lead + poll_interval)`, and in steady state loop
    /// heads are `poll_interval` apart. So at the arming instant the cached
    /// token's remaining life is in `(refresh_lead, refresh_lead +
    /// poll_interval]` — the `+ poll_interval` term is exactly the head start
    /// the phase gets before the bare lead window opens. Drop the term and the
    /// entry window opens at `expires_on - refresh_lead`, so arming can only
    /// ever be observed with remaining life *at or below* the lead. That is the
    /// b.qam no-op: the phase arms at the same instant the run-up it exists to
    /// get ahead of begins.
    ///
    /// Both sides are load-bearing. `margin > 0` is what fails the mutation
    /// above; `margin <= poll_interval + slack` is what keeps the test from
    /// passing vacuously against a build that armed the phase off some *wider*
    /// lead than the one this test configured — the role the old
    /// "…and it does eventually get withheld" control assertion played before
    /// the withholding accessor was deleted (b.mq8).
    ///
    /// Numbers: 1 s poll interval, 1 s lead, 3 s lifetime. Entry window opens
    /// at t≈1 s, the bare-lead window at t≈2 s — a full poll interval of
    /// margin, so the positive margin is not a race.
    ///
    /// Two things this deliberately does NOT observe: [`StalePhase`] itself
    /// (private, and task-local), and the entry INFO's own `secs_to_expiry`
    /// field (whole seconds, so a margin of ~1 s rounds unpredictably). The
    /// margin is computed here from `cached_token_expires_on()`, which is the
    /// same cached `expires_on` the loop head keyed its decision on.
    #[tokio::test]
    async fn stale_phase_arms_a_full_poll_interval_before_the_refresh_lead_window_opens() {
        /// Steady interval — and the whole margin under test. A moderate value
        /// (not [`LONG_POLL_INTERVAL`]) because it is one of the two terms in
        /// the entry lead being measured.
        const LEAD_POLL_INTERVAL: Duration = Duration::from_secs(1);
        /// The provider's proactive re-read lead: the *other* term in the entry
        /// lead, and the baseline the margin below is measured against. Read
        /// from this constant rather than from the provider, which no longer
        /// exposes it.
        const LEAD_SKEW: chrono::Duration = chrono::Duration::seconds(1);
        /// Token lifetime. Entry window opens at
        /// `lifetime - (lead + poll_interval)` = t≈1 s.
        const LEAD_LIFETIME: chrono::Duration = chrono::Duration::seconds(3);
        /// Budget for the entry INFO. Deliberately past the t≈2 s bare-lead
        /// window: under the mutation the phase still arms, just late, and this
        /// budget lets the test observe it and fail on the *margin* assertion
        /// with a diagnosis rather than on an ambiguous timeout.
        const ENTRY_BUDGET: Duration = Duration::from_millis(2500);
        /// Slack on the margin's upper bound. Absorbs the wall time between the
        /// fixture write and the task's first loop head plus one
        /// [`PROBE_INTERVAL`] of observation lag — both of which only *shrink*
        /// the observed margin, so this is pure headroom against runner jitter.
        /// A quarter of [`LEAD_POLL_INTERVAL`], so a phase armed off a lead even
        /// 25 % wider than configured still fails.
        const MARGIN_CEILING_SLACK: chrono::Duration = chrono::Duration::milliseconds(250);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("entry-lead-margin.json");
        write_token_file_in(&path, "lead-margin-token", LEAD_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) =
            build_provider_with_skew_and_interval(path.clone(), LEAD_POLL_INTERVAL, LEAD_SKEW);

        // Sample the remaining life inside the predicate, at the same probe
        // that first observes the entry INFO — so the recorded margin is the
        // one at (within one `PROBE_INTERVAL` of) the arming instant, not
        // whatever it happens to be several statements later.
        let mut margin_at_entry = None;
        let entered_after = poll_until(ENTRY_BUDGET, || {
            if marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0 {
                margin_at_entry = provider
                    .cached_token_expires_on()
                    .map(|expires_on| expires_on - Utc::now() - LEAD_SKEW);
                true
            } else {
                false
            }
        })
        .await
        .expect("the cached token must arm the stale-token phase before it expires");

        let margin = margin_at_entry.expect(
            "the phase can only arm off a cached expiry, so one must be readable at the arming \
             instant — a `None` here means the entry INFO fired against an empty cache",
        );
        assert!(
            margin > chrono::Duration::zero(),
            "the stale-token phase must arm while the cached token still has more than the \
             {LEAD_SKEW:?} refresh lead of life left; the observed margin was {margin} (entry \
             observed {entered_after:?} after construction). A non-positive margin is what an \
             entry lead of `refresh_lead` alone produces — the phase would arm at the same instant \
             the run-up it exists to get ahead of begins, making it a no-op on exactly the \
             deployments b.qam is about. The `+ poll_interval` term is what buys the \
             one-steady-tick head start."
        );
        assert!(
            margin
                <= chrono::Duration::from_std(LEAD_POLL_INTERVAL).expect("1 s is representable")
                    + MARGIN_CEILING_SLACK,
            "the phase must arm off THIS instance's entry lead — `refresh_lead + poll_interval`, \
             i.e. at most one {LEAD_POLL_INTERVAL:?} of margin past the {LEAD_SKEW:?} lead — but \
             the observed margin was {margin}. A margin this wide means the phase armed off some \
             wider lead than the one configured, which would let the positive assertion above pass \
             without the `+ poll_interval` term doing any of the work."
        );

        cancel.cancel();
    }

    /// Retirement on a fresher token restores the steady cadence and is
    /// silent.
    ///
    /// Two halves, both load-bearing: the phase must not keep fast-polling for
    /// the lifetime of the process once the replacement lands (a *third* token
    /// written afterwards must be invisible across several
    /// [`FAST_POLL_RETRY_INTERVAL`]s), and a recovering sidecar must not look
    /// like an operator event (no expiry WARN).
    #[tokio::test]
    async fn stale_phase_retires_silently_on_a_fresher_token() {
        const STALE_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(1500);
        const ENTRY_BUDGET: Duration = Duration::from_millis(500);
        const PICKUP_BUDGET: Duration = Duration::from_millis(1000);
        /// Four fast-poll intervals after the retirement, still far inside one
        /// [`LONG_POLL_INTERVAL`].
        const OBSERVATION_WINDOW: Duration = Duration::from_millis(1000);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("silent-retirement.json");
        write_token_file_in(&path, "stale-A", STALE_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect("the near-expiry token must arm the stale-token phase");

        write_token_file(&path, "fresh-B", 3600).await;
        poll_until(PICKUP_BUDGET, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("fresh-B")
        })
        .await
        .expect("the replacement must be picked up on the fast cadence");

        // The phase is keyed on the cached `expires_on`, which has now changed,
        // so it must be back to `Idle` — and `Idle` means the steady arm, not
        // the 250 ms one.
        write_token_file(&path, "fresh-C", 3600).await;
        let picked_up = poll_until(OBSERVATION_WINDOW, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("fresh-C")
        })
        .await;
        assert!(
            picked_up.is_none(),
            "once a fresher token lands the phase must retire and the task must return to the \
             {LONG_POLL_INTERVAL:?} steady cadence; a third token was observed after \
             {picked_up:?}, so the task is still re-reading every {FAST_POLL_RETRY_INTERVAL:?}"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            0,
            "a sidecar that rotated in time is not an operator event; the expiry WARN must not \
             fire on the happy retirement: {output}"
        );
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            1,
            "the far-future replacements are outside the entry window, so the phase must not \
             re-arm: {output}"
        );

        // Liveness control.
        provider.force_refresh().await.expect("force_refresh");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("fresh-C".to_string()),
            "the polling task must still service force_refresh after the phase retires"
        );

        cancel.cancel();
    }

    /// The chronically-short-lead sidecar: a replacement token that is ITSELF
    /// inside the entry window must re-arm the phase in the **same** loop
    /// iteration that retires it.
    ///
    /// The sibling test above covers the healthy shape — the replacement is far
    /// from expiry, so retirement leaves the task `Idle` on the steady cadence.
    /// This one covers the deployment SDD §15.4 actually warns about: a sidecar
    /// whose refresh lead is *shorter* than `expiry_skew + poll_interval`, so
    /// every token it writes is born already inside the entry window. There the
    /// loop head must do both things in one pass — reset to `Idle` because the
    /// cached expiry changed, then immediately re-evaluate the entry predicate
    /// against the new expiry and arm again — which is exactly what the
    /// production loop head does (the reset and the entry check are sequential
    /// `if`s over one `cached_expiry` read, not an `if / else`).
    ///
    /// Why it needs its own test: a refactor that made the two mutually
    /// exclusive (an `else if`, or moving the entry check above the reset)
    /// still passes every other stale-phase test in this module, because they
    /// all use far-future replacements. Here it would defer the re-arm to the
    /// *next* loop head — and once the phase is `Idle` the next loop head is
    /// the [`LONG_POLL_INTERVAL`] steady tick, 60 s out. That is b.qam,
    /// silently reintroduced for precisely the deployment that needs the fix.
    ///
    /// The 60 s steady interval does double duty here: it is the discriminator
    /// (nothing observed inside a 1 s budget can be a steady tick) *and* it
    /// makes the entry lead ~60.5 s wide, so every fixture in this test is
    /// "still stale" by the phase's own predicate without needing a sub-second
    /// lifetime that could expire mid-test and confound the count.
    #[tokio::test]
    async fn stale_phase_re_arms_in_the_same_iteration_for_a_still_stale_replacement() {
        /// Lifetime shared by all three tokens. Inside the ~60.5 s entry lead
        /// (so each one re-arms the phase) yet far longer than the whole test
        /// (so no token expires and no retirement WARN can confound the entry
        /// count).
        const STILL_STALE_LIFETIME: chrono::Duration = chrono::Duration::seconds(5);
        /// Budget for the t≈0 read plus the loop head that arms the phase.
        const ENTRY_BUDGET: Duration = Duration::from_millis(500);
        /// Budget for the re-arm. The replacement is picked up on the
        /// [`FAST_POLL_RETRY_INTERVAL`] cadence and the re-arm happens at the
        /// very next loop head, so ~250 ms is expected — and this is 60× below
        /// [`LONG_POLL_INTERVAL`], which is what makes "well inside one poll
        /// interval" a real claim rather than an arithmetic tautology.
        const RE_ARM_BUDGET: Duration = Duration::from_millis(1000);
        /// Budget for the third token, proving the fast cadence survived the
        /// re-arm rather than merely being re-logged.
        const PICKUP_BUDGET: Duration = Duration::from_millis(1000);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("short-lead-sidecar.json");
        write_token_file_in(&path, "short-lead-A", STILL_STALE_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect("the first short-lead token must arm the stale-token phase");

        // The sidecar rotates — but its lead is short, so the replacement is
        // itself inside the entry window.
        write_token_file_in(&path, "short-lead-B", STILL_STALE_LIFETIME).await;
        let re_armed = poll_until(RE_ARM_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 1
        })
        .await;
        assert!(
            re_armed.is_some(),
            "a replacement that is itself inside the entry window must re-arm the phase in the \
             same loop iteration that retires it; no second entry INFO appeared within \
             {RE_ARM_BUDGET:?}, so the re-arm was deferred to the next loop head — which, with \
             the phase back to Idle, is the {LONG_POLL_INTERVAL:?} steady tick (b.qam)"
        );
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("short-lead-B".to_string()),
            "the re-arm must be keyed on the replacement the fast cadence actually cached"
        );

        // ...and the fast cadence is still running under the re-armed phase, so
        // the next rotation is picked up sub-second too.
        write_token_file_in(&path, "short-lead-C", STILL_STALE_LIFETIME).await;
        let picked_up = poll_until(PICKUP_BUDGET, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("short-lead-C")
        })
        .await;
        assert!(
            picked_up.is_some(),
            "the re-armed phase must keep fast-polling: a third short-lead rotation was not \
             cached within {PICKUP_BUDGET:?}, so the task fell back to the \
             {LONG_POLL_INTERVAL:?} steady cadence after the first replacement"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            3,
            "one arming per distinct cached expiry, and all three of this sidecar's tokens are \
             born inside the entry window: {output}"
        );
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            0,
            "every rotation landed before its predecessor expired, so no retirement WARN may \
             fire — a WARN here means a fixture expired mid-test and the entry counts above are \
             measuring something else: {output}"
        );

        cancel.cancel();
    }

    /// The phase is retired by a cache that `force_refresh()` filled, not only
    /// by one a stale-arm read filled.
    ///
    /// Loop-head coverage, the stale-phase analog of
    /// `force_refresh_populated_cache_retires_the_fast_poll_phase`: the
    /// retirement checks live at the loop head precisely so every path that can
    /// change the cache retires the phase. Placed inside the stale timer arm
    /// instead, a token delivered by the Epic 5 401-retry path would leave the
    /// task fast-polling until the token it replaced expired.
    #[tokio::test]
    async fn force_refresh_delivered_token_retires_the_stale_phase() {
        /// Long enough that the cached token cannot expire during the test —
        /// this test is about the *fresher-token* retirement, not the expiry
        /// one, so an expiry WARN here would be a confound.
        const STALE_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(3000);
        const ENTRY_BUDGET: Duration = Duration::from_millis(500);
        /// Four fast-poll intervals, far inside one [`LONG_POLL_INTERVAL`].
        const OBSERVATION_WINDOW: Duration = Duration::from_millis(1000);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("force-refresh-stale.json");
        write_token_file_in(&path, "stale-A", STALE_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect("the near-expiry token must arm the stale-token phase");

        // The entry INFO is emitted at the loop head immediately after the t≈0
        // read, and `stale_interval.reset()` puts the first stale-arm read a
        // full `FAST_POLL_RETRY_INTERVAL` later. `poll_until` observes the INFO
        // within one `PROBE_INTERVAL` of emission, so the blocking write and
        // the refresh below land with ~245 ms of margin before any stale-arm
        // read could see the file — and `select!`'s `biased` ordering prefers a
        // queued refresh request over an already-ready timer regardless. The
        // load that populates the cache is therefore the `force_refresh()` one.
        write_token_file_blocking(&path, "refresh-B", 3600);
        provider
            .force_refresh()
            .await
            .expect("force_refresh must load the freshly written file");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("refresh-B".to_string()),
            "force_refresh must be the load that replaces the near-expiry token"
        );

        write_token_file_blocking(&path, "second-token", 3600);
        let picked_up = poll_until(OBSERVATION_WINDOW, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("second-token")
        })
        .await;
        assert!(
            picked_up.is_none(),
            "a cache refreshed through force_refresh() must retire the stale-token phase at the \
             loop-head check; a later token was observed after {picked_up:?}, so the task is \
             still on the {FAST_POLL_RETRY_INTERVAL:?} cadence"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            0,
            "the replacement landed before expiry, so the retirement must be silent: {output}"
        );

        // Liveness control.
        provider.force_refresh().await.expect("force_refresh");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("second-token".to_string()),
            "the polling task must still service force_refresh after the phase retires"
        );

        cancel.cancel();
    }

    /// The other retirement: the cached token expires with no replacement. One
    /// WARN — exactly one, however many loop heads follow — and no token bytes
    /// in it (SR-6.2).
    ///
    /// "Exactly once" is not free here: the WARN branch runs at the loop head
    /// on every iteration where the cache holds an expired token, so it is
    /// `StalePhase::RetiredFor(E)` that makes it one-shot. This test drives
    /// extra loop heads with `force_refresh()` calls — the file is unchanged,
    /// so each re-read leaves the cached `expires_on` (and therefore the keyed
    /// expiry) exactly where it was — which is what makes the count assertion
    /// non-vacuous.
    #[tokio::test]
    async fn stale_phase_warns_exactly_once_when_the_token_expires_unreplaced() {
        /// Canary token: no contiguous 8-byte slice of it may appear in the
        /// captured output. The retirement WARN carries `role`, `path` and
        /// derived durations only.
        const SECRET_TOKEN: &str = "tkn-stale-expiry-canary-0123456789ab";
        /// Sub-second so the test does not have to wait out a real token
        /// lifetime; the phase self-terminates at this instant.
        const DYING_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(600);
        const ENTRY_BUDGET: Duration = Duration::from_millis(500);
        /// The lifetime plus several fast-poll intervals of slack for the loop
        /// head that observes the expiry.
        const WARN_BUDGET: Duration = Duration::from_millis(2000);
        /// Extra loop heads driven after the WARN. Three is enough to
        /// distinguish "one-shot" from "once per loop head".
        const EXTRA_LOOP_HEADS: usize = 3;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("expires-unreplaced.json");
        write_token_file_in(&path, SECRET_TOKEN, DYING_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect("the dying token must arm the stale-token phase");

        poll_until(WARN_BUDGET, || {
            marker_count(&captured, STALE_EXPIRY_WARN_MARKER) > 0
        })
        .await
        .expect(
            "a cached token that expires with no replacement must retire the phase with one WARN \
             — without it the phase would fast-poll for the process lifetime",
        );

        // Drive further loop heads over the still-expired cache.
        for _ in 0..EXTRA_LOOP_HEADS {
            // The file is unchanged, so this re-read re-caches the SAME
            // `expires_on` and `RetiredFor` must continue to suppress.
            provider
                .force_refresh()
                .await
                .expect("force_refresh must still succeed against the unchanged file");
        }

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            1,
            "the expiry retirement must WARN exactly once per distinct cached expiry, not once \
             per loop head: {output}"
        );
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            1,
            "a retired expiry must not re-arm the phase: {output}"
        );
        assert_no_token_bytes(&output, SECRET_TOKEN);

        cancel.cancel();
    }

    /// Warm restart onto an already-expired token: the WARN fires even though
    /// the phase was never entered.
    ///
    /// Same operator-visible fault as the expiry retirement — a cached token
    /// nobody replaced — reached by a different path: a pod restart whose
    /// sidecar-written file is already stale. The t≈0 read populates the cache
    /// with a dead token, so the loop head takes the expired branch before the
    /// entry branch is ever considered (`fast_polled_secs = 0`).
    #[tokio::test]
    async fn warm_restart_into_an_already_expired_token_warns_without_entering_the_phase() {
        const WARN_BUDGET: Duration = Duration::from_millis(1000);
        /// Three is enough to distinguish "one-shot" from "once per loop head".
        const EXTRA_LOOP_HEADS: usize = 3;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("warm-restart-expired.json");
        // Already 10 s dead at construction time.
        write_token_file(&path, "dead-on-arrival", -10).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(WARN_BUDGET, || {
            marker_count(&captured, STALE_EXPIRY_WARN_MARKER) > 0
        })
        .await
        .expect(
            "a warm restart onto an already-expired token is the same operator-visible fault as \
             an unreplaced expiry and must WARN, even though the phase never entered",
        );

        for _ in 0..EXTRA_LOOP_HEADS {
            provider
                .force_refresh()
                .await
                .expect("force_refresh must still succeed against the unchanged file");
        }

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            1,
            "the warm-restart WARN must be one-shot, not once per loop head: {output}"
        );
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            0,
            "an already-expired cached token takes the expired branch; the phase must never be \
             entered for it: {output}"
        );

        cancel.cancel();
    }

    /// Degenerate-case bound: the expiry retirement is one-shot for that
    /// expiry — the phase neither re-arms nor re-WARNs for the same dead token,
    /// however many loop heads run past it.
    ///
    /// That suppression is what keeps a permanently-dead sidecar from
    /// re-entering the phase forever; the *cadence* it leaves running is
    /// separately bounded by [`STARTUP_FAST_POLL_WINDOW`] (see
    /// `post_expiry_fast_poll_stops_after_the_startup_window`).
    ///
    /// **What b.d6b changed here.** This test used to assert the complement of
    /// its final observation: that a token written after the retirement was
    /// *not* picked up on the fast cadence, because retirement dropped straight
    /// back to the steady poll. That is precisely the behaviour b.d6b removed —
    /// reverting to a 15 s cadence at the moment every consumer goes
    /// credential-less is backwards, so `RetiredFor` now keeps the 250 ms arm
    /// live for a bounded window past expiry. The replacement is therefore
    /// picked up fast, and this test asserts that instead: same fixture, same
    /// [`LONG_POLL_INTERVAL`] discriminator, inverted verdict.
    ///
    /// The named invariant survives untouched, and is now carried by the exact
    /// marker counts: across the whole post-expiry window the phase produces
    /// exactly one entry INFO and exactly one retirement WARN. A `RetiredFor`
    /// that re-armed per loop head would show up as ~4 extra of each per second
    /// — the `stale_phase_re_arms_on_the_next_rotation` sibling proves the
    /// suppression is expiry-keyed rather than latched.
    ///
    /// The `force_refresh()` drain that used to precede the observation window
    /// is gone with the assertion that needed it: the retirement's
    /// `reset_immediately()` catch-up read was a *contaminant* of a negative
    /// observation, but the observation is now positive and the steady arm is
    /// guard-disabled for the whole window anyway, so nothing to drain.
    #[tokio::test]
    async fn stale_phase_does_not_re_arm_for_the_same_expired_token() {
        /// Must comfortably exceed [`ENTRY_BUDGET`]: the entry INFO can only be
        /// observed while the cached token is still *alive*, so a lifetime at
        /// or near the budget makes a stalled t≈0 read cache an
        /// already-expired token, take the expired branch, and never arm the
        /// phase at all — failing a correct implementation. 800 ms against a
        /// 500 ms budget leaves the observation 300 ms of headroom while
        /// keeping the expiry inside [`WARN_BUDGET`].
        const DYING_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(800);
        const ENTRY_BUDGET: Duration = Duration::from_millis(500);
        const WARN_BUDGET: Duration = Duration::from_millis(2000);
        /// Budget for the post-expiry pickup: four [`FAST_POLL_RETRY_INTERVAL`]s,
        /// and 60× below [`LONG_POLL_INTERVAL`]. That ratio is what makes an
        /// observed pickup attributable to the post-expiry fast cadence and to
        /// nothing else — the steady arm is guard-disabled for the whole
        /// window, and even once it is not, its next deadline is 60 s out.
        const RECOVERY_BUDGET: Duration =
            Duration::from_millis(4 * FAST_POLL_RETRY_INTERVAL.as_millis() as u64);
        /// Extra observation held open after the pickup, during which a
        /// `RetiredFor` that re-armed per loop head would emit ~4 more entry
        /// INFOs and retirement WARNs per second against the exact counts below.
        const RE_ARM_OBSERVATION_WINDOW: Duration = Duration::from_millis(500);

        // The pickup budget must stay strictly inside the steady interval, or a
        // pass could be a steady tick rather than the fast cadence.
        assert!(
            RECOVERY_BUDGET < LONG_POLL_INTERVAL,
            "the post-expiry pickup budget must be strictly below the steady interval or it \
             stops discriminating; shrink RECOVERY_BUDGET"
        );

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("no-re-arm.json");
        write_token_file_in(&path, "dead-token", DYING_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect("the dying token must arm the stale-token phase");
        poll_until(WARN_BUDGET, || {
            marker_count(&captured, STALE_EXPIRY_WARN_MARKER) > 0
        })
        .await
        .expect("the dying token must retire the phase at its expiry");

        // The replacement lands inside the bounded post-expiry window, where
        // the 250 ms arm is still the live one and the steady arm is
        // guard-disabled. No drain is needed: the retirement's
        // `reset_immediately()` catch-up read cannot fire until the window
        // closes, 20 s out.
        write_token_file(&path, "replacement-token", 3600).await;
        let picked_up = poll_until(RECOVERY_BUDGET, || {
            provider.cached_bearer_if_unexpired().as_deref() == Some("replacement-token")
        })
        .await;
        assert!(
            picked_up.is_some(),
            "a replacement written after the expiry retirement must be picked up on the \
             post-expiry fast cadence within {RECOVERY_BUDGET:?} (b.d6b): total credential loss is \
             the MOST urgent moment to be hunting for the replacement, and reverting to the \
             {LONG_POLL_INTERVAL:?} steady poll there is what this window exists to prevent"
        );

        // Hold the window open past the pickup. `RetiredFor` re-arming per loop
        // head — the failure mode this test is named for — would show up here
        // as extra markers against the exact counts below.
        tokio::time::sleep(RE_ARM_OBSERVATION_WINDOW).await;

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            1,
            "the retired expiry must not WARN again, however many fast-cadence loop heads run \
             past it: {output}"
        );
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            1,
            "the retired expiry must not re-arm the phase — and the replacement, 3600 s from \
             expiry, must not arm it either: {output}"
        );

        // Liveness control: the task is still servicing requests, so the
        // marker counts above are a real "did not happen", not a dead task.
        provider.force_refresh().await.expect("force_refresh");
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("replacement-token".to_string()),
            "the polling task must still service force_refresh after the expiry retirement"
        );

        cancel.cancel();
    }

    /// The post-expiry fast cadence is **bounded** by
    /// [`STARTUP_FAST_POLL_WINDOW`] (b.d6b): past that window the task reverts
    /// to the steady `poll_interval`, so a sidecar that has stopped writing
    /// entirely cannot leave it stat-ing a dead path four times a second for
    /// the process lifetime.
    ///
    /// The complement of the pickup assertion in
    /// `stale_phase_does_not_re_arm_for_the_same_expired_token`, and the reason
    /// `RetiredFor` carries a `tokio::time::Instant` anchor rather than just an
    /// expiry.
    ///
    /// **Why this one CAN use a paused clock** where the rest of the
    /// stale-phase group cannot. Those tests need a cached token to travel from
    /// alive to dead, which is `Utc::now()` movement that tokio does not
    /// virtualise. This one starts from a token that is *already* dead when the
    /// sidecar file is first read — the warm-restart path, which reaches
    /// `RetiredFor` at the very first loop head with no wall-clock travel at
    /// all. Everything after that is pure `tokio::time` (the window anchor, the
    /// fast cadence, the steady interval), so 20 s of window plus a 60 s steady
    /// interval cost milliseconds of real time instead of 80 s.
    ///
    /// Structure:
    ///
    /// 1. Expired token A → the retirement WARN, which *is* the window anchor
    ///    (it is emitted by the same loop head that sets `RetiredFor`).
    /// 2. **Non-vacuity, inside the window**: a second *already-expired* token
    ///    B is picked up within a few fast intervals. A no-op fast arm would
    ///    fail here rather than passing step 4 for free. B's distinct
    ///    `expires_on` re-anchors the window, so the burn below is measured
    ///    from a known instant.
    /// 3. Burn [`STARTUP_FAST_POLL_WINDOW`] of virtual time with the file
    ///    unchanged.
    /// 4. A now-*valid* token C must NOT be picked up on the fast cadence…
    /// 5. …but the steady tick must still eventually cache it — the polling
    ///    task reverted, it did not stop.
    ///
    /// The warm-restart entry also means the phase is never `Active` here, so
    /// the entry-INFO count stays zero throughout: every read in this test is
    /// post-expiry fast polling or a steady tick, never the pre-expiry half.
    #[tokio::test(start_paused = true)]
    async fn post_expiry_fast_poll_stops_after_the_startup_window() {
        /// Virtual steps to burn the post-expiry window, plus margin for the
        /// anchor landing up to a tick after the observed WARN. Derived from
        /// the constants so a constant change cannot silently under-burn.
        const WINDOW_BURN_STEPS: usize =
            (STARTUP_FAST_POLL_WINDOW.as_millis() / FAST_POLL_RETRY_INTERVAL.as_millis()) as usize
                + 8;
        /// 20 fast-poll intervals (5 virtual seconds). Used both as the
        /// in-window pickup budget (step 2) and the post-window negative probe
        /// (step 4), so the two spans are directly comparable: the same span
        /// that comfortably contains a pickup must contain none.
        const FAST_CADENCE_PROBE_STEPS: usize = 20;
        /// Enough steps to cover two steady intervals after the fallback.
        const STEADY_CADENCE_STEPS: usize =
            (LONG_POLL_INTERVAL.as_millis() / FAST_POLL_RETRY_INTERVAL.as_millis()) as usize * 2
                + 40;
        /// Budget for a loop-head marker to appear. Generous relative to the
        /// ≤ 2-step jitter `step_paused_until` documents, so a slow runner
        /// cannot fail it.
        const MARKER_STEPS: usize = 40;

        // Everything before the liveness check must land strictly inside one
        // `LONG_POLL_INTERVAL` of task start: the steady `interval` is
        // `reset()` at the warm-up transition and never re-anchored afterwards
        // (the warm-restart path skips `reset_immediately()`, which is gated on
        // the phase having been `Active`). If the burn plus both probe spans
        // overran it, the steady read would legitimately cache token C and
        // step 4 would fail for a reason unrelated to the bound under test.
        assert!(
            FAST_POLL_RETRY_INTERVAL
                * ((WINDOW_BURN_STEPS + 2 * FAST_CADENCE_PROBE_STEPS + 2 * MARKER_STEPS) as u32)
                < LONG_POLL_INTERVAL,
            "the burn and probe spans must fit strictly inside one {LONG_POLL_INTERVAL:?} steady \
             interval; shrink them or lengthen LONG_POLL_INTERVAL"
        );

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("post-expiry-window.json");
        // Warm restart onto a token that is already 10 s dead: the t≈0 read
        // caches it and the first loop head retires straight to `RetiredFor`,
        // anchoring the post-expiry window without any wall-clock travel.
        write_token_file(&path, "dead-on-arrival-a", -10).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        assert!(
            step_paused_until(MARKER_STEPS, || marker_count(
                &captured,
                STALE_EXPIRY_WARN_MARKER
            ) > 0)
            .await
            .is_some(),
            "the already-expired token must produce the retirement WARN that anchors the \
             post-expiry window"
        );

        // Step 2 — non-vacuity. A second already-expired token, distinct only
        // in its `expires_on`, must be picked up on the fast cadence while the
        // window is open. `cached_token_expires_on()` is the observable:
        // `cached_bearer_if_unexpired()` reports `None` for both tokens (both
        // are dead), so it cannot tell them apart.
        let expiry_a = provider
            .cached_token_expires_on()
            .expect("the t≈0 read must have cached token A");
        write_token_file(&path, "dead-on-arrival-b", -5).await;
        // Bind the step count instead of discarding it. The probe's whole job
        // is to prove the *post-expiry fast arm* ran: a regressed arm
        // (`RetiredFor => false` in `wants_fast_poll`) still flips this
        // observable eventually, just off the `LONG_POLL_INTERVAL` steady
        // tick — which is why "some pickup happened" is not the assertion,
        // "a pickup inside the fast-cadence budget happened" is.
        // `step_paused_until`'s budget is hard, so the `unwrap_or_else` is
        // already that bound; stating the ceiling explicitly keeps this
        // probe's non-vacuity local rather than resting on the helper's
        // internals.
        let fast_pickup_steps = step_paused_until(FAST_CADENCE_PROBE_STEPS, || {
            provider.cached_token_expires_on() != Some(expiry_a)
        })
        .await
        .unwrap_or_else(|| {
            panic!(
                "inside the post-expiry window the task must still be re-reading every \
                 {FAST_POLL_RETRY_INTERVAL:?}; without this the negative probe below would pass on \
                 a fast arm that never ran at all"
            )
        });
        assert!(
            fast_pickup_steps <= FAST_CADENCE_PROBE_STEPS,
            "token B was picked up only after {fast_pickup_steps} × {FAST_POLL_RETRY_INTERVAL:?}, \
             past the {FAST_CADENCE_PROBE_STEPS}-step fast-cadence budget — that is a \
             {LONG_POLL_INTERVAL:?} steady-tick pickup, not the post-expiry fast arm, and it \
             leaves the negative probe below vacuous"
        );
        assert!(
            step_paused_until(MARKER_STEPS, || marker_count(
                &captured,
                STALE_EXPIRY_WARN_MARKER
            ) > 1)
            .await
            .is_some(),
            "token B's distinct expiry must produce its own retirement WARN — that WARN is the \
             instant the burn below is measured from"
        );

        // Step 3 — burn the window with the file unchanged.
        for _ in 0..WINDOW_BURN_STEPS {
            tokio::time::sleep(FAST_POLL_RETRY_INTERVAL).await;
        }

        // One `force_refresh()` round trip before the file changes, so no
        // fast-poll read from the tail of the window can still be in flight
        // when token C lands — that read would cache C legitimately and fail
        // step 4 for a reason unrelated to the bound. The reply is sent after
        // the task's read completes, so receiving it proves the task is back in
        // `select!`. It re-reads the unchanged file, so the keyed expiry does
        // not move and `RetiredFor` keeps suppressing.
        provider
            .force_refresh()
            .await
            .expect("force_refresh must still succeed against the unchanged file");

        // Step 4 — the fast cadence is gone. The `None` below is the cached
        // token B having *actually expired*, not a lead withholding a live one:
        // the bearer accessor knows only expiry (b.mq8), so a flip to `Some`
        // here can only mean the fast arm read token C.
        write_token_file(&path, "post-window-token", 3600).await;
        for _ in 0..FAST_CADENCE_PROBE_STEPS {
            tokio::time::sleep(FAST_POLL_RETRY_INTERVAL).await;
            assert_eq!(
                provider.cached_bearer_if_unexpired(),
                None,
                "after {STARTUP_FAST_POLL_WINDOW:?} past the expiry retirement the task must be \
                 back on the {LONG_POLL_INTERVAL:?} steady cadence, not still re-reading every \
                 {FAST_POLL_RETRY_INTERVAL:?} — the post-expiry window is what bounds a sidecar \
                 that has stopped writing entirely"
            );
        }

        // Step 5 — reverted, not stopped. The cache flip is published only
        // after the steady tick's blocking read returns; that read holds off
        // auto-advance while it runs, so the step budget measures virtual time
        // the task actually saw (see `step_paused_until`).
        assert!(
            step_paused_until(STEADY_CADENCE_STEPS, || provider
                .cached_bearer_if_unexpired()
                .is_some())
            .await
            .is_some(),
            "the steady-state poll never picked the token up after the post-expiry window closed \
             — the polling task stopped reading entirely rather than reverting"
        );
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("post-window-token".to_string()),
            "the steady-state poll must eventually cache the now-valid token"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            2,
            "exactly one retirement WARN per distinct cached expiry — the whole burn runs against \
             an unchanged expiry and must stay silent: {output}"
        );
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            0,
            "every token here is either already dead or 3600 s from expiry, so the pre-expiry \
             half of the phase must never arm: {output}"
        );

        cancel.cancel();
    }

    /// The suppression is keyed on the cached expiry, not latched: a
    /// *different* cached expiry re-arms the phase, and can produce its own
    /// retirement WARN.
    ///
    /// The complement of
    /// `stale_phase_does_not_re_arm_for_the_same_expired_token`. Together they
    /// pin `RetiredFor(E)` exactly: refuse re-entry for `E`, allow it for `E'`.
    /// A `bool` "already warned" flag — the obvious simpler implementation —
    /// passes the sibling test and fails this one, going permanently silent
    /// after the first late rotation.
    ///
    /// `force_refresh()` is what delivers the second token: after the first
    /// retirement the task is parked on the [`LONG_POLL_INTERVAL`] steady tick
    /// (that is the sibling test's assertion), so nothing else would drive a
    /// loop head inside this test's budget.
    #[tokio::test]
    async fn stale_phase_re_arms_on_the_next_rotation() {
        /// Same reasoning as the sibling test's constant, with one more round
        /// trip to cover: rotation B's lifetime starts when the fixture is
        /// written and its entry INFO is not observable until the write, the
        /// `force_refresh()` round trip and the loop head that follows it have
        /// all completed. 800 ms covers that against a 500 ms
        /// [`ENTRY_BUDGET`]; 400 ms did not reliably.
        const DYING_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(800);
        const ENTRY_BUDGET: Duration = Duration::from_millis(500);
        const WARN_BUDGET: Duration = Duration::from_millis(2000);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("re-arms.json");
        write_token_file_in(&path, "rotation-A", DYING_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, _metrics, cancel) = build_provider_with_skew_and_interval(
            path.clone(),
            LONG_POLL_INTERVAL,
            SUB_SECOND_SKEW,
        );

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect("rotation A must arm the phase");
        poll_until(WARN_BUDGET, || {
            marker_count(&captured, STALE_EXPIRY_WARN_MARKER) > 0
        })
        .await
        .expect("rotation A must retire the phase at its expiry");

        // Rotation B: a genuinely different `expires_on`, delivered through the
        // one path that reaches a loop head while the task is parked.
        write_token_file_in(&path, "rotation-B", DYING_LIFETIME).await;
        provider
            .force_refresh()
            .await
            .expect("force_refresh must load rotation B");

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 1
        })
        .await
        .expect(
            "a different cached expiry must re-arm the phase — `RetiredFor` refuses re-entry only \
             for the expiry it names, and a latched flag would leave every rotation after the \
             first unprotected",
        );
        poll_until(WARN_BUDGET, || {
            marker_count(&captured, STALE_EXPIRY_WARN_MARKER) > 1
        })
        .await
        .expect("rotation B's expiry must produce its own retirement WARN");

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(STALE_ENTRY_INFO_MARKER).count(),
            2,
            "exactly one arming per distinct cached expiry: {output}"
        );
        assert_eq!(
            output.matches(STALE_EXPIRY_WARN_MARKER).count(),
            2,
            "exactly one retirement WARN per distinct cached expiry: {output}"
        );

        cancel.cancel();
    }

    /// The stale arm's reads are paced by the same [`ParseFailureGate`] the
    /// cold-start arm uses.
    ///
    /// A token file that goes malformed while the cached token is expiring is
    /// exactly the fault the fast cadence amplifies: reads at
    /// [`FAST_POLL_RETRY_INTERVAL`] would emit one WARN and one
    /// `live_edit_service_token_parse_errors_total` increment each. The gate's
    /// `min_interval` *is* `poll_interval`, so the bound is "at most one per
    /// steady interval, whichever fast phase is running".
    ///
    /// This test needs a *moderate* poll interval rather than
    /// [`LONG_POLL_INTERVAL`], because the poll interval is simultaneously the
    /// gate period: with a 60 s gate period every window is trivially
    /// single-emission and the assertion would be vacuous. 1 s makes the
    /// bound 1 emission where the ungated cadence would produce 4.
    ///
    /// The steady arm is disabled while the phase is `Active`, and its
    /// `interval` is `reset()` at the retirement, so no ungated steady read can
    /// contaminate the count before the snapshot below.
    #[tokio::test]
    async fn stale_fast_poll_reads_are_paced_by_the_parse_failure_gate() {
        /// Gate period == steady interval. Four times the fast cadence, so an
        /// ungated stale arm would emit ~4× as often as a gated one.
        const GATE_POLL_INTERVAL: Duration = Duration::from_secs(1);
        /// Entry lead is this plus `GATE_POLL_INTERVAL` = 1.3 s.
        const GATE_SKEW: chrono::Duration = chrono::Duration::milliseconds(300);
        /// Inside the 1.3 s entry lead, so the phase arms at the first loop
        /// head; and the instant the phase self-terminates.
        const DYING_LIFETIME: chrono::Duration = chrono::Duration::milliseconds(1000);
        const ENTRY_BUDGET: Duration = Duration::from_millis(500);
        const WARN_BUDGET: Duration = Duration::from_millis(2000);
        /// Ceiling on gated emissions. The failing reads span roughly one gate
        /// period (first failure at ~250 ms, retirement at ~1000–1250 ms), so
        /// the expected count is 1; 2 absorbs a slow runner straddling a second
        /// gate period. The ungated count over the same span is 4.
        const MAX_GATED_INCREMENTS: u64 = 2;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("gated-stale-reads.json");
        write_token_file_in(&path, "gated-token", DYING_LIFETIME).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, metrics, cancel) =
            build_provider_with_skew_and_interval(path.clone(), GATE_POLL_INTERVAL, GATE_SKEW);

        poll_until(ENTRY_BUDGET, || {
            marker_count(&captured, STALE_ENTRY_INFO_MARKER) > 0
        })
        .await
        .expect("the dying token must arm the stale-token phase");
        assert_eq!(
            parse_error_count(&metrics),
            0,
            "the t≈0 read saw valid JSON, so the gate is unstamped and the first failing read \
             below is guaranteed to emit"
        );
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("gated-token".to_string()),
            "precondition: the cache holds the good, still-live token, so the retirement observed \
             below is that token's expiry and not some other transition"
        );

        // The sidecar starts writing garbage. Blocking so no scheduler yield
        // separates the entry observation from the corruption. Every stale-arm
        // read from here to the retirement fails to parse; the cache keeps the
        // good (dying) token by the previous-good-cache invariant, so the
        // expiry retirement still fires on schedule.
        std::fs::write(&path, b"{ not valid json").expect("write malformed fixture");

        poll_until(WARN_BUDGET, || {
            marker_count(&captured, STALE_EXPIRY_WARN_MARKER) > 0
        })
        .await
        .expect("the cached token must still retire at its expiry despite the malformed file");
        // Snapshot immediately: past the retirement the ungated steady arm
        // resumes and would add un-gated increments.
        let gated_increments = parse_error_count(&metrics);

        assert!(
            gated_increments >= 1,
            "the gate's first call always passes, so a malformed file must never be fully silent"
        );
        assert!(
            gated_increments <= MAX_GATED_INCREMENTS,
            "the stale arm must share the cold-start arm's ParseFailureGate: over ~one \
             {GATE_POLL_INTERVAL:?} gate period of {FAST_POLL_RETRY_INTERVAL:?} reads the counter \
             moved {gated_increments} times, where the gate bounds it at \
             {MAX_GATED_INCREMENTS} and an ungated arm would produce ~4"
        );

        drop(_dispatch_guard);
        cancel.cancel();
    }

    // ── Parse failure: warn + counter + recovery (acceptance #6, #9) ────────

    /// One failed read increments `live_edit_service_token_parse_errors_total`
    /// by exactly one — no more, no less.
    ///
    /// Deliberately poller-free: `Inner` is constructed by hand and
    /// `read_and_update` is driven directly, so the number of reads is exactly
    /// the number this test performs. Every provider-level test has a live
    /// polling task whose cadence can straddle a counter snapshot (an
    /// unparseable file never populates the cache, so the fast-poll phase keeps
    /// re-reading every [`FAST_POLL_RETRY_INTERVAL`]), which is why those tests
    /// can only assert `>`. The exact-count invariant lives here instead.
    #[tokio::test]
    async fn each_failed_parse_increments_the_counter_exactly_once() {
        /// Number of hand-driven failing reads. Two is the minimum that
        /// distinguishes "increments by exactly 1 per read" from "increments
        /// once, ever".
        const READS: u64 = 2;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("malformed-exact.json");
        tokio::fs::write(&path, b"{ not valid json")
            .await
            .expect("write fixture");

        let metrics = fresh_metrics();
        let inner = Inner {
            cache: RwLock::new(None),
            path,
            metrics: Arc::clone(&metrics),
            // No polling task is spawned here, so nothing ever receives on
            // this channel; the field exists only to satisfy the struct.
            force_refresh_tx: mpsc::channel(1).0,
        };

        assert_eq!(
            parse_error_count(&metrics),
            0,
            "a fresh Metrics instance must start with no recorded parse errors"
        );

        for expected in 1..=READS {
            let err = read_and_update(&inner, ReadCaller::ForceRefresh)
                .await
                .expect_err("a malformed token file must surface as Err");
            assert!(
                matches!(err, ServiceTokenError::Parse(_)),
                "expected ServiceTokenError::Parse, got {err:?}"
            );
            assert_eq!(
                parse_error_count(&metrics),
                expected,
                "read #{expected} must move the parse-error counter by exactly 1"
            );
            assert!(
                !inner.has_cached_token(),
                "a failed parse must never populate the cache"
            );
        }
    }

    #[tokio::test]
    async fn malformed_json_logs_warn_and_increments_parse_counter() {
        // The test token bytes — any 8-byte slice of this string MUST NOT
        // appear in captured log output. Hand-picked so it is improbable
        // to collide with anything tokio / tracing emits.
        const SECRET_TOKEN: &str = "tkn-deadbeef-0123456789abcdef-canary";

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("malformed.json");

        // Write a file whose contents include the canary token but is NOT
        // valid JSON (the trailing garbage breaks JSON parse). If the
        // implementation ever logs the file bytes, the canary will appear
        // in the captured output and the assertion below will fire.
        let body = format!(
            "{{\"access_token\":\"{SECRET_TOKEN}\",\"expires_on\":\"not-rfc3339\"}} <-- GARBAGE TAIL"
        );
        tokio::fs::write(&path, body).await.expect("write fixture");

        let (_dispatch_guard, captured) = capture_logs();

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();

        // Use a LONG poll interval so the steady-state `interval.tick()`
        // never fires inside the test window — the first steady tick is a
        // full 60 s out. Reads still happen more than once, though: this
        // fixture never parses, so the cache never populates and the
        // cold-start fast-poll phase stays active for the whole test,
        // re-reading every `FAST_POLL_RETRY_INTERVAL` (250 ms). That is why the
        // counter assertion below is `>` rather than `== baseline + 1`; the
        // exact-per-read count is pinned by
        // `each_failed_parse_increments_the_counter_exactly_once`.
        let provider = ServiceTokenProvider::new(
            path.clone(),
            Duration::from_secs(60),
            metrics.clone(),
            cancel.clone(),
        );

        // Wait for the polling task's IMMEDIATE first read to observe the
        // malformed file and bump the counter (cold-start scenario — there is
        // no previous-good cache to preserve).
        //
        // A bounded probe on the counter, not a fixed sleep: the read runs on
        // the blocking pool, so under a loaded runner (this module runs 37
        // tests in parallel) it can take far longer than any sleep short enough
        // to be worth writing, and the old unconditional 50 ms wait flaked
        // exactly there. The budget is a ceiling a healthy run exits in
        // single-digit milliseconds, spent in full only on a genuine hang.
        const BASELINE_BUDGET: Duration = Duration::from_millis(2000);
        poll_until(BASELINE_BUDGET, || parse_error_count(&metrics) >= 1)
            .await
            .expect(
                "the immediate first read must bump the parse-error counter (cold-start scenario, \
                 malformed file present)",
            );
        let baseline = parse_error_count(&metrics);

        // Drive a SECOND deterministic read via force_refresh — this
        // must produce at least one additional parse-error increment.
        let err = provider
            .force_refresh()
            .await
            .expect_err("malformed file must surface as Err");
        assert!(
            matches!(err, ServiceTokenError::Parse(_)),
            "expected ServiceTokenError::Parse, got {err:?}"
        );

        // Counter moved past the baseline.
        //
        // `>` rather than `== baseline + 1`: this file NEVER parses, so the
        // cache never populates and the cold-start fast-poll phase stays
        // active for the whole test, re-reading every `FAST_POLL_RETRY_INTERVAL`
        // (250 ms). The 50 ms baseline sleep above plus any scheduling stall
        // between the baseline snapshot and this assertion can straddle a
        // fast-poll read, which would add a second increment that has nothing
        // to do with the invariant under test. That invariant is "a failed
        // parse increments the counter and surfaces `Parse`" — the exact
        // per-window read count is cadence, not contract.
        // `good_cached_then_malformed_preserves_previous_token` makes the same
        // argument for the same reason.
        let after = parse_error_count(&metrics);
        assert!(
            after > baseline,
            "parse-error counter must increment after a force_refresh on a malformed file \
             (baseline={baseline}, after={after})"
        );

        // Cache stays empty (no previous-good state to preserve in this
        // cold-start scenario).
        assert_eq!(provider.cached_bearer_if_unexpired(), None);

        // Drop the dispatcher guard so it stops capturing further events
        // before we assert on the captured bytes.
        drop(_dispatch_guard);

        let guard = captured.lock().unwrap();
        let output = String::from_utf8_lossy(&guard);

        // A WARN event must have fired. The fmt subscriber prints the
        // level name; pretty-format uses "WARN".
        assert!(
            output.contains("WARN"),
            "expected at least one WARN event in captured output, got: {output}"
        );

        // Bearer-bytes-never-logged invariant — pinned at the 8-byte
        // slice granularity called out in the Subtask acceptance criteria.
        assert_no_token_bytes(&output, SECRET_TOKEN);

        cancel.cancel();
    }

    #[tokio::test]
    async fn malformed_then_valid_recovers_without_restart() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("recover.json");

        // Cold-start malformed write.
        tokio::fs::write(&path, b"garbage not json")
            .await
            .expect("write malformed");

        let (provider, metrics, cancel) = build_provider(path.clone());
        await_parse_errors_above(&metrics, 0).await;
        assert_eq!(provider.cached_bearer_if_unexpired(), None);

        // Overwrite with a valid token.
        write_token_file(&path, "recovered-token", 3600).await;
        await_bearer(&provider, "recovered-token").await;

        // A valid read must not increment the parse-error counter.
        //
        // Stated as "the counter is frozen from the recovery onward" rather
        // than "it equals the pre-recovery snapshot": the malformed file is
        // re-read every [`TEST_POLL_INTERVAL`] until the cache populates, so
        // any snapshot taken *before* the recovery races the reads still
        // arriving in between and the equality would be measuring cadence, not
        // the invariant. After the recovery every read parses, so the freeze is
        // exact.
        let counter_after_recovery = parse_error_count(&metrics);
        for _ in 0..3 {
            await_one_poll().await;
        }
        assert_eq!(
            parse_error_count(&metrics),
            counter_after_recovery,
            "parse-error counter must not increment on valid follow-up reads"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn good_cached_then_malformed_preserves_previous_token() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("cache-preserve.json");

        // Step 1 + 2: write valid JSON with token-A and let the polling
        // task pick it up. We deliberately use a SHORT interval here +
        // a settle window so the polling task's immediate first tick (and
        // any straggler ticks) all observe the VALID file and do NOT
        // bump the parse-error counter. After the settle window, we
        // snapshot the counter — that snapshot is the baseline against
        // which a single subsequent malformed read must increment by
        // exactly 1.
        write_token_file(&path, "token-A", 3600).await;

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new(
            path.clone(),
            TEST_POLL_INTERVAL,
            metrics.clone(),
            cancel.clone(),
        );
        // Step 3: wait for the cache to flip to Some("token-A").
        await_bearer(&provider, "token-A").await;
        // Counter has NOT moved — the file was valid for every poll so
        // far. This is the precondition the next step relies on.
        let counter_before_corruption = parse_error_count(&metrics);
        assert_eq!(
            counter_before_corruption, 0,
            "no parse errors should be observed while the file holds valid JSON"
        );

        // Step 4: corrupt the file, then wait for the counter to move — the
        // observable that the corruption has actually been *read*. Waiting on
        // the counter rather than on a fixed poll-interval sleep is what keeps
        // step 5's cache assertion from passing vacuously against a corruption
        // no read has seen yet.
        //
        // The counter assertion is `>` rather than `== +1` because a loaded
        // runner can fire several reads across this window; the invariant being
        // pinned is "a malformed write ticks the counter AND does not evict the
        // previous-good cache", and the exact read count is cadence, not
        // contract. "Exactly 1 per read" is pinned separately by
        // `each_failed_parse_increments_the_counter_exactly_once`, which drives
        // `read_and_update` by hand with no polling task in the picture.
        tokio::fs::write(&path, b"corrupted not json")
            .await
            .expect("write malformed");
        await_parse_errors_above(&metrics, counter_before_corruption).await;

        // Step 5 + 6: cache MUST still surface token-A — the
        // previous-good-cache invariant (SDD §14.6, module docs) means a
        // malformed update does NOT evict a previously-good token.
        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("token-A".to_string()),
            "previous-good-cache invariant: malformed update must NOT evict token-A"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn rfc3339_parse_failure_increments_parse_counter_same_as_malformed_json() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bad-expires.json");

        // Well-formed JSON shape but `expires_on` is NOT RFC3339. serde
        // will fail at the chrono `DateTime<Utc>` deserialize step — which
        // surfaces through the same `serde_json::from_str` Err path as
        // structural malformedness, so the counter SHOULD tick.
        let body = r#"{"access_token":"tok","expires_on":"not-a-timestamp"}"#;
        tokio::fs::write(&path, body).await.expect("write fixture");

        let (provider, metrics, cancel) = build_provider(path);
        await_parse_errors_above(&metrics, 0).await;

        // And the cache stays empty (cold start).
        assert_eq!(provider.cached_bearer_if_unexpired(), None);

        cancel.cancel();
    }

    // ── cached_bearer_if_unexpired(): the sole bearer accessor (b.d6b, b.mq8) ──
    //
    // One accessor, one policy: serve any cached token that has not *actually*
    // expired, with the per-instance refresh lead ignored entirely. The pair
    // below is the whole contract — a token deep inside a ≥ 3 min lead is
    // served, an expired one is not — and the *contrast* between them is what
    // makes each non-vacuous.
    //
    // This section used to be half of a two-accessor matrix: a `wide_skew_*`
    // group pinned a second, *withholding* accessor that returned `None` inside
    // the lead, and these tests were its mirror image. b.e49 retired that
    // accessor's last production consumer (storage-api checkpoint adapter,
    // health supervisor) and b.mq8 deleted it, taking the mirror group with it:
    // asserting a withholding policy no production path can reach was pinning a
    // behaviour the service does not have.

    /// The headline b.d6b contract: a token deep inside the ≥ 3 min refresh
    /// lead is still served.
    ///
    /// The lead is proactive-re-read guidance, not a validity floor. This is
    /// the fixture the deleted withholding accessor answered `None` for — the
    /// answer that failed every Valkey `AUTH` for the width of the lead — so it
    /// is the one worth keeping pinned.
    #[tokio::test]
    async fn cached_bearer_if_unexpired_serves_a_token_inside_the_refresh_lead() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("token-inside.json");
        // Expires 150 s from now — INSIDE the 3 min lead, and 150 s away from
        // being a dead credential.
        write_token_file(&path, "token-soon", 150).await;

        let (provider, _metrics, cancel) = build_provider_with_skew(path, WIDE_EXPIRY_SKEW);
        await_cached(&provider).await;

        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("token-soon".to_string()),
            "a cached token 150 s from expiry must be served even though the instance's re-read \
             lead is 180 s: the lead says when to go looking for the replacement, not when the \
             current credential stops working (b.d6b)"
        );

        cancel.cancel();
    }

    /// The boundary that *does* withhold: an actually-expired token.
    ///
    /// The complement of the test above. Without this, "ignores the lead" could
    /// be implemented as "ignores expiry", which would hand fred a dead
    /// credential instead of failing `AUTH`.
    #[tokio::test]
    async fn cached_bearer_if_unexpired_returns_none_for_an_expired_token() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("expired.json");
        // Expired 10 s ago.
        write_token_file(&path, "old-token", -10).await;

        let (provider, _metrics, cancel) = build_provider_with_skew(path, WIDE_EXPIRY_SKEW);
        // The cache must actually hold the token, or `None` below would be the
        // never-read state and prove nothing about the expiry check.
        await_cached(&provider).await;

        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            None,
            "an expired token is not a credential; ignoring the lead must not mean ignoring \
             expiry"
        );

        cancel.cancel();
    }

    /// `cached_token_expires_on()` projects the cached expiry — `None` until
    /// the first successful parse, then the token's own `expires_on`.
    ///
    /// The accessor exists so consumers (and their tests) can observe *that a
    /// token is cached* without touching bearer bytes; it projects only the
    /// `Copy` timestamp (SR-6.2). `src/storage/token_file_credentials.rs`'s
    /// tests use it as the "the polling task has read the file" precondition
    /// that its module surface previously could not express.
    #[tokio::test]
    async fn cached_token_expires_on_reports_none_then_the_cached_expiry() {
        const LIFETIME_SECS: i64 = 3600;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("expires-on.json");

        // Bracket the fixture write: it computes `expires_on = Utc::now() +
        // LIFETIME_SECS` at some instant inside this pair, so the accessor's
        // answer must land inside the bracket — an exact statement, not a
        // tolerance.
        let before = Utc::now();
        write_token_file(&path, "token-A", LIFETIME_SECS).await;
        let after = Utc::now();

        let (provider, _metrics, cancel) = build_provider(path);
        await_cached(&provider).await;

        let expires_on = provider
            .cached_token_expires_on()
            .expect("a populated cache must report its token's expiry");
        let lifetime = chrono::Duration::seconds(LIFETIME_SECS);
        assert!(
            expires_on >= before + lifetime && expires_on <= after + lifetime,
            "the reported expiry must be the fixture's own `expires_on` ({expires_on}), which was \
             computed between {before} and {after} plus {LIFETIME_SECS} s"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn cached_token_expires_on_is_none_before_the_first_successful_read() {
        let dir = tempdir().expect("tempdir");
        // Never written: the polling task's NotFound branch leaves the cache
        // empty, so there is no expiry to project.
        let path = dir.path().join("never-written.json");

        let (provider, _metrics, cancel) = build_provider(path);
        await_one_poll().await;

        assert_eq!(
            provider.cached_token_expires_on(),
            None,
            "an empty cache must report no expiry rather than a sentinel timestamp"
        );

        cancel.cancel();
    }

    // ── live_edit_service_token_expires_in_seconds (b.d6b) ──────────────────
    //
    // `metrics.rs` owns the recorder's own contract (lazy family, per-role
    // series, no clamping). The two tests here own the *wiring*: that the
    // polling task actually calls it, under the role label it was constructed
    // with, from the token it actually cached. Asserted as a bounded range
    // rather than an exact value — the contract is "a derived time-to-expiry
    // for this token", not a specific number.

    #[tokio::test]
    async fn polling_task_records_positive_time_to_expiry_under_its_role_label() {
        const LIFETIME_SECS: i64 = 3600;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("gauge-alive.json");
        write_token_file(&path, "token-A", LIFETIME_SECS).await;

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        // `new_with_role`, not `build_provider`: the label is half of what this
        // test pins, and the shared harness constructor passes
        // `ROLE_UNSPECIFIED`.
        let provider = ServiceTokenProvider::new_with_role(
            path,
            TEST_POLL_INTERVAL,
            ServiceTokenProvider::ROLE_VALKEY,
            metrics.clone(),
            cancel.clone(),
        );
        await_bearer(&provider, "token-A").await;

        assert!(
            poll_until(POLL_PICKUP_BUDGET, || expires_in_seconds(
                &metrics,
                ServiceTokenProvider::ROLE_VALKEY
            )
            .is_some())
            .await
            .is_some(),
            "the polling task must record the time-to-expiry gauge for its own role once a token \
             is cached; the family was still absent after {POLL_PICKUP_BUDGET:?}"
        );

        let secs = expires_in_seconds(&metrics, ServiceTokenProvider::ROLE_VALKEY)
            .expect("the series exists per the wait above");
        assert!(
            secs > 0.0 && secs <= LIFETIME_SECS as f64,
            "a live token must report a positive time-to-expiry no greater than its own \
             {LIFETIME_SECS} s lifetime, i.e. a value derived from the cached token rather than a \
             constant; got {secs}"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn polling_task_records_negative_time_to_expiry_for_an_expired_token() {
        /// How long ago the fixture died. Bounds the assertion below: the
        /// reported value must be at or just past `-DEAD_FOR_SECS`, so it can
        /// only have come from this token's `expires_on`.
        const DEAD_FOR_SECS: i64 = 10;
        /// Slack for the wall time between writing the fixture and the loop
        /// head that records the gauge.
        const OBSERVATION_SLACK_SECS: f64 = 60.0;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("gauge-dead.json");
        write_token_file(&path, "token-dead", -DEAD_FOR_SECS).await;

        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        let provider = ServiceTokenProvider::new_with_role(
            path,
            TEST_POLL_INTERVAL,
            ServiceTokenProvider::ROLE_STORAGE_API,
            metrics.clone(),
            cancel.clone(),
        );
        await_cached(&provider).await;

        assert!(
            poll_until(POLL_PICKUP_BUDGET, || expires_in_seconds(
                &metrics,
                ServiceTokenProvider::ROLE_STORAGE_API
            )
            .is_some())
            .await
            .is_some(),
            "an expired cached token is still a cached token; the gauge must be recorded for it \
             — that is the whole point of a signed series"
        );

        let secs = expires_in_seconds(&metrics, ServiceTokenProvider::ROLE_STORAGE_API)
            .expect("the series exists per the wait above");
        assert!(
            secs <= -(DEAD_FOR_SECS as f64)
                && secs >= -(DEAD_FOR_SECS as f64) - OBSERVATION_SLACK_SECS,
            "a token that died {DEAD_FOR_SECS} s ago must report a negative time-to-expiry at or \
             just past -{DEAD_FOR_SECS}: the zero crossing is the operator-meaningful part of this \
             series and must not be clamped away on the write path; got {secs}"
        );

        cancel.cancel();
    }

    // ── Rotation prolongation headroom (b.is1) ──────────────────────────────
    //
    // The provider measures, per observed rotation, how long before the
    // *outgoing* token's expiry the replacement was picked up, records it as
    // `live_edit_service_token_refresh_headroom_seconds{role}`, and WARNs once
    // when that headroom lands under the instance's prolongation floor. The
    // floor is per-instance and optional: the Valkey instance carries the
    // maximum re-AUTH interval its credential consumer schedules, the
    // storage-api instance carries `None`.
    //
    // Every test here asserts the gauge **value** in a bounded range, not just
    // its presence, and every fixture pairs an outgoing token with a
    // deliberately different replacement lifetime. That pairing is the whole
    // discriminator: a range containing the outgoing lifetime and excluding the
    // replacement's is what distinguishes "measured against the token being
    // retired" from "measured against the token just cached" — the latter would
    // report a comfortable headroom on exactly the deployments the WARN exists
    // to catch.
    //
    // Real clock, like the stale-phase group above and for the same reason: the
    // headroom is `Utc::now()`-derived and a paused tokio clock does not
    // virtualise that. Cost is bounded differently, though — nothing here waits
    // out a token lifetime, so each test is a few poll intervals plus, for the
    // negative observations, a fixed few hundred milliseconds.

    /// Prolongation floor used by these tests: **the** production Valkey value,
    /// read from its single definition site rather than restated.
    ///
    /// [`crate::storage::client::VALKEY_TOKEN_ROTATION_FANOUT_CEILING`] is the
    /// worst-case latency from a rotation being cached to every pooled
    /// connection having re-AUTHed with it, and is exactly what `main.rs`
    /// passes to the Redis-scoped instance. Sourcing it here means a change to
    /// the shipped ceiling moves these fixtures with it; a literal would leave
    /// them asserting a band production had stopped using, which is the failure
    /// mode where the tests stay green and the WARN threshold is wrong.
    ///
    /// It is deliberately **not** `VALKEY_TOKEN_REAUTH_MAX_INTERVAL`: b.7td
    /// split the two apart, since under the rotation gate a tick only sends an
    /// `AUTH` when the gate opens, so the fan-out takes two tick intervals, not
    /// one. Reading the tick ceiling here would put the fixtures a factor of two
    /// below the threshold production actually WARNs at.
    ///
    /// This is a `#[cfg(test)]` module of the *same crate*, so the `use` is not
    /// the dependency edge the design avoids: what `auth` must not do is read
    /// `storage` in **production** code (hence the floor being threaded in from
    /// the wiring layer as a constructor argument), and nothing here changes
    /// that — the code under test still learns the floor only from its caller.
    const TEST_PROLONGATION_FLOOR: chrono::Duration = chrono::Duration::seconds(
        crate::storage::client::VALKEY_TOKEN_ROTATION_FANOUT_CEILING.as_secs() as i64,
    );

    /// Lifetime of an outgoing token whose replacement therefore lands with
    /// headroom far *under* [`TEST_PROLONGATION_FLOOR`] — the misconfigured
    /// sidecar the WARN exists for.
    const SHORT_OUTGOING_LIFETIME_SECS: i64 = 30;

    /// Lifetime of the *middle* token in the two-rotation fixture: the token
    /// that is first a replacement and then, one write later, the outgoing one.
    ///
    /// Under [`TEST_PROLONGATION_FLOOR`] like [`SHORT_OUTGOING_LIFETIME_SECS`],
    /// so both rotations WARN — but far enough from it that the two recorded
    /// headrooms fall in ranges that do not overlap once
    /// [`HEADROOM_SLACK_SECS`] is allowed on each. That separation is what makes
    /// "the second rotation was measured afresh" distinguishable from "the first
    /// value was re-reported".
    const SECOND_OUTGOING_LIFETIME_SECS: i64 = 120;

    /// Lifetime of an outgoing token whose replacement lands with headroom
    /// comfortably *over* [`TEST_PROLONGATION_FLOOR`] — the healthy sidecar.
    const AMPLE_OUTGOING_LIFETIME_SECS: i64 = 3600;

    /// Lifetime of the *replacement* token in every rotation fixture.
    /// Deliberately unequal to either outgoing lifetime so the recorded value
    /// says which of the two expiries it was derived from.
    const REPLACEMENT_LIFETIME_SECS: i64 = 7200;

    /// Slack subtracted from the outgoing lifetime when bounding an observed
    /// headroom below. Covers the wall time between writing the fixture and the
    /// loop-head pass that observes the rotation; generous against the ~5 s
    /// worst case of the two [`POLL_PICKUP_BUDGET`] waits a test can spend.
    const HEADROOM_SLACK_SECS: f64 = 20.0;

    /// Negative observation window for "and it does not keep firing / does not
    /// fire at all": ~15 loop heads at [`TEST_POLL_INTERVAL`]. Spent in full on
    /// every run — the elapsed time is the assertion.
    const NO_MORE_OUTPUT_WINDOW: Duration = Duration::from_millis(300);

    #[tokio::test]
    async fn rotation_below_prolongation_floor_records_headroom_and_warns_once() {
        // Canaries rather than "token-A" / "token-B": the WARN is new log
        // output on a path that has the cached token in hand, so it is worth
        // re-pinning SR-6.2 against it directly. Opaque by construction — the
        // check below is an 8-byte sliding window over the token, so a canary
        // containing an English word ("…-replacement-…") collides with the
        // WARN's own prose and fails on a run where nothing leaked.
        const OUTGOING: &str = "tkn-9f3c1a7d5e2b8046-canary";
        const REPLACEMENT: &str = "tkn-4b8e6d2f0a9c7135-canary";

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("headroom-under-floor.json");
        write_token_file(&path, OUTGOING, SHORT_OUTGOING_LIFETIME_SECS).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, metrics, cancel) =
            build_provider_with_floor(path.clone(), Some(TEST_PROLONGATION_FLOOR));
        await_bearer(&provider, OUTGOING).await;

        // The rotation. The sidecar replaced the token only
        // `SHORT_OUTGOING_LIFETIME_SECS` before it would have died — well inside
        // the [`TEST_PROLONGATION_FLOOR`] band a periodic re-AUTH needs to carry
        // the new credential onto a live connection.
        write_token_file(&path, REPLACEMENT, REPLACEMENT_LIFETIME_SECS).await;
        let secs = await_headroom_change(&metrics, ServiceTokenProvider::ROLE_VALKEY, None).await;

        assert!(
            secs > SHORT_OUTGOING_LIFETIME_SECS as f64 - HEADROOM_SLACK_SECS
                && secs <= SHORT_OUTGOING_LIFETIME_SECS as f64,
            "the recorded headroom must be derived from the OUTGOING token's expiry \
             (~{SHORT_OUTGOING_LIFETIME_SECS} s away), not the replacement's \
             ({REPLACEMENT_LIFETIME_SECS} s) and not a constant; got {secs}"
        );

        // One WARN per rotation, not one per loop-head pass. Without this
        // window an implementation that re-warns on every pass while the
        // headroom sits under the floor would pass the count check below purely
        // because the test read the buffer too early — and in production would
        // emit at the poll cadence for the rest of the token's life.
        assert!(
            poll_until(NO_MORE_OUTPUT_WINDOW, || marker_count(
                &captured,
                HEADROOM_WARN_MARKER
            ) > 1)
            .await
            .is_none(),
            "the prolongation WARN must fire once per observed rotation; it repeated within \
             {NO_MORE_OUTPUT_WINDOW:?}, i.e. it is keyed on the loop head rather than on the \
             rotation"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(HEADROOM_WARN_MARKER).count(),
            1,
            "exactly one prolongation WARN for one rotation under the floor: {output}"
        );
        assert!(
            output.contains(&format!(
                "recommended_floor_secs={}",
                TEST_PROLONGATION_FLOOR.num_seconds()
            )),
            "the WARN must name the floor the observed headroom fell short of, otherwise the \
             operator cannot tell how far short: {output}"
        );
        // The other half of "how far short", and the number the message's own
        // remediation sentence points the operator at ("raise … to at least the
        // observed_headroom_secs value in this line") — so the line is unusable
        // without it. Pinned against `secs`, the value the gauge took, rather
        // than against a range: both come from the one `headroom` computed at
        // the observation site, so any divergence means the WARN and the metric
        // have started reporting different rotations.
        assert!(
            output.contains(&format!("observed_headroom_secs={}", secs as i64)),
            "the WARN must report the headroom it actually observed, and it must be the same \
             value the gauge recorded ({secs}): {output}"
        );
        // The role disambiguates which provider warned. With a second instance
        // wired (storage-api) an unlabelled line would leave an operator unable
        // to tell which credential's rotation was late.
        assert!(
            output.contains(&format!("role=\"{}\"", ServiceTokenProvider::ROLE_VALKEY)),
            "the WARN must carry the role label so the line names which provider rotated late: \
             {output}"
        );
        assert!(
            output.contains("LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS"),
            "the WARN must name the operator lever on this side of the deployment — the sidecar's \
             own lead is not a live-edit config value, so this is the only knob the message can \
             point at concretely: {output}"
        );
        // SR-6.2 for the new log line: neither the retired token nor the one
        // that replaced it may appear in the output that reports the rotation.
        assert_no_token_bytes(&output, OUTGOING);
        assert_no_token_bytes(&output, REPLACEMENT);

        cancel.cancel();
    }

    #[tokio::test]
    async fn rotation_above_prolongation_floor_records_headroom_without_warning() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("headroom-over-floor.json");
        write_token_file(&path, "healthy-outgoing", AMPLE_OUTGOING_LIFETIME_SECS).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, metrics, cancel) =
            build_provider_with_floor(path.clone(), Some(TEST_PROLONGATION_FLOOR));
        await_bearer(&provider, "healthy-outgoing").await;

        // A healthy sidecar: the replacement lands an hour before the outgoing
        // token dies, an order of magnitude clear of
        // [`TEST_PROLONGATION_FLOOR`].
        write_token_file(&path, "healthy-replacement", REPLACEMENT_LIFETIME_SECS).await;
        let secs = await_headroom_change(&metrics, ServiceTokenProvider::ROLE_VALKEY, None).await;

        assert!(
            secs > AMPLE_OUTGOING_LIFETIME_SECS as f64 - HEADROOM_SLACK_SECS
                && secs <= AMPLE_OUTGOING_LIFETIME_SECS as f64,
            "a healthy rotation must still be *measured* — the gauge is recorded on every \
             rotation, and only the WARN is conditional — at a value derived from the outgoing \
             token's ~{AMPLE_OUTGOING_LIFETIME_SECS} s remaining life; got {secs}"
        );

        assert!(
            poll_until(NO_MORE_OUTPUT_WINDOW, || marker_count(
                &captured,
                HEADROOM_WARN_MARKER
            ) > 0)
            .await
            .is_none(),
            "a rotation with headroom over the floor is the correctly-configured case and must be \
             silent; a WARN here would fire on every healthy deployment and train operators to \
             ignore the one signal this ticket added"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(HEADROOM_WARN_MARKER).count(),
            0,
            "no prolongation WARN may fire for a rotation above the floor: {output}"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn rotation_after_the_outgoing_token_expired_records_negative_headroom() {
        /// How long before the provider started the outgoing token had already
        /// died. Bounds the assertion below, so the recorded value can only
        /// have come from that token's expiry.
        const DEAD_FOR_SECS: i64 = 10;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("headroom-negative.json");
        // Warm restart onto a token that is already dead, then a replacement.
        // This is the worst case the signed series exists to name: the
        // replacement did not prolong anything, because there was nothing left
        // to prolong — every connection AUTHed with the old token was dropped
        // at its expiry and re-established.
        write_token_file(&path, "dead-outgoing", -DEAD_FOR_SECS).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, metrics, cancel) =
            build_provider_with_floor(path.clone(), Some(TEST_PROLONGATION_FLOOR));
        // An expired token is still a *cached* token, and the rotation
        // observation is keyed on the cache — the bearer accessor declines a
        // dead token, so this is the only usable precondition.
        await_cached(&provider).await;

        write_token_file(&path, "late-replacement", REPLACEMENT_LIFETIME_SECS).await;
        let secs = await_headroom_change(&metrics, ServiceTokenProvider::ROLE_VALKEY, None).await;

        assert!(
            secs <= -(DEAD_FOR_SECS as f64)
                && secs >= -(DEAD_FOR_SECS as f64) - HEADROOM_SLACK_SECS,
            "a replacement picked up after the outgoing token had been dead for \
             {DEAD_FOR_SECS} s must record a headroom at or just past -{DEAD_FOR_SECS}: the sign \
             is what distinguishes 'prolonged with little to spare' from 'the pool churned', so \
             it must not be clamped at the measurement site either; got {secs}"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(HEADROOM_WARN_MARKER).count(),
            1,
            "a negative headroom is unambiguously under any floor and must WARN exactly once, \
             like any other under-floor rotation: {output}"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn rotation_without_prolongation_floor_records_headroom_and_never_warns() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("headroom-no-floor.json");
        // The SAME under-floor fixture the first test warns on — the only
        // difference here is the absent floor, so an implementation that
        // WARNed on headroom alone rather than on `Some(floor)` fails.
        write_token_file(&path, "storage-outgoing", SHORT_OUTGOING_LIFETIME_SECS).await;

        let (_dispatch_guard, captured) = capture_logs();
        let metrics = fresh_metrics();
        let cancel = CancellationToken::new();
        // `new_with_role` is the production storage-api shape: it delegates with
        // `None`, so this also covers the delegation rather than only the
        // explicit-`None` path.
        let provider = ServiceTokenProvider::new_with_role(
            path.clone(),
            TEST_POLL_INTERVAL,
            ServiceTokenProvider::ROLE_STORAGE_API,
            metrics.clone(),
            cancel.clone(),
        );
        // `await_cached`, not `await_bearer`: the rotation observation is keyed
        // on the cached `expires_on` that the loop head reads, not on what the
        // bearer accessor hands out, so the cache is the precondition that
        // matches the thing under test.
        await_cached(&provider).await;

        write_token_file(&path, "storage-replacement", REPLACEMENT_LIFETIME_SECS).await;
        let secs =
            await_headroom_change(&metrics, ServiceTokenProvider::ROLE_STORAGE_API, None).await;

        assert!(
            secs > SHORT_OUTGOING_LIFETIME_SECS as f64 - HEADROOM_SLACK_SECS
                && secs <= SHORT_OUTGOING_LIFETIME_SECS as f64,
            "an instance with no floor must still record the headroom — the gauge is how the \
             storage-api rotation is observable at all; got {secs}"
        );
        assert!(
            refresh_headroom_seconds(&metrics, ServiceTokenProvider::ROLE_VALKEY).is_none(),
            "the point must land under this instance's own role label; a series under \
             role=\"valkey\" would silently merge the two providers' rotations"
        );

        assert!(
            poll_until(NO_MORE_OUTPUT_WINDOW, || marker_count(
                &captured,
                HEADROOM_WARN_MARKER
            ) > 0)
            .await
            .is_none(),
            "an instance with no prolongation floor has no re-AUTH band to stay ahead of, so it \
             must never emit the prolongation WARN however short the headroom"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(HEADROOM_WARN_MARKER).count(),
            0,
            "floor `None` suppresses the WARN entirely, not merely for large headrooms: {output}"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn first_token_load_is_not_a_rotation_and_records_no_headroom() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("headroom-first-load.json");
        // Under the floor, so a first load mistaken for a rotation would both
        // record a point and WARN — the failure this test is shaped to catch.
        write_token_file(&path, "only-token", SHORT_OUTGOING_LIFETIME_SECS).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, metrics, cancel) =
            build_provider_with_floor(path, Some(TEST_PROLONGATION_FLOOR));
        await_bearer(&provider, "only-token").await;

        // Spend ~15 loop heads with a single token ever cached. This window is
        // what fails an implementation that records on every pass once a token
        // is present (i.e. one missing the `prev != curr` guard) as well as one
        // that treats `None → Some` as a rotation: both would have produced a
        // point inside it.
        assert!(
            poll_until(NO_MORE_OUTPUT_WINDOW, || refresh_headroom_seconds(
                &metrics,
                ServiceTokenProvider::ROLE_VALKEY
            )
            .is_some())
            .await
            .is_none(),
            "a provider that has cached exactly one token has observed no rotation; recording a \
             headroom point here would report the token's own remaining life as if it were a \
             prolongation window"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(HEADROOM_WARN_MARKER).count(),
            0,
            "no rotation, no prolongation WARN — a cold start must not look like a late \
             rotation: {output}"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn malformed_rewrite_is_not_observed_as_a_rotation() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("headroom-malformed.json");
        write_token_file(&path, "good-token", SHORT_OUTGOING_LIFETIME_SECS).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, metrics, cancel) =
            build_provider_with_floor(path.clone(), Some(TEST_PROLONGATION_FLOOR));
        await_bearer(&provider, "good-token").await;

        // A sidecar writing a torn or truncated file is the case the
        // rotation predicate must not confuse with a rotation: the
        // previous-good-cache invariant leaves `expires_on` untouched, so the
        // cached expiry does not change and nothing is observed. Waiting on the
        // parse counter is what makes this non-vacuous — it is the observable
        // that a read has actually *seen* the malformed file.
        tokio::fs::write(&path, b"corrupted not json")
            .await
            .expect("write malformed");
        await_parse_errors_above(&metrics, 0).await;

        assert_eq!(
            provider.cached_bearer_if_unexpired(),
            Some("good-token".to_string()),
            "precondition: the malformed write must leave the previous-good token cached, which \
             is exactly why its expiry is unchanged"
        );
        assert!(
            refresh_headroom_seconds(&metrics, ServiceTokenProvider::ROLE_VALKEY).is_none(),
            "a failed parse leaves the cached expiry untouched and so is not a rotation; \
             recording a point here would report a headroom for a token that never arrived"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(HEADROOM_WARN_MARKER).count(),
            0,
            "an unparseable file is already reported by the parse-error counter and its own WARN; \
             it must not also be reported as a late rotation: {output}"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn two_consecutive_rotations_each_record_and_warn() {
        // The recurrence test. Every other test in this section observes at most
        // one rotation, so all of them pass against an implementation that
        // latches — a `warned: bool`, a `first rotation only` guard, a
        // `last_cached_expiry` left un-updated after the first observation. Such
        // an implementation would report the *second* rotation of a long-lived
        // pod not at all: the gauge would freeze on the first value and the WARN
        // would never fire again, so a sidecar that degraded after the first
        // rotation would be invisible for the life of the process. That is the
        // steady state of a real deployment, not an edge case — a 24 h token
        // lifetime means every rotation after the pod's first day is one of
        // these.
        //
        // Three writes, two rotations, both under the floor. The middle token's
        // lifetime is deliberately unequal to the first's so the two recorded
        // headrooms land in disjoint ranges: that is what proves the second
        // observation is a fresh measurement against the token that was actually
        // retired, rather than the first value re-reported.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("headroom-two-rotations.json");
        write_token_file(&path, "rotation-token-1", SHORT_OUTGOING_LIFETIME_SECS).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, metrics, cancel) =
            build_provider_with_floor(path.clone(), Some(TEST_PROLONGATION_FLOOR));
        await_bearer(&provider, "rotation-token-1").await;

        // Rotation #1: token-1 retired with ~`SHORT_OUTGOING_LIFETIME_SECS` of
        // life left.
        write_token_file(&path, "rotation-token-2", SECOND_OUTGOING_LIFETIME_SECS).await;
        let first = await_headroom_change(&metrics, ServiceTokenProvider::ROLE_VALKEY, None).await;
        assert!(
            first > SHORT_OUTGOING_LIFETIME_SECS as f64 - HEADROOM_SLACK_SECS
                && first <= SHORT_OUTGOING_LIFETIME_SECS as f64,
            "the first rotation must be measured against token-1's expiry \
             (~{SHORT_OUTGOING_LIFETIME_SECS} s away); got {first}"
        );

        // Rotation #2, written only once #1 has been *observed* — otherwise the
        // poller could coalesce the two file writes into a single pickup and
        // this would silently become another one-rotation test.
        write_token_file(&path, "rotation-token-3", REPLACEMENT_LIFETIME_SECS).await;
        let second =
            await_headroom_change(&metrics, ServiceTokenProvider::ROLE_VALKEY, Some(first)).await;
        assert!(
            second > SECOND_OUTGOING_LIFETIME_SECS as f64 - HEADROOM_SLACK_SECS
                && second <= SECOND_OUTGOING_LIFETIME_SECS as f64,
            "the second rotation must be measured against token-2's expiry \
             (~{SECOND_OUTGOING_LIFETIME_SECS} s away), not re-report the first rotation's \
             {first} and not be derived from the replacement's {REPLACEMENT_LIFETIME_SECS} s; \
             got {second}"
        );

        // Both WARNs. Waited for rather than read straight off the buffer: the
        // WARN is emitted just after the gauge record it accompanies, so a bare
        // read here would race the second line and fail intermittently on the
        // count.
        poll_until(POLL_PICKUP_BUDGET, || {
            marker_count(&captured, HEADROOM_WARN_MARKER) >= 2
        })
        .await
        .unwrap_or_else(|| {
            panic!(
                "each under-floor rotation must WARN, so two rotations must produce two lines \
                 within {POLL_PICKUP_BUDGET:?}; saw {}",
                marker_count(&captured, HEADROOM_WARN_MARKER)
            )
        });
        assert!(
            poll_until(NO_MORE_OUTPUT_WINDOW, || marker_count(
                &captured,
                HEADROOM_WARN_MARKER
            ) > 2)
            .await
            .is_none(),
            "two rotations must produce exactly two WARNs; a third within \
             {NO_MORE_OUTPUT_WINDOW:?} means the line is keyed on the loop head rather than on \
             the rotation"
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert_eq!(
            output.matches(HEADROOM_WARN_MARKER).count(),
            2,
            "one WARN per observed rotation, both of them: {output}"
        );
        // …and each line carries its own measurement. A latched implementation
        // that somehow emitted twice would repeat the first headroom here.
        assert!(
            output.contains(&format!("observed_headroom_secs={}", first as i64)),
            "the first WARN must report the first rotation's headroom ({first}): {output}"
        );
        assert!(
            output.contains(&format!("observed_headroom_secs={}", second as i64)),
            "the second WARN must report the second rotation's own headroom ({second}), not \
             repeat the first: {output}"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn under_floor_rotation_on_a_non_valkey_role_names_a_generic_lever() {
        // The WARN's lever name is keyed on the role, because
        // `LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS` drives exactly one
        // instance's lead — and the constructor that takes a floor is `pub`, so
        // nothing structurally stops another instance from carrying one. Naming
        // the Valkey variable in *that* line would send an operator to a knob
        // that does not move the deployment they are looking at, which is worse
        // than naming no knob: they would change it, observe nothing, and
        // conclude the signal is noise.
        //
        // Production wires no such instance today, which is exactly why this
        // needs a test — the branch is unreachable from `main.rs` and would
        // otherwise be pinned by nothing at all.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("headroom-non-valkey-role.json");
        write_token_file(&path, "other-role-outgoing", SHORT_OUTGOING_LIFETIME_SECS).await;

        let (_dispatch_guard, captured) = capture_logs();
        let (provider, metrics, cancel) = build_provider_with_role_lead_and_floor(
            path.clone(),
            ServiceTokenProvider::ROLE_STORAGE_API,
            SUB_SECOND_SKEW,
            Some(TEST_PROLONGATION_FLOOR),
        );
        await_bearer(&provider, "other-role-outgoing").await;

        write_token_file(&path, "other-role-replacement", REPLACEMENT_LIFETIME_SECS).await;
        await_headroom_change(&metrics, ServiceTokenProvider::ROLE_STORAGE_API, None).await;

        poll_until(POLL_PICKUP_BUDGET, || {
            marker_count(&captured, HEADROOM_WARN_MARKER) > 0
        })
        .await
        .expect(
            "an under-floor rotation must WARN whatever the role, since the floor is what \
                 gates the line",
        );

        drop(_dispatch_guard);
        let output = captured_text(&captured);
        assert!(
            !output.contains("LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS"),
            "a non-Valkey instance's WARN must NOT name the Valkey-only env var — that variable \
             does not drive this instance's lead: {output}"
        );
        assert!(
            output.contains("this instance's refresh lead"),
            "…and must still name the lever generically, so the operator knows there are two \
             terms to raise rather than only the sidecar's: {output}"
        );

        cancel.cancel();
    }

    // ── Unrepresentable refresh leads (b.is1) ────────────────────────────────

    /// A refresh lead so large that `Utc::now() + lead` falls outside chrono's
    /// `DateTime<Utc>` range: the largest `chrono::Duration` there is.
    ///
    /// `Config::from_env` rejects this range
    /// (`VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS`), so it is unreachable through the
    /// env — but the constructors are `pub` and take a caller-chosen
    /// `chrono::Duration`, and the surviving addition site is written with
    /// `checked_add_signed` precisely so it does not depend on the config gate
    /// staying in sync. This is the value that exercises it.
    const UNREPRESENTABLE_LEAD: chrono::Duration = chrono::Duration::MAX;

    /// The provider's one `Utc::now() + lead` site — the polling task's
    /// stale-phase entry check — must resolve an unaddable lead as "not inside
    /// the window" rather than panicking.
    ///
    /// It is the site that needs a test most, and since b.mq8 the only one:
    /// the task is `tokio::spawn`ed with its `JoinHandle` dropped, so a panic
    /// there is *swallowed* — the poller and `force_refresh()` die silently
    /// while the pod coasts to credential death with no log line. Nothing else
    /// observes that. (The second site was inside the withholding accessor,
    /// deleted with it; its guard had a caller-visible failure mode and needs no
    /// successor test.)
    #[tokio::test]
    async fn unrepresentable_refresh_lead_leaves_the_poller_running() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("unrepresentable-lead.json");
        write_token_file(&path, "first-token", AMPLE_OUTGOING_LIFETIME_SECS).await;

        let (provider, metrics, cancel) = build_provider_with_role_lead_and_floor(
            path.clone(),
            ServiceTokenProvider::ROLE_VALKEY,
            UNREPRESENTABLE_LEAD,
            None,
        );
        // `await_cached`: the cache is what the loop head under test reads, and
        // a populated cache is the precondition for it reaching the entry check
        // at all.
        await_cached(&provider).await;

        // Liveness, twice. The rotation-headroom gauge is the loop head's only
        // externally visible product, so a gauge move is proof that a whole pass
        // completed — and *two* moves are proof that a pass completed past the
        // stale-phase entry check, which sits after the rotation block and is
        // the second `Utc::now() + lead` site. One move alone would leave a
        // panic between the two sites indistinguishable from a healthy pass.
        //
        // The three fixtures carry deliberately unequal lifetimes so each
        // rotation's headroom differs from the last — with equal lifetimes the
        // gauge would land on the same number twice and the second wait could
        // not tell a fresh observation from a frozen series.
        write_token_file(&path, "second-token", SECOND_OUTGOING_LIFETIME_SECS).await;
        let first = await_headroom_change(&metrics, ServiceTokenProvider::ROLE_VALKEY, None).await;
        write_token_file(&path, "third-token", REPLACEMENT_LIFETIME_SECS).await;
        await_headroom_change(&metrics, ServiceTokenProvider::ROLE_VALKEY, Some(first)).await;

        cancel.cancel();
    }

    // ── Cached token claims (b.cbe) ─────────────────────────────────────────
    //
    // `decode_token_claims` projects a FIXED, NAMED set of non-secret claims
    // out of a JWT-shaped access token, and `read_and_update` runs it once at
    // cache-write time so `cached_token_claims()` is a lock-read with no
    // parsing. The consumer is the Valkey `accessTokenFile` auth diagnostic in
    // `storage::client`, which prints what lands here — so the two properties
    // under test are (a) the projection is faithful across the claim shapes a
    // real issuer emits, and (b) a token that is NOT a decodable JWT degrades
    // to `None` **without** disturbing the credential path, since an opaque
    // token is a legitimate deployment and must still authenticate.
    //
    // Split deliberately: the shape matrix drives `decode_token_claims`
    // directly (pure, no runtime, no file I/O — one case per shape, which is
    // unaffordable through the polling task), while the four provider-level
    // tests below pin the parts only the cache can answer — that the decode
    // happens at cache-write time, that a failed decode still caches, and that
    // a rotation replaces the previous claims rather than latching them.

    /// Build a JWT-shaped string carrying `claims` as its payload.
    ///
    /// Same construction as `userinfo_jwt.rs::tests::make_jwt` (fixed
    /// `{"alg":"none"}` header, empty signature, `URL_SAFE_NO_PAD` throughout)
    /// because both decoders share `deserialize_optional_string` and the same
    /// three-segment split — a fixture that drifted from that one would be
    /// testing a different wire shape than production sees.
    fn make_jwt(claims: &serde_json::Value) -> String {
        let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims).unwrap());
        let signature = URL_SAFE_NO_PAD.encode(b"");
        format!("{header}.{payload}.{signature}")
    }

    /// A JWT-shaped token whose payload segment is `payload` verbatim — the
    /// escape hatch for the malformed-payload cases, where the point is that
    /// the segment is *not* valid base64url or *not* JSON.
    fn jwt_with_raw_payload(payload: &str) -> String {
        let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
        format!("{header}.{payload}.")
    }

    #[test]
    fn decode_token_claims_full_jwt_projects_every_named_claim() {
        let jwt = make_jwt(&serde_json::json!({
            "aud": "https://redis.azure.com",
            "oid": "11111111-2222-3333-4444-555555555555",
            "appid": "99999999-8888-7777-6666-555555555555",
            "tid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "iss": "https://login.example.com/tenant/v2.0",
            "exp": 1_700_000_000_i64,
            // An unreviewed extra claim must be dropped on the floor: the
            // projection is closed by design (never a claim bag), because
            // anything it keeps can reach a log line.
            "sub": "not-projected",
        }));

        let claims = decode_token_claims(&jwt).expect("a well-formed JWT must decode");

        assert_eq!(
            claims,
            TokenClaims {
                aud: vec!["https://redis.azure.com".to_string()],
                oid: Some("11111111-2222-3333-4444-555555555555".to_string()),
                client_id_claim: Some("99999999-8888-7777-6666-555555555555".to_string()),
                tid: Some("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".to_string()),
                iss: Some("https://login.example.com/tenant/v2.0".to_string()),
                exp: Some(1_700_000_000),
            }
        );
    }

    // (The single-string `aud` shape is not tested separately: the
    // full-projection test above already carries a string `aud` and asserts the
    // whole `TokenClaims`, which subsumes a one-element-vec assertion.)

    #[test]
    fn decode_token_claims_array_audience_collects_every_string_member() {
        // RFC 7519 §4.1.3 permits an array. Collapsing it (the `Option<String>`
        // modelling this type deliberately avoids) would report "no audience"
        // for an ordinary multi-audience token — hiding exactly the wrong-scope
        // failure the diagnostic exists to surface. The embedded number is
        // skipped rather than failing the whole decode.
        let jwt = make_jwt(&serde_json::json!({
            "aud": ["https://redis.azure.com", 42, "api://live-edit"],
        }));

        let claims = decode_token_claims(&jwt).expect("decodes");

        assert_eq!(
            claims.aud,
            vec![
                "https://redis.azure.com".to_string(),
                "api://live-edit".to_string()
            ]
        );
    }

    #[test]
    fn decode_token_claims_non_string_non_array_audience_yields_empty_vec() {
        let jwt = make_jwt(&serde_json::json!({ "aud": { "nested": "object" } }));

        let claims = decode_token_claims(&jwt).expect("decodes");

        assert!(
            claims.aud.is_empty(),
            "an audience of an illegal JSON shape must collapse to absent, not fail the decode; \
             got {:?}",
            claims.aud
        );
    }

    #[test]
    fn decode_token_claims_appid_present_wins_over_azp() {
        let jwt = make_jwt(&serde_json::json!({
            "appid": "the-appid",
            "azp": "the-azp",
        }));

        let claims = decode_token_claims(&jwt).expect("decodes");

        assert_eq!(claims.client_id_claim, Some("the-appid".to_string()));
    }

    #[test]
    fn decode_token_claims_azp_used_when_appid_absent() {
        let jwt = make_jwt(&serde_json::json!({ "azp": "the-azp" }));

        let claims = decode_token_claims(&jwt).expect("decodes");

        assert_eq!(claims.client_id_claim, Some("the-azp".to_string()));
    }

    #[test]
    fn decode_token_claims_no_identity_claims_leaves_oid_and_client_id_none() {
        let jwt = make_jwt(&serde_json::json!({ "iss": "https://issuer.example" }));

        let claims = decode_token_claims(&jwt).expect("decodes");

        assert_eq!(claims.oid, None);
        assert_eq!(claims.client_id_claim, None);
    }

    #[test]
    fn decode_token_claims_empty_string_claims_are_treated_as_absent() {
        // Load-bearing for the identity self-check downstream: an empty `oid`
        // that survived as `Some("")` would compare equal to an unset
        // configured username and report a healthy identity match.
        let jwt = make_jwt(&serde_json::json!({
            "aud": ["", "api://live-edit"],
            "oid": "",
            "appid": "",
            "azp": "",
            "tid": "",
            "iss": "",
        }));

        let claims = decode_token_claims(&jwt).expect("decodes");

        assert_eq!(claims.oid, None);
        assert_eq!(claims.client_id_claim, None);
        assert_eq!(claims.tid, None);
        assert_eq!(claims.iss, None);
        assert_eq!(claims.aud, vec!["api://live-edit".to_string()]);
    }

    #[test]
    fn decode_token_claims_wrong_shaped_claims_collapse_to_missing() {
        // Per-field tolerance: one claim of an unexpected JSON type must not
        // cost the operator the rest of the diagnostic.
        let jwt = make_jwt(&serde_json::json!({
            "oid": 12345,
            "appid": ["array"],
            "tid": true,
            "iss": { "nested": "object" },
            "exp": "1700000000",
        }));

        let claims = decode_token_claims(&jwt).expect("decodes");

        assert_eq!(claims, TokenClaims::default());
    }

    #[test]
    fn decode_token_claims_fractional_exp_truncates_toward_zero() {
        let jwt = make_jwt(&serde_json::json!({ "exp": 1_700_000_000.9_f64 }));

        let claims = decode_token_claims(&jwt).expect("decodes");

        assert_eq!(claims.exp, Some(1_700_000_000));
    }

    #[test]
    fn decode_token_claims_undecodable_shapes_return_none() {
        // Every input shape that is NOT a decodable JWT payload, in one table.
        // `None` is the contract for all of them — an opaque token is a
        // legitimate issuer-agnostic deployment, so this is a degrade, not an
        // error path. Folded from nine near-identical single-assert bodies; the
        // `why` column is what a red run reads.
        let three_segment = make_jwt(&serde_json::json!({ "oid": "x" }));
        let two_segments = three_segment
            .rsplit_once('.')
            .expect("fixture has three segments")
            .0
            .to_string();
        let scalar_payload =
            |json: &str| jwt_with_raw_payload(&URL_SAFE_NO_PAD.encode(json.as_bytes()));

        let cases: [(String, &str); 9] = [
            (
                "an-entirely-opaque-token".to_string(),
                "not a JWT at all — one segment, the shape an issuer-agnostic sidecar may \
                 legitimately mint",
            ),
            (
                two_segments,
                "two segments — a truncated or hand-assembled token is not a JWS",
            ),
            (
                format!("{three_segment}.extra"),
                "four segments — a JWE (five) or a mangled concatenation must not be decoded as \
                 if its second segment were a JWS payload",
            ),
            (
                jwt_with_raw_payload("not+valid/base64url=="),
                "`+` / `/` / `=` are the standard-base64 alphabet, rejected by the \
                 `URL_SAFE_NO_PAD` engine the production decoder uses",
            ),
            (
                jwt_with_raw_payload(&URL_SAFE_NO_PAD.encode(b"this is not json")),
                "payload decodes cleanly but is not JSON",
            ),
            (
                scalar_payload("\"just-a-string\""),
                "a JSON string payload is not a claim set",
            ),
            (
                scalar_payload("42"),
                "a JSON number payload is not a claim set",
            ),
            (
                scalar_payload("null"),
                "a JSON null payload is not a claim set",
            ),
            (
                // HISTORY — read before "fixing" this row. It used to assert the
                // OPPOSITE (`decode_token_claims_json_array_payload_maps_positionally`),
                // pinning an accepted serde artefact: serde derives a *positional*
                // struct deserializer as well as a field-named one, so a JSON array
                // payload was consumed field-by-field in `RawJwtClaims`' DECLARATION
                // ORDER (`aud`, `oid`, `appid`, `azp`, `tid`, `iss`, `exp`) and
                // `["aud-slot","oid-slot","appid-slot"]` produced a real-looking `oid`
                // — enough for the Valkey diagnostic to report a confident
                // `MatchesOid` verdict by pure coincidence. Code review rejected
                // that: RFC 7519 §7.2 makes the payload a JSON object, and an
                // authoritative misdiagnosis from a diagnostic whose whole job is to
                // be trusted is worse than no diagnostic. The `value.is_object()`
                // guard in `decode_token_claims` is what makes this `None`.
                //
                // So if this row reds, the FIX IS THE GUARD, not the test.
                jwt_with_raw_payload(
                    &URL_SAFE_NO_PAD.encode(b"[\"aud-slot\",\"oid-slot\",\"appid-slot\"]"),
                ),
                "a JSON array payload maps POSITIONALLY under serde's derived impl, so accepting \
                 it would mint a coincidental `oid` and a coincidental identity verdict",
            ),
        ];

        for (token, why) in cases {
            assert_eq!(
                decode_token_claims(&token),
                None,
                "must decode to None: {why}"
            );
        }
    }

    #[tokio::test]
    async fn cached_token_claims_empty_cache_returns_none() {
        // The precondition every other claims test needs to be non-vacuous:
        // an unpopulated cache and a non-decodable token both read `None`.
        let dir = tempdir().unwrap();
        let (provider, _metrics, cancel) = build_provider(dir.path().join("absent.json"));

        assert_eq!(provider.cached_token_claims(), None);

        cancel.cancel();
    }

    #[tokio::test]
    async fn cached_token_claims_projects_the_claims_of_the_cached_token() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("token.json");
        let jwt = make_jwt(&serde_json::json!({
            "aud": "https://redis.azure.com",
            "oid": "11111111-2222-3333-4444-555555555555",
            "appid": "99999999-8888-7777-6666-555555555555",
        }));
        write_token_file(&path, &jwt, 3600).await;

        let (provider, _metrics, cancel) = build_provider(path);
        await_bearer(&provider, &jwt).await;

        let claims = provider
            .cached_token_claims()
            .expect("a JWT-shaped cached token must expose its claims");
        assert_eq!(
            claims.oid,
            Some("11111111-2222-3333-4444-555555555555".to_string())
        );
        assert_eq!(
            claims.client_id_claim,
            Some("99999999-8888-7777-6666-555555555555".to_string())
        );
        assert_eq!(claims.aud, vec!["https://redis.azure.com".to_string()]);

        cancel.cancel();
    }

    #[tokio::test]
    async fn cached_token_claims_opaque_token_returns_none_but_the_token_still_authenticates() {
        // The load-bearing negative: a decode failure is a *diagnostic*
        // degradation, never a credential one. An implementation that made the
        // cache write conditional on a successful decode would break every
        // opaque-token deployment, and `cached_token_claims() == None` alone
        // cannot tell that apart from a healthy opaque token — hence the
        // bearer assertion in the same test.
        let dir = tempdir().unwrap();
        let path = dir.path().join("token.json");
        const OPAQUE: &str = "an-entirely-opaque-sidecar-token";
        write_token_file(&path, OPAQUE, 3600).await;

        let (provider, _metrics, cancel) = build_provider(path);
        await_bearer(&provider, OPAQUE).await;

        assert_eq!(
            provider.cached_token_claims(),
            None,
            "an opaque token carries no readable claims"
        );
        assert_eq!(
            provider.cached_bearer_if_unexpired().as_deref(),
            Some(OPAQUE),
            "a token that cannot be decoded must still be served as the credential"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn cached_token_claims_rotation_replaces_the_previous_claims() {
        // Claims are cached alongside the token, so a rotation must move them
        // with it. A latched projection would keep reporting the retired
        // identity — which is precisely the case the diagnostic is consulted
        // in, since a rotation into a differently-bound identity is one of the
        // failure modes it exists to name.
        let dir = tempdir().unwrap();
        let path = dir.path().join("token.json");
        let first = make_jwt(&serde_json::json!({ "oid": "oid-before-rotation" }));
        write_token_file(&path, &first, 3600).await;

        let (provider, _metrics, cancel) = build_provider(path.clone());
        await_bearer(&provider, &first).await;
        assert_eq!(
            provider.cached_token_claims().and_then(|c| c.oid),
            Some("oid-before-rotation".to_string())
        );

        let second = make_jwt(&serde_json::json!({ "oid": "oid-after-rotation" }));
        write_token_file(&path, &second, 3600).await;
        await_bearer(&provider, &second).await;

        assert_eq!(
            provider.cached_token_claims().and_then(|c| c.oid),
            Some("oid-after-rotation".to_string()),
            "the projection must follow the rotation rather than latching the first token's claims"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn cached_token_claims_rotation_from_jwt_to_opaque_clears_the_claims() {
        // The other direction of the same invariant, and the one an
        // "only overwrite when the decode succeeds" implementation passes the
        // test above while failing: stale claims outliving the token they were
        // read from would make the diagnostic accuse an identity that is no
        // longer in play.
        let dir = tempdir().unwrap();
        let path = dir.path().join("token.json");
        let jwt = make_jwt(&serde_json::json!({ "oid": "oid-before-rotation" }));
        write_token_file(&path, &jwt, 3600).await;

        let (provider, _metrics, cancel) = build_provider(path.clone());
        await_bearer(&provider, &jwt).await;
        assert!(provider.cached_token_claims().is_some());

        const OPAQUE: &str = "an-entirely-opaque-replacement-token";
        write_token_file(&path, OPAQUE, 3600).await;
        await_bearer(&provider, OPAQUE).await;

        assert_eq!(
            provider.cached_token_claims(),
            None,
            "claims from the retired token must not survive a rotation to an opaque one"
        );

        cancel.cancel();
    }

    #[tokio::test]
    async fn cached_token_claims_never_leaks_token_bytes_into_the_logs() {
        // SR-6.2 extended to the b.cbe surface. The whole token is the canary;
        // the *claims* are expected to be readable (they are non-secret and the
        // diagnostic prints them), so the canary is deliberately the base64
        // wire form, no 8-byte window of which may reach a log line.
        let dir = tempdir().unwrap();
        let path = dir.path().join("token.json");
        let jwt = make_jwt(&serde_json::json!({
            "oid": "11111111-2222-3333-4444-555555555555",
            "appid": "99999999-8888-7777-6666-555555555555",
        }));

        let (guard, captured) = capture_logs();
        write_token_file(&path, &jwt, 3600).await;
        let (provider, _metrics, cancel) = build_provider(path);
        await_bearer(&provider, &jwt).await;
        let _ = provider.cached_token_claims();
        cancel.cancel();
        drop(guard);

        assert_no_token_bytes(&captured_text(&captured), &jwt);
    }
}
