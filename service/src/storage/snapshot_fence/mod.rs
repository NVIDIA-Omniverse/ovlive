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

//! The per-object snapshot fence's storage layer (Spec SDD SR-5.1–SR-5.4,
//! SR-5.7, SR-11.4): the six fence and intent scripts, their typed Rust
//! wrappers and the two guards, plus the cost model ([`cost`]) every fence
//! limit is computed from.
//!
//! # Who calls what
//!
//! Above the atomic-read ceiling a snapshot is proven by a **fence**
//! (`le:{oid}:obj:snapfence`): a fenced reader takes a lease
//! ([`acquire_fence`]), reads, and proves its read in the same script that
//! gives the lease up ([`FenceLease::release`]). Under the writer protocol of
//! Spec SDD SR-5.4, writers wait while any lease is live, and WATCH-path
//! writers classify their attempt with a read-only [`probe_fence_on`] and,
//! when the attempt reads a snapshot of such a layer, protect it from the
//! next epoch with an **intent** (`le:{oid}:obj:snapintent`,
//! [`register_intent`]); the cached structural route registers none
//! (structural-edits SR-8.2). Every caller goes through these wrappers;
//! nothing issues the scripts directly. The one exception to "only through a
//! wrapper" is the commit script, `update_layer_fast.lua`, which in both
//! modes reads the fence key itself (its `KEYS[UPDATE_SCRIPT_KEY_FENCE]`, at
//! step 1d) inside its own atomic unit. What the writer protocol keeps
//! across one `update_layer` call — its size class, its intent's renewal, the
//! cap on its fence waits — lives in [`writer`]. The reader side — acquire,
//! the fan-out under the lease with its renewal timer, release and retry —
//! is [`read`].
//!
//! # Every limit from one model
//!
//! Each wrapper passes the scripts the coefficient block of
//! [`coefficients`]`(client)` — the recorded coefficients at the client's
//! configured atomic-read item ceiling — and the scripts compute every
//! per-layer limit from the size the fence key stores (the caller's estimate
//! only when it stores none). Callers compute the limits they need themselves
//! (renewal periods, wait caps) from the same [`coefficients`] and the size a
//! verdict carries ([`FenceLease::size`], [`AcquireVerdict::RetryAfter`]), so
//! the two sides agree. No timing constant is chosen anywhere else (SR-5.7).
//!
//! # Pool
//!
//! Every wrapper runs on the multiplexed **data pool** (`ValkeyClient::pool`),
//! never the snapshot lane (Spec SDD SR-10.1): these are small scripts, and a
//! renewal must not queue on a connection behind MiB-sized snapshot replies.
//! The one exception is [`probe_fence_on`], which takes the connection to run
//! on, because the WATCH path runs it on its `WATCH`ed connection beside the
//! header read (SR-5.4; `EVALSHA` does not disturb a `WATCH`).
//! [`probe_fence`] is the same probe on the data pool.
//!
//! # Errors
//!
//! A failed script call is classified like the other commit-path Lua calls,
//! through [`map_transient_fred_err`]: transient I/O becomes
//! `ServiceError::Aborted` (retried by a surrounding retry loop, and
//! `UNAVAILABLE` if it escapes a handler), a store write refusal of a fence
//! script's write (`-OOM`, `-MISCONF`, …) a plain `ServiceError::Unavailable`
//! that no loop retries, anything else `ServiceError::Internal`.
//! A command timeout is counted once, by that mapper, on the data pool (or on
//! the origin [`probe_fence_on`] is handed). A reply in a shape this binary's
//! scripts never produce is a protocol error, mapped the same way, so it is
//! `Internal`.
//!
//! # Guards
//!
//! [`FenceLease`] and [`WriterIntent`] own what their scripts wrote. Dropped
//! without an explicit [`FenceLease::release`] / [`WriterIntent::clear`] — a
//! cancelled future, a panic, an early `?` — they spawn a best-effort release
//! (the release script with the size the lease was granted under) or `HDEL`
//! on the current tokio runtime. Best effort is enough: the lease and the
//! intent each lapse on their own deadline, which writers and readers judge
//! against the store's `TIME`, and the keys carry a `PEXPIRE`. A drop outside
//! any runtime spawns nothing and leaves the same backstops.
//!
//! The spawned cleanup is tracked on the client
//! (`ValkeyClient::spawn_fence_guard_cleanup`), and `ValkeyClient::quit`
//! waits for every one, bounded, before it closes the pools. So a guard that
//! graceful shutdown drops — a catch-up flight's read when its last waiter
//! ends on the service shutdown token, a checkpoint's fenced read when the
//! checkpoint loop is cancelled — reaches the store, and other replicas'
//! writers of that layer are not left paused until the lease lapses.
//!
//! # Measurement event
//!
//! Every lease release — [`FenceLease::release`] and the guard's spawned
//! release alike — ends with one DEBUG event, `snapshot fence lease`, on the
//! target [`LEASE_TRACE_TARGET`] (`live_edit::snapshot_fence_lease`), written
//! after the release script answers, on success and on error. It is
//! measurement-only: the validation tooling unions these events across
//! replicas to judge the writers' share of wall-clock time per epoch–gap
//! cycle (Spec SDD SR-5.5). Its fields:
//!
//! - `object_id`; `kind` (`started_epoch` / `joined_epoch`); `cut`;
//! - `outcome`: `verified`, `broken`, `not_found`, or `error` when the call
//!   failed or its reply was malformed;
//! - `via`: `release` (an explicit [`FenceLease::release`]) or `drop` (the
//!   guard's best-effort release). A failed explicit release logs `error`,
//!   and the guard's re-release then logs a second event with the same
//!   acquire pair;
//! - `acquire_sent_unix_us` / `acquire_replied_unix_us` and
//!   `release_sent_unix_us` / `release_replied_unix_us`: the replica's wall
//!   clock (`SystemTime::now()`, µs since the Unix epoch) read immediately
//!   before and after the acquire and release `EVALSHA`s, so each pair
//!   brackets the store-side run of its script.
//!
//! These times are the replica's wall clock and never inform a fence
//! decision: every fence decision uses the store's `TIME` (SR-5.1). The event
//! never carries the lease token, the lease's only credential. No event is
//! written for a grant that was never released through [`release_lease`] (a
//! drop outside any runtime, a killed process). The clock reads run on every
//! lease; the event's fields are evaluated only when a filter enables the
//! target. The target sits outside `live_edit::storage::`, no other target
//! starts with it, and the only target it starts with is the crate root's own
//! `live_edit`. tracing-subscriber matches a directive's target by plain
//! string prefix, so no other module's directive (`live_edit::storage=debug`,
//! for instance) enables it. `live_edit::snapshot_fence_lease=debug` does, as
//! do `live_edit=debug` and a bare `debug`.

pub mod cost;
pub mod read;
pub mod writer;

use std::borrow::Cow;
use std::sync::LazyLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fred::interfaces::HashesInterface;
use fred::types::scripts::Script;
use fred::types::Value;

use crate::error::ServiceError;
use crate::storage::client::{
    map_transient_fred_err, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA,
};
use crate::storage::keys::KeySchema;

pub use cost::{ms_to_duration, FenceCoefficients, FenceTiming, LayerSize, FENCE_READ_ATTEMPTS};

/// The `tracing` target of the measurement-only `snapshot fence lease` event,
/// at DEBUG: one event per lease release, written by [`release_lease`] after
/// the release script answers (module docs, # Measurement event). The target
/// sits outside `live_edit::storage::`, no other target starts with it, and
/// the only target it starts with is the crate root's own `live_edit`.
/// tracing-subscriber matches a directive's target by plain string prefix, so
/// no other module's directive (`live_edit::storage=debug`, for instance)
/// enables it. `live_edit::snapshot_fence_lease=debug` does, as do
/// `live_edit=debug` and a bare `debug`. Its times are the replica's wall
/// clock and never inform a fence decision.
pub const LEASE_TRACE_TARGET: &str = "live_edit::snapshot_fence_lease";

/// The replica's wall clock in µs since the Unix epoch, for the measurement
/// event's brackets only (never a fence input). A pre-epoch clock — not a
/// state any sane host is in — folds to 0 rather than panicking.
fn unix_now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// What the measurement event carries from the grant: copied out of the lease
/// for [`release_lease`], so the guard's spawned release can log it after the
/// lease is gone. Holds no token.
#[derive(Debug, Clone, Copy)]
struct LeaseTrace {
    object_id: u64,
    kind: GrantKind,
    cut: u64,
    acquire_sent_unix_us: u64,
    acquire_replied_unix_us: u64,
}

/// Which path released a lease, for the measurement event's `via` field.
#[derive(Debug, Clone, Copy)]
enum ReleaseVia {
    /// An explicit [`FenceLease::release`].
    Release,
    /// The guard's best-effort release on drop.
    Drop,
}

impl ReleaseVia {
    fn as_str(self) -> &'static str {
        match self {
            ReleaseVia::Release => "release",
            ReleaseVia::Drop => "drop",
        }
    }
}

/// `snapshot_fence_acquire.lua`; sent only by [`acquire_fence`].
static FENCE_ACQUIRE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("../scripts/snapshot_fence_acquire.lua")));

/// `snapshot_fence_renew.lua`; sent only by [`FenceLease::renew`].
static FENCE_RENEW_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("../scripts/snapshot_fence_renew.lua")));

/// `snapshot_fence_release.lua`; sent only by [`release_lease`].
static FENCE_RELEASE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("../scripts/snapshot_fence_release.lua")));

/// `snapshot_fence_probe.lua`; sent only by [`probe_fence_on`].
static FENCE_PROBE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("../scripts/snapshot_fence_probe.lua")));

/// `snapshot_intent_register.lua`; sent only by [`register_intent`].
static INTENT_REGISTER_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("../scripts/snapshot_intent_register.lua")));

/// `snapshot_intent_renew.lua`; sent only by [`WriterIntent::renew`].
static INTENT_RENEW_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("../scripts/snapshot_intent_renew.lua")));

/// The coefficients every fence script and every caller's limit is computed
/// from: the recorded set at `client`'s configured atomic-read item ceiling
/// (`LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS`).
pub fn coefficients(client: &ValkeyClient) -> FenceCoefficients {
    FenceCoefficients::recorded(client.snapshot_atomic_read_max_items)
}

/// A random 128-bit token in lowercase hex: one per fenced read, one per
/// writer intent. Never logged above DEBUG.
fn mint_token() -> String {
    format!("{:032x}", rand::random::<u128>())
}

/// Why [`acquire_fence`] asked the reader to wait (Spec SDD SR-5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryReason {
    /// A live epoch the reader arrived outside the join window of.
    Epoch,
    /// The writers' gap after the last epoch has not ended.
    Gap,
    /// A WATCH-path writer's intent is live, inside the window readers honour
    /// intents for.
    Intent,
}

impl RetryReason {
    /// The script's word for the reason: `epoch`, `gap` or `intent`.
    pub fn as_str(self) -> &'static str {
        match self {
            RetryReason::Epoch => "epoch",
            RetryReason::Gap => "gap",
            RetryReason::Intent => "intent",
        }
    }
}

/// How [`acquire_fence`] granted the fence (Spec SDD SR-5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrantKind {
    /// No reader was live and the gap and intents allowed it: a new epoch.
    StartedEpoch,
    /// A live epoch inside its join window: the reader joined it.
    JoinedEpoch,
}

impl GrantKind {
    /// The script's word for the kind: `started_epoch` or `joined_epoch`.
    pub fn as_str(self) -> &'static str {
        match self {
            GrantKind::StartedEpoch => "started_epoch",
            GrantKind::JoinedEpoch => "joined_epoch",
        }
    }
}

/// What [`acquire_fence`] answered. Only [`Self::Granted`] wrote anything.
#[derive(Debug)]
pub enum AcquireVerdict {
    /// The object header is absent, or present without a numeric `seq`.
    NotFound,
    /// Wait `hint`, then call again. `size` is the layer size the script used,
    /// the fence key's stored size when it had one.
    RetryAfter {
        hint: Duration,
        reason: RetryReason,
        size: LayerSize,
    },
    /// The reader holds a lease; the snapshot it reads under it is proven at
    /// [`FenceLease::cut`] if its release answers [`ReleaseVerdict::Verified`].
    /// Boxed: the guard carries a `ValkeyClient` handle, far larger than the
    /// other variants.
    Granted(Box<FenceLease>),
}

/// What [`FenceLease::renew`] answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenewVerdict {
    /// The lease was extended.
    Renewed,
    /// The lease is gone or had already expired; nothing was written. The read
    /// must stop: its release will answer [`ReleaseVerdict::Broken`].
    Lost,
}

/// What [`FenceLease::release`] answered (Spec SDD SR-5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseVerdict {
    /// The lease was still live and the header still at the cut: the read is
    /// proven at the cut.
    Verified,
    /// The lease was lost, or a non-commit mutator (restore, cleanup) moved or
    /// removed what the read covered. Retry the whole fenced read, up to
    /// [`FENCE_READ_ATTEMPTS`].
    Broken,
    /// The object header is gone. Nothing was written.
    NotFound,
}

/// What [`probe_fence_on`] saw. The probe writes nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum FenceProbe {
    /// No fence key: the layer has not been fenced within its last
    /// `key_linger(L)`.
    Absent,
    /// The fence key exists.
    Present {
        /// `Some(latest live deadline − now)` while a reader holds the fence.
        live_retry_after: Option<Duration>,
        /// The layer size the key stores, if any.
        stored_size: Option<LayerSize>,
        /// The bytes per item the key stores, if any.
        stored_bpi: Option<f64>,
        /// The key's epoch marker, which every fence write after a probe that
        /// found no live lease changes (see [`FenceMarker`]).
        marker: FenceMarker,
    },
    /// The fence or the writer-intent key holds a type other than a hash.
    /// `detail` is the script's `WRONGTYPE_KEY: <key> holds a <type>,
    /// expected hash` refusal — the update script's step-1d reply (either
    /// mode), byte for byte
    /// — for the caller to refuse its commit with (SDD §7.5 parity).
    WrongType { detail: String },
}

impl FenceProbe {
    /// Whether the fence key changed between `earlier` — an attempt's own
    /// probe, which found no live lease — and this later probe of the same
    /// key: it appeared or disappeared, or its [`FenceMarker`] differs.
    ///
    /// The slow path's conflict classifier reads a change as an epoch that
    /// began, or began and ended, after the attempt's `WATCH`: the fence, not
    /// a peer writer, failed the `EXEC` (Spec SDD SR-5.4). A
    /// [`FenceProbe::WrongType`] on either side is never a change: the
    /// classifier refuses it before asking, and a wrong-typed key is no
    /// evidence of an epoch.
    pub fn changed_since(&self, earlier: &FenceProbe) -> bool {
        match (earlier, self) {
            (FenceProbe::Absent, FenceProbe::Absent) => false,
            (FenceProbe::Absent, FenceProbe::Present { .. })
            | (FenceProbe::Present { .. }, FenceProbe::Absent) => true,
            (
                FenceProbe::Present { marker: before, .. },
                FenceProbe::Present { marker: after, .. },
            ) => before != after,
            (FenceProbe::WrongType { .. }, _) | (_, FenceProbe::WrongType { .. }) => false,
        }
    }
}

/// The epoch marker [`probe_fence_on`] reads from a present fence key
/// (`snapshot_fence_probe.lua`'s last three reply elements): the fields the
/// acquire and release scripts already keep, which together change on every
/// fence write after a probe that found no live lease.
///
/// - A grant then starts an epoch, so `epoch_start` takes a new value.
/// - The release that ends an epoch clears `epoch_start` and sets a later
///   `gap_until`.
/// - Any other release removes its own `r:*` field, so `readers` drops.
/// - Joins and renewals happen only inside an epoch that began after that
///   probe, so `epoch_start` has already moved; expiry or `UNLINK` removes
///   the key ([`FenceProbe::Absent`]).
///
/// No script writes anything for it, so it adds no stored state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FenceMarker {
    /// The stored `epoch_start` (ms since the Unix epoch, on the store's
    /// `TIME`), or `None` when no epoch is in progress.
    pub epoch_start: Option<u64>,
    /// The stored `gap_until` (ms, store `TIME`), or `None` when absent.
    pub gap_until: Option<u64>,
    /// The number of `r:*` reader fields, live or expired.
    pub readers: u64,
}

/// What [`register_intent`] answered (Spec SDD SR-5.4). Only
/// [`Self::Registered`] wrote anything.
#[derive(Debug)]
pub enum IntentVerdict {
    /// The object header is absent (a deleted, retired or wiped object).
    NotFound,
    /// The fence or the writer-intent key holds a type other than a hash;
    /// `detail` is the `WRONGTYPE_KEY` refusal, as for
    /// [`FenceProbe::WrongType`].
    WrongType { detail: String },
    /// A reader holds the fence; nothing was written. The latest live lease
    /// ends in `retry_after`.
    Fenced { retry_after: Duration },
    /// The intent is registered and held by the guard. Boxed for the same
    /// reason as [`AcquireVerdict::Granted`].
    Registered(Box<WriterIntent>),
}

/// The tag both [`probe_fence_on`] and [`register_intent`] recognise on a
/// wrong-typed fence or intent key — the commit paths' keyspace-type refusal.
const WRONGTYPE_KEY_TAG: &str = "WRONGTYPE_KEY";

/// What [`WriterIntent::renew`] answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentRenewVerdict {
    /// The intent was extended.
    Renewed,
    /// The intent is gone or had already lapsed; nothing was written.
    Lost,
}

/// Ask for a fenced reader's lease on `object_id`'s fence
/// (`snapshot_fence_acquire.lua`, Spec SDD SR-5.2).
///
/// `estimate` is the caller's layer size (SR-11.1, [`LayerSize::from_too_large`]);
/// the script uses the fence key's stored size instead whenever it has one,
/// and writes the estimate only on the grant that creates a fresh key. Runs on
/// the data pool. Writes nothing unless it grants.
pub async fn acquire_fence(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    estimate: LayerSize,
) -> Result<AcquireVerdict, ServiceError> {
    const OP: &str = "EVALSHA snapshot_fence_acquire";
    let oid = object_id.to_string();
    let coefficients = coefficients(client);
    let token = mint_token();
    let fence_key = keys.obj_snapshot_fence(&oid);
    let header_key = keys.obj_header(&oid);
    let script_keys = vec![
        fence_key.clone(),
        keys.obj_snapshot_intent(&oid),
        header_key.clone(),
    ];
    let mut script_argv = Vec::with_capacity(3 + cost::FENCE_COEFFICIENT_ARGC);
    script_argv.push(token.clone());
    script_argv.push(estimate.items.to_string());
    script_argv.push(estimate.bytes.to_string());
    script_argv.extend(coefficients.script_argv());

    // The measurement event's acquire bracket (module docs, # Measurement
    // event): read on every call, kept only on a grant.
    let acquire_sent_unix_us = unix_now_us();
    let reply = FENCE_ACQUIRE_SCRIPT
        .evalsha_with_reload::<Value, _, _>(client.pool.next(), script_keys, script_argv)
        .await;
    let acquire_replied_unix_us = unix_now_us();
    let reply: Value = reply.map_err(|e| data_pool_err(client, OP, e))?;
    let reply = reply.into_array();
    let verdict = match verdict_word(&reply).as_deref() {
        Some("NOT_FOUND") if reply.len() == 1 => Some(AcquireVerdict::NotFound),
        Some("RETRY_AFTER") if reply.len() == 5 => {
            let reason = match reply[2].as_str().as_deref() {
                Some("epoch") => Some(RetryReason::Epoch),
                Some("gap") => Some(RetryReason::Gap),
                Some("intent") => Some(RetryReason::Intent),
                _ => None,
            };
            match (
                reply_u64(&reply[1]),
                reason,
                reply_size(&reply[3], &reply[4]),
            ) {
                (Some(hint_ms), Some(reason), Some(size)) => Some(AcquireVerdict::RetryAfter {
                    hint: Duration::from_millis(hint_ms),
                    reason,
                    size,
                }),
                _ => None,
            }
        }
        Some("OK") if reply.len() == 5 => {
            let kind = match reply[2].as_str().as_deref() {
                Some("started_epoch") => Some(GrantKind::StartedEpoch),
                Some("joined_epoch") => Some(GrantKind::JoinedEpoch),
                _ => None,
            };
            let cut_raw = reply[1].as_string();
            let cut = cut_raw.as_deref().and_then(|s| s.parse::<u64>().ok());
            match (cut_raw, cut, kind, reply_size(&reply[3], &reply[4])) {
                (Some(cut_raw), Some(cut), Some(kind), Some(size)) => {
                    Some(AcquireVerdict::Granted(Box::new(FenceLease {
                        client: client.clone(),
                        fence_key,
                        header_key,
                        token,
                        object_id,
                        cut,
                        cut_raw,
                        kind,
                        size,
                        coefficients,
                        acquire_sent_unix_us,
                        acquire_replied_unix_us,
                        armed: true,
                    })))
                }
                _ => None,
            }
        }
        _ => None,
    };
    verdict.ok_or_else(|| unexpected_reply(client, OP, &reply))
}

/// A fenced reader's lease (Spec SDD SR-5.3), returned by a granted
/// [`acquire_fence`].
///
/// Hold it for the whole read and renew it every [`FenceTiming::lease_ms`]/4
/// ([`Self::renew_interval`]) from a concurrent timer, independently of the
/// read's windows. End it with [`Self::release`], which proves the read.
/// Dropping it unreleased spawns a best-effort release (module docs).
pub struct FenceLease {
    client: ValkeyClient,
    fence_key: String,
    header_key: String,
    token: String,
    /// The fenced object, for the measurement event (module docs,
    /// # Measurement event).
    object_id: u64,
    /// The cut, parsed.
    cut: u64,
    /// The cut exactly as the header stored it; release compares it byte for
    /// byte, so it is handed back verbatim rather than re-rendered.
    cut_raw: String,
    kind: GrantKind,
    /// The layer size the grant computed its limits from.
    size: LayerSize,
    coefficients: FenceCoefficients,
    /// The replica's wall clock (µs since the Unix epoch) immediately before
    /// the acquire `EVALSHA` was sent. Measurement only: carried into the
    /// release's event, never a fence input.
    acquire_sent_unix_us: u64,
    /// The replica's wall clock (µs since the Unix epoch) immediately after
    /// the acquire `EVALSHA` replied. Measurement only, as above.
    acquire_replied_unix_us: u64,
    /// Whether dropping the guard still owes a release.
    armed: bool,
}

// By hand: the token is the lease's only credential (whoever holds it can
// release the lease), so it stays out of `{:?}` output.
impl std::fmt::Debug for FenceLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FenceLease")
            .field("fence_key", &self.fence_key)
            .field("object_id", &self.object_id)
            .field("cut", &self.cut)
            .field("kind", &self.kind)
            .field("size", &self.size)
            .field("armed", &self.armed)
            .finish_non_exhaustive()
    }
}

impl FenceLease {
    /// The header `seq` at acquisition: the cut the read is proven at.
    pub fn cut(&self) -> u64 {
        self.cut
    }

    /// Whether the grant started an epoch or joined one.
    pub fn kind(&self) -> GrantKind {
        self.kind
    }

    /// The layer size the grant computed its limits from (the fence key's
    /// stored size when it had one).
    pub fn size(&self) -> LayerSize {
        self.size
    }

    /// Every limit for [`Self::size`], from the same coefficients the scripts
    /// were given.
    pub fn timing(&self) -> FenceTiming {
        self.coefficients.timing(self.size)
    }

    /// How often to renew: `lease(L) / 4`.
    pub fn renew_interval(&self) -> Duration {
        ms_to_duration(self.timing().renew_interval_ms())
    }

    /// Extend the lease if it is still held (`snapshot_fence_renew.lua`).
    /// [`RenewVerdict::Lost`] means stop the read. Runs on the data pool.
    pub async fn renew(&self) -> Result<RenewVerdict, ServiceError> {
        const OP: &str = "EVALSHA snapshot_fence_renew";
        let mut script_argv = Vec::with_capacity(3 + cost::FENCE_COEFFICIENT_ARGC);
        script_argv.push(self.token.clone());
        script_argv.push(self.size.items.to_string());
        script_argv.push(self.size.bytes.to_string());
        script_argv.extend(self.coefficients.script_argv());
        let reply: Value = FENCE_RENEW_SCRIPT
            .evalsha_with_reload::<Value, _, _>(
                self.client.pool.next(),
                vec![self.fence_key.clone()],
                script_argv,
            )
            .await
            .map_err(|e| data_pool_err(&self.client, OP, e))?;
        let reply = reply.into_array();
        let verdict = match verdict_word(&reply).as_deref() {
            Some("RENEWED") if reply.len() == 2 && reply_u64(&reply[1]).is_some() => {
                Some(RenewVerdict::Renewed)
            }
            Some("LOST") if reply.len() == 1 => Some(RenewVerdict::Lost),
            _ => None,
        };
        verdict.ok_or_else(|| unexpected_reply(&self.client, OP, &reply))
    }

    /// Prove the read and give the lease up, atomically
    /// (`snapshot_fence_release.lua`).
    ///
    /// `measured` is the layer's size as the read found it; the last reader of
    /// an epoch records it on the fence key for the next epoch. A read that did
    /// not complete passes `None`, and the size the lease was granted under is
    /// recorded instead. On [`ReleaseVerdict::Broken`] the read is not proven;
    /// on [`ReleaseVerdict::NotFound`] the object is gone. Either way the lease
    /// is given up, and neither writes anything on an object that cleanup or a
    /// restore wipe removed.
    ///
    /// If the call itself fails, the guard stays armed and its drop at the end
    /// of this call spawns one more best-effort release.
    pub async fn release(
        mut self,
        measured: Option<LayerSize>,
    ) -> Result<ReleaseVerdict, ServiceError> {
        let verdict = release_lease(
            &self.client,
            &self.fence_key,
            &self.header_key,
            &self.token,
            &self.cut_raw,
            measured.unwrap_or(self.size),
            &self.coefficients,
            self.trace(),
            ReleaseVia::Release,
        )
        .await?;
        self.armed = false;
        Ok(verdict)
    }

    /// The grant's part of the measurement event.
    fn trace(&self) -> LeaseTrace {
        LeaseTrace {
            object_id: self.object_id,
            kind: self.kind,
            cut: self.cut,
            acquire_sent_unix_us: self.acquire_sent_unix_us,
            acquire_replied_unix_us: self.acquire_replied_unix_us,
        }
    }
}

impl Drop for FenceLease {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // No runtime (a drop during late shutdown): spawn nothing. The lease
        // lapses on its own deadline, and the key on its PEXPIRE.
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let client = self.client.clone();
        let fence_key = std::mem::take(&mut self.fence_key);
        let header_key = std::mem::take(&mut self.header_key);
        let token = std::mem::take(&mut self.token);
        let cut_raw = std::mem::take(&mut self.cut_raw);
        let size = self.size;
        let coefficients = self.coefficients;
        let trace = self.trace();
        // Tracked, so `ValkeyClient::quit` waits for it before the pools close
        // (module docs, # Guards).
        let release = async move {
            let outcome = release_lease(
                &client,
                &fence_key,
                &header_key,
                &token,
                &cut_raw,
                size,
                &coefficients,
                trace,
                ReleaseVia::Drop,
            )
            .await;
            if let Err(error) = outcome {
                tracing::warn!(
                    fence_key = %fence_key,
                    error = %error,
                    "best-effort release of a dropped snapshot-fence lease failed; the lease \
                     lapses at its deadline, so writers of this layer may wait up to one lease \
                     longer"
                );
            }
        };
        self.client.spawn_fence_guard_cleanup(release, &runtime);
    }
}

/// The one release call site, shared by [`FenceLease::release`] and the
/// guard's drop.
///
/// It ends by writing the measurement-only `snapshot fence lease` DEBUG event
/// on [`LEASE_TRACE_TARGET`] (module docs, # Measurement event), after the
/// release `EVALSHA` returns, on success and on error: `trace` carries the
/// grant's object, kind, cut and acquire bracket, `via` the path that
/// released, and the release bracket is read around the `EVALSHA` here. The
/// brackets are the replica's wall clock and never inform a fence decision;
/// the event never carries `token`.
// Each argument is a distinct input: the release script's keys and argv, the
// measurement event's grant fields and its path. Its two callers take them
// apart from one lease (or the drop's moved-out copies), so a struct would
// only rename them at those two sites.
#[allow(clippy::too_many_arguments)]
async fn release_lease(
    client: &ValkeyClient,
    fence_key: &str,
    header_key: &str,
    token: &str,
    cut_raw: &str,
    measured: LayerSize,
    coefficients: &FenceCoefficients,
    trace: LeaseTrace,
    via: ReleaseVia,
) -> Result<ReleaseVerdict, ServiceError> {
    const OP: &str = "EVALSHA snapshot_fence_release";
    let mut script_argv = Vec::with_capacity(4 + cost::FENCE_COEFFICIENT_ARGC);
    script_argv.push(token.to_owned());
    script_argv.push(cut_raw.to_owned());
    script_argv.push(measured.items.to_string());
    script_argv.push(measured.bytes.to_string());
    script_argv.extend(coefficients.script_argv());
    let release_sent_unix_us = unix_now_us();
    let reply = FENCE_RELEASE_SCRIPT
        .evalsha_with_reload::<Value, _, _>(
            client.pool.next(),
            vec![fence_key.to_owned(), header_key.to_owned()],
            script_argv,
        )
        .await;
    let release_replied_unix_us = unix_now_us();
    let verdict = match reply {
        Err(e) => Err(data_pool_err(client, OP, e)),
        Ok(reply) => {
            let reply = reply.into_array();
            let verdict = match verdict_word(&reply).as_deref() {
                Some("VERIFIED") if reply.len() == 1 => Some(ReleaseVerdict::Verified),
                Some("BROKEN") if reply.len() == 1 => Some(ReleaseVerdict::Broken),
                Some("NOT_FOUND") if reply.len() == 1 => Some(ReleaseVerdict::NotFound),
                _ => None,
            };
            verdict.ok_or_else(|| unexpected_reply(client, OP, &reply))
        }
    };
    let outcome = match &verdict {
        Ok(ReleaseVerdict::Verified) => "verified",
        Ok(ReleaseVerdict::Broken) => "broken",
        Ok(ReleaseVerdict::NotFound) => "not_found",
        Err(_) => "error",
    };
    tracing::debug!(
        target: LEASE_TRACE_TARGET,
        object_id = trace.object_id,
        kind = trace.kind.as_str(),
        cut = trace.cut,
        outcome,
        via = via.as_str(),
        acquire_sent_unix_us = trace.acquire_sent_unix_us,
        acquire_replied_unix_us = trace.acquire_replied_unix_us,
        release_sent_unix_us,
        release_replied_unix_us,
        "snapshot fence lease"
    );
    verdict
}

/// [`probe_fence_on`] on the data pool.
pub async fn probe_fence(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<FenceProbe, ServiceError> {
    probe_fence_on(
        client.pool.next(),
        FredErrOrigin::client(client, POOL_LABEL_DATA),
        keys,
        object_id,
    )
    .await
}

/// Look at `object_id`'s fence without writing anything
/// (`snapshot_fence_probe.lua`, Spec SDD SR-5.4), on `conn`.
///
/// The WATCH path runs it on its `WATCH`ed connection, beside the header
/// read; `EVALSHA` does not disturb the `WATCH`. `origin` names that
/// connection's pool, where a command timeout is counted.
///
/// The probe also type-checks the fence and writer-intent keys: a wrong-typed
/// one answers [`FenceProbe::WrongType`] rather than an error, so the caller
/// refuses its commit with the same keyspace-type refusal the update script
/// gives in both modes.
pub async fn probe_fence_on(
    conn: &fred::clients::Client,
    origin: FredErrOrigin<'_>,
    keys: &KeySchema,
    object_id: u64,
) -> Result<FenceProbe, ServiceError> {
    const OP: &str = "EVALSHA snapshot_fence_probe";
    let oid = object_id.to_string();
    let script_keys = vec![
        keys.obj_snapshot_fence(&oid),
        keys.obj_snapshot_intent(&oid),
    ];
    let reply: Value = match FENCE_PROBE_SCRIPT
        .evalsha_with_reload::<Value, _, _>(conn, script_keys, Vec::<String>::new())
        .await
    {
        Ok(reply) => reply,
        Err(e) if e.details().starts_with(WRONGTYPE_KEY_TAG) => {
            return Ok(FenceProbe::WrongType {
                detail: e.details().to_owned(),
            })
        }
        Err(e) => return Err(map_transient_fred_err(origin, OP, e)),
    };
    let reply = reply.into_array();
    parse_probe(&reply).ok_or_else(|| {
        map_transient_fred_err(
            origin,
            OP,
            protocol_error(&format!("{OP} returned an unexpected reply: {reply:?}")),
        )
    })
}

/// Decode `snapshot_fence_probe.lua`'s `{exists, live, retry_after_ms,
/// size_e, size_b, bpi, epoch_start, gap_until, readers}`.
///
/// Any other length is a reply this binary's script never produces — a
/// 6-element reply is the earlier script's — so it answers `None` (a protocol
/// error at the caller). The script is loaded by SHA from this binary, so a
/// binary never meets another binary's reply in practice.
fn parse_probe(reply: &[Value]) -> Option<FenceProbe> {
    let [exists, live, retry_after, size_e, size_b, bpi, epoch_start, gap_until, readers] = reply
    else {
        return None;
    };
    match exists.as_i64()? {
        0 => return Some(FenceProbe::Absent),
        1 => {}
        _ => return None,
    }
    let live_retry_after = match live.as_i64()? {
        0 => None,
        1 => Some(Duration::from_millis(reply_u64(retry_after)?)),
        _ => return None,
    };
    let stored_size = match (size_e.as_i64()?, size_b.as_i64()?) {
        (-1, _) | (_, -1) => None,
        (items, bytes) => Some(LayerSize {
            items: u64::try_from(items).ok()?,
            bytes: u64::try_from(bytes).ok()?,
        }),
    };
    let bpi = bpi.as_string()?;
    let stored_bpi = if bpi.is_empty() {
        None
    } else {
        Some(bpi.parse::<f64>().ok()?)
    };
    let optional_ms = |value: &Value| match value.as_i64()? {
        -1 => Some(None),
        ms => u64::try_from(ms).ok().map(Some),
    };
    let marker = FenceMarker {
        epoch_start: optional_ms(epoch_start)?,
        gap_until: optional_ms(gap_until)?,
        readers: reply_u64(readers)?,
    };
    Some(FenceProbe::Present {
        live_retry_after,
        stored_size,
        stored_bpi,
        marker,
    })
}

/// Register a WATCH-path writer's intent on `object_id`
/// (`snapshot_intent_register.lua`, Spec SDD SR-5.4), refused while a reader
/// holds the fence, and answered [`IntentVerdict::NotFound`] — writing
/// nothing — when the object header is gone, so an intent key is never
/// recreated on a removed object.
///
/// `estimate` is the attempt's size class (the fence key's stored size, or
/// the fan-out tallies); the script uses the fence key's stored size whenever
/// it has one. Runs on the data pool.
pub async fn register_intent(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    estimate: LayerSize,
) -> Result<IntentVerdict, ServiceError> {
    const OP: &str = "EVALSHA snapshot_intent_register";
    let oid = object_id.to_string();
    let token = mint_token();
    let intent_key = keys.obj_snapshot_intent(&oid);
    let mut script_argv = Vec::with_capacity(3 + cost::FENCE_COEFFICIENT_ARGC);
    script_argv.push(token.clone());
    script_argv.push(estimate.items.to_string());
    script_argv.push(estimate.bytes.to_string());
    script_argv.extend(coefficients(client).script_argv());
    let reply: Value = match INTENT_REGISTER_SCRIPT
        .evalsha_with_reload::<Value, _, _>(
            client.pool.next(),
            vec![
                intent_key.clone(),
                keys.obj_snapshot_fence(&oid),
                keys.obj_header(&oid),
            ],
            script_argv,
        )
        .await
    {
        Ok(reply) => reply,
        Err(e) if e.details().starts_with(WRONGTYPE_KEY_TAG) => {
            return Ok(IntentVerdict::WrongType {
                detail: e.details().to_owned(),
            })
        }
        Err(e) => return Err(data_pool_err(client, OP, e)),
    };
    let reply = reply.into_array();
    let verdict = match verdict_word(&reply).as_deref() {
        Some("NOT_FOUND") if reply.len() == 1 => Some(IntentVerdict::NotFound),
        Some("FENCED") if reply.len() == 2 => {
            reply_u64(&reply[1]).map(|ms| IntentVerdict::Fenced {
                retry_after: Duration::from_millis(ms),
            })
        }
        Some("REGISTERED") if reply.len() == 2 => reply_u64(&reply[1])
            .filter(|ttl_ms| *ttl_ms > 0)
            .map(|ttl_ms| {
                IntentVerdict::Registered(Box::new(WriterIntent {
                    client: client.clone(),
                    intent_key,
                    token,
                    ttl_ms,
                    armed: true,
                }))
            }),
        _ => None,
    };
    verdict.ok_or_else(|| unexpected_reply(client, OP, &reply))
}

/// A WATCH-path writer's registered intent (Spec SDD SR-5.4), returned by
/// [`register_intent`].
///
/// Renew it every [`Self::renew_interval`] while the attempt runs and
/// [`Self::clear`] it on every exit. Dropping it uncleared spawns a
/// best-effort `HDEL` (module docs).
pub struct WriterIntent {
    client: ValkeyClient,
    intent_key: String,
    token: String,
    /// `intent_ttl(L)` as the register script computed it, whole ms.
    ttl_ms: u64,
    /// Whether dropping the guard still owes an `HDEL`.
    armed: bool,
}

// By hand, for the same reason as `FenceLease`'s: the token stays out of
// `{:?}` output.
impl std::fmt::Debug for WriterIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriterIntent")
            .field("intent_key", &self.intent_key)
            .field("ttl_ms", &self.ttl_ms)
            .field("armed", &self.armed)
            .finish_non_exhaustive()
    }
}

impl WriterIntent {
    /// `intent_ttl(L)`, as the register script computed it from the fence
    /// key's stored size (or the estimate).
    pub fn ttl(&self) -> Duration {
        Duration::from_millis(self.ttl_ms)
    }

    /// How often to renew: `intent_ttl(L) / 4`.
    pub fn renew_interval(&self) -> Duration {
        ms_to_duration(self.ttl_ms as f64 / 4.0)
    }

    /// Extend the intent if it is still live (`snapshot_intent_renew.lua`).
    /// Runs on the data pool.
    pub async fn renew(&self) -> Result<IntentRenewVerdict, ServiceError> {
        const OP: &str = "EVALSHA snapshot_intent_renew";
        let reply: Value = INTENT_RENEW_SCRIPT
            .evalsha_with_reload::<Value, _, _>(
                self.client.pool.next(),
                vec![self.intent_key.clone()],
                vec![self.token.clone(), self.ttl_ms.to_string()],
            )
            .await
            .map_err(|e| data_pool_err(&self.client, OP, e))?;
        let reply = reply.into_array();
        let verdict = match verdict_word(&reply).as_deref() {
            Some("RENEWED") if reply.len() == 2 && reply_u64(&reply[1]).is_some() => {
                Some(IntentRenewVerdict::Renewed)
            }
            Some("LOST") if reply.len() == 1 => Some(IntentRenewVerdict::Lost),
            _ => None,
        };
        verdict.ok_or_else(|| unexpected_reply(&self.client, OP, &reply))
    }

    /// Remove the intent (`HDEL`). On failure the guard stays armed, and its
    /// drop at the end of this call spawns one more best-effort `HDEL`.
    pub async fn clear(mut self) -> Result<(), ServiceError> {
        clear_intent(&self.client, &self.intent_key, &self.token).await?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for WriterIntent {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // No runtime: spawn nothing; the intent lapses at its deadline.
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let client = self.client.clone();
        let intent_key = std::mem::take(&mut self.intent_key);
        let token = std::mem::take(&mut self.token);
        // Tracked, like the lease's release (module docs, # Guards).
        let clear = async move {
            if let Err(error) = clear_intent(&client, &intent_key, &token).await {
                tracing::warn!(
                    intent_key = %intent_key,
                    error = %error,
                    "best-effort removal of a dropped writer intent failed; the intent lapses \
                     at its deadline, so readers of this layer may wait up to one intent \
                     lifetime longer"
                );
            }
        };
        self.client.spawn_fence_guard_cleanup(clear, &runtime);
    }
}

/// The one intent `HDEL`, shared by [`WriterIntent::clear`] and the guard's
/// drop.
async fn clear_intent(
    client: &ValkeyClient,
    intent_key: &str,
    token: &str,
) -> Result<(), ServiceError> {
    let _: i64 = client
        .pool
        .hdel(intent_key, format!("w:{token}"))
        .await
        .map_err(|e| data_pool_err(client, "HDEL snapshot intent", e))?;
    Ok(())
}

/// Classify a data-pool fred error (module docs, # Errors).
fn data_pool_err(client: &ValkeyClient, op: &str, e: fred::error::Error) -> ServiceError {
    map_transient_fred_err(FredErrOrigin::client(client, POOL_LABEL_DATA), op, e)
}

/// A script reply in a shape this binary's scripts never produce: a protocol
/// error, permanent by the mappers' rule.
fn unexpected_reply(client: &ValkeyClient, op: &str, reply: &[Value]) -> ServiceError {
    data_pool_err(
        client,
        op,
        protocol_error(&format!("{op} returned an unexpected reply: {reply:?}")),
    )
}

fn protocol_error(details: &str) -> fred::error::Error {
    fred::error::Error::new(fred::error::ErrorKind::Protocol, details.to_owned())
}

/// The verdict word every fence script's reply leads with.
fn verdict_word(reply: &[Value]) -> Option<Cow<'_, str>> {
    reply.first().and_then(Value::as_str)
}

/// A non-negative integer reply element.
fn reply_u64(value: &Value) -> Option<u64> {
    value.as_i64().and_then(|v| u64::try_from(v).ok())
}

/// A `size_e`, `size_b` pair of reply elements.
fn reply_size(items: &Value, bytes: &Value) -> Option<LayerSize> {
    Some(LayerSize {
        items: reply_u64(items)?,
        bytes: reply_u64(bytes)?,
    })
}

#[cfg(test)]
mod tests {
    //! The probe reply's decoding and the epoch-marker comparison the slow
    //! path's conflict classifier reads, with no store. The script that
    //! produces the reply runs against a live store in
    //! `tests/integration/storage/snapshot_fence_test.rs` (section 4), and the
    //! classifier through the commit seams in `snapshot_fence_writer_test.rs`.

    use super::*;

    fn int(n: i64) -> Value {
        Value::Integer(n)
    }

    fn text(s: &str) -> Value {
        Value::from(s)
    }

    fn marker(epoch_start: Option<u64>, gap_until: Option<u64>, readers: u64) -> FenceMarker {
        FenceMarker {
            epoch_start,
            gap_until,
            readers,
        }
    }

    /// A present, not-live probe carrying `marker`.
    fn present(marker: FenceMarker) -> FenceProbe {
        FenceProbe::Present {
            live_retry_after: None,
            stored_size: None,
            stored_bpi: None,
            marker,
        }
    }

    fn wrong_type() -> FenceProbe {
        FenceProbe::WrongType {
            detail: "WRONGTYPE_KEY: k holds a string, expected hash".to_owned(),
        }
    }

    /// The marker comparison, case by case (Spec SDD SR-5.4): a key that
    /// appeared or disappeared, or any one marker field that moved, is a
    /// change; an equal marker is none, whatever else the two probes say; a
    /// wrong-typed key on either side never is.
    #[test]
    fn changed_since_is_a_change_of_existence_or_of_any_one_marker_field() {
        let gap = marker(None, Some(1_000), 1);
        let cases: [(&str, FenceProbe, FenceProbe, bool); 14] = [
            (
                "absent, still absent",
                FenceProbe::Absent,
                FenceProbe::Absent,
                false,
            ),
            ("the key appeared", FenceProbe::Absent, present(gap), true),
            (
                "the key lapsed or was removed",
                present(gap),
                FenceProbe::Absent,
                true,
            ),
            ("nothing moved", present(gap), present(gap), false),
            (
                "a grant started an epoch",
                present(gap),
                present(marker(Some(2_000), None, 2)),
                true,
            ),
            (
                "the epoch_start alone moved",
                present(marker(Some(2_000), Some(1_000), 1)),
                present(marker(Some(2_001), Some(1_000), 1)),
                true,
            ),
            (
                "an epoch began and ended: a later gap_until alone",
                present(gap),
                present(marker(None, Some(1_500), 1)),
                true,
            ),
            (
                "the gap_until appeared",
                present(marker(None, None, 1)),
                present(marker(None, Some(1_000), 1)),
                true,
            ),
            (
                "a stale release lowered readers alone",
                present(marker(Some(2_000), None, 2)),
                present(marker(Some(2_000), None, 1)),
                true,
            ),
            (
                "the stored size moved, the marker did not",
                present(gap),
                FenceProbe::Present {
                    live_retry_after: Some(Duration::from_millis(5)),
                    stored_size: Some(LayerSize {
                        items: 150_216,
                        bytes: 1_799_752,
                    }),
                    stored_bpi: Some(11.98),
                    marker: gap,
                },
                false,
            ),
            (
                "wrong type, then absent",
                wrong_type(),
                FenceProbe::Absent,
                false,
            ),
            (
                "absent, then wrong type",
                FenceProbe::Absent,
                wrong_type(),
                false,
            ),
            (
                "present, then wrong type",
                present(gap),
                wrong_type(),
                false,
            ),
            (
                "wrong type, then present",
                wrong_type(),
                present(gap),
                false,
            ),
        ];
        for (what, earlier, later, changed) in cases {
            assert_eq!(
                later.changed_since(&earlier),
                changed,
                "{what}: {earlier:?} then {later:?}"
            );
        }
    }

    /// The script's absent reply, `{0, 0, 0, -1, -1, "", -1, -1, 0}`, is
    /// `Absent`; a present one decodes its marker, `-1` as no value.
    #[test]
    fn parse_probe_decodes_the_nine_element_reply_and_its_marker() {
        let absent = [
            int(0),
            int(0),
            int(0),
            int(-1),
            int(-1),
            text(""),
            int(-1),
            int(-1),
            int(0),
        ];
        assert_eq!(parse_probe(&absent), Some(FenceProbe::Absent));

        let in_a_gap = [
            int(1),
            int(0),
            int(0),
            int(150_216),
            int(1_799_752),
            text("11.98"),
            int(-1),
            int(1_791_241_585_054),
            int(1),
        ];
        assert_eq!(
            parse_probe(&in_a_gap),
            Some(FenceProbe::Present {
                live_retry_after: None,
                stored_size: Some(LayerSize {
                    items: 150_216,
                    bytes: 1_799_752,
                }),
                stored_bpi: Some(11.98),
                marker: marker(None, Some(1_791_241_585_054), 1),
            })
        );

        let live = [
            int(1),
            int(1),
            int(250),
            int(-1),
            int(-1),
            text(""),
            int(1_791_241_584_000),
            int(-1),
            int(3),
        ];
        assert_eq!(
            parse_probe(&live),
            Some(FenceProbe::Present {
                live_retry_after: Some(Duration::from_millis(250)),
                stored_size: None,
                stored_bpi: None,
                marker: marker(Some(1_791_241_584_000), None, 3),
            })
        );
    }

    /// Any length but nine is a reply this binary's script never gives — the
    /// earlier script's six elements included, absent or present — and so
    /// is a marker field below `-1`, or a negative reader count.
    #[test]
    fn parse_probe_refuses_the_six_element_reply_and_a_malformed_marker() {
        let six_absent = [int(0), int(0), int(0), int(-1), int(-1), text("")];
        let six_present = [int(1), int(0), int(0), int(10), int(20), text("2")];
        assert_eq!(
            parse_probe(&six_absent),
            None,
            "the earlier script's absent reply"
        );
        assert_eq!(
            parse_probe(&six_present),
            None,
            "the earlier script's present reply"
        );

        let good = [
            int(1),
            int(0),
            int(0),
            int(-1),
            int(-1),
            text(""),
            int(5),
            int(6),
            int(1),
        ];
        assert!(
            parse_probe(&good).is_some(),
            "precondition: the base reply decodes"
        );
        let mut ten = good.to_vec();
        ten.push(int(0));
        assert_eq!(parse_probe(&ten), None, "ten elements");
        assert_eq!(parse_probe(&good[..8]), None, "eight elements");
        for (at, bad) in [(6, -2), (7, -2), (8, -1)] {
            let mut reply = good.to_vec();
            reply[at] = int(bad);
            assert_eq!(parse_probe(&reply), None, "element {at} = {bad}");
        }
    }
}
