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

// Configuration loaded from environment variables.
// All variables use the LIVE_EDIT_ prefix. Missing required variables cause
// an immediate error on startup (fail-fast).

use std::env;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use crate::checkpoint::tail_codec::{TAIL_HEADER_SIZE, TAIL_RECORD_PREFIX_SIZE};
use crate::checkpoint::CheckpointAdapter;

pub mod memory_plan;
pub mod pod_memory;

pub use memory_plan::{PlanInputs, PodMemoryPlan, SUBSCRIPTION_CEILING_AT_REFERENCE_LIMIT};
pub use pod_memory::{
    PodMemoryLimit, PodMemoryLimitSource, PodMemorySource, REFERENCE_POD_MEMORY_LIMIT_BYTES,
    REFERENCE_POD_MEMORY_LIMIT_MIB,
};

/// Protocol used by the external auth clients (permissions + userinfo).
///
/// Selected at startup via `LIVE_EDIT_AUTH_PROTOCOL` (default: `rest`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthProtocol {
    Rest,
    Grpc,
}

/// Output format for structured tracing (b.3z7 H-obs-4).
///
/// Selected at startup via `LIVE_EDIT_LOG_FORMAT` (default: `json`).
/// `Pretty` is intended for local development — human-readable multi-line
/// output; `Json` is the production default so log aggregators get structured
/// records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Json,
    Pretty,
}

/// Error returned when `LIVE_EDIT_LOG_FORMAT` contains a value that is neither
/// `json` nor `pretty` (case-insensitive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseLogFormatError {
    input: String,
}

impl fmt::Display for ParseLogFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid LIVE_EDIT_LOG_FORMAT value {:?}: expected one of json, pretty (case-insensitive)",
            self.input
        )
    }
}

impl std::error::Error for ParseLogFormatError {}

impl FromStr for LogFormat {
    type Err = ParseLogFormatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(LogFormat::Json),
            "pretty" => Ok(LogFormat::Pretty),
            _ => Err(ParseLogFormatError {
                input: s.to_string(),
            }),
        }
    }
}

/// Default `LIVE_EDIT_TRACE_FILTER`: an allowlist that exports the service's
/// own INFO-and-above spans and events and nothing from any other crate.
pub const DEFAULT_TRACE_FILTER: &str = "live_edit=info";

/// Resolve `LIVE_EDIT_TRACE_FILTER` from its raw value (`None` when unset).
///
/// Unset yields [`DEFAULT_TRACE_FILTER`]. A set value is trimmed and must be a
/// non-empty, parseable `EnvFilter` directive list; anything else is an
/// error naming the variable, so a typo fails startup instead of silently
/// exporting nothing (or everything).
pub fn parse_trace_filter(raw: Option<&str>) -> Result<String, String> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_TRACE_FILTER.to_string());
    };
    let value = raw.trim();
    if value.is_empty() {
        return Err(format!(
            "invalid value for LIVE_EDIT_TRACE_FILTER: must not be empty; unset it for the \
             default ({DEFAULT_TRACE_FILTER}) or give EnvFilter directives such as \
             {DEFAULT_TRACE_FILTER}"
        ));
    }
    tracing_subscriber::EnvFilter::try_new(value).map_err(|e| {
        format!(
            "invalid value for LIVE_EDIT_TRACE_FILTER {value:?}: {e}; expected EnvFilter \
             directives such as {DEFAULT_TRACE_FILTER}"
        )
    })?;
    Ok(value.to_string())
}

/// The three logging variables the tracing subscriber is built from:
/// `LIVE_EDIT_LOG_LEVEL`, `LIVE_EDIT_LOG_FORMAT` and `LIVE_EDIT_TRACE_FILTER`.
///
/// Parsed apart from the rest of [`Config`] so `main.rs` can install the
/// subscriber *before* [`Config::from_env`] runs: a line logged while no
/// subscriber exists is silently dropped, so every WARN `from_env` emits
/// (and the startup lines `main.rs` logs right after it) reaches the log only
/// because the subscriber already exists. [`Config::from_env`] calls this same
/// function for its `log_level` / `log_format` / `trace_filter` fields, so the
/// fields can never disagree with the subscriber `main.rs` built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracingSettings {
    /// `LIVE_EDIT_LOG_LEVEL`, trimmed. Default: `"info"`. See
    /// [`Config::log_level`].
    pub log_level: String,
    /// `LIVE_EDIT_LOG_FORMAT`. Default: `json`. See [`Config::log_format`].
    pub log_format: LogFormat,
    /// `LIVE_EDIT_TRACE_FILTER`, resolved by [`parse_trace_filter`]. See
    /// [`Config::trace_filter`].
    pub trace_filter: String,
}

impl TracingSettings {
    /// Read the three logging variables from the environment.
    ///
    /// Returns `Err` naming the variable when `LIVE_EDIT_LOG_FORMAT` or
    /// `LIVE_EDIT_TRACE_FILTER` is invalid; the message is the one
    /// [`Config::from_env`] returns for the same value.
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            log_level: env::var("LIVE_EDIT_LOG_LEVEL")
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "info".to_string()),
            log_format: optional_parsed("LIVE_EDIT_LOG_FORMAT", LogFormat::Json)?,
            trace_filter: parse_trace_filter(env::var("LIVE_EDIT_TRACE_FILTER").ok().as_deref())?,
        })
    }
}

/// Error returned when `LIVE_EDIT_AUTH_PROTOCOL` contains a value that is
/// neither `rest` nor `grpc` (case-insensitive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseAuthProtocolError {
    input: String,
}

impl fmt::Display for ParseAuthProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid LIVE_EDIT_AUTH_PROTOCOL value {:?}: expected one of rest, grpc (case-insensitive)",
            self.input
        )
    }
}

impl std::error::Error for ParseAuthProtocolError {}

impl FromStr for AuthProtocol {
    type Err = ParseAuthProtocolError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "rest" => Ok(AuthProtocol::Rest),
            "grpc" => Ok(AuthProtocol::Grpc),
            _ => Err(ParseAuthProtocolError {
                input: s.to_string(),
            }),
        }
    }
}

/// Backend used by `update_layer` for committing mutations (b.67u).
///
/// - `Lua` (default): atomic server-side Lua EVAL for the field/time-sample
///   fast path; structural and reset deltas are admitted one at a time per
///   object per replica (the per-object `structural` permit), planned on the
///   per-replica structural cache and committed by the same script's
///   `seq`-fenced structural mode. An object the cache cannot hold commits
///   them on the legacy WATCH/MULTI/EXEC path.
/// - `LuaWatchStructural` (`lua-watch-structural`): the same fast path, with
///   structural and reset deltas on the legacy WATCH/MULTI/EXEC path under the
///   plain per-object writer admission — the rollback for `Lua`'s structural
///   route.
/// - `Txn`: force the legacy WATCH/MULTI/EXEC path for **every** `update_layer`
///   call, regardless of whether the delta qualifies as fast-path, under the
///   plain per-object writer admission. Intended for parity testing and as an
///   emergency rollback knob.
///
/// Selected at startup via `LIVE_EDIT_UPDATE_LAYER_BACKEND` (default: `lua`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateLayerBackend {
    Lua,
    LuaWatchStructural,
    Txn,
}

/// Error returned when `LIVE_EDIT_UPDATE_LAYER_BACKEND` contains a value that
/// is none of `lua`, `lua-watch-structural`, `txn` (case-insensitive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseUpdateLayerBackendError {
    input: String,
}

impl fmt::Display for ParseUpdateLayerBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid LIVE_EDIT_UPDATE_LAYER_BACKEND value {:?}: expected one of lua, lua-watch-structural, txn (case-insensitive)",
            self.input
        )
    }
}

impl std::error::Error for ParseUpdateLayerBackendError {}

impl FromStr for UpdateLayerBackend {
    type Err = ParseUpdateLayerBackendError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "lua" => Ok(UpdateLayerBackend::Lua),
            "lua-watch-structural" => Ok(UpdateLayerBackend::LuaWatchStructural),
            "txn" => Ok(UpdateLayerBackend::Txn),
            _ => Err(ParseUpdateLayerBackendError {
                input: s.to_string(),
            }),
        }
    }
}

/// What the CreateLayer / UpdateLayer content check does when an external
/// value fails it (SDD §11.1). The check itself — `extValueSize` equals the
/// part's length, and `valueOrExtHash` equals the part's 1 MiB block content
/// hash — runs in both modes and is counted in
/// `live_edit_ext_value_verify_total`.
///
/// - `Enforce` (default): the call is refused `INVALID_ARGUMENT` before
///   anything is stored. This is the client contract. A call that passes is
///   stored as verified, which also replaces an existing data key under one
///   of its hashes that is not a string or whose length differs from the
///   verified bytes.
/// - `Observe`: the mismatch is counted and logged at WARN and the value is
///   stored anyway. A rollout-safety setting for a fleet whose clients have
///   not been confirmed to compute the content hash; the set-if-absent store
///   still keeps any copy already stored under the hash, and, unlike
///   `Enforce`, never replaces a damaged one (a wrong-typed or wrong-length
///   data key) with the uploaded bytes. A claim that is not
///   32 bytes is refused in both modes, because it breaks the protocol's
///   32-byte rule: CreateLayer and the UpdateLayer slow path already refuse it
///   in `apply_delta`, after the value was stored and its pre-flight increment
///   paid (a leak), while the UpdateLayer Lua fast path never runs
///   `apply_delta` and would commit the padded or truncated claim. `observe`
///   therefore does not restore that fast-path behaviour for such a claim.
///   Selecting it logs one startup WARN (`Config::warn_ext_value_verify_observe`).
///
/// Selected at startup via `LIVE_EDIT_EXT_VALUE_VERIFY` (default: `enforce`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtValueVerifyMode {
    Enforce,
    Observe,
}

/// Error returned when `LIVE_EDIT_EXT_VALUE_VERIFY` is neither `enforce` nor
/// `observe` (case-insensitive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseExtValueVerifyModeError {
    input: String,
}

impl fmt::Display for ParseExtValueVerifyModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid LIVE_EDIT_EXT_VALUE_VERIFY value {:?}: expected one of enforce, observe (case-insensitive)",
            self.input
        )
    }
}

impl std::error::Error for ParseExtValueVerifyModeError {}

impl FromStr for ExtValueVerifyMode {
    type Err = ParseExtValueVerifyModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "enforce" => Ok(ExtValueVerifyMode::Enforce),
            "observe" => Ok(ExtValueVerifyMode::Observe),
            _ => Err(ParseExtValueVerifyModeError {
                input: s.to_string(),
            }),
        }
    }
}

/// Authentication mode used when connecting to the Valkey backing store
/// (SDD SR-2.1).
///
/// - `Static` (default): password/ACL credentials supplied directly (or none).
/// - `AccessTokenFile`: read a bearer token from a sidecar-written file and use
///   it as the Valkey AUTH credential (a generic file-based passwordless mode;
///   the token is conventionally minted by an Azure Entra managed-identity
///   sidecar, but from the service's runtime perspective the mode is agnostic
///   to the token issuer).
/// - `Iam`: AWS IAM token-based authentication.
///
/// Selected at startup via `LIVE_EDIT_VALKEY_AUTH_MODE` (default: `static`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValkeyAuthMode {
    Static,
    AccessTokenFile,
    Iam,
}

/// Error returned when `LIVE_EDIT_VALKEY_AUTH_MODE` contains a value that is
/// none of `static`, `accessTokenFile`, `iam` (case-insensitive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseValkeyAuthModeError {
    input: String,
}

impl fmt::Display for ParseValkeyAuthModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid LIVE_EDIT_VALKEY_AUTH_MODE value {:?}: expected one of static, accessTokenFile, iam (case-insensitive)",
            self.input
        )
    }
}

impl std::error::Error for ParseValkeyAuthModeError {}

impl FromStr for ValkeyAuthMode {
    type Err = ParseValkeyAuthModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "static" => Ok(ValkeyAuthMode::Static),
            "accesstokenfile" => Ok(ValkeyAuthMode::AccessTokenFile),
            "iam" => Ok(ValkeyAuthMode::Iam),
            _ => Err(ParseValkeyAuthModeError {
                input: s.to_string(),
            }),
        }
    }
}

/// Selects which late-join / catch-up regime `SubscribeReadLayer` uses when a
/// subscriber falls out of the live stream window (SDD §12.3).
///
/// AWS managed backend selected when `LIVE_EDIT_VALKEY_AUTH_MODE=iam`.
///
/// v1 supports both AWS managed Valkey backends behind IAM auth:
///
/// - `MemoryDb` (default): Amazon MemoryDB for Valkey (recommended).
/// - `ElastiCache`: Amazon ElastiCache for Valkey (supported).
///
/// Selected at startup via `LIVE_EDIT_VALKEY_IAM_SERVICE` (default: `memorydb`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValkeyIamService {
    MemoryDb,
    ElastiCache,
}

/// Error returned when `LIVE_EDIT_VALKEY_IAM_SERVICE` contains a value that is
/// neither `memorydb` nor `elasticache` (case-insensitive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseValkeyIamServiceError {
    input: String,
}

impl fmt::Display for ParseValkeyIamServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid LIVE_EDIT_VALKEY_IAM_SERVICE value {:?}: expected one of memorydb, elasticache (case-insensitive)",
            self.input
        )
    }
}

impl std::error::Error for ParseValkeyIamServiceError {}

impl FromStr for ValkeyIamService {
    type Err = ParseValkeyIamServiceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "memorydb" => Ok(ValkeyIamService::MemoryDb),
            "elasticache" => Ok(ValkeyIamService::ElastiCache),
            _ => Err(ParseValkeyIamServiceError {
                input: s.to_string(),
            }),
        }
    }
}

/// All runtime configuration for the Live Edit service.
#[derive(Debug, Clone)]
pub struct Config {
    /// gRPC listen port (LIVE_EDIT_PORT). Default: 50001.
    pub port: u16,
    /// Prometheus metrics HTTP port (LIVE_EDIT_METRICS_PORT). Default: 9090.
    pub metrics_port: u16,
    /// Valkey connection URL (LIVE_EDIT_VALKEY_URL). Default: redis://localhost:6379.
    pub valkey_url: String,
    /// Connection pool size; 0 means auto (cores × 2) (LIVE_EDIT_VALKEY_POOL_SIZE). Default: 0.
    pub valkey_pool_size: u32,
    /// Stream-multiplexer reader connections (LIVE_EDIT_SUBSCRIBER_POOL_SIZE).
    /// Default: 0 (auto = `cores × 2`).
    ///
    /// Each reader is one dedicated Valkey connection running one `XREAD BLOCK`
    /// over every layer stream its subscriptions need; subscriptions share
    /// readers, so this does not cap the subscription count
    /// ([`Self::subscriber_max_subscriptions`] does). Separate from the
    /// `txn_pool`, so readers cannot starve writer transactions. A reader
    /// connects when it first serves a subscription, not at startup, so this is
    /// a ceiling on reader connections. In Valkey Cluster one XREAD may only
    /// name keys of one hash slot, so a reader serves one slot (in practice one
    /// layer) at a time, connects to that slot's primary alone, and this bounds
    /// the distinct layers a replica can serve concurrently.
    pub subscriber_pool_size: u32,
    /// Concurrent `SubscribeReadLayer` subscriptions admitted per replica,
    /// `S`: derived at startup from the pod memory limit, 0.8 per MiB of it
    /// (4,096 at the chart's 5 GiB), never configured
    /// ([`PodMemoryPlan::max_subscriptions`]). An established subscription
    /// holds no Valkey connection, only memory, so its ceiling is a memory
    /// share: the plan charges `S` × the 49.8 KiB one measured on the line
    /// beside every other term (`config::memory_plan`). Always at least 1;
    /// `LIVE_EDIT_SUBSCRIBER_MAX_SUBSCRIPTIONS`, which set it before, is
    /// refused at startup.
    ///
    /// Copied from [`Self::pod_memory`] at load (`with_pod_memory_plan`); the
    /// plan is the authority.
    pub subscriber_max_subscriptions: usize,
    /// Catch-ups holding a slot at once per replica
    /// (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS). Default: 8. Zero is
    /// rejected at startup: each slot's memory, 42 ×
    /// [`Self::snapshot_atomic_read_max_bytes`] (24.22 MiB at the default
    /// ceiling), is charged to the pod memory plan, and an unlimited count
    /// cannot be. The default is CPU-derived: a catch-up build is CPU-bound,
    /// so the default allows 4 slots per CPU at the chart's 2-CPU limit: more
    /// slots do not raise catch-up throughput there, they only lengthen each
    /// catch-up and take more CPU from writes and live delivery. Raising it
    /// lowers the catch-up memory budget
    /// ([`Self::catchup_build_memory_budget_mib`]), or, past what the pod
    /// memory limit can carry, refuses startup (`config::memory_plan`).
    /// A catch-up (a re-bootstrap's included) holds a slot only
    /// across its backing-store reads and its build — and, in rollback mode
    /// (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) on a same-object reconnect,
    /// the cluster stream-reader wait — never across a send and never on the
    /// up-to-date path. A shared fenced-read flight
    /// (`storage::snapshot_flights`) is one unit of catch-up work: it holds one
    /// slot from its read, through its fence `RETRY_AFTER` sleeps and its
    /// permit and memory waits, to the last build of the catch-ups attached to
    /// it (which hold none), and each of its extra builds holds one more for
    /// that build. Its slot wait has no deadline; each attached catch-up is
    /// charged its share of it. Past the limit a catch-up waits in FIFO
    /// order within the subscription's capacity budget
    /// ([`Self::subscriber_admission_wait_ms`]; cancelled by session deletion,
    /// shutdown or client disconnect) instead of all allocating a full layer
    /// state at once, and is refused `UNAVAILABLE` (`catchup_capacity`) when the
    /// budget runs out: before any content during catch-up, in both catch-up
    /// modes; after the caught-up ack only for a steady-state overwrite
    /// re-bootstrap. A value above
    /// `tokio::sync::Semaphore::MAX_PERMITS` is rejected at startup.
    pub subscriber_max_concurrent_catchups: usize,
    /// A `SubscribeReadLayer`'s single capacity-wait budget
    /// (LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS). Default: 20000. One budget
    /// covers every wait for capacity the subscribe makes: for an admission
    /// slot, in cluster mode for a free stream reader, for each catch-up slot
    /// and for each snapshot read permit
    /// ([`Self::catchup_snapshot_read_concurrency`]), and, for a catch-up
    /// attached to a shared fenced-read flight, the time it is attached while
    /// the flight waits for its slot, its fenced read permit
    /// ([`Self::catchup_fenced_read_concurrency`]) or its memory
    /// ([`Self::catchup_build_memory_budget_mib`]); when the budget runs out
    /// that catch-up alone is refused. Time spent reading, building or sending
    /// is not charged, nor is a flight's fence sleeps, its wait for a
    /// predecessor flight or a wait for a build turn. A re-bootstrap onto an overwritten
    /// layer's new object starts a fresh budget. Zero means never wait for
    /// capacity: each of those waits takes a free slot, reader or permit or is
    /// refused at once, and no queue is joined (b.f4f).
    ///
    /// The 20 s default is derived from the tightest client retry window
    /// (SDD §16): the longest budget, rounded down to 5 s, that still leaves a
    /// client first open refused once at the budget a complete retry inside
    /// its 30 s open window.
    ///
    /// The admission wait is cancellation-aware: it races the service shutdown
    /// token (refused `UNAVAILABLE`, SDD §5.6) and then the per-session
    /// cancellation token (a deleted session, refused `NOT_FOUND`), and a
    /// client that disconnects while waiting cancels it implicitly, because
    /// hyper drops the handler future. Admission is taken *after* the
    /// read-permission check — so pod saturation is not observable to an
    /// unauthorized caller — and *before* the response channel and the
    /// catch-up spawn, so an admission refusal is a Trailers-Only status with
    /// no task spawned behind it. The reader, catch-up-slot and read-permit
    /// waits (a shared fenced-read flight's slot, permit and memory waits
    /// included, as its attached catch-ups' shares of them) come after the
    /// stream opens and race the same three endings, and their refusals
    /// (`reader_capacity`, `catchup_capacity` for the others) end the stream
    /// `UNAVAILABLE`:
    /// before any content during catch-up; after the caught-up ack only for a
    /// steady-state overwrite re-bootstrap.
    pub subscriber_admission_wait_ms: u64,
    /// How many `SubscribeReadLayer` RPCs may be queued for admission at once
    /// (LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS). `None` — the default —
    /// means "track [`Self::subscriber_max_subscriptions`]".
    ///
    /// Zero is **rejected at startup** rather than accepted: a zero cap next to
    /// a non-zero wait makes the wait silently dead code, the same failure mode
    /// [`Self::checkpoint_tail_max_bytes`]' zero-rejection exists to prevent.
    /// Fail-fast is expressed by `subscriber_admission_wait_ms = 0`.
    pub subscriber_admission_max_waiters: Option<usize>,
    /// Maintenance pool size: dedicated Valkey connections used by the
    /// background GC sweep + orphan sweep (and the reference-count reconciler's
    /// and the external-value backup repair's keyspace walks) so they cannot
    /// starve the data pool with long-running SCAN / HGETALL passes
    /// (LIVE_EDIT_MAINTENANCE_POOL_SIZE). Default: 1.
    ///
    /// One connection is enough because all of them are gated behind the
    /// `le:gc:leader` lock (b.kf1 H-stor-4) — normally one replica runs them
    /// at a time (two can briefly overlap after a lock takeover). Operators who want extra parallelism for very large
    /// keyspaces can raise this; values above ~4 are rarely useful because
    /// the sweep loop itself is single-threaded inside the leader.
    pub maintenance_pool_size: u32,
    /// Catch-up atomic snapshot reads in flight at once per replica, `P_a`
    /// (LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY; Spec SDD SR-7.1). Default:
    /// [`CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT`] = 1, the value the
    /// atomic-read ceilings were solved at. Zero is rejected at startup.
    ///
    /// The count of read permits a catch-up takes for its atomic read. A
    /// catch-up takes its permit inside its catch-up slot
    /// ([`Self::subscriber_max_concurrent_catchups`]) and holds it across the
    /// read only; the wait is a capacity wait, charged to the subscription's
    /// budget ([`Self::subscriber_admission_wait_ms`]) and refused
    /// `UNAVAILABLE` (`catchup_capacity`) before any content when it runs out.
    /// A legacy in-window replay takes one per replayed commit the same way,
    /// except for an empty commit, which reads nothing. The steady state never
    /// reads layer state and takes none. It bounds the shard-blocking atomic
    /// reads one replica issues at
    /// once, so raising it raises the writer delay a join wave can cause; the
    /// ceilings hold only at the value they were solved at.
    pub catchup_snapshot_read_concurrency: u32,
    /// Catch-up fenced snapshot reads in flight at once per replica, `P_f`
    /// (LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY; Spec SDD SR-7.1). Default:
    /// [`CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT`] = 1. Zero is rejected at
    /// startup.
    ///
    /// The count of the second read-permit class, held by the read of a layer
    /// above the atomic-read ceilings, so a multi-second fenced read never
    /// takes the permits the atomic reads of other layers need. It bounds the
    /// fenced fan-out windows one replica keeps outstanding, and the
    /// over-commit of the catch-up memory budget
    /// ([`Self::catchup_build_memory_budget_mib`]): only a fenced-permit holder
    /// runs a budgeted fenced read, the only read whose memory grows past its
    /// grant, so resident above-ceiling catch-up memory stays within
    /// `T × (1 + P_f) + P_f × X` for layers whose weight is within `T`, `X`
    /// being one read stage's fan-out window transient, which the grant does
    /// not reserve (`storage::memory_budget`). The snapshot
    /// lane's default ([`Self::snapshot_pool_size`]) grows with it. Taken by
    /// a shared fenced-read flight's read stage (`storage::snapshot_flights`),
    /// inside the flight's catch-up slot, per fence acquire attempt, with no
    /// deadline (each attached catch-up is charged its share of the wait);
    /// held through the fenced read, the fence's release and the memory
    /// adjust after it, and given back during every `RETRY_AFTER` sleep.
    ///
    /// The structural write-planning cache's fenced cold start takes one too
    /// (`storage::structural_cache_fenced`), the same way, after the cache's
    /// one-at-a-time and plain cold-start permits and with a deadline capped
    /// by its call's RPC deadline, so at most one cold start per replica waits
    /// for it, and a catch-up flight waits behind at most one cold start's
    /// fenced read. Its waits are observed under
    /// `consumer="structural_cache"`. A catch-up never waits on the cache's
    /// permits, so the two form no wait cycle.
    pub catchup_fenced_read_concurrency: u32,
    /// The per-replica catch-up memory budget `T`, in MiB (Spec SDD SR-7.2):
    /// derived at startup from the pod memory limit, never configured
    /// ([`PodMemoryPlan::catchup_budget_mib`]). It is the residual of the
    /// pod memory plan's line: the largest 0.01 step of the limit less the
    /// 155 MiB floor at which every term of the line, charged at its exact
    /// share of the limit, fits 80 % of the limit (`config::memory_plan`,
    /// *Exact shares*), so it never decreases as the limit grows; 744 at the
    /// chart's 5 GiB
    /// ([`CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT`]). Always at least 1;
    /// `LIVE_EDIT_CATCHUP_BUILD_MEMORY_BUDGET_MIB`, which set it before, is
    /// refused at startup.
    ///
    /// Sizes the `storage::memory_budget::MemoryBudget` that reservations for
    /// above-ceiling catch-up snapshots and their builds draw on. A
    /// reservation larger than `T` is clamped to it and runs alone, so no
    /// layer is refused for its size. A shared fenced-read flight
    /// (`storage::snapshot_flights`) reserves one combined grant (its snapshot
    /// plus one build) per fence acquire attempt, with no deadline, and each
    /// of its extra builds takes a non-blocking build part. The structural
    /// write-planning cache's fenced cold start draws on it too
    /// (`storage::structural_cache_fenced`): `b × (SNAPSHOT_MEMORY_FACTOR + 1)`
    /// before its acquire, adjusted after a verified read, and given back once
    /// the snapshot is decoded, before the entry is admitted. The budget's
    /// series carry no consumer label (policy decision P13): cold starts are
    /// rare, and their `P_f` wait is told apart by `consumer`.
    ///
    /// Copied from [`Self::pod_memory`] at load (`with_pod_memory_plan`); the
    /// plan is the authority.
    pub catchup_build_memory_budget_mib: u64,
    /// Snapshot-lane size: dedicated multiplexed Valkey connections for
    /// object-snapshot reads (LIVE_EDIT_SNAPSHOT_POOL_SIZE; Spec SDD SR-10.1).
    /// Default: [`snapshot_pool_size_default`] of the configured
    /// [`Self::catchup_snapshot_read_concurrency`] and
    /// [`Self::catchup_fenced_read_concurrency`], `P_a + P_f + 4` (6 at the
    /// defaults): one connection per atomic read (catch-up or per-commit) and
    /// per fenced read a replica runs at once, plus four for the structural
    /// and checkpoint reads and the ext-GC reference walks.
    ///
    /// A lane of its own so multi-MiB snapshot replies never queue in front of
    /// `UpdateLayer` traffic on the data pool. Built from the data pool's
    /// builder, so it shares that pool's credentials, TLS policy, reconnect
    /// policy and timeouts. Zero is rejected at startup.
    pub snapshot_pool_size: u32,
    /// Valkey connect bound in seconds (LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS).
    /// Default: 30. Zero is rejected at startup.
    ///
    /// The hard budget of each startup connect phase (its retries included)
    /// and each fred connection attempt's timeout; the bound on one stream
    /// reader connect; and the bound on the readers' cluster routing sync
    /// after a `MOVED` when [`Self::valkey_command_timeout_ms`] is 0. A zero
    /// would cut every one of them before it could finish.
    pub valkey_connect_timeout_secs: u64,
    /// Per-command timeout applied to every Valkey command in ms
    /// (LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS). Default: 5000. 0 disables the timeout.
    ///
    /// Prevents commands from hanging indefinitely when a pooled connection is
    /// mid-reconnect (e.g. after a Valkey restart). Pair with
    /// `valkey_reconnect_max_delay_ms` so the pool self-heals.
    pub valkey_command_timeout_ms: u64,
    /// Upper bound on the exponential reconnect backoff in ms
    /// (LIVE_EDIT_VALKEY_RECONNECT_MAX_DELAY_MS). Default: 30000.
    ///
    /// fred uses unlimited reconnect attempts (`max_attempts = 0`) starting at
    /// 100 ms and doubling up to this cap.
    pub valkey_reconnect_max_delay_ms: u32,
    /// Valkey authentication mode (LIVE_EDIT_VALKEY_AUTH_MODE). Default: Static.
    pub valkey_auth_mode: ValkeyAuthMode,
    /// Redis-scoped access-token-file path for Valkey AUTH
    /// (LIVE_EDIT_VALKEY_TOKEN_FILE). Default: None.
    pub valkey_token_file: Option<PathBuf>,
    /// Valkey AUTH username (conventionally the token issuer's identity, e.g. an
    /// Azure managed identity's Object ID)
    /// (LIVE_EDIT_VALKEY_TOKEN_USERNAME). Default: None.
    pub valkey_token_username: Option<String>,
    /// Token-refresh lead time in seconds before expiry
    /// (LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS). Default: 180 (Azure's
    /// guidance is to fetch a fresh token ≥ 3 min before expiry).
    ///
    /// Drives the Redis-scoped `ServiceTokenProvider`'s proactive re-read
    /// window only — it does NOT withhold a still-valid cached token from
    /// Valkey `AUTH` (b.d6b).
    ///
    /// Validated against zero and against timestamp representability
    /// ([`VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS`] — the lead is added to
    /// `Utc::now()` inside the provider, so the `DateTime<Utc>` range bounds it)
    /// only; there is deliberately no operational upper bound (b.is1).
    pub valkey_token_refresh_lead_secs: u64,
    /// IAM-enabled ACL username for Valkey AUTH (LIVE_EDIT_VALKEY_IAM_USERNAME). Default: None.
    pub valkey_iam_username: Option<String>,
    /// AWS region for IAM SigV4 signing (LIVE_EDIT_VALKEY_IAM_REGION). Default: None.
    pub valkey_iam_region: Option<String>,
    /// Cluster/cache **name** used as the IAM SigV4 signing host
    /// (LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST) — the short cluster name for MemoryDB
    /// (lowercased cache name for ElastiCache), NOT the endpoint FQDN (that goes
    /// in LIVE_EDIT_VALKEY_URL). Default: None.
    pub valkey_iam_cluster_host: Option<String>,
    /// AWS managed backend selected under `iam` auth mode
    /// (LIVE_EDIT_VALKEY_IAM_SERVICE). Default: MemoryDb.
    pub valkey_iam_service: ValkeyIamService,
    /// Storage service URL for path-level authorization (LIVE_EDIT_STORAGE_SERVICE_ENDPOINT).
    /// Required when `authz_enabled` is true; optional otherwise.
    pub storage_service_endpoint: Option<String>,
    /// Auth service call timeout in ms (LIVE_EDIT_AUTH_TIMEOUT_MS). Default: 5000.
    pub auth_timeout_ms: u64,
    /// Auth result cache TTL in seconds (LIVE_EDIT_AUTH_CACHE_TTL_SECS). Default: 600.
    ///
    /// Applies to positive entries (an `Allowed` permission verdict, a
    /// successful userinfo lookup). An entry for a bearer that is a JWT with a
    /// numeric `exp` ends at the earlier of this TTL and that `exp`; an opaque
    /// bearer's entry lives the full TTL. See `auth::cache::AuthCache`.
    pub auth_cache_ttl_secs: u64,
    /// Auth result cache TTL for negative decisions (Denied / Unauthenticated /
    /// PathNotFound / InvalidRequest) in seconds
    /// (LIVE_EDIT_AUTH_CACHE_NEGATIVE_TTL_SECS). Default: 30.
    ///
    /// Separate from `auth_cache_ttl_secs` so revocations propagate within
    /// seconds rather than up to the 10-minute positive TTL. Positive (Allowed)
    /// decisions are cached for `auth_cache_ttl_secs`; negatives use this
    /// shorter TTL. Set this value ≤ the positive TTL in practice — a larger
    /// value is accepted but defeats the purpose of the split.
    pub auth_cache_negative_ttl_secs: u64,
    /// Userinfo cache TTL for error decisions in seconds
    /// (LIVE_EDIT_AUTH_USERINFO_NEGATIVE_TTL_SECS). Default: 5.
    ///
    /// Pre-b.ohw the userinfo cache only memoized successful lookups, so a
    /// bearer the userinfo service rejected caused every RPC on that token to
    /// hit the userinfo backend until the token was refreshed. Caching the
    /// failure with a
    /// short TTL bounds the duplicate-request fanout while keeping the
    /// revocation-propagation window tight (default 5 s).
    pub auth_userinfo_negative_ttl_secs: u64,
    /// Maximum number of entries held by each inner auth cache
    /// (permissions + userinfo) before LRU eviction kicks in
    /// (LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES). Default: 10_000.
    ///
    /// TTL alone leaves the cache unbounded between expirations, so under
    /// token churn the cache can grow roughly linearly with insertion rate.
    /// This cap gives each inner cache a hard entry ceiling; zero is rejected
    /// at parse time because it would effectively disable the cache.
    pub auth_cache_max_entries: u64,
    /// Userinfo service URL (LIVE_EDIT_USERINFO_ENDPOINT).
    /// Required when `userinfo_service_enabled = true`; optional otherwise.
    ///
    /// Mirrors the `storage_service_endpoint` pattern: when the userinfo
    /// service is enabled, `Config::from_env` requires this var to be set
    /// and runs the fail-closed scheme validator on it. When the userinfo
    /// service is disabled, the var is read with `env::var(...).ok()` and
    /// no scheme validation is performed.
    pub userinfo_endpoint: Option<String>,
    /// Cap, in bytes, on the body of any one CreateLayer or UpdateLayer upload
    /// (LIVE_EDIT_MAX_LAYER_SIZE); a larger upload fails INVALID_ARGUMENT. It
    /// bounds a single upload, not the size a layer reaches across updates.
    /// Default: [`MAX_LAYER_SIZE_DEFAULT`] = 104857600 (100 MiB). The upload
    /// budget's floor is derived from it ([`Self::upload_memory_budget_mib`]),
    /// and the pod memory plan charges one upload of this size in the
    /// overflow slot.
    pub max_layer_size: usize,
    /// The per-replica upload memory budget `B`, in MiB: derived at startup
    /// from the pod memory limit, never configured
    /// ([`PodMemoryPlan::upload_budget_mib`]): 7.5 % of the limit (384 at the
    /// chart's 5 GiB), and never below the floor `F + A + 2 × max_layer_size`
    /// ([`upload_budget_floor_bytes`](crate::grpc::upload_admission::upload_budget_floor_bytes))
    /// rounded up to a MiB, so no upload under [`Self::max_layer_size`] is
    /// refused for its size. Every CreateLayer / UpdateLayer reserves from it
    /// after its authorization and before reading its content, and holds the
    /// reservation to the end of the call (`grpc::upload_admission`, SDD
    /// §5.4); one whose processing does not fit runs in the overflow slot, one
    /// at a time beyond the budget. `LIVE_EDIT_UPLOAD_MEMORY_BUDGET_MIB`, which
    /// set it before, is refused at startup.
    ///
    /// Copied from [`Self::pod_memory`] at load (`with_pod_memory_plan`); the
    /// plan is the authority.
    pub upload_memory_budget_mib: u64,
    /// How long an upload may wait for admission to the upload budget, in ms
    /// (LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS). Default when unset: derived,
    /// [`upload_admission_wait_ms_default`](crate::grpc::upload_admission::upload_admission_wait_ms_default)
    /// = `min(2000, rpc_deadline_ms / 2)` — 2000 at the 30 s default deadline.
    /// 0 means never wait (refused at once when capacity is full). A wait
    /// ends refused UNAVAILABLE with nothing applied. An explicitly set value
    /// at or above [`Self::rpc_deadline_ms`] is rejected at startup: the
    /// deadline would always end the wait first.
    pub upload_admission_wait_ms: u64,
    /// How many uploads may wait for admission at once
    /// (LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS). Default:
    /// [`UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT`](crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT)
    /// = 64. One more is refused UNAVAILABLE at once. Zero is rejected at
    /// startup; fail-fast is `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS=0`.
    pub upload_admission_max_waiters: usize,
    /// The entry cap on per-layer Streams retention, applied as `MAXLEN ~ N`
    /// on every `XADD` to the layer stream (LIVE_EDIT_LAYER_STREAM_MAXLEN).
    /// Default: 4096. `0` means no entry cap.
    ///
    /// Each entry carries the writer's verbatim relay `ompf` field (b.gze) —
    /// up to `LIVE_EDIT_MAX_LAYER_SIZE` of OMPF bytes per entry — so the
    /// stream's cost is in bytes, which [`Self::layer_stream_max_bytes`]
    /// bounds; this cap bounds the entry count beside it. A stream retains
    /// the smaller of the two windows. 4096 covers cursor-tracking workloads
    /// (60 fps × ~10 collaborators) with several seconds of retention headroom
    /// before subscribers fall off the window and drop to the out-of-window
    /// (coalesced) catch-up path. See `warn_layer_stream_memory_footprint` for
    /// the startup WARN guard on the worst-case footprint.
    pub layer_stream_maxlen: u64,
    /// The per-object byte budget on retained relay bytes
    /// (LIVE_EDIT_LAYER_STREAM_MAX_BYTES). Default:
    /// [`LAYER_STREAM_MAX_BYTES_DEFAULT`] (128 MiB). `0` turns it off
    /// (count-only retention, the no-redeploy rollback). Values above
    /// `2^53 - 1` are rejected (the Lua f64 integer ceiling, as for
    /// [`Self::layer_stream_maxlen`]).
    ///
    /// Every `UpdateLayer` commit folds its relay bytes into the header's
    /// `relay_gens` and trims the stream head when the newest entries' bytes
    /// pass the budget (`storage::relay_retention`), so retained relay bytes
    /// per object stay at most about this plus 2 × `LIVE_EDIT_MAX_LAYER_SIZE`,
    /// and never fewer than 2 entries are kept.
    pub layer_stream_max_bytes: u64,
    /// How many presence events a local JoinSession stream's queue cursor
    /// may fall behind before it loses any
    /// (LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY). Default:
    /// [`crate::presence::DEFAULT_BROADCAST_CHANNEL_CAPACITY`] (4096); must be
    /// greater than 0.
    ///
    /// The capacity of each session's fan-out queue, applied exactly to how
    /// far a stream's queue cursor may fall behind; up to `RECV_BATCH - 1`
    /// more events may already be taken into the stream's read-ahead buffer,
    /// and those can no longer be lost (SDD §13.1). It is a lag bound, not a
    /// reservation: beyond a small fixed floor, a queue holds memory only for
    /// the events some local stream has not read yet, so an idle session
    /// costs the same at any value. A stream that falls further behind loses the oldest
    /// events it had not read; the loss is counted on
    /// `live_edit_presence_broadcast_dropped_total`, and a stream that lost a
    /// participant's join or leave is ended `UNAVAILABLE` so the client
    /// rejoins with a fresh participant list (SDD §13.2).
    pub presence_broadcast_capacity: usize,
    /// Presence lease TTL in seconds (LIVE_EDIT_PRESENCE_LEASE_TTL_SECS).
    /// Default: 60 (b.w29 — was 30 pre-fix).
    ///
    /// Each JoinSession's register sets `le:{sid}:presence:{cid}:lease` with
    /// this TTL, and the connection's lease keeper renews it every
    /// `presence_refresh_interval_secs` while the connection is still a
    /// member. The presence sweeper treats absence of the lease key as proof
    /// that the connection is dead and reaps the membership row.
    ///
    /// The same TTL is the lifetime of the teardown marker
    /// (`le:{sid}:presence:{cid}:gone`) a connection's teardown arms, and the
    /// register's deadline: a register is refused if it reaches the store more
    /// than this many seconds, on the store's clock, after the joining
    /// connection read that clock just before registering.
    ///
    /// Tuning: TTL must be strictly greater than
    /// `presence_refresh_interval_secs` plus the worst plausible tokio
    /// scheduling pause, or a healthy session whose refresh tick was
    /// delayed will be misclassified as dead. Default ratio is
    /// `60 / 15 = 4×` so up to ~45 s of tokio jitter is tolerated before a
    /// live session is wrongly evicted. Raising the TTL further trades
    /// faster crash-recovery for slack. Zero is rejected at parse time —
    /// `SET ... EX 0` is a syntax error in Valkey.
    pub presence_lease_ttl_secs: u64,
    /// Presence lease refresh interval in seconds
    /// (LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS). Default: 15 (b.w29 —
    /// was 10 pre-fix).
    ///
    /// Each JoinSession connection's lease keeper, running from registration
    /// to teardown, renews the lease every `presence_refresh_interval_secs`.
    /// The renewal itself writes only while the connection is still a member
    /// with its info hash, and never re-creates a lease. If the connection
    /// was evicted (or lost its info hash) the keeper instead re-registers it,
    /// recreating lease, info hash and membership, until the connection's
    /// teardown begins. Must be strictly less
    /// than `presence_lease_ttl_secs` so the lease never expires under
    /// nominal scheduling; the constructor enforces a `refresh × 2 ≤ ttl`
    /// rule of thumb so a missed tick still has one more chance before the
    /// lease lapses. Zero is rejected at parse time — refreshes would
    /// queue back-to-back and saturate the data pool.
    pub presence_refresh_interval_secs: u64,
    /// `JoinSession` client keepalive interval in milliseconds
    /// (LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS). Default: 15000
    /// ([`crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS`]).
    ///
    /// A join whose header set `client_keepalive` is told this interval in
    /// `SessionConnected.keepalive_interval_ms`, and the join's replica ends
    /// it `UNAVAILABLE CLIENT_KEEPALIVE_TIMEOUT` once its request stream has
    /// carried no data for three intervals (45 s at the default), counted
    /// from the end of the membership replay. A join that did not opt in is
    /// never ended for silence.
    ///
    /// `0` turns the keepalive off: every join is told 0 and none is ended
    /// for silence, which is the server's pre-keepalive behaviour on the
    /// wire. It is the rollback if live clients are ever ended falsely.
    /// A non-zero value below
    /// [`crate::presence::MIN_PRESENCE_KEEPALIVE_INTERVAL_MS`] (1000) is
    /// rejected at parse time: it is what a value meant as seconds produces,
    /// and would end every opted-in join after milliseconds of silence.
    /// Not cross-checked against the presence lease: the deadline acts in
    /// process on the join's own replica and ends through the join's normal
    /// teardown, so it composes with the lease and the sweeper independently.
    pub presence_keepalive_interval_ms: u64,
    /// Graceful shutdown deadline in seconds (LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS). Default: 5.
    ///
    /// Applies to (a) the HTTP/2 drain window after flipping the readiness
    /// probe to NOT_SERVING — live streams keep serving through it, and start
    /// ending `UNAVAILABLE`, paced, just after it, behind the server's GOAWAY
    /// ([`Self::shutdown_stream_end_rate`]) — and (b) the per-task join budget
    /// for each background task, and for the JoinSession and
    /// SubscribeReadLayer task drains, during the shutdown sequence.
    pub shutdown_timeout_secs: u64,
    /// Overall upper bound on the post-gRPC-server shutdown sequence in
    /// seconds (LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS). Default: 30.
    ///
    /// The shutdown sequence after the gRPC server stops (layer-cleanup
    /// wait + streaming-RPC and session-checkpoint drains + background-task
    /// joins + Valkey `quit()`) runs under one `tokio::time::timeout` sized
    /// to this value; the metrics-server abort and the final telemetry flush
    /// run after it, outside this budget. Must fit inside the Helm chart's
    /// `terminationGracePeriodSeconds` (80 s by default) together with the
    /// drain window (`shutdown_timeout_secs`), the bound on the server after
    /// GOAWAY and the final flush; the startup log line reports the minimum
    /// grace this configuration needs
    /// (`grpc::transport::minimum_termination_grace`). See b.3z7 C-2.
    pub shutdown_overall_budget_secs: u64,
    /// Graceful shutdown's stream endings per second, per replica, while the
    /// ending window is under its cap (LIVE_EDIT_SHUTDOWN_STREAM_END_RATE).
    /// Default: 25. `0` ends every live stream at once.
    ///
    /// After the drain, just behind the server's GOAWAY, the live
    /// `SubscribeReadLayer` and `JoinSession` streams are ended in random
    /// order over a window of at most
    /// [`Self::shutdown_stream_end_max_window_secs`], at max(this rate,
    /// streams / that cap) per second: this rate is not a ceiling, because
    /// above rate × cap streams the cap governs and the rate rises with the
    /// stream count. Their reconnects reach the other replicas as a ramp
    /// rather than a step (`grpc::graceful_shutdown::StreamPacing`).
    pub shutdown_stream_end_rate: u32,
    /// The cap on graceful shutdown's stream-ending window, in seconds
    /// (LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS). Default: 20. `0` ends
    /// every live stream at once.
    ///
    /// The window is `streams / shutdown_stream_end_rate`, capped here. The
    /// longer of it and `LIVE_EDIT_RPC_DEADLINE_MS` bounds how long the
    /// server is waited for after GOAWAY, so it counts toward the minimum
    /// termination grace.
    pub shutdown_stream_end_max_window_secs: u32,
    /// Tracing log level filter (LIVE_EDIT_LOG_LEVEL). Default: "info".
    pub log_level: String,
    /// Output format for structured tracing (LIVE_EDIT_LOG_FORMAT). Default: `json`.
    ///
    /// `json` is the production default — log aggregators parse structured
    /// records. `pretty` emits human-readable multi-line output, intended for
    /// local development. Case-insensitive. See b.3z7 H-obs-4.
    pub log_format: LogFormat,
    /// Filter for OpenTelemetry trace export (LIVE_EDIT_TRACE_FILTER), in
    /// `EnvFilter` directive syntax. Default: [`DEFAULT_TRACE_FILTER`].
    ///
    /// Independent of `log_level` and `RUST_LOG`: it decides which spans and
    /// events reach the trace exporter, never which lines are logged. Read
    /// only when `OTEL_EXPORTER_OTLP_ENDPOINT` enables trace export, but
    /// validated on every start (see [`parse_trace_filter`]).
    pub trace_filter: String,
    /// Presence sweeper interval in seconds (LIVE_EDIT_SWEEPER_INTERVAL_SECS).
    /// Default: 15 (`presence::sweeper::DEFAULT_SWEEPER_INTERVAL_SECS`).
    ///
    /// How often each replica tries to take the presence-sweeper leader lock
    /// and, if it wins, runs one pass over the session index removing
    /// connections whose lease has expired. A lower value removes a crashed
    /// replica's connections sooner, at the cost of more passes (more Valkey
    /// traffic); a higher value lowers that traffic and lets stale
    /// connections linger longer. A pass that overruns the interval delays the
    /// next tick rather than triggering back-to-back passes. Zero is rejected
    /// at parse time — `tokio::time::interval` panics on a zero period.
    pub sweeper_interval_secs: u64,
    /// ListSessions deadline in ms (LIVE_EDIT_SESSION_LIST_TIMEOUT_MS). Default: 2000.
    pub session_list_timeout_ms: u64,
    /// Default RPC deadline in ms (LIVE_EDIT_RPC_DEADLINE_MS). Default: 30000.
    pub rpc_deadline_ms: u64,
    /// Value for the `service` Prometheus label (LIVE_EDIT_SERVICE_LABEL → HOSTNAME → "unknown").
    pub service_label: String,
    /// The backing store's TOTAL memory, in bytes, as the provider states it
    /// — for Azure Managed Redis the SKU's size, e.g. `12000000000` for
    /// `Balanced_B10`; never per shard (LIVE_EDIT_VALKEY_MEMORY_BYTES, b.yuo,
    /// b.qhk). Default: unset.
    ///
    /// Set it only on a store that does not report its own memory limit — in
    /// practice Azure Managed Redis, whose `INFO memory` has no `maxmemory`.
    /// Self-hosted Valkey and Amazon MemoryDB / ElastiCache report
    /// `maxmemory` on every primary, and a store-reported `maxmemory` wins
    /// over this setting.
    ///
    /// Read for two things, both resolved in `storage::store_memory` and
    /// re-read every 60 s: the external-value release budget
    /// ([`Self::ext_release_max_bytes`]) when that is unset and the store
    /// reports no usable `maxmemory` — one shard's share of 80 % of this
    /// value, by the smallest slot range of the cluster map, ÷ 8
    /// (`storage::store_memory::memory_bytes_budget`) — and
    /// `live_edit_store_memory_used_ratio` on such a store. Values below
    /// [`VALKEY_MEMORY_BYTES_MIN`] (100 MB) are rejected at parse time: no
    /// supported store is that small, so such a value is a units slip (`12`
    /// meant as 12 GB) that would shrink the budget to almost nothing.
    pub valkey_memory_bytes: Option<u64>,
    /// The external-value release budget `H`, in bytes
    /// (LIVE_EDIT_EXT_RELEASE_MAX_BYTES, b.yuo), when set explicitly; `None`
    /// takes the derived default. Read through
    /// [`Self::ext_release_budget`].
    ///
    /// The most bytes of released external values the backing store holds
    /// awaiting garbage collection: `CreateLayer` / `UpdateLayer` uploads that
    /// carry external values wait while the release ledger's total is at or
    /// above it, up to what is left of the call's own deadline (capped at
    /// [`Self::rpc_deadline_ms`]) less a fixed margin
    /// (`storage::ext_release_gate::EXT_RELEASE_GATE_DEADLINE_MARGIN`) — not
    /// `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS` — and are then refused
    /// `UNAVAILABLE` with nothing applied; the GC loop collects oldest
    /// first while the total is at or above half of it. **Unset:** the
    /// smallest `maxmemory` the store's primaries report in `INFO memory`,
    /// ÷ 8, when every primary reports one above zero; else, when
    /// [`Self::valkey_memory_bytes`] is set, one shard's share of it ÷ 8
    /// (`storage::store_memory::memory_bytes_budget`); else
    /// `min(8 × LIVE_EDIT_MAX_LAYER_SIZE, 100 MB)` with a startup line naming
    /// why (INFO when the store does not report its size, WARN for a fault).
    /// Re-read every 60 s (`storage::store_memory`). **`0` disables** both the
    /// upload gate and the pressure trigger.
    pub ext_release_max_bytes: Option<u64>,
    /// External-value `ref_count` reconciler interval in seconds
    /// (LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS). `0` disables the pass (b.yda),
    /// and with it the only repair of the leak-direction `ref_count`
    /// residuals it drains — among them the external-value releases of an
    /// `UpdateLayer` whose reply was lost and which the call could not
    /// recover from its own commit release entry (SDD §7.5.7): an entry
    /// evicted from the object's ring by later releasing commits, a call
    /// dropped (deadline, disconnect) before its lookup, a call ending in a
    /// way that runs no lookup, or a lookup whose read failed. With the pass
    /// off, those leaked references are never repaired.
    ///
    /// **Default: `max(3600, the derived grace below)`** — 3600 at every
    /// `LIVE_EDIT_RPC_DEADLINE_MS` up to 360000, and the grace above that. The
    /// default is derived rather than flat because a flat 3600 would fail this
    /// field's own cross-validation on a deployment that raised the RPC deadline
    /// past 360 s, refusing to boot over a variable the operator never set. An
    /// explicitly-set value is taken verbatim and then validated.
    ///
    /// Its own interval rather than a step of the GC loop: the pass walks the
    /// whole object and blob keyspace, which is far too heavy for that
    /// cadence, and the residuals it drains accumulate over hours, not seconds.
    /// It shares the `le:gc:leader` lease with the GC and orphan sweeps and the
    /// external-value backup repair — independent cadences, one lease, the
    /// SDD §22.7.2 shape.
    ///
    /// The pass's **grace** — the window a surplus must survive unchanged before
    /// it may be repaired — is derived, never configured:
    /// `storage::ext_reconcile::ext_reconcile_grace`. It is the safety guard,
    /// not a tuning knob, so it is not an operator's to shorten. This interval
    /// is cross-validated `>=` that grace at startup: a shorter one could never
    /// claim a candidate it registered.
    pub ext_reconcile_interval_secs: u64,
    /// Max attempts for auth service calls (LIVE_EDIT_AUTH_MAX_ATTEMPTS). Default: 3.
    pub auth_max_attempts: u32,
    /// Initial retry delay in ms for auth service calls (LIVE_EDIT_AUTH_RETRY_INITIAL_DELAY_MS). Default: 100.
    pub auth_retry_initial_delay_ms: u64,
    /// Whether authentication (authn) is enabled (LIVE_EDIT_AUTHN_ENABLED). Default: true.
    /// Disabling authentication is not supported; setting this to false causes a startup error.
    pub authn_enabled: bool,
    /// Whether authorization (authz) is enabled (LIVE_EDIT_AUTHZ_ENABLED). Default: true.
    /// When false, all permission checks are bypassed and `storage_service_endpoint` is not required.
    pub authz_enabled: bool,
    /// Whether the userinfo service is consulted to resolve the bearer
    /// token to a user identity (LIVE_EDIT_USERINFO_SERVICE_ENABLED).
    /// Default: false.
    ///
    /// When `true`, `userinfo_endpoint` is required at startup and the
    /// userinfo service resolves the bearer to a user identity. It does NOT
    /// by itself validate the bearer's signature or expiry: the token is
    /// validated on that path only when a validating proxy fronts the
    /// userinfo endpoint. The validating backend is the permissions service
    /// (`authz_enabled`). When
    /// `false`, the service runs without the userinfo dependency; in that
    /// configuration `authz_enabled` must be `true` so that the
    /// permissions service stays in the request path to validate the
    /// token. `Config::from_env` enforces this cross-validation: the
    /// `userinfo_service_enabled=false && authz_enabled=false`
    /// combination is rejected at startup because no backend would then
    /// validate the token — unless
    /// [`Self::test_only_dangerously_allow_unvalidated_tokens`] is set.
    pub userinfo_service_enabled: bool,
    /// Test-only escape hatch
    /// (`LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS`).
    /// Default: false.
    ///
    /// When `true`, the `userinfo_service_enabled=false &&
    /// authz_enabled=false` startup rejection is skipped and the service
    /// boots with **no backend validating the bearer token at all**: the
    /// in-process JWT client only base64-decodes the payload for a display
    /// identity (it checks no signature, issuer, audience, or expiry), and
    /// the no-op permissions client grants everything. Any well-formed
    /// three-part JWT-shaped string is accepted as a genuine caller.
    ///
    /// Exists so the standalone conformance suite can drive the public wire
    /// against a service with neither auth dependency deployed. It is
    /// deliberately not wired into the Helm chart, not documented in the
    /// operator configuration table, and not mentioned in the rejection
    /// message operators see. `main.rs` logs an ERROR-level line at startup
    /// whenever it takes effect (ERROR so that `LIVE_EDIT_LOG_LEVEL=error`
    /// cannot suppress it). It has no effect when at least one of the two
    /// auth backends is enabled.
    pub test_only_dangerously_allow_unvalidated_tokens: bool,
    /// Health-supervisor probe interval in ms
    /// (LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS). Default: 5000 (5 s).
    ///
    /// How often the supervisor pings Valkey. Lower values shorten failover
    /// detection but cost more Valkey traffic. Zero is rejected at startup
    /// because it would run the probe loop in a tight spin.
    pub health_probe_interval_ms: u64,
    /// Per-probe deadline in ms
    /// (LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS). Default: 2000 (2 s).
    ///
    /// A probe that doesn't return inside this window counts as a failure.
    /// Must be strictly less than `health_probe_interval_ms` so a slow probe
    /// can't queue past the next tick; the constructor enforces this. Zero
    /// is rejected.
    pub health_probe_timeout_ms: u64,
    /// Protocol used by the external auth clients (LIVE_EDIT_AUTH_PROTOCOL).
    /// Accepts `rest` or `grpc` case-insensitively. Default: `rest`.
    pub auth_protocol: AuthProtocol,
    /// Maximum attempts `update_layer` will make on a WATCH/MULTI/EXEC conflict
    /// before giving up; the exhausted call reaches the client as gRPC
    /// `UNAVAILABLE` (retriable) (LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS).
    /// Default: 128.
    ///
    /// The value counts the total attempts (first try + retries). Each retry
    /// sleeps a full-jittered backoff before re-acquiring an exclusive txn
    /// connection and restarting the WATCH/HGETALL/snapshot/MULTI/EXEC
    /// sequence with freshly loaded state. Raised 32 → 128 in b.3gx to
    /// pair with the per-object admission semaphore (see
    /// `update_layer_per_object_concurrency`); at `K=16` concurrent writers
    /// per object, 128 attempts gives `(15/16)^128 ≈ 0.028%` exhaustion
    /// probability — still effectively zero. See b.vi2 (retry loop) and
    /// b.3gx (admission + raised cap).
    pub update_layer_retry_max_attempts: u32,
    /// Per-object in-process admission cap for `update_layer`
    /// (LIVE_EDIT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY). Default: 16.
    ///
    /// Caps the number of concurrent WATCH/MULTI/EXEC attempts allowed
    /// against any single `object_id`. Waiters queue on an in-process
    /// `tokio::sync::Semaphore` held BEFORE the exclusive txn-pool slot is
    /// acquired, so they cannot starve txn-pool connections while blocked.
    /// Paired with the raised retry budget (see
    /// `update_layer_retry_max_attempts`) to bound the WATCH-contention
    /// tail that dominates `update_layer` failure rate at N ≥ 50 writers
    /// on one layer. Zero is rejected at startup. See b.3gx.
    pub update_layer_per_object_concurrency: u32,
    /// Backend used by `update_layer` for committing mutations
    /// (LIVE_EDIT_UPDATE_LAYER_BACKEND). Default: `lua`.
    ///
    /// `lua` uses the atomic server-side Lua EVAL for the field/time-sample
    /// fast path (b.67u), admits structural and reset deltas one at a time
    /// per object per replica, and commits them on WATCH/MULTI/EXEC.
    /// `lua-watch-structural` keeps the fast path and sends structural and
    /// reset deltas down WATCH/MULTI/EXEC under the plain per-object writer
    /// admission. `txn` forces every `update_layer` call down the legacy
    /// WATCH/MULTI/EXEC path — used for parity testing against the Lua
    /// implementation and as an emergency rollback knob. See
    /// [`UpdateLayerBackend`].
    pub update_layer_backend: UpdateLayerBackend,
    /// What the upload content check does on a failing external value
    /// (LIVE_EDIT_EXT_VALUE_VERIFY). Default: `enforce`. See
    /// [`ExtValueVerifyMode`].
    pub ext_value_verify: ExtValueVerifyMode,
    /// Late-join / catch-up snapshot switch (LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED).
    /// Default: `true` (on).
    ///
    /// `true` (default) routes late-join / rejoin catch-up through the faithful
    /// builders: a full-state snapshot for a brand-new subscriber (SDD §12.9)
    /// and one coalesced diff reconciled in place for a reconnector (SDD
    /// §12.10). `false` selects the legacy builders (SDD §12.3). The routing
    /// that acts on this flag lives in the subscribe handler.
    pub catchup_snapshot_enabled: bool,
    /// Item ceiling for one atomic object-snapshot read
    /// (LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS; Spec SDD SR-1.9). Default:
    /// [`SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT`] = 37,554.
    ///
    /// One unit for both of the read's item checks: the early-out's upper bound
    /// from the object's cardinalities, and the running item tally. Either
    /// ceiling tripping answers `TOO_LARGE`, and the consumer takes its
    /// above-ceiling path. It also sets the bounded fan-out's command window
    /// ([`Self::snapshot_fanout_window`], SR-11.3), up to the recorded cap
    /// [`FANOUT_WINDOW_MAX`]. The default is derived from
    /// the measured read cost on AMR (see the constant); raise it only with your
    /// own measurement of a faster store. Zero is rejected at startup.
    pub snapshot_atomic_read_max_items: u64,
    /// Byte ceiling for one atomic object-snapshot read
    /// (LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES; Spec SDD SR-1.9). Default:
    /// [`SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT`] = 604,619.
    ///
    /// Checked against the read's running byte tally; it is what bounds the
    /// reply size, the script's Lua memory and in-flight read memory. It is also
    /// the bounded fan-out's byte window
    /// ([`Self::snapshot_fanout_window_bytes`], SR-11.3). Zero is rejected at
    /// startup.
    pub snapshot_atomic_read_max_bytes: u64,
    /// Memory budget, in MiB, of the per-replica decoded-state cache that
    /// structural `UpdateLayer` planning runs on (structural-edits spec
    /// SR-3.3): derived at startup from the pod memory limit, 5 % of it, never
    /// configured ([`PodMemoryPlan::structural_cache_budget_mib`]); 256 at the
    /// chart's 5 GiB ([`STRUCTURAL_CACHE_MAX_MIB_DEFAULT`]).
    /// `LIVE_EDIT_STRUCTURAL_CACHE_MAX_MIB`, which set it before, is refused
    /// at startup; the cache stays empty under the
    /// `lua-watch-structural` update backend.
    ///
    /// Bounds the estimated weight of the cached states; one object's state
    /// heavier than a quarter of it, or than `u32::MAX` bytes (about 4 GiB,
    /// the cap from a 16 GiB budget up), is never cached. Each cached state
    /// also holds plan indexes, so the cache's resident weight is bounded by
    /// (1 + φ) times the budget, φ =
    /// `storage::structural_cache::PLAN_INDEX_BUDGET_CENTI` / 100, whatever
    /// an entry's indexes weigh; an entry is never made uncacheable for its
    /// indexes, and a heavier-indexed one has the cache evict others sooner
    /// (on every measured layer shape but tiny ones the indexes weigh at most
    /// φ of the state; a four-section rig's weigh 0.24 of it). One whose
    /// indexes have grown it past that whole resident budget (or past
    /// 4 × `u32::MAX` bytes, the lower figure from a budget of about 13.2 GiB
    /// up) is evicted when it is stored, and is loaded again, its indexes
    /// rebuilt, at its next structural edit. `0` disables the cache (only a
    /// `Config` built field by field can hold it). In bytes:
    /// [`Self::structural_cache_budget_bytes`].
    ///
    /// Copied from [`Self::pod_memory`] at load (`with_pod_memory_plan`); the
    /// plan is the authority.
    pub structural_cache_max_mib: u64,
    /// Probability that a committed structural call starts a sampled audit of
    /// its cache entry (LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE; structural-edits
    /// spec SR-3.6). Default: [`STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT`] = 0.001.
    ///
    /// The audit is a metric-only correctness check; at most one runs per
    /// replica at a time. `0` never audits; `1` audits every committed
    /// structural call that finds no audit running. A value outside `[0, 1]`,
    /// or not finite, is rejected at startup.
    pub structural_cache_audit_rate: f64,
    /// Whether the sampled audit may read an object above the plain
    /// atomic-read ceilings, with the fenced read
    /// (LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING; structural-edits spec
    /// SR-3.6). Default: `true`, since the cold start caches such objects
    /// with the same read. The audit's fenced read pauses the layer's
    /// colliding writers for about one polling step (project SDD §21.2), so
    /// it is paced to one per replica per
    /// `STRUCTURAL_CACHE_AUDIT_FENCED_MIN_INTERVAL` (10 minutes), never waits
    /// for capacity (it skips), and charges its reply to the catch-up memory
    /// budget (`storage::structural_cache_audit`). `false` counts such an
    /// audit `skipped`, leaving the cache's largest entries unchecked.
    pub structural_cache_audit_above_ceiling: bool,
    /// Maximum attempts `create_session` will make on a WATCH/MULTI/EXEC
    /// conflict (or connection-drop-at-EXEC) before giving up; the exhausted
    /// call reaches the client as gRPC `UNAVAILABLE` (retriable)
    /// (LIVE_EDIT_SESSION_RETRY_MAX_ATTEMPTS). Default: 32.
    ///
    /// Counts the total attempts (first try + retries). Each retry sleeps a
    /// full-jittered backoff (see `session_retry_initial_delay_ms`) before
    /// re-acquiring an exclusive txn connection and restarting the
    /// WATCH/SMEMBERS/MULTI/EXEC sequence. Shares the `retry_aborted_with_jitter`
    /// helper with `update_layer`. Zero is rejected at startup. See b.91j
    /// H-stor-3.
    pub session_retry_max_attempts: u32,
    /// Starting full-jitter backoff window in ms for `create_session` retries
    /// (LIVE_EDIT_SESSION_RETRY_INITIAL_DELAY_MS). Default: 1.
    ///
    /// Matches `update_layer`'s base delay. The actual sleep before retry N is
    /// drawn uniformly from `[0, min(2^(N-1) * base, cap)]`. See b.91j H-stor-3.
    pub session_retry_initial_delay_ms: u64,
    /// Whether the periodic checkpoint loop is enabled
    /// (LIVE_EDIT_CHECKPOINT_ENABLED). Default: `false`.
    ///
    /// Independent of `checkpoint_adapter` — selecting an adapter without
    /// flipping this flag leaves the loop dormant. The flag exists so an
    /// operator can stand up the adapter configuration in a deployment
    /// manifest and toggle the loop on/off without re-rolling the pod.
    pub checkpoint_enabled: bool,
    /// Checkpoint-store adapter selector (LIVE_EDIT_CHECKPOINT_ADAPTER).
    /// Default: `CheckpointAdapter::Disabled`.
    ///
    /// When set to `LocalDisk`, `LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH` is
    /// required to be non-empty (after trim). When set to `StorageApi`,
    /// both `LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT` (with an explicit
    /// `https://` or `http://` scheme) and `LIVE_EDIT_CHECKPOINT_BASE_PATH`
    /// are required at startup — see Epic `t1.fgj.jd` and SDD SR-4.5 /
    /// SR-4.6.
    pub checkpoint_adapter: CheckpointAdapter,
    /// Per-layer checkpoint cadence in seconds — the upper bound on the gap
    /// between two checkpoints of the same layer, measured from when the
    /// last one landed (LIVE_EDIT_CHECKPOINT_INTERVAL_SECS). Default: 60.
    /// A layer under continuous write is still checkpointed on this cadence.
    ///
    /// Must be strictly greater than `checkpoint_tick_secs` so the periodic
    /// loop can complete one tick before the next interval fires; enforced
    /// by `validate_checkpoint_budgets`.
    pub checkpoint_interval_secs: u64,
    /// Per-object quiesce window in seconds — once writes to an object have
    /// been idle this long, the loop checkpoints it early rather than
    /// waiting out the rest of `checkpoint_interval_secs`
    /// (LIVE_EDIT_CHECKPOINT_QUIESCE_SECS). Default: 30. Zero is rejected.
    /// OR-gated with the interval bound, not a precondition for it — see
    /// SDD §22.7 step 5.
    pub checkpoint_quiesce_secs: u64,
    /// Tick cadence in seconds for the checkpoint loop's quiesce-watch
    /// poll (LIVE_EDIT_CHECKPOINT_TICK_SECS). Default: 10. Zero is rejected
    /// and the value must be strictly less than `checkpoint_interval_secs`.
    pub checkpoint_tick_secs: u64,
    /// Number of historical checkpoint generations the GC retains per
    /// object after a successful new write
    /// (LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS). Default: 2.
    ///
    /// Zero is rejected at startup because retention of 0 would prune
    /// every just-written checkpoint and leave the object with no
    /// recoverable blob.
    pub checkpoint_retain_generations: u32,
    /// Safety margin added to the restore floor when reseeding
    /// `sequence_num` on a checkpoint-restored object
    /// (LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN). Default: 1000. Zero is
    /// rejected.
    ///
    /// `apply_checkpoint` seeds
    /// `seeded_seq = max(snap_seq, max_seq_observed) + margin`. The margin
    /// is **additive above** the highest sequence number the restore can
    /// prove was issued, not an alternative floor: commits that landed
    /// between the last checkpoint and the crash are lost with the live
    /// state, so their sequence numbers are unprovable, and resuming at
    /// exactly the provable floor would re-issue them against different
    /// content while pre-restore subscribers still hold the originals.
    /// The margin is the guard band covering that RPO window; zero means
    /// no guard band at all, which is why it is rejected at startup.
    pub checkpoint_seq_safety_margin: u64,
    /// Target interval in seconds between counters-sync ticks of the
    /// checkpoint leader loop (LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS),
    /// each of which writes the counters-sidecar JSON blob only when the
    /// live `le:counter:object_id` is above the stored sidecar (SDD SR-5.1,
    /// SR-2.4). Default: 60. Zero is rejected: it would put the sidecar
    /// writer into a tight loop and saturate the maintenance pool.
    ///
    /// This is a target, not a bound on how far the sidecar's
    /// `max_object_id_observed` lags the live counter. The counters tick
    /// shares one task with the per-layer checkpoint tick, so a long
    /// per-layer tick delays it; a failing checkpoint store stops its writes
    /// until the store recovers; and a write already in flight can land late
    /// and leave an earlier reading in place until the next tick.
    ///
    /// For config validation only: SDD SR-5.1 intentionally does NOT impose
    /// a cross-validation against `checkpoint_interval_secs` or
    /// `checkpoint_tick_secs`, so this knob is not checked against the
    /// quiesce/checkpoint cadence enforced by `validate_checkpoint_budgets`.
    pub checkpoint_counters_sync_secs: u64,
    /// Cadence in seconds of the checkpoint-store **reconciliation** pass —
    /// the sweep that reclaims orphaned partial-publish artifacts and collects
    /// tombstoned ext blobs (LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS).
    /// Default: `max(3600, derived reconciliation grace)`. Zero is rejected.
    ///
    /// The default is *derived*, not flat, because the bound this knob is
    /// validated against depends on a different variable
    /// (`LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS`, via the StorageApi adapter's
    /// publish arithmetic). A flat 3600 would refuse to boot once that timeout
    /// pushes the grace past an hour — on a variable the operator never set.
    ///
    /// Runs on the same `le:checkpoint:leader` lock as the per-layer and
    /// counters ticks, on its own timer. Hourly by default because the pass is
    /// pure garbage reclamation on a cold path: its input is a full enumeration
    /// of the adapter's `layers/` namespace plus one live-set read, both of
    /// which are LIST-shaped and cost real money on a blob backend, and nothing
    /// downstream depends on its latency. Raising the cadence buys faster
    /// reclamation of storage that is already paid for; lowering it multiplies
    /// LIST traffic.
    ///
    /// Cross-validated by `validate_checkpoint_budgets` to be **at least the
    /// derived reconciliation grace** (`3 ×` the adapter's own
    /// `publish_worst_case`, floored). Below the grace, every pass would find
    /// its candidates still inside their grace window and defer them forever —
    /// the sweep would burn its LIST budget and reclaim nothing, silently. The
    /// grace itself is deliberately NOT configurable: it is a property of the
    /// adapter's retry arithmetic, and an operator who set it too low would get
    /// half-published generations reaped out from under their own writer with
    /// no signal.
    pub checkpoint_reconcile_interval_secs: u64,
    /// Safety margin added to the counters-sidecar's
    /// `max_object_id_observed` value before reseeding
    /// `le:counter:object_id` at service startup
    /// (LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN). Default:
    /// [`CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT`] (1,000,000). Zero is
    /// rejected, and so is anything above
    /// [`CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX`] (10^9; its doc gives the
    /// reasoning).
    ///
    /// Defeats the whole AOF-off recovery feature when set to 0 — a fresh
    /// `CreateLayer` issued immediately after restart could hand out an
    /// `object_id` already referenced by a still-live checkpoint blob,
    /// silently overwriting persisted state (SDD SR-5.3). The margin
    /// covers the ids allocated after the most recent sidecar write that
    /// completed in Valkey before the loss but were never mirrored to the
    /// sidecar. There is one object-id counter per deployment and every
    /// `CreateLayer` allocates from it, so the margin must exceed the
    /// deployment's `CreateLayer` count over the sidecar's lag (SDD §22.7.1
    /// holds the windows). A late sidecar write's worst window is the
    /// counters cadence plus up to three adapter calls, about 600 s, which
    /// 1,000,000 covers up to about 1,666 creations a second. A counters tick
    /// held up behind one layer's assembly in the per-layer pass (up to
    /// 540 s) stretches that window to about 1,140 s, covered up to about 877
    /// a second, and a pass over many due layers can run longer, because the
    /// lease is refreshed between layers. The earlier 1000 covered under 2 a
    /// second over the 600 s window. While the checkpoint store is
    /// down the sidecar stops advancing, so the lag grows with the outage
    /// and no margin covers an unbounded one.
    ///
    /// The cost of a large margin is id-space only: the startup seed runs on
    /// every pod start and raises the counter to `sidecar + margin` when it is
    /// below that, so object ids can jump by up to the margin per pod start.
    pub checkpoint_counter_safety_margin: u64,
    /// Master opt-in for stream-replay tail capture
    /// (LIVE_EDIT_CHECKPOINT_TAIL_ENABLED). Default: `true`.
    ///
    /// Shipped as an explicit opt-in at introduction (SDD SR-8.1) so the
    /// feature could run dark and be enabled per-deployment; the default was
    /// promoted to `true` on 2026-09-11 (b.3n1) after the dev soak. The knob
    /// stays — setting it `false` restores the original dark behaviour, with
    /// every pre-SR-8.1 path unchanged. Parsed with the plain stdlib
    /// `bool::from_str` like every other bool knob here, so only the exact
    /// strings `true` / `false` are accepted.
    pub checkpoint_tail_enabled: bool,
    /// Maximum number of tail generations that may chain onto one base before
    /// the capture path refuses (K) (LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN).
    /// Default: 8. Zero is rejected.
    ///
    /// Bounds restore cost: a restore composes the base plus every tail in the
    /// chain, so K caps how many artifacts one recovery must fetch and apply.
    /// Zero is rejected because a chain bound of zero admits no tail at all —
    /// every eligible tick would refuse and the feature would be silently inert
    /// while reporting itself enabled.
    pub checkpoint_tail_max_chain: u32,
    /// Maximum number of stream entries one captured tail may carry
    /// (LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES). Default: 4096. Zero is
    /// rejected, as is any value above `2^53 - 1`.
    ///
    /// The default is pinned to the `LIVE_EDIT_LAYER_STREAM_MAXLEN` default
    /// (also 4096) deliberately, so `warn_tail_entries_exceed_stream_retention`
    /// is silent at stock defaults: a tail can never be longer than stream
    /// retention allows, and matching the two keeps the shipped pairing
    /// self-consistent. Retention can hold fewer entries than the entry cap —
    /// the relay byte budget (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`) trims too —
    /// and the byte pairing has its own WARN,
    /// `warn_tail_bytes_exceed_stream_retention`.
    ///
    /// The ceiling exists because a tail's entry count is bounded by the stream
    /// sequence span it covers, and that span is already Lua-f64-bounded at
    /// `2^53 - 1` by both commit paths — a cap above it can never bind.
    pub checkpoint_tail_max_entries: u64,
    /// Maximum size in bytes of one captured tail artifact
    /// (LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES). Default: 67108864 (64 MiB).
    /// Zero is rejected, as is any value below the one-record floor
    /// (`TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE + 1`).
    ///
    /// 64 MiB is pinned deliberately: the SDD's measured tail sizes are
    /// 2.6–4.9 MiB for a 4096-entry / 1 KiB-per-entry tail, so the default
    /// gives an order of magnitude of headroom while keeping one tail artifact
    /// comfortably below the 100 MB `LIVE_EDIT_MAX_LAYER_SIZE` base-blob scale
    /// — a tail must never cost more to fetch than the base it layers onto.
    ///
    /// The floor is a cross-field bound on the tail codec's own framing rather
    /// than a round number: below one tail header plus one record prefix plus a
    /// single payload byte, the capture path's headroom derivation yields a
    /// payload budget of zero, so no verbatim commit is representable and every
    /// capture would refuse observably as `refused_missing_bytes`. Rejecting at
    /// startup turns a permanent silent refusal loop into a loud boot failure.
    pub checkpoint_tail_max_bytes: usize,
    /// Filesystem path used by the `LocalDisk` checkpoint adapter
    /// (LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH). Default: `None`.
    ///
    /// Required (non-empty after trim) when `checkpoint_adapter` is
    /// `LocalDisk`; ignored otherwise. The value is stored without
    /// further filesystem validation here — the adapter constructor
    /// owns existence / permission checks at startup.
    pub checkpoint_local_disk_path: Option<String>,
    /// Endpoint URL targeted by the `StorageApi` checkpoint adapter
    /// (LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT). Default: `None`.
    /// Epic `t1.fgj.jd` / SDD SR-4.5.
    ///
    /// Required (non-empty after trim) when `checkpoint_adapter` is
    /// `StorageApi`; ignored otherwise (silent-tolerance contract — the
    /// value is parsed unconditionally so the field is always populated
    /// when set, but only validated and consumed under StorageApi). The
    /// scheme is validated at startup via the shared `validate_auth_endpoint`
    /// wrapper so bare hostnames are rejected fail-closed — the value
    /// transits a bearer-authenticated channel to the storage-api service
    /// and a silent plaintext downgrade would leak the token. Task
    /// `t2.fgj.jd.8e` consumes the parsed string into
    /// `StorageApiCheckpointStore::new`.
    pub checkpoint_storage_api_endpoint: Option<String>,
    /// Base path used by the `StorageApi` checkpoint adapter to render
    /// per-object blob locations (LIVE_EDIT_CHECKPOINT_BASE_PATH).
    /// Default: `None`. Epic `t1.fgj.jd` / SDD SR-4.5 / SR-4.6.
    ///
    /// Required (non-empty after trim) when `checkpoint_adapter` is
    /// `StorageApi`; ignored otherwise (silent-tolerance contract). The
    /// adapter renders `<base>/.live-edit/...` from this value, which is
    /// why the field is `PathBuf` rather than `String` — Task
    /// `t2.fgj.jd.8e` consumes the path for segment-rendering. Trimmed-
    /// empty → `None`, mirroring the `service_token_file` and
    /// `checkpoint_local_disk_path` idioms.
    pub checkpoint_base_path: Option<PathBuf>,
    /// Connect-phase timeout in ms for the `StorageApi` checkpoint adapter's
    /// outbound HTTP client (LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS).
    /// Default: 5000. Epic `t1.fgj.jd`.
    ///
    /// Bounds the time the adapter waits for the TCP/TLS handshake before
    /// surfacing a transient error. Zero is rejected at parse time —
    /// a zero connect timeout would fail every checkpoint call
    /// synchronously. Task `t2.fgj.jd.8e` passes the value into
    /// `StorageApiCheckpointStore::new`.
    pub checkpoint_connect_timeout_ms: u64,
    /// Per-request timeout in ms for the `StorageApi` checkpoint adapter's
    /// outbound HTTP client (LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS).
    /// Default: 30000. Epic `t1.fgj.jd` / SDD SR-4.2.
    ///
    /// Checkpoint blobs are multi-MB, so the per-request budget is
    /// intentionally longer than the auth-RPC pattern (`auth_timeout_ms`
    /// defaults to 5 s). Zero is rejected at parse time — a zero request
    /// timeout would fail every checkpoint call synchronously. Task
    /// `t2.fgj.jd.8e` passes the value into
    /// `StorageApiCheckpointStore::new`.
    pub checkpoint_request_timeout_ms: u64,
    /// Number of worker tasks that drain the external-value blob mirror
    /// queue (LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS). Default: 2, at most
    /// [`crate::checkpoint::ext_mirror::EXT_MIRROR_MAX_WORKERS`] (16).
    /// Epic `t1.fgj.zc` / SDD §22.17.
    ///
    /// The ext-value mirror pipeline is a bounded mpsc fed by
    /// `ValkeyExternalValueStore` after every committed store (so every
    /// Valkey-side blob is offered to the checkpoint store). This value sets
    /// the worker-pool fan-out that drains the queue; each worker calls
    /// `CheckpointStore::put_ext_blob` per item. Zero is rejected at parse
    /// time — zero workers would spawn no drainer tasks and silently drop
    /// every enqueue once the bounded channel filled (the
    /// `checkpoint_ext_mirror_dropped_total` counter would tick continuously
    /// with no progress). Above 16 is rejected too: the values the workers
    /// hold are inside the mirror's byte budget, but each worker's adapter
    /// connection keeps buffers that are not, so the count must stay bounded.
    pub checkpoint_ext_mirror_workers: usize,
    /// Bounded queue capacity for the external-value blob mirror channel
    /// (LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY). Default: 1000.
    /// Epic `t1.fgj.zc` / SDD §22.17.
    ///
    /// Sets the `tokio::sync::mpsc::channel` capacity bridging
    /// `ValkeyExternalValueStore` producers to the mirror worker pool. A
    /// limit on items, not on memory: the bytes the mirror holds are bounded
    /// by its byte budget, `M_ext`, derived from the pod memory limit
    /// ([`PodMemoryPlan::ext_mirror_bytes`]). Senders never wait: a full
    /// channel or a full budget surfaces as a drop — counted by
    /// `checkpoint_ext_mirror_dropped_total` by reason — rather than
    /// back-pressuring the hot write path. Zero is rejected at parse time
    /// because `tokio::sync::mpsc::channel(0)` panics.
    pub checkpoint_ext_mirror_queue_capacity: usize,
    /// Filesystem path to the sidecar-written service-identity token file
    /// (LIVE_EDIT_SERVICE_TOKEN_FILE). Default: `None`. b.fgj / SDD SR-6.1.
    ///
    /// When `Some`, `main.rs` constructs a `ServiceTokenProvider` that
    /// polls this path for a JSON-encoded `{access_token, expires_on}`
    /// blob written by a credential-provisioner sidecar (atomic-rename
    /// semantics). The cached token is consumed by the Epic 5 outbound
    /// storage-api adapter to authenticate checkpoint calls. No scheme
    /// validation runs here — it's a filesystem path, not an endpoint
    /// URL. Required when `checkpoint_adapter` is `StorageApi`: `from_env`
    /// refuses that adapter without it.
    pub service_token_file: Option<PathBuf>,
    /// Poll interval in seconds for the `ServiceTokenProvider` to re-read
    /// `service_token_file` (LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS).
    /// Default: 15. b.fgj / SDD SR-6.2.
    ///
    /// The polling task uses `tokio::time::interval` with
    /// `MissedTickBehavior::Skip`. Zero is rejected at parse time
    /// (mirrors `presence_lease_ttl_secs`) because a zero-interval
    /// `tokio::time::interval` panics and a tight-loop poll would
    /// saturate the runtime.
    pub service_token_poll_interval_secs: u64,
    /// Forward-compat signal that the operator selected the OAuth
    /// client-credentials flow for outbound service identity
    /// (LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED). Default:
    /// `false`. b.fgj / SDD SR-6.4.
    ///
    /// V1 is NOT implementing the in-process client-credentials flow —
    /// service identity comes from the credential-provisioner sidecar
    /// only (see `service_token_file`). When set to `true`,
    /// `log_service_token_startup_warnings` emits a WARN at startup so
    /// operators see the misconfiguration without the service failing
    /// closed. The flag has no other effect: outbound storage-api calls
    /// authenticate only with the token read from `service_token_file`.
    /// It is carried in `Config` so the warn-helper can branch on it; no
    /// other code path consults it in V1.
    pub service_token_client_credentials_enabled: bool,
    /// `session_tokens` crash-recovery sweep interval in seconds
    /// (LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS). Default: 60.
    ///
    /// `session_tokens` is the per-replica `DashMap<session_id, SessionCancel>`
    /// of streams this replica is currently servicing. Entries are pruned by
    /// the `le:session:deleted` pub/sub fan-out, but that only fires for
    /// explicit `DeleteSession` calls — sessions whose owning client crashes
    /// (or that are simply never deleted) would otherwise stay in the map
    /// until the process exits. The sweeper iterates the live entries, HGETs
    /// `le:{session_id}:sess` once per entry, and drops + cancels any entry
    /// whose underlying session is gone. Higher values lower Valkey load but
    /// extend the window over which a crashed client's `session_tokens`
    /// entry lingers, and over which streams outlive a deleted session whose
    /// `le:session:deleted` notice this replica lost without detecting it (a
    /// detected loss — a relay lag or a pub/sub reconnect — triggers a prompt
    /// deleted-only pass instead, within seconds). Zero is rejected at startup
    /// because a zero-interval `tokio::time::interval` panics. b.j5t item 5 /
    /// b.f36 A6.
    pub session_tokens_sweep_interval_secs: u64,
    /// The pod memory plan: the pod memory limit read at startup, every
    /// pod-memory budget derived from it and the term list they close
    /// (`config::memory_plan`). [`Self::from_env`] derives it and copies its
    /// budgets into [`Self::upload_memory_budget_mib`],
    /// [`Self::catchup_build_memory_budget_mib`],
    /// [`Self::structural_cache_max_mib`] and
    /// [`Self::subscriber_max_subscriptions`]; a `Config` built field by field
    /// carries [`PodMemoryPlan::reference`]. A budget without one of those
    /// fields is read from here (`config.pod_memory.<budget>_bytes()`).
    pub pod_memory: PodMemoryPlan,
}

/// Largest accepted `LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS`: 1e12 seconds,
/// roughly 31,700 years (b.is1).
///
/// A **representability** bound, not an operational one, and derived from the
/// `DateTime<Utc>` range rather than from `chrono::Duration`'s range. The lead
/// becomes a `chrono::Duration` at the `main.rs` wiring site and is then *added
/// to a `DateTime<Utc>`* inside the token provider — `now + stale_entry_lead`
/// (`stale_entry_lead = refresh_lead + poll_interval`) at the polling task's
/// loop head, the sole such site since b.mq8 deleted the withholding accessor
/// that also added the bare lead to `now`. That deletion does not move this
/// bound: the surviving site adds a value `>= refresh_lead` to the same
/// `DateTime<Utc>`, so the binding range is unchanged.
/// `DateTime<Utc> + TimeDelta` is `checked_add_signed(rhs).expect(..)`, and
/// `DateTime::<Utc>::MAX_UTC` is year +262143 — 8_210_298_412_799 s past the
/// epoch — so that addition, not the `chrono::Duration` conversion, is the
/// binding constraint. It is ~6 orders of magnitude tighter than
/// `chrono::Duration`'s `i64::MAX / 1_000` s (≈9.22e15) second-range, which an
/// earlier form of this gate used: that admitted leads the provider could not
/// add to `Utc::now()` at all.
///
/// Deliberately a static const rather than a `Utc::now()`-relative computation:
/// startup validation should be deterministic, and the margin is wide enough
/// that the wall clock cannot matter. The const-assert below pins the bound
/// under one eighth of the `MAX_UTC` timestamp, leaving >7.2e12 s (~228,000
/// years) of headroom for the `Utc::now()` operand.
///
/// This gate is **not** what makes the provider panic-free — the addition site
/// uses `checked_add_signed` and treats `None` as "not inside the window",
/// so panic-freedom does not depend on this file staying in sync. The gate
/// exists so an operator typo surfaces as a startup config error instead of a
/// provider whose proactive re-read window can never open, and so `main.rs`'s
/// `try_seconds(..).expect(..)` keeps a contract it can rely on.
///
/// There is still deliberately **no operational ceiling** — no "plausible token
/// lifetime" bound. The precedent in this file is that a ceiling needs a derived
/// harm (`AUTH_CACHE_MAX_ENTRIES_CEILING` bounds memory); an oversized lead has
/// none, because post-b.d6b the lead withholds no credential — it only makes the
/// token-file poller run its 250 ms cadence for longer, which is a `stat` plus a
/// small read whose log amplification is already bounded by the polling task's
/// parse-failure gate.
const VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS: u64 = 1_000_000_000_000;

const _: () = assert!(
    VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS
        <= (chrono::DateTime::<chrono::Utc>::MAX_UTC.timestamp() as u64) / 8,
    "VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS must stay far below the DateTime<Utc> range so \
     `Utc::now() + lead` is representable for any wall clock this process can observe"
);

/// The headroom `LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS` must leave above
/// `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS` (`Config::validate_shutdown_budgets`):
/// room for `ValkeyClient::quit()`'s sequential internal bounds plus slack for
/// the shutdown steps with no timeout of their own. The `const` assertion
/// after `STREAM_MUX_SHUTDOWN_GRACE` in `storage::client` sums those bounds
/// against this value, so a bound raised or added in `quit()` without raising
/// this fails the build.
pub(crate) const MIN_SHUTDOWN_BUDGET_HEADROOM_SECS: u64 = 15;

/// Default `LIVE_EDIT_MAX_LAYER_SIZE`: 104,857,600 bytes (100 MiB), the cap on
/// any one CreateLayer or UpdateLayer upload body. The same number as
/// [`crate::grpc::upload_admission::MAX_LAYER_SIZE_DEFAULT_BYTES`], from which
/// it is taken.
pub const MAX_LAYER_SIZE_DEFAULT: usize =
    crate::grpc::upload_admission::MAX_LAYER_SIZE_DEFAULT_BYTES as usize;

/// Default `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS`: 37,554 items.
///
/// The AMR operating ceiling for one atomic snapshot read (Spec SDD SR-1.9,
/// SR-2.5, SR-11.2): condition (i) solved on dev AMR at `P_a` = `P_f` = 1 against
/// `W0_shard` + `W_wave` minus the measured excess allowance, scaled by the
/// AMR/sandbox cost ratio, taking the x(15) end of the interval 37,554–39,430.
/// The chart carries no per-platform default: the slower platform's ceiling is
/// safe on a faster store, where a layer above it takes the slower above-ceiling
/// path but stays correct. Recorded in
/// `docs-internal/validation/proven-snapshot-reads-20261001/report.md`,
/// `## Recorded inputs (SR-11.8)` and `## AMR ceiling re-solve against the
/// excess allowance`; the inputs it is solved from are Epic 1's
/// (`docs-internal/validation/snapshot-read-cost-20260927/report.md` §8.2–§8.3).
pub const SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT: u64 = 37_554;

/// Default `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES`: 604,619 B.
///
/// **A reply-size and memory bound, not a time bound.** It is
/// [`SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT`] on the real-id ray:
/// floor(37,554 items × 16.1 B/item), the bytes per pre-read bound item of a
/// Kitchen_set live layer whose section ids are real, full-width random
/// 64-bit integers (19–20 digits), as both clients allocate them (Spec SDD
/// SR-1.9, SR-11.2; project SDD §21.5). It bounds an atomic read's reply
/// size, Lua memory and in-flight read memory, and the memory terms of
/// project SDD §16 and §21.2 are derived from it. The item ceiling is the
/// time bound: the script's server time scales with items, and bytes add
/// 6–9 % of it, so the byte ceiling enters no time solve.
///
/// It replaced 449,938 (37,554 × 11.981096), set on Epic 1's Kitchen_set
/// probe layer with short sequential section ids ("1".."2108"), which
/// under-stated real layers' bytes by about a third and capped them at
/// about 74 % of the item ceiling. Recorded, with the id-length evidence and
/// the memory terms, in
/// `docs-internal/validation/real-id-byte-ceiling-20261006/report.md`
/// (`derivation.json`). The raise adds no read time: real ids cost 1.03–1.05×
/// short ids per bound item (that report, §3.1), and a short-id layer already
/// reached the item ceiling under 449,938.
pub const SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT: u64 = 604_619;

/// The per-replica structural-planning cache's budget at the reference pod
/// memory limit, 256 MiB: 5 % of 5,120 MiB (structural-edits spec SR-3.3;
/// `config::memory_plan`). What a `Config` built field by field, rather than
/// derived from a limit, sets [`Config::structural_cache_max_mib`] to.
pub const STRUCTURAL_CACHE_MAX_MIB_DEFAULT: u64 = 256;

/// Default `LIVE_EDIT_LAYER_STREAM_MAX_BYTES`: 128 MiB of retained relay bytes
/// per object (`storage::relay_retention`). At it a layer still keeps the full
/// default 4096-entry window while its mean relay is at most 32 KiB, and one
/// checkpoint tail at its own 64 MiB cap, plus one generation of rounding,
/// still fits. Also the budget a `LayerStore` built without
/// `LayerStore::with_layer_stream_max_bytes` uses. `0` turns the byte budget
/// off (count-only retention).
pub const LAYER_STREAM_MAX_BYTES_DEFAULT: u64 = 134_217_728;

/// Default `LIVE_EDIT_RPC_DEADLINE_MS`: 30 s. Also the cap a `LayerStore` built
/// without `LayerStore::with_fence_wait_deadline` puts on one `update_layer`
/// call's snapshot-fence waiting (Spec SDD SR-5.4), as in a store a test
/// constructs directly, and one input of the commit release ring TTL such a
/// store uses (`storage::keys::commit_release_ring_ttl`).
pub const RPC_DEADLINE_MS_DEFAULT: u64 = 30_000;

/// Default `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`: 5 s. Also the other input of
/// the commit release ring TTL a `LayerStore` built without
/// `LayerStore::with_commit_release_ring_ttl` uses
/// (`storage::keys::commit_release_ring_ttl`).
pub const VALKEY_COMMAND_TIMEOUT_MS_DEFAULT: u64 = 5_000;

/// The structural-planning cache's budget in bytes for a budget of `max_mib`
/// MiB ([`Config::structural_cache_max_mib`]): the per-entry cap and the
/// state budget, which the cache raises by its plan-index allowance for the
/// resident weight (`storage::structural_cache::resident_budget_bytes`).
/// Saturates rather than overflowing; a derived budget, 5 % of the pod memory
/// limit, is far below that.
pub const fn structural_cache_budget_bytes(max_mib: u64) -> u64 {
    max_mib.saturating_mul(1024 * 1024)
}

/// Default `LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE`: one committed structural
/// call in a thousand starts a sampled audit of its cache entry
/// (structural-edits spec SR-3.6).
pub const STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT: f64 = 0.001;

/// Snapshot items per entry, in thousandths: 5.266 items per entry.
///
/// Kitchen_set's marked path, 26,767 items over 5,083 entries. It converts the
/// item ceiling into the fan-out window's command count (Spec SDD SR-11.3, see
/// [`snapshot_fanout_window`]); it is not a limit on its own. Held as an integer
/// ratio so the window is an exact floor rather than a floating-point one.
/// Recorded in `docs-internal/validation/proven-snapshot-reads-20261001/report.md`,
/// `## Recorded inputs (SR-11.8)`, from Epic 1's report §8.2
/// (`docs-internal/validation/snapshot-read-cost-20260927/report.md`).
pub const SNAPSHOT_ITEMS_PER_ENTRY_MILLI: u64 = 5_266;

/// `FANOUT_WINDOW_MAX`: the most commands one bounded fan-out window holds,
/// whatever the item ceiling, 2,141 (Spec SDD SR-11.3, SR-11.6 (i)). A recorded
/// derived constant, not a setting: [`snapshot_fanout_window`] caps the
/// ceiling-derived window at it.
///
/// **Why a cap.** SR-11.6 (i) bounds the queue one atomic read can wait behind
/// in a join wave: `R_design × (P_a × A(e_c) + P_f × A_window) + A(e_c)` (the
/// last term the checkpoint read) must stay within `W0_shard + W_wave −` the
/// scaled excess allowance, 501.614 − 24.823 = 476.790 ms on AMR (Epic 2's
/// re-solve). SR-5.2 wakes every waiting reader at one shared instant, so every
/// replica's fenced read of a reconnecting layer starts its first window
/// together: the worst-case alignment is the designed case. The shipped
/// 37,554-item ceiling is where (i)'s model queue, with Epic 1's per-window
/// term (ρ × `A`, from a replica of the command shape), reaches that limit
/// exactly. The implemented reader's windows cost 2–3× that term, so an
/// uncapped window at the shipped ceiling (7,131 commands) fails (i) on AMR;
/// the cap restores it without moving the ceiling, which decides which layers
/// are fenced.
///
/// **Derivation** (the Task 10 section of
/// `docs-internal/validation/fenced-snapshot-reads-20261002/report.md`; full
/// precision in that folder's `raw/task10-results/results.json`,
/// `fanout_window_cap`):
/// - the per-stream allowance at the shipped ceiling, AMR:
///   (476.790 − 11 × 39.328) ÷ 10 = 4.418 ms (`R_design` = 10, `P_a` = `P_f` =
///   1, `A(e_c)` = 39.328 ms);
/// - the conservative per-command window cost: the largest measured full
///   Kitchen-shaped window, 9.553 ms of shard time over 6,957 commands on the
///   sandbox, 1.373 µs per command, × the AMR / sandbox cost ratio 1.502 =
///   2.063 µs;
/// - 4.418 ms ÷ 2.063 µs = 2,141 commands (rounded down).
///
/// With the cap the queue is at most 11 × `A(e)` + 44.2 ms ≤ 476.8 ms at the
/// shipped ceiling and every lowered one, at SR-11.6's average entry size
/// (the measured inline-heavy window after a family change costs about 1.23×
/// the allowance, once per family transition; per-family slow-start is
/// declined); below 11,275 items the uncapped window is the smaller and
/// applies. The cost is round trips only: about 1.5 %
/// on a fenced hold at AMR's round trip. Fenced catch-up, checkpoint and
/// structural reads all run the same reader, so all take the cap; the ext-GC
/// reference walk (`storage::retirement`) reads at most half as many value
/// rows at once, each two wire commands, so it stays within it too.
///
/// **Re-derive it** (and re-record it) when any input moves: `P_a`, `P_f`,
/// `R_design`, the shipped item ceiling, `W_wave` or the excess allowance, or
/// the measured per-command window cost.
pub const FANOUT_WINDOW_MAX: u64 = 2_141;

/// `FANOUT_INDEX_WINDOW_MEMBERS`: the most index members one window of the
/// bounded fan-out's round 1 asks for, 26,855 (Spec SDD SR-11.3, SR-11.6 (i)).
/// A recorded derived constant, not a setting, beside [`FANOUT_WINDOW_MAX`]
/// and derived the same way.
///
/// **Why a bound.** Round 1 reads the object's four indices (sections,
/// deleted sections, fields, time samples), whose replies are round 2's
/// commands. [`FANOUT_WINDOW_MAX`] bounds a window in commands, and round 1
/// is a handful of commands whatever the layer's size, so read whole its
/// window grows with the layer: about 66–110 ns of shard time per index
/// member on the sandbox, 6.5–7.9 ms on AMR at ten times the shipped ceiling,
/// past the same per-stream allowance the command cap keeps every other
/// window under. SR-5.2 wakes every waiting replica's fenced read at one
/// instant, so round 1 is the window that lines up across replicas. So round
/// 1 reads the four member counts first (`SCARD` / `ZCARD`, beside the
/// header), then the members in pages — `ZRANGE` by rank for the sorted
/// indices, `SSCAN … COUNT` (half the page, `COUNT` being approximate) for
/// the sets — at most this many members per window, all indices together.
///
/// **Derivation** (the Task 10 section of
/// `docs-internal/validation/fenced-snapshot-reads-20261002/report.md`; full
/// precision in that folder's `raw/task10-results/results.json`,
/// `fanout_index_window_members`):
/// - the per-stream allowance at the shipped ceiling, AMR: 4.418 ms, as for
///   [`FANOUT_WINDOW_MAX`];
/// - the conservative per-member cost: the largest measured whole-index round
///   1 on a Kitchen-shaped layer, 2.360 ms of shard time over the 4× layer's
///   21,550 index members on the sandbox, 0.1095 µs per member, × the AMR /
///   sandbox cost ratio 1.502 = 0.1645 µs (inline-heavy layers' whole round 1
///   is under 1.3 ms, its cost per member set by the window's fixed part);
/// - 4.418 ms ÷ 0.1645 µs = 26,855 members (rounded down, at the record's full
///   precision).
///
/// Fenced catch-up, checkpoint and structural reads all run the same reader,
/// so all take the bound. (The ext-GC reference walk, `storage::retirement`,
/// pages its indices at its own, smaller page.) The cost is round trips only: one for the
/// counts, then about one per 26,855 members. The structural update's read
/// also bounds each page in bytes, at each index's per-member bound
/// (`storage::diff_read::FanoutReplies::Bounded`), so at the default byte
/// window its fields-index pages hold at most 4,057 members (604,619 B ÷
/// 149 B) while names are within `NAME_BYTES_SIZING_FLOOR`. A set index small
/// enough for the store's compact encoding (`set-max-intset-entries`,
/// `set-max-listpack-entries`) answers its whole membership to one `SSCAN`
/// whatever the `COUNT`, so such a page can exceed its share by up to that
/// encoding limit; a hashtable-encoded set's step can exceed it by a few
/// members, since the store adds whole bucket chains until it has `COUNT`.
///
/// **Re-derive it** (and re-record it) when any input moves: `P_a`, `P_f`,
/// `R_design`, the shipped item ceiling, `W_wave` or the excess allowance, or
/// the measured per-member round-1 cost.
pub const FANOUT_INDEX_WINDOW_MEMBERS: u64 = 26_855;

/// `τ`: the most shard time one script window of the structural WATCH read
/// may take on Azure Managed Redis, by the recorded model below, in ns:
/// 4,418,408 (SDD §10.5). A recorded derived constant, not a setting.
///
/// The structural `UpdateLayer` WATCH read (`storage::diff_read`,
/// `FanoutReplies::Bounded`) issues each window of its rounds 2 and 3 as one
/// `EVALSHA` of `scripts/fanout_window_read.lua`, which blocks its shard for
/// the whole window, until the read falls back to plain commands. τ is
/// SR-11.6 (i)'s per-stream allowance at the shipped ceiling, the same
/// allowance [`FANOUT_WINDOW_MAX`] and
/// [`FANOUT_INDEX_WINDOW_MEMBERS`] are sized from (the Task 10 section of
/// `docs-internal/validation/fenced-snapshot-reads-20261002/report.md`;
/// `raw/task10-results/results.json`, `fanout_window_cap.allowance_ms`
/// 4.418408945, rounded down to the ns), so a script window blocks its shard
/// no longer than one fenced window may.
///
/// A script window holds entries while its modelled shard time
/// ([`fanout_script_window_modelled_shard_ns`]) stays within τ, besides
/// [`FANOUT_WINDOW_MAX`]'s wire commands and the byte window. A lone entry
/// whose modelled time alone exceeds τ — a long children list — is issued as
/// one plain `LRANGE` instead.
///
/// **Re-derive it** when SR-11.6 (i)'s inputs move: `P_a`, `P_f`, `R_design`,
/// the shipped item ceiling, `W_wave` or the excess allowance.
pub const FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS: u64 = 4_418_408;

// The four coefficients of a script window's modelled shard time
// ([`fanout_script_window_modelled_shard_ns`]), in AMR ns: recorded derived
// constants, not settings.
//
// **Derivation** (`docs-internal/validation/watch-window-script-b-gvf-20261010/`,
// `results.json` `fanout_script_window`, `raw/m1-fit.json`, script
// `scripts/m1_fit.py`): 8,930 script windows of the structural WATCH read,
// each one `EVALSHA` whose shard time the store's `SLOWLOG` recorded
// (the structural cache's measurement tool, its WATCH-read mode with the
// slowlog read, release build, Valkey
// 9.0.3 on the shared sandbox), over synthetic layers of one window kind each
// — inline and external field rows, time samples, sections named
// `MAX_NAME_BYTES` long, children lists from 1 to 30,000 entries — and the
// ref-xform layers of 1,000 and 5,000 placed assets (each one prim with a
// reference and four attributes) and the whole Kitchen_set stage, flattened;
// each window's shard time × the AMR / sandbox cost ratio 1.502323. The
// coefficients are the non-negative values minimising the windows' summed
// modelled ÷ measured time with every window at or below its model (the
// upper envelope, a linear program), rounded up to whole ns. They are an
// envelope, not a decomposition of the cost: the per-entry cost of a value
// row, a section and a list entry differ more than the bound's bytes say, so
// the fixed term absorbs the slowest windows. The list term fitted to 0: the
// reply-byte term, 20 B of bound per list entry, already covers an
// in-script `LRANGE`'s growth with its length. Windows that brought names
// longer than any their read had seen (their bytes past their bound) are the
// cut's residual, left out of the envelope and recorded apart.
//
// Out of sample (`scripts/m1_check.py`, `raw/m1-check-watch-read.json`), 14
// of 6,490 windows run on these values took longer than their model, at most
// 1.747×, all inside reads slowed as a whole by contention on the shared
// host. The decider ruled (2026-10-10) that these values stand under the
// strict every-window envelope: they over-model an uncontended window's
// shard time, so they cost smaller windows and more round trips, not longer
// blocks; a fit filtered to uncontended reads was rejected. An uncontended
// re-fit does not displace them: the same strict-envelope fit over 8,930
// windows measured with the host to itself (`raw/m1-fit-clean.json`) cuts
// more script windows than these values on the ref-xform 5,000-asset layer
// and the flattened Kitchen_set (382 against 363, 523 against 496;
// `raw/m1-compare.json`), and every one of those windows these values model
// within τ ran at or under its model, at most 0.935×
// (`raw/m1-clean-vs-wired.json`).
//
// Re-derive them when the measured in-script costs change (a new store
// version, a change to `scripts/fanout_window_read.lua` or to what a window
// carries), or when the AMR / sandbox cost ratio is re-measured.

/// A script window's fixed shard cost, in AMR ns: 836,886 (derivation above).
pub const FANOUT_SCRIPT_WINDOW_FIXED_NS: u64 = 836_886;

/// A script window's shard cost per wire command (two per value row, one per
/// section, tombstone or children-list entry), in AMR ns: 2,860 (derivation
/// above). It covers the value rows' in-script follow-ups, which are not
/// counted as wire commands.
pub const FANOUT_SCRIPT_COMMAND_NS: u64 = 2_860;

/// A script window's shard cost per children-list entry its ranged `LRANGE`s
/// return, in AMR ns: 0 (derivation above). An in-script `LRANGE`'s cost
/// grows with its length (each element converted to Lua and back), which a
/// cut in commands alone cannot bound; the fit covers that growth through
/// [`FANOUT_SCRIPT_REPLY_BYTE_NS`] × its 20 B of reply bound per entry, so
/// this term is 0 until a re-fit needs it.
pub const FANOUT_SCRIPT_LIST_ENTRY_NS: u64 = 0;

/// A script window's shard cost per byte of its summed reply bound, in AMR
/// ns: 35 (derivation above).
pub const FANOUT_SCRIPT_REPLY_BYTE_NS: u64 = 35;

/// A script window's modelled shard time on AMR, in ns:
/// [`FANOUT_SCRIPT_WINDOW_FIXED_NS`] + [`FANOUT_SCRIPT_COMMAND_NS`] × wire
/// commands + [`FANOUT_SCRIPT_LIST_ENTRY_NS`] × the entries its ranged
/// `LRANGE`s ask for + [`FANOUT_SCRIPT_REPLY_BYTE_NS`] × its summed reply
/// bound, saturating. The structural WATCH read cuts each script window so
/// this stays within [`FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS`].
pub const fn fanout_script_window_modelled_shard_ns(
    wire_commands: u64,
    list_entries: u64,
    reply_bound_bytes: u64,
) -> u64 {
    FANOUT_SCRIPT_WINDOW_FIXED_NS
        .saturating_add(FANOUT_SCRIPT_COMMAND_NS.saturating_mul(wire_commands))
        .saturating_add(FANOUT_SCRIPT_LIST_ENTRY_NS.saturating_mul(list_entries))
        .saturating_add(FANOUT_SCRIPT_REPLY_BYTE_NS.saturating_mul(reply_bound_bytes))
}

/// Default `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`: catch-up atomic
/// snapshot reads in flight per replica (`P_a`), 1.
///
/// The value the ceilings above are solved at (Spec SDD SR-2.5, SR-11.6; report
/// §8.3 step 6 of Epic 1, and `## Recorded inputs (SR-11.8)` of
/// `docs-internal/validation/proven-snapshot-reads-20261001/report.md`). It
/// enters [`SNAPSHOT_POOL_SIZE_DEFAULT`].
pub const CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT: u32 = 1;

/// Default `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`: catch-up fenced
/// snapshot reads in flight per replica (`P_f`), 1.
///
/// Recorded beside `P_a` in the same report section; the fenced read it counts
/// is Spec SDD SR-5's. It enters [`SNAPSHOT_POOL_SIZE_DEFAULT`] and the pod
/// memory plan's catch-up terms (`config::memory_plan`).
pub const CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT: u32 = 1;

/// The catch-up memory budget `T` at the reference pod memory limit, 744 MiB
/// (Spec SDD SR-7.2): what the pod memory plan derives at 5,120 MiB for the
/// chart's configuration (`config::memory_plan`), and what a `Config` built
/// field by field, rather than derived from a limit, sets
/// [`Config::catchup_build_memory_budget_mib`] to.
///
/// The plan's rule at that limit: `T` = floor(f × (limit − floor)) =
/// floor(0.15 × (5,120 − 155)) = 744, with a 155 MiB steady-state floor.
/// f = 0.15 is the largest 0.01 step for which one term list, every consumer
/// the build funds in that limit, fits with each share of the limit charged
/// exactly (`config::memory_plan`, *Exact shares*). At the values the
/// service builds, the list is:
/// floor + `T × (1 + P_f)` (1,488 MiB at `P_f` = 1) + `P_f × X` (one read
/// stage's fan-out window transient, which the flight's grant does not
/// reserve; 143.0 MiB) + one checkpoint fenced read of the reference layer,
/// 4× the item ceiling (172.9 MiB) + the load terms SDD §16 funds in that
/// limit (684.8 MiB) + the full structural write-planning cache at its
/// resident budget, the configured 256 MiB × (1 + φ) for its entries' plan
/// indexes (309.76 MiB, φ = 0.21,
/// `storage::structural_cache::resident_budget_bytes`) + the cache's
/// cold-start reads (11.7 MiB: each of the two plain reads' reply, the state
/// decoded from it and the plan indexes' build at its peak) and its sampled
/// audit (128.6 MiB), both outside the cache's budget (SDD §21.2) + the
/// uploads (857.1 MiB: the
/// upload memory budget, 384 MiB, its 64 queued waiters at the 1 MiB HTTP/2
/// stream window each, and one `LIVE_EDIT_MAX_LAYER_SIZE` single-value
/// upload in the overflow slot, 409.1 MiB) + the Valkey connections' at-rest
/// buffers on a 10-primary cluster at the chart's pools (127.7 MiB,
/// `storage::connection_buffers::at_rest_bound_bytes`) + its 20 % margin
/// (1,024 MiB) = 5,102.7 ≤ 5,120 MiB; at f = 0.16 (`T` = 794) the same list
/// is 5,202.7 MiB. At exact shares, the shares unrounded, the list is
/// 5,104.1 MiB at f = 0.15, 15.9 MiB under the limit, and 5,203.4 MiB at
/// f = 0.16. With the uploads and the connections' buffers in the list, a
/// `T` the rule accepts closes SDD §21.2's combined line by construction.
/// At other limits the same rule gives 433 at 4,096 MiB and 1,607 at
/// 8,192 MiB.
///
/// `X` is 143.0 MiB, one term the catch-up, the structural cache's cold start
/// and the structural WATCH read share. Every fenced fan-out pass runs on
/// the one `fanout-read` thread and every unfenced one on one of a fixed
/// four `watch-read` threads (`storage::fanout_thread`), and in the service
/// binary all of those threads, with `snapshot-io`, allocate from one
/// dedicated jemalloc arena with a 1,000 ms decay (`src/allocator.rs`, *The
/// fan-out arena*), so `X` grows neither with the replica's worker count nor
/// with the number of fan-out threads. Measured on that build, the largest
/// of the three reads' `X` is the WATCH read's 134.5 MiB; with the WATCH
/// read's rounds 2 and 3 through window scripts, that read's `X` is 7.3 MiB
/// at most (96 reads of 4x and 10x layers at 2 and 6 CPUs), and the read
/// issues plain windows only for round 1, a lone children list over τ, and
/// every window after a script fallback. Those reads ran on the window
/// model's coefficients as wired ([`fanout_script_window_modelled_shard_ns`]).
/// On any coefficients, the WATCH read's `X` stays at or below the plain WATCH
/// read's 134.5 MiB above [model]: every script window is cut by the same
/// command cap ([`FANOUT_WINDOW_MAX`]) and byte cut
/// ([`snapshot_fanout_window_bytes`]) as that read's plain windows (and by τ
/// and its own request bytes besides), with the cut's two exceptions the same
/// on both; the read holds one window at a time on both; and a script window
/// holds one EVALSHA future, an argument list within the byte cut and its
/// replies, where a plain window held one issuer future and one fred command
/// per command beside the same replies. The record states the argument in
/// full (`derivation.json` `x_watch_script.shipped_build_bound`). So the
/// record's rule (the largest of those and the catch-up's 142.1 MiB model)
/// gives 142.1 MiB. The term is instead the 143.0 MiB WATCH maximum of the build
/// before the arena (the fan-out threads on jemalloc's default arenas, at
/// their 10 s decay), kept as the conservative bound: it rests on three
/// times as many WATCH reads, the arena only lowers what a read leaves
/// resident, and the plain WATCH read it measured still ships, as the
/// fallback. It costs 0.9 MiB of the combined line's headroom. The plan
/// carries it as `memory_plan::WINDOW_TRANSIENT_BYTES`.
///
/// The record: `docs-internal/validation/watch-window-script-b-gvf-20261010/`,
/// `derivation.json` (the 5,120 MiB list, its inputs and sources, copied from
/// `docs-internal/validation/structural-edit-inplace-b-ug2-20261010/derivation.json`
/// — whose `report.md` §5 gives the cache and cold-start terms, and which
/// supersedes the list of `docs-internal/validation/fanout-executor-b26e-s2-20261009/`,
/// the source of the window, connection and upload terms, unchanged — with
/// the window-script build's WATCH read added to the window term's inputs).
/// Its
/// reply-byte terms are on real-id bytes, from
/// `docs-internal/validation/real-id-byte-ceiling-20261006/` (`derivation.json`
/// `memory_mib`, the 4,096 MiB list that gave 788 before the uploads and the
/// connections' buffers joined it), whose base is the Task 10 section of
/// `docs-internal/validation/fenced-snapshot-reads-20261002/report.md`
/// (`raw/task10-results/results.json` `memory.by_pod_limit`, short-id bytes:
/// 283 at 2,048 MiB, 985 at 4,096 MiB). That record's rule took the largest
/// 0.05 step; the plan takes the largest 0.01 step at which the line fits at
/// exact shares, which gives the same 744 at 5,120 MiB.
pub const CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT: u64 = 744;

/// Snapshot-lane connections for the readers that are not catch-up reads: the
/// structural path, the checkpoint assembler and old-object ext-GC scheduling
/// (Spec SDD SR-10.1).
pub const SNAPSHOT_POOL_NON_CATCHUP_CONNECTIONS: u32 = 4;

/// `LIVE_EDIT_SNAPSHOT_POOL_SIZE` at the default `P_a` and `P_f`:
/// `P_a + P_f + 4` = 6 (Spec SDD SR-10.1).
pub const SNAPSHOT_POOL_SIZE_DEFAULT: u32 = snapshot_pool_size_default(
    CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
    CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
);

/// Default `LIVE_EDIT_SNAPSHOT_POOL_SIZE` for a configured
/// `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` (`P_a`) and
/// `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY` (`P_f`): `P_a + P_f + 4` (Spec
/// SDD SR-10.1), so the lane grows with the atomic and fenced reads a replica
/// runs at once. Saturates rather than overflowing near `u32::MAX`.
pub const fn snapshot_pool_size_default(
    catchup_snapshot_read_concurrency: u32,
    catchup_fenced_read_concurrency: u32,
) -> u32 {
    catchup_snapshot_read_concurrency
        .saturating_add(catchup_fenced_read_concurrency)
        .saturating_add(SNAPSHOT_POOL_NON_CATCHUP_CONNECTIONS)
}

/// `FANOUT_WINDOW`: the most snapshot commands one bounded fan-out read keeps
/// outstanding, derived from the configured item ceiling (Spec SDD SR-11.3).
///
/// min([`snapshot_fanout_window_uncapped`], [`FANOUT_WINDOW_MAX`]): the
/// ceiling's window, floor(`max_items` ÷ 5.266), capped at 2,141 commands so
/// the fenced windows of one join wave stay within SR-11.6 (i)'s queue limit
/// (see the cap). 2,141 at the default ceiling (uncapped it would be 7,131);
/// an operator who lowers the ceiling below 11,275 items narrows the window
/// with it.
pub fn snapshot_fanout_window(max_items: u64) -> u64 {
    snapshot_fanout_window_uncapped(max_items).min(FANOUT_WINDOW_MAX)
}

/// The ceiling's own fan-out window, before [`FANOUT_WINDOW_MAX`]:
/// floor(`max_items` ÷ 5.266) ([`SNAPSHOT_ITEMS_PER_ENTRY_MILLI`]), the
/// commands of one atomic read's worth of entries at the fitted items per
/// entry (Epic 2's recorded `fanout_window`, 7,131 at the default ceiling).
/// Never below 1: a ceiling under 5.266 items floors to zero commands, and a
/// window that admits none would never read anything.
pub fn snapshot_fanout_window_uncapped(max_items: u64) -> u64 {
    // u128 so the scaling cannot overflow; the quotient is below `max_items`,
    // so it always fits back in u64.
    let window = u128::from(max_items) * 1_000 / u128::from(SNAPSHOT_ITEMS_PER_ENTRY_MILLI);
    (window as u64).max(1)
}

/// `FANOUT_WINDOW_BYTES`: the reply-byte target one bounded fan-out window is
/// cut to. It equals the configured byte ceiling (Spec SDD SR-11.3).
///
/// A target, not a hard maximum: the reader sizes each window from the average
/// reply bytes per command it has observed so far
/// (`storage::diff_read::fanout_window_len`), so a window's replies come to
/// about this figure when its entries are near that average and more when they
/// are larger — up to `FANOUT_WINDOW` × the largest entry in the window. A
/// window always holds at least one command, so once the observed average
/// exceeds this target the windows shrink to one command each; but a window is
/// sized before its own replies arrive, so a reply larger than this can still
/// share its window with up to `FANOUT_WINDOW − 1` other commands. The hard
/// bound is [`snapshot_fanout_window`]'s command count.
///
/// That is the whole-row read's rule (the fenced read's pass, ext-GC). The
/// structural update's read, whose replies are bounded per command, sizes
/// each window and index page so the sum of its commands' reply bounds stays
/// within this figure (`storage::diff_read::FanoutReplies::Bounded`), so for
/// it this is a bound, except for names longer than any it has seen yet and
/// a single command whose bound alone exceeds it, which runs alone.
pub fn snapshot_fanout_window_bytes(max_bytes: u64) -> u64 {
    max_bytes
}

impl Config {
    /// [`snapshot_fanout_window`] at this configuration's
    /// [`Self::snapshot_atomic_read_max_items`].
    pub fn snapshot_fanout_window(&self) -> u64 {
        snapshot_fanout_window(self.snapshot_atomic_read_max_items)
    }

    /// [`snapshot_fanout_window_bytes`] at this configuration's
    /// [`Self::snapshot_atomic_read_max_bytes`].
    pub fn snapshot_fanout_window_bytes(&self) -> u64 {
        snapshot_fanout_window_bytes(self.snapshot_atomic_read_max_bytes)
    }

    /// [`structural_cache_budget_bytes`] at this configuration's
    /// [`Self::structural_cache_max_mib`]: the structural-planning cache's
    /// budget in bytes, `0` when the cache is disabled.
    pub fn structural_cache_budget_bytes(&self) -> u64 {
        structural_cache_budget_bytes(self.structural_cache_max_mib)
    }

    /// Whether the checkpoint base bootstrap runs: checkpointing enabled, a
    /// checkpoint adapter wired, and the tail enabled (SR-7.1: the unit
    /// serves the tail path, so it is off whenever that is). The one
    /// definition of the condition: `main.rs` builds `BaseBootstrap` with it,
    /// and the pod memory plan charges the base bootstraps' margin term
    /// exactly when it holds ([`PlanInputs::from_config`]).
    pub fn checkpoint_base_bootstrap_enabled(&self) -> bool {
        self.checkpoint_adapter_wired() && self.checkpoint_tail_enabled
    }

    /// Whether checkpointing is enabled with a checkpoint adapter wired
    /// (anything but `Disabled`). The one definition of the condition the
    /// pod memory plan charges the external-value mirror's byte budget on
    /// ([`PlanInputs::from_config`]), so `main.rs` builds the mirror exactly
    /// when [`PodMemoryPlan::ext_mirror_bytes`] is `Some`.
    pub fn checkpoint_adapter_wired(&self) -> bool {
        self.checkpoint_enabled && !matches!(self.checkpoint_adapter, CheckpointAdapter::Disabled)
    }

    /// The backup store's identity: 16 lowercase hex characters naming the
    /// place the checkpoint adapter writes to. External-value backup
    /// confirmations are written under it (`storage::ext_backup`), and the
    /// backup repair treats a confirmation under any other identity as no
    /// confirmation, so pointing the adapter at a different store makes every
    /// value owed again and the repair re-checks each one against the new
    /// store (SDD §22.17).
    ///
    /// The first 8 bytes of SHA-256 over `<adapter>|<address>`, the address
    /// trimmed of surrounding whitespace and trailing `/`: the endpoint and
    /// base path for `StorageApi`, the directory for `LocalDisk`. `InMemory`
    /// keeps its blobs in this process, so its identity is random per process
    /// and no other process's confirmation counts for it. `Disabled` writes
    /// nowhere; its identity is fixed and never confirmed against.
    pub fn checkpoint_store_identity(&self) -> String {
        use sha2::{Digest, Sha256};
        fn normalised(value: &str) -> &str {
            value.trim().trim_end_matches('/')
        }
        let address = match self.checkpoint_adapter {
            CheckpointAdapter::Disabled => "disabled|".to_owned(),
            CheckpointAdapter::InMemory => {
                static PROCESS_ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
                let process = PROCESS_ID.get_or_init(|| uuid::Uuid::new_v4().to_string());
                format!("in_memory|{process}")
            }
            CheckpointAdapter::LocalDisk => format!(
                "local_disk|{}",
                normalised(self.checkpoint_local_disk_path.as_deref().unwrap_or(""))
            ),
            CheckpointAdapter::StorageApi => format!(
                "storage_api|{}|{}",
                normalised(
                    self.checkpoint_storage_api_endpoint
                        .as_deref()
                        .unwrap_or("")
                ),
                normalised(
                    &self
                        .checkpoint_base_path
                        .as_ref()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default()
                )
            ),
        };
        Sha256::digest(address.as_bytes())[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// `true` for the one shape the test-only override admits: the userinfo
    /// service is off, authorization is off, and
    /// [`Self::test_only_dangerously_allow_unvalidated_tokens`] let that pair
    /// through `from_env`.
    ///
    /// This is the only shape in which the override changes behaviour. With
    /// either auth backend enabled the flag is inert and this returns `false`
    /// even when the env var is set.
    ///
    /// It is NOT a general "no backend validates the bearer" predicate. With
    /// the userinfo service on and authorization off, nothing validates the
    /// bearer's signature or expiry either, unless a validating proxy fronts
    /// the userinfo endpoint — the userinfo service resolves identity, it does
    /// not validate. That shape needs no override, is accepted by `from_env`,
    /// and this function returns `false` for it. A known gap in the current
    /// behaviour.
    pub fn unvalidated_tokens_in_effect(&self) -> bool {
        !self.userinfo_service_enabled
            && !self.authz_enabled
            && self.test_only_dangerously_allow_unvalidated_tokens
    }

    /// Emit the startup log line for the test-only unvalidated-token override.
    ///
    /// Lives on `Config` (rather than inline in `from_env`) so it is a
    /// single, separately callable line: `main.rs` calls it right after
    /// `from_env` returns, ahead of every dependency step and INFO line
    /// (only WARNs `from_env` itself emitted can precede it). `main.rs` is
    /// the production call site; integration tests drive it directly with a
    /// synthetic `Config` and capture the output.
    ///
    /// Fires once at ERROR when the override is in effect — ERROR rather than
    /// WARN because this is the only runtime signal that the override let the
    /// userinfo-off, authz-off shape start without token validation, and
    /// `LIVE_EDIT_LOG_LEVEL=error` must not be able to hide it. (The
    /// userinfo-on, authz-off shape without a validating proxy also validates
    /// no tokens and emits no such signal; see
    /// [`Self::unvalidated_tokens_in_effect`].) Fires once at WARN (with
    /// different wording) when the variable is set but inert, so a stray copy
    /// of it in a real deployment's environment is still visible in startup
    /// logs.
    pub fn log_test_only_auth_override_startup_warning(&self) {
        if self.unvalidated_tokens_in_effect() {
            tracing::error!(
                "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS=true with \
                 LIVE_EDIT_USERINFO_SERVICE_ENABLED=false and LIVE_EDIT_AUTHZ_ENABLED=false: \
                 NO backend validates bearer tokens (signature, issuer, audience, expiry). Any \
                 well-formed JWT-shaped string is accepted as a genuine caller and every \
                 permission check is granted. This mode exists only for the conformance test \
                 suite and must never be used outside a test deployment."
            );
        } else if self.test_only_dangerously_allow_unvalidated_tokens {
            tracing::warn!(
                "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS=true is set but has no \
                 effect because at least one auth backend is enabled \
                 (LIVE_EDIT_USERINFO_SERVICE_ENABLED={}, LIVE_EDIT_AUTHZ_ENABLED={}). Unset it: \
                 the variable is a test-only hatch and should not appear in a real deployment.",
                self.userinfo_service_enabled,
                self.authz_enabled,
            );
        }
    }

    /// Load configuration from environment variables, with every pod-memory
    /// budget derived from the memory limit of the cgroup this process runs
    /// in (`config::pod_memory`, `config::memory_plan`).
    ///
    /// Returns `Err` immediately if any required variable is absent, with a message
    /// that names the missing variable; when the pod memory plan cannot close
    /// at the limit, naming the smallest limit the configuration needs; and
    /// when a removed memory-budget variable is set, naming it.
    pub fn from_env() -> Result<Self, String> {
        Self::from_env_with(PodMemorySource::Cgroup)
    }

    /// [`Self::from_env`] with the pod memory limit taken from `pod_memory`
    /// rather than from the process's cgroup.
    ///
    /// Tests call this with [`PodMemorySource::Reference`] or
    /// [`PodMemorySource::Fixed`], so the limit of the machine running them
    /// never changes their outcome.
    pub fn from_env_with(pod_memory: PodMemorySource) -> Result<Self, String> {
        let authn_enabled = optional_parsed("LIVE_EDIT_AUTHN_ENABLED", true)?;
        if !authn_enabled {
            return Err(
                "LIVE_EDIT_AUTHN_ENABLED=false is not supported: disabling authentication is not allowed".to_string(),
            );
        }

        let userinfo_service_enabled =
            optional_parsed("LIVE_EDIT_USERINFO_SERVICE_ENABLED", false)?;
        let authz_enabled = optional_parsed("LIVE_EDIT_AUTHZ_ENABLED", true)?;

        // Test-only override of the SR-1.5 rejection below. Parsed
        // unconditionally so a malformed value is reported even when the
        // override would be inert, and so the field reflects exactly what the
        // operator set. The rejection message deliberately does not name this
        // variable: it is a conformance-suite hatch, not an operator remedy.
        let test_only_dangerously_allow_unvalidated_tokens = optional_parsed(
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            false,
        )?;

        // SR-1.5 cross-validation (b.g3z). With both userinfo and authz
        // disabled, no backend validates the bearer token (signature,
        // issuer, audience, expiry) — `authn_enabled` alone only enforces
        // header presence, not token validity. Enabling userinfo instead does
        // not close that by itself: with authz off, the bearer is validated
        // only if a validating proxy fronts the userinfo endpoint (see
        // `Config::unvalidated_tokens_in_effect`). Run this check inline
        // (per SR-1.4) and before any endpoint validation work so the
        // failure message is the first error an operator sees.
        //
        // The override is checked here and nowhere else: the runtime already
        // handles the off/off shape (`NoOpPermissionsClient` +
        // `JwtUserinfoClient`), so relaxing the parse-time guard is the
        // whole mechanism. The startup ERROR line is
        // `Config::log_test_only_auth_override_startup_warning`, which
        // `main.rs` calls right after `from_env` returns.
        if !userinfo_service_enabled
            && !authz_enabled
            && !test_only_dangerously_allow_unvalidated_tokens
        {
            return Err(
                "LIVE_EDIT_USERINFO_SERVICE_ENABLED=false requires LIVE_EDIT_AUTHZ_ENABLED=true. \
                 With both disabled, no backend validates the bearer token (signature, issuer, audience, expiry) — \
                 note that LIVE_EDIT_AUTHN_ENABLED only enforces header presence, not token validity. \
                 Set LIVE_EDIT_AUTHZ_ENABLED=true (permissions service validates tokens). \
                 Setting LIVE_EDIT_USERINFO_SERVICE_ENABLED=true instead resolves user identity \
                 but does not itself validate the bearer's signature or expiry; tokens are then \
                 validated only if a validating proxy fronts the userinfo endpoint."
                    .to_string(),
            );
        }

        let storage_service_endpoint = if authz_enabled {
            Some(required("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT")?)
        } else {
            env::var("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT").ok()
        };
        let userinfo_endpoint = if userinfo_service_enabled {
            Some(required("LIVE_EDIT_USERINFO_ENDPOINT")?)
        } else {
            env::var("LIVE_EDIT_USERINFO_ENDPOINT").ok()
        };

        // Fail-closed scheme validation (b.ohw C-6 / H-auth-1). Bare
        // hostnames are rejected at parse time so a misconfigured
        // deployment cannot silently downgrade the auth channel to plaintext
        // — bearer tokens transit those connections, and a downgrade would
        // leak them on the wire. The same validator is also called inside
        // the auth-client constructors as defense-in-depth.
        if let Some(ep) = storage_service_endpoint.as_deref() {
            validate_auth_endpoint("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", ep)?;
        }
        if userinfo_service_enabled {
            if let Some(ep) = userinfo_endpoint.as_deref() {
                validate_auth_endpoint("LIVE_EDIT_USERINFO_ENDPOINT", ep)?;
            }
        }

        // Checkpoint adapter selection + dependent-var guards.
        //
        // Resolved up-front (mirroring storage/userinfo endpoint handling)
        // so the LocalDisk and StorageApi dependent-var requirements can
        // fail fast — even if `LIVE_EDIT_CHECKPOINT_ENABLED=false` — before
        // any other field-level parsing work that might error first and
        // obscure the real problem.
        let checkpoint_adapter =
            optional_parsed("LIVE_EDIT_CHECKPOINT_ADAPTER", CheckpointAdapter::Disabled)?;
        let checkpoint_local_disk_path = env::var("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if checkpoint_adapter == CheckpointAdapter::LocalDisk
            && checkpoint_local_disk_path.is_none()
        {
            return Err(
                "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH is required (non-empty) when \
                 LIVE_EDIT_CHECKPOINT_ADAPTER=localdisk. Set the variable to the \
                 filesystem path the LocalDisk adapter should write checkpoints to."
                    .to_string(),
            );
        }
        // StorageApi dependent-var resolution (Epic t1.fgj.jd / SDD SR-4.5
        // / SR-4.6). Reads happen unconditionally so the values are
        // populated on the `Config` regardless of which adapter is
        // selected (silent-tolerance contract — Task `t2.fgj.jd.8e`
        // consumes them only under StorageApi, but downstream tooling
        // can inspect the parsed values for any adapter without a
        // separate plumbing pass).
        let checkpoint_storage_api_endpoint = env::var("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let checkpoint_base_path = env::var("LIVE_EDIT_CHECKPOINT_BASE_PATH")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let checkpoint_connect_timeout_ms = {
            let n = optional_parsed("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS", 5_000u64)?;
            if n == 0 {
                return Err(
                    "invalid value for LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS: must be greater than 0 (a zero connect-phase timeout would fail every StorageApi checkpoint call synchronously)".to_string(),
                );
            }
            n
        };
        let checkpoint_request_timeout_ms = {
            let n = optional_parsed("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", 30_000u64)?;
            if n == 0 {
                return Err(
                    "invalid value for LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS: must be greater than 0 (a zero per-request timeout would fail every StorageApi checkpoint call synchronously)".to_string(),
                );
            }
            n
        };
        // Ext-value blob mirror knobs (Epic `t1.fgj.zc` / SDD SR-3.1).
        // Both are zero-rejected: zero workers would silently drop every
        // enqueue once the bounded channel filled, and
        // `tokio::sync::mpsc::channel(0)` panics outright. The worker count
        // is also capped (`EXT_MIRROR_MAX_WORKERS`): each worker's adapter
        // connection holds buffers outside the mirror's byte budget. Parsed
        // up-front alongside the other checkpoint-config fields so the
        // dependent ext_mirror module / ValkeyExternalValueStore integration
        // land these values via `Config`. See `checkpoint_ext_mirror_workers`
        // and `checkpoint_ext_mirror_queue_capacity` rustdoc above.
        let checkpoint_ext_mirror_workers = {
            let n = optional_parsed("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS", 2usize)?;
            if n == 0 {
                return Err(
                    "invalid value for LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS: must be greater than 0 (zero would spawn no workers and silently drop every enqueue once the bounded channel filled)".to_string(),
                );
            }
            let most = crate::checkpoint::ext_mirror::EXT_MIRROR_MAX_WORKERS;
            if n > most {
                return Err(format!(
                    "invalid value for LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS: {n} is above the maximum of {most} (each worker's connection to the backup store holds memory the pod memory plan does not budget for; more workers only add concurrent uploads to the backup store)"
                ));
            }
            n
        };
        let checkpoint_ext_mirror_queue_capacity = {
            let n = optional_parsed("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY", 1_000usize)?;
            if n == 0 {
                return Err(
                    "invalid value for LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY: must be greater than 0 (tokio::sync::mpsc::channel(0) panics)".to_string(),
                );
            }
            n
        };
        // Epic t1.fgj.dp / SDD SR-6.1 / SR-6.2 / SR-6.4. Filesystem path —
        // no scheme validation. Trimmed-empty → `None` so a manifest line
        // like `LIVE_EDIT_SERVICE_TOKEN_FILE=" "` doesn't hand a meaningless
        // path to the polling task. Hoisted out of the struct literal because
        // the StorageApi validation below requires it.
        let service_token_file: Option<PathBuf> = env::var("LIVE_EDIT_SERVICE_TOKEN_FILE")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        // StorageApi dependent-var validation. Error ordering matches
        // the STORAGE_SERVICE_ENDPOINT convention used above (missing
        // first, then shape): missing-endpoint → missing-base-path →
        // missing-token-file → bare-hostname. The token file is required
        // because the adapter authenticates outbound calls only with the
        // token the credential-provisioner sidecar writes there; this is the
        // primary gate, and `build_checkpoint_store` refuses the same
        // combination again as defense in depth. The bare-hostname
        // rejection flows through `validate_auth_endpoint`, which delegates to
        // `crate::auth::validate_auth_endpoint_uri` — operator-facing
        // wording is therefore byte-identical to the existing
        // `LIVE_EDIT_STORAGE_SERVICE_ENDPOINT` rejection aside from the
        // env-var name in the `"invalid value for {var}: ..."` prefix.
        if checkpoint_adapter == CheckpointAdapter::StorageApi {
            if checkpoint_storage_api_endpoint.is_none() {
                return Err(
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT is required (non-empty) when \
                     LIVE_EDIT_CHECKPOINT_ADAPTER=storageapi. Set the variable to the \
                     storage-api endpoint URL the adapter should target (with an explicit \
                     https:// or http:// scheme)."
                        .to_string(),
                );
            }
            if checkpoint_base_path.is_none() {
                return Err(
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH is required (non-empty) when \
                     LIVE_EDIT_CHECKPOINT_ADAPTER=storageapi. Set the variable to the \
                     base path under which the adapter renders per-object checkpoint \
                     blob locations (e.g. `/projects/my-project`)."
                        .to_string(),
                );
            }
            if service_token_file.is_none() {
                return Err("LIVE_EDIT_SERVICE_TOKEN_FILE is required (non-empty) when \
                     LIVE_EDIT_CHECKPOINT_ADAPTER=storageapi. Set it to the token file the \
                     credential-provisioner sidecar writes (Helm: \
                     serviceIdentity.accessTokenFile.enabled: true)."
                    .to_string());
            }
            if let Some(ep) = checkpoint_storage_api_endpoint.as_deref() {
                validate_auth_endpoint("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT", ep)?;
            }
        }

        let service_label = env::var("LIVE_EDIT_SERVICE_LABEL")
            .or_else(|_| env::var("HOSTNAME"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        // Hoisted out of the struct literal because the ref_count reconciler's
        // interval default is derived from it (b.yda) — same shape and same
        // reason as `checkpoint_request_timeout_ms` above, whose reconciliation
        // grace feeds the checkpoint reconcile interval's default.
        let rpc_deadline_ms: u64 =
            optional_parsed("LIVE_EDIT_RPC_DEADLINE_MS", RPC_DEADLINE_MS_DEFAULT)?;

        // Hoisted for the same reason: the upload budget's default and floor
        // are derived from it.
        let max_layer_size: usize =
            optional_parsed("LIVE_EDIT_MAX_LAYER_SIZE", MAX_LAYER_SIZE_DEFAULT)?;

        // Hoisted for the same reason: `LIVE_EDIT_SNAPSHOT_POOL_SIZE`'s default
        // is derived from these two (`P_a + P_f + 4`, Spec SDD SR-10.1).
        let catchup_fenced_read_concurrency: u32 = {
            let n = optional_parsed(
                "LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY",
                CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            )?;
            if n == 0 {
                return Err(
                    "invalid value for LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY: must be greater than 0 (with no fenced read permit no catch-up of a layer above the atomic-read ceilings could ever read it, so every such subscribe would be refused)".to_string(),
                );
            }
            n
        };
        let catchup_snapshot_read_concurrency: u32 = {
            let n = optional_parsed(
                "LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY",
                CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            )?;
            if n == 0 {
                return Err(
                    "invalid value for LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY: must be greater than 0 (with no read permit no catch-up could ever read its layer, so every subscribe that has state to catch up on would be refused)".to_string(),
                );
            }
            n
        };

        // Assigned inside the struct literal, at the `log_level` field, so
        // the logging variables are validated at the same point as before.
        let tracing_settings: TracingSettings;
        let config = Config {
            port: optional_parsed("LIVE_EDIT_PORT", 50001u16)?,
            metrics_port: optional_parsed("LIVE_EDIT_METRICS_PORT", 9090u16)?,
            valkey_url: env::var("LIVE_EDIT_VALKEY_URL")
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            valkey_pool_size: optional_parsed("LIVE_EDIT_VALKEY_POOL_SIZE", 0u32)?,
            subscriber_pool_size: optional_parsed("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", 0u32)?,
            // Derived from the pod memory limit below, with the other budgets.
            subscriber_max_subscriptions: SUBSCRIPTION_CEILING_AT_REFERENCE_LIMIT,
            subscriber_max_concurrent_catchups: {
                let n = optional_semaphore_capacity(
                    "LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS",
                    8usize,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS: must be \
                         greater than 0 (each catch-up slot's memory is charged to the pod memory \
                         plan, and an unlimited count cannot be charged); set the number of \
                         catch-ups that may hold a slot at once"
                            .to_string(),
                    );
                }
                n
            },
            subscriber_admission_wait_ms: optional_parsed(
                "LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS",
                20_000u64,
            )?,
            subscriber_admission_max_waiters: {
                // Unset (or set to whitespace) means "follow the subscription
                // ceiling", `subscriber_max_subscriptions`.
                match env::var("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS") {
                    Ok(raw) if !raw.trim().is_empty() => {
                        let n =
                            optional_parsed("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS", 0usize)?;
                        if n == 0 {
                            return Err(
                                "invalid value for LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS: \
                                 must be greater than 0 (a zero waiter cap alongside a non-zero \
                                 LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS makes the admission wait \
                                 silently dead code — every saturated subscribe would be refused \
                                 without ever queueing, while the configured wait reported itself \
                                 as in effect). Express fail-fast with \
                                 LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS=0 instead, and leave this \
                                 unset to track the subscription ceiling."
                                    .to_string(),
                            );
                        }
                        Some(n)
                    }
                    _ => None,
                }
            },
            maintenance_pool_size: optional_parsed("LIVE_EDIT_MAINTENANCE_POOL_SIZE", 1u32)?,
            catchup_snapshot_read_concurrency,
            catchup_fenced_read_concurrency,
            // Derived from the pod memory limit below.
            catchup_build_memory_budget_mib: CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: {
                let n = optional_parsed(
                    "LIVE_EDIT_SNAPSHOT_POOL_SIZE",
                    snapshot_pool_size_default(
                        catchup_snapshot_read_concurrency,
                        catchup_fenced_read_concurrency,
                    ),
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_SNAPSHOT_POOL_SIZE: must be greater than 0 (the snapshot lane cannot be an empty connection pool: the Valkey client refuses to build one, so the service could not connect)".to_string(),
                    );
                }
                n
            },
            valkey_connect_timeout_secs: {
                let secs = optional_parsed("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", 30u64)?;
                if secs == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS: must be greater than 0 (it bounds every startup connect phase and every stream reader connect, so 0 would cut each of them before it could finish)".to_string(),
                    );
                }
                secs
            },
            valkey_command_timeout_ms: optional_parsed(
                "LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS",
                VALKEY_COMMAND_TIMEOUT_MS_DEFAULT,
            )?,
            valkey_reconnect_max_delay_ms: optional_parsed(
                "LIVE_EDIT_VALKEY_RECONNECT_MAX_DELAY_MS",
                30_000u32,
            )?,
            valkey_auth_mode: optional_parsed(
                "LIVE_EDIT_VALKEY_AUTH_MODE",
                ValkeyAuthMode::Static,
            )?,
            valkey_token_file: env::var("LIVE_EDIT_VALKEY_TOKEN_FILE")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
            valkey_token_username: env::var("LIVE_EDIT_VALKEY_TOKEN_USERNAME")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            valkey_token_refresh_lead_secs: {
                let n = optional_parsed("LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS", 180u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS: must be greater than 0 (Azure requires refreshing the token at least 3 minutes before expiry; a zero lead time would let the token lapse before renewal)".to_string(),
                    );
                }
                // Representability, NOT an operational ceiling (b.is1). The
                // bound and its full derivation live on
                // `VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS`: it is the
                // `DateTime<Utc>`-range bound (the provider *adds* the lead to
                // `Utc::now()`), which is far tighter than the
                // `chrono::Duration` conversion's own `i64::MAX / 1_000` range.
                // Values above `2^63` would additionally sign-wrap through
                // `main.rs`'s `as i64` cast into a *negative* lead.
                if n > VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS: {n} exceeds \
                         {VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS}, the largest lead the token \
                         provider can add to the current time and still land inside chrono's \
                         representable DateTime<Utc> range (chrono::DateTime::<Utc>::MAX_UTC is \
                         year +262143). This is a representability bound only — no operational \
                         ceiling on the refresh lead is enforced, because the lead withholds no \
                         credential and an oversized value is otherwise harmless."
                    ));
                }
                n
            },
            valkey_iam_username: env::var("LIVE_EDIT_VALKEY_IAM_USERNAME")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            valkey_iam_region: env::var("LIVE_EDIT_VALKEY_IAM_REGION")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            valkey_iam_cluster_host: env::var("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            valkey_iam_service: optional_parsed(
                "LIVE_EDIT_VALKEY_IAM_SERVICE",
                ValkeyIamService::MemoryDb,
            )?,
            storage_service_endpoint,
            auth_timeout_ms: optional_parsed("LIVE_EDIT_AUTH_TIMEOUT_MS", 5000u64)?,
            auth_cache_ttl_secs: optional_parsed("LIVE_EDIT_AUTH_CACHE_TTL_SECS", 600u64)?,
            auth_cache_negative_ttl_secs: optional_parsed(
                "LIVE_EDIT_AUTH_CACHE_NEGATIVE_TTL_SECS",
                30u64,
            )?,
            auth_userinfo_negative_ttl_secs: optional_parsed(
                "LIVE_EDIT_AUTH_USERINFO_NEGATIVE_TTL_SECS",
                5u64,
            )?,
            auth_cache_max_entries: {
                // Upper bound (b.ohw medium): a misconfigured 100M-entry cap
                // would make the cache itself a memory-pressure liability —
                // 100M `PermissionKey`s × ~80 bytes each is ~8 GB resident
                // before any actual values land in it. Cap at 10M so an
                // operator typo can't cause the service to OOM trying to
                // grow into the configured ceiling under sustained churn.
                const AUTH_CACHE_MAX_ENTRIES_CEILING: u64 = 10_000_000;
                let n = optional_parsed("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES", 10_000u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES: must be greater than 0 (zero would disable the cache)".to_string(),
                    );
                }
                if n > AUTH_CACHE_MAX_ENTRIES_CEILING {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES: \
                         {n} exceeds the {AUTH_CACHE_MAX_ENTRIES_CEILING} hard ceiling \
                         (each entry consumes ~80 bytes; values above this risk OOM \
                         under churn). Lower the value or split traffic across replicas."
                    ));
                }
                n
            },
            userinfo_endpoint,
            max_layer_size,
            // Derived from the pod memory limit below, never below the size
            // cap's floor.
            upload_memory_budget_mib:
                crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            upload_admission_wait_ms: {
                // Unset ⇒ derive; set ⇒ take the operator's number and judge
                // it against the RPC deadline. The
                // `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS` shape below, for the
                // same reason: the bound is a function of a different
                // variable, so a flat 2000 default would refuse to boot on a
                // tightened `LIVE_EDIT_RPC_DEADLINE_MS`, citing a variable the
                // operator never set. Deriving the default keeps "I didn't
                // configure this" a permanently valid state.
                //
                // The derived `min(2000, deadline / 2)` is below every
                // non-zero deadline. It is 0 — never wait — only for a
                // deadline under 2 ms, where a capacity wait could not fit
                // anyway.
                match env::var("LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS") {
                    Ok(_) => {
                        let n = optional_parsed("LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS", 0u64)?;
                        if n >= rpc_deadline_ms {
                            return Err(format!(
                                "invalid value for LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS: {n} must be \
                                 below LIVE_EDIT_RPC_DEADLINE_MS ({rpc_deadline_ms}); the RPC \
                                 deadline would always end the admission wait first. Lower it, \
                                 or unset it to take the derived default of min({}, \
                                 LIVE_EDIT_RPC_DEADLINE_MS / 2)",
                                crate::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
                            ));
                        }
                        n
                    }
                    Err(_) => crate::grpc::upload_admission::upload_admission_wait_ms_default(
                        rpc_deadline_ms,
                    ),
                }
            },
            upload_admission_max_waiters: optional_parsed(
                "LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS",
                crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            )?,
            layer_stream_maxlen: {
                let n = optional_parsed("LIVE_EDIT_LAYER_STREAM_MAXLEN", 4096u64)?;
                // b.qw6 item 3: reject MAXLEN values above the Lua f64
                // integer-precision ceiling (2^53 - 1). The Lua fast-path
                // script reads this value via `tonumber(ARGV[4])` and
                // round-trips it through Lua's `number` (f64) when forming
                // the `MAXLEN ~ N` argument to `XADD`. A configured value
                // above 2^53 - 1 silently rounds when crossed through
                // `tonumber`, so we reject at startup instead of letting
                // the precision loss alter the stream-trim behavior at
                // runtime. The 2^53 ceiling is many orders of magnitude
                // beyond any sane retention depth — even bounding each entry
                // at the smallest plausible payload, 2^53 entries is an
                // absurd stream depth — so any value past the ceiling is
                // fat-fingered input rather than a legitimate workload. (Note
                // this ceiling guards Lua f64 precision on the entry *count*
                // only; per-entry byte cost is a separate concern now that
                // each entry carries the b.gze relay `ompf` payload — see the
                // `warn_layer_stream_memory_footprint` startup WARN guard for
                // the worst-case `maxlen × max_layer_size` memory check.)
                // Matches the same ceiling that gates `seq` and
                // `next_set_order` inside `update_layer_fast.lua` (b.k1r
                // H-stor-9 and b.qw6 item 2).
                const LUA_F64_INTEGER_CEILING: u64 = (1u64 << 53) - 1;
                if n > LUA_F64_INTEGER_CEILING {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_LAYER_STREAM_MAXLEN: \
                         {n} exceeds the Lua f64 integer-precision ceiling \
                         ({LUA_F64_INTEGER_CEILING} = 2^53 - 1). Values above \
                         this round when the Lua fast-path script in \
                         update_layer_fast.lua converts the ARGV string \
                         through `tonumber`, silently altering the \
                         MAXLEN trim behavior. Lower the value — any \
                         realistic retention budget is many orders of \
                         magnitude below this ceiling."
                    ));
                }
                n
            },
            layer_stream_max_bytes: {
                let n = optional_parsed(
                    "LIVE_EDIT_LAYER_STREAM_MAX_BYTES",
                    LAYER_STREAM_MAX_BYTES_DEFAULT,
                )?;
                // The same Lua f64 ceiling as LIVE_EDIT_LAYER_STREAM_MAXLEN
                // above: the commit script reads the budget from its ARGV
                // through `tonumber` and refuses a value past 2^53 - 1 as an
                // encoder bug, which would fail every UpdateLayer. `0` is
                // accepted: it turns the byte budget off.
                const LUA_F64_INTEGER_CEILING: u64 = (1u64 << 53) - 1;
                if n > LUA_F64_INTEGER_CEILING {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_LAYER_STREAM_MAX_BYTES: {n} exceeds the Lua \
                         f64 integer-precision ceiling ({LUA_F64_INTEGER_CEILING} = 2^53 - 1) the \
                         update_layer_fast.lua commit script reads it through. Lower the value, or \
                         set 0 to turn the byte budget off."
                    ));
                }
                n
            },
            presence_broadcast_capacity: {
                let n = optional_parsed(
                    "LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY",
                    crate::presence::DEFAULT_BROADCAST_CHANNEL_CAPACITY,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY: must be greater than 0 (with 0 every presence event would be lost before a connection could read it)".to_string(),
                    );
                }
                n
            },
            presence_lease_ttl_secs: {
                let n = optional_parsed(
                    "LIVE_EDIT_PRESENCE_LEASE_TTL_SECS",
                    crate::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_PRESENCE_LEASE_TTL_SECS: must be greater than 0 (SET ... EX 0 is a Valkey syntax error)".to_string(),
                    );
                }
                n
            },
            presence_refresh_interval_secs: {
                let n = optional_parsed(
                    "LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS",
                    crate::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS: must be greater than 0 (refreshes would queue back-to-back and saturate the data pool)".to_string(),
                    );
                }
                n
            },
            // 0 is accepted: it turns the client keepalive off. 1..floor is
            // refused: a value meant as seconds would end every opted-in join
            // after a few milliseconds of silence.
            presence_keepalive_interval_ms: {
                let n = optional_parsed(
                    "LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS",
                    crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
                )?;
                let floor = crate::presence::MIN_PRESENCE_KEEPALIVE_INTERVAL_MS;
                if n != 0 && n < floor {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS: {n} milliseconds \
                         is below the {floor}-millisecond minimum. The interval is in milliseconds, \
                         not seconds: set at least {floor} (e.g. 15000 for 15 s), or 0 to turn the \
                         client keepalive off"
                    ));
                }
                n
            },
            shutdown_timeout_secs: optional_parsed("LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS", 5u64)?,
            shutdown_overall_budget_secs: optional_parsed(
                "LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS",
                30u64,
            )?,
            shutdown_stream_end_rate: optional_parsed("LIVE_EDIT_SHUTDOWN_STREAM_END_RATE", 25u32)?,
            shutdown_stream_end_max_window_secs: optional_parsed(
                "LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS",
                20u32,
            )?,
            // The same parse `main.rs` runs before installing the subscriber,
            // so these fields cannot disagree with it. Evaluated here, at the
            // fields' own position in the literal, so the order in which
            // invalid variables are reported is unchanged.
            log_level: {
                tracing_settings = TracingSettings::from_env()?;
                tracing_settings.log_level
            },
            log_format: tracing_settings.log_format,
            trace_filter: tracing_settings.trace_filter,
            sweeper_interval_secs: {
                let n = optional_parsed(
                    "LIVE_EDIT_SWEEPER_INTERVAL_SECS",
                    crate::presence::sweeper::DEFAULT_SWEEPER_INTERVAL_SECS,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_SWEEPER_INTERVAL_SECS: must be greater than 0 (zero would panic tokio::time::interval and disable the presence sweep)".to_string(),
                    );
                }
                n
            },
            session_list_timeout_ms: optional_parsed("LIVE_EDIT_SESSION_LIST_TIMEOUT_MS", 2000u64)?,
            rpc_deadline_ms,
            service_label,
            valkey_memory_bytes: match env::var("LIVE_EDIT_VALKEY_MEMORY_BYTES") {
                Ok(raw) => {
                    let n = raw.trim().parse::<u64>().map_err(|e| {
                        format!(
                            "invalid value for LIVE_EDIT_VALKEY_MEMORY_BYTES: {e}; it is the \
                             backing store's total memory in bytes, as the provider states it \
                             (for example 12000000000 for a 12 GB cache)"
                        )
                    })?;
                    if n < VALKEY_MEMORY_BYTES_MIN {
                        return Err(format!(
                            "invalid value for LIVE_EDIT_VALKEY_MEMORY_BYTES: {n} is below the \
                             minimum of {VALKEY_MEMORY_BYTES_MIN} bytes (100 MB); it is the \
                             backing store's total memory in bytes, as the provider states it \
                             (for example 12000000000 for a 12 GB cache), so a smaller value is \
                             almost certainly a units slip. Set the total in bytes, or unset it \
                             on a store that reports its own maxmemory"
                        ));
                    }
                    Some(n)
                }
                Err(_) => None,
            },
            ext_release_max_bytes: match env::var("LIVE_EDIT_EXT_RELEASE_MAX_BYTES") {
                Ok(raw) => Some(raw.trim().parse::<u64>().map_err(|e| {
                    format!("invalid value for LIVE_EDIT_EXT_RELEASE_MAX_BYTES: {e}")
                })?),
                Err(_) => None,
            },
            ext_reconcile_interval_secs: {
                // Unset ⇒ derive; set ⇒ take the operator's number verbatim and
                // let `validate_ext_reconcile_interval` judge it. Exactly the
                // `LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS` shape above,
                // for exactly the same reason.
                //
                // The floor cannot be the bare 3600 constant, because the bound
                // this knob is validated against — the reconciler's derived
                // grace, `max(600 s, 10 x LIVE_EDIT_RPC_DEADLINE_MS)` — is a
                // function of a *different* variable. Raise the RPC deadline
                // past 360 s and the grace exceeds an hour, at which point a
                // flat 3600 default would fail the cross-validation and refuse
                // to boot on a variable the operator never touched, citing a
                // variable they never set. Deriving the default keeps "I didn't
                // configure this" a permanently valid state.
                //
                // `env::var(..).is_ok()` semantics rather than a sentinel, and
                // `0` stays the documented off switch on the explicit path only
                // — a derived default can never produce it.
                match env::var("LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS") {
                    Ok(raw) => raw.trim().parse::<u64>().map_err(|e| {
                        format!("invalid value for LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS: {e}")
                    })?,
                    Err(_) => EXT_RECONCILE_INTERVAL_DEFAULT_SECS.max(
                        crate::storage::ext_reconcile::ext_reconcile_grace(rpc_deadline_ms)
                            .as_secs(),
                    ),
                }
            },
            auth_max_attempts: optional_parsed("LIVE_EDIT_AUTH_MAX_ATTEMPTS", 3u32)?,
            auth_retry_initial_delay_ms: optional_parsed(
                "LIVE_EDIT_AUTH_RETRY_INITIAL_DELAY_MS",
                100u64,
            )?,
            authn_enabled,
            authz_enabled,
            userinfo_service_enabled,
            test_only_dangerously_allow_unvalidated_tokens,
            health_probe_interval_ms: {
                let n = optional_parsed("LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS", 5_000u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS: must be greater than 0 (zero would run the probe loop in a tight spin)".to_string(),
                    );
                }
                n
            },
            health_probe_timeout_ms: {
                let n = optional_parsed("LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS", 2_000u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS: must be greater than 0 (zero would fail every probe synchronously)".to_string(),
                    );
                }
                n
            },
            auth_protocol: optional_parsed("LIVE_EDIT_AUTH_PROTOCOL", AuthProtocol::Rest)?,
            update_layer_retry_max_attempts: {
                let n = optional_parsed("LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS", 128u32)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS: must be greater than 0 (zero would disable the first attempt)".to_string(),
                    );
                }
                n
            },
            update_layer_per_object_concurrency: {
                let n = optional_parsed("LIVE_EDIT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY", 16u32)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY: must be greater than 0 (zero would block every update_layer caller indefinitely)".to_string(),
                    );
                }
                n
            },
            update_layer_backend: optional_parsed(
                "LIVE_EDIT_UPDATE_LAYER_BACKEND",
                UpdateLayerBackend::Lua,
            )?,
            ext_value_verify: optional_parsed(
                "LIVE_EDIT_EXT_VALUE_VERIFY",
                ExtValueVerifyMode::Enforce,
            )?,
            catchup_snapshot_enabled: optional_parsed("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", true)?,
            snapshot_atomic_read_max_items: {
                let n = optional_parsed(
                    "LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS",
                    SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS: must be greater than 0 (a zero item ceiling would answer TOO_LARGE for every object, so no snapshot read could ever be atomic)".to_string(),
                    );
                }
                n
            },
            snapshot_atomic_read_max_bytes: {
                let n = optional_parsed(
                    "LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES",
                    SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES: must be greater than 0 (a zero byte ceiling would answer TOO_LARGE for every object, so no snapshot read could ever be atomic)".to_string(),
                    );
                }
                n
            },
            // Derived from the pod memory limit below.
            structural_cache_max_mib: STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: {
                let rate = optional_parsed(
                    "LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE",
                    STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
                )?;
                // The rate is a probability: the audit samples it with
                // `rand`'s `random_bool`, which panics outside [0, 1]. NaN
                // and the infinities parse as `f64` and fail this range test.
                if !(0.0..=1.0).contains(&rate) {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE: {rate} is not a probability; set a value from 0 (never audit) to 1 (audit every committed structural edit)"
                    ));
                }
                rate
            },
            structural_cache_audit_above_ceiling: optional_parsed(
                "LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING",
                true,
            )?,
            session_retry_max_attempts: {
                let n = optional_parsed("LIVE_EDIT_SESSION_RETRY_MAX_ATTEMPTS", 32u32)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_SESSION_RETRY_MAX_ATTEMPTS: must be greater than 0 (zero would disable the first attempt)".to_string(),
                    );
                }
                n
            },
            session_retry_initial_delay_ms: optional_parsed(
                "LIVE_EDIT_SESSION_RETRY_INITIAL_DELAY_MS",
                1u64,
            )?,
            session_tokens_sweep_interval_secs: {
                let n = optional_parsed(
                    "LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS",
                    crate::grpc::session::SESSION_TOKENS_SWEEP_INTERVAL_SECS,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS: must be greater than 0 (zero would panic tokio::time::interval and disable the sweep)".to_string(),
                    );
                }
                n
            },
            checkpoint_enabled: optional_parsed("LIVE_EDIT_CHECKPOINT_ENABLED", false)?,
            checkpoint_adapter,
            checkpoint_interval_secs: optional_parsed("LIVE_EDIT_CHECKPOINT_INTERVAL_SECS", 60u64)?,
            checkpoint_quiesce_secs: {
                let n = optional_parsed("LIVE_EDIT_CHECKPOINT_QUIESCE_SECS", 30u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_QUIESCE_SECS: must be greater than 0 (a zero-length quiesce window would mark every object as a checkpoint candidate immediately after any write)".to_string(),
                    );
                }
                n
            },
            checkpoint_tick_secs: {
                let n = optional_parsed("LIVE_EDIT_CHECKPOINT_TICK_SECS", 10u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_TICK_SECS: must be greater than 0 (zero would panic tokio::time::interval and spin the checkpoint loop)".to_string(),
                    );
                }
                n
            },
            checkpoint_retain_generations: {
                let n = optional_parsed("LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS", 2u32)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS: must be greater than 0 (retention of 0 would prune every just-written checkpoint and leave the object with no recoverable blob)".to_string(),
                    );
                }
                n
            },
            checkpoint_seq_safety_margin: {
                let n = optional_parsed("LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN", 1000u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN: must be greater than 0 (a zero margin makes a checkpoint-restored layer resume at exactly the highest sequence number the restore can prove, leaving no guard band for the commits lost inside the RPO window — subscribers that saw those commits would receive the reused sequence numbers attached to different content)".to_string(),
                    );
                }
                n
            },
            checkpoint_counters_sync_secs: {
                let n = optional_parsed("LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS", 60u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS: must be greater than 0 (zero would tight-loop the sidecar writer and saturate the maintenance pool)".to_string(),
                    );
                }
                n
            },
            checkpoint_reconcile_interval_secs: {
                // Unset ⇒ derive; set ⇒ take the operator's number verbatim and
                // let `validate_checkpoint_budgets` judge it.
                //
                // The floor cannot be the bare 3600 constant, because the bound
                // this knob is validated against (the derived reconciliation
                // grace) is itself a function of a *different* variable —
                // `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS`, via the StorageApi
                // adapter's publish arithmetic. Raise that timeout past ~200 s
                // and the grace exceeds an hour, at which point a flat 3600
                // default would fail the cross-validation and refuse to boot on
                // a variable the operator never touched, citing a variable they
                // never set. Deriving the default keeps "I didn't configure
                // this" a permanently valid state.
                //
                // `env::var(..).is_ok()` rather than a sentinel: matches
                // `optional_parsed`'s own absent-or-non-unicode ⇒ default rule,
                // so the two agree on what "unset" means.
                match env::var("LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS") {
                    Ok(raw) => {
                        let n = raw.trim().parse::<u64>().map_err(|e| {
                            format!(
                                "invalid value for LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS: {e}"
                            )
                        })?;
                        if n == 0 {
                            return Err(
                                "invalid value for LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS: must be greater than 0 (zero would panic tokio::time::interval and spin the reconciliation pass, whose every iteration is a full enumeration of the checkpoint store's layers/ namespace)".to_string(),
                            );
                        }
                        n
                    }
                    Err(_) => CHECKPOINT_RECONCILE_INTERVAL_DEFAULT_SECS.max(
                        derived_reconcile_grace(checkpoint_adapter, checkpoint_request_timeout_ms)
                            .as_secs(),
                    ),
                }
            },
            checkpoint_counter_safety_margin: {
                let n = optional_parsed(
                    "LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN",
                    CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT,
                )?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN: must be greater than 0 (a zero margin defeats the AOF-off recovery feature — a fresh CreateLayer could collide with a checkpoint-referenced object_id)".to_string(),
                    );
                }
                // See `CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX` for why the bound
                // is 10^9 and not the 2^53 - 1 id ceiling itself.
                if n > CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN: {n} exceeds \
                         the maximum {CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX}. A pod start can \
                         move the object-id counter up by as much as the margin, and object ids \
                         must stay below 2^53 - 1, so a larger margin uses up the id space in \
                         too few starts. The default {CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT} \
                         already covers far more layer creations than any deployment makes."
                    ));
                }
                n
            },
            // SDD SR-8.1 — stream-replay tail capture. Shipped as an explicit
            // opt-in; default promoted to `true` on 2026-09-11 (b.3n1) after the
            // dev soak, with `false` retained as the opt-out. The plain stdlib
            // `bool` parse matches every other bool knob here
            // (LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED, LIVE_EDIT_CHECKPOINT_ENABLED),
            // so only the exact strings `true` / `false` are accepted.
            checkpoint_tail_enabled: optional_parsed("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", true)?,
            checkpoint_tail_max_chain: {
                let n = optional_parsed("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN", 8u32)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN: must be greater than 0 (a chain bound of zero admits no tail at all, so every eligible tick would refuse and the feature would be silently inert while reporting itself enabled)".to_string(),
                    );
                }
                n
            },
            checkpoint_tail_max_entries: {
                let n = optional_parsed("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", 4096u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES: must be greater than 0 (a zero entry cap admits no stream entry into a tail, so every eligible tick would refuse and the feature would be silently inert while reporting itself enabled)".to_string(),
                    );
                }
                // A tail's entry count is bounded by the stream sequence span it
                // covers, and that span is already Lua-f64-bounded at 2^53 - 1
                // by both commit paths (`seq` in update_layer_fast.lua, and the
                // same ceiling on the WATCH/MULTI/EXEC path), so a cap above it
                // can never bind and is fat-fingered input. Declared block-local
                // mirroring the `LUA_F64_INTEGER_CEILING` precedent on
                // `layer_stream_maxlen` above — deliberately NOT
                // `tail_codec::TAIL_MAX_SEQUENCE`, which bounds a *sequence
                // number* rather than a count and shares this value only by
                // coincidence; config must not take an edge on the codec for a
                // numeric coincidence.
                const TAIL_MAX_ENTRIES_CEILING: u64 = (1u64 << 53) - 1;
                if n > TAIL_MAX_ENTRIES_CEILING {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES: \
                         {n} exceeds the {TAIL_MAX_ENTRIES_CEILING} ceiling \
                         (= 2^53 - 1). A tail's entry count is bounded by the \
                         stream sequence span it covers, and that span is \
                         already Lua-f64-bounded at 2^53 - 1 by both commit \
                         paths, so a cap above this can never bind. Lower the \
                         value — any realistic per-tail budget is many orders \
                         of magnitude below this ceiling."
                    ));
                }
                n
            },
            checkpoint_tail_max_bytes: {
                let n = optional_parsed("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", 67_108_864usize)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES: must be greater than 0 (a zero byte cap admits no tail record at all, so the capture path would refuse every tick and the feature would be silently inert while reporting itself enabled)".to_string(),
                    );
                }
                // Cross-field floor on the tail codec's own framing. Below one
                // header plus one record prefix plus a single payload byte, the
                // capture path's headroom derivation yields a payload budget of
                // zero: only empty-delta records would fit, so no verbatim
                // commit is representable and every capture refuses observably
                // as `refused_missing_bytes`. The `+ 1` is load-bearing — at
                // exactly the envelope size the budget is still zero. Spelled
                // via the imported constants so the numbers live in exactly one
                // place (`tail_codec`).
                const TAIL_MAX_BYTES_FLOOR: usize = TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE + 1;
                if n < TAIL_MAX_BYTES_FLOOR {
                    return Err(format!(
                        "invalid value for LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES: \
                         {n} is below the {TAIL_MAX_BYTES_FLOOR}-byte one-record \
                         floor (tail header {TAIL_HEADER_SIZE} + record prefix \
                         {TAIL_RECORD_PREFIX_SIZE} + at least one payload byte). \
                         Below it the capture path's payload budget derives to \
                         zero, so no verbatim commit is representable and every \
                         capture would refuse as refused_missing_bytes forever. \
                         Raise the value — the 67108864 default leaves an order \
                         of magnitude of headroom over measured tail sizes."
                    ));
                }
                n
            },
            checkpoint_local_disk_path,
            checkpoint_storage_api_endpoint,
            checkpoint_base_path,
            checkpoint_connect_timeout_ms,
            checkpoint_request_timeout_ms,
            checkpoint_ext_mirror_workers,
            checkpoint_ext_mirror_queue_capacity,
            service_token_file,
            service_token_poll_interval_secs: {
                let n = optional_parsed("LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS", 15u64)?;
                if n == 0 {
                    return Err(
                        "invalid value for LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS: must be greater than 0 (a zero-interval tokio::time::interval panics and a tight-loop poll would saturate the runtime)".to_string(),
                    );
                }
                n
            },
            service_token_client_credentials_enabled: optional_parsed(
                "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED",
                false,
            )?,
            // Replaced below by the plan derived for this pod's limit.
            pod_memory: PodMemoryPlan::reference(),
        };
        // The pod memory plan: every pod-memory budget from the pod memory
        // limit, after the settings it charges are parsed and before the
        // validation chain, so the chain's errors keep their order.
        let plan = PodMemoryPlan::derive(pod_memory.resolve(), &PlanInputs::from_config(&config))?;
        refuse_removed_memory_settings(&plan)?;
        let config = config.with_pod_memory_plan(plan);
        config
            .validate_shutdown_budgets()?
            .validate_upload_admission()?
            .validate_health_probe_budgets()?
            .validate_presence_lease_budgets()?
            .validate_pool_size_vs_store_values_inflight()?
            // SR-1.4 (b.tr5): reject a `rediss://` / `valkeys://` URL on a
            // binary built without a Valkey TLS feature, before fred can
            // silently fall back to a plaintext connection. Runs on the
            // already-`.trim()`-ed `valkey_url`.
            .validate_valkey_tls_compiled()?
            // SR-2.3 (b.tr5): reject an incompletely/inconsistently configured
            // passwordless Valkey auth mode. Ordered *after*
            // `validate_valkey_tls_compiled` so a TLS-uncompiled build surfaces
            // "TLS not compiled in" first, not a downstream auth-mode complaint
            // about the same URL.
            .validate_valkey_auth_mode()?
            .validate_checkpoint_budgets()
            .and_then(Config::validate_ext_reconcile_interval)
            .map(Config::warn_layer_stream_memory_footprint)
            .map(Config::warn_tail_entries_exceed_stream_retention)
            .map(Config::warn_tail_bytes_exceed_stream_retention)
            .map(Config::warn_ext_reconcile_disabled)
            .map(Config::warn_ext_value_verify_observe)
    }

    /// `self` with `plan` as its pod memory plan, its budgets copied into the
    /// fields their consumers read.
    fn with_pod_memory_plan(mut self, plan: PodMemoryPlan) -> Self {
        self.upload_memory_budget_mib = plan.upload_budget_mib();
        self.catchup_build_memory_budget_mib = plan.catchup_budget_mib();
        self.structural_cache_max_mib = plan.structural_cache_budget_mib();
        self.subscriber_max_subscriptions =
            usize::try_from(plan.max_subscriptions()).unwrap_or(usize::MAX);
        self.pod_memory = plan;
        self
    }

    /// The upload budget in bytes: `upload_memory_budget_mib` × 1 MiB,
    /// saturating.
    pub fn upload_memory_budget_bytes(&self) -> u64 {
        self.upload_memory_budget_mib.saturating_mul(1024 * 1024)
    }

    /// The upload admission gate's settings (`grpc::upload_admission`).
    pub fn upload_admission_config(&self) -> crate::grpc::upload_admission::UploadAdmissionConfig {
        crate::grpc::upload_admission::UploadAdmissionConfig {
            budget_bytes: self.upload_memory_budget_bytes(),
            wait: std::time::Duration::from_millis(self.upload_admission_wait_ms),
            max_waiters: self.upload_admission_max_waiters,
            max_upload_bytes: u64::try_from(self.max_layer_size).unwrap_or(u64::MAX),
        }
    }

    /// The external-value release budget `H` the configuration alone gives,
    /// and where it came from (b.yuo, b.qhk): `LIVE_EDIT_EXT_RELEASE_MAX_BYTES`
    /// when set ([`ExtReleaseBudgetSource::Explicit`]; `0` disables the
    /// budget), else the fallback `min(8 × LIVE_EDIT_MAX_LAYER_SIZE, 100 MB)`
    /// ([`EXT_RELEASE_BUDGET_MAX_LAYER_MULTIPLE`],
    /// [`EXT_RELEASE_BUDGET_FALLBACK_CAP_BYTES`], b.1jt) — at most eight
    /// maximum-size uploads' worth of released blobs, capped so it is safe on
    /// the smallest store that does not report its size.
    ///
    /// The fallback is open to the store: startup reads every primary's
    /// `INFO memory` (`storage::store_memory::startup_store_memory`), and the
    /// smallest `maxmemory`, ÷ 8, replaces it
    /// ([`ExtReleaseBudgetSource::StoreInfo`]) when every primary reports one
    /// above zero; otherwise, when [`Self::valkey_memory_bytes`] is set, one
    /// shard's share of it ÷ 8 does ([`ExtReleaseBudgetSource::ValkeyMemory`]),
    /// which needs the store's cluster map; otherwise the fallback stands and
    /// startup logs the reason. Both are re-read every 60 s.
    ///
    /// One eighth of one primary's memory leaves the rest of that primary for
    /// live state, streams and the sizing model's other terms; the budget is a
    /// bound on garbage, not on the store.
    pub fn ext_release_budget(&self) -> ExtReleaseBudget {
        if let Some(bytes) = self.ext_release_max_bytes {
            return ExtReleaseBudget {
                bytes,
                source: ExtReleaseBudgetSource::Explicit,
            };
        }
        ExtReleaseBudget {
            bytes: u64::try_from(self.max_layer_size)
                .unwrap_or(u64::MAX)
                .saturating_mul(EXT_RELEASE_BUDGET_MAX_LAYER_MULTIPLE)
                .min(EXT_RELEASE_BUDGET_FALLBACK_CAP_BYTES),
            source: ExtReleaseBudgetSource::Fallback,
        }
    }

    /// Log the external-value release budget in force at startup, `budget`,
    /// once (b.yuo): INFO with the budget and its source (`source` =
    /// `override`, `info` or `fallback`, the `live_edit_ext_release_budget_bytes`
    /// label), or a WARN when it is disabled.
    ///
    /// When it is the fallback, `min(8 × LIVE_EDIT_MAX_LAYER_SIZE, 100 MB)` —
    /// `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` is unset and the store's memory could
    /// not be used: no usable `maxmemory`, and `LIVE_EDIT_VALKEY_MEMORY_BYTES`
    /// unset or no reading of the store's cluster map — `fallback_problem`
    /// says why, and the line's level follows it (b.1jt): INFO for a store
    /// that reports no `maxmemory` field (expected on a managed store such as
    /// Azure Managed Redis), WARN for a fault (`maxmemory:0`, a malformed
    /// value, no routing table or primary, an `INFO` that failed or was
    /// denied) and for an unknown reason. For any other source
    /// `fallback_problem` is not read here: a problem the startup store read
    /// had is logged by `storage::store_memory::log_store_memory_startup`.
    ///
    /// A budget derived from `LIVE_EDIT_VALKEY_MEMORY_BYTES` also names the
    /// cluster map it was derived from: `shards` (slot-range entries),
    /// `endpoints` (distinct primaries, which may front several shards each),
    /// `per_shard_bytes` and `total_bytes`.
    ///
    /// Called by `main` after tracing is initialised, not from
    /// [`Self::from_env`]'s validation chain, so the line is never lost to a
    /// subscriber that does not exist yet.
    pub fn log_ext_release_budget_startup(
        &self,
        budget: &ExtReleaseBudget,
        fallback_problem: Option<&crate::storage::store_memory::StoreMemoryProblem>,
    ) {
        match budget.source {
            ExtReleaseBudgetSource::Fallback => {
                let reason = fallback_problem
                    .map_or_else(|| "no reason recorded".to_owned(), ToString::to_string);
                let fault = fallback_problem.is_none_or(|p| p.kind().is_fault());
                if fault {
                    tracing::warn!(
                        ext_release_max_bytes = budget.bytes,
                        pressure_threshold_bytes = budget.bytes / 2,
                        source = budget.source.label(),
                        max_layer_size = self.max_layer_size,
                        reason = %reason,
                        valkey_memory_bytes = self.valkey_memory_bytes,
                        "external-value release budget is the fallback, min(8 x \
                         LIVE_EDIT_MAX_LAYER_SIZE, 100 MB): LIVE_EDIT_EXT_RELEASE_MAX_BYTES is not \
                         set, and the store's memory could not be used because of a fault (see \
                         reason). The GC collects released blobs at any age once they total half \
                         the budget, and uploads carrying external values wait while they total \
                         all of it. By reason: for maxmemory 0 or a malformed maxmemory, set \
                         maxmemory on every primary; for a failed or refused INFO, let this \
                         service's user run INFO, or, on a store that does not report maxmemory \
                         (Azure Managed Redis), set LIVE_EDIT_VALKEY_MEMORY_BYTES to the store's \
                         total memory, which needs only the cluster map; for an unknown routing \
                         table or one with no primary, check the store's cluster state (it \
                         usually clears at a later read). Or set LIVE_EDIT_EXT_RELEASE_MAX_BYTES \
                         for a fixed budget. The store is asked again every 60 s"
                    );
                } else {
                    tracing::info!(
                        ext_release_max_bytes = budget.bytes,
                        pressure_threshold_bytes = budget.bytes / 2,
                        source = budget.source.label(),
                        max_layer_size = self.max_layer_size,
                        reason = %reason,
                        "external-value release budget is the fallback, min(8 x \
                         LIVE_EDIT_MAX_LAYER_SIZE, 100 MB): the store does not report its size \
                         (expected on a managed store such as Azure Managed Redis) and neither \
                         LIVE_EDIT_EXT_RELEASE_MAX_BYTES nor LIVE_EDIT_VALKEY_MEMORY_BYTES is set. \
                         The GC collects released blobs at any age once they total half the \
                         budget, and uploads carrying external values wait while they total all \
                         of it. On such a store, setting LIVE_EDIT_VALKEY_MEMORY_BYTES to the \
                         store's total memory as the provider states it (for example \
                         12000000000 for a 12 GB cache) is recommended, for the release budget \
                         and for live_edit_store_memory_used_ratio; it is not required"
                    );
                }
            }
            _ if budget.bytes == 0 => tracing::warn!(
                source = budget.source.label(),
                "LIVE_EDIT_EXT_RELEASE_MAX_BYTES=0 disables the external-value release budget: \
                 uploads never wait for garbage collection, and released blobs awaiting it are \
                 bounded only by the GC loop's pace, so a large-value workload can fill a \
                 noeviction store"
            ),
            source => tracing::info!(
                ext_release_max_bytes = budget.bytes,
                source = source.label(),
                derived_from = ?source,
                valkey_memory_bytes = self.valkey_memory_bytes,
                shards = source.cluster_map().map(|(d, _)| d.shards),
                endpoints = source.cluster_map().map(|(_, endpoints)| endpoints),
                per_shard_bytes = source.cluster_map().map(|(d, _)| d.per_shard_bytes),
                total_bytes = source.cluster_map().map(|(d, _)| d.total_bytes),
                rpc_deadline_ms = self.rpc_deadline_ms,
                deadline_margin_ms = u64::try_from(
                    crate::storage::ext_release_gate::EXT_RELEASE_GATE_DEADLINE_MARGIN.as_millis()
                )
                .unwrap_or(u64::MAX),
                "external-value release budget configured; uploads carrying external values \
                 wait at it for up to their remaining deadline less the margin"
            ),
        }
    }

    /// Reject an upload admission configuration that cannot work: a waiter
    /// cap of 0, which would make any non-zero wait dead code (fail-fast is
    /// `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS=0`).
    ///
    /// The budget needs no check: the pod memory plan derives it at or above
    /// the floor `F + A + 2 × LIVE_EDIT_MAX_LAYER_SIZE`, so no upload under the
    /// size cap is refused for its size. An explicitly set admission wait at or
    /// above `LIVE_EDIT_RPC_DEADLINE_MS` is refused where it is parsed, in
    /// [`Self::from_env`]: only an explicit value can fail, since the unset
    /// default is derived below every non-zero deadline.
    fn validate_upload_admission(self) -> Result<Self, String> {
        if self.upload_admission_max_waiters == 0 {
            return Err(
                "invalid value for LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS: must be greater than 0 \
                 (a zero waiter cap makes any non-zero LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS dead \
                 code). Express fail-fast with LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS=0 instead."
                    .to_string(),
            );
        }
        Ok(self)
    }

    /// Cross-validate `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS` against the
    /// **derived** reconciler grace (b.yda).
    ///
    /// An interval shorter than the grace cannot repair anything: a candidate
    /// becomes claimable only a full grace after its first observation, so every
    /// pass would register and none would ever act — a reconciler that runs
    /// constantly and reconciles nothing, which is worse than one that is
    /// switched off, because it looks alive. Reject rather than clamp: an
    /// operator who set a short interval wanted more frequent repair, and
    /// silently running at the grace instead would hide that they cannot have it.
    ///
    /// `0` is the documented off switch and is exempt — see
    /// [`Self::warn_ext_reconcile_disabled`].
    ///
    /// Only an EXPLICIT value can fail this: the unset default is
    /// `max(EXT_RECONCILE_INTERVAL_DEFAULT_SECS, grace)`, which satisfies the
    /// bound by construction at every RPC deadline.
    fn validate_ext_reconcile_interval(self) -> Result<Self, String> {
        if self.ext_reconcile_interval_secs == 0 {
            return Ok(self);
        }
        let grace = crate::storage::ext_reconcile::ext_reconcile_grace(self.rpc_deadline_ms);
        if self.ext_reconcile_interval_secs < grace.as_secs() {
            return Err(format!(
                "invalid value for LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS: {} is shorter than the \
                 derived reconciler grace of {}s (max(600, 10 x LIVE_EDIT_RPC_DEADLINE_MS={})). A \
                 pass shorter than the grace can never claim a candidate it registered, so it \
                 would repair nothing at all. Raise the interval to at least the grace, unset it \
                 to take the derived default of {}s, or set 0 to disable the reconciler \
                 deliberately",
                self.ext_reconcile_interval_secs,
                grace.as_secs(),
                self.rpc_deadline_ms,
                EXT_RECONCILE_INTERVAL_DEFAULT_SECS.max(grace.as_secs()),
            ));
        }
        Ok(self)
    }

    /// Emit a startup WARN (not a rejection) when the reference-count
    /// reconciler is switched off (b.yda).
    ///
    /// `0` is a supported, deliberate off switch — the
    /// `LIVE_EDIT_AUTH_USERINFO_NEGATIVE_TTL_SECS` precedent — but it disables
    /// the ONLY thing in the service that drains the documented leak-direction
    /// `ref_count` residuals, so it must not be silent. A deployment that runs
    /// this way will accumulate `ext:meta` entries no live object references,
    /// with no error and no metric movement to see it by.
    fn warn_ext_reconcile_disabled(self) -> Self {
        if self.ext_reconcile_interval_secs == 0 {
            tracing::warn!(
                "LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0 disables the external-value ref_count \
                 reconciler. Nothing else in the service compares ext:meta ref_count against the \
                 objects that actually reference each hash, so the documented leak-direction \
                 residuals (among them a crash between a commit's EXEC and its release \
                 enqueue, a DR restore's suppressed surplus, references stranded by corrupt \
                 keyspace, and an UpdateLayer's owed releases when its reply was lost and the call \
                 could not recover them itself: its commit release entry was evicted from the \
                 object's ring, the call was dropped before its lookup, the call ended in a way \
                 that runs no lookup, or the lookup's read failed) will accumulate unbounded on a \
                 noeviction store"
            );
        }
        self
    }

    /// Emit a startup WARN (not a rejection) when the upload content check runs
    /// in `observe` mode (SDD §11.1).
    ///
    /// `observe` is a supported, deliberate relaxation — the rollout setting for
    /// clients not yet confirmed to compute the content hash — but it lets a
    /// value that fails the check become the first, and so the only, copy every
    /// reader of its hash gets, so it must not be silent. Nothing else signals
    /// it while no mismatch occurs: the per-call WARN and the
    /// `live_edit_ext_value_verify_total` mismatch cells move only when a
    /// failing value actually arrives. `enforce` emits nothing.
    fn warn_ext_value_verify_observe(self) -> Self {
        if self.ext_value_verify == ExtValueVerifyMode::Observe {
            tracing::warn!(
                "LIVE_EDIT_EXT_VALUE_VERIFY=observe: external values that fail the upload content \
                 check (size or content hash) are counted and stored instead of refused. A \
                 failing value stored first under a hash is the copy every reader of that hash \
                 gets, and clients that check content refuse it on read. Return to enforce once \
                 live_edit_ext_value_verify_total shows no mismatches"
            );
        }
        self
    }

    /// Emit a startup WARN (not a rejection) when the configured per-tail entry
    /// cap exceeds what the live stream actually retains (SDD SR-8.1).
    ///
    /// A tail is captured by replaying entries out of
    /// `le:{object_id}:obj:stream`, which is bounded by `MAXLEN ~
    /// layer_stream_maxlen`. So a tail can never be longer than stream
    /// retention allows: when `checkpoint_tail_max_entries` exceeds
    /// `layer_stream_maxlen` the effective cap is the retention depth and the
    /// configured number is misleading — it reads like a capacity the
    /// deployment has when it does not.
    ///
    /// WARN rather than reject, matching
    /// `warn_layer_stream_memory_footprint`: the pairing is inconsistent, not
    /// unsafe. Capture stays correct (it simply yields fewer entries than the
    /// cap allows), and an operator mid-way through raising both knobs should
    /// not be locked out of booting.
    ///
    /// Fires **unconditionally** — deliberately NOT gated on
    /// `checkpoint_tail_enabled`. This is a config-consistency warning, the
    /// operator may flip the feature flag at any time, and gating would hide
    /// the inconsistency until exactly the moment it starts mattering. The
    /// shipped defaults (4096 == 4096) make it silent out of the box.
    fn warn_tail_entries_exceed_stream_retention(self) -> Self {
        if self.checkpoint_tail_max_entries > self.layer_stream_maxlen {
            tracing::warn!(
                checkpoint_tail_max_entries = self.checkpoint_tail_max_entries,
                layer_stream_maxlen = self.layer_stream_maxlen,
                "configured per-tail entry cap \
                 (LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES) exceeds live-stream \
                 retention (LIVE_EDIT_LAYER_STREAM_MAXLEN): a tail is replayed \
                 out of the per-object stream, so it can never be longer than \
                 retention allows. The effective cap is the retention depth and \
                 the configured number is misleading. Raise \
                 LIVE_EDIT_LAYER_STREAM_MAXLEN to match, or lower \
                 LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES."
            );
        }
        self
    }

    /// Emit a startup WARN (not a rejection) when a tail's byte cap exceeds
    /// what the live stream retains in bytes: `checkpoint_tail_max_bytes` >
    /// `layer_stream_max_bytes` > 0.
    ///
    /// A tail is captured out of `le:{object_id}:obj:stream`, whose relay
    /// bytes the byte budget bounds (`storage::relay_retention`), so a tail
    /// can never carry more relay bytes than the stream retains: the larger
    /// cap does nothing, and reads like a capacity the deployment does not
    /// have. The byte twin of `warn_tail_entries_exceed_stream_retention`,
    /// WARN rather than reject for the same reasons, and unconditional like
    /// it. Silent with the byte budget off (`0`), and at the shipped defaults
    /// (64 MiB tail cap, 128 MiB budget).
    fn warn_tail_bytes_exceed_stream_retention(self) -> Self {
        if self.layer_stream_max_bytes > 0
            && self.checkpoint_tail_max_bytes as u64 > self.layer_stream_max_bytes
        {
            tracing::warn!(
                checkpoint_tail_max_bytes = self.checkpoint_tail_max_bytes,
                layer_stream_max_bytes = self.layer_stream_max_bytes,
                "configured per-tail byte cap \
                 (LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES) exceeds the live stream's \
                 byte budget (LIVE_EDIT_LAYER_STREAM_MAX_BYTES): a tail is \
                 replayed out of the per-object stream, so it can never carry \
                 more relay bytes than the stream retains. The effective cap is \
                 the byte budget and the configured number is misleading. Raise \
                 LIVE_EDIT_LAYER_STREAM_MAX_BYTES to match, or lower \
                 LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES."
            );
        }
        self
    }

    /// Emit a startup WARN (not a rejection) when the worst-case per-object
    /// live-stream memory footprint is alarmingly large (b.gze follow-up).
    ///
    /// Since b.gze, every entry on `le:{object_id}:obj:stream` carries the
    /// writer's verbatim relay `ompf` field — up to `LIVE_EDIT_MAX_LAYER_SIZE`
    /// of OMPF bytes. The worst case is every retained entry a maximum-size
    /// upload: with the relay byte budget on (`layer_stream_max_bytes` > 0)
    /// a single hot object retains at most
    /// `min(layer_stream_maxlen × max_layer_size, layer_stream_max_bytes + 2 ×
    /// max_layer_size)` relay bytes (the budget, plus the two entries a trim
    /// always keeps; `storage::relay_retention`), and with it off
    /// `layer_stream_maxlen × max_layer_size`.
    ///
    /// This is deliberately a WARN, not a hard reject:
    /// - The worst case is theoretical. Real edit deltas are far smaller than
    ///   `max_layer_size` (which caps any one upload, CreateLayer and
    ///   UpdateLayer alike, so it is sized for the largest upload, not the
    ///   typical edit delta), and `MAXLEN ~ N` is an upper bound on retained
    ///   entries, not a steady-state count.
    /// - Legitimate deployments may intentionally pair a large
    ///   `MAX_LAYER_SIZE` with a deep stream and a large byte budget and have
    ///   provisioned Valkey memory accordingly. Rejecting would break those
    ///   operators.
    ///
    /// Threshold: 4 GiB per object. Defensible because it is roughly the point
    /// at which a single object's worst-case stream rivals the whole-instance
    /// memory budget of a modest Valkey node. Silent at the defaults (128 MiB
    /// plus 2 × 100 MB ≈ 334 MB); with the byte budget off the default pairing
    /// (4096 × 100 MB ≈ 400 GB) *will* warn, flagging that count-only
    /// retention is safe only while real deltas are tiny. The constant is a
    /// heuristic for a log line, so a hardcoded value with this justification
    /// is sufficient — no new env var.
    fn warn_layer_stream_memory_footprint(self) -> Self {
        const WORST_CASE_FOOTPRINT_WARN_BYTES: u128 = 4 * 1024 * 1024 * 1024;
        let max_layer_size = self.max_layer_size as u128;
        let count_bound = (self.layer_stream_maxlen as u128).saturating_mul(max_layer_size);
        let worst_case = if self.layer_stream_max_bytes > 0 {
            let byte_bound = (self.layer_stream_max_bytes as u128)
                .saturating_add(max_layer_size.saturating_mul(2));
            // An entry cap of 0 is no cap: only the byte bound applies.
            if self.layer_stream_maxlen == 0 {
                byte_bound
            } else {
                count_bound.min(byte_bound)
            }
        } else {
            count_bound
        };
        if worst_case > WORST_CASE_FOOTPRINT_WARN_BYTES {
            tracing::warn!(
                layer_stream_maxlen = self.layer_stream_maxlen,
                layer_stream_max_bytes = self.layer_stream_max_bytes,
                max_layer_size = self.max_layer_size,
                worst_case_bytes = %worst_case,
                threshold_bytes = %WORST_CASE_FOOTPRINT_WARN_BYTES,
                "worst-case per-object live-stream footprint is large: each \
                 stream entry carries the writer's verbatim relay OMPF, \
                 so a single hot object could retain this many bytes in Valkey \
                 if every retained entry is a maximum-size upload — the smaller \
                 of LIVE_EDIT_LAYER_STREAM_MAXLEN × LIVE_EDIT_MAX_LAYER_SIZE and \
                 LIVE_EDIT_LAYER_STREAM_MAX_BYTES + 2 × LIVE_EDIT_MAX_LAYER_SIZE \
                 (the first alone when LIVE_EDIT_LAYER_STREAM_MAX_BYTES is 0). \
                 This worst case is theoretical (real edit deltas are far \
                 smaller than max_layer_size); ensure Valkey is provisioned for \
                 it or lower LIVE_EDIT_LAYER_STREAM_MAX_BYTES / \
                 LIVE_EDIT_LAYER_STREAM_MAXLEN / LIVE_EDIT_MAX_LAYER_SIZE."
            );
        }
        self
    }

    /// Enforce that the presence lease TTL leaves slack for at least one
    /// missed refresh tick before the sweeper starts evicting (b.w29).
    ///
    /// Pre-b.w29 the defaults were TTL=30 / refresh=10 and the sweeper ran
    /// every 15 s. A single tokio scheduling pause of 20 s between two
    /// refresh attempts (e.g., a stop-the-world allocator backpressure
    /// event) left the lease expired by the time the next refresh fired,
    /// the sweeper saw the missing lease, and a perfectly healthy session
    /// was evicted. The post-fix defaults (60 / 15) give a 4× ratio so a
    /// missed refresh still has one more chance before the lease lapses.
    ///
    /// Encoded rule of thumb: `refresh × 2 ≤ ttl`. Operators who flip both
    /// knobs must keep them inside this band or accept eviction noise.
    fn validate_presence_lease_budgets(self) -> Result<Self, String> {
        let ttl = self.presence_lease_ttl_secs;
        let refresh = self.presence_refresh_interval_secs;
        if refresh.saturating_mul(2) > ttl {
            return Err(format!(
                "invalid configuration: LIVE_EDIT_PRESENCE_LEASE_TTL_SECS \
                 ({ttl}) must be at least 2 × LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS \
                 ({refresh}) so a single missed refresh tick (e.g. due to a \
                 tokio scheduling pause) does not let the lease expire before \
                 the next refresh runs. Got {ttl} < {floor}. Raise the TTL or \
                 lower the refresh interval.",
                ttl = ttl,
                refresh = refresh,
                floor = refresh.saturating_mul(2),
            ));
        }
        Ok(self)
    }

    /// Enforce the health-probe budget invariant (b.avd item 9).
    ///
    /// `health_probe_timeout_ms` is the per-probe deadline; if it exceeds
    /// the inter-tick window (`health_probe_interval_ms`), a single slow
    /// probe queues past the next scheduled tick and the supervisor's
    /// `interval()` falls behind. Fail fast at boot rather than silently
    /// every probe cycle.
    fn validate_health_probe_budgets(self) -> Result<Self, String> {
        if self.health_probe_timeout_ms >= self.health_probe_interval_ms {
            return Err(format!(
                "invalid configuration: LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS \
                 ({timeout}) must be strictly less than \
                 LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS ({interval}) so a slow \
                 probe does not queue past the next tick.",
                timeout = self.health_probe_timeout_ms,
                interval = self.health_probe_interval_ms,
            ));
        }
        Ok(self)
    }

    /// Enforce the post-server shutdown-budget invariant (b.3z7 non-blocking
    /// nit from PM review).
    ///
    /// `shutdown_overall_budget_secs` wraps the whole post-server sequence
    /// in a single `tokio::time::timeout`; the metrics-server abort and the
    /// provider flush run after it, outside the budget. In order, the
    /// sequence is: the layer-cleanup wait; the JoinSession and
    /// SubscribeReadLayer task drains (concurrently); the session-checkpoint
    /// drain; firing every background task's cancellation token; the
    /// background-task joins (one concurrent group, then the ref-count
    /// reconciler, the checkpoint loop and each ext-mirror worker in turn);
    /// and `ValkeyClient::quit()`. Every wait before `quit()` has its own
    /// `shutdown_timeout_secs` budget. The steps with no timeout of their
    /// own are the cancellation fires, the tracker close and the
    /// result reporting between the waits; `quit()` is bounded by its
    /// internal bounds below. A budget below the floor fails at boot rather
    /// than at shutdown, where the outer wrapper could fire before the
    /// per-task waits and `quit()` complete. A raise of `timeout_secs`
    /// that stays above the floor still boots, and makes a "shutdown overall
    /// budget exceeded" warning more likely on a shutdown whose waits run long
    /// (a healthy shutdown finishes well inside the budget).
    ///
    /// The floor is `timeout_secs +` [`MIN_SHUTDOWN_BUDGET_HEADROOM_SECS`]
    /// (15 s): room for one full per-task wait plus 15 s. The waits run one
    /// after another, so the floor is not their worst-case sum. 15 s covers
    /// `ValkeyClient::quit()`'s sequential internal bounds — 2 s
    /// member-replacer drain, 2 s router-supervisor drain, 2 s slot-recycle
    /// drain (the snapshot-fence guard cleanup drain runs beside it and is no
    /// longer), 2 s UNWATCH drain, 1 s stream-multiplexer grace, then 2.5 s
    /// for the parallel QUITs (each pool's 2 s `QUIT_TIMEOUT`; a stream
    /// reader's retire may first wait 0.5 s for its router watcher before its
    /// own 2 s `QUIT`): 11.5 s in all — plus slack for the steps with no
    /// timeout of their own. The `const` assertion after
    /// `STREAM_MUX_SHUTDOWN_GRACE` in `storage::client` checks that sum against
    /// the headroom at build time.
    ///
    /// Whether the whole shutdown fits the pod's termination grace is not
    /// checked here — the binary does not know the grace. The chart refuses
    /// to render a grace below the minimum, and the startup log line reports
    /// the minimum this configuration needs.
    fn validate_shutdown_budgets(self) -> Result<Self, String> {
        let floor = self
            .shutdown_timeout_secs
            .saturating_add(MIN_SHUTDOWN_BUDGET_HEADROOM_SECS);
        if self.shutdown_overall_budget_secs < floor {
            return Err(format!(
                "invalid configuration: LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS \
                 ({overall}) must be at least \
                 LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS ({timeout}) + \
                 {headroom} s of headroom for Valkey quit() and the \
                 shutdown steps with no timeout of their own. \
                 Got {overall} < {floor}. \
                 Raise LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS or lower \
                 LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS.",
                overall = self.shutdown_overall_budget_secs,
                timeout = self.shutdown_timeout_secs,
                headroom = MIN_SHUTDOWN_BUDGET_HEADROOM_SECS,
                floor = floor,
            ));
        }
        Ok(self)
    }

    /// Enforce that the configured Valkey pool size is large enough to
    /// fit a full `store_values` wave (b.avd second follow-up; rationale
    /// re-argued for the data pool by b.rej).
    ///
    /// `ValkeyExternalValueStore::store_values` issues at most
    /// `STORE_VALUES_MAX_INFLIGHT` (currently 8) per-hash blob writes
    /// concurrently. Post-b.rej each is one `EVALSHA` on the MULTIPLEXED
    /// data pool rather than a `MULTI/EXEC` holding an exclusive
    /// `txn_pool` slot, and `valkey_pool_size` sizes both pools (see
    /// `ValkeyClient::connect` in `src/storage/client.rs`), so the bound
    /// still binds — for a different reason. A multi-MB blob occupies the
    /// connection fred routed it onto for the whole of its transfer, and
    /// every read and every Lua fast-path commit multiplexed onto that
    /// same connection waits behind it. What this gate enforces is that
    /// the pool is AT LEAST AS WIDE AS one wave — so a wave can never
    /// exceed the pool it has to fit into. Below the floor it does: a
    /// single big `CreateLayer` upload has a blob on every connection in
    /// the pool with more still queued behind them, so concurrent
    /// `UpdateLayer` / `CreateSession` callers head-of-line-block behind
    /// the wave — turning the pool's tail latency into a stall hazard with
    /// no operational surface to discover it.
    ///
    /// Equality is accepted and is NOT claimed to be comfortable: at
    /// exactly the floor a full wave can momentarily occupy every
    /// data-pool connection, and reads and Lua fast-path commits queue
    /// behind blob transfers until it drains. Reaching that boundary takes
    /// either an explicit `LIVE_EDIT_VALKEY_POOL_SIZE=8` or a host
    /// reporting ≤ 4 cores on the auto path — including a CPU-limited
    /// container, since `available_parallelism` honours the cgroup quota —
    /// where `effective_pool_size` floors at exactly the
    /// same bound. The floor is a hard lower limit, not a sizing
    /// recommendation — production should sit well above it, and the Helm
    /// chart sets `64`.
    ///
    /// Only validate when the operator set an explicit non-zero
    /// `LIVE_EDIT_VALKEY_POOL_SIZE`. Zero means "auto = cores × 2"
    /// (resolved at connect time by `effective_pool_size`); for the
    /// auto path we additionally floor the resolved size at the
    /// inflight bound so a host reporting ≤ 4 cores (auto → 2..8) still
    /// satisfies the invariant. That floor is the runtime safety net for the auto
    /// path; this startup check is the misconfiguration safety net for
    /// the explicit path.
    ///
    /// The current production defaults (auto on dev, 64 on prod Helm)
    /// both already clear the bound by a wide margin, so this gate is
    /// defensive — but it makes the operational contract explicit and
    /// fails the pod at boot rather than silently degrading throughput
    /// under load.
    fn validate_pool_size_vs_store_values_inflight(self) -> Result<Self, String> {
        let floor = u32::try_from(crate::storage::external_values::STORE_VALUES_MAX_INFLIGHT)
            .expect("STORE_VALUES_MAX_INFLIGHT must fit in u32 — it is a small bounded constant");
        if self.valkey_pool_size > 0 && self.valkey_pool_size < floor {
            return Err(format!(
                "invalid configuration: LIVE_EDIT_VALKEY_POOL_SIZE \
                 ({configured}) must be at least {floor} so the data pool \
                 is at least as wide as a single store_values wave \
                 (capacity STORE_VALUES_MAX_INFLIGHT = {floor}) and the \
                 wave can never exceed the pool it has to fit into, \
                 head-of-line-blocking concurrent UpdateLayer / \
                 CreateSession callers behind blob transfers with more \
                 blobs still queued. \
                 Raise LIVE_EDIT_VALKEY_POOL_SIZE to at least {floor}, or \
                 unset it to enable the auto = cores × 2 default which is \
                 floored at the same bound. {floor} is a hard lower limit, \
                 not a sizing recommendation — at exactly that value a full \
                 wave can momentarily occupy every connection, so set it \
                 well above the floor in production (the Helm chart uses \
                 64).",
                configured = self.valkey_pool_size,
                floor = floor,
            ));
        }
        Ok(self)
    }

    /// Enforce that the checkpoint tick cadence stays strictly shorter than
    /// the inter-pass interval (b.fgj Epic 1), and that the reconciliation
    /// cadence is at least the derived reconciliation grace (b.9qd).
    ///
    /// The periodic checkpoint loop ticks every `checkpoint_tick_secs` to
    /// re-evaluate which objects have quiesced; a full pass kicks off every
    /// `checkpoint_interval_secs`. If the tick window is greater than or
    /// equal to the interval, the loop never gets a chance to poll
    /// quiesce-watch state between passes — passes either fire on every
    /// tick (tick = interval) or skip entirely (tick > interval), depending
    /// on how the scheduler aligns. Fail fast at boot rather than letting
    /// the misconfiguration alter steady-state cadence in production.
    ///
    /// The reconciliation bound is the same class of failure with a quieter
    /// symptom. That pass acts on a candidate only on the **second**
    /// observation at least one grace window after the first — that is the
    /// entire safety argument, since no adapter offers a trustworthy artifact
    /// timestamp. With the pass interval below the grace, no candidate can ever
    /// be observed twice a grace apart before the next pass re-observes it
    /// inside the window, so every pass defers everything and the sweep spends
    /// a full namespace enumeration to reclaim nothing, forever, with no error.
    fn validate_checkpoint_budgets(self) -> Result<Self, String> {
        if self.checkpoint_tick_secs >= self.checkpoint_interval_secs {
            return Err(format!(
                "invalid configuration: LIVE_EDIT_CHECKPOINT_TICK_SECS \
                 ({tick}) must be strictly less than \
                 LIVE_EDIT_CHECKPOINT_INTERVAL_SECS ({interval}) so the \
                 checkpoint loop can poll quiesce-watch state between \
                 passes. With tick >= interval, passes either fire on \
                 every tick or skip entirely depending on how the \
                 scheduler aligns — either way, the configured cadence \
                 is not what the operator asked for.",
                tick = self.checkpoint_tick_secs,
                interval = self.checkpoint_interval_secs,
            ));
        }
        let grace = self.checkpoint_reconcile_grace();
        let reconcile = Duration::from_secs(self.checkpoint_reconcile_interval_secs);
        if reconcile < grace {
            return Err(format!(
                "invalid configuration: LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS \
                 ({reconcile_secs}) must be at least the derived reconciliation grace \
                 ({grace_secs}s) for the configured checkpoint adapter \
                 ({adapter:?}). The grace is {multiple} × that adapter's own \
                 worst-case single-artifact publish (floored at {floor_secs}s) and is \
                 deliberately not configurable — it is what separates 'this \
                 artifact's sibling publish is still in flight' from 'this \
                 artifact's publish died'. A pass interval below it means no \
                 candidate is ever observed twice a grace apart, so the sweep \
                 defers everything forever and reclaims nothing while still \
                 paying for a full namespace enumeration each pass. On this \
                 configuration the grace resolves from \
                 LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS={timeout_ms}. Raise \
                 the interval to at least {grace_secs}, or unset the variable \
                 to take the derived default of {default_secs}s.",
                reconcile_secs = self.checkpoint_reconcile_interval_secs,
                grace_secs = grace.as_secs(),
                adapter = self.checkpoint_adapter,
                multiple = crate::checkpoint::store::RECONCILE_GRACE_MULTIPLE,
                floor_secs = crate::checkpoint::store::RECONCILE_GRACE_FLOOR.as_secs(),
                timeout_ms = self.checkpoint_request_timeout_ms,
                // What unsetting the variable would actually yield — NOT the
                // bare baseline. Advising "unset it for the 3600s default" is
                // wrong advice whenever the grace exceeds 3600, because that
                // default would fail this very check.
                default_secs = CHECKPOINT_RECONCILE_INTERVAL_DEFAULT_SECS.max(grace.as_secs()),
            ));
        }
        Ok(self)
    }

    /// The reconciliation grace window this configuration implies.
    ///
    /// Resolved from the configured adapter's publish arithmetic, not stored:
    /// `StorageApi` derives its bound from `checkpoint_request_timeout_ms` (the
    /// same product `StorageApiCheckpointStore::publish_worst_case` reports at
    /// runtime, so config-time validation and runtime behaviour cannot drift),
    /// and the local adapters take the conservative floor. Exposed on `Config`
    /// so `validate_checkpoint_budgets` can enforce the bound before any adapter
    /// has been constructed.
    pub fn checkpoint_reconcile_grace(&self) -> Duration {
        derived_reconcile_grace(self.checkpoint_adapter, self.checkpoint_request_timeout_ms)
    }

    /// The enforced per-adapter-call wall-clock ceiling this configuration
    /// implies (b.gg7).
    ///
    /// Same relationship to `CheckpointStore::publish_worst_case` that
    /// [`Self::checkpoint_reconcile_grace`] has to the reconciliation pass:
    /// one expression, evaluated either from a `Config` (startup validation,
    /// config-echo gauges, the startup INFO line) or from a constructed adapter
    /// (the `tokio::time::timeout` that actually enforces it), and structurally
    /// unable to disagree because both route through
    /// [`derived_adapter_call_ceiling`].
    pub fn checkpoint_adapter_call_ceiling(&self) -> Duration {
        derived_adapter_call_ceiling(self.checkpoint_adapter, self.checkpoint_request_timeout_ms)
    }

    /// Wall-clock budget a **lazy-recovery** caller gives a checkpoint-store
    /// read before abandoning it (b.gg7 review round 2).
    ///
    /// `max(rpc_deadline_ms, 2 × checkpoint_request_timeout_ms)` — 60 s at the
    /// defaults. Both terms are load-bearing and the `max` is what makes the
    /// budget correct under either tuning direction.
    ///
    /// **Why a floor at all.** The two recovery hooks
    /// (`RecoveryHook::try_restore_layer`, `SessionRecoveryHook::try_recover_
    /// session`) are reached from the *deadline-exempt* RPCs —
    /// `SubscribeReadLayer` and `JoinSession` — which the tower deadline layer
    /// deliberately does not bound (`EXEMPT_FROM_DEADLINE` in
    /// `src/grpc/metrics_layer.rs`). Something has to, because both hooks hold
    /// a K=1 admission mutex across the fetch, so an unbounded read costs every
    /// waiter for the same layer/session the adapter's full enumeration ceiling
    /// in series.
    ///
    /// **A third call site, also off the RPC clock.** `b.b2j`'s
    /// `CheckpointObjectIdCounterRepair` uses this same budget for its
    /// counters-sidecar read even though it is reached from `CreateLayer`, which
    /// is NOT deadline-exempt. That is consistent, not an exception: the repair
    /// runs the read on a **detached** `tokio::spawn`ed task precisely so no RPC
    /// deadline can drop it mid-fetch, and the requesting handler waits on the
    /// separate, strictly-sub-deadline
    /// [`Self::object_id_counter_repair_wait_budget`]. So the invariant this
    /// budget assumes — "nothing shorter is already bounding this fetch" — holds
    /// at all three sites; what differs is only *why*. That sub-deadline wait
    /// shapes the *caller's* error, not the fetch: a `CreateLayer` that
    /// allocates late in its deadline window — or under a shorter client
    /// `grpc-timeout` — is dropped by the deadline layer mid-wait anyway, and
    /// the detached read still runs to completion on this budget (SDD §22.18).
    ///
    /// **Why not `rpc_deadline_ms` alone.** The layer hook's read is a
    /// *multi-request* sequence on the StorageApi adapter: one paginated LIST,
    /// then — since Epic 4's composed restore (SR-3.1) — up to one artifact GET
    /// **and** one meta GET per lineage *member*, a member being the base or one
    /// of the tails replayed onto it. SR-4.2 sizes
    /// `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS` (30 s, same as the default RPC
    /// deadline) so that the OMPF GET *alone* may legitimately consume all of
    /// it. A budget of one `rpc_deadline_ms` therefore has negative headroom on
    /// exactly the large layers DR exists for: the fetch would elapse
    /// deterministically, every recovery would return `UNAVAILABLE`, and the
    /// client would reconnect into the same wall — a retry storm dressed as a
    /// timeout. `2 ×` the per-request timeout buys the first member fetch its
    /// full design budget plus one more for the LIST and the walk around it,
    /// without pretending an arbitrary number of retries fits.
    ///
    /// **Why the member count does not multiply this number.** A composed
    /// lineage holds up to `K + 1` members and 60 s does not fund `K + 1`
    /// fetches — deliberately, because multiplying the budget by `K` would hand
    /// straight back the head-of-line stall on the per-`object_id` admission
    /// mutex that b.gg7 added it to cap. Instead this budget is now **both** a
    /// preemptive cap *and* a cooperative step-down trigger:
    /// `checkpoint::lineage::gather_restore_lineage` takes this value as a
    /// deadline alongside a per-member reserve of one
    /// `checkpoint_request_timeout_ms`, and a candidate lineage that cannot fund
    /// its next member fetch inside the remaining budget is **abandoned and
    /// stepped down** to the next older restorable generation rather than
    /// erroring. Running out of time is a step-down signal, not a stranded
    /// recovery.
    ///
    /// The reserve arithmetic is what makes that safe, and it fits at the
    /// shipped defaults: budget `= max(30 s, 2 × 30 s) = 60 s`, reserve
    /// `= 30 s`. The formula's `2 ×` term is why the budget is **always** at
    /// least two reserves, whatever the two knobs are set to — which is exactly
    /// the `budget >= 2 × reserve` precondition the gather states. A
    /// candidate's first member fetch is un-gated until the selection has
    /// abandoned something — un-gated meaning *not charged against the base's
    /// reservation*, not un-capped: it is still bounded by the live remainder
    /// like every other fetch. So an untroubled base-newest listing costs
    /// exactly one such fetch, at up to one full reserve — the pre-Epic-4 cost
    /// exactly.
    ///
    /// **How the floor answer is guaranteed — accounting, not a bigger
    /// number.** The floor answer is the newest base alone, byte-for-byte the
    /// pre-Epic-4 selection result. The gather reserves its fetch out of this
    /// budget *by member kind*, at the point of spend: a `Base` fetch is capped
    /// at `min(reserve, remaining)` and is never declined while any budget is
    /// left, while a speculative `Tail` fetch is capped at
    /// `min(reserve, remaining − reserve)` and so can never take the remainder
    /// below the base's reservation. However many **allowanced** members wedge
    /// above it, the base the walk reaches is therefore funded with a *full*
    /// reserve — the tail cap is taken against the live remainder, so the LIST
    /// and the other un-capped wall clock above it are already netted out of it
    /// and are not charged to the base a second time. The selection's one
    /// un-gated first fetch is the exception: when it lands on a tail that then
    /// wedges, the base behind it gets only `min(reserve, remaining)`, short of
    /// a full reserve by the LIST plus that fetch's own cost. The guarantee is
    /// deliberately stated with that exception spelled out rather than as a
    /// flat "the base always gets a full reserve": on that one arm the residue
    /// is real, it is just not proportional to how badly the adapter is
    /// behaving.
    ///
    /// This is why the fix for b.bu8 was accounting rather than a larger
    /// budget. The b.gg7 argument above is unchanged and now load-bearing in
    /// the other direction: enlarging this budget to buy the guarantee headroom
    /// would hand straight back the head-of-line stall it exists to cap (and a
    /// `3 ×` formula would only convert a one-wedged-member loss into a
    /// two-wedged-member loss). The per-fetch cap remains required for its own
    /// reason — the adapter's enforced per-call ceiling
    /// ([`Self::checkpoint_adapter_call_ceiling`], 180 s at stock knobs) is 6×
    /// the reserve, so one uncapped degraded fetch would outlive this whole
    /// budget and elapse the preemptive wrapper before any cooperative check
    /// ran. Each declined candidate still records its own budget step-down (a
    /// verdict, never an error), every cap — the un-gated first fetch included
    /// — is bounded by the live remainder (see `gather_restore_lineage`'s
    /// property (i) for the exact deadline-precision statement), and the
    /// preemptive wrapper remains the backstop. That bound on
    /// the first fetch is what keeps the claim true when the LIST is slow: the
    /// adapter's enumeration is bounded by its own 540 s ceiling rather than by
    /// this budget, so a LIST can return with less than one reserve left, and
    /// the first fetch is then clamped to the live remainder instead of running
    /// a flat reserve regardless of it. The degenerate outcome there — the base
    /// behind a wedged first fetch declined on a zero allowance, or funded on a
    /// near-zero one — is documented at `gather_restore_lineage`; it is a
    /// step-down either way, never a stranded error. The caller's answer on
    /// that trace is still the retriable one: an exhaustion that left a listed
    /// full generation unread is `Exhaustion::BudgetInconclusive`, which
    /// `select_restore_source` reports as `Transient` → `UNAVAILABLE`, because
    /// a base the walk never read is no evidence that the layer has no backup.
    /// Only an exhaustion that adjudicated every listed full generation ends at
    /// the terminal `NotRecoverable` → `NOT_FOUND`. So budget exhaustion does
    /// not decide the status code here — what the walk learned about the
    /// artifacts that can root a lineage does.
    ///
    /// **Why still a ceiling.** 60 s is an order of magnitude under the 540 s
    /// enumeration ceiling the adapter would otherwise impose, so the
    /// head-of-line cost per mutex holder stays bounded at something a client
    /// reconnect can outlast.
    ///
    /// Derived here rather than at the two call sites so the layer and session
    /// hooks cannot drift apart: they bound different reads (a lineage walk vs.
    /// one GET) but the same hazard, and a single expression is what keeps the
    /// rationale above true for both.
    pub fn checkpoint_recovery_fetch_budget(&self) -> Duration {
        Duration::from_millis(
            self.rpc_deadline_ms
                .max(self.checkpoint_request_timeout_ms.saturating_mul(2)),
        )
    }

    /// Wall-clock budget for `acquire_session_cancel`'s recovery pass: the one
    /// re-resolve through the session recovery hook it runs when a streaming
    /// RPC's setup finds its session missing from the live store.
    ///
    /// See [`session_recovery_pass_budget`] for the expression and its
    /// derivation; this is the same value computed from a finished `Config`.
    pub fn session_recovery_pass_budget(&self) -> Duration {
        session_recovery_pass_budget(
            self.presence_refresh_interval_secs,
            self.presence_lease_ttl_secs,
        )
    }

    /// Wall-clock budget a `CreateLayer` gives the in-flight object-id counter
    /// repair before returning `UNAVAILABLE` and letting the client retry
    /// (b.b2j review round 2).
    ///
    /// `⅔ × rpc_deadline_ms` — 20 s at the 30 s default. No env var: the number
    /// is not independently meaningful, it is a *fraction* of the deadline, and
    /// a knob would only create the opportunity to set it above one.
    ///
    /// **Why strictly below the deadline.** `CreateLayer` is not in
    /// `EXEMPT_FROM_DEADLINE` (`src/grpc/metrics_layer.rs`), so the tower layer
    /// bounds the handler future at `rpc_deadline_ms` and drops it on elapse.
    /// The two budgets produce very different outcomes. This one returns a
    /// deliberate `UNAVAILABLE` naming the pending repair, leaves the repair
    /// running on its detached task to record its verdict, and lets the retry
    /// join that attempt. The deadline layer instead drops the future and yields
    /// a bare `DEADLINE_EXCEEDED` with nothing said about the counter.
    ///
    /// **How far that ordering reaches.** Only from the allocation call. The
    /// tower layer bounds the whole handler future from *RPC start*, and
    /// allocation is the terminal step of `handle_create_layer` — auth, the
    /// upload admission wait, the content read, userinfo, OMPF parse and
    /// ext-value storage all precede it — so a request that already spent more
    /// than a third of its
    /// deadline before allocating, or one whose client set a shorter
    /// `grpc-timeout`, is still dropped mid-wait: bare `DEADLINE_EXCEEDED`, no
    /// `wait_timeout` tick. The ⅔ ordering therefore makes the
    /// deliberate-`UNAVAILABLE` shape the outcome when the repair starts early
    /// in the deadline window; what it guarantees unconditionally is that no
    /// repair is ever abandoned half-finished — the attempt is detached and
    /// books its own outcome either way. Full argument: SDD §22.18.
    ///
    /// **Why not something like `deadline - 1s`.** A fixed subtraction inverts
    /// under a short deadline (`500 ms` would go negative), and a saturating one
    /// silently collapses to zero — which elapses instantly and would fail every
    /// `CreateLayer` on a deployment that merely tightened its deadline. A
    /// fraction scales in both directions and can never cross over. The `max(1)`
    /// floor exists only so a sub-millisecond deadline cannot produce a
    /// zero-duration timeout, which `tokio::time::timeout` treats as
    /// already-elapsed.
    ///
    /// **Why ⅔ rather than ⅞.** The remaining third is the rest of the
    /// `CreateLayer`: the allocation retry after the repair, the MULTI/EXEC that
    /// writes the object, the mapping swap. A budget that consumed the whole
    /// deadline would hand back a repaired counter with no time left to use it.
    pub fn object_id_counter_repair_wait_budget(&self) -> Duration {
        self.object_id_counter_repair_wait_budget_within(None)
    }

    /// [`Self::object_id_counter_repair_wait_budget`], narrowed to what is
    /// actually left of *this* request's deadline (b.hka (f)).
    ///
    /// `⅔ × min(rpc_deadline_ms, remaining)`, with the same `max(1 ms)` floor.
    /// `remaining = None` — no ambient per-request deadline is in scope —
    /// collapses to exactly the fixed budget, which is why that method
    /// delegates here rather than the other way round.
    ///
    /// **What this fixes.** The fixed ⅔ ordering only holds when the wait
    /// starts at RPC t=0. It does not: allocation is the terminal step of
    /// `handle_create_layer`, so auth, the upload admission wait, the content
    /// read, userinfo, OMPF parse and ext-value storage all run first, and a
    /// client may have set a
    /// `grpc-timeout` shorter than the server deadline anyway. Either way the
    /// fixed budget can outlast what remains, and the tower deadline layer —
    /// not this budget — ends the wait, dropping the handler future and
    /// yielding a bare `DEADLINE_EXCEEDED` with no `wait_timeout` tick and no
    /// mention of the pending repair. Measuring the fraction against the
    /// *actual* remaining time makes the deliberate-`UNAVAILABLE` shape
    /// unconditional instead of merely typical.
    ///
    /// **Why the same ⅔ and not "all of what is left".** The remaining third is
    /// funded out of real remaining time for the same reason as in the fixed
    /// case: the allocation retry after the repair, the MULTI/EXEC that writes
    /// the object, and the mapping swap all still have to happen inside the
    /// deadline. Handing back a repaired counter with no time left to use it
    /// converts one clean `UNAVAILABLE` into a `DEADLINE_EXCEEDED` a step
    /// later.
    ///
    /// **No env var.** Same reasoning as the fixed budget: the number is a
    /// *fraction*, not an independently meaningful duration, and a knob would
    /// only create the opportunity to set it above one. The `remaining` input
    /// is observed, not configured.
    pub fn object_id_counter_repair_wait_budget_within(
        &self,
        remaining: Option<Duration>,
    ) -> Duration {
        let deadline = Duration::from_millis(self.rpc_deadline_ms);
        let effective = match remaining {
            Some(remaining) => deadline.min(remaining),
            None => deadline,
        };
        // Integer-millisecond arithmetic, matching the fixed budget exactly on
        // the `None` arm: `(ms / 3 * 2)`, truncating, then floored at 1 ms so a
        // sub-millisecond window cannot produce a zero-duration timeout (which
        // `tokio::time::timeout` treats as already elapsed and would turn every
        // `CreateLayer` on a nearly-spent deadline into an instant give-up
        // rather than a short wait).
        Duration::from_millis(((effective.as_millis() as u64) / 3 * 2).max(1))
    }

    /// SR-1.4 (b.tr5) chain link: reject a TLS Valkey URL on a binary built
    /// without a Valkey TLS feature.
    ///
    /// This is the sole place the `valkey-tls` cargo feature is consulted via
    /// `cfg!`. That feature is the single route by which fred receives a TLS
    /// backend (see `Cargo.toml`'s `valkey-tls = ["fred/enable-rustls-ring"]`
    /// alias and its rationale block), so the `cfg!` here is a faithful
    /// compile-time proxy for "fred's TLS support is compiled in". The actual
    /// decision lives in the pure, `cfg!`-free `validate_valkey_tls_scheme`, so
    /// the reject branch is directly unit-testable in a build where the feature
    /// is on.
    fn validate_valkey_tls_compiled(self) -> Result<Self, String> {
        validate_valkey_tls_scheme(&self.valkey_url, cfg!(feature = "valkey-tls"))?;
        Ok(self)
    }

    /// SR-2.3 (b.tr5) fail-closed chain link: reject an incompletely or
    /// inconsistently configured passwordless Valkey auth mode at startup,
    /// mirroring `validate_auth_endpoint`'s fail-closed posture.
    ///
    /// Runs *after* `validate_valkey_tls_compiled` in the `from_env` chain so an
    /// operator on a TLS-uncompiled build sees "TLS not compiled in" first,
    /// rather than a downstream auth-mode complaint about the same URL.
    ///
    /// By mode:
    /// - `Static` — no additional inputs required.
    /// - `AccessTokenFile` — both a token-file path and the AUTH username are
    ///   co-required: the provider cannot `AUTH` without the username, so
    ///   failing closed at startup beats failing at connect.
    /// - `Iam` — region, ACL username, and cluster/cache name are all required:
    ///   the SigV4 auth token cannot be built without the cluster/cache name
    ///   (the signing host), so failing closed at startup beats failing at
    ///   connect.
    ///
    /// Both passwordless modes additionally require a TLS URL (managed
    /// passwordless endpoints are TLS-only). That check reuses the canonical
    /// `valkey_url_uses_tls` predicate negated, so `valkeys://` is correctly
    /// treated as TLS; the rejection never echoes the URL (it carries inline
    /// credentials — mirror the scheme-only redaction in
    /// `validate_valkey_tls_scheme`).
    fn validate_valkey_auth_mode(self) -> Result<Self, String> {
        match self.valkey_auth_mode {
            ValkeyAuthMode::Static => {}
            ValkeyAuthMode::AccessTokenFile => {
                if self.valkey_token_file.is_none() {
                    return Err(
                        "invalid configuration: LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile requires \
                         LIVE_EDIT_VALKEY_TOKEN_FILE (the path to the sidecar-written \
                         access-token file); it is unset. Set it to the token-file path, or switch \
                         LIVE_EDIT_VALKEY_AUTH_MODE back to `static`."
                            .to_string(),
                    );
                }
                if self.valkey_token_username.is_none() {
                    return Err(
                        "invalid configuration: LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile requires \
                         LIVE_EDIT_VALKEY_TOKEN_USERNAME (the AUTH username, conventionally the \
                         token issuer's identity such as a managed identity's Object ID); it is \
                         unset. The provider cannot AUTH without it. Set it to the AUTH username, \
                         or switch LIVE_EDIT_VALKEY_AUTH_MODE back to `static`."
                            .to_string(),
                    );
                }
            }
            ValkeyAuthMode::Iam => {
                if self.valkey_iam_region.is_none() {
                    return Err(
                        "invalid configuration: LIVE_EDIT_VALKEY_AUTH_MODE=iam requires \
                         LIVE_EDIT_VALKEY_IAM_REGION (the AWS region used to build the SigV4 auth \
                         token); it is unset. Set it to the cluster's AWS region, or switch \
                         LIVE_EDIT_VALKEY_AUTH_MODE back to `static`."
                            .to_string(),
                    );
                }
                if self.valkey_iam_username.is_none() {
                    return Err(
                        "invalid configuration: LIVE_EDIT_VALKEY_AUTH_MODE=iam requires \
                         LIVE_EDIT_VALKEY_IAM_USERNAME (the IAM-enabled Valkey ACL user to \
                         authenticate as); it is unset. Set it to the ACL username, or switch \
                         LIVE_EDIT_VALKEY_AUTH_MODE back to `static`."
                            .to_string(),
                    );
                }
                if self.valkey_iam_cluster_host.is_none() {
                    return Err(
                        "invalid configuration: LIVE_EDIT_VALKEY_AUTH_MODE=iam requires \
                         LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST (the cluster/cache name used as the \
                         SigV4 signing host — for MemoryDB the short cluster name, NOT the \
                         endpoint FQDN, which goes in LIVE_EDIT_VALKEY_URL); it is unset. Set it \
                         to the cluster/cache name, or switch LIVE_EDIT_VALKEY_AUTH_MODE back to \
                         `static`."
                            .to_string(),
                    );
                }
            }
        }

        if matches!(
            self.valkey_auth_mode,
            ValkeyAuthMode::AccessTokenFile | ValkeyAuthMode::Iam
        ) && !valkey_url_uses_tls(&self.valkey_url)
        {
            return Err("invalid configuration: a passwordless Valkey auth mode \
                 (LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile|iam) requires a TLS LIVE_EDIT_VALKEY_URL, but \
                 the configured URL is plaintext. Managed passwordless endpoints are TLS-only and \
                 the auth token would otherwise transit in the clear. Use a `rediss://` URL, or \
                 switch LIVE_EDIT_VALKEY_AUTH_MODE back to `static`."
                .to_string());
        }

        Ok(self)
    }
}

/// Validate that an auth endpoint env-var contains an explicit `http://` or
/// `https://` scheme (b.ohw C-6 / H-auth-1).
///
/// Bare hostnames are rejected at config-parse time — without this, a
/// misconfigured deployment that drops the scheme by accident would let
/// `auth::normalize_endpoint` (pre-b.ohw) silently rewrite to `http://...`,
/// transmitting bearer tokens in cleartext. Calling the same validator that
/// the auth-client constructors use means the failure mode is identical and
/// the error message is consistent.
fn validate_auth_endpoint(var: &str, value: &str) -> Result<(), String> {
    crate::auth::validate_auth_endpoint_uri(value)
        .map_err(|reason| format!("invalid value for {var}: {reason}"))
}

const REDIS_TLS_SCHEME: &str = "rediss";
const VALKEY_TLS_SCHEME: &str = "valkeys";

/// The single canonical Valkey-URL TLS-scheme predicate for this module.
///
/// Returns `true` iff `valkey_url`'s scheme means TLS to fred. This mirrors
/// fred's own detection (`fred-10.1.0/src/utils.rs:697-700`) exactly, and any
/// other Valkey-URL scheme test in `src/config.rs` MUST reuse this function
/// rather than deriving its own. In particular, **Epic 2's
/// `validate_valkey_auth_mode` (SR-2.3) MUST call this predicate (negated) for
/// its "plaintext URL with a passwordless mode" check** rather than writing a
/// fresh `starts_with("redis://")`-style test — doing otherwise reintroduces
/// the `valkeys://` blind spot documented below.
///
/// Three bypasses a naive reimplementation would miss, each load-bearing:
///
/// - fred's TLS schemes are **both** `"rediss"` and `"valkeys"`
///   (`fred-10.1.0/src/utils.rs:55-56`). `valkeys://` is a TLS scheme to fred
///   yet appears nowhere in this repo's docs, chart, or tests — so a guard
///   written from our own documentation would hardcode only `rediss`, pass
///   review, and still ship the plaintext-fallback hole.
/// - fred matches with `starts_with` on the whole scheme token, so
///   `rediss-cluster`, `rediss-sentinel`, `rediss+unix`, `valkeys-cluster`, …
///   all count (fred's `CLUSTER_SCHEME_SUFFIX = "-cluster"` /
///   `SENTINEL_SCHEME_SUFFIX = "-sentinel"` / `UNIX_SCHEME_SUFFIX = "+unix"`,
///   `fred-10.1.0/src/utils.rs:57-59`). We use the same `starts_with` and do
///   NOT hardcode a suffix list.
/// - `url::Url::parse` is used rather than raw-string prefix matching because
///   the `url` crate lowercases the scheme (`url-2.5.8/src/parser.rs:413`) and
///   skips ASCII tab/CR/LF anywhere in the input
///   (`url-2.5.8/src/parser.rs:329-331`). `REDISS://…` and `red\tiss://…` are
///   both TLS URLs to fred; a `starts_with` on the raw input would leak either.
///
/// Detection is **scheme-only** in the other direction too:
/// `redis://rediss.example.com:6379` is plaintext to fred, and a
/// `contains("rediss")`-style guard would over-fire and hard-fail a legitimate
/// plaintext deployment at startup.
fn valkey_url_uses_tls(valkey_url: &str) -> bool {
    let Ok(parsed) = url::Url::parse(valkey_url) else {
        return false; // defer malformed URLs to fred's own BadUrl error
    };
    let scheme = parsed.scheme();
    scheme.starts_with(REDIS_TLS_SCHEME) || scheme.starts_with(VALKEY_TLS_SCHEME)
}

/// Pure SR-1.4 (b.tr5) decision helper: reject a TLS Valkey URL when no Valkey
/// TLS feature is compiled in.
///
/// `tls_compiled` is injected (not read via `cfg!` here) so the reject branch
/// is directly unit-testable in a build where the feature is on. The compiled
/// state is supplied by the sole caller, `validate_valkey_tls_compiled`, via
/// a `cfg!` on the `valkey-tls` cargo feature.
///
/// The rejection message deliberately carries only the env-var name and the
/// offending scheme token — never the URL. `LIVE_EDIT_VALKEY_URL` routinely
/// carries inline credentials (cf. `src/storage/client.rs`'s
/// `redact_rediss_tls_scheme_strips_credentials`), and schemes carry no
/// secrets, so echoing the scheme alone respects the token-secrecy invariant.
fn validate_valkey_tls_scheme(valkey_url: &str, tls_compiled: bool) -> Result<(), String> {
    if tls_compiled || !valkey_url_uses_tls(valkey_url) {
        return Ok(());
    }
    // Safe to name the scheme: `valkey_url_uses_tls` returned true, so the URL
    // parsed and its scheme is a TLS scheme token (no credential material).
    let scheme = url::Url::parse(valkey_url)
        .map(|u| u.scheme().to_string())
        .unwrap_or_else(|_| REDIS_TLS_SCHEME.to_string());
    Err(format!(
        "invalid configuration: LIVE_EDIT_VALKEY_URL requests TLS (scheme `{scheme}`) but this \
         binary was built without a Valkey TLS feature. fred silently falls back to a plaintext \
         connection in that case, so Valkey credentials and layer data would transit in the clear \
         on a URL that says otherwise. Rebuild with the `valkey-tls` cargo feature enabled (it is \
         on by default), or use a plaintext `redis://` URL if plaintext is genuinely intended."
    ))
}

/// Return the value of a required environment variable, or an error naming it.
///
/// Surrounding whitespace is trimmed before the value is returned so that a
/// copy-paste with a trailing newline or a space in a manifest (e.g. a quoted
/// `" redis://..."`) does not silently fail downstream parsing (b.3z7
/// medium).
fn required(var: &str) -> Result<String, String> {
    env::var(var)
        .map(|v| v.trim().to_string())
        .map_err(|_| format!("required environment variable {var} is not set"))
}

/// Return the value of an optional environment variable parsed into `T`, falling back to
/// `default` if the variable is absent. Returns `Err` if the variable is present but
/// unparseable.
///
/// The variable is trimmed of surrounding whitespace before being parsed, so
/// a value like `" 100 "` parses as `100` and no downstream component has to
/// worry about stray whitespace from manifest quoting (b.3z7 medium).
/// Baseline cadence of the checkpoint-store reconciliation pass, in seconds,
/// when `LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS` is unset.
///
/// A *baseline*, not the effective default: the pass is only useful at an
/// interval at or above the derived reconciliation grace, so the resolved
/// default is `max(this, derived_grace)`. See the parse arm in
/// [`Config::from_env`] and [`derived_reconcile_grace`].
pub const CHECKPOINT_RECONCILE_INTERVAL_DEFAULT_SECS: u64 = 3600;

/// Default `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN`: how far past the
/// counters sidecar's `max_object_id_observed` the object-id counter is
/// seeded. See [`Config::checkpoint_counter_safety_margin`] for what it must
/// cover and why it is this large.
pub const CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT: u64 = 1_000_000;

/// Largest `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN` accepted (10^9).
///
/// Object ids must stay at or below 2^53 − 1, the Lua f64 integer ceiling
/// every id crosses on its way through `EVAL` (SDD §22.18), and the seed adds
/// the margin to the sidecar value with nothing at runtime to stop the sum
/// passing that ceiling. A pod start that runs the startup seed can move the
/// counter up by up to the whole margin, not only a start after a keyspace
/// loss: the live counter normally sits less than one margin above the
/// sidecar, so `sidecar + margin` lands above it. Pods starting within one
/// counters-sync tick read the same sidecar, so only the first of them moves
/// the counter, and with no sidecar the seed does not move it at all. The
/// bound therefore has to leave the id space room for a lifetime of pod
/// starts:
///
/// - at 10^9 per start, 2^53 − 1 (≈ 9.0 × 10^15) holds about 9 million
///   starts in the worst case: about 82 years at 300 pod starts a day, or
///   about 27 years at 900 a day;
/// - `sidecar + margin` can pass the ceiling only once the counter is within
///   10^9 of it, i.e. after the id space is already more than 99.99% spent;
/// - 10^9 is 1000× the default and covers about 1.67 million `CreateLayer`
///   calls a second over a late sidecar write's 600 s window, or about
///   877,000 a second over the 1,140 s window of a counters tick held up
///   behind an assembly (SDD §22.7.1), far beyond any real rate, so the
///   bound refuses only a mistyped value.
pub const CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX: u64 = 1_000_000_000;

/// Baseline cadence of the external-value `ref_count` reconciler, in seconds,
/// when `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS` is unset (b.yda).
///
/// A *baseline*, not the effective default, for the same reason as
/// [`CHECKPOINT_RECONCILE_INTERVAL_DEFAULT_SECS`]: an interval below the pass's
/// derived grace can never claim a candidate it registered, so the resolved
/// default is `max(this, ext_reconcile_grace(rpc_deadline_ms))`. At the default
/// `LIVE_EDIT_RPC_DEADLINE_MS=30000` the grace is 600 s and this baseline wins;
/// above `LIVE_EDIT_RPC_DEADLINE_MS=360000` the grace does. See the parse arm in
/// [`Config::from_env`] and `storage::ext_reconcile::ext_reconcile_grace`.
pub const EXT_RECONCILE_INTERVAL_DEFAULT_SECS: u64 = 3600;

/// The divisor that derives the external-value release budget from one
/// primary's memory when `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` is unset (b.yuo):
/// one eighth of the smallest primary's store-reported `maxmemory`, or of one
/// shard's share of `LIVE_EDIT_VALKEY_MEMORY_BYTES`
/// (`storage::store_memory::memory_bytes_budget`).
pub const EXT_RELEASE_BUDGET_MEMORY_DIVISOR: u64 = 8;

/// The smallest `LIVE_EDIT_VALKEY_MEMORY_BYTES` accepted at parse time, in
/// bytes (b.qhk): 100 MB. The setting is the store's total memory; no
/// supported store is smaller, so a value below this — `12` meant as 12 GB —
/// is a units slip, refused at startup rather than shrinking the release
/// budget to a few bytes.
pub const VALKEY_MEMORY_BYTES_MIN: u64 = 100_000_000;

/// The multiple of `LIVE_EDIT_MAX_LAYER_SIZE` the external-value release
/// budget falls back to, capped at [`EXT_RELEASE_BUDGET_FALLBACK_CAP_BYTES`],
/// when `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` is unset and the store's memory
/// gives no budget (b.yuo). See [`Config::ext_release_budget`].
pub const EXT_RELEASE_BUDGET_MAX_LAYER_MULTIPLE: u64 = 8;

/// The cap on the external-value release budget's fallback (b.1jt): with
/// `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` unset, no usable `maxmemory` from the
/// store, and no budget from `LIVE_EDIT_VALKEY_MEMORY_BYTES`,
/// `H = min(8 × LIVE_EDIT_MAX_LAYER_SIZE, 100 MB)`.
///
/// The fallback must be safe on the smallest supported store that does not
/// report its size. Azure Managed Redis reports no `maxmemory`, and its
/// smallest cache, `Balanced_B1` (1 GB), holds about 0.8 GB per primary; one
/// eighth of that — the one-eighth-of-a-primary rule
/// ([`EXT_RELEASE_BUDGET_MEMORY_DIVISOR`]) — is 100 MB, while the uncapped
/// `8 × 100 MiB` (839 MB) exceeds its usable memory. No minimum cache size
/// can be stated instead: AMR grows its Balanced and Compute Optimized tiers
/// by adding shards, not by enlarging them, so no SKU there reaches the
/// 6.7 GB primary `8 × LIVE_EDIT_MAX_LAYER_SIZE` would need under the same
/// rule.
///
/// The cost: the GC collects released blobs at any age once they total half
/// the budget (50 MB: reached above about 8.3 MB/s of released
/// external-value bytes, 50 MB over the 5 s age floor plus a pass), and
/// uploads carrying external values wait while they total 100 MB. On a store
/// that does not report its size, setting `LIVE_EDIT_VALKEY_MEMORY_BYTES` to
/// its total memory is recommended for large-value workloads; it is never
/// required.
pub const EXT_RELEASE_BUDGET_FALLBACK_CAP_BYTES: u64 = 100_000_000;

/// The external-value release budget `H` a configuration resolves to (b.yuo),
/// from [`Config::ext_release_budget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtReleaseBudget {
    /// `H`, in bytes; `0` disables the upload gate and the GC pressure trigger.
    pub bytes: u64,
    /// Which setting produced it.
    pub source: ExtReleaseBudgetSource,
}

/// Where [`ExtReleaseBudget::bytes`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtReleaseBudgetSource {
    /// `LIVE_EDIT_EXT_RELEASE_MAX_BYTES`, set explicitly.
    Explicit,
    /// `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` is unset, the store reported no
    /// usable `maxmemory`, and `LIVE_EDIT_VALKEY_MEMORY_BYTES` (the store's
    /// total memory) is set: one shard's share of it ÷ 8, by the store's
    /// cluster map (`storage::store_memory::memory_bytes_budget`). Re-read
    /// every 60 s while the service runs.
    ValkeyMemory {
        /// How `storage::store_memory::memory_bytes_budget` derived it.
        derivation: crate::storage::store_memory::MemoryBytesDerivation,
        /// The distinct primaries (`host:port`) the cluster map listed, for
        /// the log lines only: one endpoint can front several shards, so this
        /// never enters the arithmetic.
        endpoints: usize,
    },
    /// `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` is unset, and every primary of the
    /// store reported a `maxmemory` above zero in `INFO memory`: the smallest
    /// of them, `min_maxmemory_bytes`, ÷ 8 (`storage::store_memory`), whether
    /// or not `LIVE_EDIT_VALKEY_MEMORY_BYTES` is set. Re-read every 60 s while
    /// the service runs.
    StoreInfo {
        /// The smallest `maxmemory` the store's primaries reported, in bytes.
        min_maxmemory_bytes: u64,
        /// How many primaries reported.
        primaries: usize,
    },
    /// `min(8 × LIVE_EDIT_MAX_LAYER_SIZE, 100 MB)`
    /// ([`EXT_RELEASE_BUDGET_FALLBACK_CAP_BYTES`]): `LIVE_EDIT_EXT_RELEASE_MAX_BYTES`
    /// is unset, the store reported no usable `maxmemory`, and
    /// `LIVE_EDIT_VALKEY_MEMORY_BYTES` is unset or the store's cluster map
    /// could not be read. Startup logs the reason — INFO for a store that does
    /// not report its size (expected on a managed store), WARN for a fault.
    Fallback,
}

impl ExtReleaseBudgetSource {
    /// The `source` label of `live_edit_ext_release_budget_bytes` and of the
    /// startup line: `override` for a budget from configuration (the explicit
    /// one, or the one derived from `LIVE_EDIT_VALKEY_MEMORY_BYTES`), `info`
    /// for the store's own `maxmemory`, `fallback` for the fallback.
    pub fn label(self) -> &'static str {
        use crate::observability::metrics::{
            EXT_RELEASE_BUDGET_SOURCE_FALLBACK, EXT_RELEASE_BUDGET_SOURCE_INFO,
            EXT_RELEASE_BUDGET_SOURCE_OVERRIDE,
        };
        match self {
            ExtReleaseBudgetSource::Explicit | ExtReleaseBudgetSource::ValkeyMemory { .. } => {
                EXT_RELEASE_BUDGET_SOURCE_OVERRIDE
            }
            ExtReleaseBudgetSource::StoreInfo { .. } => EXT_RELEASE_BUDGET_SOURCE_INFO,
            ExtReleaseBudgetSource::Fallback => EXT_RELEASE_BUDGET_SOURCE_FALLBACK,
        }
    }

    /// For a budget derived from `LIVE_EDIT_VALKEY_MEMORY_BYTES`, the cluster
    /// map it was derived from and the distinct primaries that map listed:
    /// the `shards`, `endpoints`, `per_shard_bytes` and `total_bytes` fields of the startup
    /// and refresh lines. `None` for every other source.
    pub fn cluster_map(
        &self,
    ) -> Option<(&crate::storage::store_memory::MemoryBytesDerivation, usize)> {
        match self {
            ExtReleaseBudgetSource::ValkeyMemory {
                derivation,
                endpoints,
            } => Some((derivation, *endpoints)),
            _ => None,
        }
    }
}

/// The reconciliation grace implied by an adapter selection plus the
/// StorageApi request timeout.
///
/// Free function rather than a method so [`Config::from_env`] can call it while
/// the `Config` is still being built — the reconcile-interval default depends on
/// it, and both inputs are already resolved as locals by that point.
/// [`Config::checkpoint_reconcile_grace`] is the same computation on a finished
/// `Config`, so config-time validation, the boot-time default, and the runtime
/// `publish_worst_case` all agree by construction.
fn derived_reconcile_grace(adapter: CheckpointAdapter, request_timeout_ms: u64) -> Duration {
    crate::checkpoint::store::reconcile_grace(derived_adapter_call_ceiling(
        adapter,
        request_timeout_ms,
    ))
}

/// The per-adapter-call wall-clock ceiling implied by an adapter selection plus
/// the StorageApi request timeout.
///
/// Extracted from [`derived_reconcile_grace`] by b.gg7 so the *same* expression
/// serves three consumers that must not diverge: the reconcile-interval
/// validation (via the grace), the enforced `tokio::time::timeout` inside
/// `StorageApiCheckpointStore::send_classified`, and the
/// `live_edit_checkpoint_config_adapter_call_ceiling_seconds` config-echo gauge.
/// The adapter method (`CheckpointStore::publish_worst_case`) is the runtime
/// face of the same arithmetic; this is the face reachable from a `Config`
/// alone, before any adapter exists.
///
/// The non-StorageApi arms return [`LOCAL_PUBLISH_WORST_CASE`] — the local
/// adapters do not retry and have no enforced ceiling, so the gauge echoes the
/// conservative bound they report rather than pretending they are bounded by a
/// timeout they do not install.
///
/// [`LOCAL_PUBLISH_WORST_CASE`]: crate::checkpoint::store::LOCAL_PUBLISH_WORST_CASE
fn derived_adapter_call_ceiling(adapter: CheckpointAdapter, request_timeout_ms: u64) -> Duration {
    match adapter {
        CheckpointAdapter::StorageApi => {
            crate::checkpoint::adapters::storage_api_publish_worst_case(Duration::from_millis(
                request_timeout_ms,
            ))
        }
        _ => crate::checkpoint::store::LOCAL_PUBLISH_WORST_CASE,
    }
}

/// Budget for `acquire_session_cancel`'s recovery pass, from the presence lease
/// knobs: `min(refresh / 2, ttl − refresh − 1 ms)`, floored at 1 ms. 7.5 s at
/// the defaults (refresh 15 s, TTL 60 s).
///
/// **Why the presence knobs.** `JoinSession` runs that helper at SR-4 step 6,
/// after step 4 registered the connection with a lease of one TTL and before
/// step 7 starts the lease keeper, whose first renewal comes at most one
/// refresh interval after it starts. Nothing renews the lease during step 6.
/// So the lease outlives the join's setup only if step 6 takes less than
/// `ttl − refresh`; the clamp guarantees that whenever `ttl > refresh` (no
/// budget can when it is not, and the 1 ms floor only keeps the timeout from
/// being zero, which `tokio::time::timeout` treats as already elapsed). Past it,
/// the sweeper could reap the connection (a spurious `PeerLeft`) before the
/// keeper ever runs.
///
/// **Why half a refresh interval.** `validate_presence_lease_budgets` enforces
/// `2 × refresh ≤ ttl` so that one missed renewal still leaves a second chance
/// before the lease lapses. A pass that could take up to `ttl − refresh` would
/// spend that slack on setup. Half an interval delays the first renewal by at
/// most half the cadence and leaves most of the slack. Under the validated
/// knobs `refresh / 2 < ttl − refresh` always holds, so the clamp binds only
/// for a configuration built without that validation.
///
/// **What an elapse costs.** The helper returns `UNAVAILABLE` and cancels
/// nothing. The pass already dropped the replica's base-path cache entry, so
/// the client's retry resolves the session at the handler's own lookup, where
/// recovery is bounded only by [`Config::checkpoint_recovery_fetch_budget`].
/// `SubscribeReadLayer` shares the helper and the budget. A subscription holds
/// no lease, but its client waits on the same pass.
///
/// A free function so `LiveEditService`, which holds the two knobs but not the
/// `Config`, evaluates the same expression as
/// [`Config::session_recovery_pass_budget`].
pub fn session_recovery_pass_budget(refresh_interval_secs: u64, lease_ttl_secs: u64) -> Duration {
    Duration::from_millis(
        (refresh_interval_secs.saturating_mul(1000) / 2)
            .min(
                lease_ttl_secs
                    .saturating_mul(1000)
                    .saturating_sub(refresh_interval_secs.saturating_mul(1000))
                    .saturating_sub(1),
            )
            .max(1),
    )
}

fn optional_parsed<T>(var: &str, default: T) -> Result<T, String>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(var) {
        Ok(val) => val
            .trim()
            .parse::<T>()
            .map_err(|e| format!("invalid value for {var}: {e}")),
        Err(_) => Ok(default),
    }
}

/// [`optional_parsed`] for a capacity that becomes a `tokio::sync::Semaphore`
/// size at connect (its caller refuses `0`). `Semaphore::new` panics above
/// `Semaphore::MAX_PERMITS`, so a larger value is refused here, naming the
/// variable, instead of crashing the process at connect.
fn optional_semaphore_capacity(var: &str, default: usize) -> Result<usize, String> {
    let n = optional_parsed(var, default)?;
    let max = tokio::sync::Semaphore::MAX_PERMITS;
    if n > max {
        return Err(format!(
            "invalid value for {var}: {n} is above the largest supported limit ({max})"
        ));
    }
    Ok(n)
}

/// Refuse startup when a memory-budget variable the service no longer reads
/// is set, naming it, the budget's derived value at this pod's limit, and
/// what to do instead: these budgets are derived from the pod memory limit
/// (`config::memory_plan`), and a set variable left in place would read as if
/// it still applied.
fn refuse_removed_memory_settings(plan: &PodMemoryPlan) -> Result<(), String> {
    use memory_plan::grouped;
    let limit_mib = grouped(plan.limit().mib());
    let removed: [(&str, &str, String, &str); 4] = [
        (
            "LIVE_EDIT_UPLOAD_MEMORY_BUDGET_MIB",
            "upload memory budget",
            format!("{} MiB", grouped(plan.upload_budget_mib())),
            "budget",
        ),
        (
            "LIVE_EDIT_CATCHUP_BUILD_MEMORY_BUDGET_MIB",
            "catch-up memory budget",
            format!("{} MiB", grouped(plan.catchup_budget_mib())),
            "budget",
        ),
        (
            "LIVE_EDIT_STRUCTURAL_CACHE_MAX_MIB",
            "structural cache's memory budget",
            format!("{} MiB", grouped(plan.structural_cache_budget_mib())),
            "budget",
        ),
        (
            "LIVE_EDIT_SUBSCRIBER_MAX_SUBSCRIPTIONS",
            "subscription ceiling",
            format!("{} subscriptions", grouped(plan.max_subscriptions())),
            "ceiling",
        ),
    ];
    for (var, budget, value, noun) in removed {
        if env::var_os(var).is_some() {
            return Err(format!(
                "{var} is no longer read: the {budget} is derived from the container's memory \
                 limit at startup ({value} at this pod's {limit_mib} MiB). Unset it; to change \
                 the {noun}, change the pod's memory limit."
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::ParseCheckpointAdapterError;

    /// Minimum vars for Config::from_env() to succeed with defaults.
    ///
    /// Post-b.g3z the AUTHZ default flipped to `true`, so
    /// STORAGE_SERVICE_ENDPOINT is now strictly required (not just
    /// convenience). USERINFO_ENDPOINT remains here because
    /// `LIVE_EDIT_USERINFO_SERVICE_ENABLED` defaults to `false` and the
    /// SR-1.5 cross-validation rule then forces `LIVE_EDIT_AUTHZ_ENABLED`
    /// to be `true` — which is the default. Callers that want the
    /// pre-b.g3z authz-off behaviour must explicitly pin both
    /// `LIVE_EDIT_AUTHZ_ENABLED=false` AND
    /// `LIVE_EDIT_USERINFO_SERVICE_ENABLED=true` so the cross-validation
    /// rule passes.
    const REQUIRED_VARS: [(&str, Option<&str>); 3] = [
        (
            "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
            Some("http://auth.example.com"),
        ),
        (
            "LIVE_EDIT_USERINFO_ENDPOINT",
            Some("http://userinfo.example.com"),
        ),
        // Pinned unset so the default-value assertions below cannot be
        // satisfied or broken by an ambient copy of the test-only hatch.
        (
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            None,
        ),
    ];

    // ---- t3.1it.rr.ng.9e: happy path and defaults ----

    #[test]
    fn config_from_env_all_defaults_match_sdd() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(cfg.port, 50001, "port default");
            assert_eq!(cfg.metrics_port, 9090, "metrics_port default");
            assert_eq!(
                cfg.valkey_url, "redis://localhost:6379",
                "valkey_url default"
            );
            assert_eq!(cfg.valkey_pool_size, 0, "valkey_pool_size default");
            assert_eq!(
                cfg.valkey_connect_timeout_secs, 30,
                "valkey_connect_timeout_secs default"
            );
            assert_eq!(
                cfg.valkey_command_timeout_ms, 5000,
                "valkey_command_timeout_ms default"
            );
            assert_eq!(
                cfg.valkey_reconnect_max_delay_ms, 30_000,
                "valkey_reconnect_max_delay_ms default"
            );
            assert_eq!(
                cfg.storage_service_endpoint,
                Some("http://auth.example.com".to_string())
            );
            assert_eq!(cfg.auth_timeout_ms, 5000, "auth_timeout_ms default");
            assert_eq!(cfg.auth_cache_ttl_secs, 600, "auth_cache_ttl_secs default");
            assert_eq!(
                cfg.userinfo_endpoint.as_deref(),
                Some("http://userinfo.example.com")
            );
            assert_eq!(
                cfg.max_layer_size, 104_857_600,
                "max_layer_size default (100 MB)"
            );
            assert_eq!(
                cfg.shutdown_timeout_secs, 5,
                "shutdown_timeout_secs default"
            );
            assert_eq!(cfg.log_level, "info", "log_level default");
            assert_eq!(
                cfg.session_list_timeout_ms, 2000,
                "session_list_timeout_ms default"
            );
            assert_eq!(cfg.rpc_deadline_ms, 30000, "rpc_deadline_ms default");
            assert_eq!(cfg.auth_max_attempts, 3, "auth_max_attempts default");
            assert_eq!(
                cfg.auth_retry_initial_delay_ms, 100,
                "auth_retry_initial_delay_ms default"
            );
            assert!(cfg.authn_enabled, "authn_enabled default");
            assert!(
                cfg.authz_enabled,
                "authz_enabled default flipped to true post-b.g3z"
            );
            assert!(
                !cfg.userinfo_service_enabled,
                "userinfo_service_enabled default is false (b.g3z) — \
                 service runs without an external userinfo backend by default"
            );
            assert!(
                !cfg.test_only_dangerously_allow_unvalidated_tokens,
                "test_only_dangerously_allow_unvalidated_tokens must default to false"
            );
            assert!(
                !cfg.unvalidated_tokens_in_effect(),
                "a default config must never run without token validation"
            );
            // b.fgj / SDD SR-6.1 / SR-6.2 / SR-6.4 — service-identity token
            // env vars. Pinned here so adding a future field upstream
            // surfaces a compile/assertion failure in this table-shaped
            // defaults test instead of silently slipping through.
            assert_eq!(
                cfg.service_token_file, None,
                "service_token_file default is None"
            );
            assert_eq!(
                cfg.service_token_poll_interval_secs, 15,
                "service_token_poll_interval_secs default is 15"
            );
            assert!(
                !cfg.service_token_client_credentials_enabled,
                "service_token_client_credentials_enabled default is false (V1: sidecar-only)"
            );
            // b.msb Epic — late-join/catch-up snapshot switch
            // (LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED). Default `true` = the
            // corrected full-state snapshot path (the shipped fix); `false`
            // selects legacy per-commit reconstruction. Pinned here so a future
            // default flip surfaces in this table-shaped defaults test.
            assert!(
                cfg.catchup_snapshot_enabled,
                "catchup_snapshot_enabled default is true (corrected snapshot on)"
            );
            // b.dfh Epic 3 / SDD SR-8.1 — stream-replay tail capture. The
            // spec fixes only two of these four defaults (`TAIL_ENABLED`, now
            // = true per b.3n1, and `TAIL_MAX_CHAIN` = 8); `TAIL_MAX_ENTRIES` /
            // `TAIL_MAX_BYTES` are implementation choices within the spec's
            // "zero rejected / <= 2^53-1" envelope, so this table is the
            // single place they are pinned. `TailCaptureConfig::default()`
            // restates all four — `tail_capture_config_default_matches_
            // config_from_env_defaults` below is the lockstep check.
            assert!(
                cfg.checkpoint_tail_enabled,
                "checkpoint_tail_enabled default is true (promoted 2026-09-11 \
                 after the dev soak; SR-8.1 shipped it as explicit opt-in)"
            );
            assert_eq!(
                cfg.checkpoint_tail_max_chain, 8,
                "checkpoint_tail_max_chain default is 8 (K; SR-8.1's measured \
                 restore-cost rationale)"
            );
            assert_eq!(
                cfg.checkpoint_tail_max_entries, 4096,
                "checkpoint_tail_max_entries default is 4096 — deliberately equal \
                 to the layer_stream_maxlen default so the cross-field retention \
                 WARN is silent at stock defaults"
            );
            assert_eq!(
                cfg.checkpoint_tail_max_bytes, 67_108_864,
                "checkpoint_tail_max_bytes default is 67108864 (64 MiB)"
            );
            assert_eq!(
                cfg.checkpoint_tail_max_entries, cfg.layer_stream_maxlen,
                "the two defaults must stay equal — the tail-entries default is \
                 pinned TO the stream-retention default precisely so \
                 `warn_tail_entries_exceed_stream_retention` never fires out of \
                 the box; drifting either one alone reintroduces a stock-config WARN"
            );
        });
    }

    #[test]
    fn config_from_env_explicit_port_overrides_default() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_PORT", Some("51000")),
            ],
            || {
                let cfg =
                    Config::from_env_with(PodMemorySource::Reference).expect("should succeed");
                assert_eq!(cfg.port, 51000);
            },
        );
    }

    #[test]
    fn config_from_env_explicit_overrides_all_fields() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.test"),
                ),
                ("LIVE_EDIT_USERINFO_ENDPOINT", Some("http://userinfo.test")),
                ("LIVE_EDIT_PORT", Some("51234")),
                ("LIVE_EDIT_METRICS_PORT", Some("9999")),
                ("LIVE_EDIT_VALKEY_URL", Some("redis://valkey.test:6380")),
                ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("8")),
                ("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", Some("5")),
                ("LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS", Some("750")),
                ("LIVE_EDIT_VALKEY_RECONNECT_MAX_DELAY_MS", Some("12000")),
                ("LIVE_EDIT_AUTH_TIMEOUT_MS", Some("1000")),
                ("LIVE_EDIT_AUTH_CACHE_TTL_SECS", Some("120")),
                ("LIVE_EDIT_MAX_LAYER_SIZE", Some("1048576")),
                ("LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS", Some("10")),
                ("LIVE_EDIT_LOG_LEVEL", Some("debug")),
                ("LIVE_EDIT_SESSION_LIST_TIMEOUT_MS", Some("500")),
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some("15000")),
                ("LIVE_EDIT_SERVICE_LABEL", Some("pod-abc")),
                ("LIVE_EDIT_AUTH_MAX_ATTEMPTS", Some("5")),
                ("LIVE_EDIT_AUTH_RETRY_INITIAL_DELAY_MS", Some("200")),
                ("LIVE_EDIT_AUTHN_ENABLED", Some("true")),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            ],
            || {
                let cfg =
                    Config::from_env_with(PodMemorySource::Reference).expect("should succeed");
                assert_eq!(cfg.port, 51234);
                assert_eq!(cfg.metrics_port, 9999);
                assert_eq!(cfg.valkey_url, "redis://valkey.test:6380");
                assert_eq!(cfg.valkey_pool_size, 8);
                assert_eq!(cfg.valkey_connect_timeout_secs, 5);
                assert_eq!(cfg.valkey_command_timeout_ms, 750);
                assert_eq!(cfg.valkey_reconnect_max_delay_ms, 12_000);
                assert_eq!(
                    cfg.storage_service_endpoint,
                    Some("http://auth.test".to_string())
                );
                assert_eq!(cfg.auth_timeout_ms, 1000);
                assert_eq!(cfg.auth_cache_ttl_secs, 120);
                assert_eq!(
                    cfg.userinfo_endpoint.as_deref(),
                    Some("http://userinfo.test")
                );
                assert_eq!(cfg.max_layer_size, 1_048_576);
                assert_eq!(cfg.shutdown_timeout_secs, 10);
                assert_eq!(cfg.log_level, "debug");
                assert_eq!(cfg.session_list_timeout_ms, 500);
                assert_eq!(cfg.rpc_deadline_ms, 15000);
                assert_eq!(cfg.service_label, "pod-abc");
                assert_eq!(cfg.auth_max_attempts, 5);
                assert_eq!(cfg.auth_retry_initial_delay_ms, 200);
                assert!(cfg.authn_enabled);
                assert!(cfg.authz_enabled);
            },
        );
    }

    /// `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS=0` is refused at startup: it
    /// bounds every startup connect phase and every stream reader connect,
    /// and (with the command timeout disabled) the readers' routing sync, so
    /// zero would cut each before it could finish. One second, the smallest
    /// bound, is accepted as set.
    #[test]
    fn config_from_env_valkey_connect_timeout_zero_rejected_and_one_accepted() {
        let err = assert_snapshot_knob_rejected(
            "LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS",
            "0",
            "a zero connect bound would cut every connect before it could finish",
        );
        assert!(
            err.contains("must be greater than 0"),
            "the reject says what the operator must change; got: {err}"
        );
        temp_env::with_vars(
            snapshot_knob_env("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", "1"),
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("a one-second connect bound is accepted");
                assert_eq!(cfg.valkey_connect_timeout_secs, 1);
            },
        );
    }

    #[test]
    fn config_from_env_service_label_falls_back_to_hostname() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_LABEL", None),
                ("HOSTNAME", Some("my-pod-xyz")),
            ],
            || {
                let cfg =
                    Config::from_env_with(PodMemorySource::Reference).expect("should succeed");
                assert_eq!(cfg.service_label, "my-pod-xyz");
            },
        );
    }

    #[test]
    fn config_from_env_service_label_falls_back_to_unknown_when_no_hostname() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_LABEL", None),
                ("HOSTNAME", None),
            ],
            || {
                let cfg =
                    Config::from_env_with(PodMemorySource::Reference).expect("should succeed");
                assert_eq!(cfg.service_label, "unknown");
            },
        );
    }

    #[test]
    fn config_from_env_service_label_prefers_live_edit_service_label_over_hostname() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_LABEL", Some("explicit-label")),
                ("HOSTNAME", Some("should-not-be-used")),
            ],
            || {
                let cfg =
                    Config::from_env_with(PodMemorySource::Reference).expect("should succeed");
                assert_eq!(cfg.service_label, "explicit-label");
            },
        );
    }

    // ---- t3.1it.rr.ng.fw: fail-fast on missing required vars ----

    #[test]
    fn config_from_env_missing_storage_service_endpoint_succeeds_when_authz_disabled() {
        // Preserves the original intent (storage endpoint is optional on
        // the authz-off branch). Post-b.g3z the default flipped to
        // authz=true, so we pin authz=false explicitly here. The SR-1.5
        // cross-validation rule then forces us to also pin
        // userinfo_service_enabled=true (otherwise no backend would
        // validate the token).
        temp_env::with_vars(
            [
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("false")),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
                ("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", None),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed without STORAGE_SERVICE_ENDPOINT when authz disabled");
                assert_eq!(cfg.storage_service_endpoint, None);
            },
        );
    }

    #[test]
    fn config_from_env_missing_storage_service_endpoint_returns_err_when_authz_enabled() {
        temp_env::with_vars(
            [
                ("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", None),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail without STORAGE_SERVICE_ENDPOINT when authz enabled");
                assert!(
                    err.contains("STORAGE_SERVICE_ENDPOINT"),
                    "error should name STORAGE_SERVICE_ENDPOINT, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_missing_userinfo_endpoint_returns_err_naming_var() {
        // Pin `userinfo_service_enabled=true` so USERINFO_ENDPOINT is
        // required at parse time. Post-b.g3z the default is `false`, in
        // which case USERINFO_ENDPOINT is optional and this test would
        // wrongly stop catching a missing-required-var regression.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
                ("LIVE_EDIT_USERINFO_ENDPOINT", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail without USERINFO_ENDPOINT");
                assert!(
                    err.contains("USERINFO_ENDPOINT"),
                    "error should name USERINFO_ENDPOINT, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_missing_both_required_vars_returns_err() {
        // Both endpoints must be required for this test to exercise its
        // intent. Under b.g3z defaults `userinfo_service_enabled=false`
        // makes USERINFO_ENDPOINT optional — pin it to `true` so the
        // assertion still observes both vars as required.
        temp_env::with_vars(
            [
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
                ("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", None::<&str>),
                ("LIVE_EDIT_USERINFO_ENDPOINT", None),
            ],
            || {
                let result = Config::from_env_with(PodMemorySource::Reference);
                assert!(
                    result.is_err(),
                    "should fail when both required vars are absent"
                );
            },
        );
    }

    #[test]
    fn config_from_env_invalid_port_returns_err() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_PORT", Some("not_a_number")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail with non-numeric port");
                assert!(
                    err.contains("LIVE_EDIT_PORT"),
                    "error should name LIVE_EDIT_PORT, got: {err}"
                );
            },
        );
    }

    // ---- t3.5ci.ny.o9.4u: auth toggle config validation ----

    #[test]
    fn config_from_env_authn_disabled_returns_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHN_ENABLED", Some("false")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail when authn is disabled");
                assert!(
                    err.contains("LIVE_EDIT_AUTHN_ENABLED"),
                    "error should name LIVE_EDIT_AUTHN_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_authn_enabled_explicitly_true_succeeds() {
        // Provide STORAGE_SERVICE_ENDPOINT because the b.g3z default of
        // `authz_enabled=true` now requires it. The test intent here is
        // the AUTHN_ENABLED=true toggle path, not the storage-endpoint
        // requirement.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHN_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with authn explicitly enabled");
                assert!(cfg.authn_enabled);
            },
        );
    }

    #[test]
    fn config_from_env_authz_enabled_with_storage_endpoint_succeeds() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with authz enabled and storage endpoint set");
                assert!(cfg.authz_enabled);
                assert_eq!(
                    cfg.storage_service_endpoint,
                    Some("http://auth.example.com".to_string())
                );
            },
        );
    }

    #[test]
    fn config_from_env_authz_disabled_without_storage_endpoint_succeeds() {
        // SR-1.5 (b.g3z): `authz_enabled=false` requires
        // `userinfo_service_enabled=true` (which validates the bearer only
        // when a validating proxy fronts the userinfo endpoint). Pin both
        // explicitly.
        temp_env::with_vars(
            [
                ("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", None),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("false")),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with authz disabled and no storage endpoint");
                assert!(!cfg.authz_enabled);
                assert_eq!(cfg.storage_service_endpoint, None);
            },
        );
    }

    #[test]
    fn config_from_env_invalid_authn_enabled_returns_parse_err() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHN_ENABLED", Some("yes")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail with invalid bool for AUTHN_ENABLED");
                assert!(
                    err.contains("LIVE_EDIT_AUTHN_ENABLED"),
                    "error should name LIVE_EDIT_AUTHN_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_invalid_authz_enabled_returns_parse_err() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("yes")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail with invalid bool for AUTHZ_ENABLED");
                assert!(
                    err.contains("LIVE_EDIT_AUTHZ_ENABLED"),
                    "error should name LIVE_EDIT_AUTHZ_ENABLED, got: {err}"
                );
            },
        );
    }

    // ---- t3.g3z.zq.7q.h9: SR-5.3 userinfo-service toggle + SR-1.5
    //                       cross-validation (b.g3z) ----

    #[test]
    fn config_from_env_userinfo_service_enabled_default_is_false() {
        // SR-1.1 (b.g3z): post-flip the default for
        // `LIVE_EDIT_USERINFO_SERVICE_ENABLED` is `false`. Pin
        // `AUTHZ_ENABLED=true` (its own new default) so the SR-1.5
        // cross-validation rule is satisfied while we exercise the
        // userinfo toggle's default specifically.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with userinfo flag unset (default applies)");
                assert!(
                    !cfg.userinfo_service_enabled,
                    "userinfo_service_enabled default must be false (b.g3z SR-1.1)"
                );
            },
        );
    }

    #[test]
    fn config_from_env_authz_enabled_default_is_true() {
        // SR-1.2 (b.g3z): post-flip the default for `LIVE_EDIT_AUTHZ_ENABLED`
        // is `true`. Drop the explicit pin and confirm the default lands.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHZ_ENABLED", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with AUTHZ_ENABLED unset (default applies)");
                assert!(
                    cfg.authz_enabled,
                    "authz_enabled default must be true (b.g3z SR-1.2)"
                );
            },
        );
    }

    #[test]
    fn config_from_env_invalid_userinfo_service_enabled_returns_parse_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("yes")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail with invalid bool for USERINFO_SERVICE_ENABLED");
                assert!(
                    err.contains("LIVE_EDIT_USERINFO_SERVICE_ENABLED"),
                    "error should name LIVE_EDIT_USERINFO_SERVICE_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_userinfo_service_disabled_endpoint_absent_succeeds() {
        // SR-1.3 (b.g3z): with the userinfo flag off, the userinfo endpoint
        // is optional. AUTHZ_ENABLED=true keeps SR-1.5 satisfied.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_USERINFO_ENDPOINT", None),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("false")),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed when userinfo flag off and userinfo endpoint absent");
                assert_eq!(
                    cfg.userinfo_endpoint, None,
                    "userinfo_endpoint must be None when flag is off and var unset"
                );
            },
        );
    }

    #[test]
    fn config_from_env_userinfo_service_disabled_endpoint_present_ignored() {
        // SR-1.3 (b.g3z): "parsed if present but otherwise ignored". When
        // the flag is off, scheme validation does NOT run, so a bare
        // hostname (which would normally be rejected) is accepted and
        // stored verbatim as `Some(raw_value)`. Pin the exact shape — not
        // `None` — because operators rely on round-tripping the raw value
        // through configuration without it being silently dropped.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("userinfo.prod.internal"),
                ),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("false")),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "should succeed: scheme validation is gated by userinfo_service_enabled",
                );
                assert_eq!(
                    cfg.userinfo_endpoint.as_deref(),
                    Some("userinfo.prod.internal"),
                    "userinfo_endpoint must round-trip raw value when flag is off (SR-1.3)"
                );
            },
        );
    }

    #[test]
    fn config_from_env_userinfo_service_enabled_endpoint_absent_returns_err_naming_var() {
        // Symmetric to the b.g3z change: when the flag is ON, the userinfo
        // endpoint is required (the original pre-flip behaviour). Pin the
        // flag explicitly so this guarantees the required-var rejection
        // still fires.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
                ("LIVE_EDIT_USERINFO_ENDPOINT", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail without USERINFO_ENDPOINT when userinfo flag is on");
                assert!(
                    err.contains("LIVE_EDIT_USERINFO_ENDPOINT"),
                    "error should name LIVE_EDIT_USERINFO_ENDPOINT, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_userinfo_service_enabled_endpoint_bare_hostname_rejected() {
        // Symmetric to the b.g3z change: when the flag is ON, the scheme
        // validator on the userinfo endpoint runs (fail-closed). A bare
        // hostname must be rejected with the validator's reason string so
        // operators see why their config was refused.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("userinfo.prod.internal"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "bare hostname must be rejected when userinfo flag is on (b.ohw C-6)",
                );
                assert!(
                    err.contains("LIVE_EDIT_USERINFO_ENDPOINT"),
                    "error should name LIVE_EDIT_USERINFO_ENDPOINT, got: {err}"
                );
                assert!(
                    err.contains("explicit scheme"),
                    "error should surface the validator's reason, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_both_userinfo_and_authz_disabled_returns_err_with_sr_1_5_message() {
        // SR-1.5 (b.g3z): with both userinfo and authz disabled, no
        // backend validates the bearer token (signature, issuer, audience,
        // expiry) — `authn_enabled` alone only enforces header presence.
        // Pin the message's content: both env-var names, the
        // AUTHN_ENABLED stub disclaimer, and the remedy wording (the
        // permissions-service bullet, what USERINFO_SERVICE_ENABLED=true
        // actually does, and that it validates only behind a validating
        // proxy). A regression that drops any of these (e.g., quietly
        // removing the disclaimer or one of the remedy bullets) fails this
        // test. Two strings are pinned ABSENT: the old claim that the
        // userinfo service validates tokens, and the test-only override.
        temp_env::with_vars(
            [
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("false")),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("false")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail with both userinfo and authz disabled (SR-1.5)");
                assert!(
                    err.contains("LIVE_EDIT_USERINFO_SERVICE_ENABLED"),
                    "error should name LIVE_EDIT_USERINFO_SERVICE_ENABLED, got: {err}"
                );
                assert!(
                    err.contains("LIVE_EDIT_AUTHZ_ENABLED"),
                    "error should name LIVE_EDIT_AUTHZ_ENABLED, got: {err}"
                );
                assert!(
                    err.contains("LIVE_EDIT_AUTHN_ENABLED only enforces header presence"),
                    "error must preserve the AUTHN_ENABLED stub disclaimer, got: {err}"
                );
                assert!(
                    err.contains("permissions service validates tokens"),
                    "error must preserve the AUTHZ_ENABLED suggested-fix bullet, got: {err}"
                );
                assert!(
                    err.contains(
                        "LIVE_EDIT_USERINFO_SERVICE_ENABLED=true instead resolves user identity"
                    ),
                    "error must still explain what USERINFO_SERVICE_ENABLED=true does, got: {err}"
                );
                assert!(
                    err.contains("validating proxy fronts the userinfo endpoint"),
                    "error must say the userinfo path validates tokens only behind a validating \
                     proxy, got: {err}"
                );
                assert!(
                    !err.contains("userinfo service validates tokens"),
                    "error must not claim the userinfo service validates tokens, got: {err}"
                );
                // The test-only hatch is deliberately NOT advertised to
                // operators as a remedy for this rejection.
                assert!(
                    !err.contains("TEST_ONLY"),
                    "rejection message must not name the test-only override, got: {err}"
                );
            },
        );
    }

    // ---- LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS ----

    /// The off/off auth shape with neither endpoint set: exactly what the
    /// conformance suite runs against. Both endpoint vars are pinned to
    /// `None` so the test also proves the override does not smuggle in an
    /// endpoint requirement.
    const OFF_OFF_AUTH_VARS: [(&str, Option<&str>); 4] = [
        ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("false")),
        ("LIVE_EDIT_AUTHZ_ENABLED", Some("false")),
        ("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", None),
        ("LIVE_EDIT_USERINFO_ENDPOINT", None),
    ];

    #[test]
    fn config_from_env_test_only_override_allows_both_auth_backends_disabled() {
        let mut vars = OFF_OFF_AUTH_VARS.to_vec();
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("true"),
        ));
        temp_env::with_vars(vars, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("off/off auth must parse when the test-only override is set");
            assert!(
                cfg.test_only_dangerously_allow_unvalidated_tokens,
                "override field must reflect the env var"
            );
            assert!(!cfg.authz_enabled, "authz must stay disabled");
            assert!(
                !cfg.userinfo_service_enabled,
                "userinfo service must stay disabled"
            );
            assert!(
                cfg.unvalidated_tokens_in_effect(),
                "off/off + override is the one shape where the override takes effect"
            );
            assert_eq!(
                cfg.storage_service_endpoint, None,
                "no storage endpoint may be required in the off/off shape"
            );
            assert_eq!(
                cfg.userinfo_endpoint, None,
                "no userinfo endpoint may be required in the off/off shape"
            );
        });
    }

    #[test]
    fn config_from_env_test_only_override_explicit_false_still_rejects_off_off() {
        // Setting the variable to `false` must be indistinguishable from
        // leaving it unset: the SR-1.5 rejection stands.
        let mut vars = OFF_OFF_AUTH_VARS.to_vec();
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("false"),
        ));
        temp_env::with_vars(vars, || {
            let err = Config::from_env_with(PodMemorySource::Reference)
                .expect_err("override=false must not relax the off/off rejection");
            assert!(
                err.contains("LIVE_EDIT_USERINFO_SERVICE_ENABLED=false requires LIVE_EDIT_AUTHZ_ENABLED=true"),
                "expected the SR-1.5 rejection, got: {err}"
            );
        });
    }

    #[test]
    fn config_from_env_test_only_override_accepts_surrounding_whitespace() {
        // Same trimming contract as every other `optional_parsed` var
        // (b.3z7): manifest quoting must not turn the hatch into a parse
        // error or, worse, a silent no-op.
        let mut vars = OFF_OFF_AUTH_VARS.to_vec();
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("  true  "),
        ));
        temp_env::with_vars(vars, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("whitespace-padded `true` must parse");
            assert!(cfg.unvalidated_tokens_in_effect());
        });
    }

    #[test]
    fn config_from_env_test_only_override_rejects_unparseable_value() {
        // `1`, `yes`, `TRUE` etc. are not accepted by Rust's `bool` parser,
        // matching every other boolean env var in this file. A typo must
        // fail loudly rather than be read as either `true` or `false`. The
        // empty string is the boundary a templated manifest actually
        // produces (`value: ""`); it is present-but-empty, not unset, and
        // must be a hard error too.
        for bad in ["1", "yes", "TRUE", "on", ""] {
            let mut vars = OFF_OFF_AUTH_VARS.to_vec();
            vars.push((
                "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
                Some(bad),
            ));
            temp_env::with_vars(vars, || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("unparseable override value must be a startup error");
                assert!(
                    err.contains("LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS"),
                    "error must name the offending variable for value {bad:?}, got: {err}"
                );
            });
        }
    }

    #[test]
    fn config_from_env_test_only_override_unparseable_value_rejected_even_when_inert() {
        // Parsed unconditionally: a malformed value is reported even in a
        // shape where the override would change nothing. Otherwise a typo
        // could ride along unnoticed into a real deployment.
        let mut vars = REQUIRED_VARS.to_vec();
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("yes"),
        ));
        temp_env::with_vars(vars, || {
            let err = Config::from_env_with(PodMemorySource::Reference)
                .expect_err("unparseable override value must be rejected regardless of auth shape");
            assert!(
                err.contains("LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS"),
                "error must name the offending variable, got: {err}"
            );
        });
    }

    #[test]
    fn config_from_env_test_only_override_is_inert_when_authz_enabled() {
        // Default auth shape (userinfo off, authz on) plus the override: the
        // service must still require the storage endpoint and must still
        // report that tokens are validated. The field records the operator's
        // setting so the startup WARN can call out the stray variable.
        let mut vars = REQUIRED_VARS.to_vec();
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("true"),
        ));
        temp_env::with_vars(vars, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("default auth shape + override must parse");
            assert!(cfg.authz_enabled);
            assert!(cfg.test_only_dangerously_allow_unvalidated_tokens);
            assert!(
                !cfg.unvalidated_tokens_in_effect(),
                "override must be inert while the permissions service validates tokens"
            );
        });
    }

    #[test]
    fn config_from_env_test_only_override_does_not_relax_storage_endpoint_requirement() {
        // The override relaxes exactly one guard (SR-1.5). With authz on the
        // storage endpoint is still mandatory, override or not.
        let mut vars = REQUIRED_VARS.to_vec();
        vars.push(("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", None));
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("true"),
        ));
        temp_env::with_vars(vars, || {
            let err = Config::from_env_with(PodMemorySource::Reference)
                .expect_err("authz on still requires the storage endpoint, override or not");
            assert!(
                err.contains("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT"),
                "got: {err}"
            );
        });
    }

    #[test]
    fn config_from_env_test_only_override_does_not_relax_authn_rejection() {
        // `LIVE_EDIT_AUTHN_ENABLED=false` is the one startup rejection the
        // service treats as absolute. The override must not widen to it.
        let mut vars = OFF_OFF_AUTH_VARS.to_vec();
        vars.push(("LIVE_EDIT_AUTHN_ENABLED", Some("false")));
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("true"),
        ));
        temp_env::with_vars(vars, || {
            let err = Config::from_env_with(PodMemorySource::Reference)
                .expect_err("disabling authn must be rejected regardless of the override");
            assert!(
                err.contains("LIVE_EDIT_AUTHN_ENABLED=false is not supported"),
                "expected the authn rejection, got: {err}"
            );
        });
    }

    #[test]
    fn config_from_env_authn_rejection_precedes_override_parse() {
        // Pins the ordering: `from_env` returns the authn rejection BEFORE it
        // parses the override. With an unparseable override value the only
        // way to get the authn error (rather than the parse error) is for
        // the authn guard to run first. A reorder that hoists the override
        // parse above the guard flips which error comes back.
        let mut vars = OFF_OFF_AUTH_VARS.to_vec();
        vars.push(("LIVE_EDIT_AUTHN_ENABLED", Some("false")));
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("not-a-bool"),
        ));
        temp_env::with_vars(vars, || {
            let err = Config::from_env_with(PodMemorySource::Reference)
                .expect_err("authn off must be rejected");
            assert!(
                err.contains("LIVE_EDIT_AUTHN_ENABLED=false is not supported"),
                "authn rejection must win over the override parse error, got: {err}"
            );
            assert!(
                !err.contains("LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS"),
                "override must not have been parsed yet, got: {err}"
            );
        });
    }

    #[test]
    fn config_from_env_test_only_override_is_inert_when_userinfo_enabled() {
        let mut vars = REQUIRED_VARS.to_vec();
        vars.push(("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")));
        vars.push(("LIVE_EDIT_AUTHZ_ENABLED", Some("false")));
        vars.push((
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            Some("true"),
        ));
        temp_env::with_vars(vars, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("userinfo on + authz off + override must parse");
            assert!(cfg.userinfo_service_enabled);
            assert!(!cfg.authz_enabled);
            assert!(cfg.test_only_dangerously_allow_unvalidated_tokens);
            assert!(
                !cfg.unvalidated_tokens_in_effect(),
                "override must be inert while the userinfo service is enabled"
            );
        });
    }

    #[test]
    fn unvalidated_tokens_in_effect_requires_all_three_conditions() {
        // Pure predicate check over all eight combinations of the three
        // inputs. The expected column is a literal table, not a re-statement
        // of the predicate, so a sign flip applied to both sides is caught.
        let base = temp_env::with_vars(REQUIRED_VARS, || {
            Config::from_env_with(PodMemorySource::Reference).expect("baseline config")
        });
        // (userinfo_service_enabled, authz_enabled, override_set, expected)
        const TABLE: [(bool, bool, bool, bool); 8] = [
            (false, false, false, false),
            (false, false, true, true), // the ONLY in-effect row
            (false, true, false, false),
            (false, true, true, false),
            (true, false, false, false),
            (true, false, true, false),
            (true, true, false, false),
            (true, true, true, false),
        ];
        for (userinfo, authz, override_set, expected) in TABLE {
            let mut cfg = base.clone();
            cfg.userinfo_service_enabled = userinfo;
            cfg.authz_enabled = authz;
            cfg.test_only_dangerously_allow_unvalidated_tokens = override_set;
            assert_eq!(
                cfg.unvalidated_tokens_in_effect(),
                expected,
                "userinfo={userinfo} authz={authz} override={override_set}"
            );
        }
    }

    #[test]
    fn config_from_env_userinfo_disabled_authz_enabled_succeeds() {
        // The new-default safe-pin combination (b.g3z): userinfo service
        // disabled, authz enabled, storage endpoint set, userinfo
        // endpoint absent. This is the "no userinfo backend" deployment
        // shape that SR-1.5 leaves valid.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_USERINFO_ENDPOINT", None),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("false")),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("new-default safe-pin (userinfo off + authz on) must parse cleanly");
                assert_eq!(
                    cfg.userinfo_endpoint, None,
                    "userinfo_endpoint must be None when flag is off and var unset"
                );
                assert!(
                    cfg.authz_enabled,
                    "authz_enabled must be true on this safe-pin combination"
                );
                assert!(
                    !cfg.userinfo_service_enabled,
                    "userinfo_service_enabled must be false on this safe-pin combination"
                );
            },
        );
    }

    // ---- t3.943.oz.ar.5w: AuthProtocol FromStr pure parsing ----

    #[test]
    fn auth_protocol_from_str_accepts_lowercase_rest() {
        assert_eq!(AuthProtocol::from_str("rest").unwrap(), AuthProtocol::Rest);
    }

    #[test]
    fn auth_protocol_from_str_accepts_uppercase_rest() {
        assert_eq!(AuthProtocol::from_str("REST").unwrap(), AuthProtocol::Rest);
    }

    #[test]
    fn auth_protocol_from_str_accepts_mixed_case_rest() {
        assert_eq!(AuthProtocol::from_str("Rest").unwrap(), AuthProtocol::Rest);
        assert_eq!(AuthProtocol::from_str("ReSt").unwrap(), AuthProtocol::Rest);
        assert_eq!(AuthProtocol::from_str("rEST").unwrap(), AuthProtocol::Rest);
    }

    #[test]
    fn auth_protocol_from_str_accepts_lowercase_grpc() {
        assert_eq!(AuthProtocol::from_str("grpc").unwrap(), AuthProtocol::Grpc);
    }

    #[test]
    fn auth_protocol_from_str_accepts_uppercase_grpc() {
        assert_eq!(AuthProtocol::from_str("GRPC").unwrap(), AuthProtocol::Grpc);
    }

    #[test]
    fn auth_protocol_from_str_accepts_mixed_case_grpc() {
        assert_eq!(AuthProtocol::from_str("Grpc").unwrap(), AuthProtocol::Grpc);
        assert_eq!(AuthProtocol::from_str("GrPc").unwrap(), AuthProtocol::Grpc);
        assert_eq!(AuthProtocol::from_str("gRPC").unwrap(), AuthProtocol::Grpc);
    }

    #[test]
    fn auth_protocol_from_str_rejects_unknown() {
        let err = AuthProtocol::from_str("http").expect_err("should reject unknown value");
        assert!(matches!(err, ParseAuthProtocolError { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("rest"),
            "error should mention `rest`, got: {msg}"
        );
        assert!(
            msg.contains("grpc"),
            "error should mention `grpc`, got: {msg}"
        );
        assert!(
            msg.contains("http"),
            "error should echo the original input `http`, got: {msg}"
        );

        let err2 = AuthProtocol::from_str("graphql").expect_err("should reject unknown value");
        assert!(err2.to_string().contains("graphql"));

        let err3 = AuthProtocol::from_str("Rest!").expect_err("should reject unknown value");
        assert!(err3.to_string().contains("Rest!"));
    }

    #[test]
    fn auth_protocol_from_str_rejects_empty_string() {
        let err = AuthProtocol::from_str("").expect_err("should reject empty string");
        assert!(matches!(err, ParseAuthProtocolError { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("rest") && msg.contains("grpc"),
            "error should list accepted values, got: {msg}"
        );
    }

    // ---- UpdateLayerBackend FromStr pure parsing + Config::from_env ----
    //
    // Every accepted value, in every casing, maps to its own variant, so a
    // rename or drop of a knob value — `lua-watch-structural` is the
    // structural route's rollback — fails here rather than at an operator's
    // startup.

    #[test]
    fn update_layer_backend_from_str_accepts_each_value_case_insensitively() {
        let cases = [
            ("lua", UpdateLayerBackend::Lua),
            ("LUA", UpdateLayerBackend::Lua),
            ("Lua", UpdateLayerBackend::Lua),
            (
                "lua-watch-structural",
                UpdateLayerBackend::LuaWatchStructural,
            ),
            (
                "LUA-WATCH-STRUCTURAL",
                UpdateLayerBackend::LuaWatchStructural,
            ),
            (
                "Lua-Watch-Structural",
                UpdateLayerBackend::LuaWatchStructural,
            ),
            ("txn", UpdateLayerBackend::Txn),
            ("TXN", UpdateLayerBackend::Txn),
            ("Txn", UpdateLayerBackend::Txn),
        ];
        for (input, expected) in cases {
            assert_eq!(
                UpdateLayerBackend::from_str(input),
                Ok(expected),
                "{input:?} must parse as {expected:?}"
            );
        }
    }

    /// The accepted `LIVE_EDIT_UPDATE_LAYER_BACKEND` values an error message
    /// names, as whole words once the echoed input is removed — so `lua` is
    /// not found inside `lua-watch-structural`, nor inside the input itself.
    fn update_layer_backend_values_named(msg: &str, input: &str) -> Vec<&'static str> {
        let listed = msg.replacen(&format!("{input:?}"), "", 1);
        let words: std::collections::HashSet<&str> = listed
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
            .collect();
        ["lua", "lua-watch-structural", "txn"]
            .into_iter()
            .filter(|value| words.contains(value))
            .collect()
    }

    #[test]
    fn update_layer_backend_from_str_rejects_unknown_naming_every_accepted_value() {
        // Near-misses of the three values, plus the empty string: none may be
        // accepted as a prefix, a separator variant or a fragment.
        for input in [
            "",
            "lua-watch",
            "lua_watch_structural",
            "luawatchstructural",
            "watch-structural",
            "multi",
        ] {
            let err = UpdateLayerBackend::from_str(input)
                .expect_err("an unknown backend value must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("{input:?}")),
                "the error must echo the rejected input {input:?}; got: {msg}"
            );
            assert_eq!(
                update_layer_backend_values_named(&msg, input),
                ["lua", "lua-watch-structural", "txn"],
                "the error for {input:?} must name every accepted value; got: {msg}"
            );
        }
    }

    #[test]
    fn config_from_env_update_layer_backend_defaults_to_lua_and_reads_each_value() {
        let cases = [
            (None, UpdateLayerBackend::Lua),
            (Some("lua"), UpdateLayerBackend::Lua),
            (
                Some("Lua-Watch-Structural"),
                UpdateLayerBackend::LuaWatchStructural,
            ),
            (Some(" txn "), UpdateLayerBackend::Txn),
        ];
        for (value, expected) in cases {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_UPDATE_LAYER_BACKEND", value),
                ],
                || {
                    let cfg = Config::from_env_with(PodMemorySource::Reference)
                        .expect("a valid backend value must be accepted");
                    assert_eq!(
                        cfg.update_layer_backend, expected,
                        "LIVE_EDIT_UPDATE_LAYER_BACKEND={value:?}"
                    );
                },
            );
        }
    }

    #[test]
    fn config_from_env_rejects_unknown_update_layer_backend_naming_the_var_and_values() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_UPDATE_LAYER_BACKEND", Some("lua-structural")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("an unknown backend must refuse startup");
                assert!(
                    err.contains("LIVE_EDIT_UPDATE_LAYER_BACKEND"),
                    "the error must name the variable; got: {err}"
                );
                assert_eq!(
                    update_layer_backend_values_named(&err, "lua-structural"),
                    ["lua", "lua-watch-structural", "txn"],
                    "the error must name every accepted value; got: {err}"
                );
            },
        );
    }

    // ---- t3.943.oz.ar.5w: Config::from_env AuthProtocol integration ----

    #[test]
    fn config_from_env_default_auth_protocol_is_rest() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_PROTOCOL", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(cfg.auth_protocol, AuthProtocol::Rest);
            },
        );
    }

    #[test]
    fn config_from_env_explicit_rest() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_PROTOCOL", Some("rest")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with explicit rest");
                assert_eq!(cfg.auth_protocol, AuthProtocol::Rest);
            },
        );
    }

    #[test]
    fn config_from_env_explicit_grpc() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_PROTOCOL", Some("grpc")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with explicit grpc");
                assert_eq!(cfg.auth_protocol, AuthProtocol::Grpc);
            },
        );
    }

    #[test]
    fn config_from_env_explicit_rest_uppercase() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_PROTOCOL", Some("REST")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should accept uppercase REST");
                assert_eq!(cfg.auth_protocol, AuthProtocol::Rest);
            },
        );
    }

    #[test]
    fn config_from_env_explicit_grpc_uppercase() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_PROTOCOL", Some("GRPC")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should accept uppercase GRPC");
                assert_eq!(cfg.auth_protocol, AuthProtocol::Grpc);
            },
        );
    }

    // ---- b.1o4: LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES parsing ----

    #[test]
    fn config_from_env_auth_cache_max_entries_default_is_10_000() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(
                    cfg.auth_cache_max_entries, 10_000,
                    "missing LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES should default to 10_000 (documented default)",
                );
            },
        );
    }

    #[test]
    fn config_from_env_auth_cache_max_entries_parses_explicit_u64() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES", Some("25000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should accept explicit numeric max_entries");
                assert_eq!(cfg.auth_cache_max_entries, 25_000u64);
            },
        );
    }

    #[test]
    fn config_from_env_auth_cache_max_entries_above_ceiling_rejected() {
        // b.ohw medium replaces the previous "accepts u64" pin: values above
        // the 10M ceiling are now rejected at parse time so an operator
        // typo cannot drive the cache toward an OOM-shaped config. We pick
        // a value just above the ceiling (10M + 1) — large enough to clear
        // the boundary, small enough that the failure mode is the ceiling
        // check, not a parse error.
        let above_ceiling = 10_000_001u64;
        let value = above_ceiling.to_string();
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES", Some(value.as_str())),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("values above the 10M ceiling must be rejected (b.ohw medium)");
                assert!(
                    err.contains("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES"),
                    "error should name LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES, got: {err}"
                );
                assert!(
                    err.contains("ceiling"),
                    "error should mention the ceiling, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_auth_cache_max_entries_zero_rejected_with_meaningful_error() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero should be rejected — it would disable the cache");
                assert!(
                    err.contains("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES"),
                    "error should name LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES, got: {err}",
                );
                // Sanity check that the message explains why, not just that it
                // failed — operators who hit this need to know zero is invalid.
                let lowered = err.to_lowercase();
                assert!(
                    lowered.contains("zero")
                        || lowered.contains("0")
                        || lowered.contains("greater than"),
                    "error should explain that zero is not allowed, got: {err}",
                );
            },
        );
    }

    #[test]
    fn config_from_env_auth_cache_max_entries_non_numeric_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES", Some("not-a-number")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("non-numeric LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES"),
                    "error should name LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES, got: {err}",
                );
            },
        );
    }

    #[test]
    fn config_from_env_auth_cache_max_entries_negative_rejected() {
        // u64 parsing rejects negative numbers; this test guards that the
        // field type hasn't been silently changed to a signed integer.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES", Some("-1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("negative LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES"),
                    "error should name LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES, got: {err}",
                );
            },
        );
    }

    /// b.ohw "Test coverage to add" line: AUTH_CACHE_MAX_ENTRIES=100_000_000
    /// rejected at config parse. Distinct from the boundary test
    /// (`above_ceiling_rejected` at 10M+1) — this is the realistic
    /// operator-typo case (added an extra digit) that the ceiling guards
    /// against. We assert against the same ceiling-mention contract so a
    /// future ceiling change still surfaces a meaningful error.
    #[test]
    fn config_from_env_auth_cache_max_entries_100m_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES", Some("100000000")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "100M (10x ceiling) must be rejected at parse — bounds the operator-typo \
                     blast radius before the cache can balloon to ~8GB resident",
                );
                assert!(
                    err.contains("LIVE_EDIT_AUTH_CACHE_MAX_ENTRIES"),
                    "error must name the variable, got: {err}"
                );
                assert!(
                    err.contains("ceiling"),
                    "error must mention the ceiling so operators understand why, got: {err}"
                );
                assert!(
                    err.contains("10000000") || err.contains("10_000_000"),
                    "error must include the 10M ceiling value, got: {err}"
                );
            },
        );
    }

    /// b.ohw C-6: bare hostnames rejected at config parse for both auth
    /// endpoint variables. This is the fail-closed config validation that
    /// catches an operator typo before the service ever opens an auth
    /// connection — the same `validate_auth_endpoint_uri` runs again inside
    /// the auth-client constructors as defense-in-depth.

    #[test]
    fn config_from_env_storage_service_endpoint_bare_hostname_rejected() {
        temp_env::with_vars(
            [
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("storage.prod.internal"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "bare hostname for storage endpoint must be rejected (b.ohw C-6) — \
                     the pre-fix path silently rewrote to http:// and would have transmitted \
                     bearer tokens in plaintext",
                );
                assert!(
                    err.contains("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT"),
                    "error must name the offending variable, got: {err}"
                );
                assert!(
                    err.contains("explicit scheme"),
                    "error must surface the validator's reason, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_userinfo_endpoint_bare_hostname_rejected() {
        // Pin `userinfo_service_enabled=true` — post-b.g3z the scheme
        // validator on the userinfo endpoint only runs when the
        // userinfo backend is enabled, and the default flipped to
        // `false`. Without this pin the bare-hostname rejection test
        // would silently stop exercising the validator.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("userinfo.prod.internal"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("bare hostname for userinfo endpoint must be rejected (b.ohw C-6)");
                assert!(err.contains("LIVE_EDIT_USERINFO_ENDPOINT"), "got: {err}");
                assert!(err.contains("explicit scheme"), "got: {err}");
            },
        );
    }

    /// Both schemes (https://, http://) are accepted at config parse so
    /// operators can still configure HTTP-only auth backends in test
    /// clusters. This positive case complements the rejection tests above.
    #[test]
    fn config_from_env_explicit_https_endpoints_accepted() {
        temp_env::with_vars(
            [
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("https://storage.prod.internal"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("https://userinfo.prod.internal"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit https:// scheme must pass config validation");
                assert_eq!(
                    cfg.storage_service_endpoint.as_deref(),
                    Some("https://storage.prod.internal")
                );
                assert_eq!(
                    cfg.userinfo_endpoint.as_deref(),
                    Some("https://userinfo.prod.internal")
                );
            },
        );
    }

    #[test]
    fn config_from_env_invalid_auth_protocol_errors() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTH_PROTOCOL", Some("http")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("should fail with invalid AUTH_PROTOCOL value");
                assert!(
                    err.contains("LIVE_EDIT_AUTH_PROTOCOL"),
                    "error should name LIVE_EDIT_AUTH_PROTOCOL, got: {err}"
                );
            },
        );
    }

    // ---- t3.pev: LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED boolean parsing ----
    //
    // The catch-up switch is now a plain stdlib `bool` (b.pev), replacing the
    // former `CatchupExperiment { Off, Snapshot }` enum. Default is `true`
    // (corrected full-state snapshot — the shipped fix); `false` selects the
    // legacy per-commit reconstruction path. Mirrors the
    // `LIVE_EDIT_CHECKPOINT_ENABLED` boolean coverage above: stdlib
    // `FromStr for bool` accepts EXACTLY `"true"` / `"false"` and rejects all
    // aliases and casings, so a silent swap to a permissive parser — or an
    // accidental default flip — fails CI fast.

    #[test]
    fn config_from_env_catchup_snapshot_enabled_accepts_lowercase_true() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with explicit `true`");
                assert!(cfg.catchup_snapshot_enabled);
            },
        );
    }

    #[test]
    fn config_from_env_catchup_snapshot_enabled_accepts_lowercase_false() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", Some("false")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with explicit `false`");
                assert!(!cfg.catchup_snapshot_enabled);
            },
        );
    }

    #[test]
    fn config_from_env_catchup_snapshot_enabled_rejects_capitalized_true() {
        // Stdlib `FromStr for bool` is case-sensitive: only the exact
        // lowercase strings `"true"` and `"false"` parse. Pin this so a
        // regression that silently swaps in a case-insensitive parser
        // doesn't drift the contract out from under operators.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", Some("True")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`True` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED"),
                    "error should name the env var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_catchup_snapshot_enabled_rejects_uppercase_false() {
        // Same rationale as the capitalized-`True` test — stdlib
        // `FromStr for bool` rejects `"FALSE"`.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", Some("FALSE")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`FALSE` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED"),
                    "error should name the env var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_catchup_snapshot_enabled_rejects_yes_returns_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", Some("yes")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`yes` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED"),
                    "error should name LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_catchup_snapshot_enabled_rejects_numeric_one_returns_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", Some("1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`1` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED"),
                    "error should name LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_catchup_snapshot_enabled_rejects_empty_returns_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", Some("")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("empty string must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED"),
                    "error should name LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_default_catchup_snapshot_enabled_is_true() {
        // Regression guard for the b.pev default flip: with the env var UNSET
        // the config default is `true` (corrected snapshot ON) — the behavior
        // change this ticket ships. This is the test that would have failed
        // before the fix (former default was legacy/off) and passes after.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert!(
                    cfg.catchup_snapshot_enabled,
                    "default (env unset) must be true — corrected snapshot on"
                );
            },
        );
    }

    #[test]
    fn config_from_env_explicit_catchup_snapshot_enabled_false() {
        // `false` opts back into the legacy per-commit reconstruction path.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED", Some("false")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with explicit `false`");
                assert!(!cfg.catchup_snapshot_enabled);
            },
        );
    }

    // ---- LIVE_EDIT_SUBSCRIBER_POOL_SIZE parsing ----

    #[test]
    fn config_from_env_subscriber_pool_size_default_is_auto() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(
                    cfg.subscriber_pool_size, 0,
                    "missing LIVE_EDIT_SUBSCRIBER_POOL_SIZE should default to 0 (auto = cores × 2 at connect time); \
                     production sets an explicit ceiling via the Helm chart",
                );
            },
        );
    }

    #[test]
    fn config_from_env_subscriber_pool_size_parses_explicit_u32() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some("1024")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should accept explicit numeric subscriber_pool_size");
                assert_eq!(cfg.subscriber_pool_size, 1024u32);
            },
        );
    }

    #[test]
    fn config_from_env_subscriber_pool_size_zero_means_auto() {
        // Zero is accepted at parse time (it's a u32 value) — the "auto"
        // expansion happens inside `ValkeyClient::connect`. Parsing should
        // preserve the zero here.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some("0")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("zero subscriber_pool_size means 'auto' at connect");
                assert_eq!(cfg.subscriber_pool_size, 0u32);
            },
        );
    }

    #[test]
    fn config_from_env_subscriber_pool_size_non_numeric_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some("not-a-number")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("non-numeric LIVE_EDIT_SUBSCRIBER_POOL_SIZE should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_SUBSCRIBER_POOL_SIZE"),
                    "error should name LIVE_EDIT_SUBSCRIBER_POOL_SIZE, got: {err}",
                );
            },
        );
    }

    #[test]
    fn config_from_env_subscriber_pool_size_negative_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some("-1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("negative LIVE_EDIT_SUBSCRIBER_POOL_SIZE should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_SUBSCRIBER_POOL_SIZE"),
                    "error should name LIVE_EDIT_SUBSCRIBER_POOL_SIZE, got: {err}",
                );
            },
        );
    }

    // ---- t2.ebu.qb.57: LIVE_EDIT_LAYER_STREAM_MAXLEN parsing ----

    #[test]
    fn config_from_env_layer_stream_maxlen_default_is_4096() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LAYER_STREAM_MAXLEN", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(
                    cfg.layer_stream_maxlen, 4096,
                    "missing LIVE_EDIT_LAYER_STREAM_MAXLEN should default to 4096 (SDD-documented)",
                );
            },
        );
    }

    #[test]
    fn config_from_env_layer_stream_maxlen_parses_explicit_u64() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LAYER_STREAM_MAXLEN", Some("8192")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should accept explicit numeric layer_stream_maxlen");
                assert_eq!(cfg.layer_stream_maxlen, 8192u64);
            },
        );
    }

    #[test]
    fn config_from_env_layer_stream_maxlen_non_numeric_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LAYER_STREAM_MAXLEN", Some("not-a-number")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("non-numeric LIVE_EDIT_LAYER_STREAM_MAXLEN should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_LAYER_STREAM_MAXLEN"),
                    "error should name LIVE_EDIT_LAYER_STREAM_MAXLEN, got: {err}",
                );
            },
        );
    }

    #[test]
    fn config_from_env_layer_stream_maxlen_negative_rejected() {
        // u64 parsing rejects negative numbers; this test guards against the
        // field being silently widened to a signed integer.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LAYER_STREAM_MAXLEN", Some("-1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("negative LIVE_EDIT_LAYER_STREAM_MAXLEN should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_LAYER_STREAM_MAXLEN"),
                    "error should name LIVE_EDIT_LAYER_STREAM_MAXLEN, got: {err}",
                );
            },
        );
    }

    // ── b.qw6 AC #3: LIVE_EDIT_LAYER_STREAM_MAXLEN ceiling (2^53 - 1) ─────
    //
    // The Lua fast-path script reads the configured MAXLEN through Lua's
    // `tonumber(...)`, which round-trips integers through IEEE 754 f64.
    // Values above 2^53 - 1 silently round when crossed by that
    // conversion, altering the actual stream-trim behavior in ways that
    // operators cannot debug from logs. Reject at startup so misconfigured
    // values fail fast instead of nibbling at production retention.

    #[test]
    fn config_from_env_layer_stream_maxlen_at_lua_ceiling_accepted() {
        // 2^53 - 1 itself is on the safe side of the ceiling — the
        // `tonumber` round-trip preserves this value bit-for-bit, so
        // accepting it pins the inclusive boundary.
        let ceiling = (1u64 << 53) - 1;
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_LAYER_STREAM_MAXLEN",
                    Some(ceiling.to_string().as_str()),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("MAXLEN at 2^53-1 must be accepted (inclusive ceiling)");
                assert_eq!(cfg.layer_stream_maxlen, ceiling);
            },
        );
    }

    #[test]
    fn config_from_env_layer_stream_maxlen_above_lua_ceiling_rejected() {
        // 2^53 is the first value past the ceiling. Reject at startup
        // with a message that names the offending env var and the
        // ceiling so operators can fix the config without diving into
        // the Lua script.
        let above_ceiling = 1u64 << 53;
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_LAYER_STREAM_MAXLEN",
                    Some(above_ceiling.to_string().as_str()),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("MAXLEN above 2^53-1 must be rejected at startup");
                assert!(
                    err.contains("LIVE_EDIT_LAYER_STREAM_MAXLEN"),
                    "error must name the env var, got: {err}"
                );
                assert!(
                    err.contains("2^53") || err.contains("9007199254740991"),
                    "error must reference the 2^53 ceiling so operators can locate \
                     the constraint, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_layer_stream_maxlen_u64_max_rejected() {
        // u64::MAX is the worst-case fat-finger input. Confirm the
        // boundary check fires here too rather than producing a generic
        // parse error or — worse — silently accepting and round-tripping
        // through Lua's tonumber.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_LAYER_STREAM_MAXLEN",
                    Some(u64::MAX.to_string().as_str()),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("MAXLEN at u64::MAX must be rejected at startup");
                assert!(
                    err.contains("LIVE_EDIT_LAYER_STREAM_MAXLEN"),
                    "error must name the env var, got: {err}"
                );
            },
        );
    }

    // ── b.yvh: LIVE_EDIT_LAYER_STREAM_MAX_BYTES (the relay byte budget) ─────
    //
    // Default 128 MiB; `0` turns the budget off (the no-redeploy rollback to
    // count-only retention); the commit script reads it through `tonumber`, so
    // the same 2^53 − 1 ceiling as MAXLEN applies.

    /// `Config::from_env` with the required vars plus
    /// `LIVE_EDIT_LAYER_STREAM_MAX_BYTES` = `value` (`None` = unset).
    fn from_env_with_stream_max_bytes(value: Option<&str>) -> Result<Config, String> {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LAYER_STREAM_MAX_BYTES", value),
            ],
            || Config::from_env_with(PodMemorySource::Reference),
        )
    }

    #[test]
    fn config_from_env_layer_stream_max_bytes_default_is_128_mib() {
        let cfg = from_env_with_stream_max_bytes(None).expect("required vars set");
        assert_eq!(cfg.layer_stream_max_bytes, 134_217_728);
        assert_eq!(
            cfg.layer_stream_max_bytes, LAYER_STREAM_MAX_BYTES_DEFAULT,
            "the parse default is the documented constant"
        );
    }

    #[test]
    fn config_from_env_layer_stream_max_bytes_zero_turns_the_budget_off() {
        let cfg = from_env_with_stream_max_bytes(Some("0"))
            .expect("0 must be accepted: it is the no-redeploy rollback to count-only retention");
        assert_eq!(cfg.layer_stream_max_bytes, 0);
    }

    #[test]
    fn config_from_env_layer_stream_max_bytes_parses_explicit_u64() {
        let cfg = from_env_with_stream_max_bytes(Some("8388608")).expect("an explicit budget");
        assert_eq!(cfg.layer_stream_max_bytes, 8_388_608);
    }

    #[test]
    fn config_from_env_layer_stream_max_bytes_at_lua_ceiling_accepted() {
        let ceiling = (1u64 << 53) - 1;
        let cfg = from_env_with_stream_max_bytes(Some(ceiling.to_string().as_str()))
            .expect("2^53 - 1 must be accepted (inclusive ceiling)");
        assert_eq!(cfg.layer_stream_max_bytes, ceiling);
    }

    #[test]
    fn config_from_env_layer_stream_max_bytes_above_lua_ceiling_rejected() {
        // Accepted, it would make the commit script refuse every UpdateLayer
        // (`ENCODING`, a malformed budget slot) instead of failing startup.
        for above in [1u64 << 53, u64::MAX] {
            let err = from_env_with_stream_max_bytes(Some(above.to_string().as_str()))
                .expect_err("a budget above 2^53 - 1 must be rejected at startup");
            assert!(
                err.contains("LIVE_EDIT_LAYER_STREAM_MAX_BYTES"),
                "error must name the env var, got: {err}"
            );
            assert!(
                err.contains("2^53") || err.contains("9007199254740991"),
                "error must reference the 2^53 ceiling, got: {err}"
            );
        }
    }

    #[test]
    fn config_from_env_layer_stream_max_bytes_non_numeric_and_negative_rejected() {
        for value in ["128MiB", "not-a-number", "-1", "1.5", ""] {
            let err = from_env_with_stream_max_bytes(Some(value))
                .expect_err("a non-numeric budget must fail parse");
            assert!(
                err.contains("LIVE_EDIT_LAYER_STREAM_MAX_BYTES"),
                "{value:?}: error should name LIVE_EDIT_LAYER_STREAM_MAX_BYTES, got: {err}"
            );
        }
    }

    // ── LogFormat FromStr pure parsing (b.3z7 H-obs-4) ────────────────────────
    //
    // The FromStr impl is the only surface through which
    // LIVE_EDIT_LOG_FORMAT reaches the rest of the program, so covering it
    // at the pure-parsing level pins the contract (case-insensitive +
    // whitespace-tolerant) independently of any env-var plumbing.

    #[test]
    fn log_format_from_str_accepts_lowercase_json() {
        assert_eq!(LogFormat::from_str("json").unwrap(), LogFormat::Json);
    }

    #[test]
    fn log_format_from_str_accepts_lowercase_pretty() {
        assert_eq!(LogFormat::from_str("pretty").unwrap(), LogFormat::Pretty);
    }

    #[test]
    fn log_format_from_str_accepts_uppercase_variants() {
        assert_eq!(LogFormat::from_str("JSON").unwrap(), LogFormat::Json);
        assert_eq!(LogFormat::from_str("PRETTY").unwrap(), LogFormat::Pretty);
    }

    #[test]
    fn log_format_from_str_accepts_mixed_case_variants() {
        assert_eq!(LogFormat::from_str("Json").unwrap(), LogFormat::Json);
        assert_eq!(LogFormat::from_str("Pretty").unwrap(), LogFormat::Pretty);
        assert_eq!(LogFormat::from_str("jSoN").unwrap(), LogFormat::Json);
        assert_eq!(LogFormat::from_str("pReTtY").unwrap(), LogFormat::Pretty);
    }

    #[test]
    fn log_format_from_str_trims_surrounding_whitespace() {
        // A value surrounded by whitespace — e.g. copy-pasted from a
        // manifest block that quoted `" pretty "` — must parse cleanly.
        assert_eq!(
            LogFormat::from_str("  pretty  ").unwrap(),
            LogFormat::Pretty
        );
        assert_eq!(LogFormat::from_str("\tjson\n").unwrap(), LogFormat::Json);
    }

    #[test]
    fn log_format_from_str_rejects_unknown() {
        let err = LogFormat::from_str("logfmt").expect_err("should reject unknown value");
        assert!(matches!(err, ParseLogFormatError { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("json") && msg.contains("pretty"),
            "error should list accepted values, got: {msg}"
        );
        assert!(
            msg.contains("logfmt"),
            "error should echo the rejected input, got: {msg}"
        );
    }

    #[test]
    fn log_format_from_str_rejects_empty_string() {
        let err = LogFormat::from_str("").expect_err("should reject empty string");
        assert!(matches!(err, ParseLogFormatError { .. }));
    }

    // ── Config::from_env LogFormat integration (b.3z7 H-obs-4) ───────────────

    #[test]
    fn config_from_env_default_log_format_is_json() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with defaults");
            assert_eq!(
                cfg.log_format,
                LogFormat::Json,
                "production default must be JSON so log aggregators get structured records"
            );
        });
    }

    #[test]
    fn config_from_env_log_format_pretty_parsed() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LOG_FORMAT", Some("pretty")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with LIVE_EDIT_LOG_FORMAT=pretty");
                assert_eq!(cfg.log_format, LogFormat::Pretty);
            },
        );
    }

    #[test]
    fn config_from_env_log_format_case_insensitive() {
        for raw in ["PRETTY", "Pretty", "pRetty"] {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_LOG_FORMAT", Some(raw)),
                ],
                || {
                    let cfg = Config::from_env_with(PodMemorySource::Reference)
                        .unwrap_or_else(|e| panic!("should parse LIVE_EDIT_LOG_FORMAT={raw}: {e}"));
                    assert_eq!(cfg.log_format, LogFormat::Pretty);
                },
            );
        }
    }

    #[test]
    fn config_from_env_log_format_with_surrounding_whitespace_parses() {
        // LIVE_EDIT_LOG_FORMAT=" pretty " should trim-then-parse instead of
        // silently falling back to JSON. Covers the b.3z7 env-trim medium
        // as applied to LogFormat specifically.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LOG_FORMAT", Some("  pretty  ")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "whitespace-wrapped pretty should parse, not silently fall back to JSON",
                );
                assert_eq!(cfg.log_format, LogFormat::Pretty);
            },
        );
    }

    #[test]
    fn config_from_env_log_format_invalid_returns_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LOG_FORMAT", Some("logfmt")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("invalid LIVE_EDIT_LOG_FORMAT value should fail startup");
                assert!(
                    err.contains("LIVE_EDIT_LOG_FORMAT"),
                    "error should name LIVE_EDIT_LOG_FORMAT, got: {err}"
                );
            },
        );
    }

    // ── LIVE_EDIT_TRACE_FILTER ────────────────────────────────────────────────

    /// `Config::from_env` with `LIVE_EDIT_TRACE_FILTER` set to `value` (or
    /// unset) on top of the required variables.
    fn from_env_with_trace_filter(value: Option<&str>) -> Result<Config, String> {
        let mut vars = REQUIRED_VARS.to_vec();
        vars.push(("LIVE_EDIT_TRACE_FILTER", value));
        temp_env::with_vars(vars, || Config::from_env_with(PodMemorySource::Reference))
    }

    #[test]
    fn config_from_env_trace_filter_unset_defaults_to_live_edit_info() {
        let cfg = from_env_with_trace_filter(None).expect("defaults must load");
        assert_eq!(cfg.trace_filter, "live_edit=info");
        assert_eq!(cfg.trace_filter, DEFAULT_TRACE_FILTER);
    }

    #[test]
    fn config_from_env_trace_filter_is_trimmed() {
        let cfg = from_env_with_trace_filter(Some("  live_edit=debug,h2=off \t"))
            .expect("a whitespace-wrapped filter must load");
        assert_eq!(cfg.trace_filter, "live_edit=debug,h2=off");
    }

    /// An empty value is refused rather than read as "export nothing" (or,
    /// as an `EnvFilter` with no directives, the ERROR-only default).
    #[test]
    fn config_from_env_trace_filter_empty_fails_naming_the_variable() {
        for raw in ["", "   "] {
            let err = from_env_with_trace_filter(Some(raw))
                .expect_err("an empty LIVE_EDIT_TRACE_FILTER must fail startup");
            assert!(
                err.contains("LIVE_EDIT_TRACE_FILTER"),
                "error for {raw:?} must name LIVE_EDIT_TRACE_FILTER; got: {err}"
            );
        }
    }

    /// An unparseable directive fails startup, instead of `EnvFilter::new`'s
    /// silent drop of the bad directive.
    #[test]
    fn config_from_env_trace_filter_unparseable_fails_naming_the_variable() {
        let err = from_env_with_trace_filter(Some("live_edit=[bad"))
            .expect_err("an unparseable LIVE_EDIT_TRACE_FILTER must fail startup");
        assert!(
            err.contains("LIVE_EDIT_TRACE_FILTER"),
            "error must name LIVE_EDIT_TRACE_FILTER; got: {err}"
        );
    }

    // ── TracingSettings / Config parity ──────────────────────────────────────
    //
    // `main.rs` builds the subscriber from `TracingSettings::from_env` before
    // `Config::from_env` runs. The two must read the logging variables the
    // same way, or `Config`'s `log_level` / `log_format` / `trace_filter`
    // would describe a subscriber that was never built.

    /// `TracingSettings::from_env` and `Config::from_env` under one
    /// environment: the required variables plus `logging`.
    fn tracing_settings_and_config_from_env(
        logging: [(&str, Option<&str>); 3],
    ) -> (Result<TracingSettings, String>, Result<Config, String>) {
        let mut vars: Vec<(&str, Option<&str>)> = REQUIRED_VARS.to_vec();
        vars.extend(logging);
        temp_env::with_vars(vars, || {
            (
                TracingSettings::from_env(),
                Config::from_env_with(PodMemorySource::Reference),
            )
        })
    }

    /// Assert `Config`'s three logging fields equal `settings`.
    fn assert_config_matches_tracing_settings(cfg: &Config, settings: &TracingSettings) {
        assert_eq!(
            TracingSettings {
                log_level: cfg.log_level.clone(),
                log_format: cfg.log_format,
                trace_filter: cfg.trace_filter.clone(),
            },
            *settings,
            "Config's logging fields must equal the TracingSettings the subscriber is built from"
        );
    }

    #[test]
    fn tracing_settings_from_env_matches_config_at_defaults() {
        let (settings, cfg) = tracing_settings_and_config_from_env([
            ("LIVE_EDIT_LOG_LEVEL", None),
            ("LIVE_EDIT_LOG_FORMAT", None),
            ("LIVE_EDIT_TRACE_FILTER", None),
        ]);
        let settings = settings.expect("TracingSettings must load at defaults");
        assert_eq!(
            settings,
            TracingSettings {
                log_level: "info".to_string(),
                log_format: LogFormat::Json,
                trace_filter: DEFAULT_TRACE_FILTER.to_string(),
            }
        );
        assert_config_matches_tracing_settings(&cfg.expect("Config must load"), &settings);
    }

    #[test]
    fn tracing_settings_from_env_matches_config_for_whitespace_wrapped_values() {
        let (settings, cfg) = tracing_settings_and_config_from_env([
            ("LIVE_EDIT_LOG_LEVEL", Some("  debug \t")),
            ("LIVE_EDIT_LOG_FORMAT", Some(" Pretty ")),
            (
                "LIVE_EDIT_TRACE_FILTER",
                Some("  live_edit=debug,h2=off \t"),
            ),
        ]);
        let settings = settings.expect("TracingSettings must load trimmed values");
        assert_eq!(
            settings,
            TracingSettings {
                log_level: "debug".to_string(),
                log_format: LogFormat::Pretty,
                trace_filter: "live_edit=debug,h2=off".to_string(),
            }
        );
        assert_config_matches_tracing_settings(&cfg.expect("Config must load"), &settings);
    }

    #[test]
    fn tracing_settings_from_env_invalid_log_format_fails_with_configs_error() {
        let (settings, cfg) = tracing_settings_and_config_from_env([
            ("LIVE_EDIT_LOG_LEVEL", None),
            ("LIVE_EDIT_LOG_FORMAT", Some("logfmt")),
            ("LIVE_EDIT_TRACE_FILTER", None),
        ]);
        let err = settings.expect_err("an invalid LIVE_EDIT_LOG_FORMAT must fail");
        assert!(
            err.contains("LIVE_EDIT_LOG_FORMAT"),
            "error must name LIVE_EDIT_LOG_FORMAT; got: {err}"
        );
        assert_eq!(
            Err(err),
            cfg.map(|_| ()),
            "main.rs prints whichever of the two fails first, so their messages must match"
        );
    }

    #[test]
    fn tracing_settings_from_env_invalid_trace_filter_fails_with_configs_error() {
        let (settings, cfg) = tracing_settings_and_config_from_env([
            ("LIVE_EDIT_LOG_LEVEL", None),
            ("LIVE_EDIT_LOG_FORMAT", None),
            ("LIVE_EDIT_TRACE_FILTER", Some("live_edit=[bad")),
        ]);
        let err = settings.expect_err("an unparseable LIVE_EDIT_TRACE_FILTER must fail");
        assert!(
            err.contains("LIVE_EDIT_TRACE_FILTER"),
            "error must name LIVE_EDIT_TRACE_FILTER; got: {err}"
        );
        assert_eq!(
            Err(err),
            cfg.map(|_| ()),
            "main.rs prints whichever of the two fails first, so their messages must match"
        );
    }

    #[test]
    fn tracing_settings_from_env_loads_without_the_rest_of_the_configuration() {
        // The subscriber must be installable before the rest of the
        // configuration loads, so a configuration error (here: the authz
        // endpoint the default auth shape requires) cannot stop it. This is
        // the `env -i` shape the Dockerfile's startup gates run.
        temp_env::with_vars(
            [
                ("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", None::<&str>),
                ("LIVE_EDIT_USERINFO_ENDPOINT", None),
                ("LIVE_EDIT_LOG_LEVEL", None),
                ("LIVE_EDIT_LOG_FORMAT", None),
                ("LIVE_EDIT_TRACE_FILTER", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("precondition: the rest of the configuration must fail to load");
                assert!(
                    err.contains("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT"),
                    "precondition: the failure must be the missing authz endpoint, got: {err}"
                );
                TracingSettings::from_env()
                    .expect("TracingSettings must load whatever the rest of the configuration is");
            },
        );
    }

    // ── Sweeper interval override (b.3z7 H-obs-5) ─────────────────────────────

    #[test]
    fn config_from_env_default_sweeper_interval_matches_presence_default() {
        // The default must track `presence::sweeper::DEFAULT_SWEEPER_INTERVAL_SECS`
        // so the fix preserves pre-b.3z7 behavior when the env var is absent.
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with defaults");
            assert_eq!(
                cfg.sweeper_interval_secs,
                crate::presence::sweeper::DEFAULT_SWEEPER_INTERVAL_SECS,
                "default sweeper interval must match the presence-module constant so \
                 un-configured replicas keep the pre-b.3z7 15s cadence"
            );
        });
    }

    #[test]
    fn config_from_env_explicit_sweeper_interval_overrides_default() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SWEEPER_INTERVAL_SECS", Some("2")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with LIVE_EDIT_SWEEPER_INTERVAL_SECS=2");
                assert_eq!(
                    cfg.sweeper_interval_secs, 2,
                    "explicit env override must take effect so operators can tune sweep cadence \
                     without a code change"
                );
            },
        );
    }

    #[test]
    fn config_from_env_invalid_sweeper_interval_returns_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SWEEPER_INTERVAL_SECS", Some("not-a-number")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("non-numeric LIVE_EDIT_SWEEPER_INTERVAL_SECS should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_SWEEPER_INTERVAL_SECS"),
                    "error should name LIVE_EDIT_SWEEPER_INTERVAL_SECS, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_zero_sweeper_interval_returns_err_naming_var() {
        // Zero would panic `tokio::time::interval` in the presence sweeper's
        // spawn, so the replica would run with no crash-recovery sweep: fail
        // at startup instead, naming the variable to fix.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SWEEPER_INTERVAL_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero LIVE_EDIT_SWEEPER_INTERVAL_SECS must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_SWEEPER_INTERVAL_SECS"),
                    "error must name the env var so operators know what to fix, got: {err}"
                );
            },
        );
    }

    // ── session_tokens sweep interval (b.j5t item 5 follow-up) ────────────────
    //
    // `LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS` lets operators tune the
    // cadence of the `session_tokens` crash-recovery sweep (see
    // `grpc::session::spawn_session_tokens_sweeper`). The default is the
    // 60 s `SESSION_TOKENS_SWEEP_INTERVAL_SECS` constant, the env var
    // overrides, and zero is rejected because a zero-interval
    // `tokio::time::interval` panics + would effectively disable the sweep.

    #[test]
    fn config_from_env_default_session_tokens_sweep_interval_matches_constant() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg =
                Config::from_env_with(PodMemorySource::Reference).expect("defaults must succeed");
            assert_eq!(
                cfg.session_tokens_sweep_interval_secs,
                crate::grpc::session::SESSION_TOKENS_SWEEP_INTERVAL_SECS,
                "default must track the module-level constant so unconfigured \
                 replicas keep the documented 60 s cadence"
            );
        });
    }

    #[test]
    fn config_from_env_explicit_session_tokens_sweep_interval_overrides_default() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS", Some("7")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS must succeed");
                assert_eq!(
                    cfg.session_tokens_sweep_interval_secs, 7,
                    "operator override must reach the sweeper without a code change"
                );
            },
        );
    }

    #[test]
    fn config_from_env_zero_session_tokens_sweep_interval_returns_err_naming_var() {
        // Zero would panic `tokio::time::interval` and effectively disable
        // the sweep — fail fast at startup so operators don't ship a replica
        // with the crash-recovery sweep silently broken.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "zero LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS must be rejected",
                );
                assert!(
                    err.contains("LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS"),
                    "error must name the env var so operators know what to fix, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_invalid_session_tokens_sweep_interval_returns_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS",
                    Some("not-a-number"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "non-numeric LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS must fail parse",
                );
                assert!(
                    err.contains("LIVE_EDIT_SESSION_TOKENS_SWEEP_INTERVAL_SECS"),
                    "error must name the env var, got: {err}"
                );
            },
        );
    }

    // ── Shutdown overall budget (b.3z7 C-2) ───────────────────────────────────

    #[test]
    fn config_from_env_default_shutdown_overall_budget_is_30() {
        // The 30 s default is one term of the minimum termination grace —
        // the drain, the bound on the server after GOAWAY, this budget and
        // the final flush; 76.25 s at the defaults, under the Helm chart's
        // 80 s `terminationGracePeriodSeconds`
        // (`grpc::transport::minimum_termination_grace`).
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with defaults");
            assert_eq!(cfg.shutdown_overall_budget_secs, 30);
        });
    }

    #[test]
    fn config_from_env_explicit_shutdown_overall_budget_overrides_default() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS", Some("45")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS=45");
                assert_eq!(cfg.shutdown_overall_budget_secs, 45);
            },
        );
    }

    /// Shutdown budgets must satisfy
    /// `shutdown_overall_budget_secs >= shutdown_timeout_secs + 15` —
    /// otherwise the overall timeout can fire before the per-task
    /// drains + Valkey quit have room to run. A
    /// misconfig is a hard startup error so operators can't ship a
    /// replica that will always SIGKILL mid-cleanup.
    #[test]
    fn config_from_env_shutdown_budget_below_floor_returns_err() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                // overall = 10, timeout = 5 → floor is 20, so 10 < 20 fails.
                ("LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS", Some("10")),
                ("LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS", Some("5")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("budget below timeout+15 headroom must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS"),
                    "error must name the overall-budget env var so operators know what to raise, got: {err}"
                );
                assert!(
                    err.contains("LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS"),
                    "error must also name the timeout env var (the other side of the invariant), got: {err}"
                );
            },
        );
    }

    /// The floor is inclusive: `overall == timeout + 15` is accepted.
    /// Protects against an off-by-one in the validator that would
    /// either reject a well-sized default config or accept a
    /// one-second-too-small one.
    #[test]
    fn config_from_env_shutdown_budget_exactly_at_floor_accepted() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS", Some("20")),
                ("LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS", Some("5")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("overall == timeout + 15 must be accepted");
                assert_eq!(cfg.shutdown_overall_budget_secs, 20);
                assert_eq!(cfg.shutdown_timeout_secs, 5);
            },
        );
    }

    /// The floor's headroom is 15 s, not less: `overall == timeout + 14` is
    /// refused. With the exactly-at-floor test above this pins the headroom at
    /// 15 from both sides — the room `ValkeyClient::quit()`'s sequential
    /// internal bounds (11.5 s) need, plus slack.
    #[test]
    fn config_from_env_shutdown_budget_one_below_floor_returns_err() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS", Some("19")),
                ("LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS", Some("5")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("overall == timeout + 14 is below the 15 s headroom");
                assert!(
                    err.contains("Got 19 < 20"),
                    "the error must state the budget and the floor (timeout + 15), got: {err}"
                );
            },
        );
    }

    // ── Paced stream ending on shutdown ────────────────────────────────────────

    /// The paced ending's two settings default to 25 endings per second over
    /// at most 20 s.
    #[test]
    fn config_from_env_default_shutdown_stream_end_pacing_is_25_per_sec_over_20_s() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg =
                Config::from_env_with(PodMemorySource::Reference).expect("defaults must succeed");
            assert_eq!(cfg.shutdown_stream_end_rate, 25);
            assert_eq!(cfg.shutdown_stream_end_max_window_secs, 20);
        });
    }

    /// Explicit values override the defaults, and `0` — the rollback switch
    /// that ends every live stream at once — is accepted for each.
    #[test]
    fn config_from_env_shutdown_stream_end_pacing_accepts_explicit_values_and_zero() {
        for (rate, window) in [("100", "5"), ("0", "20"), ("25", "0"), ("0", "0")] {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_SHUTDOWN_STREAM_END_RATE", Some(rate)),
                    (
                        "LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS",
                        Some(window),
                    ),
                ],
                || {
                    let cfg = Config::from_env_with(PodMemorySource::Reference)
                        .unwrap_or_else(|e| panic!("rate={rate} window={window}: {e}"));
                    assert_eq!(cfg.shutdown_stream_end_rate.to_string(), rate);
                    assert_eq!(cfg.shutdown_stream_end_max_window_secs.to_string(), window);
                },
            );
        }
    }

    /// A malformed, negative or out-of-range (above `u32`) value is refused at
    /// startup, and the error names the variable.
    #[test]
    fn config_from_env_shutdown_stream_end_pacing_rejects_bad_values_naming_the_variable() {
        for var in [
            "LIVE_EDIT_SHUTDOWN_STREAM_END_RATE",
            "LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS",
        ] {
            for bad in ["fast", "-1", "2.5", "4294967296"] {
                temp_env::with_vars(
                    [
                        (
                            "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                            Some("http://auth.example.com"),
                        ),
                        (
                            "LIVE_EDIT_USERINFO_ENDPOINT",
                            Some("http://userinfo.example.com"),
                        ),
                        (var, Some(bad)),
                    ],
                    || {
                        let err = Config::from_env_with(PodMemorySource::Reference)
                            .expect_err("a bad pacing value must be refused at startup");
                        assert!(
                            err.contains(var),
                            "{var}={bad}: the error must name the variable; got: {err}"
                        );
                    },
                );
            }
        }
    }

    // ── Health probe interval / timeout (b.avd item 9) ─────────────────────────
    //
    // Operators can tune `LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS` and
    // `LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS` to trade detection latency
    // against backend QPS. Both reject zero; timeout must be strictly less
    // than interval so a slow probe can't queue past the next tick.

    #[test]
    fn config_from_env_default_health_probe_interval_is_5000ms() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg =
                Config::from_env_with(PodMemorySource::Reference).expect("defaults must succeed");
            assert_eq!(cfg.health_probe_interval_ms, 5_000);
            assert_eq!(cfg.health_probe_timeout_ms, 2_000);
        });
    }

    #[test]
    fn config_from_env_explicit_health_probe_interval_and_timeout_override_defaults() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS", Some("1000")),
                ("LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS", Some("500")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit values must succeed");
                assert_eq!(cfg.health_probe_interval_ms, 1_000);
                assert_eq!(cfg.health_probe_timeout_ms, 500);
            },
        );
    }

    #[test]
    fn config_from_env_health_probe_interval_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero interval must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS"),
                    "got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_health_probe_timeout_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero timeout must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS"),
                    "got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_health_probe_timeout_ge_interval_rejected() {
        // timeout = interval → reject (a probe that hangs equal to the
        // next-tick window means the loop perpetually skips ticks).
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS", Some("2000")),
                ("LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS", Some("2000")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("timeout == interval must be rejected (b.avd item 9)");
                assert!(
                    err.contains("LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS")
                        && err.contains("LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS"),
                    "error must name both env vars, got: {err}"
                );
            },
        );

        // timeout > interval → reject too.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS", Some("1000")),
                ("LIVE_EDIT_HEALTH_PROBE_TIMEOUT_MS", Some("5000")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("timeout > interval must be rejected");
                assert!(err.contains("LIVE_EDIT_HEALTH_PROBE"), "got: {err}");
            },
        );
    }

    // ── Presence fan-out capacity (LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY) ────

    /// `REQUIRED_VARS` plus the capacity variable, pinned to `value` (unset
    /// for `None`, so an ambient copy cannot satisfy the default assertion).
    fn presence_capacity_env(value: Option<&str>) -> [(&'static str, Option<&str>); 4] {
        let [a, b, c] = REQUIRED_VARS;
        [a, b, c, ("LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY", value)]
    }

    #[test]
    fn config_from_env_presence_broadcast_capacity_defaults_to_the_router_constant() {
        temp_env::with_vars(presence_capacity_env(None), || {
            let cfg =
                Config::from_env_with(PodMemorySource::Reference).expect("defaults must parse");
            assert_eq!(
                cfg.presence_broadcast_capacity,
                crate::presence::DEFAULT_BROADCAST_CHANNEL_CAPACITY,
                "the environment default and the router's code default must be one constant"
            );
        });
    }

    #[test]
    fn config_from_env_presence_broadcast_capacity_zero_rejected_with_explanation() {
        temp_env::with_vars(presence_capacity_env(Some("0")), || {
            let err = Config::from_env_with(PodMemorySource::Reference)
                .expect_err("a zero capacity must be rejected");
            assert!(
                err.contains("LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY"),
                "error must name the env var, got: {err}"
            );
            assert!(
                err.contains("must be greater than 0") && err.contains("would be lost"),
                "error must say why zero is refused (every event lost), got: {err}"
            );
        });
    }

    #[test]
    fn config_from_env_presence_broadcast_capacity_explicit_value_is_kept_exactly() {
        // The capacity is applied exactly, so a value that is not a power of
        // two must reach the router unchanged.
        temp_env::with_vars(presence_capacity_env(Some("1000")), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("a positive capacity must parse");
            assert_eq!(cfg.presence_broadcast_capacity, 1000);
        });
    }

    // ── Presence lease TTL / refresh validation (b.w29 item 3) ────────────────
    //
    // The Engineer picked "Path A" (raise the defaults): `TTL=60 / refresh=15`
    // (up from `TTL=30 / refresh=10`) with the rule `refresh × 2 ≤ ttl` so a
    // single missed refresh tick caused by a tokio scheduling pause cannot
    // evict a healthy session. These tests pin (a) the post-fix defaults,
    // (b) zero-rejection for both knobs, and (c) the `refresh × 2 ≤ ttl`
    // invariant enforcement.

    #[test]
    fn config_from_env_presence_lease_defaults_post_b_w29() {
        // No env override: defaults must be the b.w29 values (60 / 15) — if
        // either constant is silently changed back to the pre-fix value (30
        // / 10) this assertion fires and reminds the change-author to also
        // re-evaluate the `sweeper_interval + lease_ttl` recovery upper
        // bound documented in `src/presence/mod.rs`.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
            ],
            || {
                let cfg =
                    Config::from_env_with(PodMemorySource::Reference).expect("defaults must parse");
                assert_eq!(
                    cfg.presence_lease_ttl_secs, 60,
                    "default presence lease TTL must be 60 s post-b.w29 \
                     (was 30 s pre-fix; raising it preserves the recovery \
                     upper bound while quadrupling the scheduler-jitter \
                     headroom)"
                );
                assert_eq!(
                    cfg.presence_refresh_interval_secs, 15,
                    "default refresh interval must be 15 s post-b.w29 \
                     (was 10 s pre-fix; matches sweeper interval so every \
                     sweep tick observes a lease refreshed in the past \
                     window)"
                );
            },
        );
    }

    #[test]
    fn config_from_env_presence_lease_ttl_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_PRESENCE_LEASE_TTL_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero TTL must be rejected: SET ... EX 0 is a Valkey syntax error");
                assert!(
                    err.contains("LIVE_EDIT_PRESENCE_LEASE_TTL_SECS"),
                    "error must name the env var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_presence_refresh_interval_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "zero refresh must be rejected: would queue refreshes back-to-back \
                     and saturate the data pool",
                );
                assert!(
                    err.contains("LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS"),
                    "error must name the env var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_presence_lease_refresh_ratio_violation_rejected() {
        // refresh × 2 > ttl → reject. Pre-b.w29 the implicit ratio was
        // 30 / 10 = 3× and the explicit floor was unchecked; a single
        // missed refresh let the lease lapse. This test pins the fix:
        // operator-supplied values that violate the rule of thumb must
        // fail boot, not silently degrade in steady state.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_PRESENCE_LEASE_TTL_SECS", Some("10")),
                ("LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS", Some("6")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("refresh × 2 (12) > ttl (10) violates the b.w29 rule of thumb");
                assert!(
                    err.contains("LIVE_EDIT_PRESENCE_LEASE_TTL_SECS")
                        && err.contains("LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS"),
                    "error must name both env vars so the operator can \
                     see which knob to adjust, got: {err}"
                );
            },
        );

        // refresh × 2 == ttl → ACCEPT (the rule is "≤", not "<"). This is
        // the boundary case operators are most likely to land on — pin
        // it so a future tightening to strict `<` is a deliberate choice.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_PRESENCE_LEASE_TTL_SECS", Some("10")),
                ("LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS", Some("5")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("refresh × 2 == ttl must be accepted (rule is `≤`, not `<`)");
                assert_eq!(cfg.presence_lease_ttl_secs, 10);
                assert_eq!(cfg.presence_refresh_interval_secs, 5);
            },
        );
    }

    #[test]
    fn config_from_env_presence_lease_refresh_explicit_values_within_band_accepted() {
        // Sanity: a generous operator override (TTL=120 / refresh=20) must
        // succeed — ratio 6× is well within `refresh × 2 ≤ ttl`. Pins that
        // the validator only fires on actual violations, not on any
        // non-default configuration.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_PRESENCE_LEASE_TTL_SECS", Some("120")),
                ("LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS", Some("20")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit values inside the band must parse");
                assert_eq!(cfg.presence_lease_ttl_secs, 120);
                assert_eq!(cfg.presence_refresh_interval_secs, 20);
            },
        );
    }

    // ── JoinSession client keepalive interval ────────────────────────────────
    //
    // `LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS`: default 15000, any `u64`
    // from `MIN_PRESENCE_KEEPALIVE_INTERVAL_MS` (1000) up, `0` accepted as
    // "off" (the no-redeploy rollback); 1–999 and anything that is not a
    // `u64` refused at startup naming the variable.

    /// [`REQUIRED_VARS`] plus the keepalive interval pinned to `value` (unset
    /// for `None`, so an ambient copy cannot reach the test).
    fn keepalive_interval_env(value: Option<&str>) -> Vec<(&str, Option<&str>)> {
        let mut env = REQUIRED_VARS.to_vec();
        env.push(("LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS", value));
        env
    }

    #[test]
    fn config_from_env_presence_keepalive_interval_defaults_to_15000_ms() {
        temp_env::with_vars(keepalive_interval_env(None), || {
            let cfg =
                Config::from_env_with(PodMemorySource::Reference).expect("defaults must parse");
            assert_eq!(
                cfg.presence_keepalive_interval_ms, 15_000,
                "the default interval is 15 s, so the deadline is 45 s"
            );
            assert_eq!(
                cfg.presence_keepalive_interval_ms,
                crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS
            );
        });
    }

    #[test]
    fn config_from_env_presence_keepalive_interval_takes_a_set_value() {
        temp_env::with_vars(keepalive_interval_env(Some("2500")), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("a set interval must parse");
            assert_eq!(cfg.presence_keepalive_interval_ms, 2_500);
        });
    }

    /// `0` is accepted, unlike the presence lease knobs, and is exempt from
    /// the 1000 ms floor: it turns the client keepalive off.
    #[test]
    fn config_from_env_presence_keepalive_interval_zero_is_accepted_as_off() {
        temp_env::with_vars(keepalive_interval_env(Some("0")), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("0 turns the keepalive off and must parse");
            assert_eq!(cfg.presence_keepalive_interval_ms, 0);
        });
    }

    /// The floor: a non-zero interval under
    /// [`crate::presence::MIN_PRESENCE_KEEPALIVE_INTERVAL_MS`] (1000) — what a
    /// value meant as seconds produces — is refused at startup, naming the
    /// variable and its unit, at both ends of the refused range.
    #[test]
    fn config_from_env_presence_keepalive_interval_below_1000_ms_is_rejected_naming_the_unit() {
        assert_eq!(crate::presence::MIN_PRESENCE_KEEPALIVE_INTERVAL_MS, 1_000);
        for value in ["1", "15", "999"] {
            temp_env::with_vars(keepalive_interval_env(Some(value)), || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(&format!(
                    "LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS={value} is under the 1000 ms floor \
                     and must be rejected"
                ));
                assert!(
                    err.contains("LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS")
                        && err.contains("milliseconds"),
                    "the error must name the env var and say milliseconds, got: {err}"
                );
            });
        }
    }

    /// The floor itself is accepted: 1000 ms is the smallest non-zero
    /// interval.
    #[test]
    fn config_from_env_presence_keepalive_interval_of_exactly_1000_ms_is_accepted() {
        temp_env::with_vars(keepalive_interval_env(Some("1000")), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("the 1000 ms floor itself must parse");
            assert_eq!(cfg.presence_keepalive_interval_ms, 1_000);
        });
    }

    #[test]
    fn config_from_env_presence_keepalive_interval_not_a_u64_is_rejected_naming_the_variable() {
        for value in ["fast", "-1", "1.5", "15s"] {
            temp_env::with_vars(keepalive_interval_env(Some(value)), || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(&format!(
                    "LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS={value:?} must be rejected"
                ));
                assert!(
                    err.contains("LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS"),
                    "the error must name the env var, got: {err}"
                );
            });
        }
    }

    // ── Env-var whitespace trim (b.3z7 medium) ────────────────────────────────
    //
    // Pre-fix, a manifest value like `LIVE_EDIT_VALKEY_URL=" redis://... "`
    // would pass through the whitespace verbatim and downstream fred
    // parsing would fail with an opaque error. The fix trims both required
    // and optional env vars in `required` / `optional_parsed` — these
    // tests pin that behavior on representative vars so regressions
    // surface immediately instead of as silent runtime failures.

    #[test]
    fn config_from_env_trims_whitespace_from_valkey_url() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_URL", Some("  redis://valkey.test:6379\n")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("trimmed valkey_url should parse");
                assert_eq!(
                    cfg.valkey_url, "redis://valkey.test:6379",
                    "valkey_url must be trimmed so surrounding whitespace from manifest \
                     quoting doesn't reach the fred parser"
                );
            },
        );
    }

    #[test]
    fn config_from_env_trims_whitespace_from_log_level() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_LOG_LEVEL", Some("  debug  ")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("trimmed log_level should parse");
                assert_eq!(cfg.log_level, "debug");
            },
        );
    }

    #[test]
    fn config_from_env_trims_whitespace_from_service_label() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_LABEL", Some(" pod-abc \n")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("trimmed service_label should parse");
                assert_eq!(cfg.service_label, "pod-abc");
            },
        );
    }

    #[test]
    fn config_from_env_trims_whitespace_from_numeric_vars() {
        // optional_parsed should trim before parsing so a whitespace-wrapped
        // numeric env var doesn't surface as "invalid value".
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_PORT", Some("  51000  ")),
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some("\t2500\n")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("trimmed numeric values should parse");
                assert_eq!(cfg.port, 51000);
                assert_eq!(cfg.rpc_deadline_ms, 2500);
            },
        );
    }

    #[test]
    fn config_from_env_trims_whitespace_from_required_userinfo_endpoint() {
        // The `required` helper must trim — otherwise a leading/trailing
        // space in LIVE_EDIT_USERINFO_ENDPOINT would be passed through to
        // the HTTP client builder and fail with an opaque URL parse error.
        // Pin `userinfo_service_enabled=true` so the `required` branch is
        // exercised (post-b.g3z the default is `false`, which routes
        // through `env::var(...).ok()` and would silently skip the trim).
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("\thttp://userinfo.example.com\n"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("trimmed userinfo endpoint should parse");
                assert_eq!(
                    cfg.userinfo_endpoint.as_deref(),
                    Some("http://userinfo.example.com")
                );
            },
        );
    }

    #[test]
    fn config_from_env_trims_whitespace_from_storage_service_endpoint_when_authz_enabled() {
        // With authz enabled, STORAGE_SERVICE_ENDPOINT flows through the
        // `required` helper and therefore is trimmed. Covers the
        // authz-enabled branch so the whitespace-trim guarantee holds
        // end-to-end on the path that actually matters in production.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("  http://auth.example.com  "),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("trimmed auth endpoint should parse");
                assert_eq!(
                    cfg.storage_service_endpoint,
                    Some("http://auth.example.com".to_string())
                );
            },
        );
    }

    // ---- b.avd second follow-up: pool-size vs store_values inflight ----

    /// `LIVE_EDIT_VALKEY_POOL_SIZE = 7` is below the
    /// `STORE_VALUES_MAX_INFLIGHT = 8` floor and must be rejected at
    /// startup. The fail-fast surface beats the silent stall hazard of a
    /// `CreateLayer` whose blob wave is WIDER than the pool it has to fit
    /// into — post-b.rej the wave rides the multiplexed data pool, so below
    /// the floor a big upload has a blob on every connection with more still
    /// queued, and concurrent `UpdateLayer` / `CreateSession` callers
    /// head-of-line-block behind the transfers.
    #[test]
    fn config_from_env_valkey_pool_size_below_store_values_inflight_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("7")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "valkey_pool_size below STORE_VALUES_MAX_INFLIGHT must be rejected",
                );
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_POOL_SIZE"),
                    "error must name the env var so operators know what to fix, got: {err}"
                );
                assert!(
                    err.contains("8"),
                    "error must name the floor (STORE_VALUES_MAX_INFLIGHT = 8), got: {err}"
                );
                assert!(
                    err.contains("store_values"),
                    "error must explain which contract binds (the store_values wave), got: {err}"
                );
            },
        );
    }

    /// Boundary case: exactly the floor is accepted. The bound is
    /// inclusive — eight connections are exactly enough for one full
    /// `store_values` wave with no headroom for concurrent callers, which is
    /// a tail-latency concern rather than the cliff the validation guards
    /// against. Accepted, and deliberately not claimed to be comfortable:
    /// at exactly the floor a full wave can momentarily carry a blob on
    /// every data-pool connection. It is also the value the auto path
    /// resolves to on a host reporting ≤ 4 cores, so rejecting equality
    /// would reject that host.
    #[test]
    fn config_from_env_valkey_pool_size_equal_to_store_values_inflight_accepted() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("8")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("valkey_pool_size == STORE_VALUES_MAX_INFLIGHT must be accepted");
                assert_eq!(cfg.valkey_pool_size, 8);
            },
        );
    }

    /// Boundary case: well above the floor is accepted. Pins the
    /// production Helm value (`valkey.poolSize = 64`, see
    /// `helm/live-edit-service/values.yaml`) so a future change to the
    /// validation can't silently raise the floor past production — the
    /// gate's own message points operators at that number as the shape to
    /// aim for, so a build that rejected it would contradict its own advice.
    #[test]
    fn config_from_env_valkey_pool_size_production_default_accepted() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("64")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("production valkey_pool_size (Helm's 64) must be accepted");
                assert_eq!(cfg.valkey_pool_size, 64);
            },
        );
    }

    /// Zero means "auto = cores × 2" and must NOT trip the explicit
    /// validation — the auto path applies its own floor at connect
    /// time (`effective_pool_size` in `src/storage/client.rs`), which
    /// also matches the `STORE_VALUES_MAX_INFLIGHT` bound so the
    /// invariant holds without operator action.
    #[test]
    fn config_from_env_valkey_pool_size_zero_means_auto_and_is_not_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("0")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("valkey_pool_size = 0 (auto) must be accepted by validation");
                assert_eq!(cfg.valkey_pool_size, 0);
            },
        );
    }

    /// `LIVE_EDIT_VALKEY_POOL_SIZE = 1` (the minimum non-zero value)
    /// is the most pathological misconfiguration the floor exists to
    /// catch. Pin the error so a future refactor cannot accidentally
    /// pass it through.
    #[test]
    fn config_from_env_valkey_pool_size_one_rejected_with_clear_remediation() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("valkey_pool_size = 1 must be rejected (well below floor)");
                assert!(
                    err.contains("Raise LIVE_EDIT_VALKEY_POOL_SIZE") || err.contains("auto"),
                    "error must include remediation guidance, got: {err}"
                );
            },
        );
    }

    // ---- t3.fgj.ya.99.e7: checkpoint env-var smoke tests ----
    //
    // Comprehensive coverage lives in sibling subtasks (`.eu` config
    // parsing, `.b8` cross-validation). These tests confirm the happy
    // path lands the documented defaults and each rejection path names
    // its env var(s).

    #[test]
    fn config_from_env_checkpoint_defaults_match_sdd() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert!(!cfg.checkpoint_enabled, "checkpoint_enabled default");
            assert_eq!(
                cfg.checkpoint_adapter,
                CheckpointAdapter::Disabled,
                "checkpoint_adapter default"
            );
            assert_eq!(
                cfg.checkpoint_interval_secs, 60,
                "checkpoint_interval_secs default"
            );
            assert_eq!(
                cfg.checkpoint_quiesce_secs, 30,
                "checkpoint_quiesce_secs default"
            );
            assert_eq!(cfg.checkpoint_tick_secs, 10, "checkpoint_tick_secs default");
            assert_eq!(
                cfg.checkpoint_retain_generations, 2,
                "checkpoint_retain_generations default"
            );
            assert_eq!(
                cfg.checkpoint_seq_safety_margin, 1000,
                "checkpoint_seq_safety_margin default"
            );
            assert_eq!(
                cfg.checkpoint_local_disk_path, None,
                "checkpoint_local_disk_path default"
            );
            assert_eq!(
                cfg.checkpoint_ext_mirror_workers, 2,
                "checkpoint_ext_mirror_workers default (Epic t1.fgj.zc / SDD SR-3.1)"
            );
            assert_eq!(
                cfg.checkpoint_ext_mirror_queue_capacity, 1000,
                "checkpoint_ext_mirror_queue_capacity default (Epic t1.fgj.zc / SDD SR-3.1)"
            );
        });
    }

    #[test]
    fn config_from_env_checkpoint_ext_mirror_workers_zero_rejected_with_explanation() {
        // Epic t1.fgj.zc / SDD SR-3.1: zero workers would spawn no drainer
        // tasks; the bounded channel would fill and silently drop every
        // enqueue. Pin both the env-var name (so the operator can find
        // the misconfiguration) and the "drop" / "silently" prose so a
        // future refactor doesn't collapse the message into a generic
        // "must be greater than 0".
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("ext_mirror_workers=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS"),
                    "error must name the offending var, got: {err}"
                );
                assert!(
                    err.contains("silently drop"),
                    "error must explain the drop-on-full failure mode, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_ext_mirror_queue_capacity_zero_rejected_with_explanation() {
        // Epic t1.fgj.zc / SDD SR-3.1: `tokio::sync::mpsc::channel(0)`
        // panics at runtime, so a zero capacity must fail closed at
        // startup. Pin both the env-var name and the "panics" prose so a
        // future refactor doesn't collapse the message into a generic
        // "must be greater than 0".
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("ext_mirror_queue_capacity=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY"),
                    "error must name the offending var, got: {err}"
                );
                assert!(
                    err.contains("panics"),
                    "error must explain the channel(0)-panic failure mode, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_ext_mirror_explicit_override_round_trips() {
        // Operator-set non-zero values must parse through unchanged.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS", Some("8")),
                (
                    "LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY",
                    Some("4096"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit overrides must succeed");
                assert_eq!(cfg.checkpoint_ext_mirror_workers, 8);
                assert_eq!(cfg.checkpoint_ext_mirror_queue_capacity, 4096);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tick_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TICK_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("tick=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TICK_SECS"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_quiesce_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_QUIESCE_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("quiesce=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_QUIESCE_SECS"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_retain_zero_rejected_with_explanation() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("retain=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS"),
                    "error must name the offending var, got: {err}"
                );
                assert!(
                    err.contains("prune"),
                    "error must explain that retain=0 prunes every just-written checkpoint, got: {err}"
                );
            },
        );
    }

    // ---- t3.fgj.mh.s3.xp: counters-sync + safety-margin Config plumbing ----
    //
    // Epic t1.fgj.mh / PRD AC #6 / SDD SR-5.1 / SR-5.2 / SR-5.3: the
    // counters sidecar exposes two operator knobs —
    // `LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS` (cadence of the
    // sidecar-write tick) and `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN`
    // (additive margin used at startup recovery before seeding
    // `le:counter:object_id`). Both must default cleanly and reject `0`
    // (the margin also rejects anything above
    // `CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX`)
    // with an error string that names the offending env var so an
    // operator can locate the misconfiguration without grepping the
    // codebase. The round-trip tests guard against a future refactor
    // collapsing the parse path and silently dropping the override.

    #[test]
    fn config_from_env_counters_sync_and_safety_margin_defaults_match_sdd() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                cfg.checkpoint_counters_sync_secs, 60,
                "checkpoint_counters_sync_secs default (Epic t1.fgj.mh / SDD SR-5.1)"
            );
            assert_eq!(
                cfg.checkpoint_counter_safety_margin, CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT,
                "checkpoint_counter_safety_margin default (Epic t1.fgj.mh / SDD SR-5.2)"
            );
            // The chart's `checkpoint.counterSafetyMargin` default and the CI
            // helm-lint render check pin the same number, so a change to the
            // constant is a deliberate edit here too.
            assert_eq!(CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT, 1_000_000);
        });
    }

    #[test]
    fn config_from_env_checkpoint_counters_sync_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("counters_sync=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_counter_safety_margin_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("counter_safety_margin=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    /// b.8y5: `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN=0` must be rejected
    /// at startup, matching its two siblings above.
    ///
    /// The margin is the additive guard band `apply_checkpoint` puts above
    /// the restore floor (`max(snap_seq, max_seq_observed)`). Zero means no
    /// guard band at all: the restored layer resumes at exactly the highest
    /// sequence number the restore can prove was issued, so the first
    /// post-restore commit re-issues a sequence number that subscribers who
    /// saw the (now-lost) in-RPO-window commits already hold attached to
    /// different content. That is a silent correctness knob, not a tuning
    /// one, so it fails closed rather than accepting `0`.
    #[test]
    fn config_from_env_checkpoint_seq_safety_margin_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("seq_safety_margin=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    /// b.8y5 boundary: `1` is the smallest accepted margin. Pinned so a
    /// future refactor that raises the floor (or reintroduces `0`) is a
    /// deliberate edit here rather than a silent config-surface change.
    #[test]
    fn config_from_env_checkpoint_seq_safety_margin_one_boundary_accepted() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN", Some("1")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("seq_safety_margin=1 (boundary) must be accepted");
                assert_eq!(cfg.checkpoint_seq_safety_margin, 1);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_counters_sync_explicit_override_round_trips() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS", Some("120")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit counters_sync=120 should parse");
                assert_eq!(cfg.checkpoint_counters_sync_secs, 120);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_counter_safety_margin_explicit_override_round_trips() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN", Some("5000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit counter_safety_margin=5000 should parse");
                assert_eq!(cfg.checkpoint_counter_safety_margin, 5000);
            },
        );
    }

    /// The largest accepted counter safety margin is
    /// `CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX` (10^9) itself: the bound is
    /// inclusive. Zero, the lower refusal, is
    /// `config_from_env_checkpoint_counter_safety_margin_zero_rejected`.
    #[test]
    fn config_from_env_checkpoint_counter_safety_margin_max_boundary_accepted() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN",
                    Some("1000000000"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("counter_safety_margin=10^9 (the maximum) must be accepted");
                assert_eq!(cfg.checkpoint_counter_safety_margin, 1_000_000_000);
                assert_eq!(
                    cfg.checkpoint_counter_safety_margin,
                    CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX
                );
            },
        );
    }

    /// One past the maximum is refused at startup, with an error that names
    /// the variable and the maximum so an operator can find and fix it. A
    /// margin that large would let pod starts use up the 2^53 - 1 object-id
    /// space (`CHECKPOINT_COUNTER_SAFETY_MARGIN_MAX`'s doc).
    #[test]
    fn config_from_env_checkpoint_counter_safety_margin_above_max_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN",
                    Some("1000000001"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("counter_safety_margin above the maximum must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN"),
                    "error must name the offending var, got: {err}"
                );
                assert!(
                    err.contains("1000000000"),
                    "error must state the maximum, got: {err}"
                );
            },
        );
    }

    // ---- b.yda: external-value ref_count reconciler cadence ---------------
    //
    // `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS` is the b.9qd knob's twin one
    // subsystem over, and it is deliberately validated the same way — with one
    // inversion that is the whole point of the block below. The bound is the
    // reconciler's DERIVED grace, `max(600 s, 10 x LIVE_EDIT_RPC_DEADLINE_MS)`,
    // a function of a variable this knob does not name; and `0` is a supported
    // off switch rather than a rejected tight loop, because a reconciler that
    // walks the whole blob keyspace is something an operator may legitimately
    // want silenced.
    //
    // Every test here goes through `temp_env::with_vars`, which serialises the
    // process-global environment across the suite. `Config::from_env` reads
    // `LIVE_EDIT_RPC_DEADLINE_MS` and `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS`
    // as a PAIR — the first sets the bound the second is judged against — so a
    // test that set one of them without the guard could observe the other
    // test's value and assert a bound that was never in force.

    /// Unset, at the shipped RPC deadline, resolves to the flat baseline.
    ///
    /// The grace at `LIVE_EDIT_RPC_DEADLINE_MS=30000` is
    /// `max(600, 10 x 30) = 600` s, comfortably under the baseline, so
    /// `max(EXT_RECONCILE_INTERVAL_DEFAULT_SECS, grace)` resolves to the
    /// baseline. Pinned as a literal rather than off the constant so a change to
    /// the constant is a deliberate edit here.
    #[test]
    fn config_from_env_ext_reconcile_interval_default_is_hourly() {
        temp_env::with_vars(
            [
                REQUIRED_VARS[0],
                REQUIRED_VARS[1],
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some("30000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(
                    cfg.ext_reconcile_interval_secs, 3600,
                    "ext_reconcile_interval_secs default (b.yda): the pass walks the whole object \
                     and blob keyspace and the residuals it drains accumulate over hours, so the \
                     default trades repair latency for SCAN traffic. At the shipped 30 s RPC \
                     deadline the derived grace is 600 s and the flat baseline wins"
                );
                assert_eq!(
                    crate::storage::ext_reconcile::ext_reconcile_grace(cfg.rpc_deadline_ms),
                    Duration::from_secs(600),
                    "sanity: the 600 s floor dominates 10 x the 30 s deadline"
                );
            },
        );
    }

    /// An **unset** interval must never refuse to boot, however large the
    /// derived grace gets — the b.9qd R2 argument, transplanted.
    ///
    /// The bound is a function of `LIVE_EDIT_RPC_DEADLINE_MS`, not of this
    /// variable, so a flat 3600 default would fail this field's own
    /// cross-validation on a deployment that raised the RPC deadline past 360 s
    /// — refusing to boot over a variable the operator never set, citing a
    /// variable they never touched.
    #[test]
    fn config_from_env_ext_reconcile_interval_default_derives_from_a_grace_above_the_baseline() {
        temp_env::with_vars(
            [
                REQUIRED_VARS[0],
                REQUIRED_VARS[1],
                // 400 s deadline → grace = 10 x 400 = 4000 s, above the 3600 s
                // baseline.
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some("400000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "an unset ext-reconcile interval must ALWAYS boot: the default is derived \
                     from the same grace it is cross-validated against, so it cannot fail that \
                     validation at any RPC deadline",
                );
                assert_eq!(
                    crate::storage::ext_reconcile::ext_reconcile_grace(cfg.rpc_deadline_ms),
                    Duration::from_secs(4000),
                    "sanity: 10 x the 400 s RPC deadline, above the 600 s floor"
                );
                assert_eq!(
                    cfg.ext_reconcile_interval_secs, 4000,
                    "the default is max(3600, grace), not the bare 3600 baseline — a flat \
                     baseline here would be BELOW the grace and would refuse to start on a \
                     variable the operator never set"
                );
            },
        );
    }

    /// An explicit value at or above the grace is taken verbatim.
    #[test]
    fn config_from_env_ext_reconcile_interval_explicit_override_round_trips() {
        temp_env::with_vars(
            [
                REQUIRED_VARS[0],
                REQUIRED_VARS[1],
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some("30000")),
                ("LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS", Some("7200")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit ext_reconcile_interval=7200 should parse");
                assert_eq!(
                    cfg.ext_reconcile_interval_secs, 7200,
                    "an explicit value clearing the grace is taken as written, not clamped to \
                     the derived default"
                );
            },
        );
    }

    /// Below the derived grace → rejected, with an error that names the
    /// variable, the bound, and the variable the bound resolves from.
    ///
    /// Deriving the *default* must not weaken the *explicit* validation: an
    /// operator who writes a number below the grace still gets the startup
    /// error. The error has to name `LIVE_EDIT_RPC_DEADLINE_MS` too, or the
    /// operator raises the interval by trial and error against a bound whose
    /// source is invisible.
    #[test]
    fn config_from_env_ext_reconcile_interval_below_the_derived_grace_is_rejected() {
        temp_env::with_vars(
            [
                REQUIRED_VARS[0],
                REQUIRED_VARS[1],
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some("30000")),
                ("LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS", Some("100")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("an interval below the derived grace must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS"),
                    "error must name the offending var, got: {err}"
                );
                assert!(
                    err.contains("600"),
                    "error must state the derived grace the operator has to clear — a pass \
                     shorter than the grace can never claim a candidate it registered, so it \
                     would repair nothing at all. Got: {err}"
                );
                assert!(
                    err.contains("LIVE_EDIT_RPC_DEADLINE_MS"),
                    "and must name the variable the bound actually resolves from, got: {err}"
                );
            },
        );
    }

    /// `0` is the documented off switch and is exempt from the grace bound.
    ///
    /// Unlike its b.9qd sibling, where zero is a tight-loop guard and is
    /// REJECTED. The asymmetry is deliberate — this pass is the only thing in
    /// the service that compares `ext:meta` `ref_count` against live references,
    /// and an operator with a very large blob keyspace may want it silenced —
    /// so the two knobs' zero handling must not be assumed to match.
    #[test]
    fn config_from_env_ext_reconcile_interval_zero_is_accepted_as_the_off_switch() {
        temp_env::with_vars(
            [
                REQUIRED_VARS[0],
                REQUIRED_VARS[1],
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some("30000")),
                ("LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS", Some("0")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "0 is the documented off switch (WARNed at startup, not rejected) — it is \
                     exempt from the grace cross-validation, which would otherwise refuse every \
                     deliberate disable",
                );
                assert_eq!(cfg.ext_reconcile_interval_secs, 0);
            },
        );
    }

    /// The reconciler knob must not disturb `LIVE_EDIT_MAINTENANCE_POOL_SIZE`.
    ///
    /// The pass runs on the maintenance pool, and the obvious way to make room
    /// for a third consumer of that pool would have been to raise its default.
    /// That was deliberately not done — the pass is hourly and leader-gated, so
    /// it contends with the other leader-gated maintenance-pool tasks (the two
    /// sweeps, and since the backup repair, its walk) only in the ticks where
    /// they land together —
    /// and a silent default bump would change the connection footprint of every
    /// deployment.
    #[test]
    fn config_from_env_ext_reconcile_leaves_the_maintenance_pool_default_alone() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                cfg.maintenance_pool_size, 1,
                "the b.yda reconciler shares the existing maintenance pool; its default is not \
                 the reconciler's to raise"
            );
        });
    }

    // ── Snapshot lane, read permits and atomic-read limits
    //    (Spec SDD SR-1.9, SR-7.1, SR-10.1, SR-11) ───────────────────────────
    //
    // Five knobs, one shape: a positive integer with a ruled default, zero
    // rejected, non-numeric rejected. The cases share one env fixture and one
    // assertion helper per direction, as the refresh-lead cluster below does,
    // rather than a dozen near-identical `temp_env::with_vars` blocks.

    /// Reads the `Config` field a snapshot knob lands in.
    type SnapshotKnobField = fn(&Config) -> u64;

    /// The five snapshot knobs, each with the `Config` field it lands in.
    const SNAPSHOT_KNOBS: [(&str, SnapshotKnobField); 5] = [
        ("LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY", |c| {
            u64::from(c.catchup_snapshot_read_concurrency)
        }),
        ("LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY", |c| {
            u64::from(c.catchup_fenced_read_concurrency)
        }),
        ("LIVE_EDIT_SNAPSHOT_POOL_SIZE", |c| {
            u64::from(c.snapshot_pool_size)
        }),
        ("LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS", |c| {
            c.snapshot_atomic_read_max_items
        }),
        ("LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES", |c| {
            c.snapshot_atomic_read_max_bytes
        }),
    ];

    /// [`REQUIRED_VARS`] plus `var=value`.
    fn snapshot_knob_env<'a>(var: &'a str, value: &'a str) -> Vec<(&'a str, Option<&'a str>)> {
        let mut env = REQUIRED_VARS.to_vec();
        env.push((var, Some(value)));
        env
    }

    /// Assert `Config::from_env` rejects `var=value`, naming the variable, and
    /// return the message for any case-specific check.
    fn assert_snapshot_knob_rejected(var: &str, value: &str, why: &str) -> String {
        temp_env::with_vars(snapshot_knob_env(var, value), || {
            let err = Config::from_env_with(PodMemorySource::Reference).expect_err(&format!(
                "{var}={value:?} must be rejected at startup: {why}"
            ));
            assert!(
                err.contains(var),
                "the reject for {var}={value:?} must name the variable, got: {err}"
            );
            err
        })
    }

    #[test]
    fn config_from_env_snapshot_knobs_default_to_the_ruled_values() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                cfg.catchup_snapshot_read_concurrency, 1,
                "LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY defaults to P_a = 1 (SR-7.1)"
            );
            assert_eq!(
                cfg.catchup_fenced_read_concurrency, 1,
                "LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY defaults to P_f = 1 (SR-7.1)"
            );
            assert_eq!(
                cfg.snapshot_pool_size, 6,
                "LIVE_EDIT_SNAPSHOT_POOL_SIZE defaults to P_a + P_f + 4 = 1 + 1 + 4 (SR-10.1)"
            );
            assert_eq!(
                cfg.snapshot_atomic_read_max_items, 37_554,
                "LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS defaults to the AMR operating ceiling"
            );
            assert_eq!(
                cfg.snapshot_atomic_read_max_bytes, 604_619,
                "LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES defaults to the item ceiling on the \
                 real-id ray (37,554 × 16.1)"
            );
        });
    }

    /// The lane's default at the shipped `P_a` and `P_f` defaults
    /// (`LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` and
    /// `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`). Pinned at the values the
    /// ceilings were solved at, so a change to either default moves the lane
    /// size deliberately rather than as a side effect.
    #[test]
    fn snapshot_pool_size_default_counts_one_atomic_and_one_fenced_read_plus_four() {
        assert_eq!(CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT, 1, "P_a");
        assert_eq!(CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT, 1, "P_f");
        assert_eq!(
            SNAPSHOT_POOL_NON_CATCHUP_CONNECTIONS, 4,
            "structural, checkpoint and ext-GC readers"
        );
        assert_eq!(SNAPSHOT_POOL_SIZE_DEFAULT, 6);
    }

    /// The lane's default follows the configured `P_a` (SR-10.1): a replica
    /// that runs more atomic reads at once gets one more connection for each,
    /// and an explicit lane size still wins.
    #[test]
    fn config_from_env_snapshot_pool_size_default_follows_the_configured_read_concurrency() {
        temp_env::with_vars(
            snapshot_knob_env("LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY", "3"),
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("an explicit read concurrency parses");
                assert_eq!(cfg.catchup_snapshot_read_concurrency, 3);
                assert_eq!(
                    cfg.snapshot_pool_size, 8,
                    "LIVE_EDIT_SNAPSHOT_POOL_SIZE defaults to P_a + P_f + 4 = 3 + 1 + 4"
                );
            },
        );
        let mut env = snapshot_knob_env("LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY", "3");
        env.push(("LIVE_EDIT_SNAPSHOT_POOL_SIZE", Some("2")));
        temp_env::with_vars(env, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference).expect("both knobs parse");
            assert_eq!(
                cfg.snapshot_pool_size, 2,
                "an explicit LIVE_EDIT_SNAPSHOT_POOL_SIZE is taken as set"
            );
        });
        assert_eq!(
            snapshot_pool_size_default(u32::MAX, CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT),
            u32::MAX,
            "the derived default saturates instead of overflowing"
        );
    }

    /// The lane's default follows the configured `P_f` too (SR-10.1): one
    /// connection per fenced read a replica runs at once, added to `P_a`'s.
    #[test]
    fn config_from_env_snapshot_pool_size_default_follows_the_configured_fenced_read_concurrency() {
        temp_env::with_vars(
            snapshot_knob_env("LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY", "3"),
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("an explicit fenced read concurrency parses");
                assert_eq!(cfg.catchup_fenced_read_concurrency, 3);
                assert_eq!(
                    cfg.snapshot_pool_size, 8,
                    "LIVE_EDIT_SNAPSHOT_POOL_SIZE defaults to P_a + P_f + 4 = 1 + 3 + 4"
                );
            },
        );
        let mut env = snapshot_knob_env("LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY", "2");
        env.push(("LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY", Some("3")));
        temp_env::with_vars(env, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("both read concurrencies parse");
            assert_eq!(
                cfg.snapshot_pool_size, 9,
                "LIVE_EDIT_SNAPSHOT_POOL_SIZE defaults to P_a + P_f + 4 = 3 + 2 + 4"
            );
        });
        assert_eq!(
            snapshot_pool_size_default(CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT, u32::MAX),
            u32::MAX,
            "the derived default saturates on P_f as well"
        );
    }

    // ── The chart's values for P_f and the pod memory limit (Spec SDD SR-7.1,
    //    SR-7.2) ──────────────────────────────────────────────────────────────
    //
    // The binary's default P_f must equal the chart's: the chart renders the
    // variable on every install, so a chart value that drifted from the
    // binary's would silently override it (the `supply-chain:helm-lint` job
    // pins the rendered output; these pin the binary side). The reference pod
    // memory limit, what a pod with no readable limit is sized as, is the
    // chart's default limit. `helm/` is part of the published source tree, so
    // these never skip.

    const CHART_VALUES: &str = "helm/live-edit-service/values.yaml";

    /// The scalar at `path` (one key per nesting level) in the chart's
    /// `values.yaml`. A minimal reader for plain two-space block mappings,
    /// which is all the paths read here need: a key is matched only at its own
    /// level's indentation, and a block ends at the first line indented less.
    fn chart_value(path: &[&str]) -> String {
        values_file_value(CHART_VALUES, path)
            .unwrap_or_else(|| panic!("{CHART_VALUES} has no {path:?}"))
    }

    /// The scalar at `path` in the values file `file` (relative to the
    /// crate root), by [`chart_value`]'s reader; `None` when the file sets
    /// no such key (an override file sets only what it overrides).
    fn values_file_value(file: &str, path: &[&str]) -> Option<String> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("the chart values {file}: {e}"));
        let mut depth = 0;
        for line in text.lines() {
            let body = line.trim_start();
            if body.is_empty() || body.starts_with('#') {
                continue;
            }
            let indent = line.len() - body.len();
            if indent < 2 * depth {
                // The `path[..depth]` block ended without the key.
                return None;
            }
            if indent != 2 * depth {
                continue;
            }
            let Some((key, value)) = body.split_once(':') else {
                continue;
            };
            if key != path[depth] {
                continue;
            }
            depth += 1;
            if depth == path.len() {
                return Some(value.split(" #").next().unwrap_or("").trim().to_string());
            }
        }
        None
    }

    fn chart_u64(path: &[&str]) -> u64 {
        let raw = chart_value(path);
        raw.parse()
            .unwrap_or_else(|e| panic!("{CHART_VALUES} {path:?} = {raw:?} is not an integer: {e}"))
    }

    /// The chart's pod memory limit (`resources.limits.memory`) in MiB.
    fn chart_pod_memory_limit_mib() -> u64 {
        memory_quantity_mib(&chart_value(&["resources", "limits", "memory"]))
    }

    /// A Kubernetes memory quantity in `Gi` or `Mi`, in MiB.
    fn memory_quantity_mib(raw: &str) -> u64 {
        let raw = raw.trim_matches('"');
        let (number, mib_per_unit) = if let Some(n) = raw.strip_suffix("Gi") {
            (n, 1024)
        } else if let Some(n) = raw.strip_suffix("Mi") {
            (n, 1)
        } else {
            panic!("resources.limits.memory = {raw:?} is in neither Gi nor Mi");
        };
        number
            .parse::<u64>()
            .unwrap_or_else(|e| panic!("resources.limits.memory = {raw:?}: {e}"))
            * mib_per_unit
    }

    #[test]
    fn catchup_fenced_read_default_and_reference_pod_memory_limit_equal_the_chart_values() {
        assert_eq!(
            REFERENCE_POD_MEMORY_LIMIT_MIB,
            chart_pod_memory_limit_mib(),
            "the reference pod memory limit must be the chart's resources.limits.memory"
        );
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                u64::from(cfg.catchup_fenced_read_concurrency),
                chart_u64(&["valkey", "catchupFencedReadConcurrency"]),
                "the binary's default P_f must equal the chart's \
                 valkey.catchupFencedReadConcurrency"
            );
        });
    }

    // ── T, the memory factors and the window cap against the Task 10 record
    //    (Spec SDD SR-7.2, SR-11.3, SR-11.6) ─────────────────────────────────
    //
    // `T` is derived from the chart's pod memory limit (over the record's
    // funded terms on real-id bytes, the structural cache's two terms outside
    // its budget, the uploads and the Valkey connections' buffers; the chart
    // limit's list is recorded in its own record), the two memory factors are
    // recorded figures (measured as their combined
    // factor and kept), and `FANOUT_WINDOW_MAX` is derived from the measured
    // per-command window cost; the record's inputs are committed in the
    // fenced-snapshot-reads report's Task 10 record,
    // `raw/task10-results/results.json`. Read at run time and skipped in the
    // published tree, like the SR-11.8 records below.

    const TASK10_RESULTS_JSON: &str =
        "docs-internal/validation/fenced-snapshot-reads-20261002/raw/task10-results/results.json";

    /// The Task 10 record's funded terms this build replaces with their
    /// real-id values from the real-id byte-ceiling record (`memory_mib`):
    /// its load terms (catch-up slots at the short-id 13.4 MiB) and its two
    /// measured reply-byte terms, which scale with the reply bytes.
    const RECORD_LOAD_TERMS: &str = "SDD §16 terms";
    const RECORD_WINDOW_TRANSIENT: &str = "P_f x X";
    const RECORD_REFERENCE_READ: &str = "reference read";

    /// `value` rounded to one decimal place, the precision the records and
    /// the SDD state MiB figures at.
    fn mib_1dp(value: f64) -> f64 {
        (value * 10.0).round() / 10.0
    }

    /// SR-7.2's rule for `T`, as the Task 10 record applies it
    /// (`t10_results.derive_t`): `T` = floor(f × (limit − floor)), f the
    /// largest 0.05 step for which `fixed_mib` (every funded term but
    /// `T × (1 + P_f)`, the floor included) + `T × (1 + P_f)` fits `limit_mib`.
    /// Returns f, `T` and the funded total at that `T`.
    fn t_by_the_recorded_rule(
        limit_mib: f64,
        floor_mib: f64,
        p_f: f64,
        fixed_mib: f64,
    ) -> (f64, u64, f64) {
        let head = limit_mib - floor_mib;
        let f_max = (limit_mib - fixed_mib) / (1.0 + p_f) / head;
        let steps = (f_max / 0.05 + 1e-9).floor();
        assert!(
            steps >= 1.0,
            "no 0.05 step of the headroom fits {limit_mib} MiB beside {fixed_mib} MiB of terms"
        );
        let t = (steps * head / 20.0).floor() as u64;
        (steps / 20.0, t, fixed_mib + (1.0 + p_f) * t as f64)
    }

    /// The record of the chart limit's term list:
    /// `T`, the combined line and the subscription allowance at 5,120 MiB, with
    /// the uploads and the Valkey connections' buffers counted, the
    /// structural cache at its resident budget (the configured budget ×
    /// (1 + φ)) and its cold-start reads carrying the plan indexes' build. The
    /// window script's record carries the list (copied from
    /// `docs-internal/validation/structural-edit-inplace-b-ug2-20261010/derivation.json`,
    /// which supersedes `fanout-executor-b26e-s2-20261009/derivation.json`, the
    /// source of its window, connection and upload terms, kept unchanged) with
    /// the structural WATCH read's `X` on the window-script build added to the
    /// window term's inputs (`x_watch_script`).
    const CHART_LIMIT_DERIVATION_JSON: &str =
        "docs-internal/validation/watch-window-script-b-gvf-20261010/derivation.json";

    /// The measurements the chart limit's record derives its cold-start
    /// build ratio `b` from: `shapes[].b_build_peak_over_state`, the plan
    /// indexes' build peak (`PlanIndexes::build_peak_bytes_for`) over the
    /// state's weight per weighed layer shape.
    const CHART_LIMIT_DERIVE_INPUTS_JSON: &str =
        "docs-internal/validation/structural-edit-inplace-b-ug2-20261010/raw/derive.json";

    /// The Task 10 record's entry the rule is checked against: 4,096 MiB,
    /// whose funded terms give its `T_mib` 985.
    const TASK10_RULE_CHECK_LIMIT_MIB: u64 = 4_096;

    /// The HTTP/2 stream window each queued upload waiter's unread content is
    /// bounded by: hyper's server default, 1 MiB per stream, which
    /// `production_server_builder` leaves as it is (SDD §5.4).
    const UPLOAD_WAITER_WINDOW_MIB: f64 = 1.0;

    /// The uploads' terms of the chart limit's list, derived from the binary's
    /// configuration at the chart's values ([`chart_uploads`]).
    struct ChartUploads {
        /// The upload memory budget `B`, in MiB.
        budget_mib: u64,
        /// The queued upload waiters' cap.
        waiters: usize,
        /// The charge of one `LIVE_EDIT_MAX_LAYER_SIZE` upload of one
        /// external value in the overflow slot, in MiB.
        overflow_slot_mib: f64,
        /// `B` + the waiters at [`UPLOAD_WAITER_WINDOW_MIB`] each + the
        /// overflow slot, in MiB.
        total_mib: f64,
        /// The connection pools at the chart's pool sizes, for `C_conn`.
        pools: crate::storage::connection_buffers::PoolSizes,
        /// `T` as the pod memory plan derives it at the reference limit for
        /// the chart's pools.
        plan_catchup_mib: u64,
    }

    /// The uploads' terms at the chart's values: `B` is the pod memory plan's
    /// at the reference limit, the waiters' cap and the largest upload are the
    /// binary's defaults and equal the chart's, and the pools are the chart's
    /// sizes.
    fn chart_uploads() -> ChartUploads {
        const MIB: f64 = 1_048_576.0;
        let pools = |name: &str| {
            let raw = chart_value(&["valkey", name]);
            (raw != "null").then_some(raw)
        };
        let pool_env: Vec<(String, Option<String>)> = [
            ("LIVE_EDIT_VALKEY_POOL_SIZE", "poolSize"),
            ("LIVE_EDIT_MAINTENANCE_POOL_SIZE", "maintenancePoolSize"),
            ("LIVE_EDIT_SNAPSHOT_POOL_SIZE", "snapshotPoolSize"),
            ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", "subscriberPoolSize"),
        ]
        .into_iter()
        .map(|(var, name)| (var.to_owned(), pools(name)))
        .chain(
            REQUIRED_VARS
                .iter()
                .map(|(var, value)| ((*var).to_owned(), value.map(str::to_owned))),
        )
        .collect();
        let (budget_mib, waiters, max_layer_size, chart_pools, plan_catchup_mib) =
            temp_env::with_vars(pool_env, || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("the chart's pool sizes parse");
                (
                    cfg.upload_memory_budget_mib,
                    cfg.upload_admission_max_waiters,
                    cfg.max_layer_size as u64,
                    crate::storage::connection_buffers::PoolSizes::from_config(&cfg),
                    cfg.catchup_build_memory_budget_mib,
                )
            });
        assert_eq!(
            budget_mib,
            crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            "the pod memory plan's upload budget at the reference limit is 384 MiB"
        );
        assert_eq!(
            waiters as u64,
            chart_u64(&["layer", "uploadAdmissionMaxWaiters"])
        );
        assert_eq!(max_layer_size, chart_u64(&["layer", "maxSize"]));
        let overflow_slot_mib =
            crate::grpc::upload_admission::parse_charge(max_layer_size, 0, max_layer_size) as f64
                / MIB;
        ChartUploads {
            budget_mib,
            waiters,
            overflow_slot_mib,
            total_mib: budget_mib as f64
                + waiters as f64 * UPLOAD_WAITER_WINDOW_MIB
                + overflow_slot_mib,
            pools: chart_pools,
            plan_catchup_mib,
        }
    }

    /// The chart limit's record ([`CHART_LIMIT_DERIVATION_JSON`]) is the term
    /// list the configuration derives at the chart's pod memory limit, by the
    /// Task 10 record's own rule (its 0.05 step), and its `T` is the one the
    /// pod memory plan derives there and the reference constant
    /// [`CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT`] carries. The plan itself,
    /// term by term against the same record, is
    /// [`pod_memory_plan_at_the_chart_limit_reproduces_its_records_line_term_by_term`].
    ///
    /// 1. The rule is checked against the Task 10 record: over its terms for
    ///    4,096 MiB alone it reproduces that entry's `T_mib` (985, f = 0.25,
    ///    4,032.9 MiB).
    /// 2. The real-id terms: `P_f × X` and the reference read are the Task 10
    ///    record's measured terms scaled by the byte ceiling's
    ///    real-id/short-id ratio, and every term that grows with the byte
    ///    ceiling (the catch-up slots, the load terms, the structural cache's
    ///    cold-start reads and sampled audit) is derived here from the shipped
    ///    configuration, each checked against its figure in the real-id
    ///    record (`memory_mib`).
    /// 3. The chart limit's list, read from its record
    ///    ([`CHART_LIMIT_DERIVATION_JSON`], whose `limit_mib` must be the
    ///    chart's): the floor, those real-id terms, the window term (the
    ///    largest `X` the record carries: the catch-up's model, the measured
    ///    catch-up, structural cold start and WATCH read of the build with
    ///    the fan-out arena, the WATCH read of the window-script build, and
    ///    the watch-read pool build's WATCH maximum, kept as the conservative
    ///    bound; the window-script read's shipped-build bound is the arena
    ///    build's plain WATCH read, within the rule's term, and its own
    ///    measurement no larger), the structural cache at its
    ///    resident budget (`resident_budget_bytes`: the configured budget ×
    ///    (1 + φ), for its entries' plan indexes), its two plain cold-start
    ///    reads — each the reply, the state decoded from it (`r`, the
    ///    real-id record's entry-to-read ratio) and the plan indexes' build at
    ///    its peak (`b` × the state, `b` = ⌈1.25 × the largest build-peak-to-
    ///    state ratio measured⌉ in 0.01 steps, re-derived here from the
    ///    record's inputs, [`CHART_LIMIT_DERIVE_INPUTS_JSON`]) — and its
    ///    sampled audit, the uploads — the upload memory budget, its queued
    ///    waiters' unread content and one `LIVE_EDIT_MAX_LAYER_SIZE`
    ///    single-value upload's charge in the overflow slot, each derived from
    ///    configuration and code — the Valkey connections' at-rest buffers on
    ///    a 10-primary cluster at the chart's pools, and the 20 % margin. It
    ///    gives 744 (f = 0.15, 5,102.7 MiB; f = 0.20 would need 5,600.7). The
    ///    record's connection-buffer figures and total are the shipped
    ///    build's, with the `valkey-tls` feature on, and are compared only in
    ///    that shape; without it `C_conn` is smaller, and the list is derived
    ///    and must fit at that shape's own `C_conn`.
    ///
    /// Moving the chart's limit without a new record, re-recording a term
    /// without moving `T`, or changing the byte ceiling, φ, a cache, upload or
    /// pool setting fails here (a pool setting, and a re-recorded `C_conn` or
    /// total, in the shipped `valkey-tls` shape only; without it a pool
    /// setting fails only where it moves `f`'s 0.05 step).
    // Guard test: pinned by name in scripts/guard-tests.sh and in the
    // validation-records guard (docs-internal/engineering-best-practices.md
    // § Validation records); rename or remove it there too.
    #[test]
    fn chart_limit_records_term_list_derives_from_configuration_by_its_0_05_rule() {
        let Some(results) = sr_11_8_record(TASK10_RESULTS_JSON) else {
            return;
        };
        let limit_mib = TASK10_RULE_CHECK_LIMIT_MIB;
        let rec = format!("/memory/by_pod_limit/{limit_mib}");
        let number = |pointer: &str| {
            recorded(&results, pointer)
                .as_f64()
                .unwrap_or_else(|| panic!("{pointer} is recorded as a number"))
        };
        let (limit, floor) = (
            number(&format!("{rec}/limit_mib")),
            number(&format!("{rec}/floor_mib")),
        );
        assert_eq!(
            limit, limit_mib as f64,
            "the record's entry is for the rule check's limit"
        );
        let p_f = f64::from(CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT);
        let budget_bound = "T x (1 + P_f)";
        let terms = recorded(&results, &format!("{rec}/funded_terms_mib"));
        let terms = terms
            .as_object()
            .unwrap_or_else(|| panic!("{rec}/funded_terms_mib is recorded as an object"));
        let term = |name: &str| {
            terms[name]
                .as_f64()
                .unwrap_or_else(|| panic!("funded term {name:?} is recorded as a number"))
        };
        let recorded_fixed: f64 = terms
            .keys()
            .filter(|name| *name != budget_bound)
            .map(|name| term(name))
            .sum();

        let record_t = recorded_u64(&results, &format!("{rec}/T_mib"));
        let (_, rule_t, _) = t_by_the_recorded_rule(limit, floor, p_f, recorded_fixed);
        assert_eq!(
            rule_t, record_t,
            "over the record's own funded terms the rule reproduces its T_mib"
        );
        assert_eq!(
            term(budget_bound),
            (1.0 + p_f) * record_t as f64,
            "the record's budget bound is its T at this build's P_f"
        );

        let Some(real_id) = sr_11_8_record(REAL_ID_DERIVATION_JSON) else {
            return;
        };
        let real = |name: &str| {
            recorded(&real_id, &format!("/memory_mib/{name}"))
                .as_f64()
                .unwrap_or_else(|| panic!("memory_mib.{name} is recorded as a number"))
        };
        let scale = recorded_u64(&real_id, "/byte_ceiling/bytes") as f64
            / recorded_u64(&real_id, "/byte_ceiling/previous") as f64;
        let window_transient = term(RECORD_WINDOW_TRANSIENT) * scale;
        let reference_read = term(RECORD_REFERENCE_READ) * scale;
        assert_eq!(
            mib_1dp(window_transient),
            real("p_f_times_x"),
            "memory_mib.p_f_times_x is the record's P_f x X on real-id reply bytes"
        );
        assert_eq!(
            mib_1dp(reference_read),
            real("checkpoint_reference_layer_4x"),
            "memory_mib.checkpoint_reference_layer_4x is the record's reference read on \
             real-id reply bytes"
        );
        // The terms that grow with the byte ceiling, from the shipped
        // configuration, each checked against its recorded figure: the
        // catch-up slots (42 × the ceiling each), the cache's plain
        // cold-start reads (each reply plus the state decoded from it) and
        // the sampled audit (two entries at the per-entry cap plus one
        // reply).
        const MIB: f64 = 1_048_576.0;
        let (ceiling, slots, cache_budget) = temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            (
                cfg.snapshot_atomic_read_max_bytes as f64,
                cfg.subscriber_max_concurrent_catchups as f64,
                cfg.structural_cache_max_mib * 1_048_576,
            )
        });
        assert_eq!(
            ceiling,
            recorded_u64(&real_id, "/byte_ceiling/bytes") as f64,
            "the terms are derived at the shipped byte ceiling"
        );
        let factor = (crate::storage::memory_budget::SNAPSHOT_MEMORY_FACTOR
            + crate::storage::memory_budget::BUILD_MEMORY_FACTOR) as f64;
        assert_eq!(factor, real("snapshot_plus_build_factor"));
        let slot = factor * ceiling / MIB;
        assert_eq!(
            (slot * 100.0).round() / 100.0,
            real("catchup_slot_at_ceiling")
        );
        assert_eq!(mib_1dp(slots * slot), real("catchup_slots_8"));
        let load_terms = real("subscriptions_4096_at_49_8_kib")
            + slots * slot
            + real("readers_replies")
            + real("lagging_layer_hard_cap")
            + real("registration_to_first_poll");
        assert_eq!(
            mib_1dp(load_terms),
            real("load_terms"),
            "memory_mib.load_terms is the sum of its parts at the shipped byte ceiling"
        );
        let cold_start_reads = crate::storage::structural_cache::STRUCTURAL_COLD_START_READ_PERMITS
            as f64
            * ceiling
            * (1.0 + real("cold_start_entry_to_read_ratio"))
            / MIB;
        assert_eq!(mib_1dp(cold_start_reads), real("cold_start_plain_reads"));
        let entry_cap = crate::storage::structural_cache::StructuralCache::new(cache_budget)
            .max_entry_bytes() as f64;
        let audit = (2.0 * entry_cap + ceiling) / MIB;
        assert_eq!(mib_1dp(audit), real("sampled_audit"));
        let merged_fixed = recorded_fixed
            - term(RECORD_LOAD_TERMS)
            - term(RECORD_WINDOW_TRANSIENT)
            - term(RECORD_REFERENCE_READ)
            + load_terms
            + window_transient
            + reference_read
            + cold_start_reads
            + audit;
        // The real-id record's own list, at its 4,096 MiB, is unchanged: a
        // historical record, still the source of the real-id terms.
        let (_, t_4096, total_4096) = t_by_the_recorded_rule(limit, floor, p_f, merged_fixed);
        assert_eq!(
            mib_1dp(total_4096),
            real("t_table_total"),
            "memory_mib.t_table_total is the merged term list's total at T = {t_4096}"
        );
        assert_eq!(
            (1.0 + p_f) * t_4096 as f64,
            real("t_times_1_plus_pf"),
            "memory_mib.t_times_1_plus_pf is the merged list's T x (1 + P_f) at 4,096 MiB"
        );

        // The chart's limit: its record's list, every term derived here.
        let Some(chart) = sr_11_8_record(CHART_LIMIT_DERIVATION_JSON) else {
            return;
        };
        let at = |pointer: &str| {
            recorded(&chart, pointer)
                .as_f64()
                .unwrap_or_else(|| panic!("{pointer} is recorded as a number"))
        };
        let chart_limit_mib = chart_pod_memory_limit_mib();
        assert_eq!(
            recorded_u64(&chart, "/limit_mib"),
            chart_limit_mib,
            "{CHART_LIMIT_DERIVATION_JSON} is the term list of the chart's pod memory limit; \
             a new limit needs its own record"
        );
        let chart_limit = chart_limit_mib as f64;
        assert_eq!(at("/floor"), floor, "the same steady-state floor");
        assert_eq!(at("/p_f"), p_f);
        assert_eq!(at("/load_terms/value"), mib_1dp(load_terms));
        assert_eq!(at("/x_catchup/value"), mib_1dp(window_transient));
        assert_eq!(at("/reference_read"), mib_1dp(reference_read));
        // The structural cache at its resident budget: the configured budget
        // × (1 + φ), which its weigher bounds its entries' state and plan
        // indexes by.
        use crate::storage::structural_cache::{resident_budget_bytes, PLAN_INDEX_BUDGET_CENTI};
        let resident_bytes = resident_budget_bytes(cache_budget)
            .expect("the default budget's resident budget fits in a u64");
        assert_eq!(
            recorded_u64(&chart, "/structural_cache_budget_mib"),
            cache_budget / 1_048_576,
            "the record's configured cache budget is the shipped default"
        );
        assert_eq!(at("/phi"), PLAN_INDEX_BUDGET_CENTI as f64 / 100.0);
        assert_eq!(
            recorded_u64(&chart, "/structural_cache_bytes"),
            resident_bytes,
            "structural_cache_bytes is resident_budget_bytes of the default budget"
        );
        let resident_cache = resident_bytes as f64 / MIB;
        assert_eq!(
            at("/structural_cache"),
            (resident_cache * 100.0).round() / 100.0,
            "structural_cache is the resident budget in MiB at two decimals"
        );
        // The plain cold-start reads: each the reply, the state decoded from
        // it and the plan indexes' build at its peak, `b` × the state, `b`
        // re-derived from the record's inputs by its rule.
        let r = real("cold_start_entry_to_read_ratio");
        assert_eq!(at("/cold_start_entry_to_read_ratio"), r);
        let b = at("/cold_start_index_build_peak_ratio");
        // The refused plain cold start's hold allowance
        // (`structural_cache_cold_start`, seam A) is built from the same `r`
        // and `b`, so the term and the guard cannot drift apart.
        use crate::storage::structural_cache_cold_start::{
            plain_cold_start_hold_allowance_bytes, COLD_START_BUILD_PEAK_CENTI,
            COLD_START_ENTRY_TO_READ_CENTI,
        };
        assert_eq!(
            COLD_START_ENTRY_TO_READ_CENTI as f64 / 100.0,
            r,
            "COLD_START_ENTRY_TO_READ_CENTI is the record's r"
        );
        assert_eq!(
            COLD_START_BUILD_PEAK_CENTI as f64 / 100.0,
            b,
            "COLD_START_BUILD_PEAK_CENTI is the record's b"
        );
        if let Some(inputs) = sr_11_8_record(CHART_LIMIT_DERIVE_INPUTS_JSON) {
            let largest = recorded(&inputs, "/shapes")
                .as_array()
                .expect("shapes is recorded as an array")
                .iter()
                .map(|shape| {
                    shape["b_build_peak_over_state"]
                        .as_f64()
                        .expect("each shape records b_build_peak_over_state")
                })
                .fold(0.0, f64::max);
            assert!(largest > 0.0, "the inputs record at least one shape");
            assert_eq!(
                b,
                (largest * 1.25 * 100.0 - 1e-9).ceil() / 100.0,
                "b is ceil(1.25 x the largest build-peak-to-state ratio, {largest}) in 0.01 \
                 steps"
            );
        }
        let chart_cold_start_reads =
            crate::storage::structural_cache::STRUCTURAL_COLD_START_READ_PERMITS as f64
                * ceiling
                * (1.0 + r * (1.0 + b))
                / MIB;
        assert_eq!(
            at("/cold_start_plain_reads"),
            mib_1dp(chart_cold_start_reads),
            "cold_start_plain_reads is 2 x the ceiling x (1 + r x (1 + b))"
        );
        // Each read's term less its reply is the allowance, at the shipped
        // ceiling 5,535,891 B.
        let allowance = plain_cold_start_hold_allowance_bytes(ceiling as u64);
        assert_eq!(allowance, 5_535_891, "the allowance at the shipped ceiling");
        assert_eq!(
            allowance as f64,
            (ceiling * r * (1.0 + b)).floor(),
            "the allowance is the ceiling x r x (1 + b), floored"
        );
        assert_eq!(
            at("/cold_start_plain_reads"),
            mib_1dp(
                crate::storage::structural_cache::STRUCTURAL_COLD_START_READ_PERMITS as f64
                    * (ceiling + allowance as f64)
                    / MIB
            ),
            "cold_start_plain_reads is 2 x (the ceiling + the allowance)"
        );
        assert_eq!(at("/sampled_audit"), mib_1dp(audit));
        // A measured `X` of the record, in MiB: from its bytes when it
        // records them (exact), else its one-decimal value; `None` while it
        // is null (not measured).
        let measured_x = |pointer: &str| {
            let value = recorded(&chart, pointer);
            if value.is_null() {
                return None;
            }
            let x = match value.pointer("/bytes").and_then(serde_json::Value::as_i64) {
                Some(bytes) => {
                    let mib = bytes as f64 / MIB;
                    assert_eq!(
                        value.pointer("/value").and_then(serde_json::Value::as_f64),
                        Some(mib_1dp(mib)),
                        "{pointer}.value is its bytes in MiB at one decimal"
                    );
                    mib
                }
                None => value
                    .pointer("/value")
                    .and_then(serde_json::Value::as_f64)
                    .or_else(|| value.as_f64())
                    .unwrap_or_else(|| panic!("{pointer} is recorded as a number")),
            };
            Some(x)
        };
        // The window term by the rule: the largest `X` recorded — the
        // catch-up's model, the measured catch-up, structural cold start and
        // WATCH read of the build with the fan-out arena, and the WATCH read
        // of the window-script build.
        let window_term_by_rule = [
            "/x_catchup_postfix",
            "/x_sc_postfix",
            "/x_watch_postfix",
            "/x_watch_script",
        ]
        .into_iter()
        .filter_map(measured_x)
        .fold(window_transient, f64::max);
        assert_eq!(at("/window_term_by_rule"), mib_1dp(window_term_by_rule));
        // The window term in the list: the rule's, or the watch-read pool
        // build's larger-sample WATCH maximum where that is larger, kept as
        // the conservative bound (the record's PD-X2).
        let window_term = measured_x("/x_watch_pool_build")
            .into_iter()
            .fold(window_term_by_rule, f64::max);
        assert_eq!(at("/window_term"), mib_1dp(window_term));
        // The window-script build's WATCH read on the shipped build (b.gvf):
        // its `X` on any window-model coefficients is bounded by the plain
        // WATCH read's, measured on the fan-out arena build
        // (`x_watch_script.shipped_build_bound`, the record's argument), so
        // the bound is that measurement, and it sits under the rule's term;
        // the script read's own measurement, on the wired coefficients, is
        // no larger.
        let bytes_at = |pointer: &str| {
            recorded(&chart, pointer)
                .as_i64()
                .unwrap_or_else(|| panic!("{pointer} is recorded as a whole number of bytes"))
        };
        let watch_plain = bytes_at("/x_watch_postfix/bytes");
        let shipped_bound = bytes_at("/x_watch_script/shipped_build_bound/bytes");
        assert_eq!(
            shipped_bound, watch_plain,
            "the window-script read's shipped-build bound is the plain WATCH read's measured X"
        );
        assert_eq!(
            at("/x_watch_script/shipped_build_bound/mib"),
            mib_1dp(shipped_bound as f64 / MIB)
        );
        assert!(
            shipped_bound as f64 / MIB <= window_term_by_rule,
            "the shipped-build bound ({} MiB) is within the window term by the rule ({} MiB)",
            shipped_bound as f64 / MIB,
            window_term_by_rule
        );
        assert!(
            bytes_at("/x_watch_script/bytes") <= watch_plain,
            "the window-script read's measured X is at most the plain read's"
        );

        // The uploads, from configuration and code, at the chart's values.
        let ChartUploads {
            budget_mib: upload_budget_mib,
            waiters,
            overflow_slot_mib: overflow_slot,
            total_mib: uploads,
            pools: chart_pools,
            plan_catchup_mib,
        } = chart_uploads();
        assert_eq!(at("/uploads/budget"), upload_budget_mib as f64);
        assert_eq!(at("/uploads/waiters"), waiters as f64);
        assert_eq!(at("/uploads/waiter_window_mib"), UPLOAD_WAITER_WINDOW_MIB);
        assert_eq!(
            at("/uploads/overflow_slot_reference"),
            mib_1dp(overflow_slot)
        );
        assert_eq!(at("/uploads/total"), mib_1dp(uploads));

        // The Valkey connections' at-rest buffers at the chart's pools, per
        // topology; the 10-primary figure is in T's list. Computed in every
        // feature shape; the record's figures are the shipped build's, with
        // the `valkey-tls` feature on, so only that shape compares them:
        // without it fred's `Command`, whose size is one in-flight slot of the
        // bound (`IN_FLIGHT_SLOT_BYTES`), is smaller, and so is `C_conn`.
        // Every shape still derives T with its own `C_conn` and checks the
        // list fits (below).
        use crate::storage::connection_buffers::{at_rest_bound_bytes, Topology};
        let c_conn = |topology: Topology| at_rest_bound_bytes(chart_pools, topology) as f64 / MIB;
        let mib_2dp = |value: f64| (value * 100.0).round() / 100.0;
        let shipped_shape = cfg!(feature = "valkey-tls");
        if shipped_shape {
            for (name, topology) in [
                ("standalone", Topology::standalone()),
                ("cluster_1", Topology::cluster(1)),
                ("cluster_3", Topology::cluster(3)),
                ("cluster_10", Topology::cluster(10)),
            ] {
                assert_eq!(
                    at(&format!("/c_conn_mib/{name}")),
                    mib_2dp(c_conn(topology)),
                    "c_conn_mib.{name} is at_rest_bound_bytes at the chart's pools"
                );
            }
        }
        assert_eq!(
            recorded(&chart, "/c_conn_in_t_list"),
            serde_json::json!("cluster_10")
        );
        let margin = 0.2 * chart_limit;
        assert_eq!(at("/margin_20pct"), margin);

        let chart_fixed = floor
            + load_terms
            + window_term
            + reference_read
            + resident_cache
            + chart_cold_start_reads
            + audit
            + uploads
            + c_conn(Topology::cluster(10))
            + margin;
        let (f, t, total) = t_by_the_recorded_rule(chart_limit, floor, p_f, chart_fixed);
        assert!(
            total <= chart_limit,
            "the chart limit's term list fits at T = {t}: {total} MiB"
        );
        assert_eq!(at("/f"), f);
        assert_eq!(recorded_u64(&chart, "/t"), t);
        assert_eq!(at("/t_times_1_plus_pf"), (1.0 + p_f) * t as f64);
        // The total carries `C_conn`: the record's is the shipped shape's.
        if shipped_shape {
            assert_eq!(
                mib_1dp(total),
                at("/t_table_total"),
                "t_table_total is the chart limit's term list total at T = {t}"
            );
        }
        assert_eq!(
            CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT, t,
            "the reference T must be the term list's T for the chart's {chart_limit_mib} MiB pod \
             memory limit (f = {f}, {total:.1} MiB: the real-id terms, the structural cache at \
             its resident budget with its cold-start reads, their index build included, and \
             sampled audit, the uploads and the Valkey connections' \
             buffers on a 10-primary cluster); a limit or term change needs a new record, the \
             pod memory plan's and SDD §16's figures moving with it"
        );
        // The pod memory plan, the runtime form of this list at 0.01 steps,
        // derives the same T at the reference limit for the chart's pools.
        assert_eq!(
            plan_catchup_mib, CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            "the pod memory plan's T at the reference limit for the chart's configuration"
        );
    }

    /// The plan's term of `plan`'s line named `name`, in bytes.
    fn line_term(plan: &PodMemoryPlan, name: &str) -> u64 {
        plan.line_terms()
            .iter()
            .find(|term| term.name == name)
            .unwrap_or_else(|| panic!("the plan's line has a {name:?} term"))
            .bytes
    }

    /// W2: the pod memory plan the service derives at the chart's pod memory
    /// limit for the chart's configuration reproduces the chart limit's
    /// record ([`CHART_LIMIT_DERIVATION_JSON`]) term by term — the plan
    /// replaces that hand-derived list at run time, so every figure of the
    /// record, at the precision it records, must be the plan's:
    ///
    /// - the budgets: `T` 744 at f = 0.15, `B` 384, `C` 256 (its resident
    ///   budget × (1 + φ) to the byte), `S` 4,096;
    /// - each line term: the floor, the five subscription load terms and their
    ///   sum, the cache, its cold-start reads and audit, `T × (1 + P_f)`, the
    ///   window term (to the byte: the watch-read pool build's `X`), the
    ///   reference read, the uploads (the overflow slot to the byte) and
    ///   `C_conn` on 10 primaries;
    /// - the line: its 80 % limit, its total before `C_conn`, its fenced-read
    ///   terms, and its headroom on a standalone store and on 1, 3 and 10
    ///   primaries, which must not be negative;
    /// - the subscription ceiling's allowance per subscription on each
    ///   topology, floored to 0.1 KiB as recorded and at least the measured
    ///   49.8 KiB, so the ceiling holds; and the catch-up slots the line
    ///   funds, at least the CPU-derived default of 8, which so binds.
    ///
    /// The plan is derived at the limit `Config::from_env` resolves with no
    /// readable limit (the reference, the chart's), for the chart's
    /// configuration (`PlanInputs::reference`, with the pools read from the
    /// chart rather than this machine's "auto" ones), so a drift between the
    /// plan's rules and the record fails here. The figures that carry
    /// `C_conn` are the shipped build's, with the `valkey-tls` feature on,
    /// and are compared only in that shape; every shape checks the line
    /// closes on every topology, the ceiling holds and the CPU default binds.
    // Guard test: pinned by name in scripts/guard-tests.sh and in the
    // validation-records guard (docs-internal/engineering-best-practices.md
    // § Validation records); rename or remove it there too.
    #[test]
    fn pod_memory_plan_at_the_chart_limit_reproduces_its_records_line_term_by_term() {
        use crate::storage::connection_buffers::{at_rest_bound_bytes, Topology};
        const MIB: u64 = 1024 * 1024;
        let mib = |bytes: u64| bytes as f64 / MIB as f64;
        let mib_2dp = |bytes: u64| (mib(bytes) * 100.0).round() / 100.0;
        let Some(chart) = sr_11_8_record(CHART_LIMIT_DERIVATION_JSON) else {
            return;
        };
        let at = |pointer: &str| {
            recorded(&chart, pointer)
                .as_f64()
                .unwrap_or_else(|| panic!("{pointer} is recorded as a number"))
        };
        let ChartUploads { pools, .. } = chart_uploads();
        let (plan, cpu_slots) = temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            (
                cfg.pod_memory.clone(),
                cfg.subscriber_max_concurrent_catchups as u64,
            )
        });
        // The chart's pools, not this machine's "auto" ones.
        let plan = PodMemoryPlan::derive(
            plan.limit().clone(),
            &PlanInputs {
                pools,
                ..PlanInputs::reference()
            },
        )
        .expect("the plan closes at the chart's limit");
        assert_eq!(
            plan.limit().mib(),
            recorded_u64(&chart, "/limit_mib"),
            "the plan is at the record's limit, the chart's"
        );

        // The budgets.
        assert_eq!(plan.catchup_budget_mib(), recorded_u64(&chart, "/t"));
        assert_eq!(plan.catchup_step_centi() as f64 / 100.0, at("/f"));
        assert_eq!(plan.upload_budget_mib() as f64, at("/uploads/budget"));
        assert_eq!(
            plan.structural_cache_budget_mib(),
            recorded_u64(&chart, "/structural_cache_budget_mib")
        );
        assert_eq!(
            plan.structural_cache_resident_budget_bytes(),
            recorded_u64(&chart, "/structural_cache_bytes")
        );
        assert_eq!(plan.max_subscriptions() as f64, at("/subscription/ceiling"));

        // Each line term, at the record's precision.
        let term = |name: &str| line_term(&plan, name);
        assert_eq!(mib_1dp(mib(term("steady-state floor"))), at("/floor"));
        let load_parts = [
            (
                "established subscriptions",
                "subscriptions_4096_at_49_8_kib",
            ),
            ("catch-up slots", "catchup_slots_8"),
            ("stream readers", "readers_replies"),
            ("one lagging layer", "lagging_layer_hard_cap"),
            (
                "catch-ups before their first poll",
                "registration_to_first_poll",
            ),
        ];
        for (name, part) in load_parts {
            assert_eq!(
                mib_1dp(mib(term(name))),
                at(&format!("/load_terms/parts/{part}")),
                "{name}"
            );
        }
        let load_terms: u64 = load_parts.iter().map(|(name, _)| term(name)).sum();
        assert_eq!(mib_1dp(mib(load_terms)), at("/load_terms/value"));
        assert_eq!(mib_2dp(term("structural cache")), at("/structural_cache"));
        assert_eq!(
            mib_1dp(mib(term("structural cache cold-start reads"))),
            at("/cold_start_plain_reads")
        );
        assert_eq!(
            mib_1dp(mib(term("structural cache audit"))),
            at("/sampled_audit")
        );
        assert_eq!(
            mib(term("catch-up memory budget")),
            at("/t_times_1_plus_pf")
        );
        assert_eq!(
            term("fan-out window transient"),
            recorded_u64(&chart, "/x_watch_pool_build/bytes"),
            "the window term is the watch-read pool build's X, to the byte"
        );
        assert_eq!(
            mib_1dp(mib(term("fan-out window transient"))),
            at("/window_term")
        );
        assert_eq!(
            mib_1dp(mib(term("checkpoint reference read"))),
            at("/reference_read")
        );
        assert_eq!(
            term("one maximum-size upload"),
            recorded_u64(&chart, "/uploads/overflow_slot_reference_bytes")
        );
        let uploads =
            term("upload budget") + term("queued uploads") + term("one maximum-size upload");
        assert_eq!(mib_1dp(mib(uploads)), at("/uploads/total"));
        assert_eq!(
            mib(term("queued uploads")),
            at("/uploads/waiters") * at("/uploads/waiter_window_mib")
        );

        // The line: limit, total before C_conn, fenced-read terms, headroom
        // per topology.
        let c_conn_10 = term("connection pools");
        let line_limit = plan.line_limit_bytes();
        let before_c_conn = plan.line_total_bytes() - c_conn_10;
        assert_eq!(mib(line_limit), at("/line/limit_80pct"));
        assert_eq!(mib_1dp(mib(before_c_conn)), at("/line/total_before_c_conn"));
        assert_eq!(
            mib_1dp(mib(line_limit - before_c_conn)),
            at("/line/headroom_before_c_conn")
        );
        // The record sums the fenced-read terms at the 0.1 MiB it records
        // each at (1,488 + 143.0 + 172.9).
        assert_eq!(
            mib_1dp(
                mib(term("catch-up memory budget"))
                    + mib_1dp(mib(term("fan-out window transient")))
                    + mib_1dp(mib(term("checkpoint reference read")))
            ),
            at("/line/terms/fenced_read_catchup_terms")
        );
        assert_eq!(mib_1dp(mib(uploads)), at("/line/terms/uploads"));
        assert_eq!(recorded(&chart, "/line/closes"), serde_json::json!(true));
        let shipped_shape = cfg!(feature = "valkey-tls");
        let topologies = [
            ("standalone", Topology::standalone()),
            ("cluster_1", Topology::cluster(1)),
            ("cluster_3", Topology::cluster(3)),
            ("cluster_10", Topology::cluster(10)),
        ];
        assert_eq!(
            c_conn_10,
            at_rest_bound_bytes(pools, Topology::cluster(10)),
            "the plan's line carries C_conn on 10 primaries"
        );
        let subscriptions = term("established subscriptions");
        let slots = PlanInputs::reference().catchup_slots;
        let slot = term("catch-up slots") / slots;
        assert_eq!(slot * slots, term("catch-up slots"));
        assert_eq!(
            mib_2dp(slot),
            at("/subscription/catchup_slot_at_ceiling_mib")
        );
        let mut funded_slots = u64::MAX;
        for (name, topology) in topologies {
            let c_conn = at_rest_bound_bytes(pools, topology);
            let line = before_c_conn + c_conn;
            assert!(
                line <= line_limit,
                "the line must close at the chart's limit on {name}: {} MiB over {} MiB",
                mib(line),
                mib(line_limit)
            );
            let headroom = line_limit - line;
            // The ceiling's allowance: the line's room for the established
            // subscriptions, over the ceiling, in KiB.
            let allowance_kib =
                mib(headroom + subscriptions) * 1024.0 / plan.max_subscriptions() as f64;
            assert!(
                allowance_kib >= at("/subscription/measured_kib"),
                "{name}: an established subscription may cost {allowance_kib:.2} KiB, under the \
                 one measured: the ceiling does not hold"
            );
            funded_slots = funded_slots.min((headroom + term("catch-up slots")) / slot);
            if shipped_shape {
                assert_eq!(
                    mib_2dp(c_conn),
                    at(&format!("/c_conn_mib/{name}")),
                    "{name}"
                );
                assert_eq!(
                    mib_1dp(mib(headroom)),
                    at(&format!("/line/headroom_after_c_conn/{name}")),
                    "{name}"
                );
                assert_eq!(
                    (allowance_kib * 10.0).floor() / 10.0,
                    at(&format!("/subscription/allowance_kib/{name}")),
                    "{name}: floored to 0.1 KiB"
                );
            }
        }
        assert_eq!(
            line_limit - plan.line_total_bytes(),
            plan.line_headroom_bytes(),
            "the plan's headroom is its 10-primary line's"
        );
        if shipped_shape {
            assert_eq!(funded_slots as f64, at("/subscription/memory_slot_bound"));
        }
        assert!(
            funded_slots >= cpu_slots,
            "the line funds {funded_slots} catch-up slots, under the {cpu_slots} the CPU bound \
             sets: memory would bind"
        );
    }

    // ── The pod memory plan against its own record and the chart (b.fj4) ────
    //
    // The plan's record (`POD_MEMORY_PLAN_DERIVATION_JSON`) is the plan at
    // 2,048 / 2,706 / 4,096 / 5,120 / 8,192 MiB and at its minimum for the
    // chart's configuration, and at 2,709 / 2,808 / 4,096 / 5,120 / 8,192 MiB
    // and at its minimum for that configuration with a backup store on (its
    // `backup_store` block), built and at exact shares, written by a script
    // that re-implements the plan's rules in Python (`scripts/derive_plan.py`
    // in the record's folder) from `POD_MEMORY_PLAN_INPUTS_JSON`. The two
    // derivations agree term by term, and the record's inputs are the code's.

    /// The pod memory plan's record, generated by its folder's
    /// `scripts/derive_plan.py`.
    const POD_MEMORY_PLAN_DERIVATION_JSON: &str =
        "docs-internal/validation/pod-memory-plan-b-fj4-20261014/derivation.json";

    /// The record's hand-assembled inputs: the configuration, the constants
    /// and the code-computed terms the script derives from.
    const POD_MEMORY_PLAN_INPUTS_JSON: &str =
        "docs-internal/validation/pod-memory-plan-b-fj4-20261014/inputs.json";

    /// The `(name, bytes)` list of a recorded `line_terms` / `margin_terms`.
    fn recorded_terms(doc: &str, pointer: &str) -> Vec<(String, u64)> {
        recorded(doc, pointer)
            .as_array()
            .unwrap_or_else(|| panic!("{pointer} is recorded as an array"))
            .iter()
            .map(|term| {
                (
                    term["name"]
                        .as_str()
                        .unwrap_or_else(|| panic!("{pointer}: a term has a name"))
                        .to_owned(),
                    term["bytes"]
                        .as_u64()
                        .unwrap_or_else(|| panic!("{pointer}: a term has its bytes")),
                )
            })
            .collect()
    }

    /// That the record's reduced fraction at `pointer`, `"numerator/
    /// denominator"`, in units of `unit` bytes, equals `exact`, an exact value
    /// of the plan in 1/EXACT_DEN byte: compared cross-multiplied, exactly.
    fn assert_exact_is_recorded(doc: &str, pointer: &str, exact: u128, unit: u128) {
        let value = recorded(doc, pointer);
        let text = value
            .as_str()
            .unwrap_or_else(|| panic!("{pointer} is recorded as a \"numerator/denominator\""));
        let (num, den) = text
            .split_once('/')
            .unwrap_or_else(|| panic!("{pointer}: {text:?} is not \"numerator/denominator\""));
        let parse = |part: &str| -> u128 {
            part.parse()
                .unwrap_or_else(|e| panic!("{pointer}: {part:?} is not a whole number ({e})"))
        };
        let (num, den) = (parse(num), parse(den));
        let product = |a: u128, b: u128| {
            a.checked_mul(b)
                .unwrap_or_else(|| panic!("{pointer}: {a} × {b} overflows u128"))
        };
        assert_eq!(
            product(exact, den),
            product(product(num, memory_plan::EXACT_DEN), unit),
            "{pointer}: the plan's exact value, {exact} / EXACT_DEN B, is not the record's {text}"
        );
    }

    fn plan_terms(terms: &[memory_plan::PlanTerm]) -> Vec<(String, u64)> {
        terms
            .iter()
            .map(|term| (term.name.to_owned(), term.bytes))
            .collect()
    }

    /// The configuration of the record's `backup_store` block: the chart's
    /// with a backup store on — checkpointing enabled with an adapter and the
    /// tail — so the margin charges the base bootstraps and the mirror's
    /// `M_ext` (b.mhq). `Config::from_env` derives exactly these switches
    /// for such a configuration
    /// (`config_from_env_funds_the_ext_mirror_exactly_when_checkpointing_is_on_with_an_adapter`).
    fn record_backup_store_inputs() -> PlanInputs {
        PlanInputs {
            base_bootstrap: true,
            ext_mirror: true,
            ..PlanInputs::reference()
        }
    }

    /// W3: the plan the service derives for the chart's configuration
    /// (`PlanInputs::reference`, pinned to the chart by
    /// [`plan_inputs_reference_is_the_charts_configuration`]), and for it
    /// with a backup store on ([`record_backup_store_inputs`], the record's
    /// `backup_store` block), at each limit the record lists for it is the
    /// record's, term by term and to the byte: a limit below the minimum is
    /// refused, and at every other the budgets — the mirror's `M_ext` among
    /// them, absent without a backup store — `T` and its step, every line and
    /// margin term in order, the line's total, limit and headroom on each
    /// topology, the margin left and the minimum are the record's, and so,
    /// exactly, are the line at exact shares at `T`'s step and at the next
    /// and the room it leaves, which closure and `T`'s step are decided on.
    /// The record's `M_ext` floor is the code's. The record is derived by an
    /// independent Python implementation of the plan's rules, so a change to
    /// either side's rules fails here. Its terms carry `C_conn`, which is the
    /// shipped build's (the `valkey-tls` feature on). Without that feature the
    /// connection pools' buffers are smaller, so the build's own minimum is at
    /// most the record's (equal to it with a backup store, where the margin,
    /// which no connection buffer enters, sets it): there a refusal must name
    /// the build's minimum, not the record's, a limit the record refuses must
    /// refuse only when it is below the build's minimum (one at or above it
    /// closes), every other listed limit must close with the record's `M_ext`,
    /// and the record's minimum and each closing limit's other terms are not
    /// compared.
    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn pod_memory_plan_matches_its_record_term_by_term() {
        let Some(record) = sr_11_8_record(POD_MEMORY_PLAN_DERIVATION_JSON) else {
            return;
        };
        assert_plan_matches_its_record(
            &record,
            "/plan",
            &PlanInputs::reference(),
            &[2_048, 4_096, 5_120, 8_192],
        );
        let backup = record_backup_store_inputs();
        assert_plan_matches_its_record(
            &record,
            "/plan/backup_store",
            &backup,
            &[2_709, 2_808, 4_096, 5_120, 8_192],
        );
        assert_eq!(
            recorded_u64(&record, "/plan/backup_store/minimum_limit_mib"),
            2_809,
            "the chart's configuration with a backup store starts from 2,809 MiB"
        );
        assert_eq!(
            recorded_u64(&record, "/plan/ext_mirror_term/floor_mib"),
            memory_plan::ext_mirror_floor_mib(&backup),
            "the record's M_ext floor is the code's"
        );
        assert_eq!(
            recorded(&record, "/plan/ext_mirror_term/side"),
            serde_json::json!("margin")
        );
    }

    /// [`pod_memory_plan_matches_its_record_term_by_term`] for one recorded
    /// configuration: `inputs`, recorded under `prefix`, listing at least
    /// `required` limits and its minimum.
    fn assert_plan_matches_its_record(
        record: &str,
        prefix: &str,
        inputs: &PlanInputs,
        required: &[u64],
    ) {
        use crate::storage::connection_buffers::{at_rest_bound_bytes, Topology};
        let inputs = *inputs;
        let minimum = recorded_u64(record, &format!("{prefix}/minimum_limit_mib"));
        let limits = recorded(record, &format!("{prefix}/limits"));
        let limits = limits
            .as_object()
            .unwrap_or_else(|| panic!("{prefix}/limits is recorded as an object"));
        let listed: Vec<u64> = limits
            .keys()
            .map(|mib| mib.parse().expect("a limit in MiB"))
            .collect();
        for required in required.iter().copied().chain([minimum]) {
            assert!(
                listed.contains(&required),
                "{prefix}: the record lists {required} MiB"
            );
        }
        let shipped_shape = cfg!(feature = "valkey-tls");
        // This build's own minimum, read off the plan at the record's: the
        // record's in the shipped shape, at most the record's without
        // `valkey-tls` (smaller pool buffers).
        let build_minimum =
            PodMemoryPlan::derive(PodMemorySource::Fixed(minimum << 20).resolve(), &inputs)
                .unwrap_or_else(|e| {
                    panic!("{prefix}: the record's minimum, {minimum} MiB, closes: {e}")
                })
                .minimum_limit_mib()
                .expect("a plan that closes has a minimum");
        if shipped_shape {
            assert_eq!(build_minimum, minimum, "{prefix}: the record's minimum");
        } else {
            assert!(
                build_minimum <= minimum,
                "{prefix}: without valkey-tls the minimum, {build_minimum} MiB, is at most the \
                 record's {minimum} MiB"
            );
        }
        let mut compared = 0;
        for mib in listed {
            let at = |field: &str| format!("{prefix}/limits/{mib}/{field}");
            let plan = PodMemoryPlan::derive(PodMemorySource::Fixed(mib << 20).resolve(), &inputs);
            if recorded(record, &at("closes")) == serde_json::json!(false) {
                // Without `valkey-tls` a limit the record refuses closes when
                // it is at or above this build's lower minimum: it must close,
                // and nothing else of it is compared.
                if !shipped_shape && mib >= build_minimum {
                    plan.unwrap_or_else(|e| {
                        panic!("{prefix}: {mib} MiB is at or above this build's minimum: {e}")
                    });
                    continue;
                }
                let refusal = plan.expect_err("the record says this limit does not close");
                assert!(
                    refusal.contains(&format!(
                        "raise resources.limits.memory to at least {} MiB",
                        memory_plan::grouped(build_minimum)
                    )),
                    "{prefix}: {mib} MiB: the refusal names the minimum, {build_minimum} MiB: \
                     {refusal}"
                );
                compared += 1;
                continue;
            }
            let plan =
                plan.unwrap_or_else(|e| panic!("{prefix}: {mib} MiB closes in the record: {e}"));
            // `M_ext` depends on the limit and the size cap alone, so it is
            // compared in every build shape: null in the record exactly when
            // the plan funds no mirror.
            assert_eq!(
                plan.ext_mirror_bytes().map(|bytes| bytes >> 20),
                recorded(record, &at("budgets/ext_mirror_mib")).as_u64(),
                "{prefix}: {mib} MiB: M_ext"
            );
            if !shipped_shape {
                continue;
            }
            assert_eq!(
                plan.minimum_limit_mib(),
                Some(minimum),
                "{prefix}: {mib} MiB"
            );
            let budget = |name: &str| recorded_u64(record, &at(&format!("budgets/{name}")));
            assert_eq!(
                (
                    plan.max_subscriptions(),
                    plan.structural_cache_budget_mib(),
                    plan.upload_budget_mib(),
                    plan.retirement_budget_bytes() >> 20,
                    plan.catchup_budget_mib(),
                    plan.catchup_step_centi(),
                ),
                (
                    budget("max_subscriptions"),
                    budget("structural_cache_mib"),
                    budget("upload_mib"),
                    budget("retirement_mib"),
                    budget("catchup_mib"),
                    budget("catchup_step_centi"),
                ),
                "{prefix}: {mib} MiB: S, C, B, R_ref, T and T's step"
            );
            assert_eq!(
                plan_terms(plan.line_terms()),
                recorded_terms(record, &at("line_terms")),
                "{prefix}: {mib} MiB: the line's terms"
            );
            assert_eq!(
                plan_terms(plan.margin_terms()),
                recorded_terms(record, &at("margin_terms")),
                "{prefix}: {mib} MiB: the margin's terms"
            );
            for (field, value) in [
                ("limit_bytes", plan.limit().bytes),
                ("line_total_bytes", plan.line_total_bytes()),
                ("line_limit_bytes", plan.line_limit_bytes()),
                ("line_headroom_bytes/cluster_10", plan.line_headroom_bytes()),
                ("margin_named_bytes", plan.margin_named_bytes()),
                ("margin_left_bytes", plan.margin_left_bytes()),
            ] {
                assert_eq!(
                    value,
                    recorded_u64(record, &at(field)),
                    "{prefix}: {mib} MiB: {field}"
                );
            }
            // The other topologies' headroom: the line with its 10-primary
            // connection term replaced by theirs.
            let c_conn_10 = at_rest_bound_bytes(inputs.pools, Topology::cluster(10));
            for (name, topology) in [
                ("standalone", Topology::standalone()),
                ("cluster_1", Topology::cluster(1)),
                ("cluster_3", Topology::cluster(3)),
            ] {
                let line = plan.line_total_bytes() - c_conn_10
                    + at_rest_bound_bytes(inputs.pools, topology);
                assert_eq!(
                    plan.line_limit_bytes() - line,
                    recorded_u64(record, &at(&format!("line_headroom_bytes/{name}"))),
                    "{prefix}: {mib} MiB: headroom on {name}"
                );
            }
            // The exact line, `T`'s term included, at `T`'s step and at the
            // next, and the room it leaves within 80 % of the limit: what
            // closure and `T`'s step are decided on, the plan's exact values
            // (in 1/EXACT_DEN byte) against the record's reduced fractions.
            let exact_line = |step: u64| -> u128 {
                memory_plan::exact_terms_at(plan.limit().bytes, &inputs, step)
                    .0
                    .iter()
                    .map(|term| term.exact)
                    .sum()
            };
            let step = plan.catchup_step_centi();
            let at_step = exact_line(step);
            let allowance = memory_plan::LINE_ALLOWANCE.exact_units_scaled(plan.limit().bytes, 0);
            assert_exact_is_recorded(record, &at("exact/line_total_bytes"), at_step, 1);
            if step < 100 {
                assert_exact_is_recorded(
                    record,
                    &at("exact/line_total_next_step_bytes"),
                    exact_line(step + 1),
                    1,
                );
            } else {
                assert_eq!(
                    recorded(record, &at("exact/line_total_next_step_bytes")),
                    serde_json::Value::Null,
                    "{prefix}: {mib} MiB: T's step is 1.00, with no next step"
                );
            }
            assert_exact_is_recorded(
                record,
                &at("exact/line_slack_mib"),
                allowance
                    .checked_sub(at_step)
                    .unwrap_or_else(|| panic!("{prefix}: {mib} MiB: the exact line fits")),
                1 << 20,
            );
            compared += 1;
        }
        if shipped_shape {
            assert_eq!(
                compared,
                limits.len(),
                "{prefix}: every recorded limit compared"
            );
        }
    }

    /// The record's inputs are the code's: its configuration is
    /// `PlanInputs::reference`, each constant is the one the plan's terms use,
    /// and each code-computed term is what the code computes at that
    /// configuration (`C_conn` in the shipped `valkey-tls` shape, whose
    /// figure the record holds). A changed constant or rule input fails here
    /// first, before the term-by-term comparison, naming the input to
    /// re-record before the record is re-derived.
    // Guard test: pinned by name in scripts/guard-tests.sh and in the
    // validation-records guard (docs-internal/engineering-best-practices.md
    // § Validation records); rename or remove it there too.
    #[test]
    fn pod_memory_plan_record_inputs_are_the_codes() {
        use crate::storage::connection_buffers::{at_rest_bound_bytes, Topology};
        use memory_plan::*;
        let Some(doc) = sr_11_8_record(POD_MEMORY_PLAN_INPUTS_JSON) else {
            return;
        };
        let reference = PlanInputs::reference();
        let input = |pointer: &str| recorded_u64(&doc, pointer);
        let pools = reference.pools;
        for (field, value) in [
            ("/configuration/pools/data", pools.data as u64),
            ("/configuration/pools/txn", pools.txn as u64),
            ("/configuration/pools/maintenance", pools.maintenance as u64),
            ("/configuration/pools/snapshot", pools.snapshot as u64),
            ("/configuration/pools/readers", pools.readers as u64),
            ("/configuration/catchup_slots", reference.catchup_slots),
            (
                "/configuration/fenced_read_concurrency",
                reference.fenced_read_concurrency,
            ),
            ("/configuration/byte_ceiling", reference.byte_ceiling),
            ("/configuration/upload_waiters", reference.upload_waiters),
            ("/configuration/max_layer_size", reference.max_layer_size),
            ("/constants/floor_bytes", POD_MEMORY_FLOOR_BYTES),
            (
                "/constants/subscription_bytes_deci",
                SUBSCRIPTION_BYTES_DECI,
            ),
            (
                "/constants/queued_entry_bytes_deci",
                QUEUED_ENTRY_BYTES_DECI,
            ),
            (
                "/constants/pre_poll_entries_per_slot_centi",
                PRE_POLL_ENTRIES_PER_SLOT_CENTI,
            ),
            (
                "/constants/pre_poll_shared_payload_bytes",
                PRE_POLL_SHARED_PAYLOAD_BYTES,
            ),
            (
                "/constants/catchup_slot_bytes_per_ceiling_byte",
                CATCHUP_SLOT_BYTES_PER_CEILING_BYTE,
            ),
            (
                "/constants/xread_reply_target_bytes",
                crate::storage::stream_mux::XREAD_REPLY_TARGET_BYTES,
            ),
            (
                "/constants/consumer_backlog_bytes",
                crate::storage::notifications::CONSUMER_BACKLOG_BYTES,
            ),
            (
                "/constants/consumer_channel_capacity",
                crate::storage::notifications::CONSUMER_CHANNEL_CAPACITY as u64,
            ),
            (
                "/constants/backlog_hard_cap_factor",
                crate::storage::stream_mux::BACKLOG_HARD_CAP_FACTOR as u64,
            ),
            (
                "/constants/plan_index_budget_centi",
                crate::storage::structural_cache::PLAN_INDEX_BUDGET_CENTI,
            ),
            (
                "/constants/cold_start_read_permits",
                crate::storage::structural_cache::STRUCTURAL_COLD_START_READ_PERMITS as u64,
            ),
            (
                "/constants/cold_start_entry_to_read_centi",
                crate::storage::structural_cache_cold_start::COLD_START_ENTRY_TO_READ_CENTI,
            ),
            (
                "/constants/cold_start_build_peak_centi",
                crate::storage::structural_cache_cold_start::COLD_START_BUILD_PEAK_CENTI,
            ),
            ("/constants/window_transient_bytes", WINDOW_TRANSIENT_BYTES),
            (
                "/constants/reference_read_short_id_bytes",
                REFERENCE_READ_SHORT_ID_BYTES,
            ),
            (
                "/constants/record_short_id_byte_ceiling",
                RECORD_SHORT_ID_BYTE_CEILING,
            ),
            ("/constants/record_byte_ceiling", RECORD_BYTE_CEILING),
            (
                "/constants/upload_waiter_window_bytes",
                UPLOAD_WAITER_WINDOW_BYTES,
            ),
            ("/constants/allocator_slack_bytes", ALLOCATOR_SLACK_BYTES),
            (
                "/constants/bootstrap_max_inflight",
                crate::checkpoint::BOOTSTRAP_MAX_INFLIGHT as u64,
            ),
            (
                "/constants/ext_mirror_item_overhead_kib",
                crate::checkpoint::ext_mirror::ITEM_OVERHEAD_KIB,
            ),
            (
                "/code_terms/upload_budget_floor_bytes",
                crate::grpc::upload_admission::upload_budget_floor_bytes(reference.max_layer_size),
            ),
            (
                "/code_terms/one_maximum_size_upload_bytes",
                crate::grpc::upload_admission::parse_charge(
                    reference.max_layer_size,
                    0,
                    reference.max_layer_size,
                ),
            ),
        ] {
            assert_eq!(input(field), value, "{POD_MEMORY_PLAN_INPUTS_JSON} {field}");
        }
        let backup = record_backup_store_inputs();
        for (field, value) in [
            ("/configuration/base_bootstrap", reference.base_bootstrap),
            ("/configuration/ext_mirror", reference.ext_mirror),
            ("/backup_store/base_bootstrap", backup.base_bootstrap),
            ("/backup_store/ext_mirror", backup.ext_mirror),
        ] {
            assert_eq!(
                recorded(&doc, field),
                serde_json::json!(value),
                "{POD_MEMORY_PLAN_INPUTS_JSON} {field}"
            );
        }
        // The mirror's charge for one maximum-size value, as the script
        // computes it from these inputs, is the sender's.
        assert_eq!(
            (reference.max_layer_size.div_ceil(1024)
                + input("/constants/ext_mirror_item_overhead_kib"))
                * 1024,
            crate::checkpoint::ext_mirror::item_charge_kib(reference.max_layer_size) * 1024
        );
        // The plan's rule for the cold-start term is the hold allowance's.
        assert_eq!(
            crate::storage::structural_cache_cold_start::plain_cold_start_hold_allowance_bytes(
                reference.byte_ceiling
            ),
            reference.byte_ceiling
                * input("/constants/cold_start_entry_to_read_centi")
                * (100 + input("/constants/cold_start_build_peak_centi"))
                / 10_000
        );
        if cfg!(feature = "valkey-tls") {
            for (name, topology) in [
                ("standalone", Topology::standalone()),
                ("cluster_1", Topology::cluster(1)),
                ("cluster_3", Topology::cluster(3)),
                ("cluster_10", Topology::cluster(10)),
            ] {
                assert_eq!(
                    input(&format!("/code_terms/c_conn_bytes/{name}")),
                    at_rest_bound_bytes(reference.pools, topology),
                    "C_conn on {name}"
                );
            }
        }
    }

    /// `PlanInputs::reference`, the configuration the reference plan and the
    /// plan's record are derived for, is the configuration the chart renders
    /// at its defaults: `Config::from_env` given the chart's values for every
    /// setting the plan charges derives exactly those inputs, so the chart and
    /// the hard-coded reference cannot drift apart.
    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn plan_inputs_reference_is_the_charts_configuration() {
        let chart = |path: &[&str]| {
            let raw = chart_value(path);
            (raw != "null").then_some(raw)
        };
        let env: Vec<(String, Option<String>)> = [
            ("LIVE_EDIT_VALKEY_POOL_SIZE", &["valkey", "poolSize"][..]),
            (
                "LIVE_EDIT_MAINTENANCE_POOL_SIZE",
                &["valkey", "maintenancePoolSize"],
            ),
            (
                "LIVE_EDIT_SNAPSHOT_POOL_SIZE",
                &["valkey", "snapshotPoolSize"],
            ),
            (
                "LIVE_EDIT_SUBSCRIBER_POOL_SIZE",
                &["valkey", "subscriberPoolSize"],
            ),
            (
                "LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS",
                &["valkey", "subscriberMaxConcurrentCatchups"],
            ),
            (
                "LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY",
                &["valkey", "catchupSnapshotReadConcurrency"],
            ),
            (
                "LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY",
                &["valkey", "catchupFencedReadConcurrency"],
            ),
            (
                "LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES",
                &["valkey", "snapshotAtomicReadMaxBytes"],
            ),
            ("LIVE_EDIT_MAX_LAYER_SIZE", &["layer", "maxSize"]),
            (
                "LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS",
                &["layer", "uploadAdmissionMaxWaiters"],
            ),
            ("LIVE_EDIT_CHECKPOINT_ENABLED", &["checkpoint", "enabled"]),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", &["checkpoint", "adapter"]),
            (
                "LIVE_EDIT_CHECKPOINT_TAIL_ENABLED",
                &["checkpoint", "tail", "enabled"],
            ),
        ]
        .into_iter()
        .map(|(var, path)| (var.to_owned(), chart(path)))
        .chain(
            REQUIRED_VARS
                .iter()
                .map(|(var, value)| ((*var).to_owned(), value.map(str::to_owned))),
        )
        .collect();
        let inputs = temp_env::with_vars(env, || {
            PlanInputs::from_config(
                &Config::from_env_with(PodMemorySource::Reference)
                    .expect("the chart's values parse"),
            )
        });
        assert_eq!(
            inputs,
            PlanInputs::reference(),
            "PlanInputs::reference must be the chart's configuration; change the two together"
        );
    }

    /// The chart's local values file (`values-local.yaml`, the local-cluster
    /// install) sets a smaller pod memory limit than the chart's: the
    /// service must start there. It overrides none of the settings the plan
    /// charges (checked key by key), so the plan is the chart configuration's
    /// at its limit, and it closes.
    #[test]
    fn pod_memory_plan_closes_at_the_local_values_files_memory_limit() {
        const LOCAL_VALUES: &str = "helm/live-edit-service/values-local.yaml";
        for path in [
            &["valkey", "poolSize"][..],
            &["valkey", "maintenancePoolSize"],
            &["valkey", "snapshotPoolSize"],
            &["valkey", "subscriberPoolSize"],
            &["valkey", "subscriberMaxConcurrentCatchups"],
            &["valkey", "catchupSnapshotReadConcurrency"],
            &["valkey", "catchupFencedReadConcurrency"],
            &["valkey", "snapshotAtomicReadMaxBytes"],
            &["layer", "maxSize"],
            &["layer", "uploadAdmissionMaxWaiters"],
            &["checkpoint", "enabled"],
            &["checkpoint", "adapter"],
            &["checkpoint", "tail", "enabled"],
        ] {
            assert_eq!(
                values_file_value(LOCAL_VALUES, path),
                None,
                "{LOCAL_VALUES} overrides {path:?}: derive this test's plan with that value too"
            );
        }
        let limit_mib = memory_quantity_mib(
            &values_file_value(LOCAL_VALUES, &["resources", "limits", "memory"])
                .expect("the local values file sets resources.limits.memory"),
        );
        let plan = PodMemoryPlan::derive(
            PodMemorySource::Fixed(limit_mib << 20).resolve(),
            &PlanInputs::reference(),
        )
        .unwrap_or_else(|e| panic!("the plan must close at {LOCAL_VALUES}'s {limit_mib} MiB: {e}"));
        assert!(
            plan.minimum_limit_mib().is_some_and(|m| m <= limit_mib),
            "{limit_mib} MiB is at or above the minimum"
        );
    }

    /// W6: each of the four removed memory-budget variables refuses startup
    /// whatever its value, naming the variable, what replaced it (the budget
    /// derived from the container's memory limit, with its value at this
    /// pod's limit) and what to do instead. At an 8 GiB pod with the chart's
    /// pools (`T` depends on them), so the value named is the one derived
    /// there, not the reference pod's. The values span what each variable
    /// once meant: `0` (the cache disabled, unlimited subscriptions), its old
    /// defaults, larger values, non-numbers, an empty value and `u64::MAX`.
    #[test]
    fn config_from_env_refuses_each_removed_memory_budget_variable_naming_its_replacement() {
        for (var, derived) in [
            ("LIVE_EDIT_UPLOAD_MEMORY_BUDGET_MIB", "614 MiB"),
            ("LIVE_EDIT_CATCHUP_BUILD_MEMORY_BUDGET_MIB", "1,607 MiB"),
            ("LIVE_EDIT_STRUCTURAL_CACHE_MAX_MIB", "409 MiB"),
            (
                "LIVE_EDIT_SUBSCRIBER_MAX_SUBSCRIPTIONS",
                "6,553 subscriptions",
            ),
        ] {
            for value in [
                "0",
                "256",
                "384",
                "512",
                "744",
                "4096",
                "10000",
                "abc",
                "lots",
                "",
                "18446744073709551615",
            ] {
                let mut env = snapshot_knob_env(var, value);
                env.push(("LIVE_EDIT_VALKEY_POOL_SIZE", Some("64")));
                env.push(("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some("256")));
                temp_env::with_vars(env, || {
                    let err = Config::from_env_with(PodMemorySource::Fixed(8 << 30))
                        .expect_err("a removed variable refuses startup");
                    for part in [
                        format!("{var} is no longer read"),
                        "derived from the container's memory limit at startup".to_owned(),
                        format!("({derived} at this pod's 8,192 MiB)"),
                        "Unset it".to_owned(),
                        "change the pod's memory limit".to_owned(),
                    ] {
                        assert!(err.contains(&part), "{var}={value:?}: {part:?} in {err}");
                    }
                });
            }
        }
    }

    /// `checkpoint_base_bootstrap_enabled` holds exactly when checkpointing is
    /// enabled, an adapter is wired and the tail is on: its full truth table.
    #[test]
    fn checkpoint_base_bootstrap_enabled_needs_checkpointing_an_adapter_and_the_tail() {
        let base = temp_env::with_vars(REQUIRED_VARS, || {
            Config::from_env_with(PodMemorySource::Reference).expect("defaults parse")
        });
        for enabled in [false, true] {
            for adapter in [CheckpointAdapter::Disabled, CheckpointAdapter::InMemory] {
                for tail in [false, true] {
                    let cfg = Config {
                        checkpoint_enabled: enabled,
                        checkpoint_adapter: adapter,
                        checkpoint_tail_enabled: tail,
                        ..base.clone()
                    };
                    assert_eq!(
                        cfg.checkpoint_base_bootstrap_enabled(),
                        enabled && !matches!(adapter, CheckpointAdapter::Disabled) && tail,
                        "enabled={enabled} adapter={adapter:?} tail={tail}"
                    );
                }
            }
        }
    }

    /// The pod memory plan `Config::from_env` derives charges the checkpoint
    /// base bootstraps' margin term — 2 × `BOOTSTRAP_MAX_INFLIGHT` maximum-size
    /// layers, 400 MiB at the default size cap — exactly when the bootstrap
    /// runs: with checkpointing enabled, an adapter wired and the tail on, and
    /// not when any one of the three is off.
    #[test]
    fn config_from_env_charges_the_base_bootstraps_exactly_when_the_bootstrap_runs() {
        for (enabled, adapter, tail, charged) in [
            ("true", "InMemory", "true", true),
            ("false", "InMemory", "true", false),
            ("true", "Disabled", "true", false),
            ("true", "InMemory", "false", false),
        ] {
            let mut env = REQUIRED_VARS.to_vec();
            env.extend([
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some(enabled)),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some(adapter)),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some(tail)),
            ]);
            let cfg = temp_env::with_vars(env, || {
                Config::from_env_with(PodMemorySource::Fixed(8 << 30)).expect("the config parses")
            });
            let term = cfg
                .pod_memory
                .margin_terms()
                .iter()
                .find(|t| t.name == "checkpoint base bootstraps")
                .map(|t| t.bytes);
            let case = format!("enabled={enabled} adapter={adapter} tail={tail}");
            assert_eq!(cfg.checkpoint_base_bootstrap_enabled(), charged, "{case}");
            assert_eq!(term, charged.then_some(400 << 20), "{case}");
        }
    }

    /// The environment that sets `LIVE_EDIT_CHECKPOINT_ENABLED` to `enabled`
    /// and `LIVE_EDIT_CHECKPOINT_ADAPTER` to `adapter`, with whatever that
    /// adapter needs to parse, the tail as `tail`, and the pools pinned small
    /// so the plan's line does not depend on the machine's CPUs.
    fn checkpoint_env(
        enabled: &'static str,
        adapter: &'static str,
        tail: &'static str,
    ) -> Vec<(&'static str, Option<&'static str>)> {
        let mut env = REQUIRED_VARS.to_vec();
        env.extend([
            ("LIVE_EDIT_CHECKPOINT_ENABLED", Some(enabled)),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some(adapter)),
            ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some(tail)),
            ("LIVE_EDIT_VALKEY_POOL_SIZE", Some("8")),
            ("LIVE_EDIT_SUBSCRIBER_POOL_SIZE", Some("4")),
        ]);
        match adapter {
            "localdisk" => env.push((
                "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                Some("/var/lib/live-edit/checkpoints"),
            )),
            "StorageApi" => env.extend([
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/checkpoints")),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
            ]),
            _ => {}
        }
        env
    }

    /// T12 (c), through `Config::from_env` (b.mhq): the pod memory plan funds
    /// the external-value mirror — `ext_mirror_bytes` is `Some(M_ext)`, the
    /// margin carries its term, and `checkpoint_adapter_wired` holds, which
    /// is what `main.rs` builds the mirror on — exactly when checkpointing is
    /// enabled with an adapter, over every adapter with checkpointing on and
    /// off. The tail does not matter to it: with the tail off the mirror's
    /// term is charged and the base bootstraps' is not. The margin's named
    /// bytes are always its terms'. `M_ext` at 8 GiB is 204 MiB.
    #[test]
    fn config_from_env_funds_the_ext_mirror_exactly_when_checkpointing_is_on_with_an_adapter() {
        const MIRROR_TERM: &str = "backup copies of external values";
        let mut cases = Vec::new();
        for enabled in ["false", "true"] {
            for adapter in ["Disabled", "InMemory", "localdisk", "StorageApi"] {
                let wired = enabled == "true" && adapter != "Disabled";
                cases.push((enabled, adapter, "true", wired, wired));
            }
        }
        cases.push(("true", "InMemory", "false", true, false));
        for (enabled, adapter, tail, mirror, bootstraps) in cases {
            let case = format!("enabled={enabled} adapter={adapter} tail={tail}");
            let cfg = temp_env::with_vars(checkpoint_env(enabled, adapter, tail), || {
                Config::from_env_with(PodMemorySource::Fixed(8 << 30))
                    .unwrap_or_else(|e| panic!("{case}: the config parses: {e}"))
            });
            let margin = cfg.pod_memory.margin_terms();
            let term = |name: &str| margin.iter().find(|t| t.name == name).map(|t| t.bytes);
            assert_eq!(cfg.checkpoint_adapter_wired(), mirror, "{case}");
            assert_eq!(
                cfg.pod_memory.ext_mirror_bytes(),
                mirror.then_some(204 << 20),
                "{case}: the mirror's budget"
            );
            assert_eq!(
                term(MIRROR_TERM),
                mirror.then_some(204 << 20),
                "{case}: the margin's mirror term"
            );
            assert_eq!(
                term("checkpoint base bootstraps"),
                bootstraps.then_some(400 << 20),
                "{case}: the margin's bootstrap term"
            );
            assert_eq!(
                cfg.pod_memory.margin_named_bytes(),
                margin.iter().map(|t| t.bytes).sum::<u64>(),
                "{case}: the margin's named bytes are its terms'"
            );
        }
    }

    /// I5 (b.mhq): the mirror runs exactly when the plan funds it. The
    /// service's wiring, `ext_mirror_channel_for_plan` over the plan
    /// `Config::from_env` derives (as `main.rs` builds it), returns the
    /// mirror's channel exactly when checkpointing is on with an adapter,
    /// with the plan's `M_ext` (204 MiB at 8 GiB) as its byte budget and the
    /// queue capacity setting as its item bound, and nothing otherwise.
    #[test]
    fn ext_mirror_channel_for_plan_builds_the_mirror_exactly_when_the_plan_funds_it() {
        let metrics = std::sync::Arc::new(crate::observability::metrics::metrics_scrape_only(
            "ext-mirror-wiring",
        ));
        for enabled in ["false", "true"] {
            for adapter in ["Disabled", "InMemory", "localdisk"] {
                let case = format!("enabled={enabled} adapter={adapter}");
                let mut env = checkpoint_env(enabled, adapter, "true");
                env.push(("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY", Some("3")));
                let cfg = temp_env::with_vars(env, || {
                    Config::from_env_with(PodMemorySource::Fixed(8 << 30))
                        .unwrap_or_else(|e| panic!("{case}: the config parses: {e}"))
                });
                let funded = enabled == "true" && adapter != "Disabled";
                let channel = crate::checkpoint::ext_mirror_channel_for_plan(
                    &cfg.pod_memory,
                    cfg.checkpoint_ext_mirror_queue_capacity,
                    std::sync::Arc::clone(&metrics),
                );
                assert_eq!(channel.is_some(), funded, "{case}: the mirror is built");
                let Some((sender, mut rx)) = channel else {
                    continue;
                };
                assert_eq!(
                    Some(sender.budget_bytes()),
                    cfg.pod_memory.ext_mirror_bytes(),
                    "{case}: the mirror's budget is the plan's M_ext"
                );
                assert_eq!(sender.budget_bytes(), 204 << 20, "{case}: M_ext at 8 GiB");
                // The item bound is the capacity setting: three tiny values
                // fill the channel, and a fourth finds it full.
                for i in 0..4u8 {
                    sender.send(format!("{i:064x}"), sender.copy_value(&[i]), "0".to_owned());
                }
                let mut queued = 0;
                while rx.try_recv().is_ok() {
                    queued += 1;
                }
                assert_eq!(
                    queued, 3,
                    "{case}: the queue capacity setting bounds the items"
                );
            }
        }
    }

    /// T12 (d), through `Config::from_env` (b.mhq): with a backup store on
    /// (checkpointing enabled, an adapter, the tail on) the smallest limit
    /// that starts is 2,809 MiB, the margin's. At 2,808 MiB startup is refused
    /// naming the mirror's term with the settings that switch it on and size
    /// it, and the remedy, raising the limit first, never turning off
    /// checkpointing or the adapter; the same
    /// limit starts with the backup store off, funding no mirror.
    #[test]
    fn config_from_env_with_a_backup_store_refuses_one_mib_below_its_minimum_naming_the_mirror_term(
    ) {
        let parse = |enabled: &'static str, mib: u64| {
            temp_env::with_vars(checkpoint_env(enabled, "InMemory", "true"), || {
                Config::from_env_with(PodMemorySource::Fixed(mib << 20))
            })
        };
        let err = parse("true", 2_808).expect_err("one MiB below the minimum");
        for part in [
            "is below the 2,809 MiB this configuration needs, 1 MiB short",
            "backup copies of external values 101 MiB (LIVE_EDIT_CHECKPOINT_ENABLED, \
             LIVE_EDIT_CHECKPOINT_ADAPTER, LIVE_EDIT_MAX_LAYER_SIZE)",
            "raise resources.limits.memory to at least 2,809 MiB, which costs nothing else; \
             failing that, lower LIVE_EDIT_MAX_LAYER_SIZE, or turn off incremental backups \
             (LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false), though recovery points then come only \
             from full backup passes, so a newly created or just-restored layer has no backup of \
             its current state until the next pass",
        ] {
            assert!(err.contains(part), "{part:?} in {err}");
        }
        let at_minimum = parse("true", 2_809).expect("the minimum starts");
        assert_eq!(at_minimum.pod_memory.ext_mirror_bytes(), Some(101 << 20));
        let off = parse("false", 2_808).expect("the same limit starts with the backup store off");
        assert_eq!(off.pod_memory.ext_mirror_bytes(), None);
    }

    /// W6, through `Config::from_env`: a pool, slot, waiter or upload-size
    /// setting raised past what the reference pod can carry refuses startup,
    /// naming the smallest limit that configuration needs and the term the
    /// setting sizes with the settings that size it; the setting reaches the
    /// plan from its variable. (Where
    /// closure ends for each, and that just inside it `T` shrinks, is
    /// `memory_plan`'s
    /// `raising_a_setting_past_closure_refuses_naming_the_minimum_and_just_inside_t_shrinks`.)
    #[test]
    fn config_from_env_refuses_a_setting_past_closure_naming_the_minimum_and_its_term() {
        for (var, value, term) in [
            (
                "LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS",
                "200",
                "catch-up slots",
            ),
            ("LIVE_EDIT_VALKEY_POOL_SIZE", "4000", "connection pools"),
            (
                "LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS",
                "4000",
                "queued uploads",
            ),
            (
                "LIVE_EDIT_MAX_LAYER_SIZE",
                "1073741824",
                "one maximum-size upload",
            ),
        ] {
            temp_env::with_vars(snapshot_knob_env(var, value), || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("the reference pod cannot carry this setting");
                for part in [
                    "this pod's memory limit, 5,120 MiB (no readable limit; the 5,120 MiB \
                     reference size), is below the"
                        .to_owned(),
                    "MiB this configuration needs".to_owned(),
                    format!("the largest fixed terms: {term} "),
                    "raise resources.limits.memory to at least".to_owned(),
                    // The term names the setting that sizes it; the pools'
                    // term names every pool setting.
                    if term == "connection pools" {
                        "MiB (LIVE_EDIT_VALKEY_POOL_SIZE, LIVE_EDIT_MAINTENANCE_POOL_SIZE, \
                         LIVE_EDIT_SNAPSHOT_POOL_SIZE, LIVE_EDIT_SUBSCRIBER_POOL_SIZE)"
                            .to_owned()
                    } else {
                        format!("MiB ({var})")
                    },
                ] {
                    assert!(err.contains(&part), "{var}={value}: {part:?} in {err}");
                }
            });
        }
    }

    #[test]
    fn catchup_memory_factors_equal_the_recorded_task10_split() {
        use crate::storage::memory_budget::{BUILD_MEMORY_FACTOR, SNAPSHOT_MEMORY_FACTOR};
        let Some(inputs) = sr_11_8_record(TASK10_RESULTS_JSON) else {
            return;
        };
        let factor = |pointer: &str| {
            recorded(&inputs, pointer)
                .as_f64()
                .unwrap_or_else(|| panic!("{pointer} is recorded as a number"))
        };
        assert_eq!(
            SNAPSHOT_MEMORY_FACTOR as f64,
            factor("/memory/SNAPSHOT_MEMORY_FACTOR")
        );
        assert_eq!(
            BUILD_MEMORY_FACTOR as f64,
            factor("/memory/BUILD_MEMORY_FACTOR")
        );
        assert_eq!(
            SNAPSHOT_MEMORY_FACTOR + BUILD_MEMORY_FACTOR,
            recorded_u64(&inputs, "/memory/combined_factor"),
            "the two factors split the recorded combined factor"
        );
    }

    /// Each knob lands unchanged. At a 1 PiB pod memory limit, so the memory
    /// a million read permits or lane connections would take is charged
    /// without the pod memory plan refusing it (`memory_plan`). `P_f` stops at
    /// 50: the catch-up budget's bound `T × (1 + P_f)` at its smallest step,
    /// 1 % of the limit, outgrows the line beside the shares of the limit the
    /// other budgets take once `P_f` passes 61 (at the chart's other
    /// settings; about that at these), whatever the limit.
    #[test]
    fn config_from_env_snapshot_knobs_explicit_values_round_trip() {
        for (var, field) in SNAPSHOT_KNOBS {
            let largest = if var == "LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY" {
                50
            } else {
                1_000_000
            };
            for value in [1_u64, 3, largest] {
                let raw = value.to_string();
                temp_env::with_vars(snapshot_knob_env(var, &raw), || {
                    let cfg = Config::from_env_with(PodMemorySource::Fixed(1 << 50))
                        .unwrap_or_else(|e| panic!("{var}={raw} must parse, got: {e}"));
                    assert_eq!(field(&cfg), value, "{var}={raw} must land unchanged");
                });
            }
        }
    }

    #[test]
    fn config_from_env_snapshot_knobs_reject_zero() {
        for (var, _) in SNAPSHOT_KNOBS {
            let err = assert_snapshot_knob_rejected(
                var,
                "0",
                "an empty lane cannot be built, a zero ceiling would refuse every atomic read, \
                 and with no read permit of either class no catch-up could read its layer",
            );
            assert!(
                err.contains("must be greater than 0"),
                "the zero reject for {var} must state the constraint, got: {err}"
            );
        }
    }

    #[test]
    fn config_from_env_snapshot_knobs_reject_non_numeric_values() {
        for (var, _) in SNAPSHOT_KNOBS {
            for garbage in ["abc", "-1", "1.5", "1e6", ""] {
                let _ = assert_snapshot_knob_rejected(var, garbage, "it is not a positive integer");
            }
        }
    }

    // ── SR-11.8: shipped defaults against the recorded derivation ───────────
    //
    // The two ceilings are derived, not picked: Task 1 of the proven-snapshot-
    // reads Epic re-solved the AMR ceiling against the scaled measured excess
    // allowance and committed the result as `operating` in `resolve.json`
    // (reproduced by `excess_allowance.py` beside it). Items per entry is Epic
    // 1's, in `derivation.json`. An edit to either side — the constant or the
    // committed record — fails here.
    //
    // The records live under `docs-internal/`, which `.publishignore` leaves
    // out of the GitHub source release, so they are read at run time rather
    // than embedded: the published tree still compiles its unit tests, and
    // these three skip there. Only a missing `docs-internal/` skips them; in
    // this repository a moved or missing record fails.

    const SR_11_8_RESOLVE_JSON: &str = "docs-internal/validation/proven-snapshot-reads-20261001/raw/task1-excess-allowance/resolve.json";
    const REAL_ID_DERIVATION_JSON: &str =
        "docs-internal/validation/real-id-byte-ceiling-20261006/derivation.json";
    const SR_11_8_DERIVATION_JSON: &str =
        "docs-internal/validation/snapshot-read-cost-20260927/raw/task8-derivation/derivation.json";

    /// The committed record at `path` (relative to the crate root), or `None`
    /// in the published source tree, which has no `docs-internal/`.
    fn sr_11_8_record(path: &str) -> Option<String> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("docs-internal").is_dir() {
            eprintln!("skipped: no docs-internal/ (published source tree), so no {path}");
            return None;
        }
        Some(
            std::fs::read_to_string(root.join(path))
                .unwrap_or_else(|e| panic!("the committed derivation record {path}: {e}")),
        )
    }

    /// The value at JSON `pointer` in the recorded document `doc`.
    fn recorded(doc: &str, pointer: &str) -> serde_json::Value {
        let parsed: serde_json::Value =
            serde_json::from_str(doc).expect("the committed derivation record is valid JSON");
        parsed
            .pointer(pointer)
            .unwrap_or_else(|| panic!("the derivation record carries {pointer}"))
            .clone()
    }

    fn recorded_u64(doc: &str, pointer: &str) -> u64 {
        recorded(doc, pointer)
            .as_u64()
            .unwrap_or_else(|| panic!("{pointer} is recorded as a non-negative integer"))
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn snapshot_atomic_read_default_limits_equal_the_recorded_amr_operating_ceiling() {
        let Some(resolve) = sr_11_8_record(SR_11_8_RESOLVE_JSON) else {
            return;
        };
        let items = recorded_u64(&resolve, "/operating/items");
        // The byte ceiling is the item ceiling on the real-id ray, recorded
        // beside its evidence; resolve.json's operating.bytes is the
        // short-id ray it replaced.
        let Some(real_id) = sr_11_8_record(REAL_ID_DERIVATION_JSON) else {
            return;
        };
        let bytes = recorded_u64(&real_id, "/byte_ceiling/bytes");
        let bpi = recorded(&real_id, "/real_id_bytes_per_item/value")
            .as_f64()
            .expect("the real-id ray is recorded as a number");
        assert_eq!(
            recorded_u64(&real_id, "/item_ceiling/items"),
            items,
            "the real-id record must sit on the shipped item ceiling"
        );
        assert_eq!(
            bytes,
            (items as f64 * bpi).floor() as u64,
            "the recorded byte ceiling must be floor(items × the real-id bytes per item)"
        );
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                cfg.snapshot_atomic_read_max_items, items,
                "the shipped item ceiling must be resolve.json's operating.items"
            );
            assert_eq!(
                cfg.snapshot_atomic_read_max_bytes, bytes,
                "the shipped byte ceiling must be the real-id record's byte_ceiling.bytes"
            );
        });
    }

    /// The fence cost model's `bpi` for an unrecorded layer is the byte
    /// ceiling's real-id ray: one record, so the two never disagree.
    #[test]
    fn fitted_bytes_per_item_is_the_recorded_real_id_ray() {
        let Some(real_id) = sr_11_8_record(REAL_ID_DERIVATION_JSON) else {
            return;
        };
        let bpi = recorded(&real_id, "/real_id_bytes_per_item/value")
            .as_f64()
            .expect("the real-id ray is recorded as a number");
        assert_eq!(
            crate::storage::snapshot_fence::cost::FITTED_BYTES_PER_ITEM,
            bpi,
            "FITTED_BYTES_PER_ITEM must be derivation.json's real_id_bytes_per_item.value"
        );
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// The constant holds the ratio in thousandths (5.266, the report's
    /// figure), so the comparison rule is "the recorded ratio rounded to the
    /// nearest thousandth". That is exact, not a tolerance: any edit of the
    /// constant fails, and so does any re-recording of the ratio that moves it
    /// by half a thousandth or more. A smaller change cannot move the constant's
    /// resolution, and the window test below catches one that moves the shipped
    /// window.
    #[test]
    fn snapshot_items_per_entry_constant_is_the_recorded_ratio_in_thousandths() {
        let Some(derivation) = sr_11_8_record(SR_11_8_DERIVATION_JSON) else {
            return;
        };
        let ratio = recorded(&derivation, "/items_per_entry/value")
            .as_f64()
            .expect("items_per_entry.value is recorded as a number");
        assert_eq!(
            SNAPSHOT_ITEMS_PER_ENTRY_MILLI,
            (ratio * 1_000.0).round() as u64,
            "SNAPSHOT_ITEMS_PER_ENTRY_MILLI must be derivation.json's items_per_entry.value \
             ({ratio}) rounded to the nearest thousandth"
        );
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// The uncapped window at the shipped item ceiling is Epic 2's recorded
    /// `operating.fanout_window` (7,131), and it is also floor(items ÷ the
    /// *unrounded* recorded ratio) — so rounding the ratio to thousandths moves
    /// no window, and a ratio drift that would is caught even when it is too
    /// small to change the thousandths. The window the shipped configuration
    /// actually uses is the Task 10 cap (2,141): the uncapped 7,131 fails
    /// SR-11.6 (i) on AMR.
    #[test]
    fn snapshot_fanout_window_at_the_default_ceiling_equals_the_recorded_window() {
        let (Some(resolve), Some(derivation), Some(results)) = (
            sr_11_8_record(SR_11_8_RESOLVE_JSON),
            sr_11_8_record(SR_11_8_DERIVATION_JSON),
            sr_11_8_record(TASK10_RESULTS_JSON),
        ) else {
            return;
        };
        let items = recorded_u64(&resolve, "/operating/items");
        let uncapped = recorded_u64(&resolve, "/operating/fanout_window");
        let cap = recorded_u64(&results, "/fanout_window_cap/cap");
        let ratio = recorded(&derivation, "/items_per_entry/value")
            .as_f64()
            .expect("items_per_entry.value is recorded as a number");
        assert_eq!(
            snapshot_fanout_window_uncapped(items),
            uncapped,
            "the uncapped window at the shipped ceiling must be resolve.json's \
             operating.fanout_window"
        );
        assert_eq!(
            uncapped,
            (items as f64 / ratio).floor() as u64,
            "the recorded window must be the item ceiling over derivation.json's unrounded \
             items_per_entry.value"
        );
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                cfg.snapshot_fanout_window(),
                cap,
                "FANOUT_WINDOW at the shipped ceiling must be the Task 10 record's \
                 fanout_window_cap.cap, not the uncapped {uncapped}"
            );
        });
    }

    /// `FANOUT_WINDOW_MAX` is the cap the Task 10 record derives from the
    /// recorded inputs (`fanout_window_cap.cap`): re-deriving the cap without
    /// changing the constant, or the reverse, fails here.
    #[test]
    fn fanout_window_max_is_the_recorded_task10_cap() {
        let Some(results) = sr_11_8_record(TASK10_RESULTS_JSON) else {
            return;
        };
        assert_eq!(
            FANOUT_WINDOW_MAX,
            recorded_u64(&results, "/fanout_window_cap/cap"),
            "FANOUT_WINDOW_MAX must be results.json's fanout_window_cap.cap"
        );
        assert_eq!(
            snapshot_fanout_window_uncapped(recorded_u64(
                &results,
                "/fanout_window_cap/smallest_ceiling_items_the_cap_binds"
            )),
            FANOUT_WINDOW_MAX,
            "the record's smallest binding ceiling must be where the uncapped window first \
             reaches the cap"
        );
    }

    /// `FANOUT_INDEX_WINDOW_MEMBERS` is the round-1 member bound the Task 10
    /// record derives (`fanout_index_window_members.members`, 26,855), and
    /// the record's own inputs re-derive it: the per-stream allowance SR-11.6
    /// (i) leaves at the shipped ceiling — the same allowance the window cap
    /// is sized from — over the largest measured whole-index round-1 cost per
    /// member (2.360 ms of sandbox shard time over 21,550 members) × the AMR /
    /// sandbox cost ratio, rounded down. One member more would pass the
    /// allowance. Editing the constant, or re-recording any input without it,
    /// fails here.
    #[test]
    fn fanout_index_window_members_is_the_recorded_task10_bound() {
        let Some(results) = sr_11_8_record(TASK10_RESULTS_JSON) else {
            return;
        };
        let number = |pointer: &str| {
            recorded(&results, pointer)
                .as_f64()
                .unwrap_or_else(|| panic!("{pointer} is recorded as a number"))
        };
        assert_eq!(
            FANOUT_INDEX_WINDOW_MEMBERS,
            recorded_u64(&results, "/fanout_index_window_members/members"),
            "FANOUT_INDEX_WINDOW_MEMBERS must be results.json's fanout_index_window_members.members"
        );
        assert_eq!(FANOUT_INDEX_WINDOW_MEMBERS, 26_855);

        let rec = "/fanout_index_window_members";
        let (target, a_level) = (
            number(&format!("{rec}/target_ms")),
            number(&format!("{rec}/A_level_ms")),
        );
        let (r, p_a, p_f) = (
            number(&format!("{rec}/R_design")),
            number(&format!("{rec}/P_a")),
            number(&format!("{rec}/P_f")),
        );
        let allowance = (target - (r * p_a + 1.0) * a_level) / (r * p_f);
        assert!(
            (allowance - number(&format!("{rec}/allowance_ms"))).abs() < 1e-9,
            "the recorded allowance is SR-11.6 (i)'s per-stream allowance from its own inputs"
        );
        assert!(
            (allowance - number("/fanout_window_cap/allowance_ms")).abs() < 1e-9,
            "the member bound is sized from the same allowance as FANOUT_WINDOW_MAX"
        );
        let (shard_ms, members, ratio) = (
            number(&format!("{rec}/window_shard_ms")),
            number(&format!("{rec}/window_members")),
            number(&format!("{rec}/amr_cost_ratio")),
        );
        assert_eq!(
            (shard_ms, members),
            (2.36, 21_550.0),
            "the conservative per-member cost: the 4x layer's whole-index round 1"
        );
        let per_member_amr_ms = shard_ms / members * ratio;
        let bound = (allowance / per_member_amr_ms).floor() as u64;
        assert_eq!(
            bound, FANOUT_INDEX_WINDOW_MEMBERS,
            "floor(allowance {allowance} ms ÷ {per_member_amr_ms} ms per member)"
        );
        assert!(
            bound as f64 * per_member_amr_ms <= allowance
                && (bound + 1) as f64 * per_member_amr_ms > allowance,
            "the bound is the most members whose round-1 window fits the allowance on AMR"
        );
    }

    /// The window script's record (b.gvf): the window model's fit, τ and its
    /// cut.
    const WINDOW_SCRIPT_RESULTS_JSON: &str =
        "docs-internal/validation/watch-window-script-b-gvf-20261010/results.json";

    /// τ and the window model of the structural WATCH read's script windows
    /// are the recorded figures (b.gvf): τ is the Task 10 record's per-stream
    /// allowance at the shipped ceiling (`fanout_window_cap.allowance_ms`,
    /// the allowance `FANOUT_WINDOW_MAX` is sized from) in ns, rounded down,
    /// and the window script's record carries it; each of the four
    /// coefficients is the window script's record's (`fanout_script_window`)
    /// and its unrounded fit rounded up to the ns; and the derivation record
    /// lists the same five constants. `fanout_script_window_modelled_shard_ns`
    /// is the model the record states, saturating. Re-fitting the record
    /// without the constants, or the reverse, fails here.
    #[test]
    fn fanout_script_window_constants_are_the_recorded_window_model_b_gvf() {
        let (Some(task10), Some(results), Some(derivation)) = (
            sr_11_8_record(TASK10_RESULTS_JSON),
            sr_11_8_record(WINDOW_SCRIPT_RESULTS_JSON),
            sr_11_8_record(CHART_LIMIT_DERIVATION_JSON),
        ) else {
            return;
        };
        let number = |doc: &str, pointer: &str| {
            recorded(doc, pointer)
                .as_f64()
                .unwrap_or_else(|| panic!("{pointer} is recorded as a number"))
        };

        let allowance_ms = number(&task10, "/fanout_window_cap/allowance_ms");
        assert_eq!(
            FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS,
            (allowance_ms * 1e6).floor() as u64,
            "τ is SR-11.6 (i)'s per-stream allowance in ns, rounded down"
        );
        assert_eq!(
            FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS,
            recorded_u64(&results, "/fanout_script_window/tau_ns")
        );
        assert_eq!(
            number(&results, "/fanout_script_window/tau_source/value_ms"),
            allowance_ms,
            "the window script's record cites the Task 10 allowance"
        );

        for (constant, key, unrounded, name) in [
            (
                FANOUT_SCRIPT_WINDOW_FIXED_NS,
                "fixed_ns",
                "fixed",
                "FANOUT_SCRIPT_WINDOW_FIXED_NS",
            ),
            (
                FANOUT_SCRIPT_COMMAND_NS,
                "command_ns",
                "command",
                "FANOUT_SCRIPT_COMMAND_NS",
            ),
            (
                FANOUT_SCRIPT_LIST_ENTRY_NS,
                "list_entry_ns",
                "list_entry",
                "FANOUT_SCRIPT_LIST_ENTRY_NS",
            ),
            (
                FANOUT_SCRIPT_REPLY_BYTE_NS,
                "reply_byte_ns",
                "reply_byte",
                "FANOUT_SCRIPT_REPLY_BYTE_NS",
            ),
        ] {
            assert_eq!(
                constant,
                recorded_u64(&results, &format!("/fanout_script_window/{key}")),
                "{name} must be results.json's fanout_script_window.{key}"
            );
            let fitted = number(
                &results,
                &format!("/fanout_script_window/fitted_ns_unrounded/{unrounded}"),
            );
            assert_eq!(
                constant,
                fitted.ceil() as u64,
                "{name} is its fit ({fitted} ns) rounded up to the ns"
            );
            assert_eq!(
                constant,
                recorded_u64(
                    &derivation,
                    &format!("/fanout_script_window_model/constants/{name}")
                ),
                "derivation.json lists {name} as wired"
            );
        }
        assert_eq!(
            FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS,
            recorded_u64(
                &derivation,
                "/fanout_script_window_model/constants/FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS"
            )
        );

        assert_eq!(
            fanout_script_window_modelled_shard_ns(3, 5, 7),
            FANOUT_SCRIPT_WINDOW_FIXED_NS
                + 3 * FANOUT_SCRIPT_COMMAND_NS
                + 5 * FANOUT_SCRIPT_LIST_ENTRY_NS
                + 7 * FANOUT_SCRIPT_REPLY_BYTE_NS,
            "FIXED + CMD × wire + LIST × list entries + BYTE × reply bound"
        );
        let unbounded = u128::from(FANOUT_SCRIPT_WINDOW_FIXED_NS)
            + u128::from(u64::MAX)
                * u128::from(
                    FANOUT_SCRIPT_COMMAND_NS
                        + FANOUT_SCRIPT_LIST_ENTRY_NS
                        + FANOUT_SCRIPT_REPLY_BYTE_NS,
                );
        assert_eq!(
            u128::from(fanout_script_window_modelled_shard_ns(
                u64::MAX,
                u64::MAX,
                u64::MAX
            )),
            unbounded.min(u128::from(u64::MAX)),
            "saturates rather than wrapping"
        );
    }

    // ── FANOUT_WINDOW / FANOUT_WINDOW_BYTES derivation (SR-11.3) ────────────

    #[test]
    fn snapshot_fanout_window_scales_with_the_configured_item_limit() {
        // Whole multiples of 5.266 items give whole windows; anything between
        // floors. Below the cap the window follows the ceiling.
        assert_eq!(snapshot_fanout_window(5_266), 1_000);
        assert_eq!(snapshot_fanout_window(10_532), 2_000);
        // 11,275 items is the smallest ceiling whose window reaches the cap
        // (11,275 ÷ 5.266 = 2,141.09); one item less stays under it.
        assert_eq!(snapshot_fanout_window(11_274), 2_140);
        assert_eq!(snapshot_fanout_window(11_275), FANOUT_WINDOW_MAX);
        // Above it the ceiling's own window is larger, and the cap applies.
        assert_eq!(snapshot_fanout_window_uncapped(52_660), 10_000);
        assert_eq!(snapshot_fanout_window_uncapped(52_659), 9_999);
        assert_eq!(
            snapshot_fanout_window(52_660),
            FANOUT_WINDOW_MAX,
            "a ceiling whose own window exceeds the cap takes the cap"
        );
        temp_env::with_vars(
            snapshot_knob_env("LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS", "10532"),
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("an explicit item ceiling parses");
                assert_eq!(
                    cfg.snapshot_fanout_window(),
                    2_000,
                    "an operator who lowers the item ceiling narrows the window with it"
                );
            },
        );
    }

    /// An item ceiling under 5.266 floors to zero commands, and a window that
    /// admits none would never read anything; the window is held at one.
    #[test]
    fn snapshot_fanout_window_never_falls_below_one_command() {
        for items in 1..=5 {
            assert_eq!(
                snapshot_fanout_window(items),
                1,
                "an item ceiling of {items} must still admit one command"
            );
        }
        assert_eq!(
            snapshot_fanout_window(6),
            1,
            "6 / 5.266 floors to 1 unclamped"
        );
        assert_eq!(
            snapshot_fanout_window(11),
            2,
            "the one-command clamp is a floor: it does not hold larger windows at one"
        );
        assert_eq!(
            snapshot_fanout_window_uncapped(1),
            1,
            "the uncapped window is held at one command too"
        );
    }

    #[test]
    fn snapshot_fanout_window_does_not_overflow_at_the_largest_item_limit() {
        assert_eq!(
            snapshot_fanout_window_uncapped(u64::MAX),
            (u128::from(u64::MAX) * 1_000 / 5_266) as u64,
            "the uncapped scaling must not overflow at the largest item ceiling"
        );
        assert_eq!(
            snapshot_fanout_window(u64::MAX),
            FANOUT_WINDOW_MAX,
            "the largest item ceiling still takes the cap"
        );
    }

    #[test]
    fn snapshot_fanout_window_bytes_equals_the_configured_byte_limit() {
        temp_env::with_vars(
            snapshot_knob_env("LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES", "123456"),
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("an explicit byte ceiling parses");
                assert_eq!(cfg.snapshot_fanout_window_bytes(), 123_456);
            },
        );
    }

    // ── Structural write-planning cache budget (structural-edits SR-3.3) ────
    //
    // Its removed variable's refusal is in the W6 table,
    // `config_from_env_refuses_each_removed_memory_budget_variable_naming_its_replacement`.

    #[test]
    fn config_from_env_structural_cache_max_mib_defaults_to_256() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(cfg.structural_cache_max_mib, 256);
            assert_eq!(
                cfg.structural_cache_budget_bytes(),
                256 * 1024 * 1024,
                "the budget in bytes is the MiB value times 2^20"
            );
        });
    }

    // ── Structural cache sampled audit (structural-edits SR-3.6) ────────────

    const STRUCTURAL_CACHE_AUDIT_RATE: &str = "LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_RATE";
    const STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING: &str =
        "LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING";

    #[test]
    fn config_from_env_structural_cache_audit_knobs_default_to_one_in_a_thousand_and_on() {
        assert_eq!(
            STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT, 0.001,
            "the ruled default rate (E4): one committed structural call in a thousand"
        );
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                cfg.structural_cache_audit_rate,
                STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT
            );
            assert!(
                cfg.structural_cache_audit_above_ceiling,
                "the audit checks the entries the fenced cold start creates: on by default"
            );
        });
    }

    /// Both ends of the probability range are accepted as given: `0` is how an
    /// operator turns the audit off, `1` how a correctness run audits every
    /// call. An interior value round-trips too, in decimal and exponent form.
    #[test]
    fn config_from_env_structural_cache_audit_rate_accepts_zero_one_and_values_between() {
        for (raw, rate) in [
            ("0.0", 0.0),
            ("0", 0.0),
            ("1.0", 1.0),
            ("0.25", 0.25),
            ("1e-4", 1e-4),
        ] {
            temp_env::with_vars(snapshot_knob_env(STRUCTURAL_CACHE_AUDIT_RATE, raw), || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).unwrap_or_else(|e| {
                    panic!("{STRUCTURAL_CACHE_AUDIT_RATE}={raw} must parse: {e}")
                });
                assert_eq!(cfg.structural_cache_audit_rate, rate, "{raw}");
            });
        }
    }

    /// A rate the audit's sampler cannot take — above 1, below 0, NaN or an
    /// infinity, all of which parse as `f64` — fails startup naming the
    /// variable, rather than panicking at the first committed structural call.
    #[test]
    fn config_from_env_structural_cache_audit_rate_rejects_values_outside_zero_to_one() {
        for raw in ["1.5", "-0.1", "NaN", "inf", "-inf", "1.0000001"] {
            let err = assert_snapshot_knob_rejected(
                STRUCTURAL_CACHE_AUDIT_RATE,
                raw,
                "it is not a probability in [0, 1]",
            );
            assert!(
                err.contains("not a probability"),
                "{raw}: the reject must say why, got: {err}"
            );
        }
        for garbage in ["abc", "", "0.1%"] {
            let _ = assert_snapshot_knob_rejected(
                STRUCTURAL_CACHE_AUDIT_RATE,
                garbage,
                "it is not a number",
            );
        }
    }

    /// `true` / `false` only: a `1` or `yes` is rejected rather than read as
    /// on, so a typo cannot silently change whether the audit reads above the
    /// ceilings.
    #[test]
    fn config_from_env_structural_cache_audit_above_ceiling_parses_true_and_false_only() {
        for (raw, on) in [("true", true), ("false", false)] {
            temp_env::with_vars(
                snapshot_knob_env(STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING, raw),
                || {
                    let cfg =
                        Config::from_env_with(PodMemorySource::Reference).unwrap_or_else(|e| {
                            panic!("{STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING}={raw} must parse: {e}")
                        });
                    assert_eq!(cfg.structural_cache_audit_above_ceiling, on, "{raw}");
                },
            );
        }
        for garbage in ["1", "0", "yes", "on", ""] {
            let _ = assert_snapshot_knob_rejected(
                STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING,
                garbage,
                "it is not `true` or `false`",
            );
        }
    }

    /// The chart renders `LIVE_EDIT_STRUCTURAL_CACHE_AUDIT_ABOVE_CEILING` from
    /// `layer.structuralCacheAuditAboveCeiling` on every install, so a
    /// chart value that drifted from the binary's default would silently
    /// override it: the two must be changed together.
    #[test]
    fn structural_cache_audit_above_ceiling_default_equals_the_chart_value() {
        let raw = chart_value(&["layer", "structuralCacheAuditAboveCeiling"]);
        let chart: bool = raw.parse().unwrap_or_else(|e| {
            panic!(
                "{CHART_VALUES} layer.structuralCacheAuditAboveCeiling = {raw:?} is not \
                 `true` or `false`: {e}"
            )
        });
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                cfg.structural_cache_audit_above_ceiling, chart,
                "the binary's default above-ceiling switch must equal the chart's \
                 layer.structuralCacheAuditAboveCeiling; change the two together"
            );
        });
    }

    // ---- b.9qd: checkpoint-store reconciliation cadence -------------------
    //
    // `LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS` has three guards, and the
    // third is the interesting one. Zero is a tight-loop guard like its
    // siblings. The floor — the interval must be at least the DERIVED
    // reconciliation grace — guards a failure with no error at all: the pass
    // acts on a candidate only on its second observation a grace apart, so an
    // interval below the grace means every pass re-observes everything inside
    // its window, defers all of it, and reclaims nothing forever while still
    // paying for a full namespace enumeration each time.

    #[test]
    fn config_from_env_checkpoint_reconcile_interval_default_is_hourly() {
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            assert_eq!(
                cfg.checkpoint_reconcile_interval_secs, 3600,
                "checkpoint_reconcile_interval_secs default (b.9qd): the pass is pure garbage \
                 reclamation on a cold path and its input is two LIST-shaped enumerations, so \
                 the default trades reclamation latency for LIST traffic"
            );
        });
    }

    #[test]
    fn config_from_env_checkpoint_reconcile_interval_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("reconcile_interval=0 must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_reconcile_interval_explicit_override_round_trips() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS", Some("900")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit reconcile_interval=900 should parse");
                assert_eq!(cfg.checkpoint_reconcile_interval_secs, 900);
            },
        );
    }

    /// Below the derived grace → rejected, with an error that names the var,
    /// the bound, and how to satisfy it.
    ///
    /// On the local adapters the grace is the floor
    /// (`RECONCILE_GRACE_FLOOR` = 60 s), so `59` is the first rejected value
    /// and `60` the first accepted one. Pinning both sides means a future
    /// change to either constant is a deliberate edit here rather than a
    /// silent widening of the window in which a half-published generation can
    /// be reaped.
    #[test]
    fn config_from_env_checkpoint_reconcile_interval_below_the_derived_grace_is_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS", Some("59")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("an interval below the derived grace must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS"),
                    "error must name the offending var, got: {err}"
                );
                assert!(
                    err.contains("60"),
                    "error must state the bound the operator has to clear, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_reconcile_interval_at_the_derived_grace_is_accepted() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS", Some("60")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "an interval exactly at the derived grace is the boundary and must \
                             be accepted",
                );
                assert_eq!(cfg.checkpoint_reconcile_interval_secs, 60);
                assert_eq!(
                    cfg.checkpoint_reconcile_grace(),
                    Duration::from_secs(60),
                    "the local adapters take the floor"
                );
            },
        );
    }

    /// b.9qd R2 — an **unset** reconcile interval must never refuse to boot,
    /// however large the derived grace gets.
    ///
    /// The bound this knob is validated against is a function of a *different*
    /// variable: the grace comes from `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS`
    /// via the StorageApi publish arithmetic (`3 × timeout × 3 retries × 2 auth
    /// attempts` = `18 × timeout`). Raise that timeout past ~200 s and the grace
    /// exceeds an hour — at which point a flat 3600 default would fail
    /// `validate_checkpoint_budgets` and the service would refuse to start,
    /// citing a variable the operator never set, over a variable they never
    /// touched. Deriving the default (`max(3600, grace)`) is what keeps "I did
    /// not configure this" a permanently valid state.
    ///
    /// The second half is the part that keeps the first from being a loophole:
    /// deriving the *default* must not weaken the *explicit* validation. An
    /// operator who writes a number below the grace still gets the startup
    /// error, and it still names the bound and the derived default that would
    /// clear it.
    #[test]
    fn config_from_env_checkpoint_reconcile_interval_default_derives_from_a_grace_above_the_baseline(
    ) {
        // 300 s request timeout → publish worst case 300 × 3 × 2 = 1800 s →
        // grace 3 × 1800 = 5400 s, comfortably above the 3600 s baseline.
        let storage_api_vars = |interval: Option<&'static str>| {
            vec![
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("StorageApi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/checkpoints")),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("300000")),
                ("LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS", interval),
            ]
        };

        temp_env::with_vars(storage_api_vars(None), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                "an unset reconcile interval must ALWAYS boot: the default is derived from the \
                 grace, so it cannot fail the very cross-validation it feeds",
            );
            assert_eq!(
                cfg.checkpoint_reconcile_grace(),
                Duration::from_secs(5400),
                "sanity: 18 × the 300 s request timeout"
            );
            assert_eq!(
                cfg.checkpoint_reconcile_interval_secs, 5400,
                "the default is max(3600, grace), not the bare 3600 baseline — a flat baseline \
                 here would be BELOW the grace and would refuse to start on a variable the \
                 operator never set"
            );
        });

        // And the explicit path is untouched: 3600 would have been the old
        // default, and it is still rejected when set by hand.
        temp_env::with_vars(storage_api_vars(Some("3600")), || {
            let err = Config::from_env_with(PodMemorySource::Reference)
                .expect_err("deriving the DEFAULT must not weaken the explicit-value validation");
            assert!(
                err.contains("5400"),
                "the error must state the bound the operator has to clear, got: {err}"
            );
            assert!(
                err.contains("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS"),
                "and must name the variable the bound actually resolves from — otherwise the \
                 operator raises the interval by trial and error, got: {err}"
            );
        });
    }

    /// The recovery fetch budget is `max(rpc_deadline, 2 × request_timeout)`,
    /// and **both** terms have to stay live (b.gg7 review round 2).
    ///
    /// The two recovery hooks share this one expression and each of them
    /// scales both knobs in its own stall test, precisely so the `max` is not
    /// the thing under test there — which leaves the expression itself pinned
    /// nowhere. It needs pinning because each term guards a different
    /// direction, and a regression to either bare term is silent:
    ///
    /// - Drop the `2 × request_timeout` term and a deployment whose per-request
    ///   timeout is sized for multi-MB OMPF blobs (SR-4.2 sizes it so the blob
    ///   GET alone may consume it) gets a budget with negative headroom: every
    ///   lazy recovery of a large layer elapses deterministically and the
    ///   client reconnects into the same wall.
    /// - Drop the `rpc_deadline_ms` term and an operator who raises the RPC
    ///   deadline for slow clients silently does not raise the budget those
    ///   clients' recoveries run under.
    ///
    /// Asserted through `from_env` rather than on a hand-built `Config` so the
    /// env plumbing for both variables is covered with it.
    #[test]
    fn config_recovery_fetch_budget_takes_the_larger_of_its_two_terms() {
        let vars = |deadline: &'static str, request_timeout: &'static str| {
            vec![
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some(deadline)),
                (
                    "LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS",
                    Some(request_timeout),
                ),
            ]
        };

        // The defaults: 30 s deadline against 2 × 30 s request timeout — the
        // request-timeout term wins, which is the case the hooks actually run
        // in and the one a bare-`rpc_deadline_ms` regression would halve.
        temp_env::with_vars(vars("30000", "30000"), || {
            assert_eq!(
                Config::from_env_with(PodMemorySource::Reference)
                    .expect("defaults must parse")
                    .checkpoint_recovery_fetch_budget(),
                Duration::from_secs(60),
                "at the shipped defaults the budget is twice the per-request timeout, not one \
                 RPC deadline — the layer hook's fetch is a lineage walk (one LIST plus an \
                 allowance-bounded fetch per member) and SR-4.2 lets the blob GET alone \
                 consume a whole request timeout, so the per-member reserve that funds the \
                 floor answer is itself a whole request timeout. The `2 ×` term is what makes \
                 the budget at least TWO reserves at every tuning, which is the gather's stated \
                 `budget >= 2 × reserve` PRECONDITION: it is what leaves a full reserve of \
                 speculative budget above the base's own. It is not the mechanism — what holds \
                 the base's reserve back is the gather's kind-dependent allowance accounting, \
                 which caps a speculative tail at `min(reserve, remaining − reserve)` so a \
                 wedged member cannot eat it (see `gather_restore_lineage`, property (ii))"
            );
        });

        // A deployment with a long client deadline and a short backend timeout:
        // the deadline term wins and the budget follows it.
        temp_env::with_vars(vars("120000", "5000"), || {
            assert_eq!(
                Config::from_env_with(PodMemorySource::Reference)
                    .expect("must parse")
                    .checkpoint_recovery_fetch_budget(),
                Duration::from_secs(120),
                "raising the RPC deadline must raise the budget with it — a recovery bounded \
                 below what the caller is willing to wait for is a self-inflicted failure"
            );
        });

        // And the converse: raising only the backend timeout raises it too.
        temp_env::with_vars(vars("30000", "45000"), || {
            assert_eq!(
                Config::from_env_with(PodMemorySource::Reference)
                    .expect("must parse")
                    .checkpoint_recovery_fetch_budget(),
                Duration::from_secs(90),
                "both terms are live; neither is a constant the other can shadow"
            );
        });
    }

    /// `acquire_session_cancel`'s recovery-pass budget is
    /// `min(refresh / 2, ttl − refresh − 1 ms)`, floored at 1 ms, from the two
    /// presence lease knobs.
    ///
    /// Each case pins one term of the derivation:
    ///
    /// - **The defaults (15 s / 60 s) → 7.5 s**, half a refresh interval, via
    ///   `from_env` so the `Config` method and the env plumbing agree with the
    ///   free function `LiveEditService` evaluates.
    /// - **Half a refresh interval wins whenever `ttl ≥ 1.5 × refresh`**,
    ///   which every validated configuration satisfies (`2 × refresh ≤ ttl`).
    /// - **The clamp binds when `ttl < 1.5 × refresh`** (plus the 1 ms): a
    ///   join runs the pass with its lease set and no keeper yet, and the
    ///   keeper's first renewal comes up to one interval after it starts, so
    ///   the pass must finish before `ttl − refresh`. Only a configuration
    ///   built without the lease-budget validation gets here.
    /// - **The 1 ms floor.** `ttl ≤ refresh` leaves no budget that protects
    ///   the lease, and a zero `tokio::time::timeout` is already elapsed, so
    ///   every pass would fail without trying.
    /// - **Saturation.** Knobs near `u64::MAX` must neither panic nor
    ///   overflow.
    #[test]
    fn config_session_recovery_pass_budget_is_half_a_refresh_clamped_below_the_lease() {
        use crate::presence::{
            DEFAULT_PRESENCE_LEASE_TTL_SECS, DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        };

        // Defaults, through `from_env` and the `Config` method.
        temp_env::with_vars(REQUIRED_VARS, || {
            let config =
                Config::from_env_with(PodMemorySource::Reference).expect("defaults must parse");
            assert_eq!(
                (
                    config.presence_refresh_interval_secs,
                    config.presence_lease_ttl_secs
                ),
                (
                    DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
                    DEFAULT_PRESENCE_LEASE_TTL_SECS
                ),
                "precondition: the shipped presence defaults"
            );
            assert_eq!(
                config.session_recovery_pass_budget(),
                Duration::from_millis(7_500),
                "at 15 s / 60 s the pass gets half a refresh interval"
            );
        });
        assert_eq!(
            session_recovery_pass_budget(
                DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
                DEFAULT_PRESENCE_LEASE_TTL_SECS
            ),
            Duration::from_millis(7_500),
            "the free function and the Config method are one expression"
        );

        // Half a refresh interval, while it is under `ttl − refresh − 1 ms`.
        assert_eq!(
            session_recovery_pass_budget(10, 20),
            Duration::from_millis(5_000),
            "10 s / 20 s, the validated floor `2 × refresh = ttl`: half (5 s) is under \
             ttl − refresh − 1 ms (9.999 s) — under validated knobs the clamp never binds"
        );
        assert_eq!(
            session_recovery_pass_budget(10, 16),
            Duration::from_millis(5_000),
            "10 s / 16 s: half (5 s) is still under ttl − refresh − 1 ms (5.999 s)"
        );

        // The clamp: from `ttl = 1.5 × refresh` down, `ttl − refresh − 1 ms` is
        // the smaller term.
        assert_eq!(
            session_recovery_pass_budget(10, 15),
            Duration::from_millis(4_999),
            "10 s / 15 s, ttl = 1.5 × refresh: the clamp (4.999 s) undercuts half (5 s) by its \
             1 ms, so the pass ends strictly before ttl − refresh"
        );
        assert_eq!(
            session_recovery_pass_budget(10, 12),
            Duration::from_millis(1_999),
            "10 s / 12 s: ttl − refresh − 1 ms (1.999 s) is under half (5 s), so the clamp binds \
             and the pass ends before the unrenewed lease can lapse"
        );

        // The 1 ms floor: no lease slack at all.
        for (refresh, ttl) in [(10, 10), (10, 5), (0, 0), (1, 1)] {
            assert_eq!(
                session_recovery_pass_budget(refresh, ttl),
                Duration::from_millis(1),
                "refresh {refresh} s / ttl {ttl} s leaves no slack; the budget floors at 1 ms, \
                 never zero"
            );
        }

        // Saturating arithmetic: no panic, no overflow.
        assert_eq!(
            session_recovery_pass_budget(u64::MAX, u64::MAX),
            Duration::from_millis(1),
            "u64::MAX / u64::MAX saturates both terms to the same value: ttl − refresh is 0, so \
             the floor applies"
        );
        assert_eq!(
            session_recovery_pass_budget(1, u64::MAX),
            Duration::from_millis(500),
            "a ttl whose milliseconds saturate leaves the half-interval term in charge"
        );
    }

    /// The repair wait budget is `⅔ × rpc_deadline_ms`, and it must stay
    /// **strictly below** the deadline at every tuning (b.b2j review round 2).
    ///
    /// Both halves need pinning, and neither is covered by the integration
    /// tests: those construct `CheckpointObjectIdCounterRepair` with hand-picked
    /// budgets so the derivation itself is exercised nowhere.
    ///
    /// **The fraction.** No env var backs this — the number is not
    /// independently meaningful, it is a share of the deadline — so a regression
    /// to a hard-coded constant is invisible to operators until a deployment
    /// tightens or loosens `LIVE_EDIT_RPC_DEADLINE_MS` and the budget silently
    /// does not follow.
    ///
    /// **The ordering.** `CreateLayer` is not in `EXEMPT_FROM_DEADLINE`
    /// (`src/grpc/metrics_layer.rs`), so whichever of the two budgets is shorter
    /// is the one that actually decides the outcome — and they produce very
    /// different outcomes. This budget returns a deliberate `UNAVAILABLE` naming
    /// the pending repair, leaves the repair running on its detached task to
    /// record its verdict, and lets the retry join that attempt. The tower
    /// deadline layer instead DROPS the handler future and yields a bare
    /// `DEADLINE_EXCEEDED` with nothing said about the counter. So `budget <
    /// deadline` is the contract, not a tuning preference, and the cases below
    /// walk it down to a sub-millisecond deadline where a naive
    /// `deadline - 1s` formulation would invert or collapse to zero (which
    /// `tokio::time::timeout` treats as already-elapsed, failing every
    /// `CreateLayer` on a deployment that merely tightened its deadline).
    ///
    /// Asserted through `from_env` so the env plumbing is covered with it.
    #[test]
    fn config_object_id_counter_repair_wait_budget_is_two_thirds_of_the_rpc_deadline() {
        let vars = |deadline: &'static str| {
            vec![
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some(deadline)),
            ]
        };

        // The shipped default: 30 s deadline → 20 s budget.
        temp_env::with_vars(vars("30000"), || {
            assert_eq!(
                Config::from_env_with(PodMemorySource::Reference)
                    .expect("defaults must parse")
                    .object_id_counter_repair_wait_budget(),
                Duration::from_secs(20),
                "at the shipped 30 s deadline the caller waits 20 s. The remaining third is \
                 the rest of CreateLayer — the allocation retry, the MULTI/EXEC that writes \
                 the object, the mapping swap — so a budget that consumed the whole deadline \
                 would hand back a repaired counter with no time left to use it"
            );
        });

        // Scales with the deadline in both directions.
        for (deadline, expect_ms) in [("120000", 80_000u64), ("6000", 4_000), ("1500", 1_000)] {
            temp_env::with_vars(vars(deadline), || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect("must parse");
                assert_eq!(
                    cfg.object_id_counter_repair_wait_budget(),
                    Duration::from_millis(expect_ms),
                    "the budget is a FRACTION of the deadline, not a constant: at \
                     rpc_deadline_ms={deadline} it must be {expect_ms}ms"
                );
            });
        }

        // The ordering invariant, walked down to the degenerate end. A fixed
        // subtraction (`deadline - 1s`) goes negative at 500 ms and a saturating
        // one collapses to zero, which elapses instantly; the fraction cannot
        // cross over, and the `max(1)` floor keeps it non-zero.
        for deadline in ["120000", "30000", "5000", "500", "10", "2"] {
            temp_env::with_vars(vars(deadline), || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect("must parse");
                let budget = cfg.object_id_counter_repair_wait_budget();
                let deadline_dur = Duration::from_millis(cfg.rpc_deadline_ms);
                assert!(
                    budget < deadline_dur,
                    "the wait budget ({budget:?}) MUST be strictly below the RPC deadline \
                     ({deadline_dur:?}) at rpc_deadline_ms={deadline}. At or above it, the \
                     tower deadline layer wins the race and drops the handler future — the \
                     client gets a bare DEADLINE_EXCEEDED, the repair attempt is cancelled \
                     mid-flight, and nothing is recorded about why"
                );
                assert!(
                    !budget.is_zero(),
                    "and it must never be zero: `tokio::time::timeout(ZERO, ..)` is \
                     already-elapsed, so every CreateLayer on a deployment that merely \
                     tightened its deadline would fail without waiting on the repair at all \
                     (rpc_deadline_ms={deadline})"
                );
            });
        }

        // The one deadline where strictness genuinely does not hold, pinned
        // explicitly rather than skipped. At `rpc_deadline_ms = 1` the fraction
        // floors to `0` and the `max(1)` floor lifts it back to `1 ms`, so
        // budget == deadline and which of the two ends the wait is a race.
        //
        // Deliberately left as documentation, not a bug filed against the
        // formula: a one-millisecond RPC deadline fails every `CreateLayer` on
        // the MULTI/EXEC alone, long before the counter repair is reachable, so
        // there is no configuration in which this boundary is the operative
        // problem. `Config::from_env` applies no lower bound to
        // `LIVE_EDIT_RPC_DEADLINE_MS` (only `0` would be a parse of the
        // default), which is why the case is reachable at all — if a minimum is
        // ever introduced, this assertion is where the interaction surfaces.
        temp_env::with_vars(vars("1"), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference).expect("must parse");
            assert_eq!(
                cfg.object_id_counter_repair_wait_budget(),
                Duration::from_millis(1),
                "at rpc_deadline_ms=1 the ⅔ fraction floors to 0 and the `max(1)` floor \
                 lifts it back to 1 ms — equal to the deadline, the single input where the \
                 strict-ordering invariant above degenerates. Pinned so a future lower bound \
                 on LIVE_EDIT_RPC_DEADLINE_MS, or a change to the floor, has to acknowledge \
                 this corner rather than silently redefine it"
            );
        });
    }

    // ---- b.715: upload admission settings ----

    /// The environment for an upload-admission case: the required endpoint,
    /// the RPC deadline and the two upload settings, with the size cap the
    /// budget's floor derives from and the removed budget setting (refused
    /// when set; the W6 table tests it) pinned unset — `None` pins a variable
    /// unset, so an ambient copy cannot satisfy or break a default.
    fn upload_admission_env<'a>(
        deadline_ms: Option<&'a str>,
        wait_ms: Option<&'a str>,
        max_waiters: Option<&'a str>,
    ) -> Vec<(&'static str, Option<&'a str>)> {
        vec![
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://auth.example.com"),
            ),
            ("LIVE_EDIT_RPC_DEADLINE_MS", deadline_ms),
            ("LIVE_EDIT_UPLOAD_MEMORY_BUDGET_MIB", None),
            ("LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS", wait_ms),
            ("LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS", max_waiters),
            ("LIVE_EDIT_MAX_LAYER_SIZE", None),
        ]
    }

    /// The environment for a budget case: the size cap (`None` = the 100 MiB
    /// default), every other upload setting and the removed budget setting
    /// unset.
    fn upload_budget_env(max_layer_size: Option<&str>) -> Vec<(&'static str, Option<&str>)> {
        vec![
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://auth.example.com"),
            ),
            ("LIVE_EDIT_RPC_DEADLINE_MS", None),
            ("LIVE_EDIT_UPLOAD_MEMORY_BUDGET_MIB", None),
            ("LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS", None),
            ("LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS", None),
            ("LIVE_EDIT_MAX_LAYER_SIZE", max_layer_size),
        ]
    }

    #[test]
    fn config_upload_admission_defaults_match_the_gate_defaults() {
        use crate::grpc::upload_admission::{
            UploadAdmissionConfig, UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            UPLOAD_ADMISSION_WAIT_MS_DEFAULT, UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
        };
        temp_env::with_vars(upload_admission_env(None, None, None), || {
            let cfg =
                Config::from_env_with(PodMemorySource::Reference).expect("defaults must parse");
            assert_eq!(
                cfg.upload_memory_budget_mib,
                UPLOAD_MEMORY_BUDGET_MIB_DEFAULT
            );
            assert_eq!(cfg.upload_memory_budget_mib, 384);
            assert_eq!(
                cfg.upload_admission_wait_ms,
                UPLOAD_ADMISSION_WAIT_MS_DEFAULT
            );
            assert_eq!(cfg.upload_admission_wait_ms, 2_000);
            assert_eq!(
                cfg.upload_admission_max_waiters,
                UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT
            );
            assert_eq!(cfg.upload_admission_max_waiters, 64);
            assert_eq!(
                cfg.upload_admission_config(),
                UploadAdmissionConfig::default(),
                "at the default 30 s deadline the configured gate is the gate's own default"
            );
            assert_eq!(cfg.upload_memory_budget_bytes(), 384 * 1024 * 1024);
        });
    }

    /// Unset, the admission wait is DERIVED from the RPC deadline —
    /// `min(2000, deadline / 2)` — so it follows a tightened or loosened
    /// deadline without the operator setting it, at several deadlines.
    #[test]
    fn config_upload_admission_wait_unset_is_derived_from_the_rpc_deadline() {
        for (deadline, want) in [
            ("120000", 2_000u64),
            ("30000", 2_000),
            ("4000", 2_000),
            ("3000", 1_500),
            ("1000", 500),
            ("10", 5),
            ("1", 0),
        ] {
            temp_env::with_vars(upload_admission_env(Some(deadline), None, None), || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("an unset wait never fails validation");
                assert_eq!(
                    cfg.upload_admission_wait_ms, want,
                    "unset LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS at rpc_deadline_ms={deadline}"
                );
            });
        }
    }

    /// The deadline sweep of
    /// [`config_object_id_counter_repair_wait_budget_is_two_thirds_of_the_rpc_deadline`],
    /// for the unset upload admission wait: at every swept deadline it parses
    /// and stays strictly below the deadline, so the RPC deadline never ends
    /// a capacity wait before the wait's own refusal can answer.
    #[test]
    fn config_upload_admission_wait_unset_stays_below_the_rpc_deadline_at_every_deadline() {
        for deadline in [
            "120000", "30000", "6000", "5000", "1500", "500", "10", "2", "1",
        ] {
            temp_env::with_vars(upload_admission_env(Some(deadline), None, None), || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect("must parse");
                assert!(
                    cfg.upload_admission_wait_ms < cfg.rpc_deadline_ms,
                    "the derived wait ({} ms) must stay below rpc_deadline_ms={deadline}",
                    cfg.upload_admission_wait_ms
                );
            });
        }
    }

    /// An explicit wait at or above the RPC deadline is refused at startup,
    /// naming both settings; one below it is taken as given.
    #[test]
    fn config_upload_admission_wait_explicit_at_or_above_the_deadline_is_refused_naming_both() {
        for wait in ["5000", "6000"] {
            temp_env::with_vars(upload_admission_env(Some("5000"), Some(wait), None), || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("a wait the deadline always ends first must be refused");
                assert!(
                    err.contains("LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS")
                        && err.contains("LIVE_EDIT_RPC_DEADLINE_MS"),
                    "the refusal must name both settings; got: {err}"
                );
            });
        }
        temp_env::with_vars(
            upload_admission_env(Some("5000"), Some("4999"), None),
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("a wait below the deadline is valid");
                assert_eq!(cfg.upload_admission_wait_ms, 4_999);
            },
        );
    }

    /// An explicit 0 — never wait, refuse at once when capacity is full — is
    /// accepted, and reaches the gate as a zero wait.
    #[test]
    fn config_upload_admission_wait_explicit_zero_is_accepted() {
        temp_env::with_vars(upload_admission_env(None, Some("0"), None), || {
            let cfg =
                Config::from_env_with(PodMemorySource::Reference).expect("0 means never wait");
            assert_eq!(cfg.upload_admission_wait_ms, 0);
            assert!(cfg.upload_admission_config().wait.is_zero());
        });
    }

    /// At the reference pod memory limit the derived budget is
    /// `max(384, ⌈floor ÷ 1 MiB⌉)` — 384 at the default 100 MiB cap and
    /// below it, larger once the cap's floor passes 384 MiB — so raising the
    /// size cap alone never leaves a budget below its floor.
    #[test]
    fn config_upload_memory_budget_unset_is_derived_from_the_size_cap() {
        use crate::grpc::upload_admission::upload_budget_floor_bytes;
        const MIB: u64 = 1024 * 1024;
        for (cap_env, want_mib) in [
            (None, 384u64),
            (Some("10485760"), 384),
            // 300 MiB: floor = 8 MiB + 90,200 B + 600 MiB = 608.09 MiB → 609.
            (Some("314572800"), 609),
        ] {
            temp_env::with_vars(upload_budget_env(cap_env), || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("the plan closes at the reference limit with this size cap");
                assert_eq!(
                    cfg.upload_memory_budget_mib, want_mib,
                    "the derived upload budget at LIVE_EDIT_MAX_LAYER_SIZE={cap_env:?}"
                );
                let cap = u64::try_from(cfg.max_layer_size).expect("fits");
                assert!(cfg.upload_memory_budget_mib * MIB >= upload_budget_floor_bytes(cap));
                assert_eq!(cfg.upload_admission_config().max_upload_bytes, cap);
            });
        }
    }

    /// A waiter cap of 0 is refused at startup; fail-fast is a 0 wait.
    #[test]
    fn config_upload_admission_max_waiters_zero_is_refused() {
        temp_env::with_vars(upload_admission_env(None, None, Some("0")), || {
            let err = Config::from_env_with(PodMemorySource::Reference)
                .expect_err("a zero waiter cap must be refused");
            assert!(
                err.contains("LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS"),
                "got: {err}"
            );
        });
        temp_env::with_vars(upload_admission_env(None, None, Some("1")), || {
            let cfg =
                Config::from_env_with(PodMemorySource::Reference).expect("one waiter is valid");
            assert_eq!(cfg.upload_admission_max_waiters, 1);
        });
    }

    /// A non-numeric value is refused, naming the variable.
    #[test]
    fn config_upload_admission_non_numeric_values_are_refused() {
        for (wait, waiters, var) in [
            (Some("soon"), None, "LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS"),
            (None, Some("many"), "LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS"),
        ] {
            temp_env::with_vars(upload_admission_env(None, wait, waiters), || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("a non-numeric value must be refused");
                assert!(err.contains(var), "got: {err}");
            });
        }
    }

    /// `object_id_counter_repair_wait_budget_within` narrows the fixed budget
    /// to what is actually left of *this* request's deadline (b.hka (f)) — and
    /// collapses to the fixed budget when there is no deadline to narrow to.
    ///
    /// Four cases, one per branch of the formula, because each answers a
    /// different question about the refinement:
    ///
    /// 1. **`None` is exactly the fixed budget.** This is the compatibility
    ///    claim the whole design rests on: streaming RPCs, background tasks and
    ///    direct unit-test calls have no ambient deadline, and the wiring in
    ///    `main.rs` swaps the fixed budget for this method unconditionally. If
    ///    the two disagreed, that swap would silently change behaviour on every
    ///    non-RPC caller.
    /// 2. **Plenty of time left → still the fixed budget.** The narrowing is a
    ///    `min`, not a replacement; a request that has barely started must not
    ///    be given a *longer* wait than the configured share of the deadline
    ///    just because `remaining` happens to exceed it.
    /// 3. **Little time left → ⅔ of what is left.** The case the whole change
    ///    exists for. Allocation is the terminal step of `handle_create_layer`,
    ///    so by the time it runs the request has already spent auth, the
    ///    upload admission wait, the content read and ext-value storage; the
    ///    fixed budget would
    ///    outlast the remainder and the tower layer — not this budget — would
    ///    end the wait, producing a bare `DEADLINE_EXCEEDED` with no
    ///    `wait_timeout` tick and nothing said about the pending repair.
    /// 4. **Sub-millisecond remaining → the 1 ms floor.** `tokio::time::timeout`
    ///    treats a zero duration as already elapsed, so a truncating fraction of
    ///    a nearly-spent deadline would turn every `CreateLayer` into an instant
    ///    give-up that never joined the repair at all.
    #[test]
    fn config_object_id_counter_repair_wait_budget_within_narrows_to_the_remaining_deadline() {
        temp_env::with_vars(
            vec![
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some("30000")),
            ],
            || {
                let cfg =
                    Config::from_env_with(PodMemorySource::Reference).expect("defaults must parse");

                // 1 — no deadline in scope.
                assert_eq!(
                    cfg.object_id_counter_repair_wait_budget_within(None),
                    cfg.object_id_counter_repair_wait_budget(),
                    "`None` MUST collapse to exactly the fixed budget. `main.rs` installs the \
                     deadline-aware budget unconditionally, so any divergence here silently \
                     changes the wait for every caller with no ambient deadline — streaming \
                     RPCs, background tasks, direct calls"
                );

                // 2 — more remaining than the deadline itself: the `min` holds.
                assert_eq!(
                    cfg.object_id_counter_repair_wait_budget_within(Some(Duration::from_secs(60))),
                    Duration::from_secs(20),
                    "with 60 s left against a 30 s deadline the budget MUST stay at the \
                     configured ⅔ × 30 s. This is a narrowing, never a widening — a request may \
                     not buy itself a longer wait than the deadline funds"
                );

                // 3 — the case the refinement exists for.
                assert_eq!(
                    cfg.object_id_counter_repair_wait_budget_within(Some(Duration::from_millis(
                        900
                    ))),
                    Duration::from_millis(600),
                    "with 900 ms left the budget MUST be ⅔ × 900 ms, not ⅔ × 30 s. The fixed \
                     budget would outlast the remainder, so the tower deadline layer would end \
                     the wait by DROPPING the handler future: a bare DEADLINE_EXCEEDED, no \
                     `wait_timeout` tick, and nothing said about the repair still running"
                );

                // 4 — the floor.
                for remaining in [
                    Duration::ZERO,
                    Duration::from_micros(1),
                    Duration::from_micros(999),
                ] {
                    assert_eq!(
                        cfg.object_id_counter_repair_wait_budget_within(Some(remaining)),
                        Duration::from_millis(1),
                        "a sub-millisecond remainder ({remaining:?}) MUST floor at 1 ms — \
                         `tokio::time::timeout(ZERO, ..)` is already-elapsed, which would fail \
                         the CreateLayer without ever joining the in-flight repair"
                    );
                }
            },
        );
    }

    /// The bound is **adapter-dependent**, and StorageApi is where that bites.
    ///
    /// Its publish bound is `request_timeout × MAX_RETRY_ATTEMPTS ×
    /// AUTH_ATTEMPTS` = 30 s × 3 × 2 = 180 s at the defaults — since b.gg7 an
    /// enforced `tokio::time::timeout` in `send_classified`, not merely a
    /// description of the retry ladder — and the grace is
    /// `RECONCILE_GRACE_MULTIPLE` on top of that — 540 s, nine times the local
    /// floor — so an interval that is perfectly legal on LocalDisk is rejected
    /// here. `Config::checkpoint_reconcile_grace` composes it from the same
    /// arithmetic `StorageApiCheckpointStore::publish_worst_case` reports at
    /// runtime, which is what keeps config-time validation and runtime
    /// behaviour from drifting.
    #[test]
    fn config_from_env_checkpoint_reconcile_grace_tracks_the_storage_api_request_timeout() {
        let storage_api_vars = |interval: &'static str, timeout: Option<&'static str>| {
            vec![
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("StorageApi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/checkpoints")),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", timeout),
                (
                    "LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS",
                    Some(interval),
                ),
            ]
        };

        // 120 s clears the 60 s local floor but not StorageApi's 540 s grace.
        temp_env::with_vars(storage_api_vars("120", None), || {
            let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                "120 s is below the StorageApi grace (3 × (30 s × 3 retries × 2 auth attempts)) \
                 and must be rejected even though it would be legal on a local adapter",
            );
            assert!(
                err.contains("540"),
                "error must state the adapter-derived bound, got: {err}"
            );
        });

        temp_env::with_vars(storage_api_vars("540", None), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("540 s is exactly the derived grace");
            assert_eq!(
                cfg.checkpoint_reconcile_grace(),
                Duration::from_secs(540),
                "the grace must be live arithmetic over the configured request timeout, not a \
                 frozen constant — an operator who raises the timeout raises this bound with it"
            );
        });

        // Halve the request timeout → the grace halves with it, and an
        // interval the default timeout would have rejected becomes legal.
        temp_env::with_vars(storage_api_vars("270", Some("15000")), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("halving the request timeout halves the derived grace to 270 s");
            assert_eq!(cfg.checkpoint_reconcile_grace(), Duration::from_secs(270));
        });
    }

    // ---- t3.fgj.zc.nx.gc: ext-mirror env-var validation gatekeepers ----
    //
    // Pin Acceptance Criterion #1 of the parent Task t2.fgj.zc.nx behind
    // dedicated filterable names so `cargo test -p live-edit
    // config_from_env_ext_mirror` selects exactly these four. The
    // `config_from_env_checkpoint_ext_mirror_*` tests added in nx.p7
    // assert stronger prose ("silently drop", "panics"); these four keep
    // assertions to the var-name substring so the gatekeeper survives
    // future error-wording cleanups.

    #[test]
    fn config_from_env_ext_mirror_workers_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("ext_mirror_workers=0 must be rejected at startup");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_ext_mirror_queue_capacity_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("ext_mirror_queue_capacity=0 must be rejected at startup");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_ext_mirror_defaults_applied_when_unset() {
        // Both vars left unset; Config must surface the documented
        // defaults (2 workers, 1000-deep queue) per SDD SR-3.1.
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("required vars set; defaults must succeed");
            assert_eq!(
                cfg.checkpoint_ext_mirror_workers, 2,
                "ext_mirror_workers default"
            );
            assert_eq!(
                cfg.checkpoint_ext_mirror_queue_capacity, 1000,
                "ext_mirror_queue_capacity default"
            );
        });
    }

    /// T12 (e) (b.mhq, PD3): up to 16 mirror workers are accepted and 17 is
    /// refused at startup, naming the setting and the maximum: each worker's
    /// connection to the backup store holds memory outside the mirror's
    /// byte budget, so the count is capped.
    #[test]
    fn config_from_env_ext_mirror_workers_accepts_16_and_refuses_17() {
        assert_eq!(crate::checkpoint::ext_mirror::EXT_MIRROR_MAX_WORKERS, 16);
        let parse = |workers: &'static str| {
            let mut env = REQUIRED_VARS.to_vec();
            env.push(("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS", Some(workers)));
            temp_env::with_vars(env, || Config::from_env_with(PodMemorySource::Reference))
        };
        assert_eq!(
            parse("16")
                .expect("16 workers are accepted")
                .checkpoint_ext_mirror_workers,
            16
        );
        let err = parse("17").expect_err("17 workers are refused");
        for part in [
            "invalid value for LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS",
            "17 is above the maximum of 16",
        ] {
            assert!(err.contains(part), "{part:?} in {err}");
        }
    }

    #[test]
    fn config_from_env_ext_mirror_valid_positive_values_accepted() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS", Some("8")),
                (
                    "LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY",
                    Some("4096"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("valid positive values must be accepted");
                assert_eq!(cfg.checkpoint_ext_mirror_workers, 8);
                assert_eq!(cfg.checkpoint_ext_mirror_queue_capacity, 4096);
            },
        );
    }

    // Epic-1 forward-pointer rejection tests removed in t3.fgj.jd.o6.5p;
    // the StorageApi accept-path + new reject-path coverage is owned by the
    // sibling Test Writer Subtasks t3.fgj.jd.o6.{qf,vh,sz,d5,1w}.

    #[test]
    fn config_from_env_checkpoint_localdisk_requires_path() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                ("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("localdisk adapter must reject missing path");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_localdisk_rejects_whitespace_path() {
        // A path that trims to empty is treated as missing.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                ("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH", Some("   ")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("localdisk adapter must reject whitespace-only path");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_localdisk_accepts_explicit_path() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                (
                    "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                    Some("/var/lib/live-edit/checkpoints"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("localdisk + explicit path should succeed");
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::LocalDisk);
                assert_eq!(
                    cfg.checkpoint_local_disk_path.as_deref(),
                    Some("/var/lib/live-edit/checkpoints")
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tick_ge_interval_rejected() {
        // Cross-validation: tick must be strictly less than interval.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TICK_SECS", Some("60")),
                ("LIVE_EDIT_CHECKPOINT_INTERVAL_SECS", Some("60")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("tick == interval must be rejected by cross-validation");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TICK_SECS")
                        && err.contains("LIVE_EDIT_CHECKPOINT_INTERVAL_SECS"),
                    "error must name both vars, got: {err}"
                );
            },
        );
    }

    // ---- t3.fgj.ya.99.b8: cross-validation tests for checkpoint interval/tick/quiesce ----
    //
    // The foundation Task lands these cross-validation rules:
    //   1. `TICK_SECS > 0 && TICK_SECS < INTERVAL_SECS`
    //   2. `QUIESCE_SECS > 0`
    //   3. `RETAIN_GENERATIONS >= 1`
    //   4. LocalDisk adapter requires a non-empty `LOCAL_DISK_PATH`
    //   5. Selecting `StorageApi` requires both
    //      `LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT` (with an explicit
    //      `https://` / `http://` scheme) and `LIVE_EDIT_CHECKPOINT_BASE_PATH`
    //      (non-empty after trim) at startup; missing or bare-hostname values
    //      are rejected. (Updated in t3.fgj.jd.o6.5p — the prior Epic-1
    //      forward-pointer rejection was replaced when Epic t1.fgj.jd landed.)
    //
    // The loop Task and adapters Task downstream assume these hold by the
    // time they construct themselves. If the validator silently accepts an
    // out-of-band combination, the loop's interval/tick scheduling math
    // breaks and the adapter constructor receives a `None` path it cannot
    // recover from. Mirrors the presence-lease cross-validation pattern.
    // The "rejection-path test asserts the env-var name appears in the
    // error string" pattern matches `presence_lease_ttl_zero_rejected`.

    #[test]
    fn config_from_env_checkpoint_tick_gt_interval_rejected() {
        // Strict-`<` rule: a tick strictly greater than interval must
        // also be rejected. Complements the `tick == interval` test
        // above by pinning the other side of the boundary so a future
        // refactor that flips the inequality to `>` (instead of `>=`)
        // is caught here.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TICK_SECS", Some("90")),
                ("LIVE_EDIT_CHECKPOINT_INTERVAL_SECS", Some("60")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("tick > interval must be rejected by cross-validation");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TICK_SECS")
                        && err.contains("LIVE_EDIT_CHECKPOINT_INTERVAL_SECS"),
                    "error must name both env vars so the operator sees \
                     which knob to adjust, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tick_just_inside_interval_accepted() {
        // Boundary just inside the accept band: tick = interval - 1.
        // The rule is strict `<`, so this value must pass and the
        // explicit values must round-trip without silent clamping.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TICK_SECS", Some("59")),
                ("LIVE_EDIT_CHECKPOINT_INTERVAL_SECS", Some("60")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("tick = interval - 1 must be accepted (strict `<`)");
                assert_eq!(
                    cfg.checkpoint_tick_secs, 59,
                    "tick must round-trip the explicit boundary value"
                );
                assert_eq!(
                    cfg.checkpoint_interval_secs, 60,
                    "interval must round-trip the explicit boundary value"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tick_interval_defaults_accepted() {
        // Canonical sanity case: documented defaults (tick=10,
        // interval=60) must pass cross-validation. If the validator's
        // inequality direction is ever inverted, this test fires on
        // the most common operator configuration before any
        // explicitly-set value ever reaches production.
        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("documented defaults (tick=10, interval=60) must satisfy cross-validation");
            assert_eq!(cfg.checkpoint_tick_secs, 10);
            assert_eq!(cfg.checkpoint_interval_secs, 60);
            assert!(
                cfg.checkpoint_tick_secs < cfg.checkpoint_interval_secs,
                "documented defaults must satisfy the strict `<` cross-validation rule"
            );
        });
    }

    #[test]
    fn config_from_env_checkpoint_quiesce_explicit_value_accepted() {
        // The quiesce-zero rejection is pinned above. This test
        // complements that by confirming a small reasonable value
        // (5s) is accepted and round-trips into the Config field —
        // i.e. the rejection isn't accidentally over-broad.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_QUIESCE_SECS", Some("5")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit non-zero quiesce must be accepted");
                assert_eq!(cfg.checkpoint_quiesce_secs, 5);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_retain_one_boundary_accepted() {
        // `RETAIN_GENERATIONS >= 1` — the boundary case is `1`, which
        // keeps exactly the most recent checkpoint and prunes
        // everything older. Pin acceptance so a future refactor that
        // raises the floor to 2 (or any other value) is a deliberate
        // choice, not a silent regression that surprises operators
        // who explicitly set this knob.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS", Some("1")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("retain=1 (boundary) must be accepted");
                assert_eq!(cfg.checkpoint_retain_generations, 1);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_retain_explicit_override_accepted() {
        // A typical operator override (5 generations) must round-trip.
        // Note: `config_from_env_checkpoint_retain_generations_explicit_override_round_trips`
        // covers the same value in the `.eu` cluster; this test pins
        // the same contract from the cross-validation lens so the
        // `.b8` cluster's coverage of "RETAIN_GENERATIONS accept band"
        // is self-contained and grep-able.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS", Some("5")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("retain=5 (operator override) must be accepted");
                assert_eq!(cfg.checkpoint_retain_generations, 5);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_localdisk_missing_path_error_names_both_vars() {
        // The LocalDisk-without-path rejection message must name both
        // `LIVE_EDIT_CHECKPOINT_ADAPTER` and
        // `LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH`, so the operator
        // sees both knobs in play. The existing
        // `_localdisk_requires_path` test only asserts on the path
        // var; this test pins the second half of the contract.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                ("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("localdisk + missing path must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH"),
                    "error must name the path var, got: {err}"
                );
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_ADAPTER"),
                    "error must name the adapter var so the operator sees \
                     why the path is required, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_localdisk_empty_string_path_rejected() {
        // An empty-string path (verbatim `""`, not whitespace) must
        // be rejected the same as a missing variable. The
        // `_rejects_whitespace_path` test above pins the
        // "trims-to-empty" case; this test pins the bare
        // empty-string case so a future change that special-cases
        // empty before the trim is caught.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                ("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH", Some("")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("localdisk + empty-string path must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_localdisk_with_path_round_trips() {
        // Positive case: LocalDisk + non-empty path must succeed and
        // round-trip the path into the Config field. Pins the
        // canonical example value called out by the Subtask body
        // (`/tmp/checkpoints`) so a regression to the path-storage
        // type or trim behavior fires here. Sibling
        // `_localdisk_accepts_explicit_path` covers a different
        // path; both targets are the same accept-branch.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                (
                    "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                    Some("/tmp/checkpoints"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("localdisk + non-empty path must be accepted");
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::LocalDisk);
                assert_eq!(
                    cfg.checkpoint_local_disk_path.as_deref(),
                    Some("/tmp/checkpoints"),
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_local_disk_path_ignored_for_disabled_adapter() {
        // The path env var is only consulted by the LocalDisk
        // adapter. Setting it alongside `disabled` (the default
        // adapter) must be accepted as a no-op rather than surfacing
        // an "unused variable" error — operators routinely leave
        // environment scaffolding in place across adapter toggles,
        // and a startup failure here would force them to scrub the
        // manifest for every cohabiting variable.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("disabled")),
                (
                    "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                    Some("/tmp/checkpoints"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("disabled + path is the cohabiting case; foundation must accept");
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::Disabled);
                // The value is captured into the field but ignored
                // by downstream adapter construction. Pinning the
                // captured value (rather than asserting it's `None`)
                // documents the actual contract: the parser doesn't
                // null out unused values, the adapter just doesn't
                // read them.
                assert_eq!(
                    cfg.checkpoint_local_disk_path.as_deref(),
                    Some("/tmp/checkpoints"),
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_local_disk_path_ignored_for_inmemory_adapter() {
        // Same rationale as the `disabled` cohabiting-case test:
        // a path env var alongside `inmemory` must not error. Pins
        // the contract for the second non-LocalDisk adapter so a
        // future change that ties the path-required guard to
        // anything other than the `LocalDisk` variant fires here.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("inmemory")),
                (
                    "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                    Some("/tmp/checkpoints"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("inmemory + path is the cohabiting case; foundation must accept");
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::InMemory);
            },
        );
    }

    // The companion "rejected_when_enabled_true" test was also removed
    // in t3.fgj.jd.o6.5p; the StorageApi accept-path under both ENABLED
    // flag values is covered by sibling Test Writer Subtasks
    // t3.fgj.jd.o6.{qf,vh,sz,d5,1w}.

    #[test]
    fn config_from_env_checkpoint_storage_api_with_endpoint_and_base_path_round_trips() {
        // Positive accept-path for the StorageApi adapter (Task
        // `t2.fgj.jd.o6` / Epic `t1.fgj.jd`). Mirrors
        // `_localdisk_with_path_round_trips` (the canonical sibling
        // positive-path pattern for the LocalDisk adapter): pin the
        // adapter variant AND every field the parser populates so a
        // future regression to `Option<String>` vs `String` on the
        // endpoint field, or a flip of the `PathBuf` type on the
        // base-path field, fires here. The two timeout fields are
        // populated unconditionally (silent-tolerance contract); pin
        // their SDD-default values (5000 / 30000) so a future default
        // drift is caught without the operator having to set the
        // env vars.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/checkpoints")),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("storageapi + endpoint + base path must be accepted");
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::StorageApi);
                // Endpoint round-trips as `Option<String>` — pin the
                // exact value AND the `Some(...)` shape so a refactor
                // that flips the field to `String` (and stores `""`
                // when unset) fails this test.
                assert_eq!(
                    cfg.checkpoint_storage_api_endpoint,
                    Some("https://storage.example.com".to_string()),
                    "endpoint must round-trip as Option<String> with the exact value",
                );
                // Base path round-trips as `Option<PathBuf>` — pin
                // the exact value AND the `Some(PathBuf::from(...))`
                // shape so a refactor that flips the field to
                // `String` or `PathBuf` (non-Option) fails this test.
                assert_eq!(
                    cfg.checkpoint_base_path,
                    Some(PathBuf::from("/checkpoints")),
                    "base path must round-trip as Option<PathBuf> with the exact value",
                );
                // Default-value behavior for both timeouts: unset env
                // vars → SDD-pinned defaults (5000 / 30000 ms). Both
                // fields are populated unconditionally per the
                // silent-tolerance contract, so the defaults fire
                // even though no `LIVE_EDIT_CHECKPOINT_*_TIMEOUT_MS`
                // var is set in this case.
                assert_eq!(
                    cfg.checkpoint_connect_timeout_ms, 5000,
                    "checkpoint_connect_timeout_ms default must match the SDD-pinned 5000 ms",
                );
                assert_eq!(
                    cfg.checkpoint_request_timeout_ms, 30000,
                    "checkpoint_request_timeout_ms default must match the SDD-pinned 30000 ms",
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_case_insensitive_adapter_value_accepted() {
        // The `CheckpointAdapter::from_str` impl lower-cases its input
        // and matches against `"storageapi"`, so the supported case-
        // insensitive variants are `storageapi`, `STORAGEAPI`, and
        // `StorageApi` (per the Engineer's Subtask body t3.fgj.jd.o6.5p
        // acceptance criterion). The Task body's prose uses
        // `storage_api` with an underscore as shorthand; the underscore
        // is NOT a supported variant of the env-var value — only the
        // contiguous `storageapi` spelling parses (see
        // `checkpoint/store.rs::FromStr for CheckpointAdapter`). This
        // test pins the case-insensitive contract end-to-end at the
        // `Config::from_env` level (the FromStr-level case-insensitivity
        // is already pinned by
        // `checkpoint_adapter_from_str_accepts_uppercase_storageapi`).
        // Use the uppercase `STORAGEAPI` variant — a casing the
        // `to_ascii_lowercase()` pipeline must collapse to the
        // canonical lowercase arm.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("STORAGEAPI")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/checkpoints")),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "uppercase STORAGEAPI must parse to CheckpointAdapter::StorageApi \
                     via case-insensitive matching",
                );
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::StorageApi);
                // Re-pin the endpoint + base-path values so the test
                // also documents that case-insensitive adapter parsing
                // does not interfere with dependent-var resolution.
                assert_eq!(
                    cfg.checkpoint_storage_api_endpoint,
                    Some("https://storage.example.com".to_string()),
                );
                assert_eq!(
                    cfg.checkpoint_base_path,
                    Some(PathBuf::from("/checkpoints")),
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_explicit_timeouts_round_trip() {
        // Both new timeout env vars must round-trip into the Config
        // fields when set to non-default values on top of the
        // storage-api accept-path. Mirrors
        // `_checkpoint_quiesce_explicit_value_accepted` — pins the
        // accept-band so a future refactor that accidentally clamps,
        // ignores, or drops the explicit value (e.g., by re-using the
        // default in `optional_parsed`'s wrong branch) fires here.
        // Values chosen are deliberately non-multiples of the defaults
        // (7500 vs 5000, 45000 vs 30000) so a "return default on
        // success" regression cannot accidentally match. Complements
        // the sibling boundary tests
        // (`_explicit_connect_timeout_accepted` and
        // `_explicit_request_timeout_accepted` pin `1` ms each); this
        // test pins the typical operator-override path where both
        // timeouts are tuned together.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("/checkpoints")),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
                ("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS", Some("7500")),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("45000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "explicit non-zero timeouts on the storageapi accept-path must be accepted",
                );
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::StorageApi);
                assert_eq!(
                    cfg.checkpoint_connect_timeout_ms, 7500,
                    "explicit LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS must round-trip",
                );
                assert_eq!(
                    cfg.checkpoint_request_timeout_ms, 45000,
                    "explicit LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS must round-trip",
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_missing_base_path_rejected() {
        // StorageApi adapter selected, endpoint set to a valid URL, but
        // LIVE_EDIT_CHECKPOINT_BASE_PATH is unset. The validator must
        // reject and the error must name BOTH knobs so an operator
        // sees the dependency relationship (mirrors the
        // `_localdisk_missing_path_error_names_both_vars` precedent).
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("storageapi + missing base path must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_BASE_PATH"),
                    "error must name the base-path var, got: {err}"
                );
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_ADAPTER"),
                    "error must name the adapter var so the operator sees \
                     why the base path is required, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_empty_base_path_rejected() {
        // An empty-string base path (verbatim `""`, not whitespace)
        // must be rejected the same as a missing variable. Mirrors
        // `_localdisk_empty_string_path_rejected` — guards against a
        // future change that special-cases empty before the trim.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("storageapi + empty-string base path must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_BASE_PATH"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_whitespace_base_path_rejected() {
        // A base path that trims to empty is treated as missing.
        // Mirrors `_localdisk_rejects_whitespace_path` — guards the
        // trim+filter pattern against a regression that would let
        // whitespace-only values slip through as a non-None Path.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("   ")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("storageapi adapter must reject whitespace-only base path");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_BASE_PATH"),
                    "error must name the offending var, got: {err}"
                );
            },
        );
    }

    // ---- t3.fgj.jd.o6.d5: zero CONNECT and REQUEST timeouts ----
    //
    // The engineer's Subtask t3.fgj.jd.o6.5p introduced the canonical
    // `optional_parsed` + inline `n == 0` zero-rejection idiom for both
    // `LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS` (default 5000) and
    // `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS` (default 30000). Mirrors
    // the existing CHECKPOINT_TICK_SECS / QUIESCE_SECS / RETAIN_GENERATIONS
    // zero-rejection contract. Zero-rejection fires *unconditionally* at
    // parse time — independent of which adapter is selected — but the
    // tests below isolate the failure mode to the timeout check by
    // setting a valid StorageApi adapter + endpoint + base-path so
    // unrelated dependent-var guards stay quiet.
    //
    // Pinning the lower-boundary accept-path at value `1` complements
    // the reject tests so a future off-by-one (e.g. flipping the
    // inequality to `<= 0` covering `1`, or `> 0` excluding `1`) is
    // caught by the accept-path test rather than silently degrading
    // operator-supplied values.

    #[test]
    fn config_from_env_checkpoint_storage_api_zero_connect_timeout_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH",
                    Some("/projects/my-project"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
                ("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("connect_timeout_ms=0 must be rejected at parse time");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS"),
                    "error must name the offending env var so the operator \
                     sees which knob to adjust, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_zero_request_timeout_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH",
                    Some("/projects/my-project"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("request_timeout_ms=0 must be rejected at parse time");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS"),
                    "error must name the offending env var so the operator \
                     sees which knob to adjust, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_explicit_connect_timeout_accepted() {
        // Lower-boundary accept-path: `1` ms is the smallest non-zero
        // value the validator must let through. If the rejection
        // inequality is ever flipped from `n == 0` to `n <= 0` (covering
        // signed reinterpretation) or to `n < 1` written as a different
        // expression, a future off-by-one regression that excludes `1`
        // fires here. The round-trip assertion also pins that the
        // explicit value is captured into the Config field — no silent
        // clamp-up to the default `5000` ms.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH",
                    Some("/projects/my-project"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
                ("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS", Some("1")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "connect_timeout_ms=1 (smallest non-zero) must be accepted; \
                     a regression that rejects `1` would silently fail every \
                     StorageApi rollout that sets aggressive timeouts",
                );
                assert_eq!(
                    cfg.checkpoint_connect_timeout_ms, 1,
                    "explicit boundary value must round-trip into the Config \
                     field without silent default-substitution or clamping",
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_explicit_request_timeout_accepted() {
        // Symmetric to the connect-timeout accept-path test above:
        // boundary value `1` ms must be accepted and round-trip into
        // `checkpoint_request_timeout_ms`. Pins that the two
        // independent zero-rejection sites have independently-correct
        // accept paths — a refactor that consolidates the two
        // validations into a single helper can't accidentally narrow
        // one accept band without firing here.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("https://storage.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH",
                    Some("/projects/my-project"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("1")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("request_timeout_ms=1 (smallest non-zero) must be accepted");
                assert_eq!(
                    cfg.checkpoint_request_timeout_ms, 1,
                    "explicit boundary value must round-trip into the Config \
                     field without silent default-substitution or clamping",
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_disabled_with_storage_api_vars_accepted() {
        // Silent-tolerance contract (Task `t2.fgj.jd.o6`): the four
        // new StorageApi env vars are parsed unconditionally so the
        // values populate on `Config` for any adapter, but the
        // endpoint+base-path validations are adapter-scoped (only
        // fire under `StorageApi`). With adapter=disabled, setting
        // a bare-hostname endpoint AND an empty-string base path
        // (both shapes that WOULD fail SR-4.5/SR-4.6 under
        // StorageApi) must be accepted — the parser keeps the
        // populated values but the runtime never consumes them. The
        // timeouts are held at their non-zero defaults because the
        // zero-check is unconditional (per the Engineer's §2
        // split), and a zero here would mask the adapter-scoped
        // assertion under test.
        //
        // Mirrors the `_local_disk_path_ignored_for_disabled_adapter`
        // precedent, extended to the new four-var cluster.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("disabled")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("")),
                ("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS", Some("5000")),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("30000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "disabled adapter must silently tolerate bare-hostname endpoint + empty \
                     base path — endpoint/base-path validation is adapter-scoped to StorageApi",
                );
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::Disabled);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_inmemory_with_storage_api_vars_accepted() {
        // Same silent-tolerance contract as
        // `_disabled_with_storage_api_vars_accepted`, pinned for the
        // second non-StorageApi adapter so a future change that
        // ties the endpoint/base-path validation to anything other
        // than the `StorageApi` variant fires here.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("inmemory")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("")),
                ("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS", Some("5000")),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("30000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "inmemory adapter must silently tolerate bare-hostname endpoint + empty \
                     base path — endpoint/base-path validation is adapter-scoped to StorageApi",
                );
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::InMemory);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_localdisk_with_storage_api_vars_accepted() {
        // Same silent-tolerance contract as the disabled/inmemory
        // tests above, pinned for the third non-StorageApi adapter.
        // `LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH` must be set to a
        // valid path so the existing LocalDisk-path validation
        // succeeds and isolates the storage-api-vars-tolerance
        // assertion from an unrelated LocalDisk validation failure.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                ("LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH", Some("/disk")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("storage.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", Some("")),
                ("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS", Some("5000")),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", Some("30000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "localdisk adapter must silently tolerate bare-hostname endpoint + empty \
                     base path — endpoint/base-path validation is adapter-scoped to StorageApi",
                );
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::LocalDisk);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_vars_default_when_adapter_not_storage_api() {
        // Dual of the explicit-set tolerance tests above: when the
        // adapter is non-StorageApi and NONE of the four new env
        // vars are set, `Config::from_env()` succeeds and each new
        // field takes its documented default value. Pins the
        // no-env-var path so a future regression that flips a
        // default or forgets to wire one in is caught by CI.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("inmemory")),
                ("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT", None),
                ("LIVE_EDIT_CHECKPOINT_BASE_PATH", None),
                ("LIVE_EDIT_CHECKPOINT_CONNECT_TIMEOUT_MS", None),
                ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "inmemory adapter with no storage-api env vars set must accept; \
                     fields take documented defaults",
                );
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::InMemory);
                assert_eq!(cfg.checkpoint_storage_api_endpoint, None);
                assert_eq!(cfg.checkpoint_base_path, None);
                assert_eq!(cfg.checkpoint_connect_timeout_ms, 5000);
                assert_eq!(cfg.checkpoint_request_timeout_ms, 30000);
            },
        );
    }

    // ---- t3.fgj.jd.o6.vh: StorageApi endpoint-rejection paths ----
    //
    // The StorageApi adapter requires `LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT`
    // with an explicit `https://` or `http://` scheme. The four tests below
    // pin each failure mode independently so a future regression in any one
    // arm fires by itself:
    //   1. Missing endpoint env var entirely.
    //   2. Bare-hostname endpoint (no scheme).
    //   3. Non-http scheme (e.g. `ftp://`).
    //   4. Empty-string endpoint.
    // Each test sets `LIVE_EDIT_CHECKPOINT_BASE_PATH` to a valid value so the
    // failure mode is isolated to the endpoint check; otherwise the base-path
    // missing-arm could race the endpoint-arm and obscure which rejection won.
    // The bare-hostname test additionally asserts the cross-cutting "explicit
    // scheme" vocabulary contract with `LIVE_EDIT_STORAGE_SERVICE_ENDPOINT`'s
    // rejection (SR-4.5), matching how `src/auth/mod.rs::bare_hostname_rejected`
    // asserts. Mirrors the LocalDisk path-rejection tests at
    // `_localdisk_requires_path`, `_localdisk_missing_path_error_names_both_vars`,
    // and `_localdisk_empty_string_path_rejected`.

    #[test]
    fn config_from_env_checkpoint_storage_api_missing_endpoint_rejected() {
        // Endpoint env var unset entirely → rejected. Mirrors
        // `_localdisk_missing_path_error_names_both_vars` in that the
        // error message must name both the missing var and the adapter
        // var so the operator sees both knobs in play.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                ("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT", None),
                (
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH",
                    Some("/projects/my-project"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("storageapi adapter must reject missing endpoint");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT"),
                    "error must name the offending endpoint var, got: {err}"
                );
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_ADAPTER"),
                    "error must name the adapter var so the operator sees \
                     why the endpoint is required, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_bare_hostname_endpoint_rejected() {
        // Bare hostname (no scheme) → rejected with the explicit-scheme
        // vocabulary. Pins the cross-cutting contract with
        // `LIVE_EDIT_STORAGE_SERVICE_ENDPOINT`'s rejection (SR-4.5):
        // the error must contain `explicit scheme` so the operator-
        // facing wording is consistent across both env-var rejections.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("storage.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH",
                    Some("/projects/my-project"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("storageapi adapter must reject bare-hostname endpoint");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT"),
                    "error must name the offending endpoint var, got: {err}"
                );
                assert!(
                    err.contains("explicit scheme"),
                    "error must use the cross-cutting `explicit scheme` \
                     vocabulary so the wording matches \
                     `LIVE_EDIT_STORAGE_SERVICE_ENDPOINT`'s rejection (SR-4.5), \
                     got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_non_http_scheme_endpoint_rejected() {
        // Non-http scheme (e.g. `ftp://`) → rejected. Mirrors
        // `src/auth/mod.rs::ftp_scheme_rejected`. The validator must
        // name the offending env var so the operator can locate the
        // knob without grepping.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                (
                    "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                    Some("ftp://storage.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH",
                    Some("/projects/my-project"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("storageapi adapter must reject non-http scheme");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT"),
                    "error must name the offending endpoint var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_empty_endpoint_rejected() {
        // Empty-string endpoint → rejected. Mirrors
        // `_localdisk_empty_string_path_rejected`. If the Engineer's
        // trim/filter logic treats empty as "missing", this failure
        // mode is identical to the `_missing_endpoint_rejected` case
        // above; pinning both means a future regression that special-
        // cases empty before the trim is caught.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
                ("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT", Some("")),
                (
                    "LIVE_EDIT_CHECKPOINT_BASE_PATH",
                    Some("/projects/my-project"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("storageapi adapter must reject empty-string endpoint");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT"),
                    "error must name the offending endpoint var, got: {err}"
                );
            },
        );
    }

    // ---- StorageApi requires LIVE_EDIT_SERVICE_TOKEN_FILE ----
    //
    // `Config::from_env` is the primary gate: the adapter authenticates its
    // outbound calls only with the token the credential-provisioner sidecar
    // writes to this file, so the combination is refused at configuration
    // load, before any port bind or backing-store connect.
    // `build_checkpoint_store`'s refusal of a missing provider is defense in
    // depth (`checkpoint::builder::tests`).

    /// A storage-api token-file path for the cases below; never read.
    const STORAGE_API_TOKEN_FILE: &str = "/var/run/oauth/access_token";

    /// `Config::from_env` with the StorageApi adapter selected and its three
    /// dependent variables set as given (`None` = unset), on top of the
    /// required variables.
    fn from_env_storage_api(
        endpoint: Option<&str>,
        base_path: Option<&str>,
        token_file: Option<&str>,
    ) -> Result<Config, String> {
        let mut vars: Vec<(&str, Option<&str>)> = REQUIRED_VARS.to_vec();
        vars.extend([
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
            ("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT", endpoint),
            ("LIVE_EDIT_CHECKPOINT_BASE_PATH", base_path),
            ("LIVE_EDIT_SERVICE_TOKEN_FILE", token_file),
        ]);
        temp_env::with_vars(vars, || Config::from_env_with(PodMemorySource::Reference))
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_missing_service_token_file_rejected() {
        let err = from_env_storage_api(
            Some("https://storage.example.com"),
            Some("/projects/my-project"),
            None,
        )
        .expect_err("storageapi without LIVE_EDIT_SERVICE_TOKEN_FILE must be rejected");
        assert!(
            err.starts_with("LIVE_EDIT_SERVICE_TOKEN_FILE is required"),
            "error must name the token-file var, got: {err}"
        );
        assert!(
            err.contains("LIVE_EDIT_CHECKPOINT_ADAPTER=storageapi"),
            "error must name the adapter so the operator sees why the token file is \
             required, got: {err}"
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_blank_service_token_file_rejected() {
        // A value that trims to empty is no token file: it collapses to
        // `None` and is refused the same as an unset variable.
        for raw in ["", "   ", " \t\n"] {
            let err = from_env_storage_api(
                Some("https://storage.example.com"),
                Some("/projects/my-project"),
                Some(raw),
            )
            .expect_err("storageapi with a blank LIVE_EDIT_SERVICE_TOKEN_FILE must be rejected");
            assert!(
                err.starts_with("LIVE_EDIT_SERVICE_TOKEN_FILE is required"),
                "error for {raw:?} must name the token-file var, got: {err}"
            );
        }
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_with_service_token_file_accepted() {
        let cfg = from_env_storage_api(
            Some("https://storage.example.com"),
            Some("/projects/my-project"),
            Some(STORAGE_API_TOKEN_FILE),
        )
        .expect("storageapi with endpoint, base path and token file must load");
        assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::StorageApi);
        assert_eq!(
            cfg.service_token_file,
            Some(PathBuf::from(STORAGE_API_TOKEN_FILE)),
            "the token file the gate checked must be the one carried in Config"
        );
    }

    #[test]
    fn config_from_env_checkpoint_storage_api_reports_missing_vars_before_endpoint_shape() {
        // Missing first, then shape: missing endpoint → missing base path →
        // missing token file → endpoint shape. Each step fixes the previous
        // step's variable and leaves every later one broken, so each assertion
        // shows its error outranks all that follow. The bare hostname is
        // invalid in shape from step 2 on.
        const BARE_HOST: &str = "storage.example.com";
        const BASE_PATH: &str = "/projects/my-project";

        let err = from_env_storage_api(None, None, None).expect_err("step 1 must fail");
        assert!(
            err.starts_with("LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT is required"),
            "a missing endpoint must be reported first, got: {err}"
        );

        let err = from_env_storage_api(Some(BARE_HOST), None, None).expect_err("step 2 must fail");
        assert!(
            err.starts_with("LIVE_EDIT_CHECKPOINT_BASE_PATH is required"),
            "a missing base path must outrank the token file and the endpoint shape, got: {err}"
        );

        let err = from_env_storage_api(Some(BARE_HOST), Some(BASE_PATH), None)
            .expect_err("step 3 must fail");
        assert!(
            err.starts_with("LIVE_EDIT_SERVICE_TOKEN_FILE is required"),
            "a missing token file must outrank the endpoint shape, got: {err}"
        );

        let err = from_env_storage_api(
            Some(BARE_HOST),
            Some(BASE_PATH),
            Some(STORAGE_API_TOKEN_FILE),
        )
        .expect_err("step 4 must fail");
        assert!(
            err.starts_with("invalid value for LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT")
                && err.contains("explicit scheme"),
            "with every variable present the endpoint shape is reported, got: {err}"
        );
    }

    #[test]
    fn config_from_env_checkpoint_multiple_violations_at_least_one_reported() {
        // Multiple bad values at once: the validator must surface
        // at least one of them rather than silently masking both
        // behind a different error. Pin this so a future refactor
        // that reorders validation passes can't hide a real
        // misconfiguration behind an unrelated error path. We
        // accept either `TICK_SECS=0` or `QUIESCE_SECS=0` in the
        // error string — the test is about "at least one violation
        // is reported", not which one wins the race.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TICK_SECS", Some("0")),
                ("LIVE_EDIT_CHECKPOINT_QUIESCE_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("two simultaneous violations must surface at least one");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TICK_SECS")
                        || err.contains("LIVE_EDIT_CHECKPOINT_QUIESCE_SECS"),
                    "error must name at least one of the offending env vars, got: {err}"
                );
            },
        );
    }

    // ---- t3.fgj.ya.99.eu: CheckpointAdapter FromStr pure parsing ----
    //
    // Mirrors the AuthProtocol / UpdateLayerBackend FromStr coverage above.
    // Pinning each variant + each casing + each rejection path means a
    // silent reorder, rename, or drop of a variant fails CI fast — every
    // downstream adapter / loop / recovery-hook test in this Epic depends
    // on this enum mapping staying stable.

    #[test]
    fn checkpoint_adapter_from_str_accepts_lowercase_disabled() {
        assert_eq!(
            CheckpointAdapter::from_str("disabled").unwrap(),
            CheckpointAdapter::Disabled
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_accepts_lowercase_inmemory() {
        assert_eq!(
            CheckpointAdapter::from_str("inmemory").unwrap(),
            CheckpointAdapter::InMemory
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_accepts_lowercase_localdisk() {
        assert_eq!(
            CheckpointAdapter::from_str("localdisk").unwrap(),
            CheckpointAdapter::LocalDisk
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_accepts_lowercase_storageapi() {
        assert_eq!(
            CheckpointAdapter::from_str("storageapi").unwrap(),
            CheckpointAdapter::StorageApi
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_accepts_uppercase_disabled() {
        assert_eq!(
            CheckpointAdapter::from_str("DISABLED").unwrap(),
            CheckpointAdapter::Disabled
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_accepts_uppercase_storageapi() {
        assert_eq!(
            CheckpointAdapter::from_str("STORAGEAPI").unwrap(),
            CheckpointAdapter::StorageApi
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_accepts_mixed_case_inmemory() {
        assert_eq!(
            CheckpointAdapter::from_str("InMemory").unwrap(),
            CheckpointAdapter::InMemory
        );
        assert_eq!(
            CheckpointAdapter::from_str("iNMeMoRy").unwrap(),
            CheckpointAdapter::InMemory
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_accepts_mixed_case_localdisk() {
        assert_eq!(
            CheckpointAdapter::from_str("LocalDisk").unwrap(),
            CheckpointAdapter::LocalDisk
        );
        assert_eq!(
            CheckpointAdapter::from_str("lOcAlDiSk").unwrap(),
            CheckpointAdapter::LocalDisk
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_rejects_unknown() {
        let err = CheckpointAdapter::from_str("nfs").expect_err("should reject unknown value");
        assert!(matches!(err, ParseCheckpointAdapterError { .. }));
        let msg = err.to_string();
        // Display must name the env var so operators can locate the
        // offending knob without grepping for the variant name.
        assert!(
            msg.contains("LIVE_EDIT_CHECKPOINT_ADAPTER"),
            "error should name LIVE_EDIT_CHECKPOINT_ADAPTER, got: {msg}"
        );
        // Display must list every accepted variant so operators don't
        // have to read source to discover the alternatives.
        assert!(
            msg.contains("disabled"),
            "error should mention `disabled`, got: {msg}"
        );
        assert!(
            msg.contains("inmemory"),
            "error should mention `inmemory`, got: {msg}"
        );
        assert!(
            msg.contains("localdisk"),
            "error should mention `localdisk`, got: {msg}"
        );
        assert!(
            msg.contains("storageapi"),
            "error should mention `storageapi`, got: {msg}"
        );
        // Display must echo the input so the operator sees what they typed.
        assert!(
            msg.contains("nfs"),
            "error should echo the original input `nfs`, got: {msg}"
        );

        let err2 = CheckpointAdapter::from_str("s3").expect_err("should reject unknown value");
        assert!(err2.to_string().contains("s3"));

        let err3 =
            CheckpointAdapter::from_str("LocalDisk!").expect_err("should reject unknown value");
        assert!(err3.to_string().contains("LocalDisk!"));
    }

    #[test]
    fn checkpoint_adapter_from_str_rejects_empty_string() {
        let err = CheckpointAdapter::from_str("").expect_err("should reject empty string");
        assert!(matches!(err, ParseCheckpointAdapterError { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("LIVE_EDIT_CHECKPOINT_ADAPTER"),
            "error should name the env var, got: {msg}"
        );
        assert!(
            msg.contains("disabled")
                && msg.contains("inmemory")
                && msg.contains("localdisk")
                && msg.contains("storageapi"),
            "error should list every accepted variant, got: {msg}"
        );
    }

    #[test]
    fn checkpoint_adapter_from_str_rejects_whitespace_only() {
        // `CheckpointAdapter::from_str` itself does NOT trim — trimming
        // happens up in `optional_parsed` before the parser sees the
        // value. So `"   "` reaches the parser verbatim and is rejected
        // as an unknown variant. Pin that contract so a future change
        // to add trimming inside `from_str` is a deliberate decision
        // (and the whitespace-trim test below still covers the
        // operationally important `Config::from_env` end-to-end path).
        let err =
            CheckpointAdapter::from_str("   ").expect_err("should reject whitespace-only input");
        assert!(matches!(err, ParseCheckpointAdapterError { .. }));
    }

    // ---- ValkeyAuthMode FromStr pure parsing ----
    //
    // Mirrors the AuthProtocol / CheckpointAdapter FromStr coverage above.
    // Pinning each variant + each casing + each rejection path means a
    // silent reorder, rename, or drop of a variant fails CI fast — the
    // Valkey auth-mode selection (SR-2.1) and every downstream credential
    // provider gated on it depend on this enum mapping staying stable.

    #[test]
    fn valkey_auth_mode_from_str_accepts_lowercase_static() {
        assert_eq!(
            ValkeyAuthMode::from_str("static").unwrap(),
            ValkeyAuthMode::Static
        );
    }

    #[test]
    fn valkey_auth_mode_from_str_accepts_lowercase_access_token_file() {
        assert_eq!(
            ValkeyAuthMode::from_str("accesstokenfile").unwrap(),
            ValkeyAuthMode::AccessTokenFile
        );
    }

    #[test]
    fn valkey_auth_mode_from_str_accepts_lowercase_iam() {
        assert_eq!(
            ValkeyAuthMode::from_str("iam").unwrap(),
            ValkeyAuthMode::Iam
        );
    }

    #[test]
    fn valkey_auth_mode_from_str_accepts_uppercase_static() {
        assert_eq!(
            ValkeyAuthMode::from_str("STATIC").unwrap(),
            ValkeyAuthMode::Static
        );
    }

    #[test]
    fn valkey_auth_mode_from_str_accepts_mixed_case_access_token_file() {
        assert_eq!(
            ValkeyAuthMode::from_str("AccessTokenFile").unwrap(),
            ValkeyAuthMode::AccessTokenFile
        );
        assert_eq!(
            ValkeyAuthMode::from_str("aCcEsStOkEnFiLe").unwrap(),
            ValkeyAuthMode::AccessTokenFile
        );
    }

    #[test]
    fn valkey_auth_mode_from_str_accepts_mixed_case_iam() {
        assert_eq!(
            ValkeyAuthMode::from_str("iAm").unwrap(),
            ValkeyAuthMode::Iam
        );
        assert_eq!(
            ValkeyAuthMode::from_str("IAM").unwrap(),
            ValkeyAuthMode::Iam
        );
        assert_eq!(
            ValkeyAuthMode::from_str("Iam").unwrap(),
            ValkeyAuthMode::Iam
        );
    }

    #[test]
    fn valkey_auth_mode_from_str_rejects_unknown() {
        let err = ValkeyAuthMode::from_str("aws").expect_err("should reject unknown value");
        assert!(matches!(err, ParseValkeyAuthModeError { .. }));
        let msg = err.to_string();
        // Display must list every accepted variant so operators don't
        // have to read source to discover the alternatives.
        assert!(
            msg.contains("static"),
            "error should mention `static`, got: {msg}"
        );
        assert!(
            msg.contains("accessTokenFile"),
            "error should mention `accessTokenFile`, got: {msg}"
        );
        assert!(
            msg.contains("iam"),
            "error should mention `iam`, got: {msg}"
        );
        // Display must echo the input so the operator sees what they typed.
        assert!(
            msg.contains("aws"),
            "error should echo the original input `aws`, got: {msg}"
        );

        let err2 = ValkeyAuthMode::from_str("managed").expect_err("should reject unknown value");
        assert!(err2.to_string().contains("managed"));

        let err3 = ValkeyAuthMode::from_str("Static!").expect_err("should reject unknown value");
        assert!(err3.to_string().contains("Static!"));
    }

    #[test]
    fn valkey_auth_mode_from_str_rejects_empty_string() {
        let err = ValkeyAuthMode::from_str("").expect_err("should reject empty string");
        assert!(matches!(err, ParseValkeyAuthModeError { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("static") && msg.contains("accessTokenFile") && msg.contains("iam"),
            "error should list every accepted variant, got: {msg}"
        );
    }

    #[test]
    fn valkey_auth_mode_from_str_rejects_whitespace_only() {
        // `ValkeyAuthMode::from_str` itself does NOT trim — trimming
        // happens up in `optional_parsed` before the parser sees the
        // value. So `"   "` reaches the parser verbatim and is rejected
        // as an unknown variant. Pin that contract so a future change
        // to add trimming inside `from_str` is a deliberate decision.
        let err = ValkeyAuthMode::from_str("   ").expect_err("should reject whitespace-only input");
        assert!(matches!(err, ParseValkeyAuthModeError { .. }));
    }

    // ---- ValkeyIamService FromStr (t3.tr5.we.6w.br) ----
    //
    // The AWS-managed-backend selector chosen by LIVE_EDIT_VALKEY_IAM_SERVICE
    // under `iam` auth mode. Mirrors the ValkeyAuthMode FromStr coverage
    // above: pin each variant × each casing plus the reject path so a silent
    // reorder / rename / drop of a variant fails CI fast. The two managed
    // backends (MemoryDb default, ElastiCache) differ in SigV4 signing host
    // shape, so this mapping staying stable is load-bearing.

    #[test]
    fn valkey_iam_service_from_str_accepts_lowercase_memorydb() {
        assert_eq!(
            ValkeyIamService::from_str("memorydb").unwrap(),
            ValkeyIamService::MemoryDb
        );
    }

    #[test]
    fn valkey_iam_service_from_str_accepts_lowercase_elasticache() {
        assert_eq!(
            ValkeyIamService::from_str("elasticache").unwrap(),
            ValkeyIamService::ElastiCache
        );
    }

    #[test]
    fn valkey_iam_service_from_str_accepts_uppercase() {
        assert_eq!(
            ValkeyIamService::from_str("MEMORYDB").unwrap(),
            ValkeyIamService::MemoryDb
        );
        assert_eq!(
            ValkeyIamService::from_str("ELASTICACHE").unwrap(),
            ValkeyIamService::ElastiCache
        );
    }

    #[test]
    fn valkey_iam_service_from_str_accepts_mixed_case() {
        assert_eq!(
            ValkeyIamService::from_str("MemoryDb").unwrap(),
            ValkeyIamService::MemoryDb
        );
        assert_eq!(
            ValkeyIamService::from_str("mEmOrYdB").unwrap(),
            ValkeyIamService::MemoryDb
        );
        assert_eq!(
            ValkeyIamService::from_str("ElastiCache").unwrap(),
            ValkeyIamService::ElastiCache
        );
        assert_eq!(
            ValkeyIamService::from_str("eLaStIcAcHe").unwrap(),
            ValkeyIamService::ElastiCache
        );
    }

    #[test]
    fn valkey_iam_service_from_str_rejects_unknown() {
        let err =
            ValkeyIamService::from_str("serverless").expect_err("should reject unknown value");
        assert!(matches!(err, ParseValkeyIamServiceError { .. }));
        let msg = err.to_string();
        // Display must list both accepted variants so operators don't have to
        // read source to discover the alternatives.
        assert!(
            msg.contains("memorydb"),
            "error should mention `memorydb`, got: {msg}"
        );
        assert!(
            msg.contains("elasticache"),
            "error should mention `elasticache`, got: {msg}"
        );
        // Display must echo the input so the operator sees what they typed.
        assert!(
            msg.contains("serverless"),
            "error should echo the original input `serverless`, got: {msg}"
        );

        let err2 = ValkeyIamService::from_str("redis").expect_err("should reject unknown value");
        assert!(err2.to_string().contains("redis"));
    }

    #[test]
    fn valkey_iam_service_from_str_rejects_empty_string() {
        let err = ValkeyIamService::from_str("").expect_err("should reject empty string");
        assert!(matches!(err, ParseValkeyIamServiceError { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("memorydb") && msg.contains("elasticache"),
            "error should list both accepted variants, got: {msg}"
        );
    }

    // ---- Config::from_env ValkeyAuthMode integration ----
    //
    // Exercise the LIVE_EDIT_VALKEY_AUTH_MODE env-var → `valkey_auth_mode`
    // Config field path through the public `Config::from_env` API (SR-2.1),
    // mirroring the AuthProtocol integration tests above. The `accessTokenFile`/`iam`
    // positive cases additionally require the mode's inputs plus a TLS
    // `rediss://` URL (managed passwordless endpoints are TLS-only per
    // `validate_valkey_auth_mode`), so they are `#[cfg(feature = "valkey-tls")]`-
    // gated like `config_from_env_rediss_valkey_url_accepted_in_default_build` —
    // under `--no-default-features` the rediss:// URL is rejected by the
    // TLS-scheme guard before the auth-mode check runs.

    #[test]
    fn config_from_env_default_valkey_auth_mode_is_static() {
        // Unset LIVE_EDIT_VALKEY_AUTH_MODE defaults to Static (SR-2.1), and
        // every SR-2.2 per-mode field must be unset/default — the default-static
        // no-change contract. Explicitly None the per-mode vars so a polluted
        // process env can't mask a leaked non-default. The refresh-lead is a
        // non-Option u64 whose default is 180.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", None),
                ("LIVE_EDIT_VALKEY_TOKEN_FILE", None),
                ("LIVE_EDIT_VALKEY_TOKEN_USERNAME", None),
                ("LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS", None),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", None),
                ("LIVE_EDIT_VALKEY_IAM_REGION", None),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::Static);
                assert_eq!(
                    cfg.valkey_token_file, None,
                    "access-token-file token-file must default to None under static"
                );
                assert_eq!(
                    cfg.valkey_token_username, None,
                    "access-token-file username must default to None under static"
                );
                assert_eq!(
                    cfg.valkey_token_refresh_lead_secs, 180,
                    "access-token-file refresh-lead must default to 180 under static"
                );
                assert_eq!(
                    cfg.valkey_iam_username, None,
                    "iam username must default to None under static"
                );
                assert_eq!(
                    cfg.valkey_iam_region, None,
                    "iam region must default to None under static"
                );
                assert_eq!(
                    cfg.valkey_iam_cluster_host, None,
                    "iam cluster-host must default to None under static"
                );
            },
        );
    }

    #[test]
    fn config_from_env_explicit_static_valkey_auth_mode() {
        // Explicit `static` needs no extra inputs and no TLS URL, so this is
        // ungated.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("static")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit static must parse");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::Static);
            },
        );
    }

    #[test]
    fn config_from_env_valkey_auth_mode_mixed_case_static() {
        // Case-insensitivity through the env boundary for the ungated variant.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("sTaTiC")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("mixed-case sTaTiC must parse to Static");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::Static);
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_explicit_access_token_file_valkey_auth_mode() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("accessTokenFile")),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_FILE",
                    Some("/var/run/secrets/access-token"),
                ),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_USERNAME",
                    Some("00000000-0000-0000-0000-000000000000"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit accessTokenFile with required inputs + TLS URL must parse");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::AccessTokenFile);
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_valkey_auth_mode_uppercase_access_token_file() {
        // Case-insensitivity survives the env-var boundary via
        // `to_ascii_lowercase` inside FromStr, reached through optional_parsed.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("ACCESSTOKENFILE")),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_FILE",
                    Some("/var/run/secrets/access-token"),
                ),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_USERNAME",
                    Some("00000000-0000-0000-0000-000000000000"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("uppercase ACCESSTOKENFILE must parse to the AccessTokenFile variant");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::AccessTokenFile);
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_explicit_iam_valkey_auth_mode() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", Some("my-cluster")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit iam with required inputs + TLS URL must parse");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::Iam);
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_valkey_auth_mode_mixed_case_iam() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("Iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", Some("my-cluster")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("mixed-case Iam must parse to the Iam variant");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::Iam);
            },
        );
    }

    #[test]
    fn config_from_env_unknown_valkey_auth_mode_errors() {
        // An unrecognized value fails in `optional_parsed`, before any
        // mode-specific validation — the error names the env-var and lists the
        // accepted values so operators can self-correct.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("bogus")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("unknown LIVE_EDIT_VALKEY_AUTH_MODE must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_AUTH_MODE"),
                    "error should name the env-var, got: {err}"
                );
                assert!(
                    err.contains("static")
                        && err.contains("accessTokenFile")
                        && err.contains("iam"),
                    "error should list the accepted values, got: {err}"
                );
            },
        );
    }

    // ---- validate_valkey_auth_mode fail-closed ----
    //
    // SR-2.3 (b.tr5): `validate_valkey_auth_mode` runs in the `Config::from_env`
    // chain *after* `validate_valkey_tls_compiled` and fail-closes when the
    // selected passwordless mode's required inputs are missing/inconsistent.
    // Each reject-branch test below isolates a single failure reason: it
    // supplies every OTHER required input for the mode and violates only the
    // branch under test, then asserts an actionable substring of the `Err`.
    //
    // The missing-input branches drive the validator through a `rediss://` URL
    // so the plaintext check does not also fire — which makes them accept a TLS
    // scheme, so they are `#[cfg(feature = "valkey-tls")]`-gated: under
    // `--no-default-features` the `rediss://` URL is rejected by
    // `validate_valkey_tls_compiled` before the auth-mode check runs. The
    // plaintext-with-passwordless-mode cases use a `redis://` URL (plaintext in
    // both builds) and are therefore feature-agnostic.

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_access_token_file_without_token_file_rejected() {
        // accessTokenFile + TLS URL + username present, token-file unset → the
        // token-file branch is the sole violation.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("accessTokenFile")),
                ("LIVE_EDIT_VALKEY_TOKEN_FILE", None),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_USERNAME",
                    Some("00000000-0000-0000-0000-000000000000"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("accessTokenFile without a token-file path must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_TOKEN_FILE"),
                    "reject must name the missing token-file input, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_access_token_file_without_username_rejected() {
        // accessTokenFile + TLS URL + token-file present, username unset → the
        // username branch is the sole violation.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("accessTokenFile")),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_FILE",
                    Some("/var/run/secrets/access-token"),
                ),
                ("LIVE_EDIT_VALKEY_TOKEN_USERNAME", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("accessTokenFile without the username must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_TOKEN_USERNAME"),
                    "reject must name the missing username input, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_iam_without_region_rejected() {
        // iam + TLS URL + ACL-username present, region unset → the region branch
        // is the sole violation.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", None),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("iam without an AWS region must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_IAM_REGION"),
                    "reject must name the missing region input, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_iam_without_acl_username_rejected() {
        // iam + TLS URL + region present, username unset → the username branch
        // is the sole violation.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("iam without an ACL username must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_IAM_USERNAME"),
                    "reject must name the missing ACL-username input, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_iam_without_cluster_host_rejected() {
        // t3.tr5.we.6w.br: iam + TLS URL + region + ACL-username present, but
        // LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST unset → the cluster-host branch (the
        // check deferred from Epic 2 and landed this Epic) is the sole violation.
        // Models `config_from_env_iam_without_region_rejected`.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", None),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("iam without a cluster host must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST"),
                    "reject must name the missing cluster-host input, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_iam_empty_cluster_host_rejected() {
        // Companion to the missing-cluster-host case: an explicitly empty
        // LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST must fail closed identically. The
        // field parse trims-then-filters empties to `None`, so `Some("")` reaches
        // the required-when-iam check as `None` and trips the same branch —
        // missing-or-empty are equivalent here.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", Some("")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("iam with an empty cluster host must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST"),
                    "reject must name the empty cluster-host input, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_iam_unknown_service_rejected() {
        // t3.tr5.we.6w.br: a fully-configured iam on a TLS URL but with an
        // unknown LIVE_EDIT_VALKEY_IAM_SERVICE value → parse-time rejection. The
        // error must name the offending var and list the accepted values so the
        // operator can self-correct without reading source.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", Some("my-cluster")),
                ("LIVE_EDIT_VALKEY_IAM_SERVICE", Some("serverless")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("an unknown iam service selector must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_IAM_SERVICE"),
                    "reject must name the offending var, got: {err}"
                );
                assert!(
                    err.contains("memorydb"),
                    "reject must list the `memorydb` accepted value, got: {err}"
                );
                assert!(
                    err.contains("elasticache"),
                    "reject must list the `elasticache` accepted value, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_iam_service_defaults_to_memorydb() {
        // t3.tr5.we.6w.br: a fully-configured iam on a TLS URL with
        // LIVE_EDIT_VALKEY_IAM_SERVICE unset → `Ok`, and the selector defaults
        // to `MemoryDb` (the recommended backend). Pins the default so a silent
        // flip of the `optional_parsed` fallback fails CI.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", Some("my-cluster")),
                ("LIVE_EDIT_VALKEY_IAM_SERVICE", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("fully-configured iam with no service selector must parse");
                assert_eq!(
                    cfg.valkey_iam_service,
                    ValkeyIamService::MemoryDb,
                    "iam service selector must default to memorydb when the var is unset"
                );
            },
        );
    }

    #[test]
    fn config_from_env_access_token_file_with_plaintext_url_rejected() {
        // SR-2.3: accessTokenFile with all inputs present but a plaintext
        // `redis://` URL → the plaintext-vs-passwordless check (reusing
        // `valkey_url_uses_tls` negated) rejects. `redis://` is never a TLS
        // scheme, so this holds in both builds — NOT feature-gated.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_URL", Some("redis://127.0.0.1:6379")),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("accessTokenFile")),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_FILE",
                    Some("/var/run/secrets/access-token"),
                ),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_USERNAME",
                    Some("00000000-0000-0000-0000-000000000000"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("accessTokenFile on a plaintext URL must be rejected");
                assert!(
                    err.contains("plaintext"),
                    "reject must name the plaintext problem, got: {err}"
                );
                assert!(
                    err.contains("TLS"),
                    "reject must point at the TLS requirement, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_iam_with_plaintext_url_rejected() {
        // Analogous to the accessTokenFile plaintext case, for `iam`. NOT feature-gated.
        // All per-mode inputs present (incl. cluster host) so the required-when-iam
        // field checks pass and the plaintext-vs-passwordless check is the sole
        // violation.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_URL", Some("redis://127.0.0.1:6379")),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", Some("my-cluster")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("iam on a plaintext URL must be rejected");
                assert!(
                    err.contains("plaintext"),
                    "reject must name the plaintext problem, got: {err}"
                );
                assert!(
                    err.contains("TLS"),
                    "reject must point at the TLS requirement, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_valkeys_url_with_passwordless_mode_not_tripped_by_plaintext_check() {
        // Regression pin for the `valkeys://` blind spot the SDD calls out: the
        // plaintext check must reuse `valkey_url_uses_tls` (which treats both
        // `rediss` and `valkeys` as TLS), NOT a fresh `starts_with("redis://")`.
        // A `valkeys://` URL with a passwordless mode + all inputs present must
        // therefore be ACCEPTED — a naive `redis://`-only guard would wrongly
        // reject it as plaintext.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("valkeys://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("accessTokenFile")),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_FILE",
                    Some("/var/run/secrets/access-token"),
                ),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_USERNAME",
                    Some("00000000-0000-0000-0000-000000000000"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "valkeys:// (a TLS scheme to fred) must not be rejected by the plaintext check",
                );
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::AccessTokenFile);
            },
        );
    }

    // ---- validate_valkey_auth_mode accept side (SR-3.1 no-regression) ----
    //
    // The accept-branch complement to the reject tests above (b.tr5,
    // t3.tr5.vc.t7.4g). Two guarantees: (a) under `static` the passwordless
    // plaintext check must NOT fire — a plaintext `redis://` URL (incl. the
    // binary default) with no per-mode inputs stays byte-identical to pre-Epic-2
    // behavior; and (b) a fully-configured `accessTokenFile`/`iam` on a `rediss://` URL is
    // accepted AND every per-mode field round-trips onto `Config` (the existing
    // `config_from_env_explicit_{access_token_file,iam}_valkey_auth_mode` tests pin only the
    // mode discriminant, not the field values). The static cases are NOT
    // feature-gated (plaintext `redis://` is accepted in both builds); the
    // passwordless cases are `#[cfg(feature = "valkey-tls")]`-gated because
    // `validate_valkey_tls_compiled` rejects the `rediss://` URL first under
    // `--no-default-features`.

    #[test]
    fn config_from_env_static_default_plaintext_url_passwordless_check_not_tripped() {
        // Mode unset (default `static`) + an explicit plaintext `redis://` URL +
        // no per-mode inputs → `Ok`. Pins that the passwordless plaintext check
        // (which reuses `valkey_url_uses_tls` negated) fires ONLY for
        // accessTokenFile|iam, never for static — the SR-3.1 no-regression contract.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_URL", Some("redis://localhost:6379")),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", None),
                ("LIVE_EDIT_VALKEY_TOKEN_FILE", None),
                ("LIVE_EDIT_VALKEY_TOKEN_USERNAME", None),
                ("LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS", None),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", None),
                ("LIVE_EDIT_VALKEY_IAM_REGION", None),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "default static on a plaintext redis:// URL with no per-mode inputs must parse",
                );
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::Static);
            },
        );
    }

    #[test]
    fn config_from_env_explicit_static_plaintext_url_accepted() {
        // Explicit `LIVE_EDIT_VALKEY_AUTH_MODE=static` + an explicit plaintext
        // `redis://` URL → `Ok`. The explicit-mode analog of the case above:
        // naming `static` outright still bypasses the passwordless plaintext
        // check.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_URL", Some("redis://127.0.0.1:6379")),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("static")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit static on a plaintext redis:// URL must parse");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::Static);
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_access_token_file_fully_configured_fields_round_trip() {
        // Fully-configured accessTokenFile on a `rediss://` URL → `Ok`, and every
        // per-mode field round-trips onto `Config`. A non-default refresh-lead
        // (300, vs. the 180 default) proves the value crosses the env boundary
        // rather than silently defaulting.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("accessTokenFile")),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_FILE",
                    Some("/var/run/secrets/access-token"),
                ),
                (
                    "LIVE_EDIT_VALKEY_TOKEN_USERNAME",
                    Some("00000000-0000-0000-0000-000000000000"),
                ),
                ("LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS", Some("300")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("fully-configured accessTokenFile on a TLS URL must parse");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::AccessTokenFile);
                assert_eq!(
                    cfg.valkey_token_file,
                    Some(PathBuf::from("/var/run/secrets/access-token")),
                    "token-file must round-trip as Option<PathBuf> with the exact value"
                );
                assert_eq!(
                    cfg.valkey_token_username,
                    Some("00000000-0000-0000-0000-000000000000".to_string()),
                    "token username must round-trip"
                );
                assert_eq!(
                    cfg.valkey_token_refresh_lead_secs, 300,
                    "refresh-lead must round-trip the non-default value, not fall back to 180"
                );
            },
        );
    }

    // ── LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS bounds ───────────────────────
    //
    // Four cases over one parameter: zero (SR-4.1 fail-closed), the accepted
    // bound, one past it, and `u64::MAX`. They differ only in the value and in
    // what the outcome must say, so they share one env fixture and one
    // assertion helper per direction rather than four near-identical
    // `temp_env::with_vars` blocks — the copy-paste-with-one-line-differences
    // shape the test-review guide flags, and the shape in which a fifth case
    // gets bolted on as a fifth copy.

    /// The minimal environment a refresh-lead bound case needs.
    ///
    /// The lead check lives in the *unconditional* field-parse, so it fires
    /// regardless of auth mode: these cases run under the default `static` mode
    /// on a plaintext URL, which needs no TLS feature gate and sets no per-mode
    /// variables — so the lead is the only thing in the environment that can be
    /// the violation. `LIVE_EDIT_VALKEY_AUTH_MODE` is deliberately *not* set:
    /// `static` is the default, and spelling it out would read as though the
    /// case depended on it.
    fn refresh_lead_env(value: &str) -> [(&str, Option<&str>); 4] {
        [
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://auth.example.com"),
            ),
            (
                "LIVE_EDIT_USERINFO_ENDPOINT",
                Some("http://userinfo.example.com"),
            ),
            ("LIVE_EDIT_VALKEY_URL", Some("redis://127.0.0.1:6379")),
            ("LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS", Some(value)),
        ]
    }

    /// Assert `Config::from_env` rejects `value` as a refresh lead, and that the
    /// message contains every string in `expected`.
    ///
    /// The substrings are the case-specific part: what a reject has to *say* is
    /// the thing an operator reads, and each case fails for a different reason.
    fn assert_refresh_lead_rejected(value: &str, expected: &[&str]) {
        temp_env::with_vars(refresh_lead_env(value), || {
            let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                "a refresh lead outside the accepted range must be rejected at startup, not \
                 carried into the wiring site",
            );
            for needle in expected {
                assert!(
                    err.contains(needle),
                    "the reject for LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS={value} must \
                     contain {needle:?}, got: {err}"
                );
            }
        });
    }

    /// Assert `Config::from_env` accepts `value` as a refresh lead, that it
    /// round-trips unclamped, and that it satisfies the datetime-range contract
    /// the wiring site and the token provider rely on.
    ///
    /// The two arithmetic checks are the reason this is not just an
    /// `expect("must parse")`. `main.rs` converts every accepted lead with
    /// `chrono::Duration::try_seconds(..).expect(..)`, and the provider then
    /// *adds* it to `Utc::now()` at the polling task's stale-entry window — the
    /// sole such site since b.mq8 deleted the withholding accessor that shared
    /// it, and the binding constraint of the two checks below, being the tighter
    /// one. An accepted value failing either would mean config called valid a
    /// value the binary aborts on, or one whose window can never open.
    ///
    /// The surviving site does not add the bare lead: `run_polling_task`
    /// precomputes `stale_entry_lead = refresh_lead + poll_interval` and adds
    /// *that* to `Utc::now()`, so `now + lead` alone is only the looser
    /// *necessary* condition. The check below therefore adds the same sum this
    /// `Config`'s own `service_token_poll_interval_secs` would feed the
    /// provider, so the assertion binds the arithmetic the named site actually
    /// performs rather than a weaker one that could pass while that site
    /// overflowed. It does not move the accepted range in practice —
    /// `VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS` leaves ~228,000 years of headroom,
    /// against which one poll interval is nothing — which is the point: the
    /// tightening costs no behaviour and closes the gap between what this
    /// asserts and what it claims to assert.
    fn assert_refresh_lead_accepted(value: u64) {
        temp_env::with_vars(refresh_lead_env(&value.to_string()), || {
            let cfg = Config::from_env_with(PodMemorySource::Reference).unwrap_or_else(|err| {
                panic!("LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS={value} must parse, got: {err}")
            });
            assert_eq!(
                cfg.valkey_token_refresh_lead_secs, value,
                "an accepted lead must round-trip exactly, not be clamped"
            );
            let lead = chrono::Duration::try_seconds(cfg.valkey_token_refresh_lead_secs as i64)
                .unwrap_or_else(|| {
                    panic!(
                        "every accepted lead must be convertible at the wiring site; otherwise \
                         main.rs's expect aborts the process on a value config called valid"
                    )
                });
            // The provider's stale-entry window opens at
            // `now + (refresh_lead + poll_interval)`, not at `now + lead`, and
            // `service_token_poll_interval_secs` is the field `main.rs` passes
            // as that `poll_interval` for both provider instances. Read it off
            // this same `cfg` so the sum tracks the config a real boot would
            // build.
            let poll_interval =
                chrono::Duration::try_seconds(cfg.service_token_poll_interval_secs as i64)
                    .expect("the poll interval's own range gate keeps this convertible");
            let stale_entry_lead = lead
                .checked_add(&poll_interval)
                .expect("refresh_lead + poll_interval must be a representable Duration");
            assert!(
                chrono::Utc::now()
                    .checked_add_signed(stale_entry_lead)
                    .is_some(),
                "every accepted lead must be addable to the current time *together with one poll \
                 interval*: the provider precomputes `stale_entry_lead = refresh_lead + \
                 poll_interval` and adds that sum to `Utc::now()` at its stale-entry window, so \
                 `now + lead` alone is only the looser necessary condition. A lead whose sum \
                 cannot be added leaves that window permanently shut"
            );
        });
    }

    #[test]
    fn config_from_env_token_refresh_lead_zero_rejected() {
        // SR-4.1 fail-closed: a zero lead would let the token lapse before
        // renewal.
        assert_refresh_lead_rejected(
            "0",
            &["LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS", "greater than 0"],
        );
    }

    /// The largest refresh lead `Config::from_env` accepts — the
    /// `DateTime<Utc>`-range representability bound the provider's
    /// `Utc::now() + lead` arithmetic has to stay inside.
    ///
    /// An alias for the production constant, not a restatement of it, so the
    /// accepted / rejected pair below pins the *shipped* bound from both sides
    /// and moves with it.
    const CHRONO_MAX_LEAD_SECS: u64 = super::VALKEY_TOKEN_REFRESH_LEAD_MAX_SECS;

    #[test]
    fn config_from_env_token_refresh_lead_at_chrono_bound_accepted() {
        // The bound is inclusive: rejecting it would be an off-by-one narrowing
        // a range the wiring site can actually represent.
        assert_refresh_lead_accepted(CHRONO_MAX_LEAD_SECS);
    }

    #[test]
    fn config_from_env_token_refresh_lead_above_chrono_bound_rejected() {
        // One past the bound. Paired with the case above, this is what makes the
        // gate an exact edge rather than an approximate one.
        assert_refresh_lead_rejected(
            &(CHRONO_MAX_LEAD_SECS + 1).to_string(),
            &[
                "LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS",
                // The bound the operator has to come under.
                &CHRONO_MAX_LEAD_SECS.to_string(),
            ],
        );
    }

    #[test]
    fn config_from_env_token_refresh_lead_u64_max_rejected() {
        // b.is1 regression, and the reason the gate cannot merely be the
        // adjacent-value edge above. `u64::MAX` is the worst case of the range
        // the old wiring cast with `as i64`: it wraps to a *negative* lead
        // rather than panicking, which is a silent fail-open — the stale-token
        // phase's entry lead collapses and the proactive re-read never arms,
        // with no startup error and no log line.
        assert_refresh_lead_rejected(
            &u64::MAX.to_string(),
            &[
                "LIVE_EDIT_VALKEY_TOKEN_REFRESH_LEAD_SECS",
                // Not an operational ceiling: the message has to say the bound
                // is chrono representability at the wiring site.
                "chrono",
            ],
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_iam_fully_configured_fields_round_trip() {
        // Fully-configured iam on a `rediss://` URL → `Ok`, and every per-mode
        // field round-trips — including `valkey_iam_cluster_host` (now required
        // under `iam`) and the `valkey_iam_service` selector. A non-default
        // `elasticache` service proves the value crosses the env boundary rather
        // than silently defaulting to `memorydb`.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
                ("LIVE_EDIT_VALKEY_AUTH_MODE", Some("iam")),
                ("LIVE_EDIT_VALKEY_IAM_USERNAME", Some("iam-acl-user")),
                ("LIVE_EDIT_VALKEY_IAM_REGION", Some("us-east-1")),
                ("LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST", Some("my-cluster")),
                ("LIVE_EDIT_VALKEY_IAM_SERVICE", Some("elasticache")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("fully-configured iam on a TLS URL must parse");
                assert_eq!(cfg.valkey_auth_mode, ValkeyAuthMode::Iam);
                assert_eq!(
                    cfg.valkey_iam_username,
                    Some("iam-acl-user".to_string()),
                    "iam ACL-username must round-trip"
                );
                assert_eq!(
                    cfg.valkey_iam_region,
                    Some("us-east-1".to_string()),
                    "iam region must round-trip"
                );
                assert_eq!(
                    cfg.valkey_iam_cluster_host,
                    Some("my-cluster".to_string()),
                    "iam cluster-host must round-trip"
                );
                assert_eq!(
                    cfg.valkey_iam_service,
                    ValkeyIamService::ElastiCache,
                    "iam service selector must round-trip the non-default `elasticache`, \
                     not fall back to memorydb"
                );
            },
        );
    }

    // ---- t3.fgj.ya.99.eu: per-env-var explicit overrides round-trip ----
    //
    // One test per new env var pins the parse path end-to-end through
    // `Config::from_env`. Defaults are pinned in
    // `config_from_env_checkpoint_defaults_match_sdd` above; these tests
    // confirm a non-default value crosses the env-var boundary without
    // silent truncation, default-substitution, or unit confusion.

    #[test]
    fn config_from_env_checkpoint_enabled_explicit_true_overrides_default() {
        // Pair `enabled=true` with `adapter=inmemory` so the LocalDisk
        // path requirement isn't tripped — the test is about the
        // boolean toggle, not the dependent-var guard.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("true")),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("inmemory")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit enabled=true should parse");
                assert!(
                    cfg.checkpoint_enabled,
                    "checkpoint_enabled must round-trip the explicit `true`"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_adapter_explicit_inmemory_overrides_default() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("inmemory")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit adapter=inmemory should parse");
                assert_eq!(cfg.checkpoint_adapter, CheckpointAdapter::InMemory);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_interval_explicit_override_round_trips() {
        // Pair an explicit interval with an explicit tick so the
        // tick < interval cross-validator stays satisfied — this test
        // is about the interval parse path, not the cross-validator
        // (that's `.b8`'s lane).
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_INTERVAL_SECS", Some("120")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit interval=120 should parse");
                assert_eq!(cfg.checkpoint_interval_secs, 120);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_quiesce_explicit_override_round_trips() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_QUIESCE_SECS", Some("45")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit quiesce=45 should parse");
                assert_eq!(cfg.checkpoint_quiesce_secs, 45);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tick_explicit_override_round_trips() {
        // Use 20 (< default interval of 60) so cross-validation passes
        // without setting the interval as well.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TICK_SECS", Some("20")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit tick=20 should parse");
                assert_eq!(cfg.checkpoint_tick_secs, 20);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_retain_generations_explicit_override_round_trips() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS", Some("5")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit retain=5 should parse");
                assert_eq!(cfg.checkpoint_retain_generations, 5);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_seq_safety_margin_explicit_override_round_trips() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN", Some("5000")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit seq_safety_margin=5000 should parse");
                assert_eq!(cfg.checkpoint_seq_safety_margin, 5000);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_local_disk_path_explicit_override_round_trips() {
        // Pin the LocalDisk-adapter path so the dependent-var guard
        // accepts it; the assertion target is the path field itself.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                (
                    "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                    Some("/srv/checkpoints/live-edit"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit local_disk_path should parse");
                assert_eq!(
                    cfg.checkpoint_local_disk_path.as_deref(),
                    Some("/srv/checkpoints/live-edit"),
                );
            },
        );
    }

    // ---- t3.fgj.ya.99.eu: LIVE_EDIT_CHECKPOINT_ENABLED boolean parsing ----
    //
    // `optional_parsed::<bool>` reaches the stdlib `FromStr for bool`
    // implementation, which accepts exactly `"true"` / `"false"` and
    // (since Rust does the lowercase comparison directly) rejects any
    // other casing. These tests pin the contract: lowercase forms are
    // accepted, common-mistake values are rejected with an error that
    // names the env var. The first surface an operator sees on a
    // typo'd manifest is this error message; if it drops the var name
    // the operator has no anchor.

    #[test]
    fn config_from_env_checkpoint_enabled_accepts_lowercase_true() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("true")),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("inmemory")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("`true` must be accepted");
                assert!(cfg.checkpoint_enabled);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_enabled_accepts_lowercase_false() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("false")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("`false` must be accepted");
                assert!(!cfg.checkpoint_enabled);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_enabled_rejects_capitalized_true() {
        // Stdlib `FromStr for bool` is case-sensitive: only the exact
        // lowercase strings `"true"` and `"false"` parse. Pin this so
        // a regression that silently swaps in a case-insensitive
        // parser doesn't drift the contract out from under operators
        // who would then be relying on undocumented behavior.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("True")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`True` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_ENABLED"),
                    "error should name the env var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_enabled_rejects_uppercase_false() {
        // Same rationale as the capitalized-`True` test above —
        // stdlib `FromStr for bool` rejects `"FALSE"`.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("FALSE")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`FALSE` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_ENABLED"),
                    "error should name the env var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_enabled_rejects_yes_returns_err_naming_var() {
        // `yes` is the most common human-intuitive substitute for
        // `true` — rejecting it with a descriptive error is the
        // mainline UX value of this contract.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("yes")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`yes` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_ENABLED"),
                    "error should name LIVE_EDIT_CHECKPOINT_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_enabled_rejects_numeric_one_returns_err_naming_var() {
        // Numeric-form booleans (`0` / `1` / `2`) all fail
        // stdlib `FromStr for bool` — pin the rejection for `2`
        // specifically because the `0` / `1` case is the most
        // confusing for operators coming from shell scripts where
        // those values mean false/true.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("2")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`2` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_ENABLED"),
                    "error should name LIVE_EDIT_CHECKPOINT_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_enabled_rejects_empty_returns_err_naming_var() {
        // Empty-string input reaches the parser after trim, so it's
        // an empty `&str` — `bool::from_str` rejects it.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ENABLED", Some("")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("empty string must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_ENABLED"),
                    "error should name LIVE_EDIT_CHECKPOINT_ENABLED, got: {err}"
                );
            },
        );
    }

    // ---- t3.fgj.ya.99.eu: env-var whitespace trim on new checkpoint vars ----
    //
    // Mirrors `config_from_env_trims_whitespace_from_valkey_url`. Manifest
    // quoting (Helm values, k8s downward-API substitution, shell heredocs)
    // routinely introduces leading/trailing whitespace that operators
    // can't see. Without the trim, those values surface as opaque
    // parse failures with no hint at the root cause. Pin the trim
    // contract on the two new string-typed checkpoint env vars so any
    // regression that drops the trim fires here, not in production.

    #[test]
    fn config_from_env_trims_whitespace_from_checkpoint_adapter() {
        // `LIVE_EDIT_CHECKPOINT_ADAPTER` flows through `optional_parsed`,
        // which trims before `CheckpointAdapter::from_str` is invoked.
        // A value like `"  inmemory  "` from a quoted manifest must
        // parse as `InMemory`.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("  inmemory  ")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("whitespace-wrapped adapter value should parse");
                assert_eq!(
                    cfg.checkpoint_adapter,
                    CheckpointAdapter::InMemory,
                    "adapter must be trimmed so surrounding whitespace from manifest \
                     quoting doesn't reach CheckpointAdapter::from_str"
                );
            },
        );
    }

    // ---- t3.fgj.dp.io.a7: service-identity token env vars (SDD SR-6.1 / SR-6.2 / SR-6.4) ----
    //
    // The engineer Subtask t3.fgj.dp.io.b4 added three new fields to
    // `Config` plumbed through `Config::from_env`:
    //
    //   * `service_token_file: Option<PathBuf>` —
    //     `LIVE_EDIT_SERVICE_TOKEN_FILE`. Default `None`. Trimmed-empty
    //     also collapses to `None` so a manifest line like
    //     `LIVE_EDIT_SERVICE_TOKEN_FILE=" "` doesn't hand a meaningless
    //     path to the polling task. No scheme validation (filesystem
    //     path, not endpoint URL).
    //   * `service_token_poll_interval_secs: u64` —
    //     `LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS`. Default 15.
    //     Zero rejected at parse time because a zero-interval
    //     `tokio::time::interval` panics and a tight-loop poll would
    //     saturate the runtime.
    //   * `service_token_client_credentials_enabled: bool` —
    //     `LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED`. Default
    //     `false`. V1 forward-compat signal only; warn-helper consults
    //     it at startup.
    //
    // The all-defaults table-shaped test above pins the defaults; the
    // tests below pin the round-trip / rejection / unset-startup
    // behaviour case by case.

    #[test]
    fn config_from_env_service_token_file_round_trip() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_FILE",
                    Some("/var/run/oauth/access_token"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit service_token_file should round-trip into Config");
                assert_eq!(
                    cfg.service_token_file,
                    Some(PathBuf::from("/var/run/oauth/access_token")),
                    "service_token_file must round-trip from LIVE_EDIT_SERVICE_TOKEN_FILE \
                     so the ServiceTokenProvider polls the operator-configured sidecar path"
                );
            },
        );
    }

    #[test]
    fn config_from_env_service_token_poll_interval_secs_round_trip() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS", Some("30")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("explicit poll-interval secs should round-trip into Config");
                assert_eq!(
                    cfg.service_token_poll_interval_secs, 30,
                    "service_token_poll_interval_secs must round-trip from \
                     LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS"
                );
            },
        );
    }

    #[test]
    fn config_from_env_service_token_poll_interval_secs_zero_rejected() {
        // Mirrors `config_from_env_missing_userinfo_endpoint_returns_err_naming_var`
        // in shape: assert both that the value is rejected and that the
        // error string names the offending env var. A bare `Err(_)`
        // would let a future refactor that conflates this with another
        // env var slip through.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "zero poll interval must be rejected — \
                     tokio::time::interval panics on zero and a tight-loop poll \
                     would saturate the runtime",
                );
                assert!(
                    err.contains("LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS"),
                    "error should name LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_service_token_poll_interval_secs_non_numeric_rejected() {
        // `optional_parsed::<u64>` should reject a non-numeric value;
        // this complements the zero-rejection test and pins the
        // parse-error path independently of the post-parse zero check.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS",
                    Some("fifteen"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("non-numeric poll interval must be rejected at parse time");
                assert!(
                    err.contains("LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS"),
                    "error should name LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_service_token_client_credentials_enabled_round_trip() {
        // Pin parse behaviour for `true`, `false`, and unset. All three
        // live in one test because `temp_env::with_vars` scopes each
        // closure independently and the underlying behaviour is the
        // same `optional_parsed::<bool>` call as `authn_enabled` and
        // friends; what we're pinning is the *wiring* between the env
        // var name and the Config field.

        // `true` round-trip.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED",
                    Some("true"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("client_credentials_enabled=true should parse");
                assert!(
                    cfg.service_token_client_credentials_enabled,
                    "service_token_client_credentials_enabled must round-trip `true` \
                     so the startup warn-helper can branch on it"
                );
            },
        );

        // Explicit `false` round-trip — pinned separately from unset to
        // catch a regression that flipped the default.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED",
                    Some("false"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("client_credentials_enabled=false should parse");
                assert!(
                    !cfg.service_token_client_credentials_enabled,
                    "service_token_client_credentials_enabled must round-trip explicit `false`"
                );
            },
        );

        // Unset → default `false`. (The all-defaults test also pins
        // this; restated here so this round-trip test stands on its own
        // when read in isolation.)
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("unset client_credentials_enabled should fall back to default");
                assert!(
                    !cfg.service_token_client_credentials_enabled,
                    "unset LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED must default to false"
                );
            },
        );
    }

    #[test]
    fn config_from_env_service_token_file_unset_does_not_refuse_startup() {
        // `LIVE_EDIT_SERVICE_TOKEN_FILE` is required only with the
        // StorageApi checkpoint adapter. Under the default adapter
        // (`Disabled`) `Config::from_env` must NOT refuse startup when it is
        // unset: this test pins that the parse step does not gate every
        // service launch on the variable.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_TOKEN_FILE", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "Config::from_env must NOT refuse startup when service_token_file is unset \
                     and the checkpoint adapter is not StorageApi",
                );
                assert_eq!(
                    cfg.service_token_file, None,
                    "unset LIVE_EDIT_SERVICE_TOKEN_FILE must surface as None"
                );
            },
        );
    }

    #[test]
    fn config_from_env_service_token_file_trimmed_empty_collapses_to_none() {
        // The engineer's parse logic trims the value and collapses an
        // empty-after-trim result to `None`. A manifest line like
        // `LIVE_EDIT_SERVICE_TOKEN_FILE=" "` from Helm-quoted values
        // must NOT reach the polling task as a meaningless path. Pin
        // that contract so a future refactor that drops the trim or
        // filter surfaces here.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_SERVICE_TOKEN_FILE", Some("   ")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "whitespace-only service_token_file must parse to None, not refuse startup",
                );
                assert_eq!(
                    cfg.service_token_file, None,
                    "trimmed-empty LIVE_EDIT_SERVICE_TOKEN_FILE must collapse to None \
                     so the polling task isn't handed a meaningless path"
                );
            },
        );
    }

    #[test]
    fn config_from_env_trims_whitespace_from_checkpoint_local_disk_path() {
        // `LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH` is read via `env::var`
        // and trimmed inline in `Config::from_env`; the trimmed-empty
        // case is the existing `checkpoint_localdisk_rejects_whitespace_path`
        // test. This complements that by pinning the non-empty case —
        // the path itself must round-trip into the Config field
        // without leading/trailing whitespace, so downstream adapter
        // constructors don't have to defensively re-trim.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("localdisk")),
                (
                    "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                    Some("\t/var/lib/live-edit/checkpoints\n"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("trimmed local_disk_path should parse");
                assert_eq!(
                    cfg.checkpoint_local_disk_path.as_deref(),
                    Some("/var/lib/live-edit/checkpoints"),
                    "local_disk_path must be trimmed so manifest-quoting whitespace \
                     doesn't reach the LocalDisk adapter constructor"
                );
            },
        );
    }

    // ---- t3.tr5.t4.pi.rv: SR-1.4 Valkey-URL TLS-scheme guard ----
    //
    // These tests exercise `validate_valkey_tls_scheme(&str, tls_compiled: bool)`
    // — the pure, `cfg!`-free decision helper — *directly*, a knowing departure
    // from the test-writing guide's "do not test private helper functions
    // directly" / "do not test configuration parsing in isolation" bullets.
    // The departure is justified and load-bearing:
    //
    // (a) WHY THE PRIVATE HELPER IS TESTED DIRECTLY. Task 1 ships the
    //     `valkey-tls` feature ON by default, so in the shipped build the guard's
    //     reject branch is UNREACHABLE through the public `Config::from_env`
    //     API — `validate_valkey_tls_compiled` passes `cfg!(feature =
    //     "valkey-tls")` == `true`. The Task therefore mandates an injected
    //     `tls_compiled: bool` precisely so the reject branch can be executed
    //     here; the guide's usual "reach it through the public API" escape hatch
    //     does not apply because the public API can never reach it in this build.
    //
    // (b) `from_env` COVERAGE SHAPE. In the default (TLS-compiled) build the
    //     `from_env` layer gets ACCEPT-side coverage only (tests 8-11); the
    //     reject branch is covered at the helper level (tests 1, 5, 6, 7). The
    //     `#[cfg(not(feature = "valkey-tls"))]` mirrors of tests 8/9/11 close the
    //     other half: under `--no-default-features` they drive the reject branch
    //     END-TO-END through `Config::from_env`, the only such coverage anywhere
    //     in this Epic. The mirrors also pin the production `cfg!` call site — a
    //     hardcoded `true` in `validate_valkey_tls_compiled` in place of the real
    //     `cfg!` read would make every mirror fail — so that wiring is a tested
    //     property, not merely a code-review check.
    //
    // (c) SCHEME-LITERAL SOURCE. The TLS-scheme truth table mirrors fred's own
    //     detection: `fred-10.1.0/src/utils.rs:55-56` (`REDIS_TLS_SCHEME =
    //     "rediss"`, `VALKEY_TLS_SCHEME = "valkeys"`) and `:699`
    //     (`scheme.starts_with(REDIS_TLS_SCHEME) || starts_with(VALKEY_TLS_SCHEME)`).
    //     Cited so a future fred bump that moves the literals has a grep anchor.

    /// Every value fred treats as a TLS scheme. Detection is prefix-based,
    /// case-insensitive-in-effect (the `url` crate lowercases the scheme), and
    /// scheme-only — the near-miss non-TLS rows in `NON_TLS_URL_VALUES` guard
    /// the other direction.
    const TLS_SCHEME_URLS: &[&str] = &[
        "rediss://valkey.example.com:6380", // documented standalone form (docs/service/docs/backing-store.md)
        "rediss-cluster://valkey.example.com:10000", // documented managed form (docs/service/docs/backing-store-azure.md)
        "valkeys://valkey.example.com:6379", // VALKEY_TLS_SCHEME; the bypass absent from all repo docs
        "valkeys-cluster://valkey.example.com:6379", // Valkey-flavoured cluster form
        "rediss-sentinel://valkey.example.com:26379?sentinelServiceName=mymaster", // sentinel suffix still routes through parse_url -> url_uses_tls
        "valkeys-sentinel://valkey.example.com:26379?sentinelServiceName=mymaster", // Valkey x sentinel cell
        "REDISS://valkey.example.com:6380", // uppercase scheme; the url crate lowercases it before fred sees it
        "ValkeyS://valkey.example.com:6379", // mixed case, Valkey flavour
        "rediss+unix:///var/run/valkey.sock", // locks starts_with prefix semantics vs equality-against-two-literals; fail-closed is correct
    ];

    /// Everything the guard must never fire on. The near-miss rows (host or
    /// credential segment contains `rediss`/`valkeys` but the scheme is
    /// plaintext) stand between a sloppy `contains`-based guard and a
    /// self-inflicted startup outage on a legitimate plaintext deployment.
    const NON_TLS_URL_VALUES: &[&str] = &[
        "redis://127.0.0.1:6379",          // the binary default shape
        "redis-cluster://127.0.0.1:7000",  // the dev-valkey.sh cluster-up shape
        "valkey://127.0.0.1:6379", // plaintext Valkey flavour; must not be caught by a sloppy `valkey` prefix check
        "valkey-cluster://127.0.0.1:7000", // plaintext Valkey cluster flavour
        "redis-sentinel://127.0.0.1:26379?sentinelServiceName=mymaster", // plaintext sentinel
        "redis://rediss.example.com:6379", // near-miss: host contains `rediss`, scheme is plaintext
        "valkey://valkeys.internal:6379", // near-miss, Valkey flavour
        "redis://user:rediss-pw@127.0.0.1:6379", // near-miss in the credential segment
        "",               // empty; from_env accepts any string today, guard must not newly reject
        "localhost:6379", // bare host:port, not a URL
        "not a url",      // unparseable
        "http://example.com", // a scheme fred parses as centralized plaintext
    ];

    #[test]
    fn valkey_tls_scheme_guard_tls_scheme_without_tls_compiled_rejects() {
        for url in TLS_SCHEME_URLS {
            assert!(
                validate_valkey_tls_scheme(url, false).is_err(),
                "TLS scheme URL {url:?} must be rejected when no TLS feature is compiled"
            );
        }
    }

    #[test]
    fn valkey_tls_scheme_guard_tls_scheme_with_tls_compiled_accepts() {
        for url in TLS_SCHEME_URLS {
            assert!(
                validate_valkey_tls_scheme(url, true).is_ok(),
                "TLS scheme URL {url:?} must be accepted when the TLS feature is compiled \
                 (the guard never fires in the shipped build)"
            );
        }
    }

    #[test]
    fn valkey_tls_scheme_guard_non_tls_value_without_tls_compiled_accepts() {
        for url in NON_TLS_URL_VALUES {
            assert!(
                validate_valkey_tls_scheme(url, false).is_ok(),
                "non-TLS value {url:?} must be accepted even without a TLS feature — a sloppy \
                 substring match here would be a startup outage on a plaintext deployment"
            );
        }
    }

    #[test]
    fn valkey_tls_scheme_guard_non_tls_value_with_tls_compiled_accepts() {
        for url in NON_TLS_URL_VALUES {
            assert!(
                validate_valkey_tls_scheme(url, true).is_ok(),
                "non-TLS value {url:?} must be accepted with the TLS feature compiled too"
            );
        }
    }

    #[test]
    fn valkey_tls_scheme_guard_rejection_message_names_variable_and_tls() {
        let err = validate_valkey_tls_scheme("rediss://valkey.example.com:6380", false)
            .expect_err("TLS scheme without TLS compiled must reject");
        // Two substring assertions only — matches the existing
        // `err.contains("LIVE_EDIT_...")` precedent and keeps brittleness low.
        assert!(
            err.contains("LIVE_EDIT_VALKEY_URL"),
            "rejection must name the offending env-var, got: {err}"
        );
        assert!(
            err.contains("TLS"),
            "rejection must name TLS as the reason, got: {err}"
        );
    }

    #[test]
    fn valkey_tls_scheme_guard_rejection_message_never_echoes_credentials() {
        // Pins CLAUDE.md's token-secrecy invariant against a future edit that
        // interpolates the URL into the message for "better diagnostics".
        let err =
            validate_valkey_tls_scheme("rediss://admin:hunter2@valkey.example.com:6380", false)
                .expect_err("TLS scheme without TLS compiled must reject");
        assert!(
            !err.contains("hunter2"),
            "rejection must not echo the password, got: {err}"
        );
        assert!(
            !err.contains("admin"),
            "rejection must not echo the username, got: {err}"
        );
    }

    #[test]
    fn valkey_tls_scheme_guard_whitespace_wrapped_tls_url_without_tls_compiled_rejects() {
        // Defense-in-depth. `Config::from_env` already trims `valkey_url` inside
        // the `Config { ... }` literal (src/config.rs:827-829) BEFORE the
        // validator chain (src/config.rs:1127-1133) runs, so a from_env-level
        // test cannot observe whether the guard saw the trimmed value. The
        // helper must tolerate surrounding whitespace itself so a future
        // refactor that reorders trim-vs-guard cannot silently open the hole;
        // this is the only observable form of that trim contract given the
        // reject branch is unreachable from from_env in the default build.
        let wrapped = "  rediss://valkey.example.com:6380\n";
        assert!(
            validate_valkey_tls_scheme(wrapped, false).is_err(),
            "whitespace-wrapped TLS URL must still be rejected without a TLS feature"
        );
        assert!(
            validate_valkey_tls_scheme(wrapped, true).is_ok(),
            "whitespace-wrapped TLS URL must be accepted with the TLS feature compiled"
        );
    }

    // `Config::from_env` integration level (accept-side only in the default
    // build). Tests 8, 9, 11 assert `Ok` for a `rediss://` URL, which is true
    // only when the `valkey-tls` feature is compiled; under
    // `--no-default-features` the guard correctly rejects, so each is gated on
    // `#[cfg(feature = "valkey-tls")]` with a `#[cfg(not(...))]` reject mirror.

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_rediss_valkey_url_accepted_in_default_build() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("rediss:// URL must be accepted in the TLS build");
                assert_eq!(
                    cfg.valkey_url, "rediss://valkey.example.com:6380",
                    "valkey_url must round-trip verbatim"
                );
            },
        );
    }

    #[cfg(not(feature = "valkey-tls"))]
    #[test]
    fn config_from_env_rediss_valkey_url_rejected_without_tls_feature() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss://valkey.example.com:6380"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("rediss:// URL must be rejected end-to-end without a TLS feature");
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_URL"),
                    "from_env reject must name the env-var, got: {err}"
                );
                assert!(
                    err.contains("TLS"),
                    "from_env reject must name TLS, got: {err}"
                );
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_rediss_cluster_valkey_url_with_credentials_accepted() {
        // The real managed-service shape from docs/service/docs/backing-store-azure.md
        // § Access-key fallback.
        // Also proves the guard does not choke on a userinfo segment.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss-cluster://default:s3cr3t@cache.example.net:10000"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "rediss-cluster:// URL with credentials must be accepted in the TLS build",
                );
                assert_eq!(
                    cfg.valkey_url, "rediss-cluster://default:s3cr3t@cache.example.net:10000",
                    "valkey_url must round-trip verbatim, credentials included"
                );
            },
        );
    }

    #[cfg(not(feature = "valkey-tls"))]
    #[test]
    fn config_from_env_rediss_cluster_valkey_url_with_credentials_rejected_without_tls_feature() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("rediss-cluster://default:s3cr3t@cache.example.net:10000"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "rediss-cluster:// URL must be rejected end-to-end without a TLS feature",
                );
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_URL"),
                    "from_env reject must name the env-var, got: {err}"
                );
                assert!(
                    err.contains("TLS"),
                    "from_env reject must name TLS, got: {err}"
                );
                // The reject message must not echo the credential segment either.
                assert!(
                    !err.contains("s3cr3t"),
                    "from_env reject must not echo the password, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_plaintext_valkey_url_still_accepted_after_tls_guard() {
        // Pins that adding the SR-1.4 validator to the chain did not break the
        // default plaintext path. Feature-agnostic: `redis://` is never a TLS
        // scheme, so this holds in both builds.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_URL", Some("redis://127.0.0.1:6379")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("plaintext redis:// URL must remain accepted");
                assert_eq!(cfg.valkey_url, "redis://127.0.0.1:6379");
            },
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn config_from_env_trims_whitespace_from_rediss_valkey_url() {
        // Sibling of `config_from_env_trims_whitespace_from_valkey_url`, on the
        // TLS scheme. Accept-side only: valid in the TLS-compiled build.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("  rediss://valkey.test:6380\n"),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("trimmed rediss:// URL should parse");
                assert_eq!(
                    cfg.valkey_url, "rediss://valkey.test:6380",
                    "valkey_url must be trimmed so surrounding whitespace from manifest \
                     quoting doesn't reach the fred parser"
                );
            },
        );
    }

    #[cfg(not(feature = "valkey-tls"))]
    #[test]
    fn config_from_env_trims_whitespace_from_rediss_valkey_url_rejected_without_tls_feature() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_VALKEY_URL",
                    Some("  rediss://valkey.test:6380\n"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "trimmed rediss:// URL must still be rejected end-to-end without a TLS feature",
                );
                assert!(
                    err.contains("LIVE_EDIT_VALKEY_URL"),
                    "from_env reject must name the env-var, got: {err}"
                );
                assert!(
                    err.contains("TLS"),
                    "from_env reject must name TLS, got: {err}"
                );
            },
        );
    }

    // ══ b.dfh Epic 3 / SDD SR-8.1: LIVE_EDIT_CHECKPOINT_TAIL_* knobs ════════
    //
    // Four knobs gate stream-replay tail capture. Coverage below mirrors the
    // two established shapes in this file case-for-case:
    //   * boolean  → the LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED cluster (stdlib
    //     `FromStr for bool`: exactly `true` / `false`, every alias rejected)
    //   * sizing   → the LIVE_EDIT_LAYER_STREAM_MAXLEN cluster (default /
    //     explicit round-trip / non-numeric / negative / boundary trio)
    //
    // Everything here is pure `Config::from_env` parsing — no Valkey, no
    // checkpoint loop.

    // ── LIVE_EDIT_CHECKPOINT_TAIL_ENABLED (bool, default true) ──────────────

    #[test]
    fn config_from_env_default_checkpoint_tail_enabled_is_true() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert!(
                    cfg.checkpoint_tail_enabled,
                    "unset LIVE_EDIT_CHECKPOINT_TAIL_ENABLED must default to true — \
                     the default was promoted 2026-09-11 (b.3n1) after the dev soak, \
                     so an accidental flip back would silently disable tail capture \
                     fleet-wide"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_enabled_accepts_lowercase_true() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("true")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with explicit `true`");
                assert!(cfg.checkpoint_tail_enabled);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_enabled_accepts_lowercase_false() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("false")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with explicit `false`");
                assert!(!cfg.checkpoint_tail_enabled);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_enabled_rejects_capitalized_true() {
        // Stdlib `FromStr for bool` is case-sensitive. Pinned so a regression
        // that swaps in a permissive/case-insensitive parser cannot silently
        // widen the accepted vocabulary for a feature flag.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("True")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`True` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED"),
                    "error should name the env var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_enabled_rejects_uppercase_true() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("TRUE")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`TRUE` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED"),
                    "error should name the env var, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_enabled_rejects_yes_returns_err_naming_var() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("yes")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`yes` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_enabled_rejects_numeric_one_returns_err_naming_var() {
        // `1` is the single most likely fat-finger for a feature flag, and
        // silently accepting it as `true` would enable the feature on a
        // deployment that never asked for it.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("`1` must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_ENABLED, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_enabled_rejects_empty_returns_err_naming_var() {
        // An empty value is NOT treated as unset: `optional_parsed` only
        // defaults on an absent var, so `FOO=` reaches the parser and must
        // fail loudly rather than silently resolving to the default.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("empty string must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_ENABLED, got: {err}"
                );
            },
        );
    }

    // ── LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN (u32, default 8, zero rejected) ─

    #[test]
    fn config_from_env_checkpoint_tail_max_chain_default_is_8() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(
                    cfg.checkpoint_tail_max_chain, 8,
                    "unset LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN must default to K = 8 \
                     (SR-8.1: restore stays <= 9 artifact GETs)"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_chain_parses_explicit_u32() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN", Some("3")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should accept an explicit numeric checkpoint_tail_max_chain");
                assert_eq!(cfg.checkpoint_tail_max_chain, 3u32);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_chain_zero_rejected() {
        // K = 0 admits no tail at all: every eligible tick would refuse and
        // the feature would report itself enabled while being inert.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero chain bound must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN"),
                    "error must name the env var, got: {err}"
                );
                assert!(
                    err.contains("must be greater than 0"),
                    "error must state the constraint in the house message shape, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_chain_non_numeric_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN", Some("not-a-number")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "non-numeric LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN should fail parse",
                );
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_chain_negative_rejected() {
        // u32 parsing rejects negatives; this guards against the field being
        // silently widened to a signed integer.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN", Some("-1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("negative LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN, got: {err}"
                );
            },
        );
    }

    // ── LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES (u64, default 4096, zero
    //    rejected, inclusive 2^53-1 ceiling) ───────────────────────────────

    #[test]
    fn config_from_env_checkpoint_tail_max_entries_default_is_4096() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(
                    cfg.checkpoint_tail_max_entries, 4096,
                    "unset LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES must default to 4096, \
                     matching the LIVE_EDIT_LAYER_STREAM_MAXLEN default"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_entries_parses_explicit_u64() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", Some("2048")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should accept an explicit numeric checkpoint_tail_max_entries");
                assert_eq!(cfg.checkpoint_tail_max_entries, 2048u64);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_entries_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero entry cap must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES"),
                    "error must name the env var, got: {err}"
                );
                assert!(
                    err.contains("must be greater than 0"),
                    "error must state the constraint in the house message shape, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_entries_non_numeric_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES",
                    Some("not-a-number"),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "non-numeric LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES should fail parse",
                );
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_entries_negative_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", Some("-1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("negative LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_entries_at_lua_ceiling_accepted() {
        // 2^53 - 1 is on the safe side of the ceiling — the same inclusive
        // boundary the LIVE_EDIT_LAYER_STREAM_MAXLEN cluster pins, and for the
        // same reason: a tail's entry count is bounded by the stream sequence
        // span it covers, which both commit paths already bound here.
        let ceiling = (1u64 << 53) - 1;
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES",
                    Some(ceiling.to_string().as_str()),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("entry cap at 2^53-1 must be accepted (inclusive ceiling)");
                assert_eq!(cfg.checkpoint_tail_max_entries, ceiling);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_entries_above_lua_ceiling_rejected() {
        // 2^53 is the first value past the ceiling.
        let above_ceiling = 1u64 << 53;
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES",
                    Some(above_ceiling.to_string().as_str()),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("entry cap above 2^53-1 must be rejected at startup");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES"),
                    "error must name the env var, got: {err}"
                );
                assert!(
                    err.contains("2^53") || err.contains("9007199254740991"),
                    "error must reference the 2^53 ceiling so operators can locate \
                     the constraint, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_entries_u64_max_rejected() {
        // The worst-case fat-finger input: confirm the ceiling check fires
        // rather than the value being accepted whole.
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES",
                    Some(u64::MAX.to_string().as_str()),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("entry cap at u64::MAX must be rejected at startup");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES"),
                    "error must name the env var, got: {err}"
                );
            },
        );
    }

    // ── LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES (usize, default 67108864, zero
    //    rejected, one-record floor) ────────────────────────────────────────

    #[test]
    fn config_from_env_checkpoint_tail_max_bytes_default_is_64_mib() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(
                    cfg.checkpoint_tail_max_bytes, 67_108_864,
                    "unset LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES must default to 64 MiB"
                );
            },
        );
    }

    // ── b.f4f: subscriber admission knobs ────────────────────────────────────

    /// Env vars every `Config::from_env` test in this block must set, because
    /// `from_env` requires them.
    const REQUIRED_ENV: [(&str, Option<&str>); 2] = [
        (
            "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
            Some("http://auth.example.com"),
        ),
        (
            "LIVE_EDIT_USERINFO_ENDPOINT",
            Some("http://userinfo.example.com"),
        ),
    ];

    /// The shipped defaults for the subscriber admission knobs: the
    /// subscription cap, the catch-up slot cap, the capacity budget and the
    /// waiter cap.
    ///
    /// All are operator-visible contract (`docs/service/docs/configuration.md`,
    /// SDD §16),
    /// so a silent drift changes how a saturated pod behaves in production
    /// without any code review noticing. `None` for the waiter cap means "track
    /// the subscription cap".
    #[test]
    fn config_from_env_subscriber_admission_defaults() {
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS", None),
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS", None),
                ("LIVE_EDIT_SUBSCRIBER_MAX_SUBSCRIPTIONS", None),
                ("LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should succeed with required vars set");
                assert_eq!(
                    cfg.subscriber_max_subscriptions, 4096,
                    "subscriptions are capped at the memory-derived ceiling by default"
                );
                assert_eq!(
                    cfg.subscriber_max_concurrent_catchups, 8,
                    "unset LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS must default to 8 slots, \
                     the CPU-derived default: 4 per CPU at the chart's 2-CPU limit (SDD §16)"
                );
                assert_eq!(
                    cfg.subscriber_admission_wait_ms, 20_000,
                    "unset LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS must default to 20000 ms, the \
                     capacity budget derived from the tightest client retry window"
                );
                assert_eq!(
                    cfg.subscriber_admission_max_waiters, None,
                    "unset LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS must stay `None`, which \
                     `ValkeyClient::connect` resolves to the subscription cap"
                );
            },
        );
    }

    /// Zero is fail-fast expressed on the **wait**, and is accepted there.
    #[test]
    fn config_from_env_subscriber_admission_wait_zero_is_the_fail_fast_policy() {
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS", Some("0")),
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS", None),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("a zero admission wait is the fail-fast policy, not an error");
                assert_eq!(cfg.subscriber_admission_wait_ms, 0);
            },
        );
    }

    /// The subscription ceiling is derived from the pod memory limit (its old
    /// variable's refusal is in the W6 table,
    /// `config_from_env_refuses_each_removed_memory_budget_variable_naming_its_replacement`),
    /// so a slot setting leaves it at the reference pod's 4,096. The catch-up
    /// slot count stays a setting: an explicit value is read (deliberately
    /// not the default 8, so a parse that fell back to it could not pass),
    /// and `0`, an unlimited count the pod memory plan cannot charge, is
    /// refused.
    #[test]
    fn config_from_env_subscription_ceiling_is_derived_and_catch_up_slots_refuse_zero() {
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero slots cannot be charged");
                assert!(
                    err.contains("LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS")
                        && err.contains("must be greater than 0"),
                    "got {err}"
                );
            },
        );
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS", Some("16")),
            ],
            || {
                let cfg =
                    Config::from_env_with(PodMemorySource::Reference).expect("explicit caps parse");
                assert_eq!(cfg.subscriber_max_subscriptions, 4_096);
                assert_eq!(cfg.subscriber_max_concurrent_catchups, 16);
            },
        );
    }

    /// The slot count becomes a `tokio::sync::Semaphore` size at connect, and
    /// `Semaphore::new` panics above `Semaphore::MAX_PERMITS`. So a value above
    /// it is refused where it is parsed, naming the variable, instead of
    /// crashing the process at connect. `MAX_PERMITS` itself parses, and the
    /// pod memory plan then refuses the memory that many slots would take.
    #[test]
    fn config_from_env_catch_up_slots_above_the_semaphore_limit_are_rejected() {
        let max = tokio::sync::Semaphore::MAX_PERMITS;
        let over = (max + 1).to_string();
        let var = "LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS";
        temp_env::with_vars(
            [REQUIRED_ENV[0], REQUIRED_ENV[1], (var, Some(over.as_str()))],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("a cap above MAX_PERMITS is refused");
                assert_eq!(
                    err,
                    format!(
                        "invalid value for {var}: {over} is above the largest supported \
                         limit ({max})"
                    ),
                );
            },
        );
        let at_max = max.to_string();
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                (var, Some(at_max.as_str())),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("the plan cannot charge MAX_PERMITS slots");
                assert!(
                    err.contains("no pod memory limit can fit this configuration"),
                    "{var}={max} parses, and the pod memory plan refuses it: {err}"
                );
            },
        );
    }

    /// A zero **waiter cap** is rejected at startup rather than accepted.
    ///
    /// Accepting it would make a configured non-zero wait silently dead code:
    /// every saturated subscribe would be refused without ever queueing, while
    /// the wait reported itself as in effect. That is the same failure mode
    /// `checkpoint_tail_max_bytes`' zero-rejection exists to prevent, and the
    /// error must say so — an operator who wanted fail-fast has a correct way to
    /// ask for it, and the message names it.
    #[test]
    fn config_from_env_subscriber_admission_max_waiters_zero_rejected() {
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("a zero waiter cap must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS"),
                    "error must name the env var, got: {err}"
                );
                assert!(
                    err.contains("must be greater than 0"),
                    "error must state the constraint in the house message shape, got: {err}"
                );
                assert!(
                    err.contains("LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS=0"),
                    "the error must point the operator at the knob that DOES express fail-fast, \
                     or the rejection reads as 'this configuration is impossible' rather than \
                     'you are asking on the wrong knob'. Got: {err}"
                );
            },
        );
    }

    /// **An empty value means unset.** The chart leaves the variable out
    /// when the cap is unset, but an env file or a chart rendered before
    /// that (which emitted `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS: ""`
    /// for the default `0`) can still carry it empty, and such a pod must
    /// start with the default cap.
    ///
    /// It has to be an explicit guard because `optional_parsed` does not treat
    /// empty as unset: it only falls back on `env::var` returning `Err`, so an
    /// empty-but-present variable reaches `"".parse::<usize>()` and fails.
    /// Delete the `!raw.trim().is_empty()` condition and such a pod stops
    /// booting.
    #[test]
    fn config_from_env_subscriber_admission_max_waiters_empty_means_unset() {
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS", Some("")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                    "an empty LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS must be treated as \
                     unset, as an env file or an earlier chart can carry it empty",
                );
                assert_eq!(
                    cfg.subscriber_admission_max_waiters, None,
                    "empty must resolve the same as absent — `None`, which \
                     `ValkeyClient::connect` turns into the resolved pool size"
                );
            },
        );
        // Whitespace is the same case: a chart that renders a padded value, or a
        // hand-written env file, must not boot into a parse failure either.
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS", Some("   ")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("a whitespace-only waiter cap must be treated as unset");
                assert_eq!(cfg.subscriber_admission_max_waiters, None);
            },
        );
    }

    /// The **wait** has no such guard, and does not need one — but the asymmetry
    /// is worth pinning, because it is one chart edit away from mattering.
    ///
    /// `subscriber_admission_wait_ms` goes through `optional_parsed` directly, so
    /// an empty-but-present value is a startup ERROR rather than a fallback to
    /// 20000. That is safe today only because the chart renders this one
    /// UNCONDITIONALLY from a non-empty default
    /// (`LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS: "{{ .Values.valkey.subscriberAdmissionWaitMs }}"`,
    /// default `20000`), so `""` is not a path any shipped deployment takes.
    ///
    /// This test asserts the behaviour that EXISTS rather than the one the
    /// sibling above has. If someone makes the wait conditional the way the cap
    /// is — or sets the value to `null` — this test is the record that doing so
    /// requires adding the empty-means-unset guard too, and the failure it
    /// produces is a refusal to boot, not a silent default.
    #[test]
    fn config_from_env_subscriber_admission_wait_empty_is_a_startup_error_not_a_default() {
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS", Some("")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "an empty admission wait is a parse failure, not a fallback to the 20000 ms \
                     default — `optional_parsed` falls back only on an ABSENT variable. If this \
                     now succeeds, someone added an empty-means-unset guard: good, but update \
                     this test to say so",
                );
                assert!(
                    err.contains("LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS"),
                    "the error must name the variable so an operator can find it, got: {err}"
                );
            },
        );
    }

    /// An explicit positive cap is taken as given.
    #[test]
    fn config_from_env_subscriber_admission_max_waiters_parses_explicit_value() {
        temp_env::with_vars(
            [
                REQUIRED_ENV[0],
                REQUIRED_ENV[1],
                ("LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS", Some("64")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("an explicit positive cap must be accepted");
                assert_eq!(cfg.subscriber_admission_max_waiters, Some(64));
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_bytes_parses_explicit_usize() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", Some("1048576")),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("should accept an explicit numeric checkpoint_tail_max_bytes");
                assert_eq!(cfg.checkpoint_tail_max_bytes, 1_048_576usize);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_bytes_zero_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", Some("0")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("zero byte cap must be rejected");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES"),
                    "error must name the env var, got: {err}"
                );
                assert!(
                    err.contains("must be greater than 0"),
                    "error must state the constraint in the house message shape, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_bytes_non_numeric_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", Some("not-a-number")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                    "non-numeric LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES should fail parse",
                );
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES, got: {err}"
                );
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_bytes_negative_rejected() {
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", Some("-1")),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("negative LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES should fail parse");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES"),
                    "error should name LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES, got: {err}"
                );
            },
        );
    }

    // The byte cap carries a second, non-zero lower bound the other sizing
    // knobs do not: a one-record floor derived from the tail codec's own
    // framing (`TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE + 1`). Below it the
    // capture path's payload budget derives to zero, so no verbatim commit is
    // representable and every capture would refuse forever as
    // `refused_missing_bytes` — a silent permanent wedge that startup
    // rejection converts into a loud boot failure. The floor is spelled here
    // from the same imported constants the validation uses, so a codec framing
    // change moves both together; the `+ 1` is asserted by the adjacent
    // accept/reject pair straddling the exact boundary.

    #[test]
    fn config_from_env_checkpoint_tail_max_bytes_at_one_record_floor_accepted() {
        let floor = TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE + 1;
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES",
                    Some(floor.to_string().as_str()),
                ),
            ],
            || {
                let cfg = Config::from_env_with(PodMemorySource::Reference)
                    .expect("byte cap at exactly the one-record floor must be accepted");
                assert_eq!(cfg.checkpoint_tail_max_bytes, floor);
            },
        );
    }

    #[test]
    fn config_from_env_checkpoint_tail_max_bytes_below_one_record_floor_rejected() {
        // One byte under the floor — the envelope size exactly, at which the
        // derived payload budget is still zero.
        let below_floor = TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE;
        temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                (
                    "LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES",
                    Some(below_floor.to_string().as_str()),
                ),
            ],
            || {
                let err = Config::from_env_with(PodMemorySource::Reference)
                    .expect_err("byte cap below the one-record floor must be rejected at startup");
                assert!(
                    err.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES"),
                    "error must name the env var, got: {err}"
                );
                assert!(
                    err.contains(&(below_floor + 1).to_string()),
                    "error must state the floor so an operator can raise the value \
                     without reading the codec, got: {err}"
                );
            },
        );
    }

    // ── TailCaptureConfig::default() lockstep ───────────────────────────────

    #[test]
    fn tail_capture_config_default_matches_config_from_env_defaults() {
        // The four tail defaults are restated in several places across this
        // Epic (the parse blocks, this struct's hand-written `Default`, the
        // doc tables). `TailCaptureConfig::default()` is what fixture
        // construction sites use, so a drift between it and `Config::from_env`
        // would have every test in the Epic exercising a configuration no
        // deployment can actually boot with — silently, since both halves
        // would be internally consistent. This is the only automated check
        // that ties the two together.
        use crate::checkpoint::TailCaptureConfig;

        temp_env::with_vars(REQUIRED_VARS, || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("should succeed with required vars set");
            let fixture = TailCaptureConfig::default();

            assert_eq!(
                fixture.enabled, cfg.checkpoint_tail_enabled,
                "TailCaptureConfig::default().enabled drifted from the \
                 LIVE_EDIT_CHECKPOINT_TAIL_ENABLED default"
            );
            // `max_chain` widens u32 -> usize at the CheckpointLoopConfig
            // boundary, exactly as `retain_generations` does.
            assert_eq!(
                fixture.max_chain, cfg.checkpoint_tail_max_chain as usize,
                "TailCaptureConfig::default().max_chain drifted from the \
                 LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN default"
            );
            assert_eq!(
                fixture.max_entries, cfg.checkpoint_tail_max_entries,
                "TailCaptureConfig::default().max_entries drifted from the \
                 LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES default"
            );
            assert_eq!(
                fixture.max_bytes, cfg.checkpoint_tail_max_bytes,
                "TailCaptureConfig::default().max_bytes drifted from the \
                 LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES default"
            );

            // A derived `Default` would give 0/0/0 on the three sizing knobs —
            // precisely the values `from_env` rejects. Pin that the hand-written
            // impl is still in place, since the assertions above would keep
            // passing only for as long as the `from_env` defaults are non-zero.
            assert!(
                fixture.max_chain > 0 && fixture.max_entries > 0 && fixture.max_bytes > 0,
                "TailCaptureConfig::default() must not be `#[derive(Default)]`-shaped: \
                 zeros on the sizing knobs encode a configuration the service refuses \
                 to boot with"
            );
        });
    }

    // ══ Cross-field WARN: TAIL_MAX_ENTRIES vs LAYER_STREAM_MAXLEN (SR-8.1) ══
    //
    // TECHNIQUE NOTE (load-bearing). Other startup WARNs can fire beside the
    // tail WARN — `warn_layer_stream_memory_footprint` does on any count-only
    // config (LIVE_EDIT_LAYER_STREAM_MAX_BYTES=0: 4096 x 104_857_600 ~ 400 GB,
    // threshold 4 GiB) — so a global "no WARN was emitted" assertion is
    // confounded. Every case below therefore asserts on a **tail-specific
    // substring** of the tail WARN's message rather than on a global WARN
    // count — no pinning of LIVE_EDIT_MAX_LAYER_SIZE is needed, and the
    // footprint warn may fire freely alongside without perturbing the result.

    /// Distinctive substring of `warn_tail_entries_exceed_stream_retention`'s
    /// message. Deliberately a phrase from the tail warn only — the footprint
    /// warn mentions `LIVE_EDIT_LAYER_STREAM_MAXLEN` too, so matching on the
    /// env-var name alone would not discriminate.
    const TAIL_RETENTION_WARN_MARKER: &str = "exceeds live-stream retention";

    /// A `MakeWriter` that captures everything written into a shared buffer.
    ///
    /// Duplicated (~25 lines) from the same pattern in
    /// `src/grpc/auth_interceptor.rs::tests` / `src/auth/service_token.rs::tests`
    /// / `tests/misc/service_token_startup_warns.rs`: a shared test helper
    /// would have to live in the non-test tree or in `test-support`, which is
    /// a worse trade than the duplication for a self-contained 25 lines.
    #[derive(Clone)]
    struct CapturingWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl CapturingWriter {
        fn new() -> (Self, std::sync::Arc<std::sync::Mutex<Vec<u8>>>) {
            let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            (Self(buf.clone()), buf)
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
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

    /// Run `f` with a WARN-level `fmt` subscriber installed as the
    /// thread-local default, returning `(f`'s value, captured output)`.
    ///
    /// `with_default` scopes the dispatcher to this thread, so parallel test
    /// threads cannot cross-contaminate. ANSI is off so substring assertions
    /// are not defeated by colour escapes.
    fn capture_warn_output<T>(f: impl FnOnce() -> T) -> (T, String) {
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let value = tracing::subscriber::with_default(subscriber, f);
        let text = String::from_utf8(captured.lock().unwrap().clone())
            .expect("captured log output must be valid UTF-8");
        (value, text)
    }

    /// Number of tail-retention WARN events in `output`.
    fn tail_warn_count(output: &str) -> usize {
        output.matches(TAIL_RETENTION_WARN_MARKER).count()
    }

    #[test]
    fn warn_tail_entries_exceed_stream_retention_warns_once_when_over_threshold() {
        // The point of this case is the *warn, not reject* distinction: a
        // half-migrated pairing must still boot.
        let (result, output) = capture_warn_output(|| {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", Some("8192")),
                    ("LIVE_EDIT_LAYER_STREAM_MAXLEN", Some("4096")),
                ],
                || Config::from_env_with(PodMemorySource::Reference),
            )
        });

        let cfg = result.expect(
            "an entry cap above stream retention must WARN, not reject — an operator \
             mid-way through raising both knobs must not be locked out of booting",
        );
        assert_eq!(cfg.checkpoint_tail_max_entries, 8192);
        assert_eq!(cfg.layer_stream_maxlen, 4096);

        assert_eq!(
            tail_warn_count(&output),
            1,
            "exactly one tail-retention WARN expected, got output: {output}"
        );

        // Scope the content assertions to the tail WARN's own line: the
        // footprint warn also carries a `layer_stream_maxlen` field, so an
        // unscoped `contains` would not prove the tail warn carried it.
        let line = output
            .lines()
            .find(|l| l.contains(TAIL_RETENTION_WARN_MARKER))
            .expect("the tail WARN line must be present");
        assert!(
            line.contains("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES"),
            "the WARN must name the tail knob so the operator knows which value to \
             lower, got: {line}"
        );
        assert!(
            line.contains("LIVE_EDIT_LAYER_STREAM_MAXLEN"),
            "the WARN must name the retention knob so the operator knows the other \
             half of the pairing, got: {line}"
        );
        assert!(
            line.contains("checkpoint_tail_max_entries=8192"),
            "the WARN must carry the configured entry cap as a structured field, got: {line}"
        );
        assert!(
            line.contains("layer_stream_maxlen=4096"),
            "the WARN must carry the effective retention depth as a structured field, \
             got: {line}"
        );
    }

    #[test]
    fn warn_tail_entries_exceed_stream_retention_silent_at_equality() {
        // Boundary pin: the condition is `>` (strictly greater), so an exactly
        // matched pairing — the shipped default relationship — is consistent
        // and must stay silent.
        let (result, output) = capture_warn_output(|| {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", Some("4096")),
                    ("LIVE_EDIT_LAYER_STREAM_MAXLEN", Some("4096")),
                ],
                || Config::from_env_with(PodMemorySource::Reference),
            )
        });
        result.expect("an equal pairing must be accepted");
        assert_eq!(
            tail_warn_count(&output),
            0,
            "entries == maxlen is consistent and must not warn, got output: {output}"
        );
    }

    #[test]
    fn warn_tail_entries_exceed_stream_retention_silent_when_below_retention() {
        let (result, output) = capture_warn_output(|| {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", Some("2048")),
                    ("LIVE_EDIT_LAYER_STREAM_MAXLEN", Some("4096")),
                ],
                || Config::from_env_with(PodMemorySource::Reference),
            )
        });
        result.expect("an entry cap below retention must be accepted");
        assert_eq!(
            tail_warn_count(&output),
            0,
            "entries < maxlen is consistent and must not warn, got output: {output}"
        );
    }

    #[test]
    fn warn_tail_entries_exceed_stream_retention_silent_at_stock_defaults() {
        // The engineer pinned the tail-entries default TO the retention default
        // specifically so the shipped configuration is quiet. Pin that claim:
        // a future bump to either default alone would start warning every
        // operator at boot for a pairing they never chose.
        let (result, output) = capture_warn_output(|| {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", None),
                    ("LIVE_EDIT_LAYER_STREAM_MAXLEN", None),
                ],
                || Config::from_env_with(PodMemorySource::Reference),
            )
        });
        result.expect("stock defaults must be accepted");
        assert_eq!(
            tail_warn_count(&output),
            0,
            "the shipped defaults must not emit the tail-retention WARN, got output: {output}"
        );
    }

    #[test]
    fn warn_tail_entries_exceed_stream_retention_fires_even_when_tail_disabled() {
        // Deliberately NOT gated on LIVE_EDIT_CHECKPOINT_TAIL_ENABLED: this is
        // a config-consistency warning, the operator can flip the feature flag
        // at any time, and gating would hide the inconsistency until exactly
        // the moment it starts costing coverage. Pinned because the opposite
        // choice — warn only when enabled — is the intuitive one and would be
        // an easy "cleanup" for a later reader to make.
        let (result, output) = capture_warn_output(|| {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some("false")),
                    ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", Some("8192")),
                    ("LIVE_EDIT_LAYER_STREAM_MAXLEN", Some("4096")),
                ],
                || Config::from_env_with(PodMemorySource::Reference),
            )
        });
        let cfg = result.expect("the pairing must still be accepted with the flag off");
        assert!(!cfg.checkpoint_tail_enabled, "the flag must be off here");
        assert_eq!(
            tail_warn_count(&output),
            1,
            "the cross-field WARN fires unconditionally, not only when the feature \
             flag is on, got output: {output}"
        );
    }

    // ══ b.yvh: the relay byte budget's two startup WARNs ══
    //
    // `warn_tail_bytes_exceed_stream_retention` (a tail's byte cap above the
    // stream's byte budget) and `warn_layer_stream_memory_footprint` (the
    // worst-case per-object footprint, now bounded by the budget when it is
    // on). Each case counts its own WARN by a phrase only that WARN carries,
    // per the TECHNIQUE NOTE above.

    /// Distinctive substring of `warn_tail_bytes_exceed_stream_retention`.
    const TAIL_BYTES_WARN_MARKER: &str = "exceeds the live stream's byte budget";
    /// Distinctive substring of `warn_layer_stream_memory_footprint`.
    const FOOTPRINT_WARN_MARKER: &str = "worst-case per-object live-stream footprint is large";

    /// `Config::from_env` with the required vars plus the four sizing knobs
    /// the two WARNs read (`None` = unset), and the WARN output.
    fn from_env_with_retention_knobs(
        tail_max_bytes: Option<&str>,
        stream_max_bytes: Option<&str>,
        stream_maxlen: Option<&str>,
        max_layer_size: Option<&str>,
    ) -> (Result<Config, String>, String) {
        capture_warn_output(|| {
            temp_env::with_vars(
                [
                    (
                        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                        Some("http://auth.example.com"),
                    ),
                    (
                        "LIVE_EDIT_USERINFO_ENDPOINT",
                        Some("http://userinfo.example.com"),
                    ),
                    ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", tail_max_bytes),
                    ("LIVE_EDIT_LAYER_STREAM_MAX_BYTES", stream_max_bytes),
                    ("LIVE_EDIT_LAYER_STREAM_MAXLEN", stream_maxlen),
                    ("LIVE_EDIT_MAX_LAYER_SIZE", max_layer_size),
                ],
                || Config::from_env_with(PodMemorySource::Reference),
            )
        })
    }

    #[test]
    fn warn_tail_bytes_exceed_stream_retention_warns_once_when_the_tail_cap_is_above_the_budget() {
        // 64 MiB tail cap over a 32 MiB budget: a tail can never carry more
        // relay bytes than the stream retains. A WARN, not a rejection.
        let (result, output) =
            from_env_with_retention_knobs(Some("67108864"), Some("33554432"), None, None);
        let cfg = result.expect("a tail cap above the byte budget must WARN, not reject");
        assert_eq!(cfg.layer_stream_max_bytes, 33_554_432);
        assert_eq!(
            output.matches(TAIL_BYTES_WARN_MARKER).count(),
            1,
            "exactly one tail-bytes WARN expected, got output: {output}"
        );
        let line = output
            .lines()
            .find(|l| l.contains(TAIL_BYTES_WARN_MARKER))
            .expect("the tail-bytes WARN line must be present");
        for wanted in [
            "LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES",
            "LIVE_EDIT_LAYER_STREAM_MAX_BYTES",
            "checkpoint_tail_max_bytes=67108864",
            "layer_stream_max_bytes=33554432",
        ] {
            assert!(
                line.contains(wanted),
                "the WARN must carry {wanted:?}, got: {line}"
            );
        }
    }

    #[test]
    fn warn_tail_bytes_exceed_stream_retention_silent_at_defaults_at_equality_and_budget_off() {
        for (tail, budget, why) in [
            (
                None,
                None,
                "the shipped defaults (64 MiB tail, 128 MiB budget)",
            ),
            (Some("33554432"), Some("33554432"), "an equal pairing"),
            (Some("67108864"), Some("0"), "the byte budget off"),
        ] {
            let (result, output) = from_env_with_retention_knobs(tail, budget, None, None);
            result.unwrap_or_else(|e| panic!("{why} must be accepted: {e}"));
            assert_eq!(
                output.matches(TAIL_BYTES_WARN_MARKER).count(),
                0,
                "{why} must not emit the tail-bytes WARN, got output: {output}"
            );
        }
    }

    #[test]
    fn warn_layer_stream_memory_footprint_silent_at_defaults() {
        // min(4096 × 100 MiB, 128 MiB + 2 × 100 MiB) ≈ 328 MiB, under 4 GiB.
        let (result, output) = from_env_with_retention_knobs(None, None, None, None);
        result.expect("stock defaults must be accepted");
        assert_eq!(
            output.matches(FOOTPRINT_WARN_MARKER).count(),
            0,
            "the shipped defaults must not emit the footprint WARN, got output: {output}"
        );
    }

    #[test]
    fn warn_layer_stream_memory_footprint_with_the_budget_off_is_the_count_bound_as_before() {
        // Count-only retention: 4096 × 100 MiB ≈ 400 GB, the pre-budget WARN.
        let (result, output) = from_env_with_retention_knobs(None, Some("0"), None, None);
        result.expect("the budget off must be accepted");
        assert_eq!(
            output.matches(FOOTPRINT_WARN_MARKER).count(),
            1,
            "with the budget off the count bound alone applies and must warn, got: {output}"
        );
        // And silent when the count bound alone is small: 16 × 1 MiB.
        let (result, output) =
            from_env_with_retention_knobs(None, Some("0"), Some("16"), Some("1048576"));
        result.expect("a small count bound must be accepted");
        assert_eq!(
            output.matches(FOOTPRINT_WARN_MARKER).count(),
            0,
            "16 × 1 MiB is far under the threshold, got: {output}"
        );
    }

    #[test]
    fn warn_layer_stream_memory_footprint_uses_the_smaller_bound_with_the_budget_on() {
        // An 8 GiB budget: min(400 GB, 8 GiB + 200 MiB) is above 4 GiB.
        let (result, output) = from_env_with_retention_knobs(None, Some("8589934592"), None, None);
        result.expect("a large budget must be accepted");
        assert_eq!(
            output.matches(FOOTPRINT_WARN_MARKER).count(),
            1,
            "a byte bound above 4 GiB must warn, got: {output}"
        );
        // An entry cap of 0 is no cap: the byte bound alone applies, and at the
        // default budget it is under the threshold.
        let (result, output) = from_env_with_retention_knobs(None, None, Some("0"), None);
        result.expect("MAXLEN 0 with the default budget must be accepted");
        assert_eq!(
            output.matches(FOOTPRINT_WARN_MARKER).count(),
            0,
            "MAXLEN 0 leaves the default byte bound (≈ 328 MiB), under the threshold, got: \
             {output}"
        );
    }

    // ── LIVE_EDIT_EXT_VALUE_VERIFY (enforce | observe, default enforce) ─────
    //
    // The upload content check's mode (SDD §11.1). `enforce` is the client
    // contract and the default; `observe` is a rollout relaxation that stores
    // what fails the check, so it must be chosen deliberately: any other value,
    // including an empty one, fails startup, and selecting it logs a WARN.

    /// `Config::from_env` with the required vars plus `value` for
    /// `LIVE_EDIT_EXT_VALUE_VERIFY` (`None` = unset), and the WARN output.
    fn from_env_with_ext_value_verify(value: Option<&str>) -> (Result<Config, String>, String) {
        capture_warn_output(|| {
            temp_env::with_vars(
                [
                    REQUIRED_VARS[0],
                    REQUIRED_VARS[1],
                    REQUIRED_VARS[2],
                    ("LIVE_EDIT_EXT_VALUE_VERIFY", value),
                ],
                || Config::from_env_with(PodMemorySource::Reference),
            )
        })
    }

    /// Distinctive opening of `warn_ext_value_verify_observe`'s message.
    const EXT_VALUE_VERIFY_OBSERVE_WARN_MARKER: &str = "LIVE_EDIT_EXT_VALUE_VERIFY=observe:";

    #[test]
    fn config_from_env_ext_value_verify_unset_defaults_to_enforce() {
        let (result, _) = from_env_with_ext_value_verify(None);
        assert_eq!(
            result.expect("unset must be accepted").ext_value_verify,
            ExtValueVerifyMode::Enforce,
            "unset LIVE_EDIT_EXT_VALUE_VERIFY must default to enforce — the client contract; a \
             default of observe would store every value that fails the content check"
        );
    }

    #[test]
    fn config_from_env_ext_value_verify_parses_either_mode_case_insensitively() {
        for (raw, expected) in [
            ("observe", ExtValueVerifyMode::Observe),
            ("OBSERVE", ExtValueVerifyMode::Observe),
            ("Enforce", ExtValueVerifyMode::Enforce),
        ] {
            let (result, _) = from_env_with_ext_value_verify(Some(raw));
            assert_eq!(
                result
                    .unwrap_or_else(|e| panic!("{raw:?} must be accepted, got: {e}"))
                    .ext_value_verify,
                expected,
                "LIVE_EDIT_EXT_VALUE_VERIFY={raw:?}"
            );
        }
    }

    #[test]
    fn config_from_env_ext_value_verify_rejects_empty_and_unknown_values_naming_both_modes() {
        for raw in ["", "audit"] {
            let (result, _) = from_env_with_ext_value_verify(Some(raw));
            let err = result
                .err()
                .unwrap_or_else(|| panic!("LIVE_EDIT_EXT_VALUE_VERIFY={raw:?} must fail startup"));
            assert!(
                err.contains("LIVE_EDIT_EXT_VALUE_VERIFY"),
                "the error must name the variable, got: {err}"
            );
            assert!(
                err.contains("enforce, observe"),
                "the error must name the accepted values, got: {err}"
            );
        }
    }

    #[test]
    fn warn_ext_value_verify_observe_fires_once_under_observe() {
        let (result, output) = from_env_with_ext_value_verify(Some("observe"));
        result.expect("observe must be accepted, with a WARN rather than a rejection");
        assert_eq!(
            output.matches(EXT_VALUE_VERIFY_OBSERVE_WARN_MARKER).count(),
            1,
            "observe stores values that fail the content check, so selecting it must log \
             exactly one startup WARN, got output: {output}"
        );
    }

    #[test]
    fn warn_ext_value_verify_observe_is_silent_under_enforce() {
        for value in [None, Some("enforce")] {
            let (result, output) = from_env_with_ext_value_verify(value);
            result.expect("enforce must be accepted");
            assert!(
                !output.contains(EXT_VALUE_VERIFY_OBSERVE_WARN_MARKER),
                "enforce ({value:?}) must not log the observe WARN, got output: {output}"
            );
        }
    }

    // ---- b.yuo: the external-value release budget ----

    /// `Config::from_env` with the required vars plus `extra`, the two
    /// release-budget variables pinned unset unless `extra` sets them.
    fn from_env_with_release_budget(extra: &[(&str, Option<&str>)]) -> Result<Config, String> {
        let mut vars: Vec<(&str, Option<&str>)> = REQUIRED_VARS.to_vec();
        vars.push(("LIVE_EDIT_VALKEY_MEMORY_BYTES", None));
        vars.push(("LIVE_EDIT_EXT_RELEASE_MAX_BYTES", None));
        vars.push(("LIVE_EDIT_MAX_LAYER_SIZE", None));
        for &(name, value) in extra {
            vars.retain(|(n, _)| *n != name);
            vars.push((name, value));
        }
        // A 16 GiB pod: a 1 GiB LIVE_EDIT_MAX_LAYER_SIZE needs more memory
        // than the reference pod's for its upload terms.
        temp_env::with_vars(vars, || {
            Config::from_env_with(PodMemorySource::Fixed(16 << 30))
        })
    }

    /// Distinctive phrase of both fallback lines `log_ext_release_budget_startup`
    /// logs when `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` is unset and the store's
    /// memory gave no budget — no usable `maxmemory`, and
    /// `LIVE_EDIT_VALKEY_MEMORY_BYTES` unset or no cluster map to divide it by
    /// (b.1jt: INFO or WARN by reason).
    const RELEASE_BUDGET_FALLBACK_MARKER: &str =
        "external-value release budget is the fallback, min(8 x";

    /// Distinctive phrase of the fallback line's WARN form: the store's
    /// `maxmemory` could not be used because of a fault.
    const RELEASE_BUDGET_FALLBACK_WARN_MARKER: &str = "could not be used because of a fault";

    /// Distinctive phrase of the fallback line's INFO form: the store does not
    /// report its size (expected on a managed store).
    const RELEASE_BUDGET_FALLBACK_INFO_MARKER: &str = "the store does not report its size";

    /// Distinctive phrase of the WARN it logs when the budget is disabled.
    const RELEASE_BUDGET_DISABLED_WARN_MARKER: &str = "disables the external-value release budget";

    /// T12 (b.1jt): unset, the budget is the fallback
    /// `min(8 × LIVE_EDIT_MAX_LAYER_SIZE, 100 MB)`, labelled `fallback`: the
    /// default 100 MiB layer size gives the 100 MB cap (one eighth of a 1 GB
    /// Azure Managed Redis `Balanced_B1`'s usable 0.8 GB), a 1 MiB layer size
    /// stays below the cap at 8 MiB, and a 1 GiB layer size is capped too.
    ///
    /// **Fails on the build before b.1jt**: the fallback was the uncapped
    /// `8 × LIVE_EDIT_MAX_LAYER_SIZE`, 838,860,800 at the default.
    #[test]
    fn release_budget_unset_falls_back_to_eight_max_uploads_capped_at_100_mb() {
        let cfg = from_env_with_release_budget(&[]).expect("the defaults must load");
        assert_eq!(
            (cfg.valkey_memory_bytes, cfg.ext_release_max_bytes),
            (None, None)
        );
        assert_eq!(
            cfg.ext_release_budget(),
            ExtReleaseBudget {
                bytes: 100_000_000,
                source: ExtReleaseBudgetSource::Fallback,
            },
            "the default layer size's 8 × 104,857,600 is capped at 100 MB"
        );
        assert_eq!(cfg.ext_release_budget().source.label(), "fallback");
        assert_eq!(EXT_RELEASE_BUDGET_FALLBACK_CAP_BYTES, 100_000_000);

        for (max_layer_size, expected) in [("1048576", 8 * 1_048_576), ("1073741824", 100_000_000)]
        {
            let cfg =
                from_env_with_release_budget(&[("LIVE_EDIT_MAX_LAYER_SIZE", Some(max_layer_size))])
                    .expect("the max layer size must load");
            assert_eq!(
                cfg.ext_release_budget(),
                ExtReleaseBudget {
                    bytes: expected,
                    source: ExtReleaseBudgetSource::Fallback,
                },
                "LIVE_EDIT_MAX_LAYER_SIZE={max_layer_size}"
            );
        }
    }

    /// b.1jt, the startup half: the fallback line's level follows why the
    /// store's `maxmemory` was unusable. A store with no `maxmemory` field —
    /// the expected case on Azure Managed Redis — logs exactly one INFO and no
    /// WARN; `maxmemory:0`, a failed or denied `INFO`, and a fallback with no
    /// reason recorded log one WARN. Each line names the reason and `source`
    /// `fallback`.
    ///
    /// **Fails on the build before b.1jt**: every fallback WARNed, whatever
    /// the reason.
    #[test]
    fn the_release_budget_fallback_logs_info_for_an_absent_field_and_warn_for_a_fault() {
        use crate::storage::store_memory::{StoreMemoryProblem, UnusableReason};
        let cfg = from_env_with_release_budget(&[]).expect("the defaults must load");
        let budget = cfg.ext_release_budget();

        let absent = StoreMemoryProblem::Unusable(UnusableReason::FieldAbsent {
            primary: "the store".to_owned(),
        });
        let ((), warns) =
            capture_warn_output(|| cfg.log_ext_release_budget_startup(&budget, Some(&absent)));
        assert!(
            warns.is_empty(),
            "a store that does not report its size is expected, not a fault: no WARN; got \
             {warns}"
        );
        let ((), infos) =
            capture_info_output(|| cfg.log_ext_release_budget_startup(&budget, Some(&absent)));
        assert_eq!(infos.lines().count(), 1, "one startup line; got {infos}");
        assert!(infos.contains(" INFO "), "{infos}");
        assert_eq!(
            infos.matches(RELEASE_BUDGET_FALLBACK_INFO_MARKER).count(),
            1,
            "{infos}"
        );
        assert!(
            infos.contains("reports no maxmemory field") && infos.contains("source=\"fallback\""),
            "the INFO names why and the source; got {infos}"
        );

        let faults = [
            Some(StoreMemoryProblem::Unusable(UnusableReason::ZeroLimit {
                primary: "the store".to_owned(),
            })),
            Some(StoreMemoryProblem::ReadFailed(
                "NOPERM this user has no permissions to run the 'info' command".to_owned(),
            )),
            None,
        ];
        for problem in &faults {
            let ((), output) = capture_info_output(|| {
                cfg.log_ext_release_budget_startup(&budget, problem.as_ref())
            });
            assert_eq!(
                output.lines().count(),
                1,
                "{problem:?}: one line; got {output}"
            );
            assert!(
                output.contains(" WARN "),
                "{problem:?}: a fault WARNs; got {output}"
            );
            assert_eq!(
                output.matches(RELEASE_BUDGET_FALLBACK_WARN_MARKER).count(),
                1,
                "{problem:?}: {output}"
            );
            assert!(
                output.contains(RELEASE_BUDGET_FALLBACK_MARKER),
                "{problem:?}: {output}"
            );
        }
        let ((), output) =
            capture_info_output(|| cfg.log_ext_release_budget_startup(&budget, None));
        assert!(
            output.contains("reason=no reason recorded"),
            "a fallback with no reason says so; got {output}"
        );
    }

    /// With `LIVE_EDIT_VALKEY_MEMORY_BYTES` set, the fallback is still
    /// possible — no cluster map to divide the setting by (the routing table
    /// not known yet, or no primary) — and its WARN names the reason and the
    /// setting it could not use.
    #[test]
    fn the_release_budget_fallback_warns_with_the_store_total_set_and_no_map() {
        use crate::storage::store_memory::{StoreMemoryProblem, UnusableReason};
        let cfg =
            from_env_with_release_budget(&[("LIVE_EDIT_VALKEY_MEMORY_BYTES", Some("12000000000"))])
                .expect("a store total must load");
        let budget = cfg.ext_release_budget();
        assert_eq!(
            budget.source,
            ExtReleaseBudgetSource::Fallback,
            "precondition"
        );
        for reason in [UnusableReason::RoutingUnknown, UnusableReason::NoPrimary] {
            let problem = StoreMemoryProblem::Unusable(reason);
            let ((), output) =
                capture_info_output(|| cfg.log_ext_release_budget_startup(&budget, Some(&problem)));
            assert_eq!(
                output.lines().count(),
                1,
                "{problem}: one line; got {output}"
            );
            assert!(
                output.contains(" WARN ")
                    && output.contains(RELEASE_BUDGET_FALLBACK_WARN_MARKER)
                    && output.contains(&problem.to_string())
                    && output.contains("valkey_memory_bytes=12000000000"),
                "{problem}: got {output}"
            );
        }
    }

    /// With the store's total memory set, the configuration alone still
    /// answers the fallback — the budget derived from the setting needs the
    /// store's cluster map and loses to a store-reported `maxmemory`, so the
    /// store-memory resolution answers it — and that derived budget's startup
    /// line (T9) is one INFO, `source` `override`, naming the map it was
    /// derived from: the shards (map entries) and the endpoints apart, the
    /// per-shard bytes and the total, with no WARN. The map is the 24 GB
    /// cache measured on 2026-10-13: 8 entries of 2,048 slots on 2 endpoints.
    #[test]
    fn a_budget_derived_from_the_store_total_logs_its_cluster_map_at_startup() {
        use crate::storage::store_memory::memory_bytes_budget;
        let cfg =
            from_env_with_release_budget(&[("LIVE_EDIT_VALKEY_MEMORY_BYTES", Some("24000000000"))])
                .expect("a store memory must load");
        assert_eq!(cfg.valkey_memory_bytes, Some(24_000_000_000));
        assert_eq!(
            cfg.ext_release_budget().source,
            ExtReleaseBudgetSource::Fallback,
            "the configuration alone never answers the setting's arm"
        );

        let ranges: Vec<(u16, u16)> = (0..8u16).map(|i| (i * 2048, i * 2048 + 2047)).collect();
        let (bytes, derivation) =
            memory_bytes_budget(&ranges, 24_000_000_000).expect("eight ranges give a budget");
        let budget = ExtReleaseBudget {
            bytes,
            source: ExtReleaseBudgetSource::ValkeyMemory {
                derivation,
                endpoints: 2,
            },
        };
        let ((), output) =
            capture_info_output(|| cfg.log_ext_release_budget_startup(&budget, None));
        assert_eq!(output.lines().count(), 1, "one startup line; got {output}");
        assert!(output.contains(" INFO "), "{output}");
        for field in [
            "source=\"override\"",
            "ext_release_max_bytes=300000000",
            "shards=8",
            "endpoints=2",
            "per_shard_bytes=2400000000",
            "total_bytes=24000000000",
        ] {
            assert!(
                output.contains(field),
                "{field}: the line names the cluster map the budget was derived from; got \
                 {output}"
            );
        }
        assert!(!output.contains(RELEASE_BUDGET_FALLBACK_MARKER), "{output}");
    }

    /// An explicit budget is the configuration's answer even with the store
    /// total (`LIVE_EDIT_VALKEY_MEMORY_BYTES`) set, and `0` disables the
    /// budget with a startup WARN.
    #[test]
    fn an_explicit_release_budget_is_honoured_and_zero_disables_it() {
        let cfg = from_env_with_release_budget(&[
            ("LIVE_EDIT_VALKEY_MEMORY_BYTES", Some("3221225472")),
            ("LIVE_EDIT_EXT_RELEASE_MAX_BYTES", Some("123456789")),
        ])
        .expect("an explicit budget must load");
        assert_eq!(
            cfg.ext_release_budget(),
            ExtReleaseBudget {
                bytes: 123_456_789,
                source: ExtReleaseBudgetSource::Explicit,
            }
        );

        let cfg = from_env_with_release_budget(&[("LIVE_EDIT_EXT_RELEASE_MAX_BYTES", Some("0"))])
            .expect("0 must load: it is the documented off switch");
        assert_eq!(
            cfg.ext_release_budget(),
            ExtReleaseBudget {
                bytes: 0,
                source: ExtReleaseBudgetSource::Explicit,
            }
        );
        let ((), output) = capture_warn_output(|| {
            cfg.log_ext_release_budget_startup(&cfg.ext_release_budget(), None)
        });
        assert_eq!(
            output.matches(RELEASE_BUDGET_DISABLED_WARN_MARKER).count(),
            1,
            "a disabled budget must WARN once; got {output}"
        );
        assert!(!output.contains(RELEASE_BUDGET_FALLBACK_MARKER));
    }

    /// T8: a store memory that is unparseable or below 100,000,000 bytes (0
    /// included), or an unparseable budget, is refused at load, naming the
    /// variable — the store memory's error also says it is the store's total
    /// and gives an example; a store memory of exactly 100,000,000 loads.
    ///
    /// **Fails on the build before b.qhk** at `99999999` (and `12`): the floor
    /// was 8 bytes, so they loaded.
    #[test]
    fn a_bad_release_budget_setting_is_refused_naming_the_variable() {
        for (name, value) in [
            ("LIVE_EDIT_EXT_RELEASE_MAX_BYTES", "lots"),
            ("LIVE_EDIT_EXT_RELEASE_MAX_BYTES", "-5"),
        ] {
            let err = from_env_with_release_budget(&[(name, Some(value))])
                .expect_err("a bad release-budget setting must be refused");
            assert!(
                err.contains(name),
                "{name}={value}: the error must name the variable; got {err}"
            );
        }
        for value in ["abc", "3gb", "-1", "12GB"] {
            let err =
                from_env_with_release_budget(&[("LIVE_EDIT_VALKEY_MEMORY_BYTES", Some(value))])
                    .expect_err("a store memory that is not a number must be refused");
            assert!(
                err.contains("LIVE_EDIT_VALKEY_MEMORY_BYTES")
                    && err.contains("total memory in bytes")
                    && err.contains("12000000000"),
                "LIVE_EDIT_VALKEY_MEMORY_BYTES={value}: the error must name the variable, say it \
                 is the store's total and give an example; got {err}"
            );
        }

        // The setting is the store's total memory: below 100 MB it is a
        // units slip, refused at startup naming the variable, the value, the
        // minimum and an example of a real total.
        for value in ["0", "1", "7", "12", "99999999"] {
            let err =
                from_env_with_release_budget(&[("LIVE_EDIT_VALKEY_MEMORY_BYTES", Some(value))])
                    .expect_err("a store memory below 100 MB must be refused");
            assert!(
                err.contains("LIVE_EDIT_VALKEY_MEMORY_BYTES")
                    && err.contains(&format!("{value} is below the minimum of 100000000 bytes"))
                    && err.contains("12000000000"),
                "LIVE_EDIT_VALKEY_MEMORY_BYTES={value}: the error must name the variable, the \
                 value, the minimum and an example; got {err}"
            );
        }
        let cfg =
            from_env_with_release_budget(&[("LIVE_EDIT_VALKEY_MEMORY_BYTES", Some("100000000"))])
                .expect("a store memory of 100 MB loads");
        assert_eq!(cfg.valkey_memory_bytes, Some(VALKEY_MEMORY_BYTES_MIN));
    }

    /// [`capture_warn_output`] at INFO, for the release budget's startup line.
    fn capture_info_output<T>(f: impl FnOnce() -> T) -> (T, String) {
        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .finish();
        let value = tracing::subscriber::with_default(subscriber, f);
        let text = String::from_utf8(captured.lock().unwrap().clone())
            .expect("captured log output must be valid UTF-8");
        (value, text)
    }

    /// A budget the store's own `maxmemory` set (decider ruling 4) is logged
    /// once at startup, at INFO, with `source` `info` and the reading it
    /// derives from — and with no fallback WARN.
    #[test]
    fn a_release_budget_from_the_store_logs_its_info_source_at_startup() {
        let cfg = from_env_with_release_budget(&[]).expect("the defaults must load");
        let budget = ExtReleaseBudget {
            bytes: 3_221_225_472 / 8,
            source: ExtReleaseBudgetSource::StoreInfo {
                min_maxmemory_bytes: 3_221_225_472,
                primaries: 3,
            },
        };
        let ((), output) =
            capture_info_output(|| cfg.log_ext_release_budget_startup(&budget, None));
        assert_eq!(output.lines().count(), 1, "one startup line; got {output}");
        assert!(output.contains(" INFO "), "{output}");
        assert!(output.contains("source=\"info\""), "{output}");
        assert!(
            output.contains("ext_release_max_bytes=402653184"),
            "{output}"
        );
        assert!(
            output.contains("min_maxmemory_bytes: 3221225472, primaries: 3"),
            "the line names the reading it derives from; got {output}"
        );
        assert!(!output.contains(RELEASE_BUDGET_FALLBACK_MARKER), "{output}");
    }

    /// The fallback WARN carries the store's reason it could not be used, and
    /// `source` `fallback`.
    #[test]
    fn the_release_budget_fallback_warn_names_the_stores_reason() {
        let cfg = from_env_with_release_budget(&[]).expect("the defaults must load");
        let reason = "primary 10.0.0.2:6379 reports maxmemory 0 (no memory limit configured)";
        let problem = crate::storage::store_memory::StoreMemoryProblem::Unusable(
            crate::storage::store_memory::UnusableReason::ZeroLimit {
                primary: "primary 10.0.0.2:6379".to_owned(),
            },
        );
        let ((), output) = capture_warn_output(|| {
            cfg.log_ext_release_budget_startup(&cfg.ext_release_budget(), Some(&problem))
        });
        assert_eq!(
            output.matches(RELEASE_BUDGET_FALLBACK_WARN_MARKER).count(),
            1,
            "{output}"
        );
        assert!(output.contains(reason), "the WARN names why; got {output}");
        assert!(output.contains("source=\"fallback\""), "{output}");
    }

    /// `LIVE_EDIT_GC_POLL_INTERVAL_SECS` is retired (policy 4): the service no
    /// longer reads it, so even a value that would not parse cannot refuse a
    /// boot. Before b.yuo `"abc"` failed `Config::from_env`.
    #[test]
    fn the_retired_gc_poll_interval_is_no_longer_read() {
        from_env_with_release_budget(&[("LIVE_EDIT_GC_POLL_INTERVAL_SECS", Some("abc"))])
            .expect("the retired GC poll interval must be ignored, not parsed");
    }

    // ── The backup store's identity (b.wsx) ──────────────────────────────

    /// A loaded `Config` whose checkpoint adapter and address fields are then
    /// set directly: the identity reads only those.
    fn identity_config(
        adapter: CheckpointAdapter,
        endpoint: Option<&str>,
        base_path: Option<&str>,
        local_disk_path: Option<&str>,
    ) -> Config {
        let mut cfg = from_env_with_release_budget(&[]).expect("the defaults must load");
        cfg.checkpoint_adapter = adapter;
        cfg.checkpoint_storage_api_endpoint = endpoint.map(str::to_owned);
        cfg.checkpoint_base_path = base_path.map(PathBuf::from);
        cfg.checkpoint_local_disk_path = local_disk_path.map(str::to_owned);
        cfg
    }

    fn storage_api_identity(endpoint: &str, base_path: &str) -> String {
        identity_config(
            CheckpointAdapter::StorageApi,
            Some(endpoint),
            Some(base_path),
            None,
        )
        .checkpoint_store_identity()
    }

    fn local_disk_identity(path: &str) -> String {
        identity_config(CheckpointAdapter::LocalDisk, None, None, Some(path))
            .checkpoint_store_identity()
    }

    /// T6 (b.wsx): the identity backup confirmations are written under is 16
    /// lowercase hex characters, and names the place the adapter writes to:
    /// the same address spelled with surrounding whitespace or a trailing `/`
    /// is the same store, while another endpoint, another base path, another
    /// directory or another adapter is another store, so every confirmation
    /// written under the old identity counts as none. Settings that do not
    /// change the address (a timeout) leave it alone.
    #[test]
    fn checkpoint_store_identity_names_the_adapters_address_and_ignores_its_spelling() {
        let endpoint = "https://storage.example.com";
        let base = "/checkpoints/live-edit";
        let id = storage_api_identity(endpoint, base);
        assert_eq!(id.len(), 16, "{id}");
        assert!(
            id.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "lowercase hex: {id}"
        );

        assert_eq!(
            storage_api_identity(" https://storage.example.com/ ", "/checkpoints/live-edit/"),
            id,
            "whitespace and a trailing `/` do not change the store"
        );
        let mut retimed = identity_config(
            CheckpointAdapter::StorageApi,
            Some(endpoint),
            Some(base),
            None,
        );
        retimed.checkpoint_request_timeout_ms += 1_000;
        assert_eq!(
            retimed.checkpoint_store_identity(),
            id,
            "a setting that is not the address does not change the store"
        );

        let others = [
            storage_api_identity("https://other-storage.example.com", base),
            storage_api_identity(endpoint, "/checkpoints/other"),
            local_disk_identity("/checkpoints/live-edit"),
            local_disk_identity("/var/lib/live-edit"),
        ];
        for other in &others {
            assert_ne!(other, &id, "another address is another store");
        }
        assert_ne!(others[2], others[3], "another directory is another store");
        assert_eq!(
            local_disk_identity("/var/lib/live-edit/"),
            others[3],
            "a directory's trailing `/` does not change the store"
        );
    }

    /// The in-memory adapter keeps its blobs in this process, so its identity
    /// is the process's own: stable across calls within the process, and
    /// unlike any address-derived identity. `Disabled` has a fixed one.
    #[test]
    fn checkpoint_store_identity_of_the_in_memory_adapter_is_stable_within_the_process() {
        let in_memory = identity_config(CheckpointAdapter::InMemory, None, None, None);
        let first = in_memory.checkpoint_store_identity();
        assert_eq!(first.len(), 16);
        assert_eq!(in_memory.checkpoint_store_identity(), first);
        assert_eq!(
            identity_config(CheckpointAdapter::InMemory, None, None, None)
                .checkpoint_store_identity(),
            first,
            "every Config of this process names the same in-memory store"
        );
        let disabled = identity_config(CheckpointAdapter::Disabled, None, None, None)
            .checkpoint_store_identity();
        assert_eq!(
            disabled,
            identity_config(CheckpointAdapter::Disabled, None, None, None)
                .checkpoint_store_identity()
        );
        assert_ne!(disabled, first);
        for address_derived in [
            local_disk_identity(""),
            local_disk_identity("/var/lib/live-edit"),
            storage_api_identity("", ""),
            storage_api_identity("https://storage.example.com", "/checkpoints/live-edit"),
        ] {
            assert_ne!(
                address_derived, first,
                "the in-memory store is no address-derived store"
            );
        }
    }
}
