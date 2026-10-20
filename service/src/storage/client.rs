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

// Valkey connection pool and client wrapper (fred crate).
//
// Provides:
// - `ValkeyClient` — cheaply cloneable wrapper around a data `Pool` and a
//   `SubscriberClient` for pub/sub.
// - `connect()` — reads config, creates the pools and the pub/sub connection
//   with exponential-backoff retry (200 ms initial, 5 s cap, a hard per-phase
//   budget), validates each with PING, and returns a ready `ValkeyClient`. The
//   stream readers are not connected here: `ReaderConnector` connects each one
//   when it first serves a subscription.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use aws_credential_types::provider::SharedCredentialsProvider;
#[cfg(feature = "valkey-tls")]
use fred::types::config::TlsHostMapping;
use fred::{
    clients::{Client, Pool, SubscriberClient, Transaction},
    error::ErrorKind as FredErrorKind,
    interfaces::{ClientLike, ClusterInterface, EventInterface},
    live_edit::liveness::PeerReplies,
    prelude::{Builder, ReconnectPolicy},
    types::{
        config::{
            Blocking, Config as FredConfig, CredentialProvider, PerformanceConfig, Server,
            ServerConfig,
        },
        ClientState, ClusterHash, ConnectHandle, FromValue, RespVersion,
    },
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::auth::service_token::{ServiceTokenProvider, TokenClaims};
use crate::config::{Config, ValkeyAuthMode};
use crate::observability::metrics::SharedMetrics;
use crate::storage::checkout::{CheckedOutClient, ExclusiveCheckout};
use crate::storage::connection_buffers;
use crate::storage::diff_read::ObjectSnapshotReadFaults;
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
use crate::storage::diff_read::UnprovableCutFaults;
use crate::storage::external_values::{ReadHoldTakeHolds, ReleaseEnqueueFaults};
use crate::storage::fanout_thread::{connect_router_on, FanoutThreads};
use crate::storage::iam_credentials::IamCredentialProvider;
use crate::storage::member_replacement::{MemberReplacer, VerdictSink};
use crate::storage::memory_budget::MemoryBudget;
use crate::storage::no_refresh_credentials::NoRefreshCredentialProvider;
use crate::storage::router_supervisor::{
    AbortCurrent, RetirableWatch, RouterRestart, RouterSupervisor, ROUTER_ABORT_TIMEOUT,
};
use crate::storage::stream_mux::{StreamMux, StreamMuxKnobs};
use crate::storage::subscriber_admission::{
    AdmissionPolicy, AdmissionRefusal, CatchUpGate, CatchUpReadGate, SubscriberAdmission,
    SubscriberAdmissionGate,
};
use crate::storage::token_file_credentials::{
    valkey_identity_verdict, IdentityVerdict, TokenFileCredentialProvider,
};
use crate::storage::txn_checkout::{
    is_state_proof_reason, probe_txn_slot, should_escalate_ineffective_repair,
    should_recycle_unknown_state, state_proof_repair_is_ineffective, CheckoutVerdict, ProbeFaults,
    TXN_CHECKOUT_PROBE_CEILING, TXN_CHECKOUT_PROBE_TIMEOUT, TXN_RECYCLE_REARM_EVERY,
    TXN_STATE_PROOF_REPAIR_ATTEMPTS,
};

// `valkey-tls` MUST imply a fred TLS feature — `src/config.rs`'s SR-1.4 guard
// trusts `cfg!(feature = "valkey-tls")` as a proxy for it. `Config::tls` is
// #[cfg]-gated behind fred's TLS features (fred-10.1.0/src/types/config.rs:633),
// so this fails to compile if the two ever drift apart.
#[cfg(feature = "valkey-tls")]
const _: () = {
    #[allow(dead_code)]
    fn assert_fred_tls_compiled(c: &FredConfig) -> &Option<fred::types::config::TlsConfig> {
        &c.tls
    }
};

/// Pool label for the WATCH/MULTI/EXEC transaction pool.
pub const POOL_LABEL_TXN: &str = crate::observability::metrics::VALKEY_POOL_TXN;
/// Pool label for the long-lived subscriber `XREAD BLOCK` pool.
pub const POOL_LABEL_SUBSCRIBER: &str = crate::observability::metrics::VALKEY_POOL_SUBSCRIBER_POOL;
/// Pool label for the dedicated GC + orphan-sweep maintenance pool (b.kf1
/// H-stor-5; gauge wiring added in b.mfn).
pub const POOL_LABEL_MAINTENANCE: &str = crate::observability::metrics::VALKEY_POOL_MAINTENANCE;
/// Pool label for the multiplexed data pool ([`ValkeyClient::pool`]).
pub const POOL_LABEL_DATA: &str = crate::observability::metrics::VALKEY_POOL_DATA;
/// Pool label for the multiplexed snapshot lane
/// ([`ValkeyClient::snapshot_pool`]).
pub const POOL_LABEL_SNAPSHOT: &str = crate::observability::metrics::VALKEY_POOL_SNAPSHOT;
/// Pool label for the dedicated pub/sub connection
/// ([`ValkeyClient::subscriber`]). Not to be confused with
/// [`POOL_LABEL_SUBSCRIBER`], the `XREAD BLOCK` pool.
pub const POOL_LABEL_PUBSUB: &str = crate::observability::metrics::VALKEY_POOL_SUBSCRIBER;

/// Where service code received a fred error: the pool whose connection issued
/// the command, how to reach the metrics that count a command timeout on that
/// pool, and the store out-of-memory signal a refused write is noted on.
///
/// fred applies its per-command timeout as a bare `tokio::time::timeout` around
/// the reply and hands the resulting `ErrorKind::Timeout` to the caller only —
/// it never reaches the connection-event hook [`spawn_error_metric`] listens to.
/// So a timed-out command is counted where service code first receives the
/// error: the three mappers below and `SweepError::from_fred` take one of
/// these as their leading argument, `WatchedTxn::exec` takes one and counts a
/// timed-out `EXEC` where it builds `WatchedExec::InDoubt`, and every other
/// consumer calls [`Self::observe_error`] itself. That puts the count on
/// `valkey_errors_total{pool, kind="command_timeout"}`.
///
/// The same receipt point notes a store write refused `-OOM` (b.pa1):
/// [`Self::observe_error`] hands it to the origin's
/// [`StoreOomSignal`](crate::storage::store_oom::StoreOomSignal), which
/// requests the GC's out-of-memory collection (`storage::store_oom`). An
/// origin built with [`Self::client`] takes the client's signal; one built
/// with [`Self::metrics`] must name its signal, so every receipt site covers
/// the trigger by construction. The counted upload write sites note it
/// through [`refuse_store_write`], which takes an origin too.
///
/// A site that hands the error on to a mapper, or receives it from
/// `WatchedTxn::exec`, must not also count it; each received error is counted
/// once, where it was received. Noting a refusal twice is harmless (the
/// signal rate-limits its requests), so an origin that counts nothing still
/// carries a signal.
#[derive(Clone, Copy)]
pub struct FredErrOrigin<'a> {
    pool: &'static str,
    metrics: OriginMetrics<'a>,
}

/// How a [`FredErrOrigin`] reaches the metrics handle and the store
/// out-of-memory signal.
#[derive(Clone, Copy)]
enum OriginMetrics<'a> {
    /// Resolved through [`ValkeyClient::metrics_snapshot`], on the error path
    /// only, so a successful command pays nothing; the signal is the client's
    /// ([`ValkeyClient::store_oom_signal`]).
    Client(&'a ValkeyClient),
    /// Threaded explicitly by sites that already carry the handle; `None` when
    /// the caller runs without metrics. The signal is named by the site.
    Explicit(
        Option<&'a SharedMetrics>,
        &'a crate::storage::store_oom::StoreOomSignal,
    ),
}

impl<'a> FredErrOrigin<'a> {
    /// A command issued on `pool` of `client`.
    pub fn client(client: &'a ValkeyClient, pool: &'static str) -> Self {
        FredErrOrigin {
            pool,
            metrics: OriginMetrics::Client(client),
        }
    }

    /// A command issued on `pool`, counted on `metrics` (nothing is counted
    /// when it is `None`), with a store write refused `-OOM` noted on
    /// `store_oom` — the issuing client's signal
    /// ([`ValkeyClient::store_oom_signal`]), or
    /// [`StoreOomSignal::unwired`](crate::storage::store_oom::StoreOomSignal::unwired)
    /// where the store cannot answer `-OOM`.
    pub fn metrics(
        metrics: Option<&'a SharedMetrics>,
        pool: &'static str,
        store_oom: &'a crate::storage::store_oom::StoreOomSignal,
    ) -> Self {
        FredErrOrigin {
            pool,
            metrics: OriginMetrics::Explicit(metrics, store_oom),
        }
    }

    /// Count `e` on `valkey_errors_total{pool, kind="command_timeout"}` when it
    /// is fred's `Timeout`, and note it on the origin's store out-of-memory
    /// signal when it is a store write refused `-OOM`
    /// ([`store_write_refusal`]); do nothing for any other kind. `IO` and
    /// `Canceled` are left alone because fred already broadcasts the connection
    /// event behind them, which [`spawn_error_metric`] counts.
    ///
    /// Observing only: it borrows the error, returns nothing and steers
    /// nothing. The note requests an out-of-memory collection and wakes this
    /// replica's GC loop, at most once a second, without waiting
    /// (`storage::store_oom`).
    pub fn observe_error(&self, e: &fred::error::Error) {
        if store_write_refusal(e) == Some(StoreWriteRefusal::OutOfMemory) {
            self.note_out_of_memory();
            return;
        }
        if !matches!(e.kind(), FredErrorKind::Timeout) {
            return;
        }
        match self.metrics {
            OriginMetrics::Client(client) => {
                if let Some(m) = client.metrics_snapshot() {
                    m.inc_valkey_command_timeout(self.pool);
                }
            }
            OriginMetrics::Explicit(Some(m), _) => m.inc_valkey_command_timeout(self.pool),
            OriginMetrics::Explicit(None, _) => {}
        }
    }

    /// Note a store write refused `-OOM` on this origin's signal.
    fn note_out_of_memory(&self) {
        match self.metrics {
            OriginMetrics::Client(client) => client.store_oom_signal().note(),
            OriginMetrics::Explicit(_, signal) => signal.note(),
        }
    }

    /// Count one store write refusal on
    /// `live_edit_store_write_refused_total{stage, reason}` through this
    /// origin's metrics (none when it has none).
    fn count_store_write_refused(&self, stage: &'static str, reason: &'static str) {
        match self.metrics {
            OriginMetrics::Client(client) => {
                if let Some(m) = client.metrics_snapshot() {
                    m.inc_store_write_refused(stage, reason);
                }
            }
            OriginMetrics::Explicit(Some(m), _) => m.inc_store_write_refused(stage, reason),
            OriginMetrics::Explicit(None, _) => {}
        }
    }

    /// **Test-only.** Count `e` on the client's
    /// [`ValkeyClient::watched_exec_timeouts_for_test`] when it is fred's
    /// `Timeout` and this origin names a client ([`Self::client`]); do nothing
    /// otherwise. Called by `WatchedTxn::exec` / `exec_overridden` alone, beside
    /// [`Self::observe_error`], so the count holds their in-doubt `EXEC`
    /// timeouts and nothing else. Compiled only for tests and the
    /// `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn count_watched_exec_timeout_for_test(&self, e: &fred::error::Error) {
        if !matches!(e.kind(), FredErrorKind::Timeout) {
            return;
        }
        if let OriginMetrics::Client(client) = self.metrics {
            client
                .watched_exec_timeouts_for_test
                .fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// Budget for the `QUIT` half of a dirty-slot recycle (b.7td).
///
/// fred's `quit()` waits for the server's reply under the client-wide
/// `default_command_timeout`; this caps it independently so a connection whose
/// reply stream is *already* offset — the case that got us here — cannot stall
/// a checkout for the whole configured command timeout. Exceeding it is not
/// fatal: the repair then aborts the slot's still-running router
/// (`RouterSupervisor::abort_current`) and `connect()` starts a new one.
const TXN_RECYCLE_QUIT_TIMEOUT: Duration = Duration::from_secs(2);

/// Budget for the reconnect half of a dirty-slot recycle (b.7td): how long the
/// repair waits for the restarted router to report its connection.
///
/// Covers a full TCP connect, TLS handshake and `AUTH`/`HELLO` through the
/// configured credential provider, which on a managed backing store is
/// comfortably a second or two. Sized under the auth-RPC family's 5 s so a
/// wedged rebuild surfaces well inside `LIVE_EDIT_RPC_DEADLINE_MS` rather than
/// consuming it. Exceeding it ends only the wait: the router keeps connecting,
/// and the router supervisor watches it either way.
const TXN_RECYCLE_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// The `reason` a `txn` slot repair logs when the router supervisor restarts
/// the slot's ended router task (`storage::router_supervisor`), beside the
/// dirty-checkout reasons a probe-driven recycle logs. A log value only, never
/// a metric label: the restart is counted on
/// `live_edit_valkey_router_restarts_total{pool="txn"}`.
const TXN_RECYCLE_REASON_ROUTER_ENDED: &str = "router_ended";

/// Ceiling on how many distinct nodes
/// [`ValkeyClient::txn_state_proof_streaks`] tracks at once (b.289 sub-task 4).
///
/// The map is keyed by `host:port` and only ever grows on a state-proof dirty
/// verdict, so on any healthy deployment it is empty. The cap bounds the
/// pathological case: a long series of reshardings against a large cluster,
/// each one leaving a stale entry for a node that no longer owns the slot.
/// 64 is far above any real primary count this service targets (the Helm
/// chart's largest documented shape is single-digit shards) and small enough
/// that a full map is a rounding error next to one pooled connection. Past it,
/// a verdict against a node with no bucket of its own folds into the shared
/// `""` bucket rather than being dropped — and so does that node's next `Clean`
/// checkout, because both sides resolve the bucket through the one
/// [`ValkeyClient::bucket_key`] helper. A cap-overflowed node therefore shares
/// the fallback bucket's streak *and* its re-arm; it does not become a node
/// whose breaker can never be re-armed.
const TXN_STATE_PROOF_STREAK_MAX_NODES: usize = 64;

/// How long [`ValkeyClient::quit`] gives in-flight slot recycles to finish
/// before it closes the pools anyway (b.7td round 2).
///
/// A recycle runs in a spawned task that **owns** the slot's
/// `OwnedMutexGuard<Client>` (that is what makes the checkout cancellation-safe
/// — see [`ValkeyClient::recycle_txn_slot`]), so `ExclusivePool::quit` would
/// otherwise block on the same mutex until the recycle finished. Draining the
/// tracker first turns that into a bounded, explained wait instead of a
/// mysterious `txn pool quit timed out`.
///
/// Deliberately shorter than a recycle's own worst case (2 s `QUIT` + 0.5 s
/// router abort + 5 s reconnect): at shutdown the repair has no value left —
/// the connection is about to be closed either way — so this budget buys an
/// orderly hand-off, not a completed repair. Sized to match
/// [`UNWATCH_DRAIN_TIMEOUT`], its sibling in the same `quit()` sequence.
///
/// The knob both are written against is
/// **`LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS`** (30 s default), not
/// `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`: `ValkeyClient::quit` runs inside
/// `main.rs`'s `shutdown_sequence`, which is wrapped in one
/// `tokio::time::timeout(shutdown_overall_budget, …)`. The per-task budget
/// derived from `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS` bounds the individual
/// background-task joins earlier in that sequence, never this pool teardown.
/// The arithmetic is unchanged either way — 2 s + 2 s of drain is a small
/// fraction of the 30 s the whole sequence gets, and `Config::from_env` rejects
/// `overall < timeout + 15` so the overall budget cannot be configured below
/// 15 s.
const TXN_RECYCLE_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// How long [`ValkeyClient::quit`] gives the snapshot-fence guards' dropped
/// cleanups — a dropped `FenceLease`'s release script, a dropped
/// `WriterIntent`'s `HDEL` (`storage::snapshot_fence`), and a WATCH-path
/// attempt's `IntentRenewal` task still running, which clears its intent when
/// its attempt ends — to reach the store before it closes the pools anyway.
///
/// Each cleanup is one small command on the data pool, so a healthy store
/// answers in milliseconds; a renewal task ends with its attempt, which
/// graceful shutdown has normally ended or dropped before `quit()`. The bound
/// only bites on a stalled store, or an attempt that outlived the server
/// drain, where the lease and the intent then lapse at their own deadlines as
/// they would have without the drain. It runs concurrently with the [`TXN_RECYCLE_DRAIN_TIMEOUT`]
/// drain and is no longer than it, so it adds nothing to `quit()`'s worst case
/// — the 11.5 s sum `Config::from_env`'s `overall >= timeout + 15` floor is
/// written against, which the `quit()` bounds assertion after
/// `STREAM_MUX_SHUTDOWN_GRACE` checks with this drain counted inside the
/// recycle drain's term. The assertion below makes lengthening it past that
/// drain a compile error rather than a silent erosion of that floor.
const FENCE_GUARD_CLEANUP_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

const _: () = assert!(
    FENCE_GUARD_CLEANUP_DRAIN_TIMEOUT.as_millis() <= TXN_RECYCLE_DRAIN_TIMEOUT.as_millis(),
    "ValkeyClient::quit runs the snapshot-fence guard cleanup drain beside the txn slot-recycle \
     drain, so it must be no longer than that drain or it lengthens quit()'s worst case past the \
     sum Config::from_env's shutdown-budget floor (overall >= timeout + 15) is written against"
);

/// How long [`ValkeyClient::quit`] gives each pool's `quit()` before it logs a
/// warning and leaves that pool's connections to be closed by process exit.
///
/// All the pools quit concurrently, so this bounds the whole pool teardown, not
/// each pool in turn.
pub(crate) const QUIT_TIMEOUT: Duration = Duration::from_secs(2);

/// A dirty `txn` checkout's worst case, mechanized so a reviewer does not have
/// to sum four constants across three files to bound it: both probe commands
/// timing out, then a `QUIT` that never answers, then an aborted router that
/// does not end, then a reconnect that never completes. It must stay a clear
/// fraction of the default `LIVE_EDIT_RPC_DEADLINE_MS` (30 s) — the whole point
/// of the repair is to keep a corrupt slot from becoming a client-visible
/// `DEADLINE_EXCEEDED`, so the repair itself must not be able to cause one.
const _: () = assert!(
    TXN_CHECKOUT_PROBE_CEILING.as_millis()
        + TXN_RECYCLE_QUIT_TIMEOUT.as_millis()
        + ROUTER_ABORT_TIMEOUT.as_millis()
        + TXN_RECYCLE_CONNECT_TIMEOUT.as_millis()
        <= 10_000,
    "the worst-case dirty-checkout repair (probe ceiling + QUIT budget + router abort budget + \
     reconnect budget) must stay well inside the 30 s default LIVE_EDIT_RPC_DEADLINE_MS. The \
     repair exists to stop a corrupt txn slot from producing a client-visible DEADLINE_EXCEEDED; a \
     repair that can itself cause one has inverted the fix. Lower one of the four budgets."
);

/// How long [`ValkeyClient::quit`] waits for the stream multiplexer's readers
/// to stop after waking their blocked XREADs (`StreamMux::shutdown`).
const STREAM_MUX_SHUTDOWN_GRACE: Duration = Duration::from_secs(1);

/// The least time [`ValkeyClient::quit`]'s sequential bounds must leave inside
/// [`MIN_SHUTDOWN_BUDGET_HEADROOM_SECS`] for the shutdown steps with no timeout
/// of their own (closing gates and trackers, aborting task handles, firing
/// cancellation tokens, reporting results) and for the runtime to schedule
/// them on a loaded process at SIGTERM. 2 s is one drain's worth: the headroom
/// leaves 3.5 s today, so adding another 2 s sequential drain to `quit()`
/// without raising the headroom fails the build, while a sub-second bound does
/// not.
///
/// [`MIN_SHUTDOWN_BUDGET_HEADROOM_SECS`]: crate::config::MIN_SHUTDOWN_BUDGET_HEADROOM_SECS
const QUIT_MIN_UNTIMED_SLACK: Duration = Duration::from_secs(2);

/// The longer of two durations, usable in a `const` (`Ord::max` is not).
const fn longer(a: Duration, b: Duration) -> Duration {
    if a.as_nanos() >= b.as_nanos() {
        a
    } else {
        b
    }
}

/// [`ValkeyClient::quit`]'s worst case, mechanized so the shutdown-budget
/// floor cannot drift below it. `quit()` runs, one after another: the
/// member-replacer drain, the router-supervisor drain, the
/// slot-recycle drain (with the snapshot-fence guard cleanup drain beside it,
/// no longer than it by the assertion after
/// `FENCE_GUARD_CLEANUP_DRAIN_TIMEOUT`, so it has no term of its own), the
/// UNWATCH drain, the stream-multiplexer grace, and
/// then one concurrent stage in which every pool's `QUIT` runs under
/// [`QUIT_TIMEOUT`] while each connected stream reader's retire first waits up
/// to [`ROUTER_ABORT_TIMEOUT`] for its router watcher and then sends its own
/// `QUIT` under [`QUIT_TIMEOUT`] — so that stage costs the longer of the two.
/// The sum plus [`QUIT_MIN_UNTIMED_SLACK`] must fit inside the headroom
/// `Config::validate_shutdown_budgets` requires above
/// `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS`.
const _: () = assert!(
    crate::storage::member_replacement::MEMBER_REPLACER_DRAIN_TIMEOUT.as_millis()
        + crate::storage::router_supervisor::ROUTER_SUPERVISOR_DRAIN_TIMEOUT.as_millis()
        + TXN_RECYCLE_DRAIN_TIMEOUT.as_millis()
        + UNWATCH_DRAIN_TIMEOUT.as_millis()
        + STREAM_MUX_SHUTDOWN_GRACE.as_millis()
        + longer(
            QUIT_TIMEOUT,
            ROUTER_ABORT_TIMEOUT.saturating_add(QUIT_TIMEOUT),
        )
        .as_millis()
        + QUIT_MIN_UNTIMED_SLACK.as_millis()
        <= crate::config::MIN_SHUTDOWN_BUDGET_HEADROOM_SECS as u128 * 1_000,
    "ValkeyClient::quit()'s sequential bounds plus QUIT_MIN_UNTIMED_SLACK exceed \
     MIN_SHUTDOWN_BUDGET_HEADROOM_SECS: the overall shutdown budget would fire inside quit(). \
     Raise the headroom in src/config.rs (and the docs that state it) or lower a quit() bound."
);

/// RAII token that decrements `valkey_pool_in_use{pool}` when it drops
/// (b.3z7 H-obs-3).
///
/// **It is no longer a value call sites carry themselves.** Through b.f4f the
/// token travelled beside the `OwnedMutexGuard<Client>` as the second half of a
/// returned tuple, and fourteen call sites had to keep the two in the same scope
/// so the gauge decrement landed with the mutex release — a rule eight of them
/// got backwards, because Rust drops *locals* in reverse declaration order and
/// `let (client, _pool_token) = …` therefore dropped the token first. The token
/// now lives inside [`CheckedOutClient`], between the guard and the semaphore
/// permit, where field declaration order fixes the sequence once for every site:
/// mutex released, gauge decremented, permit freed.
///
/// It stays a separate *value* (rather than being folded into the guard's
/// `Drop`) for the reason it always was: [`WatchGuard`] moves the whole checkout
/// into a spawned UNWATCH task on Drop, so the decrement must follow the
/// **actual** release point — after UNWATCH completes — and not the synchronous
/// `Drop::drop` return. Moving one composite value into that task body is
/// strictly easier than moving two, and `Policy decision 2` of b.f4f makes the
/// permit ride along with it, so the `txn_pool` slot's entitlement is released
/// only once UNWATCH has actually happened.
///
/// When metrics are not yet installed (startup PINGs, tests that don't wire
/// observability), the inner `metrics` is `None` and `Drop` is a no-op.
pub struct PoolUseToken {
    metrics: Option<SharedMetrics>,
    pool_label: &'static str,
}

impl PoolUseToken {
    /// Mint a token for `pool_label`, incrementing the gauge now when metrics
    /// are installed. Called only from `storage::checkout`, which pairs it with
    /// the slot guard inside one [`CheckedOutClient`], from the stream
    /// multiplexer's readers (one token while a reader has a stream), and from
    /// [`ValkeyClient::acquire_maintenance`] and
    /// [`ValkeyClient::acquire_snapshot`], whose "slot" is logical.
    pub(crate) fn new(metrics: Option<SharedMetrics>, pool_label: &'static str) -> Self {
        if let Some(m) = &metrics {
            m.inc_valkey_pool_in_use(pool_label);
        }
        PoolUseToken {
            metrics,
            pool_label,
        }
    }
}

impl Drop for PoolUseToken {
    fn drop(&mut self) {
        if let Some(metrics) = self.metrics.take() {
            metrics.dec_valkey_pool_in_use(self.pool_label);
        }
    }
}

/// Classify a fred error as transient I/O that is safe to retry on a fresh
/// connection.
///
/// Used by `update_layer` (b.1dz) so brief Valkey blips during the
/// WATCH/HGETALL/snapshot-load phase turn into retries rather than surfacing
/// as `Internal`. The classification deliberately covers only connection-level
/// / flow-control conditions; protocol errors, deserialization failures, and
/// configuration problems must still bubble up so real bugs don't get masked
/// by silent retries.
///
/// Transient (retry):
/// - `Timeout` — per-command deadline, most often a pooled connection still
///   mid-reconnect
/// - `IO` — underlying TCP error (connection reset, broken pipe)
/// - `Canceled` — in-flight command canceled because the connection closed
/// - `Cluster` — a cluster state transition that reached the caller as a server
///   error: fred maps `-MOVED` / `-ASK` / `-CLUSTERDOWN` to this kind
///   (`fred-10.1.0/src/protocol/utils.rs:113`). In practice the router
///   intercepts the two redirects before the caller sees them, so what arrives
///   here is `CLUSTERDOWN` or a redirect that escaped interception; either way a
///   retry re-resolves on the healed topology. The redirects fred intercepted
///   but could not *complete* arrive under the next bullet instead.
/// - `Backpressure` — command queue full; a short retry lets the queue drain
/// - fred's "no live connection to route to" error, matched on its *message*
///   rather than its kind — see [`is_router_no_connection_error`] for why the
///   kind alone cannot express it
/// - fred's router **retry-budget exhaustion** family — the errors a `MOVED` /
///   `ASK` redirect turns into when the command has no write attempt left to
///   follow the hop with, or when it is a script, which the vendored fred
///   never lets follow an `ASK`. Also message-matched; see
///   [`is_router_retry_budget_exhausted_error`], which is where the
///   Valkey-Cluster topology-change story actually lives (`Cluster` above covers
///   only the redirects fred *did* follow).
/// - fred's **`Transaction` router** giving up — a fred `Transaction`
///   (`client.multi()` … `exec()`, on the `txn_pool`) met a `MOVED` / `ASK`
///   past its zero redirection budget, or ran out of attempts with no
///   underlying error to forward. Also message-matched; see
///   [`is_transaction_router_exhausted_error`], which states what may have
///   run.
/// - the backing store **not serving** commands for a while — `-LOADING` (a
///   restarted node replaying its dataset) or `-BUSY` (a long-running script or
///   module command), bare or after an `EXEC`'s `EXECABORT Transaction
///   discarded because of: ` prefix. Also message-matched, on the token with
///   its trailing space; see [`is_store_not_serving_reply`].
/// - a cluster node refusing a **multi-key command during a slot migration**
///   — `-TRYAGAIN`, answered before the command runs, while the command's
///   keys are split between the migrating slot's source and target. Also
///   message-matched, on the token with its trailing space; see
///   [`is_tryagain_reply`].
///
/// Not transient (surface as `Internal`, except where noted):
/// - `Config`, `Url`, `Auth`, `Tls` — fatal configuration / auth failures,
///   including one met opening a connection to a cluster `-ASK` target: the
///   vendored fred keeps exactly these four kinds there and answers every
///   other connect or setup failure `IO`, transient (fred's
///   `live_edit::ask`, which gives the reasons). Under the rustls backend
///   `Tls` is only a TLS configuration fault, such as an invalid server
///   name: a certificate or handshake failure reaches fred as an I/O error
///   and arrives here as `IO`, transient, on every connect
/// - `Protocol`, `Parse` — malformed frames or deserialization bugs. The one
///   `Protocol` fred raises opening an `-ASK` target's connection (its setup's
///   "Invalid or missing cluster state.", a node still joining the cluster)
///   never reaches here: that path answers it `IO`, as above
/// - `InvalidArgument`, `InvalidCommand` — bad call at the fred layer
/// - `Routing`, `Sentinel`, `NotFound`, `Unknown` — not transient *by kind*,
///   with five message-matched carve-out families
///   ([`is_router_no_connection_error`],
///   [`is_router_retry_budget_exhausted_error`],
///   [`is_transaction_router_exhausted_error`],
///   [`is_store_not_serving_reply`] and [`is_tryagain_reply`], all folded
///   into this function below).
///   Absent those carve-outs these kinds are either deterministic failures or
///   ambiguous cases where silent retry would hide a real issue, so the
///   blanket kind-level classification stands.
/// - A **store write refusal** (`-OOM`, `-MISCONF`, `-READONLY`, `-NOREPLICAS`;
///   [`store_write_refusal`]) is deliberately not transient either: those
///   conditions can persist, and a retry loop would spin against them. The
///   mappers ([`map_snapshot_load_err`], [`map_fred_err_for_grpc`]) answer it
///   with a plain `Unavailable` that no loop retries, instead of `Internal`.
///
/// b.k1r low: `fred::error::ErrorKind::Replica` would be the natural
/// addition here (a brief window during replica failover reads as transient),
/// but the variant is gated behind fred's `replicas` feature which this
/// service does not enable. If the service ever adopts replica-aware reads,
/// adding `FredErrorKind::Replica` here (and switching the `cfg` gate on
/// the feature) is the correct next step.
pub fn is_transient_fred_error(e: &fred::error::Error) -> bool {
    matches!(
        e.kind(),
        FredErrorKind::Timeout
            | FredErrorKind::IO
            | FredErrorKind::Canceled
            | FredErrorKind::Cluster
            | FredErrorKind::Backpressure,
    ) || is_router_no_connection_error(e)
        || is_router_retry_budget_exhausted_error(e)
        || is_transaction_router_exhausted_error(e)
        || is_store_not_serving_reply(e)
        || is_tryagain_reply(e)
}

/// Whether a fred error is a cluster node's `-TRYAGAIN`: a **multi-key
/// command refused during a slot migration**, before it ran.
///
/// # The reply (Valkey 9.0.3)
///
/// `TRYAGAIN Multiple keys request during rehashing of slot`, from
/// `clusterRedirectClient` (`src/cluster.c:1316-1320`), its only emitter. It
/// answers a command whose keys all hash to one slot that is mid-migration,
/// when only some of them are on the node it reached: on the migrating slot's
/// source when some keys are still there and some have moved
/// (`getNodeByQuery`, `:1265-1274`), and on the target, after `ASKING`, when
/// some have not arrived yet (`:1280-1287`). A command whose keys are all
/// absent from the source gets `-ASK` instead, which fred's router follows.
///
/// fred's `pretty_error` maps no `TRYAGAIN` token
/// (`fred-10.1.0/src/protocol/utils.rs:119`, `_ => ErrorKind::Unknown`) and
/// keeps the whole reply as `details`, so it arrives as `ErrorKind::Unknown`
/// with the token first. A Lua script's own `redis.call` inside a migrating
/// slot fails differently ("Unable to execute command … rehashing", handled
/// by `storage::diff_read`); this predicate is about the reply to the
/// top-level command.
///
/// # Why this is transient
///
/// `processCommand` answers it from its pre-execution cluster redirect check
/// (`src/server.c:4310-4324`): the command never runs, and an `EXEC` that
/// meets it discards its transaction first, so nothing a `-TRYAGAIN`
/// answers was applied — the safety basis of the `-ASK` exhaustion shapes
/// [`is_router_retry_budget_exhausted_error`] already classifies transient.
/// It clears by itself when the migration moves the rest of the slot's keys
/// or ends, so the remedy is to back off and re-issue within the caller's own
/// budget. Every multi-key call site gets the same verdict: the `update_layer`
/// commit (the `WATCH` and the commit scripts) retries it in its loop, and a
/// single-shot site answers a retryable `UNAVAILABLE` instead of `INTERNAL`.
///
/// **Remediation class: backoff only, never a forced reconnect,** as for the
/// `-ASK` shapes: the slot is mid-migration, and a forced reconnect would
/// rebuild the connection map against a topology about to change again.
///
/// # Anti-creep
///
/// Kind-gated and prefix-anchored on the token **with its trailing space**,
/// never a substring search, for the reasons [`is_store_not_serving_reply`]
/// gives. Not matched after an `EXECABORT` prefix: a node answers
/// `-TRYAGAIN` to the command itself (an `EXEC` among them), never inside an
/// `EXECABORT`'s cause. Real-cluster evidence comes from the cluster-gated
/// `slot_migration_disruptive` tests in CI's `test-cluster-disruptive` job,
/// which stage a `MIGRATING` slot and put the real reply through this
/// classifier; elsewhere it is reproduced by a test proxy sending the verbatim
/// text.
pub fn is_tryagain_reply(e: &fred::error::Error) -> bool {
    matches!(e.kind(), FredErrorKind::Unknown) && e.details().starts_with("TRYAGAIN ")
}

/// The reply [`is_tryagain_reply`] exists for, verbatim as Valkey 9.0.3
/// sends it (without the leading `-`; `src/cluster.c:1320`). It arrives as
/// `ErrorKind::Unknown`. Kept beside the predicate for the reason
/// [`STORE_NOT_SERVING_SHAPES`] is.
#[cfg(any(test, feature = "test-support", feature = "test-fault-injection"))]
pub const TRYAGAIN_REPLY: &str = "TRYAGAIN Multiple keys request during rehashing of slot";

/// Whether a fred error is the backing store **not serving commands for a
/// while**: `-LOADING` or `-BUSY`, bare or after an `EXEC`'s `EXECABORT
/// Transaction discarded because of: ` prefix (b.xd1).
///
/// # The two replies (Valkey `unstable`, `src/server.c`)
///
/// - `-LOADING <name> is loading the dataset in memory` — a node that
///   restarted (or a replica resyncing) is replaying its RDB / AOF.
///   `processCommand` (`:4962-4971`) answers it to every command without the
///   `loading` flag, until the replay ends.
/// - `-BUSY …` — a script, a function or a module command has run past
///   `busy-reply-threshold` (default 5000 ms, `valkey.conf`). `processCommand`
///   (`:4980-4991`) answers it to every command without the `allow-busy`
///   flag, until the long command ends or is killed. Three fixed texts
///   (`… busy running a script. You can only call SCRIPT KILL or SHUTDOWN
///   NOSAVE.`, the same with `FUNCTION KILL`, `… busy running a module
///   command.`) and a module's own `BUSY <text>`.
///
/// The fixed texts are built in `createSharedObjectsForCompat` (`:2216-2235`),
/// with `<name>` `Valkey`, or `Redis` under `extended-redis-compatibility`.
/// `STORE_NOT_SERVING_SHAPES` (compiled for tests and the test features
/// only) records every text verbatim.
///
/// fred's `pretty_error` maps neither leading token
/// (`fred-10.1.0/src/protocol/utils.rs:119`, `_ => ErrorKind::Unknown`) and
/// keeps the whole reply as `details`, so each arrives as
/// `ErrorKind::Unknown` with the token first.
///
/// # Why this is transient
///
/// Both refuse every data command on every connection to the node — they
/// say nothing about the request or the connection, and they clear by
/// themselves, so the right remedy is the one for the other transient
/// conditions of a restart: back off and re-issue, within the caller's own
/// budget. Nothing ran: the store refuses before executing the command (or,
/// after the `EXECABORT` prefix, before running any queued command). The
/// service's own scripts can produce `-BUSY` for other clients when one runs
/// long; `MULTI`, `WATCH`, `DISCARD`, `UNWATCH` and `HELLO` carry the
/// `loading` and `allow-busy` flags, so a transaction's bookkeeping and a
/// reconnect still work while the node refuses (valkey `src/commands/*.json`).
/// `EXEC` carries `loading` but not `allow-busy`: under `-LOADING` the queued
/// commands are refused and the `EXEC` answers `EXECABORT Transaction
/// discarded because of previous errors.`; under `-BUSY` the `EXEC` itself is
/// refused too, with the prefixed form this predicate also matches. When a
/// queued command was refused, that refusal is the transaction's cause
/// ([`crate::storage::watched_txn::WatchedExec::Discarded`]; fred's
/// `Transaction::exec(true)` returns it as the error). When `-BUSY` began
/// only after the commands were queued, the `EXEC` alone is refused, and the
/// cause is that prefixed `EXEC` reply itself (`Discarded` falls back to
/// `exec`; fred returns the last frame). The predicate matches both.
///
/// **Remediation class: backoff only, never a forced reconnect.** The
/// connection is healthy and the node is answering; a reconnect lands on the
/// same node and the same refusal. The stream readers keep their
/// subscriptions under their transient budget, as for the router carve-outs.
///
/// # Anti-creep
///
/// Kind-gated and prefix-anchored, never a substring search: `ErrorKind::Unknown`
/// also carries `-UNBLOCKED`, the router's message-matched failures, this
/// service's own tagged Lua verdicts and every other un-parsed server error.
/// Each token carries its **trailing space**, so `-BUSYGROUP` and `-BUSYKEY`
/// (a consumer-group or `RESTORE` conflict — permanent, and issued by no call
/// site today) do not match. The four store write refusals
/// ([`store_write_refusal`]) are deliberately not matched here: they can
/// persist, so they are not transient. A future addition needs the same
/// evidence: a named server origin at a file:line that refuses before
/// anything runs, plus a real-store observation or an explicit note that
/// there is none. **Neither reply has been observed on a real store**: both
/// were reproduced only by a test proxy injecting the verbatim text.
pub fn is_store_not_serving_reply(e: &fred::error::Error) -> bool {
    if !matches!(e.kind(), FredErrorKind::Unknown) {
        return false;
    }
    let details = e.details();
    let reply = details
        .strip_prefix(EXECABORT_DISCARDED_PREFIX)
        .unwrap_or(details);
    reply.starts_with("LOADING ") || reply.starts_with("BUSY ")
}

/// Every reply text [`is_store_not_serving_reply`] exists for, verbatim as
/// Valkey sends it (without the leading `-`), and the single record of where
/// each comes from (b.xd1). Every entry arrives as `ErrorKind::Unknown`.
///
/// Kept beside the predicate for the reason [`BUDGET_EXHAUSTED_SHAPES`] is:
/// tests that pin the predicate build these errors by hand, so one copy here
/// is the only version that can go stale as a unit.
///
/// | # | Text | Origin (Valkey `unstable`, `src/server.c`) |
/// |---|---|---|
/// | 0 | `LOADING Valkey is loading the dataset in memory` | `loadingerr`, `:2218-2219`; answered at `:4963`, `:4969` |
/// | 1 | `LOADING Redis is loading the dataset in memory` | the same under `extended-redis-compatibility` |
/// | 2 | `BUSY Valkey is busy running a script. You can only call SCRIPT KILL or SHUTDOWN NOSAVE.` | `slowevalerr`, `:2220-2222`; answered at `:4988` |
/// | 3 | `BUSY Redis is busy running a script. You can only call SCRIPT KILL or SHUTDOWN NOSAVE.` | the same under `extended-redis-compatibility` |
/// | 4 | `BUSY Valkey is busy running a script. You can only call FUNCTION KILL or SHUTDOWN NOSAVE.` | `slowscripterr`, `:2223-2225`; answered at `:4990` |
/// | 5 | `BUSY Redis is busy running a script. You can only call FUNCTION KILL or SHUTDOWN NOSAVE.` | the same under `extended-redis-compatibility` |
/// | 6 | `BUSY Valkey is busy running a module command.` | `slowmoduleerr`, `:2226-2227`; answered at `:4986` |
/// | 7 | `BUSY Redis is busy running a module command.` | the same under `extended-redis-compatibility` |
/// | 8 | `BUSY the module is compacting its index` | a module's own `busy_module_yield_reply`, `-BUSY %s` at `:4982`; the text after `BUSY ` is the module's (this one is illustrative) |
///
/// **Entry 0 is the shape a backing-store restart produces** with Valkey's
/// default configuration, so call sites that want "the restart reply" index
/// `[0]`; keep it first.
#[cfg(any(test, feature = "test-support", feature = "test-fault-injection"))]
pub const STORE_NOT_SERVING_SHAPES: [&str; 9] = [
    "LOADING Valkey is loading the dataset in memory",
    "LOADING Redis is loading the dataset in memory",
    "BUSY Valkey is busy running a script. You can only call SCRIPT KILL or SHUTDOWN NOSAVE.",
    "BUSY Redis is busy running a script. You can only call SCRIPT KILL or SHUTDOWN NOSAVE.",
    "BUSY Valkey is busy running a script. You can only call FUNCTION KILL or SHUTDOWN NOSAVE.",
    "BUSY Redis is busy running a script. You can only call FUNCTION KILL or SHUTDOWN NOSAVE.",
    "BUSY Valkey is busy running a module command.",
    "BUSY Redis is busy running a module command.",
    "BUSY the module is compacting its index",
];

/// Whether a fred error is the router's "no live connection to route this
/// command to" failure (b.c7a).
///
/// Scope: this predicate matches **one message**, `"Failed to route command…"`,
/// not fred's routing-error family as a whole. fred emits that message from
/// three sites, all in `fred-10.1.0/src/router/commands.rs`:
///
/// - `:150` — `Error::new(ErrorKind::Unknown, "Failed to route command.")`, the
///   `#[cfg(not(feature = "replicas"))]` branch of `write_command`. This is the
///   shape **this service compiles**: fred's `replicas` feature is not enabled.
/// - `:83` — the same message under `ErrorKind::Routing`, in the
///   `create_replica_connection` fallthrough the `replicas` feature routes to.
/// - `:76` — `"Failed to route command to replica."`, `ErrorKind::Routing`,
///   also `replicas`-gated.
///
/// The kind is matched permissively (`Unknown` *or* `Routing`) so a future fred
/// that flips the variant — or a future decision to enable `replicas` — keeps
/// classifying correctly instead of silently regressing to "permanent". The
/// *message* prefix, not the kind, is what does the discriminating.
///
/// **Deliberately NOT covered: the router's retry-budget exhaustion family.**
/// The redirect-exhaustion messages (`"Too many failed write attempts."`,
/// `"Too many redirections."`, `"Max attempts reached."`, `"Unable to route
/// command or reconnect."`) are a *different* condition with a *different*
/// remediation, and b.cv4 settled them into their own sibling predicate,
/// [`is_router_retry_budget_exhausted_error`]. None of them starts with
/// `"Failed to route command"`, so this predicate cannot and must not match
/// them; read that one for the mechanism and the real-cluster datapoint.
///
/// **Why this is transient.** `Router::route`
/// (`fred-10.1.0/src/router/mod.rs:176-207`) returns `None` only when there is
/// no connection to hand back at all. In clustered mode it does *not* require
/// the slot's owner: when the slot's server is missing from the connection map
/// it falls back to `writers.values_mut().next()`, so `None` means the map is
/// **empty** — the router is mid-rebuild after a disconnect storm. That is a
/// connection-level condition, exactly what this classifier's charter covers,
/// and fred agrees: the same code path calls
/// `utils::defer_reconnection` (`fred-10.1.0/src/router/utils.rs:118-149`),
/// which enqueues `RouterCommand::Reconnect { force: false }` (deduped through
/// `pending_reconnection`). fred is already healing itself; the caller just has
/// to wait and retry. It surfaces to the caller at all only because
/// `finish_or_retry_command` found the command's `attempts_remaining` spent —
/// so callers that pin `max_attempts = 1` (the stream-multiplexer readers, b.9uq)
/// see it on the very first miss, with no fred-internal cushion.
///
/// **Why this one belongs in the shared classifier and `-UNBLOCKED` does not.**
/// The routing miss is emitted by `write_command`, which every command on every
/// pool goes through — the write path can observe it exactly as the subscriber
/// can, and both want the same verdict. `-UNBLOCKED`
/// ([`is_client_unblocked_error`]) is structurally confined to the stream
/// reader connections, the only clients with `Blocking::Interrupt` and the
/// only ones issuing a blocking command, so folding *that* in would reclassify a genuine
/// `Unknown` on the write path for no caller's benefit.
pub fn is_router_no_connection_error(e: &fred::error::Error) -> bool {
    matches!(e.kind(), FredErrorKind::Unknown | FredErrorKind::Routing)
        && e.details().starts_with("Failed to route command")
}

/// Whether a fred error is the router giving up on a **redirect it had no
/// attempt budget left to follow** (b.cv4).
///
/// This is predominantly the Valkey-Cluster topology-change error family — the
/// name reflects the dominant cause, not the only one (see **Honest scope**
/// below). `ErrorKind::Cluster` in [`is_transient_fred_error`] covers only the
/// redirects fred successfully *followed*; when the command's budget is spent,
/// the same physical event — a slot moved — instead surfaces as one of four
/// exhaustion messages, under two different kinds, none of which is transient by
/// kind. Without this predicate every one of them is permanent, and a routine
/// resharding drops every live subscriber (`permanent_exits` → terminal
/// `Unavailable`) and turns every single-shot read into a non-retriable
/// `Internal`.
///
/// # The origins (fred 10.1.0 with the vendored patch, verified against the source)
///
/// Four messages, five origins: `"Max attempts reached."` has two.
///
/// - `protocol/command.rs:1778` — `ErrorKind::Unknown`, `"Too many failed write
///   attempts."`, from `Command::decr_check_attempted` (fn at `:1776`). This is
///   the **only** origin of that message.
/// - `protocol/command.rs:1791` — `ErrorKind::Routing`, `"Too many
///   redirections."`, from `Command::decr_check_redirections` (fn at `:1789`).
/// - `router/utils.rs:469` — `ErrorKind::Routing`, `"Max attempts reached."`,
///   from `send_asking_with_policy`, raised for an `-ASK` to a command with no
///   write attempt left, before anything is written.
/// - `live_edit/ask.rs:180-183` (the vendored patch's `live_edit::ask::follow`)
///   — `ErrorKind::Routing`, `"Max attempts reached."`, raised for an `-ASK` to
///   a **script** command (`EVAL`, `EVALSHA`, their `_RO` forms, `FCALL`,
///   `FCALL_RO`) that still has a write attempt left, before anything is
///   written or sent to the target: a script is never followed, because the
///   target would read the undeclared keys that have not arrived as absent
///   (`vendor/fred-10.1.0.SOURCE.md`, change 8). The text is the first
///   origin's on purpose: it is the answer a script got before the patch.
/// - `router/utils.rs:454` — `ErrorKind::Routing`, `"Unable to route command or
///   reconnect."`, the `next_sleep!` fallback in `send_asking_with_policy`,
///   which the vendored fred no longer reaches (below).
///
/// # The mechanism: attempts, not redirections
///
/// `write_command` (`router/commands.rs:100`) calls `decr_check_attempted` at
/// `:111` on **every** write — including the re-write that *follows* a redirect.
/// So a redirect costs an attempt, and a command pinned to one attempt can never
/// complete a hop:
///
/// - `-MOVED` → `process_moved` (`router/commands.rs:288`) syncs the cluster,
///   passes `decr_check_redirections` (`:304`), and re-enters `write_command` at
///   `:309` with `attempts_remaining` already 0 → dies `Unknown` / `"Too many
///   failed write attempts."`.
/// - `-ASK` → `process_ask` (`router/commands.rs:251`) passes
///   `decr_check_redirections` (`:261`). A script command with a write
///   attempt left fails `Routing` / `"Max attempts reached."` at once, with
///   nothing written or sent (fred's `live_edit::ask::follow`,
///   `live_edit/ask.rs:180-183`; see the origins above). Any other command
///   with a write attempt left is followed by the vendored patch (`:265`,
///   `live_edit::ask::follow`): it writes a pipelined `ASKING` and then the
///   command to the `ASK` target's connection, opening one when the client has
///   none, and takes the command's attempt as `write_command` would. (Stock
///   fred 10.1.0 re-sent the command to the slot's owner, the source, which
///   answered `-ASK` again until the attempts ran out; no `-ASK` was ever
///   followed. `vendor/fred-10.1.0.SOURCE.md` records the change.) With the
///   attempts already at zero, it forwards them into `send_asking_with_policy`,
///   which returns `Routing` / `"Max attempts reached."` at `utils.rs:468-469`
///   before sending anything, and `process_ask` answers the caller with that
///   error (`:274-277`). Stock fred 10.1.0 then also returned it, which ended
///   the client's router task for good while the client still read
///   `Connected`; the vendored patch makes it return `Ok(())`, as
///   `process_moved` does, so the `ASK` fails this one command and the router
///   keeps running. The caller sees the same error either way.
///
/// This service reaches that state deliberately: every stream-multiplexer
/// reader pins `CONSUMER_XREAD_MAX_ATTEMPTS = 1` (`storage::stream_mux::reader`,
/// b.9uq) on its `XREAD`, so it can never follow a redirect internally. The
/// pin stays — it is what stops an orphaned blocking `XREAD` being resurrected
/// onto another connection — and this classification is what makes the
/// reader's *own* cursor-aware retry the recovery path instead of a tear-down.
///
/// **`max_redirections: Some(0)` is inert.** `Command::inherit_options`
/// (`protocol/command.rs:1882-1888`) treats `0` as "unset" for *both* counters
/// and restores the client default (`max_redirections: 5`,
/// `types/config.rs:480`). Anything trying to forbid redirects by zeroing that
/// field is silently getting the default; the attempt budget is the only lever.
///
/// # Honest scope
///
/// **Not every one of these is a redirect.** `decr_check_attempted` fires on
/// *any* entry to `write_command` with a spent budget, and a redirect is only
/// one of the ways to get there. The other is a **dropped connection**:
/// `Router::retry_buffer` (`router/mod.rs:373-390`) replays a disconnected
/// connection's in-flight commands straight back into `write_command`, filtering
/// only on `timed_out` — there is no `should_finish_with_error` /
/// `attempts_remaining` check on that path. So a `max_attempts: 1` command whose
/// connection is killed comes back as `Unknown` / `"Too many failed write
/// attempts."` with no `MOVED` or `ASK` anywhere in the story. The
/// classification is right either way — a dropped connection is exactly as
/// retriable as a moved slot — but do not read this predicate's name as a
/// guarantee about *why* it matched. In particular
/// `cluster_subscriber_observed_errors_are_all_classified_transient`, which
/// disturbs a cluster with `CLIENT KILL` and no resharding at all, can reach
/// this predicate by precisely that route.
///
/// The same is true **off** a cluster entirely: on a standalone deployment a
/// command that fails to write repeatedly burns fred's default
/// `max_command_attempts` (3, `types/config.rs:481`) and ends at the same
/// `decr_check_attempted` — no cluster, no redirect, no `max_attempts` override
/// needed. Transient is still the right verdict there: repeated write failures
/// are a connection-level condition, which is exactly this classifier's charter,
/// and both consumers of the verdict bound their own retrying (the stream
/// multiplexer's readers by `MAX_CONSECUTIVE_TRANSIENT_FAILURES`, `update_layer` by
/// `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS`), so a genuinely dead backend
/// still surfaces rather than looping forever.
///
/// **What has been measured.** The `Unknown` /
/// `"Too many failed write attempts."` shape was **observed** on a real 3-shard
/// cluster (pipeline 64932151, job 414867156 — an `XLEN` pinned to a non-owner
/// primary). `"Max attempts reached."` (a one-attempt command given `-ASK`) and
/// `"Too many redirections."` (an `-ASK` on the `txn_pool`, whose client config
/// sets `max_redirections = 0`) were reproduced against a mock cluster node,
/// with clients built as [`ValkeyClient::connect`] builds them (the evidence
/// table in `vendor/fred-10.1.0.SOURCE.md`). `"Max attempts reached."` was
/// also seen on a real cluster before the vendored `-ASK` follow, from a
/// three-attempt plain `SCARD` that stock fred re-sent to the source until its
/// attempts ran out (CI pipeline 73100357, job 481449796); its script origin
/// is pinned by a two-node mock cluster test, not seen on a real cluster.
/// `"Unable to route command or reconnect."` is read from fred's
/// source only, and is *unreachable* in the vendored fred: it is only the
/// `unwrap_or_else` fallback taken when `router.route()` returned `None`
/// **and** `next_reconnection_delay` errored, and `send_asking_with_policy` is
/// now called only for a command with no attempt left, which returns before
/// routing anything. (Before that, this service's
/// `ReconnectPolicy::new_exponential(0, …)` — unlimited attempts, see
/// [`ValkeyClient::connect`] — already kept `next_reconnection_delay` from
/// erroring.) It is matched for completeness, not because it is expected.
///
/// # Remediation class
///
/// Class 2, **backoff only** — this must NOT trigger `force_reconnection()`,
/// the same asymmetry [`is_router_no_connection_error`] carries. fred is already
/// healing underneath on both routes into this predicate: on the redirect route
/// `process_moved` ran `sync_cluster_with_policy` before it gave up, and on the
/// connection-drop route the replay only happens *because* the router is
/// reconnecting. Forcing on top would tear down connections mid-rebuild, at the
/// moment the cluster is weakest. The caller's job is to wait and re-issue from
/// its durable cursor.
///
/// The `-ASK` shapes (`"Too many redirections."` from `process_ask`'s budget
/// check; `"Max attempts reached."` from `send_asking_with_policy`, or from
/// the vendored `live_edit::ask::follow` for a script command) are the
/// exception to "already healing": they are raised before any sync or
/// reconnect. Nothing needs healing either: with the vendored patch the router
/// answers the caller and keeps running, and the connection map is intact. (On
/// stock fred 10.1.0 the router task ended on the second and nothing healed
/// the client; `storage::router_supervisor` now restarts a router that ends any
/// other way.) Forcing is still wrong there: an `ASK` means the slot is
/// mid-migration, so a forced reconnect would rebuild the connection map
/// against a topology about to change again. Backing off and re-issuing is the
/// remedy on every shape.
///
/// # Anti-creep
///
/// The match is **kind-gated and prefix-anchored**, never a substring search and
/// never a bare kind widening: `ErrorKind::Unknown` also carries `-UNBLOCKED`
/// (see [`is_client_unblocked_error`]) and every un-parsed server error, and
/// `ErrorKind::Routing` also carries the `replicas`-gated routing failures. A
/// future addition here needs the same shape and the same evidence: a named
/// fred origin at a file:line, plus a real-cluster observation or an explicit
/// note that there is none.
pub fn is_router_retry_budget_exhausted_error(e: &fred::error::Error) -> bool {
    match e.kind() {
        FredErrorKind::Unknown => e.details().starts_with("Too many failed write attempts"),
        FredErrorKind::Routing => {
            let details = e.details();
            details.starts_with("Too many redirections")
                || details.starts_with("Max attempts reached")
                || details.starts_with("Unable to route command or reconnect")
        }
        _ => false,
    }
}

/// The authoritative `(kind, message)` table for
/// [`is_router_retry_budget_exhausted_error`], and the single record of where
/// each shape comes from in fred (b.cv4).
///
/// # Why this is here rather than in the tests
///
/// The test sites that pin this contract — the unit block at the bottom of this
/// file and `tests/integration/storage/transient_classification_test.rs` — use
/// **only hand-built errors**, because every origin lives in fred's private
/// `mod router` / `mod protocol`, or in the vendored patch's
/// `live_edit::ask::follow`, which a test reaches only through a mock cluster
/// node, and none can be provoked from outside the crate on its own.
/// That has a sharp consequence: if fred reworded one of these messages, none of
/// those tests would fail. They restate the predicate's assumption rather than
/// verifying it. Independent copies of an unverifiable assumption are chances
/// to update some of them and leave another quietly describing a contract that
/// no longer exists. One copy, here, next to the predicate it describes, is the
/// only version that can go stale as a unit.
///
/// The real closing guard is a live cluster: a cluster-gated test in
/// `tests/integration/storage/cluster_subscriber_test.rs` pins a command on a
/// layer-stream key to a non-owner primary with `max_attempts = 1`, so the
/// real `-MOVED` is not followed, and puts the error an actual 3-shard cluster
/// returns through [`is_transient_fred_error`].
///
/// # Provenance (fred 10.1.0, verified against the vendored source)
///
/// | # | Kind | Message | Origin |
/// |---|---|---|---|
/// | 0 | `Unknown` | `"Too many failed write attempts."` | `protocol/command.rs:1778`, `Command::decr_check_attempted` (fn at `:1776`) — the **only** origin of this text |
/// | 1 | `Routing` | `"Too many redirections."` | `protocol/command.rs:1791`, `Command::decr_check_redirections` (fn at `:1789`) — what an `-ASK` gives a `txn_pool` command (`max_redirections = 0`); the router survives |
/// | 2 | `Routing` | `"Max attempts reached."` | Two origins, both raised before anything is written: `router/utils.rs:469`, `send_asking_with_policy` — the `-ASK` twin, for a command with no write attempt left; with the vendored patch it fails only the command (stock fred 10.1.0 also ended the client's router). And `live_edit/ask.rs:180-183`, the vendored `live_edit::ask::follow` — an `-ASK` to a script command with a write attempt left, which is never followed |
/// | 3 | `Routing` | `"Unable to route command or reconnect."` | `router/utils.rs:454`, the same function's `next_sleep!` fallback; unreachable in the vendored fred, which calls that function only with no attempt left |
///
/// **Entry 0 is load-bearing by position.** It is the only shape ever observed
/// on a real cluster (pipeline 64932151, job 414867156) and the only one
/// reachable without a redirect at all — see the `retry_buffer` replay path in
/// [`is_router_retry_budget_exhausted_error`]'s **Honest scope**. Call sites
/// that want "the one shape we have actually seen" index `[0]` rather than
/// re-spelling the string, so keep it first.
///
/// The messages are stored **without** trailing-period sensitivity in mind: the
/// predicate matches by `starts_with`, so these full strings exercise the real
/// prefixes rather than the prefixes alone.
///
/// Also compiled under `test-fault-injection`: the stream multiplexer's
/// `MuxFault::Hang` seam fails a hung read with entry 0 on a forced reconnect.
#[cfg(any(test, feature = "test-support", feature = "test-fault-injection"))]
pub const BUDGET_EXHAUSTED_SHAPES: [(FredErrorKind, &str); 4] = [
    (FredErrorKind::Unknown, "Too many failed write attempts."),
    (FredErrorKind::Routing, "Too many redirections."),
    (FredErrorKind::Routing, "Max attempts reached."),
    (
        FredErrorKind::Routing,
        "Unable to route command or reconnect.",
    ),
];

/// The message fred's `Transaction` router fails a transaction with when it
/// meets a `-MOVED` or `-ASK` past its redirection budget (`"Max redirections
/// exceeded"`, `vendor/fred-10.1.0/src/router/transactions.rs:65-67`). On the
/// `txn_pool` that budget is zero, so this is what any redirect gives a fred
/// `Transaction` there. See [`is_transaction_router_exhausted_error`].
const TRANSACTION_REDIRECTIONS_EXHAUSTED: &str = "Max redirections exceeded";

/// The message fred's `Transaction` router fails a transaction with when it
/// runs out of attempts on an arm that has no underlying error to forward
/// (`"Max attempts exceeded"`, `transactions.rs:59-63`). See
/// [`is_transaction_router_exhausted_error`].
const TRANSACTION_ATTEMPTS_EXHAUSTED: &str = "Max attempts exceeded";

/// Whether a fred error is fred's **`Transaction` router** giving up — out of
/// redirections or out of attempts (b.bm5 sub-task (a), b.cyu).
///
/// fred runs a `Transaction` (`client.multi()` … `exec()`) through its own
/// router loop, `router/transactions.rs::send`, not through `write_command`,
/// so it spells its exhaustion differently from the per-command family
/// [`is_router_retry_budget_exhausted_error`] matches. Both texts are
/// `ErrorKind::Unknown`, which is not transient by kind.
///
/// # The origins (vendored fred 10.1.0)
///
/// - `"Max redirections exceeded"` — `max_redirections_error`
///   (`transactions.rs:65-67`), sent when a `-MOVED` (`:212-217`) or an
///   `-ASK` (`:240-243`) in the transaction's replies takes the redirection
///   count past the last command's budget. The `txn_pool` pins that budget
///   to zero (`ValkeyClient::connect`), so **every** redirect a fred
///   `Transaction` meets there ends this way: a slot that moved, or a slot
///   mid-migration. The router gives up before it re-syncs anything, which is
///   why every caller sends its `EXEC` through `ValkeyClient::exec_txn`, which
///   re-syncs the checkout's routing on this error.
/// - `"Max attempts exceeded"` — `max_attempts_error`'s fallback
///   (`transactions.rs:59-63`), sent only from the `retry!(None)` arms: a
///   failed `router.drain_all` (`:142-145`) or no connection to route the
///   transaction to (`:147-156`). Every other attempt failure forwards its
///   real error (`retry!(Some(err))`), an `IO` or `Timeout` the kind-level
///   classification already covers.
///
/// # Why this is transient
///
/// **Within one attempt, nothing ran.** fred reads every reply of the
/// attempt, `EXEC`'s included, before it examines any (`:30-51`), and returns
/// the first redirect (`process_responses`, `:73-97`). A redirect at the
/// `EXEC` position is answered by Valkey's pre-execution cluster check, which
/// discards the transaction first (Valkey 9.0.3 `server.c:4315-4316`); a
/// redirect at a queued command (Valkey ≤ 8.x and Redis 7.x check each
/// queued command) flags the transaction, and the `EXEC` answers `EXECABORT`
/// (`multi.c:124-129`, `:185-193`). Commands that `EXEC` runs are never
/// re-checked (`multi.c:240-243`), and fred reads only top-level frames for a
/// redirect. The `retry!(None)` arms write nothing in their attempt.
///
/// **Across attempts, something may have run.** fred replays the whole
/// transaction once on this pool (`:115-133`; the txn pool's COST comment in
/// [`ValkeyClient::connect`]): when the first send's `EXEC` landed and its
/// reply was lost, the replay's redirect or failed drain is what the caller
/// sees. So this verdict means "retry", never "nothing was applied" — the
/// same rule `storage::watched_txn` keeps by mapping only Valkey's own bare
/// `TRYAGAIN ` to `Discarded`. Each fred `Transaction` user tolerates that:
///
/// - `SessionStore::delete` is idempotent: a re-send after a landed `EXEC`
///   finds the session gone.
/// - `LayerStore::write_object` (the `CreateLayer` materialization) fails
///   before the create's mapping flip, so no layer is made. A landed first
///   send leaves an unmapped, complete orphan object, which the orphan sweep
///   reclaims; a re-send creates the layer under a new `object_id` (with the
///   same `operation_id` it answers `APPLIED`).
/// - The session checkpoint applier (`checkpoint::session`) does not consult
///   this classifier at all: it grades every `EXEC` error `Transient`, and its
///   writes are idempotent against their own earlier send.
///
/// **Remediation class: backoff only, never a forced reconnect.** A redirect
/// means the slot moved or is moving; the fix for a moved slot is a routing
/// sync of the one member that met it, which the callers run, not a rebuilt
/// connection map. The `retry!(None)` arms have already reconnected once
/// inside fred.
///
/// # Anti-creep
///
/// Kind-gated (`Unknown` only) and prefix-anchored on the two exact texts,
/// never a substring search, for the reasons
/// [`is_router_retry_budget_exhausted_error`] gives. Kept out of
/// [`BUDGET_EXHAUSTED_SHAPES`], whose tests index its entries by position;
/// [`TRANSACTION_ROUTER_EXHAUSTED_SHAPES`] is this predicate's own table.
///
/// **What has been observed.** `"Max redirections exceeded"` is put through
/// this classifier on a real cluster by the cluster-gated
/// `slot_migration_disruptive` tests in CI's `test-cluster-disruptive` job,
/// which stage a migrating slot under `DeleteSession`; elsewhere it is
/// reproduced by a test proxy answering a transaction with a redirect.
/// `"Max attempts exceeded"` is read from fred's source only.
pub fn is_transaction_router_exhausted_error(e: &fred::error::Error) -> bool {
    if !matches!(e.kind(), FredErrorKind::Unknown) {
        return false;
    }
    let details = e.details();
    details.starts_with(TRANSACTION_REDIRECTIONS_EXHAUSTED)
        || details.starts_with(TRANSACTION_ATTEMPTS_EXHAUSTED)
}

/// Whether a fred error is fred's `Transaction` router out of
/// **redirections** — the one [`is_transaction_router_exhausted_error`] shape
/// a routing sync can help. Same kind gate and prefix anchoring.
fn is_transaction_redirections_exhausted_error(e: &fred::error::Error) -> bool {
    matches!(e.kind(), FredErrorKind::Unknown)
        && e.details().starts_with(TRANSACTION_REDIRECTIONS_EXHAUSTED)
}

/// The authoritative `(kind, message)` table for
/// [`is_transaction_router_exhausted_error`], verbatim as fred builds each
/// error (`vendor/fred-10.1.0/src/router/transactions.rs:61`, `:66`). Kept
/// beside the predicate for the reason [`BUDGET_EXHAUSTED_SHAPES`] is: the
/// tests that pin the predicate build these errors by hand.
#[cfg(any(test, feature = "test-support", feature = "test-fault-injection"))]
pub const TRANSACTION_ROUTER_EXHAUSTED_SHAPES: [(FredErrorKind, &str); 2] = [
    (FredErrorKind::Unknown, TRANSACTION_REDIRECTIONS_EXHAUSTED),
    (FredErrorKind::Unknown, TRANSACTION_ATTEMPTS_EXHAUSTED),
];

/// Why a `txn_pool` checkout's routing sync
/// ([`ValkeyClient::sync_txn_checkout_routing`]) did not complete. Each
/// caller logs it at WARN with its own fields and carries on: a sync never
/// changes the caller's answer.
#[derive(Debug)]
pub(crate) enum TxnRoutingSyncFailure {
    /// The member's router failed the sync (it ended or closed while the sync
    /// was in it, or the sync never reached it).
    Failed(fred::error::Error),
    /// The sync outlived the client's command timeout, `timeout_ms`; it keeps
    /// running in the member's router.
    TimedOut { timeout_ms: u64 },
}

/// Whether a fred error is the server's `-UNBLOCKED` reply to a `CLIENT
/// UNBLOCK ... ERROR` (b.9uq round 4).
///
/// fred's [`Blocking::Interrupt`](fred::types::config::Blocking::Interrupt) —
/// which the stream reader connections and only they enable, see
/// `ValkeyClient::reader_builder` — sends `CLIENT UNBLOCK <id> ERROR` when a command
/// would otherwise queue behind a blocking one on the same client. The blocked
/// command then fails with `-UNBLOCKED client unblocked via CLIENT UNBLOCK`,
/// which fred's `pretty_error` classifies as `ErrorKind::Unknown` (its match
/// arms cover `ERR` / `WRONGTYPE` / `NOAUTH` / `WRONGPASS` / `MOVED` / `ASK` /
/// `CLUSTERDOWN` / `Invalid …` and nothing else). `Unknown` is deliberately
/// *not* transient by kind in [`is_transient_fred_error`] — and that
/// classifier's `Unknown`-keyed message carve-outs
/// ([`is_router_no_connection_error`],
/// [`is_router_retry_budget_exhausted_error`],
/// [`is_transaction_router_exhausted_error`],
/// [`is_store_not_serving_reply`] and [`is_tryagain_reply`]) all key on
/// different messages — so without
/// this predicate a stream reader would treat a deliberate unblock as a
/// permanent failure and tear its subscriptions down.
///
/// Retrying is unambiguously correct here: the reader re-plans its `XREAD` from
/// its subscriptions' durable `{sequence_num}-0` cursors, so an interrupted read
/// loses nothing. The predicate is deliberately kept *out* of
/// [`is_transient_fred_error`] — no other pool issues a blocking command, so no
/// other caller can legitimately observe this error, and widening the shared
/// classifier would silently reclassify a genuine `Unknown` on the write path.
pub fn is_client_unblocked_error(e: &fred::error::Error) -> bool {
    matches!(e.kind(), FredErrorKind::Unknown) && e.details().starts_with("UNBLOCKED")
}

/// Whether a fred error is a raw `-MOVED` reply: the server saying another
/// node now owns the command's hash slot.
///
/// fred's `pretty_error` maps `MOVED`, `ASK` and `CLUSTERDOWN` all to
/// `ErrorKind::Cluster` and keeps the whole reply as `details`
/// (`fred-10.1.0/src/protocol/utils.rs:113`, `:123-127`), so the `"MOVED "`
/// prefix is what tells a redirect from the other two.
///
/// Only a single-node client ever sees one: a cluster client follows the
/// redirect itself, and one with no write attempt left reports it as
/// [`is_router_retry_budget_exhausted_error`]'s shapes instead. The stream
/// readers are single-node clients — on a cluster each is connected to its
/// bound slot's primary alone (`storage::stream_mux`) — so a slot that moves
/// reaches a reader as this error, and the reader re-resolves the slot's owner
/// and moves its connection there rather than retrying the same node.
///
/// Reader-only, like [`is_client_unblocked_error`]: [`is_transient_fred_error`]
/// is not changed and still calls the error transient by its `Cluster` kind.
pub fn is_moved_redirect(e: &fred::error::Error) -> bool {
    matches!(e.kind(), FredErrorKind::Cluster) && e.details().starts_with("MOVED ")
}

/// Whether a fred error is the **server** rejecting the credential we sent —
/// `-WRONGPASS` or `-NOAUTH` (b.cbe).
///
/// # Provenance
///
/// fred's `pretty_error` (`fred-10.1.0/src/protocol/utils.rs:96-129`) splits the
/// error frame on whitespace and maps the leading token: `"NOAUTH" |
/// "WRONGPASS" => ErrorKind::Auth` at `:112`. The `details` it stores is the
/// **whole** response string (`:123-127`), so the server's leading token is
/// still there to anchor on.
///
/// # Why the prefix anchor is load-bearing, not decoration
///
/// `ErrorKind::Auth` alone is **not** enough, and this is the whole reason the
/// predicate exists rather than a bare `matches!`. This service raises
/// `ErrorKind::Auth` itself: `TokenFileCredentialProvider::fetch` returns
/// `Error::new(ErrorKind::Auth, "accessTokenFile valkey credential
/// unavailable…")` when no usable token is cached (b.d6b). That is the exact
/// *opposite* fault — *we have no credential to send*, not *the server rejected
/// the credential we sent* — and it already carries its own diagnosis. Firing
/// the four-cause operator checklist on it would send an operator hunting an
/// access-policy assignment when the real problem is a sidecar that stopped
/// writing the token file. The prefix separates the two.
///
/// # Anti-creep
///
/// Kind-gated and prefix-anchored, never a substring search — the same rule the
/// [`is_router_retry_budget_exhausted_error`] family follows. A future addition
/// needs a named fred origin at a file:line, and must be a token the **server**
/// emits.
///
/// # Both tokens here; the steady-state caller narrows further
///
/// This predicate keeps `NOAUTH` because the **connect** path needs it: a
/// `NOAUTH` on the very first connection is terminal, not something the client
/// absorbs. The steady-state hook in [`spawn_error_metric`] deliberately adds a
/// `WRONGPASS`-only condition on top of this call — see the comment there before
/// "simplifying" that site back to a bare call.
pub fn is_server_auth_rejection(e: &fred::error::Error) -> bool {
    matches!(e.kind(), FredErrorKind::Auth)
        && (e.details().starts_with("WRONGPASS") || e.details().starts_with("NOAUTH"))
}

/// Whether a fred error is the **server**'s `-WRONGTYPE` reply — the operation
/// was issued against a key holding a different Valkey type (b.atj).
///
/// # Counting and logging ONLY — never control flow
///
/// This predicate exists so a snapshot read refused on a wrong-typed key can
/// tick a counter and name the key in a log line. It must **never** decide what
/// the caller does next. The b.459 / b.zvd fail-closed posture (SDD §7.5.4) is
/// that a `WRONGTYPE` on a snapshot read's commands is an `Internal`
/// refusal exactly like every other permanent error, precisely *because* a
/// message-matched classification that mis-fires re-opens the silent structural
/// truncation b.zvd closed — a mis-detected `WRONGTYPE` treated as "no
/// children" lets a structural commit `DEL` a real children list under a valid
/// `WATCH`. A mis-detection *here* mis-ticks a counter and mis-writes a log
/// line; it cannot truncate data. That asymmetry is the whole licence for the
/// predicate, so keep [`map_snapshot_load_err`] free of it.
///
/// # Provenance (fred 10.1.0, verified against the vendored source)
///
/// `pretty_error` (`fred-10.1.0/src/protocol/utils.rs:96-128`) splits the error
/// frame on whitespace and maps the leading token: `"WRONGTYPE" =>
/// ErrorKind::InvalidArgument` at `:111`, storing the **whole** reply as
/// `details` (`:123-127`), so the server's leading token is still there to
/// anchor on.
///
/// # Why the prefix anchor is load-bearing
///
/// `ErrorKind::InvalidArgument` alone over-matches: the same `pretty_error`
/// maps a leading `"Invalid"` token onto it (`:114-116`), and fred raises it
/// from client-side argument validation that never reached the server. The
/// kind check is equally load-bearing in the other direction — it separates the
/// server's reply from this service's *own* `WRONGTYPE_KEY:`-tagged Lua verdict
/// (`storage::layer::KEYSPACE_WRONGTYPE_TAG`), which arrives as
/// `ErrorKind::Unknown` and is already counted at its own site by
/// [`crate::observability::metrics::Metrics::inc_keyspace_type_error`]. Double
/// counting one refusal on two counters would break both baselines.
///
/// # Upgrade path
///
/// If a future fred stops setting `InvalidArgument` for `-WRONGTYPE`, replace
/// the whole predicate with an explicit `TYPE <key>` probe at the tick site
/// rather than widening the match — the probe is a real answer where a broader
/// message match is a guess. Deliberately kept out of
/// [`is_transient_fred_error`]: a wrong-typed key is permanent by definition.
pub fn is_server_wrongtype_error(e: &fred::error::Error) -> bool {
    matches!(e.kind(), FredErrorKind::InvalidArgument) && e.details().starts_with("WRONGTYPE ")
}

/// Whether a fred error is the server's `WRONGTYPE` reply, for the sites that
/// turn a wrong-typed key into a **refusal**: fred's `InvalidArgument` kind
/// AND `details` prefix-anchored on `WRONGTYPE`, never a bare substring
/// search (the kind and anchor reasons are [`is_server_wrongtype_error`]'s).
///
/// Its callers: the checkpoint restore's object-key reads
/// (`checkpoint::apply::classify_object_key_read_error`) and the restore
/// fence's header reads (`storage::layer`'s
/// `classify_restore_header_read_error`), which refuse the restore as
/// permanent instead of retrying it, and the structural cache cold start's
/// commit-token `HGET` (`storage::structural_cache_cold_start`), which answers
/// the commit paths' `WRONGTYPE_KEY` refusal. Each turns one refusal into
/// another, never into "no data", so it fails closed whichever way it
/// matches. That is why these sites may branch on it, while
/// [`is_server_wrongtype_error`] stays for counting and logging only (its
/// documentation gives the snapshot-read hazard a branch there would open).
/// Its anchor has no trailing space, as each of those sites tested before it
/// was shared.
pub(crate) fn is_wrongtype_reply(e: &fred::error::Error) -> bool {
    matches!(e.kind(), FredErrorKind::InvalidArgument) && e.details().starts_with("WRONGTYPE")
}

/// The prefix Valkey puts on an `EXEC` it refused at `EXEC` time, before any
/// queued command ran: `execCommandAbort` (valkey `src/multi.c`) answers
/// `-EXECABORT Transaction discarded because of: <the refusal>`, the refusal
/// with its leading `-` stripped — the shape a write-gated `EXEC` takes when
/// `processCommand` rejects it as a whole (`rejectCommand` → `execCommandAbort`,
/// valkey `src/server.c`).
const EXECABORT_DISCARDED_PREFIX: &str = "EXECABORT Transaction discarded because of: ";

/// Why the backing store refused a write before applying anything: one of the
/// four write-gating refusals [`store_write_refusal`] recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreWriteRefusal {
    /// `-OOM`: `used_memory` is above `maxmemory`. The required `noeviction`
    /// policy evicts nothing. Noting it starts the GC's out-of-memory
    /// collection (`storage::store_oom`), which frees what released external
    /// values hold; memory held by data in use stays until an operator raises
    /// `maxmemory` or removes data.
    OutOfMemory,
    /// `-MISCONF`: the store cannot persist (a failing RDB save under
    /// `stop-writes-on-bgsave-error`, or a failing AOF write).
    Misconf,
    /// `-READONLY`: the write reached a read-only replica.
    ReadOnly,
    /// `-NOREPLICAS`: too few replicas for `min-replicas-to-write`.
    NoReplicas,
}

impl StoreWriteRefusal {
    /// The `reason` label of `live_edit_store_write_refused_total`, one of
    /// [`crate::observability::metrics::STORE_WRITE_REFUSED_REASONS`].
    pub fn reason_label(self) -> &'static str {
        match self {
            Self::OutOfMemory => {
                crate::observability::metrics::STORE_WRITE_REFUSED_REASON_OUT_OF_MEMORY
            }
            Self::Misconf => crate::observability::metrics::STORE_WRITE_REFUSED_REASON_MISCONF,
            Self::ReadOnly => crate::observability::metrics::STORE_WRITE_REFUSED_REASON_READONLY,
            Self::NoReplicas => {
                crate::observability::metrics::STORE_WRITE_REFUSED_REASON_NOREPLICAS
            }
        }
    }

    /// The condition the client-facing message names.
    fn condition(self) -> &'static str {
        match self {
            Self::OutOfMemory => "it is out of memory",
            Self::Misconf => "it cannot persist data to disk",
            Self::ReadOnly => "the node it reached is a read-only replica",
            Self::NoReplicas => "too few of its replicas are reachable to accept writes",
        }
    }

    /// The answer to a refused store write: `UNAVAILABLE`, saying the store
    /// refused, that nothing was applied, and to retry — the upload-capacity
    /// refusals' convention (`grpc::upload_admission`). Worded for any RPC,
    /// not only an upload: the `ext_store` stage also serves a read's lazy
    /// external-value rewrite after a checkpoint recovery
    /// (`external_values::recover_ext_value_via_checkpoint`), where the write
    /// is one the request needed rather than the request itself. The
    /// message never starts with the overtaken prefix
    /// (`layer::OPERATION_OVERTAKEN_MESSAGE`), which clients match on.
    ///
    /// It carries the retry hint
    /// [`STORE_WRITE_REFUSED`](crate::error::refusal_reason::STORE_WRITE_REFUSED)
    /// (metadata `condition` = [`Self::reason_label`]), whose `ErrorInfo` on
    /// an upload tells the client this call applied nothing: every site that
    /// answers it is one where the refused write leaves nothing of the call a
    /// client can observe ([`store_write_refusal`] lists them). An
    /// `UpdateLayer` commit refused after a counted retry loses the hint at
    /// the call's tail (`layer::outcome_unknown_after_retry`).
    pub fn into_service_error(self) -> crate::error::ServiceError {
        use crate::error::{refusal_metadata, refusal_reason, RefusalDetail};
        RefusalDetail::new(
            refusal_reason::STORE_WRITE_REFUSED,
            Some(crate::error::STORE_WRITE_REFUSED_RETRY_DELAY),
        )
        .with_metadata(refusal_metadata::CONDITION, self.reason_label())
        .into_unavailable_error(format!(
            "the backing store refused a write this request needed because {}{}",
            self.condition(),
            STORE_WRITE_NOTHING_APPLIED
        ))
    }

    /// The answer at the `create_commit` stage: plain `UNAVAILABLE` saying
    /// the store refused, with no retry hint and no nothing-applied claim.
    ///
    /// A `CreateLayer` refusal at its commit stage never carries the hint
    /// (SDD §18.1; its pre-commit refusals — admission, receive growth, the
    /// external-value pre-store — keep theirs): a wrong
    /// "applied nothing" invites a re-send that can replace a layer, which is
    /// destructive, while a missing one only leaves the outcome unknown. And
    /// one of the stage's writes cannot prove it: the overwrite `superseded`
    /// mark is a data-pool `HSET` fred replays after a dropped connection, so
    /// a refused replay can follow a mark that landed.
    pub fn into_unhinted_service_error(self) -> crate::error::ServiceError {
        crate::error::ServiceError::Unavailable(format!(
            "the backing store refused a write this request needed because {}; retry",
            self.condition()
        ))
    }

    /// The mappers' answer ([`store_write_refusal`], "Two answers"): plain
    /// `UNAVAILABLE` naming the failed command `op` and the condition, ending
    /// `; retry`, with no retry hint and no nothing-applied claim (b.xd1).
    /// The message starts with `op`, a command label, so never with the
    /// overtaken prefix (`layer::OPERATION_OVERTAKEN_MESSAGE`), which clients
    /// match on.
    fn into_mapped_service_error(self, op: &str) -> crate::error::ServiceError {
        crate::error::ServiceError::Unavailable(format!(
            "{op}: the backing store refused a write this request needed because {}; retry",
            self.condition()
        ))
    }
}

/// The end of [`StoreWriteRefusal::into_service_error`]'s message: its
/// nothing-applied claim, which `layer::outcome_unknown_after_retry` strips.
pub(crate) const STORE_WRITE_NOTHING_APPLIED: &str = "; nothing was applied, retry";

/// Whether a fred error is the backing store refusing a **write** before
/// applying anything: `-OOM`, `-MISCONF`, `-READONLY` or `-NOREPLICAS`, or
/// the same token after an `EXEC`'s `EXECABORT Transaction discarded because
/// of: ` prefix ([`EXECABORT_DISCARDED_PREFIX`]).
///
/// fred's `pretty_error` maps none of the four tokens, so each arrives as
/// `ErrorKind::Unknown` with the whole reply in `details`
/// (`fred-10.1.0/src/protocol/utils.rs:96-129`). Valkey
/// raises them only before a write runs — `processCommand` /
/// `scriptVerifyWriteCommandAllow` / `scriptVerifyOOM` for a command or a
/// script's first write, `rejectCommand` → `execCommandAbort` for a whole
/// `EXEC` — so the write they answer applied nothing.
///
/// # Two answers
///
/// **Everywhere, through the mappers**: [`map_snapshot_load_err`] (and its
/// alias [`map_transient_fred_err`]) and [`map_fred_err_for_grpc`] answer a
/// store write refusal with a plain `Unavailable` naming the failed command
/// and the condition, ending `; retry` (b.xd1). It is never `Aborted`, so no
/// in-process retry loop retries it — the conditions can persist (a full
/// store under `noeviction`, a broken disk), and a loop would spin against
/// them — and it carries no retry hint and no nothing-applied claim, and
/// ticks no counter: a mapper site is not audited for either. The client
/// retries `UNAVAILABLE` itself.
///
/// **Counted, at the write sites below, before any mapper**: all reached
/// from the upload RPCs and each provably nothing-applied: the external-value
/// pre-store (`external_values::ValkeyExternalValueStore::store_values`), the
/// commit script's untagged errors (`LayerStore::map_update_script_error`),
/// the WATCH path's discarded transaction
/// ([`crate::storage::watched_txn::WatchedExec::Discarded`]), and
/// `CreateLayer`'s object-id allocation (with the counter repair writes an
/// absent counter triggers, `CounterStore::initialize_object_id_counter_if_absent`
/// and `CounterStore::seed_object_id_if_below`), materialization `EXEC`,
/// overwrite `superseded` mark and new-layer commit. None is retried
/// in-process. The upload's pre-store and the `UpdateLayer` commit answer
/// [`StoreWriteRefusal::into_service_error`], with the retry hint; every
/// `CreateLayer` site, at `stage="create_commit"`, answers
/// [`StoreWriteRefusal::into_unhinted_service_error`], without it
/// ([`refuse_store_write`]).
///
/// Two of those sites are shared primitives, so the same answer also reaches
/// callers outside the upload RPCs:
/// - `store_values` also serves the DR restore's pre-store and a read's lazy
///   external-value rewrite after checkpoint recovery, both counted at
///   `stage="ext_store"` like the upload pre-store. The restore then maps
///   every pre-store error to its own retryable `Transient`.
/// - The two counter writes also run at startup: the counter-seed routine
///   (`checkpoint::counter_seed`, `initialize_object_id_counter_if_absent` in
///   its absent-sidecar arm and `seed_object_id_if_below`), and the
///   Disabled-adapter `SET NX` in `main.rs`. Those `CounterStore`s carry no
///   metrics handle, so a refusal there answers the same `UNAVAILABLE`, which
///   is logged, but counts nothing (`CounterStore::with_metrics`).
///
/// A new consumer of any of these sites inherits the answer, and belongs in
/// this list.
///
/// [`is_transient_fred_error`] deliberately does not match these four: it
/// would make the update retry loop spin against a full store. Its
/// not-serving carve-out ([`is_store_not_serving_reply`], `-LOADING` /
/// `-BUSY`) covers the conditions that clear by themselves.
///
/// # Anti-creep
///
/// Kind-gated and prefix-anchored, never a substring search and never a bare
/// kind widening — `storage::diff_read::is_backend_unavailable_reply`'s rule,
/// for the same reason: `ErrorKind::Unknown` also carries `-UNBLOCKED`, the
/// router's message-matched failures, this service's own tagged Lua verdicts
/// (`STREAM_ID_COLLISION:`, `WRONGTYPE_KEY:` …) and every other un-parsed
/// server error. Each token carries its trailing space, as every Valkey reply
/// of the four does, so no longer code that merely starts with the same
/// letters matches. A future addition needs the same evidence: a named
/// server origin that refuses before anything is applied.
pub fn store_write_refusal(e: &fred::error::Error) -> Option<StoreWriteRefusal> {
    if !matches!(e.kind(), FredErrorKind::Unknown) {
        return None;
    }
    let details = e.details();
    let reply = details
        .strip_prefix(EXECABORT_DISCARDED_PREFIX)
        .unwrap_or(details);
    if reply.starts_with("OOM ") {
        Some(StoreWriteRefusal::OutOfMemory)
    } else if reply.starts_with("MISCONF ") {
        Some(StoreWriteRefusal::Misconf)
    } else if reply.starts_with("READONLY ") {
        Some(StoreWriteRefusal::ReadOnly)
    } else if reply.starts_with("NOREPLICAS ") {
        Some(StoreWriteRefusal::NoReplicas)
    } else {
        None
    }
}

/// [`store_write_refusal`] at one of its write sites (the upload writes, and
/// the writes sharing their stages: the DR restore's pre-store and a read's
/// recovery rewrite at `ext_store`, the object-id counter repair writes at
/// `create_commit`): when `e` is a store write refusal, tick
/// `live_edit_store_write_refused_total{stage, reason}` on `origin`'s metrics
/// (when it has them), note an `-OOM` on `origin`'s store out-of-memory signal
/// (b.pa1; [`FredErrOrigin`]), and return the `UNAVAILABLE` answer; otherwise
/// `None`, and the caller classifies `e` as before. It counts no command
/// timeout: the caller's mapper does, for the errors it hands on. `stage` is
/// one of [`crate::observability::metrics::STORE_WRITE_REFUSED_STAGES`], and
/// picks the answer: `create_commit` the unhinted one
/// ([`StoreWriteRefusal::into_unhinted_service_error`]), every other stage
/// the hinted one ([`StoreWriteRefusal::into_service_error`]).
pub(crate) fn refuse_store_write(
    origin: FredErrOrigin<'_>,
    stage: &'static str,
    e: &fred::error::Error,
) -> Option<crate::error::ServiceError> {
    let refusal = store_write_refusal(e)?;
    origin.count_store_write_refused(stage, refusal.reason_label());
    if refusal == StoreWriteRefusal::OutOfMemory {
        origin.note_out_of_memory();
    }
    Some(
        if stage == crate::observability::metrics::STORE_WRITE_REFUSED_STAGE_CREATE_COMMIT {
            refusal.into_unhinted_service_error()
        } else {
            refusal.into_service_error()
        },
    )
}

/// Minimum spacing between steady-state auth-rejection hints across the entire
/// process (b.cbe).
///
/// The steady-state surface is unbounded in a way the connect surface is not:
/// [`spawn_error_metric`] is installed on every client of every pool — with the
/// Helm production sizing that is up to ~392 per replica (≈1,176 across the
/// chart's 3 replicas) once every stream reader has connected, by the
/// per-process formula
/// `P = 2 × POOL_SIZE + SUBSCRIBER_POOL_SIZE + SNAPSHOT_POOL_SIZE + 2` (136 at
/// startup: readers connect lazily) — and
/// `reconnect_on_auth_error` makes each one retry forever, so a revoked
/// access-policy assignment produces a continuous cross-product of rejections.
/// One gate shared by every hook (it lives on the single
/// [`ValkeyAuthDiagnostics`] the client holds) bounds the whole process to one
/// multi-line hint per interval.
///
/// Five minutes: long enough that the hint cannot crowd out other logs at that
/// fan-out, short enough that an operator tailing logs during an incident sees
/// it without waiting.
const STEADY_STATE_AUTH_HINT_MIN_INTERVAL: Duration = Duration::from_secs(300);

/// Rate-limit gate for the steady-state auth-rejection hint.
///
/// The mechanism lives in [`crate::storage::log_gate::LogGate`], which carries
/// the rationale for both of its properties (first call always passes so a
/// rejection is never fully silent; a poisoned lock stays quiet rather than
/// panicking) and for the `tokio::time::Instant` clock that makes a 5-minute
/// interval testable under `start_paused`. This alias keeps the name the hint's
/// prose and tests use; the interval is supplied at construction
/// ([`STEADY_STATE_AUTH_HINT_MIN_INTERVAL`]), which is what lets
/// `storage::watched_txn`'s "defence layer disarmed" warnings share the same
/// type with their own spacing.
type AuthHintGate = crate::storage::log_gate::LogGate;

/// Everything needed to turn an opaque server-side `WRONGPASS` / `NOAUTH` into
/// a cause-specific operator hint (b.cbe).
///
/// # Why this type exists
///
/// A managed backing store returns **one** string — `WRONGPASS invalid
/// username-password pair` — for every distinct cause of a rejected AUTH, so the
/// discriminating information is not in the server's response and no amount of
/// parsing it can recover which cause fired. It *is* available here: this
/// service holds its own configuration and the token bytes it sent. This struct
/// is the pairing of those two halves, held across the whole client lifetime so
/// both the connect path and the steady-state error hook can render the same
/// diagnosis.
///
/// # Evaluation is lazy, on purpose
///
/// Construction captures only the two static halves (the mode and the
/// configured username); no diagnosis is computed there. The claims are read out
/// of the provider cache **at the moment a rejection is observed**, which is what makes
/// the strongest statement in the hint — "the token's identity checks out and
/// the server still rejected it, so suspect the access-policy assignment" — a
/// *derived* statement about the credential actually in play rather than a
/// snapshot from before the connect attempt. It also means the token can rotate
/// underneath and the next hint describes the new one.
///
/// # Secrecy
///
/// Reads only [`ServiceTokenProvider::cached_token_claims`], which projects a
/// fixed named set of non-secret claims and can never yield the token bytes
/// (SR-6.2). The configured username is never echoed either — every statement
/// about it is made relationally; see `IdentityVerdict::guidance`.
pub(crate) struct ValkeyAuthDiagnostics {
    /// Which of the three Valkey credential surfaces is in play. Selects the
    /// hint arm: only `accessTokenFile` has a token whose claims can be read.
    auth_mode: ValkeyAuthMode,
    /// `LIVE_EDIT_VALKEY_TOKEN_USERNAME`. `None` outside `accessTokenFile`
    /// (config validation makes it `Some` under that mode).
    configured_username: Option<String>,
    /// Claims-read handle. `None` outside `accessTokenFile` — the `iam` mode's
    /// SigV4 credential is not a JWT and `static` has no token at all.
    token_provider: Option<Arc<ServiceTokenProvider>>,
    /// Shared across every [`spawn_error_metric`] hook on every pool. See
    /// [`STEADY_STATE_AUTH_HINT_MIN_INTERVAL`].
    steady_state_gate: AuthHintGate,
}

impl std::fmt::Debug for ValkeyAuthDiagnostics {
    /// Elides `configured_username` and reports the token handle as a boolean.
    /// `ValkeyClient`'s own `Debug` reaches this type, and neither value has any
    /// business appearing in a `{:?}` dump.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValkeyAuthDiagnostics")
            .field("auth_mode", &self.auth_mode)
            .field("configured_username", &"<redacted>")
            .field("token_provider", &self.token_provider.is_some())
            .finish()
    }
}

/// Rendering of a token's non-secret claims, one preformatted string per field,
/// ready to attach as structured tracing fields.
///
/// Every field is a `String` and never empty: an absent claim renders as
/// [`ABSENT_CLAIM_DISPLAY`] and a mode with no readable token renders as
/// [`NOT_APPLICABLE_DISPLAY`], because an empty value in a log line reads as
/// "this is blank" rather than "there is nothing to read here".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClaimsDescriptor {
    pub(crate) aud: String,
    pub(crate) oid: String,
    pub(crate) client_id_claim: String,
    pub(crate) tid: String,
    pub(crate) iss: String,
    pub(crate) exp: String,
}

/// Rendering for a claim the token does not carry.
const ABSENT_CLAIM_DISPLAY: &str = "<absent>";
/// Rendering for a claim field under a mode with no readable token
/// (`static`, `iam`), or when nothing is cached yet.
const NOT_APPLICABLE_DISPLAY: &str = "<n/a>";
/// `identity_verdict` field value under a mode where no identity self-check is
/// defined. Distinct from `IdentityVerdict::NotAJwt`'s `"not-a-jwt"`, which
/// means "we looked and the token was not decodable".
const VERDICT_NOT_APPLICABLE: &str = "n/a";

impl ClaimsDescriptor {
    /// The all-`<n/a>` descriptor used by the `static` / `iam` arms.
    fn not_applicable() -> Self {
        let na = || NOT_APPLICABLE_DISPLAY.to_string();
        Self {
            aud: na(),
            oid: na(),
            client_id_claim: na(),
            tid: na(),
            iss: na(),
            exp: na(),
        }
    }

    fn from_claims(claims: Option<&TokenClaims>) -> Self {
        let Some(claims) = claims else {
            return Self::not_applicable();
        };
        let field = |v: &Option<String>| {
            v.clone()
                .unwrap_or_else(|| ABSENT_CLAIM_DISPLAY.to_string())
        };
        Self {
            aud: if claims.aud.is_empty() {
                ABSENT_CLAIM_DISPLAY.to_string()
            } else {
                claims.aud.join(", ")
            },
            oid: field(&claims.oid),
            client_id_claim: field(&claims.client_id_claim),
            tid: field(&claims.tid),
            iss: field(&claims.iss),
            exp: match claims.exp {
                // Both renderings, because they answer different questions: the
                // raw NumericDate is what an operator greps the sidecar's own
                // logs for, the RFC 3339 form is what tells them at a glance
                // whether the token is simply stale.
                Some(secs) => match chrono::DateTime::from_timestamp(secs, 0) {
                    Some(dt) => format!("{secs} ({})", dt.to_rfc3339()),
                    None => secs.to_string(),
                },
                None => ABSENT_CLAIM_DISPLAY.to_string(),
            },
        }
    }
}

/// A fully rendered operator hint: the structured fields plus the prose body.
pub(crate) struct ValkeyAuthHint {
    pub(crate) verdict_label: &'static str,
    pub(crate) claims: ClaimsDescriptor,
    pub(crate) message: String,
}

impl ValkeyAuthDiagnostics {
    /// Pair the configured Valkey credential surface with the handle that can
    /// read the token's claims.
    ///
    /// `token_provider` is the same `Arc` the credential seam installs; this
    /// takes a **clone of the handle**, never builds one, so it cannot become a
    /// second construction site for the shared credential source.
    pub(crate) fn new(config: &Config, token_provider: Option<Arc<ServiceTokenProvider>>) -> Self {
        Self {
            auth_mode: config.valkey_auth_mode,
            configured_username: config.valkey_token_username.clone(),
            token_provider,
            steady_state_gate: AuthHintGate::new(STEADY_STATE_AUTH_HINT_MIN_INTERVAL),
        }
    }

    fn auth_mode_label(&self) -> &'static str {
        match self.auth_mode {
            ValkeyAuthMode::Static => "static",
            ValkeyAuthMode::AccessTokenFile => "accessTokenFile",
            ValkeyAuthMode::Iam => "iam",
        }
    }

    /// Read the live cached claims. `None` under any mode without a readable
    /// token, and under `accessTokenFile` when nothing is cached yet or the
    /// cached token is not a decodable JWT.
    fn cached_claims(&self) -> Option<TokenClaims> {
        self.token_provider
            .as_ref()
            .and_then(|p| p.cached_token_claims())
    }

    /// Read the cached claims **once** and derive the verdict from that same
    /// read.
    ///
    /// Both consumers need the verdict *and* the claims themselves, and the
    /// cache can rotate between two reads — so taking one read and returning
    /// both is what stops a hint from printing one token's claims underneath
    /// another token's verdict.
    ///
    /// `configured_username` is `None` only outside `accessTokenFile`, where
    /// config validation does not require it; the empty-string fallback then
    /// matches no claim and yields `MatchesNeither`, which prints the token's
    /// real `oid` and tells the operator to set the variable to it — the right
    /// advice for an unset username, and unreachable in practice because this
    /// path only runs under `accessTokenFile`, where validation forces it
    /// `Some`.
    fn evaluate(&self) -> (IdentityVerdict, Option<TokenClaims>) {
        let claims = self.cached_claims();
        let verdict = valkey_identity_verdict(
            claims.as_ref(),
            self.configured_username.as_deref().unwrap_or(""),
        );
        (verdict, claims)
    }

    /// Render the hint for the mode in play, evaluating the token claims *now*.
    pub(crate) fn render_hint(&self) -> ValkeyAuthHint {
        match self.auth_mode {
            ValkeyAuthMode::AccessTokenFile => self.render_access_token_file_hint(),
            ValkeyAuthMode::Iam => ValkeyAuthHint {
                verdict_label: VERDICT_NOT_APPLICABLE,
                claims: ClaimsDescriptor::not_applicable(),
                message: IAM_AUTH_REJECTION_HINT.to_string(),
            },
            ValkeyAuthMode::Static => ValkeyAuthHint {
                verdict_label: VERDICT_NOT_APPLICABLE,
                claims: ClaimsDescriptor::not_applicable(),
                message: STATIC_AUTH_REJECTION_HINT.to_string(),
            },
        }
    }

    /// The `accessTokenFile` arm: the identity self-check, then the four-cause
    /// checklist with each cause annotated by what the self-check proved.
    fn render_access_token_file_hint(&self) -> ValkeyAuthHint {
        let (verdict, claims) = self.evaluate();
        let descriptor = ClaimsDescriptor::from_claims(claims.as_ref());

        // Per-cause status keyed by the verdict. Causes (1) and (3) are the two
        // the self-check can actually decide; (2) and (4) are reported and
        // narrowed, never decided — there is no audience allowlist here (the
        // mode is issuer-agnostic, SDD §15.3/§15.4) and the access-policy
        // assignment lives entirely on the backing store's side.
        let (cause_1, cause_3, cause_4) = match &verdict {
            IdentityVerdict::MatchesOid => (
                "RULED OUT — the configured username is the token's `oid`, not its client id.",
                "RULED OUT — the token's `oid` equals the configured username.",
                "MOST LIKELY REMAINING CAUSE — causes (1) and (3) are ruled out from this \
                 service's own data, so a token that is correctly minted and correctly bound is \
                 being refused by the data plane.",
            ),
            IdentityVerdict::MatchesClientIdClaim { .. } => (
                "THIS IS THE CAUSE — see the self-check above; fix this first.",
                "Follows from (1): the configured username is the client id, so it is necessarily \
                 not the `oid`.",
                "Not yet in question — resolve (1) first, then retry.",
            ),
            IdentityVerdict::MatchesNeither { .. } => (
                "RULED OUT — the configured username is not the token's client id either.",
                "THIS IS THE LIKELY CAUSE — the configured username matches neither identity \
                 claim, so the sidecar minted a token for a different identity than the one \
                 configured. A missing workload-identity binding on the pod is the usual reason.",
                "Not yet in question — resolve (3) first, then retry.",
            ),
            IdentityVerdict::NoIdentityClaim | IdentityVerdict::NotAJwt => (
                "NOT CHECKABLE — no identity claim was readable; verify by hand.",
                "NOT CHECKABLE — no identity claim was readable; verify by hand.",
                "NOT NARROWED — with no readable identity claim this service cannot rule the \
                 other causes out; check all four.",
            ),
        };

        // Cause (4) is the one an operator has to go look up in the cloud
        // console, so name the subject concretely whenever the token tells us
        // what it is — an Object ID they can paste into a search box beats
        // "the identity", which is the phrasing that made this a 40-minute
        // chase in the first place.
        let policy_subject = match claims.as_ref().and_then(|c| c.oid.as_deref()) {
            Some(oid) => format!("object ID `{oid}`"),
            None => "the identity the credential sidecar authenticates as".to_string(),
        };

        let message = format!(
            "Valkey rejected the credential this service sent (WRONGPASS/NOAUTH) under \
             LIVE_EDIT_VALKEY_AUTH_MODE=accessTokenFile. A managed backing store returns the \
             same opaque string for every cause, so the following is derived from this \
             service's own configuration and from the non-secret claims of the token it sent \
             (the token bytes are never logged).\n\
             \n\
             Token-identity self-check: {guidance}\n\
             \n\
             Four known causes:\n\
             (1) LIVE_EDIT_VALKEY_TOKEN_USERNAME set to the identity's client id instead of its \
             directory Object ID. {cause_1}\n\
             (2) Wrong token audience/scope. The token's `aud` is {aud}. The credential sidecar \
             must request a token scoped to the backing store's own data plane — for Azure \
             Managed Redis that is scope `https://redis.azure.com/.default`; other issuers use \
             their own value, so compare against the scope your store's provider documents \
             (docs/service/docs/backing-store.md links each provider's setup page) rather than \
             against a fixed string. This service deliberately has no expected-`aud` \
             allowlist: the mode is issuer-agnostic, so it reports the audience and does not \
             judge it.\n\
             (3) Token `oid` does not match LIVE_EDIT_VALKEY_TOKEN_USERNAME (federation \
             misbinding — most often the pod is missing its workload-identity binding, so the \
             sidecar falls back to a different identity). {cause_3}\n\
             (4) No data-plane access-policy assignment for this identity on the cache, or one \
             whose provisioning state is not yet `Succeeded`. {cause_4} Verify an access-policy \
             assignment exists for {policy_subject} on this cache and reports `Succeeded`; a \
             flawless token is still refused without it.\n\
             \n\
             Token claims used above (all non-secret): aud={aud} oid={oid} \
             client_id_claim={client_id} tid={tid} iss={iss} exp={exp}",
            guidance = verdict.guidance(),
            aud = descriptor.aud,
            oid = descriptor.oid,
            client_id = descriptor.client_id_claim,
            tid = descriptor.tid,
            iss = descriptor.iss,
            exp = descriptor.exp,
        );

        ValkeyAuthHint {
            verdict_label: verdict.label(),
            claims: descriptor,
            message,
        }
    }

    /// Emit the hint with the structured claim fields attached, at a level
    /// selected by `phase`.
    ///
    /// `phase` distinguishes the two surfaces (`"connect"` /
    /// `"steady-state"`); `pool` names the connection that observed the
    /// rejection.
    ///
    /// # Why connect is WARN and steady-state is ERROR
    ///
    /// The connect-path hint fires on the **first** failing attempt while
    /// [`connect_with_retry`] keeps retrying. A boot that recovers on a later
    /// attempt — a credential sidecar that had not written the token file yet, an
    /// access-policy assignment still finishing provisioning — would otherwise
    /// end with an ERROR hint, the WARN retry line, and a contradicting INFO
    /// healthy-connect line all describing one condition, and ERROR-level
    /// alerting would page on a pod that came up fine. At WARN the hint is still
    /// in the log an operator reads after a boot that genuinely fails, without
    /// asserting failure before the retry loop has finished.
    ///
    /// The steady-state surface has no such recovery narrative: the pool was
    /// already up and authenticated, and its call site only fires on the
    /// persistent (`WRONGPASS`) rejection shape, so it keeps ERROR.
    fn emit_hint(&self, phase: &'static str, pool: &str) {
        let hint = self.render_hint();
        // tracing's level is compile-time metadata, so the two levels are two
        // macro call sites; the body is shared through the local macro.
        macro_rules! emit_at {
            ($level:ident) => {
                tracing::$level!(
                    valkey_auth_mode = self.auth_mode_label(),
                    phase,
                    pool,
                    identity_verdict = hint.verdict_label,
                    token_aud = %hint.claims.aud,
                    token_oid = %hint.claims.oid,
                    token_client_id_claim = %hint.claims.client_id_claim,
                    token_tid = %hint.claims.tid,
                    token_iss = %hint.claims.iss,
                    token_exp = %hint.claims.exp,
                    "{}",
                    hint.message
                )
            };
        }
        if phase == "connect" {
            emit_at!(warn);
        } else {
            emit_at!(error);
        }
    }

    /// Connect-path emission, rate-limited to once per `connect()` invocation by
    /// the caller-owned `already_emitted` flag.
    ///
    /// The flag is passed in rather than held on `self` deliberately: a
    /// process-global `Once` would make the hint untestable, because `connect`
    /// also runs from `storage::test_support` and from three integration
    /// binaries, all of which share a process under `cargo test`.
    fn emit_connect_hint(&self, pool: &str, already_emitted: &AtomicBool) {
        if already_emitted.swap(true, Ordering::Relaxed) {
            return;
        }
        self.emit_hint("connect", pool);
    }

    /// Steady-state emission from the connection-error hook, bounded by the
    /// shared [`AuthHintGate`].
    fn emit_steady_state_hint(&self, pool: &str) {
        if !self.steady_state_gate.allow() {
            return;
        }
        self.emit_hint("steady-state", pool);
    }

    /// One INFO on a successful connect under `accessTokenFile`, recording the
    /// identity verdict for the credential that was just **accepted** (b.cbe).
    ///
    /// Sibling of the `"Valkey transport configured"` TLS line: it states what
    /// the service ended up doing, so a later failure has a healthy baseline to
    /// diff against. It is also the only place the healthy verdict is ever
    /// printed — the hint path only runs on rejection — which matters because
    /// `matches-oid` on a working deployment is the evidence an operator needs
    /// to trust the same field on a broken one.
    ///
    /// Silent under `static` / `iam`: neither has a token whose claims could be
    /// reported, so a line there would carry nothing but the mode.
    fn log_healthy_connect(&self) {
        if self.auth_mode != ValkeyAuthMode::AccessTokenFile {
            return;
        }
        let (verdict, claims) = self.evaluate();
        let descriptor = ClaimsDescriptor::from_claims(claims.as_ref());
        tracing::info!(
            valkey_auth_mode = "accessTokenFile",
            identity_verdict = verdict.label(),
            token_aud = %descriptor.aud,
            token_oid = %descriptor.oid,
            token_client_id_claim = %descriptor.client_id_claim,
            token_tid = %descriptor.tid,
            token_iss = %descriptor.iss,
            token_exp = %descriptor.exp,
            "Valkey accepted the accessTokenFile credential; token-identity self-check recorded \
             (non-secret claims only)"
        );
    }
}

/// The `iam` arm of the auth-rejection hint.
///
/// Deliberately short and deliberately **not** a four-cause checklist: the
/// causes enumerated for `accessTokenFile` are properties of an OAuth-style
/// bearer token and a data-plane access-policy assignment, and none of them
/// transfers to a SigV4-signed credential. Inventing an IAM cause list from the
/// shape of the Azure one would be a guess printed with the authority of a
/// diagnosis. A real one is deferred to its own ticket (b.r5d); until then this
/// arm names the mode and the exact env vars that feed the signature, which is
/// true and useful.
const IAM_AUTH_REJECTION_HINT: &str = "Valkey rejected the credential this service sent \
     (WRONGPASS/NOAUTH) under LIVE_EDIT_VALKEY_AUTH_MODE=iam. There is no claims-level \
     self-check for this mode — the credential is a SigV4-signed string, not a JWT, so it \
     carries no readable identity — which makes the check manual. Confirm that: \
     LIVE_EDIT_VALKEY_IAM_USERNAME names an IAM-enabled ACL user that exists on this cluster; \
     LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST is the cluster/cache NAME used for signing, not the \
     endpoint FQDN (the FQDN belongs in LIVE_EDIT_VALKEY_URL); LIVE_EDIT_VALKEY_IAM_REGION is \
     the cluster's region; LIVE_EDIT_VALKEY_IAM_SERVICE (memorydb | elasticache) matches the \
     service family this cluster belongs to, since it selects the SigV4 service name, the IAM \
     action and the signing-host family; and the pod's AWS identity is granted the connect \
     action for that user on that cluster.";

/// The `static` arm of the auth-rejection hint.
///
/// The credential rides `LIVE_EDIT_VALKEY_URL`, so there is nothing for this
/// service to introspect and — importantly — nothing it may echo: that URL
/// carries the password inline, which is why every other log line about it goes
/// through `redact_url_credentials`.
const STATIC_AUTH_REJECTION_HINT: &str = "Valkey rejected the credential this service sent \
     (WRONGPASS/NOAUTH) under LIVE_EDIT_VALKEY_AUTH_MODE=static, where the credential is carried \
     inline on LIVE_EDIT_VALKEY_URL. Check the username and password embedded in that URL \
     against the backing store's ACL — note the URL is never logged and must not be pasted into \
     a ticket, because the password is part of it. If this cluster expects a passwordless \
     identity instead, set LIVE_EDIT_VALKEY_AUTH_MODE to accessTokenFile or iam and supply that \
     mode's variables.";

/// The slow unresponsive bound (fred's `UnresponsiveConfig::max_timeout`): a
/// connection whose oldest outstanding command has had no reply progress for
/// this long is closed and rebuilt whatever its peers are doing — the bound for
/// a shard that is silent to every connection. It is also the bound on every
/// wait in fred's router on one connection's I/O outside its read loop (a
/// drain, a broadcast reply, an `ASKING` reply, a transaction's `EXEC`, a write
/// or flush). Must exceed the longest legitimate single-command wait — a stream
/// reader's `XREAD BLOCK` budget, `XREAD_BLOCK_MS + 5 s` = 35 s — with margin.
/// See [`apply_connect_timeout`].
pub(crate) const UNRESPONSIVE_MAX_TIMEOUT: Duration = Duration::from_secs(45);

/// The client-time allowance, in µs, a whole-write transaction's timeout adds
/// per command it queues ([`TXN_TIMEOUT_PER_COMMAND`]). Measured, not tuned:
/// 2 × the slowest client wait per command measured on an unoptimised build,
/// rounded up to 10 µs (measurement M1′, the record
/// `docs-internal/validation/txn-timeout-m1prime-20261013/`, whose
/// `derivation.json` this value is pinned against).
pub(crate) const TXN_TIMEOUT_PER_COMMAND_US: u64 = 120;

/// [`TXN_TIMEOUT_PER_COMMAND_US`] as a `Duration`.
///
/// fred puts ONE timer around a whole transaction: from the router taking it,
/// through writing every queued command, the store running the `EXEC`, to the
/// reply. Almost all of that time is the client's, not the store's: at the
/// per-delta limit (about 300,000 commands) the store's `EXEC` took about
/// 0.3–0.4 s of a 6–7 s wait in an unoptimised build. So the flat per-command
/// timeout (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, 5 s by default) cannot
/// bound a transaction whose size is set by its content, and
/// [`txn_timeout`] scales it.
pub(crate) const TXN_TIMEOUT_PER_COMMAND: Duration =
    Duration::from_micros(TXN_TIMEOUT_PER_COMMAND_US);

/// The client timeout of one whole-write transaction that queues `commands`
/// commands: `command_timeout + commands × TXN_TIMEOUT_PER_COMMAND`, capped at
/// the larger of `command_timeout` and [`UNRESPONSIVE_MAX_TIMEOUT`].
///
/// - A `command_timeout` of zero means "no timeout" (fred's convention for
///   `default_command_timeout`) and stays zero.
/// - Saturating, so no count can wrap it.
/// - While `command_timeout` is at most [`UNRESPONSIVE_MAX_TIMEOUT`] (the
///   default 5 s is), the cap keeps the caller answered before fred can start
///   its one whole-transaction replay: every router wait on a `Transaction`'s
///   connection is bounded by [`UNRESPONSIVE_MAX_TIMEOUT`] and starts at or
///   after the send, so a timeout no longer than that bound expires first
///   (SDD §15.3), and fred writes nothing of an attempt that starts after its
///   caller gave up (the vendored fred's change 9), so that replay is never
///   written. A `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` above 45 s is its own
///   cap, and then a replay can start, and be written, before the caller is
///   answered, as it already can for every other `Transaction` user at that
///   setting. A
///   transaction that needs more client time than the cap is answered as any
///   timed-out transaction is: `write_object` answers `UNAVAILABLE`, and the
///   WATCH commit reconciles its in-doubt `EXEC`. The cap binds on large WATCH
///   commits: their count is bounded by the layer, not by one delta (a reset
///   clearing 11,000 placed assets queues about 750,000 commands, whose client
///   wait measured 5–6.5 s in an optimised build and 27–30 s in an unoptimised
///   one).
///
/// Used by exactly two transactions, each passing its own count: `write_object`
/// (`CreateLayer`'s object write) and the WATCH-route `UpdateLayer` commit
/// (WA3's queued prediction). fred reads a transaction's timeout from its
/// first command, which each sets it on.
pub(crate) fn txn_timeout(command_timeout: Duration, commands: u64) -> Duration {
    if command_timeout.is_zero() {
        return Duration::ZERO;
    }
    let per_command = u32::try_from(commands)
        .map(|n| TXN_TIMEOUT_PER_COMMAND.saturating_mul(n))
        .unwrap_or(Duration::MAX);
    command_timeout
        .saturating_add(per_command)
        .min(command_timeout.max(UNRESPONSIVE_MAX_TIMEOUT))
}

/// The fast unresponsive bound: a connection whose oldest outstanding command
/// has had no reply progress for this long is closed and rebuilt if the same
/// cluster slot range answered another connection at least
/// [`UNRESPONSIVE_PEER_EVIDENCE_LAG`] after the stall began. Off on the stream
/// readers. With one [`UNRESPONSIVE_CHECK_INTERVAL`] and a reconnect it must
/// leave room inside the default command timeout for fred's re-send of the
/// in-flight commands. See [`apply_connect_timeout`].
pub(crate) const UNRESPONSIVE_FAST_TIMEOUT: Duration = Duration::from_secs(2);

/// How long after a connection's stall began a peer's reply from the same slot
/// range must arrive to count as evidence that the shard is answering, so a
/// reply that left the shard just before it went busy does not count.
pub(crate) const UNRESPONSIVE_PEER_EVIDENCE_LAG: Duration = Duration::from_secs(1);

/// How often fred re-polls every connection of an idle client, which runs the
/// unresponsive check (fred's `UnresponsiveConfig::interval`); a busy client
/// polls on every event anyway. At most a quarter of
/// [`UNRESPONSIVE_FAST_TIMEOUT`], so a fast verdict lands within one interval
/// of the bound.
pub(crate) const UNRESPONSIVE_CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// Test-scale replacements for the unresponsive bounds
/// [`apply_unresponsive_policy`] sets, for a test that cannot wait out the
/// production ones. Every `Builder` [`ValkeyClient::connect`] makes while
/// [`with_liveness_overrides_for_test`] runs takes them, and so does every
/// client later cloned from one (a member replacement's fresh client, a stream
/// reader's connection). A `None` field keeps the production value; the fast
/// bound stays off where the policy turns it off (the readers).
/// **Test-only** (`test-support`).
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LivenessOverrides {
    /// Replaces [`UNRESPONSIVE_FAST_TIMEOUT`].
    pub fast_timeout: Option<Duration>,
    /// Replaces [`UNRESPONSIVE_MAX_TIMEOUT`], the slow bound and the bound on
    /// every router wait on one connection's I/O.
    pub slow_timeout: Option<Duration>,
    /// Replaces [`UNRESPONSIVE_CHECK_INTERVAL`].
    pub check_interval: Option<Duration>,
    /// Replaces [`UNRESPONSIVE_PEER_EVIDENCE_LAG`].
    pub peer_evidence_lag: Option<Duration>,
}

#[cfg(any(test, feature = "test-support"))]
tokio::task_local! {
    static LIVENESS_OVERRIDES: LivenessOverrides;
}

/// Run `connect` — a future that calls [`ValkeyClient::connect`] — with
/// `overrides` applied to every unresponsive policy it sets
/// ([`LivenessOverrides`]). Scoped to the future's task, so concurrent tests
/// do not see each other's overrides. **Test-only** (`test-support`).
#[cfg(any(test, feature = "test-support"))]
pub async fn with_liveness_overrides_for_test<F: std::future::Future>(
    overrides: LivenessOverrides,
    connect: F,
) -> F::Output {
    LIVENESS_OVERRIDES.scope(overrides, connect).await
}

/// Which unresponsive verdicts the connections of a fred `Builder` may reach
/// (see [`apply_connect_timeout`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnresponsiveVerdicts {
    /// The fast, peer-gated verdict and the slow bound: every pool and the
    /// pub/sub client.
    FastAndSlow,
    /// The slow bound only: the stream readers, whose own connection checks
    /// (`storage::stream_mux`) cover them sooner.
    SlowOnly,
}

/// How long [`ValkeyClient::quit`] gives the `WatchGuard` UNWATCH
/// `TaskTracker` to drain before it closes the `txn` pool anyway (b.k1r
/// H-stor-2).
///
/// Module-visible rather than function-local (b.ji8 round 3) because
/// `storage::watch_guard` sizes its own bounds against it: a `const` assertion
/// there requires `UNWATCH_FUNNEL_TIMEOUT + UNWATCH_DRAIN_SCHEDULING_RESERVE <=
/// this`, so raising a teardown bound past the budget shutdown allows is a
/// compile error rather than a shutdown-shaped race nobody observes. It is
/// also one of `quit()`'s sequential bounds, which a second `const` assertion
/// (after `STREAM_MUX_SHUTDOWN_GRACE`) sums against the shutdown-budget headroom
/// (SDD §15.3, §20).
pub(crate) const UNWATCH_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Apply the per-attempt TCP/TLS `connection_timeout` derived from
/// `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` — and enable
/// `reconnect_on_auth_error` — on a fred `Builder`'s `ConnectionConfig`.
///
/// fred 10.x's `ConnectionConfig::connection_timeout` defaults to a
/// hardcoded 10 s. Without this wiring, any operator who set the env var
/// below 10 s would have been silently overridden — a real footgun fixed in
/// b.1kf. Centralising the mutation in one helper keeps every `Builder` —
/// the data pool's, the pub/sub subscriber's and the stream readers'
/// template (`ValkeyClient::reader_builder`) — in sync and gives unit tests a
/// regression-guard seam that fails closed if a future refactor drops the
/// call from [`ValkeyClient::connect`].
///
/// The helper also sets `reconnect_on_auth_error = true` (a `ConnectionConfig`
/// field defaulting `false`, not a cargo feature). SR-4.2/SR-5.1 require fred
/// to re-establish a connection after an auth error so the credential-provider
/// refresh loop can re-AUTH on live connections; without it a rotated /
/// refreshed credential could never take effect on an already-open connection.
/// Because every call site routes through this single seam, every
/// connection inherits the behaviour.
///
/// Finally the helper applies the unresponsive-connection policy
/// ([`apply_unresponsive_policy`]). fred's per-command timeout — whether the
/// global `default_command_timeout` or a per-call `Options::timeout` — is a
/// bare `tokio::time::timeout` around the response future: it fails the
/// *caller*, but never closes or reconnects the socket. On a managed frontend
/// (Azure Managed Redis) a connection can go silent without any FIN/RST
/// reaching fred's reader, so nothing else in the stack notices, and fred's
/// round-robin keeps handing it a share of the pool's commands. The
/// unresponsive check closes such a connection and lets the reconnect policy
/// rebuild it.
///
/// The check is the vendored fred's (`vendor/fred-10.1.0/src/live_edit/`,
/// recorded in `vendor/fred-10.1.0.SOURCE.md`). A connection's *stall* is the
/// time since the later of two events: its oldest outstanding non-pubsub
/// command reaching the socket, and its last read progress (a reply decoded, or
/// reply bytes arriving). Connections that are `blocked` (a blocking command in
/// flight, such as a reader's `XREAD BLOCK`) are not judged. Two verdicts close
/// a connection, each with its own error text, which
/// `fred::live_edit::liveness::verdict_of` tells apart (never compare the
/// texts):
///
/// - **fast** ([`UnresponsiveVerdicts::FastAndSlow`] only): the stall exceeds
///   [`UNRESPONSIVE_FAST_TIMEOUT`] and another connection — of any client built
///   with the same `peer_replies` registry, which [`ValkeyClient::connect`]
///   shares across every pool and the readers — received a reply from the same
///   cluster slot range at least [`UNRESPONSIVE_PEER_EVIDENCE_LAG`] after the
///   stall began. The shard answers others, so this socket is the dead one. A
///   command with a hash slot — from a key, a `ClusterHash::Custom` pin (the
///   `WatchedTxn` `MULTI`/`CLIENT ID`/`EXEC`, the presence `TIME`, the
///   checkout's `DISCARD`/`UNWATCH`) or an argument fred hashes — is judged by
///   its slot's range when the client's cluster routing places that slot on
///   the connection's node, because that slot's shard runs it; any other (one
///   with nothing to hash, one whose slot another node owns, any before the
///   client has cluster routing) is judged by every reply from the
///   connection's node. A standalone store is one node owning every slot.
///   Error text `"Unresponsive connection (fast verdict)."`.
/// - **slow**: the stall exceeds [`UNRESPONSIVE_MAX_TIMEOUT`], whatever peers
///   are doing — a shard silent to everyone. Error text stock fred's
///   `"Unresponsive connection."`.
///
/// The peer evidence is what keeps the fast bound safe. A shard busy with a
/// long script or frozen for seconds answers no connection, so nothing is
/// closed and its connections are not rebuilt at once (the connect/AUTH storm
/// b.azh guards against); a reply that keeps arriving counts as progress, so a
/// large reply is never cut; and a command's wait starts when its frame reaches
/// the socket, so a large upload's own write time does not count — nor does a
/// command fed behind one, which is measured from the connection's last
/// completed write while it waits in fred's buffer. A
/// `WatchedTxn` pipeline is judged like any pipeline: each queued command is
/// answered `QUEUED` as it arrives, and its `EXEC` runs on the shard, which
/// answers no one meanwhile. A fred `Transaction` (`client.multi()`) is read
/// outside the read loop, under the bounded waits below.
///
/// What a verdict does is fred's existing path for a lost connection: the
/// connection is dropped and rebuilt, and every in-flight command with write
/// attempts left is re-sent on the replacement — about 2.5–3 s after the stall
/// began with a fast verdict, inside the 5 s command timeout. A command with
/// one attempt (the `UpdateLayer` commit `EVALSHA`, the operation-id start
/// record `HGET`, and on the `txn` pool every single command and every
/// `WatchedTxn` pipeline) fails with that IO error instead of timing out; its
/// callers already handle it. (The GC decrement `EVALSHA` is not one of them:
/// it keeps fred's attempts, and its re-send answers `replayed`.) A fred
/// `Transaction` on the `txn` pool is the exception: no verdict judges it
/// (it is read outside the read loop), and when one of its writes or bounded
/// reads below fails, fred reconnects and replays the whole transaction once,
/// even at the pool's one attempt (`router/transactions.rs` gives up only
/// once its failures exceed the attempt budget, so at one attempt the second
/// failure ends it). What that replay does to each `Transaction` user is
/// stated at the `txn` pool's config in [`ValkeyClient::connect`] (*ONE
/// EXCEPTION*) and in SDD §15.3: not every user is safe under it — the
/// session checkpoint applier's Step B mapping `SET`s can overwrite a
/// `DeleteLayer` or `CreateLayer` writer that landed in the reconnect window.
/// A `CreateLayer` materialization whose landed `EXEC` reply was lost, and
/// whose replay then reaches the store, answers `UNAVAILABLE` and leaves an
/// unmapped orphan object for the orphan sweep; a re-send creates the layer.
/// The materialization's caller waits its size-scaled timeout ([`txn_timeout`]),
/// which is capped at the slow bound below while the command timeout is at
/// most that bound (the default is), so it is answered no later than a replay
/// can start; a command timeout set above the slow bound is its own cap, and
/// then a replay can start first, as for every other `Transaction` user.
///
/// The same slow bound caps every wait in fred's router on one connection's
/// I/O outside its read loop: a drain (which the cluster credential refresh's
/// `_AuthAllCluster` broadcast runs first), the broadcast's replies, an
/// `ASKING` reply, a transaction's `EXEC` and skip reads, and every write and
/// flush. When it expires that connection fails with the slow verdict's error;
/// the router keeps running. Every later write, flush or read on that
/// connection then fails at once with the same error, without waiting, until
/// fred closes it, which it does without waiting on it either (no flush of
/// what fred still holds and no TLS `close_notify`, as for any connection
/// fred has judged dead, by an expired wait or a verdict) — so a fred
/// `Transaction`'s `DISCARD` on the connection whose `EXEC` read expired, a
/// re-flush or a drain never waits a second time, and one silent connection
/// holds the router for at most one slow bound per wait chain (per attempt,
/// for a `Transaction`), its close included. Because the check judges the
/// in-flight queue rather than a write timestamp, a reply read by such a drain
/// clears the stall like any other, so a refresh no longer leaves a healthy
/// connection to be declared unresponsive later.
///
/// Every verdict and expired wait logs one fred WARN recording where the stuck
/// bytes sit — fred's buffers, the command queue, the client's other
/// connections, the kernel's socket queues (field list in
/// `vendor/fred-10.1.0/src/live_edit/liveness.rs`) — a stall of over a second
/// that had peer evidence and then recovered logs a rate-limited fred INFO
/// "near miss", and each verdict ticks
/// `live_edit_valkey_unresponsive_total{pool,verdict}` (see
/// `spawn_error_metric`).
///
/// A member of the `data`, `maintenance` or `snapshot` pool, or a `txn` slot,
/// whose client reaches its second fast verdict within an hour is replaced by
/// a fresh client (`storage::member_replacement`: the trigger, cooldown,
/// retirement grace and replacement budget are its constants).
///
/// The stream readers get [`UnresponsiveVerdicts::SlowOnly`]: their own
/// connection checks in `storage::stream_mux` — the idle keepalive, the
/// idle→active probe, the liveness wake with the wake-ack watchdog, and the
/// XREAD's own timeout — catch a dead reader connection sooner, each ending in
/// a forced reconnect: a `force_reconnection` self-heal, or, on a cluster,
/// retiring the connection instead when its node no longer serves the reader
/// (a verdict that finds the bound slot's owner moved retires it and
/// retargets; an idle reader whose node is no longer a primary just retires
/// it). Their replies still stamp the shared registry.
pub(crate) fn apply_connect_timeout(
    builder: &mut Builder,
    config: &Config,
    peer_replies: &PeerReplies,
    verdicts: UnresponsiveVerdicts,
) {
    let connection_timeout = Duration::from_secs(config.valkey_connect_timeout_secs);
    builder.with_connection_config(|cfg| {
        cfg.connection_timeout = connection_timeout;
        cfg.reconnect_on_auth_error = true;
        apply_unresponsive_policy(cfg, peer_replies, verdicts);
    });
}

/// Set the unresponsive-connection policy [`apply_connect_timeout`] describes
/// on a fred `ConnectionConfig`: the slow bound and scan interval on fred's
/// `UnresponsiveConfig`, and the fast bound, evidence lag and shared
/// `peer_replies` registry on the vendored fred's liveness settings. Under
/// [`UnresponsiveVerdicts::SlowOnly`] the fast bound is left off; the registry
/// is set either way, so the connection's replies count as evidence for
/// others.
pub(crate) fn apply_unresponsive_policy(
    cfg: &mut fred::types::config::ConnectionConfig,
    peer_replies: &PeerReplies,
    verdicts: UnresponsiveVerdicts,
) {
    cfg.unresponsive = fred::types::config::UnresponsiveConfig {
        max_timeout: Some(UNRESPONSIVE_MAX_TIMEOUT),
        interval: UNRESPONSIVE_CHECK_INTERVAL,
    };
    cfg.live_edit.liveness = fred::live_edit::liveness::Settings {
        fast_timeout: match verdicts {
            UnresponsiveVerdicts::FastAndSlow => Some(UNRESPONSIVE_FAST_TIMEOUT),
            UnresponsiveVerdicts::SlowOnly => None,
        },
        peer_evidence_lag: UNRESPONSIVE_PEER_EVIDENCE_LAG,
        peer_replies: Some(peer_replies.clone()),
    };
    #[cfg(any(test, feature = "test-support"))]
    let _ = LIVENESS_OVERRIDES.try_with(|overrides| {
        if let Some(slow) = overrides.slow_timeout {
            cfg.unresponsive.max_timeout = Some(slow);
        }
        if let Some(interval) = overrides.check_interval {
            cfg.unresponsive.interval = interval;
        }
        if let (Some(fast), UnresponsiveVerdicts::FastAndSlow) = (overrides.fast_timeout, verdicts)
        {
            cfg.live_edit.liveness.fast_timeout = Some(fast);
        }
        if let Some(lag) = overrides.peer_evidence_lag {
            cfg.live_edit.liveness.peer_evidence_lag = lag;
        }
    });
}

/// The `FredConfig` values every Valkey connection derives from.
///
/// They are produced together by [`build_fred_config`] — the single override
/// seam — so the derivation cannot drift from the shared policy.
///
/// * `shared` — the data, txn, maintenance and snapshot pools.
/// * `pubsub` — the dedicated pub/sub `SubscriberClient` only. Identical to
///   `shared` in every field except that a `credential_provider`, when present
///   and the protocol is RESP2, is wrapped in a
///   [`NoRefreshCredentialProvider`] so fred schedules no periodic re-`AUTH` on
///   that one connection (b.j64; see [`derive_pubsub_config`]).
/// * `reader` — the stream readers' template, before `connect` sets its
///   `blocking`. Identical to `shared` in every field except that a
///   `credential_provider`, when present, is wrapped in a
///   [`NoRefreshCredentialProvider`] under any RESP version, so fred schedules
///   no periodic re-`AUTH` on any reader client: each reader re-`AUTH`s its
///   own connection between reads instead (b.19i; see
///   [`derive_reader_config`]).
pub(crate) struct FredConfigs {
    /// Policy for the data / txn / maintenance / snapshot pools.
    pub(crate) shared: FredConfig,
    /// Policy for the dedicated pub/sub `SubscriberClient`.
    pub(crate) pubsub: FredConfig,
    /// `true` when `pubsub`'s credential provider was wrapped, i.e. the periodic
    /// re-`AUTH` is suppressed on the pub/sub connection. Read by
    /// [`ValkeyClient::connect`] for the one startup log line that makes the
    /// divergence greppable, and by the unit tests that pin the wrapping.
    pub(crate) pubsub_refresh_disabled: bool,
    /// Policy for the stream readers' connections (their template).
    pub(crate) reader: FredConfig,
    /// The *unwrapped* provider `reader`'s wrapper delegates to — the same
    /// `Arc` `shared` carries — or `None` when there is none (`static` mode).
    /// Handed to the readers' `ReaderConnector`, so a reader can schedule its
    /// own re-`AUTH` ticks (`refresh_interval()`) and consult the provider's
    /// gate (`fetch(None)`) as fred's refresh task would. `Some` exactly when
    /// fred's periodic re-`AUTH` is suppressed on the readers; read by
    /// [`ValkeyClient::connect`] for its startup log line.
    pub(crate) reader_reauth: Option<Arc<dyn CredentialProvider>>,
}

/// Hand-written **redacting** `Debug`. The trait is required only so
/// `Result<FredConfigs, _>::expect_err` compiles in the fail-closed unit tests;
/// it must never be a route by which a credential reaches a log. A `#[derive]`
/// would be exactly that: fred's `Config` derives `Debug` and carries the
/// URL-embedded `username` / `password` in the clear, so deriving here would
/// print them on any `{:?}`. Prints only the non-secret cadence facts.
impl std::fmt::Debug for FredConfigs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FredConfigs")
            .field("shared", &"<redacted>")
            .field("pubsub", &"<redacted>")
            .field("pubsub_refresh_disabled", &self.pubsub_refresh_disabled)
            .field("reader", &"<redacted>")
            .field("reader_refresh_disabled", &self.reader_reauth.is_some())
            .finish()
    }
}

/// Derive the pub/sub `SubscriberClient`'s `FredConfig` from the shared one.
///
/// Returns the derived config plus whether the periodic re-`AUTH` was actually
/// suppressed on it.
///
/// **Only the `credential_provider` differs, and only its *cadence*.** The
/// wrapper delegates `fetch()` to the same inner `Arc`, so the credential the
/// pub/sub connection presents at handshake is byte-identical to every other
/// connection's — which is why this does not weaken the "one credential-provider
/// policy across all six connections" guarantee (SDD §15.3). Every other field
/// — TLS, server, blocking, version, fail_fast — rides the `clone()` untouched.
///
/// **Why the pub/sub connection alone.** It is permanently in RESP2 subscribe
/// state (`main.rs` subscribes `le:session:deleted` at startup), where Valkey
/// refuses `AUTH` outright (`only (P|S)SUBSCRIBE / (P|S)UNSUBSCRIBE / PING /
/// QUIT / RESET are allowed in this context`). fred parses that as the non-fatal
/// `ErrorKind::Unknown`, so the refresh task neither disconnects nor records
/// anything — it just `warn!`s every `[300 s, 360 s]`, forever, on every
/// passwordless deployment. The refresh there was already a no-op: that
/// connection renews its credential on the *reconnect* path
/// (`reconnect_on_auth_error = true` + handshake `AUTH`, which calls `fetch()`
/// afresh, + `manage_subscriptions()` re-subscribing). See
/// [`crate::storage::no_refresh_credentials`] for why log filtering was
/// disqualified.
///
/// **RESP2 gate.** The refusal above is a property of the *connection's RESP
/// version*, not of which command fred picks. Valkey applies the
/// subscribe-context command allowlist only to RESP2 connections: a connection
/// that handshook `HELLO 3` is never placed in subscribe-restricted state at
/// all — measured on Valkey 9.0.3, a RESP3 connection that has already issued
/// `SUBSCRIBE` still executes `AUTH`, `HELLO 3 AUTH …`, and even `GET`. A RESP2
/// connection in subscribed state refuses everything outside the pub/sub
/// allowlist, `HELLO` included, with the same error text as `AUTH` — so fred's
/// RESP3-only switch from `AUTH` to `HELLO`
/// (`fred-10.1.0/src/modules/inner.rs:412-424`) is not what makes the refresh
/// land; the RESP3 handshake is. The conclusion is unchanged either way: under
/// RESP3 the periodic refresh genuinely reaches the server and must not be
/// suppressed, so wrapping is gated on `version == RESP2`. Gating on the version
/// means a future protocol flip restores the periodic refresh automatically
/// rather than silently masking it.
///
/// Under `Static` auth mode there is no provider at all and the two configs are
/// byte-identical.
pub(crate) fn derive_pubsub_config(shared: &FredConfig) -> (FredConfig, bool) {
    let mut pubsub = shared.clone();
    let disabled = match shared.credential_provider.as_ref() {
        Some(provider) if shared.version == RespVersion::RESP2 => {
            pubsub.credential_provider =
                Some(Arc::new(NoRefreshCredentialProvider::new(provider.clone())));
            true
        }
        // No provider (Static mode) — nothing to wrap, byte-identical clone.
        // RESP3 — a RESP3 connection is never put in subscribe-restricted
        // context, so the refresh actually reaches the server: the schedule is
        // load-bearing and must survive untouched.
        _ => false,
    };
    (pubsub, disabled)
}

/// Derive the stream readers' `FredConfig` from the shared one (b.19i), and
/// hand back the provider a reader consults for its own re-`AUTH`.
///
/// Returns the derived config plus the shared config's provider *unwrapped*
/// (the same `Arc`), or `None` when the shared config has none.
///
/// **The fourth sanctioned divergence: only the `credential_provider`'s
/// *cadence*.** When a provider is present it is wrapped in a
/// [`NoRefreshCredentialProvider`] — the same inner `Arc`, `refresh_interval()
/// = None` — so fred spawns no credential-refresh task on any reader client.
/// The wrapper delegates `fetch()` verbatim, so a reader's handshake presents a
/// credential byte-identical to every other connection's, and every other
/// field rides the `clone()` untouched. Unlike [`derive_pubsub_config`] this
/// applies under any RESP version: the reason is not a server refusal but the
/// reader's own command discipline.
///
/// **Why the readers.** A reader keeps its one connection parked in `XREAD
/// BLOCK`. fred's refresh is a second, independent task writing to that
/// connection, and fred decides whether to interrupt the block at send time,
/// outside the router, where it can race the read (a refresh that checks
/// before the router marks the read blocked, or one that loses to the reader's
/// re-issued read after an interrupt, queues behind a fresh 30 s `BLOCK`). With
/// fred's refresh off, a reader takes the same tick itself — the provider's
/// `refresh_interval()` and `fetch(None)` gate, through the provider returned
/// here — and sends the `AUTH` on its own connection between reads, waking an
/// in-flight read first (`storage::stream_mux`'s reader).
///
/// Under `Static` auth mode there is no provider: the config is a
/// byte-identical clone and the returned provider is `None`.
pub(crate) fn derive_reader_config(
    shared: &FredConfig,
) -> (FredConfig, Option<Arc<dyn CredentialProvider>>) {
    let mut reader = shared.clone();
    let inner = shared.credential_provider.clone();
    if let Some(provider) = inner.as_ref() {
        reader.credential_provider =
            Some(Arc::new(NoRefreshCredentialProvider::new(provider.clone())));
    }
    (reader, inner)
}

#[cfg(any(test, feature = "test-support"))]
tokio::task_local! {
    /// Test seam: a caller-supplied provider [`build_fred_config`] installs on
    /// the shared config, after the per-mode overrides, for every connection a
    /// [`ValkeyClient::connect`] polled inside
    /// [`with_credential_provider_for_test`] builds.
    static TEST_CREDENTIAL_PROVIDER: Arc<dyn CredentialProvider>;
}

/// Runs `fut` — typically a [`ValkeyClient::connect`] call — with `provider`
/// installed through [`build_fred_config`]: the shared config carries it, and
/// the pub/sub and reader configs are derived from that as in production. So
/// a test can drive the passwordless path (the readers' own re-`AUTH`, fred's
/// pooled refresh) against a store with no ACL, with any provider — for
/// example a `TokenFileCredentialProvider` with a sub-second cadence. The
/// provider replaces whatever the configured auth mode installed; the
/// scope covers only `fut`.
#[cfg(any(test, feature = "test-support"))]
pub async fn with_credential_provider_for_test<F: std::future::Future>(
    provider: Arc<dyn CredentialProvider>,
    fut: F,
) -> F::Output {
    TEST_CREDENTIAL_PROVIDER.scope(provider, fut).await
}

/// Build the `FredConfig`s every Valkey connection derives from.
///
/// [`ValkeyClient::connect`] MUST obtain its `FredConfig`s only through this
/// function. That makes the "all six connections share one policy" property
/// structural rather than a review-time convention: the data / txn /
/// maintenance / snapshot pools, the stream readers and the pub/sub subscriber
/// client are all built from values taken *after* the overrides are applied —
/// the four pools from clones of [`FredConfigs::shared`], the readers from
/// [`FredConfigs::reader`] and the pub/sub client from [`FredConfigs::pubsub`],
/// both derived from `shared` inside this same seam (by
/// [`derive_reader_config`] and [`derive_pubsub_config`]).
///
/// `redis_token_provider` is the Redis-scoped [`ServiceTokenProvider`] the
/// startup-wiring layer constructs for `auth_mode=accessTokenFile` (it needs
/// `Metrics` + a `CancellationToken` this module does not own). It is `None` in
/// every other mode and threaded straight into [`apply_fred_config_overrides`],
/// where the access-token-file arm wraps it in a [`TokenFileCredentialProvider`].
///
/// `iam_credentials_provider` is the AWS credential source the startup-wiring
/// layer resolves for `auth_mode=iam` (the standard provider chain / IRSA — its
/// STS round-trip needs the tokio runtime `main.rs` owns). It is `None` in every
/// other mode and threaded straight into [`apply_fred_config_overrides`], where
/// the `iam` arm wraps it in an [`IamCredentialProvider`]. Distinct from the
/// access-token-file `ServiceTokenProvider`: a different type for a different
/// backend, so it rides its own parameter rather than a shared one.
///
/// **TLS trust roots.** On a TLS URL, `FredConfig::from_url` installs
/// `TlsConnector::default_rustls()`, which loads its roots with
/// `rustls_native_certs::load_native_certs()`
/// (`vendor/fred-10.1.0/src/protocol/tls.rs`). That reads the runtime image's
/// system store or, when `SSL_CERT_FILE` or `SSL_CERT_DIR` is set, only those
/// locations instead; any load error fails this build
/// (`ConnectError::BadUrl`), so a bad path fails startup. Nothing here reads a
/// CA file the store's own chart mounts. The rule across the service's other
/// TLS clients is SDD §15.3.
pub(crate) fn build_fred_config(
    config: &Config,
    redis_token_provider: Option<Arc<ServiceTokenProvider>>,
    iam_credentials_provider: Option<SharedCredentialsProvider>,
) -> Result<FredConfigs, ConnectError> {
    let mut shared = FredConfig::from_url(&config.valkey_url)
        .map_err(|e| ConnectError::BadUrl(e.to_string()))?;
    apply_fred_config_overrides(
        &mut shared,
        config,
        redis_token_provider,
        iam_credentials_provider,
    )?;
    #[cfg(any(test, feature = "test-support"))]
    if let Ok(provider) = TEST_CREDENTIAL_PROVIDER.try_with(Arc::clone) {
        shared.credential_provider = Some(provider);
    }
    let (pubsub, pubsub_refresh_disabled) = derive_pubsub_config(&shared);
    let (reader, reader_reauth) = derive_reader_config(&shared);
    Ok(FredConfigs {
        shared,
        pubsub,
        pubsub_refresh_disabled,
        reader,
        reader_reauth,
    })
}

/// Apply live-edit's overrides to a fred `Config` parsed from
/// `LIVE_EDIT_VALKEY_URL`, before any `Builder::from_config` derives from it.
///
/// **The `FredConfig` mutation seam is [`build_fred_config`]** — this function,
/// plus the [`derive_pubsub_config`] and [`derive_reader_config`] calls it makes
/// immediately afterwards. Anything applied outside that seam — in particular
/// after the `fred_config.clone()` in [`ValkeyClient::connect`] — would leave
/// the six connection sets disagreeing on policy: a partial-TLS (or
/// partial-auth) deployment. New per-mode policy, such as the
/// `credential_provider` the passwordless auth modes set below, MUST extend
/// this function rather than add a second, competing mutation site.
///
/// [`derive_pubsub_config`] and [`derive_reader_config`] are the two sanctioned
/// *derived* writes, and neither is a competing site: each runs inside the same
/// seam, strictly after this function has made `shared` final, and touches
/// exactly one field on its own clone — re-wrapping the very `Arc` this
/// function installed, in a delegating wrapper that narrows only the re-`AUTH`
/// cadence. There is no second credential source, and every non-auth field
/// rides the clones untouched. (Under `test-support`, a test may replace the
/// installed provider through `with_credential_provider_for_test`, in
/// [`build_fred_config`] before either derivation.)
///
/// `Builder`-level overrides (connect timeout, reconnect policy, performance
/// config) are a separate, later phase — see [`apply_connect_timeout`].
///
/// Dispatches on [`ValkeyAuthMode`](crate::config::ValkeyAuthMode): `Static`
/// (the default) installs no `credential_provider`, keeping the `FredConfig`
/// byte-identical to a passwordless / URL-embedded-password deployment;
/// `AccessTokenFile` wraps the Redis-scoped [`ServiceTokenProvider`] in a
/// [`TokenFileCredentialProvider`] and installs it as the `credential_provider`
/// (SR-4.1 / SR-4.2); `Iam` wraps the AWS credential source in an
/// [`IamCredentialProvider`] and installs it as the `credential_provider`
/// (SDD §15.5). Every arm that installs a provider does so *here*, before
/// [`build_fred_config`] hands the finished configs to the pool-builder
/// `.clone()`s in [`ValkeyClient::connect`], so all six connections resolve the
/// same credential from the same provider `Arc`.
///
/// **Every non-erroring arm falls through to the shared
/// `apply_tls_host_mapping` tail — no arm early-returns `Ok`.** The
/// access-token-file and `iam` arms in particular set `credential_provider` and
/// then fall out of the `match`; early-returning `Ok(())` from inside either
/// would skip the SR-1.5 `TlsHostMapping::DefaultHost` step and break clustered
/// TLS on the managed backend (cert-name mismatch against discovered bare-IP
/// nodes) — a rollout-gated failure CI cannot catch (§15.3).
// Under --no-default-features the TLS tail is compiled out, but the
// access-token-file and `iam` arms still read `config` + their credential-source param and write
// `fred_config`, so every parameter remains used in every feature configuration
// and needs no unused-variable suppression.
pub(crate) fn apply_fred_config_overrides(
    fred_config: &mut FredConfig,
    config: &Config,
    redis_token_provider: Option<Arc<ServiceTokenProvider>>,
    iam_credentials_provider: Option<SharedCredentialsProvider>,
) -> Result<(), ConnectError> {
    match config.valkey_auth_mode {
        ValkeyAuthMode::Static => {
            // No `credential_provider` — byte-identical to the pre-auth-mode
            // FredConfig. Password (if any) travels via the URL as before
            // (SR-3.1). `redis_token_provider` is ignored here.
        }
        ValkeyAuthMode::AccessTokenFile => {
            // Install fred's `credential_provider` so every `AUTH` / `HELLO`
            // carries a live file-sourced token (SR-4.1 / SR-4.2). The
            // Redis-scoped `ServiceTokenProvider` is built at startup (it needs
            // `Metrics` + a `CancellationToken` this module does not own); fail
            // closed if it — or the username — is absent, rather than silently
            // building a provider-less (unauthenticated) connection.
            let provider = redis_token_provider.ok_or_else(|| {
                ConnectError::Build(
                    "accessTokenFile Valkey auth mode requires a Redis-scoped \
                     ServiceTokenProvider, but none was wired into connect"
                        .to_string(),
                )
            })?;
            let username = config.valkey_token_username.clone().ok_or_else(|| {
                ConnectError::Build(
                    "accessTokenFile Valkey auth mode requires \
                     LIVE_EDIT_VALKEY_TOKEN_USERNAME (the AUTH username, \
                     conventionally the token issuer's identity), but it was unset"
                        .to_string(),
                )
            })?;

            // Readiness path (no new `ProbeConfig` slot): when no unexpired
            // sidecar token is cached — not yet loaded, or the cached one has
            // actually expired; the ≥ 3 min refresh lead is a proactive re-read
            // trigger and no longer withholds a live token (b.d6b) —
            // `TokenFileCredentialProvider::fetch` returns `Err(Auth)`, the
            // connection's AUTH fails, and the existing Valkey PING readiness
            // probe (`probe_valkey` via `pool.ping`, src/observability/health.rs)
            // trips, flipping the replica to `NOT_SERVING`. The single existing
            // `ProbeConfig` slot remains the storage-api token's, unchanged.
            // `refresh_interval()` draws fresh per-call entropy, so each client
            // staggers independently across the jitter band (SR-4.2) — no seed
            // is threaded in here.
            fred_config.credential_provider = Some(Arc::new(TokenFileCredentialProvider::new(
                provider,
                username,
                VALKEY_TOKEN_REAUTH_BASE_INTERVAL,
                VALKEY_TOKEN_REAUTH_JITTER_BOUND,
                VALKEY_TOKEN_AUTH_FANOUT_WINDOW,
            )));
            // Fall through to the shared `apply_tls_host_mapping` tail below.
        }
        ValkeyAuthMode::Iam => {
            // Install fred's `credential_provider` so every `AUTH` / `HELLO`
            // carries a freshly minted AWS SigV4 pre-signed token (SDD §15.5).
            // The AWS credential source is resolved at startup (the IRSA/STS
            // default chain needs the tokio runtime `main.rs` owns); fail closed
            // if it — or any required signing input — is absent, rather than
            // silently building a provider-less (unauthenticated) connection.
            // Mirrors the access-token-file arm's `ok_or_else` guards above.
            let credentials_provider = iam_credentials_provider.ok_or_else(|| {
                ConnectError::Build(
                    "iam Valkey auth mode requires an AWS credential source, but \
                     none was wired into connect"
                        .to_string(),
                )
            })?;
            let username = config.valkey_iam_username.clone().ok_or_else(|| {
                ConnectError::Build(
                    "iam Valkey auth mode requires LIVE_EDIT_VALKEY_IAM_USERNAME \
                     (the IAM-enabled Valkey ACL username), but it was unset"
                        .to_string(),
                )
            })?;
            let region = config.valkey_iam_region.clone().ok_or_else(|| {
                ConnectError::Build(
                    "iam Valkey auth mode requires LIVE_EDIT_VALKEY_IAM_REGION \
                     (the AWS region for SigV4 signing), but it was unset"
                        .to_string(),
                )
            })?;
            let cluster_host = config.valkey_iam_cluster_host.clone().ok_or_else(|| {
                ConnectError::Build(
                    "iam Valkey auth mode requires \
                     LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST (the cluster host for \
                     SigV4 signing), but it was unset"
                        .to_string(),
                )
            })?;

            // Readiness path (no new `ProbeConfig` slot): when AWS credentials
            // cannot be resolved or signing fails, `IamCredentialProvider::fetch`
            // returns `Err(Auth)`, the connection's AUTH fails, and the existing
            // Valkey PING readiness probe trips the replica to `NOT_SERVING` —
            // same posture as the access-token-file arm. `refresh_interval()` draws fresh
            // per-call entropy, so each client staggers independently across the
            // jitter band — no seed is threaded in here.
            fred_config.credential_provider = Some(Arc::new(IamCredentialProvider::new(
                credentials_provider,
                username,
                region,
                cluster_host,
                config.valkey_iam_service,
                VALKEY_IAM_REAUTH_BASE_INTERVAL,
                VALKEY_IAM_REAUTH_JITTER_BOUND,
            )));
            // Fall through to the shared `apply_tls_host_mapping` tail below.
        }
    }

    #[cfg(feature = "valkey-tls")]
    apply_tls_host_mapping(fred_config);

    Ok(())
}

/// Pin fred's cluster TLS host-mapping policy to `DefaultHost` on any
/// TLS-enabled `FredConfig`, overriding the `TlsHostMapping::None` that
/// `FredConfig::from_url`'s auto-TLS path installs for a `rediss://` /
/// `rediss-cluster://` URL.
///
/// **Consumption path.** The policy is read during cluster discovery:
/// `modify_cluster_slot_hostnames` (`fred-10.1.0/src/protocol/cluster.rs:226`)
/// → `Server::set_tls_server_name` (`.../protocol/types.rs:144`), and the
/// resulting name is used to build the rustls `ServerName` at
/// `.../protocol/connection.rs:380`.
///
/// **It only changes the TLS server name, never the TCP target.**
/// `set_tls_server_name` writes `tls_server_name` and leaves `host` untouched,
/// so only the SNI / certificate-verification name is affected; the connection
/// still dials the same address `CLUSTER SLOTS` returned.
///
/// **No-op when discovery returns hostnames.** If a node advertises a hostname
/// — directly or via the `CLUSTER SLOTS` `metadata` block
/// (`.../protocol/cluster.rs:57-90`) — `DefaultHost` changes nothing; fred
/// already prefers that hostname over an IP regardless of policy. `DefaultHost`
/// only takes effect when discovery yields a bare IP, where it substitutes the
/// front-door host from `LIVE_EDIT_VALKEY_URL` (the host of the connection that
/// answered `CLUSTER SLOTS`) as the verification name.
///
/// **Why it is required, not defensive.** Azure Managed Redis under OSS cluster
/// policy is observed to return bare private IPv4 addresses with no `hostname`
/// metadata (`redis-rs` issue #1963: a Rust client against `*.redis.azure.net`
/// fails with `certificate not valid for name "<ip>"`). Under `None`, rustls
/// builds an `ServerName::IpAddress` from that literal, so verification succeeds
/// only if the node certificate carries a matching IP SAN; where it does not,
/// every post-discovery (non-seed) connection fails the handshake. `DefaultHost`
/// makes clustered TLS work on AMR at all. MemoryDB is understood to return
/// hostnames, where this is a no-op — but that half is inference, not observed,
/// and the exact AMR / MemoryDB `CLUSTER SLOTS` shape is confirmed only by
/// capturing real output at rollout. Because `DefaultHost` is a provable no-op
/// in the hostname case, the choice is correct under either answer.
///
/// **Residual case.** A self-hosted TLS cluster advertising bare IPs whose node
/// certs carry matching IP SANs but not the front-door name would work under
/// `None` and fail under `DefaultHost`. No such deployment exists today; this
/// seam makes reversal a one-line change.
#[cfg(feature = "valkey-tls")]
pub(crate) fn apply_tls_host_mapping(fred_config: &mut FredConfig) {
    // `from_url` leaves `.tls` at `None` for plaintext `redis://` /
    // `redis-cluster://` URLs, so this writes nothing on the plaintext path.
    if let Some(tls) = fred_config.tls.as_mut() {
        tls.hostnames = TlsHostMapping::DefaultHost;
    }
}

/// Describe the effective Valkey transport policy as `(tls, host_mapping)`
/// for the startup log.
///
/// Emitted before the first connect attempt so that a clustered-TLS
/// certificate failure — where the seed connection succeeds and only the
/// post-`CLUSTER SLOTS` node connections fail — can be read against the
/// policy that was actually in effect, rather than inferred from the build.
#[cfg(feature = "valkey-tls")]
pub(crate) fn describe_tls_policy(fred_config: &FredConfig) -> (&'static str, &'static str) {
    match fred_config.tls.as_ref() {
        None => ("disabled", "n/a"),
        Some(tls) => (
            "enabled",
            match tls.hostnames {
                TlsHostMapping::None => "none",
                TlsHostMapping::DefaultHost => "default-host",
                TlsHostMapping::Custom(_) => "custom",
            },
        ),
    }
}

/// No TLS feature compiled — fred drops a `rediss://` URL to plaintext
/// silently, which is what `Config::from_env`'s SR-1.4 guard rejects.
#[cfg(not(feature = "valkey-tls"))]
pub(crate) fn describe_tls_policy(_fred_config: &FredConfig) -> (&'static str, &'static str) {
    ("unavailable", "n/a")
}

// ─── fred-error → `ServiceError` mappers ────────────────────────────────────
//
// Overview of the three names below (b.fd9 follow-up — the naming evolved
// across three issues and the overlap deserves a single source-of-truth
// reference). All three classify fred `ErrorKind`s via
// [`is_transient_fred_error`]; they differ only in what `ServiceError`
// variant they emit on the transient side.
//
// **Three verdicts** (b.xd1), decided in this order:
//
//   1. A store write refusal ([`store_write_refusal`]: `-OOM`, `-MISCONF`,
//      `-READONLY`, `-NOREPLICAS`) → plain `ServiceError::Unavailable`
//      "`{op}`: the backing store refused a write this request needed because
//      `{condition}`; retry", from **all three** names alike. Never `Aborted`,
//      so no in-process loop retries it (the condition can persist); no retry
//      hint, no nothing-applied claim, no count. The sites that can prove
//      nothing was applied answer before reaching a mapper
//      ([`refuse_store_write`]).
//   2. Transient ([`is_transient_fred_error`], including the store's
//      not-serving replies `-LOADING` / `-BUSY`) → the variant below.
//   3. Anything else → `ServiceError::Internal`.
//
//   * [`map_snapshot_load_err`] — *original* (b.1dz). Named for its first
//     call site: the `update_layer` snapshot-load loop's WATCH / HGETALL /
//     HSET-queue round-trips. Transient → `ServiceError::Aborted` with the
//     `"transient I/O:"` prefix so the outer `retry_aborted_with_jitter`
//     helper picks it up alongside WATCH-conflict aborts. Permanent →
//     `ServiceError::Internal` and the retry loop exits. The name persists
//     for source-archaeology continuity; new call sites should prefer the
//     alias [`map_transient_fred_err`] which describes the *intent* (use
//     this when calling from inside a retry loop) rather than the
//     *original site* (snapshot load).
//   * [`map_transient_fred_err`] — *alias* of `map_snapshot_load_err`,
//     introduced in b.k1r H-stor-6 when the same classifier was extended
//     to EXEC-reply / MULTI-queue sites that aren't part of the snapshot-
//     load phase. Use this from any site that sits inside a
//     [`crate::storage::retry::retry_aborted_with_jitter`] loop — the
//     `Aborted` variant is the helper's in-band retry signal.
//   * [`map_fred_err_for_grpc`] — b.fd9. Same transient/permanent
//     classification, different variant: transient →
//     `ServiceError::Unavailable` (gRPC `Code::Unavailable`, 14 — retriable
//     per the standard gRPC client retry contract, and by the Kit client).
//     Use this from single-shot call sites whose result surfaces directly
//     to a gRPC handler (no in-process retry budget interposed) — e.g.
//     `LayerStore::list_layer_names`, `SessionStore::get_base_path`,
//     `JoinSession`'s presence reads/writes.
//
// **Wire code.** Both variants reach the client as `UNAVAILABLE`, never as
// gRPC `ABORTED` (the service's one `ABORTED` is a chained `UpdateLayer`'s
// predecessor refusal, which no fred error produces), because
// `From<ServiceError> for tonic::Status`
// (`src/error.rs`) maps an `Aborted` that escapes a handler to `UNAVAILABLE`
// with its message verbatim. An `Aborted` escapes from a retry loop that ran
// out of attempts, or from a single-attempt site that returns the in-loop
// signal: the SubscribeReadLayer catch-up's snapshot read (which reuses the
// in-loop mapper) and `SessionStore::delete`'s `EXEC` arms (which
// raise `Aborted` directly). The checkpoint recovery reads are not on that
// list: each runs inside a `retry_aborted_with_jitter` loop whose exhausted
// result the restore consumes rather than returns, so none escapes as
// `Aborted`. An exhausted read feeds the restore's own decision, and when that
// decision is a refusal the result reaches the client through the restore
// error contract (`ServiceError::from_recovery_error`: `Transient` →
// `UNAVAILABLE`, anything else → `INTERNAL`), never through the `Aborted` arm.
// The one handler that rewraps an `Aborted` itself is DeleteSession's cascade
// (`delete_session_inner`, `src/grpc/session.rs`), which turns a layer's
// `Aborted` / `Unavailable` into an `Unavailable` carrying the progress counts
// — the same wire code. The only client-visible difference between the two
// mappers is the `"transient I/O:"` message prefix.
//
// Why not collapse the two `transient_*` names into one parameterised
// helper? The split is intentional: the *output variant* encodes the
// *in-process* contract. `Aborted` is the in-band signal
// [`crate::storage::retry::retry_aborted_with_jitter`] retries on, so a
// `map_fred_err_for_grpc` call inside a retry loop would emit `Unavailable`
// that the helper does not retry on and silently disable the loop's
// transient-I/O retry. The opposite mistake — `map_transient_fred_err` at a
// single-shot site — is harmless on the wire (same `UNAVAILABLE`), but the
// `Aborted` it returns reads to the next maintainer as "a loop retries this",
// so single-shot sites should still use `map_fred_err_for_grpc`. Two named
// functions force the reviewer to think about which contract applies at each
// site; one parameterised function would invite a flag-default mistake. The
// cross-helper classifier symmetry is locked in by the
// `transient_classification_test::map_transient_fred_err_and_map_fred_err_for_grpc_agree_on_transient_set`
// regression test (b.fd9).

/// Map a fred error produced during any Valkey round-trip in the
/// `update_layer` commit path (WATCH, HGETALL, MULTI queue, EXEC, EVALSHA)
/// to a `ServiceError`. Transient I/O failures become
/// `ServiceError::Aborted` with a `"transient I/O: ..."` prefix so the outer
/// retry loop picks them up alongside WATCH-conflict aborts. A store write
/// refusal ([`store_write_refusal`]) becomes a plain
/// `ServiceError::Unavailable` naming `op` and the condition, which exits the
/// retry loop without being retried (see the mapper overview above).
/// Everything else surfaces as `ServiceError::Internal` (persistent bug, bad
/// data, protocol error) and exits the retry loop.
///
/// "Transient" is whatever [`is_transient_fred_error`] says, which is a set of
/// `ErrorKind`s **plus four message-matched carve-out families**
/// ([`is_router_no_connection_error`], b.c7a,
/// [`is_router_retry_budget_exhausted_error`], b.cv4,
/// [`is_store_not_serving_reply`], b.xd1, and [`is_tryagain_reply`], b.vn4)
/// — so this helper is not a pure kind-level split. A `Routing` / `Unknown`
/// error carrying fred's `"Failed to route command"` message retries here
/// rather than exiting the loop, and so does one carrying any of the router's
/// retry-budget-exhaustion messages — which is how a Valkey-Cluster topology
/// change, or a dropped connection whose command is re-dispatched with a spent
/// attempt budget, reaches this path — and so does the store's `-LOADING` /
/// `-BUSY` while a node replays its dataset or runs a long script, and a
/// cluster node's `-TRYAGAIN` to a multi-key command during a slot migration.
/// See those predicates for why.
///
/// `op` is a short human-readable label for the Valkey command being wrapped
/// (e.g. `"HGETALL obj header"`) — it appears verbatim in every message so
/// operators can tell which call failed.
///
/// b.k1r H-stor-6: this helper is also used on the EXEC reply classification
/// alongside the dedicated `is_canceled()` branch, so a Valkey restart that
/// drops the connection between MULTI-queue completion and EXEC dispatch is
/// retried on a fresh txn-pool slot rather than surfacing as `Internal`. See
/// [`map_transient_fred_err`] for the alias re-exported at the call site.
///
/// `origin` counts a command timeout (see [`FredErrOrigin`]); it never changes
/// the returned error.
///
/// See `b.1dz` for the original classification rationale.
pub fn map_snapshot_load_err(
    origin: FredErrOrigin<'_>,
    op: &str,
    e: fred::error::Error,
) -> crate::error::ServiceError {
    origin.observe_error(&e);
    if let Some(refusal) = store_write_refusal(&e) {
        refusal.into_mapped_service_error(op)
    } else if is_transient_fred_error(&e) {
        crate::error::ServiceError::Aborted(format!(
            "transient I/O: {op} failed ({kind}): {e}",
            kind = e.kind().to_str()
        ))
    } else {
        crate::error::ServiceError::Internal(format!("{op} failed: {e}"))
    }
}

/// Alias for [`map_snapshot_load_err`] used by non-snapshot-phase commit-path
/// error sites (EXEC reply, MULTI queue return, EVALSHA) so the classification
/// is uniform across every Valkey round-trip in `update_layer` (b.k1r H-stor-6).
///
/// Pre-b.k1r the EXEC reply ran a local `Err(e) => ServiceError::Internal(...)`
/// arm that did not consult [`is_transient_fred_error`]. A Valkey restart
/// mid-EXEC therefore surfaced as `Internal` even though `WATCH / HGETALL /
/// HSET-queue` snapshot-phase failures with the same fred `ErrorKind` already
/// mapped to `Aborted` (retried). This alias fixes the asymmetry so all four
/// round-trip types route through the same classifier.
///
/// **Use [`map_transient_fred_err`] from inside a retry loop** — `Aborted` is
/// the in-band retry signal for [`crate::storage::retry::retry_aborted_with_jitter`].
/// For Valkey round-trips that run **outside** any retry loop (single-shot
/// reads / writes that surface directly to a gRPC handler), use
/// [`map_fred_err_for_grpc`] instead, so the returned variant does not claim a
/// loop will retry it. Either way the client sees `Code::Unavailable`: an
/// `Aborted` that escapes a handler is mapped to `UNAVAILABLE` at the wire
/// (`src/error.rs`), never to `ABORTED`.
pub fn map_transient_fred_err(
    origin: FredErrOrigin<'_>,
    op: &str,
    e: fred::error::Error,
) -> crate::error::ServiceError {
    map_snapshot_load_err(origin, op, e)
}

/// Map a fred error produced by a Valkey round-trip that runs **outside** any
/// retry loop into a `ServiceError` suitable for direct gRPC return (b.fd9).
///
/// Used by call sites that surface their result straight to a gRPC handler —
/// e.g. `LayerStore::list_layer_names`, `SessionStore::get_base_path`, every
/// JoinSession presence read/write. These have no in-process retry budget, so
/// the classification needs to land at a gRPC code that **standard client
/// retry policies treat as retriable**:
///
/// - Transient (`is_transient_fred_error`) → [`ServiceError::Unavailable`] →
///   gRPC `Code::Unavailable` (14, retriable per the gRPC retry contract — see
///   b.ohw H-grpc-1 for the same code path on auth-backend outages). Six
///   things land here:
///   - by `ErrorKind`: IO / Timeout / Canceled / Cluster / Backpressure;
///   - by *message*: fred's router miss — a `Routing` or `Unknown` error whose
///     text starts `"Failed to route command"`
///     ([`is_router_no_connection_error`], b.c7a);
///   - by *message*: fred's router retry-budget exhaustion — a `Routing` or
///     `Unknown` error carrying one of four prefixes
///     ([`is_router_retry_budget_exhausted_error`], b.cv4), which is how a
///     Valkey-Cluster `MOVED` / `ASK` the router could not complete arrives;
///   - by *message*: fred's `Transaction` router giving up — an `Unknown`
///     error carrying `"Max redirections exceeded"` or `"Max attempts
///     exceeded"` ([`is_transaction_router_exhausted_error`], b.cyu), which is
///     how a `MOVED` / `ASK` reaches a fred `Transaction` on the `txn_pool`;
///   - by *message*: the store not serving — an `Unknown` error carrying
///     `-LOADING` or `-BUSY` ([`is_store_not_serving_reply`], b.xd1);
///   - by *message*: a multi-key command refused during a slot migration —
///     an `Unknown` error carrying `-TRYAGAIN` ([`is_tryagain_reply`], b.vn4).
///
///   The message matches are what break the otherwise kind-level split, so
///   the last bullet is not the whole story for `Routing` / `Unknown`.
/// - Store write refusal ([`store_write_refusal`]: `-OOM` / `-MISCONF` /
///   `-READONLY` / `-NOREPLICAS`, checked first) → a plain
///   [`ServiceError::Unavailable`] naming `op` and the condition, ending
///   `; retry` (b.xd1) — the same answer [`map_transient_fred_err`] gives, so
///   the client retries it and no in-process loop does.
/// - Permanent (every other fred `ErrorKind`: Protocol / Parse / Auth / Config /
///   Url / Routing / Sentinel / NotFound / Unknown / Invalid*, **less the
///   message-matched carve-outs above**) → [`ServiceError::Internal`] → gRPC
///   `Code::Internal` (13, non-retriable; indicates a real bug or contract
///   violation that warrants operator attention).
///
/// The transient bucket is identical to [`map_transient_fred_err`] —
/// classification is unified via [`is_transient_fred_error`] — but the
/// **in-process variant differs**:
///
/// - Inside a retry loop, transient → `Aborted` (loop catches and retries;
///   on exhaustion it reaches the client as `UNAVAILABLE`).
/// - Outside a retry loop, transient → `Unavailable` (client retries).
///
/// Both land on gRPC `UNAVAILABLE`, never on `ABORTED` (see the mapper
/// overview above and `From<ServiceError> for tonic::Status`).
///
/// `op` is a short human-readable label for the Valkey command being wrapped
/// (e.g. `"SMEMBERS layers index"`); it appears verbatim in the message body
/// so operators can tell which call site produced the error from logs alone.
/// The `Unavailable` variant preserves the message body (operator-facing); the
/// `Internal` variant has its message scrubbed in `ServiceError::from` so the
/// label only reaches operator logs, never the client.
///
/// `origin` counts a command timeout (see [`FredErrOrigin`]); it never changes
/// the returned error.
pub fn map_fred_err_for_grpc(
    origin: FredErrOrigin<'_>,
    op: &str,
    e: fred::error::Error,
) -> crate::error::ServiceError {
    origin.observe_error(&e);
    if let Some(refusal) = store_write_refusal(&e) {
        refusal.into_mapped_service_error(op)
    } else if is_transient_fred_error(&e) {
        crate::error::ServiceError::Unavailable(format!(
            "{op} failed ({kind}): {e}",
            kind = e.kind().to_str()
        ))
    } else {
        crate::error::ServiceError::Internal(format!("{op} failed: {e}"))
    }
}

/// Minimum reconnect backoff in ms (exponential backoff starts here).
const RECONNECT_MIN_DELAY_MS: u32 = 100;
/// Exponential base for the reconnect backoff (delay *= base each attempt).
const RECONNECT_BASE: u32 = 2;

/// The reconnect policy every connection's `Builder` carries — the pools',
/// the pub/sub client's and the stream readers' template's: unlimited
/// attempts, exponential backoff from [`RECONNECT_MIN_DELAY_MS`] up to
/// `valkey_reconnect_max_delay_ms`. fred ships with `policy = None` by
/// default, meaning a dropped connection is never re-established — we need
/// an explicit policy so the pool self-heals after a Valkey restart.
fn shared_reconnect_policy(config: &Config) -> ReconnectPolicy {
    ReconnectPolicy::new_exponential(
        0, // max_attempts: 0 = unlimited
        RECONNECT_MIN_DELAY_MS,
        config.valkey_reconnect_max_delay_ms,
        RECONNECT_BASE,
    )
}

/// The performance config every connection's `Builder` carries (the pub/sub
/// client's raises its broadcast channel capacity on top of it): the
/// per-command timeout, which keeps commands from hanging indefinitely while
/// waiting for a pooled connection to reconnect. fred's default is 0 (no
/// timeout), which turns every server outage into an indefinite hang for any
/// command dispatched during the gap.
fn shared_performance_config(config: &Config) -> PerformanceConfig {
    PerformanceConfig {
        default_command_timeout: Duration::from_millis(config.valkey_command_timeout_ms),
        ..Default::default()
    }
}

/// fred `PerformanceConfig::broadcast_channel_capacity` for the dedicated
/// pub/sub `SubscriberClient` ([`ValkeyClient::subscriber`]) — the only client
/// in this service that subscribes, so the only one that receives pub/sub
/// messages. A constant, not an env var: like the presence replay's page
/// size, a capacity becomes an operator knob only if measurement shows a need.
///
/// **Why.** fred delivers every pub/sub message on that connection through one
/// tokio `broadcast` channel whose receivers are the replica-wide relays: the
/// presence relay (`presence::spawn_presence_subscriber`) and the
/// session-deleted relay (`grpc::session::spawn_session_delete_subscriber`).
/// A relay that falls more than the channel's capacity behind gets
/// `Lagged` and the skipped messages are gone for this replica — for
/// presence, missing for every local `JoinSession` stream in the session.
/// fred's default is 32 (`fred-10.1.0/src/types/config.rs:540`), which a
/// 50-client simultaneous join overran. 4096 matches the default lag bound
/// of the per-session presence fan-out downstream of the presence relay
/// (`LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY`), which, unlike this channel,
/// allocates only for undelivered events. A lag here is counted on
/// `live_edit_pubsub_relay_lagged_total`.
///
/// **The relays drain the channel; its capacity is not their per-turn
/// limit.** Each relay reads through `presence::PubsubRelayReceiver`: after
/// each awaited `recv` it takes everything already queued with `try_recv`,
/// which tokio's cooperative budget does not limit, and yields only after one
/// capacity's worth of messages (this constant, read back from the client's
/// `PerformanceConfig`). Awaiting `recv` alone capped a relay at the budget's
/// 128 messages per task poll, after which it waited a full run-queue cycle.
/// In the 30,000-connection join and close waves of the keeper-load runs a
/// relay lagged at 4096 with its lags milliseconds apart, far inside the time
/// a wave takes to fill 4096 slots; from that and the code, the cap, not this
/// capacity, is inferred to be what let it fall behind. That is not measured
/// at that scale; at 1,000 streams under 1,000 messages/s neither relay
/// lagged (docs-internal/validation/relay-drain-20261005/report.md §2). This
/// capacity stays 4096: measure before any deployment expecting more than
/// about 5,000 concurrent JoinSession streams per replica, and raise it if a
/// relay lags there (SDD §13.3).
///
/// **Memory, ≈ 2.9 MiB per replica.** fred sizes all eight broadcast channels
/// of a client's `Notifications` (`fred-10.1.0/src/modules/inner.rs`
/// `Notifications::new`: errors, pubsub, keyspace, reconnect, cluster change,
/// connect, close, unresponsive) with this one capacity, and tokio 1.53
/// allocates each channel's ring eagerly: `capacity.next_power_of_two()`
/// `Mutex<Slot<T>>` elements, every one initialised
/// (`tokio/src/sync/broadcast.rs`, `new_with_receiver_count`), so the pages
/// are resident at build time. With tokio's `parking_lot` mutex (feature
/// `full`) a slot is `size_of::<T>() + 24` B rounded to alignment, measured
/// against fred 10.1.0 with this crate's features: `Message` (pubsub) 192 B,
/// `(Error, Option<Server>)` 128 B, `Server` 96 B (reconnect, unresponsive),
/// `KeyspaceEvent` 88 B, `Result<(), Error>` 56 B,
/// `Vec<ClusterStateChange>` 48 B, `()` 32 B — 736 B per slot index across
/// the eight. At 4096 slots that is 4096 × 736 = 3,014,656 B requested
/// (786,432 B of it the pubsub ring), and 3,047,424 B in jemalloc size
/// classes (only the keyspace ring, 360,448 B, rounds up, to 393,216 B), on
/// the one client per replica. At fred's default of 32 it was 23,552 B.
/// Added to that are the payloads of messages not yet read by every relay
/// (at most 4096 of them), which a ring slot holds until its last receiver
/// reads it.
///
/// Applied to the pub/sub client alone: on the command pools, which
/// never subscribe and whose event channels carry only connection events,
/// the same capacity would allocate those 2.9 MiB once per pooled connection
/// for nothing.
pub(crate) const PUBSUB_BROADCAST_CHANNEL_CAPACITY: usize = 4096;

/// Base tick cadence for the access-token-file Valkey credential provider
/// (SR-4.2, re-sized in b.7td).
///
/// **This is how often the provider is consulted, not how often fred sends an
/// `AUTH`.** Pre-b.7td the two were the same thing and the constant was `300 s`:
/// every tick on every connection produced a re-`AUTH`, which on a cluster fans
/// out to every primary (`_AuthAllCluster`). That broadcast is where the field
/// failure surfaced — it is unbounded head-of-line work with no deadline of its
/// own, it is the command most likely to land on a connection carrying leaked
/// transaction state, and it wipes the watch set of any connection it forces a
/// reconnect on. `TokenFileCredentialProvider::fetch` now gates the broadcast on
/// an actual token rotation, so the overwhelming majority of ticks send nothing.
///
/// With the broadcast gated, a *short* cadence is both cheap and strictly
/// better. What a re-AUTH buys is *prolongation*: a connection AUTHed with the
/// old token is dropped by the backing store at that token's expiry, so only a
/// re-AUTH landing after the sidecar rotates and before the old token dies
/// carries the fresh credential onto the existing connection. The width of that
/// window is set by the **sidecar's** refresh lead less up to one
/// `LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS` (15 s by default) — not by this
/// service's ≥ 3 min proactive re-read lead, which withholds nothing (`fetch()`
/// reads [`ServiceTokenProvider::cached_bearer_if_unexpired`], so a cached token
/// authenticates for its whole real lifetime). A `300–360 s` cadence usually
/// missed that window outright; `60–90 s` usually lands in it. So the change
/// cuts broadcasts *and* improves pickup.
///
/// Missing it still costs a reconnect, not an outage. Every pooled connection
/// was AUTHed with the same token and shares its `expires_on`, so on a total
/// miss they are dropped together at expiry and immediately re-established — the
/// fresh token is cached by then, so the reconnect AUTHs successfully. One brief
/// connection blip per token lifetime (~24 h), not a failed-auth window.
///
/// Sizing the sidecar lead to cover the band turns that blip into a seamless
/// prolongation; that is a deployment recommendation and lives in the operator
/// docs. It is deliberately NOT a startup check (b.is1): the term it constrains
/// is the *sidecar's* refresh lead, which is not a live-edit config value and is
/// invisible at boot. Instead [`VALKEY_TOKEN_ROTATION_FANOUT_CEILING`] is passed
/// to the Redis-scoped [`ServiceTokenProvider`] as its prolongation floor (wired
/// in `main.rs`, so `auth` gains no dependency edge on `storage`), and the
/// provider observes the real per-rotation headroom at runtime — recording
/// `live_edit_service_token_refresh_headroom_seconds{role="valkey"}` and
/// WARNing once per rotation that lands under it.
///
/// Private: the wiring layer consumes the derived
/// [`VALKEY_TOKEN_ROTATION_FANOUT_CEILING`] instead, so the ceiling has exactly
/// one definition site and `main.rs` cannot re-derive it out of step.
///
/// Paired with [`VALKEY_TOKEN_REAUTH_JITTER_BOUND`] for the per-client
/// thundering-herd stagger.
///
/// [`ServiceTokenProvider`]: crate::auth::service_token::ServiceTokenProvider
/// [`ServiceTokenProvider::cached_bearer_if_unexpired`]: crate::auth::service_token::ServiceTokenProvider::cached_bearer_if_unexpired
const VALKEY_TOKEN_REAUTH_BASE_INTERVAL: Duration = Duration::from_secs(60);
/// Upper bound on the per-client stagger added to
/// [`VALKEY_TOKEN_REAUTH_BASE_INTERVAL`] (SR-4.2). fred adds no jitter of its
/// own, so a fixed interval would tick every replica's every connection in
/// lockstep — the synchronized-refresh thundering herd Azure warns against. The
/// realized tick interval is `base + uniform(0, this)`; `refresh_interval()`
/// draws fresh entropy per call, so each client staggers independently (the
/// provider is one shared `Arc`, so a per-process/per-instance seed would give
/// every client in a replica the same interval — no per-client spread).
///
/// Held at half the base so the band stays narrow enough for
/// [`VALKEY_TOKEN_AUTH_FANOUT_WINDOW`] to cover it comfortably (see the
/// assertion below) while still spreading clients across 30 s.
///
/// Private alongside [`VALKEY_TOKEN_REAUTH_BASE_INTERVAL`], and for the same
/// reason: what the wiring layer needs is published once, as
/// [`VALKEY_TOKEN_ROTATION_FANOUT_CEILING`].
const VALKEY_TOKEN_REAUTH_JITTER_BOUND: Duration = Duration::from_secs(30);

/// Largest realized provider-tick interval on the access-token-file path:
/// [`VALKEY_TOKEN_REAUTH_BASE_INTERVAL`] plus the whole
/// [`VALKEY_TOKEN_REAUTH_JITTER_BOUND`] (`90 s`).
///
/// Named for what it is — the tick ceiling — and no longer, post-b.7td, the
/// prolongation floor: a tick only *sends* an `AUTH` when the rotation gate
/// opens, so the latency a rotation actually takes to reach every connection is
/// [`VALKEY_TOKEN_ROTATION_FANOUT_CEILING`], not this.
pub const VALKEY_TOKEN_REAUTH_MAX_INTERVAL: Duration =
    VALKEY_TOKEN_REAUTH_BASE_INTERVAL.saturating_add(VALKEY_TOKEN_REAUTH_JITTER_BOUND);

/// Worst-case latency from "the polling task caches a rotated token" to "every
/// pooled connection has re-`AUTH`ed with it" (`180 s`) — and therefore the
/// **prolongation floor** handed to the Redis-scoped [`ServiceTokenProvider`].
///
/// Two tick intervals, not one, and the doubling is the whole point of the
/// constant (b.7td). Under the rotation gate a fan-out has two serial stages:
///
/// 1. Up to one [`VALKEY_TOKEN_REAUTH_MAX_INTERVAL`] passes before *any* client
///    ticks and first observes the new token — that observation is what opens
///    [`VALKEY_TOKEN_AUTH_FANOUT_WINDOW`].
/// 2. Up to one more passes before the *last* client takes its next tick inside
///    that window and re-`AUTH`s.
///
/// A rotation picked up with less headroom than this before the outgoing token's
/// expiry cannot be guaranteed to reach every live connection, so the stragglers
/// are dropped and re-established at expiry instead of prolonged — which is
/// exactly the condition the provider's per-rotation headroom WARN reports.
///
/// Still less than the `360 s` the pre-b.7td (ungated, `300–360 s`) path
/// required, so the operator-facing sidecar-lead recommendation derived from it
/// gets *looser*, not tighter.
///
/// Exists so the floor has **one** definition site. `main.rs` passes it whole to
/// the Redis-scoped [`ServiceTokenProvider`] (`auth` therefore gains no
/// dependency edge on `storage`); the part constants stay private so no caller
/// can re-derive it and drift. `pub` for that wiring: `main.rs` is a separate
/// crate from the library, so `pub(crate)` would not reach it.
///
/// [`ServiceTokenProvider`]: crate::auth::service_token::ServiceTokenProvider
pub const VALKEY_TOKEN_ROTATION_FANOUT_CEILING: Duration =
    VALKEY_TOKEN_REAUTH_MAX_INTERVAL.saturating_mul(2);

/// How long after a rotation is first observed the access-token-file provider
/// keeps answering the periodic tick with a credential (b.7td).
///
/// The provider is one shared `Arc` across every pool and client, so it has no
/// per-client identity and cannot track who has already re-`AUTH`ed. A time
/// window is the only mechanism available, and it must be at least
/// [`VALKEY_TOKEN_REAUTH_MAX_INTERVAL`]: a client whose tick fired microseconds
/// before the rotation was observed will not tick again for that long.
///
/// 150 s covers the shipped 60 s + 30 s cadence with 60 s of slack for task
/// scheduling and clock jitter. The slack is cheap and bounded — at worst one
/// extra broadcast per client per rotation, i.e. once per token lifetime,
/// against the ~288 per client per day the ungated 300 s cadence issued.
///
/// Lives here, next to the cadence it must dominate, rather than in
/// `storage::token_file_credentials`: all three numbers are one design, the
/// provider takes the window as a constructor argument like the other two, and
/// co-locating them is what lets the pairing below be a `const` assertion.
const VALKEY_TOKEN_AUTH_FANOUT_WINDOW: Duration = Duration::from_secs(150);

// The cadence / fan-out-window pairing, mechanized (b.7td). The provider's own
// `debug_assert!` catches this too, but only in a debug build that actually
// constructs the provider — which is not the access-token-file path unless the
// deployment is in that mode. Getting it wrong fails silently and slowly: a
// client whose tick fires microseconds before a rotation is observed simply
// never receives the new credential and is dropped at expiry instead of
// prolonged, which looks like ordinary connection churn.
const _: () = assert!(
    VALKEY_TOKEN_REAUTH_MAX_INTERVAL.as_millis() <= VALKEY_TOKEN_AUTH_FANOUT_WINDOW.as_millis(),
    "VALKEY_TOKEN_AUTH_FANOUT_WINDOW must be at least the largest realized tick interval \
     (VALKEY_TOKEN_REAUTH_BASE_INTERVAL + VALKEY_TOKEN_REAUTH_JITTER_BOUND), or a client whose \
     tick fired just before a rotation was observed gets no tick inside the window and never \
     re-AUTHs with the new token. Lower the cadence, or raise the window (and re-check \
     VALKEY_TOKEN_ROTATION_FANOUT_CEILING, which is derived from the cadence, not the window)."
);

/// Base re-AUTH cadence for the IAM Valkey credential provider (SDD §15.5).
///
/// The AWS analog of [`VALKEY_TOKEN_REAUTH_BASE_INTERVAL`]. `IamCredentialProvider`
/// mints a fresh SigV4 pre-signed token — valid for 15 minutes (`900 s`,
/// [`TOKEN_VALIDITY_SECS`](super::iam_credentials)) — on every `AUTH` fred
/// re-issues at [`CredentialProvider::refresh_interval`](fred::types::config::CredentialProvider::refresh_interval)
/// intervals. Sized well under that 15-minute expiry (`300 s + up to 60 s ≈
/// 360 s ≪ 900 s`) so re-AUTH always fires with ample headroom before a token
/// would expire, and so the periodic re-AUTH prolongs the connection ahead of
/// AWS's 12-hour force-disconnect. Paired with [`VALKEY_IAM_REAUTH_JITTER_BOUND`]
/// for the per-client thundering-herd stagger.
///
/// **Deliberately left at `300 s` when b.7td shortened the access-token-file
/// pair to `60 s`.** The two are no longer analogous, because the rotation gate
/// that justifies the short cadence there is inapplicable here:
/// `IamCredentialProvider` mints a *freshly signed* token on every `fetch()`, so
/// its credential is "rotated" every tick by construction and there is nothing
/// to gate on. Every IAM tick therefore still produces a real `AUTH` — which is
/// exactly what SDD §15.5's 12-hour force-disconnect prolongation invariant
/// depends on, and shortening the cadence here would multiply that broadcast for
/// no benefit.
const VALKEY_IAM_REAUTH_BASE_INTERVAL: Duration = Duration::from_secs(300);
/// Upper bound on the per-client stagger added to
/// [`VALKEY_IAM_REAUTH_BASE_INTERVAL`]. The AWS analog of
/// [`VALKEY_TOKEN_REAUTH_JITTER_BOUND`]: fred adds no jitter of its own, so a
/// fixed interval would re-AUTH every replica's every connection in lockstep —
/// a synchronized-refresh thundering herd. The realized re-AUTH interval is
/// `base + uniform(0, this)`; `IamCredentialProvider::refresh_interval()` draws
/// fresh entropy per call, so each client staggers independently (the provider
/// is one shared `Arc`, so a per-process seed would give every client the same
/// interval — no per-client spread).
const VALKEY_IAM_REAUTH_JITTER_BOUND: Duration = Duration::from_secs(60);

/// Error type for Valkey connection failures.
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error("invalid Valkey URL: {0}")]
    BadUrl(String),
    #[error("failed to build Valkey client: {0}")]
    Build(String),
    #[error("Valkey unreachable after {elapsed:.1}s: {source}")]
    Timeout {
        elapsed: f64,
        source: fred::error::Error,
    },
    #[error("Valkey PING failed on {connection}: {source}")]
    Ping {
        connection: &'static str,
        source: fred::error::Error,
    },
}

/// A cheaply cloneable Valkey client.
///
/// Wraps a round-robin `Pool` for data commands, a dedicated `SubscriberClient`
/// for pub/sub (Valkey requires separate connections for subscriptions and
/// regular commands), an `ExclusivePool` `txn_pool` for short-lived
/// WATCH/MULTI/EXEC transactions (UpdateLayer), and the stream multiplexer's
/// reader connections for `XREAD BLOCK` (SubscribeReadLayer), kept separate
/// from `txn_pool` so readers cannot starve writer transactions.
///
/// The `ExclusivePool` is required because fred's regular `Pool` multiplexes
/// multiple logical operations onto a shared underlying connection — which
/// makes WATCH broken under concurrency, since WATCH state is per-connection
/// in Valkey, and interleaving two transactions on the same connection means
/// neither WATCH ever trips. A checkout hands out an `OwnedMutexGuard<Client>`
/// that holds exclusive ownership for the lifetime of the transaction,
/// guaranteeing the WATCH/read/EXEC sequence shares one Valkey session.
///
/// The txn pool, the only exclusive pool, is reached only through
/// [`ExclusiveCheckout`] (b.f4f),
/// which owns the slot selection itself instead of calling
/// `ExclusivePool::acquire()`. fred's `acquire` scans once and then awaits ONE
/// selected mutex, so a queued caller can stay parked while every other slot
/// goes idle; the replacement takes a FIFO semaphore permit and then scans all
/// slots non-blockingly, and fred's blocking fallback is absent from the call
/// path. See `storage::checkout`'s module docs for the soundness argument and
/// for the two allow-listed sites that lock these mutexes directly.
#[derive(Clone)]
pub struct ValkeyClient {
    /// Round-robin data connection pool (shared, multiplexed). Use for
    /// non-transactional commands (single-shot reads, writes that don't need
    /// WATCH).
    pub pool: Pool,
    /// Exclusive connection pool for WATCH/MULTI/EXEC transactions, behind the
    /// service's own next-free checkout. Each checkout returns a
    /// [`CheckedOutClient`] holding one dedicated Valkey connection — and its
    /// semaphore permit — for the lifetime of the transaction. A slot's client
    /// can be replaced by a fresh one (`storage::member_replacement`), under
    /// the slot's own guard and never while it is checked out; the checkout
    /// and its permits are unchanged by it.
    ///
    /// `Arc` because `ValkeyClient` is `Clone` and every clone must share one
    /// permit semaphore; a per-clone semaphore would mean `clones × slots`
    /// permits against one set of slots, which is exactly the over-admission the
    /// permit exists to prevent.
    pub txn_pool: Arc<ExclusiveCheckout>,
    /// `SubscribeReadLayer` admission: caps concurrent subscriptions per
    /// replica at the subscription ceiling derived from the pod memory limit,
    /// independent of connections. A gate of 0, which only a `Config` built
    /// field by field can hold, admits without bound.
    pub subscriber_admission: Arc<SubscriberAdmissionGate>,
    /// Bounds `SubscribeReadLayer` catch-ups holding a slot for their
    /// backing-store reads and build — and, in rollback mode
    /// (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) on a same-object reconnect,
    /// the cluster stream-reader wait — at once. A shared fenced-read flight
    /// (`storage::snapshot_flights`) holds one slot for its whole life, fence
    /// sleeps and its attached catch-ups' build turns included, and each of
    /// its extra builds one more ([`CatchUpGate::try_acquire`]).
    pub catchup_gate: Arc<CatchUpGate>,
    /// Bounds the catch-up atomic snapshot reads in flight at once
    /// (`LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`, the atomic read-permit
    /// class; Spec SDD SR-7.1). A catch-up — a legacy in-window replay's
    /// per-commit reads included — takes a permit only while holding a
    /// [`Self::catchup_gate`] slot; the steady state never reads layer state,
    /// so it takes none. Each holds it across its atomic read alone. Closed by
    /// [`Self::quit`] with the other gates.
    pub catchup_read_gate: Arc<CatchUpReadGate>,
    /// Bounds the catch-up fenced snapshot reads in flight at once
    /// (`LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`, the fenced read-permit
    /// class; Spec SDD SR-7.1), so a multi-second fenced read of a layer
    /// above the atomic-read ceilings never holds a permit an atomic read
    /// needs. Taken by a shared fenced-read flight's read stage
    /// (`storage::snapshot_flights`), inside the flight's
    /// [`Self::catchup_gate`] slot, with no deadline; and by the structural
    /// cache's fenced cold start (`storage::structural_cache_fenced`), after
    /// the cache's one-at-a-time and plain permits and with a deadline at its
    /// waiting cap; and by the cache's sampled audit
    /// (`storage::structural_cache_audit`) for its fenced read, in the same
    /// order, only when a permit is free at once (it never waits). Closed by
    /// [`Self::quit`] with the other gates.
    pub catchup_fenced_read_gate: Arc<CatchUpReadGate>,
    /// The per-replica catch-up memory budget `T`, derived from the pod memory
    /// limit (`config::memory_plan`; Spec SDD SR-7.2): the byte bound beside
    /// the slot's CPU bound and the read permits. Closed by
    /// [`Self::quit`] with the gates.
    pub catchup_build_budget: Arc<MemoryBudget>,
    /// XREAD multiplexer over its own readers
    /// (`LIVE_EDIT_SUBSCRIBER_POOL_SIZE`), each with its own connection —
    /// opened when the reader first serves a subscription, and on a cluster to
    /// its bound slot's primary alone — separate from `txn_pool` so readers
    /// cannot starve `UpdateLayer` transactions.
    pub stream_mux: Arc<StreamMux>,
    /// Dedicated connection pool for the periodic GC sweep + orphan sweep
    /// (b.kf1 H-stor-5), and the reference-count reconciler's and the
    /// external-value backup repair's keyspace walks. Sized very small
    /// (default 1) because all of them
    /// are gated behind the `le:gc:leader` lock and normally one replica runs
    /// them at a time (two can briefly overlap after a lock takeover). Kept off the multiplexed data `pool` so that a long
    /// SCAN + HGETALL pass does not queue behind concurrent `update_layer`
    /// reads / writes that picked the same multiplexed connection.
    pub maintenance_pool: Pool,
    /// The snapshot lane: a dedicated multiplexed pool for object-snapshot
    /// reads, sized by `LIVE_EDIT_SNAPSHOT_POOL_SIZE` (Spec SDD SR-10.1). Kept
    /// off the data `pool` so a multi-MiB snapshot reply never queues in front
    /// of `update_layer` commands that picked the same multiplexed connection.
    /// Built from the data pool's builder like `maintenance_pool`, so it shares
    /// that pool's credentials, TLS policy, reconnect policy and timeouts.
    /// Borrow it through [`Self::acquire_snapshot`] so the
    /// `valkey_pool_in_use{pool="snapshot"}` gauge sees the use, and attribute
    /// its errors to [`POOL_LABEL_SNAPSHOT`]. Its members' routers run on the
    /// `snapshot-io` thread of [`Self::fanout_threads`].
    pub snapshot_pool: Pool,
    /// Dedicated pub/sub connection.
    pub subscriber: SubscriberClient,
    /// Watches the fred router task of every client above — each member of
    /// every pool and the pub/sub `subscriber` — and restarts one that ends
    /// other than by `quit()` (see `storage::router_supervisor`). `Arc` so every
    /// clone shares one set of watchers; [`Self::quit`] shuts it down before
    /// closing any pool.
    router_supervisor: Arc<RouterSupervisor>,
    /// Replaces a `data`, `maintenance` or `snapshot` pool member, or a `txn`
    /// slot's client, whose client keeps failing with a fresh client
    /// (`storage::member_replacement`).
    /// `Arc` so every clone shares one; [`Self::quit`] shuts it down before the
    /// router supervisor.
    member_replacer: Arc<MemberReplacer>,
    /// The snapshot fan-out's dedicated threads (`storage::fanout_thread`):
    /// `fanout-read`, which polls every fenced snapshot fan-out pass, the
    /// four `watch-read` threads, which poll every unfenced one, and
    /// `snapshot-io`, which hosts the snapshot lane's routers, so the fan-out
    /// allocates from a fixed set of allocator arenas whatever the worker
    /// count. Built by [`Self::connect`] before the snapshot lane connects.
    /// `Arc` so every clone shares the threads; they stop when the client's
    /// last owner drops (`owners`).
    fanout_threads: Arc<FanoutThreads>,
    /// The client's owners: `Some` on every clone held from outside the
    /// client, `None` on an internal clone ([`Self::internal_clone`]), the
    /// kind the client's own machinery keeps. When the last owner drops,
    /// [`ClientOwners`]'s drop releases the client — quit or not — and shuts
    /// the fan-out threads down. A clone keeps its kind.
    owners: Option<Arc<ClientOwners>>,
    /// Handles for fred-owned background tasks (b.3z7):
    /// - `SubscriberClient::manage_subscriptions()` — re-subscribes to tracked
    ///   pub/sub channels after a reconnect so presence + session-deleted
    ///   fan-out survives a Valkey blip.
    /// - Reconnect / error event listeners — tick metrics when fred raises a
    ///   reconnect or connection-error event.
    ///
    /// Stored inside an `Arc<Mutex<Option<_>>>` so that cloning `ValkeyClient`
    /// (required by `Clone`) does not duplicate the tasks. `quit()` drains the
    /// vec under the mutex and aborts each handle before closing the
    /// underlying connections.
    background_tasks: BackgroundTasks,
    /// Metrics handle used by [`acquire_txn`] and the stream multiplexer to
    /// populate [`PoolUseToken`]s for the `valkey_pool_in_use{pool}` gauge
    /// (b.3z7 H-obs-3). Populated by [`install_observability_hooks`] after
    /// `Metrics::new` returns; cloneable across `ValkeyClient` clones because
    /// it sits behind an `Arc<Mutex<_>>`. `None` means acquisitions don't
    /// affect the gauge — the state startup PINGs and tests that don't wire
    /// observability operate in.
    metrics: Arc<Mutex<Option<SharedMetrics>>>,
    /// The store out-of-memory signal (b.pa1; `storage::store_oom`): every
    /// [`FredErrOrigin::client`] on this client notes a store write refused
    /// `-OOM` here, which, at most once a second, requests the GC's
    /// out-of-memory collection (once startup armed it) and wakes this
    /// replica's GC loop. Built over the data
    /// `pool` by [`Self::connect`]; cheap to clone, so every clone shares one
    /// signal.
    store_oom: crate::storage::store_oom::StoreOomSignal,
    /// Tracker for `WatchGuard::Drop` spawned UNWATCH cleanup tasks
    /// (b.k1r H-stor-2). A `tokio_util::task::TaskTracker` gives us two things
    /// `tokio::spawn` alone does not:
    ///
    /// - [`TaskTracker::wait`] drained before [`ValkeyClient::quit`] returns,
    ///   so process shutdown does not tear the runtime down mid-`UNWATCH`.
    ///   Without this, an early-`?` return inside `update_layer_once_watch` at
    ///   shutdown leaves a stale WATCH on the txn-pool client, which is then
    ///   `quit()`ed before the spawned UNWATCH had a chance to reach Valkey.
    /// - A snapshot of in-flight cleanup tasks via [`TaskTracker::len`] that
    ///   powers the `live_edit_storage_watch_unwatch_inflight` gauge —
    ///   operators can see exactly how many UNWATCHes are outstanding.
    ///
    /// Wrapped in `Arc` so clones of `ValkeyClient` observe the same tracker
    /// (cleanup tasks spawned from one clone must be waited on by any other
    /// clone's `quit()`).
    pub(crate) watch_unwatch_tracker: Arc<TaskTracker>,
    /// Configuration + claims-read handle used to explain a server-side
    /// `WRONGPASS` / `NOAUTH` (b.cbe). Built in [`ValkeyClient::connect`] and
    /// held for the client's whole lifetime because the *steady-state* surface
    /// needs it: [`install_observability_hooks`](Self::install_observability_hooks)
    /// hands it to every pool's error hook, where a mid-life credential
    /// rejection — an access-policy assignment revoked or flipped to `Failed`
    /// under a running process — otherwise reconnect-loops forever with no
    /// operator-actionable signal.
    ///
    /// `Arc` so the up to ~392 hook tasks in this process (≈1,176 across the
    /// chart's 3 replicas, once every stream reader has connected) share one
    /// instance, and with it one rate-limit gate. `Option` because the
    /// diagnostic is a decoration on the error path: nothing in the connection
    /// logic may depend on it existing.
    auth_diagnostics: Option<Arc<ValkeyAuthDiagnostics>>,
    /// The reconnect + error metric-hook pair currently armed on each
    /// connection that is replaced at runtime — each `txn` slot (b.7td round 2),
    /// each stream reader's current connection, and each `data`, `maintenance`
    /// and `snapshot` member's current client (b.zt3) — keyed by fred's
    /// `ClientLike::id()`. A reader arms its connection's pair when it
    /// publishes the connection and disarms it when it retires it
    /// ([`PoolHooks`], shared with the readers' `ReaderConnector`); the member
    /// replacer arms a fresh client's pair and disarms the retired one's
    /// (`storage::member_replacement`).
    ///
    /// **This registry is what makes [`Self::rearm_pool_hooks`] idempotent, and
    /// it has to be.** A recycle is `quit()` then `connect()`, and `quit()` only
    /// swaps fred's public broadcast senders *after* its own internal await
    /// completes — so when the 2 s [`TXN_RECYCLE_QUIT_TIMEOUT`] fires (the
    /// EXPECTED path for a `probe_timeout`: a connection that would not answer
    /// a 650 ms `DISCARD` will not answer a `QUIT` either) the senders are never
    /// swapped, the old hook tasks never see `RecvError::Closed`, and they never
    /// finish. Re-arming blind would then leave two live hook pairs on one
    /// channel, permanently double-counting
    /// `valkey_errors_total{pool="txn"}` / `valkey_reconnects_total{pool="txn"}`
    /// and the b.cbe hint — and a `retain(!is_finished())` prune can never
    /// collect the orphans, because they never finish. Keyed by client id, the
    /// re-arm aborts the exact previous pair first, so it is idempotent whether
    /// or not `quit()` got as far as swapping the channels.
    ///
    /// Keyed by `ClientLike::id()` rather than by pool index because a checkout
    /// does not report which slot index it handed out (it scans and takes the
    /// first free mutex). The id is minted in `Client::new` and shared by every
    /// clone of a client, so the handle registered against the pool's client at
    /// install time is the same entry the recycle path finds through the
    /// acquired guard. A slot replacement (`storage::member_replacement`) puts
    /// a client with a new id in the slot: it arms the fresh id's pair and
    /// disarms the old one, so nothing stays under the old id.
    ///
    /// `quit()` drains and aborts the whole map alongside `background_tasks`.
    pool_hook_handles: HookRegistry,
    /// Tracker for spawned slot-recycle tasks (b.7td round 2).
    ///
    /// A recycle owns the slot's `OwnedMutexGuard<Client>` for its whole
    /// duration, so [`Self::quit`] drains this before closing the pools — same
    /// role `watch_unwatch_tracker` plays for `WatchGuard::Drop`, kept separate
    /// so the `live_edit_storage_watch_unwatch_inflight` gauge keeps meaning
    /// exactly what its name says.
    txn_recycle_tracker: Arc<TaskTracker>,
    /// Tracker for the cleanups a dropped snapshot-fence guard spawns
    /// (`storage::snapshot_fence`): a `FenceLease`'s release script and a
    /// `WriterIntent`'s `HDEL`, each spawned through
    /// [`Self::spawn_fence_guard_cleanup`]; and the WATCH path's
    /// `IntentRenewal` tasks, which clear their intent when they end
    /// ([`Self::track_fence_guard_task`]).
    ///
    /// [`Self::quit`] drains it before the pools close, so a fenced read that
    /// graceful shutdown drops — a catch-up flight's when its last waiter ends
    /// on the service shutdown token, a checkpoint's when the checkpoint loop
    /// is cancelled — gives its lease back to the store rather than leaving
    /// other replicas' writers of that layer paused until the lease lapses.
    /// Kept apart from `watch_unwatch_tracker` so the
    /// `live_edit_storage_watch_unwatch_inflight` gauge keeps meaning exactly
    /// what its name says.
    ///
    /// `Arc` so every clone's guards are waited on by any clone's `quit()`.
    fence_guard_cleanup_tracker: Arc<TaskTracker>,
    /// Consecutive *unknown-state* dirty verdicts (`probe_timeout` /
    /// `probe_error` / `topology_reply`) across the whole `txn` pool, reset by
    /// any `Clean` checkout (b.7td round 2; `topology_reply` joined the class
    /// in b.289 sub-task 3).
    ///
    /// Pool-wide, not per-slot, because the failure it rate-limits is
    /// pool-wide: a store-wide latency event times out the probe on every slot
    /// at once, and a per-slot streak would still admit `pool_size` recycles per
    /// window. Feeds
    /// [`should_recycle_unknown_state`](crate::storage::txn_checkout::should_recycle_unknown_state).
    ///
    /// `Arc` so every `ValkeyClient` clone shares one streak — the whole point
    /// is a single process-wide budget.
    txn_unknown_state_streak: Arc<AtomicU32>,
    /// Consecutive *state-proof* dirty verdicts (`leaked_multi` /
    /// `unexpected_reply`) **per cluster node**, each reset by a `Clean`
    /// checkout against that node (b.7td round 3; made per-node by b.289
    /// sub-task 4).
    ///
    /// Feeds
    /// [`state_proof_repair_is_ineffective`](crate::storage::txn_checkout::state_proof_repair_is_ineffective).
    ///
    /// **Why per-node and not pool-wide, unlike its unknown-state sibling.**
    /// The failure this bounds is a reply shape the *endpoint* produces and the
    /// classifier never accepts, so on a centralized or proxied deployment it
    /// really is a whole-pool property and one bucket is the whole story. On a
    /// real cluster it need not be: a single node mid-resharding, a single
    /// proxy shard re-rendering the `DISCARD` error, a single primary running a
    /// different server build. Pool-wide, ANY `Clean` checkout — including one
    /// pinned to a healthy node — resets the streak, so the breaker never
    /// latches and every checkout pinned to the affected node recycles forever:
    /// precisely the unbounded loop the breaker exists to bound, left unbounded
    /// in the partial-node case. Keyed by node, node A's streak survives node
    /// B's `Clean` and the breaker latches for A alone, leaving A's borrowers
    /// unrepaired (as designed) while B keeps its hygiene.
    ///
    /// Keys are `Server::to_string()` (`host:port`). The empty string is a
    /// shared fallback bucket for every checkout whose node cannot be named —
    /// a centralized or proxied deployment (no cached cluster state at all),
    /// a routing view that does not cover the slot, and the
    /// [`Self::acquire_txn`] shim, which has no key to hash.
    ///
    /// **The rule, exactly** (b.289 round 2 — round 1 shipped only the first
    /// half and the fallback bucket could never be cleared):
    ///
    /// 1. Bump and clear resolve the bucket through the same
    ///    [`Self::bucket_key`] helper — own bucket if it exists or the map has
    ///    room, else `""` — so a verdict and the `Clean` that should re-arm it
    ///    can never land in different entries.
    /// 2. ANY `Clean` checkout clears the `""` bucket, on top of clearing the
    ///    bucket its own node resolved to. That is what makes the fallback
    ///    degenerate to the pre-b.289 pool-wide behaviour, where any `Clean`
    ///    reset the streak: the verdicts in `""` have no node identity, so
    ///    there is no `Clean` "against" that bucket to pair them with, and
    ///    without this any transient unnameable window — fred's cached routing
    ///    view is `None` while it rebuilds, or does not cover the slot — could
    ///    latch the breaker permanently.
    /// 3. A cap-overflowed node therefore shares the fallback bucket's streak
    ///    *and* its re-arm, by rule 1 rather than as a special case.
    ///
    /// Bounded at [`TXN_STATE_PROOF_STREAK_MAX_NODES`] entries so a long series
    /// of reshardings cannot grow the map without limit. Stale entries for
    /// departed nodes are inert (nothing reads a key no checkout resolves to)
    /// and are bounded by the same cap.
    ///
    /// `std::sync::Mutex`, never held across an `.await` — every access is a
    /// lock, an integer edit, and an unlock inside one of the small helpers
    /// below.
    txn_state_proof_streaks: Arc<Mutex<HashMap<String, u32>>>,
    /// Test-only checkout-probe fault-injection seam (b.7td round 2). A
    /// zero-sized unit value in release builds; see
    /// [`ProbeFaults`](crate::storage::txn_checkout::ProbeFaults).
    probe_faults: ProbeFaults,
    /// Test-only fault-injection seam of
    /// [`read_object_snapshot`](crate::storage::diff_read::read_object_snapshot)
    /// and the fenced read's fan-out (Spec SDD SR-1.5). A zero-sized unit
    /// value in release builds; see
    /// [`ObjectSnapshotReadFaults`](crate::storage::diff_read::ObjectSnapshotReadFaults).
    ///
    /// Held on the client rather than on `LayerStore` because subscribe and
    /// catch-up, two of the consumers this seam covers, call
    /// `read_object_snapshot` directly and never go through a `LayerStore`.
    pub(crate) object_snapshot_read_faults: ObjectSnapshotReadFaults,
    /// The atomic snapshot read's item limit,
    /// `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS` (Spec SDD SR-1.9), copied
    /// from `Config` at connect so
    /// [`read_object_snapshot`](crate::storage::diff_read::read_object_snapshot)
    /// reads it beside the snapshot lane it runs on.
    pub(crate) snapshot_atomic_read_max_items: u64,
    /// The atomic snapshot read's byte limit,
    /// `LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES`; see
    /// [`Self::snapshot_atomic_read_max_items`].
    pub(crate) snapshot_atomic_read_max_bytes: u64,
    /// Test-only fault-injection seam of the attempt-mode release enqueue
    /// (b.yuo Part R): the retirement's attempts and the orphan sweep's
    /// re-drives. A zero-sized unit value in release builds; see
    /// [`ReleaseEnqueueFaults`](crate::storage::external_values::ReleaseEnqueueFaults).
    ///
    /// Held on the client for the same reason its snapshot-read sibling is: the
    /// enqueue that consults it is a free function reached from the retirement
    /// cleanup tasks and the orphan sweep's background task, which own no
    /// injectable handle of their own, and threading a parameter down to it
    /// would put a `#[cfg]` on their production signatures to serve one arm.
    pub(crate) release_enqueue_faults: ReleaseEnqueueFaults,
    /// Test-only seam holding back a `{gc}` read hold take for an armed object
    /// before it is sent (b.yuo; [`Self::arm_read_hold_take_hold_for_test`]).
    /// A zero-sized unit value in release builds; see
    /// [`ReadHoldTakeHolds`](crate::storage::external_values::ReadHoldTakeHolds).
    /// Held on the client because `ReadHold::take` is handed the client and
    /// nothing else injectable.
    pub(crate) read_hold_take_holds: ReadHoldTakeHolds,
    /// Test-only override of the orphan sweep's in-progress age X
    /// ([`RETIRE_INTENT_IN_PROGRESS_AGE`](crate::storage::orphan_sweep::RETIRE_INTENT_IN_PROGRESS_AGE);
    /// [`Self::set_retire_intent_in_progress_age_for_test`]). Shared behind an
    /// `Arc`, so it reaches every clone. Held on the client because the sweep
    /// phase that reads it is handed the client and nothing else injectable.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) retire_intent_in_progress_age_for_test: Arc<Mutex<Option<std::time::Duration>>>,
    /// Test-only override of the retirement intent's send window Y
    /// ([`RETIRE_INTENT_SEND_WINDOW`](crate::storage::external_values::RETIRE_INTENT_SEND_WINDOW);
    /// [`Self::set_retire_intent_send_window_for_test`]). Shared behind an
    /// `Arc`, so it reaches every clone. Held on the client for the same reason
    /// as the in-progress age beside it: the retirement and the sweep that read
    /// it are handed the client and nothing else injectable.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) retire_intent_send_window_for_test: Arc<Mutex<Option<std::time::Duration>>>,
    /// Test-only override of the orphan sweep's grace on unmapped objects and
    /// stranded overwrites
    /// ([`orphan_sweep_grace`](crate::storage::orphan_sweep::orphan_sweep_grace);
    /// [`Self::set_orphan_sweep_grace_for_test`]). Shared behind an `Arc`, so
    /// it reaches every clone. Held on the client for the same reason as the
    /// in-progress age beside it: it reaches the sweep, and the stores whose
    /// commits the sweep competes with, through the client they share.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) orphan_sweep_grace_for_test: Arc<Mutex<Option<std::time::Duration>>>,
    /// Test-only seam forcing the checkpoint assembler's read of an armed
    /// object to an unprovable cut, so a test can reach the checkpoint loop's
    /// tail-capture fallback. Absent from release builds, field and consumer
    /// alike; see
    /// [`UnprovableCutFaults`](crate::storage::diff_read::UnprovableCutFaults).
    ///
    /// Held on the client because the checkpoint loop it serves is handed a
    /// client and nothing else injectable: a parameter would put a `#[cfg]` on
    /// the loop's and the assembler's production signatures.
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    pub(crate) unprovable_cut_faults: UnprovableCutFaults,
    /// Test-only override of the `{gc}` read hold's TTL and renewal interval
    /// (b.yuo; [`Self::set_read_hold_timing_for_test`]). Shared behind an
    /// `Arc`, so it reaches every clone. Held on the client because the
    /// catch-up code that takes holds is handed the client and nothing else
    /// injectable; production builds always use
    /// [`ReadHoldTiming::PRODUCTION`](crate::storage::external_values::ReadHoldTiming::PRODUCTION).
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) read_hold_timing_for_test:
        Arc<Mutex<Option<crate::storage::external_values::ReadHoldTiming>>>,
    /// Test-only override of how the structural WATCH read issues its rounds
    /// 2 and 3 ([`Self::set_watch_read_transport_for_test`]): `None` (the
    /// production rule) issues each window as one window script but those
    /// `storage::diff_read`'s bounded fan-out names as plain, `Some(Plain)`
    /// issues every window as plain commands. Shared behind an `Arc`, so it
    /// reaches every clone. For the tests that compare the two transports and
    /// for the measurement tool's `watch-read --transport plain`; absent from
    /// release builds.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) watch_read_transport_for_test:
        Arc<Mutex<Option<crate::storage::diff_read::WindowTransport>>>,
    /// Test-only count of the watched transactions whose `EXEC` reply outlived
    /// their timeout on this client: every `WatchedExec::InDoubt` carrying
    /// fred's `Timeout` that `WatchedTxn::exec` / `exec_overridden` builds for
    /// an origin on this client ([`FredErrOrigin::client`]). A drop-time
    /// `UNWATCH` that times out is never counted here, though it is counted on
    /// `valkey_errors_total{pool="txn", kind="command_timeout"}` beside the
    /// `EXEC`, so a test can tell the two apart. Shared behind an `Arc`, so
    /// it reaches every clone. Read and reset with
    /// [`Self::watched_exec_timeouts_for_test`] and
    /// [`Self::take_watched_exec_timeouts_for_test`]; absent from release
    /// builds.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) watched_exec_timeouts_for_test: Arc<AtomicU64>,
    /// Test-only count of the `txn_pool` checkout routing syncs
    /// [`Self::sync_txn_checkout_routing`] has issued, on every clone; read by
    /// [`Self::txn_routing_syncs_for_test`]. An injected fault cannot
    /// reproduce a member's stale routing table, so this is how a test sees
    /// that a sync ran. Absent from release builds.
    #[cfg(any(test, feature = "test-fault-injection"))]
    txn_routing_syncs_for_test: Arc<std::sync::atomic::AtomicUsize>,
}

impl std::fmt::Debug for ValkeyClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Metrics` does not implement `Debug` (prometheus families are
        // opaque), so we elide the field. The other fields are already
        // `Debug`.
        f.debug_struct("ValkeyClient")
            .field("pool", &self.pool)
            .field("txn_pool", &self.txn_pool)
            .field("subscriber_admission", &self.subscriber_admission)
            .field("catchup_gate", &self.catchup_gate)
            .field("catchup_read_gate", &self.catchup_read_gate)
            .field("catchup_fenced_read_gate", &self.catchup_fenced_read_gate)
            .field("catchup_build_budget", &self.catchup_build_budget)
            .field("stream_mux", &self.stream_mux)
            .field("maintenance_pool", &self.maintenance_pool)
            .field("snapshot_pool", &self.snapshot_pool)
            .field("subscriber", &self.subscriber)
            .field("router_supervisor", &self.router_supervisor)
            .field("fanout_threads", &self.fanout_threads)
            .field("background_tasks", &"<opaque>")
            .field("metrics", &self.metrics.lock().ok().map(|g| g.is_some()))
            .field("store_oom", &self.store_oom)
            .field(
                "watch_unwatch_tracker_len",
                &self.watch_unwatch_tracker.len(),
            )
            // `ValkeyAuthDiagnostics` has its own redacting `Debug`.
            .field("auth_diagnostics", &self.auth_diagnostics)
            .field(
                "armed_hook_pairs",
                &self.pool_hook_handles.lock().map(|m| m.len()).ok(),
            )
            .field("txn_recycles_inflight", &self.txn_recycle_tracker.len())
            .field(
                "fence_guard_cleanups_inflight",
                &self.fence_guard_cleanup_tracker.len(),
            )
            .field(
                "txn_unknown_state_streak",
                &self.txn_unknown_state_streak.load(Ordering::Relaxed),
            )
            // Length, not contents: the map is keyed by `host:port` and a
            // `{:?}` of a `ValkeyClient` should not print backing-store
            // topology into a log line.
            .field(
                "txn_state_proof_streak_nodes",
                &self.txn_state_proof_streaks.lock().map(|m| m.len()).ok(),
            )
            .finish()
    }
}

impl ValkeyClient {
    /// Connect to Valkey, returning a ready `ValkeyClient`.
    ///
    /// Applies exponential backoff on connect failures:
    /// - Initial delay: 200 ms
    /// - Cap: 5 s
    /// - Total budget: `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` per phase, hard
    ///   (an attempt still running at the budget is cut off); five sequential
    ///   phases — data, txn, maintenance and snapshot pools, then the pub/sub
    ///   connection
    ///
    /// Those pools and the pub/sub connection are validated with PING before
    /// this function returns, proving every node and the credential. The
    /// stream readers are not connected here: each connects when it first
    /// serves a subscription (`ReaderConnector`), so a node that refuses
    /// reader connections surfaces at the first subscribe that needs it, not
    /// at startup.
    ///
    /// `redis_token_provider` is the Redis-scoped [`ServiceTokenProvider`] the
    /// startup-wiring layer builds for `auth_mode=accessTokenFile`; it is `None`
    /// in every other mode. `iam_credentials_provider` is the AWS credential source the
    /// startup-wiring layer resolves for `auth_mode=iam`; it is `None` in every
    /// other mode. Both are threaded straight into [`build_fred_config`] so the
    /// `credential_provider` is installed on the single `FredConfig` before the
    /// pool-builder clones — this is the *only* place either is consumed, keeping
    /// the single-mutation-site guarantee intact. That same seam also derives the
    /// pub/sub client's config ([`derive_pubsub_config`]); `connect` only
    /// consumes the two values it hands back.
    pub async fn connect(
        config: &Config,
        redis_token_provider: Option<Arc<ServiceTokenProvider>>,
        iam_credentials_provider: Option<SharedCredentialsProvider>,
    ) -> Result<Self, ConnectError> {
        // Auth diagnostics (b.cbe). Built HERE, before the seam below consumes
        // `redis_token_provider`, by cloning the `Arc` — a handle copy, never a
        // second construction of the shared token source, so the single
        // mutation-site guarantee this function is guarded on is untouched.
        // Nothing is evaluated yet: the claims are read only when a rejection is
        // actually observed. See `ValkeyAuthDiagnostics`.
        let auth_diagnostics = Arc::new(ValkeyAuthDiagnostics::new(
            config,
            redis_token_provider.as_ref().map(Arc::clone),
        ));
        // One flag for the whole `connect()` call, shared by all five retry
        // loops below, so a rejected credential produces exactly one hint per
        // invocation rather than one per pool per attempt. Deliberately owned
        // here rather than being a process-global `Once`: `connect` also runs
        // from `storage::test_support` and from three integration binaries that
        // share a process, where a global gate would be untestable.
        let auth_hint_emitted = AtomicBool::new(false);

        // Every config comes out of the one override seam. `fred_config` drives
        // the four pools; `pubsub_fred_config` drives the dedicated pub/sub
        // subscriber client and `reader_fred_config` the stream readers' template,
        // each differing from it in exactly one respect decided inside that seam —
        // see `derive_pubsub_config` and `derive_reader_config`. `reader_reauth`
        // is the provider the readers consult for their own re-AUTH.
        let FredConfigs {
            shared: fred_config,
            pubsub: pubsub_fred_config,
            pubsub_refresh_disabled,
            reader: reader_fred_config,
            reader_reauth,
        } = build_fred_config(config, redis_token_provider, iam_credentials_provider)?;

        let (tls, tls_host_mapping) = describe_tls_policy(&fred_config);
        tracing::info!(
            tls,
            tls_host_mapping,
            valkey_url = %redact_url_credentials(&config.valkey_url),
            "Valkey transport configured"
        );
        if pubsub_refresh_disabled {
            // One greppable boot line replacing ~288 WARNs/day/replica (b.j64).
            // The RESP2 pub/sub connection is permanently in subscribe context,
            // where Valkey refuses AUTH, so fred's periodic re-AUTH there could
            // only ever log a warning. That connection renews its credential on
            // the reconnect path instead. SDD §15.3.
            tracing::info!(
                pubsub_periodic_reauth = "disabled",
                pubsub_credential_renewal = "reconnect",
                "Periodic Valkey credential re-AUTH disabled on the pub/sub connection \
                 (RESP2 subscribe context refuses AUTH); it renews via reconnect"
            );
        }
        if reader_reauth.is_some() {
            // One greppable boot line for the readers' divergence (b.19i):
            // fred schedules no periodic re-AUTH on any stream reader client;
            // each reader re-AUTHs its own connection between reads, waking an
            // in-flight XREAD first. SDD §15.3.
            tracing::info!(
                reader_periodic_reauth = "reader-owned",
                "Periodic Valkey credential re-AUTH on the stream readers is sent by each reader \
                 between its reads, not by fred's refresh task"
            );
        }

        let pool_size = effective_pool_size(config.valkey_pool_size);
        let subscriber_pool_size = effective_subscriber_pool_size(config.subscriber_pool_size);
        let maintenance_pool_size = effective_maintenance_pool_size(config.maintenance_pool_size);
        // Validated non-zero by `Config::from_env`; a hand-built `Config` with 0
        // fails `build_pool` below ("Pool cannot be empty").
        let snapshot_pool_size = config.snapshot_pool_size as usize;

        // The reconnect policy and per-command timeout every Builder carries
        // (see `shared_reconnect_policy` / `shared_performance_config`; the
        // readers' template takes the same two in `Self::reader_builder`).
        let reconnect_policy = shared_reconnect_policy(config);
        let performance_config = shared_performance_config(config);
        // The one registry of reply times every connection this client opens
        // stamps and the unresponsive check reads (see
        // `apply_connect_timeout`): handed to every `Builder` below, so each
        // pool, the pub/sub client and the readers share it. It lives in their
        // fred configs and is dropped with the last of them.
        let peer_replies = PeerReplies::new();

        // Per-attempt TCP/TLS connect timeout. fred's default is a
        // hardcoded 10 s on `ConnectionConfig::connection_timeout`, so any
        // operator who sets `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` below
        // 10 s would have been silently overridden — a real operator
        // footgun (b.1kf). Wiring our configured budget into fred makes the
        // env-var honor its docstring at both ends of the range. The same
        // `budget` value below caps the outer retry loop, so the
        // per-attempt timeout never exceeds the overall budget in practice.
        // Note: with the default config (per-attempt == outer budget), a
        // single slow handshake now consumes the full budget in one
        // attempt rather than splitting across retries as it did pre-fix
        // (when fred's hidden 10 s was smaller than the 30 s outer
        // budget). This is the correct trade-off: slow-handshake retries
        // against the same endpoint rarely succeed (the slow path is
        // typically a misconfigured DNS / firewall / TLS partner, not a
        // transient flake), and honoring the operator's stated budget
        // verbatim is the more honest semantic. Fast-fail retries
        // (ECONNREFUSED) still work because `start.elapsed() < budget`
        // after a sub-budget failure.
        // See [`apply_connect_timeout`] — the shared seam called for the
        // data-pool `Builder`, the pub/sub subscriber `Builder` below and the
        // readers' template (`Self::reader_builder`).
        let mut pool_builder = Builder::from_config(fred_config.clone());
        pool_builder
            .set_policy(reconnect_policy.clone())
            .set_performance_config(performance_config.clone());
        apply_connect_timeout(
            &mut pool_builder,
            config,
            &peer_replies,
            UnresponsiveVerdicts::FastAndSlow,
        );
        // Connection buffer limits, general class (`storage::connection_buffers`).
        // Set before the txn clone and the maintenance and snapshot builds below,
        // so all four pools built from this builder carry them.
        connection_buffers::apply_limits(&mut pool_builder, connection_buffers::GENERAL_LIMITS);
        let pool = pool_builder
            .build_pool(pool_size)
            .map_err(|e| ConnectError::Build(e.to_string()))?;

        // Exclusive pool for WATCH/MULTI/EXEC transactions. Sized to the same
        // pool_size as the shared pool — each concurrent UpdateLayer takes one
        // exclusive client for its transaction window.
        //
        // ── The second deliberate per-pool config divergence (b.bhb) ─────────
        //
        // This pool alone runs at `max_command_attempts = 1` and
        // `max_redirections = 0`. Everything else is the shared data-pool
        // builder verbatim — it is a `clone()` of it, not a second
        // `Builder::from_config`, so credentials, credential provider, TLS
        // policy, reconnect policy and timeouts cannot drift from the data
        // pool's by construction.
        //
        // WHY. Valkey scopes a `WATCH` to one connection. fred's default
        // behaviour is to transparently re-issue a failed command on a fresh
        // connection (3 attempts) and to follow a `MOVED` to a fresh node (5
        // redirections) — and for a `Transaction` it replays the entire
        // `MULTI … EXEC` that way (`fred-10.1.0/src/router/transactions.rs`
        // `send`). Any such rewrite lands the commit on a socket that never
        // issued the `WATCH`, so Valkey commits it unconditionally: the
        // optimistic-concurrency contract of `LayerStore::update_layer_once_watch`
        // and `SessionStore::create_once` is silently void and a concurrent
        // writer's update is lost. The pipeline half of the fix lives in
        // `storage::watched_txn`; this config is what stops fred from rewriting
        // that pipeline.
        //
        // WHAT THIS BUYS, PRECISELY. Two things, both narrow:
        //   (a) a command whose socket write FAILS is never re-attempted on a
        //       new socket — at a budget of 1, `write_command`'s
        //       `decr_check_attempted` (`router/commands.rs:111`) leaves
        //       `attempts_remaining == 0`, so `finish_or_retry_command`
        //       (`router/commands.rs:91-97`) responds with the error instead of
        //       re-queueing, and every later frame of the same pipeline then
        //       fails to route;
        //   (b) fred's `Transaction` router never runs its explicit
        //       `MULTI … EXEC` replay loop, because the WATCHed commits no
        //       longer use a `Transaction` at all.
        // It does NOT stop a pipeline that has not yet been written from being
        // written to a REPLACEMENT socket: fred's router is one task, a
        // peer-side close enqueues a `RouterCommand::Reconnect`
        // (`router/utils.rs::process_response` → `defer_reconnection`) on the
        // same channel the pipeline is queued on, and if that message is ahead
        // of us the router reconnects first and the pipeline's write SUCCEEDS on
        // the new connection — no attempt is charged, so this cap never engages.
        // That can happen only while the pipeline's caller is still waiting:
        // once its timeout fires or its future is dropped, fred writes no frame
        // of it it has not already written (the vendored fred's change 9).
        // That residual is why `storage::watched_txn` also queues an
        // in-transaction `CLIENT ID` witness. That witness is NOT compared
        // against fred's handshake-time connection id: it is compared against a
        // baseline `CLIENT ID` sampled from inside a throwaway `MULTI … EXEC`
        // pipelined ahead of the `WATCH` on the same connection, so a proxied
        // managed store answering `CLIENT ID` from its own id space cannot turn
        // every ordinary commit into a false "proven unwatched". See
        // `storage::watched_txn`'s module docs.
        //
        // WHY 1 AND NOT 0. `router/commands.rs::write_command` calls
        // `Command::decr_check_attempted()` before writing ANY regular command
        // and errors at 0 (`protocol/command.rs:1776-1783`), so a pool at 0
        // would fail the `WATCH` itself, the reads under it, `QUIT`, and the
        // checkout hygiene probe — every ordinary command on the pool. 1 is the
        // smallest value that writes each command exactly once.
        //
        // WHY POOL-LEVEL AND NOT PER-COMMAND. `Command::inherit_options`
        // rewrites `attempts_remaining == 0` / `redirections_remaining == 0` to
        // the pool defaults (`protocol/command.rs:1882-1888`), so
        // `Options::max_redirections: Some(0)` is not expressible at the call
        // site. The zero has to be the pool default. It is load-bearing: a
        // `MOVED` on one member of a pipelined `MULTI` would otherwise be
        // re-sent ALONE to the new node as an unqueued, immediately-executed
        // write.
        //
        // COST. The non-WATCH `txn_pool` users (create_layer's mapping +
        // materialization, session delete, the counters primitives, the
        // checkpoint lane) lose fred's transparent per-command retry and
        // see a transient blip as an error instead. All of them either sit
        // inside a retry loop already, surface to a gRPC handler as
        // `Unavailable` (client retries) or re-run on the next
        // maintenance tick; `storage::counters` gained its own bounded retry for this
        // reason. Taking that cost on this pool — and NOT on the data pool,
        // whose transparent retry is load-bearing for every read — is the
        // deliberate trade.
        //
        // ONE EXCEPTION: a fred `Transaction` (`client.multi()` … `exec()`) is
        // still replayed WHOLE, once. Its router runs its own loop over the
        // whole `MULTI … EXEC` (`router/transactions.rs::send`), taking its
        // budget from the last command's `attempts_remaining` — this pool's 1 —
        // and failing only when `attempted > max_attempts`
        // (`transactions.rs:115,122-133`). A failed drain before it, a failed
        // write or a failed `EXEC` read (`transactions.rs:142,191,257`) —
        // including one whose bounded wait expired (*`UnresponsiveConfig`*,
        // at `apply_unresponsive_policy`) — therefore reconnects and
        // sends the whole transaction a second time, unless its caller has
        // given up by then: fred writes nothing of an attempt whose caller's
        // timeout fired or whose call was dropped (the vendored fred's change
        // 9), so the replay is written only inside the caller's own timeout. What
        // that one replay
        // does to each of the three `Transaction` users on this pool:
        //   - `SessionStore::delete` (`DEL` ×2 + `SET` of the deletion
        //     marker) is idempotent: a replay after a landed `EXEC` writes the
        //     same state again.
        //   - The session checkpoint applier's two transactions
        //     (`checkpoint::session::apply_session_checkpoint`: Step A's
        //     `SADD`s; Step B's `HSET` + layer-mapping `SET`s + `SADD`s) are
        //     idempotent against their OWN write, not against another
        //     writer's: Step B's mapping `SET`s are unconditional, so a replay
        //     after a landed `EXEC` re-applies them over any writer that
        //     landed in the reconnect window — a `DeleteLayer` removal, or a
        //     `CreateLayer` mapping flip once the session is visible — and
        //     that writer's change is overwritten. The same class as the
        //     unconditional-`SET` residual SDD §7.9 states for the create
        //     flow (*Residuals of the create flow*).
        //   - `LayerStore::write_object` (create materialization): when a
        //     landed `EXEC`'s reply is lost and the replay reaches the store,
        //     the replay re-applies the same state but its `XADD 1-0` is
        //     refused, the stream already holding `1-0`. That refusal is an
        //     `ERR …` element of the `EXEC` reply, which fred's `pretty_error`
        //     (`protocol/utils.rs:102-109`) maps to `ErrorKind::Unknown`;
        //     `write_object` recognises it by the refusal's text and answers
        //     `Unavailable` (an earlier send may have landed; no layer was
        //     made), before the general mapper, which would have called it
        //     `Internal`. When the replay is instead redirected (a slot that
        //     moved or is moving), fred fails it with "Max redirections
        //     exceeded", which the classifier grades transient
        //     (`is_transaction_router_exhausted_error`), so that reaches the
        //     client as `UNAVAILABLE` too. Either way the create fails before
        //     its mapping flip, and it is the lost reply, not the replay, that
        //     leaves the object it wrote: a complete, unmapped orphan for the
        //     orphan sweep. A re-send creates the layer under a new id.
        //
        // REDIRECTS. With the redirection budget at zero, fred's `Transaction`
        // router fails a transaction that meets a `MOVED` or an `ASK` at once
        // ("Max redirections exceeded", `transactions.rs:212-217`, `:240-243`)
        // and, unlike `process_moved` for a single command
        // (`router/commands.rs:298-307`), re-syncs nothing first. Each slot is
        // its own fred client with its own routing table, so a member that met
        // a `MOVED` would keep sending that slot's transactions to the old
        // owner. Every fred `Transaction` caller therefore sends its `EXEC`
        // through `ValkeyClient::exec_txn`, which re-syncs the transaction's own
        // checkout on that error before returning it to the caller.
        //
        // The one data-pool command that opts out per command is the
        // `UpdateLayer` commit, `EVALSHA update_layer_fast`, in both of the
        // script's modes (the field / time-sample fast path and the structural
        // commit, `storage::structural::commit_structural`): fred replays commands
        // still in flight when a connection drops, and a replay of a commit is
        // a second execution at an unbounded later time — possibly after the
        // same client's next commit, where no record check can recognise it.
        // It is therefore sent with `max_attempts = 1`
        // (`storage::lua::evalsha_once_with_reload`): written to the wire at
        // most once. A dropped connection fails that attempt transiently
        // instead, and so does a redirect it cannot follow — a `MOVED` as
        // `Unknown` "Too many failed write attempts.", an `-ASK` as `Routing`
        // "Max attempts reached." (the vendored fred patch keeps the router
        // running through the `-ASK`; `vendor/fred-10.1.0.SOURCE.md`).
        // `update_layer`'s own retry loop runs the next attempt, whose
        // commit-token check reports the commit if it did land. An
        // operation-id call's start-record `HGET` is sent once for the same
        // reason. "Once" bounds the writes; the caller's own timeout bounds
        // when the one write happens: a command still queued in fred's channel
        // while its router reconnects (or while the router supervisor restarts
        // a router that ended with an error) is written when the router next
        // reads the channel only if its caller is still waiting. Once the
        // caller gives up — its timeout fires or its future is dropped — fred
        // never writes it (the vendored fred's change 9). A router that
        // panicked or was aborted drops its channel; its queued commands fail
        // instead.
        let mut txn_pool_builder = pool_builder.clone();
        txn_pool_builder.with_connection_config(|connection| {
            connection.max_command_attempts = 1;
            connection.max_redirections = 0;
        });
        let txn_pool = txn_pool_builder
            .build_exclusive_pool(pool_size)
            .map_err(|e| ConnectError::Build(e.to_string()))?;

        // The stream-multiplexer readers' connections (`storage::stream_mux`),
        // one per reader, each running one multi-key `XREAD BLOCK` at a time.
        // Subscriptions share readers and hold no connection. Kept separate
        // from `txn_pool` and the data pool so a read parked in `XREAD BLOCK`
        // never sits in front of a writer's command. Nothing is connected
        // here: the `ReaderConnector` built below from this template connects
        // a reader the first time it serves a subscription — on a cluster, to
        // its bound slot's primary alone, the third sanctioned divergence (see
        // `ReaderConnector`).
        //
        // The template is `Self::reader_builder`'s, which makes the readers'
        // one per-pool config divergence (`Blocking::Interrupt`; read its docs
        // before adding a second). The readers build from `reader_fred_config`,
        // derived from the shared config inside `build_fred_config` (its
        // provider cadence-wrapped), and MOVED there; `fred_config` is dropped
        // right after its last read, so "nothing reads the shared config after
        // the readers' divergence" stays a compile-time fact rather than a
        // reading exercise.
        let cluster = matches!(fred_config.server, ServerConfig::Clustered { .. });
        drop(fred_config);
        let reader_builder = Self::reader_builder(config, reader_fred_config, &peer_replies);

        // Dedicated multiplexed pool (small, default size 1) for the GC +
        // orphan sweep tasks. Multiplexed (not exclusive) is fine because
        // the leader-election lock confines concurrent sweep activity to
        // one tick at a time per replica; the only thing that matters is
        // that this pool's connections are NEVER shared with the data
        // pool, so a SCAN page does not queue behind a writer's HSET.
        let maintenance_pool = pool_builder
            .build_pool(maintenance_pool_size)
            .map_err(|e| ConnectError::Build(e.to_string()))?;

        // The snapshot lane (Spec SDD SR-10.1): a dedicated multiplexed pool for
        // object-snapshot reads, built from the data-pool builder like
        // `maintenance_pool` — the data pool's credentials, credential provider,
        // TLS policy, reconnect policy, timeouts and command-retry defaults, with
        // no divergence. Its own connections are the point: a multi-MiB snapshot
        // reply never sits in front of a writer's command on the data pool.
        let snapshot_pool = pool_builder
            .build_pool(snapshot_pool_size)
            .map_err(|e| ConnectError::Build(e.to_string()))?;

        // Dedicated pub/sub `SubscriberClient`. It is the ONE connection that
        // builds from `pubsub_fred_config` rather than the shared config —
        // identical in every field except the re-AUTH cadence, and that
        // difference was decided inside `build_fred_config` (see
        // `derive_pubsub_config`), not here. `connect` performs no mutation of
        // its own on this value.
        //
        // Its performance config differs from the pools' in one field: the
        // broadcast channel capacity, sized for pub/sub bursts on this, the one
        // client that subscribes (see `PUBSUB_BROADCAST_CHANNEL_CAPACITY`).
        let mut subscriber_builder = Builder::from_config(pubsub_fred_config);
        subscriber_builder
            .set_policy(reconnect_policy)
            .set_performance_config(PerformanceConfig {
                broadcast_channel_capacity: PUBSUB_BROADCAST_CHANNEL_CAPACITY,
                ..performance_config
            });
        apply_connect_timeout(
            &mut subscriber_builder,
            config,
            &peer_replies,
            UnresponsiveVerdicts::FastAndSlow,
        );
        // Connection buffer limits, general class (`storage::connection_buffers`).
        connection_buffers::apply_limits(
            &mut subscriber_builder,
            connection_buffers::GENERAL_LIMITS,
        );
        let subscriber = subscriber_builder
            .build_subscriber_client()
            .map_err(|e| ConnectError::Build(e.to_string()))?;

        // Outer retry-loop budget. Held identically to the per-attempt
        // `connection_timeout` (both derived from
        // `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`) so the per-attempt
        // connect timeout never exceeds the outer retry budget — i.e. a
        // single fred handshake can never silently consume more wall
        // time than the operator allotted to the whole connect loop.
        let budget = Duration::from_secs(config.valkey_connect_timeout_secs);

        // Auth-diagnostic arguments shared by all five retry loops. Bound to
        // short names so the data-pool retry call below stays on ONE line:
        // `connect_fn_body()` (the source-inspection guard on this function)
        // ends its slice at that call's opening text, and a rustfmt-wrapped
        // multi-line form would break the anchor and fail the guard loudly.
        // (This comment deliberately does not spell the anchor out — an earlier
        // occurrence of the literal would move the slice's end.)
        let diag = Some(&*auth_diagnostics);
        let hinted = &auth_hint_emitted;

        // Every client below is started one member at a time
        // (`MemberRouters::init`, the same `Client::init()` fred's `Pool::init`
        // runs per member), so each member's router `ConnectHandle` is kept for
        // the router supervisor instead of being folded into one pool-wide
        // handle and discarded. A retry re-initialises only the members that
        // have not connected. The phases run one after another, each under its
        // own hard `budget`, so a degraded connect costs at most five budgets.
        // The stream readers are not among them: they connect lazily. The
        // `txn` slots are reached through the raw `ExclusivePool` here, before
        // `ExclusiveCheckout` wraps it, while nothing else can hold a slot.
        // The multiplexed pools' members, as fred's `Pool::clients()` snapshots
        // them (no member is replaced before `connect` returns).
        let data_clients = pool.clients();
        let maintenance_clients = maintenance_pool.clients();
        let snapshot_clients = snapshot_pool.clients();
        let data_routers = MemberRouters::new(data_clients.len());
        let mut txn_clients: Vec<Client> = Vec::with_capacity(txn_pool.size());
        for slot in txn_pool.clients() {
            txn_clients.push(slot.lock().await.clone());
        }
        let txn_routers = MemberRouters::new(txn_clients.len());
        let maintenance_routers = MemberRouters::new(maintenance_clients.len());
        // The snapshot fan-out's threads (`storage::fanout_thread`), built
        // before the snapshot lane connects: its members' routers start
        // on `snapshot-io`, so the reply frames the fan-out reads are
        // allocated from that thread's one arena.
        let fanout_threads =
            Arc::new(FanoutThreads::new().map_err(|e| ConnectError::Build(e.to_string()))?);
        let snapshot_routers =
            MemberRouters::on(snapshot_clients.len(), fanout_threads.snapshot_io().clone());
        let pubsub_routers = MemberRouters::new(1);

        // Connect data pool with retry.
        connect_with_retry("data pool", budget, diag, hinted, || {
            data_routers.init(&data_clients)
        })
        .await?;

        // Connect transaction (exclusive) pool with retry.
        connect_with_retry("txn pool", budget, diag, hinted, || {
            txn_routers.init(&txn_clients)
        })
        .await?;

        // Connect maintenance pool with retry.
        connect_with_retry("maintenance pool", budget, diag, hinted, || {
            maintenance_routers.init(&maintenance_clients)
        })
        .await?;

        // Connect the snapshot lane with retry.
        connect_with_retry("snapshot pool", budget, diag, hinted, || {
            snapshot_routers.init(&snapshot_clients)
        })
        .await?;

        // Connect subscriber with retry.
        connect_with_retry("subscriber", budget, diag, hinted, || {
            pubsub_routers.init(std::slice::from_ref(&subscriber))
        })
        .await?;

        // Validate data pool.
        pool.ping::<()>(None)
            .await
            .map_err(|e| ConnectError::Ping {
                connection: "data pool",
                source: e,
            })?;

        // Wrap the txn pool, the only exclusive pool, in the service-owned
        // next-free checkout (b.f4f) BEFORE anything takes a slot, so the
        // "every acquisition goes through the checkout" property that the
        // permit arithmetic rests on holds from the first PING onward. The
        // metrics slot is created here rather than in the struct literal below
        // because the checkout, the `SubscribeReadLayer` gates, the catch-up
        // memory budget, the router supervisor and the stream multiplexer (its
        // readers' connector included) share it with the `ValkeyClient` —
        // `install_observability_hooks` fills it in once and all of them
        // observe the same handle.
        let metrics_slot: Arc<Mutex<Option<SharedMetrics>>> = Arc::new(Mutex::new(None));
        let txn_pool = Arc::new(ExclusiveCheckout::new(
            txn_pool,
            POOL_LABEL_TXN,
            metrics_slot.clone(),
        ));
        // 0 = unlimited for both caps. `Config::from_env` never gives either 0
        // (the subscription ceiling is derived from the pod memory limit, and a
        // zero catch-up slot count is refused); a `Config` built field by field
        // can.
        let subscription_cap = (config.subscriber_max_subscriptions > 0)
            .then_some(config.subscriber_max_subscriptions);
        let subscriber_admission = Arc::new(SubscriberAdmissionGate::new(
            subscription_cap,
            AdmissionPolicy {
                wait: Duration::from_millis(config.subscriber_admission_wait_ms),
                // Default: the cap itself; unbounded (never reached) without one.
                max_waiters: config
                    .subscriber_admission_max_waiters
                    .unwrap_or(subscription_cap.unwrap_or(usize::MAX)),
            },
            metrics_slot.clone(),
        ));
        let catchup_gate = Arc::new(CatchUpGate::new(
            (config.subscriber_max_concurrent_catchups > 0)
                .then_some(config.subscriber_max_concurrent_catchups),
            metrics_slot.clone(),
        ));
        let catchup_read_gate = Arc::new(CatchUpReadGate::new(
            // `Config::from_env` rejects 0; the gate holds at least one permit
            // whatever it is handed.
            usize::try_from(config.catchup_snapshot_read_concurrency).unwrap_or(usize::MAX),
            crate::observability::metrics::CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC,
            metrics_slot.clone(),
        ));
        let catchup_fenced_read_gate = Arc::new(CatchUpReadGate::new(
            // As above: `Config::from_env` rejects 0.
            usize::try_from(config.catchup_fenced_read_concurrency).unwrap_or(usize::MAX),
            crate::observability::metrics::CATCHUP_SNAPSHOT_READ_CLASS_FENCED,
            metrics_slot.clone(),
        ));
        // `T`, from the pod memory plan (`config::memory_plan`), in KiB.
        let catchup_build_budget = Arc::new(MemoryBudget::new(
            config.catchup_build_memory_budget_mib.saturating_mul(1024),
            metrics_slot.clone(),
        ));

        // Validate transaction pool by PINGing one acquired client.
        {
            let client = txn_pool.acquire().await;
            client
                .ping::<()>(None)
                .await
                .map_err(|e| ConnectError::Ping {
                    connection: "txn pool",
                    source: e,
                })?;
        }

        // Start the readers, none of them connected: each connects through the
        // connector when it first serves a subscription, so startup opens no
        // reader connection and proves none (a node refusing reader sockets
        // surfaces at the first subscribe: a WARN, the reader-connect metric
        // and a retryable UNAVAILABLE). The connector reaches the router
        // supervisor and the per-connection hook registry this client owns,
        // so they are created here, ahead of the struct literal below.
        let router_supervisor = Arc::new(RouterSupervisor::new(metrics_slot.clone()));
        let background_tasks: BackgroundTasks = Arc::new(Mutex::new(Some(Vec::new())));
        let pool_hook_handles: HookRegistry = Arc::new(Mutex::new(HashMap::new()));
        let reader_connector = Arc::new(ReaderConnector {
            builder: reader_builder,
            reauth: reader_reauth,
            cluster,
            connect_timeout: Duration::from_secs(config.valkey_connect_timeout_secs),
            command_timeout: Duration::from_millis(config.valkey_command_timeout_ms),
            routing: pool.clone(),
            routing_sync: RoutingSync::default(),
            supervisor: Arc::clone(&router_supervisor),
            hooks: PoolHooks {
                metrics: Arc::clone(&metrics_slot),
                background_tasks: Arc::clone(&background_tasks),
                registry: Arc::clone(&pool_hook_handles),
                diagnostics: Some(Arc::clone(&auth_diagnostics)),
            },
        });
        let stream_mux = Arc::new(StreamMux::start(
            reader_connector,
            pool.clone(),
            cluster,
            subscriber_pool_size,
            StreamMuxKnobs::new(),
            metrics_slot.clone(),
        ));
        // The member replacer over the three multiplexed pools and the `txn`
        // slots (`storage::member_replacement`), sharing the router supervisor
        // and the per-connection hook registry; `start_router_supervision`
        // hands it each multiplexed member's watch and the `txn` slots' router
        // restart, `install_observability_hooks` arms the members' hooks
        // through it.
        let member_replacer = MemberReplacer::new(
            pool.clone(),
            maintenance_pool.clone(),
            snapshot_pool.clone(),
            Arc::clone(&txn_pool),
            txn_clients.clone(),
            Arc::clone(&router_supervisor),
            PoolHooks {
                metrics: Arc::clone(&metrics_slot),
                background_tasks: Arc::clone(&background_tasks),
                registry: Arc::clone(&pool_hook_handles),
                diagnostics: Some(Arc::clone(&auth_diagnostics)),
            },
            Duration::from_secs(config.valkey_connect_timeout_secs),
            Duration::from_millis(config.valkey_command_timeout_ms),
            Some(fanout_threads.snapshot_io().clone()),
        );

        // Validate maintenance pool.
        maintenance_pool
            .ping::<()>(None)
            .await
            .map_err(|e| ConnectError::Ping {
                connection: "maintenance pool",
                source: e,
            })?;

        // Validate every snapshot-lane connection, not one: a snapshot read
        // goes to whichever connection the round-robin
        // picks, so a single connection that cannot serve would fail a share of
        // reads.
        for client in &snapshot_clients {
            client
                .ping::<()>(None)
                .await
                .map_err(|e| ConnectError::Ping {
                    connection: "snapshot pool",
                    source: e,
                })?;
        }

        // Validate subscriber.
        subscriber
            .ping::<()>(None)
            .await
            .map_err(|e| ConnectError::Ping {
                connection: "subscriber",
                source: e,
            })?;

        // b.3z7 C-1: Spawn fred's internal task that re-subscribes to tracked
        // pub/sub channels on every reconnect. Without this, a Valkey failover
        // or brief network blip silently drops the presence-events and
        // session-deleted subscriptions on this replica — cross-replica
        // PeerJoined/PeerLeft + DeleteSession fan-out then breaks until the
        // pod restarts. The handle is stashed on the client so `quit()` can
        // abort the task instead of letting it run until the broadcast
        // channel closes on its own.
        let manage_subscriptions_handle = subscriber.manage_subscriptions();
        if let Ok(mut tasks) = background_tasks.lock() {
            if let Some(tasks) = tasks.as_mut() {
                tasks.push(manage_subscriptions_handle);
            }
        }

        tracing::info!(
            pool_size,
            subscriber_pool_size,
            maintenance_pool_size,
            snapshot_pool_size,
            valkey_url = %redact_url_credentials(&config.valkey_url),
            "Valkey connected"
        );

        // Every connection above AUTHed successfully, so under accessTokenFile
        // this records the identity verdict for a credential the backing store
        // demonstrably accepted (b.cbe). No-op under static / iam.
        auth_diagnostics.log_healthy_connect();

        // The returned client is the first owner; `start_router_supervision`
        // below hands the restarts it builds internal clones only.
        let owners = Arc::new(ClientOwners {
            member_replacer: Arc::clone(&member_replacer),
            router_supervisor: Arc::clone(&router_supervisor),
            fanout_threads: Arc::clone(&fanout_threads),
        });
        let store_oom = crate::storage::store_oom::StoreOomSignal::new(pool.clone());
        let client = ValkeyClient {
            pool,
            txn_pool,
            subscriber_admission,
            catchup_gate,
            catchup_read_gate,
            catchup_fenced_read_gate,
            catchup_build_budget,
            stream_mux,
            maintenance_pool,
            snapshot_pool,
            subscriber,
            router_supervisor,
            owners: Some(owners),
            member_replacer,
            fanout_threads,
            background_tasks,
            metrics: metrics_slot,
            store_oom,
            watch_unwatch_tracker: Arc::new(TaskTracker::new()),
            auth_diagnostics: Some(auth_diagnostics),
            pool_hook_handles,
            txn_recycle_tracker: Arc::new(TaskTracker::new()),
            fence_guard_cleanup_tracker: Arc::new(TaskTracker::new()),
            txn_unknown_state_streak: Arc::new(AtomicU32::new(0)),
            txn_state_proof_streaks: Arc::new(Mutex::new(HashMap::new())),
            probe_faults: ProbeFaults::none(),
            object_snapshot_read_faults: ObjectSnapshotReadFaults::none(),
            snapshot_atomic_read_max_items: config.snapshot_atomic_read_max_items,
            snapshot_atomic_read_max_bytes: config.snapshot_atomic_read_max_bytes,
            release_enqueue_faults: ReleaseEnqueueFaults::none(),
            read_hold_take_holds: ReadHoldTakeHolds::none(),
            #[cfg(any(test, feature = "test-support"))]
            retire_intent_in_progress_age_for_test: Arc::new(Mutex::new(None)),
            #[cfg(any(test, feature = "test-support"))]
            retire_intent_send_window_for_test: Arc::new(Mutex::new(None)),
            #[cfg(any(test, feature = "test-support"))]
            orphan_sweep_grace_for_test: Arc::new(Mutex::new(None)),
            #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
            unprovable_cut_faults: UnprovableCutFaults::none(),
            #[cfg(any(test, feature = "test-support"))]
            read_hold_timing_for_test: Arc::new(Mutex::new(None)),
            #[cfg(any(test, feature = "test-support"))]
            watch_read_transport_for_test: Arc::new(Mutex::new(None)),
            #[cfg(any(test, feature = "test-support"))]
            watched_exec_timeouts_for_test: Arc::new(AtomicU64::new(0)),
            #[cfg(any(test, feature = "test-fault-injection"))]
            txn_routing_syncs_for_test: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        };
        client.start_router_supervision(SupervisedRouters {
            data: data_routers.into_handles(),
            txn: txn_clients
                .into_iter()
                .zip(txn_routers.into_handles())
                .collect(),
            maintenance: maintenance_routers.into_handles(),
            snapshot: snapshot_routers.into_handles(),
            pubsub: pubsub_routers.into_handles(),
        });
        Ok(client)
    }

    /// The stream readers' client template (`ReaderConnector`'s `builder`),
    /// built from `reader_fred_config` — the readers' config
    /// `build_fred_config` returned, taken whole — with the shared reconnect
    /// policy ([`shared_reconnect_policy`]), the shared performance config
    /// ([`shared_performance_config`]) and the shared connect timeout
    /// ([`apply_connect_timeout`]) every other `Builder` carries, with the
    /// shared `peer_replies` registry from `connect` but the slow unresponsive
    /// verdict only ([`UnresponsiveVerdicts::SlowOnly`]), the reader class of
    /// connection buffer limits ([`connection_buffers::READER_LIMITS`]), plus
    /// the readers' one `FredConfig` divergence below. Called once, by
    /// [`Self::connect`].
    ///
    /// ── The readers' deliberate per-pool config divergence (b.9uq round 4) ──
    ///
    /// `Config::blocking` is set to `Blocking::Interrupt` here and NOWHERE
    /// else. This is the only place `connect` (through this fn) mutates a
    /// `FredConfig` after `build_fred_config` returned it, so read the next
    /// three paragraphs before adding a second one. (`ReaderConnector::connect`
    /// later rewrites `server` and `database` on each connection's clone of
    /// this template — the per-connection divergence, documented and pinned
    /// with `ReaderConnector`, not a per-pool one. The pub/sub client's and
    /// the readers' re-AUTH cadence divergences are NOT a second one either:
    /// both are derived inside `build_fred_config` itself, before `connect`
    /// sees any value — see `derive_pubsub_config` and `derive_reader_config`.)
    ///
    /// WHY the readers need it. Each reader client is parked in `XREAD
    /// BLOCK`, and a reader sends its own other commands — its probes, head
    /// checks, keepalive and credential re-AUTH — only between its reads.
    /// fred's periodic credential refresh no longer reaches a reader at all
    /// (b.19i: the reader config's provider is cadence-wrapped, so fred
    /// schedules no refresh there, and the reader wakes an in-flight read and
    /// re-AUTHs between reads). What is left is the residual case: the next
    /// command after a client-side XREAD timeout that did not force a
    /// reconnect, while the old read is still blocked on the server, would
    /// queue server-side behind up to 30 s of that BLOCK.
    /// `Blocking::Interrupt` makes fred send `CLIENT UNBLOCK <id> ERROR` (via
    /// its lazily opened backchannel connection) first. Every reader client is
    /// a single-node client, on a cluster too, so this is fred's non-cluster
    /// interrupt path for every deployment. The read it interrupts is the
    /// reader's own abandoned one: a client-side timeout only marks fred's
    /// command `timed_out`, so it stays first in the connection's buffer and
    /// its `-UNBLOCKED` reply goes to that command's dropped caller. The reader
    /// never sees it — nothing is re-planned and nothing is counted; the case
    /// costs the backchannel connection only. (A `-UNBLOCKED` that does reach
    /// a reader's live read comes from outside the service; the reader
    /// re-plans it cheaply from its durable per-subscription cursors —
    /// `is_client_unblocked_error` — so nothing is lost.) Safe because a
    /// reader issues its XREADs one at a time and nothing else while one is in
    /// flight.
    ///
    /// WHY it must never diverge for auth / TLS. `apply_fred_config_overrides`
    /// is the single mutation point for `FredConfig` (see
    /// `build_fred_config`) so that all six connection sets provably agree on
    /// credentials, credential provider, and TLS policy — a divergence there
    /// is a security property silently lost on one pool, and it would not
    /// fail any test that does not specifically look for it. `blocking` is
    /// categorically different: it is a client-side command-scheduling policy,
    /// invisible to the server's authn/authz, and it is meaningful only on
    /// clients that issue blocking commands — the readers and no other.
    /// Diverging it here therefore cannot weaken the shared-policy guarantee,
    /// and enabling it globally would be actively wrong: the data, txn,
    /// maintenance, snapshot and pub/sub pools are multiplexed, so a command
    /// queued behind another there is normal pipelining, not a stale block.
    ///
    /// Nothing is connected here: each reader connection is built from a
    /// clone of this template when the reader first serves a subscription
    /// (`ReaderConnector::connect`).
    fn reader_builder(
        config: &Config,
        reader_fred_config: FredConfig,
        peer_replies: &PeerReplies,
    ) -> Builder {
        let mut subscriber_fred_config = reader_fred_config;
        subscriber_fred_config.blocking = Blocking::Interrupt;
        let mut reader_builder = Builder::from_config(subscriber_fred_config);
        reader_builder
            .set_policy(shared_reconnect_policy(config))
            .set_performance_config(shared_performance_config(config));
        apply_connect_timeout(
            &mut reader_builder,
            config,
            peer_replies,
            UnresponsiveVerdicts::SlowOnly,
        );
        // Connection buffer limits, reader class
        // (`storage::connection_buffers::READER_LIMITS`): what each reader
        // connection keeps once its traffic drains. A `ConnectionConfig`
        // setting like the connect timeout above, not a `FredConfig`
        // divergence in the sense of the docs above: it changes how much a
        // connection keeps, never what it sends or how it authenticates.
        connection_buffers::apply_limits(&mut reader_builder, connection_buffers::READER_LIMITS);
        reader_builder
    }

    /// Hand every router `connect` started to the router supervisor: one
    /// watcher per member of each pool and one for the pub/sub `subscriber`.
    /// (Each stream reader's router is registered by its `ReaderConnector`
    /// when the reader first connects, and retired with its connection.)
    ///
    /// A `data`, `maintenance` or `snapshot` member is watched as a retirable
    /// watch, handed to the member replacer, which retires it when it replaces
    /// the member's client (`storage::member_replacement`). A multiplexed
    /// member is restarted with `connect()` on a clone of its client (a clone
    /// shares the client's state, so it is the same client); a `snapshot`
    /// member's on the `snapshot-io` thread, where it was started
    /// (`storage::fanout_thread`), so a restarted router keeps allocating its
    /// reply frames from that thread's arena rather than a service worker's.
    /// A `txn` slot is restarted by [`Self::restart_txn_router`], which needs
    /// this `ValkeyClient`; the watcher holds an internal clone of it
    /// ([`Self::internal_clone`]) until [`Self::quit`] cancels the watchers.
    /// The member replacer gets the same restart ([`Self::txn_router_restart`])
    /// for the fresh client a slot replacement watches under its own id, and
    /// holds it until `quit` shuts the replacer down. Either is released
    /// instead by the drop of the client's last owner ([`ClientOwners`]).
    fn start_router_supervision(&self, routers: SupervisedRouters) {
        let multiplexed: [(&'static str, Vec<Client>, Vec<ConnectHandle>); 3] = [
            (POOL_LABEL_DATA, self.pool.clients(), routers.data),
            (
                POOL_LABEL_MAINTENANCE,
                self.maintenance_pool.clients(),
                routers.maintenance,
            ),
            (
                POOL_LABEL_SNAPSHOT,
                self.snapshot_pool.clients(),
                routers.snapshot,
            ),
        ];
        for (pool_label, clients, handles) in multiplexed {
            let spawn_on = (pool_label == POOL_LABEL_SNAPSHOT)
                .then(|| self.fanout_threads.snapshot_io().clone());
            for (member, (client, handle)) in clients.into_iter().zip(handles).enumerate() {
                let client_id = ClientLike::id(&client).to_owned();
                let spawn_on = spawn_on.clone();
                let watch = self.router_supervisor.watch_retirable(
                    pool_label,
                    member,
                    &client_id,
                    handle,
                    RouterRestart::Connect(Box::new(move || {
                        connect_router_on(spawn_on.as_ref(), &client)
                    })),
                );
                self.member_replacer.adopt_watch(pool_label, member, watch);
            }
        }
        for handle in routers.pubsub {
            let subscriber = self.subscriber.clone();
            self.router_supervisor.watch(
                POOL_LABEL_PUBSUB,
                0,
                handle,
                RouterRestart::Connect(Box::new(move || ClientLike::connect(&subscriber))),
                None,
            );
        }
        for (member, (client, handle)) in routers.txn.into_iter().enumerate() {
            let client_id = ClientLike::id(&client).to_owned();
            self.router_supervisor.watch(
                POOL_LABEL_TXN,
                member,
                handle,
                self.txn_router_restart(client_id.clone()),
                Some(client_id),
            );
        }
        // The same restart for the fresh client a slot replacement swaps in,
        // watched under its own id. An internal clone: the replacer is this
        // client's, so an owner here would keep the client alive for good.
        let valkey = self.internal_clone();
        self.member_replacer
            .set_txn_restart(Arc::new(move |client_id| {
                valkey.txn_router_restart(client_id)
            }));
    }

    /// How the router of the `txn` slot whose client id is `client_id` is
    /// restarted: [`Self::restart_txn_router`], under the checkout. The watcher
    /// holds an internal clone of this `ValkeyClient` ([`Self::internal_clone`])
    /// until [`Self::quit`], or the drop of the client's last owner, cancels
    /// it.
    fn txn_router_restart(&self, client_id: String) -> RouterRestart {
        let valkey = self.internal_clone();
        RouterRestart::Repair(Box::new(move || {
            let valkey = valkey.clone();
            let client_id = client_id.clone();
            Box::pin(async move { valkey.restart_txn_router(&client_id).await })
        }))
    }

    /// Restart the ended router task of the `txn` slot whose fred client id is
    /// `client_id`, for the router supervisor.
    ///
    /// Under the checkout's own rules: a permit, then that slot's guard
    /// (`ExclusiveCheckout::checkout_slot_for_router_restart`), so no borrower
    /// holds the slot while it is rebuilt and no other slot is touched. Then
    /// the same repair a dirty checkout gets ([`Self::repair_held_txn_slot`]):
    /// `quit()` sends nothing on a client whose router has ended (it returns
    /// at once, or fails on the closed channel of an aborted router), the
    /// router abort finds it already ended, `connect()` starts the new router,
    /// the slot's hooks are re-armed, and the new handle is handed over to the
    /// slot's watcher.
    ///
    /// Skipped, with a WARN, when the pool is shutting down or no slot holds
    /// `client_id` any more: a slot replacement (`storage::member_replacement`)
    /// swapped a fresh client in before or while this waited for the slot. The
    /// ended router belonged to the client that left; the old watch was ended
    /// with it (`RouterSupervisor::retire_slot`), so its watcher exits, and the
    /// fresh client's router is watched under its own id.
    async fn restart_txn_router(&self, client_id: &str) {
        let Some(checkout) = self
            .txn_pool
            .checkout_slot_for_router_restart(client_id)
            .await
        else {
            tracing::warn!(
                client_id,
                shutting_down = self.txn_pool.is_quitting(),
                "txn slot router restart skipped: the pool is shutting down, or no slot \
                 holds this client id any more (a slot replacement swapped a fresh client in \
                 before or while this restart waited for the slot, and the fresh client's \
                 router is watched under its own id)"
            );
            return;
        };
        self.repair_held_txn_slot(&checkout, TXN_RECYCLE_REASON_ROUTER_ENDED)
            .await;
    }

    /// End the current router task of member `member` of pool `pool` (a
    /// `POOL_LABEL_*` value: `data`, `txn`, `maintenance`, `snapshot`,
    /// `subscriber` with member 0, or `subscriber_pool` with a reader index,
    /// for that reader's current client) by aborting it, as a panic in it
    /// would end it. Returns whether that member's router was found: a reader
    /// that has never connected, or whose last connection was retired, has
    /// none.
    ///
    /// **Test-only** (`test-support`). With the vendored fred patch no `-ASK`
    /// ends a router, so this is the way a test drives the router supervisor:
    /// the member's watcher sees the abort's `JoinError` and restarts it.
    #[cfg(any(test, feature = "test-support"))]
    pub fn end_router_for_test(&self, pool: &str, member: usize) -> bool {
        self.router_supervisor.end_router_for_test(pool, member)
    }

    /// The stream readers whose current client is under router supervision,
    /// each with the fred client id (`ClientLike::id`) its watch supervises,
    /// by reader index. A reader appears from its first successful connect,
    /// its entry follows each retarget to the new client, and it disappears
    /// when its connection is retired; [`Self::quit`] empties it. Compare an
    /// entry with `StreamMux::reader_client_for_test(reader)`'s id to prove
    /// the reader's current client is the supervised one.
    ///
    /// **Test-only** (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn reader_watches_for_test(&self) -> HashMap<usize, String> {
        self.router_supervisor
            .retirable_watches_for_test(POOL_LABEL_SUBSCRIBER)
    }

    /// The fred client ids (`ClientLike::id`) of every connection whose metric
    /// hook pair is registered for re-arming: the `data`, `maintenance` and
    /// `snapshot` members' current clients, the `txn` slots and the stream
    /// readers' current connections, once `install_observability_hooks` has
    /// run. A retired reader connection, and a member's client once a
    /// replacement has retired it, leave it. **Test-only** (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn armed_hook_client_ids_for_test(&self) -> Vec<String> {
        self.pool_hook_handles
            .lock()
            .map(|registry| registry.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Replace the member replacer's trigger, cooldown and retirement grace
    /// (`storage::member_replacement::ReplaceKnobs`; production:
    /// `ReplaceKnobs::production`) on this client and every clone. Takes effect
    /// from the next fast verdict. **Test-only** (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_member_replacement_knobs_for_test(
        &self,
        knobs: crate::storage::member_replacement::ReplaceKnobs,
    ) {
        self.member_replacer.set_knobs_for_test(knobs);
    }

    /// Count a fast unresponsive verdict on member `member` of pool `pool`
    /// (`data`, `maintenance`, `snapshot` or `txn`, a slot index for `txn`)
    /// against its current client, as that client's error hook does when fred
    /// closes one of its connections with a fast verdict: the member
    /// replacer's trigger, cooldown and budget apply as for a real one.
    /// **Test-only** (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn fast_verdict_for_test(&self, pool: &str, member: usize) {
        self.member_replacer.fast_verdict_for_test(pool, member);
    }

    /// The fred client id (`ClientLike::id`) each `txn` slot holds now, in
    /// slot order: a slot replacement changes its entry. **Test-only**
    /// (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn txn_slot_client_ids_for_test(&self) -> Vec<String> {
        self.txn_pool.slot_client_ids_for_test()
    }

    /// The fred client ids the router supervisor watches as `txn` slots: after
    /// a slot replacement, the fresh client's id and not the old one's.
    /// **Test-only** (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn txn_slot_watches_for_test(&self) -> Vec<String> {
        self.router_supervisor.slot_watch_ids_for_test()
    }

    /// Whether pool `pool`'s (`data`, `maintenance`, `snapshot` or `txn`)
    /// replacement slot is held: a member replacement is between its trigger
    /// and its swap. An old client's retirement, which runs after the swap,
    /// does not hold it ([`Self::member_replacement_tasks_for_test`] counts
    /// those). **Test-only** (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn member_replacement_in_flight_for_test(&self, pool: &str) -> bool {
        self.member_replacer.in_flight_for_test(pool)
    }

    /// The member-replacement tasks still running, across every pool:
    /// replacements before their swap and old clients' retirements after it.
    /// Zero once every replacement and retirement has finished. **Test-only**
    /// (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn member_replacement_tasks_for_test(&self) -> usize {
        self.member_replacer.tasks_for_test()
    }

    /// The old clients' retirements in progress, across every pool: at most
    /// `storage::member_replacement::MEMBER_RETIRING_CAP`. **Test-only**
    /// (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn member_retirements_for_test(&self) -> usize {
        self.member_replacer.retiring_for_test()
    }

    /// The fred client id each live retirable watch of pool `pool` supervises,
    /// by member: for `data`, `maintenance` and `snapshot`, each member's
    /// current client from startup on, following its replacements. A `txn`
    /// slot has no retirable watch ([`Self::txn_slot_watches_for_test`]).
    /// **Test-only** (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn member_watches_for_test(&self, pool: &str) -> HashMap<usize, String> {
        self.router_supervisor.retirable_watches_for_test(pool)
    }

    /// The number of router-watcher tasks still running: one per supervised
    /// member — each connected stream reader included — until [`Self::quit`],
    /// zero after it. **Test-only** (`test-support`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn router_watchers_for_test(&self) -> usize {
        self.router_supervisor.watchers()
    }

    /// Install the checkout-probe fault injector every `txn_pool` acquisition on
    /// this client (and on every clone of it, present or future) will consult.
    ///
    /// **Test-only** — compiled only under
    /// `#[cfg(any(test, feature = "test-fault-injection"))]`, exactly like
    /// `LayerStore`'s seams. The injector lives behind a shared `Arc`, so
    /// install order relative to building a `LayerStore` /
    /// `ValkeyExternalValueStore` on top of this client does not matter.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn install_probe_fault_injector_for_test(
        &self,
        injector: Arc<crate::storage::txn_checkout::ProbeFaultInjector>,
    ) {
        self.probe_faults.install(injector);
    }

    /// Install the fault injector every
    /// [`read_object_snapshot`](crate::storage::diff_read::read_object_snapshot)
    /// and fenced read on this client (and on every clone of it, present or
    /// future) will consult (Spec SDD SR-1.5): each queued error fails one
    /// read attempt, answering in place of the call its
    /// [`ObjectSnapshotReadFaultTarget`](crate::storage::diff_read::ObjectSnapshotReadFaultTarget)
    /// names — by default the atomic path's `EVALSHA` or, on either fan-out,
    /// the first command of the window that consults it (on a fresh
    /// read, the header `HGETALL`) — and is classified and counted as a real
    /// failure of that call would be.
    ///
    /// **Test-only** — compiled only under
    /// `#[cfg(any(test, feature = "test-fault-injection"))]`, exactly like
    /// [`Self::install_probe_fault_injector_for_test`]. The injector lives
    /// behind a shared `Arc`, so install order relative to building a
    /// `LayerStore` on top of this client does not matter.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn install_object_snapshot_read_fault_injector_for_test(
        &self,
        injector: Arc<crate::storage::diff_read::ObjectSnapshotReadFaultInjector>,
    ) {
        self.object_snapshot_read_faults.install(injector);
    }

    /// Install the attempt-mode release enqueue fault injector (b.yuo Part R)
    /// on this client and on every clone of it, present or future. Every
    /// attempt-mode enqueue — a retirement's attempt or the orphan sweep's
    /// re-drive — pops one queued answer: `Fail` (not sent) or
    /// `ExecuteThenFail` (sent, reply dropped). The commit paths' plain enqueue
    /// never consults it.
    ///
    /// **Test-only** — compiled only under
    /// `#[cfg(any(test, feature = "test-fault-injection"))]`, exactly like
    /// [`Self::install_object_snapshot_read_fault_injector_for_test`]. The
    /// injector lives behind a shared `Arc`, so install order relative to
    /// handing this client to a task does not matter.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn install_release_enqueue_fault_injector_for_test(
        &self,
        injector: Arc<crate::storage::external_values::ReleaseEnqueueFaultInjector>,
    ) {
        self.release_enqueue_faults.install(injector);
    }

    /// Arm a one-shot hold on the next `{gc}` read hold take for `object_id`
    /// made through this client or any clone of it (b.yuo; see
    /// [`ReadHoldTakeHold`](crate::storage::external_values::ReadHoldTakeHold)):
    /// that take parks before its `EVALSHA ext_read_hold` is sent, until the
    /// test releases the returned handle. Takes for other objects, and later
    /// takes for this one, are not held. Arming again for an object whose hold
    /// has not been taken yet replaces it.
    ///
    /// ```ignore
    /// let take = svc.valkey.arm_read_hold_take_hold_for_test(object_id);
    /// // ... drive the catch-up whose build misses ...
    /// take.parked().await;  // the take is parked; nothing is on {gc} yet
    /// // ... the work that must happen before the hold exists ...
    /// take.release();       // the take is sent and the catch-up goes on
    /// ```
    ///
    /// **Test-only** — compiled only under
    /// `#[cfg(any(test, feature = "test-fault-injection"))]`, like
    /// [`Self::install_release_enqueue_fault_injector_for_test`]. Shared behind
    /// an `Arc`, so arming after the client was handed to a service still
    /// reaches it.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn arm_read_hold_take_hold_for_test(
        &self,
        object_id: u64,
    ) -> Arc<crate::storage::external_values::ReadHoldTakeHold> {
        self.read_hold_take_holds.arm(object_id)
    }

    /// The orphan sweep's retirement-intent in-progress age X this client's
    /// sweeps use (b.yuo Part R):
    /// [`RETIRE_INTENT_IN_PROGRESS_AGE`](crate::storage::orphan_sweep::RETIRE_INTENT_IN_PROGRESS_AGE),
    /// or a test's override ([`Self::set_retire_intent_in_progress_age_for_test`]).
    pub(crate) fn retire_intent_in_progress_age(&self) -> std::time::Duration {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(age) = self
            .retire_intent_in_progress_age_for_test
            .lock()
            .ok()
            .and_then(|guard| *guard)
        {
            return age;
        }
        crate::storage::orphan_sweep::RETIRE_INTENT_IN_PROGRESS_AGE
    }

    /// **Test-only.** Make every orphan-sweep drain through this client (and
    /// every clone of it, present or future) treat a carrier whose newest
    /// attempt token is younger than `age` as still in progress, instead of
    /// the production 60 s (b.yuo Part R) — `Duration::ZERO` drains a carrier
    /// at once. Compiled only for tests and the `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_retire_intent_in_progress_age_for_test(&self, age: std::time::Duration) {
        if let Ok(mut guard) = self.retire_intent_in_progress_age_for_test.lock() {
            *guard = Some(age);
        }
    }

    /// The retirement intent's send window Y this client's retirements and
    /// orphan-sweep drains use (b.yuo Part R, F3):
    /// [`RETIRE_INTENT_SEND_WINDOW`](crate::storage::external_values::RETIRE_INTENT_SEND_WINDOW),
    /// or a test's override ([`Self::set_retire_intent_send_window_for_test`]).
    pub(crate) fn retire_intent_send_window(&self) -> std::time::Duration {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(window) = self
            .retire_intent_send_window_for_test
            .lock()
            .ok()
            .and_then(|guard| *guard)
        {
            return window;
        }
        crate::storage::external_values::RETIRE_INTENT_SEND_WINDOW
    }

    /// **Test-only.** Make every retirement and orphan-sweep drain through this
    /// client (and every clone of it, present or future) decide no
    /// attempt-mode enqueue later than `window` after recording its token,
    /// instead of the production 30 s (b.yuo Part R) — `Duration::ZERO` makes
    /// every such send answer `NotSent` without sending. Compiled only for
    /// tests and the `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_retire_intent_send_window_for_test(&self, window: std::time::Duration) {
        if let Ok(mut guard) = self.retire_intent_send_window_for_test.lock() {
            *guard = Some(window);
        }
    }

    /// The orphan sweep grace a test installed on this client
    /// ([`Self::set_orphan_sweep_grace_for_test`]), if any. Always `None`
    /// outside tests and the `test-support` feature; the configured grace is
    /// `LayerStore`'s
    /// ([`LayerStore::orphan_sweep_grace`](crate::storage::layer::LayerStore::orphan_sweep_grace)).
    pub(crate) fn orphan_sweep_grace_override(&self) -> Option<std::time::Duration> {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(grace) = self
            .orphan_sweep_grace_for_test
            .lock()
            .ok()
            .and_then(|guard| *guard)
        {
            return Some(grace);
        }
        None
    }

    /// **Test-only.** Make every orphan sweep, and every `CreateLayer`
    /// finisher, run through this client (and every clone of it, present or
    /// future) use `grace` instead of
    /// [`orphan_sweep_grace`](crate::storage::orphan_sweep::orphan_sweep_grace)
    /// of the RPC deadline (at least 60 s): `Duration::ZERO` lets one sweep
    /// pass veto and retire an unmapped object, or complete a stranded
    /// overwrite, the moment it is created. A finisher timing installed with
    /// `LayerStore::with_create_commit_finisher_timing_for_test` takes
    /// precedence for the finisher. Compiled only for tests and the
    /// `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_orphan_sweep_grace_for_test(&self, grace: std::time::Duration) {
        if let Ok(mut guard) = self.orphan_sweep_grace_for_test.lock() {
            *guard = Some(grace);
        }
    }

    /// The `{gc}` read hold's TTL and renewal interval this client's catch-ups
    /// use (b.yuo): [`ReadHoldTiming::PRODUCTION`](crate::storage::external_values::ReadHoldTiming::PRODUCTION),
    /// or a test's override ([`Self::set_read_hold_timing_for_test`]).
    pub(crate) fn read_hold_timing(&self) -> crate::storage::external_values::ReadHoldTiming {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(timing) = self
            .read_hold_timing_for_test
            .lock()
            .ok()
            .and_then(|guard| *guard)
        {
            return timing;
        }
        crate::storage::external_values::ReadHoldTiming::PRODUCTION
    }

    /// **Test-only.** Make every `{gc}` read hold taken through this client
    /// (and every clone of it, present or future) use `timing` instead of the
    /// production 15 s TTL and 5 s renewal (b.yuo), so a test can let a hold
    /// lapse, or outlive its TTL under renewal, in milliseconds. Compiled only
    /// for tests and the `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_read_hold_timing_for_test(
        &self,
        timing: crate::storage::external_values::ReadHoldTiming,
    ) {
        if let Ok(mut guard) = self.read_hold_timing_for_test.lock() {
            *guard = Some(timing);
        }
    }

    /// How a structural WATCH read on this client issues the windows of its
    /// rounds 2 and 3 (`storage::diff_read`, SDD §10.5):
    /// [`WindowTransport::Script`](crate::storage::diff_read::WindowTransport::Script),
    /// one window script per window, unless a test overrode it
    /// ([`Self::set_watch_read_transport_for_test`]). A lone children list
    /// over τ goes as one plain command, and a read that falls back on a
    /// cluster-state refusal issues its later windows plain, whatever this
    /// answers.
    pub(crate) fn watch_read_transport(&self) -> crate::storage::diff_read::WindowTransport {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(transport) = self
            .watch_read_transport_for_test
            .lock()
            .ok()
            .and_then(|guard| *guard)
        {
            return transport;
        }
        crate::storage::diff_read::WindowTransport::Script
    }

    /// **Test-only.** Make every structural WATCH read through this client
    /// (and every clone of it, present or future) issue the windows of its
    /// rounds 2 and 3 as `transport` says: `Plain` issues each as plain
    /// commands, the transport a read falls back to; `Script` is the
    /// production rule, under which a lone children list over τ still goes
    /// plain. For the tests that hold both transports to one
    /// projection and for the measurement tool's `watch-read --transport`.
    /// Compiled only for tests and the `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_watch_read_transport_for_test(
        &self,
        transport: crate::storage::diff_read::WindowTransport,
    ) {
        if let Ok(mut guard) = self.watch_read_transport_for_test.lock() {
            *guard = Some(transport);
        }
    }

    /// **Test-only.** How many watched transactions on this client (and every
    /// clone of it) had their `EXEC` reply outlive their timeout since the
    /// client connected or the count was last taken: the in-doubt timeouts of
    /// `WatchedTxn::exec` / `exec_overridden` alone, never a drop-time
    /// `UNWATCH`. Compiled only for tests and the `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub fn watched_exec_timeouts_for_test(&self) -> u64 {
        self.watched_exec_timeouts_for_test.load(Ordering::SeqCst)
    }

    /// **Test-only.** [`Self::watched_exec_timeouts_for_test`], resetting the
    /// count to zero in the same step, so a test can read one phase's count
    /// alone. Compiled only for tests and the `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub fn take_watched_exec_timeouts_for_test(&self) -> u64 {
        self.watched_exec_timeouts_for_test
            .swap(0, Ordering::SeqCst)
    }

    /// Arm the checkpoint-assembly seam for `object_id` on this client (and on
    /// every clone of it, present or future): until
    /// [`Self::disarm_unprovable_checkpoint_cut_for_test`], every checkpoint
    /// assembly of the object pins its header sequence at `u64::MAX`, above
    /// every cut, so its proven read — atomic or fenced, still run for real — is
    /// judged regressed and the assembly ends `SkippedNotQuiesced`, persisting
    /// nothing. The route by which a test reaches the checkpoint loop's
    /// tail-capture fallback, which since every read proves its cut nothing
    /// else but a real header-sequence regression reaches (Spec SDD SR-4.1).
    /// An absent header still reads as absent.
    ///
    /// **Test-only** — compiled only under
    /// `#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]`.
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    pub fn arm_unprovable_checkpoint_cut_for_test(&self, object_id: u64) {
        self.unprovable_cut_faults.arm(object_id);
    }

    /// Undo [`Self::arm_unprovable_checkpoint_cut_for_test`] for `object_id`:
    /// its next checkpoint assembly pins the real header sequence again.
    ///
    /// **Test-only**, under the same cfg.
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    pub fn disarm_unprovable_checkpoint_cut_for_test(&self, object_id: u64) {
        self.unprovable_cut_faults.disarm(object_id);
    }

    /// Current pool-wide consecutive unknown-state verdict streak, for tests
    /// asserting the recycle gate's suppression window.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn txn_unknown_state_streak_for_test(&self) -> u32 {
        self.txn_unknown_state_streak.load(Ordering::Relaxed)
    }

    /// The **longest** consecutive state-proof verdict streak currently held
    /// against any one node, for tests asserting the repair-ineffective breaker
    /// (b.7td round 3).
    ///
    /// The streak became per-node in b.289 sub-task 4, and this accessor keeps
    /// its pre-b.289 signature deliberately. `max` is the reading that
    /// preserves what every existing caller asserts: on a single-node,
    /// centralized or proxied deployment — which is every in-sandbox fixture —
    /// there is exactly one bucket (the shared `""` fallback), so the max IS
    /// that bucket and the old assertions hold verbatim. On a real cluster it
    /// answers "has ANY node's streak reached N", which is the question the
    /// breaker's own escalation asks. Use
    /// [`Self::txn_state_proof_streak_for_node_for_test`] when the test needs
    /// to tell two nodes apart.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn txn_state_proof_streak_for_test(&self) -> u32 {
        match self.txn_state_proof_streaks.lock() {
            Ok(streaks) => streaks.values().copied().max().unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// Consecutive state-proof verdict streak held against exactly one node
    /// (b.289 sub-task 4).
    ///
    /// `node` is a `Server::to_string()` rendering (`host:port`), or `""` for
    /// the shared fallback bucket every checkout whose node could not be named
    /// lands in — which is every checkout on a centralized or proxied
    /// deployment, and every checkout through the [`Self::acquire_txn`] shim.
    /// A node with no streak reads 0, the same as a node that has never been
    /// seen: absence and zero are the same state here, since a `Clean`
    /// checkout removes the entry.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn txn_state_proof_streak_for_node_for_test(&self, node: &str) -> u32 {
        match self.txn_state_proof_streaks.lock() {
            Ok(streaks) => streaks.get(node).copied().unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// Every node currently carrying a non-zero state-proof streak, sorted
    /// (b.289 sub-task 4).
    ///
    /// The cluster-gated test that proves node A latches while node B stays
    /// `Clean` needs to see the bucket keys, not just their values — an
    /// assertion on counts alone cannot tell "two nodes, one streak each" from
    /// "one node counted twice". `""` appears in the list when the shared
    /// fallback bucket is in use.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn txn_state_proof_streak_nodes_for_test(&self) -> Vec<String> {
        match self.txn_state_proof_streaks.lock() {
            Ok(streaks) => {
                let mut nodes: Vec<String> = streaks.keys().cloned().collect();
                nodes.sort_unstable();
                nodes
            }
            Err(_) => Vec::new(),
        }
    }

    /// Acquire a `txn_pool` slot **without** knowing which cluster slot the
    /// caller's transaction will run on.
    ///
    /// Prefer [`Self::acquire_txn_for`] in new code. This form still runs the
    /// b.7td checkout hygiene probe, but leaves fred's default routing in place
    /// for the two probe commands, which is exact on a centralized or
    /// single-endpoint proxied deployment and best-effort on a cluster (the
    /// probe lands on whichever primary the router picks, which need not be the
    /// one the caller's transaction will use).
    ///
    /// **No production call site remains** — b.4t3 migrated the last two
    /// (`src/checkpoint/session.rs`, DR-restore Steps A and B) to
    /// [`Self::acquire_txn_for`], so the earlier lane-boundary justification for
    /// keeping them here is obsolete. The form survives for the storage-lane
    /// checkout / watch-leak / metrics tests that take a raw `txn_pool` slot
    /// with no transaction of their own, and so no key to name. It does **not**
    /// survive for "some future caller that cannot know a key": the fence
    /// `no_unpinned_acquire_txn_call_site_survives_under_src` in
    /// `tests/integration/storage/txn_checkout_test.rs` is absolute over `src/`
    /// with no allow-list, so a production call site here would first require
    /// deliberately amending that test — which is the point, since a caller that
    /// *can* name a key and uses this anyway is exactly what the fence catches.
    ///
    /// b.f4f made the companion property — that no code outside
    /// `storage::checkout` can lock a pool mutex at all — **structural**: the
    /// `ExclusivePool` is private to that module and the only things it hands
    /// out are `Client` clones (for the startup metric hooks) and a `quit()`.
    /// The widened fence is the regression guard on top of that, not the proof.
    ///
    /// The returned [`CheckedOutClient`] carries the slot, the
    /// `valkey_pool_in_use{pool="txn"}` gauge token and the checkout's semaphore
    /// permit in one value whose drop order releases them in that sequence
    /// (b.f4f) — callers no longer have a scoping rule to get right.
    pub async fn acquire_txn(&self) -> CheckedOutClient {
        self.acquire_txn_inner(None).await
    }

    /// Acquire a `txn_pool` slot for a transaction that will touch `hash_key`,
    /// running the checkout hygiene probe pinned to that key's cluster slot
    /// (b.7td).
    ///
    /// `hash_key` is any one key the caller's transaction touches. Where a
    /// transaction spans several keys they must already share a hash tag (a
    /// `MULTI` spanning slots is not expressible in cluster mode), so any of
    /// them names the same slot — pass the most representative one, typically
    /// the key that would also be `WATCH`ed.
    ///
    /// This is the form that makes the probe *mean* something on a cluster: an
    /// argument-free `DISCARD` / `UNWATCH` carries no key for fred to hash, so
    /// without the pin the router sends it to an arbitrary node and the verdict
    /// describes a connection the caller will never use. Same defect b.ji8 fixed
    /// for UNWATCH teardown, same fix.
    pub async fn acquire_txn_for(&self, hash_key: &str) -> CheckedOutClient {
        self.acquire_txn_inner(Some(hash_key)).await
    }

    /// Run a fred `Transaction`'s `EXEC` on its `txn_pool` checkout — **the
    /// only way service code sends one** — and, when it fails, re-sync that
    /// checkout's cluster routing after a redirect
    /// ([`Self::resync_txn_routing_after_redirect`]) before returning the
    /// error unchanged. Returns exactly what `trx.exec::<R>(abort_on_error)`
    /// returned; a success is returned untouched, and nothing is retried.
    ///
    /// **Precondition:** `checkout` must be the checkout `trx` was opened on
    /// (`checkout.multi()`). fred's `Transaction` exposes no handle on its
    /// client, so the helper cannot check it at run time; every caller opens
    /// the transaction on the checkout it passes, a few lines above, and the
    /// `txn_exec_routing_guard_test` guard checks that statically. `key` is
    /// the key the checkout was pinned to and `op` names the transaction; both
    /// only label the re-sync's WARN.
    ///
    /// **Why one helper.** fred's `Transaction` router fails a redirect past
    /// the pool's zero budget without re-syncing
    /// (`vendor/fred-10.1.0/src/router/transactions.rs:212-217`), so each
    /// caller has to. Sending every `EXEC` through this helper puts the
    /// re-sync on the error path, before the caller sees the error, by
    /// construction. That it runs on the transaction's own checkout rests on
    /// the precondition above, not on construction. The callers then map the
    /// error exactly as before.
    pub(crate) async fn exec_txn<R>(
        &self,
        checkout: &Client,
        trx: Transaction,
        key: &str,
        op: &str,
        abort_on_error: bool,
    ) -> Result<R, fred::error::Error>
    where
        R: FromValue,
    {
        let result = trx.exec::<R>(abort_on_error).await;
        if let Err(e) = &result {
            self.resync_txn_routing_after_redirect(checkout, key, op, e)
                .await;
        }
        result
    }

    /// Re-sync a `txn_pool` checkout's cluster routing after its fred
    /// `Transaction` failed with `"Max redirections exceeded"` — the shape a
    /// `MOVED` or an `ASK` gives a transaction on this pool, whose redirection
    /// budget is zero ([`is_transaction_router_exhausted_error`]). Any other
    /// error, `"Max attempts exceeded"` included, is left alone: that one
    /// comes from a failed drain or a route miss, after which fred has already
    /// reconnected, and no redirect says the routing is stale. A no-op on a
    /// non-clustered client. Reached only through [`Self::exec_txn`].
    ///
    /// **Why it is needed.** fred's `Transaction` router fails a redirect past
    /// the budget without re-syncing
    /// (`vendor/fred-10.1.0/src/router/transactions.rs:212-217`), unlike
    /// `process_moved` for a single command (`router/commands.rs:298-307`).
    /// Each `txn_pool` slot is its own fred client with its own routing table,
    /// and the checkout's hygiene probe is keyless, so after a slot moves the
    /// member that met the `MOVED` would keep sending that slot's
    /// transactions to the old owner until something else re-synced it.
    ///
    /// The sync is [`Self::sync_txn_checkout_routing`]: bounded, logged here
    /// at WARN (with the pinned `key` as the field) when it fails, and it
    /// never changes the caller's answer. An `ASK` re-syncs too, which is
    /// harmless: the slot keeps its owner until the migration ends.
    async fn resync_txn_routing_after_redirect(
        &self,
        checkout: &Client,
        key: &str,
        op: &str,
        error: &fred::error::Error,
    ) {
        if !is_transaction_redirections_exhausted_error(error) {
            return;
        }
        match self.sync_txn_checkout_routing(checkout).await {
            Ok(()) => {}
            Err(TxnRoutingSyncFailure::Failed(e)) => tracing::warn!(
                key = %key,
                error = %e,
                kind = ?e.kind(),
                "Cluster routing sync of a txn-pool connection after the {op} transaction was \
                 redirected failed; returning the transaction's error"
            ),
            Err(TxnRoutingSyncFailure::TimedOut { timeout_ms }) => tracing::warn!(
                key = %key,
                timeout_ms,
                "Cluster routing sync of a txn-pool connection after the {op} transaction was \
                 redirected timed out; returning the transaction's error"
            ),
        }
    }

    /// Sync one `txn_pool` checkout's cluster routing (`CLUSTER SLOTS`, then
    /// the connections it implies) — the sync fred itself runs on a single
    /// command's `MOVED`, so it adds no new router failure mode. Does nothing,
    /// and answers `Ok`, on a non-clustered client.
    ///
    /// Bounded by the client's command timeout
    /// (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`; unbounded when that is `0`,
    /// like every command). A sync that fails or times out is reported to the
    /// caller, which logs it at WARN under its own field names and otherwise
    /// ignores it: the caller's own outcome stands. A timed-out or cancelled
    /// sync keeps running in the member's router, and a later command on that
    /// member queues behind it under its own timeout. It does not undo the
    /// checkout's hygiene: a connection the sync opens carries no transaction
    /// state.
    ///
    /// Shared by [`Self::resync_txn_routing_after_redirect`] and the
    /// `CreateLayer` commit's repeated runs — an overwrite's second run, and
    /// every run of its in-process finisher
    /// (`LayerStore::dispatch_create_layer_commit`).
    pub(crate) async fn sync_txn_checkout_routing(
        &self,
        checkout: &Client,
    ) -> Result<(), TxnRoutingSyncFailure> {
        if !checkout.is_clustered() {
            return Ok(());
        }
        #[cfg(any(test, feature = "test-fault-injection"))]
        self.txn_routing_syncs_for_test
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let timeout = checkout.perf_config().default_command_timeout;
        let synced = if timeout.is_zero() {
            Ok(checkout.sync_cluster().await)
        } else {
            tokio::time::timeout(timeout, checkout.sync_cluster()).await
        };
        match synced {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(TxnRoutingSyncFailure::Failed(e)),
            Err(_elapsed) => Err(TxnRoutingSyncFailure::TimedOut {
                timeout_ms: timeout.as_millis() as u64,
            }),
        }
    }

    /// How many `txn_pool` checkout routing syncs
    /// [`Self::sync_txn_checkout_routing`] has issued on this client and its
    /// clones, counted when a sync is issued, whatever its outcome. Stays at
    /// zero on a standalone store. **Test-only.**
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn txn_routing_syncs_for_test(&self) -> usize {
        self.txn_routing_syncs_for_test
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Name the cluster node a probe pinned to `hash_key` was routed to, for
    /// the per-node state-proof streak (b.289 sub-task 4).
    ///
    /// Returns the shared fallback bucket key (`""`) rather than `None`
    /// wherever the node cannot be named, so callers have one total function
    /// and no branch. Three ways to get there, and they are not the same kind
    /// of thing:
    ///
    /// - The [`Self::acquire_txn`] shim, which has no key to hash, and a
    ///   centralized / sentinel / proxied deployment, where
    ///   `cached_cluster_state()` is `None` off the cluster path. These are
    ///   *permanent*, and the one bucket is the correct answer for them: there
    ///   is exactly one endpoint to be wrong about.
    /// - A clustered client whose cached routing view is momentarily absent
    ///   (fred rebuilds it after a topology change) or does not cover the slot
    ///   (`get_server` misses). This one is *transient*, and it is why
    ///   [`Self::clear_state_proof_streak`] clears `""` on ANY `Clean`: without
    ///   that, three verdicts landing in this window would latch the breaker
    ///   for a bucket every later `Clean` walks straight past.
    ///
    /// The slot is derived exactly as the probe itself derives its routing —
    /// `ClusterHash::from(&str)` is `Custom(redis_keyslot(key))` — so the
    /// bucket cannot drift from the connection the verdict describes. The
    /// slot→server step reuses the shape `WatchFingerprint::take` already
    /// uses; `redis_protocol` is deliberately not imported here (see
    /// `storage::txn_checkout::frame_kind_name` for the same rule).
    ///
    /// **Cost, and where it is paid.** `cached_cluster_state()` clones fred's
    /// whole routing table on a clustered client, so this is called only from
    /// the `Dirty` arm of a checkout — never on the `Clean` path, which the
    /// `update_layer` WATCH/MULTI/EXEC slow path drives at the highest
    /// frequency in the process (one checkout per retry attempt, up to
    /// `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS`) — and, on the `Clean`
    /// path's reset, only when the map is non-empty and there is therefore
    /// actually a streak to clear.
    fn probe_node_bucket(client: &Client, hash_key: Option<&str>) -> String {
        let Some(key) = hash_key else {
            return String::new();
        };
        let ClusterHash::Custom(slot) = ClusterHash::from(key) else {
            return String::new();
        };
        client
            .cached_cluster_state()
            .and_then(|state| state.get_server(slot).map(|server| server.to_string()))
            .unwrap_or_default()
    }

    /// Resolve the map key a verdict against `node` belongs in — the **one**
    /// place the `""` folding rule lives (b.289 sub-task 4).
    ///
    /// `node` gets its own bucket when it already has one, or when the map has
    /// room under [`TXN_STATE_PROOF_STREAK_MAX_NODES`]; otherwise the verdict
    /// folds into the shared `""` bucket rather than being dropped.
    ///
    /// **Why this is a shared helper and not two copies.** The rule has to be
    /// applied identically by [`Self::bump_state_proof_streak`] and
    /// [`Self::clear_state_proof_streak`] or the breaker latches forever: if
    /// `bump` folds a verdict into `""` (unnameable node, or a full map) and a
    /// later `clear` removes only the exact key it was handed, the `""` bucket
    /// is never collected and its streak stays past
    /// [`TXN_STATE_PROOF_REPAIR_ATTEMPTS`] for the life of the process. That is
    /// not a corner case: `probe_node_bucket` returns `""` transiently on a
    /// real cluster whenever fred's cached routing view is absent or does not
    /// cover the slot — i.e. exactly during the reshard / reconnect windows
    /// that produce topology-shaped verdicts in the first place.
    ///
    /// The returned lifetime is tied to `node` alone, so the caller's borrow of
    /// the map ends with the call and it can go on to mutate the map.
    fn bucket_key<'n>(streaks: &HashMap<String, u32>, node: &'n str) -> &'n str {
        if streaks.contains_key(node) || streaks.len() < TXN_STATE_PROOF_STREAK_MAX_NODES {
            node
        } else {
            ""
        }
    }

    /// Advance this node's state-proof streak by one and return its new
    /// 1-based length (b.289 sub-task 4).
    ///
    /// Synchronous and `.await`-free by construction, so the streak is
    /// committed before the checkout can be cancelled: a caller dropped between
    /// the verdict and the recycle must not be able to lose the evidence the
    /// breaker is built from.
    ///
    /// The bucket is chosen by [`Self::bucket_key`], shared with
    /// [`Self::clear_state_proof_streak`] so the two cannot disagree about
    /// which entry a given verdict lives in.
    fn bump_state_proof_streak(&self, node: &str) -> u32 {
        let Ok(mut streaks) = self.txn_state_proof_streaks.lock() else {
            // Poisoned: the breaker degrades to "never latches", which is the
            // pre-b.7td-round-3 behaviour (always repair) rather than a
            // wrongly-withheld repair.
            return 1;
        };
        let key = Self::bucket_key(&streaks, node).to_string();
        let entry = streaks.entry(key).or_insert(0);
        *entry = entry.saturating_add(1);
        *entry
    }

    /// Clear the state-proof streaks a `Clean` checkout against `node` re-arms
    /// (b.289 sub-task 4).
    ///
    /// Two entries, not one:
    ///
    /// - `node`'s own bucket, resolved through [`Self::bucket_key`] so a node
    ///   whose verdicts were folded into `""` by a full map has its `Clean`
    ///   folded the same way. Without that symmetry a cap-overflowed node could
    ///   never re-arm.
    /// - The shared `""` bucket, unconditionally. That bucket holds every
    ///   verdict whose node could not be named, and there is no such thing as a
    ///   `Clean` "against" it to pair with — the whole point of the fallback is
    ///   that those verdicts have no node identity. Pre-b.289 the streak was
    ///   pool-wide and ANY `Clean` reset it; clearing `""` on any `Clean` is
    ///   what actually makes the fallback bucket degenerate to that behaviour,
    ///   which is what its rustdoc promises. Leaving it out is the b.289
    ///   round-1 defect: three unnameable verdicts during a routing-view
    ///   rebuild latched the breaker permanently, because every later `Clean`
    ///   named a real node and removed a different key.
    ///
    /// Removes rather than zeroes, so a `Clean` checkout also collects the
    /// entry and the map returns to empty on a healthy pool — which is what
    /// keeps [`Self::probe_node_bucket`]'s routing-table clone off the hot
    /// `Clean` path in steady state (see [`Self::has_state_proof_streak`]).
    fn clear_state_proof_streak(&self, node: &str) {
        if let Ok(mut streaks) = self.txn_state_proof_streaks.lock() {
            let key = Self::bucket_key(&streaks, node);
            streaks.remove(key);
            streaks.remove("");
        }
    }

    /// Whether any node currently carries a state-proof streak.
    ///
    /// The `Clean` path's cheap pre-check: on a healthy pool this is `false`
    /// and the checkout skips [`Self::probe_node_bucket`] entirely, so the
    /// highest-frequency acquirer in the process — the `update_layer`
    /// WATCH/MULTI/EXEC slow path, once per retry attempt — never pays for a
    /// `cached_cluster_state()` clone.
    fn has_state_proof_streak(&self) -> bool {
        self.txn_state_proof_streaks
            .lock()
            .map(|streaks| !streaks.is_empty())
            .unwrap_or(false)
    }

    /// Shared body of [`Self::acquire_txn`] / [`Self::acquire_txn_for`].
    ///
    /// The pool-use token is minted by the checkout itself, i.e. **before** the
    /// probe, so the gauge covers the probe and any recycle: the slot is
    /// genuinely held for that whole window, and a checkout stalled behind a
    /// wedged probe is exactly the saturation an operator needs to see. The
    /// checkout's semaphore permit spans the same window for the same reason —
    /// nothing else may be admitted against a slot this function is still
    /// probing or repairing.
    ///
    /// Infallible by design. Every existing call site takes the guard
    /// unconditionally, and a `Result` here would have to be threaded through
    /// `src/checkpoint/session.rs` too. More importantly a failed *repair* is
    /// not a reason to fail the *acquisition*: the caller's own error handling
    /// (retry loops, `map_fred_err_for_grpc`) is already the right machinery
    /// for a connection that then misbehaves, and denying the checkout would
    /// turn a recoverable slot into a hard outage for that request.
    async fn acquire_txn_inner(&self, hash_key: Option<&str>) -> CheckedOutClient {
        // The checkout mints the pool-use token itself, so the gauge still
        // covers the probe and any recycle — the slot is genuinely held for that
        // whole window.
        let mut checkout = self.txn_pool.acquire().await;

        match probe_txn_slot(
            &checkout,
            hash_key,
            &self.probe_faults,
            FredErrOrigin::client(self, POOL_LABEL_TXN),
        )
        .await
        {
            CheckoutVerdict::Clean => {
                // A slot that answered both probes correctly ends BOTH streaks,
                // and for the same reason in each case: a clean answer is
                // positive evidence that the condition the gate was bounding is
                // over. Whatever made the last few checkouts time out has
                // passed, so the next unknown-state verdict is the first of a
                // fresh streak and repairs immediately; and the classifier and
                // the server evidently do agree on this endpoint's replies, so
                // the repair-ineffective breaker re-arms and the next
                // state-proof verdict is treated as real residue again.
                //
                // The two resets have different SCOPES (b.289 sub-task 4). The
                // unknown-state streak is pool-wide, so any Clean ends it. The
                // state-proof streak is per node, and a Clean against node B
                // says nothing about node A — resetting A's would restore
                // exactly the mixed-topology gap where the breaker can never
                // latch. So this Clean clears THIS node's bucket, plus the
                // shared `""` bucket, which holds the verdicts that had no node
                // identity to begin with and therefore has no Clean of its own
                // to pair with (see `clear_state_proof_streak`).
                self.txn_unknown_state_streak.store(0, Ordering::Relaxed);
                // Guarded on the map being non-empty: naming the node costs a
                // routing-table clone on a clustered client, and this is the
                // path the `update_layer` WATCH/MULTI/EXEC slow path — the
                // highest-frequency acquirer in the process, once per retry
                // attempt — drives hardest. On a healthy pool the map
                // is empty and there is nothing to clear, so the clone is never
                // paid. When it is not empty, though, EVERY Clean checkout in
                // the process pays it, not just the one following the verdict:
                // a bucket is collected only by a Clean against that same node
                // (plus the shared `""` one), so the guard stays open for as
                // long as any bucket is uncollected — permanently once the
                // breaker has latched on a node that never answers a probe
                // correctly, since the only thing that would clear it is the
                // Clean that is not happening. Accepted: the clone is one small
                // `Vec<SlotRange>` of `Str`-backed fields plus a `to_string()`,
                // beside the two probe round trips this same checkout has
                // already paid. See `probe_node_bucket`'s rustdoc.
                if self.has_state_proof_streak() {
                    self.clear_state_proof_streak(&Self::probe_node_bucket(&checkout, hash_key));
                }
            }
            CheckoutVerdict::MootConnectionGone(err) => {
                // Not a leak: a MULTI and a WATCH both die with their
                // connection, so there is no residue for this borrower to
                // inherit and fred's own reconnect machinery owns the recovery.
                // Uncounted for the same reason `WatchGuard` leaves its moot
                // arm uncounted — this is the common failure on a pool that
                // sees failover, managed-proxy recycles and ordinary
                // reconnects, and counting it would page on routine churn.
                tracing::debug!(
                    error = %err,
                    kind = ?err.kind(),
                    pinned = hash_key.is_some(),
                    "txn checkout probe failed because the connection is already gone; no leaked \
                     transaction state can survive a torn-down connection, so the slot is not \
                     recycled"
                );
            }
            CheckoutVerdict::Dirty { reason, detail } => {
                let metrics = self.metrics_snapshot();
                // The counter ticks on EVERY dirty verdict, gated or not: it
                // means "the probe found this slot unfit", which is true
                // regardless of whether the repair was admitted. Only the
                // recycle is rate-limited.
                if let Some(metrics) = metrics.as_ref() {
                    metrics.inc_valkey_txn_dirty_checkout(reason);
                }

                // Name the node this verdict is about, once, for the log
                // fields below and — on the state-proof path — for the streak
                // bucket. Only reached from this arm: on a clustered client it
                // clones fred's routing table, which must not land on the
                // `Clean` path. Empty string on a centralized / proxied
                // deployment and through the `acquire_txn()` shim, where there
                // is exactly one node to be wrong about.
                let node = Self::probe_node_bucket(&checkout, hash_key);

                // Two independent streaks, each consulted by exactly
                // one gate; the other reads 0 on any given verdict, so the log
                // event always shows the number the decision was actually made
                // on. See `txn_checkout`'s module docs for both trades.
                //
                // - Unknown-state (`probe_timeout` / `probe_error` /
                //   `topology_reply`), POOL-WIDE: recycle on
                //   the first of a streak and every Nth after it, so a
                //   store-wide latency event is not answered with a pool-wide
                //   reconnect storm.
                // - State-proof (`leaked_multi` / `unexpected_reply`),
                //   PER NODE (b.289 sub-task 4): recycle
                //   immediately — until several consecutive verdicts against
                //   THAT node prove the recycle is not clearing anything, at
                //   which point the repair is withheld for that node rather
                //   than repeated forever. Pool-wide, a Clean checkout against
                //   any healthy node reset the streak and the breaker could
                //   never latch on a single misbehaving one.
                let mut unknown_state_streak = 0u32;
                let mut state_proof_streak = 0u32;
                let recycle;
                let repair_ineffective;
                if is_state_proof_reason(reason) {
                    // Committed synchronously, before any `.await` below, so a
                    // cancelled checkout cannot lose the evidence.
                    state_proof_streak = self.bump_state_proof_streak(&node);
                    repair_ineffective = state_proof_repair_is_ineffective(state_proof_streak);
                    recycle = !repair_ineffective;
                } else {
                    unknown_state_streak = self
                        .txn_unknown_state_streak
                        .fetch_add(1, Ordering::Relaxed)
                        .saturating_add(1);
                    repair_ineffective = false;
                    recycle = should_recycle_unknown_state(unknown_state_streak);
                }

                // One field set, three levels. `detail` renders through
                // `Display`, and `tracing` evaluates field expressions only
                // after the callsite's enabled check, so the suppressed arm
                // formats nothing under an `info` filter (b.7td round 3).
                macro_rules! dirty_event {
                    ($level:ident) => {
                        tracing::$level!(
                            reason,
                            detail = %detail,
                            pinned = hash_key.is_some(),
                            // `host:port` of the cluster node this verdict is
                            // about, or empty when it could not be named
                            // (centralized / proxied / unpinned shim). A LOG
                            // field and deliberately NOT a metric label: node
                            // identity is unbounded-cardinality over a
                            // deployment's lifetime, and the counter's job is
                            // "how often", not "where" (b.289 sub-task 4).
                            node = node.as_str(),
                            probe_timeout_ms = TXN_CHECKOUT_PROBE_TIMEOUT.as_millis() as u64,
                            suppressed = !recycle,
                            unknown_state_streak,
                            state_proof_streak,
                            recycle_rearm_every = TXN_RECYCLE_REARM_EVERY,
                            "txn pool checkout found the slot unfit for a transaction"
                        )
                    };
                }

                if repair_ineffective {
                    // Every withheld verdict ticks the counter — that is what
                    // keeps the breaker from being silent. The ERROR is emitted
                    // on the tripping verdict and every re-arm interval after
                    // it: the state that latches this breaker reproduces on
                    // every checkout in the pool, and 128 retry attempts per
                    // request would turn a per-checkout ERROR into a log flood
                    // describing one unchanging fact.
                    if let Some(metrics) = metrics.as_ref() {
                        metrics.inc_valkey_txn_repair_ineffective(reason);
                    }
                    if should_escalate_ineffective_repair(state_proof_streak) {
                        tracing::error!(
                            reason,
                            detail = %detail,
                            pinned = hash_key.is_some(),
                            // The breaker is per node (b.289 sub-task 4), so
                            // this field is the whole scope of the escalation:
                            // hygiene is now off for THIS node only, and every
                            // other node's checkouts still repair. Empty means
                            // the node could not be named — a centralized or
                            // proxied endpoint, where the scope is the pool.
                            // Not a metric label; see `dirty_event!`.
                            node = node.as_str(),
                            state_proof_streak,
                            repair_attempts = TXN_STATE_PROOF_REPAIR_ATTEMPTS,
                            escalation_rearm_every = TXN_RECYCLE_REARM_EVERY,
                            consequence = "the slot is being handed to the borrower unrepaired, \
                                           and stays that way until some checkout pinned to this \
                                           node probes Clean. A state-proof verdict that survives \
                                           repeated rebuilds \
                                           is far more likely to be the probe's classifier and \
                                           the server disagreeing about a reply shape — a managed \
                                           proxy re-rendering the DISCARD error, say — than real \
                                           leaked transaction state, which a single rebuild \
                                           clears. Compare the raw reply in `detail` against \
                                           storage::txn_checkout's classifier",
                            "txn slot recycling is not clearing the dirty verdict; repair \
                             withheld for this node to stop a reconnect loop"
                        );
                    } else {
                        dirty_event!(debug);
                    }
                } else if recycle {
                    dirty_event!(warn);
                } else {
                    // Suppressed by the unknown-state gate. `debug!`, not
                    // `warn!`: this is the store-wide-latency regime, where
                    // every checkout in the pool goes dirty and the retry loop
                    // re-enters the checkout up to 128 times per request. The
                    // counter above is the non-silent record.
                    dirty_event!(debug);
                }

                if recycle {
                    // The whole checkout — slot, gauge token and permit —
                    // travels into the repair task, so both
                    // `valkey_pool_in_use{pool="txn"}` and the permit still
                    // reflect reality when the caller is cancelled mid-repair:
                    // the slot is held by the recycle task until it finishes,
                    // and nothing else may be admitted against it (b.f4f).
                    checkout = self.recycle_txn_slot(checkout, reason).await;
                }
            }
        }

        checkout
    }

    /// Rebuild a `txn_pool` connection in place after the checkout probe found
    /// it dirty (b.7td).
    ///
    /// **Why a full quit + reconnect and not `force_reconnection()`.** fred's
    /// `force_reconnection()` routes through the same path as
    /// `drop_broken_connections`, which only acts on connections that have
    /// raised a *transport* error. A socket that is perfectly healthy at the TCP
    /// layer but has an open `MULTI` on it — or a reply stream offset against
    /// its command stream — is invisible to that path, so the one API that
    /// sounds like the fix provably cannot deliver it. `quit()` closes the
    /// connection unconditionally; `connect()` starts a new router that
    /// rebuilds it and re-runs the handshake (including `AUTH`/`HELLO` through
    /// the configured credential provider).
    ///
    /// **The hooks have to be re-armed.** fred's `quit()` calls
    /// `Notifications::close_public_receivers`, which *swaps in fresh broadcast
    /// senders* — every existing `error_rx()` / `reconnect_rx()` receiver sees
    /// `RecvError::Closed`. Both hook tasks in `spawn_error_metric` /
    /// `spawn_reconnect_metric` `break` on `Closed`, so a recycle would
    /// otherwise silently and permanently blind this slot's
    /// `valkey_errors_total{pool="txn"}`, `valkey_reconnects_total{pool="txn"}`
    /// and the b.cbe steady-state credential-rejection hint. They are therefore
    /// re-spawned against the rebuilt connection by [`Self::rearm_pool_hooks`],
    /// which **aborts the previous pair first** — read its docs before touching
    /// it: on the `QUIT`-timeout path (the expected one here) `quit()` never
    /// reaches the sender swap, the old tasks never see `Closed`, and a re-arm
    /// that merely pruned finished handles would double-count this slot's
    /// counters forever while never being able to collect the orphans.
    ///
    /// **The re-arm is unconditional, not a success action** (b.289 sub-task
    /// 5). It runs after the bounded reconnect wait resolves on *every* arm —
    /// rebuilt, reconnect-failed, and reconnect-timed-out alike — because the
    /// senders may already have been swapped by the `quit()` above whether or
    /// not the rebuild then worked. Written as a success action it left a
    /// quit-succeeded + init-failed slot with permanently dead hooks until some
    /// later recycle happened to succeed. See [`Self::repair_held_txn_slot`],
    /// where the ordering is fenced by a source-inspection guard.
    ///
    /// **Why it runs in a spawned task that owns the guard.** `quit()` + the
    /// old router's abort + `connect()` + the reconnect wait is up to 7.5 s of
    /// awaiting, and the checkout that is waiting on it is routinely cancelled
    /// — an RPC deadline expiring, a client disconnecting, a `select!` arm
    /// losing. Run inline, a cancellation anywhere in that window drops the
    /// future mid-repair:
    /// `quit()` may already have swapped fred's broadcast senders while
    /// `rearm_pool_hooks` never runs (and the new router's handle never reaches
    /// the router supervisor), and because fred's `connect()` spawns the router
    /// synchronously, the connection comes back *healthy* — so every later
    /// checkout probes `Clean`, never re-arms, and that slot's
    /// `valkey_errors_total{pool="txn"}` / `valkey_reconnects_total{pool="txn"}`
    /// are silently dead for the life of the pod. Moving the repair into a task
    /// that owns the `OwnedMutexGuard<Client>` makes the slot stay held until
    /// the repair finishes regardless of what the caller does, and makes the
    /// caller's await point a plain "wait for a handle" that is safe to drop.
    /// The guard is handed back through the `JoinHandle` so the caller resumes
    /// holding exactly the slot it acquired. Same shape as `WatchGuard::Drop`,
    /// for the same reason, and on a tracker [`Self::quit`] drains.
    ///
    /// Every failure arm logs and returns. A slot that could not be rebuilt is
    /// handed to the borrower anyway — see [`Self::acquire_txn_inner`] on why a
    /// failed repair must not become a failed acquisition.
    async fn recycle_txn_slot(
        &self,
        checkout: CheckedOutClient,
        reason: &'static str,
    ) -> CheckedOutClient {
        if self.txn_recycle_tracker.is_closed() {
            // Shutdown is in progress: the connection is about to be closed
            // either way, so a repair buys nothing and a spawn on a closed
            // tracker would not be waited on.
            tracing::debug!(
                reason,
                "txn slot recycle skipped: shutdown in progress, the connection is about to close"
            );
            return checkout;
        }

        let client = self.clone();
        // Split and reassemble around the spawn: `repair_held_txn_slot` takes a
        // `&Client`, and the composite has to be moved into the task whole so
        // the slot, the gauge and the semaphore permit all stay held for the
        // repair's duration.
        let (guard, token, entitlement) = checkout.into_parts();
        let handle = self.txn_recycle_tracker.spawn(async move {
            client.repair_held_txn_slot(&guard, reason).await;
            // All three travel back to the caller here; the slot has been held
            // — and the gauge has counted it, and the permit has withheld it
            // from every other caller — continuously since the checkout.
            CheckedOutClient::reassemble(guard, token, entitlement)
        });

        match handle.await {
            Ok(checkout) => checkout,
            Err(join_err) => {
                // Only reachable if the repair task panicked or was aborted.
                // The guard, its token AND its permit died with it — they are
                // one value, and an aborted task's `JoinHandle` resolves only
                // after the task's future has been dropped — so by the time we
                // are here the slot is already free, the gauge already
                // decremented, and the permit already returned to the
                // semaphore. Take a fresh checkout rather than failing an
                // acquisition that every call site treats as infallible; it
                // parks on the permit queue exactly like any other.
                tracing::error!(
                    reason,
                    error = %join_err,
                    "txn slot recycle task did not return the slot; acquiring a fresh one \
                     unprobed so the borrower still gets a connection"
                );
                self.txn_pool.acquire().await
            }
        }
    }

    /// The body of a recycle, run while the slot's mutex guard — and the
    /// checkout's permit — are held by the caller: [`Self::recycle_txn_slot`]'s
    /// spawned task for a dirty checkout, or [`Self::restart_txn_router`] for a
    /// router the supervisor saw end (`reason` `router_ended`).
    ///
    /// `quit()` (bounded), then end the slot's old router
    /// (`RouterSupervisor::abort_current`), then `connect()` to start the new
    /// router, then a bounded wait for it to report its connection.
    /// `connect()` rather than `init()`: `init()` returns the router's
    /// `ConnectHandle` only once the connection is up and drops it if it fails
    /// or times out, while the handle is what the router supervisor needs.
    /// `connect()` returns it at once, and the repair **hands it over** to the
    /// slot's watcher (`RouterSupervisor::hand_over`) on every arm, so the
    /// supervisor watches the router this repair started. The wait only
    /// decides what is logged; the router keeps connecting after it gives up.
    ///
    /// **Why the old router is aborted.** `quit()` does not end it in step
    /// with the repair: when the bounded `QUIT` times out the old router is
    /// still draining the command that hung, and when `QUIT` succeeds fred
    /// answers it on write and the router ends only after a further await.
    /// Left running, it would end after `connect()` and store its command
    /// receiver back on the client, a receiver `connect()` had replaced. The
    /// next repair's `quit()` would then return at once, and its `connect()`
    /// would start a router on that dead channel while the router this repair
    /// started kept serving the slot, unwatched. `abort_current` ends it, and
    /// waits for it to end, before `connect()`; see `storage::router_supervisor`'s
    /// module docs.
    async fn repair_held_txn_slot(&self, client: &Client, reason: &'static str) {
        if let Err(err) =
            tokio::time::timeout(TXN_RECYCLE_QUIT_TIMEOUT, ClientLike::quit(client)).await
        {
            tracing::warn!(
                reason,
                timeout_ms = TXN_RECYCLE_QUIT_TIMEOUT.as_millis() as u64,
                error = %err,
                "txn slot recycle: QUIT did not complete in time; aborting the slot's router and \
                 reconnecting anyway"
            );
        }

        // On every arm, not only the `QUIT` timeout: see the rustdoc above. The
        // `quit()` future has completed or been dropped by its timeout, so no
        // `quit()` is in flight; a `QUIT` command still queued in the old
        // router dies with it, and nothing awaits its reply.
        match self
            .router_supervisor
            .abort_current(ClientLike::id(client))
            .await
        {
            AbortCurrent::Ended | AbortCurrent::AlreadyEnded => {}
            AbortCurrent::StillRunning => tracing::warn!(
                reason,
                timeout_ms = ROUTER_ABORT_TIMEOUT.as_millis() as u64,
                consequence = "the old router may still store its command receiver back once it \
                               ends, after which the next repair of this slot cannot replace the \
                               router serving it",
                "txn slot recycle: the slot's previous router did not end within its abort \
                 budget; reconnecting anyway"
            ),
            AbortCurrent::Unsupervised => tracing::debug!(
                reason,
                "txn slot recycle: router supervision is shut down; the slot's previous router \
                 was not aborted"
            ),
        }

        // Polled once BEFORE `connect()`, so the wait is subscribed to the
        // connect notification before the new router can send it: fred's
        // `wait_for_connect` returns at once if the client reads `Connected`,
        // and otherwise subscribes on its first poll
        // (`fred-10.1.0/src/runtime/_tokio.rs:327-336`). Polling it only after
        // `connect()` could miss a fast connection and wait out the whole
        // budget. On the `router_ended` path a router that ended by RETURNING
        // stored its command receiver back
        // (`fred-10.1.0/src/router/commands.rs:549`, `:575`), which makes
        // `quit()` a no-op (`commands/impls/server.rs:23-27`), and a returning
        // router's task moves the state only off `Disconnecting`
        // (`runtime/_tokio.rs:311`). So a router that ended by returning AFTER
        // IT HAD CONNECTED still reads `Connected`, and the wait resolves here
        // and is not awaited. A router whose first connect failed did not:
        // under fred's default `fail_fast` it returns without ever setting
        // `Connected` (`router/commands.rs:534-549`), and its task moves the
        // `Disconnecting` left by the `quit()` of the repair that started it to
        // `Disconnected` (the connect wrapper, `runtime/_tokio.rs:299-313`).
        // The restart after that death therefore finds `Disconnected`, and its
        // wait stays pending and is awaited, for up to
        // `TXN_RECYCLE_CONNECT_TIMEOUT`, under the slot guard and a permit.
        // (Had that repair's own `quit()` been a no-op, the client still reads
        // `Connected` and the wait resolves here.) Everywhere else `quit()` ran
        // past its stored-receiver check and set `Disconnecting` before its
        // first await: a router it ended moves that to `Disconnected`, one
        // `abort_current` ended leaves it, and a router aborted before the
        // repair had dropped its receiver, so `quit()`'s send failed on the
        // closed channel after the state change. In each the wait stays
        // pending and is awaited, bounded, for the new router's connection.
        let mut connected = std::pin::pin!(ClientLike::wait_for_connect(client));
        let reported_before_restart = futures::poll!(connected.as_mut());
        let connection_task = ClientLike::connect(client);
        let init = match reported_before_restart {
            std::task::Poll::Ready(_) => None,
            std::task::Poll::Pending => {
                Some(tokio::time::timeout(TXN_RECYCLE_CONNECT_TIMEOUT, connected).await)
            }
        };

        // UNCONDITIONAL, and deliberately ahead of the `match` (b.289 sub-task
        // 5). The re-arm was written as a *success* action, but the predicate
        // it actually needs is "a `quit()` was attempted, so the broadcast
        // senders MAY have been swapped" — and that is true on every arm below.
        //
        // fred swaps the senders (`Notifications::close_public_receivers`) on
        // `quit()`'s SUCCESS path only, after the await at
        // `fred-10.1.0/src/commands/impls/server.rs:39`, at `:40-42`. So
        // quit-succeeded + reconnect-failed left this slot's old hook tasks having
        // already seen `RecvError::Closed` and exited, with nothing re-arming
        // them: `valkey_errors_total{pool="txn"}`,
        // `valkey_reconnects_total{pool="txn"}` and the b.cbe steady-state
        // credential hint were dead for that slot until some LATER recycle
        // happened to succeed. Metrics-only and self-healing, but the window is
        // unbounded — a slot whose rebuilds keep failing is exactly the slot
        // whose errors an operator most needs to see.
        //
        // Safe on the quit-timeout path too, in both worlds:
        //   - `quit()` timed out => `tokio::time::timeout` DROPPED that future,
        //     so `close_public_receivers` can never run after this line; the
        //     old tasks are still live on the still-current senders.
        //   - `quit()` succeeded => the senders were swapped before this line.
        // `rearm_pool_hooks` aborts the exact previously-registered pair before
        // inserting the new one, which is correct in both — read its rustdoc
        // before moving this call.
        //
        // The hooks are broadcast subscriptions (`client.error_rx()`), not
        // commands, so they need no live connection to arm.
        self.rearm_pool_hooks(client, POOL_LABEL_TXN);

        // The new router's handle goes to the slot's watcher on every arm, so
        // the supervisor restarts THIS router if it ends. With no watcher (the
        // supervisor shut down) the handle is dropped, which detaches the task
        // rather than aborting it; `quit()` at shutdown is what ends it.
        self.router_supervisor
            .hand_over(ClientLike::id(client), connection_task);

        // Logging only from here: the re-arm and the hand-over above are the
        // only actions, and they have already happened on every arm.
        match init {
            None => tracing::info!(
                reason,
                "txn slot recycled: router restarted and its metrics hooks re-armed; the \
                 connection was not awaited because the client still reported Connected, as it \
                 does after its previous router ended (the router supervisor watches the new one)"
            ),
            Some(Ok(Ok(()))) => tracing::info!(
                reason,
                "txn slot recycled: connection rebuilt and its metrics hooks re-armed"
            ),
            Some(Ok(Err(err))) => tracing::error!(
                reason,
                error = %err,
                kind = ?err.kind(),
                consequence = "the slot goes back to its borrower (or the pool) unrepaired; \
                               commands on it fail and route through the caller's own retry / \
                               error mapping, the router supervisor restarts its router once \
                               that router ends, and the next checkout probes it again",
                "txn slot recycle: reconnect failed"
            ),
            Some(Err(_elapsed)) => tracing::error!(
                reason,
                timeout_ms = TXN_RECYCLE_CONNECT_TIMEOUT.as_millis() as u64,
                consequence = "the slot goes back to its borrower (or the pool) before its \
                               router reported a connection; commands on it fail until it \
                               connects and route through the caller's own retry / error \
                               mapping, and the next checkout probes it again",
                "txn slot recycle: reconnect exceeded its budget"
            ),
        }
    }

    /// Attach — or re-attach — the reconnect / error metric hooks to a pooled
    /// client, aborting whatever pair was armed on it before (b.7td).
    ///
    /// This is the **only** installer `ValkeyClient` has for the `txn` pool's
    /// hooks: the initial install in [`Self::install_observability_hooks`] and
    /// every recycle in [`Self::repair_held_txn_slot`] both come through here.
    /// The one other arm of a slot's client is a slot replacement's, which
    /// arms the fresh client through the member replacer before swapping it
    /// in (`storage::member_replacement`). Both end in
    /// [`PoolHooks::rearm_with`] on the member replacer's `PoolHooks`, the
    /// installer the stream readers' connections and the multiplexed members
    /// are armed through too, so there is one place a hook can be wired
    /// without the shared [`ValkeyAuthDiagnostics`].
    ///
    /// **Idempotent by abort, not by hope.** The obvious re-arm — spawn a fresh
    /// pair and prune the finished ones — is only correct if `quit()` actually
    /// swapped fred's broadcast senders, and it does that *after* its internal
    /// await. When the [`TXN_RECYCLE_QUIT_TIMEOUT`] fires first (the expected
    /// path for a `probe_timeout`: a connection that will not answer a 650 ms
    /// `DISCARD` will not answer a `QUIT` in 2 s either) the senders are never
    /// swapped, the previous hook tasks never observe `RecvError::Closed`, and
    /// they never finish — so a blind re-arm leaves two live pairs on one
    /// channel, permanently double-counting that pool's
    /// `valkey_errors_total` / `valkey_reconnects_total` and the b.cbe hint,
    /// with a `retain(!is_finished())` prune that can never collect them.
    /// Aborting the registered pair first is correct in both worlds — which is
    /// exactly what lets [`Self::repair_held_txn_slot`] call this
    /// **unconditionally** after its reconnect wait resolves (b.289 sub-task 5)
    /// rather than only on the success arm: whether `quit()` reached the sender
    /// swap or was dropped short of it, the abort-then-insert leaves exactly
    /// one live pair on the current senders.
    ///
    /// No-op before `install_observability_hooks` has run (`metrics` is `None`),
    /// which is the state startup PINGs and metric-free tests operate in, and
    /// after [`Self::quit`] has taken `background_tasks` (shutdown: nothing
    /// worth re-arming, and nobody left to abort the new pair).
    ///
    /// A `txn` slot's pair is armed by the member replacer
    /// ([`MemberReplacer::arm_txn`]), which adds the slot's identity so the
    /// slot's fast unresponsive verdicts reach it (`storage::member_replacement`);
    /// the pair itself is [`PoolHooks::rearm_with`]'s, as for every pool. The
    /// client's own [`Self::pool_hooks`] arms any other `pool_label`; every
    /// caller today passes `POOL_LABEL_TXN`.
    fn rearm_pool_hooks(&self, client: &Client, pool_label: &'static str) {
        if pool_label == POOL_LABEL_TXN {
            self.member_replacer.arm_txn(client);
        } else {
            self.pool_hooks().rearm(client, pool_label);
        }
    }

    /// The per-connection hook registry and what arming a pair needs, as
    /// handles shared with this client (and with the stream readers'
    /// [`ReaderConnector`]).
    fn pool_hooks(&self) -> PoolHooks {
        PoolHooks {
            metrics: Arc::clone(&self.metrics),
            background_tasks: Arc::clone(&self.background_tasks),
            registry: Arc::clone(&self.pool_hook_handles),
            diagnostics: self.auth_diagnostics.clone(),
        }
    }

    /// Expose the shared `TaskTracker` that WatchGuard::Drop uses to register
    /// its background UNWATCH cleanup task (b.k1r H-stor-2).
    ///
    /// `WatchGuard::drop` does not have a handle to the `ValkeyClient` that
    /// produced it, so we pass the Arc'd tracker through to the guard at
    /// construction time via [`WatchGuard::new`]. The tracker is drained in
    /// [`ValkeyClient::quit`] before the txn-pool is closed.
    pub fn watch_unwatch_tracker(&self) -> Arc<TaskTracker> {
        self.watch_unwatch_tracker.clone()
    }

    /// Expose the current count of in-flight `WatchGuard` UNWATCH cleanup
    /// tasks (b.k1r H-stor-2). Tests and the metrics layer poll this to
    /// populate the `live_edit_storage_watch_unwatch_inflight` gauge and to
    /// assert shutdown drained the tracker.
    pub fn watch_unwatch_inflight(&self) -> usize {
        self.watch_unwatch_tracker.len()
    }

    /// Spawn a dropped snapshot-fence guard's cleanup (a `FenceLease`'s
    /// release, a `WriterIntent`'s `HDEL`) on `runtime`, tracked so
    /// [`Self::quit`] waits for it, bounded, before it closes the pools.
    ///
    /// The guard's `Drop` passes the runtime it found, because a drop has no
    /// task context of its own. Spawning on a closed tracker still runs the
    /// cleanup, so a guard dropped after `quit()` drained the tracker (a task
    /// that outlived its shutdown join) still tries; its command then fails
    /// against closed pools, and the lease or intent lapses at its own
    /// deadline.
    pub(crate) fn spawn_fence_guard_cleanup<F>(&self, cleanup: F, runtime: &tokio::runtime::Handle)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        // Detached: the tracker, not the handle, is what `quit()` waits on.
        drop(self.fence_guard_cleanup_tracker.spawn_on(cleanup, runtime));
    }

    /// Track `task`, which owns a snapshot-fence guard and clears it when it
    /// ends (a WATCH-path attempt's `IntentRenewal` task), on the tracker
    /// [`Self::spawn_fence_guard_cleanup`] uses, so [`Self::quit`] waits for
    /// its clear, bounded, before it closes the pools. The caller spawns the
    /// returned future; it counts as in flight from this call until it
    /// completes or is dropped.
    pub(crate) fn track_fence_guard_task<F: std::future::Future>(
        &self,
        task: F,
    ) -> tokio_util::task::task_tracker::TrackedFuture<F> {
        self.fence_guard_cleanup_tracker.track_future(task)
    }

    /// How many dropped snapshot-fence guards' cleanups, and live
    /// `IntentRenewal` tasks, are still in flight
    /// ([`Self::spawn_fence_guard_cleanup`], [`Self::track_fence_guard_task`]):
    /// what [`Self::quit`] waits for before it closes the pools.
    pub fn fence_guard_cleanups_inflight(&self) -> usize {
        self.fence_guard_cleanup_tracker.len()
    }

    /// Admit one `SubscribeReadLayer` (b.f4f). The returned
    /// [`SubscriberAdmission`] is held by the RPC for its whole lifetime,
    /// re-bootstraps included, and returns its slot on drop.
    ///
    /// Bounded by `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS` and
    /// `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS`; the wait races `cancel`
    /// (the per-session token, cancelled only by a session deletion) and is
    /// implicitly cancelled by a client disconnect, because hyper drops the
    /// handler future. Graceful shutdown is observed only through the permit
    /// semaphore [`Self::quit`] closes; the handler, which also holds its
    /// enrolled shutdown token, calls [`Self::admit_subscriber_until_shutdown`].
    pub async fn admit_subscriber(
        &self,
        cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<SubscriberAdmission, AdmissionRefusal> {
        self.subscriber_admission.admit(cancel).await
    }

    /// [`Self::admit_subscriber`], also racing a shutdown token (the
    /// subscription's enrolled child of the service shutdown token).
    ///
    /// `shutdown` is checked before `cancel` on entry and while queued, so a
    /// request that sees both is refused [`AdmissionRefusal::ShuttingDown`] —
    /// the refusal the client retries — rather than
    /// [`AdmissionRefusal::Cancelled`]. Both record `result="cancelled"`.
    pub async fn admit_subscriber_until_shutdown(
        &self,
        shutdown: &tokio_util::sync::CancellationToken,
        cancel: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<SubscriberAdmission, AdmissionRefusal> {
        self.subscriber_admission
            .admit_until_shutdown(Some(shutdown), cancel)
            .await
    }

    /// Borrow the multiplexed `maintenance_pool`, paired with a
    /// [`PoolUseToken`] for the `valkey_pool_in_use{pool="maintenance"}`
    /// gauge (b.mfn).
    ///
    /// Unlike `acquire_txn` — which hands out an exclusive
    /// `OwnedMutexGuard<Client>` from an `ExclusivePool` —
    /// the maintenance pool is fred's regular multiplexed `Pool`, so there
    /// is no per-connection mutex to acquire. The token therefore tracks
    /// **logical** maintenance use: a non-zero gauge means at least one
    /// callsite is currently issuing maintenance commands. Most holders hold
    /// it only around their own Valkey calls, but the external-value backup
    /// repair (`checkpoint::ext_backup_repair`) holds it for its whole pass,
    /// up to its 300 s deadline, so on the GC leader the gauge reads non-zero
    /// for up to 5 minutes at a time by design (SDD §22.8). A non-zero gauge
    /// across several repair intervals is the sign of a stuck or slow sweep.
    /// The leader-gating in `external_values::spawn_gc_task_with_leader` and
    /// `orphan_sweep::spawn_orphan_sweep` keeps concurrent uses bounded so
    /// nested calls (e.g. `sweep_orphaned_objects` →
    /// `cleanup_orphaned_object`) only multiply the gauge by a small,
    /// well-known factor.
    pub fn acquire_maintenance(&self) -> (&Pool, PoolUseToken) {
        let token = self.make_pool_use_token(POOL_LABEL_MAINTENANCE);
        (&self.maintenance_pool, token)
    }

    /// Borrow the multiplexed snapshot lane ([`Self::snapshot_pool`]), paired
    /// with a [`PoolUseToken`] for the `valkey_pool_in_use{pool="snapshot"}`
    /// gauge.
    ///
    /// The same logical-use shape as [`Self::acquire_maintenance`]: the lane is
    /// a regular multiplexed `Pool` with no per-connection mutex, so the token
    /// counts snapshot reads in flight, not connections held. Hold it for the
    /// whole read.
    pub fn acquire_snapshot(&self) -> (&Pool, PoolUseToken) {
        let token = self.make_pool_use_token(POOL_LABEL_SNAPSHOT);
        (&self.snapshot_pool, token)
    }

    /// The snapshot fan-out's threads (`storage::fanout_thread`): every
    /// snapshot fan-out pass runs through [`FanoutThreads::run_pass`].
    pub fn fanout_threads(&self) -> &FanoutThreads {
        &self.fanout_threads
    }

    /// A clone that is not one of the client's owners (the `owners` field),
    /// for a holder the client itself keeps alive — its member replacer, its
    /// router supervisor's watchers — so that holder cannot keep the client
    /// alive in turn. It works as any clone does while an owner remains;
    /// once the last owner has dropped, its fan-out threads are shut down
    /// and whatever holds it has been released. From then on a fan-out pass
    /// through it ([`FanoutThreads::run_pass`]) answers
    /// `ServiceError::Unavailable`, and any other command on the snapshot
    /// lane fails at once: the lane's routers ended with `snapshot-io`, so
    /// fred answers "Failed to send command to router." (`Unknown`, not
    /// transient → `Internal`) — failing closed, with no hang and no panic.
    /// Nothing that holds an internal clone issues a snapshot read; the
    /// holders only restart `txn` routers. Never hand one out of the client.
    fn internal_clone(&self) -> Self {
        let mut clone = self.clone();
        clone.owners = None;
        clone
    }

    /// Snapshot the metrics handle installed by `install_observability_hooks`,
    /// if any. Returns `None` before installation (and on a poisoned lock —
    /// losing a metric increment is strictly preferable to propagating a panic
    /// out of a hot path).
    ///
    /// Exists so code that needs an owned handle — work that outlives the
    /// `&ValkeyClient` borrow, or a counter ticked on an error path — can take
    /// one without threading `SharedMetrics` through every constructor. Tests
    /// routinely build clients with no metrics installed, so every caller must
    /// be `None`-tolerant.
    pub(crate) fn metrics_snapshot(&self) -> Option<SharedMetrics> {
        match self.metrics.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        }
    }

    /// This client's store out-of-memory signal (b.pa1; `storage::store_oom`),
    /// shared by every clone: what a [`FredErrOrigin`] built from this client
    /// notes a store write refused `-OOM` on, what startup arms with the
    /// collection request key, and what the GC loop wakes on.
    pub fn store_oom_signal(&self) -> &crate::storage::store_oom::StoreOomSignal {
        &self.store_oom
    }

    /// Build a `PoolUseToken` for the given pool label. Increments the gauge
    /// now if metrics have been installed; otherwise returns a no-op token
    /// whose `Drop` does nothing. Callers must hold the returned token for
    /// the full duration the slot is in use.
    fn make_pool_use_token(&self, pool_label: &'static str) -> PoolUseToken {
        PoolUseToken::new(self.metrics_snapshot(), pool_label)
    }

    /// Wire fred's reconnect / error event streams into the Prometheus metrics
    /// registry (b.3z7 H-obs-3, extended in b.avd HIGH follow-up to cover
    /// every pool). Call once at startup after `Metrics::new`.
    ///
    /// Increments `valkey_reconnects_total{pool}` on every reconnect and
    /// `valkey_errors_total{pool,kind}` on every connection-level error so
    /// operators have a first-class "Valkey in trouble" signal during an
    /// outage. Hooks cover all six pools — the pub/sub `subscriber`, the
    /// multiplexed `data`, `maintenance` and `snapshot` pools, the `txn`
    /// `ExclusivePool`, and the stream readers (`subscriber_pool`).
    ///
    /// Pre-fix only `subscriber` + `data` were hooked, so the three
    /// pre-initialised counters for the `ExclusivePool`s + `maintenance`
    /// pool sat at zero forever and any reconnect/error on those pools
    /// was silently invisible to operators — alert rules keyed off
    /// `rate(valkey_reconnects_total{pool="txn"}[5m])` could never fire.
    /// The b.avd HIGH follow-up extends coverage so the AC6 promise
    /// ("All pools' error / reconnect counters pre-registered") delivers
    /// the user-observable behavior, not just the literal pre-init.
    ///
    /// Per-pool note for the `ExclusivePool`: fred exposes it as
    /// `&[RefCount<AsyncMutex<Client>>]`. We synchronously lock each
    /// client just long enough to capture a clone of the `Client` (which
    /// shares the inner state — including the broadcast receivers — with
    /// the original), then spawn the hook task against the clone. The
    /// brief lock is taken before any user traffic starts (init-time
    /// startup), so it cannot contend with a hot-path `acquire_txn` caller.
    ///
    /// Each hook spawns a task that lives until the underlying broadcast
    /// channel closes; the `JoinHandle`s are stashed on the client so
    /// `quit()` aborts them deterministically. The pub/sub subscriber's pair
    /// goes on `background_tasks`; every other pair — the `data`,
    /// `maintenance` and `snapshot` members', the `txn` slots' and the stream
    /// readers' — in the per-connection hook registry, keyed by client id,
    /// because those clients are replaced at runtime.
    pub async fn install_observability_hooks(&self, metrics: SharedMetrics) {
        // Stash the metrics handle so future `acquire_txn` calls and the
        // stream readers populate `valkey_pool_in_use`
        // tokens (b.3z7 H-obs-3). Acquisitions happening *before* this call
        // (startup PINGs) run with a no-op token — fine because those are
        // synchronous and don't represent real application saturation.
        if let Ok(mut guard) = self.metrics.lock() {
            *guard = Some(metrics.clone());
        }

        let mut new_handles: Vec<JoinHandle<()>> = Vec::new();

        // b.cbe: every error hook below gets a clone of the SAME
        // `ValkeyAuthDiagnostics` Arc — as does every stream reader's pair,
        // armed when the reader connects — so all of them in this process (up
        // to ~392; ≈1,176 across the chart's 3 replicas) share one
        // rate-limit gate and the process emits one steady-state auth hint per
        // interval rather than one per connection.
        let diagnostics = self.auth_diagnostics.clone();

        // Pub/sub connection.
        new_handles.push(spawn_reconnect_metric(
            &self.subscriber,
            POOL_LABEL_PUBSUB,
            metrics.clone(),
        ));
        new_handles.push(spawn_error_metric(
            &self.subscriber,
            POOL_LABEL_PUBSUB,
            metrics.clone(),
            diagnostics.clone(),
            None,
        ));

        // Data, maintenance and snapshot pools — each member's current client,
        // under its own pool label (so a reconnect storm caused by a
        // long-running GC sweep is attributable to the maintenance pool, and a
        // snapshot-read connection's errors are not folded into the data
        // pool's series). Armed through the per-connection hook registry by the
        // member replacer, keyed by client id, with each member's identity:
        // the error hook hands the member's fast unresponsive verdicts to the
        // replacer, and a replacement re-arms the pair on the fresh client and
        // disarms the old one's (`storage::member_replacement`).
        self.member_replacer.arm_hooks();

        // ExclusivePool: we need an `&Client` (which is what `EventInterface` is
        // implemented on) to attach the hook.
        // `ExclusiveCheckout::clients_for_startup_hooks` hands back one `Client`
        // *clone* per slot — fred Clients are `Clone` and a clone shares inner
        // state, so the hook subscribes to the same broadcast channels the
        // pooled client emits on. It deliberately exposes no mutex: this is the
        // only route from outside `storage::checkout` to a pooled client, which
        // is what makes "every acquisition of a pool mutex goes through the
        // checkout" a fact about the types rather than a rule this comment asks
        // the reader to keep.
        //
        // It does lock every slot in turn internally, so it is startup-only and
        // says so at its definition. At startup the pool is uncontended (no
        // application code is running yet), so the awaits resolve immediately.
        // Do NOT relax that: an `install_observability_hooks` that ran after
        // traffic started would hold slots that no permit accounts for, and a
        // concurrent scanner holding a permit could then find nothing free. See
        // `storage::checkout`'s module docs.
        //
        // The `txn` pool goes through `rearm_pool_hooks` rather than pushing
        // onto `new_handles` (b.7td round 2): its slots are the ones the
        // dirty-checkout repair rebuilds and a slot replacement swaps a fresh
        // client into (`storage::member_replacement`), so their hook pair must
        // be *registered per connection* and abortable, not anonymous in the
        // shutdown join list — see `rearm_pool_hooks`.
        //
        // NOTE: `background_tasks` must already hold the `Some(_)` the sentinel
        // in `rearm_pool_hooks` checks. It does: `connect()` seeds it with the
        // `manage_subscriptions` handle and only `quit()` takes it.
        for client in self.txn_pool.clients_for_startup_hooks().await {
            self.rearm_pool_hooks(&client, POOL_LABEL_TXN);
        }
        // The stream readers' connections are replaced at runtime (a reader
        // connects lazily and retargets on a cluster), so, like the `txn`
        // slots, their pairs are registered per connection: each reader arms
        // its own when it publishes a new connection, and disarms it when it
        // retires one. Here, after the metrics are installed above, only the
        // readers already connected need arming — none in production, where
        // this runs before any subscription; a reader that connects
        // concurrently arms itself after publishing, so it is armed by one
        // side or both (the re-arm is idempotent per client). The same arm
        // counts such a connection on `live_edit_stream_mux_readers_connected`
        // when its connect, before the metrics existed, could not — once,
        // whichever side gets there first.
        self.stream_mux.arm_reader_hooks();

        if let Ok(mut guard) = self.background_tasks.lock() {
            if let Some(vec) = guard.as_mut() {
                vec.extend(new_handles);
            }
        }
    }

    /// Gracefully close all Valkey connections (SDD §5.6 step 4.4).
    ///
    /// The member replacer is shut down first (`storage::member_replacement`):
    /// no replacement starts after it, and every one in flight, and every old
    /// client's retirement in progress, is cut short at once — a fresh client
    /// not yet in its pool has its router aborted and its hooks disarmed, and
    /// each old client already replaced has its router aborted at once. Its
    /// drain covers all of them, however many, in one
    /// `MEMBER_REPLACER_DRAIN_TIMEOUT`. Then the router supervisor (`storage::router_supervisor`): its
    /// watchers are cancelled and drained before anything closes, so no router
    /// this shutdown ends is restarted.
    ///
    /// Every pool's `quit()` runs concurrently, each under `QUIT_TIMEOUT` (2 s)
    /// (b.3z7 C-2), so the worst-case cost of this step is one `QUIT_TIMEOUT`
    /// rather than the sum of per-pool timeouts. Before the pools quit, the
    /// stream multiplexer wakes every reader's blocked `XREAD BLOCK`
    /// (`StreamMux::shutdown`); otherwise a reader connection's `QUIT` would
    /// queue behind a BLOCK. The readers' connections — only those connected
    /// now; readers connect lazily — are retired concurrently, each `QUIT`
    /// under `QUIT_TIMEOUT` and its router aborted after it
    /// (`StreamMux::quit_readers`). So a timeout here means a genuine stall: a wake did not
    /// land, or a server did not answer.
    /// On a timeout we log a warning and let the process exit anyway rather
    /// than hang the shutdown indefinitely. The sum of this function's
    /// sequential bounds is checked at build time against the shutdown-budget
    /// headroom (the `const` assertion after `STREAM_MUX_SHUTDOWN_GRACE`); a new or
    /// raised bound here belongs in that sum. Process termination will close the
    /// underlying TCP connections; the server-side gRPC clients have
    /// already been notified by the graceful-shutdown GOAWAY earlier in
    /// the sequence.
    ///
    /// Any fred background tasks spawned by `install_observability_hooks`
    /// or the `manage_subscriptions` call in `connect()` are aborted before
    /// the connections close so they don't race the TCP teardown.
    ///
    /// **Without `quit()`, the last owner's drop frees the client.** Its state
    /// holds clones of itself: the member replacer (`Arc`) holds the `txn`
    /// slots' router restart (`MemberReplacer::set_txn_restart`, from
    /// [`Self::start_router_supervision`]), a closure holding a clone of this
    /// client, and the `txn` slots' router watchers hold one each, for their
    /// restart. Those are internal clones ([`Self::internal_clone`]), which do
    /// not count as owners, so the cycle they form lasts only as long as the
    /// owners do: here the replacer's and the supervisor's shutdowns release
    /// them; without `quit()`, the drop of the last owner ([`ClientOwners`])
    /// cancels both, which releases them, and shuts the fan-out threads down
    /// (`storage::fanout_thread`), ending the snapshot lane's routers with
    /// them. Nothing sends the other pools a `QUIT` then: their routers run
    /// on the runtime that started them and end with it at the latest.
    ///
    /// The cleanups dropped snapshot-fence guards spawned
    /// ([`Self::spawn_fence_guard_cleanup`]) are drained, bounded by
    /// `FENCE_GUARD_CLEANUP_DRAIN_TIMEOUT`, before any pool closes, so a fenced
    /// read graceful shutdown dropped has given its lease back to the store.
    pub async fn quit(&self) {
        // First, before any pool closes: stop member replacement, so no fresh
        // client is connected, watched or swapped in while the pools close
        // (its watches belong to the supervisor, which shuts down next).
        self.member_replacer.shutdown().await;
        // Then stop the router supervisor and wait for its watchers to exit,
        // so the routers the `QUIT`s below end are not restarted.
        self.router_supervisor.shutdown().await;
        self.subscriber_admission.close();
        self.catchup_gate.close();
        self.catchup_read_gate.close();
        self.catchup_fenced_read_gate.close();
        self.catchup_build_budget.close();
        // Abort fred-owned background tasks (manage_subscriptions + event
        // hooks) first so they don't log spurious reconnect/error events as
        // the connections tear down. Take the Vec under the mutex and drop
        // the guard before awaiting anything.
        let handles: Vec<JoinHandle<()>> = {
            match self.background_tasks.lock() {
                Ok(mut guard) => guard.take().unwrap_or_default(),
                Err(_) => Vec::new(),
            }
        };
        for h in &handles {
            h.abort();
        }

        // Drain in-flight slot recycles (b.7td round 2). Each owns an
        // `OwnedMutexGuard<Client>` on a `txn_pool` slot, so `txn_pool.quit()`
        // below would otherwise block on that mutex and surface as an
        // unexplained pool-quit timeout. Bounded, and deliberately shorter than
        // a recycle's own worst case: at shutdown the repair has no value left.
        //
        // Drained BEFORE the hook registry below so a recycle that read the
        // `background_tasks` sentinel as `Some` just before this function took
        // it cannot re-arm into a map that has already been emptied.
        //
        // Concurrently, let every dropped snapshot-fence guard's cleanup reach
        // the store while the data pool it runs on is still open. A lease
        // released here frees the other replicas' writers of its layer now;
        // one cut off by the pool teardown below would hold them paused until
        // it lapsed. By the time `main.rs` calls this, graceful shutdown has
        // dropped every fenced read whose task its joins saw end: the catch-up
        // flights with the subscriptions on the service shutdown token, the
        // checkpoint's with the checkpoint loop's cancellation. Run beside the
        // recycle drain rather than after it, so it adds nothing to this
        // function's worst case, which `Config::from_env`'s shutdown-budget
        // floor is written against.
        self.txn_recycle_tracker.close();
        self.fence_guard_cleanup_tracker.close();
        let (recycles, fence_cleanups) = tokio::join!(
            tokio::time::timeout(TXN_RECYCLE_DRAIN_TIMEOUT, self.txn_recycle_tracker.wait()),
            tokio::time::timeout(
                FENCE_GUARD_CLEANUP_DRAIN_TIMEOUT,
                self.fence_guard_cleanup_tracker.wait(),
            ),
        );
        if recycles.is_err() {
            tracing::warn!(
                inflight_remaining = self.txn_recycle_tracker.len(),
                timeout_secs = TXN_RECYCLE_DRAIN_TIMEOUT.as_secs(),
                "timed out draining in-flight txn slot recycles; their connections will be closed \
                 by the pool quit below or by process exit"
            );
        }
        if fence_cleanups.is_err() {
            tracing::warn!(
                inflight_remaining = self.fence_guard_cleanup_tracker.len(),
                timeout_secs = FENCE_GUARD_CLEANUP_DRAIN_TIMEOUT.as_secs(),
                "timed out waiting for dropped snapshot-fence leases and writer intents to be \
                 released, or for running writer-intent renewals to end and clear theirs; \
                 closing the pools anyway (each lapses at its own deadline, so other \
                 replicas' writers or readers of those layers may wait up to one lease or intent \
                 lifetime longer)"
            );
        }

        // Per-connection hook pairs (b.7td round 2): the `txn` slots', the
        // stream readers' and the `data` / `maintenance` / `snapshot` members'.
        // These live in their own registry rather than the vec above, because
        // a recycle or a replacement has to be able to abort *the pair
        // belonging to one connection* — see `rearm_pool_hooks`. Draining the whole map here keeps shutdown's
        // "every fred-owned task is aborted before the sockets close" property.
        let hook_handles: Vec<JoinHandle<()>> = match self.pool_hook_handles.lock() {
            Ok(mut registry) => registry.drain().flat_map(|(_id, pair)| pair).collect(),
            Err(_) => Vec::new(),
        };
        for h in &hook_handles {
            h.abort();
        }

        // Drain any in-flight `WatchGuard::Drop` UNWATCH cleanup tasks before
        // we quit the txn pool (b.k1r H-stor-2). An early-`?` return inside
        // `update_layer_once_watch` spawns an UNWATCH task that holds the fred
        // `OwnedMutexGuard<Client>` across its `unwatch().await`; if we
        // tore the runtime down before that task ran, we'd leave stale
        // WATCH state on a connection headed for `pool.quit()` and would
        // silently pollute the next consumer on a restart-then-reconnect
        // cycle. The tracker caps the worst-case wait by
        // `UNWATCH_DRAIN_TIMEOUT` (module-level, above — `watch_guard` sizes
        // its funnel bound against it under a `const` assertion) so shutdown
        // does not hang indefinitely if a runaway UNWATCH is stuck —
        // consistent with the per-pool `QUIT_TIMEOUT` philosophy below.
        self.watch_unwatch_tracker.close();
        let inflight = self.watch_unwatch_tracker.len();
        if inflight > 0 {
            tracing::debug!(
                inflight,
                timeout_secs = UNWATCH_DRAIN_TIMEOUT.as_secs(),
                "draining in-flight WatchGuard UNWATCH cleanup tasks before txn-pool quit"
            );
        }
        if tokio::time::timeout(UNWATCH_DRAIN_TIMEOUT, self.watch_unwatch_tracker.wait())
            .await
            .is_err()
        {
            tracing::warn!(
                inflight_remaining = self.watch_unwatch_tracker.len(),
                timeout_secs = UNWATCH_DRAIN_TIMEOUT.as_secs(),
                "timed out draining WatchGuard UNWATCH tasks; \
                 stale WATCH on any held txn-pool client will clear on reconnect"
            );
        }

        // Before the pools close: readers fail any subscription still
        // registered (the post-server drain, `grpc::transport::drain_stream_tasks`,
        // normally ended them already) and wake their blocked XREADs through the
        // data pool, so the readers' QUITs are not queued behind a BLOCK.
        self.stream_mux.shutdown(STREAM_MUX_SHUTDOWN_GRACE).await;

        // `ExclusiveCheckout::quit` closes that pool's permit semaphore before
        // delegating to fred, and that is what makes the next line safe (b.f4f).
        // fred's `quit()` locks every slot concurrently, so for its duration the
        // pool looks fully occupied to any scanner — and the slot-recycle and
        // `WatchGuard` UNWATCH drains above do NOT rule out a concurrent
        // checkout: they cover those `txn_pool` holders only (the fence-guard
        // cleanup drain beside the recycle drain covers data-pool commands, not
        // `txn_pool` holders), and background work (the GC/orphan sweeps, the
        // checkpointer) can still take a `txn_pool` checkout. Closing the
        // semaphore first means those callers observe shutdown explicitly
        // (`is_quitting`) instead of scanning an apparently-full pool and
        // tripping the escaped-slot assertion.
        //
        // Those two drains still have to precede this: a recycle or an UNWATCH task
        // holds a `CheckedOutClient`, so `ExclusivePool::quit` would block on its
        // mutex and surface as an unexplained pool-quit timeout.
        //
        // The readers' quit is bounded on its own: each connected reader's
        // retire sends its `QUIT` under `QUIT_TIMEOUT` and then aborts the
        // router, all concurrently (`StreamMux::quit_readers`).
        let (pool_res, subscriber_res, txn_res, readers_overran, maintenance_res, snapshot_res) = tokio::join!(
            tokio::time::timeout(QUIT_TIMEOUT, self.pool.quit()),
            tokio::time::timeout(QUIT_TIMEOUT, self.subscriber.quit()),
            tokio::time::timeout(QUIT_TIMEOUT, self.txn_pool.quit()),
            self.stream_mux.quit_readers(),
            tokio::time::timeout(QUIT_TIMEOUT, self.maintenance_pool.quit()),
            tokio::time::timeout(QUIT_TIMEOUT, self.snapshot_pool.quit()),
        );

        match pool_res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(error = %e, "error closing Valkey data pool"),
            Err(_) => tracing::warn!(
                timeout_secs = QUIT_TIMEOUT.as_secs(),
                "data pool quit timed out; leaving connections to be closed by process exit"
            ),
        }
        match subscriber_res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(error = %e, "error closing Valkey subscriber"),
            Err(_) => tracing::warn!(
                timeout_secs = QUIT_TIMEOUT.as_secs(),
                "subscriber quit timed out; leaving connection to be closed by process exit"
            ),
        }
        if txn_res.is_err() {
            tracing::warn!(
                timeout_secs = QUIT_TIMEOUT.as_secs(),
                "txn pool quit timed out; leaving connections to be closed by process exit"
            );
        }
        if readers_overran > 0 {
            tracing::warn!(
                readers = readers_overran,
                timeout_secs = QUIT_TIMEOUT.as_secs(),
                "stream reader connection quit timed out (a reader's blocked XREAD was not woken, or the server did not answer); those readers' routers were aborted, closing their connections"
            );
        }
        match maintenance_res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(error = %e, "error closing Valkey maintenance pool"),
            Err(_) => tracing::warn!(
                timeout_secs = QUIT_TIMEOUT.as_secs(),
                "maintenance pool quit timed out; leaving connections to be closed by process exit"
            ),
        }
        match snapshot_res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(error = %e, "error closing Valkey snapshot pool"),
            Err(_) => tracing::warn!(
                timeout_secs = QUIT_TIMEOUT.as_secs(),
                "snapshot pool quit timed out; leaving connections to be closed by process exit"
            ),
        }
    }
}

/// Spawn a background task that increments `valkey_reconnects_total{pool}`
/// on every reconnect event fred emits for `client`.
fn spawn_reconnect_metric<C>(
    client: &C,
    pool_label: &'static str,
    metrics: SharedMetrics,
) -> JoinHandle<()>
where
    C: EventInterface + Send + Sync,
{
    let mut rx = client.reconnect_rx();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(_server) => metrics.inc_valkey_reconnect(pool_label),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

/// Spawn a background task that increments `valkey_errors_total{pool,kind}`
/// on every connection-level error fred surfaces for `client`.
///
/// `diagnostics` adds the b.cbe steady-state auth hint. This hook is the *only*
/// place a **mid-life** credential rejection is observable: an access-policy
/// assignment revoked or flipped to `Failed` under an already-running process
/// produces a `WRONGPASS` on reconnect, and `reconnect_on_auth_error` then
/// retries forever. Today that costs one bump of `valkey_errors_total{kind="Auth"}`
/// and nothing else — a counter that says a number went up, not what to do about
/// it. Emission is bounded by the shared [`AuthHintGate`] on the single
/// [`ValkeyAuthDiagnostics`], which every hook on every pool holds a clone of;
/// per-hook rate limiting would be useless at this fan-out.
///
/// No new metric accompanies this: the hint is prose for a human, and
/// `valkey_errors_total{kind="Auth"}` already carries the machine-readable
/// signal an alert rule keys on.
///
/// The hook also ticks `valkey_unresponsive_total{pool,verdict}` for each
/// error `fred::live_edit::liveness::verdict_of` classifies: fred broadcasts
/// one such error when its unresponsive check closes a connection (see
/// [`apply_connect_timeout`]), and the hook is attached to every member of
/// every pool. A fast verdict also goes to `verdict_sink`, when the client is
/// a `data`, `maintenance` or `snapshot` member or a `txn` slot's, for the
/// member replacer (`storage::member_replacement`).
fn spawn_error_metric<C>(
    client: &C,
    pool_label: &'static str,
    metrics: SharedMetrics,
    diagnostics: Option<Arc<ValkeyAuthDiagnostics>>,
    verdict_sink: Option<VerdictSink>,
) -> JoinHandle<()>
where
    C: EventInterface + Send + Sync,
{
    let mut rx = client.error_rx();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok((err, _server)) => {
                    metrics.inc_valkey_error(pool_label, err.kind().to_str());
                    if let Some(verdict) = fred::live_edit::liveness::verdict_of(&err) {
                        metrics.inc_valkey_unresponsive(pool_label, verdict.as_str());
                        if verdict == fred::live_edit::liveness::Verdict::Fast {
                            if let Some(sink) = verdict_sink.as_ref() {
                                sink.fast_verdict();
                            }
                        }
                    }
                    // Narrower than [`is_server_auth_rejection`] on purpose:
                    // steady-state hints only on `WRONGPASS`, never on `NOAUTH`.
                    //
                    // A mid-life `NOAUTH` is precisely the event
                    // `reconnect_on_auth_error` (set true where the fred config is
                    // built, per SR-4.2 / SR-5.1) exists to absorb — a managed-proxy
                    // session reset, or a connection that raced a credential
                    // rotation and needs to re-`AUTH`. fred reconnects and
                    // re-authenticates and the pod is fine. Firing the four-cause
                    // misconfiguration checklist at ERROR on it is a confident,
                    // authoritative *wrong* diagnosis: it sends an operator hunting
                    // an access-policy assignment during a self-healing blip.
                    //
                    // A rejection that is genuinely persistent — revoked or
                    // `Failed` access-policy assignment, wrong username, wrong
                    // audience — presents as `WRONGPASS` (the server has a
                    // credential and refuses it), which is the shape the checklist
                    // actually diagnoses. The connect path keeps both tokens:
                    // there, a `NOAUTH` on the very first connection is terminal,
                    // not a reconnect the client will absorb.
                    if is_server_auth_rejection(&err) && err.details().starts_with("WRONGPASS") {
                        if let Some(diagnostics) = diagnostics.as_ref() {
                            diagnostics.emit_steady_state_hint(pool_label);
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

/// Run `attempt` in a retry loop with exponential backoff.
///
/// - Initial delay: 200 ms, doubled on each failure, capped at 5 s.
/// - Aborts after `budget` has elapsed. The budget is **hard**: each attempt
///   runs under `timeout(remaining budget)`, so an attempt that never resolves
///   (a handshake that hangs) is cut off at the budget instead of overrunning
///   it by up to one handshake timeout, and the phase fails at once. A cut
///   attempt's futures are dropped; the routers they had spawned are ended by
///   their owner: the failed connect drops every phase's `MemberRouters`,
///   each of which aborts every router it still holds, so a failed connect
///   leaves no router task behind. The fatal [`ConnectError::Timeout`] of a
///   cut names the phase and, when an earlier attempt of the phase failed,
///   that attempt's error, so the real cause (a TLS failure, a `WRONGPASS`)
///   is not left only in the per-attempt WARNs.
///
/// `diagnostics` / `auth_hint_emitted` drive the b.cbe operator hint: the first
/// attempt whose failure is a *server-side* credential rejection
/// ([`is_server_auth_rejection`]) renders a cause-specific explanation of the
/// otherwise-opaque `WRONGPASS`. `auth_hint_emitted` is owned by the caller and
/// shared across every pool's loop, so one `connect()` produces one hint no
/// matter how many pools and attempts observe the rejection.
///
/// The hint is evaluated **here, at failure time**, not once before the loop.
/// That is what lets it make the strongest claim in the checklist — "the token's
/// identity checks out and the server rejected it anyway, so suspect the
/// access-policy assignment" — about the credential that was actually refused,
/// rather than about whatever was cached when `connect` started. The credential
/// can also rotate mid-loop, and a failure-time read follows it.
async fn connect_with_retry<F, Fut>(
    label: &str,
    budget: Duration,
    diagnostics: Option<&ValkeyAuthDiagnostics>,
    auth_hint_emitted: &AtomicBool,
    mut attempt: F,
) -> Result<(), ConnectError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<(), fred::error::Error>>,
{
    const INITIAL_DELAY: Duration = Duration::from_millis(200);
    const MAX_DELAY: Duration = Duration::from_secs(5);

    // Tokio's clock, not `std`'s: the real clock in production, and the one a
    // paused test runtime drives, so the hard budget (and the backoff sleeps,
    // already on tokio's clock) can be tested without wall-clock waits.
    let start = tokio::time::Instant::now();
    let mut delay = INITIAL_DELAY;
    // The latest attempt's real error (TLS, `WRONGPASS`, a refused
    // connection, ...): when the budget then cuts a later attempt, the fatal
    // error carries it, so the cause survives beyond the per-attempt WARNs.
    let mut last_error: Option<fred::error::Error> = None;

    loop {
        let remaining = budget.saturating_sub(start.elapsed());
        let Ok(result) = tokio::time::timeout(remaining, attempt()).await else {
            let elapsed = start.elapsed();
            tracing::warn!(
                label,
                elapsed_ms = elapsed.as_millis(),
                budget_ms = budget.as_millis(),
                "Valkey connect attempt did not finish within the connect budget; giving up"
            );
            let cut = format!(
                "{label}: the connect attempt did not finish within the {} ms budget",
                budget.as_millis()
            );
            let details = match &last_error {
                Some(e) => format!("{cut}; the previous attempt failed: {e}"),
                None => cut,
            };
            return Err(ConnectError::Timeout {
                elapsed: elapsed.as_secs_f64(),
                source: fred::error::Error::new(FredErrorKind::Timeout, details),
            });
        };
        match result {
            Ok(()) => return Ok(()),
            Err(e) => {
                let elapsed = start.elapsed();
                tracing::warn!(
                    label,
                    error = %e,
                    elapsed_ms = elapsed.as_millis(),
                    "Valkey connect attempt failed, retrying"
                );

                // The WARN above is the whole operator signal today: an opaque
                // `WRONGPASS invalid username-password pair`, repeated once per
                // attempt until the budget expires and the process exits. Add
                // the interpretation this service is uniquely able to make.
                if is_server_auth_rejection(&e) {
                    if let Some(diagnostics) = diagnostics {
                        diagnostics.emit_connect_hint(label, auth_hint_emitted);
                    }
                }

                if elapsed >= budget {
                    return Err(ConnectError::Timeout {
                        elapsed: elapsed.as_secs_f64(),
                        source: e,
                    });
                }

                // Sleep whichever is shorter: the backoff delay or the remaining budget.
                let remaining = budget.saturating_sub(start.elapsed());
                let sleep_dur = delay.min(remaining);
                tokio::time::sleep(sleep_dur).await;

                // Double the delay, respecting the cap.
                delay = (delay * 2).min(MAX_DELAY);

                if start.elapsed() >= budget {
                    return Err(ConnectError::Timeout {
                        elapsed: start.elapsed().as_secs_f64(),
                        source: e,
                    });
                }
                last_error = Some(e);
            }
        }
    }
}

/// The router `ConnectHandle` of each member of one pool, as
/// [`ValkeyClient::connect`] starts them, for the router supervisor.
///
/// A member's handle is kept from the moment its router is spawned, not only
/// once it has connected, so a connect attempt that `connect_with_retry` cuts
/// off at its budget still leaves the handle here. Every phase's
/// `MemberRouters` is a local of `ValkeyClient::connect` and keeps its handles
/// until `connect` has succeeded and hands them all to the router supervisor
/// ([`Self::into_handles`]). Dropping one that still holds handles — the
/// connect failed, in this phase or any later one, a startup PING included —
/// aborts every router it holds, connected or not, so a failed connect leaves
/// no router task behind.
///
/// `std::sync::Mutex` so the `connect_with_retry` closure can borrow it shared;
/// never held across an `.await`.
struct MemberRouters {
    members: Mutex<Vec<MemberRouter>>,
    /// The runtime the routers are started on: `snapshot-io` for the
    /// snapshot lane ([`Self::on`]; `storage::fanout_thread`), the calling
    /// runtime for every other pool.
    spawn_on: Option<tokio::runtime::Handle>,
}

/// One member's router, and whether its connect succeeded.
#[derive(Default)]
struct MemberRouter {
    router: Option<ConnectHandle>,
    connected: bool,
}

impl MemberRouters {
    /// `members` routers, started on the calling runtime.
    fn new(members: usize) -> Self {
        Self {
            members: Mutex::new((0..members).map(|_| MemberRouter::default()).collect()),
            spawn_on: None,
        }
    }

    /// `members` routers, started on `runtime`: the snapshot lane's, on
    /// `snapshot-io`, where its reply frames are allocated from that
    /// thread's one arena rather than from whichever service worker a router
    /// would run on (`storage::fanout_thread`).
    fn on(members: usize, runtime: tokio::runtime::Handle) -> Self {
        Self {
            members: Mutex::new((0..members).map(|_| MemberRouter::default()).collect()),
            spawn_on: Some(runtime),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<MemberRouter>> {
        self.members.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Start every member of `clients` that has not connected yet,
    /// concurrently, and wait for each to connect — what fred's `Pool::init`
    /// does with `Client::init()`, except that each router's handle is kept as
    /// soon as it is spawned. The first error is returned for
    /// `connect_with_retry` to retry; members that connected keep their
    /// routers, so a retry starts only the rest.
    async fn init<C: ClientLike>(&self, clients: &[C]) -> Result<(), fred::error::Error> {
        let pending: Vec<usize> = {
            let members = self.lock();
            (0..clients.len())
                .filter(|&member| !members[member].connected)
                .collect()
        };
        let results = futures::future::join_all(
            pending
                .iter()
                .map(|&member| self.init_member(&clients[member], member)),
        )
        .await;
        results
            .into_iter()
            .find_map(Result::err)
            .map_or(Ok(()), Err)
    }

    /// `Client::init()` for one member, keeping the router's handle from its
    /// spawn, on [`Self::spawn_on`]'s runtime when set. A previous attempt's
    /// router is aborted first, so no two routers of one client ever run side
    /// by side.
    async fn init_member<C: ClientLike>(
        &self,
        client: &C,
        member: usize,
    ) -> Result<(), fred::error::Error> {
        if let Some(previous) = self.lock()[member].router.take() {
            previous.abort();
        }
        // Subscribed to the connect notification before the router can send
        // it, as `init()` does: the first poll subscribes unless the client
        // already reads `Connected`.
        let mut connected = std::pin::pin!(client.wait_for_connect());
        let before = futures::poll!(connected.as_mut());
        let router = connect_router_on(self.spawn_on.as_ref(), client);
        self.lock()[member].router = Some(router);
        let result = match before {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => connected.await,
        };
        if result.is_ok() {
            self.lock()[member].connected = true;
        }
        result
    }

    /// The handles, in member order. Every member has one once `init`
    /// succeeded, which is the only time this is called.
    fn into_handles(self) -> Vec<ConnectHandle> {
        std::mem::take(&mut *self.lock())
            .into_iter()
            .filter_map(|member| member.router)
            .collect()
    }
}

impl Drop for MemberRouters {
    fn drop(&mut self) {
        // Empty after `into_handles`; otherwise the connect failed, and
        // nothing will ever supervise or quit these routers.
        for member in self.lock().iter_mut() {
            if let Some(router) = member.router.take() {
                router.abort();
            }
        }
    }
}

/// Every router [`ValkeyClient::connect`] started, by pool, in member order:
/// what [`ValkeyClient::start_router_supervision`] watches. `txn` pairs each
/// slot's client with its handle, because the slot is restarted by client id.
/// The stream readers are not here: each reader's router is registered when
/// it first connects (`ReaderConnector`).
struct SupervisedRouters {
    data: Vec<ConnectHandle>,
    txn: Vec<(Client, ConnectHandle)>,
    maintenance: Vec<ConnectHandle>,
    snapshot: Vec<ConnectHandle>,
    pubsub: Vec<ConnectHandle>,
}

/// The owners of one `ValkeyClient` (its `owners` field): one value shared by
/// every owning clone, never by an internal one
/// ([`ValkeyClient::internal_clone`]). Its drop is the client's last owner
/// going, quit or not, and releases what the client's internal clones would
/// otherwise keep alive for the rest of the process:
///
/// - the member replacer ([`MemberReplacer::release`]): no replacement starts,
///   those in progress stop at their next await, and its `txn` slots' router
///   restart, which holds an internal clone, is dropped;
/// - the router supervisor ([`RouterSupervisor::release`]): every watcher
///   exits at its next await, and a `txn` slot's with it its internal clone;
/// - the fan-out threads ([`FanoutThreads::shut_down`]): every one of their
///   runtimes is shut down now, without blocking the dropping thread, and
///   the snapshot lane's routers on `snapshot-io` end with them. Now rather
///   than when the internal clones have gone, so the threads' end does not
///   wait on another runtime polling a cancelled watcher.
///
/// Every step is synchronous and non-blocking, so the drop is safe on any
/// thread: a service worker, a test's thread after its runtime ended, or a
/// fan-out thread itself (`fanout-read`, a `watch-read` thread) when a pass
/// held the last owner. After a `quit()`
/// the replacer and the supervisor are already shut down and only the threads
/// remain to stop.
struct ClientOwners {
    member_replacer: Arc<MemberReplacer>,
    router_supervisor: Arc<RouterSupervisor>,
    fanout_threads: Arc<FanoutThreads>,
}

impl Drop for ClientOwners {
    fn drop(&mut self) {
        self.member_replacer.release();
        self.router_supervisor.release();
        self.fanout_threads.shut_down();
    }
}

/// `ValkeyClient::background_tasks`: fred-owned tasks `quit()` aborts; `None`
/// once `quit()` has taken them (the shutdown sentinel).
pub(crate) type BackgroundTasks = Arc<Mutex<Option<Vec<JoinHandle<()>>>>>;

/// `ValkeyClient::pool_hook_handles`: the metric-hook pair armed on each
/// recyclable connection, by fred client id.
pub(crate) type HookRegistry = Arc<Mutex<HashMap<String, [JoinHandle<()>; 2]>>>;

/// The per-connection metric-hook registry and what arming one pair needs,
/// as handles shared with the `ValkeyClient` (see
/// `ValkeyClient::rearm_pool_hooks` for why pairs are registered by client id
/// and re-armed by abort). Used for every connection that is replaced at
/// runtime: the `txn` slots, the stream readers' connections, and the `data`,
/// `maintenance` and `snapshot` members (`storage::member_replacement`).
#[derive(Clone)]
pub(crate) struct PoolHooks {
    pub(crate) metrics: Arc<Mutex<Option<SharedMetrics>>>,
    pub(crate) background_tasks: BackgroundTasks,
    pub(crate) registry: HookRegistry,
    pub(crate) diagnostics: Option<Arc<ValkeyAuthDiagnostics>>,
}

impl PoolHooks {
    pub(crate) fn metrics(&self) -> Option<SharedMetrics> {
        self.metrics.lock().ok().and_then(|slot| slot.clone())
    }

    /// Arm the reconnect / error hook pair on `client`, aborting whatever pair
    /// was armed on it before. No-op before `install_observability_hooks` has
    /// installed the metrics, and after `quit()` has taken the background
    /// tasks.
    fn rearm(&self, client: &Client, pool_label: &'static str) {
        self.rearm_with(client, pool_label, None);
    }

    /// [`Self::rearm`], with the error hook handing the client's fast
    /// unresponsive verdicts to `verdict_sink` (a member the member replacer
    /// can replace: a multiplexed member or a `txn` slot).
    pub(crate) fn rearm_with(
        &self,
        client: &Client,
        pool_label: &'static str,
        verdict_sink: Option<VerdictSink>,
    ) {
        let Some(metrics) = self.metrics() else {
            return;
        };
        // `background_tasks == None` is the shutdown sentinel — `quit()` takes
        // the vec before it closes anything.
        match self.background_tasks.lock() {
            Ok(tasks) if tasks.is_some() => {}
            _ => return,
        }

        let key = ClientLike::id(client).to_string();
        let armed = [
            spawn_reconnect_metric(client, pool_label, metrics.clone()),
            spawn_error_metric(
                client,
                pool_label,
                metrics,
                self.diagnostics.clone(),
                verdict_sink,
            ),
        ];

        let previous = match self.registry.lock() {
            Ok(mut registry) => registry.insert(key, armed),
            Err(_) => {
                // Poisoned registry: we cannot record the new pair, so we must
                // not leave it running unaccounted. Abort it and leave the
                // (still live) previous pair in place — stale but singular.
                for handle in armed {
                    handle.abort();
                }
                return;
            }
        };
        if let Some(stale) = previous {
            for handle in stale {
                handle.abort();
            }
        }
    }

    /// Abort and forget the hook pair armed on the client `client_id`, if any:
    /// its connection is being retired, and a pair left behind would count a
    /// discarded client's events (or, after a `QUIT` that never swapped fred's
    /// broadcast senders, never finish).
    pub(crate) fn disarm(&self, client_id: &str) {
        let pair = self
            .registry
            .lock()
            .ok()
            .and_then(|mut registry| registry.remove(client_id));
        for handle in pair.into_iter().flatten() {
            handle.abort();
        }
    }
}

/// One stream reader's connection (see [`ReaderConnector`]): a single-node
/// fred client, the node it targets on a cluster, and its router's
/// supervision. Hand it back to [`ReaderConnector::retire`] to end it.
#[derive(Debug)]
pub struct ReaderConn {
    client: Client,
    node: Option<Server>,
    /// Taken by the retire.
    watch: Option<RetirableWatch>,
    /// Whether this connection has added one to
    /// `live_edit_stream_mux_readers_connected`: at its connect when the
    /// metrics were already installed, otherwise when its hooks are armed
    /// once they are ([`ReaderConnector::arm_hooks`]). The retire subtracts
    /// one exactly when this is set, so each connection is counted at most
    /// once and the gauge never goes negative.
    counted: bool,
}

impl ReaderConn {
    /// The reader's client: one connection, plus fred's lazily opened
    /// backchannel when a `CLIENT UNBLOCK` interrupts its XREAD.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// On a cluster, the primary this connection targets (the owner of the
    /// reader's hash slot when it connected); `None` off a cluster.
    pub fn node(&self) -> Option<&Server> {
        self.node.as_ref()
    }
}

/// Why [`ReaderConnector::connect`] returned no connection.
#[derive(Debug)]
pub(crate) enum ReaderConnectError {
    /// The reader's cancellation (shutdown) arrived mid-connect; the
    /// half-started router was aborted.
    Cancelled,
    /// The connect failed or did not finish within
    /// `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`. `node` is the primary it tried
    /// (`None`: the URL's endpoint).
    Failed {
        node: Option<Server>,
        error: fred::error::Error,
    },
}

/// A test-injected outcome for one reader connect's setup: the
/// `MuxFaultSite::Connect` fault the reader popped for the attempt
/// (`storage::stream_mux`). Uninhabited without the test seams, so a
/// production connect carries none.
#[cfg(any(test, feature = "test-fault-injection"))]
pub(crate) type ReaderConnectFault = crate::storage::stream_mux::MuxFault;
/// See the `test-fault-injection` definition: uninhabited here.
#[cfg(not(any(test, feature = "test-fault-injection")))]
pub(crate) type ReaderConnectFault = std::convert::Infallible;

/// What a reader connect waits on for its new connection's setup: fred's
/// connect notification for `client`, subscribed on the first poll, so the
/// caller polls it once before it starts the router.
#[cfg(not(any(test, feature = "test-fault-injection")))]
fn reader_connect_setup(
    client: &Client,
    fault: Option<ReaderConnectFault>,
) -> impl std::future::Future<Output = Result<(), fred::error::Error>> + Send + '_ {
    if let Some(never) = fault {
        match never {}
    }
    ClientLike::wait_for_connect(client)
}

/// What a reader connect waits on for its new connection's setup: fred's
/// connect notification for `client`, subscribed on the first poll, so the
/// caller polls it once before it starts the router — unless `fault` (a
/// `MuxFaultSite::Connect` fault) replaces what it reports. An `Err` is
/// reported after one yield, so it reaches the caller's race against its
/// cancellation and bound as a real setup failure does; a `Hang` never
/// reports, so that bound cuts it. A `Hold` reports the real setup's outcome
/// once the hold is released: the notification and the release are awaited
/// together, so the first poll still subscribes before the router starts and
/// a setup that fails while the hold is pending is reported, not missed (a
/// subscription made after the release would see only a later attempt, and
/// the bound would cut the connect as a `Timeout`).
#[cfg(any(test, feature = "test-fault-injection"))]
async fn reader_connect_setup(
    client: &Client,
    fault: Option<ReaderConnectFault>,
) -> Result<(), fred::error::Error> {
    use crate::storage::stream_mux::MuxFault;
    match fault {
        None | Some(MuxFault::Pass) => ClientLike::wait_for_connect(client).await,
        Some(MuxFault::Err(e)) => {
            tokio::task::yield_now().await;
            Err(e)
        }
        Some(MuxFault::Hang) => std::future::pending().await,
        Some(MuxFault::Hold(release)) => {
            let (setup, _released) = tokio::join!(ClientLike::wait_for_connect(client), release);
            setup
        }
        Some(MuxFault::Reply(_) | MuxFault::Panic) => {
            unreachable!("push_faults_for_test refuses Reply and Panic at the Connect site")
        }
    }
}

/// What [`ReaderConnector::retire`] observed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RetireOutcome {
    /// The `QUIT` did not complete within `QUIT_TIMEOUT` (for the caller's
    /// WARN; the client's router is aborted either way). `false` when no
    /// `QUIT` was sent because the client was not connected.
    pub quit_overran: bool,
}

/// Builds, and retires, the stream readers' connections (`storage::stream_mux`).
///
/// Readers connect lazily — the first time each serves a subscription — so
/// `ValkeyClient::connect` opens none of them, and on a cluster each connects
/// only to the primary that owns its bound hash slot. Every reader client is a
/// **single-node** client built from one template: the reader `FredConfig`
/// `build_fred_config` derived from the shared one (its credential provider
/// cadence-wrapped so fred schedules no periodic re-`AUTH` on a reader, the
/// fourth sanctioned divergence — see `derive_reader_config`), with
/// `Blocking::Interrupt` set by `ValkeyClient::reader_builder` (the readers'
/// per-pool divergence), the shared reconnect policy, performance config and
/// connect-timeout settings. On a cluster URL each
/// connect makes the **third sanctioned divergence**, on a clone of that
/// template: `server` becomes the slot owner's `Server` — taken whole from the
/// readers' routing view (below), so it carries the `tls_server_name` fred mapped
/// under `TlsHostMapping::DefaultHost` and is never rebuilt from `host:port` —
/// and `database` is cleared, because fred skips `SELECT` only for clustered
/// configs and a cluster URL's database is meaningless. Credentials, the
/// credential provider and the TLS policy ride the clone untouched. Off a
/// cluster the template is used as it is.
///
/// **One routing view.** Every question the readers ask about the cluster's
/// topology — which primary owns a slot (a connect, the connection step, the
/// failover check before a forced reconnect, a new binding's node affinity),
/// whether a node is still a primary (the idle keepalive), and the re-sync
/// after a `-MOVED` — goes to the cached routing table of ONE data-pool
/// client: member 0's current client, resolved on every call, so a
/// replacement of that member (`storage::member_replacement`) moves the view
/// to the fresh client. Each fred client keeps its own routing cache, so a
/// sync on one pool member and a read from another would usually miss the
/// sync: the reader would keep reading the stale owner after a `MOVED`. With
/// one client, the sync refreshes exactly the view every later read uses.
///
/// **One sync at a time.** fred runs a `CLUSTER SLOTS` sync inside the routing
/// client's router task, stalling that data-pool member's traffic, and a sync
/// that finds the cluster healthy also replaces its backchannel connection (a
/// fresh TCP, TLS and `AUTH` handshake). So the readers' syncs are
/// single-flight ([`Self::sync_routing`]): one caller at a time issues one,
/// and a reader that waited while another reader's sync ended — answered,
/// or cut off by its bound — does not sync again: it reads the view that
/// sync left. While syncs answer within their bound, many readers redirected
/// `-MOVED` together therefore cost the routing client one sync, not one
/// each. A sync that outlives its bound (fred reconnecting the routing client
/// during a failover, under its connect timeout) cannot be taken back: fred
/// keeps it queued and runs it. Its cut still ends it for the readers, so
/// the ones queued behind it issue nothing; they re-resolve from the view as
/// it is and, finding the owner unchanged, take the transient remedy (a
/// budget charge and a backoff). So while syncs keep being cut, a new
/// `CLUSTER SLOTS` is issued at most once per bound period, never once per
/// reader.
///
/// A connection's router is supervised, and its metric hooks are armed, only
/// after its first connect succeeded (so a first connect is never counted as
/// a reconnect), and both end when it is retired.
pub(crate) struct ReaderConnector {
    /// The reader client template (see the type docs).
    builder: Builder,
    /// The unwrapped provider behind the template's cadence-wrapped one (the
    /// same `Arc` every pool carries), or `None` under `static` auth: what a
    /// reader consults to schedule and gate its own re-`AUTH`
    /// ([`Self::reauth_provider`]).
    reauth: Option<Arc<dyn CredentialProvider>>,
    cluster: bool,
    /// Bound on one connect: `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` (never
    /// zero). Also the routing-view sync's bound when the command timeout is
    /// disabled ([`routing_sync_bound`]).
    connect_timeout: Duration,
    /// The command timeout (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`; zero:
    /// disabled), which bounds the routing-view sync after a `MOVED` — see
    /// [`routing_sync_bound`].
    command_timeout: Duration,
    /// The readers' routing view: the data pool, whose member 0's current
    /// client's cached routing table names slot owners and primaries and which
    /// alone is re-synced after a `MOVED` (see the type docs). Resolved on
    /// every call ([`Self::routing_client`]).
    routing: Pool,
    /// Makes [`Self::sync_routing`] single-flight (see the type docs).
    routing_sync: RoutingSync,
    supervisor: Arc<RouterSupervisor>,
    hooks: PoolHooks,
}

/// The single-flight state of [`ReaderConnector::sync_routing`]: created with
/// the connector and living exactly as long as it (no task, nothing to tear
/// down).
#[derive(Debug, Default)]
struct RoutingSync {
    /// Held by the one caller whose sync is in flight, across its
    /// `CLUSTER SLOTS`; every other caller waits on it, in arrival order.
    /// Released when that caller's sync is answered or its bound cuts it off
    /// (or its caller abandons it: shutdown, a panic).
    lock: tokio::sync::Mutex<()>,
    /// How many syncs have ended since the connector was built: answered
    /// (successfully or not), cut off by their bound, or abandoned by their
    /// caller. A caller reads it before it waits on `lock`; when it has moved
    /// by the time the caller holds the lock, a sync ended while the caller
    /// waited, and the caller uses the view that sync left instead of
    /// issuing its own. A cut sync moves it too, although fred may still be
    /// running that sync: otherwise every caller queued behind it would
    /// issue its own `CLUSTER SLOTS` behind the orphaned one. It moves
    /// before `lock` is released, so no caller the release lets in can miss
    /// it.
    generation: AtomicU64,
}

/// The bound on one readers' routing-view sync after a `MOVED`: the command
/// timeout, or — when the command timeout is zero (disabled) — the connect
/// timeout (`LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`, never zero). The sync is
/// always bounded, so a reader's `MOVED` recovery never waits on it forever,
/// and a disabled command timeout never reads as a zero bound that would cut
/// every sync on its first poll.
fn routing_sync_bound(command_timeout: Duration, connect_timeout: Duration) -> Duration {
    if command_timeout.is_zero() {
        connect_timeout
    } else {
        command_timeout
    }
}

impl std::fmt::Debug for ReaderConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The builder's config carries credentials: never print it.
        f.debug_struct("ReaderConnector")
            .field("cluster", &self.cluster)
            .field("connect_timeout", &self.connect_timeout)
            .finish()
    }
}

impl ReaderConnector {
    /// The client whose routing table is the readers' routing view: data-pool
    /// member 0's current client.
    fn routing_client(&self) -> &Client {
        self.routing.member(0)
    }

    /// The fred client id of [`Self::routing_client`]. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn routing_client_id_for_test(&self) -> String {
        ClientLike::id(self.routing_client()).to_owned()
    }

    /// The primary that owns hash slot `slot` in the readers' routing view,
    /// taken whole — with the TLS server name fred mapped onto it — or `None`
    /// off a cluster, while the view is absent, or when it does not cover the
    /// slot. Clones the routing table, so not for hot paths.
    pub(crate) fn slot_owner(&self, slot: u16) -> Option<Server> {
        self.routing_client()
            .cached_cluster_state()
            .and_then(|routing| routing.get_server(slot).cloned())
    }

    /// Whether the readers' routing view lists `node` as a primary; `None`
    /// when there is no view to ask.
    pub(crate) fn is_primary(&self, node: &Server) -> Option<bool> {
        self.routing_client()
            .cached_cluster_state()
            .map(|routing| routing.unique_primary_nodes().contains(node))
    }

    /// Re-sync the readers' routing view from the cluster (`CLUSTER SLOTS` on
    /// the routing client), single-flight across every reader (see the type
    /// docs).
    ///
    /// The caller waits for any sync in flight, then syncs only when none
    /// ended while it waited; otherwise it returns `Ok` at once and reads the
    /// view that sync left — refreshed, or unchanged when that sync failed or
    /// was cut off. The whole call, the wait included, is bounded by
    /// [`routing_sync_bound`] (the command timeout, or the connect timeout
    /// when the command timeout is disabled); past it the call fails
    /// `Timeout`. A sync cut off by its bound counts as ended for every
    /// caller queued behind it, although fred keeps running it in the pinned
    /// client's router (its queued `SyncCluster` cannot be taken back): those
    /// callers issue none of their own, so while syncs keep being cut a new
    /// one is issued at most once per bound period.
    pub(crate) async fn sync_routing(&self) -> Result<(), fred::error::Error> {
        /// Moves the generation when dropped: when the sync is answered, and
        /// equally when the bound (or the caller's cancellation) drops it
        /// mid-flight.
        struct SyncEnded<'a>(&'a AtomicU64);
        impl Drop for SyncEnded<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::AcqRel);
            }
        }

        let bound = routing_sync_bound(self.command_timeout, self.connect_timeout);
        let single_flight = async {
            let seen = self.routing_sync.generation.load(Ordering::Acquire);
            let _in_flight = self.routing_sync.lock.lock().await;
            if self.routing_sync.generation.load(Ordering::Acquire) != seen {
                return Ok(());
            }
            // Declared after the lock guard, so it drops first: the
            // generation moves before the lock is handed to the next caller,
            // whether the sync is answered or cut off.
            let _ended = SyncEnded(&self.routing_sync.generation);
            self.routing_client().sync_cluster().await
        };
        match tokio::time::timeout(bound, single_flight).await {
            Ok(result) => result,
            Err(_) => Err(fred::error::Error::new(
                FredErrorKind::Timeout,
                format!(
                    "the cluster routing sync did not finish within {} ms",
                    bound.as_millis()
                ),
            )),
        }
    }

    /// How many readers' routing-view syncs have ended on this connector: one
    /// per `CLUSTER SLOTS` that [`Self::sync_routing`] actually issued, once
    /// fred answered it (successfully or not) or its bound cut it off (or its
    /// caller was cancelled mid-sync). Not a count of answered syncs only. A
    /// call that found a sync ended while it waited adds none.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn routing_syncs_finished_for_test(&self) -> u64 {
        self.routing_sync.generation.load(Ordering::Acquire)
    }

    /// Holds the single-flight lock of [`Self::sync_routing`] as an in-flight
    /// sync would, so a test can queue callers behind it. Dropping the guard
    /// releases them.
    #[cfg(test)]
    pub(crate) async fn hold_routing_sync_for_test(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.routing_sync.lock.lock().await
    }

    /// Records a sync as ended, as an in-flight sync does when fred answers
    /// it or its bound cuts it off, without issuing one.
    #[cfg(test)]
    pub(crate) fn finish_routing_sync_for_test(&self) {
        self.routing_sync.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Connect a client for reader `reader`: on a cluster to the primary that
    /// owns `slot` in the readers' routing view, otherwise to the URL's
    /// endpoint. `cause` (`bind` | `retarget`) labels
    /// `live_edit_stream_mux_reader_connects_total`.
    ///
    /// The connect is bounded by `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` and
    /// raced against `cancel`; a connect that fails, times out or is
    /// cancelled aborts the router it started, so it leaves no task behind.
    /// Only a connect that succeeded registers the router for supervision.
    /// The caller arms the hooks ([`Self::arm_hooks`]) once it has published
    /// the connection.
    ///
    /// `fault` is always `None` outside the test seams. With them, it is the
    /// `MuxFaultSite::Connect` fault the reader popped for this attempt,
    /// which replaces, or (a `Hold`) delays, only what the new connection's
    /// setup reports ([`reader_connect_setup`]); the client is built, the
    /// router started, and the outcome counted, hinted and cleaned up as for
    /// a real connect.
    pub(crate) async fn connect(
        &self,
        reader: usize,
        slot: Option<u16>,
        cause: &'static str,
        fault: Option<ReaderConnectFault>,
        cancel: &CancellationToken,
    ) -> Result<ReaderConn, ReaderConnectError> {
        let mut result = self.connect_inner(reader, slot, fault, cancel).await;
        let outcome = match &result {
            Ok(_) => Some(crate::observability::metrics::READER_CONNECT_RESULT_OK),
            Err(ReaderConnectError::Failed { error, .. }) => {
                // A credential the server refused, on a connection no hook
                // watches yet: the steady-state hint is the operator's only
                // explanation of it. Rate-limited process-wide.
                // `WRONGPASS` only, as in `spawn_error_metric`: this is a
                // running process, where a `NOAUTH` is the session reset
                // `reconnect_on_auth_error` exists to absorb, not the
                // misconfiguration the checklist diagnoses.
                if is_server_auth_rejection(error) && error.details().starts_with("WRONGPASS") {
                    if let Some(diagnostics) = self.hooks.diagnostics.as_ref() {
                        diagnostics.emit_steady_state_hint(POOL_LABEL_SUBSCRIBER);
                    }
                }
                Some(crate::observability::metrics::READER_CONNECT_RESULT_FAILED)
            }
            Err(ReaderConnectError::Cancelled) => None,
        };
        if let (Some(outcome), Some(metrics)) = (outcome, self.hooks.metrics()) {
            metrics.inc_stream_mux_reader_connect(cause, outcome);
        }
        if let Ok(conn) = result.as_mut() {
            self.count_connected(conn);
        }
        result
    }

    /// Count `conn` once on `live_edit_stream_mux_readers_connected`, if it is
    /// not counted yet and the metrics are installed; [`Self::retire`]
    /// subtracts it again exactly when it was counted. Called by the connect
    /// and again by [`Self::arm_hooks`], so a connection made before
    /// `install_observability_hooks` is counted once the metrics exist —
    /// whichever of the two calls sees them first counts it, and the other
    /// finds it counted.
    fn count_connected(&self, conn: &mut ReaderConn) {
        if conn.counted {
            return;
        }
        if let Some(metrics) = self.hooks.metrics() {
            metrics.add_stream_mux_readers_connected(1);
            conn.counted = true;
        }
    }

    async fn connect_inner(
        &self,
        reader: usize,
        slot: Option<u16>,
        fault: Option<ReaderConnectFault>,
        cancel: &CancellationToken,
    ) -> Result<ReaderConn, ReaderConnectError> {
        let mut builder = self.builder.clone();
        let node = if self.cluster {
            let owner = slot.and_then(|slot| self.slot_owner(slot));
            let Some(owner) = owner else {
                return Err(ReaderConnectError::Failed {
                    node: None,
                    error: fred::error::Error::new(
                        FredErrorKind::Cluster,
                        format!(
                            "the cluster routing view names no owner for hash slot {}",
                            slot.map_or_else(|| "(none)".to_owned(), |slot| slot.to_string())
                        ),
                    ),
                });
            };
            let server = owner.clone();
            // The third sanctioned divergence (see the type docs).
            builder.with_config(|config| {
                config.server = ServerConfig::Centralized { server };
                config.database = None;
            });
            Some(owner)
        } else {
            None
        };
        let failed = |error: fred::error::Error| ReaderConnectError::Failed {
            node: node.clone(),
            error,
        };
        let client = builder.build().map_err(failed)?;

        // Subscribed to the connect notification before the router can send
        // it (a new client reads `Disconnected`, so this first poll subscribes),
        // and the router's handle kept from the start, so a connect that is
        // cut off can still be aborted — `init()` would drop it.
        let (router, outcome) = {
            let mut connected = std::pin::pin!(reader_connect_setup(&client, fault));
            let before = futures::poll!(connected.as_mut());
            let router = ClientLike::connect(&client);
            let outcome = match before {
                std::task::Poll::Ready(result) => Some(result),
                std::task::Poll::Pending => tokio::select! {
                    biased;
                    () = cancel.cancelled() => None,
                    result = tokio::time::timeout(self.connect_timeout, connected) => Some(
                        result.unwrap_or_else(|_| {
                            Err(fred::error::Error::new(
                                FredErrorKind::Timeout,
                                format!(
                                    "the reader connection was not up within {} s \
                                     (LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS)",
                                    self.connect_timeout.as_secs()
                                ),
                            ))
                        }),
                    ),
                },
            };
            (router, outcome)
        };
        match outcome {
            Some(Ok(())) if !cancel.is_cancelled() => {}
            Some(Ok(())) | None => {
                router.abort();
                return Err(ReaderConnectError::Cancelled);
            }
            Some(Err(error)) => {
                router.abort();
                return Err(failed(error));
            }
        }

        let restart_client = client.clone();
        let watch = self.supervisor.watch_retirable(
            POOL_LABEL_SUBSCRIBER,
            reader,
            ClientLike::id(&client),
            router,
            RouterRestart::Connect(Box::new(move || ClientLike::connect(&restart_client))),
        );
        Ok(ReaderConn {
            client,
            node,
            watch: Some(watch),
            counted: false,
        })
    }

    /// Arm the metric hooks (`valkey_reconnects_total` /
    /// `valkey_errors_total{pool="subscriber_pool"}`) on a reader connection,
    /// once it is published, and count it on
    /// `live_edit_stream_mux_readers_connected` if its connect could not (the
    /// metrics were not installed yet). Idempotent per connection; both are
    /// no-ops before `install_observability_hooks` installed the metrics. The
    /// caller holds the connection in the reader's slot, under the slot's
    /// lock, so it is never armed or counted after a retire took it.
    pub(crate) fn arm_hooks(&self, conn: &mut ReaderConn) {
        self.hooks.rearm(&conn.client, POOL_LABEL_SUBSCRIBER);
        self.count_connected(conn);
    }

    /// The provider a reader re-`AUTH`s its own connection from (b.19i): the
    /// same `Arc` the pools carry, unwrapped, so its `refresh_interval()`
    /// schedules the reader's ticks and its `fetch(None)` applies the same
    /// gate fred's refresh task applies on the pools. `None` under `static`
    /// auth, where no reader schedules a re-`AUTH`.
    pub(crate) fn reauth_provider(&self) -> Option<&Arc<dyn CredentialProvider>> {
        self.reauth.as_ref()
    }

    /// End a reader connection: retire its router's supervision first (so
    /// nothing below is read as a router death, counted or restarted), disarm
    /// its hooks, `QUIT` it bounded by `QUIT_TIMEOUT`, then abort its router
    /// whatever the `QUIT` did — it answered, failed or overran (an abort of
    /// a router that has already ended does nothing). Unsupervised now, a
    /// router left running after a failed `QUIT` would keep reconnecting and
    /// re-`AUTH`ing a client nothing uses; this is the same unconditional
    /// abort a `txn` slot's repair makes between its `quit()` and `connect()`.
    ///
    /// A client that is not `Connected` gets no `QUIT`: its router is aborted
    /// at once. That is the failover retarget's case — the old primary is
    /// gone and fred is in its reconnect loop, where a `QUIT` could only wait
    /// out `QUIT_TIMEOUT` and so hold up the connect to the new owner. No
    /// `QUIT` sent means none overran.
    pub(crate) async fn retire(&self, mut conn: ReaderConn) -> RetireOutcome {
        let router = match conn.watch.take() {
            Some(watch) => Some(self.supervisor.retire(watch).await),
            None => None,
        };
        self.hooks.disarm(ClientLike::id(&conn.client));
        let quit_overran = conn.client.state() == ClientState::Connected
            && tokio::time::timeout(QUIT_TIMEOUT, ClientLike::quit(&conn.client))
                .await
                .is_err();
        if let Some(router) = router {
            router.abort();
        }
        if conn.counted {
            if let Some(metrics) = self.hooks.metrics() {
                metrics.add_stream_mux_readers_connected(-1);
            }
        }
        RetireOutcome { quit_overran }
    }

    /// A connector that never connects anything useful: a default centralized
    /// template and fresh, empty supervision and hooks. For unit tests of the
    /// reader loop's pure decisions.
    #[cfg(test)]
    pub(crate) fn unconnected_for_test(routing: Pool, cluster: bool) -> Arc<Self> {
        let metrics: Arc<Mutex<Option<SharedMetrics>>> = Arc::new(Mutex::new(None));
        Arc::new(Self {
            builder: Builder::default_centralized(),
            reauth: None,
            cluster,
            connect_timeout: Duration::from_secs(1),
            command_timeout: Duration::from_secs(1),
            routing,
            routing_sync: RoutingSync::default(),
            supervisor: Arc::new(RouterSupervisor::new(Arc::clone(&metrics))),
            hooks: PoolHooks {
                metrics,
                background_tasks: Arc::new(Mutex::new(Some(Vec::new()))),
                registry: Arc::new(Mutex::new(HashMap::new())),
                diagnostics: None,
            },
        })
    }
}

/// Resolve the effective maintenance pool size. Defaults to 1 (the absolute
/// minimum a fred `Pool` accepts) which is sufficient because every task that
/// uses the pool for long walks (the GC sweep, the orphan sweep, the
/// reference-count reconciler, the external-value backup repair)
/// is gated behind the `le:gc:leader` lock, so normally one replica runs them
/// at a time (two can briefly overlap after a lock takeover). Operators can
/// raise this for very large keyspaces, but values above ~4 are rarely
/// useful given the leader-only single-threaded sweep loop.
pub(crate) fn effective_maintenance_pool_size(configured: u32) -> usize {
    (configured as usize).max(1)
}

/// Resolve the effective pool size: if config says 0 (auto), use cores × 2.
///
/// The auto branch is floored at `STORE_VALUES_MAX_INFLIGHT` (currently 8)
/// to match the b.avd second-follow-up startup-validation rule: a single
/// `ValkeyExternalValueStore::store_values` wave issues up to that many
/// concurrent blob writes, and post-b.rej they ride the multiplexed DATA
/// pool this function sizes. A multi-MB blob occupies the connection fred
/// routed it onto for its whole transfer, so the invariant the floor buys
/// is that the pool is AT LEAST AS WIDE AS one wave — the wave can never
/// exceed the pool, starve the txn/subscriber tokens, or wedge fred's queue
/// with blobs that have no connection to land on. It does NOT buy headroom:
/// at exactly the floor a full wave can momentarily carry a blob on every
/// connection, with concurrent reads and Lua fast-path commits queued behind
/// them. On a host reporting more than 4 cores (auto = 10+) the floor never
/// triggers; at or below 4 — a small dev box, or, since
/// `available_parallelism` honours the cgroup CPU quota, a CPU-limited
/// container on a large node — `(cores * 2).max(8)` resolves to exactly the
/// floor, 8, so the invariant holds without the operator having to set the
/// variable explicitly. Operators who want an
/// explicit value go through `Config::validate_pool_size_vs_store_values_inflight`,
/// which rejects any non-zero `LIVE_EDIT_VALKEY_POOL_SIZE` below the same
/// floor and accepts equality — production should sit well above it (the
/// Helm chart sets 64).
pub(crate) fn effective_pool_size(configured: u32) -> usize {
    if configured == 0 {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        (cores * 2).max(crate::storage::external_values::STORE_VALUES_MAX_INFLIGHT)
    } else {
        configured as usize
    }
}

/// Resolve the stream reader connection count. Zero means auto: `cores × 2`,
/// matching `effective_pool_size`. Subscriptions share readers, so this does
/// not bound the subscription count; in Valkey Cluster it bounds how many
/// distinct layers (hash slots) a replica reads concurrently.
pub(crate) fn effective_subscriber_pool_size(configured: u32) -> usize {
    if configured == 0 {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        (cores * 2).max(1)
    } else {
        configured as usize
    }
}

/// Strip credentials from a URL for safe logging.
///
/// Removes userinfo (everything between `://` and the last `@`) so passwords
/// in URLs like `redis://:secret@host:6379` are never written to logs. Uses
/// the *last* `@` to handle passwords that contain literal `@` characters.
/// Returns the original string unchanged if no `@` is present or parsing fails.
fn redact_url_credentials(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let after_scheme = scheme_end + 3; // position right after "://"
    let rest = &url[after_scheme..];
    let Some(at_pos) = rest.rfind('@') else {
        return url.to_string(); // no userinfo, nothing to redact
    };
    // Rebuild: scheme + "://" + everything after the '@'
    format!("{}://{}", &url[..scheme_end], &rest[at_pos + 1..])
}

#[cfg(test)]
mod tests {
    use aws_credential_types::Credentials;
    use tokio_util::sync::CancellationToken;

    use super::*;

    #[test]
    fn redact_strips_password() {
        assert_eq!(
            redact_url_credentials("redis://:secret@host:6379"),
            "redis://host:6379"
        );
    }

    #[test]
    fn redact_strips_user_and_password() {
        assert_eq!(
            redact_url_credentials("redis://user:pass@host:6379/0"),
            "redis://host:6379/0"
        );
    }

    #[test]
    fn redact_no_credentials_unchanged() {
        assert_eq!(
            redact_url_credentials("redis://host:6379"),
            "redis://host:6379"
        );
    }

    #[test]
    fn redact_no_scheme_unchanged() {
        assert_eq!(redact_url_credentials("host:6379"), "host:6379");
    }

    #[test]
    fn redact_empty_userinfo() {
        assert_eq!(
            redact_url_credentials("redis://@host:6379"),
            "redis://host:6379"
        );
    }

    #[test]
    fn redact_rediss_tls_scheme_strips_credentials() {
        assert_eq!(
            redact_url_credentials("rediss://:secret@host:6380"),
            "rediss://host:6380"
        );
    }

    #[test]
    fn redact_valkey_scheme_strips_credentials() {
        assert_eq!(
            redact_url_credentials("valkey://admin:hunter2@valkey.internal:6379/0"),
            "valkey://valkey.internal:6379/0"
        );
    }

    #[test]
    fn redact_multiple_at_signs_strips_to_last_at() {
        // A password containing a literal '@' (e.g. "p@ss") produces two '@'
        // signs. The function strips to the last '@' so no password material leaks.
        assert_eq!(
            redact_url_credentials("redis://user:p@ss@host:6379"),
            "redis://host:6379"
        );
    }

    // ── b.1dz transient-error classification ──────────────────────────────
    //
    // `is_transient_fred_error` and `map_snapshot_load_err` are the two
    // helpers added by b.1dz to turn brief Valkey I/O blips during the
    // `update_layer_once_watch` snapshot-load / MULTI-queue phases into retries
    // instead of user-visible `Internal` errors. The tests below pin the
    // classification table: each transient kind must return `true` /
    // `Aborted("transient I/O: ...")`, each non-transient kind must return
    // `false` / `Internal(...)`. A regression on either helper would let a
    // real snapshot-load timeout surface as `Internal`, which the retry
    // loop does not match and which is the precise failure mode b.1dz
    // protects against.

    fn mk_err(kind: FredErrorKind) -> fred::error::Error {
        fred::error::Error::new(kind, "synthetic test error")
    }

    // ── FredErrOrigin::observe_error ────────────────────────────────────
    //
    // The command-timeout cell of `valkey_errors_total` is fed only from
    // here, so the kind filter is the whole contract: `Timeout` ticks exactly
    // its origin's pool, and every other kind — `IO` and `Canceled` above all,
    // whose connection event fred already reports on the hook — ticks nothing.

    /// Every fred `ErrorKind` this build can construct, `Timeout` included.
    /// (`Tls` exists only when fred's TLS feature is on, i.e. under
    /// `valkey-tls`.)
    fn every_fred_error_kind() -> Vec<FredErrorKind> {
        vec![
            FredErrorKind::Config,
            FredErrorKind::Auth,
            FredErrorKind::Routing,
            FredErrorKind::IO,
            FredErrorKind::InvalidCommand,
            FredErrorKind::InvalidArgument,
            FredErrorKind::Url,
            FredErrorKind::Protocol,
            #[cfg(feature = "valkey-tls")]
            FredErrorKind::Tls,
            FredErrorKind::Canceled,
            FredErrorKind::Unknown,
            FredErrorKind::Timeout,
            FredErrorKind::Cluster,
            FredErrorKind::Parse,
            FredErrorKind::Sentinel,
            FredErrorKind::NotFound,
            FredErrorKind::Backpressure,
        ]
    }

    /// `valkey_errors_total{pool, kind="command_timeout"}` for every pool.
    fn command_timeout_cells(
        metrics: &crate::observability::metrics::Metrics,
    ) -> Vec<(String, f64)> {
        use crate::observability::metrics::test_util::{render, sample_value};
        let text = render(metrics);
        crate::observability::metrics::VALKEY_POOLS
            .iter()
            .map(|pool| {
                let v = sample_value(
                    &text,
                    "live_edit_valkey_errors_total",
                    &[("pool", pool), ("kind", "command_timeout")],
                )
                .expect("every pool's command_timeout cell is seeded in Metrics::new");
                ((*pool).to_owned(), v)
            })
            .collect()
    }

    fn only_pool(pool: &str, value: f64) -> Vec<(String, f64)> {
        crate::observability::metrics::VALKEY_POOLS
            .iter()
            .map(|p| ((*p).to_owned(), if *p == pool { value } else { 0.0 }))
            .collect()
    }

    #[test]
    fn observe_error_ticks_exactly_its_pools_command_timeout_cell_for_a_timeout() {
        for pool in crate::observability::metrics::VALKEY_POOLS {
            let metrics: SharedMetrics = Arc::new(
                crate::observability::metrics::metrics_scrape_only("count-timeout"),
            );
            FredErrOrigin::metrics(
                Some(&metrics),
                pool,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            )
            .observe_error(&mk_err(FredErrorKind::Timeout));
            assert_eq!(
                command_timeout_cells(&metrics),
                only_pool(pool, 1.0),
                "one Timeout on pool \"{pool}\" must tick exactly that pool's cell, once"
            );
        }
    }

    #[test]
    fn observe_error_ticks_nothing_for_every_other_kind() {
        let metrics: SharedMetrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "count-timeout",
        ));
        for kind in every_fred_error_kind() {
            if matches!(kind, FredErrorKind::Timeout) {
                continue;
            }
            FredErrOrigin::metrics(
                Some(&metrics),
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            )
            .observe_error(&mk_err(kind.clone()));
            assert_eq!(
                command_timeout_cells(&metrics),
                only_pool(POOL_LABEL_DATA, 0.0),
                "{kind:?} is not a command timeout and must tick nothing. IO and Canceled in \
                 particular are already counted from fred's connection-event hook; counting them \
                 here would double-count one outage"
            );
        }
    }

    /// An origin with no metrics handle is a silent no-op, not a panic — the
    /// shape every caller that runs without metrics (unit tests, the mapper
    /// test harnesses) passes.
    #[test]
    fn observe_error_without_metrics_does_nothing() {
        FredErrOrigin::metrics(
            None,
            POOL_LABEL_TXN,
            crate::storage::store_oom::StoreOomSignal::unwired(),
        )
        .observe_error(&mk_err(FredErrorKind::Timeout));
    }

    /// Each of the three mappers counts a `Timeout` exactly once on its origin
    /// and returns the very `ServiceError` a non-counting origin gets, for
    /// every kind. The counting argument is observability only; a mapper
    /// whose output moved with it would change what clients see.
    #[test]
    fn mappers_count_a_timeout_once_and_return_the_same_error_for_every_kind() {
        type Mapper = fn(FredErrOrigin<'_>, &str, fred::error::Error) -> crate::error::ServiceError;
        let mappers: [(&str, Mapper); 3] = [
            ("map_snapshot_load_err", map_snapshot_load_err),
            ("map_transient_fred_err", map_transient_fred_err),
            ("map_fred_err_for_grpc", map_fred_err_for_grpc),
        ];
        for (name, mapper) in mappers {
            for kind in every_fred_error_kind() {
                let metrics: SharedMetrics =
                    Arc::new(crate::observability::metrics::metrics_scrape_only("mapper"));
                let counted = mapper(
                    FredErrOrigin::metrics(
                        Some(&metrics),
                        POOL_LABEL_MAINTENANCE,
                        crate::storage::store_oom::StoreOomSignal::unwired(),
                    ),
                    "op",
                    mk_err(kind.clone()),
                );
                let uncounted = mapper(
                    FredErrOrigin::metrics(
                        None,
                        POOL_LABEL_MAINTENANCE,
                        crate::storage::store_oom::StoreOomSignal::unwired(),
                    ),
                    "op",
                    mk_err(kind.clone()),
                );
                // `ServiceError` is not `PartialEq`; its `Debug` form carries
                // the variant and the whole message.
                assert_eq!(
                    format!("{counted:?}"),
                    format!("{uncounted:?}"),
                    "{name}({kind:?}): the counting origin must not change the returned error"
                );
                let expected = if matches!(kind, FredErrorKind::Timeout) {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(
                    command_timeout_cells(&metrics),
                    only_pool(POOL_LABEL_MAINTENANCE, expected),
                    "{name}({kind:?}) must tick the command_timeout cell {expected} times"
                );
            }
        }
    }

    #[test]
    fn is_transient_fred_error_timeout_true() {
        assert!(is_transient_fred_error(&mk_err(FredErrorKind::Timeout)));
    }

    #[test]
    fn is_transient_fred_error_io_true() {
        assert!(is_transient_fred_error(&mk_err(FredErrorKind::IO)));
    }

    #[test]
    fn is_transient_fred_error_canceled_true() {
        assert!(is_transient_fred_error(&mk_err(FredErrorKind::Canceled)));
    }

    #[test]
    fn is_transient_fred_error_cluster_true() {
        assert!(is_transient_fred_error(&mk_err(FredErrorKind::Cluster)));
    }

    #[test]
    fn is_transient_fred_error_backpressure_true() {
        assert!(is_transient_fred_error(&mk_err(
            FredErrorKind::Backpressure
        )));
    }

    #[test]
    fn is_transient_fred_error_protocol_false() {
        // Protocol errors signal a malformed frame — a real bug, never retry.
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::Protocol)));
    }

    #[test]
    fn is_transient_fred_error_parse_false() {
        // Parse errors are deserialization bugs — never retry.
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::Parse)));
    }

    #[test]
    fn is_transient_fred_error_auth_false() {
        // Auth errors are fatal; retrying is pointless and masks config bugs.
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::Auth)));
    }

    #[test]
    fn is_transient_fred_error_config_false() {
        // Config errors shut the client down and retry can't help.
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::Config)));
    }

    #[test]
    fn is_transient_fred_error_url_false() {
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::Url)));
    }

    #[test]
    fn is_transient_fred_error_invalid_argument_false() {
        // Bad command at fred layer — programmer error, don't retry.
        assert!(!is_transient_fred_error(&mk_err(
            FredErrorKind::InvalidArgument
        )));
    }

    #[test]
    fn is_transient_fred_error_invalid_command_false() {
        assert!(!is_transient_fred_error(&mk_err(
            FredErrorKind::InvalidCommand
        )));
    }

    #[test]
    fn is_transient_fred_error_routing_false() {
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::Routing)));
    }

    #[test]
    fn is_transient_fred_error_sentinel_false() {
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::Sentinel)));
    }

    #[test]
    fn is_transient_fred_error_not_found_false() {
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::NotFound)));
    }

    #[test]
    fn is_transient_fred_error_unknown_false() {
        // "Unknown" is ambiguous — silent retry would hide real issues.
        assert!(!is_transient_fred_error(&mk_err(FredErrorKind::Unknown)));
    }

    // ── store_write_refusal (b.yvh) ─────────────────────────────────────────
    //
    // The four write-gating refusals as Valkey words them (`server.c`
    // `shared.oomerr`, `shared.bgsaveerr`, `shared.roslaveerr`,
    // `shared.noreplicaserr`), as fred hands them over: `ErrorKind::Unknown`
    // with the whole reply in `details`.

    const STORE_REFUSAL_REPLIES: [(&str, StoreWriteRefusal); 4] = [
        (
            "OOM command not allowed when used memory > 'maxmemory'.",
            StoreWriteRefusal::OutOfMemory,
        ),
        (
            "MISCONF Valkey is configured to save RDB snapshots, but it's currently unable to \
             persist to disk. Commands that may modify the data set are disabled, because this \
             instance is configured to report errors during writes if RDB snapshotting fails \
             (stop-writes-on-bgsave-error option). Please check the Valkey logs for details \
             about the RDB error.",
            StoreWriteRefusal::Misconf,
        ),
        (
            "READONLY You can't write against a read only replica.",
            StoreWriteRefusal::ReadOnly,
        ),
        (
            "NOREPLICAS Not enough good replicas to write.",
            StoreWriteRefusal::NoReplicas,
        ),
    ];

    fn unknown_err(details: &str) -> fred::error::Error {
        fred::error::Error::new(FredErrorKind::Unknown, details.to_owned())
    }

    #[test]
    fn store_write_refusal_recognises_each_refusal_bare_and_after_the_execabort_prefix() {
        for (reply, refusal) in STORE_REFUSAL_REPLIES {
            assert_eq!(
                store_write_refusal(&unknown_err(reply)),
                Some(refusal),
                "the bare refusal {reply:?}"
            );
            let discarded = format!("EXECABORT Transaction discarded because of: {reply}");
            assert_eq!(
                store_write_refusal(&unknown_err(&discarded)),
                Some(refusal),
                "an EXEC rejected as a whole by {reply:?}"
            );
        }
    }

    #[test]
    fn store_write_refusal_ignores_every_other_reply_and_every_other_kind() {
        // Anti-creep: prefix-anchored with the token's trailing space, and
        // gated on `ErrorKind::Unknown`.
        for reply in [
            "ERR OOM command not allowed when used memory > 'maxmemory'.",
            "OOMKILLED",
            "READONLYX something",
            "NOREPLICAS",
            "LOADING Valkey is loading the dataset in memory",
            "BUSY Valkey is busy running a script.",
            "EXECABORT Transaction discarded because of previous errors.",
            "EXECABORT Transaction discarded because of: ERR unknown command",
            "STREAM_ID_COLLISION: XADD 3-0 rejected",
            " OOM command not allowed",
            "",
        ] {
            assert_eq!(
                store_write_refusal(&unknown_err(reply)),
                None,
                "{reply:?} is not a store write refusal"
            );
        }
        for kind in [
            FredErrorKind::InvalidArgument,
            FredErrorKind::Protocol,
            FredErrorKind::IO,
            FredErrorKind::Timeout,
        ] {
            let err = fred::error::Error::new(
                kind.clone(),
                "OOM command not allowed when used memory > 'maxmemory'.",
            );
            assert_eq!(
                store_write_refusal(&err),
                None,
                "a {kind:?} error is not a store reply, whatever its text"
            );
        }
    }

    #[test]
    fn store_write_refusal_answers_unavailable_saying_nothing_was_applied() {
        let want = [
            (
                StoreWriteRefusal::OutOfMemory,
                "the backing store refused a write this request needed because it is out of \
                 memory; nothing was applied, retry",
            ),
            (
                StoreWriteRefusal::Misconf,
                "the backing store refused a write this request needed because it cannot persist \
                 data to disk; nothing was applied, retry",
            ),
            (
                StoreWriteRefusal::ReadOnly,
                "the backing store refused a write this request needed because the node it \
                 reached is a read-only replica; nothing was applied, retry",
            ),
            (
                StoreWriteRefusal::NoReplicas,
                "the backing store refused a write this request needed because too few of its \
                 replicas are reachable to accept writes; nothing was applied, retry",
            ),
        ];
        for (refusal, message) in want {
            match refusal.into_service_error() {
                crate::error::ServiceError::UnavailableWithDetail { message: got, .. } => {
                    assert_eq!(got, message, "{refusal:?}");
                    assert!(
                        !got.starts_with(crate::storage::layer::OPERATION_OVERTAKEN_MESSAGE),
                        "a client matches the overtaken prefix; a store refusal must never \
                         read as one"
                    );
                }
                other => panic!("{refusal:?} must answer UNAVAILABLE; got {other:?}"),
            }
        }
    }

    #[test]
    fn store_write_refusal_reason_labels_are_the_counter_vocabulary() {
        let labels: Vec<&str> = STORE_REFUSAL_REPLIES
            .iter()
            .map(|(_, refusal)| refusal.reason_label())
            .collect();
        assert_eq!(
            labels,
            crate::observability::metrics::STORE_WRITE_REFUSED_REASONS,
            "one distinct reason label per refusal, every one in the counter's vocabulary"
        );
    }

    #[test]
    fn is_transient_fred_error_keeps_every_store_write_refusal_permanent() {
        // b.xd1 made the store's not-serving `-LOADING` / `-BUSY` transient
        // and deliberately left these four out: they can persist, and folding
        // a refusal into the classifier would make the update retry loop spin
        // against a full store. The mappers answer them a plain `Unavailable`
        // instead (pinned in `transient_classification_test.rs` §5e).
        for (reply, _) in STORE_REFUSAL_REPLIES {
            assert!(
                !is_transient_fred_error(&unknown_err(reply)),
                "{reply:?} must stay non-transient"
            );
        }
    }

    #[test]
    fn refuse_store_write_ticks_its_stage_and_reason_cell_once_and_answers_unavailable() {
        use crate::observability::metrics::test_util::{render, sample_value};
        use crate::observability::metrics::{
            STORE_WRITE_REFUSED_REASONS, STORE_WRITE_REFUSED_STAGES,
            STORE_WRITE_REFUSED_STAGE_EXT_STORE,
        };
        let metrics: SharedMetrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "refuse-store-write",
        ));
        let origin = || {
            FredErrOrigin::metrics(
                Some(&metrics),
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            )
        };
        let refused = refuse_store_write(
            origin(),
            STORE_WRITE_REFUSED_STAGE_EXT_STORE,
            &unknown_err("NOREPLICAS Not enough good replicas to write."),
        );
        assert!(
            matches!(
                refused,
                Some(crate::error::ServiceError::UnavailableWithDetail { .. })
            ),
            "a refusal answers UNAVAILABLE; got {refused:?}"
        );
        assert!(
            refuse_store_write(
                origin(),
                STORE_WRITE_REFUSED_STAGE_EXT_STORE,
                &unknown_err("ERR something else"),
            )
            .is_none(),
            "anything else is left to the caller's own classifier, uncounted"
        );
        let text = render(&metrics);
        for stage in STORE_WRITE_REFUSED_STAGES {
            for reason in STORE_WRITE_REFUSED_REASONS {
                let want = if (*stage, *reason) == ("ext_store", "noreplicas") {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_store_write_refused_total",
                        &[("stage", stage), ("reason", reason)],
                    ),
                    Some(want),
                    "stage={stage} reason={reason}"
                );
            }
        }
    }

    /// b.pa1: of the store write refusals and a command timeout, only `-OOM`
    /// — bare, or after an `EXEC`'s discard prefix — notes the store
    /// out-of-memory signal, through `observe_error` and through
    /// `refuse_store_write` alike, on an origin naming its signal
    /// (`FredErrOrigin::metrics`). Each case and each path gets a signal of
    /// its own: a signal acts on at most one note per
    /// `STORE_OOM_REQUEST_MIN_INTERVAL`, so a shared one would hide every note
    /// after the first. The signals are unarmed, so a note only wakes.
    #[tokio::test]
    async fn only_an_out_of_memory_refusal_notes_the_store_out_of_memory_signal() {
        use crate::observability::metrics::STORE_WRITE_REFUSED_STAGE_EXT_STORE;
        use crate::storage::store_oom::StoreOomSignal;
        use futures::FutureExt;

        let fresh_signal = || {
            StoreOomSignal::new(
                Builder::default_centralized()
                    .build_pool(1)
                    .expect("an unconnected pool builds"),
            )
        };
        let execabort_oom = format!("{EXECABORT_DISCARDED_PREFIX}{}", STORE_REFUSAL_REPLIES[0].0);
        let mut cases: Vec<(String, fred::error::Error, bool)> = STORE_REFUSAL_REPLIES
            .iter()
            .map(|(reply, refusal)| {
                (
                    (*reply).to_owned(),
                    unknown_err(reply),
                    *refusal == StoreWriteRefusal::OutOfMemory,
                )
            })
            .collect();
        cases.push((execabort_oom.clone(), unknown_err(&execabort_oom), true));
        cases.push((
            "a command timeout".to_owned(),
            mk_err(FredErrorKind::Timeout),
            false,
        ));

        for (what, error, out_of_memory) in &cases {
            let signal = fresh_signal();
            FredErrOrigin::metrics(None, POOL_LABEL_DATA, &signal).observe_error(error);
            assert_eq!(
                signal.woken().now_or_never().is_some(),
                *out_of_memory,
                "observe_error on {what:?}: only -OOM notes the signal"
            );

            let signal = fresh_signal();
            let _ = refuse_store_write(
                FredErrOrigin::metrics(None, POOL_LABEL_DATA, &signal),
                STORE_WRITE_REFUSED_STAGE_EXT_STORE,
                error,
            );
            assert_eq!(
                signal.woken().now_or_never().is_some(),
                *out_of_memory,
                "refuse_store_write on {what:?}: only -OOM notes the signal"
            );
        }
    }

    /// The stage picks the answer: `create_commit` is plain `UNAVAILABLE`,
    /// with no retry hint and no nothing-applied claim, for each refusal;
    /// `ext_store` and `update_commit` keep the hinted answer. Both are counted.
    #[test]
    fn refuse_store_write_answers_create_commit_plain_and_the_upload_stages_hinted() {
        use crate::error::ServiceError;
        use crate::observability::metrics::test_util::{render, sample_value};
        use crate::observability::metrics::{
            STORE_WRITE_REFUSED_STAGE_CREATE_COMMIT, STORE_WRITE_REFUSED_STAGE_EXT_STORE,
            STORE_WRITE_REFUSED_STAGE_UPDATE_COMMIT,
        };
        let metrics: SharedMetrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "refuse-store-write-stage",
        ));
        let origin = || {
            FredErrOrigin::metrics(
                Some(&metrics),
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            )
        };
        for (reply, _) in STORE_REFUSAL_REPLIES {
            let refused = refuse_store_write(
                origin(),
                STORE_WRITE_REFUSED_STAGE_CREATE_COMMIT,
                &unknown_err(reply),
            );
            match refused {
                Some(ServiceError::Unavailable(message)) => {
                    assert!(
                        message
                            .starts_with("the backing store refused a write this request needed")
                            && message.ends_with("; retry")
                            && !message.contains("nothing was"),
                        "{reply:?}: {message}"
                    );
                }
                other => {
                    panic!("{reply:?}: create_commit must answer plain UNAVAILABLE; got {other:?}")
                }
            }
            for stage in [
                STORE_WRITE_REFUSED_STAGE_EXT_STORE,
                STORE_WRITE_REFUSED_STAGE_UPDATE_COMMIT,
            ] {
                let refused = refuse_store_write(origin(), stage, &unknown_err(reply));
                assert!(
                    matches!(refused, Some(ServiceError::UnavailableWithDetail { .. })),
                    "{reply:?} at {stage} keeps its retry hint; got {refused:?}"
                );
            }
        }
        let text = render(&metrics);
        for (_, refusal) in STORE_REFUSAL_REPLIES {
            let reason = refusal.reason_label();
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_store_write_refused_total",
                    &[("stage", "create_commit"), ("reason", reason)],
                ),
                Some(1.0),
                "the plain answer is still counted: {reason}"
            );
        }
    }

    // `FredErrorKind::Tls` is gated behind fred's TLS features (routed via the
    // `valkey-tls` alias → `fred/enable-rustls-ring`, t2.tr5.t4.oq), so the
    // variant is only nameable in this repo once that feature is enabled —
    // the same situation the doc comment at src/storage/client.rs:95-100
    // records for `FredErrorKind::Replica`. Each of these two tests therefore
    // carries its own `#[cfg(feature = "valkey-tls")]` (rather than a wrapping
    // block) so they stay co-located among the eleven non-transient-kind tests
    // above and preserve the one-test-per-`ErrorKind` convention. A sibling CI
    // job (t3.tr5.t4.oq.8z) compiles this crate `--no-default-features`, where
    // `FredErrorKind::Tls` does not exist.

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn is_transient_fred_error_tls_is_not_transient() {
        // The error is built by hand with the `Tls` kind. In this build (fred
        // over rustls) only a TLS configuration fault produces that kind — an
        // invalid TLS server name, or a TLS connector that does not match the
        // backend — and retrying on a fresh connection cannot fix one. A
        // certificate or handshake failure does NOT arrive as `Tls`: rustls
        // reports it to fred as an I/O error, so it reaches the classifier as
        // `IO` and is retried. Closes the t1.abb.6x §Existing-Behavior gap
        // ("keeps classifying FredErrorKind::Tls as fatal"): Task t2.tr5.t4.oq
        // is the first commit where the variant is nameable, so this is where
        // the test belongs.
        assert!(
            !is_transient_fred_error(&mk_err(FredErrorKind::Tls)),
            "FredErrorKind::Tls must stay NON-transient — a TLS configuration \
             fault (an invalid server name) retried on a fresh connection just \
             fails again and hides a real misconfiguration (t2.tr5.t4.oq)"
        );
    }

    #[cfg(feature = "valkey-tls")]
    #[test]
    fn map_fred_err_for_grpc_tls_maps_to_internal_not_unavailable() {
        // Counter-intuitive but deliberate: a `Tls`-kind error surfaces to
        // clients as gRPC INTERNAL (13, non-retriable), NOT UNAVAILABLE (14,
        // retriable). The error is built by hand: in this build (fred over
        // rustls) only a TLS configuration fault — an invalid TLS server name,
        // or a connector that does not match the backend — has the `Tls`
        // kind, and retrying cannot fix one. A certificate or handshake
        // failure is not this case: rustls reports it to fred as an I/O
        // error, which arrives as `IO`, maps to UNAVAILABLE and is retried.
        let svc_err = map_fred_err_for_grpc(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "Valkey connect",
            mk_err(FredErrorKind::Tls),
        );
        assert!(
            matches!(svc_err, crate::error::ServiceError::Internal(_)),
            "TLS error must map to ServiceError::Internal (non-retriable), not \
             Unavailable — a TLS configuration fault is not a retriable outage \
             (t2.tr5.t4.oq)"
        );
        let status = tonic::Status::from(svc_err);
        assert_eq!(
            status.code(),
            tonic::Code::Internal,
            "TLS error must present as gRPC INTERNAL (non-retriable); UNAVAILABLE \
             would tell clients to retry a permanent TLS misconfiguration \
             (t2.tr5.t4.oq)"
        );
    }

    #[test]
    fn map_snapshot_load_err_timeout_maps_to_aborted_with_prefix() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "HGETALL obj header",
            mk_err(FredErrorKind::Timeout),
        );
        match svc_err {
            crate::error::ServiceError::Aborted(msg) => {
                assert!(
                    msg.starts_with("transient I/O:"),
                    "transient fred error must produce Aborted(\"transient I/O: ...\") so the \
                     update_layer retry loop picks it up; got message: {msg:?}"
                );
                assert!(
                    msg.contains("HGETALL obj header"),
                    "Aborted message must preserve the op label for operator diagnosis; got: {msg:?}"
                );
            }
            other => panic!(
                "transient fred error must map to ServiceError::Aborted (retriable); got {other:?}"
            ),
        }
    }

    #[test]
    fn map_snapshot_load_err_io_maps_to_aborted_with_prefix() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "WATCH obj header",
            mk_err(FredErrorKind::IO),
        );
        assert!(matches!(
            svc_err,
            crate::error::ServiceError::Aborted(ref m) if m.starts_with("transient I/O:")
        ));
    }

    #[test]
    fn map_snapshot_load_err_canceled_maps_to_aborted_with_prefix() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "HSET field",
            mk_err(FredErrorKind::Canceled),
        );
        assert!(matches!(
            svc_err,
            crate::error::ServiceError::Aborted(ref m) if m.starts_with("transient I/O:")
        ));
    }

    #[test]
    fn map_snapshot_load_err_cluster_maps_to_aborted_with_prefix() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "ZADD fields index",
            mk_err(FredErrorKind::Cluster),
        );
        assert!(matches!(
            svc_err,
            crate::error::ServiceError::Aborted(ref m) if m.starts_with("transient I/O:")
        ));
    }

    #[test]
    fn map_snapshot_load_err_backpressure_maps_to_aborted_with_prefix() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "XADD queue",
            mk_err(FredErrorKind::Backpressure),
        );
        assert!(matches!(
            svc_err,
            crate::error::ServiceError::Aborted(ref m) if m.starts_with("transient I/O:")
        ));
    }

    #[test]
    fn map_snapshot_load_err_protocol_maps_to_internal() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "HGETALL obj header",
            mk_err(FredErrorKind::Protocol),
        );
        match svc_err {
            crate::error::ServiceError::Internal(msg) => {
                assert!(
                    !msg.starts_with("transient I/O:"),
                    "non-transient fred errors must NOT carry the transient I/O prefix — \
                     otherwise the retry loop would retry persistent bugs forever; got: {msg:?}"
                );
                assert!(
                    msg.contains("HGETALL obj header"),
                    "Internal message must preserve the op label; got: {msg:?}"
                );
            }
            other => panic!(
                "non-transient fred error must map to ServiceError::Internal (non-retriable); \
                 got {other:?}"
            ),
        }
    }

    #[test]
    fn map_snapshot_load_err_parse_maps_to_internal() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "HGETALL obj header",
            mk_err(FredErrorKind::Parse),
        );
        assert!(matches!(svc_err, crate::error::ServiceError::Internal(_)));
    }

    #[test]
    fn map_snapshot_load_err_auth_maps_to_internal() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "WATCH obj header",
            mk_err(FredErrorKind::Auth),
        );
        assert!(matches!(svc_err, crate::error::ServiceError::Internal(_)));
    }

    #[test]
    fn map_snapshot_load_err_config_maps_to_internal() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "WATCH obj header",
            mk_err(FredErrorKind::Config),
        );
        assert!(matches!(svc_err, crate::error::ServiceError::Internal(_)));
    }

    #[test]
    fn map_snapshot_load_err_includes_kind_label_in_transient_message() {
        // Operators triaging retries need to tell which transient kind
        // triggered the Aborted: a flood of "Timeout" reads differently from
        // a flood of "Cluster" (resharding). The engineer's format includes
        // `(<kind>)` between the op label and the underlying error — the
        // exact label text comes from `ErrorKind::to_str()`, so bind the
        // assertion to fred's own label (not a hardcoded string) so a fred
        // bump that renames the label doesn't silently break the test but
        // still leaves the production code working.
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "HGETALL obj header",
            mk_err(FredErrorKind::Timeout),
        );
        let msg = match svc_err {
            crate::error::ServiceError::Aborted(m) => m,
            other => panic!("expected Aborted, got {other:?}"),
        };
        let expected_label = FredErrorKind::Timeout.to_str();
        assert!(
            msg.contains(expected_label),
            "transient message must include the fred ErrorKind label (`{expected_label}`) so \
             operators can distinguish retry triggers; got: {msg:?}"
        );
    }

    #[test]
    fn map_snapshot_load_err_cluster_includes_fred_kind_label() {
        // Same contract as the Timeout case — pinned against fred's own
        // `ErrorKind::to_str()` so a fred label rename doesn't create a
        // fake regression.
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "WATCH obj header",
            mk_err(FredErrorKind::Cluster),
        );
        let msg = match svc_err {
            crate::error::ServiceError::Aborted(m) => m,
            other => panic!("expected Aborted, got {other:?}"),
        };
        let expected_label = FredErrorKind::Cluster.to_str();
        assert!(
            msg.contains(expected_label),
            "transient Cluster message must include fred's ErrorKind label (`{expected_label}`); \
             got: {msg:?}"
        );
    }

    #[test]
    fn map_snapshot_load_err_io_includes_fred_kind_label() {
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "HGETALL obj header",
            mk_err(FredErrorKind::IO),
        );
        let msg = match svc_err {
            crate::error::ServiceError::Aborted(m) => m,
            other => panic!("expected Aborted, got {other:?}"),
        };
        let expected_label = FredErrorKind::IO.to_str();
        assert!(
            msg.contains(expected_label),
            "transient IO message must include fred's ErrorKind label (`{expected_label}`); \
             got: {msg:?}"
        );
    }

    // ── b.1kf connect-timeout wiring ──────────────────────────────────────
    //
    // `apply_connect_timeout` is the load-bearing seam that wires
    // `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS` into fred's
    // `ConnectionConfig::connection_timeout`. fred 10.x defaults that field
    // to a hardcoded 10 s, so before b.1kf an operator who lowered the env
    // var below 10 s (e.g. for a tight CI fail-fast) was silently overridden.
    // The tests below pin the formula (`Duration::from_secs(N)`) and the
    // fred Builder-mutation API contract end-to-end: the value passed in
    // via `&Config` must be readable back through
    // `Builder::get_connection_config().connection_timeout`. A regression
    // on either side — a future refactor that drops the `.with_connection_config`
    // call, a fred upgrade that changes mutation semantics, or a typo in
    // the formula — fails these tests.

    /// Minimal `Config` for `apply_connect_timeout`. Only the
    /// `valkey_connect_timeout_secs` field is meaningful for these tests;
    /// every other field is filler that keeps the struct literal valid as
    /// new fields are added upstream (a new field surfaces as a compile
    /// error here, matching the pattern used by `tests/common::test_config`
    /// and `checkpoint::apply::tests::test_config`).
    fn connect_timeout_test_config(timeout_secs: u64) -> Config {
        Config {
            catchup_snapshot_enabled: false,
            port: 0,
            metrics_port: 0,
            valkey_url: String::new(),
            valkey_pool_size: 1,
            subscriber_pool_size: 1,
            subscriber_admission_wait_ms: 2_000,
            subscriber_max_subscriptions: 4096,
            subscriber_max_concurrent_catchups: 64,
            subscriber_admission_max_waiters: None,
            maintenance_pool_size: 1,
            catchup_snapshot_read_concurrency:
                crate::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            catchup_fenced_read_concurrency: crate::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            catchup_build_memory_budget_mib: crate::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: 1,
            snapshot_atomic_read_max_items: crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            snapshot_atomic_read_max_bytes: crate::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
            structural_cache_max_mib: crate::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: crate::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
            structural_cache_audit_above_ceiling: false,
            valkey_connect_timeout_secs: timeout_secs,
            valkey_command_timeout_ms: 5000,
            valkey_reconnect_max_delay_ms: 30_000,
            valkey_auth_mode: crate::config::ValkeyAuthMode::Static,
            valkey_token_file: None,
            valkey_token_username: None,
            valkey_token_refresh_lead_secs: 180,
            valkey_iam_username: None,
            valkey_iam_region: None,
            valkey_iam_cluster_host: None,
            valkey_iam_service: crate::config::ValkeyIamService::MemoryDb,
            storage_service_endpoint: None,
            auth_timeout_ms: 1000,
            auth_cache_ttl_secs: 60,
            auth_cache_negative_ttl_secs: 30,
            auth_userinfo_negative_ttl_secs: 5,
            auth_cache_max_entries: 1_000,
            userinfo_endpoint: None,
            max_layer_size: 1_048_576,
            upload_memory_budget_mib:
                crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            upload_admission_wait_ms:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
            upload_admission_max_waiters:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            layer_stream_maxlen: 4096,
            layer_stream_max_bytes: crate::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
            presence_broadcast_capacity: 4096,
            presence_lease_ttl_secs: crate::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs: crate::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms: crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            shutdown_timeout_secs: 1,
            shutdown_overall_budget_secs: 5,
            shutdown_stream_end_rate: 25,
            shutdown_stream_end_max_window_secs: 20,
            log_level: "warn".to_string(),
            log_format: crate::config::LogFormat::Json,
            trace_filter: crate::config::DEFAULT_TRACE_FILTER.to_string(),
            sweeper_interval_secs: 15,
            session_list_timeout_ms: 1000,
            rpc_deadline_ms: 5000,
            service_label: "test".to_string(),
            valkey_memory_bytes: None,
            ext_release_max_bytes: None,
            ext_reconcile_interval_secs: 3600,
            auth_max_attempts: 3,
            auth_retry_initial_delay_ms: 100,
            authn_enabled: true,
            authz_enabled: false,
            userinfo_service_enabled: true,
            test_only_dangerously_allow_unvalidated_tokens: false,
            health_probe_interval_ms: 5_000,
            health_probe_timeout_ms: 2_000,
            auth_protocol: crate::config::AuthProtocol::Rest,
            update_layer_retry_max_attempts: 128,
            update_layer_per_object_concurrency: 16,
            update_layer_backend: crate::config::UpdateLayerBackend::Lua,
            ext_value_verify: crate::config::ExtValueVerifyMode::Enforce,
            session_retry_max_attempts: 32,
            session_retry_initial_delay_ms: 1,
            pod_memory: crate::config::PodMemoryPlan::reference(),
            session_tokens_sweep_interval_secs:
                crate::grpc::session::SESSION_TOKENS_SWEEP_INTERVAL_SECS,
            checkpoint_enabled: false,
            checkpoint_adapter: crate::checkpoint::CheckpointAdapter::Disabled,
            checkpoint_interval_secs: 60,
            checkpoint_quiesce_secs: 30,
            checkpoint_tick_secs: 10,
            checkpoint_retain_generations: 2,
            checkpoint_seq_safety_margin: 1000,
            checkpoint_counters_sync_secs: 60,
            checkpoint_reconcile_interval_secs: 3600,
            checkpoint_counter_safety_margin: 1000,
            checkpoint_tail_enabled: false,
            checkpoint_tail_max_chain: 8,
            checkpoint_tail_max_entries: 4096,
            checkpoint_tail_max_bytes: 67_108_864,
            checkpoint_local_disk_path: None,
            checkpoint_storage_api_endpoint: None,
            checkpoint_base_path: None,
            checkpoint_connect_timeout_ms: 5_000,
            checkpoint_request_timeout_ms: 30_000,
            checkpoint_ext_mirror_workers: 2,
            checkpoint_ext_mirror_queue_capacity: 1_000,
            service_token_file: None,
            service_token_poll_interval_secs: 15,
            service_token_client_credentials_enabled: false,
        }
    }

    fn make_builder() -> Builder {
        // `from_url` is the same entry point `ValkeyClient::connect` uses,
        // so the test exercises the same Builder construction shape that
        // production hits — no hand-rolled `FredConfig::default()`.
        let fred_config =
            FredConfig::from_url("redis://127.0.0.1:6379").expect("static URL must parse");
        Builder::from_config(fred_config)
    }

    #[test]
    fn apply_connect_timeout_writes_configured_value_to_builder() {
        // 7 s is deliberately not 10 s (fred's pre-b.1kf hardcoded default)
        // — the test would silently pass on a regression that re-introduced
        // the default if we picked 10 here.
        let cfg = connect_timeout_test_config(7);
        let mut builder = make_builder();

        apply_connect_timeout(
            &mut builder,
            &cfg,
            &PeerReplies::new(),
            UnresponsiveVerdicts::FastAndSlow,
        );

        assert_eq!(
            builder.get_connection_config().connection_timeout,
            Duration::from_secs(7),
            "apply_connect_timeout must wire valkey_connect_timeout_secs into \
             ConnectionConfig::connection_timeout — fred's hardcoded 10 s default \
             must not leak through (b.1kf)"
        );
    }

    #[test]
    fn apply_connect_timeout_overrides_fred_default() {
        // Even when the operator picks a value *below* fred's 10 s default,
        // the configured value must reach `ConnectionConfig`. This is the
        // specific footgun b.1kf protects against: without the wiring, fred
        // silently kept 10 s and any sub-10 s env var was a no-op.
        let cfg = connect_timeout_test_config(3);
        let mut builder = make_builder();

        apply_connect_timeout(
            &mut builder,
            &cfg,
            &PeerReplies::new(),
            UnresponsiveVerdicts::FastAndSlow,
        );

        assert_eq!(
            builder.get_connection_config().connection_timeout,
            Duration::from_secs(3),
            "sub-10 s connect timeout must reach fred (silent-default-override \
             regression guard, b.1kf)"
        );
        assert_ne!(
            builder.get_connection_config().connection_timeout,
            Duration::from_secs(10),
            "fred's 10 s default must not be the observed value when the operator \
             configured anything else"
        );
    }

    #[test]
    fn apply_connect_timeout_enables_reconnect_on_auth_error() {
        // SR-4.2/SR-5.1: the credential-provider refresh loop can only re-AUTH a
        // rotated/refreshed credential onto an already-open connection if fred
        // re-establishes the connection after an auth error. `apply_connect_timeout`
        // flips `reconnect_on_auth_error` (a `ConnectionConfig` field, NOT a cargo
        // feature) from fred's `false` default to `true` via the same
        // `with_connection_config` closure that lands the connect timeout.
        let mut builder = make_builder();

        // Pre-state guard: fred defaults this to `false`. Asserting the default up
        // front makes the test fail closed against a fred bump that flips it — a
        // silently-true default would mask a helper that no longer sets the field.
        assert!(
            !builder.get_connection_config().reconnect_on_auth_error,
            "precondition: fred's ConnectionConfig::reconnect_on_auth_error default \
             must be false — a flipped default would hide a dropped field-set"
        );

        apply_connect_timeout(
            &mut builder,
            &connect_timeout_test_config(7),
            &PeerReplies::new(),
            UnresponsiveVerdicts::FastAndSlow,
        );

        assert!(
            builder.get_connection_config().reconnect_on_auth_error,
            "apply_connect_timeout must enable reconnect_on_auth_error so the \
             credential-provider refresh loop can re-AUTH on live connections \
             (SR-4.2/SR-5.1); fails closed if the field-set is dropped from the \
             with_connection_config closure"
        );
    }

    // ── fred's cluster-cache update delay stays zero ──────────────────────
    //
    // The vendored fred flushes every connection holding fed frames at the
    // top of each cluster-sync pass (`live_edit::before_inline_connect`,
    // change 5). A `-MOVED` first waits `ConnectionConfig::
    // cluster_cache_update_delay` (`delay_cluster_sync`), a read-polling sleep
    // that flush does not precede: a frame fed just before the `Moved` would
    // wait unsent through it, and a verdict could close a healthy connection
    // over it. The fix rests on that delay being fred's default zero in every
    // client the service builds. Two guards hold it: the configuration the
    // pools' shared builder seams produce reads zero, and no code under `src/`
    // writes the field.

    /// The token the source scan below looks for, split so this file's own
    /// mention of it is not a match.
    const CLUSTER_CACHE_UPDATE_DELAY: &str = concat!("cluster_cache_", "update_delay");

    #[test]
    fn the_pool_builder_keeps_freds_cluster_cache_update_delay_at_zero() {
        // Built as `ValkeyClient::connect` builds the pool template every pool
        // and the readers' template clone: the parsed URL config, the shared
        // reconnect policy and performance config, the connect-timeout seam
        // (either verdict set) and the connection buffer limits.
        let cfg = connect_timeout_test_config(7);
        for verdicts in [
            UnresponsiveVerdicts::FastAndSlow,
            UnresponsiveVerdicts::SlowOnly,
        ] {
            let mut builder = make_builder();
            builder
                .set_policy(shared_reconnect_policy(&cfg))
                .set_performance_config(shared_performance_config(&cfg));
            apply_connect_timeout(&mut builder, &cfg, &PeerReplies::new(), verdicts);
            connection_buffers::apply_limits(&mut builder, connection_buffers::GENERAL_LIMITS);
            assert_eq!(
                builder.get_connection_config().cluster_cache_update_delay,
                Duration::ZERO,
                "fred's cluster_cache_update_delay must stay zero: the vendored fred's \
                 inline-connect flush does not precede the sleep it adds before a \
                 cluster sync after -MOVED (vendor/fred-10.1.0.SOURCE.md change 5)"
            );
        }
    }

    #[test]
    fn no_code_under_src_writes_freds_cluster_cache_update_delay() {
        fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).expect("read src dir") {
                let path = entry.expect("read src entry").path();
                if path.is_dir() {
                    rust_files(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                }
            }
        }
        /// Whether a mention of the field, with `before` and `after` the rest
        /// of its line, writes it: a struct-literal field (`:` but not `::`);
        /// a field-init shorthand (`{ <field>, .. }` or `{ <field> }`: not a
        /// `.` access, and followed by `,`, `}` or, alone on its line, by
        /// nothing); an assignment or compound assignment (not `==`); a
        /// `&mut` borrow just before it; or a `ref mut` binding of it. It
        /// over-counts on purpose, the safe direction for a guard: a by-value
        /// destructure (`let ConnectionConfig { <field>, .. } = cfg;`) and a
        /// mention inside a trailing `// …` comment read as writes, so a
        /// failure naming one of those is the scanner, not a real write.
        fn writes(before: &str, after: &str) -> bool {
            let after = after.trim_start();
            let before = before.trim_end();
            let assigns = ["=", "+=", "-=", "*=", "/="]
                .iter()
                .any(|op| after.starts_with(op))
                && !after.starts_with("==");
            let field = after.starts_with(':') && !after.starts_with("::");
            let shorthand = !before.ends_with('.')
                && (after.starts_with(',')
                    || after.starts_with('}')
                    || (after.is_empty()
                        && (before.is_empty() || before.ends_with('{') || before.ends_with(','))));
            let borrowed = before
                .trim_end_matches(|c: char| c.is_alphanumeric() || c == '_' || c == '.')
                .trim_end()
                .ends_with("&mut");
            let ref_mut = before.ends_with("ref mut");
            assigns || field || shorthand || borrowed || ref_mut
        }

        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        assert!(
            files.len() > 50,
            "the scan must see the crate's sources; found {} files under {}",
            files.len(),
            src.display()
        );
        let mut hits = Vec::new();
        for file in &files {
            let text = std::fs::read_to_string(file).expect("read source file");
            for (number, line) in text.lines().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                for (at, _) in line.match_indices(CLUSTER_CACHE_UPDATE_DELAY) {
                    let after = &line[at + CLUSTER_CACHE_UPDATE_DELAY.len()..];
                    if writes(&line[..at], after) {
                        hits.push(format!(
                            "{}:{}: {}",
                            file.display(),
                            number + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
        assert!(
            hits.is_empty(),
            "fred's cluster_cache_update_delay must stay at its default zero: the \
             vendored fred's inline-connect flush does not precede the sleep it adds \
             before a cluster sync after -MOVED (vendor/fred-10.1.0.SOURCE.md change 5). \
             Writes found:\n{}",
            hits.join("\n")
        );
        // The scanner itself: each write shape is caught, a read is not.
        // Each assert below is caught by exactly one term of `writes`, so
        // deleting that term fails it.
        // Assignment, one per operator.
        assert!(writes("cfg.", " = Duration::from_millis(5);"));
        assert!(writes("cfg.", " += Duration::from_millis(5);"));
        assert!(writes("cfg.", " -= Duration::from_millis(5);"));
        assert!(writes("cfg.", " *= 2;"));
        assert!(writes("cfg.", " /= 2;"));
        // Struct-literal field.
        assert!(writes("        ", ": Duration::from_millis(5),"));
        // Field-init shorthand: followed by `,`, by `}`, or alone at the end of
        // its line after nothing, after `{` or after `,`.
        assert!(writes("ConnectionConfig { ", ", ..cfg }"));
        assert!(writes("ConnectionConfig { connection_timeout, ", " }"));
        assert!(writes("            ", ""));
        assert!(writes("ConnectionConfig { ", ""));
        assert!(writes("ConnectionConfig { a, ", ""));
        // `&mut` borrow.
        assert!(writes("std::mem::replace(&mut cfg.", ", d)"));
        // `ref mut` binding (ending its line after `mut`, which the shorthand
        // term does not take).
        assert!(writes("let ConnectionConfig { ref mut ", ""));
        // Reads: `==` is not an assignment, `::` is not a field, and a `.`
        // access followed by `,` is not a shorthand.
        assert!(!writes("cfg.", " == Duration::ZERO"));
        assert!(!writes("use ", "::Thing;"));
        assert!(!writes("builder.get_connection_config().", ","));
        assert!(!writes("(cfg.", ")"));
        assert!(!writes("let delay = cfg.", ";"));
    }

    // ── b.9uq unresponsive-connection check ───────────────────────────────
    //
    // `UnresponsiveConfig` is off by default in fred (`max_timeout: None`),
    // which is precisely why a connection silently invalidated backend-side
    // by a managed proxy could sit un-reaped forever: fred's per-command
    // timeouts are bare `tokio::time::timeout`s that fail the caller without
    // ever touching the socket. `apply_connect_timeout` is the single seam
    // every pool Builder routes through, so enabling the
    // check there is what makes the non-blocking pools self-heal.
    //
    // The tests below pin (a) that the seam turns the check on at the
    // documented values and (b) the two invariants those values exist to
    // satisfy — the max_timeout must clear the longest legitimate per-command
    // budget in the service (a stream multiplexer reader's `XREAD BLOCK` at
    // `XREAD_BLOCK_MS + 5 s`) so it can never reap a healthy connection, and
    // the scan interval must stay under half the max_timeout per fred's own
    // guidance.

    #[test]
    fn apply_connect_timeout_enables_unresponsive_connection_check() {
        let mut builder = make_builder();

        // Pre-state guard: fred leaves the check disabled. Asserting the
        // default up front makes the test fail closed against a fred bump
        // that flips it on — a silently-enabled default would mask a helper
        // that no longer sets the field.
        assert_eq!(
            builder.get_connection_config().unresponsive.max_timeout,
            None,
            "precondition: fred's UnresponsiveConfig::max_timeout default must be \
             None — a flipped default would hide a dropped field-set"
        );

        apply_connect_timeout(
            &mut builder,
            &connect_timeout_test_config(7),
            &PeerReplies::new(),
            UnresponsiveVerdicts::FastAndSlow,
        );

        assert_eq!(
            builder.get_connection_config().unresponsive.max_timeout,
            Some(UNRESPONSIVE_MAX_TIMEOUT),
            "apply_connect_timeout must enable fred's unresponsive check so a \
             silently-invalidated connection on a non-blocking pool is closed and \
             rebuilt instead of absorbing commands forever (b.9uq)"
        );
        assert_eq!(
            builder.get_connection_config().unresponsive.interval,
            UNRESPONSIVE_CHECK_INTERVAL,
            "apply_connect_timeout must set the unresponsive scan interval \
             explicitly rather than inheriting whatever fred's default happens \
             to be after a version bump (b.9uq)"
        );
    }

    /// The M1′ record the transaction timeout's per-command allowance is
    /// derived in (§ Validation records, R2). Kept out of the GitHub source
    /// release, so read at run time: the published tree skips the pin.
    const M1PRIME_DERIVATION_JSON: &str =
        "docs-internal/validation/txn-timeout-m1prime-20261013/derivation.json";

    /// `TXN_TIMEOUT_PER_COMMAND_US` is the value M1′ derived: re-read from the
    /// record on every run, so a re-measurement that moves it fails here until
    /// the constant moves with it.
    #[test]
    fn txn_timeout_per_command_is_the_recorded_m1prime_value() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("docs-internal").is_dir() {
            eprintln!("skipped: no docs-internal/ (published source tree)");
            return;
        }
        let text = std::fs::read_to_string(root.join(M1PRIME_DERIVATION_JSON))
            .unwrap_or_else(|e| panic!("the committed record {M1PRIME_DERIVATION_JSON}: {e}"));
        let doc: serde_json::Value =
            serde_json::from_str(&text).expect("the M1′ record is valid JSON");
        let recorded = doc
            .pointer("/txn_timeout_per_command/value_us")
            .and_then(serde_json::Value::as_u64)
            .expect("the M1′ record carries txn_timeout_per_command.value_us");
        assert_eq!(
            TXN_TIMEOUT_PER_COMMAND_US, recorded,
            "TXN_TIMEOUT_PER_COMMAND_US must be {M1PRIME_DERIVATION_JSON}'s \
             txn_timeout_per_command.value_us"
        );
        assert_eq!(
            TXN_TIMEOUT_PER_COMMAND,
            Duration::from_micros(TXN_TIMEOUT_PER_COMMAND_US)
        );
    }

    // ── `txn_timeout`: the whole-write transaction timeout (b.tbw) ─────────

    /// No commands: the transaction waits exactly the command timeout, so a
    /// small transaction fails as fast as any single command.
    #[test]
    fn txn_timeout_of_zero_commands_is_the_command_timeout() {
        let command_timeout = Duration::from_secs(5);
        assert_eq!(txn_timeout(command_timeout, 0), command_timeout);
    }

    /// The allowance grows by exactly `TXN_TIMEOUT_PER_COMMAND` per command
    /// below the cap.
    #[test]
    fn txn_timeout_grows_linearly_by_the_per_command_allowance() {
        let command_timeout = Duration::from_secs(5);
        for commands in [1u64, 2, 1_000, 100_000] {
            assert_eq!(
                txn_timeout(command_timeout, commands),
                command_timeout + TXN_TIMEOUT_PER_COMMAND * commands as u32,
                "{commands} commands"
            );
        }
        // At 120 µs: 100,000 commands add 12 s.
        assert_eq!(
            txn_timeout(command_timeout, 100_000),
            Duration::from_secs(17)
        );
    }

    /// A command timeout of zero is fred's "no timeout", and stays zero
    /// whatever the count: scaling it would turn "never time out" into a
    /// finite wait.
    #[test]
    fn txn_timeout_keeps_a_disabled_command_timeout_disabled() {
        for commands in [0u64, 1, 1_000_000, u64::MAX] {
            assert_eq!(
                txn_timeout(Duration::ZERO, commands),
                Duration::ZERO,
                "{commands} commands"
            );
        }
    }

    /// The wait is capped at the larger of the command timeout and the slow
    /// unresponsive bound: at 45 s for the default 5 s command timeout, and at
    /// the command timeout itself when an operator set it above 45 s (which
    /// then never grows).
    #[test]
    fn txn_timeout_is_capped_at_the_larger_of_the_command_timeout_and_the_slow_bound() {
        let five = Duration::from_secs(5);
        // 5 s + 333,334 × 120 µs is just over 45 s.
        assert_eq!(txn_timeout(five, 333_334), UNRESPONSIVE_MAX_TIMEOUT);
        assert_eq!(txn_timeout(five, 10_000_000), UNRESPONSIVE_MAX_TIMEOUT);
        // Just under the cap it still grows.
        assert!(txn_timeout(five, 333_333) < UNRESPONSIVE_MAX_TIMEOUT);

        let sixty = Duration::from_secs(60);
        assert_eq!(txn_timeout(sixty, 0), sixty);
        assert_eq!(
            txn_timeout(sixty, 1_000_000),
            sixty,
            "a command timeout above the slow bound is its own cap"
        );
        assert_eq!(UNRESPONSIVE_MAX_TIMEOUT, Duration::from_secs(45));
    }

    /// No count wraps it: counts past `u32` and a command timeout near
    /// `Duration::MAX` saturate into the cap instead of panicking.
    #[test]
    fn txn_timeout_saturates_on_absurd_inputs() {
        let five = Duration::from_secs(5);
        assert_eq!(
            txn_timeout(five, u64::from(u32::MAX) + 1),
            UNRESPONSIVE_MAX_TIMEOUT
        );
        assert_eq!(txn_timeout(five, u64::MAX), UNRESPONSIVE_MAX_TIMEOUT);
        assert_eq!(txn_timeout(Duration::MAX, u64::MAX), Duration::MAX);
        assert_eq!(txn_timeout(Duration::MAX, 1), Duration::MAX);
    }

    #[test]
    fn unresponsive_max_timeout_clears_longest_command_budget() {
        // A stream reader's XREAD BLOCK budget (`block_ms + 5 s`) is the
        // single longest per-command budget in `src/`; every other single
        // command is bounded by the 5 s `default_command_timeout`. (The two
        // whole-write transactions wait longer, `txn_timeout`, but at the
        // default command timeout that wait is capped at
        // `UNRESPONSIVE_MAX_TIMEOUT` itself, and their replies
        // stream back while they are written.) If
        // `UNRESPONSIVE_MAX_TIMEOUT` ever dropped to or below it, the
        // unresponsive check would start reaping perfectly healthy idle reader
        // connections mid-BLOCK.
        let longest_command_budget =
            Duration::from_millis(crate::storage::notifications::XREAD_BLOCK_MS + 5_000);
        assert!(
            UNRESPONSIVE_MAX_TIMEOUT > longest_command_budget,
            "UNRESPONSIVE_MAX_TIMEOUT ({UNRESPONSIVE_MAX_TIMEOUT:?}) must exceed the \
             longest legitimate per-command budget ({longest_command_budget:?} = \
             XREAD_BLOCK_MS + 5 s) or the check can reap healthy connections; \
             raising XREAD_BLOCK_MS requires raising UNRESPONSIVE_MAX_TIMEOUT too"
        );
    }

    #[test]
    fn unresponsive_check_interval_is_under_half_max_timeout() {
        // fred's documented guidance for `UnresponsiveConfig::interval`:
        // "usually less than half of max_timeout and always more than 1 ms".
        assert!(
            UNRESPONSIVE_CHECK_INTERVAL > Duration::from_millis(1),
            "fred requires the unresponsive scan interval to exceed 1 ms"
        );
        assert!(
            UNRESPONSIVE_CHECK_INTERVAL * 2 < UNRESPONSIVE_MAX_TIMEOUT,
            "fred's guidance is that the scan interval stay under half of \
             max_timeout so a wedge is detected within ~one interval of the \
             budget expiring; got interval {UNRESPONSIVE_CHECK_INTERVAL:?} vs \
             max_timeout {UNRESPONSIVE_MAX_TIMEOUT:?}"
        );
    }

    // ── b.9uq round 4: `Blocking::Interrupt` on the subscriber_pool ONLY ──
    //
    // `Config::blocking` is the readers' per-pool `FredConfig` divergence (the
    // other reader divergence, b.azh's per-connection `server`, is pinned by
    // `reader_connector_diverges_only_the_server_and_database_of_its_template`
    // below; the readers' credential-cadence divergence, b.19i's, is derived
    // inside `build_fred_config` and pinned in `pubsub_credential_refresh`). It
    // exists because the `subscriber_pool` clients are the stream multiplexer's
    // readers, each parked in `XREAD BLOCK`. A reader sends its own other
    // commands — its probes, head checks, keepalive and credential re-AUTH —
    // only between reads, and fred schedules no credential refresh on a reader
    // (b.19i), so what is left is the residual case: the reader's next command
    // after a client-side XREAD timeout that did not force a reconnect, while
    // the old read is still blocked on the server, would otherwise queue
    // server-side behind up to 30 s of that BLOCK. `Interrupt` makes fred send
    // `CLIENT UNBLOCK` first. The read it interrupts is that timed-out one,
    // whose `-UNBLOCKED` goes to a caller that already dropped it: nothing is
    // re-planned or counted. (A reader re-plans on `-UNBLOCKED` only when
    // something outside the service unblocks a live read.)
    //
    // The divergence must stay scoped to that one pool. The data / txn /
    // maintenance / pub-sub connections are MULTIPLEXED: a command queued behind
    // another there is normal pipelining, so `Interrupt` would make fred send
    // `CLIENT UNBLOCK` against perfectly ordinary traffic.
    //
    // Two guards, because neither alone is sufficient:
    //
    //   * the source guard below pins WHERE the mutation happens (exactly one
    //     site, on the readers' config, before the readers' Builder — the
    //     template every lazily connected reader is built from — is made) —
    //     `connect` is async, consumes its `Builder`s internally and returns
    //     only a connected `ValkeyClient`, so single-site-ness is not observable
    //     from any runtime value. Same technique, and same rationale, as the
    //     sibling `connect_applies_fred_config_overrides_before_the_first_clone`
    //     / `connect_has_no_second_credential_provider_mutation_site` guards;
    //   * the runtime guard in `tests/integration/storage/client_test.rs`
    //     (`connect_enables_blocking_interrupt_on_the_subscriber_pool_only`)
    //     reads `blocking` back off every live pool, and off a reader's client
    //     once a subscription has connected it, through fred's
    //     `ClientLike::client_config()`, which is what actually proves the
    //     divergence landed where it was meant to.

    #[test]
    fn fred_blocking_default_is_block_so_the_divergence_is_meaningful() {
        // Pre-state guard: the whole "one pool diverges" claim is only
        // meaningful while fred's default is `Block`. A fred bump that changed
        // the default would make every OTHER pool silently interrupt-capable
        // without failing any assertion that only checks the stream readers.
        let fred_config =
            FredConfig::from_url("redis://127.0.0.1:6379").expect("static URL must parse");
        assert_eq!(
            fred_config.blocking,
            Blocking::Block,
            "precondition: fred's `Config::blocking` default must be `Block`. If a fred \
             upgrade changes it, every pool but the stream readers starts sending CLIENT UNBLOCK \
             against ordinary pipelined traffic and the b.9uq round-4 divergence has to \
             be re-thought"
        );
        assert_ne!(
            Blocking::Interrupt,
            Blocking::Block,
            "sanity: the value the readers set must differ from the default it is diverging \
             from"
        );
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn connect_sets_blocking_interrupt_exactly_once_on_the_subscriber_pool_config() {
        // The whole of `connect`, not `connect_fn_body()`: the readers'
        // template is made by `ValkeyClient::reader_builder`, the associated
        // fn between `connect` and `start_router_supervision`, which this
        // slice spans (and `connect_fn_body()` does not).
        let body = connect_whole_body();

        // Exactly one `blocking` field assignment in the slice (`connect` plus
        // `reader_builder`).
        // The needle is the assignment form (`.blocking = `), so the prose in
        // the surrounding comment block — which necessarily says "blocking"
        // and "blocked" many times — cannot cross-match.
        assert_eq!(
            body.matches(".blocking = ").count(),
            1,
            "connect must contain exactly ONE `blocking` assignment. A second one is how \
             `Blocking::Interrupt` leaks onto a multiplexed pool, where a command queued \
             behind another is normal pipelining and CLIENT UNBLOCK would abort live \
             traffic (b.9uq round 4)"
        );
        assert_eq!(
            body.matches("subscriber_fred_config.blocking = Blocking::Interrupt")
                .count(),
            1,
            "the sole `blocking` assignment must target the readers' own `FredConfig` \
             (`subscriber_fred_config`) — not the shared `fred_config` every other connection \
             is built from"
        );
        // The struct-literal form of the same divergence
        // (`FredConfig { blocking: …, .. }`) would slip past the assignment
        // needle above, so exclude it outright. (`Blocking::…` alone is not a
        // usable needle: `reader_builder`'s rustdoc names the variant several
        // times, and comments are part of this slice.)
        assert_eq!(
            body.matches("blocking:").count(),
            0,
            "connect must not set `blocking` through a struct literal — the one divergence \
             goes through the documented `subscriber_fred_config.blocking = …` assignment \
             so the single-site needle above stays load-bearing"
        );

        // …and it must land BEFORE the readers' Builder is made from that
        // config, otherwise the readers inherit the default and the mutation
        // is dead code.
        let mutation_idx = body
            .find("subscriber_fred_config.blocking = Blocking::Interrupt")
            .expect("checked above");
        let builder_idx = body
            .find("Builder::from_config(subscriber_fred_config)")
            .expect(
                "connect must build the readers' Builder from the mutated \
                 `subscriber_fred_config` — an anchor-not-found panic here is an \
                 intentional loud maintenance signal, not a flake",
            );
        assert!(
            mutation_idx < builder_idx,
            "the `Blocking::Interrupt` mutation must precede `Builder::from_config` for the \
             readers — the other order silently builds them from an un-mutated config"
        );

        // b.19i: the config `blocking` is set on is the readers' own, derived
        // inside `build_fred_config` (`derive_reader_config`, its provider
        // cadence-wrapped), never the shared one. Built from `fred_config`,
        // the readers would carry the unwrapped provider and fred would run its
        // periodic refresh on them again — a second task writing to a
        // connection parked in `XREAD BLOCK`, which is what b.19i removed.
        assert_eq!(
            body.matches("let mut subscriber_fred_config = reader_fred_config;")
                .count(),
            1,
            "the readers' config must be the seam-derived `reader_fred_config`, taken whole"
        );
        assert_eq!(
            body.matches("subscriber_fred_config = fred_config").count(),
            0,
            "the readers must not be built from the shared `fred_config`, whose provider fred \
             refreshes"
        );
        assert_eq!(
            body.matches("reader_fred_config.").count(),
            0,
            "connect must not write a field of `reader_fred_config` beyond the one `blocking` \
             divergence on its moved value"
        );
        assert_eq!(
            body.matches("derive_reader_config(").count(),
            0,
            "connect must not derive the readers' config itself — that derivation lives inside \
             build_fred_config, after the shared config is final"
        );

        // The readers connect lazily (b.azh), so that Builder is no pool any
        // more: it is the template every reader connection is built from,
        // handed to the `ReaderConnector`. Pinned over the whole of `connect`
        // (`connect_fn_body()` stops at the data-pool retry, above the
        // connector): the connector must take exactly that Builder, after the
        // mutation, and `connect` must still hold exactly one `blocking`
        // assignment once the connector is built.
        let whole = connect_whole_body();
        assert_eq!(
            whole.matches(".blocking = ").count(),
            1,
            "the whole of connect must contain exactly ONE `blocking` assignment"
        );
        assert_eq!(
            whole
                .matches("let mut reader_builder = Builder::from_config(subscriber_fred_config);")
                .count(),
            1,
            "the readers' template must be made once, from the mutated config, as \
             `reader_builder` — an anchor miss here is an intentional loud maintenance signal"
        );
        assert_eq!(
            whole.matches("Self::reader_builder(").count(),
            1,
            "connect must make the readers' template exactly once, through \
             `Self::reader_builder`"
        );
        let template_idx = whole
            .find("let reader_builder = Self::reader_builder(config, reader_fred_config, &peer_replies);")
            .expect(
                "connect must bind the readers' template as `reader_builder`, from \
                 `Self::reader_builder` and the seam's `reader_fred_config` — an \
                 anchor-not-found panic here is an intentional loud maintenance signal",
            );
        assert_eq!(
            whole.matches("builder: reader_builder").count(),
            1,
            "the ReaderConnector must be built from the mutated `reader_builder`, exactly once — \
             a connector built from any other Builder connects its readers without \
             Blocking::Interrupt"
        );
        let connector_idx = whole
            .find("builder: reader_builder")
            .expect("checked above");
        assert!(
            template_idx < connector_idx,
            "the connector must be handed the readers' Builder after it was made from the \
             mutated config"
        );
        // b.19i: the connector is handed the provider the same seam returned
        // (unwrapped, the very `Arc` the pools carry), so each reader schedules
        // and gates its own re-AUTH from it. A provider from anywhere else
        // would be a second credential source.
        assert_eq!(
            whole.matches("reauth: reader_reauth").count(),
            1,
            "the ReaderConnector must take the seam's `reader_reauth`, exactly once"
        );
        assert_eq!(
            whole.matches("reader_reauth =").count(),
            0,
            "connect must not rebind `reader_reauth`: it comes from build_fred_config alone"
        );
    }

    /// From `ValkeyClient::connect`'s opener to `start_router_supervision`:
    /// the whole of `connect` plus `ValkeyClient::reader_builder`, the
    /// associated fn between them. `connect_fn_body()` deliberately
    /// stops at the data-pool retry; the reader connector is built below it.
    /// An anchor-not-found panic is an intentional loud maintenance signal.
    fn connect_whole_body() -> &'static str {
        let src = include_str!("client.rs");
        let start = src
            .find("pub async fn connect(")
            .expect("connect's signature anchor moved — update the connect source guards");
        let end = start
            + src[start..]
                .find("fn start_router_supervision(")
                .expect("the method after connect moved — update the connect source guards");
        &src[start..end]
    }

    /// The `ReaderConnector` methods that resolve a slot's owner and connect a
    /// reader: from `impl ReaderConnector {` to `arm_hooks`. Sliced, so the
    /// needles cannot match this test module's own literals.
    fn reader_connector_connect_body() -> &'static str {
        let src = include_str!("client.rs");
        let start = src.find("impl ReaderConnector {").expect(
            "the ReaderConnector impl anchor moved — update the reader-divergence guard \
             (intentional loud maintenance signal, not a flake)",
        );
        let end = start
            + src[start..].find("pub(crate) fn arm_hooks(").expect(
                "ReaderConnector::arm_hooks moved — update the reader-divergence guard \
                 (intentional loud maintenance signal, not a flake)",
            );
        &src[start..end]
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// b.azh, the third sanctioned `FredConfig` divergence: a cluster reader's
    /// connection goes to its bound slot's primary alone, so its config — a
    /// clone of the readers' template — gets that primary's `Server` and no
    /// database, and NOTHING else. The two shared-policy divergences are
    /// pinned above (`blocking`) and in `derive_pubsub_config`'s guards; this
    /// one may not grow into a credential, TLS or blocking write, which would
    /// make the readers' auth or transport differ from every other pool's.
    ///
    /// Source inspection, as for the `connect` guards: the connect needs a
    /// live cluster, and what it builds is observable only on the connected
    /// client (the runtime half is in `tests/integration/storage/`).
    #[test]
    fn reader_connector_diverges_only_the_server_and_database_of_its_template() {
        let body = reader_connector_connect_body();

        assert_eq!(
            body.matches(".with_config(").count(),
            1,
            "the reader connector must mutate its template in exactly one place"
        );
        let closure_start = body
            .find(".with_config(|config| {")
            .expect("the one mutation is a `with_config` closure over `config`");
        let closure_len = body[closure_start..]
            .find("});")
            .expect("the `with_config` closure ends with `});`");
        let closure = &body[closure_start..closure_start + closure_len];
        let writes: Vec<&str> = closure
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("config."))
            .collect();
        assert_eq!(
            writes,
            [
                "config.server = ServerConfig::Centralized { server };",
                "config.database = None;",
            ],
            "the reader divergence writes the server (the slot owner's, taken whole) and clears \
             the database, and writes nothing else"
        );
        for forbidden in [
            "credential_provider",
            ".blocking",
            ".tls =",
            "password",
            "username",
        ] {
            assert_eq!(
                body.matches(forbidden).count(),
                0,
                "the reader connector must not touch `{forbidden}`: credentials, TLS and the \
                 blocking policy ride the template clone untouched"
            );
        }
        assert_eq!(
            body.matches("self.builder.clone()").count(),
            1,
            "each reader connection is built from a clone of the one template"
        );
        assert_eq!(
            body.matches("Builder::from_config(").count(),
            0,
            "the reader connector must not build from any config but its template's clone"
        );
        assert!(
            body.contains("get_server(slot).cloned()") && !body.contains("Server::new("),
            "the slot owner's `Server` must be taken whole from the routing view (it carries the \
             TLS server name fred mapped), never rebuilt from host and port"
        );
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// The reader `server` divergence exists in exactly one place in the
    /// production code: the reader connector. A second `server` write anywhere
    /// would re-point some other pool at one node.
    #[test]
    fn the_reader_server_divergence_is_the_only_server_write_in_the_client() {
        let src = include_str!("client.rs");
        let production = &src[..src
            .find("\n#[cfg(test)]\nmod tests {")
            .expect("the test module anchor moved")];
        assert_eq!(
            production.matches(".server = ").count(),
            1,
            "exactly one `FredConfig::server` write in storage::client: the reader connector's"
        );
        assert_eq!(
            production.matches(".database = ").count(),
            1,
            "exactly one `FredConfig::database` write in storage::client: the reader connector's"
        );
        assert_eq!(
            reader_connector_connect_body()
                .matches("config.server = ")
                .count(),
            1,
            "and it is in the reader connector"
        );
    }

    // ── b.azh A2: a hard per-phase connect budget ─────────────────────────
    //
    // `connect_with_retry` used to check its budget only after an attempt
    // failed, so an attempt that never resolved (a handshake that hangs) held
    // its phase — and the pod's startup — past the budget the startup probe is
    // sized on. Each attempt now runs under `timeout(remaining budget)`. The
    // cut attempt's future is dropped mid-`init()`, after fred has spawned the
    // member's router, so the phase's `MemberRouters` keeps every router from
    // its spawn and aborts the ones it still holds when the failed connect
    // drops it.

    /// Sets its flag when dropped: proves a future holding it was dropped.
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// The phase budget the cut tests run under: short, so the cut is quick.
    const HARD_BUDGET: Duration = Duration::from_millis(200);

    /// Bound on the call under test: a deadline the healthy path never
    /// reaches. Hit only when the budget is not hard.
    const CUT_DEADLINE: Duration = Duration::from_secs(10);

    #[tokio::test]
    async fn connect_with_retry_cuts_an_attempt_that_never_resolves_at_the_budget() {
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        let dropped = Arc::new(AtomicBool::new(false));
        let flag = AtomicBool::new(false);

        let outcome = tokio::time::timeout(
            CUT_DEADLINE,
            connect_with_retry("data pool", HARD_BUDGET, None, &flag, || {
                attempts.fetch_add(1, Ordering::SeqCst);
                let guard = DropFlag(Arc::clone(&dropped));
                async move {
                    let _guard = guard;
                    std::future::pending::<Result<(), fred::error::Error>>().await
                }
            }),
        )
        .await
        .expect(
            "an attempt that never resolves must be cut at the phase budget — a soft budget, \
             checked only after an attempt fails, waits on it forever",
        );

        match outcome {
            Err(ConnectError::Timeout { source, .. }) => {
                assert!(
                    matches!(source.kind(), FredErrorKind::Timeout),
                    "the cut phase fails as a timeout: {source:?}"
                );
                assert!(
                    source.details().contains("data pool"),
                    "the error names the phase that was cut: {source:?}"
                );
                assert!(
                    !source.details().contains("previous attempt"),
                    "no attempt failed before the cut one, so the error names none: {source:?}"
                );
            }
            other => panic!("a cut phase must fail with ConnectError::Timeout; got {other:?}"),
        }
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "a cut at the budget fails the phase at once: there is no budget left to retry in"
        );
        assert!(
            dropped.load(Ordering::SeqCst),
            "the cut attempt's future is dropped, not left running"
        );
    }

    /// The budget is the PHASE's, not each attempt's: a later attempt gets only
    /// what the earlier ones left. On a paused clock, with a 1 s budget, the
    /// first attempt fails at 600 ms, the backoff sleeps 200 ms, and the second
    /// attempt — which never resolves — is cut at exactly 1 s, the budget. An
    /// attempt bounded by the whole budget instead would be cut at 1.8 s.
    #[tokio::test(start_paused = true)]
    async fn connect_with_retry_cuts_a_later_attempt_at_what_is_left_of_the_phase_budget() {
        const BUDGET: Duration = Duration::from_millis(1_000);
        const FIRST_FAILS_AFTER: Duration = Duration::from_millis(600);
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        let flag = AtomicBool::new(false);
        let start = tokio::time::Instant::now();

        let outcome = connect_with_retry("data pool", BUDGET, None, &flag, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            async move {
                if n == 0 {
                    tokio::time::sleep(FIRST_FAILS_AFTER).await;
                    Err(fred::error::Error::new(
                        FredErrorKind::IO,
                        "synthetic connection refused",
                    ))
                } else {
                    std::future::pending::<Result<(), fred::error::Error>>().await
                }
            }
        })
        .await;
        let elapsed = start.elapsed();

        assert!(
            matches!(&outcome, Err(ConnectError::Timeout { source, .. })
                if matches!(source.kind(), FredErrorKind::Timeout)),
            "the never-resolving attempt is cut and the phase fails as a timeout: {outcome:?}"
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(
            elapsed >= BUDGET && elapsed <= BUDGET + Duration::from_millis(1),
            "the phase ends at its budget, {BUDGET:?} of virtual time, not at the second \
             attempt's start plus a whole budget (1.8 s); it ended at {elapsed:?}"
        );
    }

    /// A phase whose budget cuts an attempt after earlier attempts failed
    /// keeps the real cause in its fatal error: the cut's `Timeout` names the
    /// phase first, then the LATEST failed attempt's error — here a refused
    /// credential after a refused connection — so a misconfiguration is not
    /// reported as a bare timeout. On a paused clock with a 1 s budget:
    /// attempt 1 fails at once, the backoff sleeps 200 ms, attempt 2 fails at
    /// once, the backoff sleeps 400 ms, and attempt 3, which never resolves,
    /// is cut at 1 s.
    #[tokio::test(start_paused = true)]
    async fn a_cut_connect_phase_fails_as_a_timeout_naming_the_previous_attempts_error() {
        const BUDGET: Duration = Duration::from_millis(1_000);
        let first = fred::error::Error::new(FredErrorKind::IO, "synthetic connection refused");
        let latest = fred::error::Error::new(
            FredErrorKind::Auth,
            "WRONGPASS synthetic invalid username-password pair",
        );
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        let flag = AtomicBool::new(false);

        let outcome = connect_with_retry("txn pool", BUDGET, None, &flag, || {
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            let failure = match n {
                0 => Some(first.clone()),
                1 => Some(latest.clone()),
                _ => None,
            };
            async move {
                match failure {
                    Some(e) => Err(e),
                    None => std::future::pending::<Result<(), fred::error::Error>>().await,
                }
            }
        })
        .await;

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            3,
            "precondition: two failed attempts, then the one cut"
        );
        let source = match outcome {
            Err(ConnectError::Timeout { source, .. }) => source,
            other => panic!("a cut phase fails with ConnectError::Timeout; got {other:?}"),
        };
        assert!(
            matches!(source.kind(), FredErrorKind::Timeout),
            "the cut is still a timeout, whatever came before it: {source:?}"
        );
        assert_eq!(
            source.details(),
            format!(
                "txn pool: the connect attempt did not finish within the 1000 ms budget; the \
                 previous attempt failed: {latest}"
            ),
            "the phase label leads, and the latest real error follows"
        );
        assert!(
            !source.details().contains("synthetic connection refused"),
            "only the latest failed attempt is named, not every one: {source:?}"
        );
    }

    /// A stand-in for a member's router: a task that never ends on its own.
    fn running_router() -> ConnectHandle {
        tokio::spawn(std::future::pending())
    }

    /// Polls (bounded) until `task` has ended.
    async fn await_router_ended(task: &tokio::task::AbortHandle, what: &str) {
        let deadline = tokio::time::Instant::now() + CUT_DEADLINE;
        while !task.is_finished() {
            assert!(tokio::time::Instant::now() < deadline, "{what}");
            tokio::task::yield_now().await;
        }
    }

    /// The whole A2 flow: a phase whose attempt spawned its member's router
    /// and then never resolved is cut at the budget, and dropping that phase's
    /// `MemberRouters` — as a failed `connect` does — ends the router. Left
    /// running, it would reconnect a client nothing supervises or quits.
    #[tokio::test]
    async fn a_connect_phase_cut_at_its_budget_leaves_no_router_running() {
        let routers = MemberRouters::new(1);
        let router_task: std::sync::Mutex<Option<tokio::task::AbortHandle>> =
            std::sync::Mutex::new(None);
        let flag = AtomicBool::new(false);

        let outcome = tokio::time::timeout(
            CUT_DEADLINE,
            connect_with_retry("data pool", HARD_BUDGET, None, &flag, || {
                // As `MemberRouters::init_member`: the router's handle is kept
                // from its spawn, then the connect is awaited.
                let router = running_router();
                *router_task.lock().unwrap() = Some(router.abort_handle());
                routers.lock()[0].router = Some(router);
                std::future::pending::<Result<(), fred::error::Error>>()
            }),
        )
        .await
        .expect("the attempt must be cut at the budget");
        assert!(matches!(outcome, Err(ConnectError::Timeout { .. })));
        let router_task = router_task
            .lock()
            .unwrap()
            .take()
            .expect("the attempt spawned its router");
        assert!(
            !router_task.is_finished(),
            "precondition: the cut leaves the router running until its owner ends it"
        );

        drop(routers);
        await_router_ended(
            &router_task,
            "dropping a phase's MemberRouters after a failed connect must abort the router the \
             cut attempt spawned",
        )
        .await;
    }

    /// `MemberRouters` aborts every router it still holds when dropped —
    /// connected or not, since a later phase's failure fails the connect too —
    /// and none it has handed to the router supervisor.
    #[tokio::test]
    async fn member_routers_abort_held_routers_on_drop_but_not_handed_over_ones() {
        let failed = MemberRouters::new(2);
        let (a, b) = (running_router(), running_router());
        let (a_task, b_task) = (a.abort_handle(), b.abort_handle());
        {
            let mut members = failed.lock();
            members[0].router = Some(a);
            members[0].connected = true;
            members[1].router = Some(b);
        }
        drop(failed);
        await_router_ended(&a_task, "a connected member's router must be aborted").await;
        await_router_ended(&b_task, "an unconnected member's router must be aborted").await;

        let succeeded = MemberRouters::new(1);
        succeeded.lock()[0].router = Some(running_router());
        succeeded.lock()[0].connected = true;
        let handed_over = succeeded.into_handles();
        assert_eq!(handed_over.len(), 1);
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert!(
            !handed_over[0].is_finished(),
            "a router handed to the supervisor is the supervisor's: dropping its emptied \
             MemberRouters must not abort it"
        );
        handed_over[0].abort();
    }

    // The same two guarantees through the real `MemberRouters::init` path, on
    // a real fred client: `init_member` must keep each router's handle from
    // its spawn, and abort a previous attempt's router, for the drop above to
    // have anything to abort. The client's server is a listener that accepts
    // and never replies, so each connection stalls in fred's setup (its first
    // `PING`), and fred's own setup bound is raised to a minute: within these
    // tests only an abort of a router can close a connection it opened.

    /// A listener that accepts and never replies, and its port.
    async fn silent_listener() -> (tokio::net::TcpListener, u16) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the listener must bind");
        let port = listener.local_addr().expect("a bound address").port();
        (listener, port)
    }

    /// A single-node fred client to `port` whose setup fred gives up on only
    /// after a minute.
    fn client_to_silent_listener(port: u16) -> Client {
        let mut builder = Builder::default_centralized();
        builder.with_config(|config| {
            config.server = ServerConfig::new_centralized("127.0.0.1", port);
        });
        builder.with_connection_config(|config| {
            config.internal_command_timeout = Duration::from_secs(60);
        });
        builder.build().expect("the test client must build")
    }

    /// The next connection the listener accepted, after its client wrote
    /// something on it (fred's setup `PING`): the setup is underway and stalled.
    async fn accept_stalled_setup(listener: &tokio::net::TcpListener) -> tokio::net::TcpStream {
        use tokio::io::AsyncReadExt;
        let (mut conn, _) = tokio::time::timeout(CUT_DEADLINE, listener.accept())
            .await
            .expect("the client must connect")
            .expect("accept must succeed");
        let mut buf = [0u8; 64];
        let read = tokio::time::timeout(CUT_DEADLINE, conn.read(&mut buf))
            .await
            .expect("the client must start its setup")
            .expect("the read must succeed");
        assert!(read > 0, "precondition: the client wrote its setup command");
        conn
    }

    /// Reads (bounded) until the client closes `conn`.
    async fn await_client_closed(conn: &mut tokio::net::TcpStream, what: &str) {
        use tokio::io::AsyncReadExt;
        let deadline = tokio::time::Instant::now() + CUT_DEADLINE;
        let mut buf = [0u8; 64];
        loop {
            match tokio::time::timeout_at(deadline, conn.read(&mut buf)).await {
                Err(_) => panic!("{what}"),
                Ok(Ok(0) | Err(_)) => return,
                Ok(Ok(_)) => {}
            }
        }
    }

    /// A phase whose real member stalls mid-setup is cut at its budget by
    /// `connect_with_retry`, and dropping the phase's `MemberRouters` — as a
    /// failed `connect` does — closes the member's connection: `init` kept the
    /// router's handle from its spawn. Left running, the router would hold
    /// the connection and keep setting up a client nothing supervises.
    #[tokio::test]
    async fn a_real_member_cut_mid_setup_is_closed_when_its_phases_routers_drop() {
        let (listener, port) = silent_listener().await;
        let client = client_to_silent_listener(port);
        let routers = MemberRouters::new(1);
        let flag = AtomicBool::new(false);

        let outcome = tokio::time::timeout(
            CUT_DEADLINE,
            connect_with_retry("data pool", HARD_BUDGET, None, &flag, || {
                routers.init(std::slice::from_ref(&client))
            }),
        )
        .await
        .expect("the stalled attempt must be cut at the budget");
        assert!(
            matches!(outcome, Err(ConnectError::Timeout { .. })),
            "the phase fails as a timeout: {outcome:?}"
        );
        let mut conn = accept_stalled_setup(&listener).await;
        assert!(
            routers.lock()[0]
                .router
                .as_ref()
                .is_some_and(|router| !router.is_finished()),
            "precondition: the cut attempt's router is held, and still running"
        );

        drop(routers);
        await_client_closed(
            &mut conn,
            "dropping the phase's MemberRouters must abort the router `init` spawned, closing \
             its connection",
        )
        .await;
    }

    /// A retried `init` aborts the router the previous attempt spawned before
    /// it spawns the next, so no two routers of one client run side by side:
    /// the first attempt's stalled connection closes while the `MemberRouters`
    /// is still alive, and the second attempt's router is the one it holds.
    #[tokio::test]
    async fn a_retried_member_init_aborts_the_previous_attempts_router() {
        let (listener, port) = silent_listener().await;
        let client = client_to_silent_listener(port);
        let routers = MemberRouters::new(1);
        let attempt =
            || tokio::time::timeout(HARD_BUDGET, routers.init(std::slice::from_ref(&client)));

        assert!(
            attempt().await.is_err(),
            "precondition: the first attempt stalls mid-setup"
        );
        let mut first = accept_stalled_setup(&listener).await;
        assert!(
            attempt().await.is_err(),
            "precondition: the second attempt stalls mid-setup"
        );
        let mut second = accept_stalled_setup(&listener).await;
        await_client_closed(
            &mut first,
            "the second attempt must abort the first attempt's router, closing its connection",
        )
        .await;
        assert!(
            routers.lock()[0]
                .router
                .as_ref()
                .is_some_and(|router| !router.is_finished()),
            "the second attempt's router is the one held, and still running"
        );
        drop(routers);
        await_client_closed(
            &mut second,
            "dropping the MemberRouters must abort the second attempt's router",
        )
        .await;
    }

    // The same guarantee for a stream reader's connect (b.azh):
    // `ReaderConnector::connect` keeps the router of the client it builds from
    // its spawn, and a connect that is cut at its bound or cancelled
    // mid-setup aborts it, so the connection it was opening closes. The
    // connector's template targets a `silent_listener`, so the only
    // connection the listener sees is the reader's, and fred's own setup
    // bound is a minute: within these tests only that abort can close it.
    // The integration tests stage the same cut and cancellation with the
    // `MuxFaultSite::Connect` seam, which cannot show the socket
    // (`tests/integration/storage/stream_mux_reader_connect_test.rs`).

    /// A standalone reader connector whose template targets `port`, with
    /// fred's own setup bound raised to a minute and its connect bound
    /// (`LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`) at `connect_timeout`.
    fn reader_connector_to_silent_listener(
        port: u16,
        connect_timeout: Duration,
    ) -> Arc<ReaderConnector> {
        let routing = Builder::default_centralized()
            .build_pool(1)
            .expect("an unconnected pool builds");
        let mut connector = ReaderConnector::unconnected_for_test(routing, false);
        let fresh = Arc::get_mut(&mut connector).expect("a fresh connector is not shared yet");
        fresh.builder.with_config(|config| {
            config.server = ServerConfig::new_centralized("127.0.0.1", port);
        });
        fresh.builder.with_connection_config(|config| {
            config.internal_command_timeout = Duration::from_secs(60);
        });
        fresh.connect_timeout = connect_timeout;
        connector
    }

    /// A reader connect whose new connection stalls mid-setup (the listener
    /// never answers its `PING`) is cut at the connector's own bound, failing
    /// with the bound's `Timeout`, and the cut aborts the router the connect
    /// started, closing the stalled connection. Left running, that router
    /// would hold the connection for fred's minute-long setup bound, past
    /// [`CUT_DEADLINE`].
    #[tokio::test]
    async fn a_reader_connect_cut_at_its_bound_mid_setup_closes_its_connection() {
        let (listener, port) = silent_listener().await;
        let connector = reader_connector_to_silent_listener(port, Duration::from_secs(1));
        let cancel = CancellationToken::new();

        let (outcome, mut conn) = tokio::join!(
            tokio::time::timeout(
                CUT_DEADLINE,
                connector.connect(
                    0,
                    None,
                    crate::observability::metrics::READER_CONNECT_CAUSE_BIND,
                    None,
                    &cancel,
                ),
            ),
            accept_stalled_setup(&listener),
        );
        match outcome.expect("the stalled connect must be cut at its bound") {
            Err(ReaderConnectError::Failed { error, .. }) => {
                assert!(
                    matches!(error.kind(), FredErrorKind::Timeout),
                    "the cut fails as a timeout: {error:?}"
                );
                assert_eq!(
                    error.details(),
                    "the reader connection was not up within 1 s \
                     (LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS)",
                    "the connector's own bound cut the connect, not fred's setup bound"
                );
            }
            other => panic!("a connect whose setup never answers is cut as failed; got {other:?}"),
        }
        await_client_closed(
            &mut conn,
            "the cut must abort the router the connect started, closing its connection",
        )
        .await;
    }

    /// A reader connect cancelled (the reader's shutdown) while its new
    /// connection is stalled mid-setup returns `Cancelled` and aborts the
    /// router it started, closing the stalled connection. The connect bound
    /// is a minute here, as is fred's setup bound: only the cancellation can
    /// end the connect inside [`CUT_DEADLINE`], and only the abort can close
    /// the connection.
    #[tokio::test]
    async fn a_reader_connect_cancelled_mid_setup_closes_its_connection() {
        let (listener, port) = silent_listener().await;
        let connector = reader_connector_to_silent_listener(port, Duration::from_secs(60));
        let cancel = CancellationToken::new();

        let (outcome, mut conn) = tokio::join!(
            tokio::time::timeout(
                CUT_DEADLINE,
                connector.connect(
                    0,
                    None,
                    crate::observability::metrics::READER_CONNECT_CAUSE_BIND,
                    None,
                    &cancel,
                ),
            ),
            async {
                let conn = accept_stalled_setup(&listener).await;
                cancel.cancel();
                conn
            },
        );
        match outcome.expect("the cancelled connect must return") {
            Err(ReaderConnectError::Cancelled) => {}
            other => panic!("a connect cancelled mid-setup returns Cancelled; got {other:?}"),
        }
        await_client_closed(
            &mut conn,
            "the cancellation must abort the router the connect started, closing its connection",
        )
        .await;
    }

    // A reader connect the server refuses, or whose setup connection it drops,
    // fails at once with the error fred parsed from the wire — classified as
    // the reader and its credential hint expect — rather than being retried
    // inside fred until the connect bound cuts it as a `Timeout`. What makes a
    // first connect fail at once is fred's `fail_fast` (default `true`, never
    // written by `build_fred_config`): with it, fred's router init returns the
    // first connect's error; without it, the router retries under the
    // reconnect policy (`fred-10.1.0/src/router/commands.rs:534-546`).
    // `reconnect_on_auth_error` plays no part in a first connect. The
    // connector's template is the production one: `build_fred_config`'s
    // reader config through `ValkeyClient::reader_builder`, the function
    // `connect` builds the readers' template with. The server is a listener
    // only the reader's connection reaches, answering its first setup frame
    // (its `PING`: no credentials here) itself. The integration tests stage
    // the same outcomes with the `MuxFaultSite::Connect` seam, which builds
    // these errors by hand
    // (`tests/integration/storage/stream_mux_reader_connect_test.rs`).

    /// The connect bound the rejection tests give the connector: far past
    /// [`CUT_DEADLINE`], so a connect that returns inside that deadline was not
    /// cut by it.
    const REJECTED_CONNECT_BOUND_SECS: u64 = 60;

    /// A standalone reader connector to `port`, its template built by
    /// `ValkeyClient::reader_builder` from `build_fred_config`'s reader
    /// config, as `connect` builds it, with the connect bound
    /// [`REJECTED_CONNECT_BOUND_SECS`]. Asserts the template fails fast, the
    /// property the rejection tests rest on.
    fn production_reader_connector(port: u16) -> Arc<ReaderConnector> {
        production_reader_connector_bounded(port, REJECTED_CONNECT_BOUND_SECS)
    }

    /// [`production_reader_connector`], with the connect bound `bound_secs`.
    fn production_reader_connector_bounded(port: u16, bound_secs: u64) -> Arc<ReaderConnector> {
        let mut config = connect_timeout_test_config(bound_secs);
        config.valkey_url = format!("redis://127.0.0.1:{port}");
        let FredConfigs {
            reader: reader_fred_config,
            ..
        } = build_fred_config(&config, None, None).expect("the reader config builds");
        let reader_builder =
            ValkeyClient::reader_builder(&config, reader_fred_config, &PeerReplies::new());
        assert!(
            reader_builder
                .get_config()
                .expect("the readers' template carries a config")
                .fail_fast,
            "the readers' template must fail a first connect at once (fred's `fail_fast`)"
        );

        let routing = Builder::default_centralized()
            .build_pool(1)
            .expect("an unconnected pool builds");
        let mut connector = ReaderConnector::unconnected_for_test(routing, false);
        let fresh = Arc::get_mut(&mut connector).expect("a fresh connector is not shared yet");
        fresh.builder = reader_builder;
        fresh.connect_timeout = Duration::from_secs(config.valkey_connect_timeout_secs);
        connector
    }

    /// Connects a [`production_reader_connector`] to a fresh listener that
    /// answers the reader's first setup frame with the RESP error `reply`, or,
    /// with `None`, closes the connection once that frame arrives. Returns
    /// the connect's error, asserting it came back failed inside
    /// [`CUT_DEADLINE`] — well inside the connect bound — and not as a
    /// `Timeout`.
    async fn reader_connect_answered(reply: Option<&str>) -> fred::error::Error {
        use tokio::io::AsyncWriteExt;
        let (listener, port) = silent_listener().await;
        let connector = production_reader_connector(port);
        let cancel = CancellationToken::new();

        let (outcome, _held) = tokio::join!(
            tokio::time::timeout(
                CUT_DEADLINE,
                connector.connect(
                    0,
                    None,
                    crate::observability::metrics::READER_CONNECT_CAUSE_BIND,
                    None,
                    &cancel,
                ),
            ),
            async {
                let mut conn = accept_stalled_setup(&listener).await;
                match reply {
                    Some(reply) => {
                        conn.write_all(format!("-{reply}\r\n").as_bytes())
                            .await
                            .expect("the reply must be written");
                        Some(conn)
                    }
                    None => None,
                }
            },
        );
        let error = match outcome.expect(
            "the connect must fail at once, not wait for its bound: fred retried the refused or \
             dropped setup internally",
        ) {
            Err(ReaderConnectError::Failed { error, .. }) => error,
            other => panic!("the answered connect must fail; got {other:?}"),
        };
        assert!(
            !matches!(error.kind(), FredErrorKind::Timeout),
            "the connect failed with the server's answer, not a cut: {error:?}"
        );
        error
    }

    /// A reader connect whose setup the server refuses `-WRONGPASS` fails at
    /// once with fred's parse of the reply: an `Auth` error whose details are
    /// the reply, which `is_server_auth_rejection` and the connect's
    /// `WRONGPASS` hint gate recognise, and which the reader treats as
    /// permanent.
    #[tokio::test]
    async fn a_reader_connect_refused_wrongpass_fails_at_once_as_a_server_auth_rejection() {
        const WRONGPASS: &str = "WRONGPASS invalid username-password pair or user is disabled.";
        let error = reader_connect_answered(Some(WRONGPASS)).await;
        assert!(
            is_server_auth_rejection(&error),
            "fred parses -WRONGPASS as an Auth error carrying the reply: {error:?}"
        );
        assert!(
            error.details().starts_with("WRONGPASS"),
            "the details keep the server's token, which the hint gate keys on: {error:?}"
        );
        assert!(!is_transient_fred_error(&error), "{error:?}");
    }

    /// A reader connect whose setup the server refuses `-NOAUTH` fails at once
    /// with an `Auth` error whose details start `NOAUTH`: a server rejection,
    /// which the connect's hint gate (`WRONGPASS` only) skips.
    #[tokio::test]
    async fn a_reader_connect_refused_noauth_fails_at_once_as_a_server_auth_rejection() {
        const NOAUTH: &str = "NOAUTH Authentication required.";
        let error = reader_connect_answered(Some(NOAUTH)).await;
        assert!(
            is_server_auth_rejection(&error),
            "fred parses -NOAUTH as an Auth error carrying the reply: {error:?}"
        );
        assert!(
            error.details().starts_with("NOAUTH"),
            "the details keep the server's token: {error:?}"
        );
        assert!(!is_transient_fred_error(&error), "{error:?}");
    }

    /// A reader connect whose setup connection the server drops fails at once
    /// with a transient error, which the reader retries after a backoff
    /// without ending its subscriptions.
    #[tokio::test]
    async fn a_reader_connect_whose_setup_connection_is_dropped_fails_at_once_as_transient() {
        let error = reader_connect_answered(None).await;
        assert!(
            is_transient_fred_error(&error),
            "a dropped setup connection is transient: {error:?}"
        );
    }

    // A `MuxFault::Hold` at the `Connect` site (`reader_connect_setup`) makes
    // the connect report its real setup's outcome only once the test releases
    // the hold. Here the listener refuses the setup `-WRONGPASS` while the
    // hold is still pending, so each outcome tells the hold's handling apart
    // from the setup's own: the refusal (an `Auth` failure) reported once the
    // hold is released, the connect bound's `Timeout`, or `Cancelled`. The
    // refusal-then-release cases pin why the setup's connect notification and
    // the release are awaited together: a notification subscribed only after
    // the release would miss a refusal made during the hold, and the connect
    // would wait for its bound.
    //
    // These run on a current-thread runtime. The test learns the refusal has
    // run its course from its setup connection closing, and fred drops that
    // connection and broadcasts the connect's failure in one poll of the
    // router task (the `?` on the setup in
    // `router::centralized::initialize_connection`, then `broadcast_connect`
    // in `router::commands::init`, with no await between), so on one thread
    // the test sees the close only after the failure was broadcast.

    /// How a held-connect test ends the hold.
    enum HoldEnd {
        /// Sends on the release.
        Send,
        /// Drops the release.
        Drop,
        /// Cancels the connect, keeping the release.
        Cancel,
        /// Keeps the release and does nothing more.
        Never,
    }

    /// The server's verbatim reply refusing a credential.
    const HELD_WRONGPASS: &str = "WRONGPASS invalid username-password pair or user is disabled.";

    /// Connects a [`production_reader_connector_bounded`] (connect bound
    /// `bound_secs`) to a fresh listener under a `MuxFault::Hold`. The
    /// listener refuses the setup `-WRONGPASS`; once the refused connection
    /// has closed, with the hold still pending, the test ends the hold as
    /// `end` says. Returns the connect's outcome, which must come back inside
    /// [`CUT_DEADLINE`].
    async fn held_connect_refused_wrongpass(
        bound_secs: u64,
        end: HoldEnd,
    ) -> Result<ReaderConn, ReaderConnectError> {
        use tokio::io::AsyncWriteExt;
        let (listener, port) = silent_listener().await;
        let connector = production_reader_connector_bounded(port, bound_secs);
        let cancel = CancellationToken::new();
        let (release, held) = tokio::sync::oneshot::channel::<()>();

        let (outcome, _kept_release) = tokio::join!(
            tokio::time::timeout(
                CUT_DEADLINE,
                connector.connect(
                    0,
                    None,
                    crate::observability::metrics::READER_CONNECT_CAUSE_BIND,
                    Some(crate::storage::stream_mux::MuxFault::Hold(held)),
                    &cancel,
                ),
            ),
            async {
                let mut conn = accept_stalled_setup(&listener).await;
                conn.write_all(format!("-{HELD_WRONGPASS}\r\n").as_bytes())
                    .await
                    .expect("the reply must be written");
                await_client_closed(
                    &mut conn,
                    "the refused setup must close its connection while the hold is pending",
                )
                .await;
                assert!(
                    !release.is_closed(),
                    "precondition: the connect is still held after its setup was refused"
                );
                match end {
                    HoldEnd::Send => {
                        release
                            .send(())
                            .expect("the held connect must still be waiting for its release");
                        None
                    }
                    HoldEnd::Drop => {
                        drop(release);
                        None
                    }
                    HoldEnd::Cancel => {
                        cancel.cancel();
                        Some(release)
                    }
                    HoldEnd::Never => Some(release),
                }
            },
        );
        outcome.expect(
            "the held connect must return inside the test's deadline; a refusal made while the \
             hold was pending, if missed, leaves the connect to wait out its bound",
        )
    }

    /// Asserts a held connect reported its setup's `-WRONGPASS` refusal: an
    /// `Auth` failure carrying the reply, not a `Timeout`.
    fn assert_reported_the_refusal(outcome: Result<ReaderConn, ReaderConnectError>) {
        match outcome {
            Err(ReaderConnectError::Failed { error, .. }) => {
                assert!(
                    is_server_auth_rejection(&error) && error.details().starts_with("WRONGPASS"),
                    "the held connect reports the setup's own refusal: {error:?}"
                );
            }
            other => panic!("a held connect whose setup was refused fails; got {other:?}"),
        }
    }

    /// A reader connect under a `Hold` whose setup the server refuses while
    /// the hold is still pending reports that refusal once the hold is
    /// released, at once, not as a `Timeout` at its connect bound (a minute,
    /// past [`CUT_DEADLINE`]).
    #[tokio::test]
    async fn a_held_reader_connect_refused_during_the_hold_reports_the_refusal_once_released() {
        assert_reported_the_refusal(held_connect_refused_wrongpass(60, HoldEnd::Send).await);
    }

    /// Dropping a `Hold`'s sender releases the hold as sending on it does: the
    /// refusal made during the hold is reported at once.
    #[tokio::test]
    async fn a_held_reader_connect_released_by_dropping_its_sender_reports_the_refusal() {
        assert_reported_the_refusal(held_connect_refused_wrongpass(60, HoldEnd::Drop).await);
    }

    /// A `Hold` never released is cut at the connect bound
    /// (`LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`, 3 s here) as a `Timeout`,
    /// even though the setup it delays has already been refused.
    ///
    /// The bound is real time, and the refusal (accept, setup read, the
    /// `-WRONGPASS` write, the client's close) must run its course inside it:
    /// a bound that fired first would abort the connect and drop the hold's
    /// receiver before the precondition check. 3 s leaves that room under
    /// load and stays inside [`CUT_DEADLINE`].
    #[tokio::test]
    async fn a_reader_connect_held_past_its_connect_bound_is_cut_as_a_timeout() {
        match held_connect_refused_wrongpass(3, HoldEnd::Never).await {
            Err(ReaderConnectError::Failed { error, .. }) => {
                assert!(
                    matches!(error.kind(), FredErrorKind::Timeout),
                    "the bound cuts the hold as a timeout: {error:?}"
                );
                assert_eq!(
                    error.details(),
                    "the reader connection was not up within 3 s \
                     (LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS)",
                    "the connector's own bound cut the held connect"
                );
            }
            other => panic!("a hold that outlasts the connect bound is cut; got {other:?}"),
        }
    }

    /// A reader connect cancelled (the reader's shutdown) while its `Hold`
    /// is pending returns `Cancelled`, not the refusal its setup met during
    /// the hold. The connect bound is a minute: only the cancellation ends
    /// the connect inside [`CUT_DEADLINE`].
    #[tokio::test]
    async fn a_reader_connect_cancelled_while_held_returns_cancelled() {
        match held_connect_refused_wrongpass(60, HoldEnd::Cancel).await {
            Err(ReaderConnectError::Cancelled) => {}
            other => panic!("a connect cancelled while held returns Cancelled; got {other:?}"),
        }
    }

    /// `push_faults_for_test` refuses a `MuxFault::Hold` at every site but
    /// `Connect`, queueing nothing there, and queues it at `Connect`.
    #[tokio::test]
    async fn push_faults_for_test_refuses_a_hold_at_every_site_but_connect() {
        use crate::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMux, StreamMuxKnobs};
        let pool = Builder::default_centralized()
            .build_pool(1)
            .expect("an unconnected pool builds");
        let mux = StreamMux::start(
            ReaderConnector::unconnected_for_test(pool.clone(), false),
            pool,
            false,
            1,
            StreamMuxKnobs::default(),
            Arc::new(Mutex::new(None)),
        );
        let hold = || MuxFault::Hold(tokio::sync::oneshot::channel::<()>().1);

        for site in [
            MuxFaultSite::Xread,
            MuxFaultSite::Wake,
            MuxFaultSite::HeadCheck,
            MuxFaultSite::Probe,
            MuxFaultSite::Keepalive,
            MuxFaultSite::Reauth,
        ] {
            let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                mux.push_faults_for_test(site, [hold()]);
            }))
            .expect_err("a Hold outside the Connect site must be refused");
            let message = refused
                .downcast_ref::<String>()
                .map(String::as_str)
                .unwrap_or_default();
            assert_eq!(
                message,
                format!("MuxFault::Hold applies to MuxFaultSite::Connect only, not {site:?}"),
                "the refusal names the site"
            );
            assert_eq!(
                mux.pending_faults_for_test(site),
                0,
                "{site:?}: nothing queued"
            );
        }

        mux.push_faults_for_test(MuxFaultSite::Connect, [hold()]);
        assert_eq!(mux.pending_faults_for_test(MuxFaultSite::Connect), 1);
    }

    // ── b.9uq round 4: `-UNBLOCKED` classification ────────────────────────
    //
    // `Blocking::Interrupt` above is only safe because the error it produces on
    // the interrupted command is understood downstream. Valkey replies
    // `-UNBLOCKED client unblocked via CLIENT UNBLOCK`, which fred's
    // `pretty_error` classifies as `ErrorKind::Unknown` — its match arms cover
    // only ERR / WRONGTYPE / NOAUTH / WRONGPASS / MOVED / ASK / CLUSTERDOWN /
    // `Invalid …` (`fred-10.1.0/src/protocol/utils.rs:96-128`) — and `Unknown`
    // is deliberately PERMANENT in `is_transient_fred_error`.
    //
    // The error is constructed here rather than round-tripped through
    // `pretty_error`, which lives in fred's private `mod protocol`
    // (`fred-10.1.0/src/lib.rs:55`) and is unreachable from this crate. The
    // shape — kind `Unknown`, `details` = the verbatim reply — is exactly what
    // that function returns for this reply.
    //
    // The reader-side half of the contract is pinned in
    // `storage::stream_mux::reader`'s tests: that this predicate is OR'd into
    // `is_retryable_reader_error` by
    // `tryagain_is_transient_in_the_shared_classifier_and_unblocked_stays_reader_only`,
    // and that a bare `Unknown` stays permanent by `reader_retry_classification`.

    #[test]
    fn is_client_unblocked_error_matches_the_server_unblocked_reply() {
        let e = fred::error::Error::new(
            FredErrorKind::Unknown,
            "UNBLOCKED client unblocked via CLIENT UNBLOCK",
        );
        assert!(
            is_client_unblocked_error(&e),
            "the predicate must recognise the `-UNBLOCKED` reply `Blocking::Interrupt` \
             provokes; missing it makes a stream reader treat a deliberate unblock as a \
             permanent failure and tear its subscriptions down"
        );
        assert!(
            !is_transient_fred_error(&e),
            "…and it must NOT be folded into the shared transient taxonomy: no pool other \
             than subscriber_pool issues a blocking command, so widening the shared \
             classifier would silently reclassify a genuine fred `Unknown` on the write \
             path from Internal to Unavailable"
        );
    }

    #[test]
    fn is_client_unblocked_error_rejects_other_unknown_and_other_kinds() {
        // Discrimination on the message: `Unknown` is fred's catch-all, so
        // keying on the kind alone would blanket-retry every unclassified
        // server error.
        assert!(
            !is_client_unblocked_error(&fred::error::Error::new(
                FredErrorKind::Unknown,
                "ERR unknown command 'FOO'",
            )),
            "an `Unknown`-kind error whose details do not start with `UNBLOCKED` is a \
             genuine unclassified server error and must stay permanent"
        );
        // Discrimination on the kind: a message that merely mentions the word
        // must not qualify if fred classified it as something else — a
        // `Timeout` carrying that text is a timeout, and the stream reader's
        // Timeout arm (wedge verdict → forced reconnect) must own it.
        assert!(
            !is_client_unblocked_error(&fred::error::Error::new(
                FredErrorKind::Timeout,
                "UNBLOCKED client unblocked via CLIENT UNBLOCK",
            )),
            "the predicate must require BOTH the `Unknown` kind and the `UNBLOCKED` \
             prefix — matching on the message alone would swallow a Timeout that happens \
             to quote the reply and skip the forced-reconnect self-heal"
        );
        assert!(
            !is_client_unblocked_error(&fred::error::Error::new(
                FredErrorKind::Unknown,
                "client unblocked via CLIENT UNBLOCK",
            )),
            "the match is anchored at the START of the details string (the RESP error \
             code position), not a substring search — an error merely quoting the phrase \
             must not qualify"
        );
    }

    // ── b.azh: the server's `-MOVED` reply on a single-node reader ────────
    //
    // A stream reader is a single-node client (on a cluster, connected to its
    // bound slot's primary alone), so a slot that moves reaches it as the raw
    // `-MOVED <slot> <host:port>` reply, which fred's `pretty_error` maps to
    // `ErrorKind::Cluster` with the whole reply as `details` — the same kind it
    // gives `-ASK` and `-CLUSTERDOWN` (`fred-10.1.0/src/protocol/utils.rs:113`).
    // `is_moved_redirect` is what sends the reader to re-resolve the slot's
    // owner and retarget; a miss leaves it retrying the same node until its
    // transient budget runs out, and a false positive on `-ASK` would retarget
    // a reader whose slot is only mid-migration. Hand-built errors, as for the
    // `-UNBLOCKED` block above: `pretty_error` is unreachable from this crate.

    /// The verbatim shape of a server `-MOVED` reply.
    const MOVED_REPLY: &str = "MOVED 3999 127.0.0.1:6381";

    #[test]
    fn is_moved_redirect_matches_the_server_moved_reply() {
        let e = fred::error::Error::new(FredErrorKind::Cluster, MOVED_REPLY);
        assert!(
            is_moved_redirect(&e),
            "a Cluster-kind `-MOVED` reply must be recognised: it is the reader's only signal that \
             its slot's owner moved"
        );
        assert!(
            is_transient_fred_error(&e),
            "the shared transient classifier is unchanged by the reader-only predicate: a \
             Cluster-kind error stays transient for every other caller"
        );
    }

    #[test]
    fn is_moved_redirect_rejects_ask_clusterdown_tryagain_and_the_budget_exhausted_shapes() {
        let not_moved: Vec<(FredErrorKind, &str)> = vec![
            // Same kind, other cluster replies: a slot mid-migration and a
            // cluster that is down are not an owner change.
            (FredErrorKind::Cluster, "ASK 3999 127.0.0.1:6381"),
            (FredErrorKind::Cluster, "CLUSTERDOWN The cluster is down"),
            // `-TRYAGAIN` has no `pretty_error` arm (fred classifies it
            // `Unknown`); neither kind may match.
            (
                FredErrorKind::Unknown,
                "TRYAGAIN Multiple keys request during rehashing of slot",
            ),
            (
                FredErrorKind::Cluster,
                "TRYAGAIN Multiple keys request during rehashing of slot",
            ),
            // The reply's text under another kind is not a server redirect.
            (FredErrorKind::Unknown, MOVED_REPLY),
            (FredErrorKind::Routing, MOVED_REPLY),
            // Anchored at the start, with the separating space.
            (FredErrorKind::Cluster, "ERR MOVED 3999 127.0.0.1:6381"),
            (FredErrorKind::Cluster, "MOVEDX 3999 127.0.0.1:6381"),
        ];
        for (kind, details) in not_moved
            .into_iter()
            .chain(BUDGET_EXHAUSTED_SHAPES.iter().cloned())
        {
            let e = fred::error::Error::new(kind.clone(), details);
            assert!(
                !is_moved_redirect(&e),
                "`{kind:?}` / {details:?} is not a `-MOVED` reply; matching it would retarget a \
                 reader whose slot's owner has not changed"
            );
        }
    }

    #[test]
    fn is_moved_redirect_leaves_the_shared_classifier_as_it_was_for_every_cluster_reply() {
        // The predicate is reader-only, like `is_client_unblocked_error`: the
        // shared classifier still calls every Cluster-kind reply transient
        // and `-UNBLOCKED` permanent, exactly as before the predicate existed.
        for details in [
            MOVED_REPLY,
            "ASK 3999 127.0.0.1:6381",
            "CLUSTERDOWN The cluster is down",
        ] {
            assert!(
                is_transient_fred_error(&fred::error::Error::new(FredErrorKind::Cluster, details)),
                "{details:?} stays transient in the shared taxonomy"
            );
        }
        assert!(
            !is_transient_fred_error(&fred::error::Error::new(
                FredErrorKind::Unknown,
                MOVED_REPLY
            )),
            "an Unknown-kind error is permanent whatever its text: the predicate did not widen \
             the shared classifier"
        );
    }

    // ── b.atj: the server's `-WRONGTYPE` reply ────────────────────────────
    //
    // `is_server_wrongtype_error` is the trigger for
    // `live_edit_snapshot_read_type_error_total` and for the ERROR line naming
    // the offending key. It steers no control flow — `classify_snapshot_read_err`
    // hands every error, matched or not, to the unchanged `map_snapshot_load_err`
    // — which is the whole licence for a message-matched predicate on a path
    // where b.zvd deliberately refused to add one for CLASSIFICATION. The cost
    // of a miss is therefore a mis-filed count, and the cost of a false positive
    // a spurious one; neither can truncate data. (Since b.muu a miss is not a
    // *lost* count: `count_object_key_refusal`'s second arm catches the same
    // error as `live_edit_snapshot_read_refused_total{kind="invalid_argument"}`,
    // because the two arms partition the permanent set. The type-error counter's
    // zero baseline survives a miss; what it cannot survive is a false
    // positive.)
    //
    // That bound is only as good as the predicate's discrimination, and the two
    // near-misses are what these tests pin. It must reject:
    //
    //   * this service's OWN `WRONGTYPE_KEY:`-tagged Lua verdict, which is
    //     already counted at its own site on `keyspace_type_error_total` — a
    //     match here would count one refusal on two counters and break both
    //     baselines, and the or-joined alert recipe would page twice for one key;
    //   * every other `InvalidArgument`, a kind fred also raises for client-side
    //     argument validation that never reached a server.
    //
    // As with the `-UNBLOCKED` and router-miss blocks either side of this one,
    // the errors are hand-built rather than round-tripped through fred:
    // `pretty_error` lives in fred's private `mod protocol` and is unreachable
    // from this crate. The shape asserted — kind `InvalidArgument`, `details` =
    // the verbatim reply — is what `fred-10.1.0/src/protocol/utils.rs:96-128`
    // produces for a leading `WRONGTYPE` token. These tests consequently cannot
    // detect fred changing that mapping; the closing guard is behavioural, in
    // `tests/integration/storage/b_atj_tests.rs`, where a real string is planted
    // on a real key and a real reply is classified.

    /// The verbatim reply Valkey sends for an operation against a key holding
    /// another type. Spelled out rather than paraphrased: the prefix anchor is
    /// what the predicate matches on.
    const WRONGTYPE_REPLY: &str =
        "WRONGTYPE Operation against a key holding the wrong kind of value";

    #[test]
    fn is_server_wrongtype_error_matches_the_server_wrongtype_reply() {
        let e = fred::error::Error::new(FredErrorKind::InvalidArgument, WRONGTYPE_REPLY);
        assert!(
            is_server_wrongtype_error(&e),
            "the predicate must recognise the reply every type-mismatched LRANGE / HGETALL / \
             ZRANGE on the snapshot read produces — missing it leaves the b.atj counter flat \
             through exactly the wedge it exists to page on"
        );
        assert!(
            !is_transient_fred_error(&e),
            "…and a wrong-typed key must stay PERMANENT: no retry can retype a key, so folding \
             it into the transient taxonomy would spin the update-layer retry loop against a \
             condition only an operator can clear"
        );
    }

    #[test]
    fn is_server_wrongtype_error_rejects_this_services_own_wrongtype_key_verdict() {
        // `scripts/keyspace_type_probe.lua` mints `WRONGTYPE_KEY:<key>:<actual>:
        // <expected>` and fred classifies a reply with no recognised leading
        // token as `Unknown` (`layer::classify_keyspace_probe_error` matches on
        // the tag, not the kind). It is already counted on
        // `keyspace_type_error_total` at that site.
        let e = fred::error::Error::new(
            FredErrorKind::Unknown,
            "WRONGTYPE_KEY:le:{42}:obj:fld:1:color:string:hash",
        );
        assert!(
            !is_server_wrongtype_error(&e),
            "the kind check is what separates the server's reply from this service's own tagged \
             Lua verdict. Matching it here would count one refused commit on BOTH \
             keyspace_type_error_total and snapshot_read_type_error_total, breaking both \
             baselines and double-firing the or-joined alert recipe for one key"
        );
    }

    #[test]
    fn is_server_wrongtype_error_rejects_other_invalid_argument_errors() {
        // fred maps a leading `Invalid` token onto the same kind
        // (`protocol/utils.rs:114-116`) and raises it from client-side argument
        // validation that never reached the server, so the kind alone
        // over-matches.
        assert!(
            !is_server_wrongtype_error(&fred::error::Error::new(
                FredErrorKind::InvalidArgument,
                "Invalid argument(s)",
            )),
            "`InvalidArgument` covers client-side argument validation too, so the prefix anchor \
             is load-bearing — without it a malformed command would tick a counter whose whole \
             remediation is \"an out-of-band writer put a foreign value on the le: keyspace\", \
             sending an operator to hunt a key that is fine"
        );
        // Anchored at the START of the details string (the RESP error-code
        // position), not a substring search.
        assert!(
            !is_server_wrongtype_error(&fred::error::Error::new(
                FredErrorKind::InvalidArgument,
                "ERR the reply mentioned WRONGTYPE in passing",
            )),
            "the match is anchored at the RESP error-code position; an error merely quoting the \
             word must not qualify"
        );
    }

    #[test]
    fn is_server_wrongtype_error_requires_the_invalid_argument_kind() {
        // A message that merely carries the reply must not qualify if fred
        // classified it as something else — a Timeout carrying that text is a
        // timeout, and it is TRANSIENT, so counting it would put a retriable
        // blip on a counter whose baseline is "zero forever on a healthy fleet".
        let e = fred::error::Error::new(FredErrorKind::Timeout, WRONGTYPE_REPLY);
        assert!(
            !is_server_wrongtype_error(&e),
            "the predicate must require BOTH the `InvalidArgument` kind and the `WRONGTYPE ` \
             prefix; matching on the message alone would tick the counter for a transient \
             timeout and destroy the zero baseline the alert recipe pages on"
        );
        assert!(
            is_transient_fred_error(&e),
            "…and that Timeout must stay transient, which is the reason the miss above matters: \
             the two classifications disagree, so a predicate keyed on the message alone would \
             be counting a retry as a keyspace fault"
        );
    }

    #[test]
    fn map_snapshot_load_err_wrongtype_still_maps_to_internal_unchanged() {
        // The b.zvd fail-closed contract, restated from the classifier's side:
        // b.atj added a WRONGTYPE predicate to this file and the whole safety
        // argument for it is that `map_snapshot_load_err` never learned about
        // it. This test is what makes "byte-identical refusal" a red test rather
        // than a claim in a doc-comment — the twelve sibling
        // `map_snapshot_load_err_*` tests cover every OTHER kind, and none of
        // them would fail if a WRONGTYPE carve-out were added here.
        let svc_err = map_snapshot_load_err(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "LRANGE children list (parent 0)",
            fred::error::Error::new(FredErrorKind::InvalidArgument, WRONGTYPE_REPLY),
        );
        match svc_err {
            crate::error::ServiceError::Internal(msg) => {
                assert!(
                    !msg.starts_with("transient I/O:"),
                    "a wrong-typed key is permanent — advertising a retry would spin a caller \
                     against a standing condition; got {msg:?}"
                );
                assert!(
                    msg.contains("LRANGE children list (parent 0)"),
                    "the op label routes an operator to the offending key and must survive; got \
                     {msg:?}"
                );
                assert!(
                    msg.contains("WRONGTYPE"),
                    "Valkey's own reply must reach the message so a type error is tellable from \
                     a transport blip; got {msg:?}"
                );
            }
            other => panic!(
                "a `WRONGTYPE` must still map to the non-retriable `Internal` b.zvd chose. \
                 Anything else means the counting predicate started steering classification, \
                 which re-opens the silent structural truncation b.zvd closed; got {other:?}"
            ),
        }
    }

    // ── b.c7a: fred's "Failed to route command" router miss ───────────────
    //
    // fred's router emits this from `write_command` when `Router::route` hands
    // back no connection AT ALL — the connection map is empty because the
    // router is mid-rebuild after a disconnect storm (the shape a managed-proxy
    // token rotation produces). It is not a routing *decision* failure: in
    // clustered mode `route` falls back to any writer when the slot's owner is
    // missing, so `None` means "nothing to write to", which is a connection-level
    // condition and squarely inside this classifier's charter.
    //
    // Pre-b.c7a the KIND decided it — `Unknown` in the shape this service
    // compiles (fred's `replicas` feature is off), `Routing` in the two
    // replica-gated sites — and both are permanent by kind, so a routing miss
    // surfaced as `Code::Internal` on the write path and ended live
    // subscriptions on the read path. The fix is a message-matched
    // carve-out folded into `is_transient_fred_error`; these tests pin BOTH
    // halves of it — the carve-out matches, and the kinds it rides on stay
    // permanent for every other message.
    //
    // As with the `-UNBLOCKED` block above, the errors here are hand-built
    // rather than round-tripped through fred: `write_command` lives in fred's
    // private `mod router` (`fred-10.1.0/src/lib.rs`) and is unreachable from
    // this crate. That restates the very assumption the predicate encodes, so
    // these tests cannot on their own detect fred changing the message text or
    // the kind. The closing guard is a REAL 3-shard-cluster reconnect test,
    // deliberately routed to ticket b.yxt (cluster-gated coverage) rather than
    // added here, where no router rebuild can be provoked.

    /// The verbatim message fred's router attaches to a routing miss, under
    /// whichever [`FredErrorKind`] the call site carries.
    fn router_miss(kind: FredErrorKind, details: &'static str) -> fred::error::Error {
        fred::error::Error::new(kind, details)
    }

    #[test]
    fn is_router_no_connection_error_matches_the_unknown_kind_router_miss() {
        // `fred-10.1.0/src/router/commands.rs:150` — the
        // `#[cfg(not(feature = "replicas"))]` branch, i.e. the shape THIS
        // service compiles.
        let e = router_miss(FredErrorKind::Unknown, "Failed to route command.");
        assert!(
            is_router_no_connection_error(&e),
            "the predicate must recognise fred's router miss under `Unknown` — that is the \
             exact error a post-token-rotation reconnect storm produced in the b.c7a incident"
        );
        assert!(
            is_transient_fred_error(&e),
            "…and the shared classifier must fold it in: fred calls `defer_reconnection` from \
             the same statement that emits it, so the router is already healing and the caller \
             only has to retry. Classifying it permanent surfaces `Code::Internal` on the write \
             path and clean-closes live subscriptions"
        );
    }

    #[test]
    fn is_router_no_connection_error_matches_the_routing_kind_and_the_replica_message() {
        // The two `replicas`-gated sites: `commands.rs:83` (same message, kind
        // `Routing`) and `commands.rs:76` ("… to replica."). Matched
        // permissively so enabling fred's `replicas` feature — or a future fred
        // that flips the variant — cannot silently regress the classification
        // back to permanent.
        for details in [
            "Failed to route command.",
            "Failed to route command to replica.",
        ] {
            let e = router_miss(FredErrorKind::Routing, details);
            assert!(
                is_router_no_connection_error(&e),
                "`Routing` + {details:?} is the same router miss under fred's replicas feature; \
                 the predicate must match on the message PREFIX so both site messages qualify"
            );
            assert!(
                is_transient_fred_error(&e),
                "`Routing` is permanent by kind, so the carve-out is the only thing that can \
                 make {details:?} retriable — if it stops matching, enabling fred's `replicas` \
                 feature silently reintroduces b.c7a"
            );
        }
    }

    #[test]
    fn is_router_no_connection_error_rejects_other_messages_under_the_same_kinds() {
        // Anti-creep: the carve-out is message-matched, so `Unknown` and
        // `Routing` must stay permanent for everything else. A regression that
        // widened this to the kinds themselves would blanket-retry every
        // unclassified server error — the precise failure `Unknown`'s permanent
        // classification exists to prevent.
        for (kind, details) in [
            (FredErrorKind::Unknown, "ERR unknown command 'FOO'"),
            (FredErrorKind::Routing, "Missing hash slot owner."),
            // Anchored at the START of the details string: a message that
            // merely quotes the phrase is not the router's own error.
            (
                FredErrorKind::Unknown,
                "Command failed: Failed to route command.",
            ),
        ] {
            let e = router_miss(kind.clone(), details);
            assert!(
                !is_router_no_connection_error(&e),
                "{kind:?} + {details:?} is not fred's router miss — the match is a prefix on \
                 `details`, not a kind check and not a substring search"
            );
            assert!(
                !is_transient_fred_error(&e),
                "{kind:?} + {details:?} must stay PERMANENT. b.c7a widened the classifier by \
                 exactly one message; letting the carve-out creep toward the whole kind would \
                 turn every genuine unclassified fred error into a silent retry"
            );
        }
    }

    #[test]
    fn is_router_no_connection_error_requires_one_of_the_two_router_kinds() {
        // Discrimination on the kind, mirroring the `-UNBLOCKED` predicate: an
        // error carrying the router's text under some OTHER kind is that other
        // kind's error and must be owned by its arm.
        let permanent_kind = router_miss(FredErrorKind::Protocol, "Failed to route command.");
        assert!(
            !is_router_no_connection_error(&permanent_kind),
            "a `Protocol` error is a malformed frame whatever its message says; matching on the \
             message alone would reclassify a real protocol bug as a retriable blip"
        );
        assert!(!is_transient_fred_error(&permanent_kind));

        let transient_kind = router_miss(FredErrorKind::Timeout, "Failed to route command.");
        assert!(
            !is_router_no_connection_error(&transient_kind),
            "a `Timeout` carrying that text is a timeout — the stream reader's Timeout arm (wedge \
             verdict → forced reconnect) must own it, and the routing carve-out deliberately \
             gets NO forced reconnect (fred already deferred one)"
        );
        assert!(
            is_transient_fred_error(&transient_kind),
            "…while still being transient by KIND, as it always was"
        );
    }

    #[test]
    fn router_miss_and_client_unblocked_predicates_do_not_overlap() {
        // Both carve-outs key on `ErrorKind::Unknown`, so each must reject the
        // other's message. They are not interchangeable: `-UNBLOCKED` is
        // reader-only (`is_retryable_reader_error`) because only the stream
        // readers issue blocking commands, while the router miss is in
        // the SHARED classifier because `write_command` is on every pool's path.
        // A predicate that swallowed the other's error would move an error
        // across that boundary.
        let unblocked = fred::error::Error::new(
            FredErrorKind::Unknown,
            "UNBLOCKED client unblocked via CLIENT UNBLOCK",
        );
        let routing = router_miss(FredErrorKind::Unknown, "Failed to route command.");
        assert!(
            !is_router_no_connection_error(&unblocked),
            "a `-UNBLOCKED` reply is not a router miss; folding it in here would put it into \
             the SHARED classifier by the back door, reclassifying write-path `Unknown`s"
        );
        assert!(
            !is_client_unblocked_error(&routing),
            "a router miss is not a deliberate unblock; matching it here would give it the \
             stream reader's cheap no-charge re-issue path instead of the backoff the shared \
             transient arm owes it"
        );
    }

    // ── b.cv4: fred's redirect retry-budget exhaustion family ─────────────
    //
    // The SECOND message-matched carve-out, and the one that carries the
    // Valkey-Cluster topology-change story. `ErrorKind::Cluster` in the kind
    // table above covers only the redirects fred *followed*; when the command's
    // attempt budget is spent — which for a stream reader's XREAD is always,
    // since it pins `CONSUMER_XREAD_MAX_ATTEMPTS = 1` — the same physical event
    // (a slot moved) surfaces instead as one of four exhaustion messages spread
    // over two kinds, `Unknown` and `Routing`, both permanent by kind. Without
    // the carve-out a routine resharding turns every single-shot read into a
    // non-retriable `Internal` and tears down every live subscription.
    //
    // These tests pin the same two halves the b.c7a block above does — the
    // carve-out matches its four shapes, and the kinds it rides on stay
    // permanent for every other message — plus a third the growing number of
    // `Unknown`-keyed predicates now demands: that the three carve-outs do not
    // overlap.
    //
    // As with the two blocks above, the errors are hand-built rather than
    // round-tripped through fred: every origin lives in fred's private
    // `mod router` / `mod protocol`. So these tests restate the assumption the
    // predicate encodes and cannot themselves detect fred changing the text or
    // the kind — which is exactly why the `(kind, message)` table they drive is
    // [`BUDGET_EXHAUSTED_SHAPES`], defined next to the predicate and shared with
    // the two integration sites, rather than a copy here. Its doc-comment holds
    // the per-shape provenance and the real-cluster observation; read it before
    // adding a row.

    #[test]
    fn is_router_retry_budget_exhausted_error_matches_all_four_fred_shapes() {
        for (kind, details) in &BUDGET_EXHAUSTED_SHAPES {
            let e = fred::error::Error::new(kind.clone(), *details);
            assert!(
                is_router_retry_budget_exhausted_error(&e),
                "{kind:?} + {details:?} is fred giving up on a redirect it had no attempt budget \
                 left to follow; the predicate must recognise all four shapes, not just the one \
                 a real cluster happened to produce first"
            );
            assert!(
                is_transient_fred_error(&e),
                "…and the shared classifier must fold {kind:?} + {details:?} in. Both kinds are \
                 permanent BY KIND, so this carve-out is the only thing standing between a \
                 routine resharding and every live subscriber being torn down / every \
                 single-shot read failing with a non-retriable Internal"
            );
        }
    }

    #[test]
    fn is_router_retry_budget_exhausted_error_matches_on_prefix_not_exact_equality() {
        // The match is `starts_with`, not `==`, for the same reason
        // `is_router_no_connection_error`'s is: a future fred that appends
        // context to one of these messages must not silently drop back to
        // PERMANENT. Trailing text is the one direction the anchor tolerates —
        // leading text is rejected by the anti-creep test below.
        for (kind, details) in &BUDGET_EXHAUSTED_SHAPES {
            let extended = format!("{details} (attempts_remaining: 0, server: 10.0.0.7:6379)");
            let e = fred::error::Error::new(kind.clone(), extended.clone());
            assert!(
                is_router_retry_budget_exhausted_error(&e),
                "{kind:?} + {extended:?} is the same exhaustion with context appended; a match \
                 that required exact equality would reclassify it permanent on a fred upgrade"
            );
            assert!(
                is_transient_fred_error(&e),
                "{kind:?} + {extended:?} must reach the shared classifier as transient too"
            );
        }
    }

    #[test]
    fn is_router_retry_budget_exhausted_error_requires_each_message_under_its_own_kind() {
        // The predicate matches on the KIND first and discriminates on the
        // message inside that arm — the two are not independent filters. Flip
        // each shape onto the other router kind and it must stop matching:
        // `Unknown`/"Max attempts reached." is not a shape fred emits, and
        // accepting it would be widening on inspection alone against a kind that
        // is fred's catch-all for every un-parsed server error.
        for (kind, details) in &BUDGET_EXHAUSTED_SHAPES {
            let wrong_kind = match kind {
                FredErrorKind::Unknown => FredErrorKind::Routing,
                _ => FredErrorKind::Unknown,
            };
            let e = fred::error::Error::new(wrong_kind.clone(), *details);
            assert!(
                !is_router_retry_budget_exhausted_error(&e),
                "{wrong_kind:?} + {details:?} is not a shape fred emits — {details:?} comes from \
                 exactly one site, under {kind:?}. Matching it under the sibling kind means the \
                 kind arm has stopped discriminating"
            );
            assert!(
                !is_transient_fred_error(&e),
                "{wrong_kind:?} + {details:?} must therefore stay PERMANENT"
            );
        }
    }

    #[test]
    fn is_router_retry_budget_exhausted_error_rejects_other_messages_under_the_same_kinds() {
        // Anti-creep, the half that matters: neither kind may be widened
        // wholesale. Both are load-bearing carriers of genuinely permanent
        // errors, and a regression that dropped the message match would turn
        // every one of them into a silent retry.
        for (kind, details) in [
            // A real CROSSSLOT is a caller bug — a multi-key command spanning
            // slots — and retrying it just repeats it. It arrives under
            // `Unknown` (fred's `pretty_error` has no CROSSSLOT arm), i.e. the
            // same kind as the observed exhaustion shape.
            (
                FredErrorKind::Unknown,
                "CROSSSLOT Keys in request don't hash to the same slot",
            ),
            (FredErrorKind::Unknown, "ERR unknown command 'FOO'"),
            // `Routing` also carries fred's `replicas`-gated routing failures
            // and whatever fred adds next. A future message is PERMANENT until
            // someone brings its file:line and, ideally, a cluster observation.
            (FredErrorKind::Routing, "some future fred message"),
            // Anchored at the START of the details string (the RESP error-code
            // position), never a substring search.
            (
                FredErrorKind::Unknown,
                "Command failed: Too many failed write attempts.",
            ),
            (
                FredErrorKind::Routing,
                "Command failed: Max attempts reached.",
            ),
        ] {
            let e = fred::error::Error::new(kind.clone(), details);
            assert!(
                !is_router_retry_budget_exhausted_error(&e),
                "{kind:?} + {details:?} is not a redirect budget exhaustion — the match is a \
                 kind-gated PREFIX on `details`, not a bare kind check and not a substring search"
            );
            assert!(
                !is_transient_fred_error(&e),
                "{kind:?} + {details:?} must stay PERMANENT. b.cv4 widened the classifier by \
                 exactly four messages; letting the carve-out creep toward the whole kind would \
                 turn every genuine unclassified fred error into a silent retry — and `Internal` \
                 is the signal that asks an operator for attention"
            );
        }
    }

    #[test]
    fn is_router_retry_budget_exhausted_error_rejects_every_other_kind() {
        // Discrimination on the kind, mirroring the two predicates above: an
        // error carrying an exhaustion message under some OTHER kind belongs to
        // that kind's arm, whichever way that arm classifies it.
        for (_, details) in &BUDGET_EXHAUSTED_SHAPES {
            let permanent_kind = fred::error::Error::new(FredErrorKind::Protocol, *details);
            assert!(
                !is_router_retry_budget_exhausted_error(&permanent_kind),
                "a `Protocol` error is a malformed frame whatever its message says; matching on \
                 the message alone would reclassify a real protocol bug ({details:?}) as a \
                 retriable topology blip"
            );
            assert!(
                !is_transient_fred_error(&permanent_kind),
                "…and `Protocol` + {details:?} must stay PERMANENT in the shared classifier: \
                 `Protocol` is transient by neither kind nor carve-out, so a `true` here means \
                 the carve-out stopped gating on the kind"
            );

            let transient_kind = fred::error::Error::new(FredErrorKind::Timeout, *details);
            assert!(
                !is_router_retry_budget_exhausted_error(&transient_kind),
                "a `Timeout` carrying {details:?} is a timeout — the stream reader's Timeout arm \
                 (wedge verdict → forced reconnect) must own it, and this family deliberately \
                 gets NO forced reconnect (fred already ran `sync_cluster_with_policy`)"
            );
            assert!(
                is_transient_fred_error(&transient_kind),
                "…while still being transient by KIND, as it always was"
            );
        }
    }

    #[test]
    fn budget_exhaustion_router_miss_and_client_unblocked_predicates_do_not_overlap() {
        // THREE carve-outs now key on `ErrorKind::Unknown`. Each must reject the
        // others' messages: they are not interchangeable, and they differ in
        // remediation. `-UNBLOCKED` is reader-only (cheap no-charge re-issue,
        // deliberately NOT in the shared classifier); the router miss and this
        // family are both shared-classifier + backoff with NO forced reconnect,
        // but they are distinct conditions with distinct evidence, and folding
        // one into the other's predicate would let a single future widening
        // silently cover both.
        for (kind, details) in &BUDGET_EXHAUSTED_SHAPES {
            let e = fred::error::Error::new(kind.clone(), *details);
            assert!(
                !is_router_no_connection_error(&e),
                "{kind:?} + {details:?} is a spent attempt budget, not an empty connection map; \
                 fred had somewhere to write and refused to spend the write"
            );
            assert!(
                !is_client_unblocked_error(&e),
                "{kind:?} + {details:?} is not a deliberate `CLIENT UNBLOCK`; matching it there \
                 would hand a cluster topology change the reader-only cheap re-issue path"
            );
        }

        // …and in the other direction: the router miss stays exclusively
        // `is_router_no_connection_error`'s, on every one of its three site
        // messages.
        for (kind, details) in [
            (FredErrorKind::Unknown, "Failed to route command."),
            (FredErrorKind::Routing, "Failed to route command."),
            (
                FredErrorKind::Routing,
                "Failed to route command to replica.",
            ),
        ] {
            let e = fred::error::Error::new(kind.clone(), details);
            assert!(
                is_router_no_connection_error(&e),
                "sanity: {kind:?} + {details:?} is still the b.c7a router miss"
            );
            assert!(
                !is_router_retry_budget_exhausted_error(&e),
                "…and ONLY that: a router miss is fred having no connection at all, a different \
                 condition with different evidence from a redirect it could not afford to follow"
            );
        }

        let unblocked = fred::error::Error::new(
            FredErrorKind::Unknown,
            "UNBLOCKED client unblocked via CLIENT UNBLOCK",
        );
        assert!(
            is_client_unblocked_error(&unblocked),
            "sanity: the `-UNBLOCKED` reply is still the reader-only predicate's"
        );
        assert!(
            !is_router_retry_budget_exhausted_error(&unblocked),
            "…and ONLY that: folding it in here would put a reader-only condition into the \
             SHARED classifier by the back door, reclassifying write-path `Unknown`s"
        );
    }

    // ── b.xd1: the store not serving (`-LOADING`, `-BUSY`) ─────────────────
    //
    // The THIRD message-matched carve-out family in the shared classifier, and
    // the first keyed on a reply the backing store itself sends: a node
    // replaying its dataset answers `-LOADING`, a node behind a long script or
    // module command answers `-BUSY`, and fred maps neither token, so both
    // arrive as `ErrorKind::Unknown` — permanent by kind. Without the carve-out
    // a routine backing-store restart failed every in-flight request `INTERNAL`
    // instead of being retried.
    //
    // Two halves, as for the router families above: every shape in
    // [`STORE_NOT_SERVING_SHAPES`] matches (bare, and after an `EXEC`'s
    // `EXECABORT Transaction discarded because of: ` prefix), and every near
    // miss — `-BUSYGROUP` / `-BUSYKEY`, a token without its trailing space, the
    // same text under another kind — does not. The four store write refusals
    // stay non-transient (pinned below). What the mappers answer — `Aborted` /
    // `Unavailable` for a not-serving reply, a plain `Unavailable` naming the
    // op and condition for a store write refusal — is pinned, with the gRPC
    // code, in `tests/integration/storage/transient_classification_test.rs`
    // §5d / §5e.
    //
    // The errors are hand-built, as in the blocks above: the predicate's
    // assumption about fred's mapping is restated here, and the proxy-staged
    // integration tests (real fred reply parsing) are what can catch fred
    // changing it.

    #[test]
    fn is_store_not_serving_reply_matches_every_shape_bare_and_after_the_execabort_prefix() {
        for reply in STORE_NOT_SERVING_SHAPES {
            let bare = unknown_err(reply);
            assert!(
                is_store_not_serving_reply(&bare),
                "{reply:?} is the store not serving commands for a while; the predicate must \
                 recognise every row of the shape table"
            );
            assert!(
                is_transient_fred_error(&bare),
                "…and the shared classifier must fold {reply:?} in, or a backing-store restart \
                 fails every in-flight request with a non-retriable INTERNAL"
            );

            let discarded = format!("EXECABORT Transaction discarded because of: {reply}");
            assert!(
                is_store_not_serving_reply(&unknown_err(&discarded)),
                "an EXEC refused as a whole by {reply:?} carries it after the EXECABORT prefix"
            );
            assert!(
                is_transient_fred_error(&unknown_err(&discarded)),
                "{discarded:?} ran nothing, so it is transient too"
            );
        }
    }

    #[test]
    fn store_not_serving_shapes_lead_with_the_default_restart_reply() {
        assert_eq!(
            STORE_NOT_SERVING_SHAPES[0], "LOADING Valkey is loading the dataset in memory",
            "row 0 is what call sites use as \"the restart reply\" (Valkey's default \
             configuration); it must stay first"
        );
        for reply in STORE_NOT_SERVING_SHAPES {
            assert!(
                reply.starts_with("LOADING ") || reply.starts_with("BUSY "),
                "{reply:?}: every row is one of the two tokens, with its trailing space"
            );
        }
    }

    #[test]
    fn is_store_not_serving_reply_rejects_near_misses_under_unknown() {
        // Anti-creep: prefix-anchored, on the token WITH its trailing space.
        for reply in [
            // A consumer-group or `RESTORE` conflict: permanent.
            "BUSYGROUP Consumer Group name already exists",
            "BUSYKEY Target key name already exists.",
            "EXECABORT Transaction discarded because of: BUSYKEY Target key name already exists.",
            // The token without its trailing space, or a longer token.
            "BUSY",
            "LOADING",
            "LOADINGX Valkey is loading the dataset in memory",
            // Not in the RESP error-code position.
            "ERR unknown command 'LOADING'",
            " LOADING Valkey is loading the dataset in memory",
            // `EXEC`'s reply after a queued refusal names no cause.
            "EXECABORT Transaction discarded because of previous errors.",
            // Another availability reply b.xd1 did not classify. (A slot
            // migration's `-TRYAGAIN` is transient through its own family,
            // b.vn4, below; it is still not this predicate's.)
            "MASTERDOWN Link with MASTER is down and replica-serve-stale-data is set to 'no'.",
            "",
        ] {
            let e = unknown_err(reply);
            assert!(
                !is_store_not_serving_reply(&e),
                "{reply:?} is not the store's not-serving reply"
            );
            assert!(
                !is_transient_fred_error(&e),
                "{reply:?} must stay PERMANENT in the shared classifier: the carve-out matches \
                 two tokens with their trailing space, not the `Unknown` kind"
            );
        }
    }

    #[test]
    fn is_store_not_serving_reply_requires_the_unknown_kind() {
        // The same text under another kind is that kind's error: fred hands a
        // server reply over as `Unknown`, so a `Protocol` / `Routing` /
        // `InvalidArgument` error carrying it was not the store's reply.
        for kind in [
            FredErrorKind::Protocol,
            FredErrorKind::Routing,
            FredErrorKind::InvalidArgument,
        ] {
            for reply in STORE_NOT_SERVING_SHAPES {
                let e = fred::error::Error::new(kind.clone(), reply);
                assert!(
                    !is_store_not_serving_reply(&e),
                    "{kind:?} + {reply:?} is not the store's reply"
                );
                assert!(
                    !is_transient_fred_error(&e),
                    "{kind:?} + {reply:?} must stay permanent: neither the kind nor any \
                     carve-out makes it transient"
                );
            }
        }
    }

    #[test]
    fn store_not_serving_and_the_other_message_predicates_do_not_overlap() {
        // The not-serving family is backoff-only and never a store write
        // refusal; the router families and `-UNBLOCKED` are different
        // conditions with different evidence. None may claim another's text.
        for reply in STORE_NOT_SERVING_SHAPES {
            let e = unknown_err(reply);
            assert!(!is_router_no_connection_error(&e), "{reply:?}");
            assert!(!is_router_retry_budget_exhausted_error(&e), "{reply:?}");
            assert!(!is_client_unblocked_error(&e), "{reply:?}");
            assert_eq!(
                store_write_refusal(&e),
                None,
                "{reply:?} clears by itself and is retried; a store write refusal is not"
            );
        }
        for (reply, _) in STORE_REFUSAL_REPLIES {
            assert!(
                !is_store_not_serving_reply(&unknown_err(reply)),
                "{reply:?} is a store write refusal, which can persist: it must not be matched \
                 as not-serving, or the update retry loop would spin against a full store"
            );
        }
        for (kind, details) in BUDGET_EXHAUSTED_SHAPES.iter().cloned().chain([
            (FredErrorKind::Unknown, "Failed to route command."),
            (
                FredErrorKind::Unknown,
                "UNBLOCKED client unblocked via CLIENT UNBLOCK",
            ),
        ]) {
            assert!(
                !is_store_not_serving_reply(&fred::error::Error::new(kind.clone(), details)),
                "{kind:?} + {details:?} belongs to another predicate"
            );
        }
    }

    // ── b.vn4: a slot migration's `-TRYAGAIN` ──────────────────────────────
    //
    // The FOURTH message-matched carve-out family in the shared classifier: a
    // cluster node refuses a command whose keys are split between a migrating
    // slot's source and target with `-TRYAGAIN`, before the command runs, and
    // fred maps no `TRYAGAIN` token, so it arrives as `ErrorKind::Unknown` —
    // permanent by kind. Before b.vn4 only the stream readers retried it (a
    // reader-only arm), and every other multi-key call site answered it
    // `INTERNAL`. The same two halves as the families above: the verbatim reply
    // matches, under `Unknown` only, and every near miss — a longer token, the
    // token without its trailing space, the token after another error code, a
    // script's own in-migration refusal — does not. What the mappers answer is
    // pinned in `tests/integration/storage/transient_classification_test.rs`
    // §5f; the real reply's text in the cluster-gated
    // `b_vn4_tryagain_test::slot_migration_disruptive_…` test.

    #[test]
    fn is_tryagain_reply_matches_the_verbatim_reply_under_unknown() {
        let e = unknown_err(TRYAGAIN_REPLY);
        assert!(
            is_tryagain_reply(&e),
            "{TRYAGAIN_REPLY:?} is a slot migration refusing a multi-key command"
        );
        assert!(
            is_transient_fred_error(&e),
            "…and the shared classifier must fold it in, or every multi-key call site fails a \
             live slot migration with a non-retriable INTERNAL"
        );
        assert!(
            TRYAGAIN_REPLY.starts_with("TRYAGAIN "),
            "the shared constant is the reply the predicate exists for"
        );

        // Prefix-anchored, not exact: text a future server appends after the
        // token does not drop the reply back to permanent.
        let extended = unknown_err(&format!("{TRYAGAIN_REPLY} 1234"));
        assert!(is_tryagain_reply(&extended));
        assert!(is_transient_fred_error(&extended));
    }

    #[test]
    fn is_tryagain_reply_rejects_near_misses_under_unknown() {
        for reply in [
            // A longer token, or the token without its trailing space.
            "TRYAGAINX Multiple keys request during rehashing of slot",
            "TRYAGAIN",
            // Not in the RESP error-code position.
            "ERR TRYAGAIN Multiple keys request during rehashing of slot",
            " TRYAGAIN Multiple keys request during rehashing of slot",
            // A node answers `-TRYAGAIN` to the command itself, never as an
            // `EXECABORT`'s cause.
            "EXECABORT Transaction discarded because of: TRYAGAIN Multiple keys request during \
             rehashing of slot",
            // A script's own command refused inside a migrating slot: a
            // different text, which `storage::diff_read` handles, with or
            // without the `ERR ` code `redis.pcall` adds.
            "ERR Unable to execute command 'hgetall' in script because undeclared keys were \
             accessed during rehashing of the slot",
            "Unable to execute command 'hgetall' in script because undeclared keys were accessed \
             during rehashing of the slot",
            "",
        ] {
            let e = unknown_err(reply);
            assert!(
                !is_tryagain_reply(&e),
                "{reply:?} is not a slot migration's -TRYAGAIN"
            );
            assert!(
                !is_transient_fred_error(&e),
                "{reply:?} must stay PERMANENT in the shared classifier: the carve-out matches \
                 one token with its trailing space, not the `Unknown` kind"
            );
        }
    }

    #[test]
    fn is_tryagain_reply_requires_the_unknown_kind() {
        // fred hands a server reply over as `Unknown`; the same text under
        // another kind is that kind's error, whichever way it classifies.
        for kind in [
            FredErrorKind::Protocol,
            FredErrorKind::Routing,
            FredErrorKind::InvalidArgument,
        ] {
            let e = fred::error::Error::new(kind.clone(), TRYAGAIN_REPLY);
            assert!(!is_tryagain_reply(&e), "{kind:?} + {TRYAGAIN_REPLY:?}");
            assert!(
                !is_transient_fred_error(&e),
                "{kind:?} + {TRYAGAIN_REPLY:?} must stay permanent: neither the kind nor any \
                 carve-out makes it transient"
            );
        }
        for kind in [FredErrorKind::Cluster, FredErrorKind::Timeout] {
            let e = fred::error::Error::new(kind.clone(), TRYAGAIN_REPLY);
            assert!(
                !is_tryagain_reply(&e),
                "{kind:?} + {TRYAGAIN_REPLY:?} is not the reply, however the kind classifies"
            );
            assert!(
                is_transient_fred_error(&e),
                "…and {kind:?} stays transient by KIND, as it always was"
            );
        }
    }

    #[test]
    fn tryagain_and_the_other_message_predicates_do_not_overlap() {
        // Backoff-only, never a store write refusal, never a redirect to
        // retarget on; and no other family's text is a `-TRYAGAIN`.
        let tryagain = unknown_err(TRYAGAIN_REPLY);
        assert!(!is_router_no_connection_error(&tryagain));
        assert!(!is_router_retry_budget_exhausted_error(&tryagain));
        assert!(!is_store_not_serving_reply(&tryagain));
        assert!(!is_client_unblocked_error(&tryagain));
        assert!(!is_moved_redirect(&tryagain));
        assert_eq!(
            store_write_refusal(&tryagain),
            None,
            "a migration clears by itself and is retried; a store write refusal is not"
        );
        for (kind, details) in BUDGET_EXHAUSTED_SHAPES.iter().cloned().chain([
            (FredErrorKind::Unknown, "Failed to route command."),
            (
                FredErrorKind::Unknown,
                "UNBLOCKED client unblocked via CLIENT UNBLOCK",
            ),
            (FredErrorKind::Cluster, MOVED_REPLY),
            (FredErrorKind::Cluster, "ASK 3999 127.0.0.1:6381"),
            (FredErrorKind::Cluster, "CLUSTERDOWN The cluster is down"),
        ]) {
            assert!(
                !is_tryagain_reply(&fred::error::Error::new(kind.clone(), details)),
                "{kind:?} + {details:?} belongs to another predicate"
            );
        }
        for reply in STORE_NOT_SERVING_SHAPES
            .iter()
            .copied()
            .chain(STORE_REFUSAL_REPLIES.iter().map(|(reply, _)| *reply))
        {
            assert!(
                !is_tryagain_reply(&unknown_err(reply)),
                "{reply:?} belongs to another predicate"
            );
        }
    }

    // ── b.cyu (b.bm5 (a)): fred's `Transaction` router giving up ───────────
    //
    // The FIFTH message-matched carve-out family in the shared classifier.
    // fred runs a `Transaction` (`multi()` … `exec()`) through its own router
    // loop, which spells its exhaustion differently from the per-command
    // family above: `Unknown` + "Max redirections exceeded" for a `MOVED` /
    // `-ASK` past the txn pool's zero redirection budget, and `Unknown` +
    // "Max attempts exceeded" for a failed drain or a route miss with no error
    // to forward. Before b.cyu both were permanent, so a slot migration
    // answered `DeleteSession` and `CreateLayer` `INTERNAL`. The same two
    // halves as the families above: both verbatim texts match, under
    // `Unknown` only, and no near miss or other family's text does. What the
    // mappers answer is pinned in
    // `tests/integration/storage/transient_classification_test.rs` §5g; the
    // texts reaching the call sites through a real fred `Transaction` in
    // `b_cyu_txn_redirect_test`; a real cluster's redirect in the
    // cluster-gated `b_cyu_slot_migration_test`.

    /// The two texts fred's transaction router manufactures, as the vendored
    /// fred source builds them. Read off the source rather than restated, so
    /// the table cannot drift from what fred sends.
    const FRED_TRANSACTION_ROUTER_SOURCE: &str =
        include_str!("../../vendor/fred-10.1.0/src/router/transactions.rs");

    #[test]
    fn the_transaction_router_shape_table_is_exactly_freds_two_texts() {
        assert_eq!(
            TRANSACTION_ROUTER_EXHAUSTED_SHAPES
                .iter()
                .map(|(kind, details)| (kind.clone(), *details))
                .collect::<Vec<_>>(),
            vec![
                (FredErrorKind::Unknown, "Max redirections exceeded"),
                (FredErrorKind::Unknown, "Max attempts exceeded"),
            ],
            "the table holds exactly fred's two transaction-router texts, both `Unknown`"
        );
        for (_, details) in &TRANSACTION_ROUTER_EXHAUSTED_SHAPES {
            let built = format!("Error::new(ErrorKind::Unknown, \"{details}\")");
            assert!(
                FRED_TRANSACTION_ROUTER_SOURCE.contains(&built),
                "the vendored fred transaction router must build `{built}`; if fred's wording \
                 changed, the classifier no longer matches what a redirected transaction sends"
            );
        }
    }

    #[test]
    fn is_transaction_router_exhausted_error_matches_both_texts_under_unknown() {
        for (kind, details) in &TRANSACTION_ROUTER_EXHAUSTED_SHAPES {
            let e = fred::error::Error::new(kind.clone(), *details);
            assert!(
                is_transaction_router_exhausted_error(&e),
                "{kind:?} + {details:?} is fred's transaction router giving up"
            );
            assert!(
                is_transient_fred_error(&e),
                "…and the shared classifier must fold it in, or a slot migration answers \
                 DeleteSession and CreateLayer with a non-retriable INTERNAL"
            );
            // Prefix-anchored, not exact: text appended after the message
            // keeps the match.
            let extended = unknown_err(&format!("{details} (attempt 2)"));
            assert!(is_transaction_router_exhausted_error(&extended));
            assert!(is_transient_fred_error(&extended));
        }
    }

    #[test]
    fn is_transaction_router_exhausted_error_requires_the_unknown_kind() {
        // fred builds both texts as `Unknown`; the same text under another
        // kind is that kind's error and must not ride this carve-out. The
        // non-transient kinds stay permanent.
        for kind in [
            FredErrorKind::Routing,
            FredErrorKind::Protocol,
            FredErrorKind::InvalidArgument,
            FredErrorKind::InvalidCommand,
        ] {
            for (_, details) in &TRANSACTION_ROUTER_EXHAUSTED_SHAPES {
                let e = fred::error::Error::new(kind.clone(), *details);
                assert!(
                    !is_transaction_router_exhausted_error(&e),
                    "{kind:?} + {details:?} is not fred's transaction router"
                );
                assert!(
                    !is_transient_fred_error(&e),
                    "{kind:?} + {details:?} must stay permanent: neither the kind nor any \
                     carve-out makes it transient"
                );
            }
        }
    }

    #[test]
    fn is_transaction_router_exhausted_error_rejects_near_misses_under_unknown() {
        for reply in [
            // Not at the start of the message.
            "ERR Max redirections exceeded",
            " Max redirections exceeded",
            "EXECABORT Transaction discarded because of: Max attempts exceeded",
            // A different word.
            "Max redirection exceeded",
            "Max redirections reached",
            "Max attempt exceeded",
            // Truncated.
            "Max redirections",
            "Max attempts",
            // fred's other transaction-router texts, which are not exhaustion.
            "Unexpected DISCARD response.",
            "Missing transaction response.",
            "Invalid cluster redirection.",
            "",
        ] {
            let e = unknown_err(reply);
            assert!(
                !is_transaction_router_exhausted_error(&e),
                "{reply:?} is not the transaction router giving up"
            );
            assert!(
                !is_transient_fred_error(&e),
                "{reply:?} must stay PERMANENT in the shared classifier: the carve-out matches \
                 two texts at the start, not the `Unknown` kind"
            );
        }
    }

    #[test]
    fn transaction_router_exhaustion_and_the_other_message_predicates_do_not_overlap() {
        // Backoff-only, never a store write refusal, never a redirect to
        // retarget a reader on, never a slot migration's `-TRYAGAIN`; and no
        // other family's text is the transaction router's.
        for (kind, details) in &TRANSACTION_ROUTER_EXHAUSTED_SHAPES {
            let e = fred::error::Error::new(kind.clone(), *details);
            assert!(!is_router_no_connection_error(&e), "{details:?}");
            assert!(!is_router_retry_budget_exhausted_error(&e), "{details:?}");
            assert!(!is_store_not_serving_reply(&e), "{details:?}");
            assert!(!is_tryagain_reply(&e), "{details:?}");
            assert!(!is_client_unblocked_error(&e), "{details:?}");
            assert!(!is_moved_redirect(&e), "{details:?}");
            assert_eq!(
                store_write_refusal(&e),
                None,
                "{details:?}: a moved slot clears by itself; a store write refusal does not"
            );
        }
        for (kind, details) in BUDGET_EXHAUSTED_SHAPES.iter().cloned().chain([
            (FredErrorKind::Unknown, "Failed to route command."),
            (
                FredErrorKind::Unknown,
                "UNBLOCKED client unblocked via CLIENT UNBLOCK",
            ),
            (FredErrorKind::Unknown, TRYAGAIN_REPLY),
            (FredErrorKind::Cluster, MOVED_REPLY),
            (FredErrorKind::Cluster, "ASK 3999 127.0.0.1:6381"),
        ]) {
            assert!(
                !is_transaction_router_exhausted_error(&fred::error::Error::new(
                    kind.clone(),
                    details
                )),
                "{kind:?} + {details:?} belongs to another predicate"
            );
        }
        for reply in STORE_NOT_SERVING_SHAPES
            .iter()
            .copied()
            .chain(STORE_REFUSAL_REPLIES.iter().map(|(reply, _)| *reply))
        {
            assert!(
                !is_transaction_router_exhausted_error(&unknown_err(reply)),
                "{reply:?} belongs to another predicate"
            );
        }
    }

    /// The routing re-sync after a transaction fails
    /// (`ValkeyClient::resync_txn_routing_after_redirect`) runs only for the
    /// redirect text: a redirect is the one shape that says the member's
    /// routing table is stale. "Max attempts exceeded" (a failed drain or
    /// route miss, after which fred has already reconnected) and every other
    /// error leave the routing alone.
    #[test]
    fn only_the_redirect_text_selects_the_routing_resync() {
        assert!(is_transaction_redirections_exhausted_error(&unknown_err(
            "Max redirections exceeded"
        )));
        for e in [
            unknown_err("Max attempts exceeded"),
            fred::error::Error::new(FredErrorKind::Routing, "Max redirections exceeded"),
            unknown_err("ERR Max redirections exceeded"),
            unknown_err(TRYAGAIN_REPLY),
            fred::error::Error::new(FredErrorKind::Cluster, MOVED_REPLY),
            fred::error::Error::new(FredErrorKind::IO, "connection reset"),
        ] {
            assert!(
                !is_transaction_redirections_exhausted_error(&e),
                "{e:?} does not say the routing is stale"
            );
        }
    }

    // ── b.xax access-token-file tick cadence ≥ 3 min headroom (SDD §15.4, SR-4.2) ──────
    //
    // Pins the production `VALKEY_TOKEN_REAUTH_*` constants against this module's
    // own invariant: the maximum tick interval leaves ≥ 3 min of a token's
    // lifetime unused (the 180 s floor SDD §15.4 uses for the refresh lead).
    // The complementary test
    // `token_file_credentials::tests::refresh_interval_ceiling_leaves_three_min_headroom`
    // proves the same *type property* over the provider's TEST-fixture band
    // (30s/10s) — it does NOT read these production constants, so a future bump
    // of the shipped constants into the <3-min-headroom zone would slip past it.
    // This test closes that gap by asserting over the actual shipped `Duration`s.
    #[test]
    fn token_reauth_ceiling_leaves_three_min_headroom() {
        // The largest realized tick interval is `base + max_jitter`.
        let ceiling = VALKEY_TOKEN_REAUTH_BASE_INTERVAL + VALKEY_TOKEN_REAUTH_JITTER_BOUND;

        // Formula pin: 60 s base + 30 s jitter bound = 90 s ceiling. Shortened
        // from 300 + 60 by b.7td, which gated the broadcast on an actual token
        // rotation: most ticks now send nothing, so a short cadence costs almost
        // nothing and lands inside the rotation window far more often.
        assert_eq!(
            ceiling,
            Duration::from_secs(90),
            "max access-token-file tick interval must be base (60 s) + jitter bound (30 s) = 90 s"
        );

        // The ≥ 3 min proactive re-read lead the second `ServiceTokenProvider`
        // is configured with (SDD §15.4 / SR-4.2). Post-b.d6b the lead withholds
        // nothing — `fetch()` serves any unexpired cached token — so what this
        // headroom check pins is that the tick ceiling stays far enough inside a
        // token lifetime for the provider to have re-read well before the
        // credential dies.
        const THREE_MIN_SKEW_FLOOR: Duration = Duration::from_secs(180);
        // Realistic access-token lifetime (~24 h).
        const TOKEN_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);

        let headroom = TOKEN_LIFETIME
            .checked_sub(ceiling)
            .expect("max tick interval must be shorter than the token lifetime");
        assert!(
            headroom >= THREE_MIN_SKEW_FLOOR,
            "shipped VALKEY_TOKEN_REAUTH_* constants: max tick interval ({ceiling:?}) \
             must leave ≥ 3 min ({THREE_MIN_SKEW_FLOOR:?}) of headroom before the ~24 h \
             token lifetime ({TOKEN_LIFETIME:?}); headroom was {headroom:?} \
             (SDD §15.3)"
        );
    }

    /// Pins the two **published** constants — the values that leave this module.
    ///
    /// `token_reauth_ceiling_leaves_three_min_headroom` above re-sums the two
    /// private parts; this pins the sums that are actually `pub`, and so is the
    /// test that fails if a future edit publishes something other than what the
    /// parts add up to.
    ///
    /// `VALKEY_TOKEN_ROTATION_FANOUT_CEILING` is load-bearing outside this crate
    /// in two places, and both are the reason the number is asserted rather than
    /// merely computed:
    ///
    /// - `main.rs` passes it whole to the Redis-scoped `ServiceTokenProvider` as
    ///   that instance's prolongation floor, so it is the threshold of the
    ///   per-rotation headroom WARN an operator will see in the logs.
    /// - The operator recommendation in `docs/service/docs/backing-store-azure.md`
    ///   § Token refresh for the sidecar's refresh lead (195 s) is *derived*
    ///   from it: the ceiling plus the 15 s default
    ///   `LIVE_EDIT_SERVICE_TOKEN_POLL_INTERVAL_SECS` pickup delay. Changing this constant silently invalidates that published
    ///   number, so a change here must be accompanied by a docs change.
    #[test]
    fn published_reauth_constants_match_the_shipped_parts() {
        assert_eq!(
            VALKEY_TOKEN_REAUTH_MAX_INTERVAL,
            Duration::from_secs(90),
            "the published access-token-file tick ceiling must stay 90 s (b.7td: base 60 s + \
             jitter bound 30 s)"
        );
        assert_eq!(
            VALKEY_TOKEN_REAUTH_MAX_INTERVAL,
            VALKEY_TOKEN_REAUTH_BASE_INTERVAL + VALKEY_TOKEN_REAUTH_JITTER_BOUND,
            "the published ceiling must remain exactly base + jitter bound; a `pub` constant that \
             has drifted from the parts fred actually schedules against would send both the WARN \
             threshold and the operator docs to a band no tick keeps"
        );
        assert_eq!(
            VALKEY_TOKEN_ROTATION_FANOUT_CEILING,
            Duration::from_secs(180),
            "the published rotation fan-out ceiling must stay 180 s: it is `main.rs`'s \
             prolongation floor for the Redis-scoped ServiceTokenProvider, and the operator \
             docs' sidecar-lead recommendation is this value plus the 15 s default token-file \
             poll interval"
        );
        assert_eq!(
            VALKEY_TOKEN_ROTATION_FANOUT_CEILING,
            VALKEY_TOKEN_REAUTH_MAX_INTERVAL * 2,
            "the fan-out ceiling must remain exactly TWO tick intervals — one for the first \
             client to observe a rotation (which opens the fan-out window) and one for the last \
             client to take its next tick inside it. Collapsing it to a single interval would \
             set the per-rotation headroom WARN below the latency the gate can actually deliver"
        );
    }

    // ── Shared passwordless-auth-mode fixtures ────────────────────────────────
    //
    // One copy of the inputs every passwordless-mode config test needs, used by
    // all four consumer modules: `auth_mode_dispatch`,
    // `pubsub_credential_refresh`, `token_file_tls_host_mapping_fallthrough`,
    // and `iam_tls_host_mapping_fallthrough`.
    //
    // They previously lived as near-identical per-module copies that had already
    // drifted in shape — `redis_token_provider` took a `&TempDir` in one module
    // and a `PathBuf` in another for the same job — which is the
    // copy-paste-with-one-line-differences pattern the test-review guide flags,
    // and the shape where a future edit lands in one copy and is forgotten in
    // the rest. The `PathBuf` shape won the reconciliation: it is the one that
    // lets a caller choose the filename (the parity test writes a real fixture
    // file and must name it), and `dir.path().join(...)` at the two call sites
    // that do not care is a wash.
    //
    // Ungated (no `#[cfg(feature = "valkey-tls")]`), for the same reason
    // `connect_fn_body()` is: two of the four consumer modules are themselves
    // ungated and must compile under the `--no-default-features` CI job. Nothing
    // here names a TLS symbol — each gated module keeps its own `TLS_URL`, which
    // is the only thing that actually differs between them.

    /// A managed-identity Object ID (SR-4.1 — the Object ID, not the client ID).
    /// The access-token-file arm fails closed with `ConnectError::Build` when
    /// `valkey_token_username` is `None`, so a value is required to reach that
    /// arm's provider install at all.
    const OBJECT_ID: &str = "11111111-2222-3333-4444-555555555555";

    /// The IAM-enabled Valkey ACL username (SDD §15.5). The `iam` arm's
    /// `ok_or_else` guard rejects `None`, so a value is required to reach the
    /// provider install.
    const IAM_USERNAME: &str = "live-edit-iam-user";

    /// AWS region for SigV4 signing; likewise required by the `iam` arm's guard.
    const IAM_REGION: &str = "us-east-1";

    /// Cluster host for SigV4 signing; likewise required by the `iam` arm's
    /// guard.
    const IAM_CLUSTER_HOST: &str = "valkey.example.com";

    /// Plaintext URL for the tests that do not exercise the TLS host-mapping
    /// tail. Legal here because the passwordless-mode-needs-TLS rule is enforced
    /// by `Config::from_env`, not by `build_fred_config`.
    const PLAINTEXT_URL: &str = "redis://127.0.0.1:6379";

    /// Poll cadence for a test `ServiceTokenProvider`; matches the
    /// `token_file_credentials.rs` harness.
    const TEST_POLL_INTERVAL: Duration = Duration::from_millis(20);

    /// The ≥ 3 min widened proactive re-read lead the access-token-file Valkey
    /// path uses (SR-4.1). Irrelevant to the tests that never drive `fetch()`;
    /// passed for fidelity with the production construction. It is a re-read
    /// cadence only — no accessor consults it (b.d6b, b.mq8) — so a fixture's
    /// lifetime need not clear it for the token to be served.
    const WIDE_EXPIRY_SKEW: chrono::Duration = chrono::Duration::seconds(180);

    /// Block until `provider`'s polling task has landed a servable token in its
    /// cache, or fail the test at `budget`.
    ///
    /// The test-writing guide's "wait for a background condition" shape (a
    /// bounded probe on real observable state) rather than a fixed
    /// `sleep(one interval + slack)`. The distinction is not cosmetic: a sleep
    /// that fell short of the poll on a loaded CI machine would leave the
    /// caller's `fetch()` returning the token-unavailable error, and the
    /// assertion that follows would die as a flake wearing the costume of the
    /// regression it exists to catch.
    ///
    /// `why` names what the *caller* loses when the setup fails, so a red run
    /// reads as "setup failure" rather than as the assertion under test. Shared
    /// by every caller that needs a warm provider, so the deadline arithmetic
    /// and the probe cadence live at one site.
    async fn await_token_cached(provider: &ServiceTokenProvider, budget: Duration, why: &str) {
        let deadline = tokio::time::Instant::now() + budget;
        while provider.cached_bearer_if_unexpired().is_none() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the ServiceTokenProvider polling task never loaded the fixture token within \
                 {budget:?} — {why}, so this is a setup failure, not a failure of the behaviour \
                 under test"
            );
            tokio::time::sleep(TEST_POLL_INTERVAL).await;
        }
    }

    /// A `Config` with `valkey_auth_mode` and `valkey_url` set, and nothing
    /// else touched. Delegates to the single `connect_timeout_test_config`
    /// struct literal so a new `Config` field upstream is ONE compile error, not
    /// five. The base helper leaves `valkey_url` empty, which would fail
    /// `FredConfig::from_url` with `BadUrl` before the auth-mode match is ever
    /// reached — hence the explicit `url`.
    fn auth_mode_test_config(mode: ValkeyAuthMode, url: &str) -> Config {
        let mut config = connect_timeout_test_config(1);
        config.valkey_url = url.to_string();
        config.valkey_auth_mode = mode;
        config
    }

    /// An access-token-file-mode `Config` on `url` carrying the Object-ID
    /// username — the minimum to reach that arm's provider install.
    fn token_file_test_config(url: &str) -> Config {
        let mut config = auth_mode_test_config(ValkeyAuthMode::AccessTokenFile, url);
        config.valkey_token_username = Some(OBJECT_ID.to_string());
        config
    }

    /// An iam-mode `Config` on `url` carrying every SigV4 signing input — the
    /// minimum to reach that arm's provider install.
    fn iam_test_config(url: &str) -> Config {
        let mut config = auth_mode_test_config(ValkeyAuthMode::Iam, url);
        config.valkey_iam_username = Some(IAM_USERNAME.to_string());
        config.valkey_iam_region = Some(IAM_REGION.to_string());
        config.valkey_iam_cluster_host = Some(IAM_CLUSTER_HOST.to_string());
        config
    }

    /// A Redis-scoped `ServiceTokenProvider` over `path`, plus the cancellation
    /// token the caller uses to stop its polling task.
    ///
    /// The file at `path` need not exist: the access-token-file arm only WRAPS
    /// the provider in a `TokenFileCredentialProvider`: callers that assert on
    /// the wrap alone never drive `fetch()` /
    /// `cached_bearer_if_unexpired()`. Callers that
    /// DO (the credential-parity test) write the fixture first and wait for the
    /// polling task to pick it up.
    fn redis_token_provider(
        path: std::path::PathBuf,
        label: &str,
    ) -> (Arc<ServiceTokenProvider>, CancellationToken) {
        let cancel = CancellationToken::new();
        let provider = Arc::new(ServiceTokenProvider::new_with_refresh_lead(
            path,
            TEST_POLL_INTERVAL,
            WIDE_EXPIRY_SKEW,
            Arc::new(crate::observability::metrics::metrics_scrape_only(label)),
            cancel.clone(),
        ));
        (provider, cancel)
    }

    /// A stub AWS credential source threaded into the `iam` arm so it installs
    /// the provider. The arm only WRAPS this in an `IamCredentialProvider`; no
    /// `fetch()` / SigV4 presign is driven by any test here, so the fixed static
    /// credentials never need to be valid.
    fn stub_iam_credentials() -> SharedCredentialsProvider {
        SharedCredentialsProvider::new(Credentials::new(
            "AKIAEXAMPLEACCESSKEY",
            "example-secret-access-key",
            Some("example-session-token".to_string()),
            None,
            "client-tests-stub-iam",
        ))
    }

    // ── t2.tr5.vc.aa auth-mode dispatch (apply_fred_config_overrides) ─────────
    //
    // Epic 2 Task 2 extended `apply_fred_config_overrides` with a
    // `match config.valkey_auth_mode`, and Epics 3/4 filled the passwordless
    // arms: `Static` installs no `credential_provider` (byte-identical to the
    // pre-auth-mode FredConfig), while `AccessTokenFile` / `Iam` install a real
    // `credential_provider` when their credential source is threaded in — and
    // fail CLOSED with `ConnectError::Build` when it (or a required per-mode
    // input) is absent, rather than silently building a provider-less
    // (unauthenticated) connection. These tests pin that dispatch table so a
    // regression that silently installs (or drops) a provider — or that stops
    // failing closed on a missing source — is caught. The provider-present
    // success path additionally falls through to the shared TLS host-mapping
    // tail; that fall-through is pinned by the `#[cfg(feature = "valkey-tls")]`
    // `token_file_tls_host_mapping_fallthrough` / `iam_tls_host_mapping_fallthrough`
    // modules below.
    mod auth_mode_dispatch {
        use super::*;

        #[test]
        fn static_mode_builds_ok_without_credential_provider() {
            let config = auth_mode_test_config(ValkeyAuthMode::Static, PLAINTEXT_URL);

            let configs = build_fred_config(&config, None, None).expect(
                "Static auth mode must build a FredConfig — the default \
                 passwordless / URL-embedded-password path",
            );

            // The `credential-provider` fred feature is ALWAYS-ON in this crate's
            // base feature list (Cargo.toml), not gated behind `valkey-tls`, so
            // the field is present unconditionally and this assertion needs no
            // `#[cfg]`. Static must install NO provider — byte-identical to the
            // pre-auth-mode FredConfig; a future arm that silently wires one here
            // fails this test.
            assert!(
                configs.shared.credential_provider.is_none(),
                "Static auth mode must not install a credential_provider"
            );
            // …on BOTH derived configs (b.j64). With no provider to wrap, the
            // pub/sub derivation is a straight clone and must not conjure one.
            assert!(
                configs.pubsub.credential_provider.is_none(),
                "Static auth mode must not install a credential_provider on the \
                 pub/sub config either — with nothing to wrap, the derivation is a \
                 straight clone"
            );
            assert!(
                !configs.pubsub_refresh_disabled,
                "there is no periodic re-AUTH to disable under Static — the flag \
                 must stay false so the startup log line does not claim otherwise"
            );
            // …and on the readers' config (b.19i): nothing to wrap, and no
            // provider handed to the readers, so no reader schedules a
            // re-AUTH and the readers' startup line stays silent.
            assert!(
                configs.reader.credential_provider.is_none(),
                "Static auth mode must not install a credential_provider on the readers' \
                 config — with nothing to wrap, the derivation is a straight clone"
            );
            assert!(
                configs.reader_reauth.is_none(),
                "Static auth mode hands the readers no re-AUTH provider: no reader schedules \
                 a re-AUTH tick"
            );
        }

        #[test]
        fn access_token_file_mode_without_provider_fails_closed() {
            // Epic 3 filled the access-token-file arm: it now installs a real
            // `credential_provider` when a Redis-scoped `ServiceTokenProvider`
            // is threaded in (the startup-wiring Subtask supplies it). Here we
            // pass `None` — both the interim call-site shape and the
            // misconfiguration case — and assert the arm fails CLOSED rather
            // than silently building a provider-less (unauthenticated)
            // FredConfig. (The provider-present success path, which additionally
            // wraps the token provider and preserves the TLS host-mapping tail,
            // is covered where a `ServiceTokenProvider` can be constructed.)
            let config = auth_mode_test_config(ValkeyAuthMode::AccessTokenFile, PLAINTEXT_URL);

            let err = build_fred_config(&config, None, None).expect_err(
                "access-token-file auth mode with no ServiceTokenProvider must fail closed, \
                 not build an unauthenticated FredConfig",
            );

            let display = err.to_string().to_ascii_lowercase();
            assert!(
                matches!(err, ConnectError::Build(_)),
                "a missing provider must surface as ConnectError::Build; got: {err:?}"
            );
            assert!(
                display.contains("accesstokenfile"),
                "the error must name the offending mode; got: {display:?}"
            );
        }

        #[test]
        fn iam_mode_without_credential_source_fails_closed() {
            // Epic 4 filled the iam arm: it now installs a real
            // `credential_provider` when an AWS credential source is threaded in
            // (the startup-wiring Subtask resolves it via the IRSA/STS default
            // chain). Here we pass `None` — the misconfiguration case — and
            // assert the arm fails CLOSED with `ConnectError::Build` rather than
            // silently building a provider-less (unauthenticated) FredConfig.
            // Mirrors `access_token_file_mode_without_provider_fails_closed`. (The
            // provider-present success path, which additionally wraps the
            // credential source and preserves the TLS host-mapping tail, is the
            // Test Writer's `valkey-tls` fall-through assertion.)
            let config = auth_mode_test_config(ValkeyAuthMode::Iam, PLAINTEXT_URL);

            let err = build_fred_config(&config, None, None).expect_err(
                "iam auth mode with no AWS credential source must fail closed, \
                 not build an unauthenticated FredConfig",
            );

            let display = err.to_string().to_ascii_lowercase();
            assert!(
                matches!(err, ConnectError::Build(_)),
                "a missing credential source must surface as ConnectError::Build; got: {err:?}"
            );
            assert!(
                display.contains("iam"),
                "the error must name the offending mode; got: {display:?}"
            );
        }
    }

    // ── b.j64 pub/sub re-AUTH cadence divergence (derive_pubsub_config) ───────
    //
    // `build_fred_config` returns the shared config and two derived from it:
    // the pub/sub one (here) and the readers' one (b.19i; its tests follow this
    // module's pub/sub ones). The pub/sub one differs from
    // the shared one in exactly one respect — its `credential_provider` is
    // wrapped in `NoRefreshCredentialProvider`, so fred never spawns the
    // per-client credential-refresh task for the pub/sub `SubscriberClient`.
    // That connection is permanently in RESP2 subscribe context, where Valkey
    // refuses `AUTH` outright and fred can only log a WARN once per tick
    // forever; the credential there renews on the reconnect path instead
    // (SDD §15.3).
    //
    // What these tests pin, and why each one matters:
    //
    // - The wrap happens under BOTH passwordless modes (it is a property of the
    //   RESP2 subscribe context, not of the token source).
    // - The SHARED provider's cadence is untouched — the guard against a change
    //   that accidentally disables the refresh fleet-wide, which would break the
    //   IAM 12-hour forced-disconnect prolongation (§15.5) and the access-token
    //   re-AUTH (§15.4) on the connections where they are load-bearing.
    // - Both configs present the SAME credential. This is the load-bearing
    //   security assertion: the wrapper narrows the cadence, never the
    //   credential, which is why the divergence does not weaken the "one
    //   credential-provider policy across all six connections" guarantee.
    // - Every non-auth field is at parity, so nothing else rode along on the
    //   clone.
    // - The RESP2 gate is real, and the mechanism is the RESP *version*, not the
    //   command fred picks: Valkey applies the subscribe-context command
    //   allowlist only to RESP2 connections, so a connection that handshook
    //   `HELLO 3` is never placed in subscribe-restricted state at all and its
    //   periodic refresh reaches the server. In RESP2 subscribed state `HELLO`
    //   is refused with the same error text as `AUTH` (measured on Valkey
    //   9.0.3), so fred's RESP3-only switch from `AUTH` to `HELLO` is not what
    //   makes the refresh land — the RESP3 handshake is. Under RESP3 the refresh
    //   therefore works and must NOT be suppressed; gating on the version means
    //   a future protocol flip restores it automatically rather than silently
    //   masking it.
    //
    // Ungated (no `#[cfg(feature = "valkey-tls")]`) so the guard also runs under
    // the `--no-default-features` CI job — a plaintext URL is enough here, since
    // the passwordless-mode-needs-TLS rule is enforced by `Config::from_env`,
    // not by `build_fred_config`. The single `.tls` parity assertion is gated
    // on its own because fred `#[cfg]`s that field away.
    mod pubsub_credential_refresh {
        use tempfile::tempdir;

        use super::*;

        /// The shipped tick band the named passwordless provider draws from —
        /// `[VALKEY_*_REAUTH_BASE_INTERVAL, + JITTER_BOUND]`.
        ///
        /// **The two modes no longer share one band (b.7td)** and the
        /// divergence is deliberate, so this takes the mode rather than
        /// asserting a shared precondition. The access-token-file provider ticks
        /// `[60 s, 90 s]`: its rotation gate means most ticks send nothing, so a
        /// short cadence is cheap and improves rotation pickup. The IAM provider
        /// stays at `[300 s, 360 s]`: it mints a freshly signed SigV4 token on
        /// every `fetch()`, so every one of its ticks is a real `AUTH` and the
        /// §15.5 12-hour forced-disconnect prolongation depends on it.
        ///
        /// Read from the production constants, not re-declared literals, so a
        /// constant bump moves the assertion with it.
        fn shipped_reauth_band(mode: &str) -> (Duration, Duration) {
            match mode {
                "accessTokenFile" => (
                    VALKEY_TOKEN_REAUTH_BASE_INTERVAL,
                    VALKEY_TOKEN_REAUTH_BASE_INTERVAL + VALKEY_TOKEN_REAUTH_JITTER_BOUND,
                ),
                "iam" => (
                    VALKEY_IAM_REAUTH_BASE_INTERVAL,
                    VALKEY_IAM_REAUTH_BASE_INTERVAL + VALKEY_IAM_REAUTH_JITTER_BOUND,
                ),
                other => panic!(
                    "shipped_reauth_band: unknown passwordless mode {other:?} — add its band \
                     here rather than borrowing a sibling's"
                ),
            }
        }

        /// The access-token-file / iam configs these tests drive, both on the
        /// shared plaintext URL: the cadence divergence is a property of the
        /// RESP2 subscribe context, not of the transport, so no TLS is needed
        /// (and gating this module on `valkey-tls` would drop it from the
        /// `--no-default-features` CI job).
        fn token_file_config() -> Config {
            token_file_test_config(PLAINTEXT_URL)
        }

        fn iam_config() -> Config {
            iam_test_config(PLAINTEXT_URL)
        }

        /// Write the sidecar `{"access_token","expires_on"}` JSON the
        /// `ServiceTokenProvider` reads.
        async fn write_token_file(path: &std::path::Path, token: &str, expires_in_secs: i64) {
            let expires_on = chrono::Utc::now() + chrono::Duration::seconds(expires_in_secs);
            let body = format!(
                r#"{{"access_token":"{}","expires_on":"{}"}}"#,
                token,
                expires_on.to_rfc3339()
            );
            tokio::fs::write(path, body).await.expect("write fixture");
        }

        /// Anti-hang ceiling for [`await_token_loaded`]. Deliberately generous:
        /// a bounded wait returns the instant its condition holds, so a large
        /// ceiling costs nothing on the happy path and only changes behaviour
        /// when the polling task is genuinely stuck.
        const TOKEN_LOAD_DEADLINE: Duration = Duration::from_secs(5);

        /// Block until the backing `ServiceTokenProvider`'s polling task has
        /// loaded the on-disk fixture. Thin binding of the shared
        /// [`await_token_cached`] probe to this module's budget and stake.
        async fn await_token_loaded(provider: &ServiceTokenProvider) {
            await_token_cached(
                provider,
                TOKEN_LOAD_DEADLINE,
                "the credential-parity assertion needs a real credential to compare",
            )
            .await;
        }

        /// Shared assertion body: the shared config keeps the shipped re-AUTH
        /// cadence; the pub/sub config schedules none.
        fn assert_cadence_divergence(configs: &FredConfigs, mode: &str) {
            let (floor, ceiling) = shipped_reauth_band(mode);

            let shared = configs
                .shared
                .credential_provider
                .as_ref()
                .unwrap_or_else(|| panic!("{mode} must install a credential provider"));
            let interval = shared.refresh_interval().unwrap_or_else(|| {
                panic!(
                    "{mode}: the SHARED provider must still schedule a periodic re-AUTH — \
                     disabling it fleet-wide would break the IAM 12-hour \
                     forced-disconnect prolongation (§15.5) and the access-token \
                     re-AUTH (§15.4) on the four pools where fred runs it, and on the \
                     stream readers, which tick from this same provider (b.19i)"
                )
            });
            assert!(
                interval >= floor && interval <= ceiling,
                "{mode}: the shared provider's cadence must stay in the shipped \
                 [{floor:?}, {ceiling:?}] band, got {interval:?}"
            );

            let pubsub = configs
                .pubsub
                .credential_provider
                .as_ref()
                .unwrap_or_else(|| panic!("{mode} must install a pub/sub credential provider too"));
            assert!(
                pubsub.refresh_interval().is_none(),
                "{mode}: the pub/sub provider must schedule NO periodic re-AUTH — fred \
                 only spawns the refresh task when this is Some, and on the RESP2 \
                 pub/sub connection that task's AUTH is refused by Valkey, producing \
                 one WARN per interval forever (b.j64)"
            );
            assert!(
                configs.pubsub_refresh_disabled,
                "{mode}: the flag that drives the startup log line must agree with the \
                 wrapping actually applied"
            );
        }

        #[tokio::test]
        async fn access_token_file_disables_periodic_reauth_on_the_pubsub_config_only() {
            let dir = tempdir().expect("tempdir");
            let (provider, cancel) =
                redis_token_provider(dir.path().join("token.json"), "pubsub-refresh-token-file");

            let configs = build_fred_config(&token_file_config(), Some(provider), None)
                .expect("access-token-file mode with a provider + username must build");
            assert_cadence_divergence(&configs, "accessTokenFile");

            cancel.cancel();
        }

        #[test]
        fn iam_disables_periodic_reauth_on_the_pubsub_config_only() {
            let configs = build_fred_config(&iam_config(), None, Some(stub_iam_credentials()))
                .expect("iam mode with a credential source + signing inputs must build");
            assert_cadence_divergence(&configs, "iam");
        }

        #[tokio::test]
        async fn both_configs_present_the_identical_credential() {
            // THE load-bearing security assertion. The wrapper delegates
            // `fetch()` to the SAME inner `Arc`, so the pub/sub connection —
            // and every stream reader (b.19i) — authenticates with
            // byte-identical credentials to the four pools. A wrapper that
            // cached, defaulted, or blanked the credential would be a
            // partial-auth divergence — the exact hazard §15.3's "one
            // credential-provider policy across all six connections" rule
            // exists to prevent. `derive_pubsub_config` and
            // `derive_reader_config` are *sanctioned* derived writes inside the
            // `build_fred_config` seam precisely because they narrow the
            // re-AUTH cadence and preserve the credential; break that second
            // half and the sanction no longer holds — while every other test in
            // this module still passes.
            //
            // Asked as a handshake (`fetch(Some(server))`), which is what each
            // connection presents when it authenticates, and which carries the
            // credential whatever the rotation gate says: a periodic tick on
            // the provider's first sighting of the token carries no password
            // (b.19i), so asked as a tick both sides would read `None` and the
            // parity would hold vacuously.
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("token.json");
            write_token_file(&path, "token-parity-canary", 3600).await;
            let (provider, cancel) = redis_token_provider(path, "pubsub-refresh-parity");
            await_token_loaded(&provider).await;

            let configs = build_fred_config(&token_file_config(), Some(provider), None)
                .expect("access-token-file mode must build");
            let server = Server::new("127.0.0.1", 6379);
            let handshake = |config: &FredConfig, which: &'static str| {
                let provider = config
                    .credential_provider
                    .clone()
                    .unwrap_or_else(|| panic!("{which} provider"));
                let server = server.clone();
                async move {
                    provider
                        .fetch(Some(&server))
                        .await
                        .unwrap_or_else(|e| panic!("{which} handshake fetch: {e:?}"))
                }
            };

            let shared = handshake(&configs.shared, "shared").await;
            let pubsub = handshake(&configs.pubsub, "pub/sub").await;
            let reader = handshake(&configs.reader, "reader").await;

            assert_eq!(
                shared,
                (
                    Some(OBJECT_ID.to_string()),
                    Some("token-parity-canary".to_string())
                ),
                "precondition: the shared provider must return the fixture credential"
            );
            assert_eq!(
                pubsub, shared,
                "the pub/sub provider MUST return the identical (username, password) — \
                 the wrapper narrows the re-AUTH cadence, never the credential"
            );
            assert_eq!(
                reader, shared,
                "the readers' provider MUST return the identical (username, password) — \
                 the wrapper narrows the re-AUTH cadence, never the credential (b.19i)"
            );

            cancel.cancel();
        }

        // ── b.19i reader re-AUTH cadence divergence (derive_reader_config) ──
        //
        // The readers' config differs from the shared one in exactly one
        // respect, like the pub/sub one: its provider is wrapped so fred
        // schedules no refresh on any reader client. Unlike pub/sub this holds
        // under ANY RESP version — the reason is the reader's command
        // discipline (it sends its own re-AUTH between reads), not a server
        // refusal. The seam also hands back the UNWRAPPED provider, the very
        // `Arc` the shared config carries, which each reader schedules and
        // gates its own re-AUTH from.

        /// The reader-side half of the divergence for `mode`: the readers'
        /// config schedules no fred refresh, its wrapper delegates to the very
        /// provider the shared config carries, and the provider handed to the
        /// readers is that same `Arc`, keeping the shipped cadence.
        fn assert_reader_divergence(configs: &FredConfigs, mode: &str) {
            let (floor, ceiling) = shipped_reauth_band(mode);
            let shared = configs
                .shared
                .credential_provider
                .as_ref()
                .unwrap_or_else(|| panic!("{mode} must install a credential provider"));
            let reader = configs
                .reader
                .credential_provider
                .as_ref()
                .unwrap_or_else(|| panic!("{mode} must install a reader credential provider"));
            assert!(
                reader.refresh_interval().is_none(),
                "{mode}: the readers' provider must schedule NO fred refresh — fred's refresh \
                 task would be a second writer on a connection parked in XREAD BLOCK, which \
                 races its interrupt and can queue behind a 30 s BLOCK (b.19i)"
            );
            let reauth = configs.reader_reauth.as_ref().unwrap_or_else(|| {
                panic!(
                    "{mode}: the seam must hand the readers a provider, or no reader re-AUTHs \
                     its connection at all"
                )
            });
            assert!(
                Arc::ptr_eq(reauth, shared),
                "{mode}: the provider the readers re-AUTH from must be the SAME Arc the pools \
                 carry — a second provider instance would be a second credential source"
            );
            let interval = reauth.refresh_interval().unwrap_or_else(|| {
                panic!("{mode}: the readers' re-AUTH provider must schedule ticks")
            });
            assert!(
                interval >= floor && interval <= ceiling,
                "{mode}: the readers tick on the shipped [{floor:?}, {ceiling:?}] cadence, got \
                 {interval:?}"
            );
        }

        #[tokio::test]
        async fn access_token_file_disables_fred_refresh_on_the_reader_config_only() {
            let dir = tempdir().expect("tempdir");
            let (provider, cancel) =
                redis_token_provider(dir.path().join("token.json"), "reader-refresh-token-file");

            let configs = build_fred_config(&token_file_config(), Some(provider), None)
                .expect("access-token-file mode with a provider + username must build");
            assert_reader_divergence(&configs, "accessTokenFile");
            // The shared and pub/sub sides are unchanged by the readers' wrap.
            assert_cadence_divergence(&configs, "accessTokenFile");

            cancel.cancel();
        }

        #[test]
        fn iam_disables_fred_refresh_on_the_reader_config_only() {
            let configs = build_fred_config(&iam_config(), None, Some(stub_iam_credentials()))
                .expect("iam mode with a credential source + signing inputs must build");
            assert_reader_divergence(&configs, "iam");
            assert_cadence_divergence(&configs, "iam");
        }

        #[test]
        fn resp3_still_wraps_the_reader_config() {
            // The pub/sub wrap is RESP2-only; the readers' is not. Under RESP3
            // fred's refresh would reach a reader's server just as well — and
            // still race the reader's own XREAD — so the readers' config must be
            // wrapped whatever the protocol.
            let configs = build_fred_config(&iam_config(), None, Some(stub_iam_credentials()))
                .expect("iam mode must build");
            let mut resp3 = configs.shared.clone();
            resp3.version = RespVersion::RESP3;

            let (reader, reauth) = derive_reader_config(&resp3);
            assert!(
                reader
                    .credential_provider
                    .as_ref()
                    .expect("the readers' config keeps a provider under RESP3")
                    .refresh_interval()
                    .is_none(),
                "under RESP3 the readers' provider must still schedule no fred refresh"
            );
            assert!(
                Arc::ptr_eq(
                    reauth.as_ref().expect("a provider for the readers"),
                    resp3.credential_provider.as_ref().expect("shared provider"),
                ),
                "under RESP3 the readers still re-AUTH from the shared provider itself"
            );
            assert_eq!(
                reader.version,
                RespVersion::RESP3,
                "the derivation touches the provider alone"
            );
            let (pubsub, disabled) = derive_pubsub_config(&resp3);
            assert!(
                !disabled && pubsub.credential_provider.is_some(),
                "control: under RESP3 the pub/sub derivation wraps nothing"
            );
        }

        #[tokio::test]
        async fn reader_config_fetch_reaches_the_shared_provider_on_every_call() {
            // Delegation, not just equal values: a counting provider installed
            // on a shared config is reached by the readers' wrapper on every
            // `fetch`, with the `server` argument as given. A wrapper that
            // cached would hand a reconnecting reader its boot-time credential.
            struct Counting {
                fetches: std::sync::atomic::AtomicUsize,
                handshakes: std::sync::atomic::AtomicUsize,
            }
            #[async_trait::async_trait]
            impl CredentialProvider for Counting {
                async fn fetch(
                    &self,
                    server: Option<&Server>,
                ) -> Result<(Option<String>, Option<String>), fred::error::Error> {
                    self.fetches
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if server.is_some() {
                        self.handshakes
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    Ok((Some("u".to_owned()), Some("p".to_owned())))
                }
                fn refresh_interval(&self) -> Option<Duration> {
                    Some(Duration::from_secs(60))
                }
            }
            impl std::fmt::Debug for Counting {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("Counting")
                }
            }

            let inner = Arc::new(Counting {
                fetches: std::sync::atomic::AtomicUsize::new(0),
                handshakes: std::sync::atomic::AtomicUsize::new(0),
            });
            let mut shared =
                FredConfig::from_url(PLAINTEXT_URL).expect("static plaintext URL must parse");
            shared.credential_provider = Some(inner.clone());

            let (reader, reauth) = derive_reader_config(&shared);
            let wrapped = reader.credential_provider.expect("the readers' provider");
            let server = Server::new("127.0.0.1", 6379);
            let first = wrapped.fetch(Some(&server)).await.expect("first fetch");
            let second = wrapped.fetch(None).await.expect("second fetch");
            assert_eq!(first, (Some("u".to_owned()), Some("p".to_owned())));
            assert_eq!(second, first);
            assert_eq!(
                (
                    inner.fetches.load(std::sync::atomic::Ordering::SeqCst),
                    inner.handshakes.load(std::sync::atomic::Ordering::SeqCst)
                ),
                (2, 1),
                "each fetch must reach the shared provider, with its server argument"
            );
            let reauth = reauth.expect("a provider for the readers");
            let inner_dyn: Arc<dyn CredentialProvider> = inner;
            assert!(
                Arc::ptr_eq(&reauth, &inner_dyn),
                "the readers re-AUTH from the shared provider itself, unwrapped"
            );
        }

        #[test]
        fn reader_config_matches_shared_on_every_non_auth_field() {
            // As for the pub/sub config: nothing but the provider may ride
            // along. `blocking` in particular stays at the default here —
            // `ValkeyClient::reader_builder` sets `Blocking::Interrupt` on the
            // value `connect` moves into it, the one per-pool divergence
            // pinned by the connect guards.
            let configs = build_fred_config(&iam_config(), None, Some(stub_iam_credentials()))
                .expect("iam mode must build");

            assert_eq!(configs.reader.version, configs.shared.version);
            assert_eq!(configs.reader.server, configs.shared.server);
            assert_eq!(configs.reader.blocking, configs.shared.blocking);
            assert_eq!(configs.reader.fail_fast, configs.shared.fail_fast);
            assert_eq!(configs.reader.database, configs.shared.database);
            // `assert!` on the credential-bearing fields so a failure cannot
            // print them (SR-6.2).
            assert!(
                configs.reader.username == configs.shared.username,
                "URL-embedded AUTH username must not diverge"
            );
            assert!(
                configs.reader.password == configs.shared.password,
                "URL-embedded AUTH password must not diverge"
            );
            #[cfg(feature = "valkey-tls")]
            assert_eq!(
                configs.reader.tls, configs.shared.tls,
                "TLS policy must not diverge between the readers and the pools"
            );
        }

        #[test]
        fn pubsub_config_matches_shared_on_every_non_auth_field() {
            // Nothing but the credential provider may ride along on the clone.
            // A future edit that diverged, say, `version` or `blocking` here
            // would be a post-override divergence on an auth-bearing or
            // scheduling field with no argument made for it.
            let configs = build_fred_config(&iam_config(), None, Some(stub_iam_credentials()))
                .expect("iam mode must build");

            assert_eq!(
                configs.pubsub.version, configs.shared.version,
                "`version` is auth-bearing (it decides whether the connection is \
                 subscribe-restricted at all — which is what makes or breaks the \
                 periodic re-AUTH — as well as the reply-type semantics of every \
                 command) — it must not diverge"
            );
            assert_eq!(
                configs.pubsub.server, configs.shared.server,
                "both configs must target the same server/topology"
            );
            assert_eq!(
                configs.pubsub.blocking, configs.shared.blocking,
                "`Blocking::Interrupt` belongs to the subscriber_pool alone (b.9uq \
                 round 4); the pub/sub client is multiplexed and must keep the default"
            );
            assert_eq!(
                configs.pubsub.fail_fast, configs.shared.fail_fast,
                "connect-failure posture must not diverge"
            );
            // `assert!` rather than `assert_eq!` on the two credential-bearing
            // fields: a failure message must not print them (SR-6.2,
            // bearer-bytes-never-logged — test output is logs too).
            assert!(
                configs.pubsub.username == configs.shared.username,
                "URL-embedded AUTH username must not diverge between the two configs"
            );
            assert!(
                configs.pubsub.password == configs.shared.password,
                "URL-embedded AUTH password must not diverge between the two configs"
            );
            #[cfg(feature = "valkey-tls")]
            assert_eq!(
                configs.pubsub.tls, configs.shared.tls,
                "TLS policy must not diverge — that is the partial-TLS hazard \
                 §15.3's mutation-seam rule exists to prevent: \
                 apply_fred_config_overrides writes `.tls` once, before every \
                 derivation, and derive_pubsub_config's sanctioned derived write \
                 touches credential_provider alone"
            );
        }

        #[test]
        fn fred_configs_debug_redacts_the_url_embedded_credentials() {
            // `FredConfigs`' `Debug` is hand-written and redacting, and nothing
            // else pins that. Replace it with `#[derive(Debug)]` and the crate
            // still compiles, every other test here still passes — and any
            // `{:?}` of the struct prints the URL-embedded AUTH username and
            // password in the clear, because fred's `Config` derives `Debug`
            // over both fields. The trait is required only so `expect_err`
            // compiles in the fail-closed dispatch tests, so there is no other
            // pressure keeping the redaction honest.
            //
            // Static mode: the leak hazard is the URL-embedded pair fred parses
            // on EVERY mode, so the cheapest arm that yields a built
            // `FredConfigs` is the right one to render.
            const CANARY_USERNAME: &str = "debug-redaction-canary-user";
            const CANARY_PASSWORD: &str = "debug-redaction-canary-secret";
            const CANARY_URL: &str =
                "redis://debug-redaction-canary-user:debug-redaction-canary-secret@127.0.0.1:6379";

            let configs = build_fred_config(
                &auth_mode_test_config(ValkeyAuthMode::Static, CANARY_URL),
                None,
                None,
            )
            .expect("a credential-bearing plaintext URL must build");

            // Precondition — without it the two absence assertions below would
            // pass vacuously against a URL fred never parsed credentials out of.
            // `assert!` rather than `assert_eq!` so a failure cannot print the
            // pair (SR-6.2; test output is logs too).
            assert!(
                configs.shared.username.as_deref() == Some(CANARY_USERNAME)
                    && configs.shared.password.as_deref() == Some(CANARY_PASSWORD),
                "precondition: fred must parse the URL-embedded credential pair onto \
                 the config, otherwise this test asserts the absence of something \
                 that was never there"
            );

            let rendered = format!("{configs:?}");
            // The rendered value is deliberately NOT interpolated into these
            // messages: printing it on failure would be the very leak the
            // assertion exists to catch.
            assert!(
                !rendered.contains(CANARY_USERNAME),
                "FredConfigs' Debug must not render the URL-embedded AUTH username \
                 (SR-6.2, bearer-bytes-never-logged, extended to Valkey auth) — a \
                 #[derive(Debug)] here would print it"
            );
            assert!(
                !rendered.contains(CANARY_PASSWORD),
                "FredConfigs' Debug must not render the URL-embedded AUTH password \
                 (SR-6.2, bearer-bytes-never-logged, extended to Valkey auth) — a \
                 #[derive(Debug)] here would print it"
            );
            // …while still rendering the non-secret cadence fact, so a Debug
            // that redacted by printing nothing at all does not pass as
            // compliant.
            assert!(
                rendered.contains("pubsub_refresh_disabled")
                    && rendered.contains("reader_refresh_disabled"),
                "the redacting Debug must still surface the non-secret cadence \
                 fields — that is what makes it useful in a log line; got: {rendered}"
            );
        }

        #[test]
        fn resp3_leaves_the_periodic_reauth_schedule_intact() {
            // The gate is on the protocol, not on the connection's role — and
            // the mechanism is the RESP version rather than the command fred
            // picks. Valkey applies the subscribe-context command allowlist only
            // to RESP2 connections: a connection that handshook `HELLO 3` is
            // never placed in subscribe-restricted state, so its periodic
            // refresh reaches the server. (Measured on Valkey 9.0.3: a RESP2
            // connection in subscribed state refuses `HELLO` with the same error
            // text as `AUTH`, so fred's RESP3-only `AUTH` → `HELLO` switch at
            // `fred-10.1.0/src/modules/inner.rs:412-424` is not what makes the
            // refresh land.) Under RESP3 the refresh therefore works, and
            // suppressing it would be masking a load-bearing mechanism. Pinning
            // this means a future flip of the pub/sub client to RESP3 restores
            // the periodic refresh automatically.
            let configs = build_fred_config(&iam_config(), None, Some(stub_iam_credentials()))
                .expect("iam mode must build");
            assert_eq!(
                configs.shared.version,
                RespVersion::RESP2,
                "precondition: fred's URL path yields RESP2, so the production gate is \
                 live. If a fred bump or a config change flips this default, the \
                 suppression silently stops applying and this test says so"
            );

            let mut resp3 = configs.shared.clone();
            resp3.version = RespVersion::RESP3;
            let (pubsub, disabled) = derive_pubsub_config(&resp3);

            assert!(
                !disabled,
                "under RESP3 nothing may be suppressed — the flag must be false so the \
                 startup log line does not claim a suppression that did not happen"
            );
            let interval = pubsub
                .credential_provider
                .as_ref()
                .expect("the provider must still be installed under RESP3")
                .refresh_interval();
            // This test drives `iam_config()`, so it must assert the IAM band.
            let (floor, ceiling) = shipped_reauth_band("iam");
            let interval =
                interval.expect("under RESP3 the pub/sub config must carry the UNWRAPPED provider");
            assert!(
                interval >= floor && interval <= ceiling,
                "the RESP3 pub/sub provider must keep the shipped [{floor:?}, {ceiling:?}] \
                 cadence, got {interval:?}"
            );
        }

        // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
        #[test]
        fn connect_keeps_the_pubsub_builder_on_the_reconnect_reauth_path() {
            // WHY THIS IS PART OF b.j64 AND NOT SOMEONE ELSE'S TEST. Suppressing
            // the periodic re-AUTH is only safe because the pub/sub connection
            // has a SECOND credential-renewal path: `reconnect_on_auth_error =
            // true` plus the handshake `AUTH`, which calls `fetch()` afresh.
            // That flag reaches the connection through exactly one call —
            // `apply_connect_timeout(&mut subscriber_builder, config, ..)` — and the
            // sibling `apply_connect_timeout_enables_reconnect_on_auth_error`
            // pins what that helper DOES, not that the pub/sub builder is one of
            // its callers.
            //
            // Before this fix, dropping that call cost the pub/sub connection a
            // renewal path it was not really using. After it, that call IS the
            // renewal path: lose it and the pub/sub connection presents its
            // boot-time credential forever, silently, until the next process
            // restart — no periodic refresh to fall back on, no warning, and
            // every existing test in this module still green.
            //
            // Source inspection (the established `connect_fn_body()` technique,
            // shared with the sibling ordering and single-site guards) because
            // `connect` builds and consumes the `Builder` internally and returns
            // only a connected `ValkeyClient`: which builders it wired is not
            // observable from any runtime value without a live server.
            let body = connect_fn_body();

            let pubsub_build_idx = body
                .find("Builder::from_config(pubsub_fred_config)")
                .expect(
                    "connect must build the pub/sub subscriber client from the seam-derived \
                     pubsub_fred_config — an anchor-not-found panic here is an intentional \
                     loud maintenance signal, not a flake",
                );
            let timeout_idx = body[pubsub_build_idx..]
                .find("apply_connect_timeout(\n            &mut subscriber_builder,\n            config,")
                .map(|i| pubsub_build_idx + i)
                .expect(
                    "connect must call apply_connect_timeout on the pub/sub subscriber \
                     builder — it is the sole site that sets reconnect_on_auth_error, which \
                     is the ONLY credential-renewal path left on that connection now that \
                     the periodic re-AUTH is suppressed (b.j64)",
                );
            let client_idx = body[timeout_idx..]
                .find("build_subscriber_client()")
                .map(|i| timeout_idx + i)
                .expect(
                    "connect must call apply_connect_timeout BEFORE build_subscriber_client() \
                     — a Builder consumed first carries fred's default \
                     reconnect_on_auth_error = false, so the connection would hold its \
                     boot-time credential until the process restarts",
                );
            assert!(
                pubsub_build_idx < timeout_idx && timeout_idx < client_idx,
                "the pub/sub builder must be constructed, then given the connect-timeout \
                 policy (which enables reconnect_on_auth_error), then consumed"
            );
        }
    }

    // ── t2.tr5.t4.oq TLS-feature wiring (valkey-tls → fred/enable-rustls-ring) ──
    //
    // Task t2.tr5.t4.oq is a Cargo.toml-only change: it routes fred's
    // `enable-rustls-ring` TLS feature through live-edit's defaulted
    // `valkey-tls` alias. No production Rust source changed, so without these
    // tests a future dependency edit that drops the feature regresses
    // *silently* straight back to the SR-1.1 failure mode — fred's
    // `Config::from_url_centralized` has its `tls:` initializer
    // `#[cfg]`-compiled-out, `.tls` stays `None`, and a `rediss://` URL
    // connects in PLAINTEXT WITHOUT ERRORING (the "silent plaintext downgrade").
    //
    // These tests fail *closed* two ways: they cannot compile if the feature is
    // removed (the `.tls` field and the `TlsConnector` / `TlsHostMapping` types
    // vanish behind fred's `#[cfg]` gate), and they fail if the feature is
    // present but a TLS scheme stops producing a connector.
    //
    // The whole block is gated on `#[cfg(feature = "valkey-tls")]` because it
    // names `TlsConnector`, `TlsHostMapping`, and `.tls`, all gated behind
    // fred's TLS features — a sibling CI job (t3.tr5.t4.oq.8z) compiles this
    // crate `--no-default-features`, where those symbols do not exist.
    //
    // Cross-Epic payoff: these are also the only tests that call `from_url`
    // with a TLS scheme in-process, making them the in-process canary for the
    // dual-crypto-provider hazard Epic 4 (t1.tr5.we) inherits —
    // `TlsConnector::default_rustls()` panics when both rustls `ring` and
    // `aws_lc_rs` are compiled in.
    //
    // I/O note: constructing the rustls connector reads the system CA bundle
    // (`/etc/ssl/certs/ca-certificates.crt`; CI's `test:` job installs
    // `ca-certificates`). That local read is the ONLY I/O these tests perform
    // and is inherent to the invariant being asserted — the test-writing-guide's
    // "unit tests have no I/O" bullet is knowingly relaxed here; the alternative
    // is a live TLS server, which is far more expensive (Task 3 / rollout
    // territory).
    #[cfg(feature = "valkey-tls")]
    mod tls_feature_wiring {
        use super::*;
        use fred::types::config::{TlsConnector, TlsHostMapping};

        #[test]
        fn fred_config_tls_scheme_url_populates_rustls_connector() {
            // Both a centralized `rediss://` URL and the `rediss-cluster://`
            // suffix (which fred routes to the separate `from_url_clustered`
            // initializer) must populate a rustls TLS connector.
            for url in [
                "rediss://valkey.example.com:6380",
                "rediss-cluster://valkey.example.com:6380",
            ] {
                let cfg =
                    FredConfig::from_url(url).unwrap_or_else(|e| panic!("`{url}` must parse: {e}"));
                let tls = cfg.tls.as_ref().unwrap_or_else(|| {
                    panic!(
                        "`{url}` MUST populate `.tls` — a None here is the silent \
                         plaintext downgrade SR-1.1 fixes and t2.tr5.t4.oq's \
                         enable-rustls-ring feature guards against"
                    )
                });
                assert!(
                    matches!(tls.connector, TlsConnector::Rustls(_)),
                    "`{url}` must produce a rustls connector — a native-tls or \
                     absent connector means the feature routing regressed \
                     (t2.tr5.t4.oq)"
                );
                // Pins fred's URL-path DEFAULT for `hostnames`, the baseline
                // SR-1.5 corrects. DO NOT flip or delete this when Task 3
                // (t2.tr5.t4.wz) lands: Task 3 does not touch `from_url`, it
                // mutates the returned config afterwards inside
                // `ValkeyClient::connect`, so `from_url(...).tls.hostnames`
                // stays `TlsHostMapping::None` post-Task-3. Task 3's own tests
                // assert the post-mutation policy separately. This is the only
                // thing in the suite pinning fred's default — the entire premise
                // for SR-1.5 existing.
                assert_eq!(
                    tls.hostnames,
                    TlsHostMapping::None,
                    "`{url}` must carry fred's URL-path default hostname mapping \
                     (TlsHostMapping::None) — the SR-1.5 baseline; must NOT be \
                     flipped to DefaultHost when Task 3 (t2.tr5.t4.wz) lands"
                );
            }
        }

        #[test]
        fn fred_config_plaintext_scheme_url_leaves_tls_unset() {
            // The feature must not force TLS onto the existing plaintext path.
            for url in ["redis://127.0.0.1:6379", "redis-cluster://127.0.0.1:7000"] {
                let cfg =
                    FredConfig::from_url(url).unwrap_or_else(|e| panic!("`{url}` must parse: {e}"));
                assert!(
                    cfg.tls.is_none(),
                    "plaintext `{url}` must leave `.tls` None — enable-rustls-ring \
                     must NOT force TLS onto the plaintext path (t2.tr5.t4.oq)"
                );
            }
        }

        #[test]
        fn apply_connect_timeout_preserves_tls_connector_on_builder() {
            // Guards the shared `Builder::from_config` seam that
            // `ValkeyClient::connect` uses (src/storage/client.rs:460, :495) —
            // the same seam Epic 2 (t1.tr5.vc) mutates for its credential
            // provider. A refactor that rebuilds the config and drops `.tls`
            // (the silent plaintext downgrade) fails here.
            let mut builder = Builder::from_config(
                FredConfig::from_url("rediss://valkey.example.com:6380")
                    .expect("static rediss URL must parse"),
            );
            apply_connect_timeout(
                &mut builder,
                &connect_timeout_test_config(7),
                &PeerReplies::new(),
                UnresponsiveVerdicts::FastAndSlow,
            );
            assert!(
                builder
                    .get_config()
                    .expect("builder must carry a Config")
                    .tls
                    .is_some(),
                "apply_connect_timeout must not drop the TLS connector from the \
                 Builder — dropping it is the silent plaintext downgrade \
                 (t2.tr5.t4.oq)"
            );
            assert_eq!(
                builder.get_connection_config().connection_timeout,
                Duration::from_secs(7),
                "apply_connect_timeout must still wire the connect timeout on a \
                 TLS builder (t2.tr5.t4.oq / b.1kf)"
            );
        }
    }

    /// Slice `connect`'s function body out of this module's own source.
    ///
    /// Factored out so Epic 2 (`t1.tr5.vc`) can add its
    /// `credential_provider`-before-the-clone assertion against the same
    /// helper instead of re-deriving the slice. Slicing to `connect`'s body
    /// is what keeps the needles below from matching this test module's own
    /// string literals and the `Builder::from_config(fred_config)`
    /// occurrence in `make_builder` (currently `src/storage/client.rs:1722`).
    /// An anchor-not-found panic here is an INTENTIONAL loud maintenance
    /// signal (the `connect` signature or the data-pool anchor moved), not a
    /// flake. The likeliest trigger is not a *moved* anchor at all: the end
    /// anchor requires the data-pool retry call in `connect` to stay on ONE
    /// source line, so adding a parameter to that call must be paired with
    /// short-named local bindings that keep it unwrapped — the `diag` / `hinted`
    /// precedent, currently `src/storage/client.rs:2117`. Let rustfmt wrap that
    /// call across lines and this slicer panics with the anchor still present.
    ///
    /// Lives at `mod tests` scope (ungated), not inside the
    /// `#[cfg(feature = "valkey-tls")]` `tls_host_mapping_policy` module, so
    /// BOTH the gated ordering guard and the ungated credential-provider
    /// single-site guard reach the same slicer. The latter must run under the
    /// `--no-default-features` CI job (the credential-provider seam is not
    /// TLS-gated), and this helper uses only `include_str!` + string matching,
    /// so it compiles with no TLS symbols.
    fn connect_fn_body() -> &'static str {
        let src = include_str!("client.rs");
        // Anchor on the `connect(` opener only: the signature spans multiple
        // lines once the `redis_token_provider` parameter is present, so the
        // full one-line signature is no longer a contiguous substring. The real
        // fn definition precedes this test's own string literal in the file, so
        // `find` (first match) still lands on the function.
        let fn_start = src.find("pub async fn connect(").expect(
            "connect's signature anchor moved — update the connect source guards \
             (intentional loud maintenance signal, not a flake)",
        );
        let fn_end = fn_start
            + src[fn_start..]
                .find("connect_with_retry(\"data pool\"")
                .expect(
                    "connect's data-pool anchor moved — update the connect source guards \
                     (intentional loud maintenance signal, not a flake). Check first whether \
                     the anchor is still there but rustfmt wrapped the data-pool retry call \
                     across lines: the anchor requires that call to stay on ONE source line, so \
                     a new parameter must be paired with short-named bindings like the \
                     `diag` / `hinted` ones above it",
                );
        &src[fn_start..fn_end]
    }

    // ── t3.tr5.t4.wz SR-1.5 explicit TLS host-mapping policy ──────────────────
    //
    // These cover `apply_tls_host_mapping` (which runs in the
    // `apply_fred_config_overrides` tail — the one place a shared `FredConfig`
    // field is written), its propagation to every `Builder`, the
    // mutation-before-every-derivation ordering inside `ValkeyClient::connect`,
    // and `describe_tls_policy`'s startup-log descriptor. "Before every
    // derivation" rather than "before the clone" because `build_fred_config`
    // also derives the pub/sub config (`derive_pubsub_config`) strictly earlier
    // than the data-pool clone: a mutation landing between the two would satisfy
    // a before-the-clone rule and still leave the pub/sub connection without the
    // policy the five pools carry.
    //
    // Provable in-process (here), needs no infrastructure: which policy value
    // the code SETS (`DefaultHost` on any TLS config); that the override
    // PRESERVES the system-cert rustls connector `from_url` installed; that
    // plaintext configs keep `.tls == None`; that the policy reaches every
    // `Builder`; and that the mutation is applied before `fred_config` is cloned.
    //
    // NOT provable anywhere in CI (rollout-only, Task 4 / `t2.tr5.t4.nq`): that
    // `DefaultHost` actually makes TLS cert verification SUCCEED against a real
    // managed cluster whose `CLUSTER SLOTS` returns bare IPs. CI has no TLS
    // Valkey — `scripts/dev-valkey.sh cluster-up` and CI's `test-cluster:` job
    // (`.gitlab-ci.yml:546`, `LIVE_EDIT_TEST_CLUSTER_URL: "redis-cluster://..."`)
    // are PLAINTEXT clusters. That end-to-end success is confirmed only by
    // capturing real `CLUSTER SLOTS` output at rollout.
    //
    // No `Custom(Arc<dyn HostMapping>)` test double exists. fred consumes the
    // policy in `modify_cluster_slot_hostnames`
    // (`fred-10.1.0/src/protocol/cluster.rs:226`), which lives in `mod protocol`
    // — declared PRIVATE at `fred-10.1.0/src/lib.rs:55` — and delegates to
    // `TlsHostMapping::map` / `Server::set_tls_server_name`, both `pub(crate)`.
    // None is reachable from live-edit, so the only way to observe the mapping
    // being CONSULTED is a live clustered-TLS connect; a `Custom` double could
    // only be invoked by our own test calling `HostMapping::map` directly, which
    // is the test-review guide's "testing mocks instead of code" anti-pattern.
    // `TlsHostMapping`'s hand-written `PartialEq` also makes
    // `assert_eq!(hostnames, TlsHostMapping::DefaultHost)` exclude every
    // `Custom(_)` for free, so the negative case is covered without a double.
    //
    // fred grep anchors for a future bump: the `None` default the override
    // corrects is hardcoded in the `Into<TlsConfig>` conversion at
    // `fred-10.1.0/src/protocol/tls.rs:105-111`; the consumption point is
    // `fred-10.1.0/src/protocol/cluster.rs:226`.
    //
    // The whole block is gated `#[cfg(feature = "valkey-tls")]` because every
    // test names `TlsHostMapping`, `TlsConnector`, or `.tls`, all `#[cfg]`-gated
    // behind fred's TLS features — a sibling CI job (`t3.tr5.t4.oq.8z`) compiles
    // this crate `--no-default-features`, where those symbols do not exist. Test
    // 5 (the source-order ordering guard) uses `include_str!` + string matching
    // and would compile ungated, but is kept inside the gate for cohesion: the
    // ordering property it guards only exists when the feature is on.
    //
    // I/O note: constructing a TLS config via `from_url` reads the system CA
    // bundle through `TlsConnector::default_rustls()` — the same known,
    // deliberate relaxation `t3.tr5.t4.oq.9p`'s block already documents; that
    // local read is the only I/O here.
    #[cfg(feature = "valkey-tls")]
    mod tls_host_mapping_policy {
        use super::*;
        use fred::types::config::{TlsConnector, TlsHostMapping};

        // TLS schemes whose `from_url` path installs a rustls connector with
        // fred's default `hostnames: None` — the baseline SR-1.5 corrects.
        const TLS_URLS: &[&str] = &[
            // Centralized `rediss://` → `from_url_centralized`
            // (`fred-10.1.0/src/types/config.rs:861`, `tls:` initializer `:877`).
            "rediss://valkey.example.com:6380",
            // Clustered `rediss-cluster://` → the separate `from_url_clustered`
            // (`:902`, `tls:` initializer `:921`); the production managed shape.
            "rediss-cluster://valkey.example.com:10000",
            // The Valkey-flavoured TLS scheme (`VALKEY_TLS_SCHEME`,
            // `fred-10.1.0/src/utils.rs:56`), absent from every doc in this repo
            // and therefore the easiest form for an implementation to miss.
            "valkeys-cluster://valkey.example.com:6379",
        ];

        // Plaintext schemes whose `from_url` path leaves `.tls == None`.
        const PLAINTEXT_URLS: &[&str] = &[
            "redis://127.0.0.1:6379",         // binary default shape
            "redis-cluster://127.0.0.1:7000", // byte-identical to CI's LIVE_EDIT_TEST_CLUSTER_URL (.gitlab-ci.yml:546)
            "valkey://127.0.0.1:6379",
            "valkey-cluster://127.0.0.1:7000",
        ];

        #[test]
        fn apply_tls_host_mapping_tls_scheme_config_sets_explicit_policy() {
            // Mechanical guard against the silent-compile-out landmine in
            // `t3.tr5.t4.wz.ee`: if the helper's `if let Some(tls) = ...`
            // predicate were wrong the override would be compiled out and this
            // test would fail outright.
            for url in TLS_URLS {
                let mut cfg =
                    FredConfig::from_url(url).unwrap_or_else(|e| panic!("`{url}` must parse: {e}"));
                // Pre-state: fred's URL-path default that the override corrects.
                assert_eq!(
                    cfg.tls.as_ref().unwrap().hostnames,
                    TlsHostMapping::None,
                    "`{url}` pre-state must be fred's URL-path default \
                     TlsHostMapping::None"
                );

                apply_tls_host_mapping(&mut cfg);

                assert_eq!(
                    cfg.tls.as_ref().unwrap().hostnames,
                    TlsHostMapping::DefaultHost,
                    "`{url}` must carry TlsHostMapping::DefaultHost after the \
                     override — under the inherited None, post-topology-discovery \
                     connections to non-seed cluster nodes advertised as bare IPs \
                     hit a cert-SAN mismatch and fail the TLS handshake (SR-1.5)"
                );
            }
        }

        #[test]
        fn apply_tls_host_mapping_tls_scheme_config_preserves_rustls_connector() {
            let mut cfg = FredConfig::from_url("rediss-cluster://valkey.example.com:10000")
                .expect("static rediss-cluster URL must parse");

            apply_tls_host_mapping(&mut cfg);

            // A variant match is the STRONGEST available check: `TlsConnector`'s
            // `PartialEq` returns `true` unconditionally
            // (`fred-10.1.0/src/protocol/tls.rs:122-126`), so `assert_eq!` on
            // connectors would be vacuous, and a `format!("{:?}", ...)` compare
            // would pin fred's Debug string rather than the invariant. This
            // catches an implementation that assigns a whole fresh
            // `TlsConfig { connector, hostnames }` and silently swaps or drops
            // the system-cert connector `from_url` installed.
            assert!(
                matches!(cfg.tls.as_ref().unwrap().connector, TlsConnector::Rustls(_)),
                "the override must preserve the system-cert rustls connector \
                 that from_url installed — a swapped/dropped connector is a \
                 silent cert-verification regression (SR-1.5)"
            );
        }

        #[test]
        fn apply_tls_host_mapping_plaintext_config_leaves_tls_unset() {
            // SR-1.5's plaintext non-regression at unit scope: fails loudly on an
            // implementation that unconditionally constructs a `TlsConfig` (which
            // would upgrade every plaintext deployment to TLS at startup) and on
            // one that does `cfg.tls.as_mut().unwrap()` (which would panic on
            // every plaintext connect).
            for url in PLAINTEXT_URLS {
                let mut cfg =
                    FredConfig::from_url(url).unwrap_or_else(|e| panic!("`{url}` must parse: {e}"));
                assert!(
                    cfg.tls.is_none(),
                    "plaintext `{url}` must leave `.tls` None before the override"
                );

                apply_tls_host_mapping(&mut cfg);

                assert!(
                    cfg.tls.is_none(),
                    "plaintext `{url}` must STILL leave `.tls` None after the \
                     override — the helper must not construct a TlsConfig on the \
                     plaintext path (would force TLS onto every plaintext \
                     deployment) nor panic (SR-1.5)"
                );
            }
        }

        #[test]
        fn tls_host_mapping_survives_both_builder_from_config_shapes() {
            // Reproduces `ValkeyClient::connect`'s exact two-builder construction
            // against one config built through `build_fred_config`'s override
            // path. This PROVES `Clone` carries `.tls` — the mechanism by which a
            // config mutated before the clone reaches all six connection sets
            // (data pool, txn pool, the stream readers, maintenance pool,
            // snapshot pool, subscriber client) — and would fail on a fred bump
            // that changed clone
            // semantics. It does NOT prove the mutation happens before the clone
            // inside `connect`; test 5 is what proves that.
            let mut fred_config = FredConfig::from_url("rediss-cluster://valkey.example.com:10000")
                .expect("static rediss-cluster URL must parse");
            apply_tls_host_mapping(&mut fred_config);
            let pool_builder = Builder::from_config(fred_config.clone()); // mirrors connect's data-pool builder
            let subscriber_builder = Builder::from_config(fred_config); // mirrors connect's subscriber builder

            let pool_tls = pool_builder
                .get_config()
                .expect("pool builder must carry a Config")
                .tls
                .clone();
            let sub_tls = subscriber_builder
                .get_config()
                .expect("subscriber builder must carry a Config")
                .tls
                .clone();

            assert!(
                pool_tls.is_some() && sub_tls.is_some(),
                "both builders must carry a TLS config after Clone"
            );
            assert_eq!(
                pool_tls.as_ref().unwrap().hostnames,
                TlsHostMapping::DefaultHost,
                "the pool builder's config must carry the DefaultHost policy"
            );
            assert_eq!(
                sub_tls.as_ref().unwrap().hostnames,
                TlsHostMapping::DefaultHost,
                "the subscriber builder's config must carry the DefaultHost policy"
            );
            // `TlsConfig` derives `PartialEq` (`fred-10.1.0/src/protocol/tls.rs:97`).
            assert_eq!(
                pool_tls, sub_tls,
                "the cloned and moved TlsConfig values must be identical — a \
                 divergence would mean the two builders disagree on transport \
                 policy (partial-TLS deployment)"
            );
        }

        // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
        #[test]
        fn connect_applies_fred_config_overrides_before_the_first_clone() {
            // A deliberate, precedented departure from "assert on behavior, not
            // implementation": `connect` is async, builds and consumes both
            // `Builder`s internally, and returns only a connected `ValkeyClient`,
            // so the mutate-before-clone ordering is not observable from any
            // runtime value without a live server. Same technique as
            // `src/auth/rest.rs`'s
            // `classify_request_error_calls_without_url_on_the_status_bearing_branch`
            // and `src/domain/ompf.rs`'s `source_has_no_try_into_unwrap_on_input_slices`.
            let body = connect_fn_body();

            let seam_idx = body.find("build_fred_config(").expect(
                "connect must build its FredConfigs through build_fred_config — a \
                 missing single-construction seam risks a partial-TLS (later \
                 partial-auth) deployment where some of the six connections carry \
                 the policy and others do not",
            );
            // b.j64: `build_fred_config` now returns BOTH configs — `shared` for
            // the five pools and `pubsub` for the dedicated pub/sub client, the
            // latter derived from the former INSIDE the seam. `connect`
            // destructures them; it must not construct either itself. The
            // destructure needle is what proves the pub/sub config came from the
            // seam rather than from a second, competing construction.
            let destructure_idx = body.find("let FredConfigs {").expect(
                "connect must destructure the FredConfigs the single seam returns — \
                 an anchor-not-found panic here is an intentional loud maintenance \
                 signal, not a flake",
            );
            let clone_idx = body
                .find("Builder::from_config(fred_config.clone())")
                .expect("connect must build the data-pool Builder from a CLONE of fred_config");
            // The pub/sub subscriber client is the ONE connection built from the
            // derived config. The needle cannot cross-match the data-pool form
            // (different identifier) nor the subscriber-POOL form
            // (`Builder::from_config(subscriber_fred_config)`).
            let pubsub_idx = body
                .find("Builder::from_config(pubsub_fred_config)")
                .expect(
                    "connect must build the pub/sub subscriber client from the seam-derived \
                     pubsub_fred_config",
                );

            assert!(
                seam_idx < clone_idx,
                "build_fred_config (the sole override site) must run BEFORE the \
                 first fred_config.clone() — otherwise the data pool would be \
                 built from an un-mutated config, yielding a partial-TLS \
                 deployment where some of the six connections carry the policy \
                 and others do not"
            );
            assert!(
                destructure_idx < seam_idx && seam_idx < clone_idx,
                "the `let FredConfigs {{ … }} = build_fred_config(…)` destructure must be \
                 the statement that binds BOTH configs, and it must complete before the \
                 first builder — both configs must be in hand, from the one seam, before \
                 any connection is built (the pattern precedes the call in source order \
                 because it is one `let` statement)"
            );
            assert!(
                clone_idx < pubsub_idx,
                "the data-pool clone must precede the pub/sub subscriber build — a \
                 reorder would not compile, but this pins the observable source \
                 order for the guard"
            );

            assert_eq!(
                body.matches("build_fred_config(").count(),
                1,
                "connect must have exactly one FredConfig construction site \
                 (build_fred_config). The whole mutation seam lives behind that \
                 one call — apply_fred_config_overrides writing the shared \
                 config, plus the sanctioned derived writes of \
                 derive_pubsub_config and derive_reader_config — \
                 so a second, competing site inside connect is the \
                 partial-TLS/partial-auth hazard t3.tr5.t4.wz.ee closes and Epic 2 \
                 is directed to EXTEND, not duplicate"
            );
            // b.j64: `connect` receives the pub/sub config fully formed and must
            // not touch it. A field write here (`pubsub_fred_config.tls = …`,
            // `.version = …`, anything) would be a post-seam divergence on an
            // auth- or TLS-bearing field — precisely the partial-auth hazard the
            // sibling `connect_has_no_second_credential_provider_mutation_site`
            // guard exists for, in a shape that guard's needle would miss.
            assert_eq!(
                body.matches("pubsub_fred_config.").count(),
                0,
                "connect must NOT mutate pubsub_fred_config — every difference \
                 between it and the shared config is decided by \
                 derive_pubsub_config, the one sanctioned derived write inside \
                 the build_fred_config seam"
            );
            assert_eq!(
                body.matches("derive_pubsub_config(").count(),
                0,
                "connect must NOT derive the pub/sub config itself — that derivation \
                 belongs inside build_fred_config so it cannot drift from the \
                 override it is derived from"
            );
            assert_eq!(
                body.matches("FredConfig::from_url").count(),
                0,
                "connect must NOT call FredConfig::from_url directly — that would \
                 bypass build_fred_config's override and reintroduce the \
                 partial-TLS deployment hazard"
            );
            // The same holds past the data-pool anchor, where the readers'
            // template and their `ReaderConnector` are built: a config parsed
            // afresh there would connect every stream reader without the
            // override.
            assert_eq!(
                connect_whole_body().matches("FredConfig::from_url").count(),
                0,
                "the whole of connect, the readers' template and connector included, must \
                 not call FredConfig::from_url directly"
            );
        }

        #[test]
        fn describe_tls_policy_reports_effective_transport_faithfully() {
            // Pure, offline branch coverage of the startup-log descriptor. The
            // `--no-default-features` `("unavailable", "n/a")` arm
            // (`#[cfg(not(feature = "valkey-tls"))]`) is unreachable in a
            // default-feature run and is not tested here; it is exercised by the
            // `--no-default-features` clippy job compiling that arm.
            let plaintext = FredConfig::from_url("redis://127.0.0.1:6379")
                .expect("static plaintext URL must parse");
            assert_eq!(
                describe_tls_policy(&plaintext),
                ("disabled", "n/a"),
                "a plaintext config must describe as (disabled, n/a)"
            );

            let mut tls_mapped = FredConfig::from_url("rediss-cluster://valkey.example.com:10000")
                .expect("static rediss-cluster URL must parse");
            apply_tls_host_mapping(&mut tls_mapped);
            assert_eq!(
                describe_tls_policy(&tls_mapped),
                ("enabled", "default-host"),
                "a post-apply_tls_host_mapping TLS config must describe as \
                 (enabled, default-host)"
            );

            // Not mapped: proves the descriptor reports fred's default faithfully
            // rather than hardcoding the expected answer.
            let tls_default = FredConfig::from_url("rediss://valkey.example.com:6380")
                .expect("static rediss URL must parse");
            assert_eq!(
                describe_tls_policy(&tls_default),
                ("enabled", "none"),
                "an un-mapped TLS config must describe fred's default as \
                 (enabled, none)"
            );
        }
    }

    // ── t3.tr5.vc.aa credential-provider single mutation-site guard ───────────
    //
    // Epic 2's load-bearing cross-Epic contract: the shared `credential_provider`
    // is installed ONLY inside `apply_fred_config_overrides`, and the two other
    // writes of that field in the service — `derive_pubsub_config` (b.j64) and
    // `derive_reader_config` (b.19i), each re-seating the SAME provider `Arc`
    // wrapped for cadence — are sanctioned derived writes inside the same
    // `build_fred_config` seam. (The `test-support` seam that replaces the
    // installed provider also lives there, before both derivations.) All run
    // before the pool-builder `.clone()` in `ValkeyClient::connect`, so all six connection
    // sets (data pool, txn pool, the stream readers, maintenance pool, snapshot
    // pool, subscriber client) inherit one credential. `connect` itself must
    // contain neither. A
    // SECOND `FredConfig` / `Builder` mutation site inside `connect` — one that
    // seats a credential provider on some builders but not others — is how a
    // partial-auth deployment (some pools authenticated, some not) becomes
    // possible, and it slips past the sibling ordering guard
    // (`connect_applies_fred_config_overrides_before_the_first_clone`) whose
    // needles only check `build_fred_config` ordering, not the ABSENCE of a
    // competing credential-provider write.
    //
    // Ungated (NOT `#[cfg(feature = "valkey-tls")]`): the credential-provider
    // seam is not TLS-gated, so this guard must also run under the
    // `--no-default-features` CI job. It reuses the `mod tests`-scope
    // `connect_fn_body()` slicer (its anchor-not-found `.expect(...)`s are the
    // loud maintenance signal on `connect`-signature / data-pool-anchor drift),
    // which carries no TLS symbols and compiles feature-free.
    mod connect_credential_provider_single_site {
        use super::*;

        // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
        #[test]
        fn connect_has_no_second_credential_provider_mutation_site() {
            // Same source-inspection technique, and same rationale, as the
            // sibling `connect_applies_fred_config_overrides_before_the_first_clone`:
            // `connect` is async, builds and consumes both `Builder`s
            // internally, and returns only a connected `ValkeyClient`, so "no
            // second credential mutation site" is not observable from any
            // runtime value without a live server. `connect_fn_body()` panics
            // loudly if the `connect` anchors drift.
            let body = connect_fn_body();

            // No `credential_provider` token anywhere in `connect`'s body: every
            // write of that field — `apply_fred_config_overrides`' install and
            // the sanctioned cadence re-seats of `derive_pubsub_config` and
            // `derive_reader_config` — lives behind `build_fred_config`,
            // OUTSIDE this sliced body. Any occurrence
            // here — a `credential_provider = ...` assignment, a
            // `Config { credential_provider: .. }` re-init, or a fred setter
            // naming the field — is the partial-auth second-site hazard. This
            // bare-token needle is strictly stronger than the `credential_provider =`
            // form the Subtask names, and cannot cross-match the legitimate
            // `build_fred_config(` seam or the
            // `Builder::from_config(fred_config[.clone()])` construction lines
            // (none of them contain the substring).
            assert_eq!(
                body.matches("credential_provider").count(),
                0,
                "connect must NOT name `credential_provider` — every write of \
                 that field lives behind build_fred_config \
                 (apply_fred_config_overrides' install, plus the sanctioned \
                 cadence re-seats of the same Arc by derive_pubsub_config and \
                 derive_reader_config), applied BEFORE the fred_config.clone() \
                 so all six connections inherit one credential. A write inside \
                 connect authenticates \
                 some pools and not others (partial-auth deployment) and slips \
                 past the ordering guard, whose needles only check \
                 build_fred_config ordering"
            );
            // The same holds past the data-pool anchor, where the readers'
            // template and connector are built (b.19i hands the connector the
            // seam's provider; it must not name the field either).
            assert_eq!(
                connect_whole_body().matches("credential_provider").count(),
                0,
                "the whole of connect, the readers' template included, must not name \
                 `credential_provider`"
            );

            // No `Builder::with_config(...)` second config-install site: the two
            // builders are derived ONLY via `Builder::from_config(...)` from the
            // single, already-overridden `fred_config`. A `.with_config(` on a
            // builder would re-seat a (possibly credential-less) config after the
            // clone — the same partial-auth hazard in a different shape. The
            // `.with_config(` needle cannot cross-match `Builder::from_config(`
            // or `.set_performance_config(` (neither contains the `.with_config(`
            // substring).
            assert_eq!(
                body.matches(".with_config(").count(),
                0,
                "connect must NOT re-seat a Builder config via `.with_config(` — \
                 both builders must derive from the single build_fred_config \
                 output so every connection carries the same credential_provider; \
                 a second with_config site is the partial-auth hazard"
            );
            // Also past the data-pool anchor, over the readers' template and
            // the `ReaderConnector` construction: the connector's own
            // per-connection divergence lives in `ReaderConnector`'s methods
            // (pinned by `reader_connector_diverges_only_the_server_and_database_of_its_template`),
            // never in `connect`.
            assert_eq!(
                connect_whole_body().matches(".with_config(").count(),
                0,
                "the whole of connect, the readers' template and connector included, must \
                 not re-seat a Builder config via `.with_config(`"
            );
        }
    }

    // ── t3.tr5.jv.2b.61 access-token-file arm falls through to apply_tls_host_mapping ──────
    //
    // Cross-Epic contract recorded as Subtask 61's additional acceptance
    // criterion (Epic 2 PM review): the `access-token-file` arm of
    // `apply_fred_config_overrides` MUST set `credential_provider` and then FALL
    // THROUGH the `match` to the shared `#[cfg(feature = "valkey-tls")]
    // apply_tls_host_mapping(fred_config)` tail — it must NOT early-`return
    // Ok(())` after installing the provider. Early-returning drops the SR-1.5
    // `TlsHostMapping::DefaultHost` step and breaks clustered TLS on Azure
    // Managed Redis (bare-IP `CLUSTER SLOTS` → cert-SAN mismatch), a
    // rollout-gated failure CI cannot catch (§15.3).
    //
    // The dispatch tests (`auth_mode_dispatch`) pin provider-install /
    // fail-closed but use a plaintext URL; the host-mapping tests
    // (`tls_host_mapping_policy`) pin the tail but only on the STATIC path.
    // Neither drives the access-token-file arm THROUGH `build_fred_config` with a TLS URL,
    // so the fall-through is untested without this block. THE REGRESSION THIS
    // GUARDS: rewrite the access-token-file arm to `credential_provider = Some(..); return
    // Ok(())` and `describe_tls_policy` reports `"none"` instead of
    // `"default-host"` — every assertion below fails.
    //
    // Gated `#[cfg(feature = "valkey-tls")]` (on top of the enclosing
    // `#[cfg(test)]`, so effectively `#[cfg(all(test, feature = "valkey-tls"))]`)
    // because it needs the `apply_tls_host_mapping` tail compiled in and names
    // `describe_tls_policy`'s TLS descriptor — the `--no-default-features`
    // sibling CI job (`t3.tr5.t4.oq.8z`) has neither. `#[tokio::test]` because
    // `ServiceTokenProvider::new_with_refresh_lead` spawns a polling task (needs
    // a runtime); the token file need not exist — the access-token-file arm only WRAPS the
    // provider in `TokenFileCredentialProvider::new`, it never calls `fetch()` /
    // `cached_bearer_if_unexpired()` here.
    //
    // I/O note: `FredConfig::from_url` on a TLS scheme reads the system CA
    // bundle through `TlsConnector::default_rustls()` — the same deliberate
    // relaxation the sibling `tls_host_mapping_policy` block already documents.
    #[cfg(feature = "valkey-tls")]
    mod token_file_tls_host_mapping_fallthrough {
        use tempfile::tempdir;

        use super::*;

        /// TLS (`rediss-cluster://`) — the production Azure Managed Redis shape
        /// — so `apply_tls_host_mapping`'s `if let Some(tls) = ...` branch
        /// fires. A plaintext URL leaves `.tls == None`, making the tail a
        /// no-op that could not distinguish fall-through from an early return.
        /// The one input this module does not share with its siblings.
        const TLS_URL: &str = "rediss-cluster://valkey.example.com:10000";

        /// Filename under the per-test `TempDir` handed to the shared
        /// `redis_token_provider`. The file is never created: the
        /// access-token-file arm only WRAPS the provider, so no `fetch()` /
        /// `cached_bearer_if_unexpired()` runs here.
        const TOKEN_FILE_NAME: &str = "redis-token.json";

        /// The shared access-token-file `Config` builder, pinned to this
        /// module's TLS URL — the minimum to reach the arm's provider install
        /// and fall through to the TLS tail.
        fn token_file_tls_config() -> Config {
            token_file_test_config(TLS_URL)
        }

        #[tokio::test]
        async fn access_token_file_path_preserves_default_host_tls_mapping() {
            let dir = tempdir().expect("tempdir");
            let (provider, cancel) = redis_token_provider(
                dir.path().join(TOKEN_FILE_NAME),
                "token-file-tls-fallthrough-test",
            );

            let fred_config = build_fred_config(&token_file_tls_config(), Some(provider), None)
                .expect(
                    "access-token-file mode with a provider + Object-ID username + TLS URL must \
                 build a FredConfig",
                )
                .shared;

            // The provider WAS installed — proves control took the access-token-file arm
            // (not that the mapping happened to be applied on a provider-less
            // path).
            assert!(
                fred_config.credential_provider.is_some(),
                "access-token-file mode must install the credential_provider (SR-4.1)"
            );

            // The load-bearing assertion: the access-token-file arm FELL THROUGH to
            // apply_tls_host_mapping. An early `return Ok(())` after installing
            // the provider leaves fred's URL-path default
            // (`TlsHostMapping::None` → "none"), dropping the SR-1.5 DefaultHost
            // step and breaking clustered TLS on Azure Managed Redis.
            assert_eq!(
                describe_tls_policy(&fred_config),
                ("enabled", "default-host"),
                "the access-token-file arm must fall through to apply_tls_host_mapping — a \
                 host_mapping of \"none\" means it early-returned Ok(()) after \
                 installing the credential_provider, dropping the SR-1.5 \
                 DefaultHost step and breaking clustered TLS on Azure Managed \
                 Redis (§15.3)"
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn access_token_file_path_does_not_regress_host_mapping_relative_to_static() {
            // Contrast: the STATIC path on the SAME TLS URL already reaches the
            // tail (no provider install, straight fall-through). Pinning that the
            // access-token-file path yields the identical ("enabled", "default-host")
            // isolates the fall-through as the property under test — access-token-file must
            // not REGRESS the host mapping relative to the non-access-token-file TLS path.
            // This pins the fall-through, not TLS itself.
            let mut static_config = token_file_tls_config();
            static_config.valkey_auth_mode = ValkeyAuthMode::Static;
            let static_fred = build_fred_config(&static_config, None, None)
                .expect("static mode on a TLS URL must build a FredConfig")
                .shared;
            assert_eq!(
                describe_tls_policy(&static_fred),
                ("enabled", "default-host"),
                "the static TLS path must reach apply_tls_host_mapping — the \
                 fall-through baseline the access-token-file path must match"
            );

            let dir = tempdir().expect("tempdir");
            let (provider, cancel) = redis_token_provider(
                dir.path().join(TOKEN_FILE_NAME),
                "token-file-tls-regress-test",
            );
            let token_file_fred = build_fred_config(&token_file_tls_config(), Some(provider), None)
                .expect("access-token-file mode on a TLS URL must build a FredConfig")
                .shared;

            assert_eq!(
                describe_tls_policy(&token_file_fred),
                describe_tls_policy(&static_fred),
                "access-token-file must not regress the TLS host-mapping policy relative to \
                 the static path — both must fall through to the same \
                 (enabled, default-host)"
            );

            cancel.cancel();
        }
    }

    // ── t3.tr5.we.6w.ii IAM arm falls through to apply_tls_host_mapping ──────
    //
    // The direct AWS analog of `token_file_tls_host_mapping_fallthrough` above. The
    // `iam` arm of `apply_fred_config_overrides` installs a `credential_provider`
    // (wrapping the resolved AWS credential source in an `IamCredentialProvider`)
    // and then MUST fall through to the shared `apply_tls_host_mapping` tail — it
    // must NOT early-`return Ok(())`. Skipping the tail drops the SR-1.5
    // `TlsHostMapping::DefaultHost` step and silently breaks clustered TLS on AWS
    // (bare-IP `CLUSTER SLOTS` → cert-SAN mismatch on the post-discovery node
    // connections), a rollout-gated failure CI cannot catch (§15.3).
    //
    // Gated on `valkey-tls` because it names the effective TLS policy via
    // `describe_tls_policy`, whose `TlsHostMapping` arms only exist behind fred's
    // TLS features; the `--no-default-features` CI job compiles those out.
    //
    // I/O note: `FredConfig::from_url` on a TLS scheme reads the system CA bundle
    // through `TlsConnector::default_rustls()` — the same deliberate relaxation
    // the sibling `access-token-file`/`tls_host_mapping_policy` blocks already document. No
    // tokio runtime is needed here: unlike the access-token-file `ServiceTokenProvider` (which
    // spawns a polling task), the AWS credential source is a plain
    // `SharedCredentialsProvider` the arm only wraps — no `fetch()` is driven.
    #[cfg(feature = "valkey-tls")]
    mod iam_tls_host_mapping_fallthrough {
        use super::*;

        /// TLS (`rediss-cluster://`) — the production AWS MemoryDB / ElastiCache
        /// cluster shape — so `apply_tls_host_mapping`'s `if let Some(tls) = ...`
        /// branch fires. A plaintext URL leaves `.tls == None`, making the tail a
        /// no-op that could not distinguish fall-through from an early return.
        /// The one input this module does not share with its siblings.
        const TLS_URL: &str = "rediss-cluster://valkey.example.com:6379";

        /// The shared iam `Config` builder, pinned to this module's TLS URL —
        /// the minimum to reach the arm's provider install and fall through to
        /// the TLS tail.
        fn iam_tls_config() -> Config {
            iam_test_config(TLS_URL)
        }

        #[test]
        fn iam_path_preserves_default_host_tls_mapping() {
            let fred_config =
                build_fred_config(&iam_tls_config(), None, Some(stub_iam_credentials()))
                    .expect(
                        "iam mode with an AWS credential source + username/region/\
                         cluster-host + TLS URL must build a FredConfig",
                    )
                    .shared;

            // The provider WAS installed — proves control took the iam arm (not
            // that the mapping happened to be applied on a provider-less path).
            assert!(
                fred_config.credential_provider.is_some(),
                "iam mode must install the credential_provider (SDD §15.3)"
            );

            // The load-bearing assertion: the iam arm FELL THROUGH to
            // apply_tls_host_mapping. An early `return Ok(())` after installing
            // the provider leaves fred's URL-path default
            // (`TlsHostMapping::None` → "none"), dropping the SR-1.5 DefaultHost
            // step and breaking clustered TLS on AWS MemoryDB / ElastiCache.
            assert_eq!(
                describe_tls_policy(&fred_config),
                ("enabled", "default-host"),
                "the iam arm must fall through to apply_tls_host_mapping — a \
                 host_mapping of \"none\" means it early-returned Ok(()) after \
                 installing the credential_provider, dropping the SR-1.5 \
                 DefaultHost step and breaking clustered TLS on AWS (§15.3)"
            );
        }

        #[test]
        fn iam_path_does_not_regress_host_mapping_relative_to_static() {
            // Contrast: the STATIC path on the SAME TLS URL already reaches the
            // tail (no provider install, straight fall-through). Pinning that the
            // iam path yields the identical ("enabled", "default-host") isolates
            // the fall-through as the property under test — iam must not REGRESS
            // the host mapping relative to the non-iam TLS path. This pins the
            // fall-through, not TLS itself.
            let mut static_config = iam_tls_config();
            static_config.valkey_auth_mode = ValkeyAuthMode::Static;
            let static_fred = build_fred_config(&static_config, None, None)
                .expect("static mode on a TLS URL must build a FredConfig")
                .shared;
            assert_eq!(
                describe_tls_policy(&static_fred),
                ("enabled", "default-host"),
                "the static TLS path must reach apply_tls_host_mapping — the \
                 fall-through baseline the iam path must match"
            );

            let iam_fred = build_fred_config(&iam_tls_config(), None, Some(stub_iam_credentials()))
                .expect("iam mode on a TLS URL must build a FredConfig")
                .shared;

            assert_eq!(
                describe_tls_policy(&iam_fred),
                describe_tls_policy(&static_fred),
                "iam must not regress the TLS host-mapping policy relative to the \
                 static path — both must fall through to the same \
                 (enabled, default-host)"
            );
        }
    }

    // ── b.cbe auth-rejection diagnostics ──────────────────────────────────────
    //
    // A managed backing store answers every distinct cause of a rejected AUTH
    // with the same opaque `WRONGPASS invalid username-password pair`, so the
    // discriminating information is not in the response. It is here: this
    // service holds its own configuration and the non-secret claims of the
    // token it sent. `ValkeyAuthDiagnostics` pairs the two and renders a
    // cause-specific operator hint.
    //
    // What is provable in-process, and covered below:
    //
    // - `is_server_auth_rejection` — the classifier that decides whether a
    //   fred error is the SERVER refusing our credential. Its prefix anchor is
    //   load-bearing rather than decorative, so the negative case is driven
    //   through the REAL `TokenFileCredentialProvider::fetch()` error rather
    //   than a hand-copied string.
    // - `render_hint` — per-mode dispatch, per-verdict substitutions, the
    //   `exp` rendering, and (SR-6.2) that no token bytes reach any arm.
    // - the two rate-limit gates — the per-`connect()` `AtomicBool` and the
    //   process-wide `AuthHintGate` — asserted on the emitted log lines.
    // - `log_healthy_connect` — emits under `accessTokenFile`, silent
    //   elsewhere.
    // - the emission LEVELS — connect at WARN (a boot that recovers on a later
    //   retry must not have paged), steady-state at ERROR.
    //
    // NOT OBSERVABLE at runtime here, and SOURCE-GUARDED instead: that
    // `spawn_error_metric` routes a live fred error event into
    // `emit_steady_state_hint`, that `ValkeyClient::connect` calls
    // `log_healthy_connect`, and that `install_observability_hooks` hands the
    // shared diagnostics to every hook. fred's error events arrive on a
    // broadcast channel owned by a connected `Client`, and `EventInterface`
    // cannot be implemented by a stub without `ClientLike`'s whole connection
    // surface, so *running* those paths needs a credential-rejecting server that
    // no environment here provides. The `wiring_guards` submodule below closes
    // the gap the only way left: `include_str!` + string matching over the three
    // function bodies, the same technique the sibling `connect_fn_body()` guards
    // in this file already use. Those guards pin the WIRING; the behaviour each
    // wired call has is covered by the runtime tests here.
    // ── b.7td round 2: the per-connection hook-handle registry ──────────────
    //
    // Lives in-crate rather than in `tests/integration/storage/txn_checkout_test.rs`
    // because `pool_hook_handles` is a private field with no accessor, and the
    // property under test is precisely that the *map* stays bounded — which no
    // public observable reports. Deliberately no test-only accessor was added
    // for it: an in-crate `#[cfg(test)]` module already reaches the field, so
    // widening the production API would buy nothing.
    /// The per-node state-proof streak map's bucket-resolution rule (b.289
    /// sub-task 4).
    ///
    /// [`ValkeyClient::bucket_key`] is the single place the `""`-folding rule
    /// lives, and both `bump_state_proof_streak` and `clear_state_proof_streak`
    /// go through it. That sharing is the whole correctness argument — if bump
    /// folds a verdict into `""` and clear removes only the exact key it was
    /// handed, the shared bucket is never collected and the breaker latches for
    /// the life of the process — so it is pinned here as a pure function over a
    /// hand-built map, plus a source fence that the two call sites still route
    /// through it.
    ///
    /// The map itself lives on a constructed `ValkeyClient`, which needs a live
    /// Valkey, so the bump/clear round trip is pinned in
    /// `tests/integration/storage/b_289_tests.rs` instead. What is unit-testable
    /// without I/O is the rule, and the rule is where the round-1 defect was.
    mod state_proof_streak_buckets {
        use super::*;

        /// A map holding exactly `n` distinct node names, none of them `""`.
        fn streaks_with(n: usize) -> HashMap<String, u32> {
            (0..n).map(|i| (format!("10.0.0.{i}:6379"), 1u32)).collect()
        }

        /// A node that already owns a bucket keeps it, whatever the map's
        /// occupancy — including at the cap.
        ///
        /// This is what keeps a node that latched the breaker *before* the map
        /// filled from having its later verdicts (and, worse, its later
        /// `Clean`s) silently redirected into the shared bucket, which would
        /// leave its own entry permanently uncollectable.
        #[test]
        fn a_node_that_already_has_a_bucket_keeps_it_even_at_the_cap() {
            let mut full = streaks_with(TXN_STATE_PROOF_STREAK_MAX_NODES);
            assert_eq!(
                full.len(),
                TXN_STATE_PROOF_STREAK_MAX_NODES,
                "precondition: the fixture must actually be at the cap, or the assertion below \
                 is about the has-room branch instead"
            );
            let existing = full
                .keys()
                .next()
                .expect("the fixture is non-empty")
                .clone();

            assert_eq!(
                ValkeyClient::bucket_key(&full, &existing),
                existing.as_str(),
                "a node with a live bucket must resolve to its OWN bucket even when the map is \
                 full: the cap bounds how many NEW nodes may be tracked, not whether an already \
                 tracked one keeps its streak. Redirecting it to the shared bucket would make its \
                 own entry unreachable by any later Clean and latch the breaker for that node \
                 permanently"
            );

            // And the same node resolves the same way once the entry is gone —
            // there is now room, so it takes its own bucket by the other branch.
            full.remove(&existing);
            assert_eq!(
                ValkeyClient::bucket_key(&full, &existing),
                existing.as_str(),
                "with the entry removed the map is one under the cap, so the node takes its own \
                 bucket by the has-room branch — the two branches must agree for this node or a \
                 bump and the Clean that follows it would land in different entries"
            );
        }

        /// An unseen node gets its own bucket while the map has room. This is
        /// the ordinary cluster case and the whole point of the per-node split:
        /// node A's streak must not be reset by node B's `Clean`.
        #[test]
        fn an_unseen_node_gets_its_own_bucket_while_there_is_room() {
            let below_cap = streaks_with(TXN_STATE_PROOF_STREAK_MAX_NODES - 1);
            assert_eq!(
                ValkeyClient::bucket_key(&below_cap, "10.9.9.9:6379"),
                "10.9.9.9:6379",
                "with room under the cap a new node must be tracked separately; folding it into \
                 the shared bucket would restore the pool-wide behaviour b.289 sub-task 4 exists \
                 to end"
            );
        }

        /// At the cap an unseen node folds into the shared bucket rather than
        /// being dropped.
        ///
        /// Folding, not dropping, is the deliberate choice: a verdict that
        /// vanished would make the breaker unreachable for every node past the
        /// 64th, and the shared bucket is re-armed by *any* `Clean`, so the
        /// worst case degenerates to the pre-b.289 pool-wide contract rather
        /// than to no contract at all.
        #[test]
        fn an_unseen_node_folds_into_the_shared_bucket_at_the_cap() {
            let full = streaks_with(TXN_STATE_PROOF_STREAK_MAX_NODES);
            assert_eq!(
                ValkeyClient::bucket_key(&full, "10.9.9.9:6379"),
                "",
                "at the cap a new node must fold into the shared bucket. Dropping the verdict \
                 instead would silently disable the breaker for every node past the cap; giving \
                 it a 65th entry would make the map unbounded over a deployment's lifetime of \
                 node churn"
            );
        }

        /// The shared bucket resolves to itself in every occupancy, so the
        /// unnameable-node path is total.
        #[test]
        fn the_shared_bucket_always_resolves_to_itself() {
            for streaks in [
                HashMap::new(),
                streaks_with(1),
                streaks_with(TXN_STATE_PROOF_STREAK_MAX_NODES - 1),
                streaks_with(TXN_STATE_PROOF_STREAK_MAX_NODES),
            ] {
                assert_eq!(
                    ValkeyClient::bucket_key(&streaks, ""),
                    "",
                    "the shared bucket must be its own key at every occupancy — it is where \
                     `probe_node_bucket` sends every verdict whose node cannot be named \
                     (centralized / proxied endpoints, the `acquire_txn()` shim, and a clustered \
                     client whose routing view is momentarily absent), and a fallback that could \
                     itself be redirected would have nowhere to go"
                );
            }
        }

        // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
        /// Both the bump and the clear must route through
        /// [`ValkeyClient::bucket_key`], and the clear must additionally collect
        /// the shared bucket.
        ///
        /// A source fence rather than a behavioural test because the failure it
        /// guards is *asymmetry between two functions*, and the asymmetry is
        /// invisible on a single-endpoint fixture: there, `probe_node_bucket`
        /// returns `""` for everything, so a `clear` that removed only its
        /// argument still removed the right entry. The b.289 round-1 defect —
        /// `clear` removing the exact key only — passed the whole single-node
        /// suite and latched the breaker permanently on a real cluster whose
        /// routing view was momentarily unavailable. `b_289_tests.rs` pins the
        /// behaviour that a `Clean` collects the shared bucket; this pins that
        /// the two functions cannot drift apart again.
        #[test]
        fn bump_and_clear_both_resolve_through_the_shared_bucket_key_helper() {
            let src: &'static str = include_str!("client.rs");

            // `include_str!` pulls in THIS module too, so every signature below
            // occurs twice in `src`: once at its production definition and once
            // as the string literal a few lines down that this fence searches
            // for. Searching the whole file would let the fence retire itself —
            // rename the production `fn bump_state_proof_streak` and `find`
            // re-anchors on the test's own literal, whose surrounding text
            // contains `Self::bucket_key(` and `streaks.remove("")` (they are
            // spelled in the assertions), so BOTH needles pass while the
            // function they are about no longer exists. Cutting the haystack at
            // the test-module boundary is what makes a rename loud.
            const TEST_MODULE_BOUNDARY: &str = "\n#[cfg(test)]\nmod tests {";
            let boundary = src.find(TEST_MODULE_BOUNDARY).expect(
                "src/storage/client.rs must still open its unit-test module with a top-level \
                 `#[cfg(test)]` + `mod tests {` on consecutive lines — this fence slices the file \
                 there so it cannot match its own string literals. If the module was renamed or \
                 re-indented, update TEST_MODULE_BOUNDARY; do NOT fall back to searching the whole \
                 file, which makes the fence self-satisfying",
            );
            let production = &src[..boundary];

            fn body<'a>(src: &'a str, signature: &str, what: &str) -> &'a str {
                let start = src.find(signature).unwrap_or_else(|| {
                    panic!(
                        "source-inspection anchor {signature:?} is gone from the PRODUCTION half \
                         of src/storage/client.rs (everything above `#[cfg(test)] mod tests`) — \
                         INTENTIONAL loud maintenance signal, not a flake. The signature is \
                         searched there and only there so a rename cannot re-anchor the fence on \
                         this test's own string literals. Maintenance action: re-anchor this \
                         fence on {what}'s moved or renamed definition — update the literal below \
                         — rather than deleting it or widening the search back to the whole file"
                    )
                });
                let rest = &src[start..];
                let end = rest
                    .find("\n    }\n")
                    .expect("every method in this file closes at four-space indentation");
                &rest[..end]
            }

            let bump = body(
                production,
                "fn bump_state_proof_streak(&self, node: &str) -> u32 {",
                "bump_state_proof_streak",
            );
            let clear = body(
                production,
                "fn clear_state_proof_streak(&self, node: &str) {",
                "clear_state_proof_streak",
            );

            for (what, source) in [("bump", bump), ("clear", clear)] {
                assert!(
                    source.contains("Self::bucket_key("),
                    "{what}_state_proof_streak must resolve its map key through \
                     `Self::bucket_key(..)`. The `\"\"`-folding rule has to be applied \
                     identically by both, or a verdict folded into the shared bucket by one is \
                     looked up under a different key by the other and its streak is never \
                     collected — which latches the repair-ineffective breaker for the life of \
                     the process. Body read: {source:?}"
                );
            }

            assert!(
                clear.contains(r#"streaks.remove("")"#),
                "clear_state_proof_streak must ALSO remove the shared `\"\"` bucket \
                 unconditionally. That bucket holds every verdict whose node could not be named, \
                 so there is no `Clean` \"against\" it to pair with — clearing it on any Clean is \
                 what makes the fallback degenerate to the pre-b.289 pool-wide contract its \
                 rustdoc promises, and leaving it out is precisely the b.289 round-1 defect. \
                 Body read: {clear:?}"
            );
        }
    }

    mod txn_slot_hook_registry {
        use std::sync::atomic::AtomicBool;

        use super::*;
        use crate::storage::test_support::TestValkey;
        use crate::storage::txn_checkout::{ProbeFault, ProbeFaultInjector};

        /// How many consecutive dirty checkouts the registry must survive
        /// unchanged. Three, not one: the defect this pins is *accumulation*,
        /// and a single recycle cannot distinguish "replaced the entry" from
        /// "added one more".
        const RECYCLE_ROUNDS: usize = 3;

        /// Ceiling on the "the aborted sentinel task actually unwound" wait.
        ///
        /// `JoinHandle::abort` schedules the cancellation; the future is dropped
        /// when the runtime next polls that task, so the observation needs a
        /// yield, not a delay. A healthy run satisfies it on the first probe —
        /// this is a hang ceiling, well clear of nextest's 120 s hard kill.
        const ABORT_OBSERVATION_BUDGET: Duration = Duration::from_secs(5);

        /// Probe granularity for [`ABORT_OBSERVATION_BUDGET`].
        const ABORT_PROBE_INTERVAL: Duration = Duration::from_millis(5);

        /// Sets its flag when dropped. Held by a sentinel task that otherwise
        /// never completes, so the flag flips **only** if something aborted the
        /// task — which is the whole assertion.
        struct DropWitness(Arc<AtomicBool>);

        impl Drop for DropWitness {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        /// Spawn a task that never finishes on its own and report the flag its
        /// cancellation would set.
        ///
        /// This models the orphan case exactly: on the `QUIT`-timeout path
        /// fred's `quit()` never reaches the sender swap, so the previous hook
        /// tasks never observe `RecvError::Closed` and never finish. A re-arm
        /// that merely pruned finished handles could not collect them — only an
        /// abort can, and only an abort flips this flag.
        fn spawn_immortal_sentinel() -> (JoinHandle<()>, Arc<AtomicBool>) {
            let flag = Arc::new(AtomicBool::new(false));
            let witness = DropWitness(Arc::clone(&flag));
            let handle = tokio::spawn(async move {
                let _witness = witness;
                std::future::pending::<()>().await;
            });
            (handle, flag)
        }

        /// A recycle must **replace** the slot's hook pair, not accumulate one.
        ///
        /// `rearm_pool_hooks` aborts the registered pair before inserting the
        /// new one, and the abort — not a `retain(!is_finished())` prune — is
        /// what makes it correct. `quit()` swaps fred's broadcast senders only
        /// *after* its own internal await completes, so on the expected
        /// `probe_timeout` path (a connection that will not answer a 650 ms
        /// `DISCARD` will not answer a `QUIT` in 2 s either) the
        /// [`TXN_RECYCLE_QUIT_TIMEOUT`] fires first, the senders are never
        /// swapped, and the old hook tasks run forever. Re-arming blind would
        /// then leave two live pairs on one channel — permanently
        /// double-counting `valkey_errors_total{pool="txn"}` /
        /// `valkey_reconnects_total{pool="txn"}` and the b.cbe hint, with
        /// orphans no prune can ever reach.
        ///
        /// The test drives that state directly rather than waiting for a real
        /// `QUIT` timeout: before each checkout it swaps the registered pair for
        /// two tasks that provably never finish, so "the previous pair was
        /// aborted" is observable as a flag rather than inferred from a handle
        /// that would have finished on its own anyway. A `quit()` against a
        /// healthy local Valkey always completes, so without the sentinels this
        /// test could not tell abort-before-respawn from blind respawn at all.
        ///
        /// `pool_size = 1` throughout: one txn slot, so one registry entry, and
        /// the id the fixture reads is the id every checkout re-arms under.
        #[tokio::test]
        async fn a_recycle_replaces_the_slots_hook_pair_rather_than_accumulating_one() {
            let tv = TestValkey::setup_with_pool_size(1).await;
            let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
                "txn-hook-registry-test",
            ));
            tv.client()
                .install_observability_hooks(Arc::clone(&metrics))
                .await;
            let key = tv.key_schema().obj_header("7td-hook-registry");

            let injector = Arc::new(ProbeFaultInjector::new());
            tv.client()
                .install_probe_fault_injector_for_test(Arc::clone(&injector));

            // The registry key: fred's own client id, stable across `quit()` +
            // `init()` because it is minted in `Client::new` and shared by every
            // clone.
            let slot_id = {
                let checkout = tv.client().acquire_txn_for(&key).await;
                ClientLike::id(&*checkout).to_string()
            };
            assert!(
                registry_snapshot(tv.client()).contains(&slot_id),
                "precondition: `install_observability_hooks` must have registered a hook pair for \
                 the txn slot under its fred client id ({slot_id}). If it did not, every \
                 assertion below is about an entry production never writes"
            );

            for round in 1..=RECYCLE_ROUNDS {
                // Swap the registered pair for two tasks that never finish on
                // their own, keeping the real ones from piling up.
                let (first, first_flag) = spawn_immortal_sentinel();
                let (second, second_flag) = spawn_immortal_sentinel();
                let displaced = tv
                    .client()
                    .pool_hook_handles
                    .lock()
                    .expect("the registry mutex is never poisoned in this test")
                    .insert(slot_id.clone(), [first, second]);
                for handle in displaced.into_iter().flatten() {
                    handle.abort();
                }

                // A `+OK` from `DISCARD` is the state-proof `leaked_multi`
                // verdict, so this checkout recycles unconditionally — no
                // streak gate, no suppression window to reason about.
                injector.push_discard(ProbeFault::ok());
                {
                    let _checkout = tv.client().acquire_txn_for(&key).await;
                }
                assert_eq!(
                    injector.remaining(),
                    0,
                    "round {round}: the queued DISCARD fault must have been consumed by the \
                     checkout — if it was not, no dirty verdict was reached and this round \
                     recycled nothing"
                );

                let aborted = poll_until(ABORT_OBSERVATION_BUDGET, || {
                    first_flag.load(Ordering::SeqCst) && second_flag.load(Ordering::SeqCst)
                })
                .await;
                assert!(
                    aborted,
                    "round {round}: the recycle must ABORT the hook pair it found registered for \
                     this slot before arming the replacement. These two sentinel tasks never \
                     finish on their own — exactly like the real hook tasks do not when `quit()` \
                     times out before swapping fred's broadcast senders — so a re-arm that \
                     pruned finished handles instead would leave them live here and, in \
                     production, leave two hook pairs on one channel double-counting \
                     valkey_errors_total{{pool=\"txn\"}} forever"
                );

                let registry = registry_snapshot(tv.client());
                assert_eq!(
                    registry,
                    vec![slot_id.clone()],
                    "round {round}: the registry must hold exactly one entry — this slot's. It is \
                     keyed by fred's client id, which survives quit + init, so a growing map \
                     means recycles are accumulating entries no shutdown path can attribute"
                );
                assert_eq!(
                    live_handles_for(tv.client(), &slot_id),
                    2,
                    "round {round}: exactly two live handles must be armed on the slot after the \
                     recycle — the reconnect hook and the error hook. Fewer means the recycle \
                     blinded this connection's health signal; more is not expressible in the \
                     registry's `[JoinHandle; 2]` and would mean an orphan escaped it"
                );
            }
        }

        /// Every client id currently carrying a registered hook pair, sorted,
        /// less the `data`, `maintenance` and `snapshot` members' (registered
        /// at install too, so a replacement can re-arm them; none is replaced
        /// in this test).
        fn registry_snapshot(client: &ValkeyClient) -> Vec<String> {
            let members: Vec<String> = [
                &client.pool,
                &client.maintenance_pool,
                &client.snapshot_pool,
            ]
            .into_iter()
            .flat_map(|pool| pool.clients())
            .map(|member| ClientLike::id(&member).to_owned())
            .collect();
            let registry = client
                .pool_hook_handles
                .lock()
                .expect("the registry mutex is never poisoned in this test");
            let mut ids: Vec<String> = registry
                .keys()
                .filter(|id| !members.contains(id))
                .cloned()
                .collect();
            ids.sort();
            ids
        }

        /// How many of `slot_id`'s registered handles are still running.
        fn live_handles_for(client: &ValkeyClient, slot_id: &str) -> usize {
            let registry = client
                .pool_hook_handles
                .lock()
                .expect("the registry mutex is never poisoned in this test");
            registry
                .get(slot_id)
                .map(|handles| handles.iter().filter(|h| !h.is_finished()).count())
                .unwrap_or(0)
        }

        /// Bounded probe loop — the test-writing guide's "wait for a background
        /// condition" shape.
        async fn poll_until(budget: Duration, mut predicate: impl FnMut() -> bool) -> bool {
            let started = std::time::Instant::now();
            loop {
                if predicate() {
                    return true;
                }
                if started.elapsed() >= budget {
                    return false;
                }
                tokio::time::sleep(ABORT_PROBE_INTERVAL).await;
            }
        }
    }

    mod auth_rejection_diagnostics {
        use std::io::Write;

        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;
        use fred::types::config::CredentialProvider;
        use tempfile::tempdir;
        use tracing_subscriber::fmt::MakeWriter;

        use super::*;

        /// The identity's **client id** — the value an operator wrongly puts in
        /// `LIVE_EDIT_VALKEY_TOKEN_USERNAME` in the cause-(1) case. Distinct
        /// from the shared `OBJECT_ID` fixture, which is what the variable is
        /// supposed to carry.
        const CLIENT_ID: &str = "99999999-8888-7777-6666-555555555555";
        const TENANT_ID: &str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
        const ISSUER: &str = "https://login.example.com/tenant-fixture/v2.0";
        const AUDIENCE: &str = "https://redis.azure.com";

        /// A fixed JWT `exp` (NumericDate) and its RFC 3339 rendering. Pinned
        /// as a pair so the "both renderings" contract is asserted against a
        /// value, not against whatever the formatter happens to produce.
        const EXP_NUMERIC_DATE: i64 = 1_700_000_000;
        const EXP_RFC3339: &str = "2023-11-14T22:13:20+00:00";

        /// Distinctive marker of the rejection hint's opening sentence, shared
        /// by all three mode arms. Used to COUNT emissions for the rate-limit
        /// tests; nothing asserts the sentence itself.
        const HINT_MARKER: &str = "Valkey rejected the credential this service sent";

        /// Distinctive marker of the healthy-connect INFO.
        const HEALTHY_MARKER: &str = "Valkey accepted the accessTokenFile credential";

        /// Ceiling for the "polling task has loaded the fixture" waits. A
        /// bounded probe on observable state per the test-writing guide's
        /// § Synchronization Patterns; a healthy run exits in milliseconds and
        /// only a genuinely stuck read spends the budget.
        const TOKEN_LOAD_BUDGET: Duration = Duration::from_secs(5);

        // ── Local harness ───────────────────────────────────────────────────

        /// Captures every byte written to it. Local copy of the
        /// `CapturingWriter` in `auth::service_token::tests` /
        /// `storage::token_file_credentials::tests`, which live in private
        /// `#[cfg(test)]` blocks and are not reachable cross-module.
        #[derive(Clone)]
        struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

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

        /// Install a thread-local capturing subscriber; drop the guard before
        /// reading the buffer. ANSI off so structured `field=value` pairs stay
        /// matchable by a plain `contains`.
        fn capture_logs() -> (tracing::subscriber::DefaultGuard, Arc<Mutex<Vec<u8>>>) {
            let buf = Arc::new(Mutex::new(Vec::new()));
            let subscriber = tracing_subscriber::fmt()
                .with_writer(CapturingWriter(buf.clone()))
                .with_ansi(false)
                .with_max_level(tracing::Level::TRACE)
                .finish();
            (tracing::subscriber::set_default(subscriber), buf)
        }

        fn captured_text(buf: &Arc<Mutex<Vec<u8>>>) -> String {
            String::from_utf8_lossy(&buf.lock().unwrap()).into_owned()
        }

        /// Assert no contiguous 8-byte slice of `canary` appears in `output` —
        /// the bearer-bytes-never-logged invariant (SR-6.2), at the same
        /// granularity the sibling auth modules use.
        fn assert_no_secret_bytes(output: &str, canary: &str, what: &str) {
            let bytes = canary.as_bytes();
            assert!(
                bytes.len() >= 8,
                "canary must be at least 8 bytes wide for the slice search to be meaningful"
            );
            for window in bytes.windows(8) {
                let slice = std::str::from_utf8(window).expect("ASCII canary");
                assert!(
                    !output.contains(slice),
                    "rendered output must not contain any 8-byte slice of the {what} \
                     (slice={slice:?}, canary={canary:?}, output={output})"
                );
            }
        }

        /// Build a JWT-shaped token carrying `claims`. Same construction as
        /// `auth::service_token::tests::make_jwt`, which is what the production
        /// decoder under test was written against.
        fn make_jwt(claims: &serde_json::Value) -> String {
            let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
            let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims).unwrap());
            format!("{header}.{payload}.")
        }

        /// The standard fixture token: every claim the descriptor renders,
        /// with `oid` = `OBJECT_ID` and the client-id claim = [`CLIENT_ID`].
        fn full_claims_jwt() -> String {
            make_jwt(&serde_json::json!({
                "aud": AUDIENCE,
                "oid": OBJECT_ID,
                "appid": CLIENT_ID,
                "tid": TENANT_ID,
                "iss": ISSUER,
                "exp": EXP_NUMERIC_DATE,
            }))
        }

        async fn write_token_file(path: &std::path::Path, token: &str, expires_in_secs: i64) {
            let expires_on = chrono::Utc::now() + chrono::Duration::seconds(expires_in_secs);
            tokio::fs::write(
                path,
                format!(
                    r#"{{"access_token":"{}","expires_on":"{}"}}"#,
                    token,
                    expires_on.to_rfc3339()
                ),
            )
            .await
            .expect("write fixture");
        }

        /// A `ServiceTokenProvider` whose cache already holds `token`.
        ///
        /// Waiting on the bearer accessor rather than sleeping a poll interval:
        /// the claims are written into the cache in the same guarded write as
        /// the token, so a loaded bearer is proof the projection has landed
        /// too, and a stalled read extends the wait instead of failing a
        /// diagnostic assertion with a setup problem in disguise. The probe
        /// itself is the shared [`await_token_cached`] helper.
        async fn provider_holding(
            dir: &std::path::Path,
            token: &str,
        ) -> (Arc<ServiceTokenProvider>, CancellationToken) {
            let path = dir.join("token.json");
            write_token_file(&path, token, 3600).await;
            let (provider, cancel) = redis_token_provider(path, "auth-diagnostics-test");
            await_token_cached(
                &provider,
                TOKEN_LOAD_BUDGET,
                "every claims assertion below needs a populated cache",
            )
            .await;
            (provider, cancel)
        }

        /// `accessTokenFile` diagnostics whose configured username is
        /// `username` and whose provider holds `token`.
        async fn access_token_file_diagnostics(
            dir: &std::path::Path,
            username: &str,
            token: &str,
        ) -> (ValkeyAuthDiagnostics, CancellationToken) {
            let (provider, cancel) = provider_holding(dir, token).await;
            let mut config = token_file_test_config(PLAINTEXT_URL);
            config.valkey_token_username = Some(username.to_string());
            (ValkeyAuthDiagnostics::new(&config, Some(provider)), cancel)
        }

        fn mk_auth_err(details: &'static str) -> fred::error::Error {
            fred::error::Error::new(FredErrorKind::Auth, details)
        }

        // ── is_server_auth_rejection ────────────────────────────────────────

        #[test]
        fn is_server_auth_rejection_wrongpass_auth_error_is_true() {
            // Verbatim the string a managed backing store returns.
            assert!(is_server_auth_rejection(&mk_auth_err(
                "WRONGPASS invalid username-password pair or user is disabled."
            )));
        }

        #[test]
        fn is_server_auth_rejection_noauth_auth_error_is_true() {
            assert!(is_server_auth_rejection(&mk_auth_err(
                "NOAUTH Authentication required."
            )));
        }

        #[tokio::test]
        async fn is_server_auth_rejection_token_file_fetch_failure_is_false() {
            // THE load-bearing case, and the reason the predicate is not a bare
            // `matches!(kind, Auth)`. This service raises `ErrorKind::Auth`
            // itself when it has no credential to send — the OPPOSITE fault,
            // already self-diagnosing — and firing the four-cause access-policy
            // checklist on it sends an operator hunting a policy assignment
            // when the real problem is a sidecar that stopped writing the file.
            //
            // Driven through the REAL `fetch()` rather than a hand-copied
            // message, so a reworded error cannot silently start matching.
            let dir = tempdir().expect("tempdir");
            let (provider, cancel) =
                redis_token_provider(dir.path().join("absent.json"), "fetch-failure-classifier");
            let credential = TokenFileCredentialProvider::new(
                provider,
                OBJECT_ID.to_string(),
                VALKEY_TOKEN_REAUTH_BASE_INTERVAL,
                VALKEY_TOKEN_REAUTH_JITTER_BOUND,
                VALKEY_TOKEN_AUTH_FANOUT_WINDOW,
            );

            let err = credential
                .fetch(None)
                .await
                .expect_err("no token file means no credential");

            assert_eq!(
                *err.kind(),
                FredErrorKind::Auth,
                "precondition: this error IS ErrorKind::Auth, which is exactly why a kind-only \
                 classifier would misfire on it"
            );
            assert!(
                !is_server_auth_rejection(&err),
                "the service's own token-unavailable error must never be classified as a server \
                 rejection: {err:?}"
            );

            cancel.cancel();
        }

        #[test]
        fn is_server_auth_rejection_non_auth_kind_with_wrongpass_details_is_false() {
            // The kind gate half of the predicate: the prefix alone must not
            // promote an unrelated error into an auth diagnosis.
            assert!(!is_server_auth_rejection(&fred::error::Error::new(
                FredErrorKind::Unknown,
                "WRONGPASS invalid username-password pair"
            )));
        }

        #[test]
        fn is_server_auth_rejection_token_not_at_the_start_is_false() {
            // The anchor half: `starts_with`, never a substring search. A
            // server token only counts where the server puts it.
            assert!(!is_server_auth_rejection(&mk_auth_err(
                "connection closed after WRONGPASS was observed upstream"
            )));
            assert!(!is_server_auth_rejection(&mk_auth_err(
                "retrying after NOAUTH from a peer"
            )));
        }

        // ── render_hint: accessTokenFile arm ────────────────────────────────

        #[tokio::test]
        async fn render_hint_username_equal_to_oid_rules_out_two_causes_and_names_the_policy() {
            // The healthy-identity path. Its whole value is the DERIVED claim:
            // the token is correctly minted and correctly bound, so a rejection
            // must be an access-policy problem — and cause (4) names the Object
            // ID an operator can paste into a cloud console.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;

            let hint = diag.render_hint();

            assert_eq!(hint.verdict_label, "matches-oid");
            assert_eq!(
                hint.message.matches("RULED OUT").count(),
                2,
                "EXACTLY causes (1) and (3) may be ruled out — one occurrence means a cause \
                 silently stopped being narrowed, three means a cause this self-check cannot \
                 decide (the audience, or the access-policy assignment on the backing store's \
                 side) is being dismissed on no evidence: {}",
                hint.message
            );
            assert!(
                hint.message.contains("MOST LIKELY REMAINING CAUSE"),
                "the derived narrowing to cause (4) is the point of this arm: {}",
                hint.message
            );
            assert!(
                hint.message.contains(&format!("object ID `{OBJECT_ID}`")),
                "cause (4) must name the policy subject concretely: {}",
                hint.message
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_username_equal_to_client_id_declares_it_the_cause() {
            // The exact-cause path: the one verdict where this service can say
            // "this is the cause" rather than narrowing.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), CLIENT_ID, &full_claims_jwt()).await;

            let hint = diag.render_hint();

            assert_eq!(hint.verdict_label, "matches-client-id-claim");
            assert!(
                hint.message.contains("THIS IS THE CAUSE"),
                "cause (1) must be flagged unambiguously: {}",
                hint.message
            );
            assert!(
                hint.message.contains(OBJECT_ID),
                "the `oid` to substitute must be printed: {}",
                hint.message
            );
            assert!(
                !hint.message.contains("MOST LIKELY REMAINING CAUSE"),
                "cause (4) must NOT also be advertised once (1) is established — two 'most \
                 likely' causes is worse than none: {}",
                hint.message
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_username_matching_neither_claim_points_at_the_federation_binding() {
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) = access_token_file_diagnostics(
                dir.path(),
                "00000000-0000-0000-0000-000000000000",
                &full_claims_jwt(),
            )
            .await;

            let hint = diag.render_hint();

            assert_eq!(hint.verdict_label, "matches-neither");
            assert!(
                hint.message.contains("THIS IS THE LIKELY CAUSE"),
                "cause (3) — federation misbinding — must be the flagged one: {}",
                hint.message
            );
            assert!(
                hint.message.contains("workload-identity binding"),
                "the usual concrete reason must be named: {}",
                hint.message
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_token_without_identity_claims_narrows_nothing() {
            // A decodable JWT that carries no identity claim: the self-check is
            // unanswerable, so every cause must stay open. Claiming otherwise
            // would be a diagnosis this service did not earn.
            let dir = tempdir().expect("tempdir");
            let jwt = make_jwt(&serde_json::json!({ "aud": AUDIENCE, "iss": ISSUER }));
            let (diag, cancel) = access_token_file_diagnostics(dir.path(), OBJECT_ID, &jwt).await;

            let hint = diag.render_hint();

            assert_eq!(hint.verdict_label, "no-identity-claim");
            assert_eq!(
                hint.message.matches("NOT CHECKABLE").count(),
                2,
                "EXACTLY the two causes the self-check would otherwise decide — (1) and (3) — \
                 must be reported as uncheckable; fewer means one silently claims a verdict it \
                 did not earn: {}",
                hint.message
            );
            assert!(
                hint.message.contains("NOT NARROWED"),
                "cause (4) must say the checklist could not be narrowed at all: {}",
                hint.message
            );
            assert!(
                !hint.message.contains("RULED OUT"),
                "nothing may be ruled out from an unanswerable self-check: {}",
                hint.message
            );
            assert_eq!(
                hint.claims.aud, AUDIENCE,
                "the reported-but-unjudged claims must still be printed"
            );
            assert_eq!(hint.claims.oid, "<absent>");

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_opaque_cached_token_degrades_to_the_mode_level_advice() {
            // An opaque token is a legitimate issuer-agnostic deployment, so
            // the hint must degrade rather than accuse. `<n/a>` (nothing to
            // read) is deliberately distinct from `<absent>` (read, not there).
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, "an-opaque-sidecar-token")
                    .await;

            let hint = diag.render_hint();

            assert_eq!(hint.verdict_label, "not-a-jwt");
            assert_eq!(
                hint.claims,
                ClaimsDescriptor::not_applicable(),
                "no readable token means every claim field renders as not-applicable"
            );
            assert!(
                hint.message.contains("not a decodable JWT"),
                "the hint must say why no self-check is available: {}",
                hint.message
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_with_no_cached_token_still_renders_the_checklist() {
            // Nothing cached at all — the shape a rejection observed before the
            // sidecar's first write takes. The hint must still be emitted: an
            // operator staring at an opaque WRONGPASS gets the four causes even
            // when the identity half cannot be evaluated.
            let dir = tempdir().expect("tempdir");
            let (provider, cancel) =
                redis_token_provider(dir.path().join("absent.json"), "no-cached-token-hint");
            let diag =
                ValkeyAuthDiagnostics::new(&token_file_test_config(PLAINTEXT_URL), Some(provider));

            let hint = diag.render_hint();

            assert_eq!(hint.verdict_label, "not-a-jwt");
            assert!(
                hint.message.contains("Four known causes:"),
                "the checklist is the floor of what this hint owes an operator: {}",
                hint.message
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_renders_exp_as_numeric_date_and_rfc3339() {
            // Both renderings, because they answer different questions: the raw
            // NumericDate is what an operator greps the sidecar's own logs for,
            // the RFC 3339 form tells them at a glance whether the token is
            // simply stale.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;

            let hint = diag.render_hint();

            assert_eq!(
                hint.claims.exp,
                format!("{EXP_NUMERIC_DATE} ({EXP_RFC3339})")
            );
            assert!(
                hint.message.contains(&format!("exp={EXP_NUMERIC_DATE}")),
                "the claims trailer must carry the exp: {}",
                hint.message
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_reports_every_named_claim_in_the_trailer() {
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;

            let hint = diag.render_hint();

            assert_eq!(
                hint.claims,
                ClaimsDescriptor {
                    aud: AUDIENCE.to_string(),
                    oid: OBJECT_ID.to_string(),
                    client_id_claim: CLIENT_ID.to_string(),
                    tid: TENANT_ID.to_string(),
                    iss: ISSUER.to_string(),
                    exp: format!("{EXP_NUMERIC_DATE} ({EXP_RFC3339})"),
                }
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_token_without_an_audience_renders_the_absent_marker() {
            // Cause (2) prints the audience inline ("The token's `aud` is {aud}"),
            // so a token carrying none must render `<absent>` — the marker that
            // says "we looked and it is not there". An empty string would read
            // as "The token's `aud` is ." and an operator would not know
            // whether the claim was missing or the diagnostic was broken.
            //
            // `<absent>` is also deliberately distinct from the `<n/a>` the
            // opaque-token / static / iam arms print, which means "there was
            // nothing to read at all" — the sibling
            // `render_hint_opaque_cached_token_degrades_to_the_mode_level_advice`
            // pins that half.
            let dir = tempdir().expect("tempdir");
            let jwt = make_jwt(&serde_json::json!({ "oid": OBJECT_ID }));
            let (diag, cancel) = access_token_file_diagnostics(dir.path(), OBJECT_ID, &jwt).await;

            let hint = diag.render_hint();

            assert_eq!(hint.claims.aud, "<absent>");
            assert!(
                hint.message.contains("The token's `aud` is <absent>."),
                "cause (2) must print the absent marker inline rather than a blank: {}",
                hint.message
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_out_of_range_exp_falls_back_to_bare_seconds() {
            // The `exp` renderer's failure arm. `chrono::DateTime::from_timestamp`
            // returns `None` outside its representable range, and the fallback
            // prints the raw NumericDate alone. That is the right degrade: the
            // raw value is what an operator greps the credential sidecar's own
            // logs for, so a nonsense `exp` must still be REPORTED rather than
            // rendered as absent — "the token says it expires at
            // 9223372036854775807" is itself the finding.
            let dir = tempdir().expect("tempdir");
            let jwt = make_jwt(&serde_json::json!({
                "oid": OBJECT_ID,
                "exp": i64::MAX,
            }));
            let (diag, cancel) = access_token_file_diagnostics(dir.path(), OBJECT_ID, &jwt).await;

            let hint = diag.render_hint();

            assert_eq!(
                hint.claims.exp,
                i64::MAX.to_string(),
                "an unrepresentable NumericDate must render as bare seconds — not as `<absent>`, \
                 which would hide the very anomaly worth reporting"
            );
            assert!(
                !hint.claims.exp.contains('('),
                "the RFC 3339 parenthetical must be omitted rather than faked: {}",
                hint.claims.exp
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn render_hint_multi_audience_token_lists_every_audience() {
            // Cause (2) is "wrong token audience/scope", and it is REPORTED,
            // never judged — so a multi-audience token must show all of them
            // rather than an arbitrary first.
            let dir = tempdir().expect("tempdir");
            let jwt = make_jwt(&serde_json::json!({
                "aud": [AUDIENCE, "api://live-edit"],
                "oid": OBJECT_ID,
            }));
            let (diag, cancel) = access_token_file_diagnostics(dir.path(), OBJECT_ID, &jwt).await;

            let hint = diag.render_hint();

            assert_eq!(hint.claims.aud, format!("{AUDIENCE}, api://live-edit"));
            assert!(
                hint.message.contains("no expected-`aud` allowlist"),
                "the mode is issuer-agnostic: the audience is reported, not judged: {}",
                hint.message
            );

            cancel.cancel();
        }

        // ── render_hint: iam / static arms ──────────────────────────────────

        #[test]
        fn render_hint_iam_mode_names_every_signing_input_and_no_claims() {
            // Deliberately NOT the four-cause checklist: those causes are
            // properties of an OAuth bearer token and a data-plane policy
            // assignment, none of which transfers to a SigV4 credential.
            let diag = ValkeyAuthDiagnostics::new(&iam_test_config(PLAINTEXT_URL), None);

            let hint = diag.render_hint();

            assert_eq!(hint.verdict_label, "n/a");
            assert_eq!(hint.claims, ClaimsDescriptor::not_applicable());
            for var in [
                "LIVE_EDIT_VALKEY_IAM_USERNAME",
                "LIVE_EDIT_VALKEY_IAM_CLUSTER_HOST",
                "LIVE_EDIT_VALKEY_IAM_REGION",
                "LIVE_EDIT_VALKEY_IAM_SERVICE",
            ] {
                assert!(
                    hint.message.contains(var),
                    "the iam arm must name {var}, the manual check it substitutes for a \
                     claims-level one: {}",
                    hint.message
                );
            }
            assert!(
                !hint.message.contains("Four known causes:"),
                "the accessTokenFile cause list must not be printed under iam — an invented \
                 cause list carries the authority of a diagnosis: {}",
                hint.message
            );
        }

        #[test]
        fn render_hint_static_mode_names_the_url_variable_without_echoing_the_url() {
            // The static credential rides `LIVE_EDIT_VALKEY_URL` inline, so the
            // one thing this arm must never do is quote it. The canary is the
            // password embedded in the fixture URL.
            // Opaque on purpose: a canary containing an English word ("password",
            // "username") would collide with the hint's own prose and fail this
            // assertion for a reason that has nothing to do with a leak.
            const URL_PASSWORD: &str = "hntr2StaticUrlCanary0f1e2d3c4b5a69";
            let mut config = auth_mode_test_config(
                ValkeyAuthMode::Static,
                &format!("redis://admin:{URL_PASSWORD}@valkey.example.com:6379"),
            );
            config.valkey_token_username = None;

            let hint = ValkeyAuthDiagnostics::new(&config, None).render_hint();

            assert_eq!(hint.verdict_label, "n/a");
            assert_eq!(hint.claims, ClaimsDescriptor::not_applicable());
            assert!(
                hint.message.contains("LIVE_EDIT_VALKEY_URL"),
                "the static arm must name where the credential lives: {}",
                hint.message
            );
            assert_no_secret_bytes(&hint.message, URL_PASSWORD, "static URL password");
        }

        // ── SR-6.2: no secret bytes on any new emission surface ─────────────

        #[tokio::test]
        async fn render_hint_never_emits_token_bytes_in_any_verdict() {
            // The token's *claims* are non-secret and are printed on purpose;
            // the token's WIRE FORM is the secret. So the canary is the whole
            // JWT string, and the assertion sweeps every verdict arm, because
            // each one formats a different subset of the same source data.
            let dir = tempdir().expect("tempdir");
            let jwt = full_claims_jwt();
            for username in [OBJECT_ID, CLIENT_ID, "an-unrelated-username"] {
                let sub = dir.path().join(username);
                std::fs::create_dir_all(&sub).expect("subdir");
                let (diag, cancel) = access_token_file_diagnostics(&sub, username, &jwt).await;

                let hint = diag.render_hint();

                assert_no_secret_bytes(&hint.message, &jwt, "access token");
                assert_no_secret_bytes(&hint.claims.oid, &jwt, "access token");
                assert_no_secret_bytes(&hint.claims.client_id_claim, &jwt, "access token");
                cancel.cancel();
            }
        }

        #[tokio::test]
        async fn emitted_hints_carry_neither_token_bytes_nor_the_configured_username() {
            // Sweeps the three EMISSION sites (`emit_connect_hint`,
            // `emit_steady_state_hint`, `log_healthy_connect`) rather than the
            // renderer, because a structured tracing field is a second way for
            // a value to reach a log line and the renderer's return value does
            // not cover it.
            //
            // Two canaries. The token's wire form must never appear at all. The
            // configured username must not either: every statement about it is
            // made RELATIONALLY, matching the redacting `Debug` on
            // `TokenFileCredentialProvider` — so this uses a `MatchesNeither`
            // fixture, the one verdict where the username equals no printed
            // claim and a leak would therefore be visible.
            // Opaque for the same reason as the static-URL canary above: an
            // English substring would collide with the hint's own prose.
            const USERNAME_CANARY: &str = "cfgUsrLeakCanary0f1e2d3c4b5a69788796";
            let dir = tempdir().expect("tempdir");
            let jwt = full_claims_jwt();
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), USERNAME_CANARY, &jwt).await;

            let (guard, captured) = capture_logs();
            diag.emit_connect_hint("data pool", &AtomicBool::new(false));
            diag.emit_steady_state_hint("data");
            diag.log_healthy_connect();
            drop(guard);
            let output = captured_text(&captured);

            assert!(
                output.contains(HINT_MARKER) && output.contains(HEALTHY_MARKER),
                "precondition: all three emission sites must have produced output, or the \
                 canary sweep below passes vacuously; got: {output}"
            );
            assert_no_secret_bytes(&output, &jwt, "access token");
            assert_no_secret_bytes(&output, USERNAME_CANARY, "configured AUTH username");

            cancel.cancel();
        }

        #[tokio::test]
        async fn diagnostics_debug_redacts_the_username_and_the_token_handle() {
            // `ValkeyClient`'s own `Debug` reaches this type, so a `{:?}` dump
            // anywhere in the service would otherwise carry the AUTH username.
            const USERNAME_CANARY: &str = "dbgUsrLeakCanary0f1e2d3c4b5a69788796";
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), USERNAME_CANARY, &full_claims_jwt())
                    .await;

            let rendered = format!("{diag:?}");

            assert!(
                !rendered.contains(USERNAME_CANARY),
                "Debug must not print the configured AUTH username: {rendered}"
            );
            assert!(
                rendered.contains("<redacted>") && rendered.contains("AccessTokenFile"),
                "Debug must still identify the mode it is diagnosing: {rendered}"
            );

            cancel.cancel();
        }

        // ── Rate limiting ───────────────────────────────────────────────────

        #[tokio::test]
        async fn connect_hint_is_emitted_once_per_connect_across_every_pool() {
            // `connect()` drives five retry loops sharing ONE flag, so a
            // rejected credential must produce one hint per invocation, not one
            // per pool per attempt (five loops × a 30 s budget of 200 ms-plus
            // retries is a log flood, and the message is ~20 lines).
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;
            let flag = AtomicBool::new(false);

            let (guard, captured) = capture_logs();
            for pool in ["data pool", "txn pool", "maintenance pool", "data pool"] {
                diag.emit_connect_hint(pool, &flag);
            }
            drop(guard);

            assert_eq!(
                captured_text(&captured).matches(HINT_MARKER).count(),
                1,
                "one connect() invocation must yield exactly one hint however many pools and \
                 attempts observe the rejection"
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn connect_hint_is_emitted_again_for_a_fresh_connect_invocation() {
            // The flag is per-`connect()` and deliberately not a process-global
            // `Once`: a later reconnect attempt on a still-broken credential
            // must speak up again, and a global gate would also make this
            // untestable (three integration binaries share a process).
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;

            let (guard, captured) = capture_logs();
            diag.emit_connect_hint("data pool", &AtomicBool::new(false));
            diag.emit_connect_hint("data pool", &AtomicBool::new(false));
            drop(guard);

            assert_eq!(
                captured_text(&captured).matches(HINT_MARKER).count(),
                2,
                "a second connect() invocation owns a fresh flag and must emit its own hint"
            );

            cancel.cancel();
        }

        /// Retry budget for the `connect_with_retry` tests below, which run on
        /// a paused clock (`start_paused`).
        ///
        /// `connect_with_retry` measures its budget, sleeps its backoff and
        /// bounds each attempt on tokio's clock, so a paused runtime drives the
        /// whole loop in virtual time: no test here waits in real time, and no
        /// runner stall can change how many attempts a call makes.
        ///
        /// Derivation. Every attempt is rejected at once, and attempts fire at
        /// t = 0, 200, 600 ms (backoff 200 → 400 → 800, each clamped to the
        /// remaining budget); at t = 1 s the budget is spent. So a call makes
        /// exactly **three** attempts, and the two-call test asserts exactly
        /// six.
        const RETRY_BUDGET: Duration = Duration::from_millis(1000);

        #[tokio::test(start_paused = true)]
        async fn connect_with_retry_emits_one_hint_across_pools_and_attempts() {
            // The real integration point, not the emission helper in isolation:
            // `connect_with_retry` classifies the failure and drives the gate.
            // Two calls stand in for two of `connect()`'s five retry loops
            // sharing the one caller-owned flag.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;
            let flag = AtomicBool::new(false);
            let attempts = std::sync::atomic::AtomicUsize::new(0);
            let reject = || async {
                attempts.fetch_add(1, Ordering::Relaxed);
                Err(fred::error::Error::new(
                    FredErrorKind::Auth,
                    "WRONGPASS invalid username-password pair",
                ))
            };

            let (guard, captured) = capture_logs();
            for label in ["data pool", "txn pool"] {
                let outcome =
                    connect_with_retry(label, RETRY_BUDGET, Some(&diag), &flag, reject).await;
                assert!(
                    matches!(outcome, Err(ConnectError::Timeout { .. })),
                    "a credential the server keeps refusing must still exhaust the budget and \
                     fail the connect: {outcome:?}"
                );
            }
            drop(guard);

            assert_eq!(
                attempts.load(Ordering::Relaxed),
                6,
                "precondition: three rejections per call on the paused clock — more rejections \
                 than pools, or 'one hint across attempts' is vacuous"
            );
            assert_eq!(
                captured_text(&captured).matches(HINT_MARKER).count(),
                1,
                "one connect() invocation emits exactly one hint however many pools and \
                 attempts observe the rejection"
            );

            cancel.cancel();
        }

        #[tokio::test(start_paused = true)]
        async fn connect_with_retry_stays_quiet_for_a_non_auth_failure() {
            // The ordinary case — Valkey not up yet — must not print a
            // twenty-line credential checklist. This is the classifier's gate
            // observed at its real call site.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;
            let flag = AtomicBool::new(false);

            let (guard, captured) = capture_logs();
            let outcome =
                connect_with_retry("data pool", RETRY_BUDGET, Some(&diag), &flag, || async {
                    Err(fred::error::Error::new(
                        FredErrorKind::IO,
                        "connection refused",
                    ))
                })
                .await;
            drop(guard);

            assert!(matches!(outcome, Err(ConnectError::Timeout { .. })));
            assert!(
                !captured_text(&captured).contains(HINT_MARKER),
                "a transient I/O failure is not a credential rejection: {}",
                captured_text(&captured)
            );
            assert!(
                !flag.load(Ordering::Relaxed),
                "the per-connect flag must stay unarmed so a genuine rejection later in the same \
                 connect() is not swallowed"
            );

            cancel.cancel();
        }

        #[tokio::test]
        async fn auth_hint_gate_first_call_passes() {
            // Never fully silent: the first rejection always speaks, whatever
            // the interval.
            let gate = AuthHintGate::new(Duration::from_secs(3600));

            assert!(gate.allow());
        }

        #[tokio::test(start_paused = true)]
        async fn auth_hint_gate_blocks_inside_the_interval_and_reopens_after_it() {
            // Pinned against the PRODUCTION constant, so this covers both the
            // gate's semantics and the shipped 5-minute spacing. The clock is
            // `tokio::time::Instant`, which `start_paused` virtualises — that
            // choice is what makes a 5-minute interval testable in
            // microseconds.
            let gate = AuthHintGate::new(STEADY_STATE_AUTH_HINT_MIN_INTERVAL);
            assert!(gate.allow(), "first call always passes");

            tokio::time::sleep(STEADY_STATE_AUTH_HINT_MIN_INTERVAL - Duration::from_secs(1)).await;
            assert!(
                !gate.allow(),
                "a second hint one second short of the interval must be suppressed"
            );

            tokio::time::sleep(Duration::from_secs(2)).await;
            assert!(
                gate.allow(),
                "past the interval the gate must reopen — an operator tailing logs during an \
                 incident has to see the hint eventually"
            );
        }

        #[tokio::test(start_paused = true)]
        async fn steady_state_hint_is_bounded_by_the_shared_gate_across_pools() {
            // The steady-state surface is up to ~392 error hooks per replica
            // once every stream reader has connected (≈1,176 across the
            // chart's 3 replicas), all holding a
            // clone of the SAME `ValkeyAuthDiagnostics`, and
            // `reconnect_on_auth_error` retries forever — so the gate must be
            // shared, not per-hook. Five different pool labels, one hint.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;

            let (guard, captured) = capture_logs();
            for pool in ["data", "txn", "subscriber", "maintenance", "snapshot"] {
                diag.emit_steady_state_hint(pool);
            }
            drop(guard);

            assert_eq!(
                captured_text(&captured).matches(HINT_MARKER).count(),
                1,
                "every pool's hook shares one gate, so the process emits one steady-state hint \
                 per interval rather than one per connection"
            );

            cancel.cancel();
        }

        #[tokio::test(start_paused = true)]
        async fn connect_and_steady_state_gates_are_independent() {
            // Different surfaces with different bounds: the connect flag is
            // per-invocation, the steady-state gate is time-based. One having
            // fired must not silence the other, or a rejection observed at
            // startup would suppress the mid-life hint that is the only signal
            // for a revoked policy assignment.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;

            let (guard, captured) = capture_logs();
            diag.emit_connect_hint("data pool", &AtomicBool::new(false));
            diag.emit_steady_state_hint("data");
            drop(guard);
            let output = captured_text(&captured);

            assert_eq!(output.matches(HINT_MARKER).count(), 2);
            assert!(
                output.contains("phase=\"connect\"") && output.contains("phase=\"steady-state\""),
                "each emission must name its own surface so an operator can tell a boot-time \
                 rejection from a mid-life one: {output}"
            );

            cancel.cancel();
        }

        // ── Steady-state gate: WRONGPASS only, never NOAUTH ─────────────────

        /// The steady-state call-site condition, restated verbatim from
        /// [`spawn_error_metric`].
        ///
        /// `spawn_error_metric` itself cannot be *run* here — its input is a
        /// fred broadcast channel owned by a connected `Client` — so the gate is
        /// driven directly, the same way the sibling rate-limit tests drive
        /// `emit_steady_state_hint` directly. What keeps that from degenerating
        /// into a test of its own mirror is
        /// `wiring_guards::spawn_error_metric_gates_the_steady_state_hint_on_wrongpass`,
        /// which pins that the production site still composes these same two
        /// terms and calls `emit_steady_state_hint` behind them.
        fn steady_state_gate_admits(err: &fred::error::Error) -> bool {
            is_server_auth_rejection(err) && err.details().starts_with("WRONGPASS")
        }

        #[tokio::test]
        async fn steady_state_hint_fires_on_wrongpass_and_never_on_noauth() {
            // The steady-state surface is narrower than the classifier on
            // purpose. A mid-life `NOAUTH` is exactly what
            // `reconnect_on_auth_error` exists to absorb — a managed-proxy
            // session reset, or a connection that raced a credential rotation —
            // and fred re-`AUTH`s and the pod is fine. Firing the four-cause
            // misconfiguration checklist at ERROR on a self-healing blip is a
            // confident, authoritative WRONG diagnosis. `WRONGPASS` is the
            // persistent shape (the server has a credential and refuses it),
            // which is the one the checklist actually diagnoses.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;
            let noauth = mk_auth_err("NOAUTH Authentication required.");
            let wrongpass =
                mk_auth_err("WRONGPASS invalid username-password pair or user is disabled.");

            assert!(
                is_server_auth_rejection(&noauth),
                "precondition: the shared classifier DOES accept NOAUTH — the narrowing under \
                 test lives at the steady-state call site, not in the classifier, so this test \
                 is vacuous if the classifier already rejected it"
            );

            // NOAUTH first, while the shared gate is still unarmed: a hint here
            // could only come from the gate admitting it.
            let (guard, captured) = capture_logs();
            if steady_state_gate_admits(&noauth) {
                diag.emit_steady_state_hint("data");
            }
            drop(guard);
            assert_eq!(
                captured_text(&captured).matches(HINT_MARKER).count(),
                0,
                "a mid-life NOAUTH is absorbed by reconnect_on_auth_error and must not print the \
                 four-cause checklist: {}",
                captured_text(&captured)
            );

            let (guard, captured) = capture_logs();
            if steady_state_gate_admits(&wrongpass) {
                diag.emit_steady_state_hint("data");
            }
            drop(guard);
            assert_eq!(
                captured_text(&captured).matches(HINT_MARKER).count(),
                1,
                "a persistent WRONGPASS is the shape the checklist diagnoses, and the gate was \
                 still unarmed, so it must speak: {}",
                captured_text(&captured)
            );

            cancel.cancel();
        }

        #[tokio::test(start_paused = true)]
        async fn connect_with_retry_still_hints_on_noauth() {
            // The other side of the asymmetry, at its real call site: the
            // CONNECT path keeps both tokens. A `NOAUTH` on the very first
            // connection is terminal — there is no established, authenticated
            // pool for `reconnect_on_auth_error` to repair — so suppressing the
            // hint there would leave a boot failure with nothing but the opaque
            // one-line WARN.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;
            let flag = AtomicBool::new(false);

            let (guard, captured) = capture_logs();
            let outcome =
                connect_with_retry("data pool", RETRY_BUDGET, Some(&diag), &flag, || async {
                    Err(fred::error::Error::new(
                        FredErrorKind::Auth,
                        "NOAUTH Authentication required.",
                    ))
                })
                .await;
            drop(guard);
            let output = captured_text(&captured);

            assert!(matches!(outcome, Err(ConnectError::Timeout { .. })));
            assert_eq!(
                output.matches(HINT_MARKER).count(),
                1,
                "the connect path must hint on NOAUTH: {output}"
            );
            assert!(
                output.contains("phase=\"connect\""),
                "and must attribute it to the connect surface: {output}"
            );

            cancel.cancel();
        }

        // ── Emission levels ─────────────────────────────────────────────────

        #[tokio::test]
        async fn emission_levels_are_warn_on_connect_error_on_steady_state_and_info_when_healthy() {
            // The level is not decoration: it decides whether a line pages.
            //
            // Connect is WARN because that hint fires on the FIRST failing
            // attempt while `connect_with_retry` keeps retrying, so a boot that
            // recovers later — a credential sidecar that had not written the
            // token file yet, an access-policy assignment still provisioning —
            // would otherwise end with an ERROR hint, a WARN retry line and a
            // contradicting INFO healthy-connect line all describing one
            // condition, and ERROR-level alerting would page on a pod that came
            // up fine. Steady-state is ERROR because there is no such recovery
            // narrative: the pool was already up and authenticated, and its call
            // site only fires on the persistent WRONGPASS shape.
            //
            // Asserted on the level token the formatter writes rather than on
            // message text, because the message is identical across the two
            // phases — the level is the ONLY thing that differs.
            const LEVELS: [&str; 3] = ["WARN", "ERROR", "INFO"];

            /// One emission site: what it is, the level it must carry, the
            /// marker proving it fired at all, and the call that fires it.
            type EmissionCase<'a> = (&'a str, &'a str, &'a str, Box<dyn Fn() + 'a>);

            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;

            let emissions: [EmissionCase<'_>; 3] = [
                (
                    "connect-phase hint",
                    "WARN",
                    HINT_MARKER,
                    Box::new(|| diag.emit_connect_hint("data pool", &AtomicBool::new(false))),
                ),
                (
                    "steady-state hint",
                    "ERROR",
                    HINT_MARKER,
                    Box::new(|| diag.emit_steady_state_hint("data")),
                ),
                (
                    "healthy-connect line",
                    "INFO",
                    HEALTHY_MARKER,
                    Box::new(|| diag.log_healthy_connect()),
                ),
            ];

            for (what, expected_level, marker, emit) in emissions {
                let (guard, captured) = capture_logs();
                emit();
                drop(guard);
                let output = captured_text(&captured);

                assert!(
                    output.contains(marker),
                    "precondition: the {what} must have been emitted at all, or the level \
                     assertion below passes vacuously; got: {output}"
                );
                assert!(
                    output.contains(expected_level),
                    "the {what} must be emitted at {expected_level}: {output}"
                );
                for other in LEVELS.iter().filter(|l| **l != expected_level) {
                    assert!(
                        !output.contains(other),
                        "the {what} must be emitted at {expected_level} ONLY, but the capture \
                         also carries {other}: {output}"
                    );
                }
            }

            cancel.cancel();
        }

        // ── Wiring guards (source inspection) ───────────────────────────────
        //
        // Three call sites this module cannot observe at runtime — reaching
        // them needs a Valkey that both DEMANDS and REJECTS (or accepts) a
        // token-file credential, which no environment here provides, and
        // `spawn_error_metric`'s input is a fred broadcast channel owned by a
        // connected `Client` that no stub can implement (`EventInterface`
        // requires `ClientLike`'s whole connection surface). Every one of them
        // can be deleted today with the whole suite still green while the
        // feature silently stops working in production.
        //
        // So they are pinned by source inspection — `include_str!` + string
        // matching over the function bodies — which is the established technique
        // in this file (`connect_fn_body()` and its three existing consumers).
        // The guards pin WIRING only: what each wired call then DOES is covered
        // by the runtime tests above.
        //
        // An anchor-not-found panic here is an INTENTIONAL loud maintenance
        // signal, not a flake: it means a guard can no longer find the code it
        // pins, and the fix is to re-anchor it on the moved definition — never
        // to delete the guard.
        mod wiring_guards {
            // No `use super::*`: these guards read the file as TEXT and touch no
            // production symbol at all, which is the whole point — they hold
            // when the code they pin cannot be linked, let alone run.

            /// Slice `src/storage/client.rs` between two anchors.
            ///
            /// Generalises the `connect_fn_body()` slicer above, which pins one
            /// specific span; these guards need three. Slicing (rather than
            /// searching the whole file) is what keeps the needles from matching
            /// this test module's own string literals.
            fn source_span(start_anchor: &str, end_anchor: &str) -> &'static str {
                let src: &'static str = include_str!("client.rs");
                let start = src.find(start_anchor).unwrap_or_else(|| {
                    panic!(
                        "source-inspection start anchor {start_anchor:?} is gone from \
                         src/storage/client.rs — INTENTIONAL loud maintenance signal, not a \
                         flake: a wiring guard can no longer find the function it pins. \
                         Re-anchor it on the moved definition"
                    )
                });
                let end = src[start..].find(end_anchor).unwrap_or_else(|| {
                    panic!(
                        "source-inspection end anchor {end_anchor:?} is gone from \
                         src/storage/client.rs after {start_anchor:?} — INTENTIONAL loud \
                         maintenance signal, not a flake: the sliced span no longer terminates \
                         where the guard expects. Re-anchor it"
                    )
                });
                &src[start..start + end]
            }

            // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
            #[test]
            fn spawn_error_metric_gates_the_steady_state_hint_on_wrongpass() {
                // The hook is the ONLY place a mid-life credential rejection is
                // observable: an access-policy assignment revoked or flipped to
                // `Failed` under an already-running process produces a
                // `WRONGPASS` on reconnect, and without this call the whole
                // steady-state half of b.cbe is dead code that every test in
                // this module still exercises directly.
                //
                // Three needles, in call-site form so they cannot match the
                // surrounding explanatory comments (which name
                // `is_server_auth_rejection`, `WRONGPASS` and `NOAUTH` in
                // prose):
                let body = source_span(
                    "fn spawn_error_metric<C>(",
                    "/// Run `attempt` in a retry loop with exponential backoff.",
                );

                let classifier = body.find("is_server_auth_rejection(&err)").expect(
                    "spawn_error_metric must classify the fred error with \
                     is_server_auth_rejection — without it a bare ErrorKind::Auth match would \
                     also fire on THIS service's own token-unavailable error",
                );
                let wrongpass_gate = body
                    .find(r#"err.details().starts_with("WRONGPASS")"#)
                    .expect(
                        "spawn_error_metric must narrow to WRONGPASS on top of the classifier — a \
                     mid-life NOAUTH is what reconnect_on_auth_error absorbs, and firing the \
                     four-cause checklist at ERROR on a self-healing blip is an authoritative \
                     wrong diagnosis",
                    );
                let emit = body.find("emit_steady_state_hint(pool_label)").expect(
                    "spawn_error_metric must call emit_steady_state_hint — this hook is the only \
                     place a mid-life credential rejection is observable, so without the call \
                     the steady-state hint is unreachable in production while every unit test \
                     that drives it directly stays green",
                );

                assert!(
                    classifier < wrongpass_gate && wrongpass_gate < emit,
                    "the emission must sit BEHIND both gates — classifier, then the WRONGPASS \
                     narrowing, then the hint"
                );
                assert_eq!(
                    body.matches("emit_steady_state_hint").count(),
                    1,
                    "exactly one emission site in this hook: a second, ungated call would \
                     reintroduce the NOAUTH misdiagnosis by another route"
                );
            }

            // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
            #[test]
            fn connect_records_the_healthy_verdict_after_every_pool_authenticates() {
                // `log_healthy_connect` is the only place the HEALTHY verdict is
                // ever printed — the hint path runs on rejection only — which is
                // precisely what makes the same `identity_verdict` field
                // trustworthy on a BROKEN deployment. Drop this call and an
                // operator loses the baseline to diff against, with no test
                // failure anywhere, because the unit tests above call
                // `log_healthy_connect()` themselves.
                //
                // Spans the WHOLE of `connect` rather than reusing
                // `connect_fn_body()`: that slicer deliberately stops at the
                // data-pool retry call, and this call site is far below it, just
                // before the `Ok(ValkeyClient { .. })` return.
                let body = source_span("pub async fn connect(", "pub async fn acquire_txn(");

                assert_eq!(
                    body.matches("auth_diagnostics.log_healthy_connect();")
                        .count(),
                    1,
                    "connect must record the identity verdict for the credential the backing \
                     store demonstrably ACCEPTED — exactly once, after every pool has PINGed, so \
                     the line is a statement about a working connection rather than a hopeful \
                     one"
                );
            }

            // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
            #[test]
            fn repair_rearms_the_hooks_and_hands_over_the_router_before_it_branches_on_the_init_result(
            ) {
                // b.289 sub-task 5. The bug this pins is a DEAD ZONE, not a
                // wrong value: `quit()` succeeding swaps fred's broadcast
                // senders (`fred-10.1.0/src/commands/impls/server.rs:40-42`,
                // after the await at `:39`), so the old hook tasks see
                // `RecvError::Closed` and exit. If the reconnect (`connect()`
                // plus its bounded `wait_for_connect()`, bound to `init`) then
                // fails or times out and the re-arm sits inside the success arm,
                // that slot's `valkey_errors_total{pool="txn"}` /
                // `valkey_reconnects_total{pool="txn"}` / b.cbe hint stay dead
                // until some LATER recycle happens to succeed.
                //
                // Pinned by source inspection because the triggering sequence
                // is not stageable: producing a real quit-ok + connect-timeout
                // needs a server that accepts a QUIT and then refuses the
                // reconnect for longer than TXN_RECYCLE_CONNECT_TIMEOUT, which
                // no fixture here can arrange. Regressing this is silent by
                // construction — the metric that would report it is the metric
                // that goes dead.
                let body = source_span(
                    "    async fn repair_held_txn_slot(",
                    "    /// Attach — or re-attach — the reconnect / error metric hooks",
                );

                assert_eq!(
                    body.matches("self.rearm_pool_hooks(client, POOL_LABEL_TXN);")
                        .count(),
                    1,
                    "repair_held_txn_slot must re-arm exactly once: a second call would be a \
                     second abort-then-insert on the same slot, and zero calls leaves the slot's \
                     hooks dead after any recycle that reached the sender swap"
                );

                let rearm = body
                    .find("self.rearm_pool_hooks(client, POOL_LABEL_TXN);")
                    .expect("checked by the count assertion above");
                let branch = body.find("match init {").expect(
                    "repair_held_txn_slot must bind the connection-wait result as `init` and \
                     branch on it separately — the re-arm has to sit BETWEEN the await and the \
                     branch, so the branch cannot be the thing that gates it",
                );

                assert!(
                    rearm < branch,
                    "the re-arm must run BEFORE the branch on the connection-wait result, i.e. \
                     unconditionally. Inside an `Ok(Ok(_))` arm it is a success action, but the \
                     predicate it actually needs is `quit() was attempted, so the broadcast \
                     senders may already have been swapped` — true on every arm"
                );

                assert!(
                    body[..rearm].contains("ClientLike::connect(client)"),
                    "the re-arm must sit after the reconnect (`connect()`) has been issued, not \
                     before it: re-arming against a client whose reconnect has not started yet \
                     would race the sender swap `quit()` may still be performing"
                );

                // Position alone is not the property. `rearm < branch` is still
                // satisfied by `if init.is_ok() { self.rearm_pool_hooks(..) }`
                // written ABOVE the `match` — which would reintroduce the exact
                // dead zone, in a shape that reads as if it had been fixed. So
                // also pin that the span between the bounded connection wait
                // (`connect()` then `wait_for_connect()` under a timeout, bound
                // to `init`) and the re-arm is STRAIGHT-LINE: comments, blank
                // lines and the closing braces of that binding only, no branch,
                // no early exit, no result inspection.
                const INIT_AWAIT: &str =
                    "Some(tokio::time::timeout(TXN_RECYCLE_CONNECT_TIMEOUT, connected).await)";
                let init_end = body.find(INIT_AWAIT).expect(
                    "repair_held_txn_slot must bind the bounded connection wait as `let init = \
                     .. Some(tokio::time::timeout(TXN_RECYCLE_CONNECT_TIMEOUT, connected).await) \
                     ..;` — the straight-line guard below slices from the end of that await",
                ) + INIT_AWAIT.len();
                assert!(
                    init_end < rearm,
                    "the re-arm must follow the connection wait, not precede it"
                );

                // b.v6d: the hand-over of the router this repair's `connect()`
                // started to the slot's supervisor watcher is the same kind of
                // action, needed on every arm. The watcher waits for it after
                // every recycle and restart; inside the success arm it would be
                // skipped on the `None` arm and on the reconnect-failed and
                // timeout arms, leaving the watcher waiting forever and the
                // slot's router unsupervised. The `None` arm is the restart of
                // a router that ended by returning AFTER IT HAD CONNECTED: it
                // stored its command channel back, so `quit()` returns at once
                // and the client still reads `Connected`. A router whose first
                // connect failed does not get there: under fred's default
                // `fail_fast` it returns without ever setting `Connected`
                // (`fred-10.1.0/src/router/commands.rs:534-549`), and its task
                // moves the `Disconnecting` left by the `quit()` of the repair
                // that started it to `Disconnected` (`runtime/_tokio.rs:299-313`),
                // so the restart after it awaits the new connection, for up to
                // `TXN_RECYCLE_CONNECT_TIMEOUT`. Only the success arm is staged
                // (`b_v6d_router_test`'s T38 and
                // `ended_txn_router_is_restarted_and_counted`: a router ended
                // by an abort is marked `Disconnecting` by the repair's own
                // `quit()`, so the repair awaits the new connection). The other
                // three arms cannot be staged with the seams there, so this
                // source check pins all of them.
                const HAND_OVER: &str = ".hand_over(ClientLike::id(client), connection_task);";
                assert_eq!(
                    body.matches(HAND_OVER).count(),
                    1,
                    "repair_held_txn_slot must hand the router its connect() started over to \
                     the slot's watcher exactly once"
                );
                let hand_over = body
                    .find(HAND_OVER)
                    .expect("checked by the count assertion");
                assert!(
                    body[..hand_over]
                        .trim_end()
                        .ends_with("self.router_supervisor"),
                    "the hand-over must go to this client's router supervisor"
                );
                assert!(
                    body[..hand_over]
                        .contains("let connection_task = ClientLike::connect(client);"),
                    "the handle handed over must be the one this repair's connect() returned"
                );
                assert!(
                    rearm < hand_over && hand_over < branch,
                    "the hand-over must run after the re-arm and BEFORE the branch on the \
                     connection-wait result, i.e. on every arm"
                );

                // Strip `//` comment tails so the prose between the wait and the
                // branch — which legitimately says "if", "match" and "returns"
                // — cannot fail the guard. Blank lines survive as empty strings.
                let executable: String = body[init_end..branch]
                    .lines()
                    .map(|line| line.split("//").next().unwrap_or(""))
                    .collect::<Vec<_>>()
                    .join("\n");

                for token in [
                    "if ",
                    "match ",
                    "?",
                    ".is_ok()",
                    ".is_err()",
                    "return",
                    "else",
                ] {
                    assert!(
                        !executable.contains(token),
                        "found {token:?} between the connection wait and the branch on its \
                         result in repair_held_txn_slot. The re-arm and the hand-over must be \
                         straight-line code reached on EVERY path. The re-arm needs `a quit() \
                         was attempted, so the broadcast senders may already have been swapped`, \
                         and the hand-over needs `connect() started a router`; both are true \
                         whether the reconnect succeeded, failed or timed out. Any conditional \
                         here — including one written above the `match`, which the ordering \
                         assertions above would still accept — reinstates a dead zone: a slot's \
                         valkey_errors_total{{pool=\"txn\"}} silent until some later recycle \
                         succeeds, or a slot's router no supervisor watches. Executable text \
                         found: {executable:?}"
                    );
                }

                // b.v6d defer-9: the repair ends the slot's old router
                // (`RouterSupervisor::abort_current`) between its `quit()` and
                // its `connect()`, on every arm. A `quit()` that timed out
                // leaves the old router draining the command that hung; one
                // that succeeded leaves it ending a moment later. Either, still
                // alive at `connect()`, later stores its command receiver back
                // over the one `connect()` installed, and the next repair of the
                // slot then starts a router on that dead channel. The
                // QUIT-timeout arm is staged in `b_v6d_router_test`'s
                // `old_router_abort` tests. On the QUIT-succeeded arm the old
                // router ends on its own a moment after `quit()` returns, a
                // race no fixture there controls, so this check pins the call
                // on every arm: once, after `quit()`, before `connect()`, at the
                // top level of the function body (not inside the QUIT-timeout
                // `if let`, which would skip the QUIT-succeeded arm), and with no
                // branch or early exit between it and `connect()`. The `match`
                // on the abort's own result only picks a log line.
                const QUIT: &str = "ClientLike::quit(client)";
                const ABORT: &str = ".abort_current(ClientLike::id(client))";
                const CONNECT: &str = "let connection_task = ClientLike::connect(client);";
                // `//` comment tails stripped, as above: the prose around the
                // call names `quit()`, `connect()` and `abort_current`.
                let code: String = body
                    .lines()
                    .map(|line| line.split("//").next().unwrap_or(""))
                    .collect::<Vec<_>>()
                    .join("\n");
                assert_eq!(
                    code.matches(ABORT).count(),
                    1,
                    "repair_held_txn_slot must end the slot's old router exactly once, through \
                     abort_current"
                );
                let quit = code
                    .find(QUIT)
                    .expect("repair_held_txn_slot must quit the client");
                let abort = code.find(ABORT).expect("checked by the count assertion");
                let connect = code
                    .find(CONNECT)
                    .expect("repair_held_txn_slot must start the new router with connect()");
                assert!(
                    quit < abort && abort < connect,
                    "abort_current must run after quit() and before connect(): after connect() \
                     the old router can already have stored its receiver back, and before \
                     quit() the QUIT that closes the old connection would find no router left \
                     to write it"
                );
                assert!(
                    code[..abort].trim_end().ends_with(".router_supervisor"),
                    "the abort must go to this client's router supervisor"
                );

                // Brace depth of `code[..at]`, outside string literals: 1 is the
                // top level of the function body.
                fn depth_at(code: &str, at: usize) -> i32 {
                    let (mut depth, mut in_string, mut escaped) = (0, false, false);
                    for c in code[..at].chars() {
                        if in_string {
                            match c {
                                _ if escaped => escaped = false,
                                '\\' => escaped = true,
                                '"' => in_string = false,
                                _ => {}
                            }
                            continue;
                        }
                        match c {
                            '"' => in_string = true,
                            '{' => depth += 1,
                            '}' => depth -= 1,
                            _ => {}
                        }
                    }
                    depth
                }
                assert_eq!(
                    (depth_at(&code, abort), depth_at(&code, connect)),
                    (1, 1),
                    "abort_current and connect() must both sit at the top level of \
                     repair_held_txn_slot's body. Inside the QUIT-timeout `if let` the abort \
                     would skip the QUIT-succeeded arm, whose old router can also still be \
                     running at connect()"
                );
                // From the `quit()` to the abort, only the QUIT-timeout WARN and
                // the `match` heading the abort itself: no early exit.
                let before = &code[quit..abort];
                for token in ["else", "return", "?", "break", "continue"] {
                    assert!(
                        !before.contains(token),
                        "found {token:?} between quit() and abort_current in \
                         repair_held_txn_slot: an arm that leaves there reaches no abort. \
                         Executable text found: {before:?}"
                    );
                }
                let between = &code[abort..connect];
                for token in [
                    "if ", "else", "match ", "return", "?", "break", "continue", "loop", "while ",
                ] {
                    assert!(
                        !between.contains(token),
                        "found {token:?} between abort_current and connect() in \
                         repair_held_txn_slot: nothing may branch or exit between ending the old \
                         router and starting the new one, or some arm reaches connect() with the \
                         old router still running. Executable text found: {between:?}"
                    );
                }

                // The `StillRunning` arm is the only signal an operator gets
                // that a repair reconnected beside an old router it could not
                // end, which may yet store its command receiver back. No real
                // fred router outlives the abort budget, so the arm cannot be
                // staged through a client (`router_supervisor`'s unit tests
                // pin the classification); pin that it logs at WARN, with its
                // message, inside that arm. String continuations (`\` at a
                // line end, then the next line's indent) are joined first, as
                // the compiler does, so the message is matched as logged.
                let mut joined = String::new();
                for line in between.lines() {
                    if joined.ends_with('\\') {
                        joined.pop();
                        joined.push_str(line.trim_start());
                    } else {
                        joined.push('\n');
                        joined.push_str(line);
                    }
                }
                const STILL_RUNNING_ARM: &str = "AbortCurrent::StillRunning => tracing::warn!(";
                const STILL_RUNNING_MSG: &str =
                    "the slot's previous router did not end within its abort budget";
                let arm = joined.find(STILL_RUNNING_ARM).unwrap_or_else(|| {
                    panic!(
                        "repair_held_txn_slot's StillRunning arm must log at WARN \
                         ({STILL_RUNNING_ARM:?}): it is the only record of a repair that \
                         reconnected beside a router it could not end. Text found: {joined:?}"
                    )
                });
                let arm_end = joined[arm..]
                    .find("AbortCurrent::Unsupervised")
                    .map_or(joined.len(), |end| arm + end);
                assert!(
                    joined[arm..arm_end].contains(STILL_RUNNING_MSG),
                    "repair_held_txn_slot's StillRunning WARN must say {STILL_RUNNING_MSG:?}. \
                     Arm found: {:?}",
                    &joined[arm..arm_end]
                );
            }

            // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
            #[test]
            fn install_observability_hooks_gives_every_error_hook_the_shared_diagnostics() {
                // The rate-limit gate lives on the single `ValkeyAuthDiagnostics`
                // the client holds, so sharing is what bounds the process to one
                // hint per interval at a fan-out of up to ~392 hooks per
                // replica. A hook wired with `None` instead is worse than a
                // noisy one: it is silent, and the pool it belongs to is exactly
                // the one whose rejection nobody will ever see explained.
                //
                // Three installers, each pinned below:
                //
                // 1. `install_observability_hooks` installs the pub/sub
                //    subscriber's hooks itself.
                // 2. `PoolHooks::rearm_with` is the SOLE installer for the hooks
                //    of the connections replaced at runtime: each `txn` slot (its
                //    startup install and every dirty-slot recycle go through
                //    `rearm_pool_hooks`, b.7td round 2, which hands it to the
                //    member replacer's `arm_txn`; a slot replacement arms the
                //    fresh client through the member replacer too), each stream reader's
                //    current connection (armed when the reader connects lazily
                //    or retargets, b.azh), and each `data`, `maintenance` and
                //    `snapshot` member's current client (armed by the member
                //    replacer at install and on every replacement, b.zt3). It
                //    arms with the `PoolHooks`' diagnostics.
                // 3. So every `PoolHooks` built in production must carry the
                //    client's one diagnostics `Arc`: the client's own
                //    (`pool_hooks()`, for any label `rearm_pool_hooks` gets
                //    other than `txn`), the readers' connector's and the
                //    member replacer's (the `txn` slots' and the multiplexed
                //    members'; both built in `connect`, beside the client).
                //
                // Two accepted argument forms, because rustfmt picks between
                // them by line width: the exploded call ends the argument with
                // `diagnostics.clone(),` and the one-line call (which is what
                // `PoolHooks::rearm` fits into) ends it with
                // `diagnostics.clone())`. Both are ARGUMENT positions; neither
                // can cross-match the single `let diagnostics =
                // self.auth_diagnostics.clone();` binding, which ends in `;`.
                let shared_args = |body: &str| {
                    body.matches("diagnostics.clone(),").count()
                        + body.matches("diagnostics.clone())").count()
                };

                let install = source_span(
                    "    pub async fn install_observability_hooks(",
                    "pub async fn quit(&self) {",
                );
                let hooks = install.matches("spawn_error_metric(").count();
                assert!(
                    hooks >= 1,
                    "expected install_observability_hooks to install the pub/sub subscriber's \
                     error hook itself; found {hooks}"
                );
                assert_eq!(
                    hooks,
                    shared_args(install),
                    "every spawn_error_metric call must be handed the SHARED diagnostics clone — \
                     a hook wired without it is permanently silent for its pool, and one wired \
                     with a separate ValkeyAuthDiagnostics would carry its own rate-limit gate \
                     and multiply the hint by the pool count"
                );
                assert_eq!(
                    install
                        .matches("self.rearm_pool_hooks(&client, POOL_LABEL_TXN);")
                        .count()
                        + install
                            .matches("self.stream_mux.arm_reader_hooks();")
                            .count()
                        + install.matches("self.member_replacer.arm_hooks();").count(),
                    3,
                    "install_observability_hooks must arm the txn slots, the readers already \
                     connected and the data / maintenance / snapshot members through the \
                     registered (PoolHooks) path"
                );

                let rearm = source_span("impl PoolHooks {", "    pub(crate) fn disarm(");
                assert_eq!(
                    rearm.matches("spawn_error_metric(").count(),
                    1,
                    "PoolHooks::rearm_with installs one error hook per connection"
                );
                assert_eq!(
                    shared_args(rearm),
                    1,
                    "PoolHooks::rearm_with must hand its error hook the PoolHooks' diagnostics clone"
                );

                let client_hooks =
                    source_span("    fn pool_hooks(&self) -> PoolHooks {", "\n    }");
                assert!(
                    client_hooks.contains("diagnostics: self.auth_diagnostics.clone(),"),
                    "the client's own PoolHooks must carry the client's own diagnostics Arc: \
                     {client_hooks}"
                );
                let connect = source_span("pub async fn connect(", "fn start_router_supervision(");
                assert_eq!(
                    connect
                        .matches("diagnostics: Some(Arc::clone(&auth_diagnostics)),")
                        .count(),
                    2,
                    "the readers' connector and the member replacer must carry the very \
                     diagnostics Arc the client is given (`auth_diagnostics: \
                     Some(auth_diagnostics)`), not a second instance"
                );
                assert_eq!(
                    connect
                        .matches("auth_diagnostics: Some(auth_diagnostics),")
                        .count(),
                    1,
                    "the client holds the same diagnostics Arc the readers' connector shares"
                );
                assert_eq!(
                    connect.matches("ValkeyAuthDiagnostics::new(").count()
                        + connect.matches("ValkeyAuthDiagnostics::from_").count(),
                    1,
                    "connect builds exactly one ValkeyAuthDiagnostics, shared by every hook"
                );
            }
        }

        // ── Healthy-connect INFO ────────────────────────────────────────────

        #[tokio::test]
        async fn log_healthy_connect_records_the_verdict_for_the_accepted_credential() {
            // The only place the HEALTHY verdict is ever printed — the hint
            // path runs on rejection only — which is what makes the same field
            // on a broken deployment trustworthy.
            //
            // Unit-scope on purpose: reaching this line through a real
            // `ValkeyClient::connect` would need a Valkey that both demands and
            // accepts a token-file credential, which no test environment here
            // provides (the sandbox instance is unauthenticated). What connect
            // owns beyond this is the call site itself.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) =
                access_token_file_diagnostics(dir.path(), OBJECT_ID, &full_claims_jwt()).await;

            let (guard, captured) = capture_logs();
            diag.log_healthy_connect();
            drop(guard);
            let output = captured_text(&captured);

            assert_eq!(
                output.matches(HEALTHY_MARKER).count(),
                1,
                "exactly one baseline line per successful connect: {output}"
            );
            assert!(
                output.contains("identity_verdict=\"matches-oid\""),
                "the healthy verdict is the baseline a later failure is diffed against: {output}"
            );
            assert!(
                output.contains(&format!("token_oid={OBJECT_ID}")),
                "the structured claim fields must ride the same line: {output}"
            );

            cancel.cancel();
        }

        #[test]
        fn log_healthy_connect_is_silent_under_static_and_iam() {
            // Neither mode has a token whose claims could be reported, so a
            // line there would carry nothing but the mode — noise on every
            // successful boot of every deployment that does not use
            // accessTokenFile.
            for config in [
                auth_mode_test_config(ValkeyAuthMode::Static, PLAINTEXT_URL),
                iam_test_config(PLAINTEXT_URL),
            ] {
                let mode = config.valkey_auth_mode;
                let diag = ValkeyAuthDiagnostics::new(&config, None);

                let (guard, captured) = capture_logs();
                diag.log_healthy_connect();
                drop(guard);

                assert_eq!(
                    captured_text(&captured),
                    "",
                    "{mode:?} must emit no healthy-connect line"
                );
            }
        }

        #[tokio::test]
        async fn log_healthy_connect_reports_a_mismatched_identity_it_does_not_understand() {
            // A credential the server ACCEPTED whose username matches neither
            // claim is a real and confusing state (the backing store's ACL may
            // key off something else entirely). The line must record what it
            // found rather than suppressing itself — a baseline that only
            // prints when it agrees with its own expectation is not a baseline.
            let dir = tempdir().expect("tempdir");
            let (diag, cancel) = access_token_file_diagnostics(
                dir.path(),
                "some-other-acl-user",
                &full_claims_jwt(),
            )
            .await;

            let (guard, captured) = capture_logs();
            diag.log_healthy_connect();
            drop(guard);
            let output = captured_text(&captured);

            assert!(
                output.contains("identity_verdict=\"matches-neither\""),
                "the healthy-connect line must report the verdict it actually derived: {output}"
            );

            cancel.cancel();
        }
    }

    // ── The readers' connector: routing-view sync and retire ─────────────────
    //
    // `ReaderConnector::sync_routing` is single-flight: one `CLUSTER SLOTS` at
    // a time on the pinned client, and a caller that waited while another's
    // sync ended — answered, cut off by its bound, or abandoned by its caller —
    // returns `Ok` without issuing its own. These tests run on
    // `unconnected_for_test`, whose pinned client was never connected: fred
    // queues its `SyncCluster` for a router that never runs, so every sync
    // actually issued hangs until its bound cuts it — 1 s, unless a test
    // changes the connector's timeouts first (`Err(Timeout)`, one
    // more on `routing_syncs_finished_for_test`). A caller that issued none
    // returns `Ok` at once. The clock is paused, so a cut costs no wall time.
    mod reader_connector_routing_sync {
        use super::*;

        /// Callers queued on the sync lock in each test.
        const QUEUED: usize = 4;

        /// The connector's bound on one sync (`unconnected_for_test`'s 1 s
        /// command timeout).
        const SYNC_BOUND_MS: u128 = 1_000;

        fn connector() -> Arc<ReaderConnector> {
            let routing = Builder::default_centralized()
                .build_pool(1)
                .expect("an unconnected pool builds");
            ReaderConnector::unconnected_for_test(routing, true)
        }

        /// The error a sync cut off by its bound returns.
        fn assert_cut(result: Result<(), fred::error::Error>, what: &str) {
            let e = result.expect_err(what);
            assert!(
                matches!(e.kind(), FredErrorKind::Timeout),
                "{what}: a cut sync fails as a timeout: {e:?}"
            );
            assert_eq!(
                e.details(),
                format!("the cluster routing sync did not finish within {SYNC_BOUND_MS} ms"),
                "{what}"
            );
        }

        /// First-polls each caller in order — each reads the sync generation
        /// and joins the lock's queue — asserting it is waiting.
        async fn first_poll_all<F: std::future::Future + Unpin>(callers: &mut [F]) {
            for (i, caller) in callers.iter_mut().enumerate() {
                assert!(
                    futures::poll!(tokio::task::coop::unconstrained(caller)).is_pending(),
                    "caller {i} must wait"
                );
            }
        }

        #[test]
        fn routing_sync_bound_is_the_command_timeout_or_else_the_connect_timeout() {
            let connect = Duration::from_secs(30);
            assert_eq!(
                routing_sync_bound(Duration::from_millis(750), connect),
                Duration::from_millis(750),
                "a command timeout bounds the sync"
            );
            assert_eq!(
                routing_sync_bound(Duration::ZERO, connect),
                connect,
                "a disabled (zero) command timeout falls back to the connect timeout, never a \
                 zero bound that would cut every sync on its first poll"
            );
        }

        /// Non-vacuity for the tests below: a caller that did not wait behind
        /// another's sync issues its own, so the count moves once per call.
        #[tokio::test(start_paused = true)]
        async fn a_caller_that_waited_behind_no_sync_issues_its_own() {
            let connector = connector();
            assert_eq!(connector.routing_syncs_finished_for_test(), 0);
            for n in 1..=2u64 {
                assert_cut(connector.sync_routing().await, "an unanswered sync");
                assert_eq!(
                    connector.routing_syncs_finished_for_test(),
                    n,
                    "each call that waited behind nothing issues a sync of its own"
                );
            }
        }

        /// `sync_routing` bounds the sync it issues by `routing_sync_bound`:
        /// the command timeout, or the connect timeout when the command
        /// timeout is disabled (zero). Each case uses a connect timeout unlike
        /// its command timeout, so the bound the call site picks shows in the
        /// cut's message and in the paused time it took:
        ///
        /// - command 1 s, connect 3 s: cut at 1 s. Bounded by the connect
        ///   timeout (or with the arguments swapped) it would wait 3 s.
        /// - command 0 (disabled), connect 2 s: cut at 2 s. Bounded by the zero
        ///   command timeout it would be cut on its first poll, before fred
        ///   could answer any sync at all.
        ///
        /// On a paused clock, so the waits cost no wall time.
        #[tokio::test(start_paused = true)]
        async fn sync_routing_is_bounded_by_the_command_timeout_or_else_the_connect_timeout() {
            #[rustfmt::skip]
            let cases = [
                // case,                      command,                   connect,                  cut at
                ("command timeout set",       Duration::from_secs(1),    Duration::from_secs(3),   Duration::from_secs(1)),
                ("command timeout disabled",  Duration::ZERO,            Duration::from_secs(2),   Duration::from_secs(2)),
            ];
            for (case, command, connect, cut_at) in cases {
                let mut connector = connector();
                {
                    let fresh =
                        Arc::get_mut(&mut connector).expect("a fresh connector is not shared yet");
                    fresh.command_timeout = command;
                    fresh.connect_timeout = connect;
                }
                let start = tokio::time::Instant::now();

                let e = connector
                    .sync_routing()
                    .await
                    .expect_err("an unanswered sync is cut at its bound");
                let elapsed = start.elapsed();
                assert!(matches!(e.kind(), FredErrorKind::Timeout), "{case}: {e:?}");
                assert_eq!(
                    e.details(),
                    format!(
                        "the cluster routing sync did not finish within {} ms",
                        cut_at.as_millis()
                    ),
                    "{case}: the sync's bound"
                );
                assert!(
                    elapsed >= cut_at,
                    "{case}: the sync was cut at {cut_at:?}, not sooner; cut after {elapsed:?}"
                );
                // The timeout the bound must NOT be. When it is the longer one,
                // the sync must have been cut before it; when it is the zero
                // command timeout, the check above already rules it out.
                let other = if cut_at == command { connect } else { command };
                if other > cut_at {
                    assert!(
                        elapsed < other,
                        "{case}: the sync was cut at {cut_at:?}, before the {other:?} timeout \
                         it must not be bounded by; cut after {elapsed:?}"
                    );
                }
                assert_eq!(
                    connector.routing_syncs_finished_for_test(),
                    1,
                    "{case}: one sync issued"
                );
            }
        }

        /// Callers queued while a sync is in flight issue none of their own
        /// once it ends: they return `Ok` and read the view that sync left.
        #[tokio::test(start_paused = true)]
        async fn callers_queued_behind_a_sync_that_ends_return_ok_without_syncing() {
            let connector = connector();
            let in_flight = connector.hold_routing_sync_for_test().await;
            let mut callers: Vec<_> = (0..QUEUED)
                .map(|_| Box::pin(connector.sync_routing()))
                .collect();
            first_poll_all(&mut callers).await;

            connector.finish_routing_sync_for_test();
            drop(in_flight);
            for (i, caller) in callers.into_iter().enumerate() {
                assert!(
                    caller.await.is_ok(),
                    "caller {i} waited while a sync ended: it uses that sync's view"
                );
            }
            assert_eq!(
                connector.routing_syncs_finished_for_test(),
                1,
                "{QUEUED} callers cost the one sync in flight, none of their own"
            );
        }

        /// Concurrent callers cost one `CLUSTER SLOTS`: the first issues it,
        /// the rest wait on the lock. Its bound cuts it, and the cut counts as
        /// the sync ending for every caller queued behind it — fred may still
        /// run the orphaned sync, and they issue none behind it.
        #[tokio::test(start_paused = true)]
        async fn a_sync_cut_at_its_bound_ends_it_for_every_caller_queued_behind_it() {
            let connector = connector();
            let mut callers: Vec<_> = (0..QUEUED)
                .map(|_| Box::pin(connector.sync_routing()))
                .collect();
            first_poll_all(&mut callers).await;

            let mut callers = callers.into_iter();
            let first = callers.next().expect("QUEUED > 0");
            assert_cut(first.await, "the sync the first caller issued");
            for (i, caller) in callers.enumerate() {
                assert!(
                    caller.await.is_ok(),
                    "queued caller {} returns at once once the sync it waited on was cut",
                    i + 1
                );
            }
            assert_eq!(
                connector.routing_syncs_finished_for_test(),
                1,
                "one sync issued for {QUEUED} concurrent callers"
            );
        }

        /// A caller cancelled mid-sync (a reader shut down, say) ends its sync
        /// for the callers queued behind it, as an answer or a cut would: the
        /// generation moves, the lock is released, and they return `Ok`
        /// without syncing.
        #[tokio::test(start_paused = true)]
        async fn a_sync_whose_caller_is_cancelled_mid_flight_ends_it_for_the_callers_behind() {
            let connector = connector();
            let mut callers: Vec<_> = (0..QUEUED)
                .map(|_| Box::pin(connector.sync_routing()))
                .collect();
            first_poll_all(&mut callers).await;
            assert_eq!(
                connector.routing_syncs_finished_for_test(),
                0,
                "precondition: the first caller's sync is in flight"
            );

            let mut callers = callers.into_iter();
            drop(callers.next());
            assert_eq!(
                connector.routing_syncs_finished_for_test(),
                1,
                "cancelling the caller mid-sync ends its sync"
            );
            for (i, caller) in callers.enumerate() {
                assert!(
                    caller.await.is_ok(),
                    "queued caller {} returns at once: the sync it waited on ended",
                    i + 1
                );
            }
            assert_eq!(connector.routing_syncs_finished_for_test(), 1);
        }

        /// A reader connection whose client is not `Connected` — fred is still
        /// setting it up, or reconnecting it after a failover took its node —
        /// is retired with no `QUIT`: none was sent, so none overran, and the
        /// router is aborted at once, closing the connection. A `QUIT` there
        /// would queue behind the stalled setup and wait out `QUIT_TIMEOUT`.
        #[tokio::test]
        async fn retire_sends_no_quit_to_a_reader_client_that_is_not_connected() {
            let (listener, port) = silent_listener().await;
            let client = client_to_silent_listener(port);
            let connector = connector();
            let router = ClientLike::connect(&client);
            let mut setup = accept_stalled_setup(&listener).await;
            assert_ne!(
                client.state(),
                ClientState::Connected,
                "precondition: the client's setup is stalled, so it is not connected"
            );
            let restart_client = client.clone();
            let watch = connector.supervisor.watch_retirable(
                POOL_LABEL_SUBSCRIBER,
                0,
                ClientLike::id(&client),
                router,
                RouterRestart::Connect(Box::new(move || ClientLike::connect(&restart_client))),
            );
            let conn = ReaderConn {
                client,
                node: None,
                watch: Some(watch),
                counted: false,
            };

            let outcome = tokio::time::timeout(CUT_DEADLINE, connector.retire(conn))
                .await
                .expect("the retire must return");
            assert!(
                !outcome.quit_overran,
                "no QUIT is sent to a client that is not connected, so none overran"
            );
            await_client_closed(
                &mut setup,
                "the retire must abort the client's router, closing its connection",
            )
            .await;
            assert_eq!(connector.supervisor.watchers(), 0, "the watch is retired");
        }
    }

    // b.26e Stage 2 — the client's owners (`ClientOwners`): without `quit()`,
    // the drop of a client's last owner cancels its router watchers and its
    // member replacer's tasks, frees the client its own machinery's internal
    // clones held in a cycle, and stops its fan-out threads even while an
    // internal clone remains. Here rather than in `tests/` because each test
    // keeps a part of the client past its drop — the router supervisor, the
    // member replacer, a weak reference — or an internal clone, which the
    // public API cannot reach. The fan-out threads' stop on an un-quit drop,
    // from a service worker, from a thread whose runtime has ended and from
    // a fan-out thread itself (`fanout-read`, a `watch-read` thread), is
    // pinned on the public API in
    // `tests/integration/storage/b26e_fanout_thread_test.rs`.
    //
    // Each test needs `TEST_VALKEY_URL`; every wait is bounded by `SETTLE`.
    mod client_owners {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use super::*;
        use crate::error::ServiceError;
        use crate::storage::fanout_thread::{PassAnswer, PassLane, FANOUT_RUNTIMES};
        use crate::storage::member_replacement::{
            ReplaceKnobs, MEMBER_REPLACE_AFTER_FAST_VERDICTS, MEMBER_REPLACE_COOLDOWN,
            MEMBER_REPLACE_WINDOW,
        };

        /// Bound on each polled condition.
        const SETTLE: Duration = Duration::from_secs(10);

        /// A retirement grace no test here reaches: a retiring old client is
        /// still inside it when the client drops.
        const LONG_GRACE: Duration = Duration::from_secs(600);

        /// A client of its own on `TEST_VALKEY_URL`, whose one owner is the
        /// value returned.
        async fn own_client() -> ValkeyClient {
            let mut config = connect_timeout_test_config(30);
            config.valkey_url = std::env::var("TEST_VALKEY_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned());
            crate::storage::test_support::connect_test_client(&config)
                .await
                .expect("the client must connect")
        }

        /// Poll `done` until it holds; fail naming `what` after `SETTLE`.
        async fn settles(what: &str, mut done: impl FnMut() -> bool) {
            let deadline = std::time::Instant::now() + SETTLE;
            while !done() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "{what}: not within {SETTLE:?}"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }

        /// `client`'s live fan-out thread count, once every one of the
        /// `FANOUT_RUNTIMES` runtimes' worker threads has started: tokio
        /// counts a thread from inside it, not when its runtime is built.
        async fn running_fan_out_threads(client: &ValkeyClient) -> Arc<AtomicUsize> {
            let alive = client.fanout_threads().alive_threads();
            settles("every fan-out runtime's worker thread starts", || {
                alive.load(Ordering::SeqCst) >= FANOUT_RUNTIMES
            })
            .await;
            alive
        }

        /// The router supervisor outlives the client here; its watchers —
        /// a `txn` slot's holding an internal clone of the client for its
        /// restart — all exit once the client's only owner drops un-quit.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn dropping_the_last_owner_without_quit_cancels_every_router_watcher_b26e() {
            let client = own_client().await;
            let supervisor = Arc::clone(&client.router_supervisor);
            assert!(
                !client.txn_slot_watches_for_test().is_empty(),
                "precondition: the `txn` slots' watchers run"
            );
            assert!(supervisor.watchers() > 0, "precondition: watchers run");
            drop(client);
            settles(
                "every router watcher exits once the last owner drops",
                || supervisor.watchers() == 0,
            )
            .await;
        }

        /// A data member's old client retiring inside a grace no wait here
        /// reaches is the member replacer's task; it ends once the client's
        /// only owner drops un-quit, rather than at the grace's end.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn dropping_the_last_owner_without_quit_cancels_the_member_replacers_tasks_b26e() {
            let client = own_client().await;
            client.set_member_replacement_knobs_for_test(ReplaceKnobs {
                after_fast_verdicts: MEMBER_REPLACE_AFTER_FAST_VERDICTS,
                window: MEMBER_REPLACE_WINDOW,
                cooldown: MEMBER_REPLACE_COOLDOWN,
                grace: LONG_GRACE,
            });
            let replacer = Arc::clone(&client.member_replacer);
            for _ in 0..MEMBER_REPLACE_AFTER_FAST_VERDICTS {
                client.fast_verdict_for_test(POOL_LABEL_DATA, 0);
            }
            settles(
                "precondition: the replaced data member's old client retires in its grace",
                || client.member_retirements_for_test() == 1,
            )
            .await;
            assert!(
                replacer.tasks_for_test() >= 1,
                "precondition: the retirement is a live replacer task"
            );
            drop(client);
            settles(
                "every member-replacer task ends once the last owner drops",
                || replacer.tasks_for_test() == 0,
            )
            .await;
        }

        /// The client's own machinery holds clones of it — the `txn` slots'
        /// router restart, in the member replacer and in each `txn` slot's
        /// watcher — so without `quit()` the client is freed only if those
        /// clones are released once its last owner drops: every clone is gone
        /// (the shared fan-out threads' last reference with them), no cycle
        /// left.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn dropping_the_last_owner_without_quit_frees_the_client_b26e() {
            let client = own_client().await;
            assert!(
                !client.txn_slot_watches_for_test().is_empty(),
                "precondition: the `txn` slots' watchers, each holding an internal clone, run"
            );
            let threads = Arc::downgrade(&client.fanout_threads);
            drop(client);
            settles(
                "every clone of the client, its internal ones included, is dropped",
                || threads.strong_count() == 0,
            )
            .await;
        }

        /// An internal clone is not an owner: with one still held, the drop of
        /// the client's only owner stops every fan-out thread, and a pass run
        /// through the internal clone answers `Unavailable` on either lane —
        /// a fenced pass for `fanout-read`, an unfenced one for the
        /// `watch-read` threads.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn an_internal_clone_alone_does_not_keep_the_fan_out_threads_alive_b26e() {
            let client = own_client().await;
            let alive = running_fan_out_threads(&client).await;
            let internal = client.internal_clone();
            drop(client);
            settles(
                "every fan-out thread stops while only an internal clone remains",
                || alive.load(Ordering::SeqCst) == 0,
            )
            .await;
            for lane in [PassLane::Fenced, PassLane::Unfenced] {
                let ran = tokio::time::timeout(
                    SETTLE,
                    internal.fanout_threads().run_pass(
                        lane,
                        |_link| async { Ok::<_, ServiceError>(()) },
                        |_| PassAnswer::GoOn,
                    ),
                )
                .await
                .unwrap_or_else(|_| panic!("{lane:?}: the pass answers within {SETTLE:?}"));
                assert!(
                    matches!(ran, Err(ServiceError::Unavailable(_))),
                    "{lane:?}: a pass through an internal clone after the last owner dropped \
                     answers Unavailable: {ran:?}"
                );
            }
        }
    }
}
