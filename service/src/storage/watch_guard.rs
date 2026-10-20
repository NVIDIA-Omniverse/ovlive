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

// RAII guard that wraps a `txn_pool` checkout and guarantees UNWATCH is issued
// before the underlying connection is returned to the pool.
//
// Context (b.jov). A checkout hands out an `OwnedMutexGuard<Client>` (inside a
// `CheckedOutClient` since b.f4f). When that guard drops, the connection goes
// back to the pool **without** any `UNWATCH`. Any caller that `WATCH`ed and then
// lost the guard before a matching `UNWATCH` / successful `EXEC` (panic,
// `tokio::time::timeout`-triggered future drop, early `?` return) leaves stale
// WATCH state on the connection. The next consumer of that slot inherits the
// watch set, and their `EXEC` can return nil even though they watched nothing —
// surfacing as spurious `ServiceError::Aborted` ("concurrent write conflict").
//
// `WatchGuard` newtypes the `CheckedOutClient` and, on Drop, spawns a tokio task
// that performs `UNWATCH` **with the checkout moved into the spawned task**.
// Because the checkout is not released until the async body completes, the pool
// mutex — and, since b.f4f, the checkout's semaphore permit — remains held
// across the UNWATCH round-trip. This closes the ordering race that plain "spawn
// UNWATCH and drop the guard" would have:
//
//   - fred routes all commands for one `Client` through a single FIFO
//     mpsc → one TCP socket. Ordering on the wire is determined by mpsc
//     enqueue order.
//   - If we released the mutex before the UNWATCH was scheduled, a waiting
//     consumer's checkout could resume and enqueue its WATCH on the same mpsc
//     first — there is no tokio ordering guarantee between a just-spawned task
//     and a task resuming from a pending checkout.
//   - Holding the checkout across the UNWATCH await serializes UNWATCH
//     before any subsequent checkout, so the next consumer's WATCH
//     is strictly later on the mpsc. The permit's release is what makes that
//     true of a *waiting* consumer too, and not only of one that happens to
//     scan this slot: no waiter can even be handed an entitlement until this
//     task's `CheckedOutClient` has dropped.
//
// The `has_watched` flag avoids wasted UNWATCH round-trips on the happy
// path. Callers flip it to `true` (via [`WatchGuard::mark_watched`]) after a
// successful `watch(...)` call. It clears again either inside
// [`WatchGuard::unwatch_pinned`] (which disarms itself on `Ok` and stays armed
// on `Err`) or via an explicit [`WatchGuard::mark_unwatched`] after a
// successful `EXEC` (which implicitly clears WATCH on the Valkey side). If the
// guard drops with
// `has_watched = false` (no WATCH ever issued, or already cleared), Drop
// releases the mutex synchronously without spawning anything.
//
// Cluster-mode teardown routing (b.ji8). The single-socket premise above is
// true of a centralized deployment and FALSE of a clustered one. In cluster
// mode a fred `Client` holds one connection *per node*, and each command is
// routed by its hash slot. UNWATCH carries no arguments — fred builds it via
// `ok_cmd!(unwatch, Unwatch)` with an empty argument list — so the default
// `ClusterHash::FirstKey` policy finds no key to hash, `Command::cluster_hash()`
// yields `None`, and the router falls through to
// `writers.values_mut().next()` (fred-10.1.0 `src/router/mod.rs`), i.e. an
// ARBITRARY node. The UNWATCH therefore frequently never reaches the node that
// is actually holding the WATCH, and stale WATCH state accumulates on the
// shared `txn_pool` connections. The next borrower of that slot is the one
// punished: on a proxied deployment (Redis Enterprise / Azure Managed Redis)
// the residue surfaces as a `CROSSSLOT` naming a key the new borrower never
// mentioned — e.g. `SessionStore::create_once`'s leaked
// `le:path:{...}:names` WATCH reported against an unrelated `{object_id}`
// command. (The accused call sites are themselves slot-clean; the defect is
// entirely in teardown routing.)
//
// The fix: record the watched slot at `mark_watched` time and pin every
// UNWATCH to it via fred's per-command `Options::cluster_hash`, which
// `Options::apply` writes straight into `command.hasher` — the exact field
// the router reads. `mark_watched` *requires* at least one watched key — the
// signature is `(first_key, rest)` so an empty list cannot be written — and it
// `debug_assert!`s that all supplied keys hash to one slot (the `{object_id}`
// co-location assumption behind the three-key `WATCH` (header, stream,
// snapshot fence) in `LayerStore::update_layer_once_watch_attempt` becomes an
// executable invariant).
// The *commit* side needed no fixing then and still needs none, but the reason
// has changed and the old one-liner ("`Transaction::exec` already pins
// correctly via its own hash slot") is now only half the inventory (b.289
// sub-task 1). Two shapes exist:
//
//   - The non-WATCHed commit sites — two in `src/storage` (`layer`,
//     `session`; there were four until b.rej moved `external_values`' per-hash
//     blob store to a data-pool `EVALSHA` and b.v6d made `CreateLayer`'s
//     mapping `MULTI` the `scripts/create_layer_commit.lua` `EVALSHA`) plus
//     two in the checkpoint lane
//     (`checkpoint::session`) — still use fred's
//     `Transaction`, which pins correctly:
//     `Transaction::update_hash_slot` writes a `ClusterHash::Custom(slot)` onto
//     every queued command *including* the keyless `MULTI`, and `EXEC` goes to
//     the connection the transaction chose.
//   - The four WATCHed sites — `SessionStore::create_once`,
//     `LayerStore::update_layer_once_watch`, `LayerStore::write_restored_object`
//     and the DR residue wipe in `checkpoint::residue` — do not use
//     `Transaction`. They send `MULTI` / `CLIENT ID` / `EXEC` as slot-pinned
//     `CustomCommand`s inside a fred `Pipeline` (`storage::watched_txn`),
//     because a `Transaction` replays the whole `MULTI … EXEC` onto a
//     connection that never held the `WATCH`. Those keyless commands are
//     pinned by `pinned_options` + the hash on each `CustomCommand`, i.e. by
//     the same mechanism as the teardown below, not by `Transaction`.
//     `CLIENT ID` is itself a keyless command, at three sites there: two
//     baseline probes and one in-`MULTI` witness.
//
// The full per-command routing inventory lives in SDD §15.3.
//
// Two further fences keep it that way. `clippy.toml` lists fred's keyless
// `unwatch` under `disallowed-methods`, so the only call site that compiles is
// `unwatch_attempt` below, which carries the lone `#[allow]`. And teardown is
// bounded twice over: each attempt carries its own `Options::timeout`
// ([`UNWATCH_ATTEMPT_TIMEOUT`]) and the whole funnel runs under
// [`UNWATCH_FUNNEL_TIMEOUT`], with the retry living only at the `Drop` site.
//
// Teardown outcomes are three-way, not two (b.ji8 round 3). A failure whose
// error says the *connection itself* is gone (a lost connection's kind, or
// fred's routing miss when the router holds no connection; see
// `watch_died_with_the_connection`) is not a leak at all — a
// WATCH cannot outlive its connection — so it is logged at `debug` and left
// uncounted ([`TeardownOutcome::MootConnectionGone`]). Only a genuine give-up
// ticks `live_edit_storage_watch_unwatch_failures_total{site}`. That
// distinction is load-bearing for alerting: the counter is documented as
// "any non-zero value at site=drop is actionable" and `Canceled` is fred's
// most common error on a pool that sees failover, proxy recycles and ordinary
// reconnects, so counting connection churn would page on routine events.

use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::error::ErrorKind;
use fred::types::config::Options;
use fred::types::ClusterHash;
// `ClientLike` (for `with_options`) and `KeysInterface` (for `unwatch`) are
// imported inside `unwatch_attempt`, the single funnel through which every
// UNWATCH in this module is issued.
use tokio_util::task::TaskTracker;

use crate::observability::metrics::{SharedMetrics, UNWATCH_SITE_DROP, UNWATCH_SITE_EXPLICIT};
use crate::storage::checkout::CheckedOutClient;
use crate::storage::client::{FredErrOrigin, POOL_LABEL_TXN, UNWATCH_DRAIN_TIMEOUT};

/// Wall-clock ceiling on one whole teardown — first attempt plus, at the
/// `Drop` site, its retry (b.ji8 round 2).
///
/// Sized against `ValkeyClient::quit`'s [`UNWATCH_DRAIN_TIMEOUT`] (2 s,
/// `src/storage/client.rs`), which is the budget shutdown gives the
/// `TaskTracker` to drain every in-flight `Drop`-spawned UNWATCH before the
/// `txn` pool is closed. If a teardown can outlive that budget, `quit()`
/// abandons the drain and releases connections mid-UNWATCH — precisely the
/// outcome the b.k1r drain exists to prevent.
///
/// 1 500 ms covers both attempts — see [`UNWATCH_ATTEMPT_TIMEOUT`], which makes
/// "both attempts" a deterministic 1 300 ms rather than something inherited
/// from a config knob — and still leaves
/// [`UNWATCH_DRAIN_SCHEDULING_RESERVE`] of the drain budget for task
/// scheduling, guard drop and pool-token teardown. It is deliberately NOT
/// derived from `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`: raising that knob must
/// not be able to push a teardown past a shutdown budget it knows nothing
/// about. The `const` assertion below mechanizes the pairing.
pub(crate) const UNWATCH_FUNNEL_TIMEOUT: Duration = Duration::from_millis(1_500);

/// Per-attempt deadline, set on the `Options` every UNWATCH in this module
/// carries (b.ji8 round 3).
///
/// Without it each attempt inherits fred's client-wide
/// `default_command_timeout` (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, default
/// 5 000 ms), which is more than triple [`UNWATCH_FUNNEL_TIMEOUT`] — so at the
/// `Drop` site the funnel deadline would always fire *during attempt one* and
/// the retry the site exists for could never run. The retry was reachable only
/// under a test that reconfigured the command timeout down.
///
/// 650 ms makes attempt + retry (1 300 ms) fit inside the 1 500 ms funnel at
/// **any** configured command timeout, leaving 200 ms of slack for scheduling
/// between the two attempts. fred documents `Options::timeout` as more
/// cancellation-safe than wrapping the future in `tokio::time::timeout`, so
/// this is the per-attempt bound and the funnel timeout stays as the backstop
/// covering everything the per-command deadline cannot (a wedged fred router
/// task, a future that never gets polled).
const UNWATCH_ATTEMPT_TIMEOUT: Duration = Duration::from_millis(650);

/// How much of `quit()`'s UNWATCH drain budget must stay unclaimed by the
/// funnel: the drain also has to schedule the `Drop`-spawned task and, after
/// the funnel returns, release the mutex guard and the pool-use token.
const UNWATCH_DRAIN_SCHEDULING_RESERVE: Duration = Duration::from_millis(500);

// The two couplings above, mechanized (b.ji8 round 3). Both relate constants a
// reviewer has to hold in their head — one across two files — and both regress
// on a one-token edit, so they are checked by the compiler rather than by a
// comment. Failing here is not a mystery: the assertion message names the fix.
const _: () = assert!(
    2 * UNWATCH_ATTEMPT_TIMEOUT.as_millis() <= UNWATCH_FUNNEL_TIMEOUT.as_millis(),
    "UNWATCH_ATTEMPT_TIMEOUT * 2 (the Drop site's attempt plus its one retry) must fit inside \
     UNWATCH_FUNNEL_TIMEOUT, or the funnel deadline pre-empts the retry mid-flight and the \
     Drop site's second attempt can never run. Lower the per-attempt bound or raise the funnel \
     bound (and re-check the drain assertion below)."
);
const _: () = assert!(
    UNWATCH_FUNNEL_TIMEOUT.as_millis() + UNWATCH_DRAIN_SCHEDULING_RESERVE.as_millis()
        <= UNWATCH_DRAIN_TIMEOUT.as_millis(),
    "UNWATCH_FUNNEL_TIMEOUT plus UNWATCH_DRAIN_SCHEDULING_RESERVE must fit inside \
     ValkeyClient::quit's UNWATCH_DRAIN_TIMEOUT (src/storage/client.rs). A teardown able to \
     outlive the drain makes quit() give up waiting and close the txn pool while an UNWATCH is \
     still in flight, releasing connections with the stale WATCH state the UNWATCH exists to \
     clear. Lower the funnel bound, or raise the drain budget (and LIVE_EDIT_SHUTDOWN_* \
     above it) deliberately, and update SDD §15.3."
);

/// Errors that mean the connection carrying the WATCH is already gone.
///
/// Valkey scopes WATCH state to a connection, so a torn-down connection has no
/// WATCH left to clear: fred reporting [`ErrorKind::Canceled`] ("Connection
/// closed unexpectedly") or [`ErrorKind::IO`] is itself the evidence that the
/// residue this funnel exists to prevent cannot occur. So is fred's routing
/// miss ([`is_router_no_connection_error`](crate::storage::client::is_router_no_connection_error)):
/// the `UNWATCH` carries a cluster hash, never a named node, so the router
/// misses only when it holds no connection at all — the same rule as the
/// checkout probe's (`storage::txn_checkout`). That makes such a failure
/// *moot* rather than actionable — it is neither retried nor counted; see
/// [`TeardownOutcome::MootConnectionGone`].
///
/// [`ErrorKind::Timeout`] is deliberately excluded: a command that expired
/// without a reply leaves the connection's WATCH state *unknown*, which is
/// exactly the case worth retrying and worth counting.
fn watch_died_with_the_connection(err: &fred::error::Error) -> bool {
    matches!(err.kind(), ErrorKind::Canceled | ErrorKind::IO)
        || crate::storage::client::is_router_no_connection_error(err)
}

/// How one whole teardown ended (b.ji8 round 3).
///
/// The three arms exist because "did the UNWATCH return `Ok`?" is the wrong
/// question for both logging and alerting. A failure on a connection fred has
/// already torn down leaves nothing behind, and on a pool that sees failover,
/// managed-proxy recycles and ordinary reconnects it is the *common* failure —
/// folding it in with a genuine give-up would page an operator on routine
/// connection churn.
enum TeardownOutcome {
    /// The `UNWATCH` was acknowledged. Nothing to log, nothing to count.
    Cleared,
    /// The attempt failed, but with an error that says the connection carrying
    /// the WATCH is already gone ([`watch_died_with_the_connection`]: a lost
    /// connection's kind, or fred's routing miss) — so there was no residue to
    /// clear.
    /// Logged at `debug`; deliberately does NOT tick the failure counter.
    MootConnectionGone(fred::error::Error),
    /// The teardown ended with the WATCH's fate genuinely unknown. Logged at
    /// `error` and counted.
    GaveUp {
        err: fred::error::Error,
        /// Populated only when a retry happened, so `None` is itself evidence
        /// the site made exactly one attempt.
        first_err: Option<fred::error::Error>,
        /// `true` when [`UNWATCH_FUNNEL_TIMEOUT`] elapsed with a command still
        /// in flight, as opposed to an attempt returning an error. Changes what
        /// is true about the connection — see the `consequence` field on the
        /// terminal log.
        funnel_elapsed: bool,
    },
}

/// One `UNWATCH`, pinned to the recorded slot when we have one and bounded by
/// [`UNWATCH_ATTEMPT_TIMEOUT`].
///
/// `hash` is `None` only when no key was ever recorded — [`WatchGuard::mark_watched`]
/// always stores a `ClusterHash::Custom(slot)`, so `None` never means
/// "centralized deployment"; it means this guard has no outstanding WATCH to
/// route back to. Leaving `Options::cluster_hash` unset there restores fred's
/// default routing, which is the right (and harmless) call for a connection
/// with nothing to clear.
///
/// This is the ONLY function in the repo permitted to name fred's `unwatch` —
/// `clippy.toml` lists it under `disallowed-methods`, so every other call site
/// is a hard lint error and a future edit cannot reintroduce the keyless,
/// arbitrarily-routed teardown b.ji8 fixed.
#[allow(clippy::disallowed_methods)]
async fn unwatch_attempt(
    client: &Client,
    hash: Option<&ClusterHash>,
) -> Result<(), fred::error::Error> {
    use fred::interfaces::{ClientLike, KeysInterface};
    let options = Options {
        cluster_hash: hash.cloned(),
        // Per-attempt bound. Without it the attempt inherits
        // `default_command_timeout` and the funnel deadline fires mid-attempt,
        // making the Drop site's retry unreachable — see
        // `UNWATCH_ATTEMPT_TIMEOUT`.
        timeout: Some(UNWATCH_ATTEMPT_TIMEOUT),
        ..Default::default()
    };
    client.with_options(&options).unwatch().await
}

/// Issue `UNWATCH` pinned to the slot the WATCH was routed to, under an
/// explicit wall-clock bound.
///
/// `retry_once` is the whole retry policy (b.ji8 round 2). The explicit
/// call-site path passes `false`: it runs inline on the RPC request path, and
/// `Drop` — which still holds the pool mutex — *is* its retry, so a second
/// in-line attempt would only add latency to a request that is already
/// returning an error. The `Drop` path passes `true`, because that is the last
/// moment the process holds the connection. Either way a connection-teardown
/// error short-circuits the retry (see [`watch_died_with_the_connection`]),
/// each attempt is bounded by [`UNWATCH_ATTEMPT_TIMEOUT`], and the whole funnel
/// is capped by [`UNWATCH_FUNNEL_TIMEOUT`].
///
/// The `Err` return is deliberately kept for *every* non-`Cleared` outcome,
/// including the moot one: the explicit call site relies on `Err` to leave the
/// guard armed so `Drop` gets its own attempt while the pool mutex is still
/// held. What the outcome changes is the log level and whether the counter
/// ticks, not the caller's control flow.
async fn unwatch_pinned_on(
    client: &Client,
    hash: Option<&ClusterHash>,
    metrics: Option<&SharedMetrics>,
    site: &'static str,
    retry_once: bool,
) -> Result<(), fred::error::Error> {
    // Each attempt that times out is counted on
    // `valkey_errors_total{pool="txn", kind="command_timeout"}`. The synthetic
    // funnel-deadline error below is not: it is this function's own deadline,
    // not a command's. `UNWATCH` outside a transaction is not a memory-growing
    // write, so the store never refuses it `-OOM`: no store out-of-memory
    // signal to note on (b.pa1).
    let origin = FredErrOrigin::metrics(
        metrics,
        POOL_LABEL_TXN,
        crate::storage::store_oom::StoreOomSignal::unwired(),
    );
    let attempts = async {
        let Err(first_err) = unwatch_attempt(client, hash).await else {
            return TeardownOutcome::Cleared;
        };
        origin.observe_error(&first_err);
        // Checked before the `retry_once` split so BOTH sites classify it the
        // same way: a WATCH that died with its connection is moot wherever the
        // teardown was issued from.
        if watch_died_with_the_connection(&first_err) {
            return TeardownOutcome::MootConnectionGone(first_err);
        }
        if !retry_once {
            return TeardownOutcome::GaveUp {
                err: first_err,
                first_err: None,
                funnel_elapsed: false,
            };
        }
        tracing::debug!(
            error = %first_err,
            kind = ?first_err.kind(),
            site,
            "WatchGuard: pinned UNWATCH failed; retrying once"
        );
        let retried = unwatch_attempt(client, hash).await;
        if let Err(e) = &retried {
            origin.observe_error(e);
        }
        match retried {
            Ok(()) => TeardownOutcome::Cleared,
            Err(e) if watch_died_with_the_connection(&e) => TeardownOutcome::MootConnectionGone(e),
            Err(e) => TeardownOutcome::GaveUp {
                err: e,
                first_err: Some(first_err),
                funnel_elapsed: false,
            },
        }
    };

    let outcome = match tokio::time::timeout(UNWATCH_FUNNEL_TIMEOUT, attempts).await {
        Ok(outcome) => outcome,
        Err(_elapsed) => TeardownOutcome::GaveUp {
            err: fred::error::Error::new(
                ErrorKind::Timeout,
                "WatchGuard UNWATCH exceeded its funnel deadline",
            ),
            first_err: None,
            funnel_elapsed: true,
        },
    };

    match outcome {
        TeardownOutcome::Cleared => Ok(()),
        TeardownOutcome::MootConnectionGone(err) => {
            tracing::debug!(
                error = %err,
                kind = ?err.kind(),
                site,
                "WatchGuard: pinned UNWATCH failed because the connection carrying the WATCH is \
                 already gone; Valkey scopes WATCH state to a connection, so there is nothing \
                 left to clear and nothing for the next borrower to inherit"
            );
            Err(err)
        }
        TeardownOutcome::GaveUp {
            err,
            first_err,
            funnel_elapsed,
        } => {
            // What is actually true depends on the site and on how the teardown
            // ended, so it is carried as a field rather than baked into a
            // message that would be false at two of the three combinations
            // (b.ji8 round 3). The `error!` message itself stays constant so
            // operators keep one thing to grep for.
            let consequence = match (site, funnel_elapsed) {
                (UNWATCH_SITE_EXPLICIT, _) => {
                    "no connection has been released: the guard stays armed on Err, so Drop \
                     retries this teardown while it still holds the pool mutex. Actionable only \
                     if site=\"drop\" also ticks for the same connection"
                }
                (_, true) => {
                    "the funnel deadline elapsed with the UNWATCH still in flight. If fred had \
                     already written it, it lands ahead of the next borrower's WATCH on that \
                     connection's FIFO; if it was still queued in fred, fred never writes it now \
                     that the funnel gave up on it, and the pooled connection is released with \
                     stale WATCH state that only the next checkout's hygiene probe (DISCARD + \
                     UNWATCH) clears. Either way the process no longer waits for it, so the \
                     ordering guarantee this guard exists to provide is gone"
                }
                (_, false) => {
                    "the pooled connection is being released with stale WATCH state, which will \
                     surface as a spurious EXEC=nil (centralized) or CROSSSLOT (clustered) on \
                     the next borrower"
                }
            };
            tracing::error!(
                error = %err,
                kind = ?err.kind(),
                first_error = ?first_err,
                site,
                funnel_elapsed,
                funnel_timeout_ms = UNWATCH_FUNNEL_TIMEOUT.as_millis() as u64,
                attempt_timeout_ms = UNWATCH_ATTEMPT_TIMEOUT.as_millis() as u64,
                consequence,
                "WatchGuard: pinned UNWATCH gave up"
            );
            if let Some(metrics) = metrics {
                metrics.inc_storage_watch_unwatch_failure(site);
            }
            Err(err)
        }
    }
}

/// RAII wrapper around a fred `ExclusivePool`-acquired `Client` that
/// UNWATCHes on drop when a WATCH was issued. See module-level docs.
///
/// Construction:
///
/// ```ignore
/// let checkout = self.client.acquire_txn().await;
/// let mut client = WatchGuard::new(checkout, self.client.watch_unwatch_tracker())
///     .with_metrics(self.client.metrics_snapshot());
/// client.watch::<&str>(&key).await?;
/// client.mark_watched(key.as_str(), &[]); // records the slot for teardown
/// // ... MULTI/EXEC ...
/// client.mark_unwatched(); // after successful EXEC
///
/// // Early-return paths UNWATCH through the pinned funnel. It disarms itself
/// // on success and stays armed on failure, so Drop still gets a shot:
/// let _ = client.unwatch_pinned().await;
/// ```
///
/// The [`CheckedOutClient`] carries the slot, the `valkey_pool_in_use{pool}`
/// gauge token (b.3z7 H-obs-3) and the checkout's semaphore permit (b.f4f) in
/// one value. All three must be released at the same point the underlying
/// `OwnedMutexGuard<Client>` is — so the Drop impl below moves the whole
/// checkout into the spawned UNWATCH task whenever UNWATCH is needed, and drops
/// it synchronously on the fast/no-watch path.
///
/// That the permit rides along is load-bearing, not incidental: it means the
/// `txn_pool` entitlement is returned only after UNWATCH has completed and the
/// mutex has been released, so the next waiter's scan cannot wake into a slot
/// this guard is still holding. `WatchGuard` is txn-pool-only, so the subscriber
/// pool has no deferred release at all.
pub struct WatchGuard {
    /// Slot + gauge token + permit, in one value whose drop order releases them
    /// in that sequence. Moved into the spawned UNWATCH task when
    /// `has_watched = true` so all three track the *actual* mutex-release point,
    /// not the synchronous return of `Drop::drop`.
    checkout: Option<CheckedOutClient>,
    /// Shared task tracker for the Drop-spawned UNWATCH cleanup task
    /// (b.k1r H-stor-2). `ValkeyClient::quit` waits on this tracker before
    /// quitting the txn pool so shutdown cannot tear the runtime down
    /// mid-`unwatch().await` and leave stale WATCH state on a connection
    /// that was about to be closed. `None` is accepted for test-only
    /// constructions that do not go through a `ValkeyClient`.
    unwatch_tracker: Option<Arc<TaskTracker>>,
    has_watched: bool,
    /// Cluster slot the outstanding WATCH was routed to, recorded by
    /// [`mark_watched`] from the watched key(s) (b.ji8). Every UNWATCH this
    /// guard issues — explicit or `Drop`-time — is pinned to it, because
    /// UNWATCH carries no key of its own and would otherwise be routed to an
    /// arbitrary cluster node. `None` means no WATCH has been recorded yet.
    watched_hash: Option<ClusterHash>,
    /// Optional metrics handle used to tick
    /// `live_edit_storage_watch_unwatch_failures_total` when a pinned UNWATCH
    /// fails twice. `None` on test constructions and before
    /// `install_observability_hooks` has run.
    metrics: Option<SharedMetrics>,
}

impl WatchGuard {
    /// Wrap a freshly-acquired exclusive-pool checkout (from
    /// [`ValkeyClient::acquire_txn`]).
    ///
    /// The guard is constructed with `has_watched = false` — callers must
    /// explicitly flip the flag via [`mark_watched`] after a successful
    /// `client.watch(...)` call so that an early return before WATCH never
    /// triggers a redundant UNWATCH round-trip.
    ///
    /// `unwatch_tracker` is the shared tracker from
    /// [`ValkeyClient::watch_unwatch_tracker`] (b.k1r H-stor-2). Spawning
    /// the Drop-time UNWATCH on that tracker lets
    /// [`ValkeyClient::quit`] await it before closing the txn pool,
    /// eliminating the runtime-teardown / mid-UNWATCH race that otherwise
    /// leaves stale WATCH state on the dropped connection.
    pub fn new(checkout: CheckedOutClient, unwatch_tracker: Arc<TaskTracker>) -> Self {
        Self {
            checkout: Some(checkout),
            unwatch_tracker: Some(unwatch_tracker),
            has_watched: false,
            watched_hash: None,
            metrics: None,
        }
    }

    /// Attach the process metrics handle (typically
    /// [`ValkeyClient::metrics_snapshot`]) so a doubly-failed UNWATCH ticks
    /// `live_edit_storage_watch_unwatch_failures_total{site}`. Optional: a
    /// guard constructed without it still runs the same retry policy and still
    /// logs at `error`.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Option<SharedMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Mark the guarded connection as having an outstanding WATCH, recording
    /// the cluster slot the WATCH landed on. Drop will spawn a slot-pinned
    /// UNWATCH cleanup task when this is set.
    ///
    /// `first_key` plus `rest` must be exactly the keys passed to the
    /// preceding `watch(...)` call. Taking them here (rather than a bare flag)
    /// is the compiler-enforced fence from b.ji8: a new WATCH site cannot
    /// forget to record the slot its UNWATCH has to be routed back to.
    ///
    /// The key list is split into a mandatory head and a (possibly empty) tail
    /// rather than taken as one `&[&str]` so that "no keys" is
    /// *unrepresentable* — a `&[]` would arm the guard with `watched_hash =
    /// None`, silently restoring the unpinned teardown this whole fix removes,
    /// and the single-slot `debug_assert!` below would pass vacuously on it.
    /// Single-key sites therefore read `mark_watched(key, &[])`.
    ///
    /// All supplied keys must hash to a single slot — a WATCH spanning slots
    /// is not expressible as one command in cluster mode anyway, and no
    /// single-slot UNWATCH could undo it. The `debug_assert!` pins that
    /// requirement (e.g. the header, stream and snapshot-fence keys in
    /// `LayerStore::update_layer_once_watch_attempt`, co-located by the
    /// `{object_id}` hash tag) as an executable invariant in debug and test
    /// builds.
    pub fn mark_watched(&mut self, first_key: &str, rest: &[&str]) {
        let hash = ClusterHash::from(first_key);
        debug_assert!(
            rest.iter().all(|key| ClusterHash::from(*key) == hash),
            "WatchGuard::mark_watched: watched keys span multiple cluster slots \
             (hash tags must co-locate them): first={first_key:?}, rest={rest:?}"
        );
        // Cross-call companion to the check above (b.ji8 round 3). This method
        // overwrites `watched_hash` unconditionally, and the guard records ONE
        // slot, so a second `watch(...)` on a different slot against the same
        // guard would silently strand the first WATCH: the overwrite discards
        // the only route back to the node holding it, and every teardown from
        // then on is pinned to the second slot. That is b.ji8's defect
        // re-created from inside the fix. The single-slot rule is therefore a
        // property of the guard's whole lifetime, not of one call — a call site
        // that legitimately re-WATCHes must `mark_unwatched()` (post-`EXEC`) or
        // succeed at `unwatch_pinned()` first, both of which clear the slot.
        debug_assert!(
            self.watched_hash
                .as_ref()
                .is_none_or(|outstanding| *outstanding == hash),
            "WatchGuard::mark_watched: this guard already has an outstanding WATCH on \
             {outstanding:?} and is now being marked for {hash:?} (first={first_key:?}). One \
             guard records one slot, so the earlier WATCH would be stranded on a node no \
             later UNWATCH can reach. Tear the first WATCH down (mark_unwatched after EXEC, or \
             a successful unwatch_pinned) before watching a different slot.",
            outstanding = self.watched_hash,
        );
        self.has_watched = true;
        self.watched_hash = Some(hash);
    }

    /// Clear the "needs UNWATCH" flag after a successful `EXEC`, which
    /// implicitly clears WATCH on the Valkey side.
    ///
    /// This is the *post-EXEC* disarm only. Explicit teardown disarms itself —
    /// [`Self::unwatch_pinned`] clears the flag on `Ok` and deliberately keeps
    /// it set on `Err`, which is what gives `Drop` a second chance to clear the
    /// connection while the pool mutex is still held. Do not pair a manual
    /// `mark_unwatched()` with `unwatch_pinned()`; doing so is how a failed
    /// teardown gets silently disarmed.
    pub fn mark_unwatched(&mut self) {
        self.has_watched = false;
        self.watched_hash = None;
    }

    /// Issue one `UNWATCH` pinned to the slot recorded by
    /// [`Self::mark_watched`], disarming the guard on success. This is the only
    /// teardown API on `WatchGuard` — never reach through `Deref` to fred's
    /// `Client::unwatch()`, which is routed to an arbitrary cluster node
    /// (b.ji8; see module docs, and `clippy.toml`, which makes that a lint
    /// error).
    ///
    /// No retry here, by design (b.ji8 round 2): this runs inline on the RPC
    /// request path, and `Drop` — still holding the pool mutex — is the retry.
    /// On `Err` the guard stays armed precisely so that happens; call sites
    /// therefore need no disarm bookkeeping of their own and can simply write
    /// `let _ = client.unwatch_pinned().await;`.
    ///
    /// A guard that was never armed returns `Ok(())` without touching the
    /// connection — mirroring `Drop`'s `has_watched` fast path. That is not
    /// just an optimisation (b.ji8 round 3): an unarmed guard has
    /// `watched_hash = None`, so falling through would issue an UNWATCH with no
    /// `cluster_hash` — the keyless, arbitrarily-routed teardown this whole
    /// module exists to remove — against a connection with nothing to clear.
    pub async fn unwatch_pinned(&mut self) -> Result<(), fred::error::Error> {
        if !self.has_watched {
            return Ok(());
        }
        let result = unwatch_pinned_on(
            self.checkout
                .as_deref()
                .expect("WatchGuard::unwatch_pinned after Drop"),
            self.watched_hash.as_ref(),
            self.metrics.as_ref(),
            UNWATCH_SITE_EXPLICIT,
            false,
        )
        .await;
        if result.is_ok() {
            self.mark_unwatched();
        }
        result
    }
}

impl Deref for WatchGuard {
    type Target = Client;

    fn deref(&self) -> &Self::Target {
        // `checkout` is only `None` while Drop is in progress. Any deref
        // reaching this point must observe `Some(..)`.
        self.checkout
            .as_deref()
            .expect("WatchGuard dereferenced after Drop")
    }
}

impl Drop for WatchGuard {
    fn drop(&mut self) {
        let Some(checkout) = self.checkout.take() else {
            return;
        };
        let unwatch_tracker = self.unwatch_tracker.take();
        let watched_hash = self.watched_hash.take();
        let metrics = self.metrics.take();
        if !self.has_watched {
            // No WATCH was issued (or the caller already UNWATCHed /
            // committed). Drop the checkout synchronously — the mutex releases
            // immediately with a clean connection, then the gauge ticks down,
            // then the permit frees, in that order.
            drop(checkout);
            drop(unwatch_tracker);
            // `watched_hash` / `metrics` fall out of scope here; neither
            // carries a side effect worth an explicit `drop`.
            return;
        }

        // Move the guard INTO the spawned task so the ExclusivePool mutex
        // is held until UNWATCH completes. This is what serializes the
        // UNWATCH before any subsequent `acquire()` on this slot — see
        // module-level docs.
        //
        // `try_current()` handles the rare case where the guard drops
        // outside a tokio runtime (e.g. during late-stage shutdown after
        // the runtime has exited). In that case we drop the guard
        // synchronously — stale WATCH will be cleared by connection
        // teardown anyway.
        //
        // The gauge token and the checkout's semaphore permit move into the
        // spawned task inside the same `CheckedOutClient`, so
        // `valkey_pool_in_use{pool}` decrements and the `txn_pool` entitlement
        // is returned at the same point the mutex is released (after UNWATCH
        // completes), not at the synchronous return of this `Drop::drop`. The
        // permit ordering is the load-bearing half: releasing it early would
        // wake the next waiter into a scan of a pool where this slot is still
        // locked (b.f4f).
        //
        // The spawn goes through `TaskTracker::spawn` (b.k1r H-stor-2) so
        // `ValkeyClient::quit` awaits the task before closing the txn pool.
        // A bare `tokio::spawn` would race shutdown: the runtime can tear
        // down mid-`unwatch().await`, leaving stale WATCH on the
        // about-to-be-closed connection. If the tracker is absent (test
        // harnesses constructing a WatchGuard without a ValkeyClient) or
        // has already been closed (quit in progress), fall back to plain
        // `tokio::spawn` — the fallback preserves the pre-b.k1r behavior
        // rather than dropping UNWATCH entirely.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let body = async move {
                    // Pinned to the WATCHed slot (b.ji8) — a bare
                    // `guard.unwatch()` is keyless and would be routed to an
                    // arbitrary cluster node, leaving the WATCH standing on
                    // the node that actually holds it. Failure is logged at
                    // `error` and counted inside the funnel; there is nothing
                    // further to do here, since this is the last point at
                    // which we hold the connection.
                    let _ = unwatch_pinned_on(
                        &checkout,
                        watched_hash.as_ref(),
                        metrics.as_ref(),
                        UNWATCH_SITE_DROP,
                        // Retry only here: this is the last moment the process
                        // holds the connection, so there is no later site to
                        // defer to. `UNWATCH_ATTEMPT_TIMEOUT` is what makes the
                        // retry actually reachable (round 3) and
                        // `UNWATCH_FUNNEL_TIMEOUT` is what keeps the pair inside
                        // `quit()`'s `UNWATCH_DRAIN_TIMEOUT` — both mechanized
                        // by the `const` assertions at the top of this module.
                        true,
                    )
                    .await;
                    // `checkout` drops here — releasing the ExclusivePool mutex
                    // *after* UNWATCH has been queued and processed, then
                    // ticking `valkey_pool_in_use` down, then returning the
                    // permit, in that field order.
                    drop(checkout);
                };
                match unwatch_tracker {
                    Some(tracker) if !tracker.is_closed() => {
                        tracker.spawn(body);
                    }
                    _ => {
                        handle.spawn(body);
                    }
                }
            }
            Err(_) => {
                // No runtime — e.g. dropping during test-harness teardown
                // or after `Runtime::shutdown_*`. Best-effort: let the
                // checkout drop synchronously. Reconnect / process exit will
                // clear any stale WATCH. The gauge token and the permit drop
                // with it, so neither gets stuck over-counted at shutdown.
                tracing::debug!("WatchGuard: no tokio runtime available on drop; skipping UNWATCH");
                drop(checkout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The teardown's three-way verdict hangs on [`watch_died_with_the_connection`]:
    /// a `true` makes a failed `UNWATCH` moot (DEBUG, uncounted, no `Drop`
    /// retry), a `false` a counted give-up at ERROR. Pure, so driven here with
    /// fred errors built directly; the integration tests in
    /// `tests/integration/storage/b_ji8_tests.rs` reach the `Timeout` arm through
    /// a real closed connection.
    ///
    /// The routing miss is the row the vendored fred's change 9 made reachable
    /// on the `txn` pool: a fred `Transaction` cut partway closes its connection
    /// and defers the reconnect, so an `UNWATCH` routed ahead of that reconnect
    /// finds no connection and fails "Failed to route command." — in either
    /// kind `storage::client::is_router_no_connection_error` accepts. No
    /// connection means no WATCH left to clear.
    #[test]
    fn watch_died_with_the_connection_for_a_lost_connection_or_a_routing_miss_only() {
        let error = |kind: ErrorKind, details: &'static str| fred::error::Error::new(kind, details);

        for (kind, label) in [(ErrorKind::Canceled, "Canceled"), (ErrorKind::IO, "IO")] {
            assert!(
                watch_died_with_the_connection(&error(kind, "Connection closed unexpectedly")),
                "{label} means the connection carrying the WATCH is gone: moot"
            );
        }
        for (kind, label) in [
            (ErrorKind::Unknown, "Unknown"),
            (ErrorKind::Routing, "Routing"),
        ] {
            let miss = error(kind, "Failed to route command.");
            assert!(
                crate::storage::client::is_router_no_connection_error(&miss),
                "precondition: the fixture is the routing miss the shared classifier names \
                 ({label})"
            );
            assert!(
                watch_died_with_the_connection(&miss),
                "a {label} routing miss means the router holds no connection, so no WATCH is \
                 left: moot, not a counted give-up with an ERROR line"
            );
        }

        assert!(
            !watch_died_with_the_connection(&error(ErrorKind::Timeout, "Request timed out.")),
            "a Timeout leaves the WATCH's fate unknown: retried at Drop and counted, never moot"
        );
        assert!(
            !watch_died_with_the_connection(&error(
                ErrorKind::Unknown,
                "Expected OK, found Queued."
            )),
            "only the routing miss's text makes an Unknown moot: an UNWATCH answered with \
             anything but OK (a QUEUED inside an open MULTI) is a counted give-up"
        );
    }
}
