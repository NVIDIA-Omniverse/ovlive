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

// Non-replayable WATCHed transactions (b.bhb).
//
// # The defect this module exists to remove
//
// Valkey scopes a `WATCH` to the *connection* that issued it, and clears the
// watch set the moment that connection goes away. Both WATCHed commit paths —
// `LayerStore::update_layer_once_watch`'s structural slow path and
// `SessionStore::create_once` — used to send the commit through fred's
// `Transaction` (`client.multi()` … `exec()`), which is a SEPARATE router
// command from the `WATCH`. fred's transaction router replays that whole
// `MULTI … EXEC` on a fresh connection after any reconnect
// (`ConnectionConfig::max_command_attempts`, default 3) and on a fresh node
// after a `MOVED` (`max_redirections`, default 5) — see
// `fred-10.1.0/src/router/transactions.rs:115-131`. A replayed `EXEC` runs
// **unwatched**: Valkey commits it unconditionally, so the optimistic-locking
// contract both call sites are built on is silently void. No pre-`EXEC` check
// in our code can observe that, because the rewrite happens after the commands
// are handed to the router.
//
// The damage differs by site and neither is benign:
//
// - `create_once` loses a duplicate-name check (two sessions, one name).
// - `update_layer_once_watch` produces a TORN commit: its `HSET`s land while the
//   `XADD {seq}-0` is refused by the winner's entry, so materialized state and
//   the live relay stream diverge.
//
// Every WATCHed transaction now commits through this module: those two paths,
// plus the DR restore's residue wipe (`checkpoint::residue`) and
// `LayerStore::write_restored_object`.
//
// # The mechanism
//
// Three layers, cheapest first. Together they turn "we hope the WATCH held"
// into "the server named the connection that ran the transaction".
//
// **L1 — nothing issued under a WATCH is *re-issued* after a failed write.**
// The `txn_pool` is built from its own fred `Builder` with
// `max_command_attempts = 1` and `max_redirections = 0`
// (`ValkeyClient::connect`), and the commit is sent as ONE fred `Pipeline` of
// regular commands — `MULTI`, the queued mutations, `CLIENT ID`, `EXEC` —
// rather than through fred's `Transaction`.
//
// L1 buys exactly two things, and it is worth being precise about them because
// the third thing it looks like it buys, it does not:
//
// 1. *A command whose socket write FAILED is never re-attempted on a new
//    socket.* `router/commands.rs::write_command` calls
//    `Command::decr_check_attempted()` (line 111) before writing, so at a
//    budget of 1 the command reaches the socket with `attempts_remaining == 0`;
//    when the write errors, `finish_or_retry_command` (line 91-97) responds to
//    the caller instead of calling `router.retry_command`. The connection is
//    then dropped and the reconnect DEFERRED through the router channel
//    (`router/utils.rs::defer_reconnection`, line 118-149), so the later frames
//    of the same pipeline hit `route() -> None` (line 142-155) and error too —
//    no frame of a partially-written pipeline can land on the replacement
//    socket.
// 2. *fred's `Transaction` router never replays the whole `MULTI … EXEC`.*
//    `router/transactions.rs::send` runs its own explicit replay loop
//    (`attempted += 1; if attempted > max_attempts`) on top of the per-command
//    budget. Sending the commit as a `Pipeline` of regular commands removes
//    that loop from the picture entirely; the attempt cap removes the
//    per-command half.
//
// **What L1 does NOT prevent: a pipeline that has not been written yet being
// written to a replacement socket.** fred's router is a single task whose
// `tokio::select!` is `biased` with socket reads FIRST
// (`router/commands.rs::read_or_write`, line 463-478). When the peer closes the
// socket, the reader arm resolves before the command-channel arm, and
// `router/utils.rs::process_response`'s `None` arm (line 286-290) runs
// `drop_connection` + `defer_reconnection`, which pushes a
// `RouterCommand::Reconnect` onto the same channel our pipeline is queued on.
// If that message is AHEAD of the pipeline, the router reconnects first and
// then writes the pipeline onto the NEW connection: the write succeeds, so no
// attempt is ever "charged" and `max_command_attempts = 1` never engages. The
// Test Writer reproduced exactly this — killing the WATCH's connection in the
// pre-dispatch hook *without* waiting for the reconnect still ran the whole
// `MULTI … EXEC`, and the server answered `ERR The ID specified in XADD is
// equal or smaller than the target stream top item`: an unwatched commit, which
// L3 then caught. That window closes when the caller gives up: once the
// transaction's timeout fires or its future is dropped, fred writes no frame
// of the pipeline it has not already written, and when it had written the
// `MULTI` but not the `EXEC` it follows with a `DISCARD` on that connection
// (the vendored fred's change 9). So an unwatched commit is written, if at
// all, inside the transaction's own timeout.
//
// `max_command_attempts` must be **1**, never 0: `write_command` calls
// `Command::decr_check_attempted()` before writing ANY regular command and
// errors at 0 (`fred-10.1.0/src/protocol/command.rs:1776-1783`), which would
// break every ordinary command on the pool (the `WATCH` itself, the reads under
// it, the checkout hygiene probe). Neither zero is expressible per command
// either: `Command::inherit_options` rewrites `attempts_remaining == 0` and
// `redirections_remaining == 0` back to the pool defaults
// (`command.rs:1882-1888`), which is why the ceiling lives on the pool config
// and the per-command [`pinned_options`] only re-states it.
//
// **A `MULTI … EXEC` is one router command and is never split across router
// commands.** Everything above rests on the whole transaction reaching fred's
// router as ONE `RouterCommand::Pipeline`, written in one router turn. Split
// into chunks — to bound the client memory a large transaction's queued
// commands hold, say — each chunk becomes a router command of its own, and two
// guarantees go:
//
// - *A reconnect can land between two chunks.* The router may process a queued
//   `RouterCommand::Reconnect` (the window described above) after one chunk and
//   before the next, and then writes the next chunk onto a fresh connection
//   with no `MULTI` open: every command in it executes at once and
//   unconditionally, and its `EXEC` answers `ERR EXEC without MULTI`. On
//   `update_layer` that is a torn commit — rows land with no `XADD` fence — and
//   on a restore it is the generation mix the header-absence fence exists to
//   prevent. L3's witness would sit only in the last chunk, after the earlier
//   rows had already landed.
// - *A credential-refresh `AUTH` could interleave inside the `MULTI`.* It
//   cannot today only because the transaction is one router command.
//
// Bounded chunks are safe in exactly two shapes: complete, independent
// transactions, each its own `MULTI … EXEC` with its own guard; and pure reads
// outside the `MULTI` that complete before it opens — the keyspace type probe
// (`LayerStore::probe_keyspace_types`) and the external-value pre-image read
// (`LayerStore::read_ext_pre_image`).
//
// **L2 — connection identity is re-checked before dispatch.** See
// [`dispatch_verdict`]: two local reads, no round trip, taken right after
// `mark_watched` and re-taken immediately before the pipeline is sent —
// [`WatchFingerprint`] over fred's backchannel connection-id map, plus
// `ClientLike::is_connected`. Both are *narrowing* checks, not a fence: neither
// flips in the window described above. The connection-id map is only rewritten
// by `Connections::initialize` / `Router::sync_cluster`
// (`router/connections.rs:168`, `router/mod.rs:322`) — i.e. once a reconnect has
// COMPLETED — and `disconnect`'s `backchannel.remove_connection_id`
// (`router/utils.rs:224`) is a no-op in fred 10.1.0 (`backchannel.rs:109-111`
// calls `.get()` where `.remove()` was meant), so a dropped-but-not-yet-replaced
// connection still reads as present with its old id. `is_connected` reads
// `inner.state`, which becomes `Connecting` only inside `reconnect_once`
// (`router/utils.rs:187`) — after the router has dequeued the `Reconnect` — and
// is back to `Connected` before the pipeline is written. So the two together
// catch a reconnect that finished, or one that is mid-handshake right now, and
// nothing else.
//
// L2 is also the layer that MUST NOT fail closed when it simply has no data.
// fred's connection-id map is populated from a `CLIENT ID` issued during the
// handshake, and `Connection::cache_connection_id`
// (`fred-10.1.0/src/protocol/connection.rs:541-551`) swallows any non-integer
// reply into `self.id = None` while still returning `Ok`; `connection_ids()`
// (`router/connections.rs:297-322`) then inserts only `Some` ids. On a store
// where `CLIENT ID` is ACL-denied or rejected by a proxy, that map is
// **permanently empty on a perfectly healthy connection** — so a check that
// read "no id" as "the connection is gone" would refuse every dispatch and
// take `CreateSession` and every structural `UpdateLayer` to retry exhaustion,
// forever. [`WatchFingerprint::Unnamed`] is that state, and it is
// *un-comparable*, not negative: L2 is skipped, the attempt dispatches, and
// `fingerprint_unavailable` is ticked so a deployment running without L2 is
// visible rather than silent. Only `is_connected() == false` refuses.
//
// **L3 — the transaction names its own connection, against a baseline taken on
// the WATCH's own connection.** `CLIENT ID` is queued as the LAST command
// inside the `MULTI`, so the `EXEC` array's last element is the id of the
// connection that actually executed the transaction. That witness is compared
// against a [`WitnessBaseline`] — a second `CLIENT ID`, read from inside a
// throwaway `MULTI … EXEC` **pipelined ahead of the `WATCH` itself**
// ([`watch_with_baseline`]), i.e. sampled from the same id space, in the same
// in-transaction context, on the same connection. It is deliberately NOT
// compared against L2's handshake-time fingerprint: on a directly-addressed
// Valkey the two coincide, but a proxied managed store (Azure Managed Redis,
// Redis Enterprise) may answer `CLIENT ID` from its own id space, and then
// every ordinary commit would read as a proven unwatched commit — compensating
// `create_session` to retry exhaustion and paging on `committed_unwatched`
// forever. Comparing like against like removes that failure mode by
// construction.
//
// A store that *refuses* `CLIENT ID` inside `MULTI` (the probe's `EXEC` comes
// back `EXECABORT`) or hands out a different id per command (the probe's two
// ids disagree) disarms L3 for that attempt: the baseline is
// [`WitnessBaseline::Unavailable`], the commit pipeline then queues no
// `CLIENT ID` at all — queueing one would make every real commit `EXECABORT`
// too — the witness is `None`, nothing is ever accused, and
// `witness_unavailable` is ticked. The service runs on L1+L2 there, which is
// exactly what the metric says.
//
// A mismatch against a `Trusted` baseline is *proof* of an unwatched commit,
// which the two call sites answer with compensation (`create_once`) or the
// retirement re-check (`update_layer_once_watch`) — never by reporting the commit as
// conditional. This is the only layer that closes the residual, and it closes
// it after the fact.
//
// # Why every command carries an explicit cluster hash
//
// `MULTI`, `EXEC` and `CLIENT ID` are keyless. Left to fred's default
// `ClusterHash::FirstKey` policy they hash nothing, and the cluster router
// falls through to `writers.values_mut().next()` — an ARBITRARY node (the same
// defect b.ji8 fixed for the keyless `UNWATCH` teardown). Pinning them to the
// watched key's slot is what makes the pipeline one transaction on one node.
// The baseline probe in [`watch_with_baseline`] is pinned for a second reason
// on top of that one: it only means anything if it samples the identity of the
// connection the `WATCH` immediately after it lands on.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use fred::clients::{Client, Pipeline, WithOptions};
use fred::interfaces::{ClientInterface, ClientLike, ClusterInterface};
use fred::types::config::Options;
use fred::types::{ClusterHash, CustomCommand, Value};

use crate::storage::client::FredErrOrigin;
use crate::storage::log_gate::LogGate;

/// Per-command `Options` every command of a watched transaction carries.
///
/// `cluster_hash` pins the keyless `MULTI` / `CLIENT ID` / `EXEC` to the watched
/// key's slot (see module docs). `max_attempts: Some(1)` re-states the txn
/// pool's `max_command_attempts = 1` ceiling at the call site so a future
/// pool-config edit cannot silently re-enable replay on the one path where a
/// replay is a correctness bug. `Some(0)` would be a no-op — `inherit_options`
/// rewrites it to the pool default (`fred-10.1.0/src/protocol/command.rs:1883`)
/// — and would break the command outright if it survived
/// (`decr_check_attempted` errors at 0).
///
/// `timeout` is left to the configured command timeout here; the one
/// transaction that needs longer, the WATCH-route `UpdateLayer` commit, adds
/// its size-scaled timeout on top ([`WatchedTxn::open_with_timeout`]).
pub(crate) fn pinned_options(hash: &ClusterHash) -> Options {
    Options {
        cluster_hash: Some(hash.clone()),
        max_attempts: Some(1),
        ..Default::default()
    }
}

/// Identity of the connection a `WATCH` was issued on, read from fred's local
/// backchannel map (`ClientInterface::connection_ids`,
/// `fred-10.1.0/src/commands/interfaces/client.rs:41-43` — a `HashMap` clone,
/// no round trip). fred rebuilds that map at every handshake
/// (`router/connections.rs:168`, `router/mod.rs:322`), so a completed reconnect
/// changes it.
///
/// Deliberately NOT a `CLIENT ID` round trip: that command is keyless (see the
/// module docs on routing) and it would widen the very window it measures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WatchFingerprint {
    /// fred's recorded connection id for the connection that owns the watched
    /// slot. Exact: changes only when *that* connection is re-established.
    Slot(i64),
    /// The slot's owner could not be named — fred's cached cluster routing view
    /// was unavailable on a multi-node client. Falls back to a digest of the
    /// whole connection-id map, which moves on any reconnect anywhere.
    /// Conservative: costs at most an extra `Aborted` retry, never a missed
    /// detection.
    AnyConnection(u64),
    /// fred holds **no id** for the connection this transaction routes to —
    /// either the map is empty, or the routing view named a server that has no
    /// entry in it.
    ///
    /// This is "L2 has no data", NOT "the connection is gone" (that is
    /// `is_connected() == false`, checked separately). fred learns a
    /// connection's id from a handshake `CLIENT ID` whose failure
    /// `Connection::cache_connection_id`
    /// (`fred-10.1.0/src/protocol/connection.rs:541-551`) swallows into
    /// `self.id = None`, and `connection_ids()`
    /// (`router/connections.rs:297-322`) only inserts `Some` ids — so a store
    /// that ACL-denies or proxy-rejects `CLIENT ID` produces this variant on
    /// every healthy connection, forever. Comparing it as "not held" would
    /// refuse 100% of dispatches on such a store. It is therefore treated as
    /// **un-comparable** by [`dispatch_verdict`]: L2 is skipped, the commit
    /// dispatches, and the condition is counted. L3 is unaffected — its
    /// baseline comes from the wire ([`watch_with_baseline`]), not from this
    /// map.
    Unnamed,
}

impl WatchFingerprint {
    /// Snapshot the identity of the connection commands hashed to `hash` are
    /// routed to.
    pub(crate) fn take(client: &Client, hash: &ClusterHash) -> Self {
        let ids = client.connection_ids();
        match ids.len() {
            0 => Self::Unnamed,
            // Centralized / sentinel: one connection, and it is by construction
            // the one the WATCH went to.
            1 => ids.into_values().next().map_or(Self::Unnamed, Self::Slot),
            _ => {
                // Clustered. `connection_ids` is keyed by `Server`, which
                // carries no slot ownership, so slot→server has to come from
                // fred's cached routing view — the same table its router
                // dispatches by, which is what makes the comparison meaningful.
                //
                // Two failures hide behind one `None` if they are not split
                // apart, and they call for opposite answers: a routing view we
                // could not read leaves the whole map as the best available
                // proxy (the conservative digest), whereas a server we DID name
                // but which carries no id is precisely the "fred never learned
                // this connection's id" state — the digest would then be
                // actively misleading, since it moves only when some *other*
                // node's id changes.
                let named = match hash {
                    ClusterHash::Custom(slot) => client
                        .cached_cluster_state()
                        .and_then(|state| state.get_server(*slot).cloned())
                        .map(|server| ids.get(&server).copied()),
                    // `WatchGuard::mark_watched` always records a
                    // `Custom(slot)`; any other policy means the caller built
                    // its own hash and the owner cannot be named.
                    _ => None,
                };
                match named {
                    Some(Some(id)) => Self::Slot(id),
                    Some(None) => Self::Unnamed,
                    None => {
                        let mut hasher = DefaultHasher::new();
                        let mut pairs: Vec<(String, i64)> = ids
                            .into_iter()
                            .map(|(server, id)| (server.to_string(), id))
                            .collect();
                        pairs.sort_unstable();
                        pairs.hash(&mut hasher);
                        Self::AnyConnection(hasher.finish())
                    }
                }
            }
        }
    }

    /// Whether `self` (taken at `WATCH` time) and `later` (taken immediately
    /// before dispatch) are the same live connection. [`Self::Unnamed`] is never
    /// held, on either side — callers must route that case through
    /// [`Self::comparable_with`] first, because "not held" and "cannot tell"
    /// are different verdicts here.
    pub(crate) fn still_holds(&self, later: &Self) -> bool {
        !matches!(self, Self::Unnamed) && self == later
    }

    /// Whether two fingerprints can be compared at all. `false` exactly when
    /// either side is [`Self::Unnamed`].
    pub(crate) fn comparable_with(&self, other: &Self) -> bool {
        !matches!(self, Self::Unnamed) && !matches!(other, Self::Unnamed)
    }

    /// Re-take the fingerprint and report whether it MOVED since `self`.
    ///
    /// Diagnostic only — never a decision input. A completed reconnect is the
    /// one thing that rewrites fred's connection-id map
    /// (`router/connections.rs:168`, `router/mod.rs:322`), so a `Some(true)`
    /// alongside a proven-unwatched commit says the two observations agree on
    /// the cause; a `Some(false)` says the connection swap (if any) happened
    /// and completed entirely inside the dispatch window, which is worth
    /// knowing when reading the log after the fact. `None` means the comparison
    /// was not available (see [`Self::Unnamed`]).
    pub(crate) fn moved_since(&self, client: &Client, hash: &ClusterHash) -> Option<bool> {
        let now = Self::take(client, hash);
        self.comparable_with(&now).then(|| *self != now)
    }
}

/// Outcome of the L2 pre-dispatch gate — three-valued, because "cannot compare"
/// must not be answered like "do not dispatch".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchVerdict {
    /// L2 was evaluated and agrees the connection is unchanged and live.
    Dispatch,
    /// fred is connected, but it holds no id for this connection, so L2 has
    /// nothing to compare (see [`WatchFingerprint::Unnamed`]). Dispatch — the
    /// alternative is a permanent outage on any store that does not answer
    /// `CLIENT ID` at handshake — and tick `fingerprint_unavailable`.
    DispatchUncomparable,
    /// The connection carrying the `WATCH` is gone or has been replaced. The
    /// watch set is void; abort into the caller's retry.
    Refuse,
}

/// The whole L2 pre-dispatch gate: may this commit still be sent on `client`?
///
/// `recorded` is the fingerprint taken right after `mark_watched`. Two local
/// reads, neither a round trip:
///
/// - `ClientLike::is_connected` — a read of fred's `inner.state`
///   (`runtime/_tokio.rs:262-264`), which `reconnect_once` sets to `Connecting`
///   before it dials (`router/utils.rs:187`). Catches a reconnect that is **in
///   progress right now**, which the fingerprint alone cannot see: the map still
///   holds the dead connection's id until the new handshake finishes, and
///   `disconnect`'s `backchannel.remove_connection_id` (`router/utils.rs:224`)
///   does not actually remove it — fred 10.1.0's implementation calls `.get()`
///   where `.remove()` was intended (`modules/backchannel.rs:109-111`). This is
///   the ONLY conjunct that may refuse.
/// - [`WatchFingerprint::still_holds`] against a fresh snapshot — catches a
///   reconnect that has **completed**, because fred rewrites its backchannel
///   connection-id map only at a successful handshake
///   (`router/connections.rs:168`, `router/mod.rs:322`). Evaluated only when
///   both snapshots are comparable; when either is
///   [`WatchFingerprint::Unnamed`] the answer is
///   [`DispatchVerdict::DispatchUncomparable`], never `Refuse` — see that
///   variant for the outage a fail-closed reading would cause.
///
/// A false `Refuse` costs one `Aborted` retry with a fresh checkout and a fresh
/// `WATCH`; a false `Dispatch` is answered downstream by the L3 witness.
/// Neither closes the window in which fred's router processes a queued
/// `Reconnect` between this call and the socket write — see the module docs.
pub(crate) fn dispatch_verdict(
    client: &Client,
    recorded: &WatchFingerprint,
    hash: &ClusterHash,
) -> DispatchVerdict {
    if !client.is_connected() {
        return DispatchVerdict::Refuse;
    }
    let now = WatchFingerprint::take(client, hash);
    if !recorded.comparable_with(&now) {
        return DispatchVerdict::DispatchUncomparable;
    }
    if recorded.still_holds(&now) {
        DispatchVerdict::Dispatch
    } else {
        DispatchVerdict::Refuse
    }
}

/// `reason` for a baseline whose two probe ids disagreed — the store hands out
/// a different `CLIENT ID` per command, so no witness could ever match one.
pub(crate) const BASELINE_UNSTABLE: &str = "unstable";
/// `reason` for a baseline whose probe `EXEC` reported an error, which is what
/// a store that refuses `CLIENT ID` inside `MULTI` produces (`EXECABORT`).
pub(crate) const BASELINE_EXEC_ERROR: &str = "exec_error";
/// `reason` for a baseline whose probe `EXEC` came back nil. Nothing was
/// watched at that point, so this is not a concurrency outcome — it is a store
/// answering `EXEC` in a shape this code cannot interpret.
pub(crate) const BASELINE_EXEC_DISCARDED: &str = "exec_discarded";
/// `reason` for a probe `EXEC` array that was not exactly two integers.
pub(crate) const BASELINE_UNEXPECTED_SHAPE: &str = "unexpected_shape";
/// `reason` for a probe whose reply set was shorter than the five frames sent.
pub(crate) const BASELINE_SHORT_REPLY: &str = "short_reply";

/// What the L3 witness is compared against: a `CLIENT ID` sampled from inside a
/// `MULTI … EXEC` on the connection the `WATCH` is about to land on.
///
/// Taking the baseline this way rather than reusing L2's handshake-time
/// [`WatchFingerprint`] is the whole point — see the module docs' L3 paragraph.
/// The two ids come from the same command, in the same context, on the same
/// connection, so a proxy that renumbers `CLIENT ID` cannot manufacture a
/// mismatch, and a proxy that refuses `CLIENT ID` inside `MULTI` is *detected*
/// here rather than turning every later commit into an `EXECABORT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WitnessBaseline {
    /// Two `CLIENT ID`s inside one `MULTI` returned the same integer. A commit
    /// witness that differs from it is proof the commit ran elsewhere.
    Trusted(i64),
    /// No usable baseline. L3 is disarmed for this attempt: the commit
    /// pipeline queues no `CLIENT ID` (queueing one would `EXECABORT` the real
    /// commit on the very stores this variant identifies), the witness is
    /// `None`, and nothing is ever accused. Carries one of the `BASELINE_*`
    /// reasons.
    Unavailable(&'static str),
}

impl WitnessBaseline {
    /// Whether the in-transaction `CLIENT ID` witness *contradicts* the
    /// baseline — i.e. the transaction provably ran on a connection other than
    /// the one that took the `WATCH`.
    ///
    /// `false` whenever the two cannot be compared, which is the honest answer
    /// rather than an accusation: [`Self::Unavailable`] never accuses (L3 is
    /// off), and a missing witness means either L3 was never armed or the reply
    /// shape was not what `CLIENT ID` returns.
    pub(crate) fn contradicted_by(&self, witness: Option<i64>) -> bool {
        matches!((self, witness), (Self::Trusted(base), Some(actual)) if *base != actual)
    }

    /// `Some(reason)` when L3 is disarmed for this attempt, for the metric cell
    /// and the log field. `None` when the baseline is trusted.
    pub(crate) fn unavailable_reason(&self) -> Option<&'static str> {
        match self {
            Self::Trusted(_) => None,
            Self::Unavailable(reason) => Some(reason),
        }
    }

    /// Whether the commit pipeline should queue its `CLIENT ID` witness.
    fn armed(&self) -> bool {
        matches!(self, Self::Trusted(_))
    }
}

/// `WATCH` the given keys **and** calibrate the L3 witness baseline in ONE
/// pinned pipeline — no extra round trip over the bare `WATCH` it replaces.
///
/// The five frames, in the order they must be sent:
///
/// | # | frame | why here |
/// |---|---|---|
/// | 1 | `MULTI` | the baseline must be read from inside a transaction, because that is the context the witness is read from |
/// | 2 | `CLIENT ID` | first sample |
/// | 3 | `CLIENT ID` | second sample — two are needed to detect a store that renumbers per command |
/// | 4 | `EXEC` | yields `[id, id]` |
/// | 5 | `WATCH keys…` | LAST, because `EXEC` clears the connection's watch set: a `WATCH` sent before the probe would be silently discarded by it |
///
/// Being one pipeline is what makes the sample meaningful: fred writes these
/// frames back-to-back on one socket, so frame 5's watch set is established on
/// exactly the connection frames 2-3 named. (Empirically verified against a
/// real store: the pipeline answers `[+OK, +QUEUED, +QUEUED, [id, id], +OK]`,
/// and a subsequent tripped `EXEC` still returns nil — the probe's `EXEC` does
/// not disturb the `WATCH` that follows it.)
///
/// Returns `Err` only when the `WATCH` itself did not succeed, so call sites
/// classify it exactly as they classified a failed `client.watch(...)`. A
/// baseline that could not be read is NOT an error — it is
/// [`WitnessBaseline::Unavailable`], and the caller proceeds with L3 disarmed.
///
/// On a torn write the whole pipeline collapses to a single `Err`
/// (`clients/pipeline.rs:263-288`) and this returns it. Neither way a pipeline
/// can stop partway leaves a half-open `MULTI`: a socket that dies
/// mid-pipeline takes the connection, and its transaction state, with it; and
/// a pipeline whose caller gives up partway (its timeout fires or its future
/// is dropped) has its unwritten frames skipped and its open `MULTI` closed by
/// fred itself, with a `DISCARD` on the same connection (the vendored fred's
/// change 9). The `txn` pool's per-checkout
/// `DISCARD`+`UNWATCH` hygiene probe (b.7td) is the backstop for anything that
/// still slips through.
pub(crate) async fn watch_with_baseline(
    client: &Client,
    hash: &ClusterHash,
    keys: &[&str],
) -> Result<WitnessBaseline, fred::error::Error> {
    let pipe = client.pipeline().with_options(&pinned_options(hash));
    queue_keyless_into(&pipe, hash, "MULTI").await?;
    queue_keyless_into(&pipe, hash, "CLIENT ID").await?;
    queue_keyless_into(&pipe, hash, "CLIENT ID").await?;
    queue_keyless_into(&pipe, hash, "EXEC").await?;
    // Queued as a `CustomCommand` rather than through `KeysInterface::watch`.
    // fred builds `WATCH` with `ok_cmd!`, which asserts the reply is `+OK` —
    // true on a client, false inside a pipeline, where every queued command
    // responds `QUEUED` early (`clients/pipeline.rs:96-114`) and the assertion
    // fails locally with `Expected OK, found Queued`. `custom` decodes to
    // `Value`, which accepts the early response; the real reply still comes
    // back in `try_all`'s vector. Keyed, so fred would route it correctly
    // anyway — the hash on the command and the pinned options only re-state the
    // same slot and the one-attempt ceiling.
    let watch_cmd = CustomCommand::new_static("WATCH", hash.clone(), false);
    let watch_args: Vec<Value> = keys.iter().map(|key| Value::from(*key)).collect();
    let _: Value = pipe.custom(watch_cmd, watch_args).await?;

    let mut results = pipe.try_all::<Value>().await;
    // The `WATCH` is the last frame, and it is the only one whose failure the
    // caller must see as a failure: without it there is no optimistic lock at
    // all, and the pre-b.bhb call sites treated exactly this as their error
    // path. A short reply set means the pipeline never produced one.
    let Some(watch_result) = results.pop() else {
        return Err(fred::error::Error::new(
            fred::error::ErrorKind::Protocol,
            "WATCH pipeline returned an empty reply set",
        ));
    };
    let _watched: Value = watch_result?;
    if results.len() < 4 {
        // The WATCH answered but earlier frames are missing — a shape fred is
        // not documented to produce. The lock is live, so this is not an error;
        // L3 simply has no baseline.
        return Ok(WitnessBaseline::Unavailable(BASELINE_SHORT_REPLY));
    }
    Ok(baseline_from_probe_exec(results.remove(3)))
}

/// Decode the baseline probe's `EXEC` reply. Split out from
/// [`watch_with_baseline`] so the decision table is testable without a server.
fn baseline_from_probe_exec(exec: Result<Value, fred::error::Error>) -> WitnessBaseline {
    match exec {
        Ok(Value::Array(items)) => match items.as_slice() {
            // Both samples agree: the store answers `CLIENT ID` inside `MULTI`
            // from a stable id space, which is all L3 needs.
            [first, second] => match (first.as_i64(), second.as_i64()) {
                (Some(a), Some(b)) if a == b => WitnessBaseline::Trusted(a),
                // A store handing out a different id per command cannot be
                // compared against — every commit would look unwatched.
                (Some(_), Some(_)) => WitnessBaseline::Unavailable(BASELINE_UNSTABLE),
                _ => WitnessBaseline::Unavailable(BASELINE_UNEXPECTED_SHAPE),
            },
            _ => WitnessBaseline::Unavailable(BASELINE_UNEXPECTED_SHAPE),
        },
        // An error on the probe's `EXEC`. Two causes:
        //
        // - The store refused `CLIENT ID` at queue time inside the `MULTI`
        //   (a store that denies the command), so `EXEC` answered
        //   `EXECABORT`. This is the shape that MUST disarm L3: queueing the
        //   witness into the real commit would abort every structural
        //   `UpdateLayer` and every `CreateSession` with a terminal
        //   `INTERNAL`.
        // - The store is `-BUSY` behind a long script: `CLIENT ID` carries
        //   `loading` and `allow-busy` and queues fine, but `EXEC` does not
        //   carry `allow-busy`, so the whole `EXEC` is refused as `EXECABORT
        //   Transaction discarded because of: BUSY …` (valkey
        //   `commands/exec.json`). Under `-LOADING` the probe succeeds (`EXEC`
        //   carries `loading`). Disarming is harmless here — the commit that
        //   follows meets the same refusal and is retried as transient
        //   (`storage::client::is_store_not_serving_reply`) — but each
        //   retried attempt re-runs this probe, so a `-BUSY` window ticks
        //   `live_edit_storage_watch_voided_total{disposition=
        //   "witness_unavailable"}` once per attempt.
        Err(_) => WitnessBaseline::Unavailable(BASELINE_EXEC_ERROR),
        Ok(Value::Null) => WitnessBaseline::Unavailable(BASELINE_EXEC_DISCARDED),
        Ok(_) => WitnessBaseline::Unavailable(BASELINE_UNEXPECTED_SHAPE),
    }
}

/// Minimum wall-clock gap between two emissions of either "L2 disarmed" /
/// "L3 disarmed" warning.
///
/// These describe a *property of the deployment*, not an event: on a store that
/// denies `CLIENT ID` every single watched commit hits them, at whatever rate
/// the fleet writes. One line per five minutes is enough to make the condition
/// discoverable in a log tail while leaving the per-attempt record to the
/// metric, which is what a dev-AMR run reads anyway. Same interval as
/// `storage::client`'s steady-state auth hint, and the same [`LogGate`] behind
/// it — including its `tokio::time::Instant` clock.
const DISARMED_WARN_MIN_INTERVAL: Duration = Duration::from_secs(300);

/// One [`LogGate`] per "a defence layer is disarmed on this store" warning,
/// each process-global on purpose: the condition is a property of the backing
/// store, so per-call-site gating would multiply the same line by the number of
/// paths. The *metric* is the per-attempt, per-path record and the testable
/// signal; these are decoration. `LogGate::new` is `const`, so the gates are
/// plain `static`s rather than `LazyLock`s.
static WITNESS_UNAVAILABLE_GATE: LogGate = LogGate::new(DISARMED_WARN_MIN_INTERVAL);
static FINGERPRINT_UNAVAILABLE_GATE: LogGate = LogGate::new(DISARMED_WARN_MIN_INTERVAL);

/// Rate-limited notice that L3 is disarmed on this backing store.
pub(crate) fn warn_witness_unavailable(path: &'static str, reason: &'static str) {
    if !WITNESS_UNAVAILABLE_GATE.allow() {
        return;
    }
    tracing::warn!(
        path,
        reason,
        "watched-transaction witness baseline unavailable: this backing store does not answer a \
         stable CLIENT ID from inside MULTI, so the in-transaction witness (L3) is disarmed and \
         the optimistic-locking guarantee rests on L1 (no replay) plus L2 (pre-dispatch \
         connection identity) alone. A commit that runs on a connection which never held its \
         WATCH would not be detected. Rate-limited to one line per 5 minutes; the per-attempt \
         record is live_edit_storage_watch_voided_total{{disposition=\"witness_unavailable\"}}"
    );
}

/// Rate-limited notice that L2 is disarmed on this backing store.
pub(crate) fn warn_fingerprint_unavailable(path: &'static str) {
    if !FINGERPRINT_UNAVAILABLE_GATE.allow() {
        return;
    }
    tracing::warn!(
        path,
        "watched-transaction connection fingerprint unavailable: fred holds no CLIENT ID for the \
         connection carrying the WATCH, which is what a store that ACL-denies or proxy-rejects \
         CLIENT ID at handshake produces on every healthy connection. The pre-dispatch identity \
         check (L2) is skipped rather than failing closed — failing closed here would abort every \
         CreateSession and every structural UpdateLayer to retry exhaustion — so the attempt \
         dispatches and detection rests on L1 plus the in-transaction witness (L3). Rate-limited \
         to one line per 5 minutes; the per-attempt record is \
         live_edit_storage_watch_voided_total{{disposition=\"fingerprint_unavailable\"}}"
    );
}

/// Shape of the two test-only hooks the WATCHed commit paths expose
/// (`with_watch_window_hook_for_test`, `with_pre_dispatch_hook_for_test` on
/// `LayerStore` and `SessionStore`).
///
/// The hook is handed the *guarded* fred `Client` — the one actually carrying
/// the `WATCH` — so a test can read its `CLIENT ID`, `CLIENT KILL ID` it from a
/// second connection, and land a competing write. That is the real mechanism
/// this fix defends against; none of the pre-existing injector seams
/// (`fault_injector_for_test`, `fred_error_injector_for_test`,
/// `create_fault_injector_for_test`, `delete_exec_fault_injector_for_test`,
/// `ProbeFaultInjector`) can express a window *between* two commands on one
/// connection, which is why this one exists alongside them.
///
/// The `for<'a>` bound is what lets the hook borrow the client for the duration
/// of its future rather than forcing a clone.
#[cfg(any(test, feature = "test-fault-injection"))]
pub type WatchWindowHook =
    std::sync::Arc<dyn for<'a> Fn(&'a Client) -> futures::future::BoxFuture<'a, ()> + Send + Sync>;

/// A `MULTI` … `EXEC` under a `WATCH`, sent as one non-replayable pipeline.
///
/// Build with [`WatchedTxn::open`], queue mutations against
/// [`WatchedTxn::commands`] exactly as they were queued against fred's
/// `Transaction`, then consume with [`WatchedTxn::exec`].
pub(crate) struct WatchedTxn {
    pipe: WithOptions<Pipeline<Client>>,
    hash: ClusterHash,
    /// Whether [`Self::exec`] queues the L3 `CLIENT ID` witness, decided by the
    /// baseline handed to [`Self::open`]. `false` is not merely "skip an
    /// optional check": on a store that refuses `CLIENT ID` inside `MULTI`,
    /// queueing it would make the server abort the whole transaction
    /// (`EXECABORT`), turning every structural `UpdateLayer` and every
    /// `CreateSession` into a terminal `INTERNAL`.
    witness_armed: bool,
}

impl WatchedTxn {
    /// Open the transaction: build the pinned pipeline and queue `MULTI`.
    ///
    /// Queueing into a fred `Pipeline` is a local buffer push that responds
    /// early with `QUEUED` (`fred-10.1.0/src/clients/pipeline.rs:96-114`), so
    /// nothing reaches the wire until [`Self::exec`].
    ///
    /// `baseline` is the value [`watch_with_baseline`] returned for this
    /// attempt's `WATCH`. It decides whether the L3 witness is queued at all —
    /// see [`Self::witness_armed`].
    pub(crate) async fn open(
        client: &Client,
        hash: &ClusterHash,
        baseline: &WitnessBaseline,
    ) -> Result<Self, fred::error::Error> {
        Self::open_with_options(client, hash, baseline, pinned_options(hash)).await
    }

    /// [`Self::open`] for a transaction whose client timeout is `timeout`
    /// instead of the configured command timeout: the WATCH-route
    /// `UpdateLayer` commit, whose size-scaled timeout
    /// (`storage::client::txn_timeout`) covers sending every queued command,
    /// the store's `EXEC` and the reply. fred reads a pipeline's timeout from
    /// its FIRST command (`RouterCommand::timeout_dur`), the `MULTI` queued
    /// here; every command carries it, through the pinned options.
    pub(crate) async fn open_with_timeout(
        client: &Client,
        hash: &ClusterHash,
        baseline: &WitnessBaseline,
        timeout: Duration,
    ) -> Result<Self, fred::error::Error> {
        let options = Options {
            timeout: Some(timeout),
            ..pinned_options(hash)
        };
        Self::open_with_options(client, hash, baseline, options).await
    }

    async fn open_with_options(
        client: &Client,
        hash: &ClusterHash,
        baseline: &WitnessBaseline,
        options: Options,
    ) -> Result<Self, fred::error::Error> {
        let pipe = client.pipeline().with_options(&options);
        let mut txn = Self {
            pipe,
            hash: hash.clone(),
            witness_armed: baseline.armed(),
        };
        txn.queue_keyless("MULTI").await?;
        Ok(txn)
    }

    /// Shortest reply set this transaction can legally produce: `MULTI`, at
    /// least one queued mutation, the optional `CLIENT ID`, and `EXEC`.
    ///
    /// Derived from the instance rather than hard-coded, because the witness is
    /// conditional: a constant sized for the armed shape would read a
    /// disarmed-but-complete reply set as truncated and send a *landed* commit
    /// into the in-doubt reconciliation path.
    fn min_replies(&self) -> usize {
        3 + usize::from(self.witness_armed)
    }

    /// The interface mutations are queued against. Implements the same fred
    /// command traits a `Transaction` does, so existing queue helpers
    /// (`execute_mutation`, `sadd`, `xadd`, …) accept it unchanged.
    pub(crate) fn commands(&self) -> &WithOptions<Pipeline<Client>> {
        &self.pipe
    }

    /// Queue a keyless command as a slot-pinned `custom_raw`.
    async fn queue_keyless(&mut self, cmd: &'static str) -> Result<(), fred::error::Error> {
        queue_keyless_into(&self.pipe, &self.hash, cmd).await
    }

    /// Queue the L3 witness (when armed) and `EXEC`, send the whole pipeline,
    /// and decode the reply set.
    ///
    /// `origin` names the pool the transaction runs on. A
    /// [`WatchedExec::InDoubt`] carrying fred's `Timeout` — the `EXEC` reply
    /// outlived the transaction's timeout (the configured command timeout, or
    /// the size-scaled one from [`Self::open_with_timeout`]) — is counted on
    /// it here, the one place
    /// every `InDoubt` is built, so no call site counts it and a call site's
    /// in-doubt arm must not count it again. The other variants are not
    /// counted: [`WatchedExec::ExecError`] and [`WatchedExec::Discarded`] need
    /// a full reply set, so they hold server error frames and are never a
    /// timeout.
    pub(crate) async fn exec(self, origin: FredErrOrigin<'_>) -> WatchedExec {
        count_in_doubt_timeout(self.exec_uncounted().await, origin)
    }

    /// [`Self::exec`] without the timeout count.
    async fn exec_uncounted(mut self) -> WatchedExec {
        if self.witness_armed {
            if let Err(e) = self.queue_keyless("CLIENT ID").await {
                return WatchedExec::InDoubt(e);
            }
        }
        if let Err(e) = self.queue_keyless("EXEC").await {
            return WatchedExec::InDoubt(e);
        }
        let (min_replies, witness_armed) = (self.min_replies(), self.witness_armed);
        decode(
            self.pipe.try_all::<Value>().await,
            min_replies,
            witness_armed,
        )
    }

    /// Test-only: force [`WatchedExec::InDoubt`] instead of the real verdict,
    /// so the two in-doubt reconciliation arms (`reconcile_in_doubt_commit`,
    /// `reconcile_in_doubt_create`) can be driven deterministically (b.bhb).
    ///
    /// A real `InDoubt` comes from a truncated reply set — the socket dying with
    /// the pipeline in flight, or the `EXEC` reply outliving the transaction's
    /// timeout (the configured command timeout, or the size-scaled one from
    /// [`Self::open_with_timeout`]). A test can stage the timeout by stalling
    /// the connection
    /// (`test_support::ConnStall`), but only as "the pipeline reached the
    /// server and will run once released"; it cannot stage "no reply at all"
    /// with the keyspace chosen, mutated or untouched. This seam does that, and
    /// does it without timing. See [`WatchedExecOverride`] for what each variant
    /// models and why the keyspace-mutating and non-mutating pair are both
    /// needed (whether the keyspace was mutated is exactly the question the
    /// reconciliation arms exist to answer).
    ///
    /// The substituted `InDoubt` is counted on `origin` exactly as
    /// [`Self::exec`] counts a real one, so a call site behaves identically
    /// with and without an override installed.
    ///
    /// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub(crate) async fn exec_overridden(
        self,
        over: WatchedExecOverride,
        origin: FredErrOrigin<'_>,
    ) -> WatchedExec {
        count_in_doubt_timeout(self.exec_overridden_uncounted(over).await, origin)
    }

    /// [`Self::exec_overridden`] without the timeout count.
    #[cfg(any(test, feature = "test-fault-injection"))]
    async fn exec_overridden_uncounted(self, over: WatchedExecOverride) -> WatchedExec {
        // Kind `Canceled` is what fred reports when a connection dies with
        // commands outstanding (`router/utils.rs::disconnect` →
        // `check_final_write_attempt`), so the synthesized error classifies
        // identically to the real thing at every downstream site.
        let lost = || {
            fred::error::Error::new(
                fred::error::ErrorKind::Canceled,
                "injected: watched transaction reply lost (in-doubt seam)",
            )
        };
        match over {
            WatchedExecOverride::ReplaceOutcomeAfterExec => {
                // The real pipeline runs and commits; only the verdict handed
                // back is replaced. Discarding a `Conflict` / `ExecError` here
                // is deliberate — the seam's contract is "you got no answer",
                // and the reconciliation arm must reach that conclusion from
                // the keyspace alone.
                let _ = self.exec_uncounted().await;
                WatchedExec::InDoubt(lost())
            }
            WatchedExecOverride::SkipExecReturnInDoubt => {
                // Dropping the pipeline sends nothing: `Pipeline::send_command`
                // only pushes onto a local buffer and responds `QUEUED` early
                // (`fred-10.1.0/src/clients/pipeline.rs:96-114`), so the server
                // never sees `MULTI`. The `WATCH` is therefore STILL LIVE on
                // this connection — the call site's `InDoubt` arm must leave the
                // `WatchGuard` armed so `Drop` emits the slot-pinned `UNWATCH`.
                drop(self);
                WatchedExec::InDoubt(lost())
            }
            WatchedExecOverride::SkipExecReturnTimedOut => {
                // Same keyspace outcome as `SkipExecReturnInDoubt` — nothing is
                // sent, the `WATCH` stays live — but with the error kind fred
                // reports when the reply outlives the transaction's timeout
                // (the configured command timeout, or the size-scaled one
                // from `open_with_timeout`).
                drop(self);
                WatchedExec::InDoubt(fred::error::Error::new(
                    fred::error::ErrorKind::Timeout,
                    "injected: watched transaction reply timed out (in-doubt seam)",
                ))
            }
        }
    }
}

/// Count `outcome` on `origin` when it is an [`WatchedExec::InDoubt`] carrying
/// fred's `Timeout` (see [`WatchedTxn::exec`]), and hand it back unchanged.
/// Under tests and the `test-support` feature it is also counted on the
/// origin's client alone
/// (`ValkeyClient::watched_exec_timeouts_for_test`), which, unlike the
/// metric, no drop-time `UNWATCH` reaches.
fn count_in_doubt_timeout(outcome: WatchedExec, origin: FredErrOrigin<'_>) -> WatchedExec {
    if let WatchedExec::InDoubt(e) = &outcome {
        origin.observe_error(e);
        #[cfg(any(test, feature = "test-support"))]
        origin.count_watched_exec_timeout_for_test(e);
    }
    outcome
}

/// Queue one keyless command into a pinned pipeline as a slot-tagged
/// `custom_raw`. Shared by the commit pipeline and the baseline probe.
async fn queue_keyless_into(
    pipe: &WithOptions<Pipeline<Client>>,
    hash: &ClusterHash,
    cmd: &'static str,
) -> Result<(), fred::error::Error> {
    // The slot rides on the `CustomCommand` as well as the `Options`, on
    // purpose: `Command::cluster_hash` consults `custom_hash_slot()` FIRST
    // (`protocol/command.rs:535-543`), so whichever fred reads, the command
    // lands on the watched node.
    let command = CustomCommand::new_static(cmd, hash.clone(), false);
    // Resolves immediately — `Pipeline::send_command` responds `QUEUED`
    // without touching the wire. An `Err` here is a local queueing failure.
    let _: Value = pipe.custom(command, Vec::<Value>::new()).await?;
    Ok(())
}

/// Test-only substitution for a watched transaction's `EXEC` verdict (b.bhb).
///
/// Every variant hands the call site `WatchedExec::InDoubt`. The first two
/// differ in what they leave behind in Valkey, which is the whole point — the
/// in-doubt reconciliation arms decide by *reading the keyspace*, so coverage
/// that only ever staged one of the two answers would pin half the contract.
/// The third differs from the second only in the error's kind.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchedExecOverride {
    /// Models **"the `EXEC` ran and the reply was lost."** The pipeline is sent
    /// exactly as production sends it — `MULTI`, the queued mutations,
    /// `CLIENT ID`, `EXEC` all reach the server and commit — and only the
    /// decoded outcome is discarded. Reconciliation must find the write and
    /// report success rather than retrying (on the layer path a retry would
    /// only rediscover the commit through the per-call commit-token check, minus
    /// the external-value releases the reconcile enqueues; on the session path
    /// it would hand the caller `AlreadyExists` for the session it just
    /// created). Valkey cleared the watch set on that `EXEC`, but the call
    /// site's in-doubt arm does not call `mark_unwatched` — matching production,
    /// where the same arm cannot know either — so `Drop` pays one redundant
    /// `UNWATCH`. That is the pre-existing b.jov posture, not a leak.
    ReplaceOutcomeAfterExec,
    /// Models **"the connection died before the write."** The pipeline is
    /// dropped without reaching the wire, so the server never sees `MULTI` and
    /// the keyspace is untouched. Reconciliation must find nothing and return
    /// the retriable answer. The `WATCH` is still live on the connection here,
    /// so the call site must leave the guard armed for `Drop` to tear down.
    SkipExecReturnInDoubt,
    /// [`Self::SkipExecReturnInDoubt`]'s keyspace outcome with fred's
    /// `ErrorKind::Timeout` instead of `Canceled`: the in-doubt verdict a
    /// reply that outlived the transaction's timeout (the configured command
    /// timeout, or the size-scaled one from `WatchedTxn::open_with_timeout`)
    /// produces. Exists so the
    /// timeout count every `InDoubt` passes through can be driven
    /// deterministically; the reconciliation arms treat it exactly like the
    /// `Canceled` form.
    SkipExecReturnTimedOut,
}

/// FIFO queue of [`WatchedExecOverride`]s installed on `LayerStore` /
/// `SessionStore` via `with_watched_exec_override_for_test`. One entry is
/// consumed per watched commit attempt; an empty queue lets the attempt run the
/// real pipeline and decode the real verdict.
#[cfg(any(test, feature = "test-fault-injection"))]
pub type WatchedExecOverrideQueue =
    std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<WatchedExecOverride>>>;

/// What the server said about a watched transaction.
pub(crate) enum WatchedExec {
    /// `EXEC` returned nil: a watched key changed between `WATCH` and `EXEC`.
    /// Nothing ran, and Valkey clears the watch set either way, so the
    /// connection is clean.
    Conflict,
    /// `EXEC` returned an array — every queued command ran. `witness` is the
    /// `CLIENT ID` reported from *inside* the transaction, i.e. the connection
    /// that actually executed it.
    Committed { witness: Option<i64> },
    /// `EXEC` ran but reported an error — one element of the array was
    /// (`WRONGTYPE`, an `XADD` refused for an equal-or-smaller id), or the
    /// `MULTI` itself failed, or the reply had an unexpected shape. fred
    /// collapses a nested error to the first one (`protocol/utils.rs:375-380`),
    /// so the array is not recoverable here; the commit's siblings still
    /// executed. An `EXECABORT`, and the cluster check's `-TRYAGAIN` to the
    /// `EXEC`, are [`Self::Discarded`] instead. Any error fred builds itself
    /// never lands here: it arrives as the whole call's single `Err`, which is
    /// [`Self::InDoubt`].
    ExecError(fred::error::Error),
    /// The server discarded the transaction and **no queued command ran**.
    /// Either `EXEC` answered `EXECABORT` — a command was rejected at queue
    /// time (`flagTransaction` → `-EXECABORT Transaction discarded because
    /// of previous errors.`), or the `EXEC` itself was rejected as a whole
    /// (`rejectCommand` → `execCommandAbort` → `-EXECABORT Transaction
    /// discarded because of: <refusal>`; valkey `src/multi.c`,
    /// `src/server.c`) — or the cluster check refused the `EXEC` with a bare
    /// `-TRYAGAIN` (a slot mid-migration; `processCommand` discards the
    /// transaction before it answers, valkey 9.0.3 `src/server.c:4310-4324`;
    /// see `is_exec_refused_by_cluster_check`). Every way Valkey ran
    /// `discardTransaction`, which also clears the connection's watch set.
    ///
    /// `exec` is the `EXEC` reply's error; `cause` is the first queued
    /// command's error when one was rejected at queue time, else `exec`
    /// itself (whose text then names the refusal after the `EXECABORT`
    /// prefix, or is the `-TRYAGAIN` refusal itself). A caller
    /// classifies `cause` (`storage::client::store_write_refusal` recognises a
    /// store write refusal in either shape) and must not run tear handling: a
    /// discarded transaction wrote nothing.
    Discarded {
        exec: fred::error::Error,
        cause: fred::error::Error,
    },
    /// The pipeline never produced a full reply set: the connection died with
    /// the transaction in flight, or the reply timed out. **Whether `EXEC` ran
    /// is unknown** — callers must reconcile against the keyspace rather than
    /// blind-retry.
    InDoubt(fred::error::Error),
}

/// Decode `try_all`'s per-command results.
///
/// fred delivers a pipeline all-or-nothing, which is what makes the reply
/// length the whole signal. The caller's responder fires only once every frame
/// has arrived (`ResponseKind::Buffer`, `received == expected`), and a write
/// failure responds on that same *shared* sender
/// (`protocol/command.rs:1954-1979`), collapsing the entire call to
/// `vec![Err(e)]` (`clients/pipeline.rs:263-288`). The vector is therefore
/// either one entry per queued command or exactly one `Err` — no partial set
/// exists. Every pipeline this module builds is at least `MULTI`, one mutation,
/// the optional `CLIENT ID`, and `EXEC`, so anything shorter than `min_replies`
/// is the failure shape.
///
/// `min_replies` and `witness_armed` both come from [`WatchedTxn::open`]'s
/// state rather than from a constant, because the witness frame is conditional
/// (see [`WitnessBaseline`]). Getting `witness_armed` wrong is not cosmetic in
/// either direction: with no witness queued, the last element of the `EXEC`
/// array is a *mutation's* reply, and an `HSET` answers with an integer — read
/// as a witness it would be compared against the baseline and manufacture a
/// false `committed_unwatched`.
fn decode(
    mut results: Vec<Result<Value, fred::error::Error>>,
    min_replies: usize,
    witness_armed: bool,
) -> WatchedExec {
    if results.len() < min_replies {
        let err = results
            .into_iter()
            .find_map(Result::err)
            .unwrap_or_else(|| {
                fred::error::Error::new(
                    fred::error::ErrorKind::Protocol,
                    "watched transaction returned a short reply set",
                )
            });
        return WatchedExec::InDoubt(err);
    }
    // The FIRST element is `MULTI`. An error there means the server never
    // opened a transaction, so everything queued behind it ran as ordinary
    // unconditional commands and the trailing `EXEC` frame — whatever it says —
    // is not describing a transaction at all. That is a torn commit, not a
    // conflict and not an in-doubt: the writes landed, unwatched. Report it as
    // an execution error so the call sites keep their non-retriable
    // classification instead of either accepting it or replaying it.
    if results[0].is_err() {
        let err = results
            .into_iter()
            .next()
            .and_then(Result::err)
            .expect("index 0 checked as Err immediately above");
        return WatchedExec::ExecError(err);
    }
    // The last element is `EXEC`. Everything between it and `MULTI` is one
    // `+QUEUED` per queued command; a rejection among those aborts the
    // transaction at queue time and `EXEC` answers `EXECABORT`, so the `EXEC`
    // element already carries that verdict — and the rejection itself, the
    // discarded transaction's cause, is the first error among them.
    let exec = results
        .pop()
        .expect("len >= min_replies >= 3, checked above");
    match exec {
        Ok(Value::Null) => WatchedExec::Conflict,
        Ok(Value::Array(items)) => WatchedExec::Committed {
            witness: witness_armed
                .then(|| items.last().and_then(|v| v.as_i64()))
                .flatten(),
        },
        Ok(other) => WatchedExec::ExecError(fred::error::Error::new(
            fred::error::ErrorKind::Protocol,
            format!("EXEC returned an unexpected reply shape: {other:?}"),
        )),
        Err(e) if is_execabort(&e) => {
            let cause = results
                .into_iter()
                .skip(1)
                .find_map(Result::err)
                .unwrap_or_else(|| e.clone());
            WatchedExec::Discarded { exec: e, cause }
        }
        // The cluster check refused the `EXEC` as a whole: Valkey discarded
        // the transaction first, so nothing ran, and the `EXEC` reply is the
        // refusal itself, which is also its cause.
        Err(e) if is_exec_refused_by_cluster_check(&e) => WatchedExec::Discarded {
            exec: e.clone(),
            cause: e,
        },
        Err(e) => WatchedExec::ExecError(e),
    }
}

/// Whether an `EXEC` reply's error is `EXECABORT`: the server discarded the
/// transaction without running a queued command. Kind-gated and
/// prefix-anchored like `storage::client`'s reply predicates: fred maps the
/// unrecognised `EXECABORT` token to `ErrorKind::Unknown`
/// (`fred-10.1.0/src/protocol/utils.rs:96-129`).
fn is_execabort(e: &fred::error::Error) -> bool {
    matches!(e.kind(), fred::error::ErrorKind::Unknown) && e.details().starts_with("EXECABORT ")
}

/// Whether an `EXEC` reply's error is Valkey's cluster check refusing the
/// `EXEC` as a whole, before anything ran: `-TRYAGAIN`
/// ([`crate::storage::client::is_tryagain_reply`], the same kind-gated,
/// prefix-anchored match). `processCommand` runs `getNodeByQuery` over every
/// queued command at `EXEC` and, on a refusal, calls `discardTransaction`
/// before `clusterRedirectClient` writes the bare reply (Valkey 9.0.3
/// `src/server.c:4310-4324`; `src/multi.c` has no cluster check of its own),
/// so the transaction is gone, its watch set with it, and no queued command
/// ran. `cluster.c:1320` is that reply's only emitter, and a command run
/// inside `EXEC` goes through `call()`, never that check, so an executed
/// `EXEC` array cannot carry it either (fred collapses such an array to its
/// first error, `fred-10.1.0/src/protocol/utils.rs:375-380`).
///
/// Only Valkey's own reply may match, because the unsafe direction is
/// "nothing ran". It is Valkey's: in a full reply set every element is a
/// reply frame parsed by fred's `pretty_error`, which keeps the reply text
/// verbatim (`fred-10.1.0/src/clients/pipeline.rs:283-287`,
/// `protocol/utils.rs:95-128`), and every error fred builds itself — a
/// spent redirect or attempt budget, a routing miss, a connection error, a
/// timeout — reaches the caller on the pipeline's shared sender as the whole
/// call's single `Err` (`clients/pipeline.rs:272-281`), which [`decode`]
/// answers [`WatchedExec::InDoubt`] before it looks at the `EXEC` element. No
/// fred-built text starts with `TRYAGAIN` anyway.
///
/// The cluster check's other refusals of an `EXEC` are left out because fred
/// never delivers them in the `EXEC` position: a `-CLUSTERDOWN` frame is
/// fatal to its connection in fred's read loop (`router/responses.rs:256-291`,
/// via `preprocess_frame`, `router/utils.rs:259`), which drops the connection
/// and fails or replays the in-flight pipeline — a single `Err`, so
/// `InDoubt`; and a `-MOVED` / `-ASK` is intercepted as a redirect before the
/// pipeline's reply buffer (`router/clustered.rs:322-325`), where the `txn`
/// pool's zero redirection budget (`connection.max_redirections = 0` in
/// `storage::client`'s `txn` client config) fails it `"Too many
/// redirections."` on the same shared sender. A `-CLUSTERDOWN` text that does
/// reach the caller comes through fred's connection drop, which a prefix
/// cannot tell from Valkey's reply, so matching these here would buy an arm
/// no reply in the `EXEC` position reaches.
///
/// **Re-verify on any fred upgrade.** That only `TRYAGAIN ` reaches [`decode`]
/// in the `EXEC` position rests on three fred 10.1.0 behaviours, each of which
/// a new release could change:
///
/// - a `-CLUSTERDOWN` frame is fatal to its connection
///   (`router/responses.rs:256-291`, via `preprocess_frame`), so it never
///   arrives as an element of a full reply set;
/// - a `-MOVED` / `-ASK` is intercepted by the cluster router
///   (`router/clustered.rs:322-325`), and the `txn` pool's
///   `max_redirections = 0` turns it into fred's own error;
/// - every error fred builds arrives as the pipeline call's single `Err`
///   (`clients/pipeline.rs:272-286`), which is [`WatchedExec::InDoubt`].
///
/// If a fred upgrade lets any of these reach the `EXEC` position as Valkey's
/// reply, decide whether it belongs here; if one lets a fred-built error reach
/// it, make sure no fred-built text can match this predicate.
fn is_exec_refused_by_cluster_check(e: &fred::error::Error) -> bool {
    crate::storage::client::is_tryagain_reply(e)
}

#[cfg(test)]
mod tests {
    //! b.bhb — the reply-shape decision table and the two identity
    //! comparisons that ride on it.
    //!
    //! [`decode`] is the single place the service decides whether a watched
    //! transaction committed, was cancelled, was discarded without running,
    //! failed while executing, or left the question open. Each verdict routes to a *different* disposition at
    //! both call sites — accept, retry, non-retriable `Internal`, or reconcile
    //! against the keyspace — so a shape landing in the wrong arm is not a
    //! cosmetic error: `Conflict` misread as `Committed` is the b.8q1
    //! silent-dropped-write, and `InDoubt` misread as `Conflict` sends a
    //! landed commit back through the retry loop without reconciling it — on
    //! the layer path the next attempt's per-call commit-token check then
    //! rediscovers the commit, but without the external-value releases the
    //! reconcile enqueues; on the session path the caller is handed
    //! `AlreadyExists` for the session it just created.
    //!
    //! These are pure decoding tests over hand-built reply vectors — no
    //! Valkey, no fred client. The live mechanism (a real `CLIENT KILL` inside
    //! the WATCH window, a real racing writer) is driven end-to-end in
    //! `tests/integration/storage/b_bhb_tests.rs`; this module pins the
    //! decision table those tests exercise only one branch of per run.
    use super::*;
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};

    /// Shortest reply set an **armed** commit pipeline produces: `MULTI`, one
    /// queued mutation, `CLIENT ID`, `EXEC`. Restated here rather than read off
    /// a [`WatchedTxn`] so the decode tests need no client at all; the two are
    /// kept honest against each other by
    /// [`min_replies_tracks_whether_the_witness_is_queued`].
    const ARMED_MIN_REPLIES: usize = 4;
    /// Same for a **disarmed** pipeline, which queues no `CLIENT ID`
    /// ([`WitnessBaseline::Unavailable`]).
    const DISARMED_MIN_REPLIES: usize = 3;

    /// The shortest reply set a real armed pipeline can produce: `MULTI` + one
    /// queued mutation + `CLIENT ID` + `EXEC`, with `exec` substituted in as
    /// the last element.
    fn replies(exec: Result<Value, FredError>) -> Vec<Result<Value, FredError>> {
        vec![
            Ok(Value::from_static_str("OK")),
            Ok(Value::from_static_str("QUEUED")),
            Ok(Value::from_static_str("QUEUED")),
            exec,
        ]
    }

    /// [`decode`] exactly as [`WatchedTxn::exec`] calls it for a commit whose
    /// baseline was [`WitnessBaseline::Trusted`].
    fn decode_armed(exec: Result<Value, FredError>) -> WatchedExec {
        decode(replies(exec), ARMED_MIN_REPLIES, true)
    }

    fn fred_err(kind: FredErrorKind, msg: &'static str) -> FredError {
        FredError::new(kind, msg)
    }

    #[test]
    fn decode_null_exec_is_a_watch_conflict() {
        // Valkey answers `EXEC` with a nil bulk reply when a watched key
        // changed: nothing ran. This is the ordinary optimistic-concurrency
        // loss, and it MUST NOT reach any of the other three arms — reading it
        // as `Committed` is b.8q1 (the caller acks a sequence number for a
        // transaction that never executed).
        assert!(matches!(
            decode_armed(Ok(Value::Null)),
            WatchedExec::Conflict
        ));
    }

    #[test]
    fn decode_array_exec_is_committed_with_the_trailing_client_id_witness() {
        // The `EXEC` array holds one element per queued command, and this
        // module queues `CLIENT ID` LAST — so the array's tail is the id of
        // the connection that actually ran the transaction (L3).
        const WITNESS: i64 = 7_734;
        let exec = Value::Array(vec![
            Value::from_static_str("OK"),
            Value::from_static_str("1-0"),
            Value::Integer(WITNESS),
        ]);
        match decode_armed(Ok(exec)) {
            WatchedExec::Committed { witness } => assert_eq!(
                witness,
                Some(WITNESS),
                "the witness must be read from the LAST array element — reading any other \
                 position would compare the WATCH's connection id against an unrelated \
                 command's reply and manufacture a false `committed_unwatched`"
            ),
            _ => panic!("an EXEC array is a commit"),
        }
    }

    #[test]
    fn decode_array_exec_without_a_readable_witness_commits_with_none() {
        // A witness that is not an integer cannot name a connection. The
        // honest answer is `None` — `contradicted_by` treats that as "cannot
        // compare", never as an accusation — and the commit still stands,
        // because the array itself is the server saying every queued command
        // ran.
        match decode_armed(Ok(Value::Array(vec![]))) {
            WatchedExec::Committed { witness } => assert_eq!(witness, None),
            _ => panic!("an EXEC array is a commit even when the witness is unreadable"),
        }
    }

    #[test]
    fn decode_error_exec_is_an_exec_error_carrying_the_original_error() {
        // An error element inside the array that fred collapsed to the head
        // error (`WRONGTYPE`, an `XADD` refused for an equal-or-smaller id).
        // The transaction RAN; the caller must keep the non-retriable
        // classification rather than retry. (An `EXECABORT` is
        // `WatchedExec::Discarded`: nothing ran.)
        let err = fred_err(
            FredErrorKind::Unknown,
            "ERR The ID specified in XADD is equal or smaller",
        );
        match decode_armed(Err(err)) {
            WatchedExec::ExecError(e) => assert!(
                e.details().contains("equal or smaller"),
                "the original error must be propagated verbatim so the existing \
                 stream-collision / WRONGTYPE classifiers still recognise it; got {e}"
            ),
            _ => {
                panic!("an error at the EXEC position is an ExecError, not a conflict or a commit")
            }
        }
    }

    /// The refusal Valkey sends a write when `used_memory` is above
    /// `maxmemory` (`server.c`, `shared.oomerr`).
    const OOM_REPLY: &str = "OOM command not allowed when used memory > 'maxmemory'.";

    /// [`decode`] of a full armed reply set whose queued mutation was answered
    /// `queued` and whose `EXEC` was answered `exec`.
    fn decode_with_queued(
        queued: Result<Value, FredError>,
        exec: Result<Value, FredError>,
    ) -> WatchedExec {
        decode(
            vec![
                Ok(Value::from_static_str("OK")),
                queued,
                Ok(Value::from_static_str("QUEUED")),
                exec,
            ],
            ARMED_MIN_REPLIES,
            true,
        )
    }

    #[test]
    fn decode_execabort_after_a_queue_time_refusal_is_discarded_with_the_refusal_as_its_cause() {
        // A store write refusal rejects the queued write itself (`-OOM` in
        // its `+QUEUED` slot); Valkey flags the transaction and answers
        // `EXEC` with the generic `EXECABORT`. Nothing ran, so this is not the
        // torn `ExecError`, and the cause the caller classifies is the queued
        // command's refusal — the `EXEC` reply alone does not name it.
        let discarded = decode_with_queued(
            Err(fred_err(FredErrorKind::Unknown, OOM_REPLY)),
            Err(fred_err(
                FredErrorKind::Unknown,
                "EXECABORT Transaction discarded because of previous errors.",
            )),
        );
        match discarded {
            WatchedExec::Discarded { exec, cause } => {
                assert!(
                    exec.details().starts_with("EXECABORT "),
                    "`exec` is the EXEC reply; got {exec}"
                );
                assert_eq!(
                    cause.details(),
                    OOM_REPLY,
                    "the cause is the first queued command's error, verbatim"
                );
                assert_eq!(
                    crate::storage::client::store_write_refusal(&cause),
                    Some(crate::storage::client::StoreWriteRefusal::OutOfMemory),
                    "the cause classifies as a store write refusal"
                );
            }
            _ => panic!("an EXECABORT after a queue-time refusal is a discarded transaction"),
        }
    }

    #[test]
    fn decode_execabort_of_the_exec_itself_is_discarded_with_the_exec_reply_as_its_cause() {
        // A write-gated `EXEC` is rejected as a whole (`rejectCommand` →
        // `execCommandAbort`): every queued command answered `+QUEUED`, and
        // the `EXEC` reply names the refusal after its prefix. With no queued
        // error, the cause is the `EXEC` reply itself.
        let reply =
            "EXECABORT Transaction discarded because of: OOM command not allowed when used \
                     memory > 'maxmemory'.";
        let discarded = decode_with_queued(
            Ok(Value::from_static_str("QUEUED")),
            Err(fred_err(FredErrorKind::Unknown, reply)),
        );
        match discarded {
            WatchedExec::Discarded { exec, cause } => {
                assert_eq!(exec.details(), reply);
                assert_eq!(
                    cause.details(),
                    reply,
                    "with no queue-time refusal the cause is the EXEC reply"
                );
                assert_eq!(
                    crate::storage::client::store_write_refusal(&cause),
                    Some(crate::storage::client::StoreWriteRefusal::OutOfMemory),
                    "the refusal after the EXECABORT prefix classifies as a store write refusal"
                );
            }
            _ => panic!("an EXEC rejected as a whole is a discarded transaction"),
        }
    }

    #[test]
    fn decode_execabort_with_any_other_cause_is_still_discarded() {
        // Discarded is about what ran (nothing), not about why: a queue-time
        // rejection that is no store refusal (an unknown command) still
        // discards the whole transaction. Its cause is not a store write
        // refusal, so the caller keeps its non-retriable answer.
        let discarded = decode_with_queued(
            Err(fred_err(
                FredErrorKind::Unknown,
                "ERR unknown command 'NOSUCH', with args beginning with: ",
            )),
            Err(fred_err(
                FredErrorKind::Unknown,
                "EXECABORT Transaction discarded because of previous errors.",
            )),
        );
        match discarded {
            WatchedExec::Discarded { cause, .. } => {
                assert!(cause.details().starts_with("ERR unknown command"));
                assert_eq!(crate::storage::client::store_write_refusal(&cause), None);
            }
            _ => panic!("an EXECABORT is a discarded transaction whatever its cause"),
        }
    }

    #[test]
    fn decode_execabort_for_the_store_not_serving_carries_a_transient_cause() {
        // b.xd1: the call sites classify a discarded transaction from its
        // cause, so the store's `-LOADING` / `-BUSY` — a queued command
        // refused, or the `EXEC` itself refused behind a long script — must
        // reach them as the cause, where the shared classifier calls it
        // transient and the call is retried.
        const LOADING: &str = "LOADING Valkey is loading the dataset in memory";
        const EXECABORT_BUSY: &str = "EXECABORT Transaction discarded because of: BUSY Valkey is \
                                      busy running a script. You can only call SCRIPT KILL or \
                                      SHUTDOWN NOSAVE.";
        for (queued, exec, want) in [
            (
                Err(fred_err(FredErrorKind::Unknown, LOADING)),
                fred_err(
                    FredErrorKind::Unknown,
                    "EXECABORT Transaction discarded because of previous errors.",
                ),
                LOADING,
            ),
            (
                Ok(Value::from_static_str("QUEUED")),
                fred_err(FredErrorKind::Unknown, EXECABORT_BUSY),
                EXECABORT_BUSY,
            ),
        ] {
            match decode_with_queued(queued, Err(exec)) {
                WatchedExec::Discarded { cause, .. } => {
                    assert_eq!(cause.details(), want);
                    assert!(
                        crate::storage::client::is_transient_fred_error(&cause),
                        "{want:?} ran nothing and clears by itself: transient"
                    );
                }
                _ => panic!("an EXECABORT is a discarded transaction: {want:?}"),
            }
        }
    }

    // ── b.vn4: the cluster check's `-TRYAGAIN` to the `EXEC` ───────────────
    //
    // A slot mid-migration refuses an `EXEC` whose queued keys are split
    // between its source and target: Valkey discards the transaction, then
    // answers the bare `-TRYAGAIN` in the `EXEC` position. Nothing ran, so it
    // is `Discarded`, its own cause — never the torn `ExecError`, whose tear
    // handling (an ERROR line, a record retract, a reference-index recount)
    // would run for a commit that did not happen. Only Valkey's own reply,
    // under `Unknown` with the token's trailing space, may match: the unsafe
    // direction is reading "nothing ran" into a reply that does not prove it.

    #[test]
    fn decode_tryagain_exec_is_discarded_with_the_refusal_as_its_cause() {
        use crate::storage::client::TRYAGAIN_REPLY;
        let disarmed = decode(
            vec![
                Ok(Value::from_static_str("OK")),
                Ok(Value::from_static_str("QUEUED")),
                Err(fred_err(FredErrorKind::Unknown, TRYAGAIN_REPLY)),
            ],
            DISARMED_MIN_REPLIES,
            false,
        );
        for (pipeline, decoded) in [
            (
                "armed",
                decode_armed(Err(fred_err(FredErrorKind::Unknown, TRYAGAIN_REPLY))),
            ),
            ("disarmed", disarmed),
        ] {
            match decoded {
                WatchedExec::Discarded { exec, cause } => {
                    assert_eq!(exec.details(), TRYAGAIN_REPLY, "{pipeline}: the EXEC reply");
                    assert_eq!(
                        cause.details(),
                        TRYAGAIN_REPLY,
                        "{pipeline}: with every command queued, the cause is the refusal itself"
                    );
                    assert!(
                        crate::storage::client::is_transient_fred_error(&cause),
                        "{pipeline}: the cause is transient, so the caller retries on a fresh \
                         WATCH"
                    );
                }
                _ => panic!(
                    "{pipeline}: an EXEC refused -TRYAGAIN ran nothing; it is a discarded \
                     transaction, never a torn commit"
                ),
            }
        }
    }

    #[test]
    fn decode_tryagain_is_matched_by_kind_and_leading_token_only() {
        use crate::storage::client::TRYAGAIN_REPLY;
        // Anti-creep: each of these stays the torn `ExecError`.
        for (kind, msg) in [
            // The token without its trailing space.
            (FredErrorKind::Unknown, "TRYAGAIN"),
            // The reply's text under a kind fred never gives it.
            (FredErrorKind::Protocol, TRYAGAIN_REPLY),
            (FredErrorKind::Cluster, TRYAGAIN_REPLY),
            // Not in the RESP error-code position.
            (
                FredErrorKind::Unknown,
                "ERR something about TRYAGAIN Multiple keys request during rehashing of slot",
            ),
            // The cluster check's other refusals: fred never delivers them in
            // the `EXEC` position (a `CLUSTERDOWN` frame drops the connection,
            // a `MOVED` / `ASK` is a redirect the txn pool cannot follow), so
            // nothing keyed on `ErrorKind::Cluster` may match.
            (FredErrorKind::Cluster, "CLUSTERDOWN The cluster is down"),
            (FredErrorKind::Cluster, "MOVED 3999 127.0.0.1:6381"),
            (FredErrorKind::Cluster, "ASK 3999 127.0.0.1:6381"),
            // An error fred builds itself, of the `Cluster` kind
            // (`modules/inner.rs:332`), in the `EXEC` position.
            (FredErrorKind::Cluster, "Missing cluster routing state."),
        ] {
            assert!(
                matches!(
                    decode_armed(Err(fred_err(kind.clone(), msg))),
                    WatchedExec::ExecError(_)
                ),
                "{kind:?} {msg:?} in the EXEC position must stay an ExecError"
            );
        }
    }

    #[test]
    fn decode_a_single_error_is_in_doubt_whatever_its_text() {
        use crate::storage::client::TRYAGAIN_REPLY;
        // fred answers a pipeline whose redirect or attempt budget ran out, or
        // whose connection dropped, with ONE error for the whole call: a short
        // reply set, so whether the `EXEC` ran is unknown. Not even a
        // `-TRYAGAIN` text there may read as "nothing ran".
        for (kind, msg) in [
            (FredErrorKind::Routing, "Too many redirections."),
            (FredErrorKind::Unknown, TRYAGAIN_REPLY),
            (FredErrorKind::Cluster, "CLUSTERDOWN The cluster is down"),
        ] {
            match decode(
                vec![Err(fred_err(kind.clone(), msg))],
                ARMED_MIN_REPLIES,
                true,
            ) {
                WatchedExec::InDoubt(e) => {
                    assert_eq!(e.details(), msg, "{kind:?}: the error survives verbatim")
                }
                _ => panic!("{kind:?} {msg:?} as the whole call's one error is in-doubt"),
            }
        }
    }

    #[test]
    fn decode_execabort_behind_a_queued_tryagain_is_discarded_with_that_cause() {
        use crate::storage::client::TRYAGAIN_REPLY;
        // A queued command refused `-TRYAGAIN` flags the transaction, and the
        // `EXEC` answers the generic `EXECABORT`: the `EXECABORT` arm, with the
        // refusal as cause. (Valkey 9.0.3 counts a transaction's keys at the
        // `EXEC`, not while queueing, so this shape is defensive; the new
        // `-TRYAGAIN` arm must not take it over.)
        match decode_with_queued(
            Err(fred_err(FredErrorKind::Unknown, TRYAGAIN_REPLY)),
            Err(fred_err(
                FredErrorKind::Unknown,
                "EXECABORT Transaction discarded because of previous errors.",
            )),
        ) {
            WatchedExec::Discarded { exec, cause } => {
                assert!(exec.details().starts_with("EXECABORT "));
                assert_eq!(cause.details(), TRYAGAIN_REPLY);
            }
            _ => panic!("an EXECABORT is a discarded transaction, whatever its cause"),
        }
    }

    #[test]
    fn decode_execabort_on_a_disarmed_pipeline_is_discarded() {
        // No `CLIENT ID` witness queued: `MULTI`, one mutation, `EXEC`.
        let discarded = decode(
            vec![
                Ok(Value::from_static_str("OK")),
                Err(fred_err(FredErrorKind::Unknown, OOM_REPLY)),
                Err(fred_err(
                    FredErrorKind::Unknown,
                    "EXECABORT Transaction discarded because of previous errors.",
                )),
            ],
            DISARMED_MIN_REPLIES,
            false,
        );
        match discarded {
            WatchedExec::Discarded { cause, .. } => assert_eq!(cause.details(), OOM_REPLY),
            _ => panic!("an EXECABORT is a discarded transaction on a disarmed pipeline too"),
        }
    }

    #[test]
    fn decode_execabort_is_matched_by_kind_and_leading_token_only() {
        // Anti-creep: an error that merely mentions EXECABORT, or one of a
        // kind fred never gives an EXECABORT, stays the torn `ExecError` —
        // whose tear handling a wrongly-discarded transaction would skip.
        for (kind, msg) in [
            (
                FredErrorKind::Unknown,
                "ERR something about EXECABORT Transaction discarded",
            ),
            (
                FredErrorKind::Protocol,
                "EXECABORT Transaction discarded because of previous errors.",
            ),
            (FredErrorKind::Unknown, "EXECABORTED"),
        ] {
            assert!(
                matches!(
                    decode_armed(Err(fred_err(kind.clone(), msg))),
                    WatchedExec::ExecError(_)
                ),
                "{kind:?} {msg:?} must stay an ExecError"
            );
        }
    }

    #[test]
    fn decode_non_array_non_null_exec_is_an_exec_error() {
        // Nothing else is a legal `EXEC` reply. Treating an unexpected shape
        // as a commit would report success on a reply the service cannot
        // interpret; treating it as in-doubt would send a possibly-landed
        // commit back through the retry loop. It is neither: the reply set was
        // complete, so the transaction is over.
        assert!(matches!(
            decode_armed(Ok(Value::from_static_str("OK"))),
            WatchedExec::ExecError(_)
        ));
    }

    #[test]
    fn decode_short_reply_set_is_in_doubt_and_carries_the_write_error() {
        // fred collapses a pipeline write failure to a single `Err`
        // (`clients/pipeline.rs:263-288`), so the reply *count* is the whole
        // signal that the transaction never produced a verdict. Whether `EXEC`
        // ran is unknown — this must be `InDoubt`, never `Conflict`, or a
        // landed commit is applied twice.
        let err = fred_err(FredErrorKind::Canceled, "Connection closed.");
        match decode(vec![Err(err)], ARMED_MIN_REPLIES, true) {
            WatchedExec::InDoubt(e) => assert_eq!(
                e.kind(),
                &FredErrorKind::Canceled,
                "the underlying fred error must survive: the layer call site feeds it to \
                 `map_transient_fred_err`, so a substituted kind would change a retriable \
                 `Aborted` into a terminal `Internal`"
            ),
            _ => panic!("a short reply set is in-doubt"),
        }
    }

    #[test]
    fn decode_short_all_ok_reply_set_is_in_doubt_with_a_synthesised_error() {
        // A truncated-but-error-free vector is not a shape fred is documented
        // to produce; there is no error to propagate, so the arm has to
        // synthesise one rather than fall through to indexing a missing EXEC.
        match decode(
            vec![Ok(Value::from_static_str("OK"))],
            ARMED_MIN_REPLIES,
            true,
        ) {
            WatchedExec::InDoubt(e) => assert_eq!(e.kind(), &FredErrorKind::Protocol),
            _ => panic!("a short reply set is in-doubt whatever it contains"),
        }
    }

    #[test]
    fn decode_multi_error_is_an_exec_error_rather_than_a_commit() {
        // `results[0]` is the `MULTI` frame. An error there means the server
        // never opened a transaction, so every mutation behind it ran as an
        // ordinary unconditional command — a torn, unwatched commit. Reading
        // the trailing `EXEC` frame at face value would report that as a
        // perfectly good `Committed` (with a witness that matches, because the
        // connection never changed), which is the one shape the L3 witness
        // cannot catch: nothing about the *connection* went wrong.
        let err = fred_err(FredErrorKind::Unknown, "ERR MULTI calls can not be nested");
        let mut results = replies(Ok(Value::Array(vec![
            Value::from_static_str("OK"),
            Value::Integer(4_242),
        ])));
        results[0] = Err(err);
        match decode(results, ARMED_MIN_REPLIES, true) {
            WatchedExec::ExecError(e) => assert!(
                e.details().contains("MULTI"),
                "the MULTI error itself must be propagated so the call sites' non-retriable \
                 classification names what actually failed; got {e}"
            ),
            _ => panic!(
                "a failed MULTI means the queued mutations ran unconditionally — that is an \
                 execution error, never a commit, a conflict, or an in-doubt"
            ),
        }
    }

    #[test]
    fn decode_disarmed_never_reads_a_mutation_reply_as_a_witness() {
        // The regression this argument pair exists for. With no `CLIENT ID`
        // queued, the last element of the `EXEC` array is a *mutation's* reply
        // — and `HSET` answers with an integer, which is indistinguishable
        // from a connection id by shape alone. Reading it as a witness would
        // compare an `HSET` field count against the baseline, find them
        // unequal, and manufacture a `committed_unwatched` accusation on every
        // single commit taken on a store where L3 is disarmed: exactly the
        // fleet-wide false-positive the disarm exists to avoid.
        //
        // Note the reply set is only THREE long here (`MULTI`, one mutation,
        // `EXEC`) and must still decode as complete — a `min_replies` fixed at
        // the armed shape would call this truncated and send a landed commit
        // into the in-doubt reconciliation path.
        let exec = Value::Array(vec![Value::Integer(1)]);
        let disarmed = vec![
            Ok(Value::from_static_str("OK")),
            Ok(Value::from_static_str("QUEUED")),
            Ok(exec),
        ];
        match decode(disarmed, DISARMED_MIN_REPLIES, false) {
            WatchedExec::Committed { witness } => assert_eq!(
                witness, None,
                "a disarmed commit has no witness at all, so nothing may ever be accused from \
                 it. `Some(1)` here is the HSET reply misread as a connection id"
            ),
            _ => panic!("a complete disarmed reply set whose EXEC is an array is a commit"),
        }
    }

    #[test]
    fn baseline_from_probe_exec_decides_every_reply_shape() {
        // The full decision table for the L3 calibration probe. Getting any
        // row wrong is a fleet-scale failure in one direction or the other:
        // a false `Trusted` arms the witness against an id nothing will match
        // (every commit accused), and a false `Unavailable` silently drops L3
        // on a store that supports it.
        let table: Vec<(&str, Result<Value, FredError>, WitnessBaseline)> = vec![
            (
                "two equal integers — the only row that arms L3",
                Ok(Value::Array(vec![Value::Integer(77), Value::Integer(77)])),
                WitnessBaseline::Trusted(77),
            ),
            (
                "two DIFFERENT integers: the store renumbers CLIENT ID per command, so no \
                 witness could ever match one and every commit would read as unwatched",
                Ok(Value::Array(vec![Value::Integer(77), Value::Integer(78)])),
                WitnessBaseline::Unavailable(BASELINE_UNSTABLE),
            ),
            (
                "an element that is not an integer at all",
                Ok(Value::Array(vec![
                    Value::Integer(77),
                    Value::from_static_str("not-an-id"),
                ])),
                WitnessBaseline::Unavailable(BASELINE_UNEXPECTED_SHAPE),
            ),
            (
                "the wrong arity — two CLIENT IDs were queued, so anything but two elements \
                 means the probe did not run the way this code believes it did",
                Ok(Value::Array(vec![Value::Integer(77)])),
                WitnessBaseline::Unavailable(BASELINE_UNEXPECTED_SHAPE),
            ),
            (
                "a non-array, non-null EXEC reply",
                Ok(Value::from_static_str("OK")),
                WitnessBaseline::Unavailable(BASELINE_UNEXPECTED_SHAPE),
            ),
            (
                "a nil EXEC. Nothing was watched at probe time, so this is not a concurrency \
                 outcome — it is a store answering in a shape this code cannot interpret",
                Ok(Value::Null),
                WitnessBaseline::Unavailable(BASELINE_EXEC_DISCARDED),
            ),
            (
                "an EXEC error — EXECABORT, which is what a store that ACL-denies or \
                 proxy-rejects CLIENT ID inside MULTI answers. THE load-bearing row: arming \
                 the witness here would make every real commit EXECABORT too, turning every \
                 CreateSession and every structural UpdateLayer into a terminal INTERNAL",
                Err(fred_err(
                    FredErrorKind::Unknown,
                    "EXECABORT Transaction discarded because of previous errors.",
                )),
                WitnessBaseline::Unavailable(BASELINE_EXEC_ERROR),
            ),
        ];

        for (why, exec, expected) in table {
            assert_eq!(
                baseline_from_probe_exec(exec),
                expected,
                "probe EXEC row {why:?} decoded to the wrong baseline"
            );
        }
    }

    #[test]
    fn only_a_trusted_baseline_arms_the_witness_and_reports_no_reason() {
        // `armed()` decides whether `CLIENT ID` is queued into the REAL commit,
        // and `unavailable_reason()` is what the metric cell and the log line
        // carry. They must agree, in both directions, for every variant: an
        // armed baseline that also reports a reason would tick
        // `witness_unavailable` on a healthy store, and a disarmed one that
        // reports none would leave the disarm invisible.
        assert!(WitnessBaseline::Trusted(5).armed());
        assert_eq!(WitnessBaseline::Trusted(5).unavailable_reason(), None);
        for reason in [
            BASELINE_UNSTABLE,
            BASELINE_EXEC_ERROR,
            BASELINE_EXEC_DISCARDED,
            BASELINE_UNEXPECTED_SHAPE,
            BASELINE_SHORT_REPLY,
        ] {
            let baseline = WitnessBaseline::Unavailable(reason);
            assert!(
                !baseline.armed(),
                "queueing the witness under {reason:?} is what makes a real commit EXECABORT \
                 on the very stores this variant identifies"
            );
            assert_eq!(baseline.unavailable_reason(), Some(reason));
        }
    }

    /// `min_replies` must follow the instance, not a constant.
    ///
    /// Both directions are a live defect. Sized for the armed shape while
    /// disarmed, a *complete* three-frame reply set reads as truncated and a
    /// landed commit is sent into the in-doubt reconciliation path. Sized for
    /// the disarmed shape while armed, a genuinely truncated four-frame
    /// pipeline is decoded as if the `EXEC` frame had arrived.
    ///
    /// `Client::default()` never dials, and it does not need to: queueing into
    /// a fred `Pipeline` is a local buffer push that responds `QUEUED` early
    /// (`fred-10.1.0/src/clients/pipeline.rs:96-114`), so `open` reaches the
    /// wire only at `exec`, which this test never calls.
    #[tokio::test]
    async fn min_replies_tracks_whether_the_witness_is_queued() {
        let client = Client::default();
        let hash = ClusterHash::from("le:{bhb}:obj:hdr");

        let armed = WatchedTxn::open(&client, &hash, &WitnessBaseline::Trusted(1))
            .await
            .expect("queueing MULTI into a pipeline never touches the wire");
        assert!(armed.witness_armed);
        assert_eq!(
            armed.min_replies(),
            ARMED_MIN_REPLIES,
            "MULTI + one mutation + CLIENT ID + EXEC"
        );

        let disarmed = WatchedTxn::open(
            &client,
            &hash,
            &WitnessBaseline::Unavailable(BASELINE_EXEC_ERROR),
        )
        .await
        .expect("queueing MULTI into a pipeline never touches the wire");
        assert!(!disarmed.witness_armed);
        assert_eq!(
            disarmed.min_replies(),
            DISARMED_MIN_REPLIES,
            "no CLIENT ID is queued, so a complete reply set is one frame shorter — reading it \
             as truncated would send a landed commit into in-doubt reconciliation"
        );
    }

    #[test]
    fn fingerprint_still_holds_only_for_an_unchanged_live_connection() {
        assert!(WatchFingerprint::Slot(1).still_holds(&WatchFingerprint::Slot(1)));
        assert!(!WatchFingerprint::Slot(1).still_holds(&WatchFingerprint::Slot(2)));
        assert!(
            !WatchFingerprint::Slot(1).still_holds(&WatchFingerprint::Unnamed),
            "fred holding no id for the connection is not evidence the WATCH still holds"
        );
        assert!(
            !WatchFingerprint::Unnamed.still_holds(&WatchFingerprint::Unnamed),
            "`Unnamed` must never compare as held on EITHER side — two unknowns are not a \
             match. It is routed through `comparable_with` first precisely because \
             `still_holds` answers `false` here, and `false` alone would be read as `Refuse`"
        );
        assert!(WatchFingerprint::AnyConnection(9).still_holds(&WatchFingerprint::AnyConnection(9)));
        assert!(
            !WatchFingerprint::AnyConnection(9).still_holds(&WatchFingerprint::AnyConnection(10)),
            "the fallback digest moves on any reconnect anywhere — conservative by design"
        );
    }

    #[test]
    fn fingerprint_comparability_splits_cannot_tell_from_not_held() {
        // The predicate pair [`dispatch_verdict`] composes once it is past the
        // `is_connected` gate, and the reason it is a pair: `still_holds`
        // answers `false` for BOTH "the connection was replaced" and "fred
        // never learned this connection's id". Collapsing them is a 100%
        // outage on any store that ACL-denies or proxy-rejects `CLIENT ID` at
        // handshake — `connection_ids()` is permanently empty there, so every
        // `CreateSession` and every structural `UpdateLayer` would refuse to
        // dispatch and run to retry exhaustion, forever.
        for other in [
            WatchFingerprint::Slot(1),
            WatchFingerprint::AnyConnection(9),
            WatchFingerprint::Unnamed,
        ] {
            assert!(
                !WatchFingerprint::Unnamed.comparable_with(&other),
                "`Unnamed` on the recorded side is un-comparable against everything, {other:?} \
                 included"
            );
            assert!(
                !other.comparable_with(&WatchFingerprint::Unnamed),
                "…and on the fresh side too: {other:?} vs `Unnamed`"
            );
        }
        assert!(WatchFingerprint::Slot(1).comparable_with(&WatchFingerprint::Slot(2)));
        assert!(
            WatchFingerprint::Slot(1).comparable_with(&WatchFingerprint::AnyConnection(9)),
            "two NAMED fingerprints are always comparable even across variants — the answer \
             is a genuine `false` from `still_holds`, i.e. `Refuse`, not `cannot tell`"
        );
    }

    #[test]
    fn dispatch_verdict_refuses_a_client_fred_reports_as_not_connected() {
        // `Client::default()` is a fully-constructed client that has never
        // dialed: `is_connected()` reads `inner.state == Disconnected` and
        // `connection_ids()` is empty. That is the same pair of local reads
        // [`dispatch_verdict`] takes in production, so this pins the one row
        // of its decision table that needs no live server — **a commit is
        // never dispatched onto a client fred does not consider connected.**
        // Dispatching there is precisely the b.bhb failure mode: a `WATCH`
        // cannot outlive its connection, so the `EXEC` would commit
        // unconditionally.
        //
        // `Refuse` and not `DispatchUncomparable` is the load-bearing half.
        // `WatchFingerprint::take` on this client yields `Unnamed`, so the
        // comparability arm would also match — and answering it would send the
        // commit onto a dead client. The `is_connected` gate must therefore be
        // evaluated FIRST, which is what this asserts.
        //
        // The recorded fingerprint is varied across all three variants because
        // the gate must refuse on every one of them — including `Slot`, the
        // variant that *would* compare equal if the connection map still held
        // the id.
        let dead = Client::default();
        let hash = ClusterHash::from("le:{bhb}:obj:hdr");
        for recorded in [
            WatchFingerprint::Slot(42),
            WatchFingerprint::AnyConnection(7),
            WatchFingerprint::Unnamed,
        ] {
            assert_eq!(
                dispatch_verdict(&dead, &recorded, &hash),
                DispatchVerdict::Refuse,
                "a client fred does not consider connected must be refused for every recorded \
                 fingerprint, `Unnamed` included — the un-comparable arm may only be reached \
                 on a LIVE connection; got a non-Refuse verdict for {recorded:?}"
            );
        }

        // What this test deliberately does NOT claim: that the
        // `is_connected()` conjunct is load-bearing *on its own*. Isolating it
        // needs the state to be `Connecting` while the backchannel map still
        // holds the OLD connection's id — fred's mid-handshake window, which
        // opens inside `reconnect_once` and closes when the new socket is up
        // (a few hundred microseconds on loopback). No seam in this repo can
        // hold a call site inside it, so that row is unreachable by
        // construction rather than merely untested. The two arms this cannot
        // reach are covered against a real server in
        // `tests/integration/storage/b_bhb_tests.rs`: `Refuse` on a genuinely
        // killed connection, and `DispatchUncomparable` (→
        // `fingerprint_unavailable`, commit still succeeds) on a fixture whose
        // fred clients authenticate as an ACL user denied the `client` command
        // family.
    }

    #[test]
    fn witness_baseline_contradicted_by_accuses_only_a_trusted_mismatch() {
        // The single input that may tick the alertable `committed_unwatched`
        // cell and compensate a `create_session`. Every other row must answer
        // `false`, and each for its own reason.
        assert!(
            WitnessBaseline::Trusted(1).contradicted_by(Some(2)),
            "a witness naming a connection other than the one the WATCH's own probe sampled \
             is PROOF the transaction ran elsewhere"
        );
        assert!(
            !WitnessBaseline::Trusted(1).contradicted_by(Some(1)),
            "the ordinary healthy commit"
        );
        assert!(
            !WitnessBaseline::Trusted(1).contradicted_by(None),
            "an unreadable witness is not evidence of anything; accusing on it would \
             compensate a perfectly good CreateSession"
        );
        for reason in [
            BASELINE_UNSTABLE,
            BASELINE_EXEC_ERROR,
            BASELINE_EXEC_DISCARDED,
            BASELINE_UNEXPECTED_SHAPE,
            BASELINE_SHORT_REPLY,
        ] {
            let baseline = WitnessBaseline::Unavailable(reason);
            assert!(
                !baseline.contradicted_by(Some(2)),
                "L3 is disarmed under {reason:?}: there is no id to compare against, so an \
                 accusation here would page an operator on a property of the store"
            );
            assert!(!baseline.contradicted_by(None));
        }
    }
}
