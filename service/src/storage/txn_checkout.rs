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

// Checkout hygiene for the `txn_pool`: prove a freshly-acquired exclusive
// connection is free of leaked transaction state before the borrower opens
// their own `MULTI` (b.7td).
//
// ## Why a probe exists at all
//
// The field report was an `AUTH` rejected by the server with "AUTH ... not
// allowed in transactions", followed by credential-refresh timeouts,
// "Unresponsive connection" events and a `DEADLINE_EXCEEDED` cascade on a
// cluster-mode managed backing store. The tempting reading — "the periodic
// re-AUTH interleaved itself into our MULTI" — is **false** for fred 10.1.0:
// `MULTI`..`EXEC` is shipped as ONE router command, so nothing can interleave
// between them. An `AUTH` rejected *inside a transaction* is therefore not the
// fault; it is the first thing to *touch* a connection that was already
// carrying an open `MULTI` left behind by an earlier borrower, and it is the
// most likely thing to touch it because the credential-refresh path fans out to
// every primary.
//
// Two further fred facts make the leak permanent once it happens:
//
// - **Transaction error recovery cannot resynchronize a connection.** On the
//   transaction write path `skip_results()` is a no-op (the raw frames never
//   enter the buffer it drains) and `discard()` reads exactly one frame. A
//   connection whose reply stream has drifted out of step with its command
//   stream stays that way.
// - **fred never rebuilds a dirty-but-healthy socket.**
//   `drop_broken_connections` only acts on transport errors, `force_reconnection()`
//   routes through the same path, and transaction frames never enter the
//   in-flight queue the unresponsive check judges, so it is blind to them.
//
// So the residue cannot be cleaned in place by fred and cannot be detected by
// fred. Both halves have to be ours: this module detects, and
// `ValkeyClient::acquire_txn*` repairs by recycling the slot.
//
// ## The probe
//
// Two argument-free commands, issued back to back on the just-acquired guard,
// each pinned to the borrower's own cluster slot and each bounded by
// [`TXN_CHECKOUT_PROBE_TIMEOUT`]:
//
// 1. `DISCARD` — the verdict. On a clean connection the server answers
//    `-ERR DISCARD without MULTI`; **that error IS the clean verdict.** A `+OK`
//    is only reachable when a `MULTI` was already open, which is precisely the
//    leak. Anything else means the reply stream is offset.
// 2. `UNWATCH` — the repair for the *other* residue shape. A leaked `WATCH`
//    with no `MULTI` behind it produces no distinguishable reply (there is no
//    "WATCH state" query in the protocol), so it cannot be detected — but it
//    can be cleared unconditionally, and by then step 1 has proven no `MULTI`
//    is open, so this `UNWATCH` executes rather than being queued into one.
//    This is belt-and-braces behind b.ji8's slot-pinned teardown: it covers the
//    residue b.ji8's `Drop` funnel gives up on (`site="drop"` ticked) and
//    anything that bypassed a `WatchGuard` entirely.
//
// Both are idempotent and O(1) server-side. `DISCARD` runs first deliberately:
// run the other way around, `UNWATCH` on a connection with an open `MULTI`
// answers `+QUEUED` and clears nothing.
//
// ## Cost, and who actually pays it
//
// Two round-trips per `txn_pool` acquisition — and the `txn_pool` is a much
// narrower surface than "every write":
//
// - **`UpdateLayer`'s default backend never acquires a slot.** The Lua fast
//   path dispatches its `EVALSHA` on the multiplexed data pool
//   (`storage::layer`, `self.client.pool.next()`).
// - **The highest-frequency acquirer is the `update_layer` WATCH/MULTI/EXEC
//   slow path** (`storage::layer`) — structural-command deltas, resets, and
//   `LIVE_EDIT_UPDATE_LAYER_BACKEND=txn`. One slot per RETRY ATTEMPT, up to
//   `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (default 128) per call, so a
//   contended object multiplies the probe by its conflict rate rather than by
//   its request rate. `create_layer` is next: two checkouts per call (the
//   mapping + layers-index transaction and the object materialization), each
//   paid once.
// - **`store_values` no longer acquires at all.** Until b.rej it was the
//   highest-frequency acquirer by a wide margin — one slot per unique blob,
//   `STORE_VALUES_MAX_INFLIGHT` (8) in flight, so a `CreateLayer` carrying N
//   unique external values paid `2N` extra round-trips, `⌈N / 8⌉ × 2` of them
//   on the critical path. Its per-hash `MULTI`..`EXEC` is now one `EVALSHA` of
//   `scripts/ext_store_value.lua` on the multiplexed data pool, which needs no
//   exclusive session because Lua is already atomic. A blob store is one
//   round-trip again, and the whole `2N` term is gone.
//
// Measured 2026-09-01 on the sandbox bridge network (single-node,
// unauthenticated, no TLS): an argument-free `DISCARD` / `UNWATCH` round-trips
// in **30–33 µs** serially (2 000 iterations each), so the probe's floor is
// ≈ 63 µs per checkout. **That ratio does not transfer to the deployment this
// ticket came from.** On a TLS'd cluster-mode managed store the round-trip, not
// the work, is the unit of cost, so the same two probes become a leading term
// of whatever they are attached to rather than a rounding error — which is why
// b.rej detached them from the per-blob path entirely.
//
// One honest caveat on that number. The bench (`benches/ext_mirror_store_values`,
// which drove `store_values` and therefore no longer exercises the probe at all
// post-b.rej) could never express the probe as a *delta*: the probe landed
// before the bench was ever run here, so criterion had no pre-probe baseline in
// `target/criterion/` to diff against, and its own run-to-run spread (10 samples
// against a shared backing store; ±40 % confidence intervals) is far wider than
// the effect. The 63 µs above is arithmetic over the measured per-command
// round-trip, not a measured difference.
//
// It is still the right trade — the same one the stream multiplexer's readers
// make with their idle→active connection probe (one `PING` before a reader's
// first XREAD after it had nothing to read, or on a cluster after it moved to
// another hash slot;
// `storage::stream_mux::READER_PROBE_TIMEOUT`) — because the failure it
// prevents is strictly worse than its cost: an undetected dirty slot does not
// cost one request, it costs every request that lands on that slot until the
// pod restarts, and it presents as an unrelated `CROSSSLOT` /
// `DEADLINE_EXCEEDED` on a caller that did nothing wrong. The probe is not
// conditioned on cluster mode, auth mode, or a rollback knob for the same
// reason the reader's is not: detection must not depend on which regime the
// deployment happens to be in.
//
// ## The *repair* is rate-limited; the *detection* never is
//
// Every checkout probes and every dirty verdict is counted. What is gated is
// the `quit()` + `init()` recycle, and only for the three verdicts that do not
// prove anything about the connection:
//
// - **State-proof** — [`TXN_DIRTY_REASON_LEAKED_MULTI`] and
//   [`TXN_DIRTY_REASON_UNEXPECTED_REPLY`]. The probe saw concrete evidence (a
//   `DISCARD` that was acknowledged; a reply of a shape no verdict allows), so
//   the slot is known-corrupt and is recycled immediately, with no rate limit.
//   Suppressing one of these on a *timer* would hand a provably poisoned
//   connection to a borrower — the exact outcome the module exists to prevent.
//   The only thing that stops one is proof that the repair does not work; see
//   the breaker below.
// - **Unknown-state** — [`TXN_DIRTY_REASON_PROBE_TIMEOUT`],
//   [`TXN_DIRTY_REASON_PROBE_ERROR`] and [`TXN_DIRTY_REASON_TOPOLOGY_REPLY`].
//   The first two say only "no answer arrived", which
//   is *also* what a store-wide latency event says. Under one of those every
//   checkout across the pool times out at once, and an ungated repair would
//   answer a slow backing store with a pool-wide `QUIT` + reconnect storm —
//   compounding, since `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128)
//   re-acquires per attempt and the b.7td field report is *exactly* the regime
//   where an argument-free command stops answering inside 650 ms.
//   [`should_recycle_unknown_state`] therefore admits the first of a streak and
//   every [`TXN_RECYCLE_REARM_EVERY`]th thereafter, mirroring
//   `storage::stream_mux::reader::should_force_reconnect` — same shape, same reason,
//   and the streak is process-wide across the pool because the storm is
//   pool-wide.
//
//   [`TXN_DIRTY_REASON_TOPOLOGY_REPLY`] joins them for a different reason
//   (b.289): its frame WAS received, but what it proves is that the *node*
//   declined the command — a resharding, a `LOADING` replica, a demoted
//   primary — not that this connection's reply stream is offset. Filing it
//   state-proof would let a routine topology change latch the breaker below,
//   whose escalation accuses the classifier. Same rate-limited repair, same
//   pool-wide streak, because a topology change is a pool-wide event too. It is
//   also the only `reason` cell an operator should ever expect to be non-zero.
//
// **What the gate trades away.** A single genuinely wedged slot that lands
// inside a suppression window is handed to its borrower unrepaired: their
// commands run on a connection whose state is unknown and will fail under their
// own timeouts and retry discipline, and the next checkout re-probes it. That
// is bounded (one borrower, their own deadline) and self-healing (the streak
// re-arms). The alternative — recycling unconditionally — is unbounded and
// self-amplifying, because the input that triggers it is the backing store
// being slow. Suppression is never silent: the `reason` cell is ticked either
// way and the log event carries `suppressed=true`, so the counter still means
// "the probe found this slot unfit" and only the repair is throttled.
//
// **Log level follows the repair, not the verdict** (b.7td round 3). An admitted
// recycle logs at `warn!`; a suppressed one logs the same fields at `debug!`.
// The counter is what makes suppression non-silent, and it has to be: the regime
// the gate exists for is a store-wide latency event, which turns EVERY checkout
// in the pool dirty while `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128)
// re-enters the checkout per retry attempt — a per-checkout WARN there floods
// the log pipeline at the moment it is least able to absorb it, describing a
// decision that was *not* to act. The diagnostic string is built lazily
// ([`DirtyDetail`]) so a suppressed verdict under an `info` filter formats
// nothing at all.
//
// ## When the repair itself is the problem
//
// One recycle clears real residue: an abandoned `MULTI` dies with the socket and
// the next checkout reads `Clean`, so a state-proof streak never gets past 1.
// The immediate recycle inverts, though, when the verdict comes from the
// **classifier and the server disagreeing** rather than from residue. Suppose a
// managed proxy renders the clean `DISCARD` answer without naming the command
// (`-ERR without multi`): [`is_discard_without_multi`] — deliberately strict
// since round 2 — files it as [`TXN_DIRTY_REASON_UNEXPECTED_REPLY`], the slot is
// recycled, and the FRESH connection answers exactly the same way. Every
// checkout that lands on an endpoint rendering the reply that way then recycles,
// for the life of the deployment. That is the same unbounded self-amplification
// the unknown-state gate bounds, reached through the one class the gate
// deliberately does not cover.
//
// [`state_proof_repair_is_ineffective`] is the breaker for it, and it is
// evidence-driven rather than periodic: after [`TXN_STATE_PROOF_REPAIR_ATTEMPTS`]
// state-proof verdicts against **one cluster node** since the last `Clean`
// checkout against that node (both state-proof reasons count toward the one
// streak; interleaved unknown-state or moot verdicts neither advance nor reset
// it) — i.e. after rebuilt connections have re-produced the verdict — the
// repair stops for that class on that node, the slot is
// handed over as-is, and the checkout escalates: a distinct `error!` naming
// classifier-vs-server disagreement as the likely cause and carrying the node in
// its `node` field, plus
// `valkey_txn_repair_ineffective_total{reason}`, which ticks on every withheld
// verdict (the `error!` is emitted on the tripping verdict and every
// [`TXN_RECYCLE_REARM_EVERY`]th after it, same anti-flood reasoning as above).
//
// **The streak is per node** (b.289 sub-task 4), and only a `Clean` checkout
// against the SAME node re-arms it — pool-wide, a `Clean` against any healthy
// primary reset the streak, so a single misbehaving node could recycle forever
// and the breaker could never latch. `storage::client`'s
// `ValkeyClient::probe_node_bucket` names the node from the borrower's own
// hash key, so the bucket always describes the connection the verdict is about.
// Verdicts whose node cannot be named — a centralized or proxied endpoint,
// which never has one; a clustered client whose cached routing view is
// momentarily absent or does not cover the slot; the keyless
// `ValkeyClient::acquire_txn` shim — share ONE fallback bucket, and **any**
// `Clean` checkout re-arms that bucket, whichever node it was pinned to. On a
// single-endpoint deployment that is the whole map, so the contract there reads
// exactly as the pre-b.289 pool-wide one did; on a cluster it is what keeps a
// transient unnameable window from latching a streak that no later `Clean` can
// reach. Bump and clear resolve the bucket through one shared helper
// (`ValkeyClient::bucket_key`) so the two can never disagree.
//
// **What the breaker trades away.** Once it is latched for a node, a genuinely
// leaked `MULTI` arriving on that node in the same window is handed to its
// borrower unrepaired — the outcome this module exists to prevent. That is
// accepted only because the precondition is three state-proof verdicts against
// that node (no intervening `Clean` against it) with rebuilds between them: on a
// node where recycling works that state is unreachable (the rebuild clears the
// residue and the next verdict is `Clean`), so the breaker can only latch where
// the repair is already demonstrably not repairing anything. Choosing the other
// way — recycle forever — does not save that borrower either, since their slot
// is rebuilt into a connection that answers identically, and it costs a quit +
// reconnect per checkout per retry attempt on top. The second thing traded, and
// it is the price of making the breaker able to latch at all, is that a node
// whose probes all read `Clean` can no longer re-arm a *sibling* node's withheld
// repair: hygiene stays switched off for the affected node until that node
// itself answers a probe correctly, and the rest of the cluster cannot vouch for
// it. The escalation is deliberately loud, names the node, and
// deliberately names the classifier, because what resolves the state that
// latches this is a classifier change or an endpoint configuration change, not
// something the service can retry its way out of.
//
// ## Routing
//
// `DISCARD` and `UNWATCH` carry no key, so fred's default `ClusterHash::FirstKey`
// policy finds nothing to hash and the router falls through to
// `writers.values_mut().next()` — an ARBITRARY node (fred-10.1.0
// `src/router/mod.rs`). That is the same defect b.ji8 fixed for teardown, and
// it is fatal here too: probing node A proves nothing about the node the
// borrower's transaction will run on. [`probe_txn_slot`] therefore takes the
// borrower's own key and pins both commands to its slot via
// `CustomCommand::cluster_hash`, which `CommandKind::custom_hash_slot()` reads
// ahead of every other hashing policy.
//
// `hash_key = None` is the shim case — see [`ValkeyClient::acquire_txn`]. It
// leaves fred's default routing in place, which is exact on a centralized or
// proxied single-endpoint deployment and best-effort on a cluster.
//
// **Redirects are never followed on this pool.** The `txn` pool is configured
// `max_command_attempts = 1, max_redirections = 0` (SDD §15.3), and
// [`issue_probe`] leaves `Options::max_redirections` at `None`, which
// `Options::apply` treats as "inherit" (`fred-10.1.0/src/types/config.rs:1450-1452`)
// — so the probe runs on the pool's zero. `decr_check_redirections`
// (`protocol/command.rs:1789-1796`) therefore errs `Routing "Too many
// redirections."` on the FIRST `MOVED` / `ASK` the router does intercept,
// rather than chasing it to another node. That is the property that makes an
// intercepted redirect unconditionally benign here (an unknown-state
// `probe_error`, streak-gated) — not the probe's own `max_attempts = 1`, which
// bounds replays of the *same* command, not redirect hops. The replies the
// router does NOT intercept are the subject of
// [`is_topology_transition_reply`]; [`is_state_proof_reason`] carries the
// verified matrix of which is which.
//
// ## Driving the verdicts in a test
//
// Five of the six outcomes this module can reach are unreachable against a
// healthy Valkey: there is no way to make a real server answer `DISCARD` with
// an integer, no way to make it answer with a `-MOVED`, and no way to make an
// argument-free command hang. The [`ProbeFaultInjector`] seam — compiled only
// under `#[cfg(any(test, feature = "test-fault-injection"))]`, so it is absent
// from the release library — swaps a canned outcome in for the round-trip,
// **keyed per probe command** so the two distinct `unexpected_reply` arms
// (`DISCARD`'s and `UNWATCH`'s), and the two `topology_reply` arms beside them,
// can be driven apart. `ProbeFault::simple_error` is the faithful driver for
// the topology arms: it produces the same `Ok(Resp3Frame::SimpleError { .. })`
// `custom_raw` hands back for an un-intercepted server error.
//
// A fault can be a frame, an arbitrary `fred` error kind, or a
// delay; a delay longer than [`TXN_CHECKOUT_PROBE_TIMEOUT`] runs through the
// *real* `tokio::time::timeout`, so the resulting `Timeout` is produced by the
// production mechanism rather than asserted into existence. With no injector
// installed — or with its queue drained — every probe takes the real path, so
// the seam is invisible to every other test.

use fred::clients::Client;
use fred::error::ErrorKind;
use fred::types::config::Options;
use fred::types::{ClusterHash, CustomCommand, Resp3Frame, Value};
use std::time::Duration;

use crate::observability::metrics::{
    TXN_DIRTY_REASON_LEAKED_MULTI, TXN_DIRTY_REASON_PROBE_ERROR, TXN_DIRTY_REASON_PROBE_TIMEOUT,
    TXN_DIRTY_REASON_TOPOLOGY_REPLY, TXN_DIRTY_REASON_UNEXPECTED_REPLY, TXN_STATE_PROOF_REASONS,
};
use crate::storage::client::FredErrOrigin;

/// Per-command deadline carried by both probe commands.
///
/// Sized on the same argument as `storage::watch_guard::UNWATCH_ATTEMPT_TIMEOUT`
/// and deliberately **not** derived from `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`
/// (5 000 ms by default): the probe's whole job is to fail *fast* on a wedged
/// connection so the slot can be recycled, and inheriting a knob that an
/// operator may raise for large-payload commands would turn a checkout into a
/// multi-second stall on exactly the connections that are already broken.
///
/// `DISCARD` and `UNWATCH` take no arguments and touch no data, so 650 ms is
/// several orders of magnitude above any healthy round-trip; a timeout here is
/// a wedge verdict, not a slow link. Worst case for the whole probe is two
/// attempts — [`TXN_CHECKOUT_PROBE_CEILING`].
pub(crate) const TXN_CHECKOUT_PROBE_TIMEOUT: Duration = Duration::from_millis(650);

/// Wall-clock ceiling the whole probe can add to one `txn_pool` checkout: both
/// commands timing out back to back. Exposed so the recycle path in
/// `storage::client` can be sized against it and so a reviewer can bound the
/// checkout's worst case without re-deriving it.
pub(crate) const TXN_CHECKOUT_PROBE_CEILING: Duration =
    TXN_CHECKOUT_PROBE_TIMEOUT.saturating_mul(2);

/// Re-arm interval for *unknown-state* slot recycles, in unknown-state
/// verdicts since the last `Clean` checkout (interleaved state-proof or moot
/// verdicts neither advance nor reset the streak) (b.7td round 2).
///
/// The recycle fires on the **first** unknown-state verdict of a streak and
/// again on every Nth thereafter, exactly as
/// `storage::stream_mux::reader::FORCE_RECONNECT_REARM_EVERY` gates a stream
/// reader's timeout-driven forced reconnects — and for the same reason, one
/// layer down. A
/// [`TXN_DIRTY_REASON_PROBE_TIMEOUT`] means "an argument-free command did not
/// answer in 650 ms", which on a healthy backing store means a wedged
/// connection but during a store-wide latency event means *the store*. Ungated,
/// the second case turns every checkout across the pool into a `QUIT` +
/// reconnect, which is load added to a backing store that is already the
/// bottleneck — and `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128) re-enters
/// the checkout per attempt, so the amplification is per-retry, not per-request.
///
/// Held at the same 8 as the reader's re-arm deliberately: both are "one
/// repair per streak, plus a periodic re-arm in case the first repair raced a
/// transition", and a reviewer comparing the two should find one number, not
/// two that need reconciling.
pub(crate) const TXN_RECYCLE_REARM_EVERY: u32 = 8;

/// Whether an unknown-state dirty verdict at the given streak length (counted
/// since the last `Clean` checkout) should actually recycle the slot: the
/// first of the streak, then every [`TXN_RECYCLE_REARM_EVERY`]th. `streak` is
/// 1-based.
///
/// Applies **only** to the three unknown-state reasons —
/// [`TXN_DIRTY_REASON_PROBE_TIMEOUT`], [`TXN_DIRTY_REASON_PROBE_ERROR`] and
/// [`TXN_DIRTY_REASON_TOPOLOGY_REPLY`] (b.289 sub-task 3, which routed the
/// third one through this gate) — see [`is_state_proof_reason`]. Deliberately
/// identical in shape to `storage::stream_mux::reader::should_force_reconnect`.
pub(crate) fn should_recycle_unknown_state(streak: u32) -> bool {
    streak >= 1 && (streak - 1).is_multiple_of(TXN_RECYCLE_REARM_EVERY)
}

/// Whether a dirty verdict's `reason` is one the probe *proved* rather than
/// merely failed to disprove.
///
/// State-proof verdicts ([`TXN_DIRTY_REASON_LEAKED_MULTI`],
/// [`TXN_DIRTY_REASON_UNEXPECTED_REPLY`]) rest on a reply that was actually
/// received and can only mean one thing, so they recycle immediately: the slot
/// is known-corrupt and handing it out is the failure this module exists to
/// prevent. The other three rest on something *weaker* than that — the absence
/// of a reply ([`TXN_DIRTY_REASON_PROBE_TIMEOUT`] /
/// [`TXN_DIRTY_REASON_PROBE_ERROR`]), which a slow backing store produces just
/// as readily as a wedged connection, or a reply about the *node* rather than
/// the connection ([`TXN_DIRTY_REASON_TOPOLOGY_REPLY`]) — so they go through
/// [`should_recycle_unknown_state`].
///
/// "Immediately", not "unconditionally" — [`state_proof_repair_is_ineffective`]
/// is the one thing that can stop a state-proof recycle, and only after the
/// repair has been *demonstrated* not to work.
///
/// ## Why cluster-topology replies are NOT state-proof (b.289 sub-task 3)
///
/// The premise this comment used to carry — "fred's router intercepts topology
/// replies before `custom_raw` returns" — is **false outright for four of the
/// seven prefixes, and only conditionally true for the other three** (the
/// conditions are the middle column of the table below), verified against the
/// pinned fred 10.1.0 / redis-protocol 6.0.0 sources. The
/// probe issues `DISCARD` / `UNWATCH` through `custom_raw`, whose reply path
/// (`commands::server::custom_raw` → `utils::request_response` →
/// `utils::basic_request_response`, `fred-10.1.0/src/utils.rs:411-441`) performs
/// **no** error-frame conversion: `responders.rs:442` does `tx.send(Ok(frame))`,
/// so any frame the router did not act on arrives at [`classify`] as
/// `Ok(Resp3Frame::SimpleError { .. })` — the success arm, which before this
/// change fell through to state-proof [`TXN_DIRTY_REASON_UNEXPECTED_REPLY`].
///
/// What the router actually intercepts, on the `txn` pool
/// (`max_command_attempts = 1`, `max_redirections = 0` — see the module docs'
/// **Routing** section):
///
/// | Reply | Interception | `custom_raw` returns | Verdict |
/// |---|---|---|---|
/// | `-MOVED <slot> <host:port>` | intercepted **only when canonical**: exactly 3 space-separated tokens AND the `MOVED` prefix in exact case — `is_redirection` (`redis-protocol-6.0.0/src/utils.rs:67-73`) is a case-**sensitive** `starts_with` plus `payload.split(' ').count() == 3` — and only on a clustered client, since `clustered::process_response_frame`'s `frame.is_redirection()` (`router/clustered.rs:322-325`) sits behind the `is_clustered()` split at `router/utils.rs:274`. Then `redirect_command` → `process_moved` (`router/commands.rs:288-306`) errs because `decr_check_redirections` (`protocol/command.rs:1789-1796`) has none left | `Err(Routing)`, or `Err(Timeout)` if the `sync_cluster` it kicks off outlasts the 650 ms budget | `probe_error` / `probe_timeout` |
/// | `-ASK <slot> <host:port>` | same conditions, same case-sensitive 3-token test, via `process_ask` (`router/commands.rs:260-264`) | `Err(Routing)` | `probe_error` |
/// | `-CLUSTERDOWN …` | intercepted **unconditionally, case-sensitively**: `is_clusterdown_error` (`router/responses.rs:256-279`) tests `data.trim().starts_with("CLUSTERDOWN")` on `SimpleError` AND `BlobError` with no token-count condition, and it runs from `check_fatal_errors` (`:282-292`) → `preprocess_frame` (`:298-306`) on **every** read frame (`router/utils.rs:259`), i.e. ahead of the clustered/centralized split. Then disconnect → `check_final_write_attempt` (`router/utils.rs:153-172`) | `Err(Cluster)` | `probe_error` |
/// | `-NOAUTH` / `-WRONGPASS` | `parse_auth_error` (`router/responses.rs:215-227`), live because we set `reconnect_on_auth_error` on every pool | `Err(Auth)` | `probe_error` |
/// | `-MOVED` / `-ASK` at any **other** token count | **none** — see the 3-token requirement above | `Ok(SimpleError)` | **this arm** |
/// | `-moved` / `-ask` / `-clusterdown` in any non-canonical **case** | **none** — every test above is `starts_with` on the exact spelling; this is the only escape `CLUSTERDOWN` has | `Ok(SimpleError)` | **this arm** |
/// | `-LOADING` / `-TRYAGAIN` / `-MASTERDOWN` / `-READONLY` | **none** — `check_global_reconnect_errors`'s default `[ClusterDown, Loading, ReadOnly]` list lives behind fred's `custom-reconnect-errors` feature (`router/responses.rs:229`, `types/config.rs:493-495`), which is not a fred default (`fred Cargo.toml:82-85`) and is not enabled here; the compiled definition is the no-op stub at `responses.rs:251-254` | `Ok(SimpleError)` | **this arm** |
///
/// [`is_topology_transition_reply`] matches its leading token
/// ASCII-case-**insensitively**, so for all seven prefixes it is strictly WIDER
/// than fred's interception — every reply fred acts on is one this classifier
/// would also have caught, and the casing escapes above are caught only here.
/// That is the fail-toward-unknown-state direction, and it is deliberate. The
/// one rendering neither side catches is a code that is not the leading token
/// at all (`-ERR MOVED 1234 host:port`): fred's `starts_with` misses it and so
/// does our first-word match, and it falls through to `unexpected_reply`.
///
/// So a routine resharding, a `LOADING` replica coming back from an RDB load,
/// or a `READONLY` write to a demoted primary can each put a raw error frame in
/// a probe slot. Filed as state-proof, three of them against one node with no
/// intervening `Clean` against that node latch
/// [`state_proof_repair_is_ineffective`] — a breaker whose `error!` accuses the
/// *classifier* — for the duration of an event the operator did on purpose. And
/// a resharding is exactly the window in which the node often cannot be named
/// at all, so those verdicts land in the shared fallback bucket, where a latch
/// would be at its widest.
///
/// **The trade this makes.** A `MOVED`-shaped frame in the `DISCARD` slot could
/// also be an offset reply stream that happens to be carrying somebody else's
/// redirect, and filing it unknown-state means that slot is recycled only on
/// the first of its streak and every [`TXN_RECYCLE_REARM_EVERY`]th after.
/// That costs at most one borrower a re-probe on the next checkout. Filing it
/// state-proof costs a latched breaker — for the node the probe was pinned to,
/// or for the shared fallback bucket when it could not be named — during every
/// topology change, and the cell ticks either way, so the evidence an operator
/// needs survives in both directions and only the repair cadence differs.
pub(crate) fn is_state_proof_reason(reason: &str) -> bool {
    TXN_STATE_PROOF_REASONS.contains(&reason)
}

/// Leading tokens of the server error replies that mean "this node will not
/// serve this command right now" rather than "this connection is broken"
/// (b.289 sub-task 3).
///
/// `MOVED` / `ASK` / `CLUSTERDOWN` are here even though fred *does* intercept
/// their canonical renderings, because each interception has an escape that
/// lands the frame in the probe's classifier instead:
///
/// - `MOVED` / `ASK` — intercepted only on a clustered client, and only when
///   `is_redirection` matches: a case-**sensitive** prefix AND exactly three
///   space-separated tokens. So `-MOVED 1234 host:port (proxied)`, a bare
///   `-MOVED` with no arguments, or a lowercase `-moved …` all sail past it.
/// - `CLUSTERDOWN` — has **no** token-count escape: `is_clusterdown_error`
///   runs unconditionally on every read frame, ahead of the
///   clustered/centralized split. Its `starts_with` is case-sensitive too,
///   though, so a non-canonically-cased `-clusterdown …` reaches this
///   classifier. That casing gap is its whole reachability story.
///
/// The other four (`LOADING`, `TRYAGAIN`, `MASTERDOWN`, `READONLY`) are never
/// intercepted at all on this build. Because
/// [`is_topology_transition_reply`] compares case-insensitively, this set is
/// strictly wider than fred's interception for all seven — the
/// fail-toward-unknown-state direction. See [`is_state_proof_reason`] for the
/// per-reply provenance and the full matrix.
const TOPOLOGY_TRANSITION_PREFIXES: &[&str] = &[
    "MOVED",
    "ASK",
    "CLUSTERDOWN",
    "LOADING",
    "TRYAGAIN",
    "MASTERDOWN",
    "READONLY",
];

/// Whether a server error payload announces a cluster-topology / node-state
/// transition (b.289 sub-task 3).
///
/// Matches the **leading whitespace-delimited token**, ASCII-case-insensitively,
/// against [`TOPOLOGY_TRANSITION_PREFIXES`]. Leading-token and not
/// `contains(..)`: RESP error codes are the first word by protocol convention,
/// and a substring match would reclassify a genuinely offset reply stream that
/// merely happens to mention one of these words in free text — e.g. an
/// `-ERR unknown command 'READONLY'` — as a benign topology event, which is the
/// [`is_discard_without_multi`] laundering bug (b.7td round 2) reintroduced one
/// classifier over.
fn is_topology_transition_reply(detail: &str) -> bool {
    let Some(token) = detail.split_ascii_whitespace().next() else {
        return false;
    };
    TOPOLOGY_TRANSITION_PREFIXES
        .iter()
        .any(|prefix| token.eq_ignore_ascii_case(prefix))
}

/// The textual payload of an error frame, if the frame is one.
///
/// Covers `BlobError` alongside `SimpleError` because RESP3 lets a server send
/// either for the same condition and the **topology** classifier must not
/// diverge on the encoding: a `$`-framed `-MOVED` is the same topology event as
/// a `-`-framed one, and reading one of them as `unexpected_reply` would file a
/// routine reshard as state-proof. Renders `BlobError`'s bytes lossily — the
/// payload is only ever fed to [`is_topology_transition_reply`], which reads the
/// first token, and never to a log (that stays [`describe_frame`]'s job).
///
/// **The asymmetry with the CLEAN-verdict classifier is deliberate.**
/// [`is_discard_without_multi`] — the only other payload classifier here — is
/// reachable only from the `SimpleError` arm of [`probe_txn_slot`]'s `DISCARD`
/// match and stays `SimpleError`-only on purpose. Widening it would be the *fail-open*
/// direction b.7td round 2 closed: it turns a dirty verdict into `Clean`, so a
/// new frame shape reaching it hands a possibly-poisoned slot to a borrower.
/// This helper only ever moves a verdict from one DIRTY reason to another
/// (`unexpected_reply` → `topology_reply`, changing the repair cadence and
/// nothing else), which is why widening it is safe and widening that one is not.
fn error_frame_detail(frame: &Resp3Frame) -> Option<std::borrow::Cow<'_, str>> {
    match frame {
        Resp3Frame::SimpleError { data, .. } => Some(std::borrow::Cow::Borrowed(data)),
        Resp3Frame::BlobError { data, .. } => Some(String::from_utf8_lossy(data.as_ref())),
        _ => None,
    }
}

/// Whether a reply frame is an error frame announcing a topology transition.
/// Thin composition of [`error_frame_detail`] and
/// [`is_topology_transition_reply`], written once so both probe classifiers
/// cannot drift.
fn is_topology_transition_frame(frame: &Resp3Frame) -> bool {
    error_frame_detail(frame).is_some_and(|detail| is_topology_transition_reply(&detail))
}

/// How many state-proof verdicts **against one cluster node** (since the last
/// `Clean` checkout against that node — both state-proof reasons count toward
/// one streak, and interleaved unknown-state or moot verdicts neither advance
/// nor reset it) are repaired before the breaker concludes the repair is not
/// working (b.7td round 3; made per-node by b.289 sub-task 4 — see
/// [`state_proof_repair_is_ineffective`] for the bucketing rule, including the
/// shared fallback bucket any `Clean` re-arms).
///
/// Three, and the number that matters is what it implies on the far side:
/// verdicts 1..=3 each get a recycle, so the 4th verdict of the streak is the
/// third one produced by a **freshly rebuilt connection** (state-proof verdicts
/// 2, 3 and 4 each land on a socket rebuilt since the previous one). Three
/// post-recycle re-observations of the same answer is the evidence; one could be
/// a race with a borrower dying mid-transaction again, and the module's whole
/// premise is that a single rebuild clears real residue.
/// See [`state_proof_repair_is_ineffective`].
pub(crate) const TXN_STATE_PROOF_REPAIR_ATTEMPTS: u32 = 3;

/// Whether a state-proof verdict at the given consecutive-streak length should
/// have its recycle **withheld** because rebuilding the connection has already
/// been shown not to clear it. `streak` is 1-based and counts state-proof
/// verdicts (both reasons together) **against one cluster node** since the last
/// `Clean` checkout against that node; interleaved unknown-state or moot
/// verdicts neither advance nor reset it.
///
/// **The one-shot repair path is untouched.** Streaks 1..=[`TXN_STATE_PROOF_REPAIR_ATTEMPTS`]
/// recycle exactly as before; this only fires from the streak after that.
///
/// **What re-arms it** (b.289 sub-task 4). The streak lives in
/// `storage::client`'s per-node map, keyed by the node the probe was pinned to.
/// A `Clean` checkout against THAT node clears its bucket and re-arms the
/// repair for it; a `Clean` against a different node deliberately does not,
/// which is the whole reason the breaker can latch on a single misbehaving
/// primary rather than being reset forever by its healthy siblings. Verdicts
/// whose node cannot be named — a centralized or proxied endpoint, a routing
/// view that is momentarily absent or does not cover the slot, the keyless
/// `ValkeyClient::acquire_txn` shim — share one fallback bucket that **any**
/// `Clean` checkout re-arms, so a single-endpoint deployment behaves exactly as
/// the pre-b.289 pool-wide contract did and a transient unnameable window on a
/// cluster cannot latch a streak nothing can clear.
///
/// **Why a breaker exists at all.** Recycling unconditionally is correct for
/// real residue — one rebuild clears an abandoned `MULTI` and the next checkout
/// reads `Clean`, so the streak never gets past 1. It is self-amplifying when
/// the verdict comes from the *classifier* and the server disagreeing instead:
/// a managed proxy that renders the `DISCARD` error without naming `DISCARD`
/// (`-ERR without multi`, say) falls through to
/// [`TXN_DIRTY_REASON_UNEXPECTED_REPLY`] forever, and a freshly built connection
/// answers identically — so every checkout landing on that endpoint recycles
/// without bound, for as long as the deployment renders that reply that way.
/// That is the same failure shape [`should_recycle_unknown_state`] bounds one
/// verdict class over, reached by a different route.
///
/// The two are deliberately *not* the same mechanism. The unknown-state gate
/// re-arms periodically because its input (a slow store) is expected to pass;
/// this one latches until a `Clean` proves the connection can answer correctly,
/// because its input (a reply the classifier cannot accept) does not heal on a
/// timer and re-trying it on a cadence would just re-open the storm at 1/8th
/// rate.
pub(crate) fn state_proof_repair_is_ineffective(streak: u32) -> bool {
    streak > TXN_STATE_PROOF_REPAIR_ATTEMPTS
}

/// Whether a withheld state-proof repair at this streak should be escalated
/// loudly (a distinct `error!`) rather than recorded quietly at `debug!`.
///
/// The counter ticks on **every** withheld verdict — that is the non-silent
/// record — but the log does not, for the same reason the suppressed
/// unknown-state arm dropped to `debug!` in round 3: the regime that trips this
/// breaker is one where *every* checkout in the pool produces the verdict, and
/// `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128) re-enters the checkout per
/// retry attempt, so a per-checkout `error!` is a log flood at exactly the
/// moment the pipeline is least able to absorb one. So: loud on the verdict that
/// trips the breaker, then again every [`TXN_RECYCLE_REARM_EVERY`]th, matching
/// the cadence the sibling gate already publishes.
pub(crate) fn should_escalate_ineffective_repair(streak: u32) -> bool {
    state_proof_repair_is_ineffective(streak)
        && (streak - TXN_STATE_PROOF_REPAIR_ATTEMPTS - 1).is_multiple_of(TXN_RECYCLE_REARM_EVERY)
}

/// What the checkout probe concluded about a freshly-acquired `txn_pool` slot.
///
/// Three arms, not two, for the same reason
/// `storage::watch_guard::TeardownOutcome` has three: a failure whose fred
/// error kind says the connection is already gone is not evidence of leaked
/// state — a `MULTI` and a `WATCH` both die with their connection — and
/// counting it would page an operator on routine failover, managed-proxy
/// recycles and ordinary reconnects.
#[derive(Debug)]
pub(crate) enum CheckoutVerdict {
    /// The slot answered both probes exactly as a clean connection must. Any
    /// residual `WATCH` was cleared in passing.
    Clean,
    /// A probe failed with an error that says the connection carrying any
    /// residue is already gone: a lost connection's error kind, or fred's
    /// routing miss (the router holds no connection at all). Nothing to clear,
    /// nothing to count; fred's own reconnect machinery owns the recovery.
    MootConnectionGone(fred::error::Error),
    /// The slot is carrying leaked transaction state, or its reply stream is
    /// offset, or it is wedged. The caller must recycle it before handing it to
    /// the borrower.
    Dirty {
        /// Stable `reason` label for `valkey_txn_dirty_checkouts_total`.
        reason: &'static str,
        /// Human-readable detail for the log event. Carries reply shapes and
        /// fred error text only — never key contents, never credential
        /// material.
        detail: DirtyDetail,
    },
}

/// The diagnostic half of a [`CheckoutVerdict::Dirty`], rendered **on demand**
/// (b.7td round 3).
///
/// Was a `String` formatted eagerly inside [`classify`]. That cost was paid on
/// every dirty verdict including the ones the recycle gate then suppressed — and
/// the regime the gate exists for is a store-wide latency event, where every
/// checkout in the pool goes dirty and
/// `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128) re-enters the checkout per
/// retry attempt. Holding the *inputs* instead and formatting through
/// [`std::fmt::Display`] means the string is built only if a log level that
/// wants it is enabled: `tracing`'s macros evaluate their field expressions
/// after the callsite's enabled check, so a suppressed verdict logged at
/// `debug!` under an `info` filter formats nothing at all.
///
/// [`std::fmt::Debug`] delegates to `Display` rather than being derived, so a
/// `{:?}` of a verdict cannot print a reply frame's payload bytes — see
/// [`describe_frame`], which is the only renderer any arm uses.
pub(crate) enum DirtyDetail {
    /// `DISCARD` was acknowledged: a `MULTI` was already open.
    LeakedMulti,
    /// A probe answered with a frame its verdict does not allow. `command` is
    /// the probe's log-facing name; `frame` is rendered by [`describe_frame`],
    /// never printed whole.
    UnexpectedReply {
        command: &'static str,
        frame: Resp3Frame,
    },
    /// A probe was answered with an un-intercepted cluster-topology /
    /// node-state error frame (b.289). Same fields and same renderer as
    /// [`DirtyDetail::UnexpectedReply`] — a distinct arm only so the log line
    /// says what actually happened, since the two carry opposite operator
    /// advice (routine during a resharding vs. investigate the classifier).
    TopologyReply {
        command: &'static str,
        frame: Resp3Frame,
    },
    /// A probe hit [`TXN_CHECKOUT_PROBE_TIMEOUT`] without answering.
    ProbeTimeout { command: &'static str },
    /// A probe failed with a fred error that does not say the connection is
    /// gone.
    ProbeError {
        command: &'static str,
        error: fred::error::Error,
    },
}

impl std::fmt::Display for DirtyDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirtyDetail::LeakedMulti => f.write_str(
                "DISCARD was acknowledged, so a MULTI was already open on this pooled connection \
                 when it was checked out",
            ),
            DirtyDetail::UnexpectedReply { command, frame } => {
                let expectation = if command.starts_with(PROBE_DISCARD) {
                    "which is neither +OK nor the 'without MULTI' error"
                } else {
                    "which is not +OK"
                };
                write!(
                    f,
                    "{command} answered {}, {expectation}; the connection's reply stream is \
                     offset against its command stream",
                    describe_frame(frame)
                )
            }
            DirtyDetail::TopologyReply { command, frame } => write!(
                f,
                "{command} answered {}, a cluster-topology / node-state transition reply that \
                 fred's router did not intercept; the node declined to serve the command, which \
                 says nothing about this connection's own state",
                describe_frame(frame)
            ),
            DirtyDetail::ProbeTimeout { command } => write!(
                f,
                "{command} exceeded its {TXN_CHECKOUT_PROBE_TIMEOUT:?} probe deadline"
            ),
            DirtyDetail::ProbeError { command, error } => {
                write!(f, "{command} failed: kind={:?}: {error}", error.kind())
            }
        }
    }
}

impl std::fmt::Debug for DirtyDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

/// Errors that mean the connection is already gone, so any leaked transaction
/// state went with it: the error kinds fred reports for a connection it lost
/// ([`ErrorKind::Canceled`], [`ErrorKind::IO`]), and fred's routing miss
/// ([`is_router_no_connection_error`](crate::storage::client::is_router_no_connection_error)).
///
/// A routing miss means the client's router holds no connection at all: the
/// probes carry a cluster hash, never a named node, and `Router::route`
/// returns `None` for such a command only when it has no connection to hand
/// back (`fred-10.1.0/src/router/mod.rs:176-207`; a cluster client falls back
/// to any connection it holds). A `MULTI` or a `WATCH` lives on a connection,
/// so none can be left. The usual cause is the vendored fred's change 9: a
/// fred `Transaction` cut partway through its write closes its connection and
/// defers the reconnect, so a borrower that takes the slot in between can send
/// its probe ahead of that reconnect; the reconnect is already queued ahead of
/// the borrower's next command.
///
/// Mirrors `storage::watch_guard::watch_died_with_the_connection`, including
/// its exclusion of [`ErrorKind::Timeout`]: a command that expired without a
/// reply leaves the connection's state *unknown*, which is exactly the case
/// worth recycling and worth counting.
fn residue_died_with_the_connection(err: &fred::error::Error) -> bool {
    matches!(err.kind(), ErrorKind::Canceled | ErrorKind::IO)
        || crate::storage::client::is_router_no_connection_error(err)
}

/// Name of the first probe command. A `&'static str` rather than an enum
/// because it is also the WARN's `command` prefix and the `CustomCommand`'s
/// wire name; the fault seam keys off the same value so a test and the
/// production path cannot disagree about which probe they mean.
const PROBE_DISCARD: &str = "DISCARD";
/// Name of the second probe command. See [`PROBE_DISCARD`].
const PROBE_UNWATCH: &str = "UNWATCH";

// ── Probe fault-injection seam (b.7td round 2) ──────────────────────────────
//
// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`.
// The release library has neither the types nor the branch; `ProbeFaults`
// degenerates to a zero-sized unit struct so no `#[cfg]` leaks into
// `ValkeyClient`'s field list or its constructor.

/// One canned outcome for one probe round-trip (b.7td round 2).
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug)]
pub enum ProbeFault {
    /// Answer the probe with this exact RESP frame.
    Frame(Resp3Frame),
    /// Fail the probe with this exact fred error, of any [`ErrorKind`].
    Error(fred::error::Error),
    /// Take this long to answer. Run through the *real*
    /// [`TXN_CHECKOUT_PROBE_TIMEOUT`], so a delay at or above it yields a
    /// genuine `ErrorKind::Timeout` produced by the production mechanism, and a
    /// delay below it resolves to the command's own clean reply.
    Delay(Duration),
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ProbeFault {
    /// `+OK` — the leak verdict from `DISCARD`, the clean verdict from
    /// `UNWATCH`.
    pub fn ok() -> Self {
        ProbeFault::Frame(Resp3Frame::SimpleString {
            data: bytes::Bytes::from_static(b"OK"),
            attributes: None,
        })
    }

    /// An arbitrary `+`-status reply.
    pub fn simple_string(text: &str) -> Self {
        ProbeFault::Frame(Resp3Frame::SimpleString {
            data: bytes::Bytes::from(text.to_string()),
            attributes: None,
        })
    }

    /// An arbitrary `-`-error reply — e.g. `"ERR DISCARD without MULTI"` (the
    /// clean verdict) or `"ERR EXEC without MULTI"` (which must NOT read as
    /// clean; see [`is_discard_without_multi`]).
    pub fn simple_error(text: &str) -> Self {
        ProbeFault::Frame(Resp3Frame::SimpleError {
            data: text.to_string().into(),
            attributes: None,
        })
    }

    /// An integer reply — the canonical "reply stream is offset" shape.
    pub fn number(value: i64) -> Self {
        ProbeFault::Frame(Resp3Frame::Number {
            data: value,
            attributes: None,
        })
    }

    /// A fred error of any kind: `Timeout`, `Canceled`, `IO`, `Auth`, …
    pub fn error(kind: ErrorKind, detail: &str) -> Self {
        ProbeFault::Error(fred::error::Error::new(kind, detail.to_string()))
    }

    /// A probe that never answers inside its deadline. Twice
    /// [`TXN_CHECKOUT_PROBE_TIMEOUT`] so the margin does not depend on timer
    /// granularity.
    pub fn hang() -> Self {
        ProbeFault::Delay(TXN_CHECKOUT_PROBE_TIMEOUT.saturating_mul(2))
    }

    /// Answer this fault, applying the real per-command deadline to the delay
    /// arm.
    async fn resolve(self, command: &'static str) -> Result<Resp3Frame, fred::error::Error> {
        match self {
            ProbeFault::Frame(frame) => Ok(frame),
            ProbeFault::Error(err) => Err(err),
            ProbeFault::Delay(delay) => {
                match tokio::time::timeout(TXN_CHECKOUT_PROBE_TIMEOUT, tokio::time::sleep(delay))
                    .await
                {
                    Ok(()) => Ok(clean_reply_for(command)),
                    Err(_elapsed) => Err(fred::error::Error::new(
                        ErrorKind::Timeout,
                        format!("injected {command} delay exceeded the probe deadline"),
                    )),
                }
            }
        }
    }
}

/// The reply a healthy server gives to `command` — used by the delay arm so a
/// sub-deadline delay is a *slow but clean* probe rather than a second failure
/// shape.
#[cfg(any(test, feature = "test-fault-injection"))]
fn clean_reply_for(command: &'static str) -> Resp3Frame {
    if command.starts_with(PROBE_DISCARD) {
        Resp3Frame::SimpleError {
            data: "ERR DISCARD without MULTI".to_string().into(),
            attributes: None,
        }
    } else {
        Resp3Frame::SimpleString {
            data: bytes::Bytes::from_static(b"OK"),
            attributes: None,
        }
    }
}

/// Per-probe-command queues of canned outcomes (b.7td round 2).
///
/// Keyed **per command**, not per checkout: `DISCARD` and `UNWATCH` reach two
/// distinct `unexpected_reply` arms, and a single shared queue could not drive
/// the `UNWATCH` one without first satisfying `DISCARD`. Each queue is consumed
/// front-to-back and an empty queue falls through to the real round-trip, so a
/// test can script "first checkout dirty, every checkout after it real".
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
pub struct ProbeFaultInjector {
    discard: std::sync::Mutex<std::collections::VecDeque<ProbeFault>>,
    unwatch: std::sync::Mutex<std::collections::VecDeque<ProbeFault>>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ProbeFaultInjector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue one outcome for the next `DISCARD` probe.
    pub fn push_discard(&self, fault: ProbeFault) -> &Self {
        if let Ok(mut q) = self.discard.lock() {
            q.push_back(fault);
        }
        self
    }

    /// Queue one outcome for the next `UNWATCH` probe.
    pub fn push_unwatch(&self, fault: ProbeFault) -> &Self {
        if let Ok(mut q) = self.unwatch.lock() {
            q.push_back(fault);
        }
        self
    }

    /// Queue the same outcome for the next `n` `DISCARD` probes — the shape the
    /// streak gate needs, since it takes several consecutive unknown-state
    /// verdicts to cross a re-arm boundary.
    pub fn push_discard_repeated(&self, n: usize, mut fault: impl FnMut() -> ProbeFault) {
        for _ in 0..n {
            self.push_discard(fault());
        }
    }

    /// How many canned outcomes are still queued across both commands. Zero
    /// means every subsequent probe takes the real path.
    pub fn remaining(&self) -> usize {
        let discard = self.discard.lock().map(|q| q.len()).unwrap_or(0);
        let unwatch = self.unwatch.lock().map(|q| q.len()).unwrap_or(0);
        discard + unwatch
    }

    /// Pop the next outcome for `command`, if any.
    fn take(&self, command: &'static str) -> Option<ProbeFault> {
        let queue = if command.starts_with(PROBE_DISCARD) {
            &self.discard
        } else {
            &self.unwatch
        };
        queue.lock().ok().and_then(|mut q| q.pop_front())
    }
}

/// The handle [`crate::storage::client::ValkeyClient`] carries.
///
/// Shared behind an `Arc` so an injector installed on one clone is visible to
/// every other clone — a `LayerStore` built before the injector was installed
/// must still see it, or the seam would be a construction-order trap.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Debug, Default)]
pub(crate) struct ProbeFaults {
    slot: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<ProbeFaultInjector>>>>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ProbeFaults {
    /// The inert seam — no injector installed, so every probe takes the real
    /// path. Named rather than `Default::default()` because the construction
    /// site in `ValkeyClient` is shared by both cfg shapes, and the release
    /// shape's unit struct makes `::default()` a clippy error
    /// (`default_constructed_unit_structs`).
    pub(crate) fn none() -> Self {
        Self::default()
    }

    /// Install (or replace) the injector every probe on this client consults.
    pub(crate) fn install(&self, injector: std::sync::Arc<ProbeFaultInjector>) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = Some(injector);
        }
    }

    fn take(&self, command: &'static str) -> Option<ProbeFault> {
        let injector = self.slot.lock().ok()?.clone()?;
        injector.take(command)
    }
}

/// Production stand-in for the fault seam: a zero-sized unit struct, so the
/// field on `ValkeyClient` costs nothing and needs no `#[cfg]` at its
/// construction site.
#[cfg(not(any(test, feature = "test-fault-injection")))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProbeFaults;

#[cfg(not(any(test, feature = "test-fault-injection")))]
impl ProbeFaults {
    /// Shape-independent counterpart of the test-shape constructor, so
    /// `ValkeyClient`'s construction site needs no `#[cfg]`.
    pub(crate) fn none() -> Self {
        Self
    }
}

/// Build one argument-free probe command, pinned to the borrower's slot when
/// one was supplied.
///
/// `ClusterHash::Custom(slot)` is read by `CommandKind::custom_hash_slot()`
/// ahead of `Options::cluster_hash` and ahead of the argument hasher, so
/// putting the slot here — rather than on the `Options` — is the unambiguous
/// placement for a `_Custom` command.
fn probe_command(name: &'static str, hash_key: Option<&str>) -> CustomCommand {
    let hash = match hash_key {
        // `ClusterHash::from(&str)` is `Custom(redis_keyslot(key))` — the same
        // conversion `WatchGuard::mark_watched` records the teardown slot with,
        // so probe routing and teardown routing cannot drift.
        Some(key) => ClusterHash::from(key),
        None => ClusterHash::FirstKey,
    };
    CustomCommand::new_static(name, hash, false)
}

/// Issue one probe command under [`TXN_CHECKOUT_PROBE_TIMEOUT`], returning the
/// raw RESP frame.
///
/// `custom_raw` (not `custom`) because the *frame shape* is the verdict: fred's
/// `Value` conversion turns `-ERR DISCARD without MULTI` into an
/// `ErrorKind::Unknown` error indistinguishable, at the type level, from a
/// dozen other server errors, whereas the raw frame lets a clean verdict be
/// recognised positively.
///
/// `max_attempts = 1` is load-bearing, not a tuning choice. The probe is only
/// idempotent in the sense that re-running it is *safe*; it is not idempotent
/// in its *verdict*. A retried `DISCARD` that already succeeded answers
/// `-ERR ... without MULTI` the second time, i.e. a retry would launder the
/// leaked-`MULTI` verdict into a clean one.
async fn issue_probe(
    client: &Client,
    name: &'static str,
    hash_key: Option<&str>,
    faults: &ProbeFaults,
) -> Result<Resp3Frame, fred::error::Error> {
    // Fault seam (b.7td round 2). Absent from release builds; a checkout with
    // no injector installed, or whose queue is drained, falls straight through
    // to the real round-trip below.
    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(fault) = faults.take(name) {
        return fault.resolve(name).await;
    }
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    let _ = faults;

    use fred::interfaces::ClientLike;
    let options = Options {
        timeout: Some(TXN_CHECKOUT_PROBE_TIMEOUT),
        max_attempts: Some(1),
        ..Default::default()
    };
    client
        .with_options(&options)
        .custom_raw(probe_command(name, hash_key), Vec::<Value>::new())
        .await
}

/// Render a reply frame for the WARN without ever printing payload bytes.
///
/// Only the frame *kind* and, for the two textual arms, the server's own status
/// / error string are emitted. A `txn_pool` connection can be mid-anything when
/// it is found dirty, so a `{:?}` of the frame could print key names or user
/// data; this keeps the diagnostic to the shape, which is what identifies the
/// fault.
fn describe_frame(frame: &Resp3Frame) -> String {
    match frame {
        Resp3Frame::SimpleString { data, .. } => {
            format!(
                "SimpleString({})",
                String::from_utf8_lossy(data.as_ref()).trim()
            )
        }
        Resp3Frame::SimpleError { data, .. } => format!("SimpleError({})", data.trim()),
        other => frame_kind_name(other).to_string(),
    }
}

/// Name a RESP frame variant without touching its payload.
///
/// Hand-rolled rather than going through `redis_protocol`'s `Resp3Frame` trait
/// (`frame.kind()`): that trait is a transitive dependency reached only through
/// fred's re-export, and naming it here would make this module the one place in
/// live-edit that imports `redis_protocol` directly. The variant list is small
/// and stable, and the catch-all keeps a future variant compiling.
fn frame_kind_name(frame: &Resp3Frame) -> &'static str {
    match frame {
        Resp3Frame::BlobString { .. } => "BlobString",
        Resp3Frame::BlobError { .. } => "BlobError",
        Resp3Frame::SimpleString { .. } => "SimpleString",
        Resp3Frame::SimpleError { .. } => "SimpleError",
        Resp3Frame::Boolean { .. } => "Boolean",
        Resp3Frame::Null => "Null",
        Resp3Frame::Number { .. } => "Number",
        Resp3Frame::Double { .. } => "Double",
        Resp3Frame::BigNumber { .. } => "BigNumber",
        Resp3Frame::VerbatimString { .. } => "VerbatimString",
        Resp3Frame::Array { .. } => "Array",
        Resp3Frame::Map { .. } => "Map",
        Resp3Frame::Set { .. } => "Set",
        Resp3Frame::Push { .. } => "Push",
        Resp3Frame::Hello { .. } => "Hello",
        Resp3Frame::ChunkedString(_) => "ChunkedString",
    }
}

/// Whether an error frame is the server's "there was no transaction to
/// discard" answer — the clean verdict.
///
/// **Two substrings, not one.** The tail `without multi` alone is what carries
/// the *meaning*, and matching on it rather than the whole string is deliberate:
/// the exact prefix and casing of `ERR DISCARD without MULTI` is not something
/// the protocol pins and a managed proxy is free to re-render it. But `DISCARD`
/// is not the only command that produces that tail — `EXEC` and `UNWATCH` do
/// too — and an offset reply stream is precisely a stream that answers our
/// `DISCARD` with the reply to somebody else's command. On the bare-tail match,
/// a `-ERR EXEC without MULTI` arriving in the `DISCARD` slot read as **clean**
/// and shipped the slot unrepaired: the exact fault class the probe exists to
/// catch, laundered by the classifier (b.7td round 2). Requiring the error to
/// name `DISCARD` closes that: a "without multi" naming any other command falls
/// through to [`TXN_DIRTY_REASON_UNEXPECTED_REPLY`], which is the correct
/// verdict for it.
fn is_discard_without_multi(detail: &str) -> bool {
    let lowered = detail.to_ascii_lowercase();
    lowered.contains("without multi") && lowered.contains("discard")
}

/// Whether a simple-string frame is the server's `+OK`.
fn is_ok_status(frame: &Resp3Frame) -> bool {
    matches!(frame, Resp3Frame::SimpleString { data, .. } if data.as_ref().eq_ignore_ascii_case(b"OK"))
}

/// Classify a probe's `Result` into a verdict, given the frame classifier for
/// the success case.
///
/// Allocation-free on every arm: the failure arms move the fred error / the
/// reply frame into a [`DirtyDetail`] that renders itself only if something
/// actually logs it (b.7td round 3).
fn classify(
    result: Result<Resp3Frame, fred::error::Error>,
    command: &'static str,
    on_frame: impl FnOnce(Resp3Frame) -> Option<CheckoutVerdict>,
) -> Option<CheckoutVerdict> {
    match result {
        Ok(frame) => on_frame(frame),
        Err(err) if residue_died_with_the_connection(&err) => {
            Some(CheckoutVerdict::MootConnectionGone(err))
        }
        Err(err) if *err.kind() == ErrorKind::Timeout => Some(CheckoutVerdict::Dirty {
            reason: TXN_DIRTY_REASON_PROBE_TIMEOUT,
            detail: DirtyDetail::ProbeTimeout { command },
        }),
        Err(err) => Some(CheckoutVerdict::Dirty {
            reason: TXN_DIRTY_REASON_PROBE_ERROR,
            detail: DirtyDetail::ProbeError {
                command,
                error: err,
            },
        }),
    }
}

/// Probe a freshly-acquired `txn_pool` slot. See the module docs for the
/// protocol and the routing rules.
///
/// `hash_key` is any key the borrower's transaction will touch; both probe
/// commands are pinned to its slot. Pass `None` only from the shim
/// [`ValkeyClient::acquire_txn`](crate::storage::client::ValkeyClient::acquire_txn),
/// where the caller's slot is not known.
///
/// `faults` is the test-only injection seam; in a release build it is a
/// zero-sized value and the branch that reads it does not exist.
///
/// `origin` counts a probe command whose timeout expired, on
/// `valkey_errors_total{pool="txn", kind="command_timeout"}`; the verdict does
/// not depend on it.
///
/// Total cost is bounded by [`TXN_CHECKOUT_PROBE_CEILING`]. The function is
/// total: it never panics and never returns without a verdict.
pub(crate) async fn probe_txn_slot(
    client: &Client,
    hash_key: Option<&str>,
    faults: &ProbeFaults,
    origin: FredErrOrigin<'_>,
) -> CheckoutVerdict {
    // Step 1 — DISCARD. The error IS the clean verdict; the `+OK` is the leak.
    let discard = issue_probe(client, PROBE_DISCARD, hash_key, faults).await;
    if let Err(e) = &discard {
        origin.observe_error(e);
    }
    if let Some(verdict) = classify(discard, "DISCARD probe", |frame| match frame {
        // A `MULTI` was open on this connection when we checked it out. The
        // previous borrower is gone, so nobody is coming back for it.
        f if is_ok_status(&f) => Some(CheckoutVerdict::Dirty {
            reason: TXN_DIRTY_REASON_LEAKED_MULTI,
            detail: DirtyDetail::LeakedMulti,
        }),
        Resp3Frame::SimpleError { ref data, .. } if is_discard_without_multi(data) => None,
        // Ordered AFTER the two verdicts above so no existing classification
        // changes: a `+OK` is still the leak and a `DISCARD without MULTI` is
        // still clean, neither of which can be a topology reply. Only the
        // former `unexpected_reply` fallthrough is narrowed (b.289).
        f if is_topology_transition_frame(&f) => Some(CheckoutVerdict::Dirty {
            reason: TXN_DIRTY_REASON_TOPOLOGY_REPLY,
            detail: DirtyDetail::TopologyReply {
                command: "DISCARD probe",
                frame: f,
            },
        }),
        other => Some(CheckoutVerdict::Dirty {
            reason: TXN_DIRTY_REASON_UNEXPECTED_REPLY,
            detail: DirtyDetail::UnexpectedReply {
                command: "DISCARD probe",
                frame: other,
            },
        }),
    }) {
        return verdict;
    }

    // Step 2 — UNWATCH. Step 1 proved no MULTI is open, so this executes rather
    // than being queued, and clears any residual WATCH before the borrower's
    // own WATCH lands.
    let unwatch = issue_probe(client, PROBE_UNWATCH, hash_key, faults).await;
    if let Err(e) = &unwatch {
        origin.observe_error(e);
    }
    classify(unwatch, "UNWATCH probe", |frame| {
        if is_ok_status(&frame) {
            None
        } else if is_topology_transition_frame(&frame) {
            // Same ordering rule as the DISCARD slot: `+OK` first, so the only
            // verdict this narrows is the `unexpected_reply` fallthrough
            // (b.289).
            Some(CheckoutVerdict::Dirty {
                reason: TXN_DIRTY_REASON_TOPOLOGY_REPLY,
                detail: DirtyDetail::TopologyReply {
                    command: "UNWATCH probe",
                    frame,
                },
            })
        } else {
            Some(CheckoutVerdict::Dirty {
                reason: TXN_DIRTY_REASON_UNEXPECTED_REPLY,
                detail: DirtyDetail::UnexpectedReply {
                    command: "UNWATCH probe",
                    frame,
                },
            })
        }
    })
    .unwrap_or(CheckoutVerdict::Clean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    // Not imported by the module itself (nothing outside `Metrics::new`
    // consumes the whole list), but the two label-set constants are one
    // contract and `topology_reply_is_a_dirty_cell_but_never_a_state_proof_one`
    // has to read both halves of it.
    use crate::observability::metrics::TXN_DIRTY_REASONS;

    /// The clean verdict has to survive the server rendering its own error
    /// text, which is why the match is on the tail rather than the whole
    /// string. Both the upstream wording and a plausible proxy re-render must
    /// read as clean; an unrelated error must not.
    #[test]
    fn discard_without_multi_is_matched_on_the_distinguishing_tail() {
        assert!(is_discard_without_multi("ERR DISCARD without MULTI"));
        assert!(is_discard_without_multi("ERR discard without multi"));
        assert!(is_discard_without_multi(
            "ERR DISCARD without MULTI (proxied)"
        ));
        assert!(!is_discard_without_multi("ERR unknown command 'DISCARD'"));
        assert!(!is_discard_without_multi("MOVED 1234 10.0.0.1:6379"));
    }

    /// The tail alone is not enough (b.7td round 2). `EXEC` and `UNWATCH`
    /// produce the same "without MULTI" tail, and an offset reply stream is by
    /// definition one that answers our `DISCARD` with the reply to somebody
    /// else's command — so accepting a bare tail classified the *exact* fault
    /// the probe exists to catch as `Clean` and shipped the slot unrepaired.
    #[test]
    fn a_without_multi_error_naming_another_command_is_not_the_clean_verdict() {
        assert!(
            !is_discard_without_multi("ERR EXEC without MULTI"),
            "an `EXEC without MULTI` arriving where the DISCARD reply belongs IS the offset \
             reply stream; reading it as clean hands the borrower an unrepaired slot"
        );
        assert!(!is_discard_without_multi("ERR UNWATCH without MULTI"));
        assert!(!is_discard_without_multi("ERR without multi"));
    }

    /// Which verdicts recycle unconditionally and which go through the streak
    /// gate. Getting this split backwards is silent in both directions: gate a
    /// state-proof verdict and a provably poisoned slot reaches a borrower;
    /// un-gate an unknown-state one and a slow backing store gets answered with
    /// a pool-wide reconnect storm.
    #[test]
    fn only_the_verdicts_that_saw_evidence_are_state_proof() {
        assert!(
            is_state_proof_reason(TXN_DIRTY_REASON_LEAKED_MULTI),
            "an acknowledged DISCARD is direct evidence of an open MULTI — recycle always"
        );
        assert!(
            is_state_proof_reason(TXN_DIRTY_REASON_UNEXPECTED_REPLY),
            "a reply of a shape no verdict allows is direct evidence the stream is offset — \
             recycle always"
        );
        assert!(
            !is_state_proof_reason(TXN_DIRTY_REASON_PROBE_TIMEOUT),
            "a timeout says only 'no answer arrived', which a store-wide latency event produces \
             on every slot at once — this is the storm the gate exists to bound"
        );
        assert!(
            !is_state_proof_reason(TXN_DIRTY_REASON_PROBE_ERROR),
            "an unclassified fred error leaves the state unknown, not proven"
        );
        assert!(
            !is_state_proof_reason(TXN_DIRTY_REASON_TOPOLOGY_REPLY),
            "a topology / node-state error frame (b.289 sub-task 3) DID arrive, but what it \
             proves is that the NODE declined the command — a reshard, a LOADING replica, a \
             demoted primary — not that this connection's reply stream is offset. Filing it \
             state-proof would let a routine, operator-initiated topology change latch a breaker \
             whose escalation ERROR accuses the classifier, and a reshard is exactly the window \
             in which the node often cannot be named, so the latch would land on the shared \
             fallback bucket where it is widest"
        );
    }

    /// The two label-set constants must agree with
    /// [`is_state_proof_reason`]'s split, in both directions.
    ///
    /// The registry-side twin of this lives in
    /// `tests/integration/observability_metrics_test.rs` (which pins the
    /// *exposition*'s cells); this one pins the consts the classifier and
    /// `Metrics::new` both read, so a cell added to one list and not the other
    /// fails here rather than in a downstream counter's label set.
    #[test]
    fn topology_reply_is_a_dirty_cell_but_never_a_state_proof_one() {
        assert!(
            TXN_DIRTY_REASONS.contains(&TXN_DIRTY_REASON_TOPOLOGY_REPLY),
            "the classifier files verdicts under this reason, so it must be in the set \
             `Metrics::new` pre-registers at zero — otherwise the series only materialises \
             mid-reshard and the sustained-rate alert built on it has no derivative to fire on"
        );
        assert!(
            !TXN_STATE_PROOF_REASONS.contains(&TXN_DIRTY_REASON_TOPOLOGY_REPLY),
            "TXN_STATE_PROOF_REASONS is the single source for BOTH `is_state_proof_reason` and \
             the repair-ineffective breaker's pre-registered cells. Listing topology_reply here \
             would simultaneously route it to the immediate-repair path and advertise a breaker \
             cell for it"
        );
        for reason in TXN_STATE_PROOF_REASONS {
            assert!(
                TXN_DIRTY_REASONS.contains(reason),
                "{reason:?} is state-proof but not a dirty cell at all — the breaker's label set \
                 must stay a strict SUBSET of the dirty counter's"
            );
        }
    }

    /// The prefix table, exhaustively, in both directions.
    ///
    /// The positives are the seven codes a node sends when it will not serve a
    /// command *right now*; each has to be recognised through the renderings a
    /// managed proxy actually produces, because fred's own interception is
    /// case-sensitive (and, for `MOVED` / `ASK`, also token-count sensitive), so
    /// every rendering that escapes fred is precisely the one this classifier
    /// must catch.
    ///
    /// The negatives matter more. `is_topology_transition_reply` moves a verdict
    /// off the state-proof path, so a false positive here silently downgrades
    /// the repair for a genuinely offset reply stream — the b.7td round-2
    /// laundering bug reintroduced one classifier over. `ERR unknown command
    /// 'READONLY'` is the worked example: a `contains(..)` match would read it
    /// as a benign topology event, and `ERR MOVED 1234 host:port` is the same
    /// hazard with the code in argument position.
    #[test]
    fn topology_prefixes_match_the_leading_token_only() {
        for canonical in [
            "MOVED 1234 10.0.0.1:6379",
            "ASK 1234 10.0.0.1:6379",
            "CLUSTERDOWN The cluster is down",
            "LOADING Redis is loading the dataset in memory",
            "TRYAGAIN Multiple keys request during rehashing of slot",
            "MASTERDOWN Link with MASTER is down and replica-serve-stale-data is set to 'no'",
            "READONLY You can't write against a read only replica",
        ] {
            assert!(
                is_topology_transition_reply(canonical),
                "{canonical:?} is a node-state transition reply and must be filed unknown-state"
            );
        }

        for rendering in [
            // Case-insensitive: fred's own interception is a case-SENSITIVE
            // `starts_with`, so a lower-cased rendering is exactly the one that
            // reaches this classifier instead of being intercepted upstream.
            "clusterdown The cluster is down",
            "moved 1234 10.0.0.1:6379",
            "Loading the dataset in memory",
            // No argument at all: not a three-token payload, so
            // `is_redirection` skips it and it arrives here.
            "MOVED",
            // Leading whitespace, which `split_ascii_whitespace` skips.
            "   MOVED 1234 10.0.0.1:6379",
            // Proxy-suffixed: four tokens, so fred's three-token
            // `is_redirection` test does not match it either.
            "MOVED 1234 host:port (proxied)",
        ] {
            assert!(
                is_topology_transition_reply(rendering),
                "{rendering:?} is a rendering fred's own interception does NOT catch, which is \
                 the whole reason this classifier is deliberately wider than fred's"
            );
        }

        for not_topology in [
            "",
            "   ",
            // One character past a prefix: a leading-token match is not a
            // prefix-of-token match.
            "MOVEDX 1 a",
            "READONLYISH nope",
            // The clean verdict and its b.7td round-2 impostor must both keep
            // their existing classifications.
            "ERR DISCARD without MULTI",
            "ERR EXEC without MULTI",
            // The laundering case: a topology code inside free text. A
            // `contains(..)` match would read this offset reply stream as a
            // benign reshard and downgrade its repair.
            "ERR unknown command 'READONLY'",
            // Same hazard, code in argument position rather than quoted.
            "ERR MOVED 1234 host:port",
            "WRONGTYPE Operation against a key holding the wrong kind of value",
            "NOAUTH Authentication required.",
            "NOSCRIPT No matching script",
        ] {
            assert!(
                !is_topology_transition_reply(not_topology),
                "{not_topology:?} must NOT read as a topology transition: this predicate moves a \
                 verdict off the state-proof path, so a false positive downgrades the repair for \
                 a reply stream that is genuinely offset"
            );
        }
    }

    /// RESP3 lets a server send either error framing for the same condition, so
    /// the topology classifier must not diverge on the encoding: a `$`-framed
    /// `-MOVED` is the same reshard as a `-`-framed one, and reading one of them
    /// as `unexpected_reply` would file a routine topology change as state-proof.
    ///
    /// The negative half is the other contract: a *non*-error frame must not be
    /// mined for a payload at all. `+MOVED …` is a status reply, not a
    /// rejection — treating it as one would launder the canonical
    /// "reply stream is offset" shape into a benign verdict.
    #[test]
    fn both_error_framings_classify_identically_and_non_errors_are_not_errors() {
        const PAYLOAD: &str = "MOVED 1234 10.0.0.1:6379";

        let simple = Resp3Frame::SimpleError {
            data: PAYLOAD.to_string().into(),
            attributes: None,
        };
        let blob = Resp3Frame::BlobError {
            data: Bytes::from_static(b"MOVED 1234 10.0.0.1:6379"),
            attributes: None,
        };

        assert_eq!(
            error_frame_detail(&simple).as_deref(),
            Some(PAYLOAD),
            "a SimpleError's payload is its `data` verbatim"
        );
        assert_eq!(
            error_frame_detail(&blob).as_deref(),
            Some(PAYLOAD),
            "a BlobError carrying the same bytes must yield the same payload — the two framings \
             are one condition and a classifier that saw only one of them would file the other \
             under `unexpected_reply`"
        );
        assert!(
            is_topology_transition_frame(&simple) && is_topology_transition_frame(&blob),
            "both framings must reach the same verdict"
        );

        for not_an_error in [
            Resp3Frame::SimpleString {
                data: Bytes::from_static(b"MOVED 1234 10.0.0.1:6379"),
                attributes: None,
            },
            Resp3Frame::Number {
                data: 7,
                attributes: None,
            },
            Resp3Frame::BlobString {
                data: Bytes::from_static(b"MOVED 1234 10.0.0.1:6379"),
                attributes: None,
            },
        ] {
            assert!(
                error_frame_detail(&not_an_error).is_none(),
                "{} is not an error frame, so it has no rejection payload to classify",
                describe_frame(&not_an_error)
            );
            assert!(
                !is_topology_transition_frame(&not_an_error),
                "a non-error frame in a probe slot IS the canonical offset-reply-stream shape; \
                 mining it for a topology prefix would launder the exact fault the probe exists \
                 to catch into a benign, streak-gated verdict. Frame: {}",
                describe_frame(&not_an_error)
            );
        }
    }

    /// The gate is the same "first of the streak, then every Nth" contract
    /// `storage::stream_mux::reader::should_force_reconnect` uses, and the first
    /// unknown-state verdict must still repair immediately — delaying it would
    /// reintroduce the wedged-slot window the probe was written for.
    #[test]
    fn unknown_state_recycles_on_the_first_of_a_streak_then_every_nth() {
        assert!(
            should_recycle_unknown_state(1),
            "the first unknown-state verdict of a streak must recycle immediately"
        );
        for streak in 2..TXN_RECYCLE_REARM_EVERY {
            assert!(
                !should_recycle_unknown_state(streak),
                "streak {streak} sits inside the first re-arm window and must be suppressed — \
                 under LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS=128 an ungated recycle is a \
                 quit+reconnect per retry attempt per slot"
            );
        }
        for multiple in 0..4u32 {
            let rearm = 1 + multiple * TXN_RECYCLE_REARM_EVERY;
            assert!(
                should_recycle_unknown_state(rearm),
                "streak {rearm} is a re-arm point, so a genuinely wedged slot that was suppressed \
                 still heals"
            );
            assert!(
                !should_recycle_unknown_state(rearm + 1),
                "streak {} sits one past a re-arm point and must be suppressed",
                rearm + 1
            );
        }
        assert!(
            !should_recycle_unknown_state(0),
            "a zero-length streak means no unknown-state verdict was observed"
        );
    }

    /// The fault seam must be inert until an injector is installed, and must
    /// return each queued outcome exactly once, per command. A seam that leaked
    /// a `DISCARD` fault into the `UNWATCH` probe could not drive the two
    /// distinct `unexpected_reply` arms apart.
    #[test]
    fn probe_faults_are_keyed_per_command_and_consumed_once() {
        let faults = ProbeFaults::default();
        assert!(
            faults.take(PROBE_DISCARD).is_none(),
            "with no injector installed every probe must take the real path"
        );

        let injector = std::sync::Arc::new(ProbeFaultInjector::new());
        injector.push_discard(ProbeFault::ok());
        injector.push_unwatch(ProbeFault::number(7));
        faults.install(std::sync::Arc::clone(&injector));

        assert!(
            matches!(
                faults.take(PROBE_DISCARD),
                Some(ProbeFault::Frame(Resp3Frame::SimpleString { .. }))
            ),
            "the DISCARD queue must answer the DISCARD probe"
        );
        assert!(
            faults.take(PROBE_DISCARD).is_none(),
            "a queued fault is consumed once; the next checkout falls through to the real probe"
        );
        assert!(
            matches!(
                faults.take(PROBE_UNWATCH),
                Some(ProbeFault::Frame(Resp3Frame::Number { .. }))
            ),
            "the UNWATCH queue is independent — draining DISCARD must not drain it"
        );
        assert_eq!(injector.remaining(), 0);
    }

    /// A delay fault must produce its `Timeout` through the *real*
    /// `tokio::time::timeout`, and a sub-deadline delay must resolve to the
    /// command's own clean reply — otherwise the seam would be asserting the
    /// timeout classification into existence rather than exercising it.
    #[tokio::test(start_paused = true)]
    async fn a_delay_fault_times_out_through_the_production_deadline() {
        let hung = ProbeFault::hang()
            .resolve(PROBE_DISCARD)
            .await
            .expect_err("a delay past the probe deadline must fail");
        assert_eq!(*hung.kind(), ErrorKind::Timeout);

        let slow = ProbeFault::Delay(TXN_CHECKOUT_PROBE_TIMEOUT / 2)
            .resolve(PROBE_DISCARD)
            .await
            .expect("a delay inside the deadline still answers");
        assert!(
            matches!(slow, Resp3Frame::SimpleError { ref data, .. } if is_discard_without_multi(data)),
            "a slow-but-answering DISCARD is the clean verdict, not a second failure shape"
        );

        let slow_unwatch = ProbeFault::Delay(TXN_CHECKOUT_PROBE_TIMEOUT / 2)
            .resolve(PROBE_UNWATCH)
            .await
            .expect("a delay inside the deadline still answers");
        assert!(is_ok_status(&slow_unwatch));
    }

    /// `+OK` from `DISCARD` is the leak verdict, so recognising it must not
    /// depend on the server's casing.
    #[test]
    fn ok_status_recognised_regardless_of_case() {
        let ok = Resp3Frame::SimpleString {
            data: Bytes::from_static(b"OK"),
            attributes: None,
        };
        let lower = Resp3Frame::SimpleString {
            data: Bytes::from_static(b"ok"),
            attributes: None,
        };
        let queued = Resp3Frame::SimpleString {
            data: Bytes::from_static(b"QUEUED"),
            attributes: None,
        };
        assert!(is_ok_status(&ok));
        assert!(is_ok_status(&lower));
        assert!(!is_ok_status(&queued));
    }

    /// The WARN detail must never carry payload bytes — a `txn` connection can
    /// be mid-anything when it is found dirty. Only the frame kind escapes for
    /// non-textual replies.
    #[test]
    fn describe_frame_prints_shape_not_payload() {
        let blob = Resp3Frame::BlobString {
            data: Bytes::from_static(b"le:42:obj:hdr secret-value"),
            attributes: None,
        };
        let rendered = describe_frame(&blob);
        assert!(
            !rendered.contains("secret-value") && !rendered.contains("le:42"),
            "describe_frame leaked payload bytes: {rendered}"
        );
    }

    /// A pinned probe must route by the borrower's slot, and the unpinned shim
    /// must leave fred's default policy alone rather than inventing a slot.
    #[test]
    fn probe_commands_pin_the_callers_slot() {
        let pinned = probe_command("DISCARD", Some("le:{42}:obj:hdr"));
        assert_eq!(pinned.cluster_hash, ClusterHash::from("le:{42}:obj:hdr"));
        assert!(matches!(pinned.cluster_hash, ClusterHash::Custom(_)));

        let shim = probe_command("DISCARD", None);
        assert_eq!(shim.cluster_hash, ClusterHash::FirstKey);
    }

    /// Two keys sharing a hash tag must produce the same probe slot — the
    /// property every migrated call site relies on when it passes just one of
    /// the keys its transaction touches.
    #[test]
    fn hash_tagged_siblings_share_a_probe_slot() {
        let header = probe_command("DISCARD", Some("le:{7}:obj:hdr"));
        let stream = probe_command("DISCARD", Some("le:{7}:obj:stream"));
        assert_eq!(header.cluster_hash, stream.cluster_hash);
    }

    /// A probe that fails must be sorted by *what the failure says about the
    /// connection*, not merely counted.
    ///
    /// The three-way split is the whole reason [`CheckoutVerdict`] has three
    /// arms. `Canceled` / `IO`, and fred's routing miss (`Unknown` or `Routing`
    /// "Failed to route command.", the router holding no connection at all),
    /// mean the connection is already gone, so no residue survived it and
    /// nothing is counted — this is the common outcome on a pool that sees
    /// failover, managed-proxy recycles and ordinary reconnects, and, for the
    /// routing miss, a borrower whose probe overtakes the deferred reconnect
    /// after a fred `Transaction` cut partway closed its connection (the
    /// vendored fred's change 9). Counting it would page an operator on routine
    /// churn. `Timeout` is the opposite: an argument-free `DISCARD` / `UNWATCH`
    /// that never answered leaves the connection's state *unknown*, which is
    /// exactly the case worth recycling. Everything else — an `Unknown` with any
    /// other text included — is an unknown-state failure too, filed separately
    /// so an operator can tell a wedge from a rejection.
    ///
    /// Driven through [`classify`] rather than [`probe_txn_slot`] because these
    /// are transport-level failures with no server-side trigger: there is no
    /// way to make a healthy local Valkey produce a `Canceled` on demand, and a
    /// verdict that mis-sorts them is silent by construction.
    ///
    /// The one premise this leaves implicit — that an expired per-command
    /// `Options::timeout` really does arrive here as [`ErrorKind::Timeout`] —
    /// holds in fred 10.1.0 by construction: `utils::basic_request_response`
    /// (`src/utils.rs:391-421`) wraps the response oneshot in `utils::timeout`
    /// (`src/utils.rs:286-298`), whose only elapsed arm is
    /// `Error::new(ErrorKind::Timeout, "Request timed out.")`.
    #[test]
    fn probe_failures_are_sorted_by_what_they_say_about_the_connection() {
        fn verdict_for_error(error: fred::error::Error) -> CheckoutVerdict {
            classify(Err(error), "DISCARD probe", |_| {
                panic!("the frame classifier must not run on an Err result")
            })
            .expect("an Err result always yields a verdict")
        }
        fn verdict_for(kind: ErrorKind) -> CheckoutVerdict {
            verdict_for_error(fred::error::Error::new(kind, "probe fixture"))
        }

        for gone in [ErrorKind::Canceled, ErrorKind::IO] {
            // Named before the move: `ErrorKind` is not `Copy`.
            let kind = format!("{gone:?}");
            assert!(
                matches!(verdict_for(gone), CheckoutVerdict::MootConnectionGone(_)),
                "{kind} means the connection is already gone: a MULTI and a WATCH both die \
                 with their connection, so there is no residue to clear and nothing to count. \
                 Counting it would fire the dirty-checkout alert on routine failover and \
                 managed-proxy recycles"
            );
        }

        // fred's routing miss, in both kinds `is_router_no_connection_error`
        // accepts: the router holds no connection, so no MULTI or WATCH can be
        // left for the probe to find.
        for (kind, label) in [
            (ErrorKind::Unknown, "Unknown"),
            (ErrorKind::Routing, "Routing"),
        ] {
            let miss = fred::error::Error::new(kind, "Failed to route command.");
            assert!(
                crate::storage::client::is_router_no_connection_error(&miss),
                "precondition: the fixture is the routing miss the shared classifier names \
                 ({label})"
            );
            assert!(
                matches!(
                    verdict_for_error(miss),
                    CheckoutVerdict::MootConnectionGone(_)
                ),
                "a {label} \"Failed to route command.\" means the router holds no connection at \
                 all, so no transaction state can be left: moot, uncounted, no recycle — never a \
                 probe_error that feeds the slot's recycle gate"
            );
        }

        // The boundary: an `Unknown` that is not the routing miss keeps its
        // unknown-state verdict.
        assert!(
            matches!(
                verdict_for_error(fred::error::Error::new(
                    ErrorKind::Unknown,
                    "Unexpected DISCARD response."
                )),
                CheckoutVerdict::Dirty {
                    reason: TXN_DIRTY_REASON_PROBE_ERROR,
                    ..
                }
            ),
            "only the routing miss's text makes an Unknown moot; any other Unknown leaves the \
             slot's state unknown and stays a counted probe_error"
        );

        assert!(
            matches!(
                verdict_for(ErrorKind::Timeout),
                CheckoutVerdict::Dirty {
                    reason: TXN_DIRTY_REASON_PROBE_TIMEOUT,
                    ..
                }
            ),
            "a Timeout is NOT 'the connection died' — the command expired with no reply, so the \
             connection's state is unknown, which is precisely the state worth recycling and \
             counting. Mirrors `watch_guard::watch_died_with_the_connection`, which excludes \
             Timeout for the same reason"
        );

        assert!(
            matches!(
                verdict_for(ErrorKind::Auth),
                CheckoutVerdict::Dirty {
                    reason: TXN_DIRTY_REASON_PROBE_ERROR,
                    ..
                }
            ),
            "any other fred error leaves the slot's state unknown, which is treated as dirty — \
             and filed under its own reason so an operator can tell it apart from a wedge"
        );
    }

    /// The per-attempt bound and the advertised ceiling must stay in step, and
    /// the bound must stay well under fred's default command timeout — the
    /// whole point is to fail faster than the knob a wedged connection would
    /// otherwise stall for.
    #[test]
    fn probe_ceiling_is_two_attempts() {
        assert_eq!(TXN_CHECKOUT_PROBE_CEILING, TXN_CHECKOUT_PROBE_TIMEOUT * 2);
        assert!(TXN_CHECKOUT_PROBE_CEILING < Duration::from_secs(5));
    }
}
