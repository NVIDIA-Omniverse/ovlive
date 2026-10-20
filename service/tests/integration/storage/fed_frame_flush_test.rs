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

//! A write fred fed but did not flush never waits through an inline connect,
//! nor through the router's wait on another connection's I/O.
//!
//! # The contract (SDD §15.3, *Vendored fred* change 5)
//!
//! fred writes a command without flushing it when its command-queue counter
//! says another command follows, and leaves the flush to that next command.
//! Until a flush sends it, the frame waits on whatever holds the router, and
//! on a cluster client that can take tens of seconds: an inline connect, or a
//! wait of up to one liveness bound on each other connection's write or
//! flush. By then the write's caller may long have given up while the write
//! still lands. On a clustered client, every connection holding fed frames is
//! flushed first, all of them at once, at each place the router could wait on
//! something else:
//!
//! - **before an inline connect** (b.gbt). Once running, a cluster client's
//!   router connects inline, holding the router, in three places: a
//!   reconnect attempt (`reconnect_with_policy`), a cluster-sync attempt
//!   (`sync_cluster_with_policy`) and the on-demand connect to an `-ASK`
//!   target (`live_edit::ask::follow`). All three call
//!   `live_edit::before_inline_connect` first. The router command that leads
//!   to each — a `Reconnect`, a `SyncCluster`, an `Ask` — is itself queued,
//!   so the data command routed just ahead of it is only fed, and while the
//!   connect runs the router polls no connection.
//! - **before a write that can reach a socket** (b.bo2). At the top of the
//!   router's single-connection write, `live_edit::before_socket_write`
//!   routes the command and asks whether its write can wait on its own
//!   socket: fred's own flush rule sends it (the command-queue counter at
//!   zero, the destination's feed count past `max_feed_count`, or a command
//!   kind fred always flushes, every custom command among them), or it is a
//!   feed onto a write buffer at tokio-util's 8 KiB backpressure boundary,
//!   which flushes that buffer first. If so, and another connection holds
//!   fed frames, every connection holding them is flushed before the write.
//!   `ask::follow`'s `ASKING` and command (`live_edit::before_write_to`)
//!   flush first whenever another connection holds fed frames.
//! - **before the router waits** (`live_edit::before_wait`, when its command
//!   channel is empty), and in fred's router-wide flush (`Connections::flush`,
//!   through `live_edit::flush_router_wide`).
//!
//! Every one of those flushes runs through
//! `live_edit::liveness::flush_fed_concurrently`: each connection's bounded
//! flush (change 3) at once, so the frame reaches its node's socket within
//! one bound, or its connection is failed, however many of the client's
//! connections stall, and the flush holds the router for at most one bound
//! for all of them.
//!
//! # The cluster facade
//!
//! Four [`ScriptReplyProxy`]s, nodes A to D, in front of the one standalone
//! test store answer fred's cluster discovery (`CLUSTER SLOTS`: each node
//! owns [`SLOTS_PER_NODE`] slots, A the lowest; `CLUSTER INFO`:
//! `cluster_state:ok`) and relay everything else to the store, so a client
//! connected with a `redis-cluster://` URL runs fred's clustered router
//! against the real store. A fifth node, [`SilentNode`], accepts connections
//! and never answers. The facade:
//!
//! - holds the reply to a `CLUSTER SLOTS` sent on a connection accepted after
//!   [`Facade::hold_new_slots_queries`], behind a [`ReplyGate`] the test opens.
//!   A reconnect or sync attempt's `cluster_slots_backchannel`
//!   (`router/clustered.rs:405-498`) sends that query on the new connection
//!   `connect_any` (`:359-397`) opened for it (`:471-476`), and waits on the
//!   reply, under fred's `internal_command_timeout`, as long as the test holds
//!   it;
//! - stalls ([`ProxyAction::Stall`]) or holds the replies of
//!   ([`ProxyAction::HoldReplies`]) one marked `GET` on one node, or holds
//!   A's `-ASK` reply to one marked command behind a gate
//!   ([`ProxyAction::GatedEval`]), naming the silent node or another proxied
//!   node as the target;
//! - after answering a marker `ECHO`, reads nothing more from that connection
//!   ([`ProxyAction::StopReading`]), so what the client writes on it next
//!   backs up in the socket buffers, and a write larger than those buffers
//!   ([`UNSENDABLE_VALUE_LEN`]) can only expire at the liveness bound;
//! - records, when the probe write's frame reaches A, when that was, whether
//!   the test had released the held `CLUSTER SLOTS` yet and whether the client
//!   had already closed its connection to the silent node ([`Arrival`]). The
//!   last is read from the kernel (a non-blocking read on the silent node's
//!   end of each connection), so it orders the two events without a clock;
//! - records when a marker command's frame reaches any node
//!   ([`Facade::watch_marker`]).
//!
//! Every fault aims at the subject (test-writing guide § Valkey Integration
//! Tests): the marked commands, the probe and the markers name keys or
//! strings only the test uses, the `CLIENT KILL` names the subject's own
//! connection id, and the held `CLUSTER SLOTS` is armed only for connections
//! accepted after arming.
//!
//! # The scenarios
//!
//! The subject is a `TestValkey` connected through A with one data member. Its
//! router is made busy first: a fred `Transaction` starts with `drain_all`
//! (`router/transactions.rs:142`), which waits on every connection's in-flight
//! replies, and one of them is held by the facade. The test then sends, in a
//! set order by polling each future once (fred sends a command to its router
//! on the first poll), the probe — a `SET` of a key on A with a
//! [`PROBE_TIMEOUT`] — and the commands each scenario needs behind it; the
//! command-queue counter (`cmd_buffer_len`) self-checks that they are
//! queued. Then it releases the busy step, so the router writes the probe only
//! fed: a command still follows it.
//!
//! ## Before an inline connect (b.gbt)
//!
//! The busy step is released in a way that queues the connect's router
//! command behind the probe:
//!
//! | Test | Item | Busy step on B or A, and how the connect is queued | What it pins |
//! |---|---|---|---|
//! | [`a_write_fed_behind_a_reconnect_reaches_its_node_before_the_reconnect_connects`] | (R) | a `GET` on B stalled; `CLIENT KILL` of the subject's B connection ends the drain, and the router reads B's end of file after it, queueing a `Reconnect` behind the probe | the probe reaches A while the reconnect's `CLUSTER SLOTS` is still held; once released, the subject answers on A and B |
//! | [`a_write_fed_behind_a_cluster_sync_reaches_its_node_before_the_sync_connects`] | (S) | a `GET` on B whose reply is held; a `SyncCluster` queued behind the probe; the held reply released | the probe reaches A while the sync's `CLUSTER SLOTS` is still held; once released, the sync succeeds |
//! | [`a_write_fed_behind_an_ask_redirect_reaches_its_node_before_the_ask_target_connect_ends`] | (K) | A's `-ASK` reply to a `GET` held; released, the drain queues an `Ask` (target: the silent node) behind the probe | the probe reaches A before the client closes its connect to the silent node, cut at `ASK_TARGET_CONNECT_CAP` |
//! | [`fed_frame_flush_disruptive_a_write_fed_behind_a_reconnect_to_a_paused_node_lands_before_the_pause_ends`] | T2 | a real three-primary cluster (`LIVE_EDIT_TEST_CLUSTER_URL`): `BLPOP` on C; `CLIENT KILL` of the subject's B connection and `CLIENT PAUSE` on B; `LPUSH` releases the drain | the probe is readable on A while a witness command on B is still held by the pause, and within half the pause |
//! | [`an_inline_connect_flushes_two_silent_connections_holding_fed_frames_in_one_bound`] | F3 | A and B stop reading; a first `SyncCluster` held at its `CLUSTER SLOTS`; a large one-attempt `SET` to each of A and B and a second `SyncCluster` queued behind it, then the hold released | with a 4 s liveness bound, both `SET`s fail with the slow verdict (both flushes expired), and the second sync's `CLUSTER SLOTS` arrives at least one bound and less than one bound plus 2 s after the release: the two expired flushes cost one bound together, not two |
//!
//! Each of (R), (S) and (K) fails with only its own hook removed — the
//! `before_inline_connect` call at the top of `reconnect_with_policy`'s loop,
//! the one at the top of `sync_cluster_with_policy`'s, and the one in
//! `ask::follow` before its connect: the probe is then flushed only once the
//! connect ends (by the retry buffer's flush, `router/mod.rs:396`, or for (K)
//! by `before_wait`, since a connect cut at the cap ends the `Ask` before its
//! writes), so it reaches A after the held `CLUSTER SLOTS` is released, or
//! after the client closed its connection to the silent node. F3 drives the
//! sync site, and so pins the one flush function every site shares: flushed
//! one connection after another, it fails (as do b.bo2 T1, T3 and T4 below).
//!
//! T2 is `#[ignore]`d (CI's `test-cluster-disruptive:` job runs it) because
//! `CLIENT PAUSE` suspends a whole node, and it prints `SKIP:` when
//! `LIVE_EDIT_TEST_CLUSTER_URL` is unset and `CLUSTER-GATED RAN:` when it runs.
//! The pause expires on its own, so a retry finds the cluster healthy.
//!
//! ## Before a write, and before the router waits (b.bo2)
//!
//! The subject runs with F3's 4 s liveness bound ([`F3_SLOW_BOUND`]), and
//! some of its connections stop reading. The busy step is a `GET` on A whose
//! replies the facade holds, under a `Transaction` on A (for b.bo2 T5, A's
//! held `-ASK` reply, under a `Transaction` on C). Behind the probe the test
//! queues the commands that make the router wait on a silent connection,
//! each with one write attempt, so a write whose bound expired fails its
//! caller with the slow verdict. T1 times the router's hold by when its last
//! `SET` fails; T4, whose writes fail only at a later read pass, sends a
//! marker instead, a custom `ECHO` routed to A, one second after the release
//! ([`MARKER_AFTER`]): fred always sends a custom command at once, so the
//! marker reaches A as soon as the router is free again, and its arrival is
//! when the hold ended.
//!
//! | Test | Item | The wait on a silent connection, and the one term that makes the write that waits reach its socket | What it pins |
//! |---|---|---|---|
//! | [`a_write_fed_ahead_of_a_send_to_a_silent_node_reaches_its_node_before_that_send_waits`] | b.bo2 T1 | B, C and D stop reading; a 16 MiB typed `SET` to each: B's and C's only fed, D's the last command, so sent (the counter at zero) | the probe reaches A within half a bound of the release; all three `SET`s fail with the slow verdict; D's fails at least two bounds and less than two bounds plus [`F3_SLACK`] after the release: B's and C's flushes expired together, in one bound, then D's send in a second |
//! | [`a_write_fed_ahead_of_a_custom_command_to_a_silent_node_is_sent_first_and_its_connection_is_kept`] | b.bo2 T2 | B stops reading; a 16 MiB custom `SET` to B, a command queued behind it (the command-kind term) | the probe reaches A within half a bound; the `SET` fails with the slow verdict; for two bounds after the release the subject's error stream names no error on A; its connection to A is the one it had; the probe landed |
//! | [`a_write_fed_ahead_of_a_feed_at_the_backpressure_boundary_is_sent_first_and_its_connection_is_kept`] | b.bo2 T3 | B stops reading; a 16 MiB typed `SET` and then a small one to B, a command queued behind (the boundary term, for the small `SET`) | as T2; both `SET`s fail with the slow verdict |
//! | [`a_write_fed_ahead_of_a_send_past_the_feed_count_is_sent_first_and_its_connection_is_kept`] | b.bo2 T6 | B stops reading; a pipeline of 201 small typed `GET`s fed to B, then a 16 MiB typed `SET` to B, a command queued behind (the feed-count term, fred's `max_feed_count` being 200: the `SET` would be sent; its check flushes first, which resets B's count, so it is only fed and B's bound expires in the flush before the last command) | as T2; the `SET` and every pipelined `GET` fail with the slow verdict |
//! | [`the_flush_before_the_router_waits_flushes_two_silent_connections_in_one_bound`] | b.bo2 T4 | B and C stop reading; a 16 MiB typed `SET` to each, then a `Transaction` whose caller has already given up, which the router refuses without I/O: every write is only fed (the counter above zero at each), so the check before a write never flushes | the probe reaches A within half a bound; both `SET`s fail with the slow verdict; the marker arrives at least one bound and less than one bound plus [`F3_SLACK`] after the release: the flush before the router waits ran the two expiring flushes at once |
//! | [`a_write_fed_ahead_of_an_ask_redirect_to_a_silent_node_reaches_its_node_before_the_redirected_write_waits`] | b.bo2 T5 | B stops reading; A answers a 16 MiB typed `SET` (two attempts) `-ASK` naming B, a node the client already holds a connection to | the probe reaches A within half a bound; the `SET` fails with the slow verdict (its write to B expired) |
//!
//! Each fails with only its own piece reverted:
//!
//! - T1 with the `live_edit::before_socket_write` call in `write_command`
//!   removed, or only its counter-at-zero term (`queue_empty`) made false:
//!   nothing is flushed before D's send, so the probe waits out D's bound,
//!   fed on A. (Flushed one connection after another, B's and C's expiries
//!   would also fail D's `SET` only three bounds after the release.)
//! - T2, T3 and T6 with that call removed, or only their own term made false
//!   (`flushes_by_kind`, `at_backpressure`, `feed_count_full`): the probe
//!   waits out B's bound, fed on A. A fed frame's wait counts from when it was
//!   fed, so the read pass after B's expiry can judge A silent and replace
//!   it, and the probe, whose caller has given up, is not sent again.
//! - T4 with `before_wait`'s flush reverted to one connection after another:
//!   B's and C's expiries add up, so the marker arrives two bounds after the
//!   release, and the probe waits too when the walk reaches B or C first.
//! - T5 with the `live_edit::before_write_to` call in `ask::follow` removed:
//!   the probe waits out the redirected write's bound, fed on A.
//!
//! In T1, T2, T3 and T6 the named term is the only one under which the check
//! before the named write finds that write able to reach the socket (in T6,
//! the `SET` fred would send past the feed count, which the check's flush
//! leaves only fed), so no other test of the table fails with that term alone
//! made false.
//!
//! # Not covered
//!
//! - The replica loops (`add_replica_with_policy`, `sync_replicas_with_policy`)
//!   and replica routing are compiled only with fred's `replicas` feature,
//!   which is not built.
//! - A `-MOVED`'s `delay_cluster_sync` sleep: `cluster_cache_update_delay` is
//!   fred's default 0, which the service leaves unset.
//! - The documented exceptions, which still wait on one connection while
//!   another may hold fed frames: the flush before a `CLIENT UNBLOCK` under
//!   `Blocking::Interrupt` (`write_command`, and its copy in `ask::follow`),
//!   and fred's stock close, bounded at 5 s, in `drop_connection` (after a
//!   read error or end of file) and `disconnect_all`.
//! - That every flush runs each connection's flush to its own end
//!   (`join_all`, not stock's `try_join_all`), so a fast error on one never
//!   cuts another's short — in `Connections::flush` (where a `drain_all`
//!   now waits for the others before returning the error) as in the flushes
//!   the patch adds. No construction through the public client puts a
//!   connection whose flush fails at once ahead of one still pending at the
//!   same flush: in F3, T1 and T4 the silent connections' flushes expire
//!   together, and in every other test one connection is silent.
//! - The flush before `abandon::after_pipeline`'s `DISCARD`
//!   (`live_edit::before_write_to`): reaching it needs a `WatchedTxn`
//!   pipeline whose caller gives up after its `MULTI` was fed, with a frame
//!   fed to another node meanwhile, inside one router step.
//! - That `ask::follow`'s inline connect flushes every connection at once: F3
//!   drives the sync site only, and (K) passes with any flush before the
//!   connect.
//! - The counter-at-zero term's share of `write_command`'s flush of the
//!   destination after a command it skipped (its caller gave up): that flush
//!   can wait only on a destination whose write buffer holds a frame its
//!   socket cannot take, and every such buffer here is past the 8 KiB
//!   boundary, so the boundary term flushes first too. T1 pins the term on a
//!   send instead.
//! - The cluster-only gates in `live_edit::before_inline_connect`,
//!   `before_socket_write` and `before_write_to`: no test notices if a
//!   centralized or sentinel client also runs them. Its one connection has
//!   no other connection's frames to wait behind; the inline-connect gate only
//!   keeps a close-first reconnect's wait at fred's 5 s close timeout instead
//!   of the slower liveness bound.
//!
//! # Wait budget
//!
//! Every wait in a test runs under that test's one deadline,
//! [`SCENARIO_BUDGET`] (60 s) from its first line, connects included; the
//! hold window ([`HOLD_WINDOW`], 5 s) is one of those waits. After the
//! deadline comes only the fixtures' drop: each purge round trip bounded at
//! 5 s and each client's quit at 10 s. Inside nextest's 120 s hang-killer. A
//! healthy run takes well under a second for (R) and (S); waits out one
//! [`F3_SLOW_BOUND`] for F3, T4 and T5, two for T1 and, through the window
//! in which A must stay unjudged, for T2, T3 and T6; and waits out the 2 s
//! connect cap once for (K) and the pause (at most [`PAUSE_MS`]) once for
//! T2, both by construction.

use std::future::Future;
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use fred::clients::{Client, Pipeline};
use fred::interfaces::{
    ClientInterface, ClientLike, ClusterInterface, EventInterface, KeysInterface, ListInterface,
    TransactionInterface,
};
use fred::live_edit::liveness::{verdict_of, Verdict};
use fred::types::config::{Options, Server};
use fred::types::{ClusterHash, CustomCommand, Value};
use futures::future::BoxFuture;
use futures::FutureExt;
use tokio::time::Instant;

use live_edit::storage::client::{
    is_transient_fred_error, with_liveness_overrides_for_test, LivenessOverrides,
};
use live_edit::storage::test_support::TestValkey;

use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ConnProxyRule, ProxyAction, ReplyGate, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and constants
// ═════════════════════════════════════════════════════════════════════════════

/// Each test's one deadline; see the module's *Wait budget*.
const SCENARIO_BUDGET: Duration = Duration::from_secs(60);

/// How long a held `CLUSTER SLOTS` stays held while the test waits for the
/// probe to reach A. A deadline, not a speed claim: with the flush the probe
/// arrives within milliseconds, before the attempt even reaches its
/// `CLUSTER SLOTS`; without it the probe stays in fred's write buffer however
/// long the hold lasts. Under fred's 10 s `internal_command_timeout`, so the
/// held query is still the attempt's first when the test releases it.
const HOLD_WINDOW: Duration = Duration::from_secs(5);

/// The probe `SET`'s own command timeout: once the router is held past it,
/// the caller has given up while the write still waits, the case the flush
/// exists for. It orders nothing (the gates do); it only has to outlast the
/// router's write of the probe, which follows the release of the busy step by
/// milliseconds. A runner too slow for that skips the write (fred writes no
/// command whose caller gave up) and fails the arrival precondition; it never
/// passes falsely.
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Interval between probes of a polled condition.
const POLL: Duration = Duration::from_millis(2);

/// The facade's proxied nodes, A to D.
const NODES: usize = 4;

/// The slots each proxied node owns: node `i` owns
/// `[i * SLOTS_PER_NODE, (i + 1) * SLOTS_PER_NODE)`.
const SLOTS_PER_NODE: u16 = 16384 / NODES as u16;

/// The node ids the facade's `CLUSTER SLOTS` reply names, in [`Node::ALL`]'s
/// order.
const NODE_IDS: [&str; NODES] = [
    "b0gbt0000000000000000000000000000000aaa0",
    "b0gbt0000000000000000000000000000000bbb0",
    "b0gbt0000000000000000000000000000000ccc0",
    "b0gbt0000000000000000000000000000000ddd0",
];

/// The value the probe `SET` writes.
const PROBE_VALUE: &str = "gbt-fed-probe";

/// The liveness bound F3 and the b.bo2 tests run under
/// (`UnresponsiveConfig::max_timeout`, the bound on every router wait on one
/// connection's I/O): small, so one bound against two is seconds apart, and
/// the same 4 s the expired-flush tests in `stream_mux_reader_connect_test.rs`
/// use.
const F3_SLOW_BOUND: Duration = Duration::from_secs(4);

/// The slack over a whole number of bounds that a timed hold may take, as
/// those tests allow: the router's own steps between the release and the
/// timed event take milliseconds; one more bound would add another
/// [`F3_SLOW_BOUND`], twice the slack.
const F3_SLACK: Duration = Duration::from_secs(2);

/// How soon after the release the probe must reach A in the b.bo2 tests:
/// half a bound. Flushed before the wait, it arrives within milliseconds;
/// left fed, it waits out at least one bound.
const PROBE_DEADLINE: Duration = Duration::from_secs(2);

/// When b.bo2 T4 sends its marker, after the release: well past the router's
/// taking every queued command (milliseconds), so the marker never changes
/// how they are written, and well inside the bound of the hold it times.
const MARKER_AFTER: Duration = Duration::from_secs(1);

/// A value larger than every socket buffer between the client and a proxy that
/// stopped reading (`tcp_wmem` and the receive window together hold a few MiB
/// on loopback), so a flush of it cannot complete.
const UNSENDABLE_VALUE_LEN: usize = 16 * 1024 * 1024;

/// b.bo2 T6's fed `GET`s: one more than fred's `max_feed_count` (200, fred's
/// default, which the service leaves unset), so fred would send the next
/// write to the node (`Counters::should_send`) unless a flush first resets
/// the node's feed count.
const FEEDS_PAST_THE_COUNT: usize = 201;

/// T2's `CLIENT PAUSE` on B, in milliseconds: long enough that a probe still
/// fed through the reconnect would visibly wait for it, short enough to stay
/// inside the 5 s command timeout of every client on the cluster (the witness
/// `PING` waits it out).
const PAUSE_MS: u64 = 4_000;

// ═════════════════════════════════════════════════════════════════════════════
// Deadlines
// ═════════════════════════════════════════════════════════════════════════════

/// `fut` under the test's one `deadline`, failing naming `what` past it.
async fn by<T>(deadline: Instant, what: &str, fut: impl Future<Output = T>) -> T {
    tokio::time::timeout_at(deadline, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not finish within the test's {SCENARIO_BUDGET:?}"))
}

/// Poll `holds` every [`POLL`] until it is true or `until` passes; whether it
/// became true.
async fn eventually(until: Instant, mut holds: impl FnMut() -> bool) -> bool {
    loop {
        if holds() {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        tokio::time::sleep(POLL).await;
    }
}

/// [`eventually`], failing naming `what` when `holds` is still false at the
/// test's `deadline`.
async fn until(deadline: Instant, what: &str, holds: impl FnMut() -> bool) {
    assert!(
        eventually(deadline, holds).await,
        "{what} did not happen within the test's {SCENARIO_BUDGET:?}"
    );
}

/// Poll `fut` once, from the test's own task, and require it to be pending: a
/// fred command sends itself to the router on its first poll, so the order of
/// these polls is the order the router's channel holds the commands in.
async fn send_now<T>(fut: &mut BoxFuture<'static, T>, what: &str) {
    assert!(
        futures::poll!(fut).is_pending(),
        "{what} must still wait on the busy router after its first poll"
    );
}

/// fred's command-queue counter: commands sent to the router and not yet taken
/// off its channel.
fn queued(client: &Client) -> usize {
    client.inner().counters.cmd_buffer_len.load(SeqCst)
}

// ═════════════════════════════════════════════════════════════════════════════
// The facade
// ═════════════════════════════════════════════════════════════════════════════

/// A listener that accepts every connection and never writes a byte on it: a
/// connect to it succeeds, and the connection's setup exchange never
/// completes, so only a timeout ends it. It keeps each accepted connection
/// (non-blocking), so [`any_closed`] can read whether the client
/// has closed one.
struct SilentNode {
    port: u16,
    accepted: Arc<AtomicUsize>,
    streams: Arc<Mutex<Vec<std::net::TcpStream>>>,
    accept: tokio::task::JoinHandle<()>,
}

impl SilentNode {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the silent listener must bind");
        let port = listener
            .local_addr()
            .expect("the silent listener has an address")
            .port();
        let accepted = Arc::new(AtomicUsize::new(0));
        let streams: Arc<Mutex<Vec<std::net::TcpStream>>> = Arc::default();
        let (counted, kept) = (Arc::clone(&accepted), Arc::clone(&streams));
        let accept = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                // `into_std` keeps the socket non-blocking.
                let stream = stream
                    .into_std()
                    .expect("an accepted stream converts to a std stream");
                kept.lock().expect("never poisoned").push(stream);
                counted.fetch_add(1, SeqCst);
            }
        });
        Self {
            port,
            accepted,
            streams,
            accept,
        }
    }

    fn accepted(&self) -> usize {
        self.accepted.load(SeqCst)
    }
}

impl Drop for SilentNode {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

/// Whether the client has closed any connection the silent node accepted:
/// read each one without blocking, discarding what the client sent (its setup
/// `PING`), until it would block (open) or reports end of file (closed).
fn any_closed(streams: &Mutex<Vec<std::net::TcpStream>>) -> bool {
    let streams = streams.lock().expect("never poisoned");
    streams.iter().any(|stream| {
        let mut reader: &std::net::TcpStream = stream;
        let mut buf = [0u8; 512];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => return true,
                Ok(_) => continue,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return false,
                Err(_) => return true,
            }
        }
    })
}

/// A proxied node of the facade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Node {
    A,
    B,
    C,
    D,
}

impl Node {
    const ALL: [Node; NODES] = [Node::A, Node::B, Node::C, Node::D];

    fn index(self) -> usize {
        self as usize
    }

    /// The node whose range holds `slot`.
    fn owning(slot: u16) -> Node {
        Node::ALL[usize::from(slot / SLOTS_PER_NODE)]
    }
}

/// What A recorded when the probe's frame reached it.
#[derive(Clone, Copy, Debug)]
struct Arrival {
    /// When.
    at: Instant,
    /// The test had already released the held `CLUSTER SLOTS`.
    slots_released: bool,
    /// The client had already closed a connection to the silent node.
    silent_closed: bool,
}

/// The `CLUSTER SLOTS` held for new connections: those the proxy numbered past
/// every connection that had sent a frame when the hold was armed. Every
/// connection a fixture opens sends its setup frames at once, so these are the
/// connections opened after arming.
struct SlotsHold {
    gate: ReplyGate,
    /// Per node, the first connection index the hold applies to.
    from: [usize; NODES],
}

/// A's `-ASK` answer to one command, held behind a gate.
struct AskTrap {
    /// The command's first argument (its key).
    key: String,
    gate: ReplyGate,
    /// The port the `-ASK` names.
    target: u16,
}

/// The facade's shared state, read by every proxy's rule.
struct State {
    /// The proxied nodes' ports, in [`Node::ALL`]'s order, and the silent
    /// node's.
    ports: OnceLock<([u16; NODES], u16)>,
    /// Per node, one past the highest connection index that sent a frame.
    seen: [AtomicUsize; NODES],
    slots_hold: Mutex<Option<SlotsHold>>,
    /// `CLUSTER SLOTS` frames held so far.
    slots_held: AtomicUsize,
    /// Set by the test just before it opens the held `CLUSTER SLOTS`.
    slots_released: AtomicBool,
    /// The probe's key: a `SET` of it on A is recorded in [`Self::arrival`].
    probe: Mutex<Option<String>>,
    arrival: Mutex<Option<Arrival>>,
    /// A `GET` of this key on this node takes this action, once.
    get_trap: Mutex<Option<(Node, String, ProxyAction)>>,
    get_trapped: AtomicUsize,
    /// A command on A whose first argument is the trap's key is answered
    /// `-ASK` naming the trap's target, held behind its gate, once.
    ask_trap: Mutex<Option<AskTrap>>,
    ask_trapped: AtomicUsize,
    silent: Arc<Mutex<Vec<std::net::TcpStream>>>,
    /// An `ECHO` of one of these, on any node, is forwarded and answered,
    /// after which the proxy reads nothing more from that connection
    /// ([`ProxyAction::StopReading`]); each marker once.
    stop_reading: Mutex<Vec<String>>,
    /// When each `CLUSTER SLOTS` frame the hold applies to (a connection at or
    /// past its `from`) reached any node, in arrival order: a query from
    /// one of the fixture's other pools, on a connection it already held, is
    /// neither held nor recorded.
    slots_queries: Mutex<Vec<Instant>>,
    /// Watched markers, and when the first command on any node whose first
    /// argument is the marker arrived.
    markers: Mutex<Vec<(String, Option<Instant>)>>,
}

impl State {
    fn rule(self: &Arc<Self>, me: Node) -> ConnProxyRule {
        let state = Arc::clone(self);
        Arc::new(move |conn: usize, args: &[Vec<u8>]| state.on_frame(me, conn, args))
    }

    /// The facade's `CLUSTER SLOTS` reply, as a script: one range per node.
    fn slots_script(ports: &[u16; NODES]) -> String {
        let ranges: Vec<String> = Node::ALL
            .iter()
            .map(|node| {
                let first = node.index() as u16 * SLOTS_PER_NODE;
                format!(
                    "{{{first}, {}, {{'127.0.0.1', {}, '{}'}}}}",
                    first + SLOTS_PER_NODE - 1,
                    ports[node.index()],
                    NODE_IDS[node.index()]
                )
            })
            .collect();
        format!("return {{{}}}", ranges.join(", "))
    }

    fn on_frame(&self, me: Node, conn: usize, args: &[Vec<u8>]) -> Option<ProxyAction> {
        self.seen[me.index()].fetch_max(conn + 1, SeqCst);
        let (ports, _) = *self
            .ports
            .get()
            .expect("the ports are recorded before any client connects");
        let name = args.first()?.to_ascii_uppercase();
        let arg = |i: usize| args.get(i).map(|a| String::from_utf8_lossy(a).into_owned());
        let first = arg(1);
        if let Some(first) = first.as_deref() {
            let mut markers = self.markers.lock().expect("never poisoned");
            if let Some((_, at)) = markers
                .iter_mut()
                .find(|(marker, at)| marker == first && at.is_none())
            {
                *at = Some(Instant::now());
            }
        }
        if me == Node::A && name == b"SET" {
            let probe = self.probe.lock().expect("never poisoned");
            if probe.is_some() && first == *probe {
                let mut arrival = self.arrival.lock().expect("never poisoned");
                if arrival.is_none() {
                    *arrival = Some(Arrival {
                        at: Instant::now(),
                        slots_released: self.slots_released.load(SeqCst),
                        silent_closed: any_closed(&self.silent),
                    });
                }
            }
        }
        if me == Node::A {
            let mut trap = self.ask_trap.lock().expect("never poisoned");
            if trap
                .as_ref()
                .is_some_and(|t| Some(&t.key) == first.as_ref())
            {
                let trap = trap.take().expect("checked above");
                self.ask_trapped.fetch_add(1, SeqCst);
                let slot = fred::util::redis_keyslot(trap.key.as_bytes());
                let ask = error_reply_script(&format!("ASK {slot} 127.0.0.1:{}", trap.target));
                return Some(ProxyAction::GatedEval(trap.gate, ask));
            }
        }
        if name == b"ECHO" {
            let marker = first.as_deref()?;
            let mut markers = self.stop_reading.lock().expect("never poisoned");
            if let Some(at) = markers.iter().position(|m| m == marker) {
                markers.remove(at);
                return Some(ProxyAction::StopReading);
            }
        }
        if name == b"CLUSTER" {
            return match first?.to_ascii_uppercase().as_str() {
                "SLOTS" => {
                    let slots = Self::slots_script(&ports);
                    let hold = self.slots_hold.lock().expect("never poisoned");
                    match hold.as_ref() {
                        Some(hold) if conn >= hold.from[me.index()] => {
                            self.slots_queries
                                .lock()
                                .expect("never poisoned")
                                .push(Instant::now());
                            self.slots_held.fetch_add(1, SeqCst);
                            Some(ProxyAction::GatedEval(hold.gate.clone(), slots))
                        }
                        _ => Some(ProxyAction::Eval(slots)),
                    }
                }
                "INFO" => Some(ProxyAction::Eval(format!(
                    "return 'cluster_state:ok\\r\\ncluster_slots_assigned:16384\\r\\n\
                     cluster_known_nodes:{NODES}\\r\\ncluster_size:{NODES}\\r\\n'"
                ))),
                _ => None,
            };
        }
        if name == b"GET" {
            let key = first?;
            let mut trap = self.get_trap.lock().expect("never poisoned");
            if trap
                .as_ref()
                .is_some_and(|(node, k, _)| *node == me && *k == key)
            {
                let (_, _, action) = trap.take().expect("checked above");
                self.get_trapped.fetch_add(1, SeqCst);
                return Some(action);
            }
        }
        None
    }
}

/// See the module docs.
struct Facade {
    _proxies: Vec<ScriptReplyProxy>,
    silent: SilentNode,
    state: Arc<State>,
    ports: [u16; NODES],
}

impl Facade {
    async fn start() -> Self {
        let silent = SilentNode::start().await;
        let state = Arc::new(State {
            ports: OnceLock::new(),
            seen: Default::default(),
            slots_hold: Mutex::new(None),
            slots_held: AtomicUsize::new(0),
            slots_released: AtomicBool::new(false),
            probe: Mutex::new(None),
            arrival: Mutex::new(None),
            get_trap: Mutex::new(None),
            get_trapped: AtomicUsize::new(0),
            ask_trap: Mutex::new(None),
            ask_trapped: AtomicUsize::new(0),
            silent: Arc::clone(&silent.streams),
            stop_reading: Mutex::new(Vec::new()),
            slots_queries: Mutex::new(Vec::new()),
            markers: Mutex::new(Vec::new()),
        });
        let mut proxies = Vec::with_capacity(NODES);
        for node in Node::ALL {
            proxies.push(
                ScriptReplyProxy::start_per_connection(test_valkey_addr(), state.rule(node)).await,
            );
        }
        let mut ports = [0; NODES];
        for (port, proxy) in ports.iter_mut().zip(&proxies) {
            *port = proxy.port();
        }
        state
            .ports
            .set((ports, silent.port))
            .expect("the ports are recorded once");
        Self {
            _proxies: proxies,
            silent,
            state,
            ports,
        }
    }

    fn port(&self, node: Node) -> u16 {
        self.ports[node.index()]
    }

    /// A [`TestValkey`] connected through A in cluster mode with one data
    /// member, and that member: the subject. The proxies select
    /// `TEST_VALKEY_URL`'s database, so the fixture shares it.
    async fn subject(&self, deadline: Instant) -> (TestValkey, Client) {
        self.subject_with(deadline, LivenessOverrides::default())
            .await
    }

    /// [`Self::subject`] with [`F3_SLOW_BOUND`] as its liveness bound.
    async fn subject_with_the_short_bound(&self, deadline: Instant) -> (TestValkey, Client) {
        self.subject_with(
            deadline,
            LivenessOverrides {
                slow_timeout: Some(F3_SLOW_BOUND),
                ..Default::default()
            },
        )
        .await
    }

    /// [`Self::subject`], with `overrides` applied to its liveness bounds.
    async fn subject_with(
        &self,
        deadline: Instant,
        overrides: LivenessOverrides,
    ) -> (TestValkey, Client) {
        let tv = by(
            deadline,
            "connecting the subject through the facade",
            with_liveness_overrides_for_test(
                overrides,
                TestValkey::setup_with_url(
                    format!("redis-cluster://127.0.0.1:{}", self.port(Node::A)),
                    1,
                ),
            ),
        )
        .await;
        let subject = tv.client().pool.member(0).clone();
        assert!(
            subject.is_clustered(),
            "fixture self-check: the facade is read as a cluster"
        );
        (tv, subject)
    }

    /// From now on, a `CLUSTER SLOTS` on a connection accepted from now on is
    /// held until the returned guard opens it.
    fn hold_new_slots_queries(&self) -> SlotsRelease {
        let gate = ReplyGate::new();
        let mut from = [0; NODES];
        for (from, seen) in from.iter_mut().zip(&self.state.seen) {
            *from = seen.load(SeqCst);
        }
        *self.state.slots_hold.lock().expect("never poisoned") = Some(SlotsHold {
            gate: gate.clone(),
            from,
        });
        SlotsRelease {
            gate,
            state: Arc::clone(&self.state),
        }
    }

    fn slots_held(&self) -> usize {
        self.state.slots_held.load(SeqCst)
    }

    /// Record when the `SET` of `key` reaches A.
    fn watch_probe(&self, key: &str) {
        *self.state.probe.lock().expect("never poisoned") = Some(key.to_owned());
    }

    fn arrival(&self) -> Option<Arrival> {
        *self.state.arrival.lock().expect("never poisoned")
    }

    /// Record when the first command whose first argument is `marker`
    /// reaches any node.
    fn watch_marker(&self, marker: &str) {
        self.state
            .markers
            .lock()
            .expect("never poisoned")
            .push((marker.to_owned(), None));
    }

    fn marker_arrived(&self, marker: &str) -> Option<Instant> {
        self.state
            .markers
            .lock()
            .expect("never poisoned")
            .iter()
            .find(|(m, _)| m == marker)
            .and_then(|(_, at)| *at)
    }

    /// The next `GET` of `key` on `node` takes `action`.
    fn trap_get_on(&self, node: Node, key: &str, action: ProxyAction) {
        *self.state.get_trap.lock().expect("never poisoned") = Some((node, key.to_owned(), action));
    }

    fn get_trapped(&self) -> usize {
        self.state.get_trapped.load(SeqCst)
    }

    /// The next command on A whose first argument is `key` is answered `-ASK`
    /// naming the node listening on `target`, once `gate` opens.
    fn trap_on_a_with_ask(&self, key: &str, gate: &ReplyGate, target: u16) {
        *self.state.ask_trap.lock().expect("never poisoned") = Some(AskTrap {
            key: key.to_owned(),
            gate: gate.clone(),
            target,
        });
    }

    fn ask_trapped(&self) -> usize {
        self.state.ask_trapped.load(SeqCst)
    }

    /// Make `subject`'s connection to the node owning `marker` stop being
    /// read: an `ECHO` of `marker`, routed by it, is answered, and the proxy
    /// then reads nothing more from that connection, so what the client writes
    /// on it next backs up in the socket buffers.
    async fn stop_reading_after_echo(&self, deadline: Instant, subject: &Client, marker: &str) {
        self.state
            .stop_reading
            .lock()
            .expect("never poisoned")
            .push(marker.to_owned());
        let echoed: String = by(
            deadline,
            "the ECHO after which the node stops reading",
            subject.custom(
                CustomCommand::new_static("ECHO", ClusterHash::FirstKey, false),
                vec![marker.to_owned()],
            ),
        )
        .await
        .expect("fixture self-check: the node answers the ECHO after which it stops reading");
        assert_eq!(echoed, marker);
        assert!(
            self.state
                .stop_reading
                .lock()
                .expect("never poisoned")
                .is_empty(),
            "fixture self-check: the ECHO took its marker"
        );
    }

    /// [`Self::stop_reading_after_echo`] on `subject`'s connection to `node`.
    async fn stop_reading(&self, deadline: Instant, tv: &TestValkey, subject: &Client, node: Node) {
        let marker = self.key_on(tv, &format!("stop-{node:?}"), node);
        self.stop_reading_after_echo(deadline, subject, &marker)
            .await;
    }

    /// When the first `CLUSTER SLOTS` frame the hold applies to reached any
    /// node after `since` (see [`State::slots_queries`]).
    fn first_slots_query_since(&self, since: Instant) -> Option<Instant> {
        self.state
            .slots_queries
            .lock()
            .expect("never poisoned")
            .iter()
            .copied()
            .find(|at| *at > since)
    }

    /// The first key `<prefix>gbt-<what>-<i>` whose slot `node` owns.
    fn key_on(&self, tv: &TestValkey, what: &str, node: Node) -> String {
        (0..)
            .map(|i| format!("{}gbt-{what}-{i}", tv.prefix()))
            .find(|key| Node::owning(fred::util::redis_keyslot(key.as_bytes())) == node)
            .expect("keys hash to every node's range")
    }
}

/// Opens the held `CLUSTER SLOTS`, recording first that it was released; on
/// drop too, so a failing test still lets the subject finish its connect and
/// the fixture's drop-time cleanup run.
struct SlotsRelease {
    gate: ReplyGate,
    state: Arc<State>,
}

impl SlotsRelease {
    fn release(&self) {
        self.state.slots_released.store(true, SeqCst);
        self.gate.open();
    }
}

impl Drop for SlotsRelease {
    fn drop(&mut self) {
        self.release();
    }
}

/// Opens a gate on drop, for the same reason as [`SlotsRelease`].
struct OpenOnDrop(ReplyGate);

impl Drop for OpenOnDrop {
    fn drop(&mut self) {
        self.0.open();
    }
}

/// `CLIENT KILL ID <id>` on the standalone store behind the facade: the
/// proxy's upstream connection for one subject connection, so the proxy then
/// ends that connection.
async fn kill(deadline: Instant, direct: &TestValkey, id: i64) {
    let killed: i64 = by(
        deadline,
        "CLIENT KILL",
        direct.client().pool.next().custom(
            CustomCommand::new_static("CLIENT KILL", None::<u16>, false),
            vec!["ID".to_owned(), id.to_string()],
        ),
    )
    .await
    .expect("CLIENT KILL ID must succeed");
    assert_eq!(
        killed, 1,
        "CLIENT KILL must close exactly the subject's connection"
    );
}

/// The subject's connection id on the facade node listening on `port`.
fn connection_id(subject: &Client, port: u16) -> i64 {
    let ids = subject.connection_ids();
    *ids.iter()
        .find(|(server, _)| server.port == port)
        .map(|(_, id)| id)
        .unwrap_or_else(|| panic!("the subject holds a connection to port {port}: {ids:?}"))
}

/// A fred `Transaction` of one `SET` of `key`, sent now: polled once, which
/// queues it with the router, and returned still pending. The busy step
/// awaits it: its `drain_all` waits on every in-flight reply of the subject
/// before it writes anything. b.bo2 T4 drops it instead, so its caller has
/// given up and the router refuses it before any `drain_all`.
async fn send_transaction(
    subject: &Client,
    key: &str,
) -> BoxFuture<'static, Result<Value, fred::error::Error>> {
    let trx = subject.multi();
    let _: () = trx
        .set(key, "busy", None, None, false)
        .await
        .expect("queueing into a transaction is local");
    let mut busy = async move { trx.exec::<Value>(true).await }.boxed();
    send_now(&mut busy, "the transaction").await;
    busy
}

/// The probe: a `SET` of `key` with [`PROBE_TIMEOUT`], sent now, then driven
/// by a task of its own so its timeout runs.
async fn send_probe(
    subject: &Client,
    key: &str,
) -> tokio::task::JoinHandle<Result<(), fred::error::Error>> {
    let short = Options {
        timeout: Some(PROBE_TIMEOUT),
        ..Default::default()
    };
    let (client, key) = (subject.with_options(&short), key.to_owned());
    let mut probe = async move {
        client
            .set::<(), _, _>(&key, PROBE_VALUE, None, None, false)
            .await
    }
    .boxed();
    send_now(&mut probe, "the probe SET").await;
    tokio::spawn(probe)
}

/// Wait, at most [`HOLD_WINDOW`], for the probe to reach A while the
/// `CLUSTER SLOTS` stays held; release it; then return the probe's arrival,
/// failing when it never reached A at all.
async fn arrival_across_the_hold(
    deadline: Instant,
    facade: &Facade,
    held: &SlotsRelease,
) -> Arrival {
    let window = (Instant::now() + HOLD_WINDOW).min(deadline);
    eventually(window, || facade.arrival().is_some()).await;
    held.release();
    until(
        deadline,
        "precondition: the probe SET reaching A at all (it must be written before its \
         PROBE_TIMEOUT, or fred skips it)",
        || facade.arrival().is_some(),
    )
    .await;
    facade.arrival().expect("checked above")
}

/// The subject answers on both nodes once its connect finished, and A holds
/// the probe's value: the write landed.
async fn assert_subject_answers(
    deadline: Instant,
    facade: &Facade,
    tv: &TestValkey,
    subject: &Client,
    probe: &str,
) {
    let on_b = facade.key_on(tv, "after-on-b", Node::B);
    let read_b: Option<String> = by(deadline, "a GET on B after the connect", subject.get(&on_b))
        .await
        .expect("the subject must answer on B once its connect finished");
    assert_eq!(read_b, None, "the B key was never written");
    let read_a: Option<String> = by(deadline, "a GET on A after the connect", subject.get(probe))
        .await
        .expect("the subject must answer on A once its connect finished");
    assert_eq!(
        read_a.as_deref(),
        Some(PROBE_VALUE),
        "the probe SET landed on A"
    );
}

/// The probe's caller ends answered, or failed `Timeout` when its
/// [`PROBE_TIMEOUT`] ran out while the router, held in the connect, could not
/// read the reply. Which of the two is not the subject: the arrival is.
async fn settle_probe(
    deadline: Instant,
    probe: tokio::task::JoinHandle<Result<(), fred::error::Error>>,
) {
    let got = by(deadline, "the probe's caller", probe)
        .await
        .expect("the probe's task must not panic");
    if let Err(e) = got {
        assert_eq!(
            *e.kind(),
            fred::error::ErrorKind::Timeout,
            "the probe's caller ends answered or timed out; got {e:?}"
        );
    }
}

/// A `GET` the facade stalled or held, once its connection was replaced or its
/// reply released: answered, or failed transiently (its retry raced its
/// timeout). Never a hard error.
fn assert_answered_or_transient(what: &str, got: Result<Option<String>, fred::error::Error>) {
    if let Err(e) = got {
        assert!(
            is_transient_fred_error(&e),
            "{what} may fail only transiently; got {e:?}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// (R) a reconnect
// ═════════════════════════════════════════════════════════════════════════════

/// The probe is fed behind a `Reconnect`: the subject's router waits in a
/// `Transaction`'s drain on a stalled `GET` on B; `CLIENT KILL` of the
/// subject's B connection ends the drain (B's end of file reads as drained),
/// and the router reads that end of file again once the transaction is done,
/// before it takes the queued probe, so the `Reconnect` it defers lands behind
/// the probe. The reconnect attempt is then held at its `CLUSTER SLOTS`. The
/// probe must reach A while that attempt is held; once released, the subject
/// answers on both nodes.
///
/// Without the flush at the top of `reconnect_with_policy`'s loop the probe
/// stays in fred's write buffer until the attempt's retry-buffer flush, after
/// the release.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_fed_behind_a_reconnect_reaches_its_node_before_the_reconnect_connects() {
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let facade = Facade::start().await;
    let (tv, subject) = facade.subject(deadline).await;
    let direct = by(deadline, "connecting to the store", TestValkey::setup()).await;
    let stalled_key = facade.key_on(&tv, "stalled-on-b", Node::B);
    let busy_key = facade.key_on(&tv, "busy-on-a", Node::A);
    let probe_key = facade.key_on(&tv, "probe-on-a", Node::A);
    let b_id = connection_id(&subject, facade.port(Node::B));
    facade.watch_probe(&probe_key);
    facade.trap_get_on(Node::B, &stalled_key, ProxyAction::Stall);

    let stalled = {
        let (client, key) = (subject.clone(), stalled_key.clone());
        tokio::spawn(async move { client.get::<Option<String>, _>(&key).await })
    };
    until(deadline, "the stalled GET reaching B", || {
        facade.get_trapped() == 1
    })
    .await;
    let busy = send_transaction(&subject, &busy_key).await;
    until(deadline, "the router taking the transaction", || {
        queued(&subject) == 0
    })
    .await;
    let probe = send_probe(&subject, &probe_key).await;
    assert_eq!(
        queued(&subject),
        1,
        "self-check: the probe is queued behind the router's drain"
    );
    let held = facade.hold_new_slots_queries();
    kill(deadline, &direct, b_id).await;
    until(
        deadline,
        "the reconnect attempt's held CLUSTER SLOTS",
        || facade.slots_held() >= 1,
    )
    .await;

    let arrival = arrival_across_the_hold(deadline, &facade, &held).await;

    assert!(
        !arrival.slots_released,
        "the probe SET, fed just ahead of the Reconnect, must reach A before the reconnect \
         connects; it reached A only once the attempt's CLUSTER SLOTS, held for \
         {HOLD_WINDOW:?}, was released, after its caller's {PROBE_TIMEOUT:?} timeout: \
         {arrival:?}"
    );
    let busy = by(deadline, "the transaction", busy).await;
    assert!(
        busy.is_ok(),
        "precondition: the busy step ran; got {busy:?}"
    );
    settle_probe(deadline, probe).await;
    assert_answered_or_transient(
        "the stalled GET",
        by(deadline, "the stalled GET", stalled)
            .await
            .expect("the stalled GET's task must not panic"),
    );
    assert_subject_answers(deadline, &facade, &tv, &subject, &probe_key).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// (S) a cluster sync
// ═════════════════════════════════════════════════════════════════════════════

/// The probe is fed behind a `SyncCluster`: the subject's router waits in a
/// `Transaction`'s drain on a `GET` on B whose reply the facade holds; the
/// probe and then a `SyncCluster` are queued; the held reply is released, so
/// the router writes the probe only fed and runs the sync, which is held at
/// its `CLUSTER SLOTS`. The probe must reach A while the sync is held; once
/// released, the sync succeeds.
///
/// Without the flush at the top of `sync_cluster_with_policy`'s loop the
/// probe stays in fred's write buffer until the sync's retry-buffer flush,
/// after the release.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_fed_behind_a_cluster_sync_reaches_its_node_before_the_sync_connects() {
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let facade = Facade::start().await;
    let (tv, subject) = facade.subject(deadline).await;
    let held_key = facade.key_on(&tv, "held-on-b", Node::B);
    let busy_key = facade.key_on(&tv, "busy-on-a", Node::A);
    let probe_key = facade.key_on(&tv, "probe-on-a", Node::A);
    let replies = ReplyGate::new();
    let _replies_open = OpenOnDrop(replies.clone());
    facade.watch_probe(&probe_key);
    facade.trap_get_on(
        Node::B,
        &held_key,
        ProxyAction::HoldReplies(replies.clone()),
    );

    let held_get = {
        let (client, key) = (subject.clone(), held_key.clone());
        tokio::spawn(async move { client.get::<Option<String>, _>(&key).await })
    };
    until(deadline, "the held GET reaching B", || {
        facade.get_trapped() == 1
    })
    .await;
    let busy = send_transaction(&subject, &busy_key).await;
    until(deadline, "the router taking the transaction", || {
        queued(&subject) == 0
    })
    .await;
    let probe = send_probe(&subject, &probe_key).await;
    let mut sync = {
        let client = subject.clone();
        async move { client.sync_cluster().await }.boxed()
    };
    send_now(&mut sync, "the cluster sync").await;
    assert_eq!(
        queued(&subject),
        2,
        "self-check: the probe and then the sync are queued behind the router's drain"
    );
    let held = facade.hold_new_slots_queries();
    replies.open();
    until(deadline, "the sync attempt's held CLUSTER SLOTS", || {
        facade.slots_held() >= 1
    })
    .await;

    let arrival = arrival_across_the_hold(deadline, &facade, &held).await;

    assert!(
        !arrival.slots_released,
        "the probe SET, fed just ahead of the SyncCluster, must reach A before the sync \
         connects; it reached A only once the sync's CLUSTER SLOTS, held for \
         {HOLD_WINDOW:?}, was released, after its caller's {PROBE_TIMEOUT:?} timeout: \
         {arrival:?}"
    );
    let synced = by(deadline, "the cluster sync", sync).await;
    assert!(
        synced.is_ok(),
        "the released sync must succeed; got {synced:?}"
    );
    let busy = by(deadline, "the transaction", busy).await;
    assert!(
        busy.is_ok(),
        "precondition: the busy step ran; got {busy:?}"
    );
    settle_probe(deadline, probe).await;
    assert_answered_or_transient(
        "the held GET",
        by(deadline, "the held GET", held_get)
            .await
            .expect("the held GET's task must not panic"),
    );
    assert_subject_answers(deadline, &facade, &tv, &subject, &probe_key).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// (K) an -ASK target's on-demand connect
// ═════════════════════════════════════════════════════════════════════════════

/// The probe is fed behind an `Ask`: A holds its `-ASK` reply to a `GET` (the
/// target is the silent node), so the subject's router waits in a
/// `Transaction`'s drain on A; the probe is queued; the `-ASK` is released, so
/// the drain queues the `Ask` behind the probe, the router writes the probe
/// only fed, and the `Ask` opens a connection to the silent node, which never
/// answers, until `ASK_TARGET_CONNECT_CAP` (2 s) cuts it. The probe must reach
/// A before the client closes that connection: the silent node's end of it is
/// read when the probe arrives.
///
/// Without the flush in `ask::follow` before its connect the probe stays in
/// fred's write buffer until `before_wait` flushes it, after the connect was
/// cut.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_fed_behind_an_ask_redirect_reaches_its_node_before_the_ask_target_connect_ends() {
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let facade = Facade::start().await;
    let (tv, subject) = facade.subject(deadline).await;
    let asked_key = facade.key_on(&tv, "asked-on-a", Node::A);
    let busy_key = facade.key_on(&tv, "busy-on-b", Node::B);
    let probe_key = facade.key_on(&tv, "probe-on-a", Node::A);
    let ask = ReplyGate::new();
    let _ask_open = OpenOnDrop(ask.clone());
    facade.watch_probe(&probe_key);
    facade.trap_on_a_with_ask(&asked_key, &ask, facade.silent.port);

    let asked = {
        let (client, key) = (subject.clone(), asked_key.clone());
        tokio::spawn(async move { client.get::<Option<String>, _>(&key).await })
    };
    until(deadline, "the asked GET reaching A", || {
        facade.ask_trapped() == 1
    })
    .await;
    let busy = send_transaction(&subject, &busy_key).await;
    until(deadline, "the router taking the transaction", || {
        queued(&subject) == 0
    })
    .await;
    let probe = send_probe(&subject, &probe_key).await;
    assert_eq!(
        queued(&subject),
        1,
        "self-check: the probe is queued behind the router's drain"
    );
    ask.open();
    until(deadline, "the connect to the -ASK target", || {
        facade.silent.accepted() >= 1
    })
    .await;
    until(deadline, "the probe SET reaching A", || {
        facade.arrival().is_some()
    })
    .await;
    let arrival = facade.arrival().expect("checked above");

    assert!(
        !arrival.silent_closed,
        "the probe SET, fed just ahead of the Ask, must reach A before the -ASK target's \
         connect ends; it reached A only after the client closed that connection at the cap: \
         {arrival:?}"
    );
    let asked = by(deadline, "the asked GET", asked)
        .await
        .expect("the asked GET's task must not panic");
    let err = asked.expect_err("the -ASK target never answers");
    assert!(
        err.details().starts_with("Failed connecting to ASK target"),
        "precondition: the -ASK was followed into a connect cut at the cap; got {err:?}"
    );
    let busy = by(deadline, "the transaction", busy).await;
    assert!(
        busy.is_ok(),
        "precondition: the busy step ran; got {busy:?}"
    );
    settle_probe(deadline, probe).await;
    assert_subject_answers(deadline, &facade, &tv, &subject, &probe_key).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// F3: several silent connections cost one bound
// ═════════════════════════════════════════════════════════════════════════════

/// Two of the subject's connections hold fed frames their peers will not read,
/// and an inline connect follows: the flush before it costs the router one
/// liveness bound, not one per connection.
///
/// A and B each stop reading after a marker `ECHO`. A first `SyncCluster`
/// keeps the router busy, held at its `CLUSTER SLOTS`; behind it are queued a
/// one-attempt [`UNSENDABLE_VALUE_LEN`] `SET` to A, one to B, and a second
/// `SyncCluster`. Once the hold is released the router writes both `SET`s
/// only fed (commands still follow each), then runs the second sync, whose
/// flush before its attempt can complete on neither connection: each expires
/// at [`F3_SLOW_BOUND`], and only then does the attempt send its
/// `CLUSTER SLOTS`.
///
/// - **Count:** both `SET`s fail with the slow verdict, so both flushes ran
///   and expired (k = 2).
/// - **One bound:** the second sync's `CLUSTER SLOTS` reaches the facade at
///   least one bound after the release (the flush did wait out the bound) and
///   less than one bound plus [`F3_SLACK`] after it. Flushed one after the
///   other, the two connections would hold the router for two bounds.
///
/// A count or order needs a clock here: one expired bound against two is a
/// difference in time, and the expiry is the only event the stalled
/// connections produce. The margin is the one `stream_mux_reader_connect_test`'s
/// expired-flush tests use (one bound plus 2 s), against a regression that adds
/// a whole further bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_inline_connect_flushes_two_silent_connections_holding_fed_frames_in_one_bound() {
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let facade = Facade::start().await;
    let (tv, subject) = facade.subject_with_the_short_bound(deadline).await;
    let unsendable_on_a = facade.key_on(&tv, "unsendable-on-a", Node::A);
    let unsendable_on_b = facade.key_on(&tv, "unsendable-on-b", Node::B);
    facade.stop_reading(deadline, &tv, &subject, Node::A).await;
    facade.stop_reading(deadline, &tv, &subject, Node::B).await;

    let held = facade.hold_new_slots_queries();
    let mut first_sync = {
        let client = subject.clone();
        async move { client.sync_cluster().await }.boxed()
    };
    send_now(&mut first_sync, "the first sync").await;
    let first_sync = tokio::spawn(first_sync);
    until(deadline, "the first sync's held CLUSTER SLOTS", || {
        facade.slots_held() >= 1
    })
    .await;
    let value = "x".repeat(UNSENDABLE_VALUE_LEN);
    let mut unsendable = Vec::new();
    for key in [&unsendable_on_a, &unsendable_on_b] {
        unsendable.push(send_set(&subject, key, &value, SetAs::Typed, 1).await);
    }
    let mut second_sync = {
        let client = subject.clone();
        async move { client.sync_cluster().await }.boxed()
    };
    send_now(&mut second_sync, "the second sync").await;
    assert_eq!(
        queued(&subject),
        3,
        "self-check: both SETs and then the second sync are queued behind the first sync"
    );
    let second_sync = tokio::spawn(second_sync);

    let released = Instant::now();
    held.release();
    until(deadline, "the second sync's CLUSTER SLOTS", || {
        facade.first_slots_query_since(released).is_some()
    })
    .await;
    let attempt_after = facade
        .first_slots_query_since(released)
        .expect("checked above")
        - released;

    for (node, set) in ["A", "B"].into_iter().zip(unsendable) {
        assert_slow(deadline, &format!("{node}'s unsendable SET"), set).await;
    }
    assert!(
        attempt_after >= F3_SLOW_BOUND,
        "precondition: the flush before the second sync waited out the bound; its attempt \
         started {attempt_after:?} after the release"
    );
    assert!(
        attempt_after < F3_SLOW_BOUND + F3_SLACK,
        "the flush before an inline connect must cost one bound ({F3_SLOW_BOUND:?}) however many \
         connections are silent; with two, the second sync's attempt started {attempt_after:?} \
         after the release (two bounds, one after the other, is {:?})",
        F3_SLOW_BOUND * 2
    );
    for (what, sync) in [("first", first_sync), ("second", second_sync)] {
        let synced = by(deadline, "a sync", sync)
            .await
            .expect("the sync's task must not panic");
        assert!(synced.is_ok(), "the {what} sync succeeds; got {synced:?}");
    }
    assert_answers_on(deadline, &facade, &tv, &subject, &[Node::A, Node::B]).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// b.bo2: before a write that can reach a socket, and before the router waits
// ═════════════════════════════════════════════════════════════════════════════

/// How a [`send_set`] is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SetAs {
    /// fred's typed `SET`, which fred sends only by its counter and feed-count
    /// rule.
    Typed,
    /// The same command as a custom command, which fred always sends
    /// (`CommandKind::should_flush`).
    Custom,
}

/// A [`send_set`]'s task: the `SET`'s outcome, and when its caller had it.
type SentSet = tokio::task::JoinHandle<(Result<(), fred::error::Error>, Instant)>;

/// A `SET` of `value` to `key` with `attempts` write attempts (one: a failed
/// write fails it, with that write's error) and the test's whole budget as
/// its timeout, sent now and then driven by a task of its own.
async fn send_set(subject: &Client, key: &str, value: &str, how: SetAs, attempts: u32) -> SentSet {
    let options = Options {
        max_attempts: Some(attempts),
        timeout: Some(SCENARIO_BUDGET),
        ..Default::default()
    };
    let (client, key, value) = (
        subject.with_options(&options),
        key.to_owned(),
        value.to_owned(),
    );
    let mut set = async move {
        let got = match how {
            SetAs::Typed => {
                client
                    .set::<(), _, _>(&key, value.as_str(), None, None, false)
                    .await
            }
            SetAs::Custom => client
                .custom::<Value, _>(
                    CustomCommand::new_static("SET", ClusterHash::FirstKey, false),
                    vec![key, value],
                )
                .await
                .map(|_| ()),
        };
        (got, Instant::now())
    }
    .boxed();
    send_now(&mut set, "a SET").await;
    tokio::spawn(set)
}

/// A typed `GET` of `key`, sent now and then driven by a task of its own: the
/// command queued last, behind the writes a test stages.
async fn send_get(
    subject: &Client,
    key: &str,
) -> tokio::task::JoinHandle<Result<Option<String>, fred::error::Error>> {
    let (client, key) = (subject.clone(), key.to_owned());
    let mut get = async move { client.get::<Option<String>, _>(&key).await }.boxed();
    send_now(&mut get, "a GET").await;
    tokio::spawn(get)
}

/// The marker: once [`MARKER_AFTER`] has passed since `released`, a custom
/// `ECHO` of `marker` (routed by it), with the test's whole budget as its
/// timeout, from a task of its own. fred sends a custom command at once, so
/// the marker reaches its node as soon as the router is free to write it.
async fn send_marker(
    subject: &Client,
    marker: &str,
    released: Instant,
) -> tokio::task::JoinHandle<Result<String, fred::error::Error>> {
    tokio::time::sleep_until(released + MARKER_AFTER).await;
    let patient = Options {
        timeout: Some(SCENARIO_BUDGET),
        ..Default::default()
    };
    let (client, marker) = (subject.with_options(&patient), marker.to_owned());
    tokio::spawn(async move {
        client
            .custom::<String, _>(
                CustomCommand::new_static("ECHO", ClusterHash::FirstKey, false),
                vec![marker],
            )
            .await
    })
}

/// Whether the probe reached A within [`PROBE_DEADLINE`] of `released`,
/// waiting no longer than that (nor than the test's `deadline`): `Ok` with
/// how long after `released` it arrived, or `Err` naming when it arrived, if
/// it ever did. Left fed past the deadline, the probe may never be sent: a
/// connection judged silent is replaced, and fred does not send again a
/// command whose caller gave up.
async fn probe_within_its_deadline(
    deadline: Instant,
    facade: &Facade,
    released: Instant,
) -> Result<Duration, String> {
    eventually((released + PROBE_DEADLINE).min(deadline), || {
        facade.arrival().is_some()
    })
    .await;
    match facade.arrival() {
        Some(arrival) if arrival.at - released < PROBE_DEADLINE => Ok(arrival.at - released),
        Some(arrival) => Err(format!(
            "it reached A {:?} after the release",
            arrival.at - released
        )),
        None => Err(format!(
            "it had not reached A {PROBE_DEADLINE:?} after the release"
        )),
    }
}

/// How long after `released` the marker reached its node, failing when it
/// never did by the test's `deadline`.
async fn marker_arrived_after(
    deadline: Instant,
    facade: &Facade,
    marker: &str,
    released: Instant,
) -> Duration {
    until(deadline, "the marker reaching its node", || {
        facade.marker_arrived(marker).is_some()
    })
    .await;
    facade.marker_arrived(marker).expect("checked above") - released
}

/// `what`, a `SET` with one write attempt left when its write to a silent
/// connection waited out the bound, fails with the slow verdict: the bound
/// expired on its connection, which then failed. Returns when its caller had
/// the error.
async fn assert_slow(deadline: Instant, what: &str, task: SentSet) -> Instant {
    let (got, ended) = by(deadline, what, task)
        .await
        .unwrap_or_else(|e| panic!("{what}'s task must not panic: {e}"));
    let err = got.expect_err("a write whose bound expired cannot succeed");
    assert_eq!(
        verdict_of(&err),
        Some(Verdict::Slow),
        "precondition: {what} waited out the bound on its silent connection and failed with the \
         slow verdict; got {err:?}"
    );
    ended
}

/// The subject answers a `GET` of a never-written key on each of `nodes`, so
/// every silent connection was replaced.
async fn assert_answers_on(
    deadline: Instant,
    facade: &Facade,
    tv: &TestValkey,
    subject: &Client,
    nodes: &[Node],
) {
    for node in nodes {
        let key = facade.key_on(tv, "after", *node);
        let read: Option<String> = by(deadline, "a GET after the expiries", subject.get(&key))
            .await
            .expect("the subject must answer once its silent connections were replaced");
        assert_eq!(read, None, "the {node:?} key was never written");
    }
}

/// The probe landed: the subject reads its value on A.
async fn assert_probe_landed(deadline: Instant, subject: &Client, probe: &str) {
    let read: Option<String> = by(deadline, "a GET of the probe key", subject.get(probe))
        .await
        .expect("the subject must answer on A");
    assert_eq!(
        read.as_deref(),
        Some(PROBE_VALUE),
        "the probe SET landed on A"
    );
}

/// The b.bo2 tests' busy step: a `GET` on A whose replies the facade holds
/// behind a gate, then a `Transaction` on A, whose `drain_all` waits on that
/// reply. Opening the gate releases both.
struct Busy {
    transaction: BoxFuture<'static, Result<Value, fred::error::Error>>,
    held_get: tokio::task::JoinHandle<Result<Option<String>, fred::error::Error>>,
    replies: ReplyGate,
    _open: OpenOnDrop,
}

impl Busy {
    async fn on_a(deadline: Instant, facade: &Facade, tv: &TestValkey, subject: &Client) -> Self {
        let held_key = facade.key_on(tv, "held-on-a", Node::A);
        let busy_key = facade.key_on(tv, "busy-on-a", Node::A);
        let replies = ReplyGate::new();
        facade.trap_get_on(
            Node::A,
            &held_key,
            ProxyAction::HoldReplies(replies.clone()),
        );
        let held_get = {
            let (client, key) = (subject.clone(), held_key.clone());
            tokio::spawn(async move { client.get::<Option<String>, _>(&key).await })
        };
        until(deadline, "the held GET reaching A", || {
            facade.get_trapped() == 1
        })
        .await;
        let transaction = send_transaction(subject, &busy_key).await;
        until(deadline, "the router taking the transaction", || {
            queued(subject) == 0
        })
        .await;
        Self {
            transaction,
            held_get,
            _open: OpenOnDrop(replies.clone()),
            replies,
        }
    }

    /// Release the busy step; the instant it was released.
    fn release(&self) -> Instant {
        let released = Instant::now();
        self.replies.open();
        released
    }

    /// The busy step ran: the transaction and the held `GET` were answered.
    async fn assert_ran(self, deadline: Instant) {
        let busy = by(deadline, "the transaction", self.transaction).await;
        assert!(
            busy.is_ok(),
            "precondition: the busy step ran; got {busy:?}"
        );
        let held = by(deadline, "the held GET", self.held_get)
            .await
            .expect("the held GET's task must not panic");
        assert!(
            held.is_ok(),
            "precondition: the held GET was answered once released; got {held:?}"
        );
    }
}

/// b.bo2 T1, the realistic path and the counter-at-zero term: B, C and D stop
/// reading; behind the busy step the probe is queued, then a one-attempt
/// [`UNSENDABLE_VALUE_LEN`] typed `SET` to each of B, C and D. Released, the
/// router writes the probe and B's and C's `SET`s only fed (commands still
/// follow each), and D's, the last command, as a send: the command-queue
/// counter is at zero, and that is the only reason it is sent (D's buffer is
/// empty, it has no feeds, and a typed `SET` is no kind fred always sends).
/// Before that send can wait on D's socket, every connection holding fed
/// frames is flushed at once: the probe leaves at once, and B's and C's
/// flushes expire together at the bound; then D's send expires a bound later,
/// and its write error fails D's `SET` at once.
///
/// - **The probe** reaches A within [`PROBE_DEADLINE`] of the release.
///   Without the flush before D's send it waits out D's bound, fed on A.
/// - **Count:** all three `SET`s fail with the slow verdict.
/// - **The hold:** D's `SET` fails at least two bounds after the release (the
///   flush waited out B's and C's bound, then D's send its own) and less than
///   two bounds plus [`F3_SLACK`] after it. Flushed one after the other, B
///   and C would hold the router for three bounds before D's send ended.
///
/// No marker times this hold, unlike T4: a command queued while the router
/// is inside the flush raises the counter, so D's write would be only fed,
/// and the marker's own check, not D's send, would wait on D.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_fed_ahead_of_a_send_to_a_silent_node_reaches_its_node_before_that_send_waits() {
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let facade = Facade::start().await;
    let (tv, subject) = facade.subject_with_the_short_bound(deadline).await;
    let probe_key = facade.key_on(&tv, "probe-on-a", Node::A);
    for node in [Node::B, Node::C, Node::D] {
        facade.stop_reading(deadline, &tv, &subject, node).await;
    }
    facade.watch_probe(&probe_key);
    let busy = Busy::on_a(deadline, &facade, &tv, &subject).await;
    let probe = send_probe(&subject, &probe_key).await;
    let value = "x".repeat(UNSENDABLE_VALUE_LEN);
    let mut unsendable = Vec::new();
    for node in [Node::B, Node::C, Node::D] {
        let key = facade.key_on(&tv, "unsendable", node);
        unsendable.push((
            node,
            send_set(&subject, &key, &value, SetAs::Typed, 1).await,
        ));
    }
    assert_eq!(
        queued(&subject),
        4,
        "self-check: the probe and the three SETs are queued behind the router's drain"
    );

    let released = busy.release();

    if let Err(late) = probe_within_its_deadline(deadline, &facade, released).await {
        panic!(
            "the probe SET, fed ahead of a SET sent to a silent node, must reach A within \
             {PROBE_DEADLINE:?} of the release, before that send waits on the silent socket; \
             {late}"
        );
    }
    let mut d_failed = None;
    for (node, set) in unsendable {
        let ended = assert_slow(deadline, &format!("{node:?}'s unsendable SET"), set).await;
        if node == Node::D {
            d_failed = Some(ended - released);
        }
    }
    let hold = d_failed.expect("D's SET is one of the three");
    assert!(
        hold >= F3_SLOW_BOUND * 2,
        "precondition: the router waited out B's and C's flushes, then D's send; D's SET failed \
         {hold:?} after the release"
    );
    assert!(
        hold < F3_SLOW_BOUND * 2 + F3_SLACK,
        "the flush before a send must cost one bound however many fed connections are silent: \
         with B and C silent, then D's own send, D's SET must fail two bounds ({:?}) after the \
         release; it failed {hold:?} after it (one connection after another is three bounds)",
        F3_SLOW_BOUND * 2
    );
    busy.assert_ran(deadline).await;
    settle_probe(deadline, probe).await;
    assert_probe_landed(deadline, &subject, &probe_key).await;
    assert_answers_on(deadline, &facade, &tv, &subject, &Node::ALL).await;
}

/// The writes to B in b.bo2 T2, T3 and T6, each of which the check before a
/// write finds able to reach B's silent socket by one of its terms alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriteToB {
    /// A custom `SET`, which fred always sends: the command-kind term.
    CustomSend,
    /// A typed `SET` of the unsendable value, fed, then a small typed `SET`
    /// fed onto that full write buffer: the backpressure term.
    FeedAtTheBoundary,
    /// [`FEEDS_PAST_THE_COUNT`] small typed `GET`s fed in one pipeline, then a
    /// typed `SET` of the unsendable value, which fred would send because B's
    /// feed count has passed `max_feed_count`: the feed-count term. With the
    /// fix the `SET`'s check flushes A and B first, the `GET`s fit B's socket
    /// buffers and B's feed count resets, so the `SET` is only fed; B's bound
    /// then expires in the flush before the last `GET` (the counter at zero).
    SendPastTheFeedCount,
}

/// The first short key `f<i>` (outside the test's prefix, and read only:
/// nothing writes it) whose slot `node` owns. Short, so that
/// [`FEEDS_PAST_THE_COUNT`] fed `GET`s of it stay well below the 8 KiB
/// backpressure boundary (about 20 bytes each).
fn short_key_on(node: Node) -> String {
    (0..)
        .map(|i| format!("f{i}"))
        .find(|key| Node::owning(fred::util::redis_keyslot(key.as_bytes())) == node)
        .expect("short keys hash to every node's range")
}

/// b.bo2 T2, T3 and T6: B stops reading; behind the busy step the probe is
/// queued, then `write`'s commands to B, each with one write attempt, then a
/// `GET` on A. That last command keeps the counter above zero and the writes
/// to B below fred's feed count until `write`'s last one, so only `write`'s
/// own term makes the check before that write find it able to reach B's
/// socket. Released, the router writes the probe only fed, and that check
/// flushes A (and B, when B holds fed frames) before any wait on B's socket:
/// in T2 the custom `SET`'s own send then waits out B's bound; in T3 B's
/// flush in the check itself does; in T6 that flush completes (the fed
/// `GET`s fit B's socket buffers, and B's feed count resets, so the `SET` is
/// only fed), and B's bound expires in the flush before the last `GET`.
/// Either way the probe left first, so:
///
/// - **the probe** reaches A within [`PROBE_DEADLINE`] of the release;
/// - **no verdict on A:** for two bounds after the release, past the read
///   pass that follows B's expiry, the subject's error stream names no error
///   on A, and the subject's connection to A is the one it had. Left fed
///   through B's bound, the probe would make A read as silent there (a fed
///   frame's wait counts from when it was fed), and A would get the slow
///   verdict and be replaced;
/// - **the probe landed**, and every write to B failed with the slow verdict.
async fn assert_a_write_fed_ahead_of_a_write_to_a_silent_node_is_sent_first(write: WriteToB) {
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let facade = Facade::start().await;
    let (tv, subject) = facade.subject_with_the_short_bound(deadline).await;
    let probe_key = facade.key_on(&tv, "probe-on-a", Node::A);
    let unsendable_key = facade.key_on(&tv, "unsendable-on-b", Node::B);
    let small_key = facade.key_on(&tv, "small-on-b", Node::B);
    let last_key = facade.key_on(&tv, "last-on-a", Node::A);
    facade.stop_reading(deadline, &tv, &subject, Node::B).await;
    facade.watch_probe(&probe_key);
    let a_id = connection_id(&subject, facade.port(Node::A));
    let mut errors = subject.error_rx();
    let busy = Busy::on_a(deadline, &facade, &tv, &subject).await;
    let probe = send_probe(&subject, &probe_key).await;
    let value = "x".repeat(UNSENDABLE_VALUE_LEN);
    let mut to_b = Vec::new();
    let mut pipelined = None;
    match write {
        WriteToB::CustomSend => {
            to_b.push(send_set(&subject, &unsendable_key, &value, SetAs::Custom, 1).await);
        }
        WriteToB::FeedAtTheBoundary => {
            to_b.push(send_set(&subject, &unsendable_key, &value, SetAs::Typed, 1).await);
            to_b.push(send_set(&subject, &small_key, "small", SetAs::Typed, 1).await);
        }
        WriteToB::SendPastTheFeedCount => {
            let one_attempt = Options {
                max_attempts: Some(1),
                timeout: Some(SCENARIO_BUDGET),
                ..Default::default()
            };
            let pipeline = Pipeline::from(subject.with_options(&one_attempt));
            let fed = short_key_on(Node::B);
            for _ in 0..FEEDS_PAST_THE_COUNT {
                let _: () = pipeline
                    .get(&fed)
                    .await
                    .expect("queueing into a pipeline is local");
            }
            let mut gets = async move { pipeline.try_all::<Option<String>>().await }.boxed();
            send_now(&mut gets, "the pipeline of fed GETs").await;
            pipelined = Some(tokio::spawn(gets));
            to_b.push(send_set(&subject, &unsendable_key, &value, SetAs::Typed, 1).await);
        }
    }
    let last = send_get(&subject, &last_key).await;
    let staged = 2 + to_b.len() + usize::from(pipelined.is_some());
    assert_eq!(
        queued(&subject),
        staged,
        "self-check: the probe, the writes to B and the last GET are queued behind the \
         router's drain"
    );

    let released = busy.release();
    if let Err(late) = probe_within_its_deadline(deadline, &facade, released).await {
        panic!(
            "{write:?}: the probe SET, fed ahead of a write that waits on B's silent socket, \
             must reach A within {PROBE_DEADLINE:?} of the release, before that write waits; \
             {late}"
        );
    }
    for (i, set) in to_b.into_iter().enumerate() {
        assert_slow(deadline, &format!("{write:?}: write {i} to B"), set).await;
    }
    if let Some(pipelined) = pipelined {
        let gets = by(deadline, "the pipeline of fed GETs", pipelined)
            .await
            .expect("the pipeline's task must not panic");
        assert!(!gets.is_empty(), "precondition: the pipeline answered");
        for got in gets {
            let err =
                got.expect_err("a GET fed to a connection whose bound expired cannot succeed");
            assert_eq!(
                verdict_of(&err),
                Some(Verdict::Slow),
                "precondition: the pipelined GETs were fed to B and failed with its slow \
                 verdict; got {err:?}"
            );
        }
    }
    tokio::time::sleep_until(released + F3_SLOW_BOUND * 2).await;
    let mut on_a = Vec::new();
    loop {
        match errors.try_recv() {
            Ok((err, server)) => {
                if server.as_ref().map(|s| s.port) == Some(facade.port(Node::A)) {
                    on_a.push(err);
                }
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }
    assert!(
        on_a.is_empty(),
        "{write:?}: A answered the probe as soon as it was sent, so no verdict may close A; the \
         subject's error stream named A: {on_a:?}"
    );
    assert_eq!(
        connection_id(&subject, facade.port(Node::A)),
        a_id,
        "{write:?}: the subject's connection to A must be the one it had"
    );
    let last = by(deadline, "the last GET", last)
        .await
        .expect("the last GET's task must not panic");
    assert!(
        matches!(last, Ok(None)),
        "the last GET, on A, is answered; got {last:?}"
    );
    busy.assert_ran(deadline).await;
    settle_probe(deadline, probe).await;
    assert_probe_landed(deadline, &subject, &probe_key).await;
    assert_answers_on(deadline, &facade, &tv, &subject, &[Node::B]).await;
}

/// b.bo2 T2, the command-kind term: the write to B is a custom `SET`, which
/// fred always sends (see
/// [`assert_a_write_fed_ahead_of_a_write_to_a_silent_node_is_sent_first`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_fed_ahead_of_a_custom_command_to_a_silent_node_is_sent_first_and_its_connection_is_kept(
) {
    assert_a_write_fed_ahead_of_a_write_to_a_silent_node_is_sent_first(WriteToB::CustomSend).await;
}

/// b.bo2 T3, the backpressure term: the unsendable `SET` is fed to B, and the
/// small `SET` after it is a feed onto a write buffer past the 8 KiB
/// boundary, which first flushes that buffer (see
/// [`assert_a_write_fed_ahead_of_a_write_to_a_silent_node_is_sent_first`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_fed_ahead_of_a_feed_at_the_backpressure_boundary_is_sent_first_and_its_connection_is_kept(
) {
    assert_a_write_fed_ahead_of_a_write_to_a_silent_node_is_sent_first(WriteToB::FeedAtTheBoundary)
        .await;
}

/// b.bo2 T6, the feed-count term: [`FEEDS_PAST_THE_COUNT`] small `GET`s are
/// fed to B, so fred would send the unsendable `SET` after them although
/// commands still follow it. With the fix the `SET`'s check flushes A and B
/// first, which resets B's feed count, so the `SET` is only fed and B's bound
/// expires in the flush before the last `GET`; with the term made false, the
/// `SET` is sent and waits out B's bound with the probe still fed on A (see
/// [`assert_a_write_fed_ahead_of_a_write_to_a_silent_node_is_sent_first`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_fed_ahead_of_a_send_past_the_feed_count_is_sent_first_and_its_connection_is_kept()
{
    assert_a_write_fed_ahead_of_a_write_to_a_silent_node_is_sent_first(
        WriteToB::SendPastTheFeedCount,
    )
    .await;
}

/// b.bo2 T4, the flush before the router waits: B and C stop reading; behind
/// the busy step the probe is queued, then a one-attempt
/// [`UNSENDABLE_VALUE_LEN`] typed `SET` to each of B and C, then a fred
/// `Transaction` whose caller has already given up (its future is dropped
/// after the poll that queued it, which sets its give-up flag). Released, the
/// router writes the probe and both `SET`s with that transaction still
/// queued, so the counter is above zero at each, and none of them can reach a
/// socket (no counter at zero, no feed count, no kind fred always sends, every
/// buffer below the boundary): all three are only fed, and the check before a
/// write flushes nothing. The router then refuses the given-up transaction at
/// the top of its attempt, before its `drain_all`, with no I/O and no check
/// before a write (`live_edit::abandon::refuse_transaction`). With its channel
/// empty, it flushes every fed connection before it waits (`before_wait`):
/// the probe leaves at once, and B's and C's flushes expire together. The
/// counter here is fred's own, never a value the test stores, so no
/// scheduling of the test's threads can turn the last `SET` into a send.
///
/// - **The probe** reaches A within [`PROBE_DEADLINE`] of the release.
/// - **Count:** both `SET`s fail with the slow verdict.
/// - **One bound:** the marker reaches A at least one bound after the release
///   (the flush waited out the bound) and less than one bound plus
///   [`F3_SLACK`] after it. Flushed one after the other, B and C would hold
///   the router for two bounds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_flush_before_the_router_waits_flushes_two_silent_connections_in_one_bound() {
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let facade = Facade::start().await;
    let (tv, subject) = facade.subject_with_the_short_bound(deadline).await;
    let probe_key = facade.key_on(&tv, "probe-on-a", Node::A);
    let marker = facade.key_on(&tv, "marker-on-a", Node::A);
    for node in [Node::B, Node::C] {
        facade.stop_reading(deadline, &tv, &subject, node).await;
    }
    facade.watch_probe(&probe_key);
    facade.watch_marker(&marker);
    let busy = Busy::on_a(deadline, &facade, &tv, &subject).await;
    let probe = send_probe(&subject, &probe_key).await;
    let value = "x".repeat(UNSENDABLE_VALUE_LEN);
    let mut unsendable = Vec::new();
    for node in [Node::B, Node::C] {
        let key = facade.key_on(&tv, "unsendable", node);
        unsendable.push((
            node,
            send_set(&subject, &key, &value, SetAs::Typed, 1).await,
        ));
    }
    let given_up = send_transaction(&subject, &facade.key_on(&tv, "given-up-on-a", Node::A)).await;
    drop(given_up);
    assert_eq!(
        queued(&subject),
        4,
        "self-check: the probe, the two SETs and the given-up transaction are queued behind the \
         router's drain"
    );

    let released = busy.release();
    let marker_sent = send_marker(&subject, &marker, released).await;
    if let Err(late) = probe_within_its_deadline(deadline, &facade, released).await {
        panic!(
            "the probe SET, fed with nothing behind it, must be flushed before the router waits, \
             reaching A within {PROBE_DEADLINE:?} of the release; {late}"
        );
    }
    let hold = marker_arrived_after(deadline, &facade, &marker, released).await;
    for (node, set) in unsendable {
        assert_slow(deadline, &format!("{node:?}'s unsendable SET"), set).await;
    }
    assert!(
        hold >= F3_SLOW_BOUND,
        "precondition: the flush before the router waits waited out the bound; the marker \
         reached A {hold:?} after the release"
    );
    assert!(
        hold < F3_SLOW_BOUND + F3_SLACK,
        "the flush before the router waits must cost one bound ({F3_SLOW_BOUND:?}) however many \
         connections are silent; with two, the marker reached A {hold:?} after the release (two \
         bounds, one after the other, is {:?})",
        F3_SLOW_BOUND * 2
    );
    let echoed = by(deadline, "the marker", marker_sent)
        .await
        .expect("the marker's task must not panic");
    assert_eq!(echoed.ok().as_deref(), Some(marker.as_str()));
    busy.assert_ran(deadline).await;
    settle_probe(deadline, probe).await;
    assert_probe_landed(deadline, &subject, &probe_key).await;
    assert_answers_on(deadline, &facade, &tv, &subject, &[Node::B, Node::C]).await;
}

/// b.bo2 T5, the write to an `-ASK` target the client already holds a
/// connection to (`ask::follow`'s `live_edit::before_write_to`): B stops
/// reading; a typed [`UNSENDABLE_VALUE_LEN`] `SET` of a key on A, with two
/// write attempts (one for A, one for the redirect), is answered by A `-ASK`
/// naming B, held behind a gate, so a `Transaction` on C waits in its drain
/// on A; the probe is queued. Released, the drain queues the `Ask` behind the
/// probe, the router writes the probe only fed, and the `Ask` writes `ASKING`
/// and the `SET` to B, whose send waits out the bound on B's silent socket.
/// Before those writes every connection holding fed frames is flushed, so:
///
/// - **the probe** reaches A within [`PROBE_DEADLINE`] of the release;
/// - the redirected `SET` fails with the slow verdict: its write to B waited
///   out the bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_write_fed_ahead_of_an_ask_redirect_to_a_silent_node_reaches_its_node_before_the_redirected_write_waits(
) {
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let facade = Facade::start().await;
    let (tv, subject) = facade.subject_with_the_short_bound(deadline).await;
    let asked_key = facade.key_on(&tv, "asked-on-a", Node::A);
    let busy_key = facade.key_on(&tv, "busy-on-c", Node::C);
    let probe_key = facade.key_on(&tv, "probe-on-a", Node::A);
    facade.stop_reading(deadline, &tv, &subject, Node::B).await;
    let ask = ReplyGate::new();
    let _ask_open = OpenOnDrop(ask.clone());
    facade.watch_probe(&probe_key);
    facade.trap_on_a_with_ask(&asked_key, &ask, facade.port(Node::B));

    let value = "x".repeat(UNSENDABLE_VALUE_LEN);
    let asked = send_set(&subject, &asked_key, &value, SetAs::Typed, 2).await;
    until(deadline, "the asked SET reaching A", || {
        facade.ask_trapped() == 1
    })
    .await;
    let busy = send_transaction(&subject, &busy_key).await;
    until(deadline, "the router taking the transaction", || {
        queued(&subject) == 0
    })
    .await;
    let probe = send_probe(&subject, &probe_key).await;
    assert_eq!(
        queued(&subject),
        1,
        "self-check: the probe is queued behind the router's drain"
    );

    let released = Instant::now();
    ask.open();

    if let Err(late) = probe_within_its_deadline(deadline, &facade, released).await {
        panic!(
            "the probe SET, fed just ahead of an Ask whose target is a silent node the client is \
             connected to, must reach A within {PROBE_DEADLINE:?} of the release, before the \
             redirected write waits on that node's socket; {late}"
        );
    }
    assert_slow(deadline, "the redirected SET", asked).await;
    let busy = by(deadline, "the transaction", busy).await;
    assert!(
        busy.is_ok(),
        "precondition: the busy step ran; got {busy:?}"
    );
    settle_probe(deadline, probe).await;
    assert_probe_landed(deadline, &subject, &probe_key).await;
    assert_answers_on(deadline, &facade, &tv, &subject, &[Node::B]).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T2: a real cluster
// ═════════════════════════════════════════════════════════════════════════════

/// `node` pinned, one attempt: the command goes to that node only.
fn pinned_to(node: &Server) -> Options {
    Options {
        cluster_node: Some(node.clone()),
        max_attempts: Some(1),
        ..Default::default()
    }
}

/// The first key `<prefix>gbt-<what>-<i>` whose slot `node` serves, from
/// fred's cached routing table.
fn cluster_key_on(tv: &TestValkey, subject: &Client, what: &str, node: &Server) -> String {
    let routing = subject
        .cached_cluster_state()
        .expect("a connected clustered client has a cached CLUSTER SLOTS view");
    (0..)
        .map(|i| format!("{}gbt-{what}-{i}", tv.prefix()))
        .find(|key| routing.get_server(fred::util::redis_keyslot(key.as_bytes())) == Some(node))
        .expect("keys hash onto every primary")
}

/// A custom command on `node` from `observer`.
async fn on_node<R: fred::types::FromValue>(
    deadline: Instant,
    observer: &Client,
    node: &Server,
    command: &'static str,
    args: Vec<String>,
) -> R {
    by(
        deadline,
        command,
        observer
            .with_options(&pinned_to(node))
            .custom(CustomCommand::new_static(command, None::<u16>, false), args),
    )
    .await
    .unwrap_or_else(|e| panic!("{command} on {node} must succeed: {e}"))
}

/// The real-cluster form of (R). On a three-primary cluster the subject's
/// router waits in a `Transaction`'s drain on a `BLPOP` on C; the probe (a
/// key on A) is queued; an observer kills the subject's B connection and
/// pauses B (`CLIENT PAUSE ALL`, [`PAUSE_MS`]), sends a witness `PING` to B
/// that the pause holds, and releases the `BLPOP` with an `LPUSH`. The router
/// then writes the probe only fed, ahead of the `Reconnect` B's end of file
/// queued, and the reconnect's connects to B wait out the pause. The probe
/// must be readable on A while the witness is still held, and within half the
/// pause of the `CLIENT PAUSE` reply.
///
/// Without the flush the probe is sent by the reconnect's retry-buffer flush,
/// after the pause ended and the subject's setup exchange with B completed, so
/// the witness is answered first and the pause's first half is long over.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, CLIENT PAUSE suspends a whole cluster node"]
async fn fed_frame_flush_disruptive_a_write_fed_behind_a_reconnect_to_a_paused_node_lands_before_the_pause_ends(
) {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster-disruptive:` job \
             run this"
        );
        return;
    };
    let deadline = Instant::now() + SCENARIO_BUDGET;
    let tv = by(
        deadline,
        "connecting the subject",
        TestValkey::setup_with_url(url.clone(), 1),
    )
    .await;
    let watcher = by(
        deadline,
        "connecting the observer",
        TestValkey::setup_with_url(url, 1),
    )
    .await;
    let subject = tv.client().pool.member(0).clone();
    let observer = watcher.client().pool.member(0).clone();
    assert!(
        subject.is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (`redis-cluster://...`)"
    );
    let primaries = subject
        .cached_cluster_state()
        .expect("a connected clustered client has a cached CLUSTER SLOTS view")
        .unique_primary_nodes();
    assert!(
        primaries.len() >= 3,
        "the scenario needs three primaries; got {primaries:?}"
    );
    let (a, b, c) = (&primaries[0], &primaries[1], &primaries[2]);
    let probe_key = cluster_key_on(&tv, &subject, "probe-on-a", a);
    let list_key = cluster_key_on(&tv, &subject, "list-on-c", c);
    let busy_key = cluster_key_on(&tv, &subject, "busy-on-c", c);
    let ids = subject.connection_ids();
    let (b_id, c_id) = (ids[b], ids[c]);

    let blocked = {
        let (client, key) = (subject.clone(), list_key.clone());
        let patient = Options {
            timeout: Some(SCENARIO_BUDGET),
            ..Default::default()
        };
        tokio::spawn(async move {
            client
                .with_options(&patient)
                .blpop::<Value, _>(&key, 0.0)
                .await
        })
    };
    let blocked_on_c = format!("id={c_id} ");
    let mut listed = String::new();
    while !(listed.contains(&blocked_on_c) && listed.contains("cmd=blpop")) {
        assert!(
            Instant::now() < deadline,
            "the subject's BLPOP never blocked on C; CLIENT LIST ID {c_id}: {listed}"
        );
        tokio::time::sleep(POLL).await;
        listed = on_node(
            deadline,
            &observer,
            c,
            "CLIENT",
            vec!["LIST".to_owned(), "ID".to_owned(), c_id.to_string()],
        )
        .await;
    }
    let busy = send_transaction(&subject, &busy_key).await;
    until(deadline, "the router taking the transaction", || {
        queued(&subject) == 0
    })
    .await;
    let probe = send_probe(&subject, &probe_key).await;
    assert_eq!(
        queued(&subject),
        1,
        "self-check: the probe is queued behind the router's drain"
    );

    let killed: i64 = on_node(
        deadline,
        &observer,
        b,
        "CLIENT",
        vec!["KILL".to_owned(), "ID".to_owned(), b_id.to_string()],
    )
    .await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL must close exactly the subject's B connection"
    );
    let _: String = on_node(
        deadline,
        &observer,
        b,
        "CLIENT",
        vec!["PAUSE".to_owned(), PAUSE_MS.to_string(), "ALL".to_owned()],
    )
    .await;
    let paused = Instant::now();
    let half_pause = paused + Duration::from_millis(PAUSE_MS / 2);
    let witness = {
        let (client, b) = (observer.clone(), b.clone());
        tokio::spawn(async move {
            client
                .with_options(&pinned_to(&b))
                .custom::<String, String>(
                    CustomCommand::new_static("PING", None::<u16>, false),
                    Vec::new(),
                )
                .await
        })
    };
    let _: i64 = by(
        deadline,
        "the LPUSH releasing the BLPOP",
        observer.lpush(&list_key, "released"),
    )
    .await
    .expect("LPUSH on C must succeed");

    // A read counts only if, once it has returned, the witness is still
    // unanswered and no more than half the pause has passed since the
    // `CLIENT PAUSE` reply. The two cover each other's gaps: a GET sent just
    // before the pause ended can reach A after the retry buffer's late flush
    // and read the probe, so the witness is checked after the read, not before
    // it; and `is_finished` turns true only once the witness's task has run,
    // which can lag its reply, so the read must also land in the first half of
    // the pause, long before any flush after it could.
    let mut landed = None;
    let mut out_of_window = false;
    while landed.is_none() && !out_of_window {
        assert!(
            Instant::now() < deadline,
            "neither the probe nor the witness settled"
        );
        let read = by(
            deadline,
            "the observer's GET on A",
            observer.get::<Option<String>, _>(&probe_key),
        )
        .await
        .expect("the observer's GET on A must succeed");
        out_of_window = witness.is_finished() || Instant::now() > half_pause;
        if !out_of_window {
            landed = read;
        }
        tokio::time::sleep(POLL).await;
    }

    assert_eq!(
        landed.as_deref(),
        Some(PROBE_VALUE),
        "the probe SET, fed just ahead of the Reconnect, must land on A while the reconnect waits \
         on the paused B; it was not read on A within {}ms of the CLIENT PAUSE reply with the \
         witness PING on B still held by the pause",
        PAUSE_MS / 2
    );
    let witnessed = by(deadline, "the witness PING", witness)
        .await
        .expect("the witness task must not panic");
    assert!(
        witnessed.is_ok(),
        "the witness is answered once the pause ends: {witnessed:?}"
    );
    let busy = by(deadline, "the transaction", busy).await;
    assert!(
        busy.is_ok(),
        "precondition: the busy step ran; got {busy:?}"
    );
    let blocked = by(deadline, "the BLPOP", blocked)
        .await
        .expect("the BLPOP's task must not panic");
    assert!(
        blocked.is_ok(),
        "precondition: the LPUSH released the BLPOP; got {blocked:?}"
    );
    settle_probe(deadline, probe).await;
    let read_a: Option<String> = by(deadline, "the subject's GET on A", subject.get(&probe_key))
        .await
        .expect("the subject must answer on A once its reconnect finished");
    assert_eq!(read_a.as_deref(), Some(PROBE_VALUE));
    let read_b: Option<String> = by(
        deadline,
        "the subject's GET on B",
        subject.get(cluster_key_on(&tv, &subject, "after-on-b", b)),
    )
    .await
    .expect("the subject must answer on B once its reconnect finished");
    assert_eq!(read_b, None, "the B key was never written");
    println!(
        "CLUSTER-GATED RAN: storage::fed_frame_flush_test::fed_frame_flush_disruptive_a_write_fed_behind_a_reconnect_to_a_paused_node_lands_before_the_pause_ends shards={}",
        primaries.len()
    );
}
