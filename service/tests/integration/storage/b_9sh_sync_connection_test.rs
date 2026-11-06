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

//! b.9sh — a cluster sync drops the connection it opens for `CLUSTER SLOTS`.
//!
//! # The contract (vendored fred change 11)
//!
//! Every cluster sync reads the topology with one `CLUSTER SLOTS` in
//! `clustered::cluster_slots_backchannel`, on a connection `connect_any` opens
//! for it. Stock fred stored that connection as the client's backchannel
//! (`inner.update_backchannel`), so every clustered client held one more
//! socket, idle between syncs. The vendored fred drops it when the sync is
//! done with it (`live_edit::end_sync_backchannel`), so after a sync — on
//! every exit — the client's backchannel holds no connection
//! (`Backchannel::transport` is `None`), while the router's connection-id map
//! (`Backchannel::connection_ids`, what `ClientLike::connection_ids` reads)
//! still names a connection per primary.
//!
//! # The facade
//!
//! [`SyncFacade`]: two [`ScriptReplyProxy`]s, A and B, in front of the one
//! standalone test store, each answering `CLUSTER INFO` with
//! `cluster_state:ok` and `CLUSTER SLOTS` as the test stages it: the two-node
//! map (A owns `[0, 8192)`, B the rest), an error reply, no reply at all
//! (the frame stalled), a null, or a reply `ClusterRouting::rebuild` cannot
//! parse. It can also refuse the setup `PING` of every connection opened from
//! then on with `-LOADING`, so `connect_any` finds no node it can use. Every
//! other frame is relayed to the store. The client is a raw fred cluster
//! client on the facade's URL, so every sync here is one the test starts
//! (`sync_cluster`), and the client's backchannel is read directly
//! (`ClientLike::inner`); no service seam is involved.
//!
//! | Test | Exit of the sync | What it pins |
//! |---|---|---|
//! | [`a_sync_that_rebuilds_the_routing_holds_no_connection_after_it_and_keeps_the_id_map`] | success (the connect's sync, then a caller's) | no stored connection after either, a `CLUSTER SLOTS` reached the facade each time, and the id map names A and B |
//! | [`a_sync_answered_an_error_frame_holds_no_connection_after_it`] | an error reply (`frame_to_results`) | the sync fails with the staged `ERR b9sh …` text; no stored connection |
//! | [`a_sync_whose_cluster_slots_times_out_holds_no_connection_after_it`] | the request's `internal_command_timeout` | the sync fails `Timeout`; no stored connection |
//! | [`a_sync_that_connects_to_no_node_holds_no_connection_after_it`] | `connect_any` failure | every node's setup `PING` refused; the sync fails with the `-LOADING` refusal and sends no `CLUSTER SLOTS`; no stored connection |
//! | [`a_sync_answered_a_null_holds_no_connection_after_it`] | the null-response return | the sync fails `Protocol` "Invalid or missing CLUSTER SLOTS response."; no stored connection |
//! | [`a_sync_whose_reply_does_not_parse_holds_no_connection_after_it`] | a `rebuild` error | the sync fails `Protocol` "Invalid CLUSTER SLOTS response."; no stored connection |
//! | [`a_sync_whose_router_is_aborted_mid_request_holds_no_connection_after_it`] | the router task aborted while the `CLUSTER SLOTS` is in flight | no stored connection |
//!
//! Which exits tell the patch from stock fred. Stock fred stored the
//! connection (`update_backchannel`, `router/clustered.rs:478` before the
//! patch) as soon as its `CLUSTER SLOTS` reply was read, before the reply was
//! checked. So on the success exit, the error-reply exit (its
//! `frame_to_results(frame)?`, `:483`) and the unparseable-reply exit (its
//! `new_cache.rebuild(...)?`, `:496`) it left the connection stored, and
//! nothing cleared it afterwards (the router's `disconnect_all` leaves the
//! backchannel alone): each of those three tests fails at its own post-sync
//! check on stock fred, besides its opening one. The timeout and
//! `connect_any` exits return before any store, the null exit clears the
//! stored connection (`check_and_disconnect`, `:487`), and the abort drops it
//! with the router task's future, so on those four stock fred stored nothing
//! either: they pin that the patch leaves those exits clean, not the
//! difference. Every test first checks that the connect's own sync, which
//! succeeded, left nothing stored, so with
//! `inner.update_backchannel(transport).await;` restored at `:478` (the path
//! a sync with no stored connection takes) every test fails at that opening
//! check. The `:462` site is taken only when a stored connection answers
//! `CLUSTER SLOTS` with an error, which needs a backchannel command the
//! service never sends; it is not staged here.
//!
//! # Wait budget
//!
//! `CONNECT_BUDGET` (30 s) for the client's connect, then each sync, abort or
//! quit under [`SETTLE`] (10 s): at most 30 + 10 × 3 = 60 s per test, under
//! the 120 s hang-killer. These are deadlines; the healthy path takes
//! milliseconds, the timeout case its 1 s `internal_command_timeout` per try
//! (two tries). The other failing cases run at a 2 s
//! `internal_command_timeout` ([`FAILING_SYNC_COMMAND_TIMEOUT`]), so the
//! failed sync's retry waits on nothing longer. After a failed sync the
//! closing `quit` can wait out its whole [`SETTLE`] (observed: about 10 s in
//! some runs of the timeout and `connect_any` cases); its result is not
//! checked, and the case's worst case above counts it.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use fred::clients::Client;
use fred::error::ErrorKind;
use fred::interfaces::{ClientInterface, ClientLike, ClusterInterface};
use fred::prelude::{Builder, ReconnectPolicy};
use fred::types::config::{Config as FredConfig, Server};

use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ConnProxyRule, ProxyAction, ScriptReplyProxy,
};

/// Bound on connecting the client.
const CONNECT_BUDGET: Duration = Duration::from_secs(30);

/// Bound on each sync, abort or quit. A deadline, not a speed claim.
const SETTLE: Duration = Duration::from_secs(10);

/// `internal_command_timeout` for the cases whose sync fails on a reply or on
/// its connects: whatever of the failing sync's retry, and of the closing
/// `quit`, waits on it is cut short, well inside [`SETTLE`].
const FAILING_SYNC_COMMAND_TIMEOUT: Duration = Duration::from_secs(2);

/// Slots below this are A's, the rest B's.
const SPLIT: u16 = 8192;

const NODE_A_ID: &str = "b9sh0000000000000000000000000000000000aa";
const NODE_B_ID: &str = "b9sh0000000000000000000000000000000000bb";

/// Valkey's reply to a command without the `loading` flag while it loads.
const LOADING_REPLY: &str = "LOADING Valkey is loading the dataset in memory";

/// The error reply [`SlotsReply::ErrorFrame`] answers `CLUSTER SLOTS` with.
const STAGED_REFUSAL: &str = "ERR b9sh staged CLUSTER SLOTS refusal";

/// What the facade answers `CLUSTER SLOTS` with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotsReply {
    /// The two-node map.
    Map,
    /// `-ERR …`, an error reply.
    ErrorFrame,
    /// Nothing: the frame is never answered ([`ProxyAction::Stall`]).
    Stall,
    /// A null reply.
    Null,
    /// An array `ClusterRouting::rebuild` cannot parse: one slot range of one
    /// element, where a range needs a start, an end and a primary.
    Malformed,
}

/// See the module docs.
struct SyncFacade {
    _a: ScriptReplyProxy,
    _b: ScriptReplyProxy,
    port_a: u16,
    port_b: u16,
    reply: Arc<Mutex<SlotsReply>>,
    /// When set, every setup `PING` is answered `-LOADING`.
    refuse_ping: Arc<AtomicBool>,
    /// `CLUSTER SLOTS` frames A and B received.
    slots_seen: Arc<AtomicUsize>,
    /// Setup `PING`s refused.
    pings_refused: Arc<AtomicUsize>,
}

impl SyncFacade {
    async fn start() -> Self {
        let ports: Arc<OnceLock<(u16, u16)>> = Arc::new(OnceLock::new());
        let reply = Arc::new(Mutex::new(SlotsReply::Map));
        let refuse_ping = Arc::new(AtomicBool::new(false));
        let slots_seen = Arc::new(AtomicUsize::new(0));
        let pings_refused = Arc::new(AtomicUsize::new(0));
        let rule = || -> ConnProxyRule {
            let (ports, reply) = (Arc::clone(&ports), Arc::clone(&reply));
            let (refuse_ping, slots_seen, pings_refused) = (
                Arc::clone(&refuse_ping),
                Arc::clone(&slots_seen),
                Arc::clone(&pings_refused),
            );
            Arc::new(move |_conn: usize, args: &[Vec<u8>]| {
                let (port_a, port_b) = *ports
                    .get()
                    .expect("the ports are recorded before any client connects");
                let name = args.first()?.to_ascii_uppercase();
                if name == b"PING" && refuse_ping.load(Ordering::SeqCst) {
                    pings_refused.fetch_add(1, Ordering::SeqCst);
                    return Some(ProxyAction::Eval(error_reply_script(LOADING_REPLY)));
                }
                if name != b"CLUSTER" {
                    return None;
                }
                match args.get(1)?.to_ascii_uppercase().as_slice() {
                    b"INFO" => Some(ProxyAction::Eval(
                        "return 'cluster_state:ok\\r\\ncluster_slots_assigned:16384\\r\\n\
                         cluster_known_nodes:2\\r\\ncluster_size:2\\r\\n'"
                            .to_owned(),
                    )),
                    b"SLOTS" => {
                        slots_seen.fetch_add(1, Ordering::SeqCst);
                        Some(match *reply.lock().expect("never poisoned") {
                            SlotsReply::Map => ProxyAction::Eval(format!(
                                "return {{{{0, {}, {{'127.0.0.1', {port_a}, '{NODE_A_ID}'}}}}, \
                                 {{{SPLIT}, 16383, {{'127.0.0.1', {port_b}, '{NODE_B_ID}'}}}}}}",
                                SPLIT - 1
                            )),
                            SlotsReply::ErrorFrame => {
                                ProxyAction::Eval(error_reply_script(STAGED_REFUSAL))
                            }
                            SlotsReply::Stall => ProxyAction::Stall,
                            SlotsReply::Null => ProxyAction::Eval("return nil".to_owned()),
                            SlotsReply::Malformed => {
                                ProxyAction::Eval("return {{'b9sh'}}".to_owned())
                            }
                        })
                    }
                    _ => None,
                }
            })
        };
        let a = ScriptReplyProxy::start_per_connection(test_valkey_addr(), rule()).await;
        let b = ScriptReplyProxy::start_per_connection(test_valkey_addr(), rule()).await;
        let (port_a, port_b) = (a.port(), b.port());
        ports
            .set((port_a, port_b))
            .expect("the ports are recorded once");
        Self {
            _a: a,
            _b: b,
            port_a,
            port_b,
            reply,
            refuse_ping,
            slots_seen,
            pings_refused,
        }
    }

    fn answer_cluster_slots_with(&self, reply: SlotsReply) {
        *self.reply.lock().expect("never poisoned") = reply;
    }

    fn slots_seen(&self) -> usize {
        self.slots_seen.load(Ordering::SeqCst)
    }

    /// The two primaries, as the client's routing names them.
    fn primaries(&self) -> BTreeSet<Server> {
        [self.port_a, self.port_b]
            .into_iter()
            .map(|port| Server::new("127.0.0.1", port))
            .collect()
    }

    /// A raw fred cluster client on the facade, connected, with its router's
    /// handle. `internal_command_timeout` bounds each `CLUSTER SLOTS` request
    /// (and each setup command). One reconnection attempt: a failing sync
    /// retries once, then returns its error to `sync_cluster`'s caller.
    async fn client(
        &self,
        internal_command_timeout: Duration,
    ) -> (Client, fred::types::ConnectHandle) {
        let config = FredConfig::from_url(&format!("redis-cluster://127.0.0.1:{}", self.port_a))
            .expect("the facade URL parses");
        let mut builder = Builder::from_config(config);
        builder
            .with_connection_config(|c| c.internal_command_timeout = internal_command_timeout)
            .set_policy(ReconnectPolicy::new_constant(1, 10));
        let client = builder.build().expect("the raw cluster client builds");
        let router = client.connect();
        tokio::time::timeout(CONNECT_BUDGET, client.wait_for_connect())
            .await
            .expect("the raw cluster client must connect within its budget")
            .expect("the raw cluster client connects through the facade");
        assert!(
            client.is_clustered(),
            "fixture self-check: a cluster client"
        );
        (client, router)
    }
}

/// Whether `client`'s backchannel holds no connection: what every exit of a
/// sync must leave.
async fn backchannel_holds_no_connection(client: &Client) -> bool {
    client.inner().backchannel.transport.read().await.is_none()
}

/// The servers `client`'s connection-id map names, each with a positive id.
fn id_map_servers(client: &Client) -> BTreeSet<Server> {
    let ids = client.connection_ids();
    assert!(
        ids.values().all(|id| *id > 0),
        "the id map holds the connections' CLIENT IDs: {ids:?}"
    );
    ids.into_keys().collect()
}

/// The connect's sync succeeded and left no connection stored: the
/// precondition of every failing case.
async fn assert_connected_clean(facade: &SyncFacade, client: &Client) {
    assert!(
        facade.slots_seen() >= 1,
        "fixture self-check: the connect read the topology with CLUSTER SLOTS"
    );
    assert!(
        backchannel_holds_no_connection(client).await,
        "the connect's sync must not keep its CLUSTER SLOTS connection as the backchannel"
    );
}

/// How a staged fault must end the caller's sync.
struct FailingSync {
    /// What the fault is, for the messages.
    what: &'static str,
    /// The error `sync_cluster` returns: its kind and details, exactly.
    kind: ErrorKind,
    details: &'static str,
    /// Whether the sync reaches its `CLUSTER SLOTS` (every exit but a
    /// `connect_any` failure).
    sends_cluster_slots: bool,
}

/// A caller's `sync_cluster` under the facade's staged fault: it fails with
/// the fault's own error, reached (or, for a `connect_any` failure, did not
/// reach) the facade with a `CLUSTER SLOTS`, and leaves no connection stored.
async fn assert_failing_sync_holds_no_connection(
    facade: &SyncFacade,
    client: &Client,
    expected: FailingSync,
) {
    let FailingSync {
        what,
        kind,
        details,
        sends_cluster_slots,
    } = expected;
    let seen = facade.slots_seen();
    let result = tokio::time::timeout(SETTLE, client.sync_cluster())
        .await
        .unwrap_or_else(|_| panic!("{what}: the sync must end within {SETTLE:?}"));
    let err = result.expect_err(what);
    assert_eq!(
        (err.kind(), err.details()),
        (&kind, details),
        "{what}: the sync must fail with the fault's own error; got {err:?}"
    );
    if sends_cluster_slots {
        assert!(
            facade.slots_seen() > seen,
            "fixture self-check: {what}: the sync sent its CLUSTER SLOTS"
        );
    } else {
        assert_eq!(
            facade.slots_seen(),
            seen,
            "fixture self-check: {what}: no CLUSTER SLOTS was sent, so the sync ended in \
             connect_any"
        );
    }
    assert!(
        backchannel_holds_no_connection(client).await,
        "{what}: a failed sync must leave no connection stored"
    );
}

async fn quit(client: Client) {
    let _ = tokio::time::timeout(SETTLE, client.quit()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sync_that_rebuilds_the_routing_holds_no_connection_after_it_and_keeps_the_id_map() {
    let facade = SyncFacade::start().await;
    let (client, _router) = facade.client(Duration::from_secs(10)).await;
    assert_connected_clean(&facade, &client).await;
    assert_eq!(
        id_map_servers(&client),
        facade.primaries(),
        "after the connect's sync the id map names a connection per primary"
    );

    let seen = facade.slots_seen();
    tokio::time::timeout(SETTLE, client.sync_cluster())
        .await
        .expect("the sync must end within its budget")
        .expect("the sync succeeds");
    assert!(
        facade.slots_seen() > seen,
        "fixture self-check: the caller's sync sent a CLUSTER SLOTS"
    );
    assert!(
        backchannel_holds_no_connection(&client).await,
        "a successful sync must drop its CLUSTER SLOTS connection, not store it as the \
         backchannel (stock fred stored it: one more socket per clustered client)"
    );
    assert_eq!(
        id_map_servers(&client),
        facade.primaries(),
        "the router's connection-id map is still kept: a connection per primary"
    );
    quit(client).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sync_answered_an_error_frame_holds_no_connection_after_it() {
    let facade = SyncFacade::start().await;
    let (client, _router) = facade.client(FAILING_SYNC_COMMAND_TIMEOUT).await;
    assert_connected_clean(&facade, &client).await;
    facade.answer_cluster_slots_with(SlotsReply::ErrorFrame);
    assert_failing_sync_holds_no_connection(
        &facade,
        &client,
        FailingSync {
            what: "an error reply",
            kind: ErrorKind::Unknown,
            details: STAGED_REFUSAL,
            sends_cluster_slots: true,
        },
    )
    .await;
    facade.answer_cluster_slots_with(SlotsReply::Map);
    quit(client).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sync_whose_cluster_slots_times_out_holds_no_connection_after_it() {
    let facade = SyncFacade::start().await;
    // 1 s per CLUSTER SLOTS request: the stalled one times out well inside
    // the sync's budget, twice (one retry).
    let (client, _router) = facade.client(Duration::from_secs(1)).await;
    assert_connected_clean(&facade, &client).await;
    facade.answer_cluster_slots_with(SlotsReply::Stall);
    assert_failing_sync_holds_no_connection(
        &facade,
        &client,
        FailingSync {
            what: "a timed-out request",
            kind: ErrorKind::Timeout,
            details: "Request timed out.",
            sends_cluster_slots: true,
        },
    )
    .await;
    facade.answer_cluster_slots_with(SlotsReply::Map);
    quit(client).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sync_that_connects_to_no_node_holds_no_connection_after_it() {
    let facade = SyncFacade::start().await;
    let (client, _router) = facade.client(FAILING_SYNC_COMMAND_TIMEOUT).await;
    assert_connected_clean(&facade, &client).await;
    facade.refuse_ping.store(true, Ordering::SeqCst);
    assert_failing_sync_holds_no_connection(
        &facade,
        &client,
        FailingSync {
            what: "connect_any failing on every node",
            kind: ErrorKind::Unknown,
            details: LOADING_REPLY,
            sends_cluster_slots: false,
        },
    )
    .await;
    assert!(
        facade.pings_refused.load(Ordering::SeqCst) >= 2,
        "fixture self-check: the sync's connect_any tried both nodes and was refused"
    );
    facade.refuse_ping.store(false, Ordering::SeqCst);
    quit(client).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sync_answered_a_null_holds_no_connection_after_it() {
    let facade = SyncFacade::start().await;
    let (client, _router) = facade.client(FAILING_SYNC_COMMAND_TIMEOUT).await;
    assert_connected_clean(&facade, &client).await;
    facade.answer_cluster_slots_with(SlotsReply::Null);
    assert_failing_sync_holds_no_connection(
        &facade,
        &client,
        FailingSync {
            what: "a null reply",
            kind: ErrorKind::Protocol,
            details: "Invalid or missing CLUSTER SLOTS response.",
            sends_cluster_slots: true,
        },
    )
    .await;
    facade.answer_cluster_slots_with(SlotsReply::Map);
    quit(client).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sync_whose_reply_does_not_parse_holds_no_connection_after_it() {
    let facade = SyncFacade::start().await;
    let (client, _router) = facade.client(FAILING_SYNC_COMMAND_TIMEOUT).await;
    assert_connected_clean(&facade, &client).await;
    facade.answer_cluster_slots_with(SlotsReply::Malformed);
    assert_failing_sync_holds_no_connection(
        &facade,
        &client,
        FailingSync {
            what: "a reply rebuild cannot parse",
            kind: ErrorKind::Protocol,
            details: "Invalid CLUSTER SLOTS response.",
            sends_cluster_slots: true,
        },
    )
    .await;
    facade.answer_cluster_slots_with(SlotsReply::Map);
    quit(client).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sync_whose_router_is_aborted_mid_request_holds_no_connection_after_it() {
    let facade = SyncFacade::start().await;
    let (client, router) = facade.client(Duration::from_secs(10)).await;
    assert_connected_clean(&facade, &client).await;
    facade.answer_cluster_slots_with(SlotsReply::Stall);
    let seen = facade.slots_seen();
    let sync = {
        let client = client.clone();
        tokio::spawn(async move { client.sync_cluster().await })
    };
    let deadline = tokio::time::Instant::now() + SETTLE;
    while facade.slots_seen() == seen {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the caller's sync never sent its CLUSTER SLOTS"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // The CLUSTER SLOTS is in flight, stalled, on the sync's own connection.
    router.abort();
    let ended = tokio::time::timeout(SETTLE, router).await;
    assert!(
        ended.is_ok(),
        "the aborted router task must end within {SETTLE:?}"
    );
    assert!(
        backchannel_holds_no_connection(&client).await,
        "a router aborted mid-sync must leave no connection stored"
    );
    sync.abort();
    facade.answer_cluster_slots_with(SlotsReply::Map);
}
