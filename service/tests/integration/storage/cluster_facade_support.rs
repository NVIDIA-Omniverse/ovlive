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

//! A cluster facade over the one standalone test store, shared by the tests of
//! the vendored fred's redirect handling: `b_vn4_ask_follow_test` (an `-ASK`
//! followed to its target) and `b_myh_redirect_frame_test` (only an error
//! frame is a redirect).
//!
//! # The facade
//!
//! Three [`ScriptReplyProxy`]s in front of the one standalone test store,
//! each answering fred's cluster discovery (`CLUSTER SLOTS`: node A owns slots
//! `[0, SPLIT)`, node B owns `[SPLIT, 16384)`; `CLUSTER INFO`:
//! `cluster_state:ok`), so a client connected with a `redis-cluster://` URL
//! holds one connection to A and one to B per pooled client and runs fred's
//! clustered router against the real store. Node C is listed nowhere: a node
//! that owns no slot yet, as in a scale-out.
//!
//! A answers EVERY frame naming the trapped key, on every attempt, with a
//! redirect text naming the trap's target: `ASK <slot> <target>` (as a
//! migrating slot's source does for a key it no longer holds) or
//! `MOVED <slot> <target>` ([`Redirect`]). [`TrapReply`] says in which shape:
//! as an error reply (`redis.error_reply`, a real cluster refusal; the
//! default), as a status reply (`redis.status_reply`, data), or not at all,
//! the frame forwarded to the store so the client reads whatever the key
//! holds. A records each such frame either way, and counts any `ASKING` it
//! receives; it can hold a substituted reply behind a gate.
//!
//! The target (B or C) answers `ASKING` with `+OK` itself (the store is not a
//! cluster node, and would refuse it), relays everything else to the store,
//! and records every frame it receives with the connection it came on, in
//! arrival order. B can also hold the replies behind one marked `GET`
//! ([`ReplyGate`]), so a command is still in flight on B's connection when a
//! redirect lands, or drop its connection on an `ASKING`. C can refuse the
//! setup exchange of a connection opened to it ([`SetupFault`]): its `PING`
//! answered `-NOAUTH` or `-LOADING`, or its `CLUSTER INFO` answered
//! `cluster_state:fail`, each in the text Valkey 9.0.3 sends. A fourth node,
//! [`SilentNode`], accepts connections and never answers.
//!
//! A and B count the `CLUSTER SLOTS` frames they answer
//! ([`AskFacade::cluster_slots`]): one per cluster sync the client runs
//! against them.
//!
//! The knobs are test-only: nothing here models a server the service talks
//! to, beyond the shape of its replies.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use fred::interfaces::{ClientLike, KeysInterface};

use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::test_support::TestValkey;

use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ConnProxyRule, ProxyAction, ReplyGate, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and constants
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on connecting one client.
pub(crate) const CONNECT_BUDGET: Duration = Duration::from_secs(30);

/// Upper bound on each awaited command or read. A deadline, not a speed
/// claim: the healthy path takes milliseconds. A redirect that is never
/// answered fails here instead of at the hang-killer.
pub(crate) const SETTLE: Duration = Duration::from_secs(10);

/// Slots below this are A's, the rest B's.
pub(crate) const SPLIT: u16 = 8192;

/// The node ids the facade's `CLUSTER SLOTS` reply names.
const NODE_A_ID: &str = "cf00000000000000000000000000000000000aaa";
const NODE_B_ID: &str = "cf00000000000000000000000000000000000bbb";

/// `live_edit_valkey_router_restarts_total`.
const ROUTER_RESTARTS: &str = "live_edit_valkey_router_restarts_total";

/// Every `pool` label, restated (`VALKEY_POOLS`).
const POOLS: [&str; 6] = [
    "subscriber",
    "data",
    "txn",
    "subscriber_pool",
    "maintenance",
    "snapshot",
];

/// Valkey 9.0.3's reply to a command other than `AUTH` / `HELLO` on a
/// connection that has not authenticated (`server.c`, `shared.noautherr`).
pub(crate) const NOAUTH_REPLY: &str = "NOAUTH Authentication required.";

/// Valkey 9.0.3's reply to a command without the `loading` flag, `PING`
/// among them, while the node loads its dataset (`server.c`,
/// `shared.loadingerr`).
pub(crate) const LOADING_REPLY: &str = "LOADING Valkey is loading the dataset in memory";

// ═════════════════════════════════════════════════════════════════════════════
// The facade
// ═════════════════════════════════════════════════════════════════════════════

/// The node A's redirect names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// B, a primary the client already holds a connection to.
    B,
    /// C, a node no `CLUSTER SLOTS` reply names.
    C,
    /// A node no `CLUSTER SLOTS` reply names that accepts a connection and
    /// never answers on it ([`SilentNode`]).
    Silent,
}

/// The redirect text A answers a trapped key with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Redirect {
    /// `ASK <slot> <host:port>`.
    Ask,
    /// `MOVED <slot> <host:port>`.
    Moved,
}

/// The shape of A's answer to a frame naming the trapped key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrapReply {
    /// The redirect text as an error reply (`redis.error_reply`, `-<text>`):
    /// a real cluster refusal.
    ErrorFrame,
    /// The redirect text as a status reply (`redis.status_reply`, `+<text>`):
    /// data.
    StatusReply,
    /// No reply of A's own: the frame is forwarded to the store, so the
    /// client reads whatever the key holds there (a test stores the redirect
    /// text at the key with [`store_directly`] and
    /// [`AskFacade::redirect_text`]). Never held behind the gate.
    Forward,
}

/// What A answers for, and how.
#[derive(Clone, Debug)]
struct Trap {
    key: String,
    target: Target,
    redirect: Redirect,
    reply: TrapReply,
}

/// The redirect text naming `port` on `127.0.0.1` for `key`'s slot.
fn redirect_text(redirect: Redirect, key: &str, port: u16) -> String {
    let slot = fred::util::redis_keyslot(key.as_bytes());
    let name = match redirect {
        Redirect::Ask => "ASK",
        Redirect::Moved => "MOVED",
    };
    format!("{name} {slot} 127.0.0.1:{port}")
}

/// A Lua body answering `redis.status_reply(message)`, which Valkey returns
/// as a status reply (`+<message>`).
fn status_reply_script(message: &str) -> String {
    let escaped = message.replace('\\', "\\\\").replace('\'', "\\'");
    format!("return redis.status_reply('{escaped}')")
}

/// A listener that accepts every connection and never writes a byte on it,
/// holding each one open: a connect to it succeeds, and the connection's setup
/// exchange then never completes. Unlike a refused or reset connect, which
/// fails at once, it is only ever ended by a timeout.
pub(crate) struct SilentNode {
    pub(crate) port: u16,
    /// Connections accepted so far.
    accepted: Arc<AtomicUsize>,
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
        let counted = Arc::clone(&accepted);
        let accept = tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                counted.fetch_add(1, Ordering::SeqCst);
                held.push(stream);
            }
        });
        Self {
            port,
            accepted,
            accept,
        }
    }

    pub(crate) fn accepted(&self) -> usize {
        self.accepted.load(Ordering::SeqCst)
    }
}

impl Drop for SilentNode {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

/// One frame a node received: the client connection it came on (the
/// proxy's accept order) and its arguments.
#[derive(Clone, Debug)]
pub(crate) struct Received {
    pub(crate) conn: usize,
    pub(crate) args: Vec<String>,
}

impl Received {
    pub(crate) fn is(&self, name: &str) -> bool {
        self.args
            .first()
            .is_some_and(|a| a.eq_ignore_ascii_case(name))
    }

    pub(crate) fn names(&self, key: &str) -> bool {
        self.args[1..].iter().any(|a| a == key)
    }
}

type Log = Arc<Mutex<Vec<Received>>>;

/// How C refuses the setup exchange of a connection opened to it, as a real
/// node does while it is not ready to serve one ([`AskFacade::refuse_setup_on_c`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SetupFault {
    /// `PING` answered `-NOAUTH`, a node that requires a password the client
    /// does not send ([`NOAUTH_REPLY`]).
    NoAuth,
    /// `CLUSTER INFO` answered `cluster_state:fail`, a node still joining the
    /// cluster.
    ClusterStateFail,
    /// `PING` answered `-LOADING`, a node replaying its dataset
    /// ([`LOADING_REPLY`]).
    Loading,
}

/// See the module docs.
pub(crate) struct AskFacade {
    _a: ScriptReplyProxy,
    _b: ScriptReplyProxy,
    _c: ScriptReplyProxy,
    pub(crate) silent: SilentNode,
    port_a: u16,
    port_b: u16,
    pub(crate) port_c: u16,
    /// The key A answers for, the node its redirect names, and how.
    trap: Arc<Mutex<Option<Trap>>>,
    /// Frames naming the trapped key that reached A: each one write.
    a_writes: Arc<AtomicUsize>,
    /// Those frames themselves, in arrival order.
    a_log: Log,
    /// When set, C refuses each connection's setup exchange this way.
    c_setup_fault: Arc<Mutex<Option<SetupFault>>>,
    /// Setup commands C has refused: one per connection opened to it.
    c_setup_refusals: Arc<AtomicUsize>,
    /// `ASKING` frames that reached A. A is never an `-ASK` target here, so
    /// one is an `ASKING` routed by its slot (a replay) instead of followed.
    pub(crate) a_askings: Arc<AtomicUsize>,
    /// When set, A holds each reply it substitutes for a trapped frame back
    /// until the gate opens ([`ProxyAction::GatedEval`]).
    pub(crate) ask_gate: Arc<Mutex<Option<ReplyGate>>>,
    b_log: Log,
    c_log: Log,
    /// The key whose next `GET` on B is forwarded with every reply on its
    /// connection held behind the gate ([`ProxyAction::HoldReplies`]).
    pub(crate) hold_on_b: Arc<Mutex<Option<(String, ReplyGate)>>>,
    /// Set once B's rule has matched that `GET`: it is on B's wire.
    pub(crate) held_on_b: Arc<AtomicUsize>,
    /// When armed, B drops the connection in place of the next `ASKING` it
    /// receives ([`ProxyAction::Close`]), with that `ASKING` and the command
    /// behind it still in flight.
    pub(crate) close_on_b_asking: Arc<AtomicBool>,
    /// `ASKING` frames B dropped its connection on.
    pub(crate) closed_on_b: Arc<AtomicUsize>,
    /// `CLUSTER SLOTS` frames A and B answered.
    cluster_slots: Arc<AtomicUsize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Node {
    A,
    B,
    C,
}

impl AskFacade {
    pub(crate) async fn start() -> Self {
        let ports: Arc<OnceLock<(u16, u16, u16, u16)>> = Arc::new(OnceLock::new());
        let trap: Arc<Mutex<Option<Trap>>> = Arc::new(Mutex::new(None));
        let a_writes = Arc::new(AtomicUsize::new(0));
        let a_askings = Arc::new(AtomicUsize::new(0));
        let a_log: Log = Arc::default();
        let ask_gate: Arc<Mutex<Option<ReplyGate>>> = Arc::default();
        let b_log: Log = Arc::default();
        let c_log: Log = Arc::default();
        let c_setup_fault: Arc<Mutex<Option<SetupFault>>> = Arc::default();
        let c_setup_refusals = Arc::new(AtomicUsize::new(0));
        let hold_on_b: Arc<Mutex<Option<(String, ReplyGate)>>> = Arc::default();
        let held_on_b = Arc::new(AtomicUsize::new(0));
        let close_on_b_asking = Arc::new(AtomicBool::new(false));
        let closed_on_b = Arc::new(AtomicUsize::new(0));
        let cluster_slots = Arc::new(AtomicUsize::new(0));
        let rule = |me: Node| -> ConnProxyRule {
            let (ports, trap, a_writes) =
                (Arc::clone(&ports), Arc::clone(&trap), Arc::clone(&a_writes));
            let (a_askings, ask_gate) = (Arc::clone(&a_askings), Arc::clone(&ask_gate));
            let a_log = Arc::clone(&a_log);
            let (c_setup_fault, c_setup_refusals) =
                (Arc::clone(&c_setup_fault), Arc::clone(&c_setup_refusals));
            let (close_on_b_asking, closed_on_b) =
                (Arc::clone(&close_on_b_asking), Arc::clone(&closed_on_b));
            let cluster_slots = Arc::clone(&cluster_slots);
            let log = match me {
                Node::A => None,
                Node::B => Some(Arc::clone(&b_log)),
                Node::C => Some(Arc::clone(&c_log)),
            };
            let (hold_on_b, held_on_b) = (Arc::clone(&hold_on_b), Arc::clone(&held_on_b));
            Arc::new(move |conn: usize, args: &[Vec<u8>]| {
                let (port_a, port_b, port_c, port_silent) = *ports
                    .get()
                    .expect("the ports are recorded before any client connects");
                let name = args.first()?.to_ascii_uppercase();
                if me == Node::C {
                    let fault = *c_setup_fault.lock().expect("never poisoned");
                    let is_cluster_info = name == b"CLUSTER"
                        && args
                            .get(1)
                            .is_some_and(|sub| sub.eq_ignore_ascii_case(b"INFO"));
                    let refusal = match fault {
                        Some(SetupFault::NoAuth) if name == b"PING" => {
                            Some(error_reply_script(NOAUTH_REPLY))
                        }
                        Some(SetupFault::Loading) if name == b"PING" => {
                            Some(error_reply_script(LOADING_REPLY))
                        }
                        Some(SetupFault::ClusterStateFail) if is_cluster_info => Some(
                            "return 'cluster_state:fail\\r\\ncluster_slots_assigned:16384\\r\\n\
                             cluster_known_nodes:3\\r\\ncluster_size:2\\r\\n'"
                                .to_owned(),
                        ),
                        _ => None,
                    };
                    if let Some(script) = refusal {
                        c_setup_refusals.fetch_add(1, Ordering::SeqCst);
                        return Some(ProxyAction::Eval(script));
                    }
                }
                if name == b"CLUSTER" {
                    return match args.get(1)?.to_ascii_uppercase().as_slice() {
                        b"SLOTS" => {
                            if me != Node::C {
                                cluster_slots.fetch_add(1, Ordering::SeqCst);
                            }
                            Some(ProxyAction::Eval(format!(
                                "return {{{{0, {}, {{'127.0.0.1', {port_a}, '{NODE_A_ID}'}}}}, \
                                 {{{SPLIT}, 16383, {{'127.0.0.1', {port_b}, '{NODE_B_ID}'}}}}}}",
                                SPLIT - 1
                            )))
                        }
                        b"INFO" => Some(ProxyAction::Eval(
                            "return 'cluster_state:ok\\r\\ncluster_slots_assigned:16384\\r\\n\
                             cluster_known_nodes:2\\r\\ncluster_size:2\\r\\n'"
                                .to_owned(),
                        )),
                        _ => None,
                    };
                }
                if me == Node::A {
                    if name == b"ASKING" {
                        a_askings.fetch_add(1, Ordering::SeqCst);
                        return Some(ProxyAction::Eval(
                            "return redis.status_reply('OK')".to_owned(),
                        ));
                    }
                    let trap = trap.lock().expect("never poisoned");
                    let trap = trap.as_ref()?;
                    if !args[1..].iter().any(|a| a == trap.key.as_bytes()) {
                        return None;
                    }
                    a_writes.fetch_add(1, Ordering::SeqCst);
                    a_log.lock().expect("never poisoned").push(Received {
                        conn,
                        args: args
                            .iter()
                            .map(|a| String::from_utf8_lossy(a).into_owned())
                            .collect(),
                    });
                    let port = match trap.target {
                        Target::B => port_b,
                        Target::C => port_c,
                        Target::Silent => port_silent,
                    };
                    let text = redirect_text(trap.redirect, &trap.key, port);
                    let reply = match trap.reply {
                        TrapReply::ErrorFrame => error_reply_script(&text),
                        TrapReply::StatusReply => status_reply_script(&text),
                        TrapReply::Forward => return None,
                    };
                    return Some(match ask_gate.lock().expect("never poisoned").as_ref() {
                        Some(gate) => ProxyAction::GatedEval(gate.clone(), reply),
                        None => ProxyAction::Eval(reply),
                    });
                }
                let log = log.as_ref().expect("B and C keep a log");
                log.lock().expect("never poisoned").push(Received {
                    conn,
                    args: args
                        .iter()
                        .map(|a| String::from_utf8_lossy(a).into_owned())
                        .collect(),
                });
                if name == b"ASKING" {
                    if me == Node::B && close_on_b_asking.swap(false, Ordering::SeqCst) {
                        closed_on_b.fetch_add(1, Ordering::SeqCst);
                        return Some(ProxyAction::Close);
                    }
                    // The store is not a cluster node and would refuse it.
                    return Some(ProxyAction::Eval(
                        "return redis.status_reply('OK')".to_owned(),
                    ));
                }
                if me == Node::B && name == b"GET" {
                    let held = hold_on_b.lock().expect("never poisoned");
                    if let Some((key, gate)) = held.as_ref() {
                        if args.get(1).is_some_and(|a| a == key.as_bytes()) {
                            held_on_b.fetch_add(1, Ordering::SeqCst);
                            return Some(ProxyAction::HoldReplies(gate.clone()));
                        }
                    }
                }
                None
            })
        };
        let a = ScriptReplyProxy::start_per_connection(test_valkey_addr(), rule(Node::A)).await;
        let b = ScriptReplyProxy::start_per_connection(test_valkey_addr(), rule(Node::B)).await;
        let c = ScriptReplyProxy::start_per_connection(test_valkey_addr(), rule(Node::C)).await;
        let silent = SilentNode::start().await;
        let (port_a, port_b, port_c) = (a.port(), b.port(), c.port());
        ports
            .set((port_a, port_b, port_c, silent.port))
            .expect("the ports are recorded once");
        Self {
            _a: a,
            _b: b,
            _c: c,
            silent,
            port_a,
            port_b,
            port_c,
            trap,
            a_writes,
            a_log,
            c_setup_fault,
            c_setup_refusals,
            a_askings,
            ask_gate,
            b_log,
            c_log,
            hold_on_b,
            held_on_b,
            close_on_b_asking,
            closed_on_b,
            cluster_slots,
        }
    }

    /// A [`TestValkey`] connected through A in cluster mode, with a
    /// `pool_size` data pool. The proxies select `TEST_VALKEY_URL`'s database,
    /// so the fixture shares it.
    pub(crate) async fn test_valkey(&self, pool_size: u32) -> TestValkey {
        let tv = tokio::time::timeout(
            CONNECT_BUDGET,
            TestValkey::setup_with_url(
                format!("redis-cluster://127.0.0.1:{}", self.port_a),
                pool_size,
            ),
        )
        .await
        .expect("a client must connect through the cluster facade");
        assert!(
            tv.client().pool.next().is_clustered(),
            "fixture self-check: the facade is read as a cluster"
        );
        tv
    }

    /// From now on A answers every frame naming `key` with `-ASK` to `target`.
    pub(crate) fn trap(&self, key: &str, target: Target) {
        self.trap_as(key, target, Redirect::Ask, TrapReply::ErrorFrame);
    }

    /// From now on A answers every frame naming `key` with `redirect`'s text
    /// naming `target` ([`Self::redirect_text`]), in the shape `reply` says.
    pub(crate) fn trap_as(&self, key: &str, target: Target, redirect: Redirect, reply: TrapReply) {
        *self.trap.lock().expect("never poisoned") = Some(Trap {
            key: key.to_owned(),
            target,
            redirect,
            reply,
        });
    }

    /// Stop answering for the trapped key, so the fixture's drop-time cleanup
    /// reaches it by its slot.
    pub(crate) fn disarm(&self) {
        *self.trap.lock().expect("never poisoned") = None;
    }

    /// The text A's `redirect` for `key` names `target` with:
    /// `<ASK|MOVED> <key's slot> 127.0.0.1:<target's port>`.
    pub(crate) fn redirect_text(&self, key: &str, redirect: Redirect, target: Target) -> String {
        let port = match target {
            Target::B => self.port_b,
            Target::C => self.port_c,
            Target::Silent => self.silent.port,
        };
        redirect_text(redirect, key, port)
    }

    pub(crate) fn a_writes(&self) -> usize {
        self.a_writes.load(Ordering::SeqCst)
    }

    /// The frames naming the trapped key that reached A, answered or
    /// forwarded, in arrival order.
    pub(crate) fn a_log(&self) -> Vec<Received> {
        self.a_log.lock().expect("never poisoned").clone()
    }

    /// `CLUSTER SLOTS` frames A and B have answered so far: one per cluster
    /// sync the client ran against them.
    pub(crate) fn cluster_slots(&self) -> usize {
        self.cluster_slots.load(Ordering::SeqCst)
    }

    /// From now on C refuses each connection's setup exchange as `fault` says.
    pub(crate) fn refuse_setup_on_c(&self, fault: SetupFault) {
        *self.c_setup_fault.lock().expect("never poisoned") = Some(fault);
    }

    pub(crate) fn c_setup_refusals(&self) -> usize {
        self.c_setup_refusals.load(Ordering::SeqCst)
    }

    pub(crate) fn log(&self, target: Target) -> Vec<Received> {
        match target {
            Target::B => &self.b_log,
            Target::C => &self.c_log,
            Target::Silent => panic!("the silent node reads no frame, so it keeps no log"),
        }
        .lock()
        .expect("never poisoned")
        .clone()
    }
}

/// The first key `<prefix>cf-<what>-<i>` whose slot `on_a` says A owns (or,
/// with `on_a` false, B owns).
pub(crate) fn key_on(tv: &TestValkey, what: &str, on_a: bool) -> String {
    (0..)
        .map(|i| format!("{}cf-{what}-{i}", tv.prefix()))
        .find(|key| (fred::util::redis_keyslot(key.as_bytes()) < SPLIT) == on_a)
        .expect("keys hash to both halves of the slot space")
}

/// `key`'s value written straight to the store, past the facade: a write
/// through the facade to a trapped key would be redirected itself.
pub(crate) async fn store_directly(direct: &TestValkey, key: &str, value: &str) {
    direct
        .client()
        .pool
        .set::<(), _, _>(key, value, None, None, false)
        .await
        .expect("SET straight to the store must succeed");
}

// ═════════════════════════════════════════════════════════════════════════════
// Shared checks
// ═════════════════════════════════════════════════════════════════════════════

fn restarts_row(metrics: &Metrics) -> Vec<(&'static str, u64)> {
    let text = render(metrics);
    POOLS
        .iter()
        .map(|pool| {
            let n = sample_value(&text, ROUTER_RESTARTS, &[("pool", pool)])
                .unwrap_or_else(|| panic!("{ROUTER_RESTARTS}{{pool={pool:?}}} is seeded"));
            (*pool, n as u64)
        })
        .collect()
}

/// No router of any pool was restarted.
pub(crate) fn assert_no_router_restart(metrics: &Metrics) {
    assert_eq!(
        restarts_row(metrics),
        POOLS.iter().map(|p| (*p, 0)).collect::<Vec<_>>(),
        "handling a redirect-shaped reply must not end any router: no restart may be counted"
    );
}

/// Metrics hooked onto `tv`'s client.
pub(crate) async fn hooked(tv: &TestValkey) -> Arc<Metrics> {
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    metrics
}

/// A `GET`'s answer, its error as text, to compare against [`value`].
pub(crate) fn answered(
    got: Result<Option<String>, fred::error::Error>,
) -> Result<Option<String>, String> {
    got.map_err(|e| e.to_string())
}

/// The answer of a `GET` that read `stored`.
pub(crate) fn value(stored: &str) -> Result<Option<String>, String> {
    Ok(Some(stored.to_owned()))
}

/// `fut`, bounded by [`SETTLE`].
pub(crate) async fn settle<T>(what: &str, fut: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(SETTLE, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not settle within {SETTLE:?}"))
}
