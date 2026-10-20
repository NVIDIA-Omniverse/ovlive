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

//! b.vn4 — a pooled cluster client follows an `-ASK` redirect to its target.
//!
//! # The contract (SDD §15.3)
//!
//! A command with a write attempt left that is answered `-ASK <slot> <target>`
//! is written next to the target's connection, right behind a pipelined
//! `ASKING`, and is answered there. It is never re-routed by slot, and no
//! reply on the target's connection is read outside the ordered reply
//! matching. A command with no write attempt left still fails `Routing` /
//! `"Max attempts reached."` with nothing written. The mechanism is the
//! vendored fred's `live_edit::ask::follow`, called from `process_ask`
//! (`vendor/fred-10.1.0.SOURCE.md`).
//!
//! Two exceptions. A **script** (`EVAL`, `EVALSHA`, their `_RO` forms,
//! `FCALL`, `FCALL_RO`, by fred kind or by custom-command name in any case)
//! is never followed: it fails `Routing` "Max attempts reached." (transient)
//! with nothing sent to the target, because on the target it inherits
//! `ASKING` and an undeclared key still on the source reads as absent. And
//! a **failed connect** to an `-ASK` target answers the command with an
//! error naming the target, `IO` (transient) unless the failure is a
//! credential or configuration fault (`Auth`, `Tls`, `Config`, `Url`), which
//! keeps its kind; the failure is broadcast on the client's error events
//! with its own kind either way.
//!
//! Stock fred 10.1.0 sent the `ASKING` to the target and then the command to
//! the slot's owner again, which answered `-ASK` again until the command's
//! write attempts ran out: no `-ASK` was ever followed, so during a live slot
//! migration every pooled command on a key already moved, or never written,
//! failed. It also read the `ASKING` reply synchronously off the target's
//! connection, which takes the reply of any command already in flight there.
//!
//! # The two-node `-ASK` facade
//!
//! Three [`ScriptReplyProxy`]s in front of the one standalone test store,
//! each answering fred's cluster discovery (`CLUSTER SLOTS`: node A owns slots
//! `[0, SPLIT)`, node B owns `[SPLIT, 16384)`; `CLUSTER INFO`:
//! `cluster_state:ok`), so a client connected with a `redis-cluster://` URL
//! holds one connection to A and one to B per pooled client and runs fred's
//! clustered router against the real store. Node C is listed nowhere: a node
//! that owns no slot yet, as in a scale-out.
//!
//! A answers `-ASK <slot> <target>` to EVERY frame naming the trapped key, on
//! every attempt, as a migrating slot's source does for a key it no longer
//! holds. The target (B or C) answers `ASKING` with `+OK` itself (the store
//! is not a cluster node, and would refuse it), relays everything else to the
//! store, and records every frame it receives with the connection it came on,
//! in arrival order. B can also hold the replies behind one marked `GET`
//! ([`ReplyGate`]), so a command is still in flight on B's connection when
//! the redirect lands, or drop its connection on an `ASKING`. A counts any
//! `ASKING` it receives, records every frame it answers `-ASK`, and can hold
//! its `-ASK` behind a gate. C can refuse the setup exchange of a connection
//! opened to it ([`SetupFault`]): its `PING` answered `-NOAUTH` or
//! `-LOADING`, or its `CLUSTER INFO` answered `cluster_state:fail`, each in
//! the text Valkey 9.0.3 sends. A fourth node, [`SilentNode`], accepts
//! connections and never answers.
//!
//! | Test | Item | What it pins |
//! |---|---|---|
//! | [`a_command_answered_ask_is_answered_by_the_ask_target_right_behind_asking`] | T1 (a), (b), (f) | a default-budget `GET` on the trapped key returns the key's value; A received one write; B received `ASKING` and right behind it, on the same connection, the `GET`; no router restart |
//! | [`a_one_attempt_command_answered_ask_still_fails_max_attempts_reached_and_sends_nothing_to_the_target`] | T1 (c) | a command with one write attempt (`lua::evalsha_once_with_reload`'s budget) fails `Routing` "Max attempts reached.", transient; nothing reaches B |
//! | [`a_command_in_flight_on_the_ask_target_and_the_redirected_command_each_get_their_own_reply`] | T1 (d), (f) | with a `GET` in flight on B and its reply held behind a gate, `ASKING` and the redirected `GET` arrive right behind it before the gate opens; each gets its own value; no router restart |
//! | [`an_ask_to_a_node_that_owns_no_slot_is_followed_through_a_connection_opened_for_it`] | T1 (e), (f) | the target is C: a connection is opened to it (`valkey_reconnects_total{pool="data"}` +1), `ASKING` and the `GET` arrive on it back to back, and the `GET` returns the value |
//! | [`an_ask_to_a_node_that_never_answers_fails_the_command_at_the_cap_and_stalls_the_client_no_longer`] | T1 (e′), (f), A′ | the target accepts and never answers: the connect is cut at `ASK_TARGET_CONNECT_CAP` (2 s), inside the 5 s command timeout, not at the client's 30 s connection timeout; the `GET` fails `IO` "Failed connecting to ASK target …", transient, no sooner than the cap; a command on the same pooled client issued during the connect is answered with its value; a second redirect to the node connects anew (nothing was left in the router's map) and fails the same way |
//! | [`ask_target_connect_timeout_is_the_shorter_of_the_client_timeout_and_the_cap`] | A′ | the bound is min(client timeout, cap), and the cap for a client timeout of zero (fred's "none") |
//! | [`an_asking_in_flight_when_the_targets_connection_drops_is_not_replayed_to_the_slots_owner`] | T1 (h), (f) | B drops its connection on the followed redirect's `ASKING`: the replay discards that `ASKING` (written with no attempt left), so A never receives one; no router restart |
//! | [`a_command_that_timed_out_before_its_ask_arrived_writes_nothing_to_the_target`] | T1 (i), (f) | A holds its `-ASK` behind a gate until the `GET`'s caller has received `Timeout`: once released, nothing of the `GET` — no `ASKING`, no `GET` — reaches B |
//! | [`a_structural_read_whose_deleted_sections_count_is_answered_ask_answers_the_objects_state`] | T1 (g) | `read_object_snapshot_slim(StructuralUpdate)` with `obj:dels` trapped answers the atomic read's compact projection, its `SCARD obj:dels` answered by B right behind `ASKING` |
//! | [`a_structural_reads_window_script_answered_ask_is_not_followed_and_the_read_fails_retryable`] | T1 (j) | the same read with the header trapped: its window script (a custom `EVALSHA`, default budget) reaches A once and fails the read `Aborted` on "Max attempts reached."; no script reaches B; the read's plain commands on the header still follow |
//! | [`an_atomic_snapshot_reads_script_answered_ask_is_not_followed_and_the_read_fails_retryable`] | T1 (j) | the catch-up atomic read (fred's `EVALSHA` kind) with the header trapped fails `Aborted` on "Max attempts reached."; A received the script once; nothing — no `ASKING`, nothing naming the header — reaches B |
//! | [`every_script_command_form_answered_ask_fails_at_once_and_a_custom_plain_command_still_follows`] | T1 (j) | fred's `EVAL`, `EVALSHA`, `FCALL`, `FCALL_RO` and custom `evalsha`, `EVAL_RO`, `evalsha_RO`, `Fcall_Ro` each fail `Routing` "Max attempts reached.", transient, sent to A once, nothing naming the key at B, a writing script's key left unwritten; no `ASKING` at B; a custom `GET` still follows |
//! | [`an_ask_target_refusing_noauth_fails_the_command_auth_naming_it_and_the_service_answers_internal`] | T1 (k) | C answers setup's `PING` `-NOAUTH`: the `GET` fails `Auth`, naming C and "Authentication Error"; not transient; `map_fred_err_for_grpc` answers `Internal` (gRPC `INTERNAL`); one error event, kind `Auth`, naming C, and `live_edit_valkey_errors_total{pool="data",kind="Authentication Error"}` 1; no reconnect event |
//! | [`an_ask_target_reporting_cluster_state_fail_fails_the_command_io_and_the_service_retries`] | T1 (k) | C answers setup's `CLUSTER INFO` `cluster_state:fail`: the `GET` fails `IO`, naming C and "Invalid or missing cluster state."; transient; `Unavailable`; the event and `{kind="Protocol Error"}` keep `Protocol` |
//! | [`an_ask_target_loading_its_dataset_fails_the_command_io_and_the_service_retries`] | T1 (k) | C answers setup's `PING` `-LOADING`: the `GET` fails `IO`, naming C and the `LOADING` reply; transient; `Unavailable`; the event and `{kind="Unknown Error"}` keep `Unknown` |
//!
//! (a), (d), (e) and (g) fail on stock fred: (a), (e) and (g) with `Routing`
//! "Max attempts reached." once A has answered each of the command's three
//! writes `-ASK`, and (d) because the router blocks reading the `ASKING`
//! reply off B's connection, behind the held reply, so the redirected `GET`
//! never reaches B. The one-node facade in `b_v6d_router_test.rs` cannot show
//! any of it: its `-ASK` names the node itself, and it traps one-attempt
//! commands only.
//!
//! (j) fails with `live_edit::ask::follow`'s script rule removed: every
//! script follows to B, where it runs on the whole store (the reads answer,
//! the `SET` script writes its key) or, for the `FCALL` forms, whose function
//! is not loaded, is refused there — either way it reached the target. The
//! custom forms in mixed or lower case fail when the name match is
//! case-sensitive. (k)'s `-NOAUTH` case fails when the connect error is
//! always `IO`; its `cluster_state` and `-LOADING` cases fail when it keeps
//! every kind, and when the broadcast carries the answered kind instead of
//! the failure's own.
//!
//! # Not staged
//!
//! Of `live_edit::ask` (`vendor/fred-10.1.0/src/live_edit/ask.rs`), these are
//! not staged here:
//!
//! - the on-demand connection's close at the next cluster sync that does not
//!   list its node: stock fred's sync (`router/clustered.rs:205-218`), not
//!   code of the follow;
//! - the two write-error branches (writing the `ASKING`, writing the command):
//!   they need a client-side write to fail at that exact moment, and they
//!   reuse stock fred's drop-and-reconnect shape;
//! - "No connection to ASK target": unreachable once `add_connection`
//!   succeeded;
//! - the TLS branch of `target_server`: there is no TLS cluster fixture;
//! - the kept `Tls`, `Config` and `Url` kinds of a failed connect: `Tls`
//!   needs a TLS cluster fixture (and only an invalid TLS server name
//!   produces it under rustls: a bad certificate arrives as `IO`), and
//!   `Config` / `Url` come from the client's own configuration, which a
//!   facade node's replies cannot stage; `Auth` stands for the allowlist's
//!   kept kinds;
//! - a `-WRONGPASS` setup refusal: a client with no password sends no
//!   `AUTH`, so C refuses its `PING` `-NOAUTH`, the same `Auth` kind;
//! - a script that writes, followed into a partial state: the facade's
//!   target relays to the whole store, so it shows the rule (nothing sent),
//!   not the partial read the rule prevents.
//!
//! # Wait budget
//!
//! [`CONNECT_BUDGET`] (30 s) on the one client a test connects, then each
//! awaited command or bounded wait under [`SETTLE`] (10 s). Worst cases: (a),
//! (c), (e), (h) 40 s; (i) 60 s (three commands); (d) 70 s (the held
//! command's arrival, the redirect's landing, then the two replies); (e′) 70 s
//! (the silent node's accept, the command beside the connect, the redirected
//! command, the second redirected command); (g) 30 s connect plus its layer
//! search (up to [`MAX_LAYERS_PER_SEARCH`] creates, each bounded by fred's 5 s
//! command timeout but milliseconds in practice) and two reads under
//! [`SETTLE`]; the two (j) reads the same with one read each; (j)'s forms
//! 30 s plus one [`SETTLE`] over all eight forms and one over the control
//! (50 s); (k) 30 s plus two commands and the metric wait (60 s). Inside
//! nextest's 120 s hang-killer. The healthy path takes
//! well under a second, except (e′), which waits out the 2 s connect cap
//! twice, and (i), which waits out its `GET`'s 1 s timeout, both by
//! construction.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use fred::interfaces::{
    ClientLike, EventInterface, FunctionInterface, KeysInterface, LuaInterface,
};
use fred::types::config::{Options, Server};
use fred::types::{ClusterHash, CustomCommand};

use fred::live_edit::ask::{ask_target_connect_timeout, ASK_TARGET_CONNECT_CAP};
use live_edit::domain::types::{ObjectState, Section};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::{
    is_transient_fred_error, map_fred_err_for_grpc, FredErrOrigin, POOL_LABEL_DATA,
};
use live_edit::storage::diff_read::{
    read_object_snapshot_slim, SlimSnapshotReadConsumer, SnapshotReadOf,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::UPDATE_LAYER_COMMIT_MAX_ATTEMPTS;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{compact_snapshot, TestValkey};

use crate::storage::snapshot_read_support::{
    error_reply_script, read_atomic, read_err, test_valkey_addr, ConnProxyRule, ProxyAction,
    ReplyGate, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and constants
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on connecting one client.
const CONNECT_BUDGET: Duration = Duration::from_secs(30);

/// Upper bound on each awaited command or read. A deadline, not a speed
/// claim: the healthy path takes milliseconds. A redirect that is never
/// answered fails here instead of at the hang-killer.
const SETTLE: Duration = Duration::from_secs(10);

/// Slots below this are A's, the rest B's.
const SPLIT: u16 = 8192;

/// Upper bound on layers created while looking for one whose slot A owns.
/// Each try lands there with even odds; the bound turns an impossible search
/// into a failure instead of a hang.
const MAX_LAYERS_PER_SEARCH: usize = 64;

/// The node ids the facade's `CLUSTER SLOTS` reply names.
const NODE_A_ID: &str = "b0vn40000000000000000000000000000000aaa0";
const NODE_B_ID: &str = "b0vn40000000000000000000000000000000bbb0";

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

/// The value stored at the trapped key: what the redirected command must
/// return.
const ASKED_VALUE: &str = "bvn4-asked-value";

/// The value stored at the key of the command held in flight on B.
const IN_FLIGHT_VALUE: &str = "bvn4-in-flight-value";

// ═════════════════════════════════════════════════════════════════════════════
// The facade
// ═════════════════════════════════════════════════════════════════════════════

/// Where A's `-ASK` sends the trapped key's commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// B, a primary the client already holds a connection to.
    B,
    /// C, a node no `CLUSTER SLOTS` reply names.
    C,
    /// A node no `CLUSTER SLOTS` reply names that accepts a connection and
    /// never answers on it ([`SilentNode`]).
    Silent,
}

/// A listener that accepts every connection and never writes a byte on it,
/// holding each one open: a connect to it succeeds, and the connection's setup
/// exchange then never completes. Unlike a refused or reset connect, which
/// fails at once, it is only ever ended by a timeout.
struct SilentNode {
    port: u16,
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

    fn accepted(&self) -> usize {
        self.accepted.load(Ordering::SeqCst)
    }
}

impl Drop for SilentNode {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

/// One frame a target node received: the client connection it came on (the
/// proxy's accept order) and its arguments.
#[derive(Clone, Debug)]
struct Received {
    conn: usize,
    args: Vec<String>,
}

impl Received {
    fn is(&self, name: &str) -> bool {
        self.args
            .first()
            .is_some_and(|a| a.eq_ignore_ascii_case(name))
    }

    fn names(&self, key: &str) -> bool {
        self.args[1..].iter().any(|a| a == key)
    }
}

type Log = Arc<Mutex<Vec<Received>>>;

/// How C refuses the setup exchange of a connection opened to it, as a real
/// node does while it is not ready to serve one ([`AskFacade::refuse_setup_on_c`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SetupFault {
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

/// Valkey 9.0.3's reply to a command other than `AUTH` / `HELLO` on a
/// connection that has not authenticated (`server.c`, `shared.noautherr`).
const NOAUTH_REPLY: &str = "NOAUTH Authentication required.";

/// Valkey 9.0.3's reply to a command without the `loading` flag, `PING`
/// among them, while the node loads its dataset (`server.c`,
/// `shared.loadingerr`).
const LOADING_REPLY: &str = "LOADING Valkey is loading the dataset in memory";

/// fred's text for a connection whose `CLUSTER INFO` does not report
/// `cluster_state:ok` (`src/protocol/connection.rs:615`).
const CLUSTER_STATE_INVALID: &str = "Invalid or missing cluster state.";

/// See the module docs.
struct AskFacade {
    _a: ScriptReplyProxy,
    _b: ScriptReplyProxy,
    _c: ScriptReplyProxy,
    silent: SilentNode,
    port_a: u16,
    port_c: u16,
    /// The key A answers `-ASK` for, and the node it names.
    trap: Arc<Mutex<Option<(String, Target)>>>,
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
    a_askings: Arc<AtomicUsize>,
    /// When set, A holds each `-ASK` back until the gate opens
    /// ([`ProxyAction::GatedEval`]).
    ask_gate: Arc<Mutex<Option<ReplyGate>>>,
    b_log: Log,
    c_log: Log,
    /// The key whose next `GET` on B is forwarded with every reply on its
    /// connection held behind the gate ([`ProxyAction::HoldReplies`]).
    hold_on_b: Arc<Mutex<Option<(String, ReplyGate)>>>,
    /// Set once B's rule has matched that `GET`: it is on B's wire.
    held_on_b: Arc<AtomicUsize>,
    /// When armed, B drops the connection in place of the next `ASKING` it
    /// receives ([`ProxyAction::Close`]), with that `ASKING` and the command
    /// behind it still in flight.
    close_on_b_asking: Arc<AtomicBool>,
    /// `ASKING` frames B dropped its connection on.
    closed_on_b: Arc<AtomicUsize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Node {
    A,
    B,
    C,
}

impl AskFacade {
    async fn start() -> Self {
        let ports: Arc<OnceLock<(u16, u16, u16, u16)>> = Arc::new(OnceLock::new());
        let trap: Arc<Mutex<Option<(String, Target)>>> = Arc::new(Mutex::new(None));
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
        let rule = |me: Node| -> ConnProxyRule {
            let (ports, trap, a_writes) =
                (Arc::clone(&ports), Arc::clone(&trap), Arc::clone(&a_writes));
            let (a_askings, ask_gate) = (Arc::clone(&a_askings), Arc::clone(&ask_gate));
            let a_log = Arc::clone(&a_log);
            let (c_setup_fault, c_setup_refusals) =
                (Arc::clone(&c_setup_fault), Arc::clone(&c_setup_refusals));
            let (close_on_b_asking, closed_on_b) =
                (Arc::clone(&close_on_b_asking), Arc::clone(&closed_on_b));
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
                        b"SLOTS" => Some(ProxyAction::Eval(format!(
                            "return {{{{0, {}, {{'127.0.0.1', {port_a}, '{NODE_A_ID}'}}}}, \
                             {{{SPLIT}, 16383, {{'127.0.0.1', {port_b}, '{NODE_B_ID}'}}}}}}",
                            SPLIT - 1
                        ))),
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
                    let (key, target) = trap.as_ref()?;
                    if !args[1..].iter().any(|a| a == key.as_bytes()) {
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
                    let slot = fred::util::redis_keyslot(key.as_bytes());
                    let port = match target {
                        Target::B => port_b,
                        Target::C => port_c,
                        Target::Silent => port_silent,
                    };
                    let ask = error_reply_script(&format!("ASK {slot} 127.0.0.1:{port}"));
                    return Some(match ask_gate.lock().expect("never poisoned").as_ref() {
                        Some(gate) => ProxyAction::GatedEval(gate.clone(), ask),
                        None => ProxyAction::Eval(ask),
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
        let (port_a, port_c) = (a.port(), c.port());
        ports
            .set((port_a, b.port(), port_c, silent.port))
            .expect("the ports are recorded once");
        Self {
            _a: a,
            _b: b,
            _c: c,
            silent,
            port_a,
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
        }
    }

    /// A [`TestValkey`] connected through A in cluster mode, with a
    /// `pool_size` data pool. The proxies select `TEST_VALKEY_URL`'s database,
    /// so the fixture shares it.
    async fn test_valkey(&self, pool_size: u32) -> TestValkey {
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
    fn trap(&self, key: &str, target: Target) {
        *self.trap.lock().expect("never poisoned") = Some((key.to_owned(), target));
    }

    /// Stop answering `-ASK`, so the fixture's drop-time cleanup reaches the
    /// trapped key by its slot.
    fn disarm(&self) {
        *self.trap.lock().expect("never poisoned") = None;
    }

    fn a_writes(&self) -> usize {
        self.a_writes.load(Ordering::SeqCst)
    }

    /// The frames A answered `-ASK`, in arrival order.
    fn a_log(&self) -> Vec<Received> {
        self.a_log.lock().expect("never poisoned").clone()
    }

    /// From now on C refuses each connection's setup exchange as `fault` says.
    fn refuse_setup_on_c(&self, fault: SetupFault) {
        *self.c_setup_fault.lock().expect("never poisoned") = Some(fault);
    }

    fn c_setup_refusals(&self) -> usize {
        self.c_setup_refusals.load(Ordering::SeqCst)
    }

    fn log(&self, target: Target) -> Vec<Received> {
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

/// The first key `<prefix>bvn4-<what>-<i>` whose slot `on_a` says A owns (or,
/// with `on_a` false, B owns).
fn key_on(tv: &TestValkey, what: &str, on_a: bool) -> String {
    (0..)
        .map(|i| format!("{}bvn4-{what}-{i}", tv.prefix()))
        .find(|key| (fred::util::redis_keyslot(key.as_bytes()) < SPLIT) == on_a)
        .expect("keys hash to both halves of the slot space")
}

/// `key`'s value written straight to the store, past the facade: a write
/// through the facade to a trapped key would be redirected itself.
async fn store_directly(direct: &TestValkey, key: &str, value: &str) {
    direct
        .client()
        .pool
        .set::<(), _, _>(key, value, None, None, false)
        .await
        .expect("SET straight to the store must succeed");
}

/// The frames `log` holds that name `key`, each with the frame received
/// right before it on the same connection.
fn with_predecessor<'a>(
    log: &'a [Received],
    key: &str,
) -> Vec<(Option<&'a Received>, &'a Received)> {
    log.iter()
        .enumerate()
        .filter(|(_, frame)| frame.names(key))
        .map(|(at, frame)| {
            let before = log[..at]
                .iter()
                .rev()
                .find(|prior| prior.conn == frame.conn);
            (before, frame)
        })
        .collect()
}

/// Every frame naming `key` that `target` received came right behind an
/// `ASKING` on its own connection, and there was at least one: the redirect
/// was followed, with `ASKING` pipelined in front of the command.
fn assert_each_right_behind_asking(facade: &AskFacade, target: Target, key: &str, what: &str) {
    let log = facade.log(target);
    let framed = with_predecessor(&log, key);
    assert!(
        !framed.is_empty(),
        "{what}: the {target:?} node never received a command naming {key}: the -ASK was not \
         followed (stock fred re-sends it to the slot's owner). Frames it received: {log:?}"
    );
    for (before, frame) in framed {
        assert!(
            before.is_some_and(|b| b.is("ASKING") && b.args.len() == 1),
            "{what}: {frame:?} must come right behind an ASKING on its own connection \
             (connection {}); the frame before it there was {before:?}",
            frame.conn
        );
    }
}

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

fn assert_no_router_restart(metrics: &Metrics) {
    assert_eq!(
        restarts_row(metrics),
        POOLS.iter().map(|p| (*p, 0)).collect::<Vec<_>>(),
        "following an -ASK must not end any router: no restart may be counted"
    );
}

fn data_reconnects(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_reconnects_total",
        &[("pool", "data")],
    )
    .expect("valkey_reconnects_total{pool=\"data\"} is seeded") as u64
}

async fn hooked(tv: &TestValkey) -> Arc<Metrics> {
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    metrics
}

/// A `GET`'s answer, its error as text, to compare against [`value`].
fn answered(got: Result<Option<String>, fred::error::Error>) -> Result<Option<String>, String> {
    got.map_err(|e| e.to_string())
}

/// The answer of a `GET` that read `stored`.
fn value(stored: &str) -> Result<Option<String>, String> {
    Ok(Some(stored.to_owned()))
}

async fn settle<T>(what: &str, fut: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(SETTLE, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not settle within {SETTLE:?}"))
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (a), (b), (f)
// ═════════════════════════════════════════════════════════════════════════════

/// A default-budget command (fred's three write attempts) on a key the
/// slot's owner answers `-ASK` for, every time, is answered by the `-ASK`
/// target: it returns the key's value, A received exactly one write of it,
/// and B received it right behind an `ASKING` on the same connection. No
/// router is restarted.
///
/// On stock fred the `GET` goes back to A after each `ASKING`, A answers
/// `-ASK` three times, and the `GET` fails `Routing` "Max attempts reached.".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_command_answered_ask_is_answered_by_the_ask_target_right_behind_asking() {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    let key = key_on(&tv, "asked", true);
    store_directly(&direct, &key, ASKED_VALUE).await;
    facade.trap(&key, Target::B);

    let got = settle(
        "the redirected GET",
        tv.client().pool.next().get::<Option<String>, _>(&key),
    )
    .await;
    facade.disarm();

    assert_eq!(
        answered(got),
        value(ASKED_VALUE),
        "the GET must be answered through the -ASK target with the key's value"
    );
    assert_eq!(
        facade.a_writes(),
        1,
        "the slot's owner received the GET once: the -ASK was followed to its target, not \
         re-sent to the owner"
    );
    assert_eq!(
        with_predecessor(&facade.log(Target::B), &key).len(),
        1,
        "B received the GET once"
    );
    assert_each_right_behind_asking(&facade, Target::B, &key, "the redirected GET");
    assert_no_router_restart(&metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (c)
// ═════════════════════════════════════════════════════════════════════════════

/// A command with one write attempt — the budget
/// `lua::evalsha_once_with_reload` gives the commit `EVALSHA`
/// (`UPDATE_LAYER_COMMIT_MAX_ATTEMPTS`), here on a `GET` so no script is
/// involved — is answered `-ASK` with no attempt left to follow it: it fails
/// `Routing` "Max attempts reached." (transient), A received its one write,
/// and nothing — neither `ASKING` nor the command — reaches B. Unchanged from
/// stock fred: the follow is only for a command with an attempt left.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_one_attempt_command_answered_ask_still_fails_max_attempts_reached_and_sends_nothing_to_the_target(
) {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let key = key_on(&tv, "one-attempt", true);
    facade.trap(&key, Target::B);

    let once = Options {
        max_attempts: Some(UPDATE_LAYER_COMMIT_MAX_ATTEMPTS),
        ..Default::default()
    };
    let got = settle(
        "the one-attempt GET",
        tv.client()
            .pool
            .next()
            .with_options(&once)
            .get::<Option<String>, _>(&key),
    )
    .await;
    facade.disarm();

    let err = got.expect_err("a one-attempt command cannot follow the -ASK");
    assert_eq!(
        (err.kind(), err.details()),
        (&fred::error::ErrorKind::Routing, "Max attempts reached."),
        "the -ASK to a command with no attempt left fails as before; got {err:?}"
    );
    assert!(
        is_transient_fred_error(&err),
        "…and stays transient, so the caller retries it"
    );
    assert_eq!(facade.a_writes(), 1, "A received the command's one write");
    let b = facade.log(Target::B);
    assert!(
        !b.iter().any(|f| f.is("ASKING") || f.names(&key)),
        "nothing may reach B for a command with no attempt left; B received {b:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (d), (f)
// ═════════════════════════════════════════════════════════════════════════════

/// The frames B received on the connection the `GET` of `held_key` came on,
/// from that `GET` on.
fn behind_held<'a>(log: &'a [Received], held_key: &str) -> Vec<&'a Received> {
    let Some(at) = log.iter().position(|f| f.is("GET") && f.names(held_key)) else {
        return Vec::new();
    };
    let conn = log[at].conn;
    log[at..].iter().filter(|f| f.conn == conn).collect()
}

/// Whether `behind` (from [`behind_held`]) shows the redirect landed right
/// behind the held `GET`: `ASKING`, then the `GET` of `asked_key`.
fn redirect_landed_behind(behind: &[&Received], asked_key: &str) -> bool {
    behind.len() >= 3 && behind[1].is("ASKING") && behind[2].is("GET") && behind[2].names(asked_key)
}

/// A `GET` in flight on B's connection — forwarded, its reply held by B
/// behind a [`ReplyGate`] — and a second `GET` on the same pooled client that
/// A answers `-ASK` to B. The test waits, bounded, until B has received
/// `ASKING` and the redirected `GET` right behind the held one on its
/// connection, with the held `GET` still unanswered, and only then opens the
/// gate: each `GET` gets its own value. No router is restarted.
///
/// On stock fred the router reads the `ASKING` reply synchronously off B's
/// connection, so with the held `GET`'s reply behind the gate it blocks, and
/// the redirected `GET` never reaches B: the bounded wait fails. (Were the
/// replies released, it would take the held `GET`'s reply for the `ASKING`'s,
/// and the held `GET` would read a reply that was not its own.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_command_in_flight_on_the_ask_target_and_the_redirected_command_each_get_their_own_reply()
{
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    let asked = key_on(&tv, "asked", true);
    let in_flight = key_on(&tv, "in-flight", false);
    store_directly(&direct, &asked, ASKED_VALUE).await;
    store_directly(&direct, &in_flight, IN_FLIGHT_VALUE).await;
    let gate = ReplyGate::new();
    *facade.hold_on_b.lock().expect("never poisoned") = Some((in_flight.clone(), gate.clone()));
    facade.trap(&asked, Target::B);

    // One pooled client: both commands share its one connection to B.
    let client = tv.client().pool.next().clone();
    let held = {
        let (client, key) = (client.clone(), in_flight.clone());
        tokio::spawn(async move { client.get::<Option<String>, _>(&key).await })
    };
    let deadline = tokio::time::Instant::now() + SETTLE;
    while facade.held_on_b.load(Ordering::SeqCst) == 0 {
        if tokio::time::Instant::now() >= deadline {
            gate.open();
            panic!("the held GET never reached B");
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let redirected = {
        let (client, key) = (client.clone(), asked.clone());
        tokio::spawn(async move { client.get::<Option<String>, _>(&key).await })
    };
    // The event the test is about: the redirect written behind the held GET,
    // whose reply has not been released.
    let deadline = tokio::time::Instant::now() + SETTLE;
    loop {
        let b = facade.log(Target::B);
        if redirect_landed_behind(&behind_held(&b, &in_flight), &asked) {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            gate.open();
            panic!(
                "the redirect never landed behind the GET held in flight on B: no ASKING and \
                 redirected GET on its connection within {SETTLE:?}; B received {b:?}"
            );
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert!(
        !held.is_finished(),
        "precondition: the held GET is still unanswered when the redirect lands"
    );
    gate.open();
    let redirected = settle("the redirected GET", redirected)
        .await
        .expect("the redirected GET's task must not panic");
    let held = settle("the held GET", held)
        .await
        .expect("the held GET's task must not panic");
    facade.disarm();

    assert_eq!(
        answered(held),
        value(IN_FLIGHT_VALUE),
        "the GET in flight on B when the -ASK landed must get its own reply; anything else is \
         another command's reply taken off the connection out of order"
    );
    assert_eq!(
        answered(redirected),
        value(ASKED_VALUE),
        "the redirected GET must get its own reply, through B"
    );
    assert_no_router_restart(&metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (e), (f)
// ═════════════════════════════════════════════════════════════════════════════

/// An `-ASK` naming a node the client holds no connection to — C, which owns
/// no slot (a scale-out's new node) — is followed through a connection opened
/// for it: the `GET` returns the value, C received `ASKING` and the `GET`
/// back to back on one connection, and the opened connection is announced as
/// a reconnect of the data pool. No router is restarted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_ask_to_a_node_that_owns_no_slot_is_followed_through_a_connection_opened_for_it() {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    let key = key_on(&tv, "asked", true);
    store_directly(&direct, &key, ASKED_VALUE).await;
    let reconnects_before = data_reconnects(&metrics);
    facade.trap(&key, Target::C);

    let got = settle(
        "the GET redirected to C",
        tv.client().pool.next().get::<Option<String>, _>(&key),
    )
    .await;
    facade.disarm();

    assert_eq!(
        answered(got),
        value(ASKED_VALUE),
        "the GET must be answered through the connection opened to C"
    );
    assert_eq!(facade.a_writes(), 1, "A received the GET once");
    assert_each_right_behind_asking(&facade, Target::C, &key, "the GET redirected to C");
    assert_eq!(
        data_reconnects(&metrics),
        reconnects_before + 1,
        "the connection opened to C is announced once, as a cluster sync announces a \
         connection it adds"
    );
    assert!(
        facade.log(Target::B).iter().all(|f| !f.names(&key)),
        "nothing naming the key reached B: the -ASK named C"
    );
    assert_no_router_restart(&metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (e′): an ASK target that never answers stalls the client for the cap only
// ═════════════════════════════════════════════════════════════════════════════

/// An `-ASK` naming a node that accepts the connection and never answers
/// ([`SilentNode`]): the on-demand connection to it is cut at
/// [`ASK_TARGET_CONNECT_CAP`] (2 s), not at the client's own 30 s connection
/// timeout. The redirected command fails transiently — `IO`, "Failed
/// connecting to ASK target <node>" — no sooner than the cap; a command
/// issued on the same pooled client while the connect is running is answered
/// with its value; a second command redirected to the same node fails the
/// same way over a connect of its own (the first left nothing in the
/// router's map); no router is restarted.
///
/// **The bound is the client's command timeout, not a wall clock.** The test
/// asserts no upper time bound (a gating suite carries no timing assertion).
/// Its precondition is cap < command timeout (5 s) < connection timeout
/// (30 s). Without the cap, the connect runs on toward the connection
/// timeout, so both commands reach their command timeout first and fail
/// `Timeout` ("Request timed out.") instead of the `IO` error and the value
/// asserted here: the command timeout is what bounds this test, at the scale
/// of the stall the cap exists to prevent.
///
/// **Why "answered", not "answered before the redirected one fails".** The
/// connection is opened inline on the client's one router task, so while it
/// runs that client writes and reads nothing else: the concurrent command
/// (issued once the silent node has accepted, so the router is inside the
/// connect) can only be written, and its reply read, after the cap fires. The
/// contract is that this stall ends at the cap, inside the command timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_ask_to_a_node_that_never_answers_fails_the_command_at_the_cap_and_stalls_the_client_no_longer(
) {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    let asked = key_on(&tv, "asked-silent", true);
    let beside = key_on(&tv, "beside-silent", false);
    store_directly(&direct, &beside, IN_FLIGHT_VALUE).await;
    facade.trap(&asked, Target::Silent);

    // One pooled client: both commands go through its one router.
    let client = tv.client().pool.next().clone();
    let connect_timeout = client.connection_config().connection_timeout;
    let command_timeout = client.perf_config().default_command_timeout;
    assert!(
        ASK_TARGET_CONNECT_CAP < command_timeout && command_timeout < connect_timeout,
        "precondition: cap ({ASK_TARGET_CONNECT_CAP:?}) < command timeout ({command_timeout:?}) \
         < connection timeout ({connect_timeout:?}), so a connect bounded by the client's \
         connection timeout instead of the cap shows as both commands timing out"
    );

    let started = tokio::time::Instant::now();
    let redirected = {
        let (client, key) = (client.clone(), asked.clone());
        tokio::spawn(async move {
            let got = client.get::<Option<String>, _>(&key).await;
            (got, started.elapsed())
        })
    };
    // The router is inside the connect once the silent node has accepted.
    let deadline = started + SETTLE;
    while facade.silent.accepted() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the -ASK to the silent node was never followed to it"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let beside_got = settle(
        "the command beside the stalled connect",
        client.get::<Option<String>, _>(&beside),
    )
    .await;
    let (redirected_got, redirected_done) = settle("the redirected GET", redirected)
        .await
        .expect("the redirected GET's task must not panic");

    let silent = format!("127.0.0.1:{}", facade.silent.port);
    let assert_cut_at_the_cap = |got: Result<Option<String>, fred::error::Error>, what: &str| {
        let err = got.expect_err("an -ASK target that never answers cannot serve the GET");
        assert!(
            matches!(err.kind(), fred::error::ErrorKind::IO)
                && err.details().starts_with("Failed connecting to ASK target")
                && err.details().contains(&silent),
            "{what} fails as a connect to its ASK target cut short, naming it; got {err:?}"
        );
        assert!(
            is_transient_fred_error(&err),
            "…and transiently, so its caller retries it"
        );
    };
    assert_cut_at_the_cap(redirected_got, "the redirected GET");
    assert!(
        redirected_done >= ASK_TARGET_CONNECT_CAP,
        "the redirected GET fails when the connect is cut at the cap, not sooner; it took \
         {redirected_done:?}"
    );
    assert_eq!(
        answered(beside_got),
        value(IN_FLIGHT_VALUE),
        "the command beside the stalled connect must be answered once the cap ends the stall, \
         inside its own {command_timeout:?} command timeout"
    );

    // A cut-short connect leaves nothing in the router's map: the next
    // redirect to the node opens a connect of its own and is cut the same
    // way. A half-made connection left in the map would instead have the
    // `ASKING` and the `GET` written to it, and the `GET` time out.
    let again = settle(
        "the second GET redirected to the silent node",
        client.get::<Option<String>, _>(&asked),
    )
    .await;
    facade.disarm();
    assert_cut_at_the_cap(again, "the second redirected GET");
    assert_eq!(
        facade.silent.accepted(),
        2,
        "the second redirect connected anew: nothing of the first was kept"
    );
    assert_no_router_restart(&metrics);
}

/// `ask_target_connect_timeout` is the shorter of the client's connection
/// timeout and [`ASK_TARGET_CONNECT_CAP`], and the cap when the client's is
/// zero — fred's "no timeout", which a plain `min` would hand on as no bound
/// at all.
#[test]
fn ask_target_connect_timeout_is_the_shorter_of_the_client_timeout_and_the_cap() {
    assert_eq!(
        ask_target_connect_timeout(Duration::ZERO),
        ASK_TARGET_CONNECT_CAP,
        "a client with no connection timeout still gets the cap"
    );
    assert_eq!(
        ask_target_connect_timeout(Duration::from_secs(1)),
        Duration::from_secs(1),
        "a client timeout below the cap is kept"
    );
    assert_eq!(
        ask_target_connect_timeout(Duration::from_secs(30)),
        ASK_TARGET_CONNECT_CAP,
        "a client timeout above the cap is cut to it"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (h): the pipelined ASKING is never replayed to the slot's owner
// ═════════════════════════════════════════════════════════════════════════════

/// B drops its connection in place of the `ASKING` a followed redirect sends
/// it, so that `ASKING` and the redirected `GET` behind it are both in flight
/// when the connection ends, and fred replays the connection's in-flight
/// commands. The `ASKING` is written with no write attempt left, so the
/// replay discards it: A, the slot's owner, never receives an `ASKING` (one
/// there would be routed by its slot instead of following a redirect). No
/// router is restarted, and the `GET` ends answered or with a transient
/// error — its replay spends its attempts as any replay does.
///
/// The drop is on the `ASKING` itself, not on the `GET` behind it: the
/// `ASKING` is then certainly unanswered when the connection ends, where on
/// the `GET` its `+OK` might already have reached the client.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_asking_in_flight_when_the_targets_connection_drops_is_not_replayed_to_the_slots_owner()
{
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    let key = key_on(&tv, "asked-dropped", true);
    store_directly(&direct, &key, ASKED_VALUE).await;
    facade.close_on_b_asking.store(true, Ordering::SeqCst);
    facade.trap(&key, Target::B);

    let got = settle(
        "the GET whose ASK target dropped",
        tv.client().pool.next().get::<Option<String>, _>(&key),
    )
    .await;
    facade.disarm();

    assert_eq!(
        facade.closed_on_b.load(Ordering::SeqCst),
        1,
        "precondition: B dropped its connection on the followed redirect's ASKING"
    );
    assert_eq!(
        facade.a_askings.load(Ordering::SeqCst),
        0,
        "the replay must discard the ASKING, never route it by its slot to the owner"
    );
    match &got {
        Ok(found) => assert_eq!(found.as_deref(), Some(ASKED_VALUE)),
        Err(e) => assert!(
            is_transient_fred_error(e),
            "the GET may run out of attempts on its replay, but only transiently; got {e:?}"
        ),
    }
    assert_no_router_restart(&metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (i): a command its caller gave up on follows no redirect
// ═════════════════════════════════════════════════════════════════════════════

/// The per-command timeout of the `GET` in
/// [`a_command_that_timed_out_before_its_ask_arrived_writes_nothing_to_the_target`].
/// It orders nothing against the `-ASK` (a gate does); it only has to outlast
/// the `GET`'s write to A, which takes milliseconds and which the test checks
/// (`a_writes`), so a runner too slow for it fails a precondition, never
/// passes falsely.
const TIMED_OUT_GET_TIMEOUT: Duration = Duration::from_secs(1);

/// A holds its `-ASK` behind a gate ([`ProxyAction::GatedEval`]) until the
/// `GET`'s caller has received `Timeout`: fred marks the command timed out in
/// the caller's task before it returns that error, so when the gate then
/// opens and the redirect reaches the router, the `GET` has been given up
/// on, and it writes nothing, not even its `ASKING`, to B.
///
/// Ordered by events, not time. A answers in order on one connection, so a
/// command sent to A after the gate opened (the first fence) is answered only
/// after the held `-ASK`, which the router has then queued as a redirect. A
/// third command, to B (the second fence), is queued behind that redirect on
/// the client's one router and written to B after anything the redirect
/// writes there. So once it is answered, B's record is complete.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_command_that_timed_out_before_its_ask_arrived_writes_nothing_to_the_target() {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    let key = key_on(&tv, "asked-late", true);
    let fence_a = key_on(&tv, "fence-a", true);
    let fence_b = key_on(&tv, "fence-b", false);
    store_directly(&direct, &key, ASKED_VALUE).await;
    store_directly(&direct, &fence_b, IN_FLIGHT_VALUE).await;
    let client = tv.client().pool.next().clone();
    let gate = ReplyGate::new();
    *facade.ask_gate.lock().expect("never poisoned") = Some(gate.clone());
    facade.trap(&key, Target::B);

    let short = Options {
        timeout: Some(TIMED_OUT_GET_TIMEOUT),
        ..Default::default()
    };
    let timed_out = settle(
        "the GET whose -ASK is held back",
        client.with_options(&short).get::<Option<String>, _>(&key),
    )
    .await;
    // The caller has its `Timeout`, so the command is marked timed out: only
    // now may its `-ASK` reach the router.
    gate.open();
    let first_fence = settle("the fence on A", client.get::<Option<String>, _>(&fence_a)).await;
    let second_fence = settle("the fence on B", client.get::<Option<String>, _>(&fence_b)).await;
    facade.disarm();

    let err = timed_out.expect_err("the -ASK is held until the GET has timed out");
    assert!(
        matches!(err.kind(), fred::error::ErrorKind::Timeout),
        "precondition: the GET timed out before its -ASK was released; got {err:?}"
    );
    assert_eq!(
        answered(first_fence),
        Ok(None),
        "the fence on A is answered"
    );
    assert_eq!(
        answered(second_fence),
        value(IN_FLIGHT_VALUE),
        "the fence on B is answered"
    );
    assert_eq!(
        facade.a_writes(),
        1,
        "precondition: A received the GET once"
    );
    let b = facade.log(Target::B);
    assert!(
        b.iter().any(|f| f.is("GET") && f.names(&fence_b)),
        "precondition: B received the fence"
    );
    assert!(
        !b.iter().any(|f| f.is("ASKING") || f.names(&key)),
        "a command its caller gave up on must write nothing to the -ASK target; B received {b:?}"
    );
    assert_no_router_restart(&metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (g)
// ═════════════════════════════════════════════════════════════════════════════

fn state() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    for (id, parent) in [(1u64, 0u64), (2, 1)] {
        state.sections.insert(
            id,
            Section {
                section_id: id,
                parent_id: parent,
                name: format!("s{id}"),
                section_type: 1,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
    }
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);
    state
}

/// A layer holding [`state`], created through the facade, whose object header
/// (and so every key of the object: they share its hash tag) is in a slot A
/// owns.
async fn layer_on_a(tv: &TestValkey) -> u64 {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bvn4-ask-read", "/bvn4-ask-read", "bvn4-user")
        .await
        .expect("session creation must succeed");
    let layers = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    for i in 0..MAX_LAYERS_PER_SEARCH {
        let created = layers
            .create_layer(
                &session_id,
                &format!("layer-{i}"),
                "bvn4-user",
                &state(),
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed");
        let header = tv.key_schema().obj_header(&created.to_string());
        if fred::util::redis_keyslot(header.as_bytes()) < SPLIT {
            return created;
        }
    }
    panic!("no layer in {MAX_LAYERS_PER_SEARCH} landed in a slot A owns")
}

/// End to end: the structural `UpdateLayer` read
/// (`read_object_snapshot_slim(StructuralUpdate)`) of an object whose slot A
/// owns, with A answering `-ASK` to every command naming its `obj:dels` — as
/// a migrating slot's source answers for a key it does not hold — answers the
/// atomic read's compact projection. Its round-1 `SCARD obj:dels` (a plain,
/// default-budget command) is answered by B right behind an `ASKING`.
///
/// On stock fred that `SCARD` fails "Max attempts reached." and the read
/// fails with it, before any window script runs: what the b.gvf cluster test
/// saw as zero fallbacks during a real slot migration.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_structural_read_whose_deleted_sections_count_is_answered_ask_answers_the_objects_state()
{
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(4).await;
    let oid = layer_on_a(&tv).await;
    let oid_str = oid.to_string();
    let atomic = settle(
        "the atomic read",
        read_atomic(tv.client(), tv.key_schema(), oid),
    )
    .await;
    let dels = tv.key_schema().obj_deleted_sections_index(&oid_str);
    facade.trap(&dels, Target::B);

    let read = settle(
        "the structural read",
        read_object_snapshot_slim(
            tv.client(),
            tv.key_schema(),
            &oid_str,
            SlimSnapshotReadConsumer::StructuralUpdate,
        ),
    )
    .await;
    facade.disarm();

    match read {
        Ok(SnapshotReadOf::Snapshot(read)) => assert_eq!(
            read.state,
            compact_snapshot(&atomic.state),
            "the structural read answers the atomic read's compact projection"
        ),
        other => panic!(
            "the structural read must answer with the object's state; got {:?}",
            other.map(|_| ())
        ),
    }
    assert!(
        facade.a_writes() >= 1,
        "precondition: the read sent A a command naming {dels}"
    );
    assert!(
        facade
            .log(Target::B)
            .iter()
            .any(|f| f.is("SCARD") && f.names(&dels)),
        "the read's SCARD of {dels} was answered by B"
    );
    assert_each_right_behind_asking(&facade, Target::B, &dels, "the read's commands on obj:dels");
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (j): a script answered -ASK is never followed
// ═════════════════════════════════════════════════════════════════════════════

/// The script commands, by name: what a frame that runs a script is called on
/// the wire, whatever form the client built it in.
const SCRIPT_NAMES: [&str; 6] = [
    "EVAL",
    "EVALSHA",
    "EVAL_RO",
    "EVALSHA_RO",
    "FCALL",
    "FCALL_RO",
];

impl Received {
    fn is_script(&self) -> bool {
        SCRIPT_NAMES.iter().any(|name| self.is(name))
    }
}

/// The script frames in `log`.
fn script_frames(log: &[Received]) -> Vec<&Received> {
    log.iter().filter(|f| f.is_script()).collect()
}

/// The script `-ASK` rule, end to end through the structural `UpdateLayer`
/// read (`read_object_snapshot_slim(StructuralUpdate)`), whose round-2
/// window script is a custom `EVALSHA` with fred's default budget of three
/// write attempts (`lua::evalsha_reload_rebuilding`, `CommandKind::_Custom`).
/// A answers `-ASK` to every frame naming the object's header, the window
/// script's one declared key: the script is not followed, so the read fails
/// `Aborted` — retryable — on `Routing` "Max attempts reached.", A received
/// the script once, and no script reached B. The read's plain commands on
/// the header still follow to B, each right behind an `ASKING`.
///
/// Why a script is never followed: the source routes it by its declared keys
/// only, and on the target it inherits `ASKING`, so an undeclared key still
/// on the source reads as absent there — a partial answer. Here B relays to
/// the whole store, so a followed window script would answer in full: with
/// the rule removed the read answers the object's state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_structural_reads_window_script_answered_ask_is_not_followed_and_the_read_fails_retryable(
) {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(4).await;
    let oid = layer_on_a(&tv).await;
    let oid_str = oid.to_string();
    let header = tv.key_schema().obj_header(&oid_str);
    // B's frames from here on: the layer search ran scripts on B's own slots.
    let b_from = facade.log(Target::B).len();
    facade.trap(&header, Target::B);

    let read = settle(
        "the structural read",
        read_object_snapshot_slim(
            tv.client(),
            tv.key_schema(),
            &oid_str,
            SlimSnapshotReadConsumer::StructuralUpdate,
        ),
    )
    .await;
    facade.disarm();

    let a_scripts: Vec<Received> = script_frames(&facade.a_log())
        .into_iter()
        .cloned()
        .collect();
    assert!(
        !a_scripts.is_empty() && a_scripts.iter().all(|f| f.is("EVALSHA")),
        "precondition: the read sent A its window script's EVALSHA, and no other script; got \
         {a_scripts:?}"
    );
    match read {
        Err(ServiceError::Aborted(msg)) => assert!(
            msg.contains("Max attempts reached.") && msg.contains("EVALSHA fanout_window_read"),
            "the read fails retryable on the window script's unfollowed -ASK; got Aborted({msg:?})"
        ),
        Err(other) => panic!(
            "the read must fail Aborted (retryable) on the window script's unfollowed -ASK; got \
             {other:?}"
        ),
        Ok(_) => panic!(
            "the read must fail: its window script was answered -ASK and must not be followed, \
             yet the read answered"
        ),
    }
    assert_eq!(
        a_scripts.len(),
        1,
        "the window script reached A once: answered -ASK, it fails at once instead of being \
         re-sent or followed; A received {a_scripts:?}"
    );
    let b = facade.log(Target::B)[b_from..].to_vec();
    assert!(
        script_frames(&b).is_empty(),
        "no script may reach the -ASK target; B received {:?}",
        script_frames(&b)
    );
    assert_each_right_behind_asking(
        &facade,
        Target::B,
        &header,
        "the read's plain commands on the header",
    );
}

/// The same rule through the atomic snapshot read (`read_object_snapshot`,
/// catch-up), whose script is sent by fred's own `EVALSHA`
/// (`Script::evalsha_with_reload`, `CommandKind::EvalSha`) with the default
/// budget: A answers `-ASK` to the script, which names the object's header,
/// and the read fails `Aborted` on `Routing` "Max attempts reached.", with
/// the script sent to A once and nothing — no `ASKING`, no command naming
/// the header — sent to B.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_atomic_snapshot_reads_script_answered_ask_is_not_followed_and_the_read_fails_retryable()
{
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let oid = layer_on_a(&tv).await;
    let header = tv.key_schema().obj_header(&oid.to_string());
    facade.trap(&header, Target::B);

    let err = settle(
        "the atomic read",
        read_err(tv.client(), tv.key_schema(), oid),
    )
    .await;
    facade.disarm();

    let a = facade.a_log();
    assert!(
        a.len() == 1 && a[0].is("EVALSHA"),
        "A received the read's script once, and nothing else naming the header; got {a:?}"
    );
    match &err {
        ServiceError::Aborted(msg) => assert!(
            msg.contains("Max attempts reached."),
            "the read fails retryable on the script's unfollowed -ASK; got Aborted({msg:?})"
        ),
        other => panic!(
            "the read must fail Aborted (retryable) on the script's unfollowed -ASK; got {other:?}"
        ),
    }
    let b = facade.log(Target::B);
    assert!(
        !b.iter().any(|f| f.is("ASKING") || f.names(&header)),
        "nothing may reach the -ASK target for a script; B received {b:?}"
    );
}

/// A script that writes `KEYS[1]` (`SET KEYS[1] ARGV[1]`): a followed one
/// would leave the key written in the store.
const SET_SCRIPT: &str = "return redis.call('SET', KEYS[1], ARGV[1])";

/// A script that reads `KEYS[1]`, for the read-only forms.
const GET_SCRIPT: &str = "return redis.call('GET', KEYS[1])";

/// The value [`SET_SCRIPT`] writes.
const SCRIPT_MARK: &str = "bvn4-script-mark";

/// One way a client builds a script command.
#[derive(Clone, Copy, Debug)]
enum ScriptForm {
    /// fred's `eval` (`CommandKind::Eval`).
    Eval,
    /// fred's `evalsha` (`CommandKind::EvalSha`).
    Evalsha,
    /// fred's `fcall` (`CommandKind::FCall`).
    Fcall,
    /// fred's `fcall_ro` (`CommandKind::FCallRO`).
    FcallRo,
    /// A custom command named `evalsha`, lower case.
    CustomEvalshaLower,
    /// A custom command named `EVAL_RO`, which has no fred kind.
    CustomEvalRo,
    /// A custom command named `evalsha_RO`, mixed case, which has no fred kind.
    CustomEvalshaRoMixed,
    /// A custom command named `Fcall_Ro`, mixed case.
    CustomFcallRoMixed,
}

impl ScriptForm {
    const ALL: [ScriptForm; 8] = [
        ScriptForm::Eval,
        ScriptForm::Evalsha,
        ScriptForm::Fcall,
        ScriptForm::FcallRo,
        ScriptForm::CustomEvalshaLower,
        ScriptForm::CustomEvalRo,
        ScriptForm::CustomEvalshaRoMixed,
        ScriptForm::CustomFcallRoMixed,
    ];

    /// Whether the script this form runs writes [`SCRIPT_MARK`] to its key.
    fn writes(self) -> bool {
        matches!(
            self,
            ScriptForm::Eval | ScriptForm::Evalsha | ScriptForm::CustomEvalshaLower
        )
    }

    /// Send this form, with `key` its one declared key.
    async fn send(
        self,
        client: &fred::clients::Client,
        key: &str,
    ) -> Result<fred::types::Value, fred::error::Error> {
        let custom = |name: &'static str, args: Vec<fred::types::Value>| {
            let command = CustomCommand::new_static(name, ClusterHash::from(key), false);
            client.custom::<fred::types::Value, _>(command, args)
        };
        let keyed = |script: String| -> Vec<fred::types::Value> {
            vec![script.into(), 1_i64.into(), key.into(), SCRIPT_MARK.into()]
        };
        match self {
            ScriptForm::Eval => client.eval(SET_SCRIPT, vec![key], vec![SCRIPT_MARK]).await,
            ScriptForm::Evalsha => {
                client
                    .evalsha(
                        fred::util::sha1_hash(SET_SCRIPT),
                        vec![key],
                        vec![SCRIPT_MARK],
                    )
                    .await
            }
            ScriptForm::Fcall => client.fcall("bvn4_fn", vec![key], vec![SCRIPT_MARK]).await,
            ScriptForm::FcallRo => {
                client
                    .fcall_ro("bvn4_fn", vec![key], vec![SCRIPT_MARK])
                    .await
            }
            ScriptForm::CustomEvalshaLower => {
                custom("evalsha", keyed(fred::util::sha1_hash(SET_SCRIPT))).await
            }
            ScriptForm::CustomEvalRo => custom("EVAL_RO", keyed(GET_SCRIPT.to_owned())).await,
            ScriptForm::CustomEvalshaRoMixed => {
                custom("evalsha_RO", keyed(fred::util::sha1_hash(GET_SCRIPT))).await
            }
            ScriptForm::CustomFcallRoMixed => custom("Fcall_Ro", keyed("bvn4_fn".to_owned())).await,
        }
    }
}

/// Every form a script command takes — fred's `EVAL`, `EVALSHA`, `FCALL` and
/// `FCALL_RO` kinds, and custom commands named after a script command in any
/// case, `EVAL_RO` and `EVALSHA_RO` among them — answered `-ASK` with
/// fred's default budget fails `Routing` "Max attempts reached.", transient,
/// at A's first `-ASK`: A received it once, nothing naming its key and no
/// `ASKING` reached B, and a script that writes its key left it unwritten.
/// A custom command that is not a script (`GET`) still follows the `-ASK`,
/// so the rule matches script names only.
///
/// The `SET` script is loaded on the store first, so a followed `EVALSHA`
/// would run there and write the key. No function is loaded: a followed
/// `FCALL` form would be refused at the store, so B's record is what shows
/// it was followed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_script_command_form_answered_ask_fails_at_once_and_a_custom_plain_command_still_follows(
) {
    let facade = AskFacade::start().await;
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    for script in [SET_SCRIPT, GET_SCRIPT] {
        let _: String = direct
            .client()
            .pool
            .next()
            .script_load(script)
            .await
            .expect("SCRIPT LOAD straight to the store must succeed");
    }
    let client = tv.client().pool.next().clone();

    // Every form is run and judged before anything is asserted, so a failure
    // names every form that broke the rule, not only the first. One bound
    // covers them all.
    let mut broken: Vec<(ScriptForm, Vec<String>)> = Vec::new();
    settle("the script forms", async {
        for (i, form) in ScriptForm::ALL.into_iter().enumerate() {
            let key = key_on(&tv, &format!("script-{i}"), true);
            facade.trap(&key, Target::B);
            let got = form.send(&client, &key).await;
            facade.disarm();

            let mut wrong = Vec::new();
            match &got {
                Err(err)
                    if (err.kind(), err.details())
                        == (&fred::error::ErrorKind::Routing, "Max attempts reached.") =>
                {
                    if !is_transient_fred_error(err) {
                        wrong.push(format!("{err:?} is not transient, so no caller retries it"));
                    }
                }
                other => wrong.push(format!(
                "answered {other:?}, not Routing \"Max attempts reached.\" as before the follow \
                 existed"
            )),
            }
            let at_a = facade.a_log().iter().filter(|f| f.names(&key)).count();
            if at_a != 1 {
                wrong.push(format!("A received it {at_a} times, not once"));
            }
            let at_b: Vec<Received> = facade
                .log(Target::B)
                .into_iter()
                .filter(|f| f.names(&key))
                .collect();
            if !at_b.is_empty() {
                wrong.push(format!("it reached the -ASK target: B received {at_b:?}"));
            }
            if form.writes() {
                let stored: Option<String> = direct
                    .client()
                    .pool
                    .next()
                    .get(&key)
                    .await
                    .expect("GET straight to the store must succeed");
                if stored.is_some() {
                    wrong.push(format!("its script wrote the key: {stored:?}"));
                }
            }
            if !wrong.is_empty() {
                broken.push((form, wrong));
            }
        }
    })
    .await;
    assert!(
        broken.is_empty(),
        "a script answered -ASK must fail at once, sending nothing to the -ASK target and \
         writing nothing; these forms did not: {broken:#?}"
    );
    let b = facade.log(Target::B);
    assert!(
        !b.iter().any(|f| f.is("ASKING")),
        "no ASKING may be sent for any script; B received {b:?}"
    );

    // The control: a custom command that is not a script follows.
    let key = key_on(&tv, "custom-plain", true);
    store_directly(&direct, &key, ASKED_VALUE).await;
    facade.trap(&key, Target::B);
    let got = settle(
        "the custom GET",
        client.custom::<Option<String>, _>(
            CustomCommand::new_static("GET", ClusterHash::from(key.as_str()), false),
            vec![key.clone()],
        ),
    )
    .await;
    facade.disarm();
    assert_eq!(
        answered(got),
        value(ASKED_VALUE),
        "a custom command that is not a script follows the -ASK and is answered by B"
    );
    assert_each_right_behind_asking(&facade, Target::B, &key, "the custom GET");
    assert_no_router_restart(&metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 (k): an ASK target that refuses its connection's setup
// ═════════════════════════════════════════════════════════════════════════════

/// What a redirected `GET` met when C refused its connection's setup.
struct RefusedFollow {
    /// The `GET`'s error.
    err: fred::error::Error,
    /// Every error event the client broadcast from the `GET` until a later
    /// command was answered: complete, because the router writes that
    /// command only after the step that followed the `-ASK` returned.
    errors: Vec<(fred::error::Error, Option<Server>)>,
    /// Reconnect events broadcast over the same span.
    reconnects: usize,
    metrics: Arc<Metrics>,
    /// `127.0.0.1:<C's port>`.
    target: String,
    /// The client and the direct fixture, held until the test ends so that
    /// the error hook keeps running and the keys are cleaned up after it.
    _tv: TestValkey,
    _direct: TestValkey,
}

/// A `GET` on a key A answers `-ASK` to C for, with C refusing the setup of
/// the connection opened to it as `fault` says; then a fence `GET` to B on
/// the same pooled client.
async fn follow_into_refused_setup(facade: &AskFacade, fault: SetupFault) -> RefusedFollow {
    let tv = facade.test_valkey(1).await;
    let direct = TestValkey::setup().await;
    let metrics = hooked(&tv).await;
    let key = key_on(&tv, "asked-refused", true);
    let fence = key_on(&tv, "fence-refused", false);
    store_directly(&direct, &key, ASKED_VALUE).await;
    store_directly(&direct, &fence, IN_FLIGHT_VALUE).await;
    let client = tv.client().pool.next().clone();
    let mut error_rx = client.error_rx();
    let mut reconnect_rx = client.reconnect_rx();
    facade.refuse_setup_on_c(fault);
    facade.trap(&key, Target::C);

    let got = settle(
        "the GET redirected to C",
        client.get::<Option<String>, _>(&key),
    )
    .await;
    let fenced = settle("the fence on B", client.get::<Option<String>, _>(&fence)).await;
    facade.disarm();

    assert_eq!(
        answered(fenced),
        value(IN_FLIGHT_VALUE),
        "precondition: the fence on B is answered, so the client kept running"
    );
    assert_eq!(
        facade.c_setup_refusals(),
        1,
        "precondition: C refused the setup of exactly one connection, the one opened to follow \
         the -ASK"
    );
    assert_eq!(facade.a_writes(), 1, "A received the GET once");
    let c = facade.log(Target::C);
    assert!(
        !c.iter().any(|f| f.is("ASKING") || f.names(&key)),
        "nothing of the GET may reach C, whose connection never opened; C received {c:?}"
    );
    let mut errors = Vec::new();
    while let Ok(event) = error_rx.try_recv() {
        errors.push(event);
    }
    let mut reconnects = 0;
    while reconnect_rx.try_recv().is_ok() {
        reconnects += 1;
    }
    RefusedFollow {
        err: got.expect_err("C refused its connection's setup, so it cannot answer the GET"),
        errors,
        reconnects,
        metrics,
        target: format!("127.0.0.1:{}", facade.port_c),
        _tv: tv,
        _direct: direct,
    }
}

/// `live_edit_valkey_errors_total{pool="data", kind}`, absent read as 0.
fn data_errors(metrics: &Metrics, kind: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_errors_total",
        &[("pool", "data"), ("kind", kind)],
    )
    .unwrap_or(0.0) as u64
}

impl RefusedFollow {
    /// The connect failure was broadcast once, naming C, with `kind` — its
    /// own kind, whatever the `GET` was answered with — and counted once on
    /// `live_edit_valkey_errors_total{pool="data", kind}`. No reconnect was
    /// announced: no connection was opened.
    async fn assert_broadcast_once_as(&self, kind: fred::error::ErrorKind) {
        let label = kind.to_str();
        assert_eq!(
            self.errors
                .iter()
                .map(|(e, server)| (e.kind().clone(), server.as_ref().map(|s| s.to_string())))
                .collect::<Vec<_>>(),
            vec![(kind.clone(), Some(self.target.clone()))],
            "the failed connect is broadcast once, naming C, with its own kind; got {:?}",
            self.errors
        );
        assert_eq!(
            self.reconnects, 0,
            "a connect that failed opened nothing, so no reconnect is announced"
        );
        // The hook counts the event on a task of its own: wait for it,
        // bounded, then check it counted it once.
        let deadline = tokio::time::Instant::now() + SETTLE;
        while data_errors(&self.metrics, label) == 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "live_edit_valkey_errors_total{{pool=\"data\", kind={label:?}}} was never ticked"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(
            data_errors(&self.metrics, label),
            1,
            "live_edit_valkey_errors_total{{pool=\"data\", kind={label:?}}} ticks once"
        );
    }
}

/// C answers the setup's `PING` `-NOAUTH`: a node that requires a password
/// the client does not have. A credential fault keeps its own kind, so the
/// redirected `GET` fails `Auth`, its text naming C and the
/// "Authentication Error"; the service's classifier treats it as permanent
/// and its gRPC mapping answers `Internal`; and the failure is counted once
/// on `live_edit_valkey_errors_total{pool="data", kind="Authentication Error"}`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_ask_target_refusing_noauth_fails_the_command_auth_naming_it_and_the_service_answers_internal(
) {
    let facade = AskFacade::start().await;
    let refused = follow_into_refused_setup(&facade, SetupFault::NoAuth).await;

    let err = &refused.err;
    assert!(
        matches!(err.kind(), fred::error::ErrorKind::Auth)
            && err.details().starts_with(&format!(
                "Failed connecting to ASK target {}: ",
                refused.target
            ))
            && err
                .details()
                .contains(&format!("Authentication Error: {NOAUTH_REPLY}")),
        "a credential refusal from the ASK target keeps its kind and names the target; got \
         {err:?}"
    );
    assert!(
        !is_transient_fred_error(err),
        "…and is not transient: retrying cannot fix a credential"
    );
    let mapped = map_fred_err_for_grpc(
        FredErrOrigin::metrics(
            None,
            POOL_LABEL_DATA,
            live_edit::storage::store_oom::StoreOomSignal::unwired(),
        ),
        "GET",
        err.clone(),
    );
    assert!(
        matches!(mapped, ServiceError::Internal(_)),
        "the service answers a credential refusal Internal; got {mapped:?}"
    );
    assert_eq!(tonic::Status::from(mapped).code(), tonic::Code::Internal);
    refused
        .assert_broadcast_once_as(fred::error::ErrorKind::Auth)
        .await;
}

/// The answer the two transient setup refusals share: the redirected `GET`
/// fails `IO`, its text naming C and the refusal (`refusal`), which the
/// service's classifier treats as transient and its gRPC mapping answers
/// `Unavailable`.
fn assert_answered_io_transient(refused: &RefusedFollow, refusal: &str) {
    let err = &refused.err;
    assert!(
        matches!(err.kind(), fred::error::ErrorKind::IO)
            && err.details().starts_with(&format!(
                "Failed connecting to ASK target {}: ",
                refused.target
            ))
            && err.details().contains(refusal),
        "a node not yet ready to serve answers the redirected command IO, naming the target \
         and the refusal {refusal:?}; got {err:?}"
    );
    assert!(
        is_transient_fred_error(err),
        "…transiently: retrying is what fixes a node that is not ready yet"
    );
    let mapped = map_fred_err_for_grpc(
        FredErrOrigin::metrics(
            None,
            POOL_LABEL_DATA,
            live_edit::storage::store_oom::StoreOomSignal::unwired(),
        ),
        "GET",
        err.clone(),
    );
    assert!(
        matches!(mapped, ServiceError::Unavailable(_)),
        "the service answers it Unavailable (retryable); got {mapped:?}"
    );
    assert_eq!(tonic::Status::from(mapped).code(), tonic::Code::Unavailable);
}

/// C answers the setup's `CLUSTER INFO` `cluster_state:fail`, as a node does
/// while it joins the cluster; fred's setup fails that connection `Protocol`
/// "Invalid or missing cluster state.". The redirected `GET` fails `IO`,
/// transient — a kind-preserving answer would make it permanent — while the
/// broadcast keeps `Protocol`, counted once on
/// `live_edit_valkey_errors_total{pool="data", kind="Protocol Error"}`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_ask_target_reporting_cluster_state_fail_fails_the_command_io_and_the_service_retries() {
    let facade = AskFacade::start().await;
    let refused = follow_into_refused_setup(&facade, SetupFault::ClusterStateFail).await;

    assert_answered_io_transient(
        &refused,
        &format!("Protocol Error: {CLUSTER_STATE_INVALID}"),
    );
    refused
        .assert_broadcast_once_as(fred::error::ErrorKind::Protocol)
        .await;
}

/// C answers the setup's `PING` `-LOADING`, as a node does while it replays
/// its dataset; fred's setup fails that connection `Unknown`. The redirected
/// `GET` fails `IO`, transient, while the broadcast keeps `Unknown`, counted
/// once on `live_edit_valkey_errors_total{pool="data", kind="Unknown Error"}`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_ask_target_loading_its_dataset_fails_the_command_io_and_the_service_retries() {
    let facade = AskFacade::start().await;
    let refused = follow_into_refused_setup(&facade, SetupFault::Loading).await;

    assert_answered_io_transient(&refused, &format!("Unknown Error: {LOADING_REPLY}"));
    refused
        .assert_broadcast_once_as(fred::error::ErrorKind::Unknown)
        .await;
}
