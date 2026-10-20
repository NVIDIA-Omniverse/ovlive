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

//! b.1qo — a command, pipeline or transaction handed to fred is never written
//! after its caller gave up on it: its timeout fired, fred answered it with an
//! error, or its future was dropped (the vendored fred's change 9,
//! `vendor/fred-10.1.0/src/live_edit/abandon.rs`). A pipeline cut during the
//! router step that writes it, or a fred `Transaction` cut partway, never
//! leaves its `MULTI` open.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`watched_txn_queued_behind_a_stalled_reconnect_is_never_written_after_its_timeout`] | T1: the service's `WatchedTxn` pipeline, queued in fred's channel behind a reconnect the store refuses, times out in doubt; once the store is back, no frame of it is written and its write never lands |
//! | [`create_layer_object_write_cut_mid_transaction_is_never_replayed_after_its_timeout`] | T2: `CreateLayer`'s object write (a fred `Transaction`) loses its connection after its `MULTI`; the create answers `UNAVAILABLE` while fred's one whole-transaction replay waits behind the refused reconnect, and the replay is never written: no `EXEC`, no orphan object |
//! | [`single_command_dropped_while_queued_behind_a_reconnect_is_never_written`] | T3: a single `SET` whose future is dropped (an outer `tokio::time::timeout` shorter than the command timeout) is never written once the store is back |
//! | [`pipeline_whose_caller_is_aborted_while_queued_behind_a_reconnect_is_never_written`] | T3: the same for a pipeline whose awaiting task is aborted |
//! | [`pipeline_lost_with_its_connection_is_not_replayed_after_its_timeout`] | T4: a data-pool pipeline (commands with write attempts left) lost with its connection waits in the retry buffer; after its caller's timeout the replay is skipped |
//! | [`transaction_cut_after_its_multi_is_closed_without_its_exec`] / [`transaction_cut_before_a_queued_command_is_closed_without_writing_it`] | T5: a fred `Transaction` whose write stalls after its `MULTI` and whose caller times out is never sent its `EXEC` — nor, when the stall falls before another queued command, that command: the connection is closed (the store discards the `MULTI`) and replaced, counted once as a `Canceled` txn error and once as a txn reconnect |
//! | [`watched_pipeline_cut_after_its_multi_is_discarded_and_leaves_no_open_multi`] | T6: the same cut on a `WatchedTxn` pipeline: fred writes a `DISCARD` instead of the `EXEC`, so the `WatchGuard`'s `UNWATCH` is acknowledged and the next checkout's probe finds no leaked `MULTI` |
//!
//! What none of these can stop, and none asserts against: a frame fred had
//! already handed to the socket before its caller gave up still reaches the
//! store (the residual; `tests/integration/storage/txn_timeout_test.rs` pins
//! that an `EXEC` written before the timer fired still lands). That a frame
//! fred had only fed, not yet flushed, before its caller gave up is sent
//! before a cluster client's reconnect, cluster sync or `-ASK`-target connect
//! rather than after it, and before the router waits on another connection's
//! write or flush rather than after it (so it reaches its socket within one
//! liveness bound however many of the client's connections stall), is pinned
//! in `tests/integration/storage/fed_frame_flush_test.rs`.
//!
//! # The harness
//!
//! [`GatedProxy`], a TCP proxy in front of the test store, carries every
//! connection of one `ValkeyClient` the test owns, and only that client's:
//!
//! - a rule cuts the connection that carries one frame — a marker `GET` the
//!   test sends on the client under test, or the first frame of the batch
//!   under test — forwarding nothing from it, and closes the **accept gate**
//!   at the same moment. With the gate closed every new connection is
//!   accepted and closed at once, so fred's reconnect keeps failing fast and
//!   whatever the client queues meanwhile waits in its router;
//! - a rule can **pause** a connection right after forwarding one frame (a
//!   `MULTI`): the proxy stops reading from the client until the test
//!   resumes it. Its listener's small receive buffer limits what the kernel
//!   absorbs meanwhile, so the client's write of the [`BIG_VALUE_BYTES`]
//!   value after the `MULTI` blocks and the caller's timer fires mid-batch;
//! - it **logs** every frame it forwards, with the index of the connection it
//!   came on (connections numbered from 0 in the order the proxy relayed
//!   them). Arguments are logged truncated to [`LOGGED_ARG_BYTES`].
//!
//! Every frame a test asserts on is named by a key only that test writes, or
//! is a `MULTI` or `EXEC` (which nothing else on the client sends), or a
//! `DISCARD` after the batch's `MULTI` (the checkout probe's own `DISCARD`
//! comes before it), so the log tells it apart from the client's own traffic.
//!
//! # Barrier, not sleeps
//!
//! fred's router reads its command channel in order. After the gate reopens,
//! a `GET` of a fence key sent on the same fred client is answered only once
//! the router has processed everything queued before it, the abandoned batch
//! included ([`fence`]): from then on the log and the store show whether the
//! batch was written. When a fred `Transaction` cut partway closed its
//! connection, the router defers its reconnect, and a fence routed ahead of
//! that reconnect fails fred's routing miss ("Failed to route command."); the
//! fence is re-sent then, and only then.
//!
//! # Clients
//!
//! Command timeout `T` = [`COMMAND_TIMEOUT`], reconnect backoff capped at
//! [`RECONNECT_MAX_DELAY_MS`], a data pool of one member and a `txn` pool of
//! one slot, so the connection a test cuts is the one its batch rides. The
//! fast unresponsive verdict is moved out of reach ([`liveness`]): a paused
//! connection holds its replies for about `T`, and no test here is about a
//! verdict closing one.
//!
//! # Wait budget
//!
//! Every wait in a test runs under that test's one deadline,
//! [`SCENARIO_BUDGET`] (60 s) from its first line ([`by`], [`until`]). A
//! healthy run waits `T` once (the caller's own timeout, which every test
//! stages on purpose; T3's two tests wait at most [`DROP_AFTER`] instead) plus
//! a few reconnect delays of at most [`RECONNECT_MAX_DELAY_MS`]; T5 and T6
//! also move [`BIG_VALUE_BYTES`] through the proxy once. After the deadline
//! comes only `TestValkey`'s drop: its purge, each round trip bounded at 5 s,
//! and its client's quit, bounded at 10 s. So each test's worst case is
//! 60 + 5 + 10 = 75 s plus the purge's answered round trips, inside nextest's
//! 120 s hang-killer.

use std::future::Future;
use std::net::SocketAddr;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::BytesMut;
use fred::clients::Client;
use fred::error::ErrorKind;
use fred::interfaces::{ClientLike, KeysInterface, TransactionInterface};
use fred::types::config::Options;
use fred::types::{ClusterHash, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpSocket, TcpStream};
use tokio::time::Instant;

use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::{
    is_router_no_connection_error, with_liveness_overrides_for_test, LivenessOverrides,
    ValkeyClient,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{
    connect_test_client, exec_watched_writes_for_test, TestValkey, WatchedExecForTest,
};
use live_edit::storage::watch_guard::WatchGuard;

use super::snapshot_read_support::{test_valkey_addr, test_valkey_db};

/// The command timeout `T` of every client under test.
const COMMAND_TIMEOUT: Duration = Duration::from_millis(1_500);

/// The reconnect backoff's cap: fred retries a refused reconnect at least
/// this often, so a reopened gate is reached quickly.
const RECONNECT_MAX_DELAY_MS: u32 = 200;

/// T3's outer timeout around a single `SET`: far under [`COMMAND_TIMEOUT`],
/// so the call is dropped before fred's own timer could set anything.
const DROP_AFTER: Duration = Duration::from_millis(300);

/// The large value T5 and T6 write after their `MULTI`, so the write blocks
/// while the proxy reads nothing: far above what the kernel's socket buffers
/// hold (the proxy's receive buffer is [`PROXY_RECV_BUFFER`], and a send
/// buffer's ceiling is a few MiB on common kernels). A kernel that took it
/// all would let fred write the `EXEC` before the caller's timer fired, and
/// the test would fail, not pass.
const BIG_VALUE_BYTES: usize = 64 << 20;

/// The receive buffer of every connection the proxy accepts (set on its
/// listener, so the kernel does not auto-tune it).
const PROXY_RECV_BUFFER: u32 = 64 << 10;

/// How much of each argument the proxy's frame log keeps.
const LOGGED_ARG_BYTES: usize = 256;

/// A fence `GET`'s own command timeout: generous, so a slow reconnect is not
/// read as a failed barrier.
const FENCE_TIMEOUT: Duration = Duration::from_secs(10);

/// Each test's one deadline; see the module's *Wait budget*.
const SCENARIO_BUDGET: Duration = Duration::from_secs(60);

/// Interval between probes of a polled condition.
const POLL: Duration = Duration::from_millis(10);

/// The layer T2 creates.
const LAYER: &str = "abandoned-write-layer";

/// The user T2 creates its session and layer as.
const USER: &str = "abandoned-write-user";

// ═════════════════════════════════════════════════════════════════════════════
// Deadlines
// ═════════════════════════════════════════════════════════════════════════════

/// The test's one deadline: [`SCENARIO_BUDGET`] from now.
fn test_deadline() -> Instant {
    Instant::now() + SCENARIO_BUDGET
}

/// `fut` under the test's one `deadline`, failing naming `what` past it.
async fn by<T>(deadline: Instant, what: &str, fut: impl Future<Output = T>) -> T {
    tokio::time::timeout_at(deadline, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not finish within the test's {SCENARIO_BUDGET:?}"))
}

/// Poll `holds` every [`POLL`] until it is true, by `deadline`.
async fn until(deadline: Instant, what: &str, mut holds: impl FnMut() -> bool) {
    while !holds() {
        assert!(
            Instant::now() < deadline,
            "{what} did not happen within the test's {SCENARIO_BUDGET:?}"
        );
        tokio::time::sleep(POLL).await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// GatedProxy
// ═════════════════════════════════════════════════════════════════════════════

/// What the proxy does with a frame its rule matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Close the accept gate, forward nothing more from this connection and
    /// close it (the matched frame included): the client's connection is
    /// lost with the frame unsent, and its reconnects are refused until
    /// [`GatedProxy::open_gate`].
    Cut,
    /// Forward the frame, then read nothing more from this client until
    /// [`GatedProxy::resume`]. The store's replies still flow back.
    PauseAfter,
}

/// Decides, per client→server frame, what the proxy does with it: the
/// connection's index and the frame's arguments (`args[0]` the command name
/// as sent), each truncated to [`LOGGED_ARG_BYTES`].
type Rule = Arc<dyn Fn(usize, &[Vec<u8>]) -> Option<Action> + Send + Sync>;

/// One frame the proxy forwarded to the store.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Forwarded {
    /// The connection it came on.
    conn: usize,
    /// Its arguments, lossily decoded and truncated to [`LOGGED_ARG_BYTES`].
    args: Vec<String>,
}

impl Forwarded {
    /// The command name, upper-cased.
    fn name(&self) -> String {
        self.args
            .first()
            .map(|a| a.to_ascii_uppercase())
            .unwrap_or_default()
    }

    /// Whether any argument contains `needle`.
    fn names(&self, needle: &str) -> bool {
        self.args.iter().any(|a| a.contains(needle))
    }
}

/// State every relayed connection shares with the test.
struct ProxyState {
    rule: Rule,
    /// The accept gate: while closed, a new connection is accepted and closed
    /// at once.
    gate_open: AtomicBool,
    /// Connections relayed so far: the next one's index.
    relayed: AtomicUsize,
    /// Connections accepted and closed at once while the gate was closed.
    refused: AtomicUsize,
    /// Frames the rule matched.
    hits: AtomicUsize,
    /// Set by a [`Action::PauseAfter`] match, cleared by
    /// [`GatedProxy::resume`].
    paused: tokio::sync::watch::Sender<bool>,
    forwarded: Mutex<Vec<Forwarded>>,
    /// Bytes read from each client connection, by connection index: what the
    /// client handed the proxy, whole frames or not.
    received: Mutex<std::collections::HashMap<usize, usize>>,
    /// Every relay task, aborted when the proxy drops.
    relays: Mutex<Vec<tokio::task::AbortHandle>>,
}

/// See the module's *The harness*.
struct GatedProxy {
    addr: SocketAddr,
    state: Arc<ProxyState>,
    accept: tokio::task::JoinHandle<()>,
}

impl GatedProxy {
    async fn start(rule: Rule) -> Self {
        let socket = TcpSocket::new_v4().expect("the proxy socket must open");
        socket
            .set_recv_buffer_size(PROXY_RECV_BUFFER)
            .expect("the proxy's receive buffer must be settable");
        socket
            .bind("127.0.0.1:0".parse().expect("a loopback address"))
            .expect("the proxy listener must bind");
        let listener = socket.listen(64).expect("the proxy must listen");
        let addr = listener
            .local_addr()
            .expect("the proxy must have an address");
        let state = Arc::new(ProxyState {
            rule,
            gate_open: AtomicBool::new(true),
            relayed: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            paused: tokio::sync::watch::channel(false).0,
            forwarded: Mutex::new(Vec::new()),
            received: Mutex::default(),
            relays: Mutex::new(Vec::new()),
        });
        let accept_state = Arc::clone(&state);
        let accept = tokio::spawn(async move {
            loop {
                let Ok((client, _)) = listener.accept().await else {
                    return;
                };
                if !accept_state.gate_open.load(SeqCst) {
                    accept_state.refused.fetch_add(1, SeqCst);
                    drop(client);
                    continue;
                }
                let conn = accept_state.relayed.fetch_add(1, SeqCst);
                let relay_state = Arc::clone(&accept_state);
                let relay = tokio::spawn(async move {
                    // Surfaced, not swallowed: a relay that could not reach or
                    // select the store otherwise looks like a closed connection.
                    if let Err(e) = relay(&relay_state, conn, client).await {
                        eprintln!("GatedProxy: connection {conn} ended in error: {e}");
                    }
                });
                accept_state
                    .relays
                    .lock()
                    .unwrap()
                    .push(relay.abort_handle());
            }
        });
        Self {
            addr,
            state,
            accept,
        }
    }

    /// A `redis://` URL naming the proxy, with no database path: the proxy
    /// selects `TEST_VALKEY_URL`'s database on every upstream connection.
    fn url(&self) -> String {
        format!("redis://{}", self.addr)
    }

    fn open_gate(&self) {
        self.state.gate_open.store(true, SeqCst);
    }

    fn resume(&self) {
        self.state.paused.send_replace(false);
    }

    fn hits(&self) -> usize {
        self.state.hits.load(SeqCst)
    }

    fn refused(&self) -> usize {
        self.state.refused.load(SeqCst)
    }

    fn relayed(&self) -> usize {
        self.state.relayed.load(SeqCst)
    }

    fn forwarded(&self) -> Vec<Forwarded> {
        self.state.forwarded.lock().unwrap().clone()
    }

    /// Bytes the proxy read from client connection `conn`.
    fn received_on(&self, conn: usize) -> usize {
        self.state
            .received
            .lock()
            .unwrap()
            .get(&conn)
            .copied()
            .unwrap_or(0)
    }

    /// Every forwarded frame any of whose arguments contains `needle`.
    fn forwarded_naming(&self, needle: &str) -> Vec<Forwarded> {
        self.forwarded()
            .into_iter()
            .filter(|f| f.names(needle))
            .collect()
    }

    /// Every forwarded frame of the command `name`.
    fn forwarded_commands(&self, name: &str) -> Vec<Forwarded> {
        self.forwarded()
            .into_iter()
            .filter(|f| f.name() == name)
            .collect()
    }

    /// Every frame of the command `name` forwarded after the first forwarded
    /// `MULTI`, on that `MULTI`'s connection: what the batch's router step
    /// and everything after it wrote there (the checkout probe's own
    /// `DISCARD` before it excluded).
    fn forwarded_after_the_multi(&self, name: &str) -> Vec<Forwarded> {
        let log = self.forwarded();
        let Some(at) = log.iter().position(|f| f.name() == "MULTI") else {
            return Vec::new();
        };
        let conn = log[at].conn;
        log.into_iter()
            .skip(at + 1)
            .filter(|f| f.conn == conn && f.name() == name)
            .collect()
    }
}

impl Drop for GatedProxy {
    fn drop(&mut self) {
        self.accept.abort();
        for relay in self.state.relays.lock().unwrap().drain(..) {
            relay.abort();
        }
    }
}

/// Aborts its task when dropped, so a relay that ends — or is aborted — takes
/// its store→client copy with it and both sockets close.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Relay one client connection to the store, applying the rule to each
/// client→server frame.
async fn relay(state: &ProxyState, conn: usize, client: TcpStream) -> std::io::Result<()> {
    let mut upstream = TcpStream::connect(test_valkey_addr()).await?;
    if let Some(db) = test_valkey_db() {
        // Before any client frame, so the client never sees this reply.
        let db = db.to_string();
        upstream
            .write_all(&encode(&[b"SELECT", db.as_bytes()]))
            .await?;
        let mut reply = [0u8; 5];
        upstream.read_exact(&mut reply).await?;
        if &reply != b"+OK\r\n" {
            return Err(std::io::Error::other(format!(
                "SELECT {db} through the proxy was refused: {}",
                String::from_utf8_lossy(&reply)
            )));
        }
    }
    let (mut client_read, mut client_write) = client.into_split();
    let (mut upstream_read, mut upstream_write) = upstream.into_split();
    let _downstream = AbortOnDrop(tokio::spawn(async move {
        let _ = tokio::io::copy(&mut upstream_read, &mut client_write).await;
    }));

    let mut buf = BytesMut::with_capacity(64 << 10);
    loop {
        while let Some(frame) = parse_command(&buf)? {
            let bytes = buf.split_to(frame.len);
            let log = || {
                state.forwarded.lock().unwrap().push(Forwarded {
                    conn,
                    args: frame
                        .args
                        .iter()
                        .map(|a| String::from_utf8_lossy(a).into_owned())
                        .collect(),
                });
            };
            match (state.rule)(conn, &frame.args) {
                None => {
                    log();
                    upstream_write.write_all(&bytes).await?;
                }
                Some(Action::Cut) => {
                    state.gate_open.store(false, SeqCst);
                    state.hits.fetch_add(1, SeqCst);
                    // Returning drops both halves of both sockets.
                    return Ok(());
                }
                Some(Action::PauseAfter) => {
                    log();
                    upstream_write.write_all(&bytes).await?;
                    state.paused.send_replace(true);
                    state.hits.fetch_add(1, SeqCst);
                    // The sender lives as long as the proxy's state, which
                    // this relay holds, so the wait cannot fail.
                    let _ = state.paused.subscribe().wait_for(|paused| !paused).await;
                }
            }
        }
        let read = client_read.read_buf(&mut buf).await?;
        if read == 0 {
            return Ok(());
        }
        *state.received.lock().unwrap().entry(conn).or_default() += read;
    }
}

/// One complete client→server frame at the front of a buffer.
struct Parsed {
    /// Its length in bytes.
    len: usize,
    /// Its arguments, each truncated to [`LOGGED_ARG_BYTES`].
    args: Vec<Vec<u8>>,
}

/// The complete RESP array of bulk strings at the front of `buf`, or `None`
/// while it is incomplete. fred sends every command so. Cheap on an
/// incomplete frame — headers only, no argument copied whole — so a frame of
/// [`BIG_VALUE_BYTES`] arriving in pieces is not re-copied per piece.
fn parse_command(buf: &[u8]) -> std::io::Result<Option<Parsed>> {
    let Some(&first) = buf.first() else {
        return Ok(None);
    };
    if first != b'*' {
        return Err(std::io::Error::other(
            "the proxy relays RESP arrays of bulk strings only",
        ));
    }
    let Some((count, mut pos)) = header_number(buf, 1)? else {
        return Ok(None);
    };
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        match buf.get(pos) {
            None => return Ok(None),
            Some(b'$') => {}
            Some(_) => {
                return Err(std::io::Error::other(
                    "the proxy relays RESP arrays of bulk strings only",
                ))
            }
        }
        let Some((len, start)) = header_number(buf, pos + 1)? else {
            return Ok(None);
        };
        let end = start + len;
        if buf.len() < end + 2 {
            return Ok(None);
        }
        args.push(buf[start..end.min(start + LOGGED_ARG_BYTES)].to_vec());
        pos = end + 2;
    }
    Ok(Some(Parsed { len: pos, args }))
}

/// The decimal number on the header line starting at `start`, and where the
/// line after it starts; `None` while the line is incomplete.
fn header_number(buf: &[u8], start: usize) -> std::io::Result<Option<(usize, usize)>> {
    let Some(line) = buf.get(start..) else {
        return Ok(None);
    };
    let Some(end) = line.windows(2).position(|w| w == b"\r\n") else {
        return Ok(None);
    };
    let n = std::str::from_utf8(&line[..end])
        .ok()
        .and_then(|text| text.parse().ok())
        .ok_or_else(|| std::io::Error::other("a RESP header carries no number"))?;
    Ok(Some((n, start + end + 2)))
}

/// `args` as a RESP array of bulk strings.
fn encode(args: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for arg in args {
        out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        out.extend_from_slice(arg);
        out.extend_from_slice(b"\r\n");
    }
    out
}

// ═════════════════════════════════════════════════════════════════════════════
// Rules
// ═════════════════════════════════════════════════════════════════════════════

/// Cut the connection carrying the first frame whose command is `command`
/// and whose first key is `key`; forward everything else.
fn cut_at(command: &'static str, key: &str) -> Rule {
    let key = key.as_bytes().to_vec();
    let armed = AtomicBool::new(true);
    Arc::new(move |_conn, args: &[Vec<u8>]| {
        let matches = args.first()?.eq_ignore_ascii_case(command.as_bytes())
            && args.get(1).is_some_and(|k| *k == key);
        (matches && armed.swap(false, SeqCst)).then_some(Action::Cut)
    })
}

/// Pause the connection carrying the first `MULTI`, right after forwarding it.
fn pause_after_the_first_multi() -> Rule {
    let armed = AtomicBool::new(true);
    Arc::new(move |_conn, args: &[Vec<u8>]| {
        (args.first()?.eq_ignore_ascii_case(b"MULTI") && armed.swap(false, SeqCst))
            .then_some(Action::PauseAfter)
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// Clients and steps
// ═════════════════════════════════════════════════════════════════════════════

/// The unresponsive bounds of every client here: the fast verdict out of
/// reach (see the module's *Clients*), the rest production's.
fn liveness() -> LivenessOverrides {
    LivenessOverrides {
        fast_timeout: Some(SCENARIO_BUDGET),
        ..LivenessOverrides::default()
    }
}

/// A client whose every connection runs through `proxy`, with the settings
/// the module's *Clients* lists.
async fn client_through(proxy: &GatedProxy) -> ValkeyClient {
    let mut config = crate::common::test_config();
    config.valkey_url = proxy.url();
    config.valkey_pool_size = 1;
    config.valkey_command_timeout_ms = COMMAND_TIMEOUT.as_millis() as u64;
    config.valkey_reconnect_max_delay_ms = RECONNECT_MAX_DELAY_MS;
    with_liveness_overrides_for_test(liveness(), connect_test_client(&config))
        .await
        .expect("the client under test must connect through the proxy")
}

/// A key under the fixture's prefix, so its purge covers it.
fn key(tv: &TestValkey, name: &str) -> String {
    format!("{}{name}", tv.prefix())
}

/// Lose `client`'s connection and keep its reconnects failing: a `GET` of
/// `marker` (one write attempt, so it fails at the cut instead of waiting for
/// a replay) is cut by the proxy's [`cut_at`] rule, which closes the gate.
/// Returns once the proxy has refused a reconnect, so the router is in its
/// inline reconnect loop and anything sent on `client` from now on waits in
/// its command channel.
async fn stall_reconnects(deadline: Instant, proxy: &GatedProxy, client: &Client, marker: &str) {
    let once = Options {
        max_attempts: Some(1),
        ..Options::default()
    };
    let marker_get: Result<Option<String>, fred::error::Error> = by(
        deadline,
        "the marker GET",
        client.with_options(&once).get(marker),
    )
    .await;
    assert!(
        marker_get.is_err(),
        "precondition: the proxy cut the connection under the marker GET; got {marker_get:?}"
    );
    assert_eq!(proxy.hits(), 1, "precondition: the marker GET was cut");
    until(deadline, "a reconnect refused at the closed gate", || {
        proxy.refused() >= 1
    })
    .await;
}

/// The barrier (module docs, *Barrier, not sleeps*): a `GET` of `fence_key`
/// on `client`, answered only after the router processed everything queued
/// before it. Re-sent only after fred's routing miss.
async fn fence(deadline: Instant, client: &Client, fence_key: &str) {
    let options = Options {
        timeout: Some(FENCE_TIMEOUT),
        max_attempts: Some(1),
        ..Options::default()
    };
    loop {
        let answered: Result<Option<String>, fred::error::Error> = by(
            deadline,
            "the fence GET",
            client.with_options(&options).get(fence_key),
        )
        .await;
        match answered {
            Ok(_) => return,
            Err(e) if is_router_no_connection_error(&e) => {
                tokio::time::sleep(POLL).await;
            }
            Err(e) => panic!("the fence GET must be answered once the store is back: {e:?}"),
        }
    }
}

/// Reopen the gate and wait, through [`fence`], until `client`'s router has
/// processed everything queued before the fence.
async fn reopen_and_fence(deadline: Instant, proxy: &GatedProxy, client: &Client, fence_key: &str) {
    proxy.open_gate();
    fence(deadline, client, fence_key).await;
}

/// Whether `key` exists, read on the fixture's own client (not through the
/// proxy).
async fn exists(deadline: Instant, tv: &TestValkey, key: &str) -> bool {
    let n: u64 = by(deadline, "an EXISTS", tv.client().pool.next().exists(key))
        .await
        .expect("EXISTS on the fixture's client must succeed");
    n > 0
}

/// A value of [`BIG_VALUE_BYTES`].
fn big_value() -> String {
    "x".repeat(BIG_VALUE_BYTES)
}

/// The connection the first forwarded `MULTI` came on.
fn multi_connection(proxy: &GatedProxy) -> usize {
    proxy
        .forwarded_commands("MULTI")
        .first()
        .expect("precondition: the proxy forwarded the MULTI")
        .conn
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 — a WatchedTxn queued behind a stalled reconnect
// ═════════════════════════════════════════════════════════════════════════════

/// T1: the service's `WatchedTxn` (`MULTI`, a `SET`, `EXEC`, one fred
/// pipeline), sent on a `txn` slot whose connection was lost and whose
/// reconnects are refused, waits in fred's channel until its timeout fires
/// and answers `InDoubt` with fred's `Timeout`. Once the store is back, the
/// router skips every frame of it: none is forwarded, and its `SET` never
/// lands.
///
/// Stock fred never set a pipeline's flags, so it wrote the whole pipeline on
/// the new connection, unwatched, long after its caller had given up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watched_txn_queued_behind_a_stalled_reconnect_is_never_written_after_its_timeout() {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let marker = key(&tv, "t1-marker");
    let sentinel = key(&tv, "t1-sentinel");
    let proxy = GatedProxy::start(cut_at("GET", &marker)).await;
    let own = by(deadline, "the client", client_through(&proxy)).await;
    let slot = by(deadline, "the txn slot", own.acquire_txn()).await;

    stall_reconnects(deadline, &proxy, &slot, &marker).await;
    let verdict = by(
        deadline,
        "the watched transaction",
        exec_watched_writes_for_test(
            &slot,
            &ClusterHash::from(sentinel.as_str()),
            None,
            &[(sentinel.as_str(), "t1")],
        ),
    )
    .await;
    assert!(
        matches!(&verdict, WatchedExecForTest::InDoubt(e) if *e.kind() == ErrorKind::Timeout),
        "precondition: the transaction waited behind the refused reconnect until its own timeout; \
         got {verdict:?}"
    );

    reopen_and_fence(deadline, &proxy, &slot, &key(&tv, "t1-fence")).await;

    assert_eq!(
        proxy.forwarded_naming(&sentinel),
        Vec::<Forwarded>::new(),
        "no frame of the abandoned transaction may be written once its caller gave up"
    );
    assert_eq!(
        (
            proxy.forwarded_commands("MULTI"),
            proxy.forwarded_commands("EXEC")
        ),
        (Vec::new(), Vec::new()),
        "neither the abandoned transaction's MULTI nor its EXEC was written"
    );
    assert!(
        !exists(deadline, &tv, &sentinel).await,
        "the abandoned transaction's SET never landed"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T2 — CreateLayer's object write: a fred Transaction and an upload commit
// ═════════════════════════════════════════════════════════════════════════════

/// T2, end to end through `LayerStore::create_layer`: the proxy cuts the
/// object write's connection at the first keyed frame after its `MULTI`
/// (the store discards the `MULTI` with the connection) and refuses the
/// reconnect. fred's transaction router reconnects inline to replay the whole
/// transaction once (the `txn` pool's one attempt), so the replay waits
/// behind the refused reconnect while the create's size-scaled timeout fires:
/// the create answers `UNAVAILABLE` naming the timed-out object write. Once
/// the store is back, the replay's attempt is refused before any frame is
/// written: no `EXEC` ever reaches the store, and no object materializes —
/// no orphan for the orphan sweep.
///
/// Stock fred's transaction router never read the flag, so the replay wrote
/// the whole transaction on the new connection and the object materialized,
/// unmapped, after the create had answered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_object_write_cut_mid_transaction_is_never_replayed_after_its_timeout() {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let session_id = by(
        deadline,
        "the session",
        SessionStore::new(tv.client().clone(), tv.key_schema().clone()).create(
            "t2-session",
            "/t2-session",
            USER,
        ),
    )
    .await
    .expect("the session must be created on the fixture's client");

    // The first keyed frame after a MULTI, on the connection that sent the
    // MULTI: only the object write sends a MULTI through this client.
    let prefix = tv.key_schema().prefix().as_bytes().to_vec();
    let in_multi: Mutex<std::collections::HashSet<usize>> = Mutex::default();
    let armed = AtomicBool::new(true);
    let rule: Rule = Arc::new(move |conn, args: &[Vec<u8>]| {
        let name = args.first()?;
        if name.eq_ignore_ascii_case(b"MULTI") {
            in_multi.lock().unwrap().insert(conn);
            return None;
        }
        let keyed = args.get(1).is_some_and(|k| k.starts_with(&prefix));
        (keyed && in_multi.lock().unwrap().contains(&conn) && armed.swap(false, SeqCst))
            .then_some(Action::Cut)
    });
    let proxy = GatedProxy::start(rule).await;
    let own = by(deadline, "the client", client_through(&proxy)).await;
    assert_eq!(
        own.txn_pool.slots(),
        1,
        "precondition: one txn slot, so the object write rides the slot the fence uses"
    );
    let slot_client: Client = {
        let slot = by(deadline, "the txn slot", own.acquire_txn()).await;
        slot.deref().clone()
    };
    let store = LayerStore::new(own.clone(), tv.key_schema().clone(), 4096);
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 1,
        ..Default::default()
    };

    let err = by(
        deadline,
        "the create",
        store.create_layer(
            &session_id,
            LAYER,
            USER,
            &state,
            Arc::new(MockExternalValueStore::new()),
        ),
    )
    .await
    .expect_err("an object write cut mid-transaction must not answer OK");
    let prefix = format!(
        "EXEC object write failed ({}): ",
        ErrorKind::Timeout.to_str()
    );
    assert!(
        matches!(&err, ServiceError::Unavailable(m) if m.starts_with(&prefix)),
        "UNAVAILABLE naming the object write's timeout; got {err:?}"
    );
    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the object write's connection was cut"
    );
    assert!(
        proxy.refused() >= 1,
        "precondition: the replay waited behind a refused reconnect when the create gave up"
    );

    reopen_and_fence(deadline, &proxy, &slot_client, &key(&tv, "t2-fence")).await;

    assert_eq!(
        proxy.forwarded_commands("EXEC"),
        Vec::<Forwarded>::new(),
        "the replay of a transaction whose caller gave up must write nothing: no EXEC"
    );
    assert_eq!(
        crate::common::scan_keys(
            tv.client(),
            &format!("{}{{*}}:obj:hdr", tv.key_schema().prefix())
        )
        .await,
        Vec::<fred::types::Key>::new(),
        "no object materialized: no orphan header"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T3 — dropped calls
// ═════════════════════════════════════════════════════════════════════════════

/// T3, a single command: a `SET` sent on a data-pool member whose reconnect
/// is refused, inside a `tokio::time::timeout` of [`DROP_AFTER`] — far under
/// the command timeout, so fred's own timer never fires — is dropped while it
/// waits in fred's channel. Once the store is back the router skips it: never
/// forwarded, never landed.
///
/// Stock fred set the flag only when the awaited call ended with an error;
/// a dropped call never set it, so the `SET` was written after the reconnect.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn single_command_dropped_while_queued_behind_a_reconnect_is_never_written() {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let marker = key(&tv, "t3-single-marker");
    let dropped_key = key(&tv, "t3-single-dropped");
    let proxy = GatedProxy::start(cut_at("GET", &marker)).await;
    let own = by(deadline, "the client", client_through(&proxy)).await;
    let client = own.pool.next().clone();

    stall_reconnects(deadline, &proxy, &client, &marker).await;
    let dropped = tokio::time::timeout(
        DROP_AFTER,
        client.set::<(), _, _>(&dropped_key, "t3", None, None, false),
    )
    .await;
    assert!(
        dropped.is_err(),
        "precondition: the SET was still waiting behind the reconnect when its future was \
         dropped; got {dropped:?}"
    );

    reopen_and_fence(deadline, &proxy, &client, &key(&tv, "t3-single-fence")).await;

    assert_eq!(
        proxy.forwarded_naming(&dropped_key),
        Vec::<Forwarded>::new(),
        "a command whose future was dropped must never be written"
    );
    assert!(
        !exists(deadline, &tv, &dropped_key).await,
        "the dropped SET never landed"
    );
}

/// T3, a pipeline: a task awaiting `try_all` on a two-`SET` pipeline, sent on
/// a data-pool member whose reconnect is refused, is aborted once the
/// pipeline waits in fred's channel. Once the store is back the router skips
/// both commands: neither is forwarded, neither lands.
///
/// Stock fred never set a pipeline's flags, so both `SET`s were written after
/// the reconnect.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_whose_caller_is_aborted_while_queued_behind_a_reconnect_is_never_written() {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let marker = key(&tv, "t3-pipeline-marker");
    let first = key(&tv, "t3-pipeline-first");
    let second = key(&tv, "t3-pipeline-second");
    let proxy = GatedProxy::start(cut_at("GET", &marker)).await;
    let own = by(deadline, "the client", client_through(&proxy)).await;
    let client = own.pool.next().clone();

    stall_reconnects(deadline, &proxy, &client, &marker).await;
    let queued = || client.inner().counters.cmd_buffer_len.load(SeqCst);
    let before = queued();
    let call = tokio::spawn({
        let (client, first, second) = (client.clone(), first.clone(), second.clone());
        async move {
            let pipeline = client.pipeline();
            let _: () = pipeline
                .set(&first, "t3", None, None, false)
                .await
                .expect("queueing into a pipeline is local");
            let _: () = pipeline
                .set(&second, "t3", None, None, false)
                .await
                .expect("queueing into a pipeline is local");
            pipeline.try_all::<Value>().await
        }
    });
    until(
        deadline,
        "the pipeline queued in the router's channel",
        || queued() > before,
    )
    .await;
    call.abort();
    assert!(
        call.await.unwrap_err().is_cancelled(),
        "precondition: the task was aborted while it awaited try_all"
    );

    reopen_and_fence(deadline, &proxy, &client, &key(&tv, "t3-pipeline-fence")).await;

    assert_eq!(
        (
            proxy.forwarded_naming(&first),
            proxy.forwarded_naming(&second)
        ),
        (Vec::new(), Vec::new()),
        "no command of a pipeline whose caller was aborted may be written"
    );
    assert!(
        !exists(deadline, &tv, &first).await && !exists(deadline, &tv, &second).await,
        "neither SET of the aborted pipeline landed"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — the retry buffer's replay
// ═════════════════════════════════════════════════════════════════════════════

/// T4: a two-`SET` pipeline on a data-pool member, whose commands keep fred's
/// write attempts, is swallowed by the proxy (its first `SET` matches the
/// [`cut_at`] rule, so nothing of it is forwarded) and its connection cut
/// with the gate closed. fred keeps both commands for a replay on the next
/// connection, so the caller waits out its own timeout (fred's `Timeout`,
/// not the cut's error). Once the store is back, the replay skips both: none
/// is forwarded, none lands.
///
/// Stock fred never set a pipeline's flags, so the reconnect replayed both
/// `SET`s after the caller had timed out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_lost_with_its_connection_is_not_replayed_after_its_timeout() {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let first = key(&tv, "t4-first");
    let second = key(&tv, "t4-second");
    let proxy = GatedProxy::start(cut_at("SET", &first)).await;
    let own = by(deadline, "the client", client_through(&proxy)).await;
    let client = own.pool.next().clone();

    let pipeline = client.pipeline();
    let _: () = pipeline
        .set(&first, "t4", None, None, false)
        .await
        .expect("queueing into a pipeline is local");
    let _: () = pipeline
        .set(&second, "t4", None, None, false)
        .await
        .expect("queueing into a pipeline is local");
    let result = by(deadline, "the pipeline", pipeline.all::<Value>()).await;
    assert!(
        matches!(&result, Err(e) if *e.kind() == ErrorKind::Timeout),
        "precondition: the pipeline was kept for a replay and waited out its own timeout, not \
         failed at the cut; got {result:?}"
    );
    assert_eq!(proxy.hits(), 1, "precondition: the pipeline was swallowed");
    assert!(
        proxy.refused() >= 1,
        "precondition: the replay waited behind a refused reconnect when the caller gave up"
    );

    reopen_and_fence(deadline, &proxy, &client, &key(&tv, "t4-fence")).await;

    assert_eq!(
        (
            proxy.forwarded_naming(&first),
            proxy.forwarded_naming(&second)
        ),
        (Vec::new(), Vec::new()),
        "a pipeline whose caller timed out must not be replayed from the retry buffer"
    );
    assert!(
        !exists(deadline, &tv, &first).await && !exists(deadline, &tv, &second).await,
        "neither SET of the timed-out pipeline landed"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T5 — a fred Transaction cut partway through its write
// ═════════════════════════════════════════════════════════════════════════════

/// Where T5's stall falls: which of the router's checks first sees the
/// caller gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransactionCut {
    /// `MULTI`, the sentinel's `SET`, the large `SET`: the large `SET` is the
    /// last command, so every per-command check runs before the timer fires
    /// and the check before `EXEC` is the one that closes the connection.
    BeforeExec,
    /// `MULTI`, the large `SET`, the sentinel's `SET` (also of
    /// [`BIG_VALUE_BYTES`]): the sentinel's `SET` is still to be written when
    /// the timer fires, so the per-command check before it closes the
    /// connection — the shape of a large object write cut partway through its
    /// commands.
    ///
    /// The sentinel's value is large on purpose. Without the per-command
    /// check fred writes the sentinel's `SET` and then closes at the check
    /// before `EXEC`, with the store's replies unread on its socket, so the
    /// kernel resets the connection and discards whatever the proxy had not
    /// yet read: a small `SET` written that way can vanish before the proxy
    /// logs it. A large one cannot be written without the proxy reading most
    /// of it, which [`GatedProxy::received_on`] counts.
    BeforeANextCommand,
}

/// Read `live_edit_valkey_errors_total{pool="txn", kind="Canceled"}`: fred's
/// error event for a connection closed under an abandoned transaction.
fn txn_canceled_errors(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_errors_total",
        &[("pool", "txn"), ("kind", "Canceled")],
    )
    .unwrap_or(0.0) as u64
}

/// Read `live_edit_valkey_reconnects_total{pool="txn"}`.
fn txn_reconnects(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_reconnects_total",
        &[("pool", "txn")],
    )
    .unwrap_or(0.0) as u64
}

/// T5: a fred `Transaction` on a `txn` slot — `MULTI`, a `SET` of the
/// sentinel and a `SET` of [`BIG_VALUE_BYTES`] in the order `cut` names,
/// `EXEC` — whose connection the proxy stops reading right after the `MULTI`.
/// fred's write of the large `SET` blocks, the caller's timeout fires
/// meanwhile, and the proxy then resumes. The router finishes that write,
/// finds the caller gone before its next frame (the sentinel's `SET`, or the
/// `EXEC`), and closes the connection instead (the store discards the `MULTI`
/// with it):
///
/// - no frame after the large `SET` is written: no `EXEC` is forwarded, and in
///   [`TransactionCut::BeforeANextCommand`] the proxy never receives the
///   sentinel's `SET` (it reads less than the `MULTI`, one large value and
///   half another on the cut connection);
/// - nothing the transaction queued lands;
/// - the slot's next command — the fence — runs on a new connection;
/// - the cut counts once on `live_edit_valkey_errors_total{pool="txn",
///   kind="Canceled"}` and once on `live_edit_valkey_reconnects_total{pool="txn"}`
///   (SDD §15.3).
///
/// Stock fred wrote every remaining frame, `EXEC` included, as soon as the
/// large `SET` was through, and the transaction ran after its caller had
/// timed out.
async fn transaction_cut_scenario(cut: TransactionCut) {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let sentinel = key(&tv, "t5-sentinel");
    let large = key(&tv, "t5-large");
    let fence_key = key(&tv, "t5-fence");
    let proxy = GatedProxy::start(pause_after_the_first_multi()).await;
    let own = by(deadline, "the client", client_through(&proxy)).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    own.install_observability_hooks(Arc::clone(&metrics)).await;
    let slot = by(deadline, "the txn slot", own.acquire_txn()).await;
    let value = big_value();
    let (canceled_before, reconnects_before) =
        (txn_canceled_errors(&metrics), txn_reconnects(&metrics));

    let trx = slot.multi();
    let big = (large.as_str(), value.as_str());
    let queued = match cut {
        TransactionCut::BeforeExec => [(sentinel.as_str(), "t5"), big],
        TransactionCut::BeforeANextCommand => [big, (sentinel.as_str(), value.as_str())],
    };
    for (k, v) in queued {
        let _: () = trx
            .set(k, v, None, None, false)
            .await
            .expect("queueing into a transaction is local");
    }
    let result = by(deadline, "the transaction", trx.exec::<Value>(true)).await;
    assert!(
        matches!(&result, Err(e) if *e.kind() == ErrorKind::Timeout),
        "precondition: the transaction's write was held after its MULTI past its timeout; got \
         {result:?}"
    );
    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the proxy paused after the MULTI"
    );
    let cut_conn = multi_connection(&proxy);

    proxy.resume();
    fence(deadline, &slot, &fence_key).await;

    assert_eq!(
        proxy.forwarded_commands("EXEC"),
        Vec::<Forwarded>::new(),
        "a transaction whose caller gave up after its MULTI was written must never be sent its \
         EXEC"
    );
    assert!(
        !exists(deadline, &tv, &sentinel).await && !exists(deadline, &tv, &large).await,
        "nothing the cut transaction queued landed"
    );
    let fenced_on: Vec<usize> = proxy
        .forwarded_naming(&fence_key)
        .iter()
        .map(|f| f.conn)
        .collect();
    assert!(
        !fenced_on.is_empty() && fenced_on.iter().all(|&conn| conn != cut_conn),
        "the cut connection was closed and replaced: the fence ran on a new connection, not on \
         connection {cut_conn}; it ran on {fenced_on:?} of {} relayed",
        proxy.relayed()
    );
    if cut == TransactionCut::BeforeANextCommand {
        let received = proxy.received_on(cut_conn);
        assert!(
            proxy.forwarded_naming(&sentinel).is_empty()
                && received < BIG_VALUE_BYTES + BIG_VALUE_BYTES / 2,
            "the command queued after the stall is never written once the caller gave up: the \
             proxy read {received} bytes on the cut connection, at most the MULTI and one large \
             SET may arrive"
        );
    }

    // Both counters tick from the client's event hooks, on their own tasks:
    // wait for each to move, read the deltas, then make one more round trip
    // on the slot and read them again, so a duplicate tick arriving late is
    // caught rather than missed.
    until(
        deadline,
        "the cut's error and reconnect events counted",
        || {
            txn_canceled_errors(&metrics) > canceled_before
                && txn_reconnects(&metrics) > reconnects_before
        },
    )
    .await;
    let deltas = || {
        (
            txn_canceled_errors(&metrics) - canceled_before,
            txn_reconnects(&metrics) - reconnects_before,
        )
    };
    let first = deltas();
    fence(deadline, &slot, &key(&tv, "t5-fence-again")).await;
    assert_eq!(
        (first, deltas()),
        ((1, 1), (1, 1)),
        "the cut counts once on valkey_errors_total{{pool=\"txn\",kind=\"Canceled\"}} and once \
         on valkey_reconnects_total{{pool=\"txn\"}}, and stays at once after a further round \
         trip on the slot"
    );
}

/// T5 with the stall on the last command: the check before `EXEC` closes the
/// connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transaction_cut_after_its_multi_is_closed_without_its_exec() {
    transaction_cut_scenario(TransactionCut::BeforeExec).await;
}

/// T5 with a command still queued behind the stall: the per-command check
/// closes the connection before writing it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transaction_cut_before_a_queued_command_is_closed_without_writing_it() {
    transaction_cut_scenario(TransactionCut::BeforeANextCommand).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T6 — a WatchedTxn pipeline cut with its MULTI open
// ═════════════════════════════════════════════════════════════════════════════

/// Read `live_edit_valkey_txn_dirty_checkouts_total{reason}`.
fn dirty_checkouts(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_txn_dirty_checkouts_total",
        &[("reason", reason)],
    )
    .unwrap_or(0.0) as u64
}

/// Read `live_edit_storage_watch_unwatch_failures_total{site}`.
fn unwatch_failures(metrics: &Metrics, site: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_storage_watch_unwatch_failures_total",
        &[("site", site)],
    )
    .unwrap_or(0.0) as u64
}

/// T6: the WATCH-route commit's shape — a `WatchGuard` holding a `WATCH`, then
/// a `WatchedTxn` on its connection (`MULTI`, the sentinel's `SET`, a `SET`
/// of [`BIG_VALUE_BYTES`], `EXEC`) — whose connection the proxy stops reading
/// right after the `MULTI`, as in T5. The caller times out in doubt while
/// fred's write of the large `SET` blocks; the proxy then resumes. fred
/// skips the `EXEC` and writes a `DISCARD` on the `MULTI`'s connection
/// instead, so:
///
/// - the store runs nothing the transaction queued;
/// - the guard's pinned `UNWATCH` is acknowledged, not answered `QUEUED`
///   inside an open `MULTI` (a counted give-up and a WatchGuard ERROR);
/// - the slot's next checkout probe finds no leaked `MULTI`;
/// - the connection is kept, unlike a cut fred `Transaction`'s (T5): the
///   fence runs on the `MULTI`'s own connection, and neither the `Canceled`
///   txn error nor the txn reconnect counter moves. Closing it under every
///   cut pipeline would fail other callers' commands sharing it.
///
/// Without the `DISCARD` the `MULTI` stayed open on the pooled connection
/// until the next checkout's probe found it and recycled the slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watched_pipeline_cut_after_its_multi_is_discarded_and_leaves_no_open_multi() {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let watched = key(&tv, "t6-watched");
    let sentinel = key(&tv, "t6-sentinel");
    let large = key(&tv, "t6-large");
    let proxy = GatedProxy::start(pause_after_the_first_multi()).await;
    let own = by(deadline, "the client", client_through(&proxy)).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    own.install_observability_hooks(Arc::clone(&metrics)).await;
    let value = big_value();

    let fence_key = key(&tv, "t6-fence");

    let checkout = by(deadline, "the txn slot", own.acquire_txn_for(&watched)).await;
    let (canceled_before, reconnects_before) =
        (txn_canceled_errors(&metrics), txn_reconnects(&metrics));
    let mut guard = WatchGuard::new(checkout, own.watch_unwatch_tracker())
        .with_metrics(Some(Arc::clone(&metrics)));
    by(deadline, "the WATCH", guard.deref().watch::<&str>(&watched))
        .await
        .expect("WATCH must succeed");
    guard.mark_watched(&watched, &[]);

    let verdict = by(
        deadline,
        "the watched transaction",
        exec_watched_writes_for_test(
            &guard,
            &ClusterHash::from(watched.as_str()),
            None,
            &[(sentinel.as_str(), "t6"), (large.as_str(), value.as_str())],
        ),
    )
    .await;
    assert!(
        matches!(&verdict, WatchedExecForTest::InDoubt(e) if *e.kind() == ErrorKind::Timeout),
        "precondition: the transaction's write was held after its MULTI past its timeout; got \
         {verdict:?}"
    );
    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the proxy paused after the MULTI"
    );
    let cut = multi_connection(&proxy);

    proxy.resume();
    fence(deadline, &guard, &fence_key).await;

    let fenced_on: Vec<usize> = proxy
        .forwarded_naming(&fence_key)
        .iter()
        .map(|f| f.conn)
        .collect();
    assert_eq!(
        fenced_on,
        vec![cut],
        "a cut pipeline keeps its connection: the fence ran on the MULTI's connection {cut}, \
         not on a replacement"
    );
    assert_eq!(
        proxy.forwarded_after_the_multi("DISCARD").len(),
        1,
        "fred must write one DISCARD on the cut MULTI's connection {cut}, after the MULTI"
    );
    assert_eq!(
        proxy.forwarded_commands("EXEC"),
        Vec::<Forwarded>::new(),
        "the abandoned transaction's EXEC is never written"
    );
    let unwatched = by(deadline, "the pinned UNWATCH", guard.unwatch_pinned()).await;
    assert!(
        unwatched.is_ok(),
        "the guard's UNWATCH must be acknowledged on a connection with no MULTI open; got \
         {unwatched:?}"
    );
    assert_eq!(
        (
            unwatch_failures(&metrics, "explicit"),
            unwatch_failures(&metrics, "drop")
        ),
        (0, 0),
        "no WatchGuard teardown gave up"
    );
    drop(guard);

    let _next = by(deadline, "the next checkout", own.acquire_txn_for(&watched)).await;
    assert_eq!(
        dirty_checkouts(&metrics, "leaked_multi"),
        0,
        "the next checkout's probe must find no MULTI left open by the cut pipeline"
    );
    assert!(
        !exists(deadline, &tv, &sentinel).await && !exists(deadline, &tv, &large).await,
        "nothing the discarded transaction queued landed"
    );
    // Read last, after several further round trips on the slot, so an event
    // tick a closed connection would have caused has had them to land.
    assert_eq!(
        (
            txn_canceled_errors(&metrics) - canceled_before,
            txn_reconnects(&metrics) - reconnects_before
        ),
        (0, 0),
        "a cut pipeline closes no connection: no Canceled txn error and no txn reconnect"
    );
}
