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

//! Shared harness written for the atomic snapshot read's tests (b.y3m Epic 2
//! Task 4) — `snapshot_read_atomic_test.rs` and the cluster-gated
//! `snapshot_read_cluster_test.rs` — and used as well by the other suites
//! that compare snapshots, tap `MONITOR`, or stage replies through
//! [`ScriptReplyProxy`] (`b_v6d_router_test.rs` builds its clustered `-ASK`
//! facade on it). The snapshot-fence suites borrow two of its pieces:
//! `client_with_limits`, and `MonitorTap`, through which
//! `snapshot_fence_writer_test.rs` watches a background commit's attempts.
//! `snapshot_read_fenced_test.rs` uses all three instruments and the read
//! helpers: the atomic read is its oracle, and its proxy fails a fence
//! release, fails renewals, or only counts them. The tests that stop a read or
//! a call by polling it by hand (`structural_cache_cold_start_test.rs`,
//! `snapshot_flights_test.rs`, and one in `snapshot_fence_writer_test.rs`)
//! run it through [`on_current_thread`].
//!
//! Three instruments, each standing in for something no `TestValkey` call can
//! observe:
//!
//! - [`SnapshotView`] — a `Debug + PartialEq` projection of
//!   `storage::diff_read::ObjectSnapshot`, which derives neither, so two reads
//!   can be compared whole and a mismatch prints both sides.
//! - [`MonitorTap`] — a raw `MONITOR` connection to one Valkey node. Commands a
//!   script issues are invisible to the client that ran the script; the
//!   server's own command feed is the only record of which keys
//!   `object_snapshot_read.lua` touched and which commands it issued. Lines are
//!   collected up to a fence command the test issues after the read, so the
//!   record is complete without a sleep.
//! - [`ScriptReplyProxy`] — a TCP proxy in front of the real Valkey that
//!   rewrites the command frames its rule matches into an `EVAL` of a canned
//!   one-line script (or delays, stalls or drops the connection, or holds or
//!   trickles the replies on it). Every other
//!   frame is forwarded byte for byte, so `ValkeyClient::connect` and every
//!   other command run against the real store; only the matched command's
//!   reply is substituted, in its own position on the connection. It is how a
//!   test stages a reply the test store never sends — a `-LOADING`, a `-BUSY`,
//!   a cluster refusal, a reply in another format — with the real
//!   reply-decoding and classification code downstream of it. The rule is any
//!   predicate on a frame's arguments. Suites match script frames
//!   (`snapshot_read_atomic_test.rs` through [`script_frames_naming`], the
//!   `b_v6d` suites by an `EVALSHA` of a known SHA), a `GET` of one key
//!   (`b_v6d_tests.rs`), `CLUSTER SLOTS` / `INFO` and `DISCARD`
//!   (`b_v6d_router_test.rs`), `CONFIG GET appendonly`
//!   (`durability_guard_test.rs`, which stages the server's answer without
//!   changing its configuration), and `CLUSTER SLOTS`, `XREAD`, `PING`,
//!   `AUTH` and `CLIENT` frames (`stream_mux_reader_connect_test.rs`, which
//!   builds a two-node cluster facade on it, and `stream_mux_reauth_test.rs`).
//!   Its [`ProxyAction::HoldReplies`]
//!   forwards the matched command unchanged and holds the replies behind a
//!   [`ReplyGate`] instead: the command has run on the store, and the client
//!   has not yet seen its answer — the window in which a test interleaves
//!   another actor's write after a read and before the reader acts on it.
//!   Each upstream connection first selects the database `TEST_VALKEY_URL`
//!   names ([`test_valkey_db`]), so a client through the proxy — standalone,
//!   or clustered on a facade, which never sends `SELECT` — shares the
//!   fixture's database. A rule may also see which client connection a
//!   frame came on ([`ScriptReplyProxy::start_per_connection`]: the b.vn4
//!   `-ASK` facade pins `ASKING` and the command right behind it on one
//!   connection), and a proxy that parses the store's replies can put a reply
//!   of its own in a command's position
//!   ([`ScriptReplyProxy::start_rewriting_replies`],
//!   [`ProxyAction::DiscardReplying`]: an `EXEC` refused `-TRYAGAIN` after
//!   the real connection discarded its transaction;
//!   [`ProxyAction::RefuseReplying`]: a queued command refused, so its
//!   transaction's `EXEC` answers `EXECABORT`), or close the connection once
//!   a command's reply has arrived, that reply undelivered
//!   ([`ProxyAction::CloseAfterReply`]: a lost `EXEC` reply).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::BytesMut;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use live_edit::config::Config;
use live_edit::domain::types::{Field, Section, TimeSample};
use live_edit::error::ServiceError;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{
    read_object_snapshot, AtomicSnapshot, ObjectSnapshot, SnapshotRead, SnapshotReadConsumer,
    SnapshotTooLarge,
};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::test_support::{connect_test_client, read_object_state_for_test};

// ═════════════════════════════════════════════════════════════════════════════
// SnapshotView
// ═════════════════════════════════════════════════════════════════════════════

/// Everything an `ObjectSnapshot` carries that a consumer reads, in a form that
/// compares whole.
///
/// Order-free where the snapshot's order carries no meaning — sections and
/// children lists keyed by id, tombstones sorted (they come from `SMEMBERS`) —
/// and order-preserving where it does: each children list is the list's own
/// order, and fields and time samples keep the order the reader produced them
/// in (both readers take it from `ZRANGE`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SnapshotView {
    pub(crate) object_id: u64,
    pub(crate) sequence_num: u64,
    pub(crate) first_version: u64,
    pub(crate) source_format: String,
    pub(crate) next_set_order: u64,
    pub(crate) sections: BTreeMap<u64, Section>,
    pub(crate) children: BTreeMap<u64, Vec<u64>>,
    pub(crate) fields: Vec<Field>,
    pub(crate) time_samples: Vec<TimeSample>,
    pub(crate) tombstones: Vec<(u64, u64, u64)>,
}

pub(crate) fn view(s: &ObjectSnapshot) -> SnapshotView {
    let mut tombstones = s.tombstones.clone();
    tombstones.sort_unstable();
    SnapshotView {
        object_id: s.object_id,
        sequence_num: s.sequence_num,
        first_version: s.first_version,
        source_format: s.source_format.clone(),
        next_set_order: s.next_set_order,
        sections: s.sections.iter().map(|(k, v)| (*k, v.clone())).collect(),
        children: s.children.iter().map(|(k, v)| (*k, v.clone())).collect(),
        fields: s.fields.clone(),
        time_samples: s.time_samples.clone(),
        tombstones,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Reads
// ═════════════════════════════════════════════════════════════════════════════

/// `read_object_snapshot` for `consumer`, which must answer with a snapshot.
pub(crate) async fn read_atomic_as(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    consumer: SnapshotReadConsumer,
) -> AtomicSnapshot {
    match read_object_snapshot(client, keys, &object_id.to_string(), consumer).await {
        Ok(SnapshotRead::Snapshot(snapshot)) => *snapshot,
        Ok(SnapshotRead::TooLarge(too_large)) => panic!(
            "the atomic read of object {object_id} must answer with its state; it answered \
             TOO_LARGE {too_large:?}"
        ),
        Err(e) => panic!("the atomic read of object {object_id} must succeed; got {e:?}"),
    }
}

/// [`read_atomic_as`] for the catch-up consumer.
pub(crate) async fn read_atomic(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> AtomicSnapshot {
    read_atomic_as(client, keys, object_id, SnapshotReadConsumer::Catchup).await
}

/// `read_object_snapshot` for the catch-up consumer, which must answer
/// `TOO_LARGE`.
pub(crate) async fn read_too_large(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> SnapshotTooLarge {
    match read_object_snapshot(
        client,
        keys,
        &object_id.to_string(),
        SnapshotReadConsumer::Catchup,
    )
    .await
    {
        Ok(SnapshotRead::TooLarge(too_large)) => too_large,
        Ok(SnapshotRead::Snapshot(snapshot)) => panic!(
            "object {object_id} must answer TOO_LARGE under this client's limits; it answered \
             with its state ({} items, {} bytes, {} entries)",
            snapshot.item_count, snapshot.reply_bytes, snapshot.entry_count
        ),
        Err(e) => panic!("object {object_id} must answer TOO_LARGE, not fail; got {e:?}"),
    }
}

/// `read_object_snapshot` for the catch-up consumer, which must fail.
pub(crate) async fn read_err(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> ServiceError {
    match read_object_snapshot(
        client,
        keys,
        &object_id.to_string(),
        SnapshotReadConsumer::Catchup,
    )
    .await
    {
        Err(e) => e,
        Ok(SnapshotRead::Snapshot(_)) => {
            panic!("the atomic read of object {object_id} must fail; it answered with a snapshot")
        }
        Ok(SnapshotRead::TooLarge(t)) => {
            panic!("the atomic read of object {object_id} must fail; it answered TOO_LARGE {t:?}")
        }
    }
}

/// The SR-1.2 equality: on a state the object's own writes produced, the
/// atomic read returns exactly what an independent plain read of every key
/// returns — the test-support reference reader
/// (`test_support::read_object_state_for_test`: unbounded pipelined reads on
/// the data pool, a parent set of the root plus every indexed section, no
/// `obj:chs`), which shares only the per-entry decoders with the atomic script.
///
/// Only meaningful on a quiescent object — the caller holds the only writer.
/// The field and time-sample *order* is compared too, through [`view`]: both
/// readers take it from `ZRANGE`, so a difference would be a decoder change,
/// not noise.
pub(crate) async fn assert_atomic_equals_reference(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    context: &str,
) -> AtomicSnapshot {
    let reference = match read_object_state_for_test(client, keys, &object_id.to_string()).await {
        Ok(s) => s,
        Err(e) => {
            panic!("{context}: the reference read of object {object_id} must succeed; got {e:?}")
        }
    };
    let atomic = read_atomic(client, keys, object_id).await;
    assert_eq!(
        view(&atomic.state),
        view(&reference),
        "{context}: the atomic read must return exactly the state the reference reader \
         (`read_object_state_for_test`) returns for a state the object's own writes produced \
         (Spec SDD SR-1.2) — catch-up and checkpoint payloads are built from it"
    );
    atomic
}

/// A client whose atomic-read limits are `max_items` / `max_bytes`.
///
/// The limits are copied from `Config` onto `ValkeyClient` at connect and are
/// not settable afterwards, so a test that needs small ones connects its own
/// client. It reads and writes the same store as the caller's `TestValkey`,
/// under that fixture's key prefix, so the fixture still owns the cleanup.
pub(crate) async fn client_with_limits(max_items: u64, max_bytes: u64) -> ValkeyClient {
    let mut config: Config = crate::common::test_config();
    config.snapshot_atomic_read_max_items = max_items;
    config.snapshot_atomic_read_max_bytes = max_bytes;
    connect_test_client(&config)
        .await
        .expect("a client with small atomic-read limits must connect")
}

// ═════════════════════════════════════════════════════════════════════════════
// Hand-polled reads
// ═════════════════════════════════════════════════════════════════════════════

/// Run `scenario` to its end on a current-thread runtime of its own, on a
/// blocking thread, while the test's `TestValkey` stays on the test's
/// multi-thread runtime, whose `Drop` removes its keys.
///
/// For tests that poll a read or a call by hand and stop it between two
/// store awaits (`structural_cache_cold_start_test.rs`,
/// `snapshot_flights_test.rs`, `snapshot_fence_writer_test.rs`). Every
/// client whose reads the scenario polls is connected inside it, so that
/// client's connection tasks run on this runtime too. No other task of a
/// current-thread runtime runs while the test polls, so a reply cannot be
/// delivered inside the poll that sent its request: each such poll ends
/// `Pending`, and the test sees every step between two store awaits. On a
/// multi-thread runtime a worker can deliver the reply while the polling
/// thread is descheduled, and the read then runs past the step in that one
/// poll.
///
/// `spawn_blocking` cannot be cancelled, so a deadline over the scenario
/// belongs inside it: a timeout around this call would return while the
/// scenario's thread runs on. A panic in the scenario fails the test with its
/// own message.
pub(crate) async fn on_current_thread<F, Fut>(scenario: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()>,
{
    let joined = tokio::task::spawn_blocking(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a current-thread runtime builds")
            .block_on(scenario())
    })
    .await;
    if let Err(e) = joined {
        match e.try_into_panic() {
            Ok(payload) => std::panic::resume_unwind(payload),
            Err(e) => panic!("the scenario's thread ended without finishing: {e}"),
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Valkey endpoints
// ═════════════════════════════════════════════════════════════════════════════

/// The single-node test store's URL, `TEST_VALKEY_URL` (default
/// `redis://127.0.0.1:6379`), resolved as `TestValkey::setup` resolves it.
fn test_valkey_url() -> url::Url {
    let raw =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let url = url::Url::parse(&raw).unwrap_or_else(|e| panic!("TEST_VALKEY_URL {raw:?}: {e}"));
    assert!(
        url.password().is_none(),
        "these instruments open raw connections and send no AUTH; TEST_VALKEY_URL must name an \
         unauthenticated store (the sandbox and CI stores are)"
    );
    url
}

/// `host:port` of the single-node test store, from `TEST_VALKEY_URL`.
pub(crate) fn test_valkey_addr() -> String {
    let url = test_valkey_url();
    format!(
        "{}:{}",
        url.host_str().expect("TEST_VALKEY_URL must name a host"),
        url.port().unwrap_or(6379)
    )
}

/// The logical database `TEST_VALKEY_URL` names (`redis://host:port/<db>`),
/// or `None` for the default database 0. [`ScriptReplyProxy`] selects it on
/// every upstream connection, so a client connected through the proxy — whose
/// URL names the proxy, never a database, and on a cluster facade could not
/// select one anyway — reads and writes the same database a `TestValkey` set
/// up from `TEST_VALKEY_URL` does.
pub(crate) fn test_valkey_db() -> Option<u32> {
    let url = test_valkey_url();
    let path = url.path().trim_start_matches('/');
    if path.is_empty() {
        return None;
    }
    let db: u32 = path
        .parse()
        .unwrap_or_else(|e| panic!("TEST_VALKEY_URL {url} names database {path:?}: {e}"));
    (db != 0).then_some(db)
}

// ═════════════════════════════════════════════════════════════════════════════
// MonitorTap
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on draining a `MONITOR` feed up to its fence. Unreachable on a healthy
/// store — the fence is issued after the read returned, so it is at most a few
/// lines behind — and well inside nextest's 120 s hang-killer.
pub(crate) const MONITOR_DRAIN_BUDGET: Duration = Duration::from_secs(20);

/// One command from the `MONITOR` feed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MonitoredCommand {
    /// The issuing client as the feed names it: `lua` for a command a script
    /// issued, `addr:port` for a client.
    pub(crate) client: String,
    /// The command and its arguments, unquoted.
    pub(crate) args: Vec<String>,
}

impl MonitoredCommand {
    pub(crate) fn issued_by_script(&self) -> bool {
        self.client == "lua"
    }

    pub(crate) fn name(&self) -> String {
        self.args
            .first()
            .map(|a| a.to_ascii_uppercase())
            .unwrap_or_default()
    }
}

/// A `MONITOR` connection to one node.
pub(crate) struct MonitorTap {
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
    _writer: tokio::net::tcp::OwnedWriteHalf,
}

impl MonitorTap {
    /// Open a raw connection to `addr` and switch it to `MONITOR`. Returns once
    /// the node has acknowledged, so every command it executes from then on is
    /// in the feed.
    pub(crate) async fn start(addr: &str) -> Self {
        let stream = TcpStream::connect(addr)
            .await
            .unwrap_or_else(|e| panic!("MONITOR connection to {addr} must open: {e}"));
        let (read, mut write) = stream.into_split();
        write
            .write_all(b"*1\r\n$7\r\nMONITOR\r\n")
            .await
            .expect("MONITOR must be sent");
        let mut reader = BufReader::new(read);
        let mut ack = String::new();
        tokio::time::timeout(MONITOR_DRAIN_BUDGET, reader.read_line(&mut ack))
            .await
            .expect("MONITOR must be acknowledged within the budget")
            .expect("MONITOR acknowledgement must be readable");
        assert_eq!(
            ack.trim_end(),
            "+OK",
            "the node must acknowledge MONITOR before the test issues the command it records"
        );
        Self {
            reader,
            _writer: write,
        }
    }

    /// Every command the node executed from the acknowledgement up to and
    /// including the first one with an argument containing `fence`.
    pub(crate) async fn drain_until(&mut self, fence: &str) -> Vec<MonitoredCommand> {
        let mut out = Vec::new();
        let drain = async {
            loop {
                let mut line = String::new();
                let n = self
                    .reader
                    .read_line(&mut line)
                    .await
                    .expect("the MONITOR feed must stay readable");
                assert!(
                    n > 0,
                    "the MONITOR connection closed before the fence arrived"
                );
                let Some(cmd) = parse_monitor_line(&line) else {
                    continue;
                };
                let fenced = cmd.args.iter().any(|a| a.contains(fence));
                out.push(cmd);
                if fenced {
                    return;
                }
            }
        };
        tokio::time::timeout(MONITOR_DRAIN_BUDGET, drain)
            .await
            .unwrap_or_else(|_| {
                panic!("the MONITOR feed did not reach the fence {fence:?} within the budget")
            });
        out
    }

    /// Every command the node executed since the last drain, up to and
    /// including the `count`-th one `matches` accepts — for waiting on a
    /// command a background task issues (a commit attempt, a probe) rather
    /// than one the test issues itself. `what` names the awaited command in
    /// the panic if it does not arrive within [`MONITOR_DRAIN_BUDGET`].
    pub(crate) async fn drain_until_nth(
        &mut self,
        count: usize,
        what: &str,
        mut matches: impl FnMut(&MonitoredCommand) -> bool,
    ) -> Vec<MonitoredCommand> {
        let mut out = Vec::new();
        let mut seen = 0;
        let drain = async {
            while seen < count {
                let mut line = String::new();
                let n = self
                    .reader
                    .read_line(&mut line)
                    .await
                    .expect("the MONITOR feed must stay readable");
                assert!(n > 0, "the MONITOR connection closed before {what} arrived");
                let Some(cmd) = parse_monitor_line(&line) else {
                    continue;
                };
                if matches(&cmd) {
                    seen += 1;
                }
                out.push(cmd);
            }
        };
        if tokio::time::timeout(MONITOR_DRAIN_BUDGET, drain)
            .await
            .is_err()
        {
            panic!(
                "the MONITOR feed did not show {what} {count} time(s) within the budget; saw it \
                 {seen} time(s)"
            );
        }
        out
    }
}

/// A fence token unique to this call: issue a command naming it after the
/// operation under test, then [`MonitorTap::drain_until`] it.
pub(crate) fn monitor_fence(prefix: &str) -> String {
    format!("{prefix}monitor-fence:{}", uuid::Uuid::new_v4())
}

/// Parse one `MONITOR` line:
/// `+1696000000.123456 [0 lua] "HGETALL" "key"` (a script's command) or
/// `+… [0 127.0.0.1:5000] "EVALSHA" …` (a client's).
fn parse_monitor_line(line: &str) -> Option<MonitoredCommand> {
    let line = line.trim_end_matches(['\r', '\n']).strip_prefix('+')?;
    let open = line.find('[')?;
    let close = open + line[open..].find(']')?;
    let tag = &line[open + 1..close];
    let client = tag.split_once(' ').map_or(tag, |(_db, c)| c).to_string();
    let args = parse_quoted_args(&line[close + 1..])?;
    Some(MonitoredCommand { client, args })
}

/// Parse the feed's `sdscatrepr`-quoted arguments: `"a" "b\"c" "\x01"`.
fn parse_quoted_args(rest: &str) -> Option<Vec<String>> {
    let mut args = Vec::new();
    let mut chars = rest.chars().peekable();
    loop {
        while matches!(chars.peek(), Some(' ')) {
            chars.next();
        }
        match chars.next() {
            None => return Some(args),
            Some('"') => {}
            Some(_) => return None,
        }
        let mut arg = String::new();
        loop {
            match chars.next()? {
                '"' => break,
                '\\' => match chars.next()? {
                    'n' => arg.push('\n'),
                    'r' => arg.push('\r'),
                    't' => arg.push('\t'),
                    'a' => arg.push('\u{7}'),
                    'b' => arg.push('\u{8}'),
                    'x' => {
                        let hi = chars.next()?;
                        let lo = chars.next()?;
                        let byte = u8::from_str_radix(&format!("{hi}{lo}"), 16).ok()?;
                        arg.push(char::from(byte));
                    }
                    other => arg.push(other),
                },
                c => arg.push(c),
            }
        }
        args.push(arg);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ScriptReplyProxy
// ═════════════════════════════════════════════════════════════════════════════

/// What the proxy does with a frame its rule matched.
#[derive(Debug, Clone)]
pub(crate) enum ProxyAction {
    /// Send `EVAL <script> 0` upstream in the frame's place, so the reply the
    /// client receives — in the frame's position — is that script's.
    Eval(String),
    /// Forward nothing more on this connection: the matched command never
    /// gets a reply.
    Stall,
    /// Drop the connection, both directions, in place of the matched command:
    /// the client sees its connection close with the command in flight.
    Close,
    /// Forward the matched command unchanged, then hold every reply the
    /// store sends on this connection from then on until the gate opens. The
    /// gate reports when the first held reply bytes arrive: the command has
    /// run. Meant for a command that is the only one in flight on its
    /// connection — any reply still owed to an earlier command on the same
    /// connection would be held (and reported) too.
    HoldReplies(ReplyGate),
    /// Wait, then forward the frame unchanged: the command's reply arrives
    /// late, so the command stays in flight on its connection meanwhile. Every
    /// frame behind it on the connection waits too.
    Delay(Duration),
    /// Wait, then behave as [`ProxyAction::Eval`]: the substituted reply
    /// arrives late, so the client can put more commands on the connection
    /// before it lands.
    DelayedEval(Duration, String),
    /// Wait until the gate opens, then behave as [`ProxyAction::Eval`]: the
    /// substituted reply arrives only once the test lets it, ordered by an
    /// event rather than a delay. Every frame behind it on the connection
    /// waits too.
    GatedEval(ReplyGate, String),
    /// Forward the matched command unchanged, then deliver every reply byte
    /// the store sends on this connection from then on in pieces of at most
    /// `chunk` bytes, one piece per `every`: a reply that keeps arriving,
    /// slowly. Meant, like [`ProxyAction::HoldReplies`], for a command that is
    /// the only one in flight on its connection.
    TrickleReplies { chunk: usize, every: Duration },
    /// Forward the matched command unchanged, then read nothing more from the
    /// client on this connection, while still relaying the store's replies:
    /// the client's later bytes back up in the kernel's socket buffers, so a
    /// write larger than those buffers never completes, as against a peer
    /// whose receive window stays shut. (`Stall`, by contrast, keeps reading
    /// and discarding, so the client's writes always complete.) The relay
    /// stays parked until the runtime ends.
    StopReading,
    /// Send `DISCARD` upstream in the frame's place, and hand the client
    /// `-<message>` as the frame's reply instead of `DISCARD`'s `+OK`. Meant
    /// for an `EXEC`: the real connection leaves `MULTI` (and drops its watch
    /// set) as Valkey's own `discardTransaction` would before a cluster
    /// check's refusal, and the client reads the refusal in the `EXEC`'s
    /// position. Only a proxy started with
    /// [`ScriptReplyProxy::start_rewriting_replies`] can place the reply, so
    /// any other proxy panics on it.
    DiscardReplying(String),
    /// Send a command the store refuses before queueing it (an arity error)
    /// in the frame's place, and hand the client `-<message>` as the frame's
    /// reply. Meant for a command queued inside `MULTI`: the store flags the
    /// transaction, so the `EXEC` that follows answers `EXECABORT` and runs
    /// nothing, while the client reads the refusal in the queued command's
    /// position — the shape a cluster node that checks each queued command
    /// (Valkey ≤ 8.x, Redis 7.x) gives a queued command naming a key it must
    /// redirect. Only a proxy started with
    /// [`ScriptReplyProxy::start_rewriting_replies`] can place the reply, so
    /// any other proxy panics on it.
    RefuseReplying(String),
    /// Forward the matched command unchanged, then close the client's
    /// connection once the store's reply to it has arrived, without
    /// delivering that reply or anything after it: the command ran, and the
    /// client never learns its outcome (a lost reply). The replies to earlier
    /// commands on the connection are delivered first. Only a proxy started
    /// with [`ScriptReplyProxy::start_rewriting_replies`] can find the reply,
    /// so any other proxy panics on it.
    CloseAfterReply,
}

/// The gate a [`ProxyAction::HoldReplies`] holds replies behind, and a
/// [`ProxyAction::GatedEval`] holds its substituted command behind. Cheap to
/// clone; every clone is the same gate. One-shot: once opened, it stays open.
#[derive(Debug, Clone)]
pub(crate) struct ReplyGate(Arc<ReplyGateState>);

#[derive(Debug)]
struct ReplyGateState {
    /// Notified (a stored permit, so a later waiter still sees it) when reply
    /// bytes arrive while the gate is closed.
    held: tokio::sync::Notify,
    open: tokio::sync::watch::Sender<bool>,
}

impl ReplyGate {
    pub(crate) fn new() -> Self {
        Self(Arc::new(ReplyGateState {
            held: tokio::sync::Notify::new(),
            open: tokio::sync::watch::channel(false).0,
        }))
    }

    /// Wait until the store's reply to the held command has reached the
    /// proxy — so the command has run — within [`MONITOR_DRAIN_BUDGET`].
    ///
    /// Only a [`ProxyAction::HoldReplies`] gate ever signals this. A
    /// [`ProxyAction::GatedEval`] gate holds its command before it is sent,
    /// so no reply arrives to signal it: called on one, this waits out
    /// [`MONITOR_DRAIN_BUDGET`] and panics.
    pub(crate) async fn reply_held(&self) {
        tokio::time::timeout(MONITOR_DRAIN_BUDGET, self.0.held.notified())
            .await
            .expect("the held command's reply must reach the proxy within the budget");
    }

    /// Release every held reply, and every later one.
    pub(crate) fn open(&self) {
        self.0.open.send_replace(true);
    }

    async fn wait_open(&self) {
        let mut open = self.0.open.subscribe();
        // The sender lives as long as `self`, so the wait cannot fail.
        let _ = open.wait_for(|&o| o).await;
    }
}

/// Decides, per client→server command, whether the proxy rewrites it. `args[0]`
/// is the command name as the client sent it.
pub(crate) type ProxyRule = Arc<dyn Fn(&[Vec<u8>]) -> Option<ProxyAction> + Send + Sync>;

/// [`ProxyRule`] that also sees which client connection the command came on:
/// the connection's index, from 0, in the order the proxy accepted them. For
/// a test that must tell one connection's frames from another's (two frames
/// back to back on ONE connection, a reply on the connection a command was
/// sent on).
pub(crate) type ConnProxyRule = Arc<dyn Fn(usize, &[Vec<u8>]) -> Option<ProxyAction> + Send + Sync>;

/// See the module header.
pub(crate) struct ScriptReplyProxy {
    addr: std::net::SocketAddr,
    hits: Arc<AtomicUsize>,
    accept: tokio::task::JoinHandle<()>,
}

impl ScriptReplyProxy {
    pub(crate) async fn start(upstream: String, rule: ProxyRule) -> Self {
        Self::start_per_connection(upstream, Arc::new(move |_conn, args| rule(args))).await
    }

    /// [`Self::start`] with a rule that sees each command's connection.
    pub(crate) async fn start_per_connection(upstream: String, rule: ConnProxyRule) -> Self {
        Self::start_inner(upstream, rule, false).await
    }

    /// [`Self::start_per_connection`], also able to place a reply of the
    /// proxy's own in a command's position ([`ProxyAction::DiscardReplying`],
    /// [`ProxyAction::RefuseReplying`]) or close the connection at a reply
    /// ([`ProxyAction::CloseAfterReply`]). It parses every reply the store sends, one per command forwarded (RESP2
    /// and RESP3; a RESP3 push frame is passed through uncounted), so it is
    /// only for connections that never enter a mode where a command has no
    /// reply or several (`SUBSCRIBE`, `MONITOR`, `CLIENT REPLY OFF`).
    pub(crate) async fn start_rewriting_replies(upstream: String, rule: ConnProxyRule) -> Self {
        Self::start_inner(upstream, rule, true).await
    }

    async fn start_inner(upstream: String, rule: ConnProxyRule, rewrite_replies: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the proxy listener must bind");
        let addr = listener
            .local_addr()
            .expect("the proxy must have an address");
        let hits = Arc::new(AtomicUsize::new(0));
        let task_hits = Arc::clone(&hits);
        let db = test_valkey_db();
        let accept = tokio::spawn(async move {
            let mut next_conn = 0usize;
            loop {
                let Ok((client, _)) = listener.accept().await else {
                    return;
                };
                let conn = next_conn;
                next_conn += 1;
                let upstream = upstream.clone();
                let rule = Arc::clone(&rule);
                let hits = Arc::clone(&task_hits);
                tokio::spawn(async move {
                    // Surfaced, not swallowed: a relay that could not reach or
                    // select the upstream otherwise looks to the test like a
                    // client whose connection just closed.
                    let relayed = Relay {
                        conn,
                        db,
                        rule,
                        hits,
                        rewrite_replies,
                    };
                    if let Err(e) = relayed.run(client, upstream).await {
                        eprintln!("ScriptReplyProxy: a relayed connection ended in error: {e}");
                    }
                });
            }
        });
        Self { addr, hits, accept }
    }

    /// A `redis://` URL naming the proxy, with no database path: the proxy
    /// itself selects `TEST_VALKEY_URL`'s database on every upstream
    /// connection ([`test_valkey_db`]), so a client built on this URL reads
    /// and writes the database the caller's `TestValkey` fixture does.
    pub(crate) fn url(&self) -> String {
        format!("redis://{}", self.addr)
    }

    /// The port the proxy listens on, on `127.0.0.1`.
    pub(crate) fn port(&self) -> u16 {
        self.addr.port()
    }

    /// How many frames the rule has matched so far — the vacuity guard: a test
    /// whose rule never matched staged nothing.
    pub(crate) fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

impl Drop for ScriptReplyProxy {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

/// A client connected through `proxy`, with the default atomic-read limits, on
/// the database `TEST_VALKEY_URL` names ([`ScriptReplyProxy::url`] carries its
/// path), so it sees the keys a `TestValkey` fixture writes.
pub(crate) async fn client_through(proxy: &ScriptReplyProxy) -> ValkeyClient {
    client_through_with_limits(
        proxy,
        live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
    )
    .await
}

/// A client connected through `proxy`, whose atomic-read limits are
/// `max_items` / `max_bytes` (as [`client_with_limits`]).
pub(crate) async fn client_through_with_limits(
    proxy: &ScriptReplyProxy,
    max_items: u64,
    max_bytes: u64,
) -> ValkeyClient {
    let mut config: Config = crate::common::test_config();
    config.valkey_url = proxy.url();
    config.snapshot_atomic_read_max_items = max_items;
    config.snapshot_atomic_read_max_bytes = max_bytes;
    connect_test_client(&config)
        .await
        .expect("a client must connect through the reply proxy")
}

/// A rule matching `EVALSHA` / `EVAL` frames any of whose arguments contains
/// `needle`.
pub(crate) fn script_frames_naming(needle: String, action: ProxyAction) -> ProxyRule {
    Arc::new(move |args: &[Vec<u8>]| {
        let name = args.first()?.to_ascii_uppercase();
        if name != b"EVALSHA" && name != b"EVAL" {
            return None;
        }
        args.iter()
            .any(|a| String::from_utf8_lossy(a).contains(needle.as_str()))
            .then(|| action.clone())
    })
}

/// A Lua body answering `redis.error_reply(message)`, which Valkey returns
/// verbatim as the error reply (`-<message>`).
pub(crate) fn error_reply_script(message: &str) -> String {
    let escaped = message.replace('\\', "\\\\").replace('\'', "\\'");
    format!("return redis.error_reply('{escaped}')")
}

/// One relayed client connection.
struct Relay {
    /// The connection's index, in the order the proxy accepted it.
    conn: usize,
    db: Option<u32>,
    rule: ConnProxyRule,
    hits: Arc<AtomicUsize>,
    /// Parse the replies, so a reply of the proxy's own can take a command's
    /// position ([`ScriptReplyProxy::start_rewriting_replies`]).
    rewrite_replies: bool,
}

/// The replies a [`ProxyAction::DiscardReplying`] or
/// [`ProxyAction::RefuseReplying`] replaced, and the reply a
/// [`ProxyAction::CloseAfterReply`] closes the connection at, on one
/// connection.
#[derive(Default)]
struct ReplyRewrites {
    /// Commands forwarded upstream so far: the reply index the next one's
    /// reply will have (each forwarded command gets exactly one reply).
    forwarded: u64,
    /// Reply index → the bytes the client gets in that reply's place.
    replace: BTreeMap<u64, Vec<u8>>,
    /// The reply index at which the client's connection is closed, that
    /// reply undelivered.
    close_at: Option<u64>,
}

impl Relay {
    async fn run(self, client: TcpStream, upstream: String) -> std::io::Result<()> {
        let Relay {
            conn,
            db,
            rule,
            hits,
            rewrite_replies,
        } = self;
        let mut server = TcpStream::connect(upstream).await?;
        if let Some(db) = db {
            // Before any client frame, so the client never sees this reply.
            let db = db.to_string();
            server
                .write_all(&encode_command(&[b"SELECT", db.as_bytes()]))
                .await?;
            let mut reply = Vec::new();
            while !reply.ends_with(b"\r\n") {
                let mut byte = [0u8; 1];
                if server.read(&mut byte).await? == 0 {
                    return Ok(());
                }
                reply.push(byte[0]);
            }
            if reply != b"+OK\r\n" {
                return Err(std::io::Error::other(format!(
                    "SELECT {db} through the proxy was refused: {}",
                    String::from_utf8_lossy(&reply)
                )));
            }
        }
        let (mut client_read, mut client_write) = client.into_split();
        let (mut server_read, mut server_write) = server.into_split();
        // The gate a `HoldReplies` match armed on this connection, if any: set
        // before the matched frame is forwarded, so its reply cannot slip past.
        let hold: Arc<std::sync::Mutex<Option<ReplyGate>>> = Arc::new(std::sync::Mutex::new(None));
        let downstream_hold = Arc::clone(&hold);
        // The pace a `TrickleReplies` match set on this connection, if any: set
        // before the matched frame is forwarded, as `hold` is.
        let trickle: Arc<std::sync::Mutex<Option<(usize, Duration)>>> =
            Arc::new(std::sync::Mutex::new(None));
        let downstream_trickle = Arc::clone(&trickle);
        // The replies a `DiscardReplying` / `RefuseReplying` match replaced
        // (and where a `CloseAfterReply` closes): the upstream side
        // records the replaced reply's index before it forwards the `DISCARD`, so
        // the reply cannot arrive before its entry.
        let rewrites: Arc<std::sync::Mutex<ReplyRewrites>> = Arc::default();
        let downstream_rewrites = Arc::clone(&rewrites);
        let downstream = tokio::spawn(async move {
            let mut chunk = vec![0u8; 16 * 1024];
            // With `rewrite_replies`: the bytes of a reply not yet complete, and
            // the index of the next reply.
            let mut partial = BytesMut::new();
            let mut index = 0u64;
            loop {
                let n = match server_read.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => n,
                };
                // Set when a `CloseAfterReply` reply arrived: the replies
                // before it are delivered, then the connection is closed.
                let mut close = false;
                let out: Vec<u8> = if rewrite_replies {
                    partial.extend_from_slice(&chunk[..n]);
                    let mut out = Vec::with_capacity(partial.len());
                    while let Some(len) = reply_frame_len(&partial) {
                        let frame = partial.split_to(len);
                        // A RESP3 push frame answers no command.
                        if frame[0] == b'>' {
                            out.extend_from_slice(&frame);
                            continue;
                        }
                        let substitute = {
                            let mut rewrites = downstream_rewrites.lock().unwrap();
                            if rewrites.close_at == Some(index) {
                                close = true;
                                break;
                            }
                            rewrites.replace.remove(&index)
                        };
                        index += 1;
                        out.extend_from_slice(substitute.as_deref().unwrap_or(&frame));
                    }
                    out
                } else {
                    chunk[..n].to_vec()
                };
                if close {
                    // Dropping the write half on return sends the client a
                    // FIN: its connection ends with the reply unread.
                    let _ = client_write.write_all(&out).await;
                    return;
                }
                let gate = downstream_hold.lock().unwrap().clone();
                if let Some(gate) = gate {
                    gate.0.held.notify_one();
                    gate.wait_open().await;
                }
                let pace = *downstream_trickle.lock().unwrap();
                match pace {
                    Some((piece, every)) => {
                        for part in out.chunks(piece.max(1)) {
                            if client_write.write_all(part).await.is_err() {
                                return;
                            }
                            tokio::time::sleep(every).await;
                        }
                    }
                    None => {
                        if client_write.write_all(&out).await.is_err() {
                            return;
                        }
                    }
                }
            }
        });
        // Counts a command forwarded upstream: its reply's index.
        let forwarded = |rewrites: &Arc<std::sync::Mutex<ReplyRewrites>>| {
            rewrites.lock().unwrap().forwarded += 1;
        };

        let mut buf = BytesMut::with_capacity(16 * 1024);
        let mut stalled = false;
        loop {
            if client_read.read_buf(&mut buf).await? == 0 {
                return Ok(());
            }
            if stalled {
                buf.clear();
                continue;
            }
            loop {
                match parse_command_frame(&buf) {
                    Frame::Incomplete => break,
                    Frame::Opaque => {
                        // Not a RESP array: forward as is. Its replies cannot be
                        // counted, so a reply-rewriting relay refuses it.
                        if rewrite_replies {
                            return Err(std::io::Error::other(
                                "a reply-rewriting proxy cannot count the replies of a frame that \
                             is not a RESP array",
                            ));
                        }
                        server_write.write_all(&buf).await?;
                        buf.clear();
                        break;
                    }
                    Frame::Command { len, args } => {
                        let frame = buf.split_to(len);
                        match rule(conn, &args) {
                            None => {
                                forwarded(&rewrites);
                                server_write.write_all(&frame).await?
                            }
                            Some(ProxyAction::Eval(script)) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                forwarded(&rewrites);
                                server_write
                                    .write_all(&encode_command(&[b"EVAL", script.as_bytes(), b"0"]))
                                    .await?;
                            }
                            Some(ProxyAction::Delay(pause)) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                tokio::time::sleep(pause).await;
                                forwarded(&rewrites);
                                server_write.write_all(&frame).await?;
                            }
                            Some(ProxyAction::DelayedEval(pause, script)) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                tokio::time::sleep(pause).await;
                                forwarded(&rewrites);
                                server_write
                                    .write_all(&encode_command(&[b"EVAL", script.as_bytes(), b"0"]))
                                    .await?;
                            }
                            Some(ProxyAction::GatedEval(gate, script)) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                gate.wait_open().await;
                                forwarded(&rewrites);
                                server_write
                                    .write_all(&encode_command(&[b"EVAL", script.as_bytes(), b"0"]))
                                    .await?;
                            }
                            Some(ProxyAction::DiscardReplying(message)) => {
                                assert!(
                                    rewrite_replies,
                                    "ProxyAction::DiscardReplying needs a proxy started with \
                                 ScriptReplyProxy::start_rewriting_replies"
                                );
                                hits.fetch_add(1, Ordering::SeqCst);
                                {
                                    let mut rewrites = rewrites.lock().unwrap();
                                    let index = rewrites.forwarded;
                                    rewrites
                                        .replace
                                        .insert(index, format!("-{message}\r\n").into_bytes());
                                    rewrites.forwarded += 1;
                                }
                                server_write
                                    .write_all(&encode_command(&[b"DISCARD"]))
                                    .await?;
                            }
                            Some(ProxyAction::RefuseReplying(message)) => {
                                assert!(
                                    rewrite_replies,
                                    "ProxyAction::RefuseReplying needs a proxy started with \
                                 ScriptReplyProxy::start_rewriting_replies"
                                );
                                hits.fetch_add(1, Ordering::SeqCst);
                                {
                                    let mut rewrites = rewrites.lock().unwrap();
                                    let index = rewrites.forwarded;
                                    rewrites
                                        .replace
                                        .insert(index, format!("-{message}\r\n").into_bytes());
                                    rewrites.forwarded += 1;
                                }
                                // `GET` with no key: refused for its arity before
                                // it is queued, which inside `MULTI` flags the
                                // transaction (`EXECABORT` at its `EXEC`).
                                server_write.write_all(&encode_command(&[b"GET"])).await?;
                            }
                            Some(ProxyAction::CloseAfterReply) => {
                                assert!(
                                    rewrite_replies,
                                    "ProxyAction::CloseAfterReply needs a proxy started with \
                                 ScriptReplyProxy::start_rewriting_replies"
                                );
                                hits.fetch_add(1, Ordering::SeqCst);
                                {
                                    let mut rewrites = rewrites.lock().unwrap();
                                    rewrites.close_at = Some(rewrites.forwarded);
                                    rewrites.forwarded += 1;
                                }
                                server_write.write_all(&frame).await?;
                            }
                            Some(ProxyAction::Stall) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                stalled = true;
                                buf.clear();
                                break;
                            }
                            Some(ProxyAction::Close) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                // Dropping the copy task drops the client's write
                                // half; returning drops its read half.
                                downstream.abort();
                                return Ok(());
                            }
                            Some(ProxyAction::HoldReplies(gate)) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                *hold.lock().unwrap() = Some(gate);
                                forwarded(&rewrites);
                                server_write.write_all(&frame).await?;
                            }
                            Some(ProxyAction::StopReading) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                forwarded(&rewrites);
                                server_write.write_all(&frame).await?;
                                // Keep both client halves open, unread.
                                let _unread = client_read;
                                std::future::pending::<()>().await;
                                unreachable!("a pending future never completes");
                            }
                            Some(ProxyAction::TrickleReplies { chunk, every }) => {
                                hits.fetch_add(1, Ordering::SeqCst);
                                *trickle.lock().unwrap() = Some((chunk, every));
                                forwarded(&rewrites);
                                server_write.write_all(&frame).await?;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The length of the one complete RESP2 / RESP3 reply frame at the front of
/// `buf`, or `None` while it is incomplete. The store sends only well-formed
/// frames, so a malformed one also reads as incomplete. An attribute frame
/// (`|`) is counted with the frame it annotates.
fn reply_frame_len(buf: &[u8]) -> Option<usize> {
    fn line_end(buf: &[u8], from: usize) -> Option<usize> {
        buf.get(from..)?
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|at| from + at)
    }
    fn frame_end(buf: &[u8], at: usize) -> Option<usize> {
        let kind = *buf.get(at)?;
        let header_end = line_end(buf, at + 1)?;
        let header = std::str::from_utf8(&buf[at + 1..header_end]).ok()?;
        let body = header_end + 2;
        let elements = |count: usize, mut at: usize| -> Option<usize> {
            for _ in 0..count {
                at = frame_end(buf, at)?;
            }
            Some(at)
        };
        match kind {
            b'+' | b'-' | b':' | b'_' | b',' | b'#' | b'(' => Some(body),
            b'$' | b'!' | b'=' => match header.parse::<i64>().ok()? {
                len if len < 0 => Some(body),
                len => {
                    let end = body + len as usize + 2;
                    (buf.len() >= end).then_some(end)
                }
            },
            b'*' | b'~' | b'>' => match header.parse::<i64>().ok()? {
                count if count < 0 => Some(body),
                count => elements(count as usize, body),
            },
            b'%' => elements(2 * header.parse::<usize>().ok()?, body),
            b'|' => {
                let annotated = elements(2 * header.parse::<usize>().ok()?, body)?;
                frame_end(buf, annotated)
            }
            _ => None,
        }
    }
    frame_end(buf, 0)
}

enum Frame {
    Incomplete,
    Opaque,
    Command { len: usize, args: Vec<Vec<u8>> },
}

/// Parse one client→server RESP command (`*N\r\n` then N bulk strings) off
/// the front of `buf`.
fn parse_command_frame(buf: &[u8]) -> Frame {
    fn line(buf: &[u8], at: usize) -> Option<(&[u8], usize)> {
        let end = buf[at..].windows(2).position(|w| w == b"\r\n")?;
        Some((&buf[at..at + end], at + end + 2))
    }
    fn number(bytes: &[u8]) -> Option<usize> {
        std::str::from_utf8(bytes).ok()?.parse().ok()
    }
    if buf.is_empty() {
        return Frame::Incomplete;
    }
    if buf[0] != b'*' {
        return Frame::Opaque;
    }
    let Some((count, mut at)) = line(buf, 1) else {
        return Frame::Incomplete;
    };
    let Some(count) = number(count) else {
        return Frame::Opaque;
    };
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        if at >= buf.len() {
            return Frame::Incomplete;
        }
        if buf[at] != b'$' {
            return Frame::Opaque;
        }
        let Some((len, start)) = line(buf, at + 1) else {
            return Frame::Incomplete;
        };
        let Some(len) = number(len) else {
            return Frame::Opaque;
        };
        if buf.len() < start + len + 2 {
            return Frame::Incomplete;
        }
        args.push(buf[start..start + len].to_vec());
        at = start + len + 2;
    }
    Frame::Command { len: at, args }
}

fn encode_command(args: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for arg in args {
        out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        out.extend_from_slice(arg);
        out.extend_from_slice(b"\r\n");
    }
    out
}

// The frame parser is the one piece of the proxy whose failure would be
// silent (a frame split wrong is forwarded as garbage), so it is pinned here.
#[test]
fn parse_command_frame_splits_pipelined_commands_and_waits_for_partial_ones() {
    let a = encode_command(&[b"GET", b"k"]);
    let b = encode_command(&[b"EVALSHA", b"abc", b"1", b"{1}:obj:hdr"]);
    let mut both = a.clone();
    both.extend_from_slice(&b);

    let Frame::Command { len, args } = parse_command_frame(&both) else {
        panic!("a complete leading frame must parse");
    };
    assert_eq!(len, a.len());
    assert_eq!(args, vec![b"GET".to_vec(), b"k".to_vec()]);
    let Frame::Command { args, .. } = parse_command_frame(&both[len..]) else {
        panic!("the pipelined second frame must parse");
    };
    assert_eq!(args[3], b"{1}:obj:hdr".to_vec());
    assert!(matches!(
        parse_command_frame(&b[..b.len() - 1]),
        Frame::Incomplete
    ));
}

// The reply parser is what places a `DiscardReplying` reply; a frame split
// wrong would hand the substitute to the wrong command, silently.
#[test]
fn reply_frame_len_splits_resp2_and_resp3_replies_and_waits_for_partial_ones() {
    let replies: [&[u8]; 9] = [
        b"+OK\r\n",
        b"-TRYAGAIN Multiple keys request during rehashing of slot\r\n",
        b":42\r\n",
        b"$5\r\nhello\r\n",
        b"$-1\r\n",
        b"*2\r\n$1\r\na\r\n*1\r\n:1\r\n",
        b"*-1\r\n",
        b"%1\r\n+server\r\n+valkey\r\n",
        b"|1\r\n+ttl\r\n:3\r\n$2\r\nhi\r\n",
    ];
    for reply in replies {
        let mut both = reply.to_vec();
        both.extend_from_slice(b"+QUEUED\r\n");
        assert_eq!(
            reply_frame_len(&both),
            Some(reply.len()),
            "{:?} is one frame, followed by another",
            String::from_utf8_lossy(reply)
        );
        assert_eq!(
            reply_frame_len(&reply[..reply.len() - 1]),
            None,
            "{:?} cut short is incomplete",
            String::from_utf8_lossy(reply)
        );
    }
}

#[test]
fn parse_monitor_line_reads_script_and_client_commands() {
    let script = parse_monitor_line("+1727.1 [0 lua] \"HGETALL\" \"p{7}:obj:hdr\"\r\n")
        .expect("a script line must parse");
    assert!(script.issued_by_script());
    assert_eq!(script.args, vec!["HGETALL", "p{7}:obj:hdr"]);

    let client = parse_monitor_line("+1727.2 [0 10.0.0.1:5000] \"GET\" \"a\\\"b\" \"\\x01\"")
        .expect("a client line must parse");
    assert!(!client.issued_by_script());
    assert_eq!(client.client, "10.0.0.1:5000");
    assert_eq!(client.args, vec!["GET", "a\"b", "\u{1}"]);
}
