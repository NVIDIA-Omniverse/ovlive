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

//! A dead read path comes back by itself (SDD SR-6.5, §22.21; regression pin
//! T-R2e): the hold poll reads through the replica's own client, real fred,
//! and when every connection of that client is cut and reconnects are
//! refused, the replica goes stale and does not exit while the store stays
//! unreachable; once connections are accepted again, the client reconnects
//! by itself, the next hold read succeeds, and the replica exits 0 after its
//! jitter. The test also counts the connections the proxy relays: once as
//! many have opened after the cut as before it, and for a few seconds more,
//! the count after the cut equals the count before it. That rules out a
//! client of the hold's own whose first connection opens after the cut and
//! within that window, which would show as one more. The count does not see
//! two others:
//!
//! - a client the hold builds whose only connect attempt falls inside the
//!   outage and is never retried: it opens no connection. It serves no read
//!   either, so a hold that read through it would never exit, and the test
//!   would fail at [`EXIT_DEADLINE`];
//! - a client the hold builds and connects before the cut: its connections
//!   are in the count before the cut, and it reconnects one for one with the
//!   replica's client, so the counts stay equal even if the hold reads
//!   through it.
//!
//! The client reaches the store through [`SeveringProxy`], a TCP relay that
//! can cut every relayed connection and refuse new ones (accept, then close
//! at once) until it is opened again. The client's reconnect backoff is
//! capped at 1 s (`LIVE_EDIT_VALKEY_RECONNECT_MAX_DELAY_MS`), so it is back
//! within about a second of the proxy opening.
//!
//! What the stale entry does to the gate, the streams, the writers and
//! readiness is `hold_entry_test.rs`'s
//! (`a_dead_hold_poll_connection_fails_the_replica_closed_and_its_next_read_restarts_it`,
//! a stalled connection rather than a cut one); that a stale replica waits on
//! no clock for an hour is pinned on virtual time in `hold_reactor_test.rs`.
//!
//! # Waits
//!
//! The 30 s stale threshold is the production constant, paid in every run,
//! with [`AFTER_STALE`] more of the outage and up to 5 s of jitter at the
//! exit, then [`RECONNECT_SETTLE`] (3 s) after the client's last reconnect.
//! Against a healthy store the connects take about a second, so a passing run
//! takes about 40 to 45 s.
//!
//! The test's own bounds, each at its full budget, total 91 s: two
//! [`SETTLE`]s for the setup (the startup read, the first publication),
//! [`STALE_DEADLINE`] (45 s) for the stale entry, [`AFTER_STALE`] (3 s),
//! [`EXIT_DEADLINE`] (20 s) for the exit, the rest of [`RECONNECT_DEADLINE`]
//! (counted from the same instant as the exit's, so nothing past it) and
//! [`RECONNECT_SETTLE`] (3 s). The two connects, `TestValkey::setup`'s and
//! the replica client's, and the fixture's key cleanup on drop come on top,
//! and the test does not bound them. Each connect runs up to five phases,
//! each under its own budget: 30 s a phase for the fixture's, 5 s for the
//! replica client's. The cleanup bounds each `UNLINK` and each wait for the
//! walk's next key, but not the key walk as a whole. So a run near its bounds, with its connects slowed by suite
//! load, can reach nextest's 120 s hang-killer.
//!
//! fred's reconnect policy is not the only way back: the router supervisor
//! (`storage::router_supervisor`) restarts a member's router task that ended,
//! and that reconnects too. Only with both gone does the replica stay stale
//! here.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use live_edit::dr::hold::{evaluate_hold_at_startup, spawn_hold_poll, HoldPollSettings};
use live_edit::grpc::graceful_shutdown::GracefulShutdown;
use live_edit::grpc::hold::{
    spawn_hold_reactor, HoldEntryParts, HoldGate, HoldGateState, HoldReactorSettings,
    HoldServingPoint, HoldWriterAborts,
};
use live_edit::storage::test_support::{connect_test_client, TestValkey};

use super::hold_support::answering;
use super::support::RecordingExit;
use crate::storage::snapshot_read_support::{test_valkey_addr, test_valkey_db};

/// Bound on each setup wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// The hold poll's interval: short, so a read goes out soon after the
/// client is back.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The client's Valkey command timeout: a read sent while the store is
/// unreachable fails within it.
const COMMAND_TIMEOUT_MS: u64 = 2_000;

/// The client's reconnect backoff cap.
const RECONNECT_MAX_DELAY_MS: u32 = 1_000;

/// Bound on the stale entry after the cut: the 30 s threshold, counted from
/// the last read that answered, and slack.
const STALE_DEADLINE: Duration = Duration::from_secs(45);

/// How long the outage lasts past the stale entry.
const AFTER_STALE: Duration = Duration::from_secs(3);

/// Bound on the exit once the proxy accepts again: the 1 s reconnect cap, a
/// poll interval, the 5 s jitter bound, and slack for a loaded runner.
const EXIT_DEADLINE: Duration = Duration::from_secs(20);

/// Bound, from the proxy accepting again, on every connection the client had
/// before the outage coming back: a few reconnect attempts at the 1 s cap.
const RECONNECT_DEADLINE: Duration = Duration::from_secs(10);

/// How long past the last of the client's reconnects the test watches for
/// one more connection: past the 1 s reconnect cap, so a client the hold
/// built and left retrying would have connected by then.
const RECONNECT_SETTLE: Duration = Duration::from_secs(3);

/// A TCP relay to the test store that can cut every connection it relays
/// and refuse new ones.
struct SeveringProxy {
    addr: SocketAddr,
    accepting: Arc<AtomicBool>,
    relayed: Arc<AtomicUsize>,
    refused: Arc<AtomicUsize>,
    relays: Arc<Mutex<Vec<tokio::task::AbortHandle>>>,
    accept: tokio::task::JoinHandle<()>,
}

impl SeveringProxy {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the proxy listener binds");
        let addr = listener.local_addr().expect("the proxy's address");
        let accepting = Arc::new(AtomicBool::new(true));
        let relayed = Arc::new(AtomicUsize::new(0));
        let refused = Arc::new(AtomicUsize::new(0));
        let relays: Arc<Mutex<Vec<tokio::task::AbortHandle>>> = Arc::default();
        let accept = tokio::spawn({
            let (accepting, relayed, refused, relays) = (
                Arc::clone(&accepting),
                Arc::clone(&relayed),
                Arc::clone(&refused),
                Arc::clone(&relays),
            );
            async move {
                let upstream = test_valkey_addr();
                loop {
                    let Ok((mut client, _)) = listener.accept().await else {
                        return;
                    };
                    if !accepting.load(SeqCst) {
                        refused.fetch_add(1, SeqCst);
                        drop(client);
                        continue;
                    }
                    relayed.fetch_add(1, SeqCst);
                    let upstream = upstream.clone();
                    let relay = tokio::spawn(async move {
                        let Ok(mut server) = TcpStream::connect(upstream).await else {
                            return;
                        };
                        let _ = tokio::io::copy_bidirectional(&mut client, &mut server).await;
                    });
                    relays.lock().unwrap().push(relay.abort_handle());
                }
            }
        });
        Self {
            addr,
            accepting,
            relayed,
            refused,
            relays,
            accept,
        }
    }

    /// A `redis://` URL naming the proxy and the database `TEST_VALKEY_URL`
    /// names, which the client selects on every connection it opens.
    fn url(&self) -> String {
        match test_valkey_db() {
            Some(db) => format!("redis://{}/{db}", self.addr),
            None => format!("redis://{}", self.addr),
        }
    }

    /// Cut every relayed connection, and refuse new ones until
    /// [`Self::accept_again`].
    fn cut_and_refuse(&self) {
        self.accepting.store(false, SeqCst);
        for relay in self.relays.lock().unwrap().drain(..) {
            relay.abort();
        }
    }

    fn accept_again(&self) {
        self.accepting.store(true, SeqCst);
    }

    fn relayed(&self) -> usize {
        self.relayed.load(SeqCst)
    }

    fn refused(&self) -> usize {
        self.refused.load(SeqCst)
    }
}

impl Drop for SeveringProxy {
    fn drop(&mut self) {
        self.accept.abort();
        for relay in self.relays.lock().unwrap().drain(..) {
            relay.abort();
        }
    }
}

/// A serving replica whose every store connection is cut and whose
/// reconnects are refused: about 30 s after its last read it goes stale, and
/// it exits nothing while the outage lasts, past the stale entry. Once the
/// proxy accepts again, the client reconnects on its own (backoff capped at
/// 1 s), the next hold read succeeds, and the replica exits 0. Once the proxy
/// has relayed as many connections after the cut as before it, and for a few
/// seconds more, the two counts are equal: no client of the hold's own made
/// its first connection after the cut, which it would do through the same
/// proxy. Two clients of the hold's own go unseen. One whose only connect
/// attempt fell inside the outage, never retried, opens no connection; it
/// serves no read, so a hold reading through it would fail the exit wait
/// instead. One connected before the cut is in the count before it and
/// reconnects one for one with the replica's client.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replica_whose_store_connections_are_cut_goes_stale_and_exits_zero_once_a_read_succeeds()
{
    let tv = TestValkey::setup().await;
    let proxy = SeveringProxy::start().await;
    let mut config = crate::common::test_config();
    config.valkey_url = proxy.url();
    config.valkey_command_timeout_ms = COMMAND_TIMEOUT_MS;
    config.valkey_reconnect_max_delay_ms = RECONNECT_MAX_DELAY_MS;
    let client = connect_test_client(&config)
        .await
        .expect("the replica's client connects through the proxy");

    let initial = tokio::time::timeout(
        SETTLE,
        evaluate_hold_at_startup(&client, tv.key_schema(), answering(COMMAND_TIMEOUT_MS)),
    )
    .await
    .expect("the startup evaluation returns")
    .expect("the startup evaluation answers");
    assert!(!initial.verdict.is_held(), "fixture: no marker");
    let shutdown = CancellationToken::new();
    let (verdicts, _poll) = spawn_hold_poll(
        client.clone(),
        tv.key_schema().clone(),
        HoldPollSettings::new(POLL_INTERVAL, COMMAND_TIMEOUT_MS),
        initial,
        shutdown.clone(),
    );
    let point = HoldServingPoint::new();
    let (calls, mut exits) = mpsc::unbounded_channel();
    let _reactor = spawn_hold_reactor(
        verdicts.clone(),
        point.clone(),
        shutdown.clone(),
        HoldReactorSettings {
            heartbeat_interval: Duration::from_secs(300),
            exit: Arc::new(RecordingExit { calls }),
            verdict_stale: watch::Sender::new(false),
            held: watch::Sender::new(false),
        },
    );
    let gate = HoldGate::new(verdicts.clone());
    assert!(point.reach(HoldEntryParts {
        gate: gate.clone(),
        streams: GracefulShutdown::new(),
        writers: Arc::new(HoldWriterAborts::new()),
        metrics: Arc::new(crate::common::metrics_scrape_only("hold-poll-reconnect")),
    }));
    // Reads are answering: a publication after the startup one.
    let mut published = verdicts.clone();
    tokio::time::timeout(SETTLE, published.changed())
        .await
        .expect("a hold read answers through the proxy")
        .expect("the poll runs");
    let connections_before = proxy.relayed();

    // The outage: every connection cut, every reconnect refused.
    proxy.cut_and_refuse();
    let cut = Instant::now();
    while gate.state() != HoldGateState::Stale {
        assert!(
            cut.elapsed() < STALE_DEADLINE,
            "no stale entry within {STALE_DEADLINE:?} of the cut"
        );
        assert!(
            exits.try_recv().is_err(),
            "a replica whose reads fail does not exit"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(AFTER_STALE).await;
    assert!(
        exits.try_recv().is_err(),
        "a stale replica exits nothing while the store stays unreachable"
    );
    assert!(
        proxy.refused() > 0,
        "the client tried to reconnect during the outage"
    );

    // The store reachable again: the client comes back by itself.
    proxy.accept_again();
    let reopened = Instant::now();
    let (code, _at) = tokio::time::timeout(EXIT_DEADLINE, exits.recv())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "no exit within {EXIT_DEADLINE:?} of the store accepting again ({} connections \
                 relayed since)",
                proxy.relayed() - connections_before
            )
        })
        .expect("the reactor keeps its exit seam");
    assert_eq!(code, 0, "a stale replica whose read succeeds exits 0");
    assert_eq!(
        gate.state(),
        HoldGateState::Stale,
        "the gate never reopens in place"
    );
    // The exit can come once one member reads, before the others are back.
    // Wait until the proxy has relayed as many connections since the cut as
    // before it, then a little longer, so a client of the hold's own whose
    // first connection opens after the cut would show as one more. A client
    // of the hold's own connected before the cut is in `connections_before`
    // and is not seen here.
    let reconnects = || proxy.relayed() - connections_before;
    while reconnects() < connections_before {
        assert!(
            reopened.elapsed() < RECONNECT_DEADLINE,
            "only {} of the client's {connections_before} connections came back within \
             {RECONNECT_DEADLINE:?} of the store accepting again",
            reconnects()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(RECONNECT_SETTLE).await;
    assert_eq!(
        reconnects(),
        connections_before,
        "the proxy relayed as many connections after the cut as before it: no client of the \
         hold's own made its first connection after the cut"
    );
    shutdown.cancel();
}
