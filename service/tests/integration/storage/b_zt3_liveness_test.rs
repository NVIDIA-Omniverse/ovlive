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

//! b.zt3 — what the unresponsive verdict does NOT close, how it tells its
//! verdicts apart, what it records, and that a connection whose bounded wait
//! expired is not waited on again.
//!
//! The fast verdict closes a connection only when it has been silent past the
//! fast bound while the same slot range answered another connection (the
//! peer evidence). These tests stage, through a [`ScriptReplyProxy`] in front
//! of the real store, the cases where it must stay its hand, plus the
//! classifier, the WARN a verdict logs, and a fred `Transaction` whose `EXEC`
//! read expired:
//!
//! | Test | Design item | What it pins |
//! |---|---|---|
//! | [`a_store_answering_no_connection_for_three_seconds_gets_no_verdict_and_every_queued_command_is_answered`] | T3 | the proxy holds every reply on every connection for 3 s (a shard busy with a long script answers nobody): with commands queued on both data members, no verdict, and every command is answered once the hold ends |
//! | [`a_reply_whose_bytes_keep_arriving_for_three_seconds_gets_no_verdict`] | T4 | a 48 KiB reply trickled over about 3 s while another member's reads are answered: reply bytes arriving are progress, so no verdict, and the value arrives whole |
//! | [`connections_silent_while_no_connection_answers_get_the_slow_verdict_and_no_replacement`] | T5 | both data members silent at once, twice over: each connection is closed by the slow verdict (here a 3 s test bound), never the fast one, and two slow verdicts on a member replace nothing |
//! | [`a_transaction_whose_exec_read_expired_is_replayed_without_waiting_again_on_the_failed_connection`] | T-X (and T8's `Transaction` half) | a `txn` checkout's `MULTI`/`GET`/`EXEC` stalled at the `GET`: the `EXEC` read expires at a 4 s test bound, the `DISCARD` on that connection fails at once, and the replay on a fresh connection and a follow-up command finish within the bound + 2 s; one `(expired ` WARN, the marker `GET` seen twice, `unresponsive_total{pool="txn",verdict="slow"}` +1 |
//! | [`verdict_of_tells_fast_from_slow_and_keeps_stock_freds_text_for_slow`] | T12 | the classifier and the texts; pure |
//! | [`a_fast_verdict_logs_one_warn_carrying_every_field_the_design_lists`] | T19 | the fred WARN of one fast verdict, field by field, with the kernel's socket row |
//!
//! The busy store of T3 is a proxy hold rather than a real Lua busy-loop: a
//! script that busies the shared test store for 3 s stalls every concurrent
//! test's commands, a node-wide disturbance the test-writing guide keeps off
//! the shared server. From the verdict's point of view the two are the same —
//! no connection of the shard gets a reply. The other half of T3, a second
//! node's replies never counting as evidence for the first, needs two nodes
//! and lives with the cluster facade in `stream_mux_reader_connect_test.rs`.
//!
//! T19's other half — the kernel fields reading `unavailable` when the socket
//! has no `/proc/self/net/tcp` row — is not covered: every connection a test
//! can make here is a loopback TCP socket, which always has a row.
//!
//! The guard on expired connections sits in four I/O wrappers in vendored
//! fred (`live_edit::liveness::{send, feed, flush, next}`), which has no unit
//! harness of its own, so each site is pinned through an integration test
//! that reaches it after an expiry:
//!
//! - `send`: T-X here (the `DISCARD`'s write on the connection whose `EXEC`
//!   read expired).
//! - `flush` and `feed`: the expired-flush tests in
//!   `stream_mux_reader_connect_test.rs`, which need a proxy that stops reading
//!   a connection (`ProxyAction::StopReading`; `Stall` keeps draining client
//!   bytes, so a flush through it never expires) and the two-node facade (the
//!   flush that expires is the router's flush of every connection holding fed
//!   frames before a write to the other node).
//! - `next`: not pinned. Every path to a read on a connection runs a write or
//!   flush on it first (`drain_all` flushes before it drains; a transaction's
//!   `DISCARD`, an `ASKING` retry and a cluster-wide command write before they
//!   read), so with the other guards in place a read on an expired connection
//!   is never reached, and removing the `next` guard alone changes nothing a
//!   test can observe.
//!
//! The close of a connection judged dead (rev 4, `liveness::close`) skips
//! fred's flush and TLS shutdown on either of two markers:
//!
//! - `expired`: pinned by the expired-flush tests, whose closes would
//!   otherwise wait out fred's 5 s close bound on the 16 MiB still buffered.
//! - `closed_by` (a read-loop verdict): not pinned. On loopback TCP, a
//!   read-loop verdict closes a connection holding bytes its kernel refuses
//!   only when the stall crosses its bound inside one router step, between a
//!   feed and the next flush; no fixture can time that. The other case it
//!   covers, TLS's `close_notify`, needs a TLS store, which this suite does
//!   not have. The one assignment in `defer` is left to code review.
//!
//! The 3 s windows are staging (how long the store stays busy, how long the
//! reply trickles), each longer than the 2 s fast bound by construction. The
//! one deadline, in T-X, separates one expired bound from two; it is not a
//! latency measurement.
//!
//! # Wait budget
//!
//! Every wait at its full bound — the `TestValkey` fixture's connect (30 s),
//! the proxied client's (30 s), fred's 5 s command timeout per command (12 s
//! in T-X, which raises it), the staging windows, a polled verdict (10 s
//! each), the closing `quit()` (11.5 s, the sum of its internal bounds: it has
//! no outer one) and the key purge (10 s):
//!
//! | Test | Worst case |
//! |---|---|
//! | `a_store_answering_no_connection_…` | 30 + 5 × 6 + 30 + hold 20 + 3 + 5 + quit 11.5 + purge 10 = 139.5 s |
//! | `a_reply_whose_bytes_keep_arriving_…` | 30 + 5 + 30 + 5 + quit 11.5 + purge 10 = 91.5 s |
//! | `connections_silent_while_no_connection_answers_…` | 30 + 5 × 4 + 30 + round 10 × 2 + quit 11.5 + purge 10 = 121.5 s (a round is its GETs, 5 s each, beside its verdict waits, 10 s each, all concurrent) |
//! | `a_transaction_whose_exec_read_expired_…` | 30 + 5 + 30 + checkout 10 (its probe and a worst-case slot repair) + `EXEC` 12 + `PING` 12 + polled metric 10 + quit 11.5 + purge 10 = 130.5 s |
//! | `verdict_of_…` | none |
//! | `a_fast_verdict_logs_one_warn_…` | 30 + 5 + 30 + 5 + quit 11.5 + purge 10 = 91.5 s |
//!
//! under the 180 s kill deadline `.config/nextest.toml` sets for the `b_zt3_`
//! modules.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::error::{Error, ErrorKind};
use fred::interfaces::{ClientLike, EventInterface, KeysInterface, TransactionInterface};
use fred::live_edit::liveness::{verdict_of, Verdict, UNRESPONSIVE, UNRESPONSIVE_FAST};
use fred::types::config::ConnectionConfig;
use tokio::sync::broadcast::error::TryRecvError;
use tokio::time::Instant;

use live_edit::observability::metrics::{
    Metrics, VALKEY_MEMBER_REPLACEMENT_OUTCOMES, VALKEY_POOL_DATA, VALKEY_POOL_TXN,
    VALKEY_UNRESPONSIVE_VERDICT_FAST, VALKEY_UNRESPONSIVE_VERDICT_SLOW,
};
use live_edit::storage::client::{LivenessOverrides, ValkeyClient};
use live_edit::storage::test_support::TestValkey;

use super::b_zt3_support::{
    await_until, bridge_fred_log, messages_containing, proxied_client, proxied_client_with,
    stall_marker_gets, write_keys, PeerReads, MEMBER_REPLACEMENTS,
    UNRESPONSIVE as UNRESPONSIVE_TOTAL,
};
use super::mux_test_helpers::read_series;
use super::snapshot_read_support::{
    test_valkey_addr, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};
use crate::grpc::log_capture::{purge_keys, ThreadCapture};

/// How long the staged busy store answers nobody, and about how long the
/// trickled reply takes: past the 2 s fast bound.
const STAGED: Duration = Duration::from_secs(3);

/// Bound on a verdict arriving on a member's error stream.
const VERDICT_WAIT: Duration = Duration::from_secs(10);

/// T5's slow bound: short enough to wait out, past the fast bound and its
/// evidence lag so the fast rule would have had its chance.
const TEST_SLOW_BOUND: Duration = Duration::from_secs(3);

/// T-X's slow bound, the bound on each wait on one connection's I/O: below
/// fred's 10 s `internal_command_timeout`, so without the guard on expired
/// connections the `DISCARD`'s read would wait this bound out a second time.
const EXPIRY_TEST_SLOW_BOUND: Duration = Duration::from_secs(4);

/// T-X's slack over one [`EXPIRY_TEST_SLOW_BOUND`] for the reconnect, the
/// replay and a follow-up command: well under a second bound.
const EXPIRY_TEST_SLACK: Duration = Duration::from_secs(2);

/// T-X's command timeout, in place of the 5 s default: above two expired
/// bounds plus the replay, so the transaction completes either way and the
/// test's own assertions, not the command timeout, tell one wait from two.
const EXPIRY_TEST_COMMAND_TIMEOUT: Duration = Duration::from_secs(12);

/// The value T-X's marker key holds, so the replayed `GET` is seen returning
/// it.
const MARKER_VALUE: &str = "zt3-marker-value";

fn data_verdicts(metrics: &Metrics, verdict: &str) -> i64 {
    read_series(
        metrics,
        UNRESPONSIVE_TOTAL,
        &[("pool", VALKEY_POOL_DATA), ("verdict", verdict)],
    )
}

/// Whether `client`'s error stream has received nothing.
fn quiet(
    errors: &mut tokio::sync::broadcast::Receiver<(Error, Option<fred::types::config::Server>)>,
) -> Result<(), String> {
    match errors.try_recv() {
        Err(TryRecvError::Empty) => Ok(()),
        other => Err(format!("{other:?}")),
    }
}

async fn hooked(client: &ValkeyClient) -> Arc<Metrics> {
    let metrics = Arc::new(crate::common::metrics_scrape_only("zt3"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    metrics
}

/// T3: a store that answers no connection — the proxy holds every reply on
/// every connection of the client for [`STAGED`] — closes none of them: with
/// no peer evidence the fast verdict never fires, whatever the stall, and the
/// queued commands are all answered once the store answers again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_store_answering_no_connection_for_three_seconds_gets_no_verdict_and_every_queued_command_is_answered(
) {
    let tv = TestValkey::setup().await;
    let keys: Vec<String> = (0..6)
        .map(|i| format!("{}zt3:busy:{i}", tv.prefix()))
        .collect();
    write_keys(tv.client(), &keys, "v").await;
    let busy = Arc::new(AtomicBool::new(false));
    let gate = ReplyGate::new();
    let rule: ProxyRule = {
        let (busy, gate) = (Arc::clone(&busy), gate.clone());
        Arc::new(move |_args: &[Vec<u8>]| {
            busy.load(Ordering::SeqCst)
                .then(|| ProxyAction::HoldReplies(gate.clone()))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = proxied_client(&proxy, 2, LivenessOverrides::default()).await;
    let metrics = hooked(&client).await;
    let members: [Client; 2] = [client.pool.member(0).clone(), client.pool.member(1).clone()];
    let mut errors = [members[0].error_rx(), members[1].error_rx()];

    busy.store(true, Ordering::SeqCst);
    let reads = async {
        let per_member = members.iter().enumerate().map(|(m, member)| {
            let keys = &keys[m * 3..m * 3 + 3];
            futures::future::join_all(keys.iter().map(|key| member.get::<Option<String>, _>(key)))
        });
        futures::future::join_all(per_member).await
    };
    let release = async {
        gate.reply_held().await;
        tokio::time::sleep(STAGED).await;
        busy.store(false, Ordering::SeqCst);
        gate.open();
    };
    let (replies, ()) = tokio::join!(reads, release);

    for (m, member_replies) in replies.into_iter().enumerate() {
        for reply in member_replies {
            assert_eq!(
                reply
                    .unwrap_or_else(|e| panic!("member {m}'s queued GET must be answered: {e:?}"))
                    .as_deref(),
                Some("v")
            );
        }
    }
    for (m, errors) in errors.iter_mut().enumerate() {
        quiet(errors).unwrap_or_else(|e| {
            panic!("no connection of member {m} may be closed while the store answers nobody: {e}")
        });
    }
    assert_eq!(data_verdicts(&metrics, VALKEY_UNRESPONSIVE_VERDICT_FAST), 0);
    assert_eq!(data_verdicts(&metrics, VALKEY_UNRESPONSIVE_VERDICT_SLOW), 0);
    assert!(proxy.hits() >= 6, "every queued GET must have been held");
    client.quit().await;
}

/// T4: a reply whose bytes keep arriving — 48 KiB, delivered 512 bytes per
/// 16 ms, about 3 s — while the other member's reads are answered (peer
/// evidence present all along) is never cut: arriving bytes are read
/// progress, so the connection is not silent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reply_whose_bytes_keep_arriving_for_three_seconds_gets_no_verdict() {
    const VALUE_LEN: usize = 48 * 1024;
    const PIECE: usize = 512;
    let piece_every = STAGED / (VALUE_LEN / PIECE) as u32;
    let tv = TestValkey::setup().await;
    let key = format!("{}zt3:trickled", tv.prefix());
    let peer_key = format!("{}zt3:peer", tv.prefix());
    let value = "x".repeat(VALUE_LEN);
    write_keys(tv.client(), std::slice::from_ref(&key), &value).await;
    let trickled = AtomicBool::new(false);
    let rule: ProxyRule = {
        let key = key.clone();
        Arc::new(move |args: &[Vec<u8>]| {
            let hit = args.first()?.eq_ignore_ascii_case(b"GET")
                && args.get(1).is_some_and(|k| k == key.as_bytes())
                && !trickled.swap(true, Ordering::SeqCst);
            hit.then_some(ProxyAction::TrickleReplies {
                chunk: PIECE,
                every: piece_every,
            })
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = proxied_client(&proxy, 2, LivenessOverrides::default()).await;
    let _peer = PeerReads::start(client.pool.member(1).clone(), peer_key);
    let reader = client.pool.member(0);
    let mut errors = reader.error_rx();

    let got: Option<String> = reader
        .get(&key)
        .await
        .expect("the trickled GET must be answered within the command timeout");

    assert_eq!(
        got.map(|v| v.len()),
        Some(VALUE_LEN),
        "the value must arrive whole"
    );
    quiet(&mut errors).unwrap_or_else(|e| {
        panic!("a connection whose reply keeps arriving must not be closed: {e}")
    });
    assert_eq!(proxy.hits(), 1, "the GET's reply must have been trickled");
    client.quit().await;
}

/// T5: both data members' connections silent at once — each carrying a
/// stalled `GET`, nothing else answered on any connection — is a store
/// silent to everyone: no peer evidence, so each is closed by the slow
/// verdict (at a 3 s test bound), never the fast one. Twice over, so member 0
/// gets two verdicts; slow verdicts do not count toward replacement, so its
/// client is kept.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn connections_silent_while_no_connection_answers_get_the_slow_verdict_and_no_replacement() {
    let tv = TestValkey::setup().await;
    let markers: Vec<String> = (0..4)
        .map(|i| format!("{}zt3:silent:{i}", tv.prefix()))
        .collect();
    write_keys(tv.client(), &markers, "v").await;
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), stall_marker_gets(&markers)).await;
    let client = proxied_client(
        &proxy,
        2,
        LivenessOverrides {
            slow_timeout: Some(TEST_SLOW_BOUND),
            ..LivenessOverrides::default()
        },
    )
    .await;
    let metrics = hooked(&client).await;
    let ids: Vec<String> = (0..2)
        .map(|m| ClientLike::id(client.pool.member(m)).to_owned())
        .collect();

    for round in 0..2 {
        let members = [client.pool.member(0).clone(), client.pool.member(1).clone()];
        let mut errors = [members[0].error_rx(), members[1].error_rx()];
        let gets = futures::future::join_all(members.iter().enumerate().map(|(m, member)| {
            let marker = markers[round * 2 + m].clone();
            async move { member.get::<Option<String>, _>(marker).await }
        }));
        let verdicts = futures::future::join_all(errors.iter_mut().map(|errors| async {
            tokio::time::timeout(VERDICT_WAIT, errors.recv())
                .await
                .expect("the silent connection must be closed within the bound")
                .expect("the error stream must stay open")
                .0
        }));
        // The GETs' own outcome is not this test's subject: each is re-sent
        // after its verdict, near the command timeout.
        let (_gets, verdicts) = tokio::join!(gets, verdicts);
        for (m, error) in verdicts.iter().enumerate() {
            assert_eq!(
                verdict_of(error),
                Some(Verdict::Slow),
                "round {round}: member {m}'s connection, silent with nothing answered anywhere, \
                 must be closed by the slow verdict, not the fast one: {error:?}"
            );
        }
    }

    assert_eq!(proxy.hits(), 4, "every marker GET must have been stalled");
    assert_eq!(data_verdicts(&metrics, VALKEY_UNRESPONSIVE_VERDICT_FAST), 0);
    assert_eq!(data_verdicts(&metrics, VALKEY_UNRESPONSIVE_VERDICT_SLOW), 4);
    for outcome in VALKEY_MEMBER_REPLACEMENT_OUTCOMES {
        assert_eq!(
            read_series(
                &metrics,
                MEMBER_REPLACEMENTS,
                &[("pool", VALKEY_POOL_DATA), ("outcome", *outcome)]
            ),
            0,
            "slow verdicts must not lead to a replacement ({outcome})"
        );
    }
    assert!(!client.member_replacement_in_flight_for_test(VALKEY_POOL_DATA));
    assert_eq!(
        client.member_replacement_tasks_for_test(),
        0,
        "no replacement started, so no replacement or retirement is running"
    );
    for (m, id) in ids.iter().enumerate() {
        assert_eq!(
            ClientLike::id(client.pool.member(m)),
            id,
            "member {m} keeps its client after two slow verdicts"
        );
    }
    client.quit().await;
}

/// T-X (b.zt3 rev 3; also T8's fred-`Transaction` half): once a bounded wait
/// on a connection has expired, nothing waits on that connection again.
///
/// A fred `Transaction` (`MULTI`, `GET <marker>`, `EXEC`) runs on a `txn` slot
/// taken through `acquire_txn_for`, the production checkout, with the slow
/// bound at [`EXPIRY_TEST_SLOW_BOUND`] and fred's `internal_command_timeout` at
/// its default 10 s, as in the service. The proxy stalls the connection at the
/// marker `GET`, so the `EXEC`'s reply read expires at the bound and marks the
/// connection failed. fred then sends `DISCARD` on that connection; it must
/// fail at once rather than wait out another bound, so the router reconnects
/// and replays the transaction on a fresh connection, where the `GET` is
/// forwarded. Without the guard the `DISCARD`'s read waits a second bound and
/// logs a second `(expired ` WARN, and the replay lands after about twice the
/// bound. The client's command timeout is raised to
/// [`EXPIRY_TEST_COMMAND_TIMEOUT`] so the transaction completes either way and
/// those two observations decide the result.
///
/// The [`EXPIRY_TEST_SLACK`] deadline separates one bound from two; it is
/// not a latency measurement. Current-thread runtime, so the thread-local
/// capture sees fred's router task.
#[tokio::test]
async fn a_transaction_whose_exec_read_expired_is_replayed_without_waiting_again_on_the_failed_connection(
) {
    bridge_fred_log();
    let capture = ThreadCapture::start();
    let tv = TestValkey::setup().await;
    let marker = format!("{}zt3:expired-exec", tv.prefix());
    write_keys(tv.client(), std::slice::from_ref(&marker), MARKER_VALUE).await;
    let marker_gets = Arc::new(AtomicUsize::new(0));
    let rule: ProxyRule = {
        let stall_once = stall_marker_gets(std::slice::from_ref(&marker));
        let (marker, gets) = (marker.clone(), Arc::clone(&marker_gets));
        Arc::new(move |args: &[Vec<u8>]| {
            let is_marker_get = args.first().is_some_and(|n| n.eq_ignore_ascii_case(b"GET"))
                && args.get(1).is_some_and(|key| key == marker.as_bytes());
            if is_marker_get {
                gets.fetch_add(1, Ordering::SeqCst);
            }
            stall_once(args)
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = proxied_client_with(
        &proxy,
        1,
        LivenessOverrides {
            slow_timeout: Some(EXPIRY_TEST_SLOW_BOUND),
            ..LivenessOverrides::default()
        },
        |config| {
            config.valkey_command_timeout_ms = EXPIRY_TEST_COMMAND_TIMEOUT.as_millis() as u64;
        },
    )
    .await;
    let metrics = hooked(&client).await;
    let txn_slow_verdicts = || {
        read_series(
            &metrics,
            UNRESPONSIVE_TOTAL,
            &[
                ("pool", VALKEY_POOL_TXN),
                ("verdict", VALKEY_UNRESPONSIVE_VERDICT_SLOW),
            ],
        )
    };
    let slow_before = txn_slow_verdicts();
    let checkout = client.acquire_txn_for(&marker).await;
    let id = ClientLike::id(&*checkout).to_owned();
    let internal_command_timeout = checkout.connection_config().internal_command_timeout;
    assert_eq!(
        internal_command_timeout,
        ConnectionConfig::default().internal_command_timeout,
        "fixture self-check: fred's internal command timeout is its default, as in the service"
    );
    assert!(
        internal_command_timeout > EXPIRY_TEST_SLOW_BOUND,
        "fixture self-check: the slow bound, not the internal command timeout, bounds the \
         DISCARD's read without the guard"
    );
    assert!(
        EXPIRY_TEST_COMMAND_TIMEOUT > EXPIRY_TEST_SLOW_BOUND * 2 + EXPIRY_TEST_SLACK,
        "fixture self-check: even two expired bounds and the replay fit in the command \
         timeout, so the WARN count and the deadline decide the result, not the timeout"
    );

    let sent = Instant::now();
    let trx = checkout.multi();
    let _: () = trx
        .get(&marker)
        .await
        .expect("queueing a GET into the transaction does no I/O");
    let replayed: Result<Vec<Option<String>>, _> = trx.exec(true).await;
    let follow_up: Result<String, _> = checkout.ping(None).await;
    let took = sent.elapsed();

    let captured = capture.snapshot();
    let expired: Vec<String> = messages_containing(&captured, "Unresponsive connection to ")
        .into_iter()
        .filter(|line| line.starts_with(&format!("{id}: ")) && line.contains("(expired "))
        .collect();
    assert_eq!(
        expired.len(),
        1,
        "one expired wait, one WARN; nothing waits on the failed connection again: {expired:#?}"
    );
    assert!(
        took <= EXPIRY_TEST_SLOW_BOUND + EXPIRY_TEST_SLACK,
        "one expired bound, then the replay and a follow-up, must fit in {:?}: took {took:?}",
        EXPIRY_TEST_SLOW_BOUND + EXPIRY_TEST_SLACK
    );
    assert_eq!(
        replayed.expect("the transaction must be replayed and answered on a fresh connection"),
        [Some(MARKER_VALUE.to_owned())]
    );
    follow_up.expect("the checked-out client must serve a follow-up command");
    assert_eq!(
        marker_gets.load(Ordering::SeqCst),
        2,
        "the marker GET must reach the proxy twice: stalled, then replayed and forwarded"
    );
    await_until(
        "the failed connection's close must tick unresponsive_total{pool=\"txn\",verdict=\"slow\"}",
        Instant::now() + VERDICT_WAIT,
        || txn_slow_verdicts() > slow_before,
    )
    .await;
    assert_eq!(
        txn_slow_verdicts() - slow_before,
        1,
        "one failed connection, one slow verdict"
    );
    drop(checkout);
    client.quit().await;
    purge_keys(&tv).await;
}

/// T12: `verdict_of` reads the verdict off an IO error's text; the slow
/// verdict (and every expired bounded wait, which shares it) keeps stock
/// fred's text; anything else is no verdict.
#[test]
fn verdict_of_tells_fast_from_slow_and_keeps_stock_freds_text_for_slow() {
    assert_eq!(
        UNRESPONSIVE, "Unresponsive connection.",
        "stock fred's text"
    );
    assert_ne!(UNRESPONSIVE_FAST, UNRESPONSIVE);
    assert_eq!(
        verdict_of(&Error::new(ErrorKind::IO, UNRESPONSIVE_FAST)),
        Some(Verdict::Fast)
    );
    assert_eq!(
        verdict_of(&Error::new(ErrorKind::IO, UNRESPONSIVE)),
        Some(Verdict::Slow)
    );
    for (kind, text) in [
        (ErrorKind::Timeout, UNRESPONSIVE_FAST),
        (ErrorKind::Unknown, UNRESPONSIVE),
        (ErrorKind::IO, "Connection reset by peer"),
        (ErrorKind::IO, "unresponsive connection."),
        (ErrorKind::Canceled, ""),
    ] {
        assert_eq!(
            verdict_of(&Error::new(kind.clone(), text)),
            None,
            "{kind:?} {text:?} is no verdict"
        );
    }
    assert_eq!(Verdict::Fast.as_str(), VALKEY_UNRESPONSIVE_VERDICT_FAST);
    assert_eq!(Verdict::Slow.as_str(), VALKEY_UNRESPONSIVE_VERDICT_SLOW);
}

/// T19: one fast verdict logs one fred WARN naming the closed connection's
/// client and node and carrying every field the design lists — the event,
/// this connection's queue and buffers, this client's counter and channel,
/// the other connections, and the kernel's view of the socket, read from its
/// `/proc/self/net/tcp` row. The line is the deliverable (it is what an
/// operator reads to place the stuck bytes), so it is pinned field by field.
/// Current-thread runtime, so the thread-local capture sees fred's router
/// tasks.
#[tokio::test]
async fn a_fast_verdict_logs_one_warn_carrying_every_field_the_design_lists() {
    bridge_fred_log();
    let capture = ThreadCapture::start();
    let tv = TestValkey::setup().await;
    let marker = format!("{}zt3:logged", tv.prefix());
    write_keys(tv.client(), std::slice::from_ref(&marker), "v").await;
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        stall_marker_gets(std::slice::from_ref(&marker)),
    )
    .await;
    let client = proxied_client(&proxy, 2, LivenessOverrides::default()).await;
    let _peer = PeerReads::start(
        client.pool.member(1).clone(),
        format!("{}zt3:peer", tv.prefix()),
    );
    let silent = client.pool.member(0);
    let id = ClientLike::id(silent).to_owned();

    let _: Option<String> = silent
        .get(&marker)
        .await
        .expect("the stalled GET must be rescued");

    let captured = capture.snapshot();
    let lines = messages_containing(&captured, "(fast verdict): no reply progress for ");
    assert_eq!(
        lines.len(),
        1,
        "one WARN per verdict; full capture:\n{}",
        captured.text
    );
    let line = &lines[0];
    let event = captured
        .events
        .iter()
        .find(|e| e["fields"]["message"].as_str() == Some(line.as_str()))
        .expect("the line's event");
    assert_eq!(event["level"], "WARN", "{event}");
    assert!(
        line.starts_with(&format!("{id}: Unresponsive connection to ")),
        "the line names the client and the node: {line}"
    );
    for field in [
        "(fast verdict): no reply progress for ",
        "newest reply from the same slot range on another connection: ",
        "This connection: 1 in flight; oldest outstanding command GET, sent ",
        " commands fed since the last send or flush, ",
        " bytes in the write buffer, feed count ",
        " bytes in the read buffer, last read progress ",
        "subscribed false, 0 pubsub-shaped frames received; ",
        "longest wait since the stall began: ",
        "longest time a command sent since the stall began sat fed before it was sent: ",
        "This client: command-queue counter ",
        ", command channel length ",
        " when the router last waited, last command taken by the router ",
        "Other connections: ",
        "Kernel socket: tx_queue ",
        " bytes, rx_queue ",
        " bytes, timer ",
        ", retrnsmt ",
        "Closing it and reconnecting.",
    ] {
        assert!(
            line.contains(field),
            "the verdict line must carry `{field}`: {line}"
        );
    }
    assert!(
        !line.contains("Kernel socket: unavailable"),
        "a loopback TCP socket has a /proc/self/net/tcp row: {line}"
    );
    client.quit().await;
    purge_keys(&tv).await;
}
