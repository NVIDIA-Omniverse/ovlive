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

//! b.19i — a stream reader re-AUTHs its own connection, between its reads.
//!
//! Under a credential provider fred schedules no periodic re-AUTH on a reader
//! client (its config carries the provider cadence-wrapped). Each reader takes
//! the provider's tick itself: `fetch(None)`, gated by the provider as fred's
//! pooled refresh is; with a credential due and a read in flight it sends that
//! read a credential wake, and it sends `AUTH` on its own connection only with
//! no read in flight. Before b.19i fred's refresh task wrote the `AUTH` to the
//! reader's connection itself, racing the reader's `XREAD BLOCK`: it interrupted
//! the read through `CLIENT UNBLOCK` on a backchannel (an `-UNBLOCKED` re-plan)
//! or, losing the race, queued behind a fresh 30 s BLOCK.
//!
//! # Fixtures
//!
//! * **A counting proxy** ([`Wire`]): a [`ScriptReplyProxy`] in front of the
//!   one standalone test store that forwards every frame and counts the `AUTH`
//!   and `CLIENT UNBLOCK` frames it relays — the wire-level evidence of which
//!   `AUTH`s went out, and that no backchannel `CLIENT UNBLOCK` did.
//! * **A scripted provider** ([`ScriptedProvider`]), installed on every
//!   connection through `TestValkey::setup_with_url_readers_and_credential_provider`.
//!   Its handshake answer is always `default` / a password (the store's
//!   `default` user has no password, so any password authenticates). Its
//!   periodic answer carries a password only for the credentials the test
//!   [grants](ScriptedProvider::grant), so a test decides exactly which tick
//!   sends. Its cadence is off until the test starts it, after the fixture
//!   connected: the pools, connected first, get no fred refresh task unless a
//!   router restarts later. Whether fred schedules one on a reader is read
//!   off the reader client itself ([`assert_no_fred_refresh_on_reader`]).
//! * **The real gate**: [`a_reader_follows_the_rotation_gate_no_auth_before_a_rotation_and_auth_inside_its_window`]
//!   installs a `TokenFileCredentialProvider` on a sub-second band instead.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`a_reauth_due_during_a_blocked_read_wakes_it_and_auths_between_reads`] | busy reader: a credential wake, one `AUTH` on the wire carrying the fetched credential, answered well inside the BLOCK; no `CLIENT UNBLOCK`, no `-UNBLOCKED`; no fred refresh scheduled on the reader's client; nothing lost |
//! | [`a_reader_follows_the_rotation_gate_no_auth_before_a_rotation_and_auth_inside_its_window`] | the real gate: no `AUTH` and no wake while the token is the first one seen (b.19i's first-sighting rule); after a rotation, `AUTH`s inside the window, every one carrying the rotated token, at least one through a credential wake, and no `CLIENT UNBLOCK` |
//! | [`a_failed_credential_fetch_counts_failed_and_sends_nothing`] | a provider error: `failed` +1, no `AUTH`, no wake |
//! | [`a_readers_reauth_schedule_follows_its_connection`] | M1: no schedule before the reader connects (the provider ticking all along) or under `static`; set once connected; cleared at `quit()` |
//! | `faults::an_idle_readers_reauth_is_sent_at_once_with_no_wake` | idle reader: `AUTH` with no wake and no read |
//! | `faults::a_failed_credential_wake_leaves_the_auth_for_when_the_read_returns` | 3× `Wake` faults: WARN `reason="credential"`, `AUTH` once the read returns |
//! | `faults::a_failed_reauth_counts_failed_warns_once_and_keeps_the_connection` | `Reauth` `Err`: `failed` +1, one WARN naming the error, no forced reconnect, no budget charge, the subscription kept |
//! | `faults::a_hung_reauth_is_cut_at_the_probe_timeout_and_counts_failed` | `Reauth` `Hang`: the same, its WARN naming the shortened `probe_timeout` that cut it |
//! | `faults::a_restarted_reader_still_holding_its_connection_keeps_reauthing` | a reader restarted after a panic, its connection held, schedules its ticks again |
//! | `faults::a_hung_credential_fetch_is_cut_at_the_probe_timeout_and_reads_carry_on` | a provider that never answers: the fetch is cut at the shortened `probe_timeout` (its DEBUG line names it), `failed` +1, and the reader goes on delivering |
//! | `faults::a_credential_due_while_a_wake_is_outstanding_sends_no_second_wake` | one wake per in-flight read: with a coverage wake outstanding, a due credential sends none; its `AUTH` goes out when the read returns |
//! | `faults::a_reauth_due_while_the_reader_backs_off_is_sent_without_waiting_out_the_backoff` | a tick due during a backoff sends its `AUTH` at once, with no read |
//! | `faults::no_reauth_credential_reaches_the_logs` | a distinctive password never appears in any log line, across the ok, failed-`AUTH` and failed-fetch paths |
//!
//! The cluster twins are in `stream_mux_reader_connect_test.rs` (the two-node
//! facade) and `cluster_subscriber_test.rs` (a real cluster).
//!
//! # Waits
//!
//! Every wait is a deadline ([`SETTLE`], 10 s), not a measurement. "Answered
//! well inside the BLOCK" is that deadline against the production 30 s
//! `XREAD BLOCK`: without the credential wake the `AUTH` would wait for the read
//! to return by itself, 30 s on a quiet layer. The healthy path takes a second
//! or two per test.
//!
//! Worst case — every wait at its full bound: [`CONNECT_BUDGET`] (30 s) per
//! client connected; one [`SETTLE`] per polled condition, subscription start,
//! delivery and `quit()`; fred's 5 s command timeout per one-shot command (an
//! `XADD`, a poll's last `CLIENT LIST`); and `TestValkey`'s drop-time purge
//! (a `SCAN` page and an `UNLINK`, 5 s each). [`busy_reader`] alone is
//! 10 + 5 + 10 + (10 + 5) = 40 s. The largest:
//!
//! | Test | Sum | Worst case |
//! |---|---|---|
//! | the outstanding-wake test | connect 30 + busy reader 40 + joiner subscribe 10 + wake sent 10 + credential pending 10 + tick 10 + `XADD` 5 + delivery 10 + `AUTH` 10 + purge 10 | 145 s |
//! | the schedule-lifecycle test | two connects 60 + two subscribes 20 + two connected polls 20 + scheduled poll 10 + quit 10 + cleared poll 10 + purge 10 | 140 s |
//! | the rotation-gate test | token cached 10 + connect 30 + busy reader 40 + ticks 10 + rotated token cached 10 + `AUTH` 10 + `XADD` 5 + delivery 10 + purge 10 | 135 s |
//! | the restarted-reader test | connect 30 + subscribe 10 + ending 10 + tick 10 + busy reader 40 + `AUTH` 10 + `XADD` 5 + delivery 10 + purge 10 | 135 s |
//! | the busy-reader test | connect 30 + busy reader 40 + `AUTH` 10 + tick 10 + `XADD` 5 + delivery 10 + purge 10 | 115 s |
//!
//! That is past the global 120 s hang-killer, which kills a test silently, so
//! `.config/nextest.toml` gives this module a 180 s kill deadline (60 s × 3),
//! as it does `stream_mux_reader_connect_test`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[cfg(feature = "test-fault-injection")]
use fred::error::ErrorKind as FredErrorKind;
use fred::types::config::CredentialProvider;
use tokio::time::Instant;

use live_edit::observability::metrics::{
    Metrics, REAUTH_RESULT_FAILED, REAUTH_RESULT_OK, WAKE_OUTCOME_ERROR, WAKE_OUTCOME_OK,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::test_support::{connect_test_client_with_credential_provider, TestValkey};
use live_edit::storage::{consume_layer_stream, LayerStreamConsumer, StreamCursor};

use super::mux_test_helpers::{
    assert_no_fred_refresh_on_reader, await_mux, await_reader_blocked_in_xread,
    forced_reconnect_row, install_metrics, read_unblocked_retries, reauths, recv_seqs, wakes,
    xadd_update, ScriptedProvider, SETTLE,
};
use super::snapshot_read_support::{test_valkey_addr, ProxyRule, ScriptReplyProxy};

/// Bound on connecting one client (the fixture's own connect budget).
const CONNECT_BUDGET: Duration = Duration::from_secs(30);

/// The password [`ScriptedProvider`] hands out. Any password authenticates
/// the store's password-less `default` user.
const PASSWORD: &str = "b19i-scripted-reader-password";

/// A forwarding proxy in front of the test store that records the arguments
/// of every `AUTH` frame it relays and counts the `CLIENT UNBLOCK` frames,
/// from every connection through it.
struct Wire {
    proxy: ScriptReplyProxy,
    /// Each relayed `AUTH`'s arguments (`[username, password]`), in order.
    auths: Arc<Mutex<Vec<Vec<String>>>>,
    unblocks: Arc<AtomicUsize>,
}

impl Wire {
    async fn start() -> Self {
        let auths: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let unblocks = Arc::new(AtomicUsize::new(0));
        let rule: ProxyRule = {
            let (auths, unblocks) = (Arc::clone(&auths), Arc::clone(&unblocks));
            Arc::new(move |args: &[Vec<u8>]| {
                let name = args.first()?.to_ascii_uppercase();
                if name == b"AUTH" {
                    auths.lock().expect("never poisoned").push(
                        args[1..]
                            .iter()
                            .map(|a| String::from_utf8_lossy(a).into_owned())
                            .collect(),
                    );
                } else if name == b"CLIENT"
                    && args
                        .get(1)
                        .is_some_and(|sub| sub.eq_ignore_ascii_case(b"UNBLOCK"))
                {
                    unblocks.fetch_add(1, Ordering::SeqCst);
                }
                None
            })
        };
        Self {
            proxy: ScriptReplyProxy::start(test_valkey_addr(), rule).await,
            auths,
            unblocks,
        }
    }

    fn url(&self) -> String {
        self.proxy.url()
    }

    fn auths(&self) -> usize {
        self.auths.lock().expect("never poisoned").len()
    }

    /// The arguments of every `AUTH` relayed after the first `from`.
    fn auth_args_since(&self, from: usize) -> Vec<Vec<String>> {
        self.auths.lock().expect("never poisoned")[from..].to_vec()
    }

    fn client_unblocks(&self) -> usize {
        self.unblocks.load(Ordering::SeqCst)
    }
}

/// A one-reader `TestValkey` behind a [`Wire`], with `provider` on every
/// connection, and its metrics.
async fn fixture(provider: Arc<dyn CredentialProvider>) -> (Wire, TestValkey, Arc<Metrics>) {
    let wire = Wire::start().await;
    let tv = tokio::time::timeout(
        CONNECT_BUDGET,
        TestValkey::setup_with_url_readers_and_credential_provider(wire.url(), 4, 1, provider),
    )
    .await
    .expect("the fixture must connect through the proxy");
    let metrics = install_metrics(&tv).await;
    (wire, tv, metrics)
}

/// Subscribes to `object_id` from `0-0`.
async fn subscribe(tv: &TestValkey, object_id: u64) -> LayerStreamConsumer {
    consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".into()),
        Instant::now() + SETTLE,
    )
    .await
    .expect("the subscription must start")
}

/// Subscribes to `object_id`, has update 1 delivered, and returns once the
/// server shows reader 0's connection blocked in its next `XREAD`: a busy
/// reader on a quiet layer, its read parked for the 30 s BLOCK.
async fn busy_reader(tv: &TestValkey, object_id: u64) -> LayerStreamConsumer {
    let c = subscribe(tv, object_id).await;
    xadd_update(tv, object_id, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("reader 0 holds a connection");
    await_reader_blocked_in_xread(tv.client().pool.next(), &reader).await;
    c
}

/// Polls until `cond` holds, for one [`SETTLE`].
async fn await_until(what: &str, cond: impl Fn() -> bool) {
    let deadline = Instant::now() + SETTLE;
    while !cond() {
        assert!(Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Waits until reader 0 has started `n` more re-AUTH ticks than `from` and
/// holds no credential pending. A tick is counted as it starts, before its
/// `fetch`, and the reader runs one at a time, so every tick before the last
/// of them has run to its end; the last may still be fetching.
async fn await_ticks_past(client: &ValkeyClient, from: u64, n: u64) {
    await_until(
        &format!("reader 0 must take {n} more re-AUTH tick(s)"),
        || {
            let snap = client.stream_mux.reauth_for_test(0);
            snap.ticks >= from + n && !snap.pending
        },
    )
    .await;
}

/// The busy reader (design item 3, first case). A re-AUTH that falls due
/// during a read blocked on a quiet layer sends that read a credential wake,
/// and the `AUTH` goes out on the reader's own connection once the read has
/// returned:
///
/// - `reauths_total{ok}` +1 and exactly one `AUTH` frame on the wire;
/// - one wake (`wakes_total{ok}` +1), so the read returned at once rather than
///   after its 30 s BLOCK (the whole wait is one [`SETTLE`], 10 s);
/// - no `CLIENT UNBLOCK` frame (no backchannel) and no `-UNBLOCKED` re-plan;
/// - fred schedules no refresh on the reader's client: the provider its
///   config carries has no cadence ([`assert_no_fred_refresh_on_reader`]),
///   so every periodic `fetch` made for the reader is one of its own ticks;
/// - nothing forced a reconnect, and the next update is delivered.
///
/// Before b.19i fred's refresh task wrote the `AUTH` to the blocked
/// connection: it interrupted the read through `CLIENT UNBLOCK` on a
/// backchannel (one `-UNBLOCKED` re-plan, and a second connection's handshake
/// `AUTH` on the wire), on a cadence read from the reader client's own
/// provider.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reauth_due_during_a_blocked_read_wakes_it_and_auths_between_reads() {
    const OID: u64 = 3;
    let provider = ScriptedProvider::new(PASSWORD);
    let (wire, tv, metrics) = fixture(provider.clone()).await;
    provider.start_ticking();
    let client = tv.client();

    let c = busy_reader(&tv, OID).await;
    let reauth = client.stream_mux.reauth_for_test(0);
    assert!(
        reauth.scheduled && !reauth.pending,
        "precondition: the connected reader has a tick scheduled and nothing pending; got \
         {reauth:?}"
    );
    let auths_before = wire.auths();
    let wakes_before = wakes(&metrics, WAKE_OUTCOME_OK);
    let xreads_before = client.stream_mux.snapshot().xreads();

    provider.grant(1);
    await_until(
        "the granted credential's AUTH must be answered well inside the 30 s BLOCK: the read \
         must have been woken for it",
        || reauths(&metrics, REAUTH_RESULT_OK) == 1,
    )
    .await;
    await_ticks_past(client, client.stream_mux.reauth_for_test(0).ticks, 1).await;

    assert_eq!(
        reauths(&metrics, REAUTH_RESULT_OK),
        1,
        "one credential, one AUTH"
    );
    assert_eq!(reauths(&metrics, REAUTH_RESULT_FAILED), 0);
    assert_eq!(
        wire.auth_args_since(auths_before),
        vec![vec!["default".to_owned(), PASSWORD.to_owned()]],
        "exactly the reader's one AUTH reached the wire — no backchannel connection's \
         handshake, no fred refresh AUTH — and it carried the credential the tick fetched"
    );
    assert_eq!(
        wakes(&metrics, WAKE_OUTCOME_OK) - wakes_before,
        1,
        "the due credential woke the in-flight read: one credential wake"
    );
    assert_eq!(wakes(&metrics, WAKE_OUTCOME_ERROR), 0);
    assert!(
        client.stream_mux.snapshot().xreads() > xreads_before,
        "the woken read returned and the reader issued its next one"
    );
    assert_eq!(
        wire.client_unblocks(),
        0,
        "no CLIENT UNBLOCK: nothing but the reader's own XREAD was ever queued on its \
         connection, so fred never opened a backchannel"
    );
    assert_eq!(
        read_unblocked_retries(&metrics),
        0,
        "no read was interrupted with -UNBLOCKED"
    );
    assert_no_fred_refresh_on_reader(
        &client
            .stream_mux
            .reader_client_for_test(0)
            .expect("the reader still holds its connection"),
        &provider,
    );
    assert!(
        forced_reconnect_row(&metrics).iter().all(|(_, n)| *n == 0),
        "a re-AUTH forces no reconnect: {:?}",
        forced_reconnect_row(&metrics)
    );

    xadd_update(&tv, OID, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2], "nothing lost across the wake");
    assert!(c.take_terminal_exit().is_none());
}

/// The real gate (design item 3, gated tick, and b.19i Part A end to end): a
/// `TokenFileCredentialProvider` on a sub-second band, installed on every
/// connection.
///
/// 1. While the token is the one every connection authenticated with at its
///    handshake — the provider's first sighting — the reader's ticks fetch no
///    password: no `AUTH` on the wire from anyone, no wake, no re-AUTH counted.
///    The window is `GATE_WINDOW`, a minute — far longer than setup plus the
///    ticks this phase watches — so if the first sighting opened one (the
///    pre-b.19i rule) it would still be open here and the ticks inside it
///    would send. Nothing in the test waits the window out.
/// 2. The sidecar rotates the token. The next observation opens the window,
///    and the reader's ticks inside it send `AUTH` carrying the rotated token,
///    as every `AUTH` on the wire then does. The reader's read is in flight,
///    so at least one of them comes through a credential wake; no
///    `CLIENT UNBLOCK` crosses the wire at any point.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reader_follows_the_rotation_gate_no_auth_before_a_rotation_and_auth_inside_its_window() {
    use live_edit::auth::service_token::ServiceTokenProvider;
    use live_edit::storage::token_file_credentials::TokenFileCredentialProvider;

    /// The provider's tick band: every client ticks every 200–300 ms.
    const BASE: Duration = Duration::from_millis(200);
    const JITTER: Duration = Duration::from_millis(100);
    /// The fan-out window. Long enough that a window a first sighting opened
    /// at the fixture's handshakes would still be open through phase 1 however
    /// slow the setup — otherwise a pre-b.19i window could close before the
    /// phase's baseline and the phase would pass vacuously. Nothing waits it
    /// out.
    const GATE_WINDOW: Duration = Duration::from_secs(60);
    const TOKEN_A: &str = "tkn-b19i-gate-alpha";
    const TOKEN_B: &str = "tkn-b19i-gate-bravo";
    const OID: u64 = 5;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("token.json");
    let write_token = |token: &'static str| {
        let path = path.clone();
        async move {
            let expires_on = chrono::Utc::now() + chrono::Duration::seconds(3600);
            let body = format!(
                r#"{{"access_token":"{token}","expires_on":"{}"}}"#,
                expires_on.to_rfc3339()
            );
            tokio::fs::write(&path, body)
                .await
                .expect("write the token file");
        }
    };
    write_token(TOKEN_A).await;
    let cancel = tokio_util::sync::CancellationToken::new();
    let tokens = Arc::new(ServiceTokenProvider::new_with_refresh_lead(
        path.clone(),
        Duration::from_millis(20),
        chrono::Duration::seconds(180),
        Arc::new(crate::common::metrics_scrape_only("test")),
        cancel.clone(),
    ));
    let cached = |want: &'static str| {
        let tokens = Arc::clone(&tokens);
        move || tokens.cached_bearer_if_unexpired().as_deref() == Some(want)
    };
    await_until("the token file must be cached", cached(TOKEN_A)).await;
    let provider = Arc::new(TokenFileCredentialProvider::new(
        Arc::clone(&tokens),
        "default".to_owned(),
        BASE,
        JITTER,
        GATE_WINDOW,
    ));
    let (wire, tv, metrics) = fixture(provider).await;
    let client = tv.client();

    // Phase 1: the first token, unchanged.
    let c = busy_reader(&tv, OID).await;
    let auths_before = wire.auths();
    let wakes_before = wakes(&metrics, WAKE_OUTCOME_OK);
    let ticks_before = client.stream_mux.reauth_for_test(0).ticks;
    await_ticks_past(client, ticks_before, 3).await;
    assert_eq!(
        (
            reauths(&metrics, REAUTH_RESULT_OK),
            reauths(&metrics, REAUTH_RESULT_FAILED)
        ),
        (0, 0),
        "a gated tick sends nothing and counts nothing"
    );
    assert_eq!(
        wire.auths(),
        auths_before,
        "no AUTH on the wire while the token is the one every connection authenticated with \
         — not from the reader's ticks, not from the pools' fred refresh. A first sighting \
         that opened the fan-out window would re-AUTH them all here"
    );
    assert_eq!(
        wakes(&metrics, WAKE_OUTCOME_OK),
        wakes_before,
        "a gated tick wakes nothing"
    );

    // Phase 2: the sidecar rotates the token.
    write_token(TOKEN_B).await;
    await_until("the rotated token must be cached", cached(TOKEN_B)).await;
    await_until(
        "a reader tick inside the rotation's fan-out window must send AUTH",
        || reauths(&metrics, REAUTH_RESULT_OK) >= 1,
    )
    .await;
    assert!(
        wakes(&metrics, WAKE_OUTCOME_OK) > wakes_before,
        "the reader's read was in flight, so its re-AUTH came through a credential wake"
    );
    assert_eq!(reauths(&metrics, REAUTH_RESULT_FAILED), 0);
    // Every AUTH since phase 1 — the reader's and the pools' fred refresh
    // alike — carries the rotated token, and the reader's are among them: a
    // relayed frame is recorded before its answer is counted, so there are at
    // least as many as the reader's answered re-AUTHs read first.
    let answered = reauths(&metrics, REAUTH_RESULT_OK) as usize;
    let sent = wire.auth_args_since(auths_before);
    assert!(
        sent.len() >= answered,
        "{answered} answered reader re-AUTH(s) but only {} AUTH frame(s) on the wire",
        sent.len()
    );
    assert!(
        sent.iter()
            .all(|args| *args == ["default".to_owned(), TOKEN_B.to_owned()]),
        "every AUTH after the rotation must carry the rotated token as `default`'s \
         password; got {sent:?}"
    );
    assert_eq!(
        wire.client_unblocks(),
        0,
        "no CLIENT UNBLOCK at any point: no backchannel interrupt"
    );
    assert_eq!(read_unblocked_retries(&metrics), 0);

    xadd_update(&tv, OID, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2]);
    assert!(c.take_terminal_exit().is_none());
    cancel.cancel();
}

/// A tick whose `fetch` fails counts `failed` and sends nothing: no `AUTH`,
/// no wake. The connection is not touched — the pools' refresh warnings and
/// readiness report a credential outage, not the reader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_credential_fetch_counts_failed_and_sends_nothing() {
    const OID: u64 = 7;
    let provider = ScriptedProvider::new(PASSWORD);
    let (wire, tv, metrics) = fixture(provider.clone()).await;
    provider.start_ticking();
    let client = tv.client();

    let c = busy_reader(&tv, OID).await;
    let auths_before = wire.auths();
    let wakes_before = wakes(&metrics, WAKE_OUTCOME_OK);

    provider.fail(1);
    await_until("the failed fetch must be counted", || {
        reauths(&metrics, REAUTH_RESULT_FAILED) == 1
    })
    .await;
    await_ticks_past(client, client.stream_mux.reauth_for_test(0).ticks, 1).await;

    assert_eq!(reauths(&metrics, REAUTH_RESULT_FAILED), 1);
    assert_eq!(reauths(&metrics, REAUTH_RESULT_OK), 0);
    assert_eq!(wire.auths(), auths_before, "no credential, no AUTH");
    assert_eq!(
        wakes(&metrics, WAKE_OUTCOME_OK),
        wakes_before,
        "no credential, no wake"
    );
    assert!(forced_reconnect_row(&metrics).iter().all(|(_, n)| *n == 0));
    xadd_update(&tv, OID, 2).await;
    assert_eq!(recv_seqs(&c, 1).await, [2]);
}

/// M1, the schedule's lifecycle: a reader with no connection has no re-AUTH
/// tick scheduled, although its provider ticks from before the client
/// connects; connecting schedules one; `quit()` clears it. Under `static`
/// auth (no provider) a connected reader schedules none.
///
/// The reader publishes its connection, then arms its hooks, then schedules,
/// so the schedule is awaited once the connection shows, not read at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_readers_reauth_schedule_follows_its_connection() {
    // Under a provider, ticking before anything connects: a reader that
    // scheduled whenever its provider ticks, connection or not, would show
    // a schedule from the start.
    let provider = ScriptedProvider::new(PASSWORD);
    provider.start_ticking();
    let mut config = crate::common::test_config();
    config.subscriber_pool_size = 1;
    let client = tokio::time::timeout(
        CONNECT_BUDGET,
        connect_test_client_with_credential_provider(&config, provider.clone()),
    )
    .await
    .expect("connect must settle")
    .expect("connect must succeed");
    assert!(
        !client.stream_mux.reauth_for_test(0).scheduled,
        "a reader with no connection schedules no re-AUTH"
    );
    let schema =
        live_edit::storage::keys::KeySchema::new(&format!("test_{}:", uuid::Uuid::new_v4()));
    let sub = consume_layer_stream(
        &client,
        &schema,
        0,
        StreamCursor::From("0-0".into()),
        Instant::now() + SETTLE,
    )
    .await
    .expect("the subscription must start");
    await_mux(&client, "reader 0 connected", |s| s.readers[0].connected).await;
    await_until(
        "connecting schedules the connection's first re-AUTH tick",
        || client.stream_mux.reauth_for_test(0).scheduled,
    )
    .await;
    drop(sub);
    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");
    await_until(
        "the connection's re-AUTH schedule, and anything pending, must end with it",
        || {
            let snap = client.stream_mux.reauth_for_test(0);
            !snap.scheduled && !snap.pending
        },
    )
    .await;

    // Under `static` auth: no provider, no schedule.
    let tv = tokio::time::timeout(CONNECT_BUDGET, TestValkey::setup_with_readers(1))
        .await
        .expect("the static fixture must connect");
    let c = subscribe(&tv, 1).await;
    await_mux(tv.client(), "reader 0 connected", |s| {
        s.readers[0].connected
    })
    .await;
    assert_eq!(
        tv.client().stream_mux.reauth_for_test(0),
        live_edit::storage::stream_mux::ReauthSnapshot {
            scheduled: false,
            pending: false,
            ticks: 0
        },
        "under static auth a connected reader schedules no re-AUTH"
    );
    drop(c);
}

/// The cases that need the multiplexer's fault sites or knobs.
#[cfg(feature = "test-fault-injection")]
mod faults {
    use super::*;

    use fred::interfaces::ClientInterface;
    use live_edit::observability::metrics::STREAM_EXIT_ERROR;
    use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMuxKnobs};

    use crate::grpc::log_capture::{assert_field, purge_keys, ThreadCapture};
    use crate::storage::mux_test_helpers::{
        assert_ends_with, await_faults_consumed, await_lines, fault, push_faults, read_series,
        QUIET,
    };

    /// The WARN a reader logs when its `AUTH` fails.
    const REAUTH_FAILED_MSG: &str = "stream reader credential re-AUTH failed; the connection is \
                                     kept and nothing is reconnected (the next rotation or \
                                     reconnect re-authenticates it)";

    /// The WARN a failed credential wake logs.
    const CREDENTIAL_WAKE_FAILED_MSG: &str = "stream multiplexer wake failed; the reader's \
                                              credential re-AUTH waits until the in-flight XREAD \
                                              returns (up to XREAD_BLOCK_MS), and is sent then";

    /// The DEBUG line of an answered re-AUTH.
    const REAUTH_ANSWERED_MSG: &str = "stream reader credential re-AUTH answered";

    /// The DEBUG line of a tick whose fetch failed.
    const FETCH_FAILED_MSG: &str = "stream reader could not fetch a credential for its re-AUTH; \
                                    nothing is sent, and the next tick tries again";

    /// The shortened `probe_timeout` the hang cases run on: well below the
    /// production 2 s, and named in the failure the reader reports, so the
    /// tests can tell which bound cut the hang.
    const SHORT_PROBE_TIMEOUT: Duration = Duration::from_millis(200);

    /// Sets reader knobs on `tv`'s multiplexer.
    fn set_knobs(tv: &TestValkey, f: impl FnOnce(StreamMuxKnobs) -> StreamMuxKnobs) {
        let mux = &tv.client().stream_mux;
        mux.set_knobs_for_test(f(mux.knobs()));
    }

    /// An idle reader (connected, nothing to read): a granted tick sends its
    /// `AUTH` at once, with no wake (there is no read to wake) and no read.
    /// The idle keepalive ticking is the proof the reader is idle: it ticks
    /// only with nothing to read and no read in flight.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_idle_readers_reauth_is_sent_at_once_with_no_wake() {
        use live_edit::observability::metrics::KEEPALIVE_RESULT_OK;
        const OID: u64 = 11;
        let keepalives_ok = |metrics: &Metrics| {
            read_series(
                metrics,
                "live_edit_stream_mux_keepalives_total",
                &[("result", KEEPALIVE_RESULT_OK)],
            )
        };
        let provider = ScriptedProvider::new(PASSWORD);
        let (wire, tv, metrics) = fixture(provider.clone()).await;
        provider.start_ticking();
        // A short BLOCK, so the reader goes idle soon after its subscription
        // leaves.
        set_knobs(&tv, |k| StreamMuxKnobs { block_ms: 100, ..k });
        let client = tv.client();
        let c = subscribe(&tv, OID).await;
        xadd_update(&tv, OID, 1).await;
        assert_eq!(recv_seqs(&c, 1).await, [1]);
        drop(c);
        await_mux(client, "the subscription left", |s| s.subscriptions() == 0).await;
        // Idle: the keepalive, short now, ticks, which it does only with
        // nothing to read and no read in flight.
        let ticks = keepalives_ok(&metrics);
        set_knobs(&tv, |k| StreamMuxKnobs {
            keepalive_interval: Some(Duration::from_millis(50)),
            ..k
        });
        await_until("the idle reader's keepalive must tick", || {
            keepalives_ok(&metrics) > ticks
        })
        .await;
        let last = client.stream_mux.snapshot().xreads();
        assert!(client.stream_mux.reauth_for_test(0).scheduled);
        let auths_before = wire.auths();
        let wakes_before = wakes(&metrics, WAKE_OUTCOME_OK) + wakes(&metrics, WAKE_OUTCOME_ERROR);

        provider.grant(1);
        await_until("the idle reader's AUTH must be answered", || {
            reauths(&metrics, REAUTH_RESULT_OK) == 1
        })
        .await;
        await_ticks_past(client, client.stream_mux.reauth_for_test(0).ticks, 1).await;

        assert_eq!(wire.auths() - auths_before, 1, "one AUTH on the wire");
        assert_eq!(
            wakes(&metrics, WAKE_OUTCOME_OK) + wakes(&metrics, WAKE_OUTCOME_ERROR),
            wakes_before,
            "an idle reader has no read to wake"
        );
        assert_eq!(
            client.stream_mux.snapshot().xreads(),
            last,
            "the re-AUTH issued no read: the reader stays idle"
        );
        assert_eq!(wire.client_unblocks(), 0);
    }

    /// A credential wake that fails (all three attempts) leaves the `AUTH`
    /// pending until the read returns by itself, then sends it: one WARN with
    /// `reason="credential"` naming that cost, `wakes_total{error}` +1, and the
    /// `AUTH` answered once the (shortened) BLOCK has run out. No wake landed,
    /// so the wake-ack watchdog has nothing to judge.
    #[tokio::test]
    async fn a_failed_credential_wake_leaves_the_auth_for_when_the_read_returns() {
        const OID: u64 = 13;
        let capture = ThreadCapture::start();
        let provider = ScriptedProvider::new(PASSWORD);
        let (wire, tv, metrics) = fixture(provider.clone()).await;
        provider.start_ticking();
        // The read returns by itself after 1.5 s: the AUTH's only way out once
        // its wake has failed.
        set_knobs(&tv, |k| StreamMuxKnobs {
            block_ms: 1_500,
            ..k
        });
        let c = busy_reader(&tv, OID).await;
        let auths_before = wire.auths();
        tv.client().stream_mux.push_faults_for_test(
            MuxFaultSite::Wake,
            (0..3).map(|_| fault(FredErrorKind::IO, "synthetic wake failure")),
        );

        provider.grant(1);
        let lines = await_lines(&capture, CREDENTIAL_WAKE_FAILED_MSG, 1).await;
        // Bounded quiet window: no second WARN follows for the one failure.
        tokio::time::sleep(QUIET).await;
        assert_eq!(
            capture
                .snapshot()
                .with_message(CREDENTIAL_WAKE_FAILED_MSG)
                .len(),
            1,
            "one WARN for the one failed wake"
        );
        assert_eq!(lines[0]["level"], "WARN");
        assert_field(&lines[0], "reason", "credential".into());
        await_until(
            "the AUTH must go out once the read returns by itself",
            || reauths(&metrics, REAUTH_RESULT_OK) == 1,
        )
        .await;

        assert_eq!(wakes(&metrics, WAKE_OUTCOME_ERROR), 1);
        assert_eq!(wakes(&metrics, WAKE_OUTCOME_OK), 0);
        assert_eq!(
            tv.client()
                .stream_mux
                .pending_faults_for_test(MuxFaultSite::Wake),
            0,
            "the wake made all three attempts"
        );
        assert_eq!(wire.auths() - auths_before, 1);
        assert_eq!(wire.client_unblocks(), 0);
        assert!(forced_reconnect_row(&metrics).iter().all(|(_, n)| *n == 0));
        xadd_update(&tv, OID, 2).await;
        assert_eq!(recv_seqs(&c, 1).await, [2]);
        drop(c);
        purge_keys(&tv).await;
    }

    /// Shared body of the two failed-`AUTH` cases: one `Reauth` fault, a
    /// granted credential on a busy reader. The `AUTH` counts `failed`, logs
    /// one WARN naming the reader, the node (none, off a cluster) and
    /// `failure` — exactly — and forces no reconnect. It charges nothing to
    /// the transient budget either: the budget here allows no transient
    /// failure, so a charge would end the subscription, which instead carries
    /// on, on the same connection.
    async fn a_failed_reauth_keeps_the_connection(
        injected: MuxFault,
        probe_timeout: Option<Duration>,
        failure: &str,
    ) {
        const OID: u64 = 17;
        let capture = ThreadCapture::start();
        let provider = ScriptedProvider::new(PASSWORD);
        let (_wire, tv, metrics) = fixture(provider.clone()).await;
        provider.start_ticking();
        set_knobs(&tv, |k| StreamMuxKnobs {
            max_transient_failures: 0,
            probe_timeout: probe_timeout.unwrap_or(k.probe_timeout),
            ..k
        });
        let client = tv.client();
        let c = busy_reader(&tv, OID).await;
        let connection = client
            .stream_mux
            .reader_client_for_test(0)
            .expect("connected")
            .connection_ids();
        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Reauth, [injected]);

        provider.grant(1);
        await_until("the failed AUTH must be counted", || {
            reauths(&metrics, REAUTH_RESULT_FAILED) == 1
        })
        .await;
        let lines = await_lines(&capture, REAUTH_FAILED_MSG, 1).await;
        // Bounded quiet window: no second WARN for the one failure.
        tokio::time::sleep(QUIET).await;
        let lines_after = capture.snapshot();
        assert_eq!(
            lines_after.with_message(REAUTH_FAILED_MSG).len(),
            1,
            "one WARN per failed AUTH"
        );
        assert_eq!(lines[0]["level"], "WARN");
        assert_field(&lines[0], "reader", 0.into());
        assert_field(&lines[0], "node", "default".into());
        assert_field(&lines[0], "failure", failure.into());

        assert_eq!(reauths(&metrics, REAUTH_RESULT_OK), 0);
        assert_eq!(
            client
                .stream_mux
                .pending_faults_for_test(MuxFaultSite::Reauth),
            0
        );
        assert!(
            forced_reconnect_row(&metrics).iter().all(|(_, n)| *n == 0),
            "a failed AUTH forces no reconnect: {:?}",
            forced_reconnect_row(&metrics)
        );
        assert_eq!(
            client
                .stream_mux
                .reader_client_for_test(0)
                .expect("still connected")
                .connection_ids(),
            connection,
            "the reader keeps its connection"
        );
        xadd_update(&tv, OID, 2).await;
        assert_eq!(recv_seqs(&c, 1).await, [2], "the subscription carries on");
        assert!(c.take_terminal_exit().is_none());
        drop(c);
        purge_keys(&tv).await;
    }

    /// `Reauth` fault `Err`: the server refused, say. The WARN names fred's
    /// error as it displays.
    #[tokio::test]
    async fn a_failed_reauth_counts_failed_warns_once_and_keeps_the_connection() {
        let refused =
            fred::error::Error::new(FredErrorKind::Auth, "synthetic WRONGPASS on re-AUTH");
        a_failed_reauth_keeps_the_connection(
            MuxFault::Err(refused.clone()),
            None,
            &refused.to_string(),
        )
        .await;
    }

    /// `Reauth` fault `Hang`: the `AUTH` never answers and is cut at the
    /// knobs' `probe_timeout` — shortened here to [`SHORT_PROBE_TIMEOUT`],
    /// which the WARN names, so the production 2 s bound cutting it instead
    /// would fail — then handled as a failure.
    #[tokio::test]
    async fn a_hung_reauth_is_cut_at_the_probe_timeout_and_counts_failed() {
        a_failed_reauth_keeps_the_connection(
            MuxFault::Hang,
            Some(SHORT_PROBE_TIMEOUT),
            "no answer within 200 ms",
        )
        .await;
    }

    /// M1 on a restart: a reader panics and is restarted while still holding
    /// its connection (the connection outlives the reader task). The
    /// restarted reader schedules its re-AUTH again from that connection: it
    /// keeps ticking, and a granted credential is still sent. Without that the
    /// restarted reader's schedule starts empty, and the connection never
    /// re-AUTHs again until it reconnects.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_restarted_reader_still_holding_its_connection_keeps_reauthing() {
        const OID: u64 = 19;
        let provider = ScriptedProvider::new(PASSWORD);
        let (_wire, tv, metrics) = fixture(provider.clone()).await;
        provider.start_ticking();
        let client = tv.client();
        // The reader connects for this subscription, then its first read
        // panics.
        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Panic]);
        let first = subscribe(&tv, OID).await;
        assert_ends_with(&first, STREAM_EXIT_ERROR).await;
        assert!(
            client.stream_mux.reader_client_for_test(0).is_some(),
            "precondition: the restarted reader still holds its connection"
        );
        await_ticks_past(client, client.stream_mux.reauth_for_test(0).ticks, 1).await;

        let c = busy_reader(&tv, OID).await;
        provider.grant(1);
        await_until("the restarted reader's AUTH must be answered", || {
            reauths(&metrics, REAUTH_RESULT_OK) == 1
        })
        .await;
        xadd_update(&tv, OID, 2).await;
        assert_eq!(recv_seqs(&c, 1).await, [2]);
    }

    /// A provider whose periodic `fetch` never returns does not freeze the
    /// reader. The tick awaits the provider inline, with the read's result
    /// waiting behind it, so the fetch is bounded by the knobs'
    /// `probe_timeout` — [`SHORT_PROBE_TIMEOUT`] here, which the tick's DEBUG
    /// line names as the failure, so the production 2 s bound cutting it
    /// instead would fail: it counts `failed`, sends nothing, and the reader
    /// goes on to deliver the next update. Unbounded, the reader would sit in
    /// the tick forever and deliver nothing more.
    ///
    /// Current-thread, so the [`ThreadCapture`] sees the reader task's DEBUG
    /// line; keys are purged explicitly.
    #[tokio::test]
    async fn a_hung_credential_fetch_is_cut_at_the_probe_timeout_and_reads_carry_on() {
        const OID: u64 = 29;
        let capture = ThreadCapture::start_debug();
        let provider = ScriptedProvider::new(PASSWORD);
        let (wire, tv, metrics) = fixture(provider.clone()).await;
        provider.start_ticking();
        set_knobs(&tv, |k| StreamMuxKnobs {
            probe_timeout: SHORT_PROBE_TIMEOUT,
            ..k
        });
        let c = busy_reader(&tv, OID).await;
        let auths_before = wire.auths();

        provider.hang(1);
        await_until(
            "the hung fetch must be cut at the probe timeout and counted",
            || reauths(&metrics, REAUTH_RESULT_FAILED) == 1,
        )
        .await;
        let lines = await_lines(&capture, FETCH_FAILED_MSG, 1).await;
        assert_eq!(lines[0]["level"], "DEBUG");
        assert_field(&lines[0], "reader", 0.into());
        assert_field(&lines[0], "failure", "no answer within 200 ms".into());
        xadd_update(&tv, OID, 2).await;
        assert_eq!(
            recv_seqs(&c, 1).await,
            [2],
            "the reader carries on reading after the cut fetch"
        );
        assert_eq!(reauths(&metrics, REAUTH_RESULT_OK), 0);
        assert_eq!(wire.auths(), auths_before, "no credential, no AUTH");
        assert!(forced_reconnect_row(&metrics).iter().all(|(_, n)| *n == 0));
        assert!(c.take_terminal_exit().is_none());
        drop(c);
        drop(capture);
        purge_keys(&tv).await;
    }

    /// Only one wake per in-flight read. With a coverage wake already
    /// outstanding for the read (a joining subscription's wake, held unlanded
    /// by a `Hang` fault), a credential that falls due sends no second wake:
    /// the credential waits, pending, and the outstanding wake — or here the
    /// read returning with data — serves it. The `AUTH` then goes out between
    /// reads.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_credential_due_while_a_wake_is_outstanding_sends_no_second_wake() {
        const OID: u64 = 31;
        const JOINER: u64 = 37;
        let provider = ScriptedProvider::new(PASSWORD);
        let (wire, tv, metrics) = fixture(provider.clone()).await;
        provider.start_ticking();
        let client = tv.client();
        let c = busy_reader(&tv, OID).await;
        let wakes_sent = || client.stream_mux.snapshot().readers[0].wakes;
        let before = wakes_sent();

        // A second subscription joins the one reader while its read is in
        // flight: a coverage wake, which hangs and never reports back.
        client
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Wake, [MuxFault::Hang]);
        let joiner = subscribe(&tv, JOINER).await;
        await_until("the joiner's coverage wake must be sent and hang", || {
            wakes_sent() == before + 1
                && client
                    .stream_mux
                    .pending_faults_for_test(MuxFaultSite::Wake)
                    == 0
        })
        .await;

        provider.grant(1);
        await_until("a tick must fetch the granted credential", || {
            provider.grants_left() == 0
        })
        .await;
        // The next tick starts only once the one that took the credential —
        // and any wake it sent — has finished.
        let ticks = client.stream_mux.reauth_for_test(0).ticks;
        await_until("the reader must take one more tick", || {
            client.stream_mux.reauth_for_test(0).ticks > ticks
        })
        .await;
        assert_eq!(
            wakes_sent(),
            before + 1,
            "a wake is already outstanding for this read: the credential sends no second one"
        );
        assert!(
            client.stream_mux.reauth_for_test(0).pending,
            "the credential is held while the read is in flight"
        );
        assert_eq!(
            reauths(&metrics, REAUTH_RESULT_OK),
            0,
            "the AUTH waits for the read to return"
        );

        // Data returns the read; the AUTH goes out between reads.
        let auths_before = wire.auths();
        xadd_update(&tv, OID, 2).await;
        assert_eq!(recv_seqs(&c, 1).await, [2]);
        await_until("the pending AUTH must go out once the read returns", || {
            reauths(&metrics, REAUTH_RESULT_OK) == 1
        })
        .await;
        assert_eq!(wire.auths() - auths_before, 1);
        assert_eq!(wakes_sent(), before + 1, "still the one wake");
        drop((c, joiner));
    }

    /// The backoff wait takes a due re-AUTH tick too. A reader backing off
    /// after a whole-reader failure (a minute here) sends a granted
    /// credential's `AUTH` at once rather than after the backoff, and issues
    /// no read for it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reauth_due_while_the_reader_backs_off_is_sent_without_waiting_out_the_backoff() {
        const OID: u64 = 41;
        let provider = ScriptedProvider::new(PASSWORD);
        let (wire, tv, metrics) = fixture(provider.clone()).await;
        provider.start_ticking();
        let client = tv.client();
        set_knobs(&tv, |k| StreamMuxKnobs {
            backoff_initial: Duration::from_secs(60),
            backoff_cap: Duration::from_secs(60),
            ..k
        });
        push_faults(client, 1, FredErrorKind::IO, "synthetic connection reset");
        let c = subscribe(&tv, OID).await;
        await_faults_consumed(client).await;
        assert!(
            client.stream_mux.reauth_for_test(0).scheduled,
            "precondition: the reader holds its connection while it backs off"
        );
        let xreads = client.stream_mux.snapshot().xreads();
        let auths_before = wire.auths();

        provider.grant(1);
        await_until(
            "the AUTH must go out during the minute-long backoff, not after it",
            || reauths(&metrics, REAUTH_RESULT_OK) == 1,
        )
        .await;
        assert_eq!(
            wire.auth_args_since(auths_before),
            vec![vec!["default".to_owned(), PASSWORD.to_owned()]]
        );
        assert_eq!(
            client.stream_mux.snapshot().xreads(),
            xreads,
            "still backing off: the re-AUTH issued no read"
        );
        drop(c);
    }

    /// The re-AUTH's credential is never logged: a distinctive password, sent
    /// on the answered path, the failed-`AUTH` path and around a failed fetch,
    /// appears nowhere in the captured output at DEBUG. Non-vacuous: the
    /// capture holds each path's own line.
    #[tokio::test]
    async fn no_reauth_credential_reaches_the_logs() {
        const OID: u64 = 23;
        const CANARY: &str = "b19i-reauth-canary-5f3a9e27c1d84b06";
        let capture = ThreadCapture::start_debug();
        let provider = ScriptedProvider::new(CANARY);
        let (_wire, tv, metrics) = fixture(provider.clone()).await;
        provider.start_ticking();
        let client = tv.client();
        let c = busy_reader(&tv, OID).await;

        // Answered.
        provider.grant(1);
        await_until("the AUTH must be answered", || {
            reauths(&metrics, REAUTH_RESULT_OK) == 1
        })
        .await;
        // Failed AUTH.
        client.stream_mux.push_faults_for_test(
            MuxFaultSite::Reauth,
            [fault(FredErrorKind::Auth, "synthetic refusal")],
        );
        provider.grant(1);
        await_until("the failed AUTH must be counted", || {
            reauths(&metrics, REAUTH_RESULT_FAILED) == 1
        })
        .await;
        // Failed fetch.
        provider.fail(1);
        await_until("the failed fetch must be counted", || {
            reauths(&metrics, REAUTH_RESULT_FAILED) == 2
        })
        .await;

        let captured = capture.snapshot();
        for msg in [REAUTH_ANSWERED_MSG, REAUTH_FAILED_MSG, FETCH_FAILED_MSG] {
            assert!(
                !captured.with_message(msg).is_empty(),
                "non-vacuity: the capture must hold `{msg}`; full capture:\n{}",
                captured.text
            );
        }
        captured.assert_absent_everywhere(CANARY);
        drop(c);
        purge_keys(&tv).await;
    }
}
