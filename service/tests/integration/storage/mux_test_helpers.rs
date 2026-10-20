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

//! Shared harness for stream-multiplexer tests: the storage-level suites and
//! the gRPC suites that drive a real `SubscribeReadLayer` against injected
//! reader faults.

use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{ClientInterface, StreamsInterface};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::notifications::{consume_layer_stream, LayerNotification, StreamCursor};
use live_edit::storage::stream_mux::MuxSnapshot;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::LayerStreamConsumer;

/// Upper bound on any wait in these suites; unreachable on the healthy path.
pub(crate) const SETTLE: Duration = Duration::from_secs(10);

/// XADDs a well-formed `update` entry with the deterministic `{seq}-0` id.
pub(crate) async fn xadd_update(tv: &TestValkey, object_id: u64, seq: u64) {
    let stream_key = tv.key_schema().layer_stream(object_id);
    let id = format!("{seq}-0");
    let payload = format!("update|{object_id}|{seq}|probe|0|0");
    let _: String = tv
        .client()
        .pool
        .xadd(
            stream_key,
            false,
            None,
            id.as_str(),
            vec![("payload", payload)],
        )
        .await
        .expect("XADD must succeed");
}

pub(crate) async fn install_metrics(tv: &TestValkey) -> Arc<Metrics> {
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    metrics
}

fn read_counter(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> u64 {
    read_series(metrics, name, labels) as u64
}

/// One pre-initialised series. Panics on an absent one rather than reading
/// zero, so a misspelled name or label cannot satisfy an `== 0` assertion.
pub(crate) fn read_series(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> i64 {
    sample_value(&render(metrics), name, labels)
        .unwrap_or_else(|| panic!("{name}{labels:?} is pre-initialised in Metrics::new")) as i64
}

/// Polls one pre-initialised series until it reads `want`.
pub(crate) async fn await_series(
    metrics: &Metrics,
    name: &str,
    labels: &[(&str, &str)],
    want: i64,
) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    loop {
        let got = read_series(metrics, name, labels);
        if got == want {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{name}{labels:?} stayed at {got}, want {want}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// `live_edit_valkey_consumer_forced_reconnects_total` on every phase, in
/// `CONSUMER_FORCED_RECONNECT_PHASES` order.
pub(crate) fn forced_reconnect_row(metrics: &Metrics) -> Vec<(&'static str, u64)> {
    live_edit::observability::metrics::CONSUMER_FORCED_RECONNECT_PHASES
        .iter()
        .map(|phase| (*phase, read_forced_reconnects(metrics, phase)))
        .collect()
}

/// The [`forced_reconnect_row`] with `n` on `phase` and zero elsewhere.
pub(crate) fn forced_reconnects_only_on(phase: &str, n: u64) -> Vec<(&'static str, u64)> {
    live_edit::observability::metrics::CONSUMER_FORCED_RECONNECT_PHASES
        .iter()
        .map(|p| (*p, if *p == phase { n } else { 0 }))
        .collect()
}

#[cfg(feature = "test-fault-injection")]
pub(crate) fn read_wakes(metrics: &Metrics, outcome: &str) -> u64 {
    read_counter(
        metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", outcome)],
    )
}

pub(crate) fn read_forced_reconnects(metrics: &Metrics, phase: &str) -> u64 {
    read_counter(
        metrics,
        "live_edit_valkey_consumer_forced_reconnects_total",
        &[("phase", phase)],
    )
}

pub(crate) fn read_permanent_exits(metrics: &Metrics, phase: &str) -> u64 {
    read_counter(
        metrics,
        "live_edit_valkey_consumer_permanent_exits_total",
        &[("phase", phase)],
    )
}

/// `live_edit_valkey_consumer_unblocked_retries_total`: live reader reads that
/// ended in `-UNBLOCKED` and were re-planned. In production only an outside
/// `CLIENT UNBLOCK` does that; in tests, a separate client's `CLIENT UNBLOCK`
/// or a command sent at a blocked reader. The residual `Blocking::Interrupt`
/// case never ticks it: that `-UNBLOCKED` goes to the abandoned read's
/// dropped caller. Since b.19i no credential refresh produces one either — a
/// reader re-AUTHs between its reads.
pub(crate) fn read_unblocked_retries(metrics: &Metrics) -> u64 {
    read_counter(
        metrics,
        "live_edit_valkey_consumer_unblocked_retries_total",
        &[],
    )
}

/// `live_edit_stream_mux_reauths_total{result}` (b.19i).
pub(crate) fn reauths(metrics: &Metrics, result: &str) -> i64 {
    read_series(
        metrics,
        "live_edit_stream_mux_reauths_total",
        &[("result", result)],
    )
}

/// `live_edit_stream_mux_wakes_total{outcome}`.
pub(crate) fn wakes(metrics: &Metrics, outcome: &str) -> i64 {
    read_series(
        metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", outcome)],
    )
}

/// The stream readers' re-AUTH cadence under [`ScriptedProvider`]: short, so
/// a tick falls due soon after a credential is granted, and long enough that
/// a test reads the counters between two ticks.
pub(crate) const REAUTH_TICK: Duration = Duration::from_millis(250);

/// A credential provider whose periodic answers a test scripts (b.19i), for
/// driving the stream readers' own re-AUTH against the password-less test
/// store.
///
/// - Its handshake answer (`fetch(Some(_))`) is always `default` and the
///   password: the store's `default` user has no password, so any password
///   authenticates.
/// - Its periodic answer (`fetch(None)`) carries the password only for the
///   credentials [granted](Self::grant), fails for the ones
///   [failed](Self::fail), never returns for the ones [hung](Self::hang), and
///   otherwise carries none — so a test decides exactly which tick sends.
/// - Its cadence (`refresh_interval()`) is `None` until
///   [`Self::start_ticking`]: the clients a fixture connects first (the
///   pools) read `None` at their router start and run no fred refresh.
///   Readers connect lazily, after the fixture, so they tick. A pool whose
///   router restarts after that (a reconnect) reads the cadence afresh and
///   starts a fred refresh of its own, so the provider's total of periodic
///   `fetch`es is not the readers' alone; [`assert_no_fred_refresh_on_reader`]
///   checks the reader's side instead.
pub(crate) struct ScriptedProvider {
    password: String,
    cadence: std::sync::Mutex<Option<Duration>>,
    grants: std::sync::atomic::AtomicUsize,
    failures: std::sync::atomic::AtomicUsize,
    hangs: std::sync::atomic::AtomicUsize,
    periodic: std::sync::atomic::AtomicUsize,
}

impl ScriptedProvider {
    pub(crate) fn new(password: &str) -> Arc<Self> {
        Arc::new(Self {
            password: password.to_owned(),
            cadence: std::sync::Mutex::new(None),
            grants: std::sync::atomic::AtomicUsize::new(0),
            failures: std::sync::atomic::AtomicUsize::new(0),
            hangs: std::sync::atomic::AtomicUsize::new(0),
            periodic: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// The next `n` periodic `fetch`es never return: a provider that hangs
    /// (a stuck token source). Only the caller's own bound ends them.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    pub(crate) fn hang(&self, n: usize) {
        self.hangs.fetch_add(n, std::sync::atomic::Ordering::SeqCst);
    }

    /// From now on `refresh_interval()` is [`REAUTH_TICK`]: a reader that
    /// connects (or reschedules) from here ticks. A client already connected
    /// keeps the `None` it read at its router start.
    pub(crate) fn start_ticking(&self) {
        *self.cadence.lock().expect("never poisoned") = Some(REAUTH_TICK);
    }

    /// The next `n` periodic answers carry the password.
    pub(crate) fn grant(&self, n: usize) {
        self.grants
            .fetch_add(n, std::sync::atomic::Ordering::SeqCst);
    }

    /// The next `n` periodic answers fail with `ErrorKind::Auth`.
    pub(crate) fn fail(&self, n: usize) {
        self.failures
            .fetch_add(n, std::sync::atomic::Ordering::SeqCst);
    }

    /// Granted credentials no periodic `fetch` has taken yet.
    #[cfg_attr(not(feature = "test-fault-injection"), allow(dead_code))]
    pub(crate) fn grants_left(&self) -> usize {
        self.grants.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Every periodic `fetch` so far, from any client: the readers' own ticks,
    /// and any fred refresh a pool started at a router (re)start after
    /// [`Self::start_ticking`]. Not a count of the readers' fetches alone, so
    /// only a diagnostic.
    pub(crate) fn periodic_fetches(&self) -> usize {
        self.periodic.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn take(counter: &std::sync::atomic::AtomicUsize) -> bool {
        counter
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |n| n.checked_sub(1),
            )
            .is_ok()
    }
}

#[async_trait::async_trait]
impl fred::types::config::CredentialProvider for ScriptedProvider {
    async fn fetch(
        &self,
        server: Option<&fred::types::config::Server>,
    ) -> Result<(Option<String>, Option<String>), fred::error::Error> {
        let user = Some("default".to_owned());
        if server.is_some() {
            return Ok((user, Some(self.password.clone())));
        }
        self.periodic
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if Self::take(&self.failures) {
            return Err(fred::error::Error::new(
                fred::error::ErrorKind::Auth,
                "scripted credential outage",
            ));
        }
        if Self::take(&self.hangs) {
            std::future::pending::<()>().await;
        }
        if Self::take(&self.grants) {
            return Ok((user, Some(self.password.clone())));
        }
        Ok((user, None))
    }

    fn refresh_interval(&self) -> Option<Duration> {
        *self.cadence.lock().expect("never poisoned")
    }
}

impl std::fmt::Debug for ScriptedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ScriptedProvider")
    }
}

/// Asserts fred schedules no credential refresh on reader client `reader`
/// (b.19i): the provider its config carries — the one fred reads at every
/// router (re)start, and the cadence it schedules its refresh from — has none,
/// although `provider` ticks by now. So every periodic `fetch` made for the
/// reader is one of its own ticks (`StreamMux::reauth_for_test`), however
/// many the pools' own refreshes add to the provider's total.
pub(crate) fn assert_no_fred_refresh_on_reader(reader: &Client, provider: &ScriptedProvider) {
    use fred::interfaces::ClientLike;
    use fred::types::config::CredentialProvider;
    assert_eq!(
        provider.refresh_interval(),
        Some(REAUTH_TICK),
        "precondition: the provider ticks"
    );
    let config = reader.client_config();
    let carried = config
        .credential_provider
        .as_ref()
        .expect("a reader client carries the provider for its handshake AUTH");
    assert_eq!(
        carried.refresh_interval(),
        None,
        "fred must schedule no refresh on a reader client: the provider its config carries must \
         have no cadence (the provider has seen {} periodic fetch(es) in all)",
        provider.periodic_fetches()
    );
}

/// Next delivery, or `None` if nothing arrives within `within` (or the stream
/// ended).
pub(crate) async fn recv_within(
    consumer: &LayerStreamConsumer,
    within: Duration,
) -> Option<(String, LayerNotification)> {
    tokio::time::timeout(within, consumer.recv())
        .await
        .ok()
        .flatten()
}

/// Receives `n` deliveries and returns their `sequence_num`s (`Deleted` → 0).
pub(crate) async fn recv_seqs(consumer: &LayerStreamConsumer, n: usize) -> Vec<u64> {
    let mut seqs = Vec::with_capacity(n);
    for i in 0..n {
        let (_, notification) = recv_within(consumer, SETTLE)
            .await
            .unwrap_or_else(|| panic!("delivery {i} of {n} did not arrive (got {seqs:?})"));
        seqs.push(match notification {
            LayerNotification::Update { sequence_num, .. } => sequence_num,
            LayerNotification::Deleted => 0,
        });
    }
    seqs
}

/// Asserts nothing further is delivered within `window`.
pub(crate) async fn assert_quiet(consumer: &LayerStreamConsumer, window: Duration, what: &str) {
    if let Some(extra) = recv_within(consumer, window).await {
        panic!("{what}: unexpected delivery {extra:?}");
    }
}

/// Polls the multiplexer snapshot until `cond` holds.
pub(crate) async fn await_mux(
    client: &ValkeyClient,
    what: &str,
    cond: impl Fn(&MuxSnapshot) -> bool,
) -> MuxSnapshot {
    let deadline = tokio::time::Instant::now() + SETTLE;
    loop {
        let snap = client.stream_mux.snapshot();
        if cond(&snap) {
            return snap;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what}: condition not reached; last snapshot {snap:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Connects stream reader 0 of `client` and leaves it with no subscription
/// and its read blocked on the server. Readers connect lazily, the first time
/// they serve a subscription (b.azh), and standalone routing sends object 0 to
/// reader 0 (`object_id % readers`).
///
/// The subscription is dropped only once the server shows the reader's
/// connection blocked in its `XREAD` ([`await_reader_blocked_in_xread`]):
/// before that the reader may not have issued its read, or fred may not yet
/// have marked it blocked, and a command sent then would queue behind the
/// 30 s BLOCK. A subscription leaving does not wake the read, so it stays
/// blocked, and the next command on the reader's client interrupts it
/// (`Blocking::Interrupt`); the reader, with nothing to read, then goes idle.
/// The subscription names a fresh key prefix and writes nothing, so it leaves
/// no key behind.
pub(crate) async fn connect_reader_zero(client: &ValkeyClient) {
    let schema = KeySchema::new(&format!("test_{}:", uuid::Uuid::new_v4()));
    let subscription = consume_layer_stream(
        client,
        &schema,
        0,
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + SETTLE,
    )
    .await
    .expect("the subscription must start");
    await_mux(
        client,
        "reader 0 connected to serve its subscription",
        |s| s.readers[0].connected,
    )
    .await;
    let reader = client
        .stream_mux
        .reader_client_for_test(0)
        .expect("reader 0 holds a connection");
    await_reader_blocked_in_xread(client.pool.next(), &reader).await;
    drop(subscription);
}

/// Polls (for one [`SETTLE`]) until the server shows `reader`'s one connection
/// blocked in an `XREAD` (`CLIENT LIST`: `cmd=xread`, flag `b`), asked over
/// `observer`. A reader's XREAD count rises before the command is sent, and
/// fred marks the connection blocked only just after writing it, so neither
/// the count nor a connected reader proves the read is parked: a command sent
/// in that window queues behind the BLOCK instead of interrupting it. Once the
/// server shows it blocked, fred has marked it too, and any command on the
/// reader's client interrupts it (`Blocking::Interrupt`).
pub(crate) async fn await_reader_blocked_in_xread(observer: &Client, reader: &Client) {
    let ids = reader.connection_ids();
    assert_eq!(
        ids.len(),
        1,
        "a reader client holds one connection; got {ids:?}"
    );
    let id = *ids.values().next().expect("checked above");
    let field = |line: &str, name: &str| -> String {
        line.split_whitespace()
            .find_map(|kv| kv.strip_prefix(&format!("{name}=")[..]).map(str::to_owned))
            .unwrap_or_default()
    };
    let deadline = tokio::time::Instant::now() + SETTLE;
    loop {
        let listing: String = observer
            .client_list::<String, String>(None, Some(vec![id.to_string()]))
            .await
            .expect("CLIENT LIST ID must succeed");
        if listing
            .lines()
            .any(|l| field(l, "cmd") == "xread" && field(l, "flags").contains('b'))
        {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "reader connection {id} never blocked in its XREAD; last CLIENT LIST:\n{listing}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Waits until the readers have issued at least one XREAD and then none for
/// 150 ms: every reader is parked in a block over its current key set.
pub(crate) async fn await_readers_parked(client: &ValkeyClient) {
    await_mux(client, "first XREAD", |s| s.xreads() >= 1).await;
    let deadline = tokio::time::Instant::now() + SETTLE;
    let mut last = client.stream_mux.snapshot().xreads();
    loop {
        tokio::time::sleep(Duration::from_millis(150)).await;
        let now = client.stream_mux.snapshot().xreads();
        if now == last {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "readers never went quiet"
        );
        last = now;
    }
}

#[cfg(feature = "test-fault-injection")]
pub(crate) use faults::*;

#[cfg(feature = "test-fault-injection")]
mod faults {
    use super::*;
    use crate::grpc::log_capture::ThreadCapture;
    use live_edit::storage::notifications::{consume_layer_stream, StreamCursor};
    use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMuxKnobs};

    /// Bounded observation window for the fault suites' negative checks ("no
    /// forced reconnect happened", "no keepalive tick happened"): many
    /// multiples of the shortened timer each one watches, so it can only let
    /// a regression through under extreme load, never fail a correct build.
    pub(crate) const QUIET: Duration = Duration::from_millis(500);

    /// `-UNBLOCKED`, as the server replies to `CLIENT UNBLOCK <id> ERROR`.
    pub(crate) const UNBLOCKED_REPLY: &str = "UNBLOCKED client unblocked via CLIENT UNBLOCK";

    /// The DEBUG line a reader logs when it backs off after a whole-reader
    /// failure.
    pub(crate) const WHOLE_READER_BACKOFF_MSG: &str =
        "backing off before the next multiplexed XREAD after a whole-reader failure";

    /// The WARN a reader logs on a wake-ack verdict.
    pub(crate) const WAKE_ACK_WEDGE_MSG: &str =
        "multiplexed XREAD did not return within the wake-ack timeout of its wake landing; \
         treating the reader connection as wedged";

    /// `live_edit_stream_mux_streams`: the streams the readers read.
    pub(crate) const STREAMS: &str = "live_edit_stream_mux_streams";

    /// Subscribes to `object_id`'s layer stream from `0-0`.
    pub(crate) async fn consume(tv: &TestValkey, object_id: u64) -> LayerStreamConsumer {
        consume_layer_stream(
            tv.client(),
            tv.key_schema(),
            object_id,
            StreamCursor::From("0-0".into()),
            tokio::time::Instant::now() + std::time::Duration::from_secs(2),
        )
        .await
        .unwrap()
    }

    /// An injected fred error of `kind`.
    pub(crate) fn fault(kind: fred::error::ErrorKind, detail: &'static str) -> MuxFault {
        MuxFault::Err(fred::error::Error::new(kind, detail))
    }

    /// Asserts the subscription ended with `reason` and a retryable
    /// `Unavailable` — carrying the shutdown retry hint for the `shutdown`
    /// ending and no refusal detail for any other — and returns that status's
    /// message.
    pub(crate) async fn assert_ends_with(consumer: &LayerStreamConsumer, reason: &str) -> String {
        assert!(
            recv_within(consumer, SETTLE).await.is_none(),
            "the subscription must end"
        );
        let exit = consumer
            .take_terminal_exit()
            .expect("a reader-side ending must never read as a clean close");
        assert_eq!(exit.reason, reason);
        crate::common::refusal_detail::mux_ending_message(exit.error, reason)
    }

    /// The captured lines carrying `msg`, once at least `n` have appeared.
    pub(crate) async fn await_lines(
        capture: &ThreadCapture,
        msg: &str,
        n: usize,
    ) -> Vec<serde_json::Value> {
        let deadline = tokio::time::Instant::now() + SETTLE;
        loop {
            let captured = capture.snapshot();
            let lines: Vec<serde_json::Value> =
                captured.with_message(msg).into_iter().cloned().collect();
            if lines.len() >= n {
                return lines;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "expected {n} `{msg}` line(s); full capture:\n{}",
                captured.text
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Queues `n` copies of a fred error for the next XREADs.
    pub(crate) fn push_faults(
        client: &ValkeyClient,
        n: usize,
        kind: fred::error::ErrorKind,
        detail: &'static str,
    ) {
        client.stream_mux.push_faults_for_test(
            MuxFaultSite::Xread,
            (0..n).map(|_| MuxFault::Err(fred::error::Error::new(kind.clone(), detail))),
        );
    }

    pub(crate) fn push_timeout_faults(client: &ValkeyClient, n: usize) {
        push_faults(
            client,
            n,
            fred::error::ErrorKind::Timeout,
            "synthetic wedged-connection timeout",
        );
    }

    /// Waits until every queued fault has been consumed by an XREAD.
    pub(crate) async fn await_faults_consumed(client: &ValkeyClient) {
        let deadline = tokio::time::Instant::now() + SETTLE;
        while client
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Xread)
            > 0
        {
            assert!(
                tokio::time::Instant::now() < deadline,
                "{} injected fault(s) were never consumed — no reader is issuing XREADs",
                client
                    .stream_mux
                    .pending_faults_for_test(MuxFaultSite::Xread)
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Current knobs with a 20 ms transient backoff, so retry tests run fast.
    pub(crate) fn fast_backoff_knobs(client: &ValkeyClient) -> StreamMuxKnobs {
        StreamMuxKnobs {
            backoff_initial: Duration::from_millis(20),
            backoff_cap: Duration::from_millis(20),
            ..client.stream_mux.knobs()
        }
    }
}
