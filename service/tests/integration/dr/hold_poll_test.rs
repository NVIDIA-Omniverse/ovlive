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

//! The fleet hold poll against a real store (`live_edit::dr::hold`; SDD
//! SR-2.2, SR-2.7, SR-4.1, SR-5.1, SR-6.5).
//!
//! The poll reads the hold marker in one `EVALSHA` of `dr_hold_poll.lua`,
//! turns the reply into a verdict, and publishes it on a `watch` channel only
//! when the read answered. These tests drive the real script against the
//! fixture's Valkey:
//!
//! - **Verdicts.** No marker reads `Serving`; a marker reads `Held` with its
//!   `incident`, `reason` and `detected_at_ms`; the running poll follows a
//!   marker written and then removed; and every answered read republishes,
//!   so the observation's instant is always the last successful read.
//! - **Errors are never a verdict (SR-2.7).** A `-LOADING`, `-BUSY` or
//!   `CLUSTERDOWN` reply, a stalled connection ([`ConnStall`]) and fred's own
//!   command timeout each make the startup evaluation fail, never answer
//!   `Serving`, and make the running poll publish nothing across several
//!   failed reads: the channel's version does not move and its observation
//!   keeps its instant. Each poll then publishes again once the fault is
//!   lifted, so the silence is the fault's doing, not a dead poll. A failed
//!   redirect is not staged: the poll handles every failed read through the
//!   one branch these five shapes exercise, whatever the error.
//! - **The per-call bound (SR-6.5 "Poll bounds").** A single read against a
//!   stalled connection ends with the poll's own `Timeout` at its bound,
//!   never at fred's longer command timeout; and the running poll's reads, on
//!   a stalled client with no command timeout at all, end only because the
//!   poll bounds each one. Which of the command timeout and the interval
//!   sets the bound is pinned where it is computed
//!   (`dr::hold::verdict::tests`, and
//!   [`hold_poll_settings_from_config_bound_each_read_by_the_smaller_timeout`]).
//! - **Shutdown.** Cancelling the token ends the poll even while its read is
//!   stuck on a connection with no command timeout at all.
//! - **The counter's slot.** A cluster-gated test checks that the script runs
//!   on the primary serving the object-id counter, the slot the full hold
//!   predicate will read the counter in. The unit-level check is
//!   `storage::keys::tests::dr_hold_marker_shares_the_object_id_counters_slot`.
//!
//! Store replies are staged with [`ScriptReplyProxy`] on the poll's own
//! script frames: they name the test's marker key, which carries the test's
//! unique prefix, so no other client's command can take the fault, and the
//! proxy's hit count shows each one was taken. Every marker is written
//! through `test_support::write_hold_marker`, so each test seeds the one
//! shape the service will see.
//!
//! # Waits
//!
//! No assertion is on wall-clock time; every wait is a deadline a healthy
//! store never reaches. [`WAIT_BUDGET`] (20 s) bounds each awaited event (a
//! publication, a count of staged faults taken, a stalled connection's
//! received bytes, the poll's end), [`HANG_BUDGET`] (10 s) each read the
//! test makes itself, and `test_support::CONN_STALL_WAIT` (10 s) the parking
//! and the release of each stalled connection. The stall tests stall a
//! one-connection client of their own ([`own_client`]), so each pays one
//! park and one release.
//!
//! Every wait at its full budget, per test, against nextest's 120 s
//! hang-killer (fixture setup and connects not counted):
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | stalled connection, fred timeout | 2 reads + park + 3 events + release | 100 s |
//! | `-LOADING`, `-BUSY`, `CLUSTERDOWN` | 2 reads + 3 events | 80 s |
//! | marker appears and is removed, republish | 1 read + 3 events | 70 s |
//! | shutdown | 1 read + park + 2 events + release | 70 s |
//! | cluster slot | `MONITOR` acknowledgement + 5 s read bound + `MONITOR` drain (20 s each) | 45 s |
//! | startup Held / Serving | 3 reads | 30 s |
//! | per-call bound | park + 1 read + release | 30 s |
//! | the rest | at most one read | 10 s |
//!
//! The `CLUSTERDOWN` test is the slow one when it passes, about 10 s: it
//! polls at the default 2 s interval (see [`CLUSTERDOWN_INTERVAL`]).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::interfaces::{ClientLike, ClusterInterface, KeysInterface};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use live_edit::dr::hold::{
    evaluate_hold_at_startup, poll_hold_once, spawn_hold_poll, HoldObservation, HoldPollError,
    HoldPollSettings, HoldVerdict,
};
use live_edit::storage::client::{
    is_store_not_serving_reply, ValkeyClient, STORE_NOT_SERVING_SHAPES,
};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::test_support::{
    connect_test_client, release_all, stall_pool, ConnStall, HoldMarkerSeed, TestValkey,
};

use super::hold_support::answering;
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, monitor_fence, test_valkey_addr, MonitorTap, ProxyAction,
    ProxyRule, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// The poll interval of the tests that run the poll: short, so several reads
/// happen in a fraction of a second.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The fixture clients' command timeout (`valkey_command_timeout_ms` in
/// `TestValkey` and `common::test_config`).
const FIXTURE_COMMAND_TIMEOUT_MS: u64 = 5_000;

/// Bound on each awaited event. Unreachable on a healthy store: the slowest
/// event, the `CLUSTERDOWN` test's (`FAILED_READS` + 1) refused reads at the
/// 2 s interval, comes about 8.5 s after the poll starts.
const WAIT_BUDGET: Duration = Duration::from_secs(20);

/// Bound on one read the test makes itself. Every read under test ends at its
/// own per-call bound or at fred's 5 s command timeout, both below this.
const HANG_BUDGET: Duration = Duration::from_secs(10);

/// Failed reads the running poll must make before the test asserts it
/// published nothing.
const FAILED_READS: usize = 3;

/// Gap between two probes of a condition the test waits for.
const PROBE_EVERY: Duration = Duration::from_millis(5);

/// A `detected_at_ms` in the past, for seeds that pin it.
const DETECTED_AT_MS: u64 = 1_760_000_000_123;

/// The reply of a cluster that has lost a slot's coverage.
const CLUSTERDOWN_REPLY: &str = "CLUSTERDOWN The cluster is down";

/// The poll interval of the `-LOADING` and `-BUSY` tests: room for the
/// store's refusal to arrive well inside the per-call bound on a loaded
/// runner.
const REFUSAL_INTERVAL: Duration = Duration::from_millis(500);

/// The poll interval of the `CLUSTERDOWN` test: the default, 2 s. fred
/// treats `CLUSTERDOWN` as a dead connection, reconnects and re-sends the
/// read, up to its attempt cap (three attempts in about 0.3 s, measured
/// against the test store), so the bound must leave room for all of them.
const CLUSTERDOWN_INTERVAL: Duration =
    Duration::from_millis(live_edit::config::DR_HOLD_POLL_INTERVAL_MS_DEFAULT);

/// The timing of the tests that run the poll: [`POLL_INTERVAL`], so the
/// per-call bound is the interval (below the fixture's command timeout).
fn settings() -> HoldPollSettings {
    HoldPollSettings::new(POLL_INTERVAL, FIXTURE_COMMAND_TIMEOUT_MS)
}

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

/// The startup evaluation against a store that answers.
async fn startup(client: &ValkeyClient, keys: &KeySchema) -> HoldObservation {
    tokio::time::timeout(
        HANG_BUDGET,
        evaluate_hold_at_startup(client, keys, answering(FIXTURE_COMMAND_TIMEOUT_MS)),
    )
    .await
    .expect("the startup evaluation must return within its bound")
    .expect("the startup evaluation must answer against a healthy store")
}

/// The startup evaluation, bounded by `settings`, against a store that does
/// not answer: its error.
async fn startup_error(
    client: &ValkeyClient,
    keys: &KeySchema,
    settings: HoldPollSettings,
    what: &str,
) -> HoldPollError {
    let result = tokio::time::timeout(
        HANG_BUDGET,
        evaluate_hold_at_startup(client, keys, settings),
    )
    .await
    .unwrap_or_else(|_| panic!("{what}: the startup evaluation must return within its bound"));
    match result {
        Ok(observation) => panic!(
            "{what}: a failed read must be an error, never a verdict (fail-closed startup); got \
             {observation:?}"
        ),
        Err(e) => e,
    }
}

/// Wait until `cond` holds, probing every [`PROBE_EVERY`].
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT_BUDGET;
    while !cond() {
        assert!(Instant::now() < deadline, "{what} within {WAIT_BUDGET:?}");
        tokio::time::sleep(PROBE_EVERY).await;
    }
}

/// A running hold poll and the token that stops it.
struct Poll {
    rx: watch::Receiver<HoldObservation>,
    handle: JoinHandle<()>,
    shutdown: CancellationToken,
}

impl Poll {
    fn spawn(
        client: &ValkeyClient,
        keys: &KeySchema,
        settings: HoldPollSettings,
        initial: HoldObservation,
    ) -> Self {
        let shutdown = CancellationToken::new();
        let (rx, handle) = spawn_hold_poll(
            client.clone(),
            keys.clone(),
            settings,
            initial,
            shutdown.clone(),
        );
        Self {
            rx,
            handle,
            shutdown,
        }
    }

    /// The next observation the poll publishes.
    async fn next(&mut self, what: &str) -> HoldObservation {
        tokio::time::timeout(WAIT_BUDGET, self.rx.changed())
            .await
            .unwrap_or_else(|_| panic!("{what}: the poll published nothing within {WAIT_BUDGET:?}"))
            .unwrap_or_else(|_| panic!("{what}: the poll ended"));
        self.rx.borrow_and_update().clone()
    }

    /// The first published observation whose verdict is `want`.
    async fn until_verdict(&mut self, want: &HoldVerdict, what: &str) -> HoldObservation {
        tokio::time::timeout(WAIT_BUDGET, self.rx.wait_for(|o| o.verdict == *want))
            .await
            .unwrap_or_else(|_| panic!("{what}: no {want:?} within {WAIT_BUDGET:?}"))
            .unwrap_or_else(|_| panic!("{what}: the poll ended"))
            .clone()
    }

    /// Nothing has been published since `last`: the channel's version has not
    /// moved, and it still holds `last`, instant included. `has_changed` tells
    /// a publication apart from a republished identical value, which a
    /// comparison of verdicts alone could not.
    fn assert_nothing_published_since(&self, last: &HoldObservation, what: &str) {
        assert!(
            !self.handle.is_finished(),
            "{what}: the poll must still run"
        );
        let changed = self
            .rx
            .has_changed()
            .unwrap_or_else(|_| panic!("{what}: the poll ended"));
        assert!(
            !changed,
            "{what}: a failed read published a value; an error is never a verdict (SR-2.7); \
             now {:?}, was {last:?}",
            *self.rx.borrow()
        );
        assert_eq!(
            *self.rx.borrow(),
            *last,
            "{what}: the observation changed with no publication"
        );
    }

    /// Cancel the token and wait for the poll to end.
    async fn stop(self) {
        self.shutdown.cancel();
        tokio::time::timeout(WAIT_BUDGET, self.handle)
            .await
            .expect("the poll must end once its shutdown token is cancelled")
            .expect("the poll task must not panic");
    }
}

/// A client of the test's own on the fixture's store: one data connection,
/// whose stall parks and releases within one `CONN_STALL_WAIT` each, and a
/// command timeout of `command_timeout_ms` (0: none). A stall test stalls
/// this client, never the fixture's pool, so a failure leaves the fixture's
/// pool answering for its cleanup.
async fn own_client(command_timeout_ms: u64) -> ValkeyClient {
    let mut config = crate::common::test_config();
    config.valkey_pool_size = 1;
    config.valkey_command_timeout_ms = command_timeout_ms;
    connect_test_client(&config)
        .await
        .expect("the test's own client must connect")
}

/// [`stall_pool`] over `client`'s data connection, observed from the
/// fixture's maintenance pool.
async fn stall_own_client(tv: &TestValkey, client: &ValkeyClient, label: &str) -> Vec<ConnStall> {
    let observer = tv.client().maintenance_pool.next().clone();
    let stalls = stall_pool(
        &client.pool,
        &observer,
        &format!("{}conn-stall:{label}:", tv.prefix()),
    )
    .await;
    assert_eq!(
        stalls.len(),
        1,
        "the test's own client has one data connection"
    );
    stalls
}

/// Every byte the store has read from the stalled connections, commands
/// queued behind each stall included.
async fn bytes_received(stalls: &[ConnStall]) -> u64 {
    let mut total = 0;
    for stall in stalls {
        total += stall
            .net_in()
            .await
            .expect("a stalled connection must stay open");
    }
    total
}

/// Wait until the stalled connections have received `reads` more reads than
/// they had at `base` bytes. Every read the poll sends is the same frame (one
/// script, one key), so the first growth is a whole number of frames, and
/// `reads` times it is at least `reads` frames. The poll sends a read only
/// once the one before it has returned, so `reads` frames mean `reads - 1`
/// reads have failed.
async fn wait_for_queued_reads(stalls: &[ConnStall], base: u64, reads: u64, what: &str) {
    let deadline = Instant::now() + WAIT_BUDGET;
    let first = loop {
        let now = bytes_received(stalls).await;
        if now > base {
            break now - base;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: the poll sent no read within {WAIT_BUDGET:?}"
        );
        tokio::time::sleep(PROBE_EVERY).await;
    };
    while bytes_received(stalls).await - base < reads * first {
        assert!(
            Instant::now() < deadline,
            "{what}: the poll sent fewer than {reads} reads within {WAIT_BUDGET:?}"
        );
        tokio::time::sleep(PROBE_EVERY).await;
    }
}

// ── Staged store replies ────────────────────────────────────────────────────

/// The proxy's staged fault: while armed, every `EVALSHA` / `EVAL` frame
/// naming `marker` meets the action; disarmed, every frame passes through.
#[derive(Clone, Default)]
struct Stage(Arc<Mutex<Option<ProxyAction>>>);

impl Stage {
    fn rule(&self, marker: String) -> ProxyRule {
        let staged = Arc::clone(&self.0);
        Arc::new(move |args: &[Vec<u8>]| {
            let name = args.first()?.to_ascii_uppercase();
            if name != b"EVALSHA" && name != b"EVAL" {
                return None;
            }
            if !args.iter().any(|a| a.as_slice() == marker.as_bytes()) {
                return None;
            }
            staged.lock().unwrap().clone()
        })
    }

    fn arm(&self, action: ProxyAction) {
        *self.0.lock().unwrap() = Some(action);
    }

    fn disarm(&self) {
        *self.0.lock().unwrap() = None;
    }
}

/// The fixture's store, and a client whose every connection runs through a
/// [`ScriptReplyProxy`] staging faults on the hold poll's script frames.
/// Seeding and cleanup go straight to the store; only the poll meets the
/// proxy.
struct Proxied {
    // Declared first, so it drops before the proxy it connects through.
    client: ValkeyClient,
    stage: Stage,
    proxy: ScriptReplyProxy,
    tv: TestValkey,
}

impl Proxied {
    async fn new() -> Self {
        let tv = TestValkey::setup().await;
        let stage = Stage::default();
        let proxy = ScriptReplyProxy::start(
            test_valkey_addr(),
            stage.rule(tv.key_schema().dr_hold_marker()),
        )
        .await;
        let client = client_through(&proxy).await;
        Self {
            client,
            stage,
            proxy,
            tv,
        }
    }
}

/// With `reply` staged on the poll's reads, polled every `interval`: the
/// startup evaluation fails with the store's refusal as `expect` says, and
/// the running poll publishes nothing across [`FAILED_READS`] refused reads;
/// lifted, the poll publishes again.
///
/// `interval` must leave fred room to finish every attempt it makes at one
/// read inside the per-call bound, so the startup read ends with fred's error
/// and leaves no retry behind. The proxy's hits for that read are then the
/// most any read can take: every read meets the same refusal and fred handles
/// it the same way, returning `-LOADING` and `-BUSY` at once and re-sending
/// after `CLUSTERDOWN` up to its per-command attempt cap. So
/// (`FAILED_READS` + 1) × that many further hits mean the poll sent at least
/// `FAILED_READS` + 1 reads, and as it sends a read only once the one before
/// it returned, at least [`FAILED_READS`] of them failed before the
/// assertion.
async fn staged_reply_publishes_no_verdict(
    reply: &str,
    interval: Duration,
    what: &str,
    expect: impl Fn(&HoldPollError) -> bool,
) {
    let fx = Proxied::new().await;
    let keys = fx.tv.key_schema();
    let settings = HoldPollSettings::new(interval, FIXTURE_COMMAND_TIMEOUT_MS);
    let initial = startup(&fx.client, keys).await;
    assert_eq!(
        initial.verdict,
        HoldVerdict::Serving,
        "{what}: precondition"
    );
    fx.stage.arm(ProxyAction::Eval(error_reply_script(reply)));

    let err = startup_error(&fx.client, keys, settings, what).await;
    assert!(expect(&err), "{what}: unexpected startup error {err:?}");
    let per_read = fx.proxy.hits();
    assert!(
        per_read >= 1,
        "{what}: the startup read met the staged reply"
    );

    let mut poll = Poll::spawn(&fx.client, keys, settings, initial.clone());
    let reads = FAILED_READS + 1;
    wait_until(
        &format!("{what}: the poll's reads meet the staged reply {reads} × {per_read} times"),
        || fx.proxy.hits() >= per_read + reads * per_read,
    )
    .await;
    poll.assert_nothing_published_since(&initial, what);

    fx.stage.disarm();
    let recovered = poll.next(&format!("{what}, lifted")).await;
    assert_eq!(recovered.verdict, HoldVerdict::Serving, "{what}, lifted");
    assert!(
        recovered.at > initial.at,
        "{what}, lifted: the new observation is stamped at its read"
    );
    poll.stop().await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Verdicts
// ═════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poll_hold_once_without_a_marker_observes_serving_stamped_at_the_read() {
    let tv = TestValkey::setup().await;
    let before = Instant::now();

    let observation = poll_hold_once(
        tv.client(),
        tv.key_schema(),
        answering(FIXTURE_COMMAND_TIMEOUT_MS).call_timeout,
    )
    .await
    .expect("the read must answer");

    assert_eq!(observation.verdict, HoldVerdict::Serving);
    assert!(
        observation.at >= before && observation.at <= Instant::now(),
        "the observation is stamped when its read answered"
    );
}

/// `Held` carries the marker's `incident`, `reason` and `detected_at_ms`.
/// Each seeded field is distinct, so a field read from the wrong slot of the
/// script's reply fails the comparison.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poll_hold_once_with_a_marker_observes_held_with_its_incident_reason_and_detection_time() {
    let tv = TestValkey::setup().await;
    let seed = HoldMarkerSeed::default()
        .with_incident("inc-present")
        .with_reason("reason-present")
        .detected_at(DETECTED_AT_MS);
    assert_eq!(tv.write_hold_marker(&seed).await, DETECTED_AT_MS);

    let observation = poll_hold_once(
        tv.client(),
        tv.key_schema(),
        answering(FIXTURE_COMMAND_TIMEOUT_MS).call_timeout,
    )
    .await
    .expect("the read must answer");

    assert_eq!(
        observation.verdict,
        HoldVerdict::Held {
            incident: Some("inc-present".to_owned()),
            reason: Some("reason-present".to_owned()),
            evidence: Some(seed.evidence.clone()),
            detected_at_ms: Some(DETECTED_AT_MS),
        }
    );
}

/// A marker written with the store's clock carries that clock's reading
/// through the read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poll_hold_once_with_a_marker_on_the_stores_clock_observes_its_detection_time() {
    let tv = TestValkey::setup().await;
    let seed = HoldMarkerSeed::default();
    let detected_at_ms = tv.write_hold_marker(&seed).await;

    let observation = poll_hold_once(
        tv.client(),
        tv.key_schema(),
        answering(FIXTURE_COMMAND_TIMEOUT_MS).call_timeout,
    )
    .await
    .expect("the read must answer");

    assert_eq!(
        observation.verdict,
        HoldVerdict::Held {
            incident: Some(seed.incident.clone()),
            reason: Some(seed.reason.clone()),
            evidence: Some(seed.evidence.clone()),
            detected_at_ms: Some(detected_at_ms),
        }
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evaluate_hold_at_startup_answers_held_with_the_marker_and_serving_without_it() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    assert_eq!(
        startup(tv.client(), keys).await.verdict,
        HoldVerdict::Serving
    );

    let seed = HoldMarkerSeed::default()
        .with_incident("inc-startup")
        .detected_at(DETECTED_AT_MS);
    tv.write_hold_marker(&seed).await;
    assert_eq!(
        startup(tv.client(), keys).await.verdict,
        HoldVerdict::Held {
            incident: Some("inc-startup".to_owned()),
            reason: Some(seed.reason.clone()),
            evidence: Some(seed.evidence.clone()),
            detected_at_ms: Some(DETECTED_AT_MS),
        }
    );

    assert!(tv.remove_hold_marker().await, "the marker existed");
    assert_eq!(
        startup(tv.client(), keys).await.verdict,
        HoldVerdict::Serving
    );
}

/// A key of the wrong type at the marker's name makes the script fail: an
/// error, so a process must not serve, never a `Serving` read of a key it
/// could not decode.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn evaluate_hold_at_startup_marker_of_the_wrong_type_is_an_error_never_serving() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let _: () = tv
        .client()
        .pool
        .set(keys.dr_hold_marker(), "not-a-hash", None, None, false)
        .await
        .expect("SET of a string at the marker's name must succeed");

    let err = startup_error(
        tv.client(),
        keys,
        answering(FIXTURE_COMMAND_TIMEOUT_MS),
        "a string at the marker's name",
    )
    .await;

    assert!(
        matches!(err, HoldPollError::Store(_)),
        "the script's own failure is a store error, got {err:?}"
    );
}

/// The running poll follows the store: `Held`, with the new marker's fields,
/// once one is written, and `Serving` on the next read after it is removed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_publishes_held_once_a_marker_appears_and_serving_once_it_is_removed() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let initial = startup(tv.client(), keys).await;
    assert_eq!(initial.verdict, HoldVerdict::Serving, "precondition");
    let mut poll = Poll::spawn(tv.client(), keys, settings(), initial);

    let seed = HoldMarkerSeed::default()
        .with_incident("inc-appears")
        .with_reason("reason-appears")
        .detected_at(DETECTED_AT_MS);
    tv.write_hold_marker(&seed).await;
    let held = HoldVerdict::Held {
        incident: Some("inc-appears".to_owned()),
        reason: Some("reason-appears".to_owned()),
        evidence: Some(seed.evidence.clone()),
        detected_at_ms: Some(DETECTED_AT_MS),
    };
    let held_at = poll.until_verdict(&held, "marker written").await.at;

    assert!(tv.remove_hold_marker().await, "the marker existed");
    let serving = poll
        .until_verdict(&HoldVerdict::Serving, "marker removed")
        .await;
    assert!(
        serving.at > held_at,
        "the Serving read follows the Held one"
    );
    poll.stop().await;
}

/// Every answered read publishes a new observation, the verdict unchanged
/// included, so the channel's instant is always the last successful read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_republishes_an_unchanged_verdict_with_each_reads_instant() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let initial = startup(tv.client(), keys).await;
    let mut poll = Poll::spawn(tv.client(), keys, settings(), initial.clone());

    let first = poll.next("first read").await;
    let second = poll.next("second read").await;

    assert_eq!(first.verdict, HoldVerdict::Serving);
    assert_eq!(second.verdict, HoldVerdict::Serving);
    assert!(
        first.at > initial.at,
        "the first read's instant follows startup's"
    );
    assert!(second.at > first.at, "each read stamps its own instant");
    poll.stop().await;
}

#[test]
fn hold_poll_settings_from_config_bound_each_read_by_the_smaller_timeout() {
    let mut config = crate::common::test_config();
    config.dr_hold_poll_interval_ms = 1_500;
    config.valkey_command_timeout_ms = 400;
    assert_eq!(
        HoldPollSettings::from_config(&config),
        HoldPollSettings {
            interval: Duration::from_millis(1_500),
            call_timeout: Duration::from_millis(400),
        },
        "a command timeout below the interval bounds each read"
    );

    config.valkey_command_timeout_ms = 5_000;
    assert_eq!(
        HoldPollSettings::from_config(&config).call_timeout,
        Duration::from_millis(1_500),
        "an interval below the command timeout bounds each read"
    );

    config.valkey_command_timeout_ms = 0;
    assert_eq!(
        HoldPollSettings::from_config(&config).call_timeout,
        Duration::from_millis(1_500),
        "with fred's timeout off, the interval bounds each read"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Errors are never a verdict (SR-2.7)
// ═════════════════════════════════════════════════════════════════════════════

/// The store's not-serving refusal, as the shared classifier reads it.
fn is_not_serving_refusal(e: &HoldPollError) -> bool {
    matches!(e, HoldPollError::Store(e) if is_store_not_serving_reply(e))
        && e.class() == "store_not_serving"
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_loading_reply_publishes_no_verdict_and_startup_fails() {
    staged_reply_publishes_no_verdict(
        STORE_NOT_SERVING_SHAPES[0],
        REFUSAL_INTERVAL,
        "-LOADING",
        is_not_serving_refusal,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_busy_reply_publishes_no_verdict_and_startup_fails() {
    staged_reply_publishes_no_verdict(
        STORE_NOT_SERVING_SHAPES[2],
        REFUSAL_INTERVAL,
        "-BUSY",
        is_not_serving_refusal,
    )
    .await;
}

/// Slow (about 10 s): it polls at the default 2 s interval, which fred's
/// reconnect-and-retry on `CLUSTERDOWN` needs (see [`CLUSTERDOWN_INTERVAL`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_clusterdown_reply_publishes_no_verdict_and_startup_fails() {
    staged_reply_publishes_no_verdict(
        CLUSTERDOWN_REPLY,
        CLUSTERDOWN_INTERVAL,
        "CLUSTERDOWN",
        |e| {
            matches!(e, HoldPollError::Store(e) if *e.kind() == fred::error::ErrorKind::Cluster)
                && e.class() == "transient"
        },
    )
    .await;
}

/// The poll's data connection stalled: the startup read and each of the
/// running poll's reads end at the per-call bound, and nothing is published
/// until the connection answers again.
///
/// The client has no command timeout, so only the poll's own bound can end a
/// stalled read: a running poll that bounded its reads by anything longer
/// than the test's waits would leave its first read stuck, never send the
/// next, and fail [`wait_for_queued_reads`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_stalled_connection_publishes_no_verdict_and_startup_fails() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let client = own_client(0).await;
    let initial = startup(&client, keys).await;
    let stalls = stall_own_client(&tv, &client, "poll").await;

    let err = startup_error(&client, keys, settings(), "stalled").await;
    assert!(
        matches!(err, HoldPollError::Timeout(d) if d == settings().call_timeout),
        "the stalled startup read ends at the per-call bound, got {err:?}"
    );
    let base = bytes_received(&stalls).await;
    let mut poll = Poll::spawn(&client, keys, settings(), initial.clone());
    wait_for_queued_reads(&stalls, base, FAILED_READS as u64 + 1, "stalled").await;
    poll.assert_nothing_published_since(&initial, "stalled");

    release_all(stalls).await;
    let recovered = poll.next("stall released").await;
    assert_eq!(recovered.verdict, HoldVerdict::Serving);
    assert!(recovered.at > initial.at);
    poll.stop().await;
}

/// fred's own command timeout, shorter than the per-call bound here, fails
/// each read first: still no verdict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_fred_command_timeout_publishes_no_verdict_and_startup_fails() {
    /// The client's command timeout: well below the per-call bound, so
    /// fred's timer always fires first.
    const FRED_TIMEOUT_MS: u64 = 50;
    /// The poll's interval, and its per-call bound.
    const INTERVAL: Duration = Duration::from_millis(500);

    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let client = own_client(FRED_TIMEOUT_MS).await;
    let settings = HoldPollSettings::new(INTERVAL, FIXTURE_COMMAND_TIMEOUT_MS);
    assert_eq!(settings.call_timeout, INTERVAL, "precondition");
    // The poll's seed, read on the fixture's client: its 5 s timeout cannot
    // turn the answer into an error on a loaded runner.
    let initial = startup(tv.client(), keys).await;
    let stalls = stall_own_client(&tv, &client, "fred-timeout").await;

    let err = startup_error(&client, keys, settings, "fred timeout").await;
    assert!(
        matches!(&err, HoldPollError::Store(e) if *e.kind() == fred::error::ErrorKind::Timeout)
            && err.class() == "command_timeout",
        "fred's command timeout ends the read, got {err:?}"
    );
    let base = bytes_received(&stalls).await;
    let mut poll = Poll::spawn(&client, keys, settings, initial.clone());
    wait_for_queued_reads(&stalls, base, FAILED_READS as u64 + 1, "fred timeout").await;
    poll.assert_nothing_published_since(&initial, "fred timeout");

    release_all(stalls).await;
    let recovered = poll.next("stall released").await;
    assert_eq!(recovered.verdict, HoldVerdict::Serving);
    poll.stop().await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The per-call bound (SR-6.5 "Poll bounds")
// ═════════════════════════════════════════════════════════════════════════════

/// An interval below the command timeout bounds the read: on a stalled
/// connection it ends with the poll's own `Timeout` at the interval, not at
/// the client's 5 s command timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poll_hold_once_stalled_read_ends_at_the_interval_when_it_is_below_the_command_timeout() {
    let settings = HoldPollSettings::new(Duration::from_millis(250), 60_000);
    assert_eq!(
        settings.call_timeout,
        Duration::from_millis(250),
        "precondition"
    );
    let tv = TestValkey::setup().await;
    let client = own_client(FIXTURE_COMMAND_TIMEOUT_MS).await;
    let stalls = stall_own_client(&tv, &client, "per-call").await;

    let result = tokio::time::timeout(
        HANG_BUDGET,
        poll_hold_once(&client, tv.key_schema(), settings.call_timeout),
    )
    .await
    .expect("a stalled read must end within the hang budget");
    release_all(stalls).await;

    let err = result.expect_err("a stalled read is an error, never a verdict");
    assert!(
        matches!(err, HoldPollError::Timeout(d) if d == Duration::from_millis(250)),
        "the per-call bound ends the read, got {err:?}"
    );
    assert_eq!(err.class(), "call_timeout");
}

// ═════════════════════════════════════════════════════════════════════════════
// Shutdown
// ═════════════════════════════════════════════════════════════════════════════

/// The poll ends when its token is cancelled while its read is stuck on a
/// connection with no command timeout and a per-call bound past the test's
/// budget: only the cancellation can end it. The receivers then see the
/// channel closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_stops_on_shutdown_with_its_read_stuck() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let client = own_client(0).await;
    let initial = startup(&client, keys).await;
    let stalls = stall_own_client(&tv, &client, "shutdown").await;
    let base = bytes_received(&stalls).await;
    let stuck = HoldPollSettings {
        interval: POLL_INTERVAL,
        call_timeout: Duration::from_secs(3_600),
    };
    let mut poll = Poll::spawn(&client, keys, stuck, initial);
    wait_for_queued_reads(&stalls, base, 1, "shutdown").await;

    poll.shutdown.cancel();
    tokio::time::timeout(WAIT_BUDGET, &mut poll.handle)
        .await
        .expect("the poll must end on cancellation with its read stuck")
        .expect("the poll task must not panic");
    assert!(
        poll.rx.changed().await.is_err(),
        "the receivers see the channel closed"
    );

    release_all(stalls).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The counter's slot, on a cluster
// ═════════════════════════════════════════════════════════════════════════════

/// On a multi-shard cluster, the store's own slot computation puts the marker
/// in the object-id counter's slot, and the poll's script runs on the primary
/// serving that slot: its `EVALSHA` names the marker and its `HMGET` reads it
/// there, as the `MONITOR` record of that primary shows.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when
/// it is unset; prints `CLUSTER-GATED RAN:` when it ran, which the `ci`
/// profile's success-output override in `.config/nextest.toml` keeps in the
/// job log and the JUnit XML.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_poll_script_runs_on_the_object_id_counters_primary_in_a_cluster() {
    let Some(url) = live_edit::storage::test_support::test_cluster_url() else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run this"
        );
        return;
    };
    let tv = TestValkey::setup_with_url(url, 4).await;
    let conn = tv.client().pool.next().clone();
    assert!(
        conn.is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment"
    );
    let keys = tv.key_schema();
    let marker = keys.dr_hold_marker();
    let counter = keys.counter_object_id();

    let marker_slot: u16 = conn
        .cluster_keyslot(marker.as_str())
        .await
        .expect("CLUSTER KEYSLOT of the marker");
    let counter_slot: u16 = conn
        .cluster_keyslot(counter.as_str())
        .await
        .expect("CLUSTER KEYSLOT of the counter");
    assert_eq!(
        marker_slot, counter_slot,
        "the store must hash {marker} into {counter}'s slot"
    );
    let state = conn
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view");
    let shards = state.unique_primary_nodes().len();
    let owner = state
        .get_server(counter_slot)
        .unwrap_or_else(|| panic!("no primary serves slot {counter_slot}"))
        .clone();
    let owner = format!("{}:{}", owner.host, owner.port);

    let seed = HoldMarkerSeed::default();
    tv.write_hold_marker(&seed).await;
    let mut tap = MonitorTap::start(&owner).await;
    let observation = poll_hold_once(
        tv.client(),
        keys,
        answering(FIXTURE_COMMAND_TIMEOUT_MS).call_timeout,
    )
    .await
    .expect("the read must answer on the cluster");
    assert_eq!(observation.verdict.incident(), Some(seed.incident.as_str()));

    // The fence carries the counter key as its hash tag, so it lands on the
    // same primary as the read.
    let fence = monitor_fence(&format!("{}{{{counter}}}:", tv.prefix()));
    let _: i64 = tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("the fence EXISTS must succeed");
    let seen = tap.drain_until(&fence).await;
    let start = seen
        .iter()
        .rposition(|c| c.name() == "EVALSHA" && c.args.get(3) == Some(&marker))
        .unwrap_or_else(|| panic!("{owner}: no EVALSHA naming {marker} in {seen:?}"));
    let script: Vec<_> = seen[start + 1..]
        .iter()
        .take_while(|c| c.issued_by_script())
        .collect();
    assert!(
        script
            .iter()
            .any(|c| c.name() == "HMGET" && c.args.get(1) == Some(&marker)),
        "{owner}: the script must read the marker on the counter's primary; it ran {script:?}"
    );

    println!(
        "CLUSTER-GATED RAN: dr::hold_poll_test::hold_poll_script_runs_on_the_object_id_counters_primary_in_a_cluster shards={shards}"
    );
}
