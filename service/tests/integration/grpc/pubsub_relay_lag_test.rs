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

// The replica-wide pub/sub relays: `presence` (`spawn_presence_subscriber`,
// into the local presence router) and `session_deleted`
// (`spawn_session_delete_subscriber`, cancelling this replica's session
// tokens). Both read the subscriber client's one message channel through
// `PubsubRelayReceiver`. Covered here:
//
// - `live_edit_pubsub_relay_lagged_total{relay, service}`: the messages a
//   relay skipped because it fell more than the channel's capacity behind,
//   each relay counting its own; and the production client's larger capacity,
//   on the pub/sub subscriber client only;
// - the drain, through each real relay: a relay takes every message already
//   queued each time it runs, not tokio's cooperative budget of 128 per poll
//   (`PubsubRelayReceiver`'s own batch semantics — lag sums, the yield bound,
//   a closed channel — are unit-tested in `src/presence/relay.rs`);
// - eager receivers: a relay sees every message published once its spawn
//   function returns, before its task first runs;
// - a lost deletion notice: the session-deleted relay requests a deleted-only
//   `session_tokens` pass (`SessionRevalidation`) when it lags and after a
//   pub/sub reconnect, and that pass ends the streams of a session deleted
//   meanwhile. The pass's own scope is tested in `session_tokens_sweep_test.rs`;
// - the re-SUBSCRIBE after a reconnect, held in flight through the relay's
//   test seam (`ResubscribeHold`): the relay keeps applying notices meanwhile,
//   requests the pass only once the SUBSCRIBE completes, owes exactly one more
//   SUBSCRIBE for any number of reconnects during it, still stops on its
//   token, and requests the pass if its channel closes during it.
//
// ── Holding a relay ──────────────────────────────────────────────────────────
//
// A relay under test reads a subscriber client the test builds with a message
// channel of a chosen capacity, and is spawned on a second, current-thread
// runtime (`HeldRuntime`) that runs only while the test drives it. So the
// relay is held still while the test publishes:
//
//   1. the relay is driven until it has forwarded an attach probe and then a
//      fence message: it is running and has read everything published so far;
//   2. with the relay held, the test publishes, and a second receiver on the
//      same client is read until the last message, so all of them are in the
//      channel;
//   3. the relay is driven again.
//
// ── Forcing an exact lag ─────────────────────────────────────────────────────
//
// The lag tests use a channel of `CAPACITY` (4) messages and publish `BURST`
// in step 2. tokio's broadcast channel keeps the newest `CAPACITY` messages
// (the capacity is a power of two, which tokio would otherwise round up to),
// so in step 3 the relay's next receive reports exactly
// `Lagged(BURST - CAPACITY)` and it then forwards the last `CAPACITY`.
//
// ── Counting a relay's turns ─────────────────────────────────────────────────
//
// The drain tests queue `DRAIN_BURST` messages in a channel of
// `DRAIN_CAPACITY` and then run the held runtime with a competitor task that
// yields in a loop, counting its turns, until an observer has every message.
// A relay that takes the whole queue in one turn finishes within
// `MAX_COMPETITOR_TURNS`; one that awaits `recv()` per message is cut off by
// tokio's cooperative budget after 128 messages per poll and deferred behind
// the competitor each time, so it needs at least 7 more turns
// (`DRAIN_BURST` / 128, rounded up, is 8 polls). They count scheduler turns,
// never wall-clock time.
//
// ── Waits ────────────────────────────────────────────────────────────────────
//
// Attaching a relay drives it for `ATTACH_WINDOW` (50 ms) per try, publishing
// a fresh probe each try, for at most `ATTACH_TRIES` tries. A relay takes its
// receiver before it is spawned, so the first try normally succeeds; the
// retry keeps the tests other than the eager-receiver ones independent of
// that. After a reconnect the same retry waits out fred's own
// resubscription, before which a probe reaches no subscriber. Every other
// wait is bounded by `SETTLE` (10 s), and returns in milliseconds on the
// healthy path. Each scenario runs under `SCENARIO_BUDGET`.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use fred::clients::SubscriberClient;
use fred::interfaces::{ClientLike, EventInterface, PubsubInterface};
use fred::types::config::{Config, PerformanceConfig};
use fred::types::{Builder, CustomCommand, Message, RespVersion};
use futures::{FutureExt, StreamExt};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tonic::Request;

use live_edit::grpc::proto::{DeleteSessionRequest, SubscribeReadLayerRequest};
use live_edit::grpc::session::{
    handle_delete_session, spawn_session_delete_subscriber,
    spawn_session_delete_subscriber_with_resubscribe_hold, spawn_session_tokens_sweeper,
    subscribe_session_deleted_channel, ResubscribeHold, SessionCancel, SessionCancelCause,
    SessionRevalidation,
};
use live_edit::grpc::session_cache::SessionBasePathCache;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::{
    SharedMetrics, PUBSUB_RELAYS, PUBSUB_RELAY_PRESENCE, PUBSUB_RELAY_SESSION_DELETED,
};
use live_edit::presence::{
    spawn_presence_subscriber, LocalPresenceRouter, PresenceEvent, PresenceReceiver,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{connect_test_client, TestValkey};
use live_edit::storage::{consume_layer_stream, StreamCursor};

use super::helpers::*;

/// The lag tests' subscriber-client message-channel capacity. A power of two,
/// so tokio's broadcast channel holds exactly this many.
const CAPACITY: usize = 4;

/// Messages published while the relay is held: the relay skips all but the
/// newest [`CAPACITY`].
const BURST: usize = 3 * CAPACITY;

/// The drain tests' subscriber-client message-channel capacity, which is also
/// the relay's yield bound. The relay counts toward that bound every message
/// it has handled since it last yielded, which in these tests is all of them:
/// up to [`ATTACH_TRIES`] attach messages and the fence before the burst, then
/// [`DRAIN_BURST`]. This capacity is above that total, so the burst cannot end
/// a drain with a yield partway through, and no message lags.
const DRAIN_CAPACITY: usize = 2048;

/// Messages queued for one drain: more than seven times tokio's cooperative
/// budget of 128 per poll, and with the attach messages and fence, under
/// [`DRAIN_CAPACITY`], so nothing lags.
const DRAIN_BURST: usize = 1000;

/// The most competitor turns a relay that drains its queue may take to
/// deliver [`DRAIN_BURST`]: one turn for the relay, one of slack for where the
/// competitor lands in the run queue. A relay capped at 128 messages per poll
/// takes at least 7.
const MAX_COMPETITOR_TURNS: usize = 2;

/// Deletion notices published after the lost one, so it is among the skipped:
/// twice [`CAPACITY`], so the relay keeps only these.
const FILLERS: usize = 2 * CAPACITY;

/// The sweeper's periodic interval in the tests that need only its requested
/// pass: no periodic pass comes within a test.
const NO_PERIODIC_PASS: Duration = Duration::from_secs(3600);

/// The `service` label of the relays' metrics.
const SERVICE: &str = "relay-lag";

/// Bound on any single wait.
const SETTLE: Duration = Duration::from_secs(10);

/// How long one attach try drives the relay.
const ATTACH_WINDOW: Duration = Duration::from_millis(50);

/// Attach tries before giving up: [`SETTLE`] in [`ATTACH_WINDOW`]s.
const ATTACH_TRIES: u32 = 200;

/// Interval between reads of a metric a test waits on.
const METRIC_POLL: Duration = Duration::from_millis(5);

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// A current-thread runtime that runs its tasks only while the test drives it
/// ([`Self::drive`]), so a relay spawned on it is held still in between.
struct HeldRuntime(Option<Arc<tokio::runtime::Runtime>>);

impl HeldRuntime {
    fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("building the held runtime must succeed");
        Self(Some(Arc::new(runtime)))
    }

    fn runtime(&self) -> &Arc<tokio::runtime::Runtime> {
        self.0
            .as_ref()
            .expect("the held runtime is live until dropped")
    }

    /// Call `spawn` with this runtime entered, so the task it spawns with
    /// `tokio::spawn` runs here.
    fn spawn_on<R>(&self, spawn: impl FnOnce() -> R) -> R {
        let _entered = self.runtime().enter();
        spawn()
    }

    /// Run this runtime, and every task on it, until `fut` completes.
    async fn drive<F>(&self, fut: F) -> F::Output
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let runtime = Arc::clone(self.runtime());
        tokio::task::spawn_blocking(move || runtime.block_on(fut))
            .await
            .expect("driving the held runtime must not panic")
    }
}

impl Drop for HeldRuntime {
    /// A runtime cannot be dropped in an async context; shut it down in the
    /// background instead.
    fn drop(&mut self) {
        if let Some(runtime) = self.0.take() {
            match Arc::try_unwrap(runtime) {
                Ok(runtime) => runtime.shutdown_background(),
                // A drive still holds it (a failed test): leak it rather than
                // drop it here.
                Err(runtime) => std::mem::forget(runtime),
            }
        }
    }
}

/// A subscriber client on the test Valkey whose message channel holds
/// `capacity` messages.
async fn subscriber_with_capacity(capacity: usize) -> SubscriberClient {
    let url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let config = Config::from_url(&url).expect("TEST_VALKEY_URL must parse as a fred config");
    let mut builder = Builder::from_config(config);
    builder
        .with_config(|c| c.version = RespVersion::RESP2)
        .set_performance_config(PerformanceConfig {
            broadcast_channel_capacity: capacity,
            ..Default::default()
        });
    let subscriber = builder
        .build_subscriber_client()
        .expect("building the subscriber client must succeed");
    subscriber
        .init()
        .await
        .expect("the subscriber client must connect");
    subscriber
}

/// A subscriber client whose message channel holds [`CAPACITY`] messages.
async fn small_subscriber() -> SubscriberClient {
    subscriber_with_capacity(CAPACITY).await
}

fn new_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(SERVICE))
}

/// `(presence, session_deleted)` on `metrics`.
fn lagged(metrics: &SharedMetrics) -> (u64, u64) {
    (
        pubsub_relay_lagged(metrics, SERVICE, PUBSUB_RELAY_PRESENCE),
        pubsub_relay_lagged(metrics, SERVICE, PUBSUB_RELAY_SESSION_DELETED),
    )
}

/// Read `rx`, a receiver on the subscriber client itself, until it holds the
/// message whose payload `is_last` accepts: everything published before that
/// message is in the client's channel. Skips the lags this receiver takes.
async fn wait_in_channel(
    rx: &mut broadcast::Receiver<Message>,
    is_last: impl Fn(&str) -> bool,
    what: &str,
) {
    tokio::time::timeout(SETTLE, async {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    let payload: String = msg.value.convert().expect("a string payload");
                    if is_last(&payload) {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => {
                    panic!("{what}: the subscriber client's channel closed")
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: the burst did not reach the client within {SETTLE:?}"));
}

/// Run `held`, beside a competitor task that yields in a loop and counts its
/// turns, until `done` completes. Returns `done`'s output and the competitor's
/// turns by then, or `None` if `done` did not complete within [`SETTLE`].
///
/// `done` runs unconstrained by tokio's cooperative budget, so reading the
/// relay's output never costs a turn of its own: only the relay's turns move
/// the count.
async fn competitor_turns_until<F>(held: &HeldRuntime, done: F) -> Option<(F::Output, usize)>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    held.drive(async move {
        let turns = Arc::new(AtomicUsize::new(0));
        let competitor = tokio::spawn({
            let turns = Arc::clone(&turns);
            async move {
                loop {
                    turns.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                }
            }
        });
        let output = tokio::time::timeout(SETTLE, tokio::task::coop::unconstrained(done)).await;
        let turns = turns.load(Ordering::Relaxed);
        competitor.abort();
        output.ok().map(|output| (output, turns))
    })
    .await
}

/// Wrap `message` in a request carrying a bearer token.
fn with_bearer<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═════════════════════════════════════════════════════════════════════════════
// Seeding
// ═════════════════════════════════════════════════════════════════════════════

/// Both relays' cells are served at zero before any lag, so the series exists
/// from boot.
#[test]
fn pubsub_relay_lagged_cells_are_seeded_at_zero() {
    let metrics = new_metrics();
    assert_eq!(
        PUBSUB_RELAYS,
        &[PUBSUB_RELAY_PRESENCE, PUBSUB_RELAY_SESSION_DELETED],
        "the relay vocabulary"
    );
    assert_eq!(lagged(&metrics), (0, 0));
}

// ═════════════════════════════════════════════════════════════════════════════
// The production client's message-channel capacities
// ═════════════════════════════════════════════════════════════════════════════

/// The smallest message-channel capacity the pub/sub subscriber client may
/// have: the 4096 `storage::client::PUBSUB_BROADCAST_CHANNEL_CAPACITY` sets.
/// At fred's default of 32 the presence relay loses events in a 50-join burst
/// (the storm test in `presence_replay_dedup_test` catches that only in about
/// one run in six, so this test is the deterministic gate).
const MIN_PUBSUB_CHANNEL_CAPACITY: usize = 4096;

/// The client `ValkeyClient::connect` builds (which `TestValkey` uses) gives
/// its pub/sub subscriber client a message channel of at least
/// [`MIN_PUBSUB_CHANNEL_CAPACITY`], and only that client: the other clients,
/// which subscribe to nothing, keep fred's default, so no pooled connection
/// allocates a large channel it never uses. Checked on the data pool (whose
/// builder the transaction pool is cloned from), the maintenance pool and
/// every stream-multiplexer reader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_the_pubsub_subscriber_client_has_the_larger_message_channel() {
    let tv = TestValkey::setup().await;
    let fred_default = PerformanceConfig::default().broadcast_channel_capacity;
    assert!(
        fred_default < MIN_PUBSUB_CHANNEL_CAPACITY,
        "precondition: fred's default ({fred_default}) is below the pub/sub client's"
    );

    let subscriber_capacity = tv
        .client()
        .subscriber
        .perf_config()
        .broadcast_channel_capacity;
    assert!(
        subscriber_capacity >= MIN_PUBSUB_CHANNEL_CAPACITY,
        "the pub/sub subscriber client's message channel must hold at least \
         {MIN_PUBSUB_CHANNEL_CAPACITY} messages; it holds {subscriber_capacity}"
    );
    assert_eq!(
        tv.client()
            .pool
            .next()
            .perf_config()
            .broadcast_channel_capacity,
        fred_default,
        "the data pool's clients keep fred's default message channel"
    );
    assert_eq!(
        tv.client()
            .maintenance_pool
            .next()
            .perf_config()
            .broadcast_channel_capacity,
        fred_default,
        "the maintenance pool's clients keep fred's default message channel"
    );
    // Readers connect lazily, on their first subscription, so subscribe until
    // every reader holds a connection. Reading a client's performance config
    // issues no command, so it leaves a reader parked in `XREAD BLOCK`
    // undisturbed.
    let mux = &tv.client().stream_mux;
    let reader_count = mux.snapshot().readers.len();
    assert!(
        reader_count > 0,
        "precondition: the stream multiplexer has readers"
    );
    let deadline = tokio::time::Instant::now() + SETTLE;
    let mut subscriptions = Vec::new();
    for object_id in 1..=(4 * reader_count as u64) {
        subscriptions.push(
            consume_layer_stream(
                tv.client(),
                tv.key_schema(),
                object_id,
                StreamCursor::From("0-0".into()),
                deadline,
            )
            .await
            .expect("the subscription must start"),
        );
    }
    while mux.connected_readers() < reader_count {
        assert!(
            tokio::time::Instant::now() < deadline,
            "precondition: {} of {reader_count} readers connected within {SETTLE:?}",
            mux.connected_readers()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let readers = mux.reader_clients_for_test();
    assert_eq!(
        readers.len(),
        reader_count,
        "precondition: every reader holds a connection"
    );
    for (i, reader) in readers.iter().enumerate() {
        assert_eq!(
            reader.perf_config().broadcast_channel_capacity,
            fred_default,
            "stream-multiplexer reader {i} keeps fred's default message channel"
        );
    }
    drop(subscriptions);
}

// ═════════════════════════════════════════════════════════════════════════════
// The presence relay
// ═════════════════════════════════════════════════════════════════════════════

/// Drive the presence relay until `observer`, a receiver on the relay's
/// router, gets the `PeerJoined` of `want`, or `within` passes. Returns the
/// receiver, the `PeerJoined` ids it got, and whether `want` was among them.
async fn drive_presence_until(
    held: &HeldRuntime,
    mut observer: PresenceReceiver,
    want: String,
    within: Duration,
) -> (PresenceReceiver, Vec<String>, bool) {
    held.drive(async move {
        let mut forwarded = Vec::new();
        let found = tokio::time::timeout(within, async {
            loop {
                match observer.recv().await.as_deref() {
                    Ok(PresenceEvent::PeerJoined(user)) => {
                        let done = user.connection_id == want;
                        forwarded.push(user.connection_id.clone());
                        if done {
                            return;
                        }
                    }
                    Ok(_) => {}
                    Err(e) => panic!("the router's receiver failed: {e:?}"),
                }
            }
        })
        .await
        .is_ok();
        (observer, forwarded, found)
    })
    .await
}

/// Steps 1 of the module header for the presence relay on `session_id`:
/// drive it until it has forwarded an attach probe and then a fence to
/// `observer`. Returns the receiver.
async fn attach_presence_relay(
    tv: &TestValkey,
    held: &HeldRuntime,
    mut observer: PresenceReceiver,
    session_id: &str,
) -> PresenceReceiver {
    let mut attached = false;
    for attempt in 0..ATTACH_TRIES {
        let probe = format!("attach-{attempt}");
        publish_probe(tv, session_id, &probe).await;
        let (back, _, found) = drive_presence_until(held, observer, probe, ATTACH_WINDOW).await;
        observer = back;
        if found {
            attached = true;
            break;
        }
    }
    assert!(
        attached,
        "the presence relay never forwarded an attach probe"
    );
    publish_probe(tv, session_id, "drained").await;
    let (observer, _, drained) =
        drive_presence_until(held, observer, "drained".to_owned(), SETTLE).await;
    assert!(drained, "the relay must forward the fence");
    observer
}

/// The presence relay, held while [`BURST`] presence events are published on a
/// subscriber client that keeps [`CAPACITY`], reports one `Lagged` of
/// `BURST - CAPACITY` and adds exactly that to its `presence` cell; it then
/// forwards the newest `CAPACITY` events to the router. The `session_deleted`
/// cell is untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn presence_relay_lag_adds_the_skipped_message_count_to_its_cell() {
    const SID: &str = "relay-lag-presence";
    scenario(async {
        let tv = TestValkey::setup().await;
        let metrics = new_metrics();
        let subscriber = small_subscriber().await;
        subscriber
            .subscribe(tv.key_schema().presence_events_channel(SID))
            .await
            .expect("SUBSCRIBE must succeed");
        let router = Arc::new(LocalPresenceRouter::new());
        let (observer, _) = router.subscribe(SID);

        let held = HeldRuntime::new();
        let cancel = CancellationToken::new();
        let relay = held.spawn_on(|| {
            spawn_presence_subscriber(
                subscriber.clone(),
                Arc::clone(&router),
                tv.key_schema().clone(),
                Arc::clone(&metrics),
                cancel.clone(),
            )
        });

        // 1. The relay is attached and has read everything published so far.
        let observer = attach_presence_relay(&tv, &held, observer, SID).await;
        assert_eq!(lagged(&metrics), (0, 0), "precondition: no lag yet");

        // 2. With the relay held, the burst fills the client's channel.
        let burst: Vec<String> = (0..BURST).map(|i| format!("burst-{i:02}")).collect();
        let mut tap = subscriber.message_rx();
        for cid in &burst {
            publish_probe(&tv, SID, cid).await;
        }
        let last = burst.last().expect("a burst").clone();
        wait_in_channel(
            &mut tap,
            |payload| {
                matches!(PresenceEvent::parse(payload),
                    Ok(PresenceEvent::PeerJoined(user)) if user.connection_id == last)
            },
            "the presence burst",
        )
        .await;

        // 3. The relay runs again.
        let (_observer, forwarded, found) =
            drive_presence_until(&held, observer, last.clone(), SETTLE).await;
        assert!(found, "the relay must forward the burst's last event");
        assert_eq!(
            forwarded,
            burst[BURST - CAPACITY..].to_vec(),
            "the relay skips all but the newest {CAPACITY} events"
        );
        assert_eq!(
            lagged(&metrics),
            ((BURST - CAPACITY) as u64, 0),
            "the presence cell grows by the {} events the relay skipped; the session_deleted \
             cell is untouched",
            BURST - CAPACITY
        );

        cancel.cancel();
        held.drive(relay)
            .await
            .expect("the relay must stop on its token");
        let _ = subscriber.quit().await;
    })
    .await;
}

/// The presence relay, held while [`DRAIN_BURST`] events queue in its client's
/// channel, forwards all of them, in order, in one turn: a competitor task on
/// the same runtime gets at most [`MAX_COMPETITOR_TURNS`] turns before the
/// router's receiver has the last. A relay limited to tokio's cooperative
/// budget of 128 messages per poll would need eight polls, with the
/// competitor's turn between each. Nothing lags.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn presence_relay_forwards_every_queued_event_in_one_turn() {
    const SID: &str = "relay-drain-presence";
    scenario(async {
        let tv = TestValkey::setup().await;
        let metrics = new_metrics();
        let subscriber = subscriber_with_capacity(DRAIN_CAPACITY).await;
        subscriber
            .subscribe(tv.key_schema().presence_events_channel(SID))
            .await
            .expect("SUBSCRIBE must succeed");
        let router = Arc::new(LocalPresenceRouter::new());
        let (observer, _) = router.subscribe(SID);

        let held = HeldRuntime::new();
        let cancel = CancellationToken::new();
        let relay = held.spawn_on(|| {
            spawn_presence_subscriber(
                subscriber.clone(),
                Arc::clone(&router),
                tv.key_schema().clone(),
                Arc::clone(&metrics),
                cancel.clone(),
            )
        });
        let mut observer = attach_presence_relay(&tv, &held, observer, SID).await;

        // With the relay held, the burst queues in the client's channel.
        let burst: Vec<String> = (0..DRAIN_BURST).map(|i| format!("drain-{i:04}")).collect();
        let mut tap = subscriber.message_rx();
        for cid in &burst {
            publish_probe(&tv, SID, cid).await;
        }
        let last = burst.last().expect("a burst").clone();
        wait_in_channel(
            &mut tap,
            |payload| {
                matches!(PresenceEvent::parse(payload),
                    Ok(PresenceEvent::PeerJoined(user)) if user.connection_id == last)
            },
            "the presence burst",
        )
        .await;
        drop(tap);

        let (forwarded, turns) = competitor_turns_until(&held, async move {
            let mut forwarded = Vec::with_capacity(DRAIN_BURST);
            while forwarded.len() < DRAIN_BURST {
                match observer.recv().await.as_deref() {
                    Ok(PresenceEvent::PeerJoined(user)) => {
                        forwarded.push(user.connection_id.clone())
                    }
                    Ok(other) => panic!("only the burst's joins were published: {other:?}"),
                    Err(e) => panic!("the router's receiver failed: {e:?}"),
                }
            }
            forwarded
        })
        .await
        .unwrap_or_else(|| {
            panic!("the router's receiver did not get the {DRAIN_BURST} events within {SETTLE:?}")
        });

        assert_eq!(
            forwarded, burst,
            "the relay forwards every queued event, in order"
        );
        assert!(
            turns <= MAX_COMPETITOR_TURNS,
            "the relay must forward all {DRAIN_BURST} queued events in one turn, but a competitor \
             got {turns} turns meanwhile (at most {MAX_COMPETITOR_TURNS} allowed)"
        );
        assert_eq!(lagged(&metrics), (0, 0), "nothing lags");

        cancel.cancel();
        held.drive(relay)
            .await
            .expect("the relay must stop on its token");
        let _ = subscriber.quit().await;
    })
    .await;
}

/// The presence relay takes its receiver before its task first runs: an event
/// published after `spawn_presence_subscriber` returns, and already in the
/// client's channel before the relay has ever run, is forwarded.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn presence_relay_forwards_an_event_published_before_its_task_first_runs() {
    const SID: &str = "relay-eager-presence";
    scenario(async {
        let tv = TestValkey::setup().await;
        let subscriber = small_subscriber().await;
        subscriber
            .subscribe(tv.key_schema().presence_events_channel(SID))
            .await
            .expect("SUBSCRIBE must succeed");
        let router = Arc::new(LocalPresenceRouter::new());
        let (observer, _) = router.subscribe(SID);

        let held = HeldRuntime::new();
        let cancel = CancellationToken::new();
        let relay = held.spawn_on(|| {
            spawn_presence_subscriber(
                subscriber.clone(),
                Arc::clone(&router),
                tv.key_schema().clone(),
                new_metrics(),
                cancel.clone(),
            )
        });

        // The relay has not run; the event reaches the client's channel.
        let mut tap = subscriber.message_rx();
        publish_probe(&tv, SID, "early").await;
        wait_in_channel(
            &mut tap,
            |payload| {
                matches!(PresenceEvent::parse(payload),
                    Ok(PresenceEvent::PeerJoined(user)) if user.connection_id == "early")
            },
            "the early event",
        )
        .await;

        let (_observer, forwarded, found) =
            drive_presence_until(&held, observer, "early".to_owned(), SETTLE).await;
        assert!(
            found,
            "the relay must forward an event published after its spawn returned, before its task \
             first ran; it forwarded {forwarded:?}"
        );

        cancel.cancel();
        held.drive(relay)
            .await
            .expect("the relay must stop on its token");
        let _ = subscriber.quit().await;
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The session-deleted relay
// ═════════════════════════════════════════════════════════════════════════════

/// Register a session token for `session_id` in `tokens`, as a session handled
/// here would have; the returned token is cancelled when the relay applies a
/// deletion of the session.
fn track(tokens: &DashMap<String, SessionCancel>, session_id: &str) -> CancellationToken {
    let token = CancellationToken::new();
    tokens.insert(session_id.to_owned(), token.clone().into());
    token
}

/// Publish a deletion notice for `session_id` on the session-deleted channel,
/// as `DeleteSession` on another replica would.
async fn publish_deletion(tv: &TestValkey, session_id: &str) {
    let _: i64 = store_reader(tv)
        .publish(tv.key_schema().session_deleted_channel(), session_id)
        .await
        .expect("PUBLISH must succeed");
}

/// Drive the session-deleted relay until `token` is cancelled, or `within`
/// passes; whether it was.
async fn drive_deleted_until(
    held: &HeldRuntime,
    token: &CancellationToken,
    within: Duration,
) -> bool {
    let token = token.clone();
    held.drive(async move {
        tokio::time::timeout(within, token.cancelled_owned())
            .await
            .is_ok()
    })
    .await
}

/// Publish a deletion of a fresh session `<label>-<try>` and drive the
/// session-deleted relay over `tokens` for [`ATTACH_WINDOW`], until it applies
/// one, for at most [`ATTACH_TRIES`] tries; whether it did.
async fn apply_a_fresh_notice(
    tv: &TestValkey,
    held: &HeldRuntime,
    tokens: &DashMap<String, SessionCancel>,
    label: &str,
) -> bool {
    for attempt in 0..ATTACH_TRIES {
        let session_id = format!("{label}-{attempt}");
        let token = track(tokens, &session_id);
        publish_deletion(tv, &session_id).await;
        if drive_deleted_until(held, &token, ATTACH_WINDOW).await {
            return true;
        }
    }
    false
}

/// Step 1 of the module header for the session-deleted relay over `tokens`:
/// drive it until it has applied an attach deletion and then a fence.
async fn attach_deleted_relay(
    tv: &TestValkey,
    held: &HeldRuntime,
    tokens: &DashMap<String, SessionCancel>,
) {
    assert!(
        apply_a_fresh_notice(tv, held, tokens, "attach").await,
        "the session-deleted relay never applied an attach deletion"
    );
    let drained = track(tokens, "drained");
    publish_deletion(tv, "drained").await;
    assert!(
        drive_deleted_until(held, &drained, SETTLE).await,
        "the relay must apply the fence"
    );
}

/// Drive the session-deleted relay until `metrics` reads `want`, or
/// [`SETTLE`] passes; whether it did. The relay adds a drain batch's lag and
/// requests a revalidation in the same step, so once the cell moves the
/// request has been made.
async fn drive_until_lagged(held: &HeldRuntime, metrics: &SharedMetrics, want: (u64, u64)) -> bool {
    let metrics = Arc::clone(metrics);
    held.drive(async move {
        tokio::time::timeout(SETTLE, async {
            while lagged(&metrics) != want {
                tokio::time::sleep(METRIC_POLL).await;
            }
        })
        .await
        .is_ok()
    })
    .await
}

/// Whether `revalidation` has a pass requested that no sweeper has taken.
/// Polls `SessionRevalidation::requested` once, which is ready at once when
/// one is pending. For a handle no sweeper waits on only: a poll that finds
/// none pending registers, briefly, as a waiter.
fn revalidation_requested(revalidation: &SessionRevalidation) -> bool {
    revalidation.requested().now_or_never().is_some()
}

/// The session-deleted relay, held while [`BURST`] deletions are published on
/// a subscriber client that keeps [`CAPACITY`], reports one `Lagged` of
/// `BURST - CAPACITY` and adds exactly that to its `session_deleted` cell; it
/// then applies the newest `CAPACITY` deletions, and the skipped sessions'
/// tokens are left as they were, for the revalidation the lag requests. The
/// `presence` cell is untouched.
///
/// The skipped sessions never existed, so a requested pass, which cancels only
/// marked sessions, would leave them registered too; this test runs no
/// sweeper, and checks the request itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_deleted_relay_lag_adds_the_skipped_message_count_to_its_cell() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let metrics = new_metrics();
        let subscriber = small_subscriber().await;
        subscribe_session_deleted_channel(&subscriber, tv.key_schema())
            .await
            .expect("SUBSCRIBE must succeed");
        let tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
        let revalidation = SessionRevalidation::new();

        let held = HeldRuntime::new();
        let cancel = CancellationToken::new();
        let relay = held.spawn_on(|| {
            spawn_session_delete_subscriber(
                subscriber.clone(),
                Arc::clone(&tokens),
                SessionBasePathCache::new(),
                tv.key_schema().clone(),
                Arc::clone(&metrics),
                revalidation.clone(),
                cancel.clone(),
            )
        });

        // 1. The relay is attached and has read everything published so far.
        attach_deleted_relay(&tv, &held, &tokens).await;
        assert_eq!(lagged(&metrics), (0, 0), "precondition: no lag yet");
        assert!(
            !revalidation_requested(&revalidation),
            "precondition: a relay that has not lagged requests no revalidation"
        );

        // 2. With the relay held, the burst fills the client's channel.
        let burst: Vec<String> = (0..BURST).map(|i| format!("burst-{i:02}")).collect();
        let burst_tokens: Vec<CancellationToken> =
            burst.iter().map(|sid| track(&tokens, sid)).collect();
        let mut tap = subscriber.message_rx();
        for session_id in &burst {
            publish_deletion(&tv, session_id).await;
        }
        let last = burst.last().expect("a burst").clone();
        wait_in_channel(&mut tap, |payload| payload == last, "the deletion burst").await;

        // 3. The relay runs again.
        assert!(
            drive_deleted_until(&held, &burst_tokens[BURST - 1], SETTLE).await,
            "the relay must apply the burst's last deletion"
        );
        let applied: Vec<bool> = burst_tokens.iter().map(|t| t.is_cancelled()).collect();
        let expected: Vec<bool> = (0..BURST).map(|i| i >= BURST - CAPACITY).collect();
        assert_eq!(
            applied, expected,
            "the relay applies only the newest {CAPACITY} deletions"
        );
        for (session_id, was_applied) in burst.iter().zip(&expected) {
            assert_eq!(
                tokens.contains_key(session_id),
                !was_applied,
                "{session_id}: a skipped deletion leaves its token registered"
            );
        }
        assert_eq!(
            lagged(&metrics),
            (0, (BURST - CAPACITY) as u64),
            "the session_deleted cell grows by the {} deletions the relay skipped; the presence \
             cell is untouched",
            BURST - CAPACITY
        );
        assert!(
            revalidation_requested(&revalidation),
            "a lag on the session-deleted relay must request a revalidation of the replica's \
             sessions"
        );

        cancel.cancel();
        held.drive(relay)
            .await
            .expect("the relay must stop on its token");
        let _ = subscriber.quit().await;
    })
    .await;
}

/// The session-deleted relay, held while [`DRAIN_BURST`] deletions queue in
/// its client's channel, applies all of them in one turn: a competitor task on
/// the same runtime gets at most [`MAX_COMPETITOR_TURNS`] turns before the last
/// session's token is cancelled. A relay limited to tokio's cooperative budget
/// of 128 messages per poll would need eight polls, with the competitor's turn
/// between each. Nothing lags.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_deleted_relay_applies_every_queued_deletion_in_one_turn() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let metrics = new_metrics();
        let subscriber = subscriber_with_capacity(DRAIN_CAPACITY).await;
        subscribe_session_deleted_channel(&subscriber, tv.key_schema())
            .await
            .expect("SUBSCRIBE must succeed");
        let tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());

        let held = HeldRuntime::new();
        let cancel = CancellationToken::new();
        let relay = held.spawn_on(|| {
            spawn_session_delete_subscriber(
                subscriber.clone(),
                Arc::clone(&tokens),
                SessionBasePathCache::new(),
                tv.key_schema().clone(),
                Arc::clone(&metrics),
                SessionRevalidation::new(),
                cancel.clone(),
            )
        });
        attach_deleted_relay(&tv, &held, &tokens).await;

        // With the relay held, the burst queues in the client's channel.
        let burst: Vec<String> = (0..DRAIN_BURST).map(|i| format!("drain-{i:04}")).collect();
        let burst_tokens: Vec<CancellationToken> =
            burst.iter().map(|sid| track(&tokens, sid)).collect();
        let mut tap = subscriber.message_rx();
        for session_id in &burst {
            publish_deletion(&tv, session_id).await;
        }
        let last = burst.last().expect("a burst").clone();
        wait_in_channel(&mut tap, |payload| payload == last, "the deletion burst").await;
        drop(tap);

        let watched = burst_tokens.clone();
        let ((), turns) = competitor_turns_until(&held, async move {
            for token in &watched {
                token.cancelled().await;
            }
        })
        .await
        .unwrap_or_else(|| {
            panic!("the relay did not apply the {DRAIN_BURST} deletions within {SETTLE:?}")
        });

        assert!(
            burst.iter().all(|sid| !tokens.contains_key(sid)),
            "every queued deletion removes its session's entry"
        );
        assert!(
            turns <= MAX_COMPETITOR_TURNS,
            "the relay must apply all {DRAIN_BURST} queued deletions in one turn, but a \
             competitor got {turns} turns meanwhile (at most {MAX_COMPETITOR_TURNS} allowed)"
        );
        assert_eq!(lagged(&metrics), (0, 0), "nothing lags");

        cancel.cancel();
        held.drive(relay)
            .await
            .expect("the relay must stop on its token");
        let _ = subscriber.quit().await;
    })
    .await;
}

/// The session-deleted relay takes its receiver before its task first runs: a
/// deletion published after `spawn_session_delete_subscriber` returns, and
/// already in the client's channel before the relay has ever run, is applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_deleted_relay_applies_a_deletion_published_before_its_task_first_runs() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let subscriber = small_subscriber().await;
        subscribe_session_deleted_channel(&subscriber, tv.key_schema())
            .await
            .expect("SUBSCRIBE must succeed");
        let tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
        let early = track(&tokens, "early");

        let held = HeldRuntime::new();
        let cancel = CancellationToken::new();
        let relay = held.spawn_on(|| {
            spawn_session_delete_subscriber(
                subscriber.clone(),
                Arc::clone(&tokens),
                SessionBasePathCache::new(),
                tv.key_schema().clone(),
                new_metrics(),
                SessionRevalidation::new(),
                cancel.clone(),
            )
        });

        // The relay has not run; the deletion reaches the client's channel.
        let mut tap = subscriber.message_rx();
        publish_deletion(&tv, "early").await;
        wait_in_channel(&mut tap, |payload| payload == "early", "the early deletion").await;

        assert!(
            drive_deleted_until(&held, &early, SETTLE).await,
            "the relay must apply a deletion published after its spawn returned, before its task \
             first ran"
        );
        assert!(!tokens.contains_key("early"));

        cancel.cancel();
        held.drive(relay)
            .await
            .expect("the relay must stop on its token");
        let _ = subscriber.quit().await;
    })
    .await;
}

// ── A lost deletion notice ───────────────────────────────────────────────────

/// The layer every lost-notice scenario's session holds.
const LAYER: &str = "lost-notice-layer";

/// What happens on replica B when its session-deleted relay lags past the
/// notice of a session another replica, A, deleted. B holds a
/// `SubscribeReadLayer` and a `JoinSession` on the session.
///
/// - The subscription does not depend on the notice: `DeleteSession` deletes
///   the session's layers first, and each deletion ends the layer's stream
///   with a sentinel, so it ends `NOT_FOUND` while B's relay is still held and
///   B's session token is still live.
/// - The join tails no layer stream, so the token is its only deletion signal.
///   The relay's lag requests a revalidation pass; B's sweeper, whose periodic
///   pass never comes within the test, runs it, finds the session missing and
///   marked, and cancels the token as a deletion. The join ends `OK`.
///
/// The notice is lost by arithmetic, not by timing: the relay is held while
/// the notice and [`FILLERS`] more reach a channel that keeps [`CAPACITY`], so
/// it skips exactly the notice and the oldest fillers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deleted_relay_lag_over_a_deletion_still_ends_the_join_on_the_lagging_replica() {
    scenario(async {
        let tv = TestValkey::setup().await;
        // Two replicas on one store: A serves DeleteSession, B holds the
        // streams. Each has its own session_tokens.
        let svc_a = build_service(&tv, Arc::new(AllowAllPermissions));
        let svc_b = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc_a, "lost-notice", LAYER).await;

        // B's relay, held, on a client whose channel keeps CAPACITY; B's
        // sweeper, sharing the relay's request handle.
        let metrics = new_metrics();
        let subscriber = small_subscriber().await;
        subscribe_session_deleted_channel(&subscriber, tv.key_schema())
            .await
            .expect("SUBSCRIBE must succeed");
        let revalidation = SessionRevalidation::new();
        let held = HeldRuntime::new();
        let relay_stop = CancellationToken::new();
        let relay = held.spawn_on(|| {
            spawn_session_delete_subscriber(
                subscriber.clone(),
                Arc::clone(&svc_b.session_tokens),
                svc_b.session_base_path_cache.clone(),
                tv.key_schema().clone(),
                Arc::clone(&metrics),
                revalidation.clone(),
                relay_stop.clone(),
            )
        });
        let sweeper_stop = CancellationToken::new();
        let sweeper = spawn_session_tokens_sweeper(
            Arc::clone(&svc_b.session_tokens),
            Arc::clone(&svc_b.session_store),
            svc_b.session_base_path_cache.clone(),
            NO_PERIODIC_PASS,
            revalidation.clone(),
            sweeper_stop.clone(),
        );
        attach_deleted_relay(&tv, &held, &svc_b.session_tokens).await;

        // B's streams on the session.
        let mut read = handle_subscribe_read_layer(
            &svc_b,
            with_bearer(SubscribeReadLayerRequest {
                session_id: session_id.clone(),
                layer_name: LAYER.to_owned(),
                object_id,
                sequence_num: 1,
                client_id: "reader-b".to_owned(),
            }),
        )
        .await
        .expect("SubscribeReadLayer on B must succeed")
        .into_inner();
        tokio::time::timeout(SETTLE, read.next())
            .await
            .expect("the subscription's ack must arrive")
            .expect("the subscription must not end before its ack")
            .expect("the subscription's ack must be Ok");
        let (mut join, _join_client) = open_join(&svc_b, &session_id, "lost-notice-app").await;
        expect_session_connected(&mut join, SETTLE).await;
        let token = svc_b
            .session_tokens
            .get(&session_id)
            .expect("B's streams registered the session's token")
            .value()
            .clone();

        // 1. With B's relay held, A deletes the session; FILLERS later notices
        //    push its notice out of B's channel.
        let mut tap = subscriber.message_rx();
        handle_delete_session(
            &svc_a,
            with_bearer(DeleteSessionRequest {
                session_id: session_id.clone(),
            }),
        )
        .await
        .expect("DeleteSession on A must succeed");
        wait_in_channel(&mut tap, |payload| payload == session_id, "A's notice").await;
        let fillers: Vec<String> = (0..FILLERS).map(|i| format!("filler-{i}")).collect();
        for filler in &fillers {
            publish_deletion(&tv, filler).await;
        }
        let last = fillers.last().expect("fillers").clone();
        wait_in_channel(&mut tap, |payload| payload == last, "the fillers").await;

        // 2. The subscription has ended on the layer's deletion; the join's
        //    token is untouched.
        let ending = tokio::time::timeout(SETTLE, async {
            loop {
                match read.next().await {
                    None => break None,
                    Some(Err(status)) => break Some(status),
                    Some(Ok(_)) => {}
                }
            }
        })
        .await
        .expect("the subscription must end on the layer's deletion, without the notice");
        assert_eq!(
            ending.map(|status| status.code()),
            Some(tonic::Code::NotFound),
            "B's subscription ends NOT_FOUND on the layer's deletion"
        );
        assert!(
            !token.is_cancelled() && svc_b.session_tokens.contains_key(&session_id),
            "precondition: before B's relay runs, nothing on B has seen the deletion"
        );

        // 3. B's relay runs, skips the notice, and requests a revalidation.
        let skipped = (1 + FILLERS - CAPACITY) as u64;
        assert!(
            drive_until_lagged(&held, &metrics, (0, skipped)).await,
            "B's relay must skip the notice and the oldest fillers: {skipped} messages, got {:?}",
            lagged(&metrics)
        );
        let (_, status) = read_to_end_within(&mut join, "B's join", SETTLE).await;
        assert!(
            status.is_none(),
            "B's join must end OK, as for a deletion, once the requested pass finds the session \
             deleted; it ended {status:?}"
        );
        assert!(
            token.is_cancelled() && !svc_b.session_tokens.contains_key(&session_id),
            "the requested pass removes the session's entry on B and cancels its token"
        );
        assert_eq!(
            token.cause(),
            SessionCancelCause::Deleted,
            "the requested pass cancels the session as a deletion"
        );
        drain_tracker(&svc_b).await;

        relay_stop.cancel();
        held.drive(relay)
            .await
            .expect("the relay must stop on its token");
        sweeper_stop.cancel();
        tokio::time::timeout(SETTLE, sweeper)
            .await
            .expect("the sweeper must stop on its token")
            .expect("the sweeper must not panic");
        let _ = subscriber.quit().await;
    })
    .await;
}

/// A session id that never existed, so it is missing from the store with no
/// deletion marker.
const UNMARKED: &str = "never-existed";

/// A `ValkeyClient` of the test's own for a session-deleted relay, its pub/sub
/// connection subscribed to the session-deleted channel, and that connection's
/// id, for [`kill_connection`]. After a drop, fred reconnects within its
/// 500 ms delay cap and re-subscribes on its own.
async fn relay_client(tv: &TestValkey) -> (ValkeyClient, i64) {
    let mut config = crate::common::test_config();
    config.valkey_pool_size = 1;
    config.valkey_reconnect_max_delay_ms = 500;
    let replica = connect_test_client(&config)
        .await
        .expect("the relay's ValkeyClient must connect");
    // Valkey refuses CLIENT ID on a connection in subscribe mode, so read the
    // connection's id before subscribing.
    let victim_id: i64 = replica
        .subscriber
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
        .expect("CLIENT ID must succeed");
    subscribe_session_deleted_channel(&replica.subscriber, tv.key_schema())
        .await
        .expect("SUBSCRIBE must succeed");
    (replica, victim_id)
}

/// Drop the relay's pub/sub connection, `victim_id`, from the server side, so
/// its client reconnects.
async fn kill_connection(tv: &TestValkey, victim_id: i64) {
    let killed: i64 = tv
        .client()
        .pool
        .next()
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT KILL", None::<u16>, false),
            vec!["ID".to_string(), victim_id.to_string()],
        )
        .await
        .expect("CLIENT KILL ID must succeed");
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID {victim_id} must kill exactly the relay's pub/sub connection"
    );
}

/// After a pub/sub reconnect, the session-deleted relay requests a
/// revalidation pass, and that pass cancels, as a deletion, a session deleted
/// while the connection could not have delivered its notice. An entry missing
/// without a deletion marker is left to the periodic pass.
///
/// The relay runs on a `ValkeyClient` of its own, whose pub/sub connection the
/// test kills by id; fred reconnects within its 500 ms delay cap. The session
/// is deleted through the store, with no notice published at all, before the
/// kill: the store state a pass then reads is the same as for a notice lost in
/// the gap, without racing the reconnect. The handle starts a reconnect's pass
/// with no random delay, and the sweeper's periodic pass never comes within
/// the test.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deleted_relay_reconnect_requests_a_pass_that_cancels_a_session_deleted_meanwhile()
{
    scenario(async {
        let tv = TestValkey::setup().await;
        let (replica, victim_id) = relay_client(&tv).await;

        let store = Arc::new(SessionStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
        let cache = SessionBasePathCache::new();
        let revalidation = SessionRevalidation::without_reconnect_jitter();
        let relay_stop = CancellationToken::new();
        let relay = spawn_session_delete_subscriber(
            replica.subscriber.clone(),
            Arc::clone(&tokens),
            cache.clone(),
            tv.key_schema().clone(),
            new_metrics(),
            revalidation.clone(),
            relay_stop.clone(),
        );
        let sweeper_stop = CancellationToken::new();
        let sweeper = spawn_session_tokens_sweeper(
            Arc::clone(&tokens),
            Arc::clone(&store),
            cache,
            NO_PERIODIC_PASS,
            revalidation,
            sweeper_stop.clone(),
        );

        let deleted_id = store
            .create("reconnect-deleted", "/p", "tester")
            .await
            .expect("create must succeed");
        let deleted = SessionCancel::new();
        tokens.insert(deleted_id.clone(), deleted.clone());
        let unmarked = SessionCancel::new();
        tokens.insert(UNMARKED.to_owned(), unmarked.clone());
        store
            .delete(&deleted_id)
            .await
            .expect("delete must succeed");
        assert!(
            !deleted.is_cancelled(),
            "precondition: no notice was published, so nothing has cancelled the session"
        );

        kill_connection(&tv, victim_id).await;

        tokio::time::timeout(SETTLE, deleted.cancelled())
            .await
            .expect(
                "after the pub/sub reconnect, the requested pass must cancel the session deleted \
                 meanwhile",
            );
        // The pass runs inside the sweeper's loop, which sees its stop token
        // only between passes: once the sweeper has stopped, the pass that
        // cancelled the session has finished with every entry.
        sweeper_stop.cancel();
        tokio::time::timeout(SETTLE, sweeper)
            .await
            .expect("the sweeper must stop on its token")
            .expect("the sweeper must not panic");
        assert_eq!(
            deleted.cause(),
            SessionCancelCause::Deleted,
            "the requested pass cancels the session as a deletion"
        );
        assert!(!tokens.contains_key(&deleted_id));
        assert!(
            !unmarked.is_cancelled() && tokens.contains_key(UNMARKED),
            "a requested pass leaves an entry missing without a deletion marker to the periodic \
             pass"
        );

        relay_stop.cancel();
        tokio::time::timeout(SETTLE, relay)
            .await
            .expect("the relay must stop on its token")
            .expect("the relay must not panic");
        replica.quit().await;
    })
    .await;
}

// ── A re-SUBSCRIBE in flight ─────────────────────────────────────────────────

/// Deletion notices published together while a re-SUBSCRIBE is held, after
/// the first one through, so the relay drains a batch during the hold.
const HELD_NOTICES: usize = 8;

/// Reconnects forced while the first re-SUBSCRIBE is held, after the one that
/// started it: more than one, so they must collapse into one owed SUBSCRIBE.
const FURTHER_RECONNECTS: usize = 2;

/// A session-deleted relay on a `ValkeyClient` of its own ([`relay_client`]),
/// spawned on a [`HeldRuntime`] through the relay's test seam, so every
/// re-SUBSCRIBE it starts after a reconnect stays in flight until the test
/// releases it on `hold`. No sweeper runs unless a test spawns one, so a
/// requested pass stays pending for the test to read; the handle's reconnect
/// requests carry no random delay.
struct HeldResubscribeRelay {
    held: HeldRuntime,
    replica: ValkeyClient,
    victim_id: i64,
    tokens: Arc<DashMap<String, SessionCancel>>,
    revalidation: SessionRevalidation,
    hold: ResubscribeHold,
    stop: CancellationToken,
    relay: Option<tokio::task::JoinHandle<()>>,
}

impl HeldResubscribeRelay {
    /// Spawn the relay and attach it (step 1 of the module header).
    async fn start(tv: &TestValkey) -> Self {
        let (replica, victim_id) = relay_client(tv).await;
        let tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
        let revalidation = SessionRevalidation::without_reconnect_jitter();
        let hold = ResubscribeHold::new();
        let stop = CancellationToken::new();
        let held = HeldRuntime::new();
        let relay = held.spawn_on(|| {
            spawn_session_delete_subscriber_with_resubscribe_hold(
                replica.subscriber.clone(),
                Arc::clone(&tokens),
                SessionBasePathCache::new(),
                tv.key_schema().clone(),
                new_metrics(),
                revalidation.clone(),
                stop.clone(),
                hold.clone(),
            )
        });
        attach_deleted_relay(tv, &held, &tokens).await;
        Self {
            held,
            replica,
            victim_id,
            tokens,
            revalidation,
            hold,
            stop,
            relay: Some(relay),
        }
    }

    /// Kill the relay's pub/sub connection and drive the relay until it has
    /// started its re-SUBSCRIBE, which the hold keeps in flight.
    async fn reconnect_and_hold(&self, tv: &TestValkey) {
        assert_eq!(
            self.hold.started(),
            0,
            "precondition: no reconnect yet, so no re-SUBSCRIBE"
        );
        kill_connection(tv, self.victim_id).await;
        assert!(
            self.drive_until_started(1).await,
            "after a pub/sub reconnect, the relay must start a re-SUBSCRIBE"
        );
    }

    /// Drive the relay until it has started `count` re-SUBSCRIBEs, or
    /// [`SETTLE`] passes; whether it did.
    async fn drive_until_started(&self, count: usize) -> bool {
        let hold = self.hold.clone();
        self.held
            .drive(async move {
                tokio::time::timeout(SETTLE, hold.wait_started(count))
                    .await
                    .is_ok()
            })
            .await
    }

    /// Drive the relay until it has requested a pass, or [`SETTLE`] passes;
    /// whether it did. The relay requests the pass and starts an owed
    /// re-SUBSCRIBE in one step, so once the request is seen, so is that start.
    async fn drive_until_pass_requested(&self) -> bool {
        let revalidation = self.revalidation.clone();
        self.held
            .drive(async move {
                tokio::time::timeout(SETTLE, revalidation.requested())
                    .await
                    .is_ok()
            })
            .await
    }

    /// Cancel the relay's token and drive it until its task ends, or
    /// [`SETTLE`] passes; whether it ended, without panicking, by then.
    async fn cancel(&mut self) -> bool {
        self.stop.cancel();
        self.drive_to_end().await
    }

    /// Drive the relay until its task ends, or [`SETTLE`] passes; whether it
    /// ended, without panicking, by then.
    async fn drive_to_end(&mut self) -> bool {
        let relay = self.relay.take().expect("the relay ends once");
        self.held
            .drive(async move { matches!(tokio::time::timeout(SETTLE, relay).await, Ok(Ok(()))) })
            .await
    }

    /// Stop the relay, if the test has not, and close its client.
    async fn finish(mut self) {
        if self.relay.is_some() {
            assert!(self.cancel().await, "the relay must stop on its token");
        }
        self.replica.quit().await;
    }
}

/// While its re-SUBSCRIBE after a reconnect is in flight, the session-deleted
/// relay keeps draining its channel: a deletion notice published then cancels
/// its session's token, and a batch of [`HELD_NOTICES`] more is applied in
/// full. The hold is still in place at each check, and no pass has been
/// requested: the relay requests the pass only once the test releases the
/// hold and the SUBSCRIBE completes.
///
/// A relay that awaits the re-SUBSCRIBE inside its reconnect handling applies
/// none of these until the SUBSCRIBE completes, which in production can take
/// up to the command timeout; the applied tokens are what this test
/// discriminates on. The lag that stall causes under presence traffic is not
/// reproduced here: the client keeps the production channel capacity, far
/// above the few messages published, so the notices would only queue. The
/// first notice after the reconnect comes through fred's own resubscription,
/// which the held re-SUBSCRIBE does not depend on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deleted_relay_keeps_applying_notices_while_its_resubscribe_after_a_reconnect_is_in_flight(
) {
    scenario(async {
        let tv = TestValkey::setup().await;
        let fx = HeldResubscribeRelay::start(&tv).await;
        fx.reconnect_and_hold(&tv).await;

        assert!(
            apply_a_fresh_notice(&tv, &fx.held, &fx.tokens, "held").await,
            "while its re-SUBSCRIBE after a reconnect is in flight, the relay must keep applying \
             deletion notices"
        );
        let notices: Vec<String> = (0..HELD_NOTICES)
            .map(|i| format!("held-batch-{i}"))
            .collect();
        let notice_tokens: Vec<CancellationToken> =
            notices.iter().map(|sid| track(&fx.tokens, sid)).collect();
        for session_id in &notices {
            publish_deletion(&tv, session_id).await;
        }
        let last = notice_tokens.last().expect("notices");
        assert!(
            drive_deleted_until(&fx.held, last, SETTLE).await,
            "the relay must apply the batch published while its re-SUBSCRIBE is in flight"
        );
        assert!(
            notice_tokens.iter().all(CancellationToken::is_cancelled),
            "every notice published while the re-SUBSCRIBE is in flight is applied"
        );
        assert_eq!(
            fx.hold.started(),
            1,
            "precondition: the notices were applied while the one re-SUBSCRIBE was still held"
        );
        assert!(
            !revalidation_requested(&fx.revalidation),
            "no pass is requested before the re-SUBSCRIBE completes"
        );

        fx.hold.release(1);
        assert!(
            fx.drive_until_pass_requested().await,
            "once the re-SUBSCRIBE completes, the relay must request a pass"
        );
        fx.finish().await;
    })
    .await;
}

/// Reconnects while a re-SUBSCRIBE is in flight neither replace it nor stack
/// more: [`FURTHER_RECONNECTS`] of them leave one re-SUBSCRIBE started, and
/// owe exactly one more, which the relay starts when the first completes. Each
/// completion requests a pass: the first's is taken by a sweeper pass that
/// cancels a session deleted before the reconnects, and the owed one's
/// completion then requests another.
///
/// The first reconnect is a `CLIENT KILL` of the relay's pub/sub connection;
/// the further ones are forced on the client (`force_reconnection`), because
/// Valkey refuses the `CLIENT ID` a kill needs on the reconnected connection,
/// which fred has put back in subscribe mode. A forced reconnect notifies the
/// relay as a dropped connection does, and its notification is queued for the
/// relay once `force_reconnection` returns, which the test checks on a
/// receiver of its own. The relay is held meanwhile; it
/// then takes every ready notification and message in one turn, so once it
/// has applied a notice published after them, it has handled the reconnects.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deleted_relay_reconnects_during_a_resubscribe_owe_exactly_one_more_with_its_own_pass(
) {
    scenario(async {
        let tv = TestValkey::setup().await;
        let mut fx = HeldResubscribeRelay::start(&tv).await;
        let store = Arc::new(SessionStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let deleted_id = store
            .create("owed-resubscribe", "/p", "tester")
            .await
            .expect("create must succeed");
        let deleted = SessionCancel::new();
        fx.tokens.insert(deleted_id.clone(), deleted.clone());
        store
            .delete(&deleted_id)
            .await
            .expect("delete must succeed");

        // 1. A reconnect starts a re-SUBSCRIBE, held; more reconnects follow.
        fx.reconnect_and_hold(&tv).await;
        let mut reconnect_tap = fx.replica.subscriber.reconnect_rx();
        for _ in 0..FURTHER_RECONNECTS {
            fx.replica
                .subscriber
                .force_reconnection()
                .await
                .expect("forcing a reconnect must succeed");
        }
        let notified = std::iter::from_fn(|| reconnect_tap.try_recv().ok()).count();
        assert!(
            notified >= FURTHER_RECONNECTS,
            "precondition: each forced reconnect notifies the client's reconnect receivers, the \
             relay's included; {notified} of {FURTHER_RECONNECTS} did"
        );
        attach_deleted_relay(&tv, &fx.held, &fx.tokens).await;
        assert_eq!(
            fx.hold.started(),
            1,
            "a reconnect while a re-SUBSCRIBE is in flight must not start another"
        );
        assert!(
            !revalidation_requested(&fx.revalidation),
            "no pass is requested before the re-SUBSCRIBE completes"
        );

        // 2. The first completes: its pass is requested, and the owed one
        //    starts.
        fx.hold.release(1);
        assert!(
            fx.drive_until_started(2).await,
            "once the in-flight re-SUBSCRIBE completes, the relay must start the one the later \
             reconnects owe"
        );
        assert!(
            revalidation_requested(&fx.revalidation),
            "the first re-SUBSCRIBE's completion requests a pass"
        );

        // A sweeper takes that request: its pass cancels the deleted session,
        // and once it has stopped, the pass has finished.
        let sweeper_stop = CancellationToken::new();
        let sweeper = spawn_session_tokens_sweeper(
            Arc::clone(&fx.tokens),
            Arc::clone(&store),
            SessionBasePathCache::new(),
            NO_PERIODIC_PASS,
            fx.revalidation.clone(),
            sweeper_stop.clone(),
        );
        tokio::time::timeout(SETTLE, deleted.cancelled())
            .await
            .expect("the requested pass must cancel the session deleted before the reconnects");
        sweeper_stop.cancel();
        tokio::time::timeout(SETTLE, sweeper)
            .await
            .expect("the sweeper must stop on its token")
            .expect("the sweeper must not panic");
        assert!(
            !revalidation_requested(&fx.revalidation),
            "precondition: the sweeper took the first completion's request"
        );

        // 3. The owed one completes: a pass of its own, and no third
        //    re-SUBSCRIBE.
        fx.hold.release(1);
        assert!(
            fx.drive_until_pass_requested().await,
            "the owed re-SUBSCRIBE's completion must request a pass of its own"
        );
        assert_eq!(
            fx.hold.started(),
            2,
            "the {FURTHER_RECONNECTS} reconnects during the first re-SUBSCRIBE owe one more, not \
             one each"
        );

        assert!(fx.cancel().await, "the relay must stop on its token");
        fx.finish().await;
    })
    .await;
}

/// Cancelling the session-deleted relay's token while its re-SUBSCRIBE after a
/// reconnect is in flight ends the task at once: the hold is never released,
/// so the SUBSCRIBE never completes, and the task still ends within
/// [`SETTLE`].
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deleted_relay_cancel_while_its_resubscribe_is_in_flight_ends_the_task() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let mut fx = HeldResubscribeRelay::start(&tv).await;
        fx.reconnect_and_hold(&tv).await;
        assert!(
            !revalidation_requested(&fx.revalidation),
            "precondition: the re-SUBSCRIBE has not completed"
        );

        assert!(
            fx.cancel().await,
            "cancelling the relay while its re-SUBSCRIBE is in flight must end its task"
        );
        fx.finish().await;
    })
    .await;
}

/// When its client's message channel closes while its re-SUBSCRIBE after a
/// reconnect is in flight, the session-deleted relay requests that
/// reconnect's pass before its task ends, rather than leave the reconnect
/// without one. The channel closes as the client quits (fred closes its
/// receivers then); the relay's token is never cancelled and the hold never
/// released, so the task ends on the closed channel alone, with the
/// re-SUBSCRIBE still in flight.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deleted_relay_channel_closed_while_its_resubscribe_is_in_flight_requests_the_pass()
{
    scenario(async {
        let tv = TestValkey::setup().await;
        let mut fx = HeldResubscribeRelay::start(&tv).await;
        fx.reconnect_and_hold(&tv).await;
        assert!(
            !revalidation_requested(&fx.revalidation),
            "precondition: the re-SUBSCRIBE has not completed, so no pass is requested yet"
        );

        fx.replica.quit().await;
        assert!(
            fx.drive_to_end().await,
            "once its client's message channel closes, the relay's task must end on its own"
        );
        assert!(
            revalidation_requested(&fx.revalidation),
            "a relay that ends on a closed channel with a re-SUBSCRIBE in flight must request \
             that reconnect's pass"
        );
    })
    .await;
}
