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

// Regression test: ValkeyClient::quit() must return within a bounded time
// even while a stream reader is parked in a 30 s `XREAD BLOCK`.
//
// fred's QUIT drains in-flight commands first, so a reader connection's quit
// behind a pending BLOCK would wait out the block. The stream multiplexer wakes
// its in-flight reads on shutdown, and every pool quit — and each connected
// reader's — is additionally bounded by a 2 s timeout, after which process
// exit closes the TCP connections (a reader's router is aborted at once).
//
// Test construction note: we build a standalone `ValkeyClient` directly
// rather than going through `TestValkey`. `TestValkey::Drop` cleans up
// keys via the shared pool; after `quit()` tears the pool down, its `SCAN`
// gets no answer within `CLEANUP_STEP_BUDGET`, so `Drop` panics and fails a
// test that otherwise passed (it only reports it when the test is already
// failing). Owning the client directly lets us call `quit()` and then just
// drop without a cleanup step. There are no keys to clean up because the
// test doesn't write any.

use std::time::Duration;

use live_edit::config::Config;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::notifications::{consume_layer_stream, StreamCursor};

/// b.ohv item 8: delegate to `common::test_config()`. The test exercises the
/// quit-timeout path with a single-connection pool, so the override list is
/// just `valkey_pool_size = 1`.
fn test_config() -> Config {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = 1;
    cfg
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn client_quit_completes_even_with_active_subscriber() {
    let config = test_config();
    let client = ValkeyClient::connect(&config, None, None)
        .await
        .expect("connect must succeed");
    let prefix = format!("test_quit_{}:", uuid::Uuid::new_v4());
    let keys = KeySchema::new(&prefix);
    let object_id: u64 = 9_999;

    let consumer = consume_layer_stream(
        &client,
        &keys,
        object_id,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");

    // Wait until the server shows the reader parked in its (production, 30 s)
    // XREAD BLOCK: the read is blocked, so a quit that waits behind it would
    // wait out the BLOCK.
    await_a_reader_blocked_in_xread(&client, client.pool.next()).await;

    // Drop without calling unsubscribe(), as when the gRPC handler future is
    // dropped: the registration goes, but the reader's XREAD stays blocked.
    drop(consumer);

    // Must complete well within a few seconds even though the block has
    // ~30 s to run.
    let start = std::time::Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(10), client.quit()).await;
    let elapsed = start.elapsed();

    result.expect("quit must return; if this times out the shutdown-hang regression is back");

    // Quits run in parallel under one 2 s timeout, after a bounded (1 s)
    // multiplexer shutdown.
    assert!(
        elapsed < Duration::from_secs(5),
        "quit took {elapsed:?}; a blocked reader must not hold shutdown for its BLOCK"
    );
}

// ── C-2 regression: parallel quit bounded by single per-pool timeout ──────────
//
// Pre-C-2, `ValkeyClient::quit()` ran its pool quits sequentially, each with
// its own 2 s timeout, for a worst-case ~8 s of shutdown cost — past the K8s
// default `terminationGracePeriodSeconds: 30` once you stack it on top of the
// background-task join budgets, leading to SIGKILL during cleanup. The fix
// wraps every pool's `quit()` in a single `tokio::join!` so they proceed in
// parallel and the worst case is one 2 s timeout total.

/// Waits until the server reports one of `client`'s stream reader connections
/// blocked in its `XREAD`, polled over `observer`. The reader's XREAD count
/// rises before the command is sent, so the count alone does not prove the
/// read is blocked on the server. Readers connect lazily, when they first
/// serve a subscription, so this first waits for the subscribed reader to
/// hold its connection.
async fn await_a_reader_blocked_in_xread(client: &ValkeyClient, observer: &fred::clients::Client) {
    use fred::interfaces::ClientInterface;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while client.stream_mux.connected_readers() == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "no reader connected to serve the subscription"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let ids: Vec<String> = client
        .stream_mux
        .reader_clients_for_test()
        .iter()
        .flat_map(|c| c.connection_ids().into_values().map(|id| id.to_string()))
        .collect();
    assert!(
        !ids.is_empty(),
        "fred caches every connected reader's CLIENT ID on connect"
    );
    let field = |line: &str, name: &str| -> String {
        line.split_whitespace()
            .find_map(|kv| kv.strip_prefix(&format!("{name}=")[..]).map(str::to_owned))
            .unwrap_or_default()
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let listing: String = observer
            .client_list::<String, String>(None, Some(ids.clone()))
            .await
            .expect("CLIENT LIST ID must succeed on the observer connection");
        if listing
            .lines()
            .any(|l| field(l, "cmd") == "xread" && field(l, "flags").contains('b'))
        {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no reader connection blocked in its XREAD; last CLIENT LIST:\n{listing}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Every connection pool stalls at once, and `quit()` still returns within one
/// per-pool timeout (2 s) plus the shutdown wake's 0.5 s timeout — which only a
/// quit that runs its pools in parallel can do. (The pub/sub client is not a
/// pool and quits cleanly.)
///
/// Five pools stall, each a different way:
///
/// * **txn** — the test holds a slot, and fred's quit waits on every slot's
///   mutex.
/// * **data**, **maintenance** and the **snapshot** lane — every connection is
///   parked in a server-side `BLPOP`
///   ([`stall_pool`](live_edit::storage::test_support::stall_pool)), so each
///   `QUIT` queues behind the block.
/// * **stream readers** — the subscription's reader (connected when it first
///   served it: readers connect lazily) is blocked in its 30 s `XREAD BLOCK`.
///   The multiplexer's shutdown wake goes through the stalled data pool,
///   cannot land, and gives up after 0.5 s, so the reader connection's `QUIT`
///   queues behind the block too, and is cut off at its own 2 s bound.
///
/// A serial quit pays three or more 2 s timeouts, past the 4.5 s bound; a
/// parallel one pays one, plus the 0.5 s wake. The capture proves each of the
/// five pools really did time out and that the wake did not land, so a quick
/// quit cannot pass because something was not stalled after all — and that the
/// snapshot lane is quit at all, inside the same parallel join.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn client_quit_parallel_bound_holds_when_all_pools_stall() {
    use live_edit::storage::test_support::{release_all, stall_pool};

    let config = test_config();
    let client = ValkeyClient::connect(&config, None, None)
        .await
        .expect("connect must succeed");
    // Polls and releases the stalls; it must belong to neither stalled pool.
    let observer_client = ValkeyClient::connect(&config, None, None)
        .await
        .expect("connect must succeed");
    let observer = observer_client.pool.next().clone();
    let prefix = format!("test_quit_{}:", uuid::Uuid::new_v4());

    let _consumer = consume_layer_stream(
        &client,
        &KeySchema::new(&prefix),
        4_242,
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");
    await_a_reader_blocked_in_xread(&client, &observer).await;
    let held_txn_slot = client.acquire_txn().await;
    let mut stalls = stall_pool(&client.pool, &observer, &format!("{prefix}stall:data:")).await;
    stalls.extend(
        stall_pool(
            &client.maintenance_pool,
            &observer,
            &format!("{prefix}stall:maintenance:"),
        )
        .await,
    );
    stalls.extend(
        stall_pool(
            &client.snapshot_pool,
            &observer,
            &format!("{prefix}stall:snapshot:"),
        )
        .await,
    );

    let start = std::time::Instant::now();
    let (quit, captured) = crate::grpc::log_capture::capture_debug(tokio::time::timeout(
        Duration::from_secs(20),
        client.quit(),
    ))
    .await;
    let elapsed = start.elapsed();
    quit.expect("quit must complete within 20 s even with every pool stalled");

    for timed_out in [
        "data pool quit timed out; leaving connections to be closed by process exit",
        "txn pool quit timed out; leaving connections to be closed by process exit",
        "stream reader connection quit timed out (a reader's blocked XREAD was not woken, or \
         the server did not answer); those readers' routers were aborted, closing their \
         connections",
        "maintenance pool quit timed out; leaving connections to be closed by process exit",
        "snapshot pool quit timed out; leaving connections to be closed by process exit",
    ] {
        captured.only(timed_out);
    }
    let woke = captured.only("woke blocked stream reads before the reader pool QUIT");
    assert_eq!(
        woke["fields"]["woken"], 0,
        "the shutdown wake must not land through the stalled data pool; full capture:\n{}",
        captured.text
    );

    // One 2 s pool timeout after the 0.5 s shutdown wake, against a 4.5 s
    // bound: 2 s of margin. A serial quit of the five stalled pools takes
    // 10.5 s.
    assert!(
        elapsed < Duration::from_millis(4_500),
        "quit() took {elapsed:?} with every pool stalled; pool quits must run in \
         parallel under one timeout"
    );

    drop(held_txn_slot);
    // Each stall held through the quit; releasing it lets the queued QUIT run.
    release_all(stalls).await;
    observer_client.quit().await;
}

/// `QUIT_TIMEOUT` (`storage::client`, crate-private): the bound on each reader
/// connection's `QUIT` at retire.
const READER_QUIT_TIMEOUT: Duration = Duration::from_secs(2);

/// `StreamMux::quit_readers` retires every connected reader concurrently: two
/// readers whose `QUIT`s both queue behind a blocked `XREAD` both overrun
/// their bound, and the step takes one `QUIT_TIMEOUT`, not one per reader.
///
/// Measured on tokio's paused clock, so the bound is exact and no wall-clock
/// time is asserted: the readers are connected and blocked first, on the
/// real clock, and the clock is paused only for the step. Paused, the runtime
/// jumps to its next timer whenever every task is waiting — here the two
/// `QUIT_TIMEOUT`s, both armed at the same instant when the retires run
/// together, one after the other when they do not. Nothing the reader
/// connections wait on can answer meanwhile: each read is blocked for the
/// production 30 s, and nothing wakes it, since `quit_readers` is called on
/// its own (no `StreamMux::shutdown` wake first).
///
/// Current-thread: `tokio::time::pause` requires it.
#[tokio::test]
async fn quit_readers_retires_stalled_readers_concurrently_within_one_quit_timeout() {
    const SETTLE: Duration = Duration::from_secs(10);
    let mut config = test_config();
    config.subscriber_pool_size = 2;
    let client = tokio::time::timeout(
        Duration::from_secs(30),
        live_edit::storage::test_support::connect_test_client(&config),
    )
    .await
    .expect("connect must settle")
    .expect("connect must succeed");
    let schema = KeySchema::new(&format!("test_quit_{}:", uuid::Uuid::new_v4()));

    // Standalone routing sends object `n` to reader `n % 2`.
    let mut subscriptions = Vec::new();
    for object_id in [0, 1] {
        subscriptions.push(
            consume_layer_stream(
                &client,
                &schema,
                object_id,
                StreamCursor::From("0-0".into()),
                tokio::time::Instant::now() + SETTLE,
            )
            .await
            .expect("the subscription must start"),
        );
    }
    let deadline = tokio::time::Instant::now() + SETTLE;
    while client.stream_mux.connected_readers() < 2 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "both readers must connect to serve their subscriptions"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for reader in 0..2 {
        let reader = client
            .stream_mux
            .reader_client_for_test(reader)
            .expect("the reader holds a connection");
        super::mux_test_helpers::await_reader_blocked_in_xread(client.pool.next(), &reader).await;
    }
    // A subscription leaving does not wake the read: both stay blocked.
    drop(subscriptions);

    tokio::time::pause();
    let start = tokio::time::Instant::now();
    let overran = client.stream_mux.quit_readers().await;
    let took = start.elapsed();
    tokio::time::resume();

    assert_eq!(
        overran, 2,
        "precondition: both readers' QUITs queued behind their blocked XREADs and overran"
    );
    assert!(
        took >= READER_QUIT_TIMEOUT && took < 2 * READER_QUIT_TIMEOUT,
        "the two retires run concurrently: one QUIT_TIMEOUT ({READER_QUIT_TIMEOUT:?}) of tokio \
         time, not one per reader; took {took:?}"
    );
    assert_eq!(client.stream_mux.connected_readers(), 0);

    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");
}

// ── The two `SubscribeReadLayer` gates close at quit ──────────────────────────

/// `ValkeyClient::quit` closes the subscription admission gate and the
/// catch-up gate, so a subscribe parked in either wait is released at once as
/// a shutdown — the admission waiter `ShuttingDown`, the catch-up waiter with
/// no slot — instead of waiting out its bound. A subscribe arriving after the
/// close is refused as a shutdown too, and recorded under
/// `result="cancelled"` (this replica is going away), never
/// `refused_saturated` (this replica is over capacity).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quit_releases_parked_admission_and_catch_up_waiters_as_shutdown() {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::observability::metrics::{
        ADMISSION_RESULT_ADMITTED, ADMISSION_RESULT_CANCELLED, ADMISSION_RESULT_REFUSED_SATURATED,
    };
    use live_edit::storage::AdmissionRefusal;

    const SETTLE: Duration = Duration::from_secs(10);

    let mut config = test_config();
    config.subscriber_max_subscriptions = 1;
    config.subscriber_max_concurrent_catchups = 1;
    // Long enough that the parked admission cannot refuse itself first.
    config.subscriber_admission_wait_ms = 60_000;
    let client = live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("connect must succeed");
    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;
    let admissions = |result: &str| {
        sample_value(
            &render(&metrics),
            "live_edit_subscriber_admission_total",
            &[("result", result)],
        )
        .expect("every result is pre-initialised") as u64
    };

    let _admitted = client
        .admit_subscriber(None)
        .await
        .expect("the one admission slot is free");
    // Far enough away that the parked catch-up cannot time out first.
    let catch_up_deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    let _catching_up = client
        .catchup_gate
        .acquire(catch_up_deadline)
        .await
        .expect("the one catch-up slot is free");

    let mut parked_admission = std::pin::pin!(client.admit_subscriber(None));
    assert!(
        futures::poll!(parked_admission.as_mut()).is_pending(),
        "past the cap the admission waits"
    );
    let mut parked_catch_up = std::pin::pin!(client.catchup_gate.acquire(catch_up_deadline));
    assert!(
        futures::poll!(parked_catch_up.as_mut()).is_pending(),
        "past the limit the catch-up waits"
    );

    client.quit().await;

    let admission = tokio::time::timeout(SETTLE, parked_admission)
        .await
        .expect("quit must release the parked admission");
    assert!(
        matches!(admission, Err(AdmissionRefusal::ShuttingDown)),
        "the parked admission is refused as a shutdown"
    );
    let catch_up = tokio::time::timeout(SETTLE, parked_catch_up)
        .await
        .expect("quit must release the parked catch-up");
    assert!(
        matches!(
            catch_up,
            Err(live_edit::storage::subscriber_admission::CatchUpRefusal::Closed)
        ),
        "the parked catch-up gets no slot"
    );

    assert!(matches!(
        client.admit_subscriber(None).await,
        Err(AdmissionRefusal::ShuttingDown)
    ));
    assert_eq!(admissions(ADMISSION_RESULT_ADMITTED), 1);
    assert_eq!(
        admissions(ADMISSION_RESULT_CANCELLED),
        2,
        "the released waiter and the late arrival both record `cancelled`"
    );
    assert_eq!(
        admissions(ADMISSION_RESULT_REFUSED_SATURATED),
        0,
        "a replica shutting down is never reported as over capacity"
    );
}

// ── The fenced read-permit gate and the catch-up memory budget ────────────────

/// `ValkeyClient` builds the fenced read-permit gate with
/// `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY` permits, as a gate of its own
/// beside the atomic one and counted under `class="fenced"` (Spec SDD SR-7.1),
/// and the catch-up memory budget at `Config::catchup_build_memory_budget_mib`
/// (SR-7.2). `quit` closes both: a fenced-permit wait and a memory wait parked
/// at the time end `Closed`, and every later acquisition, reservation and
/// non-blocking grant is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_fenced_read_gate_and_memory_budget_follow_their_configuration_and_close_at_quit() {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::observability::metrics::{
        CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC, CATCHUP_SNAPSHOT_READ_CLASS_FENCED,
    };
    use live_edit::storage::diff_read::SnapshotReadConsumer;
    use live_edit::storage::subscriber_admission::CatchUpRefusal;

    const SETTLE: Duration = Duration::from_secs(10);
    const FENCED_PERMITS: u32 = 2;
    const BUDGET_MIB: u64 = 3;
    const MIB: u64 = 1024 * 1024;
    const CATCHUP: SnapshotReadConsumer = SnapshotReadConsumer::Catchup;

    let mut config = test_config();
    config.catchup_snapshot_read_concurrency = 1;
    config.catchup_fenced_read_concurrency = FENCED_PERMITS;
    config.catchup_build_memory_budget_mib = BUDGET_MIB;
    let client = live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("connect must succeed");
    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;
    let in_flight = |class: &str| {
        sample_value(
            &render(&metrics),
            "live_edit_catchup_snapshot_reads_in_flight",
            &[("class", class)],
        )
        .expect("both classes are pre-registered") as i64
    };
    // Far enough away that no wait here can time out first.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);

    let fenced = &client.catchup_fenced_read_gate;
    assert_eq!(
        (fenced.limit(), fenced.available()),
        (FENCED_PERMITS as usize, FENCED_PERMITS as usize),
        "the fenced gate holds the configured number of permits, all free"
    );
    let held: Vec<_> =
        futures::future::join_all((0..FENCED_PERMITS).map(|_| fenced.acquire(deadline, CATCHUP)))
            .await
            .into_iter()
            .map(|r| r.expect("within the fenced limit"))
            .collect();
    assert_eq!(in_flight(CATCHUP_SNAPSHOT_READ_CLASS_FENCED), 2);
    assert_eq!(in_flight(CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC), 0);
    let atomic = match futures::poll!(std::pin::pin!(client
        .catchup_read_gate
        .acquire(deadline, CATCHUP)))
    {
        std::task::Poll::Ready(Ok(p)) => p,
        std::task::Poll::Ready(Err(e)) => panic!("the atomic read was refused: {e:?}"),
        std::task::Poll::Pending => {
            panic!("an exhausted fenced gate must not delay the client's atomic gate")
        }
    };
    drop(atomic);

    let budget = &client.catchup_build_budget;
    assert_eq!(
        budget.total_bytes(),
        BUDGET_MIB * MIB,
        "the budget is the configured MiB"
    );
    let full = budget
        .reserve(BUDGET_MIB * MIB, None)
        .await
        .expect("the whole budget is free");

    let mut parked_permit = std::pin::pin!(fenced.acquire(deadline, CATCHUP));
    assert!(
        futures::poll!(parked_permit.as_mut()).is_pending(),
        "past the fenced limit the read waits"
    );
    let mut parked_memory = std::pin::pin!(budget.reserve(1, None));
    assert!(
        futures::poll!(parked_memory.as_mut()).is_pending(),
        "past the budget the reservation waits"
    );

    client.quit().await;

    assert!(matches!(
        tokio::time::timeout(SETTLE, parked_permit)
            .await
            .expect("quit must release the parked fenced-permit wait"),
        Err(CatchUpRefusal::Closed)
    ));
    assert!(matches!(
        tokio::time::timeout(SETTLE, parked_memory)
            .await
            .expect("quit must release the parked memory wait"),
        Err(CatchUpRefusal::Closed)
    ));
    drop((held, full));
    assert!(matches!(
        fenced.acquire(deadline, CATCHUP).await,
        Err(CatchUpRefusal::Closed)
    ));
    assert!(matches!(
        budget.reserve(1, None).await,
        Err(CatchUpRefusal::Closed)
    ));
    assert!(
        budget.try_grant(1).is_none(),
        "a closed budget grants nothing"
    );
}
