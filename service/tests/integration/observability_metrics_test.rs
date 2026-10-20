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

// Regression tests for b.3z7 H-obs-3: observability metrics.
//
// Covered metrics:
//   - `live_edit_valkey_reconnects_total{pool}` — ticks on every fred
//     reconnect event. Test: force a CLIENT KILL, wait for fred to
//     reconnect, observe the counter.
//   - `live_edit_valkey_errors_total{pool, kind}` — pre-initialised with its
//     one service-owned kind, `command_timeout`, on every pool (b.21r). The
//     per-funnel counting tests live beside each fault seam (see
//     `live_edit::storage::test_support::command_timeout_row`).
//   - `live_edit_background_task_panics_total{task}` — ticks when the
//     shutdown supervisor's `JoinHandle::await` resolves to
//     `Err(JoinError)` where `is_panic()` is true. Test: feed a synthetic
//     panicked JoinError through `Metrics::inc_background_task_panic`
//     directly and observe the counter. (The production code path —
//     `report_background_task_result` in `src/main.rs` — is private to
//     the binary crate and not callable from integration tests; the
//     metric-side contract is what matters for operators.)

use std::time::Duration;

use fred::interfaces::ClientLike;
use fred::types::CustomCommand;

use live_edit::config::Config;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::notifications::{consume_layer_stream, StreamCursor};

/// Extract the set of values for `label` across all rendered series of
/// `metric` in the Prometheus text exposition. Used by the pre-registration
/// tests that pin the bounded `pool` label set.
fn label_value_set(text: &str, metric: &str, label: &str) -> std::collections::HashSet<String> {
    let needle = format!("{label}=\"");
    let mut set = std::collections::HashSet::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let Some(rest) = line.strip_prefix(metric) else {
            continue;
        };
        if !rest.starts_with('{') {
            continue;
        }
        if let Some(i) = line.find(&needle) {
            let after = &line[i + needle.len()..];
            if let Some(end) = after.find('"') {
                set.insert(after[..end].to_string());
            }
        }
    }
    set
}

/// b.ohv item 8: delegate to `common::test_config()` and override only the
/// fields this test cares about. The pre-migration version of this helper
/// duplicated every `Config` field — keeping it in sync with new fields
/// upstream was a recurring chore the canonical helper now owns.
///
/// Reconnect-max-delay is tightened to 500 ms because this test deliberately
/// kills the Valkey connection (`CLIENT KILL`) and waits for the reconnect
/// counter to tick. The production default of 30 s would stall the test for
/// half a minute on a transient drop.
fn test_config() -> Config {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = 1;
    cfg.subscriber_pool_size = 1;
    cfg.valkey_reconnect_max_delay_ms = 500;
    cfg
}

async fn maybe_connect() -> Option<ValkeyClient> {
    let cfg = test_config();
    match tokio::time::timeout(
        Duration::from_secs(5),
        live_edit::storage::test_support::connect_test_client(&cfg),
    )
    .await
    {
        Ok(Ok(c)) => Some(c),
        _ => None,
    }
}

/// Read `live_edit_valkey_reconnects_total{pool}` from the rendered scrape
/// exposition. Pre-initialised in `Metrics::new`, so present from boot.
fn read_reconnects(metrics: &Metrics, pool: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_reconnects_total",
        &[("pool", pool)],
    )
    .unwrap_or(0.0) as u64
}

/// Read `live_edit_background_task_panics_total{task}` from the rendered scrape
/// exposition. Not pre-initialised (the `task` label is opened lazily), so an
/// absent series reads as zero.
fn read_panic_count(metrics: &Metrics, task: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_background_task_panics_total",
        &[("task", task)],
    )
    .unwrap_or(0.0) as u64
}

/// Read the victim pool-client's Valkey-assigned connection ID so we
/// can kill it by ID — scoped to exactly this test, not any sibling
/// test's pool connection.
async fn pool_client_connection_id(victim: &ValkeyClient) -> i64 {
    victim
        .pool
        .next()
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
        .expect("CLIENT ID must succeed")
}

/// Kill the specific connection identified by `victim_id` using
/// `CLIENT KILL ID <id>`. Scoping by ID avoids cross-test interference
/// that `CLIENT KILL TYPE normal` causes when other tests run in
/// parallel with their own pool connections.
async fn kill_connection_by_id(killer: &ValkeyClient, victim_id: i64) -> i64 {
    killer
        .pool
        .next()
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT KILL", None::<u16>, false),
            vec!["ID".to_string(), victim_id.to_string()],
        )
        .await
        .expect("CLIENT KILL ID must succeed")
}

/// After the reconnect event hooks are installed, a CLIENT KILL on the
/// data pool causes fred to emit a reconnect event, which the hook
/// increments into `valkey_reconnects_total{pool="data"}`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_metric_increments_on_data_pool_kill() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let Some(killer) = maybe_connect().await else {
        eprintln!("SKIP: second Valkey connection not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));

    // Wire fred's event streams into the metrics registry. This is the
    // call main.rs makes at startup.
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    // Baseline: pre-init the label so reads below see 0 rather than an
    // unregistered-series panic.
    let before = read_reconnects(&metrics, "data");

    // Scope the kill to the victim's specific pool connection so we
    // don't knock out other parallel tests' pool clients.
    let victim_id = pool_client_connection_id(&client).await;
    let killed = kill_connection_by_id(&killer, victim_id).await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID must close exactly the victim connection"
    );

    // fred's reconnect policy is `valkey_reconnect_max_delay_ms = 500` so
    // the reconnect settles quickly. Poll the counter for up to 5 s.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut observed_increment = false;
    while std::time::Instant::now() < deadline {
        if read_reconnects(&metrics, "data") > before {
            observed_increment = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    assert!(
        observed_increment,
        "valkey_reconnects_total{{pool=\"data\"}} must increment after CLIENT KILL + fred reconnect — \
         pre-H-obs-3 there was no reconnect instrumentation so operators had no \"Valkey in trouble\" signal"
    );

    client.quit().await;
    killer.quit().await;
}

/// Poll `valkey_reconnects_total{pool=<label>}` for up to `deadline`,
/// returning true once it exceeds `baseline`. Shared by the five
/// sibling tests below so the polling shape stays identical across
/// pools — only the label and the kill mechanism differ.
async fn wait_for_reconnect_increment(
    metrics: &Metrics,
    pool: &str,
    baseline: u64,
    deadline: Duration,
) -> bool {
    let stop = std::time::Instant::now() + deadline;
    while std::time::Instant::now() < stop {
        if read_reconnects(metrics, pool) > baseline {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

/// Maintenance pool variant of `reconnect_metric_increments_on_data_pool_kill`.
///
/// The `maintenance_pool` is fred's multiplexed `Pool` (same shape as
/// `pool`), hooked separately in `install_observability_hooks` so a
/// reconnect storm caused by the periodic GC + orphan sweeps is
/// attributable to the maintenance pool in dashboards. This test pins
/// the b.avd HIGH follow-up contract that the `for client in
/// self.maintenance_pool.clients()` hook-installation block actually
/// drives the `valkey_reconnects_total{pool="maintenance"}` counter on
/// a real reconnect — a future regression that drops the block from
/// `install_observability_hooks` would slip past the
/// `reconnect_metric_uses_bounded_pool_label_set` pre-init test (which
/// only checks the registration step, not the hook).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_metric_increments_on_maintenance_pool_kill() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let Some(killer) = maybe_connect().await else {
        eprintln!("SKIP: second Valkey connection not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    let before = read_reconnects(&metrics, "maintenance");

    // CLIENT ID on the maintenance pool's connection — `maintenance_pool`
    // is exposed as a multiplexed `Pool`, so `.next()` returns the active
    // `Client`.
    let victim_id: i64 = client
        .maintenance_pool
        .next()
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
        .expect("CLIENT ID must succeed on maintenance pool");
    let killed = kill_connection_by_id(&killer, victim_id).await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID must close exactly the maintenance victim connection"
    );

    assert!(
        wait_for_reconnect_increment(&metrics, "maintenance", before, Duration::from_secs(5))
            .await,
        "valkey_reconnects_total{{pool=\"maintenance\"}} must increment after CLIENT KILL + fred \
         reconnect; pre-b.avd HIGH the maintenance pool was never hooked into the metric stream so \
         alerts keyed on rate(valkey_reconnects_total{{pool=\"maintenance\"}}[5m]) could never fire"
    );

    client.quit().await;
    killer.quit().await;
}

/// Snapshot-lane variant of `reconnect_metric_increments_on_maintenance_pool_kill`.
///
/// The `snapshot_pool` is the same multiplexed `Pool` shape, hooked in its own
/// `install_observability_hooks` block under `pool="snapshot"` so a reconnect
/// on a snapshot-read connection is not folded into the data pool's series.
/// The pre-init test only proves the label is registered; this one proves the
/// hook block exists and fires on a real reconnect.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_metric_increments_on_snapshot_pool_kill() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let Some(killer) = maybe_connect().await else {
        eprintln!("SKIP: second Valkey connection not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    let before_snapshot = read_reconnects(&metrics, "snapshot");
    let before_data = read_reconnects(&metrics, "data");

    let victim_id: i64 = client
        .snapshot_pool
        .next()
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
        .expect("CLIENT ID must succeed on the snapshot lane");
    let killed = kill_connection_by_id(&killer, victim_id).await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID must close exactly the snapshot-lane victim connection"
    );

    assert!(
        wait_for_reconnect_increment(
            &metrics,
            "snapshot",
            before_snapshot,
            Duration::from_secs(5)
        )
        .await,
        "valkey_reconnects_total{{pool=\"snapshot\"}} must increment after CLIENT KILL + fred \
         reconnect; without the snapshot-lane block in install_observability_hooks the label is \
         registered but never moves"
    );
    assert_eq!(
        read_reconnects(&metrics, "data"),
        before_data,
        "a snapshot-lane reconnect must not be counted against the data pool"
    );

    client.quit().await;
    killer.quit().await;
}

/// Subscriber (pub/sub `SubscriberClient`) variant. The subscriber
/// connection is a separate TCP session from the multiplexed pools,
/// hooked at line 599-608 in `install_observability_hooks`. This test
/// pins that hook fires on real reconnects so the pre-init label
/// `valkey_reconnects_total{pool="subscriber"}` actually grows on a
/// pub/sub disconnect.
///
/// Reads CLIENT ID over the subscriber connection itself (pre-subscribe
/// — subscribe-mode rejects `CLIENT ID`), then issues CLIENT KILL ID
/// from a separate `ValkeyClient`'s data pool to scope the kill to
/// exactly this victim and not collateral-kill other parallel tests'
/// pub/sub connections.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_metric_increments_on_subscriber_pool_kill() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let Some(killer) = maybe_connect().await else {
        eprintln!("SKIP: second Valkey connection not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    let before = read_reconnects(&metrics, "subscriber");

    // CLIENT ID over the pub/sub connection itself — must be issued
    // BEFORE any SUBSCRIBE (subscribe-mode rejects `CLIENT ID`).
    let victim_id: i64 = client
        .subscriber
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
        .expect("CLIENT ID must succeed on subscriber connection");
    let killed = kill_connection_by_id(&killer, victim_id).await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID must close exactly the subscriber victim connection"
    );

    assert!(
        wait_for_reconnect_increment(&metrics, "subscriber", before, Duration::from_secs(5)).await,
        "valkey_reconnects_total{{pool=\"subscriber\"}} must increment after CLIENT KILL + fred \
         reconnect; pins the subscriber hook in install_observability_hooks against future \
         regressions"
    );

    client.quit().await;
    killer.quit().await;
}

/// `txn_pool` (ExclusivePool) variant. The txn pool is hooked at line
/// 642-646 in `install_observability_hooks` — fred exposes it as
/// `&[RefCount<AsyncMutex<Client>>]`, so the hook installation locks
/// each pooled client briefly to clone the inner `Client` and spawn
/// the reconnect/error tasks against the clone. Pre-b.avd HIGH this
/// block did not exist, so a connection drop on a txn-pool slot
/// silently failed the `rate(valkey_reconnects_total{pool="txn"}[5m])`
/// alert.
///
/// `pool_size=1` (from `test_config`) means the txn pool has exactly
/// one underlying connection; we acquire the slot, read CLIENT ID,
/// release the guard, and kill via a separate client. The guard MUST
/// be dropped before the kill so fred is free to drive the reconnect
/// — otherwise we'd hold the slot across the kill and the reconnect
/// would queue behind our own borrow.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_metric_increments_on_txn_pool_kill() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let Some(killer) = maybe_connect().await else {
        eprintln!("SKIP: second Valkey connection not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    let before = read_reconnects(&metrics, "txn");

    // Acquire the txn slot, read CLIENT ID over it, then drop the guard
    // so fred's reconnect path has the slot to itself.
    let victim_id: i64 = {
        let checkout = client.acquire_txn().await;
        checkout
            .custom::<i64, String>(
                CustomCommand::new_static("CLIENT ID", None::<u16>, false),
                vec![],
            )
            .await
            .expect("CLIENT ID must succeed on txn-pool connection")
    };
    let killed = kill_connection_by_id(&killer, victim_id).await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID must close exactly the txn-pool victim connection"
    );

    assert!(
        wait_for_reconnect_increment(&metrics, "txn", before, Duration::from_secs(5)).await,
        "valkey_reconnects_total{{pool=\"txn\"}} must increment after CLIENT KILL + fred reconnect; \
         pre-b.avd HIGH the ExclusivePool hook block did not exist, so the txn-pool's reconnect \
         events were silently invisible to operators despite the label being pre-registered"
    );

    client.quit().await;
    killer.quit().await;
}

/// Stream reader pool (`pool="subscriber_pool"`, distinct from the pub/sub
/// `subscriber` SubscriberClient) variant. The two labels are the ones most
/// likely to be confused in code review, so a regression that dropped one or
/// the other would slip past a quick scan — this test pins the reader half
/// independently (the `subscriber` sibling test above guards the other half).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_metric_increments_on_stream_reader_kill() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let Some(killer) = maybe_connect().await else {
        eprintln!("SKIP: second Valkey connection not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    // Readers connect lazily, the first time they serve a subscription, and
    // arm their hooks then: connect reader 0 (standalone routing sends object 0
    // to it), then let the subscription go.
    crate::storage::mux_test_helpers::connect_reader_zero(&client).await;

    let before = read_reconnects(&metrics, "subscriber_pool");

    // The reader's read is still blocked on the server after its subscription
    // left; `Blocking::Interrupt` makes CLIENT ID interrupt it rather than
    // queue behind the XREAD BLOCK.
    let victim_id: i64 = client
        .stream_mux
        .reader_client_for_test(0)
        .expect("reader 0 holds a connection")
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
        .expect("CLIENT ID must succeed on a stream reader connection");
    let killed = kill_connection_by_id(&killer, victim_id).await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID must close exactly the stream reader victim connection"
    );

    assert!(
        wait_for_reconnect_increment(&metrics, "subscriber_pool", before, Duration::from_secs(5))
            .await,
        "valkey_reconnects_total{{pool=\"subscriber_pool\"}} must increment after CLIENT KILL + \
         fred reconnect; pins the stream reader hook block separately from the txn one so a \
         half-applied regression cannot slip through"
    );

    client.quit().await;
    killer.quit().await;
}

/// The `background_task_panics_total{task}` counter must tick when
/// `Metrics::inc_background_task_panic` is called — this is the API
/// surface the shutdown supervisor (`report_background_task_result` in
/// `src/main.rs`) uses on panicked joins. Operators rely on this counter
/// firing an alert when a background task unexpectedly dies.
///
/// The shutdown supervisor itself is a `fn` in the binary crate
/// (`main.rs`) and not callable from integration tests, so this test
/// exercises the metric-side contract: given a panicked `JoinError`, the
/// counter increments with the right label. If the implementation path
/// misnames the task label or increments a different metric, operators
/// get the wrong alert.
#[tokio::test]
async fn background_task_panic_metric_increments_by_task_label() {
    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));

    // Construct a genuine panicked JoinError by spawning a task that
    // panics, then awaiting the handle. This is the exact shape
    // `report_background_task_result` sees during shutdown when a
    // background task has panicked.
    let handle = tokio::spawn(async {
        panic!("deliberate panic for background-task-panic metric test");
    });
    let join_err = handle
        .await
        .expect_err("spawned task must yield a JoinError on panic");
    assert!(
        join_err.is_panic(),
        "JoinError from panicking task must report is_panic()"
    );

    // Pre-increment: counter for "gc" starts at zero.
    let before = read_panic_count(&metrics, "gc");
    assert_eq!(before, 0, "counter must start at zero");

    // Replicate the `report_background_task_result` production shape: on
    // a panicked JoinError, increment the counter using the task label.
    metrics.inc_background_task_panic("gc");

    let after = read_panic_count(&metrics, "gc");
    assert_eq!(
        after, 1,
        "background_task_panics_total{{task=\"gc\"}} must increment by 1"
    );

    // A second panic on a different task must only increment its own label.
    metrics.inc_background_task_panic("orphan_sweep");
    assert_eq!(
        read_panic_count(&metrics, "orphan_sweep"),
        1,
        "second task label must increment independently"
    );
    assert_eq!(
        read_panic_count(&metrics, "gc"),
        1,
        "first task label must not be double-counted"
    );

    // b.yda added a third `le:gc:leader` consumer with its own shutdown join,
    // so `report_background_task_result` gained a task label. Pinned here
    // because that label is what an operator alerts on: a reconciler that dies
    // silently leaves the ONLY thing in the service that drains the
    // leak-direction ref_count residuals switched off, with no error and no
    // other metric movement to see it by.
    metrics.inc_background_task_panic("ext_ref_reconcile");
    assert_eq!(
        read_panic_count(&metrics, "ext_ref_reconcile"),
        1,
        "the ref_count reconciler's task label must tick independently of the two sweeps it \
         shares the leader lease with — they run on three separate cadences and a shared label \
         would page for the wrong task"
    );
    for other in ["gc", "orphan_sweep"] {
        assert_eq!(
            read_panic_count(&metrics, other),
            1,
            "task=\"{other}\" must not absorb the reconciler's tick"
        );
    }
}

/// Read `live_edit_valkey_pool_in_use{pool}` from the rendered scrape
/// exposition. Pre-initialised for `txn`/`subscriber_pool`/`maintenance`/
/// `snapshot`.
fn read_pool_in_use(metrics: &Metrics, pool: &str) -> i64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_pool_in_use",
        &[("pool", pool)],
    )
    .unwrap_or(0.0) as i64
}

/// `valkey_pool_in_use{pool="txn"}` must tick up while a `txn_pool` slot
/// is held and back down when the checkout drops (b.3z7 H-obs-3 item 1).
/// Pre-rework the gauge was registered but never populated; post-rework
/// `acquire_txn` returns a `CheckedOutClient` that bundles the slot guard with
/// the `PoolUseToken` whose `Drop` decrements (b.f4f replaced the previous
/// `(guard, token)` tuple, whose drop order was inverted at eight call sites).
/// This test exercises the exact path production code takes: acquire → observe
/// inc → drop → observe dec.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn valkey_pool_in_use_txn_ticks_up_while_held_and_down_on_drop() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));

    // Install hooks — stashes the metrics handle inside ValkeyClient so
    // `acquire_txn` populates non-no-op tokens.
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    // Baseline must be zero. Pre-registration of the label (in Metrics::new)
    // means the series exists even before any increment.
    assert_eq!(
        read_pool_in_use(&metrics, "txn"),
        0,
        "pool-in-use gauge must start at 0 before any acquire"
    );

    // Acquire a txn slot. The token bundled inside the `CheckedOutClient`
    // increments the gauge in its construction path.
    let checkout = client.acquire_txn().await;
    assert_eq!(
        read_pool_in_use(&metrics, "txn"),
        1,
        "acquire_txn must increment valkey_pool_in_use{{pool=\"txn\"}}"
    );

    // A second concurrent acquire (pool_size=1 → this would block, so use
    // a separate client for the parallel slot demonstration).
    let client2 = maybe_connect()
        .await
        .expect("second connection must be reachable");
    client2
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;
    let checkout2 = client2.acquire_txn().await;
    assert_eq!(
        read_pool_in_use(&metrics, "txn"),
        2,
        "second concurrent acquire must bump the gauge to 2 (shared registry)"
    );

    // Drop the second checkout first — gauge should dec to 1.
    drop(checkout2);
    assert_eq!(
        read_pool_in_use(&metrics, "txn"),
        1,
        "dropping the second checkout must decrement the gauge to 1"
    );

    // Drop the first checkout — gauge back to 0.
    drop(checkout);
    assert_eq!(
        read_pool_in_use(&metrics, "txn"),
        0,
        "dropping the last checkout must decrement the gauge back to 0"
    );

    client.quit().await;
    client2.quit().await;
}

/// `valkey_pool_in_use{pool="subscriber_pool"}` counts stream readers that
/// have at least one stream to read. Readers update it asynchronously, so the
/// test polls.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn valkey_pool_in_use_subscriber_pool_ticks_up_and_down() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    async fn await_gauge(metrics: &Metrics, want: i64, what: &str) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let got = read_pool_in_use(metrics, "subscriber_pool");
            if got == want {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "{what}: subscriber_pool gauge stuck at {got}, want {want}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    assert_eq!(
        read_pool_in_use(&metrics, "subscriber_pool"),
        0,
        "subscriber_pool gauge starts at 0"
    );

    let admission = client
        .admit_subscriber(None)
        .await
        .expect("an idle gate must admit");
    assert_eq!(
        read_pool_in_use(&metrics, "subscriber_pool"),
        0,
        "an admission is not a reader; it must not move the gauge"
    );

    let keys = KeySchema::new(&format!("test_gauge_{}:", uuid::Uuid::new_v4()));
    let first = consume_layer_stream(
        &client,
        &keys,
        1,
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume must succeed");
    await_gauge(&metrics, 1, "a subscription activates its reader").await;
    let second = consume_layer_stream(
        &client,
        &keys,
        2,
        StreamCursor::From("0-0".into()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume must succeed");
    // One reader (`subscriber_pool_size = 1`) serves both streams. Wait until
    // its read names both keys, so the gauge below is read after the reader
    // took the second stream on.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while client.stream_mux.snapshot().readers[0].keys != 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "the reader never read both streams: {:?}",
            client.stream_mux.snapshot()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        read_pool_in_use(&metrics, "subscriber_pool"),
        1,
        "a second stream on the same reader must not count twice"
    );

    drop(first);
    drop(second);
    await_gauge(&metrics, 0, "unsubscribing every stream idles the reader").await;
    drop(admission);

    client.quit().await;
}

/// `valkey_pool_in_use{pool="maintenance"}` regression test (b.mfn). The
/// maintenance pool is fred's multiplexed `Pool`, so `acquire_maintenance`
/// is synchronous (no per-connection mutex) and the token tracks **logical**
/// maintenance use. The contract still mirrors `acquire_txn`: while at least
/// one borrower is alive, the
/// gauge is non-zero; when every paired token has dropped, the gauge
/// returns to zero. Pre-b.mfn the label did not exist on the gauge at all,
/// so operators had no saturation signal on the GC + orphan-sweep pool.
///
/// Nested borrows (e.g. `sweep_orphaned_objects` →
/// `cleanup_orphaned_object`) deliberately stack — the gauge counts each
/// outstanding token, so a sweep that descends into a per-object cleanup
/// shows up as 2.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn valkey_pool_in_use_maintenance_ticks_up_and_down() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    assert_eq!(
        read_pool_in_use(&metrics, "maintenance"),
        0,
        "maintenance gauge must start at 0; pre-b.mfn the label did not exist at all"
    );

    let (_pool_ref, token) = client.acquire_maintenance();
    assert_eq!(
        read_pool_in_use(&metrics, "maintenance"),
        1,
        "acquire_maintenance must increment valkey_pool_in_use{{pool=\"maintenance\"}}"
    );

    // Nested borrow — mirrors `sweep_orphaned_objects` calling into
    // `cleanup_orphaned_object` on the same client.
    let (_pool_ref2, token2) = client.acquire_maintenance();
    assert_eq!(
        read_pool_in_use(&metrics, "maintenance"),
        2,
        "nested acquire_maintenance must stack — gauge reflects logical \
         maintenance use, not exclusive-pool slot count"
    );

    drop(token2);
    assert_eq!(
        read_pool_in_use(&metrics, "maintenance"),
        1,
        "dropping the inner token must decrement the gauge back to 1"
    );

    drop(token);
    assert_eq!(
        read_pool_in_use(&metrics, "maintenance"),
        0,
        "dropping the last token must decrement the gauge back to 0"
    );

    client.quit().await;
}

/// `Metrics::new` must pre-register `pool="maintenance"` on the
/// `valkey_pool_in_use` gauge so dashboards can plot it as a zero-line on
/// fresh pods that haven't run a sweep yet (b.mfn). Pre-b.mfn the label
/// only appeared on first `acquire_maintenance`, breaking the
/// pre-init-zero-line contract that `txn` and `subscriber_pool` already
/// honored.
#[tokio::test]
async fn valkey_pool_in_use_maintenance_label_is_pre_registered() {
    let metrics = crate::common::metrics_scrape_only("test");

    let labels = label_value_set(&render(&metrics), "live_edit_valkey_pool_in_use", "pool");

    assert!(
        labels.contains("maintenance"),
        "pool=\"maintenance\" must be pre-registered (b.mfn); got labels {labels:?}"
    );
    // Sanity: the two pre-existing labels still pre-register too.
    assert!(labels.contains("txn"), "pool=\"txn\" still required");
    assert!(
        labels.contains("subscriber_pool"),
        "pool=\"subscriber_pool\" still required"
    );
}

/// `valkey_pool_in_use{pool="snapshot"}` counts snapshot reads in flight on the
/// multiplexed snapshot lane: `acquire_snapshot` pairs the lane with a token,
/// borrows stack, and every dropped token gives its count back. Same logical-use
/// shape as the maintenance pool, under its own label.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn valkey_pool_in_use_snapshot_ticks_up_and_down() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    assert_eq!(read_pool_in_use(&metrics, "snapshot"), 0);

    let (lane, token) = client.acquire_snapshot();
    assert!(
        std::ptr::eq(lane, &client.snapshot_pool),
        "acquire_snapshot must hand out the snapshot lane, not another pool"
    );
    assert_eq!(
        read_pool_in_use(&metrics, "snapshot"),
        1,
        "acquire_snapshot must increment valkey_pool_in_use{{pool=\"snapshot\"}}"
    );

    let (_lane2, token2) = client.acquire_snapshot();
    assert_eq!(
        read_pool_in_use(&metrics, "snapshot"),
        2,
        "concurrent snapshot reads stack — the gauge counts reads in flight, not connections"
    );
    assert_eq!(
        read_pool_in_use(&metrics, "maintenance"),
        0,
        "a snapshot read must not be counted against the maintenance pool"
    );

    drop(token2);
    assert_eq!(read_pool_in_use(&metrics, "snapshot"), 1);
    drop(token);
    assert_eq!(
        read_pool_in_use(&metrics, "snapshot"),
        0,
        "dropping the last token must return the gauge to 0"
    );

    client.quit().await;
}

/// `Metrics::new` pre-registers `pool="snapshot"` on the `valkey_pool_in_use`
/// gauge at zero, so the lane plots as a zero-line on a pod that has served no
/// snapshot read yet, like the other gauged pools.
#[tokio::test]
async fn valkey_pool_in_use_snapshot_label_is_pre_registered() {
    let metrics = crate::common::metrics_scrape_only("test");

    assert_eq!(
        sample_value(
            &render(&metrics),
            "live_edit_valkey_pool_in_use",
            &[("pool", "snapshot")],
        ),
        Some(0.0),
        "pool=\"snapshot\" must be pre-registered at zero on valkey_pool_in_use"
    );
}

/// When no `install_observability_hooks` call happens (e.g. startup
/// PINGs or tests that don't wire metrics), acquires return a no-op
/// token whose Drop does nothing. The gauge stays at its
/// pre-initialised zero and the acquire path never panics from a null
/// metrics handle.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn valkey_pool_in_use_without_metrics_hooks_installed_is_noop() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    // Deliberately do NOT call install_observability_hooks — the client's
    // internal metrics slot stays None, so acquire_txn returns a no-op
    // token.

    let checkout = client.acquire_txn().await;
    // There is no metrics registry to observe, but acquire + drop must
    // complete without panicking.
    drop(checkout);

    client.quit().await;
}

/// The per-pool label set on `valkey_reconnects_total` must be bounded to the
/// six pools the service opens (`VALKEY_POOLS`). A regression that
/// accidentally introduced attacker-influenced labels (e.g. a per-client-id
/// label) would explode cardinality the same way the `endpoint` label did
/// pre-H-obs-1. This test guards the shape.
#[tokio::test]
async fn reconnect_metric_uses_bounded_pool_label_set() {
    let metrics = crate::common::metrics_scrape_only("test");

    // After `Metrics::new`, the counter is pre-initialised with every valid
    // pool label so zero-scrape dashboards work on a fresh pod.
    let pool_labels = label_value_set(
        &render(&metrics),
        "live_edit_valkey_reconnects_total",
        "pool",
    );

    // b.avd item 7: pre-init now covers every pool whose reconnect/error
    // events are reported by `install_observability_hooks` — namely the
    // pub/sub `subscriber`, the multiplexed `data` pool, the `txn`
    // ExclusivePool, the stream multiplexer's reader connections
    // (`subscriber_pool`), the `maintenance` pool and the `snapshot` lane.
    // Pre-fix only `subscriber` + `data` were pre-inited, so the other pools'
    // series materialised only on the first reconnect/error event and broke
    // `rate()` continuity for any operator alert.
    let expected: std::collections::HashSet<String> = [
        "subscriber".to_string(),
        "data".to_string(),
        "txn".to_string(),
        "subscriber_pool".to_string(),
        "maintenance".to_string(),
        "snapshot".to_string(),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        pool_labels, expected,
        "valkey_reconnects_total must pre-initialise exactly the bounded label set \
         {{'subscriber', 'data', 'txn', 'subscriber_pool', 'maintenance', 'snapshot'}} \
         (b.avd item 7); got {pool_labels:?}"
    );
}

/// `valkey_errors_total` is pre-initialised on the same six pools as
/// `valkey_reconnects_total`, and only with its one service-owned `kind`,
/// `command_timeout` (b.21r). The fred-named kinds (`to_str()` of fred's
/// `ErrorKind`) are unbounded from this crate's point of view and are not
/// seeded, so on a fresh pod the whole family is exactly six
/// `command_timeout` cells at zero.
#[tokio::test]
async fn valkey_errors_metric_pre_initialises_command_timeout_on_every_pool() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let pool_labels = label_value_set(&text, "live_edit_valkey_errors_total", "pool");
    let kind_labels = label_value_set(&text, "live_edit_valkey_errors_total", "kind");
    let expected_pools: std::collections::HashSet<String> =
        live_edit::observability::metrics::VALKEY_POOLS
            .iter()
            .map(|p| (*p).to_string())
            .collect();
    assert_eq!(
        pool_labels, expected_pools,
        "valkey_errors_total must be pre-initialised on exactly the pools the service opens; got \
         {pool_labels:?}"
    );
    assert_eq!(
        kind_labels,
        std::iter::once("command_timeout".to_string()).collect(),
        "only the service-owned `command_timeout` kind is seeded; got {kind_labels:?}"
    );
    for pool in live_edit::observability::metrics::VALKEY_POOLS {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_valkey_errors_total",
                &[("pool", pool), ("kind", "command_timeout")],
            ),
            Some(0.0),
            "pool=\"{pool}\" must start at zero"
        );
    }
}

/// b.s5y: `live_edit_checkpoint_lease_total` must pre-initialise all three
/// `outcome` values at zero.
///
/// A checkpoint replica in steady state never loses its lease, so without the
/// zero-seed the `lost` and `refresh_error` series would not exist until the
/// first incident — and a Prometheus `rate(...[5m])` alert on a series that
/// materialises at the moment of the incident has no derivative to fire on. The
/// bounded label set is also the guard: `outcome` is a pinned three-value
/// vocabulary, never a passthrough of anything caller-supplied.
#[tokio::test]
async fn checkpoint_lease_total_pre_initialises_its_bounded_outcome_set() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let outcomes = label_value_set(&text, "live_edit_checkpoint_lease_total", "outcome");
    let expected: std::collections::HashSet<String> = ["renewed", "lost", "refresh_error"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(
        outcomes, expected,
        "live_edit_checkpoint_lease_total must pre-initialise exactly \
         {{'renewed', 'lost', 'refresh_error'}} (b.s5y); got {outcomes:?}"
    );

    for outcome in ["renewed", "lost", "refresh_error"] {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_checkpoint_lease_total",
                &[("outcome", outcome), ("service", "test")],
            ),
            Some(0.0),
            "outcome=\"{outcome}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// b.443 / b.hka (a): `live_edit_checkpoint_session_publish_superseded_total`
/// must pre-initialise all four `result` values at zero.
///
/// The alertable cells are `result="exhausted"` — a repair loop that gave up,
/// leaving a durable session checkpoint stale — and `result="stale_incarnation"`
/// — the `le:{sid}:ckpt` coordination key was lost or re-created under a build
/// that was already in flight (b.hka (a)). A healthy fleet never emits either,
/// so without the zero-seed the series would materialise at the exact
/// moment of the incident, and a Prometheus `rate(...[5m])` alert on a
/// series that did not exist a scrape ago has no derivative to fire on.
/// `write` and `delete` are seeded for the opposite reason: a steady trickle
/// there is the protocol working, and an operator needs the baseline to
/// judge a spike against.
///
/// The `result` vocabulary is also the guard — a pinned four-value set
/// (`PendingOp::kind()`'s two spellings plus the literals `"stale_incarnation"`
/// and `"exhausted"`), never a passthrough of anything caller-supplied. It is
/// asserted against `CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS` rather than
/// a re-spelled literal list, so the exported constant and the zero-seed pass
/// cannot drift apart silently.
#[tokio::test]
async fn checkpoint_session_publish_superseded_pre_initialises_its_bounded_result_set() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let results = label_value_set(
        &text,
        "live_edit_checkpoint_session_publish_superseded_total",
        "result",
    );
    let expected: std::collections::HashSet<String> =
        live_edit::observability::metrics::CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS
            .iter()
            .map(|r| (*r).to_string())
            .collect();
    assert_eq!(
        results, expected,
        "live_edit_checkpoint_session_publish_superseded_total must pre-initialise exactly the \
         CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS vocabulary (b.443, b.hka (a)); got \
         {results:?}"
    );
    assert!(
        results.contains("stale_incarnation"),
        "b.hka (a): the keyspace-loss refusal has its own cell and must NOT be folded into \
         `write` — `write` means the ordering protocol caught a genuine race between two live \
         builds, `stale_incarnation` means the coordination key itself was lost, which is a \
         keyspace-integrity event an operator alerts on separately; got {results:?}"
    );

    for result in live_edit::observability::metrics::CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_checkpoint_session_publish_superseded_total",
                &[("result", result)],
            ),
            Some(0.0),
            "result=\"{result}\" must read 0 on a fresh registry, not be absent — an alert on \
             `exhausted` or `stale_incarnation` cannot fire on a series that only appears at \
             incident time"
        );
    }
}

/// b.ji8: `live_edit_storage_watch_unwatch_failures_total` must pre-initialise
/// both `site` values at zero.
///
/// Every cell here is alertable and none of them is ever emitted by a healthy
/// replica: a non-zero value means a `txn`-pool connection was released still
/// carrying WATCH state the service believes it cleared, which punishes the
/// *next* borrower with a spurious EXEC=nil (an in-process transaction abort and
/// retry) or, on a proxied cluster,
/// a CROSSSLOT naming a key that request never touched. A series that
/// materialises at the moment of the incident gives a Prometheus
/// `increase(...[5m])` alert nothing to fire on, so the zero-seed is what makes
/// the alert expressible at all.
///
/// The two-value `site` vocabulary is the other half of the guard: `explicit`
/// and `drop` are a pinned pair (recoverable vs. terminal — `Drop` still gets a
/// retry after an explicit failure, but nothing follows a drop-site failure),
/// never a passthrough of anything caller-supplied.
#[tokio::test]
async fn storage_watch_unwatch_failures_pre_initialises_its_bounded_site_set() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let sites = label_value_set(
        &text,
        "live_edit_storage_watch_unwatch_failures_total",
        "site",
    );
    let expected: std::collections::HashSet<String> =
        ["explicit", "drop"].into_iter().map(String::from).collect();
    assert_eq!(
        sites, expected,
        "live_edit_storage_watch_unwatch_failures_total must pre-initialise exactly \
         {{'explicit', 'drop'}} (b.ji8); got {sites:?}"
    );

    for site in ["explicit", "drop"] {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_storage_watch_unwatch_failures_total",
                &[("site", site)],
            ),
            Some(0.0),
            "site=\"{site}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// b.7td: `live_edit_valkey_txn_dirty_checkouts_total` must pre-initialise all
/// five `reason` values at zero.
///
/// Every cell is alertable and — with the one carve-out below — a healthy
/// replica emits none of them: each means a `txn`-pool connection was checked
/// out carrying leaked transaction state — an open `MULTI`, a reply stream
/// offset against its command stream, or a wedged socket — and had to be
/// recycled before the borrower could use it. This is the *first-order*
/// detector for the fault b.7td was filed on; before it, the only evidence was
/// fred's second-order `Failed to refresh credentials` warning, which named the
/// credential refresh rather than the leak and only appeared on the passwordless
/// auth modes. A series that materialises at the moment of the incident gives a
/// Prometheus `increase(...[5m])` alert nothing to fire on, so the zero-seed is
/// what makes the alert expressible at all.
///
/// The five-value `reason` vocabulary is the other half of the guard: it is a
/// pinned set naming the five distinguishable verdicts, never a passthrough of
/// anything caller-supplied. The moot arm (`Canceled` / `IO`, or fred's routing
/// miss) — the connection was already gone, so no residue survived it — is
/// deliberately absent, on the
/// same reasoning as the sibling UNWATCH counter: counting it would page on
/// routine failover and proxy-recycle churn.
///
/// `topology_reply` (b.289 sub-task 3) is the fifth, and the one cell **expected
/// to be non-zero**: a `-MOVED` / `-LOADING` / `-READONLY`-shaped error frame in
/// a probe slot that fred's router did not intercept ticks it during any
/// resharding, failover or replica promotion. The zero-seed matters for the same
/// reason regardless — an operator's alert on it is a *sustained-rate* alert,
/// which needs a series that exists before the reshard starts.
///
/// # Why the expected set is restated rather than read from `TXN_DIRTY_REASONS`
///
/// Two reasons, and the second is the load-bearing one. First, mechanically:
/// `TXN_DIRTY_REASONS` is `pub(crate)` in `observability::metrics` and an
/// integration test is a separate crate, so it cannot be imported here at all.
/// Second, and why it should stay restated even if that changed: this fence
/// exists to catch a cell added to (or renamed in) that const **without a
/// deliberate decision**, and `Metrics::new` seeds the registry *from* the const.
/// Comparing the registry against the const would compare the const with itself
/// — green for any edit, including the one this test is here to stop. The
/// duplication is the assertion.
#[tokio::test]
async fn valkey_txn_dirty_checkouts_pre_initialises_its_bounded_reason_set() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let reasons = label_value_set(
        &text,
        "live_edit_valkey_txn_dirty_checkouts_total",
        "reason",
    );
    let expected: std::collections::HashSet<String> = [
        "leaked_multi",
        "unexpected_reply",
        "probe_timeout",
        "probe_error",
        "topology_reply",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(
        reasons, expected,
        "live_edit_valkey_txn_dirty_checkouts_total must pre-initialise exactly \
         {{'leaked_multi', 'unexpected_reply', 'probe_timeout', 'probe_error', \
         'topology_reply'}} (b.7td; 'topology_reply' added by b.289 sub-task 3); \
         got {reasons:?}"
    );

    for reason in [
        "leaked_multi",
        "unexpected_reply",
        "probe_timeout",
        "probe_error",
        "topology_reply",
    ] {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_valkey_txn_dirty_checkouts_total",
                &[("reason", reason)],
            ),
            Some(0.0),
            "reason=\"{reason}\" must read 0 on a fresh registry, not be absent — an alert on \
             a series that only appears at incident time has no derivative to fire on"
        );
    }
}

/// b.7td: `live_edit_valkey_txn_repair_ineffective_total` must pre-initialise
/// both state-proof `reason` values at zero.
///
/// This counter is the breaker's non-silent record: after several consecutive
/// state-proof verdicts have each survived a full connection rebuild, the
/// service stops rebuilding and hands the slot to the borrower unrepaired. The
/// accompanying `error!` is rate-limited to one in eight so it cannot flood a
/// pipeline where every checkout in the pool is producing the verdict and
/// `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS` (128) re-enters the checkout per
/// retry attempt — which leaves this counter as the only continuous signal an
/// operator's alert can fire on between escalations. A series that materialises
/// at the moment of the incident gives `increase(...[5m])` nothing to fire on.
///
/// **The two-value vocabulary is the other half of the guard, and it is a
/// strict subset of its sibling's five.** Only `leaked_multi` and
/// `unexpected_reply` can reach this breaker: those verdicts rest on a reply
/// that was actually received, so a rebuilt connection answering identically is
/// evidence the *classifier* (or the endpoint's reply rendering) is the problem
/// rather than leaked transaction state. The three unknown-state verdicts —
/// `probe_timeout` / `probe_error` / `topology_reply` — do not: the first two
/// rest on the absence of a reply, which a slow backing store produces just as
/// readily, and `topology_reply` (b.289 sub-task 3) rests on a reply that proves
/// what the *node* would not do rather than what this connection's reply stream
/// is doing. All three are bounded by a differently-shaped mechanism that never
/// stops repairing (a re-arming streak gate). Seeding all five here would
/// advertise three cells nothing can ever write, and a dashboard reading a flat
/// `probe_timeout` line would take it as evidence rather than as an
/// impossibility.
///
/// `topology_reply`'s absence is load-bearing beyond tidiness: a routine
/// resharding produces that verdict on many checkouts at once, and if it could
/// reach this breaker an operator-initiated topology change would latch a
/// breaker whose escalation `error!` accuses the classifier.
///
/// A non-zero rate here is a **different investigation** from a non-zero rate on
/// `live_edit_valkey_txn_dirty_checkouts_total`: that one says the probe found
/// slots unfit, this one says rebuilding them changed nothing.
#[tokio::test]
async fn valkey_txn_repair_ineffective_pre_initialises_its_state_proof_reason_set() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let reasons = label_value_set(
        &text,
        "live_edit_valkey_txn_repair_ineffective_total",
        "reason",
    );
    let expected: std::collections::HashSet<String> = ["leaked_multi", "unexpected_reply"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(
        reasons, expected,
        "live_edit_valkey_txn_repair_ineffective_total must pre-initialise exactly \
         {{'leaked_multi', 'unexpected_reply'}} — the state-proof subset of the dirty-checkout \
         reasons, and no more: the unknown-state verdicts have their own re-arming rate limit and \
         can never reach this breaker (b.7td); got {reasons:?}"
    );

    for reason in ["leaked_multi", "unexpected_reply"] {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_valkey_txn_repair_ineffective_total",
                &[("reason", reason)],
            ),
            Some(0.0),
            "reason=\"{reason}\" must read 0 on a fresh registry, not be absent — the escalation \
             ERROR is rate-limited to one in eight, so between escalations this counter is the \
             only thing an alert has, and a series that only appears at incident time has no \
             derivative to fire on"
        );
    }
}

/// b.bhb: `live_edit_storage_watch_voided_total` must pre-initialise the full
/// `path` × `disposition` cross product at zero.
///
/// A `WATCH` dies with its connection, so any reconnect between `WATCH` and
/// `EXEC` turns an optimistic-concurrency commit into an unconditional one.
/// This counter is what the three defence layers report, and
/// `disposition="committed_unwatched"` is the alertable one: it means the
/// server confirmed a commit ran on a connection that never held the `WATCH`,
/// so a duplicate session name (`create_session`) or a commit torn between
/// materialized state and the live stream (`update_layer`) may have landed. A
/// series that materialises at the moment of the incident gives
/// `increase(...[5m])` nothing to fire on, which is what makes the zero-seed
/// the thing that makes the alert expressible at all.
///
/// **The cross product is seeded deliberately, including the three unreachable
/// cells.** `{path="update_layer", disposition="compensated"}` can never tick —
/// a structural commit cannot be undone from the client, which is why the layer
/// path answers a proven-unwatched commit with the retirement re-check instead
/// — and neither can `{path="restore_residue_wipe", disposition="compensated"}`,
/// because an `UNLINK` cannot be undone at all, nor
/// `{path="restore_write_object", disposition="compensated"}`, because a landed
/// restore materialization cannot be un-merged: once two replicas' rebuilds are
/// in one keyspace, which member came from which generation is not recoverable.
/// Seeding them keeps one alert expression valid for every path. That is
/// the opposite of the `txn_repair_ineffective` decision one metric above, and
/// the difference is the point: there the unseeded cells were *impossible
/// verdicts* a flat line would be read as evidence for; here the missing cell
/// would only force the alert to be written four times.
///
/// **Why the path vocabulary is four (b.f4t, b.h4m).** Both additions are on the
/// DR restore path and neither is redundant with the other:
///
/// * `restore_residue_wipe` — the restore's pre-write residue wipe
///   (`src/checkpoint/residue.rs`), the only watched site whose committed effect
///   is an irreversible delete: a `committed_unwatched` there means the
///   `obj:hdr` fence that was supposed to stop the wipe from removing a peer
///   replica's freshly landed restore did not actually run.
/// * `restore_write_object` — the restore's own materialization
///   (`LayerStore::write_restored_object`), added by b.h4m, and the site whose
///   `committed_unwatched` cell the counter's own rustdoc names as the
///   correctness failure behind `live_edit_checkpoint_restore_fenced_total`: the
///   layer may now be a MIXTURE of two backup generations, with no client error
///   and no other counter moving.
///
/// Both must be seeded for exactly the same reason as the original two — the
/// alert needs a series before the first incident, not at it.
///
/// **Why the vocabulary is five and not three.** The two `*_unavailable`
/// dispositions report that a defence layer could not be *armed* on this
/// backing store rather than that a `WATCH` was voided: `witness_unavailable`
/// when the store will not answer a stable `CLIENT ID` from inside a `MULTI`
/// (L3 disarmed), `fingerprint_unavailable` when fred holds no `CLIENT ID` for
/// the connection at all (L2 skipped). Both are near-zero or near-total by
/// construction — a property of the store, not an event — and both are the
/// cells a DR / dev-AMR run reads to characterise a proxied managed store. They
/// must be seeded for the same reason as the other three, and they must be
/// *distinguishable* from them: an operator paging on `committed_unwatched`
/// must not have a disarm folded into that cell, and an operator reading a flat
/// `committed_unwatched` must be able to tell "no incidents" from "the layer
/// that would have detected one was never armed".
#[tokio::test]
async fn storage_watch_voided_pre_initialises_its_full_path_disposition_matrix() {
    const PATHS: [&str; 4] = [
        "update_layer",
        "create_session",
        "restore_residue_wipe",
        "restore_write_object",
    ];
    const DISPOSITIONS: [&str; 5] = [
        "aborted_pre_exec",
        "committed_unwatched",
        "compensated",
        "witness_unavailable",
        "fingerprint_unavailable",
    ];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let paths = label_value_set(&text, "live_edit_storage_watch_voided_total", "path");
    let expected_paths: std::collections::HashSet<String> =
        PATHS.into_iter().map(String::from).collect();
    assert_eq!(
        paths, expected_paths,
        "live_edit_storage_watch_voided_total must pre-initialise exactly \
         {{'update_layer', 'create_session', 'restore_residue_wipe', 'restore_write_object'}} — \
         the four WATCHed sites, never a passthrough of anything caller-supplied (b.bhb, b.f4t, \
         b.h4m); got {paths:?}"
    );

    let dispositions =
        label_value_set(&text, "live_edit_storage_watch_voided_total", "disposition");
    let expected_dispositions: std::collections::HashSet<String> =
        DISPOSITIONS.into_iter().map(String::from).collect();
    assert_eq!(
        dispositions, expected_dispositions,
        "the disposition vocabulary is a pinned set of five: a caught window (nothing \
         written), an undone one, the residual that reached a client, and the two \
         `*_unavailable` cells saying a defence layer could not be armed on this store. Only \
         `committed_unwatched` is alertable, so an unexpected sixth value would silently \
         widen what an operator is paging on — and a MISSING `*_unavailable` cell would hide \
         a store on which the detection is off entirely; got {dispositions:?}"
    );

    for path in PATHS {
        for disposition in DISPOSITIONS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_storage_watch_voided_total",
                    &[("path", path), ("disposition", disposition)],
                ),
                Some(0.0),
                "path=\"{path}\", disposition=\"{disposition}\" must read 0 on a fresh \
                 registry, not be absent"
            );
        }
    }
}

/// b.459 / b.zvd: `live_edit_keyspace_type_error_total` must pre-initialise
/// every reachable cell at zero — and only those.
///
/// Every cell here is alertable and none of them is ever emitted by a healthy
/// replica: a non-zero value means a key on the `le:` keyspace holds a foreign
/// value, which nothing in this service can produce — it retypes no key it owns.
/// The layer stays unwritable until an operator deletes the key named in the
/// accompanying WARN, so this is the counter that turns "writes to one layer
/// silently stopped" into a page. A series that materialises at the moment of
/// the incident gives a Prometheus `increase(...[5m])` alert nothing to fire on,
/// which is what makes the zero-seed the thing that makes the alert expressible.
///
/// The bounded label set is the other half of the guard. `backend` is a pinned
/// two-value vocabulary — both commit paths refuse the same keyspace on the same
/// message (b.459 parity), and the label is what tells an operator which one
/// refused. `site` is a pinned two-value vocabulary too, since b.zvd gave
/// `LayerStore::write_object` the same pre-`MULTI` probe: `update` for
/// `update_layer`'s commit paths, `create` for the CreateLayer materialization and
/// the DR restore that share `write_object`. The two create entry points do NOT
/// split the label — they ride the refusal log line's `entry_point` field
/// instead, which is what keeps this label set bounded.
///
/// The `lua`/`update` cell is the update script's, and the script refuses in
/// both of its modes, so that cell alone is split by `mode` (`fast` |
/// `structural`) and the seeded set is four series: `lua`/`update` once per
/// mode, plus `txn`/`update` and `txn`/`create`, which no script mode raises
/// and which therefore carry no `mode` label. There is no `lua` create backend
/// (`create` is served exclusively by `write_object`'s MULTI/EXEC), so a
/// `{backend="lua", site="create"}` cell would publish a series that can never
/// move. The exact-set comparison excludes it for the same reason it requires
/// the present cells: a zero-seeded series that cannot move trains operators
/// to ignore the metric. The cell set is pinned by
/// [`assert_update_script_error_cells_seeded`].
#[tokio::test]
async fn keyspace_type_error_total_pre_initialises_its_bounded_backend_set() {
    assert_update_script_error_cells_seeded("live_edit_keyspace_type_error_total");
}

/// b.ph9: `live_edit_stream_id_collision_total` must pre-initialise the same
/// four series as `live_edit_keyspace_type_error_total`, at zero, and no
/// other. A non-zero value means a layer's stream already holds the ID a
/// commit was about to write, which leaves that layer unwritable until the
/// stream is cleared; the counter's value rests on a zero baseline an
/// `increase(...)` alert can fire from. Its `lua`/`update` cell is split by the
/// update script's `mode`, since both modes reach the collision arm; the
/// `txn`/`update` probe and `write_object`'s `txn`/`create` guard carry no
/// `mode`; and there is no `lua` create path.
#[tokio::test]
async fn stream_id_collision_total_pre_initialises_its_backend_site_and_mode_cells() {
    assert_update_script_error_cells_seeded("live_edit_stream_id_collision_total");
}

/// The `{backend="lua", site="update"}` cell of
/// `live_edit_keyspace_type_error_total` exists only per `mode`, so the
/// two-label `inc_keyspace_type_error` must refuse it: a tick there would open
/// a `mode`-less series no seeding covers and no per-mode reading sums. The
/// `debug_assert!` is the only runtime guard, so this exists only in builds
/// that keep it.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the lua/update keyspace_type_error cell carries `mode`")]
fn inc_keyspace_type_error_refuses_the_lua_update_cell_without_a_mode() {
    let metrics = crate::common::metrics_scrape_only("test");
    metrics.inc_keyspace_type_error("lua", "update");
}

/// The `live_edit_stream_id_collision_total` twin of
/// [`inc_keyspace_type_error_refuses_the_lua_update_cell_without_a_mode`].
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the lua/update stream_id_collision cell carries `mode`")]
fn inc_stream_id_collision_refuses_the_lua_update_cell_without_a_mode() {
    let metrics = crate::common::metrics_scrape_only("test");
    metrics.inc_stream_id_collision("lua", "update");
}

/// Every series of `metric` in rendered exposition `text`, each as its whole
/// label set (name → value), so a test pins which labels each seeded cell
/// carries and not only which values one label takes. Label values here are
/// plain identifiers, so the body is split on `,` and `=` without unescaping.
fn series_label_sets(
    text: &str,
    metric: &str,
) -> std::collections::BTreeSet<std::collections::BTreeMap<String, String>> {
    text.lines()
        .filter_map(|line| line.strip_prefix(metric)?.strip_prefix('{'))
        .map(|rest| {
            let body = &rest[..rest
                .find('}')
                .expect("a labelled series closes its label set")];
            body.split(',')
                .filter(|pair| !pair.is_empty())
                .map(|pair| {
                    let (name, value) = pair.split_once('=').expect("a label is name=\"value\"");
                    (name.to_owned(), value.trim_matches('"').to_owned())
                })
                .collect()
        })
        .collect()
}

/// The cells of `metric`, one of the two counters the update script shares
/// with the other commit paths, are exactly `{backend="lua", site="update",
/// mode}` for each update-script mode, `{backend="txn", site="update"}` and
/// `{backend="txn", site="create"}` — the `txn` cells with no `mode` label —
/// and every one reads 0 on a fresh registry. The `mode` vocabulary is
/// restated and compared with `UPDATE_SCRIPT_MODES`, so a rename (which moves
/// the const and the seeding together) fails here.
fn assert_update_script_error_cells_seeded(metric: &str) {
    const MODES: [&str; 2] = ["fast", "structural"];
    let mut vocabulary = live_edit::observability::metrics::UPDATE_SCRIPT_MODES.to_vec();
    vocabulary.sort_unstable();
    assert_eq!(
        vocabulary, MODES,
        "the `mode` vocabulary of {metric} is operator-facing; renaming, adding or dropping a \
         value is a deliberate change that must update this restated list too"
    );
    let cell = |labels: &[(&str, &str)]| -> std::collections::BTreeMap<String, String> {
        labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    };
    let expected: std::collections::BTreeSet<_> = MODES
        .iter()
        .map(|mode| cell(&[("backend", "lua"), ("site", "update"), ("mode", mode)]))
        .chain([
            cell(&[("backend", "txn"), ("site", "update")]),
            cell(&[("backend", "txn"), ("site", "create")]),
        ])
        .collect();

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    assert_eq!(
        series_label_sets(&text, metric),
        expected,
        "{metric} must pre-initialise exactly the update script's per-mode lua/update cells and \
         the two mode-less txn cells — a lua/update cell without `mode`, a `mode` on a txn cell, \
         or a lua/create cell is a series no producer ticks"
    );
    for labels in &expected {
        let labels: Vec<(&str, &str)> = labels
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            sample_value(&text, metric, &labels),
            Some(0.0),
            "{metric}{labels:?} must read 0 on a fresh registry, not be absent — an alert on a \
             series that did not exist a scrape ago has no derivative to fire on"
        );
    }
}

/// b.75i: `live_edit_update_layer_commit_already_landed_total` must
/// pre-initialise exactly its two `backend` cells, `lua` and `txn`, at zero.
///
/// Both backends run the per-call commit-token check, so both cells are
/// reachable and both must exist before the first match — a cell that appears
/// only at incident time gives `increase(...)` nothing to compare against. The
/// rendered set is checked against `UPDATE_LAYER_BACKENDS` (catches a seeding
/// loop that drifted from the vocabulary) and that const against a list
/// restated here (catches a rename, which moves the const and the seeding
/// together and would otherwise go unnoticed).
#[tokio::test]
async fn update_layer_commit_already_landed_total_pre_initialises_both_backend_cells() {
    const METRIC: &str = "live_edit_update_layer_commit_already_landed_total";
    const BACKENDS: [&str; 2] = ["lua", "txn"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut vocabulary: Vec<&str> =
        live_edit::observability::metrics::UPDATE_LAYER_BACKENDS.to_vec();
    vocabulary.sort_unstable();
    assert_eq!(
        vocabulary, BACKENDS,
        "the `backend` vocabulary is operator-facing; renaming a value is a breaking change"
    );

    let rendered = label_value_set(&text, METRIC, "backend");
    let expected: std::collections::HashSet<String> =
        BACKENDS.into_iter().map(String::from).collect();
    assert_eq!(
        rendered, expected,
        "{METRIC} must pre-initialise exactly {{'lua', 'txn'}}; got {rendered:?}"
    );
    for backend in BACKENDS {
        assert_eq!(
            sample_value(&text, METRIC, &[("backend", backend)]),
            Some(0.0),
            "backend=\"{backend}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// `live_edit_update_layer_lost_reply_release_lookup_total` must
/// pre-initialise exactly its three `outcome` cells — `enqueued`, `absent`
/// and `failed` — at zero.
///
/// Every outcome is reachable on any store (a lookup finds an entry, finds
/// none, or cannot read it), and an operator alerting on `failed`, or
/// watching `enqueued` as the lost-reply rate, needs each series to exist
/// before its first tick. The rendered set is checked against
/// `LOST_REPLY_RELEASE_LOOKUP_OUTCOMES` (a seeding loop that drifted from the
/// vocabulary) and that const against a list restated here (a rename, which
/// moves the const and the seeding together).
#[tokio::test]
async fn update_layer_lost_reply_release_lookup_total_pre_initialises_its_three_outcome_cells() {
    const METRIC: &str = "live_edit_update_layer_lost_reply_release_lookup_total";
    const OUTCOMES: [&str; 3] = ["absent", "enqueued", "failed"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut vocabulary: Vec<&str> =
        live_edit::observability::metrics::LOST_REPLY_RELEASE_LOOKUP_OUTCOMES.to_vec();
    vocabulary.sort_unstable();
    assert_eq!(
        vocabulary, OUTCOMES,
        "the `outcome` vocabulary is operator-facing; renaming a value is a breaking change"
    );

    let rendered = label_value_set(&text, METRIC, "outcome");
    let expected: std::collections::HashSet<String> =
        OUTCOMES.into_iter().map(String::from).collect();
    assert_eq!(
        rendered, expected,
        "{METRIC} must pre-initialise exactly {{'enqueued', 'absent', 'failed'}}; got \
         {rendered:?}"
    );
    for outcome in OUTCOMES {
        assert_eq!(
            sample_value(&text, METRIC, &[("outcome", outcome)]),
            Some(0.0),
            "outcome=\"{outcome}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// The `path` vocabulary of every `UpdateLayer` commit metric, restated rather
/// than re-read from `UPDATE_LAYER_PATHS` so a rename fails the tests below.
/// Five values although this build emits only `fast` and `watch`: the
/// counters are seeded over every path so the series the later commit paths
/// emit exist, at zero, before those paths ship.
const UPDATE_LAYER_PATHS_RESTATED: [&str; 5] =
    ["fast", "watch", "structural", "structural_full", "reset"];

/// Assert `vocabulary` (a `metrics` const) equals `restated` as a set, and that
/// `label` of the rendered `metric` carries exactly that set.
fn assert_label_vocabulary(
    text: &str,
    metric: &str,
    label: &str,
    vocabulary: &[&str],
    restated: &[&str],
) {
    let mut sorted_vocabulary = vocabulary.to_vec();
    sorted_vocabulary.sort_unstable();
    let mut sorted_restated = restated.to_vec();
    sorted_restated.sort_unstable();
    assert_eq!(
        sorted_vocabulary, sorted_restated,
        "the `{label}` vocabulary of {metric} is operator-facing — dashboards and the b.2wo \
         analysis scripts select on it — so renaming, adding or dropping a value is a \
         deliberate change that must update this restated list too"
    );
    let rendered = label_value_set(text, metric, label);
    let expected: std::collections::HashSet<String> =
        restated.iter().map(|v| v.to_string()).collect();
    assert_eq!(
        rendered, expected,
        "{metric} must pre-initialise exactly the `{label}` values {restated:?}; got {rendered:?}"
    );
}

/// `live_edit_update_layer_retries_total{path,outcome}` is zero-seeded over its
/// whole `path` × `outcome` cross product (Spec SDD SR-2.1), including the
/// three paths this build cannot emit. The vocabulary consts are pinned against
/// restated lists (which catches a rename — the const and the seeding loop
/// would move together), the rendered label sets against the same lists
/// (which catches a seeding loop that drifted from its const), and every cell
/// is read at zero (which catches a loop that seeds only part of the product).
#[tokio::test]
async fn update_layer_retries_total_pre_initialises_its_full_path_outcome_matrix() {
    const METRIC: &str = "live_edit_update_layer_retries_total";
    const OUTCOMES: [&str; 2] = ["committed", "exhausted"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    assert_label_vocabulary(
        &text,
        METRIC,
        "path",
        live_edit::observability::metrics::UPDATE_LAYER_PATHS,
        &UPDATE_LAYER_PATHS_RESTATED,
    );
    assert_label_vocabulary(
        &text,
        METRIC,
        "outcome",
        live_edit::observability::metrics::UPDATE_LAYER_RETRY_OUTCOMES,
        &OUTCOMES,
    );
    for path in UPDATE_LAYER_PATHS_RESTATED {
        for outcome in OUTCOMES {
            assert_eq!(
                sample_value(&text, METRIC, &[("path", path), ("outcome", outcome)]),
                Some(0.0),
                "path=\"{path}\", outcome=\"{outcome}\" must read 0 on a fresh registry, not \
                 be absent — a per-path baseline compares series that must exist before the \
                 first retry on that path"
            );
        }
    }
}

/// `live_edit_update_layer_conflicts_total{path,kind}` is zero-seeded over its
/// whole `path` × `kind` cross product (Spec SDD SR-2.3) — 25 cells, of which
/// this build can tick only `{path="watch"}` × `{watch_conflict,
/// watch_read_moved}`. Pinned the same three ways as the retry counter above.
#[tokio::test]
async fn update_layer_conflicts_total_pre_initialises_its_full_path_kind_matrix() {
    const METRIC: &str = "live_edit_update_layer_conflicts_total";
    const KINDS: [&str; 5] = [
        "watch_conflict",
        "watch_read_moved",
        "seq_fence",
        "cache_diverged",
        "snapshot_fence_wait",
    ];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    assert_label_vocabulary(
        &text,
        METRIC,
        "path",
        live_edit::observability::metrics::UPDATE_LAYER_PATHS,
        &UPDATE_LAYER_PATHS_RESTATED,
    );
    assert_label_vocabulary(
        &text,
        METRIC,
        "kind",
        live_edit::observability::metrics::UPDATE_LAYER_CONFLICT_KINDS,
        &KINDS,
    );
    for path in UPDATE_LAYER_PATHS_RESTATED {
        for kind in KINDS {
            assert_eq!(
                sample_value(&text, METRIC, &[("path", path), ("kind", kind)]),
                Some(0.0),
                "path=\"{path}\", kind=\"{kind}\" must read 0 on a fresh registry, not be absent"
            );
        }
    }
}

/// b.26e Stage 2: `live_edit_update_layer_recharges_total{path}` is zero-seeded
/// over the closed `path` vocabulary, so a `rate(...) > 0` over a structural
/// commit's waits for a larger upload charge evaluates a present series from
/// the first scrape; and its sibling histogram
/// `live_edit_upload_object_state_bytes{path}`, observed once per call that
/// returns its own commit, renders no series before its first observation (a
/// histogram cannot be seeded without a fake sample). The `path` vocabulary is
/// pinned against the restated list, as on the counters above.
#[tokio::test]
async fn update_layer_recharges_total_pre_initialises_every_path_and_the_object_state_histogram_none(
) {
    use live_edit::observability::metrics::test_util::has_metric;
    const METRIC: &str = "live_edit_update_layer_recharges_total";

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    assert_label_vocabulary(
        &text,
        METRIC,
        "path",
        live_edit::observability::metrics::UPDATE_LAYER_PATHS,
        &UPDATE_LAYER_PATHS_RESTATED,
    );
    for path in UPDATE_LAYER_PATHS_RESTATED {
        assert_eq!(
            sample_value(&text, METRIC, &[("path", path)]),
            Some(0.0),
            "path=\"{path}\" must read 0 on a fresh registry, not be absent"
        );
    }
    assert_eq!(
        text.lines()
            .filter(|l| l.starts_with(&format!("{METRIC}{{")))
            .count(),
        UPDATE_LAYER_PATHS_RESTATED.len(),
        "one cell per path and no other label"
    );
    assert!(
        !has_metric(&text, "live_edit_upload_object_state_bytes"),
        "live_edit_upload_object_state_bytes must have no series on a fresh registry — a \
         pre-registered histogram carries a fake sample in its count and buckets"
    );
}

/// The two `UpdateLayer` commit histograms are NOT pre-registered (Spec SDD
/// SR-2.1, project SDD §17.1): an OTel histogram cannot be seeded without
/// recording a sample, and a fake sample would corrupt every quantile. So a
/// fresh registry renders neither family. The commit-duration `outcome`
/// vocabulary, which no seeding exposes, is pinned here against a restated
/// list instead.
#[tokio::test]
async fn update_layer_commit_histograms_render_no_series_before_their_first_observation() {
    use live_edit::observability::metrics::test_util::has_metric;

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    for family in [
        "live_edit_update_layer_attempts_histogram",
        "live_edit_update_layer_commit_duration_seconds",
    ] {
        assert!(
            !has_metric(&text, family),
            "{family} must have no series on a fresh registry — a pre-registered histogram \
             carries a fake sample in its count and buckets"
        );
    }

    let mut outcomes = live_edit::observability::metrics::UPDATE_LAYER_COMMIT_OUTCOMES.to_vec();
    outcomes.sort_unstable();
    assert_eq!(
        outcomes,
        ["error", "ok"],
        "the commit-duration `outcome` vocabulary is operator-facing; changing it is a \
         deliberate change that must update this list"
    );
}

/// b.v6d T18: `live_edit_operation_id_match_total` must pre-initialise all four
/// `(rpc, outcome)` cells at zero, and no others.
///
/// Every cell is reachable — both RPCs answer a matched re-send
/// (`already_committed`) and refuse a mismatched one (`conflict`) — and the
/// `conflict` rate is a client-bug alert, so each series must exist before its
/// first tick. Both vocabularies are restated here as well as read from their
/// constants, the same rename guard as the test above.
#[tokio::test]
async fn operation_id_match_total_pre_initialises_all_four_cells() {
    const METRIC: &str = "live_edit_operation_id_match_total";
    const RPCS: [&str; 2] = ["create_layer", "update_layer"];
    const OUTCOMES: [&str; 2] = ["already_committed", "conflict"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut rpcs: Vec<&str> = live_edit::observability::metrics::OPERATION_ID_RPCS.to_vec();
    rpcs.sort_unstable();
    assert_eq!(rpcs, RPCS, "the `rpc` vocabulary is operator-facing");
    let mut outcomes: Vec<&str> = live_edit::observability::metrics::OPERATION_ID_OUTCOMES.to_vec();
    outcomes.sort_unstable();
    assert_eq!(
        outcomes, OUTCOMES,
        "the `outcome` vocabulary is operator-facing"
    );

    for (label, values) in [("rpc", RPCS), ("outcome", OUTCOMES)] {
        let rendered = label_value_set(&text, METRIC, label);
        let expected: std::collections::HashSet<String> =
            values.into_iter().map(String::from).collect();
        assert_eq!(
            rendered, expected,
            "{METRIC} must render exactly these `{label}` values; got {rendered:?}"
        );
    }
    for rpc in RPCS {
        for outcome in OUTCOMES {
            assert_eq!(
                sample_value(&text, METRIC, &[("rpc", rpc), ("outcome", outcome)]),
                Some(0.0),
                "rpc=\"{rpc}\", outcome=\"{outcome}\" must read 0 on a fresh registry, not be \
                 absent"
            );
        }
    }
}

/// b.v6d T31: `live_edit_update_layer_operation_overtaken_total` must
/// pre-initialise exactly its two `backend` cells, `lua` and `txn`, at zero.
///
/// Both commit paths run the start-record check, so both cells are reachable,
/// and the counter is the only aggregate sign that operation-id calls are
/// being refused because their `client_id`'s commit record moved while they
/// were in flight — a cell that appears only at
/// incident time gives `increase(...)` nothing to compare against. Checked
/// against `UPDATE_LAYER_BACKENDS` and against a restated list, as the
/// already-landed counter above is.
#[tokio::test]
async fn update_layer_operation_overtaken_total_pre_initialises_both_backend_cells() {
    const METRIC: &str = "live_edit_update_layer_operation_overtaken_total";
    const BACKENDS: [&str; 2] = ["lua", "txn"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut vocabulary: Vec<&str> =
        live_edit::observability::metrics::UPDATE_LAYER_BACKENDS.to_vec();
    vocabulary.sort_unstable();
    assert_eq!(
        vocabulary, BACKENDS,
        "the `backend` vocabulary is operator-facing; renaming a value is a breaking change"
    );

    let rendered = label_value_set(&text, METRIC, "backend");
    let expected: std::collections::HashSet<String> =
        BACKENDS.into_iter().map(String::from).collect();
    assert_eq!(
        rendered, expected,
        "{METRIC} must pre-initialise exactly {{'lua', 'txn'}}; got {rendered:?}"
    );
    for backend in BACKENDS {
        assert_eq!(
            sample_value(&text, METRIC, &[("backend", backend)]),
            Some(0.0),
            "backend=\"{backend}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// b.v6d T39: `live_edit_valkey_router_restarts_total` must pre-initialise all
/// six `pool` cells at zero, and no others.
///
/// Its baseline is zero and any rise means a fred router task ended — the
/// operator alert is `increase(...) > 0`, which needs every series to exist
/// before its first tick. The vocabulary is `VALKEY_POOLS`, shared with
/// `valkey_reconnects_total` / `valkey_errors_total`; it is restated here too,
/// so a rename that moves the const and the seeding together still fails.
#[tokio::test]
async fn valkey_router_restarts_total_pre_initialises_all_six_pool_cells() {
    const METRIC: &str = "live_edit_valkey_router_restarts_total";
    const POOLS: [&str; 6] = [
        "data",
        "maintenance",
        "snapshot",
        "subscriber",
        "subscriber_pool",
        "txn",
    ];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut vocabulary: Vec<&str> = live_edit::observability::metrics::VALKEY_POOLS.to_vec();
    vocabulary.sort_unstable();
    assert_eq!(
        vocabulary, POOLS,
        "the `pool` vocabulary is operator-facing; renaming a value is a breaking change"
    );

    let rendered = label_value_set(&text, METRIC, "pool");
    let expected: std::collections::HashSet<String> = POOLS.into_iter().map(String::from).collect();
    assert_eq!(
        rendered, expected,
        "{METRIC} must pre-initialise exactly the six pools; got {rendered:?}"
    );
    for pool in POOLS {
        assert_eq!(
            sample_value(&text, METRIC, &[("pool", pool)]),
            Some(0.0),
            "pool=\"{pool}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// b.y3m Epic 3 Task 3 (Spec SDD SR-8.3): the writer-side fence counters are
/// present at zero from the first scrape — `live_edit_update_layer_fence_intent_total`
/// with exactly its two `result` cells, `registered` and `refused_fenced`, and
/// the unlabelled `live_edit_update_layer_fence_timeout_total` — so an
/// `increase(...)` over either has a series to compare against before the
/// first fenced commit. The rendered `result` set is checked against
/// `UPDATE_LAYER_FENCE_INTENT_RESULTS` and that const against a list restated
/// here, as for the already-landed counter above. (The fence-wait histogram
/// has no zero observation to seed and is not covered here.)
#[tokio::test]
async fn update_layer_fence_intent_and_timeout_totals_pre_initialise_at_zero() {
    const INTENT: &str = "live_edit_update_layer_fence_intent_total";
    const TIMEOUT: &str = "live_edit_update_layer_fence_timeout_total";
    const RESULTS: [&str; 2] = ["refused_fenced", "registered"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut vocabulary: Vec<&str> =
        live_edit::observability::metrics::UPDATE_LAYER_FENCE_INTENT_RESULTS.to_vec();
    vocabulary.sort_unstable();
    assert_eq!(
        vocabulary, RESULTS,
        "the `result` vocabulary is operator-facing; renaming a value is a breaking change"
    );

    let rendered = label_value_set(&text, INTENT, "result");
    let expected: std::collections::HashSet<String> =
        RESULTS.into_iter().map(String::from).collect();
    assert_eq!(
        rendered, expected,
        "{INTENT} must pre-initialise exactly {{'registered', 'refused_fenced'}}; got \
         {rendered:?}"
    );
    for result in RESULTS {
        assert_eq!(
            sample_value(&text, INTENT, &[("result", result)]),
            Some(0.0),
            "result=\"{result}\" must read 0 on a fresh registry, not be absent"
        );
    }
    assert_eq!(
        sample_value(&text, TIMEOUT, &[]),
        Some(0.0),
        "{TIMEOUT} must read 0 on a fresh registry, not be absent"
    );
}

/// b.y3m Epic 3 Task 4 (Spec SDD SR-8.5): the catch-up memory budget's series
/// are present at zero from the first scrape, before any catch-up reserves
/// memory — the unlabelled `live_edit_catchup_build_budget_reserved_bytes`,
/// `live_edit_catchup_build_budget_overcommit_bytes` and
/// `live_edit_catchup_builds_in_progress` gauges, and
/// `live_edit_catchup_build_admission_total` with exactly its four `result`
/// cells. The rendered set is checked against `CATCHUP_BUILD_ADMISSION_RESULTS`
/// and that const against a list restated here, so a rename is caught. (The
/// wait histogram has no zero observation to seed and is not covered here.)
#[tokio::test]
async fn catchup_build_budget_gauges_and_admission_total_pre_initialise_at_zero() {
    const ADMISSIONS: &str = "live_edit_catchup_build_admission_total";
    const GAUGES: [&str; 3] = [
        "live_edit_catchup_build_budget_reserved_bytes",
        "live_edit_catchup_build_budget_overcommit_bytes",
        "live_edit_catchup_builds_in_progress",
    ];
    const RESULTS: [&str; 4] = ["cancelled", "immediate", "refused_timeout", "waited"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut vocabulary: Vec<&str> =
        live_edit::observability::metrics::CATCHUP_BUILD_ADMISSION_RESULTS.to_vec();
    vocabulary.sort_unstable();
    assert_eq!(
        vocabulary, RESULTS,
        "the `result` vocabulary is operator-facing; renaming a value is a breaking change"
    );

    let rendered = label_value_set(&text, ADMISSIONS, "result");
    let expected: std::collections::HashSet<String> =
        RESULTS.into_iter().map(String::from).collect();
    assert_eq!(
        rendered, expected,
        "{ADMISSIONS} must pre-initialise exactly {RESULTS:?}; got {rendered:?}"
    );
    for result in RESULTS {
        assert_eq!(
            sample_value(&text, ADMISSIONS, &[("result", result)]),
            Some(0.0),
            "result=\"{result}\" must read 0 on a fresh registry, not be absent"
        );
    }
    for gauge in GAUGES {
        assert_eq!(
            sample_value(&text, gauge, &[]),
            Some(0.0),
            "{gauge} must read 0 on a fresh registry, not be absent"
        );
    }
}

/// b.y3m Epic 3 Task 6 (Spec SDD SR-8.4): `live_edit_catchup_snapshot_flight_total`
/// is present from the first scrape with exactly its two `role` cells at zero,
/// so the read hit-rate recipe (`joined_read` ÷ (`leader` + `joined_read`))
/// divides present series before any catch-up shares a read. The rendered set
/// is checked against `CATCHUP_SNAPSHOT_FLIGHT_ROLES` and that const against a
/// list restated here, so a rename — or an unemittable `joined_build` cell —
/// is caught.
#[tokio::test]
async fn catchup_snapshot_flight_total_pre_initialises_both_roles_at_zero() {
    const FLIGHTS: &str = "live_edit_catchup_snapshot_flight_total";
    const ROLES: [&str; 2] = ["joined_read", "leader"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut vocabulary: Vec<&str> =
        live_edit::observability::metrics::CATCHUP_SNAPSHOT_FLIGHT_ROLES.to_vec();
    vocabulary.sort_unstable();
    assert_eq!(
        vocabulary, ROLES,
        "the `role` vocabulary is operator-facing; renaming a value is a breaking change"
    );

    let rendered = label_value_set(&text, FLIGHTS, "role");
    let expected: std::collections::HashSet<String> = ROLES.into_iter().map(String::from).collect();
    assert_eq!(
        rendered, expected,
        "{FLIGHTS} must pre-initialise exactly {ROLES:?}; got {rendered:?}"
    );
    for role in ROLES {
        assert_eq!(
            sample_value(&text, FLIGHTS, &[("role", role)]),
            Some(0.0),
            "role=\"{role}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// b.azh: `live_edit_stream_mux_reader_connects_total` pre-initialises its
/// whole `cause` × `result` matrix at zero — exactly `bind` / `retarget` ×
/// `ok` / `failed` — and `live_edit_stream_mux_readers_connected` reads 0 on a
/// fresh registry. Readers connect lazily, so a replica that has served no
/// subscription shows the series at zero rather than absent, and a
/// `failed` rate alert has a series to fire on from its first tick. The label
/// values are restated here, so a rename that moves the const and the seeding
/// together still fails.
#[tokio::test]
async fn stream_mux_reader_connect_series_pre_initialise_at_zero() {
    const CONNECTS: &str = "live_edit_stream_mux_reader_connects_total";
    const CONNECTED: &str = "live_edit_stream_mux_readers_connected";
    const CAUSES: [&str; 2] = ["bind", "retarget"];
    const RESULTS: [&str; 2] = ["failed", "ok"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut causes = live_edit::observability::metrics::READER_CONNECT_CAUSES.to_vec();
    causes.sort_unstable();
    let mut results = live_edit::observability::metrics::READER_CONNECT_RESULTS.to_vec();
    results.sort_unstable();
    assert_eq!(
        (causes, results),
        (CAUSES.to_vec(), RESULTS.to_vec()),
        "the label vocabularies are operator-facing; renaming a value is a breaking change"
    );
    let expected = |values: [&str; 2]| -> std::collections::HashSet<String> {
        values.into_iter().map(String::from).collect()
    };
    assert_eq!(label_value_set(&text, CONNECTS, "cause"), expected(CAUSES));
    assert_eq!(
        label_value_set(&text, CONNECTS, "result"),
        expected(RESULTS)
    );
    for cause in CAUSES {
        for result in RESULTS {
            assert_eq!(
                sample_value(&text, CONNECTS, &[("cause", cause), ("result", result)]),
                Some(0.0),
                "cause=\"{cause}\", result=\"{result}\" must read 0 on a fresh registry"
            );
        }
    }
    assert_eq!(
        sample_value(&text, CONNECTED, &[]),
        Some(0.0),
        "{CONNECTED} must read 0 on a fresh registry, not be absent"
    );
}

/// b.19i: `live_edit_stream_mux_reauths_total` pre-initialises exactly its
/// `result` cells — `ok` and `failed` — at zero. Under `static` auth no reader
/// ever re-AUTHs, and under the passwordless modes a `failed` is rare, so
/// without seeding the series would not exist for a `failed` rate alert to
/// fire on. The label values are restated here, so a rename that moves the
/// const and the seeding together still fails.
#[tokio::test]
async fn stream_mux_reauths_total_pre_initialises_ok_and_failed_at_zero() {
    const REAUTHS: &str = "live_edit_stream_mux_reauths_total";
    const RESULTS: [&str; 2] = ["failed", "ok"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let mut results = live_edit::observability::metrics::REAUTH_RESULTS.to_vec();
    results.sort_unstable();
    assert_eq!(
        results,
        RESULTS.to_vec(),
        "the label vocabulary is operator-facing; renaming a value is a breaking change"
    );
    assert_eq!(
        label_value_set(&text, REAUTHS, "result"),
        RESULTS.into_iter().map(String::from).collect(),
        "exactly the two result cells are seeded"
    );
    for result in RESULTS {
        assert_eq!(
            sample_value(&text, REAUTHS, &[("result", result)]),
            Some(0.0),
            "result=\"{result}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// b.atj: `live_edit_snapshot_read_type_error_total` must pre-initialise every
/// `key_family` cell at zero — exactly the twelve registered families (b.atj's
/// ten, plus `children_index`, the `obj:chs` read only `read_object_snapshot`
/// issues — its atomic read and its unfenced fan-out, on a marked object — and
/// `ext_index`, the `obj:ext` reads of external-value clean-up's size read and
/// reference walk, which no snapshot read touches, b.fsz MR-D4), no
/// thirteenth, and no `service` label.
///
/// The read-side twin of the test above, and zero-seeding matters here for the
/// same reason plus one more. The shared reason: the shipped alert recipe is an
/// or-join of the two counters at `increase(...) > 0`, and a Prometheus
/// `increase` over a series that did not exist a scrape ago has no derivative to
/// fire on — so a cell that materialises at incident time is a cell the alert
/// cannot page on, which is precisely the flat-line failure b.atj exists to end.
/// The extra reason: this counter is the ONLY signal for a wedge that can take
/// down up to four snapshot consumers of the object-state reads at once
/// (subscribe / catch-up reconstruction and the checkpoint assembler through
/// `read_object_snapshot`'s atomic script, or above the atomic-read limits its
/// fenced read; structural `UpdateLayer` through its unfenced fan-out) and the
/// external-value release of a deleted or replaced layer's clean-up — the
/// last of which is a background task whose only other evidence is a log line
/// nobody is watching. The authoritative list, with which keys stop which
/// consumers, is on the `snapshot_read_type_error_total` field doc in
/// `src/observability/metrics.rs`.
///
/// **Both directions of the set equality are asserted, and against two different
/// sources.** The rendered set is compared with
/// `metrics::SNAPSHOT_READ_KEY_FAMILIES`, which catches a pre-init loop that
/// drifted from the vocabulary it is supposed to seed; and that `const` is
/// compared with a list restated here, which is what catches a *rename*. The
/// rendered-vs-const comparison alone cannot: both sides are fed from the same
/// array, so renaming an entry moves them together and the test agrees with
/// itself. The label values are the operator-facing contract — they appear in
/// the alert's `by (key_family)` breakdown and in every dashboard built on it —
/// so a rename is a breaking change that has to be loud somewhere, and this is
/// the somewhere. (`tests/integration/storage/b_atj_tests.rs` reads the `const`
/// directly, so this test is the only place the two spellings are forced to
/// agree.)
///
/// The `service` label is asserted ABSENT, matching the commit-side sibling. A
/// per-replica breakdown would be actively harmful on this metric: the condition
/// is a property of the shared keyspace, not of the replica that happened to
/// read it, so splitting by replica fragments one incident across N series and
/// invites an alert that fires N times for one bad key.
#[tokio::test]
async fn snapshot_read_type_error_total_pre_initialises_its_bounded_key_family_set() {
    /// The twelve `key_family` cells, restated. See the doc-comment above for
    /// why this is a literal rather than a re-read of the `const` under test.
    const KEY_FAMILIES: [&str; 12] = [
        "header",
        "sections_index",
        "deleted_sections_index",
        "section",
        "deleted_section",
        "children",
        "children_index",
        "fields_index",
        "time_samples_index",
        "field",
        "time_sample",
        "ext_index",
    ];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let registered: std::collections::HashSet<String> =
        live_edit::observability::metrics::SNAPSHOT_READ_KEY_FAMILIES
            .iter()
            .map(|family| (*family).to_string())
            .collect();
    let expected: std::collections::HashSet<String> =
        KEY_FAMILIES.into_iter().map(String::from).collect();
    assert_eq!(
        registered, expected,
        "metrics::SNAPSHOT_READ_KEY_FAMILIES must be exactly the ten families named in the b.atj \
         contract plus the atomic read's `children_index` (Spec SDD SR-1.4) and external-value \
         clean-up's `ext_index` (b.fsz MR-D4). A difference \
         here is a RENAME (or an added/removed family), which silently \
         breaks every dashboard and alert grouping by key_family and which the \
         rendered-vs-const comparison below cannot see, because both of its sides read this same \
         array; got {registered:?}"
    );

    let families = label_value_set(
        &text,
        "live_edit_snapshot_read_type_error_total",
        "key_family",
    );
    assert_eq!(
        families, expected,
        "live_edit_snapshot_read_type_error_total must pre-initialise exactly the registered \
         key_family vocabulary (b.atj). A MISSING cell is invisible until its first refusal, \
         which is the absent-series hole the zero-seed exists to close; an EXTRA one is either a \
         family emitted from a `SnapshotKeyFamily` variant that never reached \
         SNAPSHOT_READ_KEY_FAMILIES or an unbounded string on the label; got {families:?}"
    );

    for family in KEY_FAMILIES {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_snapshot_read_type_error_total",
                &[("key_family", family)],
            ),
            Some(0.0),
            "key_family=\"{family}\" must read 0 on a fresh registry, not be absent — \"no \
             snapshot read has ever been refused by a wrong-typed key\" is an assertion an \
             operator needs to be able to make from a FIRST scrape, and it is the baseline the \
             `increase(...) > 0` recipe measures against"
        );
    }

    assert!(
        label_value_set(&text, "live_edit_snapshot_read_type_error_total", "service").is_empty(),
        "this counter carries no `service` label, matching its commit-side sibling \
         live_edit_keyspace_type_error_total. The condition it reports is a property of the \
         shared keyspace rather than of the replica that read it, so a per-replica split would \
         fragment one incident across N series and page N times for one bad key"
    );
}

/// b.y3m Epics 2–3 (Spec SDD SR-8.1, §17.1's seeding rule):
/// `live_edit_snapshot_read_total` must pre-initialise exactly the cells the
/// built read paths can emit, at zero, and no others — fifty-four series:
///
/// - `path="atomic"`: the five atomic consumers (catch-up, legacy per-commit,
///   checkpoint, the structural cache's cold start, `structural_cache`, and its
///   sampled audit, `structural_cache_audit`) × the five atomic outcomes
///   (twenty-five);
/// - `path="unfenced"`: the `structural_update` and ext-GC consumers × the
///   four outcomes `ok`, `not_found`, `transient_error` and `permanent_error`
///   (eight), plus `structural_update`'s `too_large` (nine) — the fan-out has
///   no ceiling, but a structural `UpdateLayer`'s read stops before its next
///   window when its upload charge cannot grow to cover it (b.26e); the ext-GC
///   consumer, external-value clean-up's reference walk (b.fsz MR-D4), carries
///   no allowance, so it has no `too_large` cell, and no unfenced read has a
///   `fence_broken` one, because it holds no fence;
/// - `path="fenced"`: the four consumers whose reads take the fence — the
///   catch-up consumer, through its shared fenced-read flight; the checkpoint
///   consumer, through the assembler's fenced read of a layer above the
///   atomic-read limits; the structural cache's cold start
///   (`structural_cache`), through its own hook, for a layer above the plain
///   limits or predicted over the plain byte limit; and the cache's sampled
///   audit (`structural_cache_audit`), through the same hook, for a layer
///   above the plain limits with its above-ceiling switch on — × the five
///   fenced outcomes (twenty) — no `too_large`, because the fenced read has no
///   ceiling. A rollback replay's per-commit read is never fenced (it hands
///   its catch-up to a flight, counted under `catchup`).
///
/// Zero-seeding matters for the reason it does on the b.atj counter above: the
/// `too_large` rate is the signal that layers are taking their above-ceiling
/// path, and the fenced `fence_broken` and error cells are what an alert on a
/// misbehaving fence evaluates; an alert over a cell that appears only at its
/// first tick has no `increase` to evaluate before then. Seeding NO other cell
/// matters just as much: the `structural_update` and ext-GC consumers never
/// take the atomic path, the catch-up, checkpoint and both structural-cache
/// consumers never take the unfenced one, and no consumer but catch-up,
/// checkpoint and the structural cache's cold start and audit takes the fenced
/// one, so a
/// seeded cell for any of them would be a series that can only ever read
/// zero — a dashboard row claiming coverage of a path that does not run.
///
/// The vocabularies are restated as literals and compared with the exported
/// `const`s, for the rename-guard reason the b.atj test gives; the rendered
/// set is compared with the restated one. The three histograms record at the
/// first read, so a fresh registry carries none of them.
#[tokio::test]
async fn snapshot_read_total_pre_initialises_exactly_the_atomic_unfenced_and_fenced_cells() {
    use live_edit::observability::metrics as m;

    const CONSUMERS: [&str; 7] = [
        "catchup",
        "catchup_legacy_per_commit",
        "checkpoint",
        "structural_update",
        "ext_gc",
        "structural_cache",
        "structural_cache_audit",
    ];
    const PATHS: [&str; 3] = ["atomic", "fenced", "unfenced"];
    const OUTCOMES: [&str; 6] = [
        "ok",
        "not_found",
        "too_large",
        "transient_error",
        "permanent_error",
        "fence_broken",
    ];
    const ATOMIC_CONSUMERS: [&str; 5] = [
        "catchup",
        "catchup_legacy_per_commit",
        "checkpoint",
        "structural_cache",
        "structural_cache_audit",
    ];
    const ATOMIC_OUTCOMES: [&str; 5] = [
        "ok",
        "not_found",
        "too_large",
        "transient_error",
        "permanent_error",
    ];
    const UNFENCED_CONSUMERS: [&str; 2] = ["structural_update", "ext_gc"];
    const UNFENCED_OUTCOMES: [&str; 5] = [
        "ok",
        "not_found",
        "too_large",
        "transient_error",
        "permanent_error",
    ];
    /// The one unfenced consumer whose read can stop `too_large`: the
    /// structural `UpdateLayer`'s, which carries its upload charge's allowance.
    const UNFENCED_TOO_LARGE_CONSUMER: &str = "structural_update";
    const FENCED_CONSUMERS: [&str; 4] = [
        "catchup",
        "checkpoint",
        "structural_cache",
        "structural_cache_audit",
    ];
    const FENCED_OUTCOMES: [&str; 5] = [
        "ok",
        "not_found",
        "transient_error",
        "permanent_error",
        "fence_broken",
    ];
    let set = |values: &[&str]| -> std::collections::HashSet<String> {
        values.iter().map(|v| v.to_string()).collect()
    };
    for (exported, restated, what) in [
        (m::SNAPSHOT_READ_CONSUMERS, &CONSUMERS[..], "consumer"),
        (m::SNAPSHOT_READ_PATHS, &PATHS[..], "path"),
        (m::SNAPSHOT_READ_OUTCOMES, &OUTCOMES[..], "outcome"),
        (
            m::SNAPSHOT_READ_ATOMIC_CONSUMERS,
            &ATOMIC_CONSUMERS[..],
            "atomic consumer",
        ),
        (
            m::SNAPSHOT_READ_ATOMIC_OUTCOMES,
            &ATOMIC_OUTCOMES[..],
            "atomic outcome",
        ),
        (
            m::SNAPSHOT_READ_UNFENCED_CONSUMERS,
            &UNFENCED_CONSUMERS[..],
            "unfenced consumer",
        ),
        (
            m::SNAPSHOT_READ_UNFENCED_OUTCOMES,
            &UNFENCED_OUTCOMES[..],
            "unfenced outcome",
        ),
        (
            m::SNAPSHOT_READ_FENCED_CONSUMERS,
            &FENCED_CONSUMERS[..],
            "fenced consumer",
        ),
        (
            m::SNAPSHOT_READ_FENCED_OUTCOMES,
            &FENCED_OUTCOMES[..],
            "fenced outcome",
        ),
    ] {
        assert_eq!(
            set(exported),
            set(restated),
            "the {what} vocabulary of the live_edit_snapshot_read_* series is an operator-facing \
             contract; a difference here is a rename or an added/removed value"
        );
    }

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);
    const METRIC: &str = "live_edit_snapshot_read_total";

    assert_eq!(label_value_set(&text, METRIC, "consumer"), set(&CONSUMERS));
    assert_eq!(label_value_set(&text, METRIC, "path"), set(&PATHS));
    assert_eq!(label_value_set(&text, METRIC, "outcome"), set(&OUTCOMES));
    for (path, consumers, outcomes) in [
        ("atomic", &ATOMIC_CONSUMERS[..], &ATOMIC_OUTCOMES[..]),
        ("unfenced", &UNFENCED_CONSUMERS[..], &UNFENCED_OUTCOMES[..]),
        ("fenced", &FENCED_CONSUMERS[..], &FENCED_OUTCOMES[..]),
    ] {
        for &consumer in consumers {
            for &outcome in outcomes {
                let cell = sample_value(
                    &text,
                    METRIC,
                    &[("consumer", consumer), ("path", path), ("outcome", outcome)],
                );
                if path == "unfenced"
                    && outcome == "too_large"
                    && consumer != UNFENCED_TOO_LARGE_CONSUMER
                {
                    assert_eq!(
                        cell, None,
                        "consumer={consumer} carries no allowance, so its unfenced read never \
                         stops too_large and the cell must not be seeded"
                    );
                    continue;
                }
                assert_eq!(
                    cell,
                    Some(0.0),
                    "consumer={consumer} outcome={outcome} on path={path} must read 0 on a fresh \
                     registry, not be absent"
                );
            }
        }
    }
    let series = text
        .lines()
        .filter(|l| l.starts_with(&format!("{METRIC}{{")))
        .count();
    assert_eq!(
        series, 54,
        "exactly the twenty-five atomic, nine unfenced (structural_update's too_large included) \
         and twenty fenced (catch-up, checkpoint, structural_cache and structural_cache_audit) \
         cells may be seeded — a fifty-fifth is an atomic structural_update/ext-GC cell, an \
         unfenced catch-up/checkpoint/structural_cache/structural_cache_audit cell, an unfenced \
         ext-GC or fenced too_large, a fence_broken cell off the fenced path, or a fenced cell of \
         a consumer that does not run the fenced read"
    );
    assert!(
        label_value_set(&text, METRIC, "service").is_empty(),
        "no `service` label, matching the sibling snapshot-read counters"
    );
    for histogram in [
        "live_edit_snapshot_read_duration_seconds",
        "live_edit_snapshot_read_reply_bytes",
        "live_edit_snapshot_read_entries",
    ] {
        assert!(
            !text.lines().any(|l| l.starts_with(histogram)),
            "{histogram} records at the first read; a fresh registry must carry none of it"
        );
    }
}

/// b.dkt Epic 4 (structural-edits SDD SR-2.5):
/// `live_edit_structural_cold_start_reads_total` must pre-initialise every
/// `route` — `atomic`, `fenced`, `uncacheable` — at zero, so a dashboard row
/// and an `uncacheable > 0` rule evaluate a present series from the first
/// scrape, and no other cell: no `atomic_raised`, since no cold start takes
/// the raised-ceiling read any more (structural-edits C1; nor does the
/// sampled audit, which reads above the plain ceilings fenced too); and one
/// observation per route must open exactly one
/// counter cell and one duration series per route.
///
/// The vocabulary is restated as literals and compared with the exported
/// `const`, for the rename-guard reason the b.atj test gives. The duration
/// histogram records at the first cold start, so a fresh registry carries
/// none of it.
#[tokio::test]
async fn structural_cold_start_reads_total_pre_initialises_every_route_and_nothing_else() {
    use live_edit::observability::metrics as m;

    const ROUTES: [&str; 3] = ["atomic", "fenced", "uncacheable"];
    const COUNTER: &str = "live_edit_structural_cold_start_reads_total";
    const HISTOGRAM: &str = "live_edit_structural_cold_start_read_duration_seconds";
    let set = |values: &[&str]| -> std::collections::HashSet<String> {
        values.iter().map(|v| v.to_string()).collect()
    };
    let series = |text: &str, metric: &str| {
        text.lines()
            .filter(|l| l.starts_with(&format!("{metric}{{")))
            .count()
    };
    assert_eq!(
        set(m::STRUCTURAL_COLD_START_ROUTES),
        set(&ROUTES),
        "the route vocabulary of the live_edit_structural_cold_start_* series is an \
         operator-facing contract; a difference here is a rename or an added/removed value"
    );

    let metrics = crate::common::metrics_scrape_only("test");
    let fresh = render(&metrics);
    for route in ROUTES {
        assert_eq!(
            sample_value(&fresh, COUNTER, &[("route", route)]),
            Some(0.0),
            "route={route} must read 0 on a fresh registry, not be absent"
        );
    }
    assert_eq!(
        series(&fresh, COUNTER),
        ROUTES.len(),
        "exactly the three route cells may be seeded"
    );
    assert_eq!(
        sample_value(&fresh, COUNTER, &[("route", "atomic_raised")]),
        None,
        "route=atomic_raised is retired: no cold start takes the raised-ceiling read"
    );
    assert!(
        !fresh.lines().any(|l| l.starts_with(HISTOGRAM)),
        "{HISTOGRAM} records at the first cold start; a fresh registry must carry none of it"
    );

    for &route in m::STRUCTURAL_COLD_START_ROUTES {
        metrics.observe_structural_cold_start(route, std::time::Duration::from_millis(3));
    }
    let ticked = render(&metrics);
    for route in ROUTES {
        assert_eq!(
            (
                sample_value(&ticked, COUNTER, &[("route", route)]),
                sample_value(&ticked, &format!("{HISTOGRAM}_count"), &[("route", route)]),
            ),
            (Some(1.0), Some(1.0)),
            "route={route}: one observation is one tick and one duration"
        );
    }
    assert_eq!(
        (
            series(&ticked, COUNTER),
            series(&ticked, &format!("{HISTOGRAM}_count"))
        ),
        (ROUTES.len(), ROUTES.len()),
        "an observation per route opens no series outside the route vocabulary"
    );
    assert!(
        label_value_set(&ticked, COUNTER, "service").is_empty(),
        "no `service` label, matching the snapshot-read counters"
    );
}

/// b.dkt Epic 4 (structural-edits SDD SR-2.4): the structural cache's five
/// series. `live_edit_structural_cache_lookups_total` pre-initialises every
/// `outcome` and `live_edit_structural_cache_reloads_total` every `reason` at
/// zero — `diverged` and `uncacheable` included, though only the structural
/// commit's caller ticks them — and nothing else; the unlabelled
/// `live_edit_structural_cache_divergence_total` reads 0, so its `> 0` alert
/// evaluates a present series from the first scrape. The two repair
/// histograms record at the first repair, so a fresh registry carries none of
/// them; one observation lands in the buckets the dashboards read (`le=1024`
/// is the repair budget, `le=4194304` one read's byte bound).
///
/// The vocabularies are restated as literals and compared with the exported
/// `const`s, for the rename-guard reason the b.atj test gives.
#[tokio::test]
async fn structural_cache_series_pre_initialise_every_outcome_and_reason_and_nothing_else() {
    use live_edit::observability::metrics as m;

    const OUTCOMES: [&str; 4] = ["hit", "repaired", "cold", "reload"];
    const REASONS: [&str; 11] = [
        "identity",
        "regressed",
        "gap",
        "tombstone",
        "parse",
        "replay_err",
        "too_far",
        "too_large",
        "dangling_adopted",
        "diverged",
        "uncacheable",
    ];
    const LOOKUPS: &str = "live_edit_structural_cache_lookups_total";
    const RELOADS: &str = "live_edit_structural_cache_reloads_total";
    const DIVERGENCE: &str = "live_edit_structural_cache_divergence_total";
    const REPLAYED: &str = "live_edit_structural_cache_replayed_entries";
    const REPAIR_BYTES: &str = "live_edit_structural_cache_repair_bytes";
    let set = |values: &[&str]| -> std::collections::HashSet<String> {
        values.iter().map(|v| v.to_string()).collect()
    };
    let series = |text: &str, metric: &str| {
        text.lines()
            .filter(|l| {
                l.starts_with(&format!("{metric}{{")) || l.starts_with(&format!("{metric} "))
            })
            .count()
    };
    assert_eq!(
        set(m::STRUCTURAL_CACHE_LOOKUP_OUTCOMES),
        set(&OUTCOMES),
        "the outcome vocabulary of {LOOKUPS} is an operator-facing contract"
    );
    assert_eq!(
        m::STRUCTURAL_CACHE_RELOAD_REASONS,
        &REASONS,
        "the reason vocabulary of {RELOADS} is an operator-facing contract, in the spec's order"
    );

    let metrics = crate::common::metrics_scrape_only("test");
    let fresh = render(&metrics);
    for outcome in OUTCOMES {
        assert_eq!(
            sample_value(&fresh, LOOKUPS, &[("outcome", outcome)]),
            Some(0.0),
            "outcome={outcome} must read 0 on a fresh registry, not be absent"
        );
    }
    for reason in REASONS {
        assert_eq!(
            sample_value(&fresh, RELOADS, &[("reason", reason)]),
            Some(0.0),
            "reason={reason} must read 0 on a fresh registry, not be absent"
        );
    }
    assert_eq!(
        sample_value(&fresh, DIVERGENCE, &[]),
        Some(0.0),
        "{DIVERGENCE} must read 0 on a fresh registry, not be absent"
    );
    assert_eq!(
        (
            series(&fresh, LOOKUPS),
            series(&fresh, RELOADS),
            series(&fresh, DIVERGENCE)
        ),
        (OUTCOMES.len(), REASONS.len(), 1),
        "exactly the vocabulary cells, and one divergence series, may be seeded"
    );
    for histogram in [REPLAYED, REPAIR_BYTES] {
        assert!(
            !fresh.lines().any(|l| l.starts_with(histogram)),
            "{histogram} records at the first repair; a fresh registry must carry none of it"
        );
    }

    for &outcome in m::STRUCTURAL_CACHE_LOOKUP_OUTCOMES {
        metrics.inc_structural_cache_lookup(outcome);
    }
    for &reason in m::STRUCTURAL_CACHE_RELOAD_REASONS {
        metrics.inc_structural_cache_reload(reason);
    }
    metrics.inc_structural_cache_divergence();
    metrics.observe_structural_cache_repair(1_024, 4 * 1024 * 1024);
    let ticked = render(&metrics);
    for outcome in OUTCOMES {
        assert_eq!(
            sample_value(&ticked, LOOKUPS, &[("outcome", outcome)]),
            Some(1.0),
            "outcome={outcome}: one increment is one tick"
        );
    }
    for reason in REASONS {
        assert_eq!(
            sample_value(&ticked, RELOADS, &[("reason", reason)]),
            Some(1.0),
            "reason={reason}: one increment is one tick"
        );
    }
    assert_eq!(sample_value(&ticked, DIVERGENCE, &[]), Some(1.0));
    assert_eq!(
        (series(&ticked, LOOKUPS), series(&ticked, RELOADS)),
        (OUTCOMES.len(), REASONS.len()),
        "an increment per vocabulary member opens no series outside it"
    );
    for (histogram, at_bound, below_bound) in [
        (REPLAYED, "1024", "512"),
        (REPAIR_BYTES, "4194304", "1048576"),
    ] {
        let bucket =
            |le: &str| sample_value(&ticked, &format!("{histogram}_bucket"), &[("le", le)]);
        assert_eq!(
            (bucket(below_bound), bucket(at_bound)),
            (Some(0.0), Some(1.0)),
            "{histogram}: one repair at the bound lands in the le={at_bound} bucket, not below"
        );
        assert_eq!(
            sample_value(&ticked, &format!("{histogram}_count"), &[]),
            Some(1.0),
            "{histogram}: one repair is one observation"
        );
    }
    for metric in [LOOKUPS, RELOADS] {
        assert!(
            label_value_set(&ticked, metric, "service").is_empty(),
            "{metric}: no `service` label, matching the cold-start counters"
        );
    }
}

/// `live_edit_structural_cache_uncacheable_total` pre-initialises every
/// `reason` — `over_weight`, `no_chs_complete`, `reload_limit`,
/// `fenced_read_failed`, `over_deadline` — at zero, so a per-reason
/// `increase(...) > 0` rule
/// evaluates a present series from the first scrape, and no other cell: no
/// `disabled` (a disabled cache is a configuration, never a verdict, and a
/// cell this binary cannot emit is not seeded), and no
/// `above_cold_start_ceiling` (retired with the cold start's raised-ceiling
/// read: an object above the plain ceilings now takes the fenced read,
/// structural-edits C1). One increment per reason opens no series outside
/// them, and the counter carries no `service` label.
///
/// The vocabulary is restated as literals and compared with the exported
/// `const`, for the rename-guard reason the b.atj test gives.
#[tokio::test]
async fn structural_cache_uncacheable_total_pre_initialises_every_reason_and_nothing_else() {
    use live_edit::observability::metrics as m;

    const REASONS: [&str; 5] = [
        "over_weight",
        "no_chs_complete",
        "reload_limit",
        "fenced_read_failed",
        "over_deadline",
    ];
    const METRIC: &str = "live_edit_structural_cache_uncacheable_total";
    let series = |text: &str| {
        text.lines()
            .filter(|l| {
                l.starts_with(&format!("{METRIC}{{")) || l.starts_with(&format!("{METRIC} "))
            })
            .count()
    };
    assert_eq!(
        m::STRUCTURAL_CACHE_UNCACHEABLE_REASONS,
        &REASONS,
        "the reason vocabulary of {METRIC} is an operator-facing contract, in \
         UncacheableReason's order"
    );

    let metrics = crate::common::metrics_scrape_only("test");
    let fresh = render(&metrics);
    for reason in REASONS {
        assert_eq!(
            sample_value(&fresh, METRIC, &[("reason", reason)]),
            Some(0.0),
            "reason={reason} must read 0 on a fresh registry, not be absent"
        );
    }
    assert_eq!(
        series(&fresh),
        REASONS.len(),
        "exactly the five reason cells may be seeded: no `disabled`"
    );
    assert_eq!(
        sample_value(&fresh, METRIC, &[("reason", "above_cold_start_ceiling")]),
        None,
        "reason=above_cold_start_ceiling is retired with the cold start's raised-ceiling read"
    );

    for &reason in m::STRUCTURAL_CACHE_UNCACHEABLE_REASONS {
        metrics.inc_structural_cache_uncacheable(reason);
    }
    let ticked = render(&metrics);
    for reason in REASONS {
        assert_eq!(
            sample_value(&ticked, METRIC, &[("reason", reason)]),
            Some(1.0),
            "reason={reason}: one increment is one tick"
        );
    }
    assert_eq!(
        series(&ticked),
        REASONS.len(),
        "an increment per reason opens no series outside the vocabulary"
    );
    assert!(
        label_value_set(&ticked, METRIC, "service").is_empty(),
        "no `service` label, matching the other structural cache counters"
    );
}

/// `live_edit_update_layer_structural_fallbacks_total` pre-initialises every
/// `reason` — `repeat_divergence`, and the structural size gate's
/// `over_work_limit` (a plan over the work limit) and `reset_over_work_limit`
/// (a cold reset estimated over it) — at zero, so a per-reason dashboard row
/// evaluates a present series from the first scrape, and no other cell. One
/// increment per reason is one tick and opens no series outside them.
///
/// The vocabulary is restated as literals and compared with the exported
/// `const`, for the rename-guard reason the b.atj test gives.
#[tokio::test]
async fn update_layer_structural_fallbacks_total_pre_initialises_every_reason_and_nothing_else() {
    use live_edit::observability::metrics as m;

    const REASONS: [&str; 3] = [
        "repeat_divergence",
        "over_work_limit",
        "reset_over_work_limit",
    ];
    const METRIC: &str = "live_edit_update_layer_structural_fallbacks_total";
    let series = |text: &str| {
        text.lines()
            .filter(|l| {
                l.starts_with(&format!("{METRIC}{{")) || l.starts_with(&format!("{METRIC} "))
            })
            .count()
    };
    assert_eq!(
        m::UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS,
        &REASONS,
        "the reason vocabulary of {METRIC} is an operator-facing contract"
    );

    let metrics = crate::common::metrics_scrape_only("test");
    let fresh = render(&metrics);
    for reason in REASONS {
        assert_eq!(
            sample_value(&fresh, METRIC, &[("reason", reason)]),
            Some(0.0),
            "reason={reason} must read 0 on a fresh registry, not be absent"
        );
    }
    assert_eq!(
        series(&fresh),
        REASONS.len(),
        "exactly the three reason cells may be seeded"
    );

    for &reason in m::UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS {
        metrics.inc_update_layer_structural_fallback(reason);
    }
    let ticked = render(&metrics);
    for reason in REASONS {
        assert_eq!(
            sample_value(&ticked, METRIC, &[("reason", reason)]),
            Some(1.0),
            "reason={reason}: one increment is one tick"
        );
    }
    assert_eq!(
        series(&ticked),
        REASONS.len(),
        "an increment per reason opens no series outside the vocabulary"
    );
}

/// `live_edit_update_layer_structural_commit_seconds` has bucket boundaries
/// at the structural size gate's decision lines — 2 s (the slow-commit
/// WARN), 2.5 s (the hold that is the trigger to lower the work limit) and
/// 5 s (the command timeout and `lua-time-limit`) — among its others, and is
/// not pre-registered: a fresh registry carries none of it. A 2.2 s sample
/// lands at `le=2.5` and not at `le=2`, which only those two boundaries make
/// visible. The `le` values are compared as numbers, so the exporter's float
/// formatting does not matter.
#[tokio::test]
async fn update_layer_structural_commit_seconds_buckets_are_the_size_gates_decision_lines() {
    const HISTOGRAM: &str = "live_edit_update_layer_structural_commit_seconds";
    const BOUNDS: [f64; 17] = [
        0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
        8.0,
    ];
    let metrics = crate::common::metrics_scrape_only("test");
    assert!(
        !render(&metrics).lines().any(|l| l.starts_with(HISTOGRAM)),
        "{HISTOGRAM} records at the first structural commit; a fresh registry must carry none \
         of it"
    );

    metrics.observe_update_layer_structural_commit(Duration::from_millis(2_200));
    let text = render(&metrics);
    let bucket = format!("{HISTOGRAM}_bucket");
    let mut les: Vec<(f64, String)> = label_value_set(&text, &bucket, "le")
        .into_iter()
        .filter(|le| le != "+Inf")
        .map(|le| {
            let bound = le
                .parse()
                .unwrap_or_else(|_| panic!("le={le:?} is not a number"));
            (bound, le)
        })
        .collect();
    les.sort_by(|a, b| a.0.total_cmp(&b.0));
    let bounds: Vec<f64> = les.iter().map(|(bound, _)| *bound).collect();
    assert_eq!(bounds, BOUNDS, "{HISTOGRAM}'s bucket boundaries");

    let count_at = |bound: f64| {
        let (_, le) = les
            .iter()
            .find(|(b, _)| *b == bound)
            .unwrap_or_else(|| panic!("no le={bound} bucket"));
        sample_value(&text, &bucket, &[("le", le.as_str())])
    };
    assert_eq!(
        (count_at(2.0), count_at(2.5)),
        (Some(0.0), Some(1.0)),
        "a 2.2 s commit lands in the 2.5 bucket, not the 2 one"
    );
    assert_eq!(
        sample_value(&text, &format!("{HISTOGRAM}_count"), &[]),
        Some(1.0),
        "one commit is one observation"
    );
}

/// b.dkt Epic 4 Task 6 (structural-edits SDD SR-2.4, SR-3.6):
/// `live_edit_structural_cache_audit_total` pre-initialises every `outcome` —
/// `match`, `diverged`, `skipped` — at zero, so the `diverged > 0` alert
/// evaluates a present series from the first scrape, and no other cell; one
/// increment per outcome opens no series outside them.
///
/// The vocabulary is restated as literals and compared with the exported
/// `const`, for the rename-guard reason the b.atj test gives.
#[tokio::test]
async fn structural_cache_audit_total_pre_initialises_every_outcome_and_nothing_else() {
    use live_edit::observability::metrics as m;

    const OUTCOMES: [&str; 3] = ["match", "diverged", "skipped"];
    const METRIC: &str = "live_edit_structural_cache_audit_total";
    let set = |values: &[&str]| -> std::collections::HashSet<String> {
        values.iter().map(|v| v.to_string()).collect()
    };
    let series = |text: &str| {
        text.lines()
            .filter(|l| {
                l.starts_with(&format!("{METRIC}{{")) || l.starts_with(&format!("{METRIC} "))
            })
            .count()
    };
    assert_eq!(
        set(m::STRUCTURAL_CACHE_AUDIT_OUTCOMES),
        set(&OUTCOMES),
        "the outcome vocabulary of {METRIC} is an operator-facing contract"
    );

    let metrics = crate::common::metrics_scrape_only("test");
    let fresh = render(&metrics);
    for outcome in OUTCOMES {
        assert_eq!(
            sample_value(&fresh, METRIC, &[("outcome", outcome)]),
            Some(0.0),
            "outcome={outcome} must read 0 on a fresh registry, not be absent"
        );
    }
    assert_eq!(
        series(&fresh),
        OUTCOMES.len(),
        "exactly the three outcome cells may be seeded"
    );

    for &outcome in m::STRUCTURAL_CACHE_AUDIT_OUTCOMES {
        metrics.inc_structural_cache_audit(outcome);
    }
    let ticked = render(&metrics);
    for outcome in OUTCOMES {
        assert_eq!(
            sample_value(&ticked, METRIC, &[("outcome", outcome)]),
            Some(1.0),
            "outcome={outcome}: one increment is one tick"
        );
    }
    assert_eq!(
        series(&ticked),
        OUTCOMES.len(),
        "an increment per outcome opens no series outside the vocabulary"
    );
    assert!(
        label_value_set(&ticked, METRIC, "service").is_empty(),
        "no `service` label, matching the other structural cache counters"
    );
}

/// Structural-edits SDD SR-3.6 (the audit's fenced read):
/// `live_edit_structural_cache_audit_fenced_total` pre-initialises every
/// `result` — `read`, `paced`, `busy`, `failed` — at zero, so a rate of
/// fenced audit reads (each one a pause of the layer's colliding writers) and
/// a `failed` alert evaluate a present series from the first scrape, and no
/// other cell; one increment per result opens no series outside them.
///
/// The vocabulary is restated as literals and compared with the exported
/// `const`, for the rename-guard reason the b.atj test gives.
#[tokio::test]
async fn structural_cache_audit_fenced_total_pre_initialises_every_result_and_nothing_else() {
    use live_edit::observability::metrics as m;

    const RESULTS: [&str; 4] = ["read", "paced", "busy", "failed"];
    const METRIC: &str = "live_edit_structural_cache_audit_fenced_total";
    let set = |values: &[&str]| -> std::collections::HashSet<String> {
        values.iter().map(|v| v.to_string()).collect()
    };
    let series = |text: &str| {
        text.lines()
            .filter(|l| {
                l.starts_with(&format!("{METRIC}{{")) || l.starts_with(&format!("{METRIC} "))
            })
            .count()
    };
    assert_eq!(
        set(m::STRUCTURAL_CACHE_AUDIT_FENCED_RESULTS),
        set(&RESULTS),
        "the result vocabulary of {METRIC} is an operator-facing contract"
    );

    let metrics = crate::common::metrics_scrape_only("test");
    let fresh = render(&metrics);
    for result in RESULTS {
        assert_eq!(
            sample_value(&fresh, METRIC, &[("result", result)]),
            Some(0.0),
            "result={result} must read 0 on a fresh registry, not be absent"
        );
    }
    assert_eq!(
        series(&fresh),
        RESULTS.len(),
        "exactly the four result cells may be seeded"
    );

    for &result in m::STRUCTURAL_CACHE_AUDIT_FENCED_RESULTS {
        metrics.inc_structural_cache_audit_fenced(result);
    }
    let ticked = render(&metrics);
    for result in RESULTS {
        assert_eq!(
            sample_value(&ticked, METRIC, &[("result", result)]),
            Some(1.0),
            "result={result}: one increment is one tick"
        );
    }
    assert_eq!(
        series(&ticked),
        RESULTS.len(),
        "an increment per result opens no series outside the vocabulary"
    );
    assert!(
        label_value_set(&ticked, METRIC, "service").is_empty(),
        "no `service` label, matching the other structural cache counters"
    );
}

/// b.y3m Epic 3 (Spec SDD SR-8.3, §17.1's seeding rule; defer-13): now that a
/// production read — a catch-up's shared fenced-read flight — takes the
/// fence, `live_edit_snapshot_fence_acquire_total` must pre-initialise every
/// `result` the acquire script can answer a reader with, at zero, and no
/// other: the two grant kinds and the three `RETRY_AFTER` reasons, with no
/// `service` label. A rising `retry_after_*` rate is how an operator sees
/// readers being held off, so each cell must exist before its first tick.
///
/// The fence's reader-wait histogram is the other half of defer-13's units
/// decision: one observation per read attempt, nothing else, so like every
/// histogram it records at its first observation and a fresh registry carries
/// none of it (a catch-up's cap ending is counted on
/// `live_edit_subscriber_stream_exits_total{reason="fenced_read_wait_cap"}`,
/// which `subscribe_admission_test.rs` checks is seeded).
#[tokio::test]
async fn snapshot_fence_acquire_total_pre_initialises_every_acquire_result_at_zero() {
    use live_edit::observability::metrics as m;

    const RESULTS: [&str; 5] = [
        "started_epoch",
        "joined_epoch",
        "retry_after_epoch",
        "retry_after_gap",
        "retry_after_intent",
    ];
    const METRIC: &str = "live_edit_snapshot_fence_acquire_total";
    let set = |values: &[&str]| -> std::collections::HashSet<String> {
        values.iter().map(|v| v.to_string()).collect()
    };
    assert_eq!(
        set(m::SNAPSHOT_FENCE_ACQUIRE_RESULTS),
        set(&RESULTS),
        "the acquire `result` vocabulary is an operator-facing contract; a difference here is a \
         rename or an added/removed value"
    );

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);
    assert_eq!(label_value_set(&text, METRIC, "result"), set(&RESULTS));
    for result in RESULTS {
        assert_eq!(
            sample_value(&text, METRIC, &[("result", result)]),
            Some(0.0),
            "{METRIC}{{result=\"{result}\"}} must read 0 on a fresh registry, not be absent"
        );
    }
    let series = text
        .lines()
        .filter(|l| l.starts_with(&format!("{METRIC}{{")))
        .count();
    assert_eq!(
        series,
        RESULTS.len(),
        "exactly one series per acquire result"
    );
    assert!(
        label_value_set(&text, METRIC, "service").is_empty(),
        "no `service` label, matching the sibling snapshot-read counters"
    );
    assert!(
        !text
            .lines()
            .any(|l| l.starts_with("live_edit_snapshot_fence_reader_wait_seconds")),
        "the reader-wait histogram records per read attempt, at the first one; a fresh registry \
         must carry none of it"
    );
}

/// b.muu: `live_edit_snapshot_read_refused_total` must pre-initialise every
/// `kind` cell at zero — exactly the twelve registered kinds, no thirteenth,
/// `tls` included on a build with no TLS feature, and no `service` label — and
/// one increment per vocabulary member must open exactly twelve series.
///
/// The twin of the b.atj test above, with one extra property to defend and one
/// weaker guard to compensate for.
///
/// **The extra property: `tls` is seeded unconditionally.** The `ErrorKind::Tls`
/// variant is feature-gated in fred and reached only through this crate's
/// `valkey-tls` feature, so the `snapshot_refusal_kind_label` arm that returns
/// `"tls"` is `#[cfg]`-gated to match. The *cell* deliberately is not: a
/// published series set that changes shape with the build shape means an alert
/// expression, a dashboard panel or a recording rule silently covers a different
/// set of series depending on how the binary was compiled, and the operator has
/// no way to tell.
///
/// **Be precise about what that assertion is worth here.** `valkey-tls` is a
/// DEFAULT feature (`Cargo.toml` `default = ["valkey-tls"]`), and every test
/// tier — the narrow runs, **Full test**, CI's `test:` / `test-cluster:` — runs
/// with default features on. So in the shape this test normally executes in,
/// `tls` is present whether or not the pre-init loop is `#[cfg]`-gated, and the
/// assertion below is a *vocabulary* check rather than a build-shape one. It
/// does discriminate in the third shape CLAUDE.md's **Lint** entry names, and
/// that was verified rather than assumed: gating the seed on
/// `cfg!(feature = "valkey-tls")` leaves this test green under the default
/// features and red under
/// `cargo nextest run -p live-edit --no-default-features --features test-support
/// --test integration`, which is the only invocation that exercises it. Nothing
/// in CI runs the suite in that shape today, so the unconditional seed is
/// guarded by this test plus review, not by a gate — worth knowing before
/// treating a green pipeline as proof of it.
///
/// **The weaker guard: there is no enum here.** b.atj's `key_family` vocabulary
/// is backed by `SnapshotKeyFamily`, whose `_`-less `label()` match makes a new
/// variant a compile error until it has a constant. `ErrorKind` is fred's enum,
/// so `snapshot_refusal_kind_label` keeps a `_ => …UNKNOWN` catch-all on purpose
/// (a fred bump that adds a kind must not break the build) — which means nothing
/// in the compiler forces a new label into `SNAPSHOT_READ_REFUSAL_KINDS`. What
/// takes its place is bounded cardinality: every emittable value is one of these
/// twelve constants, so the exact-set assertion below IS the cardinality guard
/// the alert recipe rests on, and it is load-bearing in a way its b.atj
/// counterpart is not.
///
/// **Both directions of the set equality are asserted, against two different
/// sources**, for the reason the b.atj test's doc-comment spells out: the
/// rendered set is compared with `metrics::SNAPSHOT_READ_REFUSAL_KINDS` (which
/// catches a pre-init loop that drifted from the vocabulary it seeds), and that
/// `const` is compared with a list restated here (which catches a RENAME — a
/// rendered-vs-const comparison alone cannot, because both sides read the same
/// array). `tests/integration/storage/b_muu_tests.rs` asserts against restated
/// literals too, so this is where the two spellings are forced to agree.
///
/// **The seeded set is only half the cardinality claim; the increment direction
/// is the other half.** Everything above reads a registry nothing has ticked, so
/// it pins the cells the pre-init loop OPENS and says nothing about the cells an
/// increment opens. Those can differ: `sample_value` matches a label subset, so
/// a `kind` cell that had gained a second label dimension — a `service` label
/// copied in from a neighbouring counter, say, or a per-object label — would
/// still answer every lookup above while multiplying the published series by the
/// cardinality of that second label the moment a real refusal ticked. The
/// closing block therefore drives `inc_snapshot_read_refused` once per
/// vocabulary member and counts the RENDERED series (shape borrowed from
/// `src/observability/metrics.rs`'s
/// `inc_checkpoint_restore_lineage_opens_no_series_outside_the_pinned_vocabulary`):
/// twelve increments over twelve members must leave exactly twelve series, each
/// reading 1. Two members collapsing onto one cell reads 2 and is caught by the
/// per-cell half; a member opening a series the pre-init loop did not is caught
/// by the count.
#[tokio::test]
async fn snapshot_read_refused_total_pre_initialises_its_bounded_kind_set() {
    /// The twelve `kind` cells, restated. See the doc-comment above for why
    /// this is a literal rather than a re-read of the `const` under test.
    ///
    /// The twelfth is `backend_out_of_memory`: an `-OOM` reply is the backing
    /// store answering that it is full, split out of `unknown` by its own
    /// predicate and deliberately NOT folded into `backend_unavailable`, because
    /// under the `maxmemory-policy: noeviction` this project mandates a store
    /// full of data in use stays refused until an operator acts (the GC's
    /// out-of-memory collection frees only released external values).
    const KINDS: [&str; 12] = [
        "protocol",
        "parse",
        "auth",
        "config",
        "tls",
        "invalid_argument",
        "invalid_command",
        "routing",
        "not_found",
        "backend_unavailable",
        "backend_out_of_memory",
        "unknown",
    ];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let registered: std::collections::HashSet<String> =
        live_edit::observability::metrics::SNAPSHOT_READ_REFUSAL_KINDS
            .iter()
            .map(|kind| (*kind).to_string())
            .collect();
    let expected: std::collections::HashSet<String> = KINDS.into_iter().map(String::from).collect();
    assert_eq!(
        registered, expected,
        "metrics::SNAPSHOT_READ_REFUSAL_KINDS must be exactly the twelve kinds named in the b.muu \
         contract. A difference here is a RENAME (or an added/removed kind), which silently breaks \
         every dashboard and alert grouping by kind and which the rendered-vs-const comparison \
         below cannot see, because both of its sides read this same array; got {registered:?}"
    );

    let kinds = label_value_set(&text, "live_edit_snapshot_read_refused_total", "kind");
    assert_eq!(
        kinds, expected,
        "live_edit_snapshot_read_refused_total must pre-initialise exactly the registered kind \
         vocabulary (b.muu). A MISSING cell is invisible until its first refusal, which is the \
         absent-series hole the zero-seed exists to close; an EXTRA one means an unregistered \
         label reached the counter, and since `snapshot_refusal_kind_label` returns only these \
         constants that would be a second, unbounded emitter — the cardinality fault the whole \
         vocabulary exists to prevent; got {kinds:?}"
    );

    for kind in KINDS {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_snapshot_read_refused_total",
                &[("kind", kind)],
            ),
            Some(0.0),
            "kind=\"{kind}\" must read 0 on a fresh registry, not be absent — \"the backing store \
             has never permanently refused a touch of an object-state key\" is an assertion an \
             operator needs to be able to make from a FIRST scrape, and it is the baseline the \
             `increase(...) > 0` recipe measures against for the eleven non-backend_unavailable \
             cells"
        );
    }

    assert_eq!(
        sample_value(
            &text,
            "live_edit_snapshot_read_refused_total",
            &[("kind", "tls")],
        ),
        Some(0.0),
        "the `tls` cell must be seeded whatever the build shape. The label arm that emits it is \
         `#[cfg(feature = \"valkey-tls\")]`-gated because fred's `ErrorKind` variant is; the CELL \
         is registered unconditionally so the published series set does not depend on the cargo \
         features the binary was compiled with. Gate the pre-init and an alert expression, \
         dashboard panel or recording rule silently covers a different set of series per binary, \
         with nothing to tell the operator which. NOTE: `valkey-tls` is a DEFAULT feature, so this \
         line only discriminates under `--no-default-features` — see the doc-comment"
    );

    assert!(
        label_value_set(&text, "live_edit_snapshot_read_refused_total", "service").is_empty(),
        "this counter carries no `service` label, matching both its b.atj sibling and the \
         commit-side live_edit_keyspace_type_error_total. The condition it reports — a backing \
         store refusing to serve — is a property of the shared backend rather than of the replica \
         that met it, so a per-replica split would fragment one incident across N series and page \
         N times for one outage"
    );

    // ── The increment direction ─────────────────────────────────────────────
    //
    // Everything above reads a registry nothing has ticked. Drive one increment
    // per vocabulary member and count the rendered series: `sample_value` is a
    // SUBSET match over labels, so a second label dimension appearing on this
    // counter — the cardinality fault the whole bounded vocabulary exists to
    // prevent — answers every lookup above unchanged and is invisible until the
    // series are counted.
    for &kind in live_edit::observability::metrics::SNAPSHOT_READ_REFUSAL_KINDS {
        metrics.inc_snapshot_read_refused(kind);
    }
    let ticked = render(&metrics);

    let series: Vec<&str> = ticked
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("live_edit_snapshot_read_refused_total{"))
        .collect();
    assert_eq!(
        series.len(),
        KINDS.len(),
        "driving every registered kind exactly once must leave exactly {} series — one per cell, \
         no more. A larger count is a SECOND label dimension on this counter (a `service` label \
         copied in from a neighbour, an object id, anything unbounded), which the `sample_value` \
         lookups above cannot see because they match a label SUBSET, and which turns the \
         `increase(...) > 0` recipe into a per-dimension page storm; got {} series: {series:?}",
        KINDS.len(),
        series.len()
    );

    for kind in KINDS {
        assert_eq!(
            sample_value(
                &ticked,
                "live_edit_snapshot_read_refused_total",
                &[("kind", kind)],
            ),
            Some(1.0),
            "kind=\"{kind}\" must read exactly 1 after one increment. A 2 means two vocabulary \
             members collapsed onto one cell — a duplicate in SNAPSHOT_READ_REFUSAL_KINDS, or a \
             `Metrics::inc_snapshot_read_refused` that ignored its argument — which would make \
             every per-kind assertion in `b_muu_tests` agree with a counter that had stopped \
             distinguishing a credential fault from a replaying primary"
        );
    }
}

/// b.gvf: `live_edit_snapshot_read_window_fallback_total` — structural WATCH
/// reads that fell back from their per-window read script to plain commands
/// on a cluster-state refusal — must exist at zero on a fresh registry, as
/// one unlabelled series (no `service` label, no other dimension), so an
/// `increase(...) > 0` alert evaluates a present series from the first
/// scrape; and `inc_snapshot_read_window_fallback` must tick that one series,
/// opening no other.
#[tokio::test]
async fn snapshot_read_window_fallback_total_pre_initialises_one_unlabelled_series() {
    const NAME: &str = "live_edit_snapshot_read_window_fallback_total";
    let series = |text: &str| -> Vec<String> {
        text.lines()
            .map(str::trim)
            .filter(|line| {
                line.strip_prefix(NAME)
                    .is_some_and(|rest| rest.starts_with('{') || rest.starts_with(' '))
            })
            .map(str::to_owned)
            .collect()
    };

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);
    assert_eq!(
        series(&text).len(),
        1,
        "{NAME} must render exactly one series on a fresh registry; got {:?}",
        series(&text)
    );
    assert_eq!(
        sample_value(&text, NAME, &[]),
        Some(0.0),
        "{NAME} must read 0 on a fresh registry, not be absent"
    );
    assert!(
        label_value_set(&text, NAME, "service").is_empty(),
        "{NAME} carries no `service` label: a fallback is a property of the backing store's \
         cluster state, not of the replica that met it"
    );

    metrics.inc_snapshot_read_window_fallback();
    let ticked = render(&metrics);
    assert_eq!(
        series(&ticked).len(),
        1,
        "one increment must tick the one series and open no other; got {:?}",
        series(&ticked)
    );
    assert_eq!(sample_value(&ticked, NAME, &[]), Some(1.0));
}

/// b.qpf: the two external-value release counters must exist, at zero, on a
/// fresh registry — `live_edit_ext_ref_released_total` over exactly its three
/// `reason` cells, and the unlabelled
/// `live_edit_ext_ref_release_enqueue_failed_total`.
///
/// The pair is read as a ratio, which is what makes zero-seeding load-bearing
/// on BOTH halves rather than a formality on one. The failure counter is the
/// leak signal and is alertable at `increase(...) > 0`; an alert on a series
/// that did not exist a scrape ago has no derivative to fire on, so an absent
/// baseline is the difference between "the enqueue has never failed" and "the
/// alert has never been able to fire". The released counter is the denominator:
/// an operator comparing `ext:meta` growth against release volume has to be
/// able to tell "no rewrite has happened" from "the rewrite cell is missing",
/// and only a seeded zero says the former.
///
/// `reason` is asserted as an exact set, not a subset. It is a BOUNDED label —
/// the three values come from one `const` array in `src/observability/metrics.rs`
/// and every emitter is fed from it — so a fourth value appearing here means
/// either a new reason that was never added to the pre-init list (its series
/// would then be absent until the first occurrence, exactly the hole this test
/// closes) or an unbounded string reaching the label, which is a cardinality
/// fault.
#[tokio::test]
async fn ext_ref_release_counters_pre_initialise_their_bounded_reason_set() {
    /// Every `reason` cell `live_edit_ext_ref_released_total` may carry.
    ///
    /// Restated rather than imported: the label VALUES are the operator-facing
    /// contract (they appear in dashboards and alert expressions), so a test
    /// that read them from the same `const` the emitter reads could only ever
    /// agree with itself and could not fail when one was renamed.
    /// b.yda added the fourth: a reference handed back out of band by the
    /// reference-count reconciler, which cannot attribute it to any object.
    const REASONS: [&str; 4] = ["rewrite", "surplus_acquire", "retirement", "reconcile"];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let reasons = label_value_set(&text, "live_edit_ext_ref_released_total", "reason");
    let expected: std::collections::HashSet<String> =
        REASONS.into_iter().map(String::from).collect();
    assert_eq!(
        reasons, expected,
        "live_edit_ext_ref_released_total must pre-initialise exactly {expected:?} (b.qpf) — \
         a missing cell is invisible until its first occurrence, and an extra one is either a \
         reason that escaped the pre-init list or an unbounded string on the label; got \
         {reasons:?}"
    );

    for reason in REASONS {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_ext_ref_released_total",
                &[("reason", reason)],
            ),
            Some(0.0),
            "reason=\"{reason}\" must read 0 on a fresh registry, not be absent"
        );
    }

    assert_eq!(
        sample_value(&text, "live_edit_ext_ref_release_enqueue_failed_total", &[]),
        Some(0.0),
        "live_edit_ext_ref_release_enqueue_failed_total is the b.qpf LEAK signal and is \
         alertable at `increase(...) > 0`. It must read 0 on a fresh registry: the commit it \
         belongs to has already landed and is never failed for the enqueue, so this counter is \
         the only thing that says a ref_count will not be decremented — and an alert on a \
         series that appears only at incident time cannot fire"
    );

    assert!(
        label_value_set(
            &text,
            "live_edit_ext_ref_release_enqueue_failed_total",
            "reason"
        )
        .is_empty(),
        "the failure counter is deliberately UNLABELLED — the enqueue is one call with one \
         failure mode, and splitting it by reason would invite a per-reason alert on a signal \
         whose remediation does not depend on which release was lost"
    );
}

/// b.yda: the reference-count reconciler's whole observability surface must
/// exist, at zero, on a fresh registry — before any pass has run.
///
/// Every cell here is one an operator reads on a FIRST scrape to make a negative
/// assertion, which a missing series cannot support and a seeded zero can:
///
/// * `ext_reconcile_total{outcome}` over its full ten-cell vocabulary. Two of
///   the ten are alertable (`enumeration_failed`, `deficit_observed`) and both
///   are supposed to read zero forever on a healthy deployment — which is
///   exactly the shape whose series would otherwise first appear at incident
///   time, with no derivative for `increase(...) > 0` to fire on.
/// * The per-object reference-index recount counters,
///   `ext_index_recount_total{site, outcome}` and
///   `ext_index_drift_total{site, direction}`, over every (site, outcome) and
///   (site, direction) pair. `direction="under"` is the dangerous direction
///   and `outcome="failed"` / `outcome="withheld"` the ones that leave drift
///   unrepaired, and all three are supposed to stay flat after the first
///   post-upgrade pass — the same shape.
/// * `ext_ref_undecodable_total{site}` over its four corrupt-keyspace drop
///   sites. Same argument, sharper: these fire only on a keyspace that has
///   already been damaged.
/// * The three unlabelled retire-carrier lifecycle counters. They are read as a
///   trio (`deferred ~= recovered + cancelled`, and a standing gap is an
///   obligation stuck on a carrier), so a missing term makes the arithmetic
///   undefined rather than merely incomplete.
///
/// `outcome` and `site` are asserted as EXACT sets, not subsets, and the values
/// are restated here rather than imported: the label values are the
/// operator-facing contract, so a test reading them from the same `const` the
/// emitter reads could only ever agree with itself.
#[tokio::test]
async fn ext_reconcile_counters_pre_initialise_their_bounded_label_sets() {
    /// The ten `outcome` cells of `live_edit_ext_reconcile_total`.
    const OUTCOMES: [&str; 10] = [
        "candidate_registered",
        "repaired",
        "healed",
        "deferred",
        "skipped_recent_store",
        // b.rej renamed this cell from `skipped_no_stamp` when an ABSENT
        // `last_store_ms` became ELIGIBLE: what remains ineligible is a stamp
        // that is PRESENT and unparseable, which is a keyspace-integrity signal
        // rather than the pass's steady state. The two verdicts are opposite
        // now, so the name had to stop saying "no stamp".
        "skipped_unreadable_stamp",
        "enumeration_failed",
        "deficit_observed",
        "deficit_cleared",
        // A surplus held because an object's per-object reference index
        // disagreed with its rows on the hash and the recount could not repair
        // it this pass: a leak held, not a loss.
        "skipped_index_drift",
    ];
    /// The two `site` cells of the per-object reference-index recount counters.
    const RECOUNT_SITES: [&str; 2] = ["reconcile", "torn_commit"];
    /// The `site` cells of `live_edit_ext_index_drift_total` alone: the
    /// structural commit's raise-only `obj:ext` settle runs no recount, so it
    /// has no `live_edit_ext_index_recount_total` cells.
    const DRIFT_ONLY_SITES: [&str; 1] = ["structural_settle"];
    /// The six `outcome` cells of `live_edit_ext_index_recount_total`.
    /// `withheld` is any recount that saw a malformed `hash` column — an
    /// operator signal, so it needs its zero from boot like `failed`.
    const RECOUNT_OUTCOMES: [&str; 6] = [
        "repaired",
        "clean",
        "too_large",
        "not_live",
        "failed",
        "withheld",
    ];
    /// The two `direction` cells of `live_edit_ext_index_drift_total`.
    const DRIFT_DIRECTIONS: [&str; 2] = ["under", "over"];
    /// The four `site` cells of `live_edit_ext_ref_undecodable_total`.
    const SITES: [&str; 4] = [
        "lua_token",
        "row_hash",
        "ext_index_field",
        "retirement_index_field",
    ];

    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let outcomes = label_value_set(&text, "live_edit_ext_reconcile_total", "outcome");
    let expected: std::collections::HashSet<String> =
        OUTCOMES.into_iter().map(String::from).collect();
    assert_eq!(
        outcomes, expected,
        "live_edit_ext_reconcile_total must pre-initialise exactly {expected:?} (b.yda) — a \
         missing cell is invisible until its first occurrence, and an extra one is either an \
         outcome that escaped the pre-init list or an unbounded string on the label; got \
         {outcomes:?}"
    );
    for outcome in OUTCOMES {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_ext_reconcile_total",
                &[("outcome", outcome)]
            ),
            Some(0.0),
            "outcome=\"{outcome}\" must read 0 on a fresh registry, not be absent"
        );
    }

    // The recount counters carry two labels each, so the exact label sets are
    // checked per label and then every PAIR is checked present at zero: two
    // complete label sets do not prove the product is complete.
    let set_of = |values: &[&str]| -> std::collections::HashSet<String> {
        values.iter().map(|v| (*v).to_owned()).collect()
    };
    for (metric, label, expected) in [
        (
            "live_edit_ext_index_recount_total",
            "site",
            set_of(&RECOUNT_SITES),
        ),
        (
            "live_edit_ext_index_recount_total",
            "outcome",
            set_of(&RECOUNT_OUTCOMES),
        ),
        (
            "live_edit_ext_index_drift_total",
            "site",
            set_of(&RECOUNT_SITES)
                .union(&set_of(&DRIFT_ONLY_SITES))
                .cloned()
                .collect(),
        ),
        (
            "live_edit_ext_index_drift_total",
            "direction",
            set_of(&DRIFT_DIRECTIONS),
        ),
    ] {
        let got = label_value_set(&text, metric, label);
        assert_eq!(
            got, expected,
            "{metric} must pre-initialise exactly {expected:?} on `{label}` — a missing cell is \
             invisible until its first occurrence, and an extra one is a value that escaped the \
             pre-init list; got {got:?}"
        );
    }
    for site in DRIFT_ONLY_SITES {
        for direction in DRIFT_DIRECTIONS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_ext_index_drift_total",
                    &[("site", site), ("direction", direction)],
                ),
                Some(0.0),
                "drift-only site=\"{site}\", direction=\"{direction}\" must read 0 on a fresh \
                 registry"
            );
        }
    }
    for site in RECOUNT_SITES {
        for outcome in RECOUNT_OUTCOMES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_ext_index_recount_total",
                    &[("site", site), ("outcome", outcome)],
                ),
                Some(0.0),
                "site=\"{site}\", outcome=\"{outcome}\" must read 0 on a fresh registry"
            );
        }
        for direction in DRIFT_DIRECTIONS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_ext_index_drift_total",
                    &[("site", site), ("direction", direction)],
                ),
                Some(0.0),
                "site=\"{site}\", direction=\"{direction}\" must read 0 on a fresh registry — \
                 `direction=\"under\"` recurring is the signal that something still produces \
                 drift in the dangerous direction, and it needs a baseline from boot"
            );
        }
    }

    let sites = label_value_set(&text, "live_edit_ext_ref_undecodable_total", "site");
    let expected: std::collections::HashSet<String> = SITES.into_iter().map(String::from).collect();
    assert_eq!(
        sites, expected,
        "live_edit_ext_ref_undecodable_total must pre-initialise exactly {expected:?} (b.yda); \
         got {sites:?}"
    );
    for site in SITES {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_ext_ref_undecodable_total",
                &[("site", site)]
            ),
            Some(0.0),
            "site=\"{site}\" must read 0 on a fresh registry — these fire only on an already \
             damaged keyspace, so the baseline has to predate the damage"
        );
    }

    for name in [
        "live_edit_ext_ref_release_deferred_total",
        "live_edit_ext_ref_release_recovered_total",
        "live_edit_ext_ref_release_cancelled_total",
    ] {
        assert_eq!(
            sample_value(&text, name, &[]),
            Some(0.0),
            "{name} must read 0 on a fresh registry. The three are read as one identity — \
             `deferred ~= recovered + cancelled` — so an absent term does not merely hide a \
             number, it makes the identity unevaluable"
        );
        assert!(
            label_value_set(&text, name, "reason").is_empty(),
            "{name} is deliberately UNLABELLED, matching its sibling \
             live_edit_ext_ref_release_enqueue_failed_total: one obligation, one lifecycle, and \
             the remediation does not depend on which release was deferred"
        );
    }
}

/// b.8jc: the reconciler's signed-zero merge counter,
/// `live_edit_time_sample_signed_zero_merge_total{outcome}`, must exist at
/// zero over its whole seven-cell vocabulary on a fresh registry, before any
/// pass has run.
///
/// `undecodable` and `failed` are the cells that need an operator when they
/// recur, and `merged` / `rekeyed` recurring after the whole fleet runs this
/// build means something still writes `-0.0`: each is read as an increase
/// over a baseline, which a series that first appears at incident time cannot
/// give. The set is EXACT, and the values are restated rather than imported,
/// for the reason `ext_reconcile_counters_pre_initialise_their_bounded_label_sets`
/// gives.
#[tokio::test]
async fn time_sample_signed_zero_merge_counter_pre_initialises_its_outcomes() {
    const OUTCOMES: [&str; 7] = [
        "merged",
        "rekeyed",
        "changed",
        "fenced",
        "not_live",
        "undecodable",
        "failed",
    ];
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let outcomes = label_value_set(
        &text,
        "live_edit_time_sample_signed_zero_merge_total",
        "outcome",
    );
    let expected: std::collections::HashSet<String> =
        OUTCOMES.into_iter().map(String::from).collect();
    assert_eq!(
        outcomes, expected,
        "live_edit_time_sample_signed_zero_merge_total must pre-initialise exactly {expected:?}; \
         got {outcomes:?}"
    );
    for outcome in OUTCOMES {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_time_sample_signed_zero_merge_total",
                &[("outcome", outcome)]
            ),
            Some(0.0),
            "outcome=\"{outcome}\" must read 0 on a fresh registry, not be absent"
        );
    }
}

/// b.4et: the two DR stream-adjudication counters must pre-initialise every
/// cell at zero — `live_edit_checkpoint_recovery_stream_unresolved_total` over
/// its whole pinned `reason` vocabulary, and its sibling
/// `live_edit_checkpoint_recovery_stream_refused_total`.
///
/// These are the sharpest possible case for zero-seeding, and the reason both
/// are pinned in one test is that they are the two halves of one operator
/// question: "has DR ever had to do something unusual with a surviving
/// per-object stream?" A healthy fleet answers no on every cell of both,
/// forever — so the assertion an operator needs to be able to make is over a
/// FIRST scrape, before any restore has run, which a series that materialises
/// only at incident time cannot support. A Prometheus `increase(...[5m])` on an
/// absent series has no derivative to fire on.
///
/// The three reasons are deliberately NOT merged, and the split is not
/// cosmetic: two of them (`probe_unreadable`, `unlink_unverified`) mean the
/// restore was REFUSED and the layer did not come back, while the third
/// (`wrongtype_key`) means it DID come back and every write to it now fails
/// closed. Same family, opposite incidents, different remediations.
///
/// The refused counter is the third distinct outcome and belongs beside them:
/// the restore landed AND a surviving stream was verifiably discarded. Its
/// label set is `service` alone — no `reason` — because there is only one way
/// to get there (the plausibility gate), and an unlabelled cell is what lets a
/// dashboard sum it against the `unresolved` family without a join.
#[tokio::test]
async fn checkpoint_recovery_stream_adjudication_counters_are_zero_seeded() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let reasons = label_value_set(
        &text,
        "live_edit_checkpoint_recovery_stream_unresolved_total",
        "reason",
    );
    let expected: std::collections::HashSet<String> =
        live_edit::observability::metrics::CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS
            .iter()
            .map(|reason| (*reason).to_string())
            .collect();
    assert_eq!(
        reasons, expected,
        "live_edit_checkpoint_recovery_stream_unresolved_total must pre-initialise exactly the \
         pinned reason vocabulary (b.4et); got {reasons:?}. A reason minted at an increment site \
         but absent from CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS renders a series only \
         once that arm first fires, which is indistinguishable from a code path that was never \
         wired"
    );

    for reason in live_edit::observability::metrics::CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_checkpoint_recovery_stream_unresolved_total",
                &[("reason", reason)],
            ),
            Some(0.0),
            "reason=\"{reason}\" must read 0 on a fresh registry, not be absent — \"DR has never \
             had to refuse a restore for want of stream evidence\" is an assertion an operator \
             needs to be able to make from a FIRST scrape"
        );
    }

    assert_eq!(
        sample_value(
            &text,
            "live_edit_checkpoint_recovery_stream_refused_total",
            &[("service", "test")],
        ),
        Some(0.0),
        "the sibling `..._refused_total` (b.8y5) must read 0 on a fresh registry too. It counts \
         the third outcome — the restore LANDED after verifiably discarding a stale stream — and \
         a dashboard that can read the two refusal reasons but not this one cannot tell \
         \"nothing unusual happened\" from \"the series has not been created yet\""
    );
}

/// b.f4t: `live_edit_checkpoint_restore_residue_wiped_total` must pre-initialise
/// its single `{service}` cell at zero.
///
/// Same operator question as the two stream-adjudication counters above, one
/// step earlier in the same restore: "has DR ever found a keyspace still alive
/// under an object it was about to materialize?" A healthy fleet answers no
/// forever, so the answer has to be readable from a FIRST scrape — an absent
/// series gives `increase(...[5m])` no derivative, and a counter that only
/// appears at incident time is indistinguishable from a wipe path that was
/// never wired.
///
/// The label set is `service` alone, deliberately: there is one way to reach
/// the increment (`ResidueOutcome::Wiped`), and the per-family breakdown of
/// what survived rides the accompanying WARN rather than a label — a
/// section/field/tombstone dimension here would multiply cardinality without
/// changing any alert.
#[tokio::test]
async fn checkpoint_restore_residue_wiped_counter_is_zero_seeded() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    assert_eq!(
        sample_value(
            &text,
            "live_edit_checkpoint_restore_residue_wiped_total",
            &[("service", "test")],
        ),
        Some(0.0),
        "live_edit_checkpoint_restore_residue_wiped_total{{service=\"test\"}} must read 0 on a \
         fresh registry, not be absent (b.f4t). It is the only signal that a restore had to \
         REPLACE a surviving keyspace rather than write into an empty one — the b.kf1 orphan \
         class — and an operator has to be able to read \"that has never happened here\" before \
         the first occurrence, not only after it"
    );
}

/// b.h4m: `live_edit_checkpoint_restore_fenced_total` must pre-initialise every
/// `{stage}` cell of its pinned vocabulary at zero — and only those.
///
/// The sharpest form of the argument the residue-wiped counter one test above
/// makes. On a single-replica or serialized deployment every cell here is
/// expected to read ZERO forever, and "no two replicas have ever raced to
/// restore the same layer" is exactly the assertion an operator wants to make
/// from a FIRST scrape — before any restore has run at all. An absent series
/// gives `increase(...[5m])` no derivative and makes "never happened" impossible
/// to tell from "this fence was never wired".
///
/// **The three stages are deliberately not merged, and the vocabulary is pinned
/// rather than passed through.** They are ordered by how late the discovery is,
/// and the distance between them is the operator's whole read on how tight the
/// race is: `presence_probe` (a point-in-time read before any work — unremarkable
/// on a lazily-recovering fleet), `residue_wipe` (the peer's header landed under
/// the wipe's `WATCH`, reachable only when residue survived), and `object_write`
/// (the peer's header landed under the materialization's own `WATCH` — the
/// innermost fence, the only one of the two `WATCH`-held fences a total-loss
/// restore can reach, and the one that closes the mixture-of-two-backups hazard;
/// `presence_probe` is a non-fenced point-in-time check every restore runs, total
/// loss included). A fourth `stage` minted at an
/// increment site but absent from `CHECKPOINT_RESTORE_FENCED_STAGES` would render
/// a series only once that arm first fires, and would silently widen the label
/// set an operator sums over.
///
/// Deliberately additive to `live_edit_checkpoint_recovery_attempts_total`, never
/// a substitute: a fenced attempt still reports `result="restored"` there, since
/// the layer IS restored — by the other replica. The two counters answer
/// different questions and both must be readable from the same scrape.
#[tokio::test]
async fn checkpoint_restore_fenced_counter_is_zero_seeded_across_its_stage_vocabulary() {
    let metrics = crate::common::metrics_scrape_only("test");
    let text = render(&metrics);

    let stages = label_value_set(&text, "live_edit_checkpoint_restore_fenced_total", "stage");
    let expected: std::collections::HashSet<String> =
        live_edit::observability::metrics::CHECKPOINT_RESTORE_FENCED_STAGES
            .iter()
            .map(|stage| (*stage).to_string())
            .collect();
    assert_eq!(
        stages, expected,
        "live_edit_checkpoint_restore_fenced_total must pre-initialise exactly the pinned stage \
         vocabulary (b.h4m); got {stages:?}"
    );

    // The cardinality is pinned as a literal as well as against the const,
    // because asserting only "the rendered set equals the const" is satisfied by
    // a const that lost a member: both sides move together. Three is the number
    // of points inside `apply_restore` at which a peer's landed rebuild can be
    // discovered, and a change to it is a change to the fence topology that
    // should have to be restated here.
    assert_eq!(
        live_edit::observability::metrics::CHECKPOINT_RESTORE_FENCED_STAGES.len(),
        3,
        "the fence topology is three stages — apply_restore's step-1 presence probe, the step-3b \
         residue wipe, and write_restored_object's own WATCH. A fourth means a new fence was \
         added (state it here) and a second means one was removed (which one, and what now \
         catches its case?)"
    );
    assert_eq!(
        live_edit::observability::metrics::CHECKPOINT_RESTORE_FENCED_STAGES,
        ["presence_probe", "residue_wipe", "object_write"],
        "the stage values are pinned label text an operator's dashboards and alerts match on, \
         ordered outermost-fence-first; renaming one silently breaks every expression built on it"
    );

    for stage in live_edit::observability::metrics::CHECKPOINT_RESTORE_FENCED_STAGES {
        assert_eq!(
            sample_value(
                &text,
                "live_edit_checkpoint_restore_fenced_total",
                &[("stage", stage), ("service", "test")],
            ),
            Some(0.0),
            "stage=\"{stage}\" must read 0 on a fresh registry, not be absent — on a serialized \
             deployment every cell is expected to stay 0 forever, so a flat line IS the \
             assertion, and it cannot be made against a series that does not exist yet"
        );
    }
}

/// `live_edit_snapshot_read_duration_seconds` resolves the band an atomic
/// catch-up read takes under a join burst (around 100–200 ms end to end):
/// the boundaries 0.075, 0.15 and 0.2 sit inside it, alongside the original
/// set. The `le` values are compared as numbers, so the exporter's float
/// formatting does not matter. A 120 ms read lands at `le=0.15` and not at
/// `le=0.1`, which only the 0.15 boundary makes visible.
#[tokio::test]
async fn snapshot_read_duration_buckets_resolve_the_catch_up_read_band() {
    const HISTOGRAM: &str = "live_edit_snapshot_read_duration_seconds";
    const BOUNDS: [f64; 18] = [
        0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0, 30.0,
    ];
    let metrics = crate::common::metrics_scrape_only("test");
    metrics.observe_snapshot_read(
        "catchup",
        "atomic",
        "ok",
        std::time::Duration::from_millis(120),
    );
    let text = render(&metrics);
    let bucket = format!("{HISTOGRAM}_bucket");

    // The exported `le` strings, each with the boundary it parses to.
    let mut les: Vec<(f64, String)> = label_value_set(&text, &bucket, "le")
        .into_iter()
        .filter(|le| le != "+Inf")
        .map(|le| {
            let bound = le
                .parse()
                .unwrap_or_else(|_| panic!("le={le:?} is not a number"));
            (bound, le)
        })
        .collect();
    les.sort_by(|a, b| a.0.total_cmp(&b.0));
    let bounds: Vec<f64> = les.iter().map(|(bound, _)| *bound).collect();
    assert_eq!(bounds, BOUNDS, "{HISTOGRAM}'s bucket boundaries");

    let count_at = |bound: f64| {
        let (_, le) = les
            .iter()
            .find(|(b, _)| *b == bound)
            .unwrap_or_else(|| panic!("no le={bound} bucket"));
        sample_value(&text, &bucket, &[("le", le.as_str())])
    };
    assert_eq!(
        (count_at(0.1), count_at(0.15)),
        (Some(0.0), Some(1.0)),
        "a 120 ms read lands in the 0.15 bucket, not the 0.1 one"
    );
}

/// `live_edit_subscribe_catchups_pending` is pre-initialised at zero (the
/// autoscaler dashboard and any alert evaluate a present series from the
/// first scrape), and moves with `begin_pending_catchup` /
/// `end_pending_catchup` together with the `/scaling/v1` counters it
/// mirrors: a pending catch-up that ends completed adds one to the completed
/// count, one that ends without its ack adds nothing.
#[tokio::test]
async fn subscribe_catchups_pending_is_seeded_at_zero_and_moves_with_its_scaling_mirror() {
    const GAUGE: &str = "live_edit_subscribe_catchups_pending";
    let metrics = crate::common::metrics_scrape_only("test");
    let pending = |m: &Metrics| sample_value(&render(m), GAUGE, &[]);
    assert_eq!(
        pending(&metrics),
        Some(0.0),
        "{GAUGE} must read 0, not be absent"
    );

    metrics.begin_pending_catchup();
    metrics.begin_pending_catchup();
    let counters = metrics.scaling_counters();
    assert_eq!(pending(&metrics), Some(2.0));
    assert_eq!(
        (counters.catchups_pending(), counters.catchups_completed()),
        (2, 0)
    );

    metrics.end_pending_catchup(true);
    metrics.end_pending_catchup(false);
    let counters = metrics.scaling_counters();
    assert_eq!(pending(&metrics), Some(0.0));
    assert_eq!(
        (counters.catchups_pending(), counters.catchups_completed()),
        (0, 1),
        "only the catch-up that reached its ack counts as completed"
    );

    metrics.add_subscriber_admitted(2);
    metrics.add_subscriber_admitted(-1);
    assert_eq!(metrics.scaling_counters().subscriptions_admitted(), 1);
    assert_eq!(
        sample_value(&render(&metrics), "live_edit_subscriber_admitted", &[]),
        Some(1.0),
        "the admitted gauge and its mirror move together"
    );
}
