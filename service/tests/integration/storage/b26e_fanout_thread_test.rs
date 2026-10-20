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

//! b.26e Stage 2 (r14 B.1 item 4) — the snapshot lane's Valkey routers run
//! on the client's `snapshot-io` thread (`storage::fanout_thread`), and no
//! other pool's do: after `connect`, after a router supervisor restart of a
//! snapshot member's ended router, and after a snapshot member's replacement
//! by a fresh client. And every one of the client's [`FANOUT_RUNTIMES`]
//! fan-out threads — `fanout-read`, `snapshot-io` and the
//! [`WATCH_READ_THREADS`] `watch-read` threads — stops once the client's
//! last owner is dropped, quit or not, whichever thread drops it.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`snapshot_lane_routers_run_on_snapshot_io_after_connect_restart_and_replacement_b26e`] | `snapshot-io` runs exactly one task per snapshot member — its router — after each of the three spawn sites, and `fanout-read` and every `watch-read` thread run none while no read does |
//! | [`snapshot_lane_credential_refresh_tasks_run_beside_their_routers_on_snapshot_io_b26e`] | with a ticking credential provider each snapshot member's connection also runs fred's credential-refresh task there, spawned from its router: two tasks per member, still no other pool's |
//! | [`a_dropped_fixture_stops_its_clients_fan_out_threads_b26e`] | every thread of every fan-out runtime has exited once a `TestValkey` is dropped |
//! | [`dropping_the_last_client_clone_stops_every_fan_out_thread_b26e`] | the threads run while the client does — every runtime still runs a task after a `quit` — and every thread of every runtime has exited once the client's last owner is dropped, here after that `quit` |
//! | [`dropping_an_unquit_clients_last_owner_stops_its_fan_out_threads_b26e`] | without `quit`, the snapshot lane still serves once one of two owners has dropped, and every fan-out thread exits once the last one drops on a service worker |
//! | [`dropping_an_unquit_clients_last_owner_after_its_runtime_ended_stops_its_fan_out_threads_b26e`] | the threads outlive the runtime the client was connected on, and exit once the un-quit client's last owner drops on a thread with no runtime |
//! | [`dropping_the_last_owner_on_fanout_read_neither_panics_nor_blocks_b26e`] | the last owner dropped by a task on `fanout-read` itself returns there, without a panic, and every fan-out thread then exits |
//! | [`dropping_the_last_owner_on_a_watch_read_thread_neither_panics_nor_blocks_b26e`] | the same, the last owner dropped by a task on the last `watch-read` thread, as when an unfenced pass held it |
//!
//! Each lifecycle test first waits until all [`FANOUT_RUNTIMES`] worker
//! threads have started, so its stop covers every runtime, the `watch-read`
//! ones included.
//!
//! The un-quit drop's release of the router watchers and the member
//! replacer, and an internal clone's not counting as an owner, are pinned in
//! `src/storage/client.rs` (`tests::client_owners`), which can hold those
//! parts past the client's drop.
//!
//! **How a thread's tasks are counted.** `snapshot-io` is a runtime of one
//! worker thread, so its alive-task count (`RuntimeMetrics::num_alive_tasks`)
//! is the set of tasks spawned on it. It is read from inside a task of its
//! own ([`tasks_on`]): that task runs on the thread, so every task the thread
//! finished before it — an aborted router's included, which completes and
//! leaves the runtime's task list in one step on that thread — is already off
//! the count, and the reading task itself is taken off. A router started on
//! the caller's runtime instead is missing from it, and any other pool's
//! router started there is extra.
//!
//! **Waits.** Every wait is bounded by [`SETTLE`]; a client's connect by its
//! own connect timeout.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{ClientLike, KeysInterface};
use tokio::time::Instant;

use live_edit::observability::metrics::{Metrics, VALKEY_POOL_SNAPSHOT};
use live_edit::storage::client::{ValkeyClient, POOL_LABEL_SNAPSHOT};
use live_edit::storage::fanout_thread::{
    watch_read_thread_name, FANOUT_READ_THREAD_NAME, FANOUT_RUNTIMES, SNAPSHOT_IO_THREAD_NAME,
    WATCH_READ_THREADS,
};
use live_edit::storage::member_replacement::{
    ReplaceKnobs, MEMBER_REPLACE_AFTER_FAST_VERDICTS, MEMBER_REPLACE_COOLDOWN,
    MEMBER_REPLACE_WINDOW,
};
use live_edit::storage::test_support::{
    connect_test_client, connect_test_client_with_credential_provider, TestValkey,
};

use super::b_zt3_support::{await_until, MEMBER_REPLACEMENTS, ROUTER_RESTARTS};
use super::mux_test_helpers::{read_series, ScriptedProvider};

/// Bound on each polled condition, each count and each `quit()`.
const SETTLE: Duration = Duration::from_secs(10);

/// The snapshot lane's members here: more than one, so a missing or extra
/// router is told from a pool of one, and fewer than the data pool's, so a
/// data router started on `snapshot-io` cannot hide in the count.
const SNAPSHOT_MEMBERS: u32 = 3;

/// The data pool's members.
const DATA_MEMBERS: u32 = 4;

/// A client of [`SNAPSHOT_MEMBERS`] snapshot and [`DATA_MEMBERS`] data
/// members, its metrics hooks installed, and the metrics.
async fn snapshot_lane_client(
    provider: Option<Arc<ScriptedProvider>>,
) -> (ValkeyClient, Arc<Metrics>) {
    let mut config = crate::common::test_config();
    config.snapshot_pool_size = SNAPSHOT_MEMBERS;
    config.valkey_pool_size = DATA_MEMBERS;
    let client = match provider {
        Some(provider) => connect_test_client_with_credential_provider(&config, provider).await,
        None => connect_test_client(&config).await,
    }
    .expect("the client must connect");
    let metrics = Arc::new(crate::common::metrics_scrape_only("b26e-fanout-thread"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    (client, metrics)
}

/// The tasks alive on the one-worker runtime `handle`, the reading task
/// itself not counted (see the module docs), with the name of the thread it
/// ran on. A failure is prefixed with `what`: a task spawned on a runtime
/// that has been shut down is cancelled, so the read fails there.
async fn tasks_on(handle: &tokio::runtime::Handle, what: &str) -> (usize, Option<String>) {
    let read = handle.spawn(async {
        (
            tokio::runtime::Handle::current()
                .metrics()
                .num_alive_tasks()
                - 1,
            std::thread::current().name().map(str::to_owned),
        )
    });
    match tokio::time::timeout(SETTLE, read).await {
        Ok(Ok(read)) => read,
        Ok(Err(e)) => panic!("{what}: the counting task did not finish: {e}"),
        Err(_) => panic!("{what}: the counting task did not run within {SETTLE:?}"),
    }
}

/// The tasks on `client`'s `snapshot-io` thread, checked to be read there.
async fn snapshot_io_tasks(client: &ValkeyClient) -> usize {
    let (tasks, thread) = tasks_on(
        client.fanout_threads().snapshot_io_handle(),
        "counting `snapshot-io`'s tasks",
    )
    .await;
    assert_eq!(thread.as_deref(), Some(SNAPSHOT_IO_THREAD_NAME));
    tasks
}

/// The tasks on `client`'s `fanout-read` thread, checked to be read there.
async fn fanout_read_tasks(client: &ValkeyClient) -> usize {
    let (tasks, thread) = tasks_on(
        client.fanout_threads().fanout_read_handle(),
        "counting `fanout-read`'s tasks",
    )
    .await;
    assert_eq!(thread.as_deref(), Some(FANOUT_READ_THREAD_NAME));
    tasks
}

/// Every snapshot member answers a command through its current client: each
/// router, wherever it runs, serves.
async fn every_snapshot_member_serves(client: &ValkeyClient, what: &str) {
    for member in 0..SNAPSHOT_MEMBERS as usize {
        let _: Option<String> = tokio::time::timeout(
            SETTLE,
            client
                .snapshot_pool
                .member(member)
                .get("b26e-fanout-thread:never-written"),
        )
        .await
        .unwrap_or_else(|_| panic!("{what}: snapshot member {member} answers in time"))
        .unwrap_or_else(|e| panic!("{what}: snapshot member {member} serves: {e:?}"));
    }
}

/// `client`'s fan-out runtimes that poll passes, each with its thread's name:
/// `fanout-read`, then the `watch-read` threads by index.
fn pass_runtimes(client: &ValkeyClient) -> Vec<(&tokio::runtime::Handle, String)> {
    let threads = client.fanout_threads();
    std::iter::once((
        threads.fanout_read_handle(),
        FANOUT_READ_THREAD_NAME.to_owned(),
    ))
    .chain(
        threads
            .watch_read_handles()
            .iter()
            .enumerate()
            .map(|(i, handle)| (handle, watch_read_thread_name(i))),
    )
    .collect()
}

/// `snapshot-io` holds exactly `per_member` tasks per snapshot member, and
/// `fanout-read` and every `watch-read` thread none.
async fn assert_lane_tasks(client: &ValkeyClient, per_member: usize, what: &str) {
    every_snapshot_member_serves(client, what).await;
    assert_eq!(
        snapshot_io_tasks(client).await,
        per_member * SNAPSHOT_MEMBERS as usize,
        "{what}: `{SNAPSHOT_IO_THREAD_NAME}` runs the snapshot lane's {SNAPSHOT_MEMBERS} members' \
         tasks ({per_member} each) and nothing of the other pools: a router started on the \
         caller's runtime is missing from it, another pool's router there is extra"
    );
    assert_eq!(
        fanout_read_tasks(client).await,
        0,
        "{what}: `{FANOUT_READ_THREAD_NAME}` runs nothing while no read does"
    );
    for (handle, name) in pass_runtimes(client).into_iter().skip(1) {
        let (tasks, thread) = tasks_on(handle, &format!("counting `{name}`'s tasks")).await;
        assert_eq!(thread.as_deref(), Some(name.as_str()));
        assert_eq!(
            tasks, 0,
            "{what}: `{name}` runs nothing while no read does: no router runs there"
        );
    }
}

/// The production replacement trigger and window, with no cooldown and no
/// retirement grace, so a replacement ends with its old client's retirement.
fn quick_replacements() -> ReplaceKnobs {
    ReplaceKnobs {
        after_fast_verdicts: MEMBER_REPLACE_AFTER_FAST_VERDICTS,
        window: MEMBER_REPLACE_WINDOW,
        cooldown: MEMBER_REPLACE_COOLDOWN,
        grace: Duration::ZERO,
    }
}

/// End snapshot member 0's router and wait until the supervisor's restart
/// is counted.
async fn restart_snapshot_router(client: &ValkeyClient, metrics: &Metrics) {
    let restarts = || read_series(metrics, ROUTER_RESTARTS, &[("pool", VALKEY_POOL_SNAPSHOT)]);
    let before = restarts();
    assert!(
        client.end_router_for_test(POOL_LABEL_SNAPSHOT, 0),
        "precondition: snapshot member 0 has a supervised router"
    );
    await_until(
        "the ended snapshot router's restart must be counted",
        Instant::now() + SETTLE,
        || restarts() == before + 1,
    )
    .await;
}

/// Replace snapshot member 0 by a fresh client and wait until the
/// replacement, its old client's retirement included, has finished.
async fn replace_snapshot_member(client: &ValkeyClient, metrics: &Metrics) {
    let old = ClientLike::id(client.snapshot_pool.member(0)).to_owned();
    for _ in 0..MEMBER_REPLACE_AFTER_FAST_VERDICTS {
        client.fast_verdict_for_test(POOL_LABEL_SNAPSHOT, 0);
    }
    await_until(
        "snapshot member 0 must be pointed at a fresh client",
        Instant::now() + SETTLE,
        || ClientLike::id(client.snapshot_pool.member(0)) != old,
    )
    .await;
    await_until(
        "the replacement's retirement must finish",
        Instant::now() + SETTLE,
        || client.member_replacement_tasks_for_test() == 0,
    )
    .await;
    assert_eq!(
        read_series(
            metrics,
            MEMBER_REPLACEMENTS,
            &[("pool", VALKEY_POOL_SNAPSHOT), ("outcome", "replaced")]
        ),
        1,
        "fixture: one snapshot member replaced"
    );
}

/// The snapshot lane's routers run on `snapshot-io` from each of the three
/// places one is started: `connect` (`MemberRouters`), the router
/// supervisor's restart of an ended router, and a member replacement's fresh
/// client (and that client's own restart). After each, `snapshot-io` runs
/// exactly one task per snapshot member and nothing else — so no data,
/// maintenance, `txn` or pub/sub router runs there — and every member
/// serves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_lane_routers_run_on_snapshot_io_after_connect_restart_and_replacement_b26e() {
    let (client, metrics) = snapshot_lane_client(None).await;
    client.set_member_replacement_knobs_for_test(quick_replacements());
    assert_lane_tasks(&client, 1, "after connect").await;

    restart_snapshot_router(&client, &metrics).await;
    assert_lane_tasks(&client, 1, "after a supervisor restart").await;

    replace_snapshot_member(&client, &metrics).await;
    assert_lane_tasks(&client, 1, "after a member replacement").await;

    restart_snapshot_router(&client, &metrics).await;
    assert_lane_tasks(&client, 1, "after the replaced member's restart").await;

    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");
}

/// With a credential provider whose refresh cadence ticks from before the
/// client connects, fred runs a credential-refresh task per connection,
/// spawned from the connection's router onto the router's runtime: on
/// `snapshot-io`, two tasks per snapshot member — its router and its
/// connection's refresh — and still nothing of the other pools, whose
/// routers and refreshes run on the caller's runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_lane_credential_refresh_tasks_run_beside_their_routers_on_snapshot_io_b26e() {
    let provider = ScriptedProvider::new("b26e-fanout-thread-password");
    provider.start_ticking();
    let (client, _metrics) = snapshot_lane_client(Some(provider)).await;
    for member in 0..SNAPSHOT_MEMBERS as usize {
        let member = client.snapshot_pool.member(member);
        await_until(
            "each snapshot member's credential-refresh task starts",
            Instant::now() + SETTLE,
            || member.inner().credentials_task.read().is_some(),
        )
        .await;
    }
    assert_lane_tasks(&client, 2, "with credential refresh").await;
    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");
}

/// Wait, within [`SETTLE`], until every thread `alive` counts has exited.
async fn fan_out_threads_stop(alive: &AtomicUsize, what: &str) {
    let deadline = Instant::now() + SETTLE;
    while alive.load(Ordering::SeqCst) != 0 {
        assert!(
            Instant::now() < deadline,
            "{} fan-out thread(s) still run {SETTLE:?} after {what}",
            alive.load(Ordering::SeqCst)
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Wait, within [`SETTLE`], until every one of the [`FANOUT_RUNTIMES`]
/// runtimes' worker threads has started. A wait, not a single read: tokio
/// counts a thread from inside it (`on_thread_start` runs in the spawned
/// worker), not when its runtime is built, so a correct build can still read
/// fewer when `connect` returns.
async fn fan_out_threads_start(alive: &AtomicUsize) {
    let deadline = Instant::now() + SETTLE;
    while alive.load(Ordering::SeqCst) < FANOUT_RUNTIMES {
        assert!(
            Instant::now() < deadline,
            "all {FANOUT_RUNTIMES} fan-out runtimes' worker threads start within {SETTLE:?}: {} \
             alive",
            alive.load(Ordering::SeqCst)
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// A dropped [`TestValkey`] leaves no fan-out thread of its client running:
/// over a one-process `cargo test` run every fixture's threads would
/// otherwise pile up. (The fixture quits its client first; the stop itself
/// is the last owner's drop, quit or not — see the un-quit tests below.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropped_fixture_stops_its_clients_fan_out_threads_b26e() {
    let tv = TestValkey::setup().await;
    let alive = tv.client().fanout_threads().alive_threads();
    fan_out_threads_start(&alive).await;
    drop(tv);
    fan_out_threads_stop(&alive, "the fixture was dropped").await;
}

/// The fan-out threads live as long as their client's owners: every
/// runtime's worker thread runs while it does — through a `quit`, whose
/// `QUIT`s the snapshot lane's routers on `snapshot-io` serve — and once its
/// last owner is dropped, quit or not, every thread of every runtime has
/// exited, the drop shutting every runtime down without blocking the
/// dropping task.
///
/// After the `quit` every runtime still runs a task ([`tasks_on`]): a task
/// spawned on a runtime that has been shut down is cancelled, so a `quit`
/// that shut any runtime down — before the snapshot members' `QUIT`s were
/// served there, or at all while the client is still owned — fails here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_the_last_client_clone_stops_every_fan_out_thread_b26e() {
    let (client, _metrics) = snapshot_lane_client(None).await;
    let alive = client.fanout_threads().alive_threads();
    fan_out_threads_start(&alive).await;
    tokio::time::timeout(SETTLE, client.quit())
        .await
        .expect("quit must settle");
    let runtimes = std::iter::once((
        client.fanout_threads().snapshot_io_handle(),
        SNAPSHOT_IO_THREAD_NAME.to_owned(),
    ))
    .chain(pass_runtimes(&client));
    for (handle, name) in runtimes {
        let (_, thread) = tasks_on(
            handle,
            &format!("`{name}` no longer runs tasks after quit while its client is still owned"),
        )
        .await;
        assert_eq!(
            thread.as_deref(),
            Some(name.as_str()),
            "the task spawned on `{name}` after the quit ran there"
        );
    }
    assert!(
        alive.load(Ordering::SeqCst) >= FANOUT_RUNTIMES,
        "every fan-out thread still runs after the quit: {} alive",
        alive.load(Ordering::SeqCst)
    );
    drop(client);
    fan_out_threads_stop(&alive, "the last client clone was dropped").await;
}

/// Without `quit()`, the fan-out threads live exactly as long as the client's
/// owners. With two owners, dropping one leaves the snapshot lane serving
/// (its routers run on `snapshot-io`); dropping the last, on a worker of the
/// test's still-running runtime, stops every thread of every runtime. Before
/// the owners split, the client's own machinery held owning clones of it and
/// an un-quit client's threads never stopped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_an_unquit_clients_last_owner_stops_its_fan_out_threads_b26e() {
    let (client, _metrics) = snapshot_lane_client(None).await;
    let alive = client.fanout_threads().alive_threads();
    fan_out_threads_start(&alive).await;
    let other_owner = client.clone();
    drop(client);
    every_snapshot_member_serves(&other_owner, "with one of two owners dropped").await;
    drop(other_owner);
    fan_out_threads_stop(&alive, "the un-quit client's last owner was dropped").await;
}

/// The un-quit client's last owner dropped on a thread whose runtime has
/// ended: the threads outlive the runtime the client was connected on —
/// they are the client's, not that runtime's — and exit once the owner drops
/// on a thread with no runtime at all. Built on a runtime of its own rather
/// than `#[tokio::test]`, since that runtime must end before the drop.
#[test]
fn dropping_an_unquit_clients_last_owner_after_its_runtime_ended_stops_its_fan_out_threads_b26e() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("the test's runtime builds");
    let (client, alive) = runtime.block_on(async {
        let (client, _metrics) = snapshot_lane_client(None).await;
        let alive = client.fanout_threads().alive_threads();
        fan_out_threads_start(&alive).await;
        (client, alive)
    });
    drop(runtime);
    assert!(
        alive.load(Ordering::SeqCst) >= FANOUT_RUNTIMES,
        "the fan-out threads outlive the runtime the client was connected on: {} alive",
        alive.load(Ordering::SeqCst)
    );
    drop(client);
    let deadline = std::time::Instant::now() + SETTLE;
    while alive.load(Ordering::SeqCst) != 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "{} fan-out thread(s) still run {SETTLE:?} after the un-quit client's last owner \
             was dropped on a thread with no runtime",
            alive.load(Ordering::SeqCst)
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Drop `client`, the last owner, from a task on its fan-out thread `name`
/// (`on` picks that thread's runtime among the client's), and check that the
/// drop returns there, in time and without a panic, and that every fan-out
/// thread then exits.
async fn drop_the_last_owner_on(
    client: ValkeyClient,
    name: &str,
    on: impl FnOnce(&ValkeyClient) -> tokio::runtime::Handle,
) {
    let alive = client.fanout_threads().alive_threads();
    fan_out_threads_start(&alive).await;
    let runtime = on(&client);
    let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
    // The task's own outcome is not awaited: the runtime it runs on is one
    // its drop shuts down. The channel reports that the drop returned.
    let _task = runtime.spawn(async move {
        let thread = std::thread::current().name().map(str::to_owned);
        drop(client);
        let _ = dropped_tx.send(thread);
    });
    let thread = tokio::time::timeout(SETTLE, dropped_rx)
        .await
        .unwrap_or_else(|_| {
            panic!("the last owner's drop on `{name}` returns in time, not blocking")
        })
        .unwrap_or_else(|_| panic!("the last owner's drop on `{name}` returns without a panic"));
    assert_eq!(thread.as_deref(), Some(name), "the drop ran on `{name}`");
    fan_out_threads_stop(&alive, &format!("the last owner was dropped on `{name}`")).await;
}

/// The last owner dropped by a task on `fanout-read` — as when a fenced pass
/// held it — shuts down the very runtime that task runs on: the drop returns
/// there without a panic (a plain runtime drop panics in an async context)
/// and without blocking, and every fan-out thread then exits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_the_last_owner_on_fanout_read_neither_panics_nor_blocks_b26e() {
    let (client, _metrics) = snapshot_lane_client(None).await;
    drop_the_last_owner_on(client, FANOUT_READ_THREAD_NAME, |client| {
        client.fanout_threads().fanout_read_handle().clone()
    })
    .await;
}

/// The same on a `watch-read` thread — as when an unfenced pass (a WATCH read
/// or an ext-GC reference walk) held the last owner — here the last one, so
/// the drop runs on a thread other than `fanout-read` and other than the
/// first `watch-read` thread: it returns there without a panic and without
/// blocking, and every fan-out thread then exits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_the_last_owner_on_a_watch_read_thread_neither_panics_nor_blocks_b26e() {
    let (client, _metrics) = snapshot_lane_client(None).await;
    let last = WATCH_READ_THREADS - 1;
    drop_the_last_owner_on(client, &watch_read_thread_name(last), |client| {
        client.fanout_threads().watch_read_handles()[last].clone()
    })
    .await;
}
