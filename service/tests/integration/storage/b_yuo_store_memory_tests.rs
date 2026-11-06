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

//! b.yuo decider ruling 4, b.qhk — the external-value release budget read
//! from the store's own memory (`storage::store_memory`), against a real
//! Valkey.
//!
//! The configuration's half of the precedence (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES`
//! over the fallback), the startup line and its level by reason (b.1jt: INFO
//! for a store that does not report its size, WARN for a fault) are pinned
//! beside `Config` (`src/config.rs`); the parses, the budget one reading gives
//! (the store's `maxmemory` over `LIVE_EDIT_VALKEY_MEMORY_BYTES` by the
//! cluster map) and the refresh rule beside the module
//! (`src/storage/store_memory.rs`). Here the store answers:
//!
//! - the shared test store, read as it is (standalone; it is never
//!   reconfigured — no `CONFIG SET`);
//! - a reply staged through `ScriptReplyProxy`, which rewrites only `INFO
//!   memory` frames: a limit, a 0, a store without the field, a malformed
//!   value, an error reply;
//! - [`AmrStore`], a proxy answering like Azure Managed Redis: `INFO memory`
//!   with no `maxmemory` field, `INFO commandstats` with a replica
//!   acknowledgement count the test moves, and a placeholder `INFO
//!   replication` the service must never ask for;
//! - [`MapFacade`], a proxy playing a cluster whose `CLUSTER SLOTS` the test
//!   rewrites (every entry on the one endpoint, as an Azure Managed Redis node
//!   fronts several shards), fails, or answers malformed, for the `CLUSTER
//!   SLOTS` every clustered read sends, whatever is configured, and whose
//!   `INFO memory` reports no `maxmemory` field or a staged limit;
//! - [`ScaleOutFacade`] (b.xb7), three proxies playing a cluster that gains a
//!   primary after the client connected, which the client's routing table
//!   does not list — the read must confirm its `INFO` targets against the
//!   map, re-syncing the table once — whose new node can refuse its
//!   connections' setup, so that re-sync cannot complete, and whose replies
//!   can be held back, to carry startup's read past its deadline;
//! - a 3-shard cluster, when `LIVE_EDIT_TEST_CLUSTER_URL` is set (skipped
//!   otherwise), and, in CI's `test-cluster-disruptive:` job, one whose slot
//!   moves under a read's stale routing table (`test-fault-injection`).
//!
//! Startup reads the store once whatever is configured, and the store's
//! memory replaces the budget unless it is explicit (`startup_store_memory`),
//! driven against the proxy, which counts every `INFO memory` it sees. The
//! refresh task runs through its test-only interval seam
//! (`spawn_store_memory_refresh_every_for_test`, about 50 ms instead of
//! 60 s), so the real loop moves the gate, re-records the metrics and logs;
//! [`a_refresh_moves_the_gate_and_the_gc_threshold_and_keeps_the_last_good_value`]
//! adds that the GC loop's threshold follows the gate, and
//! [`upload_admission_follows_a_moved_budget`] that upload admission does.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use fred::interfaces::{ClientLike, ClusterInterface, KeysInterface};
use fred::types::InfoKind;
use futures::FutureExt;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::fmt::MakeWriter;

use live_edit::config::{ExtReleaseBudget, ExtReleaseBudgetSource};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::ext_release_gate::{
    ExtReleaseGate, ExtReleaseRefusal, TotalReaderForTest, EXT_RELEASE_GATE_DEADLINE_MARGIN,
};
use live_edit::storage::external_values::{
    ExtRelease, ExternalValueStore, GcPassState, GcPolicy, ValkeyExternalValueStore,
};
use live_edit::storage::scan_guard::RoutingRefresh;
use live_edit::storage::store_memory::{
    budget_from_reading, memory_bytes_budget, parse_info_maxmemory, read_store_memory,
    refreshed_budget, resolve_startup, spawn_store_memory_refresh_every_for_test,
    startup_store_memory, store_memory_gauge, ClusterMapProblem, ClusterMapProblemKind, InfoUnread,
    MaxmemoryField, PrimariesUnconfirmed, RefreshSeed, ReplconfCalls, StoreMemoryGaugeArm,
    StoreMemoryProblem, StoreMemoryProblemKind, StoreMemoryReading, StoreMemoryRefresh,
};
use live_edit::storage::test_support::{connect_test_client_or_skip, TestValkey};

#[cfg(feature = "test-fault-injection")]
use crate::storage::scan_guard_disruptive_test::{another_primary, move_slot};
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

/// The floor a configuration with neither budget setting answers, here: any
/// value that no store reading divides to.
const FALLBACK: u64 = 8 * 1_048_576;

/// The fallback budget, as `Config::ext_release_budget` answers it with no
/// budget setting.
fn floor() -> ExtReleaseBudget {
    ExtReleaseBudget {
        bytes: FALLBACK,
        source: ExtReleaseBudgetSource::Fallback,
    }
}

/// Startup's store read with neither `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` nor
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES` set: the budget and the problem.
async fn discover(client: &ValkeyClient) -> (ExtReleaseBudget, Option<StoreMemoryProblem>) {
    let resolution = startup_store_memory(client, floor(), None).await;
    (resolution.budget, resolution.problem)
}

/// 3 GiB, Valkey's `maxmemory 3072mb`.
const THREE_GIB: u64 = 3_221_225_472;

/// A Lua body answering an `INFO memory` reply whose `maxmemory` line is
/// `maxmemory_line` (or absent, for `None`), framed as Valkey frames one,
/// with `used_memory:1048576`.
fn info_reply(maxmemory_line: Option<&str>) -> ProxyAction {
    memory_reply(maxmemory_line, 1_048_576)
}

/// [`info_reply`] with `used_memory:{used}`, beside its `used_memory_human`
/// and `used_memory_peak` neighbours.
fn memory_reply(maxmemory_line: Option<&str>, used: u64) -> ProxyAction {
    let line = maxmemory_line.map_or(String::new(), |line| format!("{line}\\r\\n"));
    ProxyAction::Eval(format!(
        "return '# Memory\\r\\nused_memory:{used}\\r\\nused_memory_human:1.00M\\r\\n\
         used_memory_peak:99999999999\\r\\n{line}maxmemory_policy:noeviction\\r\\n'"
    ))
}

/// A proxy whose `INFO memory` replies are whatever `staged` holds when each
/// arrives; every other frame reaches the store unchanged.
async fn staged_store(staged: Arc<Mutex<ProxyAction>>) -> (ScriptReplyProxy, ValkeyClient) {
    let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
        let is_info_memory = args.len() == 2
            && args[0].eq_ignore_ascii_case(b"INFO")
            && args[1].eq_ignore_ascii_case(b"memory");
        is_info_memory.then(|| staged.lock().unwrap().clone())
    });
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;
    (proxy, client)
}

async fn staged_once(action: ProxyAction) -> (ScriptReplyProxy, ValkeyClient) {
    staged_store(Arc::new(Mutex::new(action))).await
}

/// Standalone, unstaged: the shared test store's own `maxmemory`, whatever it
/// is, decides — a limit above zero gives one eighth of it, and 0 (the usual
/// test store, which sets no limit) keeps the floor with the reason, never
/// "unlimited". The expectation is read from the store's own `INFO memory`
/// first, so the test holds on either kind of store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_test_stores_own_maxmemory_decides_only_when_above_zero() {
    let tv = TestValkey::setup().await;
    let info: String = tv
        .client()
        .pool
        .next()
        .info(Some(InfoKind::Memory))
        .await
        .expect("INFO memory");
    let own = match parse_info_maxmemory(&info) {
        MaxmemoryField::Value(bytes) => bytes,
        other => panic!("the test store reports a maxmemory; got {other:?}"),
    };

    let reading = read_store_memory(tv.client(), None).await.reading;
    assert!(
        reading.primaries().is_some(),
        "the read succeeds: {reading:?}"
    );
    let (budget, reason) = discover(tv.client()).await;

    if own == 0 {
        let Err(why) = reading.maxmemory_limit() else {
            panic!("a store at maxmemory 0 is unusable; got {reading:?}");
        };
        assert!(
            why.to_string()
                .starts_with("the store reports maxmemory 0 (no memory limit configured)"),
            "a standalone store is named as itself, with no \"primary \" prefix: {why}"
        );
        assert_eq!(
            (budget.bytes, budget.source),
            (FALLBACK, ExtReleaseBudgetSource::Fallback),
            "0 keeps the floor"
        );
        assert!(reason.is_some_and(|r| r.to_string().contains("reports maxmemory 0")));
    } else {
        assert_eq!(reading.maxmemory_limit(), Ok((own, 1)));
        assert_eq!(budget.bytes, (own / 8).max(1));
        assert_eq!(budget.source.label(), "info");
        assert_eq!(reason, None);
    }
}

/// Every primary reporting a limit: one eighth of it, under the `info`
/// source, with no reason — and recorded on
/// `live_edit_ext_release_budget_bytes` under `info` alone, as startup
/// records it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_limit_gives_one_eighth_under_the_info_source() {
    let (proxy, client) = staged_once(info_reply(Some(&format!("maxmemory:{THREE_GIB}")))).await;

    let (budget, reason) = discover(&client).await;

    assert!(
        proxy.hits() >= 1,
        "precondition: the INFO memory read was staged"
    );
    assert_eq!(budget.bytes, THREE_GIB / 8);
    assert_eq!(
        budget.source,
        ExtReleaseBudgetSource::StoreInfo {
            min_maxmemory_bytes: THREE_GIB,
            primaries: 1
        }
    );
    assert_eq!(reason, None);

    let metrics = crate::common::metrics_scrape_only("b-yuo-store-memory");
    metrics.record_ext_release_budget(budget.source.label(), budget.bytes);
    let text = render(&metrics);
    let cell = |source: &str| {
        sample_value(
            &text,
            "live_edit_ext_release_budget_bytes",
            &[("source", source)],
        )
        .unwrap_or_else(|| panic!("source={source} is published"))
    };
    assert_eq!(
        (cell("info"), cell("override"), cell("fallback")),
        ((THREE_GIB / 8) as f64, 0.0, 0.0),
        "exactly the info source carries the budget"
    );
}

/// A primary reporting 0, no `maxmemory` field (a managed store that does not
/// expose it), or a value that is not a byte count: the floor, with a reason
/// saying what the store (standalone here, so named as "the store") reported
/// — never 0 and never unlimited.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unusable_maxmemory_keeps_the_floor_with_the_reason() {
    for (line, why) in [
        (Some("maxmemory:0"), "reports maxmemory 0"),
        (None, "reports no maxmemory field"),
        (Some("maxmemory:3gb"), "not a number of bytes"),
    ] {
        let (proxy, client) = staged_once(info_reply(line)).await;
        let (budget, reason) = discover(&client).await;
        assert!(proxy.hits() >= 1, "{line:?}: precondition: staged");
        assert_eq!(
            (budget.bytes, budget.source),
            (FALLBACK, ExtReleaseBudgetSource::Fallback),
            "{line:?}: the floor stands"
        );
        let reason = reason
            .unwrap_or_else(|| panic!("{line:?}: the fallback names why"))
            .to_string();
        assert!(
            reason.starts_with("the store ") && reason.contains(why),
            "{line:?}: {reason}"
        );
    }
}

/// An `INFO` the store refuses (an ACL user denied it) keeps the floor, with
/// the error as the reason.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_info_read_keeps_the_floor_with_the_error() {
    let (proxy, client) = staged_once(ProxyAction::Eval(error_reply_script(
        "NOPERM User default has no permissions to run the 'info' command",
    )))
    .await;

    let (budget, reason) = discover(&client).await;

    assert!(proxy.hits() >= 1, "precondition: staged");
    assert_eq!(budget.source, ExtReleaseBudgetSource::Fallback);
    assert_eq!(budget.bytes, FALLBACK);
    let reason = reason.expect("the fallback names why").to_string();
    assert!(
        reason.starts_with("the INFO memory read failed: ") && reason.contains("NOPERM"),
        "{reason}"
    );
}

/// The refresh's composition: a store resized in place moves the gate's
/// budget `H` — and the GC loop's pressure threshold `T = H / 2`, read off
/// the gate every pass — and a later reading of 0, or a failed one, keeps
/// the last good value instead of stepping down to the floor.
///
/// The ledger holds 3 000 released bytes, aged under a minute, with an
/// hour's age floor and no pressure floor: below `T` nothing is collected;
/// at or above it the pass collects.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refresh_moves_the_gate_and_the_gc_threshold_and_keeps_the_last_good_value() {
    let tv = TestValkey::setup().await;
    let staged = Arc::new(Mutex::new(info_reply(Some("maxmemory:80000"))));
    let (proxy, info_client) = staged_store(Arc::clone(&staged)).await;

    let (current, _) = discover(&info_client).await;
    assert_eq!(current.bytes, 10_000, "80 000 / 8");
    let gate = Arc::new(ExtReleaseGate::new(
        tv.client().clone(),
        tv.key_schema(),
        current.bytes,
        Duration::from_secs(30),
        None,
    ));
    let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_ext_release_gate(Arc::clone(&gate));
    let h = [0x5B; 32];
    store
        .store_values(&[(&h, vec![0x5a; 3_000].as_slice())])
        .await
        .expect("store");
    store
        .schedule_release(&[ExtRelease::retirement(h, 9_900)])
        .await
        .expect("release");
    let policy = GcPolicy {
        min_age: Duration::from_secs(3600),
        pressure_min_age: Duration::ZERO,
        ..GcPolicy::IMMEDIATE
    };
    let pass = || async {
        store
            .process_pending(&policy, &mut GcPassState::new())
            .await
            .expect("a GC pass")
    };

    assert_eq!(
        pass().await.processed,
        0,
        "3 000 bytes are below T = 5 000: the hour's floor applies"
    );

    // The store shrinks: 32 000 / 8 = 4 000, T = 2 000.
    *staged.lock().unwrap() = info_reply(Some("maxmemory:32000"));
    let next = refreshed_budget(
        &current,
        &read_store_memory(&info_client, None).await.reading,
        None,
    )
    .expect("a changed limit moves the budget");
    gate.set_budget_bytes(next.bytes);
    assert_eq!(gate.budget_bytes(), 4_000);
    assert_eq!(
        pass().await.processed,
        1,
        "3 000 bytes are at or above the new T = 2 000: the pass collects"
    );

    // A reading of 0, then a failed one: the gate keeps 4 000.
    for action in [
        info_reply(Some("maxmemory:0")),
        ProxyAction::Eval(error_reply_script("ERR staged failure")),
    ] {
        *staged.lock().unwrap() = action;
        let reading = read_store_memory(&info_client, None).await.reading;
        assert_eq!(
            refreshed_budget(&next, &reading, None),
            None,
            "nothing usable keeps the last good budget: {reading:?}"
        );
    }
    assert_eq!(gate.budget_bytes(), 4_000);
    assert!(proxy.hits() >= 4, "precondition: every read was staged");
}

/// Upload admission follows a moved budget, as the GC threshold does: the
/// gate's `admit` — what CreateLayer / UpdateLayer call for an upload
/// carrying external values — admits at once while the release total is
/// below `H`; once a refresh lowers `H` below the total it waits and is
/// refused `Full` when the wait runs out; and a refresh that raises `H`
/// again admits a call already waiting, on the poller's next reading, with
/// the total unchanged.
///
/// The total is planted on `pending:total` (5 000 bytes); the gate's RPC
/// deadline is its margin plus 300 ms, so a call with no deadline of its own
/// waits 300 ms — except the last, whose gate waits up to a minute and is
/// admitted long before. That last gate reads the same planted total through
/// a counting reader, so the test moves `H` only once the call is known to be
/// waiting: the gate's shared poller, which only a waiting call starts, has
/// read the total after the call's own first read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upload_admission_follows_a_moved_budget() {
    let tv = TestValkey::setup().await;
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().ext_pending_total(),
            "5000",
            None,
            None,
            false,
        )
        .await
        .expect("SET pending:total");
    let shutdown = CancellationToken::new();
    let gate = Arc::new(ExtReleaseGate::new(
        tv.client().clone(),
        tv.key_schema(),
        10_000,
        EXT_RELEASE_GATE_DEADLINE_MARGIN + Duration::from_millis(300),
        None,
    ));
    assert_eq!(
        gate.admit(None, &shutdown).await,
        Ok(()),
        "5 000 bytes are below H = 10 000"
    );

    gate.set_budget_bytes(4_000);
    assert_eq!(
        gate.admit(None, &shutdown).await,
        Err(ExtReleaseRefusal::Full),
        "lowered below the total, H holds the upload and refuses it when its wait runs out"
    );

    let reads = Arc::new(AtomicUsize::new(0));
    let reader: TotalReaderForTest = {
        let (client, key, reads) = (
            tv.client().clone(),
            tv.key_schema().ext_pending_total(),
            Arc::clone(&reads),
        );
        Arc::new(
            move || -> futures::future::BoxFuture<'static, Option<u64>> {
                let (client, key, reads) = (client.clone(), key.clone(), Arc::clone(&reads));
                Box::pin(async move {
                    let total: Option<u64> = client.pool.get(&key).await.ok().flatten();
                    reads.fetch_add(1, Ordering::SeqCst);
                    total
                })
            },
        )
    };
    let waiting = Arc::new(ExtReleaseGate::with_total_reader_for_test(
        reader,
        4_000,
        EXT_RELEASE_GATE_DEADLINE_MARGIN + Duration::from_secs(60),
        None,
    ));
    let call = tokio::spawn({
        let (waiting, shutdown) = (Arc::clone(&waiting), shutdown.clone());
        async move { waiting.admit(None, &shutdown).await }
    });
    // Evidence of waiting: a second read is the shared poller's, and only a
    // call that read 5 000 at H = 4 000 and joined the wait starts it. The
    // sleep is the probe interval, not a coordination primitive.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while reads.load(Ordering::SeqCst) < 2 {
        assert!(
            !call.is_finished(),
            "the call ended before it waited at H = 4 000"
        );
        assert!(
            tokio::time::Instant::now() < deadline,
            "the call never waited at H = 4 000"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    waiting.set_budget_bytes(8_000);
    let admitted = tokio::time::timeout(Duration::from_secs(30), call)
        .await
        .expect("the waiting call ends once H is raised")
        .expect("the call did not panic");
    assert_eq!(
        admitted,
        Ok(()),
        "raised above the total, H admits the waiting upload"
    );
}

/// Startup reads the store once whatever is configured (b.qhk). With
/// `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` set, that budget stands — the staged
/// store, which answers 3 GiB, never replaces it — and the read gives the
/// fullness ratio; since that store reports its limit, the refresh runs.
/// Without it, the store's `maxmemory` sets the budget under `info`, whether
/// or not `LIVE_EDIT_VALKEY_MEMORY_BYTES` is set: a store-reported limit wins
/// over the setting. Such a store is never sent `INFO commandstats`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn startup_reads_the_store_whatever_is_configured_and_its_limit_wins() {
    let (proxy, client) = staged_once(info_reply(Some(&format!("maxmemory:{THREE_GIB}")))).await;
    let config_with = |max_bytes: Option<u64>, memory: Option<u64>| {
        let mut cfg = crate::common::test_config();
        cfg.ext_release_max_bytes = max_bytes;
        cfg.valkey_memory_bytes = memory;
        cfg
    };
    // The staged reply's used_memory over its maxmemory.
    let ratio = 1_048_576.0 / THREE_GIB as f64;

    for (cfg, what) in [
        (config_with(Some(123_456), None), "an explicit budget"),
        (
            config_with(Some(123_456), Some(12_000_000_000)),
            "an explicit budget and the store total",
        ),
    ] {
        let from = proxy.hits();
        let configured = cfg.ext_release_budget();
        let resolution = startup_store_memory(&client, configured, cfg.valkey_memory_bytes).await;
        assert!(proxy.hits() > from, "{what}: the store is read once");
        assert_eq!(resolution.budget, configured, "{what}: the budget stands");
        assert_eq!(resolution.budget.source.label(), "override", "{what}");
        assert_eq!(resolution.used_ratio, Some(ratio), "{what}");
        assert!(resolution.refresh_needed, "{what}: the gauge needs it");
    }

    for (cfg, what) in [
        (config_with(None, None), "no setting"),
        (
            config_with(None, Some(12_000_000_000)),
            "the store total set",
        ),
    ] {
        let resolution =
            startup_store_memory(&client, cfg.ext_release_budget(), cfg.valkey_memory_bytes).await;
        assert_eq!(
            (
                resolution.budget.bytes,
                resolution.budget.source.label(),
                resolution.problem
            ),
            (THREE_GIB / 8, "info", None),
            "{what}"
        );
        assert_eq!(resolution.used_ratio, Some(ratio), "{what}");
    }

    // M5: `INFO commandstats` belongs to the store-total arm only; a store
    // that reports its limit is never asked for it, the setting set or not.
    let reading = read_store_memory(&client, Some(DEV_TOTAL)).await.reading;
    assert!(
        matches!(
            reading,
            StoreMemoryReading::Read {
                replconf_calls: ReplconfCalls::NotRead,
                ..
            }
        ),
        "a store limit: no INFO commandstats; got {reading:?}"
    );
}

/// A capturing `MakeWriter` for the refresh task's log lines.
#[derive(Clone)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl<'a> MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Wait until `proxy` has staged at least `reads` more `INFO memory` replies
/// than `from`.
async fn wait_for_reads(proxy: &ScriptReplyProxy, from: usize, reads: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while proxy.hits() < from + reads {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the refresh task stopped reading the store"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The refresh task itself, at a 50 ms interval: a store resized in place
/// moves the gate's budget and re-records `live_edit_ext_release_budget_bytes`
/// under `info`, with one INFO line; then readings of 0, and failed readings,
/// keep the last good budget, and each kind of problem WARNs once and logs
/// DEBUG on its repeats.
///
/// Current-thread runtime with a thread-local subscriber: the task is
/// spawned, so it is polled on the thread the capture is installed on.
/// Each problem phase waits for four reads after the switch, so at least
/// three of them saw the new reply even if one was already in flight.
#[tokio::test]
async fn the_refresh_task_follows_the_store_and_keeps_the_last_good_value() {
    const WARN_TEXT: &str = "store memory read gave nothing usable: keeping the last good";
    const DEBUG_TEXT: &str =
        "store memory re-read gave nothing usable again: keeping the last good";
    let tv = TestValkey::setup().await;
    let staged = Arc::new(Mutex::new(info_reply(Some("maxmemory:80000"))));
    let (proxy, info_client) = staged_store(Arc::clone(&staged)).await;
    let captured = Arc::new(Mutex::new(Vec::new()));
    let _capture = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_writer(CapturingWriter(Arc::clone(&captured)))
            .with_env_filter(tracing_subscriber::EnvFilter::new(
                "live_edit::storage::store_memory=debug",
            ))
            .with_ansi(false)
            .finish(),
    );

    let (initial, _) = discover(&info_client).await;
    assert_eq!(initial.bytes, 10_000, "precondition: 80 000 / 8");
    let metrics = Arc::new(crate::common::metrics_scrape_only("b-yuo-refresh"));
    let gate = Arc::new(ExtReleaseGate::new(
        tv.client().clone(),
        tv.key_schema(),
        initial.bytes,
        Duration::from_secs(30),
        None,
    ));
    let cell = |source: &str| {
        sample_value(
            &render(&metrics),
            "live_edit_ext_release_budget_bytes",
            &[("source", source)],
        )
        .unwrap_or(0.0)
    };

    *staged.lock().unwrap() = info_reply(Some("maxmemory:32000"));
    let cancel = CancellationToken::new();
    let task = spawn_store_memory_refresh_every_for_test(
        info_client.clone(),
        StoreMemoryRefresh {
            gate: Some(Arc::clone(&gate)),
            initial,
            memory_bytes: None,
            seed: RefreshSeed::default(),
            metrics: Some(Arc::clone(&metrics)),
        },
        cancel.clone(),
        Duration::from_millis(50),
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while gate.budget_bytes() != 4_000 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the refresh never moved the gate"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        (cell("info"), cell("fallback"), cell("override")),
        (4_000.0, 0.0, 0.0),
        "the refresh re-records the budget under info"
    );

    for action in [
        info_reply(Some("maxmemory:0")),
        ProxyAction::Eval(error_reply_script("ERR staged failure")),
    ] {
        *staged.lock().unwrap() = action;
        let from = proxy.hits();
        wait_for_reads(&proxy, from, 4).await;
        assert_eq!(gate.budget_bytes(), 4_000, "the last good budget stands");
        assert_eq!(cell("info"), 4_000.0);
    }
    cancel.cancel();
    task.await.expect("the refresh task ends on cancel");

    let output = String::from_utf8(captured.lock().unwrap().clone()).expect("UTF-8 log output");
    let lines = |level: &str, text: &str, about: &str| {
        output
            .lines()
            .filter(|line| line.contains(level) && line.contains(text) && line.contains(about))
            .count()
    };
    assert_eq!(
        output
            .lines()
            .filter(|l| {
                l.contains(" INFO ") && l.contains("external-value release budget re-read")
            })
            .count(),
        1,
        "one INFO for the one change; got {output}"
    );
    for about in ["maxmemory 0", "the INFO memory read failed"] {
        assert_eq!(
            lines(" WARN ", WARN_TEXT, about),
            1,
            "{about}: one WARN for the first of its kind; got {output}"
        );
        assert!(
            lines(" DEBUG ", DEBUG_TEXT, about) >= 1,
            "{about}: its repeats log at DEBUG; got {output}"
        );
    }
    drop(proxy);
}

/// b.1jt: a store that does not report its size (no `maxmemory` field —
/// Azure Managed Redis) is the expected case, not a fault. Startup logs it
/// once, at INFO, and the refresh task — seeded with the startup problem's
/// kind — logs nothing more for it however many re-reads say the same, and
/// never WARNs. A fault (`maxmemory:0`, then an `INFO` an ACL denies) WARNs
/// once per kind as it appears, and when the store goes back to not
/// reporting its size, that change of reason is logged again at its own
/// level, INFO.
///
/// Current-thread runtime with a thread-local subscriber, as above: the
/// refresh task is spawned, so it is polled on the thread the capture is
/// installed on. Each phase waits for four re-reads after its switch.
///
/// **On the build before b.1jt** every unusable reason logged WARN, and the
/// refresh task started with no remembered problem, so a store with no
/// `maxmemory` field WARNed at startup and again at the first re-read.
#[tokio::test]
async fn a_store_that_does_not_report_its_size_logs_one_info_and_never_warns() {
    const ABSENT_TEXT: &str = "does not report its size";
    let staged = Arc::new(Mutex::new(info_reply(None)));
    let (proxy, info_client) = staged_store(Arc::clone(&staged)).await;
    let tv = TestValkey::setup().await;
    let captured = Arc::new(Mutex::new(Vec::new()));
    let _capture = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_writer(CapturingWriter(Arc::clone(&captured)))
            .with_env_filter(tracing_subscriber::EnvFilter::new(
                "live_edit::storage::store_memory=debug,live_edit::config=info",
            ))
            .with_ansi(false)
            .finish(),
    );

    let config = crate::common::test_config();
    let configured = config.ext_release_budget();
    assert!(
        configured.source == ExtReleaseBudgetSource::Fallback
            && config.valkey_memory_bytes.is_none(),
        "precondition: neither budget setting is set"
    );
    let resolution = startup_store_memory(&info_client, configured, None).await;
    let budget = resolution.budget;
    let seed = resolution.seed();
    let problem = resolution
        .problem
        .expect("a store with no maxmemory field leaves the fallback");
    assert_eq!(problem.kind(), StoreMemoryProblemKind::FieldAbsent);
    config.log_ext_release_budget_startup(&budget, Some(&problem));
    let gate = Arc::new(ExtReleaseGate::new(
        tv.client().clone(),
        tv.key_schema(),
        budget.bytes,
        Duration::from_secs(30),
        None,
    ));

    let cancel = CancellationToken::new();
    let task = spawn_store_memory_refresh_every_for_test(
        info_client.clone(),
        StoreMemoryRefresh {
            gate: Some(Arc::clone(&gate)),
            initial: budget,
            memory_bytes: None,
            seed,
            metrics: None,
        },
        cancel.clone(),
        Duration::from_millis(50),
    );
    let log = || String::from_utf8(captured.lock().unwrap().clone()).expect("UTF-8 log output");
    let count = |level: &str, text: &str| {
        log()
            .lines()
            .filter(|line| line.contains(level) && line.contains(text))
            .count()
    };

    wait_for_reads(&proxy, proxy.hits(), 4).await;
    assert_eq!(
        count(" INFO ", ABSENT_TEXT),
        1,
        "the startup INFO alone: the refresh inherits its reason; got {}",
        log()
    );
    assert_eq!(count(" WARN ", ""), 0, "no WARN; got {}", log());

    for (action, about) in [
        (info_reply(Some("maxmemory:0")), "maxmemory 0"),
        (
            ProxyAction::Eval(error_reply_script(
                "NOPERM User default has no permissions to run the 'info' command",
            )),
            "NOPERM",
        ),
    ] {
        *staged.lock().unwrap() = action;
        wait_for_reads(&proxy, proxy.hits(), 4).await;
        assert_eq!(
            count(" WARN ", about),
            1,
            "{about}: a fault WARNs once; got {}",
            log()
        );
    }

    *staged.lock().unwrap() = info_reply(None);
    wait_for_reads(&proxy, proxy.hits(), 4).await;
    cancel.cancel();
    task.await.expect("the refresh task ends on cancel");
    assert_eq!(
        count(" INFO ", ABSENT_TEXT),
        2,
        "the reason changed back, so it is logged again, at INFO; got {}",
        log()
    );
    assert_eq!(count(" WARN ", ABSENT_TEXT), 0, "{}", log());
    assert_eq!(gate.budget_bytes(), budget.bytes, "the fallback stands");
    drop(proxy);
}

// ── b.qhk: the fullness gauge, the store total and the cluster map ─────────

/// Dev's `Balanced_B10`, 12 GB: `LIVE_EDIT_VALKEY_MEMORY_BYTES` as the
/// provider states it.
const DEV_TOTAL: u64 = 12_000_000_000;

/// 80 % of [`DEV_TOTAL`]: the part of it the gauge counts as usable.
const DEV_USABLE: f64 = 9_600_000_000.0;

/// Dev's measured whole-database `used_memory`, high availability on.
const DEV_USED: u64 = 470_333_232;

/// The 24 GB cache measured on 2026-10-13.
const X20_TOTAL: u64 = 24_000_000_000;

/// The interval the refresh tests run the task at, instead of 60 s.
const REFRESH_EVERY: Duration = Duration::from_millis(50);

/// The deadline of every wait for the refresh task, or for a client to
/// connect: unreachable on the healthy path, which takes milliseconds.
const SETTLE: Duration = Duration::from_secs(30);

/// `live_edit_store_memory_used_ratio` as `metrics` renders it now.
fn used_ratio(metrics: &Metrics) -> Option<f64> {
    sample_value(&render(metrics), "live_edit_store_memory_used_ratio", &[])
}

/// Wait until `metrics` renders `live_edit_store_memory_used_ratio` within
/// rounding of `want`, by [`SETTLE`]: the refresh task re-recorded it.
async fn wait_for_ratio(metrics: &Metrics, want: f64, what: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    loop {
        let got = used_ratio(metrics);
        if got.is_some_and(|got| (got - want).abs() < 1e-12) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what}: the gauge stayed at {got:?}, want {want}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Wait until `counter` has counted `more` past `from`, by [`SETTLE`].
async fn wait_for_count(counter: &AtomicUsize, from: usize, more: usize, what: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while counter.load(Ordering::SeqCst) < from + more {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what}: the refresh task stopped reading the store"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// An upload gate at `budget_bytes` whose release total is never read.
fn unread_gate(budget_bytes: u64) -> Arc<ExtReleaseGate> {
    Arc::new(ExtReleaseGate::with_total_reader_for_test(
        Arc::new(|| futures::future::ready(Some(0)).boxed()),
        budget_bytes,
        Duration::from_secs(30),
        None,
    ))
}

fn explicit(bytes: u64) -> ExtReleaseBudget {
    ExtReleaseBudget {
        bytes,
        source: ExtReleaseBudgetSource::Explicit,
    }
}

/// T10: with `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` set, the store is still read
/// every interval for `live_edit_store_memory_used_ratio`, which follows the
/// store's `used_memory`, while the budget stands — and with it set to `0`
/// there is no gate at all, and the gauge is recorded just the same.
///
/// **Fails under the rule before b.qhk**, where the refresh ran only for a
/// budget that was not explicit (seen red with `store_memory_refresh_needed`
/// reduced to that rule).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_an_explicit_budget_the_refresh_still_reads_the_store_for_the_gauge() {
    for budget in [123_456, 0] {
        let staged = Arc::new(Mutex::new(memory_reply(
            Some(&format!("maxmemory:{THREE_GIB}")),
            1_073_741_824,
        )));
        let (proxy, client) = staged_store(Arc::clone(&staged)).await;
        let configured = explicit(budget);
        let resolution = startup_store_memory(&client, configured, None).await;
        assert!(
            resolution.refresh_needed,
            "budget {budget}: the gauge can use the read"
        );
        assert_eq!(resolution.budget, configured, "budget {budget}");
        assert_eq!(resolution.used_ratio, Some(1.0 / 3.0), "budget {budget}");

        // As startup builds it: no gate for a disabled budget.
        let gate = (budget > 0).then(|| unread_gate(budget));
        let metrics = Arc::new(crate::common::metrics_scrape_only("b-qhk-explicit"));
        let cancel = CancellationToken::new();
        let task = spawn_store_memory_refresh_every_for_test(
            client.clone(),
            StoreMemoryRefresh {
                gate: gate.clone(),
                initial: resolution.budget,
                memory_bytes: None,
                seed: resolution.seed(),
                metrics: Some(Arc::clone(&metrics)),
            },
            cancel.clone(),
            REFRESH_EVERY,
        );
        for used in [2_147_483_648_u64, 3_221_225_472 + 1_073_741_824] {
            *staged.lock().unwrap() = memory_reply(Some(&format!("maxmemory:{THREE_GIB}")), used);
            #[allow(clippy::cast_precision_loss)]
            let want = used as f64 / THREE_GIB as f64;
            wait_for_ratio(&metrics, want, &format!("budget {budget}, used {used}")).await;
        }
        cancel.cancel();
        task.await.expect("the refresh task ends on cancel");
        if let Some(gate) = gate {
            assert_eq!(gate.budget_bytes(), budget, "the explicit budget stands");
        }
        assert!(proxy.hits() >= 3, "precondition: every read was staged");
    }
}

/// A standalone store through a proxy that answers like Azure Managed Redis:
/// `INFO memory` with no `maxmemory` field and `used_memory:{used}`; `INFO
/// commandstats` with `cmdstat_replconf` at [`AmrStore::replconf`]'s count
/// (no such line when it is `None`), raised by [`AmrStore::step`] after each
/// read, as replicas acknowledging raise it; and `INFO replication` with the
/// placeholder replica Azure reports whether or not replication runs.
struct AmrStore {
    proxy: ScriptReplyProxy,
    client: ValkeyClient,
    commandstats_reads: Arc<AtomicUsize>,
    replication_reads: Arc<AtomicUsize>,
    replconf: Arc<Mutex<Option<u64>>>,
    step: Arc<AtomicU64>,
}

impl AmrStore {
    async fn start(used: u64, replconf: Option<u64>) -> Self {
        let commandstats_reads = Arc::new(AtomicUsize::new(0));
        let replication_reads = Arc::new(AtomicUsize::new(0));
        let replconf = Arc::new(Mutex::new(replconf));
        let step = Arc::new(AtomicU64::new(0));
        let rule: ProxyRule = {
            let (commandstats_reads, replication_reads, replconf, step) = (
                Arc::clone(&commandstats_reads),
                Arc::clone(&replication_reads),
                Arc::clone(&replconf),
                Arc::clone(&step),
            );
            Arc::new(move |args: &[Vec<u8>]| {
                if args.len() != 2 || !args[0].eq_ignore_ascii_case(b"INFO") {
                    return None;
                }
                let section = args[1].to_ascii_lowercase();
                match section.as_slice() {
                    b"memory" => Some(memory_reply(None, used)),
                    b"commandstats" => {
                        commandstats_reads.fetch_add(1, Ordering::SeqCst);
                        let mut count = replconf.lock().unwrap();
                        let line = count.map_or(String::new(), |calls| {
                            format!(
                                "cmdstat_replconf:calls={calls},usec=2449185,\
                                 usec_per_call=1.54\\r\\r\\n"
                            )
                        });
                        if let Some(calls) = count.as_mut() {
                            *calls += step.load(Ordering::SeqCst);
                        }
                        Some(ProxyAction::Eval(format!(
                            "return '# Commandstats\\r\\r\\n{line}\
                             cmdstat_info:calls=42,usec=100,usec_per_call=2.38\\r\\r\\n'"
                        )))
                    }
                    b"replication" => {
                        replication_reads.fetch_add(1, Ordering::SeqCst);
                        Some(ProxyAction::Eval(
                            "return '# Replication\\r\\r\\nrole:master\\r\\r\\n\
                             connected_slaves:1\\r\\r\\n\
                             slave0:ip=0.0.0.0,port=0,state=online,offset=0,lag=0\\r\\r\\n'"
                                .to_owned(),
                        ))
                    }
                    _ => None,
                }
            })
        };
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        let client = client_through(&proxy).await;
        Self {
            proxy,
            client,
            commandstats_reads,
            replication_reads,
            replconf,
            step,
        }
    }

    /// Spawn the refresh task on this store, seeded as startup seeds it, at
    /// [`REFRESH_EVERY`], recording on `metrics`.
    fn refresh(
        &self,
        initial: ExtReleaseBudget,
        seed: RefreshSeed,
        metrics: &Arc<Metrics>,
        cancel: &CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        spawn_store_memory_refresh_every_for_test(
            self.client.clone(),
            StoreMemoryRefresh {
                gate: Some(unread_gate(initial.bytes)),
                initial,
                memory_bytes: Some(DEV_TOTAL),
                seed,
                metrics: Some(Arc::clone(metrics)),
            },
            cancel.clone(),
            REFRESH_EVERY,
        )
    }
}

/// T10 (T6's first read, second read): on a store with no `maxmemory` field
/// and `LIVE_EDIT_VALKEY_MEMORY_BYTES` set, startup reads the replica
/// acknowledgement count (100) and has no figure yet; the budget is one
/// shard's share of the total (a store that is not clustered is one shard);
/// once the count rises (160, then on by 60 a read) the refresh records the
/// whole-database `used_memory` ÷ 2 ÷ 80 % of the setting, and keeps it while
/// the count keeps rising. `INFO replication` is never asked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_a_store_without_maxmemory_the_gauge_appears_at_the_second_read_with_two_copies() {
    let store = AmrStore::start(DEV_USED, Some(100)).await;
    let configured = crate::common::test_config().ext_release_budget();

    let startup = startup_store_memory(&store.client, configured, Some(DEV_TOTAL)).await;

    assert!(
        store.commandstats_reads.load(Ordering::SeqCst) >= 1,
        "precondition: startup read the count"
    );
    assert_eq!(
        (startup.gauge_arm, startup.used_ratio),
        (StoreMemoryGaugeArm::StoreTotalPending, None),
        "absent at the first read"
    );
    assert_eq!(startup.seed().replconf_calls, Some(100));
    assert_eq!(
        startup.budget.bytes, 1_200_000_000,
        "12 GB x 0.8 / 8, one shard"
    );
    assert_eq!(
        startup
            .budget
            .source
            .cluster_map()
            .map(|(d, e)| (d.shards, e)),
        Some((1, 1))
    );
    assert_eq!(startup.budget.source.label(), "override");

    *store.replconf.lock().unwrap() = Some(160);
    store.step.store(60, Ordering::SeqCst);
    let metrics = Arc::new(crate::common::metrics_scrape_only("b-qhk-amr"));
    let cancel = CancellationToken::new();
    let task = store.refresh(startup.budget, startup.seed(), &metrics, &cancel);
    #[allow(clippy::cast_precision_loss)]
    let two_copies = DEV_USED as f64 / 2.0 / DEV_USABLE;
    wait_for_ratio(&metrics, two_copies, "the count rose").await;
    let from = store.commandstats_reads.load(Ordering::SeqCst);
    wait_for_count(&store.commandstats_reads, from, 3, "rising").await;
    assert_eq!(
        used_ratio(&metrics),
        Some(two_copies),
        "a count that keeps rising keeps copies 2"
    );
    cancel.cancel();
    task.await.expect("the refresh task ends on cancel");
    assert_eq!(
        store.replication_reads.load(Ordering::SeqCst),
        0,
        "INFO replication is never sent"
    );
    drop(store.proxy);
}

/// T10, the discriminator against counting `connected_slaves`: a store whose
/// `INFO replication` reports the placeholder replica, but whose
/// commandstats list no `cmdstat_replconf` line (no replica ever
/// acknowledged), reads copies 1 — the whole-database `used_memory` ÷ 80 % of
/// the setting — from the refresh's first pass on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_whose_replicas_never_acknowledge_reads_one_copy() {
    let store = AmrStore::start(DEV_USED, None).await;
    let configured = crate::common::test_config().ext_release_budget();
    let startup = startup_store_memory(&store.client, configured, Some(DEV_TOTAL)).await;
    assert_eq!(
        startup.seed().replconf_calls,
        Some(0),
        "no replconf line reads 0"
    );

    let metrics = Arc::new(crate::common::metrics_scrape_only("b-qhk-amr-one"));
    let cancel = CancellationToken::new();
    let task = store.refresh(startup.budget, startup.seed(), &metrics, &cancel);
    #[allow(clippy::cast_precision_loss)]
    let one_copy = DEV_USED as f64 / DEV_USABLE;
    wait_for_ratio(&metrics, one_copy, "a standing count").await;
    cancel.cancel();
    task.await.expect("the refresh task ends on cancel");
    assert_eq!(
        store.replication_reads.load(Ordering::SeqCst),
        0,
        "INFO replication is never sent"
    );
    drop(store.proxy);
}

/// T10 (PD3): with `LIVE_EDIT_VALKEY_MEMORY_BYTES` set, an `INFO` the store
/// refuses at startup does not stop the budget the setting gives over the
/// map: startup starts on it, under `override`, naming the failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_info_with_the_store_total_set_starts_on_the_map() {
    let (proxy, client) = staged_once(ProxyAction::Eval(error_reply_script(
        "NOPERM User default has no permissions to run the 'info' command",
    )))
    .await;

    let resolution = startup_store_memory(&client, floor(), Some(DEV_TOTAL)).await;

    assert!(proxy.hits() >= 1, "precondition: staged");
    assert_eq!(
        (resolution.budget.bytes, resolution.budget.source.label()),
        (1_200_000_000, "override")
    );
    assert_eq!(
        resolution.problem.map(|p| p.kind()),
        Some(StoreMemoryProblemKind::ReadFailed)
    );
    assert_eq!(resolution.used_ratio, None, "no figure from a failed read");
}

/// The node id the map facade's `CLUSTER SLOTS` reply names for itself.
const MAP_FACADE_NODE_ID: &str = "b0qhk00000000000000000000000000000map000";

/// A [`ScriptReplyProxy`] in front of the test store that plays a cluster
/// whose every slot it owns: `CLUSTER SLOTS` answers the ranges `map` holds
/// when it arrives, each its own entry and every entry on the proxy itself —
/// one endpoint fronting several shards, as an Azure Managed Redis node does
/// — or an error while `slots_fail` is set, or, while `slots_malformed` is
/// set, a reply whose second entry names no primary; `INFO memory` answers
/// `used_memory:470333232` and no `maxmemory` field, or `maxmemory:<n>` while
/// `maxmemory` holds `Some(n)`; `INFO commandstats` lists no
/// `cmdstat_replconf` line. Counts the `CLUSTER SLOTS`, the `INFO memory`
/// and the `INFO commandstats` it receives.
struct MapFacade {
    proxy: ScriptReplyProxy,
    map: Arc<Mutex<Vec<(u16, u16)>>>,
    slots_fail: Arc<AtomicBool>,
    slots_malformed: Arc<AtomicBool>,
    maxmemory: Arc<Mutex<Option<u64>>>,
    slots_reads: Arc<AtomicUsize>,
    info_memory_reads: Arc<AtomicUsize>,
    commandstats_reads: Arc<AtomicUsize>,
}

impl MapFacade {
    async fn start(map: Vec<(u16, u16)>) -> Self {
        let port_slot: Arc<OnceLock<u16>> = Arc::new(OnceLock::new());
        let map = Arc::new(Mutex::new(map));
        let slots_fail = Arc::new(AtomicBool::new(false));
        let slots_malformed = Arc::new(AtomicBool::new(false));
        let maxmemory = Arc::new(Mutex::new(None::<u64>));
        let slots_reads = Arc::new(AtomicUsize::new(0));
        let info_memory_reads = Arc::new(AtomicUsize::new(0));
        let commandstats_reads = Arc::new(AtomicUsize::new(0));
        let rule: ProxyRule = {
            let (port_slot, map, slots_fail, slots_reads) = (
                Arc::clone(&port_slot),
                Arc::clone(&map),
                Arc::clone(&slots_fail),
                Arc::clone(&slots_reads),
            );
            let (slots_malformed, maxmemory, info_memory_reads, commandstats_reads) = (
                Arc::clone(&slots_malformed),
                Arc::clone(&maxmemory),
                Arc::clone(&info_memory_reads),
                Arc::clone(&commandstats_reads),
            );
            Arc::new(move |args: &[Vec<u8>]| {
                let port = *port_slot
                    .get()
                    .expect("the proxy port is recorded before any client connects");
                let name = args.first()?.to_ascii_uppercase();
                let sub = args.get(1).map(|a| a.to_ascii_lowercase());
                match (name.as_slice(), sub.as_deref()) {
                    (b"CLUSTER", Some(b"slots")) => {
                        slots_reads.fetch_add(1, Ordering::SeqCst);
                        if slots_fail.load(Ordering::SeqCst) {
                            return Some(ProxyAction::Eval(error_reply_script(
                                "ERR staged CLUSTER SLOTS failure",
                            )));
                        }
                        if slots_malformed.load(Ordering::SeqCst) {
                            return Some(ProxyAction::Eval(format!(
                                "return {{{{0, 8191, {{'127.0.0.1', {port}, \
                                 '{MAP_FACADE_NODE_ID}'}}}}, {{8192, 16383}}}}"
                            )));
                        }
                        let entries = map
                            .lock()
                            .unwrap()
                            .iter()
                            .map(|(start, end)| {
                                format!(
                                    "{{{start}, {end}, {{'127.0.0.1', {port}, \
                                     '{MAP_FACADE_NODE_ID}'}}}}"
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        Some(ProxyAction::Eval(format!("return {{{entries}}}")))
                    }
                    (b"CLUSTER", Some(b"info")) => Some(ProxyAction::Eval(
                        "return 'cluster_state:ok\\r\\ncluster_slots_assigned:16384\\r\\n\
                         cluster_known_nodes:1\\r\\ncluster_size:1\\r\\n'"
                            .to_owned(),
                    )),
                    (b"INFO", Some(b"memory")) => {
                        info_memory_reads.fetch_add(1, Ordering::SeqCst);
                        let line = maxmemory.lock().unwrap().map(|n| format!("maxmemory:{n}"));
                        Some(memory_reply(line.as_deref(), DEV_USED))
                    }
                    (b"INFO", Some(b"commandstats")) => {
                        commandstats_reads.fetch_add(1, Ordering::SeqCst);
                        Some(ProxyAction::Eval(
                            "return '# Commandstats\\r\\r\\n\
                             cmdstat_info:calls=42,usec=100,usec_per_call=2.38\\r\\r\\n'"
                                .to_owned(),
                        ))
                    }
                    _ => None,
                }
            })
        };
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
        port_slot
            .set(proxy.port())
            .expect("the port is recorded exactly once");
        Self {
            proxy,
            map,
            slots_fail,
            slots_malformed,
            maxmemory,
            slots_reads,
            info_memory_reads,
            commandstats_reads,
        }
    }

    fn info_memory_reads(&self) -> usize {
        self.info_memory_reads.load(Ordering::SeqCst)
    }

    /// A [`TestValkey`] connected through the facade in cluster mode.
    async fn test_valkey(&self) -> TestValkey {
        tokio::time::timeout(
            SETTLE,
            TestValkey::setup_with_url(
                format!("redis-cluster://127.0.0.1:{}", self.proxy.port()),
                1,
            ),
        )
        .await
        .expect("a client must connect through the map facade")
    }

    fn slots_reads(&self) -> usize {
        self.slots_reads.load(Ordering::SeqCst)
    }
}

/// 8 entries of 2,048 slots: the 24 GB cache measured on 2026-10-13.
fn eight_shards() -> Vec<(u16, u16)> {
    (0..8u16).map(|i| (i * 2048, i * 2048 + 2047)).collect()
}

/// 2 entries of 8,192 slots: dev's `Balanced_B10`.
fn two_shards() -> Vec<(u16, u16)> {
    vec![(0, 8191), (8192, 16383)]
}

/// The slot ranges and endpoints of a read that got its map.
fn map_of(reading: &StoreMemoryReading) -> (Vec<(u16, u16)>, usize) {
    match reading {
        StoreMemoryReading::Read {
            slot_ranges,
            endpoints,
            ..
        } => (slot_ranges.clone(), *endpoints),
        other => panic!("a read with a map; got {other:?}"),
    }
}

/// Why a reading has no `INFO memory` answers, when it is that its pass
/// could not confirm its targets (b.xb7); `None` for any other reading.
fn unconfirmed(reading: &StoreMemoryReading) -> Option<&PrimariesUnconfirmed> {
    match reading {
        StoreMemoryReading::Read {
            info: Err(InfoUnread::Unconfirmed(cause)),
            ..
        } => Some(cause),
        _ => None,
    }
}

/// T5 (PD4), I4, b.xb7: each read on a cluster sends its own `CLUSTER SLOTS`,
/// whatever is configured, and takes the map from the reply, so a store
/// resharded from 2 entries to 8 since the client connected is seen at once —
/// 8 shards on 1 endpoint, the budget one eighth of an eighth of 80 % of
/// 24 GB, never divided by the endpoint. The 8 entries still sit on the one
/// endpoint the client's routing table lists, so the table divides the slots
/// as the map does: the read is confirmed and its `INFO` answered, the setting
/// set or not.
///
/// **Fails if the read took the client's cached routing table instead**: the
/// reads still see 2 entries, and the budget is 1,200,000,000. **Fails on the
/// build before b.xb7** (stated, not run), which sent no `CLUSTER SLOTS` with
/// the setting unset and so carried no ranges.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_read_sends_its_own_cluster_slots_whatever_is_configured() {
    let facade = MapFacade::start(two_shards()).await;
    let tv = facade.test_valkey().await;
    let first = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
    assert_eq!(
        (map_of(&first.reading), first.map_problem),
        ((two_shards(), 1), None),
        "precondition: 2 entries on the one endpoint"
    );

    *facade.map.lock().unwrap() = eight_shards();
    let before = facade.slots_reads();
    let unset = read_store_memory(tv.client(), None).await;
    assert_eq!(
        (
            facade.slots_reads() - before,
            map_of(&unset.reading),
            unset.map_problem,
            unset.reading.primaries().map(<[_]>::len),
        ),
        (1, (eight_shards(), 1), None, Some(1)),
        "the setting unset: (CLUSTER SLOTS sent, map, map problem, INFO memory answered)"
    );

    let before = facade.slots_reads();
    let read = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
    assert_eq!(
        facade.slots_reads() - before,
        1,
        "the setting set: the read sends one CLUSTER SLOTS"
    );
    assert_eq!(read.map_problem, None);
    assert_eq!(map_of(&read.reading), (eight_shards(), 1));
    let budget = budget_from_reading(&read.reading, Some(X20_TOTAL))
        .expect("a store with no maxmemory field and a map takes rule 3");
    assert_eq!(
        (
            budget.bytes,
            budget.source.cluster_map().map(|(d, e)| (d.shards, e))
        ),
        (300_000_000, Some((8, 1)))
    );
    drop(tv);
    drop(facade.proxy);
}

/// T5 (PD4), b.xb7: a `CLUSTER SLOTS` the store keeps refusing leaves the
/// pass with no map — the failure reported, no ranges, no budget from rule 3
/// — and, with no map to confirm its targets against, no `INFO` at all: its
/// primaries are unconfirmed. The `CLUSTER SLOTS` is sent once, as an
/// ordinary command, so it never holds up the shared maintenance client the
/// way fred's `sync_cluster` would, retrying inside the client's router for
/// as long as the store refuses: a command sent on that client right after
/// is answered. The next read once the store answers gets the map, and its
/// `INFO`.
///
/// **Fails with the map read through `sync_cluster`**: fred retries the
/// refused `CLUSTER SLOTS` inside the client's router (more than one reaches
/// the store), and the `PING` queued behind that sync times out at the
/// client's command timeout. Both are asserted as outcomes — a count and a
/// failed command — never as elapsed time; [`SETTLE`] is only a hang bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_cluster_slots_leaves_no_map_and_does_not_hold_up_the_client() {
    let facade = MapFacade::start(two_shards()).await;
    let tv = facade.test_valkey().await;
    *facade.map.lock().unwrap() = eight_shards();
    facade.slots_fail.store(true, Ordering::SeqCst);
    let (slots_before, info_before) = (facade.slots_reads(), facade.info_memory_reads());

    let failed = tokio::time::timeout(SETTLE, read_store_memory(tv.client(), Some(X20_TOTAL)))
        .await
        .expect("the read ends");
    let (maintenance, _token) = tv.client().acquire_maintenance();
    let follow_up: Result<String, _> = tokio::time::timeout(SETTLE, maintenance.next().ping(None))
        .await
        .expect("the PING ends");

    assert_eq!(
        (
            facade.slots_reads() - slots_before,
            facade.info_memory_reads() - info_before,
            follow_up.map_err(|e| e.to_string()),
        ),
        (1, 0, Ok("PONG".to_owned())),
        "(CLUSTER SLOTS sent, INFO memory sent, the next command on that client): sent \
         once, not retried, no INFO, and nothing held up; got {failed:?}"
    );
    assert_eq!(
        failed.map_problem.as_ref().map(ClusterMapProblem::kind),
        Some(ClusterMapProblemKind::ReadFailed),
        "the failure is reported: {failed:?}"
    );
    assert!(
        matches!(
            unconfirmed(&failed.reading),
            Some(PrimariesUnconfirmed::NoMap(ClusterMapProblem::ReadFailed(
                _
            )))
        ),
        "no map, so the primaries are unconfirmed: {failed:?}"
    );
    assert_eq!(map_of(&failed.reading), (Vec::new(), 1), "no map");
    assert_eq!(
        budget_from_reading(&failed.reading, Some(X20_TOTAL)),
        None,
        "no map, no INFO: no budget from the store"
    );

    facade.slots_fail.store(false, Ordering::SeqCst);
    let recovered = tokio::time::timeout(SETTLE, read_store_memory(tv.client(), Some(X20_TOTAL)))
        .await
        .expect("a CLUSTER SLOTS the store answers is bounded");
    assert_eq!(
        (
            map_of(&recovered.reading),
            recovered.map_problem,
            recovered.reading.primaries().map(<[_]>::len),
        ),
        ((eight_shards(), 1), None, Some(1)),
        "the next read sees the resharded map, and its INFO memory is answered"
    );
    drop(tv);
    drop(facade.proxy);
}

/// On a staged cluster whose primary reports a `maxmemory` above zero, with
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES` set, the read still sends its `CLUSTER
/// SLOTS` and keeps the 8-entry map, but the store's own limit wins over the
/// setting: the budget is `info` (one eighth of 3 GiB), the fullness gauge
/// takes the limit arm, and no `INFO commandstats` is sent. CI's real cluster
/// sets no `maxmemory`, so this is where that arm runs on every pipeline.
/// Asserted as one tuple, so a run that breaks several parts shows each.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_a_staged_cluster_reporting_maxmemory_the_limit_wins_over_the_store_total() {
    let facade = MapFacade::start(eight_shards()).await;
    *facade.maxmemory.lock().unwrap() = Some(THREE_GIB);
    let tv = facade.test_valkey().await;
    let slots_before = facade.slots_reads();
    let commandstats_before = facade.commandstats_reads.load(Ordering::SeqCst);

    let read = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
    let resolution = resolve_startup(floor(), &read, Some(X20_TOTAL));

    let not_read = matches!(
        read.reading,
        StoreMemoryReading::Read {
            replconf_calls: ReplconfCalls::NotRead,
            ..
        }
    );
    #[allow(clippy::cast_precision_loss)]
    let ratio = DEV_USED as f64 / THREE_GIB as f64;
    assert_eq!(
        (
            facade.slots_reads() > slots_before,
            map_of(&read.reading),
            resolution.budget,
            resolution.gauge_arm,
            resolution.used_ratio,
            not_read,
            facade.commandstats_reads.load(Ordering::SeqCst) - commandstats_before,
        ),
        (
            true,
            (eight_shards(), 1),
            ExtReleaseBudget {
                bytes: THREE_GIB / 8,
                source: ExtReleaseBudgetSource::StoreInfo {
                    min_maxmemory_bytes: THREE_GIB,
                    primaries: 1,
                },
            },
            StoreMemoryGaugeArm::Maxmemory,
            Some(ratio),
            true,
            0,
        ),
        "(CLUSTER SLOTS sent, map, budget, gauge arm, gauge, replconf not read, \
         INFO commandstats sent)"
    );
    drop(tv);
    drop(facade.proxy);
}

/// A malformed `CLUSTER SLOTS` reply — here a second entry naming no primary
/// — is no map at all, never part of one. At startup the read reports
/// `Malformed` and carries no ranges; with no map to confirm its targets
/// against it sends no `INFO memory` (b.xb7); and the budget is the fallback
/// — not one from the client's cached routing table, which still lists the 2
/// entries it connected with. On a refresh the last good budget (8 entries:
/// 300,000,000) stands, and the malformed map WARNs once however many passes
/// see it.
///
/// Current-thread runtime with a thread-local subscriber: the refresh task is
/// spawned, so it is polled on the thread the capture is installed on.
#[tokio::test]
async fn a_malformed_cluster_slots_reply_is_no_map_and_warns_once() {
    const MALFORMED_WARN: &str = "the cluster map was malformed";
    let facade = MapFacade::start(two_shards()).await;
    let tv = facade.test_valkey().await;
    *facade.map.lock().unwrap() = eight_shards();
    let good = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
    let last_good = resolve_startup(floor(), &good, Some(X20_TOTAL));
    assert_eq!(
        last_good.budget.bytes, 300_000_000,
        "precondition: the 8-entry map"
    );

    facade.slots_malformed.store(true, Ordering::SeqCst);
    let info_before = facade.info_memory_reads();
    let malformed = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
    assert_eq!(malformed.map_problem, Some(ClusterMapProblem::Malformed));
    assert_eq!(map_of(&malformed.reading).0, Vec::new(), "no ranges");
    assert_eq!(
        (
            facade.info_memory_reads() - info_before,
            unconfirmed(&malformed.reading),
        ),
        (
            0,
            Some(&PrimariesUnconfirmed::NoMap(ClusterMapProblem::Malformed))
        ),
        "(INFO memory sent, why none): {malformed:?}"
    );
    let startup = resolve_startup(floor(), &malformed, Some(X20_TOTAL));
    assert_eq!(startup.budget, floor(), "startup on no map: the fallback");
    assert_eq!(
        (startup.seed().problem, startup.seed().map_problem),
        (
            Some(StoreMemoryProblemKind::PrimariesUnconfirmed),
            Some(ClusterMapProblemKind::Malformed)
        )
    );

    let captured = Arc::new(Mutex::new(Vec::new()));
    let _capture = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_writer(CapturingWriter(Arc::clone(&captured)))
            .with_env_filter(tracing_subscriber::EnvFilter::new(
                "live_edit::storage::store_memory=debug",
            ))
            .with_ansi(false)
            .finish(),
    );
    let gate = unread_gate(last_good.budget.bytes);
    let cancel = CancellationToken::new();
    let task = spawn_store_memory_refresh_every_for_test(
        tv.client().clone(),
        StoreMemoryRefresh {
            gate: Some(Arc::clone(&gate)),
            initial: last_good.budget,
            memory_bytes: Some(X20_TOTAL),
            seed: last_good.seed(),
            metrics: None,
        },
        cancel.clone(),
        REFRESH_EVERY,
    );
    wait_for_count(&facade.slots_reads, facade.slots_reads(), 4, "malformed").await;
    cancel.cancel();
    task.await.expect("the refresh task ends on cancel");

    let output = String::from_utf8(captured.lock().unwrap().clone()).expect("UTF-8 log output");
    assert_eq!(
        gate.budget_bytes(),
        300_000_000,
        "the last good budget stands"
    );
    assert_eq!(
        output
            .lines()
            .filter(|l| l.contains(" WARN ") && l.contains(MALFORMED_WARN))
            .count(),
        1,
        "one WARN for the kind; got {output}"
    );
    facade.slots_malformed.store(false, Ordering::SeqCst);
    drop(tv);
    drop(facade.proxy);
}

/// b.xb7: a pass with no map — its `CLUSTER SLOTS` malformed or refused —
/// cannot confirm its `INFO` targets, so it sends no `INFO` and takes no
/// rule-2 figure even from a store that reports a `maxmemory`: on a staged
/// cluster reporting 3 GiB, with `LIVE_EDIT_VALKEY_MEMORY_BYTES` set,
/// startup starts on the fallback, the problem is the unconfirmed primaries,
/// and the map problem is still reported. With the map back, the same store
/// starts on `info` (one eighth of 3 GiB).
///
/// **Fails on the build before b.xb7** (stated, not run), where a pass with no
/// map still sent its `INFO`s and started on `info`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_staged_cluster_with_no_map_sends_no_info_and_starts_on_the_fallback() {
    let facade = MapFacade::start(eight_shards()).await;
    *facade.maxmemory.lock().unwrap() = Some(THREE_GIB);
    let tv = facade.test_valkey().await;
    let info_budget = ExtReleaseBudget {
        bytes: THREE_GIB / 8,
        source: ExtReleaseBudgetSource::StoreInfo {
            min_maxmemory_bytes: THREE_GIB,
            primaries: 1,
        },
    };

    let mut outcomes = Vec::new();
    for (what, malformed, refused) in [
        ("malformed", true, false),
        ("refused", false, true),
        ("the map back", false, false),
    ] {
        facade.slots_malformed.store(malformed, Ordering::SeqCst);
        facade.slots_fail.store(refused, Ordering::SeqCst);
        let info_before = facade.info_memory_reads();
        let read = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
        let resolution = resolve_startup(floor(), &read, Some(X20_TOTAL));
        outcomes.push((
            what,
            read.map_problem.as_ref().map(ClusterMapProblem::kind),
            facade.info_memory_reads() - info_before,
            resolution.problem.as_ref().map(StoreMemoryProblem::kind),
            resolution.budget,
        ));
    }
    let unconfirmed = Some(StoreMemoryProblemKind::PrimariesUnconfirmed);
    assert_eq!(
        outcomes,
        [
            (
                "malformed",
                Some(ClusterMapProblemKind::Malformed),
                0,
                unconfirmed,
                floor()
            ),
            (
                "refused",
                Some(ClusterMapProblemKind::ReadFailed),
                0,
                unconfirmed,
                floor()
            ),
            ("the map back", None, 1, None, info_budget),
        ],
        "(case, map problem, INFO memory sent, problem, startup budget)"
    );
    facade.slots_malformed.store(false, Ordering::SeqCst);
    facade.slots_fail.store(false, Ordering::SeqCst);
    drop(tv);
    drop(facade.proxy);
}

// ── b.xb7: a primary the routing table does not list ───────────────────────

/// The `maxmemory` the scale-out facade's second node reports: 1 GiB, below
/// the first node's 3 GiB, so a budget taken over both primaries is its ÷ 8
/// and one taken over the first alone is not.
const NEW_PRIMARY_MAXMEMORY: u64 = 1_073_741_824;

/// The node ids the scale-out facade's `CLUSTER SLOTS` reply names.
const SCALE_OUT_IDS: [&str; 3] = [
    "b0xb7000000000000000000000000000000first",
    "b0xb70000000000000000000000000000second",
    "b0xb7000000000000000000000000000000third",
];

/// The reconnect backoff of the scale-out tests' clients: fred retries a
/// cluster sync it could not complete inside the member's router at this
/// pace (`valkey_reconnect_max_delay_ms`, which is also the floor of the
/// exponential backoff), so a sync left running once the new node refuses
/// its setup completes within one step of the node accepting.
const SCALE_OUT_RECONNECT_DELAY_MS: u32 = 100;

/// How long the scale-out facade holds each reply of a kind before it sends
/// it (zero: at once). A held reply is still answered, so a client sees a
/// slow command, never a failed one, while the delay stays below its command
/// timeout.
#[derive(Clone, Copy, Default)]
struct StepDelays {
    cluster_slots: Duration,
    info_memory: Duration,
    commandstats: Duration,
}

/// `script`'s reply, held for `delay` first when it is not zero.
fn after(delay: Duration, script: String) -> ProxyAction {
    if delay.is_zero() {
        ProxyAction::Eval(script)
    } else {
        ProxyAction::DelayedEval(delay, script)
    }
}

/// Three [`ScriptReplyProxy`]s in front of the test store playing a cluster
/// that scales out. `CLUSTER SLOTS`, answered alike by every node, gives
/// dev's two entries (2 × 8,192 slots): both on the first node while `moved`
/// is clear, the second on the second node once it is set — a primary added
/// after the client connected, which its routing table does not list. While
/// `alternate` is set, the `CLUSTER SLOTS` the nodes answer from then on
/// alternate between those maps, whatever `moved` says: the 1st, 3rd, … the
/// scaled-out one, the 2nd, 4th, … the old one. While `three_primaries` is
/// set, whatever the others say, it gives three entries (5,461 / 5,462 /
/// 5,461 slots), one on each node; the third node is in no other map. `INFO
/// memory` answers `maxmemory` [`NEW_PRIMARY_MAXMEMORY`] on the second node
/// and 3 GiB on the others — or, while `report_maxmemory` is clear, no
/// `maxmemory` field on any, as Azure Managed Redis answers — each node
/// counting its own; `INFO commandstats` answers a `cmdstat_replconf` count
/// of 5. `delays` holds those three replies back. While `refuse_setup` is
/// set, the second node answers `CLUSTER INFO` — the last step of every
/// connection's setup in fred — with `cluster_state:fail` (a node still
/// joining), so no connection to it opens; it counts those refusals, and the
/// `cluster_state:ok` answers it gives otherwise. The nodes count the
/// `CLUSTER SLOTS` they answer in one count.
struct ScaleOutFacade {
    nodes: [ScriptReplyProxy; 3],
    moved: Arc<AtomicBool>,
    alternate: Arc<AtomicBool>,
    three_primaries: Arc<AtomicBool>,
    refuse_setup: Arc<AtomicBool>,
    report_maxmemory: Arc<AtomicBool>,
    delays: Arc<Mutex<StepDelays>>,
    slots_reads: Arc<AtomicUsize>,
    info_memory_reads: [Arc<AtomicUsize>; 3],
    setup_refusals: Arc<AtomicUsize>,
    setups_answered: Arc<AtomicUsize>,
}

impl ScaleOutFacade {
    async fn start() -> Self {
        let ports: Arc<OnceLock<[u16; 3]>> = Arc::new(OnceLock::new());
        let moved = Arc::new(AtomicBool::new(false));
        let alternate = Arc::new(AtomicBool::new(false));
        let alternated = Arc::new(AtomicUsize::new(0));
        let three_primaries = Arc::new(AtomicBool::new(false));
        let refuse_setup = Arc::new(AtomicBool::new(false));
        let report_maxmemory = Arc::new(AtomicBool::new(true));
        let delays = Arc::new(Mutex::new(StepDelays::default()));
        let slots_reads = Arc::new(AtomicUsize::new(0));
        let info_memory_reads: [Arc<AtomicUsize>; 3] = Default::default();
        let setup_refusals = Arc::new(AtomicUsize::new(0));
        let setups_answered = Arc::new(AtomicUsize::new(0));
        let rule = |node: usize| -> ProxyRule {
            let (ports, moved, refuse_setup) = (
                Arc::clone(&ports),
                Arc::clone(&moved),
                Arc::clone(&refuse_setup),
            );
            let three_primaries = Arc::clone(&three_primaries);
            let (alternate, alternated, slots_reads) = (
                Arc::clone(&alternate),
                Arc::clone(&alternated),
                Arc::clone(&slots_reads),
            );
            let (report_maxmemory, delays) = (Arc::clone(&report_maxmemory), Arc::clone(&delays));
            let info_memory_reads = Arc::clone(&info_memory_reads[node]);
            let (setup_refusals, setups_answered) =
                (Arc::clone(&setup_refusals), Arc::clone(&setups_answered));
            Arc::new(move |args: &[Vec<u8>]| {
                let ports = *ports
                    .get()
                    .expect("the ports are recorded before any client connects");
                let name = args.first()?.to_ascii_uppercase();
                let sub = args.get(1).map(|a| a.to_ascii_lowercase());
                let delays = *delays.lock().unwrap();
                match (name.as_slice(), sub.as_deref()) {
                    (b"CLUSTER", Some(b"slots")) => {
                        slots_reads.fetch_add(1, Ordering::SeqCst);
                        let scaled_out = if alternate.load(Ordering::SeqCst) {
                            alternated.fetch_add(1, Ordering::SeqCst) % 2 == 0
                        } else {
                            moved.load(Ordering::SeqCst)
                        };
                        let second_owner = usize::from(scaled_out);
                        let entry = |start: u16, end: u16, owner: usize| {
                            format!(
                                "{{{start}, {end}, {{'127.0.0.1', {}, '{}'}}}}",
                                ports[owner], SCALE_OUT_IDS[owner]
                            )
                        };
                        let entries = if three_primaries.load(Ordering::SeqCst) {
                            [
                                entry(0, 5460, 0),
                                entry(5461, 10922, 1),
                                entry(10923, 16383, 2),
                            ]
                            .join(", ")
                        } else {
                            [entry(0, 8191, 0), entry(8192, 16383, second_owner)].join(", ")
                        };
                        Some(after(delays.cluster_slots, format!("return {{{entries}}}")))
                    }
                    (b"CLUSTER", Some(b"info")) => {
                        if node == 1 && refuse_setup.load(Ordering::SeqCst) {
                            setup_refusals.fetch_add(1, Ordering::SeqCst);
                            return Some(ProxyAction::Eval(
                                "return 'cluster_state:fail\\r\\ncluster_slots_assigned:16384\\r\\n\
                                 cluster_known_nodes:2\\r\\ncluster_size:2\\r\\n'"
                                    .to_owned(),
                            ));
                        }
                        if node == 1 {
                            setups_answered.fetch_add(1, Ordering::SeqCst);
                        }
                        Some(ProxyAction::Eval(
                            "return 'cluster_state:ok\\r\\ncluster_slots_assigned:16384\\r\\n\
                             cluster_known_nodes:2\\r\\ncluster_size:2\\r\\n'"
                                .to_owned(),
                        ))
                    }
                    (b"INFO", Some(b"memory")) => {
                        info_memory_reads.fetch_add(1, Ordering::SeqCst);
                        let maxmemory = [THREE_GIB, NEW_PRIMARY_MAXMEMORY, THREE_GIB][node];
                        let line = report_maxmemory
                            .load(Ordering::SeqCst)
                            .then(|| format!("maxmemory:{maxmemory}"));
                        let ProxyAction::Eval(script) = memory_reply(line.as_deref(), DEV_USED)
                        else {
                            unreachable!("memory_reply answers a script");
                        };
                        Some(after(delays.info_memory, script))
                    }
                    (b"INFO", Some(b"commandstats")) => Some(after(
                        delays.commandstats,
                        "return '# Commandstats\\r\\n\
                         cmdstat_replconf:calls=5,usec=1,usec_per_call=0.20\\r\\n'"
                            .to_owned(),
                    )),
                    _ => None,
                }
            })
        };
        let first = ScriptReplyProxy::start(test_valkey_addr(), rule(0)).await;
        let second = ScriptReplyProxy::start(test_valkey_addr(), rule(1)).await;
        let third = ScriptReplyProxy::start(test_valkey_addr(), rule(2)).await;
        ports
            .set([first.port(), second.port(), third.port()])
            .expect("the ports are recorded exactly once");
        Self {
            nodes: [first, second, third],
            moved,
            alternate,
            three_primaries,
            refuse_setup,
            report_maxmemory,
            delays,
            slots_reads,
            info_memory_reads,
            setup_refusals,
            setups_answered,
        }
    }

    fn port(&self, node: usize) -> u16 {
        self.nodes[node].port()
    }

    /// `INFO memory` frames each node has received: `[first, second, third]`.
    fn info_memory_reads(&self) -> [usize; 3] {
        [0, 1, 2].map(|node| self.info_memory_reads[node].load(Ordering::SeqCst))
    }

    /// `CLUSTER SLOTS` frames the nodes have answered.
    fn slots_reads(&self) -> usize {
        self.slots_reads.load(Ordering::SeqCst)
    }

    /// A client connected through the first node in cluster mode, with one
    /// pooled client per pool, the [`SCALE_OUT_RECONNECT_DELAY_MS`] backoff
    /// and the test configuration's 5 s command timeout; `None` after a
    /// `SKIP:` line when no store was promised.
    async fn client(&self) -> Option<ValkeyClient> {
        self.client_with_command_timeout(crate::common::test_config().valkey_command_timeout_ms)
            .await
    }

    /// [`Self::client`] with a command timeout of `command_timeout_ms`.
    async fn client_with_command_timeout(&self, command_timeout_ms: u64) -> Option<ValkeyClient> {
        let mut cfg = crate::common::test_config();
        cfg.valkey_url = format!("redis-cluster://127.0.0.1:{}", self.port(0));
        cfg.valkey_pool_size = 1;
        cfg.subscriber_pool_size = 1;
        cfg.valkey_reconnect_max_delay_ms = SCALE_OUT_RECONNECT_DELAY_MS;
        cfg.valkey_command_timeout_ms = command_timeout_ms;
        connect_test_client_or_skip(&cfg).await
    }
}

/// The ports of the distinct primaries `client`'s maintenance member's
/// routing table lists, sorted.
fn listed_primary_ports(client: &ValkeyClient) -> Vec<u16> {
    let mut ports: Vec<u16> = client
        .maintenance_pool
        .next()
        .cached_cluster_state()
        .expect("a connected cluster client has a routing table")
        .unique_primary_nodes()
        .iter()
        .map(|server| server.port)
        .collect();
    ports.sort_unstable();
    ports
}

/// The `info` budget `maxmemory` gives over `primaries` primaries.
fn info_budget(maxmemory: u64, primaries: usize) -> ExtReleaseBudget {
    ExtReleaseBudget {
        bytes: maxmemory / 8,
        source: ExtReleaseBudgetSource::StoreInfo {
            min_maxmemory_bytes: maxmemory,
            primaries,
        },
    }
}

/// b.xb7 T4, the Issue's regression test: a primary added after the client
/// connected, which the client's routing table does not list, is read in the
/// same pass with `LIVE_EDIT_VALKEY_MEMORY_BYTES` unset. The client connects
/// while every slot is the first node's; the map then gives the second entry
/// to the second node, which reports the smaller `maxmemory`. Startup's one
/// read sends its `CLUSTER SLOTS`, finds the maintenance member's table
/// dividing the slots otherwise, re-syncs it once, and sends `INFO memory` to
/// both primaries: the budget is the smaller `maxmemory` ÷ 8, under `info`,
/// from 2 primaries, and the member's table lists the second node afterwards.
/// The pass sends exactly two `CLUSTER SLOTS`: its own, and the one fred's
/// completed sync sends — the count
/// [`a_read_whose_one_re_sync_still_leaves_the_table_differing_sends_no_info_and_syncs_once`]
/// relies on. Counts and budgets, never elapsed time; [`SETTLE`] only bounds
/// the read.
///
/// **Fails on the build before b.xb7** (stated, not run): it sent no `CLUSTER
/// SLOTS` with the setting unset, sent `INFO memory` only to the primaries the
/// table listed — the first node — and budgeted from its 3 GiB. Seen red by
/// in-place perturbation: with `info_targets` answering the cached table's
/// own primaries without the comparison, the second node gets no `INFO
/// memory`, the budget is 3 GiB ÷ 8 from one primary, and the table still
/// lists only the first node.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_read_reaches_a_primary_its_routing_table_does_not_list_and_budgets_from_it() {
    let facade = ScaleOutFacade::start().await;
    let Some(client) = facade.client().await else {
        return;
    };
    assert_eq!(
        listed_primary_ports(&client),
        vec![facade.port(0)],
        "precondition: the maintenance member's table lists the first node alone"
    );
    facade.moved.store(true, Ordering::SeqCst);
    let slots_before = facade.slots_reads();

    let startup = tokio::time::timeout(SETTLE, startup_store_memory(&client, floor(), None))
        .await
        .expect("the read ends");

    let mut both = vec![facade.port(0), facade.port(1)];
    both.sort_unstable();
    assert_eq!(
        (
            facade.info_memory_reads(),
            startup.budget,
            startup.problem,
            listed_primary_ports(&client),
            facade.slots_reads() - slots_before,
        ),
        (
            [1, 1, 0],
            info_budget(NEW_PRIMARY_MAXMEMORY, 2),
            None,
            both,
            2
        ),
        "(INFO memory per node, startup budget, problem, the table's primaries after the read, \
         CLUSTER SLOTS sent: the read's and its one sync's)"
    );
    drop(client);
}

/// b.xb7: a pass re-syncs its routing table at most once. The facade answers
/// the pass's own `CLUSTER SLOTS` with the scaled-out map and the next one —
/// the one fred's sync sends — with the old map, so the sync completes
/// (`Refreshed`) to a table that still divides the slots otherwise than the
/// pass's map. The pass is unconfirmed, naming the completed re-sync; it
/// sends no `INFO memory` to any node; and it sends exactly two `CLUSTER
/// SLOTS` — its own and the one sync's (a completed sync sends one: see
/// [`a_read_reaches_a_primary_its_routing_table_does_not_list_and_budgets_from_it`]),
/// so a second sync, which would send a third, is ruled out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_read_whose_one_re_sync_still_leaves_the_table_differing_sends_no_info_and_syncs_once() {
    let facade = ScaleOutFacade::start().await;
    let Some(client) = facade.client().await else {
        return;
    };
    facade.alternate.store(true, Ordering::SeqCst);
    let slots_before = facade.slots_reads();

    let read = tokio::time::timeout(SETTLE, read_store_memory(&client, None))
        .await
        .expect("the read ends");

    let refresh = match unconfirmed(&read.reading) {
        Some(PrimariesUnconfirmed::TableDiffers { refresh, .. }) => refresh.clone(),
        _ => panic!("a table that still differs, unconfirmed; got {read:?}"),
    };
    assert_eq!(
        (
            refresh,
            facade.info_memory_reads(),
            facade.slots_reads() - slots_before,
            listed_primary_ports(&client),
        ),
        (
            Some(RoutingRefresh::Refreshed),
            [0, 0, 0],
            2,
            vec![facade.port(0)]
        ),
        "(the re-sync, INFO memory per node, CLUSTER SLOTS sent, the table's primaries after)"
    );
    drop(client);
}

/// b.xb7 T4, the variant: the new primary refuses its connections' setup, so
/// the read's one re-sync of the routing table cannot complete — fred has
/// written the new table, but holds no connection to the new node — and the
/// read is unconfirmed: no `INFO memory` reaches any node that pass, the
/// problem is the unconfirmed primaries, naming a re-sync that timed out, and
/// a refresh keeps the `info` budget startup took from the first node. The
/// re-sync cannot fail instead: the client's reconnect policy allows
/// unlimited attempts, and fred retries a `cluster_state` refusal, so the
/// sync never returns before the bound cuts the wait. Once the node accepts
/// and the sync still running in the member's router completes, the next
/// read is confirmed and reaches both primaries, and a refresh moves the
/// budget to the smaller `maxmemory`.
///
/// The read's re-sync is cut off at the client's command timeout (5 s), and
/// fred keeps retrying it inside the member's router every
/// [`SCALE_OUT_RECONNECT_DELAY_MS`]; the test waits for the new node to answer
/// a connection's setup (`cluster_state:ok`) before the next read, so that
/// read finds the sync done. Only that member's sync connects to the new
/// node: no other client of the facade draws a `-MOVED` or reconnects, so the
/// refusals are its sync's. Counts and budgets, never elapsed time; [`SETTLE`]
/// only bounds each wait.
///
/// Fail-on-old, by in-place perturbation: with the table taken and compared
/// again after a re-sync that timed out, as after a completed one, the table
/// fred wrote matches, so the read sends `INFO memory` to its primaries, the
/// new node among them, while the cut-off sync still holds the member's
/// router: the reading is a failed `INFO` (it timed out), not an unconfirmed
/// one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_read_whose_routing_refresh_cannot_reach_the_new_primary_sends_no_info_until_it_can() {
    let facade = ScaleOutFacade::start().await;
    facade.refuse_setup.store(true, Ordering::SeqCst);
    let Some(client) = facade.client().await else {
        return;
    };
    let startup = tokio::time::timeout(SETTLE, startup_store_memory(&client, floor(), None))
        .await
        .expect("the startup read ends");
    assert_eq!(
        (startup.budget, facade.info_memory_reads()),
        (info_budget(THREE_GIB, 1), [1, 0, 0]),
        "precondition: startup read the first node alone"
    );
    facade.moved.store(true, Ordering::SeqCst);

    let cut_off = tokio::time::timeout(SETTLE, read_store_memory(&client, None))
        .await
        .expect("the read ends");

    let refresh = match unconfirmed(&cut_off.reading) {
        Some(PrimariesUnconfirmed::TableDiffers { refresh, .. }) => refresh.clone(),
        _ => panic!("a table that still differs, unconfirmed; got {cut_off:?}"),
    };
    assert!(
        matches!(refresh, Some(RoutingRefresh::TimedOut { .. })),
        "the re-sync timed out: {refresh:?}"
    );
    assert_eq!(
        (
            facade.info_memory_reads(),
            StoreMemoryProblem::of_reading(&cut_off.reading).map(|p| p.kind()),
            refreshed_budget(&startup.budget, &cut_off.reading, None),
            facade.setup_refusals.load(Ordering::SeqCst) > 0,
        ),
        (
            [1, 0, 0],
            Some(StoreMemoryProblemKind::PrimariesUnconfirmed),
            None,
            true,
        ),
        "(INFO memory per node, problem, refreshed budget, the new node refused a setup)"
    );

    facade.refuse_setup.store(false, Ordering::SeqCst);
    let deadline = tokio::time::Instant::now() + SETTLE;
    while facade.setups_answered.load(Ordering::SeqCst) == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the sync left running never connected to the new node"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let confirmed = tokio::time::timeout(SETTLE, read_store_memory(&client, None))
        .await
        .expect("the next read ends");

    assert_eq!(
        (
            facade.info_memory_reads(),
            confirmed.reading.primaries().map(<[_]>::len),
            refreshed_budget(&startup.budget, &confirmed.reading, None),
        ),
        (
            [2, 1, 0],
            Some(2),
            Some(info_budget(NEW_PRIMARY_MAXMEMORY, 2)),
        ),
        "(INFO memory per node, primaries read, refreshed budget) once the node accepts"
    );
    drop(client);
}

// ── b.xb7: startup's read bounded as a whole ───────────────────────────────

/// The command timeout T of the startup-deadline tests' clients, 4 s: startup's
/// deadline is 2T, 8 s. Each held reply below is held 0.55T (2.2 s), under T by
/// 1.8 s, so a held command is answered, never failed by its own timeout, and
/// only the read's deadline — the sum of four held replies, 2.2T — cuts a
/// step off.
const DEADLINE_TEST_COMMAND_TIMEOUT_MS: u64 = 4_000;

/// A share of [`DEADLINE_TEST_COMMAND_TIMEOUT_MS`], in thousandths.
fn of_command_timeout(thousandths: u64) -> Duration {
    Duration::from_millis(DEADLINE_TEST_COMMAND_TIMEOUT_MS * thousandths / 1000)
}

/// b.xb7 2(b) and 2(d): startup's read is bounded as a whole at two command
/// timeouts (2T), a refresh pass is not. On a cluster of three primaries,
/// each of the read's four steps — the `CLUSTER SLOTS` and an `INFO memory`
/// on each primary — is held 0.55T, each answered well within T, together
/// 2.2T. Startup's read is cut off at 2T in its third `INFO memory`: a failed
/// `INFO memory` naming the deadline, the map kept — so with
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES` set the budget is rule 3 from its three
/// entries, `override`, not the fallback — the gauge absent, the resolution
/// marked `cut_off_at_deadline`, and its seed carrying no problem. The same
/// four steps read as a refresh pass (`read_store_memory`, on a second
/// client whose connections hold no reply still owed) are waited for past
/// 2T: all three primaries answer, and the budget is the smallest
/// `maxmemory`'s.
///
/// Slow by design (about 17 s): the cut-off needs replies held past 2T, and
/// each must stay under T. Outcomes only, never elapsed time; [`SETTLE`]
/// bounds each read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_startup_read_cut_off_in_info_memory_keeps_the_map_and_a_refresh_read_is_not_cut_off() {
    let facade = ScaleOutFacade::start().await;
    facade.three_primaries.store(true, Ordering::SeqCst);
    let (startup_client, refresh_client) = tokio::join!(
        facade.client_with_command_timeout(DEADLINE_TEST_COMMAND_TIMEOUT_MS),
        facade.client_with_command_timeout(DEADLINE_TEST_COMMAND_TIMEOUT_MS),
    );
    let (Some(startup_client), Some(refresh_client)) = (startup_client, refresh_client) else {
        return;
    };
    *facade.delays.lock().unwrap() = StepDelays {
        cluster_slots: of_command_timeout(550),
        info_memory: of_command_timeout(550),
        commandstats: Duration::ZERO,
    };

    let startup = tokio::time::timeout(
        SETTLE,
        startup_store_memory(&startup_client, floor(), Some(DEV_TOTAL)),
    )
    .await
    .expect("the startup read ends");

    let why = match &startup.problem {
        Some(StoreMemoryProblem::ReadFailed(why)) => why.clone(),
        other => panic!("a failed INFO memory; got {other:?} in {startup:?}"),
    };
    assert!(why.contains("reached its deadline"), "{why}");
    assert_eq!(
        (
            startup.cut_off_at_deadline,
            startup.budget.bytes,
            startup.budget.source.label(),
            startup.used_ratio,
            startup.seed().problem,
        ),
        // 12 GB x 0.8 over the smallest of the three entries, 5,461 slots, ÷ 8.
        (true, 399_975_585, "override", None, None),
        "(cut off, budget, its source — rule 3 from the kept map — gauge, seeded problem)"
    );

    let refresh = tokio::time::timeout(SETTLE, read_store_memory(&refresh_client, None))
        .await
        .expect("the refresh read ends");
    assert_eq!(
        (
            refresh.reading.primaries().map(<[_]>::len),
            budget_from_reading(&refresh.reading, None),
        ),
        (Some(3), Some(info_budget(NEW_PRIMARY_MAXMEMORY, 3))),
        "a refresh read is waited for past 2T: (primaries read, budget); got {refresh:?}"
    );
    drop(startup_client);
    drop(refresh_client);
}

/// b.xb7 2(b): on a store that reports no `maxmemory` field, with
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES` set, startup's read also sends `INFO
/// commandstats` to each primary. With each `INFO memory` and each `INFO
/// commandstats` held 0.55T (together 2.2T, each well within T), the read is
/// cut off at 2T in its second `INFO commandstats`: the count is unreadable,
/// naming the deadline, while both `INFO memory` answers are kept — the
/// problem is the absent field, the budget rule 3, the gauge's arm the store
/// total's, pending — and the resolution is marked `cut_off_at_deadline`,
/// its seed not carrying the unreadable count.
///
/// Slow by design (about 8 s), for the reason above. Outcomes only;
/// [`SETTLE`] bounds the read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_startup_read_cut_off_in_info_commandstats_reads_the_count_as_unreadable() {
    let facade = ScaleOutFacade::start().await;
    facade.moved.store(true, Ordering::SeqCst);
    facade.report_maxmemory.store(false, Ordering::SeqCst);
    let Some(client) = facade
        .client_with_command_timeout(DEADLINE_TEST_COMMAND_TIMEOUT_MS)
        .await
    else {
        return;
    };
    *facade.delays.lock().unwrap() = StepDelays {
        cluster_slots: Duration::ZERO,
        info_memory: of_command_timeout(550),
        commandstats: of_command_timeout(550),
    };

    let startup = tokio::time::timeout(
        SETTLE,
        startup_store_memory(&client, floor(), Some(DEV_TOTAL)),
    )
    .await
    .expect("the startup read ends");

    let why = startup
        .commandstats_unreadable
        .clone()
        .unwrap_or_else(|| panic!("an unreadable count; got {startup:?}"));
    assert!(why.contains("reached its deadline"), "{why}");
    assert_eq!(
        (
            startup.cut_off_at_deadline,
            facade.info_memory_reads(),
            startup.problem.as_ref().map(StoreMemoryProblem::kind),
            startup.budget.bytes,
            startup.gauge_arm,
            startup.replconf_calls,
            startup.seed().commandstats_unreadable,
        ),
        (
            true,
            [1, 1, 0],
            Some(StoreMemoryProblemKind::FieldAbsent),
            600_000_000,
            StoreMemoryGaugeArm::StoreTotalPending,
            None,
            false,
        ),
        "(cut off, INFO memory per node, problem, budget, gauge arm, count, seeded unreadable)"
    );
    drop(client);
}

/// The slot b.xb7 T6 moves: the first primary's pin slot, as
/// `scan_guard_stale_table_test` moves it.
#[cfg(feature = "test-fault-injection")]
const MOVED_SLOT: u16 = 0;

/// The bound on T6's one store-memory read: its `CLUSTER SLOTS`, one re-sync
/// and an `INFO memory` per primary, each bounded by the 5 s command timeout
/// and each milliseconds when healthy.
#[cfg(feature = "test-fault-injection")]
const READ_BUDGET: Duration = Duration::from_secs(15);

/// b.xb7 T6, on a real cluster (`LIVE_EDIT_TEST_CLUSTER_URL`; skipped
/// otherwise): the reader's maintenance member connects first, so its routing
/// table is built; another client then moves slot 0 from the first primary
/// A to another primary B. The member sends nothing keyed, so its table
/// still maps slot 0 to A. One `read_store_memory` with
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES` unset sends its `CLUSTER SLOTS`, finds the
/// table grouping slot 0 with A's slots where the map groups it with B's,
/// re-syncs the table once and is confirmed: `INFO memory` on every primary,
/// and the member's table then routes slot 0 to B. Slot 0 is moved back
/// before any assertion, so a failure leaves the cluster as it found it.
///
/// **Disruptive and twice gated**: `#[ignore]`d with the
/// `ci: test-cluster-disruptive` label and selected by that job's `-E` filter
/// through `slot_migration_disruptive` (which also puts it in
/// `.config/nextest.toml`'s one-at-a-time `disruptive-cluster` group and keeps
/// its `CLUSTER-GATED RAN:` line on a pass), and skipped without
/// `LIVE_EDIT_TEST_CLUSTER_URL`. Compiled only with `test-fault-injection`,
/// the feature the shared `SlotMigration` helper is built under: a run
/// without it compiles the test out and reports nothing.
///
/// The slot moves go through `scan_guard_disruptive_test`'s `move_slot`, and
/// the second primary is its `another_primary`.
///
/// **Worst case**, every wait at its budget, against nextest's 120 s kill:
/// two moves, each `SlotMigration`'s drain (bounded at 30 s) and a
/// convergence wait of `scan_guard_disruptive_test`'s `CONVERGENCE_BUDGET`
/// (10 s), and the read's
/// [`READ_BUDGET`] (15 s): 95 s; well under a second healthy.
///
/// **Fail-on-old** (stated, not run: this sandbox has no cluster): the build
/// before b.xb7 sent no `CLUSTER SLOTS` with the setting unset and read the
/// primaries its stale table listed — every primary here, so its `INFO`s were
/// answered — but never re-synced the table, so slot 0 stays on A.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, migrates a hash slot between shards"]
async fn store_memory_cluster_slot_migration_disruptive_a_read_on_a_stale_table_resyncs_it_and_reads_every_primary(
) {
    let Some(url) = live_edit::storage::test_support::test_cluster_url() else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return;
    };
    let reader = TestValkey::setup_with_url(url.clone(), 1).await;
    let admin = TestValkey::setup_with_url(url, 1).await;
    let slot_owner = || {
        reader
            .client()
            .maintenance_pool
            .next()
            .cached_cluster_state()
            .expect("a connected cluster client has a routing table")
            .get_server(MOVED_SLOT)
            .cloned()
            .expect("the table names the slot's owner")
    };
    let a = slot_owner();
    let primaries = admin
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a routing table")
        .unique_primary_nodes()
        .len();
    let b = another_primary(&admin, &a);

    move_slot(&admin, MOVED_SLOT, &a, &b).await;
    let stale = slot_owner();
    let read = tokio::time::timeout(READ_BUDGET, read_store_memory(reader.client(), None)).await;
    let after = slot_owner();
    move_slot(&admin, MOVED_SLOT, &b, &a).await;

    assert_eq!(
        stale, a,
        "fixture: the reader's maintenance member still maps slot {MOVED_SLOT} to its old owner"
    );
    let read = read.expect("the read ends within its budget");
    let answered = read.reading.primaries().map(<[_]>::len);
    assert_eq!(
        (read.map_problem, answered, after),
        (None, Some(primaries), b),
        "(map problem, primaries read, slot {MOVED_SLOT}'s owner in the member's table after the \
         read); got {:?}",
        read.reading
    );
    println!(
        "CLUSTER-GATED RAN: store_memory_cluster_slot_migration_disruptive_a_read_on_a_stale_table_resyncs_it_and_reads_every_primary"
    );
}

/// T11, on a real cluster (`LIVE_EDIT_TEST_CLUSTER_URL`; skipped otherwise):
/// the store is read on every primary, pinned per node, and each reports
/// `used_memory`; the map's entries cover every slot once, at least one per
/// primary. The map comes from the read's own `CLUSTER SLOTS`, which every
/// clustered read sends.
///
/// b.xb7 T7: a read with `LIVE_EDIT_VALKEY_MEMORY_BYTES` unset is confirmed
/// on every primary too — its map, read from real Valkey's own `CLUSTER
/// SLOTS` spelling, divides the slots as the client's freshly connected
/// routing table does — so a comparison that read the two as different on
/// every pass (and so never sent an `INFO`) fails here. Prints
/// `CLUSTER-GATED RAN:` when it ran (`.config/nextest.toml` keeps that line
/// on a pass).
///
/// CI's cluster (and `scripts/dev-valkey.sh cluster-up`'s) sets no
/// `maxmemory`, so there only the unusable arm runs: the reading names the
/// primary that made its `maxmemory` unusable — not "the store" — the
/// setting divides by the map's entries, never by the primaries, no `INFO
/// commandstats` is sent (the store reports a `maxmemory` field, 0), and the
/// fullness gauge has no figure. The other arm — a store-reported limit
/// winning over the setting — cannot run there, since a `CONFIG SET
/// maxmemory` on the shared cluster is fenced; that precedence is pinned by
/// `a_store_reported_maxmemory_wins_over_the_store_total`
/// (`src/storage/store_memory.rs`), by
/// [`startup_reads_the_store_whatever_is_configured_and_its_limit_wins`], and,
/// on a cluster, by the staged
/// [`on_a_staged_cluster_reporting_maxmemory_the_limit_wins_over_the_store_total`]
/// and [`a_staged_cluster_with_no_map_sends_no_info_and_starts_on_the_fallback`].
/// The arm is kept for a cluster that does report a limit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_store_is_read_on_every_primary() {
    let Some(cluster_url) = live_edit::storage::test_support::test_cluster_url() else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    let tv = TestValkey::setup_with_url(cluster_url, 2).await;
    let node_client = tv.client().pool.next();
    assert!(node_client.is_clustered(), "a cluster URL is required");
    let primaries = node_client
        .cached_cluster_state()
        .expect("the routing table is known after connect")
        .unique_primary_nodes()
        .len();
    assert!(primaries >= 2, "precondition: a multi-shard cluster");

    let read = read_store_memory(tv.client(), Some(DEV_TOTAL)).await;
    assert_eq!(read.map_problem, None, "CLUSTER SLOTS gives a map");
    let reading = read.reading;
    let StoreMemoryReading::Read {
        info: Ok(answered),
        slot_ranges,
        endpoints,
        replconf_calls,
    } = &reading
    else {
        panic!("INFO memory on every primary of a connected cluster; got {reading:?}");
    };
    assert_eq!(answered.len(), primaries, "every primary was read");
    assert_eq!(*endpoints, primaries);
    assert!(
        answered.iter().all(|p| p.used_memory.is_some()),
        "every primary reports used_memory: {answered:?}"
    );
    assert!(
        slot_ranges.len() >= primaries,
        "the map lists at least one slot range per primary: {slot_ranges:?}"
    );
    let covered: u64 = slot_ranges
        .iter()
        .map(|&(start, end)| u64::from(end - start) + 1)
        .sum();
    assert_eq!(
        covered, 16_384,
        "the entries cover every slot once: {slot_ranges:?}"
    );

    let budget = budget_from_reading(&reading, Some(DEV_TOTAL))
        .expect("a store limit or the map gives a budget");
    match reading.maxmemory_limit() {
        Ok((min, read)) => {
            assert_eq!(read, primaries, "every primary reported");
            assert_eq!(
                budget.source,
                ExtReleaseBudgetSource::StoreInfo {
                    min_maxmemory_bytes: min,
                    primaries
                },
                "a store-reported limit wins over the setting"
            );
            assert_eq!(*replconf_calls, ReplconfCalls::NotRead);
            assert_eq!(
                store_memory_gauge(&reading, Some(DEV_TOTAL), None).0,
                StoreMemoryGaugeArm::Maxmemory
            );
        }
        Err(reason) => {
            assert!(
                reason.to_string().starts_with("primary ")
                    && !reason.to_string().starts_with("primary the store"),
                "an unusable reading names its primary: {reason}"
            );
            let (bytes, derivation) =
                memory_bytes_budget(slot_ranges, DEV_TOTAL).expect("a non-empty map");
            assert_eq!(budget.bytes, bytes);
            assert_eq!(
                budget.source.cluster_map().map(|(d, e)| (d.shards, e)),
                Some((slot_ranges.len(), primaries)),
                "the shards are the map's entries: {derivation:?}"
            );
            if reason.kind() != StoreMemoryProblemKind::FieldAbsent {
                assert_eq!(
                    *replconf_calls,
                    ReplconfCalls::NotRead,
                    "INFO commandstats is sent only to a store with no maxmemory field"
                );
                assert_eq!(
                    store_memory_gauge(&reading, Some(DEV_TOTAL), Some(2)),
                    (StoreMemoryGaugeArm::Absent, None),
                    "{reason}: no arm gives a figure"
                );
            }
        }
    }

    let unset = read_store_memory(tv.client(), None).await;
    let StoreMemoryReading::Read {
        info: Ok(answered),
        slot_ranges,
        ..
    } = &unset.reading
    else {
        panic!(
            "the setting unset: INFO memory on every primary, confirmed against the map; got \
             {unset:?}"
        );
    };
    assert_eq!(
        (unset.map_problem.clone(), answered.len()),
        (None, primaries),
        "the setting unset: (map problem, primaries read)"
    );
    assert_eq!(
        slot_ranges
            .iter()
            .map(|&(start, end)| u64::from(end - start) + 1)
            .sum::<u64>(),
        16_384,
        "the setting unset: the map covers every slot once: {slot_ranges:?}"
    );
    println!("CLUSTER-GATED RAN: a_cluster_store_is_read_on_every_primary");
}
