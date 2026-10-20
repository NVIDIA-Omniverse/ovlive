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
//!   SLOTS` each read with `LIVE_EDIT_VALKEY_MEMORY_BYTES` set sends, and
//!   whose `INFO memory` reports no `maxmemory` field or a staged limit;
//! - a 3-shard cluster, when `LIVE_EDIT_TEST_CLUSTER_URL` is set (skipped
//!   otherwise).
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
use live_edit::storage::store_memory::{
    budget_from_reading, memory_bytes_budget, parse_info_maxmemory, read_store_memory,
    refreshed_budget, resolve_startup, spawn_store_memory_refresh_every_for_test,
    startup_store_memory, store_memory_gauge, ClusterMapProblem, ClusterMapProblemKind,
    MaxmemoryField, RefreshSeed, ReplconfCalls, StoreMemoryGaugeArm, StoreMemoryProblem,
    StoreMemoryProblemKind, StoreMemoryReading, StoreMemoryRefresh,
};
use live_edit::storage::test_support::TestValkey;

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
/// `cmdstat_replconf` line. Counts the `CLUSTER SLOTS` and the `INFO
/// commandstats` it receives.
struct MapFacade {
    proxy: ScriptReplyProxy,
    map: Arc<Mutex<Vec<(u16, u16)>>>,
    slots_fail: Arc<AtomicBool>,
    slots_malformed: Arc<AtomicBool>,
    maxmemory: Arc<Mutex<Option<u64>>>,
    slots_reads: Arc<AtomicUsize>,
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
        let commandstats_reads = Arc::new(AtomicUsize::new(0));
        let rule: ProxyRule = {
            let (port_slot, map, slots_fail, slots_reads) = (
                Arc::clone(&port_slot),
                Arc::clone(&map),
                Arc::clone(&slots_fail),
                Arc::clone(&slots_reads),
            );
            let (slots_malformed, maxmemory, commandstats_reads) = (
                Arc::clone(&slots_malformed),
                Arc::clone(&maxmemory),
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
            commandstats_reads,
        }
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

/// T5 (PD4), I4: with `LIVE_EDIT_VALKEY_MEMORY_BYTES` set, each read sends
/// its own `CLUSTER SLOTS` and takes the map from the reply, so a store
/// resharded from 2 entries to 8 since the client connected is seen at once —
/// 8 shards on 1 endpoint, the budget one eighth of an eighth of 80 % of
/// 24 GB, never divided by the endpoint. With the setting unset, nothing needs
/// the map: no `CLUSTER SLOTS` is sent and the reading carries no ranges.
///
/// **Fails if the read took the client's cached routing table instead**: the
/// third read still sees 2 entries, and the budget is 1,200,000,000.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_the_store_total_set_each_read_sends_its_own_cluster_slots() {
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
        facade.slots_reads(),
        before,
        "the setting unset: no CLUSTER SLOTS"
    );
    assert_eq!(
        (map_of(&unset.reading), unset.map_problem),
        ((Vec::new(), 1), None),
        "the setting unset: no map, and none needed"
    );

    let read = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
    assert!(
        facade.slots_reads() > before,
        "the setting set: the read sends CLUSTER SLOTS"
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

/// T5 (PD4): a `CLUSTER SLOTS` the store keeps refusing leaves the pass with
/// no map — the failure reported, no ranges, no budget from rule 3 — while
/// the `INFO`s behind it are answered at once: the command is an ordinary
/// one, so it never holds up the shared maintenance client the way fred's
/// `sync_cluster` would, retrying inside the client's router for as long as
/// the store refuses. The next read once the store answers gets the map.
///
/// **Fails with the map read through `sync_cluster`**: fred retries the
/// refused `CLUSTER SLOTS` inside the client's router (more than one reaches
/// the store), and the `INFO memory` queued behind that sync times out, so
/// the reading has no primaries. Both are asserted as outcomes — a count
/// and a failed `INFO` — never as elapsed time; [`SETTLE`] is only a hang
/// bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_cluster_slots_leaves_no_map_and_does_not_hold_up_the_client() {
    let facade = MapFacade::start(two_shards()).await;
    let tv = facade.test_valkey().await;
    *facade.map.lock().unwrap() = eight_shards();
    facade.slots_fail.store(true, Ordering::SeqCst);
    let before = facade.slots_reads();

    let failed = tokio::time::timeout(SETTLE, read_store_memory(tv.client(), Some(X20_TOTAL)))
        .await
        .expect("the read ends");

    assert_eq!(
        (
            facade.slots_reads() - before,
            failed.reading.primaries().is_some(),
        ),
        (1, true),
        "(CLUSTER SLOTS sent, the INFO memory behind it answered): sent once, not \
         retried, and nothing held up; got {failed:?}"
    );
    assert_eq!(
        failed.map_problem.as_ref().map(ClusterMapProblem::kind),
        Some(ClusterMapProblemKind::ReadFailed),
        "the failure is reported: {failed:?}"
    );
    assert_eq!(map_of(&failed.reading), (Vec::new(), 1), "no map");
    assert_eq!(
        budget_from_reading(&failed.reading, Some(X20_TOTAL)),
        None,
        "no map, no maxmemory: no budget from the store"
    );

    facade.slots_fail.store(false, Ordering::SeqCst);
    let recovered = tokio::time::timeout(SETTLE, read_store_memory(tv.client(), Some(X20_TOTAL)))
        .await
        .expect("a CLUSTER SLOTS the store answers is bounded");
    assert_eq!(
        (map_of(&recovered.reading), recovered.map_problem),
        ((eight_shards(), 1), None),
        "the next read sees the resharded map"
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
/// `Malformed` and carries no ranges, its `INFO memory` is still answered,
/// and the budget is the fallback — not one from the client's cached routing
/// table, which still lists the 2 entries it connected with. On a refresh
/// the last good budget (8 entries: 300,000,000) stands, and the malformed
/// map WARNs once however many passes see it.
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
    let malformed = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
    assert_eq!(malformed.map_problem, Some(ClusterMapProblem::Malformed));
    assert_eq!(map_of(&malformed.reading).0, Vec::new(), "no ranges");
    assert!(
        malformed.reading.primaries().is_some(),
        "INFO memory is still answered: {malformed:?}"
    );
    let startup = resolve_startup(floor(), &malformed, Some(X20_TOTAL));
    assert_eq!(startup.budget, floor(), "startup on no map: the fallback");
    assert_eq!(
        startup.seed().map_problem,
        Some(ClusterMapProblemKind::Malformed)
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

/// A pass with no map — its `CLUSTER SLOTS` malformed or refused — still
/// takes rule 2 when the store reports a `maxmemory`: on a staged cluster
/// reporting 3 GiB, with `LIVE_EDIT_VALKEY_MEMORY_BYTES` set, startup starts
/// on `info` (one eighth of 3 GiB), not the fallback, and the map problem is
/// still reported.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_staged_cluster_with_no_map_still_takes_the_stores_limit() {
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
    for (what, malformed) in [("malformed", true), ("refused", false)] {
        facade.slots_malformed.store(malformed, Ordering::SeqCst);
        facade.slots_fail.store(!malformed, Ordering::SeqCst);
        let read = read_store_memory(tv.client(), Some(X20_TOTAL)).await;
        let resolution = resolve_startup(floor(), &read, Some(X20_TOTAL));
        outcomes.push((
            what,
            read.map_problem.as_ref().map(ClusterMapProblem::kind),
            resolution.budget,
        ));
    }
    assert_eq!(
        outcomes,
        [
            (
                "malformed",
                Some(ClusterMapProblemKind::Malformed),
                info_budget
            ),
            (
                "refused",
                Some(ClusterMapProblemKind::ReadFailed),
                info_budget
            ),
        ],
        "(case, map problem, startup budget)"
    );
    facade.slots_malformed.store(false, Ordering::SeqCst);
    facade.slots_fail.store(false, Ordering::SeqCst);
    drop(tv);
    drop(facade.proxy);
}

/// T11, on a real cluster (`LIVE_EDIT_TEST_CLUSTER_URL`; skipped otherwise):
/// the store is read on every primary, pinned per node, and each reports
/// `used_memory`; the map's entries cover every slot once, at least one per
/// primary. With `LIVE_EDIT_VALKEY_MEMORY_BYTES` set the map comes from the
/// read's own `CLUSTER SLOTS`.
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
/// and [`a_staged_cluster_with_no_map_still_takes_the_stores_limit`]. The arm
/// is kept for a cluster that does report a limit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_store_is_read_on_every_primary() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
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
}
