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

//! Every `UpdateLayer` commit whose landing the call observes is counted
//! exactly once, where it lands (Spec SDD SR-2.6), and its log line carries the
//! call's attempts, path, conflicts by kind and commit size (SR-1.2).
//!
//! # The defect
//!
//! The attempts histogram and `retries_total{outcome="committed"}` used to be
//! observed when the call returned, and the two plain commit routes' INFO
//! lines logged, only after the commit's post-commit awaits (the
//! external-value release enqueue). A call dropped
//! there — its RPC deadline elapsed, its client went away — left a commit that
//! had landed counted by neither the metrics nor the log. Each route now counts
//! and logs its commit synchronously at the point it knows the commit landed,
//! before its next await.
//!
//! # What lives here
//!
//! - **Per-route exactly-once.** The eight routes that end an `update_layer`
//!   call with a landed commit — a Lua commit, a Lua `ALREADY_LANDED` reply, a
//!   txn `EXEC`, a txn commit-token match, a reconciled in-doubt `EXEC`, an
//!   accepted unwatched commit, a structural-route commit, and a structural
//!   lost reply answered by the retry's read-side token check — each run
//!   twice: to completion, and dropped at
//!   the landing (the `with_commit_landed_hook_for_test` seam pauses the call
//!   right after the count, before the post-commit awaits; the test then drops
//!   the call's future, as an RPC deadline would). Both endings must show the
//!   header `seq` advanced by exactly 1, one attempts sample whose value is the
//!   call's attempt number, that number minus one committed retries, one
//!   `live_edit_update_layer_relay_bytes` sample of the call's relay length
//!   (b.yvh: its `_count` equals the attempts `_count`), exactly one
//!   commit log line, and one duration sample — `ok` for a completed call,
//!   `error` for a dropped one (Task 1's drop guard). Every scenario takes one
//!   injected retriable failure first, so the retries count is never a
//!   vacuous 0. Each route's own counter or seam queue proves the route was
//!   really taken. Every route's line also carries the landed commit's
//!   `records` (the delta's planned non-header mutations) and `bytes` (read
//!   off a `MONITOR` feed of the store) — on the token-landed routes, attempt
//!   2's commit, which attempt 3 must not reset. The in-doubt and unwatched
//!   WATCH routes run a structural delta on `lua-watch-structural`, the
//!   backend whose structural deltas take the WATCH path those seams stage.
//! - **The commit line's fields.** A WATCH-path call that commits on attempt 2
//!   after a nil `EXEC`, and a fast-path call that commits on attempt 2 after a
//!   transient failure: the line carries `attempt`, `path`, the five
//!   per-kind conflict counts, and `records` / `bytes` equal to what the commit
//!   actually sent.
//! - **WATCH-path size per mutation family.** A section tree's creation, a
//!   cascading section delete (sections, tombstones, children lists, fields,
//!   time samples), a time-sample write, and an external-value rewrite (the
//!   `obj:ext` index writes): each commit line's `records` and `bytes` match
//!   the plan and the `MONITOR` count, so every byte-counting site is pinned.
//!
//! The mapping of each conflict kind to its own field — on the commit line and
//! on the failure line — is pinned with distinct per-kind counts by unit tests
//! beside `log_landed_commit!` (`src/storage/layer.rs`) and the failure-line
//! emitter (`src/grpc/rpc_log.rs`); `watch_read_moved` on the call record by
//! the inline torn-snapshot test in `src/storage/layer.rs`.
//!
//! The exhausted call's failure line is pinned end to end in
//! `tests/integration/grpc/rpc_failure_log_test.rs`, which reuses this file's
//! `MONITOR` byte counters.
//!
//! # Not tested, by design
//!
//! A call dropped before it observes the landing — awaiting its own `EVALSHA`
//! / `EXEC` reply, or during the unwatched commit's retirement re-reads or the
//! in-doubt reconcile `XRANGE` — leaves a landed commit uncounted. That is the
//! documented limit of a landing-point count, not a defect.
//!
//! # Wait budget
//!
//! Each test makes at most two `update_layer` calls (a family test's setup
//! commit and its measured one), each under [`CALL_BUDGET`] (25 s); the
//! unwatched route's staged reconnect runs inside its call. Every test adds
//! the `MONITOR` acknowledgement and drain, each bounded by
//! `MONITOR_DRAIN_BUDGET` (20 s). Worst case 90 s, under nextest's 120 s
//! hang-killer; the healthy path takes milliseconds.
//!
//! Gated behind `test-fault-injection` (see `mod.rs`): every seam used here
//! exists only under that feature.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::clients::Client;
use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use futures::FutureExt;
use serde_json::Value;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::{apply_delta, StorageMutation};
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{Metrics, UPDATE_LAYER_CONFLICT_KINDS, UPDATE_LAYER_PATHS};
use live_edit::storage::diff_read::snapshot_to_object_state;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{CommitLandedHook, LayerStore};
use live_edit::storage::lua::{UpdateScriptMode, UPDATE_LAYER_FAST_SCRIPT};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::{relay_ompf, RelayedUpdates};
use live_edit::storage::test_support::{read_object_state_for_test, TestValkey};
use live_edit::storage::watched_txn::{
    WatchWindowHook, WatchedExecOverride, WatchedExecOverrideQueue,
};
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, OwnedDelta, SetField, StructCommand, TimeSample,
};

use crate::grpc::log_capture::{capture, Captured};
use crate::storage::b_bhb_tests::{
    await_reconnect, guarded_connection, kill_connection, once_only, POOL_SIZE,
};
use crate::storage::b_qpf_tests::{
    ext_hash, ext_hash_hex, ext_hash_vec, ext_store, obj_ext_sorted, pre_store,
};
use crate::storage::snapshot_read_support::{
    monitor_fence, test_valkey_addr, MonitorTap, MonitoredCommand,
};
use crate::storage::update_layer_commit_metrics_test::{
    aborted_injector, attempts_count, attempts_count_all_paths, attempts_sum, duration_count,
    relay_bytes_count, relay_bytes_sum, retries,
};

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// Deadline on the one `update_layer` call each test makes (or on its run up
/// to the landing, for a dropped call). The healthy path is a handful of round
/// trips, a few millisecond-scale retry backoffs and, on the unwatched route,
/// one fred reconnect; 25 s is two orders of magnitude of headroom.
const CALL_BUDGET: Duration = Duration::from_secs(25);

/// The writer of every call here.
const WRITER: &str = "commit-count-writer";

/// The writer's client timestamp.
const WRITER_TS: u64 = 1_000;

/// `path` of a commit on the Lua field fast path.
const FAST: &str = "fast";
/// `path` of a commit on the WATCH/MULTI/EXEC slow path.
const WATCH: &str = "watch";
/// `path` of a structural-route commit planned on a cache hit or repair.
const STRUCTURAL: &str = "structural";
/// `path` of a structural-route commit planned on a cold start or reload.
const STRUCTURAL_FULL: &str = "structural_full";

/// The header `seq` every fixture object starts at (CreateLayer's commit), and
/// the one its single `update_layer` commit must leave.
const SEQ_BEFORE: u64 = 1;
const SEQ_AFTER: u64 = 2;

// ═════════════════════════════════════════════════════════════════════════════
// The eight landing routes
// ═════════════════════════════════════════════════════════════════════════════

/// A route by which an `update_layer` call observes its commit land.
#[derive(Clone, Copy, Debug)]
enum Route {
    /// The fast-path script commits and its reply is decoded.
    LuaCommit,
    /// The fast-path script commits, its reply is lost, and the retry's script
    /// answers `ALREADY_LANDED`.
    LuaAlreadyLanded,
    /// A WATCH-path `EXEC` commits (txn backend, field delta).
    TxnCommit,
    /// A WATCH-path `EXEC` commits, its reply is lost, the reconcile read
    /// fails, and the retry's commit-token check finds the commit.
    TxnTokenLanded,
    /// A WATCH-path `EXEC` commits, its reply is lost, and the reconcile
    /// `XRANGE` confirms the commit.
    InDoubtReconciled,
    /// The connection holding the WATCH is replaced before dispatch, the
    /// `EXEC` runs unwatched, and the commit is accepted on a live object.
    UnwatchedAccepted,
    /// The structural route on the `lua` backend (a cold-started cache entry,
    /// so `path="structural_full"`): `commit_structural`'s script commits and
    /// its reply is decoded.
    StructuralCommit,
    /// The structural route: the commit lands, its reply is lost, and the
    /// retry's read side answers this call's token (`StructuralRead::
    /// AlreadyLanded`, before it plans) on a cache hit, so
    /// `path="structural"`.
    StructuralAlreadyLanded,
}

impl Route {
    fn layer(self) -> &'static str {
        match self {
            Route::LuaCommit => "count-lua-commit",
            Route::LuaAlreadyLanded => "count-lua-already-landed",
            Route::TxnCommit => "count-txn-commit",
            Route::TxnTokenLanded => "count-txn-token-landed",
            Route::InDoubtReconciled => "count-in-doubt-reconciled",
            Route::UnwatchedAccepted => "count-unwatched-accepted",
            Route::StructuralCommit => "count-structural-commit",
            Route::StructuralAlreadyLanded => "count-structural-already-landed",
        }
    }

    /// The commit path every metric of the call is labelled with.
    fn path(self) -> &'static str {
        match self {
            Route::LuaCommit | Route::LuaAlreadyLanded => FAST,
            Route::StructuralCommit => STRUCTURAL_FULL,
            Route::StructuralAlreadyLanded => STRUCTURAL,
            _ => WATCH,
        }
    }

    /// The backend the route's store runs: the three in-doubt and unwatched
    /// WATCH routes stage the WATCH path's own `EXEC` seams, which a
    /// structural delta reaches under `lua-watch-structural`; the default
    /// `lua` backend plans it on the structural cache instead.
    fn backend(self) -> UpdateLayerBackend {
        match self {
            Route::TxnTokenLanded | Route::InDoubtReconciled | Route::UnwatchedAccepted => {
                UpdateLayerBackend::LuaWatchStructural
            }
            Route::TxnCommit => UpdateLayerBackend::Txn,
            _ => UpdateLayerBackend::Lua,
        }
    }

    /// The attempt that observes the landing: one injected failure, then the
    /// committing attempt, then — on the three lost-reply routes — the retry
    /// that finds the commit.
    fn attempts(self) -> u32 {
        match self {
            Route::LuaAlreadyLanded | Route::TxnTokenLanded | Route::StructuralAlreadyLanded => 3,
            _ => 2,
        }
    }

    /// The message of the route's commit log line.
    fn commit_line(self) -> &'static str {
        match self {
            Route::LuaCommit => "UpdateLayer committed (lua)",
            Route::TxnCommit => "UpdateLayer committed",
            Route::StructuralCommit => "UpdateLayer committed (structural)",
            Route::LuaAlreadyLanded | Route::TxnTokenLanded | Route::StructuralAlreadyLanded => {
                "UpdateLayer attempt found this call's commit already landed; reporting it"
            }
            Route::InDoubtReconciled => {
                "UpdateLayer EXEC reply was lost but the commit landed; reporting success"
            }
            Route::UnwatchedAccepted => {
                "UpdateLayer committed on a connection that did not hold its WATCH; the XADD was \
                 accepted so no writer took this sequence, and the object is still live — \
                 accepting the commit"
            }
        }
    }
}

/// How the call under test ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ending {
    /// Awaited to its return.
    Completed,
    /// Its future dropped while the landing hook holds it, right after the
    /// commit landed — what an RPC deadline or a client cancel does to a call
    /// in its post-commit awaits.
    DroppedAtLanding,
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures
// ═════════════════════════════════════════════════════════════════════════════

type FredQueue = Arc<Mutex<VecDeque<FredError>>>;

struct Fixture {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    session_id: String,
    layer: &'static str,
    object_id: u64,
}

/// A fresh object at `seq` [`SEQ_BEFORE`] with one stream entry (`1-0`).
async fn fixture(layer: &'static str) -> Fixture {
    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(layer, &format!("/{layer}"), "commit-count-user")
        .await
        .expect("session creation must succeed");
    let state = ObjectState {
        sequence_num: SEQ_BEFORE,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 1,
        ..Default::default()
    };
    let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            layer,
            "commit-count-user",
            &state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    Fixture {
        tv,
        metrics,
        session_id,
        layer,
        object_id,
    }
}

/// A store on `backend` reporting into the fixture's registry.
fn store(fx: &Fixture, backend: UpdateLayerBackend) -> LayerStore {
    LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
        .with_metrics(Arc::clone(&fx.metrics))
}

/// Two inline set fields: fast-path eligible on the Lua backend, the WATCH
/// path on the txn backend. Two records, so a `records` of 1 cannot pass.
fn field_delta() -> OwnedDelta {
    let field = |key_name: &str, value: u8| SetField {
        section_id: 1,
        key_name: key_name.to_owned(),
        value_or_ext_hash: vec![value],
        ext_value_size: 0,
        ext_value_index: 0,
        set_order: 0,
    };
    OwnedDelta {
        set_fields: vec![field("first", 0x61), field("second", 0x62)],
        ..Default::default()
    }
}

/// One `CreateSection`: the structural route on the `lua` backend, the WATCH
/// path on `lua-watch-structural` and `txn`.
fn section_delta() -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 1,
            section_name: "CommitCountSection".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    }
}

/// `n` fred `Timeout`s — "the request went out and no reply came back".
fn timeouts(n: usize) -> FredQueue {
    Arc::new(Mutex::new(
        (0..n)
            .map(|_| FredError::new(FredErrorKind::Timeout, "commit-count: reply lost"))
            .collect(),
    ))
}

fn overrides(items: &[WatchedExecOverride]) -> WatchedExecOverrideQueue {
    Arc::new(Mutex::new(items.iter().copied().collect()))
}

/// Assert a seam queue drained, so the disturbance it stages really happened.
fn assert_drained<T: std::fmt::Debug>(queue: &Mutex<VecDeque<T>>, seam: &str) {
    let left = queue.lock().unwrap();
    assert!(
        left.is_empty(),
        "every queued {seam} entry must have been consumed; {left:?} left — the route under \
         test was not taken"
    );
}

/// The landing hook for `ending`: counts every landing it is called for and,
/// for a call to be dropped, signals `landed` and then never returns.
fn landing_hook(
    ending: Ending,
    landings: Arc<AtomicUsize>,
    landed: Arc<tokio::sync::Notify>,
) -> CommitLandedHook {
    Arc::new(move || {
        let landings = Arc::clone(&landings);
        let landed = Arc::clone(&landed);
        async move {
            landings.fetch_add(1, Ordering::SeqCst);
            if ending == Ending::DroppedAtLanding {
                landed.notify_one();
                std::future::pending::<()>().await;
            }
        }
        .boxed()
    })
}

/// A pre-dispatch hook that kills the connection holding the `WATCH` and waits
/// for fred to replace it, once: the `EXEC` then runs unwatched (the b.bhb
/// staging). `kills` counts the disturbances staged.
fn kill_watch_connection_once(fx: &Fixture, kills: Arc<AtomicUsize>) -> WatchWindowHook {
    let killer = fx.tv.client().pool.next().clone();
    let header_key = fx.tv.key_schema().obj_header(&fx.object_id.to_string());
    once_only(Arc::new(move |client: &Client| {
        let killer = killer.clone();
        let header_key = header_key.clone();
        let kills = Arc::clone(&kills);
        async move {
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            await_reconnect(client, &node, conn_id).await;
            kills.fetch_add(1, Ordering::SeqCst);
        }
        .boxed()
    }))
}

/// One route's store and delta, plus the seam queues that must drain.
struct Staged {
    store: LayerStore,
    delta: OwnedDelta,
    injected: Arc<Mutex<VecDeque<ServiceError>>>,
    lost_replies: Option<FredQueue>,
    exec_overrides: Option<WatchedExecOverrideQueue>,
    reconcile_failures: Option<FredQueue>,
    kills: Option<Arc<AtomicUsize>>,
}

async fn stage(route: Route, fx: &Fixture, hook: CommitLandedHook) -> Staged {
    let injected = aborted_injector(1);
    let mut staged = Staged {
        store: store(fx, route.backend()),
        delta: section_delta(),
        injected: Arc::clone(&injected),
        lost_replies: None,
        exec_overrides: None,
        reconcile_failures: None,
        kills: None,
    };
    match route {
        Route::LuaCommit => staged.delta = field_delta(),
        Route::LuaAlreadyLanded => {
            let lost = timeouts(1);
            staged.store = staged
                .store
                .with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&lost));
            staged.delta = field_delta();
            staged.lost_replies = Some(lost);
        }
        Route::TxnCommit => staged.delta = field_delta(),
        Route::StructuralCommit => {}
        Route::StructuralAlreadyLanded => {
            // One queue for both script modes: the structural commit's reply
            // is the one lost.
            let lost = timeouts(1);
            staged.store = staged
                .store
                .with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&lost));
            staged.lost_replies = Some(lost);
        }
        Route::TxnTokenLanded => {
            let exec = overrides(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
            let reconcile = timeouts(1);
            staged.store = staged
                .store
                .with_watched_exec_override_for_test(Arc::clone(&exec))
                .with_reconcile_xrange_fred_error_injector_for_test(Arc::clone(&reconcile));
            staged.exec_overrides = Some(exec);
            staged.reconcile_failures = Some(reconcile);
        }
        Route::InDoubtReconciled => {
            let exec = overrides(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
            staged.store = staged
                .store
                .with_watched_exec_override_for_test(Arc::clone(&exec));
            staged.exec_overrides = Some(exec);
        }
        Route::UnwatchedAccepted => {
            // The b.bhb witness needs the client's connection-id map and the
            // observability hooks that keep it current.
            fx.tv
                .client()
                .install_observability_hooks(Arc::clone(&fx.metrics))
                .await;
            let kills = Arc::new(AtomicUsize::new(0));
            staged.store =
                staged
                    .store
                    .with_pre_dispatch_hook_for_test(kill_watch_connection_once(
                        fx,
                        Arc::clone(&kills),
                    ));
            staged.kills = Some(kills);
        }
    }
    staged.store = staged
        .store
        .with_fault_injector_for_test(injected)
        .with_commit_landed_hook_for_test(hook);
    staged
}

// ═════════════════════════════════════════════════════════════════════════════
// Readers
// ═════════════════════════════════════════════════════════════════════════════

/// A zero-seeded counter cell, which must be present.
fn counter(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> f64 {
    sample_value(&render(metrics), name, labels)
        .unwrap_or_else(|| panic!("{name}{labels:?} is absent; its cells are zero-seeded"))
}

async fn header_seq(fx: &Fixture) -> u64 {
    let seq: String = fx
        .tv
        .client()
        .pool
        .hget(
            fx.tv.key_schema().obj_header(&fx.object_id.to_string()),
            "seq",
        )
        .await
        .expect("HGET seq must succeed");
    seq.parse().expect("header seq must be numeric")
}

/// The layer stream as `(id, client_id)` per entry, in order.
async fn stream_writers(fx: &Fixture) -> Vec<(String, String)> {
    // Values read as bytes: a commit's relay `ompf` field is binary.
    let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = fx
        .tv
        .client()
        .pool
        .xrange_values(
            fx.tv.key_schema().layer_stream(fx.object_id),
            "-",
            "+",
            None,
        )
        .await
        .expect("XRANGE over the layer stream must succeed");
    entries
        .into_iter()
        .map(|(id, fields)| {
            // `update|{oid}|{seq}|{client_id}|…`; CreateLayer's entry has none.
            let writer = fields
                .get("payload")
                .and_then(|p| std::str::from_utf8(p).ok()?.split('|').nth(3))
                .unwrap_or_default()
                .to_owned();
            (id, writer)
        })
        .collect()
}

/// Every commit log line in `captured`: an `UpdateLayer …` event carrying the
/// call summary's `attempt` field.
fn commit_lines(captured: &Captured) -> Vec<&Value> {
    captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with("UpdateLayer "))
                && e["fields"].get("attempt").is_some()
        })
        .collect()
}

/// The single commit log line, asserting there is exactly one.
fn only_commit_line(captured: &Captured) -> &Value {
    let lines = commit_lines(captured);
    assert_eq!(
        lines.len(),
        1,
        "a landed commit must be logged exactly once; got {} commit lines — full capture:\n{}",
        lines.len(),
        captured.text
    );
    lines[0]
}

// ═════════════════════════════════════════════════════════════════════════════
// MONITOR byte counters — what a commit actually sent
// ═════════════════════════════════════════════════════════════════════════════

/// One argument's length in bytes as the store received it. The feed escapes
/// every non-printable byte as one `\xHH` (or `\n`-style) sequence and
/// `MonitorTap` decodes each escape to one `char`, so the byte length is the
/// `char` count.
fn sent_len(arg: &str) -> u64 {
    arg.chars().count() as u64
}

/// Bytes of the one WATCH-path commit that landed `stream_id` on
/// `stream_key`, as `UpdateLayerCallSummary::bytes` defines them: every key
/// and argument of the commands executed between its `MULTI` and `EXEC`, but
/// not command names, the `XADD` trim clause, or the transaction's own
/// `CLIENT ID` witness (queued by `WatchedTxn`, not by the commit).
///
/// The feed shows a transaction's commands when `EXEC` runs them, so a
/// transaction a nil `EXEC` discarded shows none, and exactly one transaction
/// in `feed` can carry the landed `XADD`.
pub(crate) fn watch_commit_sent_bytes(
    feed: &[MonitoredCommand],
    stream_key: &str,
    stream_id: &str,
) -> u64 {
    let mut open: HashMap<&str, Vec<&MonitoredCommand>> = HashMap::new();
    let mut executed: Vec<Vec<&MonitoredCommand>> = Vec::new();
    for cmd in feed.iter().filter(|c| !c.issued_by_script()) {
        match cmd.name().as_str() {
            "MULTI" => {
                open.insert(cmd.client.as_str(), Vec::new());
            }
            "EXEC" => {
                if let Some(txn) = open.remove(cmd.client.as_str()) {
                    executed.push(txn);
                }
            }
            _ => {
                if let Some(txn) = open.get_mut(cmd.client.as_str()) {
                    txn.push(cmd);
                }
            }
        }
    }
    let lands = |cmd: &&MonitoredCommand| {
        cmd.name() == "XADD"
            && cmd.args.get(1).map(String::as_str) == Some(stream_key)
            && cmd.args.iter().any(|a| a == stream_id)
    };
    let commits: Vec<&Vec<&MonitoredCommand>> = executed
        .iter()
        .filter(|txn| txn.iter().any(lands))
        .collect();
    assert_eq!(
        commits.len(),
        1,
        "exactly one executed transaction may carry the XADD of {stream_id} on {stream_key}; \
         got {commits:#?}"
    );
    commits[0]
        .iter()
        .filter(|cmd| cmd.name() != "CLIENT")
        .map(|cmd| {
            if cmd.name() == "XADD" {
                // `XADD key [MAXLEN ~ n] id field value …`: the key, then the
                // id and everything after it.
                let id_at = cmd
                    .args
                    .iter()
                    .position(|a| a == stream_id)
                    .expect("the landed XADD names its id");
                sent_len(&cmd.args[1]) + cmd.args[id_at..].iter().map(|a| sent_len(a)).sum::<u64>()
            } else {
                cmd.args[1..].iter().map(|a| sent_len(a)).sum()
            }
        })
        .sum()
}

/// Bytes of the fast-path commit on `stream_key`, as
/// `UpdateLayerCallSummary::bytes` defines them: the script's `KEYS` and
/// `ARGV` — every argument of the client's `EVALSHA` after the script hash and
/// key count. A NOSCRIPT reload sends the same arguments twice; the last send
/// is the one that ran.
pub(crate) fn fast_commit_sent_bytes(feed: &[MonitoredCommand], stream_key: &str) -> u64 {
    let sent = feed
        .iter()
        .rfind(|c| {
            !c.issued_by_script()
                && matches!(c.name().as_str(), "EVALSHA" | "EVAL")
                && c.args.iter().any(|a| a == stream_key)
        })
        .unwrap_or_else(|| panic!("no script call naming {stream_key} in the feed: {feed:#?}"));
    let numkeys: usize = sent.args[2]
        .parse()
        .unwrap_or_else(|e| panic!("EVALSHA's third argument must be its key count: {e}"));
    assert!(
        sent.args.len() > 3 + numkeys,
        "the script call must carry its KEYS and ARGV: {sent:?}"
    );
    sent.args[3..].iter().map(|a| sent_len(a)).sum()
}

/// Bytes of the structural-mode commit on `stream_key`, as
/// `UpdateLayerCallSummary::bytes` defines them: the `KEYS` and `ARGV` of the
/// client's `EVALSHA` of `update_layer_fast.lua` carrying the `structural`
/// mode slot. Picked out by script and mode, because the structural route's
/// read side runs its own scripts (the cold start's atomic read, the repair
/// read) naming the stream key around the commit. Exactly one such commit
/// may be in `feed`: a structural route that sent a second would have
/// committed, or tried to commit, twice.
pub(crate) fn structural_commit_sent_bytes(feed: &[MonitoredCommand], stream_key: &str) -> u64 {
    let sha = UPDATE_LAYER_FAST_SCRIPT.sha1().to_owned();
    let commits: Vec<&MonitoredCommand> = feed
        .iter()
        .filter(|c| {
            !c.issued_by_script()
                && c.name() == "EVALSHA"
                && c.args.get(1).is_some_and(|s| s.eq_ignore_ascii_case(&sha))
                && c.args.iter().any(|a| a == stream_key)
                && c.args
                    .iter()
                    .any(|a| a == UpdateScriptMode::Structural.as_str())
        })
        .collect();
    let [sent] = commits.as_slice() else {
        panic!(
            "exactly one structural-mode commit naming {stream_key} must be in the feed; got \
             {commits:#?}"
        );
    };
    sent.args[3..].iter().map(|a| sent_len(a)).sum()
}

/// What `delta` plans against `object_id`'s stored state, read just before
/// the commit: the kind of every planned mutation and the number of mutation
/// records — every mutation but the header bump, which is how
/// `UpdateLayerCallSummary::records` counts them (the fast path folds the bump
/// into its script). The plan comes from the same `apply_delta` the commit
/// path runs, over the state the commit will read.
async fn planned(
    tv: &TestValkey,
    object_id: u64,
    delta: &OwnedDelta,
) -> (u64, BTreeSet<&'static str>) {
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &object_id.to_string())
        .await
        .expect("the object must be readable before the commit");
    let mut state = snapshot_to_object_state(&snapshot);
    let mutations =
        apply_delta(&mut state, Some(delta.clone())).expect("the delta must apply to the state");
    let kinds: BTreeSet<&'static str> = mutations.iter().map(mutation_kind).collect();
    let records = mutations
        .iter()
        .filter(|m| !matches!(m, StorageMutation::SetObjectHeader { .. }))
        .count() as u64;
    (records, kinds)
}

fn mutation_kind(m: &StorageMutation) -> &'static str {
    match m {
        StorageMutation::SetSection { .. } => "SetSection",
        StorageMutation::DeleteSection { .. } => "DeleteSection",
        StorageMutation::CreateTombstone { .. } => "CreateTombstone",
        StorageMutation::DeleteChildrenList { .. } => "DeleteChildrenList",
        StorageMutation::SetChildren { children, .. } if children.is_empty() => {
            "SetChildren(empty)"
        }
        StorageMutation::SetChildren { .. } => "SetChildren",
        StorageMutation::SetField { .. } => "SetField",
        StorageMutation::DeleteField { .. } => "DeleteField",
        StorageMutation::SetTimeSample { .. } => "SetTimeSample",
        StorageMutation::DeleteTimeSample { .. } => "DeleteTimeSample",
        StorageMutation::SetObjectHeader { .. } => "SetObjectHeader",
    }
}

/// Issue a fence command, after the operation under test, and drain `tap` up
/// to it.
async fn drain_to_fence(tv: &TestValkey, tap: &mut MonitorTap) -> Vec<MonitoredCommand> {
    let fence = monitor_fence(tv.prefix());
    let _: Option<String> = tv
        .client()
        .pool
        .get(&fence)
        .await
        .expect("the fence GET must succeed");
    tap.drain_until(&fence).await
}

// ═════════════════════════════════════════════════════════════════════════════
// Per-route exactly-once, completed and dropped at the landing
// ═════════════════════════════════════════════════════════════════════════════

/// Run `route`'s one `update_layer` call to `ending` and assert its commit was
/// counted, and logged, exactly once. See the module header.
async fn assert_route_counts_its_commit_once(route: Route, ending: Ending) {
    let fx = fixture(route.layer()).await;
    let landings = Arc::new(AtomicUsize::new(0));
    let landed = Arc::new(tokio::sync::Notify::new());
    let staged = stage(
        route,
        &fx,
        landing_hook(ending, Arc::clone(&landings), Arc::clone(&landed)),
    )
    .await;
    let (records, _) = planned(&fx.tv, fx.object_id, &staged.delta).await;
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let call = staged.store.update_layer_relayed(
        &fx.session_id,
        fx.layer,
        fx.object_id,
        Some(staged.delta.clone()),
        WRITER,
        WRITER_TS,
    );

    let (result, captured) = capture(async {
        match ending {
            Ending::Completed => Some(
                tokio::time::timeout(CALL_BUDGET, call)
                    .await
                    .expect("the call must settle within the budget"),
            ),
            Ending::DroppedAtLanding => {
                let mut call = std::pin::pin!(call);
                tokio::time::timeout(CALL_BUDGET, async {
                    tokio::select! {
                        biased;
                        () = landed.notified() => {}
                        returned = &mut call => panic!(
                            "{route:?}: the call must pause at its landing, not return; it \
                             returned {returned:?}"
                        ),
                    }
                })
                .await
                .expect("the call must reach its landing within the budget");
                // `call` is dropped here, at the landing — as an RPC deadline
                // would drop it during the post-commit awaits.
                None
            }
        }
    })
    .await;

    // The route was taken: its seams fired, and its own counters say so.
    assert_drained(&staged.injected, "injected Aborted");
    for (queue, seam) in [
        (&staged.lost_replies, "post-EVALSHA lost reply"),
        (&staged.reconcile_failures, "reconcile XRANGE failure"),
    ] {
        if let Some(queue) = queue {
            assert_drained(queue, seam);
        }
    }
    if let Some(queue) = &staged.exec_overrides {
        assert_drained(queue, "watched-EXEC override");
    }
    if let Some(kills) = &staged.kills {
        assert_eq!(
            kills.load(Ordering::SeqCst),
            1,
            "{route:?}: the WATCH connection must have been replaced once before dispatch"
        );
    }
    let already_landed = |backend: &str| {
        counter(
            &fx.metrics,
            "live_edit_update_layer_commit_already_landed_total",
            &[("backend", backend)],
        )
    };
    assert_eq!(
        (already_landed("lua"), already_landed("txn")),
        match route {
            Route::LuaAlreadyLanded | Route::StructuralAlreadyLanded => (1.0, 0.0),
            Route::TxnTokenLanded => (0.0, 1.0),
            _ => (0.0, 0.0),
        },
        "{route:?}: commit_already_landed_total names the route that reported the commit"
    );
    assert_eq!(
        counter(
            &fx.metrics,
            "live_edit_storage_watch_voided_total",
            &[
                ("path", "update_layer"),
                ("disposition", "committed_unwatched")
            ],
        ),
        if matches!(route, Route::UnwatchedAccepted) {
            1.0
        } else {
            0.0
        },
        "{route:?}: only the unwatched route commits on a connection without its WATCH"
    );
    assert_eq!(
        landings.load(Ordering::SeqCst),
        1,
        "{route:?}: exactly one attempt must observe the call's commit land"
    );

    // One commit landed: `seq` advanced by exactly 1, one stream entry.
    assert_eq!(
        header_seq(&fx).await,
        SEQ_AFTER,
        "{route:?}/{ending:?}: the header must advance exactly once"
    );
    assert_eq!(
        stream_writers(&fx).await,
        vec![
            ("1-0".to_owned(), String::new()),
            (format!("{SEQ_AFTER}-0"), WRITER.to_owned()),
        ],
        "{route:?}/{ending:?}: exactly one commit entry, the writer's"
    );
    if let Some(result) = result {
        let (seq, _) = result.unwrap_or_else(|e| {
            panic!("{route:?}: the commit landed, so the completed call must succeed; got {e:?}")
        });
        assert_eq!(
            seq, SEQ_AFTER,
            "{route:?}: the call reports the landed commit"
        );
    }

    // Counted exactly once, on the route's path, with the observing attempt.
    let (path, attempts) = (route.path(), route.attempts());
    for other in UPDATE_LAYER_PATHS.iter().filter(|p| **p != path) {
        assert_eq!(
            attempts_count(&fx.metrics, other),
            None,
            "{route:?}/{ending:?}: no attempts sample may land on path {other:?}"
        );
    }
    assert_eq!(
        (
            attempts_count(&fx.metrics, path),
            attempts_sum(&fx.metrics, path)
        ),
        (Some(1.0), Some(f64::from(attempts))),
        "{route:?}/{ending:?}: the landed commit must be exactly one attempts sample, of the \
         observing attempt's number ({attempts}). None means a commit that landed was never \
         counted — the defect, when the call was dropped — and a count of 2 means it was \
         counted twice"
    );
    assert_eq!(
        (
            retries(&fx.metrics, path, "committed"),
            retries(&fx.metrics, path, "exhausted")
        ),
        (f64::from(attempts - 1), 0.0),
        "{route:?}/{ending:?}: the commit's attempts minus one are its committed retries"
    );
    // b.yvh: the relay-bytes histogram samples exactly the commits the
    // attempts histogram counts — one sample, of the call's relay length, at
    // the same landing, so its `_count` equals the attempts `_count` summed
    // over `path`. On the three lost-reply routes the committing attempt
    // sampled nothing, so the observing attempt's report is the one sample.
    let relay_len = relay_ompf(&staged.delta).len() as f64;
    assert_eq!(
        (relay_bytes_count(&fx.metrics), relay_bytes_sum(&fx.metrics)),
        (Some(1.0), Some(relay_len)),
        "{route:?}/{ending:?}: the landed commit must be exactly one relay-bytes sample, of its \
         relay length ({relay_len}). None is a landing route that samples nothing, 2.0 a \
         commit sampled twice"
    );
    assert_eq!(
        relay_bytes_count(&fx.metrics),
        attempts_count_all_paths(&fx.metrics),
        "{route:?}/{ending:?}: the relay-bytes _count must equal the attempts _count"
    );

    // One duration sample: `ok` on a return, `error` on a drop (Task 1's drop
    // guard), so a dropped commit is an attempts sample beside an `error`
    // duration sample.
    assert_eq!(
        (
            duration_count(&fx.metrics, path, "ok"),
            duration_count(&fx.metrics, path, "error")
        ),
        match ending {
            Ending::Completed => (Some(1.0), None),
            Ending::DroppedAtLanding => (None, Some(1.0)),
        },
        "{route:?}/{ending:?}: one commit-duration sample per call"
    );

    // Logged exactly once, by the route, with the observing attempt.
    let line = only_commit_line(&captured);
    assert_eq!(
        line["fields"]["message"],
        route.commit_line(),
        "{route:?}: the commit line is the route's own; full event: {line}"
    );
    assert_eq!(
        (
            &line["fields"]["attempt"],
            &line["fields"]["path"],
            &line["fields"]["sequence_num"]
        ),
        (
            &Value::from(attempts),
            &Value::from(path),
            &Value::from(SEQ_AFTER)
        ),
        "{route:?}: the commit line carries the observing attempt, the path and the landed \
         sequence; full event: {line}"
    );

    // The commit's size, as the store received it. On the WATCH routes the
    // landed transaction executed — on the token-landed route that is attempt
    // 2's, and attempt 3, which builds nothing, must not have reset it. On
    // the Lua already-landed route the size is the observing attempt's own
    // script call, the last one sent. On the structural routes it is the one
    // structural-mode commit sent — on the already-landed route attempt 2's,
    // which attempt 3, answered by its read side, must not have reset.
    let feed = drain_to_fence(&fx.tv, &mut tap).await;
    let stream_key = fx.tv.key_schema().layer_stream(fx.object_id);
    let sent = match path {
        FAST => fast_commit_sent_bytes(&feed, &stream_key),
        STRUCTURAL | STRUCTURAL_FULL => structural_commit_sent_bytes(&feed, &stream_key),
        _ => watch_commit_sent_bytes(&feed, &stream_key, &format!("{SEQ_AFTER}-0")),
    };
    assert_eq!(
        (&line["fields"]["records"], &line["fields"]["bytes"]),
        (&Value::from(records), &Value::from(sent)),
        "{route:?}/{ending:?}: the commit line carries the landed commit's records (the \
         delta's non-header mutations) and the bytes the store received; full event: {line}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_commit_is_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::LuaCommit, Ending::Completed).await;
}

/// The SR-2.6 regression: a call dropped after its commit landed, in the
/// post-commit work, has still counted and logged that commit once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_commit_dropped_after_landing_is_still_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::LuaCommit, Ending::DroppedAtLanding).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_already_landed_report_is_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::LuaAlreadyLanded, Ending::Completed).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_already_landed_report_dropped_after_landing_is_still_counted_exactly_once(
) {
    assert_route_counts_its_commit_once(Route::LuaAlreadyLanded, Ending::DroppedAtLanding).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_commit_is_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::TxnCommit, Ending::Completed).await;
}

/// The SR-2.6 regression on the WATCH path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_commit_dropped_after_landing_is_still_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::TxnCommit, Ending::DroppedAtLanding).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_token_landed_report_is_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::TxnTokenLanded, Ending::Completed).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_token_landed_report_dropped_after_landing_is_still_counted_exactly_once()
{
    assert_route_counts_its_commit_once(Route::TxnTokenLanded, Ending::DroppedAtLanding).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_in_doubt_reconciled_commit_is_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::InDoubtReconciled, Ending::Completed).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_in_doubt_reconciled_commit_dropped_after_landing_is_still_counted_exactly_once(
) {
    assert_route_counts_its_commit_once(Route::InDoubtReconciled, Ending::DroppedAtLanding).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_unwatched_accepted_commit_is_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::UnwatchedAccepted, Ending::Completed).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_unwatched_accepted_commit_dropped_after_landing_is_still_counted_exactly_once(
) {
    assert_route_counts_its_commit_once(Route::UnwatchedAccepted, Ending::DroppedAtLanding).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_commit_is_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::StructuralCommit, Ending::Completed).await;
}

/// The SR-2.6 regression on the structural route: `commit_structural`'s
/// observer counts and logs the commit before the landed-commit hook.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_commit_dropped_after_landing_is_still_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::StructuralCommit, Ending::DroppedAtLanding).await;
}

/// A lost structural reply, answered by the retry's read-side token check: one
/// attempts sample, `commit_already_landed_total{backend="lua"}` = 1 and one
/// commit line — the read side's `AlreadyLanded` never reaches the commit's
/// observer, so the route reports it itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_already_landed_report_is_counted_exactly_once() {
    assert_route_counts_its_commit_once(Route::StructuralAlreadyLanded, Ending::Completed).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_already_landed_report_dropped_after_landing_is_still_counted_exactly_once(
) {
    assert_route_counts_its_commit_once(Route::StructuralAlreadyLanded, Ending::DroppedAtLanding)
        .await;
}

/// b.yvh (INV-9): one registry over a mixed run — a first-try fast commit, a
/// retried one, a fast lost reply, a WATCH commit, a lost `EXEC` answered by
/// the retry's commit-token check, a lost `EXEC` the reconcile confirms, a
/// structural commit and a structural lost reply. After every call
/// `live_edit_update_layer_relay_bytes_count` equals the attempts
/// histogram's `_count` summed over `path`, and at the end its `_sum` is the
/// calls' relay bytes. A route that samples nothing, or a lost-reply commit
/// sampled at both its committing and its reporting attempt, breaks the
/// equality at that call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_relay_bytes_count_equals_the_attempts_count_across_a_mixed_run() {
    let fx = fixture("count-relay-bytes-mixed").await;
    let numbered_section = |section_id: u64| OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id,
            section_name: format!("MixedRunSection{section_id}"),
            section_type: 1,
        })],
        ..Default::default()
    };
    let lost_exec = || overrides(&[WatchedExecOverride::ReplaceOutcomeAfterExec]);
    let steps: Vec<(&str, LayerStore, OwnedDelta)> = vec![
        (
            "fast, first try",
            store(&fx, UpdateLayerBackend::Lua),
            field_delta(),
        ),
        (
            "fast, retried",
            store(&fx, UpdateLayerBackend::Lua).with_fault_injector_for_test(aborted_injector(1)),
            field_delta(),
        ),
        (
            "fast, lost reply",
            store(&fx, UpdateLayerBackend::Lua)
                .with_lua_post_eval_fred_error_injector_for_test(timeouts(1)),
            field_delta(),
        ),
        ("WATCH", store(&fx, UpdateLayerBackend::Txn), field_delta()),
        (
            "WATCH, lost EXEC, token answer",
            store(&fx, UpdateLayerBackend::LuaWatchStructural)
                .with_watched_exec_override_for_test(lost_exec())
                .with_reconcile_xrange_fred_error_injector_for_test(timeouts(1)),
            numbered_section(2),
        ),
        (
            "WATCH, lost EXEC, reconciled",
            store(&fx, UpdateLayerBackend::LuaWatchStructural)
                .with_watched_exec_override_for_test(lost_exec()),
            numbered_section(3),
        ),
        (
            "structural",
            store(&fx, UpdateLayerBackend::Lua),
            numbered_section(4),
        ),
        (
            "structural, lost reply",
            store(&fx, UpdateLayerBackend::Lua)
                .with_lua_post_eval_fred_error_injector_for_test(timeouts(1)),
            numbered_section(5),
        ),
    ];

    let mut relay_bytes = 0u64;
    for (calls, (what, store, delta)) in steps.into_iter().enumerate() {
        relay_bytes += relay_ompf(&delta).len() as u64;
        tokio::time::timeout(
            CALL_BUDGET,
            store.update_layer_relayed(
                &fx.session_id,
                fx.layer,
                fx.object_id,
                Some(delta),
                WRITER,
                WRITER_TS,
            ),
        )
        .await
        .expect("the call must settle within the budget")
        .unwrap_or_else(|e| panic!("{what}: the call must commit; got {e:?}"));
        let landed = (calls + 1) as f64;
        assert_eq!(
            attempts_count_all_paths(&fx.metrics),
            Some(landed),
            "{what}: fixture: every call so far landed one counted commit"
        );
        assert_eq!(
            relay_bytes_count(&fx.metrics),
            Some(landed),
            "{what}: the relay-bytes _count must equal the attempts _count after every call"
        );
    }
    assert_eq!(
        relay_bytes_sum(&fx.metrics),
        Some(relay_bytes as f64),
        "the relay-bytes _sum is every landed call's relay length"
    );
    assert_eq!(
        (
            counter(
                &fx.metrics,
                "live_edit_update_layer_commit_already_landed_total",
                &[("backend", "lua")],
            ),
            counter(
                &fx.metrics,
                "live_edit_update_layer_commit_already_landed_total",
                &[("backend", "txn")],
            ),
        ),
        (2.0, 1.0),
        "fixture: the two script lost replies and the token answer each reported a landed commit"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The commit line's fields
// ═════════════════════════════════════════════════════════════════════════════

/// Assert `line` carries the per-kind conflict counts `want` (kind → count),
/// and 0 for every kind `want` does not name. Each kind is a field of its own,
/// present even at 0.
fn assert_conflict_fields(line: &Value, want: &[(&str, u64)]) {
    for kind in UPDATE_LAYER_CONFLICT_KINDS {
        let expected = want.iter().find(|(k, _)| k == kind).map_or(0, |(_, n)| *n);
        assert_eq!(
            line["fields"][*kind],
            Value::from(expected),
            "conflict kind `{kind}` must be logged as {expected}; full event: {line}"
        );
    }
}

/// A call whose first `EXEC` is cancelled by a write to its WATCHed header
/// (a nil `EXEC`) commits on attempt 2. Its commit line carries `attempt=2`,
/// `path="watch"`, `watch_conflict=1` and 0 for every other kind, and
/// `records` / `bytes` equal to the transaction that landed: one record per
/// field the delta sets (the header bump is not one), and the bytes of every
/// key and argument the store executed between that transaction's `MULTI` and
/// `EXEC`, read off the store's `MONITOR` feed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_commit_line_after_a_nil_exec_carries_attempt_path_conflicts_and_sent_size() {
    let fx = fixture("commit-line-nil-exec").await;
    let writer = fx.tv.client().pool.next().clone();
    let header_key = fx.tv.key_schema().obj_header(&fx.object_id.to_string());
    let hook: WatchWindowHook = once_only(Arc::new(move |_watched: &Client| {
        let writer = writer.clone();
        let header_key = header_key.clone();
        async move {
            // Re-set `seq` to its own value: any write to a WATCHed key
            // cancels the EXEC, and attempt 2 meets the same state.
            let seq: String = writer
                .hget(&header_key, "seq")
                .await
                .expect("HGET seq must succeed");
            let _: () = writer
                .hset(&header_key, [("seq", seq)])
                .await
                .expect("HSET seq must succeed");
        }
        .boxed()
    }));
    let ls = store(&fx, UpdateLayerBackend::Txn).with_watch_window_hook_for_test(hook);
    let delta = field_delta();
    let records = delta.set_fields.len() as u64;

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let (result, captured) = capture(tokio::time::timeout(
        CALL_BUDGET,
        ls.update_layer_relayed(
            &fx.session_id,
            fx.layer,
            fx.object_id,
            Some(delta),
            WRITER,
            WRITER_TS,
        ),
    ))
    .await;
    let (seq, _) = result
        .expect("the call must settle within the budget")
        .expect("the call must commit on attempt 2");
    let feed = drain_to_fence(&fx.tv, &mut tap).await;
    assert_eq!(seq, SEQ_AFTER);
    assert_eq!(
        counter(
            &fx.metrics,
            "live_edit_update_layer_conflicts_total",
            &[("path", WATCH), ("kind", "watch_conflict")],
        ),
        1.0,
        "precondition: attempt 1's EXEC must have been cancelled exactly once"
    );
    let sent = watch_commit_sent_bytes(
        &feed,
        &fx.tv.key_schema().layer_stream(fx.object_id),
        &format!("{SEQ_AFTER}-0"),
    );

    let line = only_commit_line(&captured);
    assert_eq!(line["level"], "INFO", "full event: {line}");
    assert_eq!(line["fields"]["message"], "UpdateLayer committed");
    assert_eq!(line["fields"]["attempt"], 2, "full event: {line}");
    assert_eq!(line["fields"]["path"], WATCH, "full event: {line}");
    assert_eq!(
        line["fields"]["sequence_num"], SEQ_AFTER,
        "full event: {line}"
    );
    assert_conflict_fields(line, &[("watch_conflict", 1)]);
    assert_eq!(
        line["fields"]["records"], records,
        "`records` is the commit's mutation records — one per field set, not the header \
         bump; full event: {line}"
    );
    assert_eq!(
        line["fields"]["bytes"], sent,
        "`bytes` must be the keys and arguments the landed transaction sent, as the store's \
         MONITOR feed recorded them; full event: {line}"
    );
}

/// A fast-path call whose first attempt fails transiently commits on attempt
/// 2. Its commit line carries `attempt=2`, `path="fast"`, every conflict kind
/// at 0 (a transient failure is not a conflict), and `records` / `bytes` equal
/// to the script call that ran: one record per field set, and the bytes of the
/// `EVALSHA`'s `KEYS` and `ARGV` read off the store's `MONITOR` feed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_path_commit_line_carries_the_records_and_bytes_its_script_was_sent() {
    let fx = fixture("commit-line-fast").await;
    let transient: FredQueue = Arc::new(Mutex::new(VecDeque::from([FredError::new(
        FredErrorKind::IO,
        "injected transient I/O failure",
    )])));
    let ls = store(&fx, UpdateLayerBackend::Lua)
        .with_lua_fred_error_injector_for_test(Arc::clone(&transient));
    let delta = field_delta();
    let records = delta.set_fields.len() as u64;

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let (result, captured) = capture(tokio::time::timeout(
        CALL_BUDGET,
        ls.update_layer_relayed(
            &fx.session_id,
            fx.layer,
            fx.object_id,
            Some(delta),
            WRITER,
            WRITER_TS,
        ),
    ))
    .await;
    let (seq, _) = result
        .expect("the call must settle within the budget")
        .expect("the call must commit on attempt 2");
    let feed = drain_to_fence(&fx.tv, &mut tap).await;
    assert_eq!(seq, SEQ_AFTER);
    assert_drained(&transient, "injected EVALSHA failure");
    let sent = fast_commit_sent_bytes(&feed, &fx.tv.key_schema().layer_stream(fx.object_id));

    let line = only_commit_line(&captured);
    assert_eq!(line["level"], "INFO", "full event: {line}");
    assert_eq!(line["fields"]["message"], "UpdateLayer committed (lua)");
    assert_eq!(line["fields"]["attempt"], 2, "full event: {line}");
    assert_eq!(line["fields"]["path"], FAST, "full event: {line}");
    assert_conflict_fields(line, &[]);
    assert_eq!(
        line["fields"]["records"], records,
        "`records` is the script's mutation records, one per field set; full event: {line}"
    );
    assert_eq!(
        line["fields"]["bytes"], sent,
        "`bytes` must be the KEYS and ARGV the script call sent, as the store's MONITOR feed \
         recorded them; full event: {line}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// WATCH-path commit size, per mutation family
// ═════════════════════════════════════════════════════════════════════════════
//
// The WATCH path adds up `bytes` at every site that queues a command between
// `MULTI` and `EXEC` — one per mutation family in `execute_mutation`, plus the
// external-value index writes. Each test below commits one delta whose plan
// reaches a family's sites (checked against the plan before the commit, so a
// delta that drifted away from its family fails as a precondition, not as a
// pass), and asserts the commit line's `records` is the plan's non-header
// mutation count and its `bytes` the `MONITOR` count of the landed
// transaction. A site that forgets a key or an argument reads short.

/// One `update_layer` commit of a family test: the delta, and the external
/// hashes whose pre-flight `store_values` increment the gRPC layer would have
/// paid before dispatching it.
struct Commit {
    delta: OwnedDelta,
    ext_hashes: Vec<[u8; 32]>,
}

impl From<OwnedDelta> for Commit {
    fn from(delta: OwnedDelta) -> Self {
        Self {
            delta,
            ext_hashes: Vec::new(),
        }
    }
}

/// Commit every `setup` delta, then `measured` under a `MONITOR` tap, on the
/// txn backend, and assert `measured`'s commit line carries its planned
/// records and the bytes the store received. `must_plan` names the
/// [`mutation_kind`]s the measured delta must plan. Returns the fixture for
/// any further keyspace assertion.
async fn assert_watch_commit_size_matches_sent(
    layer: &'static str,
    setup: Vec<Commit>,
    measured: Commit,
    must_plan: &[&str],
) -> Fixture {
    let fx = fixture(layer).await;
    let ls = store(&fx, UpdateLayerBackend::Txn);
    let ext = ext_store(&fx.tv);
    for commit in setup {
        pre_store(&ext, &commit.ext_hashes).await;
        tokio::time::timeout(
            CALL_BUDGET,
            ls.update_layer_relayed(
                &fx.session_id,
                fx.layer,
                fx.object_id,
                Some(commit.delta),
                WRITER,
                WRITER_TS,
            ),
        )
        .await
        .expect("a setup commit must settle within the budget")
        .expect("a setup commit must land");
    }

    let (records, kinds) = planned(&fx.tv, fx.object_id, &measured.delta).await;
    for kind in must_plan {
        assert!(
            kinds.contains(kind),
            "{layer}: precondition — the measured delta must plan `{kind}`, or this test does \
             not reach that family's byte-counting sites; it plans {kinds:?}"
        );
    }
    pre_store(&ext, &measured.ext_hashes).await;

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let (result, captured) = capture(tokio::time::timeout(
        CALL_BUDGET,
        ls.update_layer_relayed(
            &fx.session_id,
            fx.layer,
            fx.object_id,
            Some(measured.delta),
            WRITER,
            WRITER_TS,
        ),
    ))
    .await;
    let (seq, _) = result
        .expect("the measured commit must settle within the budget")
        .expect("the measured commit must land");
    let feed = drain_to_fence(&fx.tv, &mut tap).await;
    let sent = watch_commit_sent_bytes(
        &feed,
        &fx.tv.key_schema().layer_stream(fx.object_id),
        &format!("{seq}-0"),
    );

    let line = only_commit_line(&captured);
    assert_eq!(line["fields"]["path"], WATCH, "full event: {line}");
    assert_eq!(
        line["fields"]["records"], records,
        "{layer}: `records` is the commit's non-header mutations ({kinds:?}); full event: {line}"
    );
    assert_eq!(
        line["fields"]["bytes"], sent,
        "{layer}: `bytes` must be every key and argument the landed transaction sent, as the \
         store's MONITOR feed recorded them; full event: {line}"
    );
    fx
}

fn create_section(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: name.to_owned(),
        section_type: 1,
    })
}

fn inline_field(section_id: u64, key_name: &str, value: u8) -> SetField {
    SetField {
        section_id,
        key_name: key_name.to_owned(),
        value_or_ext_hash: vec![value],
        ext_value_size: 0,
        ext_value_index: 0,
        set_order: 0,
    }
}

fn inline_sample(section_id: u64, time: f64, value: u8) -> TimeSample {
    TimeSample {
        section_id,
        time,
        value_or_ext_hash: vec![value],
        ext_value_size: 0,
        ext_value_index: 0,
    }
}

/// A field holding the external value with hash byte `hash`.
fn external_field(section_id: u64, key_name: &str, hash: u8) -> SetField {
    SetField {
        section_id,
        key_name: key_name.to_owned(),
        value_or_ext_hash: ext_hash_vec(hash),
        ext_value_size: 64,
        ext_value_index: 1,
        set_order: 0,
    }
}

/// Creating a section with a child: each section's `HSET` and its `obj:secs`
/// `SADD`, and each non-empty children list's `DEL`, `RPUSH` and `obj:chs`
/// `SADD`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_watch_commit_size_of_a_section_tree_creation_matches_what_it_sent() {
    assert_watch_commit_size_matches_sent(
        "size-section-create",
        Vec::new(),
        OwnedDelta {
            struct_commands: vec![
                create_section(0, 1, "Parent"),
                create_section(1, 2, "Child"),
            ],
            ..Default::default()
        }
        .into(),
        &["SetSection", "SetChildren"],
    )
    .await;
}

/// Deleting a section that has a child, fields and time samples: the
/// sections' `DEL` and `SREM`, their tombstones' `HSET` and `SADD`, the
/// deleted parent's children-list `DEL` and `SREM`, the emptied root list's
/// `DEL` and `SREM`, and each purged field's and time sample's `DEL` and
/// `ZREM`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_watch_commit_size_of_a_cascading_section_delete_matches_what_it_sent() {
    let setup = OwnedDelta {
        struct_commands: vec![
            create_section(0, 1, "Parent"),
            create_section(1, 2, "Child"),
        ],
        set_fields: vec![
            inline_field(1, "parent-field", 0x71),
            inline_field(2, "child-field", 0x72),
        ],
        time_samples: vec![inline_sample(1, 1.0, 0x73), inline_sample(2, 2.5, 0x74)],
        ..Default::default()
    };
    assert_watch_commit_size_matches_sent(
        "size-section-delete",
        vec![setup.into()],
        OwnedDelta {
            struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 0,
                section_id: 1,
            })],
            ..Default::default()
        }
        .into(),
        &[
            "DeleteSection",
            "CreateTombstone",
            "DeleteChildrenList",
            "SetChildren(empty)",
            "DeleteField",
            "DeleteTimeSample",
        ],
    )
    .await;
}

/// Writing a time sample: its `HDEL` of the value shape, its `HSET`, and its
/// index `ZADD` (key, score as sent, member).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_watch_commit_size_of_a_time_sample_write_matches_what_it_sent() {
    assert_watch_commit_size_matches_sent(
        "size-time-sample",
        vec![OwnedDelta {
            struct_commands: vec![create_section(0, 1, "Animated")],
            ..Default::default()
        }
        .into()],
        OwnedDelta {
            time_samples: vec![inline_sample(1, 3.25, 0x75), inline_sample(1, 4.0, 0x76)],
            ..Default::default()
        }
        .into(),
        &["SetTimeSample"],
    )
    .await;
}

/// Rewriting a field from one external value to another: beside the field's
/// own writes, the external-value index `HSET` of the new hash and `HDEL` of
/// the superseded one (`queue_ext_index_writes`). The index naming only the
/// new hash afterwards — over a setup commit that indexed the old one (pinned
/// in `b_qpf_tests`) — shows both were queued by the measured commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_watch_commit_size_of_an_external_value_rewrite_matches_what_it_sent() {
    const OLD: u8 = 0xA1;
    const NEW: u8 = 0xB2;
    let setup = Commit {
        delta: OwnedDelta {
            struct_commands: vec![create_section(0, 1, "Textured")],
            set_fields: vec![external_field(1, "texture", OLD)],
            ..Default::default()
        },
        ext_hashes: vec![ext_hash(OLD)],
    };
    let measured = Commit {
        delta: OwnedDelta {
            set_fields: vec![external_field(1, "texture", NEW)],
            ..Default::default()
        },
        ext_hashes: vec![ext_hash(NEW)],
    };

    let fx = assert_watch_commit_size_matches_sent(
        "size-external-rewrite",
        vec![setup],
        measured,
        &["SetField"],
    )
    .await;
    assert_eq!(
        obj_ext_sorted(&fx.tv, fx.object_id).await,
        vec![(ext_hash_hex(NEW), 1)],
        "precondition: the measured commit must have rewritten the external-value index — \
         the new hash set, the superseded one deleted — or its index writes were not in the \
         transaction measured above"
    );
}
