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

//! b.tbw part one — the two whole-write transactions wait a timeout scaled to
//! the commands they queue (`storage::client::txn_timeout`: the command
//! timeout plus a per-command allowance, capped at the larger of the command
//! timeout and 45 s), not the flat per-command timeout.
//!
//! fred puts ONE timer around a whole transaction: sending every queued
//! command, the store's `EXEC` and the reply. Under the flat timeout a large
//! write's reply could outlive it although its `EXEC` landed: the WATCH-route
//! `UpdateLayer` commit then went in doubt, reconciled before the `EXEC` had
//! run, and retried over the whole layer (`watch_read_moved`, then
//! `commit_already_landed`); a `CreateLayer` answered `UNAVAILABLE` and left
//! its object behind, unmapped.
//!
//! Each test delays one transaction's reply, deterministically, on the
//! connection carrying that transaction only. Two delay a LARGE transaction
//! past the low command timeout but within its scaled one: the lower bound
//! (the timeout grows with the work). Two delay a SMALL transaction, whose
//! scaled timeout is the command timeout plus a few milliseconds, past that:
//! the upper bound (the timeout is still a finite one at each call site, so a
//! transaction left with no timeout, or one miscounted up to the 45 s cap,
//! fails them), and the expiry exits with the timeouts each counts: exactly
//! one per in-doubt `EXEC` or object write, and on the WATCH route also the
//! drop-time `UNWATCH` attempts that fire inside the guard's funnel, a number
//! scheduling decides and so only bounded. Two pin the residuals.
//!
//! | Test | Pins |
//! |---|---|
//! | [`watch_route_commit_whose_exec_reply_outlives_the_command_timeout_commits_on_its_first_attempt`] | T9: the WATCH commit is answered on its first attempt; no `command_timeout` on any pool, no `watch_read_moved`, no `commit_already_landed`; the sequence advances by exactly one |
//! | [`watch_route_commit_whose_exec_reply_outlives_its_scaled_timeout_is_reconciled_retried_and_lands_once`] | the WATCH route's expiry: a one-section commit held past its scaled timeout times out in doubt (exactly one in-doubt `EXEC` timeout on the client; `command_timeout` on txn alone, one to three with the drop-time `UNWATCH` attempts), is reconciled "not landed" and retried (two dispatches), and lands exactly once: one sequence step, one stream entry |
//! | [`create_layer_whose_exec_reply_outlives_the_command_timeout_commits_and_leaves_no_orphan`] | T10: the create answers OK, its layer is mapped, and it is the only object written |
//! | [`create_layer_whose_exec_reply_outlives_its_scaled_timeout_answers_unavailable_and_its_object_still_materializes_unmapped`] | the create's expiry, with no seam: a small create held past its scaled timeout answers `UNAVAILABLE`, counts one `command_timeout` on txn, and its object lands unmapped |
//! | [`create_layer_whose_object_write_times_out_answers_unavailable_and_its_object_still_materializes_unmapped`] | T11 (a): the residual — a create whose timeout is forced shorter than its `EXEC` (the seam) answers `UNAVAILABLE`, counts one `command_timeout` on txn, and the object still lands, unmapped, for the orphan sweep |
//! | [`update_layer_past_the_client_deadline_lands_once_and_its_resend_is_already_committed_end_to_end`] | T11 (b): a client deadline shorter than a WATCH-route reset reads `DEADLINE_EXCEEDED`; the reset lands once; a re-send with the same `operation_id` is OK `ALREADY_COMMITTED` at the original sequence |
//!
//! # How each reply is delayed
//!
//! - **The WATCH commit**: the commit's pre-dispatch hook parks a server-side
//!   `BLPOP` on the very connection that holds its `WATCH`
//!   (`test_support::ConnStall`), so the pipeline queues behind it. A blocked
//!   connection is not judged by the liveness check. The hold starts once the
//!   pipeline's first bytes reach the server (the stalled connection's
//!   `tot-net-in` grows), so the reply arrives at least the hold after fred's
//!   timer started.
//! - **The create**: `write_object`'s fred `Transaction` takes a txn-pool
//!   slot the test cannot reach before it is sent, so the client runs through
//!   a `ScriptReplyProxy` whose rule delays the transaction's `XADD` — the
//!   last command before `EXEC`, on a stream key only this test's prefix
//!   names — by [`EXEC_DELAY`]: the `EXEC` reaches the store, and its reply
//!   the client, at least that long after the send began. No liveness verdict
//!   judges a fred `Transaction`'s connection.
//!
//! # The margins
//!
//! The scaled timeout is `COMMAND_TIMEOUT + commands × 120 µs`.
//!
//! - T9's delta queues about 100,000 commands (20,000 sections under the
//!   root, each an `HSET` and an index `SADD`, and 20,000 fields, each three
//!   commands): about 1.5 s + 12 s = 13.5 s, against a reply at about
//!   [`HOLD`] (4 s) plus the `EXEC`.
//! - T10's object queues about 150,000 (30,000 sections, each two commands
//!   and one `RPUSH` into the root's list, and 20,000 fields): about 1.5 s +
//!   18 s = 19.5 s, against a reply at the send time plus [`EXEC_DELAY`] (4 s).
//! - Measured in an unoptimised build on a shared sandbox store, T9's
//!   transaction waited about 6 s (the 4 s hold, then about 1.5 s for the
//!   store to run the 100,000 queued commands) and T10's about 6.5 s (about
//!   2 s sending through the proxy, the 4 s delay, the `EXEC`): each under
//!   half its scaled timeout, while the flat [`COMMAND_TIMEOUT`] is always
//!   passed.
//! - The two expiry tests queue a few dozen commands, so their scaled timeout
//!   is about [`COMMAND_TIMEOUT`] + 5 ms; their replies are held at least
//!   [`EXEC_DELAY`] (the create) or until the call has answered (the WATCH
//!   commit, at most [`MAX_HOLD`]), both more than twice that.
//!
//! # Wait budget
//!
//! Each test's every wait, from its first line to its last assertion — the
//! fixture's setup included — runs under ONE deadline, [`SCENARIO_BUDGET`]
//! (90 s) from the test's start ([`by`], or a poll loop that checks the same
//! deadline). After it comes only `TestValkey`'s drop:
//!
//! - its purge (`cleanup_keys`) runs `SCAN MATCH <prefix>*` with a page hint
//!   of 100 and `UNLINK`s the matches in batches of 100, each page and each
//!   `UNLINK` one round trip, and fails at the first step unanswered for 5 s.
//!   `SCAN` walks the whole shared keyspace, not just this test's keys, so
//!   the pages number about the database's key count / 100: no fixed bound.
//!   The `UNLINK`s number this test's keys / 100. The largest object here is
//!   T10's (`queue_object_state`, plus the create's stream): one hash per
//!   section (30,000) and per field (20,000), plus the header, the sections
//!   index, the root's children list, the children index, the fields index
//!   and the stream (6), so 30,000 + 20,000 + 6 = about 50,000 keys (plus the
//!   session's few): about 500 `UNLINK`s;
//! - then it quits its client, bounded at 10 s.
//!
//! So each test's worst case is 90 + 5 + 10 = 105 s plus the purge's answered
//! round trips; `.config/nextest.toml` kills these tests at 240 s, which
//! leaves 135 s for those. The healthy runs take 2–31 s unoptimised on a
//! shared store.
//!
//! A stall a failed test leaves parked is not ended by the test: its `BLPOP`
//! holds its connection until the test process exits or the `BLPOP`'s own
//! 60 s server-side timeout fires. It holds a txn-pool connection, never one
//! of the fixture's data pool, which the purge uses, so it cannot stall the
//! purge; the quit that might queue behind it is the bounded 10 s above.
//!
//! Compiled only under `test-fault-injection` (the pre-dispatch hook and the
//! create's timeout seam), which CI's test job enables.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{ClientInterface, HashesInterface, KeysInterface, StreamsInterface};
use futures::FutureExt;
use tokio::time::Instant;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::{
    command_timeout_row, command_timeouts_only_on, connect_test_client, ConnStall, TestValkey,
};
use live_edit::storage::watched_txn::WatchWindowHook;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::snapshot_read_support::{test_valkey_addr, ProxyAction, ScriptReplyProxy};

/// The command timeout of the client under test: low, so a staged reply
/// outlives it by construction.
const COMMAND_TIMEOUT: Duration = Duration::from_millis(1_500);

/// How long T9 holds its stall after the pipeline's first bytes reach the
/// server.
const HOLD: Duration = Duration::from_secs(4);

/// The longest the WATCH-route expiry test holds its stall after the
/// pipeline's first bytes reach the server. On the healthy path it releases
/// as soon as the call has answered and its first attempt's connection is
/// back in the pool (about 3 s); a call that did not time out (a transaction
/// left with no timeout) never answers while held, and is released here so
/// the test fails on its assertions rather than its deadline.
const MAX_HOLD: Duration = Duration::from_secs(10);

/// How long the proxy holds the create's `XADD` (and so its `EXEC`).
const EXEC_DELAY: Duration = Duration::from_secs(4);

/// T11 (a)'s forced object-write timeout: far under [`EXEC_DELAY`].
const FORCED_OBJECT_WRITE_TIMEOUT: Duration = Duration::from_millis(300);

/// Each test's one deadline, from its first line to its last assertion; see
/// the module's "Wait budget".
const SCENARIO_BUDGET: Duration = Duration::from_secs(90);

/// Interval between probes of a polled condition.
const POLL: Duration = Duration::from_millis(10);

/// The layer every test writes.
const LAYER: &str = "txn-timeout-layer";

/// `fut` under the test's one `deadline`, failing naming `what` past it.
async fn by<T>(deadline: Instant, what: &str, fut: impl Future<Output = T>) -> T {
    tokio::time::timeout_at(deadline, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not finish within the test's {SCENARIO_BUDGET:?}"))
}

/// The test's one deadline: [`SCENARIO_BUDGET`] from now.
fn test_deadline() -> Instant {
    Instant::now() + SCENARIO_BUDGET
}

/// A client with [`COMMAND_TIMEOUT`] on `url` (the fixture's store, or a
/// proxy in front of it), counting into `metrics`.
async fn client_on(url: Option<String>, metrics: &Arc<Metrics>) -> ValkeyClient {
    let mut config = crate::common::test_config();
    if let Some(url) = url {
        config.valkey_url = url;
    }
    config.valkey_command_timeout_ms = COMMAND_TIMEOUT.as_millis() as u64;
    let client = connect_test_client(&config)
        .await
        .expect("the client under test must connect");
    client
        .install_observability_hooks(Arc::clone(metrics))
        .await;
    client
}

/// A session, created on the fixture's own client.
async fn new_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), "txn-timeout-user")
        .await
        .expect("session creation must succeed")
}

/// An empty layer at sequence 1, created on the fixture's own client.
async fn new_layer(tv: &TestValkey, session_id: &str) -> u64 {
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 1,
        ..Default::default()
    };
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            session_id,
            LAYER,
            "txn-timeout-user",
            &state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed")
}

/// `sections` created under the root (ids 1..=sections) and `fields` inline
/// fields on section 1.
fn sections_and_fields(sections: u64, fields: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: (1..=sections)
            .map(|id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: id,
                    section_name: format!("s{id}"),
                    section_type: 1,
                })
            })
            .collect(),
        set_fields: (0..fields)
            .map(|i| SetField {
                section_id: 1,
                key_name: format!("k{i}"),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            })
            .collect(),
        ..Default::default()
    }
}

/// The object state `delta` builds on an empty layer, at sequence 1.
fn state_of(delta: OwnedDelta) -> ObjectState {
    let mut state = ObjectState::default();
    live_edit::domain::delta::apply_delta(&mut state, Some(delta)).expect("the delta applies");
    state
}

/// `(sequence number, stream length)` of `object_id`.
async fn seq_and_stream(tv: &TestValkey, object_id: u64) -> (u64, u64) {
    let client = tv.client().pool.next();
    let seq: String = client
        .hget(tv.key_schema().obj_header(&object_id.to_string()), "seq")
        .await
        .expect("HGET seq");
    let stream: u64 = client
        .xlen(tv.key_schema().layer_stream(object_id))
        .await
        .expect("XLEN layer stream");
    (seq.parse().expect("seq is an integer"), stream)
}

/// The object [`LAYER`] is mapped to in `session_id`, if any.
async fn mapped_object(tv: &TestValkey, session_id: &str) -> Option<u64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .next()
        .get(tv.key_schema().layer_mapping(session_id, LAYER))
        .await
        .expect("GET layer mapping");
    raw.map(|v| v.parse().expect("a mapping names an object id"))
}

/// Every object header under the fixture's prefix: the objects written.
async fn object_headers(tv: &TestValkey) -> Vec<String> {
    let mut keys: Vec<String> = crate::common::scan_keys(
        tv.client(),
        &format!("{}{{*}}:obj:hdr", tv.key_schema().prefix()),
    )
    .await
    .into_iter()
    .map(|k| k.as_str().expect("a key is UTF-8").to_owned())
    .collect();
    keys.sort();
    keys
}

/// Poll [`object_headers`] until an object has been written, by `deadline`.
async fn first_object_written(tv: &TestValkey, deadline: Instant) -> Vec<String> {
    loop {
        let headers = by(deadline, "the object scan", object_headers(tv)).await;
        if !headers.is_empty() {
            return headers;
        }
        assert!(
            Instant::now() < deadline,
            "the delayed transaction never landed"
        );
        tokio::time::sleep(POLL).await;
    }
}

/// `live_edit_update_layer_conflicts_total{path="watch", kind}`.
fn watch_conflicts(metrics: &Metrics, kind: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_conflicts_total",
        &[("path", "watch"), ("kind", kind)],
    )
    .unwrap_or_else(|| panic!("the {kind} cell is seeded at zero")) as u64
}

/// `live_edit_update_layer_commit_already_landed_total{backend="txn"}`.
fn commit_already_landed(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_commit_already_landed_total",
        &[("backend", "txn")],
    )
    .expect("the txn cell is seeded at zero") as u64
}

/// `live_edit_valkey_pool_in_use{pool="txn"}`: txn-pool slots checked out.
fn txn_slots_in_use(metrics: &Metrics) -> i64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_pool_in_use",
        &[("pool", "txn")],
    )
    .expect("the txn cell is present once observability hooks are installed") as i64
}

/// What the first attempt's pre-dispatch hook leaves behind: the stall it
/// parked, and the connection it parked it on.
type Parked = Arc<tokio::sync::Mutex<Option<(ConnStall, Client)>>>;

/// A pre-dispatch hook that, on the first attempt only, parks a
/// [`ConnStall`] on the connection holding the attempt's `WATCH` and hands it,
/// with that connection, to `parked`; `dispatches` counts every attempt that
/// reached dispatch.
fn stall_first_dispatch(
    tv: &TestValkey,
    key: &str,
    parked: Parked,
    dispatches: Arc<AtomicUsize>,
) -> WatchWindowHook {
    let observer = tv.client().maintenance_pool.next().clone();
    let key = format!("{}{key}", tv.prefix());
    Arc::new(move |guarded: &Client| {
        let (parked, dispatches) = (Arc::clone(&parked), Arc::clone(&dispatches));
        let (observer, key, guarded) = (observer.clone(), key.clone(), guarded.clone());
        async move {
            if dispatches.fetch_add(1, Ordering::SeqCst) != 0 {
                return;
            }
            let stall = ConnStall::park(&guarded, &observer, key).await;
            *parked.lock().await = Some((stall, guarded));
        }
        .boxed()
    })
}

/// Wait until `parked` holds a stall, then take it.
async fn wait_parked(parked: &Parked, deadline: Instant) -> (ConnStall, Client) {
    loop {
        if let Some(held) = parked.lock().await.take() {
            return held;
        }
        assert!(
            Instant::now() < deadline,
            "the first attempt never reached dispatch"
        );
        tokio::time::sleep(POLL).await;
    }
}

/// Wait until the pipeline queued behind `stall` starts reaching the server.
async fn wait_first_bytes(stall: &ConnStall, deadline: Instant) {
    let parked_bytes = by(deadline, "a CLIENT LIST read", stall.net_in())
        .await
        .expect("the stalled connection stays open");
    while by(deadline, "a CLIENT LIST read", stall.net_in())
        .await
        .expect("the stalled connection stays open")
        == parked_bytes
    {
        assert!(
            Instant::now() < deadline,
            "the commit pipeline never reached the stalled connection"
        );
        tokio::time::sleep(POLL).await;
    }
}

/// Release `stall`, asserting it held until now.
async fn release(stall: ConnStall, deadline: Instant) {
    let id = stall.conn_id().to_owned();
    let reply = by(deadline, "the stall's release", stall.release()).await;
    assert!(
        matches!(&reply, Ok(v) if !v.is_null()),
        "the stall on connection {id} must have held the commit until its release; got {reply:?}"
    );
}

/// One round trip on `conn`: answered only after every command queued on it
/// before has run.
async fn drain(conn: &Client, deadline: Instant) {
    let _: i64 = by(deadline, "the drain round trip", conn.client_id())
        .await
        .expect("CLIENT ID on the released connection");
}

// ═════════════════════════════════════════════════════════════════════════════
// The WATCH-route commit: T9, and its expiry
// ═════════════════════════════════════════════════════════════════════════════

/// A WATCH-route commit (`txn` backend) whose `EXEC` reply arrives [`HOLD`]
/// after its pipeline reached the server — past the [`COMMAND_TIMEOUT`],
/// within its scaled timeout — is answered on its first attempt: one
/// dispatch, no in-doubt `EXEC` counted as a `command_timeout` on any pool,
/// no `watch_read_moved`, no `commit_already_landed`, and the sequence and
/// stream advance by exactly one.
///
/// Fails before b.tbw: the pipeline waited the flat command timeout, so
/// fred answered `Timeout` at 1.5 s; `WatchedTxn::exec` counted
/// `valkey_errors_total{pool="txn", kind="command_timeout"}` (the
/// `command_timeout_row` assertion), and `reconcile_in_doubt_commit` read the
/// stream while the `EXEC` was still parked, answered "not landed", and the
/// call retried: a second dispatch (the `dispatches` assertion), whose
/// WATCHed read saw the landed commit (`watch_read_moved`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_route_commit_whose_exec_reply_outlives_the_command_timeout_commits_on_its_first_attempt(
) {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let session_id = by(deadline, "the session", new_session(&tv, "t9-watch")).await;
    let object_id = by(deadline, "the layer", new_layer(&tv, &session_id)).await;
    let before = by(deadline, "a store read", seq_and_stream(&tv, object_id)).await;

    let parked: Parked = Arc::default();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let store = LayerStore::new(
        by(deadline, "the client", client_on(None, &metrics)).await,
        tv.key_schema().clone(),
        4096,
    )
    .with_update_layer_backend(UpdateLayerBackend::Txn)
    .with_metrics(Arc::clone(&metrics))
    .with_pre_dispatch_hook_for_test(stall_first_dispatch(
        &tv,
        "t9-txn-stall",
        Arc::clone(&parked),
        Arc::clone(&dispatches),
    ));

    let hold_past_the_command_timeout = async {
        let (stall, _) = wait_parked(&parked, deadline).await;
        // The hold starts once the pipeline reaches the server, so fred's
        // timer is running for at least HOLD before the reply can come back.
        wait_first_bytes(&stall, deadline).await;
        assert!(
            HOLD > COMMAND_TIMEOUT * 2,
            "precondition: the hold passes the flat timeout"
        );
        tokio::time::sleep(HOLD).await;
        release(stall, deadline).await;
    };
    let (outcome, ()) = tokio::join!(
        tokio::time::timeout_at(
            deadline,
            store.update_layer_with_summary_relayed(
                &session_id,
                LAYER,
                object_id,
                Some(sections_and_fields(20_000, 20_000)),
                "t9-writer",
                1_000,
                None,
            ),
        ),
        hold_past_the_command_timeout,
    );
    let (result, summary) = outcome.expect("the update did not settle within the scenario budget");
    let commit = result.unwrap_or_else(|e| panic!("the update must commit; got {e:?}"));

    assert_eq!(
        dispatches.load(Ordering::SeqCst),
        1,
        "answered on its first attempt: one dispatch"
    );
    let summary = summary.expect("a call that committed reports its summary");
    assert_eq!(summary.attempts, 1, "{summary:?}");
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on("txn", 0),
        "no command timed out on any pool: the EXEC reply was within the scaled timeout"
    );
    assert_eq!(watch_conflicts(&metrics, "watch_read_moved"), 0);
    assert_eq!(commit_already_landed(&metrics), 0);
    assert_eq!(
        commit.seq,
        before.0 + 1,
        "the sequence advanced by exactly one"
    );
    assert_eq!(
        by(deadline, "a store read", seq_and_stream(&tv, object_id)).await,
        (before.0 + 1, before.1 + 1),
        "one commit landed: one header bump, one stream entry"
    );
}

/// The WATCH route's expiry exit, with no seam: a one-section commit, whose
/// scaled timeout is [`COMMAND_TIMEOUT`] plus a few milliseconds, held past
/// it. Its `EXEC` times out in doubt — `WatchedTxn::exec` counts one
/// `command_timeout` on the txn pool — and `reconcile_in_doubt_commit` reads
/// the stream while the `EXEC` is still parked, answers "not landed", and the
/// call retries on another txn slot (the first slot stays held by its
/// `WatchGuard`'s `UNWATCH`, queued behind the stall): two dispatches, the
/// second committing. The delta lands exactly once: the sequence and the
/// stream advance by one. (The held `EXEC`, released after the call answered,
/// is expected to be voided by its `WATCH`, since the retry moved the header,
/// but the final read does not tell a void from a replay: the commit's writes
/// are absolute `HSET`s at the same sequence, and a replayed `XADD` at the
/// same id fails and adds no entry. The assertion pins one sequence step and
/// one stream entry, not the void.)
///
/// The `EXEC`'s own in-doubt timeout is pinned exactly, on the client the
/// store runs on: `ValkeyClient::watched_exec_timeouts_for_test` counts the
/// in-doubt timeouts `WatchedTxn::exec` builds and nothing else, and reads
/// one for the call. It is counted in `count_in_doubt_timeout` beside the
/// `EXEC`'s tick on `valkey_errors_total{pool="txn", kind="command_timeout"}`.
/// That metric cell also counts the first slot's drop-time `UNWATCH`
/// attempts, queued behind the stall, which is released only once that slot
/// is back in the pool. There are zero to two of them: the attempt and its
/// one retry, each counted only when its 650 ms per-attempt timeout
/// (`UNWATCH_ATTEMPT_TIMEOUT`) fires inside the guard's 1,500 ms funnel
/// (`UNWATCH_FUNNEL_TIMEOUT`); the funnel cancels the one in flight uncounted.
/// How many fire in time depends on scheduling (even the first does only
/// while scheduling adds under 850 ms to it), so the cell is pinned only to
/// what holds at any scheduling delay: at least one, the `EXEC`'s tick, and
/// at most three, the `EXEC` plus both attempts. No other pool counts one. A
/// lone `UNWATCH` tick also meets the lower bound, so this test does not pin
/// the `EXEC`'s own metric tick;
/// `b_bhb_tests::layer_in_doubt_exec_timeout_counts_one_txn_command_timeout`
/// pins it exactly, through the same `count_in_doubt_timeout`.
///
/// The upper bound at this call site: with the commit's transaction left
/// without a timeout, or miscounted up to the 45 s cap, the first attempt
/// waits out the hold (released at [`MAX_HOLD`]) and commits: one dispatch,
/// nothing in doubt, and no `command_timeout` at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_route_commit_whose_exec_reply_outlives_its_scaled_timeout_is_reconciled_retried_and_lands_once(
) {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let session_id = by(deadline, "the session", new_session(&tv, "expiry-watch")).await;
    let object_id = by(deadline, "the layer", new_layer(&tv, &session_id)).await;
    let before = by(deadline, "a store read", seq_and_stream(&tv, object_id)).await;

    let parked: Parked = Arc::default();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let client = by(deadline, "the client", client_on(None, &metrics)).await;
    // A clone shares the client's in-doubt EXEC timeout count.
    let watched = client.clone();
    let store = LayerStore::new(client, tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(Arc::clone(&metrics))
        .with_pre_dispatch_hook_for_test(stall_first_dispatch(
            &tv,
            "expiry-txn-stall",
            Arc::clone(&parked),
            Arc::clone(&dispatches),
        ));

    assert_eq!(
        watched.take_watched_exec_timeouts_for_test(),
        0,
        "precondition: nothing on the client under test was in doubt before the call"
    );

    let answered = AtomicBool::new(false);
    let update = async {
        let outcome = tokio::time::timeout_at(
            deadline,
            store.update_layer_with_summary_relayed(
                &session_id,
                LAYER,
                object_id,
                Some(sections_and_fields(1, 0)),
                "expiry-writer",
                1_000,
                None,
            ),
        )
        .await;
        answered.store(true, Ordering::SeqCst);
        outcome
    };
    // Held until the call has answered and the first attempt's slot is back
    // in the pool (its UNWATCH given up), or MAX_HOLD; then released, and the
    // connection drained so the held EXEC has run before anything is read.
    let hold_past_the_scaled_timeout = async {
        let (stall, conn) = wait_parked(&parked, deadline).await;
        wait_first_bytes(&stall, deadline).await;
        let cap = Instant::now() + MAX_HOLD;
        while !(answered.load(Ordering::SeqCst) && txn_slots_in_use(&metrics) == 0)
            && Instant::now() < cap
        {
            assert!(Instant::now() < deadline, "the hold outlived the test");
            tokio::time::sleep(POLL).await;
        }
        release(stall, deadline).await;
        drain(&conn, deadline).await;
    };
    let (outcome, ()) = tokio::join!(update, hold_past_the_scaled_timeout);
    let (result, summary) = outcome.expect("the update did not settle within the scenario budget");
    let commit = result.unwrap_or_else(|e| panic!("the update must commit; got {e:?}"));

    assert_eq!(
        dispatches.load(Ordering::SeqCst),
        2,
        "timed out in doubt, reconciled not landed, retried: two dispatches"
    );
    let summary = summary.expect("a call that committed reports its summary");
    assert_eq!(summary.attempts, 2, "{summary:?}");
    assert_eq!(
        watched.watched_exec_timeouts_for_test(),
        1,
        "exactly one EXEC timed out in doubt: the first attempt's"
    );
    // The EXEC's tick plus 0..=2 drop-time UNWATCH attempts, however many fired
    // inside the guard's funnel; see the test's doc.
    let row = command_timeout_row(&metrics);
    let txn = row
        .iter()
        .find_map(|&(pool, n)| (pool == "txn").then_some(n))
        .expect("the txn pool has a command_timeout cell");
    assert!(
        (1..=3).contains(&txn),
        "txn command_timeout: the EXEC plus at most two drop-time UNWATCH attempts; got {row:?}"
    );
    assert_eq!(
        row,
        command_timeouts_only_on("txn", txn),
        "no other pool counted a command_timeout"
    );
    assert_eq!(
        commit.seq,
        before.0 + 1,
        "the retry committed at the next sequence"
    );
    assert_eq!(
        by(deadline, "a store read", seq_and_stream(&tv, object_id)).await,
        (before.0 + 1, before.1 + 1),
        "the delta landed exactly once: one sequence step, one stream entry"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The create's object write: T10, its expiry, T11 (a)
// ═════════════════════════════════════════════════════════════════════════════

/// A proxy in front of the fixture's store that delays every `XADD` on a key
/// under `prefix` — in a create, only `write_object`'s `XADD 1-0`, the last
/// command before its `EXEC` — by [`EXEC_DELAY`].
async fn exec_delaying_proxy(prefix: &str) -> ScriptReplyProxy {
    let prefix = prefix.as_bytes().to_vec();
    ScriptReplyProxy::start(
        test_valkey_addr(),
        Arc::new(move |args: &[Vec<u8>]| {
            (args.first()?.eq_ignore_ascii_case(b"XADD") && args.get(1)?.starts_with(&prefix))
                .then_some(ProxyAction::Delay(EXEC_DELAY))
        }),
    )
    .await
}

/// A `CreateLayer` whose object write's `EXEC` reaches the store
/// [`EXEC_DELAY`] after the transaction was sent — past the
/// [`COMMAND_TIMEOUT`], within its scaled timeout — answers OK: the layer is
/// mapped to the object it created, which is the only object written (no
/// orphan), with no `command_timeout` counted.
///
/// Fails before b.tbw: the transaction waited the flat command timeout, so
/// `create_layer` answered `UNAVAILABLE "EXEC object write failed (Timeout
/// Error)"` at 1.5 s (the `unwrap_or_else` below), while the delayed `EXEC`
/// still ran and left the object unmapped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_layer_whose_exec_reply_outlives_the_command_timeout_commits_and_leaves_no_orphan() {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let session_id = by(deadline, "the session", new_session(&tv, "t10-create")).await;
    let proxy = exec_delaying_proxy(tv.key_schema().prefix()).await;
    let store = LayerStore::new(
        by(
            deadline,
            "the client",
            client_on(Some(proxy.url()), &metrics),
        )
        .await,
        tv.key_schema().clone(),
        4096,
    )
    .with_metrics(Arc::clone(&metrics));
    let state = state_of(sections_and_fields(30_000, 20_000));
    assert!(
        EXEC_DELAY > COMMAND_TIMEOUT * 2,
        "precondition: the delay passes the flat timeout"
    );

    let object_id = by(
        deadline,
        "the create",
        store.create_layer(
            &session_id,
            LAYER,
            "txn-timeout-user",
            &state,
            Arc::new(MockExternalValueStore::new()),
        ),
    )
    .await
    .unwrap_or_else(|e| panic!("the create must commit; got {e:?}"));

    assert_eq!(
        proxy.hits(),
        1,
        "vacuity guard: the object write's XADD was delayed"
    );
    assert_eq!(
        by(deadline, "a store read", mapped_object(&tv, &session_id)).await,
        Some(object_id)
    );
    assert_eq!(
        by(deadline, "the object scan", object_headers(&tv)).await,
        vec![tv.key_schema().obj_header(&object_id.to_string())],
        "no orphan: the mapped object is the only one written"
    );
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on("txn", 0),
        "no command timed out on any pool"
    );
}

/// The prefix of the `UNAVAILABLE` a timed-out object write answers.
fn object_write_timeout_prefix() -> String {
    format!(
        "EXEC object write failed ({}): ",
        fred::error::ErrorKind::Timeout.to_str()
    )
}

/// The create's expiry exit, with no seam: a small create (3 sections and 3
/// fields, about 22 commands, so its scaled timeout is [`COMMAND_TIMEOUT`]
/// plus about 3 ms) whose `EXEC` is held [`EXEC_DELAY`] past it answers
/// `UNAVAILABLE` naming the timed-out object write, counts exactly one
/// `command_timeout`, on the txn pool, and its transaction still lands: one
/// object, unmapped. fred wrote the whole transaction, its `EXEC` included,
/// before the timer fired — only the proxy holds it — and a frame fred has
/// already written is not recalled when its caller gives up (the residual of
/// the vendored fred's change 9).
///
/// The upper bound at this call site: with the create's transaction left
/// without a timeout (`Some(Duration::ZERO)` is fred's "none"), or its count
/// saturated up to the 45 s cap, the create waits out the delay and answers
/// OK (the `expect_err` below), counting nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_layer_whose_exec_reply_outlives_its_scaled_timeout_answers_unavailable_and_its_object_still_materializes_unmapped(
) {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let session_id = by(deadline, "the session", new_session(&tv, "expiry-create")).await;
    let proxy = exec_delaying_proxy(tv.key_schema().prefix()).await;
    let store = LayerStore::new(
        by(
            deadline,
            "the client",
            client_on(Some(proxy.url()), &metrics),
        )
        .await,
        tv.key_schema().clone(),
        4096,
    )
    .with_metrics(Arc::clone(&metrics));
    let state = state_of(sections_and_fields(3, 3));
    assert!(
        EXEC_DELAY > COMMAND_TIMEOUT * 2,
        "precondition: the delay passes the small create's scaled timeout"
    );

    let err = by(
        deadline,
        "the create",
        store.create_layer(
            &session_id,
            LAYER,
            "txn-timeout-user",
            &state,
            Arc::new(MockExternalValueStore::new()),
        ),
    )
    .await
    .expect_err("an object write past its scaled timeout must not answer OK");

    let prefix = object_write_timeout_prefix();
    assert!(
        matches!(&err, ServiceError::Unavailable(m) if m.starts_with(&prefix)),
        "UNAVAILABLE naming the timed-out object write; got {err:?}"
    );
    assert_eq!(
        proxy.hits(),
        1,
        "vacuity guard: the object write's XADD was delayed"
    );
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on("txn", 1),
        "the object write's timeout counts once, on the txn pool"
    );
    let headers = first_object_written(&tv, deadline).await;
    assert_eq!(headers.len(), 1, "one object materialized: {headers:?}");
    assert_eq!(
        by(deadline, "a store read", mapped_object(&tv, &session_id)).await,
        None,
        "the mapping is written only after the object write answers OK"
    );
}

/// The residual the scaled timeout leaves (T11 a): a create whose object
/// write's timeout is forced shorter than its `EXEC` (the test seam
/// `with_object_write_timeout_for_test`) answers `UNAVAILABLE` naming the
/// timed-out `EXEC`, counts one `command_timeout` on the txn pool, and the
/// transaction, which fred wrote in full before its timer fired, still lands:
/// the object materializes, unmapped, for the orphan sweep.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_layer_whose_object_write_times_out_answers_unavailable_and_its_object_still_materializes_unmapped(
) {
    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let session_id = by(deadline, "the session", new_session(&tv, "t11a-create")).await;
    let proxy = exec_delaying_proxy(tv.key_schema().prefix()).await;
    let store = LayerStore::new(
        by(
            deadline,
            "the client",
            client_on(Some(proxy.url()), &metrics),
        )
        .await,
        tv.key_schema().clone(),
        4096,
    )
    .with_metrics(Arc::clone(&metrics))
    .with_object_write_timeout_for_test(FORCED_OBJECT_WRITE_TIMEOUT);
    let state = state_of(sections_and_fields(3, 3));

    let err = by(
        deadline,
        "the create",
        store.create_layer(
            &session_id,
            LAYER,
            "txn-timeout-user",
            &state,
            Arc::new(MockExternalValueStore::new()),
        ),
    )
    .await
    .expect_err("an object write past its timeout must not answer OK");

    let prefix = object_write_timeout_prefix();
    assert!(
        matches!(&err, ServiceError::Unavailable(m) if m.starts_with(&prefix)),
        "UNAVAILABLE naming the timed-out object write; got {err:?}"
    );
    assert_eq!(
        proxy.hits(),
        1,
        "vacuity guard: the object write's XADD was delayed"
    );
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on("txn", 1),
        "the object write's timeout counts once, on the txn pool"
    );
    let headers = first_object_written(&tv, deadline).await;
    assert_eq!(headers.len(), 1, "one object materialized: {headers:?}");
    assert_eq!(
        by(deadline, "a store read", mapped_object(&tv, &session_id)).await,
        None,
        "the mapping is written only after the object write answers OK"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T11 (b) — a client deadline shorter than a WATCH-route reset (b.gvf item 4)
// ═════════════════════════════════════════════════════════════════════════════

/// The served stack's RPC deadline in T11 (b): what ends the client's call.
const RPC_DEADLINE_MS: u64 = 2_000;

/// The bearer every served call carries.
const BEARER: &str = "Bearer txn-timeout-test";

/// The full server stack around `svc` — `MetricsLayer` with its RPC deadline,
/// and the LiveService `transport::live_service_server` builds — on a local
/// port.
async fn serve(
    svc: live_edit::grpc::service::LiveEditService,
    deadline: Instant,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    use live_edit::grpc::metrics_layer::MetricsLayer;
    use live_edit::grpc::transport::{live_service_server, production_server_builder};

    let layer_metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let metrics_layer = MetricsLayer::new(layer_metrics, "test", RPC_DEADLINE_MS, 2_000);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a random port");
    let addr = listener.local_addr().expect("local addr");
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let _ = ready_tx.send(());
        production_server_builder()
            .layer(metrics_layer)
            .add_service(live_service_server(svc))
            .serve_with_incoming(incoming)
            .await
            .expect("test server error");
    });
    by(deadline, "the server's readiness", ready_rx)
        .await
        .expect("the server task dropped its ready channel");
    (addr, task)
}

/// One `UpdateLayer` of a reset delta through a tonic client at `addr`, with
/// no client-side timer: the server's RPC deadline is what can end it.
async fn served_reset(
    addr: std::net::SocketAddr,
    session_id: &str,
    object_id: u64,
    operation_id: &str,
) -> Result<live_edit::grpc::proto::UpdateLayerResponse, tonic::Status> {
    use live_edit::grpc::proto::{
        live_service_client::LiveServiceClient, update_layer_request, UpdateLayerChunk,
        UpdateLayerHeader, UpdateLayerRequest,
    };
    let reset = OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: (1..=10u64)
            .map(|id| {
                StructCommand::DiffChangeSection(oodf_ffi::CommandDiffChangeSection {
                    parent_id: 0,
                    section_id: id,
                    section_name: format!("r{id}"),
                    section_type: 1,
                    section_order: id as u32 - 1,
                })
            })
            .collect(),
        source_format: "usda".to_owned(),
        ..Default::default()
    };
    let messages = vec![
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Header(
                UpdateLayerHeader {
                    session_id: session_id.to_owned(),
                    layer_name: LAYER.to_owned(),
                    object_id,
                    timestamp_microseconds_since_epoch: 1_000,
                    client_id: "t11b-writer".to_owned(),
                    operation_id: operation_id.to_owned(),
                },
            )),
        },
        UpdateLayerRequest {
            message_type: Some(update_layer_request::MessageType::Chunk(UpdateLayerChunk {
                data: crate::common::owned_delta::delta_to_ompf_bytes(&reset, &[]),
            })),
        },
    ];
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .expect("valid URI")
        .connect()
        .await
        .expect("connect to the test server");
    let mut request = tonic::Request::new(tokio_stream::iter(messages));
    request
        .metadata_mut()
        .insert("authorization", BEARER.parse().expect("ASCII bearer"));
    LiveServiceClient::new(channel)
        .update_layer(request)
        .await
        .map(tonic::Response::into_inner)
}

/// b.gvf item 4, end to end: a WATCH-route reset whose commit is held past
/// the client's deadline (its connection parked by the pre-dispatch hook) is
/// answered `DEADLINE_EXCEEDED` — the outcome unknown to the client — yet the
/// commit, already written to its stalled connection before the deadline,
/// lands once when released. A re-send
/// with the same `operation_id` is answered OK `ALREADY_COMMITTED` at the
/// original sequence number, and applies nothing more.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_past_the_client_deadline_lands_once_and_its_resend_is_already_committed_end_to_end(
) {
    use crate::grpc::helpers::{build_disabled_recovery_hook, build_service, AllowAllPermissions};
    use live_edit::grpc::proto::OperationOutcome;

    let deadline = test_deadline();
    let tv = by(deadline, "the fixture", TestValkey::setup()).await;
    let session_id = by(deadline, "the session", new_session(&tv, "t11b-deadline")).await;
    let object_id = by(deadline, "the layer", new_layer(&tv, &session_id)).await;
    let before = by(deadline, "a store read", seq_and_stream(&tv, object_id)).await;

    let parked: Parked = Arc::default();
    let dispatches = Arc::new(AtomicUsize::new(0));
    let base = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Txn)
            .with_metrics(Arc::clone(&base.metrics))
            .with_pre_dispatch_hook_for_test(stall_first_dispatch(
                &tv,
                "t11b-txn-stall",
                Arc::clone(&parked),
                Arc::clone(&dispatches),
            )),
    );
    let svc = live_edit::grpc::service::LiveEditService {
        recovery: build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&base.metrics)),
        layer_store,
        ..base
    };
    let (addr, server) = serve(svc, deadline).await;

    let (first, (stall, _)) = tokio::join!(
        tokio::time::timeout_at(
            deadline,
            served_reset(addr, &session_id, object_id, "t11b-op")
        ),
        wait_parked(&parked, deadline),
    );
    let status = first
        .expect("the server did not answer within the scenario budget")
        .expect_err("the call must end at its deadline while its commit is held");
    assert_eq!(
        status.code(),
        tonic::Code::DeadlineExceeded,
        "the client reads its deadline, the outcome unknown: {status:?}"
    );
    assert_eq!(
        by(deadline, "a store read", seq_and_stream(&tv, object_id)).await,
        before,
        "the commit is still held when the client is answered"
    );

    release(stall, deadline).await;
    let landed = (before.0 + 1, before.1 + 1);
    while by(deadline, "a store read", seq_and_stream(&tv, object_id)).await != landed {
        assert!(
            Instant::now() < deadline,
            "the held reset never landed once released"
        );
        tokio::time::sleep(POLL).await;
    }

    let resent = by(
        deadline,
        "the re-send",
        served_reset(addr, &session_id, object_id, "t11b-op"),
    )
    .await
    .unwrap_or_else(|s| panic!("the re-send must be answered OK; got {s:?}"));
    assert_eq!(
        (
            OperationOutcome::try_from(resent.operation_outcome),
            resent.sequence_num
        ),
        (Ok(OperationOutcome::AlreadyCommitted), landed.0),
        "the re-send is answered ALREADY_COMMITTED at the original sequence"
    );
    assert_eq!(
        by(deadline, "a store read", seq_and_stream(&tv, object_id)).await,
        landed,
        "the reset landed once: the re-send applied nothing"
    );
    assert_eq!(
        dispatches.load(Ordering::SeqCst),
        1,
        "one dispatch: the held one; the re-send was answered from its record"
    );
    server.abort();
}
