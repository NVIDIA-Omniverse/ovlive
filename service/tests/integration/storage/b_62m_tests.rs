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

//! b.62m — a transient Valkey failure on a unary RPC's round trip must answer
//! a retriable class, not a hand-wrapped `INTERNAL`.
//!
//! # The defect
//!
//! Twelve data-pool round trips — on the CreateSession, DeleteSession,
//! ListSessions and CreateLayer paths, and in `load_values` — wrapped every fred
//! error as `ServiceError::Internal(format!("… failed: {e}"))`. A timeout, I/O
//! or connection error on one of them answered a redacted `INTERNAL` and
//! logged one ERROR `internal error` line, for what is a retriable
//! backing-store blip. The source-shape guard in `mapper_routing_test.rs` had
//! excused them under a "MULTI/EXEC queue-time" rationale that did not hold for
//! any of them.
//!
//! A site that is the first data-pool command on its path fails alone under a
//! whole-pool stall. Four such sites are driven directly below: `delete`'s
//! `HGETALL`, `list`'s `SMEMBERS`, `get_session_meta` and `load_values`.
//! CreateSession's metadata `HSET` is another — `create_once`'s `WATCH`,
//! duplicate-scan `SMEMBERS` and `EXEC` all run on the txn checkout, never the
//! data pool — and so, on the in-doubt path, is the reconciling `SISMEMBER`; the
//! `undo` module drives both. It also drives the in-doubt re-scan `SMEMBERS`,
//! which does sit behind a data-pool command (that `SISMEMBER`), by letting the
//! `SISMEMBER` through a single connection (see *Staging*). The remaining
//! converted sites — `delete`'s two `SREM`s, `list`'s per-session `HGETALL`,
//! `create_layer`'s mapping `GET` / `SET`, and the overwrite's
//! `HSET obj:hdr superseded` (moved off the in-loop mapper) — each sit behind
//! an earlier data-pool command and are guarded per site by
//! `mapper_routing_test.rs` alone. The CreateLayer overwrite's old-stream
//! `XADD` is driven through its fred-error seam in
//! `update_layer_admission_test.rs`, and the ListSessions handler's log line in
//! `grpc/rpc_failure_log_test.rs`.
//!
//! CreateSession's post-commit failures fail through
//! `SessionStore::undo_create_attempt`: the call retries only once the undo's
//! names `SREM` has released the attempt's member, and otherwise answers
//! `UNAVAILABLE`, never `Aborted` (why, per caller: see `undo_create_attempt` /
//! SDD §7.10 step 4). The `undo` module drives both outcomes through each of
//! the three callers — the metadata `HSET`, the in-doubt `SISMEMBER`, and the
//! in-doubt re-scan `SMEMBERS` that follows a `SISMEMBER` reading "present" —
//! and the failed-undo cases pin the not-released WARN verbatim. The shape of
//! the three call sites (no direct mapper call, one `undo_create_attempt` per
//! round trip, each with its op) is pinned separately in
//! `mapper_routing_test.rs`.
//!
//! The partial-undo verdict — names `SREM` succeeds, `path_index` `SREM` and
//! `DEL` fail — is driven through the metadata `HSET` only: the call must still
//! retry under a new id into a successful create, with one partial-undo WARN,
//! and the failed commands' leftovers stay under the abandoned id. One caller
//! is enough because the verdict is decided inside `undo_create_attempt`, after
//! the caller has handed over.
//!
//! Two tests pin the per-attempt `session_id` (see `undo_create_attempt` /
//! SDD §7.10): in each, a command of an abandoned attempt reaches the store
//! after a later attempt has moved on — the partial undo's timed-out `SREM` and
//! `DEL` landing after the call returned `Ok`, and an in-doubt `EXEC` landing
//! after its `SISMEMBER` read "absent" — and must leave the call's result
//! intact.
//!
//! # What lives here
//!
//! | Test | Site | Pre-fix answer |
//! |---|---|---|
//! | [`session_delete_timed_out_meta_read_answers_unavailable_and_changes_nothing`] | `delete` step 1 `HGETALL` | `Internal` |
//! | [`session_list_timed_out_names_read_answers_unavailable`] | `list` `SMEMBERS` | `Internal` |
//! | [`session_get_meta_timed_out_read_answers_unavailable`] | `get_session_meta` `HGETALL` | `Internal` |
//! | [`ext_load_values_timed_out_read_answers_unavailable`] | `load_values` `GET ext_data` | `Internal` |
//! | [`undo::create_session_timed_out_meta_write_undoes_the_attempt_and_retries_to_success`] | `finish_create` `HSET`, undo releases the name | `Err(Internal)` |
//! | [`undo::create_session_timed_out_meta_write_with_a_failed_undo_answers_unavailable`] | `finish_create` `HSET`, undo fails | `Err(Internal)` |
//! | [`undo::create_in_doubt_timed_out_reconcile_read_undoes_the_landed_commit_and_retries_to_success`] | in-doubt `SISMEMBER`, undo releases the name | `Err(AlreadyExists)` |
//! | [`undo::create_in_doubt_timed_out_reconcile_read_with_a_failed_undo_answers_unavailable`] | in-doubt `SISMEMBER`, undo fails | `Err(AlreadyExists)` |
//! | [`undo::create_in_doubt_timed_out_rescan_undoes_the_landed_commit_and_retries_to_success`] | in-doubt re-scan `SMEMBERS`, undo releases the name | `Err(Internal)` |
//! | [`undo::create_in_doubt_timed_out_rescan_with_a_failed_undo_answers_unavailable`] | in-doubt re-scan `SMEMBERS`, undo fails | `Err(Internal)` |
//! | [`undo::create_session_timed_out_meta_write_with_a_partial_undo_releases_the_name_and_retries_to_success`] | `finish_create` `HSET`, undo releases the name but its other two commands fail | `Err(Internal)` |
//! | [`undo::create_session_partial_undo_commands_landing_after_ok_leave_the_new_session_intact`] | the same, with the failed undo commands landing after `Ok` | with the undo but a retry reusing its id: `Ok`, then that session left a ghost |
//! | [`undo::create_in_doubt_exec_landing_late_does_not_make_the_call_answer_already_exists`] | an in-doubt `EXEC` landing after its `SISMEMBER` read "absent" | with a retry reusing its id: `Err(AlreadyExists)` |
//!
//! Every test also compares the WHOLE `command_timeout` row, so a manual
//! `observe_error` left beside a mapper (which already counts) shows up as a
//! double count, and a tick on the wrong pool as a wrong cell.
//!
//! # Staging
//!
//! Every timeout is fred's real one. [`TestValkey::stall_data_pool`] parks a
//! server-side `BLPOP` on each data-pool connection, so any data-pool command
//! queues behind it and outlives the fixture's 5 s command timeout. Every
//! CreateSession test except the late-`EXEC` one stalls the data pool from a
//! one-shot `with_pre_dispatch_hook_for_test` hook, i.e. after the duplicate
//! scan and before the commit, and leaves the transaction pool carrying
//! `WATCH` / `EXEC` unstalled; the first data-pool command after the stall is
//! exactly the one under test. The in-doubt pair additionally replaces the
//! commit's verdict with `ReplaceOutcomeAfterExec` (the commit lands, its
//! reply is reported lost), which is what routes the
//! attempt into `reconcile_in_doubt_create`'s `SISMEMBER`. The re-scan pair
//! then lets that `SISMEMBER` through on its own: fred's data `Pool` hands each
//! command to the next connection in turn, so releasing only the connection
//! the `SISMEMBER`'s bytes reached answers it ("present"), while the re-scan
//! `SMEMBERS` right behind it lands on another, still-stalled connection. That
//! connection is found from its bytes, not predicted, and is stalled again
//! straight away, so the undo meets a fully stalled pool as in every other
//! case. Each parked connection's byte count is sampled when it is parked, so
//! "new bytes" never races a command already on its way.
//!
//! The two partial-undo tests split the undo itself. Its three commands are
//! sent concurrently, each to the next data-pool connection in turn, so with
//! the fixture's four connections (asserted in `UndoSizes::new`) they land on
//! three different ones, none of them the connection holding the timed-out
//! `HSET`. They are told apart by the bytes each connection received, matched
//! against the three commands' exact RESP sizes (the names `SREM` carries a
//! longer key and member than the `path_index` `SREM`, and the `DEL` one
//! argument fewer). Only the names `SREM`'s connection is released; the other
//! two time out. What happens to them next is what separates the two tests:
//! - **kill** (the partial-undo verdict test): they never run, which pins the
//!   inert leftovers that stay under the abandoned id for good — its
//!   `path_index` member, and the orphan `session_meta` its released original
//!   `HSET` writes;
//! - **release after `Ok`** (the late-undo test): every other stall is released
//!   first, so the retry commits and `create` returns; the abandoned id's
//!   original `HSET` is then confirmed landed, and only after that are the two
//!   undo commands released and run — late, against an id the returned session
//!   does not use.
//!
//! The late-`EXEC` test stalls the transaction pool instead, and only one
//! connection of it: attempt 1's pre-dispatch hook parks the txn connection
//! carrying its own `WATCH`, so its commit pipeline queues behind the stall and
//! its reply times out in doubt, while the reconciling `SISMEMBER` runs on the
//! unstalled data pool and reads "absent". Attempt 1's `WatchGuard` keeps that
//! txn slot (and its permit) until its own `UNWATCH`, queued behind the same
//! stall, completes, so attempt 2 checks out another slot and never touches the
//! stalled connection. Attempt 2's watch-window hook then releases the stall,
//! so the parked `EXEC` commits late, inside attempt 2's `WATCH` window.
//!
//! A stall ends one of two ways. `release` lets every queued command run. The
//! "retries to success" cases release as soon as the undo's first command
//! reaches a stalled connection — detected from the connections' byte counts,
//! not from the timeout metric, because `undo_create_attempt` counts the
//! failing command's timeout only after its own commands return — so the undo
//! completes. `kill` closes the connection, so everything still queued behind
//! the `BLPOP` dies unrun; fred does not replay a command that already timed out
//! (`router/mod.rs::retry_buffer` skips `timed_out` commands). The failed-undo
//! cases kill every stall and so end with a deterministic keyspace — the
//! attempt's members and no metadata; the partial-undo verdict test kills only
//! the two undo connections, as above.
//!
//! # Feature gate
//!
//! The four single-site tests need only `test-support`. The `undo` module uses
//! `with_pre_dispatch_hook_for_test` / `with_watched_exec_override_for_test`,
//! which exist only under `test-fault-injection`, so without that feature the
//! module is compiled out and reports nothing — pass
//! `--features test-support,test-fault-injection` on any hand-rolled run.
//!
//! # Wait budgets vs. nextest's hang-killer
//!
//! The operation under test runs under ONE deadline, [`SCENARIO_BUDGET`]; in
//! the success cases the watcher waiting for the undo runs under the same one.
//! Parking and ending a stall are bounded separately by
//! `test_support::CONN_STALL_WAIT` per connection, and the post-kill reconnect
//! by `undo::RECONNECT_BUDGET`.
//!
//! | Test | Healthy path | Operation deadline |
//! |---|---|---|
//! | the four single-site tests | one command timeout, ≈ 5 s | 30 s |
//! | the three "retries to success" tests | one command timeout, ≈ 5 s | 30 s |
//! | the three "failed undo" tests | two command timeouts in sequence (the failure, then the undo), ≈ 10 s | 30 s |
//! | the two partial-undo tests | two command timeouts in sequence (the `HSET`, then two of the undo's commands), ≈ 10 s | 30 s |
//! | the late-`EXEC` test | one command timeout (the in-doubt `EXEC`, on the txn pool), ≈ 5 s | 30 s |
//!
//! Each is slow (> 1 s) only because a real command timeout is the point of the
//! test.

use std::sync::Arc;
use std::time::Duration;

use live_edit::error::ServiceError;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{
    command_timeout_row, command_timeouts_only_on, release_all, TestValkey,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets — see the module-level table
// ═════════════════════════════════════════════════════════════════════════════

/// One deadline over the operation under test. The slowest healthy path is two
/// 5 s command timeouts in sequence; 30 s leaves ample room for CI runner load
/// and keeps nextest's 120 s hang-killer a backstop.
const SCENARIO_BUDGET: Duration = Duration::from_secs(30);

const CREATED_BY: &str = "b62m-user";

// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

/// A fresh fixture with a metrics registry installed on the client, so every
/// `FredErrOrigin::client` count lands on `metrics`.
async fn fixture() -> (TestValkey, Arc<Metrics>) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    (tv, metrics)
}

fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

/// Assert `err` is the no-loop mapper's transient class for a timed-out `op`:
/// `Unavailable("{op} failed ({kind}): …")`, `kind` being fred's own spelling
/// of `ErrorKind::Timeout`. The prefix pins the exact op label (several labels
/// share a stem, e.g. `HGETALL session meta` and
/// `HGETALL session meta (delete)`), and the variant is the whole point —
/// before b.62m each of these sites answered `Internal`.
fn assert_unavailable_timeout(err: &ServiceError, op: &str) {
    let prefix = format!(
        "{op} failed ({}): ",
        fred::error::ErrorKind::Timeout.to_str()
    );
    match err {
        ServiceError::Unavailable(msg) => assert!(
            msg.starts_with(&prefix),
            "the Unavailable message must start {prefix:?}; got {msg:?}"
        ),
        other => panic!(
            "a timed-out `{op}` is a transient store failure and must answer a retriable \
             Unavailable; got {other:?}"
        ),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Single-site timeouts (no fault-injection feature needed)
// ═════════════════════════════════════════════════════════════════════════════

/// A `DeleteSession` whose first read times out answers `Unavailable` and
/// changes nothing: the read runs before any write, so the session is still
/// there, listed and complete, once the store recovers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_delete_timed_out_meta_read_answers_unavailable_and_changes_nothing() {
    const NAME: &str = "b62m-delete";
    const BASE_PATH: &str = "/b62m-delete";

    let (tv, metrics) = fixture().await;
    let store = session_store(&tv);
    let session_id = store
        .create(NAME, BASE_PATH, CREATED_BY)
        .await
        .expect("fixture session creation must succeed");

    let stalls = tv.stall_data_pool().await;
    let outcome = tokio::time::timeout(SCENARIO_BUDGET, store.delete(&session_id)).await;
    release_all(stalls).await;

    let err = outcome
        .expect("the stalled delete did not settle within the scenario budget")
        .expect_err("a timed-out first read must fail the delete");
    assert_unavailable_timeout(&err, "HGETALL session meta (delete)");
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on("data", 1),
        "the timed-out HGETALL counts exactly once, on the data pool"
    );

    let listed = store
        .list(BASE_PATH)
        .await
        .expect("list must succeed once the stall is released");
    assert_eq!(
        listed
            .present
            .iter()
            .map(|s| s.session_id.as_str())
            .collect::<Vec<_>>(),
        vec![session_id.as_str()],
        "a delete that failed on its first read must leave the session in place"
    );
    assert!(listed.ghosts.is_empty(), "nothing was half-removed");
}

/// A `ListSessions` whose names read times out answers `Unavailable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_list_timed_out_names_read_answers_unavailable() {
    let (tv, metrics) = fixture().await;
    let store = session_store(&tv);

    let stalls = tv.stall_data_pool().await;
    let outcome = tokio::time::timeout(SCENARIO_BUDGET, store.list("/b62m-list")).await;
    release_all(stalls).await;

    let err = match outcome.expect("the stalled list did not settle within the scenario budget") {
        Ok(listed) => panic!("a timed-out names read must fail the list; got {listed:?}"),
        Err(e) => e,
    };
    assert_unavailable_timeout(&err, "SMEMBERS path names index (list)");
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on("data", 1),
        "the timed-out SMEMBERS counts exactly once, on the data pool"
    );
}

/// `get_session_meta` (the ListSessions ghost re-read) whose `HGETALL` times
/// out answers `Unavailable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_get_meta_timed_out_read_answers_unavailable() {
    let (tv, metrics) = fixture().await;
    let store = session_store(&tv);
    let session_id = store
        .create("b62m-get-meta", "/b62m-get-meta", CREATED_BY)
        .await
        .expect("fixture session creation must succeed");

    let stalls = tv.stall_data_pool().await;
    let outcome = tokio::time::timeout(SCENARIO_BUDGET, store.get_session_meta(&session_id)).await;
    release_all(stalls).await;

    let err = match outcome
        .expect("the stalled get_session_meta did not settle within the scenario budget")
    {
        Ok(meta) => panic!("a timed-out HGETALL must fail the read; got {meta:?}"),
        Err(e) => e,
    };
    assert_unavailable_timeout(&err, "HGETALL session meta");
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on("data", 1),
        "the timed-out HGETALL counts exactly once, on the data pool"
    );
}

/// `load_values` whose `GET ext_data` times out answers `Unavailable`. One hash,
/// so exactly one `GET` is issued and exactly one timeout counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ext_load_values_timed_out_read_answers_unavailable() {
    let (tv, metrics) = fixture().await;
    let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone());
    let hash = [0x62u8; 32];

    let stalls = tv.stall_data_pool().await;
    let outcome = tokio::time::timeout(SCENARIO_BUDGET, store.load_values(&[&hash])).await;
    release_all(stalls).await;

    let err = match outcome.expect("the stalled load_values did not settle within the budget") {
        Ok(values) => panic!(
            "a timed-out GET must fail the load; got {} value(s)",
            values.len()
        ),
        Err(e) => e,
    };
    assert_unavailable_timeout(&err, "GET ext_data");
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on("data", 1),
        "the timed-out GET counts exactly once, on the data pool"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// CreateSession: undo the attempt after a post-commit failure
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod undo {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use fred::clients::Client;
    use fred::interfaces::{ClientLike, KeysInterface, SetsInterface};
    use futures::FutureExt;
    use tonic::Request;

    use live_edit::grpc::proto::ListSessionsRequest;
    use live_edit::grpc::session::handle_list_sessions;
    use live_edit::grpc::BearerToken;
    use live_edit::storage::test_support::ConnStall;
    use live_edit::storage::watched_txn::{
        WatchWindowHook, WatchedExecOverride, WatchedExecOverrideQueue,
    };

    use super::*;
    use crate::grpc::helpers::{build_service, AllowAllPermissions};

    /// How long fred may take to re-establish the data-pool connections a test
    /// killed. A healthy reconnect takes a few hundred milliseconds (100 ms
    /// exponential floor).
    const RECONNECT_BUDGET: Duration = Duration::from_secs(10);

    /// Poll interval for [`Staged::wait_until_the_undo_arrives`].
    const POLL_INTERVAL: Duration = Duration::from_millis(5);

    /// The names-index members holding `name` at `base_path`.
    async fn name_members(tv: &TestValkey, base_path: &str, name: &str) -> Vec<String> {
        let members: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_names_index(base_path))
            .await
            .expect("SMEMBERS on the names index must succeed");
        members
            .into_iter()
            .filter(|m| m.split('\u{0}').next() == Some(name))
            .collect()
    }

    /// Which post-commit round trip a scenario makes fail.
    #[derive(Clone, Copy)]
    enum Failing {
        /// The commit lands normally; `finish_create`'s `HSET session_meta`
        /// times out.
        MetaWrite,
        /// The commit lands but its reply is reported lost
        /// (`ReplaceOutcomeAfterExec`); `reconcile_in_doubt_create`'s
        /// `SISMEMBER` — the read that asks whether it landed — times out.
        InDoubtReconcile,
        /// The same lost reply, but the `SISMEMBER` is let through alone (it
        /// reads "present"), and the re-scan `SMEMBERS` that follows a landed
        /// commit times out. See [`Staged::let_the_sismember_through`].
        InDoubtRescan,
    }

    impl Failing {
        /// The op label the failure surfaces under.
        fn op(self) -> &'static str {
            match self {
                Failing::MetaWrite => "HSET session meta (create)",
                Failing::InDoubtReconcile => "SISMEMBER path names index (in-doubt reconcile)",
                Failing::InDoubtRescan => "SMEMBERS path names index (in-doubt re-scan)",
            }
        }
    }

    /// One stalled data-pool connection: the pool client it sits on, its
    /// stall, and the bytes the server had read from it once the stall was in
    /// place.
    ///
    /// The baseline is sampled right after parking, before the attempt sends
    /// anything more, so "this connection has new bytes" means exactly "a
    /// command of the attempt reached it" — no sampling race with a command
    /// that was already on its way.
    struct ParkedConn {
        conn: Client,
        stall: ConnStall,
        parked_bytes: u64,
    }

    impl ParkedConn {
        async fn park(conn: &Client, observer: &Client, key: String) -> Self {
            let stall = ConnStall::park(conn, observer, key).await;
            let parked_bytes = stall
                .net_in()
                .await
                .expect("a freshly stalled data-pool connection must be open");
            ParkedConn {
                conn: conn.clone(),
                stall,
                parked_bytes,
            }
        }

        async fn bytes_in(&self) -> u64 {
            self.stall
                .net_in()
                .await
                .expect("a stalled data-pool connection must stay open")
        }
    }

    /// Index of the first parked connection that has read bytes since it was
    /// parked, if any.
    async fn first_with_new_bytes(held: &[ParkedConn]) -> Option<usize> {
        for (i, parked) in held.iter().enumerate() {
            if parked.bytes_in().await != parked.parked_bytes {
                return Some(i);
            }
        }
        None
    }

    /// Bytes the server has read across every parked connection, including
    /// commands still queued behind the stalls.
    async fn total_bytes_in(held: &[ParkedConn]) -> u64 {
        let mut total = 0;
        for parked in held {
            total += parked.bytes_in().await;
        }
        total
    }

    /// A `SessionStore` whose first attempt stalls every data-pool connection
    /// just before its commit is sent, plus the handles to end the stall.
    struct Staged {
        store: SessionStore,
        failing: Failing,
        held: Arc<tokio::sync::Mutex<Vec<ParkedConn>>>,
        parked: Arc<AtomicBool>,
        overrides: Option<WatchedExecOverrideQueue>,
        observer: Client,
        key_prefix: String,
    }

    impl Staged {
        fn new(tv: &TestValkey, failing: Failing) -> Self {
            if matches!(failing, Failing::InDoubtRescan) {
                assert!(
                    tv.client().pool.clients().len() >= 2,
                    "the re-scan case needs the SMEMBERS to land on a different data-pool \
                     connection than the SISMEMBER it follows"
                );
            }
            let held: Arc<tokio::sync::Mutex<Vec<ParkedConn>>> = Arc::default();
            let parked = Arc::new(AtomicBool::new(false));
            let observer = tv.client().maintenance_pool.next().clone();
            let key_prefix = format!("{}b62m-data-stall:", tv.prefix());
            // One-shot: only the first attempt stalls, so a retry runs against
            // a store that has been released.
            let hook: WatchWindowHook = {
                let (held, parked) = (held.clone(), parked.clone());
                let data = tv.client().pool.clone();
                let (observer, key_prefix) = (observer.clone(), key_prefix.clone());
                Arc::new(move |_guarded: &Client| {
                    let (held, parked) = (held.clone(), parked.clone());
                    let (data, observer, key_prefix) =
                        (data.clone(), observer.clone(), key_prefix.clone());
                    async move {
                        if parked.load(Ordering::SeqCst) {
                            return;
                        }
                        let mut held = held.lock().await;
                        for (i, conn) in data.clients().iter().enumerate() {
                            held.push(
                                ParkedConn::park(conn, &observer, format!("{key_prefix}{i}")).await,
                            );
                        }
                        parked.store(true, Ordering::SeqCst);
                    }
                    .boxed()
                })
            };
            let mut store = session_store(tv).with_pre_dispatch_hook_for_test(hook);
            let overrides = match failing {
                Failing::MetaWrite => None,
                Failing::InDoubtReconcile | Failing::InDoubtRescan => {
                    let queue: WatchedExecOverrideQueue = Arc::new(Mutex::new(VecDeque::from([
                        WatchedExecOverride::ReplaceOutcomeAfterExec,
                    ])));
                    store = store.with_watched_exec_override_for_test(queue.clone());
                    Some(queue)
                }
            };
            Staged {
                store,
                failing,
                held,
                parked,
                overrides,
                observer,
                key_prefix,
            }
        }

        /// Wait for the pre-dispatch hook to finish parking the data pool.
        async fn wait_parked(&self, deadline: Instant) {
            while !self.parked.load(Ordering::SeqCst) {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for the pre-dispatch hook to park the data pool"
                );
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }

        /// For [`Failing::InDoubtRescan`] only (a no-op otherwise): let the
        /// reconciling `SISMEMBER` through on its own, then stall its
        /// connection again.
        ///
        /// fred's data `Pool` hands each command to the next connection in turn
        /// (`fred-10.1.0/src/clients/pool.rs`, `Pool::next`), and a `ConnStall`
        /// ends one connection at a time. Nothing uses the data pool between
        /// the park and the `SISMEMBER`, so the one connection with new bytes
        /// is the `SISMEMBER`'s; releasing only it answers the `SISMEMBER`
        /// ("present" — the commit landed), and the re-scan `SMEMBERS` sent
        /// straight after goes to the next connection, which is still stalled.
        /// The connection is found from its bytes, not predicted, so the only
        /// thing relied on is that two consecutive commands take two different
        /// connections, which holds for a pool of two or more.
        ///
        /// Re-stalling the released connection (which holds nothing else — the
        /// undo is 5 s away) leaves the whole pool stalled again, so what
        /// follows is the same shape as the other cases: the failing command,
        /// then the undo, whichever connections they land on.
        async fn let_the_sismember_through(&self) {
            if !matches!(self.failing, Failing::InDoubtRescan) {
                return;
            }
            let deadline = Instant::now() + SCENARIO_BUDGET;
            self.wait_parked(deadline).await;
            let mut held = self.held.lock().await;
            let idx = loop {
                if let Some(idx) = first_with_new_bytes(&held).await {
                    break idx;
                }
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for the reconciling SISMEMBER to reach a stalled \
                     data-pool connection"
                );
                tokio::time::sleep(POLL_INTERVAL).await;
            };
            let ParkedConn { conn, stall, .. } = held.remove(idx);
            let id = stall.conn_id().to_owned();
            let reply = stall.release().await;
            assert!(
                matches!(&reply, Ok(v) if !v.is_null()),
                "the stall on connection {id} must have held until the SISMEMBER was let \
                 through; got {reply:?}"
            );
            held.push(
                ParkedConn::park(
                    &conn,
                    &self.observer,
                    format!("{}again-{idx}", self.key_prefix),
                )
                .await,
            );
        }

        /// Assert the stall fired and any verdict override was consumed, so the
        /// scenario ran the path it claims to.
        fn assert_staged(&self) {
            assert!(
                self.parked.load(Ordering::SeqCst),
                "the pre-dispatch hook must have stalled the data pool"
            );
            if let Some(queue) = &self.overrides {
                assert!(
                    queue.lock().unwrap().is_empty(),
                    "the in-doubt verdict override must have been consumed"
                );
            }
        }

        /// Return as soon as the undo's first command reaches a stalled
        /// connection (after [`Self::let_the_sismember_through`], for the
        /// re-scan case). The caller then releases the stalls, so the undo is
        /// answered well inside its own 5 s deadline while the failing command
        /// it follows has already timed out. Waits only — it never releases —
        /// so dropping it mid-wait leaves every stall intact for
        /// [`Self::release_remaining`].
        ///
        /// Keyed to the traffic — the bytes the server has read from the
        /// stalled connections — not to the timeout metric. The metric cannot
        /// serve: `undo_create_attempt` counts the failing command's timeout
        /// only after its own three commands have returned, so a watcher
        /// waiting for that count would hold the stall until the undo had timed
        /// out too, turning every run into the failed-undo case. Nothing else
        /// uses the data pool under the stall, so the first new bytes after the
        /// park are the failing command, and the next are the undo's, which it
        /// sends only once the failing command has timed out.
        async fn wait_until_the_undo_arrives(&self) {
            self.let_the_sismember_through().await;
            let deadline = Instant::now() + SCENARIO_BUDGET;
            let wait_until = |what: &str| {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for {what} to reach a stalled data-pool connection"
                );
            };
            self.wait_parked(deadline).await;
            let held = self.held.lock().await;
            while first_with_new_bytes(&held).await.is_none() {
                wait_until("the failing command");
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            // The failing command is one small frame, and nothing more is sent
            // on the data pool until its 5 s timeout, so this settle is slack
            // against a frame split across reads, not a race.
            tokio::time::sleep(UNDO_SETTLE).await;
            let after_failing = total_bytes_in(&held).await;
            while total_bytes_in(&held).await == after_failing {
                wait_until("the undo's first command");
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }

        /// Release every stall still held; a no-op once they all are.
        async fn release_remaining(&self) {
            let held: Vec<ConnStall> = self
                .held
                .lock()
                .await
                .drain(..)
                .map(|parked| parked.stall)
                .collect();
            release_all(held).await;
        }

        /// Kill every stalled connection, so every command still queued behind
        /// a stall dies unrun.
        async fn kill_stalls(&self) {
            let held: Vec<ConnStall> = self
                .held
                .lock()
                .await
                .drain(..)
                .map(|parked| parked.stall)
                .collect();
            for stall in held {
                kill_stall(stall).await;
            }
        }

        /// For the partial-undo case: once the undo's three commands have
        /// reached three stalled connections, let only the names `SREM`
        /// through, so the undo releases the name while its `path_index`
        /// `SREM` and `DEL` time out.
        ///
        /// The three commands are sent concurrently, each to the next data-pool
        /// connection in turn, so with the fixture's four connections (asserted
        /// in [`UndoSizes::new`]) they land on three different ones — none of
        /// them the connection holding the timed-out `HSET`, which, one step
        /// further round, is also where the retry's `HSET` goes. Which command
        /// took which connection is read from the bytes
        /// each connection received, matched against the commands' exact wire
        /// sizes ([`UndoSizes`]); nothing about fred's assignment order is
        /// predicted.
        ///
        /// Before returning it reads, from the observer connection (the data
        /// pool is still stalled), what the partial undo has left in the store
        /// at that moment, and it re-baselines the remaining stalls so
        /// [`Self::wait_until_the_retry_arrives`] sees only the retry's traffic.
        async fn let_only_the_names_srem_through(
            &self,
            tv: &TestValkey,
            name: &str,
            base_path: &str,
            sizes: &UndoSizes,
        ) -> PartialUndo {
            let deadline = Instant::now() + SCENARIO_BUDGET;
            let wait_until = |what: &str| {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for {what} to reach a stalled data-pool connection"
                );
            };
            self.wait_parked(deadline).await;
            let mut held = self.held.lock().await;
            while first_with_new_bytes(&held).await.is_none() {
                wait_until("the metadata HSET");
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            // Nothing more is sent on the data pool until the HSET's 5 s
            // timeout; see `UNDO_SETTLE`.
            tokio::time::sleep(UNDO_SETTLE).await;
            let before_undo = bytes_each(&held).await;
            loop {
                let now = bytes_each(&held).await;
                let moved = now.iter().zip(&before_undo).filter(|(n, b)| n != b).count();
                if moved >= 3 {
                    break;
                }
                wait_until("the undo's three commands");
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            // Let the three small frames finish arriving before measuring them.
            tokio::time::sleep(UNDO_SETTLE).await;
            let deltas: Vec<u64> = bytes_each(&held)
                .await
                .iter()
                .zip(&before_undo)
                .map(|(now, before)| now - before)
                .collect();
            let mut received: Vec<u64> = deltas.iter().copied().filter(|d| *d > 0).collect();
            received.sort_unstable();
            let mut expected = vec![sizes.names_srem, sizes.index_srem, sizes.meta_del];
            expected.sort_unstable();
            assert_eq!(
                received, expected,
                "the undo must have sent its three commands to three different connections, \
                 with exactly the wire sizes of the names SREM, the path_index SREM and the \
                 DEL ({sizes:?}); per-connection bytes received: {deltas:?}"
            );
            let names_idx = deltas
                .iter()
                .position(|d| *d == sizes.names_srem)
                .expect("the names SREM's size is among the received ones");
            let undo_conn_ids: Vec<String> = deltas
                .iter()
                .enumerate()
                .filter(|(i, d)| **d > 0 && *i != names_idx)
                .map(|(i, _)| held[i].stall.conn_id().to_owned())
                .collect();
            let ParkedConn { stall, .. } = held.remove(names_idx);
            let id = stall.conn_id().to_owned();
            let reply = stall.release().await;
            assert!(
                matches!(&reply, Ok(v) if !v.is_null()),
                "the stall on connection {id} must have held until the names SREM was let \
                 through; got {reply:?}"
            );
            // The retry cannot start before the other two undo commands time
            // out (5 s from now), so this baseline holds only undo traffic.
            for parked in held.iter_mut() {
                parked.parked_bytes = parked.bytes_in().await;
            }
            drop(held);

            // What the partial undo left behind: the names member is gone, the
            // path_index member is still there, and there is no metadata (the
            // timed-out HSET is still queued behind its stall).
            let names_key = tv.key_schema().path_names_index(base_path);
            loop {
                let members: Vec<String> = self
                    .observer
                    .smembers(&names_key)
                    .await
                    .expect("SMEMBERS on the observer must succeed");
                if !members
                    .iter()
                    .any(|m| m.split('\u{0}').next() == Some(name))
                {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "the released names SREM never removed the attempt's names member"
                );
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            let indexed: Vec<String> = self
                .observer
                .smembers(tv.key_schema().path_index(base_path))
                .await
                .expect("SMEMBERS on the observer must succeed");
            assert_eq!(
                indexed.len(),
                1,
                "the failed path_index SREM must leave the attempt's session id indexed: \
                 {indexed:?}"
            );
            let session_id = indexed[0].clone();
            let meta_keys: i64 = self
                .observer
                .exists(tv.key_schema().session_meta(&session_id))
                .await
                .expect("EXISTS on the observer must succeed");
            assert_eq!(
                meta_keys, 0,
                "no metadata can exist yet: the timed-out HSET is still queued behind its stall"
            );
            PartialUndo {
                session_id,
                undo_conn_ids,
            }
        }

        /// Return, once any still-stalled connection has read new bytes since
        /// its last baseline, that connection's id — for the partial-undo case,
        /// the connection the retry's first data-pool command (its metadata
        /// `HSET`) reached.
        async fn wait_until_the_retry_arrives(&self) -> String {
            let deadline = Instant::now() + SCENARIO_BUDGET;
            let held = self.held.lock().await;
            loop {
                if let Some(idx) = first_with_new_bytes(&held).await {
                    return held[idx].stall.conn_id().to_owned();
                }
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for the retry's metadata HSET to reach a stalled \
                     data-pool connection"
                );
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }

        /// Release every stall still held except those in `keep_ids`, which
        /// stay parked for a later [`Self::release_remaining`].
        async fn release_all_but(&self, keep_ids: &[String]) {
            let mut held = self.held.lock().await;
            let (kept, released): (Vec<ParkedConn>, Vec<ParkedConn>) = held
                .drain(..)
                .partition(|parked| keep_ids.iter().any(|id| id == parked.stall.conn_id()));
            *held = kept;
            drop(held);
            release_all(released.into_iter().map(|parked| parked.stall).collect()).await;
        }

        /// End every stall still held: first kill the connections in
        /// `kill_ids` (their queued commands die unrun), then release the rest.
        async fn kill_then_release(&self, kill_ids: &[String]) {
            let held: Vec<ConnStall> = self
                .held
                .lock()
                .await
                .drain(..)
                .map(|parked| parked.stall)
                .collect();
            let (to_kill, to_release): (Vec<ConnStall>, Vec<ConnStall>) = held
                .into_iter()
                .partition(|stall| kill_ids.iter().any(|id| id == stall.conn_id()));
            for stall in to_kill {
                kill_stall(stall).await;
            }
            release_all(to_release).await;
        }
    }

    /// Kill one stalled connection, asserting the stall held until then.
    async fn kill_stall(stall: ConnStall) {
        let id = stall.conn_id().to_owned();
        let result = stall.kill().await;
        assert!(
            result.is_err(),
            "the stall on connection {id} must have held until it was killed; got {result:?}"
        );
    }

    /// Bytes each parked connection has read so far, in `held` order.
    async fn bytes_each(held: &[ParkedConn]) -> Vec<u64> {
        let mut out = Vec::with_capacity(held.len());
        for parked in held {
            out.push(parked.bytes_in().await);
        }
        out
    }

    /// What [`Staged::let_only_the_names_srem_through`] observed.
    struct PartialUndo {
        /// The attempt's session id, read off the `path_index` member the
        /// partial undo left behind.
        session_id: String,
        /// The connections holding the undo's timed-out `path_index` `SREM`
        /// and `DEL`.
        undo_conn_ids: Vec<String>,
    }

    /// A session id's shape as sent on the wire: a v4 UUID's 36-character
    /// hyphenated form. Only its length matters to [`UndoSizes`].
    const SESSION_ID_SHAPE: &str = "00000000-0000-0000-0000-000000000000";

    /// The exact wire size of each of `undo_create_attempt`'s three commands for
    /// a given name and base path.
    #[derive(Debug)]
    struct UndoSizes {
        names_srem: u64,
        index_srem: u64,
        meta_del: u64,
    }

    impl UndoSizes {
        fn new(tv: &TestValkey, name: &str, base_path: &str) -> Self {
            // The partial-undo staging relies on fred's round-robin over exactly
            // four data connections: the undo's three commands land on the three
            // connections after the failed `HSET`'s, and the retry's `HSET` comes
            // round to that one again. A different pool size changes both.
            assert_eq!(
                tv.client().pool.clients().len(),
                4,
                "the partial-undo staging assumes the fixture's four data-pool connections"
            );
            let keys = tv.key_schema();
            let member =
                live_edit::storage::keys::KeySchema::path_name_member(name, SESSION_ID_SHAPE);
            let sizes = UndoSizes {
                names_srem: resp_len(&["SREM", &keys.path_names_index(base_path), &member]),
                index_srem: resp_len(&["SREM", &keys.path_index(base_path), SESSION_ID_SHAPE]),
                meta_del: resp_len(&["DEL", &keys.session_meta(SESSION_ID_SHAPE)]),
            };
            assert!(
                sizes.names_srem != sizes.index_srem
                    && sizes.names_srem != sizes.meta_del
                    && sizes.index_srem != sizes.meta_del,
                "the three undo commands must differ in size to be told apart: {sizes:?}"
            );
            sizes
        }
    }

    /// Bytes a command occupies on the wire: a RESP array of bulk strings.
    fn resp_len(args: &[&str]) -> u64 {
        let header = format!("*{}\r\n", args.len()).len();
        let body: usize = args
            .iter()
            .map(|arg| format!("${}\r\n", arg.len()).len() + arg.len() + 2)
            .sum();
        (header + body) as u64
    }

    /// Pause between seeing the failing command's bytes and sampling the
    /// baseline the undo's bytes are detected against. See
    /// [`Staged::wait_until_the_undo_arrives`].
    const UNDO_SETTLE: Duration = Duration::from_millis(200);

    /// Wait for every data-pool connection to be connected again after
    /// [`Staged::kill_stalls`].
    async fn await_data_pool_reconnected(tv: &TestValkey) {
        tokio::time::timeout(RECONNECT_BUDGET, tv.client().pool.wait_for_connect())
            .await
            .expect("the data pool did not reconnect within the budget")
            .expect("the data pool reconnect must succeed");
    }

    /// Assert `session_id` is the only holder of `name` at `base_path`, listed
    /// with its full metadata, with no ghost left behind.
    async fn assert_one_complete_session(
        tv: &TestValkey,
        base_path: &str,
        name: &str,
        session_id: &str,
    ) {
        let listed = session_store(tv)
            .list(base_path)
            .await
            .expect("list must succeed once the stall is released");
        assert!(
            listed.ghosts.is_empty(),
            "the successful create wrote the metadata, so nothing may be a ghost; got {:?}",
            listed.ghosts.iter().map(|g| &g.sid).collect::<Vec<_>>()
        );
        assert_eq!(listed.present.len(), 1, "exactly one session: {listed:?}");
        let meta = &listed.present[0];
        assert_eq!(meta.session_id, session_id);
        assert_eq!(meta.name, name);
        assert_eq!(meta.base_path, base_path);
        assert_eq!(meta.created_by, CREATED_BY);
        assert!(meta.created_at > 0, "created_at must be written");
        assert_eq!(
            name_members(tv, base_path, name).await,
            vec![format!("{name}\u{0}{session_id}")],
            "exactly one names member may carry {name:?}, and it is this call's"
        );
    }

    /// One `ListSessions` of `base_path` through the real handler, whose ghost
    /// walk removes members with no metadata (the session-recovery hook is
    /// disabled, so its probe answers `NoCheckpoint`).
    async fn list_sessions_through_the_handler(
        tv: &TestValkey,
        base_path: &str,
    ) -> live_edit::grpc::proto::ListSessionsResponse {
        let svc = build_service(tv, Arc::new(AllowAllPermissions));
        let mut request = Request::new(ListSessionsRequest {
            path: base_path.to_owned(),
        });
        request
            .extensions_mut()
            .insert(BearerToken::new("b62m-bearer"));
        handle_list_sessions(&svc, request)
            .await
            .expect("ListSessions must succeed once the store is healthy")
            .into_inner()
    }

    /// The failed attempt's leftover — both members, no metadata — is a ghost
    /// that the ListSessions handler removes (the session-recovery hook is
    /// disabled, so the probe answers `NoCheckpoint`), after which the name is
    /// free again. Returns the failed attempt's `session_id`, read off the
    /// ghost.
    async fn assert_leftover_is_a_ghost_that_list_sessions_removes(
        tv: &TestValkey,
        base_path: &str,
        name: &str,
    ) -> String {
        let listed = session_store(tv)
            .list(base_path)
            .await
            .expect("list must succeed once the data pool reconnected");
        assert!(
            listed.present.is_empty(),
            "no metadata write ran (killing the stalls discarded everything queued behind \
             them), so nothing is complete: {listed:?}"
        );
        assert_eq!(
            listed.ghosts.len(),
            1,
            "the attempt's names member must be left as exactly one ghost"
        );
        let ghost_sid = listed.ghosts[0].sid.clone();
        assert_eq!(listed.ghosts[0].name, name);
        let in_path_index: bool = tv
            .client()
            .pool
            .sismember(tv.key_schema().path_index(base_path), &ghost_sid)
            .await
            .expect("SISMEMBER on the path index must succeed");
        assert!(
            in_path_index,
            "the undo could not remove the path_index member either"
        );

        let response = list_sessions_through_the_handler(tv, base_path).await;
        assert!(
            response.sessions.is_empty(),
            "a ghost with no checkpoint is removed, not listed: {:?}",
            response.sessions
        );
        assert!(
            name_members(tv, base_path, name).await.is_empty(),
            "ListSessions must have removed the ghost's names member"
        );
        let still_indexed: bool = tv
            .client()
            .pool
            .sismember(tv.key_schema().path_index(base_path), &ghost_sid)
            .await
            .expect("SISMEMBER on the path index must succeed");
        assert!(
            !still_indexed,
            "ListSessions must remove the path_index member too"
        );

        let session_id = session_store(tv)
            .create(name, base_path, CREATED_BY)
            .await
            .expect("once the ghost is gone the name is free, so a same-name create succeeds");
        assert_one_complete_session(tv, base_path, name, &session_id).await;
        ghost_sid
    }

    /// The WARN `undo_create_attempt` logs when its names `SREM` failed, so
    /// the name could not be released. Verbatim from `src/storage/session.rs`:
    /// the line is what an operator reads to learn a CreateSession may have
    /// left a leftover behind, so a reword must break this test.
    const UNDO_NAME_NOT_RELEASED_MSG: &str = "create_session failed after its commit was sent and \
         the undo could not release the name. If the commit landed, the name stays claimed: the \
         next ListSessions of this base path removes the incomplete entry if it has no metadata, \
         but if a timed-out metadata write still reached the store, the session is complete and \
         stays";

    /// The WARN `reconcile_in_doubt_create` logs when its `SISMEMBER` fails,
    /// verbatim from `src/storage/session.rs`.
    const SISMEMBER_FAILED_MSG: &str =
        "create_session EXEC reply lost and the reconciling SISMEMBER also failed; undoing \
         the attempt";

    /// The WARN `undo_create_attempt` logs when its names `SREM` succeeded but
    /// its `path_index` `SREM` or `DEL` failed. Verbatim from
    /// `src/storage/session.rs`.
    const UNDO_PARTIAL_MSG: &str = "create_session failed after its commit was sent; the name \
         was released but part of the undo failed. A transient failure retries under a new \
         session_id; whatever the failed commands leave behind belongs to this abandoned \
         session_id and stays unlisted";

    /// A distinctive fragment of the sibling WARN `undo_create_attempt` logs
    /// when the names `SREM` succeeded but another undo command failed.
    const UNDO_PARTIAL_FRAGMENT: &str = "the name was released but part of the undo failed";

    /// Drive a create whose `failing` round trip times out and whose undo is
    /// allowed to complete (the stall is released as soon as the undo's first
    /// command arrives). The undo releases the name, so the failure is retried
    /// in-process under a new id and the create succeeds with a complete
    /// session — the only one listed, whatever the abandoned attempt's stale
    /// `HSET` leaves under its own id once the stall is released. Exactly one
    /// data-pool timeout — the failing command's — is counted: the undo's
    /// commands are answered once the stall is released.
    async fn scenario_undo_releases_the_name_and_the_retry_succeeds(
        failing: Failing,
        name: &str,
        base_path: &str,
    ) {
        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, failing);

        // Release the stalls once the undo reaches the store, keeping the create
        // polled throughout. If the create returns first — a code path that
        // answers without undoing, as every pre-b.62m one did — the watcher
        // would wait for bytes that never come, so stop waiting then and report
        // the create's own answer, which is the informative failure. Either way
        // every stall is released before anything is asserted.
        let create = crate::grpc::log_capture::capture(tokio::time::timeout(
            SCENARIO_BUDGET,
            staged.store.create(name, base_path, CREATED_BY),
        ));
        tokio::pin!(create);
        let (outcome, captured) = tokio::select! {
            outcome = &mut create => outcome,
            () = staged.wait_until_the_undo_arrives() => {
                let ((), outcome) = tokio::join!(staged.release_remaining(), &mut create);
                outcome
            }
        };
        staged.release_remaining().await;
        // Which in-doubt arm ran: the `SISMEMBER`-failed arm logs this line, the
        // re-scan arm (reached only when the `SISMEMBER` answered) does not. It
        // is what tells the two in-doubt cases apart, since both otherwise end
        // in the same successful retry with one data-pool timeout.
        let sismember_failed = captured.with_message(SISMEMBER_FAILED_MSG).len();
        match failing {
            Failing::MetaWrite => {}
            Failing::InDoubtReconcile => assert_eq!(
                sismember_failed, 1,
                "the reconciling SISMEMBER must be the round trip that failed; full \
                 capture:\n{}",
                captured.text
            ),
            Failing::InDoubtRescan => assert_eq!(
                sismember_failed, 0,
                "the SISMEMBER must have been answered, so the failure is the re-scan's; full \
                 capture:\n{}",
                captured.text
            ),
        }
        let session_id = outcome
            .expect("the create did not settle within the scenario budget")
            .unwrap_or_else(|e| {
                panic!(
                    "a transient `{}` failure whose undo released the name must be retried \
                     in-process into a successful create; got {e:?}",
                    failing.op()
                )
            });
        staged.assert_staged();

        assert_one_complete_session(&tv, base_path, name, &session_id).await;
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "exactly one data-pool timeout: the `{}` itself (the undo's commands were answered \
             after the release)",
            failing.op()
        );
    }

    /// Drive a create whose `failing` round trip times out and whose undo times
    /// out too (the stall is held until the create returns), then kill the
    /// stalls so nothing queued runs. The name could not be released, so the
    /// call must end with `Unavailable`, never retry (why, per caller: see
    /// `undo_create_attempt` / SDD §7.10). A wrong retry shows up here as a
    /// different op label on the error or extra timeouts on the row. The one
    /// not-released WARN must name the failing op and the attempt.
    async fn scenario_undo_fails_and_the_create_answers_unavailable(
        failing: Failing,
        name: &str,
        base_path: &str,
    ) {
        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, failing);

        let ((outcome, captured), ()) = tokio::join!(
            crate::grpc::log_capture::capture(tokio::time::timeout(
                SCENARIO_BUDGET,
                staged.store.create(name, base_path, CREATED_BY),
            )),
            staged.let_the_sismember_through(),
        );
        // End the stalls before asserting anything, so a failing run does not
        // leave the data pool parked until the `BLPOP`s expire (60 s).
        staged.kill_stalls().await;
        await_data_pool_reconnected(&tv).await;
        let outcome = outcome.expect("the create did not settle within the scenario budget");
        staged.assert_staged();
        let err = match outcome {
            Ok(sid) => {
                panic!("the create cannot succeed while the data pool is stalled; got {sid}")
            }
            Err(e) => e,
        };
        assert!(
            !matches!(err, ServiceError::AlreadyExists(_)),
            "a failed undo must end the call with `Unavailable`. `AlreadyExists` cannot come \
             from this attempt's own member (the scans skip ids this call created), so it would \
             mean the call went on past the failed undo; got {err:?}"
        );
        assert_unavailable_timeout(&err, failing.op());
        // Four, all on data: the failing command, then the undo's three
        // concurrent commands (`SREM` names member, `SREM` path_index member,
        // `DEL` session_meta), each queued behind a stall and each counted once
        // by `undo_create_attempt`. The commit's `EXEC` ran on the txn pool,
        // which is never stalled.
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 4),
            "the `{}` and the undo's three commands each time out once on the data pool",
            failing.op()
        );

        let attempt_sid =
            assert_leftover_is_a_ghost_that_list_sessions_removes(&tv, base_path, name).await;

        let warns = captured.with_message(UNDO_NAME_NOT_RELEASED_MSG);
        assert_eq!(
            warns.len(),
            1,
            "a failed undo must log exactly one `could not release the name` line; got {} — full \
             capture:\n{}",
            warns.len(),
            captured.text
        );
        let warn = warns[0];
        assert_eq!(warn["level"], "WARN", "wrong level; full event: {warn}");
        assert_eq!(
            warn["fields"]["op"],
            failing.op(),
            "the WARN must name the round trip that failed; full event: {warn}"
        );
        assert_eq!(
            warn["fields"]["session_id"],
            attempt_sid.as_str(),
            "the WARN must name the attempt whose leftover stays behind; full event: {warn}"
        );
        assert!(
            !captured.text.contains(UNDO_PARTIAL_FRAGMENT),
            "the names SREM failed, so the partial-undo line must not fire; full capture:\n{}",
            captured.text
        );
    }

    /// A timed-out `HSET session_meta` after a committed create undoes the
    /// attempt and retries it, so `CreateSession` succeeds.
    ///
    /// Pre-fix: `Err(Internal("HSET failed: …"))`, leaving a names member with
    /// no metadata behind.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_timed_out_meta_write_undoes_the_attempt_and_retries_to_success() {
        scenario_undo_releases_the_name_and_the_retry_succeeds(
            Failing::MetaWrite,
            "b62m-hset-retried",
            "/b62m-hset-retried",
        )
        .await;
    }

    /// A timed-out `HSET session_meta` whose undo also times out answers
    /// `Unavailable`, and the leftover heals through ListSessions.
    ///
    /// Pre-fix: `Err(Internal("HSET failed: …"))`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_timed_out_meta_write_with_a_failed_undo_answers_unavailable() {
        scenario_undo_fails_and_the_create_answers_unavailable(
            Failing::MetaWrite,
            "b62m-hset-stuck",
            "/b62m-hset-stuck",
        )
        .await;
    }

    /// A commit that landed but whose reply was lost, followed by a timed-out
    /// reconciling `SISMEMBER`: the undo removes the landed members and the
    /// retry creates the session.
    ///
    /// Pre-fix: the `SISMEMBER` failure returned a bare `Aborted`, the retry's
    /// duplicate scan found the attempt's own landed member, and the caller got
    /// `Err(AlreadyExists)` for the session it was creating.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_timed_out_reconcile_read_undoes_the_landed_commit_and_retries_to_success(
    ) {
        scenario_undo_releases_the_name_and_the_retry_succeeds(
            Failing::InDoubtReconcile,
            "b62m-sismember-retried",
            "/b62m-sismember-retried",
        )
        .await;
    }

    /// The same in-doubt `SISMEMBER` timeout with an undo that also times out
    /// answers `Unavailable`, and the landed members heal through ListSessions.
    ///
    /// Pre-fix: bare `Aborted` → `Err(AlreadyExists)` after one timeout (the
    /// retry's duplicate scan runs on the unstalled txn pool and finds the
    /// landed member).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_timed_out_reconcile_read_with_a_failed_undo_answers_unavailable() {
        scenario_undo_fails_and_the_create_answers_unavailable(
            Failing::InDoubtReconcile,
            "b62m-sismember-stuck",
            "/b62m-sismember-stuck",
        )
        .await;
    }

    /// A commit that landed but whose reply was lost, a `SISMEMBER` that
    /// confirms it landed, and a re-scan `SMEMBERS` that times out: the undo
    /// removes the landed members and the retry creates the session.
    ///
    /// Pre-fix: `Err(Internal("SMEMBERS names failed: …"))`, deliberately
    /// non-retriable because a bare `Aborted` would retry into the landed
    /// member; the landed members stayed behind with no metadata.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_timed_out_rescan_undoes_the_landed_commit_and_retries_to_success() {
        scenario_undo_releases_the_name_and_the_retry_succeeds(
            Failing::InDoubtRescan,
            "b62m-rescan-retried",
            "/b62m-rescan-retried",
        )
        .await;
    }

    /// The same timed-out re-scan with an undo that also times out answers
    /// `Unavailable`, and the landed members heal through ListSessions.
    ///
    /// Pre-fix: `Err(Internal("SMEMBERS names failed: …"))`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_timed_out_rescan_with_a_failed_undo_answers_unavailable() {
        scenario_undo_fails_and_the_create_answers_unavailable(
            Failing::InDoubtRescan,
            "b62m-rescan-stuck",
            "/b62m-rescan-stuck",
        )
        .await;
    }

    /// Poll until `session_meta(session_id)` exists (`want`) or not. Both
    /// directions are one-way in the scenarios that use it, so the first
    /// matching read is final.
    async fn await_session_meta_exists(tv: &TestValkey, session_id: &str, want: bool, why: &str) {
        let key = tv.key_schema().session_meta(session_id);
        let deadline = Instant::now() + SCENARIO_BUDGET;
        loop {
            let found: i64 = tv
                .client()
                .pool
                .exists(&key)
                .await
                .expect("EXISTS on the data pool must succeed");
            if (found == 1) == want {
                return;
            }
            assert!(Instant::now() < deadline, "{why}");
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Poll until `path_index(base_path)` holds `session_id` (`want`) or not;
    /// one-way, as for [`await_session_meta_exists`].
    async fn await_path_index_holds(
        tv: &TestValkey,
        base_path: &str,
        session_id: &str,
        want: bool,
        why: &str,
    ) {
        let key = tv.key_schema().path_index(base_path);
        let deadline = Instant::now() + SCENARIO_BUDGET;
        loop {
            let holds: bool = tv
                .client()
                .pool
                .sismember(&key, session_id)
                .await
                .expect("SISMEMBER on the data pool must succeed");
            if holds == want {
                return;
            }
            assert!(Instant::now() < deadline, "{why}");
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// A timed-out metadata `HSET` whose undo releases the name but fails its
    /// other two commands: the verdict turns on the names `SREM` alone, so the
    /// call retries under a new `session_id` and succeeds, with one WARN
    /// reporting the partial undo against the abandoned id.
    ///
    /// Only the names `SREM` is let through (see
    /// [`Staged::let_only_the_names_srem_through`]); the `path_index` `SREM`
    /// and `DEL` time out. Mid-flight the store holds what the partial undo
    /// left under the abandoned id: its `path_index` member, and no names
    /// member or metadata. The retry then commits a new id.
    ///
    /// The two timed-out undo commands are then killed, so they never run,
    /// while the abandoned attempt's original `HSET` is released, so it does.
    /// That pins the inert leftovers a partial undo can leave for good — the
    /// abandoned id's `path_index` member and an orphan `session_meta` — and
    /// that neither is listed. The same undo commands landing late instead is
    /// [`create_session_partial_undo_commands_landing_after_ok_leave_the_new_session_intact`].
    ///
    /// Pre-fix: `Err(Internal("HSET failed: …"))`, with no undo at all. With
    /// the undo but a retry reusing the attempt's id, the returned id was the
    /// abandoned one and its leftovers were the new session's own entries.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_timed_out_meta_write_with_a_partial_undo_releases_the_name_and_retries_to_success(
    ) {
        const NAME: &str = "b62m-partial-undo";
        const BASE_PATH: &str = "/b62m-partial-undo";

        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, Failing::MetaWrite);
        let sizes = UndoSizes::new(&tv, NAME, BASE_PATH);

        // As in the other success cases: if the create returns before the
        // choreography completes (a path that answers without undoing), stop
        // and report its answer. Every stall is ended before any assertion,
        // the undo's two timed-out commands by a kill.
        let create = crate::grpc::log_capture::capture(tokio::time::timeout(
            SCENARIO_BUDGET,
            staged.store.create(NAME, BASE_PATH, CREATED_BY),
        ));
        tokio::pin!(create);
        let mut undo_conn_ids: Vec<String> = Vec::new();
        let mut left_behind: Option<String> = None;
        let (outcome, captured) = tokio::select! {
            outcome = &mut create => outcome,
            partial = staged.let_only_the_names_srem_through(&tv, NAME, BASE_PATH, &sizes) => {
                undo_conn_ids = partial.undo_conn_ids;
                left_behind = Some(partial.session_id);
                tokio::select! {
                    outcome = &mut create => outcome,
                    _ = staged.wait_until_the_retry_arrives() => {
                        let ((), outcome) =
                            tokio::join!(staged.kill_then_release(&undo_conn_ids), &mut create);
                        outcome
                    }
                }
            }
        };
        staged.kill_then_release(&undo_conn_ids).await;
        await_data_pool_reconnected(&tv).await;

        let session_id = outcome
            .expect("the create did not settle within the scenario budget")
            .unwrap_or_else(|e| {
                panic!(
                    "an undo that released the name must retry a transient HSET failure into a \
                     successful create, even though its other two commands failed; got {e:?}"
                )
            });
        staged.assert_staged();
        let left_behind = left_behind.expect("the partial undo was staged");
        assert_ne!(
            left_behind, session_id,
            "the retry must run under a new session_id, not the abandoned attempt's, whose \
             entries the partial undo could not clear"
        );

        // The leftovers under the abandoned id: the killed `SREM` never ran, and
        // the released original `HSET` did while the killed `DEL` did not.
        await_path_index_holds(
            &tv,
            BASE_PATH,
            &left_behind,
            true,
            "the abandoned id's path_index member must survive: its SREM was killed unrun",
        )
        .await;
        await_session_meta_exists(
            &tv,
            &left_behind,
            true,
            "the abandoned id's original HSET was released, and its DEL killed, so its \
             session_meta must exist as an orphan",
        )
        .await;
        // ...and neither is listed: the only listed session is the returned one.
        assert_one_complete_session(&tv, BASE_PATH, NAME, &session_id).await;
        await_path_index_holds(
            &tv,
            BASE_PATH,
            &session_id,
            true,
            "the retry's commit must index the returned session id",
        )
        .await;

        // Three, all on data: the HSET, then the undo's path_index SREM and
        // DEL. The names SREM was answered, and the retry's HSET was released
        // well inside its deadline.
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 3),
            "the HSET and the undo's two failed commands each time out once on the data pool"
        );

        let warns = captured.with_message(UNDO_PARTIAL_MSG);
        assert_eq!(
            warns.len(),
            1,
            "a partial undo must log exactly one partial-undo line; got {} — full capture:\n{}",
            warns.len(),
            captured.text
        );
        let warn = warns[0];
        assert_eq!(warn["level"], "WARN", "wrong level; full event: {warn}");
        assert_eq!(warn["fields"]["op"], "HSET session meta (create)");
        assert_eq!(
            warn["fields"]["session_id"],
            left_behind.as_str(),
            "the WARN names the abandoned attempt whose leftovers it reports; full event: {warn}"
        );
        for field in ["path_index_error", "session_meta_error"] {
            let error = warn["fields"][field].as_str().unwrap_or_default();
            assert!(
                error.contains("Timeout"),
                "the WARN must carry the failed undo command's error in `{field}`; full event: \
                 {warn}"
            );
        }
        assert!(
            captured.with_message(UNDO_NAME_NOT_RELEASED_MSG).is_empty(),
            "the names SREM succeeded, so the `could not release the name` line must not fire; \
             full capture:\n{}",
            captured.text
        );
    }

    /// The partial undo's timed-out `path_index` `SREM` and `DEL` reach the
    /// store only after the call has returned `Ok` — and the session it
    /// returned is untouched, because those commands target the abandoned
    /// attempt's id, which no later command of the call uses.
    ///
    /// Staged as [`create_session_timed_out_meta_write_with_a_partial_undo_releases_the_name_and_retries_to_success`]
    /// up to the names `SREM`. Then the stalls holding the two undo commands
    /// are left parked while everything else is released, so the retry's
    /// metadata `HSET` completes and `create` returns; only then are the undo
    /// commands released — not killed — and allowed to run.
    ///
    /// The retry's `HSET` must not be queued behind one of those undo
    /// commands, or releasing it would run the undo command before `Ok` and the
    /// run would not have staged the late landing at all: that fails loudly
    /// rather than passing. With the fixture's four connections fred's
    /// round-robin sends it to the connection already holding the abandoned
    /// attempt's original `HSET`, behind that `HSET`; if it went to the
    /// released names connection instead, `create` would finish while that
    /// `HSET` was still parked. Either way the test releases the original
    /// `HSET` and waits until it has landed before releasing the undo
    /// commands, so the stale `DEL` always runs after it here and no orphan
    /// `session_meta` is left in this ordering; the killed variant pins that
    /// leftover.
    ///
    /// With the undo but a retry reusing the attempt's id, the late `SREM`
    /// removed the new session's index entry and the late `DEL` its metadata:
    /// `create` returned `Ok(sid)`, and `list` then showed no session and one
    /// ghost carrying `sid`, which the next ListSessions deletes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_partial_undo_commands_landing_after_ok_leave_the_new_session_intact() {
        const NAME: &str = "b62m-late-undo";
        const BASE_PATH: &str = "/b62m-late-undo";

        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, Failing::MetaWrite);
        let sizes = UndoSizes::new(&tv, NAME, BASE_PATH);

        let create = tokio::time::timeout(
            SCENARIO_BUDGET,
            staged.store.create(NAME, BASE_PATH, CREATED_BY),
        );
        tokio::pin!(create);
        let partial = tokio::select! {
            outcome = &mut create => {
                staged.kill_stalls().await;
                panic!(
                    "the create returned before its undo reached the store, so nothing was \
                     staged; got {outcome:?}"
                );
            }
            partial = staged.let_only_the_names_srem_through(&tv, NAME, BASE_PATH, &sizes) => partial,
        };
        let abandoned_id = partial.session_id;
        let undo_conn_ids = partial.undo_conn_ids;

        // Finish the create while the two undo commands stay parked. If the
        // retry's HSET reaches the released names connection, the create
        // finishes on its own; otherwise it reaches a stalled one, which must
        // not be one holding an undo command.
        let outcome = tokio::select! {
            outcome = &mut create => outcome,
            retry_conn = staged.wait_until_the_retry_arrives() => {
                if undo_conn_ids.contains(&retry_conn) {
                    staged.kill_stalls().await;
                    panic!(
                        "NOT STAGED: the retry's metadata HSET was queued on connection \
                         {retry_conn}, behind one of the abandoned attempt's undo commands, so \
                         that command would run before `create` returned. This test needs the \
                         undo commands to land after `Ok`."
                    );
                }
                let ((), outcome) =
                    tokio::join!(staged.release_all_but(&undo_conn_ids), &mut create);
                outcome
            }
        };
        let session_id = match outcome {
            Ok(Ok(sid)) => sid,
            other => {
                staged.kill_stalls().await;
                panic!(
                    "the retry after a partial undo must succeed before the stale undo commands \
                     land; got {other:?}"
                );
            }
        };
        staged.assert_staged();

        // Pin the order of the abandoned id's own late writes before the undo
        // commands run. If the create finished without its HSET touching a
        // stall, the abandoned attempt's original HSET is still parked; release
        // it (a no-op when the retry's HSET already did) and wait until it has
        // landed, so the stale DEL below necessarily follows it.
        staged.release_all_but(&undo_conn_ids).await;
        await_session_meta_exists(
            &tv,
            &abandoned_id,
            true,
            "the abandoned attempt's original HSET, once released, must write its session_meta",
        )
        .await;

        // Now the abandoned attempt's `SREM path_index` and `DEL session_meta`
        // land, after `Ok`.
        staged.release_remaining().await;
        await_path_index_holds(
            &tv,
            BASE_PATH,
            &abandoned_id,
            false,
            "the released stale SREM must remove the abandoned id from path_index",
        )
        .await;
        await_session_meta_exists(
            &tv,
            &abandoned_id,
            false,
            "the released stale DEL must remove the abandoned id's session_meta",
        )
        .await;

        assert_one_complete_session(&tv, BASE_PATH, NAME, &session_id).await;
        assert_ne!(
            abandoned_id, session_id,
            "the retry must run under a new session_id, so the abandoned attempt's late \
             commands cannot reach the returned session"
        );
        await_path_index_holds(
            &tv,
            BASE_PATH,
            &session_id,
            true,
            "the late undo commands must leave the returned session indexed",
        )
        .await;
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 3),
            "the HSET and the undo's two failed commands each time out once on the data pool; \
             the retry's HSET and the late undo commands add none"
        );
    }

    /// An in-doubt `EXEC` that lands late — after its `SISMEMBER` read
    /// "absent" and the call moved on — does not make the call answer
    /// `AlreadyExists`: the member it writes carries the abandoned attempt's
    /// id, which the call's scans skip, and it is left as a ghost that
    /// ListSessions removes.
    ///
    /// A real late `EXEC`, not a simulated one. Attempt 1's pre-dispatch hook
    /// parks the connection carrying its own `WATCH`, so its commit pipeline
    /// queues behind the stall and its reply times out in doubt; the
    /// reconciling `SISMEMBER` runs on the (unstalled) data pool and reads
    /// "absent". Attempt 1's `WatchGuard` then holds that txn slot, and its
    /// permit, until its own `UNWATCH` — queued behind the same stall —
    /// completes, so attempt 2 checks out a different slot and its checkout
    /// probe never touches the stalled connection. Attempt 2's watch-window
    /// hook releases the stall: the parked `EXEC` runs (its `WATCH` is still
    /// intact, so it commits) and the hook waits until its member is visible.
    /// Attempt 2's scan meets that member; its `EXEC` then hits a `WATCH`
    /// conflict (the late commit touched the watched set after attempt 2's
    /// `WATCH`), and attempt 3 commits.
    ///
    /// With a retry reusing the attempt's id, attempt 2's scan found the late
    /// member under its own id and the call answered `Err(AlreadyExists)`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_exec_landing_late_does_not_make_the_call_answer_already_exists() {
        const NAME: &str = "b62m-late-exec";
        const BASE_PATH: &str = "/b62m-late-exec";

        let (tv, metrics) = fixture().await;
        let observer = tv.client().maintenance_pool.next().clone();
        let names_key = tv.key_schema().path_names_index(BASE_PATH);
        let parked_exec: Arc<tokio::sync::Mutex<Option<ConnStall>>> = Arc::default();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let watch_windows = Arc::new(AtomicUsize::new(0));
        let late_member: Arc<Mutex<Option<String>>> = Arc::default();

        // Attempt 1 only: park the connection carrying this attempt's WATCH.
        let pre_dispatch: WatchWindowHook = {
            let (parked_exec, dispatches) = (parked_exec.clone(), dispatches.clone());
            let observer = observer.clone();
            let key = format!("{}b62m-txn-stall", tv.prefix());
            Arc::new(move |guarded: &Client| {
                let (parked_exec, dispatches) = (parked_exec.clone(), dispatches.clone());
                let (observer, key) = (observer.clone(), key.clone());
                async move {
                    if dispatches.fetch_add(1, Ordering::SeqCst) != 0 {
                        return;
                    }
                    let stall = ConnStall::park(guarded, &observer, key).await;
                    *parked_exec.lock().await = Some(stall);
                }
                .boxed()
            })
        };
        // Attempt 2 only: let attempt 1's parked commit run, late, and wait for
        // its member to become visible before this attempt's scan.
        let watch_window: WatchWindowHook = {
            let (parked_exec, watch_windows) = (parked_exec.clone(), watch_windows.clone());
            let (observer, names_key, late_member) =
                (observer.clone(), names_key.clone(), late_member.clone());
            Arc::new(move |_guarded: &Client| {
                let (parked_exec, watch_windows) = (parked_exec.clone(), watch_windows.clone());
                let (observer, names_key, late_member) =
                    (observer.clone(), names_key.clone(), late_member.clone());
                async move {
                    if watch_windows.fetch_add(1, Ordering::SeqCst) != 1 {
                        return;
                    }
                    let stall = parked_exec
                        .lock()
                        .await
                        .take()
                        .expect("attempt 1 must have parked its txn connection");
                    let id = stall.conn_id().to_owned();
                    let reply = stall.release().await;
                    assert!(
                        matches!(&reply, Ok(v) if !v.is_null()),
                        "the stall on txn connection {id} must have held attempt 1's commit \
                         until now; got {reply:?}"
                    );
                    let deadline = Instant::now() + SCENARIO_BUDGET;
                    loop {
                        let members: Vec<String> = observer
                            .smembers(&names_key)
                            .await
                            .expect("SMEMBERS on the observer must succeed");
                        if let Some(member) = members
                            .into_iter()
                            .find(|m| m.split('\u{0}').next() == Some(NAME))
                        {
                            *late_member.lock().unwrap() = Some(member);
                            return;
                        }
                        assert!(
                            Instant::now() < deadline,
                            "attempt 1's released commit never wrote its names member"
                        );
                        tokio::time::sleep(POLL_INTERVAL).await;
                    }
                }
                .boxed()
            })
        };
        let store = session_store(&tv)
            .with_pre_dispatch_hook_for_test(pre_dispatch)
            .with_watch_window_hook_for_test(watch_window);

        let outcome =
            tokio::time::timeout(SCENARIO_BUDGET, store.create(NAME, BASE_PATH, CREATED_BY)).await;
        // Never leave the txn connection parked past the test, whatever happened.
        if let Some(stall) = parked_exec.lock().await.take() {
            let _ = stall.kill().await;
        }
        let session_id = outcome
            .expect("the create did not settle within the scenario budget")
            .unwrap_or_else(|e| {
                panic!(
                    "a late EXEC of an abandoned attempt writes a member under that attempt's \
                     id, which the call's scans skip, so the call must still succeed; got {e:?}"
                )
            });

        let late_member = late_member.lock().unwrap().clone().expect(
            "attempt 2's hook must have seen the late member — the late EXEC was not staged",
        );
        let abandoned_id = late_member
            .split('\u{0}')
            .nth(1)
            .expect("a names member carries NAME\\0session_id")
            .to_owned();
        assert_ne!(
            abandoned_id, session_id,
            "the late member carries the abandoned attempt's id, not the returned one"
        );
        assert_eq!(
            watch_windows.load(Ordering::SeqCst),
            3,
            "three attempts: the in-doubt one, the one whose WATCH the late commit voided, and \
             the one that committed"
        );

        // The late member is a ghost of the abandoned id beside the new session...
        let listed = session_store(&tv)
            .list(BASE_PATH)
            .await
            .expect("list must succeed");
        assert_eq!(
            listed
                .present
                .iter()
                .map(|s| s.session_id.as_str())
                .collect::<Vec<_>>(),
            vec![session_id.as_str()],
            "only the returned session is complete"
        );
        assert_eq!(
            listed
                .ghosts
                .iter()
                .map(|g| g.sid.as_str())
                .collect::<Vec<_>>(),
            vec![abandoned_id.as_str()],
            "the late EXEC leaves exactly one ghost, under the abandoned id"
        );
        // ...which ListSessions removes, leaving exactly the returned session.
        let response = list_sessions_through_the_handler(&tv, BASE_PATH).await;
        assert_eq!(
            response
                .sessions
                .iter()
                .map(|s| s.session_id.as_str())
                .collect::<Vec<_>>(),
            vec![session_id.as_str()],
            "ListSessions lists the returned session and not the ghost"
        );
        assert_one_complete_session(&tv, BASE_PATH, NAME, &session_id).await;
        await_path_index_holds(
            &tv,
            BASE_PATH,
            &abandoned_id,
            false,
            "ListSessions must remove the ghost's path_index member too",
        )
        .await;

        // One, on txn: attempt 1's in-doubt EXEC. Its UNWATCH and the late
        // commit were released well inside their deadlines.
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("txn", 1),
            "only attempt 1's EXEC times out"
        );
    }
}
