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
//! CreateSession's metadata write (`EVALSHA session_meta_write`) is another —
//! `create_once`'s `WATCH`, duplicate-scan `SMEMBERS` and `EXEC` all run on the
//! txn checkout, never the data pool — and so, on the in-doubt path, is the
//! reconciling `SISMEMBER`; the `undo` module drives both. It also drives the
//! in-doubt re-scan `SMEMBERS`, which does sit behind a data-pool command (that
//! `SISMEMBER`), by letting the `SISMEMBER` through a single connection (see
//! *Staging*). The remaining converted sites — `delete`'s two `SREM`s,
//! `list`'s per-session `HGETALL`, `create_layer`'s mapping `GET` / `SET`, and
//! the overwrite's `HSET obj:hdr superseded` (moved off the in-loop mapper) —
//! each sit behind an earlier data-pool command and are guarded per site by
//! `mapper_routing_test.rs` alone. The CreateLayer overwrite's old-stream
//! `XADD` is driven through its fred-error seam in
//! `update_layer_admission_test.rs`, and the ListSessions handler's log line in
//! `grpc/rpc_failure_log_test.rs`.
//!
//! # CreateSession's post-commit failures (b.62m, reshaped by b.2tu)
//!
//! The CreateSession retry rules live in `SessionStore::settle_failed_meta_write`'s
//! rustdoc (SDD §7.10). In short:
//! - **A failed metadata write is settled** with one `session_meta_veto.lua`
//!   run on the attempt's id. `PRESENT` (the write landed, its reply was lost)
//!   answers OK with that id, and so does `DELETED` (the write landed and
//!   another call deleted the session since; no veto, no checkpoint
//!   enqueued); `VETOED` (no write for the id can land from now
//!   on) releases the attempt's members, best effort, and a transient cause
//!   retries under a new id; a failed settle answers `UNAVAILABLE`, never
//!   `Aborted`, and releases nothing.
//! - **A failed in-doubt read** (the reconciling `SISMEMBER`, or the re-scan
//!   `SMEMBERS` after it) releases the attempt's members through one
//!   `session_index_release.lua` run; a released attempt retries, a failed
//!   release answers `UNAVAILABLE` and leaves both members or neither — never
//!   a `path_index` member alone.
//!
//! The `undo` module drives each arm:
//!
//! | Test | Staging | Verdict |
//! |---|---|---|
//! | [`undo::create_session_settle_vetoing_before_the_late_metadata_write_retries_under_a_new_id`] | write times out; the veto runs, then the late write | `VETOED`: the late write is refused, retry → OK(new id) |
//! | [`undo::create_session_metadata_write_landing_before_its_settle_answers_ok_with_its_own_id`] | write times out; the late write runs, then the veto | `PRESENT`: OK with the attempt's id, no retry |
//! | [`undo::create_session_metadata_write_landing_then_deleted_before_its_settle_answers_ok_with_its_own_id`] | write times out; the late write runs, a `DeleteSession` on another client deletes the session, then the veto | `DELETED`: OK with the attempt's id, no retry, no veto, no checkpoint enqueued |
//! | [`undo::create_session_timed_out_meta_write_with_a_failed_settle_answers_unavailable`] | write and veto both time out, killed | `UNAVAILABLE`, nothing released |
//! | [`undo::create_session_vetoed_attempt_whose_release_fails_still_retries_to_success`] | the veto runs, the release times out | `VETOED`: retry → OK, a ghost left for ListSessions |
//! | [`undo::create_session_vetoed_attempts_release_landing_after_ok_leaves_the_new_session_intact`] | the same, the release runs after OK | the late release touches only the abandoned id |
//! | [`undo::create_in_doubt_timed_out_reconcile_read_releases_the_landed_commit_and_retries_to_success`] | in-doubt `SISMEMBER` times out, the release runs | retry → OK |
//! | [`undo::create_in_doubt_timed_out_reconcile_read_with_a_failed_release_answers_unavailable`] | in-doubt `SISMEMBER` and the release time out, killed | `UNAVAILABLE`, both members left |
//! | [`undo::create_in_doubt_timed_out_rescan_releases_the_landed_commit_and_retries_to_success`] | in-doubt re-scan times out, the release runs | retry → OK |
//! | [`undo::create_in_doubt_timed_out_rescan_with_a_failed_release_answers_unavailable`] | in-doubt re-scan and the release time out, killed | `UNAVAILABLE`, both members left |
//! | [`undo::create_in_doubt_exec_landing_late_does_not_make_the_call_answer_already_exists`] | an in-doubt `EXEC` landing after its `SISMEMBER` read "absent" | retry → OK, a ghost left for ListSessions |
//!
//! Pre-b.2tu the metadata write was a plain `HSET` undone by three concurrent
//! commands: a late `HSET` after a released name left orphan metadata, an
//! `HSET` that landed with its reply lost was undone and retried under a new
//! id, and a partial undo could leave a `path_index` member without its names
//! member. Breaking the new scripts back toward that behaviour (a veto that
//! never answers `PRESENT`, a metadata write that ignores the veto) turned
//! I7, I8, P1 and the late-release test red; the other rows were not run
//! against such a perturbation.
//!
//! The shape of the call sites (no direct mapper call, one release or settle
//! per round trip, each with its op) is pinned separately in
//! `mapper_routing_test.rs`.
//!
//! # Single-site tests
//!
//! | Test | Site | Pre-fix answer |
//! |---|---|---|
//! | [`session_delete_timed_out_meta_read_answers_unavailable_and_changes_nothing`] | `delete` step 1 `HGETALL` | `Internal` |
//! | [`session_list_timed_out_names_read_answers_unavailable`] | `list` `SMEMBERS` | `Internal` |
//! | [`session_get_meta_timed_out_read_answers_unavailable`] | `get_session_meta` `HGETALL` | `Internal` |
//! | [`ext_load_values_timed_out_read_answers_unavailable`] | `load_values` `GET ext_data` | `Internal` |
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
//! exactly the one under test. The in-doubt cases additionally replace the
//! commit's verdict with `ReplaceOutcomeAfterExec` (the commit lands, its
//! reply is reported lost), which is what routes the attempt into
//! `reconcile_in_doubt_create`'s `SISMEMBER`. The re-scan cases then let that
//! `SISMEMBER` through on its own: fred's data `Pool` hands each command to
//! the next connection in turn, so releasing only the connection the
//! `SISMEMBER`'s bytes reached answers it ("present"), while the re-scan
//! `SMEMBERS` right behind it lands on another, still-stalled connection.
//! Each parked connection's byte count is sampled when it is parked, so "new
//! bytes" never races a command already on its way.
//!
//! The metadata-write cases follow the attempt one command at a time
//! ([`undo::Staged::next_arrival`]): the write, then (5 s later, once it timed
//! out) the settle's veto, then — after a `VETOED` — the release. Each is sent
//! only once the one before it has been answered or has timed out, so the
//! connection whose bytes moved is that command's, found from its bytes, not
//! predicted. Releasing one connection runs exactly the command parked on it,
//! which is how each test orders the late write against the veto. A
//! connection released after its command timed out is then sent one more
//! command (`EXISTS`), whose reply proves the late command has run.
//!
//! A stall ends one of two ways. `release` lets every queued command run.
//! `kill` closes the connection, so everything still queued behind the
//! `BLPOP` dies unrun; fred does not replay a command that already timed out
//! (`router/mod.rs::retry_buffer` skips `timed_out` commands). The failed
//! cases kill every stall and so end with a deterministic keyspace — the
//! attempt's two members and no metadata.
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
//! A late command — a metadata write or a release whose caller has already
//! given up — must run its real script, not fail `NOSCRIPT`: fred reloads a
//! script only for a caller still waiting. Otherwise "the late write wrote
//! nothing" would hold vacuously, and a late write or release the test
//! expects to land would not. Other tests in the suite run a real
//! `SCRIPT FLUSH`, so [`undo::Staged::new`]'s up-front load is not enough:
//! the session scripts are loaded again on the observer immediately before
//! each such command is released ([`undo::Staged::reload_scripts`]). The
//! window that remains is from that load to the command running, which
//! follows at once: the release's `LPUSH` ends the `BLPOP` and the server runs
//! the next queued command on that connection. A flush landing exactly there
//! still turns the command into `NOSCRIPT`, so each late command is also
//! proven to have run its script ([`undo::assert_late_script_ran`]): a fence
//! is sent on the command's own connection, and the store's `MONITOR` feed
//! must show, before it, a command the script itself issued on this test's
//! key — the metadata write's `EXISTS` of the veto key, the release's `TYPE`
//! of `path_index`. A flush in the window then fails the run with a message
//! naming it (a red run on correct code, to be rerun), never a vacuous pass of
//! I7's and P1's "the late write wrote nothing".
//!
//! # Feature gate
//!
//! The four single-site tests need only `test-support`. The `undo` module uses
//! `with_pre_dispatch_hook_for_test` / `with_watched_exec_override_for_test` /
//! `with_post_commit_hook_for_test`, which exist only under
//! `test-fault-injection`, so without that feature the module is compiled out
//! and reports nothing — pass `--features test-support,test-fault-injection`
//! on any hand-rolled run.
//!
//! # Wait budgets vs. nextest's hang-killer
//!
//! The operation under test runs under ONE deadline, [`SCENARIO_BUDGET`]; the
//! staging waits run under the same one. Parking and ending a stall are
//! bounded separately by `test_support::CONN_STALL_WAIT` per connection, and
//! the post-kill reconnect by `undo::RECONNECT_BUDGET`.
//!
//! | Test | Healthy path | Operation deadline |
//! |---|---|---|
//! | the four single-site tests | one command timeout, ≈ 5 s | 30 s |
//! | the two in-doubt "retries to success" tests, the `VETOED` / `PRESENT` / `DELETED` settle tests | one command timeout, ≈ 5 s | 30 s |
//! | the three "fails → `UNAVAILABLE`" tests, the two failed- and late-release tests | two command timeouts in sequence, ≈ 10 s | 30 s |
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
// CreateSession: settle or release the attempt after a post-commit failure
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

    use live_edit::storage::lua::{
        SESSION_INDEX_RELEASE_SCRIPT, SESSION_META_VETO_SCRIPT, SESSION_META_WRITE_SCRIPT,
    };
    use live_edit::storage::session::PostCommitHook;
    use live_edit::storage::test_support::ConnStall;
    use live_edit::storage::watched_txn::{
        WatchWindowHook, WatchedExecOverride, WatchedExecOverrideQueue,
    };

    use super::*;
    use crate::grpc::helpers::list_sessions_through_the_handler;
    use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};

    /// How long fred may take to re-establish the data-pool connections a test
    /// killed. A healthy reconnect takes a few hundred milliseconds (100 ms
    /// exponential floor).
    const RECONNECT_BUDGET: Duration = Duration::from_secs(10);

    /// Poll interval for the staging waits.
    const POLL_INTERVAL: Duration = Duration::from_millis(5);

    /// Pause between seeing a command's first bytes on a stalled connection and
    /// re-baselining every stalled connection, so a frame split across reads is
    /// counted whole. Slack, not a race: in every staging here the next command
    /// is sent only after this one is answered or has timed out (5 s).
    const FRAME_SETTLE: Duration = Duration::from_millis(200);

    /// The op label every metadata-write failure surfaces under.
    const META_WRITE_OP: &str = "EVALSHA session_meta_write (create)";

    /// The WARN `settle_failed_meta_write` logs when the settle itself failed,
    /// so the call stops without retrying. Verbatim from
    /// `src/storage/session.rs`: an operator reads it to learn a CreateSession
    /// may have completed under an id the client was not told.
    const SETTLE_FAILED_MSG: &str = "create_session metadata write failed and the settle could \
         not tell whether it landed; not retrying. If the write still lands, the session is \
         complete and listed under an id the client was not told; if not, the next ListSessions \
         of this base path removes the incomplete entry";

    /// The WARN `settle_failed_meta_write` logs when the settle found the
    /// metadata (`PRESENT`). Verbatim from `src/storage/session.rs`.
    const SETTLE_FOUND_THE_WRITE_MSG: &str = "create_session metadata write failed, but the \
         settle found the metadata: the reply was lost but the write landed; completing the create";

    /// The WARN `release_abandoned_attempt` logs when the release of a vetoed
    /// attempt failed. Verbatim from `src/storage/session.rs`.
    const VETOED_RELEASE_FAILED_MSG: &str = "create_session could not release the index members \
         of a vetoed attempt; the attempt can never complete, and the next ListSessions of this \
         base path removes its members";

    /// The WARN `release_in_doubt_attempt` logs when its release failed.
    /// Verbatim from `src/storage/session.rs`.
    const IN_DOUBT_RELEASE_FAILED_MSG: &str = "create_session lost its commit's reply, could not \
         finish reconciling it, and could not release the attempt's index members; not \
         retrying. If the commit landed, the name stays claimed until the next ListSessions of \
         this base path removes the incomplete entry";

    /// The WARN `reconcile_in_doubt_create` logs when its `SISMEMBER` fails,
    /// verbatim from `src/storage/session.rs`.
    const SISMEMBER_FAILED_MSG: &str =
        "create_session EXEC reply lost and the reconciling SISMEMBER also failed; releasing \
         the attempt's index members";

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
        /// The commit lands normally; `finish_create`'s metadata write
        /// (`EVALSHA session_meta_write`) times out.
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
                Failing::MetaWrite => META_WRITE_OP,
                Failing::InDoubtReconcile => "SISMEMBER path names index (in-doubt reconcile)",
                Failing::InDoubtRescan => "SMEMBERS path names index (in-doubt re-scan)",
            }
        }
    }

    /// One stalled data-pool connection: the pool client it sits on, its
    /// stall, and the bytes the server had read from it at its last baseline.
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

    /// Index of the first parked connection that has read bytes since its
    /// baseline, if any.
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
        /// Every attempt's `session_id` that reached `finish_create`, in
        /// order, recorded by a post-commit hook.
        attempts: Arc<Mutex<Vec<String>>>,
        observer: Client,
        key_prefix: String,
    }

    impl Staged {
        async fn new(tv: &TestValkey, failing: Failing) -> Self {
            if matches!(failing, Failing::InDoubtRescan) {
                assert!(
                    tv.client().pool.clients().len() >= 2,
                    "the re-scan case needs the SMEMBERS to land on a different data-pool \
                     connection than the SISMEMBER it follows"
                );
            }
            // Load the session scripts up front, so a late-landing command runs
            // its script rather than failing `NOSCRIPT` (see *Staging*).
            let loader = tv.client().pool.next().clone();
            for script in [
                &*SESSION_META_WRITE_SCRIPT,
                &*SESSION_META_VETO_SCRIPT,
                &*SESSION_INDEX_RELEASE_SCRIPT,
            ] {
                script
                    .load(&loader)
                    .await
                    .expect("SCRIPT LOAD of a session script must succeed");
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
            let attempts: Arc<Mutex<Vec<String>>> = Arc::default();
            let record: PostCommitHook = {
                let attempts = attempts.clone();
                Arc::new(move |session_id: String| {
                    attempts.lock().unwrap().push(session_id);
                    futures::future::ready(()).boxed()
                })
            };
            let mut store = session_store(tv)
                .with_pre_dispatch_hook_for_test(hook)
                .with_post_commit_hook_for_test(record);
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
                attempts,
                observer,
                key_prefix,
            }
        }

        /// Every attempt's `session_id` that reached `finish_create`, in order.
        fn attempts(&self) -> Vec<String> {
            self.attempts.lock().unwrap().clone()
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
        /// release is 5 s away) leaves the whole pool stalled again, so what
        /// follows is the same shape as the other cases: the failing command,
        /// then the release, whichever connections they land on.
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

        /// Return as soon as the in-doubt release's command reaches a stalled
        /// connection (after [`Self::let_the_sismember_through`], for the
        /// re-scan case). The caller then releases the stalls, so the release
        /// is answered well inside its own 5 s deadline while the failing
        /// command it follows has already timed out. Waits only — it never
        /// releases — so dropping it mid-wait leaves every stall intact for
        /// [`Self::release_remaining`].
        ///
        /// Keyed to the traffic — the bytes the server has read from the
        /// stalled connections — not to the timeout metric, which
        /// `release_in_doubt_attempt` ticks only once its release has returned.
        /// Nothing else uses the data pool under the stall, so the first new
        /// bytes after the park are the failing command, and the next are the
        /// release's, which it sends only once the failing command has timed
        /// out.
        async fn wait_until_the_release_arrives(&self) {
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
            tokio::time::sleep(FRAME_SETTLE).await;
            let after_failing = total_bytes_in(&held).await;
            while total_bytes_in(&held).await == after_failing {
                wait_until("the release");
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }

        /// Wait until a command reaches a still-stalled connection (new bytes
        /// since its baseline), let its frame finish arriving, then re-baseline
        /// every stalled connection, so the next call sees only later traffic.
        /// Returns that connection's id. `what` names the awaited command in
        /// the panic past [`SCENARIO_BUDGET`].
        async fn next_arrival(&self, what: &str) -> String {
            let deadline = Instant::now() + SCENARIO_BUDGET;
            self.wait_parked(deadline).await;
            let mut held = self.held.lock().await;
            let idx = loop {
                if let Some(idx) = first_with_new_bytes(&held).await {
                    break idx;
                }
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for {what} to reach a stalled data-pool connection"
                );
                tokio::time::sleep(POLL_INTERVAL).await;
            };
            tokio::time::sleep(FRAME_SETTLE).await;
            for parked in held.iter_mut() {
                parked.parked_bytes = parked.bytes_in().await;
            }
            held[idx].stall.conn_id().to_owned()
        }

        /// Release the stall on connection `id` alone, so exactly the commands
        /// queued on it run, and return that connection's client.
        async fn release_conn(&self, id: &str) -> Client {
            let mut held = self.held.lock().await;
            let idx = held
                .iter()
                .position(|parked| parked.stall.conn_id() == id)
                .unwrap_or_else(|| panic!("connection {id} is not stalled"));
            let ParkedConn { conn, stall, .. } = held.remove(idx);
            drop(held);
            let reply = stall.release().await;
            assert!(
                matches!(&reply, Ok(v) if !v.is_null()),
                "the stall on connection {id} must have held until its release; got {reply:?}"
            );
            conn
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

        /// Follow a [`Failing::MetaWrite`] attempt to its settle's veto: the
        /// metadata write reaches a stalled connection, times out, and the
        /// settle's veto reaches another. Returns `(write_conn, veto_conn)`.
        async fn stage_the_settle(&self) -> (String, String) {
            let write_conn = self.next_arrival("the metadata write").await;
            let veto_conn = self.next_arrival("the settle's veto").await;
            assert_ne!(
                write_conn, veto_conn,
                "NOT STAGED: the settle's veto queued behind the timed-out metadata write on \
                 one connection, so the two cannot be ordered"
            );
            (write_conn, veto_conn)
        }

        /// Load the session scripts into the store's script cache, on the
        /// observer, just before a parked command whose caller has already
        /// given up is released (see *Staging*): such a command is never
        /// reloaded after a `NOSCRIPT`, so a concurrent test's `SCRIPT FLUSH`
        /// since [`Self::new`]'s load would make it fail instead of running.
        async fn reload_scripts(&self) {
            for script in [
                &*SESSION_META_WRITE_SCRIPT,
                &*SESSION_META_VETO_SCRIPT,
                &*SESSION_INDEX_RELEASE_SCRIPT,
            ] {
                script
                    .load(&self.observer)
                    .await
                    .expect("SCRIPT LOAD of a session script must succeed");
            }
        }

        /// Run the late metadata write for `session_id` parked on `write_conn`
        /// (its caller timed out long ago) and return once it has run: the
        /// connection's next command is answered only after it. The session
        /// scripts are loaded again first, so the write runs its script, and
        /// [`assert_late_script_ran`] proves it did: the write script's own
        /// `EXISTS` of the id's veto key is on the store's `MONITOR` feed.
        async fn run_the_late_write(&self, write_conn: &str, session_id: &str) {
            let veto_key = self.store_keys().session_meta_veto(session_id);
            let mut tap = MonitorTap::start(&test_valkey_addr()).await;
            self.reload_scripts().await;
            let conn = self.release_conn(write_conn).await;
            let fence = monitor_fence(&self.key_prefix);
            let _: i64 = conn
                .exists(&fence)
                .await
                .expect("a round trip behind the late metadata write must succeed");
            assert_late_script_ran(
                &mut tap,
                &fence,
                "EXISTS",
                &veto_key,
                "the late metadata write",
            )
            .await;
        }

        /// From [`Self::stage_the_settle`]: let the veto run (`VETOED`), then
        /// run the timed-out metadata write late (it must write nothing), and
        /// wait for the vetoed attempt's release to reach a stalled connection.
        /// Returns `(abandoned_id, release_conn)`.
        async fn stage_a_vetoed_attempt_with_a_stalled_release(&self) -> (String, String) {
            let (write_conn, veto_conn) = self.stage_the_settle().await;
            let abandoned = self.attempts()[0].clone();
            self.release_conn(&veto_conn).await;
            let release_conn = self.next_arrival("the vetoed attempt's release").await;
            self.run_the_late_write(&write_conn, &abandoned).await;
            assert!(
                !self.observer_has_meta(&abandoned).await,
                "the late metadata write of a vetoed id must write nothing"
            );
            (abandoned, release_conn)
        }

        /// Whether `session_id` has metadata, read on the observer (the data
        /// pool may still be stalled).
        async fn observer_has_meta(&self, session_id: &str) -> bool {
            let found: i64 = self
                .observer
                .exists(self.store_keys().session_meta(session_id))
                .await
                .expect("EXISTS on the observer must succeed");
            found == 1
        }

        /// Poll on the observer until `key` exists.
        async fn await_key_on_observer(&self, key: String, why: &str) {
            let deadline = Instant::now() + SCENARIO_BUDGET;
            loop {
                let found: i64 = self
                    .observer
                    .exists(&key)
                    .await
                    .expect("EXISTS on the observer must succeed");
                if found == 1 {
                    return;
                }
                assert!(Instant::now() < deadline, "{why}");
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }

        fn store_keys(&self) -> live_edit::storage::keys::KeySchema {
            live_edit::storage::keys::KeySchema::new(
                self.key_prefix
                    .strip_suffix("b62m-data-stall:")
                    .expect("the stall key prefix extends the fixture prefix"),
            )
        }
    }

    /// Assert the late command released just before `fence` ran its script:
    /// the `MONITOR` feed shows, before the fence, a `command` the script
    /// itself issued naming exactly `key` (a key unique to this test, so no
    /// other test's script matches). `fence` must have been sent on the late
    /// command's own connection, so its reply came after the command's.
    ///
    /// The late command's caller gave up long ago, so a `NOSCRIPT` answer is
    /// never reloaded: if a concurrent test's `SCRIPT FLUSH` lands between
    /// [`Staged::reload_scripts`] and the command (see *Staging*), the command
    /// never runs, and an assertion that it wrote nothing would pass on a
    /// command that did nothing. This fails that run by name instead.
    async fn assert_late_script_ran(
        tap: &mut MonitorTap,
        fence: &str,
        command: &str,
        key: &str,
        what: &str,
    ) {
        let ran = tap.drain_until(fence).await.into_iter().any(|c| {
            c.issued_by_script() && c.name() == command && c.args.get(1).is_some_and(|k| k == key)
        });
        assert!(
            ran,
            "{what} never ran its script: no script-issued `{command} {key}` reached the store \
             before the fence. Its `EVALSHA` most likely failed `NOSCRIPT`, a concurrent test's \
             `SCRIPT FLUSH` landing between the reload and the command, so this run cannot show \
             what the late command does; rerun it"
        );
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

    /// Wait for every data-pool connection to be connected again after
    /// [`Staged::kill_stalls`].
    async fn await_data_pool_reconnected(tv: &TestValkey) {
        tokio::time::timeout(RECONNECT_BUDGET, tv.client().pool.wait_for_connect())
            .await
            .expect("the data pool did not reconnect within the budget")
            .expect("the data pool reconnect must succeed");
    }

    /// Whether `session_id` carries the metadata veto.
    async fn vetoed(tv: &TestValkey, session_id: &str) -> bool {
        let found: i64 = tv
            .client()
            .pool
            .exists(tv.key_schema().session_meta_veto(session_id))
            .await
            .expect("EXISTS veto");
        found == 1
    }

    /// Whether `session_id` has metadata.
    async fn has_meta(tv: &TestValkey, session_id: &str) -> bool {
        let found: i64 = tv
            .client()
            .pool
            .exists(tv.key_schema().session_meta(session_id))
            .await
            .expect("EXISTS session_meta");
        found == 1
    }

    /// Assert `session_id` is the only holder of `name` at `base_path`, listed
    /// with its full metadata, with no ghost left behind, and the only
    /// `path_index` member there.
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
        let indexed: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().path_index(base_path))
            .await
            .expect("SMEMBERS on the path index must succeed");
        assert_eq!(
            indexed,
            vec![session_id.to_owned()],
            "the path index holds exactly this call's session: no member was left without its \
             names member"
        );
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
            "the failed clean-up leaves both members, never the names member alone"
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

    /// Drive an in-doubt create whose `failing` read times out and whose
    /// release is allowed to complete (the stall is released as soon as the
    /// release's command arrives). The release frees the name, so the failure
    /// is retried in-process under a new id and the create succeeds with a
    /// complete session — the only one listed. Exactly one data-pool timeout —
    /// the failing command's — is counted: the release is answered once the
    /// stall is released.
    async fn scenario_in_doubt_release_succeeds_and_the_retry_succeeds(
        failing: Failing,
        name: &str,
        base_path: &str,
    ) {
        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, failing).await;

        // Release the stalls once the release reaches the store, keeping the
        // create polled throughout. If the create returns first — a code path
        // that answers without releasing — the watcher would wait for bytes
        // that never come, so stop waiting then and report the create's own
        // answer, which is the informative failure. Either way every stall is
        // released before anything is asserted.
        let create = crate::grpc::log_capture::capture(tokio::time::timeout(
            SCENARIO_BUDGET,
            staged.store.create(name, base_path, CREATED_BY),
        ));
        tokio::pin!(create);
        let (outcome, captured) = tokio::select! {
            outcome = &mut create => outcome,
            () = staged.wait_until_the_release_arrives() => {
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
            Failing::MetaWrite => unreachable!("the metadata write is settled, not released"),
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
                    "a transient `{}` failure whose release succeeded must be retried \
                     in-process into a successful create; got {e:?}",
                    failing.op()
                )
            });
        staged.assert_staged();

        assert_one_complete_session(&tv, base_path, name, &session_id).await;
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "exactly one data-pool timeout: the `{}` itself (the release was answered after the \
             stall was released)",
            failing.op()
        );
    }

    /// Drive a create whose `failing` round trip times out and whose clean-up
    /// — the settle's veto for the metadata write, the release for the in-doubt
    /// reads — times out too (the stall is held until the create returns),
    /// then kill the stalls so nothing queued runs. The call must end with
    /// `Unavailable`, never retry (see `settle_failed_meta_write` /
    /// `release_in_doubt_attempt`, SDD §7.10): a wrong retry shows up here as a
    /// different op label on the error or extra timeouts on the row. One WARN
    /// names the attempt; the leftover — both members, no metadata, no veto —
    /// heals through ListSessions.
    async fn scenario_clean_up_fails_and_the_create_answers_unavailable(
        failing: Failing,
        name: &str,
        base_path: &str,
    ) {
        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, failing).await;

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
            "a failed clean-up must end the call with `Unavailable`. `AlreadyExists` cannot come \
             from this attempt's own member (the scans skip ids this call created), so it would \
             mean the call went on past the failed clean-up; got {err:?}"
        );
        assert_unavailable_timeout(&err, failing.op());
        // Two, both on data: the failing command, then its clean-up's one
        // script call, queued behind a stall. The commit's `EXEC` ran on the
        // txn pool, which is never stalled.
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 2),
            "the `{}` and its clean-up each time out once on the data pool",
            failing.op()
        );
        let attempts = staged.attempts();
        match failing {
            Failing::MetaWrite => assert_eq!(
                attempts.len(),
                1,
                "one attempt reached its metadata write, and no retry followed: {attempts:?}"
            ),
            Failing::InDoubtReconcile | Failing::InDoubtRescan => assert!(
                attempts.is_empty(),
                "an in-doubt attempt never reaches its metadata write: {attempts:?}"
            ),
        }

        let attempt_sid =
            assert_leftover_is_a_ghost_that_list_sessions_removes(&tv, base_path, name).await;
        if let Failing::MetaWrite = failing {
            assert_eq!(attempts, vec![attempt_sid.clone()]);
        }

        let (msg, error_field) = match failing {
            Failing::MetaWrite => (SETTLE_FAILED_MSG, "settle_error"),
            Failing::InDoubtReconcile | Failing::InDoubtRescan => {
                (IN_DOUBT_RELEASE_FAILED_MSG, "release_error")
            }
        };
        let warns = captured.with_message(msg);
        assert_eq!(
            warns.len(),
            1,
            "a failed clean-up must log exactly one not-retrying line; got {} — full capture:\n{}",
            warns.len(),
            captured.text
        );
        let warn = warns[0];
        assert_eq!(warn["level"], "WARN", "wrong level; full event: {warn}");
        assert_eq!(
            warn["fields"]["session_id"],
            attempt_sid.as_str(),
            "the WARN must name the attempt whose leftover stays behind; full event: {warn}"
        );
        assert!(
            warn["fields"][error_field]
                .as_str()
                .is_some_and(|e| e.contains("Timeout")),
            "the WARN must carry the clean-up's own error in `{error_field}`; full event: {warn}"
        );
        if !matches!(failing, Failing::MetaWrite) {
            assert_eq!(
                warn["fields"]["op"],
                failing.op(),
                "the WARN must name the round trip that failed; full event: {warn}"
            );
        }
    }

    /// b.2tu I7 — a timed-out metadata write whose settle's veto runs before
    /// the write itself (late, after its caller gave up): the veto answers
    /// `VETOED`, the late write is refused, and the call retries under a new
    /// id and succeeds. The abandoned id never gets metadata.
    ///
    /// Pre-fix: the late `HSET` landed after the undo, leaving orphan metadata
    /// under the abandoned id.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_settle_vetoing_before_the_late_metadata_write_retries_under_a_new_id() {
        const NAME: &str = "b2tu-settle-vetoed";
        const BASE_PATH: &str = "/b2tu-settle-vetoed";

        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, Failing::MetaWrite).await;

        // Joined, not raced: the create can finish while the staging is still
        // releasing stalls, and dropping it there would leave one parked.
        let staging = async {
            let (write_conn, veto_conn) = staged.stage_the_settle().await;
            let abandoned = staged.attempts()[0].clone();
            staged.release_conn(&veto_conn).await;
            staged
                .await_key_on_observer(
                    staged.store_keys().session_meta_veto(&abandoned),
                    "the settle's veto never wrote the veto",
                )
                .await;
            staged.run_the_late_write(&write_conn, &abandoned).await;
            let late_write_wrote = staged.observer_has_meta(&abandoned).await;
            staged.release_remaining().await;
            late_write_wrote
        };
        let ((outcome, captured), late_write_wrote) = tokio::join!(
            crate::grpc::log_capture::capture(tokio::time::timeout(
                SCENARIO_BUDGET,
                staged.store.create(NAME, BASE_PATH, CREATED_BY),
            )),
            staging,
        );
        staged.release_remaining().await;

        let session_id = outcome
            .expect("the create did not settle within the scenario budget")
            .unwrap_or_else(|e| {
                panic!("a vetoed attempt must retry a transient failure into an OK; got {e:?}")
            });
        staged.assert_staged();
        let attempts = staged.attempts();
        assert_eq!(attempts.len(), 2, "one retry: {attempts:?}");
        let abandoned = &attempts[0];
        assert_eq!(attempts[1], session_id, "the OK carries the retry's id");
        assert_ne!(abandoned, &session_id);
        assert!(
            !late_write_wrote,
            "the metadata write that ran after the veto must have written nothing"
        );
        assert!(
            !has_meta(&tv, abandoned).await,
            "the vetoed id must never get metadata"
        );
        assert!(
            vetoed(&tv, abandoned).await,
            "the settle vetoed the abandoned id"
        );
        assert_one_complete_session(&tv, BASE_PATH, NAME, &session_id).await;
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "one data-pool timeout: the metadata write (the veto and the release were answered)"
        );
        assert!(
            captured.with_message(VETOED_RELEASE_FAILED_MSG).is_empty()
                && captured.with_message(SETTLE_FAILED_MSG).is_empty(),
            "the veto and the release succeeded; full capture:\n{}",
            captured.text
        );
    }

    /// b.2tu I8 — a timed-out metadata write that runs (late) before its
    /// settle's veto: the veto finds the metadata (`PRESENT`), and the call
    /// answers OK with the attempt's own id, without a retry, logging one WARN.
    ///
    /// Pre-fix: the undo deleted the landed metadata and the retry answered a
    /// different id.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_metadata_write_landing_before_its_settle_answers_ok_with_its_own_id() {
        const NAME: &str = "b2tu-settle-present";
        const BASE_PATH: &str = "/b2tu-settle-present";

        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, Failing::MetaWrite).await;

        // Joined, not raced: the create finishes as soon as the veto's stall
        // is released, while the staging may still be releasing the others;
        // dropping it there would leave one parked.
        let staging = async {
            let (write_conn, _veto_conn) = staged.stage_the_settle().await;
            let attempt = staged.attempts()[0].clone();
            staged.run_the_late_write(&write_conn, &attempt).await;
            let late_write_wrote = staged.observer_has_meta(&attempt).await;
            staged.release_remaining().await;
            late_write_wrote
        };
        let ((outcome, captured), late_write_wrote) = tokio::join!(
            crate::grpc::log_capture::capture(tokio::time::timeout(
                SCENARIO_BUDGET,
                staged.store.create(NAME, BASE_PATH, CREATED_BY),
            )),
            staging,
        );
        staged.release_remaining().await;
        assert!(
            late_write_wrote,
            "the late metadata write ran before any veto, so it must have written"
        );

        let session_id = outcome
            .expect("the create did not settle within the scenario budget")
            .unwrap_or_else(|e| {
                panic!("a settle that finds the metadata must answer OK; got {e:?}")
            });
        staged.assert_staged();
        assert_eq!(
            staged.attempts(),
            vec![session_id.clone()],
            "the OK carries the one attempt's own id, with no retry"
        );
        assert!(!vetoed(&tv, &session_id).await, "PRESENT writes no veto");
        assert_one_complete_session(&tv, BASE_PATH, NAME, &session_id).await;
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "one data-pool timeout: the metadata write, counted once by the settle"
        );
        let warn = captured.only(SETTLE_FOUND_THE_WRITE_MSG);
        assert_eq!(warn["level"], "WARN", "wrong level; full event: {warn}");
        assert_eq!(warn["fields"]["session_id"], session_id.as_str());
    }

    /// A timed-out metadata write that runs (late), whose session another
    /// call's `DeleteSession` then deletes, all before the settle's veto runs
    /// (fail-on-old): the veto finds the metadata absent and the deletion
    /// marker present (`DELETED`), so the call answers OK with the attempt's
    /// own id, without a retry. It writes no veto, enqueues no checkpoint
    /// write, and the session stays deleted. With the veto script's marker
    /// check dropped, the veto answered `VETOED` and the call retried under a
    /// new id; with the settle's `DELETED` arm enqueuing a write, the session's
    /// checkpoint stamp key appeared.
    ///
    /// The `DeleteSession` is the real `SessionStore::delete`, sent on a
    /// client of its own while the fixture's data pool is stalled. The create
    /// carries a checkpoint writer, so an enqueue would show: a write's build
    /// mints the stamp key (`KeySchema::session_checkpoint_state`) before it
    /// finds the session gone, and nothing else here writes that key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_metadata_write_landing_then_deleted_before_its_settle_answers_ok_with_its_own_id(
    ) {
        use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
        use live_edit::checkpoint::{CheckpointStore, SessionCheckpointWriter};
        use tokio_util::task::TaskTracker;

        const NAME: &str = "b62m-settle-deleted";
        const BASE_PATH: &str = "/b62m-settle-deleted";

        let (tv, metrics) = fixture().await;
        let Some(deleter) = live_edit::storage::test_support::connect_test_client_or_skip(
            &crate::common::test_config(),
        )
        .await
        else {
            return;
        };
        let mut staged = Staged::new(&tv, Failing::MetaWrite).await;
        let tracker = TaskTracker::new();
        let writer = Arc::new(SessionCheckpointWriter::new(
            Arc::new(InMemoryCheckpointStore::new()) as Arc<dyn CheckpointStore>,
            tv.client().clone(),
            tv.key_schema().clone(),
            tracker.clone(),
            Arc::new(crate::common::metrics_scrape_only(NAME)),
            NAME.to_owned(),
            true,
        ));
        staged.store = staged.store.clone().with_checkpoint_writer(writer);

        // Joined, not raced, as in the `PRESENT` case above.
        let staging = async {
            let (write_conn, _veto_conn) = staged.stage_the_settle().await;
            let attempt = staged.attempts()[0].clone();
            staged.run_the_late_write(&write_conn, &attempt).await;
            let late_write_wrote = staged.observer_has_meta(&attempt).await;
            SessionStore::new(deleter.clone(), tv.key_schema().clone())
                .delete(&attempt)
                .await
                .expect("another call's DeleteSession");
            staged.release_remaining().await;
            late_write_wrote
        };
        let ((outcome, captured), late_write_wrote) = tokio::join!(
            crate::grpc::log_capture::capture(tokio::time::timeout(
                SCENARIO_BUDGET,
                staged.store.create(NAME, BASE_PATH, CREATED_BY),
            )),
            staging,
        );
        staged.release_remaining().await;
        tracker.close();
        tokio::time::timeout(SCENARIO_BUDGET, tracker.wait())
            .await
            .expect("the checkpoint writer did not drain within the scenario budget");
        assert!(
            late_write_wrote,
            "fixture: the late metadata write ran before any veto, so it wrote"
        );

        let session_id = outcome
            .expect("the create did not settle within the scenario budget")
            .unwrap_or_else(|e| {
                panic!("a settle that finds the session deleted must answer OK; got {e:?}")
            });
        staged.assert_staged();
        assert_eq!(
            staged.attempts(),
            vec![session_id.clone()],
            "the OK carries the one attempt's own id, with no retry"
        );
        assert!(!vetoed(&tv, &session_id).await, "DELETED writes no veto");
        assert!(
            !has_meta(&tv, &session_id).await,
            "the deleted session stays deleted"
        );
        let marked: i64 = tv
            .client()
            .pool
            .exists(tv.key_schema().session_deleted_marker(&session_id))
            .await
            .expect("EXISTS deletion marker");
        assert_eq!(marked, 1, "the deletion marker stays");
        let stamped: i64 = tv
            .client()
            .pool
            .exists(tv.key_schema().session_checkpoint_state(&session_id))
            .await
            .expect("EXISTS checkpoint stamp key");
        assert_eq!(
            stamped, 0,
            "no checkpoint write is enqueued for the deleted session: a write's build mints the \
             stamp key before it finds the session gone"
        );
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "one data-pool timeout: the metadata write, counted once by the settle"
        );
        assert!(
            captured.with_message(SETTLE_FAILED_MSG).is_empty(),
            "the settle succeeded; full capture:\n{}",
            captured.text
        );
    }

    /// A timed-out metadata write whose settle times out too answers
    /// `Unavailable`, releases nothing, and the leftover heals through
    /// ListSessions.
    ///
    /// Pre-fix: the undo's three commands, not one settle, timed out (four
    /// data-pool timeouts).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_timed_out_meta_write_with_a_failed_settle_answers_unavailable() {
        scenario_clean_up_fails_and_the_create_answers_unavailable(
            Failing::MetaWrite,
            "b62m-meta-write-stuck",
            "/b62m-meta-write-stuck",
        )
        .await;
    }

    /// b.2tu P1 — a vetoed attempt whose release then times out still retries:
    /// the veto, not the release, makes the retry safe (no write for the
    /// abandoned id can land). The abandoned id's members stay as a ghost —
    /// no metadata, vetoed — which the next ListSessions removes; one WARN
    /// reports the failed release. The timed-out metadata write, run late, is
    /// refused.
    ///
    /// Pre-fix there was no veto: a partial undo that released the names
    /// member retried, while its other commands' leftovers stayed behind for
    /// good (a `path_index` member and an orphan `session_meta`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_vetoed_attempt_whose_release_fails_still_retries_to_success() {
        const NAME: &str = "b2tu-vetoed-release-fails";
        const BASE_PATH: &str = "/b2tu-vetoed-release-fails";

        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, Failing::MetaWrite).await;

        let create = crate::grpc::log_capture::capture(tokio::time::timeout(
            SCENARIO_BUDGET,
            staged.store.create(NAME, BASE_PATH, CREATED_BY),
        ));
        tokio::pin!(create);
        let mut kill_ids: Vec<String> = Vec::new();
        let (outcome, captured) = tokio::select! {
            outcome = &mut create => outcome,
            (_abandoned, release_conn) = staged.stage_a_vetoed_attempt_with_a_stalled_release() => {
                kill_ids.push(release_conn.clone());
                // The release times out (5 s); then the retry's metadata write
                // arrives. Kill the release unrun and let the retry through.
                tokio::select! {
                    outcome = &mut create => outcome,
                    retry_conn = staged.next_arrival("the retry's metadata write") => {
                        assert_ne!(
                            retry_conn, release_conn,
                            "NOT STAGED: the retry's metadata write queued behind the release \
                             this test kills"
                        );
                        let ((), outcome) =
                            tokio::join!(staged.kill_then_release(&kill_ids), &mut create);
                        outcome
                    }
                }
            }
        };
        staged.kill_then_release(&kill_ids).await;
        await_data_pool_reconnected(&tv).await;

        let session_id = outcome
            .expect("the create did not settle within the scenario budget")
            .unwrap_or_else(|e| {
                panic!(
                    "a vetoed attempt retries a transient failure into an OK even when its \
                     release failed; got {e:?}"
                )
            });
        staged.assert_staged();
        assert_eq!(kill_ids.len(), 1, "the vetoed attempt's release was staged");
        let attempts = staged.attempts();
        assert_eq!(attempts.len(), 2, "one retry: {attempts:?}");
        let abandoned = attempts[0].clone();
        assert_eq!(attempts[1], session_id);

        // The abandoned id: vetoed, no metadata, both members (its release was
        // killed unrun) — a ghost beside the new session.
        assert!(
            vetoed(&tv, &abandoned).await,
            "the settle vetoed the abandoned id"
        );
        assert!(
            !has_meta(&tv, &abandoned).await,
            "the vetoed id must never get metadata"
        );
        let listed = session_store(&tv).list(BASE_PATH).await.expect("list");
        assert_eq!(
            listed
                .present
                .iter()
                .map(|m| m.session_id.as_str())
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
            vec![abandoned.as_str()],
            "the failed release leaves the abandoned id as one ghost"
        );

        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 2),
            "two data-pool timeouts: the metadata write and the vetoed attempt's release"
        );
        let warn = captured.only(VETOED_RELEASE_FAILED_MSG);
        assert_eq!(warn["level"], "WARN", "wrong level; full event: {warn}");
        assert_eq!(warn["fields"]["session_id"], abandoned.as_str());
        assert!(
            warn["fields"]["error"]
                .as_str()
                .is_some_and(|e| e.contains("Timeout")),
            "the WARN must carry the release's error; full event: {warn}"
        );
        assert!(
            captured.with_message(SETTLE_FAILED_MSG).is_empty(),
            "the settle succeeded; full capture:\n{}",
            captured.text
        );

        // ListSessions removes the ghost, leaving exactly the returned session.
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
    }

    /// The vetoed attempt's release, timed out, reaches the store only after
    /// the call returned OK — and the returned session is untouched, because
    /// the release names the abandoned attempt's id, which no later command of
    /// the call uses.
    ///
    /// Staged as [`create_session_vetoed_attempt_whose_release_fails_still_retries_to_success`]
    /// up to the retry's metadata write, which is let through while the
    /// release stays parked; once `create` has returned, the release is
    /// released — not killed — and runs.
    ///
    /// Pre-b.2tu (the late-undo variant of this test): with the undo but a
    /// retry reusing the attempt's id, the late `SREM` removed the new
    /// session's index entry and the late `DEL` its metadata.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_session_vetoed_attempts_release_landing_after_ok_leaves_the_new_session_intact()
    {
        const NAME: &str = "b2tu-late-release";
        const BASE_PATH: &str = "/b2tu-late-release";

        let (tv, metrics) = fixture().await;
        let staged = Staged::new(&tv, Failing::MetaWrite).await;

        let create = tokio::time::timeout(
            SCENARIO_BUDGET,
            staged.store.create(NAME, BASE_PATH, CREATED_BY),
        );
        tokio::pin!(create);
        let (abandoned, release_conn) = tokio::select! {
            outcome = &mut create => {
                staged.kill_stalls().await;
                panic!(
                    "the create returned before its vetoed attempt's release reached the store, \
                     so nothing was staged; got {outcome:?}"
                );
            }
            staged_ids = staged.stage_a_vetoed_attempt_with_a_stalled_release() => staged_ids,
        };

        // Finish the create while the release stays parked.
        let keep = [release_conn.clone()];
        let outcome = tokio::select! {
            outcome = &mut create => outcome,
            retry_conn = staged.next_arrival("the retry's metadata write") => {
                if retry_conn == release_conn {
                    staged.kill_stalls().await;
                    panic!(
                        "NOT STAGED: the retry's metadata write queued behind the release, so \
                         the release would run before `create` returned"
                    );
                }
                let ((), outcome) = tokio::join!(staged.release_all_but(&keep), &mut create);
                outcome
            }
        };
        let session_id = match outcome {
            Ok(Ok(sid)) => sid,
            other => {
                staged.kill_stalls().await;
                panic!("the retry must succeed before the late release lands; got {other:?}");
            }
        };
        staged.assert_staged();
        assert_ne!(abandoned, session_id);
        assert_eq!(
            staged.attempts(),
            vec![abandoned.clone(), session_id.clone()]
        );

        // Now the abandoned attempt's release lands, after `Ok`. Its caller
        // gave up long ago, so load the scripts again first (see *Staging*),
        // and prove it ran its script: its own `TYPE` probe of `path_index`
        // is on the feed before a fence sent on its connection, whose reply
        // comes only after the release's.
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;
        staged.reload_scripts().await;
        let conn = staged.release_conn(&release_conn).await;
        let fence = monitor_fence(&staged.key_prefix);
        let _: i64 = conn
            .exists(&fence)
            .await
            .expect("a round trip behind the late release must succeed");
        assert_late_script_ran(
            &mut tap,
            &fence,
            "TYPE",
            &tv.key_schema().path_index(BASE_PATH),
            "the late release",
        )
        .await;
        staged.release_remaining().await;
        let still: bool = tv
            .client()
            .pool
            .sismember(tv.key_schema().path_index(BASE_PATH), &abandoned)
            .await
            .expect("SISMEMBER on the path index must succeed");
        assert!(
            !still,
            "the late release ran, so it must have removed the abandoned id's path_index member"
        );

        assert_one_complete_session(&tv, BASE_PATH, NAME, &session_id).await;
        assert!(
            !has_meta(&tv, &abandoned).await,
            "the vetoed id must never get metadata"
        );
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 2),
            "the metadata write and the release each time out once on the data pool; the \
             retry's write and the late release add none"
        );
    }

    /// A commit that landed but whose reply was lost, followed by a timed-out
    /// reconciling `SISMEMBER`: the release removes the landed members and the
    /// retry creates the session.
    ///
    /// Pre-fix: the `SISMEMBER` failure returned a bare `Aborted`, the retry's
    /// duplicate scan found the attempt's own landed member, and the caller got
    /// `Err(AlreadyExists)` for the session it was creating.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_timed_out_reconcile_read_releases_the_landed_commit_and_retries_to_success(
    ) {
        scenario_in_doubt_release_succeeds_and_the_retry_succeeds(
            Failing::InDoubtReconcile,
            "b62m-sismember-retried",
            "/b62m-sismember-retried",
        )
        .await;
    }

    /// The same in-doubt `SISMEMBER` timeout with a release that also times
    /// out answers `Unavailable`, and leaves both landed members — never one —
    /// which heal through ListSessions (b.2tu I9).
    ///
    /// Pre-fix: bare `Aborted` → `Err(AlreadyExists)` after one timeout (the
    /// retry's duplicate scan runs on the unstalled txn pool and finds the
    /// landed member).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_timed_out_reconcile_read_with_a_failed_release_answers_unavailable() {
        scenario_clean_up_fails_and_the_create_answers_unavailable(
            Failing::InDoubtReconcile,
            "b62m-sismember-stuck",
            "/b62m-sismember-stuck",
        )
        .await;
    }

    /// A commit that landed but whose reply was lost, a `SISMEMBER` that
    /// confirms it landed, and a re-scan `SMEMBERS` that times out: the
    /// release removes the landed members and the retry creates the session.
    ///
    /// Pre-fix: `Err(Internal("SMEMBERS names failed: …"))`, deliberately
    /// non-retriable because a bare `Aborted` would retry into the landed
    /// member; the landed members stayed behind with no metadata.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_timed_out_rescan_releases_the_landed_commit_and_retries_to_success() {
        scenario_in_doubt_release_succeeds_and_the_retry_succeeds(
            Failing::InDoubtRescan,
            "b62m-rescan-retried",
            "/b62m-rescan-retried",
        )
        .await;
    }

    /// The same timed-out re-scan with a release that also times out answers
    /// `Unavailable`, and leaves both landed members, which heal through
    /// ListSessions (b.2tu I9).
    ///
    /// Pre-fix: `Err(Internal("SMEMBERS names failed: …"))`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_in_doubt_timed_out_rescan_with_a_failed_release_answers_unavailable() {
        scenario_clean_up_fails_and_the_create_answers_unavailable(
            Failing::InDoubtRescan,
            "b62m-rescan-stuck",
            "/b62m-rescan-stuck",
        )
        .await;
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

        // One, on txn: attempt 1's in-doubt EXEC. Its UNWATCH and the late
        // commit were released well inside their deadlines.
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("txn", 1),
            "only attempt 1's EXEC times out"
        );
    }
}
