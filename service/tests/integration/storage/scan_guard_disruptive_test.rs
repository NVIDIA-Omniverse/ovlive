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

//! The serving-process guard (`live_edit::storage::scan_guard`, b.gns) on a
//! real topology change, where the walk seam cannot stand in: a hash slot
//! migrated between primaries while a walk runs, and a server process
//! restarted between two steps.
//!
//! - **A keyed walk across a slot migration** (test 10): a `ZSCAN` walk of a
//!   table-encoded sorted set whose slot moves to another primary between
//!   its first and second steps fails; it never returns as if complete.
//! - **A keyspace walk across a slot migration** (test 10): a slot that is
//!   no primary's pin slot moves while the walk runs; the end fingerprint's
//!   slot map differs from the start's, and the walk fails.
//! - **A walk across `DEBUG RESTART`** (test 11, R2): a restartable Valkey
//!   restarted between two `ZSCAN` steps comes back with a new run_id, and
//!   the walk fails. A report-only datapoint follows: the same cursor
//!   continued unguarded on the new process, members returned against
//!   `ZCARD`.
//!
//! **Disruptive and twice gated.** Every test here is `#[ignore]`d with the
//! `ci: test-cluster-disruptive` label and run only by CI's
//! `test-cluster-disruptive:` job, whose `-E` filter selects them by name:
//! the migrations by `slot_migration_disruptive` (which also puts them in
//! `.config/nextest.toml`'s one-at-a-time `disruptive-cluster` group, with
//! their output kept on a pass), the restart by
//! `serving_process_restart_disruptive` (its datapoint kept by its own
//! override). The migrations read `LIVE_EDIT_TEST_CLUSTER_URL` and skip
//! without it; the restart reads `LIVE_EDIT_TEST_RESTARTABLE_VALKEY_URL`,
//! the job's `valkey-restartable` service (`--enable-debug-command yes`),
//! skips without it outside CI and fails without it under `GITLAB_CI`. The
//! module needs `test-fault-injection`, the feature the shared
//! `SlotMigration` helper is built under.
//!
//! **Retry safety.** Each migration completes through `SlotMigration`, whose
//! `Drop` guard leaves the slot consistent if a test panics mid-move, and
//! each test moves its slot back before it ends. A restart ends with the
//! server answering under its new process.
//!
//! **Worst case per test**, every wait at its budget, against nextest's
//! 120 s kill: the keyed migration test bounds its walk and first move
//! together by [`WALK_BUDGET`] plus two [`CONVERGENCE_BUDGET`]s (35 s), then
//! moves the slot back (one convergence wait, 10 s, beside
//! `SlotMigration`'s drain, bounded at 30 s and one `GETKEYSINSLOT` on an
//! almost empty slot): about 75 s. The keyspace migration test: two moves
//! (up to 40 s each) and the rest of its walk (15 s): about 95 s. The
//! restart test: its walk and the wait for the new process together,
//! [`WALK_BUDGET`] plus [`RESTART_BUDGET`] (35 s). The healthy path takes
//! well under a second of each.
//!
//! **What a red run means.** Only the classification-independent contract is
//! asserted: the walk never returns `Ok` across the change. Which error the
//! keyed walk ends with (a change of process after fred's re-sync, or a step
//! that failed every attempt) depends on how fred meets the redirect, and is
//! printed, not asserted (`docs-internal/test-writing-guide.md`, "Don't
//! assert an open question").
//!
//! Not run in this repository's sandbox, which has neither a cluster nor a
//! restartable Valkey: these tests first run in CI.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{ClientLike, ClusterInterface, KeysInterface, SortedSetsInterface};
use fred::types::config::{Options, Server};
use fred::types::{CustomCommand, InfoKind, Value};

use live_edit::storage::scan_guard::{parse_run_id, KeyspaceWalk, ServingChange, WalkError};
use live_edit::storage::test_support::{zscan_members_by_cursor_for_test, TestValkey};

use crate::storage::cluster_subscriber_test::SlotMigration;
use crate::storage::scan_guard_test::{cluster_fixture, seed_sorted_set, table_node, COUNT};

/// The variable naming the restartable Valkey test 11 runs against.
const RESTARTABLE_VALKEY_URL: &str = "LIVE_EDIT_TEST_RESTARTABLE_VALKEY_URL";

/// A bound on one whole walk, far above a healthy walk's milliseconds.
const WALK_BUDGET: Duration = Duration::from_secs(15);

/// How long every primary may take to agree on a moved slot's new owner.
const CONVERGENCE_BUDGET: Duration = Duration::from_secs(10);

/// How long the restarted server may take to answer under a new process.
const RESTART_BUDGET: Duration = Duration::from_secs(20);

/// A primary in `tv`'s routing table other than `not`. Shared with
/// `b_yuo_store_memory_tests`' `slot_migration_disruptive` test, as is
/// [`move_slot`].
pub(crate) fn another_primary(tv: &TestValkey, not: &Server) -> Server {
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a routing table")
        .unique_primary_nodes()
        .into_iter()
        .find(|server| server != not)
        .expect("a three-shard cluster has another primary")
}

/// The port `node`'s own `CLUSTER SLOTS` names as `slot`'s owner.
async fn owner_port_seen_by(tv: &TestValkey, node: &Server, slot: u16) -> Option<i64> {
    let pinned = Options {
        cluster_node: Some(node.clone()),
        max_attempts: Some(1),
        ..Default::default()
    };
    let reply: Value = tv
        .client()
        .pool
        .next()
        .with_options(&pinned)
        .cluster_slots()
        .await
        .ok()?;
    reply.into_array().into_iter().find_map(|entry| {
        let fields = entry.into_array();
        let start = fields.first()?.as_i64()?;
        let end = fields.get(1)?.as_i64()?;
        if !(start..=end).contains(&i64::from(slot)) {
            return None;
        }
        fields.get(2)?.clone().into_array().get(1)?.as_i64()
    })
}

/// Move `slot` from `src` to `dst` (`SlotMigration`: every key in it, then
/// the slot itself), and wait until every primary names `dst` its owner,
/// so a `CLUSTER SLOTS` from any node reads the new map.
pub(crate) async fn move_slot(admin: &TestValkey, slot: u16, src: &Server, dst: &Server) {
    let mut migration = SlotMigration::begin(admin, slot, src.clone(), dst.clone()).await;
    migration.complete().await;
    drop(migration);
    let nodes = admin
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a routing table")
        .unique_primary_nodes();
    let deadline = tokio::time::Instant::now() + CONVERGENCE_BUDGET;
    for node in nodes {
        loop {
            if owner_port_seen_by(admin, &node, slot).await == Some(i64::from(dst.port)) {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "{node} did not name {dst} the owner of slot {slot} within {CONVERGENCE_BUDGET:?}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// Test 10, a keyed walk. A `ZSCAN` walk of a 200-member sorted set by
/// steps of 10 on one client, whose key's slot another client moves to
/// another primary after the walk's first step: the walk does not return
/// as if complete. The error it ends with is printed (`B.GNS DATAPOINT:`).
/// The slot is moved back before the test ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, migrates a hash slot between shards"]
async fn scan_guard_slot_migration_disruptive_a_keyed_walk_fails_when_its_slot_moves_mid_walk() {
    let Some(walker) = cluster_fixture().await else {
        return;
    };
    let Some(admin) = cluster_fixture().await else {
        return;
    };
    let key = format!("{}{{gns-keyed-move}}:index", walker.prefix());
    seed_sorted_set(&walker, &key).await;
    let slot = fred::util::redis_keyslot(key.as_bytes());
    let src = table_node(&admin, slot);
    let dst = another_primary(&admin, &src);

    let moved = Arc::new(AtomicBool::new(false));
    let walked = {
        let (moved, admin, src, dst) = (Arc::clone(&moved), &admin, src.clone(), dst.clone());
        tokio::time::timeout(
            WALK_BUDGET + 2 * CONVERGENCE_BUDGET,
            zscan_members_by_cursor_for_test(walker.client(), &key, COUNT, move |_step| {
                let first = !moved.swap(true, Ordering::SeqCst);
                let (src, dst) = (src.clone(), dst.clone());
                async move {
                    if first {
                        move_slot(admin, slot, &src, &dst).await;
                    }
                }
            }),
        )
        .await
        .expect("the walk ends within its budget")
    };
    assert!(
        moved.load(Ordering::SeqCst),
        "fixture: the slot moved after the first step"
    );

    eprintln!(
        "B.GNS DATAPOINT: a ZSCAN walk across a migration of its slot from {src} to {dst} ended \
         with {}",
        match &walked {
            Ok(members) => format!("Ok ({} members)", members.len()),
            Err(e) => format!("{e:?}"),
        }
    );
    assert!(
        !matches!(walked, Ok(_) | Err(WalkError::WitnessMissing(_))),
        "a walk whose slot moved between two steps fails as a change or a store fault, never \
         as complete and never as a missing witness: {walked:?}"
    );

    move_slot(&admin, slot, &dst, &src).await;
    println!(
        "CLUSTER-GATED RAN: scan_guard_slot_migration_disruptive_a_keyed_walk_fails_when_its_slot_moves_mid_walk"
    );
}

/// Test 10, a keyspace walk. A keyspace walk on one client, during which
/// another client moves a slot of the first primary that is not its pin
/// slot to another primary (after the walk's first page): the end
/// fingerprint's `CLUSTER SLOTS` map differs from the start's, and the walk
/// fails with `SlotMap`, naming both maps. The slot is moved back before the
/// test ends.
///
/// The test pins the `SlotMap` arm because `check_end` compares the two
/// maps before it checks each primary's ownership. With that map comparison
/// removed, the walk still fails, as `NotOwner`: the first primary's node no
/// longer serves the moved slot. The code before b.gns, with no guard,
/// returns `Ok`. (Not run in this sandbox, which has no cluster.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, migrates a hash slot between shards"]
async fn scan_guard_slot_migration_disruptive_a_keyspace_walk_fails_when_a_slot_moves_mid_walk() {
    let Some(walker) = cluster_fixture().await else {
        return;
    };
    let Some(admin) = cluster_fixture().await else {
        return;
    };
    for i in 0..60 {
        let _: () = walker
            .client()
            .pool
            .set(
                format!("{}{{t{i}}}:k", walker.prefix()),
                "1",
                None,
                None,
                false,
            )
            .await
            .expect("SET must succeed");
    }
    let pattern = format!("{}*", walker.prefix());
    let first = table_node(&admin, 0);
    // Slot 1: the first primary's, and no primary's pin slot (the first
    // primary pins slot 0).
    let slot = 1;
    let dst = another_primary(&admin, &first);

    let mut walk = KeyspaceWalk::start(walker.client().pool.next(), &pattern)
        .await
        .expect("the walk starts on a stable cluster");
    let page = walk.next_page().await;
    assert!(matches!(page, Ok(Some(_))), "the first page: {page:?}");
    move_slot(&admin, slot, &first, &dst).await;
    let rest = tokio::time::timeout(WALK_BUDGET, async {
        while walk.next_page().await?.is_some() {}
        Ok::<(), WalkError>(())
    })
    .await
    .expect("the walk ends within its budget");

    match rest {
        Err(WalkError::Changed(change)) => match *change {
            ServingChange::SlotMap {
                ref before,
                ref after,
                ..
            } => assert_ne!(before, after, "both maps are named"),
            other => panic!("a moved slot changes the slot map, got {other:?}"),
        },
        other => panic!("a walk across a slot move fails, got {other:?}"),
    }

    move_slot(&admin, slot, &dst, &first).await;
    println!(
        "CLUSTER-GATED RAN: scan_guard_slot_migration_disruptive_a_keyspace_walk_fails_when_a_slot_moves_mid_walk"
    );
}

/// The restartable Valkey's URL; `None`, after a `SKIP:` line, when the
/// variable is unset outside CI. Unset under `GITLAB_CI` it is broken job
/// wiring, and the test fails.
fn restartable_valkey_url() -> Option<String> {
    match std::env::var(RESTARTABLE_VALKEY_URL) {
        Ok(url) => Some(url),
        Err(_) => {
            assert!(
                std::env::var_os("GITLAB_CI").is_none(),
                "{RESTARTABLE_VALKEY_URL} is unset in a CI job that runs the restart test; the \
                 job must point it at its own `valkey-restartable` service"
            );
            eprintln!("SKIP: {RESTARTABLE_VALKEY_URL} not set");
            None
        }
    }
}

/// The server's run_id as `client` reads it now, or `None` while it does
/// not answer.
async fn run_id_of(client: &fred::clients::Client) -> Option<String> {
    let info: String = tokio::time::timeout(
        Duration::from_secs(2),
        client.info::<String>(Some(InfoKind::Server)),
    )
    .await
    .ok()?
    .ok()?;
    parse_run_id(&info).ok()
}

/// One unguarded `ZSCAN key cursor COUNT count` on `client`: the next
/// cursor and the step's members (scores dropped).
async fn zscan_unguarded(
    client: &fred::clients::Client,
    key: &str,
    cursor: &str,
) -> Result<(String, Vec<String>), fred::error::Error> {
    let reply: Value = client
        .custom(
            CustomCommand::new_static("ZSCAN", fred::types::ClusterHash::FirstValue, false),
            vec![
                Value::from(key),
                Value::from(cursor),
                Value::from("COUNT"),
                Value::from(i64::try_from(COUNT).expect("fits")),
            ],
        )
        .await?;
    let mut parts = reply.into_array().into_iter();
    let next = parts.next().and_then(|v| v.as_string()).unwrap_or_default();
    let elements = parts.next().map(Value::into_array).unwrap_or_default();
    let members = elements
        .into_iter()
        .step_by(2)
        .filter_map(|v| v.as_string())
        .collect();
    Ok((next, members))
}

/// Test 11 (R2). A `ZSCAN` walk of a 200-member sorted set by steps of 10
/// on a restartable Valkey, restarted (`DEBUG RESTART`, which keeps the PID
/// and the data it saves on the way down) after the walk's first step: the
/// server answers under a new run_id, and the walk fails — as a change of
/// serving process, or as a step that failed every attempt while the server
/// was down; never as complete.
///
/// Report-only (`B.GNS DATAPOINT:`, kept by `.config/nextest.toml`'s
/// override for this test): a cursor taken from the old process before the
/// restart, continued unguarded on the new one, and the distinct members it
/// returned against `ZCARD` — what the guard exists to refuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, restarts its own Valkey server (DEBUG RESTART)"]
async fn scan_guard_serving_process_restart_disruptive_a_walk_fails_across_a_debug_restart() {
    let Some(url) = restartable_valkey_url() else {
        return;
    };
    let tv = TestValkey::setup_with_url(url, 1).await;
    let key = format!("{}gns-restart:index", tv.prefix());
    seed_sorted_set(&tv, &key).await;
    let admin = tv.client().maintenance_pool.next().clone();
    let before = run_id_of(&admin)
        .await
        .expect("the server reports a run_id");

    let restarted: Arc<std::sync::Mutex<Option<(String, String)>>> =
        Arc::new(std::sync::Mutex::new(None));
    let walked = {
        let (restarted, admin, key, before) = (
            Arc::clone(&restarted),
            admin.clone(),
            key.clone(),
            before.clone(),
        );
        tokio::time::timeout(
            WALK_BUDGET + RESTART_BUDGET,
            zscan_members_by_cursor_for_test(tv.client(), &key.clone(), COUNT, move |_step| {
                let first = restarted.lock().unwrap().is_none();
                let (restarted, admin, key, before) = (
                    Arc::clone(&restarted),
                    admin.clone(),
                    key.clone(),
                    before.clone(),
                );
                async move {
                    if !first {
                        return;
                    }
                    // A cursor from the old process, for the datapoint.
                    let (cursor, _) = zscan_unguarded(&admin, &key, "0")
                        .await
                        .expect("an unguarded first step on the old process");
                    let once = Options {
                        max_attempts: Some(1),
                        timeout: Some(Duration::from_secs(5)),
                        ..Default::default()
                    };
                    // The connection drops as the server restarts, so the
                    // reply is an error; the new run_id is the evidence.
                    let _ = admin
                        .with_options(&once)
                        .custom::<Value, Value>(
                            CustomCommand::new_static("DEBUG", None::<u16>, false),
                            vec![Value::from("RESTART")],
                        )
                        .await;
                    let deadline = tokio::time::Instant::now() + RESTART_BUDGET;
                    let after = loop {
                        if let Some(now) = run_id_of(&admin).await {
                            if now != before {
                                break now;
                            }
                        }
                        assert!(
                            tokio::time::Instant::now() < deadline,
                            "the server did not answer under a new run_id within \
                             {RESTART_BUDGET:?}"
                        );
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    };
                    *restarted.lock().unwrap() = Some((cursor, after));
                }
            }),
        )
        .await
        .expect("the walk ends within its budget")
    };
    let (cursor, after) = restarted
        .lock()
        .unwrap()
        .clone()
        .expect("fixture: the server restarted after the first step");

    eprintln!(
        "B.GNS DATAPOINT: a ZSCAN walk across DEBUG RESTART (run_id {before} -> {after}) ended \
         with {}",
        match &walked {
            Ok(members) => format!("Ok ({} members)", members.len()),
            Err(e) => format!("{e:?}"),
        }
    );
    assert!(
        !matches!(walked, Ok(_) | Err(WalkError::WitnessMissing(_))),
        "a walk across a restart fails as a change or a store fault, never as complete and \
         never as a missing witness: {walked:?}"
    );

    // Report only: the old cursor continued unguarded on the new process.
    let card: i64 = admin.zcard(&key).await.unwrap_or(-1);
    let mut seen = std::collections::BTreeSet::new();
    let mut next = cursor.clone();
    let mut steps = 0;
    while next != "0" && steps < 1_000 {
        match zscan_unguarded(&admin, &key, &next).await {
            Ok((cursor, members)) => {
                seen.extend(members);
                next = cursor;
            }
            Err(e) => {
                eprintln!("B.GNS DATAPOINT: the unguarded continuation failed: {e}");
                break;
            }
        }
        steps += 1;
    }
    eprintln!(
        "B.GNS DATAPOINT: the old process's cursor {cursor}, continued unguarded on the new \
         process, returned {} distinct members in {steps} steps; ZCARD after the restart is \
         {card} (the old process's first step is not counted)",
        seen.len()
    );
}
