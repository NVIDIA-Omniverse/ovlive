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

//! The gap the keyspace walk's node pins and ownership check close
//! (`live_edit::storage::scan_guard`, b.gns test 16), on a real cluster: a
//! client whose routing table predates a slot move.
//!
//! fred routes a slot by its cached table, which only connect, reconnect, a
//! `MOVED` or an explicit sync refresh, and a keyless `SCAN`, `INFO` or
//! `CLUSTER NODES` never draws a `MOVED`. So once slot 0 moves from the first
//! primary A to another primary B, a client that sent nothing keyed for slot
//! 0 still maps it to A: the fresh `CLUSTER SLOTS` map's first primary is
//! now B, pinned at slot 0, and the stale table resolves that pin to A. A
//! walk that trusted the table would walk A twice and B not at all, with
//! every other check passing. The guard reads the pinned node's own myself
//! line, finds A's slots where B's are expected, re-syncs the client's
//! table and fails `NotOwner`; the next walk on the same client returns
//! every key.
//!
//! The coverage walk (b.hyw T8, the service's non-Job keyspace walks through
//! `orphan_sweep::scan_keyspace`) meets the same table: it leaves B unwalked,
//! returns every other key, then answers `NotOwner` after re-syncing the
//! table, and its next walk on the same client returns every key. fred's own
//! cluster-wide `SCAN`, which `scan_keyspace` used before, missed B's keys on
//! every pass with no error.
//!
//! A store whose `CLUSTER NODES` gives no usable myself line (b.87a) cannot
//! show A serving A's slots, so the coverage walk walks A on B's pin too,
//! its ownership unproven. It then compares its map with the stale table:
//! the table still groups slot 0 with A's slots, the map with B's, so the
//! table is out of date (`StaleTable`). The walk returns every key but B's,
//! re-syncs the table and does not count every primary walked; its next walk
//! on the same client, on the refreshed table, returns every key, still
//! unproven (`WitnessMissing`) but complete.
//!
//! **Disruptive and twice gated**: `#[ignore]`d with the
//! `ci: test-cluster-disruptive` label, selected by that job's `-E` filter
//! through `slot_migration_disruptive` (which also puts it in
//! `.config/nextest.toml`'s one-at-a-time `disruptive-cluster` group, its
//! output kept on a pass), and skipped without `LIVE_EDIT_TEST_CLUSTER_URL`.
//! Needs `test-fault-injection`, the feature the shared `SlotMigration`
//! helper and the walk seam (`arm_walk_fault_for_test`) are built under.
//! Self-contained, so it can be lifted out whole with its `mod` line.
//!
//! **Retry safety.** The move goes through `SlotMigration`, whose `Drop`
//! guard leaves the slot consistent if the test panics mid-move, and the
//! test moves slot 0 back before it ends.
//!
//! **Worst case** per test, every wait at its budget, against nextest's 120 s
//! kill: two walks of [`WALK_BUDGET`] (15 s each) and two moves, each a
//! convergence wait of [`CONVERGENCE_BUDGET`] (10 s) beside
//! `SlotMigration`'s drain (bounded at 30 s, and one `GETKEYSINSLOT` on an
//! almost empty slot): about 110 s at every bound, under a second healthy.
//!
//! **Fail-on-old** (not run in this repository's sandbox, which has no
//! cluster): with the ownership check removed (`read_primary`'s slot-set
//! comparison), the first walk returns `Ok` without B's keys; each coverage
//! test states its own.

use std::collections::BTreeSet;
use std::time::Duration;

use fred::interfaces::{ClientLike, ClusterInterface, KeysInterface};
use fred::types::config::{Options, Server};
use fred::types::Value;

use live_edit::storage::scan_guard::{
    arm_walk_fault_for_test, collect_keyspace, KeyspaceWalk, MyselfLineSeen, RoutingRefresh,
    ServingChange, SlotSet, WalkEnd, WalkError, WalkFault, WitnessSeen,
};
use live_edit::storage::test_support::{test_cluster_url, TestValkey};

use crate::storage::cluster_subscriber_test::SlotMigration;

/// A bound on one whole walk, far above a healthy walk's milliseconds.
const WALK_BUDGET: Duration = Duration::from_secs(15);

/// A coverage walk of `pattern` on `client`, driven to its end within
/// [`WALK_BUDGET`]: every key it handed on, the reason it answered after its
/// last key (`None` when it completed), and whether it walked every primary.
async fn coverage_walk(
    client: &fred::clients::Client,
    pattern: &str,
) -> (BTreeSet<String>, Option<WalkError>, bool) {
    tokio::time::timeout(WALK_BUDGET, async {
        let mut walk = KeyspaceWalk::start_coverage(client, pattern)
            .await
            .expect("a coverage walk starts on anything but a store fault");
        let mut found = BTreeSet::new();
        let reason = loop {
            match walk.next_page().await {
                Ok(Some(page)) => found.extend(
                    page.into_iter()
                        .map(|key| key.as_str().expect("a UTF-8 key").to_owned()),
                ),
                Ok(None) => break None,
                Err(reason) => break Some(reason),
            }
        };
        (found, reason, walk.walked_every_primary())
    })
    .await
    .expect("the coverage walk ends within its budget")
}

/// How long every primary may take to agree on a moved slot's new owner.
const CONVERGENCE_BUDGET: Duration = Duration::from_secs(10);

/// The slot the test moves: the first primary's pin slot.
const MOVED_SLOT: u16 = 0;

/// The fixture on `LIVE_EDIT_TEST_CLUSTER_URL`, one data-pool client, or
/// `None` after a `SKIP:` line.
async fn cluster_fixture() -> Option<TestValkey> {
    let Some(url) = test_cluster_url() else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return None;
    };
    let tv = TestValkey::setup_with_url(url, 1).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL names a cluster"
    );
    Some(tv)
}

/// The node `client`'s routing table names for `slot`.
fn table_node(client: &fred::clients::Client, slot: u16) -> Server {
    client
        .cached_cluster_state()
        .expect("a connected cluster client has a routing table")
        .get_server(slot)
        .cloned()
        .expect("the table names the slot's owner")
}

/// Every primary's slot set as `node`'s own `CLUSTER SLOTS` gives it, keyed
/// by the primary's port.
async fn slot_sets_seen_by(tv: &TestValkey, node: &Server) -> Option<Vec<(i64, SlotSet)>> {
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
    let mut by_port: Vec<(i64, Vec<(u16, u16)>)> = Vec::new();
    for entry in reply.into_array() {
        let fields = entry.into_array();
        let start = u16::try_from(fields.first()?.as_i64()?).ok()?;
        let end = u16::try_from(fields.get(1)?.as_i64()?).ok()?;
        let port = fields.get(2)?.clone().into_array().get(1)?.as_i64()?;
        match by_port.iter_mut().find(|(p, _)| *p == port) {
            Some((_, ranges)) => ranges.push((start, end)),
            None => by_port.push((port, vec![(start, end)])),
        }
    }
    Some(
        by_port
            .into_iter()
            .map(|(port, ranges)| (port, SlotSet::from_ranges(ranges)))
            .collect(),
    )
}

/// Move [`MOVED_SLOT`] from `src` to `dst`, then wait until every primary
/// names `dst` its owner. The slot sets every primary then agrees on, by
/// port.
async fn move_slot(admin: &TestValkey, src: &Server, dst: &Server) -> Vec<(i64, SlotSet)> {
    let mut migration = SlotMigration::begin(admin, MOVED_SLOT, src.clone(), dst.clone()).await;
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
    let mut agreed = None;
    for node in nodes {
        loop {
            if let Some(sets) = slot_sets_seen_by(admin, &node).await {
                let owner = sets
                    .iter()
                    .find(|(_, set)| {
                        set.ranges()
                            .iter()
                            .any(|&(start, end)| (start..=end).contains(&MOVED_SLOT))
                    })
                    .map(|(port, _)| *port);
                if owner == Some(i64::from(dst.port)) {
                    agreed = Some(sets);
                    break;
                }
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "{node} did not name {dst} the owner of slot {MOVED_SLOT} within \
                 {CONVERGENCE_BUDGET:?}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    agreed.expect("a primary answered")
}

/// The slot set of the primary on `port`, among `sets`.
fn slots_of(sets: &[(i64, SlotSet)], port: u16) -> SlotSet {
    sets.iter()
        .find(|(p, _)| *p == i64::from(port))
        .map(|(_, set)| set.clone())
        .unwrap_or_else(|| panic!("no primary on port {port} in {sets:?}"))
}

/// Test 16. The walk's client connects first, so its routing table is built;
/// another client then moves slot 0 from the first primary A to another
/// primary B and writes a key under each of 60 hash tags, on every shard.
/// The walk's client sends nothing keyed for slot 0, so its table still maps
/// slot 0 to A. The first walk fails `NotOwner` at its start: the map's
/// first primary is B (slots 0 and B's own range), pinned at slot 0 to A,
/// whose myself line lists A's slots; the routing table was refreshed. The
/// second walk on the same client returns every key written.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, migrates a hash slot between shards"]
async fn scan_guard_slot_migration_disruptive_a_walk_on_a_stale_routing_table_fails_not_owner() {
    let Some(walker) = cluster_fixture().await else {
        return;
    };
    let Some(admin) = cluster_fixture().await else {
        return;
    };
    let walk_client = walker.client().pool.next();
    let a = table_node(walk_client, MOVED_SLOT);
    let b = admin
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a routing table")
        .unique_primary_nodes()
        .into_iter()
        .find(|server| *server != a)
        .expect("a three-shard cluster has another primary");

    let sets = move_slot(&admin, &a, &b).await;
    let written: BTreeSet<String> = (0..60)
        .map(|i| format!("{}{{t{i}}}:k", admin.prefix()))
        .collect();
    for key in &written {
        let _: () = admin
            .client()
            .pool
            .set(key, "1", None, None, false)
            .await
            .expect("SET must succeed");
    }
    let pattern = format!("{}*", admin.prefix());
    assert_eq!(
        table_node(walk_client, MOVED_SLOT),
        a,
        "fixture: the walk's client still maps slot {MOVED_SLOT} to its old owner"
    );

    let first = tokio::time::timeout(WALK_BUDGET, collect_keyspace(walk_client, &pattern))
        .await
        .expect("the first walk ends within its budget");
    match first {
        Err(WalkError::Changed(change)) => match *change {
            ServingChange::NotOwner {
                ref node,
                ref expected,
                ref served,
                at,
                ref refresh,
                ..
            } => {
                assert_eq!(node.as_deref(), Some(a.to_string().as_str()), "pinned to A");
                assert_eq!(
                    expected,
                    &slots_of(&sets, b.port),
                    "B's slots, slot 0 among them"
                );
                assert_eq!(served.as_ref(), Some(&slots_of(&sets, a.port)), "A's slots");
                assert_eq!(at, WalkEnd::Start);
                assert_eq!(refresh, &RoutingRefresh::Refreshed);
            }
            other => panic!("NotOwner at the start, got {other:?}"),
        },
        other => panic!(
            "a walk on a stale table fails NotOwner, got {:?}",
            other.map(|keys| keys.len())
        ),
    }
    assert_eq!(
        table_node(walk_client, MOVED_SLOT),
        b,
        "the NotOwner refresh re-synced the walk's client"
    );

    let second = tokio::time::timeout(WALK_BUDGET, collect_keyspace(walk_client, &pattern))
        .await
        .expect("the second walk ends within its budget")
        .expect("the second walk, on the refreshed table, completes");
    let found: BTreeSet<String> = second
        .into_iter()
        .map(|key| key.as_str().expect("a UTF-8 key").to_owned())
        .collect();
    assert_eq!(found, written, "every key written, on every shard");

    move_slot(&admin, &b, &a).await;
    println!(
        "CLUSTER-GATED RAN: scan_guard_slot_migration_disruptive_a_walk_on_a_stale_routing_table_fails_not_owner"
    );
}

/// b.hyw T8, the Issue's regression test: the same stale table under the
/// coverage walk, the walk `orphan_sweep::scan_keyspace` takes for the
/// checkpoint tick, the backup repair, the orphan sweep and the
/// reconciler's meta walk. The map's first primary is B, pinned at slot 0
/// to A by the stale table; A's myself line lists A's slots, a proven
/// non-owner, so B is left unwalked while A and the third primary are
/// walked (A's own pin, at its lowest slot, still resolves to A). The walk
/// returns every key but B's, then `NotOwner` at its start, after a re-sync
/// that refreshed the table, and does not count every primary walked. The
/// second coverage walk on the same client, on the refreshed table, returns
/// every key with no error.
///
/// Fail-on-old (stated, not runnable: the old walk is gone, and this sandbox
/// has no cluster): the old `scan_keyspace` planned its walks with fred's
/// `scan_cluster` from the same stale table, sent the slot-0 walk to A, and
/// ended `Ok` without B's keys, on this pass and every later one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, migrates a hash slot between shards"]
async fn scan_guard_slot_migration_disruptive_a_coverage_walk_on_a_stale_routing_table_skips_only_the_moved_primary(
) {
    let Some(walker) = cluster_fixture().await else {
        return;
    };
    let Some(admin) = cluster_fixture().await else {
        return;
    };
    let walk_client = walker.client().pool.next();
    let a = table_node(walk_client, MOVED_SLOT);
    let b = admin
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a routing table")
        .unique_primary_nodes()
        .into_iter()
        .find(|server| *server != a)
        .expect("a three-shard cluster has another primary");

    let sets = move_slot(&admin, &a, &b).await;
    let b_slots = slots_of(&sets, b.port);
    let written: BTreeSet<String> = (0..60)
        .map(|i| format!("{}{{t{i}}}:k", admin.prefix()))
        .collect();
    for key in &written {
        let _: () = admin
            .client()
            .pool
            .set(key, "1", None, None, false)
            .await
            .expect("SET must succeed");
    }
    let off_b: BTreeSet<String> = written
        .iter()
        .filter(|key| {
            let slot = fred::util::redis_keyslot(key.as_bytes());
            !b_slots
                .ranges()
                .iter()
                .any(|&(start, end)| (start..=end).contains(&slot))
        })
        .cloned()
        .collect();
    assert!(
        !off_b.is_empty() && off_b.len() < written.len(),
        "fixture: B and another primary hold keys"
    );
    let pattern = format!("{}*", admin.prefix());
    assert_eq!(
        table_node(walk_client, MOVED_SLOT),
        a,
        "fixture: the walk's client still maps slot {MOVED_SLOT} to its old owner"
    );

    let (found, reason, every_primary_walked) = coverage_walk(walk_client, &pattern).await;
    assert_eq!(found, off_b, "every key but B's: B was left unwalked");
    match reason {
        Some(WalkError::Changed(change)) => match *change {
            ServingChange::NotOwner {
                ref node,
                ref expected,
                ref served,
                at,
                ref refresh,
                ..
            } => {
                assert_eq!(node.as_deref(), Some(a.to_string().as_str()), "pinned to A");
                assert_eq!(expected, &b_slots, "B's slots, slot 0 among them");
                assert_eq!(served.as_ref(), Some(&slots_of(&sets, a.port)), "A's slots");
                assert_eq!(at, WalkEnd::Start);
                assert_eq!(refresh, &RoutingRefresh::Refreshed);
            }
            other => panic!("NotOwner at the start, got {other:?}"),
        },
        other => panic!("a coverage walk on a stale table ends NotOwner, got {other:?}"),
    }
    assert!(!every_primary_walked, "B was not walked");
    assert_eq!(
        table_node(walk_client, MOVED_SLOT),
        b,
        "the NotOwner re-sync refreshed the walk's client"
    );

    let (found, reason, every_primary_walked) = coverage_walk(walk_client, &pattern).await;
    assert!(
        reason.is_none(),
        "the second walk, on the refreshed table, ends with no error: {reason:?}"
    );
    assert_eq!(found, written, "every key written, on every shard");
    assert!(every_primary_walked);

    move_slot(&admin, &b, &a).await;
    println!(
        "CLUSTER-GATED RAN: scan_guard_slot_migration_disruptive_a_coverage_walk_on_a_stale_routing_table_skips_only_the_moved_primary"
    );
}

/// b.87a, the Issue's regression test: T8's stale table on a store whose
/// `CLUSTER NODES` gives no usable myself line. Through the walk seam, every
/// `CLUSTER NODES` read of both walks (reads 0–5: three pins at the start,
/// three at the end) has no myself line, so no pin's ownership is proven.
/// The map's first primary is B, pinned at slot 0 to A by the stale table,
/// and walked there: A twice, B not at all. The walk's table check then
/// finds the table it pinned from grouping slot 0 with A's slots where the
/// map groups it with B's: `StaleTable` at the start, after a re-sync that
/// refreshed the table. The walk returns every key but B's and does not
/// count every primary walked. The second walk on the same client, its pins
/// resolved from the refreshed table, returns every key and answers only
/// the missing myself line (`WitnessMissing`), every primary walked.
///
/// Fail-on-old (stated, not runnable: this sandbox has no cluster): before
/// b.87a an unproven pin was never checked against the table, so the first
/// walk answers `WitnessMissing` with every primary walked and misses B's
/// keys, nothing re-syncs, and the second walk misses them again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, migrates a hash slot between shards"]
async fn scan_guard_slot_migration_disruptive_a_coverage_walk_with_unproven_pins_on_a_stale_routing_table_re_syncs_it(
) {
    let Some(walker) = cluster_fixture().await else {
        return;
    };
    let Some(admin) = cluster_fixture().await else {
        return;
    };
    let walk_client = walker.client().pool.next();
    let a = table_node(walk_client, MOVED_SLOT);
    let b = admin
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a routing table")
        .unique_primary_nodes()
        .into_iter()
        .find(|server| *server != a)
        .expect("a three-shard cluster has another primary");

    let sets = move_slot(&admin, &a, &b).await;
    let b_slots = slots_of(&sets, b.port);
    let written: BTreeSet<String> = (0..60)
        .map(|i| format!("{}{{t{i}}}:k", admin.prefix()))
        .collect();
    for key in &written {
        let _: () = admin
            .client()
            .pool
            .set(key, "1", None, None, false)
            .await
            .expect("SET must succeed");
    }
    let off_b: BTreeSet<String> = written
        .iter()
        .filter(|key| {
            let slot = fred::util::redis_keyslot(key.as_bytes());
            !b_slots
                .ranges()
                .iter()
                .any(|&(start, end)| (start..=end).contains(&slot))
        })
        .cloned()
        .collect();
    assert!(
        !off_b.is_empty() && off_b.len() < written.len(),
        "fixture: B and another primary hold keys"
    );
    let pattern = format!("{}*", admin.prefix());
    assert_eq!(
        table_node(walk_client, MOVED_SLOT),
        a,
        "fixture: the walk's client still maps slot {MOVED_SLOT} to its old owner"
    );

    // Held across both walks: each walk numbers its own reads from 0.
    let unusable: Vec<_> = (0..6)
        .map(|read| {
            arm_walk_fault_for_test(
                &pattern,
                WalkFault::ClusterNodesReply {
                    read,
                    reply: String::new(),
                },
            )
        })
        .collect();

    let (found, reason, every_primary_walked) = coverage_walk(walk_client, &pattern).await;
    assert_eq!(found, off_b, "every key but B's: B's pin read A");
    match reason {
        Some(WalkError::Changed(change)) => match *change {
            ServingChange::StaleTable {
                ref target,
                at,
                ref refresh,
                ..
            } => {
                assert_eq!(&**target, pattern.as_str());
                assert_eq!(at, WalkEnd::Start);
                assert_eq!(refresh, &RoutingRefresh::Refreshed);
            }
            other => panic!("StaleTable at the start, got {other:?}"),
        },
        other => {
            panic!("a walk with unproven pins on a stale table ends StaleTable, got {other:?}")
        }
    }
    assert!(!every_primary_walked, "B was not walked");
    assert_eq!(
        table_node(walk_client, MOVED_SLOT),
        b,
        "the StaleTable re-sync refreshed the walk's client"
    );

    let (found, reason, every_primary_walked) = coverage_walk(walk_client, &pattern).await;
    drop(unusable);
    assert_eq!(found, written, "every key written, on every shard");
    match reason {
        Some(WalkError::WitnessMissing(missing)) => assert_eq!(
            missing.seen,
            WitnessSeen::MyselfLine(MyselfLineSeen::NoLine)
        ),
        other => panic!(
            "the second walk, on the refreshed table, answers only the missing myself line, got \
             {other:?}"
        ),
    }
    assert!(
        every_primary_walked,
        "the refreshed table matches the map, so the unproven pins count as walked"
    );

    move_slot(&admin, &b, &a).await;
    println!(
        "CLUSTER-GATED RAN: scan_guard_slot_migration_disruptive_a_coverage_walk_with_unproven_pins_on_a_stale_routing_table_re_syncs_it"
    );
}
