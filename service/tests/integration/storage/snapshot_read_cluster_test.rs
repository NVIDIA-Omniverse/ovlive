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

//! The atomic snapshot read and its startup self-test against a real
//! multi-shard Valkey Cluster (b.y3m Epic 2 Task 4; Spec SDD SR-1.1, SR-1.8,
//! SR-2.2).
//!
//! `object_snapshot_read.lua` declares only the object's header and reads the
//! object's other keys undeclared. They carry the same `{oid}` hash tag, so
//! they sit in the declared key's slot and the shard serving it must accept
//! the reads — on every shard, since objects spread across all of them. These
//! tests prove that per primary, and that the self-test (which probes exactly
//! that acceptance before readiness) passes on each.
//!
//! The structural `UpdateLayer` WATCH read's per-window script
//! (`fanout_window_read.lua`, b.gvf) relies on the same undeclared same-slot
//! reads: one test proves its windows on every shard the same way, and a
//! disruptive one (`#[ignore]`d, `test-fault-injection`) that during a slot
//! migration the read answers the object's state: it falls back to plain
//! commands once, when the migration refuses the script's undeclared key, and
//! fred's router follows each plain command's `-ASK` to the destination.
//!
//! Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` (`bash scripts/dev-valkey.sh
//! cluster-up` → `redis-cluster://127.0.0.1:7000`) and prints a `SKIP:` line
//! when it is unset. nextest records that skip as a pass, so each test also
//! prints `CLUSTER-GATED RAN: <test> shards=<n>` when it actually runs, and
//! every one is in the `ci` profile's success-output filter in
//! `.config/nextest.toml`: CI's release-gating `test-cluster:` job (the
//! disruptive one: `test-cluster-disruptive:`) stores their output in the
//! JUnit report, which is the proof they ran.
//!
//! Containment of the key set is asserted on a `MONITOR` record of the owning
//! primary, never on a refusal: the local cluster and AMR accept an undeclared
//! other-slot key that lands on the same node, so a refusal cannot prove it.

use std::collections::BTreeMap;
use std::sync::Arc;

use fred::interfaces::{ClientLike, ClusterInterface};
use fred::types::config::Server;

use live_edit::domain::types::{ObjectState, Section};
use live_edit::storage::diff_read::{run_snapshot_read_self_test, SNAPSHOT_SELF_TEST_OBJECT_ID};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use crate::storage::snapshot_read_support::{
    assert_atomic_equals_reference, monitor_fence, read_atomic, MonitorTap,
};

/// Upper bound on layers created while looking for one per primary. With
/// three primaries and object ids spread over the slot space, a few suffice;
/// the bound turns an impossible search into a failure instead of a hang.
const MAX_LAYERS_PER_SEARCH: usize = 200;

async fn cluster() -> Option<TestValkey> {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster:` job run this"
        );
        return None;
    };
    let tv = TestValkey::setup_with_url(url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (`redis-cluster://...`); \
         a single node smuggled in through this gate would report per-shard coverage the suite \
         does not have"
    );
    Some(tv)
}

fn primaries(tv: &TestValkey) -> Vec<Server> {
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes()
}

/// The primary serving `key`'s slot, from the routing table fred dispatches by.
fn owner_of(tv: &TestValkey, key: &str) -> Server {
    let slot = fred::util::redis_keyslot(key.as_bytes());
    tv.client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .get_server(slot)
        .unwrap_or_else(|| panic!("no primary serves slot {slot}"))
        .clone()
}

fn addr(server: &Server) -> String {
    format!("{}:{}", server.host, server.port)
}

fn state() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    for (id, parent) in [(1u64, 0u64), (2, 1)] {
        state.sections.insert(
            id,
            Section {
                section_id: id,
                parent_id: parent,
                name: format!("s{id}"),
                section_type: 1,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
    }
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);
    state
}

/// For every primary, an object whose `{oid}` slot it serves: the atomic read
/// returns exactly the reference read's state (so the shard accepted every undeclared
/// same-slot read), and the `MONITOR` record of that primary shows every key
/// the script touched carrying the object's tag.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn object_snapshot_read_answers_for_an_object_on_every_shard() {
    let Some(tv) = cluster().await else { return };
    let primaries = primaries(&tv);
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("sr-cluster", "/sr-cluster", "sr-cluster-user")
        .await
        .expect("session creation must succeed");

    let mut per_shard: BTreeMap<String, u64> = BTreeMap::new();
    for i in 0..MAX_LAYERS_PER_SEARCH {
        if per_shard.len() == primaries.len() {
            break;
        }
        let oid = ls
            .create_layer(
                &session_id,
                &format!("layer-{i}"),
                "sr-cluster-user",
                &state(),
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed");
        let owner = addr(&owner_of(
            &tv,
            &tv.key_schema().obj_header(&oid.to_string()),
        ));
        per_shard.entry(owner).or_insert(oid);
    }
    assert_eq!(
        per_shard.len(),
        primaries.len(),
        "the search must find an object on every primary within {MAX_LAYERS_PER_SEARCH} layers"
    );

    for (shard, oid) in &per_shard {
        let header_key = tv.key_schema().obj_header(&oid.to_string());
        let tag = format!("{}{{{oid}}}:obj:", tv.prefix());
        let mut tap = MonitorTap::start(shard).await;
        assert_atomic_equals_reference(
            tv.client(),
            tv.key_schema(),
            *oid,
            &format!("object on {shard}"),
        )
        .await;
        // The fence must land on the same primary as the read: give it the tag.
        let fence = monitor_fence(&format!("{}{{{oid}}}:", tv.prefix()));
        let _: i64 = fred::interfaces::KeysInterface::exists(&tv.client().pool, &fence)
            .await
            .expect("the fence EXISTS must succeed");
        let seen = tap.drain_until(&fence).await;
        let start = seen
            .iter()
            .rposition(|c| c.name() == "EVALSHA" && c.args.get(3) == Some(&header_key))
            .unwrap_or_else(|| {
                panic!("{shard}: the MONITOR record holds no EVALSHA for {header_key}")
            });
        let script: Vec<_> = seen[start + 1..]
            .iter()
            .take_while(|c| c.issued_by_script())
            .collect();
        assert!(
            !script.is_empty(),
            "{shard}: the script's reads must be in the record"
        );
        for c in script {
            let key = c.args.get(1).map(String::as_str).unwrap_or_default();
            assert!(
                key.starts_with(&tag),
                "{shard}: every key the script touches must carry the object's hash tag ({tag}…); \
                 it touched {key:?} with {c:?}"
            );
        }
        read_atomic(tv.client(), tv.key_schema(), *oid).await;
    }

    println!(
        "CLUSTER-GATED RAN: storage::snapshot_read_cluster_test::object_snapshot_read_answers_for_an_object_on_every_shard shards={}",
        per_shard.len()
    );
}

/// The startup self-test passes under the production key prefix — whose probe
/// slot is fixed by `SNAPSHOT_SELF_TEST_OBJECT_ID` — and, steered to each
/// primary in turn, on every shard. The steering puts a hash tag in the key
/// prefix: the header key then hashes on the prefix's tag, and the script
/// builds its probe from the same prefix, so the probe stays an undeclared
/// same-slot key on the chosen primary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_read_self_test_passes_on_every_shard() {
    let Some(tv) = cluster().await else { return };
    let deadline = std::time::Duration::from_secs(5);
    assert_eq!(
        run_snapshot_read_self_test(tv.client(), &KeySchema::default(), deadline).await,
        Ok(()),
        "the self-test must pass under the production prefix"
    );

    let primaries = primaries(&tv);
    let mut steered: BTreeMap<String, String> = BTreeMap::new();
    for i in 0.. {
        if steered.len() == primaries.len() || i == 10_000 {
            break;
        }
        let keys = KeySchema::new(&format!("test_selftest_{{shard{i}}}:"));
        let owner = addr(&owner_of(
            &tv,
            &keys.obj_header(SNAPSHOT_SELF_TEST_OBJECT_ID),
        ));
        steered
            .entry(owner)
            .or_insert_with(|| keys.prefix().to_string());
    }
    assert_eq!(
        steered.len(),
        primaries.len(),
        "a steering prefix must exist for every primary"
    );
    for (shard, prefix) in &steered {
        assert_eq!(
            run_snapshot_read_self_test(tv.client(), &KeySchema::new(prefix), deadline).await,
            Ok(()),
            "{shard} must accept the self-test's undeclared same-slot probe"
        );
    }

    println!(
        "CLUSTER-GATED RAN: storage::snapshot_read_cluster_test::snapshot_read_self_test_passes_on_every_shard shards={}",
        steered.len()
    );
}

// ── The structural WATCH read's window script (b.gvf, SDD §10.5) ────────────
//
// Each window of the structural read's rounds 2 and 3 is one `EVALSHA` of
// `fanout_window_read.lua` (but round 1, a lone children list over τ and every
// window after a script fallback, which are plain commands). Like the atomic
// read, the script declares only the object's header and reads the object's
// other keys undeclared, in the header's slot; on a cluster-state refusal of
// one (a slot migrating, not served, or the cluster down) the read falls back
// to plain commands, whose redirects fred's router follows per key: an `-ASK`
// with a pipelined `ASKING` to the migration's target, a `-MOVED` after a
// cluster sync.

/// Whether `command` is an `EVALSHA` of the window script whose one declared
/// key is `header_key`.
fn is_window_script_run(
    command: &crate::storage::snapshot_read_support::MonitoredCommand,
    header_key: &str,
) -> bool {
    command.name() == "EVALSHA"
        && command.args.get(1).is_some_and(|sha| {
            sha.eq_ignore_ascii_case(live_edit::storage::lua::FANOUT_WINDOW_READ_SCRIPT.sha1())
        })
        && command.args.get(3).map(String::as_str) == Some(header_key)
}

/// For every primary, an object whose `{oid}` slot it serves: the
/// structural read through its window scripts returns the atomic read's
/// compact projection on each shard (so the shard accepted the script's
/// undeclared same-slot reads), and the `MONITOR` record of that primary
/// shows at least one window script naming the object's header and every
/// key each window script touched carrying the object's tag.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_window_scripts_answer_for_an_object_on_every_shard() {
    use live_edit::storage::diff_read::{
        read_object_snapshot_slim, SlimSnapshotReadConsumer, SnapshotReadOf,
    };
    use live_edit::storage::test_support::compact_snapshot;

    let Some(tv) = cluster().await else { return };
    let primaries = primaries(&tv);
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("sr-cluster-window", "/sr-cluster-window", "sr-cluster-user")
        .await
        .expect("session creation must succeed");

    let mut per_shard: BTreeMap<String, u64> = BTreeMap::new();
    for i in 0..MAX_LAYERS_PER_SEARCH {
        if per_shard.len() == primaries.len() {
            break;
        }
        let oid = ls
            .create_layer(
                &session_id,
                &format!("layer-{i}"),
                "sr-cluster-user",
                &state(),
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed");
        let owner = addr(&owner_of(
            &tv,
            &tv.key_schema().obj_header(&oid.to_string()),
        ));
        per_shard.entry(owner).or_insert(oid);
    }
    assert_eq!(
        per_shard.len(),
        primaries.len(),
        "the search must find an object on every primary within {MAX_LAYERS_PER_SEARCH} layers"
    );

    for (shard, oid) in &per_shard {
        let header_key = tv.key_schema().obj_header(&oid.to_string());
        let tag = format!("{}{{{oid}}}:obj:", tv.prefix());
        let atomic = read_atomic(tv.client(), tv.key_schema(), *oid).await;
        let mut tap = MonitorTap::start(shard).await;
        let read = read_object_snapshot_slim(
            tv.client(),
            tv.key_schema(),
            &oid.to_string(),
            SlimSnapshotReadConsumer::StructuralUpdate,
        )
        .await;
        // The fence must land on the same primary as the read: give it the tag.
        let fence = monitor_fence(&format!("{}{{{oid}}}:", tv.prefix()));
        let _: i64 = fred::interfaces::KeysInterface::exists(&tv.client().pool, &fence)
            .await
            .expect("the fence EXISTS must succeed");
        let seen = tap.drain_until(&fence).await;
        match read {
            Ok(SnapshotReadOf::Snapshot(read)) => assert_eq!(
                read.state,
                compact_snapshot(&atomic.state),
                "{shard}: the structural read is the atomic read's compact projection"
            ),
            other => panic!(
                "{shard}: the structural read must answer with state; got {:?}",
                other.map(|_| ())
            ),
        }
        let runs: Vec<usize> = seen
            .iter()
            .enumerate()
            .filter(|(_, c)| is_window_script_run(c, &header_key))
            .map(|(i, _)| i)
            .collect();
        assert!(
            !runs.is_empty(),
            "{shard}: the MONITOR record holds the read's window scripts for {header_key}"
        );
        for run in runs {
            let script: Vec<_> = seen[run + 1..]
                .iter()
                .take_while(|c| c.issued_by_script())
                .collect();
            assert!(
                !script.is_empty(),
                "{shard}: each window script's reads must be in the record"
            );
            for c in script {
                let key = c.args.get(1).map(String::as_str).unwrap_or_default();
                assert!(
                    key.starts_with(&tag),
                    "{shard}: every key a window script touches must carry the object's hash \
                     tag ({tag}…); it touched {key:?} with {c:?}"
                );
            }
        }
    }

    println!(
        "CLUSTER-GATED RAN: storage::snapshot_read_cluster_test::structural_read_window_scripts_answer_for_an_object_on_every_shard shards={}",
        per_shard.len()
    );
}

/// Whether `command`, issued by a client (not a script), names `key` as its
/// first argument and is a `name` command.
#[cfg(feature = "test-fault-injection")]
fn is_client_command_on(
    command: &crate::storage::snapshot_read_support::MonitoredCommand,
    name: &str,
    key: &str,
) -> bool {
    !command.issued_by_script()
        && command.name() == name
        && command.args.get(1).map(String::as_str) == Some(key)
}

/// In a node's `MONITOR` record, the client command `name key` is there, and
/// the command right before it from the same client is `ASKING`: the client
/// followed an `-ASK` to this node with `ASKING` pipelined in front of the
/// command, on one connection (the vendored fred's `live_edit::ask`).
#[cfg(feature = "test-fault-injection")]
fn assert_asking_right_before(
    seen: &[crate::storage::snapshot_read_support::MonitoredCommand],
    name: &str,
    key: &str,
    node: &str,
) {
    let at = seen
        .iter()
        .position(|c| is_client_command_on(c, name, key))
        .unwrap_or_else(|| {
            panic!(
                "{node}: the MONITOR record holds no `{name} {key}` from a client: the -ASK for \
                 it was not followed to the destination"
            )
        });
    let client = &seen[at].client;
    let before = seen[..at].iter().rev().find(|c| &c.client == client);
    assert!(
        before.is_some_and(|c| c.name() == "ASKING" && c.args.len() == 1),
        "{node}: `{name} {key}` from {client} must come right behind an ASKING from the same \
         client; the command before it was {before:?}"
    );
}

/// The structural read during a live slot migration (b.gvf, b.vn4): the
/// object's slot is staged `MIGRATING` on its owner and `IMPORTING` on
/// another primary, and its section row `obj:sec:2` is moved to the
/// destination. The read answers the object's state as it was before the
/// migration — the atomic read's compact projection — and falls back to plain
/// commands exactly once (`live_edit_snapshot_read_window_fallback_total`).
///
/// How it gets there. Round 1's plain commands on keys the source does not
/// hold — `SCARD obj:dels`, a set no write has created — are answered `-ASK`,
/// which fred's pooled router follows: `ASKING` pipelined, then the command,
/// on the destination (the vendored fred's `live_edit::ask`, SDD §15.3). The
/// round-2 window script, running on the source, meets the moved row it reads
/// undeclared, a cluster-state refusal, and the read re-issues the window as
/// plain commands; their `HGETALL obj:sec:2` is answered `-ASK` and followed
/// the same way. The destination's `MONITOR` record pins both follows:
/// `ASKING` right before `SCARD obj:dels`, and right before the plain
/// `HGETALL obj:sec:2`, from the same client. (Before b.vn4 fred re-sent each
/// `-ASK`ed command to the source until its attempts ran out, and the read
/// failed in round 1 with no fallback.)
///
/// The `SNAPSHOT-READ-MIGRATION DIAGNOSTIC` line (kept on a pass by the
/// nextest success-output override) is printed before any assertion, with
/// the read's result and every `snapshot_read` metric, so a red run says what
/// the read did.
///
/// **Disruptive and twice gated.** It migrates a hash slot between shards,
/// so it is `#[ignore]`d in addition to the `LIVE_EDIT_TEST_CLUSTER_URL`
/// gate, and it needs `test-fault-injection`, the feature the shared
/// [`SlotMigration`](crate::storage::cluster_subscriber_test::SlotMigration)
/// helper is built under: without the feature it is compiled out, not
/// skipped. The migration is driven through that helper, whose `Drop` guard
/// leaves the slot in a consistent end state if the test panics part-way.
/// Run it with `--run-ignored all` and the feature; CI's
/// `test-cluster-disruptive:` job selects tests by name, so its filter must
/// name this test to run it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, migrates a hash slot between shards (needs \
            test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn snapshot_read_disruptive_structural_read_falls_back_to_plain_commands_during_a_slot_migration(
) {
    use crate::storage::cluster_subscriber_test::SlotMigration;
    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::storage::diff_read::{
        read_object_snapshot_slim, SlimSnapshotReadConsumer, SnapshotReadOf,
    };
    use live_edit::storage::test_support::compact_snapshot;

    let Some(tv) = cluster().await else { return };
    let metrics = Arc::new(crate::common::metrics_scrape_only("sr-cluster-migrate"));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let fallbacks = || {
        sample_value(
            &render(&metrics),
            "live_edit_snapshot_read_window_fallback_total",
            &[],
        )
        .expect("the fallback counter is seeded at zero")
    };
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            "sr-cluster-migrate",
            "/sr-cluster-migrate",
            "sr-cluster-user",
        )
        .await
        .expect("session creation must succeed");
    let oid = ls
        .create_layer(
            &session_id,
            "layer",
            "sr-cluster-user",
            &state(),
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    let oid_str = oid.to_string();
    let header_key = tv.key_schema().obj_header(&oid_str);
    let moved_key = tv.key_schema().obj_section(&oid_str, "2");
    let dels_key = tv.key_schema().obj_deleted_sections_index(&oid_str);
    let slot = fred::util::redis_keyslot(header_key.as_bytes());
    let src = owner_of(&tv, &header_key);
    let dst = primaries(&tv)
        .into_iter()
        .find(|s| *s != src)
        .expect("a multi-shard cluster has a primary to migrate the slot to");
    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;
    let before = fallbacks();
    // Every command the destination runs from here on, up to the fence below.
    let mut tap = MonitorTap::start(&addr(&dst)).await;

    let servers = format!("src={src:?} dst={dst:?}");
    let mut migration = SlotMigration::begin(&tv, slot, src, dst.clone()).await;
    migration.move_key(&moved_key).await;
    let read = read_object_snapshot_slim(
        tv.client(),
        tv.key_schema(),
        &oid_str,
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    let fell_back = fallbacks() - before;
    let snapshot_read_metrics: Vec<String> = render(&metrics)
        .lines()
        .filter(|l| l.contains("snapshot_read") && !l.starts_with('#'))
        .map(str::to_owned)
        .collect();
    migration.complete().await;

    let read_result = match &read {
        Ok(SnapshotReadOf::Snapshot(_)) => "Ok(Snapshot)".to_owned(),
        Ok(SnapshotReadOf::TooLarge(t)) => format!("Ok(TooLarge({t:?}))"),
        Err(e) => format!("Err({e:?})"),
    };
    eprintln!(
        "SNAPSHOT-READ-MIGRATION DIAGNOSTIC: fallbacks={fell_back} slot={slot} {servers} \
         moved_key={moved_key} read={read_result}\n  metrics:\n    {}",
        snapshot_read_metrics.join("\n    ")
    );

    // The slot is the destination's now: a fence with the object's tag runs
    // there, after everything the read sent it.
    let fence = monitor_fence(&format!("{}{{{oid}}}:", tv.prefix()));
    let _: i64 = fred::interfaces::KeysInterface::exists(&tv.client().pool, &fence)
        .await
        .expect("the fence EXISTS must succeed");
    let seen = tap.drain_until(&fence).await;

    match read {
        Ok(SnapshotReadOf::Snapshot(read)) => assert_eq!(
            read.state,
            compact_snapshot(&atomic.state),
            "the read during the migration answers the object's state as it was before it"
        ),
        other => panic!(
            "the read during the migration must answer with the object's state; got {:?}",
            other.map(|_| ())
        ),
    }
    assert_eq!(
        fell_back, 1.0,
        "the read fell back once, when its window script met the moved key; read={read_result}"
    );
    let node = addr(&dst);
    assert_asking_right_before(&seen, "SCARD", &dels_key, &node);
    assert_asking_right_before(&seen, "HGETALL", &moved_key, &node);

    println!(
        "CLUSTER-GATED RAN: storage::snapshot_read_cluster_test::snapshot_read_disruptive_structural_read_falls_back_to_plain_commands_during_a_slot_migration shards={}",
        primaries(&tv).len()
    );
}
