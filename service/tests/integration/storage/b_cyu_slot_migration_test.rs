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

//! b.cyu / b.qtk — writes during a real legacy slot migration
//! (`CLUSTER SETSLOT … MIGRATING / IMPORTING` plus key-by-key `MIGRATE`).
//!
//! Each cluster test stages a slot `MIGRATING` on its owner and `IMPORTING` on
//! another primary through the shared
//! [`SlotMigration`](crate::storage::cluster_subscriber_test::SlotMigration)
//! helper, runs one write against it, completes the migration, and checks the
//! write's answer and what it left. SDD §18.1 *Slot migrations* states the
//! rules they pin.
//!
//! | Test | What it pins | Without the b.cyu change |
//! |---|---|---|
//! | T-C1 [`delete_session_slot_migration_disruptive_answers_unavailable`] | `DeleteSession` before any key moved: `Aborted`, with the real `-TRYAGAIN` on Valkey 9.x or fred's "Max redirections exceeded" on ≤ 8.x (which checks each queued command, so the absent marker's `SET` meets `-ASK`); with every key moved: `Aborted` with fred's "Max redirections exceeded", the routing re-sync run, the session kept; after the move it deletes, re-sent past each stale txn-pool member's `MOVED` (`Aborted`) | `INTERNAL` once the keys moved |
//! | T-C2 [`txn_connection_resyncs_after_flip_slot_migration_disruptive`] | a one-connection txn pool whose member still routes a flipped slot to its old owner: the first `DeleteSession` meets `MOVED` (`Aborted`) and re-syncs the member; the second deletes | the second meets `MOVED` again, and both are `INTERNAL` |
//! | T-C3 [`create_layer_slot_migration_disruptive_answers_unavailable`] | `CreateLayer` whose new object's slot is migrating: it fails closed before its transaction, at the pre-`MULTI` keyspace type probe, a script the source answers `-ASK` and fred does not follow ("Max attempts reached."): plain `Unavailable`, no txn-pool re-sync, nothing materialized or mapped; after the move it creates | — (the same pre-`MULTI` refusal) |
//! | T-C5 [`create_layer_on_a_stale_txn_connection_slot_migration_disruptive_resyncs_it`] | a one-connection txn pool whose member still routes the new object's flipped slot to its old owner: the pre-`MULTI` guards (data pool, re-synced by the move's completion) pass, the materialization `EXEC` meets `MOVED`: plain `Unavailable` with "Max redirections exceeded", `exec_txn`'s re-sync run once, nothing materialized; a re-send creates, and a transaction on the moved slot through the member lands | `INTERNAL`, and no re-sync: the member keeps meeting `MOVED` |
//! | T-C4 [`commit_and_watch_keysets_target_tryagain_slot_migration_disruptive`] | a premise guard (may pass without the change): with the object's keys moved, a one-attempt commit is `Aborted` "Max attempts reached."; on the target, `ASKING` + the commit's ten shared keys and `ASKING` + the `WATCH` set are each refused `-TRYAGAIN`, so following `-ASK` could not land them; after the move the commit lands | — |
//! | T-Q1 [`b_qtk_structural_children_rebuild_slot_migration_disruptive_refuses_before_any_write`] | a structural commit rebuilding an existing children list, every key it names but its sentinel present: refused `-TRYAGAIN`, the keyspace byte-identical | runs on the source and is refused at the `RPUSH` after its `DEL`: the stream entry published, `seq` unchanged, the children list gone |
//! | T-Q2 [`b_qtk_fast_external_value_replace_slot_migration_disruptive_refuses_before_any_write`] | a fast-mode commit replacing a layer's only external value, every key present: the same | refused at the `HGET obj:ext <new>` after the `HDEL` that emptied the index |
//! | T-S1 [`gc_drop_foreign_claims_slot_migration_disruptive_refuses_before_any_write`] | Valkey's refusal, not the caller's wiring: with every other `{gc}` key present on the migrating source, the claim-drop script called with its sentinel in `KEYS` (supplied by the test) over two members is refused `-TRYAGAIN` before it starts, nothing changed | — (the test supplies the sentinel itself). Before b.cyu the production call sent none, and the script ran and tore: the second member's `ZREM pending:age` refused after the first emptied it, that member out of `inflight`, unsettled, its bytes held |
//! | T-S2 [`gc_ledger_audit_slot_migration_disruptive_refuses_before_any_write`] | the same for the ledger audit over two members | — (the same). Before b.cyu: a partial page |
//! | T-S3 [`ext_index_recount_slot_migration_disruptive_refuses_before_any_write`] | the same for the index recount of a lowering then a raise, the header unstamped | — (the same). Before b.cyu: the header stamped, `obj:ext` emptied, the raise lost |
//! | T-R1 [`ext_store_value_heal_slot_migration_disruptive_lands_whole`] | a verified re-store over a wrong-typed blob: lands whole (the value, `ref_count` +1) | `DEL` then a refused `SET`: the blob gone, no `+1` |
//! | T-R2 [`ext_read_hold_take_slot_migration_disruptive_lands_whole`] | a hold `take` over one expired field: lands whole | the prune empties the key, the write is refused: no hold |
//! | T-R3 [`snapshot_intent_register_slot_migration_disruptive_lands_whole`] | an intent register over one expired intent and a lingering fence: `REGISTERED` | the prune empties the intent key, the write is refused |
//!
//! T-Q1 and T-Q2 stage a state in which, before the change, the commit ran on
//! the migrating source and tore. Two standalone tests (not `#[ignore]`d,
//! ordinary suite) check that staging on a stable store: the commit lands;
//! the store's `MONITOR` feed shows its `EVALSHA` sending exactly the staged
//! keys plus its sentinel, so every key it names but the sentinel was present;
//! and the feed shows the script naming the key it emptied after its `XADD` —
//! the command a migrating source refuses.
//!
//! T-S1 to T-S3 call the script statics directly, with the sentinel put in
//! `KEYS` by the test, so they pin Valkey's refusal of a script that declares
//! its sentinel, not that the production callers (`drop_foreign_claims`,
//! `gc_audit_page`, `recount_ext_index`) declare it. That wiring is pinned by
//! `script_migration_guard_test`'s
//! `the_sentinel_builder_is_called_only_at_the_sentinel_scripts_keys_assembly_sites` and
//! `each_sentinel_scripts_evalsha_keys_carry_the_sentinel`, and by the
//! standalone refusal tests in `b_cyu_script_sentinel_test`
//! (`…_without_its_migration_sentinel_is_refused_and_writes_nothing`), where a
//! call without the sentinel is refused `ENCODING` / `RECOUNT_ARGS` / `MERGE_ARGS`, so a
//! caller that dropped it would fail on every store.
//!
//! **Disruptive and twice gated**, like the b.vn4 tests: `#[ignore]`d in
//! addition to the `LIVE_EDIT_TEST_CLUSTER_URL` gate, and compiled only with
//! `test-fault-injection`, the feature `SlotMigration` is built under (without
//! it this module is compiled out, not skipped). The helper's `Drop` guard
//! rolls the staging back, or completes it once a key moved, if a test panics.
//! CI's `test-cluster-disruptive:` job selects them by the
//! `slot_migration_disruptive` in their names, and `.config/nextest.toml`
//! keeps their output on a pass. The `{gc}` tests migrate the fixed slot of
//! the tag `gc`.
//!
//! # Wait budget
//!
//! Each write is awaited under [`SETTLE`] (10 s); setup commands are bounded
//! by fred's 5 s command timeout, and the `SlotMigration` primitives by that
//! helper's 20 s per command (its completion drain by 30 s). They approach
//! those budgets only on a cluster that stopped answering, where the first one
//! fails. Healthy, each test takes about a second.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    ClientLike, ClusterInterface, HashesInterface, KeysInterface, SetsInterface,
    SortedSetsInterface, TransactionInterface,
};
use fred::types::config::Server;
use fred::types::scripts::Script;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::storage::client::TRYAGAIN_REPLY;
use live_edit::storage::external_values::{
    ExternalValueStore, MockExternalValueStore, ValkeyExternalValueStore, EXT_READ_HOLD_SCRIPT,
    EXT_STORE_VALUE_SCRIPT, GC_CLAIM_SCORE_OFFSET_MS, GC_DECREMENT_FENCE_MS,
    GC_DROP_FOREIGN_CLAIMS_SCRIPT, GC_LEDGER_AUDIT_SCRIPT,
};
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::{update_script_shared_keys, UPDATE_LAYER_FAST_SCRIPT};
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::{register_intent, IntentVerdict, LayerSize};
use live_edit::storage::structural::{CommitOutcome, StructuralCommit};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_with_set_field_external, delta_with_set_field_inline};
use crate::storage::cluster_subscriber_test::{try_keys_in_slot, SlotMigration};
use crate::storage::snapshot_read_atomic_test::{move_cmd, structural, ROOT, SIBLING, WORLD};
use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};
use crate::storage::structural_commit_test::{fixture_on, Fixture, StructuralInvocation};

/// Upper bound on one write under test. A deadline, not a speed claim.
const SETTLE: Duration = Duration::from_secs(10);

const POOL_SIZE: u32 = 4;
const CREATED_BY: &str = "bcyu-migration-user";
const WRITER: &str = "bcyu-migration-writer";

/// What fred's transaction router fails a redirected transaction with.
const REDIRECTED: &str = "Max redirections exceeded";
/// What the vendored fred answers a script `-ASK` it does not follow with.
const SCRIPT_ASK_NOT_FOLLOWED: &str = "Max attempts reached.";

/// The recount script, from the source the service compiles in (its static is
/// crate-private).
static EXT_INDEX_RECOUNT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../../src/storage/scripts/ext_index_recount.lua"
    ))
});

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// A clustered `TestValkey` with `pool` connections per pool, or `None`
/// (logged) when `LIVE_EDIT_TEST_CLUSTER_URL` is unset.
async fn cluster(pool: u32) -> Option<TestValkey> {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster-disruptive:` job \
             run this"
        );
        return None;
    };
    let tv = TestValkey::setup_with_url(url, pool).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (`redis-cluster://...`)"
    );
    Some(tv)
}

/// Where `key`'s slot lives and where to migrate it.
struct Placement {
    slot: u16,
    src: Server,
    dst: Server,
}

fn placement(tv: &TestValkey, key: &str) -> Placement {
    let state = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view");
    let slot = fred::util::redis_keyslot(key.as_bytes());
    let src = state
        .get_server(slot)
        .unwrap_or_else(|| panic!("no primary serves slot {slot}"))
        .clone();
    let dst = state
        .unique_primary_nodes()
        .into_iter()
        .find(|s| *s != src)
        .expect("a multi-shard cluster has a primary to migrate the slot to");
    Placement { slot, src, dst }
}

/// Begin migrating `key`'s slot.
async fn migrate<'a>(tv: &'a TestValkey, key: &str) -> (SlotMigration<'a>, Placement) {
    let p = placement(tv, key);
    let migration = SlotMigration::begin(tv, p.slot, p.src.clone(), p.dst.clone()).await;
    (migration, p)
}

/// Move every key the source still holds in the slot.
async fn move_every_key(migration: &mut SlotMigration<'_>, tv: &TestValkey, p: &Placement) {
    for _ in 0..32 {
        let keys = try_keys_in_slot(tv, &p.src, p.slot)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        if keys.is_empty() {
            return;
        }
        for key in keys {
            migration.move_key(&key).await;
        }
    }
    panic!("slot {} did not drain from {}", p.slot, p.src);
}

/// `DUMP` of each key: type-agnostic, `None` for an absent key, so one
/// equality covers "not changed" and "not created".
async fn dump(tv: &TestValkey, keys: &[String]) -> Vec<(String, Option<Vec<u8>>)> {
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let dumped: Option<Vec<u8>> = tv
            .client()
            .pool
            .dump(key)
            .await
            .unwrap_or_else(|e| panic!("DUMP {key} must succeed: {e}"));
        out.push((key.clone(), dumped));
    }
    out
}

async fn exists(tv: &TestValkey, key: &str) -> bool {
    let n: u64 = tv
        .client()
        .pool
        .exists(key)
        .await
        .unwrap_or_else(|e| panic!("EXISTS {key} must succeed: {e}"));
    n > 0
}

/// The message of a transient answer: `Aborted` (a retry loop's signal) or
/// `Unavailable`, both `UNAVAILABLE` on the wire.
fn transient_message(e: &ServiceError) -> Option<&str> {
    match e {
        ServiceError::Aborted(msg) | ServiceError::Unavailable(msg) => Some(msg),
        _ => None,
    }
}

/// A script refused by the migrating source's pre-execution check: the real
/// `-TRYAGAIN` (some of its keys present, the sentinel absent).
fn assert_refused_tryagain(answer: &Result<impl std::fmt::Debug, fred::error::Error>, what: &str) {
    match answer {
        Err(e) => assert!(
            e.details().starts_with(TRYAGAIN_REPLY),
            "{what}: the migrating source must refuse the script before it starts with the real \
             -TRYAGAIN (its migration sentinel is absent); got {e:?}"
        ),
        Ok(reply) => panic!(
            "{what}: the script ran on a migrating slot (reply {reply:?}); its migration sentinel \
             must keep it from starting"
        ),
    }
}

fn ran(test: &str, tv: &TestValkey) {
    let shards = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .map_or(0, |s| s.unique_primary_nodes().len());
    println!("CLUSTER-GATED RAN: storage::b_cyu_slot_migration_test::{test} shards={shards}");
}

// ═════════════════════════════════════════════════════════════════════════════
// Transactions: DeleteSession, the routing re-sync, CreateLayer
// ═════════════════════════════════════════════════════════════════════════════

/// T-C1. `DeleteSession` on a session whose slot migrates. Before any key
/// moved, the transaction names present and absent keys (the deletion marker
/// is absent): on Valkey 9.x, which checks it at `EXEC`, the source refuses
/// the `EXEC` `-TRYAGAIN`; on Valkey ≤ 8.x, which checks each queued command,
/// the marker's queued `SET` is answered `-ASK`, which the txn pool does not
/// follow, so fred fails it "Max redirections exceeded". Either is `Aborted`.
/// With every key moved, the source answers the `EXEC` `-ASK`: fred fails
/// the transaction "Max redirections exceeded", the classifier grades it
/// transient (`Aborted`, `UNAVAILABLE` on the wire; it was `INTERNAL` before
/// b.cyu), and `exec_txn` re-syncs the member's routing (the test-only sync
/// counter advances). Nothing ran: the session is kept. After the move
/// completes it deletes — possibly only on a re-send, since every txn-pool
/// member still routes the slot to its old owner until its own `MOVED`
/// re-syncs it (T-C2).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn delete_session_slot_migration_disruptive_answers_unavailable() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let ks = tv.key_schema();
    let sessions = SessionStore::new(tv.client().clone(), ks.clone());
    let session_id = sessions
        .create("bcyu-tc1", "/bcyu-tc1", CREATED_BY)
        .await
        .expect("session creation must succeed");
    let meta_key = ks.session_meta(&session_id);

    let (mut migration, p) = migrate(&tv, &meta_key).await;
    let before_move = tokio::time::timeout(SETTLE, sessions.delete(&session_id))
        .await
        .expect("the delete did not settle within its budget");
    move_every_key(&mut migration, &tv, &p).await;
    let syncs_before = tv.client().txn_routing_syncs_for_test();
    let after_move = tokio::time::timeout(SETTLE, sessions.delete(&session_id))
        .await
        .expect("the delete did not settle within its budget");
    let syncs = tv.client().txn_routing_syncs_for_test() - syncs_before;
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&meta_key)
        .await
        .expect("HGETALL meta must succeed (the data pool follows -ASK)");
    migration.complete().await;

    eprintln!(
        "B.CYU T-C1 DATAPOINT: slot={} src={} dst={}\n  before any key moved: {before_move:?}\n  \
         every key moved: {after_move:?} (routing syncs {syncs})",
        p.slot, p.src, p.dst
    );
    // Before any key moved, the shape depends on the store's version: Valkey
    // 9.x checks the transaction at `EXEC` and refuses it `-TRYAGAIN`; Valkey
    // ≤ 8.x checks each queued command, so the absent deletion marker's `SET`
    // is answered `-ASK` at queue time, which the txn pool does not follow.
    // Either is `Aborted` and nothing ran.
    match &before_move {
        Err(ServiceError::Aborted(msg)) => assert!(
            msg.contains(TRYAGAIN_REPLY) || msg.contains(REDIRECTED),
            "before any key moved the transaction meets the real -TRYAGAIN (Valkey 9.x) or a \
             queued command's -ASK, failed {REDIRECTED:?} (Valkey ≤ 8.x); got {msg:?}"
        ),
        other => panic!("before any key moved the delete must answer Aborted; got {other:?}"),
    }
    match &after_move {
        Err(ServiceError::Aborted(msg)) => {
            assert!(
                msg.contains(REDIRECTED),
                "with every key moved the EXEC is redirected and fred fails it {REDIRECTED:?}; \
                 got {msg:?}"
            );
            assert_eq!(
                tonic::Status::from(ServiceError::Aborted(msg.clone())).code(),
                tonic::Code::Unavailable
            );
        }
        other => panic!(
            "with every key moved the delete must answer Aborted (UNAVAILABLE), never Internal; \
             got {other:?}"
        ),
    }
    assert!(
        syncs >= 1,
        "the redirected transaction re-syncs its txn-pool member's routing"
    );
    assert!(!meta.is_empty(), "nothing ran: the session is kept");

    // After the move completes, every txn-pool member still routes the slot
    // to its old owner: `SlotMigration::complete` re-syncs a data-pool client
    // only, and the re-sync above ran while the slot was still the source's.
    // So a delete may first meet the old owner's `MOVED` (`Aborted`, and its
    // member re-synced, as T-C2 pins); each re-sync fixes one member, so the
    // re-sent delete lands within one attempt per member.
    let mut attempts = Vec::new();
    loop {
        let attempt = tokio::time::timeout(SETTLE, sessions.delete(&session_id))
            .await
            .expect("the delete did not settle within its budget");
        match attempt {
            Ok(()) => break,
            Err(ServiceError::Aborted(msg)) if msg.contains(REDIRECTED) => {
                attempts.push(msg);
                assert!(
                    attempts.len() <= POOL_SIZE as usize,
                    "after the move completes, a re-sent delete lands once each stale txn-pool \
                     member has met its MOVED and re-synced; got {attempts:?}"
                );
            }
            Err(other) => panic!(
                "after the move completes, a delete answers OK, or Aborted from a stale txn-pool \
                 member's MOVED; got {other:?}"
            ),
        }
    }
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&meta_key)
        .await
        .expect("HGETALL meta must succeed");
    assert!(meta.is_empty(), "the session is deleted");
    ran(
        "delete_session_slot_migration_disruptive_answers_unavailable",
        &tv,
    );
}

/// T-C2. A one-connection txn pool. The session's slot moves to another
/// primary and the move completes, but the txn pool's one member — its own
/// fred client, with its own routing table — is never told: the completion
/// re-syncs a data-pool client only. `DeleteSession` #1's transaction goes to
/// the old owner, which answers `MOVED`; the txn pool does not follow it, so
/// the call answers `Aborted` and re-syncs the member (the test-only sync
/// counter +1). `DeleteSession` #2 then reaches the new owner and deletes.
/// Without the re-sync, #2 meets the same `MOVED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn txn_connection_resyncs_after_flip_slot_migration_disruptive() {
    let Some(tv) = cluster(1).await else {
        return;
    };
    let ks = tv.key_schema();
    let sessions = SessionStore::new(tv.client().clone(), ks.clone());
    let session_id = sessions
        .create("bcyu-tc2", "/bcyu-tc2", CREATED_BY)
        .await
        .expect("session creation must succeed");
    let meta_key = ks.session_meta(&session_id);

    let (mut migration, p) = migrate(&tv, &meta_key).await;
    move_every_key(&mut migration, &tv, &p).await;
    migration.complete().await;

    let syncs_before = tv.client().txn_routing_syncs_for_test();
    let first = tokio::time::timeout(SETTLE, sessions.delete(&session_id))
        .await
        .expect("delete #1 did not settle within its budget");
    let syncs = tv.client().txn_routing_syncs_for_test() - syncs_before;
    let second = tokio::time::timeout(SETTLE, sessions.delete(&session_id))
        .await
        .expect("delete #2 did not settle within its budget");

    eprintln!(
        "B.CYU T-C2 DATAPOINT: slot={} {} -> {}\n  delete #1: {first:?} (routing syncs {syncs})\n  \
         delete #2: {second:?}",
        p.slot, p.src, p.dst
    );
    match &first {
        Err(ServiceError::Aborted(msg)) => assert!(
            msg.contains(REDIRECTED),
            "delete #1 reaches the old owner, whose MOVED the txn pool does not follow; got \
             {msg:?}"
        ),
        other => panic!("delete #1 must answer Aborted (a stale member's MOVED); got {other:?}"),
    }
    assert_eq!(syncs, 1, "delete #1 re-syncs the member's routing once");
    second.expect("after the re-sync delete #2 reaches the new owner and deletes");
    ran(
        "txn_connection_resyncs_after_flip_slot_migration_disruptive",
        &tv,
    );
}

/// The fixture shared by T-C3 and T-C5: a session, a `LayerStore`, the empty
/// state a create materializes, and the next object id the counter will mint,
/// chosen (by creating seed layers) so its slot is neither the session's nor
/// the id counter's — so moving it moves only the new object's slot.
struct NextObject {
    session_id: String,
    store: LayerStore,
    state: ObjectState,
    next: u64,
}

async fn next_object_on_its_own_slot(tv: &TestValkey, label: &str) -> NextObject {
    let ks = tv.key_schema();
    let session_id = SessionStore::new(tv.client().clone(), ks.clone())
        .create(label, &format!("/{label}"), CREATED_BY)
        .await
        .expect("session creation must succeed");
    let store = LayerStore::new(tv.client().clone(), ks.clone(), 4096);
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    };
    let session_slot = fred::util::redis_keyslot(ks.session_meta(&session_id).as_bytes());
    let counter_slot = fred::util::redis_keyslot(ks.counter_object_id().as_bytes());
    for i in 0..8 {
        let created = store
            .create_layer(
                &session_id,
                &format!("{label}-seed-{i}"),
                CREATED_BY,
                &state,
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("a seed create must succeed");
        let next = created + 1;
        let slot = fred::util::redis_keyslot(ks.obj_header(&next.to_string()).as_bytes());
        if slot != session_slot && slot != counter_slot {
            return NextObject {
                session_id,
                store,
                state,
                next,
            };
        }
    }
    panic!("eight consecutive object ids shared the session's or the counter's slot");
}

/// Every key a create of object `oid` would write first — its header, its
/// stream and its reference index — absent: nothing was materialized.
async fn assert_nothing_materialized(tv: &TestValkey, oid: u64) {
    let ks = tv.key_schema();
    let o = oid.to_string();
    for key in [
        ks.obj_header(&o),
        ks.layer_stream(oid),
        ks.obj_ext_index(&o),
    ] {
        assert!(
            !exists(tv, &key).await,
            "nothing was materialized, yet {key} exists"
        );
    }
}

/// T-C3. `CreateLayer` whose next object id's slot is migrating, nothing of
/// it moved yet. The create fails closed before its transaction opens:
/// `write_object`'s pre-`MULTI` guards run on the data pool, where the
/// `EXISTS` of the stream and the `HGETALL` of the reference index follow
/// the source's `-ASK` to the target, but the keyspace type probe is a script
/// (`EVALSHA keyspace_type_probe`) naming keys absent on the source, which
/// the source answers `-ASK` and fred never follows for a script: "Max
/// attempts reached.", answered as a plain `Unavailable`. No `MULTI` is
/// sent, so no txn-pool routing is re-synced (the test-only sync counter
/// stays put), and nothing is materialized or mapped. After the move
/// completes the create succeeds (under the next id, which the counter mints
/// afresh). The redirected materialization `EXEC` and its re-sync are T-C5's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn create_layer_slot_migration_disruptive_answers_unavailable() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let ks = tv.key_schema();
    let NextObject {
        session_id,
        store,
        state,
        next,
    } = next_object_on_its_own_slot(&tv, "bcyu-tc3").await;
    let header_key = ks.obj_header(&next.to_string());

    let (mut migration, p) = migrate(&tv, &header_key).await;
    let syncs_before = tv.client().txn_routing_syncs_for_test();
    let answer = tokio::time::timeout(
        SETTLE,
        store.create_layer(
            &session_id,
            "bcyu-tc3-layer",
            CREATED_BY,
            &state,
            Arc::new(MockExternalValueStore::new()),
        ),
    )
    .await
    .expect("the create did not settle within its budget");
    let syncs = tv.client().txn_routing_syncs_for_test() - syncs_before;
    migration.complete().await;

    eprintln!(
        "B.CYU T-C3 DATAPOINT: object {next} slot={} {} -> {}\n  CreateLayer: {answer:?} \
         (routing syncs {syncs})",
        p.slot, p.src, p.dst
    );
    match &answer {
        Err(ServiceError::Unavailable(msg)) => assert!(
            msg.contains("keyspace_type_probe") && msg.contains(SCRIPT_ASK_NOT_FOLLOWED),
            "the pre-MULTI keyspace type probe meets the source's -ASK, which fred does not \
             follow for a script ({SCRIPT_ASK_NOT_FOLLOWED:?}); got {msg:?}"
        ),
        other => panic!(
            "a create whose new object's slot migrates must answer a plain Unavailable, never \
             Internal; got {other:?}"
        ),
    }
    assert_eq!(
        syncs, 0,
        "the create failed before its transaction opened: no txn-pool routing was re-synced"
    );
    let mapping: Option<String> = tv
        .client()
        .pool
        .get(ks.layer_mapping(&session_id, "bcyu-tc3-layer"))
        .await
        .expect("GET mapping must succeed");
    assert_eq!(mapping, None, "no layer was made");
    assert_nothing_materialized(&tv, next).await;

    let created = tokio::time::timeout(
        SETTLE,
        store.create_layer(
            &session_id,
            "bcyu-tc3-layer",
            CREATED_BY,
            &state,
            Arc::new(MockExternalValueStore::new()),
        ),
    )
    .await
    .expect("the create did not settle within its budget")
    .expect("after the move completes the create succeeds");
    let mapping: Option<String> = tv
        .client()
        .pool
        .get(ks.layer_mapping(&session_id, "bcyu-tc3-layer"))
        .await
        .expect("GET mapping must succeed");
    assert_eq!(mapping, Some(created.to_string()));
    ran(
        "create_layer_slot_migration_disruptive_answers_unavailable",
        &tv,
    );
}

/// T-C5. `CreateLayer` on a one-connection txn pool whose member still routes
/// the next object id's slot to its old owner. The slot's move completes
/// first, and `SlotMigration::complete` re-syncs the (one-connection) data
/// pool only, so `write_object`'s pre-`MULTI` guards — on the data pool —
/// reach the new owner and pass. Its transaction, on the stale txn member
/// (whose checkout probe, `DISCARD` / `UNWATCH`, is keyless and so is served
/// by the old owner), goes to the old owner, which answers `MOVED`; the
/// `txn_pool` follows no redirect, so fred fails it "Max redirections
/// exceeded", and `exec_txn` re-syncs the member (the test-only sync counter
/// +1) before the call answers a plain `Unavailable`. Nothing is
/// materialized or mapped. A re-send of the create then succeeds (under the
/// next id), and a transaction on the moved slot through the same member
/// now lands — the re-sync took. Without the re-sync the counter stays put
/// and that transaction meets the same `MOVED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn create_layer_on_a_stale_txn_connection_slot_migration_disruptive_resyncs_it() {
    let Some(tv) = cluster(1).await else {
        return;
    };
    let ks = tv.key_schema();
    let NextObject {
        session_id,
        store,
        state,
        next,
    } = next_object_on_its_own_slot(&tv, "bcyu-tc5").await;
    let header_key = ks.obj_header(&next.to_string());

    let (mut migration, p) = migrate(&tv, &header_key).await;
    move_every_key(&mut migration, &tv, &p).await;
    migration.complete().await;

    let syncs_before = tv.client().txn_routing_syncs_for_test();
    let answer = tokio::time::timeout(
        SETTLE,
        store.create_layer(
            &session_id,
            "bcyu-tc5-layer",
            CREATED_BY,
            &state,
            Arc::new(MockExternalValueStore::new()),
        ),
    )
    .await
    .expect("the create did not settle within its budget");
    let syncs = tv.client().txn_routing_syncs_for_test() - syncs_before;

    eprintln!(
        "B.CYU T-C5 DATAPOINT: object {next} slot={} {} -> {}\n  CreateLayer: {answer:?} \
         (routing syncs {syncs})",
        p.slot, p.src, p.dst
    );
    match &answer {
        Err(ServiceError::Unavailable(msg)) => assert!(
            msg.contains(REDIRECTED),
            "the stale member's materialization EXEC meets the old owner's MOVED, which the \
             txn pool does not follow; got {msg:?}"
        ),
        other => panic!(
            "a create whose transaction is redirected must answer a plain Unavailable, never \
             Internal; got {other:?}"
        ),
    }
    assert_eq!(
        tonic::Status::from(answer.unwrap_err()).code(),
        tonic::Code::Unavailable
    );
    assert_eq!(
        syncs, 1,
        "`exec_txn` re-syncs the redirected txn-pool member's routing once"
    );
    let mapping: Option<String> = tv
        .client()
        .pool
        .get(ks.layer_mapping(&session_id, "bcyu-tc5-layer"))
        .await
        .expect("GET mapping must succeed");
    assert_eq!(mapping, None, "no layer was made");
    assert_nothing_materialized(&tv, next).await;

    let created = tokio::time::timeout(
        SETTLE,
        store.create_layer(
            &session_id,
            "bcyu-tc5-layer",
            CREATED_BY,
            &state,
            Arc::new(MockExternalValueStore::new()),
        ),
    )
    .await
    .expect("the re-sent create did not settle within its budget")
    .expect("the re-sent create succeeds");
    let mapping: Option<String> = tv
        .client()
        .pool
        .get(ks.layer_mapping(&session_id, "bcyu-tc5-layer"))
        .await
        .expect("GET mapping must succeed");
    assert_eq!(mapping, Some(created.to_string()));

    // The re-sync took: a transaction on the moved slot, through the same
    // (only) txn member, reaches the new owner. The key shares the moved
    // object's hash tag.
    let probe_key = format!("{header_key}:bcyu-tc5-resync-probe");
    let checkout = tv.client().acquire_txn_for(&probe_key).await;
    let trx = checkout.multi();
    let _: () = trx.hset(&probe_key, ("f", "v")).await.expect("HSET queues");
    let landed = tokio::time::timeout(SETTLE, trx.exec::<fred::types::Value>(true))
        .await
        .expect("the transaction did not settle within its budget");
    drop(checkout);
    assert!(
        landed.is_ok(),
        "after the re-sync the member routes the moved slot to its new owner; got {landed:?}"
    );
    ran(
        "create_layer_on_a_stale_txn_connection_slot_migration_disruptive_resyncs_it",
        &tv,
    );
}

/// Send `commands` on one raw connection to `node`, in order, and read the
/// first line of each reply (each is a status or an error).
async fn raw_replies(node: &Server, commands: &[Vec<String>]) -> Vec<String> {
    let stream = tokio::net::TcpStream::connect(format!("{}:{}", node.host, node.port))
        .await
        .unwrap_or_else(|e| panic!("a raw connection to {node} must open: {e}"));
    let (read, mut write) = stream.into_split();
    let mut reader = BufReader::new(read);
    let mut replies = Vec::new();
    for command in commands {
        let mut frame = format!("*{}\r\n", command.len());
        for arg in command {
            frame.push_str(&format!("${}\r\n{arg}\r\n", arg.len()));
        }
        write
            .write_all(frame.as_bytes())
            .await
            .expect("the raw command must be sent");
        let mut line = String::new();
        tokio::time::timeout(SETTLE, reader.read_line(&mut line))
            .await
            .expect("the raw reply must arrive within the budget")
            .expect("the raw reply must be readable");
        replies.push(line.trim_end().to_owned());
    }
    replies
}

/// T-C4, a premise guard (it may pass without the b.cyu change): following
/// `-ASK` could not land a commit. With every key of the object moved, a
/// one-attempt commit meets the source's `-ASK`, which the vendored fred does
/// not follow for a script: `Aborted` "Max attempts reached.". On a raw
/// connection to the target, `ASKING` then the commit `EVALSHA` with its ten
/// shared keys (the migration sentinel included), and `ASKING` then the
/// `WATCH` of the header, the stream and the fence, are each refused
/// `-TRYAGAIN`: they name keys absent on the target. After the move the
/// commit lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn commit_and_watch_keysets_target_tryagain_slot_migration_disruptive() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let ks = tv.key_schema();
    let session_id = SessionStore::new(tv.client().clone(), ks.clone())
        .create("bcyu-tc4", "/bcyu-tc4", CREATED_BY)
        .await
        .expect("session creation must succeed");
    let oid = LayerStore::new(tv.client().clone(), ks.clone(), 4096)
        .create_layer(
            &session_id,
            "bcyu-tc4-layer",
            CREATED_BY,
            &ObjectState {
                sequence_num: 1,
                first_version: 1,
                source_format: "usda".to_owned(),
                ..Default::default()
            },
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    let o = oid.to_string();
    let header_key = ks.obj_header(&o);
    let shared_keys = update_script_shared_keys(ks, oid, 0);
    assert_eq!(
        shared_keys.last(),
        Some(&ks.migration_sentinel(&o)),
        "the commit's ten shared keys end with its sentinel"
    );
    let one_attempt = LayerStore::new(tv.client().clone(), ks.clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
        .with_update_layer_retry_max_attempts(1);

    let (mut migration, p) = migrate(&tv, &header_key).await;
    move_every_key(&mut migration, &tv, &p).await;
    let commit = tokio::time::timeout(
        SETTLE,
        one_attempt.update_layer_relayed(
            &session_id,
            "bcyu-tc4-layer",
            oid,
            Some(delta_with_set_field_inline(1, "bcyu-field", vec![7], 0)),
            WRITER,
            1_000,
        ),
    )
    .await
    .expect("the commit did not settle within its budget");
    let mut evalsha = vec![
        "EVALSHA".to_owned(),
        UPDATE_LAYER_FAST_SCRIPT.sha1().to_string(),
        shared_keys.len().to_string(),
    ];
    evalsha.extend(shared_keys.iter().cloned());
    let target = raw_replies(
        &p.dst,
        &[
            vec!["ASKING".to_owned()],
            evalsha,
            vec!["ASKING".to_owned()],
            vec![
                "WATCH".to_owned(),
                header_key.clone(),
                ks.layer_stream(oid),
                ks.obj_snapshot_fence(&o),
            ],
        ],
    )
    .await;
    migration.complete().await;

    eprintln!(
        "B.CYU T-C4 DATAPOINT: slot={} {} -> {}\n  one-attempt commit, keys moved: {commit:?}\n  \
         target: {target:?}",
        p.slot, p.src, p.dst
    );
    match &commit {
        Err(ServiceError::Aborted(msg)) => {
            assert!(
                msg.contains(SCRIPT_ASK_NOT_FOLLOWED),
                "the source answers the commit -ASK, which fred does not follow for a script; \
                 got {msg:?}"
            );
            assert_eq!(
                tonic::Status::from(ServiceError::Aborted(msg.clone())).code(),
                tonic::Code::Unavailable
            );
        }
        other => panic!("a one-attempt commit with its keys moved must end Aborted; got {other:?}"),
    }
    assert_eq!(target[0], "+OK", "ASKING is accepted");
    assert!(
        target[1].starts_with(&format!("-{TRYAGAIN_REPLY}")),
        "the target refuses the commit's key set under ASKING -TRYAGAIN; got {:?}",
        target[1]
    );
    assert_eq!(target[2], "+OK", "ASKING is accepted");
    assert!(
        target[3].starts_with(&format!("-{TRYAGAIN_REPLY}")),
        "the target refuses the WATCH set under ASKING -TRYAGAIN; got {:?}",
        target[3]
    );
    let landed = tokio::time::timeout(
        SETTLE,
        LayerStore::new(tv.client().clone(), ks.clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .update_layer_relayed(
                &session_id,
                "bcyu-tc4-layer",
                oid,
                Some(delta_with_set_field_inline(1, "bcyu-field", vec![8], 0)),
                WRITER,
                2_000,
            ),
    )
    .await
    .expect("the commit did not settle within its budget");
    assert_eq!(
        landed.map(|(seq, _)| seq).map_err(|e| format!("{e:?}")),
        Ok(2),
        "after the move the commit lands"
    );
    ran(
        "commit_and_watch_keysets_target_tryagain_slot_migration_disruptive",
        &tv,
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The commit script (b.qtk): staging shared with the standalone premise checks
// ═════════════════════════════════════════════════════════════════════════════

/// Plant a type-right placeholder in each of `keys` that does not exist: the
/// keys a commit names that are normally absent, so that every key it names
/// but its sentinel is present and, without the sentinel, the commit would
/// run on a migrating source. Panics on an absent key of any other family.
///
/// The placeholders change no outcome: a fence holding only an expired lease
/// and its size refuses nothing (the script's step 1d), a live writer intent
/// is only `TYPE`-probed by the script, and another writer's commit record,
/// a ring field the script does not read and a tombstone of a section the
/// commit does not touch are ignored. The standalone premise checks prove it:
/// the staged commit lands.
async fn plant_placeholders(tv: &TestValkey, keys: &[String]) {
    let pool = &tv.client().pool;
    for key in keys {
        if exists(tv, key).await {
            continue;
        }
        if key.ends_with(":obj:snapfence") {
            let _: () = pool
                .hset(
                    key,
                    [
                        ("r:bcyu-expired", "1"),
                        ("size_e", "10"),
                        ("size_b", "1000"),
                    ],
                )
                .await
                .expect("plant the lingering fence");
        } else if key.ends_with(":obj:snapintent") {
            let _: () = pool
                .hset(key, ("w:bcyu-live", "99999999999999"))
                .await
                .expect("plant the live intent");
        } else if key.ends_with(":obj:wtok") {
            let _: () = pool
                .hset(key, ("bcyu-other-writer", "bcyu"))
                .await
                .expect("plant another writer's record");
        } else if key.ends_with(":obj:wrel") {
            let _: () = pool
                .hset(key, ("bcyu-unread", "bcyu"))
                .await
                .expect("plant an unread ring field");
        } else if key.ends_with(":obj:dels") {
            let _: () = pool
                .sadd(key, "999999")
                .await
                .expect("plant an untouched tombstone");
        } else {
            panic!("fixture: {key} is absent and has no placeholder family here");
        }
    }
}

/// The staged T-Q1 commit: a structural move of `SIBLING` from `ROOT` to
/// `WORLD` on an `every_shape_state` object — a `CHL` record rebuilding
/// `ROOT`'s existing children list (`DEL`, then `RPUSH`) — planned, then
/// every key it names but its sentinel made present.
struct ChildrenRebuild {
    fx: Fixture,
    commit: StructuralCommit,
    /// Every key the commit names but its sentinel, the stream included.
    keys: Vec<String>,
    /// `ROOT`'s children list, which the commit rebuilds.
    children: String,
}

async fn stage_children_rebuild(tv: TestValkey) -> ChildrenRebuild {
    let fx = fixture_on(tv, "bcyu-q1").await;
    let planned = fx
        .plan(
            structural(vec![move_cmd(SIBLING, ROOT, WORLD, "Sibling")]),
            "0123456789abcdef0123456789abcd01",
        )
        .await;
    let invocation = StructuralInvocation::from_commit(&fx.tv, &planned.commit, 4096);
    let sentinel = fx.keys().migration_sentinel(&fx.oid_str());
    let keys: Vec<String> = invocation
        .keys
        .iter()
        .filter(|k| **k != sentinel)
        .cloned()
        .collect();
    assert_eq!(
        keys.len() + 1,
        invocation.keys.len(),
        "the commit names its sentinel once"
    );
    let children = fx.children_key(ROOT);
    assert!(
        keys.contains(&children),
        "fixture: the commit names ROOT's children list"
    );
    plant_placeholders(&fx.tv, &keys).await;
    ChildrenRebuild {
        commit: planned.commit,
        keys,
        children,
        fx,
    }
}

/// The staged T-Q2 commit: a layer whose only external value (hash A, on one
/// field) is replaced by another (hash B) in a fast-mode commit, every key it
/// names but its sentinel made present.
struct ExternalValueReplace {
    tv: TestValkey,
    session_id: String,
    oid: u64,
    /// Every key the commit names but its sentinel, the stream included.
    keys: Vec<String>,
    /// The object's reference index.
    ext_key: String,
}

const Q2_LAYER: &str = "bcyu-q2";
const Q2_SECTION: u64 = 1;
const Q2_FIELD: &str = "mesh";
const Q2_EXT_SIZE: u64 = 1_024;

fn q2_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// Q2's replacing delta: the field takes hash B.
fn q2_replace_delta() -> oodf_ffi::OwnedDelta {
    delta_with_set_field_external(
        Q2_SECTION,
        Q2_FIELD,
        q2_hash(0xB2).to_vec(),
        Q2_EXT_SIZE,
        1,
        1,
    )
}

async fn stage_external_value_replace(tv: TestValkey) -> ExternalValueReplace {
    let ks = tv.key_schema().clone();
    let ext = Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        ks.clone(),
    ));
    let session_id = SessionStore::new(tv.client().clone(), ks.clone())
        .create(Q2_LAYER, &format!("/{Q2_LAYER}"), CREATED_BY)
        .await
        .expect("session creation must succeed");
    let ls = LayerStore::new(tv.client().clone(), ks.clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);
    let oid = ls
        .create_layer(
            &session_id,
            Q2_LAYER,
            CREATED_BY,
            &ObjectState {
                sequence_num: 1,
                first_version: 1,
                source_format: "usda".to_owned(),
                ..Default::default()
            },
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");
    // The layer's only external value, A, and an inline time sample (so the
    // time-sample index exists).
    let mut first = delta_with_set_field_external(
        Q2_SECTION,
        Q2_FIELD,
        q2_hash(0xA1).to_vec(),
        Q2_EXT_SIZE,
        1,
        0,
    );
    first.time_samples =
        crate::common::owned_delta::delta_with_time_sample_inline(Q2_SECTION, 0.5, vec![1])
            .time_samples;
    for hash in [q2_hash(0xA1), q2_hash(0xB2)] {
        ext.store_values(&[(&hash, b"bcyu-external-payload".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
    }
    ls.update_layer_relayed(&session_id, Q2_LAYER, oid, Some(first), WRITER, 1)
        .await
        .expect("the first external commit must land");
    let o = oid.to_string();
    let ext_key = ks.obj_ext_index(&o);
    let index: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&ext_key)
        .await
        .expect("HGETALL obj:ext");
    assert_eq!(
        index.keys().cloned().collect::<Vec<_>>(),
        vec![sha256_to_hex(&q2_hash(0xA1))],
        "fixture: A is the layer's only external value"
    );
    let mut keys = update_script_shared_keys(&ks, oid, 1);
    keys.pop();
    keys.push(ks.obj_field(&o, &Q2_SECTION.to_string(), Q2_FIELD));
    plant_placeholders(&tv, &keys).await;
    ExternalValueReplace {
        tv,
        session_id,
        oid,
        keys,
        ext_key,
    }
}

/// The script commands on `key` in the `MONITOR` feed `seen`, as
/// `(index, upper-cased command, args)`.
fn script_commands_on(
    seen: &[crate::storage::snapshot_read_support::MonitoredCommand],
    key: &str,
) -> Vec<(usize, String, Vec<String>)> {
    seen.iter()
        .enumerate()
        .filter(|(_, c)| c.issued_by_script() && c.args.get(1).map(String::as_str) == Some(key))
        .map(|(i, c)| (i, c.name(), c.args.clone()))
        .collect()
}

/// The premise that makes T-Q1 / T-Q2 tear before the change: the staging
/// planted every key the commit actually sends but its sentinel. A key the
/// staging missed would be absent on the migrating source, which would then
/// refuse the old commit `-TRYAGAIN` before it started, so the cluster test
/// would pass without the fix. Read off the `MONITOR` feed `seen`: every
/// update-script `EVALSHA` whose `KEYS[1]` is `header` sends exactly
/// `staged_keys` with `sentinel` in its slot, `KEYS[10]`, and there is at
/// least one.
fn assert_the_commit_sends_the_staged_keys_and_its_sentinel(
    seen: &[crate::storage::snapshot_read_support::MonitoredCommand],
    header: &str,
    staged_keys: &[String],
    sentinel: String,
) {
    use live_edit::storage::lua::{UPDATE_SCRIPT_KEY_HEADER, UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL};

    let mut expected = staged_keys.to_vec();
    expected.insert(UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL - 1, sentinel);
    // `args`: the command, the SHA, the key count, then the keys.
    let sent: Vec<Vec<String>> = seen
        .iter()
        .filter(|c| {
            c.name() == "EVALSHA"
                && c.args.get(1).map(String::as_str) == Some(UPDATE_LAYER_FAST_SCRIPT.sha1())
                && c.args.get(2 + UPDATE_SCRIPT_KEY_HEADER).map(String::as_str) == Some(header)
        })
        .map(|c| {
            let numkeys: usize = c.args[2].parse().expect("EVALSHA's key count");
            c.args[3..3 + numkeys].to_vec()
        })
        .collect();
    assert!(
        !sent.is_empty(),
        "premise: the feed shows the commit's EVALSHA on {header}"
    );
    for keys in sent {
        assert_eq!(
            keys, expected,
            "premise: the commit sends exactly the staged keys plus its sentinel at \
             KEYS[{UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL}]; a key the staging did not plant would \
             be absent on the migrating source, which would refuse the old commit before it \
             started"
        );
    }
}

/// The `MONITOR` feed of `tv`'s node from `tap`'s start up to a fence issued
/// now.
async fn drain(
    tv: &TestValkey,
    tap: &mut MonitorTap,
) -> Vec<crate::storage::snapshot_read_support::MonitoredCommand> {
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("EXISTS fence must succeed");
    tap.drain_until(&fence).await
}

/// The T-Q1 staging premise, on a stable standalone store: with every key
/// the commit names but its sentinel present, the commit lands — so before
/// b.qtk, on a migrating source, the script would have run — and the
/// `MONITOR` feed shows the commit's `EVALSHA` sending exactly the staged keys
/// plus its sentinel (so no key it names was left unplanted), then the script
/// `XADD`ing its stream entry, then `DEL`eting `ROOT`'s children list, then
/// naming the emptied list in an `RPUSH`: the command a migrating source
/// refuses, after the entry is published. The landed structural-mode commit
/// also leaves its sentinel absent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn b_qtk_children_rebuild_staging_runs_and_names_its_emptied_list_after_its_xadd() {
    let staged = stage_children_rebuild(TestValkey::setup().await).await;
    for key in &staged.keys {
        assert!(
            exists(&staged.fx.tv, key).await,
            "premise: {key} exists before the commit"
        );
    }
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let (result, _) = staged.fx.commit(&staged.commit).await;
    let seen = drain(&staged.fx.tv, &mut tap).await;

    assert!(
        matches!(result, Ok(CommitOutcome::Committed { .. })),
        "premise: the staged commit runs and lands on a stable slot; got {result:?}"
    );
    let sentinel = staged.fx.keys().migration_sentinel(&staged.fx.oid_str());
    assert_the_commit_sends_the_staged_keys_and_its_sentinel(
        &seen,
        &staged.fx.keys().obj_header(&staged.fx.oid_str()),
        &staged.keys,
        sentinel.clone(),
    );
    assert!(
        !exists(&staged.fx.tv, &sentinel).await,
        "the landed structural-mode commit leaves its migration sentinel absent"
    );
    let stream = staged.fx.keys().layer_stream(staged.fx.oid);
    let xadd = script_commands_on(&seen, &stream)
        .into_iter()
        .find(|(_, c, _)| c == "XADD")
        .map(|(i, ..)| i)
        .expect("premise: the commit XADDs its entry");
    let on_list = script_commands_on(&seen, &staged.children);
    let del = on_list
        .iter()
        .find(|(_, c, _)| c == "DEL")
        .map(|(i, ..)| *i)
        .expect("premise: the rebuild DELs the list");
    let rpush = on_list
        .iter()
        .find(|(i, c, _)| c == "RPUSH" && *i > del)
        .map(|(i, ..)| *i)
        .expect("premise: the rebuild RPUSHes the list after its DEL");
    assert!(
        xadd < del && del < rpush,
        "premise: XADD ({xadd}) < DEL ({del}) < RPUSH ({rpush}) of the emptied list"
    );
}

/// The T-Q2 staging premise, on a stable standalone store: the commit
/// replacing the layer's only external value lands, and the `MONITOR` feed
/// shows its `EVALSHA` sending exactly the staged keys plus its sentinel, then
/// the script `XADD`ing its entry, then `HDEL`ing A from the reference index
/// (emptying it), then naming the emptied index again for B — the command a
/// migrating source refuses, after the entry is published.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn b_qtk_external_value_replace_staging_runs_and_names_its_emptied_index_after_its_xadd() {
    let staged = stage_external_value_replace(TestValkey::setup().await).await;
    for key in &staged.keys {
        assert!(
            exists(&staged.tv, key).await,
            "premise: {key} exists before the commit"
        );
    }
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let committed = LayerStore::new(
        staged.tv.client().clone(),
        staged.tv.key_schema().clone(),
        4096,
    )
    .with_update_layer_backend(UpdateLayerBackend::Lua)
    .update_layer_relayed(
        &staged.session_id,
        Q2_LAYER,
        staged.oid,
        Some(q2_replace_delta()),
        WRITER,
        2,
    )
    .await;
    let seen = drain(&staged.tv, &mut tap).await;

    assert!(
        committed.is_ok(),
        "premise: the staged commit runs and lands on a stable slot; got {committed:?}"
    );
    let o = staged.oid.to_string();
    assert_the_commit_sends_the_staged_keys_and_its_sentinel(
        &seen,
        &staged.tv.key_schema().obj_header(&o),
        &staged.keys,
        staged.tv.key_schema().migration_sentinel(&o),
    );
    let stream = staged.tv.key_schema().layer_stream(staged.oid);
    let xadd = script_commands_on(&seen, &stream)
        .into_iter()
        .find(|(_, c, _)| c == "XADD")
        .map(|(i, ..)| i)
        .expect("premise: the commit XADDs its entry");
    let (a, b) = (sha256_to_hex(&q2_hash(0xA1)), sha256_to_hex(&q2_hash(0xB2)));
    let on_index = script_commands_on(&seen, &staged.ext_key);
    let emptied = on_index
        .iter()
        .find(|(_, c, args)| c == "HDEL" && args.get(2) == Some(&a))
        .map(|(i, ..)| *i)
        .expect("premise: the settle HDELs A from the index");
    let named = on_index
        .iter()
        .find(|(i, _, args)| *i > emptied && args.get(2) == Some(&b))
        .map(|(i, ..)| *i)
        .expect("premise: the commit names the index for B after emptying it");
    assert!(
        xadd < emptied && emptied < named,
        "premise: XADD ({xadd}) < HDEL A ({emptied}) < the next command on the index for B \
         ({named})"
    );
}

/// T-Q1. The staged structural commit (`ROOT`'s children list rebuilt), with
/// every key it names but its sentinel present on the migrating source: the
/// source refuses it `-TRYAGAIN` before it starts, and the object's keyspace
/// is byte-identical. Before b.qtk it ran there and tore at the `RPUSH`: the
/// stream entry published, `seq` unchanged, `ROOT`'s children list gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn b_qtk_structural_children_rebuild_slot_migration_disruptive_refuses_before_any_write() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let staged = stage_children_rebuild(tv).await;
    let tv = &staged.fx.tv;
    let before = dump(tv, &staged.keys).await;
    assert!(
        before.iter().all(|(_, d)| d.is_some()),
        "premise: every key but the sentinel is present"
    );

    let (mut migration, p) = migrate(tv, &staged.children).await;
    let (result, _) = tokio::time::timeout(SETTLE, staged.fx.commit(&staged.commit))
        .await
        .expect("the commit did not settle within its budget");
    let after = dump(tv, &staged.keys).await;
    migration.complete().await;

    eprintln!(
        "B.QTK T-Q1 DATAPOINT: slot={} {} -> {}\n  structural commit: {result:?}",
        p.slot, p.src, p.dst
    );
    match &result {
        Err(e) => assert!(
            transient_message(e).is_some_and(|m| m.contains(TRYAGAIN_REPLY)),
            "the migrating source refuses the commit -TRYAGAIN before it starts; got {e:?}"
        ),
        Ok(outcome) => panic!(
            "the commit ran on a migrating slot ({outcome:?}); its sentinel must keep it from \
             starting"
        ),
    }
    assert_eq!(
        after, before,
        "nothing was written: the keyspace is byte-identical"
    );
    ran(
        "b_qtk_structural_children_rebuild_slot_migration_disruptive_refuses_before_any_write",
        tv,
    );
}

/// T-Q2. The staged fast-mode commit (the layer's only external value
/// replaced), with every key it names but its sentinel present on the
/// migrating source: refused `-TRYAGAIN` before it starts, the keyspace
/// byte-identical. Before b.qtk it ran there and tore at the `HGET obj:ext B`
/// after the `HDEL` that emptied the index, after its `XADD`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn b_qtk_fast_external_value_replace_slot_migration_disruptive_refuses_before_any_write() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let staged = stage_external_value_replace(tv).await;
    let tv = &staged.tv;
    let before = dump(tv, &staged.keys).await;
    assert!(
        before.iter().all(|(_, d)| d.is_some()),
        "premise: every key but the sentinel is present"
    );
    let one_attempt = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
        .with_update_layer_retry_max_attempts(1);

    let (mut migration, p) = migrate(tv, &staged.ext_key).await;
    let result = tokio::time::timeout(
        SETTLE,
        one_attempt.update_layer_relayed(
            &staged.session_id,
            Q2_LAYER,
            staged.oid,
            Some(q2_replace_delta()),
            WRITER,
            2,
        ),
    )
    .await
    .expect("the commit did not settle within its budget");
    let after = dump(tv, &staged.keys).await;
    migration.complete().await;

    eprintln!(
        "B.QTK T-Q2 DATAPOINT: slot={} {} -> {}\n  fast-mode commit: {result:?}",
        p.slot, p.src, p.dst
    );
    match &result {
        Err(e) => assert!(
            transient_message(e).is_some_and(|m| m.contains(TRYAGAIN_REPLY)),
            "the migrating source refuses the commit -TRYAGAIN before it starts; got {e:?}"
        ),
        Ok(landed) => panic!(
            "the commit ran on a migrating slot ({landed:?}); its sentinel must keep it from \
             starting"
        ),
    }
    assert_eq!(
        after, before,
        "nothing was written: the keyspace is byte-identical"
    );
    ran(
        "b_qtk_fast_external_value_replace_slot_migration_disruptive_refuses_before_any_write",
        tv,
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The other sentinel scripts
// ═════════════════════════════════════════════════════════════════════════════

fn gc_keys(ks: &KeySchema) -> [String; 6] {
    [
        ks.ext_inflight(),
        ks.ext_pending(),
        ks.ext_pending_age(),
        ks.ext_pending_bytes(),
        ks.ext_pending_total(),
        ks.ext_settled(),
    ]
}

/// Stage the `{gc}` ledger with two members `m1`, `m2` (bytes 10 and 20, in
/// that insertion order), `m1` alone in `pending:age`, `inflight` holding
/// `claimed` with the bare score 1000, and every other `{gc}` key non-empty
/// with an unrelated member. Returns the two members.
async fn stage_gc_ledger(tv: &TestValkey, claimed: &[&str]) -> (String, String) {
    let ks = tv.key_schema();
    let pool = &tv.client().pool;
    let (m1, m2) = ("bcyu-m1".to_owned(), "bcyu-m2".to_owned());
    let _: () = pool
        .hset(ks.ext_pending_bytes(), (m1.as_str(), "10"))
        .await
        .expect("plant m1's bytes");
    let _: () = pool
        .hset(ks.ext_pending_bytes(), (m2.as_str(), "20"))
        .await
        .expect("plant m2's bytes");
    let _: () = pool
        .set(ks.ext_pending_total(), "30", None, None, false)
        .await
        .expect("plant the total");
    let _: () = pool
        .zadd(
            ks.ext_pending_age(),
            None,
            None,
            false,
            false,
            (1.0, m1.as_str()),
        )
        .await
        .expect("plant m1's age");
    let _: () = pool
        .sadd(ks.ext_pending(), "bcyu-unrelated")
        .await
        .expect("plant pending");
    let _: () = pool
        .zadd(
            ks.ext_settled(),
            None,
            None,
            false,
            false,
            (1.0, "bcyu-unrelated"),
        )
        .await
        .expect("plant settled");
    let _: () = pool
        .zadd(
            ks.ext_inflight(),
            None,
            None,
            false,
            false,
            (1.0, "bcyu-unrelated-claim"),
        )
        .await
        .expect("plant inflight");
    for member in claimed {
        let _: () = pool
            .zadd(
                ks.ext_inflight(),
                None,
                None,
                false,
                false,
                (1_000.0, *member),
            )
            .await
            .expect("plant an older build's claim");
    }
    (m1, m2)
}

/// T-S1. The older-build claim drop over two claims, called directly with its
/// sentinel in `KEYS` (put there by this test), every other `{gc}` key present
/// on the migrating source: Valkey refuses it `-TRYAGAIN` before it starts,
/// nothing changed. This pins the refusal of a script that declares its
/// sentinel, not that `drop_foreign_claims` declares it; that is pinned by
/// `script_migration_guard_test`'s builder-site and keys-carry tests and by
/// `gc_drop_foreign_claims_without_its_migration_sentinel_is_refused_and_writes_nothing`.
/// For the record, the unsentineled call production made before b.cyu ran
/// here and tore: `m1` dropped and settled, `pending:age` emptied, then `m2`'s
/// `ZREM pending:age` refused after its `ZREM inflight` landed — `m2` out of
/// `inflight`, unsettled, its bytes held.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn gc_drop_foreign_claims_slot_migration_disruptive_refuses_before_any_write() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let ks = tv.key_schema();
    let (m1, m2) = stage_gc_ledger(&tv, &["bcyu-m1", "bcyu-m2"]).await;
    let keys = gc_keys(ks);
    let before = dump(&tv, &keys).await;
    let mut script_keys = keys.to_vec();
    script_keys.push(ks.migration_sentinel("gc"));

    let (mut migration, p) = migrate(&tv, &ks.ext_pending()).await;
    let answer = tokio::time::timeout(
        SETTLE,
        GC_DROP_FOREIGN_CLAIMS_SCRIPT.evalsha_with_reload::<Vec<String>, _, _>(
            tv.client().pool.next(),
            script_keys,
            vec![
                GC_CLAIM_SCORE_OFFSET_MS.to_string(),
                GC_DECREMENT_FENCE_MS.to_string(),
                m1,
                "1000".to_owned(),
                m2,
                "1000".to_owned(),
            ],
        ),
    )
    .await
    .expect("the drop did not settle within its budget");
    let after = dump(&tv, &keys).await;
    migration.complete().await;

    eprintln!(
        "B.CYU T-S1 DATAPOINT: slot={} {} -> {}\n  drop: {answer:?}",
        p.slot, p.src, p.dst
    );
    assert_refused_tryagain(&answer, "gc_drop_foreign_claims");
    assert_eq!(after, before, "nothing changed");
    ran(
        "gc_drop_foreign_claims_slot_migration_disruptive_refuses_before_any_write",
        &tv,
    );
}

/// T-S2. One ledger audit page over `m1`, `m2` (in neither `pending` nor
/// `inflight`), called directly with its sentinel in `KEYS` (put there by
/// this test), every other `{gc}` key present on the migrating source: Valkey
/// refuses it `-TRYAGAIN` before it starts, nothing changed. This pins the
/// refusal, not that `gc_audit_page` declares the sentinel; that is pinned by
/// `script_migration_guard_test`'s builder-site and keys-carry tests and by
/// `gc_ledger_audit_without_its_migration_sentinel_is_refused_and_writes_nothing`.
/// For the record, the unsentineled call production made before b.cyu ran
/// here and tore: `m1` settled and `pending:age` emptied, then `m2`'s
/// `ZREM pending:age` refused — a partial page.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn gc_ledger_audit_slot_migration_disruptive_refuses_before_any_write() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let ks = tv.key_schema();
    stage_gc_ledger(&tv, &[]).await;
    let keys = gc_keys(ks);
    let before = dump(&tv, &keys).await;
    let script_keys = vec![
        ks.ext_pending(),
        ks.ext_inflight(),
        ks.ext_pending_age(),
        ks.ext_pending_bytes(),
        ks.ext_pending_total(),
        ks.ext_settled(),
        ks.migration_sentinel("gc"),
    ];

    let (mut migration, p) = migrate(&tv, &ks.ext_pending()).await;
    let answer = tokio::time::timeout(
        SETTLE,
        GC_LEDGER_AUDIT_SCRIPT.evalsha_with_reload::<(String, i64), _, _>(
            tv.client().pool.next(),
            script_keys,
            vec!["0".to_owned()],
        ),
    )
    .await
    .expect("the audit did not settle within its budget");
    let after = dump(&tv, &keys).await;
    migration.complete().await;

    eprintln!(
        "B.CYU T-S2 DATAPOINT: slot={} {} -> {}\n  audit page: {answer:?}",
        p.slot, p.src, p.dst
    );
    assert_refused_tryagain(&answer, "gc_ledger_audit");
    assert_eq!(after, before, "nothing changed");
    ran(
        "gc_ledger_audit_slot_migration_disruptive_refuses_before_any_write",
        &tv,
    );
}

/// T-S3. The external-value index recount of a live object whose index holds
/// a stale hash no row holds (a lowering) and lacks the hash its one row
/// holds (a raise), called directly with its sentinel in `KEYS` (put there by
/// this test), every other declared key present on the migrating source:
/// Valkey refuses it `-TRYAGAIN` before it starts, nothing written. This pins
/// the refusal, not that `recount_ext_index` declares the sentinel; that is
/// pinned by `script_migration_guard_test`'s builder-site and keys-carry tests
/// and by
/// `ext_index_recount_without_its_migration_sentinel_is_refused_and_writes_nothing`.
/// For the record, the unsentineled call production made before b.cyu ran
/// here and tore: the header stamped, the lowering's `HDEL` emptied
/// `obj:ext`, and the raise's `HSET` was refused — the index part-fixed,
/// against the script's "nothing written on error" contract.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn ext_index_recount_slot_migration_disruptive_refuses_before_any_write() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let staged = stage_external_value_replace(tv).await;
    let tv = &staged.tv;
    let ks = tv.key_schema();
    let o = staged.oid.to_string();
    // The one row holds A; the index forgets A and holds a stale hash.
    let stale = sha256_to_hex(&q2_hash(0x5A));
    let a = sha256_to_hex(&q2_hash(0xA1));
    let _: u64 = tv
        .client()
        .pool
        .hdel(&staged.ext_key, &a)
        .await
        .expect("drop A from the index");
    let _: () = tv
        .client()
        .pool
        .hset(&staged.ext_key, (stale.as_str(), "1"))
        .await
        .expect("plant the stale hash");
    let keys = vec![
        ks.obj_header(&o),
        ks.obj_ext_index(&o),
        ks.obj_fields_index(&o),
        ks.obj_time_samples_index(&o),
    ];
    let before = dump(tv, &keys).await;
    assert!(
        before.iter().all(|(_, d)| d.is_some()),
        "premise: every declared key is present"
    );
    let mut script_keys = keys.clone();
    script_keys.push(ks.migration_sentinel(&o));

    let (mut migration, p) = migrate(tv, &staged.ext_key).await;
    let answer = tokio::time::timeout(
        SETTLE,
        EXT_INDEX_RECOUNT.evalsha_with_reload::<Vec<String>, _, _>(
            tv.client().pool.next(),
            script_keys,
            vec![ks.prefix().to_owned(), o.clone(), "1000".to_owned()],
        ),
    )
    .await
    .expect("the recount did not settle within its budget");
    let after = dump(tv, &keys).await;
    migration.complete().await;

    eprintln!(
        "B.CYU T-S3 DATAPOINT: slot={} {} -> {}\n  recount: {answer:?}",
        p.slot, p.src, p.dst
    );
    assert_refused_tryagain(&answer, "ext_index_recount");
    assert_eq!(
        after, before,
        "nothing was written, the header stamp included"
    );
    ran(
        "ext_index_recount_slot_migration_disruptive_refuses_before_any_write",
        tv,
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The write-first scripts (decision 9)
// ═════════════════════════════════════════════════════════════════════════════

/// T-R1. A verified store over a wrong-typed blob (a hash), both keys present
/// on the migrating source: the heal's plain `SET` names a present key, so
/// the script lands whole — the blob healed, `ref_count` +1. Before b.cyu its
/// `DEL` emptied the data key and the `SET` after it was refused: the blob
/// gone, no `+1`, an error.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn ext_store_value_heal_slot_migration_disruptive_lands_whole() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let ks = tv.key_schema();
    let hex = sha256_to_hex(&q2_hash(0xE1));
    let (meta, data) = (ks.ext_meta(&hex), ks.ext_data(&hex));
    let pool = &tv.client().pool;
    let _: () = pool
        .hset(&meta, ("ref_count", "1"))
        .await
        .expect("plant meta");
    let _: () = pool
        .hset(&data, ("not", "a-blob"))
        .await
        .expect("plant the wrong-typed blob");

    let (mut migration, p) = migrate(&tv, &data).await;
    let answer = tokio::time::timeout(
        SETTLE,
        EXT_STORE_VALUE_SCRIPT.evalsha_with_reload::<(i64, String), _, _>(
            pool.next(),
            vec![meta.clone(), data.clone()],
            vec![
                fred::types::Value::from(&b"verified-bytes"[..]),
                fred::types::Value::from("1"),
            ],
        ),
    )
    .await
    .expect("the store did not settle within its budget");
    migration.complete().await;

    eprintln!(
        "B.CYU T-R1 DATAPOINT: slot={} {} -> {}\n  store: {answer:?}",
        p.slot, p.src, p.dst
    );
    let (ref_count, _) = answer.expect("the heal lands whole on the migrating source");
    assert_eq!(ref_count, 2, "the store paid its +1");
    let stored: Option<String> = pool.get(&data).await.expect("GET data");
    assert_eq!(
        stored.as_deref(),
        Some("verified-bytes"),
        "the blob is healed"
    );
    ran(
        "ext_store_value_heal_slot_migration_disruptive_lands_whole",
        &tv,
    );
}

/// T-R2. A hold `take` over a read-hold key holding one expired field only,
/// on the migrating source: the take writes its own field before pruning, so
/// no command names the key emptied and it lands whole — the own field
/// written, the expired one pruned. Before b.cyu the prune emptied the key and
/// the write was refused: no hold, an error.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn ext_read_hold_take_slot_migration_disruptive_lands_whole() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let hold = tv.key_schema().ext_read_hold();
    let pool = &tv.client().pool;
    let _: () = pool
        .hset(&hold, ("42:expired-holder", "1:2"))
        .await
        .expect("plant the expired field");
    let own = "42:bcyu-own-holder";

    let (mut migration, p) = migrate(&tv, &hold).await;
    let answer = tokio::time::timeout(
        SETTLE,
        EXT_READ_HOLD_SCRIPT.evalsha_with_reload::<i64, _, _>(
            pool.next(),
            vec![hold.clone()],
            vec!["take".to_owned(), own.to_owned(), "15000".to_owned()],
        ),
    )
    .await
    .expect("the take did not settle within its budget");
    migration.complete().await;

    eprintln!(
        "B.CYU T-R2 DATAPOINT: slot={} {} -> {}\n  take: {answer:?}",
        p.slot, p.src, p.dst
    );
    let start = answer.expect("the take lands whole on the migrating source");
    assert!(start > 0, "the take answers its start");
    let fields: HashMap<String, String> = pool.hgetall(&hold).await.expect("HGETALL hold");
    assert!(
        fields.contains_key(own) && !fields.contains_key("42:expired-holder"),
        "the own field is written and the expired one pruned: {fields:?}"
    );
    ran(
        "ext_read_hold_take_slot_migration_disruptive_lands_whole",
        &tv,
    );
}

/// T-R3. An intent register on an object whose fence lingers (no live
/// reader, its size kept) and whose intent key holds one expired intent only,
/// every key present on the migrating source: the register writes its own
/// field before pruning, so it lands whole — `REGISTERED`. Before b.cyu the
/// prune emptied the intent key and the write was refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn snapshot_intent_register_slot_migration_disruptive_lands_whole() {
    let Some(tv) = cluster(POOL_SIZE).await else {
        return;
    };
    let ks = tv.key_schema();
    let session_id = SessionStore::new(tv.client().clone(), ks.clone())
        .create("bcyu-tr3", "/bcyu-tr3", CREATED_BY)
        .await
        .expect("session creation must succeed");
    let oid = LayerStore::new(tv.client().clone(), ks.clone(), 4096)
        .create_layer(
            &session_id,
            "bcyu-tr3-layer",
            CREATED_BY,
            &ObjectState {
                sequence_num: 1,
                first_version: 1,
                source_format: "usda".to_owned(),
                ..Default::default()
            },
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    let o = oid.to_string();
    let pool = &tv.client().pool;
    let _: () = pool
        .hset(
            ks.obj_snapshot_fence(&o),
            [("size_e", "10"), ("size_b", "1000")],
        )
        .await
        .expect("plant the lingering fence");
    let _: () = pool
        .hset(ks.obj_snapshot_intent(&o), ("w:expired-writer", "1"))
        .await
        .expect("plant the expired intent");

    let (mut migration, p) = migrate(&tv, &ks.obj_header(&o)).await;
    let answer = tokio::time::timeout(
        SETTLE,
        register_intent(
            tv.client(),
            ks,
            oid,
            LayerSize {
                items: 10,
                bytes: 1_000,
            },
        ),
    )
    .await
    .expect("the register did not settle within its budget");
    let registered = matches!(answer, Ok(IntentVerdict::Registered(_)));
    let shown = format!("{answer:?}");
    if let Ok(IntentVerdict::Registered(intent)) = answer {
        // Cleared after the migration, below; dropping it would spawn the
        // clear now.
        migration.complete().await;
        intent.clear().await.expect("the intent clears");
    } else {
        migration.complete().await;
    }

    eprintln!(
        "B.CYU T-R3 DATAPOINT: slot={} {} -> {}\n  register: {shown}",
        p.slot, p.src, p.dst
    );
    assert!(
        registered,
        "the register lands whole on the migrating source; got {shown}"
    );
    ran(
        "snapshot_intent_register_slot_migration_disruptive_lands_whole",
        &tv,
    );
}
