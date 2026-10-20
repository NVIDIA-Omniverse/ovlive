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

//! b.cyu — the migration sentinel's runtime fence, and the write-first order
//! of the three decision-9 scripts, on the standalone test store.
//!
//! **The sentinel.** Five writing scripts that could stop part-way through a
//! legacy cluster slot migration declare their slot's never-written migration
//! sentinel (`KeySchema::migration_sentinel`), so the store refuses them
//! before they start while the slot moves. A standalone store has no cluster
//! check, so what can be tested here is the script's own check that the
//! caller passed the sentinel: a call without it, or with a sentinel of
//! another hash tag (which would `CROSSSLOT` on a cluster) or a key that is
//! not a sentinel, is refused before any read or write, and nothing is
//! written. This file covers the two GC ledger scripts, the external-value
//! index recount and the time-sample signed-zero merge; the commit script's
//! twin is in `update_layer_lua_test.rs`
//! (`update_layer_fast_script_without_its_migration_sentinel_is_refused_before_any_write`
//! and its sibling). The four answer `ENCODING` (the GC scripts),
//! `RECOUNT_ARGS` (the recount) and `MERGE_ARGS` (the merge); the recount's
//! and the merge's other argument refusals share their tag. Each refusal case
//! has a control: the same call with the sentinel writes, so "nothing
//! written" is not the staging writing nothing;
//! and after it lands the sentinel is still absent, since a script that wrote
//! its sentinel, however it spelled the key, would switch its own refusal off
//! for that slot.
//!
//! **Write-first order (decision 9).** Three scripts that cannot take a
//! sentinel without refusing client-visible work for a whole move are kept
//! from stopping part-way by their command order instead: no command names a
//! key the run itself emptied. `ext_store_value` heals a wrong-typed blob
//! with a plain `SET` (no `DEL` first); `ext_read_hold`'s `take` and
//! `snapshot_intent_register` write their own field before pruning the
//! others. The order is read off the store's `MONITOR` feed, where a
//! script's commands appear as `lua`'s, and the stored result is checked to
//! be what the old order stored. Each test fails on the old order. The tear
//! itself, on a real migrating slot, is pinned by the cluster-gated
//! `b_cyu_slot_migration_test` (`…_slot_migration_disruptive_lands_whole`).
//!
//! # Wait budget
//!
//! One script call or a few round trips per step against a healthy store,
//! each under fred's command timeout; the `MONITOR` drains are bounded by
//! `MONITOR_DRAIN_BUDGET`. Healthy, every test takes well under a second.

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, SortedSetsInterface};
use fred::types::scripts::Script;

use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::{
    MockExternalValueStore, EXT_READ_HOLD_SCRIPT, EXT_STORE_VALUE_SCRIPT, GC_CLAIM_SCORE_OFFSET_MS,
    GC_DECREMENT_FENCE_MS, GC_DROP_FOREIGN_CLAIMS_SCRIPT, GC_LEDGER_AUDIT_SCRIPT,
};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::{register_intent, IntentVerdict, LayerSize};
use live_edit::storage::test_support::TestValkey;

use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};

/// The recount script, loaded from the source the service compiles in (its
/// `Script` static is crate-private).
static EXT_INDEX_RECOUNT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../../src/storage/scripts/ext_index_recount.lua"
    ))
});

/// The signed-zero merge script, loaded the same way (its `Script` static is
/// crate-private too).
static TIME_SAMPLE_SIGNED_ZERO_MERGE: std::sync::LazyLock<Script> =
    std::sync::LazyLock::new(|| {
        Script::from_lua(include_str!(
            "../../../src/storage/scripts/time_sample_signed_zero_merge.lua"
        ))
    });

/// An opaque GC ledger member: the scripts never parse it.
const MEMBER: &str = "bcyu-member";
/// The bytes the staged member holds in the ledger.
const MEMBER_BYTES: i64 = 10;
/// A 64-hex external-value hash.
const HASH_HEX: &str = "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00";

/// Every `DUMP` of `keys`: type-agnostic, `None` for an absent key, so one
/// equality covers "not changed" and "not created".
async fn dump(tv: &TestValkey, keys: &[String]) -> Vec<(String, Option<Vec<u8>>)> {
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let dumped: Option<Vec<u8>> = tv.client().pool.dump(key).await.expect("DUMP must succeed");
        out.push((key.clone(), dumped));
    }
    out
}

/// After a landed run, `sentinel` is still absent: the script wrote nothing
/// to it, however it spells the key. A script that created its sentinel
/// would switch its own refusal off for that slot — the sentinel works only
/// while it is absent from every node — so this catches what the guard's
/// static scan cannot (a computed key, an alias the scan does not follow).
async fn assert_sentinel_still_absent(tv: &TestValkey, sentinel: &str, script: &str) {
    let exists: i64 = tv
        .client()
        .pool
        .exists(sentinel)
        .await
        .expect("EXISTS sentinel must succeed");
    assert_eq!(
        exists, 0,
        "{script} landed and left its migration sentinel {sentinel} absent: the sentinel is \
         never written"
    );
}

/// The three ways a call can fail to pass its sentinel as `KEYS[n]`: left
/// out, of another hash tag, or a key that is not a sentinel.
fn without_a_sentinel(
    keys: &KeySchema,
    base: &[String],
    own_tag: &str,
) -> Vec<(&'static str, Vec<String>)> {
    let with = |last: String| {
        let mut k = base.to_vec();
        k.push(last);
        k
    };
    vec![
        ("no sentinel", base.to_vec()),
        (
            "another tag's sentinel",
            with(keys.migration_sentinel(&format!("{own_tag}-other"))),
        ),
        ("a key that is not a sentinel", with(base[0].clone())),
    ]
}

// ═════════════════════════════════════════════════════════════════════════════
// The GC ledger scripts
// ═════════════════════════════════════════════════════════════════════════════

/// Stage one member in the release ledger — its bytes field, the total and
/// its age entry — and, when `claimed_at` is `Some`, an older build's claim
/// on it in `inflight` scored with that bare time.
async fn stage_ledger_member(tv: &TestValkey, claimed_at: Option<f64>) {
    let ks = tv.key_schema();
    let pool = &tv.client().pool;
    let _: () = pool
        .hset(ks.ext_pending_bytes(), (MEMBER, MEMBER_BYTES.to_string()))
        .await
        .expect("plant the ledger bytes");
    let _: () = pool
        .set(
            ks.ext_pending_total(),
            MEMBER_BYTES.to_string(),
            None,
            None,
            false,
        )
        .await
        .expect("plant the total");
    let _: () = pool
        .zadd(
            ks.ext_pending_age(),
            None,
            None,
            false,
            false,
            (1.0, MEMBER),
        )
        .await
        .expect("plant the age entry");
    if let Some(score) = claimed_at {
        let _: () = pool
            .zadd(ks.ext_inflight(), None, None, false, false, (score, MEMBER))
            .await
            .expect("plant the older build's claim");
    }
}

fn gc_ledger_keys(ks: &KeySchema) -> Vec<String> {
    vec![
        ks.ext_inflight(),
        ks.ext_pending(),
        ks.ext_pending_age(),
        ks.ext_pending_bytes(),
        ks.ext_pending_total(),
        ks.ext_settled(),
    ]
}

/// `gc_drop_foreign_claims.lua` refuses `ENCODING` before any read or write
/// when its `KEYS[7]` is not the `{gc}` migration sentinel, and writes
/// nothing; with the sentinel, the same call drops the staged claim.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_drop_foreign_claims_without_its_migration_sentinel_is_refused_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    stage_ledger_member(&tv, Some(1_000.0)).await;
    let base = gc_ledger_keys(ks);
    let argv = vec![
        GC_CLAIM_SCORE_OFFSET_MS.to_string(),
        GC_DECREMENT_FENCE_MS.to_string(),
        MEMBER.to_owned(),
        "1000".to_owned(),
    ];
    let before = dump(&tv, &base).await;

    for (case, keys) in without_a_sentinel(ks, &base, "gc") {
        let err = GC_DROP_FOREIGN_CLAIMS_SCRIPT
            .evalsha_with_reload::<Vec<String>, _, _>(tv.client().pool.next(), keys, argv.clone())
            .await
            .expect_err("a call without its migration sentinel must be refused");
        assert_eq!(
            err.details(),
            "ENCODING: gc_drop_foreign_claims expects 7 KEYS, KEYS[7] the migration sentinel of \
             KEYS[1]'s hash tag",
            "{case}: the refusal names the contract"
        );
        assert_eq!(
            dump(&tv, &base).await,
            before,
            "{case}: the refusal comes before any write"
        );
    }

    let mut keys = base.clone();
    keys.push(ks.migration_sentinel("gc"));
    let dropped: Vec<String> = GC_DROP_FOREIGN_CLAIMS_SCRIPT
        .evalsha_with_reload(tv.client().pool.next(), keys, argv)
        .await
        .expect("with its sentinel the drop runs");
    assert_eq!(
        dropped,
        vec![MEMBER.to_owned()],
        "control: the staged claim is one the drop takes"
    );
    assert_ne!(dump(&tv, &base).await, before, "control: the drop wrote");
    assert_sentinel_still_absent(&tv, &ks.migration_sentinel("gc"), "gc_drop_foreign_claims").await;
}

/// `gc_ledger_audit.lua` refuses `ENCODING` before any read or write when
/// its `KEYS[7]` is not the `{gc}` migration sentinel, and writes nothing;
/// with the sentinel, the same page settles the staged member.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_ledger_audit_without_its_migration_sentinel_is_refused_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    stage_ledger_member(&tv, None).await;
    let base = vec![
        ks.ext_pending(),
        ks.ext_inflight(),
        ks.ext_pending_age(),
        ks.ext_pending_bytes(),
        ks.ext_pending_total(),
        ks.ext_settled(),
    ];
    let before = dump(&tv, &base).await;

    for (case, keys) in without_a_sentinel(ks, &base, "gc") {
        let err = GC_LEDGER_AUDIT_SCRIPT
            .evalsha_with_reload::<(String, i64), _, _>(
                tv.client().pool.next(),
                keys,
                vec!["0".to_owned()],
            )
            .await
            .expect_err("a call without its migration sentinel must be refused");
        assert_eq!(
            err.details(),
            "ENCODING: gc_ledger_audit expects 7 KEYS, KEYS[7] the migration sentinel of \
             KEYS[1]'s hash tag",
            "{case}: the refusal names the contract"
        );
        assert_eq!(
            dump(&tv, &base).await,
            before,
            "{case}: the refusal comes before any write"
        );
    }

    let mut keys = base.clone();
    keys.push(ks.migration_sentinel("gc"));
    let (next, repaired): (String, i64) = GC_LEDGER_AUDIT_SCRIPT
        .evalsha_with_reload(tv.client().pool.next(), keys, vec!["0".to_owned()])
        .await
        .expect("with its sentinel the audit runs");
    assert_eq!(
        (next.as_str(), repaired),
        ("0", 1),
        "control: the staged member is one the audit settles"
    );
    assert_ne!(dump(&tv, &base).await, before, "control: the audit wrote");
    assert_sentinel_still_absent(&tv, &ks.migration_sentinel("gc"), "gc_ledger_audit").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The external-value index recount
// ═════════════════════════════════════════════════════════════════════════════

async fn header_and_index(
    tv: &TestValkey,
    oid: &str,
) -> (HashMap<String, String>, HashMap<String, String>) {
    let ks = tv.key_schema();
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(ks.obj_header(oid))
        .await
        .expect("HGETALL header");
    let index: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(ks.obj_ext_index(oid))
        .await
        .expect("HGETALL obj:ext");
    (header, index)
}

/// `ext_index_recount.lua` refuses `RECOUNT_ARGS` before any read or write
/// when its `KEYS[5]` is not the object's migration sentinel — left out, of
/// another tag, or another key — and writes nothing, the header stamp
/// included; with the sentinel, the same call repairs the staged drift.
///
/// The live object's index carries a count for a hash no row holds, so a
/// recount that ran would stamp the header and remove the field.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ext_index_recount_without_its_migration_sentinel_is_refused_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let session_id = SessionStore::new(tv.client().clone(), ks.clone())
        .create("bcyu-recount", "/bcyu-recount", "bcyu-user")
        .await
        .expect("session creation must succeed");
    let object_id = LayerStore::new(tv.client().clone(), ks.clone(), 4096)
        .create_layer(
            &session_id,
            "bcyu-recount-layer",
            "bcyu-user",
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
    let oid = object_id.to_string();
    let _: () = tv
        .client()
        .pool
        .hset(ks.obj_ext_index(&oid), (HASH_HEX, "3"))
        .await
        .expect("plant the over-count");
    let base = vec![
        ks.obj_header(&oid),
        ks.obj_ext_index(&oid),
        ks.obj_fields_index(&oid),
        ks.obj_time_samples_index(&oid),
    ];
    let argv = vec![ks.prefix().to_owned(), oid.clone(), "1000".to_owned()];
    let before = header_and_index(&tv, &oid).await;

    for (case, keys) in without_a_sentinel(ks, &base, &oid) {
        let err = EXT_INDEX_RECOUNT
            .evalsha_with_reload::<Vec<String>, _, _>(tv.client().pool.next(), keys, argv.clone())
            .await
            .expect_err("a call without its migration sentinel must be refused");
        assert_eq!(
            err.details(),
            format!(
                "RECOUNT_ARGS: KEYS[5] is not the migration sentinel of {}{{{oid}}}",
                ks.prefix()
            ),
            "{case}: the recount refuses with its own argument tag"
        );
        assert_eq!(
            header_and_index(&tv, &oid).await,
            before,
            "{case}: the refusal comes before any write, the header stamp included"
        );
    }

    let mut keys = base.clone();
    keys.push(ks.migration_sentinel(&oid));
    let reply: Vec<String> = EXT_INDEX_RECOUNT
        .evalsha_with_reload(tv.client().pool.next(), keys, argv)
        .await
        .expect("with its sentinel the recount runs");
    assert_eq!(
        reply.first().map(String::as_str),
        Some("REPAIRED"),
        "control: the staged drift is one the recount repairs; got {reply:?}"
    );
    let (header, index) = header_and_index(&tv, &oid).await;
    assert!(
        header.contains_key("ext_index_repaired_ms") && !index.contains_key(HASH_HEX),
        "control: the recount stamped the header and removed the over-count"
    );
    assert_sentinel_still_absent(&tv, &ks.migration_sentinel(&oid), "ext_index_recount").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The time-sample signed-zero merge
// ═════════════════════════════════════════════════════════════════════════════

/// `time_sample_signed_zero_merge.lua` refuses `MERGE_ARGS` before any read
/// or write when its `KEYS[4]` is not the object's migration sentinel — left
/// out (the row keys shift into its slot), of another tag, or another key —
/// and writes nothing, the header stamp included; with the sentinel, the same
/// call re-keys the staged legacy `-0.0` row to `+0.0`.
///
/// The live object holds one lone `-0.0` time-sample row and its `obj:tss`
/// member, as an older build wrote them, so a merge that ran would stamp the
/// header, rename the row and move its member.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn time_sample_signed_zero_merge_without_its_migration_sentinel_is_refused_and_writes_nothing(
) {
    const SECTION: &str = "7";
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let session_id = SessionStore::new(tv.client().clone(), ks.clone())
        .create("bcyu-merge", "/bcyu-merge", "bcyu-user")
        .await
        .expect("session creation must succeed");
    let object_id = LayerStore::new(tv.client().clone(), ks.clone(), 4096)
        .create_layer(
            &session_id,
            "bcyu-merge-layer",
            "bcyu-user",
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
    let oid = object_id.to_string();
    let (positive_row, negative_row) = (
        ks.obj_time_sample(&oid, SECTION, 0.0),
        ks.obj_time_sample(&oid, SECTION, -0.0),
    );
    let _: () = tv
        .client()
        .pool
        .hset(&negative_row, [("value", "AQ=="), ("version", "1")])
        .await
        .expect("plant the legacy -0.0 row");
    let _: i64 = tv
        .client()
        .pool
        .zadd(
            ks.obj_time_samples_index(&oid),
            None,
            None,
            false,
            false,
            (1.0, KeySchema::time_sample_member(SECTION, -0.0)),
        )
        .await
        .expect("plant its member");
    let seq: String = tv
        .client()
        .pool
        .hget(ks.obj_header(&oid), "seq")
        .await
        .expect("HGET seq");
    let (header, tss, fence) = (
        ks.obj_header(&oid),
        ks.obj_time_samples_index(&oid),
        ks.obj_snapshot_fence(&oid),
    );
    let rows = [positive_row.clone(), negative_row.clone()];
    let with_fourth = |fourth: Option<String>| {
        let mut keys = vec![header.clone(), tss.clone(), fence.clone()];
        keys.extend(fourth);
        keys.extend(rows.iter().cloned());
        keys
    };
    let argv = vec![
        ks.prefix().to_owned(),
        oid.clone(),
        seq,
        SECTION.to_owned(),
        "T".to_owned(),
    ];
    let watched = vec![header.clone(), tss.clone(), positive_row, negative_row];
    let before = dump(&tv, &watched).await;

    for (case, keys) in [
        ("no sentinel", with_fourth(None)),
        (
            "another tag's sentinel",
            with_fourth(Some(ks.migration_sentinel(&format!("{oid}-other")))),
        ),
        (
            "a key that is not a sentinel",
            with_fourth(Some(header.clone())),
        ),
    ] {
        let err = TIME_SAMPLE_SIGNED_ZERO_MERGE
            .evalsha_with_reload::<Vec<String>, _, _>(tv.client().pool.next(), keys, argv.clone())
            .await
            .expect_err("a call without its migration sentinel must be refused");
        assert_eq!(
            err.details(),
            format!(
                "MERGE_ARGS: KEYS[4] is not the migration sentinel of {}{{{oid}}}",
                ks.prefix()
            ),
            "{case}: the merge refuses with its own argument tag"
        );
        assert_eq!(
            dump(&tv, &watched).await,
            before,
            "{case}: the refusal comes before any write, the header stamp included"
        );
    }

    let reply: Vec<String> = TIME_SAMPLE_SIGNED_ZERO_MERGE
        .evalsha_with_reload(
            tv.client().pool.next(),
            with_fourth(Some(ks.migration_sentinel(&oid))),
            argv,
        )
        .await
        .expect("with its sentinel the merge runs");
    assert_eq!(
        reply,
        vec!["MERGED", "0", "1", "0"],
        "control: the staged -0.0 row is one the merge re-keys"
    );
    assert_ne!(
        dump(&tv, &watched).await,
        before,
        "control: the merge wrote"
    );
    assert_sentinel_still_absent(
        &tv,
        &ks.migration_sentinel(&oid),
        "time_sample_signed_zero_merge",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Write-first order (decision 9)
// ═════════════════════════════════════════════════════════════════════════════

/// The write commands a script can send.
const SCRIPT_WRITES: [&str; 7] = ["SET", "DEL", "UNLINK", "HSET", "HDEL", "PEXPIRE", "HINCRBY"];

/// The write commands scripts sent on `key`, in the order the store ran
/// them, from `tap`'s opening up to a fence issued now.
async fn script_writes_on(tv: &TestValkey, tap: &mut MonitorTap, key: &str) -> Vec<Vec<String>> {
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("EXISTS fence must succeed");
    tap.drain_until(&fence)
        .await
        .into_iter()
        .filter(|c| {
            c.issued_by_script()
                && c.args.get(1).map(String::as_str) == Some(key)
                && SCRIPT_WRITES.contains(&c.name().as_str())
        })
        .map(|c| {
            let mut args = c.args;
            args[0] = args[0].to_ascii_uppercase();
            args
        })
        .collect()
}

/// `ext_store_value.lua`'s verified heal of a wrong-typed blob is one plain
/// `SET`, with no `DEL` of the data key first: no command names the data key
/// while it is absent, so on a migrating source the heal cannot be refused
/// part-way (the old `DEL` + `SET` could leave the blob deleted with no
/// `+1`). The stored result is the old order's: the value as a string, its
/// TTL cleared, `ref_count` +1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ext_store_value_heals_a_wrong_typed_blob_without_emptying_it_first() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let (meta, data) = (ks.ext_meta(HASH_HEX), ks.ext_data(HASH_HEX));
    let pool = &tv.client().pool;
    let _: () = pool
        .hset(&meta, ("ref_count", "1"))
        .await
        .expect("plant the meta");
    let _: () = pool
        .hset(&data, ("not", "a-blob"))
        .await
        .expect("plant the wrong-typed blob");
    let _: bool = pool
        .pexpire(&data, 600_000, None)
        .await
        .expect("give it a TTL the heal must clear");
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let (ref_count, _stamp): (i64, String) = EXT_STORE_VALUE_SCRIPT
        .evalsha_with_reload(
            pool.next(),
            vec![meta.clone(), data.clone()],
            vec![
                fred::types::Value::from(&b"verified-bytes"[..]),
                fred::types::Value::from("1"),
            ],
        )
        .await
        .expect("the verified store runs");
    let writes = script_writes_on(&tv, &mut tap, &data).await;

    // The skipped `SET … NX`, then the heal's plain `SET`, and nothing that
    // empties the key: each write as its command and its options (the
    // arguments past the key and the value).
    let shapes: Vec<Vec<&str>> = writes
        .iter()
        .map(|w| {
            let mut shape = vec![w[0].as_str()];
            shape.extend(w.iter().skip(3).map(String::as_str));
            shape
        })
        .collect();
    assert_eq!(
        shapes,
        vec![vec!["SET", "NX"], vec!["SET"]],
        "the heal writes the data key with one plain SET after the skipped SET NX and never \
         empties it first; script writes on the data key: {writes:?}"
    );
    let stored: Option<String> = pool.get(&data).await.expect("GET data");
    assert_eq!(
        stored.as_deref(),
        Some("verified-bytes"),
        "the blob is healed"
    );
    let ttl: i64 = pool.pttl(&data).await.expect("PTTL data");
    assert_eq!(
        ttl, -1,
        "the heal clears the wrong-typed key's TTL, as DEL + SET did"
    );
    assert_eq!(ref_count, 2, "the store still pays its +1");
}

/// `ext_read_hold.lua`'s `take` writes the holder's own field before it
/// prunes any expired field: no prune can empty the key ahead of a command
/// naming it, so on a migrating source a `take` cannot be refused part-way
/// (the old prune-first order could delete the only, expired, field and then
/// be refused on the write). Its very first write is the out-of-memory opener
/// `HDEL <hold> :oom-gate`, a no-op that cannot grow memory and empties no
/// key, so the rest of the take runs on a full store (SDD §7.5.1, *The opener
/// rule*); the own field's `HSET` is the write right after it. The stored
/// result is the old order's: the own field live, the expired one gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ext_read_hold_take_writes_its_own_field_before_pruning_expired_ones() {
    let tv = TestValkey::setup().await;
    let hold = tv.key_schema().ext_read_hold();
    let pool = &tv.client().pool;
    // `<start>:<expires>`, long expired.
    let _: () = pool
        .hset(&hold, ("42:expired-holder", "1:2"))
        .await
        .expect("plant the expired field");
    let own = "42:bcyu-own-holder";
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let start: i64 = EXT_READ_HOLD_SCRIPT
        .evalsha_with_reload(
            pool.next(),
            vec![hold.clone()],
            vec!["take".to_owned(), own.to_owned(), "15000".to_owned()],
        )
        .await
        .expect("the take runs");
    let writes = script_writes_on(&tv, &mut tap, &hold).await;

    let opener = writes
        .iter()
        .position(|w| w[0] == "HDEL" && w.get(2).map(String::as_str) == Some(":oom-gate"));
    let first_hset = writes
        .iter()
        .position(|w| w[0] == "HSET" && w.get(2).map(String::as_str) == Some(own));
    let prune = writes
        .iter()
        .position(|w| w[0] == "HDEL" && w.get(2).map(String::as_str) == Some("42:expired-holder"));
    assert!(
        matches!((opener, first_hset, prune), (Some(0), Some(1), Some(p)) if p > 1),
        "the take's first write is the out-of-memory opener `HDEL :oom-gate`, its second its \
         own field's HSET, and the prune of the expired field comes after both; script writes \
         on the hold: {writes:?}"
    );
    let fields: HashMap<String, String> = pool.hgetall(&hold).await.expect("HGETALL hold");
    assert!(
        fields.contains_key(own) && !fields.contains_key("42:expired-holder") && start > 0,
        "the own field is taken and the expired one pruned: {fields:?}"
    );
}

/// `snapshot_intent_register.lua` writes the writer's own `w:` field before
/// it prunes any expired one, then sets the key's expiry: no prune can empty
/// the intent key ahead of a command naming it (the old prune-first order
/// could delete the only, expired, intent and then be refused on the
/// write). The stored result is the old order's: `REGISTERED`, the own field
/// present, the expired one gone, the key expiring.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_intent_register_writes_its_own_field_before_pruning_expired_ones() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let session_id = SessionStore::new(tv.client().clone(), ks.clone())
        .create("bcyu-intent", "/bcyu-intent", "bcyu-user")
        .await
        .expect("session creation must succeed");
    let object_id = LayerStore::new(tv.client().clone(), ks.clone(), 4096)
        .create_layer(
            &session_id,
            "bcyu-intent-layer",
            "bcyu-user",
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
    let intent_key = ks.obj_snapshot_intent(&object_id.to_string());
    let pool = &tv.client().pool;
    // A deadline long passed.
    let _: () = pool
        .hset(&intent_key, ("w:expired-writer", "1"))
        .await
        .expect("plant the expired intent");
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let verdict = register_intent(
        tv.client(),
        ks,
        object_id,
        LayerSize {
            items: 10,
            bytes: 1_000,
        },
    )
    .await
    .expect("the register runs");
    let writes = script_writes_on(&tv, &mut tap, &intent_key).await;

    let IntentVerdict::Registered(intent) = verdict else {
        panic!("with no live reader the intent registers; got {verdict:?}");
    };
    let kinds: Vec<&str> = writes.iter().map(|w| w[0].as_str()).collect();
    assert_eq!(
        kinds,
        vec!["HSET", "HDEL", "PEXPIRE"],
        "the register writes its own field first, then prunes the expired one, then sets the \
         expiry; script writes on the intent key: {writes:?}"
    );
    assert!(
        writes[0]
            .get(2)
            .is_some_and(|f| f.starts_with("w:") && f != "w:expired-writer")
            && writes[1].get(2).map(String::as_str) == Some("w:expired-writer"),
        "the HSET is the writer's own field and the HDEL the expired one: {writes:?}"
    );
    let fields: HashMap<String, String> = pool.hgetall(&intent_key).await.expect("HGETALL intent");
    assert!(
        fields.len() == 1 && !fields.contains_key("w:expired-writer"),
        "only the writer's own intent is left: {fields:?}"
    );
    let ttl: i64 = pool.pttl(&intent_key).await.expect("PTTL intent");
    assert!(ttl > 0, "the intent key expires: {ttl}");
    intent.clear().await.expect("the intent clears");
}
