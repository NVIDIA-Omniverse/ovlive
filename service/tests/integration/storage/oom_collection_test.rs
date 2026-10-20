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

//! b.pa1 — the GC's out-of-memory collection, against a real Valkey.
//!
//! Under the required `noeviction` policy a full shard refuses every write
//! that can grow memory. A store write refused `-OOM` anywhere in a replica
//! requests a collection (`le:ext:{gc}:oom:request`); the GC leader takes the
//! request at its next pass and collects every released external value at any
//! age (`GcPolicy::OOM_COLLECTION`), still clipped by live read holds, while
//! an upload hold (`le:ext:{gc}:oom:upload_hold`) keeps uploads carrying
//! external values back. The scripts the collection needs open with a write
//! that cannot grow memory (the "opener"), after which Valkey lets the rest
//! of the script run on a full store.
//!
//! # What lives here
//!
//! | Area | Tests |
//! |---|---|
//! | The opener changes no result (ST1, a differential against the pre-change script) | [`the_oom_opener_leaves_every_decrement_reply_and_stored_state_unchanged`] |
//! | One collection per request, rate-limited (ST4); its passes repeat while there is more | [`the_leader_runs_one_oom_collection_per_request_and_backs_off_after_a_futile_one`], [`an_oom_collection_repeats_its_passes_until_the_released_values_are_all_collected`] |
//! | Data safety | [`an_oom_collection_never_deletes_a_value_still_referenced`], [`an_oom_collection_leaves_a_release_a_live_read_hold_protects`] |
//! | The upload hold (ST5) | [`an_oom_collection_holds_uploads_only_with_a_release_budget`], [`the_upload_hold_stands_while_an_oom_collection_runs_and_is_cleared_when_it_fails`], [`the_upload_hold_is_cleared_when_an_oom_collection_is_cancelled`] |
//! | `refused`, by the decrement and by the upload hold, and the WARN once; a failed hold on the per-collection WARN | [`oom_collections_refused_by_the_store_count_refused_and_warn_once`] |
//! | A refused re-claim is skipped (F1) | [`a_refused_reclaim_skips_the_member_and_the_pass_goes_on`] |
//! | A real `-OOM` (disruptive, RT1–RT6) | the `store_write_refusal_disruptive_*` tests at the bottom |
//!
//! Elsewhere: the trigger's turn (one wake-up and one request per interval,
//! the wake after the send) in `src/storage/store_oom.rs`; an `-OOM` at an
//! upload site and at a mapper site writing the request in
//! `store_write_refusal_test.rs`; that only `-OOM` notes the signal, in
//! `src/storage/client.rs`; the gate's hold wait (GT1–GT4) in
//! `src/storage/ext_release_gate.rs`; the `OOM_COLLECTION` pin (U1), the
//! GC loop's wake-up on a note, and the retry a second after a `failed`
//! collection, in `src/storage/external_values.rs`; the
//! zero-seeded counter (U2) in `src/observability/metrics.rs`.
//!
//! # Driving it
//!
//! A test plants the collection request with `gc_oom_signal.lua`, the
//! production writer, and runs the leader's pass itself
//! (`ExternalValueStore::process_pending`, `GcPolicy::PRODUCTION`), so a
//! collection starts exactly when the test asks. The rate limit lives in the
//! pass state the caller carries (`GcPassState`), and a fresh state is a new
//! leadership, which may collect at once.
//!
//! Three tests stage a store error on one GC round trip through the
//! `test-fault-injection` seam (`ValkeyExternalValueStore::
//! with_process_pending_fault_injector_for_test`) and are compiled out
//! without that feature: the upload hold's error exit, `refused` (whose
//! upload-hold leg has a `ScriptReplyProxy` answer the hold's write `-OOM`),
//! and the skipped re-claim. The upload hold's two exits hold the
//! decrement's reply at a `ScriptReplyProxy`, so the hold is seen standing
//! mid-collection. Every item only those three use is gated with them.
//!
//! The tests whose names start `store_write_refusal_disruptive_` fill a real
//! store: they set its `maxmemory` at or below what it holds, so they are
//! `#[ignore]`d and run only against `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL`
//! (`super::disruptive_valkey`), one at a time (the
//! `disruptive-valkey-config` nextest group), in CI's
//! `test-cluster-disruptive:` job. Each first checks that a plain `SET` is
//! refused `-OOM`, so it cannot pass against a store that is not full.

use std::collections::BTreeMap;
#[cfg(feature = "test-fault-injection")]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "test-fault-injection")]
use std::sync::Mutex;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface,
};
use fred::types::scripts::Script;
use fred::types::{ClusterHash, CustomCommand, Value};

use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::ext_release_gate::ExtReleaseGate;
use live_edit::storage::external_values::{
    acquire_gc_leader_lock, ExtRelease, ExternalValueStore, GcLeaderAcquire, GcPassState, GcPolicy,
    ReadHold, ReadHoldRenewal, ValkeyExternalValueStore, GC_CLAIM_PENDING_SCRIPT,
    GC_DECREMENT_AND_COLLECT_SCRIPT, GC_DECREMENT_FENCE_MS, GC_LEADER_TAKEOVER_BELOW_MS,
    GC_OOM_SIGNAL_SCRIPT, LEADER_LOCK_TTL_MS, OOM_COLLECTION_RETRY_AFTER_FREED,
};
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::test_support::TestValkey;

use super::b_1jt_rollout_tests::{plant_own_claim, valkey_now_ms, STALE_OWN_CLAIM_AGE};
use super::disruptive_valkey::{
    assert_noeviction, disruptive_valkey_url, used_memory, RestoreMaxmemory,
};
#[cfg(feature = "test-fault-injection")]
use super::snapshot_read_support::error_reply_script;
use super::snapshot_read_support::{
    client_through, test_valkey_addr, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The object every release here names (a retirement).
const OBJ: u64 = 9_100;

/// A second releasing object, for a value with two references.
const OBJ_B: u64 = 9_101;

/// The bytes of every stored value, unless a test needs a size of its own.
const BLOB: usize = 64;

/// A release budget far above anything stored here: the gate exists (so a
/// collection sets the upload hold), and no pass is under pressure.
const LARGE_BUDGET: u64 = 1 << 30;

/// Bound on each wait for a background effect (a spawned `DEL`, a request
/// sent from a spawned task) or a staged collection.
const WAIT: Duration = Duration::from_secs(10);

/// The `-OOM` reply Valkey sends, as fred reports it.
#[cfg(feature = "test-fault-injection")]
const OOM_REPLY: &str = "OOM command not allowed when used memory > 'maxmemory'.";

/// Every `outcome` of `live_edit_gc_oom_collections_total`.
const OUTCOMES: [&str; 5] = [
    "freed",
    "freed_nothing",
    "nothing_to_collect",
    "failed",
    "refused",
];

/// The pre-b.pa1 decrement, vendored verbatim
/// (`tests/fixtures/pre_oom_opener/`): the opener's differential baseline.
/// Its reply is the old two-element `{ action, n }`.
static PRE_OPENER_DECREMENT: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../fixtures/pre_oom_opener/gc_decrement_and_collect.lua"
    ))
});

fn hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn member(h: [u8; 32], object_id: u64) -> String {
    ExtRelease::retirement(h, object_id).member()
}

fn metrics() -> Arc<Metrics> {
    Arc::new(crate::common::metrics_scrape_only("b-pa1-oom-collection"))
}

fn store(tv: &TestValkey, metrics: &Arc<Metrics>) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(metrics))
}

/// A release-budget gate at [`LARGE_BUDGET`] over `client`, so a collection
/// sets the upload hold.
fn large_gate(
    client: &live_edit::storage::client::ValkeyClient,
    keys: &KeySchema,
) -> Arc<ExtReleaseGate> {
    Arc::new(ExtReleaseGate::new(
        client.clone(),
        keys,
        LARGE_BUDGET,
        Duration::from_secs(30),
        None,
    ))
}

/// Store `len` bytes under `h` (one reference each call).
async fn put(store: &ValkeyExternalValueStore, h: [u8; 32], len: usize) {
    store
        .store_values(&[(&h, vec![0x5a; len].as_slice())])
        .await
        .expect("store_values must succeed");
}

async fn release(store: &ValkeyExternalValueStore, h: [u8; 32], object_id: u64) {
    store
        .schedule_release(&[ExtRelease::retirement(h, object_id)])
        .await
        .expect("schedule_release must succeed");
}

/// Plant a collection request, as a replica that saw `-OOM` does: one
/// `gc_oom_signal.lua` on the request key, labelled `by`, 10 s.
async fn request_collection(tv: &TestValkey, by: &str) {
    let _: i64 = GC_OOM_SIGNAL_SCRIPT
        .evalsha_with_reload(
            tv.client().pool.next(),
            vec![tv.key_schema().ext_oom_request()],
            vec![by.to_owned(), "10000".to_owned()],
        )
        .await
        .expect("gc_oom_signal must set the request");
}

async fn request_label(tv: &TestValkey) -> Option<String> {
    tv.client()
        .pool
        .get(tv.key_schema().ext_oom_request())
        .await
        .expect("GET oom:request")
}

async fn hold_present(tv: &TestValkey) -> bool {
    tv.client()
        .pool
        .exists(tv.key_schema().ext_oom_upload_hold())
        .await
        .expect("EXISTS oom:upload_hold")
}

/// One leader pass with a request check, as the GC loop runs it.
async fn leader_pass(
    store: &ValkeyExternalValueStore,
    state: &mut GcPassState,
) -> live_edit::storage::external_values::GcBatchResult {
    store
        .process_pending(&GcPolicy::PRODUCTION, state)
        .await
        .expect("the leader's pass must succeed")
}

/// `gc_claim_pending.lua` for `m`, as a pass claims it: the claim time in ms.
async fn claim(tv: &TestValkey, m: &str) -> i64 {
    let keys = tv.key_schema();
    let claim_ms: i64 = GC_CLAIM_PENDING_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_pending(),
                keys.ext_inflight(),
                keys.ext_pending_age(),
                keys.ext_pending_bytes(),
                keys.ext_pending_total(),
                keys.ext_settled(),
            ],
            vec![m.to_owned()],
        )
        .await
        .expect("gc_claim_pending must run");
    assert!(claim_ms > 0, "precondition: {m} claimed");
    claim_ms
}

/// The decrement's KEYS for hash `h`.
fn decrement_keys(keys: &KeySchema, h: [u8; 32]) -> Vec<String> {
    let hex = sha256_to_hex(&h);
    vec![
        keys.ext_meta(&hex),
        keys.ext_data(&hex),
        keys.ext_applied(&hex),
    ]
}

/// `gc_decrement_and_collect.lua` for member `m` of hash `h`: `(action, n,
/// freed_bytes)`, or the store's error.
async fn decrement(
    tv: &TestValkey,
    h: [u8; 32],
    m: &str,
    claim_ms: i64,
) -> Result<(String, i64, i64), fred::error::Error> {
    GC_DECREMENT_AND_COLLECT_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            decrement_keys(tv.key_schema(), h),
            vec![
                m.to_owned(),
                claim_ms.to_string(),
                GC_DECREMENT_FENCE_MS.to_string(),
            ],
        )
        .await
}

async fn ref_count(tv: &TestValkey, h: [u8; 32]) -> Option<i64> {
    tv.client()
        .pool
        .hget(tv.key_schema().ext_meta(&sha256_to_hex(&h)), "ref_count")
        .await
        .expect("HGET ref_count")
}

async fn data_present(tv: &TestValkey, h: [u8; 32]) -> bool {
    tv.client()
        .pool
        .exists(tv.key_schema().ext_data(&sha256_to_hex(&h)))
        .await
        .expect("EXISTS ext:data")
}

async fn meta_present(tv: &TestValkey, h: [u8; 32]) -> bool {
    tv.client()
        .pool
        .exists(tv.key_schema().ext_meta(&sha256_to_hex(&h)))
        .await
        .expect("EXISTS ext:meta")
}

async fn record_of(tv: &TestValkey, h: [u8; 32], m: &str) -> Option<f64> {
    tv.client()
        .pool
        .zscore(tv.key_schema().ext_applied(&sha256_to_hex(&h)), m)
        .await
        .expect("ZSCORE ext:applied")
}

#[cfg(feature = "test-fault-injection")]
async fn in_inflight(tv: &TestValkey, m: &str) -> bool {
    let score: Option<f64> = tv
        .client()
        .pool
        .zscore(tv.key_schema().ext_inflight(), m)
        .await
        .expect("ZSCORE inflight");
    score.is_some()
}

/// `outcome → value` of `live_edit_gc_oom_collections_total`, zero-seeded.
fn collections(metrics: &Metrics) -> BTreeMap<&'static str, u64> {
    let text = render(metrics);
    OUTCOMES
        .iter()
        .map(|outcome| {
            let v = sample_value(
                &text,
                "live_edit_gc_oom_collections_total",
                &[("outcome", outcome)],
            )
            .unwrap_or_else(|| panic!("outcome={outcome} is zero-seeded"));
            (*outcome, v as u64)
        })
        .collect()
}

/// [`collections`] with the given cells non-zero and every other at zero.
fn only(cells: &[(&'static str, u64)]) -> BTreeMap<&'static str, u64> {
    OUTCOMES
        .iter()
        .map(|outcome| {
            let v = cells
                .iter()
                .find(|(o, _)| o == outcome)
                .map_or(0, |(_, v)| *v);
            (*outcome, v)
        })
        .collect()
}

fn freed_bytes_total(metrics: &Metrics) -> u64 {
    sample_value(&render(metrics), "live_edit_gc_oom_freed_bytes_total", &[])
        .expect("live_edit_gc_oom_freed_bytes_total is zero-seeded") as u64
}

/// A fred error carrying Valkey's `-OOM` reply.
#[cfg(feature = "test-fault-injection")]
fn oom_error() -> fred::error::Error {
    fred::error::Error::new(fred::error::ErrorKind::Unknown, OOM_REPLY)
}

/// A proxy rule that holds the replies on the connection carrying the first
/// `EVALSHA` naming `meta_key` — the member's decrement — until `gate` opens.
fn hold_the_decrement(meta_key: String, gate: ReplyGate) -> ProxyRule {
    let held = std::sync::atomic::AtomicBool::new(false);
    Arc::new(move |args: &[Vec<u8>]| {
        let name = args.first()?.to_ascii_uppercase();
        (name == b"EVALSHA"
            && args.iter().any(|a| a == meta_key.as_bytes())
            && !held.swap(true, Ordering::SeqCst))
        .then(|| ProxyAction::HoldReplies(gate.clone()))
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// ST1 — the opener changes no result
// ═════════════════════════════════════════════════════════════════════════════

/// What one differential case plants before the decrement runs.
#[derive(Debug, Clone, Copy)]
struct Planted {
    /// `ref_count` of a hash meta; `None` for no meta.
    meta: Option<&'static str>,
    /// The meta key is a string, not a hash.
    meta_wrong_type: bool,
    /// The data key: a string of this many bytes; `None` for none.
    data: Option<usize>,
    /// The data key is a hash, not a string.
    data_wrong_type: bool,
    /// The replay record's score for the case's member, relative to the
    /// claim (ms); `None` for no record of it.
    record_offset: Option<i64>,
    /// Another member already on the replay record.
    other_member: bool,
    /// The applied key is a string, not a sorted set.
    applied_wrong_type: bool,
    /// The claim is older than the fence.
    stale: bool,
}

const NOTHING: Planted = Planted {
    meta: None,
    meta_wrong_type: false,
    data: None,
    data_wrong_type: false,
    record_offset: None,
    other_member: false,
    applied_wrong_type: false,
    stale: false,
};

/// Every state the decrement branches on: meta absent or present at
/// ref −1, 0, 1, 2 and 5, malformed or wrong-typed; data absent, present or
/// wrong-typed; a replay record that the claim raises, keeps or that names
/// another member, or a wrong-typed one; and a stale claim.
fn differential_cases() -> Vec<(&'static str, Planted)> {
    let with = |meta: &'static str| Planted {
        meta: Some(meta),
        data: Some(BLOB),
        ..NOTHING
    };
    vec![
        ("meta and data absent (underflow)", NOTHING),
        ("ref -1", with("-1")),
        ("ref 0", with("0")),
        ("ref 1", with("1")),
        ("ref 2", with("2")),
        ("ref 5", with("5")),
        (
            "ref 1, no data",
            Planted {
                data: None,
                ..with("1")
            },
        ),
        (
            "ref 1, wrong-typed data",
            Planted {
                data: None,
                data_wrong_type: true,
                ..with("1")
            },
        ),
        ("malformed ref", with("not-a-number")),
        (
            "wrong-typed meta",
            Planted {
                meta_wrong_type: true,
                data: Some(BLOB),
                ..NOTHING
            },
        ),
        (
            "wrong-typed applied",
            Planted {
                applied_wrong_type: true,
                ..with("1")
            },
        ),
        (
            "record below the claim (raised)",
            Planted {
                record_offset: Some(-1_000),
                ..with("2")
            },
        ),
        (
            "record above the claim (kept)",
            Planted {
                record_offset: Some(1_000),
                ..with("2")
            },
        ),
        (
            "record at the claim",
            Planted {
                record_offset: Some(0),
                ..with("2")
            },
        ),
        (
            "record, meta absent",
            Planted {
                record_offset: Some(-1_000),
                ..NOTHING
            },
        ),
        (
            "another member recorded, ref 1",
            Planted {
                other_member: true,
                ..with("1")
            },
        ),
        (
            "stale claim, ref 2",
            Planted {
                stale: true,
                ..with("2")
            },
        ),
    ]
}

/// Plant `p` under hash `h`'s keys.
async fn plant(tv: &TestValkey, h: [u8; 32], p: &Planted, m: &str, claim_ms: i64) {
    let [meta, data, applied]: [String; 3] = decrement_keys(tv.key_schema(), h)
        .try_into()
        .expect("three keys");
    let pool = &tv.client().pool;
    if let Some(rc) = p.meta {
        let _: () = pool
            .hset(
                &meta,
                [("ref_count", rc), ("last_store_ms", "1700000000000")],
            )
            .await
            .expect("HSET meta");
    }
    if p.meta_wrong_type {
        let _: () = pool
            .set(&meta, "not-a-hash", None, None, false)
            .await
            .expect("SET meta");
    }
    if let Some(len) = p.data {
        let _: () = pool
            .set(&data, vec![0x5a_u8; len], None, None, false)
            .await
            .expect("SET data");
    }
    if p.data_wrong_type {
        let _: () = pool
            .hset(&data, [("not", "a-string")])
            .await
            .expect("HSET data");
    }
    if let Some(offset) = p.record_offset {
        let _: () = pool
            .zadd(
                &applied,
                None,
                None,
                false,
                false,
                ((claim_ms + offset) as f64, m),
            )
            .await
            .expect("ZADD applied");
    }
    if p.other_member {
        let _: () = pool
            .zadd(
                &applied,
                None,
                None,
                false,
                false,
                (12_345.0, "other-member"),
            )
            .await
            .expect("ZADD applied (another member)");
    }
    if p.applied_wrong_type {
        let _: () = pool
            .set(&applied, "not-a-zset", None, None, false)
            .await
            .expect("SET applied");
    }
}

/// A key's type and whole content, comparable across two hashes' keys.
async fn dump(tv: &TestValkey, key: &str) -> String {
    let pool = &tv.client().pool;
    let kind: String = pool
        .custom(
            CustomCommand::new_static("TYPE", ClusterHash::FirstKey, false),
            vec![Value::from(key)],
        )
        .await
        .expect("TYPE");
    match kind.as_str() {
        "none" => "none".to_owned(),
        "string" => {
            let bytes: Vec<u8> = pool.get(key).await.expect("GET");
            format!("string {bytes:?}")
        }
        "hash" => {
            let fields: BTreeMap<String, String> = pool.hgetall(key).await.expect("HGETALL");
            format!("hash {fields:?}")
        }
        "zset" => {
            let entries: Vec<(String, f64)> = pool
                .zrange(key, 0, -1, None, false, None, true)
                .await
                .expect("ZRANGE WITHSCORES");
            format!("zset {entries:?}")
        }
        other => panic!("unexpected type {other} at {key}"),
    }
}

async fn dump_all(tv: &TestValkey, h: [u8; 32]) -> Vec<String> {
    let mut out = Vec::new();
    for key in decrement_keys(tv.key_schema(), h) {
        out.push(dump(tv, &key).await);
    }
    out
}

/// ST1: the out-of-memory opener (`ZREM applied ":oom-gate"`) changes no
/// result. Over every state the decrement branches on, the current script
/// and the pre-b.pa1 one (vendored verbatim) run on two copies of the same
/// planted state, with the same member and claim: both answer the same
/// `(action, n)` (for `stale`, an age past the fence from each), or both
/// fail, and leave the same three keys behind, type and content. The
/// current script's third element is the deleted blob's length on a
/// deleting branch with a string blob, and 0 otherwise.
///
/// This is the at-most-once guard's differential: the opener runs inside
/// the replay-record branch and before the decrement, so a write it made (a
/// member it added, a key it created) would show in the applied key's dump.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_oom_opener_leaves_every_decrement_reply_and_stored_state_unchanged() {
    let tv = TestValkey::setup().await;
    let m = member(hash(0xEE), OBJ);
    let now = valkey_now_ms(&tv).await as i64;

    for (i, (case, planted)) in differential_cases().into_iter().enumerate() {
        let claim_ms = if planted.stale {
            now - GC_DECREMENT_FENCE_MS as i64 - 5_000
        } else {
            now
        };
        let (old_h, new_h) = (hash(2 * i as u8 + 1), hash(2 * i as u8 + 2));
        plant(&tv, old_h, &planted, &m, claim_ms).await;
        plant(&tv, new_h, &planted, &m, claim_ms).await;
        assert_eq!(
            dump_all(&tv, old_h).await,
            dump_all(&tv, new_h).await,
            "{case}: precondition: the two copies are planted alike"
        );
        let argv = vec![
            m.clone(),
            claim_ms.to_string(),
            GC_DECREMENT_FENCE_MS.to_string(),
        ];

        let old: Result<(String, i64), _> = PRE_OPENER_DECREMENT
            .evalsha_with_reload(
                tv.client().maintenance_pool.next(),
                decrement_keys(tv.key_schema(), old_h),
                argv.clone(),
            )
            .await;
        let new: Result<(String, i64, i64), _> = GC_DECREMENT_AND_COLLECT_SCRIPT
            .evalsha_with_reload(
                tv.client().maintenance_pool.next(),
                decrement_keys(tv.key_schema(), new_h),
                argv,
            )
            .await;

        match (&old, &new) {
            (Ok((old_action, old_n)), Ok((new_action, new_n, freed))) => {
                assert_eq!(new_action, old_action, "{case}: the same action");
                if old_action == "stale" {
                    assert!(
                        *old_n > GC_DECREMENT_FENCE_MS as i64
                            && *new_n > GC_DECREMENT_FENCE_MS as i64,
                        "{case}: both answer an age past the fence; got {old_n} and {new_n}"
                    );
                } else {
                    assert_eq!(new_n, old_n, "{case}: the same n");
                }
                let deleting = new_action.starts_with("deleted");
                let want_freed = match planted.data {
                    Some(len) if deleting && !planted.data_wrong_type => len as i64,
                    _ => 0,
                };
                assert_eq!(
                    *freed, want_freed,
                    "{case}: freed_bytes is the deleted blob's length, else 0"
                );
            }
            (Err(_), Err(_)) => {}
            _ => panic!("{case}: the two scripts disagree: old {old:?}, new {new:?}"),
        }
        assert_eq!(
            dump_all(&tv, new_h).await,
            dump_all(&tv, old_h).await,
            "{case}: the opener leaves the stored state the old script leaves"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ST4 — one collection per request, rate-limited
// ═════════════════════════════════════════════════════════════════════════════

/// ST4: the leader runs one collection per request, and backs off after one
/// that freed nothing.
///
/// 1. A request with nothing owed runs a collection that ends
///    `nothing_to_collect`; the request is taken (`GETDEL`).
/// 2. A value released just now and a new request at once: the last
///    collection freed nothing, so the next waits 30 s
///    (`OOM_COLLECTION_RETRY_AFTER_FUTILE`); the request is consumed, and the
///    ordinary pass that runs instead leaves the young release to
///    `PRODUCTION`'s 5 s floor.
/// 3. A new leadership (a fresh pass state) collects at once: the request
///    runs a collection, which takes the young release at any age —
///    `freed`, and the blob's bytes counted.
/// 4. With no request the next pass is ordinary again: one collection per
///    request.
/// 5. After a collection that freed, a request one second later
///    (`OOM_COLLECTION_RETRY_AFTER_FREED`) runs the next.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_leader_runs_one_oom_collection_per_request_and_backs_off_after_a_futile_one() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let (first, second) = (hash(0x41), hash(0x42));
    let mut state = GcPassState::new();

    request_collection(&tv, "replica-a").await;
    let futile = leader_pass(&s, &mut state).await;
    assert_eq!(futile.processed, 0);
    assert_eq!(collections(&m), only(&[("nothing_to_collect", 1)]));
    assert_eq!(
        request_label(&tv).await,
        None,
        "the leader took the request"
    );

    put(&s, first, BLOB).await;
    release(&s, first, OBJ).await;
    request_collection(&tv, "replica-b").await;
    let turned_away = leader_pass(&s, &mut state).await;
    assert_eq!(
        turned_away.processed, 0,
        "no collection 30 s after a futile one: the ordinary pass keeps the young release"
    );
    assert_eq!(collections(&m), only(&[("nothing_to_collect", 1)]));
    assert_eq!(
        request_label(&tv).await,
        None,
        "a request the rate limit turns away is consumed all the same"
    );
    assert_eq!(ref_count(&tv, first).await, Some(1));

    let mut new_leadership = GcPassState::new();
    request_collection(&tv, "replica-c").await;
    let collected = leader_pass(&s, &mut new_leadership).await;
    assert_eq!(
        (
            collected.processed,
            collected.deleted,
            collected.freed_bytes
        ),
        (1, 1, BLOB as u64),
        "a new leadership collects at once, at any age"
    );
    assert!(!data_present(&tv, first).await);
    assert_eq!(
        collections(&m),
        only(&[("nothing_to_collect", 1), ("freed", 1)])
    );
    assert_eq!(freed_bytes_total(&m), BLOB as u64);

    put(&s, second, BLOB).await;
    release(&s, second, OBJ).await;
    let ordinary = leader_pass(&s, &mut new_leadership).await;
    assert_eq!(ordinary.processed, 0, "no request, no collection");
    assert_eq!(
        collections(&m),
        only(&[("nothing_to_collect", 1), ("freed", 1)])
    );

    tokio::time::sleep(OOM_COLLECTION_RETRY_AFTER_FREED + Duration::from_millis(100)).await;
    request_collection(&tv, "replica-d").await;
    let next = leader_pass(&s, &mut new_leadership).await;
    assert_eq!(
        (next.deleted, next.freed_bytes),
        (1, BLOB as u64),
        "a second after a collection that freed, the next request collects"
    );
    assert_eq!(
        collections(&m),
        only(&[("nothing_to_collect", 1), ("freed", 2)])
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Data safety
// ═════════════════════════════════════════════════════════════════════════════

/// A collection decrements a value with two references once and keeps it:
/// it frees only what is owed. Counted `freed_nothing`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_oom_collection_never_deletes_a_value_still_referenced() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0x51);
    put(&s, h, BLOB).await;
    put(&s, h, BLOB).await;
    release(&s, h, OBJ).await;

    request_collection(&tv, "replica-a").await;
    let collected = leader_pass(&s, &mut GcPassState::new()).await;

    assert_eq!(
        (
            collected.processed,
            collected.deleted,
            collected.freed_bytes
        ),
        (1, 0, 0)
    );
    assert_eq!(ref_count(&tv, h).await, Some(1), "one reference taken");
    assert!(data_present(&tv, h).await, "the value in use is kept");
    assert_eq!(collections(&m), only(&[("freed_nothing", 1)]));
}

/// A collection collects at any age but still honours read holds: a release
/// enqueued after a live hold began stays while the hold lives
/// (`nothing_to_collect`), and is collected by a collection after the hold
/// is released. The hold margin is 0 here, so the boundary is the hold's
/// start.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_oom_collection_leaves_a_release_a_live_read_hold_protects() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m).with_hold_margin_for_test(Duration::ZERO);
    let h = hash(0x52);
    put(&s, h, BLOB).await;
    let hold = ReadHold::take(tv.client(), tv.key_schema(), OBJ)
        .await
        .expect("the read hold must be taken");
    release(&s, h, OBJ).await;

    request_collection(&tv, "replica-a").await;
    let held = leader_pass(&s, &mut GcPassState::new()).await;
    assert_eq!(
        held.processed, 0,
        "a release enqueued after a live hold's start is out of the collection's selection"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));
    assert!(data_present(&tv, h).await);
    assert_eq!(collections(&m), only(&[("nothing_to_collect", 1)]));

    hold.release().await;
    request_collection(&tv, "replica-b").await;
    let collected = leader_pass(&s, &mut GcPassState::new()).await;
    assert_eq!((collected.processed, collected.deleted), (1, 1));
    assert!(!data_present(&tv, h).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// ST5 — the upload hold
// ═════════════════════════════════════════════════════════════════════════════

/// Whether `args` is an `EVALSHA` of `gc_oom_signal.lua` naming `key`.
fn is_signal_evalsha_on(args: &[Vec<u8>], key: &str) -> bool {
    let sha = GC_OOM_SIGNAL_SCRIPT.sha1().to_string();
    args.first()
        .is_some_and(|name| name.eq_ignore_ascii_case(b"EVALSHA"))
        && args
            .get(1)
            .is_some_and(|a| String::from_utf8_lossy(a).eq_ignore_ascii_case(&sha))
        && args.iter().any(|a| a == key.as_bytes())
}

/// The commands a collection sends on the upload hold key, seen by a proxy
/// that only watches: the hold's sets (`EVALSHA`s of `gc_oom_signal.lua` on
/// the key, less the re-send after a `NOSCRIPT` reload — one `SCRIPT LOAD` of
/// the script each, as a concurrent test's `SCRIPT FLUSH` can cause) and its
/// `DEL`s.
#[derive(Clone, Default)]
struct HoldCommands {
    evalshas: Arc<AtomicUsize>,
    loads: Arc<AtomicUsize>,
    dels: Arc<AtomicUsize>,
}

impl HoldCommands {
    /// Note one client→server frame.
    fn see(&self, args: &[Vec<u8>], hold_key: &str) {
        let name = args
            .first()
            .map(|a| a.to_ascii_uppercase())
            .unwrap_or_default();
        if is_signal_evalsha_on(args, hold_key) {
            self.evalshas.fetch_add(1, Ordering::SeqCst);
        } else if name == b"SCRIPT"
            && args
                .get(2)
                .is_some_and(|body| String::from_utf8_lossy(body).contains("gc_oom_signal"))
        {
            self.loads.fetch_add(1, Ordering::SeqCst);
        } else if name == b"DEL" && args.iter().skip(1).any(|a| a == hold_key.as_bytes()) {
            self.dels.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// `(sets, dels)` so far.
    fn counts(&self) -> (usize, usize) {
        (
            self.evalshas.load(Ordering::SeqCst) - self.loads.load(Ordering::SeqCst),
            self.dels.load(Ordering::SeqCst),
        )
    }
}

/// The upload hold is written only when a release budget gates uploads: a
/// collection on a store without a gate neither sets nor deletes the hold and
/// still collects; one with a gate sets the hold once and deletes it once.
/// Counted through a proxy that only watches, robust to a script reload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_oom_collection_holds_uploads_only_with_a_release_budget() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let hold_key = tv.key_schema().ext_oom_upload_hold();
    let seen = HoldCommands::default();
    let rule: ProxyRule = {
        let (hold_key, seen) = (hold_key.clone(), seen.clone());
        Arc::new(move |args: &[Vec<u8>]| {
            seen.see(args, &hold_key);
            None
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;
    let direct = store(&tv, &m);
    let (first, second) = (hash(0x61), hash(0x62));

    put(&direct, first, BLOB).await;
    release(&direct, first, OBJ).await;
    let ungated = ValkeyExternalValueStore::new(client.clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(&m));
    request_collection(&tv, "replica-a").await;
    let collected = leader_pass(&ungated, &mut GcPassState::new()).await;
    assert_eq!(collected.deleted, 1, "the collection runs without a gate");
    assert_eq!(seen.counts(), (0, 0), "no release budget, no upload hold");

    put(&direct, second, BLOB).await;
    release(&direct, second, OBJ).await;
    let gated = ungated.with_ext_release_gate(large_gate(&client, tv.key_schema()));
    request_collection(&tv, "replica-b").await;
    let collected = leader_pass(&gated, &mut GcPassState::new()).await;
    assert_eq!(collected.deleted, 1);
    assert_eq!(
        seen.counts(),
        (1, 1),
        "with a release budget the collection sets the hold once and deletes it once"
    );
    assert!(
        !hold_present(&tv).await,
        "the hold is gone after the collection"
    );
    client.quit().await;
}

/// The collection repeats its passes while one reports more work: 101
/// released values — one more than a pass's batch — and one request are
/// collected in that one collection, counted `freed` once with every
/// value's bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_oom_collection_repeats_its_passes_until_the_released_values_are_all_collected() {
    /// One more than a pass's batch (`GC_BATCH_SIZE`, 100).
    const RELEASED: u8 = 101;
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    for byte in 1..=RELEASED {
        put(&s, hash(byte), BLOB).await;
        release(&s, hash(byte), OBJ).await;
    }

    request_collection(&tv, "replica-a").await;
    let collected = leader_pass(&s, &mut GcPassState::new()).await;

    assert_eq!(
        (collected.deleted, collected.freed_bytes),
        (usize::from(RELEASED), u64::from(RELEASED) * BLOB as u64),
        "every released value is collected in the one collection"
    );
    for byte in 1..=RELEASED {
        assert!(
            !data_present(&tv, hash(byte)).await,
            "value {byte} collected"
        );
    }
    assert_eq!(collections(&m), only(&[("freed", 1)]), "one collection");
    assert_eq!(freed_bytes_total(&m), u64::from(RELEASED) * BLOB as u64);
}

/// ST5, the error exit: while a collection runs — its decrement's reply held
/// at the proxy — the upload hold stands; the completion then fails
/// (`test-fault-injection`, a non-transient error at site 2), the pass ends
/// in that error, and the hold is gone by the time the call returns,
/// counted `failed` (the pass's counts are lost with its error).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_upload_hold_stands_while_an_oom_collection_runs_and_is_cleared_when_it_fails() {
    use live_edit::storage::external_values::PendingFault;

    let tv = TestValkey::setup().await;
    let m = metrics();
    let keys = tv.key_schema().clone();
    let h = hash(0x63);
    let direct = store(&tv, &m);
    put(&direct, h, BLOB).await;
    release(&direct, h, OBJ).await;
    // Loaded up front, so the held `EVALSHA` is the one that runs the script.
    GC_DECREMENT_AND_COLLECT_SCRIPT
        .load(tv.client().pool.next())
        .await
        .expect("the decrement script must load");
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        hold_the_decrement(keys.ext_meta(&sha256_to_hex(&h)), gate.clone()),
    )
    .await;
    let client = client_through(&proxy).await;
    let faults = Arc::new(Mutex::new(VecDeque::from([(
        2,
        PendingFault::Fail(fred::error::Error::new(
            fred::error::ErrorKind::Protocol,
            "synthetic gc_complete failure",
        )),
    )])));
    let s = ValkeyExternalValueStore::new(client.clone(), keys.clone())
        .with_metrics(Arc::clone(&m))
        .with_ext_release_gate(large_gate(&client, &keys))
        .with_process_pending_fault_injector_for_test(faults);

    request_collection(&tv, "replica-a").await;
    let run = tokio::spawn(async move {
        s.process_pending(&GcPolicy::PRODUCTION, &mut GcPassState::new())
            .await
    });
    gate.reply_held().await;
    assert!(
        hold_present(&tv).await,
        "the upload hold stands while the collection runs"
    );
    gate.open();
    let ended = tokio::time::timeout(WAIT, run)
        .await
        .expect("the collection must end")
        .expect("the collection must not panic");

    assert!(
        matches!(ended, Err(live_edit::error::ServiceError::Internal(_))),
        "the completion's non-transient error ends the pass; got {ended:?}"
    );
    assert!(
        !hold_present(&tv).await,
        "the hold is deleted before a collection that failed returns"
    );
    assert_eq!(collections(&m), only(&[("failed", 1)]));
    client.quit().await;
}

/// ST5, the cancelled exit: a collection whose future is dropped mid-pass —
/// its decrement's reply held at the proxy, the hold standing — leaves no
/// hold behind: the hold's drop guard spawns the `DEL`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_upload_hold_is_cleared_when_an_oom_collection_is_cancelled() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let keys = tv.key_schema().clone();
    let h = hash(0x64);
    let direct = store(&tv, &m);
    put(&direct, h, BLOB).await;
    release(&direct, h, OBJ).await;
    GC_DECREMENT_AND_COLLECT_SCRIPT
        .load(tv.client().pool.next())
        .await
        .expect("the decrement script must load");
    let gate = ReplyGate::new();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        hold_the_decrement(keys.ext_meta(&sha256_to_hex(&h)), gate.clone()),
    )
    .await;
    let client = client_through(&proxy).await;
    let s = ValkeyExternalValueStore::new(client.clone(), keys.clone())
        .with_metrics(Arc::clone(&m))
        .with_ext_release_gate(large_gate(&client, &keys));

    request_collection(&tv, "replica-a").await;
    let run = tokio::spawn(async move {
        s.process_pending(&GcPolicy::PRODUCTION, &mut GcPassState::new())
            .await
    });
    gate.reply_held().await;
    assert!(hold_present(&tv).await, "precondition: the hold stands");

    run.abort();
    let _ = run.await;
    tokio::time::timeout(WAIT, async {
        while hold_present(&tv).await {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the dropped collection's guard must delete the upload hold");
    gate.open();
}

// ═════════════════════════════════════════════════════════════════════════════
// `refused`, and a refused re-claim
// ═════════════════════════════════════════════════════════════════════════════

/// The message prefix of the once-per-process WARN of a `refused`
/// collection.
#[cfg(feature = "test-fault-injection")]
const REFUSED_ONCE_PREFIX: &str = "the backing store refused the GC's own clean-up scripts";

/// The message prefix of every per-collection WARN.
#[cfg(feature = "test-fault-injection")]
const COLLECTION_PREFIX: &str = "the backing store refused a write for lack of memory;";

/// The captured events whose message starts with `prefix`.
#[cfg(feature = "test-fault-injection")]
fn starting_with<'a>(
    captured: &'a crate::grpc::log_capture::Captured,
    prefix: &str,
) -> Vec<&'a serde_json::Value> {
    captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|msg| msg.starts_with(prefix))
        })
        .collect()
}

/// The store refusing a collection's own scripts `-OOM` makes it `refused`,
/// not `freed_nothing`, by either leg, and the cause is named once per
/// process:
///
/// 1. **The decrement** (`test-fault-injection`, site 1): the store did not
///    run the script past its opener on a full store. The collection frees
///    nothing; the member stays in `inflight` with its value and reference
///    untouched. The first `refused` collection in the process also logs the
///    WARN naming the likely cause.
/// 2. **The upload hold**: a second collection, on a store with a release
///    budget behind a proxy that answers the hold's `gc_oom_signal.lua`
///    `-OOM`. It still runs — it collects without holding uploads back, here
///    decrementing a value with two references, so nothing is deleted — and
///    no hold is left behind. It logs only its own per-collection WARN, which
///    carries the hold's failure (`held_uploads=false`, `hold_error`): the
///    failed hold has no line of its own, and the cause was named already.
///    The first collection, with no release budget, sets no hold, and its
///    WARN has no `hold_error`.
///
/// The once-per-process WARN rides a process-wide flag, so this is the one
/// non-disruptive test that runs a `refused` collection: a second such test
/// in the binary could spend the flag first under a plain `cargo test`, where
/// tests share a process. Under nextest each test has a process of its own.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oom_collections_refused_by_the_store_count_refused_and_warn_once() {
    use crate::grpc::log_capture::capture;
    use live_edit::storage::external_values::PendingFault;

    let tv = TestValkey::setup().await;
    let m = metrics();
    let keys = tv.key_schema().clone();
    // Site 1 of the first member of a pass is its decrement.
    let faults = Arc::new(Mutex::new(VecDeque::from([(
        1,
        PendingFault::Fail(oom_error()),
    )])));
    let s = store(&tv, &m).with_process_pending_fault_injector_for_test(Arc::clone(&faults));
    let (first, second) = (hash(0x71), hash(0x72));
    put(&s, first, BLOB).await;
    release(&s, first, OBJ).await;

    request_collection(&tv, "replica-a").await;
    let (refused, logs) = capture(leader_pass(&s, &mut GcPassState::new())).await;
    assert_eq!(
        (refused.processed, refused.deleted, refused.oom_refused),
        (1, 0, 1)
    );
    assert_eq!(collections(&m), only(&[("refused", 1)]));
    assert_eq!(freed_bytes_total(&m), 0);
    assert!(
        in_inflight(&tv, &member(first, OBJ)).await,
        "left for a later pass"
    );
    assert_eq!(ref_count(&tv, first).await, Some(1));
    assert!(data_present(&tv, first).await);
    let once = starting_with(&logs, REFUSED_ONCE_PREFIX);
    assert_eq!(
        once.len(),
        1,
        "the first refused collection names the likely cause once; capture:\n{}",
        logs.text
    );
    assert_eq!(once[0]["level"], "WARN");
    let per_collection = starting_with(&logs, COLLECTION_PREFIX);
    assert_eq!(per_collection.len(), 1, "capture:\n{}", logs.text);
    assert_eq!(per_collection[0]["fields"]["outcome"], "refused");
    assert!(
        per_collection[0]["fields"].get("hold_error").is_none(),
        "no release budget, no hold to fail: `hold_error` is absent; event: {}",
        per_collection[0]
    );

    assert!(
        faults.lock().unwrap().is_empty(),
        "the decrement's refusal was taken"
    );

    let hold_key = keys.ext_oom_upload_hold();
    let refused_sets = Arc::new(AtomicUsize::new(0));
    let rule: ProxyRule = {
        let (hold_key, refused_sets) = (hold_key.clone(), Arc::clone(&refused_sets));
        Arc::new(move |args: &[Vec<u8>]| {
            is_signal_evalsha_on(args, &hold_key).then(|| {
                refused_sets.fetch_add(1, Ordering::SeqCst);
                ProxyAction::Eval(error_reply_script(OOM_REPLY))
            })
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;
    let gated = ValkeyExternalValueStore::new(client.clone(), keys.clone())
        .with_metrics(Arc::clone(&m))
        .with_ext_release_gate(large_gate(&client, &keys));
    put(&s, second, BLOB).await;
    put(&s, second, BLOB).await;
    release(&s, second, OBJ).await;
    request_collection(&tv, "replica-b").await;
    let (hold_refused, logs) = capture(leader_pass(&gated, &mut GcPassState::new())).await;
    assert_eq!(
        refused_sets.load(Ordering::SeqCst),
        1,
        "precondition: the hold's write was refused"
    );
    assert_eq!(
        (
            hold_refused.processed,
            hold_refused.deleted,
            hold_refused.oom_refused
        ),
        (1, 0, 0),
        "the member is still processed, collected without the hold"
    );
    assert_eq!(ref_count(&tv, second).await, Some(1), "one reference taken");
    assert!(!hold_present(&tv).await, "no upload hold remains");
    assert_eq!(
        collections(&m),
        only(&[("refused", 2)]),
        "a refused hold makes the collection `refused`, not `freed_nothing`"
    );
    assert!(
        starting_with(&logs, REFUSED_ONCE_PREFIX).is_empty(),
        "the cause is named once per process; capture:\n{}",
        logs.text
    );
    let per_collection = starting_with(&logs, COLLECTION_PREFIX);
    assert_eq!(per_collection.len(), 1, "capture:\n{}", logs.text);
    let fields = &per_collection[0]["fields"];
    assert_eq!(fields["outcome"], "refused");
    assert_eq!(
        fields["held_uploads"], false,
        "the collection ran without the hold; event: {}",
        per_collection[0]
    );
    assert!(
        fields["hold_error"]
            .as_str()
            .is_some_and(|error| error.contains(OOM_REPLY)),
        "the per-collection WARN carries why the hold could not be set; event: {}",
        per_collection[0]
    );
    let warnings: Vec<_> = logs
        .events
        .iter()
        .filter(|e| e["level"] == "WARN" || e["level"] == "ERROR")
        .collect();
    assert_eq!(
        warnings.len(),
        1,
        "the failed hold has no line of its own: the per-collection WARN is the only one; \
         capture:\n{}",
        logs.text
    );
    client.quit().await;
}

/// F1: a re-claim of one of this build's own stale claims that the store
/// refuses `-OOM` (`test-fault-injection`, site 0 of the re-claimed member)
/// skips that member, not the pass: the pending member beside it is still
/// collected, the skip is counted (`oom_skipped`), and the stale claim stays
/// in `inflight` as it was, for a later pass — which re-claims it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_reclaim_skips_the_member_and_the_pass_goes_on() {
    use live_edit::storage::external_values::PendingFault;

    let tv = TestValkey::setup().await;
    let m = metrics();
    let (stale, pending) = (hash(0x81), hash(0x82));
    let plain = store(&tv, &m);
    put(&plain, stale, BLOB).await;
    release(&plain, stale, OBJ).await;
    let stale_member = member(stale, OBJ);
    claim(&tv, &stale_member).await;
    plant_own_claim(&tv, &stale_member, STALE_OWN_CLAIM_AGE).await;
    let planted_score: f64 = tv
        .client()
        .pool
        .zscore::<Option<f64>, _, _>(tv.key_schema().ext_inflight(), &stale_member)
        .await
        .expect("ZSCORE inflight")
        .expect("precondition: the stale claim is in inflight");
    put(&plain, pending, BLOB).await;
    release(&plain, pending, OBJ).await;
    // The work list is the selected pending members first, then the own stale
    // claims: the re-claim is member 1, its claim site 3 * 1 + 0.
    let faults = Arc::new(Mutex::new(VecDeque::from([(
        3,
        PendingFault::Fail(oom_error()),
    )])));
    let s = store(&tv, &m).with_process_pending_fault_injector_for_test(Arc::clone(&faults));

    let pass = s
        .process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
        .await
        .expect("a re-claim refused -OOM must not end the pass");
    assert!(faults.lock().unwrap().is_empty(), "the refusal was taken");
    assert_eq!(
        (
            pass.processed,
            pass.deleted,
            pass.oom_skipped,
            pass.reclaimed
        ),
        (1, 1, 1, 0),
        "the pending member is collected; the refused re-claim is skipped, not counted"
    );
    assert!(!data_present(&tv, pending).await);
    let score: Option<f64> = tv
        .client()
        .pool
        .zscore(tv.key_schema().ext_inflight(), &stale_member)
        .await
        .expect("ZSCORE inflight");
    assert_eq!(
        score,
        Some(planted_score),
        "the stale claim is left as it was"
    );
    assert_eq!(ref_count(&tv, stale).await, Some(1));

    let later = s
        .process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
        .await
        .expect("the next pass must succeed");
    assert_eq!(
        (later.reclaimed, later.deleted),
        (1, 1),
        "a later pass re-claims and collects it"
    );
    assert!(!data_present(&tv, stale).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// A real -OOM (disruptive): RT1–RT6
// ═════════════════════════════════════════════════════════════════════════════

/// Connections of a disruptive test's fixture.
const POOL_SIZE: u32 = 4;

/// The disruptive store's fixture, checked to run `noeviction`: `None` when
/// the store is not configured (a local run without it).
async fn disruptive_fixture() -> Option<(TestValkey, fred::clients::Client)> {
    let url = disruptive_valkey_url()?;
    let tv = TestValkey::setup_with_url(url, POOL_SIZE).await;
    let admin = tv.client().pool.next().clone();
    assert_noeviction(&admin).await;
    Some((tv, admin))
}

/// Assert the store refuses a plain write `-OOM`: it is full, so what the
/// test shows runs on a full store.
async fn assert_refuses_writes(tv: &TestValkey) {
    let refused: Result<(), fred::error::Error> = tv
        .client()
        .pool
        .set(format!("{}full-probe", tv.prefix()), "x", None, None, false)
        .await;
    let error = refused.expect_err("precondition: the full store refuses a plain SET");
    assert!(
        error.details().starts_with("OOM "),
        "precondition: refused out of memory; got {error:?}"
    );
}

/// RT1: on a full store the decrement of a value's last reference runs to
/// its delete — the opener lets every later command of the script run — and
/// answers `deleted` with the blob's length; meta and data are gone, and the
/// replay record holds the claim.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_gc_decrement_deletes_a_released_blob_on_a_full_store() {
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0x91);
    put(&s, h, BLOB).await;
    release(&s, h, OBJ).await;
    let mem = member(h, OBJ);
    let claim_ms = claim(&tv, &mem).await;

    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let reply = decrement(&tv, h, &mem, claim_ms).await;
    full.restore().await;

    assert_eq!(
        reply.expect("the decrement must run on a full store"),
        ("deleted".to_owned(), 0, BLOB as i64)
    );
    assert!(!meta_present(&tv, h).await && !data_present(&tv, h).await);
    assert_eq!(
        record_of(&tv, h, &mem).await,
        Some(claim_ms as f64),
        "the replay record is written in the same run"
    );
}

/// RT2: on a full store a value with two references takes both owed
/// releases — `decremented`, then `deleted` — and a replay of the first with
/// a newer claim raises its record (the replay branch's opener) without
/// decrementing again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_gc_decrement_applies_a_shared_blobs_releases_on_a_full_store(
) {
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0x92);
    put(&s, h, BLOB).await;
    put(&s, h, BLOB).await;
    release(&s, h, OBJ).await;
    release(&s, h, OBJ_B).await;
    let (first, second) = (member(h, OBJ), member(h, OBJ_B));
    let first_claim = claim(&tv, &first).await;
    let second_claim = claim(&tv, &second).await;

    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let decremented = decrement(&tv, h, &first, first_claim).await;
    let deleted = decrement(&tv, h, &second, second_claim).await;
    let replayed = decrement(&tv, h, &first, first_claim + 1_000).await;
    full.restore().await;

    assert_eq!(
        decremented.expect("the first decrement must run on a full store"),
        ("decremented".to_owned(), 1, 0)
    );
    assert_eq!(
        deleted.expect("the second decrement must run on a full store"),
        ("deleted".to_owned(), 0, BLOB as i64)
    );
    assert_eq!(
        replayed.expect("a replay must run on a full store"),
        ("replayed".to_owned(), 0, 0)
    );
    assert!(!data_present(&tv, h).await);
    assert_eq!(
        record_of(&tv, h, &first).await,
        Some((first_claim + 1_000) as f64),
        "the replay raised the record"
    );
}

/// RT3: on a full store a catch-up's read hold is taken and renewed — the
/// take's and the renewal's opener — and released.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_a_read_hold_is_taken_and_renewed_on_a_full_store() {
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };

    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let taken = ReadHold::take(tv.client(), tv.key_schema(), OBJ).await;
    let renewed = match &taken {
        Ok(hold) => Some(hold.renew().await),
        Err(_) => None,
    };
    let fields: u64 = tv
        .client()
        .pool
        .hlen(tv.key_schema().ext_read_hold())
        .await
        .expect("HLEN read_hold");
    full.restore().await;

    let hold = taken.expect("a read hold must be taken on a full store");
    assert_eq!(fields, 1, "the take wrote its field");
    assert!(
        matches!(renewed, Some(Ok(ReadHoldRenewal::Live))),
        "the hold must be renewed on a full store; got {renewed:?}"
    );
    hold.release().await;
}

/// RT4: on a full store the GC leader lock is acquired when absent, and a
/// same-version peer's lock near expiry is taken over — the `PEXPIRE` opener
/// before each `SET` — so the GC keeps a leader while a shard is full.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_the_gc_leader_lock_is_acquired_and_taken_over_on_a_full_store(
) {
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let me = LeaderIdentity::new("pa1-rt4", "this-build");
    let peer = LeaderIdentity::new("pa1-rt4", "other-instance").gc_lock_value();
    let lock_key = tv.key_schema().gc_leader_lock();
    let acquire = || {
        acquire_gc_leader_lock(
            tv.client(),
            &lock_key,
            &me,
            LEADER_LOCK_TTL_MS,
            GC_LEADER_TAKEOVER_BELOW_MS,
        )
    };

    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let acquired = acquire().await;
    full.restore().await;
    assert_eq!(
        acquired.expect("the acquire must run on a full store"),
        GcLeaderAcquire::Acquired
    );

    let _: () = tv
        .client()
        .pool
        .set(
            &lock_key,
            peer.as_str(),
            Some(fred::types::Expiration::PX(
                (GC_LEADER_TAKEOVER_BELOW_MS / 2) as i64,
            )),
            None,
            false,
        )
        .await
        .expect("plant a same-version peer near expiry");
    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let took_over = acquire().await;
    full.restore().await;
    assert_eq!(
        took_over.expect("the takeover must run on a full store"),
        GcLeaderAcquire::TookOverExpiring {
            displaced: peer.clone()
        }
    );
    let holder: Option<String> = tv.client().pool.get(&lock_key).await.expect("GET lock");
    assert_eq!(holder, Some(me.gc_lock_value()));
}

/// RT5: on a full store a GC pass skips the `{gc}` steps the store refuses —
/// the age stamp of an older build's un-aged member and the re-claim of a
/// stale claim — instead of ending, and still collects a pending member.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_a_gc_pass_on_a_full_store_skips_stamp_and_reclaim_and_still_collects(
) {
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let m = metrics();
    let s = store(&tv, &m);
    let (collectable, unaged, stale) = (hash(0xA1), hash(0xA2), hash(0xA3));
    put(&s, collectable, BLOB).await;
    release(&s, collectable, OBJ).await;
    put(&s, unaged, BLOB).await;
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().ext_pending(), member(unaged, OBJ))
        .await
        .expect("bare SADD (an older build's enqueue, no age)");
    put(&s, stale, BLOB).await;
    release(&s, stale, OBJ).await;
    let stale_member = member(stale, OBJ);
    claim(&tv, &stale_member).await;
    plant_own_claim(&tv, &stale_member, STALE_OWN_CLAIM_AGE).await;

    let full = RestoreMaxmemory::set_full(&admin).await;
    assert_refuses_writes(&tv).await;
    let pass = s
        .process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
        .await;
    full.restore().await;

    let pass = pass.expect("a pass on a full store must skip the refused steps, not end");
    assert_eq!(
        (pass.deleted, pass.oom_skipped, pass.stamped, pass.reclaimed),
        (1, 2, 0, 0),
        "the pending member is collected; the stamp and the re-claim are skipped"
    );
    assert!(!data_present(&tv, collectable).await);
    assert_eq!(ref_count(&tv, unaged).await, Some(1));
    assert_eq!(ref_count(&tv, stale).await, Some(1));
}

/// RT6: end to end. A store filled past its limit by a 16 MiB value released
/// a moment ago (well inside `PRODUCTION`'s 5 s floor) refuses an upload's
/// pre-store `-OOM`; the refusal requests a collection (the client's armed
/// signal writes the request on the full store), the leader's next pass
/// runs it, which frees the value at once, and the retried upload lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `maxmemory` on its own \
            Valkey (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn store_write_refusal_disruptive_oom_collection_frees_a_full_store_and_the_next_upload_lands(
) {
    const LARGE: usize = 16 * 1024 * 1024;
    const HEADROOM: u64 = 4 * 1024 * 1024;
    let Some((tv, admin)) = disruptive_fixture().await else {
        return;
    };
    let m = metrics();
    let s = store(&tv, &m);
    tv.client()
        .store_oom_signal()
        .arm(tv.key_schema(), "replica-rt6");
    let (large, upload) = (hash(0xB1), hash(0xB2));
    put(&s, large, LARGE).await;
    release(&s, large, OBJ).await;

    let used = used_memory(&admin).await;
    let full = RestoreMaxmemory::set(&admin, used - HEADROOM).await;
    assert_refuses_writes(&tv).await;
    let refused = s.store_values(&[(&upload, b"rt6 upload".as_slice())]).await;
    let requested = tokio::time::timeout(WAIT, async {
        loop {
            if let Some(by) = request_label(&tv).await {
                return by;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    let collected = s
        .process_pending(&GcPolicy::PRODUCTION, &mut GcPassState::new())
        .await;
    let retried = s.store_values(&[(&upload, b"rt6 upload".as_slice())]).await;
    full.restore().await;

    assert!(
        refused.is_err(),
        "precondition: the full store refuses the upload's pre-store"
    );
    assert_eq!(
        requested.expect("the refusal must request a collection on the full store"),
        "replica-rt6"
    );
    let collected = collected.expect("the collection must run on the full store");
    assert_eq!(
        (collected.deleted, collected.freed_bytes),
        (1, LARGE as u64),
        "the young release is collected at once"
    );
    assert_eq!(collections(&m), only(&[("freed", 1)]));
    assert_eq!(freed_bytes_total(&m), LARGE as u64);
    retried.expect("the retried upload must land once the collection freed the store");
    assert!(!data_present(&tv, large).await);
}
