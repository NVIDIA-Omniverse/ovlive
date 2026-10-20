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

//! b.vn4 — a cluster node's `-TRYAGAIN` to a multi-key command is transient,
//! and an `EXEC` it refuses ran nothing.
//!
//! A slot mid-migration refuses a command whose keys are split between the
//! migrating slot's source and target with `-TRYAGAIN Multiple keys request
//! during rehashing of slot`, before the command runs; an `EXEC` that meets it
//! has had its transaction discarded first (Valkey 9.0.3 `processCommand`).
//! The shared classifier calls it transient (`storage::client::
//! is_tryagain_reply`), so the `update_layer` commit loop retries it, and
//! `storage::watched_txn::decode` reads it in the `EXEC` position as
//! `Discarded`, never a torn commit (SDD §7.5, §15.3).
//!
//! The deterministic tests stage the store's verbatim reply
//! ([`TRYAGAIN_REPLY`]) with a [`ScriptReplyProxy`] in front of the
//! standalone test store, so fred's reply parsing and the service's
//! classification run as in production; each stages it once, then asserts
//! the proxy was hit once (the vacuity guard). The cluster-gated disruptive
//! test is the real-Valkey provenance of that text: it stages a real
//! `MIGRATING` slot and captures the reply the commit meets.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`update_layer_watch_answered_tryagain_once_retries_and_commits`] | the `WATCH` route: the `WATCH` refused `-TRYAGAIN` once, the call retries and commits |
//! | [`update_layer_commit_evalsha_answered_tryagain_once_retries_and_commits`] | the `lua` route: the commit `EVALSHA` refused `-TRYAGAIN` once, the call retries and commits |
//! | [`update_layer_exec_answered_tryagain_is_discarded_and_retried_without_a_torn_commit_recount`] | the `EXEC` refused `-TRYAGAIN` once (the real connection's transaction discarded): the call retries and commits, and the torn-commit recount (`live_edit_ext_index_recount_total{site="torn_commit"}`) does not move |
//! | `slot_migration_disruptive_…` (cluster, `test-fault-injection`, `#[ignore]`d) | a real `MIGRATING` slot: the fast-path commit and a `WATCH`-route structural attempt each meet the real `-TRYAGAIN`, end `Aborted` carrying its verbatim text, never `Internal`, and both commit once the migration completes |
//!
//! The first two fail on the classifier before b.vn4, which called the reply
//! permanent: the call ends `INTERNAL` at attempt 1. The third fails without
//! `decode`'s `-TRYAGAIN` arm: the `EXEC` reads as a torn commit, which logs
//! at ERROR and recounts the object's reference index.
//!
//! # Wait budget
//!
//! Each deterministic test awaits one update under [`SETTLE`] (10 s), plus
//! setup bounded by fred's 5 s command timeout. The disruptive test awaits
//! four updates under [`SETTLE`] each (40 s), plus its setup commands (fred's
//! 5 s command timeout each) and the `SlotMigration` helper's primitives, each
//! bounded by that helper's 20 s per-command budget (its completion drain by
//! 30 s). Those approach their budget only on a cluster that has stopped
//! answering, where the first one fails and the test panics rather than
//! compounding (the reasoning of `cluster_subscriber_test.rs`'s B2 row).
//! Healthy, every test takes well under a second.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_sum};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::TRYAGAIN_REPLY;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::UPDATE_LAYER_FAST_SCRIPT;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ConnProxyRule, ProxyAction, ScriptReplyProxy,
};

/// Upper bound on one update call. A deadline, not a speed claim.
const SETTLE: Duration = Duration::from_secs(10);

const POOL_SIZE: u32 = 4;
const LAYER: &str = "bvn4-tryagain-layer";
const CREATED_BY: &str = "bvn4-tryagain-user";
const CLIENT_ID: &str = "bvn4-tryagain-writer";

/// `live_edit_ext_index_recount_total`, summed over its outcomes at
/// `site="torn_commit"`: the recount only `update_layer`'s torn-`EXEC` arm
/// runs.
fn torn_commit_recounts(metrics: &Metrics) -> f64 {
    sample_sum(
        &render(metrics),
        "live_edit_ext_index_recount_total",
        &[("site", "torn_commit")],
    )
    .expect("ext_index_recount_total{site=\"torn_commit\"} is seeded")
}

fn is_command(args: &[Vec<u8>], name: &str) -> bool {
    args.first()
        .is_some_and(|a| a.eq_ignore_ascii_case(name.as_bytes()))
}

fn names(args: &[Vec<u8>], key: &str) -> bool {
    args[1..].iter().any(|a| a.as_slice() == key.as_bytes())
}

/// A session and one layer ([`LAYER`]) at `seq = 1`, on `tv`.
async fn layer(tv: &TestValkey, name: &str) -> (String, u64) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), CREATED_BY)
        .await
        .expect("session creation must succeed");
    let object_id = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            LAYER,
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
    (session_id, object_id)
}

/// One inline-field update of [`LAYER`], under [`SETTLE`].
async fn update(
    store: &LayerStore,
    session_id: &str,
    object_id: u64,
    value: u8,
) -> Result<(u64, u64), ServiceError> {
    tokio::time::timeout(
        SETTLE,
        store.update_layer_relayed(
            session_id,
            LAYER,
            object_id,
            Some(delta_with_set_field_inline(1, "bvn4-field", vec![value], 0)),
            CLIENT_ID,
            1_000,
        ),
    )
    .await
    .expect("the update did not settle within its budget")
}

/// The `seq` an update committed at, its error as text.
fn seq_of(committed: Result<(u64, u64), ServiceError>) -> Result<u64, String> {
    committed.map(|(seq, _)| seq).map_err(|e| format!("{e:?}"))
}

/// A one-shot trap: `-TRYAGAIN` to the first frame `matches` accepts, on any
/// connection; every other frame passes through.
fn tryagain_once(matches: impl Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static) -> ConnProxyRule {
    let armed = AtomicBool::new(true);
    Arc::new(move |_conn: usize, args: &[Vec<u8>]| {
        (matches(args) && armed.swap(false, Ordering::SeqCst))
            .then(|| ProxyAction::Eval(error_reply_script(TRYAGAIN_REPLY)))
    })
}

/// A `TestValkey` whose every connection goes through `proxy`.
async fn through(proxy: &ScriptReplyProxy) -> TestValkey {
    TestValkey::setup_with_url(proxy.url(), POOL_SIZE).await
}

// ═════════════════════════════════════════════════════════════════════════════
// The commit's WATCH and EVALSHA
// ═════════════════════════════════════════════════════════════════════════════

/// The `WATCH` route (`txn` backend): the attempt's `WATCH` of the object's
/// header is answered `-TRYAGAIN` once. The attempt fails transiently, before
/// the commit marker's `MULTI` (nothing queued, nothing ran), and the call
/// retries on a fresh `WATCH` and commits at `seq = 2`.
///
/// Before b.vn4 the reply was a permanent `Unknown`: the call ended
/// `Internal` at attempt 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_watch_answered_tryagain_once_retries_and_commits() {
    let header: Arc<Mutex<Option<String>>> = Arc::default();
    let proxy = {
        let header = Arc::clone(&header);
        ScriptReplyProxy::start_per_connection(
            test_valkey_addr(),
            tryagain_once(move |args| {
                is_command(args, "WATCH")
                    && header
                        .lock()
                        .unwrap()
                        .as_ref()
                        .is_some_and(|key| names(args, key))
            }),
        )
        .await
    };
    let tv = through(&proxy).await;
    let (session_id, object_id) = layer(&tv, "bvn4-tryagain-watch").await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn);
    *header.lock().unwrap() = Some(tv.key_schema().obj_header(&object_id.to_string()));

    let committed = seq_of(update(&store, &session_id, object_id, 1).await);

    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the WATCH was refused -TRYAGAIN once; the call answered {committed:?}"
    );
    assert_eq!(
        committed,
        Ok(2),
        "a -TRYAGAIN to the WATCH is transient: the call retries and commits"
    );
}

/// The `lua` route: the commit's `EVALSHA` of `update_layer_fast.lua` is
/// answered `-TRYAGAIN` once (a multi-key command: the script declares every
/// key it writes). The refusal ran nothing, so the call retries and commits
/// at `seq = 2`.
///
/// Before b.vn4 the call ended `Internal` at attempt 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_commit_evalsha_answered_tryagain_once_retries_and_commits() {
    let sha = UPDATE_LAYER_FAST_SCRIPT.sha1().to_owned();
    let proxy = ScriptReplyProxy::start_per_connection(
        test_valkey_addr(),
        tryagain_once(move |args| {
            is_command(args, "EVALSHA")
                && args
                    .get(1)
                    .is_some_and(|a| String::from_utf8_lossy(a).eq_ignore_ascii_case(&sha))
        }),
    )
    .await;
    let tv = through(&proxy).await;
    let (session_id, object_id) = layer(&tv, "bvn4-tryagain-evalsha").await;
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);

    let committed = seq_of(update(&store, &session_id, object_id, 1).await);

    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the commit EVALSHA was refused -TRYAGAIN once; the call answered \
         {committed:?}"
    );
    assert_eq!(
        committed,
        Ok(2),
        "a -TRYAGAIN to the commit script is transient: the call retries and commits"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The EXEC (decode's Discarded arm)
// ═════════════════════════════════════════════════════════════════════════════

/// The `WATCH` route's commit `EXEC` is answered `-TRYAGAIN` once, as a
/// cluster node's redirect check answers an `EXEC` whose queued keys are
/// split by a migration. The proxy sends `DISCARD` to the store in the
/// `EXEC`'s place, so the real connection leaves `MULTI` and drops its watch
/// set, as Valkey's `discardTransaction` does before that reply; nothing
/// queued ran. The `EXEC` reads as `Discarded`: the call retries on a fresh
/// `WATCH` and commits at `seq = 2`, and the torn-commit arm — its ERROR line
/// and its reference-index recount — never runs, so
/// `live_edit_ext_index_recount_total{site="torn_commit"}` does not move.
///
/// The commit `EXEC` is the first `EXEC` after the attempt's `WATCH` on the
/// same connection (the `WATCH` pipeline's own baseline `EXEC` precedes the
/// `WATCH`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_exec_answered_tryagain_is_discarded_and_retried_without_a_torn_commit_recount(
) {
    let header: Arc<Mutex<Option<String>>> = Arc::default();
    // The connection whose attempt has WATCHed the header and not yet sent
    // its EXEC; `usize::MAX` for none. Taken by the first commit EXEC only.
    let watching = Arc::new(AtomicUsize::new(usize::MAX));
    let fired = Arc::new(AtomicBool::new(false));
    let rule: ConnProxyRule = {
        let (header, watching, fired) = (
            Arc::clone(&header),
            Arc::clone(&watching),
            Arc::clone(&fired),
        );
        Arc::new(move |conn: usize, args: &[Vec<u8>]| {
            if fired.load(Ordering::SeqCst) {
                return None;
            }
            if is_command(args, "WATCH")
                && header
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|key| names(args, key))
            {
                watching.store(conn, Ordering::SeqCst);
                return None;
            }
            (is_command(args, "EXEC")
                && watching.load(Ordering::SeqCst) == conn
                && !fired.swap(true, Ordering::SeqCst))
            .then(|| ProxyAction::DiscardReplying(TRYAGAIN_REPLY.to_owned()))
        })
    };
    let proxy = ScriptReplyProxy::start_rewriting_replies(test_valkey_addr(), rule).await;
    let tv = through(&proxy).await;
    let (session_id, object_id) = layer(&tv, "bvn4-tryagain-exec").await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(Arc::clone(&metrics));
    let recounts_before = torn_commit_recounts(&metrics);
    *header.lock().unwrap() = Some(tv.key_schema().obj_header(&object_id.to_string()));

    let committed = seq_of(update(&store, &session_id, object_id, 1).await);

    assert_eq!(
        proxy.hits(),
        1,
        "precondition: the commit EXEC was refused -TRYAGAIN once; the call answered \
         {committed:?}"
    );
    assert_eq!(
        committed,
        Ok(2),
        "an EXEC refused -TRYAGAIN ran nothing: the call retries and commits"
    );
    assert_eq!(
        torn_commit_recounts(&metrics),
        recounts_before,
        "an EXEC refused -TRYAGAIN is a discarded transaction, not a torn commit: the \
         torn-commit recount must not run"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// A real MIGRATING slot (cluster-gated, disruptive)
// ═════════════════════════════════════════════════════════════════════════════

/// A real slot migration staged `MIGRATING` on the object's owner and
/// `IMPORTING` on another primary, with **no key moved**: every key the
/// object has is still on the source, and the keys a commit names that do not
/// exist yet (the snapshot fence, the commit-token record, …) are missing
/// there. So each multi-key command a commit sends names existing and missing
/// keys of a migrating slot, and the source refuses it with the real
/// `-TRYAGAIN`:
///
/// - the fast-path commit (`lua` backend): its one `EVALSHA`;
/// - a structural `WATCH`-route attempt (`txn` backend): its `WATCH` of the
///   header, the stream and the (deleted, so missing) snapshot fence.
///
/// Each call, given one attempt, ends `Aborted` — transient, never
/// `Internal` — carrying the reply's text verbatim, which this test asserts
/// equals [`TRYAGAIN_REPLY`]: the real-Valkey provenance of the text
/// `is_tryagain_reply` matches. Nothing was applied, and once the migration
/// completes both commit.
///
/// **Disruptive and twice gated**, like the b.gvf slot-migration read in
/// `snapshot_read_cluster_test.rs`: `#[ignore]`d in addition to the
/// `LIVE_EDIT_TEST_CLUSTER_URL` gate, and compiled only with
/// `test-fault-injection`, the feature the shared
/// [`SlotMigration`](crate::storage::cluster_subscriber_test::SlotMigration)
/// helper is built under (without it the test is compiled out, not skipped).
/// The helper's `Drop` guard rolls the staging back if the test panics before
/// completing. CI's `test-cluster-disruptive:` job selects it by the
/// `slot_migration_disruptive` in its name, and `.config/nextest.toml` keeps
/// its output on a pass.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ci: test-cluster-disruptive; disruptive, stages a hash slot MIGRATING between shards \
            (needs test-fault-injection and LIVE_EDIT_TEST_CLUSTER_URL)"]
async fn slot_migration_disruptive_multi_key_commits_meet_the_real_tryagain_and_commit_after_the_migration(
) {
    use fred::interfaces::{ClientLike, ClusterInterface, HashesInterface, KeysInterface};

    use crate::common::owned_delta::delta_with_diff_change_section;
    use crate::storage::cluster_subscriber_test::SlotMigration;

    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!(
            "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured); run \
             `bash scripts/dev-valkey.sh cluster-up` or let CI's `test-cluster-disruptive:` job \
             run this"
        );
        return;
    };
    let tv = TestValkey::setup_with_url(url, POOL_SIZE).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (`redis-cluster://...`)"
    );
    let primaries = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a connected clustered client must have a cached CLUSTER SLOTS view")
        .unique_primary_nodes();
    let (session_id, object_id) = layer(&tv, "bvn4-tryagain-migration").await;
    let oid = object_id.to_string();
    let header_key = tv.key_schema().obj_header(&oid);
    let slot = fred::util::redis_keyslot(header_key.as_bytes());
    let src = tv
        .client()
        .pool
        .next()
        .cached_cluster_state()
        .expect("a cached CLUSTER SLOTS view")
        .get_server(slot)
        .unwrap_or_else(|| panic!("no primary serves slot {slot}"))
        .clone();
    let dst = primaries
        .iter()
        .find(|s| **s != src)
        .expect("a multi-shard cluster has a primary to migrate the slot to")
        .clone();
    // Missing on the source, so the WATCH of the header, the stream and the
    // fence names existing and missing keys. A fresh object holds no fence;
    // the DEL makes it so whatever ran before.
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().obj_snapshot_fence(&oid))
        .await
        .expect("DEL of the snapshot fence must succeed");

    let one_attempt = |backend| {
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend)
            .with_update_layer_retry_max_attempts(1)
    };
    let fast = one_attempt(UpdateLayerBackend::Lua);
    let watch = one_attempt(UpdateLayerBackend::Txn);
    let structural = delta_with_diff_change_section(0, 3, "bvn4-s3", 1, 0);

    let mut migration = SlotMigration::begin(&tv, slot, src.clone(), dst.clone()).await;
    let fast_refused = update(&fast, &session_id, object_id, 7).await;
    let watch_refused = tokio::time::timeout(
        SETTLE,
        watch.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(structural.clone()),
            CLIENT_ID,
            1_000,
        ),
    )
    .await
    .expect("the WATCH-route attempt did not settle within its budget");
    migration.complete().await;

    eprintln!(
        "SLOT-MIGRATION-TRYAGAIN DATAPOINT: slot={slot} src={src} dst={dst}\n  fast-path \
         commit: {fast_refused:?}\n  WATCH-route structural attempt: {watch_refused:?}"
    );
    for (route, refused) in [
        ("the fast-path commit", &fast_refused),
        ("the WATCH-route structural attempt", &watch_refused),
    ] {
        match refused {
            Err(ServiceError::Aborted(msg)) => assert!(
                msg.contains(TRYAGAIN_REPLY),
                "{route} must meet the real -TRYAGAIN, its text verbatim as \
                 `is_tryagain_reply` matches it ({TRYAGAIN_REPLY:?}); got {msg:?}"
            ),
            other => panic!(
                "{route} under a MIGRATING slot must end Aborted (transient: retried by the \
                 loop, UNAVAILABLE on exhaustion), never Internal; got {other:?}"
            ),
        }
    }
    let seq: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET seq must succeed");
    assert_eq!(
        seq.as_deref(),
        Some("1"),
        "nothing a -TRYAGAIN answered was applied"
    );

    let fast = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);
    let watch = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn);
    assert_eq!(
        seq_of(update(&fast, &session_id, object_id, 8).await),
        Ok(2),
        "the fast-path commit lands once the migration is complete"
    );
    let landed = tokio::time::timeout(
        SETTLE,
        watch.update_layer_relayed(
            &session_id,
            LAYER,
            object_id,
            Some(structural),
            CLIENT_ID,
            2_000,
        ),
    )
    .await
    .expect("the WATCH-route commit did not settle within its budget");
    assert_eq!(
        seq_of(landed),
        Ok(3),
        "the WATCH-route structural commit lands once the migration is complete"
    );

    println!(
        "CLUSTER-GATED RAN: storage::b_vn4_tryagain_test::slot_migration_disruptive_multi_key_commits_meet_the_real_tryagain_and_commit_after_the_migration shards={}",
        primaries.len()
    );
}
