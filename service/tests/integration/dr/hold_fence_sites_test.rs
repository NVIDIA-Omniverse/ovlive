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

//! The hold write fence, one check site at a time (SDD SR-6.2).
//!
//! `hold_fence_test.rs` closes each path's fence during its backup-store
//! fetch, which the fetch race alone answers. The tests here close it later —
//! after the fetch, at one check — so each pins the check that is then the
//! next one in the path's way, and drive the real `CheckpointObjectIdCounterRepair`,
//! `SessionRecoveryHook` and `RecoveryHook` rather than the storage
//! primitives, so the fence is also checked to reach the primitives they call.
//!
//! | Site | Test | Where the fence closes |
//! |---|---|---|
//! | counter repair: before the establish-at-zero `SET NX` (inner arm) | [`counter_repair_with_no_sidecar_sends_nothing_once_the_hold_begins`] | as the sidecar read answers "absent" |
//! | counter repair: before the conditional seed | [`counter_repair_with_a_sidecar_sends_nothing_once_the_hold_begins`] | as the sidecar read answers |
//! | counter repair: before its own repeat of the `SET NX` | [`counter_repair_sends_no_second_establish_once_the_hold_begins`] | while the inner `SET NX` is in flight |
//! | counter repair: the fence reaching the seed's retry | [`counter_repair_seed_retry_after_hold_entry_is_never_sent`] | while the seed's first attempt is in flight |
//! | counter repair: the fence reaching the inner `SET NX`'s retry | [`counter_repair_establish_retry_after_hold_entry_is_never_sent`] | while the first `SET NX` is in flight |
//! | counter repair: the fence reaching the repeat `SET NX`'s retry | [`counter_repair_second_establish_retry_after_hold_entry_is_never_sent`] | while the repeat's first attempt is in flight |
//! | session restore: before the apply | [`session_restore_sends_nothing_once_the_hold_begins_after_its_fetch`] | while the post-fetch deletion-marker read is in flight |
//! | session restore: before Step A's `EXEC` | [`session_restore_sends_no_step_a_once_the_hold_begins_at_its_checkout`] | while Step A's checkout probe is in flight |
//! | session restore: before the record step's (Step B's) `EVALSHA` | [`session_restore_sends_no_step_b_once_the_hold_begins_during_step_a`] | while Step A's `EXEC` is in flight |
//! | layer restore: before the high-water probe | [`layer_restore_sends_nothing_once_the_hold_begins_as_its_selection_ends`] | as the selection's blob read answers |
//! | layer restore: before each stale-stream `UNLINK`, and its refusal mapping | [`stale_stream_unlink_retry_after_hold_entry_is_refused_as_the_hold`] | inside the first `UNLINK` attempt, which fails |
//! | layer restore: the residue wipe's `EXEC` | [`residue_wipe_after_hold_entry_wipes_nothing`] | in the wipe's `WATCH` window |
//! | layer restore: before each pre-store chunk | [`prestore_chunk_after_hold_entry_is_never_handed_to_the_store`] | at the high-water probe |
//! | layer restore: the fence reaching the pre-store's store | [`prestore_noscript_reload_through_the_restore_is_never_sent`] | while a pre-store value's `EVALSHA` is in flight |
//!
//! A check whose next neighbour is another check before the same write can
//! only be told apart from it by what is sent in between: removed, the next
//! check still refuses the write, after a transaction checkout's probes or a
//! read. Those tests (the counter repair's first three, the session
//! restore's pre-apply and Step A tests, and the layer restore's pre-probe
//! test) assert that, from the moment the fence closes, the path sends **no
//! command at all** beyond the client's own connection upkeep, through a
//! [`ScriptReplyProxy`] that records every frame it forwards after the fence
//! is closed. The session restore's Step B test asserts it too: on-access
//! recovery's record step is one script, with no checkout before it.
//!
//! # Staging
//!
//! Each subject runs over a client connected through a proxy whose rule
//! counts the frames the test names and holds or substitutes the one it
//! stages ([`Stage`]); the test closes the fence while that frame is in
//! flight, then lets it answer. A backup-store read that closes the fence as
//! it answers ([`ParkedStore::close_on`]) closes it after the fetch race. The
//! restore seams (`test-checkpoint-fault-injection`) close it from inside the
//! `UNLINK` attempt, the residue wipe's window and the high-water probe.
//!
//! Every wait is a deadline a healthy run never reaches: [`SETTLE`] (10 s)
//! for the test's own waits, and the proxy's 20 s budget for a held reply.
//! The heaviest test waits about 50 s at every budget's full value, inside
//! nextest's 120 s hang-killer.

use std::collections::HashMap;
#[cfg(feature = "test-checkpoint-fault-injection")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::KeysInterface;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::{
    CheckpointError, CheckpointObjectIdCounterRepair, CheckpointStore, RecoveryHook,
    RecoveryOutcome, SessionCheckpointWriter, SessionRecoveryHook, SessionRecoveryOutcome,
};
use live_edit::config::Config;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::client::{ValkeyClient, TRYAGAIN_REPLY};
use live_edit::storage::counters::{CounterStore, ObjectIdCounterRepair};
use live_edit::storage::external_values::{ext_content_hash, ValkeyExternalValueStore};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::TestValkey;

use super::hold_support::{ParkedStore, StoreCall};
use crate::checkpoint::helpers::{
    put_base_generation, put_tail_generation_from_records, wipe_session_state,
    HAND_PUT_WRITTEN_AT_UNIX_MS,
};
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::{seed_layer_checkpoint, wipe_object_keys, RECOVERY_SERVICE_LABEL};
use crate::grpc::helpers::{build_service, create_session_and_layer, AllowAllPermissions};
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ReplyGate,
    ScriptReplyProxy,
};

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// The layer every test here uses.
const LAYER: &str = "fence-sites-layer";

/// The sequence a seeded base-only checkpoint carries.
const SNAP_SEQ: u64 = 5;

/// The counters sidecar's high-water mark and the repair's safety margin.
const SIDECAR_MAX_OBJECT_ID: u64 = 500;
const SAFETY_MARGIN: u64 = 10;

/// What the proxy answers a staged script frame with to stage a `NOSCRIPT`.
const NOSCRIPT_REPLY: &str = "NOSCRIPT No matching script. Please use EVAL.";

// ═════════════════════════════════════════════════════════════════════════════
// The proxy rule: staged frames, counted frames, frames after the fence
// ═════════════════════════════════════════════════════════════════════════════

type Matcher = Arc<dyn Fn(&[Vec<u8>]) -> bool + Send + Sync>;

/// A kind of frame the rule counts under `name`, and, when `action` is set,
/// the `at`-th of them (from 1) it stages.
struct Stage {
    name: &'static str,
    matches: Matcher,
    at: usize,
    action: Option<ProxyAction>,
}

impl Stage {
    fn count(name: &'static str, matches: Matcher) -> Self {
        Self {
            name,
            matches,
            at: 0,
            action: None,
        }
    }

    fn act(name: &'static str, matches: Matcher, at: usize, action: ProxyAction) -> Self {
        Self {
            name,
            matches,
            at,
            action: Some(action),
        }
    }
}

/// What the rule saw.
#[derive(Default)]
struct Tally {
    counts: Mutex<HashMap<&'static str, usize>>,
    /// Every frame the client sent after the fence was closed, rendered.
    after_fence: Mutex<Vec<String>>,
}

impl Tally {
    fn count(&self, name: &str) -> usize {
        self.counts.lock().unwrap().get(name).copied().unwrap_or(0)
    }

    #[track_caller]
    fn assert_nothing_after_fence(&self, what: &str) {
        let after = self.after_fence.lock().unwrap();
        assert!(
            after.is_empty(),
            "{what}: once the fence closed, nothing may be sent; sent: {after:?}"
        );
    }
}

fn render_frame(args: &[Vec<u8>]) -> String {
    args.iter()
        .take(3)
        .map(|a| String::from_utf8_lossy(&a[..a.len().min(64)]).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A rule that records every frame sent after `fence` closed, counts each
/// stage's frames and stages the configured one.
fn staged_rule(fence: CancellationToken, tally: Arc<Tally>, stages: Vec<Stage>) -> ProxyRule {
    Arc::new(move |args: &[Vec<u8>]| {
        // The client's own connection upkeep is not the subject's.
        let upkeep = ["PING", "HELLO", "CLIENT", "SELECT", "AUTH", "INFO"]
            .iter()
            .any(|name| name_is(args, name));
        if fence.is_cancelled() && !upkeep {
            tally.after_fence.lock().unwrap().push(render_frame(args));
        }
        let mut action = None;
        for stage in &stages {
            if (stage.matches)(args) {
                let n = {
                    let mut counts = tally.counts.lock().unwrap();
                    let n = counts.entry(stage.name).or_default();
                    *n += 1;
                    *n
                };
                if n == stage.at {
                    action = stage.action.clone();
                }
            }
        }
        action
    })
}

fn name_is(args: &[Vec<u8>], name: &str) -> bool {
    args.first()
        .is_some_and(|a| a.eq_ignore_ascii_case(name.as_bytes()))
}

/// A `name` frame whose first argument is `key`.
fn command_on(name: &'static str, key: String) -> Matcher {
    Arc::new(move |args| {
        name_is(args, name) && args.get(1).map(Vec::as_slice) == Some(key.as_bytes())
    })
}

/// A `name` frame with no key condition.
fn command(name: &'static str) -> Matcher {
    Arc::new(move |args| name_is(args, name))
}

/// A script frame (`EVALSHA` or `EVAL`) one of whose arguments is `key`.
fn script_naming(key: String) -> Matcher {
    Arc::new(move |args| {
        (name_is(args, "EVALSHA") || name_is(args, "EVAL"))
            && args.iter().any(|a| a.as_slice() == key.as_bytes())
    })
}

/// A proxy with `stages` over the test store, and a client through it.
async fn proxied(
    fence: &CancellationToken,
    stages: Vec<Stage>,
) -> (ScriptReplyProxy, ValkeyClient, Arc<Tally>) {
    let tally = Arc::new(Tally::default());
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        staged_rule(fence.clone(), Arc::clone(&tally), stages),
    )
    .await;
    let client = client_through(&proxy).await;
    (proxy, client, tally)
}

/// Poll `cond` until it holds, within [`SETTLE`].
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not happen within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

/// Run `subject` on its own task and return its answer within [`SETTLE`].
async fn settle<T: Send + 'static>(subject: tokio::task::JoinHandle<T>) -> T {
    tokio::time::timeout(SETTLE, subject)
        .await
        .expect("the subject must answer")
        .expect("the subject must not panic")
}

// ═════════════════════════════════════════════════════════════════════════════
// Shared fixtures
// ═════════════════════════════════════════════════════════════════════════════

fn recovery_config() -> Config {
    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = true;
    cfg.service_label = RECOVERY_SERVICE_LABEL.to_owned();
    cfg
}

fn metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(RECOVERY_SERVICE_LABEL))
}

async fn exists(tv: &TestValkey, key: &str) -> bool {
    let n: i64 = tv
        .client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed");
    n > 0
}

#[track_caller]
fn assert_fenced(err: &CheckpointError) {
    assert!(
        err.is_transient() && err.to_string().contains("disaster-recovery hold"),
        "expected the hold write fence's transient refusal; got {err:?}"
    );
}

#[track_caller]
fn assert_restore_fenced(outcome: &RecoveryOutcome) {
    match outcome {
        RecoveryOutcome::Error(err) => assert_fenced(err),
        other => panic!("a fenced layer restore ends Error(Transient); got {other:?}"),
    }
}

fn lineage(metrics: &SharedMetrics, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_checkpoint_restore_lineage_total",
        &[("outcome", outcome)],
    )
    .unwrap_or(0.0) as u64
}

/// An in-memory store with, when `present`, a counters sidecar at
/// [`SIDECAR_MAX_OBJECT_ID`].
async fn sidecar_store(present: bool) -> Arc<dyn CheckpointStore> {
    let store = InMemoryCheckpointStore::new();
    if present {
        let sidecar = CountersSidecar {
            max_object_id_observed: SIDECAR_MAX_OBJECT_ID,
            updated_at: "2026-10-15T03:47:00Z".to_owned(),
        };
        store
            .put_counters_sidecar(Bytes::from(
                serde_json::to_vec(&sidecar).expect("the sidecar serializes"),
            ))
            .await
            .expect("seeding the sidecar must succeed");
    }
    Arc::new(store)
}

/// The real counter repair over a `CounterStore` on `client`, fenced.
fn counter_repair(
    store: Arc<dyn CheckpointStore>,
    client: ValkeyClient,
    tv: &TestValkey,
    fence: &CancellationToken,
) -> Arc<CheckpointObjectIdCounterRepair> {
    Arc::new(
        CheckpointObjectIdCounterRepair::new(
            store,
            Arc::new(CounterStore::new(client, tv.key_schema().clone())),
            SAFETY_MARGIN,
            metrics(),
            SETTLE,
            SETTLE,
        )
        .with_hold_fence(fence.clone()),
    )
}

fn spawn_repair(
    repair: &Arc<CheckpointObjectIdCounterRepair>,
) -> tokio::task::JoinHandle<Result<(), ServiceError>> {
    let repair = Arc::clone(repair);
    tokio::spawn(async move { repair.repair_object_id_counter().await })
}

#[track_caller]
fn assert_repair_refused(result: &Result<(), ServiceError>) {
    assert!(
        matches!(result, Err(ServiceError::Unavailable(_))),
        "a repair stopped by the hold fails its caller UNAVAILABLE; got {result:?}"
    );
}

async fn counter_value(tv: &TestValkey) -> Option<String> {
    tv.client()
        .pool
        .get(tv.key_schema().counter_object_id())
        .await
        .expect("GET must succeed")
}

/// A session with one layer whose session-scoped state was lost, and a
/// session checkpoint for it in an in-memory store.
async fn lost_session_with_a_checkpoint(
    tv: &TestValkey,
    name: &str,
) -> (String, Arc<dyn CheckpointStore>) {
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) = create_session_and_layer(tv, &svc, name, LAYER).await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let tracker = TaskTracker::new();
    let writer = SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        metrics(),
        RECOVERY_SERVICE_LABEL.to_owned(),
        true,
    );
    writer.enqueue_session_checkpoint_write(session_id.clone());
    tracker.close();
    tokio::time::timeout(SETTLE, tracker.wait())
        .await
        .expect("the session checkpoint write must finish");
    assert!(
        store
            .get_session_checkpoint(&session_id)
            .await
            .expect("read back the session checkpoint")
            .is_some(),
        "fixture: the session has a checkpoint to restore from"
    );
    wipe_session_state(tv, &session_id).await;
    (session_id, store)
}

/// The real session restore over `client`, fenced.
fn session_hook(
    store: Arc<dyn CheckpointStore>,
    client: ValkeyClient,
    tv: &TestValkey,
    fence: &CancellationToken,
) -> Arc<SessionRecoveryHook> {
    Arc::new(
        SessionRecoveryHook::new(
            store,
            client,
            tv.key_schema().clone(),
            Arc::new(recovery_config()),
            metrics(),
        )
        .with_hold_fence(fence.clone()),
    )
}

fn spawn_session_restore(
    hook: &Arc<SessionRecoveryHook>,
    session_id: &str,
) -> tokio::task::JoinHandle<SessionRecoveryOutcome> {
    let hook = Arc::clone(hook);
    let session_id = session_id.to_owned();
    tokio::spawn(async move { hook.try_recover_session(&session_id).await })
}

#[track_caller]
fn assert_session_restore_fenced(outcome: &SessionRecoveryOutcome) {
    match outcome {
        SessionRecoveryOutcome::Error(err) => assert_fenced(err),
        other => panic!("a fenced session restore ends Error(Transient); got {other:?}"),
    }
}

/// A session with one layer whose `obj:*` keys were lost, and a base-only
/// checkpoint for it in an in-memory store.
async fn lost_layer_with_a_checkpoint(
    tv: &TestValkey,
    name: &str,
) -> (String, u64, Arc<InMemoryCheckpointStore>) {
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = create_session_and_layer(tv, &svc, name, LAYER).await;
    let store = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(&store, object_id, &session_id, LAYER, SNAP_SEQ).await;
    wipe_object_keys(tv, object_id).await;
    (session_id, object_id, store)
}

/// The base and the head of the hand-put composed lineage, and the section
/// its tail record creates.
const COMPOSED_BASE_SEQ: u64 = 1;
const COMPOSED_HEAD_SEQ: u64 = 2;
const COMPOSED_SECTION_ID: u64 = 42;

/// A session with one layer whose `obj:*` keys were lost, and a composed
/// lineage for it in an in-memory store — an empty base and one tail record
/// that creates a section and sets one external-valued field in it — so its
/// restore pre-stores that value. Returns the value's hash too.
async fn lost_layer_with_a_composed_lineage(
    tv: &TestValkey,
    name: &str,
) -> (u64, Arc<dyn CheckpointStore>, [u8; 32]) {
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = create_session_and_layer(tv, &svc, name, LAYER).await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    put_base_generation(&store, object_id, &session_id, LAYER, COMPOSED_BASE_SEQ).await;
    let blob: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    let hash = ext_content_hash(&blob);
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 1,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: COMPOSED_SECTION_ID,
            section_name: "fence-sites-section".to_owned(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: COMPOSED_SECTION_ID,
            key_name: "fence-sites-ext-field".to_owned(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: blob.len() as u64,
            ext_value_index: 1,
            set_order: 0,
        }],
        ..Default::default()
    };
    let record = Bytes::from(delta_to_ompf_bytes(&delta, &[blob.as_slice()]));
    put_tail_generation_from_records(
        &store,
        object_id,
        &session_id,
        LAYER,
        COMPOSED_BASE_SEQ,
        COMPOSED_BASE_SEQ,
        &[(COMPOSED_HEAD_SEQ, record)],
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;
    wipe_object_keys(tv, object_id).await;
    (object_id, store, hash)
}

fn layer_hook(
    store: Arc<dyn CheckpointStore>,
    client: ValkeyClient,
    tv: &TestValkey,
    metrics: &SharedMetrics,
    fence: &CancellationToken,
) -> RecoveryHook {
    RecoveryHook::new(
        store,
        Arc::new(LayerStore::new(client, tv.key_schema().clone(), 4096)),
        Arc::new(recovery_config()),
        Arc::clone(metrics),
    )
    .with_hold_fence(fence.clone())
}

// ═════════════════════════════════════════════════════════════════════════════
// The object-id counter repair
// ═════════════════════════════════════════════════════════════════════════════

/// No sidecar: the repair's inner arm checks the fence before its
/// establish-at-zero `SET NX`. The fence closes as the sidecar read answers
/// "absent", so that check is the next in the way: nothing more is sent, no
/// counter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_repair_with_no_sidecar_sends_nothing_once_the_hold_begins() {
    let tv = TestValkey::setup().await;
    let fence = CancellationToken::new();
    let store = ParkedStore::passing(sidecar_store(false).await);
    store.close_on(StoreCall::Sidecar, fence.clone());
    let (_proxy, client, tally) = proxied(&fence, Vec::new()).await;
    let repair = counter_repair(store, client, &tv, &fence);

    let result = settle(spawn_repair(&repair)).await;

    assert!(fence.is_cancelled(), "the sidecar read closed the fence");
    assert_repair_refused(&result);
    tally.assert_nothing_after_fence("the establish-at-zero check");
    assert_eq!(counter_value(&tv).await, None);
}

/// A sidecar: the fence closes as the sidecar read answers, so the check
/// before the conditional seed is the next in the way: nothing more is sent,
/// no counter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_repair_with_a_sidecar_sends_nothing_once_the_hold_begins() {
    let tv = TestValkey::setup().await;
    let fence = CancellationToken::new();
    let store = ParkedStore::passing(sidecar_store(true).await);
    store.close_on(StoreCall::Sidecar, fence.clone());
    let (_proxy, client, tally) = proxied(&fence, Vec::new()).await;
    let repair = counter_repair(store, client, &tv, &fence);

    let result = settle(spawn_repair(&repair)).await;

    assert!(fence.is_cancelled(), "the sidecar read closed the fence");
    assert_repair_refused(&result);
    tally.assert_nothing_after_fence("the conditional-seed check");
    assert_eq!(counter_value(&tv).await, None);
}

/// No sidecar: the inner `SET NX` is in flight when the hold begins, and
/// lands. The repair then repeats the `SET NX` for its own postcondition; the
/// check before that repeat stops it: nothing more is sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_repair_sends_no_second_establish_once_the_hold_begins() {
    let tv = TestValkey::setup().await;
    let fence = CancellationToken::new();
    let gate = ReplyGate::new();
    let (_proxy, client, tally) = proxied(
        &fence,
        vec![Stage::act(
            "set",
            command_on("SET", tv.key_schema().counter_object_id()),
            1,
            ProxyAction::HoldReplies(gate.clone()),
        )],
    )
    .await;
    let repair = counter_repair(sidecar_store(false).await, client, &tv, &fence);

    let subject = spawn_repair(&repair);
    gate.reply_held().await;
    fence.cancel();
    gate.open();
    let result = settle(subject).await;

    assert_repair_refused(&result);
    assert_eq!(tally.count("set"), 1, "only the inner SET NX was sent");
    tally.assert_nothing_after_fence("the repeat establish's check");
    assert_eq!(
        counter_value(&tv).await.as_deref(),
        Some("0"),
        "the SET NX sent before entry landed"
    );
}

/// A sidecar: the conditional seed's first attempt is in flight when the hold
/// begins and fails transiently; the repair handed its fence to the seed, so
/// the retry is refused unsent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_repair_seed_retry_after_hold_entry_is_never_sent() {
    let tv = TestValkey::setup().await;
    let fence = CancellationToken::new();
    let gate = ReplyGate::new();
    let (_proxy, client, tally) = proxied(
        &fence,
        vec![Stage::act(
            "seed",
            script_naming(tv.key_schema().counter_object_id()),
            1,
            ProxyAction::GatedEval(gate.clone(), error_reply_script(TRYAGAIN_REPLY)),
        )],
    )
    .await;
    let repair = counter_repair(sidecar_store(true).await, client, &tv, &fence);

    let subject = spawn_repair(&repair);
    wait_until("the seed's first EVALSHA", || tally.count("seed") >= 1).await;
    fence.cancel();
    gate.open();
    let result = settle(subject).await;

    assert_repair_refused(&result);
    assert_eq!(
        tally.count("seed"),
        1,
        "the seed's retry must never be sent"
    );
    assert_eq!(counter_value(&tv).await, None);
}

/// No sidecar: the inner `SET NX`'s first attempt is in flight when the hold
/// begins and fails transiently; its retry is refused unsent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_repair_establish_retry_after_hold_entry_is_never_sent() {
    let tv = TestValkey::setup().await;
    let fence = CancellationToken::new();
    let gate = ReplyGate::new();
    let (_proxy, client, tally) = proxied(
        &fence,
        vec![Stage::act(
            "set",
            command_on("SET", tv.key_schema().counter_object_id()),
            1,
            ProxyAction::GatedEval(gate.clone(), error_reply_script(TRYAGAIN_REPLY)),
        )],
    )
    .await;
    let repair = counter_repair(sidecar_store(false).await, client, &tv, &fence);

    let subject = spawn_repair(&repair);
    wait_until("the first SET NX", || tally.count("set") >= 1).await;
    fence.cancel();
    gate.open();
    let result = settle(subject).await;

    assert_repair_refused(&result);
    assert_eq!(
        tally.count("set"),
        1,
        "the SET NX's retry must never be sent"
    );
    assert_eq!(counter_value(&tv).await, None);
}

/// No sidecar: the inner `SET NX` lands; the repair's repeat `SET NX` is in
/// flight when the hold begins and fails transiently; its retry is refused
/// unsent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_repair_second_establish_retry_after_hold_entry_is_never_sent() {
    let tv = TestValkey::setup().await;
    let fence = CancellationToken::new();
    let gate = ReplyGate::new();
    let (_proxy, client, tally) = proxied(
        &fence,
        vec![Stage::act(
            "set",
            command_on("SET", tv.key_schema().counter_object_id()),
            2,
            ProxyAction::GatedEval(gate.clone(), error_reply_script(TRYAGAIN_REPLY)),
        )],
    )
    .await;
    let repair = counter_repair(sidecar_store(false).await, client, &tv, &fence);

    let subject = spawn_repair(&repair);
    wait_until("the repeat SET NX", || tally.count("set") >= 2).await;
    fence.cancel();
    gate.open();
    let result = settle(subject).await;

    assert_repair_refused(&result);
    assert_eq!(
        tally.count("set"),
        2,
        "the repeat SET NX's retry must never be sent"
    );
    assert_eq!(counter_value(&tv).await.as_deref(), Some("0"));
}

// ═════════════════════════════════════════════════════════════════════════════
// The session restore
// ═════════════════════════════════════════════════════════════════════════════

/// The fence closes while the restore re-reads the session's deletion marker
/// after its fetch, so the check before the apply is the next in the way:
/// nothing more is sent — no transaction checkout, no `EXEC` — and no session
/// state is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_restore_sends_nothing_once_the_hold_begins_after_its_fetch() {
    let tv = TestValkey::setup().await;
    let (session_id, store) = lost_session_with_a_checkpoint(&tv, "fence-sites-pre-apply").await;
    let fence = CancellationToken::new();
    let gate = ReplyGate::new();
    let (_proxy, client, tally) = proxied(
        &fence,
        vec![
            // Read before the fetch (step 4a) and after it (step 6a).
            Stage::act(
                "deleted-marker",
                command_on(
                    "EXISTS",
                    tv.key_schema().session_deleted_marker(&session_id),
                ),
                2,
                ProxyAction::HoldReplies(gate.clone()),
            ),
            Stage::count("exec", command("EXEC")),
        ],
    )
    .await;
    let hook = session_hook(store, client, &tv, &fence);

    let subject = spawn_session_restore(&hook, &session_id);
    gate.reply_held().await;
    fence.cancel();
    gate.open();
    let outcome = settle(subject).await;

    assert_session_restore_fenced(&outcome);
    tally.assert_nothing_after_fence("the pre-apply check");
    assert_eq!(tally.count("exec"), 0);
    assert!(!exists(&tv, &tv.key_schema().session_meta(&session_id)).await);
}

/// The fence closes while Step A's transaction checkout is probing its slot,
/// after the pre-apply check passed, so the check before Step A's `EXEC` is
/// the next in the way: no `EXEC`, no session state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_restore_sends_no_step_a_once_the_hold_begins_at_its_checkout() {
    let tv = TestValkey::setup().await;
    let (session_id, store) = lost_session_with_a_checkpoint(&tv, "fence-sites-step-a").await;
    let fence = CancellationToken::new();
    let gate = ReplyGate::new();
    let (_proxy, client, tally) = proxied(
        &fence,
        vec![
            // The checkout probe's last command, on Step A's slot.
            Stage::act(
                "unwatch",
                command("UNWATCH"),
                1,
                ProxyAction::HoldReplies(gate.clone()),
            ),
            Stage::count("exec", command("EXEC")),
        ],
    )
    .await;
    let hook = session_hook(store, client, &tv, &fence);

    let subject = spawn_session_restore(&hook, &session_id);
    gate.reply_held().await;
    fence.cancel();
    gate.open();
    let outcome = settle(subject).await;

    assert_session_restore_fenced(&outcome);
    assert_eq!(tally.count("exec"), 0, "Step A's EXEC must not be sent");
    tally.assert_nothing_after_fence("Step A's check");
    assert!(!exists(&tv, &tv.key_schema().session_meta(&session_id)).await);
}

/// Step A's `EXEC` is in flight when the hold begins, and lands; the check
/// before the record step's `EVALSHA` (on-access recovery's Step B, a script
/// on the data pool) stops it: nothing more is sent — no `EVALSHA`, no
/// `SCRIPT LOAD` — and the session's own record is not written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_restore_sends_no_step_b_once_the_hold_begins_during_step_a() {
    let tv = TestValkey::setup().await;
    let (session_id, store) = lost_session_with_a_checkpoint(&tv, "fence-sites-step-b").await;
    let fence = CancellationToken::new();
    let gate = ReplyGate::new();
    let (_proxy, client, tally) = proxied(
        &fence,
        vec![
            Stage::act(
                "exec",
                command("EXEC"),
                1,
                ProxyAction::HoldReplies(gate.clone()),
            ),
            Stage::count(
                "record-step",
                script_naming(tv.key_schema().session_meta(&session_id)),
            ),
        ],
    )
    .await;
    let hook = session_hook(store, client, &tv, &fence);

    let subject = spawn_session_restore(&hook, &session_id);
    gate.reply_held().await;
    fence.cancel();
    gate.open();
    let outcome = settle(subject).await;

    assert_session_restore_fenced(&outcome);
    assert_eq!(
        tally.count("exec"),
        1,
        "only Step A's EXEC, sent before entry"
    );
    assert_eq!(
        tally.count("record-step"),
        0,
        "the record step's EVALSHA must not be sent"
    );
    tally.assert_nothing_after_fence("the record step's check");
    assert!(
        !exists(&tv, &tv.key_schema().session_meta(&session_id)).await,
        "the record step writes the session record; it must not have run"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The layer restore
// ═════════════════════════════════════════════════════════════════════════════

/// The fence closes as the selection's blob read answers — the selection
/// finishes, past its race against the fence — so the check before the
/// stream high-water probe is the next in the way: nothing more is sent, not
/// even the probe's read, and no object is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_restore_sends_nothing_once_the_hold_begins_as_its_selection_ends() {
    let tv = TestValkey::setup().await;
    let (_session_id, object_id, inner) =
        lost_layer_with_a_checkpoint(&tv, "fence-sites-pre-probe").await;
    let fence = CancellationToken::new();
    let store = ParkedStore::passing(inner);
    store.close_on(StoreCall::LayerGet, fence.clone());
    let (_proxy, client, tally) = proxied(&fence, Vec::new()).await;
    let metrics = metrics();
    let hook = layer_hook(store, client, &tv, &metrics, &fence);

    let outcome = tokio::time::timeout(SETTLE, hook.try_restore_layer(object_id))
        .await
        .expect("the restore must answer");

    assert!(fence.is_cancelled(), "the blob read closed the fence");
    assert_eq!(
        lineage(&metrics, "gathered_base"),
        1,
        "the selection finished: the fence closed after its race"
    );
    assert_restore_fenced(&outcome);
    tally.assert_nothing_after_fence("the pre-probe check");
    assert!(!exists(&tv, &tv.key_schema().obj_header(&object_id.to_string())).await);
}

/// A restore seam injector: on its first fire at `op` it closes `fence` and
/// answers `first`; every other fire passes.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct CloseFenceAt {
    op: &'static str,
    fence: CancellationToken,
    first: fn() -> Result<(), CheckpointError>,
    fired: AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl CloseFenceAt {
    fn new(
        op: &'static str,
        fence: &CancellationToken,
        first: fn() -> Result<(), CheckpointError>,
    ) -> Arc<Self> {
        Arc::new(Self {
            op,
            fence: fence.clone(),
            first,
            fired: AtomicUsize::new(0),
        })
    }

    fn fired(&self) -> usize {
        self.fired.load(Ordering::SeqCst)
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for CloseFenceAt {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op != self.op {
            return Ok(());
        }
        if self.fired.fetch_add(1, Ordering::SeqCst) == 0 {
            self.fence.cancel();
            return (self.first)();
        }
        Ok(())
    }
}

/// A stale stream (an auto-ID `deleted` tombstone top) the restore must
/// `UNLINK`: the first attempt fails transiently as the hold begins, and the
/// retry is refused before it is sent — the seam that fires at the top of
/// every attempt fires once — the stream survives, and the restore ends with
/// the hold's refusal, not the `unlink_unverified` one.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_stream_unlink_retry_after_hold_entry_is_refused_as_the_hold() {
    use fred::interfaces::StreamsInterface;
    use live_edit::checkpoint::RECOVERY_SEAM_STALE_STREAM_UNLINK;

    let tv = TestValkey::setup().await;
    let (_session_id, object_id, store) =
        lost_layer_with_a_checkpoint(&tv, "fence-sites-unlink").await;
    let stream_key = tv.key_schema().layer_stream(object_id);
    let _: String = tv
        .client()
        .pool
        .xadd(
            &stream_key,
            false,
            None,
            "*",
            vec![("payload", fred::types::Value::from("deleted"))],
        )
        .await
        .expect("XADD of the tombstone must succeed");
    let fence = CancellationToken::new();
    let injector = CloseFenceAt::new(RECOVERY_SEAM_STALE_STREAM_UNLINK, &fence, || {
        Err(CheckpointError::Transient {
            source: "the first UNLINK attempt fails as the hold begins".into(),
        })
    });
    let metrics = metrics();
    let hook = layer_hook(store, tv.client().clone(), &tv, &metrics, &fence)
        .with_restore_fault_injector_for_test(Arc::clone(&injector) as _);

    let outcome = tokio::time::timeout(SETTLE, hook.try_restore_layer(object_id))
        .await
        .expect("the restore must answer");

    assert_eq!(
        injector.fired(),
        1,
        "the retry must be refused before its attempt"
    );
    assert_restore_fenced(&outcome);
    assert_eq!(
        sample_value(
            &render(&metrics),
            "live_edit_checkpoint_recovery_stream_unresolved_total",
            &[("reason", "unlink_unverified")],
        )
        .unwrap_or(0.0),
        0.0,
        "a refusal by the fence is the hold's, not an unverified stream removal"
    );
    assert!(exists(&tv, &stream_key).await, "no UNLINK after entry");
    assert!(!exists(&tv, &tv.key_schema().obj_header(&object_id.to_string())).await);
}

/// A residue (an `obj:*` index surviving the header) the restore wipes first:
/// the hold begins inside the wipe's `WATCH` window, after its reads, and the
/// check before its `EXEC` stops it: the residue survives.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn residue_wipe_after_hold_entry_wipes_nothing() {
    use fred::interfaces::SetsInterface;
    use live_edit::checkpoint::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC;

    let tv = TestValkey::setup().await;
    let (_session_id, object_id, store) =
        lost_layer_with_a_checkpoint(&tv, "fence-sites-residue").await;
    let residue = tv.key_schema().obj_sections_index(&object_id.to_string());
    let _: i64 = tv
        .client()
        .pool
        .sadd(&residue, "7")
        .await
        .expect("planting the residue must succeed");
    let fence = CancellationToken::new();
    let injector = CloseFenceAt::new(RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC, &fence, || Ok(()));
    let metrics = metrics();
    let hook = layer_hook(store, tv.client().clone(), &tv, &metrics, &fence)
        .with_ext_value_store(Arc::new(ValkeyExternalValueStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        )))
        .with_restore_fault_injector_for_test(Arc::clone(&injector) as _);

    let outcome = tokio::time::timeout(SETTLE, hook.try_restore_layer(object_id))
        .await
        .expect("the restore must answer");

    assert_eq!(injector.fired(), 1, "the wipe reached its EXEC window");
    assert_restore_fenced(&outcome);
    assert!(
        exists(&tv, &residue).await,
        "the wipe's EXEC must not be sent once the fence is closed"
    );
}

/// A composed restore owing one pre-stored value, through a store that
/// ignores the fence (the trait's default `store_values_unless_held`): the
/// hold begins at the high-water probe, and the check before each pre-store
/// chunk stops the pre-store before the store is ever called.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prestore_chunk_after_hold_entry_is_never_handed_to_the_store() {
    use live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE;

    use super::hold_support::ShimExtStore;

    let tv = TestValkey::setup().await;
    let (object_id, store, hash) =
        lost_layer_with_a_composed_lineage(&tv, "fence-sites-chunk").await;
    let fence = CancellationToken::new();
    let injector = CloseFenceAt::new(RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE, &fence, || Ok(()));
    let ext = Arc::new(ShimExtStore::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    )));
    let metrics = metrics();
    let hook = layer_hook(store, tv.client().clone(), &tv, &metrics, &fence)
        .with_ext_value_store(Arc::clone(&ext) as _)
        .with_restore_fault_injector_for_test(Arc::clone(&injector) as _);

    let outcome = tokio::time::timeout(SETTLE, hook.try_restore_layer(object_id))
        .await
        .expect("the restore must answer");

    assert_eq!(
        lineage(&metrics, "gathered_composed"),
        1,
        "precondition: the composed lineage, which carries a value to pre-store, was selected"
    );
    assert_restore_fenced(&outcome);
    assert_eq!(
        ext.store_values_calls(),
        0,
        "no pre-store chunk may be handed to the store after entry"
    );
    assert!(!exists(&tv, &tv.key_schema().ext_data(&sha256_to_hex(&hash))).await);
}

/// A composed restore owing one pre-stored value, through the production
/// external-value store on a proxied client: the value's `EVALSHA` is in
/// flight when the hold begins and is answered `NOSCRIPT`. The restore handed
/// its fence through to the store, so neither the reload nor the re-send is
/// sent and the value is not stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prestore_noscript_reload_through_the_restore_is_never_sent() {
    let tv = TestValkey::setup().await;
    let (object_id, store, hash) =
        lost_layer_with_a_composed_lineage(&tv, "fence-sites-prestore").await;
    let data_key = tv.key_schema().ext_data(&sha256_to_hex(&hash));
    let fence = CancellationToken::new();
    let gate = ReplyGate::new();
    let (_proxy, ext_client, tally) = proxied(
        &fence,
        vec![Stage::act(
            "store",
            script_naming(data_key.clone()),
            1,
            ProxyAction::GatedEval(gate.clone(), error_reply_script(NOSCRIPT_REPLY)),
        )],
    )
    .await;
    let metrics = metrics();
    let hook = Arc::new(
        layer_hook(store, tv.client().clone(), &tv, &metrics, &fence).with_ext_value_store(
            Arc::new(ValkeyExternalValueStore::new(
                ext_client,
                tv.key_schema().clone(),
            )),
        ),
    );

    let subject = tokio::spawn({
        let hook = Arc::clone(&hook);
        async move { hook.try_restore_layer(object_id).await }
    });
    wait_until("the pre-store's EVALSHA", || tally.count("store") >= 1).await;
    fence.cancel();
    gate.open();
    let outcome = settle(subject).await;

    assert_restore_fenced(&outcome);
    assert_eq!(tally.count("store"), 1, "no re-sent EVALSHA after entry");
    tally.assert_nothing_after_fence("the pre-store's NOSCRIPT reload");
    assert!(!exists(&tv, &data_key).await, "the value was not stored");
    assert!(!exists(&tv, &tv.key_schema().obj_header(&object_id.to_string())).await);
}
