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

//! The hold write fence of the on-access repair and restore paths
//! (`live_edit::checkpoint::hold_fence`; SDD SR-6.2): after a hold's entry
//! none of them sends a store write it has not sent yet, re-sends included.
//!
//! Each path reads its fence through one accessor (`hold_fence()` on
//! `CheckpointObjectIdCounterRepair`, `RecoveryHook` and
//! `SessionRecoveryHook`, and for the external-value fetch-on-miss rewrite
//! the store's own fence, `with_recovery_hold_fence`), so each path's test
//! closes the fence while the path is under way and shows that it writes
//! nothing; with that path's fence absent or open, the same test sees the
//! write land. The checks
//! a path makes after its fetch, one at a time, are
//! `hold_fence_sites_test.rs`'s.
//!
//! | Path, or mechanism | Test | Where the fence closes |
//! |---|---|---|
//! | object-id counter repair | [`counter_repair_fenced_during_its_sidecar_fetch_writes_no_counter`] | during the sidecar fetch |
//! | its spawn check | [`counter_repair_after_entry_starts_no_attempt`] | before the call: no attempt, no store command |
//! | session restore | [`session_restore_fenced_during_its_checkpoint_fetch_writes_no_session`] | during the checkpoint fetch |
//! | layer restore, on access | [`layer_restore_fenced_during_its_source_selection_writes_no_object`] | during the selection |
//! | layer restore, in a catch-up | [`catch_up_restore_in_flight_at_hold_entry_writes_no_object`] | the hold ending and the fence, mid-selection |
//! | the materialisation `EXEC` | [`layer_restore_fenced_just_before_its_materialisation_sends_no_exec`] | in the `WATCH` window, after every read |
//! | a transient retry (`EVALSHA`) | [`counter_seed_retry_after_hold_entry_is_never_sent`] | while the first attempt is in flight |
//! | a transient retry (`SET NX`) | [`counter_establish_retry_after_hold_entry_is_never_sent`] | while the first attempt is in flight |
//! | a `NOSCRIPT` reload, conditional seed | [`counter_seed_noscript_reload_after_hold_entry_is_never_sent`] | while the `EVALSHA` is in flight |
//! | a `NOSCRIPT` reload, pre-store value | [`prestore_noscript_reload_after_hold_entry_is_never_sent`] | while the `EVALSHA` is in flight |
//! | a `NOSCRIPT` reload, release enqueue | [`release_enqueue_noscript_reload_after_hold_entry_is_never_sent`] | while the `EVALSHA` is in flight |
//! | the release enqueue skip | [`committed_restore_whose_release_enqueue_is_unsent_at_entry_skips_and_counts_it`] | between the committed `EXEC` and the enqueue |
//! | external-value fetch-on-miss rewrite | [`fetch_on_miss_rewrite_after_hold_entry_writes_nothing_and_answers_unavailable`] | as the backup fetch answers |
//!
//! # Staging
//!
//! The fetches are parked on a [`ParkedStore`] gate, so the test closes the
//! fence while the path is provably inside its fetch. The re-sends are staged
//! with [`ScriptReplyProxy`] on frames only the subject sends — each names the
//! test's prefixed key — and the proxy holds the first one's reply behind a
//! gate ([`ProxyAction::GatedEval`]) until the test has closed the fence, then
//! answers it `-TRYAGAIN` (transient, retried) or `-NOSCRIPT` (reloaded). The
//! rule counts every later frame of the same write, and every `SCRIPT` frame
//! after the first, so "never sent" is a count of zero rather than an absence
//! of evidence. The release-enqueue skip holds the replies behind the
//! materialisation's `EXEC` ([`ProxyAction::HoldReplies`]), which has reached
//! the store when the fence closes.
//!
//! Every wait is a deadline a healthy run never reaches: [`SETTLE`] (10 s)
//! for the test's own waits, and the snapshot-read instruments' 20 s budget
//! for a `MONITOR` acknowledgement or drain and a held reply. The heaviest
//! tests — the spawn-check test (a `MONITOR` start and drain) and the
//! release-enqueue skip (a held reply, then the restore) — wait at most
//! about 40 s at every budget's full value, inside nextest's 120 s
//! hang-killer.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tonic::Request;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::{
    CheckpointError, CheckpointObjectIdCounterRepair, CheckpointStore, RecoveryHook,
    RecoveryOutcome, SessionCheckpointWriter, SessionRecoveryHook, SessionRecoveryOutcome,
};
use live_edit::config::Config;
use live_edit::error::ServiceError;
use live_edit::grpc::graceful_shutdown::HoldCause;
use live_edit::grpc::proto::SubscribeReadLayerRequest;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::client::{ValkeyClient, TRYAGAIN_REPLY};
use live_edit::storage::counters::{CounterStore, ObjectIdCounterRepair};
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::TestValkey;

use super::hold_support::{ParkedStore, StoreCall};
use crate::checkpoint::helpers::wipe_session_state;
use crate::common::{seed_layer_checkpoint, wipe_object_keys, RECOVERY_SERVICE_LABEL};
use crate::grpc::helpers::{build_service, create_session_and_layer, AllowAllPermissions};
use crate::grpc::log_capture::{purge_keys, ThreadCapture};
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, monitor_fence, test_valkey_addr, MonitorTap, ProxyAction,
    ProxyRule, ReplyGate, ScriptReplyProxy,
};

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// The layer every test here uses.
const LAYER: &str = "fence-layer";

/// The sequence the seeded layer checkpoint carries.
const SNAP_SEQ: u64 = 5;

/// The counters sidecar's high-water mark, and the repair's safety margin:
/// an unfenced repair seeds the counter to their sum.
const SIDECAR_MAX_OBJECT_ID: u64 = 500;
const SAFETY_MARGIN: u64 = 10;

/// What the proxy answers a gated script frame with to stage a `NOSCRIPT`.
const NOSCRIPT_REPLY: &str = "NOSCRIPT No matching script. Please use EVAL.";

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures
// ═════════════════════════════════════════════════════════════════════════════

/// An in-memory store holding a counters sidecar at
/// [`SIDECAR_MAX_OBJECT_ID`].
async fn store_with_a_sidecar() -> Arc<dyn CheckpointStore> {
    let store = InMemoryCheckpointStore::new();
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
    Arc::new(store)
}

/// A `Config` for the recovery hooks: checkpointing on, the shared service
/// label.
fn recovery_config() -> Config {
    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = true;
    cfg.service_label = RECOVERY_SERVICE_LABEL.to_owned();
    cfg
}

fn metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(RECOVERY_SERVICE_LABEL))
}

fn cell(metrics: &SharedMetrics, name: &str, labels: &[(&str, &str)]) -> u64 {
    sample_value(&render(metrics), name, labels).unwrap_or(0.0) as u64
}

fn counter_repairs(metrics: &SharedMetrics, outcome: &str) -> u64 {
    cell(
        metrics,
        "live_edit_object_id_counter_repair_total",
        &[("outcome", outcome)],
    )
}

fn recovery_attempts(metrics: &SharedMetrics, result: &str, scope: &str) -> u64 {
    cell(
        metrics,
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", scope),
            ("service", RECOVERY_SERVICE_LABEL),
        ],
    )
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

/// Assert `err` is a storage-side fence refusal: `UNAVAILABLE`, nothing sent.
#[track_caller]
fn assert_unsent_refusal(err: &ServiceError) {
    assert!(
        matches!(err, ServiceError::Unavailable(m) if m.contains("disaster-recovery hold")),
        "expected the hold write fence's refusal; got {err:?}"
    );
}

/// Assert `err` is a checkpoint-side fence refusal.
#[track_caller]
fn assert_fenced(err: &CheckpointError) {
    assert!(
        err.is_transient() && err.to_string().contains("disaster-recovery hold"),
        "expected the hold write fence's transient refusal; got {err:?}"
    );
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

/// What a [`gate_the_first_send`] rule saw.
#[derive(Default)]
struct Sends {
    /// Frames of the watched write, the gated first one included.
    writes: AtomicUsize,
    /// `SCRIPT` frames (a `SCRIPT LOAD`) after the first watched frame.
    reloads: AtomicUsize,
}

impl Sends {
    fn writes(&self) -> usize {
        self.writes.load(Ordering::SeqCst)
    }
    fn reloads(&self) -> usize {
        self.reloads.load(Ordering::SeqCst)
    }
}

/// A proxy rule that holds the first frame `is_write` accepts behind `gate`
/// and then answers it with `reply` (`-<reply>`), forwards every later one,
/// and counts both kinds of frame in `sends`.
fn gate_the_first_send(
    is_write: impl Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static,
    gate: ReplyGate,
    reply: &str,
    sends: Arc<Sends>,
) -> ProxyRule {
    let script = error_reply_script(reply);
    Arc::new(move |args: &[Vec<u8>]| {
        let name = args.first()?.to_ascii_uppercase();
        if name == b"SCRIPT" {
            if sends.writes() > 0 {
                sends.reloads.fetch_add(1, Ordering::SeqCst);
            }
            return None;
        }
        if !is_write(args) {
            return None;
        }
        if sends.writes.fetch_add(1, Ordering::SeqCst) == 0 {
            Some(ProxyAction::GatedEval(gate.clone(), script.clone()))
        } else {
            None
        }
    })
}

/// A script frame (`EVALSHA` or `EVAL`) one of whose arguments is `key`.
fn script_naming(key: String) -> impl Fn(&[Vec<u8>]) -> bool + Send + Sync + 'static {
    move |args: &[Vec<u8>]| {
        let name = args.first().map(|a| a.to_ascii_uppercase());
        matches!(name.as_deref(), Some(b"EVALSHA") | Some(b"EVAL"))
            && args.iter().any(|a| a.as_slice() == key.as_bytes())
    }
}

/// Stage a fenced write against the proxy: run `write` with `fence`, wait
/// until its first frame is held at the proxy, close the fence (the hold's
/// entry, while the write is in flight), then release the staged reply, and
/// return what `write` answered.
async fn close_the_fence_while_the_first_send_is_in_flight<T, F>(
    write: F,
    fence: &CancellationToken,
    gate: &ReplyGate,
    sends: &Sends,
) -> T
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let task = tokio::spawn(write);
    wait_until("the write's first frame to reach the proxy", || {
        sends.writes() >= 1
    })
    .await;
    fence.cancel();
    gate.open();
    tokio::time::timeout(SETTLE, task)
        .await
        .expect("the fenced write must finish")
        .expect("the fenced write must not panic")
}

/// A session with one layer whose `obj:*` keys were lost and whose
/// checkpoint is seeded in an in-memory store: the shape a layer restore
/// runs against.
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

// ═════════════════════════════════════════════════════════════════════════════
// The object-id counter repair
// ═════════════════════════════════════════════════════════════════════════════

/// A counter repair whose fence closes while it reads the counters sidecar
/// writes no counter — the sidecar arrives, and is not acted on — and fails
/// the caller `UNAVAILABLE`, booked `failed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_repair_fenced_during_its_sidecar_fetch_writes_no_counter() {
    let tv = TestValkey::setup().await;
    let store = ParkedStore::over(store_with_a_sidecar().await);
    let metrics = metrics();
    let fence = CancellationToken::new();
    let repair = Arc::new(
        CheckpointObjectIdCounterRepair::new(
            store.clone(),
            Arc::new(CounterStore::new(
                tv.client().clone(),
                tv.key_schema().clone(),
            )),
            SAFETY_MARGIN,
            Arc::clone(&metrics),
            SETTLE,
            SETTLE,
        )
        .with_hold_fence(fence.clone()),
    );
    let counter = tv.key_schema().counter_object_id();
    assert!(
        !exists(&tv, &counter).await,
        "fixture: the counter is absent"
    );

    let caller = tokio::spawn({
        let repair = Arc::clone(&repair);
        async move { repair.repair_object_id_counter().await }
    });
    store.sidecar.entered("the sidecar fetch").await;
    fence.cancel();
    store.sidecar.release();
    let result = tokio::time::timeout(SETTLE, caller)
        .await
        .expect("the repair must answer")
        .expect("the repair must not panic");

    assert!(
        matches!(result, Err(ServiceError::Unavailable(_))),
        "a fenced repair fails its caller UNAVAILABLE; got {result:?}"
    );
    assert!(
        !exists(&tv, &counter).await,
        "a hold that began during the sidecar fetch must stop the seed: an unfenced repair seeds \
         the counter to {}",
        SIDECAR_MAX_OBJECT_ID + SAFETY_MARGIN
    );
    assert_eq!(counter_repairs(&metrics, "failed"), 1);
    assert_eq!(counter_repairs(&metrics, "repaired_seeded"), 0);
}

/// Once the fence is closed a `CreateLayer` that finds the counter absent
/// starts no repair attempt: the caller fails at once, booked `failed`, and
/// nothing reaches the store or the backup store — not even the attempt's
/// re-read of the counter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_repair_after_entry_starts_no_attempt() {
    let tv = TestValkey::setup().await;
    let store = ParkedStore::over(store_with_a_sidecar().await);
    store.sidecar.release();
    let metrics = metrics();
    let fence = CancellationToken::new();
    let repair = CheckpointObjectIdCounterRepair::new(
        store.clone(),
        Arc::new(CounterStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        )),
        SAFETY_MARGIN,
        Arc::clone(&metrics),
        SETTLE,
        SETTLE,
    )
    .with_hold_fence(fence.clone());
    let counter = tv.key_schema().counter_object_id();
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    fence.cancel();
    let result = repair.repair_object_id_counter().await;

    assert!(
        matches!(result, Err(ServiceError::Unavailable(_))),
        "a repair after entry fails its caller UNAVAILABLE; got {result:?}"
    );
    let marker = monitor_fence(tv.prefix());
    let _: Option<String> = tv
        .client()
        .pool
        .get(&marker)
        .await
        .expect("the monitor fence GET must succeed");
    let seen = tap.drain_until(&marker).await;
    let touched: Vec<_> = seen.iter().filter(|c| c.args.contains(&counter)).collect();
    assert!(
        touched.is_empty(),
        "no attempt may start after entry, so nothing reads or writes the counter: {touched:?}"
    );
    assert!(!exists(&tv, &counter).await);
    assert_eq!(
        counter_repairs(&metrics, "failed"),
        1,
        "the refusal is booked once"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The session restore
// ═════════════════════════════════════════════════════════════════════════════

/// A session restore whose fence closes while it fetches the session's
/// checkpoint writes no session state, and ends `Error(Transient)` with the
/// fence's refusal, booked as an `error` attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_restore_fenced_during_its_checkpoint_fetch_writes_no_session() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "fence-session-restore", LAYER).await;
    let inner: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let tracker = TaskTracker::new();
    let writer = SessionCheckpointWriter::new(
        Arc::clone(&inner),
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
        inner
            .get_session_checkpoint(&session_id)
            .await
            .expect("read back the session checkpoint")
            .is_some(),
        "fixture: the session has a checkpoint to restore from"
    );
    wipe_session_state(&tv, &session_id).await;
    let meta = tv.key_schema().session_meta(&session_id);
    assert!(!exists(&tv, &meta).await, "fixture: the session is gone");

    let store = ParkedStore::over(inner);
    let metrics = metrics();
    let fence = CancellationToken::new();
    let hook = Arc::new(
        SessionRecoveryHook::new(
            store.clone(),
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(recovery_config()),
            Arc::clone(&metrics),
        )
        .with_hold_fence(fence.clone()),
    );
    let restore = tokio::spawn({
        let hook = Arc::clone(&hook);
        let session_id = session_id.clone();
        async move { hook.try_recover_session(&session_id).await }
    });
    store.session.entered("the session checkpoint fetch").await;
    fence.cancel();
    store.session.release();
    let outcome = tokio::time::timeout(SETTLE, restore)
        .await
        .expect("the restore must answer")
        .expect("the restore must not panic");

    match &outcome {
        SessionRecoveryOutcome::Error(err) => assert_fenced(err),
        other => panic!("a fenced session restore ends Error(Transient); got {other:?}"),
    }
    assert!(
        !exists(&tv, &meta).await,
        "a hold that began during the fetch must stop the restore's writes"
    );
    assert_eq!(recovery_attempts(&metrics, "error", "session"), 1);
    assert_eq!(recovery_attempts(&metrics, "restored", "session"), 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// The layer restore: on access and in a catch-up
// ═════════════════════════════════════════════════════════════════════════════

/// A layer restore whose fence closes while it selects its restore source
/// writes no object, and ends `Error(Transient)` with the fence's refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_restore_fenced_during_its_source_selection_writes_no_object() {
    let tv = TestValkey::setup().await;
    let (_session_id, object_id, inner) =
        lost_layer_with_a_checkpoint(&tv, "fence-layer-restore").await;
    let store = ParkedStore::over(inner);
    let metrics = metrics();
    let fence = CancellationToken::new();
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        4096,
    ));
    let hook = Arc::new(
        RecoveryHook::new(
            store.clone(),
            layer_store,
            Arc::new(recovery_config()),
            Arc::clone(&metrics),
        )
        .with_hold_fence(fence.clone()),
    );

    let restore = tokio::spawn({
        let hook = Arc::clone(&hook);
        async move { hook.try_restore_layer(object_id).await }
    });
    store.listing.entered("the restore-source selection").await;
    fence.cancel();
    store.listing.release();
    let outcome = tokio::time::timeout(SETTLE, restore)
        .await
        .expect("the restore must answer")
        .expect("the restore must not panic");

    match &outcome {
        RecoveryOutcome::Error(err) => assert_fenced(err),
        other => panic!("a fenced layer restore ends Error(Transient); got {other:?}"),
    }
    assert!(
        !exists(&tv, &tv.key_schema().obj_header(&object_id.to_string())).await,
        "a hold that began during the selection must stop the materialisation"
    );
    assert_eq!(recovery_attempts(&metrics, "error", "layer"), 1);
    assert_eq!(recovery_attempts(&metrics, "restored", "layer"), 0);
}

/// A `SubscribeReadLayer` catch-up already inside a layer restore when the
/// hold begins: the entry routine ends the stream and closes the fence, the
/// restore — which the stream's ending does not interrupt — finishes its
/// selection and then writes nothing, and the subscription ends
/// `UNAVAILABLE`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catch_up_restore_in_flight_at_hold_entry_writes_no_object() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "fence-catch-up-restore", LAYER).await;
    let inner = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(&inner, object_id, &session_id, LAYER, SNAP_SEQ).await;
    wipe_object_keys(&tv, object_id).await;
    let store = ParkedStore::over(inner);
    let fence = CancellationToken::new();
    svc.recovery = Arc::new(
        RecoveryHook::new(
            store.clone(),
            Arc::clone(&svc.layer_store),
            Arc::new(recovery_config()),
            Arc::clone(&svc.metrics),
        )
        .with_hold_fence(fence.clone()),
    );

    // The same-object path: the catch-up reads the header, finds it gone,
    // and restores.
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.clone(),
        layer_name: LAYER.to_owned(),
        object_id,
        sequence_num: 1,
        client_id: "fence-reader".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("fence-bearer"));
    let mut stream = handle_subscribe_read_layer(&svc, request)
        .await
        .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
        .into_inner();
    store.listing.entered("the catch-up's restore").await;

    // Hold entry, as the routine runs it for these two: the streams end, the
    // fence closes.
    svc.shutdown.end_streams_for_hold(HoldCause::Held {
        incident: Some("20261015t0347-a1b2".to_owned()),
    });
    fence.cancel();
    store.listing.release();

    let terminal = tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return Some(status);
            }
        }
        None
    })
    .await
    .expect("the subscription must end")
    .expect("the subscription must end with a status, not OK");
    assert_eq!(terminal.code(), tonic::Code::Unavailable, "{terminal:?}");
    wait_until("the restore to retire its admission entry", || {
        svc.recovery.admission_len() == 0
    })
    .await;
    assert!(
        !exists(&tv, &tv.key_schema().obj_header(&object_id.to_string())).await,
        "the restore in flight at entry must write nothing after it"
    );
}

/// A hold that begins inside the materialisation's `WATCH` window — after
/// every read and all the queueing — stops it at the check immediately
/// before the `EXEC`: no `EXEC`, no object.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_restore_fenced_just_before_its_materialisation_sends_no_exec() {
    let tv = TestValkey::setup().await;
    let (_session_id, object_id, inner) =
        lost_layer_with_a_checkpoint(&tv, "fence-materialisation").await;
    let metrics = metrics();
    let fence = CancellationToken::new();
    let close_fence: live_edit::storage::watched_txn::WatchWindowHook = {
        let fence = fence.clone();
        Arc::new(move |_: &fred::clients::Client| {
            fence.cancel();
            futures::FutureExt::boxed(async {})
        })
    };
    let layer_store = Arc::new(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_restore_write_before_exec_hook_for_test(close_fence),
    );
    let hook = RecoveryHook::new(
        inner,
        layer_store,
        Arc::new(recovery_config()),
        Arc::clone(&metrics),
    )
    .with_hold_fence(fence.clone());

    let outcome = tokio::time::timeout(SETTLE, hook.try_restore_layer(object_id))
        .await
        .expect("the restore must answer");

    assert!(
        fence.is_cancelled(),
        "the seam fired: the restore reached its EXEC"
    );
    match &outcome {
        RecoveryOutcome::Error(err) => assert_fenced(err),
        other => panic!("a fenced materialisation ends Error(Transient); got {other:?}"),
    }
    assert!(
        !exists(&tv, &tv.key_schema().obj_header(&object_id.to_string())).await,
        "the EXEC must not be sent once the fence is closed"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Re-sends (gate-tv3s9-w8x6): the transient retries
// ═════════════════════════════════════════════════════════════════════════════

/// The conditional seed's first attempt is in flight when the hold begins
/// and fails transiently (`-TRYAGAIN`): its retry is a new send after entry,
/// and is refused unsent. One `EVALSHA` in all; no counter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_seed_retry_after_hold_entry_is_never_sent() {
    let tv = TestValkey::setup().await;
    let counter = tv.key_schema().counter_object_id();
    let gate = ReplyGate::new();
    let sends = Arc::new(Sends::default());
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        gate_the_first_send(
            script_naming(counter.clone()),
            gate.clone(),
            TRYAGAIN_REPLY,
            Arc::clone(&sends),
        ),
    )
    .await;
    let counters = CounterStore::new(client_through(&proxy).await, tv.key_schema().clone());
    let fence = CancellationToken::new();

    let result = close_the_fence_while_the_first_send_is_in_flight(
        {
            let fence = fence.clone();
            async move {
                counters
                    .seed_object_id_if_below_unless_held(SIDECAR_MAX_OBJECT_ID, Some(&fence))
                    .await
            }
        },
        &fence,
        &gate,
        &sends,
    )
    .await;

    assert_unsent_refusal(&result.expect_err("the retry is refused"));
    assert_eq!(sends.writes(), 1, "the retry must never be sent");
    assert!(!exists(&tv, &counter).await, "no copy of the seed ran");
}

/// The establish-at-zero `SET NX`'s first attempt is in flight when the hold
/// begins and fails transiently: its retry is refused unsent. One `SET` in
/// all; no counter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_establish_retry_after_hold_entry_is_never_sent() {
    let tv = TestValkey::setup().await;
    let counter = tv.key_schema().counter_object_id();
    let gate = ReplyGate::new();
    let sends = Arc::new(Sends::default());
    let set_of_counter = {
        let counter = counter.clone();
        move |args: &[Vec<u8>]| {
            args.first().map(|a| a.to_ascii_uppercase()).as_deref() == Some(b"SET")
                && args.get(1).map(Vec::as_slice) == Some(counter.as_bytes())
        }
    };
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        gate_the_first_send(
            set_of_counter,
            gate.clone(),
            TRYAGAIN_REPLY,
            Arc::clone(&sends),
        ),
    )
    .await;
    let counters = CounterStore::new(client_through(&proxy).await, tv.key_schema().clone());
    let fence = CancellationToken::new();

    let result = close_the_fence_while_the_first_send_is_in_flight(
        {
            let fence = fence.clone();
            async move {
                counters
                    .initialize_object_id_counter_if_absent_unless_held(Some(&fence))
                    .await
            }
        },
        &fence,
        &gate,
        &sends,
    )
    .await;

    assert_unsent_refusal(&result.expect_err("the retry is refused"));
    assert_eq!(sends.writes(), 1, "the retry must never be sent");
    assert!(!exists(&tv, &counter).await, "no SET NX ran");
}

// ═════════════════════════════════════════════════════════════════════════════
// Re-sends (gate-tv3s9-w8x6): the NOSCRIPT reload at each fenced EVALSHA
// ═════════════════════════════════════════════════════════════════════════════

/// The conditional seed's `EVALSHA` is in flight when the hold begins, and
/// the store answers `NOSCRIPT`: neither the `SCRIPT LOAD` nor the re-sent
/// `EVALSHA` is sent, and the seed is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_seed_noscript_reload_after_hold_entry_is_never_sent() {
    let tv = TestValkey::setup().await;
    let counter = tv.key_schema().counter_object_id();
    let gate = ReplyGate::new();
    let sends = Arc::new(Sends::default());
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        gate_the_first_send(
            script_naming(counter.clone()),
            gate.clone(),
            NOSCRIPT_REPLY,
            Arc::clone(&sends),
        ),
    )
    .await;
    let counters = CounterStore::new(client_through(&proxy).await, tv.key_schema().clone());
    let fence = CancellationToken::new();

    let result = close_the_fence_while_the_first_send_is_in_flight(
        {
            let fence = fence.clone();
            async move {
                counters
                    .seed_object_id_if_below_unless_held(SIDECAR_MAX_OBJECT_ID, Some(&fence))
                    .await
            }
        },
        &fence,
        &gate,
        &sends,
    )
    .await;

    assert_unsent_refusal(&result.expect_err("the reload is refused"));
    assert_eq!(sends.reloads(), 0, "no SCRIPT LOAD after entry");
    assert_eq!(sends.writes(), 1, "no re-sent EVALSHA after entry");
    assert!(!exists(&tv, &counter).await);
}

/// A pre-store value's `EVALSHA` is in flight when the hold begins, and the
/// store answers `NOSCRIPT`: no reload, no re-send, no value stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prestore_noscript_reload_after_hold_entry_is_never_sent() {
    let tv = TestValkey::setup().await;
    let hash = [0x5Au8; 32];
    let hex = sha256_to_hex(&hash);
    let data_key = tv.key_schema().ext_data(&hex);
    let meta_key = tv.key_schema().ext_meta(&hex);
    let gate = ReplyGate::new();
    let sends = Arc::new(Sends::default());
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        gate_the_first_send(
            script_naming(data_key.clone()),
            gate.clone(),
            NOSCRIPT_REPLY,
            Arc::clone(&sends),
        ),
    )
    .await;
    let ext = ValkeyExternalValueStore::new(client_through(&proxy).await, tv.key_schema().clone());
    let fence = CancellationToken::new();

    let result = close_the_fence_while_the_first_send_is_in_flight(
        {
            let fence = fence.clone();
            async move {
                ext.store_values_unless_held(&[(&hash, b"restored-value".as_slice())], Some(&fence))
                    .await
            }
        },
        &fence,
        &gate,
        &sends,
    )
    .await;

    assert_unsent_refusal(&result.expect_err("the reload is refused"));
    assert_eq!(sends.reloads(), 0, "no SCRIPT LOAD after entry");
    assert_eq!(sends.writes(), 1, "no re-sent EVALSHA after entry");
    assert!(
        !exists(&tv, &data_key).await && !exists(&tv, &meta_key).await,
        "no copy of the value's script ran"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The release enqueue of a committed restore (gate-tv3s9-w8x5, -w8x6)
// ═════════════════════════════════════════════════════════════════════════════

/// The hash the residue's `obj:ext` index holds and the restored state does
/// not: the restore owes its release.
const RESIDUE_HASH: [u8; 32] = [0xA1; 32];

/// A lost layer with a seeded checkpoint whose residue is one `obj:ext`
/// row: the materialisation owes one `rewrite` release, enqueued after its
/// `EXEC` commits.
async fn lost_layer_owing_a_release(
    tv: &TestValkey,
    name: &str,
) -> (u64, Arc<InMemoryCheckpointStore>) {
    let (_session_id, object_id, store) = lost_layer_with_a_checkpoint(tv, name).await;
    let _: i64 = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_ext_index(&object_id.to_string()),
            (sha256_to_hex(&RESIDUE_HASH), "1"),
        )
        .await
        .expect("planting the residue's obj:ext row must succeed");
    (object_id, store)
}

/// A recovery hook over a layer store whose client goes through `proxy`,
/// fenced by `fence`, counting on `metrics`.
fn proxied_restore_hook(
    client: ValkeyClient,
    tv: &TestValkey,
    store: Arc<dyn CheckpointStore>,
    metrics: &SharedMetrics,
    fence: &CancellationToken,
) -> RecoveryHook {
    let layer_store = Arc::new(
        LayerStore::new(client, tv.key_schema().clone(), 4096).with_metrics(Arc::clone(metrics)),
    );
    RecoveryHook::new(
        store,
        layer_store,
        Arc::new(recovery_config()),
        Arc::clone(metrics),
    )
    .with_hold_fence(fence.clone())
}

async fn pending_releases(tv: &TestValkey) -> Vec<String> {
    tv.client()
        .pool
        .smembers(tv.key_schema().ext_pending())
        .await
        .expect("SMEMBERS pending must succeed")
}

fn release_enqueue_failures(metrics: &SharedMetrics) -> u64 {
    cell(
        metrics,
        "live_edit_ext_ref_release_enqueue_failed_total",
        &[],
    )
}

/// The release enqueue's `EVALSHA` is in flight when the hold begins, and
/// the store answers `NOSCRIPT`: no reload and no re-send, so nothing is
/// enqueued; the restore itself has landed, and the skipped batch is counted
/// per reference.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_enqueue_noscript_reload_after_hold_entry_is_never_sent() {
    let tv = TestValkey::setup().await;
    let (object_id, store) = lost_layer_owing_a_release(&tv, "fence-enqueue-noscript").await;
    let gate = ReplyGate::new();
    let sends = Arc::new(Sends::default());
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        gate_the_first_send(
            script_naming(tv.key_schema().ext_pending()),
            gate.clone(),
            NOSCRIPT_REPLY,
            Arc::clone(&sends),
        ),
    )
    .await;
    let metrics = metrics();
    let fence = CancellationToken::new();
    let hook = proxied_restore_hook(client_through(&proxy).await, &tv, store, &metrics, &fence);

    let outcome = close_the_fence_while_the_first_send_is_in_flight(
        async move { hook.try_restore_layer(object_id).await },
        &fence,
        &gate,
        &sends,
    )
    .await;

    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "the materialisation committed before the hold: the restore landed; got {outcome:?}"
    );
    assert!(exists(&tv, &tv.key_schema().obj_header(&object_id.to_string())).await);
    assert_eq!(sends.reloads(), 0, "no SCRIPT LOAD after entry");
    assert_eq!(sends.writes(), 1, "no re-sent EVALSHA after entry");
    assert!(
        pending_releases(&tv).await.is_empty(),
        "nothing may be enqueued after entry"
    );
    assert_eq!(
        release_enqueue_failures(&metrics),
        1,
        "the skipped release is counted, once per reference"
    );
}

/// A committed restore whose release enqueue has not sent its script by the
/// hold's entry skips it rather than send it late: the `EXEC` has reached
/// the store when the fence closes, the object lands, nothing is enqueued,
/// the batch is counted per reference on
/// `live_edit_ext_ref_release_enqueue_failed_total`, and one INFO line names
/// the object.
#[tokio::test]
async fn committed_restore_whose_release_enqueue_is_unsent_at_entry_skips_and_counts_it() {
    let capture = ThreadCapture::start();
    let tv = TestValkey::setup().await;
    let (object_id, store) = lost_layer_owing_a_release(&tv, "fence-enqueue-skip").await;
    let header = tv.key_schema().obj_header(&object_id.to_string());
    let gate = ReplyGate::new();
    let execs = Arc::new(AtomicUsize::new(0));
    // Per connection: whether the transaction being queued writes the
    // object's header — the materialisation's. Its EXEC's replies are held.
    let writes_header: Arc<Mutex<HashMap<usize, bool>>> = Arc::default();
    let rule = {
        let gate = gate.clone();
        let execs = Arc::clone(&execs);
        let header = header.clone();
        Arc::new(move |conn: usize, args: &[Vec<u8>]| {
            let name = args.first()?.to_ascii_uppercase();
            if name == b"HSET" && args.get(1).map(Vec::as_slice) == Some(header.as_bytes()) {
                writes_header.lock().unwrap().insert(conn, true);
                return None;
            }
            if name == b"EXEC" && writes_header.lock().unwrap().remove(&conn).unwrap_or(false) {
                execs.fetch_add(1, Ordering::SeqCst);
                return Some(ProxyAction::HoldReplies(gate.clone()));
            }
            None
        })
    };
    let proxy = ScriptReplyProxy::start_per_connection(test_valkey_addr(), rule).await;
    let metrics = metrics();
    let fence = CancellationToken::new();
    let hook = proxied_restore_hook(client_through(&proxy).await, &tv, store, &metrics, &fence);

    let restore = tokio::spawn(async move { hook.try_restore_layer(object_id).await });
    gate.reply_held().await;
    fence.cancel();
    gate.open();
    let outcome = tokio::time::timeout(SETTLE, restore)
        .await
        .expect("the restore must answer")
        .expect("the restore must not panic");

    assert_eq!(execs.load(Ordering::SeqCst), 1, "one materialisation EXEC");
    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "the skip does not fail the restore, whose object has landed; got {outcome:?}"
    );
    assert!(
        exists(&tv, &header).await,
        "the EXEC sent before entry committed"
    );
    assert!(
        pending_releases(&tv).await.is_empty(),
        "a release enqueue unsent at entry must be skipped, not sent late"
    );
    assert_eq!(
        release_enqueue_failures(&metrics),
        1,
        "the skipped batch is counted per reference"
    );
    let captured = capture.snapshot();
    let skipped: Vec<_> = captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with("checkpoint restore skipped its external-value"))
        })
        .collect();
    assert_eq!(
        skipped.len(),
        1,
        "one INFO line; capture:\n{}",
        captured.text
    );
    assert_eq!(skipped[0]["level"], "INFO");
    assert_eq!(skipped[0]["fields"]["object_id"], object_id);
    assert_eq!(skipped[0]["fields"]["releases"], 1);

    drop(capture);
    purge_keys(&tv).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The external-value fetch-on-miss rewrite
// ═════════════════════════════════════════════════════════════════════════════

/// The external-value fetch-on-miss rewrite (`ValkeyExternalValueStore`'s
/// recovery hold fence; SDD SR-6.2): a read that misses a value the backup
/// store holds writes it back into the store from inside the read, which can
/// be a deadline-exempt catch-up already past the gate, so neither the gate
/// nor a deadline stops it. With the fence closed (after a hold or
/// stale-verdict entry) the read sends no write: it fails `UNAVAILABLE` with
/// the fence's refusal, and nothing appears under the test's prefix. With
/// the same fence open, the same read rewrites the value and answers its
/// bytes, so the closed case's empty store is the fence's doing.
///
/// The fence closes while the read is under way, as the backup fetch answers
/// ([`ParkedStore::close_on`]): the read has already missed and fetched, so
/// only the check immediately before the rewrite's `EVALSHA` can stop it. A
/// fence read once at the read's start would let this write through.
///
/// Drives the real `ValkeyExternalValueStore`: a recording test double of
/// the store has no recovery fence to close.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fetch_on_miss_rewrite_after_hold_entry_writes_nothing_and_answers_unavailable() {
    let tv = TestValkey::setup().await;
    let bytes = b"a value only the backup store still holds".to_vec();
    let hash = ext_content_hash(&bytes);
    let hash_hex = sha256_to_hex(&hash);
    let backup = ParkedStore::passing(Arc::new(InMemoryCheckpointStore::new()));
    backup
        .put_ext_blob(&hash_hex, Bytes::from(bytes.clone()))
        .await
        .expect("seeding the backup store must succeed");
    let store_fenced_by = |fence: CancellationToken| {
        ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_checkpoint_store(Arc::clone(&backup) as Arc<dyn CheckpointStore>)
            .with_recovery_hold_fence(fence)
    };
    let data_key = tv.key_schema().ext_data(&hash_hex);

    // Hold entry during the read: the fence closes as the backup fetch
    // answers.
    let fence = CancellationToken::new();
    backup.close_on(StoreCall::ExtGet, fence.clone());
    let err = tokio::time::timeout(SETTLE, store_fenced_by(fence.clone()).read_batch(&[hash]))
        .await
        .expect("the read must answer within its bound")
        .expect_err("a read whose rewrite the fence refuses fails");
    assert!(
        fence.is_cancelled(),
        "fixture self-check: the read reached the backup fetch, which closed the fence"
    );
    assert_unsent_refusal(&err);
    let written = crate::common::scan_keys(tv.client(), &format!("{}*", tv.prefix())).await;
    assert!(
        written.is_empty(),
        "a fenced fetch-on-miss rewrite must write nothing; found {written:?}"
    );

    // Before entry: the same read, its fence open, rewrites the value.
    let read = tokio::time::timeout(
        SETTLE,
        store_fenced_by(CancellationToken::new()).read_batch(&[hash]),
    )
    .await
    .expect("the read must answer within its bound")
    .expect("an unfenced rewrite recovers the value");
    assert_eq!(read.get(&hash), Some(&bytes), "the backup store's bytes");
    assert!(
        exists(&tv, &data_key).await,
        "the unfenced rewrite stores the value: the closed case's empty store is the fence's \
         doing"
    );
}
